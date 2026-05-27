//! EvalRunner: orchestrates the evaluation pipeline.

use arrow::datatypes::{DataType, Schema};
use jammi_db::catalog::eval_repo::EvalRunRecord;
use jammi_db::catalog::status::EvalRunStatus;
use jammi_db::error::{JammiError, Result};

use crate::eval::report::{
    AggregateDelta, CompareEvalReport, EmbeddingEvalReport, InferenceAggregate,
    InferenceEvalReport, MetricDelta, PerQueryRecord, PerRecordPrediction, TableEvalReport,
};
use crate::eval::EvalTask;
use crate::model::ModelSource;
use crate::session::InferenceSession;

use jammi_numerics::classification::ClassificationMetrics;
use jammi_numerics::ner::{Entity, NerMetrics};
use jammi_numerics::retrieval::RetrievalMetrics;

use super::golden::{
    ensure_column, ensure_column_int64, load_classification_golden_from_batches,
    load_ner_golden_from_batches, load_retrieval_golden_from_batches,
};

/// Orchestrates evaluation pipelines — retrieval and classification.
pub struct EvalRunner<'a> {
    pub(crate) session: &'a InferenceSession,
}

impl<'a> EvalRunner<'a> {
    /// Evaluate embedding quality against golden relevance judgments.
    ///
    /// Returns an [`EmbeddingEvalReport`] containing both the aggregate over
    /// all queries (`report.aggregate.recall_at_k`, etc.) and the per-query
    /// arrays (`report.per_query[i].metrics.recall`). The per-query data is
    /// what sample-based statistical rules consume; the aggregate is what
    /// the catalog persists.
    ///
    /// Uses the same search infrastructure as `db.search()` — SidecarIndex for ANN
    /// when available, exact_vector_search as fallback.
    pub async fn eval_embeddings(
        &self,
        source_id: &str,
        embedding_table: Option<&str>,
        golden_source: &str,
        k: usize,
    ) -> Result<EmbeddingEvalReport> {
        // 1. Resolve embedding table.
        // result_tables.model_id stores the canonical model name (ModelSource::to_string()).
        let table = self
            .session
            .catalog()
            .resolve_embedding_table(source_id, embedding_table)
            .await?;
        let canonical_model = &table.model_id;

        let result_store = self.session.result_store();

        // 2. Load golden dataset — detect text vs image queries
        let golden_schema = self.source_schema(golden_source).await?;
        ensure_column(&golden_schema, "query_id", DataType::Utf8)?;
        ensure_column(&golden_schema, "relevant_id", DataType::Utf8)?;
        let has_grades = golden_schema.field_with_name("relevance_grade").is_ok();

        let is_image = golden_schema.field_with_name("query_image").is_ok();
        if !is_image {
            ensure_column(&golden_schema, "query_text", DataType::Utf8)?;
        }

        let query_col = if is_image {
            "\"query_image\""
        } else {
            "\"query_text\""
        };
        let grade_select = if has_grades {
            ", \"relevance_grade\""
        } else {
            ""
        };
        let batches = self
            .session
            .sql(&format!(
                "SELECT \"query_id\", {query_col}, \"relevant_id\"{grade_select} FROM {golden_source}"
            ))
            .await?;

        let golden = load_retrieval_golden_from_batches(&batches, has_grades, is_image)?;

        // 4. For each query: encode → search → compute metrics.
        let model_source = ModelSource::from_canonical(canonical_model);
        let encode_id = model_source.to_string();
        let mut query_metrics = Vec::new();
        for query in &golden.queries {
            let query_vec = match &query.input {
                super::golden::QueryInput::Text(text) => {
                    self.session.encode_text_query(&encode_id, text).await?
                }
                super::golden::QueryInput::Image(bytes) => {
                    self.session.encode_image_query(&encode_id, bytes).await?
                }
            };

            let search_results = result_store
                .search_vectors(self.session.context(), &table, &query_vec, k)
                .await?;

            let retrieved_ids: Vec<String> =
                search_results.iter().map(|(id, _)| id.clone()).collect();

            query_metrics.push(RetrievalMetrics::compute_query(
                &retrieved_ids,
                &query.judgments,
                k,
            ));
        }

        // 5. Build typed report. `per_query` is the data the old shape was
        //    discarding; the aggregate is the historical mean.
        let aggregate = RetrievalMetrics::aggregate(&query_metrics);
        let per_query: Vec<PerQueryRecord> = golden
            .queries
            .iter()
            .zip(query_metrics)
            .map(|(q, metrics)| PerQueryRecord {
                query_id: q.query_id.clone(),
                metrics,
            })
            .collect();
        let report = EmbeddingEvalReport {
            aggregate,
            per_query,
        };

        // 6. Record in catalog. Only the aggregate persists — per-query
        //    arrays are a transient response shape kept out of long-term
        //    storage because the catalog needs them only for historical
        //    aggregate trend reporting.
        self.session
            .catalog()
            .record_eval_run(&EvalRunRecord {
                eval_run_id: uuid::Uuid::new_v4().to_string(),
                eval_type: "embedding".into(),
                model_id: crate::model::to_catalog_pk(canonical_model, 1),
                source_id: source_id.into(),
                golden_source: golden_source.into(),
                k: Some(k as i32),
                metrics_json: serde_json::to_string(&report.aggregate)?,
                status: EvalRunStatus::Completed.to_string(),
                created_at: chrono::Utc::now().to_rfc3339(),
            })
            .await?;

        Ok(report)
    }

    /// Evaluate inference quality against golden labels.
    ///
    /// Returns an [`InferenceEvalReport`] carrying the task-shaped
    /// aggregate and the per-record predicted / gold pairs that
    /// sample-based statistical rules consume.
    pub async fn eval_inference(
        &self,
        model_id: &str,
        source_id: &str,
        columns: &[String],
        task: EvalTask,
        golden_source: &str,
        label_column: &str,
    ) -> Result<InferenceEvalReport> {
        // `model_id` is a user-supplied identifier (`local:/path`, `hf://owner/repo`,
        // or `owner/repo`) — parse it the same way every other public entry point
        // does. `from_canonical` is reserved for canonical names already stored in
        // the catalog (no `local:` prefix), which is not the shape callers pass here.
        let model_source = ModelSource::parse(model_id);

        let (aggregate, per_record) = match task {
            EvalTask::Classification => {
                let golden_schema = self.source_schema(golden_source).await?;
                ensure_column(&golden_schema, "id", DataType::Utf8)?;
                ensure_column(&golden_schema, label_column, DataType::Utf8)?;

                let batches = self
                    .session
                    .sql(&format!(
                        "SELECT \"id\", \"{label_column}\" AS \"label\" FROM {golden_source}"
                    ))
                    .await?;
                let golden = load_classification_golden_from_batches(&batches)?;

                let results = self
                    .session
                    .infer(
                        source_id,
                        &model_source,
                        crate::model::ModelTask::Classification,
                        columns,
                        "id",
                    )
                    .await?;

                let mut per_record: Vec<PerRecordPrediction> = Vec::new();
                let mut aligned_predicted = Vec::new();
                let mut aligned_actual = Vec::new();
                for batch in &results {
                    let ids = super::golden::extract_string_column(batch, "_row_id")?;
                    let labels = super::golden::extract_string_column(batch, "label")?;
                    for (id, pred) in ids.iter().zip(&labels) {
                        if let Some(actual) = golden.labels.get(id) {
                            aligned_predicted.push(pred.clone());
                            aligned_actual.push(actual.clone());
                            per_record.push(PerRecordPrediction::Classification {
                                record_id: id.clone(),
                                predicted: pred.clone(),
                                gold: actual.clone(),
                            });
                        }
                    }
                }

                let result = ClassificationMetrics::compute(&aligned_predicted, &aligned_actual);
                (InferenceAggregate::Classification(result), per_record)
            }
            EvalTask::Ner => {
                // NER goldens carry one entity span per row. The runner
                // joins on `id`, then groups spans into per-row entity
                // sets — the shape `NerMetrics::compute` consumes.
                let golden_schema = self.source_schema(golden_source).await?;
                ensure_column(&golden_schema, "id", DataType::Utf8)?;
                ensure_column(&golden_schema, label_column, DataType::Utf8)?;
                ensure_column_int64(&golden_schema, "start")?;
                ensure_column_int64(&golden_schema, "end")?;

                let batches = self
                    .session
                    .sql(&format!(
                        "SELECT \"id\", \"{label_column}\" AS \"label\", \"start\", \"end\" \
                         FROM {golden_source}"
                    ))
                    .await?;
                let golden = load_ner_golden_from_batches(&batches)?;

                let results = self
                    .session
                    .infer(
                        source_id,
                        &model_source,
                        crate::model::ModelTask::Ner,
                        columns,
                        "id",
                    )
                    .await?;

                let mut per_record: Vec<PerRecordPrediction> = Vec::new();
                let mut aligned_predicted: Vec<Vec<Entity>> = Vec::new();
                let mut aligned_gold: Vec<Vec<Entity>> = Vec::new();
                for batch in &results {
                    let ids = super::golden::extract_string_column(batch, "_row_id")?;
                    let statuses = super::golden::extract_string_column(batch, "_status")?;
                    let entities_col = super::golden::extract_string_column(batch, "entities")?;
                    for ((id, status), entities_json) in
                        ids.iter().zip(&statuses).zip(&entities_col)
                    {
                        // Rows that errored mid-batch carry `_status != "ok"`;
                        // skip them so a single backend error doesn't poison
                        // the aggregate. Rows without a gold counterpart are
                        // also dropped — same alignment rule the
                        // classification arm uses.
                        if status != "ok" {
                            continue;
                        }
                        let Some(gold) = golden.entities.get(id) else {
                            continue;
                        };
                        let predicted: Vec<Entity> =
                            serde_json::from_str(entities_json).map_err(|e| {
                                JammiError::Eval(format!(
                                    "NER entities JSON parse failed for row {id}: {e}"
                                ))
                            })?;
                        aligned_predicted.push(predicted.clone());
                        aligned_gold.push(gold.clone());
                        per_record.push(PerRecordPrediction::Ner {
                            record_id: id.clone(),
                            predicted,
                            gold: gold.clone(),
                        });
                    }
                }

                let metrics = NerMetrics::compute(&aligned_predicted, &aligned_gold);
                (InferenceAggregate::Ner(metrics), per_record)
            }
        };

        let report = InferenceEvalReport {
            aggregate,
            per_record,
        };

        // Record in catalog
        let canonical = model_source.to_string();
        self.session
            .catalog()
            .record_eval_run(&EvalRunRecord {
                eval_run_id: uuid::Uuid::new_v4().to_string(),
                eval_type: task.to_string(),
                model_id: crate::model::to_catalog_pk(&canonical, 1),
                source_id: source_id.into(),
                golden_source: golden_source.into(),
                k: None,
                metrics_json: serde_json::to_string(&report.aggregate)?,
                status: EvalRunStatus::Completed.to_string(),
                created_at: chrono::Utc::now().to_rfc3339(),
            })
            .await?;

        Ok(report)
    }

    /// Compare multiple embedding tables side-by-side.
    /// First table is the baseline; deltas are computed for all others.
    pub async fn eval_compare(
        &self,
        embedding_tables: &[String],
        source_id: &str,
        golden_source: &str,
        k: usize,
    ) -> Result<CompareEvalReport> {
        if embedding_tables.len() < 2 {
            return Err(JammiError::Eval(
                "eval_compare requires at least 2 embedding tables".into(),
            ));
        }

        let mut all_reports: Vec<(String, EmbeddingEvalReport)> = Vec::new();
        for table_name in embedding_tables {
            let report = self
                .eval_embeddings(source_id, Some(table_name), golden_source, k)
                .await?;
            all_reports.push((table_name.clone(), report));
        }

        let baseline_agg = all_reports[0].1.aggregate.clone();
        let per_table: Vec<TableEvalReport> = all_reports
            .into_iter()
            .enumerate()
            .map(|(i, (table_name, embedding_eval))| {
                let delta = if i == 0 {
                    None
                } else {
                    Some(AggregateDelta {
                        recall_at_k: metric_delta(
                            baseline_agg.recall_at_k,
                            embedding_eval.aggregate.recall_at_k,
                        ),
                        precision_at_k: metric_delta(
                            baseline_agg.precision_at_k,
                            embedding_eval.aggregate.precision_at_k,
                        ),
                        mrr: metric_delta(baseline_agg.mrr, embedding_eval.aggregate.mrr),
                        ndcg: metric_delta(baseline_agg.ndcg, embedding_eval.aggregate.ndcg),
                    })
                };
                TableEvalReport {
                    table_name,
                    embedding_eval,
                    delta,
                }
            })
            .collect();

        Ok(CompareEvalReport { per_table })
    }

    /// Get the schema of a registered golden source.
    async fn source_schema(&self, golden_source: &str) -> Result<Schema> {
        let batches = self
            .session
            .sql(&format!("SELECT * FROM {golden_source} LIMIT 0"))
            .await?;

        if let Some(batch) = batches.first() {
            Ok(batch.schema().as_ref().clone())
        } else {
            let df = self
                .session
                .context()
                .sql(&format!("SELECT * FROM {golden_source} LIMIT 0"))
                .await
                .map_err(|e| {
                    JammiError::Eval(format!("Failed to get schema for '{golden_source}': {e}"))
                })?;
            Ok(df.schema().as_arrow().clone())
        }
    }
}

/// Compute the absolute and relative delta between a baseline and a model
/// aggregate metric. Relative is zero when the baseline is zero.
fn metric_delta(base: f64, model: f64) -> MetricDelta {
    let absolute = model - base;
    let relative = if base.abs() > f64::EPSILON {
        absolute / base
    } else {
        0.0
    };
    MetricDelta { absolute, relative }
}
