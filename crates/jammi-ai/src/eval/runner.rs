//! EvalRunner: orchestrates the evaluation pipeline.

use std::collections::{BTreeMap, HashMap};

use arrow::datatypes::{DataType, Schema};
use jammi_db::catalog::eval_repo::{EvalRunRecord, PerQueryEvalRecord};
use jammi_db::catalog::status::EvalRunStatus;
use jammi_db::error::{JammiError, Result};
use jammi_db::sql::quote_relation;

use crate::eval::{EvalCalibrationShape, EvalTask};
use crate::model::ModelSource;
use crate::session::InferenceSession;
use jammi_wire::eval::report::{
    compute_calibration, delta_significance, AggregateDelta, CalibrationEvalReport,
    CalibrationPrediction, CompareEvalReport, EmbeddingEvalReport, InferenceAggregate,
    InferenceEvalReport, MetricDelta, PerQueryRecord, PerRecordCalibration, PerRecordPrediction,
    TableEvalReport, PER_QUERY_RECALL_KS,
};

use jammi_numerics::classification::ClassificationMetrics;
use jammi_numerics::ner::{Entity, NerMetrics};
use jammi_numerics::retrieval::RetrievalMetrics;

use super::golden::{
    ensure_column, ensure_column_int64, load_classification_golden_from_batches,
    load_ner_golden_from_batches, load_retrieval_golden_from_batches, QueryModality,
};

/// Orchestrates evaluation pipelines — retrieval and classification.
pub struct EvalRunner<'a> {
    pub(crate) session: &'a InferenceSession,
}

impl<'a> EvalRunner<'a> {
    /// Resolve a model name to the catalog PK an `eval_runs.model_id` FK binds
    /// to. The PK is read off the resolved row — tenant-qualified for a
    /// tenant-owned model, unqualified for a global one — never reconstructed,
    /// so the FK matches the same row the rest of the session resolves.
    async fn eval_model_fk(&self, model_name: &str) -> Result<String> {
        self.session
            .catalog()
            .get_model(model_name)
            .await?
            .map(|m| m.catalog_pk)
            .ok_or_else(|| {
                JammiError::Eval(format!("Model '{model_name}' not registered in catalog"))
            })
    }

    /// Evaluate embedding quality against golden relevance judgments.
    ///
    /// Returns an [`EmbeddingEvalReport`] containing both the aggregate over
    /// all queries (`report.aggregate.recall_at_k`, etc.) and the per-query
    /// arrays (`report.per_query[i].metrics.recall`). The per-query data is
    /// what sample-based statistical rules consume; the aggregate is what
    /// the `eval_runs` catalog row records.
    ///
    /// The per-query arrays are also persisted to `_jammi_eval_per_query`
    /// (spec J9): one row per query keyed by the run's `eval_run_id`, carrying
    /// Recall@{1,3,5,10}, MRR, nDCG, distance, and any opaque `cohorts` tags
    /// supplied for that `query_id`. Persistence is always-on — there is no
    /// opt-in flag. `cohorts` maps a golden-set `query_id` to an opaque
    /// `{key: value}` segment map; a query with no entry stores `{}`.
    ///
    /// Uses the same search infrastructure as `db.search()` — SidecarIndex for ANN
    /// when available, exact_vector_search as fallback.
    ///
    /// `golden_source` is an unquoted relation reference (a bare name or a
    /// dotted `<source>.public.<table>`); each part is quoted independently
    /// before interpolation, so a hyphenated or reserved name resolves verbatim.
    pub async fn eval_embeddings(
        &self,
        source_id: &str,
        embedding_table: Option<&str>,
        golden_source: &str,
        k: usize,
        cohorts: &HashMap<String, BTreeMap<String, String>>,
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

        // 2. Load golden dataset — detect query modality from the present
        //    column: query_image (binary) / query_audio (binary) / query_text.
        let golden_schema = self.source_schema(golden_source).await?;
        ensure_column(&golden_schema, "query_id", DataType::Utf8)?;
        ensure_column(&golden_schema, "relevant_id", DataType::Utf8)?;
        let has_grades = golden_schema.field_with_name("relevance_grade").is_ok();

        let modality = if golden_schema.field_with_name("query_image").is_ok() {
            QueryModality::Image
        } else if golden_schema.field_with_name("query_audio").is_ok() {
            QueryModality::Audio
        } else {
            ensure_column(&golden_schema, "query_text", DataType::Utf8)?;
            QueryModality::Text
        };

        let query_col = match modality {
            QueryModality::Image => "\"query_image\"",
            QueryModality::Audio => "\"query_audio\"",
            QueryModality::Text => "\"query_text\"",
        };
        let grade_select = if has_grades {
            ", \"relevance_grade\""
        } else {
            ""
        };
        let golden_rel = quote_relation(golden_source);
        let batches = self
            .session
            .sql(&format!(
                "SELECT \"query_id\", {query_col}, \"relevant_id\"{grade_select} FROM {golden_rel}"
            ))
            .await?;

        let golden = load_retrieval_golden_from_batches(&batches, has_grades, modality)?;

        // 4. For each query: encode → search → compute metrics.
        let model_source = ModelSource::from_canonical(canonical_model);
        let encode_id = model_source.to_string();
        let mut query_metrics = Vec::new();
        // Recall@{1,3,5,10} and the top-1 distance, captured per query so the
        // per-query record carries the multi-cutoff vector J7 re-aggregates.
        let mut per_query_recalls: Vec<Vec<(usize, f64)>> = Vec::new();
        let mut per_query_distances: Vec<f64> = Vec::new();
        for query in &golden.queries {
            let query_vec = match &query.input {
                super::golden::QueryInput::Text(text) => {
                    self.session.encode_text_query(&encode_id, text).await?
                }
                super::golden::QueryInput::Image(bytes) => {
                    self.session.encode_image_query(&encode_id, bytes).await?
                }
                super::golden::QueryInput::Audio(bytes) => {
                    self.session.encode_audio_query(&encode_id, bytes).await?
                }
            };

            let search_results = result_store
                .search_vectors(self.session.context(), &table, &query_vec, k)
                .await?;

            let retrieved_ids: Vec<String> =
                search_results.iter().map(|(id, _)| id.clone()).collect();

            // Distance = the top-1 result's score (0.0 when nothing retrieved).
            per_query_distances.push(
                search_results
                    .first()
                    .map(|(_, s)| *s as f64)
                    .unwrap_or(0.0),
            );

            // Multi-K recall reuses the numerics kernel — extended, not
            // re-implemented — at the fixed J9 cutoffs.
            per_query_recalls.push(RetrievalMetrics::recall_at_ks(
                &retrieved_ids,
                &query.judgments,
                &PER_QUERY_RECALL_KS,
            ));

            query_metrics.push(RetrievalMetrics::compute_query(
                &retrieved_ids,
                &query.judgments,
                k,
            ));
        }

        // 5. Build typed report. `per_query` carries the historical
        //    single-cutoff `metrics` plus the J9 additions (multi-K recall,
        //    top-1 distance, opaque cohort tags); the aggregate is the mean.
        let aggregate = RetrievalMetrics::aggregate(&query_metrics);
        let per_query: Vec<PerQueryRecord> = golden
            .queries
            .iter()
            .zip(query_metrics)
            .zip(per_query_recalls)
            .zip(per_query_distances)
            .map(|(((q, metrics), recall_at_ks), distance)| PerQueryRecord {
                query_id: q.query_id.clone(),
                metrics,
                recall_at_ks,
                distance,
                cohorts: cohorts.get(&q.query_id).cloned().unwrap_or_default(),
            })
            .collect();
        // The run id keys both the aggregate `eval_runs` row and the per-query
        // rows, and is surfaced on the report so callers can read the per-query
        // arrays back via `eval_per_query`.
        let eval_run_id = uuid::Uuid::new_v4().to_string();
        let report = EmbeddingEvalReport {
            eval_run_id: eval_run_id.clone(),
            aggregate,
            per_query,
        };

        // 6. Record the aggregate in the `eval_runs` catalog row (unchanged
        //    historical path), then persist the per-query arrays to
        //    `_jammi_eval_per_query` keyed by the same `eval_run_id`. Per-query
        //    persistence is always-on (spec J9) — no opt-in flag.
        let model_fk = self.eval_model_fk(canonical_model).await?;
        self.session
            .catalog()
            .record_eval_run(&EvalRunRecord {
                eval_run_id: eval_run_id.clone(),
                eval_type: "embedding".into(),
                model_id: Some(model_fk),
                source_id: source_id.into(),
                golden_source: golden_source.into(),
                k: Some(k as i32),
                metrics_json: serde_json::to_string(&report.aggregate)?,
                status: EvalRunStatus::Completed.to_string(),
                created_at: chrono::Utc::now().to_rfc3339(),
            })
            .await?;

        let per_query_rows = build_per_query_rows(&eval_run_id, &report.per_query)?;
        self.session
            .catalog()
            .record_eval_per_query(&per_query_rows)
            .await?;

        Ok(report)
    }

    /// Evaluate inference quality against golden labels.
    ///
    /// Returns an [`InferenceEvalReport`] carrying the task-shaped
    /// aggregate and the per-record predicted / gold pairs that
    /// sample-based statistical rules consume.
    ///
    /// `golden_source` is an unquoted relation reference (a bare name or a
    /// dotted `<source>.public.<table>`); each part is quoted independently
    /// before interpolation, so a hyphenated or reserved name resolves verbatim.
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

        // `golden_source` is an unquoted source reference; quote each dotted
        // part once so a hyphenated or reserved name resolves verbatim.
        let golden_rel = quote_relation(golden_source);
        let (aggregate, per_record) = match task {
            EvalTask::Classification => {
                let golden_schema = self.source_schema(golden_source).await?;
                ensure_column(&golden_schema, "id", DataType::Utf8)?;
                ensure_column(&golden_schema, label_column, DataType::Utf8)?;

                let batches = self
                    .session
                    .sql(&format!(
                        "SELECT \"id\", \"{label_column}\" AS \"label\" FROM {golden_rel}"
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
                         FROM {golden_rel}"
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
        let model_fk = self.eval_model_fk(&canonical).await?;
        self.session
            .catalog()
            .record_eval_run(&EvalRunRecord {
                eval_run_id: uuid::Uuid::new_v4().to_string(),
                eval_type: task.to_string(),
                model_id: Some(model_fk),
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

    /// Evaluate whether a predictor's *uncertainty* is honest (spec R2).
    ///
    /// This is the orthogonal sibling to `eval_embeddings`/`eval_inference`:
    /// they ask "is the prediction accurate?", this asks "does it know what it
    /// doesn't know?". The two are independent — a model can be accurate and
    /// badly calibrated, or calibrated and useless.
    ///
    /// `golden_source` is a held-out set pairing a predictive distribution with
    /// its realised `outcome`. `shape` selects the predictor's output family and
    /// the columns read:
    /// - [`EvalCalibrationShape::Gaussian`] — `record_id`, `mean`, `sd`,
    ///   `outcome` (a parametric predictive `Normal`).
    /// - [`EvalCalibrationShape::Sample`] — `record_id`, `draws` (a JSON array
    ///   of numbers per row), `outcome` (an ensemble of predictive draws).
    ///
    /// The held-out, three-way-split contract is the caller's: the predictions
    /// here must come from a test set disjoint from both the training set and
    /// any calibration set used to fit the predictor — re-using calibration
    /// points to also test inflates coverage. The harness measures what it is
    /// given; it cannot see the split.
    ///
    /// Returns a [`CalibrationEvalReport`] headlining a strictly proper score
    /// (CRPS) with NLL, the adaptive debiased PIT-calibration diagnostic, and
    /// sharpness/coverage — plus per-cohort slices and the per-record scores. As
    /// with `eval_embeddings`, per-record scores are persisted to
    /// `_jammi_eval_per_query` keyed by the run id, and `cohorts` maps a
    /// `record_id` to opaque segment tags (a record with no entry stores `{}`).
    ///
    /// `golden_source` is an unquoted relation reference (a bare name or a
    /// dotted `<source>.public.<table>`); each part is quoted independently
    /// before interpolation, so a hyphenated or reserved name resolves verbatim.
    pub async fn eval_calibration(
        &self,
        source_id: &str,
        golden_source: &str,
        shape: EvalCalibrationShape,
        cohorts: &HashMap<String, BTreeMap<String, String>>,
    ) -> Result<CalibrationEvalReport> {
        let golden_schema = self.source_schema(golden_source).await?;
        ensure_column(&golden_schema, "record_id", DataType::Utf8)?;

        // `golden_source` is an unquoted source reference; quote each dotted
        // part once so a hyphenated or reserved name resolves verbatim.
        let golden_rel = quote_relation(golden_source);
        let predictions = match shape {
            EvalCalibrationShape::Gaussian => {
                let batches = self
                    .session
                    .sql(&format!(
                        "SELECT \"record_id\", \"mean\", \"sd\", \"outcome\" FROM {golden_rel}"
                    ))
                    .await?;
                let mut predictions = Vec::new();
                for batch in &batches {
                    let ids = super::golden::extract_string_column(batch, "record_id")?;
                    let means = super::golden::extract_f64_column(batch, "mean")?;
                    let sds = super::golden::extract_f64_column(batch, "sd")?;
                    let outcomes = super::golden::extract_f64_column(batch, "outcome")?;
                    for (((id, &mean), &sd), &outcome) in
                        ids.iter().zip(&means).zip(&sds).zip(&outcomes)
                    {
                        predictions.push(CalibrationPrediction::Gaussian {
                            record_id: id.clone(),
                            mean,
                            sd,
                            outcome,
                            cohorts: cohorts.get(id).cloned().unwrap_or_default(),
                        });
                    }
                }
                predictions
            }
            EvalCalibrationShape::Sample => {
                let batches = self
                    .session
                    .sql(&format!(
                        "SELECT \"record_id\", \"draws\", \"outcome\" FROM {golden_rel}"
                    ))
                    .await?;
                let mut predictions = Vec::new();
                for batch in &batches {
                    let ids = super::golden::extract_string_column(batch, "record_id")?;
                    let draws_col = super::golden::extract_string_column(batch, "draws")?;
                    let outcomes = super::golden::extract_f64_column(batch, "outcome")?;
                    for ((id, draws_json), &outcome) in ids.iter().zip(&draws_col).zip(&outcomes) {
                        let draws: Vec<f64> = serde_json::from_str(draws_json).map_err(|e| {
                            JammiError::Eval(format!(
                                "calibration draws JSON parse failed for record {id}: {e}"
                            ))
                        })?;
                        predictions.push(CalibrationPrediction::Sample {
                            record_id: id.clone(),
                            draws,
                            outcome,
                            cohorts: cohorts.get(id).cloned().unwrap_or_default(),
                        });
                    }
                }
                predictions
            }
        };

        let eval_run_id = uuid::Uuid::new_v4().to_string();
        let report = compute_calibration(eval_run_id.clone(), &predictions)?;

        // Record the aggregate, then persist the per-record scores to
        // `_jammi_eval_per_query` keyed by the same run id — the same shape the
        // embedding eval reuses, so cohort slicing and significance read back
        // through one path. A calibration run is not model-scoped: it scores a
        // held-out predictive distribution from the golden source, not a
        // registered model, so `model_id` is `None` and `k` is empty. The
        // predictor family is carried by `eval_type` (the shape's name).
        self.session
            .catalog()
            .record_eval_run(&EvalRunRecord {
                eval_run_id: eval_run_id.clone(),
                eval_type: shape.to_string(),
                model_id: None,
                source_id: source_id.into(),
                golden_source: golden_source.into(),
                k: None,
                metrics_json: serde_json::to_string(&report.aggregate)?,
                status: EvalRunStatus::Completed.to_string(),
                created_at: chrono::Utc::now().to_rfc3339(),
            })
            .await?;

        let per_record_rows = build_calibration_per_record_rows(&eval_run_id, &report.per_record)?;
        self.session
            .catalog()
            .record_eval_per_query(&per_record_rows)
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

        // `eval_compare` does not surface cohort tagging; each per-table eval
        // persists per-query rows with empty cohorts.
        let no_cohorts: HashMap<String, BTreeMap<String, String>> = HashMap::new();
        let mut all_reports: Vec<(String, EmbeddingEvalReport)> = Vec::new();
        for table_name in embedding_tables {
            let report = self
                .eval_embeddings(source_id, Some(table_name), golden_source, k, &no_cohorts)
                .await?;
            all_reports.push((table_name.clone(), report));
        }

        let baseline_agg = all_reports[0].1.aggregate.clone();
        let baseline_per_query = all_reports[0].1.per_query.clone();
        let per_table: Vec<TableEvalReport> = all_reports
            .into_iter()
            .enumerate()
            .map(|(i, (table_name, embedding_eval))| {
                let delta = if i == 0 {
                    None
                } else {
                    // Significance pairs the baseline and treatment per-query
                    // metric arrays by `query_id`; the delta numbers stay the
                    // aggregate differences, untouched (purely additive).
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
                        significance: delta_significance(
                            &baseline_per_query,
                            &embedding_eval.per_query,
                        ),
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
    ///
    /// `golden_source` is an unquoted source reference (`<source>.public.<table>`
    /// or a bare name); each dotted part is quoted independently so a
    /// hyphenated or reserved name resolves verbatim.
    async fn source_schema(&self, golden_source: &str) -> Result<Schema> {
        let golden_rel = quote_relation(golden_source);
        let batches = self
            .session
            .sql(&format!("SELECT * FROM {golden_rel} LIMIT 0"))
            .await?;

        if let Some(batch) = batches.first() {
            Ok(batch.schema().as_ref().clone())
        } else {
            let df = self
                .session
                .context()
                .sql(&format!("SELECT * FROM {golden_rel} LIMIT 0"))
                .await
                .map_err(|e| {
                    JammiError::Eval(format!("Failed to get schema for '{golden_source}': {e}"))
                })?;
            Ok(df.schema().as_arrow().clone())
        }
    }
}

/// Serialize the per-query report records into the `_jammi_eval_per_query`
/// row shape (spec J9): one row per query, carrying a `metrics` JSON object
/// (`recall@1/3/5/10`, `mrr`, `ndcg`, `distance`) and a `cohorts` JSON object
/// (`{}` when none). The metric JSON keys are stable so a downstream consumer's
/// cohort aggregation can read them back without re-deriving the schema.
fn build_per_query_rows(
    eval_run_id: &str,
    per_query: &[PerQueryRecord],
) -> Result<Vec<PerQueryEvalRecord>> {
    per_query
        .iter()
        .map(|rec| {
            let mut metrics = serde_json::Map::new();
            for (k, recall) in &rec.recall_at_ks {
                metrics.insert(format!("recall@{k}"), serde_json::Value::from(*recall));
            }
            metrics.insert("mrr".into(), serde_json::Value::from(rec.metrics.mrr));
            metrics.insert("ndcg".into(), serde_json::Value::from(rec.metrics.ndcg));
            metrics.insert("distance".into(), serde_json::Value::from(rec.distance));

            Ok(PerQueryEvalRecord {
                eval_run_id: eval_run_id.to_string(),
                query_id: rec.query_id.clone(),
                cohorts_json: serde_json::to_string(&rec.cohorts)?,
                metrics_json: serde_json::Value::Object(metrics).to_string(),
            })
        })
        .collect()
}

/// Serialize the per-record calibration scores into the `_jammi_eval_per_query`
/// row shape — one row per held-out prediction, reusing the same table and
/// keying the embedding eval uses (spec R2/J9). The `metrics` JSON carries the
/// proper scores (`crps`, `nll`), the `pit` value, the nominal-interval
/// `coverage` (0/1 per record) and `interval_width`; `cohorts` is the opaque
/// segment map (`{}` when none). The `query_id` slot carries the `record_id`.
fn build_calibration_per_record_rows(
    eval_run_id: &str,
    per_record: &[PerRecordCalibration],
) -> Result<Vec<PerQueryEvalRecord>> {
    per_record
        .iter()
        .map(|rec| {
            let mut metrics = serde_json::Map::new();
            metrics.insert("crps".into(), serde_json::Value::from(rec.crps));
            metrics.insert("nll".into(), serde_json::Value::from(rec.nll));
            metrics.insert("pit".into(), serde_json::Value::from(rec.pit));
            metrics.insert(
                "coverage".into(),
                serde_json::Value::from(f64::from(u8::from(rec.covered))),
            );
            metrics.insert(
                "interval_width".into(),
                serde_json::Value::from(rec.interval_width),
            );

            Ok(PerQueryEvalRecord {
                eval_run_id: eval_run_id.to_string(),
                query_id: rec.record_id.clone(),
                cohorts_json: serde_json::to_string(&rec.cohorts)?,
                metrics_json: serde_json::Value::Object(metrics).to_string(),
            })
        })
        .collect()
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
