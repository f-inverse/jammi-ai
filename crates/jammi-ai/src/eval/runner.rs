//! EvalRunner: orchestrates the evaluation pipeline.

use arrow::datatypes::{DataType, Schema};
use jammi_engine::catalog::eval_repo::EvalRunRecord;
use jammi_engine::catalog::status::EvalRunStatus;
use jammi_engine::error::{JammiError, Result};

use crate::eval::EvalTask;
use crate::model::ModelSource;
use crate::session::InferenceSession;

use super::golden::{
    ensure_column, load_classification_golden_from_batches, load_retrieval_golden_from_batches,
    load_summarization_golden_from_batches,
};
use super::metrics::classification::ClassificationMetrics;
use super::metrics::retrieval::RetrievalMetrics;
use super::metrics::summarization::SummarizationMetrics;

/// Orchestrates evaluation pipelines — retrieval, classification, summarization.
pub struct EvalRunner<'a> {
    pub(crate) session: &'a InferenceSession,
}

impl<'a> EvalRunner<'a> {
    /// Evaluate embedding quality against golden relevance judgments.
    ///
    /// Uses the same search infrastructure as `db.search()` — SidecarIndex for ANN
    /// when available, exact_vector_search as fallback.
    pub async fn eval_embeddings(
        &self,
        source_id: &str,
        embedding_table: Option<&str>,
        golden_source: &str,
        k: usize,
    ) -> Result<serde_json::Value> {
        // 1. Resolve embedding table.
        // result_tables.model_id stores the canonical model name (ModelSource::to_string()).
        let table = self
            .session
            .catalog()
            .resolve_embedding_table(source_id, embedding_table)?;
        let canonical_model = &table.model_id;

        let result_store = self.session.result_store();

        // 2. Load golden dataset
        let golden_schema = self.source_schema(golden_source).await?;
        ensure_column(&golden_schema, "query_id", DataType::Utf8)?;
        ensure_column(&golden_schema, "query_text", DataType::Utf8)?;
        ensure_column(&golden_schema, "relevant_id", DataType::Utf8)?;
        let has_grades = golden_schema.field_with_name("relevance_grade").is_ok();

        let grade_select = if has_grades {
            ", \"relevance_grade\""
        } else {
            ""
        };
        let batches = self
            .session
            .sql(&format!(
                "SELECT \"query_id\", \"query_text\", \"relevant_id\"{grade_select} FROM {golden_source}"
            ))
            .await?;

        let golden = load_retrieval_golden_from_batches(&batches, has_grades)?;

        // 4. For each query: encode → search → compute metrics.
        // encode_query needs a model source; reconstruct from canonical name.
        let model_source = ModelSource::from_canonical(canonical_model);
        let encode_id = model_source.to_string();
        let mut query_metrics = Vec::new();
        for query in &golden.queries {
            let query_vec = self
                .session
                .encode_query(&encode_id, &query.query_text)
                .await?;

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

        // 5. Aggregate
        let agg = RetrievalMetrics::aggregate(&query_metrics);
        let metrics_json = serde_json::to_value(&agg)?;

        // 6. Record in catalog
        self.session.catalog().record_eval_run(&EvalRunRecord {
            eval_run_id: uuid::Uuid::new_v4().to_string(),
            eval_type: "embedding".into(),
            model_id: crate::model::to_catalog_pk(canonical_model, 1),
            source_id: source_id.into(),
            golden_source: golden_source.into(),
            k: Some(k as i32),
            metrics_json: serde_json::to_string(&agg)?,
            status: EvalRunStatus::Completed.to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
        })?;

        Ok(metrics_json)
    }

    /// Evaluate inference quality against golden labels.
    pub async fn eval_inference(
        &self,
        model_id: &str,
        source_id: &str,
        columns: &[String],
        task: EvalTask,
        golden_source: &str,
        label_column: &str,
    ) -> Result<serde_json::Value> {
        let model_source = ModelSource::from_canonical(model_id);

        let metrics_json = match task {
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

                let mut aligned_predicted = Vec::new();
                let mut aligned_actual = Vec::new();
                for batch in &results {
                    let ids = super::golden::extract_string_column(batch, "_row_id")?;
                    let labels = super::golden::extract_string_column(batch, "label")?;
                    for (id, pred) in ids.iter().zip(&labels) {
                        if let Some(actual) = golden.labels.get(id) {
                            aligned_predicted.push(pred.clone());
                            aligned_actual.push(actual.clone());
                        }
                    }
                }

                let result = ClassificationMetrics::compute(&aligned_predicted, &aligned_actual);
                serde_json::to_value(&result)?
            }
            EvalTask::Summarization => {
                let golden_schema = self.source_schema(golden_source).await?;
                ensure_column(&golden_schema, "id", DataType::Utf8)?;
                ensure_column(&golden_schema, "reference", DataType::Utf8)?;

                let batches = self
                    .session
                    .sql(&format!(
                        "SELECT \"id\", \"reference\" FROM {golden_source}"
                    ))
                    .await?;
                let golden = load_summarization_golden_from_batches(&batches)?;

                let results = self
                    .session
                    .infer(
                        source_id,
                        &model_source,
                        crate::model::ModelTask::Summarization,
                        columns,
                        "id",
                    )
                    .await?;

                let mut scores = Vec::new();
                for batch in &results {
                    let ids = super::golden::extract_string_column(batch, "_row_id")?;
                    let generated = super::golden::extract_string_column(batch, "summary")?;
                    for (id, gen) in ids.iter().zip(&generated) {
                        if let Some(reference) = golden.references.get(id) {
                            scores.push(SummarizationMetrics::rouge_l(gen, reference));
                        }
                    }
                }

                let agg = SummarizationMetrics::aggregate(&scores);
                serde_json::to_value(&agg)?
            }
        };

        // Record in catalog
        let canonical = model_source.to_string();
        self.session.catalog().record_eval_run(&EvalRunRecord {
            eval_run_id: uuid::Uuid::new_v4().to_string(),
            eval_type: task.to_string(),
            model_id: crate::model::to_catalog_pk(&canonical, 1),
            source_id: source_id.into(),
            golden_source: golden_source.into(),
            k: None,
            metrics_json: serde_json::to_string(&metrics_json)?,
            status: EvalRunStatus::Completed.to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
        })?;

        Ok(metrics_json)
    }

    /// Compare multiple embedding tables side-by-side.
    /// First table is the baseline; deltas are computed for all others.
    pub async fn eval_compare(
        &self,
        embedding_tables: &[String],
        source_id: &str,
        golden_source: &str,
        k: usize,
    ) -> Result<serde_json::Value> {
        if embedding_tables.len() < 2 {
            return Err(JammiError::Eval(
                "eval_compare requires at least 2 embedding tables".into(),
            ));
        }

        let mut all_metrics: Vec<(String, serde_json::Value)> = Vec::new();
        for table_name in embedding_tables {
            let metrics = self
                .eval_embeddings(source_id, Some(table_name), golden_source, k)
                .await?;
            all_metrics.push((table_name.clone(), metrics));
        }

        let baseline = &all_metrics[0].1;
        let metric_keys = ["recall_at_k", "precision_at_k", "mrr", "ndcg"];

        let mut results = serde_json::Map::new();
        for (name, metrics) in &all_metrics {
            results.insert(name.clone(), metrics.clone());
        }

        let mut deltas = serde_json::Map::new();
        for (name, metrics) in all_metrics.iter().skip(1) {
            let mut model_delta = serde_json::Map::new();
            for key in &metric_keys {
                let base_val = baseline.get(key).and_then(|v| v.as_f64()).unwrap_or(0.0);
                let model_val = metrics.get(key).and_then(|v| v.as_f64()).unwrap_or(0.0);

                let absolute = model_val - base_val;
                let relative = if base_val.abs() > f64::EPSILON {
                    absolute / base_val
                } else {
                    0.0
                };

                model_delta.insert(
                    key.to_string(),
                    serde_json::json!({
                        "absolute": absolute,
                        "relative": relative,
                    }),
                );
            }
            deltas.insert(name.clone(), serde_json::Value::Object(model_delta));
        }

        results.insert(
            "baseline".into(),
            serde_json::Value::String(all_metrics[0].0.clone()),
        );
        results.insert("delta".into(), serde_json::Value::Object(deltas));

        Ok(serde_json::Value::Object(results))
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
