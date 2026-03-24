use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;

use jammi_engine::catalog::result_repo::ResultTableRecord;
use jammi_engine::error::{JammiError, Result};
use jammi_engine::index::sidecar::SidecarIndex;
use jammi_engine::store::writer::ParquetResultWriter;
use jammi_engine::store::ResultStore;

use crate::model::{ModelSource, ModelTask};
use crate::operator::inference_exec::InferenceExecBuilder;
use crate::pipeline::result_sink::ResultSink;
use crate::session::InferenceSession;

/// Orchestrates `generate_embeddings()`: model → InferenceExec → ResultSink → index.
pub struct EmbeddingPipeline<'a> {
    session: &'a InferenceSession,
    result_store: &'a ResultStore,
}

impl<'a> EmbeddingPipeline<'a> {
    pub fn new(session: &'a InferenceSession, result_store: &'a ResultStore) -> Self {
        Self {
            session,
            result_store,
        }
    }

    /// Run the full embedding generation pipeline for a source.
    pub async fn run(
        &self,
        source_id: &str,
        model_id: &str,
        columns: &[String],
        key_column: &str,
    ) -> Result<ResultTableRecord> {
        // Parse model source from model_id string
        let model_source = if let Some(path) = model_id.strip_prefix("local:") {
            ModelSource::local(std::path::PathBuf::from(path))
        } else {
            ModelSource::hf(model_id)
        };

        // Pre-load model to get embedding dimensions
        let guard = self
            .session
            .model_cache()
            .get_or_load(&model_source, ModelTask::Embedding, None)
            .await?;
        let embedding_dim = guard
            .model
            .embedding_dim()
            .ok_or_else(|| JammiError::Inference("Model does not support embeddings".into()))?;
        drop(guard);

        // Create result table in catalog
        let text_cols = columns.join(",");
        let table_info = self.result_store.create_table(
            source_id,
            "embedding",
            model_id,
            Some(embedding_dim as i32),
            Some(key_column),
            Some(&text_cols),
        )?;

        // Build scan plan over source
        let table_name = self.session.find_table_name(source_id)?;
        let query = self
            .session
            .build_source_query(source_id, &table_name, key_column, columns);

        let df = self
            .session
            .context()
            .sql(&query)
            .await
            .map_err(|e| JammiError::Inference(format!("Failed to scan source: {e}")))?;
        let input_plan = df
            .create_physical_plan()
            .await
            .map_err(|e| JammiError::Inference(format!("Failed to create scan plan: {e}")))?;

        // Create InferenceExec
        let inference_exec = InferenceExecBuilder::new(
            input_plan,
            model_source,
            ModelTask::Embedding,
            columns.to_vec(),
            key_column.to_string(),
            source_id.to_string(),
            Arc::clone(self.session.model_cache()),
        )
        .batch_size(self.session.inner_config().inference.batch_size)
        .observer(self.session.observer().clone())
        .embedding_dim(Some(embedding_dim))
        .build()?;

        // Create ResultSink
        let embedding_schema = jammi_engine::store::schema::embedding_table_schema(embedding_dim);
        let writer = ParquetResultWriter::new(&table_info.parquet_path, embedding_schema)?;
        let sidecar = SidecarIndex::new(embedding_dim)?;
        let checkpoint_interval = self.session.inner_config().embedding.checkpoint_interval;
        let mut sink = ResultSink::for_embeddings(
            writer,
            sidecar,
            self.session.catalog(),
            table_info.table_name.clone(),
            checkpoint_interval,
        );

        // Execute and stream results through sink
        let task_ctx = self.session.context().task_ctx();
        let stream = inference_exec
            .execute(0, task_ctx)
            .map_err(|e| JammiError::Inference(format!("InferenceExec failed: {e}")))?;

        let batches = datafusion::physical_plan::common::collect(stream)
            .await
            .map_err(|e| JammiError::Inference(format!("Failed to collect results: {e}")))?;

        for batch in &batches {
            sink.write_batch(batch)?;
        }

        let (row_count, index) = sink.finalize()?;

        // Save sidecar index
        if let Some(ref idx) = index {
            if let Some(ref idx_path) = table_info.index_path {
                idx.save(idx_path)?;
            }
        }

        // Finalize: register in DataFusion and update catalog to 'ready'
        self.result_store
            .finalize(
                self.session.context(),
                &table_info.table_name,
                &table_info.parquet_path,
                row_count,
            )
            .await?;

        // Return the updated record
        self.session
            .catalog()
            .get_result_table(&table_info.table_name)?
            .ok_or_else(|| {
                JammiError::Catalog(format!(
                    "Result table '{}' not found after finalization",
                    table_info.table_name
                ))
            })
    }
}
