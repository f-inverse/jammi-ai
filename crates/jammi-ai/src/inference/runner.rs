use std::sync::Arc;
use std::time::Instant;

use arrow::array::{ArrayRef, RecordBatch};
use arrow::datatypes::SchemaRef;
use datafusion::execution::SendableRecordBatchStream;
use futures::StreamExt;
use jammi_engine::error::{JammiError, Result};
use tokio::sync::mpsc::Sender;

use super::adapter::{create_adapter, BackendOutput, OutputAdapter};
use super::observer::InferenceObserver;
use super::schema::build_prefix_columns;
use super::{extract_column, extract_columns, slice_columns};
use crate::model::cache::ModelCache;
use crate::model::{BackendType, LoadedModel, ModelSource, ModelTask};

/// Processes input RecordBatches through a model, handling batching,
/// error recovery, and dynamic batch sizing.
pub struct InferenceRunner {
    model_cache: Arc<ModelCache>,
    source: ModelSource,
    task: ModelTask,
    content_columns: Vec<String>,
    key_column: String,
    source_id: String,
    backend: Option<BackendType>,
    batch_size: usize,
    observer: Option<Arc<dyn InferenceObserver>>,
}

impl InferenceRunner {
    /// Create a runner for the given model, task, and column configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_cache: Arc<ModelCache>,
        source: ModelSource,
        task: ModelTask,
        content_columns: Vec<String>,
        key_column: String,
        source_id: String,
        backend: Option<BackendType>,
        batch_size: usize,
        observer: Option<Arc<dyn InferenceObserver>>,
    ) -> Self {
        Self {
            model_cache,
            source,
            task,
            content_columns,
            key_column,
            source_id,
            backend,
            batch_size,
            observer,
        }
    }

    /// Consume the input stream, run inference in sub-batches, and send results to `tx`.
    pub async fn run(
        &self,
        mut input: SendableRecordBatchStream,
        tx: Sender<datafusion::error::Result<RecordBatch>>,
        output_schema: SchemaRef,
    ) -> std::result::Result<(), datafusion::error::DataFusionError> {
        let result = self.run_inner(&mut input, &tx, &output_schema).await;
        if let Err(e) = result {
            let _ = tx
                .send(Err(datafusion::error::DataFusionError::External(Box::new(
                    e,
                ))))
                .await;
        }
        Ok(())
    }

    async fn run_inner(
        &self,
        input: &mut SendableRecordBatchStream,
        tx: &Sender<datafusion::error::Result<RecordBatch>>,
        output_schema: &SchemaRef,
    ) -> Result<()> {
        // Load model (or get from cache)
        let guard = self
            .model_cache
            .get_or_load(&self.source, self.task, self.backend)
            .await?;

        // Create output adapter for this task
        let adapter = create_adapter(self.task, &guard.model)?;

        // Track dynamic batch size
        let mut current_batch_size = self.batch_size;

        // Process input stream
        while let Some(input_batch) = input.next().await {
            let input_batch = input_batch.map_err(|e| JammiError::Inference(e.to_string()))?;

            let content = extract_columns(&input_batch, &self.content_columns)?;
            let keys = extract_column(&input_batch, &self.key_column)?;
            let row_count = input_batch.num_rows();

            // Process in sub-batches
            for chunk_start in (0..row_count).step_by(current_batch_size) {
                let chunk_end = (chunk_start + current_batch_size).min(row_count);
                let chunk_len = chunk_end - chunk_start;
                let chunk_content = slice_columns(&content, chunk_start, chunk_len);
                let chunk_keys = keys.slice(chunk_start, chunk_len);

                let start = Instant::now();
                let output_batch = self
                    .process_chunk(
                        &guard.model,
                        adapter.as_ref(),
                        &chunk_content,
                        &chunk_keys,
                        &mut current_batch_size,
                        output_schema,
                    )
                    .await?;
                let elapsed = start.elapsed();

                // Notify observer
                if let Some(obs) = &self.observer {
                    obs.on_batch(&output_batch, &self.source.to_string(), elapsed);
                }

                if tx.send(Ok(output_batch)).await.is_err() {
                    // Receiver dropped (query cancelled)
                    return Ok(());
                }
            }
        }

        Ok(())
    }

    /// Process one chunk through the model with error handling.
    async fn process_chunk(
        &self,
        model: &LoadedModel,
        adapter: &dyn OutputAdapter,
        content: &[ArrayRef],
        keys: &ArrayRef,
        current_batch_size: &mut usize,
        output_schema: &SchemaRef,
    ) -> Result<RecordBatch> {
        let row_count = keys.len();
        let start = Instant::now();

        match model.forward(content, self.task) {
            Ok(raw_output) => {
                let latency_ms = start.elapsed().as_secs_f32() * 1000.0;
                self.build_output_batch(
                    keys,
                    &raw_output,
                    adapter,
                    row_count,
                    latency_ms,
                    output_schema,
                )
            }
            Err(e) if Self::is_oom_error(&e) => {
                self.handle_oom(
                    model,
                    adapter,
                    content,
                    keys,
                    current_batch_size,
                    output_schema,
                )
                .await
            }
            Err(e) => {
                tracing::warn!(rows = row_count, "Batch inference failed: {e}");
                *current_batch_size = (*current_batch_size / 2).max(1);
                self.build_error_batch(keys, &e.to_string(), row_count, output_schema)
            }
        }
    }

    fn is_oom_error(e: &JammiError) -> bool {
        let msg = e.to_string().to_lowercase();
        msg.contains("out of memory") || msg.contains("oom") || msg.contains("cuda")
    }

    async fn handle_oom(
        &self,
        model: &LoadedModel,
        adapter: &dyn OutputAdapter,
        content: &[ArrayRef],
        keys: &ArrayRef,
        current_batch_size: &mut usize,
        output_schema: &SchemaRef,
    ) -> Result<RecordBatch> {
        let row_count = keys.len();

        // Halve batch size up to 3 times
        for attempt in 0..3 {
            *current_batch_size = (*current_batch_size / 2).max(1);
            tracing::warn!(
                attempt,
                new_batch_size = *current_batch_size,
                "GPU OOM, halving batch size"
            );

            let smaller_len = (*current_batch_size).min(row_count);
            let smaller_content = slice_columns(content, 0, smaller_len);
            let smaller_keys = keys.slice(0, smaller_len);

            match model.forward(&smaller_content, self.task) {
                Ok(raw_output) => {
                    return self.build_output_batch(
                        &smaller_keys,
                        &raw_output,
                        adapter,
                        smaller_len,
                        0.0,
                        output_schema,
                    );
                }
                Err(e) if Self::is_oom_error(&e) && *current_batch_size > 1 => continue,
                Err(e) => {
                    return self.build_error_batch(keys, &e.to_string(), row_count, output_schema);
                }
            }
        }
        self.build_error_batch(
            keys,
            "GPU OOM persists at minimum batch size",
            row_count,
            output_schema,
        )
    }

    /// Build an output RecordBatch from a successful model forward pass.
    fn build_output_batch(
        &self,
        keys: &ArrayRef,
        raw_output: &BackendOutput,
        adapter: &dyn OutputAdapter,
        row_count: usize,
        latency_ms: f32,
        output_schema: &SchemaRef,
    ) -> Result<RecordBatch> {
        let prefix = build_prefix_columns(
            keys,
            &self.source_id,
            &self.source.to_string(),
            &raw_output.row_status,
            &raw_output.row_errors,
            latency_ms,
            row_count,
        );
        let task_columns = adapter.adapt(raw_output, row_count)?;

        let mut all_columns = prefix;
        all_columns.extend(task_columns);

        RecordBatch::try_new(Arc::clone(output_schema), all_columns)
            .map_err(|e| JammiError::Inference(format!("Failed to build output batch: {e}")))
    }

    /// Build an all-error RecordBatch when the entire chunk fails.
    fn build_error_batch(
        &self,
        keys: &ArrayRef,
        error_message: &str,
        row_count: usize,
        output_schema: &SchemaRef,
    ) -> Result<RecordBatch> {
        let row_status = vec![false; row_count];
        let row_errors = vec![error_message.to_string(); row_count];

        // Extract embedding dim from the output schema (FixedSizeList field)
        let dim = output_schema
            .fields()
            .iter()
            .find_map(|f| match f.data_type() {
                arrow::datatypes::DataType::FixedSizeList(_, n) => Some(*n as usize),
                _ => None,
            })
            .unwrap_or(0);

        // Create a dummy BackendOutput with all-error rows
        let adapter = create_adapter_for_error(self.task, row_count, dim);
        let dummy_output = BackendOutput {
            float_outputs: adapter.0,
            string_outputs: adapter.1,
            row_status: row_status.clone(),
            row_errors: row_errors.clone(),
            shapes: vec![(row_count, 0)],
        };

        let task_adapter = super::adapter::create_adapter_for_schema(self.task, Some(dim));
        let task_columns = task_adapter.adapt(&dummy_output, row_count)?;

        let prefix = build_prefix_columns(
            keys,
            &self.source_id,
            &self.source.to_string(),
            &row_status,
            &row_errors,
            0.0,
            row_count,
        );

        let mut all_columns = prefix;
        all_columns.extend(task_columns);

        RecordBatch::try_new(Arc::clone(output_schema), all_columns)
            .map_err(|e| JammiError::Inference(format!("Failed to build error batch: {e}")))
    }
}

/// Create dummy float/string outputs for an error batch of a given task.
fn create_adapter_for_error(
    task: ModelTask,
    row_count: usize,
    embedding_dim: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<String>>) {
    match task {
        ModelTask::Embedding => {
            // EmbeddingAdapter expects float_outputs[0] with row_count * dim values
            (vec![vec![0.0; row_count * embedding_dim]], vec![])
        }
        ModelTask::Classification => (
            vec![vec![0.0; row_count]],
            vec![
                vec![String::new(); row_count],
                vec![String::new(); row_count],
            ],
        ),
        ModelTask::Summarization => (vec![], vec![vec![String::new(); row_count]]),
        ModelTask::TextGeneration => (
            vec![],
            vec![
                vec![String::new(); row_count],
                vec![String::new(); row_count],
            ],
        ),
        ModelTask::Ner => (vec![], vec![vec![String::new(); row_count]]),
        ModelTask::ObjectDetection => (vec![], vec![vec![String::new(); row_count]]),
    }
}
