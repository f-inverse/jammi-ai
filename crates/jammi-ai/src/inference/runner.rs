use std::sync::Arc;
use std::time::Instant;

use arrow::array::{ArrayRef, RecordBatch};
use arrow::datatypes::SchemaRef;
use datafusion::execution::SendableRecordBatchStream;
use futures::StreamExt;
use jammi_db::error::{JammiError, Result};
use tokio::sync::mpsc::Sender;

use super::adapter::{create_adapter, BackendOutput, OutputAdapter};
use super::observer::InferenceObserver;
use super::schema::build_prefix_columns;
use super::{extract_column, extract_columns, slice_columns};
use crate::model::cache::ModelCache;
use crate::model::{BackendType, LoadedModel, ModelSource, ModelTask};

/// Processes input RecordBatches through a model, handling batching and
/// dynamic batch sizing.
///
/// A model-forward failure is always systemic (a broken kernel, a
/// contiguity/PTX/dtype mismatch, or a model incapable of the requested
/// task) — it is never a per-row event, so it is never annotated as a
/// per-row `_status = error`. `run` propagates it as an `Err` sent through
/// the output stream, failing the operation loudly. The only recovery this
/// runner performs is OOM batch-halving (see [`Self::handle_oom`]), which
/// retries the SAME chunk at a smaller size; every other forward failure,
/// and a persistent OOM at the minimum batch size, propagates.
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
            if tx
                .send(Err(datafusion::error::DataFusionError::External(Box::new(
                    e,
                ))))
                .await
                .is_err()
            {
                tracing::warn!("Failed to send inference error to receiver (query cancelled)");
            }
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
            // A non-OOM forward failure is always systemic — a broken kernel,
            // a contiguity/PTX/dtype mismatch, or a model that cannot serve
            // the requested task at all — never a per-row event (per-row
            // input validation happens PRE-forward and sets
            // `row_status[i] = false` without an `Err`; over-long text is
            // truncated, not errored). Halving the batch size cannot isolate
            // a bad row here — it only shrinks FUTURE chunks, the current
            // chunk is never retried — so it is pointless for a systemic
            // failure. Propagate instead of annotating an
            // all-`_status = error` batch: `_status = error` means "this
            // row's input was bad", never "the model is broken".
            Err(e) => Err(e),
        }
    }

    fn is_oom_error(e: &JammiError) -> bool {
        // Only a genuine out-of-memory error gets the batch-halving retry. A bare
        // "cuda" substring is NOT OOM — it also matches kernel / loader failures
        // such as `CUDA_ERROR_INVALID_PTX`, which halving the batch cannot fix
        // and which must not be misrouted through the OOM path (#319). Match the
        // OOM spellings across backends: `out of memory` (spaces), the CUDA
        // `CUDA_ERROR_OUT_OF_MEMORY` (underscores), candle's `OutOfMemory`, and
        // the bare `oom` token.
        let msg = e.to_string().to_lowercase();
        msg.contains("out of memory")
            || msg.contains("out_of_memory")
            || msg.contains("outofmemory")
            || msg.contains("oom")
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
                // A non-OOM failure discovered mid-retry is systemic (see the
                // matching arm in `process_chunk`) — propagate rather than
                // annotate.
                Err(e) => return Err(e),
            }
        }
        // Persistent OOM at the minimum batch size is an unservable resource
        // failure — it must surface to the caller, not be annotated as a
        // per-row `_status = error` batch.
        Err(JammiError::Inference(
            "GPU OOM persists at minimum batch size".into(),
        ))
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `is_oom_error` must classify ONLY genuine out-of-memory errors. A CUDA
    /// kernel/loader failure (e.g. `INVALID_PTX`) is not OOM — misrouting it to
    /// the batch-halving retry (and never surfacing it) is #319.
    #[test]
    fn is_oom_error_matches_only_real_oom() {
        let oom = |m: &str| InferenceRunner::is_oom_error(&JammiError::Inference(m.into()));
        // Genuine OOM — including the CUDA OOM spelling — is caught.
        assert!(oom("CUDA_ERROR_OUT_OF_MEMORY"));
        assert!(oom("out of memory"));
        assert!(oom("GPU OOM at batch 4"));
        // Non-OOM CUDA failures must NOT be treated as OOM.
        assert!(!oom("CUDA_ERROR_INVALID_PTX"));
        assert!(!oom("a cuda kernel launch failed"));
        assert!(!oom("cuDNN not available"));
        assert!(!oom("shape mismatch"));
    }
}
