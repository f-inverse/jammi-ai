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
use crate::model::oom::is_oom_message;
use crate::model::{BackendType, ModelSource, ModelTask};

/// Processes input RecordBatches through a model, handling batching and
/// dynamic batch sizing.
///
/// A model-forward failure is always systemic (a broken kernel, a
/// contiguity/PTX/dtype mismatch, or a model incapable of the requested
/// task) — it is never a per-row event, so it is never annotated as a
/// per-row `_status = error`. `run` propagates it as an `Err` sent through
/// the output stream, failing the operation loudly. The only recovery this
/// runner performs is OOM batch-halving, folded into the cursor loop in
/// `run_chunks`: it retries the SAME unsent slice at a
/// smaller size; every other forward failure, and a persistent OOM at the
/// minimum batch size (1), propagates.
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

/// Everything needed to shape a successful forward's raw output into a
/// labeled `RecordBatch` and observe it — bundled because these are all
/// per-runner-invocation constants, distinct from the per-chunk batching
/// mechanics (`content`, `keys`, `current_batch_size`) that vary every
/// iteration of [`InferenceRunner::run_chunks`].
struct OutputContext<'a> {
    output_schema: &'a SchemaRef,
    adapter: &'a dyn OutputAdapter,
    source_id: &'a str,
    model_label: &'a str,
    observer: Option<&'a dyn InferenceObserver>,
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

        // Track dynamic batch size. A shrink from OOM recovery persists
        // across input batches (never grows back), so this is threaded
        // through every call to `run_chunks` below. Floored at 1: a misconfigured
        // batch size of 0 would make the cursor advance by 0 and spin forever
        // (the cursor loop replaced the old `step_by`, which panicked on 0) — a
        // silent hang is worse than a loud error, so treat 0 as 1.
        let mut current_batch_size = self.batch_size.max(1);
        let model_label = self.source.to_string();
        let task = self.task;
        let model = &guard.model;

        // Process input stream
        while let Some(input_batch) = input.next().await {
            let input_batch = input_batch.map_err(|e| JammiError::Inference(e.to_string()))?;

            let content = extract_columns(&input_batch, &self.content_columns)?;
            let keys = extract_column(&input_batch, &self.key_column)?;

            let ctx = OutputContext {
                output_schema,
                adapter: adapter.as_ref(),
                source_id: &self.source_id,
                model_label: &model_label,
                observer: self.observer.as_deref(),
            };

            Self::run_chunks(
                &content,
                &keys,
                &mut current_batch_size,
                &ctx,
                tx,
                |chunk_content| model.forward(chunk_content, task),
            )
            .await?;
        }

        Ok(())
    }

    /// Drive one input batch's rows through `forward` in dynamically-sized
    /// sub-batches.
    ///
    /// `current_batch_size` is read fresh for both the slice length AND the
    /// cursor advance on every iteration, so a shrink from OOM recovery is
    /// never stale for one side and fresh for the other (the old
    /// `step_by(current_batch_size)` + mutable-halving split let the two
    /// diverge and silently dropped rows). On a successful forward the
    /// cursor advances by exactly the slice that was just sent — no
    /// concatenation, no batching-of-batches: each successful sub-batch is
    /// sent as its own `RecordBatch` immediately. On OOM,
    /// `current_batch_size` halves (floored at 1) and the SAME unsent slice
    /// is retried, so no row is ever skipped or duplicated. A non-OOM error,
    /// or a persistent OOM at batch size 1, propagates rather than being
    /// annotated as a per-row `_status = error` batch (see the module doc
    /// comment). `forward` is injected so this control flow is unit-testable
    /// without a real model.
    async fn run_chunks<F>(
        content: &[ArrayRef],
        keys: &ArrayRef,
        current_batch_size: &mut usize,
        ctx: &OutputContext<'_>,
        tx: &Sender<datafusion::error::Result<RecordBatch>>,
        mut forward: F,
    ) -> Result<()>
    where
        F: FnMut(&[ArrayRef]) -> Result<BackendOutput>,
    {
        let row_count = keys.len();
        let mut chunk_start = 0;

        while chunk_start < row_count {
            let chunk_len = (*current_batch_size).min(row_count - chunk_start);
            let chunk_content = slice_columns(content, chunk_start, chunk_len);
            let chunk_keys = keys.slice(chunk_start, chunk_len);

            let start = Instant::now();
            match forward(&chunk_content) {
                Ok(raw_output) => {
                    let latency_ms = start.elapsed().as_secs_f32() * 1000.0;
                    let output_batch = Self::build_output_batch(
                        ctx,
                        &chunk_keys,
                        &raw_output,
                        chunk_len,
                        latency_ms,
                    )?;

                    if let Some(obs) = ctx.observer {
                        obs.on_batch(&output_batch, ctx.model_label, start.elapsed());
                    }

                    if tx.send(Ok(output_batch)).await.is_err() {
                        // Receiver dropped (query cancelled).
                        return Ok(());
                    }
                    chunk_start += chunk_len;
                }
                Err(e) if Self::is_oom_error(&e) && *current_batch_size > 1 => {
                    *current_batch_size = (*current_batch_size / 2).max(1);
                    tracing::warn!(
                        new_batch_size = *current_batch_size,
                        "GPU OOM, halving batch size"
                    );
                    // Do NOT advance chunk_start — retry the same slice at
                    // the smaller size.
                }
                // A non-OOM forward failure is always systemic — a broken
                // kernel, a contiguity/PTX/dtype mismatch, or a model that
                // cannot serve the requested task at all — never a per-row
                // event (per-row input validation happens PRE-forward and
                // sets `row_status[i] = false` without an `Err`; over-long
                // text is truncated, not errored). And a persistent OOM at
                // batch size 1 is an unservable resource failure. Propagate
                // instead of annotating an all-`_status = error` batch:
                // `_status = error` means "this row's input was bad", never
                // "the model is broken" or "the GPU has no memory left".
                Err(e) => return Err(e),
            }
        }

        Ok(())
    }

    /// Only a genuine out-of-memory error gets the batch-halving retry — see
    /// [`crate::model::oom`] for the shared spelling table and why this uses
    /// the retry predicate [`is_oom_message`] (matches every table entry,
    /// including the bare `oom` token) rather than the training classifier's
    /// stricter, long-spellings-only predicate (#319's misroute risk, and
    /// why a false positive here is bounded/self-correcting).
    fn is_oom_error(e: &JammiError) -> bool {
        is_oom_message(&e.to_string().to_lowercase())
    }

    /// Build an output RecordBatch from a successful model forward pass.
    fn build_output_batch(
        ctx: &OutputContext<'_>,
        keys: &ArrayRef,
        raw_output: &BackendOutput,
        row_count: usize,
        latency_ms: f32,
    ) -> Result<RecordBatch> {
        let prefix = build_prefix_columns(
            keys,
            ctx.source_id,
            ctx.model_label,
            &raw_output.row_status,
            &raw_output.row_errors,
            latency_ms,
            row_count,
        );
        let task_columns = ctx.adapter.adapt(raw_output, row_count)?;

        let mut all_columns = prefix;
        all_columns.extend(task_columns);

        RecordBatch::try_new(Arc::clone(ctx.output_schema), all_columns)
            .map_err(|e| JammiError::Inference(format!("Failed to build output batch: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{Array, StringArray};
    use arrow::datatypes::Schema;

    use super::*;
    use crate::inference::adapter::EmbeddingAdapter;
    use crate::inference::schema::build_output_schema;

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

    fn fake_backend_output(len: usize) -> BackendOutput {
        BackendOutput {
            float_outputs: vec![vec![1.0; len]],
            string_outputs: vec![],
            row_status: vec![true; len],
            row_errors: vec![String::new(); len],
            shapes: vec![(len, 1)],
        }
    }

    fn test_keys(n: usize) -> ArrayRef {
        Arc::new(StringArray::from(
            (0..n).map(|i| format!("row_{i}")).collect::<Vec<_>>(),
        ))
    }

    fn test_content(n: usize) -> Vec<ArrayRef> {
        vec![Arc::new(StringArray::from(vec!["x"; n])) as ArrayRef]
    }

    fn test_output_schema() -> SchemaRef {
        build_output_schema(
            &ModelTask::TextEmbedding,
            &Arc::new(Schema::empty()),
            "id",
            Some(1),
            None,
        )
        .expect("schema builds")
    }

    /// Drains every batch off `rx` (failing loudly on a stream-level `Err`)
    /// and returns the `_row_id` values in the order they were sent.
    async fn drain_row_ids(
        mut rx: tokio::sync::mpsc::Receiver<datafusion::error::Result<RecordBatch>>,
    ) -> Vec<String> {
        let mut ids = Vec::new();
        while let Some(batch) = rx.recv().await {
            let batch = batch.expect("no error batches expected on the success path");
            let row_id_col = batch
                .column_by_name("_row_id")
                .expect("_row_id column")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("_row_id is Utf8")
                .clone();
            ids.extend(
                row_id_col
                    .iter()
                    .map(|v| v.expect("non-null row id").to_string()),
            );
        }
        ids
    }

    /// #330: a successful OOM-halving retry must resend the FULL slice at the
    /// smaller size, and the cursor loop must read `current_batch_size`
    /// fresh on both the slice length and the advance — so a shrink never
    /// diverges from the outer cursor (the old `step_by` + mutable-halving
    /// split let the two drift apart and silently dropped rows, empirically
    /// 100 of 300). Every one of 300 input rows must appear in the output
    /// stream exactly once.
    #[tokio::test]
    async fn run_chunks_conserves_every_row_across_oom_halving() {
        let row_count = 300;
        let keys = test_keys(row_count);
        let content = test_content(row_count);
        let adapter = EmbeddingAdapter::new(1);
        let output_schema = test_output_schema();
        let ctx = OutputContext {
            output_schema: &output_schema,
            adapter: &adapter,
            source_id: "test-source",
            model_label: "test-model",
            observer: None,
        };
        let mut current_batch_size = 100;
        let oom_threshold = 64;

        let (tx, rx) = tokio::sync::mpsc::channel(row_count);
        InferenceRunner::run_chunks(
            &content,
            &keys,
            &mut current_batch_size,
            &ctx,
            &tx,
            |chunk| {
                let len = chunk[0].len();
                if len > oom_threshold {
                    Err(JammiError::Inference("out of memory".into()))
                } else {
                    Ok(fake_backend_output(len))
                }
            },
        )
        .await
        .expect("run_chunks succeeds once the batch size shrinks under the OOM threshold");
        drop(tx);

        let mut ids = drain_row_ids(rx).await;
        ids.sort();
        let mut expected: Vec<String> = (0..row_count).map(|i| format!("row_{i}")).collect();
        expected.sort();
        assert_eq!(
            ids, expected,
            "every input row must appear exactly once — no drops (#330), no duplicates"
        );
    }

    /// A persistent OOM that survives even at batch size 1 is an unservable
    /// resource failure — it must propagate rather than loop forever or
    /// silently drop the unservable slice, and the size must floor at 1
    /// (never reach 0, which would divide the input into an infinite number
    /// of empty chunks).
    #[tokio::test]
    async fn run_chunks_propagates_persistent_oom_at_minimum_batch_size() {
        let row_count = 10;
        let keys = test_keys(row_count);
        let content = test_content(row_count);
        let adapter = EmbeddingAdapter::new(1);
        let output_schema = test_output_schema();
        let ctx = OutputContext {
            output_schema: &output_schema,
            adapter: &adapter,
            source_id: "test-source",
            model_label: "test-model",
            observer: None,
        };
        let mut current_batch_size = 4;

        let (tx, _rx) = tokio::sync::mpsc::channel(row_count);
        let result = InferenceRunner::run_chunks(
            &content,
            &keys,
            &mut current_batch_size,
            &ctx,
            &tx,
            |_chunk| Err(JammiError::Inference("out of memory".into())),
        )
        .await;

        assert!(
            result.is_err(),
            "persistent OOM at batch size 1 must propagate"
        );
        assert_eq!(
            current_batch_size, 1,
            "batch size halves down to a floor of 1, never below"
        );
    }

    /// #331: a non-OOM forward failure is always systemic — it must
    /// propagate immediately, never be misrouted through the OOM-halving
    /// retry, and never emit any output batch.
    #[tokio::test]
    async fn run_chunks_propagates_non_oom_error_immediately() {
        let row_count = 10;
        let keys = test_keys(row_count);
        let content = test_content(row_count);
        let adapter = EmbeddingAdapter::new(1);
        let output_schema = test_output_schema();
        let ctx = OutputContext {
            output_schema: &output_schema,
            adapter: &adapter,
            source_id: "test-source",
            model_label: "test-model",
            observer: None,
        };
        let mut current_batch_size = 4;

        let (tx, mut rx) = tokio::sync::mpsc::channel(row_count);
        let result = InferenceRunner::run_chunks(
            &content,
            &keys,
            &mut current_batch_size,
            &ctx,
            &tx,
            |_chunk| Err(JammiError::Inference("shape mismatch".into())),
        )
        .await;

        assert!(result.is_err(), "a non-OOM forward failure must propagate");
        assert_eq!(
            current_batch_size, 4,
            "a non-OOM error must not trigger OOM halving"
        );
        drop(tx);
        assert!(
            rx.recv().await.is_none(),
            "a systemic failure must not emit any output batch"
        );
    }
}
