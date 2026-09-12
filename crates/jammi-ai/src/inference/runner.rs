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
    /// Input columns copied verbatim to the end of every emitted sub-batch
    /// (see `schema::build_output_schema`'s `passthrough`).
    passthrough: Vec<String>,
}

/// Test-only observability: a process-global count of `forward()` calls,
/// keyed by the scanned `source_id`, so an oracle can prove a refusal happened
/// BEFORE the model was ever invoked for ITS source (the
/// `KeyCheckExec`-below-the-sort contract) while sibling tests in the same
/// binary keep forwarding over their own sources in parallel — a single
/// unkeyed total would be racy in a parallel test binary.
#[cfg(feature = "test-hooks")]
pub mod test_hooks {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static FORWARD_CALLS: OnceLock<Mutex<HashMap<String, u64>>> = OnceLock::new();

    fn table() -> &'static Mutex<HashMap<String, u64>> {
        FORWARD_CALLS.get_or_init(|| Mutex::new(HashMap::new()))
    }

    pub(super) fn record_forward(source_id: &str) {
        *table()
            .lock()
            .expect("forward-call table poisoned")
            .entry(source_id.to_string())
            .or_insert(0) += 1;
    }

    /// The number of `forward()` calls over `source_id` since the last reset.
    pub fn forward_calls_for(source_id: &str) -> u64 {
        table()
            .lock()
            .expect("forward-call table poisoned")
            .get(source_id)
            .copied()
            .unwrap_or(0)
    }

    /// Reset `source_id`'s counter to zero.
    pub fn reset_forward_calls_for(source_id: &str) {
        table()
            .lock()
            .expect("forward-call table poisoned")
            .insert(source_id.to_string(), 0);
    }
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
    /// The key column's name, for the defensive null-key refusal.
    key_column: &'a str,
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
            passthrough: Vec::new(),
        }
    }

    /// Name input columns copied verbatim to the end of every emitted
    /// sub-batch, in this order (after the task columns).
    pub fn with_passthrough(mut self, passthrough: Vec<String>) -> Self {
        self.passthrough = passthrough;
        self
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
        // `_ordinal`'s running counter: one sequence for this WHOLE `run`
        // invocation (this operator is `Partitioning::UnknownPartitioning(1)`
        // — exactly one stream, never reset per input batch or per
        // OOM-halved sub-batch). See `schema::common_prefix_fields`'s doc.
        let mut next_ordinal: u64 = 0;
        let model_label = self.source.to_string();
        let task = self.task;
        let model = &guard.model;

        // Process input stream
        while let Some(input_batch) = input.next().await {
            // The structural classifier, never a stringification: a typed
            // refusal raised below this runner (`KeyCheckExec`'s
            // `InvalidKey`) must reach the caller as that variant.
            let input_batch = input_batch.map_err(JammiError::from)?;

            let content = extract_columns(&input_batch, &self.content_columns)?;
            let keys = extract_column(&input_batch, &self.key_column)?;
            let passthrough = extract_columns(&input_batch, &self.passthrough)?;

            let ctx = OutputContext {
                output_schema,
                adapter: adapter.as_ref(),
                source_id: &self.source_id,
                model_label: &model_label,
                observer: self.observer.as_deref(),
                key_column: &self.key_column,
            };

            Self::run_chunks(
                &content,
                &keys,
                &passthrough,
                &mut current_batch_size,
                &mut next_ordinal,
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
    #[allow(clippy::too_many_arguments)]
    async fn run_chunks<F>(
        content: &[ArrayRef],
        keys: &ArrayRef,
        passthrough: &[ArrayRef],
        current_batch_size: &mut usize,
        next_ordinal: &mut u64,
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
            let chunk_passthrough = slice_columns(passthrough, chunk_start, chunk_len);

            let start = Instant::now();
            #[cfg(feature = "test-hooks")]
            test_hooks::record_forward(ctx.source_id);
            match forward(&chunk_content) {
                Ok(raw_output) => {
                    let latency_ms = start.elapsed().as_secs_f32() * 1000.0;
                    let output_batch = Self::build_output_batch(
                        ctx,
                        &chunk_keys,
                        &chunk_passthrough,
                        &raw_output,
                        chunk_len,
                        latency_ms,
                        *next_ordinal,
                    )?;

                    if let Some(obs) = ctx.observer {
                        obs.on_batch(&output_batch, ctx.model_label, start.elapsed());
                    }

                    if tx.send(Ok(output_batch)).await.is_err() {
                        // Receiver dropped (query cancelled).
                        return Ok(());
                    }
                    // Advance the SAME counter `build_output_batch` just read
                    // — only on a batch that was actually sent (never on the
                    // OOM-retry arm below, which resends this identical
                    // slice at a smaller size: retried rows must reuse the
                    // ordinals their failed attempt never emitted).
                    *next_ordinal += chunk_len as u64;
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
    #[allow(clippy::too_many_arguments)]
    fn build_output_batch(
        ctx: &OutputContext<'_>,
        keys: &ArrayRef,
        passthrough: &[ArrayRef],
        raw_output: &BackendOutput,
        row_count: usize,
        latency_ms: f32,
        ordinal_start: u64,
    ) -> Result<RecordBatch> {
        let prefix = build_prefix_columns(
            keys,
            ctx.source_id,
            ctx.model_label,
            &raw_output.row_status,
            &raw_output.row_errors,
            latency_ms,
            row_count,
            ordinal_start,
        )?;
        // Defensive only: `KeyCheckExec` below the blocking sort refuses a
        // null key before any row reaches this runner, so this is unreachable
        // on every planned path — but a hand-built plan that bypasses it must
        // still get the typed refusal, never a stringly `RecordBatch::try_new`
        // "non-nullable column contains nulls".
        let null_keys = prefix[0].null_count();
        if null_keys > 0 {
            return Err(JammiError::InvalidKey {
                column: ctx.key_column.to_string(),
                null_count: null_keys as u64,
            });
        }
        let task_columns = ctx.adapter.adapt(raw_output, row_count)?;

        let mut all_columns = prefix;
        all_columns.extend(task_columns);
        all_columns.extend(passthrough.iter().cloned());

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
            &[],
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

    /// Drains every batch off `rx` and returns the `_ordinal` values in the
    /// order they were sent.
    async fn drain_ordinals(
        mut rx: tokio::sync::mpsc::Receiver<datafusion::error::Result<RecordBatch>>,
    ) -> Vec<u64> {
        let mut ordinals = Vec::new();
        while let Some(batch) = rx.recv().await {
            let batch = batch.expect("no error batches expected on the success path");
            let col = batch
                .column_by_name("_ordinal")
                .expect("_ordinal column")
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
                .expect("_ordinal is UInt64")
                .clone();
            ordinals.extend(col.values().iter().copied());
        }
        ordinals
    }

    /// `_ordinal` is a contiguous, gap-free 0-based sequence across EVERY
    /// sub-batch `run_chunks` sends for one stream — including across an
    /// OOM-halving retry, which resends the SAME row slice at a smaller
    /// size: the retried rows must get the ordinals their failed attempt
    /// never emitted, never a gap and never a value reused. Verified by
    /// reverting `run_chunks`' "advance only on a batch that was actually
    /// sent" placement (advancing `next_ordinal` before the OOM-retry check
    /// instead of after it): this test goes RED with a gap in the sequence
    /// where the failed, retried attempt's ordinals were burned and never
    /// reassigned.
    #[tokio::test]
    async fn run_chunks_ordinal_is_contiguous_across_an_oom_halving_retry() {
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
            key_column: "id",
        };
        let mut current_batch_size = 100;
        let mut next_ordinal = 0u64;
        let oom_threshold = 64;

        let (tx, rx) = tokio::sync::mpsc::channel(row_count);
        InferenceRunner::run_chunks(
            &content,
            &keys,
            &[],
            &mut current_batch_size,
            &mut next_ordinal,
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

        let ordinals = drain_ordinals(rx).await;
        let expected: Vec<u64> = (0..row_count as u64).collect();
        assert_eq!(
            ordinals, expected,
            "_ordinal must be the contiguous 0..row_count sequence with no gap or repeat, \
             even though an OOM-halving retry resent one slice more than once"
        );
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
            key_column: "id",
        };
        let mut current_batch_size = 100;
        let mut next_ordinal = 0u64;
        let oom_threshold = 64;

        let (tx, rx) = tokio::sync::mpsc::channel(row_count);
        InferenceRunner::run_chunks(
            &content,
            &keys,
            &[],
            &mut current_batch_size,
            &mut next_ordinal,
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
            key_column: "id",
        };
        let mut current_batch_size = 4;
        let mut next_ordinal = 0u64;

        let (tx, _rx) = tokio::sync::mpsc::channel(row_count);
        let result = InferenceRunner::run_chunks(
            &content,
            &keys,
            &[],
            &mut current_batch_size,
            &mut next_ordinal,
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
            key_column: "id",
        };
        let mut current_batch_size = 4;
        let mut next_ordinal = 0u64;

        let (tx, mut rx) = tokio::sync::mpsc::channel(row_count);
        let result = InferenceRunner::run_chunks(
            &content,
            &keys,
            &[],
            &mut current_batch_size,
            &mut next_ordinal,
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
