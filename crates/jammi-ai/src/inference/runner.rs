use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{ArrayRef, RecordBatch};
use arrow::datatypes::SchemaRef;
use datafusion::execution::SendableRecordBatchStream;
use futures::StreamExt;
use jammi_db::error::{JammiError, Result};
use tokio::sync::mpsc::Sender;
use tracing::Instrument;

use super::adapter::{create_adapter, BackendOutput, OutputAdapter};
use super::chunk::ChunkAssembler;
use super::observer::InferenceObserver;
use super::schema::{build_prefix_columns, ORDINAL_COLUMN};
use super::{extract_column, extract_columns, slice_columns};
use crate::concurrency::GpuScheduler;
use crate::model::oom::is_oom_message;
use crate::operator::inference_exec::{InferenceRuntime, InferenceSpec};

/// Runs one partition of an `InferenceExec`: gathers its input into forward
/// chunks by `_chunk` ([`ChunkAssembler`]) and forwards each chunk — the
/// host half ([`LoadedModel::prepare`](crate::model::LoadedModel::prepare):
/// tokenisation, decoding, the upload) before the device is admitted, the
/// device half ([`LoadedModel::forward_prepared`](crate::model::LoadedModel::forward_prepared))
/// under the admission, so one partition's preparation overlaps another's
/// forward.
///
/// A model-forward failure is always systemic (a broken kernel, a
/// contiguity/PTX/dtype mismatch, or a model incapable of the requested
/// task) — it is never a per-row event, so it is never annotated as a
/// per-row `_status = error`. `run` propagates it as an `Err` sent through
/// the output stream, failing the operation loudly. The only recovery this
/// runner performs is OOM batch-halving, folded into the cursor loop in
/// `run_chunk`: it retries the SAME unsent slice at a smaller size; every
/// other forward failure, and a persistent OOM at the minimum batch size (1),
/// propagates.
pub struct InferenceRunner {
    spec: InferenceSpec,
    runtime: InferenceRuntime,
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

    /// Test oracle: the peak number of `forward()` calls observed IN FLIGHT
    /// SIMULTANEOUSLY over `source_id` since the last reset, so a test can
    /// prove the device's forward admission (`GpuScheduler::admit_forward`)
    /// actually bounds concurrency rather than merely existing. Keyed by
    /// `source_id` for the same reason as
    /// [`forward_calls_for`] — parallel sibling tests in one binary.
    static CONCURRENT_FORWARDS: OnceLock<Mutex<HashMap<String, (u64, u64)>>> = OnceLock::new();

    fn concurrency_table() -> &'static Mutex<HashMap<String, (u64, u64)>> {
        CONCURRENT_FORWARDS.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// Record one more forward call entering `source_id`'s critical section
    /// (AFTER the device admitted it), updating the running peak.
    pub(super) fn enter_forward(source_id: &str) {
        let mut t = concurrency_table()
            .lock()
            .expect("forward-concurrency table poisoned");
        let entry = t.entry(source_id.to_string()).or_insert((0, 0));
        entry.0 += 1;
        entry.1 = entry.1.max(entry.0);
    }

    /// The peer of [`enter_forward`]: one forward call over `source_id` has
    /// returned (its admission is about to release).
    pub(super) fn exit_forward(source_id: &str) {
        let mut t = concurrency_table()
            .lock()
            .expect("forward-concurrency table poisoned");
        if let Some(entry) = t.get_mut(source_id) {
            entry.0 = entry.0.saturating_sub(1);
        }
    }

    /// The maximum number of `forward()` calls observed in flight
    /// simultaneously over `source_id` since the last reset.
    pub fn peak_concurrent_forwards_for(source_id: &str) -> u64 {
        concurrency_table()
            .lock()
            .expect("forward-concurrency table poisoned")
            .get(source_id)
            .map(|&(_, peak)| peak)
            .unwrap_or(0)
    }

    /// Reset `source_id`'s concurrency counter and peak to zero.
    pub fn reset_forward_concurrency_for(source_id: &str) {
        concurrency_table()
            .lock()
            .expect("forward-concurrency table poisoned")
            .insert(source_id.to_string(), (0, 0));
    }
}

/// The two halves of a forward and the device that admits the second:
/// `prepare` is host work over a slice's content columns, `forward` the
/// device operation over what it prepared. Injected so the chunk loop is
/// unit-testable without a real model.
struct Forwarder<'a, P, F> {
    device: &'a GpuScheduler,
    prepare: P,
    forward: F,
}

/// The columns the runner reads off one chunk, or off a sub-slice of one.
struct ChunkColumns {
    content: Vec<ArrayRef>,
    keys: ArrayRef,
    passthrough: Vec<ArrayRef>,
    ordinals: ArrayRef,
}

impl ChunkColumns {
    fn of(chunk: &RecordBatch, spec: &InferenceSpec) -> Result<Self> {
        Ok(Self {
            content: extract_columns(chunk, &spec.content_columns)?,
            keys: extract_column(chunk, &spec.key_column)?,
            passthrough: extract_columns(chunk, &spec.passthrough)?,
            ordinals: extract_column(chunk, ORDINAL_COLUMN)?,
        })
    }

    fn len(&self) -> usize {
        self.keys.len()
    }

    fn slice(&self, start: usize, len: usize) -> Self {
        Self {
            content: slice_columns(&self.content, start, len),
            keys: self.keys.slice(start, len),
            passthrough: slice_columns(&self.passthrough, start, len),
            ordinals: self.ordinals.slice(start, len),
        }
    }
}

/// Everything needed to shape a successful forward's raw output into a
/// labeled `RecordBatch` and observe it — bundled because these are all
/// per-runner-invocation constants, distinct from the per-chunk batching
/// mechanics ([`ChunkColumns`], `current_batch_size`) that vary every
/// iteration of [`InferenceRunner::run_chunk`].
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
    /// A runner computing `spec` against `runtime`.
    pub fn new(spec: InferenceSpec, runtime: InferenceRuntime) -> Self {
        Self { spec, runtime }
    }

    /// Consume the input stream, forward it chunk by chunk, and send results to `tx`.
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
        let spec = &self.spec;
        // A partition that receives no rows — a fan-out wider than the
        // input's chunk count — touches neither the model cache nor the
        // adapter: the model is bound on the first batch.
        let Some(first) = input.next().await else {
            return Ok(());
        };
        let started = tracing::debug_span!("inference.start");
        let guard = self
            .runtime
            .model_cache
            .get_or_load(&spec.source, spec.task, spec.backend)
            .instrument(started.clone())
            .await?;
        let adapter = started.in_scope(|| create_adapter(spec.task, &guard.model))?;
        drop(started);
        let model_label = spec.source.to_string();
        let ctx = OutputContext {
            output_schema,
            adapter: adapter.as_ref(),
            source_id: &spec.source_id,
            model_label: &model_label,
            observer: self.runtime.observer.as_deref(),
            key_column: &spec.key_column,
        };

        // The most rows of one forward. A chunk never exceeds the budget's
        // row cap; an OOM halves this, and the shrink persists for the rest
        // of the stream.
        let mut current_batch_size = spec.chunk.rows.get();
        let mut assembler = ChunkAssembler::try_new(input.schema())?;
        let model = &guard.model;
        let task = spec.task;

        let mut forwarder = Forwarder {
            device: guard.device(),
            prepare: |content: &[ArrayRef]| model.prepare(content, task),
            forward: |prepared| model.forward_prepared(prepared),
        };

        let mut pending = Some(first);
        let mut input_ended = false;
        while !input_ended {
            let next = match pending.take() {
                Some(first) => Some(first),
                None => input.next().await,
            };
            let chunks: Vec<RecordBatch> = match next {
                // The structural classifier, never a stringification: a
                // typed refusal raised below this runner (the numbered
                // input's `InvalidKey`) must reach the caller as that variant.
                Some(batch) => assembler.push(&batch.map_err(JammiError::from)?)?,
                None => {
                    input_ended = true;
                    assembler.finish()?.into_iter().collect()
                }
            };
            let flow = self
                .run_chunks(&chunks, &mut current_batch_size, &ctx, tx, &mut forwarder)
                .await?;
            if flow.is_break() {
                break;
            }
        }
        Ok(())
    }

    /// Forward each of `chunks` in turn. `Break` once the receiver is gone.
    async fn run_chunks<P, F, T>(
        &self,
        chunks: &[RecordBatch],
        current_batch_size: &mut usize,
        ctx: &OutputContext<'_>,
        tx: &Sender<datafusion::error::Result<RecordBatch>>,
        forwarder: &mut Forwarder<'_, P, F>,
    ) -> Result<ControlFlow<()>>
    where
        P: Fn(&[ArrayRef]) -> Result<T>,
        F: FnMut(T) -> Result<BackendOutput>,
    {
        for chunk in chunks {
            let flow = Self::run_chunk(
                &ChunkColumns::of(chunk, &self.spec)?,
                current_batch_size,
                ctx,
                tx,
                forwarder,
            )
            .await?;
            if flow.is_break() {
                return Ok(flow);
            }
        }
        Ok(ControlFlow::Continue(()))
    }

    /// Drive one chunk's rows through `prepare` then `forward` in
    /// dynamically-sized sub-batches — the whole chunk in one forward unless
    /// an OOM has shrunk `current_batch_size` below it.
    ///
    /// `current_batch_size` is read fresh for both the slice length AND the
    /// cursor advance on every iteration, so a shrink from OOM recovery is
    /// never stale for one side and fresh for the other. On a successful
    /// forward the cursor advances by exactly the slice that was just sent:
    /// each successful sub-batch is sent as its own `RecordBatch`
    /// immediately. On OOM, `current_batch_size` halves (floored at 1) and
    /// the SAME unsent slice is prepared and retried, so no row is ever
    /// skipped or duplicated; a chunk was cut under a padded-token budget,
    /// and its padded width can only fall as rows are removed, so halving
    /// its rows at least halves its padded tokens. A non-OOM error, or a
    /// persistent OOM at batch size 1, propagates rather than being annotated
    /// as a per-row `_status = error` batch (see the type's doc). `Break`
    /// once the receiver is gone (the query was cancelled).
    async fn run_chunk<P, F, T>(
        chunk: &ChunkColumns,
        current_batch_size: &mut usize,
        ctx: &OutputContext<'_>,
        tx: &Sender<datafusion::error::Result<RecordBatch>>,
        forwarder: &mut Forwarder<'_, P, F>,
    ) -> Result<ControlFlow<()>>
    where
        P: Fn(&[ArrayRef]) -> Result<T>,
        F: FnMut(T) -> Result<BackendOutput>,
    {
        let row_count = chunk.len();
        let mut chunk_start = 0;

        while chunk_start < row_count {
            let chunk_len = (*current_batch_size).min(row_count - chunk_start);
            let rows = chunk.slice(chunk_start, chunk_len);

            let start = Instant::now();
            // The host half runs before the device is asked for anything,
            // so it overlaps other partitions' forwards. The device admits
            // the forward BEFORE the model is invoked, and the permit is
            // held for the forward call alone — an OOM-halving retry below
            // is prepared and admitted afresh on its next loop iteration.
            let prepared = tracing::debug_span!("forward.prepare", rows = chunk_len)
                .in_scope(|| (forwarder.prepare)(&rows.content))?;
            let admitted = forwarder.device.admit_forward().await?;
            #[cfg(feature = "test-hooks")]
            {
                test_hooks::record_forward(ctx.source_id);
                test_hooks::enter_forward(ctx.source_id);
            }
            let forward_result = tracing::debug_span!("forward.device", rows = chunk_len)
                .in_scope(|| (forwarder.forward)(prepared));
            #[cfg(feature = "test-hooks")]
            test_hooks::exit_forward(ctx.source_id);
            drop(admitted);
            match forward_result {
                Ok(raw_output) => {
                    let latency_ms = start.elapsed().as_secs_f32() * 1000.0;
                    let output_batch = tracing::debug_span!("forward.output", rows = chunk_len)
                        .in_scope(|| Self::build_output_batch(ctx, &rows, raw_output, latency_ms))?;

                    if let Some(obs) = ctx.observer {
                        obs.on_batch(&output_batch, ctx.model_label, start.elapsed());
                    }

                    if tx.send(Ok(output_batch)).await.is_err() {
                        return Ok(ControlFlow::Break(()));
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

        Ok(ControlFlow::Continue(()))
    }

    /// Only a genuine out-of-memory error gets the batch-halving retry — see
    /// [`crate::model::oom`] for the shared spelling table and why this uses
    /// the retry predicate [`is_oom_message`] (matches every table entry,
    /// including the bare `oom` token) rather than the training classifier's
    /// stricter, long-spellings-only predicate (the misroute risk, and why a
    /// false positive here is bounded/self-correcting).
    fn is_oom_error(e: &JammiError) -> bool {
        is_oom_message(&e.to_string().to_lowercase())
    }

    /// Build an output RecordBatch from a successful model forward pass.
    fn build_output_batch(
        ctx: &OutputContext<'_>,
        rows: &ChunkColumns,
        raw_output: BackendOutput,
        latency_ms: f32,
    ) -> Result<RecordBatch> {
        let row_count = rows.len();
        let prefix = build_prefix_columns(
            &rows.keys,
            ctx.key_column,
            ctx.source_id,
            ctx.model_label,
            &raw_output.row_status,
            &raw_output.row_errors,
            latency_ms,
            row_count,
            &rows.ordinals,
        )?;
        // A keyed input refuses a null key before any row reaches this
        // runner; an arrival-order input does not check keys, so this is
        // where its null key becomes the typed refusal rather than a stringly
        // `RecordBatch::try_new` "non-nullable column contains nulls".
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
        all_columns.extend(rows.passthrough.iter().cloned());

        RecordBatch::try_new(Arc::clone(ctx.output_schema), all_columns)
            .map_err(|e| JammiError::Inference(format!("Failed to build output batch: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::inference::adapter::EmbeddingAdapter;
    use crate::inference::schema::build_output_schema;
    use crate::model::ModelTask;

    /// `is_oom_error` must classify ONLY genuine out-of-memory errors. A CUDA
    /// kernel/loader failure (e.g. `INVALID_PTX`) is not OOM — misrouting it to
    /// the batch-halving retry would hide it from the caller.
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

    /// One chunk of `n` rows, numbered from 0.
    fn test_chunk(n: usize) -> ChunkColumns {
        ChunkColumns {
            content: test_content(n),
            keys: test_keys(n),
            passthrough: Vec::new(),
            ordinals: Arc::new((0..n as u64).collect::<arrow::array::UInt64Array>()),
        }
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

    /// The chunk's `_ordinal` values reach the output exactly once each and
    /// in order across EVERY sub-batch `run_chunk` sends — including across
    /// an OOM-halving retry, which resends the SAME row slice at a smaller
    /// size: the retried rows carry the ordinals their failed attempt never
    /// emitted, never a gap and never a value repeated.
    #[tokio::test]
    async fn run_chunk_ordinal_is_contiguous_across_an_oom_halving_retry() {
        let row_count = 300;
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
        let oom_threshold = 64;

        let (tx, rx) = tokio::sync::mpsc::channel(row_count);
        let flow = InferenceRunner::run_chunk(
            &test_chunk(row_count),
            &mut current_batch_size,
            &ctx,
            &tx,
            &mut Forwarder {
                device: &GpuScheduler::new_unlimited(),
                prepare: |chunk: &[ArrayRef]| Ok(chunk[0].len()),
                forward: |len| {
                    if len > oom_threshold {
                        Err(JammiError::Inference("out of memory".into()))
                    } else {
                        Ok(fake_backend_output(len))
                    }
                },
            },
        )
        .await
        .expect("run_chunk succeeds once the batch size shrinks under the OOM threshold");
        assert!(flow.is_continue(), "the receiver is still listening");
        drop(tx);

        let ordinals = drain_ordinals(rx).await;
        let expected: Vec<u64> = (0..row_count as u64).collect();
        assert_eq!(
            ordinals, expected,
            "_ordinal must be the contiguous 0..row_count sequence with no gap or repeat, \
             even though an OOM-halving retry resent one slice more than once"
        );
    }

    /// A successful OOM-halving retry must resend the FULL slice at the
    /// smaller size, and the cursor loop must read `current_batch_size`
    /// fresh on both the slice length and the advance — so a shrink never
    /// diverges from the outer cursor (a `step_by` stride with a separately
    /// halved batch size drifts apart and silently drops rows). Every one of
    /// 300 input rows must appear in the output stream exactly once.
    #[tokio::test]
    async fn run_chunk_conserves_every_row_across_oom_halving() {
        let row_count = 300;
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
        let oom_threshold = 64;

        let (tx, rx) = tokio::sync::mpsc::channel(row_count);
        let flow = InferenceRunner::run_chunk(
            &test_chunk(row_count),
            &mut current_batch_size,
            &ctx,
            &tx,
            &mut Forwarder {
                device: &GpuScheduler::new_unlimited(),
                prepare: |chunk: &[ArrayRef]| Ok(chunk[0].len()),
                forward: |len| {
                    if len > oom_threshold {
                        Err(JammiError::Inference("out of memory".into()))
                    } else {
                        Ok(fake_backend_output(len))
                    }
                },
            },
        )
        .await
        .expect("run_chunk succeeds once the batch size shrinks under the OOM threshold");
        assert!(flow.is_continue(), "the receiver is still listening");
        drop(tx);

        let mut ids = drain_row_ids(rx).await;
        ids.sort();
        let mut expected: Vec<String> = (0..row_count).map(|i| format!("row_{i}")).collect();
        expected.sort();
        assert_eq!(
            ids, expected,
            "every input row must appear exactly once — no drops, no duplicates"
        );
    }

    /// A persistent OOM that survives even at batch size 1 is an unservable
    /// resource failure — it must propagate rather than loop forever or
    /// silently drop the unservable slice, and the size must floor at 1
    /// (never reach 0, which would divide the input into an infinite number
    /// of empty chunks).
    #[tokio::test]
    async fn run_chunk_propagates_persistent_oom_at_minimum_batch_size() {
        let row_count = 10;
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

        let (tx, _rx) = tokio::sync::mpsc::channel(row_count);
        let result = InferenceRunner::run_chunk(
            &test_chunk(row_count),
            &mut current_batch_size,
            &ctx,
            &tx,
            &mut Forwarder {
                device: &GpuScheduler::new_unlimited(),
                prepare: |chunk: &[ArrayRef]| Ok(chunk[0].len()),
                forward: |_len| Err(JammiError::Inference("out of memory".into())),
            },
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

    /// A non-OOM forward failure is always systemic — it must
    /// propagate immediately, never be misrouted through the OOM-halving
    /// retry, and never emit any output batch.
    #[tokio::test]
    async fn run_chunk_propagates_non_oom_error_immediately() {
        let row_count = 10;
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

        let (tx, mut rx) = tokio::sync::mpsc::channel(row_count);
        let result = InferenceRunner::run_chunk(
            &test_chunk(row_count),
            &mut current_batch_size,
            &ctx,
            &tx,
            &mut Forwarder {
                device: &GpuScheduler::new_unlimited(),
                prepare: |chunk: &[ArrayRef]| Ok(chunk[0].len()),
                forward: |_len| Err(JammiError::Inference("shape mismatch".into())),
            },
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

    /// A partition that receives no rows never binds the model: the runner
    /// over an empty input completes without touching a cache whose model
    /// does not exist, sending nothing.
    #[tokio::test]
    async fn an_empty_partition_never_binds_the_model() {
        use crate::model::backend::DeviceConfig;
        use crate::model::cache::ModelCache;
        use crate::model::resolver::ModelResolver;
        use crate::model::ModelSource;
        use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
        use jammi_db::store::manifest::ComputeDeviceKind;
        use jammi_numerics::ChunkBudget;
        use std::num::NonZeroUsize;

        let dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(jammi_db::catalog::Catalog::open(dir.path()).await.unwrap());
        let artifacts = Arc::new(
            jammi_db::store::ArtifactStore::with_root(
                jammi_db::storage::StorageUrl::memory("empty-partition-artifacts"),
                jammi_db::storage::StorageRegistry::new(),
                dir.path().join("artifacts"),
            )
            .unwrap(),
        );
        let hub = crate::model::hub::HubSource::from_config(
            &jammi_db::config::ModelsConfig {
                hub_cache_dir: Some(dir.path().join("hub")),
                ..Default::default()
            },
            &|_: &str| None,
        )
        .unwrap();
        let resolver = ModelResolver::new(catalog, artifacts, hub).unwrap();
        let device_config = DeviceConfig {
            gpu_device: -1,
            devices: vec![-1],
            memory_fraction: 1.0,
            require_gpu: false,
            compute_precision: jammi_numerics::ComputePrecision::F32,
        };
        let runtime = InferenceRuntime {
            model_cache: Arc::new(ModelCache::new(
                resolver,
                device_config,
                Arc::new(GpuScheduler::new_unlimited()),
            )),
            observer: None,
        };
        let spec = InferenceSpec {
            source: ModelSource::Local(dir.path().join("no-such-model")),
            task: ModelTask::TextEmbedding,
            content_columns: vec!["text".into()],
            key_column: "id".into(),
            source_id: "empty-partition".into(),
            backend: None,
            chunk: ChunkBudget {
                rows: NonZeroUsize::new(8).unwrap(),
                tokens: NonZeroUsize::new(4096).unwrap(),
            },
            embedding_dim: Some(1),
            regression_form: None,
            passthrough: Vec::new(),
            device_kind: ComputeDeviceKind::Cpu,
            partitions: NonZeroUsize::new(4).unwrap(),
        };
        let input_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
            Field::new(ORDINAL_COLUMN, DataType::UInt64, false),
            Field::new(crate::inference::chunk::CHUNK_COLUMN, DataType::UInt64, false),
        ]));
        let input: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&input_schema),
            futures::stream::empty(),
        ));
        let (tx, mut rx) = tokio::sync::mpsc::channel(2);
        InferenceRunner::new(spec, runtime)
            .run(input, tx, test_output_schema())
            .await
            .unwrap();
        assert!(rx.recv().await.is_none(), "no batch and no error for no rows");
    }

    /// `forward()` concurrency is bounded by the DEVICE's admission, never by
    /// accident of scheduling. Four concurrent `run_chunk` callers — four
    /// partitions of one plan, or of several — admit through one device that
    /// takes one forward at a time, and each sleeps on a REAL OS thread while
    /// "forwarding", so overlapping calls are actually observable — a
    /// synchronous sleep, not `tokio::time::sleep`, because the injected
    /// `forward` closure is sync and must genuinely occupy a worker thread
    /// for overlap to be possible at all. Against an unbudgeted device
    /// (`GpuScheduler::new_unlimited()`) the same four callers overlap.
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn run_chunk_bounds_concurrent_forwards_to_the_device_admission() {
        let source_id = "device-admission-unit-source";
        test_hooks::reset_forward_concurrency_for(source_id);

        let device = Arc::new(GpuScheduler::new(usize::MAX, 0.0));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let device = Arc::clone(&device);
            handles.push(tokio::spawn(async move {
                let row_count = 2;
                let adapter = EmbeddingAdapter::new(1);
                let output_schema = test_output_schema();
                let ctx = OutputContext {
                    output_schema: &output_schema,
                    adapter: &adapter,
                    source_id,
                    model_label: "test-model",
                    observer: None,
                    key_column: "id",
                };
                let mut current_batch_size = row_count;
                let (tx, mut rx) = tokio::sync::mpsc::channel(row_count);
                let flow = InferenceRunner::run_chunk(
                    &test_chunk(row_count),
                    &mut current_batch_size,
                    &ctx,
                    &tx,
                    &mut Forwarder {
                        device: &device,
                        prepare: |chunk: &[ArrayRef]| Ok(chunk[0].len()),
                        forward: |len| {
                            std::thread::sleep(std::time::Duration::from_millis(40));
                            Ok(fake_backend_output(len))
                        },
                    },
                )
                .await
                .unwrap();
                assert!(flow.is_continue(), "the receiver is still listening");
                drop(tx);
                while rx.recv().await.is_some() {}
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(
            test_hooks::peak_concurrent_forwards_for(source_id),
            1,
            "a device that admits one forward never runs two at once"
        );
    }
}
