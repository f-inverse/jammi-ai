//! The per-rank, residency-bounded stream over a materialised training set
//! (#500 U2c, M3).
//!
//! [`TrainingSetStream`] reads the SAME committed order
//! [`crate::fine_tune::training_set::read_back_sql`] reads for the eager
//! path — it delegates to that function for the relation text (the reader-
//! class allow-list gains NO new entry: see `training_set.rs`'s
//! `reader_class_allow_list` module doc) — but never collects the whole read
//! into a `Vec<RecordBatch>`. A background pump walks the DataFusion stream
//! batch by batch, decodes ONLY the rows the current step's chunk needs
//! (`crate::fine_tune::decode::append_selected_rows`, the same column
//! extractors the eager loader uses), and hands finished chunks to the
//! trainer one at a time over a bounded channel.
//!
//! # Production wiring (current state, stated plainly — #500 U2c §10/§11)
//!
//! `worker.rs::run_spec`'s `TrainingSpec::FineTune` arm binds a `Streamed`
//! [`super::source::TrainingSource`] at `W = 1` for every text arm that is
//! NOT a whole-set arm (`super::source::whole_set_arm` returns `None`) —
//! mining, GradCache, the precomputed test path, and a `GraphFineTune` run
//! stay on `Resident`, the stated exemptions. `TrainingLoop::run`
//! (`trainer.rs`) dispatches on the `TrainingSource` it is handed: the
//! `Streamed` arm opens a fresh [`Self`] each epoch
//! (`TrainingLoop::open_streamed_source`) over the train window
//! (`Slice::PerRank`) and a second one for validation (`Slice::All`) —
//! `EpochSource` gives both the `Resident` and the `Streamed` arm one
//! `next_chunk` call shape, and refuses (typed) a whole-set arm ever
//! reaching this dispatch with a `Streamed` source (F6). P6.ii's "unchanged
//! with the production path streaming" property is therefore the real
//! claim, not a vacuous one: the pinned parity fixtures
//! (`refactor_parity`/`regression_refactor_parity`) run the worker end to
//! end and are produced by this stream.
//!
//! # The W× amplification (stated, not hidden)
//!
//! At world `W`, every rank re-reads and re-scans the WHOLE window (skipping
//! the rows it does not own without decoding them) — `W` independent
//! [`TrainingSetStream`]s over the same table, not `W` slices of one shared
//! read. This unit does not change that: `TrainingSetStream::open` is called
//! once per rank, and a rank's own pump never observes another rank's rows
//! (P5). U4b is what would ever run more than one rank in production.
//!
//! # The residency bound (P3), every term named
//!
//! `live_bytes(r) ≤ S₁ + (prefetch + 1)·C + carry + Σ E` where:
//! - `S₁` = ONE in-flight DataFusion scan batch (`batch_size × row_bytes`),
//!   **not pool-accounted** — DataFusion's own pipeline, stated, never
//!   claimed as bounded by this module.
//! - `C` = one chunk of `B` rows, pool-accounted via this stream's own
//!   `MemoryConsumer("training_set_stream[...]")` reservation (never the
//!   aggregate pool high-water mark, which would also count an unrelated
//!   `SortPreservingMergeExec`/`ExternalSorter` reservation running
//!   elsewhere on the same pool).
//! - `carry` ≤ `B − 1` rows pending in the in-progress
//!   `crate::fine_tune::decode::ChunkAccumulator` (pool-accounted only once
//!   the step's chunk is finished and reserved — see `Self::open`'s pump;
//!   the in-progress accumulator itself is ordinary process heap, bounded by
//!   the same `B − 1` rows).
//! - `Σ E` = the NAMED exemptions: the regression K3 scaler's whole-prefix
//!   `Vec<f32>` pass (`crate::fine_tune::regression_loss::TargetScaler`, a
//!   SEPARATE, unfiltered pass this module never streams) and the
//!   mining/GradCache whole-set loaders. Classification is NOT an exemption
//!   (#500 U2c §11 F3 reverses that): its whole-table label vocabulary
//!   (`decode::LabelVocabulary`, `build_label_vocabulary` below) is its OWN
//!   separate, bounded-by-cardinality pass — the SAME shape as K3's scalar
//!   scan, not a reason to keep the per-step CHUNK build eager, so it does
//!   not enter `Σ E` at all. **The bound is claimed only for the non-exempt
//!   configuration** (no mining, no GradCache); with an exemption active the
//!   eager loader is pool-ACCOUNTED (a typed failure still surfaces) but not
//!   BOUNDED by this doc's inequality.
//!
//! The loader's own query plans at a SINGLE output partition —
//! `Self::open` derives a loader-local one-`target_partitions`
//! `SessionState` from the caller's session (keeping its tenant analyzer
//! rule, catalogs, and memory pool) — so on a sorted single-fragment table
//! UNDER DataFusion 54.1's default `repartition_file_min_size` (~10 MiB)
//! this plans a bare scan with NO `SortPreservingMergeExec` and NO
//! DataFusion-side reservation of its own: the merge term is ZERO by
//! construction, which is why it does not appear in the inequality above.
//! **Above that threshold it is NOT zero**: a single Parquet file larger
//! than the threshold is scanned as several read-time file groups
//! regardless of `target_partitions` — `target_partitions = 1` only bounds
//! how many partitions those groups get merged DOWN to, never whether the
//! scan itself starts multi-partition — so combining them in committed
//! order needs a real `SortPreservingMergeExec`/`ExternalSorter`
//! reservation this derivation does not eliminate (found executing #500
//! U2c §11 F1's own attempted oracle: `training_set_stream.rs`'s
//! `f1_a_table_whose_eager_read_exceeds_the_pool_trains_to_completion_
//! through_the_stream` doc records six distinct configurations tried and
//! their exact failures). Every property this crate actually pins
//! (P1–P7, `refactor_parity`/`regression_refactor_parity`) is exercised at
//! fixture sizes under that threshold; a table whose EAGER size must
//! exceed `[engine] memory_limit`'s own 64 MiB floor unavoidably exceeds it
//! too, so this merge term is UNCOVERED for that combination — stated here
//! rather than left for a reader to discover empirically.
//!
//! # The load-time pre-pass (advisory, §9)
//!
//! `Self::open` runs ONE bounded-memory aggregate pass over the window
//! before ever returning a stream: a `LIMIT 0` schema probe
//! (`crate::fine_tune::decode::check_schema_matches_format`) plus, for a
//! numeric target column, a `count`/`isnan` aggregate over exactly the
//! window's rows. A column-level refusal (a binary column under a text task,
//! a non-numeric or NaN/null regression target) therefore fires BEFORE the
//! first step — a NaN target at row 60,000 refuses at `open`, not after
//! 59,999 rows of training compute. The per-step decode keeps the SAME
//! refusals as a second line (defence in depth, not the primary detection
//! path).

use std::ops::Range;

use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use futures::StreamExt;
use jammi_db::error::{JammiError, Result};
use jammi_db::store::TrainingSetTable;

use crate::model::ModelTask;
use crate::session::InferenceSession;

use super::data::{TextChunk, TrainingDataLoader};
use super::decode::{self, ChunkAccumulator, DetectedFormat, LabelVocabulary};
use super::partition::PartitionSpec;
use super::training_set::read_back_sql;

/// The bounded prefetch depth: how many completed [`OwnedChunk`]s the pump is
/// allowed to hold in the channel ahead of the consumer. `new(0)` is refused
/// — a zero-depth channel is not a "no prefetch" request, it is an
/// unrepresentable one (`tokio::sync::mpsc::channel` itself panics on `0`).
#[derive(Debug, Clone, Copy)]
pub struct StreamConfig {
    prefetch: std::num::NonZeroUsize,
}

impl StreamConfig {
    /// `prefetch` must be `>= 1`.
    pub fn new(prefetch: usize) -> Result<Self> {
        std::num::NonZeroUsize::new(prefetch)
            .map(|prefetch| Self { prefetch })
            .ok_or_else(|| {
                JammiError::FineTune("StreamConfig::new: prefetch must be >= 1, got 0".into())
            })
    }

    /// The configured prefetch depth.
    pub fn prefetch(&self) -> usize {
        self.prefetch.get()
    }
}

/// The rows a [`TrainingSetStream`] serves, as `[start, end)` over the
/// table's COMMITTED order: the training prefix is `[0, train_count)`, the
/// validation suffix is `[train_count, total)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowWindow {
    pub start: usize,
    pub end: usize,
}

impl RowWindow {
    /// A new window; `end < start` is clamped to an empty window at `start`
    /// rather than an inverted range (K2: a degenerate window is a valid
    /// zero-row state, never a panic).
    pub fn new(start: usize, end: usize) -> Self {
        Self {
            start,
            end: end.max(start),
        }
    }

    /// The number of rows this window covers.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether this window covers no rows.
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// Which rows, within a [`RowWindow`], THIS stream keeps.
#[derive(Debug, Clone)]
pub enum Slice {
    /// One rank of a [`PartitionSpec`]-partitioned world: rank `r`'s rows at
    /// each global step, per [`PartitionSpec::rows_for_step`].
    PerRank(PartitionSpec),
    /// Every row in the window, walked in consecutive `batch`-sized chunks —
    /// the validation shape (no per-rank partition; every rank validates the
    /// same held-out rows).
    All { batch: usize },
}

impl Slice {
    /// The row range (in window-LOCAL coordinates, `0..window_len`) step
    /// `step` covers.
    fn rows_for_step(&self, window_len: usize, step: usize) -> Range<usize> {
        match self {
            Slice::PerRank(spec) => spec.rows_for_step(window_len, step),
            Slice::All { batch } => {
                if *batch == 0 {
                    return 0..0;
                }
                let start = step.saturating_mul(*batch).min(window_len);
                let end = step
                    .saturating_add(1)
                    .saturating_mul(*batch)
                    .min(window_len);
                start..end.max(start)
            }
        }
    }
}

/// One decoded, owned step's chunk, plus the pool reservation backing its
/// residency — held for as long as the consumer holds this value.
#[derive(Debug)]
pub struct OwnedChunk {
    step: usize,
    chunk: TextChunk,
    // Held only for its residency lifetime: production never reads a
    // reservation's size back, only lets it grow (at construction, via
    // `emit_step`'s `split`) and shrink (at `Drop`, via `MemoryReservation`'s
    // own `Drop` impl) with this value's own lifetime. `#[allow(dead_code)]`
    // rather than a field nobody would otherwise believe is load-bearing —
    // see `reservation_bytes` below for the test-only read.
    #[allow(dead_code)]
    reservation: MemoryReservation,
}

impl OwnedChunk {
    /// The global step this chunk was decoded for.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Borrow the decoded chunk.
    pub fn chunk(&self) -> &TextChunk {
        &self.chunk
    }

    /// This chunk's own pool reservation, in bytes — `C` in P3/P4's bound.
    /// Test-only: production never reads a reservation's size back, only
    /// lets it grow/shrink with the chunk's own lifetime.
    #[cfg(any(test, feature = "test-hooks"))]
    pub fn reservation_bytes(&self) -> usize {
        self.reservation.size()
    }

    /// Consume this value into its chunk. `self.reservation` (the OTHER
    /// field) is dropped right here as this function returns — no explicit
    /// `Drop` impl on [`OwnedChunk`] is needed because
    /// [`MemoryReservation`]'s OWN `Drop` already frees its held bytes back
    /// to the pool the instant the LAST reservation over it goes out of
    /// scope; P3/P4's oracles rely on exactly this (a chunk's bytes release
    /// the moment the consumer drops it, whether via this method or by
    /// letting the whole [`OwnedChunk`] fall out of scope unread).
    pub fn into_chunk(self) -> TextChunk {
        self.chunk
    }
}

/// Rough, cheap byte-size estimate for a decoded [`TextChunk`] — the pool
/// reservation unit `C` in P3/P4's bound. Sums owned bytes (`String`/`Vec<u8>`
/// payloads, `f32`/`u32` fields at their fixed width); does not account
/// `Vec`/`String` capacity overhead, which is bounded by a small constant
/// factor over the payload for the row counts this stream chunks at.
fn chunk_byte_size(chunk: &TextChunk) -> usize {
    fn text_bytes(v: &[String]) -> usize {
        v.iter().map(String::len).sum()
    }
    fn blob_bytes(v: &[Vec<u8>]) -> usize {
        v.iter().map(Vec::len).sum()
    }
    match chunk {
        TextChunk::Contrastive {
            texts_a,
            texts_b,
            scores,
        } => text_bytes(texts_a) + text_bytes(texts_b) + scores.len() * 4,
        TextChunk::Pairs { anchors, positives } => text_bytes(anchors) + text_bytes(positives),
        TextChunk::Triplet {
            anchors,
            positives,
            negatives,
        } => text_bytes(anchors) + text_bytes(positives) + text_bytes(negatives),
        TextChunk::MediaTriplet {
            anchors,
            positives,
            negatives,
        } => blob_bytes(anchors) + blob_bytes(positives) + blob_bytes(negatives),
        TextChunk::Classification { texts, labels } => text_bytes(texts) + labels.len() * 4,
        TextChunk::Ner {
            texts,
            entities_json,
        } => text_bytes(texts) + text_bytes(entities_json),
        TextChunk::Regression { texts, targets } => text_bytes(texts) + targets.len() * 4,
    }
}

/// Finish `acc` into a chunk, reserve its bytes against `root_reservation`
/// (splitting off exactly that many bytes into the [`OwnedChunk`]'s own
/// reservation, so its `Drop` releases exactly what it grew), and send it —
/// `try_grow` failing surfaces as a typed [`JammiError::ResourcesExhausted`]
/// down the same channel the trainer reads chunks from, never a panic.
/// Returns `false` when the send failed because the receiver already
/// dropped (the consumer stopped reading) — the pump's cue to stop too.
async fn emit_step(
    tx: &tokio::sync::mpsc::Sender<Result<OwnedChunk>>,
    step: usize,
    acc: ChunkAccumulator,
    root_reservation: &mut MemoryReservation,
) -> bool {
    let chunk = acc.into_text_chunk();
    let bytes = chunk_byte_size(&chunk);
    match root_reservation.try_grow(bytes) {
        Ok(()) => {
            let chunk_reservation = root_reservation.split(bytes);
            let owned = OwnedChunk {
                step,
                chunk,
                reservation: chunk_reservation,
            };
            tx.send(Ok(owned)).await.is_ok()
        }
        Err(e) => {
            let _ = tx.send(Err(JammiError::from(e))).await;
            false
        }
    }
}

/// The per-rank, residency-bounded stream over one materialised training
/// set. See the module doc for the residency bound and the W× amplification
/// it is claimed under.
pub struct TrainingSetStream {
    receiver: tokio::sync::mpsc::Receiver<Result<OwnedChunk>>,
    pump: tokio::task::JoinHandle<()>,
}

impl std::fmt::Debug for TrainingSetStream {
    // A manual, minimal impl (the channel/task fields carry nothing worth
    // printing) — needed only so `Result<TrainingSetStream, _>::unwrap_err()`
    // is callable from a test; production never formats this value.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainingSetStream").finish_non_exhaustive()
    }
}

impl Drop for TrainingSetStream {
    fn drop(&mut self) {
        // Aborting the pump drops its task at its next await point, which
        // drops any in-progress `MemoryReservation` the pump itself was
        // holding — the residency this stream reserved is released even
        // when the consumer never reads to the end of the window.
        self.pump.abort();
    }
}

impl TrainingSetStream {
    /// Open a stream over `table`'s `columns` (the SAME projection the
    /// eager path reads), scoped to `window`, keeping only the rows `slice`
    /// selects.
    ///
    /// Runs the load-time pre-pass (schema + null/NaN aggregate, module doc)
    /// BEFORE spawning the pump, then derives a loader-local
    /// `target_partitions = 1` `SessionState` from `session`'s own state
    /// (keeping its tenant analyzer rule, catalogs, and memory pool — see the
    /// module doc's "the loader's own query plans at a single output
    /// partition"), and spawns a pump task on the current Tokio runtime that
    /// walks the ordered read [`read_back_sql`] renders, batch by batch.
    ///
    /// `label_vocab` is required (and refused, typed, when absent) exactly
    /// when `columns`/`task` detect `DetectedFormat::Classification`
    /// (#500 U2c §11 F3) — every other format ignores it. The caller builds
    /// it from a whole-table pass BEFORE calling `open` (see
    /// `super::worker::run_spec`'s doc); this function never builds one
    /// itself, since a per-window stream cannot see rows outside its own
    /// window and a vocabulary built from less than the whole table would
    /// silently under-count `num_classes`.
    #[allow(clippy::too_many_arguments)]
    pub async fn open(
        session: &InferenceSession,
        table: &TrainingSetTable,
        columns: &[String],
        task: ModelTask,
        window: RowWindow,
        slice: Slice,
        cfg: StreamConfig,
        label_vocab: Option<LabelVocabulary>,
    ) -> Result<Self> {
        let detected = decode::detect_training_format(columns, task)?;
        // Refuses a format with no per-step shape (or a Classification
        // source with no vocabulary) at OPEN, not on the first `next_chunk`
        // — P7's "a format with no row-level chunk shape is refused at
        // open". The accumulator built here is discarded; it exists only to
        // run the check.
        drop(ChunkAccumulator::new_for(detected, label_vocab.as_ref())?);

        validate_window(session, table, columns, detected, task, window).await?;

        let derived_ctx = derive_single_partition_ctx(session);

        let query = read_back_sql(table, columns);
        let df = derived_ctx.sql(&query).await?;
        let df_stream = df.execute_stream().await?;

        let pool = session.memory_pool();
        let consumer_label = match &slice {
            Slice::PerRank(_) => "per_rank",
            Slice::All { .. } => "validation",
        };
        let root_reservation =
            MemoryConsumer::new(format!("training_set_stream[{consumer_label}]")).register(&pool);

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<OwnedChunk>>(cfg.prefetch());

        let pump = tokio::spawn(run_pump(
            df_stream,
            detected,
            task,
            window,
            slice,
            root_reservation,
            tx,
            label_vocab,
        ));

        Ok(Self { receiver: rx, pump })
    }

    /// Block for the next chunk. `Ok(None)` means the pump ended without
    /// sending anything further (the channel closed) — reached only if the
    /// pump task itself was aborted or panicked before sending a terminal
    /// chunk; the normal termination path always delivers a final `Ok`
    /// chunk with `row_count() == 0`, mirroring the eager loader's contract
    /// exactly (the trainer's own `chunk.row_count() == 0` check is what
    /// actually ends an epoch).
    ///
    /// Must be called from a blocking context (`spawn_blocking`) on a
    /// `multi_thread` Tokio runtime — the pump is a genuinely async task
    /// (`send().await` backpressure), so calling this from an async task on
    /// the SAME worker would starve the pump. Production already runs the
    /// trainer this way (`worker.rs`'s blocking-training spawn).
    pub fn next_chunk(&mut self) -> Result<Option<OwnedChunk>> {
        match self.receiver.blocking_recv() {
            Some(Ok(chunk)) => Ok(Some(chunk)),
            Some(Err(e)) => Err(e),
            None => Ok(None),
        }
    }
}

/// The pump task's body: walks `df_stream` batch by batch, decoding ONLY the
/// rows `slice` selects within `window`, one [`OwnedChunk`] per global step.
///
/// # Algorithm
///
/// Two cursors: `global_idx` (rows consumed from the WHOLE ordered read, from
/// row 0) and `local_idx` (`global_idx - window.start`, valid once
/// `global_idx >= window.start`). For every row inside the window,
/// `local_idx` is compared against the CURRENT step's `range` (from
/// [`Slice::rows_for_step`]): a row inside `range` is decoded via
/// [`decode::append_selected_rows`]; a row outside it (another rank's row,
/// world > 1) is skipped — `local_idx` still advances, but nothing is
/// cloned. The instant `local_idx` reaches `range.end`, the accumulated chunk
/// is finished and sent, `step` advances, and the next range is computed; an
/// EMPTY next range (which [`PartitionSpec::rows_for_step`]'s clamp only ever
/// produces once `range.start >= window.len()`) is the terminal state: emit
/// it (an empty [`OwnedChunk`], the trainer's own end-of-epoch signal) and
/// return — this rank has nothing left to contribute regardless of how many
/// rows the underlying read still has (a sibling rank's rows), so the pump
/// does not keep draining `df_stream` to its end.
///
/// If `df_stream` itself ends (the table had fewer rows than `window`
/// promised) while `local_idx < window.len()`, that is P7's early-end
/// refusal — the ONLY way this function's post-loop code is reached, since
/// every other termination path returns from inside the loop.
#[allow(clippy::too_many_arguments)]
async fn run_pump(
    mut df_stream: datafusion::execution::SendableRecordBatchStream,
    detected: DetectedFormat,
    task: ModelTask,
    window: RowWindow,
    slice: Slice,
    mut root_reservation: MemoryReservation,
    tx: tokio::sync::mpsc::Sender<Result<OwnedChunk>>,
    label_vocab: Option<LabelVocabulary>,
) {
    let mut step = 0usize;
    let mut range = slice.rows_for_step(window.len(), step);

    let mut acc = match ChunkAccumulator::new_for(detected, label_vocab.as_ref()) {
        Ok(a) => a,
        Err(e) => {
            let _ = tx.send(Err(e)).await;
            return;
        }
    };

    if range.is_empty() {
        // A degenerate (zero-row) window: the terminal chunk is the very
        // first one.
        emit_step(&tx, step, acc, &mut root_reservation).await;
        return;
    }

    let mut global_idx = 0usize;
    let mut local_idx = 0usize;

    while let Some(batch_result) = df_stream.next().await {
        let batch = match batch_result {
            Ok(b) => b,
            Err(e) => {
                let _ = tx.send(Err(JammiError::from(e))).await;
                return;
            }
        };
        let n = batch.num_rows();
        if n == 0 {
            continue;
        }
        for row_in_batch in 0..n {
            if global_idx < window.start {
                global_idx += 1;
                continue;
            }
            if global_idx >= window.end {
                // Past the window: nothing more for ANY rank to read here.
                return;
            }
            if local_idx >= range.start && local_idx < range.end {
                if let Err(e) = decode::append_selected_rows(
                    detected,
                    task,
                    &batch,
                    &[row_in_batch],
                    label_vocab.as_ref(),
                    &mut acc,
                ) {
                    let _ = tx.send(Err(e)).await;
                    return;
                }
            }
            local_idx += 1;
            global_idx += 1;
            if local_idx == range.end {
                let finished = std::mem::replace(
                    &mut acc,
                    match ChunkAccumulator::new_for(detected, label_vocab.as_ref()) {
                        Ok(a) => a,
                        Err(e) => {
                            let _ = tx.send(Err(e)).await;
                            return;
                        }
                    },
                );
                if !emit_step(&tx, step, finished, &mut root_reservation).await {
                    return;
                }
                step += 1;
                range = slice.rows_for_step(window.len(), step);
                if range.is_empty() {
                    // Terminal: `acc` is the fresh, empty accumulator just
                    // installed above.
                    emit_step(&tx, step, acc, &mut root_reservation).await;
                    return;
                }
            }
        }
    }

    // The underlying read ended before this rank finished its own last
    // range — the table had fewer rows than `window` promised.
    let _ = tx
        .send(Err(JammiError::FineTune(format!(
            "streamed window ended after {local_idx} of {} row(s): the table has fewer rows than \
             the window [{}, {}) requires",
            window.len(),
            window.start,
            window.end
        ))))
        .await;
}

/// Derive a loader-local, single-`target_partitions` `SessionContext` from
/// `session`'s own state — keeping its tenant analyzer rule, catalogs, and
/// memory pool (the module doc's "the loader's own query plans at a SINGLE
/// output partition"). Shared by [`TrainingSetStream::open`]'s per-step read
/// AND [`validate_window`]'s aggregate pass: BOTH queries wrap `read_back_
/// sql`'s already-`ORDER BY`'d text in a `LIMIT`/`OFFSET`, and at
/// `target_partitions > 1` DataFusion plans a real `SortPreservingMergeExec`
/// for either one — for the aggregate specifically, that merge sits behind
/// a blocking `.collect()`, so it is a genuine, real pool reservation, not
/// the per-step stream's own `S₁` (DataFusion's pipeline, explicitly NOT
/// pool-accounted). Deriving once per query is cheap (`SessionState`
/// cloning, no I/O); nothing here executes a row.
fn derive_single_partition_ctx(session: &InferenceSession) -> SessionContext {
    let base_state = session.context().state();
    let one_partition_config = base_state.config().clone().with_target_partitions(1);
    let derived_state = SessionStateBuilder::new_from_existing(base_state)
        .with_config(one_partition_config)
        .build();
    SessionContext::new_with_state(derived_state)
}

/// The load-time pre-pass (module doc): a schema check via [`read_back_sql`]'s
/// own planned schema, plus — for a numeric target column — a null/NaN
/// aggregate scoped to EXACTLY `window`'s rows.
///
/// Both queries below are built by WRAPPING [`read_back_sql`]'s own text in
/// an outer `SELECT`, never by calling [`TrainingSetTable::sql_relation`]
/// directly — the reader-class allow-list's property ("every caller reaches
/// the relation only through `read_back_sql`", B4) holds for this pre-pass
/// exactly as it does for the pump: a `LIMIT`/`OFFSET` immediately wrapping
/// an already-`ORDER BY`'d subquery is DataFusion's own idiom for "the first
/// `n` rows of this order, `LIMIT` never reshuffling what `ORDER BY` fixed.
///
/// `pub(crate)` so `super::worker::run_spec` can run it directly over
/// `RowWindow::new(0, total_rows)` — the WHOLE table, once, before the
/// first training step (#500 U2c §11 F5) — in addition to [`Self::open`]
/// running it again over each stream's own (narrower) window, which is
/// stated cost, not a bug: a `Streamed` source's train window is always a
/// SUBSET of `[0, total_rows)`, so the worker's whole-table pass already
/// covers everything the per-epoch train stream's own pass would find; the
/// duplication is the "per-epoch re-open cost (two opens + pre-pass
/// planning)" the module doc already states.
pub(crate) async fn validate_window(
    session: &InferenceSession,
    table: &TrainingSetTable,
    columns: &[String],
    detected: DetectedFormat,
    task: ModelTask,
    window: RowWindow,
) -> Result<()> {
    let ordered_sql = read_back_sql(table, columns);

    // A `DataFrame`'s logical schema is known from planning alone — no rows
    // need to execute.
    let df = session.context().sql(&ordered_sql).await?;
    let schema = df.schema().as_arrow().clone();
    decode::check_schema_matches_format(&schema, detected, task)?;

    if window.is_empty() {
        return Ok(());
    }
    if let Some(target_col) = decode::numeric_target_column(detected) {
        // The `LIMIT`/`OFFSET` must scope the ROWS the aggregate reads, not
        // the aggregate's OWN one-row output: an aggregate `SELECT` over
        // `w` always produces exactly one row, so wrapping THAT in
        // `LIMIT n OFFSET window.start` (as an earlier revision of this
        // function did) discards the one row whenever `window.start > 0` —
        // silently returning zero batches for every window that does not
        // start at row 0 (the validation suffix, always). The `LIMIT`/
        // `OFFSET` therefore apply to an INNER subquery that selects the
        // window's own rows first; the aggregate runs over THAT.
        let agg_sql = format!(
            "SELECT sum(case when \"{target_col}\" is null then 1 else 0 end) as null_count, \
             sum(case when isnan(cast(\"{target_col}\" as double)) then 1 else 0 end) as nan_count \
             FROM (SELECT * FROM ({ordered_sql}) AS w LIMIT {} OFFSET {}) AS windowed",
            window.len(),
            window.start
        );
        // Executed through the SAME loader-local, single-`target_partitions`
        // context `Self::open` derives (`derive_single_partition_ctx`) —
        // never `session.sql`, which plans at the session's OWN (often > 1)
        // partition count. At > 1 the inner `LIMIT`/`OFFSET` subquery plans
        // a real `SortPreservingMergeExec` there, and unlike the per-step
        // stream (`S₁`, DataFusion's own pipeline, explicitly NOT
        // pool-accounted per the module doc) an AGGREGATE's merge sits
        // behind a blocking `.collect()` that holds every input partition's
        // buffered rows at once — a genuine, real pool reservation this
        // pre-pass would otherwise leave unnamed in P3's inequality
        // entirely. Derived once per call (cheap: `SessionState` cloning,
        // no I/O) rather than threaded in from `open` (which needs its own
        // copy anyway, for the actual per-step read after this pre-pass).
        let single_partition_ctx = derive_single_partition_ctx(session);
        let batches = single_partition_ctx.sql(&agg_sql).await?.collect().await?;
        let batch = batches.first().ok_or_else(|| {
            JammiError::FineTune(
                "TrainingSetStream pre-pass: the null/NaN aggregate returned no batch".into(),
            )
        })?;
        let null_count = decode::extract_numeric_column(batch.column(0).as_ref())
            .ok()
            .and_then(|v| v.first().copied())
            .unwrap_or(0.0);
        let nan_count = decode::extract_numeric_column(batch.column(1).as_ref())
            .ok()
            .and_then(|v| v.first().copied())
            .unwrap_or(0.0);
        if null_count > 0.0 {
            return Err(JammiError::FineTune(format!(
                "TrainingSetStream pre-pass: the training window [{}, {}) contains {null_count} \
                 null '{target_col}' value(s); a null target cannot be coerced (it would corrupt \
                 the scaler) — remove or fill the row(s) before streaming",
                window.start, window.end
            )));
        }
        if nan_count > 0.0 {
            return Err(JammiError::FineTune(format!(
                "TrainingSetStream pre-pass: the training window [{}, {}) contains {nan_count} \
                 NaN '{target_col}' value(s); a NaN target cannot be used (it would corrupt the \
                 scaler) — remove or fix the row(s) before streaming",
                window.start, window.end
            )));
        }
    }
    Ok(())
}

/// Build a classification vocabulary from a WHOLE table's `label` column,
/// via a bounded-memory forward scan (`session.sql_stream`) over
/// [`read_back_sql`]'s ordered read — never a collected `Vec<RecordBatch>`
/// (#500 U2c §11 F3). Called ONCE, by the worker, before any per-epoch
/// stream opens (`super::worker::run_spec`'s doc).
///
/// This is a PLAIN forward scan, not a [`TrainingSetStream`] pump: the
/// vocabulary's own residency is bounded by its CARDINALITY (only the
/// distinct label SET is ever held — the row values themselves are
/// discarded per-batch, never accumulated), so none of `TrainingSetStream`'s
/// per-step chunk/prefetch machinery applies here, and building the
/// vocabulary through `open`/`ChunkAccumulator` is circular anyway: a
/// Classification `ChunkAccumulator` REQUIRES a vocabulary before it can
/// be built (see `ChunkAccumulator::new_for`'s doc) — this function is
/// what produces the vocabulary that later requirement consumes.
///
/// The relation is spelled ONLY through [`read_back_sql`] (the reader-class
/// allow-list's property, `training_set.rs`'s module doc) — order does not
/// matter for a SET, but this function still reads the SAME query text
/// every other production caller does, never a second hand-spelling.
pub async fn build_label_vocabulary(
    session: &InferenceSession,
    table: &TrainingSetTable,
    columns: &[String],
) -> Result<LabelVocabulary> {
    let query = read_back_sql(table, columns);
    let mut df_stream = session.sql_stream(&query).await?;
    let mut labels: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    while let Some(batch) = df_stream.next().await {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let label_col = batch
            .column_by_name("label")
            .ok_or_else(|| JammiError::FineTune("Missing/invalid 'label' column".into()))?;
        let label_vals = decode::extract_string_column(label_col.as_ref())
            .ok_or_else(|| JammiError::FineTune("Missing/invalid 'label' column".into()))?;
        labels.extend(label_vals);
    }
    Ok(LabelVocabulary::from_labels(
        labels.iter().map(String::as_str),
    ))
}

/// One epoch's row source for the trainer's production text loop (#500 U2c
/// §10): either an already-resident [`TrainingDataLoader`] or a fresh
/// per-epoch [`TrainingSetStream`]. [`Self::next_chunk`] gives both arms one
/// call shape, so `trainer.rs::run`'s text loop walks either one identically.
pub(crate) enum EpochSource<'a> {
    Resident(&'a TrainingDataLoader),
    Stream(TrainingSetStream),
}

impl<'a> EpochSource<'a> {
    /// The chunk this source holds for global `step`. `Ok(None)` is the
    /// end-of-epoch signal — mirroring
    /// [`TrainingDataLoader::text_chunk_for_rank`]'s own "an empty chunk
    /// means the epoch is over" contract exactly, so the caller's loop looks
    /// identical for either arm.
    ///
    /// `spec` is the Resident arm's per-rank partition (the Stream arm
    /// already baked its own [`Slice::PerRank`] in at [`TrainingSetStream::
    /// open`], so `spec` goes unused there — kept as a shared parameter
    /// rather than stored twice, once on this enum and once inside the
    /// stream that built it).
    ///
    /// The Stream arm asserts `owned.step() == step` (#500 U2c §11's
    /// advisory: "the consumer asserts `owned.step() == step`") — the pump
    /// emits steps strictly in order over one channel, so any desync here is
    /// an internal invariant violation, not a caller input error.
    pub(crate) fn next_chunk(
        &mut self,
        spec: &PartitionSpec,
        step: usize,
    ) -> Result<Option<TextChunk>> {
        match self {
            EpochSource::Resident(loader) => {
                let chunk = loader.text_chunk_for_rank(spec, step)?;
                if chunk.row_count() == 0 {
                    Ok(None)
                } else {
                    Ok(Some(chunk))
                }
            }
            EpochSource::Stream(stream) => match stream.next_chunk()? {
                None => Ok(None),
                Some(owned) => {
                    assert_eq!(
                        owned.step(),
                        step,
                        "EpochSource::Stream: pump/consumer step desync (pump emitted step {}, \
                         consumer asked for step {step})",
                        owned.step()
                    );
                    if owned.chunk().row_count() == 0 {
                        Ok(None)
                    } else {
                        Ok(Some(owned.into_chunk()))
                    }
                }
            },
        }
    }
}
