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
//! # Production wiring (current state, stated plainly)
//!
//! This module ships the data-plane PRIMITIVE — `open`/`next_chunk`, fully
//! oracled (P3–P7 below; P1's plan-shape property and P6.i's byte parity
//! against `read_back_sql` are exercised directly against this type in
//! `tests/it/training_set_stream.rs`). `TrainingLoop::run`
//! (`trainer.rs`)'s production text arm still calls the EAGER
//! `TrainingDataLoader::text_chunk_for_rank` exclusively at this commit — no
//! call site in `worker.rs` opens a `TrainingSetStream` yet. Switching the
//! W=1 non-exempt production arms onto this stream (binding `EpochSource`,
//! re-opening the stream per epoch, and re-deriving the train/validation
//! split and the K3 scaler from row COUNTS rather than an already-resident
//! `TrainingDataLoader`) is follow-on work: `trainer.rs`'s `run` is the
//! single most heavily pinned function in this crate (three byte-for-byte
//! adapter-digest oracles depend on it), and rewiring it needs its own
//! design-pressure round rather than a same-commit addition bolted onto this
//! primitive's own review. P6.ii's "unchanged with the production path
//! streaming" property is therefore vacuously true today (the production
//! path has not changed at all) rather than the stronger claim the contract
//! names — recorded here rather than left for a reader to discover by
//! grepping `worker.rs` for a call that is not there.
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
//!   SEPARATE, unfiltered pass this module never streams), the
//!   mining/GradCache whole-set loaders, and — new to this module —
//!   `crate::fine_tune::decode::DetectedFormat::Classification` (its label
//!   vocabulary is exactly the same whole-dataset-pass shape as K3; see
//!   `decode`'s module doc). **The bound is claimed only for the non-exempt
//!   configuration** (no mining, no GradCache, not classification); with an
//!   exemption active the eager loader is pool-ACCOUNTED (a typed failure
//!   still surfaces) but not BOUNDED by this doc's inequality.
//!
//! The loader's own query plans at a SINGLE output partition —
//! `Self::open` derives a loader-local one-`target_partitions`
//! `SessionState` from the caller's session (keeping its tenant analyzer
//! rule, catalogs, and memory pool) — so on a sorted single-fragment table
//! this plans a bare scan with NO `SortPreservingMergeExec` and NO
//! DataFusion-side reservation of its own: the merge term is ZERO by
//! construction, which is why it does not appear in the inequality above.
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

use super::data::TextChunk;
use super::decode::{self, ChunkAccumulator, DetectedFormat};
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
    pub async fn open(
        session: &InferenceSession,
        table: &TrainingSetTable,
        columns: &[String],
        task: ModelTask,
        window: RowWindow,
        slice: Slice,
        cfg: StreamConfig,
    ) -> Result<Self> {
        let detected = decode::detect_training_format(columns, task)?;
        // Refuses `Classification` (and any future format with no per-step
        // shape) at OPEN, not on the first `next_chunk` — P7's "a format
        // with no row-level chunk shape is refused at open". The
        // accumulator built here is discarded; it exists only to run the
        // check.
        drop(ChunkAccumulator::new_for(detected)?);

        validate_window(session, table, columns, detected, task, window).await?;

        let base_state = session.context().state();
        let one_partition_config = base_state.config().clone().with_target_partitions(1);
        let derived_state = SessionStateBuilder::new_from_existing(base_state)
            .with_config(one_partition_config)
            .build();
        let derived_ctx = SessionContext::new_with_state(derived_state);

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
) {
    let mut step = 0usize;
    let mut range = slice.rows_for_step(window.len(), step);

    let mut acc = match ChunkAccumulator::new_for(detected) {
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
                if let Err(e) =
                    decode::append_selected_rows(detected, task, &batch, &[row_in_batch], &mut acc)
                {
                    let _ = tx.send(Err(e)).await;
                    return;
                }
            }
            local_idx += 1;
            global_idx += 1;
            if local_idx == range.end {
                let finished = std::mem::replace(
                    &mut acc,
                    match ChunkAccumulator::new_for(detected) {
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
async fn validate_window(
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
        let agg_sql = format!(
            "SELECT sum(case when \"{target_col}\" is null then 1 else 0 end) as null_count, \
             sum(case when isnan(cast(\"{target_col}\" as double)) then 1 else 0 end) as nan_count \
             FROM ({ordered_sql}) AS w LIMIT {} OFFSET {}",
            window.len(),
            window.start
        );
        let batches = session.sql(&agg_sql).await?;
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
