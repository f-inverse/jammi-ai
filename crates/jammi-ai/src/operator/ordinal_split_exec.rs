//! `OrdinalSplitExec` — the N-way fan-out below `InferenceExec`, keyed on a
//! GLOBAL `_ordinal` assigned before fan-out.
//!
//! Every premise below holds against DataFusion 54.1's `EnforceDistribution`
//! / `EnforceSorting`.
//!
//! A design that pre-assigns each row to a fixed partition through a
//! bounded channel (a spawned background writer pushing onto per-partition
//! `tokio::mpsc` channels) is unsound for this node: `execute()` is not
//! guaranteed to be called for every partition (a single-consumer call
//! site's `.execute(0, ..)` alone calls it for exactly one), and a receiver that is never taken
//! keeps its channel open forever — the writer then wedges trying to fill that partition's channel
//! past capacity, and every OTHER partition wedges behind it, since the
//! writer is the single thread advancing all of them. See "Mechanism" below
//! for the design that makes this class of wedge structurally impossible.
//!
//! # Mechanism
//!
//! `OrdinalSplitExec::new(input, n)` requires `input` at a SINGLE partition
//! (`required_input_distribution = [Distribution::SinglePartition]`) and
//! produces `n` output partitions (`Partitioning::UnknownPartitioning(n)`).
//! There is no background task and no channel: a GENERATION (see below)
//! opens `input`'s single-partition stream lazily (on the first partition's
//! first poll of that generation) and holds it behind ONE
//! `tokio::sync::Mutex`; a partition's stream, EVERY time it is polled,
//! locks the mutex, pulls exactly the next batch off the shared upstream
//! stream, assigns it the next slice of the GLOBAL, contiguous, 0-based
//! `_ordinal` counter (fixed fold order — deterministic, no unseeded RNG),
//! and returns it — this is DEMAND-DRIVEN: a batch is handed to
//! WHICHEVER partition happens to poll next, never pre-assigned to a fixed
//! partition ahead of time.
//!
//! This makes the wedge class above UNREPRESENTABLE rather than merely
//! handled: the split's progress never depends on any consumer's polling
//! (there is nothing to block ON — a partition nobody ever polls simply
//! never advances the shared stream, and every OTHER partition is
//! unaffected, since each poll independently takes the mutex, does its
//! work, and releases it); a partition never polled receives nothing and
//! loses nothing (it never reserved anything); any subset of partitions
//! polled to completion yields every input row exactly once with a
//! contiguous ordinal partition (whichever partitions are polled simply
//! divide the shared stream's rows among themselves in poll order); and
//! `execute(0)` alone yields ALL of `input`'s rows (every batch request
//! from the shared stream is satisfied by partition 0's own polls) — which
//! is also the correct meaning for a single consumer, and is what makes
//! every single-consumer `.execute(0, ..)` call site (session::annotate_plan
//! via query::builder.rs's fluent chain in particular) see every row
//! regardless of `n`, with no separate coalesce needed above this node.
//!
//! ## Generations: state belongs to one execution, never to the node
//!
//! `OrdinalSplitExec` may be executed more than once, including
//! CONCURRENTLY (a caller reusing the same `Arc<dyn ExecutionPlan>` across
//! two overlapping `collect()` calls, or two genuinely interleaved runs —
//! neither forbidden by the `ExecutionPlan` trait). A GENERATION is the
//! whole shared pull state (the open input stream, the ordinal counter,
//! whether it has terminated) for exactly one such execution, shared only
//! among the partition indices that belong to it. The node holds a MAP —
//! `Mutex<HashMap<usize, Generation>>` — of every CURRENTLY-LIVE
//! generation, never a single slot: two genuinely concurrent/interleaved
//! runs must each keep their OWN generation alive at once, so a single
//! "current generation" slot would have a second run's call EVICT a
//! first, still-in-progress run's generation, making the first run's
//! LATER partition calls (still its own context) incorrectly start a
//! THIRD, fresh generation instead of rejoining their own run
//! (`two_interleaved_runs_each_see_every_row_exactly_once` pins this). A
//! `Generation` bundles the shared pull with a `claimed:
//! Vec<bool>` recording which partition indices this generation has
//! already handed a stream to; its run identity is the map's OWN key —
//! never duplicated onto the `Generation` itself.
//!
//! The map is keyed on that identity — a run's `Arc<TaskContext>` raw
//! pointer (`Arc::as_ptr(context) as usize`, stored as a bare `usize`;
//! the `Arc` itself is never kept alive by this node) — never on
//! partition-claim state alone: DataFusion hands EVERY partition of one
//! execution the SAME `Arc<TaskContext>` (`SortPreservingMergeExec` and
//! `CoalescePartitionsExec` — the parent every plan this node ships in
//! puts directly above it — each clone the ONE context they were given
//! into every child's `execute(p, Arc::clone(&context))` call; datafusion-
//! physical-plan 54.1's `sort_preserving_merge.rs`/`coalesce_partitions.rs`),
//! and each NEW logical execution (`execute_stream_partitioned`,
//! `collect`, a fresh `SessionContext::task_ctx()` call) constructs a
//! FRESH `Arc`. So `execute(p, context)` JOINS the LIVE generation for
//! `context`'s pointer if one exists — refusing typed, before a stream is
//! even constructed, if `p` was ALREADY claimed under it (a genuine
//! double-execute of one partition within one run) — and otherwise
//! STARTS a fresh generation bound to `context`, inserted into the map
//! alongside whatever other contexts' generations are ALSO currently live
//! (never evicting them).
//!
//! This makes a SECOND run's context provably never equal to a first,
//! still-in-flight run's: a partition of run 1 not yet claimed can NEVER
//! be mistaken for "still part of run 1" by a call that actually belongs
//! to an unrelated run 2, at any relative timing or interleaving of the
//! two runs' own partition calls — closing the interleaving gap a
//! claim-only rule would otherwise leave open, and (since the map never
//! evicts an unrelated context's entry) without breaking run 1's own
//! ability to resume its remaining partitions correctly AFTER run 2
//! interleaves. Two independent runs never interfere with each other's
//! shared pull, since each opens its OWN independent `input.execute(0,
//! ..)` stream: a single logical execution (every one of DataFusion's
//! `execute(0)..execute(n-1)` calls sharing ONE context, each partition
//! index asked for exactly once) divides one input's rows across its
//! partitions exactly once, regardless of how interleaved or sequential
//! the polling is; a genuinely SECOND execution (a different context)
//! always sees every row again, from a freshly reopened `input` stream
//! and a freshly zeroed ordinal counter.
//!
//! The map is pruned lazily, on every `execute()` call, of every entry
//! that can never be looked up again (every partition has claimed it AND
//! its shared pull has terminated — peeked with a non-blocking
//! `try_lock`, never forcing a wait on an actively-polling generation) —
//! this is what keeps it bounded by the number of genuinely concurrent,
//! not-yet-finished runs rather than growing with the node's own
//! lifetime call count.
//!
//! **The one residual, never closed structurally**: a caller that reuses
//! the IDENTICAL `TaskContext` `Arc` across TWO full executions of the
//! same plan, asking for a partition index NEVER claimed in the first
//! execution, JOINS the first execution's (by then possibly exhausted)
//! generation and sees zero rows rather than starting a genuinely fresh
//! one — there is no available signal distinguishing "still finishing run
//! 1, this partition simply has not been asked yet" from "a caller
//! deliberately reusing run 1's own context for an unrelated run 2." No
//! DataFusion entry point does this (`execute_stream_partitioned`,
//! `collect`, and this crate's own four call sites each construct, or
//! freshly clone from a freshly constructed `task_ctx()`, the context for
//! every actual execution).
//!
//! Each partition's OWN received ordinals are still strictly increasing
//! (a partition never polls itself concurrently, so its own successive
//! pulls happen in program order under the mutex), which is what backs
//! `InferenceExec`'s `[_ordinal ASC]` publication — though a partition's
//! own ordinals are not contiguous in general (they are
//! whichever slice of the global sequence it happened to pull), only
//! GLOBALLY contiguous, WITHIN ONE GENERATION, across the union of every
//! partition polled.
//!
//! An upstream error terminates the generation exactly like end-of-data —
//! `terminated = true`, no error is stored on `SharedPull` — and the ONE
//! partition whose poll observed it returns that OWNED `DataFusionError`
//! UNCHANGED (never re-wrapped, never stringified): a typed refusal raised
//! below this node (`KeyCheckExec`'s `InvalidKey`, a pool's
//! `ResourcesExhausted`) reaches the caller as that same variant at every
//! `N`, exactly as it would with no split at all — `jammi_db::error`'s
//! `JammiError: From<DataFusionError>` structural classifier destructures
//! an `External(Box<JammiError>)` payload BY VALUE, which only works if
//! the value crossing this node is
//! the original, not a `String` round-trip through `DataFusionError::
//! Internal(e.to_string())` — that would re-class every typed error below
//! the split into an untyped, "please file a bug report"-flavoured
//! `Internal` string, a silent wrong shape across gRPC/wire/Python, since
//! typed-vs-`Internal` is exactly what the wire and the Python bindings
//! branch on. Every
//! OTHER partition of the same generation simply ends cleanly (`None`,
//! like normal end-of-data) once `terminated` is set — never a second
//! copy of the error. This
//! composes correctly with every plan this split ships in: every root
//! puts a `SortPreservingMergeExec` (or, at `n<=1`,
//! `CoalescePartitionsExec`) directly above it, and either one propagates
//! the single `Err` from whichever partition stream produced it, so the
//! caller sees the ONE typed error exactly once.
//!
//! `_ordinal` is a NAMED INPUT COLUMN to `InferenceExec` from this node on —
//! never a `passthrough` (`InferenceExecBuilder::passthrough`'s end-of-schema
//! placement), and never generated by `InferenceRunner` when this node is the
//! child (`InferenceRunner` still self-generates a per-run sequence as a
//! documented fallback for a caller that builds `InferenceExec` directly on a
//! plan with no `_ordinal` column, e.g. unit tests and the `n == 1`
//! production shape below, which never inserts this node at all — see
//! `inference::schema::extract_or_generate_ordinals`).
//!
//! `InferenceExec` (this crate's `operator::inference_exec`) declares
//! `required_input_distribution = [UnspecifiedDistribution]`,
//! `benefits_from_input_partitioning = [false]`, `maintains_input_order =
//! [true]`, publishes `[_ordinal ASC]` on its OUTPUT equivalence properties,
//! and propagates the child's partition count as its own
//! `output_partitioning`. That declaration set keeps `EnforceDistribution`
//! from inserting a `RepartitionExec`/`SortExec`/`CoalescePartitionsExec`
//! BETWEEN this split and `InferenceExec` on the optimized plan — verified
//! non-vacuously (every one of the grid's cells contains both an
//! `InferenceExec` and an `OrdinalSplitExec` line, so the structural check
//! cannot pass by having nothing to check) on a 120-cell grid: `{PIPE,
//! UDTF} x {bare, GROUP BY, ORDER BY, WHERE, LIMIT} x N in {1,2,4} x
//! target_partitions in {1,2,4,8}` (`tests/it/rangesplit.rs`; dropping the
//! `benefits_from_input_partitioning` override fails exactly HALF of those
//! cells, 60 of 120 — the exact count shifts with which optimizer passes
//! fire). It inserts one
//! `CoalescePartitionsExec` BELOW this split instead, satisfying
//! `SinglePartition` when the split's own input plan has more than one
//! partition (the UDTF/annotate path's scan).
//!
//! A merge keyed `[_row_id, _ordinal]` DIVERGES from the 1-partition row
//! sequence on the UDTF path's unsorted input, and the optimizer deletes a
//! user's `ORDER BY _row_id` SortExec on the strength of that FALSE
//! published ordering, deadlocking the collect; `SanityCheckPlan` rejects an SPM
//! keyed `[_row_id, _ordinal]` whose child (`InferenceExec`) only publishes
//! `[_ordinal]`. A merge (and `InferenceExec`'s own published ordering)
//! keyed on `[_ordinal]` ALONE does not have this failure mode: it is
//! identical, PER ROW, over every column but `_latency_ms`, to the
//! 1-partition sequence at every `N` on both a pre-sorted and an unsorted
//! input shape (`tests/it/rangesplit.rs`'s per-row oracle), and the
//! optimizer keeps a real `SortExec` for a user `ORDER BY` rather than
//! trusting a stale claim.
//!
//! # Residency
//!
//! This node buffers NOTHING internally: a batch is handed directly from
//! the shared pull to whichever partition's poll requested it, with no
//! queue in between. This is a STRUCTURAL claim, provable by reading
//! `SharedPull`'s own field list below in full — an
//! `Option<SendableRecordBatchStream>` (the open input stream, owned by
//! `input`/DataFusion, not a buffer this node fills), a `u64` ordinal
//! counter, and a `bool` termination flag; `Generation`'s own `claimed:
//! Vec<bool>` is per-partition CLAIM bookkeeping (which partition indices
//! have been handed a stream this generation), never a batch or a
//! reference to one. None of this is a queue or a copy of a `RecordBatch`,
//! by inspection of the types alone — there is deliberately NO runtime
//! residency instrument (no counter, no gauge) anywhere in this node. A
//! per-partition "holds a batch" flag whose popcount is asserted `<= n` is
//! true of a length-`n` `Vec<bool>` unconditionally, so it measures nothing
//! about the SYSTEM; the residency bound stated above is a structural
//! argument over the field list, and must not be stated as a "measured"
//! claim without a real, falsifiable instrument backing it.
//!
//! # The wire
//!
//! `OrdinalSplitExec` has no `NodeTag`, no `plan.proto` message, and no
//! `JammiCodec` encode/decode arm — a plan containing it surfaces the SAME
//! typed "Unsupported plan node" refusal a `MaskExec` node gets when handed
//! to `jammi_ballista::codec::JammiCodec::try_encode` (see that crate's
//! `codec` module doc for WHY: in Ballista 54.1 a `SortPreservingMergeExec`
//! is a STAGE BOUNDARY, so the N-partition `InferenceExec(OrdinalSplitExec)`
//! this node feeds would become its own stage of N TASKS, each executing
//! exactly ONE partition — in a SEPARATE process per task in general,
//! which cannot share this node's in-process `tokio::sync::Mutex`-guarded
//! pull at all). `InferenceConfig::partitions` defaults to `1`, which never
//! inserts this node at all, so a wire submission (the distributed oracle's
//! `build_embedding_plan` call, `tests/distributed/main.rs`) is unaffected.
//! Configuring `partitions > 1` is an in-process-only capability; a wire
//! form would need a per-task-local split reconstructed from a contiguous
//! key RANGE rather than a shared in-process stream.

use std::collections::HashMap;
use std::fmt::{self, Formatter};
use std::sync::{Arc, Mutex};

use arrow::array::{ArrayRef, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, ExecutionPlanProperties,
    Partitioning, PlanProperties,
};
use futures::StreamExt;

/// The name of the column this node appends: the globally-assigned,
/// contiguous, 0-based row-emission ordinal (see the module doc).
pub const ORDINAL_COLUMN: &str = "_ordinal";

/// The N-way ordinal-keyed fan-out. See the module doc.
pub struct OrdinalSplitExec {
    input: Arc<dyn ExecutionPlan>,
    n: usize,
    properties: Arc<PlanProperties>,
    /// Every CURRENTLY-LIVE generation, keyed on its run's `Arc<TaskContext>`
    /// pointer (see the module doc's "Generations" section) — a MAP, never
    /// a single slot, because two genuinely concurrent/interleaved runs
    /// (different contexts) must each keep their OWN generation alive
    /// simultaneously; a single-slot design would have run 2's call evict
    /// run 1's still-in-progress generation, making run 1's LATER
    /// partition calls (still its own context) incorrectly start a THIRD,
    /// fresh generation instead of rejoining their own run. Pruned lazily
    /// (see `prune_exhausted_generations`) so a run that completes and
    /// drops its own context does not grow this map forever — the bounded
    /// term is "generations not yet fully claimed AND drained", which
    /// tracks the number of genuinely concurrent in-flight runs, not an
    /// unbounded history.
    generations: Mutex<HashMap<usize, Generation>>,
}

/// One execution's worth of shared state, plus which partition indices this
/// execution has already handed a stream to. Its run identity — the
/// `Arc<TaskContext>` pointer every partition of the run that started this
/// generation was called with — is the `OrdinalSplitExec::generations`
/// map's OWN key, never duplicated onto this struct; this node never keeps
/// a `TaskContext` alive past the call that handed it one.
struct Generation {
    shared: Arc<tokio::sync::Mutex<SharedPull>>,
    /// `claimed[p]` is `true` once `execute(p, ..)` has been called for
    /// THIS generation (see the module doc's identity rule for what makes
    /// a call "this generation" versus a new one).
    claimed: Vec<bool>,
}

/// The shared pull state for ONE generation, behind one mutex every
/// partition's poll (of that generation) takes in turn. No partition ever
/// buffers a batch it has not yet returned to its own caller; this struct
/// is the ENTIRE state one generation carries between polls (see the
/// module doc's Residency section for why this is a structural, not
/// instrumented, no-internal-buffering claim).
struct SharedPull {
    /// `None` until the first poll of this generation (any partition);
    /// `input.execute(0, ..)` is called exactly once per generation,
    /// against whichever `TaskContext` the FIRST polling partition's
    /// `execute()` call supplied.
    stream: Option<SendableRecordBatchStream>,
    /// The next value the global ordinal sequence will assign, within this
    /// generation.
    next_ordinal: u64,
    /// Set once the shared stream has yielded `None` or an `Err` — every
    /// partition's poll from then on returns `None` (a clean end, exactly
    /// like normal end-of-data) without touching the underlying stream
    /// again. On an `Err`, the ONE partition whose poll observed it returns
    /// that OWNED error immediately (before this flag is even read again)
    /// and nothing about the error is stored here — see the module doc's
    /// "An upstream error terminates..." paragraph for why no stored/
    /// re-wrapped copy exists.
    terminated: bool,
}

impl fmt::Debug for OrdinalSplitExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("OrdinalSplitExec")
            .field("n", &self.n)
            .finish_non_exhaustive()
    }
}

/// `input`'s schema with `_ordinal` (`UInt64`, non-nullable) appended.
/// `input` must not already carry an `_ordinal` column — a typed refusal
/// naming the collision, never a silent overwrite.
fn out_schema(input: &SchemaRef) -> DfResult<SchemaRef> {
    if input.field_with_name(ORDINAL_COLUMN).is_ok() {
        return Err(DataFusionError::Plan(format!(
            "OrdinalSplitExec: input schema already has an '{ORDINAL_COLUMN}' column"
        )));
    }
    let mut fields: Vec<Field> = input.fields().iter().map(|f| f.as_ref().clone()).collect();
    fields.push(Field::new(ORDINAL_COLUMN, DataType::UInt64, false));
    Ok(Arc::new(Schema::new(fields)))
}

fn with_ordinal(schema: &SchemaRef, batch: &RecordBatch, start: u64) -> DfResult<RecordBatch> {
    let end = start
        .checked_add(batch.num_rows() as u64)
        .ok_or_else(|| DataFusionError::Internal("OrdinalSplitExec: ordinal overflow".into()))?;
    let ordinals: UInt64Array = (start..end).collect();
    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
    cols.push(Arc::new(ordinals));
    RecordBatch::try_new(Arc::clone(schema), cols).map_err(DataFusionError::from)
}

fn new_generation(n: usize) -> Generation {
    Generation {
        shared: Arc::new(tokio::sync::Mutex::new(SharedPull {
            stream: None,
            next_ordinal: 0,
            terminated: false,
        })),
        claimed: vec![false; n],
    }
}

impl OrdinalSplitExec {
    /// Fan `input` out to `n` ordinal-keyed partitions. `n` is floored at 1.
    ///
    /// `required_input_distribution` declares `SinglePartition`, but every
    /// one of this crate's four call sites builds THIS node's parent tree BY
    /// HAND — never re-run through DataFusion's `EnforceDistribution`, which
    /// is what actually inserts a coalesce for a merely-DECLARED requirement
    /// in a SQL-planned query (see `operator::ordered_input`'s own doc for
    /// the identical concern over `SortExec`'s declared requirement). So
    /// `new` coalesces `input` itself, unconditionally, whenever it is not
    /// already a single partition — making the guarantee hold regardless of
    /// whether the caller happens to route this plan back through the
    /// optimizer (the SQL-planned `annotate` UDTF path does; `query::
    /// QueryBuilder::annotate`'s hand-assembled Rust chain does not).
    pub fn new(input: Arc<dyn ExecutionPlan>, n: usize) -> DfResult<Self> {
        let n = n.max(1);
        let input: Arc<dyn ExecutionPlan> = if input.output_partitioning().partition_count() > 1 {
            Arc::new(CoalescePartitionsExec::new(input))
        } else {
            input
        };
        let schema = out_schema(&input.schema())?;
        let eq = EquivalenceProperties::new(schema);
        let properties = PlanProperties::new(
            eq,
            Partitioning::UnknownPartitioning(n),
            input.pipeline_behavior(),
            input.boundedness(),
        );
        Ok(Self {
            input,
            n,
            properties: Arc::new(properties),
            generations: Mutex::new(HashMap::new()),
        })
    }

    /// The number of output partitions this split fans out to.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The shared pull for partition `p`'s NEXT `execute()` call, keyed on
    /// `context`'s IDENTITY (see the module doc): joins the LIVE generation
    /// for `context`'s pointer if one exists — refusing typed if `p` was
    /// ALREADY claimed under it (a genuine double-execute of one partition
    /// within one run) — else starts a fresh generation bound to `context`
    /// and claims `p` there. Every OTHER context's own generation, if any
    /// is concurrently live, is untouched either way.
    fn shared_pull_for(
        &self,
        partition: usize,
        context: &Arc<TaskContext>,
    ) -> DfResult<Arc<tokio::sync::Mutex<SharedPull>>> {
        let context_ptr = Arc::as_ptr(context) as usize;
        let mut gens = self
            .generations
            .lock()
            .expect("OrdinalSplitExec generations mutex poisoned");
        Self::prune_exhausted_generations(&mut gens);
        if let Some(g) = gens.get(&context_ptr) {
            if g.claimed[partition] {
                return Err(DataFusionError::Internal(format!(
                    "OrdinalSplitExec: partition {partition} executed twice within the same \
                     run (same TaskContext) — a caller may re-execute a partition to start a \
                     NEW run (a genuinely different TaskContext), but never re-poll the same \
                     (context, partition) pair of an in-progress one"
                )));
            }
        } else {
            gens.insert(context_ptr, new_generation(self.n));
        }
        let gen = gens
            .get_mut(&context_ptr)
            .expect("just inserted or confirmed present above");
        gen.claimed[partition] = true;
        Ok(Arc::clone(&gen.shared))
    }

    /// Drops every generation this map can never be asked to look up
    /// again: every partition has claimed it (`claimed.iter().all`) AND
    /// its shared pull has terminated (checked with `try_lock` — a
    /// generation currently being actively polled is simply left for a
    /// later call to prune; never blocked on). This is what keeps the map
    /// bounded by "genuinely concurrent, not-yet-finished runs" rather
    /// than growing with the node's own lifetime call count.
    fn prune_exhausted_generations(gens: &mut HashMap<usize, Generation>) {
        gens.retain(|_, g| {
            let fully_claimed = g.claimed.iter().all(|&c| c);
            if !fully_claimed {
                return true;
            }
            match g.shared.try_lock() {
                Ok(pull) => !pull.terminated,
                Err(_) => true,
            }
        });
    }
}

impl DisplayAs for OrdinalSplitExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        write!(f, "OrdinalSplitExec: n={}", self.n)
    }
}

impl ExecutionPlan for OrdinalSplitExec {
    fn name(&self) -> &str {
        "OrdinalSplitExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::SinglePartition]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(OrdinalSplitExec::new(
            Arc::clone(&children[0]),
            self.n,
        )?))
    }

    /// Demand-driven: no task is ever spawned, no channel is ever created.
    /// Every call joins or starts a generation (see the module doc) and
    /// returns immediately with a stream that, on each poll, locks that
    /// generation's shared pull, advances it by exactly one batch (opening
    /// `input`'s stream on the very first poll of that generation), stamps
    /// the batch with the next slice of the generation's ordinal sequence,
    /// and releases the lock before handing the batch to its caller. See
    /// the module doc for why this makes every wedge shape a fixed-capacity
    /// channel design is vulnerable to (a receiver never taken; a slow
    /// sequential drain; more batches than any fixed channel capacity)
    /// unrepresentable rather than merely handled.
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        if partition >= self.n {
            return Err(DataFusionError::Internal(format!(
                "OrdinalSplitExec: partition {partition} out of range for n={}",
                self.n
            )));
        }
        let schema = self.schema();
        let shared = self.shared_pull_for(partition, &context)?;
        let input = Arc::clone(&self.input);
        let out_schema = Arc::clone(&schema);
        let state = (shared, input, context, out_schema, partition);
        let stream = futures::stream::unfold(
            state,
            move |(shared, input, context, out_schema, partition)| async move {
                let mut guard = shared.lock().await;
                if guard.terminated {
                    return None;
                }
                if guard.stream.is_none() {
                    match input.execute(0, Arc::clone(&context)) {
                        Ok(s) => guard.stream = Some(s),
                        Err(e) => {
                            // Terminate the generation exactly like clean
                            // end-of-data — no error is stored — and return
                            // this OWNED error UNCHANGED: a typed refusal
                            // raised opening `input` must reach the caller
                            // as that same variant, never re-wrapped or
                            // stringified (see the module doc).
                            guard.terminated = true;
                            drop(guard);
                            return Some((Err(e), (shared, input, context, out_schema, partition)));
                        }
                    }
                }
                let next = guard.stream.as_mut().expect("just set above").next().await;
                match next {
                    None => {
                        guard.terminated = true;
                        None
                    }
                    Some(Err(e)) => {
                        // Same pass-through as above: this partition's poll
                        // is the ONE that observed the failure, and returns
                        // the OWNED `DataFusionError` unchanged. Every other
                        // partition of this generation simply sees
                        // `terminated` on its own next poll and ends
                        // cleanly (the `None` arm above) — never a second
                        // copy of this error.
                        guard.terminated = true;
                        drop(guard);
                        Some((Err(e), (shared, input, context, out_schema, partition)))
                    }
                    Some(Ok(batch)) => {
                        let start = guard.next_ordinal;
                        let rows = batch.num_rows() as u64;
                        let stamped = with_ordinal(&out_schema, &batch, start);
                        guard.next_ordinal += rows;
                        drop(guard);
                        Some((stamped, (shared, input, context, out_schema, partition)))
                    }
                }
            },
        );
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use datafusion::catalog::memory::MemorySourceConfig;
    use datafusion::physical_plan::ExecutionPlanProperties;
    use datafusion::prelude::SessionContext;

    fn in_schema() -> SchemaRef {
        Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            ArrowDataType::Int64,
            false,
        )]))
    }

    fn fixture(nb: usize, rows: usize) -> Arc<dyn ExecutionPlan> {
        let mut batches = Vec::new();
        for b in 0..nb {
            let start = (b * rows) as i64;
            let ids: Vec<i64> = (start..start + rows as i64).collect();
            batches.push(
                RecordBatch::try_new(in_schema(), vec![Arc::new(Int64Array::from(ids))]).unwrap(),
            );
        }
        MemorySourceConfig::try_new_exec(&[batches], in_schema(), None).unwrap()
    }

    /// A collision with a pre-existing `_ordinal` column is a typed refusal,
    /// never a silent overwrite of the caller's data.
    #[test]
    fn refuses_an_input_that_already_has_ordinal() {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            ORDINAL_COLUMN,
            ArrowDataType::UInt64,
            false,
        )]));
        let src = MemorySourceConfig::try_new_exec(&[vec![]], schema, None).unwrap();
        let err = OrdinalSplitExec::new(src, 2).unwrap_err();
        assert!(err.to_string().contains(ORDINAL_COLUMN));
    }

    /// The fan-out partition count is floored at 1, never 0 (a 0-partition
    /// plan can never be executed).
    #[tokio::test]
    async fn n_is_floored_at_one() {
        let split = OrdinalSplitExec::new(fixture(1, 3), 0).unwrap();
        assert_eq!(split.n(), 1);
        let split: &dyn ExecutionPlan = &split;
        assert_eq!(split.output_partitioning().partition_count(), 1);
    }

    /// Every row across every partition carries a distinct `_ordinal`, and
    /// the union of all partitions' ordinals is the contiguous `0..total_rows`
    /// set (never a gap, never a repeat) — the split's own row-conservation
    /// invariant, independent of the merge above it.
    #[tokio::test]
    async fn ordinals_are_a_contiguous_partition_of_every_row() {
        let n = 4;
        let split = Arc::new(OrdinalSplitExec::new(fixture(6, 5), n).unwrap());
        let ctx = SessionContext::new();
        // ONE context for the whole run, cloned per partition — the real
        // shape every DataFusion entry point uses (see the module doc's
        // Generations section): a generation is keyed on this Arc's
        // identity, so every partition of ONE run must share it.
        let task_ctx = ctx.task_ctx();
        let mut all_ordinals: Vec<u64> = Vec::new();
        for p in 0..n {
            let stream = split.execute(p, Arc::clone(&task_ctx)).unwrap();
            let batches = datafusion::physical_plan::common::collect(stream)
                .await
                .unwrap();
            for b in &batches {
                let col = b
                    .column_by_name(ORDINAL_COLUMN)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap();
                all_ordinals.extend(col.values().iter().copied());
            }
        }
        all_ordinals.sort_unstable();
        let expected: Vec<u64> = (0..30u64).collect();
        assert_eq!(all_ordinals, expected);
    }

    /// A consumer that executes ONLY partition 0 of an n=4 split (the
    /// single-consumer shape) must see ALL of the source's
    /// rows, and must complete — never wedge (a spawned-writer-plus-
    /// bounded-channel design wedges here: partitions 1-3's receivers are
    /// never taken, so their channels fill and the single writer thread
    /// blocks trying to fill them, starving partition 0 too).
    #[tokio::test]
    async fn execute_partition_zero_alone_sees_every_row_and_completes() {
        let n = 4;
        let split = Arc::new(OrdinalSplitExec::new(fixture(24, 5), n).unwrap());
        let ctx = SessionContext::new();
        let stream = split.execute(0, ctx.task_ctx()).unwrap();
        let batches = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            datafusion::physical_plan::common::collect(stream),
        )
        .await
        .expect("execute(0) alone must not wedge")
        .unwrap();
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 24 * 5, "partition 0 alone must see every source row");
    }

    /// A SEQUENTIAL drain (partition 0 to completion, then partition 1,
    /// ...) must complete and conserve every row exactly once — a spawned-
    /// writer-plus-bounded-channel design wedges here too: while partition
    /// 0 drains, partitions 1-3's receivers sit untaken, their channels
    /// fill, and the writer blocks. Also the load-bearing generation-
    /// boundary case: each of the 4 partition indices is asked for exactly
    /// ONCE here (never reused), so all 4 belong to the SAME generation and
    /// must divide the 120 rows exactly once, never re-reading the input.
    #[tokio::test]
    async fn sequential_drain_completes_and_conserves_every_row() {
        let n = 4;
        let split = Arc::new(OrdinalSplitExec::new(fixture(24, 5), n).unwrap());
        let ctx = SessionContext::new();
        let task_ctx = ctx.task_ctx();
        let mut total = 0usize;
        for p in 0..n {
            let stream = split.execute(p, Arc::clone(&task_ctx)).unwrap();
            let batches = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                datafusion::physical_plan::common::collect(stream),
            )
            .await
            .unwrap_or_else(|_| panic!("sequential drain must not wedge at partition {p}"))
            .unwrap();
            total += batches.iter().map(|b| b.num_rows()).sum::<usize>();
        }
        assert_eq!(total, 24 * 5, "every row must be conserved exactly once");
    }

    /// A batch count with no relationship to any fixed channel capacity
    /// (there is none any more) must still complete — a bounded-channel
    /// design's own capacity is a magic number a fixture can accidentally
    /// stay under; this fixture is sized well past any capacity that design
    /// would plausibly use.
    #[tokio::test]
    async fn a_batch_count_much_larger_than_any_old_channel_capacity_completes() {
        let n = 4;
        let split = Arc::new(OrdinalSplitExec::new(fixture(37, 5), n).unwrap());
        let ctx = SessionContext::new();
        let task_ctx = ctx.task_ctx();
        let mut total = 0usize;
        for p in 0..n {
            let stream = split.execute(p, Arc::clone(&task_ctx)).unwrap();
            let batches = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                datafusion::physical_plan::common::collect(stream),
            )
            .await
            .unwrap_or_else(|_| panic!("must not wedge at partition {p}"))
            .unwrap();
            total += batches.iter().map(|b| b.num_rows()).sum::<usize>();
        }
        assert_eq!(total, 37 * 5);
    }

    /// The generation-boundary property, both arms: (a) executing the
    /// SAME partition index again under a DIFFERENT `TaskContext` (a
    /// genuinely new run) sees every row again (a fresh generation, not
    /// `Ok(0 rows)`); (b) a partition asked for LATE under that SAME
    /// second context, in a generation another partition already
    /// exhausted, sees zero rows for THAT generation (it was never owed
    /// anything the generation did not already hand out) — this is the
    /// module doc's named residual (reusing one context across two full
    /// executions), reproduced deliberately here since it is exactly the
    /// scenario this test needs to exercise arm (b) at all. Mutation:
    /// keeping the shared pull on the node instead of per-generation makes
    /// arm (a) return `Ok(0 rows)` for run2 (state already drained by run1
    /// is still sitting on the node) — reds this test.
    #[tokio::test]
    async fn re_executing_a_partition_starts_a_fresh_generation_that_sees_every_row_again() {
        let n = 4;
        let split = Arc::new(OrdinalSplitExec::new(fixture(6, 5), n).unwrap());
        let ctx = SessionContext::new();

        // Generation 1, its own context: partition 0 alone drains
        // everything. `_run1_ctx_kept_alive` holds a strong reference for
        // the REST of this test — `execute()` only receives a clone, since
        // otherwise run1's context would be fully dropped (last strong
        // ref) the moment `s1`/`b1` go out of scope, and a general-purpose
        // allocator is then free to hand run2's `Arc::new` the IDENTICAL
        // address, defeating the pointer-identity comparison this test
        // means to exercise — a real allocator behavior, not a theoretical
        // one (observed directly: this test flaked under a full `--lib`
        // run without this guard, tripping the double-execute refusal
        // instead of starting generation 2).
        let run1_ctx = ctx.task_ctx();
        let _run1_ctx_kept_alive = Arc::clone(&run1_ctx);
        let s1 = split.execute(0, run1_ctx).unwrap();
        let b1 = datafusion::physical_plan::common::collect(s1)
            .await
            .unwrap();
        let r1: usize = b1.iter().map(|b| b.num_rows()).sum();
        assert_eq!(r1, 30, "generation 1, partition 0 alone sees every row");

        // Generation 2, a genuinely DIFFERENT context (a real second run):
        // executing partition 0 AGAIN must start over and see every row
        // again, not 0.
        let run2_ctx = ctx.task_ctx();
        let s2 = split.execute(0, Arc::clone(&run2_ctx)).unwrap();
        let b2 = datafusion::physical_plan::common::collect(s2)
            .await
            .unwrap();
        let r2: usize = b2.iter().map(|b| b.num_rows()).sum();
        assert_eq!(
            r2, 30,
            "re-executing partition 0 under a fresh context must see every row again"
        );

        // Still generation 2 (SAME `run2_ctx`, partition 0 already drained
        // it, so it is now exhausted): partition 2 has never been claimed
        // under `run2_ctx`, so it JOINS generation 2 rather than starting a
        // third one — and sees zero rows, since generation 2 already
        // handed everything to partition 0. Reusing `run2_ctx` here is
        // deliberate — the module doc's named residual, not an accident.
        let s3 = split.execute(2, run2_ctx).unwrap();
        let b3 = datafusion::physical_plan::common::collect(s3)
            .await
            .unwrap();
        let r3: usize = b3.iter().map(|b| b.num_rows()).sum();
        assert_eq!(
            r3, 0,
            "a partition asked for late, under the SAME context as an already-exhausted \
             generation, sees zero rows"
        );
    }

    /// An upstream that raises a TYPED error (an `External(Box<
    /// JammiError>)` payload, or a native `DataFusionError::
    /// ResourcesExhausted`) after `n_ok` good batches. `err` is called
    /// fresh each time `execute()` runs so a second poll (a different
    /// partition, or a re-execution) never reuses a moved value.
    struct TypedErrAfter {
        n_ok: usize,
        schema: SchemaRef,
        props: Arc<PlanProperties>,
        err: fn() -> DataFusionError,
    }
    impl std::fmt::Debug for TypedErrAfter {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            f.debug_struct("TypedErrAfter").finish_non_exhaustive()
        }
    }
    impl TypedErrAfter {
        fn new(n_ok: usize, err: fn() -> DataFusionError) -> Self {
            let schema = in_schema();
            let props = PlanProperties::new(
                EquivalenceProperties::new(Arc::clone(&schema)),
                Partitioning::UnknownPartitioning(1),
                datafusion::physical_plan::execution_plan::EmissionType::Incremental,
                datafusion::physical_plan::execution_plan::Boundedness::Bounded,
            );
            Self {
                n_ok,
                schema,
                props: Arc::new(props),
                err,
            }
        }
    }
    impl DisplayAs for TypedErrAfter {
        fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
            write!(f, "TypedErrAfter")
        }
    }
    impl ExecutionPlan for TypedErrAfter {
        fn name(&self) -> &str {
            "TypedErrAfter"
        }
        fn properties(&self) -> &Arc<PlanProperties> {
            &self.props
        }
        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![]
        }
        fn with_new_children(
            self: Arc<Self>,
            _c: Vec<Arc<dyn ExecutionPlan>>,
        ) -> DfResult<Arc<dyn ExecutionPlan>> {
            Ok(self)
        }
        fn execute(&self, _p: usize, _c: Arc<TaskContext>) -> DfResult<SendableRecordBatchStream> {
            let schema = Arc::clone(&self.schema);
            let mut items: Vec<DfResult<RecordBatch>> = (0..self.n_ok)
                .map(|i| {
                    RecordBatch::try_new(
                        Arc::clone(&schema),
                        vec![Arc::new(Int64Array::from(vec![i as i64]))],
                    )
                    .map_err(DataFusionError::from)
                })
                .collect();
            items.push(Err((self.err)()));
            let s = futures::stream::iter(items);
            Ok(Box::pin(RecordBatchStreamAdapter::new(schema, s)))
        }
    }

    fn invalid_key_error() -> DataFusionError {
        DataFusionError::External(Box::new(jammi_db::error::JammiError::InvalidKey {
            column: "id".to_string(),
            null_count: 1,
        }))
    }

    fn resources_exhausted_error() -> DataFusionError {
        DataFusionError::ResourcesExhausted("pool limit reached".to_string())
    }

    /// The variant name only — never the message, since e.g.
    /// `ResourcesExhausted`'s `detail` embeds a pool-size string that need
    /// not match byte-for-byte across two independently executed streams.
    fn classify(e: DataFusionError) -> &'static str {
        match jammi_db::error::JammiError::from(e) {
            jammi_db::error::JammiError::InvalidKey { .. } => "InvalidKey",
            jammi_db::error::JammiError::ResourcesExhausted { .. } => "ResourcesExhausted",
            other => panic!("unexpected classification: {other:?}"),
        }
    }

    /// A typed refusal raised below the split reaches the caller classified
    /// IDENTICALLY at `n=4` and with no split at all (`n=1`, still a real
    /// `OrdinalSplitExec` — this crate never omits the node,
    /// `wrap_with_split_and_merge` does that at a layer above). A
    /// `to_string()`/`Internal` re-wrap in either error arm of `execute()`'s
    /// `unfold` body would fail this test, since
    /// `DataFusionError::Internal(String)` carries no
    /// `External(Box<JammiError>)` payload for the classifier to
    /// destructure, and is never recognised by `resources_exhausted_
    /// message` either — both err sides fall through to the generic
    /// `JammiError::DataFusion(Internal(..))` arm, which `classify` panics
    /// on ("unexpected classification").
    #[tokio::test]
    async fn a_typed_refusal_below_the_split_classifies_identically_at_every_n() {
        for err_fn in [
            invalid_key_error as fn() -> DataFusionError,
            resources_exhausted_error as fn() -> DataFusionError,
        ] {
            for n in [1usize, 4] {
                let split = Arc::new(
                    OrdinalSplitExec::new(Arc::new(TypedErrAfter::new(2, err_fn)), n).unwrap(),
                );
                let ctx = SessionContext::new();
                let task_ctx = ctx.task_ctx();
                // Poll every partition (ONE shared context — one run) until
                // one observes the error (the demand-driven split hands the
                // failing final item to whichever partition happens to be
                // polling when the shared stream reaches it).
                let mut observed: Option<DataFusionError> = None;
                for p in 0..n {
                    let stream = split.execute(p, Arc::clone(&task_ctx)).unwrap();
                    if let Err(e) = datafusion::physical_plan::common::collect(stream).await {
                        observed = Some(e);
                        break;
                    }
                }
                let observed = observed.unwrap_or_else(|| {
                    panic!("n={n}: no partition observed the upstream error at all")
                });
                let no_split_err = datafusion::physical_plan::common::collect(
                    TypedErrAfter::new(2, err_fn)
                        .execute(0, ctx.task_ctx())
                        .unwrap(),
                )
                .await
                .expect_err("the direct (no-split) execution must also fail");
                assert_eq!(
                    classify(observed),
                    classify(no_split_err),
                    "n={n}: the split must classify the SAME typed error identically to no split"
                );
            }
        }
    }

    /// The generation-termination property: a failing upstream terminates
    /// the generation for every partition (the ONE whose poll observed the
    /// failure returns it typed; every OTHER partition of the same
    /// generation ends cleanly, `None`, like normal end-of-data — never a
    /// second copy of the error). Every partition returning the SAME stored
    /// error is not distinguishable from this property by ROW OUTCOME alone
    /// — a partition ending cleanly and a partition never having been
    /// polled are both `Ok` arms here; the FAILURE classification itself is
    /// covered by
    /// `a_typed_refusal_below_the_split_classifies_identically_at_every_n`
    /// above.
    #[tokio::test]
    async fn the_partition_that_observes_an_upstream_error_returns_it_typed_and_the_rest_end_cleanly(
    ) {
        let n = 4;
        let split = Arc::new(
            OrdinalSplitExec::new(Arc::new(TypedErrAfter::new(2, invalid_key_error)), n).unwrap(),
        );
        let ctx = SessionContext::new();
        let task_ctx = ctx.task_ctx();
        let mut saw_error = false;
        for p in 0..n {
            let stream = split.execute(p, Arc::clone(&task_ctx)).unwrap();
            match datafusion::physical_plan::common::collect(stream).await {
                Ok(_) => {}
                Err(e) => {
                    assert_eq!(classify(e), "InvalidKey");
                    saw_error = true;
                }
            }
        }
        assert!(
            saw_error,
            "at least one partition of the generation must observe the upstream error"
        );
    }

    /// Executing the SAME (context, partition) pair twice — never a valid
    /// shape any DataFusion entry point produces — is a genuine
    /// double-execute bug and must be a typed refusal at `execute()` call
    /// time (before a stream is even constructed), never a silent second
    /// stream. Mutation: dropping the `claimed[partition]` check in the
    /// same-context branch of `shared_pull_for` (always falling through to
    /// `Ok`) reds this with "must panic: expected an Err" — never executed
    /// as a permanent change, since a silently-accepted double-execute
    /// would mask exactly the interleaving shape the sibling test below
    /// exercises.
    #[tokio::test]
    async fn executing_the_same_partition_twice_under_the_same_context_is_a_typed_refusal() {
        let n = 4;
        let split = Arc::new(OrdinalSplitExec::new(fixture(6, 5), n).unwrap());
        let ctx = SessionContext::new();
        let task_ctx = ctx.task_ctx();
        let s1 = split.execute(0, Arc::clone(&task_ctx)).unwrap();
        datafusion::physical_plan::common::collect(s1)
            .await
            .unwrap();
        let err = match split.execute(0, task_ctx) {
            Ok(_) => panic!(
                "executing partition 0 twice under the IDENTICAL context must be a typed refusal"
            ),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("executed twice"),
            "must name the shape: {err}"
        );
    }

    /// The interleaving shape context-identity keying is built to close:
    /// run 1 (context A) executes p0, p1; run 2
    /// (context B, genuinely different) executes p0 WHILE run 1 is still
    /// mid-flight (p2, p3 not yet claimed under A); run 1 then executes p2,
    /// p3 (still context A). Both runs must see every row of their OWN
    /// independent execution of `input` exactly once — run 2's context B
    /// call must never be mistaken for "still part of run 1" merely because
    /// p2/p3 have not been claimed under A yet. Mutation: keying generations
    /// on partition-claim state ALONE (never comparing context) makes run
    /// 2's p0 call (context B, partition 0 NOT YET claimed under the
    /// current generation, since only p0/p1 of run 1 have polled) instead
    /// JOIN run 1's in-progress generation as if it were run 1's own p0 —
    /// but p0 was ALREADY claimed by run 1's own earlier call, so under the
    /// claim-only rule this reproduces the exact double-execute shape this
    /// unit's OWN typed refusal above would then incorrectly reject a
    /// legitimate second run for.
    #[tokio::test]
    async fn two_interleaved_runs_each_see_every_row_exactly_once() {
        let n = 4;
        let split = Arc::new(OrdinalSplitExec::new(fixture(24, 5), n).unwrap());
        let ctx = SessionContext::new();
        let run1_ctx = ctx.task_ctx();
        let run2_ctx = ctx.task_ctx();
        assert_ne!(
            Arc::as_ptr(&run1_ctx) as usize,
            Arc::as_ptr(&run2_ctx) as usize,
            "the probe requires two genuinely distinct TaskContext instances"
        );

        let mut run1_rows = 0usize;
        for p in [0usize, 1] {
            let stream = split.execute(p, Arc::clone(&run1_ctx)).unwrap();
            let batches = datafusion::physical_plan::common::collect(stream)
                .await
                .unwrap();
            run1_rows += batches.iter().map(|b| b.num_rows()).sum::<usize>();
        }

        let stream = split.execute(0, run2_ctx).unwrap();
        let batches = datafusion::physical_plan::common::collect(stream)
            .await
            .unwrap();
        // Run 2 only ever executes partition 0, which — as the sole
        // partition of its own independent generation — sees every row of
        // its OWN `input.execute(0, ..)` stream alone.
        let run2_rows: usize = batches.iter().map(|b| b.num_rows()).sum();

        for p in [2usize, 3] {
            let stream = split.execute(p, Arc::clone(&run1_ctx)).unwrap();
            let batches = datafusion::physical_plan::common::collect(stream)
                .await
                .unwrap();
            run1_rows += batches.iter().map(|b| b.num_rows()).sum::<usize>();
        }

        assert_eq!(
            run1_rows,
            24 * 5,
            "run 1 (context A, partitions 0,1 then 2,3) must see every row exactly once"
        );
        assert_eq!(
            run2_rows,
            24 * 5,
            "run 2 (context B) must independently see every row exactly once, unaffected by \
             run 1's interleaved partition calls"
        );
    }
}
