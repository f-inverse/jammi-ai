//! The job worker: claims durable jobs — training AND compute — under a
//! lease, reconstructs each from its persisted spec, executes it while
//! renewing the lease, and records the terminal outcome.
//!
//! One worker drives every job kind ([`COMPILED_KINDS`]). A
//! [`JobWorker::run_until`] tick — run only under
//! [`EmbeddedWorker::spawn`]/[`EmbeddedWorker::spawn_worker`]'s claimed
//! session slot (one claim loop per session is structural, not merely
//! conventional) — first reclaims expired leases
//! (re-queuing a dead worker's job, or failing it past the attempts cap),
//! then atomically
//! claims the oldest queued job of one of its configured kinds
//! (`execution = 'queued'` only — an `inline` row is never selected by the
//! poll loop). On a claim it deserialises the spec and dispatches: a
//! training kind (`fine_tune`/`graph_fine_tune`/`context_predictor`)
//! re-scopes the catalog to the job's tenant and runs a *from-scratch*
//! reconstruction — re-running the source SQL, re-reading and re-sampling
//! the graph (seeded, deterministic), or re-sampling the episodic
//! meta-dataset; a compute kind (`neighbor_graph`/`propagate`/`asof_join`)
//! dispatches through [`crate::jobs::execute_compute`]. No in-memory state
//! crosses the submit→claim boundary, so a worker can run a job submitted by
//! a now-gone session on a fresh process.
//!
//! The worker holds a [`Weak`] reference to the [`InferenceSession`]: the
//! predictor reconstruction needs an `Arc<InferenceSession>` (its sampler methods
//! take `self: &Arc<Self>`), but a strong handle would form a refcycle with the
//! session that owns the worker. Upgrading the `Weak` each tick is also the
//! worker's stop signal — when the session drops, `upgrade()` returns `None` and
//! the loop exits.
//!
//! ## Cooperative cancellation
//!
//! A `spawn_blocking` training thread cannot be force-aborted, so cancellation
//! is cooperative: the job's lease is a hold with the session's
//! [`jammi_db::catalog::lease_keeper::LeaseKeeper`] — a dedicated OS
//! thread renews it, immune to this runtime being starved by the training
//! itself — and the hold's own `lost` flag (via
//! [`jammi_db::catalog::lease_keeper::LeaseHold::lost_flag`]) is the shared
//! cancel flag the training loop checks at every epoch boundary. That
//! sentence is scoped to LEASE RENEWAL specifically: renewal itself has no
//! separate `tokio::spawn` heartbeat task anywhere in this crate (the
//! keeper's dedicated OS thread is the sole renewer). `spawn_cancel_request_watcher`
//! below IS a `tokio::spawn`'d task at that same heartbeat cadence — its
//! starvation (an unlikely, but not impossible, saturated runtime) only
//! delays *observing* a cancel request, never lease renewal, which the
//! dedicated OS thread keeps doing regardless.
//!
//! That flag has TWO writers, not one: the lease keeper flips it
//! directly on a missed renewal (a genuine lease loss), and
//! `spawn_cancel_request_watcher` flips the SAME flag, at the SAME
//! heartbeat cadence, when it observes `jobs.cancel_requested` set on this
//! job's row (an operator's `CancelJob`/`JobHandle::cancel`). Both writers
//! only ever store `true`, so there is no race to arbitrate — but they mean
//! different outcomes once the training loop bails, so
//! [`JobWorker::run_claimed_job`] tells them apart with a SECOND, one-way
//! flag the watcher alone sets right before it flips the shared one: when
//! that second flag is set, the cancellation was requested, not a lease
//! loss, and the job is recorded `failed` with
//! [`jammi_db::error::JammiError::JobCancelled`]'s message (the SAME message
//! the compute path and `InferenceSession::run_now` already record for a
//! request observed at their own checkpoints); when it is unset, the flag
//! tripped on a lease loss, and the loop bails leaving the job `running` for
//! the next `reclaim_expired_jobs` to re-queue.
//!
//! Cancellation is checked only at epoch boundaries, so a worker can still lose
//! its lease in the window between the last check and finalization. The terminal
//! write is therefore a compare-and-set: [`Catalog::finish_job_with_model`]
//! writes the output model + flips the job to `completed` only while
//! `claimed_by` is still this worker and the status is still `running`. A worker
//! that lost its lease matches zero rows and does not finalize, so two workers
//! never both finalize the same job — the re-claiming worker is the sole
//! finalizer.
//!
//! **A cancel observed after the last epoch boundary lands `completed`.**
//! [`JobWorker::run_claimed_job`]'s `Ok(artifact)` arm never consults
//! `cancel_requested_seen` — once the training loop has returned an
//! artifact, `spawn_cancel_request_watcher` may since have flipped the
//! shared flag (a request landed after the final epoch's check, in the
//! window before that watcher was aborted), but the run already has a
//! finished result and nothing left to check it against. This is the same
//! convention [`crate::jobs`]'s compute path documents for its own
//! single-shot producers: a request that lands after the producer has
//! started is honoured only in the sense that it stays recorded on the
//! row (`jobs.cancel_requested` remains `true`) — the run completes and the
//! row finishes `completed`, never retroactively `failed`.
//!
//! ## Runner roles and the job-row writers (the single-writer rule as types)
//!
//! The lease holder is the ONE writer of a job's row, of its
//! durable checkpoints and of its published artifact; every other rank of a
//! gang writes nothing durable. [`crate::fine_tune::role`] states it as two
//! types — a [`LeaseHolder`] (`LoopClaimer`, the in-process path incl. a
//! `Local` gang's rank 0; `Coordinator`, rank 0 of a `Peer` gang) and a
//! [`RunnerRole`] (`Holder(LeaseHolder)` or `Rank { rank }`) — and EVERY
//! job-row-writing site on the run path takes a `LeaseHolder` as a REQUIRED
//! parameter, so a missed site is a compile error and a `Rank` body, which
//! holds no `LeaseHolder`, has nothing to pass: the write is unreachable by
//! type. The holder of one attempt is derived ONCE, from the claimed spec
//! and this host's `[worker] local_ranks` ([`lease_holder_for`]: the
//! `Coordinator` exactly when a column-source `fine_tune` decides
//! `TopologyDecision::Peer`, the `LoopClaimer` otherwise — `W == 1` is
//! always the loop claimer and never traverses the coordinator body),
//! and threaded to every site. The sites, derived from
//! `grep -n 'record_failed(\|finish_job_with_model(\|persist_acceleration_report(\|register_job_hold_or_release(' worker.rs`
//! minus doc lines, each with the holder role(s) that can reach it:
//!
//! | # | site (function, arm) | holder(s) |
//! |---|---|---|
//! | 1 | `register_job_hold_or_release` — the lease-hold registration, the `Releasing` self-release arm, the holder accounting (`HostAdmission::job_running`) | training arm: `LoopClaimer`, `Coordinator`; compute arm: `LoopClaimer` |
//! | 2 | `run_claimed_job_under` — undeserialisable training spec (`mark_acceleration_undetermined` then `record_failed`) | `LoopClaimer` (no spec, no topology) |
//! | 3 | `run_claimed_job_under` — `Cancelled` with a cancel request observed (`record_failed`) | `LoopClaimer`, `Coordinator` |
//! | 4 | `run_claimed_job_under` — `Failed` (`record_failed`) | `LoopClaimer`, `Coordinator` |
//! | 5 | `fail_before_finalize` — `publish_and_finalize` giving up before its finalize: the final bundle could not be staged, its materialization attestation could not be written or summarised, or the job result could not be serialised (`record_failed`) | `LoopClaimer`, `Coordinator` |
//! | 6 | `publish_and_finalize` — the finalize (`finish_job_with_model`) | `LoopClaimer`, `Coordinator` |
//! | 7 | `run_claimed_compute_job` — undeserialisable compute spec (`record_failed`) | `LoopClaimer` |
//! | 8 | `run_claimed_compute_job` — a cancel observed at the post-claim checkpoint (`record_failed`) | `LoopClaimer` |
//! | 9 | `run_claimed_compute_job` — partial-result serialisation failure (`record_failed`) | `LoopClaimer` |
//! | 10 | `run_claimed_compute_job` — result serialisation failure (`record_failed`) | `LoopClaimer` |
//! | 11 | `run_claimed_compute_job` — `execute_compute` failure (`record_failed`) | `LoopClaimer` |
//! | 12 | the acceleration report: `compute_and_persist_acceleration_report` (a `Rank` computes and discards) → `persist_acceleration_report`; `mark_acceleration_not_applicable`; `mark_acceleration_undetermined` | `LoopClaimer`, `Coordinator` |
//! | 13 | `JobWorker::coordinate` — `record_assembly_outcome`, `release_job_lease` | `Coordinator` |
//! | 14 | placed hand-off: the SUBMITTER, after `WorkerJobError::HandedOff` | writes NOTHING — the row and its lease keeper registration are the placed executor's now |
//! | 15 | placed hand-off: the EXECUTOR, running [`JobWorker::run_placed_gang`] | writes as `Coordinator` (`run_claimed_job_under(.., placed = true)` is the SAME body as row 13 and every row above it) |
//!
//! `finish_job` (the compute arm's CAS) is reachable only from
//! `run_claimed_compute_job`, a `LoopClaimer` by construction. The trainer's
//! own durable writes (the resume and epoch checkpoints) are gated on the
//! same role inside `TrainingLoop` (`TrainingLoopBuilder::runner_role`),
//! never inside the store. A `Peer` member's body ([`run_member_rank`]) runs
//! as `RunnerRole::Rank` and ends its session with `RankEvent::Outcome`;
//! the coordinator publishes only on receipt of every member's `Trained`
//! outcome carrying its own artifact digest (`JobWorker::assemble_and_run`).

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

use arrow::array::{ArrayRef, RecordBatch, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use bytes::Bytes;
use datafusion::error::DataFusionError;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use jammi_db::catalog::artifact_repo::{MaterializationSummary, ReclaimDecision, StagedArtifact};
use jammi_db::catalog::instance::{
    DeviceFact, GangListing, GangMember, InstanceRegistration, PeerAddr, WorkerFacts,
};
use jammi_db::catalog::jobs_repo::{AssemblyOutcome, TrainingSetAssembly, WorkerState};
use jammi_db::catalog::lease_keeper::{HoldRelease, LeaseHold, LeaseKeeper, LeaseTarget};
use jammi_db::catalog::model_repo::ModelLocation;
use jammi_db::catalog::Catalog;
use jammi_db::config::WorkerIntervals;
use jammi_db::error::{JammiError, Result};
use jammi_db::model_task::ModelTask;
use jammi_db::sql::{quote_ident, source_relation};
use jammi_db::store::manifest::{
    ComputeDevice, GraphSampleFields, InputAnchor, ProducingDescriptor,
};
use jammi_db::store::{ArtifactStore, CachePolicy, TrainingSetInput, TrainingSetSpec};
use jammi_db::tenant::TenantId;
use tokio::sync::watch;

use crate::fine_tune::collective::{
    BlockingCall, Collective, CoordinatorLink, LocalGang, MemberEnd, MemberLink, Peer,
};
use crate::fine_tune::decode::{
    build_training_data_loader, detect_training_format, extract_string_column,
};
use crate::fine_tune::graph_sampler::{
    GraphEdge, GraphFineTuneSources, GraphSampleConfig, GraphSampler, SampledPair, TextNode,
};
use crate::fine_tune::partition::{PartitionRule, PartitionSpec};
use crate::fine_tune::role::{LeaseHolder, RunnerRole};
use crate::fine_tune::spec::{TrainingCommon, TrainingPlan, TrainingSetProducer, TrainingSpec};
use crate::fine_tune::trainer::RankContext;
use crate::fine_tune::training_set;
use crate::fine_tune::FineTuneConfig;
use crate::jobs::UnsuccessfulEnd;
use crate::model::backend::DeviceConfig;
use crate::model::hub::HubSource;
use crate::model::ModelSource;
use crate::operator::gang_exec::{GangDescriptor, PlacedOutcome};
use crate::session::InferenceSession;
use jammi_wire::proto::gang::{AbortReason, Assign};

// Lease timing is configured per deployment via `[lease]` in `JammiConfig` (the
// one lease primitive every leased row shares), the idle poll via `[worker]`
// (`idle_poll_secs`; the loop itself is gated by `[worker] enabled`), and both
// resolve to a [`WorkerIntervals`] (see
// [`jammi_db::config::WorkerConfig::worker_intervals`]). The lease is the
// window a claimed job is exclusively owned; the heartbeat renews it well
// inside that window so a single missed beat (a GC pause, a slow tick) does not
// drop the lease — the config layer enforces a ≥2× margin between the lease and
// the beat so that invariant holds for every deployment, never silently
// clamped. The idle poll is how often an idle worker checks for new work, and
// reclaim runs each idle tick so a dead worker's job is recovered within roughly
// one poll + lease. The defaults reproduce the historical 30 s / 10 s / 1 s
// timing; a short config drives lease-expiry and reclaim quickly.

/// Attempts cap before `reclaim_expired_jobs` fails a job for good.
pub(crate) const MAX_ATTEMPTS: u32 = 3;

/// Environment LABEL for this process's `instances` row — an operator's
/// human-readable name for the process (a node name, a replica slot) that
/// `ListWorkers` and logs show beside the process's minted id. It is NEVER
/// the process's identity: `instances.instance_id`/`jobs.claimed_by` are a
/// per-process UUID ([`mint_instance_id`]), so two processes sharing one
/// label (a restart, a sibling replica) are two instances and a dead one's
/// inline jobs are failed by the liveness reclaim rather than kept alive
/// by its namesake's heartbeat. Non-unique by design.
const WORKER_LABEL_ENV: &str = "JAMMI_WORKER_ID";

/// The trimmed `JAMMI_WORKER_ID` when set and non-empty, else `None` — an
/// all-whitespace value is treated as unset (a blank label labels nothing).
pub(crate) fn worker_label() -> Option<String> {
    match std::env::var(WORKER_LABEL_ENV) {
        Ok(v) if !v.trim().is_empty() => Some(v.trim().to_string()),
        _ => None,
    }
}

/// Mint this process's `instances`/`jobs.claimed_by` identity: a fresh
/// UUID, read from nothing in the environment. Called once per session
/// construction (`InferenceSession::instance_id`).
pub(crate) fn mint_instance_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// Every job kind this binary can execute — the vocabulary
/// `resolve_kinds` validates `[worker] kinds` against at startup.
/// The three training kinds dispatch through `JobWorker::run_spec`; the
/// five compute kinds (every embedded synchronous compute verb is one of
/// [`crate::jobs::ComputeSpec`]'s variants) dispatch through
/// [`crate::jobs::execute_compute`].
pub const COMPILED_KINDS: &[&str] = &[
    "fine_tune",
    "graph_fine_tune",
    "context_predictor",
    "neighbor_graph",
    "propagate",
    "asof_join",
    "embedding",
    "infer",
];

/// Whether `kind` is one of [`crate::jobs::ComputeSpec`]'s variants (dispatched
/// through [`crate::jobs::execute_compute`]) rather than a [`TrainingSpec`]
/// variant (dispatched through [`JobWorker::run_spec`]).
pub(crate) fn is_compute_kind(kind: &str) -> bool {
    matches!(
        kind,
        "neighbor_graph" | "propagate" | "asof_join" | "embedding" | "infer"
    )
}

/// Resolve `[worker] kinds` against [`COMPILED_KINDS`] at startup.
/// `WorkerKinds::All` claims every compiled
/// kind; `WorkerKinds::Only` is validated member-by-member and returned
/// as-is — an unknown name is a typed [`JammiError::Config`], never a
/// silently-ignored token.
fn resolve_kinds(kinds: &jammi_db::config::WorkerKinds) -> Result<Vec<String>> {
    use jammi_db::config::WorkerKinds;
    match kinds {
        WorkerKinds::All(_) => Ok(COMPILED_KINDS.iter().map(|s| s.to_string()).collect()),
        WorkerKinds::Only(list) => {
            for k in list {
                if !COMPILED_KINDS.contains(&k.as_str()) {
                    return Err(JammiError::Config(format!(
                        "[worker] kinds names unknown job kind '{k}' -- compiled kinds are: {}",
                        COMPILED_KINDS.join(", ")
                    )));
                }
            }
            Ok(list.clone())
        }
    }
}

/// This host's shutdown phase: `Running` until a
/// DRAIN or RELEASE begins; `Draining` finishes the in-flight job and stops
/// claiming; `Releasing` hands every lease back and stops at once. Owned by
/// the session's [`HostAdmission`] (one `watch` cell), read by the claim
/// loop's gate and by every admitted gang rank's hold loop alike — a phase
/// leaving `Running` ends a held rank with the host-initiated `Drain`
/// reason, the same instant it stops the loop from claiming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerPhase {
    Running,
    Draining,
    Releasing,
}

/// Who holds this host's single job slot — the per-process holder cell a
/// peer is defined by: a peer never claims
/// while it holds a rank, never receives a rank while it runs a loop-claimed
/// job, and is reachable whenever idle. Every transition is a
/// compare-and-set on one `watch` cell (`send_if_modified`), never a lock
/// held across an `.await`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Holder {
    /// Nobody: the loop is idle (or between claims) and no rank is held.
    Free,
    /// The claim loop is inside `claim_next` or the claim→hold prologue —
    /// a claim transaction may be in flight, so nothing aborts the loop
    /// task here and a rank waits (bounded) for the probe to resolve.
    ClaimProbe,
    /// A loop-claimed job runs under a registered lease hold.
    JobRun,
    /// A loop-claimed attempt is submitting a `GangDescriptor` through an
    /// installed `PlacedGangSubmitter`, or awaiting its stream: this host runs no
    /// compute for `(job_id, attempt)` while it waits, so it can still
    /// serve a `RunRank` session for some OTHER attempt —
    /// [`HostAdmission::try_hold_rank`] admits out of this state exactly as
    /// it does out of `Free` — a two-host fleet could not otherwise
    /// assemble if its only free-looking host were the one awaiting a
    /// placement result. [`HostAdmission::probe_claim`] still refuses it,
    /// exactly like `JobRun`.
    Awaiting { job_id: String, attempt: u32 },
    /// An admitted gang rank is held for `(job_id, attempt)`.
    Rank { job_id: String, attempt: u32 },
}

/// Why [`HostAdmission::try_hold_rank`] did not take the slot — the holder
/// it found instead. Every arm is TRANSIENT from the caller's side (the
/// coordinator retries after at least one heartbeat or picks another
/// member); none consumes any assembly budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HolderBusy {
    /// A claim probe is in flight — wait at most one heartbeat, then retry.
    ClaimProbe,
    /// A loop-claimed job is running here.
    JobRun,
    /// Another rank is held: a different job, or the SAME job at an
    /// attempt not below the caller's (an equal attempt is a duplicate
    /// assignment of a session already held; a greater one supersedes the
    /// caller). A lesser held attempt never refuses — it is taken over.
    Rank { job_id: String, attempt: u32 },
}

/// This host's admission state, owned by the session and shared (one `Arc`)
/// by the claim loop, its [`EmbeddedWorker`] guard, and the gang admission
/// handler: the shutdown [`WorkerPhase`], the slot [`Holder`], and this
/// process's fleet [`InstanceRegistration`] (the `instances`/`workers` row
/// carrier, whose worker half the claim loop alone writes).
pub struct HostAdmission {
    phase: watch::Sender<WorkerPhase>,
    holder: watch::Sender<Holder>,
    registry: Arc<InstanceRegistration>,
    /// How a coordinator on this host reaches a gang member's `RunRank`
    /// installed ONCE by the process that mounts
    /// the gang listener (`jammi-server`'s `OssServer::bind`, with
    /// `gang_rounds::dial_member` behind it — the engine crate owns no
    /// transport), absent in a library process, which therefore cannot
    /// coordinate a `Peer` gang and says so as an assembly outcome
    /// ([`CoordinatorEnd::HostCannotCoordinate`]). Write-once: a second
    /// install is refused, never a silent swap under a running body.
    dialer: OnceLock<Arc<dyn MemberDialer>>,
    /// Installed ONCE by the SCHEDULER role (`crates/jammi-ballista`): how a
    /// claimant on this host submits its OWN training job as one Ballista
    /// task instead of running it in-process. Absent on a process that hosts
    /// no scheduler.
    placed_gang_submitter: OnceLock<Arc<dyn PlacedGangSubmitter>>,
    /// Installed ONCE by the EXECUTOR role: how `GangExec::execute` — which
    /// runs with only a Ballista `TaskContext` in hand, never a session —
    /// reaches this process's coordinator body (see [`placed_gang_runner`]'s
    /// doc for the process-global seam this backs).
    placed_gang_runner: OnceLock<Arc<dyn PlacedGangRunner>>,
    /// The single claim-loop slot: `0` (free) or
    /// a nonzero GENERATION id — the id [`HostAdmission::try_claim_loop`]
    /// handed out to whichever [`EmbeddedWorker`] currently owns the slot.
    /// A second `spawn`/`spawn_worker` while a generation is live finds this
    /// nonzero and is refused with a typed error before it builds any task
    /// — "one claim loop per session" is therefore a compare-and-set on
    /// this cell, not a premise the RELEASE mechanism merely assumes (the
    /// phase/hold barrier alone is necessary but not sufficient — it cannot
    /// stop a second loop from existing in the first place).
    ///
    /// The slot is held from a successful claim until [`EmbeddedWorker::
    /// release_and_stop`] completes OR the [`EmbeddedWorker`] value is
    /// dropped, whichever comes first — both release through
    /// [`HostAdmission::release_loop_claim`], a compare-and-set against the
    /// CALLER's OWN generation id, so a release that lands after a
    /// successor has already claimed a NEW generation is a harmless no-op
    /// rather than stealing the successor's slot.
    loop_owner: AtomicU64,
    /// The next generation id [`HostAdmission::try_claim_loop`] hands out —
    /// monotonic, never reused, never zero (zero is reserved for "free").
    next_generation: AtomicU64,
    /// How many times [`HostAdmission::begin_release`] has run, on this
    /// session, ever — bumped on EVERY call, whether or not the phase
    /// actually changed. The session-scoped release barrier is EPOCH-based,
    /// not phase-based, because [`HostAdmission::try_claim_loop`] resets
    /// `phase` to `Running` for each new generation (a fresh loop must not
    /// be born already refusing every claim), so `phase() == Releasing`
    /// alone cannot distinguish "this generation was released" from "a
    /// LATER generation reset the phase after an earlier release": a claim
    /// loop's own [`WorkerShared`] instead snapshots this counter at spawn
    /// (`WorkerShared::spawn_release_epoch`) and treats ANY later value as
    /// "I have been released", however far its own task's execution lags
    /// behind an `abort()` request (`register_job_hold_or_release`'s
    /// self-release check and [`WorkerShared::admits_claim`] both read it).
    /// A generation born AFTER a release therefore snapshots the
    /// already-bumped counter and is never stopped by that release — only a
    /// LATER one.
    release_epoch: AtomicU64,
}

/// The process-global weak link to whichever session's [`HostAdmission`]
/// installed a [`PlacedGangRunner`] — set the one time
/// [`HostAdmission::install_placed_gang_runner`] succeeds anywhere in this
/// process, never a second, independent global (see
/// `crate::operator::gang_exec`'s module doc for the refutation of a
/// `TaskContext`-extension alternative).
static PLACED_GANG_HOST: OnceLock<Weak<HostAdmission>> = OnceLock::new();

/// The process's installed [`PlacedGangRunner`], if this process's session
/// hosts a Ballista executor — the ONLY way
/// `GangExec::execute` reaches it, since its
/// `execute` runs with no session in hand. `None` both when no session on
/// this process ever installed one, and when the installing session has
/// since dropped (the weak upgrade fails).
pub fn placed_gang_runner() -> Option<Arc<dyn PlacedGangRunner>> {
    PLACED_GANG_HOST
        .get()
        .and_then(Weak::upgrade)
        .and_then(|admission| admission.placed_gang_runner())
}

/// Submit a training job as one Ballista task instead of running it
/// in-process — installed by the CLIENT role (`crates/jammi-ballista`)
/// through [`HostAdmission::install_placed_gang_submitter`].
/// `run_claimed_job_under` checks this seam, before `run_spec`/topology are
/// ever reached, for every claimed `fine_tune`/`graph_fine_tune` attempt a
/// non-placed run makes: `placement_available()` true means SOME OTHER registered executor
/// exists to place the job on. The stream's items are DataFusion's own
/// `Result` — this is exactly Ballista's `execute_physical_plan` result,
/// carried unwrapped, never re-typed through `JammiError`.
pub trait PlacedGangSubmitter: Send + Sync {
    /// Submit `descriptor` and hand back the physical plan's own output
    /// stream (Ballista's), or a jammi-side error raised BEFORE any task
    /// was ever scheduled (a dial failure, a device-less cluster refusing
    /// the submission typed).
    fn submit(
        &self,
        descriptor: GangDescriptor,
    ) -> BoxFuture<
        'static,
        Result<BoxStream<'static, std::result::Result<RecordBatch, DataFusionError>>>,
    >;

    /// Whether SOME OTHER registered executor exists to place a job on
    /// right now (the client role answers this from the catalog's executor
    /// registrations) — `false` degrades every claim on this host straight
    /// to its in-process run, never a submission with nowhere to land.
    fn placement_available(&self) -> bool;
}

/// Run a placed gang's coordinator body on THIS process — installed by the
/// EXECUTOR role through [`HostAdmission::install_placed_gang_runner`];
/// `GangExec::execute` dispatches through it
/// via the process-global [`placed_gang_runner`] (that function's doc states
/// why: a Ballista executor's `TaskContext` carries no jammi session).
pub trait PlacedGangRunner: Send + Sync {
    fn run(
        &self,
        descriptor: GangDescriptor,
    ) -> BoxFuture<'static, Result<crate::operator::gang_exec::PlacedOutcome>>;
}

/// The coordinator's one transport seam: open `RunRank` on a member's
/// `peer_bind` listener with the coordinator's `Assign`, require `Admitted`,
/// and hand back the [`CoordinatorLink`] its `Peer` is built from, the
/// client's inbound decode capped at `max_message_bytes`. The engine crate
/// declares the seam; the server crate implements it over its own dialer
/// (`gang_rounds::dial_member`) and installs it through
/// [`HostAdmission::install_member_dialer`].
pub trait MemberDialer: Send + Sync {
    fn dial<'a>(
        &'a self,
        addr: &'a PeerAddr,
        assign: Assign,
        max_message_bytes: usize,
    ) -> Pin<Box<dyn Future<Output = Result<CoordinatorLink>> + Send + 'a>>;
}

impl HostAdmission {
    /// Fresh admission state: phase `Running`, holder `Free`, no dialer, no
    /// placed-gang seam.
    pub fn new(registry: Arc<InstanceRegistration>) -> Arc<Self> {
        let (phase, _) = watch::channel(WorkerPhase::Running);
        let (holder, _) = watch::channel(Holder::Free);
        Arc::new(Self {
            phase,
            holder,
            registry,
            dialer: OnceLock::new(),
            placed_gang_submitter: OnceLock::new(),
            placed_gang_runner: OnceLock::new(),
            loop_owner: AtomicU64::new(0),
            next_generation: AtomicU64::new(1),
            release_epoch: AtomicU64::new(0),
        })
    }

    /// Claim the session's single claim-loop slot. `Some((generation,
    /// release_epoch))` means this call won the slot (no loop was live):
    /// `generation` is this claim's unique, nonzero id — the caller's own
    /// key for [`Self::release_loop_claim`] — and `release_epoch` is the
    /// count of [`Self::begin_release`] calls THIS SESSION HAS EVER SEEN,
    /// snapshotted at birth (`WorkerShared::spawn_release_epoch`'s source):
    /// a fresh generation is never stopped by a release that predates it.
    /// `None` means a loop already owns the slot and the caller must not
    /// spawn a second one. Resets `phase` to `Running` on a win — a new
    /// generation must not be born already refusing every claim because a
    /// PRIOR generation's `Draining`/`Releasing` phase was left in the
    /// cell.
    pub(crate) fn try_claim_loop(&self) -> Option<(u64, u64)> {
        let candidate = self.next_generation.fetch_add(1, Ordering::SeqCst);
        self.loop_owner
            .compare_exchange(0, candidate, Ordering::SeqCst, Ordering::SeqCst)
            .ok()?;
        self.phase.send_replace(WorkerPhase::Running);
        Some((candidate, self.release_epoch.load(Ordering::SeqCst)))
    }

    /// Release the claim-loop slot ONLY if it is still held by `generation`
    /// — a compare-and-set, not an unconditional write, so a release that
    /// lands after a successor has already claimed a NEW generation (e.g.
    /// `release_and_stop` freeing it, a caller immediately spawning a
    /// successor, and only THEN this same guard's own `Drop` running) is a
    /// harmless no-op instead of stealing the successor's slot. Called from
    /// both [`EmbeddedWorker::release_and_stop`] (on success) and
    /// [`EmbeddedWorker`]'s `Drop` (unconditionally attempted, idempotent
    /// either way it lands).
    pub(crate) fn release_loop_claim(&self, generation: u64) {
        let _ = self
            .loop_owner
            .compare_exchange(generation, 0, Ordering::SeqCst, Ordering::SeqCst);
    }

    /// Install the process's [`MemberDialer`] — once. `false` when one is
    /// already installed (the first stays).
    pub fn install_member_dialer(&self, dialer: Arc<dyn MemberDialer>) -> bool {
        self.dialer.set(dialer).is_ok()
    }

    /// The installed [`MemberDialer`], if this process mounted a gang
    /// listener.
    pub fn member_dialer(&self) -> Option<Arc<dyn MemberDialer>> {
        self.dialer.get().cloned()
    }

    /// Install the process's [`PlacedGangSubmitter`] — once. `false` when
    /// one is already installed (the [`MemberDialer`] shape).
    pub fn install_placed_gang_submitter(&self, submitter: Arc<dyn PlacedGangSubmitter>) -> bool {
        self.placed_gang_submitter.set(submitter).is_ok()
    }

    /// The installed [`PlacedGangSubmitter`], if this process holds the
    /// Ballista client role.
    pub fn placed_gang_submitter(&self) -> Option<Arc<dyn PlacedGangSubmitter>> {
        self.placed_gang_submitter.get().cloned()
    }

    /// Install the process's [`PlacedGangRunner`] — once — and, on that
    /// first install only, register this admission as the process-global
    /// `PLACED_GANG_HOST` a body-less `GangExec::execute` reaches it
    /// through (`false` on a second install, the same [`MemberDialer`]
    /// shape; the global is set only alongside a WINNING install, never on
    /// a losing one).
    pub fn install_placed_gang_runner(self: &Arc<Self>, runner: Arc<dyn PlacedGangRunner>) -> bool {
        let installed = self.placed_gang_runner.set(runner).is_ok();
        if installed {
            let _ = PLACED_GANG_HOST.set(Arc::downgrade(self));
        }
        installed
    }

    /// The installed [`PlacedGangRunner`], if this process mounted a
    /// Ballista executor.
    pub fn placed_gang_runner(&self) -> Option<Arc<dyn PlacedGangRunner>> {
        self.placed_gang_runner.get().cloned()
    }

    /// `JobRun → Awaiting{job_id, attempt}` — the claim loop's own attempt
    /// is about to submit a `GangDescriptor` (the move precedes the submit)
    /// and then awaits its stream: this host runs no compute for the
    /// attempt meanwhile, so it can still serve a `RunRank` session
    /// ([`Self::try_hold_rank`]'s `Awaiting` arm admits exactly as `Free`
    /// does) — a two-host fleet could not otherwise assemble if its only
    /// free-looking host were the one busy awaiting a placement result. A
    /// refused (`Err(the holder the CAS saw)`) unless the holder is exactly `JobRun` — a direct
    /// `run_claimed_job`/an inline `run_now` (no [`ClaimGuard`]) or a slot
    /// already superseded never observes this transition. No corresponding
    /// "end awaiting" call is needed: the loop's own [`ClaimGuard`], still
    /// held across the whole submit-and-await, resets `Awaiting` to `Free`
    /// on drop exactly as it resets `ClaimProbe`/`JobRun`.
    pub(crate) fn begin_awaiting_placement(
        &self,
        job_id: &str,
        attempt: u32,
    ) -> std::result::Result<(), Holder> {
        let mut found: Option<Holder> = None;
        let moved = self.holder.send_if_modified(|h| {
            if *h == Holder::JobRun {
                *h = Holder::Awaiting {
                    job_id: job_id.to_string(),
                    attempt,
                };
                true
            } else {
                found = Some(h.clone());
                false
            }
        });
        if moved {
            Ok(())
        } else {
            // The holder the CAS saw, in the SAME critical section — the
            // caller decides on that value, never on a second read.
            Err(found.expect(
                "send_if_modified runs its closure exactly once; a refusal recorded the holder",
            ))
        }
    }

    /// This process's registration — the ONE carrier its `instances` row
    /// (and, once a claim loop runs, its `workers` row) is written from.
    pub fn registry(&self) -> &Arc<InstanceRegistration> {
        &self.registry
    }

    /// The current shutdown phase.
    pub fn phase(&self) -> WorkerPhase {
        *self.phase.borrow()
    }

    /// A receiver on the phase watch — a held rank `wait_for(|p| *p !=
    /// Running)`s on it and ends with the `Drain` reason when it fires.
    pub fn phase_receiver(&self) -> watch::Receiver<WorkerPhase> {
        self.phase.subscribe()
    }

    /// Begin a DRAIN: `Running → Draining` (a `Releasing` phase already in
    /// force is never regressed). Returns whether this call flipped it.
    pub fn begin_drain(&self) -> bool {
        self.phase.send_if_modified(|p| {
            if *p == WorkerPhase::Running {
                *p = WorkerPhase::Draining;
                true
            } else {
                false
            }
        })
    }

    /// Begin a RELEASE: the phase is `Releasing` from this instant, whatever
    /// it was (a RELEASE wins over a DRAIN in progress), AND the session's
    /// release epoch is bumped — UNCONDITIONALLY, even when the phase was
    /// already `Releasing`, since a distinct RELEASE call (e.g.
    /// `InferenceSession::release_job_leases` racing an in-flight
    /// `EmbeddedWorker::release_and_stop`) is still a distinct release event
    /// any generation born before it must be sensitive to (see the
    /// `release_epoch` field's own doc). This is the ONE session-scoped
    /// signal `WorkerShared::admits_claim` and `register_job_hold_or_
    /// release`'s self-release check both read — `phase()` alone cannot
    /// serve that role because `try_claim_loop` resets it for every new
    /// generation.
    ///
    /// The phase flip runs strictly BEFORE the epoch bump — LOAD-BEARING,
    /// not incidental: `run_placed_gang`'s own doc and `WorkerShared::
    /// for_single_run`'s (the two-catch lattice over a birth-epoch snapshot
    /// taken before `probe_claim()`) both depend on "the bump is visible ⇒
    /// the flip already happened", which only holds in THIS order. Swapping
    /// the two statements admits a gang on a releasing host: a birth-epoch
    /// snapshot taken inside the (relocated) window between the bump
    /// and the flip already contains the bump, so `released_since_birth`
    /// reads `false` downstream, while `probe_claim`'s phase check — racing
    /// the same window from the other side — still reads `Running` and
    /// admits. Pinned by `begin_release_bumps_the_epoch_strictly_after_the_
    /// phase_flip_is_already_visible` (`fine_tune::worker::tests`), which
    /// parks a live `begin_release` call between the two statements
    /// (`loop_test_hooks::ParkPoint::BeginReleaseBetweenFlipAndBump`) and
    /// asserts `probe_claim` already refuses while the epoch is still
    /// unbumped.
    pub async fn begin_release(&self) {
        self.phase.send_if_modified(|p| {
            if *p == WorkerPhase::Releasing {
                false
            } else {
                *p = WorkerPhase::Releasing;
                true
            }
        });
        #[cfg(feature = "test-hooks")]
        loop_test_hooks::maybe_park(
            &self.registry.instance_id,
            loop_test_hooks::ParkPoint::BeginReleaseBetweenFlipAndBump,
        )
        .await;
        self.release_epoch.fetch_add(1, Ordering::SeqCst);
    }

    /// How many times [`Self::begin_release`] has run on this session, ever
    /// — see the `release_epoch` field's own doc.
    pub(crate) fn release_epoch(&self) -> u64 {
        self.release_epoch.load(Ordering::SeqCst)
    }

    /// Test-only: set the phase WITHOUT any stop request or row write —
    /// the gate-direct shape no real `begin_drain`/`release_and_stop` call
    /// produces (both pair the flip with a stop), used to prove the loop's
    /// gate refuses a claim on the phase read alone.
    #[cfg(feature = "test-hooks")]
    pub fn set_phase_for_test(&self, phase: WorkerPhase) {
        self.phase.send_replace(phase);
    }

    /// The current holder (a snapshot).
    pub fn holder(&self) -> Holder {
        self.holder.borrow().clone()
    }

    /// A receiver on the holder watch — what a rank waits on while a
    /// `ClaimProbe` resolves.
    pub fn holder_receiver(&self) -> watch::Receiver<Holder> {
        self.holder.subscribe()
    }

    /// The claim loop's probe, immediately before `claim_next`: `Free →
    /// ClaimProbe`. `None` when the slot is held — a rank is admitted here
    /// (a peer never claims while it holds a rank), or a job already runs —
    /// so the loop skips this iteration's claim. The returned guard resets
    /// `ClaimProbe`/`JobRun` to `Free` when dropped, on every exit path of
    /// the iteration it covers (the claim finding nothing, the run
    /// returning, a panic, the loop future being aborted).
    pub fn probe_claim(self: &Arc<Self>) -> Option<ClaimGuard> {
        // A host that has begun a DRAIN or RELEASE admits nothing new — the
        // claim loop's own gate stops it claiming, and the placed-gang
        // runner (`JobWorker::run_placed_gang`, dialled by the executor)
        // is refused here the same way, so a gang bound to this host inside
        // its termination grace is never started on a process about to
        // exit ("finish what's running, refuse what's new" holds for every
        // entry, not only the loop's).
        if *self.phase.borrow() != WorkerPhase::Running {
            return None;
        }
        let taken = self.holder.send_if_modified(|h| {
            if *h == Holder::Free {
                *h = Holder::ClaimProbe;
                true
            } else {
                false
            }
        });
        taken.then(|| ClaimGuard {
            admission: Arc::clone(self),
        })
    }

    /// `ClaimProbe → JobRun`, at the loop's hold site once the claimed
    /// job's lease hold is registered (never earlier: the claim→hold
    /// prologue stays a `ClaimProbe`, so a RELEASE landing inside it still
    /// waits for the prologue's own self-release rather than aborting a
    /// claim whose lease would then only fall to expiry — a self-release
    /// costs zero net attempts). A direct [`JobWorker::run_claimed_job`] run, or an inline
    /// `run_now`, holds no probe and leaves the cell as it was.
    pub(crate) fn job_running(&self) {
        self.holder.send_if_modified(|h| {
            if *h == Holder::ClaimProbe {
                *h = Holder::JobRun;
                true
            } else {
                false
            }
        });
    }

    /// A rank's compare-and-set: `Free → Rank{job_id, attempt}`, or a held
    /// `Rank` of the SAME job at a LESSER attempt superseded in place (the
    /// job's attempt moved on; the elder session no longer owns the cell,
    /// its next re-verification refutes it against the row, and its guard's
    /// drop leaves the cell alone). Every other holder refuses with the
    /// [`HolderBusy`] it found — decided at once, no waiting; a caller that
    /// tolerates a `ClaimProbe` waits through [`Self::admit_rank`].
    pub fn try_hold_rank(
        self: &Arc<Self>,
        job_id: &str,
        attempt: u32,
    ) -> std::result::Result<RankHold, HolderBusy> {
        let mut busy: Option<HolderBusy> = None;
        self.holder.send_if_modified(|h| match h {
            Holder::Free => {
                *h = Holder::Rank {
                    job_id: job_id.to_string(),
                    attempt,
                };
                true
            }
            // A host awaiting its OWN placed attempt's stream runs no
            // compute meanwhile, so it can still serve a rank of some
            // OTHER attempt — admitted exactly like `Free`.
            // The awaited attempt's own eventual `ClaimGuard::drop` no
            // longer finds `Awaiting` in the cell in this case and is a
            // no-op, leaving this rank's hold untouched — the same rule a
            // superseded `Rank`'s elder guard already follows.
            Holder::Awaiting { .. } => {
                *h = Holder::Rank {
                    job_id: job_id.to_string(),
                    attempt,
                };
                true
            }
            Holder::Rank {
                job_id: held_job,
                attempt: held_attempt,
            } if held_job == job_id && *held_attempt < attempt => {
                *held_attempt = attempt;
                true
            }
            Holder::ClaimProbe => {
                busy = Some(HolderBusy::ClaimProbe);
                false
            }
            Holder::JobRun => {
                busy = Some(HolderBusy::JobRun);
                false
            }
            Holder::Rank {
                job_id: held_job,
                attempt: held_attempt,
            } => {
                busy = Some(HolderBusy::Rank {
                    job_id: held_job.clone(),
                    attempt: *held_attempt,
                });
                false
            }
        });
        match busy {
            Some(busy) => Err(busy),
            None => Ok(RankHold {
                admission: Arc::clone(self),
                job_id: job_id.to_string(),
                attempt,
            }),
        }
    }

    /// [`Self::try_hold_rank`] that waits out a `ClaimProbe`: a probe found
    /// in the cell is waited on for at most `bound` (one heartbeat — the
    /// longest a claim round trip plus its prologue takes), retrying the
    /// CAS on every holder change; the slot freeing within the bound admits,
    /// the bound elapsing (or the probe resolving into a `JobRun`) refuses
    /// with what was found. `JobRun` and another `Rank` refuse at once.
    pub async fn admit_rank(
        self: &Arc<Self>,
        job_id: &str,
        attempt: u32,
        bound: Duration,
    ) -> std::result::Result<RankHold, HolderBusy> {
        let deadline = tokio::time::Instant::now() + bound;
        let mut rx = self.holder_receiver();
        loop {
            // Mark the current value seen BEFORE the CAS, so a change that
            // lands between the CAS and the `changed().await` below is
            // still observed as a change (a `watch` receiver's `changed`
            // resolves for any version newer than the last one it marked).
            rx.borrow_and_update();
            match self.try_hold_rank(job_id, attempt) {
                Err(HolderBusy::ClaimProbe) => {
                    let now = tokio::time::Instant::now();
                    if now >= deadline {
                        return Err(HolderBusy::ClaimProbe);
                    }
                    match tokio::time::timeout(deadline - now, rx.changed()).await {
                        Ok(Ok(())) => continue,
                        // The sender dropped (the session is gone) or the
                        // bound elapsed: refuse with what was found.
                        Ok(Err(_)) | Err(_) => return Err(HolderBusy::ClaimProbe),
                    }
                }
                other => return other,
            }
        }
    }

    /// Test-only: overwrite the holder and hand back a guard that resets it
    /// to `Free` on drop — how a test manufactures a `JobRun`/`ClaimProbe`/
    /// foreign-`Rank` holder without running a job, to drive the admission
    /// lattice's contention arms deterministically.
    #[cfg(feature = "test-hooks")]
    pub fn hold_for_test(self: &Arc<Self>, holder: Holder) -> TestHold {
        self.holder.send_replace(holder);
        TestHold {
            admission: Arc::clone(self),
        }
    }
}

/// The claim loop's probe guard — see [`HostAdmission::probe_claim`].
pub struct ClaimGuard {
    admission: Arc<HostAdmission>,
}

impl Drop for ClaimGuard {
    fn drop(&mut self) {
        self.admission.holder.send_if_modified(|h| {
            if matches!(
                h,
                Holder::ClaimProbe | Holder::JobRun | Holder::Awaiting { .. }
            ) {
                *h = Holder::Free;
                true
            } else {
                false
            }
        });
    }
}

/// An admitted rank's hold on the slot — see
/// [`HostAdmission::try_hold_rank`]. Dropping it frees the slot ONLY if the
/// cell still names this exact `(job_id, attempt)`: a session superseded by
/// the same job's greater attempt leaves the successor's hold untouched.
pub struct RankHold {
    admission: Arc<HostAdmission>,
    job_id: String,
    attempt: u32,
}

impl std::fmt::Debug for RankHold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RankHold")
            .field("job_id", &self.job_id)
            .field("attempt", &self.attempt)
            .finish()
    }
}

impl RankHold {
    /// The job this hold was admitted for.
    pub fn job_id(&self) -> &str {
        &self.job_id
    }

    /// The attempt this hold was admitted at.
    pub fn attempt(&self) -> u32 {
        self.attempt
    }
}

impl Drop for RankHold {
    fn drop(&mut self) {
        self.admission.holder.send_if_modified(|h| {
            let mine = matches!(
                h,
                Holder::Rank { job_id, attempt }
                    if *job_id == self.job_id && *attempt == self.attempt
            );
            if mine {
                *h = Holder::Free;
            }
            mine
        });
    }
}

/// Test-only: see [`HostAdmission::hold_for_test`].
#[cfg(feature = "test-hooks")]
pub struct TestHold {
    admission: Arc<HostAdmission>,
}

#[cfg(feature = "test-hooks")]
impl Drop for TestHold {
    fn drop(&mut self) {
        self.admission.holder.send_replace(Holder::Free);
    }
}

/// What the loop task is doing, as observed through a `watch` the in-task
/// `LoopExitGuard` writes on EVERY exit path — a writer placed after the
/// loop's `.await` would never run when the task is aborted or panics, a
/// `Drop` guard always does. `Stopped` is the cooperative return (the stop
/// flag, the gate refusing, or the session dropping); `Aborted` is the task
/// being dropped mid-poll (an `abort()`); `Failed` is a panic unwinding
/// through the loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopState {
    Running,
    Stopped,
    Aborted,
    Failed,
}

/// State shared between the loop task, the [`EmbeddedWorker`] guard that
/// owns it, and the process's observers (`/healthz`, `/metrics`) — one
/// `Arc`, held strongly by the loop task and the guard, handed out `Weak`
/// through [`EmbeddedWorker::shared`].
///
/// * `admission` is the session's [`HostAdmission`]: the shutdown phase the
///   loop's gate reads, and the slot [`Holder`] the loop moves through
///   `Free → ClaimProbe → JobRun → Free` around every claim (an inline
///   `run_now` and a direct `run_claimed_job` never touch it).
/// * `stop` is a level-triggered `watch<bool>`: the loop's pre-claim check
///   reads it and its idle sleep is `select!`ed against `wait_for(|v| *v)`,
///   so a stop set during the sleep wakes the loop at once and a receiver
///   subscribed after the send still resolves — a wakeup cannot be lost.
/// * `state` is the [`LoopState`] watch the exit guard writes.
pub struct WorkerShared {
    admission: Arc<HostAdmission>,
    stop: watch::Sender<bool>,
    state_tx: watch::Sender<LoopState>,
    instance_id: String,
    /// The gauge sampler's last catalog snapshot (`/metrics` copies it on a
    /// scrape; the sampler task writes it every `[worker]
    /// metrics_sample_secs`).
    sample: std::sync::RwLock<WorkerSample>,
    /// How many catalog samples the sampler has taken — the oracle that a
    /// scrape issues no catalog statement of its own.
    samples_taken: AtomicU64,
    /// [`HostAdmission::release_epoch`], snapshotted at construction: any
    /// LATER value observed on `admission` means
    /// a RELEASE has happened since this shared state was born, and is
    /// treated as "I have been released" regardless of what `phase()`
    /// currently reads (a later generation may have reset it to `Running`)
    /// — see [`Self::released_since_birth`].
    spawn_release_epoch: u64,
}

/// The queue-depth snapshot the gauge sampler last read from the catalog:
/// `(kind, count)` for `queued` and for `running` loop-claimable rows
/// (`execution = 'queued'`; a held `claimable = false` row counts as
/// queued). A kind present in the previous sample but absent now is carried
/// at 0, so its gauge falls to zero instead of going stale.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorkerSample {
    pub queued: Vec<(String, i64)>,
    pub running: Vec<(String, i64)>,
}

impl WorkerShared {
    /// Fresh shared state for one loop task over the session's
    /// `admission`: stop unset, state `Running`. `spawn_release_epoch` is
    /// this state's birth snapshot of `HostAdmission::release_epoch` —
    /// callers that go through `HostAdmission::try_claim_loop`
    /// ([`EmbeddedWorker::spawn_worker`]) pass the epoch that call
    /// returned; `for_single_run` is the OTHER shape (a fresh,
    /// un-looped single-job run outside the claim-loop slot) and passes
    /// `admission.release_epoch()` read live.
    pub fn new(
        admission: Arc<HostAdmission>,
        instance_id: String,
        spawn_release_epoch: u64,
    ) -> Arc<Self> {
        let (stop, _) = watch::channel(false);
        let (state_tx, _) = watch::channel(LoopState::Running);
        Arc::new(Self {
            admission,
            stop,
            state_tx,
            instance_id,
            sample: std::sync::RwLock::new(WorkerSample::default()),
            samples_taken: AtomicU64::new(0),
            spawn_release_epoch,
        })
    }

    /// Fresh shared state for ONE claimed-job run OUTSIDE the claim-loop
    /// slot — `JobWorker::run_claimed_job`'s and `run_placed_gang`'s shared
    /// shape (`worker.rs`'s single private constructor for it, rather than
    /// each caller inlining its own `Self::new`): the hold sites read the
    /// session's phase/epoch, and — holding no claim probe — leave the
    /// session's holder exactly as they found it, sitting beside the loop's
    /// slot the way an inline `run_now` does.
    ///
    /// `birth_epoch` is THIS run's true birth snapshot — the caller's own
    /// job, not this function's: `run_claimed_job` has no earlier commit
    /// event to align to (its record is already claimed when it is called),
    /// so it reads `admission.release_epoch()` live, right before this call,
    /// matching what a bare `phase()` read would have observed before
    /// `WorkerShared` carried a birth snapshot at all. `run_placed_gang`
    /// instead reads the epoch BEFORE its own `HostAdmission::probe_claim()`
    /// call and carries that value all the way here — every byte of work
    /// after the snapshot (`probe_claim()` itself, `Catalog::transfer_claim`,
    /// `Catalog::get_job`) must be covered by the SAME birth snapshot any
    /// RELEASE landing during them has to race: because `HostAdmission::
    /// begin_release` flips the phase strictly BEFORE it bumps the epoch, a
    /// snapshot whose epoch bump is already visible implies the flip already
    /// happened too, so `probe_claim()`'s own phase check — run immediately
    /// after the snapshot — refuses it typed directly; a snapshot whose
    /// epoch bump is NOT yet visible carries no such guarantee about the
    /// phase (the flip may land at any point after the snapshot, including
    /// inside `probe_claim()`'s own check), so that case relies instead on
    /// `released_since_birth` downstream once the bump does become visible.
    /// No RELEASE lands in the gap between the two catches. The epoch must
    /// be read BEFORE `probe_claim()`, not after it: `probe_claim` commits on
    /// the phase read alone and `begin_release` bumps the epoch after
    /// flipping the phase, so a read after `probe_claim()` could snapshot an
    /// already-bumped epoch, compare the post-release epoch against itself,
    /// read `false`, and dispatch the gang on a releasing host. The
    /// flip-before-bump ORDER this relies on is pinned in
    /// `HostAdmission::begin_release`'s own doc.
    fn for_single_run(
        admission: &Arc<HostAdmission>,
        worker_id: String,
        birth_epoch: u64,
    ) -> Arc<Self> {
        Self::new(Arc::clone(admission), worker_id, birth_epoch)
    }

    /// The sampler's last snapshot (a copy).
    pub fn sample(&self) -> WorkerSample {
        self.sample
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// How many catalog samples the sampler has taken so far.
    pub fn samples_taken(&self) -> u64 {
        self.samples_taken.load(Ordering::SeqCst)
    }

    /// Fold one `count_jobs_by_kind_status` result into the snapshot,
    /// carrying every kind of the previous snapshot at 0 when absent now.
    fn record_sample(&self, rows: Vec<(String, String, i64)>) {
        let mut guard = self
            .sample
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous = std::mem::take(&mut *guard);
        let mut queued: std::collections::BTreeMap<String, i64> =
            previous.queued.into_iter().map(|(k, _)| (k, 0)).collect();
        let mut running: std::collections::BTreeMap<String, i64> =
            previous.running.into_iter().map(|(k, _)| (k, 0)).collect();
        for (kind, status, n) in rows {
            match status.as_str() {
                "queued" => {
                    queued.insert(kind, n);
                }
                "running" => {
                    running.insert(kind, n);
                }
                _ => {}
            }
        }
        *guard = WorkerSample {
            queued: queued.into_iter().collect(),
            running: running.into_iter().collect(),
        };
        drop(guard);
        self.samples_taken.fetch_add(1, Ordering::SeqCst);
    }

    /// The `claimed_by` identity of the loop this state belongs to.
    pub fn instance_id(&self) -> &str {
        &self.instance_id
    }

    /// The session's [`HostAdmission`] this loop runs under.
    pub fn admission(&self) -> &Arc<HostAdmission> {
        &self.admission
    }

    /// The current shutdown phase ([`HostAdmission::phase`]).
    pub fn phase(&self) -> WorkerPhase {
        self.admission.phase()
    }

    /// Test-only: [`HostAdmission::set_phase_for_test`] — a phase flip
    /// with `stop` deliberately left unset, the gate-direct scenario
    /// (the loop-top gate refuses a new claim on the phase read alone).
    #[cfg(feature = "test-hooks")]
    pub fn set_phase_for_test(&self, phase: WorkerPhase) {
        self.admission.set_phase_for_test(phase);
    }

    /// Whether a stop has been requested (the level the loop's pre-claim
    /// check reads).
    pub fn stop_requested(&self) -> bool {
        *self.stop.borrow()
    }

    /// Whether a RELEASE has landed on `admission` since this state was
    /// born — see the `spawn_release_epoch`
    /// field's doc for why this, rather than a bare `phase()` read, is the
    /// authoritative "have I been released" signal: `phase()` is reset to
    /// `Running` for every new generation
    /// ([`HostAdmission::try_claim_loop`]), so a STALE task whose own
    /// execution lags behind an `abort()` request could otherwise read a
    /// LATER generation's fresh `Running` phase and wrongly conclude it was
    /// never released.
    fn released_since_birth(&self) -> bool {
        self.admission.release_epoch() != self.spawn_release_epoch
    }

    /// Whether the loop may initiate a new `claim_next`: no stop has been
    /// requested, the phase is still `Running`, AND no RELEASE has landed
    /// since this state was born. `EmbeddedWorker::run_until` reads this ONE
    /// predicate at two sites — the loop's top-of-iteration gate and again,
    /// with no `.await` between that second read and the `claim_next` call
    /// itself, immediately after `reclaim_expired_jobs` returns — so the two
    /// reads can never drift apart: a
    /// RELEASE landing anywhere in the reclaim round trip is caught by the
    /// second read even when the first, now-stale read had already admitted
    /// the iteration.
    fn admits_claim(&self) -> bool {
        !self.stop_requested()
            && self.phase() == WorkerPhase::Running
            && !self.released_since_birth()
    }

    fn request_stop(&self) {
        self.stop.send_replace(true);
    }

    fn stop_receiver(&self) -> watch::Receiver<bool> {
        self.stop.subscribe()
    }

    /// The loop task's current [`LoopState`].
    pub fn loop_state(&self) -> LoopState {
        *self.state_tx.borrow()
    }

    /// A receiver on the [`LoopState`] watch — `wait_for(|s| *s !=
    /// LoopState::Running)` observes the loop's exit on every path.
    pub fn state_receiver(&self) -> watch::Receiver<LoopState> {
        self.state_tx.subscribe()
    }
}

/// Reports the loop task's terminal [`LoopState`] from inside the task, on
/// every exit path: `Stopped` when the loop returned (the guard was
/// [`Self::complete`]d), `Failed` when a panic is unwinding through it,
/// `Aborted` otherwise (the task was dropped mid-poll by an `abort()`).
struct LoopExitGuard {
    shared: Arc<WorkerShared>,
    completed: bool,
}

impl LoopExitGuard {
    fn new(shared: Arc<WorkerShared>) -> Self {
        Self {
            shared,
            completed: false,
        }
    }

    /// The loop returned cooperatively.
    fn complete(mut self) {
        self.completed = true;
    }
}

impl Drop for LoopExitGuard {
    fn drop(&mut self) {
        let state = if std::thread::panicking() {
            LoopState::Failed
        } else if self.completed {
            LoopState::Stopped
        } else {
            LoopState::Aborted
        };
        self.shared.state_tx.send_replace(state);
    }
}

/// Register a claimed job's lease hold at a loop hold site — the ONE helper
/// both sites ([`JobWorker::run_claimed_job`]'s fine-tune arm and its
/// compute arm) call, so the RELEASE check exists in exactly one place.
///
/// Registers the hold with the session's keeper, then reads whether a
/// RELEASE has landed on `shared`'s session since `shared` was born
/// ([`WorkerShared::released_since_birth`] — an
/// epoch comparison, never a bare `phase() == Releasing` read: a STALE call
/// from a task an `abort()` has not yet actually torn down could otherwise
/// observe a LATER generation's freshly-reset `Running` phase and wrongly
/// conclude it was never released): if so, the claim raced a RELEASE (it
/// committed after the keeper's per-hold pass snapshotted, or after the
/// first sweep), so this releases its OWN row through
/// `Catalog::release_job_lease` (idempotent — 0 rows if a sweep already
/// took it), drops the hold and returns `None`: the caller returns without
/// dispatching, the row is left `running` with a NULL lease for the
/// successor to claim within one idle poll, and the cap is untouched
/// (`attempts - releases` is net 0). Otherwise the holder moves `ClaimProbe
/// → JobRun` ([`HostAdmission::job_running`]) and the hold is returned; the
/// loop's [`ClaimGuard`] resets the holder to `Free` when the run returns.
///
/// An inline `run_now` registers its hold directly (`crate::jobs`) and never
/// calls this, so it never touches the holder and is never released here.
///
/// `holder` is who this attempt runs as (the module doc's writer table,
/// row 1): a `LeaseHolder`, never a rank — a rank holds no lease and
/// registers none.
async fn register_job_hold_or_release(
    session: &Arc<InferenceSession>,
    catalog: &Arc<Catalog>,
    shared: &WorkerShared,
    job_id: &str,
    attempts: u32,
    holder: LeaseHolder,
) -> Option<LeaseHold> {
    #[cfg(feature = "test-hooks")]
    loop_test_hooks::maybe_park(job_id, loop_test_hooks::ParkPoint::BeforeHold).await;
    #[cfg(feature = "test-hooks")]
    training_test_hooks::note_lease_holder(job_id, attempts, holder);
    let hold = session.lease_keeper().hold(LeaseTarget::Job {
        job_id: job_id.to_string(),
        instance_id: shared.instance_id.clone(),
        attempts,
    });
    if shared.released_since_birth() {
        match catalog
            .release_job_lease(job_id, &shared.instance_id, attempts)
            .await
        {
            Ok(true) => tracing::info!(
                job_id,
                %holder,
                "claim landed during RELEASE: lease handed back before dispatch"
            ),
            Ok(false) => tracing::debug!(
                job_id,
                "claim landed during RELEASE: lease already released by the sweep"
            ),
            Err(e) => tracing::warn!(
                job_id, error = %e,
                "claim landed during RELEASE but its release failed; left to expiry"
            ),
        }
        drop(hold);
        return None;
    }
    shared.admission.job_running();
    Some(hold)
}

/// One RELEASE sweep over this instance's rows — the two statements of
/// release steps 2c/2g in their load-bearing order: the jobs sweep
/// (`Catalog::release_jobs_claimed_by`) and then the jobs-linked building
/// sweep (`Catalog::release_building_tables_of_claimant`). `None` in a field
/// means that statement returned an error (logged; the row falls to the
/// expiry path) — distinct from `Some(0)`, a sweep that matched nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReleaseSweep {
    /// Loop-claimed `jobs` rows whose lease this sweep NULLed.
    pub jobs: Option<usize>,
    /// Linked `building` `result_tables` rows whose lease this sweep NULLed.
    pub building: Option<usize>,
}

pub(crate) async fn release_sweep(
    catalog: &Catalog,
    instance_id: &str,
    writer_id: &str,
) -> ReleaseSweep {
    let jobs = match catalog.release_jobs_claimed_by(instance_id).await {
        Ok(n) => Some(n),
        Err(e) => {
            tracing::warn!(error = %e, "RELEASE: the jobs sweep failed; leases left to expiry");
            None
        }
    };
    let building = match catalog
        .release_building_tables_of_claimant(instance_id, writer_id)
        .await
    {
        Ok(n) => Some(n),
        Err(e) => {
            tracing::warn!(
                error = %e,
                "RELEASE: the linked building-table sweep failed; the successor backs off once"
            );
            None
        }
    };
    ReleaseSweep { jobs, building }
}

/// This host's compute devices, in rank order — the `workers.devices`
/// `ListWorkers` mirror: config alone
/// decides the list, with no GPU needed to compute it. Every entry's
/// `ordinal` is [`jammi_db::config::WorkerTopology::rank_devices`]'s own
/// configured ordinal, `.max(0)` (the CPU sentinel `-1` becomes the honest
/// `0` — a `DeviceFact` ordinal is never negative); every entry's `kind` is
/// this SESSION's own [`ComputeDevice`] discriminant (one gang runs on one
/// kind of device — `GpuConfig::validate` already refuses a mixed list —
/// so the session's own backend kind applies uniformly, never re-probed
/// per ordinal). `[worker]`/`[gpu]` are validated together at
/// `InferenceSession` construction (`session.rs`'s own `worker.topology(&gpu)`
/// call), so a live session reaching the claim loop can never see the
/// `Err` arm below; it is kept typed rather than a panic so a future
/// caller that skips that validation degrades to "no devices registered"
/// instead of crashing the loop.
pub fn worker_devices(
    config: &jammi_db::config::JammiConfig,
    compute_device: ComputeDevice,
) -> Vec<DeviceFact> {
    let kind = match compute_device {
        ComputeDevice::Cpu => "cpu",
        ComputeDevice::Cuda { .. } => "cuda",
        ComputeDevice::Metal { .. } => "metal",
    };
    match config.worker.topology(&config.gpu) {
        Ok(topology) => topology
            .rank_devices()
            .iter()
            .map(|&ordinal| DeviceFact {
                kind: kind.to_string(),
                ordinal: ordinal.max(0) as u32,
            })
            .collect(),
        Err(e) => {
            tracing::error!(
                error = %e,
                "worker devices: [worker]/[gpu] topology invalid on a live session (unreachable \
                 if session construction validated it)"
            );
            Vec::new()
        }
    }
}

/// The gauge sampler: one `count_jobs_by_kind_status` per `every`, on its
/// own task — never on a `/metrics` scrape (a scrape storm must not become a
/// catalog storm) and never on the claim loop (which does not tick during a
/// run). Ends when the loop's shared state is gone.
/// Write this process's `workers` row and its registration cell as ONE fact.
/// The cell is set FIRST to the
/// facts about to be written, so a `LeaseKeeper` reregister racing this
/// write re-upserts exactly these facts and never stale ones; the
/// row is then written by [`Catalog::upsert_worker`] — an UPSERT, never a
/// bare `UPDATE` whose "zero rows matched" outcome would leave the cell
/// claiming a row that does not exist. On a failed upsert the cell is
/// REVERTED to its previous snapshot, so the cell never carries a fact no
/// row write ever succeeded with: after a failed FIRST write it is `None`
/// again (the keeper writes no row); after a failed later transition it is
/// the previous, still-true state. Every row write on the loop's lifecycle
/// (`warming`, `claiming`, `draining`) goes through here; the only other
/// row write is the delete on exit, which clears the cell first. Returns
/// whether the row write succeeded.
async fn write_worker_facts(
    catalog: &Catalog,
    registration: &InstanceRegistration,
    worker_id: &str,
    facts: WorkerFacts,
    what: &str,
) -> bool {
    let previous = registration.worker_snapshot();
    registration.set_worker(Some(facts.clone()));
    match catalog
        .upsert_worker(worker_id, &facts.kinds, facts.state, &facts.devices)
        .await
    {
        Ok(()) => true,
        Err(e) => {
            registration.set_worker(previous);
            tracing::error!(
                error = %e,
                what,
                "failed to write this process's `workers` row; the registration cell is reverted"
            );
            false
        }
    }
}

async fn sample_loop(catalog: Arc<Catalog>, shared: Weak<WorkerShared>, every: Duration) {
    loop {
        let Some(shared) = shared.upgrade() else {
            return;
        };
        match catalog.count_jobs_by_kind_status().await {
            Ok(rows) => shared.record_sample(rows),
            Err(e) => tracing::warn!(error = %e, "gauge sampler: count_jobs_by_kind_status failed"),
        }
        drop(shared);
        tokio::time::sleep(every).await;
    }
}

/// What a graceful stop found to stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopOutcome {
    /// The loop task was signalled, observed terminal, and joined; the
    /// `workers` row is deleted.
    Joined,
    /// The task had already been taken (an earlier stop or release).
    NothingToJoin,
}

/// The outcome of RELEASE step 2b: the keeper's
/// per-hold RELEASE pass (2b) either completed — carrying [`HoldRelease`]'s
/// totality-checked counts — or could not be confirmed to run at all (the
/// keeper thread was dead, or the pass did not complete within the bound).
/// The latter is UNOBSERVED, never folded into a count of zero: "no hold
/// failed" and "we don't know whether any hold failed" are different facts,
/// and collapsing them is exactly the defect this type exists to close. The
/// underlying error is logged at the call site ([`EmbeddedWorker::release_and_stop`]
/// / [`crate::session::InferenceSession::release_job_leases`]); this report
/// only needs to know the pass ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HoldReleaseOutcome {
    /// The pass completed; `Ok` from [`LeaseKeeper::release_job_holds`].
    Observed(HoldRelease),
    /// The pass could not be confirmed to run; `Err` from
    /// [`LeaseKeeper::release_job_holds`].
    Unobserved,
}

impl HoldReleaseOutcome {
    /// `true` iff the pass was observed, no hold's release attempt
    /// itself failed, AND every hold the pass started with is accounted
    /// for (`released + not_required + failed == attempted` — see
    /// [`HoldRelease::attempted`]'s doc for why this is checked here, at
    /// the consumer, rather than trusted from the pass's own internal
    /// assert alone). `false` on `Unobserved` (unobserved is not success),
    /// on `Observed` with `failed > 0`, and on an `Observed` value whose
    /// three counts do not sum to `attempted` (a hold silently dropped
    /// without being counted at all — the pass's own `assert_eq!` should
    /// already have caught this before it ever reaches a caller, but a
    /// consumer must not simply trust that).
    pub fn confirms_release(&self) -> bool {
        matches!(self, Self::Observed(hr) if hr.failed == 0
            && hr.released + hr.not_required + hr.failed == hr.attempted)
    }
}

/// What [`EmbeddedWorker::release_and_stop`] did, for the caller's log and
/// for the tests that pin each arm of the RELEASE sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReleaseReport {
    /// The loop's terminal state: `Stopped` on the cooperative arm,
    /// `Aborted` on the in-flight and timeout arms, `Failed` on a panic.
    /// May be a last-known/fallback read rather than a certain observation
    /// — see `stop_witnessed`.
    pub loop_state: LoopState,
    /// `Job` holds the keeper released (2b).
    pub holds: HoldReleaseOutcome,
    /// `true` iff no further claim by this loop can
    /// land after sweep #2 — witnessed either by 2e having resolved the
    /// task with certainty (joined, or aborted on any of its three abort
    /// arms) or by `loop_state` itself being an OBSERVED terminal state
    /// (the state-change watch fired within the bound), never the
    /// fallback/proxy read alone. A bare `loop_state != Running` comparison
    /// cannot make this distinction.
    pub stop_witnessed: bool,
    /// Sweep #1 (2c).
    pub sweep_one: ReleaseSweep,
    /// Sweep #2 (2g).
    pub sweep_two: ReleaseSweep,
}

/// Runs jobs of every kind — the three training kinds via `run_spec` AND
/// compute — from the shared catalog under a lease. Construct one per
/// process (or N for a pool); [`Self::run_until`] is the long-lived loop
/// the embedded engine and the server's worker tier both drive, spawned
/// ONLY through [`EmbeddedWorker::spawn`]/[`EmbeddedWorker::spawn_worker`]
/// (the session's single claim-loop slot) — this
/// type carries no bare, ungated entry point onto `run_until` of its own.
pub struct JobWorker {
    /// Weak back-reference to the session — upgraded each tick. `None` means the
    /// session dropped, which is the loop's exit condition (no refcycle keeps
    /// the session alive).
    session: Weak<InferenceSession>,
    /// Stable id stamped into `claimed_by` — the session's own
    /// [`InferenceSession::instance_id`], so a job this worker claims and the
    /// `instances` row the same process heartbeats share one identity.
    worker_id: String,
    /// The validated lease/heartbeat/poll timing this worker drives its loop
    /// with. `intervals.lease` is the single source of truth threaded to both
    /// the claim and the reclaim path.
    intervals: WorkerIntervals,
    /// The job kinds this worker claims (`[worker] kinds`, validated against
    /// [`COMPILED_KINDS`] by [`resolve_kinds`] at construction).
    kinds: Vec<String>,
    /// The session's [`HostAdmission`] (its own `Arc`, so holding it keeps
    /// no session alive): the phase and holder every loop this worker
    /// drives runs under.
    admission: Arc<HostAdmission>,
}

/// Re-read the node/edge sources (`GRAPH_READ_ORDER_RULE_V1`), sample
/// the graph, and materialise the pairs as a `GraphTrainingSet`-kind
/// `TrainingSet` table through the `Batches` seam —
/// the SHARED core `run_spec` (a fresh run) and
/// [`crate::pipeline::recompute`]'s `recompute_graph_training_set` (a
/// replay) both call, differing only in `inputs`: a fresh run anchors both
/// sources `UnpinnedAtInstant`, never reused; a replay re-resolves the
/// TABLE's own recorded anchor set (mirroring `recompute_training_set`'s own
/// shape). `job_id_for_hooks` labels the `test-hooks` recorders only (a
/// replay has no real job id; it passes a descriptive label instead).
///
/// # `GRAPH_READ_ORDER_RULE_V1`
///
/// Both scans below carry an explicit `ORDER BY` over the FULL projected
/// tuple, ascending, NULLS FIRST — the same shape
/// [`jammi_db::store::training_set_sort_exprs`] renders for the tabular arm's
/// own committed order. Without it the rows arrive in whatever order the
/// source's physical layout happens to hold (row-group order for Parquet,
/// file order for CSV), and `GraphSampler::sample` walks `node_ids` in
/// insertion order and each node's `out_adj` in edge-arrival order over ONE
/// seeded RNG stream — so two layouts of the identical node/edge SET would
/// sample different bytes. Ordering both scans makes "the sample is a function
/// of the input SET, not its scan order"; [`GraphSampler::build`]'s
/// duplicate-node-id refusal is the other half (a key-only order is not
/// total under a duplicate id).
#[cfg_attr(not(feature = "test-hooks"), allow(unused_variables))]
pub(crate) async fn materialize_graph_training_set(
    session: &Arc<InferenceSession>,
    job_id_for_hooks: &str,
    sources: &GraphFineTuneSources,
    sample_config: GraphSampleConfig,
    inputs: Vec<InputAnchor>,
) -> Result<jammi_db::store::TrainingSetTable> {
    let node_table = session.find_table_name(&sources.node_source).await?;
    let id_col = quote_ident(&sources.id_column);
    let text_col = quote_ident(&sources.text_column);
    let node_query = format!(
        "SELECT {id_col}, {text_col} FROM {} ORDER BY {id_col} ASC NULLS FIRST, {text_col} ASC NULLS FIRST",
        source_relation(&sources.node_source, &node_table)
    );
    let node_batches = session.sql(&node_query).await?;
    let mut nodes = Vec::new();
    for batch in &node_batches {
        let ids = batch
            .column_by_name(&sources.id_column)
            .and_then(|c| extract_string_column(c.as_ref()))
            .ok_or_else(|| {
                JammiError::FineTune(format!(
                    "node id column '{}' is not text",
                    sources.id_column
                ))
            })?;
        let texts = batch
            .column_by_name(&sources.text_column)
            .and_then(|c| extract_string_column(c.as_ref()))
            .ok_or_else(|| {
                JammiError::FineTune(format!(
                    "node text column '{}' is not text",
                    sources.text_column
                ))
            })?;
        for (id, text) in ids.into_iter().zip(texts) {
            nodes.push(TextNode::new(id, text));
        }
    }

    let edge_table = session.find_table_name(&sources.edge_source).await?;
    let src_col = quote_ident(&sources.src_column);
    let dst_col = quote_ident(&sources.dst_column);
    let edge_query = format!(
        "SELECT {src_col}, {dst_col} FROM {} ORDER BY {src_col} ASC NULLS FIRST, {dst_col} ASC NULLS FIRST",
        source_relation(&sources.edge_source, &edge_table)
    );
    let edge_batches = session.sql(&edge_query).await?;
    let mut edges = Vec::new();
    for batch in &edge_batches {
        let srcs = batch
            .column_by_name(&sources.src_column)
            .and_then(|c| extract_string_column(c.as_ref()))
            .ok_or_else(|| {
                JammiError::FineTune(format!(
                    "edge src column '{}' is not text",
                    sources.src_column
                ))
            })?;
        let dsts = batch
            .column_by_name(&sources.dst_column)
            .and_then(|c| extract_string_column(c.as_ref()))
            .ok_or_else(|| {
                JammiError::FineTune(format!(
                    "edge dst column '{}' is not text",
                    sources.dst_column
                ))
            })?;
        for (src, dst) in srcs.into_iter().zip(dsts) {
            edges.push(GraphEdge {
                src,
                dst,
                provenance: sources.provenance,
            });
        }
    }

    let sampler = GraphSampler::build(nodes, edges, sample_config)?;
    // A NAMED `MemoryConsumer` reserves the sampler's resident adjacency +
    // node-text bytes, held for THIS WHOLE FUNCTION — through the write at
    // the end, not just through sampling — via `ReservationGuard`'s `Drop`.
    // Releasing right after `sample()` would free the pool's accounting while
    // the sampler's allocation (and the batches built from it) are still
    // live, so the pool would report this job's peak bytes as free while
    // they are not. `resident_bytes` rounds up from a text-only floor by the
    // real `size_of::<String>()`/`size_of::<usize>()` per-`String`/bucket
    // overhead (that method's own doc) rather than reporting text bytes alone.
    let reservation =
        datafusion::execution::memory_pool::MemoryConsumer::new("training_set_graph_sample")
            .register(&session.memory_pool());
    let resident_bytes = sampler.resident_bytes();
    reservation
        .try_grow(resident_bytes)
        .map_err(JammiError::from)?;
    #[cfg(feature = "test-hooks")]
    training_test_hooks::note_graph_sample_reservation_bytes(job_id_for_hooks, resident_bytes);
    let _reservation_guard = ReservationGuard::new(reservation, job_id_for_hooks);

    // Materialise the sampled pairs through the SAME
    // `ResultStore::materialize_training_set` funnel the tabular arm uses,
    // via the `Batches` input seam — never a re-sample-in-place
    // `TrainingDataLoader::from_graph` (a direct, table-free constructor for
    // other callers). The format tag is the
    // CONFIG's own decision (`hard_negatives > 0`), not re-derived from any
    // particular row.
    let has_negatives = sample_config.hard_negatives > 0;
    let format_tag = crate::fine_tune::data::TrainingFormat::in_batch(has_negatives).format_tag();

    let schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("_ordinal", DataType::UInt64, false),
        Field::new("anchor", DataType::Utf8, false),
        Field::new("positive", DataType::Utf8, false),
        // Nullable: NULL for every row when `!has_negatives` (the
        // `pairs` format never carries one); the sampler's sample-time
        // refusal guarantees every row carries `Some` when `has_negatives`
        // is true, so a `None` reaching here under that config would itself
        // be an engine-invariant breach — never silently written as an
        // empty string.
        Field::new("negative", DataType::Utf8, true),
    ]));
    // 4096 rows/batch: an arbitrary, generous chunk size — no per-row
    // significance, just bounding one Arrow batch's build cost. The
    // sampler's `sample_into` streams pairs directly into THIS bounded
    // chunk buffer — no `Vec` of the whole sampled output is ever built on
    // this path (`tests::materialize_graph_training_set_streams_through_
    // sample_into_never_sample`, a source-level oracle).
    const GRAPH_BATCH_ROWS: usize = 4096;
    let mut record_batches: Vec<RecordBatch> = Vec::new();
    let mut chunk = Vec::with_capacity(GRAPH_BATCH_ROWS);
    let mut ordinal_base: u64 = 0;
    #[cfg(feature = "test-hooks")]
    let mut fingerprint_hasher = {
        use sha2::Digest;
        sha2::Sha256::new()
    };
    sampler.sample_into(|pair| {
        #[cfg(feature = "test-hooks")]
        {
            use sha2::Digest;
            fingerprint_hasher.update(pair.anchor.as_bytes());
            fingerprint_hasher.update([0u8]);
            fingerprint_hasher.update(pair.positive.as_bytes());
            fingerprint_hasher.update([0u8]);
            for negative in &pair.hard_negatives {
                fingerprint_hasher.update(negative.as_bytes());
                fingerprint_hasher.update([0u8]);
            }
            fingerprint_hasher.update([0xffu8]);
        }
        chunk.push(pair);
        if chunk.len() >= GRAPH_BATCH_ROWS {
            let batch = build_graph_training_set_batch(
                std::mem::take(&mut chunk),
                ordinal_base,
                has_negatives,
                &schema,
            )?;
            ordinal_base += batch.num_rows() as u64;
            record_batches.push(batch);
        }
        Ok(())
    })?;
    if !chunk.is_empty() {
        record_batches.push(build_graph_training_set_batch(
            chunk,
            ordinal_base,
            has_negatives,
            &schema,
        )?);
    }
    #[cfg(feature = "test-hooks")]
    {
        use sha2::Digest;
        training_test_hooks::note_graph_sample_fingerprint(
            job_id_for_hooks,
            format!("{:x}", fingerprint_hasher.finalize()),
        );
    }
    let stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        Arc::clone(&schema),
        futures::stream::iter(record_batches.into_iter().map(Ok)),
    ));

    // Exhaustive destructure, no `..` — a field
    // ADDED to `GraphSampleConfig` fails to compile HERE until it is
    // explicitly threaded into `GraphSampleFields` (moving the hash) or
    // named `_` with a reason (like `min_negatives` below), rather than
    // silently escaping the definition hash by continued dot-access on
    // only the fields this literal already named.
    let GraphSampleConfig {
        walk_length,
        walks_per_node,
        return_p,
        in_out_q,
        hard_negatives,
        exclude_hops,
        // Not output-affecting for a successful sample (`GraphTrainingSet`'s
        // own doc): the floor only gates whether sampling REFUSES, never
        // what a SUCCESSFUL sample's rows contain.
        min_negatives: _,
        seed,
    } = sample_config;
    let sample_fields = GraphSampleFields {
        seed,
        walk_length: walk_length as u64,
        walks_per_node: walks_per_node as u64,
        return_p_bits: return_p.to_bits(),
        in_out_q_bits: in_out_q.to_bits(),
        hard_negatives: hard_negatives as u64,
        exclude_hops: exclude_hops as u64,
    };
    let descriptor = ProducingDescriptor::graph_training_set(
        sources.node_source.clone(),
        sources.edge_source.clone(),
        sources.id_column.clone(),
        sources.text_column.clone(),
        sources.src_column.clone(),
        sources.dst_column.clone(),
        ModelTask::TextEmbedding,
        format_tag,
        sample_fields,
    );
    // `TrainingSetSpec::source_id` becomes part of
    // `ResultStore::create_table`'s literal table name/Parquet path
    // (`create_table`'s own doc: `"{source_id}__{task}__{model}__
    // {timestamp}_{suffix}"`, never sanitized like `model_id` is) — a
    // proper opaque lineage identifier, NOT a display sentence a human
    // reads. A space/`=`-bearing sentence ("graph node=X edge=Y") would
    // land verbatim in a file path and a SQL-registered relation name; a
    // stable `graph__node-X__edge-Y` shape names the SAME two real
    // sources (never a fabricated query string — there is none; the graph
    // arm's row source is a biased walk, not a projection) with no
    // structural character a caller reading the resulting table name would
    // have to escape.
    let source_display = format!(
        "graph__node-{}__edge-{}",
        sources.node_source, sources.edge_source
    );
    let order_columns = vec!["_ordinal".to_string()];
    let spec = TrainingSetSpec {
        source_id: &source_display,
        input: TrainingSetInput::Batches {
            schema: Arc::clone(&schema),
            stream,
        },
        columns: &order_columns,
        task: ModelTask::TextEmbedding,
        descriptor,
        inputs,
        device: session.compute_device(),
    };
    let table = session
        .result_store()
        .materialize_training_set(session.context(), spec)
        .await?;
    #[cfg(feature = "test-hooks")]
    training_test_hooks::note_graph_sample_write_committed(job_id_for_hooks);
    // `_reservation_guard` drops HERE, at the end of this function's own
    // scope — strictly after the write above committed, never before.
    Ok(table)
}

/// Build one `_ordinal`/`anchor`/`positive`/`negative` `RecordBatch` from a
/// BOUNDED chunk of sampled pairs — moves each `String`
/// out of `chunk` (`into_iter`, never `.clone()`), so building a batch never
/// doubles the chunk's own residency. `ordinal_base` is the running row
/// count already written by prior chunks (the leading `_ordinal` column is
/// GLOBAL across the whole output, not chunk-local).
fn build_graph_training_set_batch(
    chunk: Vec<SampledPair>,
    ordinal_base: u64,
    has_negatives: bool,
    schema: &SchemaRef,
) -> Result<RecordBatch> {
    let ordinals: ArrayRef = Arc::new(UInt64Array::from(
        (0..chunk.len() as u64)
            .map(|i| ordinal_base + i)
            .collect::<Vec<_>>(),
    ));
    let mut anchors = Vec::with_capacity(chunk.len());
    let mut positives = Vec::with_capacity(chunk.len());
    let mut negatives: Vec<Option<String>> = Vec::with_capacity(chunk.len());
    for pair in chunk {
        anchors.push(pair.anchor);
        positives.push(pair.positive);
        negatives.push(if has_negatives {
            pair.hard_negatives.into_iter().next()
        } else {
            None
        });
    }
    let anchors: ArrayRef = Arc::new(StringArray::from(anchors));
    let positives: ArrayRef = Arc::new(StringArray::from(positives));
    let negatives: ArrayRef = Arc::new(StringArray::from(negatives));
    RecordBatch::try_new(
        Arc::clone(schema),
        vec![ordinals, anchors, positives, negatives],
    )
    .map_err(|e| JammiError::FineTune(format!("failed to build graph training-set batch: {e}")))
}

/// Holds a [`datafusion::execution::memory_pool::MemoryReservation`] and
/// releases it on `Drop` — the reservation's release always happens at ITS
/// OWN scope's natural end, never an explicit early `drop(reservation)`
/// call whose placement a reviewer has to trust matches the write's real
/// completion.
/// Under `test-hooks`, dropping also records a release event so the
/// release-timing oracle can assert, from execution, that release happened
/// AFTER the write committed — not by reading the source and trusting the
/// ordering by eye.
struct ReservationGuard {
    reservation: Option<datafusion::execution::memory_pool::MemoryReservation>,
    #[cfg(feature = "test-hooks")]
    job_id: String,
}

impl ReservationGuard {
    fn new(
        reservation: datafusion::execution::memory_pool::MemoryReservation,
        #[cfg_attr(not(feature = "test-hooks"), allow(unused_variables))] job_id: &str,
    ) -> Self {
        Self {
            reservation: Some(reservation),
            #[cfg(feature = "test-hooks")]
            job_id: job_id.to_string(),
        }
    }
}

impl Drop for ReservationGuard {
    fn drop(&mut self) {
        self.reservation.take();
        #[cfg(feature = "test-hooks")]
        training_test_hooks::note_graph_sample_reservation_released(&self.job_id);
    }
}

impl JobWorker {
    /// Build a worker over a session, reading its lease/heartbeat timing from
    /// the session's `[lease]` configuration and its idle poll + kind
    /// selection from `[worker]`. The worker holds a
    /// [`Weak`] so it never keeps the session alive; the caller owns the strong
    /// `Arc` and the worker stops when that drops.
    ///
    /// Returns [`JammiError::Config`] if the configured timing violates the
    /// worker invariants (heartbeat margin / non-zero poll), or if `[worker]
    /// kinds` names a kind this binary does not compile. In the normal flow
    /// the timing check already ran at config load, so that half only fires
    /// for a programmatically built config that bypassed `JammiConfig::load`.
    pub fn new(session: &Arc<InferenceSession>) -> Result<Self> {
        let config = session.inner_config();
        let intervals = config.worker.worker_intervals(config.lease.intervals()?)?;
        let kinds = resolve_kinds(&config.worker.kinds)?;
        Ok(Self::with_intervals_and_kinds(session, intervals, kinds))
    }

    /// Build a worker over a session with explicit, already-validated timing
    /// and kind selection.
    ///
    /// The worker's `claimed_by` identity is the session's own
    /// [`InferenceSession::instance_id`] — never independently re-derived —
    /// so the same process's `instances` row and every job it claims agree on
    /// one id.
    pub fn with_intervals_and_kinds(
        session: &Arc<InferenceSession>,
        intervals: WorkerIntervals,
        kinds: Vec<String>,
    ) -> Self {
        Self {
            session: Arc::downgrade(session),
            worker_id: session.instance_id().to_string(),
            intervals,
            kinds,
            admission: Arc::clone(session.host_admission()),
        }
    }

    /// Build a worker over a session with explicit, already-validated timing
    /// and every compiled kind. Kept for callers that do not need to select a
    /// kind subset (most tests; a single-process deployment).
    pub fn with_intervals(session: &Arc<InferenceSession>, intervals: WorkerIntervals) -> Self {
        Self::with_intervals_and_kinds(
            session,
            intervals,
            COMPILED_KINDS.iter().map(|s| s.to_string()).collect(),
        )
    }

    /// The worker's stable id (`claimed_by` value). Exposed for tests that assert
    /// on lease ownership.
    pub fn worker_id(&self) -> &str {
        &self.worker_id
    }

    /// Run the claim→reconstruct→train loop until `shared`'s stop is set,
    /// `shared`'s phase leaves [`WorkerPhase::Running`], or the session
    /// drops.
    ///
    /// Stack-safe: a bounded `loop`, never recursion. The task's FIRST
    /// statement upserts this process's `workers` row as `warming`; it then
    /// waits on the session's worker gate (`InferenceSession::open_worker_gate`,
    /// open by default — a server closes it while it preloads models)
    /// selected against the stop, flips the row to `claiming` when the gate
    /// opens, and only then claims — one sequential chain, so `claiming` can
    /// never precede `warming` and a stop during the wait returns without a
    /// claim. `WorkerShared::admits_claim` (one private predicate, two
    /// independent signals — `stop_requested()`, the wakeup that also
    /// interrupts an idle sleep, and `phase() == Running`) is read at TWO
    /// sites before any `claim_next`: the top of the iteration, and again,
    /// with no `.await` between that second read and `claim_next` itself,
    /// immediately after `reclaim_expired_jobs` returns — so a `Releasing`
    /// (or `Draining`) phase that lands during the reclaim round trip is
    /// caught by the second read even though the first, now-stale read had
    /// already admitted the iteration; a
    /// loop already at either read point never starts a `claim_next`
    /// regardless of which of the two signals it observes first. The one
    /// residual is a claim whose own catalog round trip is already in flight
    /// when the phase flips: `register_job_hold_or_release`'s self-release
    /// arm tests `== Releasing` only, so under `Releasing` that claim
    /// self-releases, and under `Draining` it dispatches normally. On a claim it runs the
    /// job to a terminal state inline (the next claim waits for it), on no
    /// claim it sleeps the configured idle poll `select!`ed against the stop
    /// watch (level-triggered: no lost wakeup, no waiting out the poll). The
    /// catalog used for reclaim/claim is unscoped — a worker serves every
    /// tenant's queue. The terminal [`LoopState`] is written on every exit
    /// path by an in-task guard.
    pub async fn run_until(&self, shared: Arc<WorkerShared>) {
        let exit = LoopExitGuard::new(Arc::clone(&shared));
        let mut stop_rx = shared.stop_receiver();

        // The `workers` row, first: another process's `ListWorkers` sees
        // this loop from the moment it exists, and never `claiming` before.
        let Some(session) = self.session.upgrade() else {
            exit.complete();
            return;
        };
        // The row and the registration's worker cell are written as ONE
        // fact through `write_worker_facts`: a failed write leaves the cell exactly as it was
        // (`None` here), so a keeper reregister racing a still-failing loop
        // start never writes a `workers` row this loop never managed to
        // write itself.
        write_worker_facts(
            session.catalog(),
            session.instance_registration(),
            &self.worker_id,
            WorkerFacts {
                kinds: self.kinds.join(","),
                state: WorkerState::Warming,
                devices: worker_devices(session.inner_config(), session.compute_device()),
            },
            "warming",
        )
        .await;
        let mut gate_rx = session.worker_gate_receiver();
        drop(session);

        // Warm-before-claim: wait for the gate, or for a stop — whichever
        // comes first. A closed gate whose sender is gone (the session
        // dropped) can never open.
        let gate_open = tokio::select! {
            opened = gate_rx.wait_for(|open| *open) => opened.is_ok(),
            _ = stop_rx.wait_for(|stop| *stop) => false,
        };
        if !gate_open {
            exit.complete();
            return;
        }
        if let Some(session) = self.session.upgrade() {
            // The `claiming` transition is the same one-fact write as the
            // first `warming` write above: an UPSERT (so a row the first
            // write failed to create is created here, never a bare UPDATE
            // whose "zero rows" outcome the loop could not act on), with the
            // cell reverted on failure.
            write_worker_facts(
                session.catalog(),
                session.instance_registration(),
                &self.worker_id,
                WorkerFacts {
                    kinds: self.kinds.join(","),
                    state: WorkerState::Claiming,
                    devices: worker_devices(session.inner_config(), session.compute_device()),
                },
                "claiming",
            )
            .await;
        }

        loop {
            #[cfg(feature = "test-hooks")]
            loop_test_hooks::maybe_panic(&self.worker_id);
            // `admits_claim()` bundles two independent signals:
            // `stop_requested()` is the wakeup (it also interrupts an idle
            // sleep, see the `None` arm's `select!` below); `phase() !=
            // Running` refuses a new claim the instant RELEASE (or DRAIN)
            // flips the phase, even in the window before the stop watch is
            // next polled — see `EmbeddedWorker::release_and_stop`'s 2a. Read
            // again below, after `reclaim_expired_jobs`, so a RELEASE landing
            // during that round trip cannot ride this now-stale read into a
            // claim.
            if !shared.admits_claim() {
                break;
            }
            let session = match self.session.upgrade() {
                Some(s) => s,
                // The session dropped: nothing more to serve, exit the loop.
                None => break,
            };
            let catalog = session.catalog();

            if let Err(e) = catalog
                .reclaim_expired_jobs(self.intervals.lease, MAX_ATTEMPTS)
                .await
            {
                tracing::error!(worker = %self.worker_id, error = %e, "reclaim_expired_jobs failed");
            }

            #[cfg(feature = "test-hooks")]
            loop_test_hooks::maybe_park_after_reclaim(&self.worker_id).await;

            // The second read: no `.await` between this and `claim_next`
            // itself (`record_claim_next` is sync). A claim whose own
            // catalog round trip is already in flight when 2a runs is the
            // one residual neither read catches — under `Releasing` (`:522`)
            // `register_job_hold_or_release` self-releases it; under `Draining` it dispatches and runs to completion.
            if !shared.admits_claim() {
                break;
            }
            // The slot: `Free → ClaimProbe` before `claim_next` (a peer never
            // claims while it holds a rank); the guard resets the
            // holder to `Free` on every exit of this iteration. A held slot
            // (an admitted rank, or a job already running) skips the claim
            // and sleeps the idle poll exactly like a claim that found
            // nothing.
            let Some(claim) = shared.admission.probe_claim() else {
                tokio::select! {
                    _ = tokio::time::sleep(self.intervals.idle_poll) => {}
                    _ = stop_rx.wait_for(|stop| *stop) => {}
                }
                continue;
            };
            let kind_refs: Vec<&str> = self.kinds.iter().map(String::as_str).collect();
            #[cfg(feature = "test-hooks")]
            loop_test_hooks::record_claim_next(&self.worker_id);
            let claimed = match catalog
                .claim_next(&self.worker_id, &kind_refs, self.intervals.lease)
                .await
            {
                Ok(c) => c,
                Err(e) => {
                    tracing::error!(worker = %self.worker_id, error = %e, "claim_next failed");
                    None
                }
            };

            match claimed {
                Some(record) => {
                    // Drop the session strong ref before the (possibly long) run
                    // so the worker does not pin the session for the whole job —
                    // the run re-upgrades the Weak through the `Arc` it captures.
                    // The probe guard lives across the run: `ClaimProbe → JobRun`
                    // at the hold site, `→ Free` here when the run returns.
                    self.run_claimed_job_under(&session, record, &shared, false)
                        .await;
                    drop(claim);
                }
                None => {
                    drop(claim);
                    // Interruptible: a stop set mid-sleep wakes the loop now,
                    // not up to `idle_poll` later; `wait_for` is level-
                    // triggered so a stop sent before this poll resolves too.
                    tokio::select! {
                        _ = tokio::time::sleep(self.intervals.idle_poll) => {}
                        _ = stop_rx.wait_for(|stop| *stop) => {}
                    }
                }
            }
        }
        exit.complete();
    }

    /// Run one already-claimed job to a terminal state. Deserialises the spec,
    /// pins the catalog to the job's tenant (the claim is intentionally unscoped,
    /// so the worker's writes must be re-scoped) and runs the kind's
    /// reconstruction under that tenant's scope and a heartbeat — every catalog
    /// read and SQL-surface read inside the run observes the job's tenant, not
    /// the worker session's unbound default — then performs the single
    /// lease-guarded finalize —
    /// `completed` + the output model when this worker still holds the lease, or
    /// `failed` + the error on a genuine failure. A worker that lost its lease in
    /// the run window does not finalize; the job is left for `reclaim`.
    ///
    /// `record` must be a row this worker claimed (its `claimed_by` is the
    /// worker's id). The driving loop ([`Self::run_until`]) is the normal caller;
    /// it is exposed so a test can drive one claimed job in isolation.
    ///
    /// # Every failure path between the claim and the acceleration probe
    ///
    /// The acceleration report's tri-state `{"state":"pending"}` marker means "no
    /// claimant has computed a determination YET", so it must not survive onto
    /// a row that has gone terminal. `Catalog::fail_job` retires a
    /// still-`pending` report to
    /// `{"state":"undetermined","reason":"failed_before_probe"}` in the SAME
    /// lease-guarded UPDATE (see its own doc) — which covers this function's
    /// failure paths EXACTLY as long as each one goes through
    /// `record_failed`. Each one does. Enumerated, so a new path that
    /// bypasses it is visibly outside this list rather than silently uncovered:
    ///
    /// | # | failure | terminal write |
    /// |---|---|---|
    /// | 1 | no `training_spec` at all | `mark_acceleration_undetermined` (a MORE specific `failed_before_device_resolution` reason, which the catalog edge preserves) then `record_failed` |
    /// | 2 | undeserialisable `training_spec` | same as 1 |
    /// | 3 | training-set materialization / loader reconstruction error (`training_set::materialize_projection`, `build_training_data_loader`, `materialize_graph_training_set`) | `Err(Failed)` → `record_failed` |
    /// | 4 | base-model load error, incl. a missing artifact (`model_cache().get_or_load`) | `Err(Failed)` → `record_failed` |
    /// | 5 | base model exposes no embedding dim | `Err(Failed)` → `record_failed` |
    /// | 6 | device-select error (`select_device`, inside `run_fine_tune_blocking` — BEFORE the probe) | `Err(Failed)` → `record_failed` |
    /// | 7 | head/adapter construction error (`build_classification_head`, `build_distribution_head`, `build_encoder_adapters`, incl. `validate_backbone_precision`) — also before the probe | `Err(Failed)` → `record_failed` |
    /// | 8 | typed training failure after the probe | `Err(Failed)` → `record_failed` (report is already `determined`) |
    /// | 9 | `spawn_blocking` panic (caught) or join error | `Err(Failed)` → `record_failed` |
    /// | 10 | `publish_and_finalize` giving up before its finalize (`fail_before_finalize`) | `record_failed` |
    /// | 11 | `Err(Cancelled)` where `spawn_cancel_request_watcher` set `cancel_requested_seen` (a `CancelJob`/`JobHandle::cancel` request, not a lease loss) | `Err(Cancelled)` + `cancel_requested_seen` → `record_failed` with [`jammi_db::error::JammiError::JobCancelled`]'s message |
    ///
    /// Row 11's window has an edge `Ok(artifact)` never closes: a request
    /// observed only AFTER the training loop's last epoch-boundary check
    /// (the run already has a finished artifact by the time the watcher
    /// flips `cancel_requested_seen`) lands `completed`, not `failed` — the
    /// `Ok(artifact)` arm below does not consult that flag at all. See the
    /// module doc's "cancel observed after the last epoch boundary" note;
    /// this is the same single-shot-producer convention [`crate::jobs`]
    /// documents for the compute path, not a bug this table's `record_failed`
    /// column implies row 11 always wins.
    ///
    /// The deliberate exception is `Err(WorkerJobError::Cancelled)` on a
    /// genuine lease loss (row 11 above is the OTHER half — `Cancelled` has
    /// two distinguishable causes, each with its own terminal-write rule):
    /// when `spawn_cancel_request_watcher` never saw `cancel_requested`
    /// (the lease was lost, or a genuine error coincided with a lost lease —
    /// see `classify`), this writes NO terminal status, because a different
    /// worker now owns the job. Its `pending` marker is retired by the OTHER
    /// half of the same catalog-edge rule —
    /// `Catalog::reclaim_expired_jobs`' exhausted arm writes
    /// `{"state":"undetermined","reason":"lease_expired_attempts_exhausted"}`,
    /// and its requeue arm RESETS the column to `pending` for the fresh
    /// attempt that will re-probe. Adding a `record_failed` here would be
    /// wrong twice over: it would stamp `failed` over a job the re-claiming
    /// worker is running, and its lease guard would not match anyway. A
    /// `Cancelled` the watcher DID attribute to a cancel request is not this
    /// case: no other worker is coming to reclaim it, so it takes row 11's
    /// `record_failed` instead, which retires the same `pending` marker
    /// through the identical lease-guarded edge every other `record_failed`
    /// call in this table does.
    ///
    /// `Self::publish_and_finalize`'s own `finish_job_with_model`
    /// `Ok(false)`/`Err` arms likewise leave the job `running` for reclaim, so
    /// no terminal status is written on them either and the same reclaim half
    /// of the catalog-edge rule applies.
    ///
    /// # The SUCCESS path is not exempt
    ///
    /// A post-probe path's report is not guaranteed to be `determined`:
    /// `persist_acceleration_report` deliberately SWALLOWS a
    /// lease-guard miss (`Ok(false)`, e.g. a stale `attempt`) and a catalog
    /// error, by design — the write not landing must never fail training. A
    /// job whose probe write was swallowed and which then finalizes
    /// successfully reaches `completed` with the submission-time `pending`
    /// marker still on the row, which is the SAME forbidden state the failure
    /// paths above avoid, by a success path. It is covered at the same ONE
    /// catalog edge: `Catalog::finish_job_with_model` retires a still-`pending`
    /// report to `{"state":"undetermined","reason":
    /// "finalized_without_determination"}` in the SAME CAS that stamps
    /// `completed`, and preserves any already-`determined` payload
    /// byte-for-byte. `crates/jammi-ai/tests/it/acceleration_report.rs`'s
    /// `completed_job_with_a_swallowed_report_write_is_never_left_pending`
    /// drives both legs.
    #[tracing::instrument(
        skip(self, session, record),
        fields(
            worker_id = %self.worker_id,
            job_id = %record.job_id,
            tenant_id = ?record.tenant_id,
        )
    )]
    pub async fn run_claimed_job(
        &self,
        session: &Arc<InferenceSession>,
        record: jammi_db::catalog::jobs_repo::JobRecord,
    ) {
        // A caller driving one claimed job outside a loop task runs it under
        // fresh shared state over the session's own admission — see
        // `WorkerShared::for_single_run`'s own doc. No earlier commit event
        // to align to here, so the birth epoch is read live, right before
        // the call.
        let shared = WorkerShared::for_single_run(
            &self.admission,
            self.worker_id.clone(),
            self.admission.release_epoch(),
        );
        self.run_claimed_job_under(session, record, &shared, false)
            .await;
    }

    /// [`Self::run_claimed_job`] under the loop's [`WorkerShared`]: the two
    /// hold sites register through [`register_job_hold_or_release`] against
    /// `shared`'s phase and account the job in `shared.in_flight`.
    ///
    /// `placed = true` is the ONE recursion guard: a run [`Self::run_placed_gang`] is
    /// already coordinating on THIS process never re-checks the placement
    /// seam, however many gang listeners this process happens to host —
    /// every OTHER caller (the claim loop, [`Self::run_claimed_job`]) passes
    /// `false`.
    async fn run_claimed_job_under(
        &self,
        session: &Arc<InferenceSession>,
        record: jammi_db::catalog::jobs_repo::JobRecord,
        shared: &Arc<WorkerShared>,
        placed: bool,
    ) -> AttemptEnd {
        let job_id = record.job_id.clone();
        // The attempt counter makes the artifact prefix unique per (job, worker,
        // attempt): a reclaimed job re-runs under a higher `attempts`, so its
        // new attempt writes to a fresh prefix and never overwrites the prior
        // attempt's objects.
        let attempt = record.attempts;
        let catalog = Arc::new(session.catalog().pinned_to_tenant(record.tenant_id));

        if is_compute_kind(&record.kind) {
            // Unreachable for `placed`: a `GangDescriptor` only ever names a
            // `fine_tune`/`graph_fine_tune` attempt (the placement check
            // below is the only producer of one) — a compute kind never
            // reaches `run_placed_gang`.
            self.run_claimed_compute_job(
                session,
                &catalog,
                shared,
                &job_id,
                &record.spec,
                attempt,
                record.partial_result.as_deref(),
                record.tenant_id,
            )
            .await;
            return AttemptEnd::LeftForReclaim;
        }

        // `crate::jobs::JobSpec` is the one type every persisted `jobs.spec`
        // row decodes as (that type's own doc); this loop-claimer path
        // projects the decoded value to `TrainingSpec` with
        // `JobSpec::as_training_spec` rather than decoding `TrainingSpec`
        // directly, so a stray field anywhere in the row — including inside
        // a nested config struct, every one of which denies unknown
        // fields too — is caught at the SAME decode `JobSpec`'s own byte-pin
        // tests exercise, not a second, independent one that could drift.
        let job_spec: crate::jobs::JobSpec = match serde_json::from_str(&record.spec) {
            Ok(s) => s,
            Err(e) => {
                // This fails BEFORE the
                // device is ever resolved, so `run_fine_tune_blocking`'s
                // measuring probe never runs — write the honest terminal
                // marker first (still `running`, satisfying the lease guard)
                // so the record never reads `{"state":"pending"}` past this
                // job's `failed` status below.
                mark_acceleration_undetermined(
                    LeaseHolder::LoopClaimer,
                    &catalog,
                    &job_id,
                    &self.worker_id,
                    attempt,
                )
                .await;
                let reason = format!("undeserialisable training_spec: {e}");
                record_failed(
                    LeaseHolder::LoopClaimer,
                    &catalog,
                    &job_id,
                    &self.worker_id,
                    attempt,
                    reason.clone(),
                )
                .await;
                return AttemptEnd::Failed {
                    error: JammiError::FineTune(reason),
                };
            }
        };
        let Some(spec) = job_spec.as_training_spec() else {
            // `record.kind` (the `jobs.kind` column) named a training kind,
            // routing this attempt here at all (the `is_compute_kind` guard
            // above), but the persisted `spec` JSON's own `kind` decoded to
            // a compute variant — the two columns disagree. Same failure
            // shape as an undeserialisable spec: nothing has run yet.
            mark_acceleration_undetermined(
                LeaseHolder::LoopClaimer,
                &catalog,
                &job_id,
                &self.worker_id,
                attempt,
            )
            .await;
            let reason = format!(
                "training claim path: jobs.kind names a training kind but the persisted spec \
                 decoded as compute kind {}",
                job_spec.kind()
            );
            record_failed(
                LeaseHolder::LoopClaimer,
                &catalog,
                &job_id,
                &self.worker_id,
                attempt,
                reason.clone(),
            )
            .await;
            return AttemptEnd::Failed {
                error: JammiError::FineTune(reason),
            };
        };
        // Who this attempt runs as — derived ONCE from the spec and this
        // host's `[worker] local_ranks` (the module doc's writer table), and
        // threaded to every job-row-writing site below.
        let holder = lease_holder_for(&spec, session.inner_config().worker.local_ranks);
        // The job's lease is a keeper hold, not a `tokio::spawn`
        // heartbeat task — the dedicated keeper thread renews it, immune to
        // this runtime being starved by CPU-bound training. `cancel` IS the
        // hold's own `lost` flag (identity, not a poll copy): the
        // keeper flips it directly on the next renewal that misses, and both
        // training paths' epoch-boundary checks read it. `hold` must outlive the
        // run (held below) — dropping it early would stop renewal. Registered
        // through the one RELEASE-aware helper: a claim that landed during a
        // RELEASE hands its lease straight back and never dispatches.
        let Some(hold) =
            register_job_hold_or_release(session, &catalog, shared, &job_id, attempt, holder).await
        else {
            return AttemptEnd::LeftForReclaim;
        };
        let cancel = hold.lost_flag();

        // `cancel` (the lease-lost flag above) is not the ONLY source
        // that must be able to trip the training loop's epoch-boundary
        // check — a `CancelJob`/`JobHandle::cancel` request sets
        // `jobs.cancel_requested`, which only the compute path's
        // `check_cancel` reads; the training loop needs the same signal.
        // This watcher polls that column at the SAME cadence the lease keeper
        // renews at (`self.intervals.heartbeat` — never per-step, staying
        // out of the hot loop) and, on an observed request, flips the
        // identical `cancel` flag so the trainer's existing epoch-boundary
        // check (no new check needed there) bails exactly as it does on a
        // lease loss. `cancel_requested_seen` is a SEPARATE one-way flag the
        // watcher sets first, so the match below can tell "the flag tripped
        // because of a request" apart from "the flag tripped because the
        // lease was lost" and land the right terminal write for each.
        let cancel_requested_seen = Arc::new(AtomicBool::new(false));
        // `true` for the whole attempt, flipped `false` by
        // `CancelWatcherGuard::drop` — the watcher's own belt-and-braces
        // check, independent of `abort()`'s cooperative cancellation (which
        // only takes effect at the watcher's own next `.await` point). See
        // `CancelWatcherGuard`'s doc for why a bare `JoinHandle` is not
        // enough here.
        let attempt_alive = Arc::new(AtomicBool::new(true));
        let cancel_watcher = spawn_cancel_request_watcher(
            Arc::clone(&catalog),
            job_id.clone(),
            Arc::clone(&cancel),
            Arc::clone(&cancel_requested_seen),
            attempt_alive,
            self.intervals.heartbeat,
        );
        #[cfg(feature = "test-hooks")]
        training_test_hooks::record_watcher(&job_id, cancel_watcher.abort_handle(), &catalog);

        // Run the whole job in its own tenant scope. The claim is intentionally
        // unscoped (one worker drains every tenant's queue), so inside the run
        // the session's tenant binding is `None` — and the reconstruction's
        // catalog reads (`resolve_embedding_table`) and SQL-surface reads
        // (`assemble_context`, the per-member vector reads) would otherwise
        // resolve `Unscoped` and miss a tenant's rows. The session shares one
        // `TenantBinding` between its catalog and its DataFusion analyzer rule,
        // so installing the job's tenant as the task-local override for the
        // duration of the run makes every async read and write observe it.
        //
        // The write path additionally uses the sticky `pinned_to_tenant`
        // catalog (above) because a fine-tune's artifact staging and
        // `get_model` run inside (or after) a `spawn_blocking` thread, which
        // does not inherit the task-local; the predictor's async reads are
        // covered by this scope.
        // The training-set identity pair a PRIOR attempt of this job
        // recorded (write-once, `materialize_or_reuse_training_set`): a
        // retry binds THAT table rather than materializing a fresh one, so
        // the job's identity — what every member was and will be admitted
        // against — never moves under a retry. `None` for every job no
        // coordinator has run yet (and for every single-rank job).
        let recorded_pair = match (record.training_set_ref, record.training_set_location) {
            (Some(training_set_ref), Some(training_set_location)) => {
                Some(TrainingSetIdentityPair {
                    training_set_ref,
                    training_set_location,
                })
            }
            _ => None,
        };

        // Placement is decided BEFORE topology and applies to every
        // FineTune/GraphFineTune attempt this run is not itself placed — never a
        // `ContextPredictor`, which carries no `world_size`/gang concept at
        // all. `world` travels as informational only: the executor decides
        // ITS OWN topology from the spec's `world_size` and its OWN
        // `[worker] local_ranks` when it runs `run_claimed_job_under`
        // itself, so no materialization/loader work happens here for a
        // placed attempt — it happens once, on whichever process actually
        // trains.
        let placement_world = if placed {
            None
        } else {
            match &spec {
                TrainingSpec::FineTune { common, .. } => Some(common.world_size),
                TrainingSpec::GraphFineTune { common, .. } => Some(common.world_size),
                TrainingSpec::ContextPredictor { .. } => None,
            }
        };
        let placement = placement_world.and_then(|world| {
            session
                .host_admission()
                .placed_gang_submitter()
                .filter(|submitter| submitter.placement_available())
                .map(|submitter| (world, submitter))
        });

        let outcome = if let Some((world, submitter)) = placement {
            self.submit_placed(session, &catalog, &job_id, attempt, world, submitter)
                .await
        } else {
            match record.tenant_id {
                Some(tenant) => {
                    let recorded_pair = recorded_pair.clone();
                    session
                        .with_tenant_scoped(tenant, |_scope| {
                            self.run_spec(
                                session,
                                &catalog,
                                &job_id,
                                spec,
                                &cancel,
                                attempt,
                                recorded_pair,
                                holder,
                            )
                        })
                        .await
                }
                None => {
                    self.run_spec(
                        session,
                        &catalog,
                        &job_id,
                        spec,
                        &cancel,
                        attempt,
                        recorded_pair,
                        holder,
                    )
                    .await
                }
            }
        };

        // Stop renewing this attempt's lease regardless of outcome — the
        // job is about to reach a terminal write (or be left for reclaim),
        // so no further renewal is wanted either way. The watcher is stopped
        // alongside it: `CancelWatcherGuard::drop` aborting an
        // already-finished task is a harmless no-op, and there is nothing
        // left for it to watch once the run has returned. This explicit
        // drop is the ordinary exit's path through the SAME `Drop` impl
        // that also covers the extraordinary ones (a panic unwinding through
        // this scope, or this whole `.await` being dropped out from under
        // it by a caller aborting the task).
        drop(hold);
        drop(cancel_watcher);

        match outcome {
            Ok(AttemptOutput::Reused(reused)) => {
                // The job is already `completed` (the reuse probe's own
                // transaction wrote the terminal row and the output model's
                // reference). This attempt staged nothing of its own, but
                // the job's durable resume checkpoint from an earlier
                // attempt has no live stager left, exactly as after a won
                // finalize: the same sweep and the same reclaim.
                tracing::info!(
                    job_id = %job_id, worker = %self.worker_id, model_id = %reused.model_id,
                    artifact = %reused.artifact, "training job completed by reuse"
                );
                let store = session.artifact_store();
                reclaim_unpublished_artifacts(&store, &catalog, &job_id, &self.worker_id, attempt)
                    .await;
                reclaim_checkpoints(&store, &catalog, &job_id).await;
                AttemptEnd::Reused
            }
            Ok(AttemptOutput::Trained(artifact)) => {
                // Computed BEFORE the artifact's directory is handed to
                // `publish_and_finalize` (which consumes it) — the SAME
                // bytes a member/an in-process `Peer` rank digests, so
                // a placed run's `PlacedOutcome::Trained` carries an
                // identical digest without a second row read.
                let digest = adapter_files_digest(artifact.dir.path());
                match self
                    .publish_and_finalize(holder, session, &catalog, &job_id, attempt, *artifact)
                    .await
                {
                    PublishOutcome::Completed => match digest {
                        Ok(artifact_digest) => AttemptEnd::Published { artifact_digest },
                        Err(e) => {
                            // The publish itself already read every file in
                            // the SAME directory successfully (it just
                            // committed `completed`) — a digest re-read
                            // failing here is an I/O fault in the narrow
                            // window between those two reads, not a
                            // training/publish failure; the row IS
                            // genuinely `completed`. UNCOVERED: no fixture
                            // manufactures this race.
                            tracing::error!(
                                job_id = %job_id, worker = %self.worker_id, error = %e,
                                "completed job's artifact digest could not be re-read"
                            );
                            AttemptEnd::Failed {
                                error: JammiError::FineTune(format!(
                                    "artifact published but its digest could not be re-read: {e}"
                                )),
                            }
                        }
                    },
                    PublishOutcome::Failed(reason) => AttemptEnd::Failed {
                        error: JammiError::FineTune(reason),
                    },
                    PublishOutcome::LeftForReclaim => AttemptEnd::LeftForReclaim,
                }
            }
            Err(WorkerJobError::Cancelled) => {
                // No `TrainedArtifact` was ever built on this path (the run
                // bailed mid-training, or never even finished the blocking
                // call), so any epoch checkpoints this attempt staged are
                // reachable only through the catalog — which is where the
                // sweep reads them from. Reclaimed on both the lease-lost
                // and the cancel-requested arm below.
                reclaim_unpublished_artifacts(
                    &session.artifact_store(),
                    &catalog,
                    &job_id,
                    &self.worker_id,
                    attempt,
                )
                .await;
                if cancel_requested_seen.load(Ordering::SeqCst) {
                    // The flag tripped because `spawn_cancel_request_
                    // watcher` observed `jobs.cancel_requested`, not because
                    // the lease was lost — this run is not going to be
                    // reclaimed and retried, so it must land terminal here,
                    // as `cancelled` — the same end the compute path and
                    // `run_now` record for a request honoured at their own
                    // checkpoints — never the lease-lost log line below.
                    tracing::warn!(job_id = %job_id, worker = %self.worker_id, "training cancelled (cancel requested); recording cancelled");
                    record_unsuccessful_end(
                        holder,
                        &catalog,
                        &job_id,
                        &self.worker_id,
                        attempt,
                        UnsuccessfulEnd::Cancelled,
                    )
                    .await;
                    AttemptEnd::Failed {
                        error: JammiError::JobCancelled {
                            job_id: job_id.clone(),
                        },
                    }
                } else {
                    // Lease lost: leave the job `running` for reclaim to
                    // re-queue. Do not record a terminal status — a
                    // different worker now owns, or will own, this job.
                    tracing::warn!(job_id = %job_id, worker = %self.worker_id, "training cancelled (lease lost); left for reclaim");
                    AttemptEnd::LeftForReclaim
                }
            }
            Err(WorkerJobError::Abandoned(why)) => {
                // The coordinator body already recorded the attempt's
                // assembly outcome and settled the lease; nothing terminal
                // is written here — the row is `running` for reclaim (the
                // fleet's only requeue path). Any epoch checkpoint a run wrote before a
                // mid-run fault is swept exactly as on the cancelled arm.
                tracing::warn!(job_id = %job_id, worker = %self.worker_id, reason = %why, "gang attempt abandoned; left for reclaim");
                reclaim_unpublished_artifacts(
                    &session.artifact_store(),
                    &catalog,
                    &job_id,
                    &self.worker_id,
                    attempt,
                )
                .await;
                AttemptEnd::LeftForReclaim
            }
            Err(WorkerJobError::HandedOff) => {
                // The placed hand-off: this process wrote NOTHING under
                // its own worker id for this attempt (the placement check
                // runs before any materialization/training starts), so
                // there is nothing here to sweep — the row, and its lease
                // keeper registration, are the executor's now.
                tracing::info!(
                    job_id = %job_id, worker = %self.worker_id,
                    "gang attempt handed off to a placed executor"
                );
                AttemptEnd::LeftForReclaim
            }
            Err(WorkerJobError::Failed(error)) => {
                tracing::error!(job_id = %job_id, error = %error, "training job failed");
                record_failed(
                    holder,
                    &catalog,
                    &job_id,
                    &self.worker_id,
                    attempt,
                    failed_job_message(&error),
                )
                .await;
                // Same reasoning as the `Cancelled` arm above: covers a panic,
                // a `spawn_blocking` join error, and any typed training
                // failure — none of which ever produced a `TrainedArtifact`.
                reclaim_unpublished_artifacts(
                    &session.artifact_store(),
                    &catalog,
                    &job_id,
                    &self.worker_id,
                    attempt,
                )
                .await;
                AttemptEnd::Failed { error }
            }
        }
    }

    /// Submit this attempt as one Ballista task through the installed
    /// [`PlacedGangSubmitter`] and await its stream, instead of running it
    /// in-process. The submitter's exit
    /// arms are total (this function's only return values):
    ///
    /// - the stream ends with AT LEAST ONE batch → [`WorkerJobError::
    ///   HandedOff`] (the executor owns the attempt from here: no terminal
    ///   write, no release);
    /// - the stream ends in an error, or ends with no batch and no error
    ///   (the submission itself never reached a running task) → re-read the
    ///   row: `claimed_by` is STILL this instance (the transfer never
    ///   happened, or the executor refused before the CAS) →
    ///   [`WorkerJobError::Abandoned`] (left `running` for reclaim, an
    ///   attempt spent at the successor's claim);
    ///   `claimed_by` moved → [`WorkerJobError::HandedOff`] (the executor
    ///   owns the attempt; if it died, its own lease expiry requeues it,
    ///   never this instance's).
    ///
    /// This host's holder moves `JobRun → Awaiting{job_id, attempt}` BEFORE
    /// the descriptor is submitted — a CAS from exactly `JobRun`; on its
    /// refusal a `Free` holder (a direct `run_claimed_job` with no
    /// `ClaimGuard`, which admits ranks already) still submits, any other
    /// holder ends this call typed with nothing submitted
    /// (`HostAdmission::begin_awaiting_placement`;
    /// an in-process scheduler can bind the task and the placed executor can
    /// dial this host's `RunRank` before `submit()` returns): the host runs no compute while it waits, so it can
    /// still serve a `RunRank` session — a two-host fleet could not
    /// otherwise assemble if its only free-looking host were the one
    /// awaiting its own placement result.
    async fn submit_placed(
        &self,
        session: &Arc<InferenceSession>,
        catalog: &Arc<Catalog>,
        job_id: &str,
        attempt: u32,
        world: u32,
        submitter: Arc<dyn PlacedGangSubmitter>,
    ) -> std::result::Result<AttemptOutput, WorkerJobError> {
        #[cfg(feature = "test-hooks")]
        training_test_hooks::note_placed(job_id, attempt);
        let descriptor = GangDescriptor {
            job_id: job_id.to_string(),
            attempt,
            world,
            submitter: session.instance_id().to_string(),
            // The required kind is the SUBMITTER's own device, never
            // re-derived from "a GPU exists somewhere" —
            // `DevicePlacement`/`submit_physical_plan` bind/refuse on this
            // exact kind, and the executing session's device-kind check
            // compares against it the same way it does for `InferenceExec`.
            device_kind: session.compute_device().kind(),
        };
        // JobRun -> Awaiting BEFORE the plan crosses the wire, never after
        // `submit()` resolves: when the submitter's own host ALSO hosts the
        // scheduler role (`roles::host_scheduler`'s in-process case,
        // exercised end-to-end by `crates/jammi-ballista/tests/distributed`),
        // the scheduler's own binder can dispatch the task and the placed
        // executor can dial this host's RunRank BEFORE `submitter.submit`'s
        // async call returns to this line, since the round-trip and the
        // scheduler's background bind loop share the same process/runtime.
        // Moving the holder after the submit would leave it `JobRun`, and
        // `admit_rank`'s busy arm would refuse every such dial.
        if let Err(holder) = session
            .host_admission()
            .begin_awaiting_placement(job_id, attempt)
        {
            // The move is a CAS from exactly `JobRun` (the claim loop's
            // hold, `register_job_hold_or_release` -> `job_running`). Its
            // refusal has two readings: a `Free` holder is a direct
            // `run_claimed_job` with no `ClaimGuard` (the documented
            // no-op arm — `Free` admits every `RunRank` dial already, so
            // the gang can assemble and the submit proceeds); ANY other
            // holder (a rank held, a probe in flight, an `Awaiting` for
            // another job) would leave this host refusing every dial while
            // its gang assembles, so the descriptor is refused BEFORE the
            // submit, typed — the row is still this instance's claim and is
            // left for reclaim.
            if holder != Holder::Free {
                return Err(self
                    .placed_submit_end(
                        catalog,
                        job_id,
                        JammiError::FineTune(format!(
                            "submit_placed: this host's job slot is neither JobRun nor Free \
                             (holder {holder:?}); the descriptor was not submitted"
                        )),
                    )
                    .await);
            }
        }
        let mut stream = match submitter.submit(descriptor).await {
            Ok(stream) => stream,
            Err(e) => return Err(self.placed_submit_end(catalog, job_id, e).await),
        };
        use futures::StreamExt;
        let mut saw_batch = false;
        let mut end_err: Option<JammiError> = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(_batch) => saw_batch = true,
                Err(e) => {
                    end_err = Some(e.into());
                    break;
                }
            }
        }
        if saw_batch {
            // The ONE `tracing::info!` line this crate grants
            // `crates/jammi-ballista`'s distributed lane: a process-visible,
            // stdout-captured line naming the job/attempt on the placed
            // submitter's `HandedOff` arm, so a spawned worker's captured
            // log (never the in-process `training_test_hooks` recorder,
            // which a multi-process harness cannot read) can confirm this
            // process ran `run_placed_gang`/`submit_placed` and handed off.
            tracing::info!(
                job_id,
                attempt,
                world,
                "run_placed_gang: submitter HandedOff after the placed \
                 gang's stream completed"
            );
            return Err(WorkerJobError::HandedOff);
        }
        let e = end_err.unwrap_or_else(|| {
            JammiError::FineTune("the placed gang's stream ended with no batch and no error".into())
        });
        Err(self.placed_submit_end(catalog, job_id, e).await)
    }

    /// Re-read the row after a submission fault (before any batch arrived):
    /// still this instance's claim → [`WorkerJobError::Abandoned`]; moved
    /// (or the re-read itself faults) → [`WorkerJobError::HandedOff`] — see
    /// [`Self::submit_placed`]'s doc.
    async fn placed_submit_end(
        &self,
        catalog: &Arc<Catalog>,
        job_id: &str,
        e: JammiError,
    ) -> WorkerJobError {
        let still_mine = matches!(
            catalog.get_job(job_id).await,
            Ok(record) if record.claimed_by.as_deref() == Some(self.worker_id.as_str())
        );
        let end = if still_mine {
            WorkerJobError::Abandoned(format!("placement failed before transfer: {e}"))
        } else {
            // The executor owns the row and has already recorded this
            // attempt's end; the typed error it handed back as the task's
            // own is named here so the submitter's log carries the same
            // failure the row does.
            tracing::warn!(
                job_id,
                error = %e,
                "submit_placed: the placed attempt failed after its transfer; the executor \
                 recorded it"
            );
            WorkerJobError::HandedOff
        };
        #[cfg(feature = "test-hooks")]
        training_test_hooks::note_placed_submit_end(
            job_id,
            matches!(end, WorkerJobError::Abandoned(_)),
        );
        end
    }

    /// Run a placed gang's coordinator body on THIS process — the seam
    /// `GangExec::execute` dispatches through
    /// as the process's installed [`PlacedGangRunner`]. Reuses
    /// `Self::run_claimed_job_under`
    /// VERBATIM (`placed = true`, the recursion guard) — assembly →
    /// dispatch → rounds → publish → finalize, `LeaseHolder::Coordinator`
    /// — the SAME body a `Peer` gang's claimant runs, so the published
    /// bytes are the in-process `Peer` gang's. An ASSOCIATED function, not a method: the
    /// caller (the executor role, `crates/jammi-ballista`) holds only the
    /// session, never a `JobWorker`.
    ///
    /// (i) snapshots `HostAdmission::release_epoch` as this run's
    /// `WorkerShared` birth BEFORE taking this host's job slot through
    /// [`HostAdmission::probe_claim`] (`Free → ClaimProbe`; a host already
    /// holding a rank, a loop-claimed job, or another placement's
    /// probe/await refuses typed BEFORE any row write) — a two-catch
    /// lattice with no gap between the catches: because `HostAdmission::
    /// begin_release` orders the phase flip strictly BEFORE the epoch
    /// bump, a snapshot whose epoch bump IS already visible implies the
    /// flip already happened too, so `probe_claim`'s own phase check —
    /// run immediately after the snapshot — refuses it typed directly; a
    /// snapshot whose epoch bump is NOT yet visible carries no such
    /// guarantee (the flip can still land at any point up to and including
    /// inside `probe_claim`'s own check), so that case is instead caught
    /// downstream by the epoch compare (`WorkerShared::released_since_birth`)
    /// once the bump does become visible — covering `probe_claim()` itself
    /// and every byte of (ii)/(iii) below — see `WorkerShared::
    /// for_single_run`'s own doc for the prior (windowed) shape this
    /// replaced;
    /// (ii) [`Catalog::transfer_claim`] moves `claimed_by` from
    /// `descriptor.submitter` to this instance at the SAME `attempts`,
    /// arming a fresh lease (`false` — the transfer never happened, a
    /// stale runner, or a second launch of an already-transferred attempt
    /// — is a typed refusal, the slot released, no row write); (iii)
    /// re-reads the row (`Catalog::get_job`) and runs
    /// `Self::run_claimed_job_under` — which registers THIS process's own
    /// [`LeaseKeeper`] hold (`register_job_hold_or_release`, the SAME verb
    /// the claim loop uses after `claim_next`) and flips the claim guard's
    /// `ClaimProbe → JobRun` (`HostAdmission::job_running`) itself, so
    /// nothing here duplicates that registration; (iv) maps the body's
    /// `AttemptEnd` to [`PlacedOutcome`] (`Published` → `Trained`;
    /// `Reused` → `Reused`; `Failed` → `Err` carrying the attempt's own
    /// typed error, which the row already records and which reaches the
    /// submitter as the task's error; `LeftForReclaim` → a typed `Err` too,
    /// so the Ballista task itself ends in error and Ballista never re-runs
    /// it: the scheduler is configured with `task_max_failures = 0`,
    /// because a re-run would be a second attempt of the same claim, whose
    /// lease identity the first attempt still holds; jammi's own reclaim,
    /// from a FUTURE claim, is the only path back); (v) releases the slot on
    /// every exit arm (the claim guard's own `Drop`).
    pub async fn run_placed_gang(
        session: &Arc<InferenceSession>,
        descriptor: GangDescriptor,
    ) -> Result<PlacedOutcome> {
        let admission = session.host_admission();
        // The birth snapshot for this run's `WorkerShared`, read BEFORE
        // `probe_claim()` itself — see this function's own doc, point (i),
        // and `WorkerShared::for_single_run` for the lattice argument (a
        // RELEASE lands either before this read, and is then caught by
        // `probe_claim`'s own phase check below, or after it, and is then
        // caught downstream by the epoch compare — no gap between the two).
        let claim_epoch = admission.release_epoch();
        #[cfg(feature = "test-hooks")]
        loop_test_hooks::maybe_park(
            &descriptor.job_id,
            loop_test_hooks::ParkPoint::PlacedGangBeforeProbeClaim,
        )
        .await;
        let Some(claim) = admission.probe_claim() else {
            let phase = *admission.phase_receiver().borrow();
            return Err(JammiError::FineTune(if phase == WorkerPhase::Running {
                "run_placed_gang: this host's job slot is busy (a rank is held, a loop-claimed \
                 job already runs, or another placement is in flight)"
                    .into()
            } else {
                format!(
                    "run_placed_gang: this host has begun a {phase:?} and admits no new gang \
                     (refuse what's new); the row is left with its submitter"
                )
            }));
        };
        #[cfg(feature = "test-hooks")]
        loop_test_hooks::maybe_park(
            &descriptor.job_id,
            loop_test_hooks::ParkPoint::PlacedGangBeforeTransfer,
        )
        .await;
        let catalog = session.catalog();
        let lease = session.inner_config().lease.intervals()?.lease();
        let worker = JobWorker::new(session)?;
        let transferred = catalog
            .transfer_claim(
                &descriptor.job_id,
                &descriptor.submitter,
                worker.worker_id(),
                descriptor.attempt,
                lease,
            )
            .await;
        let transferred = match transferred {
            Ok(t) => t,
            Err(e) => {
                drop(claim);
                return Err(e);
            }
        };
        if !transferred {
            drop(claim);
            return Err(JammiError::FineTune(format!(
                "run_placed_gang: the transfer for job '{}' attempt {} did not land (already \
                 transferred, a stale attempt, or the row moved)",
                descriptor.job_id, descriptor.attempt
            )));
        }
        // `get_job` is tenant-SCOPED (a caller-facing read); the claim it
        // stands in for here is the unscoped kind every claim-loop read is
        // (the module doc: "the catalog used for reclaim/claim is
        // unscoped — a worker serves every tenant's queue") — admin scope
        // for this ONE re-read, exactly as a claim's own row read would see
        // it regardless of tenant.
        let record = match session
            .with_admin_scope(|_scope| catalog.get_job(&descriptor.job_id))
            .await
        {
            Ok(r) => r,
            Err(e) => {
                drop(claim);
                return Err(e);
            }
        };
        let shared = WorkerShared::for_single_run(admission, worker.worker_id.clone(), claim_epoch);
        let end = worker
            .run_claimed_job_under(session, record, &shared, true)
            .await;
        drop(claim);
        match end {
            AttemptEnd::Published { artifact_digest } => {
                Ok(PlacedOutcome::Trained { artifact_digest })
            }
            AttemptEnd::Reused => Ok(PlacedOutcome::Reused),
            AttemptEnd::Failed { error } => Err(error),
            AttemptEnd::LeftForReclaim => Err(JammiError::FineTune(format!(
                "run_placed_gang: job '{}' attempt {} left running for reclaim (no terminal \
                 write)",
                descriptor.job_id, descriptor.attempt
            ))),
        }
    }

    /// Stage a trained artifact in the object store and run the single
    /// lease-guarded finalization for every job kind — the catalog row is the
    /// commit.
    ///
    /// The worker stages the bundle under a **unique per-attempt prefix**
    /// (`{job_id}/{worker_id}/{attempt}`) — the `staged` `model_artifacts`
    /// row first, then the bytes — attests it, and then runs the
    /// lease-guarded finalize transaction
    /// ([`Catalog::finish_job_with_model`]): it flips the job to `completed`
    /// and — atomically, gated on that compare-and-set matching — publishes
    /// the artifact and writes the output `models` row referencing it, plus
    /// one row per retained epoch checkpoint. The transaction matches only
    /// while this worker still holds the lease (`claimed_by = worker_id AND
    /// status = 'running' AND attempts = attempt`), so a worker that lost its
    /// lease writes NOTHING: no job status, no `models` row, no published
    /// artifact. A `wait()` observer that sees `completed` therefore always
    /// finds the output model row, referencing the winner's complete
    /// artifact.
    ///
    /// Every terminating arm — the winner's included — ends with
    /// [`reclaim_unpublished_artifacts`]: whatever this attempt staged and
    /// the finalize did not publish is reclaimed. The winner then reclaims
    /// the job's epoch checkpoints the finalize did not publish
    /// ([`reclaim_checkpoints`]): the job is terminal, so no attempt reads
    /// them again.
    ///
    /// The `models` rows are written through the tenant-pinned `catalog`, so
    /// they land under the job's tenant.
    async fn publish_and_finalize(
        &self,
        holder: LeaseHolder,
        session: &Arc<InferenceSession>,
        catalog: &Arc<Catalog>,
        job_id: &str,
        attempt: u32,
        artifact: TrainedArtifact,
    ) -> PublishOutcome {
        let store = session.artifact_store();
        let TrainedArtifact {
            dir,
            register,
            metrics,
            epoch_checkpoints,
            materialization,
        } = artifact;
        let fail = |reason: String| {
            self.fail_before_finalize(holder, &store, catalog, job_id, attempt, reason)
        };

        // `catalog` is `pinned_to_tenant(record.tenant_id)`: its bound
        // tenant owns the staged row and names the tenant segment the bytes
        // land under, regardless of any task-local scope.
        let staged =
            match publish_artifact(&store, catalog, job_id, &self.worker_id, attempt, &dir).await {
                Ok(staged) => staged,
                Err(e) => return fail(e.to_string()).await,
            };

        // The `materialization.json` attestation is the LAST object written
        // into the bundle, before the finalize; its indexable summary rides
        // the finalize transaction onto the artifact row.
        let summary = match &materialization {
            Some(FineTuneMaterialization {
                descriptor,
                env,
                inputs,
            }) => {
                let attested = store
                    .write_model_materialization(
                        &staged,
                        jammi_db::store::manifest::Materialization {
                            descriptor,
                            env,
                            inputs: inputs.clone(),
                        },
                    )
                    .await
                    .and_then(|manifest| {
                        Ok(MaterializationSummary {
                            definition_hash: manifest.definition_hash.as_str().to_string(),
                            input_anchors_json: serde_json::to_string(&manifest.input_anchors)?,
                        })
                    });
                match attested {
                    Ok(summary) => Some(summary),
                    Err(e) => return fail(e.to_string()).await,
                }
            }
            // A context predictor: no materialization contract, so nothing
            // to attest or record.
            None => None,
        };

        // Distinct-name catalog rows for every RETAINED epoch checkpoint:
        // never an additional VERSION of the output model's name.
        let epoch_model_ids: Vec<String> = epoch_checkpoints
            .iter()
            .map(|(epoch, _)| format!("{}:epoch_{epoch}", register.model_id))
            .collect();

        // The tagged terminal payload `jobs.result` carries the model
        // metrics blob: the generalised `jobs` schema has no dedicated
        // metrics column, so it folds into `result` instead (see
        // `crate::jobs::JobResult::Model`).
        let job_result = crate::jobs::JobResult::Model {
            model_id: register.model_id.clone(),
            artifact_path: staged.artifact().to_string(),
            metrics: metrics.clone(),
            cache_outcome: jammi_db::store::CacheOutcome::Computed,
        };
        let result_json = match serde_json::to_string(&job_result) {
            Ok(j) => j,
            Err(e) => return fail(format!("job result serialisation failed: {e}")).await,
        };

        // The finalize — the module doc's writer table, row 6: reached
        // only with the attempt's `LeaseHolder` in hand.
        let finished = catalog
            .finish_job_with_model(jammi_db::catalog::jobs_repo::FinishJobWithModelParams {
                job_id,
                instance_id: &self.worker_id,
                attempts: attempt,
                result: &result_json,
                output: jammi_db::catalog::jobs_repo::ProducedModel {
                    row: register.row(&register.model_id),
                    artifact: staged,
                    materialization: summary,
                },
                epoch_checkpoints: epoch_checkpoints
                    .into_iter()
                    .zip(&epoch_model_ids)
                    .map(|((_epoch, checkpoint), name)| {
                        jammi_db::catalog::jobs_repo::ProducedModel {
                            row: jammi_db::catalog::jobs_repo::ModelRow {
                                config_json: None,
                                ..register.row(name)
                            },
                            artifact: checkpoint,
                            materialization: None,
                        }
                    })
                    .collect(),
            })
            .await;
        reclaim_unpublished_artifacts(&store, catalog, job_id, &self.worker_id, attempt).await;
        match finished {
            Ok(true) => {
                reclaim_checkpoints(&store, catalog, job_id).await;
                PublishOutcome::Completed
            }
            Ok(false) => {
                // Lost the lease before finalizing: the transaction wrote
                // nothing. Leave the job for reclaim (the re-claiming worker
                // stages its own bundle and its finalize publishes it).
                tracing::debug!(
                    job_id = %job_id,
                    worker = %self.worker_id,
                    %holder,
                    "lost lease before finalize; not finalizing (left for reclaim)"
                );
                PublishOutcome::LeftForReclaim
            }
            Err(e) => {
                tracing::error!(job_id = %job_id, %holder, error = %e, "finish_job_with_model failed");
                PublishOutcome::LeftForReclaim
            }
        }
    }

    /// Give up before the finalize: reclaim what this attempt staged, then
    /// record the terminal failure.
    async fn fail_before_finalize(
        &self,
        holder: LeaseHolder,
        store: &ArtifactStore,
        catalog: &Arc<Catalog>,
        job_id: &str,
        attempt: u32,
        reason: String,
    ) -> PublishOutcome {
        reclaim_unpublished_artifacts(store, catalog, job_id, &self.worker_id, attempt).await;
        record_failed(
            holder,
            catalog,
            job_id,
            &self.worker_id,
            attempt,
            reason.clone(),
        )
        .await;
        PublishOutcome::Failed(reason)
    }

    /// Run a claimed compute-kind job (`neighbor_graph`/`propagate`/
    /// `asof_join`/`embedding`/`infer`) to a terminal state: the
    /// attempt-algorithm dispatch on `jobs.partial_result`
    /// ([`crate::jobs::dispatch_partial_result`]) first, and only when it
    /// says to does this register the job's lease with the session's keeper
    /// (no heartbeat task) and dispatch through
    /// [`crate::jobs::execute_compute`] — then performs the single
    /// lease-guarded terminal write. A worker that lost its lease during the
    /// compute does not finalize (`finish_job`/`fail_job` match zero rows);
    /// the job is left for [`Catalog::reclaim_expired_jobs`].
    #[allow(clippy::too_many_arguments)]
    async fn run_claimed_compute_job(
        &self,
        session: &Arc<InferenceSession>,
        catalog: &Arc<Catalog>,
        shared: &Arc<WorkerShared>,
        job_id: &str,
        spec_json: &str,
        attempt: u32,
        partial_result: Option<&str>,
        tenant_id: Option<jammi_db::TenantId>,
    ) {
        // Decode the one persisted type (`crate::jobs::JobSpec`'s own doc),
        // then project to `ComputeSpec` — see the loop-claimer training path
        // above for why, and `JobSpec::as_compute_spec`'s doc.
        let job_spec: crate::jobs::JobSpec = match serde_json::from_str(spec_json) {
            Ok(s) => s,
            Err(e) => {
                record_failed(
                    LeaseHolder::LoopClaimer,
                    catalog,
                    job_id,
                    &self.worker_id,
                    attempt,
                    format!("undeserialisable compute spec: {e}"),
                )
                .await;
                return;
            }
        };
        let Some(spec) = job_spec.as_compute_spec() else {
            // `record.kind` named a compute kind, routing this attempt here
            // at all, but the persisted spec JSON decoded as a training
            // variant — the two columns disagree.
            record_failed(
                LeaseHolder::LoopClaimer,
                catalog,
                job_id,
                &self.worker_id,
                attempt,
                format!(
                    "compute claim path: jobs.kind names a compute kind but the persisted spec \
                     decoded as training kind {}",
                    job_spec.kind()
                ),
            )
            .await;
            return;
        };

        // Post-claim checkpoint: a cancel requested while the job sat
        // `queued` is honoured before any prior-attempt dispatch or
        // producer runs (`execute_compute` re-checks before dispatch).
        if let Err(e) = crate::jobs::check_cancel(catalog, job_id).await {
            record_unsuccessful_end(
                LeaseHolder::LoopClaimer,
                catalog,
                job_id,
                &self.worker_id,
                attempt,
                UnsuccessfulEnd::from(&e),
            )
            .await;
            return;
        }

        match crate::jobs::dispatch_partial_result(
            session,
            catalog,
            tenant_id,
            job_id,
            attempt,
            partial_result,
            &self.worker_id,
        )
        .await
        {
            Ok(crate::jobs::PartialResultDisposition::Ready(result)) => {
                match serde_json::to_string(&result) {
                    Ok(result_json) => {
                        match catalog
                            .finish_job(jammi_db::catalog::jobs_repo::FinishJobParams {
                                job_id,
                                instance_id: &self.worker_id,
                                attempts: attempt,
                                result: &result_json,
                            })
                            .await
                        {
                            Ok(true) => {}
                            Ok(false) => tracing::debug!(
                                job_id, worker = %self.worker_id,
                                "lost lease before finish (partial_result adopt); left for reclaim"
                            ),
                            Err(e) => {
                                tracing::error!(job_id, error = %e, "finish_job failed (partial_result adopt)")
                            }
                        }
                    }
                    Err(e) => {
                        record_failed(
                            LeaseHolder::LoopClaimer,
                            catalog,
                            job_id,
                            &self.worker_id,
                            attempt,
                            format!("result serialisation failed: {e}"),
                        )
                        .await;
                    }
                }
                return;
            }
            Ok(crate::jobs::PartialResultDisposition::BackOff) => {
                tracing::debug!(
                    job_id, worker = %self.worker_id,
                    "a prior attempt's partial_result table is still building under a \
                     live lease; backing off without double-producing"
                );
                return;
            }
            Ok(crate::jobs::PartialResultDisposition::MaterializeAnew) => {}
            Err(e) => {
                tracing::error!(job_id, error = %e, "dispatch_partial_result failed; materializing anew");
            }
        }

        let Some(hold) = register_job_hold_or_release(
            session,
            catalog,
            shared,
            job_id,
            attempt,
            LeaseHolder::LoopClaimer,
        )
        .await
        else {
            return;
        };
        let job_attempt = jammi_db::catalog::result_repo::JobAttempt {
            job_id,
            instance_id: &self.worker_id,
            attempts: attempt,
        };
        let outcome = crate::jobs::execute_compute(session, catalog, &spec, job_attempt).await;
        drop(hold);

        match outcome {
            Ok(result) => match serde_json::to_string(&result) {
                Ok(result_json) => {
                    match catalog
                        .finish_job(jammi_db::catalog::jobs_repo::FinishJobParams {
                            job_id,
                            instance_id: &self.worker_id,
                            attempts: attempt,
                            result: &result_json,
                        })
                        .await
                    {
                        Ok(true) => {}
                        Ok(false) => tracing::debug!(
                            job_id, worker = %self.worker_id,
                            "lost lease before finish; left for reclaim"
                        ),
                        Err(e) => tracing::error!(job_id, error = %e, "finish_job failed"),
                    }
                }
                Err(e) => {
                    record_failed(
                        LeaseHolder::LoopClaimer,
                        catalog,
                        job_id,
                        &self.worker_id,
                        attempt,
                        format!("result serialisation failed: {e}"),
                    )
                    .await;
                }
            },
            Err(e) => {
                record_unsuccessful_end(
                    LeaseHolder::LoopClaimer,
                    catalog,
                    job_id,
                    &self.worker_id,
                    attempt,
                    UnsuccessfulEnd::from(&e),
                )
                .await;
            }
        }
    }

    /// Dispatch a claimed spec to its kind's from-scratch reconstruction and
    /// training, returning the [`TrainedArtifact`] on success.
    #[tracing::instrument(
        skip(self, session, catalog, spec, cancel),
        fields(job_id = %job_id, worker_id = %self.worker_id)
    )]
    ///
    /// `recorded_pair` is the training-set identity pair a prior attempt of
    /// this job recorded on the row, if any: the FineTune arm then BINDS that
    /// table ([`bind_recorded_training_set`]) instead of materializing a
    /// fresh one — a registered source is anchored unpinned, so a fresh
    /// materialization would mint a new table and the write-once CAS would
    /// read the retry as a moved claim.
    ///
    /// `holder` is who this attempt runs as ([`lease_holder_for`], derived
    /// by the caller from this same spec): rank 0's `RunnerRole` on the
    /// in-process arms, and the writer of the acceleration marker on the
    /// predictor arm. The `Peer` arm is the coordinator body, which runs as
    /// `LeaseHolder::Coordinator` by its own name — the same value
    /// `lease_holder_for` derives for it, since both decide from
    /// `TopologyDecision::decide` over the same inputs.
    #[allow(clippy::too_many_arguments)]
    async fn run_spec(
        &self,
        session: &Arc<InferenceSession>,
        catalog: &Arc<Catalog>,
        job_id: &str,
        spec: TrainingSpec,
        cancel: &Arc<AtomicBool>,
        attempt: u32,
        recorded_pair: Option<TrainingSetIdentityPair>,
        holder: LeaseHolder,
    ) -> std::result::Result<AttemptOutput, WorkerJobError> {
        let kind = spec.kind();
        match spec.plan() {
            // Every kind that trains from a training-set table. The kinds
            // differ in ONE step — how the table is produced
            // (`view.producer`); from the table on there is one path, so every
            // topology serves every such kind.
            TrainingPlan::FromTrainingSet(view) => {
                let spec_canonical = crate::fine_tune::spec::fine_tune_spec_canonical(&view)
                    .map_err(WorkerJobError::from)?;
                let (columns, task, common) = (view.columns, view.task, view.common.clone());
                let detected =
                    detect_training_format(&columns, task).map_err(WorkerJobError::from)?;

                // The table: a retry binds the one the row already names
                // (the job's identity, write-once); a first attempt
                // materialises (or reuses on the engine's own key). The rows a
                // run trains on are a durable, attested artifact, not this
                // worker's private scan.
                let table = match (&recorded_pair, &view.producer) {
                    (Some(pair), _) => {
                        bind_recorded_training_set(session, catalog, pair)
                            .await
                            .map_err(WorkerJobError::from)?
                            .0
                    }
                    (None, TrainingSetProducer::Projection { source, .. }) => {
                        training_set::materialize_projection_table(
                            session,
                            source,
                            &columns,
                            task,
                            detected.format_tag(),
                        )
                        .await
                        .map_err(WorkerJobError::from)?
                    }
                    (
                        None,
                        TrainingSetProducer::GraphSample {
                            sources,
                            sample_config,
                        },
                    ) => {
                        let now = chrono::Utc::now().to_rfc3339();
                        let inputs = vec![
                            InputAnchor::unpinned_at_instant(&sources.node_source, now.clone()),
                            InputAnchor::unpinned_at_instant(&sources.edge_source, now),
                        ];
                        materialize_graph_training_set(
                            session,
                            job_id,
                            sources,
                            *sample_config,
                            inputs,
                        )
                        .await
                        .map_err(WorkerJobError::from)?
                    }
                };

                // The ONE source binding every rank of this job performs
                // over the same table — rank 0 here, a `Peer` member in
                // `run_member_rank` — so the ranks' loaders agree by
                // construction.
                let training_source =
                    bind_training_source(session, &table, &columns, task, detected, &common)
                        .await
                        .map_err(WorkerJobError::from)?;
                #[cfg(feature = "test-hooks")]
                training_test_hooks::note_source_kind(
                    job_id,
                    match &training_source {
                        crate::fine_tune::source::TrainingSource::Resident(_) => "resident",
                        crate::fine_tune::source::TrainingSource::Streamed(_) => "streamed",
                    },
                );
                #[cfg(feature = "test-hooks")]
                if let crate::fine_tune::source::TrainingSource::Streamed(streamed) =
                    &training_source
                {
                    training_test_hooks::note_streamed_total_rows(job_id, streamed.total_rows);
                }
                // The `ProducingDescriptor::FineTune` materialization
                // identity — the training-set table's own definition hash,
                // artifact digest and row count, binding this fine-tune to
                // the EXACT materialised `TrainingSet` it trained from (see
                // that descriptor variant's own doc). Recorded for the
                // column-source `FineTune` kind only: the descriptor's fields
                // name a projection's `source` and `method`, which a graph
                // sample has no value for.
                let training_set_artifact_digest = {
                    let parquet_url = jammi_db::storage::StorageUrl::parse(table.parquet_path())
                        .map_err(JammiError::from)
                        .map_err(WorkerJobError::from)?;
                    session
                        .result_store()
                        .read_materialization_manifest(&parquet_url)
                        .await
                        .map_err(WorkerJobError::from)?
                        .map(|manifest| manifest.artifact.0)
                        .ok_or_else(|| {
                            WorkerJobError::from(JammiError::FineTune(format!(
                                "training set '{}' has no materialization manifest",
                                table.table_name()
                            )))
                        })?
                };
                // The training-set identity pair a `Peer` gang's members
                // are admitted against (`GangService::run_rank`'s world>1
                // conjunct): the SAME sidecar digest recorded in the
                // materialization descriptor below, and the table's name —
                // written onto the job row by the coordinator body's CAS
                // (`materialize_or_reuse_training_set`) before any member
                // is dialed.
                let pair = TrainingSetIdentityPair {
                    training_set_ref: training_set_artifact_digest.clone(),
                    training_set_location: table.table_name().to_string(),
                };
                let materialization_source = FineTuneMaterializationSource {
                    spec_canonical,
                    training_set_definition_hash: table.definition_hash.as_str().to_string(),
                    training_set_artifact_digest,
                    training_set_row_count: table.row_count() as u64,
                };
                let topology = TopologyDecision::decide(
                    common.world_size,
                    session.inner_config().worker.local_ranks,
                );
                #[cfg(feature = "test-hooks")]
                training_test_hooks::note_topology(job_id, topology);
                // The model's identity is known BEFORE any rank trains — it
                // is a function of the training-set table, the spec, the base
                // model and the topology, all decided above — so a
                // `CachePolicy::Use` attempt probes for an already-published
                // model of that identity here, before the gang is assembled
                // and the trainer spawned. `Bypass` never probes.
                let materialization = fine_tune_materialization(
                    session,
                    &materialization_source,
                    task,
                    &common,
                    topology,
                )
                .await?;
                if common.cache == CachePolicy::Use {
                    if let Some(reused) = self
                        .reuse_published_model(
                            catalog,
                            job_id,
                            attempt,
                            task,
                            &common,
                            &materialization,
                        )
                        .await?
                    {
                        return Ok(AttemptOutput::Reused(reused));
                    }
                }
                let run = FineTuneRun {
                    task,
                    common,
                    source: training_source,
                    materialization,
                };
                let trained = match topology {
                    TopologyDecision::Single => {
                        self.train_fine_tune(
                            session,
                            catalog,
                            job_id,
                            run,
                            cancel,
                            attempt,
                            holder,
                            RankTopology::Single,
                        )
                        .await
                    }
                    TopologyDecision::Local { world } => {
                        self.train_fine_tune(
                            session,
                            catalog,
                            job_id,
                            run,
                            cancel,
                            attempt,
                            holder,
                            RankTopology::Local { world },
                        )
                        .await
                    }
                    TopologyDecision::Peer { world } => {
                        self.coordinate(
                            session, catalog, job_id, kind, run, cancel, attempt, world, pair,
                        )
                        .await
                    }
                };
                trained.map(|trained| AttemptOutput::Trained(Box::new(trained)))
            }
            TrainingPlan::ContextPredictor {
                source,
                predictor_spec,
            } => {
                // This kind never runs
                // `run_fine_tune_blocking`'s measuring probe (it has no
                // `backbone_dtype`/fused-kernel surface to measure at all) —
                // write the self-describing terminal marker up front so the
                // record never reads the submission-time `{"state":
                // "pending"}` marker past this job's eventual terminal
                // status, regardless of whether training below succeeds or
                // fails.
                mark_acceleration_not_applicable(
                    holder,
                    catalog,
                    job_id,
                    &self.worker_id,
                    attempt,
                    "context_predictor",
                )
                .await;
                // The predictor training is async (it samples through the SQL
                // surface). It checks `cancel` at every epoch boundary and
                // returns the trained weights in a local tempdir plus the model
                // registration descriptor; the worker's unified finalize
                // publishes the artifact and registers the model row through the
                // tenant-pinned catalog (the same path the fine-tune kinds take),
                // so the model lands under the job's tenant.
                session
                    .run_context_predictor_training(source, predictor_spec, cancel)
                    .await
                    .map(|trained| AttemptOutput::Trained(Box::new(trained)))
                    .map_err(|e| classify(cancel, e))
            }
        }
    }

    /// The `CachePolicy::Use` probe: finish this attempt against an
    /// already-published artifact of `materialization`'s definition, if one
    /// exists ([`Catalog::finish_job_reusing_artifact`] — the probe, the
    /// attempt-guarded job CAS and the output row's attach are one
    /// transaction). `None` is a miss: nothing was written and the attempt
    /// trains. A hit whose attempt guard did not match wrote nothing either;
    /// that is a lost lease, so the attempt ends exactly as a training run
    /// whose lease was lost does.
    async fn reuse_published_model(
        &self,
        catalog: &Arc<Catalog>,
        job_id: &str,
        attempt: u32,
        task: ModelTask,
        common: &TrainingCommon,
        materialization: &FineTuneMaterialization,
    ) -> std::result::Result<Option<ReusedModel>, WorkerJobError> {
        use jammi_db::catalog::jobs_repo::{FinishJobReusingArtifactParams, ReuseFinish};
        let model_id = crate::fine_tune::training_job::fine_tuned_model_id(job_id);
        let definition_hash = materialization
            .definition_hash()
            .map_err(WorkerJobError::from)?;
        let register = ModelRegistration {
            model_id: model_id.clone(),
            version: 1,
            model_type: "fine-tuned",
            task,
            base_model_id: Some(common.base_model.clone()),
            config_json: None,
        };
        let result_model_id = model_id.clone();
        let finish = catalog
            .finish_job_reusing_artifact(FinishJobReusingArtifactParams {
                job_id,
                instance_id: &self.worker_id,
                attempts: attempt,
                definition_hash: &definition_hash,
                inputs: &materialization.inputs,
                output: register.row(&register.model_id),
                result: Arc::new(move |artifact| {
                    Ok(serde_json::to_string(&crate::jobs::JobResult::Model {
                        model_id: result_model_id.clone(),
                        artifact_path: artifact.to_string(),
                        metrics: None,
                        cache_outcome: jammi_db::store::CacheOutcome::Reused(
                            jammi_db::store::ReusedArtifact::Model(artifact.clone()),
                        ),
                    })?)
                }),
            })
            .await
            .map_err(WorkerJobError::from)?;
        match finish {
            ReuseFinish::Reused(artifact) => {
                tracing::info!(
                    job_id = %job_id,
                    worker = %self.worker_id,
                    %artifact,
                    "fine-tune completed by reusing a published artifact of the same definition"
                );
                Ok(Some(ReusedModel { model_id, artifact }))
            }
            ReuseFinish::Miss => Ok(None),
            ReuseFinish::LostLease => Err(WorkerJobError::Cancelled),
        }
    }

    /// Load the base model, build the training target, and drive the blocking
    /// LoRA trainer — the shared tail of the two fine-tune kinds. The loop trains
    /// and persists the adapter but writes no terminal status; on a clean return
    /// the worker registers the output-model row through the tenant-pinned
    /// catalog and hands the model id + run metrics to the caller's single
    /// lease-guarded finalization.
    ///
    /// `topology` is the rank layout `run_spec` decided ([`TopologyDecision`]):
    /// a single rank, an in-process `Local`
    /// gang of `world` ranks over this host's own devices, or rank 0 of a
    /// `Peer` gang whose members the coordinator body already dialed. In
    /// every case rank 0 runs on the blocking pool under the witness minted
    /// at `BlockingCall::spawn_blocking`; a `Local` gang's other ranks are
    /// OS threads pinned to their devices, each under the witness minted at
    /// ITS OWN `BlockingCall::spawn_thread` boundary — the second
    /// production minting site (the collective module doc).
    ///
    /// `holder` is rank 0's role: the lease holder this attempt runs as
    /// (`RunnerRole::Holder(holder)`); every other local rank runs as
    /// `RunnerRole::Rank { rank }` and writes nothing durable.
    #[allow(clippy::too_many_arguments)]
    async fn train_fine_tune(
        &self,
        session: &Arc<InferenceSession>,
        catalog: &Arc<Catalog>,
        job_id: &str,
        run: FineTuneRun,
        cancel: &Arc<AtomicBool>,
        attempt: u32,
        holder: LeaseHolder,
        topology: RankTopology,
    ) -> std::result::Result<TrainedArtifact, WorkerJobError> {
        let FineTuneRun {
            task,
            common,
            source: training_source,
            materialization,
        } = run;
        let output_model_id = crate::fine_tune::training_job::fine_tuned_model_id(job_id);
        let model_source = ModelSource::parse(&common.base_model);

        // Load the base model under the task being fine-tuned so the right tower
        // (text vs audio) is materialised and `embedding_dim()` reports the
        // shared-latent width the head must match.
        let guard = session
            .model_cache()
            .get_or_load(&model_source, task, None)
            .await
            .map_err(WorkerJobError::from)?;
        let base_model_arc = Arc::clone(&guard.model);
        let hidden_size = guard.model.embedding_dim().ok_or_else(|| {
            WorkerJobError::Failed(JammiError::FineTune(
                "Base model does not support embeddings".into(),
            ))
        })?;
        drop(guard);

        // Test hook: a no-op in production (the whole call
        // compiles away without `test-hooks`). Parks here, with the job's
        // lease hold and cancel-request watcher already live and no other
        // `Arc<Catalog>` clone constructed yet (in particular, before
        // `RunFineTuneParams`'s own clone below), when a test has armed
        // `training_test_hooks::arm_pause_before_spawn_blocking` for THIS
        // job — see that function's doc.
        #[cfg(feature = "test-hooks")]
        training_test_hooks::checkpoint_before_spawn_blocking(job_id).await;

        let base_model = common.base_model.clone();
        let cancel_for_classify = Arc::clone(cancel);
        // `common.config` moves into `params` for the blocking trainer; a clone
        // survives here so a training failure can be classified against the
        // config that produced it (`classify_training_oom` names
        // `batch_size`/`max_seq_length`/`backbone_dtype` in the OOM guidance).
        let config_for_error = common.config.clone();
        let batch = common.config.batch_size;
        let rank_timeout = Duration::from_secs(session.inner_config().worker.rank_timeout_secs);
        // Every rank's parameters share this shape; rank 0's is built first
        // (it owns the eager read's pool reservation and the primary device),
        // the other local ranks replicate the source over their own devices.
        let rank_params =
            |role: RunnerRole,
             rank_ctx: Option<RankContext>,
             source: crate::fine_tune::source::TrainingSource,
             base_model_arc: Arc<crate::model::LoadedModel>,
             device_config: DeviceConfig| RunFineTuneParams {
                catalog: Arc::clone(catalog),
                artifact_store: session.artifact_store(),
                artifact_dir: session.inner_config().artifact_dir.clone(),
                job_id: job_id.to_string(),
                worker_id: self.worker_id.clone(),
                attempt,
                role,
                rank_ctx,
                base_model: base_model.clone(),
                task,
                config: config_for_error.clone(),
                source,
                base_model_arc,
                hidden_size,
                device_config,
                cancel: Arc::clone(cancel),
                hub: session.hub().clone(),
            };

        // The rank layout: rank 0's context, and — for an in-process gang —
        // the other ranks' whole parameter sets, each over its own device,
        // its own replicated source and its own model-cache entry for that
        // device (`ModelCache::get_or_load_on`).
        let (rank0_ctx, rank0_device, mut other_ranks): (
            Option<RankContext>,
            DeviceConfig,
            Vec<RunFineTuneParams>,
        ) = match topology {
            RankTopology::Single => (None, session.device_config().clone(), Vec::new()),
            RankTopology::Peer { world, coordinator } => {
                let partition = PartitionSpec::for_gang(
                    0,
                    world as usize,
                    batch,
                    PartitionRule::BlockByGlobalBatch,
                )
                .map_err(WorkerJobError::from)?;
                let collective: Arc<dyn Collective> = coordinator;
                (
                    Some(RankContext::new(collective, partition)),
                    session.device_config().clone(),
                    Vec::new(),
                )
            }
            RankTopology::Local { world } => {
                // Rank `r` on `[gpu] devices[r]` (local ranks are threads
                // pinned to devices); `[worker] local_ranks <=
                // devices.len()` is enforced at config load and `world <=
                // local_ranks` by `TopologyDecision::decide`, so every rank
                // has its own device — restated here rather than assumed.
                let devices = session.device_config().devices.clone();
                if (world as usize) > devices.len() {
                    return Err(WorkerJobError::Failed(JammiError::FineTune(format!(
                        "a Local gang of {world} ranks needs {world} configured [gpu] devices; \
                         this host lists {}",
                        devices.len()
                    ))));
                }
                let mut rank_device_configs = Vec::with_capacity(world as usize);
                let mut rank_devices = Vec::with_capacity(world as usize);
                for &device in devices.iter().take(world as usize) {
                    let device_config = session
                        .device_config()
                        .for_device(device)
                        .map_err(WorkerJobError::from)?;
                    rank_devices.push(
                        crate::model::backend::candle::select_device(&device_config)
                            .map_err(WorkerJobError::from)?,
                    );
                    rank_device_configs.push(device_config);
                }
                let gang = LocalGang::with_timeout(rank_devices, rank_timeout)
                    .map_err(WorkerJobError::from)?;
                let context_for = |rank: u32| -> std::result::Result<RankContext, WorkerJobError> {
                    let local = gang.rank(rank).map_err(WorkerJobError::from)?;
                    let partition = PartitionSpec::for_gang(
                        rank as usize,
                        world as usize,
                        batch,
                        PartitionRule::BlockByGlobalBatch,
                    )
                    .map_err(WorkerJobError::from)?;
                    Ok(RankContext::new(Arc::new(local), partition))
                };
                let mut others = Vec::with_capacity(world as usize - 1);
                for rank in 1..world {
                    let device_config = rank_device_configs[rank as usize].clone();
                    let model = session
                        .model_cache()
                        .get_or_load_on(devices[rank as usize], &model_source, task, None)
                        .await
                        .map_err(WorkerJobError::from)?;
                    let base_model_arc = Arc::clone(&model.model);
                    drop(model);
                    others.push(rank_params(
                        RunnerRole::Rank { rank },
                        Some(context_for(rank)?),
                        training_source.replicate(),
                        base_model_arc,
                        device_config,
                    ));
                }
                (
                    Some(context_for(0)?),
                    rank_device_configs[0].clone(),
                    others,
                )
            }
        };
        let params = rank_params(
            RunnerRole::Holder(holder),
            rank0_ctx,
            training_source,
            base_model_arc,
            rank0_device,
        );

        // The other local ranks, if any, start FIRST as OS threads
        // (`BlockingCall::spawn_thread` — the second production minting
        // site: each rank's `TrainingLoop::run` receives the witness minted
        // at its own boundary). A plain thread has no runtime context of its
        // own, so each enters this runtime's handle before running: the
        // trainer's checkpoint uploads and the resume discovery `block_on`
        // that handle from the rank's thread, exactly as rank 0 does from
        // the blocking pool.
        let runtime = tokio::runtime::Handle::current();
        let mut rank_threads = Vec::with_capacity(other_ranks.len());
        for rank_params in other_ranks.drain(..) {
            let runtime = runtime.clone();
            rank_threads.push((
                rank_params.role.rank(),
                BlockingCall::spawn_thread(move |call| {
                    let _runtime = runtime.enter();
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        run_fine_tune_blocking(&call, rank_params)
                    }))
                }),
            ));
        }

        // The blocking trainer runs on the blocking pool so it never starves the
        // heartbeat / poll tasks on the async runtime. Panics are caught so a
        // crashing loop still resolves to a terminal classification rather than
        // a wedged `running` row. `BlockingCall::spawn_blocking` is the first
        // production minting site of the collective's witness: rank 0's every
        // collective call takes the `call` minted here, on this blocking-pool
        // thread (the collective module doc).
        let result = crate::fine_tune::collective::BlockingCall::spawn_blocking(move |call| {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_fine_tune_blocking(&call, params)
            }));
            // Counted INSIDE the blocking closure: this thread keeps running
            // after the owning future is aborted, and a test observing that
            // the abandoned attempt never finalizes needs the exact moment the
            // thread returned.
            #[cfg(feature = "test-hooks")]
            training_test_hooks::note_training_thread_finished();
            outcome
        })
        .await;

        // Every other local rank is joined before rank 0's result is read:
        // a gang's ranks end together (lockstep, or a fault every rank
        // sees), and a rank that ended in an error or a panic makes the
        // whole run a failure — rank 0's artifact is never published over a
        // gang that did not complete. Joined off the runtime (a thread join
        // blocks).
        let mut rank_failures: Vec<String> = Vec::new();
        for (rank, thread) in rank_threads {
            match tokio::task::spawn_blocking(move || thread.join()).await {
                Ok(Ok(Ok(Ok(_result)))) => {}
                Ok(Ok(Ok(Err(e)))) => rank_failures.push(format!("rank {rank}: {e}")),
                Ok(Ok(Err(payload))) => rank_failures.push(format!(
                    "rank {rank}: Panic: {}",
                    panic_message(payload.as_ref())
                )),
                Ok(Err(_)) => rank_failures.push(format!("rank {rank}: the rank thread panicked")),
                Err(join_err) => rank_failures.push(format!("rank {rank}: join error: {join_err}")),
            }
        }

        let training = match result {
            Ok(Ok(Ok(training))) if rank_failures.is_empty() => training,
            Ok(Ok(Ok(_))) => {
                return Err(WorkerJobError::Failed(JammiError::FineTune(format!(
                    "the gang did not complete: {}",
                    rank_failures.join("; ")
                ))));
            }
            Ok(Ok(Err(e))) => {
                return Err(classify_training_error(
                    &cancel_for_classify,
                    &config_for_error,
                    e,
                ));
            }
            Ok(Err(payload)) => {
                // A panic on the blocking thread — no `TrainedArtifact` was
                // ever built. This does NOT sweep here: it
                // returns `WorkerJobError::Failed`, which propagates
                // unchanged to `run_claimed_job`'s exhaustive `Failed` arm,
                // which sweeps this exact (job_id, worker_id, attempt) range
                // — sweeping here too would be a wasted double sweep of the
                // identical range, not a second reclaim mechanism.
                return Err(WorkerJobError::Failed(JammiError::FineTune(format!(
                    "Panic: {}",
                    panic_message(payload.as_ref())
                ))));
            }
            Err(join_err) => {
                // Same reasoning as the panic arm immediately above: the
                // blocking task never returned at all, but the resulting
                // `Failed` still funnels through `run_claimed_job`'s single
                // sweep — no sweep call belongs here.
                return Err(WorkerJobError::Failed(JammiError::FineTune(format!(
                    "training task join error: {join_err}"
                ))));
            }
        };

        // Hand the worker's unified finalize the trained adapter files (in their
        // tempdir) plus the model-registration descriptor. The worker publishes
        // the files to the artifact store under a unique per-attempt prefix and
        // registers the row pointing at that prefix, both before the finalize
        // CAS — so a `wait()` observer that sees `completed` always finds a
        // registered model row backed by a complete artifact. The model id is
        // deterministic (`jammi:fine-tuned:{job_id}`) and the catalog upserts, so
        // a re-claiming worker is idempotent.
        Ok(TrainedArtifact {
            dir: training.artifact_dir,
            register: ModelRegistration {
                model_id: output_model_id,
                version: 1,
                model_type: "fine-tuned",
                task,
                base_model_id: Some(base_model),
                config_json: None,
            },
            metrics: Some(training.metrics_json),
            epoch_checkpoints: training.epoch_checkpoints,
            materialization: Some(materialization),
        })
    }
}

/// The loop task's lifecycle as tracked by its owning [`EmbeddedWorker`].
///
/// A bare `Mutex<Option<JoinHandle>>` collapses two different states into
/// one `None`: "already gracefully joined, nothing to abort" and "taken by
/// a `stop_and_join`/`release_and_stop` whose own future was then cancelled
/// by ITS caller — e.g. `tokio::select!` dropping a `stop_and_join` future
/// the instant a RELEASE signal races a DRAIN already in flight — the task
/// is still running (or was, at the moment we lost track of it) and an
/// abort is exactly what is owed. A bare `JoinHandle` drop DETACHES rather
/// than aborts, so that collapse let the loop run on, undetected, past a
/// DRAIN-then-RELEASE (SIGTERM-then-SIGINT): `release_and_stop`'s 2e found
/// `None` (already taken, never restored) and skipped its abort arm
/// entirely, and `Drop` found `None` too.
///
/// [`TakenHandle`] is the only way to read a `JoinHandle` out of this type:
/// on ordinary completion (join or an explicit synchronous abort) it leaves
/// the slot `Joined`; if the `TakenHandle`'s OWN holder is dropped before
/// that — the caller's future was itself cancelled while suspended on it —
/// its `Drop` puts the handle back as `Abandoned`, never losing it to a
/// bare detach.
enum LoopTask {
    /// The task is (as far as this guard knows) still running, and nothing
    /// has yet tried to stop it.
    Running(tokio::task::JoinHandle<()>),
    /// The handle was fully disposed of — cooperatively joined, or aborted
    /// synchronously by whichever caller last held it. Nothing owed.
    Joined,
    /// A previous attempt to stop the task (`stop_and_join` or
    /// `release_and_stop`) was itself cancelled while it held the handle,
    /// before it could join or abort it. The task's exact state is now
    /// unknown, so the next caller that observes this must abort
    /// unconditionally rather than retry the cooperative dance — always
    /// safe here (an abort can only land before a claim's `COMMIT`, which
    /// always rolls back, or between `COMMIT` and hold registration, which
    /// the reclaim path recovers with `attempts + 1`, never `failed`; see
    /// `EmbeddedWorker::release_and_stop`'s step 2e).
    Abandoned(tokio::task::JoinHandle<()>),
}

/// Takes the current [`LoopTask`]'s handle out of `slot` for direct
/// manipulation (join or abort), replacing the slot with `Joined`
/// provisionally. Implements [`Future`](std::future::Future) so `.await`ing
/// one directly resolves once the underlying task returns; if the future
/// awaiting it is itself dropped first (a `tokio::select!` losing a race),
/// [`Drop`] restores the handle into the slot as `LoopTask::Abandoned`
/// rather than letting the bare `JoinHandle` drop DETACH the task. Calling
/// [`Self::abort_now`] instead performs a synchronous abort with no `.await`
/// in between the take and the abort, so no external cancellation can land
/// in the gap.
struct TakenHandle<'a> {
    slot: &'a std::sync::Mutex<LoopTask>,
    handle: Option<tokio::task::JoinHandle<()>>,
    /// Whether this handle was reclaimed from a previously `Abandoned`
    /// slot, as opposed to a fresh `Running` one — `release_and_stop`'s 2e
    /// always aborts a reclaimed handle rather than re-attempting the
    /// cooperative wait, since the task's state is unknown.
    reclaimed: bool,
}

impl<'a> TakenHandle<'a> {
    /// `None` when the slot is already `Joined` (nothing to take).
    fn take(slot: &'a std::sync::Mutex<LoopTask>) -> Option<Self> {
        let mut guard = slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (handle, reclaimed) = match std::mem::replace(&mut *guard, LoopTask::Joined) {
            LoopTask::Running(h) => (Some(h), false),
            LoopTask::Abandoned(h) => (Some(h), true),
            LoopTask::Joined => (None, false),
        };
        drop(guard);
        handle.map(|handle| Self {
            slot,
            handle: Some(handle),
            reclaimed,
        })
    }

    /// Whether this handle came from a slot a previous caller abandoned
    /// mid-stop.
    fn reclaimed(&self) -> bool {
        self.reclaimed
    }

    /// Abort the task now, synchronously — no `.await` between taking the
    /// handle and issuing the abort, so this cannot itself be interrupted
    /// by an external cancellation. Leaves the slot `Joined`.
    fn abort_now(mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

impl<'a> std::future::Future for TakenHandle<'a> {
    type Output = std::result::Result<(), tokio::task::JoinError>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // `TakenHandle` is `Unpin` (every field is), so projecting through
        // the pin is just a reborrow.
        let this = self.get_mut();
        let handle = this
            .handle
            .as_mut()
            .expect("TakenHandle polled after its handle was already taken");
        match std::pin::Pin::new(handle).poll(cx) {
            std::task::Poll::Ready(result) => {
                // Consumed to completion: `Drop` below finds `None` and
                // leaves the slot `Joined` (already set at `take`).
                this.handle = None;
                std::task::Poll::Ready(result)
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

impl<'a> Drop for TakenHandle<'a> {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            // Still holding an un-joined, un-aborted handle: our OWN holder
            // (the `stop_and_join`/`release_and_stop` future this lived
            // inside) was cancelled before finishing with it. Restore it
            // rather than letting it detach.
            *self
                .slot
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = LoopTask::Abandoned(handle);
        }
    }
}

/// An RAII guard owning an embedded [`JobWorker`]'s background task, and
/// the process's handle on its two shutdown modes.
///
/// * [`Self::stop_and_join`] — DRAIN: signal stop, let the in-flight job
///   finish (keeper alive, heartbeats continue, every epoch bundle lands),
///   observe the loop's terminal state, join the task, delete the `workers`
///   row.
/// * [`Self::release_and_stop`] — RELEASE: hand every lease this loop holds
///   back to the catalog (both classes), stop the loop — cooperatively while
///   nothing is in flight, by abort while a job is — and delete the row. The
///   released job is claimable by a successor within one idle poll and costs
///   no attempt.
/// * `Drop` — the unattended halt: sets stop and aborts the task. Stops the
///   *loop*, not in-flight training: a job already running inside
///   `spawn_blocking` cannot be force-aborted, so aborting the loop task
///   only cancels it at the next `.await` point; a run already on the
///   blocking pool proceeds to completion and writes its terminal status
///   (the lease-guarded finalize) *after* this guard has dropped — detached
///   from the guard's lifetime. After either graceful path, or a reclaimed
///   `Abandoned` handle, `Drop` aborts it (or finds `Joined` and no-ops).
pub struct EmbeddedWorker {
    /// [`LoopTask::Joined`] whenever nothing is available to take: either
    /// [`Self::stop_and_join`] or [`Self::release_and_stop`] has fully
    /// disposed of the handle, OR one of them is CURRENTLY holding it
    /// inside a live [`TakenHandle`] — [`TakenHandle::take`] writes `Joined`
    /// provisionally for the whole take window, before the handle is
    /// joined, aborted, or (if the taker's own future is itself cancelled
    /// first) restored as `Abandoned`. So `Joined` alone does not mean the
    /// handle has been fully disposed of; it means this slot has nothing
    /// left to hand out to a concurrent caller. `Drop` reads it to skip its
    /// own abort. Guarded by a `Mutex` rather than consuming `self` because
    /// both take `&self`: the owning `Database` binding wants to
    /// signal-and-await without giving up the guard itself (its `Drop` must
    /// still run at the connection's own end of life).
    handle: std::sync::Mutex<LoopTask>,
    /// The state shared with the loop task: stop, loop state, and the
    /// session's admission (phase + holder).
    shared: Arc<WorkerShared>,
    /// The session's [`HostAdmission`] — the phase this guard flips on
    /// DRAIN/RELEASE and the holder 2e reads; captured at spawn so RELEASE
    /// needs no session.
    admission: Arc<HostAdmission>,
    /// The catalog the `workers` row was upserted into, and the id it is
    /// keyed by — so stopping the loop (graceful or `Drop`) can delete the
    /// row rather than leave a claimant advertised until its `instances`
    /// row goes stale and cascades.
    catalog: Arc<Catalog>,
    instance_id: String,
    /// This process's `InstanceRegistration`, captured at spawn so RELEASE
    /// needs no session (same rationale as `keeper`/`writer_id` below) —
    /// this guard is the worker half's OTHER owner (alongside the loop task
    /// itself): every `set_worker_state`/`delete_worker` call this guard
    /// issues writes/clears the cell FIRST, so a keeper reregister racing a
    /// DRAIN or a RELEASE never re-upserts a `workers` row this process has
    /// already stopped claiming with.
    registration: Arc<InstanceRegistration>,
    /// The session's keeper (2b releases the `Job` holds it holds) and the
    /// session's result-store writer id (the linked building sweep's
    /// `writer_id` arm) — captured at spawn so RELEASE needs no session.
    keeper: Arc<LeaseKeeper>,
    writer_id: String,
    /// The heartbeat interval: the bound on 2b's keeper pass and on 2e's
    /// cooperative wait.
    heartbeat: Duration,
    /// The gauge sampler task; aborted with the loop on every stop path.
    sampler: std::sync::Mutex<Option<tokio::task::JoinHandle<()>>>,
    /// This guard's own generation id — the key
    /// [`HostAdmission::release_loop_claim`] releases, a compare-and-set so this guard can never free a SUCCESSOR's
    /// slot.
    loop_generation: u64,
}

impl EmbeddedWorker {
    /// Spawn a worker over `session` onto the current runtime, returning the
    /// guard that owns its task. The worker holds a [`Weak`] to the session, so
    /// it never keeps `session` alive; this guard stops it when the owner drops.
    ///
    /// Reads the lease/heartbeat/poll timing and kind selection from the
    /// session's `[worker]` configuration. Returns [`JammiError::Config`] if
    /// that timing (or `kinds`) violates the worker invariants — in the
    /// normal flow `JammiConfig::load` already validated the timing, so that
    /// half only surfaces for a hand-built config. Returns
    /// [`JammiError::FineTune`] when a loop already owns this session's
    /// single claim-loop slot (`HostAdmission::try_claim_loop`): "one claim
    /// loop per session" is structural — a second
    /// spawn is refused before any task exists, never a second loop whose
    /// hold-release blast radius the phase barrier alone would have to
    /// cover. Spawn a successor only after the prior guard has fully
    /// dropped.
    ///
    /// The loop task's own first statement upserts this process's `workers`
    /// row (`warming`, then `claiming` once the session's worker gate is
    /// open — see [`JobWorker::run_until`]); nothing detached races it.
    pub fn spawn(session: &Arc<InferenceSession>) -> Result<Self> {
        let worker = JobWorker::new(session)?;
        Self::spawn_worker(session, worker)
    }

    /// Spawn an already-built worker (used by [`Self::spawn`] and any test
    /// harness that needs explicit timing/kinds via
    /// [`JobWorker::with_intervals_and_kinds`]). See [`Self::spawn`]'s doc
    /// for the single-claim-loop-slot refusal.
    pub fn spawn_worker(session: &Arc<InferenceSession>, worker: JobWorker) -> Result<Self> {
        let admission = Arc::clone(session.host_admission());
        let Some((loop_generation, spawn_release_epoch)) = admission.try_claim_loop() else {
            return Err(JammiError::FineTune(format!(
                "a claim loop is already spawned on session instance {}; only one claim loop \
                 per session may run at once (spawn a successor only after the prior \
                 EmbeddedWorker's release_and_stop completes or its guard is dropped)",
                session.instance_id()
            )));
        };
        let shared = WorkerShared::new(
            Arc::clone(&admission),
            session.instance_id().to_string(),
            spawn_release_epoch,
        );
        let heartbeat = worker.intervals.heartbeat;
        let task_shared = Arc::clone(&shared);
        let handle = tokio::spawn(async move { worker.run_until(task_shared).await });
        let every = Duration::from_secs(session.inner_config().worker.metrics_sample_secs.max(1));
        let sampler = tokio::spawn(sample_loop(
            Arc::clone(session.catalog_arc()),
            Arc::downgrade(&shared),
            every,
        ));
        Ok(Self {
            handle: std::sync::Mutex::new(LoopTask::Running(handle)),
            shared,
            admission,
            catalog: Arc::clone(session.catalog_arc()),
            instance_id: session.instance_id().to_string(),
            registration: Arc::clone(session.instance_registration()),
            keeper: Arc::clone(session.lease_keeper()),
            writer_id: session.result_store().writer_id().to_string(),
            heartbeat,
            sampler: std::sync::Mutex::new(Some(sampler)),
            loop_generation,
        })
    }

    fn stop_sampler(&self) {
        if let Some(sampler) = self
            .sampler
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            sampler.abort();
        }
    }

    /// A weak handle on the loop's shared state, for observers (`/healthz`,
    /// `/metrics`) that must never keep the loop's state alive past its
    /// owner.
    pub fn shared(&self) -> Weak<WorkerShared> {
        Arc::downgrade(&self.shared)
    }

    /// Begin a DRAIN: phase `Draining` (a later RELEASE still wins), stop
    /// requested — the loop finishes its in-flight job and claims no more,
    /// and every gang rank held on this host ends with the `Drain` reason
    /// ([`HostAdmission::begin_drain`]) — and the `workers` row flipped to
    /// `draining` (best-effort). The join is [`Self::stop_and_join`]'s.
    pub async fn begin_drain(&self) {
        self.admission.begin_drain();
        self.shared.request_stop();
        // The same one-fact write as the loop's own `warming`/`claiming`
        // writes: preserve the cell's own `kinds`, flip only `state`, and
        // write the row by UPSERT with the cell reverted on failure. A cell
        // still `None` (the loop never wrote its first row) has no row to
        // flip and nothing to race — nothing to write.
        if let Some(mut facts) = self.registration.worker_snapshot() {
            facts.state = WorkerState::Draining;
            write_worker_facts(
                &self.catalog,
                &self.registration,
                &self.instance_id,
                facts,
                "draining",
            )
            .await;
        }
    }

    /// Gracefully stop this worker and wait for its loop task to actually
    /// return, rather than `Drop`'s non-blocking signal-and-abort.
    ///
    /// This is the DRAIN primitive an explicit, deterministic teardown (a
    /// bound `Database::close()`, a `SIGTERM`'d server) needs, distinct from
    /// `Drop`'s best-effort halt for an unattended process exit: it signals
    /// `stop`, then *awaits* the loop's terminal [`LoopState`] on the shared
    /// watch (the in-flight job's own terminal write lands first: the keeper
    /// stays alive, heartbeats continue, every epoch bundle lands), then
    /// joins the task, so a caller blocking on this observes the worker's
    /// actual quiescence, not merely "the signal was sent".
    ///
    /// **Bound on how long this takes to return**: immediate if the worker is
    /// between claim attempts (its idle sleep is `select!`ed against the stop
    /// watch, so it never waits out `intervals.idle_poll`), or the remaining
    /// duration of a job already claimed and running when this is called —
    /// such a job runs to its own terminal state (finalize included) before
    /// the loop task returns. This never force-cancels an in-flight training
    /// run; it only stops the worker from picking up further work and waits
    /// for it to notice.
    ///
    /// Idempotent: a second call (concurrent or sequential) finds no handle
    /// left to take and returns [`StopOutcome::NothingToJoin`] at once. Takes
    /// `&self` rather than consuming — the caller keeps the guard (and its
    /// `Drop`) alive; `Drop` checks the same `Mutex` and no-ops the abort
    /// once this has already taken the handle.
    ///
    /// Once the loop has returned, this process's `workers` row is deleted:
    /// a process that has stopped claiming must not show up in `ListWorkers`
    /// as a claimant (the `instances` row stays — the process itself is
    /// alive). Ordered after the loop task's own `upsert_worker` by
    /// construction: the task has exited before the delete runs.
    ///
    /// Cancel-safe: if THIS future is dropped before it returns (a caller's
    /// `tokio::select!` losing a race — e.g. a RELEASE signal preempting a
    /// DRAIN already awaiting this), the handle is never lost to a bare
    /// detach. `TakenHandle` restores it as `LoopTask::Abandoned`, and the
    /// next `release_and_stop` (2e) or `Drop` aborts it unconditionally.
    pub async fn stop_and_join(&self) -> Result<StopOutcome> {
        self.shared.request_stop();
        let Some(taken) = TakenHandle::take(&self.handle) else {
            return Ok(StopOutcome::NothingToJoin);
        };
        let mut state_rx = self.shared.state_receiver();
        // The sender lives in `self.shared`, so this cannot error.
        let _ = state_rx.wait_for(|s| *s != LoopState::Running).await;
        // A join error here is a task panic inside `run_until` (every fallible
        // step inside its loop is already caught and logged, not propagated as
        // a panic — the guard reports `Failed`) or the task having been
        // aborted by a concurrent `Drop` — either is a genuine defect worth
        // surfacing, not swallowing.
        taken
            .await
            .map_err(|e| JammiError::FineTune(format!("training worker task join error: {e}")))?;
        self.stop_sampler();
        // Cell before delete: the loop has fully returned, so
        // nothing else can race a re-set of the cell after this clear.
        self.registration.set_worker(None);
        self.catalog.delete_worker(&self.instance_id).await?;
        Ok(StopOutcome::Joined)
    }

    /// RELEASE — the one mechanism, identical on the library and the server
    /// (steps 2a–2c and 2e–2i below; there is no separate 2d, see 2a): hand every lease
    /// this loop holds back to the catalog and
    /// stop the loop at once, so a successor claims the in-flight job within
    /// one idle poll (never one lease window) and the job costs no attempt
    /// — WHEN every determinant of the returned [`ReleaseReport`] confirms.
    /// When one does not, whether a given lease was in fact handed back
    /// depends on which determinant degraded and is not universal; see
    /// [`ReleaseReport`]'s own fields and, for the exit-code consumer, the
    /// server's `ShutdownOutcome::ReleaseDegraded` doc.
    ///
    /// In order:
    ///
    /// * **2a** phase `Releasing` AND stop requested, together, as
    ///   `begin_drain` does for its own phase — from this instant `claim_next`
    ///   is initiated only after a read of `WorkerShared::admits_claim` that
    ///   returned `true` with no `.await` between that read and the call:
    ///   the loop reads it at the top of
    ///   the iteration AND again, immediately after `reclaim_expired_jobs`
    ///   returns, so a phase/stop flip that lands during that reclaim round
    ///   trip is still caught by the second read even though the first,
    ///   now-stale read had already admitted the iteration. The one
    ///   residual — a claim whose own catalog round trip is already in
    ///   flight when 2a runs, so no later read of this loop's own state can
    ///   observe it — runs into `register_job_hold_or_release`, which
    ///   self-releases instead of dispatching. The stop request belongs in
    ///   2a, not in a later step: deferring it past 2b/2c opens a window in
    ///   which the loop can reclaim and re-claim the same row under
    ///   `Releasing` without ever tripping the attempts cap (`attempts −
    ///   releases` nets to 0 on every self-release), spinning for up to one
    ///   keeper pass plus one sweep.
    /// * **2b** the keeper releases every `Job` hold it holds
    ///   (`LeaseKeeper::release_job_holds`, bounded by one heartbeat): the
    ///   row's lease goes NULL and the hold's `lost` flips while the hold
    ///   still exists, so a training thread bails at its next epoch boundary
    ///   WITHOUT writing a bundle — before any abort could drop the hold and
    ///   let a detached trainer write a doomed epoch into the shared
    ///   `_resume` prefix. `ResultTable` holds are never touched (2c). The
    ///   report's `holds` field: the pass either completed, in which
    ///   case a per-hold failure is a genuine, separately-counted
    ///   determinant, distinct from `not_required` (no such hold existed),
    ///   or it could not be confirmed to run at all, which is UNOBSERVED
    ///   rather than a count of zero.
    /// * **2c** sweep #1: the jobs sweep, then the jobs-linked building-table
    ///   sweep (`release_sweep`) — covering a claim or a building row that
    ///   committed after 2b snapshotted. The loop's own `ResultTable` hold
    ///   flips `lost` through the keeper's guarded renewal within one
    ///   heartbeat.
    /// * **2e** total match on the loop task's state: a handle `Abandoned`
    ///   by a previous stop attempt this process's own caller cancelled
    ///   (e.g. a DRAIN's `stop_and_join` preempted by this RELEASE) is
    ///   aborted unconditionally, since its true state is unknown and an
    ///   abort is always safe here. A `Running` handle whose holder is not
    ///   `JobRun` (`Free`: the loop is idle or inside `reclaim_expired_jobs`;
    ///   `ClaimProbe`: inside `claim_next` or the claim→hold prologue; a
    ///   `Rank`: an admitted gang rank is held beside an idle loop — never
    ///   loop work) is never aborted while a claim transaction can be in
    ///   flight; wait one heartbeat for the cooperative exit, joining the
    ///   task on it. On timeout, abort (a claim between
    ///   COMMIT and hold registration keeps its live lease and is recovered
    ///   by the expiry path with `attempts + 1`, never `failed`). `JobRun`:
    ///   the loop is inside a job under a hold, not inside `claim_next` —
    ///   abort now; the dropped future runs the hold's and the watcher's
    ///   `Drop`. The decision reads the holder KIND, never a count. `Joined` is a
    ///   no-op — nothing left to take; see `TakenHandle`'s own doc for why
    ///   this alone does not witness `stop_witnessed` (a concurrent `stop_and_join` may
    ///   still be mid-flight holding the handle).
    /// * **2f** observe the terminal [`LoopState`] (the in-task guard reports
    ///   on every path). `stop_witnessed` is `true` when 2e itself
    ///   resolved the task (joined or aborted, any arm) OR this observation
    ///   is a genuine watch-fired transition — never when it fell back to
    ///   the last-known proxy read on a timeout/closed channel, which alone
    ///   can read `Running` on a genuine abort whose guard has not published
    ///   yet. Since 2a's gate stops an idle or between-claims loop at
    ///   once, this `wait_for` usually finds the watch ALREADY at its
    ///   terminal value by the time it is polled (an idle loop exits before
    ///   2b/2c even run) rather than observing a live transition; `wait_for`
    ///   treats an already-satisfied value as witnessed, same as a live one,
    ///   so `stop_witnessed` is unaffected.
    /// * **2g** sweep #2, unconditionally — idempotent, catches a claim or a
    ///   building row that committed after sweep #1.
    /// * **2h** delete the `workers` row — AFTER sweep #2, so the row outlives
    ///   this instance's last lease write.
    /// * **2i** release the session's claim-loop slot, so a successor
    ///   `spawn`/`spawn_worker` is admitted without waiting for this guard's
    ///   drop.
    ///
    /// Bounded by 2 × heartbeat plus the keeper's pass, never a hang. A
    /// catalog error inside any statement is logged and the arm continues
    /// (the affected lease falls to the expiry path).
    pub async fn release_and_stop(&self) -> Result<ReleaseReport> {
        // 2a — phase and stop together, in the same statement pair,
        // mirroring `begin_drain`'s own shape: `begin_release` carries a syntactic
        // `.await` (its own doc: the test-only park pinning the phase-
        // flip/epoch-bump order), but that inner future resolves within
        // this SAME poll — no genuine yield — unless a test has armed
        // `ParkPoint::BeginReleaseBetweenFlipAndBump` for THIS instance, so
        // a poll-once test proves both setters resolve this pair before
        // their first genuine yield, and exit latency after either flip is
        // bounded by the in-flight job, never by `idle_poll`.
        self.admission.begin_release().await;
        self.shared.request_stop();
        // 2b
        let holds = match self.keeper.release_job_holds(self.heartbeat).await {
            Ok(hr) => HoldReleaseOutcome::Observed(hr),
            Err(e) => {
                tracing::warn!(error = %e, "RELEASE: the keeper's per-hold release pass failed");
                HoldReleaseOutcome::Unobserved
            }
        };
        // 2c
        let sweep_one = release_sweep(&self.catalog, &self.instance_id, &self.writer_id).await;
        // 2e
        #[cfg(feature = "test-hooks")]
        loop_test_hooks::fire(&self.instance_id, loop_test_hooks::Rendezvous::ReleaseAt2e);
        // The release decision reads the HOLDER KIND: `JobRun`
        // means the loop is inside a job under a hold — abort now; anything
        // else (`Free`, a `ClaimProbe` whose claim transaction or prologue
        // may be in flight, or a `Rank` — which is never loop work, the
        // loop being idle beside it) waits one heartbeat for the
        // cooperative exit.
        let holder = self.admission.holder();
        let mut state_rx = self.shared.state_receiver();
        // `stop_witnessed`'s first disjunct: whether THIS call resolved the task with
        // certainty (joined, or aborted on any arm). `NothingToTake` — the
        // handle was already taken by a concurrent caller — is NOT itself a
        // certain witness (see `TakenHandle`'s own doc: `Joined` alone does
        // not mean the handle was fully disposed of, only that this slot had
        // nothing left to hand out); `stop_witnessed` then falls through to the second
        // disjunct below (2f's OBSERVED terminal state, never its fallback).
        let stop_resolved = if let Some(taken) = TakenHandle::take(&self.handle) {
            if taken.reclaimed() {
                // A previous stop attempt (this process's own caller
                // cancelled it) left the task's fate unresolved rather than
                // losing it to a bare `JoinHandle` drop. Abort it
                // unconditionally — see `LoopTask::Abandoned`'s doc for why
                // this is always safe.
                tracing::warn!(
                    "RELEASE: a previous stop attempt was itself cancelled before observing \
                     the loop's terminal state; aborting the loop task now"
                );
                taken.abort_now();
            } else if holder != Holder::JobRun {
                // The watch `Ref` is dropped before the join below: a guard
                // held across an `.await` would make this future `!Send`.
                let exited = tokio::time::timeout(
                    self.heartbeat,
                    state_rx.wait_for(|s| *s != LoopState::Running),
                )
                .await
                .is_ok();
                if exited {
                    if let Err(e) = taken.await {
                        tracing::error!(error = %e, "RELEASE: the loop task ended with a join error");
                    }
                } else {
                    tracing::warn!(
                        bound = ?self.heartbeat,
                        "RELEASE: the loop did not exit within one heartbeat; aborting it"
                    );
                    taken.abort_now();
                }
            } else {
                taken.abort_now();
            }
            true
        } else {
            false
        };
        // 2f
        let observed = tokio::time::timeout(
            self.heartbeat,
            state_rx.wait_for(|s| *s != LoopState::Running),
        )
        .await
        .map(|r| r.map(|state| *state));
        let (loop_state, state_witnessed) = match observed {
            Ok(Ok(state)) => (state, true),
            Ok(Err(_)) | Err(_) => {
                let observed = self.shared.loop_state();
                tracing::warn!(
                    ?observed,
                    "RELEASE: the loop's terminal state was not observed within one heartbeat"
                );
                (observed, false)
            }
        };
        // Resolved with certainty by 2e, OR the terminal state above
        // was itself an OBSERVED transition (never the fallback proxy read
        // alone — a bare `loop_state != Running` comparison cannot tell a
        // genuine abort whose guard has not published yet from a live loop).
        let stop_witnessed = stop_resolved || state_witnessed;
        // 2g
        let sweep_two = release_sweep(&self.catalog, &self.instance_id, &self.writer_id).await;
        // 2h
        self.stop_sampler();
        // Cell before delete — same as `stop_and_join`.
        self.registration.set_worker(None);
        self.catalog.delete_worker(&self.instance_id).await?;
        // 2i: the loop task is joined or aborted
        // by 2e above — this call is what it means for RELEASE to
        // "complete" the guard's slot-holding lifetime — so a successor
        // `spawn`/`spawn_worker` on this SAME session is admitted from this
        // instant, without waiting for THIS guard to be dropped. `Drop`'s
        // own release later is a compare-and-set against `loop_generation`
        // and finds the slot already free (or already reclaimed by a
        // successor), so it never double-releases or steals a successor's
        // slot.
        self.admission.release_loop_claim(self.loop_generation);
        Ok(ReleaseReport {
            loop_state,
            holds,
            stop_witnessed,
            sweep_one,
            sweep_two,
        })
    }
}

impl Drop for EmbeddedWorker {
    /// Signal the loop to stop and abort its task. This halts claiming of new
    /// jobs; an in-flight `spawn_blocking` training run is not aborted by this —
    /// it runs to completion and writes its terminal status post-drop (see the
    /// type doc).
    ///
    /// No-ops the abort when [`Self::stop_and_join`] or
    /// [`Self::release_and_stop`] already fully joined the task (the slot
    /// reads `LoopTask::Joined`) — there is nothing left to abort, and
    /// aborting a handle that already returned would be a silent no-op
    /// anyway, but the explicit check keeps the intent legible. A
    /// `LoopTask::Abandoned` handle — left behind by a stop attempt this
    /// process's own caller cancelled before it could join or abort it
    /// — is aborted here too: total match, nothing is ever silently lost to
    /// a bare `JoinHandle` drop (which would DETACH rather than abort).
    ///
    /// Also releases the session's single claim-loop slot
    /// (`HostAdmission::release_loop_claim`),
    /// compare-and-set against this guard's OWN `loop_generation` — a
    /// no-op when `release_and_stop` already released it (2i), or when a
    /// successor has since claimed a NEW generation, so this can never
    /// steal a successor's slot. `Drop` runs exactly once per guard
    /// regardless of whether `stop_and_join`, `release_and_stop`, both (one
    /// cancelled), or neither ran first, so this is the backstop for every
    /// stop path OTHER than a completed `release_and_stop`.
    fn drop(&mut self) {
        self.shared.request_stop();
        self.stop_sampler();
        self.admission.release_loop_claim(self.loop_generation);
        let task = std::mem::replace(
            &mut *self
                .handle
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            LoopTask::Joined,
        );
        if let LoopTask::Running(handle) | LoopTask::Abandoned(handle) = task {
            handle.abort();
            // Cell before delete, synchronous — `Drop` cannot
            // `.await` the delete below, but clearing the cell needs no
            // await, so it happens unconditionally here rather than only
            // once the (possibly never-scheduled) spawned task below runs.
            self.registration.set_worker(None);
            // The loop is gone, so the claimant row must go too. `Drop` is
            // synchronous: the delete rides a detached task on the current
            // runtime when there is one (the embedded engine's own runtime
            // is still up at this point); with no runtime to spawn onto the
            // row is left to the `instances` staleness cascade. A `Drop` that
            // lands while the task's own `upsert_worker` is still in flight
            // can leave a phantom `warming` row — the same cascade covers it.
            if let Ok(rt) = tokio::runtime::Handle::try_current() {
                let catalog = Arc::clone(&self.catalog);
                let instance_id = self.instance_id.clone();
                rt.spawn(async move {
                    if let Err(e) = catalog.delete_worker(&instance_id).await {
                        tracing::warn!(error = %e, "failed to delete this process's `workers` row on worker drop");
                    }
                });
            }
        }
    }
}

/// Test-only rendezvous inside the claim loop (`feature = "test-hooks"`;
/// mirrors `crate::jobs::compute_test_hooks`): a test parks the loop at a
/// documented point between its claim and its hold, or observes the exact
/// instant RELEASE reaches its 2e decision, so the shutdown arms are pinned
/// against the mechanism rather than raced against a wall clock. A
/// per-instance counter records every `claim_next` call the loop makes, so a
/// test can snapshot it beside a rendezvous and assert a delta of zero
/// across a window it controls. No production path observes anything here
/// beyond the `maybe_park` / `fire` / `record_claim_next` calls, which
/// return at once (or add one to a counter nothing reads) when nothing is
/// armed.
#[cfg(feature = "test-hooks")]
pub mod loop_test_hooks {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex, OnceLock, PoisonError};

    use tokio::sync::Notify;

    /// Where the loop parks.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ParkPoint {
        /// Inside `register_job_hold_or_release`, after the claim committed
        /// and before the hold is registered — the claim→hold prologue, on
        /// both the fine-tune and the compute path.
        BeforeHold,
        /// Inside `JobWorker::run_placed_gang`, immediately after this
        /// run's `WorkerShared` birth release-epoch is read and before
        /// `HostAdmission::probe_claim` itself runs — the window a RELEASE
        /// landing between the epoch snapshot and `probe_claim`'s own phase
        /// check must still be caught in, by `probe_claim` refusing typed
        /// (the epoch read precedes `probe_claim()`, never follows it).
        PlacedGangBeforeProbeClaim,
        /// Inside `JobWorker::run_placed_gang`, immediately after
        /// `HostAdmission::probe_claim` succeeds, before
        /// `Catalog::transfer_claim`/`Catalog::get_job` — the two catalog
        /// round trips a RELEASE landing during them must still be caught
        /// across.
        PlacedGangBeforeTransfer,
        /// Inside `HostAdmission::begin_release`, between the phase flip
        /// (→ `Releasing`) and the release-epoch bump — pins the load-
        /// bearing order the two-catch lattice in `run_placed_gang`'s own
        /// doc and `WorkerShared::for_single_run`'s depends on: while
        /// parked here the phase is already `Releasing` (a concurrent
        /// `probe_claim` refuses) but the epoch is not yet bumped (a
        /// concurrent birth-epoch snapshot would not yet contain it).
        /// Keyed by the owning `HostAdmission`'s `registry().instance_id`
        /// (`arm`'s key doubles as either a job id or an instance id — no
        /// job is claimed yet at this point).
        BeginReleaseBetweenFlipAndBump,
    }

    struct Armed {
        job_id: String,
        point: ParkPoint,
        parked: Arc<AtomicBool>,
        parked_notify: Arc<Notify>,
        released: Arc<AtomicBool>,
        release_notify: Arc<Notify>,
    }

    fn armed() -> &'static Mutex<Vec<Armed>> {
        static ARMED: OnceLock<Mutex<Vec<Armed>>> = OnceLock::new();
        ARMED.get_or_init(|| Mutex::new(Vec::new()))
    }

    /// The test's side of one armed park: wait for the loop to arrive, then
    /// let it continue. Dropping the handle without releasing leaves the
    /// loop parked — release it explicitly (or abort the task).
    pub struct ParkHandle {
        parked: Arc<AtomicBool>,
        parked_notify: Arc<Notify>,
        released: Arc<AtomicBool>,
        release_notify: Arc<Notify>,
    }

    impl ParkHandle {
        /// Resolve once the loop has reached the park point.
        pub async fn wait_parked(&self) {
            while !self.parked.load(Ordering::SeqCst) {
                self.parked_notify.notified().await;
            }
        }

        /// Whether the loop has reached the park point (non-blocking).
        pub fn is_parked(&self) -> bool {
            self.parked.load(Ordering::SeqCst)
        }

        /// Let the parked loop continue.
        pub fn release(&self) {
            self.released.store(true, Ordering::SeqCst);
            self.release_notify.notify_one();
        }
    }

    /// Arm one park for the next hold registration of `job_id` at `point`.
    /// One-shot: the park disarms as soon as the loop takes it.
    pub fn arm(job_id: &str, point: ParkPoint) -> ParkHandle {
        let parked = Arc::new(AtomicBool::new(false));
        let parked_notify = Arc::new(Notify::new());
        let released = Arc::new(AtomicBool::new(false));
        let release_notify = Arc::new(Notify::new());
        armed()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Armed {
                job_id: job_id.to_string(),
                point,
                parked: Arc::clone(&parked),
                parked_notify: Arc::clone(&parked_notify),
                released: Arc::clone(&released),
                release_notify: Arc::clone(&release_notify),
            });
        ParkHandle {
            parked,
            parked_notify,
            released,
            release_notify,
        }
    }

    pub(super) async fn maybe_park(job_id: &str, point: ParkPoint) {
        let taken = {
            let mut list = armed().lock().unwrap_or_else(PoisonError::into_inner);
            list.iter()
                .position(|a| a.job_id == job_id && a.point == point)
                .map(|i| list.remove(i))
        };
        let Some(armed) = taken else {
            return;
        };
        armed.parked.store(true, Ordering::SeqCst);
        armed.parked_notify.notify_one();
        while !armed.released.load(Ordering::SeqCst) {
            armed.release_notify.notified().await;
        }
    }

    struct ArmedInstance {
        instance_id: String,
        parked: Arc<AtomicBool>,
        parked_notify: Arc<Notify>,
        released: Arc<AtomicBool>,
        release_notify: Arc<Notify>,
    }

    fn instance_armed() -> &'static Mutex<Vec<ArmedInstance>> {
        static ARMED: OnceLock<Mutex<Vec<ArmedInstance>>> = OnceLock::new();
        ARMED.get_or_init(|| Mutex::new(Vec::new()))
    }

    /// Arm a one-shot park for the next time the loop of `instance_id`
    /// reaches the post-reclaim, pre-claim gate re-read — immediately after
    /// `reclaim_expired_jobs` returns and before `WorkerShared::admits_claim`'s
    /// second read — the reclaim-window instant a RELEASE must still be
    /// caught in. Keyed by `instance_id`
    /// (unlike [`arm`], there is no claimed job yet at this point).
    pub fn arm_after_reclaim(instance_id: &str) -> ParkHandle {
        let parked = Arc::new(AtomicBool::new(false));
        let parked_notify = Arc::new(Notify::new());
        let released = Arc::new(AtomicBool::new(false));
        let release_notify = Arc::new(Notify::new());
        instance_armed()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(ArmedInstance {
                instance_id: instance_id.to_string(),
                parked: Arc::clone(&parked),
                parked_notify: Arc::clone(&parked_notify),
                released: Arc::clone(&released),
                release_notify: Arc::clone(&release_notify),
            });
        ParkHandle {
            parked,
            parked_notify,
            released,
            release_notify,
        }
    }

    pub(super) async fn maybe_park_after_reclaim(instance_id: &str) {
        let taken = {
            let mut list = instance_armed()
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            list.iter()
                .position(|a| a.instance_id == instance_id)
                .map(|i| list.remove(i))
        };
        let Some(armed) = taken else {
            return;
        };
        armed.parked.store(true, Ordering::SeqCst);
        armed.parked_notify.notify_one();
        while !armed.released.load(Ordering::SeqCst) {
            armed.release_notify.notified().await;
        }
    }

    fn claim_next_counts() -> &'static Mutex<HashMap<String, u64>> {
        static COUNTS: OnceLock<Mutex<HashMap<String, u64>>> = OnceLock::new();
        COUNTS.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// Record one `claim_next` call by the loop of `instance_id` — called
    /// from `run_until`'s loop body, immediately before it awaits
    /// `Catalog::claim_next`. Keyed per instance so sibling tests in one
    /// binary never read each other's counts.
    pub(super) fn record_claim_next(instance_id: &str) {
        *claim_next_counts()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry(instance_id.to_string())
            .or_insert(0) += 1;
    }

    /// How many times the loop of `instance_id` has called `claim_next`
    /// since the process started (0 if it never has). A test snapshots this
    /// beside a rendezvous it controls and compares the delta across a
    /// window it also controls — never against a wall clock.
    pub fn claim_next_calls(instance_id: &str) -> u64 {
        claim_next_counts()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(instance_id)
            .copied()
            .unwrap_or(0)
    }

    fn panic_armed() -> &'static Mutex<Vec<String>> {
        static ARMED: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
        ARMED.get_or_init(|| Mutex::new(Vec::new()))
    }

    /// Make the loop task of `instance_id` panic at the top of its next tick
    /// — the way a defect inside the loop would kill it — so a test can
    /// prove the exit guard reports `LoopState::Failed` and `/healthz` reads
    /// 503. One-shot.
    pub fn arm_panic_at_next_tick(instance_id: &str) {
        panic_armed()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(instance_id.to_string());
    }

    pub(super) fn maybe_panic(instance_id: &str) {
        let armed = {
            let mut list = panic_armed().lock().unwrap_or_else(PoisonError::into_inner);
            list.iter()
                .position(|i| i == instance_id)
                .map(|i| list.remove(i))
        };
        if armed.is_some() {
            panic!("claim loop: task killed by test hook");
        }
    }

    /// A notify-only rendezvous: RELEASE fires it and never waits.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Rendezvous {
        /// The first statement of `EmbeddedWorker::release_and_stop`'s 2e —
        /// the sweeps have returned and the `in_flight` decision is about to
        /// be read. A test that parked the loop before COMMIT unparks it on
        /// THIS signal, never on a wall clock measured from the sweeps.
        ReleaseAt2e,
    }

    struct ArmedRendezvous {
        instance_id: String,
        which: Rendezvous,
        fired: Arc<AtomicBool>,
        notify: Arc<Notify>,
    }

    fn rendezvous() -> &'static Mutex<Vec<ArmedRendezvous>> {
        static ARMED: OnceLock<Mutex<Vec<ArmedRendezvous>>> = OnceLock::new();
        ARMED.get_or_init(|| Mutex::new(Vec::new()))
    }

    /// The test's side of an armed rendezvous.
    pub struct RendezvousHandle {
        fired: Arc<AtomicBool>,
        notify: Arc<Notify>,
    }

    impl RendezvousHandle {
        /// Resolve once RELEASE has fired the rendezvous.
        pub async fn wait_fired(&self) {
            while !self.fired.load(Ordering::SeqCst) {
                self.notify.notified().await;
            }
        }

        /// Whether the rendezvous has fired (non-blocking).
        pub fn has_fired(&self) -> bool {
            self.fired.load(Ordering::SeqCst)
        }
    }

    /// Arm `which` for the next RELEASE of the worker whose `claimed_by`
    /// identity is `instance_id`. One-shot per handle; keyed so sibling tests
    /// in one binary never fire each other's.
    pub fn arm_rendezvous(instance_id: &str, which: Rendezvous) -> RendezvousHandle {
        let fired = Arc::new(AtomicBool::new(false));
        let notify = Arc::new(Notify::new());
        rendezvous()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(ArmedRendezvous {
                instance_id: instance_id.to_string(),
                which,
                fired: Arc::clone(&fired),
                notify: Arc::clone(&notify),
            });
        RendezvousHandle { fired, notify }
    }

    /// Fire every handle armed for (`instance_id`, `which`). Never parks.
    pub(super) fn fire(instance_id: &str, which: Rendezvous) {
        let taken: Vec<ArmedRendezvous> = {
            let mut list = rendezvous().lock().unwrap_or_else(PoisonError::into_inner);
            let (hit, rest): (Vec<_>, Vec<_>) = list
                .drain(..)
                .partition(|a| a.which == which && a.instance_id == instance_id);
            *list = rest;
            hit
        };
        for armed in taken {
            armed.fired.store(true, Ordering::SeqCst);
            armed.notify.notify_one();
        }
    }

    /// A discrete, test-observable moment on ONE job's attempt — never a
    /// poll tick, never a wall-clock guess at when one MIGHT have happened.
    /// Generalises [`Rendezvous`]'s notify-only shape (never blocks the
    /// producer, unlike [`ParkPoint`]/[`arm`]) over an ENUM keyed by
    /// `job_id` instead of `instance_id`, so a new observable site extends
    /// this ONE `arm_observed`/`fire_observed` pair rather than growing a
    /// bespoke pair of its own (a `timeout`/deadline-loop bound that races a
    /// training-progress event's real cadence, rather than observing the
    /// event itself, is flaky by construction).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Event {
        /// [`spawn_cancel_request_watcher`] has just read
        /// `cancel_requested = true` off the row and flipped
        /// `cancel_requested_seen` — never merely "a poll tick happened" (a
        /// tick that reads `false` does not fire this).
        CancelObserved,
        /// The training loop has just written a durable resume checkpoint
        /// (`TrainingLoop::save_epoch_checkpoint`'s `stage_checkpoint`
        /// call returned `Ok`) for `job_id` — the earliest instant a test
        /// may observe `fetch_newest_checkpoint` return `Some` for it.
        ResumeCheckpointWritten,
        /// The rank body identified by the carried rank number is ABOUT TO
        /// call `discover_resume` for `job_id` — fired unconditionally,
        /// immediately before that call, so it still fires even when
        /// `discover_resume` goes on to return `Err` (a corrupted resume
        /// bundle, `artifact.rs`'s hard-error contract) or the function
        /// returns early via `?`. The rank-attributed positive proof that
        /// THIS rank's body reached the resume seam — needed because the
        /// `_checkpoints/` bundle is job-scoped, read independently by every
        /// rank, so a job's terminal `failed` row with a resume-related
        /// error cannot by itself attribute which rank's read produced it
        /// (a `Peer` gang's rank-0 coordinator and its member's rank 1 both
        /// read the identical bundle and would fail identically).
        ResumeAttempted(u32),
    }

    struct ArmedEvent {
        job_id: String,
        which: Event,
        fired: Arc<AtomicBool>,
        notify: Arc<Notify>,
    }

    fn events() -> &'static Mutex<Vec<ArmedEvent>> {
        static ARMED: OnceLock<Mutex<Vec<ArmedEvent>>> = OnceLock::new();
        ARMED.get_or_init(|| Mutex::new(Vec::new()))
    }

    /// The test's side of an armed job event.
    pub struct Observed {
        fired: Arc<AtomicBool>,
        notify: Arc<Notify>,
    }

    impl Observed {
        /// Resolve once the event has fired. A test bounds this call with
        /// its OWN generous backstop — this method itself never times out.
        pub async fn wait_fired(&self) {
            while !self.fired.load(Ordering::SeqCst) {
                self.notify.notified().await;
            }
        }
    }

    /// Arm `which` for the next occurrence on `job_id`. One-shot; keyed so
    /// sibling tests in one binary never fire each other's.
    pub fn arm_observed(job_id: &str, which: Event) -> Observed {
        let fired = Arc::new(AtomicBool::new(false));
        let notify = Arc::new(Notify::new());
        events()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(ArmedEvent {
                job_id: job_id.to_string(),
                which,
                fired: Arc::clone(&fired),
                notify: Arc::clone(&notify),
            });
        Observed { fired, notify }
    }

    /// Fire every handle armed for (`job_id`, `which`). Never parks, never
    /// blocks the caller — the opposite contract from [`maybe_park`]:
    /// firing is a notification, not a rendezvous the producer waits on.
    /// `pub(crate)` (not `pub(super)`): [`crate::fine_tune::trainer`] is a
    /// SIBLING module of `worker`, not a descendant, and is the only other
    /// caller ([`Event::ResumeCheckpointWritten`]'s fire point).
    pub(crate) fn fire_observed(job_id: &str, which: Event) {
        let taken: Vec<ArmedEvent> = {
            let mut list = events().lock().unwrap_or_else(PoisonError::into_inner);
            let (hit, rest): (Vec<_>, Vec<_>) = list
                .drain(..)
                .partition(|a| a.which == which && a.job_id == job_id);
            *list = rest;
            hit
        };
        for armed in taken {
            armed.fired.store(true, Ordering::SeqCst);
            armed.notify.notify_one();
        }
    }
}

/// The reconstructed inputs for a LoRA fine-tune run — the per-kind data
/// loader plus the task and base-model/config common bits. Bundled so the
/// shared [`JobWorker::train_fine_tune`] tail takes one job-shaped argument
/// rather than a long positional list.
struct FineTuneRun {
    task: ModelTask,
    common: TrainingCommon,
    /// What the training loop trains from — either an already in-memory
    /// [`crate::fine_tune::source::TrainingSource::Resident`] loader or a
    /// [`crate::fine_tune::source::TrainingSource::Streamed`] source.
    source: crate::fine_tune::source::TrainingSource,
    /// The identity of the model this run produces.
    materialization: FineTuneMaterialization,
}

/// The `ProducingDescriptor::FineTune`-specific inputs a training-set-trained
/// run's materialization is built from — everything [`FineTuneRun`]'s shared
/// fields (`task`, `common`) do not already carry. Both LoRA kinds train from
/// a materialised training-set table and canonicalise their spec through
/// [`crate::fine_tune::spec::fine_tune_spec_canonical`], so one descriptor
/// covers both.
struct FineTuneMaterializationSource {
    /// The spec's output-affecting fields, canonically encoded — the kind is
    /// part of the encoding, so a graph sample and a projection never share
    /// an identity.
    spec_canonical: String,
    /// The materialised `TrainingSet` table's own [`DefinitionHash`], hex.
    training_set_definition_hash: String,
    /// The materialised `TrainingSet` table's own artifact digest, hex — the
    /// [`jammi_db::store::manifest::InputAnchor::result_digest`] anchor's value.
    training_set_artifact_digest: String,
    /// The materialised `TrainingSet` table's committed row count.
    training_set_row_count: u64,
}

/// A `TrainingSpec::FineTune` run's materialization identity — the
/// `ProducingDescriptor::FineTune` descriptor, the environment and the input
/// anchors — built by [`fine_tune_materialization`] BEFORE the trainer runs.
/// Under `CachePolicy::Use` its definition hash is what the reuse probe
/// ([`JobWorker::reuse_published_model`]) looks up; on a fresh run the
/// worker writes `materialization.json` from it LAST into the published
/// bundle (before the finalize CAS) and records its summary on the artifact
/// row.
pub(crate) struct FineTuneMaterialization {
    descriptor: Box<jammi_db::store::manifest::ProducingDescriptor>,
    env: jammi_db::store::manifest::MaterializationEnv,
    inputs: Vec<jammi_db::store::manifest::InputAnchor>,
}

impl FineTuneMaterialization {
    /// The definition hash a run of this identity records — the same fold
    /// the published manifest carries, so a probe keyed on it matches
    /// exactly the bundles a fresh run would have produced.
    fn definition_hash(&self) -> Result<jammi_db::store::manifest::DefinitionHash> {
        jammi_db::store::manifest::MaterializationManifest::definition_of(
            &self.descriptor,
            &self.env,
        )
        .map_err(jammi_db::store::manifest_to_jammi)
    }
}

/// Build a `TrainingSpec::FineTune` run's materialization identity from the
/// materialised training set (`src`), the spec's task and common knobs, and
/// the topology the run will execute at — while holding the base model's
/// cache guard, because the identity folds the loaded model's
/// backend/precision/digest/quantization (the SAME uniform path
/// `pipeline::embedding`'s `embedding_definition` uses for the model it
/// invokes).
///
/// The fused-kernel admission profile is folded HERE, ex ante — before
/// training runs, alongside every other materialization fact. It is never an
/// OBSERVED per-op dispatch outcome read after training: a `DefinitionHash`
/// is what a lookup for already-existing work is keyed on, BEFORE the work
/// runs, so a value knowable only after training would make that lookup
/// impossible for the very run it describes — see
/// `jammi_kernels::admission::render_kernel_admission_profile`'s own doc.
/// Every fact folded is EX ANTE and by construction: this crate's own
/// compiled features (`jammi_kernels::admission::BUILD_FACTS`), the process's
/// `admission_mode()`, the `JAMMI_KERNELS_DISABLE` set
/// (`disabled_ops_requested()`), and this job's backbone dtype class.
///
/// The dtype folded is `common.config.backbone_dtype` — the SAME value
/// `probe_acceleration`'s own `dtype_class_of(backbone_dtype)` call resolves
/// the acceleration report's dtype class from — never
/// `guard.model.compute_precision()` (the loaded model's own ON-DISK weight
/// dtype, a DIFFERENT axis: the trainer's `VarBuilder` and LoRA adapters are
/// always `F32` regardless of `backbone_dtype`, and `guard.model` can be
/// loaded at `F32` while `backbone_dtype` casts the forward activations to
/// `Bf16`/`F16`). Passing the loaded model's own precision instead renders an
/// f16-backbone CPU job's profile at `dtype_class=F32` (tiny_bert's own
/// fixture loads at F32 on disk), so `cast_scale`/`cast_add` read `n/a` and
/// `JAMMI_KERNELS_DISABLE=cast_scale_f16_f32` never moves that job's
/// `DefinitionHash` at all.
///
/// NO input anchor is recorded for the `FineTune` materialization: one would
/// be both a FALSE ATTESTATION and REDUNDANT.
///
/// False attestation: an anchor would pair the fine-tune's own registered
/// SOURCE name (`src.source`, e.g. `"training"` — a long-lived, mutable
/// relation) with the training-set TABLE's digest (an ephemeral, single-use
/// materialization `materialize_projection` never reuses across calls;
/// anchoring on the table's own fresh, never-repeating name would defeat
/// reuse). `AnchorKind::ResultDigest`'s semantics
/// (`crate::pipeline::recompute::reresolve_recorded_anchor`) are "resolve
/// `source` as a `result_tables` row and pin its CURRENT digest" —
/// `src.source` is not that table, so a resolver that read this anchor would
/// pin the wrong relation's current state under the training-set table's old
/// digest.
///
/// Redundant: nothing needs the anchor to DISCRIMINATE.
/// `ProducingDescriptor::FineTune::training_set_artifact_digest` (folded into
/// the descriptor) is the SAME digest such an anchor would carry — two
/// fine-tunes over different training-set content already hash differently
/// without an anchor's help — and it is what makes the empty anchor set an
/// honest, PINNED request: the model's one input is named by content, so an
/// equal definition hash proves equal inputs. No CONSUMER ever reads a
/// FineTune-recorded anchor either: the model-kind's OWN replay policy is
/// retrain (`recompute_fine_tune`), which never reads a recorded anchor at
/// all — `reresolve_recorded_anchor` is reached only from the TrainingSet-
/// table replay arm, over THAT table's own separately-recorded anchors.
async fn fine_tune_materialization(
    session: &Arc<InferenceSession>,
    src: &FineTuneMaterializationSource,
    task: ModelTask,
    common: &TrainingCommon,
    topology: TopologyDecision,
) -> std::result::Result<FineTuneMaterialization, WorkerJobError> {
    let model_source = ModelSource::parse(&common.base_model);
    let guard = session
        .model_cache()
        .get_or_load(&model_source, task, None)
        .await
        .map_err(WorkerJobError::from)?;
    let canonical_model_id = model_source.to_string();
    let kernel_admission_profile = jammi_kernels::admission::render_kernel_admission_profile(
        dtype_class_of(common.config.backbone_dtype),
        jammi_kernels::admission::admission_mode(),
        &jammi_kernels::admission::disabled_ops_requested(),
    );
    let env = jammi_db::store::manifest::MaterializationEnv::new(
        session.compute_device(),
        vec![jammi_db::store::manifest::ModelIdentity {
            model_id: canonical_model_id.clone(),
            backend: guard.model.backend_kind().to_string(),
            compute_precision: guard.model.compute_precision(),
            content_digest: guard.model.content_digest().map_err(WorkerJobError::from)?,
            quantization: guard.model.quantization(),
        }],
    )
    .with_kernel_admission_profile(kernel_admission_profile);
    let descriptor = jammi_db::store::manifest::ProducingDescriptor::FineTune {
        training_set_definition_hash: src.training_set_definition_hash.clone(),
        training_set_artifact_digest: src.training_set_artifact_digest.clone(),
        training_set_row_count: src.training_set_row_count,
        spec_canonical: src.spec_canonical.clone(),
        spec_schema_version: crate::fine_tune::spec::FINE_TUNE_SPEC_SCHEMA_VERSION,
        base_model_id: canonical_model_id,
        world_size: common.world_size,
        // The topology THIS run executes at — the collective it reduces
        // over and the ranks this host runs — read off the decided
        // `topology`, never off the `[worker]` selection (`auto` resolves
        // differently on different hosts, and `[worker] local_ranks` is a
        // capacity, not what the run used); recorded alongside (never
        // instead of) the job's own declared `world_size` above.
        collective: topology.collective_token().to_string(),
        local_ranks: topology.host_ranks(),
    };
    Ok(FineTuneMaterialization {
        descriptor: Box::new(descriptor),
        env,
        inputs: Vec::new(),
    })
}

/// A completed attempt that trained nothing: its output model row references
/// an artifact another job published under the same definition. It holds the
/// artifact's reference only — never a claim on its bytes.
struct ReusedModel {
    model_id: String,
    artifact: jammi_db::catalog::artifact_repo::ArtifactRef,
}

/// What one attempt of [`JobWorker::run_spec`] produced.
enum AttemptOutput {
    /// A trained bundle awaiting the worker's publish-and-finalize.
    Trained(Box<TrainedArtifact>),
    /// The attempt is already `completed`: the reuse probe hit and the same
    /// transaction attached the output model to the published artifact.
    Reused(ReusedModel),
}

/// A successful training run's output, awaiting the worker's unified
/// publish-and-finalize.
///
/// Each kind's training path writes its final artifact files into a local
/// tempdir ([`Self::dir`]) and describes the catalog model row to register
/// ([`Self::register`]) — but does **not** publish to the object store or touch
/// the catalog terminal state. The worker reads the files out of the tempdir,
/// stages them in the artifact store under a unique per-attempt prefix, and
/// runs the single lease-guarded finalize, which publishes the artifact and
/// writes the model row referencing it — the catalog row is the commit.
/// `metrics` is the run-metrics JSON the finalize records (the fine-tune loop's loss/step/timing
/// detail; `None` for a kind that records none beyond the terminal flip).
pub struct TrainedArtifact {
    /// Local tempdir holding the final artifact files, removed on drop after
    /// the worker has published its contents under a fresh, attempt-unique
    /// prefix (see the `materialization` field below).
    pub dir: tempfile::TempDir,
    /// The catalog model row to register for this artifact.
    pub register: ModelRegistration,
    /// Run-metrics JSON recorded in the finalize CAS, or `None`.
    pub metrics: Option<String>,
    /// The training loop's RETAINED epoch checkpoints: each entry is
    /// `(epoch_index, claim)`, the claim on the bundle the TRAINER already
    /// wrote that epoch's checkpoint to
    /// (`{job_id}/_checkpoints/{attempt}/epoch_{N}/`) — a full loadable
    /// adapter beside the run's resume state. Empty for a run that did not
    /// opt in and for a kind that does not checkpoint per epoch (the
    /// context-predictor path).
    /// The worker's finalize publishes each and registers a catalog row for
    /// it — the bytes are already complete by the time this reaches
    /// `publish_and_finalize`.
    pub epoch_checkpoints: Vec<(usize, StagedArtifact)>,
    /// `Some` for the two LoRA kinds (never a context predictor) — the
    /// model-level materialization [`JobWorker::publish_and_finalize`]
    /// writes/records.
    pub(crate) materialization: Option<FineTuneMaterialization>,
}

// ── The gang's topology and the coordinator body ───────────────────────────

/// How `run_spec` lays out a claimed job's ranks —
/// decided from the job's own identity-relevant `world_size`
/// (`TrainingCommon::world_size`) and this host's `[worker] local_ranks`,
/// and nothing else:
///
/// - `world_size <= 1` → [`Self::Single`]: the single-rank path (`RankContext::single_rank`, the builder's default).
/// - `1 < world_size <= local_ranks` → [`Self::Local`]: every rank of the
///   gang runs in THIS process over a `Local` gang, rank `r` pinned to
///   `[gpu] devices[r]` (`[worker] local_ranks <= devices.len()` is enforced
///   at config load).
/// - `world_size > local_ranks` → [`Self::Peer`]: this process is rank 0,
///   the coordinator; ranks `1..world_size` are fleet members it assembles
///   and dials (`JobWorker::coordinate`).
///
/// `[distributed] max_world_size` plays no part here: it bounded the job at
/// submit (`RankAdmission`), and a claimed row is already within it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyDecision {
    Single,
    Local { world: u32 },
    Peer { world: u32 },
}

impl TopologyDecision {
    /// The rule above. `world_size == 0` is unrepresentable past the submit
    /// edge (`RankAdmission::admit` refuses it) and reads as `Single` here
    /// rather than as a gang of no ranks.
    pub fn decide(world_size: u32, local_ranks: u32) -> Self {
        if world_size <= 1 {
            Self::Single
        } else if world_size <= local_ranks {
            Self::Local { world: world_size }
        } else {
            Self::Peer { world: world_size }
        }
    }

    /// The canonical token `ProducingDescriptor::FineTune::collective`
    /// records: the collective THIS run reduces over (`Single` runs the
    /// gang-of-one `Noop`, `Local` the in-process `LocalGang`, `Peer` the
    /// coordinator's `Peer`) —
    /// a total function of the decided topology, so the manifest names what
    /// the run used rather than the `[worker] collective` selection it was
    /// configured with.
    fn collective_token(self) -> &'static str {
        match self {
            Self::Single => "noop",
            Self::Local { .. } => "local",
            Self::Peer { .. } => "peer",
        }
    }

    /// The ranks THIS HOST runs for the attempt —
    /// `ProducingDescriptor::FineTune::local_ranks`: one for `Single`, the
    /// whole gang for an in-process `Local` gang, and one (rank 0, the
    /// coordinator) for a `Peer` gang whose other ranks live on other hosts.
    fn host_ranks(self) -> u32 {
        match self {
            Self::Single | Self::Peer { .. } => 1,
            Self::Local { world } => world,
        }
    }
}

/// [`TopologyDecision`] with what `train_fine_tune` needs to spawn it: for
/// a `Peer` gang, the coordinator's collective — built by the coordinator
/// body over the members it dialed, BEFORE the blocking trainer starts.
enum RankTopology {
    Single,
    Local { world: u32 },
    Peer { world: u32, coordinator: Arc<Peer> },
}

/// The training-set identity pair the coordinator writes onto the job row
/// (write-once CAS, `Catalog::materialize_or_reuse_training_set`) and every
/// member is admitted against (`GangService::run_rank`'s world>1 conjunct):
/// the materialized table's sidecar digest and its name.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TrainingSetIdentityPair {
    training_set_ref: String,
    training_set_location: String,
}

/// Every way the coordinator body ([`JobWorker::coordinate`]) ends one
/// attempt — CLOSED, and matched TOTALLY by [`assembly_outcome`] (no
/// wildcard), so an end without a row in the assembly reason table is a
/// compile error, never a silent omission. [`Self::VARIANTS`] and
/// [`Self::ordinal`] pin the count for the table-driven oracle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CoordinatorEnd {
    /// The write-once CAS found the claim moved (another claimant/attempt
    /// holds the row, or the pair already holds different values): the
    /// attempt ends with NO write of any kind — not even an assembly
    /// outcome — since the row is no longer this attempt's to describe.
    Moved,
    /// This host cannot coordinate a `Peer` gang at all: no result-root
    /// identity to list members against (`[server] peer_advertise` unset),
    /// or no member dialer installed (no gang listener mounted in this
    /// process). Cooled, never counted: a member-capable host claims next.
    HostCannotCoordinate(String),
    /// A catalog read the assembly needed (the listing, an address
    /// resolution, the CAS) faulted — transient.
    CatalogFault(String),
    /// Fewer fresh, same-root, `claiming` members than `world - 1`.
    ShortListed { fresh: usize, needed: usize },
    /// A listed member's address no longer resolved by the time it was
    /// dialed (`peer_addr_of` → `None`: gone or stale since the listing).
    MemberUnreachable { rank: u32, instance_id: String },
    /// A member refused the dial, or ended its session before admission —
    /// `Unavailable` (a busy slot), an I-GANG refusal, a transport error.
    MemberRefused {
        rank: u32,
        instance_id: String,
        detail: String,
    },
    /// The coordinator's own `Peer` could not be built over the admitted
    /// links (a message cap below the codec's floor, no device).
    PeerRefused(String),
    /// The cancel flag was set before dispatch or tripped mid-run (a cancel
    /// request, or the lease lost).
    Cancelled,
    /// The host's phase left `Running` before dispatch.
    Drain,
    /// A member's session ended with `Aborted{reason}` while the run was in
    /// progress (the coordinator read it on a round wait): the reason maps
    /// through the table, one to one.
    MemberAborted { rank: u32, reason: AbortReason },
    /// The gang faulted mid-run for a reason that names no member abort
    /// (a transport fault, a round deadline, a descriptor disagreement, the
    /// coordinator's own refusal faulting its peers).
    LinkFault(String),
    /// Assembly proceeded and the run itself failed for a reason the gang
    /// did not fault on (a typed training refusal, a divergence, a panic):
    /// the job's own terminal failure, recorded as `failed` by the caller.
    TrainingFailed(String),
    /// Assembly proceeded, the run completed, every member ended
    /// `Outcome{Trained}` with rank 0's own artifact digest, and rank 0
    /// holds the artifact the caller publishes.
    Published,
}

impl CoordinatorEnd {
    /// How many variants this enum has — the table-driven oracle in this
    /// module's tests asserts one sample per [`Self::ordinal`] value below
    /// this count, so a variant added without a sample reds it. Read by
    /// that oracle alone.
    #[allow(dead_code)]
    pub(crate) const VARIANTS: usize = 13;

    /// A distinct index per variant, exhaustively — adding a variant fails
    /// to compile until it has one.
    pub(crate) fn ordinal(&self) -> usize {
        match self {
            Self::Moved => 0,
            Self::HostCannotCoordinate(_) => 1,
            Self::CatalogFault(_) => 2,
            Self::ShortListed { .. } => 3,
            Self::MemberUnreachable { .. } => 4,
            Self::MemberRefused { .. } => 5,
            Self::PeerRefused(_) => 6,
            Self::Cancelled => 7,
            Self::Drain => 8,
            Self::MemberAborted { .. } => 9,
            Self::LinkFault(_) => 10,
            Self::TrainingFailed(_) => 11,
            Self::Published => 12,
        }
    }
}

impl std::fmt::Display for CoordinatorEnd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Moved => f.write_str("the claim moved before assembly (no write)"),
            Self::HostCannotCoordinate(why) => write!(f, "this host cannot coordinate: {why}"),
            Self::CatalogFault(e) => write!(f, "a catalog read faulted during assembly: {e}"),
            Self::ShortListed { fresh, needed } => write!(
                f,
                "short listing: {fresh} fresh member(s) where {needed} are needed"
            ),
            Self::MemberUnreachable { rank, instance_id } => {
                write!(
                    f,
                    "rank {rank} ({instance_id}) no longer resolves to an address"
                )
            }
            Self::MemberRefused {
                rank,
                instance_id,
                detail,
            } => write!(f, "rank {rank} ({instance_id}) refused the dial: {detail}"),
            Self::PeerRefused(e) => write!(f, "the coordinator's Peer could not be built: {e}"),
            Self::Cancelled => f.write_str("cancelled"),
            Self::Drain => f.write_str("the host is draining"),
            Self::MemberAborted { rank, reason } => {
                write!(f, "rank {rank} ended its session: Aborted({reason:?})")
            }
            Self::LinkFault(e) => write!(f, "the gang faulted: {e}"),
            Self::TrainingFailed(e) => write!(f, "the run failed: {e}"),
            Self::Published => f.write_str("published"),
        }
    }
}

/// The TOTAL reason table (`AssemblyOutcome`'s own
/// doc carries the counting/cooldown rule per variant): every
/// [`CoordinatorEnd`] maps to exactly one [`AssemblyOutcome`] the body
/// records on the row — except [`CoordinatorEnd::Moved`], the one end that
/// writes nothing (`None`). No wildcard arm: a new end is a compile error
/// here until it has a row.
///
/// A member's `Aborted{reason}` maps one to one onto the outcome of the
/// same name; a reason outside the frozen set (`Unspecified`, or a value a
/// newer member sent) reads as transient (`Unavailable`).
/// `AllRootDivergent` is never produced here: root identity is a predicate
/// INSIDE `list_gang_members`, so divergent-root members are
/// invisible to the coordinator and an all-divergent fleet is a short
/// listing.
pub(crate) fn assembly_outcome(end: &CoordinatorEnd) -> Option<AssemblyOutcome> {
    Some(match end {
        CoordinatorEnd::Moved => return None,
        CoordinatorEnd::HostCannotCoordinate(_) => AssemblyOutcome::ShortListed,
        CoordinatorEnd::CatalogFault(_) => AssemblyOutcome::Unavailable,
        CoordinatorEnd::ShortListed { .. } => AssemblyOutcome::ShortListed,
        CoordinatorEnd::MemberUnreachable { .. } => AssemblyOutcome::Unavailable,
        CoordinatorEnd::MemberRefused { .. } => AssemblyOutcome::Unavailable,
        CoordinatorEnd::PeerRefused(_) => AssemblyOutcome::Unavailable,
        CoordinatorEnd::Cancelled => AssemblyOutcome::Cancelled,
        CoordinatorEnd::Drain => AssemblyOutcome::Drain,
        CoordinatorEnd::MemberAborted { reason, .. } => match reason {
            AbortReason::Refuted => AssemblyOutcome::Refuted,
            AbortReason::Unavailable => AssemblyOutcome::Unavailable,
            AbortReason::StoreUnavailable => AssemblyOutcome::StoreUnavailable,
            AbortReason::NoBody => AssemblyOutcome::NoBody,
            AbortReason::Drain => AssemblyOutcome::Drain,
            AbortReason::Cancelled => AssemblyOutcome::Cancelled,
            AbortReason::Unspecified => AssemblyOutcome::Unavailable,
        },
        CoordinatorEnd::LinkFault(_) => AssemblyOutcome::Unavailable,
        CoordinatorEnd::TrainingFailed(_) => AssemblyOutcome::Success,
        CoordinatorEnd::Published => AssemblyOutcome::Success,
    })
}

/// What the coordinator does with THIS attempt's job lease once the
/// attempt has ended — the released-vs-failed split, decided by [`lease_settlement`], a
/// TOTAL match over [`CoordinatorEnd`] (no wildcard: a new end is a
/// compile error until it has a row here too).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LeaseSettlement {
    /// `Catalog::release_job_lease` now — `releases + 1`, lease NULL: the
    /// row is reclaimable within one idle poll and the reclaim cap
    /// (`attempts - releases`) is untouched. Zero net attempts.
    Release,
    /// Left to expire: nothing is written; reclaim arm 1a requeues the row
    /// within the remaining lease window and the successor's `claim_next`
    /// spends the attempt (`attempts + 1`, `releases` unchanged). An
    /// attempt spent.
    Expire,
    /// Not this function's to settle: the row is not this attempt's
    /// (`Moved`), or the caller's own exit arm decides — `Published`
    /// finalizes, `TrainingFailed` records `failed`, `Cancelled` lands on
    /// the cancel arm (a request → `failed`; a lost lease → left for
    /// reclaim).
    Untouched,
}

/// The released-vs-failed split: how
/// the coordinator settles its lease for every way an attempt ends.
///
/// - **A member's `Aborted{Drain}`** (its host draining, a rolling restart
///   of the peer tier) → [`LeaseSettlement::Release`]: `release_job_lease`
///   (`releases + 1`, lease NULL; the CAS admits the holder) so the restart
///   costs the job zero net attempts.
/// - **Every other mid-run gang fault** — a member's `Aborted` for any
///   other reason, a stream that dropped, a rank silent past
///   `[worker] rank_timeout_secs`, a peer's round fault — is a rank failure
///   → [`LeaseSettlement::Expire`]: the lease is left to expire, reclaim
///   requeues the row within the lease window and the successor's claim
///   spends the attempt (never here; `attempts + 1` is `claim_next`'s).
///   Bounded by the reclaim cap, a member that keeps failing exhausts the
///   job's attempts instead of retrying it forever.
/// - **An assembly end** (no run started: the host cannot coordinate, a
///   catalog fault, a short listing, a member unreachable or refusing, a
///   `Peer` that could not be built, the host draining before dispatch)
///   settles by the recorded outcome's counting class
///   (`AssemblyOutcome::counts_toward_failures`): an uncounted outcome
///   hands the lease back at once (nothing was spent assembling nothing),
///   a counted one leaves it to expire.
/// - **`Moved`, `Published`, `TrainingFailed`, `Cancelled`** →
///   [`LeaseSettlement::Untouched`] (the caller's arms, see the variant).
pub(crate) fn lease_settlement(end: &CoordinatorEnd) -> LeaseSettlement {
    match end {
        CoordinatorEnd::Moved
        | CoordinatorEnd::Published
        | CoordinatorEnd::TrainingFailed(_)
        | CoordinatorEnd::Cancelled => LeaseSettlement::Untouched,
        CoordinatorEnd::MemberAborted {
            reason: AbortReason::Drain,
            ..
        } => LeaseSettlement::Release,
        CoordinatorEnd::MemberAborted {
            reason:
                AbortReason::Refuted
                | AbortReason::Unavailable
                | AbortReason::StoreUnavailable
                | AbortReason::NoBody
                | AbortReason::Cancelled
                | AbortReason::Unspecified,
            ..
        }
        | CoordinatorEnd::LinkFault(_) => LeaseSettlement::Expire,
        CoordinatorEnd::HostCannotCoordinate(_)
        | CoordinatorEnd::CatalogFault(_)
        | CoordinatorEnd::ShortListed { .. }
        | CoordinatorEnd::MemberUnreachable { .. }
        | CoordinatorEnd::MemberRefused { .. }
        | CoordinatorEnd::PeerRefused(_)
        | CoordinatorEnd::Drain => match assembly_outcome(end) {
            Some(outcome) if outcome.counts_toward_failures() => LeaseSettlement::Expire,
            Some(_) => LeaseSettlement::Release,
            None => LeaseSettlement::Untouched,
        },
    }
}

/// A listing too short for the gang.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ShortListing {
    pub(crate) fresh: usize,
    pub(crate) needed: usize,
}

/// Rank assignment — a PURE function of the membership listing: the members sorted by `instance_id` byte order (the
/// same order `Catalog::list_gang_members` already returns; sorting again
/// here makes the assignment independent of any return order), and rank
/// `r` is the `r`-th of them, `r = 1..world`. No substitution: a listing
/// shorter than `world - 1` is [`ShortListing`], never a smaller gang.
pub(crate) fn assign_ranks(
    members: &[GangMember],
    world: u32,
) -> std::result::Result<Vec<(u32, GangMember)>, ShortListing> {
    let needed = (world as usize).saturating_sub(1);
    let mut sorted: Vec<&GangMember> = members.iter().collect();
    sorted.sort_by(|a, b| a.instance_id.as_bytes().cmp(b.instance_id.as_bytes()));
    if sorted.len() < needed {
        return Err(ShortListing {
            fresh: sorted.len(),
            needed,
        });
    }
    Ok(sorted
        .into_iter()
        .take(needed)
        .enumerate()
        .map(|(index, member)| (index as u32 + 1, member.clone()))
        .collect())
}

/// Who one attempt of `spec` runs as on this host (the module doc's writer
/// table): the `Coordinator` exactly when a job that trains from a
/// training-set table decides [`TopologyDecision::Peer`] over this host's
/// `[worker] local_ranks` — the one shape `run_spec` hands to the coordinator
/// body — and the `LoopClaimer` for every other shape: a single rank, an
/// in-process `Local` gang, and a context predictor
/// (single-rank by admission). Decided from the SAME `TopologyDecision::
/// decide` call `run_spec` makes, over the same inputs, so the two cannot
/// diverge. `W == 1` is the loop claimer on every arm.
pub fn lease_holder_for(spec: &TrainingSpec, local_ranks: u32) -> LeaseHolder {
    let Some(view) = spec.training_set_view() else {
        return LeaseHolder::LoopClaimer;
    };
    match TopologyDecision::decide(view.common.world_size, local_ranks) {
        TopologyDecision::Peer { .. } => LeaseHolder::Coordinator,
        TopologyDecision::Single | TopologyDecision::Local { .. } => LeaseHolder::LoopClaimer,
    }
}

/// The training source every rank of a column-source `fine_tune` binds over
/// its (shared, attested) training-set table — `Resident` for the whole-set
/// arms (`source::whole_set_arm`: mining, GradCache — both refused at
/// `world > 1` by admission, so a gang's ranks always take the `Streamed`
/// arm), `Streamed` otherwise. Called by rank 0 in `run_spec` and by a
/// `Peer` member's rank body ([`run_member_rank`]) over the SAME table
/// (bound by name and digest through the job row), so the ranks' loaders
/// derive from one definition. Runs under the caller's tenant scope: the
/// streamed set captures `session.tenant()` here, inside that scope, and
/// re-applies it per open.
async fn bind_training_source(
    session: &Arc<InferenceSession>,
    table: &jammi_db::store::TrainingSetTable,
    columns: &[String],
    task: ModelTask,
    detected: crate::fine_tune::decode::DetectedFormat,
    common: &TrainingCommon,
) -> Result<crate::fine_tune::source::TrainingSource> {
    // The ONE predicate deciding Resident vs Streamed —
    // the SAME `source::whole_set_arm` the trainer's own dispatch
    // (`TrainingLoop::run`) refuses a mismatch against. A `FineTune` spec
    // always loads a base model (`train_fine_tune` unconditionally calls
    // `.base_model(..)`), so `has_base_model` is always `true` here.
    let whole_set_arm = crate::fine_tune::source::whole_set_arm(&common.config, true);
    if whole_set_arm.is_some() {
        // Resident: the eager arm — read the whole table back into memory,
        // HOLDING the eager read's pool reservation for the loader's own
        // lifetime rather than checking-then-releasing it
        // (`training_set::read_back`'s behavior, which every OTHER caller
        // gets).
        let (batches, reservation) =
            training_set::read_back_with_reservation(session, table).await?;
        let loader =
            build_training_data_loader(&batches, columns, task)?.with_reservation(reservation);
        // The tag the table was WRITTEN under and the shape its loader
        // reports come from one classifier, so a mismatch is a broken
        // engine invariant rather than a caller error — and it must be
        // loud: it would mean two formats sharing one definition hash.
        if loader.format().format_tag() != detected.format_tag() {
            return Err(JammiError::Other(format!(
                "training set was committed as format '{}' but its loader reports '{}': the \
                 column classifier and the loader disagree",
                detected.format_tag(),
                loader.format().format_tag()
            )));
        }
        return Ok(crate::fine_tune::source::TrainingSource::Resident(loader));
    }
    // Streamed: table only — no row is ever collected
    // into memory for this arm.
    let total_rows = table.row_count();
    let train_count =
        crate::fine_tune::data::split_index(total_rows, common.config.validation_fraction);

    // The whole-table refusal pre-pass, ONCE, over `[0, total_rows)`,
    // BEFORE the first training step — not lazily discovered mid-run.
    crate::fine_tune::stream::validate_window(
        session,
        table,
        detected,
        task,
        crate::fine_tune::stream::RowWindow::new(0, total_rows),
    )
    .await?;

    // The classification label vocabulary spans the WHOLE table (train
    // + val), built ONCE here — never re-derived per-epoch or per-window.
    let label_vocab = if matches!(
        detected,
        crate::fine_tune::decode::DetectedFormat::Classification
    ) {
        Some(crate::fine_tune::stream::build_label_vocabulary(session, table).await?)
    } else {
        None
    };

    // `PRODUCTION_PREFETCH_DEPTH` — see its own doc for why this is a named
    // constant, never a literal here.
    let stream_cfg = crate::fine_tune::stream::StreamConfig::new(
        crate::fine_tune::stream::PRODUCTION_PREFETCH_DEPTH,
    )?;

    // Captured HERE, inside the caller's task, which is still running under
    // its `with_tenant_scoped` task-local (`run_claimed_job_under`'s doc;
    // `run_member_rank`'s scope) — `session.tenant()` reads that override,
    // not the session's sticky binding. `TrainingSetStream::open` later
    // runs on the `spawn_blocking` pool via `Handle::block_on`, which does
    // NOT inherit this task-local, so the value is captured now and
    // re-applied explicitly per open.
    let tenant = session.tenant();
    let streamed = crate::fine_tune::source::StreamedSet {
        session: Arc::clone(session),
        table: table.clone(),
        columns: columns.to_vec(),
        task,
        total_rows,
        train_count,
        batch: common.config.batch_size,
        stream_cfg,
        label_vocab,
        tenant,
    };
    Ok(crate::fine_tune::source::TrainingSource::Streamed(
        Box::new(streamed),
    ))
}

/// Bind the training set a prior attempt recorded on the job row (the
/// write-once identity pair) for THIS attempt to train from — the retry's
/// half of "the coordinator materializes or reuses the training set".
/// Resolved by name through the job's own tenant-pinned
/// catalog, it must be `ready` and its sidecar must verify the recorded
/// digest — the SAME verify every member is admitted against
/// (`GangService::run_rank`'s world>1 conjunct) — else the attempt is
/// refused, typed: a job whose recorded training set is gone or no longer
/// verifies has lost its identity, and a fresh materialization would train
/// a different job under the same row. The table is bound on the session
/// context exactly as the producer's own reuse arm binds one, so the eager
/// read-back and the streamed opens resolve it like a fresh table. Also the
/// binding a `Peer` member performs ([`run_member_rank`]) over the pair it
/// was admitted against; the sidecar it verified is returned beside the
/// table so the member can verify its partition's leaves against it.
///
/// Error classes, by construction of this function: every REFUSAL — the
/// table is gone, not ready, carries no sidecar, or its digest no longer
/// verifies — is `JammiError::FineTune`; every other variant is a read
/// itself faulting (the catalog, or the store — `Storage`/`Io`). A member
/// maps the two classes to `Aborted{Refuted}` and
/// `Aborted{StoreUnavailable}`/`Aborted{Unavailable}` respectively, the
/// SAME split the gang handler's re-verification tick applies.
async fn bind_recorded_training_set(
    session: &Arc<InferenceSession>,
    catalog: &Arc<Catalog>,
    pair: &TrainingSetIdentityPair,
) -> Result<(
    jammi_db::store::TrainingSetTable,
    jammi_db::store::manifest::MaterializationManifest,
)> {
    let Some(record) = catalog
        .get_result_table(&pair.training_set_location)
        .await?
    else {
        return Err(JammiError::FineTune(format!(
            "the job row names training set '{}' but no such table resolves under the job's \
             tenant: the recorded training set is gone",
            pair.training_set_location
        )));
    };
    if record.status != jammi_db::catalog::status::ResultTableStatus::Ready.to_string() {
        return Err(JammiError::FineTune(format!(
            "the job row names training set '{}' but its status is '{}', not ready",
            pair.training_set_location, record.status
        )));
    }
    let url = jammi_db::storage::StorageUrl::parse(&record.parquet_path)?;
    let store = session.result_store();
    let Some(manifest) = store.read_materialization_manifest(&url).await? else {
        return Err(JammiError::FineTune(format!(
            "the job row names training set '{}' but it carries no verifiable sidecar",
            pair.training_set_location
        )));
    };
    if manifest.artifact.0 != pair.training_set_ref {
        return Err(JammiError::FineTune(format!(
            "the job row names training set '{}' at digest {} but its sidecar reads {}: the \
             recorded training set no longer verifies",
            pair.training_set_location, pair.training_set_ref, manifest.artifact.0
        )));
    }
    store.bind_result_table(session.context(), &record).await?;
    // `TrainingSetTable::from_record` reads its own order columns and
    // definition hash off `manifest` — never derives them
    // here and hands them in, so a manifest whose descriptor is not
    // `TrainingSet` refuses inside that constructor, typed, rather than
    // trusting this call site's own match to have gotten the refusal right.
    let outcome = jammi_db::store::CacheOutcome::Reused(jammi_db::store::ReusedArtifact::Table(
        record.name(),
    ));
    let table = jammi_db::store::TrainingSetTable::from_record(record, &manifest, outcome)
        .map_err(|e| {
            JammiError::FineTune(format!(
                "the job row names training set '{}' but {e}",
                pair.training_set_location
            ))
        })?;
    Ok((table, manifest))
}

/// End every admitted member session in `links` cooperatively (one
/// `Cancel` each) — the stream close for an attempt that ends before its
/// `Peer` exists.
fn cancel_links(links: &[CoordinatorLink]) {
    for link in links {
        link.cancel_session();
    }
}

impl JobWorker {
    /// The coordinator body: rank 0 of a `Peer` gang, in the process that claimed the job. In
    /// order: (1) the write-once CAS of the training-set identity pair
    /// (`materialize_or_reuse_training_set` — a `Moved` claim exits with
    /// no write at all); (2) the scaler — computed INSIDE the trainer's run
    /// on every rank from the training set's own targets (`TrainingLoop::
    /// run`'s scaler pass), so no value crosses the wire; (3) the
    /// membership listing with this process's own `MemberRoot` in the
    /// `GangListing` — the verb filters on kind, state, freshness, self and
    /// root identity, this body filters nothing; (4) [`assign_ranks`], the
    /// pure assignment over the sorted listing; (5) dispatch — for each
    /// assigned member, `peer_addr_of` then the installed [`MemberDialer`]
    /// (`gang_rounds::dial_member`) with the `Assign`; (6) the run as rank
    /// 0 over `Peer(coordinator)` through [`Self::train_fine_tune`]; (7)
    /// exactly one [`AssemblyOutcome`] recorded on the row through the
    /// total table ([`assembly_outcome`]) for every end but `Moved`, then
    /// the lease settled by the released-vs-failed split
    /// ([`lease_settlement`]): a member's `Aborted{Drain}` and every
    /// uncounted assembly end hand the lease back at once
    /// (`release_job_lease`, zero net attempts); every other
    /// mid-run gang fault leaves it to expire (an attempt spent at the
    /// successor's claim); either way the row stays `running` for reclaim
    /// with NO terminal write, and the next attempt re-lists once its
    /// cooldown passes. Every member session is ended (`Cancel`) whichever
    /// way the attempt ends.
    ///
    /// **The per-attempt watchdog** is the coordinator's own `Peer`: every
    /// member's stream is read by its rounds, so a member's `Aborted{reason}`
    /// (recorded typed on that member's link), a stream that dropped, or a
    /// rank silent past `[worker] rank_timeout_secs` (the round deadline)
    /// ends rank 0's collective call with the gang faulted — every member
    /// is faulted in the same round (`RoundFault`) and its session ended
    /// (`Cancel`, the stream close) — and this body classifies the end
    /// from the links (`MemberAborted`/`LinkFault`). The `Peer` is built
    /// for this attempt and dropped with it, so a fault retires exactly
    /// the attempt it belongs to. The slot discipline guarantees that no
    /// session end aborts a claim: a
    /// member's slot is `Rank` for the whole session and a peer never
    /// claims while it holds a rank (`HostAdmission`), so ending a session
    /// never aborts a claim transaction anywhere.
    /// Runs as `LeaseHolder::Coordinator` — the module doc's writer table,
    /// row 13 — on every write it makes here and on rank 0's own run.
    #[allow(clippy::too_many_arguments)]
    async fn coordinate(
        &self,
        session: &Arc<InferenceSession>,
        catalog: &Arc<Catalog>,
        job_id: &str,
        kind: &str,
        run: FineTuneRun,
        cancel: &Arc<AtomicBool>,
        attempt: u32,
        world: u32,
        pair: TrainingSetIdentityPair,
    ) -> std::result::Result<TrainedArtifact, WorkerJobError> {
        let (end, artifact) = self
            .assemble_and_run(
                session, catalog, job_id, kind, run, cancel, attempt, world, pair,
            )
            .await;
        #[cfg(feature = "test-hooks")]
        training_test_hooks::note_coordinator_end(job_id, attempt, &end);
        let outcome = assembly_outcome(&end);
        if let Some(outcome) = outcome {
            // Row 13 of the module doc's writer table: the coordinator's
            // own writes, as the coordinator.
            let holder = LeaseHolder::Coordinator;
            match catalog
                .record_assembly_outcome(job_id, attempt, outcome)
                .await
            {
                Ok(true) => {}
                Ok(false) => tracing::warn!(
                    job_id = %job_id, attempt, ?outcome, %holder,
                    "assembly outcome not recorded: the attempt moved under the coordinator"
                ),
                Err(e) => tracing::warn!(
                    job_id = %job_id, attempt, ?outcome, error = %e,
                    "assembly outcome could not be recorded"
                ),
            }
        }
        let settlement = lease_settlement(&end);
        tracing::info!(
            job_id = %job_id,
            attempt,
            world,
            end = %end,
            end_ordinal = end.ordinal(),
            ?settlement,
            "coordinator attempt ended"
        );
        match settlement {
            LeaseSettlement::Release => {
                match catalog
                    .release_job_lease(job_id, &self.worker_id, attempt)
                    .await
                {
                    Ok(true) => {}
                    Ok(false) => tracing::debug!(
                        job_id = %job_id, attempt,
                        "the lease was not this attempt's to release"
                    ),
                    Err(e) => tracing::warn!(
                        job_id = %job_id, attempt, error = %e,
                        "releasing the lease after the attempt ended failed; left to expiry"
                    ),
                }
            }
            LeaseSettlement::Expire => tracing::warn!(
                job_id = %job_id, attempt, end = %end,
                "gang attempt retired: the lease is left to expire, reclaim requeues the job \
                 and the successor's claim spends the attempt"
            ),
            LeaseSettlement::Untouched => {}
        }
        match (end, artifact) {
            (CoordinatorEnd::Published, Some(artifact)) => Ok(artifact),
            (CoordinatorEnd::Published, None) => Err(WorkerJobError::Failed(JammiError::FineTune(
                "the coordinator ended Published without an artifact".into(),
            ))),
            (CoordinatorEnd::Cancelled, _) => Err(WorkerJobError::Cancelled),
            (CoordinatorEnd::TrainingFailed(msg), _) => {
                Err(WorkerJobError::Failed(JammiError::FineTune(msg)))
            }
            (end, _) => Err(WorkerJobError::Abandoned(end.to_string())),
        }
    }

    /// [`Self::coordinate`]'s steps (1)–(6), ending in exactly one
    /// [`CoordinatorEnd`] and, for `Published`, the artifact.
    #[allow(clippy::too_many_arguments)]
    async fn assemble_and_run(
        &self,
        session: &Arc<InferenceSession>,
        catalog: &Arc<Catalog>,
        job_id: &str,
        kind: &str,
        run: FineTuneRun,
        cancel: &Arc<AtomicBool>,
        attempt: u32,
        world: u32,
        pair: TrainingSetIdentityPair,
    ) -> (CoordinatorEnd, Option<TrainedArtifact>) {
        // (1) The CAS: this attempt owns the row's training-set identity
        // pair from here on, or it never did.
        match catalog
            .materialize_or_reuse_training_set(
                job_id,
                &self.worker_id,
                attempt,
                &pair.training_set_ref,
                &pair.training_set_location,
            )
            .await
        {
            Ok(TrainingSetAssembly::Won | TrainingSetAssembly::Reused) => {}
            Ok(TrainingSetAssembly::Moved) => return (CoordinatorEnd::Moved, None),
            Err(e) => return (CoordinatorEnd::CatalogFault(e.to_string()), None),
        }

        // The pre-dispatch gates: a cancel already requested (or a lease
        // already lost), and a host that is no longer `Running`, dispatch
        // nothing.
        if cancel.load(Ordering::SeqCst) {
            return (CoordinatorEnd::Cancelled, None);
        }
        let admission = session.host_admission();
        if admission.phase() != WorkerPhase::Running {
            return (CoordinatorEnd::Drain, None);
        }

        // (3) Membership: the verb decides every predicate (kind, state,
        // freshness, self-exclusion, root identity); nothing is filtered
        // here.
        let registration = session.instance_registration();
        let Some(root) = registration.member_root.as_ref() else {
            return (
                CoordinatorEnd::HostCannotCoordinate(
                    "[server] peer_advertise is unset, so this host carries no result-root \
                     identity to list gang members against"
                        .into(),
                ),
                None,
            );
        };
        let members = match catalog
            .list_gang_members(GangListing {
                kind,
                self_instance: &self.worker_id,
                root,
                lease: self.intervals.lease,
            })
            .await
        {
            Ok(members) => members,
            Err(e) => return (CoordinatorEnd::CatalogFault(e.to_string()), None),
        };

        // (4) Assignment: pure over the sorted listing, no substitution.
        let assignment = match assign_ranks(&members, world) {
            Ok(assignment) => assignment,
            Err(ShortListing { fresh, needed }) => {
                return (CoordinatorEnd::ShortListed { fresh, needed }, None)
            }
        };
        #[cfg(feature = "test-hooks")]
        training_test_hooks::note_assembly_listing(
            job_id,
            attempt,
            assignment
                .iter()
                .map(|(rank, member)| (*rank, member.instance_id.clone()))
                .collect(),
        );

        // (5) Dispatch: one `RunRank` per member, in rank order, through
        // the installed dialer. A member that does not admit ends THIS
        // attempt (every session admitted so far is ended); the next attempt
        // re-lists.
        let Some(dialer) = admission.member_dialer() else {
            return (
                CoordinatorEnd::HostCannotCoordinate(
                    "no gang listener is mounted in this process (no member dialer installed), \
                     so it cannot dial members"
                        .into(),
                ),
                None,
            );
        };
        let max_message_bytes =
            usize::try_from(session.inner_config().server.limits.max_message_bytes)
                .unwrap_or(usize::MAX);
        let mut links: Vec<CoordinatorLink> = Vec::with_capacity(assignment.len());
        for (rank, member) in &assignment {
            let addr = match catalog
                .peer_addr_of(&member.instance_id, self.intervals.lease)
                .await
            {
                Ok(Some(addr)) => addr,
                Ok(None) => {
                    cancel_links(&links);
                    return (
                        CoordinatorEnd::MemberUnreachable {
                            rank: *rank,
                            instance_id: member.instance_id.clone(),
                        },
                        None,
                    );
                }
                Err(e) => {
                    cancel_links(&links);
                    return (CoordinatorEnd::CatalogFault(e.to_string()), None);
                }
            };
            let assign = Assign {
                job_id: job_id.to_string(),
                attempt: i64::from(attempt),
                rank: *rank,
                world,
                coordinator_instance_id: self.worker_id.clone(),
            };
            match dialer.dial(&addr, assign, max_message_bytes).await {
                Ok(link) => links.push(link),
                Err(e) => {
                    cancel_links(&links);
                    return (
                        CoordinatorEnd::MemberRefused {
                            rank: *rank,
                            instance_id: member.instance_id.clone(),
                            detail: e.to_string(),
                        },
                        None,
                    );
                }
            }
        }

        // (6) The coordinator's own collective over the admitted links, on
        // this host's primary device, at the deployment's rank timeout.
        let device = match crate::model::backend::candle::select_device(session.device_config()) {
            Ok(device) => device,
            Err(e) => {
                cancel_links(&links);
                return (CoordinatorEnd::PeerRefused(e.to_string()), None);
            }
        };
        let rank_timeout = Duration::from_secs(session.inner_config().worker.rank_timeout_secs);
        let coordinator = match Peer::coordinator(links, device, max_message_bytes)
            .and_then(|peer| peer.with_timeout(rank_timeout))
        {
            Ok(peer) => Arc::new(peer),
            Err(e) => return (CoordinatorEnd::PeerRefused(e.to_string()), None),
        };

        let result = self
            .train_fine_tune(
                session,
                catalog,
                job_id,
                run,
                cancel,
                attempt,
                LeaseHolder::Coordinator,
                RankTopology::Peer {
                    world,
                    coordinator: Arc::clone(&coordinator),
                },
            )
            .await;

        // (7) Every member's end — the terminal write on receipt: rank 0's
        // run completed, but the attempt is `Published` only once every
        // member's session ended `Outcome{Trained}` carrying rank 0's OWN
        // artifact digest (the gang converged to one artifact); a member's
        // `Outcome{Failed}`, a differing digest, an `Aborted` or a silent
        // end is that member's end of the attempt, and nothing is published
        // over a gang that did not complete.
        let result = match result {
            Ok(artifact) => match reconcile_member_ends(&coordinator, &artifact).await {
                Ok(()) => Ok(artifact),
                Err(end) => {
                    coordinator.end_members();
                    return (end, None);
                }
            },
            Err(e) => Err(e),
        };

        // (8) The stream close: every member's session is ended
        // cooperatively whichever way the run ended, so a member's slot is
        // freed now rather than left to the member's own bounds.
        coordinator.end_members();

        match result {
            Ok(artifact) => (CoordinatorEnd::Published, Some(artifact)),
            Err(WorkerJobError::Cancelled) => (CoordinatorEnd::Cancelled, None),
            // `train_fine_tune` never produces this arm (it is the
            // coordinator body's own classification); folded, not
            // wildcarded, so the match stays total.
            Err(WorkerJobError::Abandoned(why)) => (CoordinatorEnd::TrainingFailed(why), None),
            // `train_fine_tune` never produces this arm either — `HandedOff`
            // is `submit_placed`'s own classification, reached ONLY from
            // `run_claimed_job_under`'s placement check, before topology is
            // ever decided (a `Peer` rank's own `train_fine_tune` call never
            // reaches it). Folded, not wildcarded, so the match stays total.
            Err(WorkerJobError::HandedOff) => (
                CoordinatorEnd::TrainingFailed(
                    "unreachable: HandedOff surfaced from train_fine_tune".into(),
                ),
                None,
            ),
            Err(WorkerJobError::Failed(error)) => {
                if let Some((rank, raw)) = coordinator.member_aborts().into_iter().next() {
                    let reason = AbortReason::try_from(raw).unwrap_or(AbortReason::Unspecified);
                    (CoordinatorEnd::MemberAborted { rank, reason }, None)
                } else if let Some(fault) = coordinator.fault() {
                    (CoordinatorEnd::LinkFault(fault), None)
                } else {
                    (
                        CoordinatorEnd::TrainingFailed(failed_job_message(&error)),
                        None,
                    )
                }
            }
        }
    }
}

/// [`JobWorker::assemble_and_run`]'s step (7): read every member's end off
/// the coordinator's links ([`Peer::collect_member_ends`], on a blocking
/// thread under its own witness) and require each to be `Trained` with
/// rank 0's own adapter digest ([`adapter_files_digest`] over the files
/// rank 0 is about to publish). The first member that is not ends the
/// attempt, typed: a differing digest or a `Failed{reason}` is the run's
/// own failure (`TrainingFailed`, recorded `failed` by the caller), an
/// `Aborted{reason}` maps one to one through the assembly table
/// (`MemberAborted`), a closed/faulted/silent stream is a `LinkFault`.
async fn reconcile_member_ends(
    coordinator: &Arc<Peer>,
    artifact: &TrainedArtifact,
) -> std::result::Result<(), CoordinatorEnd> {
    let own = adapter_files_digest(artifact.dir.path())
        .map_err(|e| CoordinatorEnd::TrainingFailed(format!("rank 0's adapter digest: {e}")))?;
    let peer = Arc::clone(coordinator);
    let ends = BlockingCall::spawn_blocking(move |call| peer.collect_member_ends(&call))
        .await
        .map_err(|e| CoordinatorEnd::LinkFault(format!("collecting the members' outcomes: {e}")))?;
    for (rank, end) in ends {
        #[cfg(feature = "test-hooks")]
        training_test_hooks::note_member_end(&artifact.register.model_id, rank, &end);
        match end {
            MemberEnd::Trained { artifact_digest } if artifact_digest == own => {}
            MemberEnd::Trained { artifact_digest } => {
                return Err(CoordinatorEnd::TrainingFailed(format!(
                    "rank {rank} ended Trained with adapter digest {artifact_digest} where rank \
                     0's is {own}: the gang did not converge to one artifact"
                )))
            }
            MemberEnd::Failed { reason } => {
                return Err(CoordinatorEnd::TrainingFailed(format!(
                    "rank {rank}: {reason}"
                )))
            }
            MemberEnd::Aborted(raw) => {
                let reason = AbortReason::try_from(raw).unwrap_or(AbortReason::Unspecified);
                return Err(CoordinatorEnd::MemberAborted { rank, reason });
            }
            MemberEnd::Ended(why) => {
                return Err(CoordinatorEnd::LinkFault(format!(
                    "rank {rank} ended without an Outcome: {why}"
                )))
            }
        }
    }
    Ok(())
}

/// The digest of the adapter files a rank holds after its run — exactly the
/// file set `publish_artifact` publishes (every regular file directly in
/// `dir`, in name order; subdirectories are scratch and are skipped), each
/// folded as `name`, a NUL, the byte length, the bytes. What a `Peer`
/// member reports in `Outcome{Trained}` and what the coordinator computes
/// over its own files to compare against
/// (`reconcile_member_ends`): every rank holds identical weights after
/// the last step, so a gang that converged reports one
/// digest. Not the store's per-file manifest hash: this is a rank-side
/// fact about local bytes, computed by the ONE function on both sides.
pub fn adapter_files_digest(dir: &std::path::Path) -> Result<String> {
    use sha2::Digest;
    let mut names: Vec<(String, std::path::PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        names.push((
            entry.file_name().to_string_lossy().into_owned(),
            entry.path(),
        ));
    }
    names.sort_by(|a, b| a.0.cmp(&b.0));
    let mut hasher = sha2::Sha256::new();
    for (name, path) in names {
        let bytes = std::fs::read(path)?;
        hasher.update(name.as_bytes());
        hasher.update([0u8]);
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// What an admitted member session hands its rank body: the assignment the
/// coordinator's `Assign` carried and the row facts the gang handler
/// verified at admission (`GangService::run_rank`, `jammi-server`) — the
/// row's own tenant, the training-set identity pair, the `spec` column
/// verbatim. Nothing here came from the coordinator but the coordinates
/// (job, attempt, rank, world, the coordinator's instance id).
#[derive(Debug, Clone)]
pub struct MemberAssignment {
    pub job_id: String,
    pub attempt: u32,
    pub rank: u32,
    pub world: u32,
    /// The lease holder's instance id (`claimed_by`) — the coordinator.
    pub coordinator_instance_id: String,
    /// The row's own tenant, derived at admission (never the caller's).
    pub tenant: Option<TenantId>,
    /// The training-set identity pair the member was admitted against.
    pub training_set_ref: String,
    pub training_set_location: String,
    /// The row's `spec` column, verbatim.
    pub spec_json: String,
}

/// How a rank body ended — what the member's session ends with, as ONE
/// terminal stream event emitted by the session's hold loop: `Trained` and
/// `Failed` are `RankEvent::Outcome`; `Aborted` is `RankEvent::Aborted`
/// with a reason the pre-collective prologue decided (the training-set
/// identity no longer holds, this host's store faulted, the catalog did
/// not answer) — the SAME reason class the hold loop's re-verification tick
/// uses, so the wire says the same thing whichever of the two saw it first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RankOutcome {
    /// The run completed; the digest of the adapter files this rank holds
    /// ([`adapter_files_digest`]).
    Trained { artifact_digest: String },
    /// A typed failure of the run itself.
    Failed { reason: String },
    /// The prologue refused before the first collective.
    Aborted(AbortReason),
}

/// The rank body: an admitted member
/// session runs `TrainingLoop::run` as rank `assignment.rank` of a `Peer`
/// gang of `assignment.world` over `link`, and returns how it ended. In
/// order, under the row's own tenant scope: the spec is reconstructed from
/// the row (a column-source `fine_tune`, the one kind a `Peer` gang
/// serves); the recorded training set is bound by name and digest exactly
/// as a retrying coordinator binds it (`bind_recorded_training_set` —
/// the identity verified at admission, re-verified here); every leaf
/// of the table this rank's partition reads is verified against the
/// sidecar's inventory BEFORE the first collective
/// (`collective::peer::verify_partition_leaves` — under
/// `BlockByGlobalBatch` every row group carries rows of every rank, so the
/// partition's leaves are the object's; a bad leaf is the member-scoped
/// `Aborted{StoreUnavailable}`); the source is bound through the SAME
/// `bind_training_source` rank 0 used; the base model is loaded; the
/// member's `Peer` is built over `link` at the deployment's rank timeout
/// and cap; and `run_fine_tune_blocking` runs on the blocking pool under
/// the witness minted at ITS OWN `BlockingCall::spawn_blocking` — the third
/// production minting site — as `RunnerRole::Rank { rank }`: the same
/// target construction, seed split and acceleration probe as rank 0, no
/// persisted report, no checkpoint write (the trainer's role gate), no
/// job-row write (nothing here holds a `LeaseHolder` to pass), no artifact
/// publish. `cancel` is the session's: the hold loop flips it when the
/// session ends for another reason, and the trainer's epoch-boundary check
/// reads it exactly as rank 0 reads its lease flag.
pub async fn run_member_rank(
    session: Arc<InferenceSession>,
    assignment: MemberAssignment,
    link: MemberLink,
    cancel: Arc<AtomicBool>,
) -> RankOutcome {
    #[cfg(feature = "test-hooks")]
    let (job_id, rank) = (assignment.job_id.clone(), assignment.rank);
    let outcome = match assignment.tenant {
        Some(tenant) => {
            session
                .with_tenant_scoped(tenant, |_scope| {
                    member_rank_body(&session, assignment, link, cancel)
                })
                .await
        }
        None => member_rank_body(&session, assignment, link, cancel).await,
    };
    #[cfg(feature = "test-hooks")]
    training_test_hooks::note_rank_outcome(&job_id, rank, &outcome);
    outcome
}

/// [`run_member_rank`]'s body, inside the tenant scope.
async fn member_rank_body(
    session: &Arc<InferenceSession>,
    assignment: MemberAssignment,
    link: MemberLink,
    cancel: Arc<AtomicBool>,
) -> RankOutcome {
    let MemberAssignment {
        job_id,
        attempt,
        rank,
        world,
        coordinator_instance_id,
        tenant,
        training_set_ref,
        training_set_location,
        spec_json,
    } = assignment;
    let failed = |reason: String| RankOutcome::Failed {
        reason: format!("rank {rank}: {reason}"),
    };
    let catalog = Arc::new(session.catalog().pinned_to_tenant(tenant));

    // Decode the one persisted type (`crate::jobs::JobSpec`'s own doc), then
    // project to `TrainingSpec` — see the loop-claimer training path's own
    // comment for why.
    let job_spec: crate::jobs::JobSpec = match serde_json::from_str(&spec_json) {
        Ok(spec) => spec,
        Err(e) => return failed(format!("undeserialisable training_spec: {e}")),
    };
    let Some(spec) = job_spec.as_training_spec() else {
        return failed(format!(
            "a gang trains from a training-set table; the row's spec is a compute kind {}",
            job_spec.kind()
        ));
    };
    let Some(view) = spec.training_set_view() else {
        return failed(format!(
            "a gang trains from a training-set table; a {} has none",
            spec.kind()
        ));
    };
    let (columns, task, common) = (view.columns, view.task, view.common.clone());
    if common.world_size != world {
        return failed(format!(
            "the row's spec names world_size {} where the assignment names {world}",
            common.world_size
        ));
    }

    // The identity, re-verified by the SAME binding a retrying coordinator
    // performs; its refusal classes are the hold loop's re-verification
    // classes (`bind_recorded_training_set`'s doc).
    let pair = TrainingSetIdentityPair {
        training_set_ref,
        training_set_location,
    };
    let (table, manifest) = match bind_recorded_training_set(session, &catalog, &pair).await {
        Ok(bound) => bound,
        Err(JammiError::FineTune(why)) => {
            tracing::warn!(job_id = %job_id, rank, %why, "rank body: the training-set identity no longer holds");
            return RankOutcome::Aborted(AbortReason::Refuted);
        }
        Err(e @ (JammiError::Storage(_) | JammiError::Io(_))) => {
            tracing::warn!(job_id = %job_id, rank, error = %e, "rank body: this host's store faulted binding the training set");
            return RankOutcome::Aborted(AbortReason::StoreUnavailable);
        }
        Err(e) => {
            tracing::warn!(job_id = %job_id, rank, error = %e, "rank body: the catalog faulted binding the training set");
            return RankOutcome::Aborted(AbortReason::Unavailable);
        }
    };

    // The per-partition leaf verify, before the first collective.
    let parquet_url = match jammi_db::storage::StorageUrl::parse(table.parquet_path()) {
        Ok(url) => url,
        Err(e) => return failed(format!("the training set's parquet path: {e}")),
    };
    let handle = match session.result_store().open_parquet(&parquet_url) {
        Ok(handle) => handle,
        Err(e) => {
            tracing::warn!(job_id = %job_id, rank, error = %e, "rank body: this host's store cannot open the training set");
            return RankOutcome::Aborted(AbortReason::StoreUnavailable);
        }
    };
    if let Err(fault) =
        crate::fine_tune::collective::peer::verify_partition_leaves(&handle, &manifest.leaves).await
    {
        tracing::warn!(job_id = %job_id, rank, %fault, "rank body: a leaf of this rank's partition did not verify");
        return RankOutcome::Aborted(fault.abort_reason());
    }

    let detected = match detect_training_format(&columns, task) {
        Ok(detected) => detected,
        Err(e) => return failed(e.to_string()),
    };
    let training_source =
        match bind_training_source(session, &table, &columns, task, detected, &common).await {
            Ok(source) => source,
            Err(e) => return failed(e.to_string()),
        };

    let model_source = ModelSource::parse(&common.base_model);
    let guard = match session
        .model_cache()
        .get_or_load(&model_source, task, None)
        .await
    {
        Ok(guard) => guard,
        Err(e) => return failed(e.to_string()),
    };
    let base_model_arc = Arc::clone(&guard.model);
    let Some(hidden_size) = guard.model.embedding_dim() else {
        return failed("Base model does not support embeddings".into());
    };
    drop(guard);

    let device_config = session.device_config().clone();
    let device = match crate::model::backend::candle::select_device(&device_config) {
        Ok(device) => device,
        Err(e) => return failed(e.to_string()),
    };
    let max_message_bytes = usize::try_from(session.inner_config().server.limits.max_message_bytes)
        .unwrap_or(usize::MAX);
    let rank_timeout = Duration::from_secs(session.inner_config().worker.rank_timeout_secs);
    let peer = match Peer::member(rank, world, link, device, max_message_bytes)
        .and_then(|peer| peer.with_timeout(rank_timeout))
    {
        Ok(peer) => peer,
        Err(e) => return failed(e.to_string()),
    };
    let partition = match PartitionSpec::for_gang(
        rank as usize,
        world as usize,
        common.config.batch_size,
        PartitionRule::BlockByGlobalBatch,
    ) {
        Ok(partition) => partition,
        Err(e) => return failed(e.to_string()),
    };
    let collective: Arc<dyn Collective> = Arc::new(peer);
    // `test-hooks`: a chaos wrapper an oracle armed for this job's next
    // member body (the chaos rows' fault injection inside the real body).
    #[cfg(feature = "test-hooks")]
    let collective = match training_test_hooks::take_member_collective_wrap(&job_id) {
        Some(wrap) => wrap(collective),
        None => collective,
    };
    let rank_ctx = RankContext::new(collective, partition);

    let params = RunFineTuneParams {
        catalog,
        artifact_store: session.artifact_store(),
        artifact_dir: session.inner_config().artifact_dir.clone(),
        job_id: job_id.clone(),
        worker_id: coordinator_instance_id,
        attempt,
        role: RunnerRole::Rank { rank },
        rank_ctx: Some(rank_ctx),
        base_model: common.base_model.clone(),
        task,
        config: common.config,
        source: training_source,
        base_model_arc,
        hidden_size,
        device_config,
        cancel,
        hub: session.hub().clone(),
    };

    // The third production minting site: this rank's `TrainingLoop::run`
    // receives the witness minted at its own blocking-pool boundary.
    let result = BlockingCall::spawn_blocking(move |call| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_fine_tune_blocking(&call, params)
        }))
    })
    .await;
    let training = match result {
        Ok(Ok(Ok(training))) => training,
        Ok(Ok(Err(e))) => return failed(e.to_string()),
        Ok(Err(payload)) => return failed(format!("Panic: {}", panic_message(payload.as_ref()))),
        Err(join_err) => return failed(format!("training task join error: {join_err}")),
    };
    #[cfg(feature = "test-hooks")]
    if let Some(reason) = training_test_hooks::take_member_failure(&job_id) {
        return failed(reason);
    }
    match adapter_files_digest(training.artifact_dir.path()) {
        Ok(artifact_digest) => RankOutcome::Trained { artifact_digest },
        Err(e) => failed(format!("this rank's adapter digest: {e}")),
    }
}

/// The catalog model-row descriptor a training kind hands the worker's finalize.
///
/// Holds every column of the output `models` row except where its bytes
/// live. The row itself is written solely by the lease-guarded finalize
/// transaction, referencing the artifact that transaction publishes — so an
/// attempt that does not win the finalize leaves no `models` row at all.
#[derive(Debug)]
pub struct ModelRegistration {
    /// Deterministic model id (`jammi:fine-tuned:{job_id}`, or the predictor's
    /// configured id) — the finalize upserts on it.
    pub model_id: String,
    /// Catalog version this row registers under. Every training kind
    /// registers its output at `1`.
    pub version: i32,
    /// `"fine-tuned"` or `"context-predictor"`.
    pub model_type: &'static str,
    /// The model's task.
    pub task: ModelTask,
    /// The base model this was derived from, if any.
    pub base_model_id: Option<String>,
    /// Architecture/config JSON the reload path reads, if any.
    pub config_json: Option<String>,
}

impl ModelRegistration {
    /// The `models` row a finalize writes for this registration under
    /// `name`.
    fn row<'a>(&'a self, name: &'a str) -> jammi_db::catalog::jobs_repo::ModelRow<'a> {
        jammi_db::catalog::jobs_repo::ModelRow {
            model_id: name,
            version: self.version,
            model_type: self.model_type,
            backend: "candle",
            task: self.task,
            base_model_id: self.base_model_id.as_deref(),
            config_json: self.config_json.as_deref(),
        }
    }
}

/// Read every regular file directly under `dir` into `(name, bytes)` and
/// stage them as the bundle this attempt serves, under the unique
/// per-attempt prefix `{job_id}/{worker_id}/{attempt}`. The three segments
/// are jointly unique per attempt (`job_id` is the PK, `worker_id`
/// distinguishes a lost-lease worker from its re-claimer, `attempt`
/// distinguishes a reclaimed re-run), so no two attempts ever target the
/// same prefix and no object is overwritten. `catalog` is the job's
/// tenant-pinned catalog: its bound tenant owns the staged row and the
/// prefix's tenant segment.
///
/// **A bundle is a flat directory.** Only top-level files are staged (the
/// trainer's checkpoint subdirectories are training scratch, not part of the
/// served artifact): reading `dir` non-recursively here is this function's
/// half of the invariant a reclaim licence relies on — it covers exactly the
/// keys DIRECTLY inside its artifact's prefix; the job's epoch checkpoints
/// are their own artifacts under the job's own `_checkpoints` prefix
/// ([`ArtifactStore::stage_checkpoint`]), never beneath an attempt's. Proven
/// by a test that walks the physical object tree a real run writes
/// (`fine_tune_materialization::every_published_object_sits_flat_under_its_own_row`).
async fn publish_artifact(
    store: &ArtifactStore,
    catalog: &Catalog,
    job_id: &str,
    worker_id: &str,
    attempt: u32,
    dir: &tempfile::TempDir,
) -> Result<StagedArtifact> {
    let mut files: Vec<(String, Bytes)> = Vec::new();
    for entry in std::fs::read_dir(dir.path())? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let bytes = std::fs::read(entry.path())?;
        files.push((name, Bytes::from(bytes)));
    }
    store
        .stage_attempt_artifact(catalog, job_id, worker_id, attempt, &files)
        .await
}

/// Reclaim every artifact `attempt` of `job_id` staged and did not publish —
/// the ONE sweep every terminating arm of an attempt ends with, the finalize
/// winner's included. The set is read from the catalog
/// ([`Catalog::staged_artifacts_of_attempt`]), never from whatever the
/// attempt still holds in memory, so it is the same sweep whether the run
/// bailed mid-training, failed to stage its final bundle, lost its finalize,
/// or won it. The job's epoch checkpoints are not this sweep's: they belong
/// to the job, are read by its next attempt, and are reclaimed once the job
/// is terminal ([`reclaim_checkpoints`]). Each is reclaimed as the stager's
/// own bundle: the reclaim compare-and-set, then the licensed byte delete.
///
/// Best-effort: a failure leaves the artifact in the catalog for a reconcile
/// pass, and emits exactly ONE warning per sweep naming the failed-vs-
/// attempted count — never zero, never one per artifact.
async fn reclaim_unpublished_artifacts(
    store: &ArtifactStore,
    catalog: &Catalog,
    job_id: &str,
    worker_id: &str,
    attempt: u32,
) {
    let held = match catalog.staged_artifacts_of_attempt(job_id, attempt).await {
        Ok(held) => held,
        Err(e) => {
            tracing::warn!(
                job_id = %job_id,
                worker_id = %worker_id,
                attempt,
                error = %e,
                "could not list this attempt's unpublished artifacts; a reconcile pass \
                 reclaims them"
            );
            return;
        }
    };
    let attempted = held.len();
    let mut failed = 0usize;
    for staged in held {
        let artifact = staged.artifact().clone();
        let decision = catalog.reclaim_own_staged_artifact(staged).await;
        if !settle_reclaim(store, catalog, &artifact, decision).await {
            failed += 1;
        }
    }
    if failed > 0 {
        tracing::warn!(
            job_id = %job_id,
            worker_id = %worker_id,
            attempt,
            failed,
            attempted,
            "unpublished-artifact sweep: {failed} of {attempted} reclaim(s) failed — a \
             reconcile pass reclaims them"
        );
    }
}

/// The job is terminal, so no attempt will read its epoch checkpoints
/// again: whatever a finalize did not publish is reclaimed through the store
/// ([`ArtifactStore::reclaim_checkpoints`]). Best-effort like the sweep — a
/// refusal or failure leaves the epoch for a reconcile pass, and emits
/// exactly ONE warning naming how many.
async fn reclaim_checkpoints(store: &ArtifactStore, catalog: &Catalog, job_id: &str) {
    match store.reclaim_checkpoints(catalog, job_id).await {
        Ok(unsettled) if unsettled.is_empty() => {}
        Ok(unsettled) => tracing::warn!(
            job_id = %job_id,
            unsettled = unsettled.len(),
            "the ended job's checkpoints were not fully reclaimed; a reconcile pass reclaims \
             them"
        ),
        Err(e) => tracing::warn!(
            job_id = %job_id,
            error = %e,
            "could not list the ended job's checkpoints"
        ),
    }
}

/// Act on a reclaim compare-and-set's `decision` for `artifact`: delete its
/// bytes under the licence. Returns whether the artifact is settled — its
/// bytes reclaimed, or nothing of it left to reclaim. A refusal (a `models`
/// row references it, or its stager is still live) and any error are
/// unsettled, logged here with the artifact.
async fn settle_reclaim(
    store: &ArtifactStore,
    catalog: &Catalog,
    artifact: &jammi_db::catalog::artifact_repo::ArtifactRef,
    decision: Result<ReclaimDecision>,
) -> bool {
    let reclaimed = match decision {
        Ok(ReclaimDecision::Licensed(licence)) => {
            store.reclaim(catalog, licence, &[]).await.map(|_| true)
        }
        Ok(ReclaimDecision::Absent) => Ok(true),
        Ok(ReclaimDecision::Referenced | ReclaimDecision::Live) => Ok(false),
        Err(e) => Err(e),
    };
    match reclaimed {
        Ok(true) => true,
        Ok(false) => {
            tracing::debug!(%artifact, "reclaim refused: the artifact is referenced or still being written");
            false
        }
        Err(e) => {
            tracing::debug!(%artifact, error = %e, "reclaim failed");
            false
        }
    }
}

/// What [`JobWorker::publish_and_finalize`] did — the ONE fact
/// [`JobWorker::run_claimed_job_under`] needs to derive its own
/// [`AttemptEnd`] without re-deriving it from a second row read.
enum PublishOutcome {
    /// The finalize CAS committed `completed`.
    Completed,
    /// A terminal `failed` was already recorded (by this function), with
    /// this reason.
    Failed(String),
    /// No terminal write: the finalize CAS lost the race (this instance's
    /// lease was gone by the time it ran) — left `running` for reclaim.
    LeftForReclaim,
}

/// What one attempt of [`JobWorker::run_claimed_job_under`] ended as — the
/// fact [`JobWorker::run_placed_gang`] maps onto
/// [`crate::operator::gang_exec::PlacedOutcome`] without a second row read.
/// `run_claimed_job`/the claim loop discard it; both already observe every
/// row write this type merely reports.
enum AttemptEnd {
    /// The attempt published `completed`; this is the artifact's own digest
    /// (`adapter_files_digest`, computed over the SAME directory
    /// `publish_and_finalize` just uploaded from).
    Published { artifact_digest: String },
    /// The attempt completed by reusing a published artifact: no bytes of
    /// its own, so no digest of its own.
    Reused,
    /// A terminal unsuccessful status (`failed`, or `cancelled` for an
    /// honoured cancel) was recorded; `error` is the typed failure whose
    /// message ([`failed_job_message`]) the row carries — a placed attempt
    /// hands it to its submitter as the task's own error.
    Failed { error: JammiError },
    /// No terminal write: left `running` for reclaim (a lease loss, a
    /// finalize race lost, a mid-run gang abandon, or a hand-off to a
    /// placed executor).
    LeftForReclaim,
}

/// The terminal classification of a worker's run of one job.
enum WorkerJobError {
    /// The lease was lost mid-training; the job is left `running` for reclaim.
    Cancelled,
    /// The job failed for a real reason; record it as `failed` + the error's
    /// message ([`failed_job_message`]), and keep the error typed for the
    /// attempt's own end.
    Failed(JammiError),
    /// The coordinator body ended this attempt WITHOUT a run reaching a
    /// terminal state (an assembly outcome, or a gang fault mid-run — see
    /// [`CoordinatorEnd`]): no terminal write, the assembly outcome already
    /// recorded on the row, and the lease settled by the released-vs-failed
    /// split ([`lease_settlement`]) — handed back (`release_job_lease`) or
    /// left to expire — either way the row stays `running` for reclaim (the
    /// fleet's only requeue path) and the next attempt
    /// re-assembles once its cooldown passes. The string is the reason, for
    /// the log.
    Abandoned(String),
    /// The claim moved to a placed executor mid-attempt: the submitter's stream ended with at least
    /// one batch, or ended in error/emptily AFTER `Catalog::transfer_claim`
    /// already moved `claimed_by` off this instance. NO terminal write, NO
    /// release — the row is another process's now, and this process's
    /// lease keeper registration for the attempt is already dropped by the
    /// time this arm is reached (the same `drop(hold); drop(cancel_watcher);`
    /// every other end goes through) — a heartbeat from this stale holder
    /// can never resurrect the lease (`Catalog::heartbeat_job` keys on
    /// `claimed_by`, proven by
    /// `crates/jammi-ai/tests/it/gang_placed.rs::the_submitters_heartbeat_after_hand_off_never_resurrects_the_executors_lease`).
    HandedOff,
}

/// Render a terminal training failure's message for `record_failed` to
/// persist as the job's durable `error_message`.
///
/// The stored value is the RAW inner message, never pre-fixed with
/// "Fine-tune error: " here. That prefix is `JammiError::FineTune`'s own
/// `Display` output, and two callers re-wrap the stored message in a fresh
/// `JammiError::FineTune` when they read a `failed` job back —
/// `TrainingJob::wait()` (`training_job.rs`) and the Python binding's
/// `poll_until_terminal` (`jammi-python/src/job.rs`) — so those two surfaces
/// each apply the prefix exactly once, on read. Two OTHER surfaces read the
/// same durable `error_message` unprefixed and never re-wrap it: the gRPC
/// `JobStatus.error` field (`jammi-server/src/grpc/job.rs`) and
/// the Python `Database.list_training_jobs`/`get_training_job` `error` entry
/// (`jammi-python/src/database.rs`) both relay the raw column verbatim.
/// Storing `e.to_string()` unconditionally for a `FineTune`-typed source
/// error would have `wait()`'s (or `poll_until_terminal`'s) re-wrap double
/// the prefix verbatim: "Fine-tune error: Fine-tune error: …" reaching a
/// Python caller as `TrainingError("Fine-tune error: Fine-tune error:
/// …")`. Stripped to the raw inner message for that one variant here avoids
/// that double; every other variant's own (DIFFERENT) prefix is preserved
/// unchanged — "Fine-tune error: Model error: …" is one informative
/// nesting, not a literal duplicate.
fn failed_job_message(e: &JammiError) -> String {
    match e {
        JammiError::FineTune(msg) => msg.clone(),
        other => other.to_string(),
    }
}

impl From<JammiError> for WorkerJobError {
    fn from(e: JammiError) -> Self {
        WorkerJobError::Failed(e)
    }
}

/// Wrap a training failure that names an out-of-memory condition in
/// actionable guidance, so `jammi train status` / a Python `job.status()`
/// surface the config that OOM'd and what to try — instead of a raw driver
/// string — via [`crate::model::oom::is_definite_oom_message`], the strict
/// (long-spellings-only) predicate (its home explains why the training
/// classifier needs a stricter match than the inference retry's predicate:
/// this function's output is durable and caller-facing). A non-OOM error
/// passes through byte-identical.
///
/// The echoed config and the remedies both differ by adapter shape, because
/// `backbone_dtype` only takes effect on the encoder-adapters arm:
/// `build_encoder_adapters` — reached only when `config.target_modules` is
/// non-empty — is the sole caller of [`validate_backbone_precision`] and
/// `compute_precision_to_dtype`. The projection-head arm (`target_modules`
/// empty, the default) loads the frozen backbone but never re-dtypes it, so
/// `backbone_dtype` is omitted from the echoed config entirely on that arm
/// (never echoed and then disclaimed), and recommending `backbone_dtype:
/// bf16` there would be dead advice.
///
/// - **Encoder adapters** (`target_modules` non-empty): config echo includes
///   `backbone_dtype`; (1) `backbone_dtype: bf16` first — substantially
///   reduces the frozen backbone's weight and activation residency; bf16
///   requires CUDA, refused before training starts if unmet (see
///   [`validate_backbone_precision`]); (2) a smaller `batch_size`, or trade
///   batch size for `gradient_accumulation_steps`; (3) a smaller
///   `max_seq_length`.
/// - **Projection head** (`target_modules` empty): config echo omits
///   `backbone_dtype`; the message states outright that it does not apply;
///   (1) a smaller `batch_size`, or trade batch size for
///   `gradient_accumulation_steps`; (2) a smaller `max_seq_length`.
///
/// The headline names the mechanism only as strongly as the matched text
/// supports: this function never threads through which device the job
/// actually ran on, so it says "CUDA out of memory" only when the error text
/// itself mentions CUDA, and the more conservative "out of memory (device or
/// host)" otherwise.
fn classify_training_oom(config: &FineTuneConfig, e: JammiError) -> JammiError {
    let msg = e.to_string();
    let msg_lower = msg.to_lowercase();
    if !crate::model::oom::is_definite_oom_message(&msg_lower) {
        return e;
    }
    let headline = if msg_lower.contains("cuda") {
        "CUDA out of memory"
    } else {
        "out of memory (device or host)"
    };
    // The echoed config is arm-appropriate, not echo-then-disclaim: the
    // projection-head arm never reads `backbone_dtype` for anything, so it
    // is simply absent from the echo rather than named and then immediately
    // disclaimed.
    let (config_echo, remedies) = if config.target_modules.is_empty() {
        (
            format!(
                "batch_size={}, max_seq_length={}",
                config.batch_size, config.max_seq_length
            ),
            "backbone_dtype does not apply to projection-head runs. Try, in order: \
             (1) a smaller batch_size, or trade batch size for \
             gradient_accumulation_steps; (2) a smaller max_seq_length."
                .to_string(),
        )
    } else {
        (
            format!(
                "batch_size={}, max_seq_length={}, backbone_dtype={}",
                config.batch_size, config.max_seq_length, config.backbone_dtype
            ),
            "Try, in order: (1) backbone_dtype: bf16 — substantially reduces backbone \
             + activation residency; bf16 requires CUDA (refused before training \
             starts if unmet); (2) a smaller batch_size, or trade batch size for \
             gradient_accumulation_steps; (3) a smaller max_seq_length."
                .to_string(),
        )
    };
    JammiError::FineTune(format!(
        "{headline} while training ({config_echo}). {remedies} Underlying error: {msg}"
    ))
}

/// Classify a training error: a cancellation (lease lost) maps to
/// [`WorkerJobError::Cancelled`] so the job is left for reclaim; anything else is
/// a genuine failure. The cancel flag is the authoritative signal; the error
/// message is the fallback for the blocking path where the flag is not threaded
/// back to this scope.
fn classify(cancel: &AtomicBool, e: JammiError) -> WorkerJobError {
    let cancelled =
        cancel.load(Ordering::Relaxed) || e.to_string().contains("training cancelled: lease lost");
    if cancelled {
        WorkerJobError::Cancelled
    } else {
        WorkerJobError::Failed(e)
    }
}

/// The error-arm lattice for a blocking training run's `Err(JammiError)`
/// result: cancellation takes priority — a lease-lost job is left `running`
/// for reclaim, never rewritten as an OOM failure even when the error text
/// happens to look OOM-shaped — otherwise the failure is OOM-classified
/// against the config that produced it ([`classify_training_oom`]), passed
/// through byte-identical when it doesn't match.
///
/// Pulled out of `train_fine_tune`'s single call site so the full lattice —
/// cancelled+oom-text, oom, non-oom — is unit-testable directly, not only
/// exercised end-to-end through the blocking-trainer wiring. That single
/// production call site remains wiring-by-inspection: nothing here re-checks
/// that `train_fine_tune` actually calls this function on its `Err` arm.
fn classify_training_error(
    cancel: &AtomicBool,
    config: &FineTuneConfig,
    e: JammiError,
) -> WorkerJobError {
    classify(cancel, classify_training_oom(config, e))
}

// =========================================================================
// Reconstruction helpers (the data-loading + blocking-training tail; the
// worker is their only consumer). The Arrow decode
// itself (`extract_string_column`/`extract_numeric_column`/
// `build_training_data_loader`/`detect_training_format`/`DetectedFormat`/
// `NumericColumnError`) lives in `super::decode` — the ONE decoder
// both this worker's eager reconstruction and `super::stream`'s per-rank
// stream call, brought into scope below.
// =========================================================================

/// Extract a human-readable message from a panic payload.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<unknown panic payload>".into()
    }
}

/// Poll `jobs.cancel_requested` for `job_id` at `poll_interval` — the
/// SAME cadence the lease keeper renews this job's lease at
/// (`JobWorker`'s own `self.intervals.heartbeat`), never per training step —
/// and, on an observed request, flip `cancel_requested_seen` (a one-way
/// marker, set FIRST) then the shared `cancel` flag the training loop already
/// checks at every epoch boundary ([`JobWorker::run_claimed_job`]'s doc). The
/// SAME shared `cancel` flag is also the lease keeper's own `lost` flag
/// (`LeaseHold::lost_flag`): both writers only ever store `true`, so a second
/// writer here is never a race to arbitrate, only a second way to reach the
/// one outcome the training loop's check already understands — but it lets
/// [`JobWorker::run_claimed_job`] read `cancel_requested_seen` afterward to
/// tell the two causes apart and land the right terminal write for each.
///
/// One-shot: the task returns as soon as it has flipped the flag itself, or
/// as soon as it observes `cancel` already `true` for any other reason (a
/// lease loss — nothing left here to watch for), or as soon as it observes
/// `attempt_alive` gone `false` (belt-and-braces: the SAME
/// signal [`CancelWatcherGuard::drop`] flips right before it also
/// `abort()`s this task — a caller must never need this second read to
/// reclaim the task, `abort()` alone already guarantees that, but a check
/// the loop makes of its own accord costs nothing and does not depend on
/// `abort()`'s cooperative cancellation actually landing before this tick's
/// `get_job` round-trip starts). The caller holds this behind
/// [`CancelWatcherGuard`] so it never outlives the job attempt it was
/// spawned for — see that type's doc for why a bare `JoinHandle` is not
/// enough. A transient `get_job` error is logged and retried at the next
/// tick rather than treated as an observed request or a reason to stop
/// watching — a catalog hiccup must not silently disable cancellation for
/// the rest of the run.
fn spawn_cancel_request_watcher(
    catalog: Arc<Catalog>,
    job_id: String,
    cancel: Arc<AtomicBool>,
    cancel_requested_seen: Arc<AtomicBool>,
    attempt_alive: Arc<AtomicBool>,
    poll_interval: Duration,
) -> CancelWatcherGuard {
    let attempt_alive_for_task = Arc::clone(&attempt_alive);
    let handle = tokio::spawn(async move {
        loop {
            tokio::time::sleep(poll_interval).await;
            if cancel.load(Ordering::SeqCst) {
                // Already tripped by some other path (a lease loss) — nothing
                // left for this watcher to contribute.
                return;
            }
            if !attempt_alive_for_task.load(Ordering::SeqCst) {
                // Belt-and-braces: the attempt that spawned
                // this watcher is gone — `CancelWatcherGuard::drop` has
                // already called (or is concurrently calling) `abort()` on
                // this very task, but this read means the loop stops of its
                // own accord even in the narrow window before that
                // cooperative cancellation lands.
                return;
            }
            match catalog.get_job(&job_id).await {
                Ok(record) if record.cancel_requested => {
                    cancel_requested_seen.store(true, Ordering::SeqCst);
                    cancel.store(true, Ordering::SeqCst);
                    #[cfg(feature = "test-hooks")]
                    loop_test_hooks::fire_observed(&job_id, loop_test_hooks::Event::CancelObserved);
                    return;
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::debug!(
                        job_id = %job_id,
                        error = %e,
                        "cancel-request watcher: get_job failed; retrying at the next poll"
                    );
                }
            }
        }
    });
    CancelWatcherGuard {
        handle,
        attempt_alive,
    }
}

/// Abort-on-drop guard around the cancel-request watcher's
/// [`tokio::task::JoinHandle`].
///
/// A bare `JoinHandle` DETACHES its task when dropped — it does not stop it
/// (`tokio::task::JoinHandle`'s own documented behaviour) — so the explicit
/// `cancel_watcher.abort()` call this replaced, reached only on every
/// ORDINARY exit of [`JobWorker::run_claimed_job`] (`Ok`, both `Err` arms),
/// left exactly one path uncovered: that whole `.await` being dropped out
/// from under the function without any of its own code ever running again —
/// exactly what [`EmbeddedWorker::drop`] does to the loop task that owns it
/// (see that type's doc: an in-flight run is not aborted, but the LOOP TASK
/// itself is, at its next `.await` point, which is squarely inside this
/// function whenever a job is claimed). A dropped `JoinHandle` in that case
/// only detaches the watcher — never stops it — leaving it polling
/// `catalog.get_job` forever on an `Arc<Catalog>` clone that can outlive
/// [`Catalog::close`].
///
/// Wrapping the handle in this guard and holding it for the whole attempt
/// closes every exit arm at once, because Rust always runs a live local's
/// `Drop` on every one of them: the ordinary `Ok`/`Err` returns (via the
/// explicit `drop(cancel_watcher)` in [`JobWorker::run_claimed_job`]), a
/// panic unwinding through that scope, AND the future simply being
/// dropped — the one case a reachable `.abort()` call can never cover,
/// because no code gets to run to make it.
struct CancelWatcherGuard {
    handle: tokio::task::JoinHandle<()>,
    /// The write side of the watcher's belt-and-braces flag — see
    /// [`spawn_cancel_request_watcher`]'s doc for the read side.
    attempt_alive: Arc<AtomicBool>,
}

impl CancelWatcherGuard {
    /// The watcher's [`tokio::task::AbortHandle`] — `Clone`, so a test can
    /// hold one independently of this guard (which owns the only
    /// `JoinHandle`) and observe `is_finished()` after the guard drops,
    /// without racing a join.
    #[cfg(feature = "test-hooks")]
    fn abort_handle(&self) -> tokio::task::AbortHandle {
        self.handle.abort_handle()
    }
}

impl Drop for CancelWatcherGuard {
    fn drop(&mut self) {
        self.attempt_alive.store(false, Ordering::SeqCst);
        self.handle.abort();
    }
}

/// Test-only rendezvous for the cancel-watcher lifetime coverage:
/// mirrors `crate::jobs::compute_test_hooks`'s pattern (a park point a test
/// arms, then waits for) but for the training path, plus a small
/// job-id-keyed registry that hands out the primitives needed to observe
/// [`CancelWatcherGuard`] actually releasing its task and its `Arc<Catalog>`
/// clone from OUTSIDE this module — [`JobWorker::run_claimed_job`]'s own
/// `catalog` and `cancel_watcher` locals are private to that function, so a
/// test cannot reach either one directly. Compiled only under
/// `feature = "test-hooks"` (this crate's own test targets enable it
/// through a self dev-dependency); no production path observes anything
/// here.
#[cfg(feature = "test-hooks")]
pub mod training_test_hooks {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

    use jammi_db::catalog::Catalog;
    use tokio::sync::oneshot;

    static THREADS_FINISHED: AtomicUsize = AtomicUsize::new(0);

    /// How many `spawn_blocking` training threads have returned in this
    /// process — counted inside the blocking closure itself, so a thread
    /// whose owning future was aborted (a RELEASE mid-epoch) is still counted
    /// the moment it bails. A test that must prove "the abandoned attempt
    /// never finalized" waits for this to advance, then reads the row.
    pub fn training_threads_finished() -> usize {
        THREADS_FINISHED.load(Ordering::SeqCst)
    }

    pub(super) fn note_training_thread_finished() {
        THREADS_FINISHED.fetch_add(1, Ordering::SeqCst);
    }

    /// One recorded source-kind observation, keyed by the job it was
    /// selected for. A `Vec` rather than a `HashMap`, mirroring
    /// [`WatcherProbe`]'s own reasoning: a retried job can re-select a
    /// (possibly different) source kind under a later attempt, so a caller
    /// looks up the MOST RECENT entry.
    struct SourceKindProbe {
        job_id: String,
        kind: &'static str,
    }

    fn source_kinds() -> &'static Mutex<Vec<SourceKindProbe>> {
        static PROBES: OnceLock<Mutex<Vec<SourceKindProbe>>> = OnceLock::new();
        PROBES.get_or_init(|| Mutex::new(Vec::new()))
    }

    /// Record which [`crate::fine_tune::source::TrainingSource`] variant
    /// `run_spec`'s FineTune arm bound for `job_id` — `"resident"` or
    /// `"streamed"` — a `test-hooks` observation that proves the worker
    /// really bound `Streamed`. The streamed-vs-eager parity fixtures
    /// read this back through [`source_kind_for`] to prove the pinned
    /// adapter prints they assert on were actually produced by the streamed
    /// path, not a silently-unchanged eager one.
    pub(super) fn note_source_kind(job_id: &str, kind: &'static str) {
        source_kinds()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(SourceKindProbe {
                job_id: job_id.to_string(),
                kind,
            });
    }

    /// The most recently recorded source kind for `job_id` — `None` if this
    /// job never reached `run_spec`'s `TrainingSpec::FineTune` arm (a
    /// `GraphFineTune` run, or a job that hasn't been claimed yet).
    pub fn source_kind_for(job_id: &str) -> Option<&'static str> {
        source_kinds()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .rev()
            .find(|p| p.job_id == job_id)
            .map(|p| p.kind)
    }

    /// One recorded `StreamedSet::total_rows` observation, keyed by the job
    /// it was built for — the same "most recent entry wins" shape as
    /// [`SourceKindProbe`], for the same retry reason.
    struct StreamedRowsProbe {
        job_id: String,
        total_rows: usize,
    }

    fn streamed_rows() -> &'static Mutex<Vec<StreamedRowsProbe>> {
        static PROBES: OnceLock<Mutex<Vec<StreamedRowsProbe>>> = OnceLock::new();
        PROBES.get_or_init(|| Mutex::new(Vec::new()))
    }

    /// Record the row count `run_spec`'s FineTune arm built the `Streamed`
    /// source over for `job_id` — `StreamedSet::total_rows`, the catalog
    /// record's own `row_count` (the tenant-isolation oracle: the row count
    /// the stream served equals the tenant's own).
    pub(super) fn note_streamed_total_rows(job_id: &str, total_rows: usize) {
        streamed_rows()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(StreamedRowsProbe {
                job_id: job_id.to_string(),
                total_rows,
            });
    }

    /// The most recently recorded `StreamedSet::total_rows` for `job_id` —
    /// `None` if this job's FineTune arm never bound `Streamed`.
    pub fn streamed_total_rows_for(job_id: &str) -> Option<usize> {
        streamed_rows()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .rev()
            .find(|p| p.job_id == job_id)
            .map(|p| p.total_rows)
    }

    /// One recorded topology decision per `run_spec` FineTune/GraphFineTune
    /// arm entry, keyed by job — the most recent entry wins, as above.
    fn topologies() -> &'static Mutex<Vec<(String, super::TopologyDecision)>> {
        static PROBES: OnceLock<Mutex<Vec<(String, super::TopologyDecision)>>> = OnceLock::new();
        PROBES.get_or_init(|| Mutex::new(Vec::new()))
    }

    pub(super) fn note_topology(job_id: &str, topology: super::TopologyDecision) {
        topologies()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((job_id.to_string(), topology));
    }

    /// One recorded graph-sample fingerprint, keyed by job — the most recent
    /// entry wins, the same "retried job re-selects" shape as
    /// [`SourceKindProbe`]. The read-order oracle: two `GraphFineTune`
    /// runs whose node/edge sources hold the SAME set of rows in DIFFERENT
    /// physical layouts must record the SAME fingerprint here — the
    /// read-order rule making the sample a function of the input SET, not of
    /// scan order.
    struct GraphSampleProbe {
        job_id: String,
        fingerprint: String,
    }

    fn graph_samples() -> &'static Mutex<Vec<GraphSampleProbe>> {
        static PROBES: OnceLock<Mutex<Vec<GraphSampleProbe>>> = OnceLock::new();
        PROBES.get_or_init(|| Mutex::new(Vec::new()))
    }

    pub(super) fn note_graph_sample_fingerprint(job_id: &str, fingerprint: String) {
        graph_samples()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(GraphSampleProbe {
                job_id: job_id.to_string(),
                fingerprint,
            });
    }

    /// The most recently recorded graph-sample fingerprint for `job_id` —
    /// `None` if this job never sampled a graph.
    pub fn graph_sample_fingerprint_for(job_id: &str) -> Option<String> {
        graph_samples()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .rev()
            .find(|p| p.job_id == job_id)
            .map(|p| p.fingerprint.clone())
    }

    /// One recorded `training_set_graph_sample` reservation size, keyed by
    /// job: the bytes `materialize_graph_training_set` actually reserved against the NAMED `MemoryConsumer`, so a test can
    /// assert it against an independently computed lower bound rather than
    /// trust the reservation call succeeded silently.
    struct GraphReservationProbe {
        job_id: String,
        bytes: usize,
    }

    fn graph_reservations() -> &'static Mutex<Vec<GraphReservationProbe>> {
        static PROBES: OnceLock<Mutex<Vec<GraphReservationProbe>>> = OnceLock::new();
        PROBES.get_or_init(|| Mutex::new(Vec::new()))
    }

    pub(super) fn note_graph_sample_reservation_bytes(job_id: &str, bytes: usize) {
        graph_reservations()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(GraphReservationProbe {
                job_id: job_id.to_string(),
                bytes,
            });
    }

    /// The most recently recorded `training_set_graph_sample` reservation
    /// size for `job_id` — `None` if this job's reservation was never
    /// attempted (or failed before this hook fires).
    pub fn graph_sample_reservation_bytes_for(job_id: &str) -> Option<usize> {
        graph_reservations()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .rev()
            .find(|p| p.job_id == job_id)
            .map(|p| p.bytes)
    }

    /// One recorded event in a graph sample's reservation lifecycle:
    /// `"write_committed"` right after `materialize_training_
    /// set` returns, `"released"` when `ReservationGuard::drop` runs. `seq`
    /// is a single, global, monotonically increasing counter shared by
    /// every event of both kinds — so two events for the SAME job compare
    /// by real happens-before order, not by wall-clock time (which two
    /// events in the same async task can share to the clock's resolution).
    struct GraphReservationLifecycleEvent {
        job_id: String,
        seq: u64,
        released: bool,
    }

    fn graph_reservation_lifecycle() -> &'static Mutex<Vec<GraphReservationLifecycleEvent>> {
        static EVENTS: OnceLock<Mutex<Vec<GraphReservationLifecycleEvent>>> = OnceLock::new();
        EVENTS.get_or_init(|| Mutex::new(Vec::new()))
    }

    fn next_graph_reservation_lifecycle_seq() -> u64 {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }

    pub(super) fn note_graph_sample_write_committed(job_id: &str) {
        graph_reservation_lifecycle()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(GraphReservationLifecycleEvent {
                job_id: job_id.to_string(),
                seq: next_graph_reservation_lifecycle_seq(),
                released: false,
            });
    }

    pub(super) fn note_graph_sample_reservation_released(job_id: &str) {
        graph_reservation_lifecycle()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(GraphReservationLifecycleEvent {
                job_id: job_id.to_string(),
                seq: next_graph_reservation_lifecycle_seq(),
                released: true,
            });
    }

    /// The release-timing oracle: `Some(true)` iff `job_id`'s
    /// graph-sample reservation was released STRICTLY AFTER its write
    /// committed (by `seq`, not wall-clock); `Some(false)` if release
    /// preceded the write (the bug this oracle catches); `None` if either
    /// event was never recorded for this job.
    pub fn graph_sample_reservation_released_after_write_for(job_id: &str) -> Option<bool> {
        let events = graph_reservation_lifecycle()
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let write_seq = events
            .iter()
            .rev()
            .find(|e| e.job_id == job_id && !e.released)
            .map(|e| e.seq)?;
        let release_seq = events
            .iter()
            .rev()
            .find(|e| e.job_id == job_id && e.released)
            .map(|e| e.seq)?;
        Some(release_seq > write_seq)
    }

    /// The most recent [`super::TopologyDecision`] `run_spec` made for
    /// `job_id` — the oracle that a claimed job really fanned out into the
    /// layout its `world_size` and `[worker] local_ranks` imply.
    pub fn topology_for(job_id: &str) -> Option<super::TopologyDecision> {
        topologies()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .rev()
            .find(|(id, _)| id == job_id)
            .map(|(_, topology)| *topology)
    }

    /// One recorded rank assignment per coordinator attempt: `(attempt,
    /// [(rank, instance_id)])`, in the order the assignment was made.
    type Listing = (String, u32, Vec<(u32, String)>);

    fn listings() -> &'static Mutex<Vec<Listing>> {
        static PROBES: OnceLock<Mutex<Vec<Listing>>> = OnceLock::new();
        PROBES.get_or_init(|| Mutex::new(Vec::new()))
    }

    pub(super) fn note_assembly_listing(
        job_id: &str,
        attempt: u32,
        assignment: Vec<(u32, String)>,
    ) {
        listings()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((job_id.to_string(), attempt, assignment));
    }

    /// Every rank assignment the coordinator body made for `job_id`, per
    /// attempt, oldest first — the oracle that the NEXT attempt re-lists.
    pub fn assembly_listings_for(job_id: &str) -> Vec<(u32, Vec<(u32, String)>)> {
        listings()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|(id, _, _)| id == job_id)
            .map(|(_, attempt, assignment)| (*attempt, assignment.clone()))
            .collect()
    }

    /// One rank's target as built by `run_fine_tune_blocking`, before any
    /// step: the dropout seed it was given, each head layer's own dropout
    /// Philox seed (`LoraLinear::dropout_run_seed`; empty for an
    /// encoder-adapters target, whose per-site seeds are not enumerable
    /// through `AnyEncoder`), and a SHA-256 over the trainable weights in
    /// canonical name order.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct RankTarget {
        pub rank: u32,
        pub dropout_seed: u64,
        pub layer_dropout_seeds: Vec<Option<u64>>,
        pub weights_digest: String,
    }

    fn rank_targets() -> &'static Mutex<Vec<(String, RankTarget)>> {
        static PROBES: OnceLock<Mutex<Vec<(String, RankTarget)>>> = OnceLock::new();
        PROBES.get_or_init(|| Mutex::new(Vec::new()))
    }

    pub(super) fn note_rank_target(
        job_id: &str,
        rank: u32,
        dropout_seed: u64,
        target: &crate::fine_tune::target::TrainingTarget,
    ) -> jammi_db::error::Result<()> {
        use sha2::Digest;
        let layer_dropout_seeds = match target {
            crate::fine_tune::target::TrainingTarget::ProjectionHead { head } => head
                .layers
                .iter()
                .map(|(_, layer)| layer.dropout_run_seed())
                .collect(),
            crate::fine_tune::target::TrainingTarget::EncoderAdapters(_) => Vec::new(),
        };
        let weights = target.named_trainable_weights()?;
        let mut names: Vec<&String> = weights.keys().collect();
        names.sort();
        let mut hasher = sha2::Sha256::new();
        for name in names {
            hasher.update(name.as_bytes());
            hasher.update([0u8]);
            let values: Vec<f32> = weights[name]
                .flatten_all()
                .map_err(|e| jammi_db::error::JammiError::FineTune(e.to_string()))?
                .to_vec1()
                .map_err(|e| jammi_db::error::JammiError::FineTune(e.to_string()))?;
            for value in values {
                hasher.update(value.to_le_bytes());
            }
        }
        rank_targets()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((
                job_id.to_string(),
                RankTarget {
                    rank,
                    dropout_seed,
                    layer_dropout_seeds,
                    weights_digest: format!("{:x}", hasher.finalize()),
                },
            ));
        Ok(())
    }

    /// Every rank target recorded for `job_id`, in build order.
    pub fn rank_targets_for(job_id: &str) -> Vec<RankTarget> {
        rank_targets()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|(id, _)| id == job_id)
            .map(|(_, target)| target.clone())
            .collect()
    }

    /// One recorded lease holder per hold registration: `(job, attempt,
    /// holder)` — what `register_job_hold_or_release` was told this attempt
    /// runs as (the module doc's writer table, row 1).
    fn lease_holders() -> &'static Mutex<Vec<(String, u32, super::LeaseHolder)>> {
        static PROBES: OnceLock<Mutex<Vec<(String, u32, super::LeaseHolder)>>> = OnceLock::new();
        PROBES.get_or_init(|| Mutex::new(Vec::new()))
    }

    pub(super) fn note_lease_holder(job_id: &str, attempt: u32, holder: super::LeaseHolder) {
        lease_holders()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((job_id.to_string(), attempt, holder));
    }

    /// Every lease holder recorded for `job_id`, per attempt, oldest first
    /// — the oracle that a `W == 1` job is the `LoopClaimer` on every
    /// attempt; a `Peer` gang's attempts are the `Coordinator`.
    pub fn lease_holders_for(job_id: &str) -> Vec<(u32, super::LeaseHolder)> {
        lease_holders()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|(id, _, _)| id == job_id)
            .map(|(_, attempt, holder)| (*attempt, *holder))
            .collect()
    }

    /// One recorded runner role per `run_fine_tune_blocking` entry, keyed by
    /// job — every rank of a gang records its own (rank 0's holder, each
    /// other rank's `Rank { rank }`), in entry order.
    fn runner_roles() -> &'static Mutex<Vec<(String, super::RunnerRole)>> {
        static PROBES: OnceLock<Mutex<Vec<(String, super::RunnerRole)>>> = OnceLock::new();
        PROBES.get_or_init(|| Mutex::new(Vec::new()))
    }

    pub(super) fn note_runner_role(job_id: &str, role: super::RunnerRole) {
        runner_roles()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((job_id.to_string(), role));
    }

    /// Every runner role recorded for `job_id`, in entry order.
    pub fn runner_roles_for(job_id: &str) -> Vec<super::RunnerRole> {
        runner_roles()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|(id, _)| id == job_id)
            .map(|(_, role)| *role)
            .collect()
    }

    /// One recorded member end per `reconcile_member_ends` reading, keyed by
    /// the output MODEL id (`jammi:fine-tuned:{job_id}` — what the artifact
    /// in hand names; the job id is not on it): `(rank, the end's Debug)`.
    fn member_ends() -> &'static Mutex<Vec<(String, u32, String)>> {
        static PROBES: OnceLock<Mutex<Vec<(String, u32, String)>>> = OnceLock::new();
        PROBES.get_or_init(|| Mutex::new(Vec::new()))
    }

    pub(super) fn note_member_end(
        model_id: &str,
        rank: u32,
        end: &crate::fine_tune::collective::MemberEnd,
    ) {
        member_ends()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((model_id.to_string(), rank, format!("{end:?}")));
    }

    /// Every member end the coordinator read for `job_id`'s output model
    /// (`fine_tuned_model_id(job_id)`), in reading order: `(rank, Debug)`.
    pub fn member_ends_for(job_id: &str) -> Vec<(u32, String)> {
        let model_id = crate::fine_tune::training_job::fine_tuned_model_id(job_id);
        member_ends()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|(id, _, _)| *id == model_id)
            .map(|(_, rank, end)| (*rank, end.clone()))
            .collect()
    }

    /// A chaos wrapper for the NEXT member body of `job_id`: applied to the
    /// body's `Peer` right after it is built, before the first collective,
    /// so an oracle's own `Collective` (the chaos rows' `ChaosRank`) sits
    /// between the trainer and the wire — the fault injection at a chosen
    /// step of the REAL rank body. One-shot: taken by the first body of that
    /// job to reach the seam.
    pub type MemberCollectiveWrap = Box<
        dyn FnOnce(
                Arc<dyn crate::fine_tune::collective::Collective>,
            ) -> Arc<dyn crate::fine_tune::collective::Collective>
            + Send,
    >;

    fn member_wraps() -> &'static Mutex<Vec<(String, MemberCollectiveWrap)>> {
        static SLOTS: OnceLock<Mutex<Vec<(String, MemberCollectiveWrap)>>> = OnceLock::new();
        SLOTS.get_or_init(|| Mutex::new(Vec::new()))
    }

    pub fn wrap_member_collective(job_id: &str, wrap: MemberCollectiveWrap) {
        member_wraps()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((job_id.to_string(), wrap));
    }

    pub(super) fn take_member_collective_wrap(job_id: &str) -> Option<MemberCollectiveWrap> {
        let mut slots = member_wraps()
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let index = slots.iter().position(|(id, _)| id == job_id)?;
        Some(slots.remove(index).1)
    }

    /// One recorded rank-body end per `run_member_rank` return, keyed by
    /// job: `(rank, the outcome's Debug)`, in end order — how an oracle
    /// observes a member body's end whether or not the session lived to
    /// emit it.
    fn rank_outcomes() -> &'static Mutex<Vec<(String, u32, String)>> {
        static PROBES: OnceLock<Mutex<Vec<(String, u32, String)>>> = OnceLock::new();
        PROBES.get_or_init(|| Mutex::new(Vec::new()))
    }

    pub(super) fn note_rank_outcome(job_id: &str, rank: u32, outcome: &super::RankOutcome) {
        rank_outcomes()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((job_id.to_string(), rank, format!("{outcome:?}")));
    }

    /// Every rank-body end recorded for `job_id`, in end order.
    pub fn rank_outcomes_for(job_id: &str) -> Vec<(u32, String)> {
        rank_outcomes()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|(id, _, _)| id == job_id)
            .map(|(_, rank, end)| (*rank, end.clone()))
            .collect()
    }

    /// Arm a member-side failure for `job_id`: the NEXT rank body that
    /// completes its run for that job reports `Outcome{Failed{reason}}`
    /// instead of `Trained` — the fault injection that lets an oracle drive
    /// the coordinator's terminal-write-on-receipt through a member's
    /// failure without a real defect. Taken (disarmed) by the body.
    pub fn fail_member_outcome(job_id: &str, reason: &str) {
        member_failures()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((job_id.to_string(), reason.to_string()));
    }

    fn member_failures() -> &'static Mutex<Vec<(String, String)>> {
        static SLOTS: OnceLock<Mutex<Vec<(String, String)>>> = OnceLock::new();
        SLOTS.get_or_init(|| Mutex::new(Vec::new()))
    }

    pub(super) fn take_member_failure(job_id: &str) -> Option<String> {
        let mut slots = member_failures()
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let index = slots.iter().position(|(id, _)| id == job_id)?;
        Some(slots.remove(index).1)
    }

    /// One recorded coordinator end per attempt: `(attempt, the end's
    /// `Display`, its `ordinal`)`.
    type End = (String, u32, String, usize);

    fn ends() -> &'static Mutex<Vec<End>> {
        static PROBES: OnceLock<Mutex<Vec<End>>> = OnceLock::new();
        PROBES.get_or_init(|| Mutex::new(Vec::new()))
    }

    pub(super) fn note_coordinator_end(job_id: &str, attempt: u32, end: &super::CoordinatorEnd) {
        ends().lock().unwrap_or_else(PoisonError::into_inner).push((
            job_id.to_string(),
            attempt,
            end.to_string(),
            end.ordinal(),
        ));
    }

    /// Every coordinator end recorded for `job_id`, per attempt, oldest
    /// first: `(attempt, description, ordinal)`.
    pub fn coordinator_ends_for(job_id: &str) -> Vec<(u32, String, usize)> {
        ends()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|(id, _, _, _)| id == job_id)
            .map(|(_, attempt, end, ordinal)| (*attempt, end.clone(), *ordinal))
            .collect()
    }

    fn placed_submissions() -> &'static Mutex<Vec<(String, u32)>> {
        static SLOTS: OnceLock<Mutex<Vec<(String, u32)>>> = OnceLock::new();
        SLOTS.get_or_init(|| Mutex::new(Vec::new()))
    }

    /// Recorded by [`super::JobWorker::submit_placed`] the instant a claim
    /// takes the `Placed` arm — the oracle that a claim with a submitter
    /// installed took `Placed`, not [`super::JobWorker::coordinate`].
    pub(super) fn note_placed(job_id: &str, attempt: u32) {
        placed_submissions()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((job_id.to_string(), attempt));
    }

    /// Every attempt of `job_id` that took the `Placed` arm, oldest first.
    pub fn placed_attempts_for(job_id: &str) -> Vec<u32> {
        placed_submissions()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|(id, _)| id == job_id)
            .map(|(_, attempt)| *attempt)
            .collect()
    }

    fn placed_submit_ends() -> &'static Mutex<Vec<(String, bool)>> {
        static SLOTS: OnceLock<Mutex<Vec<(String, bool)>>> = OnceLock::new();
        SLOTS.get_or_init(|| Mutex::new(Vec::new()))
    }

    /// Recorded by [`super::JobWorker::placed_submit_end`]: whether the row
    /// was STILL this instance's (`Abandoned`) or had already moved
    /// (`HandedOff`) at the re-read — the oracle that the two arms are
    /// distinguished by the row's OWN `claimed_by`, never guessed.
    pub(super) fn note_placed_submit_end(job_id: &str, still_mine: bool) {
        placed_submit_ends()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((job_id.to_string(), still_mine));
    }

    /// Every `placed_submit_end` classification recorded for `job_id`, oldest
    /// first: `true` = `Abandoned` (still this instance's claim), `false` =
    /// `HandedOff` (the row had already moved).
    pub fn placed_submit_ends_for(job_id: &str) -> Vec<bool> {
        placed_submit_ends()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|(id, _)| id == job_id)
            .map(|(_, still_mine)| *still_mine)
            .collect()
    }

    /// One armed pause, keyed by the job whose `train_fine_tune` call it is
    /// for.
    struct ArmedPause {
        job_id: String,
        tx: oneshot::Sender<()>,
    }

    /// Every pause currently armed, one entry per job. Keyed by job — never
    /// a process-global "the next caller" slot — for the same reason every
    /// other park in this module is keyed (`ParkPoint` by job, the reclaim
    /// and rendezvous parks by instance): the test binary runs its tests
    /// concurrently in ONE process, so an unkeyed one-shot is taken by
    /// whichever fine-tune run happens to reach the checkpoint first — a
    /// sibling test's run as readily as the arming test's own — and a
    /// sibling that takes it parks forever before its trainer exists, so
    /// its epoch-boundary `cancel` read never happens and every bound that
    /// test placed on observing a cancel or a lease loss elapses.
    fn armed_pauses() -> &'static Mutex<Vec<ArmedPause>> {
        static ARMED: OnceLock<Mutex<Vec<ArmedPause>>> = OnceLock::new();
        ARMED.get_or_init(|| Mutex::new(Vec::new()))
    }

    /// Arm a one-shot pause just before `job_id`'s `train_fine_tune` call
    /// dispatches its training loop to `spawn_blocking`. The returned
    /// receiver resolves once THAT job's run has actually reached the
    /// checkpoint — with the job's lease hold and cancel-request watcher
    /// already constructed and live, and no training thread spawned yet —
    /// so a test can force `run_claimed_job`'s future to be dropped (or its
    /// owning task aborted, reproducing `EmbeddedWorker::drop`'s exact
    /// action) right there, with no `spawn_blocking` training thread in the
    /// picture to hold its own independent `Arc<Catalog>` clone and
    /// confound the release check this hook exists for. Any other job's run
    /// passes the checkpoint untouched (see [`armed_pauses`]).
    pub fn arm_pause_before_spawn_blocking(job_id: &str) -> oneshot::Receiver<()> {
        let (tx, rx) = oneshot::channel();
        armed_pauses()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(ArmedPause {
                job_id: job_id.to_string(),
                tx,
            });
        rx
    }

    /// Called from inside `train_fine_tune`, immediately before it
    /// dispatches `job_id`'s training loop to `spawn_blocking`. A no-op
    /// unless a pause is armed for THIS job; when one is, this takes it
    /// (disarming it), signals arrival on the armed receiver and then parks
    /// forever (never resolves on its own) — the test that armed the pause
    /// is expected to abort the task holding this `.await` (or otherwise
    /// drop the future) rather than release it, so nothing here needs a
    /// resume path.
    pub(crate) async fn checkpoint_before_spawn_blocking(job_id: &str) {
        let armed = {
            let mut list = armed_pauses()
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            list.iter()
                .position(|a| a.job_id == job_id)
                .map(|i| list.remove(i))
        };
        if let Some(ArmedPause { tx, .. }) = armed {
            let _ = tx.send(());
            std::future::pending::<()>().await;
        }
    }

    /// One recorded watcher, keyed by the job it was spawned for. A `Vec`
    /// rather than a `HashMap` because a reclaimed/retried job can spawn a
    /// new watcher under the SAME `job_id` at a higher attempt — callers
    /// look up the most recently recorded entry.
    struct WatcherProbe {
        job_id: String,
        watcher: tokio::task::AbortHandle,
        catalog: Weak<Catalog>,
    }

    fn probes() -> &'static Mutex<Vec<WatcherProbe>> {
        static PROBES: OnceLock<Mutex<Vec<WatcherProbe>>> = OnceLock::new();
        PROBES.get_or_init(|| Mutex::new(Vec::new()))
    }

    /// Record this attempt's cancel-request watcher and its per-attempt
    /// `Arc<Catalog>` — via a [`Weak`], so recording the probe never itself
    /// keeps the attempt's catalog handle alive, and the count
    /// [`catalog_strong_count`] reports is exactly the run's own remaining
    /// holders.
    pub(crate) fn record_watcher(
        job_id: &str,
        watcher: tokio::task::AbortHandle,
        catalog: &Arc<Catalog>,
    ) {
        probes()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(WatcherProbe {
                job_id: job_id.to_string(),
                watcher,
                catalog: Arc::downgrade(catalog),
            });
    }

    /// Whether the most recently recorded cancel-request watcher for
    /// `job_id` has finished — [`tokio::task::AbortHandle::is_finished`],
    /// the same primitive a `JoinHandle` exposes, reachable here because the
    /// watcher's actual `JoinHandle` is private to [`CancelWatcherGuard`],
    /// which consumes it. `None` if no watcher was ever recorded for this
    /// job id.
    pub fn watcher_is_finished(job_id: &str) -> Option<bool> {
        probes()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .rev()
            .find(|p| p.job_id == job_id)
            .map(|p| p.watcher.is_finished())
    }

    /// The live strong-reference count on the per-attempt `Arc<Catalog>`
    /// [`JobWorker::run_claimed_job`] built for `job_id`'s most recently
    /// recorded attempt. `None` if no watcher was ever recorded for this
    /// job id.
    pub fn catalog_strong_count(job_id: &str) -> Option<usize> {
        probes()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .rev()
            .find(|p| p.job_id == job_id)
            .map(|p| p.catalog.strong_count())
    }
}

#[cfg(all(test, feature = "test-hooks"))]
mod training_test_hooks_tests {
    use std::time::Duration;

    use tokio::sync::oneshot::error::TryRecvError;

    use super::training_test_hooks::{
        arm_pause_before_spawn_blocking, checkpoint_before_spawn_blocking,
    };

    /// For every armed pause and every job: the pause is taken by the
    /// checkpoint of exactly the job it was armed for — a sibling job's
    /// checkpoint passes through without signalling or parking, the armed
    /// job's checkpoint signals arrival and parks, and once taken the pause
    /// is disarmed for that job too.
    #[tokio::test]
    async fn an_armed_pause_is_taken_only_by_the_job_it_was_armed_for() {
        let mut parked = arm_pause_before_spawn_blocking("job-armed");

        tokio::time::timeout(
            Duration::from_secs(5),
            checkpoint_before_spawn_blocking("job-sibling"),
        )
        .await
        .expect("a sibling job's checkpoint must pass through a pause armed for another job");
        assert!(
            matches!(parked.try_recv(), Err(TryRecvError::Empty)),
            "the sibling's pass-through must not signal the armed receiver"
        );

        let own = tokio::spawn(checkpoint_before_spawn_blocking("job-armed"));
        tokio::time::timeout(Duration::from_secs(5), parked)
            .await
            .expect("the armed job's checkpoint must signal its arrival")
            .expect("the armed sender is only ever consumed by a send");
        assert!(
            !own.is_finished(),
            "the armed job's checkpoint parks after signalling"
        );
        own.abort();

        tokio::time::timeout(
            Duration::from_secs(5),
            checkpoint_before_spawn_blocking("job-armed"),
        )
        .await
        .expect("a taken pause is disarmed: the same job's next checkpoint passes through");
    }
}

/// Record a terminal `Failed` status for a job this worker owns, surfacing
/// the cause on `jobs.error` so a [`crate::fine_tune::training_job::TrainingJob::wait`]
/// (or [`crate::jobs::JobHandle::wait`]) observer sees the failure instead of
/// an indefinite `running` state.
///
/// The write is lease-guarded (the failure peer of the finalize CAS): it lands
/// only while this worker still holds the lease (`claimed_by = worker_id AND
/// status = 'running' AND attempts = attempt`). A worker that lost its lease
/// before failing does not stamp `failed` over a job the re-claiming worker is
/// running — that case is left for the new owner (logged at debug).
///
/// `holder` is the writer — REQUIRED at every call site (the module doc's
/// writer table): a job-row write is reachable only with the attempt's
/// `LeaseHolder` in hand, and a rank body holds none.
async fn record_failed(
    holder: LeaseHolder,
    catalog: &Arc<Catalog>,
    job_id: &str,
    worker_id: &str,
    attempt: u32,
    msg: String,
) {
    record_unsuccessful_end(
        holder,
        catalog,
        job_id,
        worker_id,
        attempt,
        UnsuccessfulEnd::Failed(msg),
    )
    .await;
}

/// [`record_failed`]'s general form: the lease-guarded terminal write for a
/// job that ended without its result, as `failed` or as `cancelled`. A call
/// site whose error can be a [`JammiError::JobCancelled`] passes
/// `UnsuccessfulEnd::from(&error)`, so the typed error decides the status.
async fn record_unsuccessful_end(
    holder: LeaseHolder,
    catalog: &Arc<Catalog>,
    job_id: &str,
    worker_id: &str,
    attempt: u32,
    end: UnsuccessfulEnd,
) {
    match end.record(catalog, job_id, worker_id, attempt).await {
        Ok(true) => {}
        Ok(false) => {
            tracing::debug!(
                job_id = %job_id,
                worker = %worker_id,
                %holder,
                "lost lease before recording failure; left for reclaim"
            );
        }
        Err(e) => {
            tracing::error!(job_id = %job_id, error = %e, "Failed to record terminal status");
        }
    }
}

/// Persists `report_json` via [`Catalog::record_acceleration_report`],
/// logging (never propagating) a lease-guard miss or a catalog error — the
/// shared write path every acceleration-report site uses, whether it
/// carries a measured `"determined"` payload
/// (`compute_and_persist_acceleration_report`) or one of the terminal
/// markers below for a job kind/path that never reaches the measuring probe
/// at all.
///
/// **Swallowing is deliberate, and it is covered downstream.** A `false`
/// (lease-guard miss: the lease was lost, or this attempt's `attempt` no
/// longer matches the row's `attempts`) or an `Err` here must never fail
/// training. The consequence — a job that goes on to complete with its
/// submission-time `{"state":"pending"}` marker never overwritten — is
/// retired at the catalog's terminal edge, not compensated for here; see
/// [`JobWorker::run_claimed_job`]'s "The SUCCESS path is not exempt"
/// section.
///
/// `holder` is the writer (the module doc's writer table, row 12): a job-row
/// write, reachable only with the attempt's `LeaseHolder`.
async fn persist_acceleration_report(
    holder: LeaseHolder,
    catalog: &Arc<Catalog>,
    job_id: &str,
    worker_id: &str,
    attempt: u32,
    report_json: &str,
) {
    match catalog
        .record_acceleration_report(job_id, worker_id, attempt, report_json)
        .await
    {
        Ok(true) => {}
        Ok(false) => {
            tracing::warn!(
                job_id = %job_id,
                worker_id = %worker_id,
                attempt,
                %holder,
                "record_acceleration_report's lease guard did not match (lease lost or \
                 stale attempt); continuing without a persisted acceleration report"
            );
        }
        Err(e) => {
            tracing::warn!(
                job_id = %job_id,
                worker_id = %worker_id,
                attempt,
                error = %e,
                "record_acceleration_report failed; continuing without a persisted \
                 acceleration report"
            );
        }
    }
}

/// `TrainingSpec::ContextPredictor` jobs never route through
/// `run_fine_tune_blocking`'s measuring probe at all (`run_spec`'s
/// `ContextPredictor` arm calls `InferenceSession::run_context_predictor_training`
/// directly, never `train_fine_tune`) — without this, the record's
/// `acceleration_report` would carry the submission-time `{"state":
/// "pending"}` marker FOREVER, even after the job reaches a terminal
/// `completed`/`failed` status, which the tri-state report's own
/// definition of "pending" (submitted, no claimant has computed a
/// determination YET) does not describe. Writes the self-describing
/// `{"state":"not_applicable","reason":"context_predictor"}` marker before
/// training starts, under the SAME lease-guarded
/// `record_acceleration_report` every other acceleration-report write uses.
async fn mark_acceleration_not_applicable(
    holder: LeaseHolder,
    catalog: &Arc<Catalog>,
    job_id: &str,
    worker_id: &str,
    attempt: u32,
    reason: &str,
) {
    persist_acceleration_report(
        holder,
        catalog,
        job_id,
        worker_id,
        attempt,
        &serde_json::json!({"state": "not_applicable", "reason": reason}).to_string(),
    )
    .await;
}

/// A job that fails in `run_claimed_job` BEFORE the device is ever resolved (no
/// `training_spec` at all, or an undeserialisable one — both happen before
/// `run_spec`/`run_fine_tune_blocking` are ever reached) never runs the
/// measuring probe either, and would otherwise carry the submission-time
/// `{"state":"pending"}` marker forever past this job's terminal `failed`
/// status. Writes the self-describing
/// `{"state":"undetermined","reason":"failed_before_device_resolution"}`
/// marker, under the SAME lease-guarded `record_acceleration_report` every
/// other acceleration-report write uses — called BEFORE `record_failed` so the write
/// still observes this attempt's `running` status (the lease guard requires
/// it; `record_failed` flips the row to `failed` immediately after).
///
/// This pre-mark is REDUNDANT-but-preferred rather than load-bearing:
/// `Catalog::fail_job` retires ANY still-`pending` report to
/// `{"state":"undetermined","reason":"failed_before_probe"}` at the catalog
/// edge, so this path is covered even without the pre-mark. It is kept
/// because its reason is strictly MORE specific (this job never even resolved
/// a device, a fact the catalog edge cannot know), and the edge's rewrite is
/// strictly `pending`-valued so it preserves this payload byte-for-byte.
///
/// **Never byte-compare a multi-key marker.** This function builds its JSON
/// with `serde_json::json!`, whose default object is a `BTreeMap`, so the
/// keys serialize ALPHABETICALLY (`{"reason":…,"state":…}`); jammi-db's own
/// terminal markers are literal strings in declaration order
/// (`{"state":…,"reason":…}`). The two producers therefore differ in byte
/// order while carrying identical JSON. Every consumer parses (the Python
/// binding, `expect_determined_report`, the catalog edge's own strictly-
/// `pending`-valued `CASE`), so this is inert — but any equality check
/// against a marker with more than one key must compare parsed values, not
/// bytes. The single-key `{"state":"pending"}` marker is the one exception,
/// and it has exactly ONE producer (jammi-db's own `INSERT` const), which is
/// what makes the catalog edge's byte match on it sound.
async fn mark_acceleration_undetermined(
    holder: LeaseHolder,
    catalog: &Arc<Catalog>,
    job_id: &str,
    worker_id: &str,
    attempt: u32,
) {
    persist_acceleration_report(
        holder,
        catalog,
        job_id,
        worker_id,
        attempt,
        &serde_json::json!({
            "state": "undetermined",
            "reason": "failed_before_device_resolution"
        })
        .to_string(),
    )
    .await;
}

/// The inputs to one blocking LoRA fine-tune run, grouped so the blocking call
/// takes a single owned argument rather than a long positional list. Built on
/// the async side and moved into the `spawn_blocking` closure.
struct RunFineTuneParams {
    catalog: Arc<Catalog>,
    artifact_store: Arc<ArtifactStore>,
    artifact_dir: std::path::PathBuf,
    job_id: String,
    worker_id: String,
    /// This claim's attempt counter (`record.attempts`) — the third segment of
    /// the attempt-unique publish prefix the trainer writes per-epoch
    /// checkpoints under (`{job_id}/{worker_id}/{attempt}/checkpoints/…`).
    attempt: u32,
    /// What this rank runs AS (`crate::fine_tune::role`): the lease holder
    /// (rank 0 — the single rank, a `Local` gang's rank 0, a `Peer` gang's
    /// coordinator) persists the acceleration report and, through the
    /// trainer's own gate, the checkpoints; a `Rank` computes the same
    /// probe (the forward/backward runs identically on every rank, so no
    /// rank's state drifts from its peers') and persists nothing.
    role: RunnerRole,
    /// This rank's [`RankContext`], or `None` for the single-rank run (the
    /// builder's own `RankContext::single_rank` default).
    rank_ctx: Option<RankContext>,
    base_model: String,
    task: ModelTask,
    config: FineTuneConfig,
    /// What the training loop trains from — see
    /// [`crate::fine_tune::source::TrainingSource`]'s own doc.
    source: crate::fine_tune::source::TrainingSource,
    base_model_arc: Arc<crate::model::LoadedModel>,
    hidden_size: usize,
    device_config: DeviceConfig,
    cancel: Arc<AtomicBool>,
    /// The session's one shared [`crate::model::hub::HubSource`] —
    /// `build_encoder_adapters`'s HF-fallback arm threads this through
    /// rather than building its own `hf_hub::api::sync::Api`.
    hub: HubSource,
}

/// Run LoRA fine-tuning in a blocking context, checking `cancel` at every epoch
/// boundary. Reconstructs the training target (projection head or encoder
/// adapters) and drives the trainer. The loop trains and persists the adapter
/// but writes no terminal status — the worker registers the output model and
/// runs the lease-guarded finalize after this returns. Returns the
/// [`crate::fine_tune::trainer::TrainingResult`] (adapter path + run metrics)
/// the worker threads into that finalization. `call` is the collective's
/// witness minted by the `spawn_blocking` boundary that entered this
/// function; `TrainingLoop::run` takes it, and through it every collective
/// verb the run makes.
fn run_fine_tune_blocking(
    call: &crate::fine_tune::collective::BlockingCall,
    params: RunFineTuneParams,
) -> Result<crate::fine_tune::trainer::TrainingResult> {
    use candle_core::DType;
    use candle_nn::VarMap;

    let RunFineTuneParams {
        catalog,
        artifact_store,
        artifact_dir,
        job_id,
        worker_id,
        attempt,
        role,
        rank_ctx,
        base_model,
        task,
        config,
        source: training_source,
        base_model_arc,
        hidden_size,
        device_config,
        cancel,
        hub,
    } = params;
    #[cfg(feature = "test-hooks")]
    training_test_hooks::note_runner_role(&job_id, role);
    // Each rank's dropout seed derives as `f(seed, rank)` —
    // `RankContext::dropout_seed` (rank 0 is the identity, so the single-rank
    // run and every rank 0 keep `config.seed` exactly). The A/B INIT seed is
    // `config.seed` on every rank (the `_for_rank` builders and
    // `LoraBuildConfig::seed`), so a gang's ranks start from byte-identical
    // adapter weights and draw distinct masks.
    let dropout_seed = rank_ctx
        .as_ref()
        .map_or(config.seed, |ctx| ctx.dropout_seed(config.seed));

    let device = crate::model::backend::candle::select_device(&device_config)?;
    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let target = if config.target_modules.is_empty() {
        let head = if task == ModelTask::Classification {
            // `num_classes` from the SAME source: a `Resident` loader's
            // `TrainingFormat::Classification { num_classes }` (built from
            // the eager `BTreeSet` pass), or a `Streamed` source's
            // `StreamedSet::num_classes()` (built from the worker's own
            // whole-table vocabulary sweep) — both are the identical
            // sorted-label-set enumeration (`decode::LabelVocabulary`'s own
            // doc), so the head is sized identically either way.
            let num_classes = match &training_source {
                crate::fine_tune::source::TrainingSource::Resident(loader) => {
                    match loader.format() {
                        crate::fine_tune::data::TrainingFormat::Classification { num_classes } => {
                            num_classes
                        }
                        _ => {
                            return Err(JammiError::FineTune(
                                "Classification task requires classification training data \
                                 format"
                                    .into(),
                            ))
                        }
                    }
                }
                crate::fine_tune::source::TrainingSource::Streamed(streamed) => {
                    streamed.num_classes().ok_or_else(|| {
                        JammiError::FineTune(
                            "Classification task requires classification training data format"
                                .into(),
                        )
                    })?
                }
            };
            crate::fine_tune::lora::build_classification_head_for_rank(
                hidden_size,
                num_classes,
                &config,
                &varmap,
                &vb,
                dropout_seed,
            )?
        } else if task == ModelTask::Regression {
            let output_dim = match config.regression_loss.unwrap_or_default() {
                crate::fine_tune::RegressionLoss::Pinball => config.quantile_levels.len(),
                _ => 2,
            };
            crate::fine_tune::lora::build_distribution_head_for_rank(
                hidden_size,
                output_dim,
                &config,
                &varmap,
                &vb,
                dropout_seed,
            )?
        } else {
            crate::fine_tune::lora::build_projection_head_for_rank(
                hidden_size,
                &config,
                &varmap,
                &vb,
                dropout_seed,
            )?
        };
        // `backbone_dtype` never takes effect on this arm (see
        // `validate_backbone_precision`'s doc), so there is no encoder to probe
        // fused-op admission against — the report still names the resolved
        // device + compiled capabilities, with an empty `ops`/honest "no probe
        // attempted" `flash` reason (never a fabricated per-op measurement for
        // an arm that never ran one — see `flash_report_no_probe_attempted`'s
        // doc).
        compute_and_persist_acceleration_report(
            &catalog,
            &job_id,
            &worker_id,
            attempt,
            role,
            &device,
            config.backbone_dtype,
            None,
            None,
        );
        crate::fine_tune::target::TrainingTarget::ProjectionHead { head }
    } else {
        let (mut encoder, adapter_cfg) = build_encoder_adapters(BuildEncoderAdaptersParams {
            base_model_id: &base_model,
            catalog: &catalog,
            artifact_store: &artifact_store,
            config: &config,
            task,
            varmap: &varmap,
            device: &device,
            hub: &hub,
            dropout_seed,
        })?;
        // Right after `build_encoder_adapters` (which calls
        // `validate_backbone_precision` and materialises the real, dtype-typed
        // encoder) and BEFORE the training loop's first step — the earliest
        // point a per-job admission determination is both possible (the real
        // encoder exists) and cheap (nothing has trained yet). `&varmap` is the
        // SAME `VarMap` `build_encoder_adapters` registered the LoRA A/B
        // trainable vars into — the probe's backward+optimizer step
        // (`run_backward_and_optimizer_probe`) snapshots and restores it.
        compute_and_persist_acceleration_report(
            &catalog,
            &job_id,
            &worker_id,
            attempt,
            role,
            &device,
            config.backbone_dtype,
            Some(&varmap),
            Some(&mut encoder),
        );
        crate::fine_tune::target::TrainingTarget::EncoderAdapters(Box::new(
            crate::fine_tune::target::EncoderAdaptersTarget {
                encoder,
                adapter_cfg,
            },
        ))
    };

    // Discover the job's newest complete epoch checkpoint. If one exists (a
    // prior attempt completed at least one epoch boundary before dying), the
    // trainer restores weights + optimizer moments + scaler + dropout
    // positions and continues from `last_completed + 1`; if none exists, it
    // trains from scratch. The discovery never perturbs the publish/serving
    // path — the checkpoint prefixes (`{job_id}/_checkpoints/`) are the job's
    // own, beside the attempts' served prefixes.
    // `catalog` is `pinned_to_tenant(record.tenant_id)` (the caller's
    // tenant-scoped catalog) — its rows name every epoch a prior attempt
    // staged, and its `current_tenant()` names the job's own tenant
    // regardless of task-local scope, so every checkpoint the trainer writes
    // lands under the SAME tenant segment the read resolves.
    // The seed-split oracle's observation point: this rank's target as
    // BUILT — the dropout seed it was given, each head layer's own dropout
    // Philox seed, and a digest of the trainable weights before any step —
    // so a gang oracle can assert distinct dropout seeds over byte-identical
    // initial weights through the real `run_spec`.
    #[cfg(feature = "test-hooks")]
    training_test_hooks::note_rank_target(&job_id, role.rank(), dropout_seed, &target)?;

    // Fired UNCONDITIONALLY, before the call: the rank-attributed proof
    // that THIS rank's body reached the resume seam, regardless of what
    // `discover_resume` goes on to return (`Ok(None)`, `Ok(Some(_))`, or an
    // `Err` this `?` propagates on a corrupted bundle) — see
    // `Event::ResumeAttempted`'s own doc.
    #[cfg(feature = "test-hooks")]
    loop_test_hooks::fire_observed(
        &job_id,
        loop_test_hooks::Event::ResumeAttempted(role.rank()),
    );

    let resume = discover_resume(&artifact_store, &catalog, &job_id, &device)?;

    let mut builder = crate::fine_tune::trainer::TrainingLoopBuilder::new(target, varmap, config)
        .base_model(base_model_arc)
        // The run's declared task is the trainer's MODALITY discriminator for
        // the media paths (`TrainingLoop::task`) — the same `task` this
        // function already gated the data loader and the encoder dispatch on.
        .task(task)
        .job_id(job_id)
        .attempt(attempt)
        // The job's tenant-pinned catalog: its bound tenant owns every
        // checkpoint the loop stages.
        .catalog(Arc::clone(&catalog))
        .artifact_dir(artifact_dir)
        .device(device.clone())
        .cancel(cancel)
        .artifact_store(Arc::clone(&artifact_store));
    // A gang rank's own context (`worker.rs`'s topology fan-out); a
    // single-rank run leaves the builder's `RankContext::single_rank`
    // default in place — byte-identical to every pre-gang run.
    if let Some(rank_ctx) = rank_ctx {
        builder = builder.rank_context(rank_ctx);
    }
    // The role, stated explicitly on every production rank: the trainer's
    // own durable-write gate (and its agreement with the rank context is
    // checked at `build`).
    builder = builder.runner_role(role);
    if let Some(restored) = resume {
        builder = builder.resume(restored);
    }
    let mut training_loop = builder.build()?;

    training_loop.run(call, training_source)
}

/// Fetch and load the checkpoint a resume of the job restores, if any: the
/// newest epoch whose manifest exists and verifies
/// ([`ArtifactStore::fetch_newest_checkpoint`] — an epoch whose write never
/// reached its manifest is skipped for the one before it). `None` only when
/// no checkpoint exists yet (from-scratch). A checkpoint that exists but
/// cannot be restored — a bundle of another schema version
/// (`JammiError::IncompatibleFormat`), a manifest whose digests do not
/// verify, torn moments — is the attempt's failure, never a silent
/// from-scratch restart.
fn discover_resume(
    store: &Arc<ArtifactStore>,
    catalog: &Catalog,
    job_id: &str,
    device: &candle_core::Device,
) -> Result<Option<crate::fine_tune::resume::RestoredCheckpoint>> {
    tokio::runtime::Handle::current()
        .block_on(store.fetch_newest_checkpoint(catalog, job_id))?
        .map(|local| crate::fine_tune::resume::load_bundle(local.dir(), device))
        .transpose()
}

/// Refuse a backbone precision the resolved device cannot compute at.
///
/// BF16 is a GPU-tier precision. candle's CPU matmul accepts `F16 | F32 | F64`
/// only and returns "unsupported dtype BF16 for op matmul" otherwise, so a BF16
/// backbone on CPU fails at the first frozen linear of the first forward —
/// after the job has been claimed, the backbone downloaded, and the adapter
/// built.
///
/// The LoRA arm keeps the frozen weight at `backbone_dtype` rather than
/// re-materialising it in F32 on every forward; an F32 upcast would mask the
/// limitation (with every ModernBERT linear LoRA-targeted, no `Frozen` arm
/// survives to hit the unsupported matmul, so a CPU BF16 fine-tune would
/// "work" only by silently discarding the precision it was asked for).
/// Honouring `backbone_dtype` means the unsupported combination is refused
/// rather than quietly ignored.
///
/// The inference path makes the same refusal at
/// `crate::model::backend::candle` when it resolves a device; this is its
/// training-side peer. It cannot move up into `FineTuneConfig::validate`,
/// which sees the config but not the device the claiming worker will resolve.
fn validate_backbone_precision(
    precision: jammi_numerics::ComputePrecision,
    device: &candle_core::Device,
) -> Result<()> {
    if precision == jammi_numerics::ComputePrecision::BF16 && !device.is_cuda() {
        return Err(JammiError::FineTune(
            "backbone_dtype=bf16 requires a CUDA device; this worker resolved a non-CUDA \
             device, whose matmul does not implement bf16. Use f16 for a reduced-precision \
             backbone on CPU, or f32."
                .into(),
        ));
    }
    Ok(())
}

// =============================================================================
// Claim-time, per-job acceleration report.
//
// A compute precision the public API accepts (`ComputePrecision`) is either
// accelerated by the fused kernels or it silently runs the eager composition.
// The fallback `tracing::warn` is deduplicated for the life of the PROCESS
// (`jammi_kernels::admission::warn_fallback_once`), so on its own a second
// f16 job would read the same silence as a first, accelerated one. This
// section computes a compact, per-JOB determination — computed from the SAME
// admission predicates the kernels use, never a parallel re-derivation of
// them — and persists it on the job's catalog record
// (`Catalog::record_acceleration_report`) before the training loop's first
// step, so a status poll mid-training always finds a `"determined"` report.
//
// **How "the same predicates" is honoured without a private admission API**:
// every per-op fused-kernel domain check in `jammi-encoders` is a *private*
// function requiring real tensors (e.g. `layer_norm.rs`'s
// `fused_admission_predicate(x, weight)`), so this module cannot call it
// directly. Instead it runs the SAME real, PUBLIC forward path
// (`AnyEncoder::forward`) the training loop itself is about to run, over a
// minimal synthetic batch (token id `0` — valid for any non-empty
// vocabulary), and reads the outcome back through the SAME public,
// process-wide dispatch registries `jammi-encoders`/`jammi-kernels` already
// expose for exactly this "durable job record" consumer (see
// `jammi_encoders::layer_norm::LN_DISPATCH_COUNTERS`'s own doc: "a durable
// job record ... uses" `jammi_encoders::ln_dispatch_snapshot`). This never
// re-derives the dtype/shape/device domain check: it exercises the real one
// and observes its real effect. A miss's `reason` is read back out of THIS
// probe's own `jammi_kernels::admission::probe_capture_begin()` window, by
// the SAME `op` key the kernel's own `admit()` call site already uses —
// reused verbatim, never invented here. That window is armed for exactly the
// forward+backward+step below and records every miss on this thread,
// independent of the log-once `(op, predicate)` dedupe
// `fallback_warnings_emitted()` applies for LOGGING. Reading the deduped
// warn list instead would attribute the most recent DIFFERENT predicate to a
// job whose own miss repeated an already-burned pair — see
// `reason_from_probe_window`'s doc. A `holds: false` op with no entry in its
// own window gets the honest `"reason_unavailable"`, never a guess.
//
// The probe runs a real forward pass PLUS one backward + optimizer step:
// `layer_norm`, `rope`, `softmax`, `geglu`, and `attention_block` dispatch
// during the forward pass; `dropout`/`low_rank_residual_linear` (both read
// from the SAME `lora_linear_fused` registry key — the separate
// `lora_dropout` counter never moves) ALSO dispatch during the forward pass
// (`LoraLinear::forward`); and `LowRankResidualLinear`'s own backward-time
// cast-boundary epilogue kernels (`cast_scale`/`cast_add`, `crates/jammi-kernels/src/ops/
// low_rank_residual_linear.rs`'s `bwd`) dispatch ONLY during backward, so
// a forward-only probe could never honestly claim to have measured them.
//
// **Which ops this probe can attribute is ONE static fact**, not a list
// re-typed here: `jammi_kernels::admission::PROBED_OPS`. `cast_scale` and
// `cast_add` admit under dtype-resolved registry keys
// (`cast_scale_bf16_f32`/`cast_scale_f16_f32`, `cast_add_bf16`/`cast_add_f16`,
// in `jammi_kernels::ops::low_rank_residual_linear`'s backward), so the
// report's key must be resolved per dtype class or an f16 job could not name
// its own cast epilogue. `rope_positions` and `scaled_cast_add` have no
// admission gate: each is a bare launcher call from inside an
// already-admitted parent's fused arm (`ProbedOpKind::InternalSubkernel`), so
// no probe can read a delta for them and they are OMITTED from `ops` — never
// fabricated as a `holds` either way (the honest negative, not a vacuous
// positive). Every kernel this build compiles is in one of those two
// positions; no compiled kernel has neither an admission gate nor an
// admitted parent (a measured CUDA census:
// `crates/jammi-kernels/artifacts/cuda-runs/2026-09-01-axpy-census-bdeb80c-a100-pcie.json`),
// so there is no class of compiled kernel this report is structurally unable
// to say anything about.
//
// The report's `ops` CANDIDATE key set is therefore a pure function of the
// job's backbone dtype class (`PROBED_OPS` filtered by
// `ProbedOp::registry_keys_for`), never a function of what else this process
// happened to run — which is what `jammi_kernels::admission::snapshot_all()`
// would give (it reflects only ops looked up at least once, so its key
// set varies with process history; see its own doc).
//
// The backward+optimizer step runs on the REAL
// production trainable weights (there is no separate throwaway model to
// probe instead — see `run_backward_and_optimizer_probe`'s doc), so it
// snapshots every trainable var with a genuine deep copy and restores it
// afterward: the training run that follows sees byte-identical initial
// weights regardless of whether this probe ran, mutation aside (its ONE
// dropout-mask RNG draw is NOT undone — see that doc's own disclosure).
// `flash` degrades to a compiled/device-level fact (`cuda_not_compiled` /
// `flash_not_compiled` / `device_is_cpu_or_metal_not_cuda` /
// `no_encoder_to_probe_projection_head_arm`) whenever the cascade admission
// path cannot even be reached, or no encoder was built to probe at all
// (distinct from a probe that WAS attempted and failed); a
// reached-but-declined cascade reads its verbatim predicate key back out of
// THIS probe's own probe-capture window — see the next paragraph — falling
// back to the coarser, honestly-labelled `"capability_or_domain_miss"` only
// when that window carries no entry for the decline.
//
// **The BERT/DistilBERT case:**
// both families' training forward always calls
// `attention_cascade::training_attention_cascade` with `flash: &FlashDecision::
// Declined { outcome: CapabilityMiss, reason: "flash_transport_not_wired" }`
// — the ONE reason value either family's `FlashDecision::Declined` ever
// carries, because neither wires the encoder-boundary flash transport
// protocol (see the `FlashDecision::Declined` construction in
// `jammi_encoders::bert` and `jammi_encoders::distilbert`).
// `jammi_kernels::admission::admit_cascade` records every decline —
// disabled, `DomainMiss`, and `CapabilityMiss` alike — into the SAME
// thread-local probe-capture sink `admit_inner` uses
// (`record_probe_miss(op, predicate_name)`), not just an atomic
// increment on `CascadeDispatchCounters`. [`flash_report`] reads that entry
// back through `jammi_kernels::admission::probe_capture_reason_for(window,
// "attention_block_flash")` on a decline, exactly the way
// [`reason_from_probe_window`] reads it for a two-arm op — so a BERT/
// DistilBERT job's `flash` field reads back verbatim as
// `"flash_transport_not_wired"` rather than the coarse
// `"capability_or_domain_miss"`. The coarse value survives only as
// [`flash_report`]'s fallback for a decline whose window happens to carry no
// entry (the window-attribution causes [`REASON_UNAVAILABLE`] already
// documents) — never fabricated in its place.
//
// **Single-worker-per-process attribution precondition**: the
// before/after dispatch-registry delta this probe reads is attributed to
// THIS job's own probe call, which is correct as long as no OTHER job's
// admission-gated dispatch races the SAME registry keys on another thread of
// the SAME process between this probe's two snapshots — true for the normal
// one-job-at-a-time-per-worker-instance shape (`JobWorker::run_until`'s
// claim→run→claim loop never overlaps two claims on one worker), but NOT
// guarded against a deployment running multiple `EmbeddedWorker`/
// `JobWorker` instances concurrently in the SAME process. A concurrent
// job's `fused`-only dispatch on the same op during this window would read
// as `holds: true` for THIS job too (`two_arm_holds` only collapses the
// ambiguous BOTH-moved case, not a fused-only race). Documented here as this
// report's attribution precondition rather than solved by a lock, since a
// snapshot-under-lock would need to serialize EVERY admission-gated call
// site workspace-wide to close it completely, not just the two reads this
// function makes.
// =============================================================================

/// This job's backbone dtype as the [`jammi_kernels::admission::DtypeClass`]
/// the probed-op table resolves registry keys against — the ONE place a
/// `ComputePrecision` becomes a dtype class for report purposes.
fn dtype_class_of(
    precision: jammi_numerics::ComputePrecision,
) -> jammi_kernels::admission::DtypeClass {
    match precision {
        jammi_numerics::ComputePrecision::F32 => jammi_kernels::admission::DtypeClass::F32,
        jammi_numerics::ComputePrecision::BF16 => jammi_kernels::admission::DtypeClass::Bf16,
        jammi_numerics::ComputePrecision::F16 => jammi_kernels::admission::DtypeClass::F16,
    }
}

/// The report keys this job's dtype class can produce a measurement for, in
/// [`jammi_kernels::admission::PROBED_OPS`] order — the report's CANDIDATE
/// `ops` key set.
///
/// A pure function of `dtype` alone: it never consults
/// `jammi_kernels::admission::snapshot_all()` (whose key set reflects only
/// ops looked up at least once in THIS process, so an identical job would get
/// a different report shape depending on what ran before it). Only [`jammi_kernels::admission::ProbedOpKind::TwoArm`] rows
/// appear: a cascade has no `fallback_warnings`-shaped reason channel (the
/// flash cascade gets the report's own dedicated top-level `flash` field
/// instead), and an `InternalSubkernel` row has no registry key for any probe
/// to read a delta from at all.
///
/// A candidate key is REALIZED into `ops` only if the probe actually moved
/// its counter one way and not the other ([`two_arm_holds`]) — an op the
/// probe never reached is omitted, never claimed as a miss.
fn probed_report_keys(
    dtype: jammi_kernels::admission::DtypeClass,
) -> Vec<(&'static str, &'static str)> {
    jammi_kernels::admission::PROBED_OPS
        .iter()
        .filter(|op| op.kind() == jammi_kernels::admission::ProbedOpKind::TwoArm)
        .filter_map(|op| {
            op.registry_keys_for(dtype)
                .next()
                .map(|key| (op.report_key(), key))
        })
        .collect()
}

/// A snapshot of every two-arm dispatch registry
/// [`jammi_kernels::admission::PROBED_OPS`] names for this job's dtype class,
/// plus the `attention_block_flash` cascade — taken once immediately before
/// and once immediately after the probe so a per-job report reads a DELTA
/// (attributable to this job's own probe call) rather than the
/// process-lifetime total (which every OTHER job sharing this process also
/// contributes to).
///
/// Keyed by REGISTRY key, not by report key: `"dropout"` and
/// `"low_rank_residual_linear"` are the same `lora_linear_fused` dispatch
/// decision, so storing one entry per registry key is what makes that a
/// structural fact rather than a match arm that has to remember it, and keeps
/// the table and the snapshot from drifting apart (one struct FIELD per op
/// would let them).
struct AdmissionProbeSnapshot {
    two_arm: std::collections::BTreeMap<&'static str, jammi_kernels::admission::DispatchSnapshot>,
    attention_block_flash: jammi_kernels::admission::CascadeDispatchSnapshot,
}

impl AdmissionProbeSnapshot {
    /// Snapshots every registry key the table names for `dtype`, straight
    /// through `counters_for(key)` — the SAME `&'static DispatchCounters` the
    /// kernels' own `admit()` sites accumulate into (the
    /// `jammi_encoders::ln_dispatch_snapshot()`-style accessors are
    /// themselves `counters_for("layer_norm_fused")` under the hood).
    fn capture(dtype: jammi_kernels::admission::DtypeClass) -> Self {
        let two_arm = probed_report_keys(dtype)
            .into_iter()
            .map(|(_, key)| (key, jammi_kernels::admission::counters_for(key).snapshot()))
            .collect();
        Self {
            two_arm,
            attention_block_flash: jammi_encoders::attention_block_flash_dispatch_snapshot(),
        }
    }

    /// The [`jammi_kernels::admission::DispatchSnapshot`] for a REGISTRY key
    /// this snapshot captured, or `None` for a key outside the captured dtype
    /// class (never reached — the caller iterates [`probed_report_keys`] with
    /// the SAME `dtype` this was captured with).
    fn two_arm(&self, registry_key: &str) -> Option<jammi_kernels::admission::DispatchSnapshot> {
        self.two_arm.get(registry_key).copied()
    }
}

/// Whether a two-arm op's DELTA between `before` and `after` shows it fired
/// fused, fired eager, or was not exercised at all: `Some(true)` (fused moved,
/// eager did not), `Some(false)` (eager moved, fused did not), or `None`
/// (neither moved — the probe never reached this op — or both moved, an
/// ambiguous signal this fn never rounds up to a clean positive).
fn two_arm_holds(
    before: jammi_kernels::admission::DispatchSnapshot,
    after: jammi_kernels::admission::DispatchSnapshot,
) -> Option<bool> {
    let fused_moved = after.fused > before.fused;
    let eager_moved = after.eager > before.eager;
    match (fused_moved, eager_moved) {
        (true, false) => Some(true),
        (false, true) => Some(false),
        _ => None,
    }
}

/// The `reason` written for a `holds: false` op whose OWN probe window
/// recorded no `(op, predicate)` entry — an honest "this report cannot say",
/// never a guess.
///
/// Reachable causes, all genuine: an admission-gated dispatch on ANOTHER
/// thread moved this registry key's `eager` counter inside this probe's
/// before/after window (the attribution precondition this section's module
/// doc already documents), or a future admission-gated op dispatches off the
/// probe's own thread (see
/// [`jammi_kernels::admission::probe_capture_begin`]'s thread-locality doc).
const REASON_UNAVAILABLE: &str = "reason_unavailable";

/// The verbatim predicate key THIS probe's own capture window recorded for
/// `registry_op_key` — the `(op, predicate)` pair the kernel's own
/// `admit()` call pushed into
/// [`jammi_kernels::admission::probe_capture_begin`]'s sink DURING this job's
/// probe, never a re-derived guess and never another job's entry.
///
/// **Never [`jammi_kernels::admission::fallback_warnings_emitted`]'s most
/// recent entry for the op.** That list is
/// process-lifetime AND deduplicated on `(op, predicate)` — a job whose miss
/// repeats a pair an earlier job already burned pushes nothing, so the "most
/// recent entry for this op" would be the most recent DIFFERENT predicate,
/// from a different job at a different dtype, persisted durably on this job's
/// record. A before/after window over that same list cannot fix it either:
/// the dedupe sits UPSTREAM of the record, so the window is empty in exactly
/// the repeat case. The capture sink is a second, undeduplicated channel that
/// exists for precisely this window.
///
/// [`REASON_UNAVAILABLE`] when the window has no entry — see its doc for the
/// causes. Never a placeholder that reads like a measured predicate.
fn reason_from_probe_window(
    window: &[jammi_kernels::admission::ProbeMiss],
    registry_op_key: &str,
) -> String {
    jammi_kernels::admission::probe_capture_reason_for(window, registry_op_key)
        .unwrap_or(REASON_UNAVAILABLE)
        .to_string()
}

/// The compiled/device-level short-circuit reasons for `"flash"`, checked
/// BEFORE any probe result is consulted: `Some(..)` when flash is not even
/// reachable on this build/device — no probe was, or could have been,
/// attempted for it — `None` when a probe's own outcome should decide the
/// field instead.
fn flash_compiled_device_reason(device: &candle_core::Device) -> Option<serde_json::Value> {
    if !jammi_kernels::admission::CUDA_COMPILED {
        return Some(serde_json::json!({"holds": false, "reason": "cuda_not_compiled"}));
    }
    if !jammi_kernels::admission::FLASH_COMPILED {
        return Some(serde_json::json!({"holds": false, "reason": "flash_not_compiled"}));
    }
    if !device.is_cuda() {
        return Some(
            serde_json::json!({"holds": false, "reason": "device_is_cpu_or_metal_not_cuda"}),
        );
    }
    None
}

/// The `"flash"` field for an arm that ran NO probe at all — no encoder was
/// ever built to probe (the projection-head arm: `backbone_dtype` never
/// takes effect there, so [`probe_acceleration`] never reaches a forward
/// call). Distinct from [`flash_report`]'s `"probe_forward_failed"`, which
/// means a probe WAS attempted and its forward pass errored: passing
/// `probe_ok = false` into `flash_report` for this arm would fabricate "the
/// probe failed" for a probe that was never attempted.
fn flash_report_no_probe_attempted(device: &candle_core::Device) -> serde_json::Value {
    flash_compiled_device_reason(device).unwrap_or_else(
        || serde_json::json!({"holds": false, "reason": "no_encoder_to_probe_projection_head_arm"}),
    )
}

/// The `"flash"` field of the acceleration report for an arm that DID attempt a
/// probe. Checks compiled/device facts FIRST (each a plain, honestly-named
/// reason no probe is needed for); only when flash is compiled AND the
/// device is CUDA does it consult the probe's own outcome: `probe_ok = false`
/// means the probe's forward pass itself errored (`"probe_forward_failed"` —
/// a real attempt that failed, never confused with
/// [`flash_report_no_probe_attempted`]'s "never even tried"); otherwise it
/// reads the `attention_block_flash` cascade delta. On a decline, `window` —
/// THIS probe's own `jammi_kernels::admission::probe_capture_begin()` capture
/// (the same one [`reason_from_probe_window`] reads for the two-arm `ops`
/// map) — is read back through
/// [`jammi_kernels::admission::probe_capture_reason_for`] for the
/// `"attention_block_flash"` cascade key: `admit_cascade` records every
/// decline into that SAME sink (see this section's module doc's "The BERT/
/// DistilBERT case" paragraph), so a BERT/DistilBERT job's
/// decline reads back verbatim as `"flash_transport_not_wired"`. The coarse
/// `"capability_or_domain_miss"` is kept ONLY as the fallback for a decline
/// whose window has no entry (the same causes [`REASON_UNAVAILABLE`]
/// documents for a two-arm op) — never a re-derived guess in its place.
fn flash_report(
    device: &candle_core::Device,
    probe_ok: bool,
    window: &[jammi_kernels::admission::ProbeMiss],
    before: jammi_kernels::admission::CascadeDispatchSnapshot,
    after: jammi_kernels::admission::CascadeDispatchSnapshot,
) -> serde_json::Value {
    if let Some(reason) = flash_compiled_device_reason(device) {
        return reason;
    }
    if !probe_ok {
        return serde_json::json!({"holds": false, "reason": "probe_forward_failed"});
    }
    let fused_moved = after.fused > before.fused;
    let declined_moved = after.declined > before.declined;
    match (fused_moved, declined_moved) {
        (true, false) => serde_json::json!({"holds": true, "reason": "domain_ok"}),
        (false, true) => {
            serde_json::json!({"holds": false, "reason": flash_cascade_decline_reason(window)})
        }
        _ => serde_json::json!({"holds": false, "reason": "flash_not_exercised_by_probe"}),
    }
}

/// The reason [`flash_report`] writes for a `holds: false` `attention_block_
/// flash` cascade delta: THIS probe's own capture window, read back for the
/// `"attention_block_flash"` registry key exactly the way
/// [`reason_from_probe_window`] reads a two-arm op's — through
/// [`jammi_kernels::admission::probe_capture_reason_for`], never a re-derived
/// guess.
///
/// Deliberately its OWN fallback, not [`REASON_UNAVAILABLE`]: the counter
/// delta already confirms a decline genuinely happened here (unlike a
/// two-arm op's `holds: false`, which can ALSO mean "never reached" —
/// [`two_arm_holds`]'s `None` case, which never calls this at all), so the
/// honest fallback for a decline whose window carries no entry is the
/// coarser-but-still-true `"capability_or_domain_miss"`, never a claim that
/// nothing can be said.
fn flash_cascade_decline_reason(window: &[jammi_kernels::admission::ProbeMiss]) -> &'static str {
    jammi_kernels::admission::probe_capture_reason_for(window, "attention_block_flash")
        .unwrap_or("capability_or_domain_miss")
}

/// A human-readable label for the resolved device: the CUDA driver's device
/// name when available (`jammi_kernels::admission::probe_cuda_device_name`),
/// else a plain `"cuda"`/`"metal"`/`"cpu"` kind — never a raw `Debug` dump
/// (candle's `Device` debug form is not designed as a durable-artifact
/// field).
fn device_report_label(device: &candle_core::Device) -> String {
    if device.is_cuda() {
        jammi_kernels::admission::probe_cuda_device_name(device).unwrap_or_else(|| "cuda".into())
    } else if device.is_metal() {
        "metal".to_string()
    } else {
        "cpu".to_string()
    }
}

/// Runs ONE backward pass + one `AdamW` step over `output` (the probe
/// forward's own pooled result), on the REAL production trainable weights
/// this job is about to train with — a forward-only probe would leave a
/// vacuous-coverage gap: backward/optimizer-time admission-gated dispatch
/// (e.g. `LowRankResidualLinear::bwd`'s `cast_add_bf16` epilogue,
/// `crates/jammi-kernels/src/ops/low_rank_residual_linear.rs`) never fires
/// during a plain forward, so a forward-only probe could not honestly claim
/// to have measured it.
///
/// **Restores every trainable var to its pre-probe value afterward**, via a
/// genuine deep copy (`Tensor::copy`, not `Tensor::clone` — candle's `clone`
/// shares the underlying storage `Arc`, so a "snapshot" taken that way would
/// silently mutate alongside the very weights `Var::set` writes into
/// in-place — `candle_core::Var::set` writes through
/// `storage_mut_and_layout`). All-or-nothing: if EVERY
/// trainable var cannot be snapshotted first, nothing is mutated at all —
/// never a partial, unrestorable snapshot. Best-effort throughout: any
/// failure (snapshot, backward, or the optimizer step) is logged and
/// swallowed, never propagated — a probe must never fail the training this
/// attempt is about to run.
///
/// **Disclosed, not eliminated**: this restores every trainable WEIGHT, but
/// not the ONE dropout-mask RNG draw the probe forward already consumed
/// (`DropoutMasks::next_key`, called once per training forward regardless of
/// which arm dispatches) — the real run's dropout stream is shifted by
/// exactly one draw relative to a build without this probe, at the same
/// seed. `crate::fine_tune::adamw::AdamW`'s own moment buffers are
/// freshly allocated inside THIS function's throwaway `AdamW` instance and
/// never shared with the real trainer's optimizer, so they leave no residue.
fn run_backward_and_optimizer_probe(varmap: &candle_nn::VarMap, output: &candle_core::Tensor) {
    let vars = varmap.all_vars();
    let snapshot: Option<Vec<candle_core::Tensor>> =
        vars.iter().map(|v| v.as_tensor().copy().ok()).collect();
    let Some(snapshot) = snapshot else {
        tracing::warn!(
            "could not snapshot every trainable var before the acceleration-report \
             probe's backward+optimizer step; skipping it entirely rather than risk an \
             unrestorable mutation of this job's real initial weights"
        );
        return;
    };

    let result = (|| -> candle_core::Result<()> {
        let loss = output
            .to_dtype(candle_core::DType::F32)?
            .sqr()?
            .mean_all()?;
        let grads = loss.backward()?;
        let mut opt = crate::fine_tune::adamw::AdamW::new(
            varmap.all_vars(),
            candle_nn::ParamsAdamW::default(),
        )?;
        opt.step(&grads)
    })();
    if let Err(e) = &result {
        tracing::warn!(
            error = %e,
            "acceleration-report probe's backward+optimizer step failed (non-fatal; \
             restoring pre-probe weights regardless)"
        );
    }

    for (var, original) in vars.iter().zip(snapshot.iter()) {
        if let Err(e) = var.set(original) {
            tracing::warn!(
                error = %e,
                "failed to restore a trainable var after the acceleration-report \
                 probe's backward+optimizer step — this job's initial weights may now differ \
                 from what it was configured with"
            );
        }
    }
}

/// Runs the acceleration probe — forward pass, then backward + one optimizer step
/// (see [`run_backward_and_optimizer_probe`]) — when `encoder`/`varmap` are
/// both `Some`, and builds the `ops` map + `flash` field from the
/// before/after dispatch-registry delta. Either is `None` on the
/// projection-head arm (`backbone_dtype` never takes effect there — see
/// `validate_backbone_precision`'s doc): `ops` is then empty and `flash`
/// reports the honest "no probe was ever attempted" reason
/// ([`flash_report_no_probe_attempted`]) — never a fabricated per-op
/// measurement, and never [`flash_report`]'s `"probe_forward_failed"` for a
/// probe that was never even tried.
fn probe_acceleration(
    device: &candle_core::Device,
    backbone_dtype: jammi_numerics::ComputePrecision,
    varmap: Option<&candle_nn::VarMap>,
    encoder: Option<&mut jammi_encoders::AnyEncoder>,
) -> (
    serde_json::Map<String, serde_json::Value>,
    serde_json::Value,
) {
    let (Some(encoder), Some(varmap)) = (encoder, varmap) else {
        return (
            serde_json::Map::new(),
            flash_report_no_probe_attempted(device),
        );
    };
    let dtype = dtype_class_of(backbone_dtype);

    // Every fused-kernel admission predicate this probe reads is gated on
    // TRAINING mode (`LayerNorm::forward`'s `(bias.is_none(), training)`
    // match; `ModernBertAttention`/`RotaryEmbedding`'s `self.training`
    // branches) — an eval-mode forward never reaches ANY of them, fused or
    // eager, regardless of dtype (`jammi_encoders::layer_norm::LayerNorm::
    // forward`'s doc: "Eval (`training == false`) NEVER reaches the fused
    // arm"). The training loop
    // built moments later (`TrainingLoopBuilder::build`) calls
    // `set_training(true)` unconditionally anyway, so flipping it here first
    // changes nothing about the run this attempt actually trains.
    encoder.set_training(true);

    let before = AdmissionProbeSnapshot::capture(dtype);
    // Arm THIS probe's own capture window before the
    // forward, and read every `holds: false` reason back out of it. The window
    // is thread-local and this whole function (forward, `Tensor::backward()`'s
    // graph walk, `AdamW::step`) runs synchronously on the ONE
    // `spawn_blocking` thread `run_fine_tune_blocking` was handed — see
    // `jammi_kernels::admission::probe_capture_begin`'s doc for the constraint
    // and exactly where it would break (an async yield inside the probe,
    // or an admission-gated op dispatched from a rayon/spawned worker).
    let capture = jammi_kernels::admission::probe_capture_begin();
    // A tiny probe batch built by the ENCODER ITSELF
    // (`AnyEncoder::probe_input`): the smallest shape-valid batch for that
    // variant's own geometry — 1x4 zero token ids for a text tower (id `0`
    // is valid for any non-empty vocabulary, so this never depends on the
    // job's tokenizer/vocab size), a `[1, 3, image_size, image_size]` pixel
    // batch for the vision tower, a `[1, 4, T, num_mel_bins]` fusion
    // spectrogram for the audio one. A hand-built token batch here would
    // shape-fail on every media tower and report `probe_forward_failed` for
    // a job whose real forward is fine — a fabricated-looking negative about
    // acceleration that the job never earned. A
    // genuine probe FAILURE still degrades to an empty `ops` map — never a
    // propagated error (this function, and its caller, are infallible by
    // construction: training must be unaffected by a report-computation
    // failure).
    let probe_ok = (|| -> Option<()> {
        let probe = encoder.probe_input(device).ok()?;
        let output = encoder.forward_input(&probe.as_input()).ok()?;
        run_backward_and_optimizer_probe(varmap, &output);
        Some(())
    })()
    .is_some();
    let after = AdmissionProbeSnapshot::capture(dtype);
    // Disarmed here, not by drop: nothing after this point may contribute to
    // this job's window, and nothing before it may be lost.
    let window = capture.finish();

    let mut ops = serde_json::Map::new();
    if probe_ok {
        for (report_key, registry_key) in probed_report_keys(dtype) {
            let (Some(b), Some(a)) = (before.two_arm(registry_key), after.two_arm(registry_key))
            else {
                continue;
            };
            if let Some(holds) = two_arm_holds(b, a) {
                let reason = if holds {
                    "domain_ok".to_string()
                } else {
                    reason_from_probe_window(&window, registry_key)
                };
                ops.insert(
                    report_key.to_string(),
                    serde_json::json!({"holds": holds, "reason": reason}),
                );
            }
        }
    }

    let flash = flash_report(
        device,
        probe_ok,
        &window,
        before.attention_block_flash,
        after.attention_block_flash,
    );
    (ops, flash)
}

/// Builds the acceleration-report JSON payload for this attempt. See
/// this section's module doc for the full design; in short, `ops`/`flash` are
/// measured by running the real, public forward path over a tiny synthetic
/// batch and reading the SAME dispatch registries the kernels themselves
/// maintain — never a re-derivation of their domain predicates.
fn build_acceleration_report_json(
    attempt: u32,
    device: &candle_core::Device,
    backbone_dtype: jammi_numerics::ComputePrecision,
    varmap: Option<&candle_nn::VarMap>,
    encoder: Option<&mut jammi_encoders::AnyEncoder>,
) -> String {
    let (ops, flash) = probe_acceleration(device, backbone_dtype, varmap, encoder);
    serde_json::json!({
        "state": "determined",
        "attempt": attempt,
        "device": device_report_label(device),
        "dtype": backbone_dtype.to_string(),
        "cuda_compiled": jammi_kernels::admission::CUDA_COMPILED,
        "flash_compiled": jammi_kernels::admission::FLASH_COMPILED,
        "ops": ops,
        "flash": flash,
    })
    .to_string()
}

/// Computes and persists this attempt's acceleration report. Runs
/// synchronously inside the blocking training thread
/// (`run_fine_tune_blocking`), right after the device is resolved and (on the
/// encoder-adapters arm) right after `validate_backbone_precision` /
/// `build_encoder_adapters` return — before the training loop's first step,
/// so a status poll mid-training finds a `"determined"` report rather than
/// the submission-time `"pending"` marker for this run's whole lifetime
/// **whenever the write lands**; when it does not (see
/// [`persist_acceleration_report`]'s swallowing note), the row keeps the
/// `pending` marker until its terminal catalog write retires it.
///
/// Never fails training: report computation is infallible by construction
/// (see [`probe_acceleration`]'s doc), and a `false`/`Err` from
/// [`Catalog::record_acceleration_report`] itself (the lease was lost, or the
/// catalog write failed) is logged and swallowed here — this attempt's
/// eventual finalize/fail is governed entirely by the training loop that
/// follows, unaffected by whether this write landed.
// 8 plain params over a private, two-call-site fn reads more directly than a
// bespoke params struct would, for two calls that already differ only in
// `varmap`/`encoder`.
#[allow(clippy::too_many_arguments)]
fn compute_and_persist_acceleration_report(
    catalog: &Arc<Catalog>,
    job_id: &str,
    worker_id: &str,
    attempt: u32,
    role: RunnerRole,
    device: &candle_core::Device,
    backbone_dtype: jammi_numerics::ComputePrecision,
    varmap: Option<&candle_nn::VarMap>,
    encoder: Option<&mut jammi_encoders::AnyEncoder>,
) {
    let report_json =
        build_acceleration_report_json(attempt, device, backbone_dtype, varmap, encoder);
    // The lease holder alone writes the row's report (one writer per
    // attempt — the module doc's writer table, row 12); every other rank of
    // a gang computed the same probe and discards it: a `Rank` holds no
    // `LeaseHolder` to write as.
    let Some(holder) = role.lease_holder() else {
        return;
    };
    tokio::runtime::Handle::current().block_on(persist_acceleration_report(
        holder,
        catalog,
        job_id,
        worker_id,
        attempt,
        &report_json,
    ));
}

/// Construct an encoder-adapters target: load the frozen backbone weights from
/// the catalog artifact path, wrap the configured target modules with LoRA, and
/// return both the resulting encoder and the persisted adapter metadata that
/// pairs with the trained tensors on disk.
///
/// # Dispatch is `(family, task)`, and there is no default arm
///
/// The tower to build is decided by the checkpoint's own architecture — the
/// SHARED [`EncoderFamily`] predicate, the same one serving branches on — and
/// the job's declared task. Every unsupported combination, and every config
/// this crate has no loader for, is a typed refusal naming what was found and
/// what is supported.
///
/// This replaces a `_ => BERT` coercion. That arm was silent and
/// output-affecting: a checkpoint whose `config.json` merely happened to
/// deserialize as a `BertConfig` (a GPT-2 config does) trained a BERT tower
/// over foreign weights and published an adapter claiming that architecture.
/// Refusing is the only honest answer — there is no BERT here to adapt.
///
/// # Zero trainable sites is a refusal, never a run
///
/// A `target_modules` list that selects NOTHING on the tower this dispatch
/// picked is refused right after the tower is built, naming the tower, the
/// submitted selectors and that tower's own
/// [`lora_site_names`](jammi_encoders::AnyEncoder::lora_site_names). Without
/// that check the job trains zero parameters, publishes an empty adapter and
/// reports success — see the refusal's own comment for why nothing
/// downstream catches it.
/// [`build_encoder_adapters`]'s inputs, grouped so the call takes a single
/// argument rather than a naturally-wide positional list (the same
/// params-struct convention `RunFineTuneParams` above uses, preferred over
/// `#[allow(clippy::too_many_arguments)]`).
struct BuildEncoderAdaptersParams<'a> {
    base_model_id: &'a str,
    catalog: &'a Arc<Catalog>,
    artifact_store: &'a Arc<ArtifactStore>,
    config: &'a FineTuneConfig,
    task: ModelTask,
    varmap: &'a candle_nn::VarMap,
    device: &'a candle_core::Device,
    /// The session's one shared [`crate::model::hub::HubSource`] —
    /// the HF-fallback arm threads this through rather than building its own
    /// `hf_hub::api::sync::Api`.
    hub: &'a HubSource,
    /// This rank's dropout-mask seed (`RankContext::dropout_seed(config.seed)`
    /// — rank 0 is the identity, so a single-rank run passes `config.seed`):
    /// the A/B init seed stays `config.seed` on every rank, so every rank of
    /// a gang starts from identical adapter weights and draws its own masks
    /// (`LoraBuildConfig::dropout_seed`).
    dropout_seed: u64,
}

fn build_encoder_adapters(
    params: BuildEncoderAdaptersParams,
) -> Result<(jammi_encoders::AnyEncoder, jammi_lora::AdapterConfig)> {
    let BuildEncoderAdaptersParams {
        base_model_id,
        catalog,
        artifact_store,
        config,
        task,
        varmap,
        device,
        hub,
        dropout_seed,
    } = params;
    use std::path::Path;

    use crate::model::arch::{self, EncoderFamily};
    use jammi_lora::Tower;

    // Interpret the base-model id through the shared `ModelSource::parse`, so the
    // catalog key here matches what the submit and load sites registered the
    // backbone under (`local:`/`file://`/`/abs` → the path string;
    // `hf://owner/repo` / `owner/repo` → the repo id).
    let source = ModelSource::parse(base_model_id);
    let catalog_model_id = source.to_string();
    let is_hf = matches!(source, ModelSource::HuggingFace(_));

    let model_record = tokio::runtime::Handle::current()
        .block_on(catalog.get_model(&catalog_model_id))?
        .ok_or_else(|| {
            JammiError::FineTune(format!("Base model '{base_model_id}' not in catalog"))
        })?;

    let artifact_dir: std::path::PathBuf = match &model_record.location {
        Some(ModelLocation::External(p)) if !p.is_empty() => {
            // A directly-registered base model (HF cache / local dir): its
            // weights already sit on a path candle can mmap. Use it in place.
            let url = jammi_db::storage::StorageUrl::parse(p)?;
            if url.scheme() != jammi_db::storage::Scheme::File {
                return Err(JammiError::FineTune(format!(
                    "Base model '{base_model_id}' is registered at '{p}', which is not a \
                     local directory"
                )));
            }
            let path = std::path::PathBuf::from(url.path());
            if path.is_dir() {
                path
            } else {
                path.parent()
                    .ok_or_else(|| {
                        JammiError::FineTune(format!(
                            "Cannot determine model dir from location '{p}'"
                        ))
                    })?
                    .to_path_buf()
            }
        }
        // The base model is itself an engine-produced artifact — fetch the
        // bundle into a local dir candle can load from, so a worker on any
        // host resolves the same backbone.
        Some(ModelLocation::Artifact(artifact)) => tokio::runtime::Handle::current()
            .block_on(artifact_store.fetch_artifact(artifact.url()))?
            .dir()
            .to_path_buf(),
        _ => {
            if is_hf {
                // `[models] offline`: the resolver's `HuggingFace`
                // arm refuses a Hub fetch identically once no catalog row
                // resolved the model — see `super::super::model::resolver`
                // and `crate::model::hub`'s module docs for why the promise
                // is Hub-only. This arm reaches the Hub exactly the same way
                // (`hub.api().model(..).get(..)`), so it must refuse
                // identically rather than silently falling through to a
                // live network fetch offline was supposed to forbid.
                if hub.offline() {
                    return Err(JammiError::Model {
                        model_id: catalog_model_id.clone(),
                        message: format!(
                            "offline: `{catalog_model_id}` was never resolved online on \
                             this catalog"
                        ),
                    });
                }
                // Shared `HubSource` — the session's ONE Hub client, not a
                // fresh `Api::new()` built here (which would not read
                // `HF_TOKEN`, would ignore `[models]` entirely, and can panic
                // outright when `HOME` is unset).
                let repo = hub.api().model(catalog_model_id.clone());
                let weights = repo.get("model.safetensors").map_err(|e| {
                    JammiError::FineTune(format!(
                        "Cannot locate '{catalog_model_id}' in HF hub cache: {e}"
                    ))
                })?;
                weights
                    .parent()
                    .ok_or_else(|| {
                        JammiError::FineTune(
                            "Cannot determine model dir from HF hub cache path".into(),
                        )
                    })?
                    .to_path_buf()
            } else {
                return Err(JammiError::FineTune(format!(
                    "Base model '{base_model_id}' has no location in catalog"
                )));
            }
        }
    };

    // Config SOURCE ORDER is the resolver's own: the catalog
    // record's stored `config_json` when it has one, else the first existing
    // candidate on disk walking the shared frozen chain (`config.json`, then
    // the OpenCLIP `open_clip_config.json`). Reading `config.json` off disk
    // unconditionally would miss an OpenCLIP checkpoint entirely (its config
    // lives only in `open_clip_config.json`) and would disagree with the
    // resolver whenever the catalog carries a config the directory does not.
    let model_config: serde_json::Value = match model_record.config_json.as_deref() {
        Some(json) => serde_json::from_str(json).map_err(|e| {
            JammiError::FineTune(format!(
                "Cannot parse the catalog's stored config_json for base model \
                 '{base_model_id}': {e}"
            ))
        })?,
        None => {
            let config_path = arch::config_candidates(&artifact_dir).ok_or_else(|| {
                JammiError::FineTune(format!(
                    "Cannot find {} or {} for base model at {artifact_dir:?}",
                    arch::CONFIG_CANDIDATE_NAMES[0],
                    arch::CONFIG_CANDIDATE_NAMES[1],
                ))
            })?;
            let text = std::fs::read_to_string(&config_path)
                .map_err(|e| JammiError::FineTune(format!("Cannot read {config_path:?}: {e}")))?;
            serde_json::from_str(&text)
                .map_err(|e| JammiError::FineTune(format!("Cannot parse {config_path:?}: {e}")))?
        }
    };

    // `model_type` as the config SPELLS it — used ONLY in refusal messages,
    // never as a dispatch key (that is `family`, below, and
    // `dispatch_model_type` for the two GGUF lookups that still key on the
    // string). `<absent>` is a message word, not an architecture: it must
    // never reach a lookup, which is why the two are separate bindings.
    let model_type = model_config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("<absent>");

    let family = EncoderFamily::from_config(&model_config).ok_or_else(|| {
        JammiError::FineTune(format!(
            "unsupported model_type '{model_type}' for encoder-adapter fine-tuning \
             (base model '{base_model_id}'); supported: bert, roberta, camembert, \
             xlm-roberta, distilbert, modernbert, an OpenCLIP checkpoint \
             (open_clip_config.json with model_cfg), or an HF-CLAP audio checkpoint \
             (clap_audio_model). Leave `target_modules` empty to train a projection \
             head on a frozen backbone instead."
        ))
    })?;

    // The string the two GGUF lookups below still key on (`GgufArchitecture`
    // and, through `gguf_num_layers`, the DistilBERT field normalization),
    // read through the SAME shared reader serving reads — so a config that
    // declares no `model_type` (admitted as `family` = Bert just above)
    // reaches those lookups as `"bert"` rather than as the `<absent>` message
    // word, which would refuse a checkpoint this function already accepted.
    let dispatch_model_type = arch::config_model_type(&model_config);

    // GGUF/QLoRA: the base artifact SELECTS this — no new
    // trainer/config knob. The FROZEN precedence lives in `model::arch`
    // (`model.safetensors` -> `open_clip_model.safetensors` -> `model.gguf`),
    // the identical chain `ModelResolver::resolve_local` walks, so a fine-tune
    // and an inference load of the same directory can never read different
    // weights.
    let weights_path = arch::weights_candidates(&artifact_dir).ok_or_else(|| {
        JammiError::FineTune(format!(
            "No weights found at {artifact_dir:?} (need one of {:?})",
            arch::CANDLE_WEIGHTS_CANDIDATE_NAMES
        ))
    })?;
    let gguf_weights_path = artifact_dir.join(arch::GGUF_WEIGHTS_FILENAME);
    let is_gguf = weights_path.ends_with(arch::GGUF_WEIGHTS_FILENAME);
    if is_gguf
        && !matches!(
            family,
            EncoderFamily::Bert | EncoderFamily::DistilBert | EncoderFamily::ModernBert
        )
    {
        return Err(JammiError::FineTune(format!(
            "quantized (GGUF) fine-tuning is not supported for this architecture \
             (model_type '{model_type}') — GGUF is threaded only through the \
             BERT-family/DistilBERT/ModernBERT text towers, matching serving's own \
             refusal"
        )));
    }

    let lora_dropout = if config.lora_dropout > 0.0 {
        Some(config.lora_dropout as f32)
    } else {
        None
    };

    let lora = jammi_lora::LoraBuildConfig {
        target_modules: &config.target_modules,
        layers_to_transform: &config.layers_to_transform,
        lora_rank: config.lora_rank,
        lora_alpha: config.lora_alpha,
        use_rslora: config.use_rslora,
        lora_dropout,
        rank_pattern: &config.rank_pattern,
        init_mode: config.init_lora_weights,
        seed: config.seed,
        dropout_seed,
    };

    validate_backbone_precision(config.backbone_dtype, device)?;
    let backbone_dtype: candle_core::DType =
        jammi_encoders::compute_precision_to_dtype(config.backbone_dtype);
    // The adapter records the FAMILY's canonical architecture id, not the
    // raw config string: `roberta` and `bert` are one family and must produce
    // adapters the load seam classifies identically, and an OpenCLIP
    // checkpoint has no `model_type` field to copy at all.
    let adapter_cfg = jammi_lora::AdapterConfig::from_build(
        family.adapter_model_type(),
        &lora,
        config.backbone_dtype,
    );

    // GGUF/QLoRA: everything the three encoder builders below
    // need to train LoRA over a `FrozenBase::Quantized` backbone — built
    // ONCE here from the GGUF file's tensor data, exactly the way
    // `CandleBackend::load`'s inference path builds it (the SAME
    // `crate::model::backend::gguf` module, so a QLoRA fine-tune and an
    // inference load of the same `model.gguf` can never silently disagree
    // on which tensors are matmul-site or which dtype loaded).
    let gguf_backbone = if is_gguf {
        let arch =
            crate::model::backend::gguf::GgufArchitecture::from_model_type(dispatch_model_type)
                .ok_or_else(|| {
                    JammiError::FineTune(format!(
                        "quantized serving not supported for this architecture \
                         (model_type '{model_type}')"
                    ))
                })?;
        // Routes through the SAME normalization + layer-count authority
        // `CandleBackend::load`'s GGUF path and `estimate_gguf_residency`
        // use (`gguf::gguf_num_layers`) — a raw, un-normalized DistilBERT
        // config declares its layer count under the DistilBERT-native
        // `n_layers` name only, so reading `num_hidden_layers`/`num_layers`
        // off the raw config here would refuse every DistilBERT GGUF
        // fine-tune outright.
        let num_layers =
            crate::model::backend::gguf::gguf_num_layers(dispatch_model_type, &model_config)
                .ok_or_else(|| {
                    JammiError::FineTune(
                        "GGUF load requires num_hidden_layers (or num_layers) in config.json"
                            .into(),
                    )
                })?;
        Some(
            crate::model::backend::gguf::load_gguf_backbone(
                &gguf_weights_path,
                arch,
                num_layers,
                backbone_dtype,
                device,
                base_model_id,
            )
            .map_err(|e| JammiError::FineTune(format!("Load GGUF backbone: {e}")))?,
        )
    } else {
        None
    };
    let gguf_lookup = gguf_backbone.as_ref().map(|b| b.lookup());
    // For a GGUF base this points at the synthesized densified safetensors
    // file `load_gguf_backbone` writes (embeddings/norms/every other
    // non-matmul-site tensor, dequantized to `backbone_dtype`); every
    // construction site below that never consults `gguf_lookup` reads this
    // exactly the way it reads a real safetensors checkpoint.
    let vb_weights_path = match &gguf_backbone {
        Some(b) => b.densified_path.clone(),
        None => weights_path.clone(),
    };
    let weights_paths: Vec<&Path> = vec![vb_weights_path.as_path()];

    let (encoder, adapter_cfg) = match (family, task) {
        // ---- OpenCLIP text tower -------------------------------------
        (EncoderFamily::OpenClip, ModelTask::TextEmbedding) => {
            let text_config = jammi_encoders::ClipTextConfig::from_open_clip_config(&model_config)
                .map_err(|e| JammiError::FineTune(format!("Parse OpenCLIP text config: {e}")))?;
            let tower = jammi_encoders::ClipText::builder()
                .lora(lora)
                .backbone_dtype(backbone_dtype)
                .build(&weights_paths, &text_config, device, varmap)
                .map_err(|e| JammiError::FineTune(format!("Build OpenCLIP text encoder: {e}")))?;
            (
                jammi_encoders::AnyEncoder::ClipText(tower),
                adapter_cfg.with_tower(Tower::Text),
            )
        }
        // ---- OpenCLIP vision tower -----------------------------------
        (EncoderFamily::OpenClip, ModelTask::ImageEmbedding) => {
            let vision_config =
                jammi_encoders::OpenClipVisionConfig::from_open_clip_config(&model_config)
                    .map_err(|e| {
                        JammiError::FineTune(format!("Parse OpenCLIP vision config: {e}"))
                    })?;
            let tower = jammi_encoders::OpenClipVisionTransformer::builder()
                .lora(lora)
                .backbone_dtype(backbone_dtype)
                .build(&weights_paths, &vision_config, device, varmap)
                .map_err(|e| JammiError::FineTune(format!("Build OpenCLIP vision encoder: {e}")))?;
            (
                jammi_encoders::AnyEncoder::OpenClipVision(tower),
                adapter_cfg.with_tower(Tower::Vision),
            )
        }
        // ---- HF-CLAP HTSAT audio tower -------------------------------
        (EncoderFamily::ClapAudio, ModelTask::AudioEmbedding) => {
            let audio_config = jammi_encoders::HtsatAudioConfig::from_hf_clap_config(&model_config)
                .map_err(|e| JammiError::FineTune(format!("Parse CLAP audio config: {e}")))?;
            let tower = jammi_encoders::HtsatAudio::builder()
                .lora(lora)
                .backbone_dtype(backbone_dtype)
                .build(&weights_paths, &audio_config, device, varmap)
                .map_err(|e| JammiError::FineTune(format!("Build CLAP audio encoder: {e}")))?;
            (
                jammi_encoders::AnyEncoder::Htsat(Box::new(tower)),
                adapter_cfg.with_tower(Tower::Audio),
            )
        }
        // ---- BERT-family text towers ---------------------------------
        //
        // Reached only for a TEXT task: the guard below refuses an
        // image/audio task on a text checkpoint before any tower is built,
        // so a media job can never train a text tower over tokenized bytes.
        (
            EncoderFamily::Bert | EncoderFamily::DistilBert | EncoderFamily::ModernBert,
            ModelTask::ImageEmbedding | ModelTask::AudioEmbedding,
        )
        | (EncoderFamily::OpenClip, _)
        | (EncoderFamily::ClapAudio, _) => {
            return Err(JammiError::FineTune(format!(
                "encoder-adapter fine-tuning does not support task {task} on this base \
                 model (model_type '{model_type}', architecture family {family:?}, towers: \
                 {}). Supported pairs: text_embedding/classification/ner/regression on a \
                 BERT-family text tower, text_embedding or image_embedding on an OpenCLIP \
                 checkpoint, audio_embedding on an HF-CLAP audio checkpoint.",
                family.towers()
            )));
        }
        (EncoderFamily::DistilBert, _) => {
            let distilbert_config: jammi_encoders::DistilBertConfig =
                serde_json::from_value(model_config.clone()).map_err(|e| {
                    JammiError::FineTune(format!("Parse DistilBert config.json: {e}"))
                })?;
            let mut builder = jammi_encoders::DistilBert::builder()
                .lora(lora)
                .backbone_dtype(backbone_dtype);
            if let Some(lookup) = &gguf_lookup {
                builder = builder.weight_source(lookup);
            }
            (
                jammi_encoders::AnyEncoder::DistilBert(
                    builder
                        .build(&weights_paths, &distilbert_config, device, varmap)
                        .map_err(|e| {
                            JammiError::FineTune(format!("Build DistilBert encoder: {e}"))
                        })?,
                ),
                adapter_cfg,
            )
        }
        (EncoderFamily::ModernBert, _) => {
            let modernbert_config: jammi_encoders::ModernBertConfig =
                serde_json::from_value(model_config.clone()).map_err(|e| {
                    JammiError::FineTune(format!("Parse ModernBert config.json: {e}"))
                })?;
            let mut builder = jammi_encoders::ModernBert::builder()
                .lora(lora)
                .backbone_dtype(backbone_dtype);
            if let Some(lookup) = &gguf_lookup {
                builder = builder.weight_source(lookup);
            }
            (
                jammi_encoders::AnyEncoder::ModernBert(
                    builder
                        .build(&weights_paths, &modernbert_config, device, varmap)
                        .map_err(|e| {
                            JammiError::FineTune(format!("Build ModernBert encoder: {e}"))
                        })?,
                ),
                adapter_cfg,
            )
        }
        (EncoderFamily::Bert, _) => {
            let bert_config: jammi_encoders::BertConfig =
                serde_json::from_value(model_config.clone())
                    .map_err(|e| JammiError::FineTune(format!("Parse Bert config.json: {e}")))?;
            let mut builder = jammi_encoders::Bert::builder()
                .lora(lora)
                .backbone_dtype(backbone_dtype);
            if let Some(lookup) = &gguf_lookup {
                builder = builder.weight_source(lookup);
            }
            (
                jammi_encoders::AnyEncoder::Bert(
                    builder
                        .build(&weights_paths, &bert_config, device, varmap)
                        .map_err(|e| JammiError::FineTune(format!("Build Bert encoder: {e}")))?,
                ),
                adapter_cfg,
            )
        }
    };

    // A selector that matched NOTHING is a hard refusal, not a run.
    //
    // This arm is reached only when `config.target_modules` is non-empty (see
    // `classify_training_oom`'s doc for that split), so an encoder with zero
    // trainable `Var`s here means the caller ASKED for adapters and got none.
    // Nothing downstream notices: candle's backward pass differentiates only
    // through `is_variable()` nodes, so `loss.backward()` populates an empty
    // `GradStore`; `optimizer::clip_and_step` treats an EMPTY `trainable_vars`
    // as the one unambiguously benign reading and does not even warn; the run
    // then completes, publishes an `adapter.safetensors` holding no A/B
    // tensors, reports SUCCESS, and serves the BASE bytes under a fine-tuned
    // model id. A typo'd selector must fail the job instead — the same
    // conclusion `jammi-bench`'s own training tier reached
    // (`finetune_run.rs`'s zero-trainable refusal), installed here at the one
    // seam every encoder-adapters job passes through.
    //
    // TWO producers land here and the message must not blame the wrong one:
    // `target_modules` matching no site on THIS tower's vocabulary, or a
    // `layers_to_transform` restriction excluding every layer the selectors
    // would otherwise have matched (an off-by-one layer index on a fixture
    // with fewer layers). The restriction is therefore named whenever it is
    // `Some`, never silently folded into "correct the selectors".
    if encoder.trainable_params().is_empty() {
        // The tower this job actually built, as the caller named it: the
        // adapter's own `tower` for a multi-tower checkpoint, and the
        // family's single tower otherwise (BERT-family adapters carry
        // `tower: None` because there is nothing to discriminate).
        let tower = match adapter_cfg.tower {
            Some(Tower::Text) => "text",
            Some(Tower::Vision) => "vision",
            Some(Tower::Audio) => "audio",
            None => family.towers(),
        };
        let restriction = match &config.layers_to_transform {
            Some(layers) => format!(" restricted to layers {layers:?}"),
            None => String::new(),
        };
        return Err(JammiError::FineTune(format!(
            "target_modules {:?}{restriction} matched no LoRA site on the '{}' {tower} tower \
             (base model '{base_model_id}', model_type '{model_type}'), so this job would \
             train zero parameters and publish an adapter that changes nothing. This \
             tower's own LoRA site names are: {:?} — a selector matches a site name \
             exactly or as a suffix of it (so 'query' selects 'attention.self.query' on a \
             BERT-family tower, whose sites are named by their full dotted checkpoint \
             path), and 'all-linear' selects every site.{}",
            config.target_modules,
            family.adapter_model_type(),
            encoder.lora_site_names(),
            if restriction.is_empty() {
                ""
            } else {
                " If the selectors are right for this architecture, check \
                 layers_to_transform for an out-of-range or off-by-one layer index."
            },
        )));
    }

    Ok((encoder, adapter_cfg))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use candle_core::Tensor;

    use super::*;

    /// A source-level oracle: `materialize_graph_training_
    /// set`'s own function body — read from disk at test time, not
    /// transcribed — calls `GraphSampler::sample_into` (the streaming,
    /// no-whole-Vec emit path) and never `GraphSampler::sample` (the
    /// whole-`Vec<SampledPair>`-collecting convenience wrapper). A grep/syn
    /// class oracle over compiled behaviour cannot see WHICH method was
    /// called if both compile fine and both satisfy the same trait-free
    /// signature elsewhere, so this reads the actual source text: a static
    /// property no unit test that only exercises inputs/outputs can prove.
    ///
    /// Replacing the production `sampler.sample_into(|pair| { ... })?;` call
    /// with `for pair in sampler.sample()? { ... }` — functionally equivalent
    /// output, but building the whole-output `Vec<SampledPair>` this
    /// property forbids — fails this test (the scan finds `.sample()`
    /// present and `.sample_into(` absent from the reduced body).
    #[test]
    fn materialize_graph_training_set_streams_through_sample_into_never_sample() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/fine_tune/worker.rs");
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let start_marker = "pub(crate) async fn materialize_graph_training_set(";
        let start = source
            .find(start_marker)
            .expect("materialize_graph_training_set must still exist under this exact name");
        // The function ends at the first top-level `\n}\n` after its own
        // opening brace, tracked by a simple brace depth counter over the
        // text from the signature onward — good enough to isolate ONE
        // function's body from its neighbours without a full parser.
        let body_start = source[start..]
            .find('{')
            .map(|i| start + i)
            .expect("the function must have a body");
        let mut depth = 0i32;
        let mut body_end = None;
        for (offset, ch) in source[body_start..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        body_end = Some(body_start + offset + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        let body_end = body_end.expect("a balanced function body must close its own brace");
        let body = &source[body_start..body_end];

        assert!(
            body.contains(".sample_into("),
            "materialize_graph_training_set must call GraphSampler::sample_into (the \
             streaming emit path) — the function body has changed shape; update this \
             oracle only if the streaming property itself is being deliberately re-verified \
             under a new name"
        );
        assert!(
            !body.contains(".sample()"),
            "materialize_graph_training_set must NEVER call GraphSampler::sample (the \
             whole-Vec<SampledPair>-collecting convenience wrapper) — that reintroduces a \
             second full-output copy this property forbids"
        );
    }

    /// The materialization descriptor's `collective`/`local_ranks` name the
    /// topology the run EXECUTES at, and the table is total: each method is
    /// exhaustive, so a new variant fails to compile before it can record
    /// the wrong token.
    #[test]
    fn topology_records_the_executed_collective_and_host_ranks() {
        let samples = [
            (TopologyDecision::Single, "noop", 1),
            (TopologyDecision::Local { world: 3 }, "local", 3),
            (TopologyDecision::Peer { world: 2 }, "peer", 1),
        ];
        for (topology, token, ranks) in samples {
            assert_eq!(topology.collective_token(), token);
            assert_eq!(topology.host_ranks(), ranks);
        }
        // Non-degeneracy: the three arms are pairwise distinct on the
        // token, so a collapsed match could not pass.
        let tokens: std::collections::BTreeSet<&str> = samples
            .iter()
            .map(|(t, _, _)| t.collective_token())
            .collect();
        assert_eq!(tokens.len(), 3);
    }

    /// Every exit arm of the coordinator body records
    /// exactly one `AssemblyOutcome` through the total table — and only the
    /// CAS `Moved` arm writes nothing. Table-driven over one sample per
    /// variant: `CoordinatorEnd::ordinal` is an exhaustive match (a variant
    /// without an arm does not compile), the ordinals of the samples must
    /// be exactly `0..VARIANTS` (a variant without a sample reds this), and
    /// each sample's outcome must be the documented row. The member-abort
    /// sub-table is one to one over every frozen `AbortReason`, with the
    /// out-of-set value reading transient. The counting class every
    /// assembly end settles its lease by
    /// (`AssemblyOutcome::counts_toward_failures`) is pinned for every
    /// variant; the settlement itself is the next oracle's.
    #[test]
    fn every_coordinator_end_records_exactly_one_assembly_outcome_and_only_moved_writes_nothing() {
        use std::collections::BTreeSet;

        use jammi_db::catalog::jobs_repo::AssemblyOutcome as O;
        use CoordinatorEnd as E;

        let samples: Vec<(E, Option<O>)> = vec![
            (E::Moved, None),
            (
                E::HostCannotCoordinate("no root".into()),
                Some(O::ShortListed),
            ),
            (E::CatalogFault("driver".into()), Some(O::Unavailable)),
            (
                E::ShortListed {
                    fresh: 0,
                    needed: 1,
                },
                Some(O::ShortListed),
            ),
            (
                E::MemberUnreachable {
                    rank: 1,
                    instance_id: "m".into(),
                },
                Some(O::Unavailable),
            ),
            (
                E::MemberRefused {
                    rank: 1,
                    instance_id: "m".into(),
                    detail: "slot busy".into(),
                },
                Some(O::Unavailable),
            ),
            (E::PeerRefused("cap".into()), Some(O::Unavailable)),
            (E::Cancelled, Some(O::Cancelled)),
            (E::Drain, Some(O::Drain)),
            (
                E::MemberAborted {
                    rank: 1,
                    reason: AbortReason::Refuted,
                },
                Some(O::Refuted),
            ),
            (E::LinkFault("deadline".into()), Some(O::Unavailable)),
            (E::TrainingFailed("diverged".into()), Some(O::Success)),
            (E::Published, Some(O::Success)),
        ];
        let ordinals: BTreeSet<usize> = samples.iter().map(|(end, _)| end.ordinal()).collect();
        assert_eq!(
            ordinals,
            (0..E::VARIANTS).collect::<BTreeSet<usize>>(),
            "one sample per variant: an end without a sample here has no row in this oracle"
        );
        assert_eq!(samples.len(), E::VARIANTS, "no variant is sampled twice");
        for (end, expected) in &samples {
            assert_eq!(assembly_outcome(end), *expected, "{end}");
        }
        assert_eq!(
            samples.iter().filter(|(_, o)| o.is_none()).count(),
            1,
            "exactly one end — the CAS Moved arm — writes nothing"
        );

        for (reason, expected) in [
            (AbortReason::Refuted, O::Refuted),
            (AbortReason::Unavailable, O::Unavailable),
            (AbortReason::StoreUnavailable, O::StoreUnavailable),
            (AbortReason::NoBody, O::NoBody),
            (AbortReason::Drain, O::Drain),
            (AbortReason::Cancelled, O::Cancelled),
            (AbortReason::Unspecified, O::Unavailable),
        ] {
            assert_eq!(
                assembly_outcome(&E::MemberAborted { rank: 2, reason }),
                Some(expected),
                "a member's Aborted({reason:?}) maps one to one"
            );
        }

        assert!(O::Refuted.counts_toward_failures());
        assert!(O::AllRootDivergent.counts_toward_failures());
        for outcome in [
            O::Unavailable,
            O::StoreUnavailable,
            O::ShortListed,
            O::NoBody,
            O::Drain,
            O::Cancelled,
            O::Success,
        ] {
            assert!(
                !outcome.counts_toward_failures(),
                "{outcome:?} is not counted toward assembly_failures"
            );
        }
    }

    /// The lease settlement is a TOTAL function of the end (`lease_settlement` is an
    /// exhaustive match; one sample per `ordinal` in `0..VARIANTS` here, so
    /// a new end without a sample reds this). The split: a member's
    /// `Aborted{Drain}` RELEASES (`releases + 1`, zero net attempts); every other mid-run gang fault — a member's `Aborted` for
    /// every other frozen reason and the out-of-set value, and a
    /// `LinkFault` (a dropped stream, a rank silent past the deadline, a
    /// peer's round fault) — leaves the lease to EXPIRE (an attempt spent
    /// at the successor's claim); an assembly end settles by its recorded
    /// outcome's counting class (every one is uncounted today → released);
    /// `Moved`/`Published`/`TrainingFailed`/`Cancelled` are the caller's
    /// arms (untouched here).
    #[test]
    fn every_coordinator_end_settles_its_lease_by_the_released_vs_failed_split() {
        use std::collections::BTreeSet;

        use LeaseSettlement as S;

        let member_aborted =
            |reason: AbortReason| CoordinatorEnd::MemberAborted { rank: 1, reason };
        let samples: Vec<(CoordinatorEnd, S)> = vec![
            (CoordinatorEnd::Moved, S::Untouched),
            (
                CoordinatorEnd::HostCannotCoordinate("no root".into()),
                S::Release,
            ),
            (CoordinatorEnd::CatalogFault("io".into()), S::Release),
            (
                CoordinatorEnd::ShortListed {
                    fresh: 0,
                    needed: 1,
                },
                S::Release,
            ),
            (
                CoordinatorEnd::MemberUnreachable {
                    rank: 1,
                    instance_id: "m".into(),
                },
                S::Release,
            ),
            (
                CoordinatorEnd::MemberRefused {
                    rank: 1,
                    instance_id: "m".into(),
                    detail: "busy".into(),
                },
                S::Release,
            ),
            (CoordinatorEnd::PeerRefused("cap".into()), S::Release),
            (CoordinatorEnd::Cancelled, S::Untouched),
            (CoordinatorEnd::Drain, S::Release),
            (member_aborted(AbortReason::Drain), S::Release),
            (CoordinatorEnd::LinkFault("timed out".into()), S::Expire),
            (CoordinatorEnd::TrainingFailed("nan".into()), S::Untouched),
            (CoordinatorEnd::Published, S::Untouched),
        ];
        let ordinals: BTreeSet<usize> = samples.iter().map(|(end, _)| end.ordinal()).collect();
        assert_eq!(
            ordinals,
            (0..CoordinatorEnd::VARIANTS).collect::<BTreeSet<_>>(),
            "one sample per CoordinatorEnd variant"
        );
        for (end, expected) in &samples {
            assert_eq!(
                lease_settlement(end),
                *expected,
                "the settlement of {end} (ordinal {})",
                end.ordinal()
            );
        }

        // The member-abort sub-table: exactly Drain releases; every other
        // frozen reason, and the out-of-set value, spends the attempt.
        for reason in [
            AbortReason::Refuted,
            AbortReason::Unavailable,
            AbortReason::StoreUnavailable,
            AbortReason::NoBody,
            AbortReason::Cancelled,
            AbortReason::Unspecified,
        ] {
            assert_eq!(
                lease_settlement(&member_aborted(reason)),
                S::Expire,
                "a member's Aborted({reason:?}) mid-run is a rank failure: the attempt is spent"
            );
        }
        assert_eq!(
            lease_settlement(&member_aborted(AbortReason::Drain)),
            S::Release
        );

        // Every assembly end settles by its outcome's counting class — the
        // one rule, never a second table: an end whose outcome counts
        // would expire, and today none of them does.
        for (end, _) in &samples {
            if matches!(
                end,
                CoordinatorEnd::HostCannotCoordinate(_)
                    | CoordinatorEnd::CatalogFault(_)
                    | CoordinatorEnd::ShortListed { .. }
                    | CoordinatorEnd::MemberUnreachable { .. }
                    | CoordinatorEnd::MemberRefused { .. }
                    | CoordinatorEnd::PeerRefused(_)
                    | CoordinatorEnd::Drain
            ) {
                let outcome = assembly_outcome(end).expect("an assembly end records an outcome");
                let expected = if outcome.counts_toward_failures() {
                    S::Expire
                } else {
                    S::Release
                };
                assert_eq!(lease_settlement(end), expected, "{end}");
            }
        }
    }

    /// The holder derivation (the module doc's writer table): the
    /// `Coordinator` exactly when a kind that trains from a training-set
    /// table — `fine_tune` and `graph_fine_tune` alike — decides `Peer` over
    /// this host's `local_ranks`; the `LoopClaimer` for every other
    /// `(kind, world_size, local_ranks)` — `W == 1` on every kind and every
    /// host, and an in-process `Local` gang.
    #[test]
    fn the_lease_holder_is_the_coordinator_exactly_when_a_training_set_kind_decides_peer() {
        use crate::fine_tune::spec::TrainingCommon;
        use crate::fine_tune::{FineTuneConfig, FineTuneMethod};
        use jammi_db::store::CachePolicy;

        let common = |world_size: u32| TrainingCommon {
            base_model: "m".into(),
            config: FineTuneConfig::default(),
            world_size,
            cache: CachePolicy::Bypass,
        };
        let fine_tune = |world_size: u32| TrainingSpec::FineTune {
            source: "s".into(),
            columns: vec!["a".into(), "b".into()],
            method: FineTuneMethod::Lora,
            task: ModelTask::TextEmbedding,
            common: common(world_size),
        };
        for local_ranks in 1..=3u32 {
            for world_size in 1..=4u32 {
                let expected = match TopologyDecision::decide(world_size, local_ranks) {
                    TopologyDecision::Peer { .. } => LeaseHolder::Coordinator,
                    _ => LeaseHolder::LoopClaimer,
                };
                assert_eq!(
                    lease_holder_for(&fine_tune(world_size), local_ranks),
                    expected,
                    "fine_tune W={world_size} L={local_ranks}"
                );
                if world_size == 1 {
                    assert_eq!(
                        expected,
                        LeaseHolder::LoopClaimer,
                        "W == 1 never coordinates"
                    );
                }
                let graph = TrainingSpec::GraphFineTune {
                    sources: crate::fine_tune::graph_sampler::GraphFineTuneSources {
                        node_source: "n".into(),
                        edge_source: "e".into(),
                        id_column: "id".into(),
                        text_column: "text".into(),
                        src_column: "src".into(),
                        dst_column: "dst".into(),
                        provenance: crate::fine_tune::graph_sampler::EdgeProvenance::Declared,
                    },
                    sample_config: crate::fine_tune::graph_sampler::GraphSampleConfig::default(),
                    common: common(world_size),
                };
                assert_eq!(
                    lease_holder_for(&graph, local_ranks),
                    expected,
                    "graph_fine_tune W={world_size} L={local_ranks}: the same rule as fine_tune"
                );
            }
        }
    }

    /// Rank assignment is a pure function of the
    /// SORTED listing — every permutation of the same members yields the
    /// same `rank -> instance_id` map; the ranks are `1..world` over the
    /// first `world - 1` members in `instance_id` byte order; a listing
    /// shorter than `world - 1` is `ShortListing`, never a smaller gang
    /// (no substitution).
    #[test]
    fn rank_assignment_is_a_pure_function_of_the_sorted_listing() {
        let member = |id: &str, port: u16| GangMember {
            instance_id: id.to_string(),
            peer_addr: PeerAddr::parse(&format!("10.0.0.1:{port}")).unwrap(),
        };
        let members = vec![
            member("m-c", 3),
            member("m-a", 1),
            member("m-d", 4),
            member("m-b", 2),
        ];
        let reference = assign_ranks(&members, 4).expect("four members serve a gang of four");
        assert_eq!(
            reference
                .iter()
                .map(|(rank, m)| (*rank, m.instance_id.as_str()))
                .collect::<Vec<_>>(),
            vec![(1, "m-a"), (2, "m-b"), (3, "m-c")],
            "rank r is the r-th member in instance_id byte order; the surplus member is unused"
        );
        // Every permutation of the DB return order: the same assignment.
        let n = members.len();
        let mut indices: Vec<usize> = (0..n).collect();
        let mut permutations = Vec::new();
        fn heap(k: usize, a: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
            if k == 1 {
                out.push(a.clone());
                return;
            }
            heap(k - 1, a, out);
            for i in 0..k - 1 {
                if k.is_multiple_of(2) {
                    a.swap(i, k - 1);
                } else {
                    a.swap(0, k - 1);
                }
                heap(k - 1, a, out);
            }
        }
        heap(n, &mut indices, &mut permutations);
        assert_eq!(permutations.len(), 24);
        for permutation in permutations {
            let permuted: Vec<GangMember> =
                permutation.iter().map(|&i| members[i].clone()).collect();
            assert_eq!(
                assign_ranks(&permuted, 4).unwrap(),
                reference,
                "the assignment must not depend on the listing's return order: {permutation:?}"
            );
        }
        // Short: three fresh members cannot serve a gang of five.
        assert_eq!(
            assign_ranks(&members[..3], 5),
            Err(ShortListing {
                fresh: 3,
                needed: 4
            })
        );
        // A gang of two over an empty listing is short, not a gang of one.
        assert_eq!(
            assign_ranks(&[], 2),
            Err(ShortListing {
                fresh: 0,
                needed: 1
            })
        );
    }

    /// The topology is decided from two inputs alone: the job's
    /// `world_size` and this host's `[worker] local_ranks`.
    #[test]
    fn topology_is_decided_from_world_size_and_local_ranks_alone() {
        use TopologyDecision as T;
        assert_eq!(T::decide(1, 1), T::Single);
        assert_eq!(
            T::decide(1, 4),
            T::Single,
            "a wider host still runs a W=1 job as one rank"
        );
        assert_eq!(T::decide(0, 1), T::Single);
        assert_eq!(T::decide(2, 2), T::Local { world: 2 });
        assert_eq!(
            T::decide(2, 4),
            T::Local { world: 2 },
            "the gang is W ranks, not local_ranks"
        );
        assert_eq!(T::decide(2, 1), T::Peer { world: 2 });
        assert_eq!(
            T::decide(3, 2),
            T::Peer { world: 3 },
            "no hybrid: one rank past local_ranks makes every other rank a member"
        );
    }

    /// `WorkerFacts.devices`: config
    /// alone decides the list, no GPU needed. A `[gpu] device = -1` (the
    /// CPU) single-device session registers one fact, ordinal `0` (never
    /// `-1`); a two-configured-device session registers two facts in RANK
    /// order — the ordinals come straight from `rank_devices()`, unaffected
    /// by which backend this build actually runs on (the session's OWN
    /// `ComputeDevice` decides `kind`, uniformly, since one gang runs on
    /// one kind of device — `GpuConfig::validate` already refuses a mixed
    /// list).
    #[test]
    fn worker_devices_is_decided_from_configuration_alone() {
        let mut config = jammi_db::config::JammiConfig {
            gpu: jammi_db::config::GpuConfig {
                device: -1,
                ..Default::default()
            },
            ..Default::default()
        };
        let devices = worker_devices(&config, ComputeDevice::Cpu);
        assert_eq!(
            devices,
            vec![DeviceFact {
                kind: "cpu".into(),
                ordinal: 0,
            }],
            "{devices:?}"
        );

        config.gpu.device = 0;
        config.gpu.devices = Some(vec![0, 1]);
        config.worker.local_ranks = 2;
        let devices = worker_devices(&config, ComputeDevice::Cpu);
        assert_eq!(
            devices,
            vec![
                DeviceFact {
                    kind: "cpu".into(),
                    ordinal: 0,
                },
                DeviceFact {
                    kind: "cpu".into(),
                    ordinal: 1,
                },
            ],
            "two configured devices register two facts in rank order: {devices:?}"
        );

        let cuda = worker_devices(&config, ComputeDevice::Cuda { ordinal: 0 });
        assert_eq!(
            cuda[0].kind, "cuda",
            "the session's own ComputeDevice decides kind, uniformly"
        );
    }

    /// The honest-negative half: [`reason_from_probe_window`] returns the window's OWN verbatim
    /// predicate for an op it recorded, and [`REASON_UNAVAILABLE`] — never a
    /// guess, and never some other op's predicate — for one it did not.
    ///
    /// The `None` branch is not reachable deterministically through the
    /// public probe (it needs a concurrent thread's dispatch to move a
    /// counter inside this probe's window, or an off-thread admission
    /// site), so it is pinned here directly rather than left as an
    /// unexercised `unwrap_or`.
    #[test]
    fn reason_from_probe_window_reads_its_own_window_or_says_unavailable() {
        let window: Vec<jammi_kernels::admission::ProbeMiss> = vec![
            (
                "attention_block_fused",
                "head_dim_is_attention_block_fixed_head_dim",
            ),
            ("layer_norm_fused", "dtype_is_f32_bf16_or_f16"),
        ];
        assert_eq!(
            reason_from_probe_window(&window, "attention_block_fused"),
            "head_dim_is_attention_block_fixed_head_dim"
        );
        assert_eq!(
            reason_from_probe_window(&window, "layer_norm_fused"),
            "dtype_is_f32_bf16_or_f16",
            "each op reads ITS OWN entry — never the most recent entry in the window"
        );
        assert_eq!(
            reason_from_probe_window(&window, "lora_linear_fused"),
            REASON_UNAVAILABLE,
            "an op this window never recorded must get the honest unavailable marker, never a \
             neighbouring op's predicate"
        );
        assert_eq!(
            reason_from_probe_window(&[], "attention_block_fused"),
            REASON_UNAVAILABLE,
            "an empty window says so"
        );
    }

    /// `admit_cascade`'s decline path records
    /// `(op, predicate)` into the SAME probe-capture window `admit_inner`
    /// uses — `jammi_kernels::admission`'s `record_probe_miss(op,
    /// predicate_name)` — which is what
    /// lets [`flash_cascade_decline_reason`] — the function [`flash_report`]
    /// itself calls on a decline — read a verbatim reason back for the
    /// `"attention_block_flash"` cascade key instead of the coarse
    /// `"capability_or_domain_miss"` fallback.
    ///
    /// This drives the REAL `jammi_kernels::admission::admit_cascade` call
    /// (never a fabricated window entry) with BERT/DistilBERT's own verbatim
    /// predicate — `"flash_transport_not_wired"`, the ONE reason value either
    /// family's `FlashDecision::Declined` ever carries (see the
    /// `FlashDecision::Declined` construction in `jammi_encoders::bert` and
    /// `jammi_encoders::distilbert`) — for a
    /// `CapabilityMiss` outcome on the `"attention_block_flash"` op, exactly
    /// as `attention_cascade::training_attention_cascade` does for a
    /// BERT-family training forward.
    ///
    /// This is the CPU-buildable half of the story
    /// `bert_family_job_reports_flash_decline_honestly`
    /// (`tests/it/acceleration_report.rs`) cannot reach: that integration
    /// test's build short-circuits on `flash_compiled_device_reason` (no CUDA
    /// device/feature) BEFORE the cascade delta — and this window — is ever
    /// consulted, so it can only pin the device-level reason. This test
    /// isolates the window-read mechanism itself, which needs no CUDA device
    /// at all: only the thread-local probe-capture sink, which is a plain
    /// `Vec` regardless of build features.
    #[test]
    fn flash_cascade_decline_reason_reads_bert_familys_verbatim_predicate_from_the_window() {
        let capture = jammi_kernels::admission::probe_capture_begin();
        let counters = jammi_kernels::admission::cascade_counters_for("attention_block_flash");
        let outcome = jammi_kernels::admission::admit_cascade(
            jammi_kernels::admission::AdmissionMode::Fallback,
            &jammi_kernels::admission::ATTENTION_BLOCK_FLASH,
            "flash_transport_not_wired",
            jammi_kernels::admission::PredicateOutcome::CapabilityMiss,
            true,
            counters,
        )
        .expect("Fallback mode's CapabilityMiss decline never errors");
        assert_eq!(
            outcome,
            jammi_kernels::admission::CascadeOutcome::Declined,
            "a CapabilityMiss outcome must decline, never fuse"
        );
        let window = capture.finish();

        assert_eq!(
            flash_cascade_decline_reason(&window),
            "flash_transport_not_wired",
            "admit_cascade's decline must thread verbatim into the SAME probe-capture window \
             flash_report reads back — never the coarse capability_or_domain_miss fallback when \
             a specific reason WAS captured"
        );
    }

    /// The fallback half of the same mechanism: an `"attention_block_flash"`
    /// decline whose window carries NO entry for it (e.g. captured on a
    /// different thread, or never captured at all) must still get the
    /// honest, coarser `"capability_or_domain_miss"` — never
    /// [`REASON_UNAVAILABLE`], which would wrongly cast doubt on whether a
    /// decline happened at all (the counter delta already confirms it did;
    /// see [`flash_cascade_decline_reason`]'s doc).
    #[test]
    fn flash_cascade_decline_reason_falls_back_to_the_coarse_reason_on_an_empty_window() {
        assert_eq!(
            flash_cascade_decline_reason(&[]),
            "capability_or_domain_miss"
        );
    }

    /// The premise the bf16-on-CPU refusal rests on, pinned rather than
    /// remembered: candle's CPU matmul does not implement BF16.
    ///
    /// If candle ever gains it, this test fails and says so — which is the
    /// signal to delete [`validate_backbone_precision`]'s CPU arm rather than
    /// leave a refusal in place for a limitation that no longer exists. A guard
    /// whose justification is only a comment outlives its reason.
    #[test]
    fn cpu_matmul_still_cannot_do_bf16() {
        use candle_core::{DType, Device, Tensor};
        let d = Device::Cpu;
        let a = Tensor::zeros((4, 4), DType::BF16, &d).unwrap();
        assert!(
            a.matmul(&a).is_err(),
            "candle CPU matmul now supports BF16 — remove the CPU arm of \
             validate_backbone_precision instead of keeping a stale refusal"
        );
        let f16 = Tensor::zeros((4, 4), DType::F16, &d).unwrap();
        assert!(
            f16.matmul(&f16).is_ok(),
            "F16 is the reduced precision the refusal steers callers to; it must work on CPU"
        );
    }

    #[test]
    fn bf16_backbone_is_refused_on_cpu_with_a_faithful_error() {
        use jammi_numerics::ComputePrecision;
        let err = validate_backbone_precision(ComputePrecision::BF16, &candle_core::Device::Cpu)
            .expect_err("bf16 on CPU must be refused, not silently downgraded");
        let msg = err.to_string();
        assert!(msg.contains("bf16"), "error must name the precision: {msg}");
        assert!(
            msg.contains("f16") || msg.contains("f32"),
            "error must name a usable alternative: {msg}"
        );
        assert!(
            matches!(err, JammiError::FineTune(_)),
            "typed error, got {err:?}"
        );
    }

    /// Positive control: the guard must refuse only the combination it targets.
    #[test]
    fn other_precisions_are_accepted_on_cpu() {
        use jammi_numerics::ComputePrecision;
        for p in [ComputePrecision::F32, ComputePrecision::F16] {
            assert!(
                validate_backbone_precision(p, &candle_core::Device::Cpu).is_ok(),
                "{p:?} must be accepted on CPU"
            );
        }
    }

    /// A panicking blocking trainer drives the job to a terminal `failed` status
    /// with the panic message recorded — never an uncaught unwind that wedges the
    /// worker loop and leaves the job stuck `running`. This runs the exact
    /// `catch_unwind` → `panic_message` → classify → `record_failed` pipeline the
    /// worker runs around [`run_fine_tune_blocking`], over a closure that panics
    /// in place of a candle/platform fault inside the trainer, and asserts on the
    /// catalog row the worker writes.
    #[tokio::test(flavor = "multi_thread")]
    async fn panicking_training_job_lands_failed_with_recorded_error() {
        use jammi_db::catalog::status::JobStatus;

        let dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(jammi_db::catalog::Catalog::open(dir.path()).await.unwrap());
        catalog
            .register_model(jammi_db::catalog::model_repo::RegisterModelParams {
                model_id: "panic-base",
                version: 1,
                model_type: "embedding",
                backend: "candle",
                task: ModelTask::TextEmbedding,
                base_model_id: None,
                external_location: None,
                config_json: None,
            })
            .await
            .unwrap();
        catalog
            .submit_job(jammi_db::catalog::jobs_repo::SubmitJobParams {
                job_id: "panic-job",
                kind: "fine_tune",
                execution: jammi_db::catalog::status::JobExecution::Queued,
                spec: "{}",
                model_ref: Some("panic-base::1"),
                output_model_id: None,
                model_source: None,
                priority: 0,
            })
            .await
            .unwrap();

        // The worker claims the job (running, leased to it) before running it —
        // the state in which a genuine failure is recorded under the lease guard.
        let claimed = catalog
            .claim_next("worker-x", &["fine_tune"], Duration::from_secs(60))
            .await
            .unwrap()
            .expect("the queued job is claimable");

        let cancel = Arc::new(AtomicBool::new(false));

        // Run the worker's blocking wrapper over a trainer that panics, then take
        // the same terminal-classification branch `train_fine_tune` does.
        let result = tokio::task::spawn_blocking(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
                panic!("simulated candle kernel fault");
            }))
        })
        .await;
        let outcome = match result {
            Ok(Ok(Ok(()))) => panic!("the closure was supposed to panic"),
            Ok(Ok(Err(e))) => classify(&cancel, e),
            Ok(Err(payload)) => WorkerJobError::Failed(JammiError::FineTune(format!(
                "Panic: {}",
                panic_message(payload.as_ref())
            ))),
            Err(join_err) => WorkerJobError::Failed(JammiError::FineTune(format!(
                "training task join error: {join_err}"
            ))),
        };

        let WorkerJobError::Failed(error) = outcome else {
            panic!("a genuine panic must classify as Failed, not Cancelled");
        };
        let msg = failed_job_message(&error);
        assert!(
            msg.contains("Panic:") && msg.contains("simulated candle kernel fault"),
            "a caught panic must carry its message into the failure, got: {msg}"
        );

        // The worker records the failure as the job's terminal status, under the
        // lease guard (it still owns the job).
        record_failed(
            LeaseHolder::LoopClaimer,
            &catalog,
            "panic-job",
            "worker-x",
            claimed.attempts,
            msg,
        )
        .await;

        let job = catalog.get_job("panic-job").await.unwrap();
        assert_eq!(
            job.status,
            JobStatus::Failed.to_string(),
            "a panicking job lands `failed`, never wedged `running`"
        );
        assert!(
            job.error
                .as_deref()
                .is_some_and(|m| m.contains("simulated candle kernel fault")),
            "the panic cause is recorded on the job, got {:?}",
            job.error
        );
    }

    /// The encoder-adapters arm (`target_modules` non-empty): a CUDA
    /// OOM-shaped failure is rewritten to name the config that OOM'd and the
    /// remedies to try, in order — `backbone_dtype: bf16` first (bf16
    /// actually takes effect here), then a smaller
    /// `batch_size`/`gradient_accumulation_steps`, then a smaller
    /// `max_seq_length`. The underlying driver text is preserved.
    #[test]
    fn classify_training_oom_encoder_adapters_arm_leads_with_bf16() {
        let config = FineTuneConfig {
            target_modules: vec!["query".into(), "value".into()],
            ..FineTuneConfig::default() // batch_size=8, max_seq_length=512, backbone_dtype=f32
        };
        let raw = JammiError::FineTune(
            "Encoder forward: cuda error: CUDA_ERROR_OUT_OF_MEMORY: out of memory".into(),
        );
        let classified = classify_training_oom(&config, raw);
        let msg = classified.to_string();

        assert!(
            msg.contains("CUDA out of memory"),
            "the matched text names CUDA, headline must say so: {msg}"
        );
        assert!(
            msg.contains("batch_size=8"),
            "must name the OOM'd batch_size, got: {msg}"
        );
        assert!(
            msg.contains("max_seq_length=512"),
            "must name the OOM'd max_seq_length, got: {msg}"
        );
        assert!(
            msg.contains("backbone_dtype=f32"),
            "must name the OOM'd backbone_dtype, got: {msg}"
        );
        assert!(
            msg.contains("backbone_dtype: bf16"),
            "first remedy on the encoder-adapters arm must be the bf16 backbone, got: {msg}"
        );
        assert!(
            msg.contains("bf16 requires CUDA"),
            "must state bf16's CUDA requirement, got: {msg}"
        );
        assert!(
            msg.contains("batch_size") && msg.contains("gradient_accumulation_steps"),
            "second remedy must name smaller batch_size / gradient_accumulation_steps, got: {msg}"
        );
        assert!(
            msg.contains("max_seq_length"),
            "third remedy must name a smaller max_seq_length, got: {msg}"
        );
        assert!(
            msg.contains("CUDA_ERROR_OUT_OF_MEMORY"),
            "underlying driver error text must survive, got: {msg}"
        );
    }

    /// The projection-head arm (`target_modules` empty, the default):
    /// `backbone_dtype` never takes effect there (only `build_encoder_adapters`
    /// re-dtypes the backbone, and it is reached only for a non-empty
    /// `target_modules`), so the remedy list must NOT suggest `backbone_dtype:
    /// bf16` — that would be dead advice on this arm. The message says so
    /// outright.
    #[test]
    fn classify_training_oom_projection_head_arm_omits_bf16() {
        let config = FineTuneConfig::default(); // target_modules empty
        let raw = JammiError::FineTune(
            "Encoder forward: cuda error: CUDA_ERROR_OUT_OF_MEMORY: out of memory".into(),
        );
        let classified = classify_training_oom(&config, raw);
        let msg = classified.to_string();

        assert!(
            msg.contains("batch_size=8") && msg.contains("max_seq_length=512"),
            "must still name the OOM'd batch_size/max_seq_length, got: {msg}"
        );
        assert!(
            !msg.contains("backbone_dtype="),
            "the echoed config must be arm-appropriate — backbone_dtype is never echoed \
             (and then disclaimed) on the projection-head arm, got: {msg}"
        );
        assert!(
            !msg.contains("backbone_dtype: bf16"),
            "bf16 is inert on the projection-head arm — must not be suggested, got: {msg}"
        );
        assert!(
            msg.contains("backbone_dtype does not apply to projection-head runs"),
            "must say outright why bf16 is absent, got: {msg}"
        );
        assert!(
            msg.contains("smaller batch_size") || msg.contains("a smaller batch_size"),
            "must still suggest a smaller batch_size, got: {msg}"
        );
        assert!(
            msg.contains("gradient_accumulation_steps"),
            "must still suggest gradient_accumulation_steps, got: {msg}"
        );
        assert!(
            msg.contains("max_seq_length"),
            "must still suggest a smaller max_seq_length, got: {msg}"
        );
    }

    /// The headline names CUDA only as strongly as the matched text supports:
    /// this function never threads through which device the job actually ran
    /// on, so an OOM-shaped message that never mentions CUDA gets the more
    /// conservative "out of memory (device or host)" headline, not an
    /// asserted "CUDA out of memory".
    #[test]
    fn classify_training_oom_headline_is_conservative_without_cuda_in_the_text() {
        let config = FineTuneConfig::default();
        let raw = JammiError::FineTune("process was killed: out of memory".into());
        let classified = classify_training_oom(&config, raw);
        let msg = classified.to_string();
        assert!(
            msg.contains("out of memory (device or host) while training"),
            "no CUDA evidence in the matched text — must not assert CUDA, got: {msg}"
        );
        assert!(
            !msg.contains("CUDA out of memory"),
            "must not upgrade to a CUDA claim the text doesn't support, got: {msg}"
        );
    }

    /// Negative control: a genuine non-OOM CUDA failure (a kernel/PTX fault —
    /// the error most at risk of being misrouted as an OOM) must pass through completely unchanged —
    /// the OOM guidance is never attached to a failure batch-halving or a
    /// backbone-dtype change cannot fix.
    #[test]
    fn classify_training_oom_leaves_non_oom_errors_unchanged() {
        let config = FineTuneConfig::default();
        let raw = JammiError::FineTune("Encoder forward: CUDA_ERROR_INVALID_PTX".into());
        let raw_msg = raw.to_string();
        let classified = classify_training_oom(&config, raw);
        assert_eq!(
            classified.to_string(),
            raw_msg,
            "a non-OOM error must not be rewritten"
        );
    }

    /// The full error-arm lattice `classify_training_error` runs: cancellation
    /// wins over an OOM-shaped message (a lease-lost job is never rewritten as
    /// an OOM failure), a genuine OOM is classified, and a non-OOM error
    /// passes through byte-identical.
    #[test]
    fn classify_training_error_lattice_cancelled_oom_and_passthrough() {
        let config = FineTuneConfig::default();

        // cancelled (flag set) + OOM-shaped text: cancellation wins.
        let cancel = AtomicBool::new(true);
        let oom_err = JammiError::FineTune("cuda_error_out_of_memory".into());
        assert!(
            matches!(
                classify_training_error(&cancel, &config, oom_err),
                WorkerJobError::Cancelled
            ),
            "a lease-lost job must classify as Cancelled even over OOM-shaped text"
        );

        // not cancelled + OOM-shaped: classified with guidance.
        let cancel = AtomicBool::new(false);
        let oom_err = JammiError::FineTune("cuda_error_out_of_memory".into());
        let WorkerJobError::Failed(oom) = classify_training_error(&cancel, &config, oom_err) else {
            panic!("a genuine OOM must classify as Failed, not Cancelled");
        };
        let msg = failed_job_message(&oom);
        assert!(
            msg.contains("out of memory") && msg.contains("batch_size=8"),
            "must carry the classified OOM guidance, got: {msg}"
        );

        // not cancelled + non-OOM: byte-identical passthrough of the RAW
        // inner message — `failed_job_message` strips `JammiError::FineTune`'s
        // own "Fine-tune error: " prefix here (see its doc): that prefix is
        // re-applied on read by `TrainingJob::wait()` and the Python
        // binding's `poll_until_terminal`, so it must not survive into the
        // stored `WorkerJobError::Failed` message a second time.
        let cancel = AtomicBool::new(false);
        let raw_inner = "Encoder forward: CUDA_ERROR_INVALID_PTX";
        let raw = JammiError::FineTune(raw_inner.into());
        let WorkerJobError::Failed(passed) = classify_training_error(&cancel, &config, raw) else {
            panic!("a non-OOM failure must classify as Failed, not Cancelled");
        };
        let msg = failed_job_message(&passed);
        assert_eq!(
            msg, raw_inner,
            "a non-OOM error must pass through unchanged, minus the redundant \
             Fine-tune-error prefix"
        );
    }

    /// The lattice cell `classify`'s docstring claims but the composed test
    /// above doesn't exercise directly: the cancel FLAG unset, but the
    /// original message satisfies `classify`'s TEXT fallback
    /// (`"training cancelled: lease lost"`) AND also matches an OOM
    /// spelling. `classify_training_oom` runs first and rewrites the
    /// message — but its rewrite always embeds the original message
    /// verbatim as the trailing `Underlying error: {msg}` clause, so the
    /// text fallback still finds `"training cancelled: lease lost"` inside
    /// the rewritten string, and the result must still be `Cancelled`, not
    /// a `Failed` OOM guidance message that has silently eaten the
    /// cancellation.
    #[test]
    fn classify_training_error_text_fallback_survives_oom_rewrite() {
        let config = FineTuneConfig::default();
        let cancel = AtomicBool::new(false); // flag unset — only the text fallback can apply
        let e = JammiError::FineTune(
            "training cancelled: lease lost (cuda out of memory during unwind)".into(),
        );
        assert!(
            matches!(
                classify_training_error(&cancel, &config, e),
                WorkerJobError::Cancelled
            ),
            "the cancellation text fallback must survive classify_training_oom's rewrite"
        );
    }

    /// End-to-end: a CUDA-OOM-shaped failure from the blocking trainer, on the
    /// default (projection-head) config, drives the job to a terminal
    /// `failed` status whose `error_message` (what `jammi train status` /
    /// Python `job.status()` read) carries the classified guidance — not the
    /// raw driver string, and without the inert `backbone_dtype: bf16`
    /// suggestion the default arm cannot act on. Mirrors
    /// `panicking_training_job_lands_failed_with_recorded_error`'s pipeline,
    /// substituting `classify_training_error` for the plain `classify` call
    /// `train_fine_tune` makes on this arm.
    #[tokio::test(flavor = "multi_thread")]
    async fn oom_training_job_lands_failed_with_classified_guidance() {
        use jammi_db::catalog::status::JobStatus;

        let dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(jammi_db::catalog::Catalog::open(dir.path()).await.unwrap());
        catalog
            .register_model(jammi_db::catalog::model_repo::RegisterModelParams {
                model_id: "oom-base",
                version: 1,
                model_type: "embedding",
                backend: "candle",
                task: ModelTask::TextEmbedding,
                base_model_id: None,
                external_location: None,
                config_json: None,
            })
            .await
            .unwrap();
        catalog
            .submit_job(jammi_db::catalog::jobs_repo::SubmitJobParams {
                job_id: "oom-job",
                kind: "fine_tune",
                execution: jammi_db::catalog::status::JobExecution::Queued,
                spec: "{}",
                model_ref: Some("oom-base::1"),
                output_model_id: None,
                model_source: None,
                priority: 0,
            })
            .await
            .unwrap();

        // The worker claims the job (running, leased to it) before running it —
        // the state in which a genuine failure is recorded under the lease guard.
        let claimed = catalog
            .claim_next("worker-x", &["fine_tune"], Duration::from_secs(60))
            .await
            .unwrap()
            .expect("the queued job is claimable");

        let cancel = Arc::new(AtomicBool::new(false));
        // The engine defaults (batch 8, seq 512, backbone F32, empty
        // target_modules => projection-head arm) that OOM on an L4 24GB card
        // at inference-side defaults.
        let config = FineTuneConfig::default();

        // Run the worker's blocking wrapper over a trainer that raises a raw
        // CUDA driver OOM, then take the same terminal-classification branch
        // `train_fine_tune` does: `classify_training_error`.
        let result = tokio::task::spawn_blocking(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
                Err(JammiError::FineTune(
                    "Encoder forward: cuda error: CUDA_ERROR_OUT_OF_MEMORY: out of memory".into(),
                ))
            }))
        })
        .await;
        let outcome = match result {
            Ok(Ok(Ok(()))) => panic!("the closure was supposed to return an OOM error"),
            Ok(Ok(Err(e))) => classify_training_error(&cancel, &config, e),
            Ok(Err(payload)) => WorkerJobError::Failed(JammiError::FineTune(format!(
                "Panic: {}",
                panic_message(payload.as_ref())
            ))),
            Err(join_err) => WorkerJobError::Failed(JammiError::FineTune(format!(
                "training task join error: {join_err}"
            ))),
        };

        let WorkerJobError::Failed(error) = outcome else {
            panic!("a genuine OOM must classify as Failed, not Cancelled");
        };
        let msg = failed_job_message(&error);
        assert!(
            msg.contains("batch_size=8")
                && msg.contains("backbone_dtype does not apply to projection-head runs")
                && !msg.contains("backbone_dtype: bf16"),
            "the classified OOM guidance must reach the terminal message, without the \
             inert bf16 remedy on the default (projection-head) arm, got: {msg}"
        );

        // The worker records the failure as the job's terminal status, under the
        // lease guard (it still owns the job).
        record_failed(
            LeaseHolder::LoopClaimer,
            &catalog,
            "oom-job",
            "worker-x",
            claimed.attempts,
            msg,
        )
        .await;

        let job = catalog.get_job("oom-job").await.unwrap();
        assert_eq!(
            job.status,
            JobStatus::Failed.to_string(),
            "an OOM'd job lands `failed`, never wedged `running`"
        );
        assert!(
            job.error
                .as_deref()
                .is_some_and(|m| m.contains("batch_size=8")
                    && m.contains("backbone_dtype does not apply to projection-head runs")),
            "`jammi train status` / job.status() must surface the classified OOM \
             guidance from the catalog's error, got {:?}",
            job.error
        );
    }

    /// The panic-payload extractor handles the two common payload shapes
    /// (`&'static str` from `panic!("…")`, `String` from `panic!("{}", x)`) and
    /// falls back for anything else, so the recorded failure is always a
    /// human-readable cause rather than an opaque type id.
    #[test]
    fn panic_message_reads_str_string_and_other_payloads() {
        let s = std::panic::catch_unwind(|| panic!("static message")).unwrap_err();
        assert_eq!(panic_message(s.as_ref()), "static message");

        let owned = std::panic::catch_unwind(|| panic!("{}", "owned".to_string())).unwrap_err();
        assert_eq!(panic_message(owned.as_ref()), "owned");

        let other = std::panic::catch_unwind(|| std::panic::panic_any(42u8)).unwrap_err();
        assert_eq!(panic_message(other.as_ref()), "<unknown panic payload>");
    }

    /// `JAMMI_WORKER_ID` is a LABEL: `worker_label` reads it trimmed when
    /// set and non-empty and `None` otherwise, while `mint_instance_id`
    /// never reads it — two mints differ from each other and from the
    /// label even while the variable is set.
    ///
    /// `JAMMI_WORKER_ID` is process-global, so the cases run in one test
    /// (parallel tests must not race the same env var) and the var is removed
    /// at the end to leave the environment clean for the rest of the suite.
    #[test]
    fn worker_id_env_is_a_label_never_the_minted_identity() {
        std::env::set_var(WORKER_LABEL_ENV, "  gpu-node-7  ");
        assert_eq!(worker_label().as_deref(), Some("gpu-node-7"));
        let a = mint_instance_id();
        let b = mint_instance_id();
        assert_ne!(a, b, "each mint is a fresh id");
        assert_ne!(a, "gpu-node-7", "the label must never become the identity");
        assert!(
            uuid::Uuid::parse_str(&a).is_ok(),
            "the identity is a UUID (the `instances` schema's promise), got {a:?}"
        );

        std::env::set_var(WORKER_LABEL_ENV, "   ");
        assert_eq!(
            worker_label(),
            None,
            "an all-whitespace label labels nothing"
        );

        std::env::remove_var(WORKER_LABEL_ENV);
        assert_eq!(worker_label(), None);
    }

    // ─── Regression detector (public on-ramp) ────────────────────────────────
    //
    // These pin the worker's column→loader detector for the regression
    // `(text, target)` format and the `extract_numeric_column` helper that feeds
    // it. They are the worker-side proof of the public on-ramp: a real
    // `db.fine_tune(task=regression)` reaches the regression loader through
    // exactly this `build_training_data_loader` dispatch. The end-to-end served
    // read (train → Infer) is pinned by the integration suite
    // (`tests/it/regression_surface.rs`).

    use arrow::array::{
        ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch as ArrowBatch,
        StringArray,
    };
    use arrow::datatypes::{DataType, Field, Schema};

    use crate::fine_tune::data::TrainingFormat;

    fn text_target_batch(texts: &[&str], target: ArrayRef) -> ArrowBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("text", DataType::Utf8, true),
            Field::new("target", target.data_type().clone(), true),
        ]));
        let text_arr = Arc::new(StringArray::from(texts.to_vec())) as ArrayRef;
        ArrowBatch::try_new(schema, vec![text_arr, target]).unwrap()
    }

    fn regression_cols() -> Vec<String> {
        vec!["text".into(), "target".into()]
    }

    /// `task=Regression` over a `(text, int64-target)` source builds a
    /// `Regression`-format loader whose targets are the years read as `f32` —
    /// the int64 arxiv-year path, the most common real target type.
    #[test]
    fn detector_builds_regression_loader_from_int64_target() {
        let target = Arc::new(Int64Array::from(vec![2017i64, 2018, 2016])) as ArrayRef;
        let batch = text_target_batch(&["a", "b", "c"], target);
        let loader =
            build_training_data_loader(&[batch], &regression_cols(), ModelTask::Regression)
                .unwrap();
        assert!(matches!(loader.format(), TrainingFormat::Regression));
        assert_eq!(loader.len(), 3);
        assert_eq!(
            loader.regression_targets().unwrap(),
            vec![2017.0, 2018.0, 2016.0]
        );
    }

    /// Float64 and Float32 target columns both reduce to the same `f32` targets —
    /// the extractor's downcast arms are width-agnostic.
    #[test]
    fn detector_reads_float64_and_float32_targets() {
        let f64_batch = text_target_batch(
            &["a", "b"],
            Arc::new(Float64Array::from(vec![1.5f64, 2.5])) as ArrayRef,
        );
        let f32_batch = text_target_batch(
            &["a", "b"],
            Arc::new(Float32Array::from(vec![1.5f32, 2.5])) as ArrayRef,
        );
        for batch in [f64_batch, f32_batch] {
            let loader =
                build_training_data_loader(&[batch], &regression_cols(), ModelTask::Regression)
                    .unwrap();
            assert_eq!(loader.regression_targets().unwrap(), vec![1.5, 2.5]);
        }
    }

    /// Int32 targets are also accepted (a narrower integer column).
    #[test]
    fn detector_reads_int32_target() {
        let target = Arc::new(Int32Array::from(vec![10i32, 20])) as ArrayRef;
        let batch = text_target_batch(&["a", "b"], target);
        let loader =
            build_training_data_loader(&[batch], &regression_cols(), ModelTask::Regression)
                .unwrap();
        assert_eq!(loader.regression_targets().unwrap(), vec![10.0, 20.0]);
    }

    /// THE headline guard: a `(text, label)` source under `task=regression` no
    /// longer falls into the classification path (which would gather a string
    /// outcome as a class index — the confirmed CUDA device-side assert). With
    /// only a `label` column and no `target`, it surfaces a typed regression
    /// error citing the missing numeric `target` column.
    #[test]
    fn task_regression_with_label_column_does_not_route_to_classification() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("text", DataType::Utf8, true),
            Field::new("label", DataType::Utf8, true),
        ]));
        let text = Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef;
        let label = Arc::new(StringArray::from(vec!["2017", "2018"])) as ArrayRef;
        let batch = ArrowBatch::try_new(schema, vec![text, label]).unwrap();
        let cols = vec!["text".to_string(), "label".to_string()];
        let err = build_training_data_loader(&[batch], &cols, ModelTask::Regression)
            .err()
            .unwrap();
        let msg = err.to_string();
        assert!(
            msg.contains("target"),
            "regression routing error must name the missing numeric 'target' column, got: {msg}"
        );
        // And it must NOT have silently produced a classification loader.
        assert!(
            !msg.contains("class"),
            "must not fall through to classification, got: {msg}"
        );
    }

    /// `(text, label)` with `task != Regression` still routes to classification,
    /// unchanged — the regression gate does not regress the existing path.
    #[test]
    fn classification_still_routes_when_task_not_regression() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("text", DataType::Utf8, true),
            Field::new("label", DataType::Utf8, true),
        ]));
        let text = Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef;
        let label = Arc::new(StringArray::from(vec!["x", "y"])) as ArrayRef;
        let batch = ArrowBatch::try_new(schema, vec![text, label]).unwrap();
        let cols = vec!["text".to_string(), "label".to_string()];
        let loader = build_training_data_loader(&[batch], &cols, ModelTask::TextEmbedding).unwrap();
        assert!(matches!(
            loader.format(),
            TrainingFormat::Classification { num_classes: 2 }
        ));
    }

    /// A null target is rejected with a typed error citing the row — never
    /// coerced to `0.0`, which would silently corrupt the scaler's μ/σ.
    #[test]
    fn null_target_is_rejected_with_typed_error() {
        let target = Arc::new(Int64Array::from(vec![Some(2017i64), None, Some(2018)])) as ArrayRef;
        let batch = text_target_batch(&["a", "b", "c"], target);
        let err = build_training_data_loader(&[batch], &regression_cols(), ModelTask::Regression)
            .err()
            .unwrap();
        let msg = err.to_string();
        assert!(
            msg.contains("null") && msg.contains("row 1"),
            "null target must be rejected citing the row, got: {msg}"
        );
    }

    /// A NaN target (float column) is likewise rejected citing the row.
    #[test]
    fn nan_target_is_rejected_with_typed_error() {
        let target = Arc::new(Float64Array::from(vec![1.0f64, f64::NAN, 3.0])) as ArrayRef;
        let batch = text_target_batch(&["a", "b", "c"], target);
        let err = build_training_data_loader(&[batch], &regression_cols(), ModelTask::Regression)
            .err()
            .unwrap();
        let msg = err.to_string();
        assert!(
            msg.contains("NaN") && msg.contains("row 1"),
            "NaN target must be rejected citing the row, got: {msg}"
        );
    }

    /// A non-numeric `target` column (strings that don't parse) is a typed
    /// "not a numeric column" error, not a panic.
    #[test]
    fn non_numeric_target_is_typed_error() {
        let target = Arc::new(StringArray::from(vec!["alpha", "beta"])) as ArrayRef;
        let batch = text_target_batch(&["a", "b"], target);
        let err = build_training_data_loader(&[batch], &regression_cols(), ModelTask::Regression)
            .err()
            .unwrap();
        assert!(
            err.to_string().contains("not a numeric"),
            "non-numeric target must be a typed error, got: {err}"
        );
    }

    /// A constant / single-value target builds a valid loader (σ=0 is floored
    /// downstream by `STD_FLOOR`); the detector itself must not choke on it.
    #[test]
    fn constant_target_builds_loader() {
        let target = Arc::new(Int64Array::from(vec![2017i64, 2017, 2017])) as ArrayRef;
        let batch = text_target_batch(&["a", "b", "c"], target);
        let loader =
            build_training_data_loader(&[batch], &regression_cols(), ModelTask::Regression)
                .unwrap();
        assert_eq!(
            loader.regression_targets().unwrap(),
            vec![2017.0, 2017.0, 2017.0]
        );
    }

    // ─────────────────────────────────────────────────────────────────
    // `build_encoder_adapters` GGUF class-closure oracle: a
    // `build_encoder_adapters` construction test for EACH of the three
    // GGUF-threaded architectures. Reading the layer count off the RAW
    // config.json refuses DistilBERT ("GGUF load requires num_hidden_layers
    // (or num_layers) in config.json"), whose checkpoint declares only the
    // DistilBERT-native `n_layers` field, while BERT and ModernBERT pass
    // (their raw config.json uses `num_hidden_layers`) — so only a
    // per-architecture oracle can catch that class of defect.
    // ─────────────────────────────────────────────────────────────────

    /// FNV-1a-seeded deterministic small-magnitude values (no
    /// unseeded RNG) — independent per-tensor-name value stream without a
    /// hand-maintained counter.
    fn gguf_fixture_tensor(name: &str, dims: &[usize], device: &candle_core::Device) -> Tensor {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in name.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        let seed = h as f64;
        let n: usize = dims.iter().product();
        let v: Vec<f32> = (0..n)
            .map(|i| (((seed % 97.0) + 1.0) * (i as f64) * 0.037 + seed * 1e-6).sin() as f32 * 0.1)
            .collect();
        Tensor::from_vec(v, dims, device).unwrap()
    }

    /// Write `dir/model.gguf`: every tensor whose name is in
    /// `matmul_weight_names` is quantized at `quant`; every other tensor
    /// (embeddings, LayerNorms, matmul-site biases) is written as an
    /// `F32`-"quantized" `QTensor` — GGUF's own convention for a dense-
    /// stored tensor. Mirrors `tests/it/gguf_qlora.rs`'s
    /// `write_gguf_checkpoint`, independently re-derived here because
    /// `build_encoder_adapters` is module-private and unreachable from
    /// that external integration-test crate.
    fn write_gguf_fixture(
        dir: &std::path::Path,
        tensors: &HashMap<String, Tensor>,
        matmul_weight_names: &[String],
        quant: candle_core::quantized::GgmlDType,
    ) {
        use candle_core::quantized::{gguf_file, QTensor};
        std::fs::create_dir_all(dir).unwrap();
        let mut names: Vec<&String> = tensors.keys().collect();
        names.sort(); // deterministic write order
        let mut qtensors: Vec<(String, QTensor)> = Vec::with_capacity(names.len());
        for name in names {
            let t = &tensors[name];
            let dtype = if matmul_weight_names.iter().any(|n| n == name) {
                quant
            } else {
                candle_core::quantized::GgmlDType::F32
            };
            qtensors.push((name.clone(), QTensor::quantize(t, dtype).unwrap()));
        }
        let file = std::fs::File::create(dir.join("model.gguf")).unwrap();
        let mut writer = std::io::BufWriter::new(file);
        let refs: Vec<(&str, &QTensor)> = qtensors.iter().map(|(n, q)| (n.as_str(), q)).collect();
        gguf_file::write(&mut writer, &[], &refs).unwrap();
    }

    const GGUF_FIXTURE_HIDDEN: usize = 32;
    const GGUF_FIXTURE_LAYERS: usize = 1;
    const GGUF_FIXTURE_HEADS: usize = 2;
    const GGUF_FIXTURE_INTERMEDIATE: usize = 64;
    const GGUF_FIXTURE_VOCAB: usize = 64;
    const GGUF_FIXTURE_MAX_POS: usize = 32;
    const GGUF_FIXTURE_TYPE_VOCAB: usize = 2;

    /// A raw (no `"bert."` wrapper) BERT-family fixture: tensors, config,
    /// and the fully-qualified matmul-site `.weight` names — mirrors
    /// `jammi_ai::model::backend::gguf::matmul_site_names`'s `Bert` arm.
    fn bert_gguf_fixture(
        device: &candle_core::Device,
    ) -> (HashMap<String, Tensor>, serde_json::Value, Vec<String>) {
        let (hidden, layers, heads, intermediate, vocab, max_pos, type_vocab) = (
            GGUF_FIXTURE_HIDDEN,
            GGUF_FIXTURE_LAYERS,
            GGUF_FIXTURE_HEADS,
            GGUF_FIXTURE_INTERMEDIATE,
            GGUF_FIXTURE_VOCAB,
            GGUF_FIXTURE_MAX_POS,
            GGUF_FIXTURE_TYPE_VOCAB,
        );
        let mut map = HashMap::new();
        let add = |map: &mut HashMap<String, Tensor>, name: String, dims: &[usize]| {
            let t = gguf_fixture_tensor(&name, dims, device);
            map.insert(name, t);
        };
        add(
            &mut map,
            "embeddings.word_embeddings.weight".into(),
            &[vocab, hidden],
        );
        add(
            &mut map,
            "embeddings.position_embeddings.weight".into(),
            &[max_pos, hidden],
        );
        add(
            &mut map,
            "embeddings.token_type_embeddings.weight".into(),
            &[type_vocab, hidden],
        );
        add(&mut map, "embeddings.LayerNorm.weight".into(), &[hidden]);
        add(&mut map, "embeddings.LayerNorm.bias".into(), &[hidden]);
        let mut matmul_weights = Vec::new();
        for n in 0..layers {
            let p = format!("encoder.layer.{n}");
            for site in [
                "attention.self.query",
                "attention.self.key",
                "attention.self.value",
                "attention.output.dense",
            ] {
                let w = format!("{p}.{site}.weight");
                add(&mut map, w.clone(), &[hidden, hidden]);
                matmul_weights.push(w);
                add(&mut map, format!("{p}.{site}.bias"), &[hidden]);
            }
            let w = format!("{p}.intermediate.dense.weight");
            add(&mut map, w.clone(), &[intermediate, hidden]);
            matmul_weights.push(w);
            add(
                &mut map,
                format!("{p}.intermediate.dense.bias"),
                &[intermediate],
            );
            let w = format!("{p}.output.dense.weight");
            add(&mut map, w.clone(), &[hidden, intermediate]);
            matmul_weights.push(w);
            add(&mut map, format!("{p}.output.dense.bias"), &[hidden]);
            for ln in ["attention.output.LayerNorm", "output.LayerNorm"] {
                add(&mut map, format!("{p}.{ln}.weight"), &[hidden]);
                add(&mut map, format!("{p}.{ln}.bias"), &[hidden]);
            }
        }
        let config = serde_json::json!({
            "model_type": "bert",
            "hidden_size": hidden,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "intermediate_size": intermediate,
            "vocab_size": vocab,
            "max_position_embeddings": max_pos,
            "type_vocab_size": type_vocab,
            "layer_norm_eps": 1e-12,
        });
        (map, config, matmul_weights)
    }

    /// A DistilBERT fixture, config.json spelled with the DistilBERT-
    /// native field names (`dim`/`n_layers`/`n_heads`/`hidden_dim`) — the
    /// RAW shape a real DistilBERT checkpoint ships, on purpose: this is
    /// exactly the config `gguf_num_layers`'s normalization step must
    /// handle for the layer-count extraction to succeed at all.
    fn distilbert_gguf_fixture(
        device: &candle_core::Device,
    ) -> (HashMap<String, Tensor>, serde_json::Value, Vec<String>) {
        let (hidden, layers, heads, intermediate, vocab, max_pos) = (
            GGUF_FIXTURE_HIDDEN,
            GGUF_FIXTURE_LAYERS,
            GGUF_FIXTURE_HEADS,
            GGUF_FIXTURE_INTERMEDIATE,
            GGUF_FIXTURE_VOCAB,
            GGUF_FIXTURE_MAX_POS,
        );
        let mut map = HashMap::new();
        let add = |map: &mut HashMap<String, Tensor>, name: String, dims: &[usize]| {
            let t = gguf_fixture_tensor(&name, dims, device);
            map.insert(name, t);
        };
        add(
            &mut map,
            "distilbert.embeddings.word_embeddings.weight".into(),
            &[vocab, hidden],
        );
        add(
            &mut map,
            "distilbert.embeddings.position_embeddings.weight".into(),
            &[max_pos, hidden],
        );
        add(
            &mut map,
            "distilbert.embeddings.LayerNorm.weight".into(),
            &[hidden],
        );
        add(
            &mut map,
            "distilbert.embeddings.LayerNorm.bias".into(),
            &[hidden],
        );
        let mut matmul_weights = Vec::new();
        for n in 0..layers {
            let p = format!("distilbert.transformer.layer.{n}");
            for site in [
                "attention.q_lin",
                "attention.k_lin",
                "attention.v_lin",
                "attention.out_lin",
            ] {
                let w = format!("{p}.{site}.weight");
                add(&mut map, w.clone(), &[hidden, hidden]);
                matmul_weights.push(w);
                add(&mut map, format!("{p}.{site}.bias"), &[hidden]);
            }
            add(&mut map, format!("{p}.sa_layer_norm.weight"), &[hidden]);
            add(&mut map, format!("{p}.sa_layer_norm.bias"), &[hidden]);
            let w = format!("{p}.ffn.lin1.weight");
            add(&mut map, w.clone(), &[intermediate, hidden]);
            matmul_weights.push(w);
            add(&mut map, format!("{p}.ffn.lin1.bias"), &[intermediate]);
            let w = format!("{p}.ffn.lin2.weight");
            add(&mut map, w.clone(), &[hidden, intermediate]);
            matmul_weights.push(w);
            add(&mut map, format!("{p}.ffn.lin2.bias"), &[hidden]);
            add(&mut map, format!("{p}.output_layer_norm.weight"), &[hidden]);
            add(&mut map, format!("{p}.output_layer_norm.bias"), &[hidden]);
        }
        let config = serde_json::json!({
            "model_type": "distilbert",
            "dim": hidden,
            "n_layers": layers,
            "n_heads": heads,
            "hidden_dim": intermediate,
            "vocab_size": vocab,
            "max_position_embeddings": max_pos,
        });
        (map, config, matmul_weights)
    }

    /// A ModernBERT fixture: bias-free matmul sites and LayerNorms
    /// (`gguf::matmul_site_names`'s `ModernBert` arm), a single layer so
    /// `attn_norm` (skipped at layer 0 — `ModernBertBuilder::build`'s own
    /// `if n == 0 { None }`) never needs a tensor.
    fn modernbert_gguf_fixture(
        device: &candle_core::Device,
    ) -> (HashMap<String, Tensor>, serde_json::Value, Vec<String>) {
        let (hidden, layers, heads, intermediate, vocab, max_pos) = (
            GGUF_FIXTURE_HIDDEN,
            GGUF_FIXTURE_LAYERS,
            GGUF_FIXTURE_HEADS,
            GGUF_FIXTURE_INTERMEDIATE,
            GGUF_FIXTURE_VOCAB,
            GGUF_FIXTURE_MAX_POS,
        );
        let mut map = HashMap::new();
        let add = |map: &mut HashMap<String, Tensor>, name: String, dims: &[usize]| {
            let t = gguf_fixture_tensor(&name, dims, device);
            map.insert(name, t);
        };
        add(
            &mut map,
            "model.embeddings.tok_embeddings.weight".into(),
            &[vocab, hidden],
        );
        add(&mut map, "model.embeddings.norm.weight".into(), &[hidden]);
        let mut matmul_weights = Vec::new();
        for n in 0..layers {
            let p = format!("model.layers.{n}");
            let w = format!("{p}.attn.Wqkv.weight");
            add(&mut map, w.clone(), &[hidden * 3, hidden]);
            matmul_weights.push(w);
            let w = format!("{p}.attn.Wo.weight");
            add(&mut map, w.clone(), &[hidden, hidden]);
            matmul_weights.push(w);
            let w = format!("{p}.mlp.Wi.weight");
            add(&mut map, w.clone(), &[intermediate * 2, hidden]);
            matmul_weights.push(w);
            let w = format!("{p}.mlp.Wo.weight");
            add(&mut map, w.clone(), &[hidden, intermediate]);
            matmul_weights.push(w);
            add(&mut map, format!("{p}.mlp_norm.weight"), &[hidden]);
        }
        add(&mut map, "model.final_norm.weight".into(), &[hidden]);
        let config = serde_json::json!({
            "model_type": "modernbert",
            "hidden_size": hidden,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "intermediate_size": intermediate,
            "vocab_size": vocab,
            "max_position_embeddings": max_pos,
        });
        (map, config, matmul_weights)
    }

    fn gguf_test_artifact_store() -> Arc<ArtifactStore> {
        let cache_dir = tempfile::tempdir().unwrap().keep();
        Arc::new(
            ArtifactStore::with_root(
                jammi_db::storage::StorageUrl::memory("worker-gguf-test-artifacts"),
                jammi_db::storage::StorageRegistry::new(),
                cache_dir,
            )
            .unwrap(),
        )
    }

    /// A [`HubSource`] rooted at a fresh tempdir, for the `build_encoder_adapters`
    /// fixtures below whose base model is always locally registered
    /// (`artifact_path` set) — the `is_hf` HF-fallback arm this threads
    /// through never runs for them, so no network/mock setup is needed, just
    /// a value of the right type.
    fn test_hub_source() -> HubSource {
        let root = tempfile::tempdir().unwrap().keep();
        HubSource::from_config(
            &jammi_db::config::ModelsConfig {
                hub_cache_dir: Some(root),
                ..Default::default()
            },
            &|_: &str| None,
        )
        .unwrap()
    }

    /// `build_encoder_adapters`'s HF-fallback arm (reached
    /// when a fine-tune base model's catalog row carries no
    /// `artifact_path`) calls `hub.api().model(..).get(..)` exactly like
    /// the resolver's own `HuggingFace` arm, so `[models] offline = true`
    /// must refuse it identically — never fall through to a live network
    /// fetch offline is supposed to forbid. No mock server is configured
    /// here at all, so an arm that ignored `offline` would attempt a real
    /// Hub network call and fail with a connection/DNS error (the wrong
    /// failure mode), not this typed offline refusal.
    #[tokio::test(flavor = "multi_thread")]
    async fn build_encoder_adapters_hf_fallback_refuses_when_offline() {
        let catalog_dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
        let base_model_id = "acme/never-resolved-base";
        catalog
            .register_model(jammi_db::catalog::model_repo::RegisterModelParams {
                model_id: base_model_id,
                version: 1,
                model_type: "embedding",
                backend: "candle",
                task: ModelTask::TextEmbedding,
                base_model_id: None,
                external_location: None,
                config_json: None,
            })
            .await
            .unwrap();
        let artifact_store = gguf_test_artifact_store();

        let root = tempfile::tempdir().unwrap().keep();
        let offline_hub = HubSource::from_config(
            &jammi_db::config::ModelsConfig {
                hub_cache_dir: Some(root),
                offline: Some(true),
                ..Default::default()
            },
            &|_: &str| None,
        )
        .unwrap();

        let owned_base_model_id = base_model_id.to_string();
        let result = tokio::task::spawn_blocking(move || {
            let training_config = FineTuneConfig::default();
            let varmap = candle_nn::VarMap::new();
            let device = candle_core::Device::Cpu;
            build_encoder_adapters(BuildEncoderAdaptersParams {
                base_model_id: &owned_base_model_id,
                catalog: &catalog,
                artifact_store: &artifact_store,
                config: &training_config,
                dropout_seed: training_config.seed,
                task: ModelTask::TextEmbedding,
                varmap: &varmap,
                device: &device,
                hub: &offline_hub,
            })
        })
        .await
        .unwrap();
        let err = match result {
            Ok(_) => panic!("offline must refuse the HF fallback, not attempt a network fetch"),
            Err(e) => e,
        };

        match err {
            JammiError::Model { model_id, message } => {
                assert_eq!(model_id, base_model_id);
                assert!(
                    message.contains("offline") && message.contains(base_model_id),
                    "expected the offline refusal to name the repo id, got: {message}"
                );
            }
            other => panic!("expected JammiError::Model, got {other:?}"),
        }
    }

    /// Register `model_id` in `catalog` with its external location pointing at
    /// `dir` (a `file://`-scheme local directory — `StorageUrl::parse`
    /// normalizes a bare absolute path to `file://...`), and return the
    /// exact `base_model_id` string `build_encoder_adapters` expects
    /// (`ModelSource::parse` maps an absolute path straight through to
    /// `Local(path)`, so the catalog key IS the path string).
    async fn register_gguf_base_model(catalog: &Arc<Catalog>, dir: &std::path::Path) -> String {
        let base_model_id = dir.to_str().unwrap().to_string();
        catalog
            .register_model(jammi_db::catalog::model_repo::RegisterModelParams {
                model_id: &base_model_id,
                version: 1,
                model_type: "embedding",
                backend: "candle",
                task: ModelTask::TextEmbedding,
                base_model_id: None,
                external_location: Some(dir.to_str().unwrap()),
                config_json: None,
            })
            .await
            .unwrap();
        base_model_id
    }

    /// Drives `build_encoder_adapters` for one architecture's GGUF fixture
    /// through the SAME `spawn_blocking` shape production code runs it
    /// under (`build_encoder_adapters` itself calls
    /// `tokio::runtime::Handle::current().block_on(..)` for its catalog
    /// reads, which panics if invoked directly on a runtime worker
    /// thread).
    async fn build_encoder_adapters_gguf(
        arch_model_type: &str,
        tensors: HashMap<String, Tensor>,
        config: serde_json::Value,
        matmul_weights: Vec<String>,
        target_modules: Vec<String>,
    ) -> Result<(jammi_encoders::AnyEncoder, jammi_lora::AdapterConfig)> {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join(arch_model_type);
        write_gguf_fixture(
            &dir,
            &tensors,
            &matmul_weights,
            candle_core::quantized::GgmlDType::Q8_0,
        );
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();

        let catalog_dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
        let base_model_id = register_gguf_base_model(&catalog, &dir).await;
        let artifact_store = gguf_test_artifact_store();

        let hub = test_hub_source();
        tokio::task::spawn_blocking(move || {
            let training_config = FineTuneConfig {
                target_modules,
                ..FineTuneConfig::default()
            };
            let varmap = candle_nn::VarMap::new();
            let device = candle_core::Device::Cpu;
            build_encoder_adapters(BuildEncoderAdaptersParams {
                base_model_id: &base_model_id,
                catalog: &catalog,
                artifact_store: &artifact_store,
                config: &training_config,
                dropout_seed: training_config.seed,
                task: ModelTask::TextEmbedding,
                varmap: &varmap,
                device: &device,
                hub: &hub,
            })
        })
        .await
        .unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn build_encoder_adapters_gguf_bert_succeeds() {
        let device = candle_core::Device::Cpu;
        let (tensors, config, matmul_weights) = bert_gguf_fixture(&device);
        let (encoder, adapter_cfg) = build_encoder_adapters_gguf(
            "bert",
            tensors,
            config,
            matmul_weights,
            vec!["query".into(), "value".into()],
        )
        .await
        .unwrap();
        assert!(
            matches!(encoder, jammi_encoders::AnyEncoder::Bert(_)),
            "expected a Bert encoder"
        );
        assert_eq!(adapter_cfg.model_type, "bert");
    }

    /// The raw config.json this fixture writes carries only `n_layers`,
    /// DistilBERT's native field name, so reading
    /// `num_hidden_layers`/`num_layers` directly off it always misses and
    /// fails with "GGUF load requires num_hidden_layers (or num_layers) in
    /// config.json". `build_encoder_adapters` routes through
    /// `gguf::gguf_num_layers`, which normalizes first.
    #[tokio::test(flavor = "multi_thread")]
    async fn build_encoder_adapters_gguf_distilbert_succeeds() {
        let device = candle_core::Device::Cpu;
        let (tensors, config, matmul_weights) = distilbert_gguf_fixture(&device);
        let (encoder, adapter_cfg) = build_encoder_adapters_gguf(
            "distilbert",
            tensors,
            config,
            matmul_weights,
            vec!["q_lin".into(), "v_lin".into()],
        )
        .await
        .unwrap();
        assert!(
            matches!(encoder, jammi_encoders::AnyEncoder::DistilBert(_)),
            "expected a DistilBert encoder"
        );
        assert_eq!(adapter_cfg.model_type, "distilbert");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn build_encoder_adapters_gguf_modernbert_succeeds() {
        let device = candle_core::Device::Cpu;
        let (tensors, config, matmul_weights) = modernbert_gguf_fixture(&device);
        let (encoder, adapter_cfg) = build_encoder_adapters_gguf(
            "modernbert",
            tensors,
            config,
            matmul_weights,
            vec!["Wqkv".into()],
        )
        .await
        .unwrap();
        assert!(
            matches!(encoder, jammi_encoders::AnyEncoder::ModernBert(_)),
            "expected a ModernBert encoder"
        );
        assert_eq!(adapter_cfg.model_type, "modernbert");
    }

    /// Drives `build_encoder_adapters` for a PRE-POPULATED base-model `dir`
    /// (config.json + weights already written by the caller) through the
    /// SAME `spawn_blocking` shape production code runs it under
    /// (`build_encoder_adapters` itself calls
    /// `tokio::runtime::Handle::current().block_on(..)` for its catalog
    /// reads, which panics if invoked directly on a runtime worker
    /// thread). `lora_dropout: 0.0` is pinned as a SIMPLIFICATION, not a
    /// flakiness necessity: `DropoutMasks` is a per-instance forward
    /// counter starting at 0, keyed by `(run_seed, layer_id, forward_idx)`
    /// through a counter-based Philox stream, and each phase below builds
    /// a fresh encoder and takes exactly one forward, so the comparison
    /// would stay bit-identical even at the default `lora_dropout` of
    /// 0.05 — pinning 0.0 just removes the dropout term from the
    /// comparison entirely. The `lora_dropout > 0` arm of
    /// `build_encoder_adapters` itself is covered by
    /// `build_encoder_adapters_gguf_bert_succeeds`,
    /// `build_encoder_adapters_gguf_distilbert_succeeds`, and
    /// `build_encoder_adapters_gguf_modernbert_succeeds` above, which all
    /// use `FineTuneConfig::default()`.
    async fn build_encoder_adapters_for_dir(
        dir: &std::path::Path,
        target_modules: Vec<String>,
    ) -> Result<(jammi_encoders::AnyEncoder, jammi_lora::AdapterConfig)> {
        let catalog_dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
        let base_model_id = register_gguf_base_model(&catalog, dir).await;
        let artifact_store = gguf_test_artifact_store();
        let hub = test_hub_source();

        tokio::task::spawn_blocking(move || {
            let training_config = FineTuneConfig {
                target_modules,
                lora_dropout: 0.0,
                ..FineTuneConfig::default()
            };
            let varmap = candle_nn::VarMap::new();
            let device = candle_core::Device::Cpu;
            build_encoder_adapters(BuildEncoderAdaptersParams {
                base_model_id: &base_model_id,
                catalog: &catalog,
                artifact_store: &artifact_store,
                config: &training_config,
                dropout_seed: training_config.seed,
                task: ModelTask::TextEmbedding,
                varmap: &varmap,
                device: &device,
                hub: &hub,
            })
        })
        .await
        .unwrap()
    }

    /// Deterministic additive-offset perturbation (no unseeded RNG) — every tensor is shifted by a fixed nonzero offset, keeping
    /// its shape and dtype intact. A `model.gguf` written from this map
    /// can NEVER forward-match a checkpoint built from the unperturbed
    /// originals, so any equality between the two proves the gguf bytes
    /// never reached the loaded weights.
    fn perturb_tensors(tensors: &HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        tensors
            .iter()
            .map(|(name, t)| (name.clone(), t.affine(1.0, 10.0).unwrap()))
            .collect()
    }

    /// One fixed `[1, 5]` token-id/mask forward through
    /// `AnyEncoder::forward`, flattened to `Vec<f32>` — the shared
    /// discriminator each phase of
    /// [`build_encoder_adapters_prefers_safetensors_over_gguf_when_both_present`]
    /// compares against the reference. Token ids stay within the fixture
    /// vocabulary (`GGUF_FIXTURE_VOCAB`).
    fn deterministic_bert_forward(
        encoder: &jammi_encoders::AnyEncoder,
        device: &candle_core::Device,
    ) -> Vec<f32> {
        let input_ids = Tensor::from_vec(vec![3u32, 7, 1, 9, 2], (1, 5), device).unwrap();
        let mask = Tensor::from_vec(vec![1u32, 1, 1, 1, 1], (1, 5), device).unwrap();
        encoder
            .forward(&input_ids, &mask)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// The set of assertions both phases of
    /// [`build_encoder_adapters_prefers_safetensors_over_gguf_when_both_present`]
    /// re-check identically — factored out so the two call sites can
    /// never silently drift apart: build must be `Ok`, the encoder must
    /// be `AnyEncoder::Bert`, `adapter_cfg.model_type` must be `"bert"`,
    /// the forward must be non-empty (an equality against empty proves
    /// nothing), and the forward must be BIT-EXACT equal to
    /// `reference_forward` — any deviation means `model.gguf`'s bytes
    /// leaked into the loaded weights.
    async fn assert_dual_format_build_phase(
        dir: &std::path::Path,
        target_modules: Vec<String>,
        device: &candle_core::Device,
        reference_forward: &[f32],
        phase: &str,
    ) {
        let (encoder, adapter_cfg) = build_encoder_adapters_for_dir(dir, target_modules)
            .await
            .unwrap_or_else(|e| {
                panic!("phase {phase}: build_encoder_adapters must succeed, got: {e}")
            });
        assert!(
            matches!(encoder, jammi_encoders::AnyEncoder::Bert(_)),
            "phase {phase}: expected a Bert encoder built from the safetensors arm"
        );
        assert_eq!(
            adapter_cfg.model_type, "bert",
            "phase {phase}: adapter_cfg.model_type"
        );
        let forward = deterministic_bert_forward(&encoder, device);
        assert!(
            !forward.is_empty(),
            "phase {phase}: the forward must be non-empty — an equality against \
             empty proves nothing"
        );
        assert_eq!(
            forward, reference_forward,
            "phase {phase}: the dual-format directory's forward must EQUAL the \
             safetensors-only reference's forward bit-for-bit — any deviation \
             means model.gguf's bytes leaked into the loaded weights"
        );
    }

    /// Dual-format precedence sweep. Corrupting `model.gguf` BEFORE ever
    /// calling `build_encoder_adapters` would leave only Ok-vs-Err to
    /// discriminate the two arms — a resolver that "tries gguf, falls back
    /// to safetensors on a gguf READ failure" (a `.ok()`-keyed fallback,
    /// not the frozen PRESENCE-keyed precedence) would ALSO fail to read
    /// the already-corrupt file and fall back to safetensors, passing
    /// identically to the correct implementation.
    ///
    /// So this test builds a safetensors-only REFERENCE encoder
    /// first (its own dir, the SAME unperturbed fixture tensors) and
    /// captures its forward, then runs the SAME assertions in two
    /// phases against ONE dual-format dir:
    ///
    /// - Phase 1 (presence-precedence): `model.safetensors` is valid and
    ///   `model.gguf` is ALSO valid, but written from PERTURBED tensors.
    ///   A presence-keyed build (the correct, frozen behavior) picks
    ///   safetensors here regardless of whether `model.gguf` is
    ///   readable, so its forward matches `reference_forward`
    ///   bit-for-bit. A read-keyed-fallback build would instead read the
    ///   valid-but-perturbed `model.gguf` successfully and its forward
    ///   would DEVIATE from `reference_forward` — the discriminator a
    ///   corrupt-before-build test lacks — and it
    ///   covers the dense-`FrozenBase` claim mechanistically:
    ///   quantized-base substitution changes the forward, not merely the
    ///   Ok/Err outcome.
    /// - Phase 2 (no-read): `model.gguf` is corrupted only AFTER phase 1
    ///   has already resolved successfully, and the identical assertions
    ///   are re-checked. This pins that the format decision never
    ///   depends on `model.gguf` being readable at all (valid-perturbed
    ///   or corrupt), i.e. that its bytes are genuinely never opened for
    ///   the decision.
    #[tokio::test(flavor = "multi_thread")]
    async fn build_encoder_adapters_prefers_safetensors_over_gguf_when_both_present() {
        let device = candle_core::Device::Cpu;
        let (tensors, config, matmul_weights) = bert_gguf_fixture(&device);
        let target_modules = vec!["query".to_string(), "value".to_string()];
        let tmp = tempfile::tempdir().unwrap();

        // The safetensors-ONLY reference: same (unperturbed) tensors, no
        // gguf sibling at all.
        let ref_dir = tmp.path().join("reference_bert");
        std::fs::create_dir_all(&ref_dir).unwrap();
        candle_core::safetensors::save(&tensors, ref_dir.join("model.safetensors")).unwrap();
        std::fs::write(
            ref_dir.join("config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();
        let (reference_encoder, _) =
            build_encoder_adapters_for_dir(&ref_dir, target_modules.clone())
                .await
                .unwrap();
        let reference_forward = deterministic_bert_forward(&reference_encoder, &device);

        // The dual-format directory under test — safetensors starts (and
        // stays) valid throughout both phases.
        let dir = tmp.path().join("dual_format_bert");
        std::fs::create_dir_all(&dir).unwrap();
        candle_core::safetensors::save(&tensors, dir.join("model.safetensors")).unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_string(&config).unwrap(),
        )
        .unwrap();

        // Phase 1: model.gguf is VALID but built from PERTURBED tensors —
        // if the gguf arm were ever (wrongly) taken, the forward would
        // deviate from `reference_forward`.
        let perturbed = perturb_tensors(&tensors);
        write_gguf_fixture(
            &dir,
            &perturbed,
            &matmul_weights,
            candle_core::quantized::GgmlDType::Q8_0,
        );
        assert_dual_format_build_phase(
            &dir,
            target_modules.clone(),
            &device,
            &reference_forward,
            "1 (gguf valid, perturbed)",
        )
        .await;

        // Corrupt the (should-be-ignored) GGUF sibling AFTER phase 1 has
        // already resolved successfully — proves phase 1's result wasn't
        // merely a byproduct of a since-corrupted file.
        std::fs::write(dir.join("model.gguf"), b"not a real gguf file").unwrap();

        // Phase 2 (no-read): re-build against the now-corrupted
        // model.gguf; the identical assertions must still hold, proving
        // the format decision never depended on being able to read
        // model.gguf's bytes.
        assert_dual_format_build_phase(
            &dir,
            target_modules,
            &device,
            &reference_forward,
            "2 (gguf corrupted)",
        )
        .await;
    }

    /// `EmbeddedWorker::stop_and_join` actually AWAITS the loop task rather
    /// than merely signalling it (`Drop`'s non-blocking shape) — the graceful
    /// primitive a deterministic `Database::close()` needs. Drives a real
    /// spawned worker with no job ever submitted (the idle path, so the loop
    /// should notice `stop` and return well inside the default 1s idle-poll
    /// window rather than hang), then checks it is safe to call twice.
    #[tokio::test(flavor = "multi_thread")]
    async fn stop_and_join_actually_awaits_the_loop_task() {
        let dir = tempfile::tempdir().unwrap();
        let config = jammi_test_utils::test_config(dir.path());
        let session = Arc::new(crate::session::InferenceSession::new(config).await.unwrap());
        let worker = EmbeddedWorker::spawn(&session).unwrap();

        tokio::time::timeout(Duration::from_secs(10), worker.stop_and_join())
            .await
            .expect("stop_and_join must not hang on an idle worker")
            .expect("an idle worker's loop task must join cleanly, not error");

        // The task is gone: this exercises `Drop`'s no-op-when-already-taken
        // arm directly (via the field it shares with `stop_and_join`) rather
        // than only trusting that dropping `worker` at scope-end never panics.
        assert!(
            matches!(*worker.handle.lock().unwrap(), LoopTask::Joined),
            "stop_and_join must leave the slot Joined so a later Drop finds nothing to abort"
        );

        // Idempotent: a second call finds no handle left and returns
        // immediately rather than blocking or erroring.
        tokio::time::timeout(Duration::from_secs(5), worker.stop_and_join())
            .await
            .expect("a second stop_and_join must not hang")
            .expect("a second stop_and_join on an already-joined worker is Ok");
    }

    /// [`reclaim_unpublished_artifacts`] is the one sweep: it reclaims
    /// exactly what the attempt staged and did not publish. Driven through a
    /// real job and a real finalize — the output and the retained checkpoint
    /// publish, the unretained checkpoint stays the attempt's own — the sweep
    /// removes the unretained bundle and its row and leaves every published
    /// byte loadable.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_sweep_reclaims_what_the_attempt_staged_and_did_not_publish() {
        use jammi_db::catalog::jobs_repo::{FinishJobWithModelParams, SubmitJobParams};
        use jammi_db::catalog::status::JobExecution;

        let dir = tempfile::tempdir().unwrap();
        let config = jammi_test_utils::test_config(dir.path());
        let session = Arc::new(crate::session::InferenceSession::new(config).await.unwrap());
        let store = session.artifact_store();
        let catalog = session.catalog_arc();
        let worker = "sweep-worker";

        catalog
            .register_model(jammi_db::catalog::model_repo::RegisterModelParams {
                model_id: "sweep-base",
                version: 1,
                model_type: "embedding",
                backend: "candle",
                task: ModelTask::TextEmbedding,
                base_model_id: None,
                external_location: None,
                config_json: None,
            })
            .await
            .unwrap();
        let job_id = uuid::Uuid::new_v4().to_string();
        let name = format!("jammi:fine-tuned:{job_id}");
        catalog
            .submit_job(SubmitJobParams {
                job_id: &job_id,
                kind: "fine_tune",
                execution: JobExecution::Queued,
                spec: "{}",
                model_ref: Some("sweep-base::1"),
                output_model_id: Some(&name),
                model_source: None,
                priority: 0,
            })
            .await
            .unwrap();
        let attempt = catalog
            .claim_next(worker, &["fine_tune"], Duration::from_secs(3600))
            .await
            .unwrap()
            .expect("the queued job is claimable")
            .attempts;

        let bundle = |tag: &str| {
            vec![(
                "adapter.safetensors".to_string(),
                Bytes::from(format!("weights:{tag}")),
            )]
        };
        let output = store
            .stage_attempt_artifact(catalog, &job_id, worker, attempt, &bundle("output"))
            .await
            .unwrap();
        let two = std::num::NonZeroUsize::new(2).unwrap();
        let unretained = store
            .stage_checkpoint(catalog, &job_id, attempt, 0, two, &bundle("epoch_0"))
            .await
            .unwrap();
        let retained = store
            .stage_checkpoint(catalog, &job_id, attempt, 1, two, &bundle("epoch_1"))
            .await
            .unwrap();
        let published = [output.artifact().clone(), retained.artifact().clone()];
        let unpublished = unretained.artifact().clone();
        let registration = ModelRegistration {
            model_id: name.clone(),
            version: 1,
            model_type: "fine-tuned",
            task: ModelTask::TextEmbedding,
            base_model_id: Some("sweep-base".to_string()),
            config_json: None,
        };
        let retained_name = format!("{name}:epoch_1");
        assert!(catalog
            .finish_job_with_model(FinishJobWithModelParams {
                job_id: &job_id,
                instance_id: worker,
                attempts: attempt,
                result: "{}",
                output: jammi_db::catalog::jobs_repo::ProducedModel {
                    row: registration.row(&name),
                    artifact: output,
                    materialization: None,
                },
                epoch_checkpoints: vec![jammi_db::catalog::jobs_repo::ProducedModel {
                    row: registration.row(&retained_name),
                    artifact: retained,
                    materialization: None,
                }],
            })
            .await
            .unwrap());

        reclaim_unpublished_artifacts(&store, catalog, &job_id, worker, attempt).await;
        store
            .fetch_artifact(unpublished.url())
            .await
            .expect("the attempt's sweep never reaches the job's own checkpoints");
        reclaim_checkpoints(&store, catalog, &job_id).await;

        assert!(
            store.fetch_artifact(unpublished.url()).await.is_err(),
            "an unpublished epoch checkpoint must be reclaimed once the job is terminal"
        );
        assert!(catalog
            .get_model_artifact(&unpublished)
            .await
            .unwrap()
            .is_none());
        for artifact in &published {
            store
                .fetch_artifact(artifact.url())
                .await
                .expect("a published bundle must survive the sweep");
        }
        assert!(catalog
            .staged_artifacts_of_attempt(&job_id, attempt)
            .await
            .unwrap()
            .is_empty());
    }

    /// `confirms_release()` checks
    /// totality (`released + not_required + failed == attempted`), not just
    /// `failed == 0` — a `HoldRelease` whose three counts undercount its own
    /// `attempted` (a hold silently dropped without being counted at all)
    /// must degrade the release even though `failed` reads zero. The
    /// production pass's own `assert_eq!` should already refuse to hand out
    /// such a value, but a consumer of the type must not simply trust that.
    #[test]
    fn confirms_release_catches_an_undercounted_attempted() {
        let undercounted = HoldReleaseOutcome::Observed(HoldRelease {
            released: 1,
            not_required: 0,
            failed: 0,
            attempted: 2,
        });
        assert!(
            !undercounted.confirms_release(),
            "one of the two attempted holds is unaccounted for; this must not confirm release"
        );

        let consistent = HoldReleaseOutcome::Observed(HoldRelease {
            released: 1,
            not_required: 1,
            failed: 0,
            attempted: 2,
        });
        assert!(
            consistent.confirms_release(),
            "every attempted hold is accounted for and none failed"
        );
    }

    fn cell() -> Arc<HostAdmission> {
        HostAdmission::new(Arc::new(InstanceRegistration::new(
            "awaiting-cell",
            None,
            None,
            None,
            None,
        )))
    }

    /// "Finish what's running, refuse what's new", for every entry: a host
    /// that has begun a DRAIN (or a RELEASE) admits no new claim through
    /// `probe_claim` — the loop's gate AND the placed-gang runner's
    /// admission are this one predicate. Mutation: drop the phase check at
    /// the top of `probe_claim` and the `Draining` assertion reds (the
    /// holder cell is `Free`, so the CAS alone would admit).
    #[tokio::test]
    async fn probe_claim_refuses_once_a_drain_or_release_has_begun() {
        let cell = cell();
        assert!(cell.probe_claim().is_some(), "Running admits");
        assert!(cell.begin_drain());
        assert!(cell.probe_claim().is_none(), "Draining refuses a new claim");
        assert_eq!(cell.holder(), Holder::Free, "the refusal moved nothing");
        cell.begin_release().await;
        assert!(
            cell.probe_claim().is_none(),
            "Releasing refuses a new claim"
        );
    }

    /// Pins `begin_release`'s two statements in order: the phase flip lands
    /// strictly before the epoch bump. The two-catch lattice
    /// `run_placed_gang`'s own doc and `WorkerShared::for_single_run`'s
    /// depend on ("a snapshot whose epoch bump is already visible implies
    /// the flip already happened too") only holds in that order, and the
    /// lattice's end-to-end test (`gang_placed::
    /// release_landing_between_the_epoch_read_and_probe_claim_is_still_
    /// refused`) cannot falsify the ORDER because its own RELEASE runs to
    /// full completion (both statements) inside one park, never observing
    /// the gap between them.
    ///
    /// This test parks a live `begin_release` call between its two
    /// statements and reads BOTH sides of the window directly: while
    /// parked, the phase must already read `Releasing` (so a concurrent
    /// `probe_claim` refuses) while the epoch must NOT yet be bumped.
    ///
    /// Mutation: swapping `begin_release`'s two statements (bump then flip,
    /// with the park between them) fails the first assertion ("phase must
    /// already be Releasing while parked between the flip and the bump",
    /// `left: Running`) — under the swap the epoch is already bumped but the
    /// phase has not yet flipped at the park, so this catches it before the
    /// `probe_claim` assertion below even runs.
    #[tokio::test(flavor = "multi_thread")]
    async fn begin_release_bumps_the_epoch_strictly_after_the_phase_flip_is_already_visible() {
        let admission = HostAdmission::new(Arc::new(InstanceRegistration::new(
            "release-window-cell",
            None,
            None,
            None,
            None,
        )));
        let birth_epoch = admission.release_epoch();
        assert_eq!(birth_epoch, 0, "a fresh admission starts at epoch 0");

        let park = loop_test_hooks::arm(
            &admission.registry().instance_id,
            loop_test_hooks::ParkPoint::BeginReleaseBetweenFlipAndBump,
        );
        let releasing = {
            let admission = Arc::clone(&admission);
            tokio::spawn(async move { admission.begin_release().await })
        };
        park.wait_parked().await;

        // Parked strictly between the flip and the bump: the flip must
        // already be visible ...
        assert_eq!(
            admission.phase(),
            WorkerPhase::Releasing,
            "phase must already be Releasing while parked between the flip and the bump"
        );
        // ... while the bump must NOT be visible yet.
        assert_eq!(
            admission.release_epoch(),
            birth_epoch,
            "the epoch must not be bumped yet while parked between the flip and the bump"
        );
        // A concurrent `probe_claim` — exactly `run_placed_gang`'s own
        // check, racing this window — must refuse on the phase alone, with
        // no epoch compare available to it at all.
        assert!(
            admission.probe_claim().is_none(),
            "probe_claim must refuse on the phase read alone while the epoch is still unbumped"
        );

        park.release();
        tokio::time::timeout(Duration::from_secs(10), releasing)
            .await
            .expect("begin_release must resume and return once released")
            .unwrap();

        assert_eq!(
            admission.release_epoch(),
            birth_epoch + 1,
            "the bump lands once begin_release resumes"
        );
    }

    /// `Awaiting` (the state a
    /// claim's own `submit_placed` puts the holder in BEFORE it submits —
    /// the move precedes the submit) admits a `RunRank` session EXACTLY like `Free`
    /// (a two-host fleet could not otherwise assemble if its only
    /// free-looking host were the one awaiting its own placement result)
    /// and refuses a second claim EXACTLY like `JobRun`; once the claim's
    /// own guard drops with no rank having taken the cell over, the host is
    /// `Free` again. Mutation: `probe_claim`'s admitting predicate widened
    /// to `matches!(h, Holder::Free | Holder::Awaiting { .. })` (admitting
    /// a SECOND claim while awaiting) reds this test's second assertion.
    #[test]
    fn awaiting_admits_a_rank_and_refuses_a_second_claim_and_frees_when_the_claim_ends() {
        let cell = cell();
        let claim = cell.probe_claim().expect("Free admits the claim's probe");
        // `register_job_hold_or_release`'s own transition, mimicked
        // directly (it needs a live catalog/session to call for real).
        cell.job_running();
        assert_eq!(cell.holder(), Holder::JobRun);
        assert_eq!(
            cell.begin_awaiting_placement("job-a", 1),
            Ok(()),
            "JobRun -> Awaiting"
        );
        assert_eq!(
            cell.begin_awaiting_placement("job-a", 1),
            Err(Holder::Awaiting {
                job_id: "job-a".into(),
                attempt: 1
            }),
            "the move is a CAS from exactly JobRun: from Awaiting it refuses with the \
             holder it SAW (one critical section, never a second read); `submit_placed` \
             submits on that refusal only for Free, and ends typed for any other holder"
        );
        assert_eq!(
            cell.holder(),
            Holder::Awaiting {
                job_id: "job-a".into(),
                attempt: 1
            }
        );
        assert!(
            cell.probe_claim().is_none(),
            "Awaiting refuses a second claim exactly like JobRun"
        );

        // A RunRank admission on the submitter's host succeeds.
        let rank = cell
            .try_hold_rank("job-b", 7)
            .expect("Awaiting admits a rank exactly like Free");
        assert_eq!(
            cell.holder(),
            Holder::Rank {
                job_id: "job-b".into(),
                attempt: 7
            }
        );
        drop(rank);
        assert_eq!(cell.holder(), Holder::Free);
        // The original claim's own guard, dropped after a rank already
        // took the cell over, is a no-op (the cell no longer names
        // `Awaiting`) — never resurrects a hold that already ended.
        drop(claim);
        assert_eq!(cell.holder(), Holder::Free);

        // The plain case: no rank ever takes the cell — the claim's own
        // guard alone returns the host to `Free` once the await ends.
        let claim2 = cell.probe_claim().expect("Free admits the claim's probe");
        cell.job_running();
        assert_eq!(cell.begin_awaiting_placement("job-c", 1), Ok(()));
        drop(claim2);
        assert_eq!(
            cell.holder(),
            Holder::Free,
            "the claim guard resets Awaiting to Free exactly as it resets JobRun"
        );
    }
}
