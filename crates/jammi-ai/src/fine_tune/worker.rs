//! The training worker: claims durable [`crate::fine_tune::spec::TrainingSpec`]
//! jobs under a lease, reconstructs each from its persisted spec, trains it
//! while heartbeating the lease, and records the terminal outcome.
//!
//! One worker drives every training verb. A [`TrainingWorker::run`] tick first
//! reclaims expired leases (re-queuing a dead worker's job, or failing it past
//! the attempts cap), then atomically claims the oldest queued job. On a claim
//! it deserialises the spec, re-scopes the catalog to the job's tenant, and
//! dispatches by kind to a *from-scratch* reconstruction — re-running the source
//! SQL, re-reading and re-sampling the graph (seeded, deterministic), or
//! re-sampling the episodic meta-dataset. No in-memory state crosses the
//! submit→claim boundary, so a worker can run a job submitted by a now-gone
//! session on a fresh process.
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
//! is cooperative: a heartbeat task renews the lease on an interval; when
//! `heartbeat_training_job` returns `false` (the lease was lost — reclaimed by
//! another worker, or expired) it sets a shared cancel flag the training loop
//! checks at every epoch boundary. The loop then bails, leaving the job
//! `running` for the next `reclaim_expired_training_jobs` to re-queue.
//!
//! Cancellation is checked only at epoch boundaries, so a worker can still lose
//! its lease in the window between the last check and finalization. The terminal
//! write is therefore a compare-and-set: [`Catalog::finalize_training_job`]
//! writes the output model + flips the job to `completed` only while
//! `claimed_by` is still this worker and the status is still `running`. A worker
//! that lost its lease matches zero rows and does not finalize, so two workers
//! never both finalize the same job — the re-claiming worker is the sole
//! finalizer.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use arrow::array::RecordBatch;
use bytes::Bytes;
use jammi_db::catalog::Catalog;
use jammi_db::config::WorkerIntervals;
use jammi_db::error::{JammiError, Result};
use jammi_db::model_task::ModelTask;
use jammi_db::sql::{quote_ident, source_relation};
use jammi_db::store::ArtifactStore;

use crate::fine_tune::data::TrainingDataLoader;
use crate::fine_tune::graph_sampler::{
    GraphEdge, GraphFineTuneSources, GraphSampleConfig, GraphSampler, TextNode,
};
use crate::fine_tune::spec::{TrainingCommon, TrainingSpec};
use crate::fine_tune::FineTuneConfig;
use crate::model::backend::DeviceConfig;
use crate::model::ModelSource;
use crate::session::InferenceSession;

// Lease timing is configured per deployment via `[training]` in `JammiConfig`
// and resolved to a [`WorkerIntervals`] (see
// [`jammi_db::config::TrainingConfig::worker_intervals`]). The lease is the
// window a claimed job is exclusively owned; the heartbeat renews it well
// inside that window so a single missed beat (a GC pause, a slow tick) does not
// drop the lease — the config layer enforces a ≥2× margin between the lease and
// the beat so that invariant holds for every deployment, never silently
// clamped. The idle poll is how often an idle worker checks for new work, and
// reclaim runs each idle tick so a dead worker's job is recovered within roughly
// one poll + lease. The defaults reproduce the historical 30 s / 10 s / 1 s
// timing; a short config drives lease-expiry and reclaim quickly.

/// Attempts cap before `reclaim_expired_training_jobs` fails a job for good.
const MAX_ATTEMPTS: u32 = 3;

/// Environment override for the worker's stable `claimed_by` identity. When set
/// (and non-empty), a worker adopts this exact id instead of minting a random
/// per-process uuid. A fleet operator uses it for stable identity in logs and
/// lease ownership across restarts; a multi-process test harness uses it to
/// assert which worker ran a given job. Unset (or empty) → the random-uuid
/// default, so a plain single-process deployment is byte-unchanged.
const WORKER_ID_ENV: &str = "JAMMI_WORKER_ID";

/// Resolve the worker's stable id: the trimmed `JAMMI_WORKER_ID` when set and
/// non-empty, otherwise a fresh `worker-{uuid}`. An all-whitespace value is
/// treated as unset — it would be a useless `claimed_by` and silently break
/// ownership assertions, so it falls back rather than seeding a blank id.
fn resolve_worker_id() -> String {
    match std::env::var(WORKER_ID_ENV) {
        Ok(v) if !v.trim().is_empty() => v.trim().to_string(),
        _ => format!("worker-{}", uuid::Uuid::new_v4()),
    }
}

/// Whether epoch checkpointing is enabled for this spec, and if so, its
/// epoch bound (`FineTuneConfig.epochs`) and retention cap
/// (`FineTuneConfig.keep_last_n_checkpoints`) — unit 348, F1/F2/F3.
///
/// `None` = DISABLED, the default: `keep_last_n_checkpoints` absent. Every
/// existing (pre-unit-348) caller and every default-configured job carries
/// no such field, so this is `None` for them — zero epoch-checkpoint bytes
/// ever written, zero catalog rows ever registered, and every derived-sweep
/// GC call site below is a bound-`0` no-op that returns immediately without
/// issuing a single store request (F1: opt-in blast radius, not a tax on
/// every legacy job). `Some((epochs, keep))` = ENABLED — read here from the
/// durable spec BEFORE it moves into the run, never a count of epochs
/// actually completed (not observable from outside a run that may never
/// reach its first epoch boundary). `ContextPredictor` has no per-epoch
/// checkpointing at all (v1 out of scope), so it is always disabled.
fn epoch_checkpointing(spec: &TrainingSpec) -> Option<(usize, u32)> {
    match spec {
        TrainingSpec::FineTune { common, .. } | TrainingSpec::GraphFineTune { common, .. } => {
            common
                .config
                .keep_last_n_checkpoints
                .map(|keep| (common.config.epochs, keep))
        }
        TrainingSpec::ContextPredictor { .. } => None,
    }
}

/// A training worker bound to a session. Claims and runs durable training jobs
/// from the shared catalog under a lease. Construct one per process (or N for a
/// pool); [`Self::run`] is the long-lived loop the embedded engine and the
/// server `train` tier both drive.
pub struct TrainingWorker {
    /// Weak back-reference to the session — upgraded each tick. `None` means the
    /// session dropped, which is the loop's exit condition (no refcycle keeps
    /// the session alive).
    session: Weak<InferenceSession>,
    /// Stable id stamped into `claimed_by` so a heartbeat / reclaim can tell
    /// this worker's leases from another's. Seeded from `JAMMI_WORKER_ID` when
    /// set, else a fresh random `worker-{uuid}` (see [`resolve_worker_id`]).
    worker_id: String,
    /// The validated lease/heartbeat/poll timing this worker drives its loop
    /// with. `intervals.lease` is the single source of truth threaded to both
    /// the claim and the heartbeat, so the renew always targets the same
    /// deadline the reclaim path compares against.
    intervals: WorkerIntervals,
}

impl TrainingWorker {
    /// Build a worker over a session, reading its lease/heartbeat/poll timing
    /// from the session's `[training]` configuration. The worker holds a
    /// [`Weak`] so it never keeps the session alive; the caller owns the strong
    /// `Arc` and the worker stops when that drops.
    ///
    /// Returns [`JammiError::Config`] if the configured timing violates the
    /// worker invariants (heartbeat margin / non-zero poll). In the normal flow
    /// the same check already ran at config load, so this only fires for a
    /// programmatically built config that bypassed `JammiConfig::load`.
    pub fn new(session: &Arc<InferenceSession>) -> Result<Self> {
        let intervals = session.inner_config().training.worker_intervals()?;
        Ok(Self::with_intervals(session, intervals))
    }

    /// Build a worker over a session with explicit, already-validated timing.
    /// The [`WorkerIntervals`] type can only be produced by
    /// [`jammi_db::config::TrainingConfig::worker_intervals`], so its invariants
    /// hold by construction.
    ///
    /// The worker's `claimed_by` identity is seeded from `JAMMI_WORKER_ID` when
    /// that env var is set and non-empty, else a fresh random `worker-{uuid}`.
    pub fn with_intervals(session: &Arc<InferenceSession>, intervals: WorkerIntervals) -> Self {
        Self {
            session: Arc::downgrade(session),
            worker_id: resolve_worker_id(),
            intervals,
        }
    }

    /// The worker's stable id (`claimed_by` value). Exposed for tests that assert
    /// on lease ownership.
    pub fn worker_id(&self) -> &str {
        &self.worker_id
    }

    /// Run the claim→reconstruct→train loop until the session drops.
    ///
    /// Equivalent to [`Self::run_until`] with a never-set stop flag — for callers
    /// that rely solely on the session dropping (the `Weak` upgrade failing) to
    /// stop the worker.
    pub async fn run(&self) {
        self.run_until(Arc::new(AtomicBool::new(false))).await
    }

    /// Run the claim→reconstruct→train loop until either `stop` is set or the
    /// session drops.
    ///
    /// Stack-safe: a bounded `loop`, never recursion. Each tick reclaims expired
    /// leases then attempts one claim; on a claim it runs the job to a terminal
    /// state inline (the next claim waits for it), on no claim it sleeps the
    /// configured idle poll. The catalog used for reclaim/claim is unscoped — a worker
    /// serves every tenant's queue.
    pub async fn run_until(&self, stop: Arc<AtomicBool>) {
        loop {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            let session = match self.session.upgrade() {
                Some(s) => s,
                // The session dropped: nothing more to serve, exit the loop.
                None => return,
            };
            let catalog = session.catalog();

            if let Err(e) = catalog.reclaim_expired_training_jobs(MAX_ATTEMPTS).await {
                tracing::error!(worker = %self.worker_id, error = %e, "reclaim_expired_training_jobs failed");
            }

            let claimed = match catalog
                .claim_next_training_job(&self.worker_id, self.intervals.lease)
                .await
            {
                Ok(c) => c,
                Err(e) => {
                    tracing::error!(worker = %self.worker_id, error = %e, "claim_next_training_job failed");
                    None
                }
            };

            match claimed {
                Some(record) => {
                    // Drop the session strong ref before the (possibly long) run
                    // so the worker does not pin the session for the whole job —
                    // the run re-upgrades the Weak through the `Arc` it captures.
                    self.run_claimed_job(&session, record).await;
                }
                None => tokio::time::sleep(self.intervals.idle_poll).await,
            }
        }
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
    /// (campaign #446 finding 1, audited exhaustively)
    ///
    /// The esc-075 tri-state contract's `{"state":"pending"}` marker means "no
    /// claimant has computed a determination YET", so it must not survive onto
    /// a row that has gone terminal. `Catalog::fail_training_job` retires a
    /// still-`pending` report to
    /// `{"state":"undetermined","reason":"failed_before_probe"}` in the SAME
    /// lease-guarded UPDATE (see its own doc) — which covers this function's
    /// failure paths EXACTLY as long as each one goes through
    /// `record_failed`. It does. Enumerated, so a path added later that
    /// skips it is visibly outside this list rather than silently uncovered:
    ///
    /// | # | failure | terminal write |
    /// |---|---|---|
    /// | 1 | no `training_spec` at all | `mark_acceleration_undetermined` (a MORE specific `failed_before_device_resolution` reason, which the catalog edge preserves) then `record_failed` |
    /// | 2 | undeserialisable `training_spec` | same as 1 |
    /// | 3 | source SQL / loader reconstruction error (`read_source_columns`, `build_training_data_loader`, `reconstruct_graph_loader`) | `Err(Failed)` → `record_failed` |
    /// | 4 | base-model load error, incl. a missing artifact (`model_cache().get_or_load`) | `Err(Failed)` → `record_failed` |
    /// | 5 | base model exposes no embedding dim | `Err(Failed)` → `record_failed` |
    /// | 6 | device-select error (`select_device`, inside `run_fine_tune_blocking` — BEFORE the probe) | `Err(Failed)` → `record_failed` |
    /// | 7 | head/adapter construction error (`build_classification_head`, `build_distribution_head`, `build_encoder_adapters`, incl. `validate_backbone_precision`) — also before the probe | `Err(Failed)` → `record_failed` |
    /// | 8 | typed training failure after the probe | `Err(Failed)` → `record_failed` (report is already `determined`) |
    /// | 9 | `spawn_blocking` panic (caught) or join error | `Err(Failed)` → `record_failed` |
    /// | 10 | final-artifact publish failure | `record_failed` |
    /// | 11 | `register_model` failure | `record_failed` |
    ///
    /// The ONE deliberate exception is `Err(WorkerJobError::Cancelled)` (the
    /// lease was lost, or a genuine error coincided with a lost lease — see
    /// `classify`): this writes NO terminal status, because a different
    /// worker now owns the job. Its `pending` marker is retired by the OTHER
    /// half of the same catalog-edge rule —
    /// `Catalog::reclaim_expired_training_jobs`' exhausted arm writes
    /// `{"state":"undetermined","reason":"lease_expired_attempts_exhausted"}`,
    /// and its requeue arm RESETS the column to `pending` for the fresh
    /// attempt that will re-probe. Adding a `record_failed` here would be
    /// wrong twice over: it would stamp `failed` over a job the re-claiming
    /// worker is running, and its lease guard would not match anyway.
    ///
    /// `Self::publish_and_finalize`'s own `finalize_training_job`
    /// `Ok(false)`/`Err` arms likewise leave the job `running` for reclaim, so
    /// no terminal status is written on them either and the same reclaim half
    /// of the catalog-edge rule applies.
    ///
    /// # The SUCCESS path is not exempt (campaign #446 round-1 audit)
    ///
    /// An earlier revision of this note claimed the post-probe paths were not
    /// at stake "because the report is already `determined`". That is not
    /// guaranteed: `persist_acceleration_report` deliberately SWALLOWS a
    /// lease-guard miss (`Ok(false)`, e.g. a stale `attempt`) and a catalog
    /// error, by design — the write not landing must never fail training. A
    /// job whose probe write was swallowed and which then finalizes
    /// successfully reaches `completed` with the submission-time `pending`
    /// marker still on the row, which is the SAME forbidden state the failure
    /// paths above avoid, by a success path. It is covered at the same ONE
    /// catalog edge: `Catalog::finalize_training_job` retires a still-`pending`
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
        record: jammi_db::catalog::training_repo::TrainingJobRecord,
    ) {
        let job_id = record.job_id.clone();
        // The attempt counter makes the artifact prefix unique per (job, worker,
        // attempt): a reclaimed job re-runs under a higher `attempts`, so its
        // new attempt writes to a fresh prefix and never overwrites the prior
        // attempt's objects.
        let attempt = record.attempts;
        let catalog = Arc::new(session.catalog().pinned_to_tenant(record.tenant_id));

        let spec_json = match record.training_spec.as_deref() {
            Some(s) => s,
            None => {
                // esc-075 (Phase-4 audit finding 4): this fails BEFORE the
                // device is ever resolved, so `run_fine_tune_blocking`'s
                // measuring probe never runs — write the honest terminal
                // marker first (still `running`, satisfying the lease guard)
                // so the record never reads `{"state":"pending"}` past this
                // job's `failed` status below.
                mark_acceleration_undetermined(&catalog, &job_id, &self.worker_id, attempt).await;
                record_failed(
                    &catalog,
                    &job_id,
                    &self.worker_id,
                    "job carries no training_spec".into(),
                )
                .await;
                return;
            }
        };
        let spec: TrainingSpec = match serde_json::from_str(spec_json) {
            Ok(s) => s,
            Err(e) => {
                // esc-075 (Phase-4 audit finding 4): same reasoning as the
                // missing-`training_spec` arm above.
                mark_acceleration_undetermined(&catalog, &job_id, &self.worker_id, attempt).await;
                record_failed(
                    &catalog,
                    &job_id,
                    &self.worker_id,
                    format!("undeserialisable training_spec: {e}"),
                )
                .await;
                return;
            }
        };
        // Whether epoch checkpointing is enabled for THIS run, and if so its
        // epoch bound and retention cap — read from the spec's own
        // `FineTuneConfig` before `spec` moves into `run_spec` below, never
        // a count of epochs actually completed (which the outer scope
        // cannot see and does not need to: sweeping the full configured
        // range is what makes the GC below correct BY CONSTRUCTION,
        // independent of how far training got or whether it ever built a
        // `TrainedArtifact` at all — see `Self::gc_epoch_checkpoints`'s doc,
        // unit 348 F1/F2/F3). `None` (the default — no `keep_last_n_
        // checkpoints`) makes every GC call below a bound-`0` no-op.
        let epoch_checkpointing = epoch_checkpointing(&spec);
        let epoch_checkpoint_bound = epoch_checkpointing.map(|(b, _)| b).unwrap_or(0);

        // The heartbeat renews the lease while training runs and sets `cancel`
        // when the lease is lost. The cancel flag threads into both training
        // paths' epoch-boundary checks.
        let cancel = Arc::new(AtomicBool::new(false));
        let heartbeat =
            self.spawn_heartbeat(Arc::clone(&catalog), job_id.clone(), Arc::clone(&cancel));

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
        // catalog (above) because a fine-tune's `register_model` /
        // `get_model` runs inside (or after) a `spawn_blocking` thread, which
        // does not inherit the task-local; the predictor's async reads are
        // covered by this scope.
        let outcome = match record.tenant_id {
            Some(tenant) => {
                session
                    .with_tenant_scoped(tenant, |_scope| {
                        self.run_spec(session, &catalog, &job_id, spec, &cancel, attempt)
                    })
                    .await
            }
            None => {
                self.run_spec(session, &catalog, &job_id, spec, &cancel, attempt)
                    .await
            }
        };

        // Stop the heartbeat regardless of outcome.
        heartbeat.abort();

        match outcome {
            Ok(artifact) => {
                self.publish_and_finalize(
                    session,
                    &catalog,
                    &job_id,
                    attempt,
                    epoch_checkpointing,
                    artifact,
                )
                .await;
            }
            Err(WorkerJobError::Cancelled) => {
                // Lease lost: leave the job `running` for reclaim to re-queue.
                // Do not record a terminal status — a different worker now owns,
                // or will own, this job. No `TrainedArtifact` was ever built on
                // this path (the run bailed mid-training, or never even
                // finished the blocking call), so any epoch checkpoints this
                // attempt wrote are reachable only by DERIVING their prefixes
                // — never from an in-memory vec that does not exist here.
                Self::gc_epoch_checkpoints(
                    &session.artifact_store(),
                    &job_id,
                    &self.worker_id,
                    attempt,
                    epoch_checkpoint_bound,
                )
                .await;
                tracing::warn!(job_id = %job_id, worker = %self.worker_id, "training cancelled (lease lost); left for reclaim");
            }
            Err(WorkerJobError::Failed(msg)) => {
                tracing::error!(job_id = %job_id, error = %msg, "training job failed");
                record_failed(&catalog, &job_id, &self.worker_id, msg).await;
                // Same reasoning as the `Cancelled` arm above: covers a panic,
                // a `spawn_blocking` join error, and any typed training
                // failure — none of which ever produced a `TrainedArtifact`.
                Self::gc_epoch_checkpoints(
                    &session.artifact_store(),
                    &job_id,
                    &self.worker_id,
                    attempt,
                    epoch_checkpoint_bound,
                )
                .await;
            }
        }
    }

    /// Publish a trained artifact to the object store and run the single
    /// lease-guarded finalization for every job kind — the catalog-pointer-as-
    /// commit path.
    ///
    /// The worker writes the artifact files to the store under a **unique
    /// per-attempt prefix** (`{job_id}/{worker_id}/{attempt}`), registers the
    /// output-model row (with **no** served path yet), then runs the lease-guarded
    /// compare-and-set that records `output_model_id`, flips the job to
    /// `completed`, and — atomically, in the same transaction, gated on that CAS
    /// matching — commits this worker's prefix as the model's served
    /// `artifact_path`. Because every attempt writes a fresh prefix, no object is
    /// ever overwritten or moved, and the served pointer is written by exactly
    /// one writer: the finalize CAS. The CAS matches only while this worker still
    /// holds the lease (`claimed_by = worker_id AND status = 'running'`), so a
    /// worker that lost its lease in the window between the last epoch check and
    /// here affects zero rows — it commits neither the job's terminal status nor
    /// any served path — and the job is left `running` for `reclaim` while its
    /// prefix is orphaned (best-effort GC'd). A `wait()` observer that sees
    /// `completed` therefore always finds the served `artifact_path` set to the
    /// winner's complete artifact.
    ///
    /// The model row is registered through the tenant-pinned `catalog` so it
    /// lands under the job's tenant. Registration is idempotent (the catalog
    /// upserts on the deterministic `model_id`) and never sets the served path,
    /// so a re-claiming worker (or a zombie loser) re-registering after a lost
    /// lease is safe: its registration cannot touch the committed pointer, and
    /// the served `artifact_path` is set only by whichever worker's finalize CAS
    /// wins. A loser's prefix is therefore never the committed pointer and is the
    /// one GC'd.
    async fn publish_and_finalize(
        &self,
        session: &Arc<InferenceSession>,
        catalog: &Arc<Catalog>,
        job_id: &str,
        attempt: u32,
        epoch_checkpointing: Option<(usize, u32)>,
        artifact: TrainedArtifact,
    ) {
        let store = session.artifact_store();
        let epoch_checkpoint_bound = epoch_checkpointing.map(|(b, _)| b).unwrap_or(0);
        let TrainedArtifact {
            dir,
            register,
            metrics,
            mut epoch_checkpoints,
        } = artifact;
        let model_id = register.model_id.clone();

        // Write the artifact under a unique per-attempt prefix, then register the
        // model row — both before the CAS, so a `completed` observer always finds
        // a registered model row. The registration does NOT carry the served
        // path: the finalize CAS is the sole writer of `artifact_path`, so a
        // loser's (or zombie's) register can never set the served pointer.
        let attempt_str = attempt.to_string();
        let prefix =
            match publish_artifact(&store, job_id, &self.worker_id, &attempt_str, &dir).await {
                Ok(p) => p,
                Err(e) => {
                    record_failed(catalog, job_id, &self.worker_id, e.to_string()).await;
                    // The training loop DID complete and DID write epoch
                    // checkpoints (we have a `TrainedArtifact`) — but the
                    // FINAL artifact publish failed, so this attempt never
                    // reaches finalize at all. Reclaim its epoch-checkpoint
                    // bytes via the derived sweep (never the vec — one
                    // reclaim path for every terminating arm, unit 348 F1/F2).
                    Self::gc_epoch_checkpoints(
                        &store,
                        job_id,
                        &self.worker_id,
                        attempt,
                        epoch_checkpoint_bound,
                    )
                    .await;
                    return;
                }
            };

        if let Err(e) = catalog.register_model(register.as_params()).await {
            // The model row could not be registered; the prefix we wrote is
            // orphaned. Best-effort GC it and fail the job.
            store.delete_artifact_prefix(&prefix).await.ok();
            Self::gc_epoch_checkpoints(
                &store,
                job_id,
                &self.worker_id,
                attempt,
                epoch_checkpoint_bound,
            )
            .await;
            record_failed(catalog, job_id, &self.worker_id, e.to_string()).await;
            return;
        }

        // Unit 348 F2: TRIM to the trailing retention window before
        // registering. `epoch_checkpoints` can hold MORE than `keep` entries
        // when a mid-run prune delete kept failing (`TrainingLoop::
        // save_epoch_checkpoint`'s retention loop leaves a failed entry in
        // place rather than dropping track of it) — a persistently-failed
        // delete must never let more than `keep` rows register just because
        // its bytes could not be deleted yet. `split_off` leaves the OLDER,
        // over-the-cap entries (if any) in `epoch_checkpoints` — exactly the
        // "non-retained" set the winner-arm sweep below reclaims — and
        // returns the trailing `keep` as `retained`, the set that actually
        // registers.
        let retained: Vec<(usize, String)> = match epoch_checkpointing {
            Some((_, keep)) => {
                let keep = keep as usize;
                if epoch_checkpoints.len() > keep {
                    epoch_checkpoints.split_off(epoch_checkpoints.len() - keep)
                } else {
                    std::mem::take(&mut epoch_checkpoints)
                }
            }
            None => Vec::new(),
        };
        // From here, `epoch_checkpoints` holds only the STALE entries (if
        // any) whose mid-run prune kept failing — never the retained set.

        // Distinct-name catalog rows for every RETAINED epoch checkpoint
        // (unit 348, CONTRACT item 4): never an additional VERSION of the
        // output model's name. Built here (owned `String`s outliving the
        // `finalize_training_job` call) so the `EpochCheckpointRow` borrows
        // are valid for the whole call.
        let epoch_model_ids: Vec<String> = retained
            .iter()
            .map(|(epoch, _)| format!("{model_id}:epoch_{epoch}"))
            .collect();
        let epoch_rows: Vec<jammi_db::catalog::training_repo::EpochCheckpointRow<'_>> = retained
            .iter()
            .zip(epoch_model_ids.iter())
            .map(|((_epoch, path), epoch_model_id)| {
                jammi_db::catalog::training_repo::EpochCheckpointRow {
                    model_id: epoch_model_id,
                    model_type: register.model_type,
                    task: register.task,
                    base_model_id: register.base_model_id.as_deref(),
                    artifact_path: path,
                }
            })
            .collect();

        match catalog
            .finalize_training_job(
                jammi_db::catalog::training_repo::FinalizeTrainingJobParams {
                    job_id,
                    worker_id: &self.worker_id,
                    output_model_id: &model_id,
                    output_model_version: register.version,
                    artifact_path: prefix.as_str(),
                    metrics: metrics.as_deref(),
                    epoch_checkpoints: &epoch_rows,
                },
            )
            .await
        {
            Ok(true) => {
                // The finalize CAS won: the job is `completed`, so its durable
                // resume checkpoint is dead. GC it (best-effort — a leftover
                // resume prefix is harmless, never on the serving path, but the
                // winner is the single point that reclaims it).
                store.delete_resume_checkpoint(job_id).await.ok();
                // Unit 348 F2: the winner is also the single point that
                // reclaims any STALE (over-the-cap, failed-to-prune-mid-run)
                // epoch checkpoints — bytes a repeatedly-failing delete left
                // durable but excluded from `retained` above. Without this,
                // a persistently-failed prune would leak forever (never
                // registered, never swept). Targets exactly the known stale
                // indices (equivalent to sweeping `[0, bound) \ retained`,
                // computed directly rather than as a broad range since the
                // stale list is already in hand).
                if !epoch_checkpoints.is_empty() {
                    Self::gc_epoch_checkpoints_by_index(
                        &store,
                        job_id,
                        &self.worker_id,
                        attempt,
                        epoch_checkpoints.into_iter().map(|(epoch, _)| epoch),
                    )
                    .await;
                }
            }
            Ok(false) => {
                // Lost the lease before finalizing: our CAS matched zero rows, so
                // we committed neither the job status nor any served path. Our
                // prefix is never the committed pointer — GC it best-effort and
                // leave the job for reclaim (the re-claiming worker writes its own
                // prefix and its CAS commits it).
                store.delete_artifact_prefix(&prefix).await.ok();
                Self::gc_epoch_checkpoints(
                    &store,
                    job_id,
                    &self.worker_id,
                    attempt,
                    epoch_checkpoint_bound,
                )
                .await;
                tracing::debug!(
                    job_id = %job_id,
                    worker = %self.worker_id,
                    "lost lease before finalize; not finalizing (left for reclaim)"
                );
            }
            Err(e) => {
                store.delete_artifact_prefix(&prefix).await.ok();
                Self::gc_epoch_checkpoints(
                    &store,
                    job_id,
                    &self.worker_id,
                    attempt,
                    epoch_checkpoint_bound,
                )
                .await;
                tracing::error!(job_id = %job_id, error = %e, "finalize_training_job failed");
            }
        }
    }

    /// Best-effort GC of THIS attempt's epoch-checkpoint bytes on any
    /// terminating path that is not the finalize-CAS winner (unit 348,
    /// F1/F2/F3). Derives every candidate prefix directly from the attempt's
    /// identity (`job_id`/`worker_id`/`attempt`) and the run's CONFIGURED
    /// epoch bound — by construction, never from the in-memory
    /// `TrainedArtifact::epoch_checkpoints` vec, which simply does not exist
    /// on most of the paths that call this: a lease-lost cancel, a typed
    /// training failure, a final-artifact publish failure, or a
    /// `register_model` failure all return (or bail) before ever building a
    /// `TrainedArtifact`. The one arm that DOES have the vec (a completed run
    /// whose finalize CAS then loses) still goes through this same derived
    /// sweep — there is exactly ONE reclaim mechanism, called from every
    /// terminating arm, not a vec-based mechanism for the lucky case and a
    /// derived one for the rest.
    ///
    /// A panic or `spawn_blocking` join error inside
    /// [`Self::train_fine_tune`] does NOT call this directly — both of those
    /// arms return `WorkerJobError::Failed`, which propagates unchanged to
    /// [`Self::run_claimed_job`]'s exhaustive `Cancelled`/`Failed` match,
    /// which DOES call this. Sweeping there too would be a harmless-but-
    /// wasteful DOUBLE sweep of the identical range for the identical
    /// attempt — one call site per termination, not two.
    ///
    /// `epoch_checkpoint_bound` is `0` whenever epoch checkpointing was never
    /// enabled for this spec (`FineTuneConfig.keep_last_n_checkpoints`
    /// absent — the default, and every pre-unit-348 caller) — this returns
    /// immediately, before even the trivial `attempt.to_string()` allocation,
    /// so a legacy/default job's terminating arm issues ZERO store requests
    /// (F1: an opt-in blast radius, not a tax on every job). For an ENABLED
    /// job, [`ArtifactStore::delete_epoch_checkpoint`] is already a no-op for
    /// any index that was never written (an absent manifest is "nothing
    /// durable to reclaim", not an error — the same rule
    /// [`ArtifactStore::delete_artifact_prefix`] applies), so sweeping the
    /// full `[0, epochs)` configured range costs at most `epochs` no-op reads
    /// beyond whatever indices actually existed — correct regardless of how
    /// far training got, or whether it ever ran a single epoch boundary. This
    /// O(epochs) failure-path reclaim cost is the accepted price of "opt-in,
    /// bounded, and never silent" — see [`Self::gc_epoch_checkpoints_by_index`]
    /// for the one-warning-per-sweep diagnostic.
    ///
    /// This is the ONE reclaim path (family E, the term that grows — every
    /// reclaimed/failed attempt's per-epoch storage — must be bounded, not
    /// left to accumulate). It reaches a LIVE process that survives to call
    /// it; a process that crashes before reaching ANY of these call sites at
    /// all is the one residual case nothing here (or the pre-existing
    /// top-level artifact-prefix GC) reaches — durable-but-permanently-
    /// unregistered, the expected residual (documented on
    /// [`jammi_db::catalog::training_repo::EpochCheckpointRow`]).
    async fn gc_epoch_checkpoints(
        store: &ArtifactStore,
        job_id: &str,
        worker_id: &str,
        attempt: u32,
        epoch_checkpoint_bound: usize,
    ) {
        if epoch_checkpoint_bound == 0 {
            return;
        }
        Self::gc_epoch_checkpoints_by_index(
            store,
            job_id,
            worker_id,
            attempt,
            0..epoch_checkpoint_bound,
        )
        .await;
    }

    /// The shared epoch-index sweep both [`Self::gc_epoch_checkpoints`] (a
    /// full `[0, bound)` range) and `publish_and_finalize`'s winner arm (the
    /// specific stale indices a persistently-failed mid-run prune left
    /// behind, unit 348 F2) drive. Attempts a best-effort delete of every
    /// index in `epochs`; if ANY fail, emits exactly ONE `tracing::warn!`
    /// naming the job/worker/attempt and the failed-vs-attempted count —
    /// never zero (a silently-swallowed sweep failure) and never one warning
    /// per failed delete (a warning storm when the whole store is down for
    /// this attempt).
    async fn gc_epoch_checkpoints_by_index(
        store: &ArtifactStore,
        job_id: &str,
        worker_id: &str,
        attempt: u32,
        epochs: impl Iterator<Item = usize>,
    ) {
        let attempt_str = attempt.to_string();
        let mut attempted = 0usize;
        let mut failed = 0usize;
        for epoch in epochs {
            attempted += 1;
            if store
                .delete_epoch_checkpoint(job_id, worker_id, &attempt_str, epoch)
                .await
                .is_err()
            {
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
                "epoch-checkpoint GC sweep: {failed} of {attempted} delete(s) failed — those \
                 bytes remain durable but unreachable by this sweep"
            );
        }
    }

    /// Spawn the lease-renewing heartbeat task. It renews on the configured
    /// heartbeat interval and, the first time `heartbeat_training_job` reports
    /// the lease lost, sets `cancel` and stops. The renewed lease window is the
    /// same `intervals.lease` the claim used, so the heartbeat and the reclaim
    /// path share one source of truth for the deadline.
    fn spawn_heartbeat(
        &self,
        catalog: Arc<Catalog>,
        job_id: String,
        cancel: Arc<AtomicBool>,
    ) -> tokio::task::JoinHandle<()> {
        let worker_id = self.worker_id.clone();
        let heartbeat = self.intervals.heartbeat;
        let lease = self.intervals.lease;
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(heartbeat).await;
                match catalog
                    .heartbeat_training_job(&job_id, &worker_id, lease)
                    .await
                {
                    Ok(true) => {}
                    Ok(false) => {
                        // Lease lost — signal the training loop to bail.
                        cancel.store(true, Ordering::Relaxed);
                        return;
                    }
                    Err(e) => {
                        tracing::error!(job_id = %job_id, error = %e, "heartbeat failed");
                    }
                }
            }
        })
    }

    /// Dispatch a claimed spec to its kind's from-scratch reconstruction and
    /// training, returning the [`TrainedArtifact`] on success.
    #[tracing::instrument(
        skip(self, session, catalog, spec, cancel),
        fields(job_id = %job_id, worker_id = %self.worker_id)
    )]
    async fn run_spec(
        &self,
        session: &Arc<InferenceSession>,
        catalog: &Arc<Catalog>,
        job_id: &str,
        spec: TrainingSpec,
        cancel: &Arc<AtomicBool>,
        attempt: u32,
    ) -> std::result::Result<TrainedArtifact, WorkerJobError> {
        match spec {
            TrainingSpec::FineTune {
                source,
                columns,
                task,
                common,
                ..
            } => {
                // Re-run the source SQL and rebuild the loader from the persisted
                // columns — the same loader the submitting `fine_tune` built, but
                // reconstructed on this worker with no carryover.
                let batches = self
                    .read_source_columns(session, &source, &columns)
                    .await
                    .map_err(WorkerJobError::from)?;
                let loader = build_training_data_loader(&batches, &columns, task)
                    .map_err(WorkerJobError::from)?;
                let run = FineTuneRun {
                    task,
                    common,
                    loader,
                };
                self.train_fine_tune(session, catalog, job_id, run, cancel, attempt)
                    .await
            }
            TrainingSpec::GraphFineTune {
                sources,
                sample_config,
                common,
            } => {
                // Re-read node/edge sources and re-sample the graph (seeded →
                // deterministic), then train on the text-embedding head.
                let loader = self
                    .reconstruct_graph_loader(session, &sources, sample_config)
                    .await
                    .map_err(WorkerJobError::from)?;
                let run = FineTuneRun {
                    task: ModelTask::TextEmbedding,
                    common,
                    loader,
                };
                self.train_fine_tune(session, catalog, job_id, run, cancel, attempt)
                    .await
            }
            TrainingSpec::ContextPredictor {
                source,
                predictor_spec,
            } => {
                // esc-075 (Phase-4 audit finding 4): this kind never runs
                // `run_fine_tune_blocking`'s measuring probe (it has no
                // `backbone_dtype`/fused-kernel surface to measure at all) —
                // write the self-describing terminal marker up front so the
                // record never reads the submission-time `{"state":
                // "pending"}` marker past this job's eventual terminal
                // status, regardless of whether training below succeeds or
                // fails.
                mark_acceleration_not_applicable(
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
                    .run_context_predictor_training(&source, &predictor_spec, cancel)
                    .await
                    .map_err(|e| classify(cancel, e))
            }
        }
    }

    /// Re-run `SELECT columns FROM source` for a tabular fine-tune.
    ///
    /// A deterministic `ORDER BY` over the **full projected column tuple** pins
    /// the row order. Without it, DataFusion gives no row-order guarantee
    /// (multi-file / multi-partition scans reorder run-to-run), which would
    /// perturb both the batching and the `TargetScaler` μ/σ reduction — breaking
    /// bit-reproducibility. The projected columns are exactly the columns that
    /// feed training, so the order is a *total* function of the trainable data:
    /// the only rows that can tie are byte-identical on every selected column,
    /// and such rows are interchangeable for both batching and the (commutative)
    /// mean/std reduction. DataFusion may permute a tie group arbitrarily, but
    /// that permutation cannot change any training output, so the result is a
    /// pure function of the row multiset. (No engine-wide stable row-identity
    /// column exists on an arbitrary registered source table, so ordering by the
    /// projected tuple is the strongest total key available here.)
    async fn read_source_columns(
        &self,
        session: &Arc<InferenceSession>,
        source: &str,
        columns: &[String],
    ) -> Result<Vec<RecordBatch>> {
        let table_name = session.find_table_name(source)?;
        let quoted: Vec<String> = columns.iter().map(|c| quote_ident(c)).collect();
        let select = quoted.join(", ");
        let order_by = quoted.join(", ");
        let query = format!(
            "SELECT {select} FROM {} ORDER BY {order_by}",
            source_relation(source, &table_name)
        );
        session.sql(&query).await
    }

    /// Re-read the node/edge sources and rebuild the deterministic graph sampler,
    /// then derive the contrastive-pair training loader from it.
    async fn reconstruct_graph_loader(
        &self,
        session: &Arc<InferenceSession>,
        sources: &GraphFineTuneSources,
        sample_config: GraphSampleConfig,
    ) -> Result<TrainingDataLoader> {
        let node_table = session.find_table_name(&sources.node_source)?;
        let node_query = format!(
            "SELECT {}, {} FROM {}",
            quote_ident(&sources.id_column),
            quote_ident(&sources.text_column),
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

        let edge_table = session.find_table_name(&sources.edge_source)?;
        let edge_query = format!(
            "SELECT {}, {} FROM {}",
            quote_ident(&sources.src_column),
            quote_ident(&sources.dst_column),
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
        TrainingDataLoader::from_graph(&sampler)
    }

    /// Load the base model, build the training target, and drive the blocking
    /// LoRA trainer — the shared tail of the two fine-tune kinds. The loop trains
    /// and persists the adapter but writes no terminal status; on a clean return
    /// the worker registers the output-model row through the tenant-pinned
    /// catalog and hands the model id + run metrics to the caller's single
    /// lease-guarded finalization.
    async fn train_fine_tune(
        &self,
        session: &Arc<InferenceSession>,
        catalog: &Arc<Catalog>,
        job_id: &str,
        run: FineTuneRun,
        cancel: &Arc<AtomicBool>,
        attempt: u32,
    ) -> std::result::Result<TrainedArtifact, WorkerJobError> {
        let FineTuneRun {
            task,
            common,
            loader,
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
            WorkerJobError::Failed("Base model does not support embeddings".into())
        })?;
        drop(guard);

        let base_model = common.base_model.clone();
        let cancel_for_classify = Arc::clone(cancel);
        // `common.config` moves into `params` for the blocking trainer; a clone
        // survives here so a training failure can be classified against the
        // config that produced it (`classify_training_oom` names
        // `batch_size`/`max_seq_length`/`backbone_dtype` in the OOM guidance).
        let config_for_error = common.config.clone();
        let params = RunFineTuneParams {
            catalog: Arc::clone(catalog),
            artifact_store: session.artifact_store(),
            artifact_dir: session.inner_config().artifact_dir.clone(),
            job_id: job_id.to_string(),
            worker_id: self.worker_id.clone(),
            attempt,
            base_model: base_model.clone(),
            task,
            config: common.config,
            loader,
            base_model_arc,
            hidden_size,
            device_config: session.device_config().clone(),
            cancel: Arc::clone(cancel),
        };

        // The blocking trainer runs on the blocking pool so it never starves the
        // heartbeat / poll tasks on the async runtime. Panics are caught so a
        // crashing loop still resolves to a terminal classification rather than
        // a wedged `running` row.
        let result = tokio::task::spawn_blocking(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_fine_tune_blocking(params)
            }))
        })
        .await;

        let training = match result {
            Ok(Ok(Ok(training))) => training,
            Ok(Ok(Err(e))) => {
                return Err(classify_training_error(
                    &cancel_for_classify,
                    &config_for_error,
                    e,
                ));
            }
            Ok(Err(payload)) => {
                // A panic on the blocking thread — no `TrainedArtifact` was
                // ever built. This does NOT sweep here (unit 348 F1): it
                // returns `WorkerJobError::Failed`, which propagates
                // unchanged to `run_claimed_job`'s exhaustive `Failed` arm,
                // which sweeps this exact (job_id, worker_id, attempt) range
                // — sweeping here too would be a wasted double sweep of the
                // identical range, not a second reclaim mechanism.
                return Err(WorkerJobError::Failed(format!(
                    "Panic: {}",
                    panic_message(payload.as_ref())
                )));
            }
            Err(join_err) => {
                // Same reasoning as the panic arm immediately above: the
                // blocking task never returned at all, but the resulting
                // `Failed` still funnels through `run_claimed_job`'s single
                // sweep — no sweep call belongs here.
                return Err(WorkerJobError::Failed(format!(
                    "training task join error: {join_err}"
                )));
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
        })
    }
}

/// An RAII guard owning an embedded [`TrainingWorker`]'s background task. On
/// drop it sets the stop flag and aborts the task, so the worker stops claiming
/// new jobs when its owner (the embedded `Session` or the Python
/// `Database`) drops.
///
/// Drop stops the *loop*, not in-flight training: a job already running inside
/// `spawn_blocking` cannot be force-aborted, so aborting the loop task only
/// cancels it at the next `.await` point. A run already on the blocking pool
/// proceeds to completion and writes its terminal status (the lease-guarded
/// finalize) *after* this guard has dropped — detached from the guard's
/// lifetime. The guard therefore bounds when the worker stops taking new work,
/// not when the current job finishes.
pub struct EmbeddedWorker {
    /// `None` once [`Self::stop_and_join`] has taken it — the marker `Drop`
    /// checks to skip its own abort (already gracefully joined, nothing left
    /// to abort). Guarded by a `Mutex` rather than consuming `self` because
    /// [`Self::stop_and_join`] takes `&self`: the owning `Database` binding
    /// wants to signal-and-await without giving up the guard itself (its
    /// `Drop` must still run at the connection's own end of life).
    handle: std::sync::Mutex<Option<tokio::task::JoinHandle<()>>>,
    stop: Arc<AtomicBool>,
}

impl EmbeddedWorker {
    /// Spawn a worker over `session` onto the current runtime, returning the
    /// guard that owns its task. The worker holds a [`Weak`] to the session, so
    /// it never keeps `session` alive; this guard stops it when the owner drops.
    ///
    /// Reads the lease/heartbeat/poll timing from the session's `[training]`
    /// configuration. Returns [`JammiError::Config`] if that timing violates the
    /// worker invariants — in the normal flow `JammiConfig::load` already
    /// validated it, so this only surfaces for a hand-built config.
    pub fn spawn(session: &Arc<InferenceSession>) -> Result<Self> {
        let worker = TrainingWorker::new(session)?;
        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_task = Arc::clone(&stop);
        let handle = tokio::spawn(async move { worker.run_until(stop_for_task).await });
        Ok(Self {
            handle: std::sync::Mutex::new(Some(handle)),
            stop,
        })
    }

    /// Gracefully stop this worker and wait for its loop task to actually
    /// return, rather than `Drop`'s non-blocking signal-and-abort.
    ///
    /// This is the primitive an explicit, deterministic teardown (a bound
    /// `Database::close()`) needs, distinct from `Drop`'s best-effort halt for
    /// an unattended process exit: it signals `stop` and then *awaits* the loop
    /// task rather than aborting it, so a caller blocking on this observes the
    /// worker's actual quiescence, not merely "the signal was sent".
    ///
    /// **Bound on how long this takes to return**, because [`TrainingWorker::
    /// run_until`]'s loop only re-checks `stop` between claim attempts (see its
    /// doc): immediate if the worker is between claim attempts (asleep for at
    /// most `intervals.idle_poll`, default 1s), or the remaining duration of a
    /// job already claimed and running when this is called — such a job runs to
    /// its own terminal state (finalize included) before the loop task returns.
    /// This never force-cancels an in-flight training run; it only stops the
    /// worker from picking up further work and waits for it to notice.
    ///
    /// Idempotent: a second call (concurrent or sequential) finds no handle left
    /// to take and returns `Ok(())` immediately. Takes `&self` rather than
    /// consuming — the caller keeps the guard (and its `Drop`) alive; `Drop`
    /// checks the same `Mutex` and no-ops the abort once this has already taken
    /// the handle.
    pub async fn stop_and_join(&self) -> Result<()> {
        self.stop.store(true, Ordering::Relaxed);
        let taken = self
            .handle
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        let Some(handle) = taken else {
            return Ok(());
        };
        // A join error here is a task panic inside `run_until` (every fallible
        // step inside its loop is already caught and logged, not propagated as
        // a panic) or the task having been aborted by a concurrent `Drop` —
        // either is a genuine defect worth surfacing, not swallowing.
        handle
            .await
            .map_err(|e| JammiError::FineTune(format!("training worker task join error: {e}")))
    }
}

impl Drop for EmbeddedWorker {
    /// Signal the loop to stop and abort its task. This halts claiming of new
    /// jobs; an in-flight `spawn_blocking` training run is not aborted by this —
    /// it runs to completion and writes its terminal status post-drop (see the
    /// type doc).
    ///
    /// No-ops the abort when [`Self::stop_and_join`] already took the handle —
    /// there is nothing left to abort, and aborting a handle that already
    /// returned would be a silent no-op anyway, but the explicit check keeps
    /// the intent legible.
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self
            .handle
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            handle.abort();
        }
    }
}

/// The reconstructed inputs for a LoRA fine-tune run — the per-kind data
/// loader plus the task and base-model/config common bits. Bundled so the
/// shared [`TrainingWorker::train_fine_tune`] tail takes one job-shaped argument
/// rather than a long positional list.
struct FineTuneRun {
    task: ModelTask,
    common: TrainingCommon,
    loader: TrainingDataLoader,
}

/// A successful training run's output, awaiting the worker's unified
/// publish-and-finalize.
///
/// Each kind's training path writes its final artifact files into a local
/// tempdir ([`Self::dir`]) and describes the catalog model row to register
/// ([`Self::register`]) — but does **not** publish to the object store or touch
/// the catalog terminal state. The worker reads the files out of the tempdir,
/// writes them to the artifact store under a unique per-attempt prefix,
/// registers the model row pointing at that prefix, and runs the single
/// lease-guarded finalize CAS — the catalog-pointer-as-commit. `metrics` is the
/// run-metrics JSON the CAS records (the fine-tune loop's loss/step/timing
/// detail; `None` for a kind that records none beyond the terminal flip).
pub struct TrainedArtifact {
    /// Local tempdir holding the final artifact files. Removed on drop, after
    /// the worker has published its contents.
    pub dir: tempfile::TempDir,
    /// The catalog model row to register for this artifact.
    pub register: ModelRegistration,
    /// Run-metrics JSON recorded in the finalize CAS, or `None`.
    pub metrics: Option<String>,
    /// The training loop's retained per-epoch checkpoints (unit 348): each
    /// entry is `(epoch_index, artifact_path)`, where `artifact_path` is the
    /// attempt-unique publish prefix the TRAINER already uploaded that
    /// epoch's full loadable adapter to
    /// (`{job_id}/{worker_id}/{attempt}/checkpoints/epoch_{N}/`). Empty for a
    /// kind that does not checkpoint per epoch (the context-predictor path;
    /// v1 out of scope there). The worker's finalize CAS registers a catalog
    /// row for each entry — never a separate publish step, since the bytes
    /// are already complete by the time this reaches `publish_and_finalize`.
    pub epoch_checkpoints: Vec<(usize, String)>,
}

/// The catalog model-row descriptor a training kind hands the worker's finalize.
///
/// Holds everything `register_model` needs to create the row *except* the served
/// artifact path. The served path is committed solely by the lease-guarded
/// finalize CAS (it takes the published prefix directly), never by this
/// registration — so a loser's or zombie's pre-finalize register can never set
/// the served pointer. The registration creates the row (so a `completed`
/// observer finds it) with the served path left unset for the CAS to fill.
pub struct ModelRegistration {
    /// Deterministic model id (`jammi:fine-tuned:{job_id}`, or the predictor's
    /// configured id) — the catalog upserts on it, so re-registration is
    /// idempotent.
    pub model_id: String,
    /// Catalog version this row registers under. Every training kind
    /// registers its output at `1` today — carried as a field (rather than
    /// hardcoded in [`Self::as_params`]) so `TrainingWorker::publish_and_finalize`
    /// can pass the SAME version into `finalize_training_job`'s version
    /// predicate that `register_model` used to create the row, closing the
    /// bare-`name` CAS-clobber gap (B5, unit 348).
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
    /// Build the [`jammi_db::catalog::model_repo::RegisterModelParams`] for this
    /// row, leaving `artifact_path` unset — the served path is committed by the
    /// finalize CAS, not by registration.
    pub fn as_params(&self) -> jammi_db::catalog::model_repo::RegisterModelParams<'_> {
        jammi_db::catalog::model_repo::RegisterModelParams {
            model_id: &self.model_id,
            version: self.version,
            model_type: self.model_type,
            backend: "candle",
            task: self.task,
            base_model_id: self.base_model_id.as_deref(),
            artifact_path: None,
            config_json: self.config_json.as_deref(),
        }
    }
}

/// Read every regular file directly under `dir` into `(name, bytes)` and write
/// them to the artifact store under the unique per-attempt prefix
/// `{job_id}/{worker_id}/{attempt}`, returning the prefix `StorageUrl` the model
/// row records. The three segments are jointly unique per attempt (`job_id` is
/// the PK, `worker_id` distinguishes a lost-lease worker from its re-claimer,
/// `attempt` distinguishes a reclaimed re-run), so no two attempts ever target
/// the same prefix and no object is overwritten. Only top-level files are
/// published (the trainer's checkpoint subdirectories are training scratch, not
/// part of the served artifact).
async fn publish_artifact(
    store: &ArtifactStore,
    job_id: &str,
    worker_id: &str,
    attempt: &str,
    dir: &tempfile::TempDir,
) -> Result<jammi_db::storage::StorageUrl> {
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
        .put_artifact(&[job_id, worker_id, attempt], &files)
        .await
}

/// The terminal classification of a worker's run of one job.
enum WorkerJobError {
    /// The lease was lost mid-training; the job is left `running` for reclaim.
    Cancelled,
    /// The job failed for a real reason; record it as `failed` + the message.
    Failed(String),
}

impl From<JammiError> for WorkerJobError {
    fn from(e: JammiError) -> Self {
        WorkerJobError::Failed(e.to_string())
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
        WorkerJobError::Failed(e.to_string())
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
// Reconstruction helpers (the data-loading + blocking-training tail moved off
// the submit path: the worker is their only consumer now).
// =========================================================================

/// Extract all string values from an Arrow column, or `None` when the column
/// is not honestly readable as text.
///
/// DataFusion 52+ returns Parquet string columns as `Utf8View` by default;
/// older versions returned `Utf8` or `LargeUtf8`. Dictionary-encoded variants
/// are also possible. Fast paths cover the three common types; the `cast`
/// fallback handles everything else.
///
/// # Two refusals the cast fallback cannot be trusted to make (family D)
///
/// `arrow::compute::cast`'s DEFAULT options are `safe: true`, which means a
/// value the target type cannot represent becomes NULL rather than an error.
/// Combined with `StringArray::value(i)` — which returns `""` for a null slot
/// rather than failing — the fallback silently turned unreadable cells into
/// empty strings:
///
/// - A **binary** column (an image or audio triplet submitted under a TEXT
///   task) cast cell-by-cell into NULLs, and every training row became the
///   empty string. The job then completed, published an adapter, and reported
///   success — a fine-tune of a text tower on nothing at all. Bytes are not
///   text: the binary families are refused OUTRIGHT here, so the caller gets
///   the typed, task-naming schema error its caller already raises.
/// - Any OTHER column whose cast introduces a null where the source had a
///   value is refused for the same reason — the empty string is a fabricated
///   input, not a reading of the caller's data.
///
/// A column that was ALREADY null keeps its historical `""` reading: that is a
/// pre-existing null-handling contract of the text path, not a value this
/// function invented.
fn extract_string_column(col: &dyn arrow::array::Array) -> Option<Vec<String>> {
    use arrow::array::{Array, LargeStringArray, StringArray, StringViewArray};
    use arrow::datatypes::DataType;

    if let Some(a) = col.as_any().downcast_ref::<StringViewArray>() {
        return Some((0..a.len()).map(|i| a.value(i).to_string()).collect());
    }
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        return Some((0..a.len()).map(|i| a.value(i).to_string()).collect());
    }
    if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
        return Some((0..a.len()).map(|i| a.value(i).to_string()).collect());
    }
    if matches!(
        col.data_type(),
        DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView
            | DataType::FixedSizeBinary(_)
    ) {
        return None;
    }
    let casted = arrow::compute::cast(col, &DataType::Utf8).ok()?;
    let a = casted.as_any().downcast_ref::<StringArray>()?;
    if (0..a.len()).any(|i| a.is_null(i) && !col.is_null(i)) {
        return None;
    }
    Some((0..a.len()).map(|i| a.value(i).to_string()).collect())
}

/// Extract a binary column into owned byte vectors, accepting the Arrow binary
/// families DataFusion produces for an audio-bytes column
/// (`Binary`/`LargeBinary`/`BinaryView`). Returns `None` for any other type so
/// the caller can surface a typed schema error.
fn extract_binary_column(col: &dyn arrow::array::Array) -> Option<Vec<Vec<u8>>> {
    use arrow::array::{Array, BinaryArray, BinaryViewArray, LargeBinaryArray};

    if let Some(a) = col.as_any().downcast_ref::<BinaryArray>() {
        return Some((0..a.len()).map(|i| a.value(i).to_vec()).collect());
    }
    if let Some(a) = col.as_any().downcast_ref::<LargeBinaryArray>() {
        return Some((0..a.len()).map(|i| a.value(i).to_vec()).collect());
    }
    if let Some(a) = col.as_any().downcast_ref::<BinaryViewArray>() {
        return Some((0..a.len()).map(|i| a.value(i).to_vec()).collect());
    }
    None
}

/// Why a numeric column could not be read into clean `f32` targets.
enum NumericColumnError {
    /// The column's Arrow type is not numeric (and the cast fallback failed).
    NotNumeric,
    /// A null target at the cited row index. Rejected rather than coerced to
    /// `0.0`, which would silently corrupt the scaler's μ/σ.
    Null(usize),
    /// A `NaN` target at the cited row index (float columns only). Rejected for
    /// the same reason as a null.
    Nan(usize),
}

/// Extract a numeric column into `Vec<f32>`, accepting the Arrow numeric
/// families DataFusion emits for a regression `target` column. Integer targets
/// (e.g. an `int64` year) are common, so the fast paths cover
/// `Int64`/`Int32`/`Float64`/`Float32`; the final `cast` fallback handles the
/// remaining numeric types (`UInt*`, `Int16`, `Decimal`, …) so a target's exact
/// Arrow width never decides whether the fine-tune is reachable.
///
/// **Null/NaN rejection is load-bearing.** `Array::value(i)` on a null slot
/// returns a zero default rather than erroring, which would silently corrupt
/// the scaler's μ/σ. A null or `NaN` target therefore returns a typed error
/// citing the row, never a coerced `0.0`.
fn extract_numeric_column(
    col: &dyn arrow::array::Array,
) -> std::result::Result<Vec<f32>, NumericColumnError> {
    use arrow::array::{Array, Float32Array, Float64Array, Int32Array, Int64Array};
    use arrow::datatypes::DataType;

    // A string/binary `target` is a schema mistake, not numeric data — reject it
    // as "not numeric" rather than letting the Float64 cast turn unparseable
    // strings into nulls (which would surface a misleading per-row null error).
    if matches!(
        col.data_type(),
        DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView
            | DataType::Boolean
            | DataType::Null
    ) {
        return Err(NumericColumnError::NotNumeric);
    }

    // Reject a null in any slot up front; `value(i)` would otherwise return a
    // garbage default for it.
    if let Some(i) = (0..col.len()).find(|&i| col.is_null(i)) {
        return Err(NumericColumnError::Null(i));
    }

    let floats: Vec<f32> = if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
        (0..a.len()).map(|i| a.value(i) as f32).collect()
    } else if let Some(a) = col.as_any().downcast_ref::<Int32Array>() {
        (0..a.len()).map(|i| a.value(i) as f32).collect()
    } else if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
        (0..a.len()).map(|i| a.value(i) as f32).collect()
    } else if let Some(a) = col.as_any().downcast_ref::<Float32Array>() {
        (0..a.len()).map(|i| a.value(i)).collect()
    } else {
        // Fallback: cast through Float64 for the remaining numeric families. A
        // cast failure means the column is not numeric.
        let casted = arrow::compute::cast(col, &DataType::Float64)
            .map_err(|_| NumericColumnError::NotNumeric)?;
        let a = casted
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or(NumericColumnError::NotNumeric)?;
        // The cast can introduce nulls (e.g. an unrepresentable value); reject
        // them with the same per-row contract.
        if let Some(i) = (0..a.len()).find(|&i| a.is_null(i)) {
            return Err(NumericColumnError::Null(i));
        }
        (0..a.len()).map(|i| a.value(i) as f32).collect()
    };

    // A NaN target (float columns only) would corrupt the scaler; reject it
    // citing the row, mirroring the null contract.
    if let Some(i) = floats.iter().position(|v| v.is_nan()) {
        return Err(NumericColumnError::Nan(i));
    }
    Ok(floats)
}

/// Build a [`TrainingDataLoader`] from query result batches.
///
/// `task` selects how `anchor`/`positive`/`negative` triplet columns are read:
/// an image or audio embedding task reads them as encoded MEDIA bytes; every
/// other task reads them as text. The column names are identical across
/// modalities (the triplet shape is the same) — only the cell decoding
/// differs, so the caller's chosen task is the discriminator, not a parallel
/// set of column names, and not a byte-header sniff (an encoded WAV and an
/// encoded PNG are both binary blobs).
fn build_training_data_loader(
    batches: &[RecordBatch],
    columns: &[String],
    task: ModelTask,
) -> Result<TrainingDataLoader> {
    let col_names: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();

    let has_contrastive = col_names.contains(&"text_a")
        && col_names.contains(&"text_b")
        && col_names.contains(&"score");
    let has_triplet = col_names.contains(&"anchor")
        && col_names.contains(&"positive")
        && col_names.contains(&"negative");
    // Pairs = anchor + positive with no negative column. In-batch negatives
    // (MultipleNegativesRanking) supply the contrast, so `negative` is absent.
    let has_pairs = col_names.contains(&"anchor")
        && col_names.contains(&"positive")
        && !col_names.contains(&"negative");
    let has_classification = col_names.contains(&"text") && col_names.contains(&"label");
    // Regression shares the `text` anchor with classification but reads a
    // numeric `target` column instead of a string `label`. The two text-outcome
    // formats are disambiguated by `task`, not by column names, exactly as the
    // audio-triplet path is task-gated below: the regression arm is gated on
    // `task == Regression` and ordered before classification, and classification
    // is gated on `task != Regression`. So `task=regression` is authoritative —
    // it can never fall into the classification path (which would gather a
    // numeric outcome as a class index and CUDA-assert), and a `label`-only
    // source under `task=regression` produces a typed "needs a numeric target"
    // error rather than a device-side assert.
    let has_regression = col_names.contains(&"text") && col_names.contains(&"target");

    if has_triplet && matches!(task, ModelTask::AudioEmbedding | ModelTask::ImageEmbedding) {
        return build_media_triplet_loader(batches, task);
    }

    if has_contrastive {
        let mut rows = Vec::new();
        for batch in batches {
            let a_col = batch
                .column_by_name("text_a")
                .ok_or_else(|| JammiError::FineTune("Missing column 'text_a'".into()))?;
            let b_col = batch
                .column_by_name("text_b")
                .ok_or_else(|| JammiError::FineTune("Missing column 'text_b'".into()))?;
            let s_col = batch
                .column_by_name("score")
                .ok_or_else(|| JammiError::FineTune("Missing column 'score'".into()))?;

            let a_vals = extract_string_column(a_col.as_ref())
                .ok_or_else(|| JammiError::FineTune("'text_a' is not a string column".into()))?;
            let b_vals = extract_string_column(b_col.as_ref())
                .ok_or_else(|| JammiError::FineTune("'text_b' is not a string column".into()))?;
            let s_arr = s_col
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .map(|arr| {
                    (0..arr.len())
                        .map(|i| arr.value(i) as f32)
                        .collect::<Vec<_>>()
                })
                .or_else(|| {
                    s_col
                        .as_any()
                        .downcast_ref::<arrow::array::Float32Array>()
                        .map(|arr| (0..arr.len()).map(|i| arr.value(i)).collect())
                })
                .ok_or_else(|| JammiError::FineTune("'score' is not a float column".into()))?;

            for (i, &score) in s_arr.iter().enumerate().take(batch.num_rows()) {
                rows.push((a_vals[i].clone(), b_vals[i].clone(), score));
            }
        }
        Ok(TrainingDataLoader::from_contrastive(rows))
    } else if has_triplet {
        let mut rows = Vec::new();
        for batch in batches {
            let schema_info = || {
                batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| format!("{}:{}", f.name(), f.data_type()))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            let anchor_vals = batch
                .column_by_name("anchor")
                .and_then(|c| extract_string_column(c.as_ref()))
                .ok_or_else(|| {
                    JammiError::FineTune(format!(
                        "Missing/invalid 'anchor' column: task {task} expects text columns; for \
                         image/audio triplets submit task=image_embedding/audio_embedding. \
                         Batch schema: [{}]",
                        schema_info()
                    ))
                })?;
            let pos_vals = batch
                .column_by_name("positive")
                .and_then(|c| extract_string_column(c.as_ref()))
                .ok_or_else(|| {
                    JammiError::FineTune(format!(
                        "Missing/invalid 'positive' column: task {task} expects text columns; for \
                         image/audio triplets submit task=image_embedding/audio_embedding. \
                         Batch schema: [{}]",
                        schema_info()
                    ))
                })?;
            let neg_vals = batch
                .column_by_name("negative")
                .and_then(|c| extract_string_column(c.as_ref()))
                .ok_or_else(|| {
                    JammiError::FineTune(format!(
                        "Missing/invalid 'negative' column: task {task} expects text columns; for \
                         image/audio triplets submit task=image_embedding/audio_embedding. \
                         Batch schema: [{}]",
                        schema_info()
                    ))
                })?;

            for i in 0..batch.num_rows() {
                rows.push((
                    anchor_vals[i].clone(),
                    pos_vals[i].clone(),
                    neg_vals[i].clone(),
                ));
            }
        }
        Ok(TrainingDataLoader::from_triplets(rows))
    } else if has_pairs {
        let mut rows = Vec::new();
        for batch in batches {
            let schema_info = || {
                batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| format!("{}:{}", f.name(), f.data_type()))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            let anchor_vals = batch
                .column_by_name("anchor")
                .and_then(|c| extract_string_column(c.as_ref()))
                .ok_or_else(|| {
                    JammiError::FineTune(format!(
                        "Missing/invalid 'anchor' column. Batch schema: [{}]",
                        schema_info()
                    ))
                })?;
            let pos_vals = batch
                .column_by_name("positive")
                .and_then(|c| extract_string_column(c.as_ref()))
                .ok_or_else(|| {
                    JammiError::FineTune(format!(
                        "Missing/invalid 'positive' column. Batch schema: [{}]",
                        schema_info()
                    ))
                })?;
            for i in 0..batch.num_rows() {
                rows.push((anchor_vals[i].clone(), pos_vals[i].clone()));
            }
        }
        Ok(TrainingDataLoader::from_pairs(rows))
    } else if task == ModelTask::Regression {
        // Regression: a string `text` column and a numeric `target` column. The
        // target is read into `f32` (handling int64/float64/float32/… via
        // `extract_numeric_column`); nulls and NaNs are rejected citing the row
        // rather than coerced, since a coerced `0.0` would silently corrupt the
        // scaler's μ/σ. A `task=regression` request with no usable `target`
        // column is a typed error here, never a fall-through to classification.
        if !has_regression {
            return Err(JammiError::FineTune(format!(
                "task=regression needs a string 'text' column and a numeric 'target' column, \
                 but the projected columns are {col_names:?}. (Classification's string 'label' \
                 is distinct: name the numeric outcome column 'target'.)"
            )));
        }
        let mut rows = Vec::new();
        for batch in batches {
            let text_vals = batch
                .column_by_name("text")
                .and_then(|c| extract_string_column(c.as_ref()))
                .ok_or_else(|| JammiError::FineTune("Missing/invalid 'text' column".into()))?;
            let target_col = batch
                .column_by_name("target")
                .ok_or_else(|| JammiError::FineTune("Missing 'target' column".into()))?;
            let target_vals = extract_numeric_column(target_col.as_ref()).map_err(|e| {
                JammiError::FineTune(match e {
                    NumericColumnError::NotNumeric => format!(
                        "regression 'target' is not a numeric column (its Arrow type is {})",
                        target_col.data_type()
                    ),
                    NumericColumnError::Null(i) => format!(
                        "regression 'target' has a null at row {i}; a null target cannot be \
                         coerced (it would corrupt the scaler) — remove or fill the row"
                    ),
                    NumericColumnError::Nan(i) => format!(
                        "regression 'target' has a NaN at row {i}; a NaN target cannot be used \
                         (it would corrupt the scaler) — remove or fix the row"
                    ),
                })
            })?;
            for i in 0..batch.num_rows() {
                rows.push((text_vals[i].clone(), target_vals[i]));
            }
        }
        Ok(TrainingDataLoader::from_regression(rows))
    } else if has_classification {
        let mut label_set = std::collections::BTreeSet::new();
        let mut rows = Vec::new();
        for batch in batches {
            let text_vals = batch
                .column_by_name("text")
                .and_then(|c| extract_string_column(c.as_ref()))
                .ok_or_else(|| JammiError::FineTune("Missing/invalid 'text' column".into()))?;
            let label_vals = batch
                .column_by_name("label")
                .and_then(|c| extract_string_column(c.as_ref()))
                .ok_or_else(|| JammiError::FineTune("Missing/invalid 'label' column".into()))?;
            for i in 0..batch.num_rows() {
                label_set.insert(label_vals[i].clone());
                rows.push((text_vals[i].clone(), label_vals[i].clone()));
            }
        }
        let label_to_idx: std::collections::HashMap<String, u32> = label_set
            .iter()
            .enumerate()
            .map(|(i, l)| (l.clone(), i as u32))
            .collect();
        let num_classes = label_to_idx.len();
        let indexed_rows: Vec<(String, u32)> = rows
            .into_iter()
            .map(|(text, label)| {
                let idx = label_to_idx[&label];
                (text, idx)
            })
            .collect();
        Ok(TrainingDataLoader::from_classification(
            indexed_rows,
            num_classes,
        ))
    } else {
        Err(JammiError::FineTune(format!(
            "Cannot detect training format from columns: {col_names:?}. \
             Expected contrastive (text_a, text_b, score), triplet (anchor, positive, negative), \
             pairs (anchor, positive), classification (text, label), or regression \
             (text, target) with task=regression. For image/audio triplets, use the \
             same (anchor, positive, negative) columns with binary cells and \
             task=image_embedding/audio_embedding."
        )))
    }
}

/// Build a MEDIA-triplet loader: read `anchor`/`positive`/`negative` as
/// encoded binary columns (audio clips or images, per `task`). Shares the
/// triplet column shape with the text path; only the cell type differs
/// (binary blobs vs strings).
fn build_media_triplet_loader(
    batches: &[RecordBatch],
    task: ModelTask,
) -> Result<TrainingDataLoader> {
    let mut rows = Vec::new();
    for batch in batches {
        let schema_info = || {
            batch
                .schema()
                .fields()
                .iter()
                .map(|f| format!("{}:{}", f.name(), f.data_type()))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let anchor_vals = batch
            .column_by_name("anchor")
            .and_then(|c| extract_binary_column(c.as_ref()))
            .ok_or_else(|| {
                JammiError::FineTune(format!(
                    "Missing/invalid binary 'anchor' column for media triplets (task \
                     {task}). Batch schema: [{}]",
                    schema_info()
                ))
            })?;
        let pos_vals = batch
            .column_by_name("positive")
            .and_then(|c| extract_binary_column(c.as_ref()))
            .ok_or_else(|| {
                JammiError::FineTune(format!(
                    "Missing/invalid binary 'positive' column for media triplets (task \
                     {task}). Batch schema: [{}]",
                    schema_info()
                ))
            })?;
        let neg_vals = batch
            .column_by_name("negative")
            .and_then(|c| extract_binary_column(c.as_ref()))
            .ok_or_else(|| {
                JammiError::FineTune(format!(
                    "Missing/invalid binary 'negative' column for media triplets (task \
                     {task}). Batch schema: [{}]",
                    schema_info()
                ))
            })?;

        for i in 0..batch.num_rows() {
            rows.push((
                anchor_vals[i].clone(),
                pos_vals[i].clone(),
                neg_vals[i].clone(),
            ));
        }
    }
    Ok(TrainingDataLoader::from_media_triplets(rows))
}

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

/// Record a terminal `Failed` status for a training job this worker owns,
/// surfacing the cause via the catalog metrics blob so a `TrainingJob::wait()`
/// observer sees the failure instead of an indefinite `running` state.
///
/// The write is lease-guarded (the failure peer of the finalize CAS): it lands
/// only while this worker still holds the lease (`claimed_by = worker_id AND
/// status = 'running'`). A worker that lost its lease before failing does not
/// stamp `failed` over a job the re-claiming worker is running — that case is
/// left for the new owner (logged at debug).
async fn record_failed(catalog: &Arc<Catalog>, job_id: &str, worker_id: &str, msg: String) {
    let metrics = serde_json::json!({ "error_message": msg }).to_string();
    match catalog
        .fail_training_job(job_id, worker_id, Some(&metrics))
        .await
    {
        Ok(true) => {}
        Ok(false) => {
            tracing::debug!(
                job_id = %job_id,
                worker = %worker_id,
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
/// shared write path every esc-075 acceleration-report site uses, whether it
/// carries a measured `"determined"` payload
/// (`compute_and_persist_acceleration_report`) or one of the terminal
/// markers below for a job kind/path that never reaches the measuring probe
/// at all (Phase-4 adversarial-audit finding 4).
///
/// **Swallowing is deliberate, and it is covered downstream.** A `false`
/// (lease-guard miss: the lease was lost, or this attempt's `attempt` no
/// longer matches the row's `attempts`) or an `Err` here must never fail
/// training. The consequence — a job that goes on to complete with its
/// submission-time `{"state":"pending"}` marker never overwritten — is
/// retired at the catalog's terminal edge, not compensated for here; see
/// [`TrainingWorker::run_claimed_job`]'s "The SUCCESS path is not exempt"
/// section.
async fn persist_acceleration_report(
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
                "esc-075: record_acceleration_report's lease guard did not match (lease lost or \
                 stale attempt); continuing without a persisted acceleration report"
            );
        }
        Err(e) => {
            tracing::warn!(
                job_id = %job_id,
                worker_id = %worker_id,
                attempt,
                error = %e,
                "esc-075: record_acceleration_report failed; continuing without a persisted \
                 acceleration report"
            );
        }
    }
}

/// esc-075 (Phase-4 adversarial-audit finding 4 — "PENDING-FOREVER"):
/// `TrainingSpec::ContextPredictor` jobs never route through
/// `run_fine_tune_blocking`'s measuring probe at all (`run_spec`'s
/// `ContextPredictor` arm calls `InferenceSession::run_context_predictor_training`
/// directly, never `train_fine_tune`) — without this, the record's
/// `acceleration_report` would carry the submission-time `{"state":
/// "pending"}` marker FOREVER, even after the job reaches a terminal
/// `completed`/`failed` status, which the tri-state contract's own
/// definition of "pending" (submitted, no claimant has computed a
/// determination YET) does not describe. Writes the self-describing
/// `{"state":"not_applicable","reason":"context_predictor"}` marker before
/// training starts, under the SAME lease-guarded
/// `record_acceleration_report` every other esc-075 write uses.
async fn mark_acceleration_not_applicable(
    catalog: &Arc<Catalog>,
    job_id: &str,
    worker_id: &str,
    attempt: u32,
    reason: &str,
) {
    persist_acceleration_report(
        catalog,
        job_id,
        worker_id,
        attempt,
        &serde_json::json!({"state": "not_applicable", "reason": reason}).to_string(),
    )
    .await;
}

/// esc-075 (Phase-4 adversarial-audit finding 4 — "PENDING-FOREVER"): a job
/// that fails in `run_claimed_job` BEFORE the device is ever resolved (no
/// `training_spec` at all, or an undeserialisable one — both happen before
/// `run_spec`/`run_fine_tune_blocking` are ever reached) never runs the
/// measuring probe either, and would otherwise carry the submission-time
/// `{"state":"pending"}` marker forever past this job's terminal `failed`
/// status. Writes the self-describing
/// `{"state":"undetermined","reason":"failed_before_device_resolution"}`
/// marker, under the SAME lease-guarded `record_acceleration_report` every
/// other esc-075 write uses — called BEFORE `record_failed` so the write
/// still observes this attempt's `running` status (the lease guard requires
/// it; `record_failed` flips the row to `failed` immediately after).
///
/// Campaign #446 finding 1 made this pre-mark REDUNDANT-but-preferred rather
/// than load-bearing: `Catalog::fail_training_job` now retires ANY
/// still-`pending` report to
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
/// `pending`-valued `CASE`), so this is inert — but a future equality check
/// against a marker with more than one key must compare parsed values, not
/// bytes. The single-key `{"state":"pending"}` marker is the one exception,
/// and it has exactly ONE producer (jammi-db's own `INSERT` const), which is
/// what makes the catalog edge's byte match on it sound.
async fn mark_acceleration_undetermined(
    catalog: &Arc<Catalog>,
    job_id: &str,
    worker_id: &str,
    attempt: u32,
) {
    persist_acceleration_report(
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
    /// checkpoints under (`{job_id}/{worker_id}/{attempt}/checkpoints/…`,
    /// unit 348).
    attempt: u32,
    base_model: String,
    task: ModelTask,
    config: FineTuneConfig,
    loader: TrainingDataLoader,
    base_model_arc: Arc<crate::model::LoadedModel>,
    hidden_size: usize,
    device_config: DeviceConfig,
    cancel: Arc<AtomicBool>,
}

/// Run LoRA fine-tuning in a blocking context, checking `cancel` at every epoch
/// boundary. Reconstructs the training target (projection head or encoder
/// adapters) and drives the trainer. The loop trains and persists the adapter
/// but writes no terminal status — the worker registers the output model and
/// runs the lease-guarded finalize after this returns. Returns the
/// [`crate::fine_tune::trainer::TrainingResult`] (adapter path + run metrics)
/// the worker threads into that finalization.
fn run_fine_tune_blocking(
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
        base_model,
        task,
        config,
        loader: data_loader,
        base_model_arc,
        hidden_size,
        device_config,
        cancel,
    } = params;

    let device = crate::model::backend::candle::select_device(&device_config)?;
    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let target = if config.target_modules.is_empty() {
        let head = if task == ModelTask::Classification {
            let num_classes = match data_loader.format() {
                crate::fine_tune::data::TrainingFormat::Classification { num_classes } => {
                    num_classes
                }
                _ => {
                    return Err(JammiError::FineTune(
                        "Classification task requires classification training data format".into(),
                    ))
                }
            };
            crate::fine_tune::lora::build_classification_head(
                hidden_size,
                num_classes,
                &config,
                &varmap,
                &vb,
            )?
        } else if task == ModelTask::Regression {
            let output_dim = match config.regression_loss.unwrap_or_default() {
                crate::fine_tune::RegressionLoss::Pinball => config.quantile_levels.len(),
                _ => 2,
            };
            crate::fine_tune::lora::build_distribution_head(
                hidden_size,
                output_dim,
                &config,
                &varmap,
                &vb,
            )?
        } else {
            crate::fine_tune::lora::build_projection_head(hidden_size, &config, &varmap, &vb)?
        };
        // esc-075: `backbone_dtype` never takes effect on this arm (see
        // `validate_backbone_precision`'s doc), so there is no encoder to probe
        // fused-op admission against — the report still names the resolved
        // device + compiled capabilities, with an empty `ops`/honest "no probe
        // attempted" `flash` reason (never a fabricated per-op measurement for
        // an arm that never ran one — see `flash_report_no_probe_attempted`'s
        // doc, Phase-4 adversarial-audit finding 3).
        compute_and_persist_acceleration_report(
            &catalog,
            &job_id,
            &worker_id,
            attempt,
            &device,
            config.backbone_dtype,
            None,
            None,
        );
        crate::fine_tune::target::TrainingTarget::ProjectionHead { head }
    } else {
        let (mut encoder, adapter_cfg) = build_encoder_adapters(
            &base_model,
            &catalog,
            &artifact_store,
            &config,
            task,
            &varmap,
            &device,
        )?;
        // esc-075: right after `build_encoder_adapters` (which calls
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

    // Discover a durable resume checkpoint for this job. If one exists (a prior
    // attempt completed at least one epoch boundary before dying), the trainer
    // restores weights + optimizer moments + scaler + dropout positions and
    // continues from `last_completed + 1`; if none exists, it trains from scratch
    // as today. The discovery never perturbs the publish/serving path — the
    // resume prefix (`{job_id}/_resume/`) is a crash-recovery side channel.
    let resume = discover_resume(&artifact_store, &job_id, &device)?;

    let mut builder = crate::fine_tune::trainer::TrainingLoopBuilder::new(target, varmap, config)
        .base_model(base_model_arc)
        // The run's declared task is the trainer's MODALITY discriminator for
        // the media paths (`TrainingLoop::task`) — the same `task` this
        // function already gated the data loader and the encoder dispatch on.
        .task(task)
        .job_id(job_id)
        .worker_id(worker_id)
        .attempt(attempt.to_string())
        .catalog(Arc::clone(&catalog))
        .artifact_dir(artifact_dir)
        .device(device.clone())
        .cancel(cancel)
        .artifact_store(Arc::clone(&artifact_store));
    if let Some(restored) = resume {
        builder = builder.resume(restored);
    }
    let mut training_loop = builder.build()?;

    training_loop.run(&data_loader)
}

/// Fetch and load a job's durable resume checkpoint, if any. `None` when no
/// checkpoint exists yet (from-scratch). A present-but-corrupt bundle surfaces as
/// a hard error from the artifact store, not a silent from-scratch restart.
fn discover_resume(
    store: &Arc<ArtifactStore>,
    job_id: &str,
    device: &candle_core::Device,
) -> Result<Option<crate::fine_tune::resume::RestoredCheckpoint>> {
    let Some(local) =
        tokio::runtime::Handle::current().block_on(store.fetch_resume_checkpoint(job_id))?
    else {
        return Ok(None);
    };
    crate::fine_tune::resume::load_bundle(local.dir(), device).map(Some)
}

/// Refuse a backbone precision the resolved device cannot compute at.
///
/// BF16 is a GPU-tier precision. candle's CPU matmul accepts `F16 | F32 | F64`
/// only and returns "unsupported dtype BF16 for op matmul" otherwise, so a BF16
/// backbone on CPU fails at the first frozen linear of the first forward —
/// after the job has been claimed, the backbone downloaded, and the adapter
/// built.
///
/// This became reachable when the LoRA arm stopped re-materialising the frozen
/// weight in F32 on every forward. That upcast was masking the limitation:
/// with every ModernBERT linear LoRA-targeted, no `Frozen` arm survived to hit
/// the unsupported matmul, so a CPU BF16 fine-tune "worked" only by silently
/// discarding the precision it was asked for. Honouring `backbone_dtype` means
/// the unsupported combination has to be refused rather than quietly ignored.
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
// esc-075: claim-time, per-job acceleration report.
//
// A compute precision the public API accepts (`ComputePrecision`) is either
// accelerated by the fused kernels or it silently runs the eager composition
// — and, before this, the ONLY signal was a `tracing::warn` deduplicated for
// the life of the PROCESS (`jammi_kernels::admission::warn_fallback_once`),
// so a second f16 job read the same silence as a first, accelerated one. This
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
// warn list instead (the pre-#446 shape) attributed the most recent
// DIFFERENT predicate to a job whose own miss repeated an already-burned
// pair — see `reason_from_probe_window`'s doc. A `holds: false` op with no
// entry in its own window gets the honest `"reason_unavailable"`, never a
// guess.
//
// The probe runs a real forward pass PLUS one backward + optimizer step
// (Phase-4 adversarial-audit finding 2, campaign #443): `layer_norm`,
// `rope`, `softmax`, `geglu`, and `attention_block` dispatch during the
// forward pass; `dropout`/`low_rank_residual_linear` (both read from the
// SAME `lora_linear_fused` registry key — `crates/jammi-lora/src/
// lora_linear.rs:37`'s own doc: the separate `lora_dropout` counter is
// "Permanently `{fused: 0, eager: 0}` today", superseded) ALSO dispatch
// during the forward pass (`LoraLinear::forward`, `lora_linear.rs:752-812`
// — confirmed by reading it: a prior revision of this comment wrongly
// claimed these needed backward, corrected during Phase-4 remediation,
// advisory 8); and `LowRankResidualLinear`'s own backward-time cast-boundary
// epilogue kernels (`cast_scale`/`cast_add`, `crates/jammi-kernels/src/ops/
// low_rank_residual_linear.rs`'s `bwd`) dispatch ONLY during backward, so
// a forward-only probe could never honestly claim to have measured them.
//
// **Which ops this probe can attribute is ONE static fact**, not a list
// re-typed here: `jammi_kernels::admission::PROBED_OPS` (campaign #446
// finding 2). A prior revision of this comment claimed `rope_positions`,
// `cast_scale` and `scaled_cast_add` all "have NO admit/admit_cascade call
// site anywhere in this workspace". That was FALSE for two of the three
// and is corrected here: `cast_scale` and `cast_add` DO admit, under
// dtype-resolved registry keys (`cast_scale_bf16_f32`/`cast_scale_f16_f32`,
// `cast_add_bf16`/`cast_add_f16` — `low_rank_residual_linear.rs:800,814,899,
// 911`), which is why an f16 job's report was structurally unable to name its
// own cast epilogue while a bf16 job's named only half of it. `rope_positions`
// and `scaled_cast_add` genuinely have no admission gate: each is a bare
// launcher call from inside an already-admitted parent's fused arm
// (`ProbedOpKind::InternalSubkernel`), so no probe can read a delta for
// them and they are OMITTED from `ops` — never fabricated as a `holds`
// either way (K-series: ship the honest negative, not a vacuous positive).
// Every kernel this build compiles is now in one of those two positions:
// campaign #446 W2 deleted the workspace's last kernel that had neither an
// admission gate nor an admitted parent — a measured CUDA census, not a
// judgement call:
// `crates/jammi-kernels/artifacts/cuda-runs/2026-09-01-axpy-census-bdeb80c-a100-pcie.json`
// — so there is no longer a class of compiled kernel this report is
// structurally unable to say anything about.
//
// The report's `ops` CANDIDATE key set is therefore a pure function of the
// job's backbone dtype class (`PROBED_OPS` filtered by
// `ProbedOp::registry_keys_for`), never a function of what else this process
// happened to run — which is what `jammi_kernels::admission::snapshot_all()`
// would have given (it reflects only ops looked up at least once, so its key
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
// (Phase-4 finding 3: distinct from a probe that WAS attempted and failed); a
// reached-but-declined cascade reads its verbatim predicate key back out of
// THIS probe's own probe-capture window — see the next paragraph — falling
// back to the coarser, honestly-labelled `"capability_or_domain_miss"` only
// when that window carries no entry for the decline.
//
// **The BERT/DistilBERT case, named honestly (issue #462/#463 follow-up):**
// both families' training forward always calls
// `attention_cascade::training_attention_cascade` with `flash: &FlashDecision::
// Declined { outcome: CapabilityMiss, reason: "flash_transport_not_wired" }`
// — the ONE reason value either family's `FlashDecision::Declined` ever
// carries, because neither wires the encoder-boundary flash transport
// protocol — see `BERT never wires the encoder-boundary flash transport` (`crates/jammi-encoders/src/bert.rs:428-420`)
// and the sibling `FlashDecision::Declined` (`crates/jammi-encoders/src/distilbert.rs:331-324`). `admit_cascade`
// (`crates/jammi-kernels/src/admission.rs:403-453`) now records every decline
// — disabled, `DomainMiss`, and `CapabilityMiss` alike — into the SAME
// thread-local probe-capture sink `admit_inner` uses
// (`record_probe_miss(op, predicate_name)`,
// `crates/jammi-kernels/src/admission.rs:427,437`), not just an atomic
// increment on `CascadeDispatchCounters`. [`flash_report`] reads that entry
// back through `jammi_kernels::admission::probe_capture_reason_for(window,
// "attention_block_flash")` on a decline, exactly the way
// [`reason_from_probe_window`] reads it for a two-arm op — so a BERT/
// DistilBERT job's `flash` field now reads back verbatim as
// `"flash_transport_not_wired"` instead of the coarse
// `"capability_or_domain_miss"`. The coarse value survives only as
// [`flash_report`]'s fallback for a decline whose window happens to carry no
// entry (the window-attribution causes [`REASON_UNAVAILABLE`] already
// documents) — never fabricated in its place.
//
// **Single-worker-per-process attribution precondition** (advisory 6): the
// before/after dispatch-registry delta this probe reads is attributed to
// THIS job's own probe call, which is correct as long as no OTHER job's
// admission-gated dispatch races the SAME registry keys on another thread of
// the SAME process between this probe's two snapshots — true for the normal
// one-job-at-a-time-per-worker-instance shape (`FineTuneWorker::run_until`'s
// claim→run→claim loop never overlaps two claims on one worker), but NOT
// guarded against a deployment running multiple `EmbeddedWorker`/
// `FineTuneWorker` instances concurrently in the SAME process. A concurrent
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
/// a different report shape depending on what ran before it — campaign #446
/// finding 2). Only [`jammi_kernels::admission::ProbedOpKind::TwoArm`] rows
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
        .filter(|op| op.kind == jammi_kernels::admission::ProbedOpKind::TwoArm)
        .filter_map(|op| {
            op.registry_keys_for(dtype)
                .next()
                .map(|key| (op.report_key, key))
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
/// structural fact rather than a match arm that has to remember it. Storing
/// one struct FIELD per op (the pre-#446 shape) is what let the table and the
/// snapshot drift apart in the first place.
struct AdmissionProbeSnapshot {
    two_arm: std::collections::BTreeMap<&'static str, jammi_kernels::admission::DispatchSnapshot>,
    attention_block_flash: jammi_kernels::admission::CascadeDispatchSnapshot,
}

impl AdmissionProbeSnapshot {
    /// Snapshots every registry key the table names for `dtype`, straight
    /// through `counters_for(key)` — the SAME `&'static DispatchCounters` the
    /// kernels' own `admit()` sites accumulate into (the
    /// `jammi_encoders::ln_dispatch_snapshot()`-style accessors this used to
    /// call are themselves `counters_for("layer_norm_fused")`,
    /// `crates/jammi-encoders/src/layer_norm.rs:129`, under the hood).
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
/// ambiguous signal this fn never rounds up to a clean positive; family D).
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
/// **This used to read
/// [`jammi_kernels::admission::fallback_warnings_emitted`] and take the most
/// recent entry for the op** (campaign #446 finding 3). That list is
/// process-lifetime AND deduplicated on `(op, predicate)` — a job whose miss
/// repeats a pair an earlier job already burned pushes nothing, so the "most
/// recent entry for this op" was the most recent DIFFERENT predicate, from a
/// different job at a different dtype, persisted durably on this job's
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
/// means a probe WAS attempted and its forward pass errored — Phase-4
/// adversarial-audit finding 3: the pre-fix code passed `probe_ok = false`
/// into `flash_report` for this arm, fabricating "the probe failed" for a
/// probe that was never attempted.
fn flash_report_no_probe_attempted(device: &candle_core::Device) -> serde_json::Value {
    flash_compiled_device_reason(device).unwrap_or_else(
        || serde_json::json!({"holds": false, "reason": "no_encoder_to_probe_projection_head_arm"}),
    )
}

/// The `"flash"` field of the esc-075 report for an arm that DID attempt a
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
/// `"attention_block_flash"` cascade key: `admit_cascade` now records every
/// decline into that SAME sink (see this section's module doc's "The BERT/
/// DistilBERT case, named honestly" paragraph), so a BERT/DistilBERT job's
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
/// this job is about to train with — closing the vacuous-coverage gap a
/// forward-only probe left (Phase-4 adversarial-audit finding 2):
/// backward/optimizer-time admission-gated dispatch (e.g.
/// `LowRankResidualLinear::bwd`'s `cast_add_bf16` epilogue,
/// `crates/jammi-kernels/src/ops/low_rank_residual_linear.rs`) never fires
/// during a plain forward, so a forward-only probe could not honestly claim
/// to have measured it.
///
/// **Restores every trainable var to its pre-probe value afterward**, via a
/// genuine deep copy (`Tensor::copy`, not `Tensor::clone` — candle's `clone`
/// shares the underlying storage `Arc`, so a "snapshot" taken that way would
/// silently mutate alongside the very weights `Var::set` writes into
/// in-place; confirmed by reading `candle_core::Var::set`'s
/// `storage_mut_and_layout` implementation). All-or-nothing: if EVERY
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
/// seed. `crates/jammi-ai/src/fine_tune/adamw.rs`'s own moment buffers are
/// freshly allocated inside THIS function's throwaway `AdamW` instance and
/// never shared with the real trainer's optimizer, so they leave no residue.
fn run_backward_and_optimizer_probe(varmap: &candle_nn::VarMap, output: &candle_core::Tensor) {
    let vars = varmap.all_vars();
    let snapshot: Option<Vec<candle_core::Tensor>> =
        vars.iter().map(|v| v.as_tensor().copy().ok()).collect();
    let Some(snapshot) = snapshot else {
        tracing::warn!(
            "esc-075: could not snapshot every trainable var before the acceleration-report \
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
            "esc-075: acceleration-report probe's backward+optimizer step failed (non-fatal; \
             restoring pre-probe weights regardless)"
        );
    }

    for (var, original) in vars.iter().zip(snapshot.iter()) {
        if let Err(e) = var.set(original) {
            tracing::warn!(
                error = %e,
                "esc-075: failed to restore a trainable var after the acceleration-report \
                 probe's backward+optimizer step — this job's initial weights may now differ \
                 from what it was configured with"
            );
        }
    }
}

/// Runs the esc-075 probe — forward pass, then backward + one optimizer step
/// (see [`run_backward_and_optimizer_probe`]) — when `encoder`/`varmap` are
/// both `Some`, and builds the `ops` map + `flash` field from the
/// before/after dispatch-registry delta. Either is `None` on the
/// projection-head arm (`backbone_dtype` never takes effect there — see
/// `validate_backbone_precision`'s doc): `ops` is then empty and `flash`
/// reports the honest "no probe was ever attempted" reason
/// ([`flash_report_no_probe_attempted`]) — never a fabricated per-op
/// measurement, and never [`flash_report`]'s `"probe_forward_failed"` for a
/// probe that was never even tried (Phase-4 adversarial-audit finding 3).
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
    // eager, regardless of dtype (verified by reading
    // `crates/jammi-encoders/src/layer_norm.rs`'s `forward` doc: "Eval
    // (`training == false`) NEVER reaches the fused arm"). The training loop
    // built moments later (`TrainingLoopBuilder::build`) calls
    // `set_training(true)` unconditionally anyway, so flipping it here first
    // changes nothing about the run this attempt actually trains.
    encoder.set_training(true);

    let before = AdmissionProbeSnapshot::capture(dtype);
    // Campaign #446 finding 3: arm THIS probe's own capture window before the
    // forward, and read every `holds: false` reason back out of it. The window
    // is thread-local and this whole function (forward, `Tensor::backward()`'s
    // graph walk, `AdamW::step`) runs synchronously on the ONE
    // `spawn_blocking` thread `run_fine_tune_blocking` was handed — see
    // `jammi_kernels::admission::probe_capture_begin`'s doc for the constraint
    // and exactly where it would break (a future async yield inside the probe,
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
    // acceleration that the job never earned (esc-075 / issue #421 A7). A
    // genuine probe FAILURE still degrades to an empty `ops` map — never a
    // propagated error (this function, and its caller, are infallible by
    // construction: esc-075 requires training to be unaffected by a
    // report-computation failure).
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

/// Builds the esc-075 acceleration-report JSON payload for this attempt. See
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

/// Computes and persists this attempt's esc-075 acceleration report. Runs
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
    device: &candle_core::Device,
    backbone_dtype: jammi_numerics::ComputePrecision,
    varmap: Option<&candle_nn::VarMap>,
    encoder: Option<&mut jammi_encoders::AnyEncoder>,
) {
    let report_json =
        build_acceleration_report_json(attempt, device, backbone_dtype, varmap, encoder);
    tokio::runtime::Handle::current().block_on(persist_acceleration_report(
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
fn build_encoder_adapters(
    base_model_id: &str,
    catalog: &Arc<Catalog>,
    artifact_store: &Arc<ArtifactStore>,
    config: &FineTuneConfig,
    task: ModelTask,
    varmap: &candle_nn::VarMap,
    device: &candle_core::Device,
) -> Result<(jammi_encoders::AnyEncoder, jammi_lora::AdapterConfig)> {
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

    let artifact_dir: std::path::PathBuf = match model_record.artifact_path.as_deref() {
        Some(p) if !p.is_empty() => {
            let url = jammi_db::storage::StorageUrl::parse(p)?;
            if url.scheme() == jammi_db::storage::Scheme::File {
                // A locally-registered base model (HF cache / local dir): its
                // weights already sit on a path candle can mmap. Use it in place.
                let path = std::path::PathBuf::from(url.path());
                if path.is_dir() {
                    path
                } else {
                    path.parent()
                        .ok_or_else(|| {
                            JammiError::FineTune(format!(
                                "Cannot determine model dir from artifact_path '{p}'"
                            ))
                        })?
                        .to_path_buf()
                }
            } else {
                // The base model's artifact lives in the object store — fetch the
                // bundle into a local cache dir candle can load from, so a
                // worker on any host resolves the same backbone.
                tokio::runtime::Handle::current()
                    .block_on(artifact_store.fetch_artifact(&url))?
                    .dir()
                    .to_path_buf()
            }
        }
        _ => {
            if is_hf {
                let api = hf_hub::api::sync::Api::new()
                    .map_err(|e| JammiError::FineTune(format!("HF hub init: {e}")))?;
                let repo = api.model(catalog_model_id.clone());
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
                    "Base model '{base_model_id}' has no artifact_path in catalog"
                )));
            }
        }
    };

    // Config SOURCE ORDER is the resolver's own (issue #421 D7): the catalog
    // record's stored `config_json` when it has one, else the first existing
    // candidate on disk walking the shared frozen chain (`config.json`, then
    // the OpenCLIP `open_clip_config.json`). Reading `config.json` off disk
    // unconditionally — the pre-#421 shape — could not see an OpenCLIP
    // checkpoint at all, and disagreed with the resolver whenever the catalog
    // carried a config the directory did not.
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

    // GGUF/QLoRA (issue #351): the base artifact SELECTS this — no new
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

    // GGUF/QLoRA (issue #351): everything the three encoder builders below
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
        // off the raw config here previously refused every DistilBERT GGUF
        // fine-tune outright (issue #351 wave 5 audit).
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

    Ok((encoder, adapter_cfg))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use candle_core::Tensor;

    use super::*;

    /// Campaign #446 finding 3, the honest-negative half:
    /// [`reason_from_probe_window`] returns the window's OWN verbatim
    /// predicate for an op it recorded, and [`REASON_UNAVAILABLE`] — never a
    /// guess, and never some other op's predicate — for one it did not.
    ///
    /// The `None` branch is not reachable deterministically through the
    /// public probe (it needs a concurrent thread's dispatch to move a
    /// counter inside this probe's window, or a future off-thread admission
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

    /// Issue #462/#463 follow-up: `admit_cascade`'s decline path now records
    /// `(op, predicate)` into the SAME probe-capture window `admit_inner`
    /// uses — `record_probe_miss(op, predicate_name)` (`crates/jammi-kernels/src/admission.rs:427,437`), which is what
    /// lets [`flash_cascade_decline_reason`] — the function [`flash_report`]
    /// itself calls on a decline — read a verbatim reason back for the
    /// `"attention_block_flash"` cascade key instead of the coarse
    /// `"capability_or_domain_miss"` fallback.
    ///
    /// This drives the REAL `jammi_kernels::admission::admit_cascade` call
    /// (never a fabricated window entry) with BERT/DistilBERT's own verbatim
    /// predicate — `"flash_transport_not_wired"`, the ONE reason value either
    /// family's `FlashDecision::Declined` ever carries — see
    /// `BERT never wires the encoder-boundary flash transport` (`crates/jammi-encoders/src/bert.rs:428-420`)
    /// and the sibling `FlashDecision::Declined` (`crates/jammi-encoders/src/distilbert.rs:331-324`) — for a
    /// `CapabilityMiss` outcome on the `"attention_block_flash"` op, exactly
    /// as `attention_cascade::training_attention_cascade` does for a
    /// BERT-family training forward (`crates/jammi-encoders/src/
    /// attention_cascade.rs:599-604`).
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
            "attention_block_flash",
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
        use jammi_db::catalog::status::TrainingJobStatus;

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
                artifact_path: None,
                config_json: None,
            })
            .await
            .unwrap();
        catalog
            .create_training_job(jammi_db::catalog::training_repo::CreateTrainingJobParams {
                job_id: "panic-job",
                base_model_id: "panic-base::1",
                training_source: "src",
                loss_type: "cosent",
                hyperparams: "{}",
                kind: "fine_tune",
                training_spec: "{}",
            })
            .await
            .unwrap();

        // The worker claims the job (running, leased to it) before running it —
        // the state in which a genuine failure is recorded under the lease guard.
        catalog
            .claim_next_training_job("worker-x", Duration::from_secs(60))
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
            Ok(Err(payload)) => {
                WorkerJobError::Failed(format!("Panic: {}", panic_message(payload.as_ref())))
            }
            Err(join_err) => {
                WorkerJobError::Failed(format!("training task join error: {join_err}"))
            }
        };

        let WorkerJobError::Failed(msg) = outcome else {
            panic!("a genuine panic must classify as Failed, not Cancelled");
        };
        assert!(
            msg.contains("Panic:") && msg.contains("simulated candle kernel fault"),
            "a caught panic must carry its message into the failure, got: {msg}"
        );

        // The worker records the failure as the job's terminal status, under the
        // lease guard (it still owns the job).
        record_failed(&catalog, "panic-job", "worker-x", msg).await;

        let job = catalog.get_training_job("panic-job").await.unwrap();
        assert_eq!(
            job.status,
            TrainingJobStatus::Failed.to_string(),
            "a panicking job lands `failed`, never wedged `running`"
        );
        assert!(
            job.error_message
                .as_deref()
                .is_some_and(|m| m.contains("simulated candle kernel fault")),
            "the panic cause is recorded on the job, got {:?}",
            job.error_message
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

    /// Negative control: a genuine non-OOM CUDA failure (a kernel/PTX fault,
    /// #319's exact misroute risk) must pass through completely unchanged —
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
        let WorkerJobError::Failed(msg) = classify_training_error(&cancel, &config, oom_err) else {
            panic!("a genuine OOM must classify as Failed, not Cancelled");
        };
        assert!(
            msg.contains("out of memory") && msg.contains("batch_size=8"),
            "must carry the classified OOM guidance, got: {msg}"
        );

        // not cancelled + non-OOM: byte-identical passthrough.
        let cancel = AtomicBool::new(false);
        let raw = JammiError::FineTune("Encoder forward: CUDA_ERROR_INVALID_PTX".into());
        let raw_msg = raw.to_string();
        let WorkerJobError::Failed(msg) = classify_training_error(&cancel, &config, raw) else {
            panic!("a non-OOM failure must classify as Failed, not Cancelled");
        };
        assert_eq!(msg, raw_msg, "a non-OOM error must pass through unchanged");
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
        use jammi_db::catalog::status::TrainingJobStatus;

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
                artifact_path: None,
                config_json: None,
            })
            .await
            .unwrap();
        catalog
            .create_training_job(jammi_db::catalog::training_repo::CreateTrainingJobParams {
                job_id: "oom-job",
                base_model_id: "oom-base::1",
                training_source: "src",
                loss_type: "cosent",
                hyperparams: "{}",
                kind: "fine_tune",
                training_spec: "{}",
            })
            .await
            .unwrap();

        // The worker claims the job (running, leased to it) before running it —
        // the state in which a genuine failure is recorded under the lease guard.
        catalog
            .claim_next_training_job("worker-x", Duration::from_secs(60))
            .await
            .unwrap()
            .expect("the queued job is claimable");

        let cancel = Arc::new(AtomicBool::new(false));
        // The engine defaults (batch 8, seq 512, backbone F32, empty
        // target_modules => projection-head arm) that OOM on an L4 24GB card
        // at inference-side defaults — issue #345's remaining repro shape.
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
            Ok(Err(payload)) => {
                WorkerJobError::Failed(format!("Panic: {}", panic_message(payload.as_ref())))
            }
            Err(join_err) => {
                WorkerJobError::Failed(format!("training task join error: {join_err}"))
            }
        };

        let WorkerJobError::Failed(msg) = outcome else {
            panic!("a genuine OOM must classify as Failed, not Cancelled");
        };
        assert!(
            msg.contains("batch_size=8")
                && msg.contains("backbone_dtype does not apply to projection-head runs")
                && !msg.contains("backbone_dtype: bf16"),
            "the classified OOM guidance must reach the terminal message, without the \
             inert bf16 remedy on the default (projection-head) arm, got: {msg}"
        );

        // The worker records the failure as the job's terminal status, under the
        // lease guard (it still owns the job).
        record_failed(&catalog, "oom-job", "worker-x", msg).await;

        let job = catalog.get_training_job("oom-job").await.unwrap();
        assert_eq!(
            job.status,
            TrainingJobStatus::Failed.to_string(),
            "an OOM'd job lands `failed`, never wedged `running`"
        );
        assert!(
            job.error_message
                .as_deref()
                .is_some_and(|m| m.contains("batch_size=8")
                    && m.contains("backbone_dtype does not apply to projection-head runs")),
            "`jammi train status` / job.status() must surface the classified OOM \
             guidance from the catalog's error_message, got {:?}",
            job.error_message
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

    /// `resolve_worker_id` honours a set, non-empty `JAMMI_WORKER_ID` verbatim
    /// (trimmed) and otherwise mints a fresh random `worker-{uuid}`. An empty /
    /// all-whitespace value falls back rather than seeding a blank `claimed_by`.
    ///
    /// `JAMMI_WORKER_ID` is process-global, so the three cases run in one test
    /// (parallel tests must not race the same env var) and the var is removed at
    /// the end to leave the environment clean for the rest of the suite.
    #[test]
    fn resolve_worker_id_honours_seed_else_random() {
        // Set + non-empty → adopted verbatim (after trimming).
        std::env::set_var(WORKER_ID_ENV, "  worker-7  ");
        assert_eq!(resolve_worker_id(), "worker-7");

        // Empty / all-whitespace → treated as unset (a blank claimed_by is useless).
        std::env::set_var(WORKER_ID_ENV, "   ");
        let blank_fallback = resolve_worker_id();
        assert!(
            blank_fallback.starts_with("worker-") && blank_fallback.len() > "worker-".len(),
            "an all-whitespace seed must fall back to a random id, got {blank_fallback:?}"
        );

        // Unset → a fresh random uuid id, and two calls differ.
        std::env::remove_var(WORKER_ID_ENV);
        let a = resolve_worker_id();
        let b = resolve_worker_id();
        assert!(a.starts_with("worker-"), "default id is worker-prefixed");
        assert_ne!(a, b, "the random default mints a distinct id per call");
    }

    // ─── Regression detector (W5-PR4 public on-ramp) ─────────────────────────
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
    // `build_encoder_adapters` GGUF class-closure oracle (issue #351 wave
    // 5 audit): a `build_encoder_adapters` construction test for EACH of
    // the three GGUF-threaded architectures. Before this wave's fix, the
    // DistilBERT arm hard-refused ("GGUF load requires num_hidden_layers
    // (or num_layers) in config.json") because it read the layer count
    // off the RAW config.json, which for a DistilBERT checkpoint declares
    // only the DistilBERT-native `n_layers` field — DistilBERT GGUF QLoRA
    // was UNREACHABLE. BERT and ModernBERT already worked (their raw
    // config.json already uses `num_hidden_layers`), which is exactly why
    // this bug was invisible on those two arches and needed a per-
    // architecture oracle to surface at all.
    // ─────────────────────────────────────────────────────────────────

    /// FNV-1a-seeded deterministic small-magnitude values (family J: no
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
        names.sort(); // deterministic write order (family J)
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

    /// Register `model_id` in `catalog` with `artifact_path` pointing at
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
                artifact_path: Some(dir.to_str().unwrap()),
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

        tokio::task::spawn_blocking(move || {
            let training_config = FineTuneConfig {
                target_modules,
                ..FineTuneConfig::default()
            };
            let varmap = candle_nn::VarMap::new();
            let device = candle_core::Device::Cpu;
            build_encoder_adapters(
                &base_model_id,
                &catalog,
                &artifact_store,
                &training_config,
                ModelTask::TextEmbedding,
                &varmap,
                &device,
            )
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

    /// RED at 32a3552c (issue #351 wave 5 audit, before this fix): fails
    /// with "GGUF load requires num_hidden_layers (or num_layers) in
    /// config.json" — the raw config.json this fixture writes carries
    /// only `n_layers`, DistilBERT's native field name, so reading
    /// `num_hidden_layers`/`num_layers` directly off it (the pre-fix
    /// `build_encoder_adapters`) always misses. GREEN after routing
    /// through `gguf::gguf_num_layers`, which normalizes first.
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

        tokio::task::spawn_blocking(move || {
            let training_config = FineTuneConfig {
                target_modules,
                lora_dropout: 0.0,
                ..FineTuneConfig::default()
            };
            let varmap = candle_nn::VarMap::new();
            let device = candle_core::Device::Cpu;
            build_encoder_adapters(
                &base_model_id,
                &catalog,
                &artifact_store,
                &training_config,
                ModelTask::TextEmbedding,
                &varmap,
                &device,
            )
        })
        .await
        .unwrap()
    }

    /// Deterministic additive-offset perturbation (family J: no unseeded
    /// RNG) — every tensor is shifted by a fixed nonzero offset, keeping
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

    /// Dual-format precedence sweep (issue #351 wave 14, round-8 audit;
    /// two-phase + forward-equality rework, round-9 audit): the
    /// single-phase predecessor of this test corrupted `model.gguf`
    /// BEFORE ever calling `build_encoder_adapters`, so only Ok-vs-Err
    /// discriminated the two arms — a resolver refactored to "try gguf,
    /// fall back to safetensors on a gguf READ failure" (a
    /// `.ok()`-keyed fallback, not the frozen PRESENCE-keyed precedence
    /// at `is_gguf = !weights_path.exists()`) would ALSO fail to read
    /// the already-corrupt file and fall back to safetensors, passing
    /// that test identically to the correct implementation.
    ///
    /// This version instead builds a safetensors-only REFERENCE encoder
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
    ///   would DEVIATE from `reference_forward` — this is the
    ///   discriminator the corrupt-before-build predecessor lost, and it
    ///   covers the dense-`FrozenBase` claim mechanistically (family F):
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
            worker.handle.lock().unwrap().is_none(),
            "stop_and_join must take the handle so a later Drop finds nothing to abort"
        );

        // Idempotent: a second call finds no handle left and returns
        // immediately rather than blocking or erroring.
        tokio::time::timeout(Duration::from_secs(5), worker.stop_and_join())
            .await
            .expect("a second stop_and_join must not hang")
            .expect("a second stop_and_join on an already-joined worker is Ok");
    }
}
