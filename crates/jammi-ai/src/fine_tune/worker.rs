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
        let output_model_id = format!("jammi:fine-tuned:{job_id}");
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
    handle: tokio::task::JoinHandle<()>,
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
        Ok(Self { handle, stop })
    }
}

impl Drop for EmbeddedWorker {
    /// Signal the loop to stop and abort its task. This halts claiming of new
    /// jobs; an in-flight `spawn_blocking` training run is not aborted by this —
    /// it runs to completion and writes its terminal status post-drop (see the
    /// type doc).
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.handle.abort();
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

/// Extract all string values from an Arrow column.
///
/// DataFusion 52+ returns Parquet string columns as `Utf8View` by default;
/// older versions returned `Utf8` or `LargeUtf8`. Dictionary-encoded variants
/// are also possible. Fast paths cover the three common types; the `cast`
/// fallback handles everything else.
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
    let casted = arrow::compute::cast(col, &DataType::Utf8).ok()?;
    let a = casted.as_any().downcast_ref::<StringArray>()?;
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
/// an audio embedding task reads them as encoded-audio bytes; every other task
/// reads them as text. The column names are identical across modalities (the
/// triplet shape is the same) — only the cell decoding differs, so the caller's
/// chosen task is the discriminator, not a parallel set of column names.
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

    if has_triplet && task == ModelTask::AudioEmbedding {
        return build_audio_triplet_loader(batches);
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
            let neg_vals = batch
                .column_by_name("negative")
                .and_then(|c| extract_string_column(c.as_ref()))
                .ok_or_else(|| {
                    JammiError::FineTune(format!(
                        "Missing/invalid 'negative' column. Batch schema: [{}]",
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
             (text, target) with task=regression. For audio triplets, use the same \
             (anchor, positive, negative) columns with task=audio_embedding."
        )))
    }
}

/// Build an audio-triplet loader: read `anchor`/`positive`/`negative` as
/// encoded-audio byte columns. Shares the triplet column shape with the text
/// path; only the cell type differs (binary clips vs strings).
fn build_audio_triplet_loader(batches: &[RecordBatch]) -> Result<TrainingDataLoader> {
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
                    "Missing/invalid binary 'anchor' column for audio triplets. Batch schema: [{}]",
                    schema_info()
                ))
            })?;
        let pos_vals = batch
            .column_by_name("positive")
            .and_then(|c| extract_binary_column(c.as_ref()))
            .ok_or_else(|| {
                JammiError::FineTune(format!(
                    "Missing/invalid binary 'positive' column for audio triplets. Batch schema: [{}]",
                    schema_info()
                ))
            })?;
        let neg_vals = batch
            .column_by_name("negative")
            .and_then(|c| extract_binary_column(c.as_ref()))
            .ok_or_else(|| {
                JammiError::FineTune(format!(
                    "Missing/invalid binary 'negative' column for audio triplets. Batch schema: [{}]",
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
    Ok(TrainingDataLoader::from_audio_triplets(rows))
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
        crate::fine_tune::target::TrainingTarget::ProjectionHead { head }
    } else {
        let (encoder, adapter_cfg) = build_encoder_adapters(
            &base_model,
            &catalog,
            &artifact_store,
            &config,
            &varmap,
            &device,
        )?;
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

/// Construct an encoder-adapters target: load the frozen backbone weights from
/// the catalog artifact path, wrap the configured target modules with LoRA, and
/// return both the resulting encoder and the persisted adapter metadata that
/// pairs with the trained tensors on disk.
fn build_encoder_adapters(
    base_model_id: &str,
    catalog: &Arc<Catalog>,
    artifact_store: &Arc<ArtifactStore>,
    config: &FineTuneConfig,
    varmap: &candle_nn::VarMap,
    device: &candle_core::Device,
) -> Result<(jammi_encoders::AnyEncoder, jammi_lora::AdapterConfig)> {
    use std::path::Path;

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

    let config_path = artifact_dir.join("config.json");
    let model_config: serde_json::Value = std::fs::read_to_string(&config_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .ok_or_else(|| {
            JammiError::FineTune(format!(
                "Cannot read config.json for base model at {config_path:?}"
            ))
        })?;

    let model_type = model_config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("bert");

    // GGUF/QLoRA (issue #351): the base artifact SELECTS this — no new
    // trainer/config knob. `model.safetensors` wins when both happen to be
    // present (mirrors the resolver's own FROZEN precedence,
    // `model::resolver::ModelResolver::resolve_local`); only when it is
    // ABSENT does `model.gguf` enter the picture at all.
    let weights_path = artifact_dir.join("model.safetensors");
    let gguf_weights_path = artifact_dir.join("model.gguf");
    let is_gguf = !weights_path.exists();
    if is_gguf && !gguf_weights_path.exists() {
        return Err(JammiError::FineTune(format!(
            "Neither model.safetensors nor model.gguf found at {artifact_dir:?}"
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
    let adapter_cfg =
        jammi_lora::AdapterConfig::from_build(model_type, &lora, config.backbone_dtype);

    // GGUF/QLoRA (issue #351): everything the three encoder builders below
    // need to train LoRA over a `FrozenBase::Quantized` backbone — built
    // ONCE here from the GGUF file's tensor data, exactly the way
    // `CandleBackend::load`'s inference path builds it (the SAME
    // `crate::model::backend::gguf` module, so a QLoRA fine-tune and an
    // inference load of the same `model.gguf` can never silently disagree
    // on which tensors are matmul-site or which dtype loaded).
    let gguf_backbone = if is_gguf {
        let arch = crate::model::backend::gguf::GgufArchitecture::from_model_type(model_type)
            .ok_or_else(|| {
                JammiError::FineTune(format!(
                    "quantized serving not supported for this architecture (model_type \
                     '{model_type}')"
                ))
            })?;
        // Routes through the SAME normalization + layer-count authority
        // `CandleBackend::load`'s GGUF path and `estimate_gguf_residency`
        // use (`gguf::gguf_num_layers`) — a raw, un-normalized DistilBERT
        // config declares its layer count under the DistilBERT-native
        // `n_layers` name only, so reading `num_hidden_layers`/`num_layers`
        // off the raw config here previously refused every DistilBERT GGUF
        // fine-tune outright (issue #351 wave 5 audit).
        let num_layers = crate::model::backend::gguf::gguf_num_layers(model_type, &model_config)
            .ok_or_else(|| {
                JammiError::FineTune(
                    "GGUF load requires num_hidden_layers (or num_layers) in config.json".into(),
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

    let encoder = match model_type {
        "distilbert" => {
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
            jammi_encoders::AnyEncoder::DistilBert(
                builder
                    .build(&weights_paths, &distilbert_config, device, varmap)
                    .map_err(|e| JammiError::FineTune(format!("Build DistilBert encoder: {e}")))?,
            )
        }
        "modernbert" => {
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
            jammi_encoders::AnyEncoder::ModernBert(
                builder
                    .build(&weights_paths, &modernbert_config, device, varmap)
                    .map_err(|e| JammiError::FineTune(format!("Build ModernBert encoder: {e}")))?,
            )
        }
        _ => {
            let bert_config: jammi_encoders::BertConfig =
                serde_json::from_value(model_config.clone())
                    .map_err(|e| JammiError::FineTune(format!("Parse Bert config.json: {e}")))?;
            let mut builder = jammi_encoders::Bert::builder()
                .lora(lora)
                .backbone_dtype(backbone_dtype);
            if let Some(lookup) = &gguf_lookup {
                builder = builder.weight_source(lookup);
            }
            jammi_encoders::AnyEncoder::Bert(
                builder
                    .build(&weights_paths, &bert_config, device, varmap)
                    .map_err(|e| JammiError::FineTune(format!("Build Bert encoder: {e}")))?,
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
}
