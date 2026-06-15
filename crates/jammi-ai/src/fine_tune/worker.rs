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
                        self.run_spec(session, &catalog, &job_id, spec, &cancel)
                    })
                    .await
            }
            None => {
                self.run_spec(session, &catalog, &job_id, spec, &cancel)
                    .await
            }
        };

        // Stop the heartbeat regardless of outcome.
        heartbeat.abort();

        match outcome {
            Ok(artifact) => {
                self.publish_and_finalize(session, &catalog, &job_id, attempt, artifact)
                    .await;
            }
            Err(WorkerJobError::Cancelled) => {
                // Lease lost: leave the job `running` for reclaim to re-queue.
                // Do not record a terminal status — a different worker now owns,
                // or will own, this job.
                tracing::warn!(job_id = %job_id, worker = %self.worker_id, "training cancelled (lease lost); left for reclaim");
            }
            Err(WorkerJobError::Failed(msg)) => {
                tracing::error!(job_id = %job_id, error = %msg, "training job failed");
                record_failed(&catalog, &job_id, &self.worker_id, msg).await;
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
        artifact: TrainedArtifact,
    ) {
        let store = session.artifact_store();
        let TrainedArtifact {
            dir,
            register,
            metrics,
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
                    return;
                }
            };

        if let Err(e) = catalog.register_model(register.as_params()).await {
            // The model row could not be registered; the prefix we wrote is
            // orphaned. Best-effort GC it and fail the job.
            store.delete_artifact_prefix(&prefix).await.ok();
            record_failed(catalog, job_id, &self.worker_id, e.to_string()).await;
            return;
        }

        match catalog
            .finalize_training_job(
                job_id,
                &self.worker_id,
                &model_id,
                prefix.as_str(),
                metrics.as_deref(),
            )
            .await
        {
            Ok(true) => {
                // The finalize CAS won: the job is `completed`, so its durable
                // resume checkpoint is dead. GC it (best-effort — a leftover
                // resume prefix is harmless, never on the serving path, but the
                // winner is the single point that reclaims it).
                store.delete_resume_checkpoint(job_id).await.ok();
            }
            Ok(false) => {
                // Lost the lease before finalizing: our CAS matched zero rows, so
                // we committed neither the job status nor any served path. Our
                // prefix is never the committed pointer — GC it best-effort and
                // leave the job for reclaim (the re-claiming worker writes its own
                // prefix and its CAS commits it).
                store.delete_artifact_prefix(&prefix).await.ok();
                tracing::debug!(
                    job_id = %job_id,
                    worker = %self.worker_id,
                    "lost lease before finalize; not finalizing (left for reclaim)"
                );
            }
            Err(e) => {
                store.delete_artifact_prefix(&prefix).await.ok();
                tracing::error!(job_id = %job_id, error = %e, "finalize_training_job failed");
            }
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
                self.train_fine_tune(session, catalog, job_id, run, cancel)
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
                self.train_fine_tune(session, catalog, job_id, run, cancel)
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
        let params = RunFineTuneParams {
            catalog: Arc::clone(catalog),
            artifact_store: session.artifact_store(),
            artifact_dir: session.inner_config().artifact_dir.clone(),
            job_id: job_id.to_string(),
            worker_id: self.worker_id.clone(),
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
            Ok(Ok(Err(e))) => return Err(classify(&cancel_for_classify, e)),
            Ok(Err(payload)) => {
                return Err(WorkerJobError::Failed(format!(
                    "Panic: {}",
                    panic_message(payload.as_ref())
                )))
            }
            Err(join_err) => {
                return Err(WorkerJobError::Failed(format!(
                    "training task join error: {join_err}"
                )))
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
                model_type: "fine-tuned",
                task,
                base_model_id: Some(base_model),
                config_json: None,
            },
            metrics: Some(training.metrics_json),
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
            version: 1,
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

    let catalog_model_id = base_model_id
        .strip_prefix("hf://")
        .or_else(|| base_model_id.strip_prefix("local:"))
        .unwrap_or(base_model_id);

    let model_record = tokio::runtime::Handle::current()
        .block_on(catalog.get_model(catalog_model_id))?
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
            let is_hf = base_model_id.starts_with("hf://")
                || (!base_model_id.starts_with('/')
                    && !std::path::Path::new(base_model_id).exists());
            if is_hf {
                let api = hf_hub::api::sync::Api::new()
                    .map_err(|e| JammiError::FineTune(format!("HF hub init: {e}")))?;
                let repo = api.model(catalog_model_id.to_string());
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

    let weights_path = artifact_dir.join("model.safetensors");
    if !weights_path.exists() {
        return Err(JammiError::FineTune(format!(
            "model.safetensors not found at {weights_path:?}"
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

    let backbone_dtype: candle_core::DType = config.backbone_dtype.into();
    let adapter_cfg =
        jammi_lora::AdapterConfig::from_build(model_type, &lora, config.backbone_dtype);

    let weights_paths: Vec<&Path> = vec![weights_path.as_path()];

    let encoder = match model_type {
        "distilbert" => {
            let distilbert_config: jammi_encoders::DistilBertConfig =
                serde_json::from_value(model_config.clone()).map_err(|e| {
                    JammiError::FineTune(format!("Parse DistilBert config.json: {e}"))
                })?;
            jammi_encoders::AnyEncoder::DistilBert(
                jammi_encoders::DistilBert::builder()
                    .lora(lora)
                    .backbone_dtype(backbone_dtype)
                    .build(&weights_paths, &distilbert_config, device, varmap)
                    .map_err(|e| JammiError::FineTune(format!("Build DistilBert encoder: {e}")))?,
            )
        }
        "modernbert" => {
            let modernbert_config: jammi_encoders::ModernBertConfig =
                serde_json::from_value(model_config.clone()).map_err(|e| {
                    JammiError::FineTune(format!("Parse ModernBert config.json: {e}"))
                })?;
            jammi_encoders::AnyEncoder::ModernBert(
                jammi_encoders::ModernBert::builder()
                    .lora(lora)
                    .backbone_dtype(backbone_dtype)
                    .build(&weights_paths, &modernbert_config, device, varmap)
                    .map_err(|e| JammiError::FineTune(format!("Build ModernBert encoder: {e}")))?,
            )
        }
        _ => {
            let bert_config: jammi_encoders::BertConfig =
                serde_json::from_value(model_config.clone())
                    .map_err(|e| JammiError::FineTune(format!("Parse Bert config.json: {e}")))?;
            jammi_encoders::AnyEncoder::Bert(
                jammi_encoders::Bert::builder()
                    .lora(lora)
                    .backbone_dtype(backbone_dtype)
                    .build(&weights_paths, &bert_config, device, varmap)
                    .map_err(|e| JammiError::FineTune(format!("Build Bert encoder: {e}")))?,
            )
        }
    };

    Ok((encoder, adapter_cfg))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

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
}
