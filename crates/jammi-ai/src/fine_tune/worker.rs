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
use std::time::Duration;

use arrow::array::RecordBatch;
use jammi_db::catalog::Catalog;
use jammi_db::error::{JammiError, Result};
use jammi_db::model_task::ModelTask;

use crate::fine_tune::data::TrainingDataLoader;
use crate::fine_tune::graph_sampler::{
    GraphEdge, GraphFineTuneSources, GraphSampleConfig, GraphSampler, TextNode,
};
use crate::fine_tune::spec::{TrainingCommon, TrainingSpec};
use crate::fine_tune::staging::StagedArtifact;
use crate::fine_tune::FineTuneConfig;
use crate::model::backend::DeviceConfig;
use crate::model::ModelSource;
use crate::session::InferenceSession;

// Lease timing. The lease is the window a claimed job is exclusively owned; the
// heartbeat renews it well inside that window so a single missed beat (GC pause,
// a slow tick) does not drop the lease. The 3× margin (30s lease / 10s beat)
// tolerates one missed beat. The idle poll is how often an idle worker checks
// for new work, and reclaim runs each idle tick so a dead worker's job is
// recovered within roughly one poll + lease.

/// How long a claim leases a job before it is considered orphaned.
const LEASE: Duration = Duration::from_secs(30);
/// Heartbeat interval — renews the lease at 3× margin inside [`LEASE`].
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(10);
/// How often an idle worker polls for a queued job (and reclaims expired leases).
const IDLE_POLL: Duration = Duration::from_secs(1);
/// Attempts cap before `reclaim_expired_training_jobs` fails a job for good.
const MAX_ATTEMPTS: u32 = 3;

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
    /// this worker's leases from another's.
    worker_id: String,
}

impl TrainingWorker {
    /// Build a worker over a session. The worker holds a [`Weak`] so it never
    /// keeps the session alive; the caller owns the strong `Arc` and the worker
    /// stops when that drops.
    pub fn new(session: &Arc<InferenceSession>) -> Self {
        Self {
            session: Arc::downgrade(session),
            worker_id: format!("worker-{}", uuid::Uuid::new_v4()),
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
    /// state inline (the next claim waits for it), on no claim it sleeps
    /// `IDLE_POLL`. The catalog used for reclaim/claim is unscoped — a worker
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
                .claim_next_training_job(&self.worker_id, LEASE)
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
                None => tokio::time::sleep(IDLE_POLL).await,
            }
        }
    }

    /// Run one already-claimed job to a terminal state. Deserialises the spec,
    /// pins the catalog to the job's tenant (the claim is intentionally unscoped,
    /// so the worker's writes must be re-scoped), runs the kind's reconstruction
    /// under a heartbeat, then performs the single lease-guarded finalize —
    /// `completed` + the output model when this worker still holds the lease, or
    /// `failed` + the error on a genuine failure. A worker that lost its lease in
    /// the run window does not finalize; the job is left for `reclaim`.
    ///
    /// `record` must be a row this worker claimed (its `claimed_by` is the
    /// worker's id). The driving loop ([`Self::run_until`]) is the normal caller;
    /// it is exposed so a test can drive one claimed job in isolation.
    pub async fn run_claimed_job(
        &self,
        session: &Arc<InferenceSession>,
        record: jammi_db::catalog::training_repo::TrainingJobRecord,
    ) {
        let job_id = record.job_id.clone();
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

        let outcome = self
            .run_spec(session, &catalog, &job_id, spec, &cancel)
            .await;

        // Stop the heartbeat regardless of outcome.
        heartbeat.abort();

        match outcome {
            Ok(success) => {
                // Single finalization for every kind: the run staged its artifact
                // privately and registered the output-model row pointing at the
                // (not-yet-written) canonical path; the worker now performs the
                // one lease-guarded compare-and-set that writes `output_model_id`
                // + flips the job to `completed`. The CAS matches only while this
                // worker still holds the lease (`claimed_by = worker_id AND
                // status = 'running'`), so a worker that lost its lease in the
                // window between the last epoch check and here affects zero rows
                // and must NOT finalize — the job is left `running` for `reclaim`
                // and the worker that re-claimed it.
                //
                // The on-disk canonical artifact mirrors that discipline: only the
                // CAS winner promotes its staging into the canonical path (an
                // atomic same-filesystem rename); the loser discards its staging
                // and never touches the shared canonical artifact. So a `wait()`
                // observer that sees `completed` always finds the complete
                // canonical artifact written by exactly one worker — the winner.
                let RunSuccess {
                    model_id,
                    metrics,
                    staged,
                } = success;
                match catalog
                    .finalize_training_job(&job_id, &self.worker_id, &model_id, metrics.as_deref())
                    .await
                {
                    Ok(true) => {
                        if let Err(e) = staged.promote() {
                            tracing::error!(job_id = %job_id, error = %e, "promote staged artifact failed");
                        }
                    }
                    Ok(false) => {
                        staged.discard();
                        tracing::debug!(
                            job_id = %job_id,
                            worker = %self.worker_id,
                            "lost lease before finalize; not finalizing (left for reclaim)"
                        );
                    }
                    Err(e) => {
                        staged.discard();
                        tracing::error!(job_id = %job_id, error = %e, "finalize_training_job failed");
                    }
                }
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

    /// Spawn the lease-renewing heartbeat task. It renews on [`HEARTBEAT_INTERVAL`]
    /// and, the first time `heartbeat_training_job` reports the lease lost, sets
    /// `cancel` and stops.
    fn spawn_heartbeat(
        &self,
        catalog: Arc<Catalog>,
        job_id: String,
        cancel: Arc<AtomicBool>,
    ) -> tokio::task::JoinHandle<()> {
        let worker_id = self.worker_id.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(HEARTBEAT_INTERVAL).await;
                match catalog
                    .heartbeat_training_job(&job_id, &worker_id, LEASE)
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
    /// training, returning the [`RunSuccess`] on success.
    async fn run_spec(
        &self,
        session: &Arc<InferenceSession>,
        catalog: &Arc<Catalog>,
        job_id: &str,
        spec: TrainingSpec,
        cancel: &Arc<AtomicBool>,
    ) -> std::result::Result<RunSuccess, WorkerJobError> {
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
                // registers the predictor's model row through the worker's
                // tenant-pinned catalog (the same catalog the fine-tune kinds
                // register through), so the model lands under the job's tenant
                // rather than the worker session's sticky scope.
                let (record, staged) = session
                    .run_context_predictor_training(
                        catalog,
                        &source,
                        &predictor_spec,
                        &self.worker_id,
                        cancel,
                    )
                    .await
                    .map_err(|e| classify(cancel, e))?;
                Ok(RunSuccess {
                    model_id: record.model_id,
                    metrics: None,
                    staged,
                })
            }
        }
    }

    /// Re-run `SELECT columns FROM source` for a tabular fine-tune.
    async fn read_source_columns(
        &self,
        session: &Arc<InferenceSession>,
        source: &str,
        columns: &[String],
    ) -> Result<Vec<RecordBatch>> {
        let table_name = session.find_table_name(source)?;
        let select = columns
            .iter()
            .map(|c| format!("\"{c}\""))
            .collect::<Vec<_>>()
            .join(", ");
        let query = format!("SELECT {select} FROM {source}.public.\"{table_name}\"");
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
            "SELECT \"{}\", \"{}\" FROM {}.public.\"{node_table}\"",
            sources.id_column, sources.text_column, sources.node_source
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
            "SELECT \"{}\", \"{}\" FROM {}.public.\"{edge_table}\"",
            sources.src_column, sources.dst_column, sources.edge_source
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
    ) -> std::result::Result<RunSuccess, WorkerJobError> {
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

        // Register the fine-tuned model row before the worker's finalize CAS so
        // a `wait()` observer that sees `completed` always finds the model row.
        // The row points at the canonical artifact path; that path is written
        // (by promoting the staged artifact) only when the finalize CAS wins, so
        // a `completed` job always has a complete canonical artifact. The id is
        // deterministic (`jammi:fine-tuned:{job_id}`) and the catalog upserts, so
        // a re-claiming worker re-registering after a lost lease is idempotent.
        // Registration goes through the tenant-pinned catalog, so the row lands
        // under the job's tenant.
        catalog
            .register_model(jammi_db::catalog::model_repo::RegisterModelParams {
                model_id: &output_model_id,
                version: 1,
                model_type: "fine-tuned",
                backend: "candle",
                task,
                base_model_id: Some(&base_model),
                artifact_path: training.adapter_path.to_str(),
                config_json: None,
            })
            .await
            .map_err(WorkerJobError::from)?;

        Ok(RunSuccess {
            model_id: output_model_id,
            metrics: Some(training.metrics_json),
            staged: training.staged,
        })
    }
}

/// An RAII guard owning an embedded [`TrainingWorker`]'s background task. On
/// drop it sets the stop flag and aborts the task, so the worker stops claiming
/// new jobs when its owner (the embedded `LocalSession` or the Python
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
    pub fn spawn(session: &Arc<InferenceSession>) -> Self {
        let worker = TrainingWorker::new(session);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_for_task = Arc::clone(&stop);
        let handle = tokio::spawn(async move { worker.run_until(stop_for_task).await });
        Self { handle, stop }
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

/// A successful training run's result: the registered output-model id, the run
/// metrics, and the worker-private staged artifact awaiting promotion. The
/// training path has staged its artifact privately and registered the model-output
/// row pointing at the (not-yet-written) canonical path (the worker for the
/// fine-tune kinds, the predictor's own `persist_predictor` for the predictor
/// kind); the worker then performs the single lease-guarded compare-and-set that
/// records these and flips the job to `completed`, and only on a CAS win promotes
/// `staged` into the canonical path (discarding it on a loss). `metrics` is the
/// run-metrics JSON the CAS writes (the fine-tune loop's loss/step/timing detail;
/// `None` for a kind that records none beyond the terminal flip).
struct RunSuccess {
    model_id: String,
    metrics: Option<String>,
    staged: StagedArtifact,
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
             pairs (anchor, positive), or classification (text, label). For audio triplets, use \
             the same (anchor, positive, negative) columns with task=audio_embedding."
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

    let device = crate::model::backend::candle::select_device(&device_config);
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
                &vb,
            )?
        } else if task == ModelTask::Regression {
            let output_dim = match config.regression_loss.unwrap_or_default() {
                crate::fine_tune::RegressionLoss::Pinball => config.quantile_levels.len(),
                _ => 2,
            };
            crate::fine_tune::lora::build_distribution_head(hidden_size, output_dim, &config, &vb)?
        } else {
            crate::fine_tune::lora::build_projection_head(hidden_size, &config, &vb)?
        };
        crate::fine_tune::target::TrainingTarget::ProjectionHead { head }
    } else {
        let (encoder, adapter_cfg) =
            build_encoder_adapters(&base_model, &catalog, &config, &varmap, &device)?;
        crate::fine_tune::target::TrainingTarget::EncoderAdapters(Box::new(
            crate::fine_tune::target::EncoderAdaptersTarget {
                encoder,
                adapter_cfg,
            },
        ))
    };

    let mut training_loop =
        crate::fine_tune::trainer::TrainingLoopBuilder::new(target, varmap, config)
            .base_model(base_model_arc)
            .job_id(job_id)
            .worker_id(worker_id)
            .catalog(Arc::clone(&catalog))
            .artifact_dir(artifact_dir)
            .device(device.clone())
            .cancel(cancel)
            .build()?;

    training_loop.run(&data_loader)
}

/// Construct an encoder-adapters target: load the frozen backbone weights from
/// the catalog artifact path, wrap the configured target modules with LoRA, and
/// return both the resulting encoder and the persisted adapter metadata that
/// pairs with the trained tensors on disk.
fn build_encoder_adapters(
    base_model_id: &str,
    catalog: &Arc<Catalog>,
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
            let path = std::path::PathBuf::from(p);
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
}
