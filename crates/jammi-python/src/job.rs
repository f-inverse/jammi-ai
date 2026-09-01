use std::sync::Arc;
use std::time::Duration;

use pyo3::prelude::*;

use jammi_ai::fine_tune::spec::TrainingSpec;
use jammi_ai::fine_tune::training_job::TrainingJob;
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::status::TrainingJobStatus;
use jammi_db::catalog::training_repo::TrainingJobRecord;
use jammi_db::error::JammiError;

use crate::convert::serializable_to_pydict;
use crate::error::to_pyerr;

/// The two ways a [`PyTrainingJob`] handle comes to exist: freshly minted by a
/// submit call (`_start_training_proto`, which already has the engine's own
/// [`TrainingJob`] to hand over), or reconstructed for a job this connection
/// never submitted (`PyDatabase::training_job`, the attach-by-id path — the
/// embedded peer of the remote client's `RemoteTrainingJob`, which can always
/// be built straight from a `job_id` because every one of its verbs re-fetches
/// state over the wire per call).
///
/// `TrainingJob::new` is `pub(crate)` to `jammi-ai` — deliberately: a
/// `TrainingJob` is minted alongside the catalog row it names, by the same
/// submit call, so its `model_id` is always the value that call itself
/// resolved, never re-derived. The `Attached` arm therefore does not attempt
/// to construct a `jammi-ai` `TrainingJob` at all; it carries exactly the
/// `job_id` + resolved `model_id` this binding needs and drives `status()`
/// `wait()` / `metrics()` straight off the catalog record, through the same
/// public `Catalog::get_training_job` read [`PyTrainingJob::metrics`] already
/// used for both arms before this split existed.
enum JobState {
    Submitted(TrainingJob),
    Attached { job_id: String, model_id: String },
}

/// Python TrainingJob handle.
#[pyclass(name = "TrainingJob")]
pub struct PyTrainingJob {
    inner: JobState,
    runtime: Arc<tokio::runtime::Runtime>,
    /// The session this handle reads through. For the `Submitted` arm this is
    /// also the session `inner` was submitted against; for `Attached` it is
    /// simply the connection `training_job(job_id)` was called on. Held
    /// directly (rather than reached through `inner`) because `TrainingJob`'s
    /// own `catalog` field is private to `jammi-ai` — this binding carries its
    /// own handle rather than growing that crate's public surface for a read
    /// only this binding needs.
    session: Arc<InferenceSession>,
}

impl PyTrainingJob {
    pub fn new(
        inner: TrainingJob,
        runtime: Arc<tokio::runtime::Runtime>,
        session: Arc<InferenceSession>,
    ) -> Self {
        Self {
            inner: JobState::Submitted(inner),
            runtime,
            session,
        }
    }

    /// Attach to an existing job by id — a handle for a job this connection
    /// never submitted, reconstructed from the catalog row alone (the embedded
    /// peer of the remote client always-attach-by-id `RemoteTrainingJob`).
    ///
    /// A nonexistent `job_id` raises the SAME typed not-found the catalog read
    /// (`Catalog::get_training_job`) itself produces — there is no separate
    /// existence check to drift from it.
    ///
    /// `model_id` is resolved eagerly here (once, synchronously) rather than
    /// on every `model_id()` call, matching how the `Submitted` arm already
    /// carries it as a plain field: prefer the catalog's own
    /// `output_model_id` once the job has completed and stamped it; before
    /// that, re-derive the same deterministic value a submit call would have
    /// handed back, from the persisted `training_spec` — see
    /// [`resolve_attach_model_id`].
    pub fn attach(
        job_id: String,
        runtime: Arc<tokio::runtime::Runtime>,
        session: Arc<InferenceSession>,
    ) -> PyResult<Self> {
        let record = runtime
            .block_on(session.catalog().get_training_job(&job_id))
            .map_err(to_pyerr)?;
        let model_id = resolve_attach_model_id(&job_id, &record)?;
        Ok(Self {
            inner: JobState::Attached { job_id, model_id },
            runtime,
            session,
        })
    }

    fn job_id_str(&self) -> &str {
        match &self.inner {
            JobState::Submitted(job) => &job.job_id,
            JobState::Attached { job_id, .. } => job_id,
        }
    }
}

#[pymethods]
impl PyTrainingJob {
    /// The unique job ID.
    #[getter]
    fn job_id(&self) -> &str {
        self.job_id_str()
    }

    /// The output model ID (set after completion).
    #[getter]
    fn model_id(&self) -> &str {
        match &self.inner {
            JobState::Submitted(job) => job.model_id(),
            JobState::Attached { model_id, .. } => model_id,
        }
    }

    /// Current status from the catalog.
    fn status(&self) -> PyResult<String> {
        match &self.inner {
            JobState::Submitted(job) => self.runtime.block_on(job.status()).map_err(to_pyerr),
            JobState::Attached { job_id, .. } => {
                let record = self
                    .runtime
                    .block_on(self.session.catalog().get_training_job(job_id))
                    .map_err(to_pyerr)?;
                Ok(record.status)
            }
        }
    }

    /// Block until the job reaches a terminal state (completed or failed).
    fn wait(&self) -> PyResult<()> {
        match &self.inner {
            JobState::Submitted(job) => self.runtime.block_on(job.wait()).map_err(to_pyerr),
            JobState::Attached { job_id, .. } => self
                .runtime
                .block_on(poll_until_terminal(&self.session, job_id))
                .map_err(to_pyerr),
        }
    }

    /// Run metrics recorded for this job, as a dict.
    ///
    /// This is exactly what the catalog's `training_jobs.metrics` column
    /// carries — the run-summary blob the trainer hands the worker at
    /// completion (`final_loss`, `early_stopping_metric`, `total_steps`,
    /// `started_at`, `completed_at`), or the `error_message` blob a failed
    /// attempt records instead. Returns `{}` for a job that has not yet
    /// recorded any metrics (e.g. still queued or running before its first
    /// stamp — the column is absent). Raises `jammi.errors.BackendError` if
    /// the column IS present but fails to parse as JSON — a catalog
    /// data-integrity fault, never silently folded into the absent `{}` case
    /// (matches the remote transport's `metrics()`).
    ///
    /// Per-epoch train/val loss curves ARE part of this surface (issue #441):
    /// the trainer accumulates `(epoch, avg_train_loss)` / `(epoch,
    /// avg_val_loss)` across `TrainingLoop::run`
    /// (`crates/jammi-ai/src/fine_tune/trainer.rs`) and folds them into the
    /// returned metrics JSON as the `train_loss_curve` / `val_loss_curve`
    /// arrays, so this dict carries them exactly like every other recorded
    /// metric.
    fn metrics(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let record = self
            .runtime
            .block_on(self.session.catalog().get_training_job(self.job_id_str()))
            .map_err(to_pyerr)?;
        let value: serde_json::Value = match record.metrics.as_deref() {
            // Absent — the job has not yet recorded any metrics. `{}`.
            None => serde_json::json!({}),
            // Present — must parse. A present-but-unparseable blob is a catalog
            // data-integrity fault, distinct from "no metrics yet" — surfaced
            // LOUDLY (never silently folded into the absent `{}` case) so it
            // cannot be mistaken for a job that simply hasn't reported yet.
            // Matches the remote transport's `metrics()`, which raises rather
            // than returning `{}` for the same malformed-but-present state.
            Some(raw) => serde_json::from_str(raw).map_err(|parse_err| {
                to_pyerr(JammiError::Catalog(format!(
                    "training job {}: metrics blob failed to parse as JSON: {parse_err}",
                    self.job_id_str(),
                )))
            })?,
        };
        serializable_to_pydict(py, &value)
    }
}

/// Poll `catalog.get_training_job(job_id)` until it reaches a terminal state,
/// mirroring `jammi_ai::fine_tune::training_job::TrainingJob::wait` exactly
/// (same 100ms cadence, same terminal classification) — reimplemented here
/// rather than called because that method lives on a type this binding cannot
/// construct for a job it did not submit (see [`JobState`]'s doc).
async fn poll_until_terminal(
    session: &InferenceSession,
    job_id: &str,
) -> jammi_db::error::Result<()> {
    loop {
        let record = session.catalog().get_training_job(job_id).await?;
        let status: TrainingJobStatus = record
            .status
            .parse()
            .map_err(|e| JammiError::FineTune(format!("{e}")))?;
        match status {
            TrainingJobStatus::Completed => return Ok(()),
            TrainingJobStatus::Failed => {
                let msg = record.error_message.unwrap_or_else(|| "Job failed".into());
                return Err(JammiError::FineTune(msg));
            }
            _ => tokio::time::sleep(Duration::from_millis(100)).await,
        }
    }
}

/// Resolve the output model id an attached handle reports, from the catalog
/// row alone — no in-memory carryover from a submit call, because attach never
/// had one.
///
/// Once the job has completed, `record.output_model_id` (stamped by the
/// worker's finalize) is authoritative and used directly. Before that, the
/// value is re-derived from the persisted `training_spec` by the SAME rule
/// [`InferenceSession::submit_fine_tune_spec`] /
/// [`InferenceSession::train_context_predictor`] apply at submit time: the two
/// LoRA fine-tune kinds mint the deterministic `jammi:fine-tuned:{job_id}`
/// (checked directly against `record.kind`, no spec decode needed — the value
/// depends only on `job_id`); the context-predictor kind carries a
/// caller-chosen id inside `predictor_spec.model_id`, which is NOT derivable
/// from `job_id` alone, so that one arm does decode the persisted spec.
fn resolve_attach_model_id(job_id: &str, record: &TrainingJobRecord) -> PyResult<String> {
    if let Some(model_id) = &record.output_model_id {
        return Ok(model_id.clone());
    }
    match record.kind.as_str() {
        "fine_tune" | "graph_fine_tune" => Ok(format!("jammi:fine-tuned:{job_id}")),
        "context_predictor" => {
            let raw = record.training_spec.as_deref().ok_or_else(|| {
                to_pyerr(JammiError::Catalog(format!(
                    "training job {job_id}: missing persisted training_spec; \
                     cannot resolve model_id before completion"
                )))
            })?;
            let spec: TrainingSpec = serde_json::from_str(raw).map_err(|parse_err| {
                to_pyerr(JammiError::Catalog(format!(
                    "training job {job_id}: training_spec failed to parse as JSON: {parse_err}"
                )))
            })?;
            match spec {
                TrainingSpec::ContextPredictor { predictor_spec, .. } => {
                    Ok(predictor_spec.model_id)
                }
                _ => Err(to_pyerr(JammiError::Catalog(format!(
                    "training job {job_id}: persisted training_spec kind does not match \
                     catalog kind 'context_predictor'"
                )))),
            }
        }
        other => Err(to_pyerr(JammiError::Catalog(format!(
            "training job {job_id}: unrecognised training job kind '{other}'"
        )))),
    }
}
