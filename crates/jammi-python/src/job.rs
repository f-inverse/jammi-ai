use std::sync::Arc;

use pyo3::prelude::*;

use jammi_ai::fine_tune::training_job::TrainingJob;
use jammi_ai::session::InferenceSession;

use crate::convert::serializable_to_pydict;
use crate::error::to_pyerr;

/// Python TrainingJob handle.
#[pyclass(name = "TrainingJob")]
pub struct PyTrainingJob {
    inner: TrainingJob,
    runtime: Arc<tokio::runtime::Runtime>,
    /// The session `inner` was submitted against, held so [`Self::metrics`]
    /// can reach the catalog's `training_jobs.metrics` column directly —
    /// `TrainingJob`'s own `catalog` field is private to `jammi-ai`, so this
    /// binding carries its own handle rather than growing that crate's
    /// public surface for a read only this binding needs.
    session: Arc<InferenceSession>,
}

impl PyTrainingJob {
    pub fn new(
        inner: TrainingJob,
        runtime: Arc<tokio::runtime::Runtime>,
        session: Arc<InferenceSession>,
    ) -> Self {
        Self {
            inner,
            runtime,
            session,
        }
    }
}

#[pymethods]
impl PyTrainingJob {
    /// The unique job ID.
    #[getter]
    fn job_id(&self) -> &str {
        &self.inner.job_id
    }

    /// The output model ID (set after completion).
    #[getter]
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    /// Current status from the catalog.
    fn status(&self) -> PyResult<String> {
        self.runtime.block_on(self.inner.status()).map_err(to_pyerr)
    }

    /// Block until the job reaches a terminal state (completed or failed).
    fn wait(&self) -> PyResult<()> {
        self.runtime.block_on(self.inner.wait()).map_err(to_pyerr)
    }

    /// Run metrics recorded for this job, as a dict.
    ///
    /// This is exactly what the catalog's `training_jobs.metrics` column
    /// carries — the run-summary blob the trainer hands the worker at
    /// completion (`final_loss`, `early_stopping_metric`, `total_steps`,
    /// `started_at`, `completed_at`), or the `error_message` blob a failed
    /// attempt records instead. Returns `{}` for a job that has not yet
    /// recorded any metrics (e.g. still queued or running before its first
    /// stamp).
    ///
    /// Per-epoch train/val loss curves are NOT part of this surface: the
    /// trainer computes `avg_train_loss`/`avg_val_loss` at every epoch
    /// boundary (`TrainingLoop::run`, `crates/jammi-ai/src/fine_tune/trainer.rs`)
    /// but only logs them (`tracing::info!`) — it never accumulates them into
    /// a `Vec` or folds them into the metrics JSON it returns, so no curve is
    /// retained anywhere this binding (or any consumer) can reach. Surfacing
    /// curves needs a `jammi-ai` change: accumulate `(epoch, avg_train_loss,
    /// avg_val_loss)` across the loop and add it to `TrainingResult::metrics_json`
    /// (`trainer.rs`, around the `let metrics_json = serde_json::json!({...})`
    /// at the end of `run`) — out of this crate's scope.
    fn metrics(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let record = self
            .runtime
            .block_on(self.session.catalog().get_training_job(&self.inner.job_id))
            .map_err(to_pyerr)?;
        let value: serde_json::Value = record
            .metrics
            .as_deref()
            .and_then(|raw| serde_json::from_str(raw).ok())
            .unwrap_or_else(|| serde_json::json!({}));
        serializable_to_pydict(py, &value)
    }
}
