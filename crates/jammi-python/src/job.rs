use std::sync::Arc;

use pyo3::prelude::*;

use jammi_ai::jobs::{JobHandle, JobResult};
use jammi_db::error::JammiError;

use crate::convert::serializable_to_pydict;
use crate::error::to_pyerr;

/// A job's recorded progress as it crosses to Python: the phase's spelling,
/// then how much of it is done and its total.
type RecordedProgress = (String, Option<u64>, Option<u64>);

/// Python `Job` handle: the engine's [`JobHandle`] — the one a submit returns
/// and an attach by id reconstructs — read through this connection's runtime.
/// Every verb reads the catalog, so a handle attached in a fresh process reads
/// what the submitter's would. The Python wrapper (`jammi.EmbeddedJob`) types
/// the spellings this binding hands across.
#[pyclass(name = "Job")]
pub struct PyJob {
    handle: JobHandle,
    runtime: Arc<tokio::runtime::Runtime>,
}

impl PyJob {
    pub fn new(handle: JobHandle, runtime: Arc<tokio::runtime::Runtime>) -> Self {
        Self { handle, runtime }
    }

    fn record(&self) -> PyResult<jammi_db::catalog::jobs_repo::JobRecord> {
        crate::released(&self.runtime, self.handle.record()).map_err(to_pyerr)
    }

    /// Decode a present JSON column of this job's row; a malformed one is a
    /// catalog data-integrity fault, never read as absent.
    fn decode_blob(&self, py: Python<'_>, what: &str, raw: &str) -> PyResult<Py<PyAny>> {
        let value: serde_json::Value = serde_json::from_str(raw).map_err(|parse_err| {
            to_pyerr(JammiError::Catalog(format!(
                "job {}: {what} blob failed to parse as JSON: {parse_err}",
                self.handle.job_id(),
            )))
        })?;
        serializable_to_pydict(py, &value)
    }
}

#[pymethods]
impl PyJob {
    /// The unique job ID.
    #[getter]
    fn job_id(&self) -> &str {
        self.handle.job_id()
    }

    /// The kind's spelling (e.g. `"fine_tune"`).
    #[getter]
    fn kind(&self) -> String {
        self.handle.kind().to_string()
    }

    /// The model a training kind registers (the two LoRA fine-tune kinds'
    /// deterministic `jammi:fine-tuned:{job_id}`, the context predictor's
    /// caller-chosen id), stamped at submission; `None` for a compute kind.
    #[getter]
    fn output_model_id(&self) -> Option<&str> {
        self.handle.output_model_id()
    }

    /// The current status's spelling.
    fn status(&self) -> PyResult<String> {
        crate::released(&self.runtime, self.handle.status())
            .map(|status| status.to_string())
            .map_err(to_pyerr)
    }

    /// The latest recorded progress as `(phase, done, total)` — the phase's
    /// spelling and how far through it is, in its unit (`total` is `None`
    /// when not known up front) — or `None` before the executor records any.
    fn progress(&self) -> PyResult<Option<RecordedProgress>> {
        Ok(self
            .record()?
            .progress
            .map(|p| (p.phase.to_string(), p.done, p.total)))
    }

    /// Request cancellation. `True` means the request took effect on a still
    /// non-terminal row: an unclaimed job is cancelled at once, a running one
    /// at its executor's next checkpoint boundary, and [`Self::wait`] then
    /// raises the cancellation. `False` when the job is already terminal.
    fn cancel(&self) -> PyResult<bool> {
        crate::released(&self.runtime, self.handle.cancel()).map_err(to_pyerr)
    }

    /// Block until the job reaches a terminal state, returning the tagged
    /// terminal result as a dict — the engine's [`JobResult`]: `{"kind":
    /// "model", "model_id", "artifact_path", "metrics", "cache_outcome"}` for
    /// a training kind (`metrics` is the raw run-summary text; read
    /// [`Self::metrics`] for the parsed form), `{"kind": "table", "table",
    /// "cache_outcome"}` for a compute kind. Raises the job's recorded error
    /// when it failed or was cancelled.
    fn wait(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let result = crate::released(&self.runtime, self.handle.wait()).map_err(to_pyerr)?;
        serializable_to_pydict(py, &result)
    }

    /// The run summary a completed training job recorded, as a dict
    /// (`best_epoch`, `best_loss`, `early_stopping_metric`, `total_steps`,
    /// `started_at`, `completed_at`, and the per-epoch `train_loss_curve` /
    /// `val_loss_curve`). `{}` before completion and for a compute kind.
    /// Raises `jammi.errors.BackendError` when the recorded result or its
    /// summary does not decode — a catalog fault, never read as absent.
    fn metrics(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match JobResult::recorded(&self.record()?).map_err(to_pyerr)? {
            Some(JobResult::Model {
                metrics: Some(raw), ..
            }) => self.decode_blob(py, "metrics", &raw),
            _ => serializable_to_pydict(py, &serde_json::json!({})),
        }
    }

    /// This job's per-attempt acceleration determination, as a dict, or
    /// `None` for a row no code has recorded one on.
    ///
    /// A present value's `"state"` is `"pending"` — the submission-time
    /// marker: the job exists and no claimant has determined anything yet —
    /// or one of the determination outcomes `"determined"`,
    /// `"not_applicable"` and `"undetermined"`, the last always carrying a
    /// `"reason"`. A job that ends without a determination has its marker
    /// retired to `"undetermined"`, and a requeued job is reset to
    /// `"pending"`, so a read can land on any of the four. Everything beside
    /// `"state"` is the producer's to define.
    ///
    /// Raises `jammi.errors.BackendError` when the column is present but
    /// does not parse as JSON.
    fn acceleration_report(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.record()?.acceleration_report {
            None => Ok(py.None()),
            Some(raw) => self.decode_blob(py, "acceleration_report", &raw),
        }
    }
}
