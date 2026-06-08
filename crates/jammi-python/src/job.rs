use std::sync::Arc;

use pyo3::prelude::*;

use jammi_ai::fine_tune::training_job::TrainingJob;

use crate::error::to_pyerr;

/// Python TrainingJob handle.
#[pyclass(name = "TrainingJob")]
pub struct PyTrainingJob {
    inner: TrainingJob,
    runtime: Arc<tokio::runtime::Runtime>,
}

impl PyTrainingJob {
    pub fn new(inner: TrainingJob, runtime: Arc<tokio::runtime::Runtime>) -> Self {
        Self { inner, runtime }
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
}
