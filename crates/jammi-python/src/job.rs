use std::sync::Arc;

use pyo3::prelude::*;

use jammi_ai::fine_tune::job::FineTuneJob;

use crate::error::to_pyerr;

/// Python FineTuneJob handle.
#[pyclass(name = "FineTuneJob")]
pub struct PyFineTuneJob {
    inner: FineTuneJob,
    runtime: Arc<tokio::runtime::Runtime>,
}

impl PyFineTuneJob {
    pub fn new(inner: FineTuneJob, runtime: Arc<tokio::runtime::Runtime>) -> Self {
        Self { inner, runtime }
    }
}

#[pymethods]
impl PyFineTuneJob {
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
        self.inner.status().map_err(to_pyerr)
    }

    /// Block until the job reaches a terminal state (completed or failed).
    fn wait(&self) -> PyResult<()> {
        self.runtime.block_on(self.inner.wait()).map_err(to_pyerr)
    }
}
