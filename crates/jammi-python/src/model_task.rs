//! Python-facing wrapper for [`jammi_ai::model::ModelTask`].
//!
//! Exposes the engine's `ModelTask` enum to Python as a pyclass enum, surfacing
//! every task variant with its canonical catalog snake-case spelling
//! (`as_db_str` / `from_str` mirror [`jammi_db::ModelTask::try_from_db_str`]) so
//! a caller can name a task as a typed value rather than a bare string.

use pyo3::prelude::*;

use jammi_ai::model::ModelTask;

use crate::error::to_pyerr;

/// Python enum surfacing every variant of the engine's [`ModelTask`].
/// Wire spelling matches the catalog's snake-case strings exactly.
#[pyclass(name = "ModelTask", eq, hash, frozen, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyModelTask {
    TextEmbedding,
    ImageEmbedding,
    AudioEmbedding,
    Classification,
    Ner,
    Regression,
}

#[pymethods]
impl PyModelTask {
    /// Canonical snake-case string stored in the catalog.
    fn as_db_str(&self) -> &'static str {
        ModelTask::from(*self).as_db_str()
    }

    fn __str__(&self) -> &'static str {
        self.as_db_str()
    }

    fn __repr__(&self) -> String {
        format!("ModelTask.{self:?}")
    }

    /// Parse a catalog string into a `ModelTask` enum value. Mirrors
    /// `jammi_db::ModelTask::try_from_db_str` exactly.
    #[staticmethod]
    fn from_str(s: &str) -> PyResult<Self> {
        ModelTask::try_from_db_str(s)
            .map(Self::from)
            .map_err(to_pyerr)
    }
}

impl From<PyModelTask> for ModelTask {
    fn from(value: PyModelTask) -> Self {
        match value {
            PyModelTask::TextEmbedding => ModelTask::TextEmbedding,
            PyModelTask::ImageEmbedding => ModelTask::ImageEmbedding,
            PyModelTask::AudioEmbedding => ModelTask::AudioEmbedding,
            PyModelTask::Classification => ModelTask::Classification,
            PyModelTask::Ner => ModelTask::Ner,
            PyModelTask::Regression => ModelTask::Regression,
        }
    }
}

impl From<ModelTask> for PyModelTask {
    fn from(value: ModelTask) -> Self {
        match value {
            ModelTask::TextEmbedding => PyModelTask::TextEmbedding,
            ModelTask::ImageEmbedding => PyModelTask::ImageEmbedding,
            ModelTask::AudioEmbedding => PyModelTask::AudioEmbedding,
            ModelTask::Classification => PyModelTask::Classification,
            ModelTask::Ner => PyModelTask::Ner,
            ModelTask::Regression => PyModelTask::Regression,
        }
    }
}
