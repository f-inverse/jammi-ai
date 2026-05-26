//! Python-facing wrapper for [`jammi_ai::model::ModelTask`].
//!
//! Exposes the engine's `ModelTask` enum to Python as a pyclass enum so call
//! sites like `db.infer(..., task=ModelTask.TextEmbedding, ...)` exchange
//! typed values across the FFI boundary. Strings are still accepted at every
//! task-bearing argument — they decode through
//! [`jammi_db::ModelTask::try_from_db_str`] exactly once at the binding
//! edge, so the Rust side never sees `&str` for `task`.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3::Borrowed;

use jammi_ai::model::ModelTask;

use crate::error::to_pyerr;

/// Python enum surfacing every variant of the engine's [`ModelTask`].
/// Wire spelling matches the catalog's snake-case strings exactly.
#[pyclass(name = "ModelTask", eq, hash, frozen, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyModelTask {
    TextEmbedding,
    ImageEmbedding,
    Classification,
    Ner,
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
            PyModelTask::Classification => ModelTask::Classification,
            PyModelTask::Ner => ModelTask::Ner,
        }
    }
}

impl From<ModelTask> for PyModelTask {
    fn from(value: ModelTask) -> Self {
        match value {
            ModelTask::TextEmbedding => PyModelTask::TextEmbedding,
            ModelTask::ImageEmbedding => PyModelTask::ImageEmbedding,
            ModelTask::Classification => PyModelTask::Classification,
            ModelTask::Ner => PyModelTask::Ner,
        }
    }
}

/// Argument shim accepted by every Python-facing `task=` parameter — either
/// a `ModelTask` enum value (preferred) or the catalog snake-case string.
/// Strings are decoded through `ModelTask::try_from_db_str` exactly once at
/// the binding boundary; the Rust call site sees a typed `ModelTask`.
pub struct ModelTaskArg(pub ModelTask);

impl<'a, 'py> FromPyObject<'a, 'py> for ModelTaskArg {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(typed) = ob.extract::<PyModelTask>() {
            return Ok(ModelTaskArg(typed.into()));
        }
        if let Ok(s) = ob.cast::<PyString>() {
            let raw: String = s.str()?.to_string();
            let task = ModelTask::try_from_db_str(&raw).map_err(to_pyerr)?;
            return Ok(ModelTaskArg(task));
        }
        Err(PyTypeError::new_err(
            "task must be a ModelTask enum value or its snake_case string \
             (text_embedding, image_embedding, classification, ner)",
        ))
    }
}
