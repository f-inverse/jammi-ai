//! Python bindings for the ephemeral session-storage primitive (spec J6).
//!
//! Surfaces `db.ephemeral_session(timeout_seconds=...)` returning a
//! context-manager `EphemeralSession`. Tables created inside the `with` block
//! are auto-deleted on exit (which calls `close()`), and every lifecycle
//! transition publishes to `jammi.audit.session_lifecycle.v1`. The handle shares
//! the `Database`'s `InferenceSession` and tokio runtime, so it observes the
//! same tenant binding and trigger broker as every other call.

use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::PyErr;
use pyo3_arrow::{PySchema, PyTable};

use jammi_db::ephemeral::{EphemeralError, EphemeralSession};

use crate::convert::batches_to_pyarrow;

/// Map an [`EphemeralError`] to a Python exception. Validation-shaped variants
/// surface as `ValueError`; everything else as `RuntimeError`.
pub(crate) fn ephemeral_err(e: EphemeralError) -> PyErr {
    match &e {
        EphemeralError::NoTenantBinding
        | EphemeralError::NameTooLong { .. }
        | EphemeralError::DuplicateTable(_)
        | EphemeralError::UnknownTable(_) => PyValueError::new_err(e.to_string()),
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

/// A session-scoped storage context. Use as a context manager:
///
/// ```python
/// with db.ephemeral_session(timeout_seconds=3600) as ephem:
///     ephem.create_ephemeral_table("query_images", schema=..., primary_key=["image_id"])
///     ephem.insert("query_images", batch)
/// # all ephemeral tables deleted on exit; a `closed` event is published
/// ```
#[pyclass(name = "EphemeralSession")]
pub struct PyEphemeralSession {
    /// `None` once the session has been closed (explicitly or via `__exit__`).
    inner: Mutex<Option<EphemeralSession>>,
    runtime: Arc<tokio::runtime::Runtime>,
}

impl PyEphemeralSession {
    pub(crate) fn new(inner: EphemeralSession, runtime: Arc<tokio::runtime::Runtime>) -> Self {
        Self {
            inner: Mutex::new(Some(inner)),
            runtime,
        }
    }

    /// Run `f` against the live session, erroring if it has been closed.
    fn with_session<T>(
        &self,
        f: impl FnOnce(&mut EphemeralSession) -> Result<T, EphemeralError>,
    ) -> PyResult<T> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("ephemeral session lock poisoned"))?;
        let session = guard
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("ephemeral session is already closed"))?;
        f(session).map_err(ephemeral_err)
    }
}

#[pymethods]
impl PyEphemeralSession {
    /// The session's unique id (UUID string).
    #[getter]
    fn session_id(&self) -> PyResult<String> {
        self.with_session(|s| Ok(s.session_id().to_string()))
    }

    /// The tenant this session is pinned to (UUID string).
    #[getter]
    fn tenant_id(&self) -> PyResult<String> {
        self.with_session(|s| Ok(s.tenant().to_string()))
    }

    /// Create a session-scoped table. `schema` is any Arrow PyCapsule schema
    /// (e.g. `pyarrow.Schema`); `primary_key` is a non-empty list of columns.
    /// The table is auto-deleted when the session ends.
    #[pyo3(signature = (name, *, schema, primary_key))]
    fn create_ephemeral_table(
        &self,
        name: &str,
        schema: PySchema,
        primary_key: Vec<String>,
    ) -> PyResult<()> {
        let schema_ref = schema.into_inner();
        let rt = Arc::clone(&self.runtime);
        self.with_session(|s| rt.block_on(s.create_ephemeral_table(name, schema_ref, primary_key)))
    }

    /// Append a `pyarrow.Table` to a named ephemeral table. The schema must
    /// match the table's. Returns the number of rows inserted.
    #[pyo3(signature = (name, *, batch))]
    fn insert(&self, name: &str, batch: PyTable) -> PyResult<u64> {
        let (batches, schema) = batch.into_inner();
        let concatenated = if batches.len() == 1 {
            batches
                .into_iter()
                .next()
                .ok_or_else(|| PyValueError::new_err("insert batch was empty"))?
        } else if batches.is_empty() {
            return Ok(0);
        } else {
            arrow::compute::concat_batches(&schema, batches.iter())
                .map_err(|e| PyRuntimeError::new_err(format!("concat insert batches: {e}")))?
        };
        let rt = Arc::clone(&self.runtime);
        self.with_session(|s| rt.block_on(s.insert(name, concatenated)))
    }

    /// Run a read query against a named ephemeral table. `{table}` in `query`
    /// is replaced by the tenant-scoped reference to the table. Returns a
    /// `pyarrow.Table`.
    fn sql(&self, py: Python<'_>, name: &str, query: &str) -> PyResult<Py<PyAny>> {
        let rt = Arc::clone(&self.runtime);
        let batches = self.with_session(|s| rt.block_on(s.sql(name, query)))?;
        batches_to_pyarrow(py, &batches)
    }

    /// Count rows currently stored in a named ephemeral table.
    fn count_rows(&self, name: &str) -> PyResult<u64> {
        let rt = Arc::clone(&self.runtime);
        self.with_session(|s| rt.block_on(s.count_rows(name)))
    }

    /// The fully qualified SQL reference (`mutable.public."<physical id>"`) for a
    /// named ephemeral table. Exposed so callers can query the underlying
    /// physical table through the parent `Database` — e.g. to prove tenant
    /// isolation, where a query run under a *different* parent tenant must scope
    /// the ephemeral rows out.
    fn physical_table_ref(&self, name: &str) -> PyResult<String> {
        self.with_session(|s| s.table_ref(name))
    }

    /// Explicitly close the session: delete all ephemeral tables and publish a
    /// `closed` lifecycle event. Idempotent — closing twice is a no-op.
    fn close(&self) -> PyResult<()> {
        let taken = {
            let mut guard = self
                .inner
                .lock()
                .map_err(|_| PyRuntimeError::new_err("ephemeral session lock poisoned"))?;
            guard.take()
        };
        match taken {
            Some(session) => self
                .runtime
                .block_on(session.close())
                .map_err(ephemeral_err),
            None => Ok(()),
        }
    }

    /// Enter the runtime context, returning `self`.
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Exit the runtime context: close the session (deleting all tables). Does
    /// not suppress exceptions raised inside the `with` block.
    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &self,
        _exc_type: Option<Py<PyAny>>,
        _exc_value: Option<Py<PyAny>>,
        _traceback: Option<Py<PyAny>>,
    ) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }
}
