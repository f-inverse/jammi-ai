//! Python bindings for the per-query audit primitive (spec J2).
//!
//! Surfaces a `PerQueryAudit` record pyclass and a `db.audit` handle with
//! `log`, `fetch_by_query_id`, and `fetch_recent`. The handle shares the
//! `Database`'s `InferenceSession` and tokio runtime so audit writes observe
//! the same tenant binding and trigger broker as every other call.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::PyErr;
use uuid::Uuid;

use jammi_ai::session::InferenceSession;
use jammi_db::audit::{self, AuditError, EnvSigningKeyStore, PerQueryAudit};

use crate::convert::serializable_to_pydict;

/// Map a substrate [`AuditError`] to a Python exception.
///
/// `AuditError` is the audit module's own taxonomy and intentionally does not
/// collapse into `JammiError` (that would force callers to substring-match on
/// generic messages), so the binding maps it directly. Validation-shaped
/// variants surface as `ValueError`; everything else as `RuntimeError`.
fn audit_err(e: AuditError) -> PyErr {
    match &e {
        AuditError::LengthMismatch { .. }
        | AuditError::LineageTooLarge { .. }
        | AuditError::SignatureMismatch(_)
        | AuditError::NoTenantBinding => pyo3::exceptions::PyValueError::new_err(e.to_string()),
        _ => pyo3::exceptions::PyRuntimeError::new_err(e.to_string()),
    }
}

/// A per-query audit record. Mirrors the Rust `PerQueryAudit` struct.
///
/// Construct one before a `db.audit.log([...])` call. `tenant_id` and
/// `signature` are populated by the substrate on write and are read-only here;
/// records returned by `fetch_*` carry both.
#[pyclass(name = "PerQueryAudit", from_py_object)]
#[derive(Clone)]
pub struct PyPerQueryAudit {
    pub(crate) inner: PerQueryAudit,
}

#[pymethods]
impl PyPerQueryAudit {
    /// Build a new, unsigned record. `query_id` is a UUID string;
    /// `query_lineage` is any JSON-serializable Python object (dict/list/etc.).
    #[new]
    #[pyo3(signature = (
        *,
        query_id,
        model_id,
        model_version,
        query_lineage,
        top_k_result_ids,
        retrieval_scores,
    ))]
    fn new(
        py: Python<'_>,
        query_id: &str,
        model_id: String,
        model_version: String,
        query_lineage: Py<PyAny>,
        top_k_result_ids: Vec<String>,
        retrieval_scores: Vec<f32>,
    ) -> PyResult<Self> {
        let qid = Uuid::parse_str(query_id).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid query_id '{query_id}': {e}"))
        })?;
        let lineage = pyobj_to_json(py, &query_lineage)?;
        let inner = PerQueryAudit::new(
            qid,
            model_id,
            model_version,
            lineage,
            top_k_result_ids,
            retrieval_scores,
        )
        .map_err(audit_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn query_id(&self) -> String {
        self.inner.query_id.to_string()
    }

    #[getter]
    fn tenant_id(&self) -> Option<String> {
        self.inner.tenant_id.clone()
    }

    #[getter]
    fn model_id(&self) -> String {
        self.inner.model_id.clone()
    }

    #[getter]
    fn model_version(&self) -> String {
        self.inner.model_version.clone()
    }

    #[getter]
    fn query_lineage(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        serializable_to_pydict(py, &self.inner.query_lineage)
    }

    #[getter]
    fn top_k_result_ids(&self) -> Vec<String> {
        self.inner.top_k_result_ids.clone()
    }

    #[getter]
    fn retrieval_scores(&self) -> Vec<f32> {
        self.inner.retrieval_scores.clone()
    }

    /// RFC3339 UTC timestamp of when the search executed.
    #[getter]
    fn executed_at(&self) -> String {
        self.inner.executed_at.to_rfc3339()
    }

    #[getter]
    fn signature(&self) -> String {
        self.inner.signature.clone()
    }

    /// Verify this record's HMAC signature, re-deriving the per-tenant secret
    /// using the env-backed audit signing key store. Raises if the signature
    /// does not verify or the master key is unavailable.
    fn verify(&self) -> PyResult<()> {
        audit::verify_with_store(&self.inner, &EnvSigningKeyStore).map_err(audit_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "PerQueryAudit(query_id={}, model_id={}, model_version={}, results={})",
            self.inner.query_id,
            self.inner.model_id,
            self.inner.model_version,
            self.inner.top_k_result_ids.len()
        )
    }
}

/// Handle to the audit primitive, returned by `Database.audit`.
#[pyclass(name = "AuditHandle")]
pub struct PyAuditHandle {
    pub(crate) session: Arc<InferenceSession>,
    pub(crate) runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl PyAuditHandle {
    /// Sign and persist a batch of `PerQueryAudit` records for the bound
    /// tenant; publishes them to the `jammi.audit.search.v1` trigger topic.
    fn log(&self, records: Vec<PyPerQueryAudit>) -> PyResult<()> {
        let recs: Vec<PerQueryAudit> = records.into_iter().map(|r| r.inner).collect();
        self.runtime
            .block_on(self.session.audit().log(recs))
            .map_err(audit_err)
    }

    /// Fetch one record by query id (tenant-scoped). Returns `None` if absent.
    fn fetch_by_query_id(&self, query_id: &str) -> PyResult<Option<PyPerQueryAudit>> {
        let qid = Uuid::parse_str(query_id).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid query_id '{query_id}': {e}"))
        })?;
        let rec = self
            .runtime
            .block_on(self.session.audit().fetch_by_query_id(qid))
            .map_err(audit_err)?;
        Ok(rec.map(|inner| PyPerQueryAudit { inner }))
    }

    /// Fetch the most recent records (tenant-scoped), newest first.
    #[pyo3(signature = (*, limit=20))]
    fn fetch_recent(&self, limit: usize) -> PyResult<Vec<PyPerQueryAudit>> {
        let recs = self
            .runtime
            .block_on(self.session.audit().fetch_recent(limit))
            .map_err(audit_err)?;
        Ok(recs
            .into_iter()
            .map(|inner| PyPerQueryAudit { inner })
            .collect())
    }
}

/// Convert a Python object to `serde_json::Value` via the `json` module so any
/// JSON-serializable object (dict/list/str/number/bool/None) is accepted for
/// `query_lineage`.
fn pyobj_to_json(py: Python<'_>, obj: &Py<PyAny>) -> PyResult<serde_json::Value> {
    let json_mod = py.import("json")?;
    let dumps = json_mod.getattr("dumps")?;
    let s: String = dumps.call1((obj.bind(py),))?.extract()?;
    serde_json::from_str(&s).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "query_lineage is not JSON-serializable: {e}"
        ))
    })
}
