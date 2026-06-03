//! Python binding for the REMOTE [`jammi_ai::Session`] transport (Shape C).
//!
//! [`PyRemoteDatabase`] is the network peer of the embedded
//! [`crate::PyDatabase`]: a Python consumer that wants to drive a *remote*
//! jammi engine over the `jammi.v1` gRPC wire opens one with
//! [`connect_remote`] and calls the transport-agnostic verbs (add-source,
//! encode-query, generate-embeddings, search, infer, the tenant trio) on it.
//!
//! It mirrors the Rust front door exactly — [`Jammi::open`] with
//! [`Target::Remote`] — and then holds the resulting [`Session`]. Every verb
//! delegates straight to the `Session` enum, which dispatches to its `Remote`
//! arm and the one Rust [`jammi_ai::RemoteSession`] gRPC client. There is no
//! second Python-side gRPC client: the binding reuses the same wire path the
//! Rust SDK and `jammi-server` integration tests exercise, so the embedded and
//! remote Python surfaces agree by construction rather than by a parallel
//! reimplementation.
//!
//! The two Flight-SQL-lane verbs (`sql` / `read_vectors`) are not wired on the
//! typed-RPC surface; reaching them on a remote session returns the engine's own
//! typed "not yet available on the remote transport" error, surfaced here as a
//! Python `RuntimeError` — the truthful answer, never a faked success.

use std::sync::Arc;

use jammi_ai::jammi::{Jammi, Target};
use jammi_ai::local_session::{Modality, QueryInput, SearchQuery, SearchRequest, Session};
use jammi_ai::model::ModelTask;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::TenantId;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3::Borrowed;
use std::str::FromStr;
use tonic::transport::Endpoint;

use crate::convert::batches_to_pyarrow;
use crate::error::to_pyerr;
use crate::model_task::ModelTaskArg;

/// Argument shim accepted by every Python-facing `modality=` parameter: the
/// snake-case string `"text"` / `"image"` / `"audio"`. Decoded once at the
/// binding boundary so the Rust call site sees a typed [`Modality`].
struct ModalityArg(Modality);

impl<'a, 'py> FromPyObject<'a, 'py> for ModalityArg {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let s = ob
            .cast::<PyString>()
            .map_err(|_| PyTypeError::new_err("modality must be a string"))?;
        let raw: String = s.str()?.to_string();
        let modality = match raw.as_str() {
            "text" => Modality::Text,
            "image" => Modality::Image,
            "audio" => Modality::Audio,
            other => {
                return Err(PyTypeError::new_err(format!(
                    "modality must be 'text', 'image', or 'audio' (got '{other}')"
                )))
            }
        };
        Ok(ModalityArg(modality))
    }
}

/// A Python `RemoteDatabase` driving a remote jammi engine over gRPC.
///
/// Holds the `Session::Remote` produced by [`Jammi::open`] plus a tokio runtime
/// that drives every wire future. The verbs are the transport-agnostic
/// [`Session`] surface; an embedded `Database` and a `RemoteDatabase` expose the
/// same vocabulary, only the transport differs.
#[pyclass(name = "RemoteDatabase")]
pub struct PyRemoteDatabase {
    session: Session,
    runtime: Arc<tokio::runtime::Runtime>,
}

impl PyRemoteDatabase {
    /// Connect to a `jammi.v1` gRPC endpoint and wrap the resulting remote
    /// [`Session`]. This is the Rust-shaped entry point the `#[pyfunction]
    /// connect_remote` wraps; it threads `Jammi::open(Target::Remote(..))`,
    /// reusing the SDK's single front door rather than constructing a transport
    /// here.
    fn open(endpoint: &str) -> PyResult<Self> {
        let runtime = Arc::new(tokio::runtime::Runtime::new().map_err(to_pyerr)?);
        let ep = Endpoint::from_shared(endpoint.to_string()).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid endpoint: {e}"))
        })?;
        let session = runtime
            .block_on(Jammi::open(Target::Remote(ep)))
            .map_err(to_pyerr)?;
        Ok(Self { session, runtime })
    }

    /// Register a file-shaped source over the wire. The Rust-shaped peer of
    /// [`Self::add_source`] that skips the PyO3 argument shims, so the
    /// integration test can drive the wired `AddSource` verb without a Python
    /// interpreter. Drives the exact same `Session::add_source` verb the Python
    /// method does.
    pub fn add_source_for_test(&self, name: &str, url: &str, format: &str) -> PyResult<()> {
        let file_format: FileFormat = format.parse().map_err(to_pyerr)?;
        let connection = SourceConnection::parse(url, file_format).map_err(to_pyerr)?;
        self.runtime
            .block_on(self.session.add_source(name, SourceType::File, connection))
            .map_err(to_pyerr)
    }

    /// Encode a query over the wire, returning the raw vector. The Rust-shaped
    /// peer of [`Self::encode_query`] that skips the PyO3 argument shims and the
    /// `pyarrow` return path, so the integration test can assert transport
    /// parity without a Python interpreter. Drives the exact same
    /// `Session::encode_query` verb the Python method does.
    pub fn encode_query_for_test(
        &self,
        model: &str,
        query: &str,
        modality: Modality,
    ) -> PyResult<Vec<f32>> {
        self.runtime
            .block_on(self.session.encode_query(
                model,
                QueryInput::Text(query.to_string()),
                modality,
            ))
            .map_err(to_pyerr)
    }

    /// Generate embeddings over the wire, returning the result table name. The
    /// Rust-shaped peer of [`Self::generate_embeddings`] for the integration
    /// test (no PyO3 shims).
    pub fn generate_embeddings_for_test(
        &self,
        source: &str,
        model: &str,
        columns: &[&str],
        key: &str,
        modality: Modality,
    ) -> PyResult<String> {
        let columns: Vec<String> = columns.iter().map(|c| c.to_string()).collect();
        let record = self
            .runtime
            .block_on(
                self.session
                    .generate_embeddings(source, model, &columns, key, modality),
            )
            .map_err(to_pyerr)?;
        Ok(record.table_name)
    }

    /// Run a vector search over the wire, returning the hit row count. The
    /// Rust-shaped peer of [`Self::search`] for the integration test (no
    /// `pyarrow` return path).
    pub fn search_for_test(&self, source: &str, query: Vec<f32>, k: usize) -> PyResult<usize> {
        let request = SearchRequest {
            source_id: source.to_string(),
            query: SearchQuery::Vector(query),
            k,
            filter: None,
            select: Vec::new(),
        };
        let batches = self
            .runtime
            .block_on(self.session.search(request))
            .map_err(to_pyerr)?;
        Ok(batches.iter().map(|b| b.num_rows()).sum())
    }
}

#[pymethods]
impl PyRemoteDatabase {
    /// Bind a tenant scope to this connection. Subsequent reads/writes are
    /// scoped to `tenant_id` (plus globally-scoped rows). Pass an empty string
    /// to clear. Maps to `SessionService.SetTenant` / `ClearTenant` over the
    /// wire, keyed by this session's id.
    fn with_tenant(&self, tenant_id: &str) -> PyResult<()> {
        if tenant_id.is_empty() {
            return self
                .runtime
                .block_on(self.session.unbind_tenant())
                .map_err(to_pyerr);
        }
        let t = TenantId::from_str(tenant_id).map_err(to_pyerr)?;
        self.runtime
            .block_on(self.session.bind_tenant(t))
            .map_err(to_pyerr)
    }

    /// The tenant currently bound to this connection, or `None`. Maps to
    /// `SessionService.GetTenant`.
    fn tenant(&self) -> PyResult<Option<String>> {
        let t = self
            .runtime
            .block_on(self.session.tenant())
            .map_err(to_pyerr)?;
        Ok(t.map(|t| t.to_string()))
    }

    /// Register a file-shaped data source on the remote engine. `url` accepts a
    /// local path (parsed into `file://...`) or any storage URL the server was
    /// compiled with: `s3://bucket/key`, `gs://bucket/key`,
    /// `azure://container/blob`. Maps to `EmbeddingService.AddSource` over the
    /// wire; the embedded peer is [`crate::PyDatabase::add_source`].
    #[pyo3(signature = (name, *, url, format))]
    fn add_source(&self, name: &str, url: &str, format: &str) -> PyResult<()> {
        self.add_source_for_test(name, url, format)
    }

    /// Encode a single query into an embedding vector using the given model.
    /// `modality` selects the tower (`"text"`/`"image"`/`"audio"`); `query` is a
    /// string for text or raw bytes for image/audio.
    #[pyo3(signature = (*, model, query, modality=None))]
    fn encode_query(
        &self,
        model: &str,
        query: QueryArg,
        modality: Option<ModalityArg>,
    ) -> PyResult<Vec<f32>> {
        let modality = modality.map(|m| m.0).unwrap_or(Modality::Text);
        self.runtime
            .block_on(self.session.encode_query(model, query.0, modality))
            .map_err(to_pyerr)
    }

    /// Generate embeddings for `columns` of a registered source with the given
    /// model and modality, persisting one vector per row on the remote engine.
    /// Returns the result table name.
    #[pyo3(signature = (*, source, model, columns, key, modality=None))]
    fn generate_embeddings(
        &self,
        source: &str,
        model: &str,
        columns: Vec<String>,
        key: &str,
        modality: Option<ModalityArg>,
    ) -> PyResult<String> {
        let modality = modality.map(|m| m.0).unwrap_or(Modality::Text);
        let record = self
            .runtime
            .block_on(
                self.session
                    .generate_embeddings(source, model, &columns, key, modality),
            )
            .map_err(to_pyerr)?;
        Ok(record.table_name)
    }

    /// Run a vector search over a source's embedding table. `query` is the query
    /// vector; `filter` is an optional SQL predicate over the hydrated results;
    /// `select` projects columns (empty keeps the keyed + scored shape).
    /// Returns a `pyarrow.Table`.
    #[pyo3(signature = (source, *, query, k, filter=None, select=None))]
    fn search(
        &self,
        py: Python<'_>,
        source: &str,
        query: Vec<f32>,
        k: usize,
        filter: Option<String>,
        select: Option<Vec<String>>,
    ) -> PyResult<Py<PyAny>> {
        let request = SearchRequest {
            source_id: source.to_string(),
            query: SearchQuery::Vector(query),
            k,
            filter,
            select: select.unwrap_or_default(),
        };
        let batches = self
            .runtime
            .block_on(self.session.search(request))
            .map_err(to_pyerr)?;
        batches_to_pyarrow(py, &batches)
    }

    /// Run inference on a registered source using a model. Returns a
    /// `pyarrow.Table`.
    #[pyo3(signature = (*, source, model, columns, task, key))]
    fn infer(
        &self,
        py: Python<'_>,
        source: &str,
        model: &str,
        columns: Vec<String>,
        task: ModelTaskArg,
        key: &str,
    ) -> PyResult<Py<PyAny>> {
        let task: ModelTask = task.0;
        let batches = self
            .runtime
            .block_on(self.session.infer(source, model, task, &columns, key))
            .map_err(to_pyerr)?;
        batches_to_pyarrow(py, &batches)
    }
}

/// Connect to a remote jammi engine over the `jammi.v1` gRPC wire and return a
/// [`PyRemoteDatabase`] (Shape C).
///
/// `endpoint` is an `http://host:port` URL. This is the remote peer of
/// [`crate::connect`]: `connect(...)` opens an embedded engine, `connect_remote`
/// talks to a server. Both return an object whose verbs are the same
/// transport-agnostic surface.
#[pyfunction]
#[pyo3(signature = (*, endpoint))]
pub fn connect_remote(endpoint: &str) -> PyResult<PyRemoteDatabase> {
    PyRemoteDatabase::open(endpoint)
}

/// Argument shim for `encode_query`'s `query=`: a string is text for the text
/// tower; raw bytes are an image/audio clip for the vision/audio tower. Decoded
/// once at the boundary into the engine's [`QueryInput`].
struct QueryArg(QueryInput);

impl<'a, 'py> FromPyObject<'a, 'py> for QueryArg {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(s) = ob.cast::<PyString>() {
            return Ok(QueryArg(QueryInput::Text(s.str()?.to_string())));
        }
        if let Ok(bytes) = ob.extract::<Vec<u8>>() {
            return Ok(QueryArg(QueryInput::Bytes(bytes)));
        }
        Err(PyTypeError::new_err(
            "query must be a str (text tower) or bytes (image/audio tower)",
        ))
    }
}
