//! `EmbeddingService` gRPC implementation.
//!
//! Each verb is a 1:1 delegation to an [`InferenceSession`] method from the
//! audio-embedding path (JA1):
//!
//! * `AddSource` — registers a data source (peer of `add_source`).
//! * `GenerateAudioEmbeddings` — scans a source's audio column, runs the
//!   decode → log-mel → audio-tower pipeline, persists one vector per row
//!   (peer of `generate_audio_embeddings`).
//! * `EncodeAudioQuery` — encodes a single clip into one vector (peer of
//!   `encode_audio_query`).
//! * `Search` — nearest-neighbor search over a source's embedding table, by a
//!   precomputed vector (peer of `search`) or by an existing row's key (peer
//!   of `search_by_id`, which resolves that row's vector inside the engine).
//!   This is the embedding-consumption verb on the gRPC-web transport edge
//!   runtimes reach; it adds no new consumption model.
//!
//! The service reimplements no embedding logic — decode, feature extraction,
//! the forward pass, and the search plan all live in the engine. This module
//! is purely the wire adapter: proto in, engine call, proto out.
//!
//! Tenant scope is read from the request's [`SessionTenant`] extension (set
//! upstream by [`crate::grpc::session::TenantInterceptor`]) and applied to
//! the engine call via `with_tenant_scoped`, matching how the Flight SQL and
//! Trigger surfaces resolve their tenant.

use std::sync::Arc;

use jammi_ai::session::InferenceSession;
use jammi_db::error::JammiError;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::TenantId;
use tonic::{Request, Response, Status};

use std::collections::HashMap;

use arrow::array::{Array, Float32Array, RecordBatch, StringArray};
use arrow::util::display::{ArrayFormatter, FormatOptions};

use jammi_ai::search::SearchBuilder;

use crate::grpc::proto::embedding::embedding_service_server::EmbeddingService;
use crate::grpc::proto::embedding::{
    search_request::Query as ProtoQuery, AddSourceRequest, EncodeAudioQueryRequest,
    EncodeAudioQueryResponse, FileFormat as ProtoFileFormat, GenerateAudioEmbeddingsRequest,
    ResultTable, SearchHit, SearchRequest, SearchResponse, SourceKind as ProtoSourceKind,
};
use crate::grpc::session::SessionTenant;

/// Server-side handler for the audio-embedding gRPC surface. Holds a shared
/// reference to the engine session it delegates every verb to.
pub struct EmbeddingServer {
    session: Arc<InferenceSession>,
}

impl EmbeddingServer {
    pub fn new(session: Arc<InferenceSession>) -> Self {
        Self { session }
    }
}

#[tonic::async_trait]
impl EmbeddingService for EmbeddingServer {
    async fn add_source(&self, request: Request<AddSourceRequest>) -> Result<Response<()>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        if req.source_id.is_empty() {
            return Err(Status::invalid_argument("source_id is required"));
        }
        let source_type = source_type_from_proto(req.source_kind)?;
        let connection = connection_from_proto(req.connection)?;

        scoped(&self.session, tenant, || {
            self.session
                .add_source(&req.source_id, source_type, connection)
        })
        .await
        .map_err(map_engine_error)?;
        Ok(Response::new(()))
    }

    async fn generate_audio_embeddings(
        &self,
        request: Request<GenerateAudioEmbeddingsRequest>,
    ) -> Result<Response<ResultTable>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        require_nonempty(&req.source_id, "source_id")?;
        require_nonempty(&req.model_id, "model_id")?;
        require_nonempty(&req.audio_column, "audio_column")?;
        require_nonempty(&req.key_column, "key_column")?;

        let record = scoped(&self.session, tenant, || {
            self.session.generate_audio_embeddings(
                &req.source_id,
                &req.model_id,
                &req.audio_column,
                &req.key_column,
            )
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(ResultTable {
            table_name: record.table_name,
            source_id: record.source_id,
            model_id: record.model_id,
            dimensions: record.dimensions.unwrap_or(0),
            row_count: record.row_count as u64,
            status: record.status,
        }))
    }

    async fn encode_audio_query(
        &self,
        request: Request<EncodeAudioQueryRequest>,
    ) -> Result<Response<EncodeAudioQueryResponse>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        require_nonempty(&req.model_id, "model_id")?;
        if req.audio_bytes.is_empty() {
            return Err(Status::invalid_argument("audio_bytes is required"));
        }

        let embedding = scoped(&self.session, tenant, || {
            self.session
                .encode_audio_query(&req.model_id, &req.audio_bytes)
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(EncodeAudioQueryResponse { embedding }))
    }

    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        require_nonempty(&req.source_id, "source_id")?;
        let query = req.query.ok_or_else(|| {
            Status::invalid_argument("query (query_vector or row_key) is required")
        })?;
        let k = req.k as usize;
        let select = req.select;
        let filter = req.filter;

        let batches = scoped(&self.session, tenant, || async {
            let builder = match query {
                ProtoQuery::QueryVector(v) => {
                    self.session.search(&req.source_id, v.values, k).await?
                }
                ProtoQuery::RowKey(key) => {
                    self.session.search_by_id(&req.source_id, &key, k).await?
                }
            };
            let builder = match filter.as_deref() {
                Some(predicate) => builder.filter(predicate)?,
                None => builder,
            };
            let builder = apply_select(builder, &select)?;
            builder.run().await
        })
        .await
        .map_err(map_engine_error)?;

        let hits = batches_to_hits(&batches, &select)?;
        Ok(Response::new(SearchResponse { hits }))
    }
}

/// Project the requested columns onto the search builder. An empty `select`
/// leaves the builder unprojected (all hydrated columns survive, so the key
/// and score are present). A non-empty `select` projects the requested columns
/// **plus** `_row_id` and `similarity` — the handler always needs those to
/// build each hit's key and score, even when the client did not list them.
fn apply_select(builder: SearchBuilder, select: &[String]) -> Result<SearchBuilder, JammiError> {
    if select.is_empty() {
        return Ok(builder);
    }
    let mut columns: Vec<String> = vec!["_row_id".to_string(), "similarity".to_string()];
    for name in select {
        if name != "_row_id" && name != "similarity" {
            columns.push(name.clone());
        }
    }
    builder.select(&columns)
}

/// Map each result row to a [`SearchHit`]: `_row_id` → key, `similarity` →
/// score, and each requested `select` column stringified into `columns`.
///
/// `select` columns are read from the projected batch via the type-general
/// Arrow formatter, so any scalar column the engine returns is carried on the
/// wire without a per-dtype branch here.
fn batches_to_hits(batches: &[RecordBatch], select: &[String]) -> Result<Vec<SearchHit>, Status> {
    let mut hits = Vec::new();
    let format = FormatOptions::default();
    for batch in batches {
        let keys = column_as::<StringArray>(batch, "_row_id")?;
        let scores = column_as::<Float32Array>(batch, "similarity")?;
        let formatters: Vec<(String, ArrayFormatter)> = select
            .iter()
            .map(|name| {
                let array = batch.column_by_name(name).ok_or_else(|| {
                    Status::invalid_argument(format!("select column '{name}' not in results"))
                })?;
                let formatter = ArrayFormatter::try_new(array.as_ref(), &format)
                    .map_err(|e| Status::internal(format!("format column '{name}': {e}")))?;
                Ok((name.clone(), formatter))
            })
            .collect::<Result<_, Status>>()?;

        for row in 0..batch.num_rows() {
            let columns: HashMap<String, String> = formatters
                .iter()
                .map(|(name, fmt)| (name.clone(), fmt.value(row).to_string()))
                .collect();
            hits.push(SearchHit {
                key: keys.value(row).to_string(),
                score: scores.value(row),
                columns,
            });
        }
    }
    Ok(hits)
}

/// Downcast a named column to a concrete Arrow array, mapping a missing or
/// wrong-typed column to an internal [`Status`] (the search plan owns these
/// columns, so a mismatch is a server-side invariant break, not a bad input).
fn column_as<'a, A: Array + 'static>(batch: &'a RecordBatch, name: &str) -> Result<&'a A, Status> {
    batch
        .column_by_name(name)
        .ok_or_else(|| Status::internal(format!("search result missing '{name}' column")))?
        .as_any()
        .downcast_ref::<A>()
        .ok_or_else(|| {
            Status::internal(format!("search result '{name}' column has unexpected type"))
        })
}

/// Read the bound tenant the [`crate::grpc::session::TenantInterceptor`]
/// attached to the request.
fn session_tenant<T>(request: &Request<T>) -> Option<TenantId> {
    request
        .extensions()
        .get::<SessionTenant>()
        .and_then(|s| s.0)
}

/// Run an engine call under the request's tenant scope.
///
/// A bound tenant installs the engine's task-local tenant override for the
/// duration of the call via `with_tenant_scoped` — the concurrency-safe form
/// the gRPC handlers must use, since they share one `Arc<InferenceSession>`
/// and the sticky `bind_tenant` would race across concurrent requests. The
/// `TenantScope` handle the closure receives is the marker that the scope is
/// active on this task; `f` calls the embedding verbs on the outer session
/// reference and observes the same task-local. An unscoped session runs the
/// call directly.
async fn scoped<F, Fut, T>(
    session: &Arc<InferenceSession>,
    tenant: Option<TenantId>,
    f: F,
) -> Result<T, JammiError>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<T, JammiError>>,
{
    match tenant {
        Some(t) => session.with_tenant_scoped(t, |_scope| f()).await,
        None => f().await,
    }
}

fn require_nonempty(value: &str, field: &str) -> Result<(), Status> {
    if value.is_empty() {
        Err(Status::invalid_argument(format!("{field} is required")))
    } else {
        Ok(())
    }
}

/// Map the proto [`SourceKind`] enum onto the engine's [`SourceType`].
/// An unspecified kind is rejected — a registration with no backend is a
/// client error, not a silent default.
fn source_type_from_proto(kind: i32) -> Result<SourceType, Status> {
    match ProtoSourceKind::try_from(kind) {
        Ok(ProtoSourceKind::File) => Ok(SourceType::File),
        Ok(ProtoSourceKind::Postgres) => Ok(SourceType::Postgres),
        Ok(ProtoSourceKind::Mysql) => Ok(SourceType::Mysql),
        Ok(ProtoSourceKind::Unspecified) | Err(_) => {
            Err(Status::invalid_argument("source_kind must be specified"))
        }
    }
}

/// Map the proto [`FileFormat`] enum onto the engine's [`FileFormat`].
fn file_format_from_proto(format: i32) -> Result<Option<FileFormat>, Status> {
    match ProtoFileFormat::try_from(format) {
        Ok(ProtoFileFormat::Parquet) => Ok(Some(FileFormat::Parquet)),
        Ok(ProtoFileFormat::Csv) => Ok(Some(FileFormat::Csv)),
        Ok(ProtoFileFormat::Json) => Ok(Some(FileFormat::Json)),
        Ok(ProtoFileFormat::Avro) => Ok(Some(FileFormat::Avro)),
        Ok(ProtoFileFormat::Unspecified) | Err(_) => Ok(None),
    }
}

/// Build the engine's [`SourceConnection`] from the proto message. Only the
/// URL + format are carried on the wire; cloud credentials are server-side.
fn connection_from_proto(
    conn: Option<crate::grpc::proto::embedding::SourceConnection>,
) -> Result<SourceConnection, Status> {
    let conn = conn.ok_or_else(|| Status::invalid_argument("connection is required"))?;
    let url = if conn.url.is_empty() {
        None
    } else {
        Some(conn.url)
    };
    Ok(SourceConnection {
        url,
        format: file_format_from_proto(conn.format)?,
        ..Default::default()
    })
}

/// Map an engine [`JammiError`] to a gRPC [`Status`], preserving the kind of
/// failure so a client can distinguish a bad request from an internal fault.
fn map_engine_error(err: JammiError) -> Status {
    match err {
        JammiError::Source { source_id, message } => {
            Status::invalid_argument(format!("source {source_id}: {message}"))
        }
        JammiError::Model { model_id, message } => {
            Status::invalid_argument(format!("model {model_id}: {message}"))
        }
        JammiError::Tenant(detail) => Status::invalid_argument(format!("tenant: {detail}")),
        JammiError::Config(detail) => Status::invalid_argument(format!("config: {detail}")),
        JammiError::Schema { .. } => Status::invalid_argument(err.to_string()),
        JammiError::Inference(detail) => Status::internal(format!("inference: {detail}")),
        other => Status::internal(other.to_string()),
    }
}
