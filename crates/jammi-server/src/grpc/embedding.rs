//! `EmbeddingService` gRPC implementation.
//!
//! Three verbs land on the wire surface, each a 1:1 delegation to an
//! [`InferenceSession`] method from the audio-embedding path (JA1):
//!
//! * `AddSource` — registers a data source (peer of `add_source`).
//! * `GenerateAudioEmbeddings` — scans a source's audio column, runs the
//!   decode → log-mel → audio-tower pipeline, persists one vector per row
//!   (peer of `generate_audio_embeddings`).
//! * `EncodeAudioQuery` — encodes a single clip into one vector (peer of
//!   `encode_audio_query`).
//!
//! The service reimplements no embedding logic — decode, feature extraction,
//! and the forward pass all live in the engine. This module is purely the
//! wire adapter: proto in, engine call, proto out.
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

use crate::grpc::proto::embedding::embedding_service_server::EmbeddingService;
use crate::grpc::proto::embedding::{
    AddSourceRequest, EncodeAudioQueryRequest, EncodeAudioQueryResponse,
    FileFormat as ProtoFileFormat, GenerateAudioEmbeddingsRequest, ResultTable,
    SourceKind as ProtoSourceKind,
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
