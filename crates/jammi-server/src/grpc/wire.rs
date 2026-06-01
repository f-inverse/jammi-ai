//! Shared wire-adapter helpers for the engine-backed gRPC services
//! (`EmbeddingService`, `InferenceService`, `EvalService`).
//!
//! These services are all thin adapters over the transport-agnostic
//! [`jammi_ai::Session`]/[`jammi_ai::LocalSession`] abstraction wrapping one
//! shared `Arc<InferenceSession>`. They share four concerns, factored here so
//! no service reimplements them:
//!
//! * [`session_tenant`] — read the tenant the [`crate::grpc::session::
//!   TenantInterceptor`] attached to the request.
//! * [`scoped`] — run a session call under that tenant via the
//!   concurrency-safe `with_tenant_scoped` task-local.
//! * [`require_nonempty`] — reject a missing required string field.
//! * [`map_engine_error`] — map an engine [`JammiError`] to a gRPC [`Status`]
//!   preserving the failure kind.
//!
//! Arrow record batches cross the wire through [`encode_ipc_stream`] /
//! [`decode_ipc_stream`], which carry a self-describing IPC stream (schema +
//! batches) in one `ArrowBatch.data_body` — the same Flight-IPC pairing
//! `TriggerService` uses.

use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::SchemaRef;
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::error::JammiError;
use jammi_db::TenantId;
use tonic::{Request, Status};

use crate::grpc::proto::inference::ModelTask as ProtoModelTask;
use crate::grpc::session::SessionTenant;

/// Read the bound tenant the [`crate::grpc::session::TenantInterceptor`]
/// attached to the request.
pub fn session_tenant<T>(request: &Request<T>) -> Option<TenantId> {
    request
        .extensions()
        .get::<SessionTenant>()
        .and_then(|s| s.0)
}

/// Run a session call under the request's tenant scope.
///
/// A bound tenant installs the engine's task-local tenant override for the
/// duration of the call via `with_tenant_scoped` — the concurrency-safe form
/// the gRPC handlers must use, since they share one `Arc<InferenceSession>`
/// and the sticky `bind_tenant` would race across concurrent requests. The
/// `TenantScope` handle the closure receives is the marker that the scope is
/// active on this task; `f` calls the verb on the [`jammi_ai::LocalSession`]
/// (which delegates to the same engine) and observes the same task-local. An
/// unscoped session runs the call directly.
pub async fn scoped<F, Fut, T>(
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

/// Map the wire [`ProtoModelTask`] onto the engine's [`ModelTask`]. An
/// unspecified task is rejected — a request that names no task is a client
/// error, not a silent default. Shared by [`crate::grpc::inference`] and
/// [`crate::grpc::fine_tune`], which both carry `jammi.v1.inference.ModelTask`.
pub fn model_task_from_proto(task: i32) -> Result<ModelTask, Status> {
    match ProtoModelTask::try_from(task) {
        Ok(ProtoModelTask::TextEmbedding) => Ok(ModelTask::TextEmbedding),
        Ok(ProtoModelTask::ImageEmbedding) => Ok(ModelTask::ImageEmbedding),
        Ok(ProtoModelTask::AudioEmbedding) => Ok(ModelTask::AudioEmbedding),
        Ok(ProtoModelTask::Classification) => Ok(ModelTask::Classification),
        Ok(ProtoModelTask::Ner) => Ok(ModelTask::Ner),
        Ok(ProtoModelTask::Unspecified) | Err(_) => {
            Err(Status::invalid_argument("task must be specified"))
        }
    }
}

/// Reject a missing required string field with `invalid_argument`.
pub fn require_nonempty(value: &str, field: &str) -> Result<(), Status> {
    if value.is_empty() {
        Err(Status::invalid_argument(format!("{field} is required")))
    } else {
        Ok(())
    }
}

/// Map an engine [`JammiError`] to a gRPC [`Status`], preserving the kind of
/// failure so a client can distinguish a bad request from an internal fault.
pub fn map_engine_error(err: JammiError) -> Status {
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
        JammiError::Eval(detail) => Status::invalid_argument(format!("eval: {detail}")),
        JammiError::Inference(detail) => Status::internal(format!("inference: {detail}")),
        other => Status::internal(other.to_string()),
    }
}

/// Encode a sequence of record batches into one self-describing Arrow IPC
/// stream (schema message followed by each batch). The result is the
/// `data_body` of an `ArrowBatch`; `data_header` stays empty because the
/// stream already carries its schema. [`decode_ipc_stream`] is the inverse.
///
/// An empty batch slice still encodes a valid stream carrying just `schema`,
/// so a zero-row inference result round-trips to an empty `Vec<RecordBatch>`
/// rather than an error.
pub fn encode_ipc_stream(schema: &SchemaRef, batches: &[RecordBatch]) -> Result<Vec<u8>, Status> {
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, schema.as_ref())
            .map_err(|e| Status::internal(format!("batch encode: {e}")))?;
        for batch in batches {
            writer
                .write(batch)
                .map_err(|e| Status::internal(format!("batch encode: {e}")))?;
        }
        writer
            .finish()
            .map_err(|e| Status::internal(format!("batch encode: {e}")))?;
    }
    Ok(buf)
}

/// Decode the `data_header` + `data_body` of an `ArrowBatch` into the record
/// batches it carries. Concatenates the two byte runs (the `TriggerService`
/// pairing puts the schema header inline in `data_body` and leaves
/// `data_header` empty, but a producer may split them) and reads every batch
/// from the resulting IPC stream. Inverse of [`encode_ipc_stream`].
pub fn decode_ipc_stream(data_header: &[u8], data_body: &[u8]) -> Result<Vec<RecordBatch>, Status> {
    // An all-empty payload carries no schema and no batches — the encoder's
    // representation of "zero rows, schema unknown". Decode it to an empty
    // batch list rather than feeding `StreamReader` a truncated stream.
    if data_header.is_empty() && data_body.is_empty() {
        return Ok(Vec::new());
    }
    let mut bytes = Vec::with_capacity(data_header.len() + data_body.len());
    bytes.extend_from_slice(data_header);
    bytes.extend_from_slice(data_body);
    let cursor = std::io::Cursor::new(bytes);
    let reader = StreamReader::try_new(cursor, None)
        .map_err(|e| Status::invalid_argument(format!("batch decode: {e}")))?;
    reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| Status::invalid_argument(format!("batch decode: {e}")))
}
