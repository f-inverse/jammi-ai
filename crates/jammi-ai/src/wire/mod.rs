//! The gRPC wire surface: the generated `jammi.v1` tonic stubs (the
//! [`proto`](crate::wire::proto) submodule) and the single home for the
//! proto↔domain conversions every gRPC consumer shares.
//!
//! Both sides of the wire live here so the conversions compile once and are
//! used in both directions: the prost types are local to this crate (generated
//! by `build.rs`) and the domain types are local too (`jammi-ai` / its
//! `jammi-db` substrate), so `From`/`TryFrom` impls satisfy the orphan rule
//! without a newtype wrapper. `jammi-server` consumes the server stubs + these
//! conversions; a future `RemoteSession` consumes the client stubs + the same
//! conversions — neither reimplements a mapping.
//!
//! Conversions are expressed as `impl From<Domain> for proto::X` (encode, never
//! fallible) and `impl TryFrom<proto::X> for Domain` (decode, fallible with a
//! [`tonic::Status`]). A decode that rejects an unspecified-enum / missing-field
//! does so with the same `invalid_argument` message the in-process and remote
//! paths both surface — the wire body stays tenant-free; tenant is stamped by
//! the receiving side, never carried in a conversion.
//!
//! Arrow record batches cross the wire through the IPC-stream helpers
//! (`encode_ipc_stream` / `decode_ipc_stream`), which carry a self-describing
//! IPC stream (schema + batches) in one `ArrowBatch.data_body` — the Flight-IPC
//! pairing the trigger, inference, and mutable-table surfaces share.

use arrow::record_batch::RecordBatch;
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::SchemaRef;
use tonic::Status;

pub mod proto;

mod audit;
mod channel;
mod embedding;
mod error;
mod eval;
mod inference;
mod mutable_table;
// The pipeline conversions touch the engine-side pipeline vocabulary
// (`BuildNeighborGraph` / `PropagateRequest` / `ContextRequest` /
// `ContextRepresentation`), which lives behind the `local` feature — the whole
// module is reachable only in a `local + wire` build (the server + embedded SDK).
#[cfg(feature = "local")]
mod pipeline;
mod training;
mod trigger;

pub use audit::{parse_query_id, record_from_wire};
pub use channel::{columns_from_proto, columns_to_proto, parse_channel_id};
pub use embedding::{
    result_table_from_proto, source_descriptor_from_proto, source_type_from_proto,
    source_type_to_proto, ProtoQueryInput,
};
pub use error::{
    attach_audit_detail, attach_error_detail, attach_trigger_detail, audit_error_from_status,
    error_from_status, trigger_error_from_status,
};
pub use eval::{
    calibration_shape_from_proto, calibration_shape_to_proto, cohorts_from_proto, cohorts_to_proto,
    eval_task_to_proto, EvalTaskFromWire,
};
pub use inference::infer_result_to_proto;
// The pipeline conversions reconstruct / project the engine pipeline request +
// response structs (`BuildNeighborGraph` / `PropagateRequest` / `ContextRequest`
// / `ContextRepresentation`), which live behind the `local` feature — reachable
// only in a `local + wire` build (the server and the embedded SDK).
#[cfg(feature = "local")]
pub use pipeline::{
    assemble_context_from_proto, assemble_context_request_from_proto, assemble_context_to_proto,
    build_neighbor_graph_from_proto, context_source_tag, propagate_request_from_proto,
    BuildNeighborGraphArgs,
};
// The spec / predict / edge-gather conversions touch the engine-side
// `TrainingSpec` / `ContextPredictorTrainConfig` / `EdgeGather` / `PredictedDistribution`
// types, which live behind the `local` feature (the engine vocabulary, as opposed
// to the always-on `FineTuneConfig` config vocabulary). They are therefore only
// reachable in a `local + wire` build (the server and the embedded SDK); a thin
// wire-only client carries no engine spec to encode.
#[cfg(feature = "local")]
pub use inference::{
    edge_gather_from_proto, edge_gather_to_proto, predicted_distribution_to_proto,
};
pub use mutable_table::{definition_from_proto, definition_to_proto, parse_table_id};
pub use training::{config_to_proto, method_from_proto, method_to_proto};
#[cfg(feature = "local")]
pub use training::{training_spec_from_proto, training_spec_to_proto};
pub use trigger::{
    decode_publish_batch, decode_subscribed_batch, encode_delivered_batch, encode_publish_batch,
    from_proto_timestamp, to_proto_timestamp, topic_from_proto, topic_to_proto,
};

/// Header name carrying the opaque session identifier. Clients mint a
/// per-connection id and include it on every request that needs tenant-scoped
/// semantics; the server's session store keys the tenant binding by it, and a
/// future remote client sets it on each outbound request. One definition shared
/// by both sides so the header name cannot drift.
pub const SESSION_HEADER: &str = "jammi-session-id";

/// Map the wire [`proto::inference::ModelTask`] onto the substrate's
/// [`jammi_db::ModelTask`]. An unspecified task is rejected — a request that
/// names no task is a client error, not a silent default. Shared by the
/// inference and fine-tune surfaces, which both carry `jammi.v1.inference
/// .ModelTask`. References the type at its substrate owner (`jammi_db`), not the
/// engine's re-export, so this conversion stays in the thin `wire`-only build.
pub fn model_task_from_proto(task: i32) -> Result<jammi_db::ModelTask, Status> {
    use jammi_db::ModelTask;
    use proto::inference::ModelTask as ProtoModelTask;
    match ProtoModelTask::try_from(task) {
        Ok(ProtoModelTask::TextEmbedding) => Ok(ModelTask::TextEmbedding),
        Ok(ProtoModelTask::ImageEmbedding) => Ok(ModelTask::ImageEmbedding),
        Ok(ProtoModelTask::AudioEmbedding) => Ok(ModelTask::AudioEmbedding),
        Ok(ProtoModelTask::Classification) => Ok(ModelTask::Classification),
        Ok(ProtoModelTask::Ner) => Ok(ModelTask::Ner),
        Ok(ProtoModelTask::Regression) => Ok(ModelTask::Regression),
        Ok(ProtoModelTask::Unspecified) | Err(_) => {
            Err(Status::invalid_argument("task must be specified"))
        }
    }
}

/// Encode the substrate's [`jammi_db::ModelTask`] onto the wire enum — the
/// inverse of [`model_task_from_proto`], for the [`crate::RemoteSession`] send
/// side. Total: every task maps to a concrete wire variant (the type has no
/// unspecified state). Shared by the inference and fine-tune send surfaces, which
/// both carry `jammi.v1.inference.ModelTask`.
pub fn model_task_to_proto(task: jammi_db::ModelTask) -> proto::inference::ModelTask {
    use jammi_db::ModelTask;
    use proto::inference::ModelTask as ProtoModelTask;
    match task {
        ModelTask::TextEmbedding => ProtoModelTask::TextEmbedding,
        ModelTask::ImageEmbedding => ProtoModelTask::ImageEmbedding,
        ModelTask::AudioEmbedding => ProtoModelTask::AudioEmbedding,
        ModelTask::Classification => ProtoModelTask::Classification,
        ModelTask::Ner => ProtoModelTask::Ner,
        ModelTask::Regression => ProtoModelTask::Regression,
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

/// Decode an Arrow IPC stream's schema message into a [`SchemaRef`]. The bytes
/// are a self-describing IPC stream (a `schema` message, optionally followed by
/// batches) — the same framing [`encode_ipc_stream`] produces; a schema-only
/// payload is `encode_ipc_stream(schema, &[])`. Used by the verbs that carry a
/// table/topic schema declaration on the wire rather than a batch of rows.
pub fn decode_ipc_schema(bytes: &[u8]) -> Result<SchemaRef, Status> {
    if bytes.is_empty() {
        return Err(Status::invalid_argument("schema is required"));
    }
    let cursor = std::io::Cursor::new(bytes.to_vec());
    let reader = StreamReader::try_new(cursor, None)
        .map_err(|e| Status::invalid_argument(format!("schema decode: {e}")))?;
    Ok(reader.schema())
}

/// Decode the `data_header` + `data_body` of an `ArrowBatch` into the record
/// batches it carries. Concatenates the two byte runs (the trigger pairing puts
/// the schema header inline in `data_body` and leaves `data_header` empty, but a
/// producer may split them) and reads every batch from the resulting IPC stream.
/// Inverse of [`encode_ipc_stream`].
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
