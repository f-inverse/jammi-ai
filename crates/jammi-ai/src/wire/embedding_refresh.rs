//! The incremental-embedding verbs' request/report seams the embedded binding
//! and the gRPC handler share (the `wire/pipeline.rs` shape): the embedded
//! binding hands serialized request bytes here and takes serialized report
//! bytes back, so both transports decode through one seam and return one
//! report shape.

use prost::Message;
use tonic::Status;

pub use jammi_wire::embedding_refresh::{
    compact_embeddings_from_proto, expire_versions_from_proto, refresh_embeddings_from_proto,
    ExpireVersionsArgs, RefreshEmbeddingsArgs,
};
use jammi_wire::embedding_refresh::{expiry_report_to_proto, refresh_report_to_proto};
use jammi_wire::proto::embedding as pb;

use crate::pipeline::embedding_refresh::{ExpiryReport, RefreshReport};

/// Decode a serialized `RefreshEmbeddingsRequest` body.
pub fn refresh_embeddings_from_bytes(body: &[u8]) -> Result<RefreshEmbeddingsArgs, Status> {
    let req = pb::RefreshEmbeddingsRequest::decode(body).map_err(|e| {
        Status::invalid_argument(format!("malformed RefreshEmbeddings request: {e}"))
    })?;
    refresh_embeddings_from_proto(req)
}

/// Decode a serialized `CompactEmbeddingsRequest` body into the table name.
pub fn compact_embeddings_from_bytes(body: &[u8]) -> Result<String, Status> {
    let req = pb::CompactEmbeddingsRequest::decode(body).map_err(|e| {
        Status::invalid_argument(format!("malformed CompactEmbeddings request: {e}"))
    })?;
    compact_embeddings_from_proto(req)
}

/// Decode a serialized `ExpireVersionsRequest` body.
pub fn expire_versions_from_bytes(body: &[u8]) -> Result<ExpireVersionsArgs, Status> {
    let req = pb::ExpireVersionsRequest::decode(body)
        .map_err(|e| Status::invalid_argument(format!("malformed ExpireVersions request: {e}")))?;
    expire_versions_from_proto(req)
}

/// Serialize a refresh/compaction report for the embedded binding.
pub fn refresh_report_to_bytes(report: &RefreshReport) -> Vec<u8> {
    refresh_report_to_proto(report).encode_to_vec()
}

/// Serialize an expiry report for the embedded binding.
pub fn expiry_report_to_bytes(report: &ExpiryReport) -> Vec<u8> {
    expiry_report_to_proto(report).encode_to_vec()
}
