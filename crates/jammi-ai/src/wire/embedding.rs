//! `EmbeddingService` request decoders: the bulk-embedding, single-query-encode,
//! and vector-search request shapes the embedded binding and the gRPC handler
//! share.
//!
//! The transport-neutral modality / query-input conversions live on the wire
//! substrate ([`jammi_wire`]); what stays here are the request decoders that
//! return the engine's flat call args (`GenerateEmbeddingsArgs`,
//! `EncodeQueryArgs`) and the [`SearchRequest`] the session search verb takes.
//!
//! The embedded binding builds each request with the same pure-Python assembly
//! the remote client uses, serializes it, and hands the bytes here — so the
//! in-process and remote paths decode through one shared seam. Modality and
//! query input are validated at decode (an unspecified modality and a
//! text/bytes-vs-modality mismatch are rejected with `invalid_argument`),
//! matching the gRPC handler's edge validation.

use prost::Message;
use tonic::Status;

use crate::local_session::{Modality, QueryInput, SearchQuery, SearchRequest};
use jammi_wire::proto::embedding as pb;
use jammi_wire::ProtoQueryInput;

/// The decoded identity + tower a `GenerateEmbeddings` request carries. The
/// engine method (`Session::generate_embeddings`) takes these separately, so the
/// decode returns them as a struct the binding destructures.
pub struct GenerateEmbeddingsArgs {
    pub source_id: String,
    pub model_id: String,
    pub columns: Vec<String>,
    pub key_column: String,
    pub modality: Modality,
}

/// Decode a serialized [`pb::GenerateEmbeddingsRequest`] body into the engine
/// [`GenerateEmbeddingsArgs`]. The embedded binding builds the request with the
/// same pure-Python assembly the remote client uses, serializes it, and hands
/// the bytes here — so the in-process and remote embedding paths decode through
/// one shared seam ([`generate_embeddings_from_proto`]). A body that is not a
/// valid `GenerateEmbeddingsRequest` is a client error (`InvalidArgument`).
pub fn generate_embeddings_from_bytes(body: &[u8]) -> Result<GenerateEmbeddingsArgs, Status> {
    let req = pb::GenerateEmbeddingsRequest::decode(body).map_err(|e| {
        Status::invalid_argument(format!("malformed GenerateEmbeddings request: {e}"))
    })?;
    generate_embeddings_from_proto(req)
}

/// Decode a [`pb::GenerateEmbeddingsRequest`] into the engine
/// [`GenerateEmbeddingsArgs`]. The required identity fields (`source_id` /
/// `model_id` / `key_column`) and a non-empty `columns` list are validated at
/// decode rather than deferred to the engine, and the modality is resolved
/// (an unspecified tower is rejected) — matching the gRPC handler's edge checks.
pub fn generate_embeddings_from_proto(
    req: pb::GenerateEmbeddingsRequest,
) -> Result<GenerateEmbeddingsArgs, Status> {
    if req.source_id.is_empty() {
        return Err(Status::invalid_argument("source_id is required"));
    }
    if req.model_id.is_empty() {
        return Err(Status::invalid_argument("model_id is required"));
    }
    if req.key_column.is_empty() {
        return Err(Status::invalid_argument("key_column is required"));
    }
    if req.columns.is_empty() {
        return Err(Status::invalid_argument("columns is required"));
    }
    Ok(GenerateEmbeddingsArgs {
        source_id: req.source_id,
        model_id: req.model_id,
        columns: req.columns,
        key_column: req.key_column,
        modality: Modality::try_from(req.modality)?,
    })
}

/// The decoded model, tower, and resolved input an `EncodeQuery` request
/// carries. The engine method (`Session::encode_query`) takes the model id, the
/// [`QueryInput`], and the [`Modality`] separately, so the decode returns them
/// as a struct the binding destructures; the query input is matched to the
/// modality at decode (text for the text tower, bytes for image/audio).
pub struct EncodeQueryArgs {
    pub model_id: String,
    pub input: QueryInput,
    pub modality: Modality,
}

/// Decode a serialized [`pb::EncodeQueryRequest`] body into the engine
/// [`EncodeQueryArgs`]. The embedded binding builds the request with the same
/// pure-Python assembly the remote client uses, serializes it, and hands the
/// bytes here — so the in-process and remote encode paths decode through one
/// shared seam ([`encode_query_from_proto`]). A body that is not a valid
/// `EncodeQueryRequest` is a client error (`InvalidArgument`).
pub fn encode_query_from_bytes(body: &[u8]) -> Result<EncodeQueryArgs, Status> {
    let req = pb::EncodeQueryRequest::decode(body)
        .map_err(|e| Status::invalid_argument(format!("malformed EncodeQuery request: {e}")))?;
    encode_query_from_proto(req)
}

/// Decode a [`pb::EncodeQueryRequest`] into the engine [`EncodeQueryArgs`]. The
/// required `model_id` is validated at decode; the `input` oneof is resolved
/// against the request's modality through the shared [`ProtoQueryInput`]
/// conversion (a missing oneof or a text/bytes-vs-modality mismatch is a client
/// error), matching the gRPC handler's edge validation.
pub fn encode_query_from_proto(req: pb::EncodeQueryRequest) -> Result<EncodeQueryArgs, Status> {
    if req.model_id.is_empty() {
        return Err(Status::invalid_argument("model_id is required"));
    }
    let modality = Modality::try_from(req.modality)?;
    let input = ProtoQueryInput {
        input: req.input,
        modality,
    }
    .try_into()?;
    Ok(EncodeQueryArgs {
        model_id: req.model_id,
        input,
        modality,
    })
}

/// Decode a serialized [`pb::SearchRequest`] body into the engine
/// [`SearchRequest`]. The embedded binding builds the request with the same
/// pure-Python assembly the remote client uses, serializes it, and hands the
/// bytes here — so the in-process and remote search paths decode the request
/// through one shared seam ([`search_from_proto`]). The response stays
/// transport-specific (the gRPC handler builds wire hits, the embedded binding
/// returns Arrow), so only the request is collapsed. A body that is not a valid
/// `SearchRequest` is a client error (`InvalidArgument`).
pub fn search_from_bytes(body: &[u8]) -> Result<SearchRequest, Status> {
    let req = pb::SearchRequest::decode(body)
        .map_err(|e| Status::invalid_argument(format!("malformed Search request: {e}")))?;
    search_from_proto(req)
}

/// Decode a [`pb::SearchRequest`] into the engine [`SearchRequest`]. The
/// required `source_id` and the `query` oneof (a precomputed vector or a row key
/// resolved in-engine) are validated at decode; an absent oneof is a client
/// error. The `k` cap widens to the engine's `usize`, and `filter` / `select` /
/// `embedding_table` carry through with their wire presence.
pub fn search_from_proto(req: pb::SearchRequest) -> Result<SearchRequest, Status> {
    use pb::search_request::Query as ProtoQuery;
    if req.source_id.is_empty() {
        return Err(Status::invalid_argument("source_id is required"));
    }
    let query = match req
        .query
        .ok_or_else(|| Status::invalid_argument("query (query_vector or row_key) is required"))?
    {
        ProtoQuery::QueryVector(v) => SearchQuery::Vector(v.values),
        ProtoQuery::RowKey(key) => SearchQuery::RowKey(key),
    };
    Ok(SearchRequest {
        source_id: req.source_id,
        query,
        k: req.k as usize,
        embedding_table: req.embedding_table,
        filter: req.filter,
        select: req.select,
    })
}
