//! `CatalogService` substrate-verb request decoders: the channel-catalog
//! (register / add-columns), mutable-table create, and topic-register request
//! shapes the embedded binding and the gRPC handler share.
//!
//! The transport-neutral channel / mutable-table / topic conversions live on the
//! wire substrate ([`jammi_wire`]) — the column, definition, and schema decodes
//! are reused here, not reimplemented. What stays in the engine crate is the
//! assembly of the engine call args that touch types behind the `local` feature
//! (`ChannelSpec`, `MutableTableDefinition`, `TopicDefinition`), reachable only in
//! an engine build (the server and the embedded SDK).
//!
//! The embedded binding builds each request with the same pure-Python assembly
//! the remote client uses, serializes it, and hands the bytes here — so the
//! in-process and remote control-plane paths decode through one shared seam.

use prost::Message;
use tonic::Status;

use jammi_db::catalog::channel_repo::{ChannelColumn, ChannelSpec};
use jammi_db::store::mutable::MutableTableDefinition;
use jammi_db::trigger::ids::TopicId;
use jammi_db::trigger::TopicDefinition;
use jammi_db::{ChannelId, TenantId};

use jammi_wire::proto::catalog as pb;
use jammi_wire::{columns_from_proto, decode_ipc_schema, definition_from_proto, parse_channel_id};

// ─── RegisterChannel ─────────────────────────────────────────────────────────

/// Decode a serialized [`pb::RegisterChannelRequest`] body into the engine
/// [`ChannelSpec`]. The embedded binding builds the request with the same
/// pure-Python assembly the remote client uses, serializes it, and hands the
/// bytes here — so the in-process and remote register-channel paths decode
/// through one shared seam ([`register_channel_from_proto`]). A body that is not
/// a valid `RegisterChannelRequest` is a client error (`InvalidArgument`).
pub fn register_channel_from_bytes(body: &[u8]) -> Result<ChannelSpec, Status> {
    let req = pb::RegisterChannelRequest::decode(body)
        .map_err(|e| Status::invalid_argument(format!("malformed RegisterChannel request: {e}")))?;
    register_channel_from_proto(req)
}

/// Decode a [`pb::RegisterChannelRequest`] into the engine [`ChannelSpec`]. The
/// id parses through the shared [`parse_channel_id`] (an empty id is a client
/// error) and the columns through [`columns_from_proto`] (a missing or
/// unspecified column type is rejected at decode), so the wire path enforces the
/// identical invariants the in-process path does.
pub fn register_channel_from_proto(req: pb::RegisterChannelRequest) -> Result<ChannelSpec, Status> {
    Ok(ChannelSpec {
        id: parse_channel_id(&req.channel_id)?,
        priority: req.priority,
        columns: columns_from_proto(req.columns)?,
    })
}

// ─── AddChannelColumns ───────────────────────────────────────────────────────

/// The decoded channel id + appended columns an `AddChannelColumns` request
/// carries. The engine method (`Session::add_channel_columns`) takes the id and
/// the column slice separately, so the decode returns them as a struct the
/// binding destructures.
pub struct AddChannelColumnsArgs {
    pub id: ChannelId,
    pub columns: Vec<ChannelColumn>,
}

/// Decode a serialized [`pb::AddChannelColumnsRequest`] body into the engine
/// [`AddChannelColumnsArgs`]. The embedded binding builds the request with the
/// same pure-Python assembly the remote client uses, serializes it, and hands
/// the bytes here — so the in-process and remote add-columns paths decode through
/// one shared seam ([`add_channel_columns_from_proto`]). A body that is not a
/// valid `AddChannelColumnsRequest` is a client error (`InvalidArgument`).
pub fn add_channel_columns_from_bytes(body: &[u8]) -> Result<AddChannelColumnsArgs, Status> {
    let req = pb::AddChannelColumnsRequest::decode(body).map_err(|e| {
        Status::invalid_argument(format!("malformed AddChannelColumns request: {e}"))
    })?;
    add_channel_columns_from_proto(req)
}

/// Decode a [`pb::AddChannelColumnsRequest`] into the engine
/// [`AddChannelColumnsArgs`]. The id parses through the shared [`parse_channel_id`]
/// and the columns through [`columns_from_proto`], so the append-only invariant
/// the engine enforces sees the identical column vocabulary on both transports.
pub fn add_channel_columns_from_proto(
    req: pb::AddChannelColumnsRequest,
) -> Result<AddChannelColumnsArgs, Status> {
    Ok(AddChannelColumnsArgs {
        id: parse_channel_id(&req.channel_id)?,
        columns: columns_from_proto(req.columns)?,
    })
}

// ─── CreateMutableTable ──────────────────────────────────────────────────────

/// Decode a serialized [`pb::CreateMutableTableRequest`] body into the engine
/// [`MutableTableDefinition`], stamping the resolved session `tenant` onto it
/// (the wire body is tenant-free). The embedded binding builds the request with
/// the same pure-Python assembly the remote client uses, serializes it, and hands
/// the bytes here — so the in-process and remote create paths decode through one
/// shared seam ([`create_mutable_table_from_proto`]). A body that is not a valid
/// `CreateMutableTableRequest` is a client error (`InvalidArgument`).
pub fn create_mutable_table_from_bytes(
    body: &[u8],
    tenant: Option<TenantId>,
) -> Result<MutableTableDefinition, Status> {
    let req = pb::CreateMutableTableRequest::decode(body).map_err(|e| {
        Status::invalid_argument(format!("malformed CreateMutableTable request: {e}"))
    })?;
    create_mutable_table_from_proto(req, tenant)
}

/// Decode a [`pb::CreateMutableTableRequest`] into the engine
/// [`MutableTableDefinition`]. An absent `definition` is a client error; the
/// definition itself decodes through the shared [`definition_from_proto`] (id /
/// schema / primary-key / index validation delegated to the engine builder), so
/// the wire path enforces the identical invariants the in-process path does.
pub fn create_mutable_table_from_proto(
    req: pb::CreateMutableTableRequest,
    tenant: Option<TenantId>,
) -> Result<MutableTableDefinition, Status> {
    let def = req
        .definition
        .ok_or_else(|| Status::invalid_argument("definition is required"))?;
    definition_from_proto(def, tenant)
}

// ─── RegisterTopic ───────────────────────────────────────────────────────────

/// Decode a serialized [`pb::RegisterTopicRequest`] body into the engine
/// [`TopicDefinition`], stamping the resolved session `tenant` onto it (the wire
/// body is tenant-free). The embedded binding builds the request with the same
/// pure-Python assembly the remote client uses, serializes it, and hands the
/// bytes here — so the in-process and remote register-topic paths decode through
/// one shared seam ([`register_topic_from_proto`]). A body that is not a valid
/// `RegisterTopicRequest` is a client error (`InvalidArgument`).
pub fn register_topic_from_bytes(
    body: &[u8],
    tenant: Option<TenantId>,
) -> Result<TopicDefinition, Status> {
    let req = pb::RegisterTopicRequest::decode(body)
        .map_err(|e| Status::invalid_argument(format!("malformed RegisterTopic request: {e}")))?;
    register_topic_from_proto(req, tenant)
}

/// Decode a [`pb::RegisterTopicRequest`] into the engine [`TopicDefinition`]. The
/// `name` is required; the schema rides as an Arrow IPC schema message decoded
/// through the shared [`decode_ipc_schema`]. A caller-supplied `topic_id` is
/// honoured so the topic's identity stays consistent across transports; an empty
/// id mints a fresh UUIDv7 here — the engine is the single source of a minted id,
/// returned to the caller after registration.
pub fn register_topic_from_proto(
    req: pb::RegisterTopicRequest,
    tenant: Option<TenantId>,
) -> Result<TopicDefinition, Status> {
    if req.name.is_empty() {
        return Err(Status::invalid_argument("name is required"));
    }
    let schema = decode_ipc_schema(&req.schema)?;
    let id = if req.topic_id.is_empty() {
        TopicId::new()
    } else {
        req.topic_id
            .parse::<TopicId>()
            .map_err(|e| Status::invalid_argument(format!("invalid topic_id: {e}")))?
    };
    Ok(TopicDefinition {
        id,
        name: req.name,
        schema,
        tenant,
        broker_metadata: req.broker_metadata.into_iter().collect(),
    })
}
