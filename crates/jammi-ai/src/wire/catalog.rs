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
/// through the shared [`decode_ipc_schema`]. The topic id is engine-assigned
/// identity, not caller input: it is always minted server-side here and any
/// `req.topic_id` on the wire is ignored. The `topics.topic_id` PK has no
/// `ON CONFLICT`, so honouring a caller id would let one tenant replay a known
/// UUID to PK-collide another tenant's registration; minting it here closes that
/// for every transport that shares this seam. The minted id is returned to the
/// caller after registration.
pub fn register_topic_from_proto(
    req: pb::RegisterTopicRequest,
    tenant: Option<TenantId>,
) -> Result<TopicDefinition, Status> {
    if req.name.is_empty() {
        return Err(Status::invalid_argument("name is required"));
    }
    let schema = decode_ipc_schema(&req.schema)?;
    Ok(TopicDefinition {
        id: TopicId::new(),
        name: req.name,
        schema,
        tenant,
        broker_metadata: req.broker_metadata.into_iter().collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_schema::{DataType, Field, Schema};
    use jammi_wire::encode_ipc_stream;
    use prost::Message;

    fn topic_schema_bytes() -> Vec<u8> {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        encode_ipc_stream(&schema, &[]).unwrap()
    }

    /// The embedded transport feeds a serialized `RegisterTopicRequest` straight
    /// into [`register_topic_from_bytes`] — the SAME seam the gRPC handler
    /// drives. A caller-supplied `topic_id` must be IGNORED and a fresh
    /// server-side id minted, so neither transport lets a caller pin (and replay)
    /// a topic's identity to PK-collide another tenant's registration.
    #[test]
    fn register_topic_from_bytes_mints_id_and_ignores_caller_id() {
        let attacker_id = TopicId::new();
        let req = pb::RegisterTopicRequest {
            name: "events".into(),
            schema: topic_schema_bytes(),
            broker_metadata: Default::default(),
            topic_id: attacker_id.to_string(),
        };

        let topic =
            register_topic_from_bytes(&req.encode_to_vec(), None).expect("decode register-topic");

        assert_ne!(
            topic.id, attacker_id,
            "the seam must mint a fresh id, never honour the caller-supplied one"
        );
        assert_eq!(topic.name, "events");
        assert_eq!(topic.tenant, None);
    }

    /// Two registrations replaying the SAME caller id each get a DISTINCT
    /// server-minted id — the property that makes the global `topics.topic_id`
    /// PK collision-proof across tenants.
    #[test]
    fn register_topic_replaying_one_caller_id_yields_distinct_minted_ids() {
        let replayed = TopicId::new().to_string();
        let make = || pb::RegisterTopicRequest {
            name: "events".into(),
            schema: topic_schema_bytes(),
            broker_metadata: Default::default(),
            topic_id: replayed.clone(),
        };

        let first = register_topic_from_bytes(&make().encode_to_vec(), None).unwrap();
        let second = register_topic_from_bytes(&make().encode_to_vec(), None).unwrap();

        assert_ne!(
            first.id, second.id,
            "two registrations replaying one caller id must get distinct minted ids"
        );
    }
}
