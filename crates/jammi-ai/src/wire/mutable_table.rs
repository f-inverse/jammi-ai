//! Mutable-table proto↔domain conversion (the `CatalogService`
//! create/drop/list-mutable-table verbs).
//!
//! The request [`pb::MutableTableDefinition`] mirrors the engine's
//! [`MutableTableDefinition`] field for field, minus `tenant`: the wire body is
//! tenant-free, and decode stamps the session's resolved tenant onto the engine
//! definition (the catalog row's `tenant_id` sink, matching the trigger DDL
//! path). The schema rides as an Arrow IPC schema message decoded through the
//! shared [`super::decode_ipc_schema`]. All schema / primary-key / order-column
//! validation is delegated to the engine builder, so the wire path enforces the
//! identical invariants the in-process path does.

use jammi_db::store::mutable::{
    MutableIndexDef, MutableTableDefinition, MutableTableDefinitionBuilder, MutableTableId,
};
use jammi_db::TenantId;
use tonic::Status;

use crate::wire::encode_ipc_stream;
use crate::wire::proto::catalog as pb;

/// Build the engine [`MutableTableDefinition`] from the wire message, stamping
/// the resolved session `tenant` onto it (the wire body is tenant-free).
pub fn definition_from_proto(
    def: pb::MutableTableDefinition,
    tenant: Option<TenantId>,
) -> Result<MutableTableDefinition, Status> {
    let id = parse_table_id(&def.id)?;
    let schema = super::decode_ipc_schema(&def.schema)?;

    let mut builder = MutableTableDefinitionBuilder::new(id, schema)
        .primary_key(def.primary_key)
        .tenant(tenant);

    for idx in def.indexes {
        builder = builder.index(MutableIndexDef {
            name: idx.name,
            columns: idx.columns,
            unique: idx.unique,
        });
    }
    if !def.order_column.is_empty() {
        builder = builder.order_column(def.order_column);
    }
    if def.chunk_size != 0 {
        builder = builder.chunk_size(def.chunk_size as usize);
    }
    if !def.user_metadata.is_empty() {
        let value: serde_json::Value = serde_json::from_str(&def.user_metadata)
            .map_err(|e| Status::invalid_argument(format!("user_metadata is not JSON: {e}")))?;
        builder = builder.user_metadata(value);
    }

    builder
        .build()
        .map_err(|e| Status::invalid_argument(e.to_string()))
}

/// Encode the engine [`MutableTableDefinition`] onto its wire message — the
/// inverse of [`definition_from_proto`], for the [`crate::RemoteSession`] send
/// side. The schema rides as a schema-only Arrow IPC stream (the framing
/// [`super::decode_ipc_schema`] reads back). The `tenant` field is intentionally
/// dropped: the wire body stays tenant-free and the server stamps the session's
/// tenant onto the catalog row, so a remote client carries no tenant in the body
/// (it rides the `SESSION_HEADER` instead). `user_metadata` serialises to its
/// JSON-object string (`"{}"` when the default empty object).
pub fn definition_to_proto(
    def: &MutableTableDefinition,
) -> Result<pb::MutableTableDefinition, Status> {
    let schema = encode_ipc_stream(&def.schema, &[])?;
    Ok(pb::MutableTableDefinition {
        id: def.id.to_string(),
        schema,
        primary_key: def.primary_key.clone(),
        indexes: def
            .indexes
            .iter()
            .map(|idx| pb::MutableIndex {
                name: idx.name.clone(),
                columns: idx.columns.clone(),
                unique: idx.unique,
            })
            .collect(),
        order_column: def.order_column.clone().unwrap_or_default(),
        chunk_size: def.chunk_size as u64,
        user_metadata: def.user_metadata.to_string(),
    })
}

/// Reconstruct an engine [`MutableTableDefinition`] from a wire message that
/// carries no tenant — the receive side of a `ListMutableTables` response, for
/// the [`crate::RemoteSession`]. The wire body is tenant-free (a list entry's
/// tenant rides the session scope, not the message), so the engine definition
/// reconstructs with `tenant = None`; a list consumer keys off the id / schema /
/// primary key / indexes, not the catalog row's tenant. Defers all schema /
/// primary-key validation to the shared [`definition_from_proto`].
pub fn definition_list_from_proto(
    def: pb::MutableTableDefinition,
) -> Result<MutableTableDefinition, Status> {
    definition_from_proto(def, None)
}

/// Parse a wire id string into a validated [`MutableTableId`]. Shared by the
/// create path (definition decode) and the drop path (which carries only the
/// id), so the id-shape check lives in one place.
pub fn parse_table_id(id: &str) -> Result<MutableTableId, Status> {
    if id.is_empty() {
        return Err(Status::invalid_argument("mutable_table_id is required"));
    }
    MutableTableId::new(id).map_err(|e| Status::invalid_argument(e.to_string()))
}
