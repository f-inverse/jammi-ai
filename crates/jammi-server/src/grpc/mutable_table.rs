//! `MutableTableService` gRPC implementation.
//!
//! Two verbs land on the wire: `CreateMutableTable` and `DropMutableTable`.
//! Each is a thin adapter over the transport-agnostic [`Session`]/
//! [`LocalSession`] abstraction (never raw [`InferenceSession`] calls): proto
//! in, one `Session::*_mutable_table` call, proto out. The service reimplements
//! no DDL, storage, or catalog logic.
//!
//! The request [`pb::MutableTableDefinition`] mirrors the Rust
//! [`jammi_db::store::mutable::MutableTableDefinition`] field for field, minus
//! `tenant`: the wire body stays tenant-free, and the handler stamps the
//! session's resolved tenant onto the engine definition before registering it
//! (the catalog row's `tenant_id` sink, matching the trigger DDL path). The
//! schema rides as an Arrow IPC schema message decoded through the shared
//! [`decode_ipc_schema`].
//!
//! Tenant scope is read from the request's [`crate::grpc::session::
//! SessionTenant`] extension (set upstream by the shared `TenantInterceptor`)
//! and applied via `with_tenant_scoped`, matching every other engine-backed
//! gRPC surface.

use std::sync::Arc;

use jammi_ai::session::InferenceSession;
use jammi_ai::{LocalSession, Session};
use jammi_db::store::mutable::{
    MutableIndexDef, MutableTableDefinition, MutableTableDefinitionBuilder, MutableTableId,
};
use jammi_db::TenantId;
use tonic::{Request, Response, Status};

use crate::grpc::proto::mutable_table as pb;
use crate::grpc::proto::mutable_table::mutable_table_service_server::MutableTableService;
use crate::grpc::wire::{
    decode_ipc_schema, map_engine_error, require_nonempty, scoped, session_tenant,
};

/// Server-side handler for the mutable-table gRPC surface. Holds a shared engine
/// session it wraps in a [`LocalSession`] per call to reach the unified
/// transport surface.
pub struct MutableTableServer {
    session: Arc<InferenceSession>,
}

impl MutableTableServer {
    pub fn new(session: Arc<InferenceSession>) -> Self {
        Self { session }
    }

    /// A [`Session`] over the shared engine; see [`crate::grpc::inference`] for
    /// the tenant-scope wiring rationale.
    fn local(&self) -> Session {
        Session::Local(LocalSession::new(Arc::clone(&self.session)))
    }
}

#[tonic::async_trait]
impl MutableTableService for MutableTableServer {
    async fn create_mutable_table(
        &self,
        request: Request<pb::CreateMutableTableRequest>,
    ) -> Result<Response<pb::CreateMutableTableResponse>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        let def_proto = req
            .definition
            .ok_or_else(|| Status::invalid_argument("definition is required"))?;
        let def = build_definition(def_proto, tenant)?;
        let session = self.local();

        let id = scoped(&self.session, tenant, || session.create_mutable_table(def))
            .await
            .map_err(map_engine_error)?;

        Ok(Response::new(pb::CreateMutableTableResponse {
            mutable_table_id: id.to_string(),
        }))
    }

    async fn drop_mutable_table(
        &self,
        request: Request<pb::DropMutableTableRequest>,
    ) -> Result<Response<()>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        let id = parse_table_id(&req.mutable_table_id)?;
        let session = self.local();

        scoped(&self.session, tenant, || session.drop_mutable_table(&id))
            .await
            .map_err(map_engine_error)?;

        Ok(Response::new(()))
    }
}

/// Build the engine [`jammi_db::store::mutable::MutableTableDefinition`] from the
/// wire message, stamping the resolved session `tenant` onto it (the wire body
/// is tenant-free). All schema/primary-key/order-column validation is delegated
/// to the engine builder so the wire path enforces the identical invariants the
/// in-process path does.
fn build_definition(
    def: pb::MutableTableDefinition,
    tenant: Option<TenantId>,
) -> Result<MutableTableDefinition, Status> {
    let id = parse_table_id(&def.id)?;
    let schema = decode_ipc_schema(&def.schema)?;

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

/// Parse a wire id string into a validated [`MutableTableId`].
fn parse_table_id(id: &str) -> Result<MutableTableId, Status> {
    require_nonempty(id, "mutable_table_id")?;
    MutableTableId::new(id).map_err(|e| Status::invalid_argument(e.to_string()))
}
