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
//! `decode_ipc_schema` helper. The conversion itself is
//! [`jammi_ai::wire::definition_from_proto`].
//!
//! Tenant scope is read from the request's [`crate::grpc::session::
//! SessionTenant`] extension (set upstream by the shared `TenantInterceptor`)
//! and applied via `with_tenant_scoped`, matching every other engine-backed
//! gRPC surface.

use std::sync::Arc;

use jammi_ai::session::InferenceSession;
use jammi_ai::wire::{definition_from_proto, definition_to_proto, parse_table_id};
use jammi_ai::{LocalSession, Session};
use tonic::{Request, Response, Status};

use crate::grpc::proto::mutable_table as pb;
use crate::grpc::proto::mutable_table::mutable_table_service_server::MutableTableService;
use crate::grpc::wire::{map_engine_error, scoped, session_tenant};

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
        let def = definition_from_proto(def_proto, tenant)?;
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

    async fn list_mutable_tables(
        &self,
        request: Request<pb::ListMutableTablesRequest>,
    ) -> Result<Response<pb::ListMutableTablesResponse>, Status> {
        let tenant = session_tenant(&request);
        let session = self.local();

        let defs = scoped(&self.session, tenant, || session.list_mutable_tables())
            .await
            .map_err(map_engine_error)?;

        // The wire body stays tenant-free (the catalog row's tenant is the
        // session scope, not a message field), so each definition encodes the
        // same projection the create path carries.
        let definitions = defs
            .iter()
            .map(definition_to_proto)
            .collect::<Result<Vec<_>, Status>>()?;
        Ok(Response::new(pb::ListMutableTablesResponse { definitions }))
    }
}
