//! `ChannelService` gRPC implementation.
//!
//! Two verbs land on the wire: `RegisterChannel` and `AddChannelColumns`. Each
//! is a thin adapter over the transport-agnostic [`Session`]/[`LocalSession`]
//! abstraction (never raw [`InferenceSession`] calls): proto in, one
//! `Session::*_channel*` call, proto out. The service reimplements no catalog
//! logic.
//!
//! The messages mirror the Rust [`jammi_db::catalog::channel_repo::ChannelSpec`]
//! / [`jammi_db::catalog::channel_repo::ChannelColumn`] field for field; the
//! proto [`pb::ChannelColumnType`] mirrors the engine's closed
//! [`jammi_db::catalog::channel_repo::ChannelColumnType`] enum.
//!
//! Tenant scope is read from the request's [`crate::grpc::session::
//! SessionTenant`] extension (set upstream by the shared `TenantInterceptor`)
//! and applied via `with_tenant_scoped`, matching every other engine-backed
//! gRPC surface.

use std::sync::Arc;

use jammi_ai::session::InferenceSession;
use jammi_ai::{LocalSession, Session};
use jammi_db::catalog::channel_repo::{ChannelColumn, ChannelColumnType, ChannelSpec};
use jammi_db::ChannelId;
use tonic::{Request, Response, Status};

use crate::grpc::proto::channel as pb;
use crate::grpc::proto::channel::channel_service_server::ChannelService;
use crate::grpc::wire::{map_engine_error, require_nonempty, scoped, session_tenant};

/// Server-side handler for the channel gRPC surface. Holds a shared engine
/// session it wraps in a [`LocalSession`] per call to reach the unified
/// transport surface.
pub struct ChannelServer {
    session: Arc<InferenceSession>,
}

impl ChannelServer {
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
impl ChannelService for ChannelServer {
    async fn register_channel(
        &self,
        request: Request<pb::RegisterChannelRequest>,
    ) -> Result<Response<()>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        let id = parse_channel_id(&req.channel_id)?;
        let columns = columns_from_proto(req.columns)?;
        let spec = ChannelSpec {
            id,
            priority: req.priority,
            columns,
        };
        let session = self.local();

        scoped(&self.session, tenant, || session.register_channel(&spec))
            .await
            .map_err(map_engine_error)?;

        Ok(Response::new(()))
    }

    async fn add_channel_columns(
        &self,
        request: Request<pb::AddChannelColumnsRequest>,
    ) -> Result<Response<()>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        let id = parse_channel_id(&req.channel_id)?;
        let columns = columns_from_proto(req.columns)?;
        let session = self.local();

        scoped(&self.session, tenant, || {
            session.add_channel_columns(&id, &columns)
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(()))
    }
}

/// Parse a wire channel id into the validated [`ChannelId`] newtype.
fn parse_channel_id(id: &str) -> Result<ChannelId, Status> {
    require_nonempty(id, "channel_id")?;
    ChannelId::new(id).map_err(|e| Status::invalid_argument(e.to_string()))
}

/// Map the wire columns onto the engine's [`ChannelColumn`], rejecting a missing
/// or unspecified column type — a column that names no type is a client error,
/// not a silent default.
fn columns_from_proto(columns: Vec<pb::ChannelColumn>) -> Result<Vec<ChannelColumn>, Status> {
    columns
        .into_iter()
        .map(|c| {
            require_nonempty(&c.name, "column name")?;
            Ok(ChannelColumn {
                name: c.name,
                data_type: column_type_from_proto(c.data_type)?,
            })
        })
        .collect()
}

/// Map the proto [`pb::ChannelColumnType`] onto the engine's closed
/// [`ChannelColumnType`]. An unspecified type is rejected.
fn column_type_from_proto(ty: i32) -> Result<ChannelColumnType, Status> {
    match pb::ChannelColumnType::try_from(ty) {
        Ok(pb::ChannelColumnType::Float32) => Ok(ChannelColumnType::Float32),
        Ok(pb::ChannelColumnType::Float64) => Ok(ChannelColumnType::Float64),
        Ok(pb::ChannelColumnType::Int32) => Ok(ChannelColumnType::Int32),
        Ok(pb::ChannelColumnType::Int64) => Ok(ChannelColumnType::Int64),
        Ok(pb::ChannelColumnType::Utf8) => Ok(ChannelColumnType::Utf8),
        Ok(pb::ChannelColumnType::Boolean) => Ok(ChannelColumnType::Boolean),
        Ok(pb::ChannelColumnType::Unspecified) | Err(_) => Err(Status::invalid_argument(
            "column data_type must be specified",
        )),
    }
}
