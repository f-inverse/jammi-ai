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
use jammi_ai::wire::{channel_to_proto, columns_from_proto, parse_channel_id};
use jammi_ai::{LocalSession, Session};
use jammi_db::catalog::channel_repo::ChannelSpec;
use tonic::{Request, Response, Status};

use crate::grpc::proto::channel as pb;
use crate::grpc::proto::channel::channel_service_server::ChannelService;
use crate::grpc::wire::{map_engine_error, scoped, session_tenant};

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

    async fn list_channels(
        &self,
        request: Request<pb::ListChannelsRequest>,
    ) -> Result<Response<pb::ListChannelsResponse>, Status> {
        let tenant = session_tenant(&request);
        let session = self.local();

        let specs = scoped(&self.session, tenant, || session.list_channels())
            .await
            .map_err(map_engine_error)?;

        let channels = specs.iter().map(channel_to_proto).collect();
        Ok(Response::new(pb::ListChannelsResponse { channels }))
    }
}
