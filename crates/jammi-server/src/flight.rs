//! Arrow Flight SQL server backed by a DataFusion `SessionContext`.
//!
//! Two service shapes are exported:
//!
//! - [`serve_flight`] — minimal single-tenant deployment. Flight SQL listens
//!   alone on `addr`; queries observe the engine's session as configured.
//! - [`serve_flight_with_session_service`] — multi-tenant deployment. The
//!   gRPC `SessionService` and Flight SQL coexist on one Tonic server
//!   behind a shared [`crate::grpc::session::TenantInterceptor`] +
//!   [`crate::grpc::session::SessionStore`]. Clients call
//!   `SessionService.SetTenant` to bind their tenant; subsequent Flight SQL
//!   queries on the same `jammi-session-id` header run scoped to that tenant
//!   via the [`TenantBoundProvider`].

use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::execution::context::{SessionContext, SessionState};
use datafusion_flight_sql_server::service::FlightSqlService;
use datafusion_flight_sql_server::session::SessionStateProvider;
use jammi_engine::tenant::TenantContext;
use jammi_engine::tenant_scope::TenantBinding;
use tonic::transport::Server;
use tonic::{Request, Status};

use crate::grpc::proto::session::session_service_server::SessionServiceServer;
use crate::grpc::session::{
    SessionId, SessionServer, SessionStore, TenantInterceptor, SESSION_HEADER,
};

/// Start an Arrow Flight SQL server alone on `addr`. Single-tenant or
/// in-process embedding shape.
pub async fn serve_flight(
    ctx: &SessionContext,
    addr: SocketAddr,
) -> Result<(), Box<dyn std::error::Error>> {
    let service = FlightSqlService::new(ctx.state());
    tracing::info!("Flight SQL server listening on {addr}");
    service.serve(addr.to_string()).await
}

/// Start Flight SQL + `SessionService` on one Tonic server, sharing a single
/// [`SessionStore`]. The Flight SQL service uses a [`TenantBoundProvider`]
/// that reads `SessionTenant` from request metadata and updates the shared
/// `TenantBinding` for the duration of the query.
pub async fn serve_flight_with_session_service(
    base_ctx: &SessionContext,
    base_tenant_binding: TenantBinding,
    addr: SocketAddr,
    store: SessionStore,
) -> Result<(), Box<dyn std::error::Error>> {
    let provider = TenantBoundProvider::new(
        base_ctx.state(),
        Arc::clone(&base_tenant_binding),
        store.clone(),
    );
    let flight = FlightSqlService::new_with_provider(Box::new(provider));
    let flight_svc = arrow_flight::flight_service_server::FlightServiceServer::new(flight);

    let interceptor = TenantInterceptor::new(store.clone());
    let session_svc =
        SessionServiceServer::with_interceptor(SessionServer::new(store), interceptor);

    tracing::info!(
        "Flight SQL + SessionService listening on {addr} \
         (tenant binding via jammi-session-id header)"
    );
    Server::builder()
        .add_service(flight_svc)
        .add_service(session_svc)
        .serve(addr)
        .await?;
    Ok(())
}

/// `SessionStateProvider` that mutates the engine's shared `TenantBinding`
/// based on the request's `jammi-session-id` → `SessionStore` lookup. The
/// mutation is process-global because the binding is shared; concurrent
/// Flight SQL requests on different tenants serialise on the write lock.
///
/// **Concurrency caveat:** if a deployment serves more than one tenant
/// concurrently through Flight SQL (rather than gRPC + per-statement
/// session bindings), the race window between binding mutation and SQL
/// execution can return rows under a stale binding. The gRPC `SessionService`
/// surface is the supported multi-tenant Flight SQL path for now; a future
/// refactor moves the binding off the shared `SessionContext` into per-plan
/// `ConfigExtension` state (SPEC-03 §13 OQ#3).
pub struct TenantBoundProvider {
    base_state: SessionState,
    binding: TenantBinding,
    store: SessionStore,
}

impl TenantBoundProvider {
    pub fn new(base_state: SessionState, binding: TenantBinding, store: SessionStore) -> Self {
        Self {
            base_state,
            binding,
            store,
        }
    }
}

#[async_trait]
impl SessionStateProvider for TenantBoundProvider {
    async fn new_context(&self, request: &Request<()>) -> Result<SessionState, Status> {
        let tenant = request
            .metadata()
            .get(SESSION_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(SessionId::new)
            .and_then(|sid| self.store.get(&sid));

        let mut binding = self
            .binding
            .write()
            .map_err(|e| Status::internal(format!("tenant binding lock poisoned: {e}")))?;
        *binding = match tenant {
            Some(t) => TenantContext::Scoped(t),
            None => TenantContext::Unscoped,
        };
        drop(binding);

        Ok(self.base_state.clone())
    }
}
