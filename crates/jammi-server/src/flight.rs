//! Arrow Flight SQL server backed by a DataFusion `SessionContext`.
//!
//! Two service shapes are exported:
//!
//! - [`serve_flight`] — minimal single-tenant deployment. Flight SQL listens
//!   alone on `addr`; queries observe the engine's session as configured.
//! - [`serve_flight_with_catalog_service`] — multi-tenant deployment. The
//!   gRPC `CatalogService` and Flight SQL coexist on one Tonic server, both
//!   binding tenants through the engine-default
//!   [`crate::grpc::session::SessionIdTenantResolver`] over a shared
//!   [`crate::grpc::session::SessionStore`]. Clients call
//!   `CatalogService.SetTenant` to bind their tenant; subsequent Flight SQL
//!   queries on the same `jammi-session-id` header run scoped to that tenant
//!   via the [`TenantBoundProvider`]. This Flight-only path mounts no engine,
//!   so only the engine-free control verbs (the tenant trio + `GetServerInfo`)
//!   answer here; the engine-backed catalog verbs are reached on the full gRPC
//!   chain instead.

use std::net::SocketAddr;

use async_trait::async_trait;
use datafusion::execution::context::{SessionContext, SessionState};
use datafusion_flight_sql_server::service::FlightSqlService;
use datafusion_flight_sql_server::session::SessionStateProvider;
use jammi_db::tenant::TenantContext;
use jammi_db::tenant_scope::TenantBinding;
use tonic::transport::Server;
use tonic::{Request, Status};

use std::sync::Arc;

use tower::Layer;

use crate::grpc::catalog::CatalogServer;
use crate::grpc::proto::catalog::catalog_service_server::CatalogServiceServer;
use crate::grpc::session::{SessionIdTenantResolver, SessionStore, TenantResolver, TenantScope};
use crate::tenant_resolver_layer::TenantResolverLayer;

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

/// Start Flight SQL + `CatalogService` on one Tonic server, sharing a single
/// [`SessionStore`]. Both surfaces bind tenants through the engine-default
/// [`SessionIdTenantResolver`]: the Flight SQL service via its
/// [`TenantBoundProvider`], the `CatalogService` via the async
/// [`TenantResolverLayer`] — the same single-binder mechanism the full gRPC
/// chain uses.
pub async fn serve_flight_with_catalog_service(
    base_ctx: &SessionContext,
    base_tenant_binding: TenantBinding,
    addr: SocketAddr,
    store: SessionStore,
) -> Result<(), Box<dyn std::error::Error>> {
    let resolver = SessionIdTenantResolver::arc(store.clone());
    let provider = TenantBoundProvider::new(
        base_ctx.state(),
        base_tenant_binding.clone(),
        Arc::clone(&resolver),
    );
    let flight = FlightSqlService::new_with_provider(Box::new(provider));
    let flight_svc = arrow_flight::flight_service_server::FlightServiceServer::new(flight);

    // This Flight-SQL-only path mounts just Flight + CatalogService — the core
    // handshake surface, no optional tiers, no engine — so it advertises core
    // only and answers the engine-free control verbs (the tenant trio +
    // `GetServerInfo`). The catalog binds tenants through the same resolver via
    // the async tenant-binding layer.
    let catalog_svc =
        TenantResolverLayer::new(resolver).layer(CatalogServiceServer::new(CatalogServer::new(
            store,
            crate::tiers::TierSet::resolve(std::iter::empty())?,
            None,
        )));

    tracing::info!(
        "Flight SQL + CatalogService listening on {addr} \
         (tenant binding via jammi-session-id header)"
    );
    Server::builder()
        .add_service(flight_svc)
        .add_service(catalog_svc)
        .serve(addr)
        .await?;
    Ok(())
}

/// `SessionStateProvider` that mutates the engine's shared `TenantBinding`
/// based on the scope the one [`TenantResolver`] resolves from each query's
/// metadata — the SAME resolver the gRPC plane binds through (the engine default
/// reads `jammi-session-id` → `SessionStore`; a downstream's authenticating
/// resolver reads its own credential). The mutation is process-global because
/// the binding is shared; concurrent Flight SQL requests on different tenants
/// serialise on the write lock.
///
/// **Concurrency caveat (inherited, tracked separately):** if a deployment serves
/// more than one tenant
/// concurrently through Flight SQL (rather than gRPC + per-statement
/// session bindings), the race window between binding mutation and SQL
/// execution can return rows under a stale binding. The gRPC `CatalogService`
/// surface is the supported multi-tenant Flight SQL path for now; a future
/// refactor moves the binding off the shared `SessionContext` into per-plan
/// `ConfigExtension` state (SPEC-03 §13 OQ#3). Downstream gRPC consumers
/// that own their own request handlers can avoid the race today by routing
/// each request through
/// [`jammi_db::session::JammiSession::with_tenant_scoped`], which
/// installs the tenant as a Tokio task-local for the duration of the
/// closure.
pub struct TenantBoundProvider {
    base_state: SessionState,
    binding: TenantBinding,
    /// The one tenant-binding resolver — the SAME resolver the gRPC plane binds
    /// through, threaded down to the Flight SQL `db.sql` lane (MUST-FIX 2, #220).
    /// It resolves each query's scope and binds it, so Flight and gRPC can never
    /// disagree about who a request is; an authenticating resolver's rejection
    /// fails the query before any binding, closing the cross-transport bypass.
    resolver: Arc<dyn TenantResolver>,
}

impl TenantBoundProvider {
    pub fn new(
        base_state: SessionState,
        binding: TenantBinding,
        resolver: Arc<dyn TenantResolver>,
    ) -> Self {
        Self {
            base_state,
            binding,
            resolver,
        }
    }
}

#[async_trait]
impl SessionStateProvider for TenantBoundProvider {
    async fn new_context(&self, request: &Request<()>) -> Result<SessionState, Status> {
        // Resolve the query's scope through the one resolver and bind it — or
        // reject (an authenticating resolver's `Err` fails the query here, so the
        // `db.sql` lane never runs unscoped on a missing/invalid credential). An
        // explicit `Global` scope binds the unscoped context.
        let ctx = match self.resolver.resolve(request.metadata()).await? {
            TenantScope::Tenant(t) => TenantContext::Scoped(t),
            TenantScope::Global => TenantContext::Unscoped,
        };
        self.binding.set_shared(ctx);

        Ok(self.base_state.clone())
    }
}
