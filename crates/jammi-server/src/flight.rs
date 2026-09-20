//! Arrow Flight SQL server backed by the engine session's query context.
//!
//! [`JammiFlightService`] is the Flight service every shape mounts: the
//! `datafusion-flight-sql-server` service, with one arm taken over — a
//! statement ticket (`CommandStatementQuery`, what a client's `execute`
//! then `do_get` carries) runs through the engine's own statement entry,
//! [`QueryContext::sql`], so the statement's class decides where its plan
//! runs exactly as it does for a statement issued in-process. Every other
//! ticket, descriptor and action is the inner service's.
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
use std::pin::Pin;
use std::sync::Arc;

use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::flight_service_server::{FlightService, FlightServiceServer};
use arrow_flight::sql::Command;
use arrow_flight::{
    Action, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo, HandshakeRequest, PollInfo,
    SchemaResult, Ticket,
};
use async_trait::async_trait;
use datafusion::execution::context::SessionState;
use datafusion::prelude::SessionContext;
use datafusion_flight_sql_server::service::FlightSqlService;
use datafusion_flight_sql_server::session::SessionStateProvider;
use datafusion_flight_sql_server::state::CommandTicket;
use futures::{Stream, StreamExt, TryStreamExt};
use jammi_db::error::JammiError;
use jammi_db::session::QueryContext;
use jammi_db::tenant::TenantContext;
use jammi_db::tenant_scope::TenantBinding;
use tonic::transport::Server;
use tonic::{Request, Response, Status, Streaming};
use tower::Layer;

use crate::grpc::catalog::CatalogServer;
use crate::grpc::proto::catalog::catalog_service_server::CatalogServiceServer;
use crate::grpc::session::{SessionIdTenantResolver, SessionStore, TenantResolver, TenantScope};
use crate::tenant_resolver_layer::TenantResolverLayer;

/// Start an Arrow Flight SQL server alone on `addr`. Single-tenant or
/// in-process embedding shape.
///
/// NOT BOUNDED: this standalone entry point does not go through
/// `assemble_grpc_chain`, so it carries NONE of `[server.limits]` — no
/// `max_message_bytes` decode cap (unlike every service `assemble_grpc_chain`
/// mounts, which is built with `.max_decoding_message_size` per the
/// `message_size` counting rule in `crate::limits`'s module docs), no in-flight/per-connection bound, no
/// wait-timeout or stream-budget enforcement. A deployment that needs those
/// bounds on its Flight SQL surface should reach it through the full chain
/// (`assemble_grpc_chain` → [`crate::runtime::AssembledChain`]) instead of
/// this function.
pub async fn serve_flight(
    ctx: &QueryContext,
    addr: SocketAddr,
) -> Result<(), Box<dyn std::error::Error>> {
    let service = JammiFlightService::new(Arc::new(StaticState(ctx.state())));
    tracing::info!("Flight SQL server listening on {addr}");
    Server::builder()
        .add_service(FlightServiceServer::new(service))
        .serve(addr)
        .await?;
    Ok(())
}

/// One state for every request: the single-tenant shape's provider.
struct StaticState(SessionState);

#[async_trait]
impl SessionStateProvider for StaticState {
    async fn new_context(&self, _request: &Request<()>) -> Result<SessionState, Status> {
        Ok(self.0.clone())
    }
}

/// Start Flight SQL + `CatalogService` on one Tonic server, sharing a single
/// [`SessionStore`]. Both surfaces bind tenants through the engine-default
/// [`SessionIdTenantResolver`]: the Flight SQL service via its
/// [`TenantBoundProvider`], the `CatalogService` via the async
/// [`TenantResolverLayer`] — the same single-binder mechanism the full gRPC
/// chain uses.
///
/// NOT BOUNDED (same caveat as [`serve_flight`]): this is a standalone
/// `Server::builder()` assembly, not `assemble_grpc_chain`, so it carries none
/// of `[server.limits]` — no `max_message_bytes` decode cap, no in-flight/
/// per-connection bound, no wait-timeout or stream-budget enforcement. A
/// deployment that needs those bounds should reach the engine through the full
/// chain instead of this function.
pub async fn serve_flight_with_catalog_service(
    base_ctx: &QueryContext,
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
    let flight_svc = FlightServiceServer::new(JammiFlightService::new(Arc::new(provider)));

    // This Flight-SQL-only path mounts just Flight + CatalogService — the core
    // handshake surface, no optional tiers, no engine — so it advertises core
    // only and answers the engine-free control verbs (the tenant trio +
    // `GetServerInfo`). The catalog binds tenants through the same resolver via
    // the async tenant-binding layer.
    let catalog_svc =
        TenantResolverLayer::new(resolver).layer(CatalogServiceServer::new(CatalogServer::new(
            store,
            crate::tiers::TierSet::resolve(std::iter::empty()),
            None,
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
/// **Concurrency caveat:** if a deployment serves more than one tenant
/// concurrently through Flight SQL (rather than gRPC + per-statement
/// session bindings), the race window between binding mutation and SQL
/// execution can return rows under a stale binding, because the binding
/// lives on the shared session. The gRPC `CatalogService` surface
/// is the supported multi-tenant path. Downstream gRPC consumers that own
/// their own request handlers avoid the race by routing each request through
/// [`jammi_db::session::JammiSession::with_tenant_scoped`], which
/// installs the tenant as a Tokio task-local for the duration of the
/// closure.
pub struct TenantBoundProvider {
    base_state: SessionState,
    binding: TenantBinding,
    /// The one tenant-binding resolver — the SAME resolver the gRPC plane binds
    /// through, threaded down to the Flight SQL `db.sql` lane.
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

/// The Flight service the engine mounts: the inner `FlightSqlService` with
/// its statement-ticket arm taken over (module doc). One provider serves
/// both — the inner's own requests and the taken-over arm resolve a
/// request's state, and its tenant, through the same object.
pub struct JammiFlightService {
    inner: FlightSqlService,
    provider: Arc<dyn SessionStateProvider>,
}

/// The one provider, shared with the inner service (which owns its copy
/// boxed).
struct SharedProvider(Arc<dyn SessionStateProvider>);

#[async_trait]
impl SessionStateProvider for SharedProvider {
    async fn new_context(&self, request: &Request<()>) -> Result<SessionState, Status> {
        self.0.new_context(request).await
    }
}

impl JammiFlightService {
    /// Mount the inner service over `provider`, taking the statement arm.
    pub fn new(provider: Arc<dyn SessionStateProvider>) -> Self {
        Self {
            inner: FlightSqlService::new_with_provider(Box::new(SharedProvider(Arc::clone(
                &provider,
            )))),
            provider,
        }
    }

    /// The statement a ticket carries, when it is a statement ticket (the
    /// inner service's own encoding of `CommandStatementQuery`); `None`
    /// for every other ticket, which the inner service answers.
    fn statement_query(ticket: &Ticket) -> Option<String> {
        match CommandTicket::try_decode(ticket.ticket.clone())
            .ok()?
            .command
        {
            Command::CommandStatementQuery(query) => Some(query.query),
            _ => None,
        }
    }

    /// Run `query` through the engine's statement entry on the request's
    /// own state and encode its rows under the plan's schema — the schema
    /// the inner service advertised for it (`get_flight_info` derives the
    /// same one from the same logical plan).
    async fn do_get_statement(
        &self,
        request: &Request<Ticket>,
        query: &str,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        let inspect =
            Request::from_parts(request.metadata().clone(), request.extensions().clone(), ());
        let state = self.provider.new_context(&inspect).await?;
        let ctx = QueryContext::from(SessionContext::new_with_state(state));
        let frame = ctx.sql(query).await.map_err(engine_status)?;
        let schema = Arc::new(frame.schema().as_arrow().clone());
        let stream = frame.execute_stream().await.map_err(engine_status)?;
        let flight = FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(stream.map_err(|e| FlightError::ExternalError(Box::new(e))))
            .map_err(Status::from)
            .boxed();
        Ok(Response::new(flight))
    }
}

/// The status a statement's DataFusion error reaches the client as: the
/// engine's own mapping over the classified `JammiError`, so a typed
/// refusal (a routed plan's among them) carries the same code and detail
/// the gRPC plane gives it.
fn engine_status(e: datafusion::error::DataFusionError) -> Status {
    crate::grpc::wire::map_engine_error(JammiError::from(e))
}

type BoxedStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

#[async_trait]
impl FlightService for JammiFlightService {
    type HandshakeStream = <FlightSqlService as FlightService>::HandshakeStream;
    type ListFlightsStream = <FlightSqlService as FlightService>::ListFlightsStream;
    type DoGetStream = BoxedStream<FlightData>;
    type DoPutStream = <FlightSqlService as FlightService>::DoPutStream;
    type DoExchangeStream = <FlightSqlService as FlightService>::DoExchangeStream;
    type DoActionStream = <FlightSqlService as FlightService>::DoActionStream;
    type ListActionsStream = <FlightSqlService as FlightService>::ListActionsStream;

    async fn handshake(
        &self,
        request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        self.inner.handshake(request).await
    }

    async fn list_flights(
        &self,
        request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        self.inner.list_flights(request).await
    }

    async fn get_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        self.inner.get_flight_info(request).await
    }

    async fn poll_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        self.inner.poll_flight_info(request).await
    }

    async fn get_schema(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        self.inner.get_schema(request).await
    }

    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        match Self::statement_query(request.get_ref()) {
            Some(query) => self.do_get_statement(&request, &query).await,
            None => self.inner.do_get(request).await,
        }
    }

    async fn do_put(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        self.inner.do_put(request).await
    }

    async fn do_exchange(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        self.inner.do_exchange(request).await
    }

    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        self.inner.do_action(request).await
    }

    async fn list_actions(
        &self,
        request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        self.inner.list_actions(request).await
    }
}
