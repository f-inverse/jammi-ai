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
//!   through [`ResolvedFlightScope`]. This Flight-only path mounts no engine,
//!   so only the engine-free control verbs (the tenant trio + `GetServerInfo`)
//!   answer here; the engine-backed catalog verbs are reached on the full gRPC
//!   chain instead.
//!
//! Every request that reaches the engine enters its own scope
//! ([`FlightScope`]) before it plans: its handler runs inside it, and so does
//! every poll of the stream it answers with. A query planned there carries
//! the scope in its state ([`jammi_db::tenant_scope::TenantConfig`]) for the
//! work DataFusion runs on other tasks. Requests share nothing that holds a
//! tenant, so concurrent requests of different tenants never see each
//! other's.

use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

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
use datafusion_flight_sql_server::service::FlightSqlService;
use datafusion_flight_sql_server::session::SessionStateProvider;
use datafusion_flight_sql_server::state::CommandTicket;
use futures::{Stream, StreamExt, TryStreamExt};
use jammi_db::error::{ErrorClass, JammiError};
use jammi_db::session::QueryContext;
use jammi_db::store::statement::StatementClass;
use jammi_db::tenant::TenantContext;
use jammi_db::tenant_scope::TenantBinding;
use prost::Message;
use tonic::metadata::MetadataMap;
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
    let service = JammiFlightService::new(Arc::new(SessionScope(ctx.clone())));
    tracing::info!("Flight SQL server listening on {addr}");
    Server::builder()
        .add_service(FlightServiceServer::new(service))
        .serve(addr)
        .await?;
    Ok(())
}

/// How a Flight request is scoped: the tenant its metadata names, and the
/// query context it plans through.
#[async_trait]
pub trait FlightScope: Send + Sync {
    /// The query context every request plans through.
    fn context(&self) -> &QueryContext;

    /// The scope `metadata` names — `None` for the session's own — or the
    /// status that refuses the request before it plans.
    async fn resolve(&self, metadata: &MetadataMap) -> Result<Option<TenantContext>, Status>;
}

/// The single-tenant shape: every request runs under the session's own
/// scope.
struct SessionScope(QueryContext);

#[async_trait]
impl FlightScope for SessionScope {
    fn context(&self) -> &QueryContext {
        &self.0
    }

    async fn resolve(&self, _metadata: &MetadataMap) -> Result<Option<TenantContext>, Status> {
        Ok(None)
    }
}

/// Start Flight SQL + `CatalogService` on one Tonic server, sharing a single
/// [`SessionStore`]. Both surfaces bind tenants through the engine-default
/// [`SessionIdTenantResolver`]: the Flight SQL service via its
/// [`ResolvedFlightScope`], the `CatalogService` via the async
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
    addr: SocketAddr,
    store: SessionStore,
) -> Result<(), Box<dyn std::error::Error>> {
    let resolver = SessionIdTenantResolver::arc(store.clone());
    let scope = ResolvedFlightScope::new(base_ctx.clone(), Arc::clone(&resolver));
    let flight_svc = FlightServiceServer::new(JammiFlightService::new(Arc::new(scope)));

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

/// The multi-tenant shape's [`FlightScope`]: each request runs under the
/// scope the one [`TenantResolver`] reads from its metadata — the SAME
/// resolver the gRPC plane binds through (the engine default reads
/// `jammi-session-id` → `SessionStore`; a downstream's authenticating
/// resolver reads its own credential), so Flight and gRPC can never disagree
/// about who a request is. An authenticating resolver's rejection fails the
/// request before it plans, closing the cross-transport bypass.
pub struct ResolvedFlightScope {
    ctx: QueryContext,
    resolver: Arc<dyn TenantResolver>,
}

impl ResolvedFlightScope {
    pub fn new(ctx: QueryContext, resolver: Arc<dyn TenantResolver>) -> Self {
        Self { ctx, resolver }
    }
}

#[async_trait]
impl FlightScope for ResolvedFlightScope {
    fn context(&self) -> &QueryContext {
        &self.ctx
    }

    async fn resolve(&self, metadata: &MetadataMap) -> Result<Option<TenantContext>, Status> {
        Ok(Some(match self.resolver.resolve(metadata).await? {
            TenantScope::Tenant(t) => TenantContext::Scoped(t),
            TenantScope::Global => TenantContext::Unscoped,
        }))
    }
}

/// The Flight service the engine mounts: the inner `FlightSqlService` with
/// its statement-ticket arm taken over (module doc). Every request that
/// reaches the engine enters the scope its [`FlightScope`] resolves, and the
/// inner service's state is the one planning under it.
pub struct JammiFlightService {
    inner: FlightSqlService,
    scope: Arc<dyn FlightScope>,
}

/// The inner service's provider: the request's state, planning under the
/// scope the request entered (installed by [`JammiFlightService`] around the
/// call that asks for it).
struct EnteredScope(Arc<dyn FlightScope>);

#[async_trait]
impl SessionStateProvider for EnteredScope {
    async fn new_context(&self, _request: &Request<()>) -> Result<SessionState, Status> {
        Ok(self.0.context().state())
    }
}

/// A response stream polled under the scope its request entered: tonic
/// polls it after the handler returned, off the handler's scope.
struct Entered<S> {
    scope: Option<TenantContext>,
    stream: S,
}

impl<S: Stream + Unpin> Stream for Entered<S> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<S::Item>> {
        let this = &mut *self;
        let stream = Pin::new(&mut this.stream);
        match this.scope {
            Some(ctx) => TenantBinding::within_sync(ctx, || stream.poll_next(cx)),
            None => stream.poll_next(cx),
        }
    }
}

impl JammiFlightService {
    /// Mount the inner service over `scope`, taking the statement arm.
    pub fn new(scope: Arc<dyn FlightScope>) -> Self {
        Self {
            inner: FlightSqlService::new_with_provider(Box::new(EnteredScope(Arc::clone(&scope)))),
            scope,
        }
    }

    /// Run `call` under `scope`, the session's own when `None`.
    async fn entered<T>(
        scope: Option<TenantContext>,
        call: impl std::future::Future<Output = Result<T, Status>>,
    ) -> Result<T, Status> {
        match scope {
            Some(ctx) => TenantBinding::within(ctx, call).await,
            None => call.await,
        }
    }

    /// Run `call` and every poll of the stream it answers with under the
    /// scope `metadata` names.
    async fn entered_stream<S, T>(
        &self,
        metadata: &MetadataMap,
        call: impl std::future::Future<Output = Result<Response<S>, Status>>,
    ) -> Result<Response<BoxedStream<T>>, Status>
    where
        S: Stream<Item = Result<T, Status>> + Send + Unpin + 'static,
        T: 'static,
    {
        let scope = self.scope.resolve(metadata).await?;
        let response = Self::entered(scope, call).await?;
        Ok(response.map(|stream| Box::pin(Entered { scope, stream }) as BoxedStream<T>))
    }

    /// Run the unary `call` under the scope `metadata` names.
    async fn entered_unary<T>(
        &self,
        metadata: &MetadataMap,
        call: impl std::future::Future<Output = Result<T, Status>>,
    ) -> Result<T, Status> {
        let scope = self.scope.resolve(metadata).await?;
        Self::entered(scope, call).await
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

    /// The statement a descriptor carries, when it is a statement descriptor
    /// (`CommandStatementQuery`, what a client's `execute` sends first);
    /// `None` for every other command, which the inner service answers.
    fn descriptor_statement(descriptor: &FlightDescriptor) -> Option<String> {
        let any = arrow_flight::sql::Any::decode(descriptor.cmd.clone()).ok()?;
        match Command::try_from(any).ok()? {
            Command::CommandStatementQuery(query) => Some(query.query),
            _ => None,
        }
    }

    /// Plan `query` through the engine's statement entry on the request's own
    /// state, executing nothing, so a statement the engine refuses is refused
    /// at the first call with the status the engine's own classification gives
    /// it — a missing relation is `NOT_FOUND` here exactly as it is in process —
    /// rather than with the inner service's planner's untyped failure.
    async fn plan_statement(&self, query: &str) -> Result<(), Status> {
        StatementClass::plan(&self.scope.context().state(), query)
            .await
            .map(|_| ())
            .map_err(engine_status)
    }

    /// Run `query` through the engine's statement entry on the request's
    /// own state and encode its rows under the plan's schema — the schema
    /// the inner service advertised for it (`get_flight_info` derives the
    /// same one from the same logical plan).
    async fn do_get_statement(&self, query: &str) -> Result<Response<BoxedStream<FlightData>>, Status> {
        let frame = self
            .scope
            .context()
            .sql(query)
            .await
            .map_err(engine_status)?;
        let schema = Arc::new(frame.schema().as_arrow().clone());
        let stream = frame.execute_stream().await.map_err(engine_status)?;
        let flight = FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(stream.map_err(|e| FlightError::from(engine_status(e))))
            .map_err(Status::from)
            .boxed();
        Ok(Response::new(flight))
    }
}

/// The status a statement's DataFusion error reaches the client as — raised
/// while planning or while its rows stream — the engine's own mapping over the
/// classified `JammiError`, so a typed refusal (a routed plan's among them,
/// which arrives mid-stream) carries the same code and detail the gRPC plane
/// gives it.
///
/// It also carries Arrow Flight's own status trailer, `x-arrow-status`. An
/// Arrow client that finds none rebuilds the error from the gRPC code alone, by
/// a table that folds `FAILED_PRECONDITION` and `RESOURCE_EXHAUSTED` into
/// `INVALID_ARGUMENT`, and drops the typed detail; with it, the client keeps the
/// `grpc-status-details-bin` envelope as the error's extra info, so a Flight
/// client classifies a statement's failure exactly as the typed gRPC lane does.
fn engine_status(e: datafusion::error::DataFusionError) -> Status {
    let err = JammiError::from(e);
    let arrow_code = arrow_status_code(err.class());
    let mut status = crate::grpc::wire::map_engine_error(err);
    status.metadata_mut().insert(
        "x-arrow-status",
        tonic::metadata::MetadataValue::from(arrow_code),
    );
    status
}

/// The Arrow `StatusCode` (`arrow/status.h`) nearest an engine error class,
/// for the Arrow clients that read `x-arrow-status` without the detail.
fn arrow_status_code(class: ErrorClass) -> u16 {
    const KEY_ERROR: u16 = 2;
    const INVALID: u16 = 4;
    const IO_ERROR: u16 = 5;
    const CANCELLED: u16 = 8;
    const ALREADY_EXISTS: u16 = 45;
    match class {
        ErrorClass::NotFound => KEY_ERROR,
        ErrorClass::InvalidArgument
        | ErrorClass::FailedPrecondition
        | ErrorClass::ResourceExhausted => INVALID,
        ErrorClass::AlreadyExists => ALREADY_EXISTS,
        ErrorClass::Cancelled => CANCELLED,
        ErrorClass::PermissionDenied
        | ErrorClass::Aborted
        | ErrorClass::Unavailable
        | ErrorClass::Internal => IO_ERROR,
    }
}

type BoxedStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

/// The handshake and the listing calls reach no engine state and run as they
/// arrive: a deployment's handshake is where its credential is presented, so
/// it is never refused for the credential it carries. Every other call
/// enters its request's scope.
#[async_trait]
impl FlightService for JammiFlightService {
    type HandshakeStream = <FlightSqlService as FlightService>::HandshakeStream;
    type ListFlightsStream = <FlightSqlService as FlightService>::ListFlightsStream;
    type DoGetStream = BoxedStream<FlightData>;
    type DoPutStream = BoxedStream<arrow_flight::PutResult>;
    type DoExchangeStream = BoxedStream<FlightData>;
    type DoActionStream = BoxedStream<arrow_flight::Result>;
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
        let metadata = request.metadata().clone();
        let statement = Self::descriptor_statement(request.get_ref());
        self.entered_unary(&metadata, async {
            if let Some(query) = statement {
                self.plan_statement(&query).await?;
            }
            self.inner.get_flight_info(request).await
        })
        .await
    }

    async fn poll_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        let metadata = request.metadata().clone();
        self.entered_unary(&metadata, self.inner.poll_flight_info(request))
            .await
    }

    async fn get_schema(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        let metadata = request.metadata().clone();
        self.entered_unary(&metadata, self.inner.get_schema(request))
            .await
    }

    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let metadata = request.metadata().clone();
        match Self::statement_query(request.get_ref()) {
            Some(query) => {
                self.entered_stream(&metadata, self.do_get_statement(&query))
                    .await
            }
            None => self.entered_stream(&metadata, self.inner.do_get(request)).await,
        }
    }

    async fn do_put(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        let metadata = request.metadata().clone();
        self.entered_stream(&metadata, self.inner.do_put(request))
            .await
    }

    async fn do_exchange(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        let metadata = request.metadata().clone();
        self.entered_stream(&metadata, self.inner.do_exchange(request))
            .await
    }

    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        let metadata = request.metadata().clone();
        self.entered_stream(&metadata, self.inner.do_action(request))
            .await
    }

    async fn list_actions(
        &self,
        request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        self.inner.list_actions(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jammi_db::error::Missing;
    use tonic::Code;

    /// A statement refused over a missing relation reaches a Flight client as
    /// `NOT_FOUND` with Arrow's own status trailer (`KeyError`), the trailer an
    /// Arrow client needs to keep the engine's typed detail as the error's
    /// extra info rather than rebuild the error from the gRPC code alone.
    #[test]
    fn a_refused_statement_carries_the_arrow_status_trailer() {
        let refused = datafusion::error::DataFusionError::External(Box::new(JammiError::NotFound(
            Missing::Relation {
                name: "mutable.public.notes".to_string(),
            },
        )));
        let status = engine_status(refused);
        assert_eq!(status.code(), Code::NotFound);
        assert_eq!(
            status
                .metadata()
                .get("x-arrow-status")
                .map(|v| v.to_str().unwrap()),
            Some("2"),
            "an Arrow client reads KeyError (2) for a missing relation"
        );
        assert!(
            !status.details().is_empty(),
            "the typed detail rides beside the trailer"
        );
    }
}
