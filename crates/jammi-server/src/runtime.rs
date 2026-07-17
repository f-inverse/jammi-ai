//! `OssServer` — the single orchestration entry-point for the OSS
//! `jammi-server` binary.
//!
//! One `OssServer` wires together:
//!
//! - the engine [`InferenceSession`] (catalog, mutable tables, broker)
//! - a [`SessionStore`] shared between Flight SQL and the gRPC services
//! - the Axum side-channel router (`/healthz`, `/readyz`, `/metrics`)
//! - one Tonic server hosting `FlightSqlService + CatalogService +
//!   TriggerService` on a single port
//! - graceful shutdown wired to SIGINT/SIGTERM via a
//!   [`tokio::sync::broadcast`] so every component drains in parallel
//!
//! The structure is intentionally flat: no `runtime/` directory, no
//! per-component sub-modules. When a second binary materialises the same
//! shape can be reused — the orchestration is the engine of last resort
//! and earns its keep by being grep-able in one place.

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;

use arrow_flight::flight_service_server::FlightServiceServer;
use async_trait::async_trait;
use axum::routing::get;
use axum::Router;
use datafusion::execution::context::SessionContext;
use datafusion_flight_sql_server::service::FlightSqlService;
use jammi_ai::session::InferenceSession;
use jammi_db::config::JammiConfig;
use tokio::net::TcpListener;
use tokio::signal;
use tokio::sync::broadcast;
use tonic::transport::server::TcpIncoming;
use tonic::transport::Server;
use tonic_web::GrpcWebLayer;
use tower::Layer;

use crate::error::fallback_handler;
use crate::flight::TenantBoundProvider;
use crate::grpc::audit::AuditServer;
use crate::grpc::catalog::CatalogServer;
use crate::grpc::embedding::EmbeddingServer;
use crate::grpc::eval::EvalServer;
use crate::grpc::inference::InferenceServer;
use crate::grpc::pipeline::PipelineServer;
use crate::grpc::proto::audit::audit_service_server::AuditServiceServer;
use crate::grpc::proto::catalog::catalog_service_server::CatalogServiceServer;
use crate::grpc::proto::embedding::embedding_service_server::EmbeddingServiceServer;
use crate::grpc::proto::eval::eval_service_server::EvalServiceServer;
use crate::grpc::proto::inference::inference_service_server::InferenceServiceServer;
use crate::grpc::proto::pipeline::pipeline_service_server::PipelineServiceServer;
#[cfg(feature = "train")]
use crate::grpc::proto::training::training_service_server::TrainingServiceServer;
use crate::grpc::proto::trigger::trigger_service_server::TriggerServiceServer;
use crate::grpc::session::{SessionIdTenantResolver, SessionStore, TenantResolver};
#[cfg(feature = "train")]
use crate::grpc::training::TrainingServer;
use crate::grpc::trigger::TriggerServer;
use crate::grpc_web_trailers::GrpcWebTrailersLayer;
use crate::metrics_layer::MetricsLayer;
use crate::routes::health::{self, MetricsRegistry};
use crate::tenant_resolver_layer::TenantResolverLayer;
use crate::tiers::{ServiceTier, TierSet};

/// Errors `OssServer::run` can surface to the binary's `main`.
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("config error: {0}")]
    Config(String),
    #[error("service tier: {0}")]
    Tier(#[from] crate::tiers::TierError),
    #[error("engine init: {0}")]
    Engine(#[from] jammi_db::error::JammiError),
    #[error("metrics registry: {0}")]
    Metrics(#[from] prometheus::Error),
    #[error("transport: {0}")]
    Transport(#[from] tonic::transport::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("addr parse: {0}")]
    AddrParse(#[from] std::net::AddrParseError),
}

/// A readiness probe: pings whatever resource readiness depends on. The
/// implementation lives behind a trait so tests can substitute a stub
/// that returns deterministic outcomes (the substrate session itself
/// has more moving parts than a probe test cares about).
#[async_trait]
pub trait ReadinessCheck: Send + Sync {
    /// `Ok(())` means the underlying resource responded; `Err(s)` is a
    /// human-readable failure reason surfaced in the `/readyz` body.
    async fn check(&self) -> Result<(), String>;
}

/// Wrapper that holds the active [`ReadinessCheck`] behind an `Arc` so
/// Axum can share it across handlers via `State`.
pub struct ReadinessProbe {
    inner: Arc<dyn ReadinessCheck>,
}

impl ReadinessProbe {
    pub fn new(inner: Arc<dyn ReadinessCheck>) -> Self {
        Self { inner }
    }

    pub async fn check(&self) -> Result<(), String> {
        self.inner.check().await
    }
}

/// Readiness check backed by the engine's catalog backend. Delegates to
/// [`jammi_db::catalog::Catalog::ping`], which issues a backend-native
/// reachability probe (no transaction, no lock) and surfaces pool failures
/// as [`jammi_db::catalog::backend::BackendError::Unavailable`].
pub struct CatalogPingProbe {
    session: Arc<InferenceSession>,
}

impl CatalogPingProbe {
    pub fn new(session: Arc<InferenceSession>) -> Self {
        Self { session }
    }
}

#[async_trait]
impl ReadinessCheck for CatalogPingProbe {
    async fn check(&self) -> Result<(), String> {
        self.session
            .catalog()
            .ping()
            .await
            .map_err(|e| e.to_string())
    }
}

/// The OSS server instance. Constructed via [`Self::new`] and consumed
/// by [`Self::run`]. Holds every long-lived dependency the binary
/// orchestrates — bind addresses, the engine session, the shared
/// SessionStore, the metrics registry, and the readiness probe.
pub struct OssServer {
    flight_addr: SocketAddr,
    health_addr: SocketAddr,
    session: Arc<InferenceSession>,
    session_store: SessionStore,
    metrics: Arc<MetricsRegistry>,
    readiness: Arc<ReadinessProbe>,
    tiers: TierSet,
}

impl OssServer {
    /// Build an OSS server from `JammiConfig`. Validates the server
    /// configuration up front (parses both bind addresses, rejects two
    /// identical FIXED bind addresses — identical `:0` ephemeral addresses
    /// are allowed, since each resolves to a distinct port at bind),
    /// constructs the engine session
    /// (catalog, mutable tables, broker), and prepares the shared
    /// metrics registry and readiness probe.
    pub async fn new(config: JammiConfig) -> Result<Self, ServerError> {
        config
            .server
            .validate()
            .map_err(|e| ServerError::Config(e.to_string()))?;
        // Reject training timing that violates the worker invariants (heartbeat
        // margin / non-zero poll) at construction, before the train tier spawns
        // its worker.
        config
            .training
            .worker_intervals()
            .map_err(|e| ServerError::Config(e.to_string()))?;

        let flight_addr: SocketAddr = config.server.flight_listen.parse()?;
        let health_addr: SocketAddr = config.server.health_listen.parse()?;

        // Resolve the mounted tier set before constructing the engine: a config
        // that names an unknown tier or one whose feature is compiled out is a
        // startup error, not a silent degrade.
        let tiers = TierSet::from_config(&config.server.services)?;

        // `open` (not `new`) registers the `annotate` query UDTF on the engine's
        // DataFusion context — the Flight SQL surface needs it. It already returns
        // an `Arc<InferenceSession>`.
        let session = InferenceSession::open(config).await?;
        let session_store = SessionStore::new();
        let metrics = Arc::new(MetricsRegistry::new()?);
        let readiness = Arc::new(ReadinessProbe::new(Arc::new(CatalogPingProbe::new(
            Arc::clone(&session),
        ))));

        Ok(Self {
            flight_addr,
            health_addr,
            session,
            session_store,
            metrics,
            readiness,
            tiers,
        })
    }

    /// Shared handle to the metrics registry. Test fixtures and the
    /// gRPC services use this to increment counters.
    pub fn metrics(&self) -> Arc<MetricsRegistry> {
        Arc::clone(&self.metrics)
    }

    /// Shared handle to the engine session. Useful in tests that want
    /// to publish to a topic or read a mutable table while the server
    /// is running.
    pub fn session(&self) -> Arc<InferenceSession> {
        Arc::clone(&self.session)
    }

    /// Override the readiness probe. Used by integration tests to make
    /// `/readyz` deterministically return 503.
    pub fn with_readiness(mut self, readiness: Arc<ReadinessProbe>) -> Self {
        self.readiness = readiness;
        self
    }

    /// Bind both listeners eagerly and return a [`BoundServer`] holding them
    /// live. The gRPC + Flight SQL surface and the HTTP side-channel are each
    /// bound here — so their ACTUAL addresses are known (the real ports even
    /// when the config requested an ephemeral `:0`) before a single connection
    /// is served, and neither port is released before
    /// [`BoundServer::serve_with_shutdown`] serves on the same listeners. A
    /// caller reads the resolved ports off the returned handle
    /// ([`BoundServer::flight_addr`] / [`BoundServer::health_addr`]) while the
    /// listeners stay bound, with no observable release-then-rebind window.
    pub async fn bind(self) -> Result<BoundServer, ServerError> {
        let health_router = self.build_health_router();
        let health_listener = TcpListener::bind(self.health_addr).await?;
        let health_addr = health_listener.local_addr()?;
        let grpc = assemble_grpc_chain(self.build_grpc_chain())?.bind().await?;
        Ok(BoundServer {
            grpc,
            health_listener,
            health_addr,
            health_router,
        })
    }

    /// Drive the server until SIGINT / SIGTERM arrives. Both the HTTP
    /// side-channel and the gRPC surface drain in parallel; the call
    /// returns when both have stopped accepting new connections and
    /// finished serving in-flight requests.
    pub async fn run(self) -> Result<(), ServerError> {
        self.bind()
            .await?
            .serve_with_shutdown(shutdown_signal())
            .await
    }

    /// Variant of [`Self::run`] that accepts a caller-provided
    /// shutdown future. Tests use this to drive deterministic
    /// teardown.
    pub async fn run_with_shutdown(
        self,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> Result<(), ServerError> {
        self.bind().await?.serve_with_shutdown(shutdown).await
    }

    fn build_health_router(&self) -> Router {
        // Two sub-routers keep the State types separated — Axum requires
        // every route in a Router to share the same State type, so the
        // readiness handler and the metrics handler are merged here
        // after each one's State is applied.
        let readyz = Router::new()
            .route("/readyz", get(health::readyz))
            .with_state(Arc::clone(&self.readiness));
        let metrics = Router::new()
            .route("/metrics", get(health::metrics))
            .with_state(Arc::clone(&self.metrics));
        Router::new()
            .route("/healthz", get(health::healthz))
            .merge(readyz)
            .merge(metrics)
            .fallback(fallback_handler)
    }

    /// Assemble the engine's [`GrpcChain`] from this server's config and engine
    /// session. Derives the event-tier trigger handles (only when the event
    /// tier is mounted) and the engine-default `jammi-session-id` tenant
    /// resolver — the OSS-cooperative multitenancy binder.
    fn build_grpc_chain(&self) -> GrpcChain {
        // The event tier (`TriggerService`) is mounted only when the deployment
        // selected it; the handles are derived from the same engine session.
        let trigger = self
            .tiers
            .contains(ServiceTier::Event)
            .then(|| crate::TriggerHandles {
                topic_repo: self.session.topic_repo(),
                publisher: self.session.publisher(),
                subscriber: self.session.subscriber(),
            });
        GrpcChain {
            addr: self.flight_addr,
            flight_ctx: self.session.context().clone(),
            flight_binding: self.session.tenant_binding_arc(),
            store: self.session_store.clone(),
            trigger,
            engine: Some(Arc::clone(&self.session)),
            tiers: self.tiers.clone(),
            metrics: Arc::clone(&self.metrics),
            // The OSS binary binds tenants with the engine-default
            // `jammi-session-id` resolver — the OSS-cooperative multitenancy
            // behavior, now a first-class resolver rather than an implicit
            // interceptor.
            tenant_resolver: SessionIdTenantResolver::arc(self.session_store.clone()),
        }
    }
}

/// An [`OssServer`] with both listeners bound and held live — the eager-bind
/// half of the serve seam. Returned by [`OssServer::bind`]. The gRPC + Flight
/// SQL surface and the HTTP side-channel are already listening:
/// [`Self::flight_addr`] / [`Self::health_addr`] report the ACTUAL bound
/// addresses (the real ports even when the config requested `:0`), and neither
/// port is released before [`Self::serve_with_shutdown`] serves on the very same
/// listeners — there is no release-then-rebind window a concurrent binder could
/// slip into.
pub struct BoundServer {
    grpc: BoundChain,
    health_listener: TcpListener,
    health_addr: SocketAddr,
    health_router: Router,
}

impl BoundServer {
    /// The ACTUAL address the gRPC + Flight SQL surface is bound to. When the
    /// config requested `:0`, this is the ephemeral port the kernel assigned —
    /// resolved at [`OssServer::bind`] and held.
    pub fn flight_addr(&self) -> SocketAddr {
        self.grpc.addr()
    }

    /// The ACTUAL address the HTTP side-channel (`/healthz`, `/readyz`,
    /// `/metrics`) is bound to. When the config requested `:0`, this is the
    /// ephemeral port the kernel assigned.
    pub fn health_addr(&self) -> SocketAddr {
        self.health_addr
    }

    /// Serve both halves on the already-bound listeners until `shutdown`
    /// resolves. The HTTP side-channel and the gRPC surface drain in parallel;
    /// the call returns when both have stopped accepting new connections and
    /// finished serving in-flight requests.
    pub async fn serve_with_shutdown(
        self,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> Result<(), ServerError> {
        // Fan out one shutdown signal to both servers. A `broadcast`
        // channel gives every subscriber an independent receiver and
        // does not require the futures to share lifetimes.
        let (shutdown_tx, _) = broadcast::channel::<()>(1);
        let mut shutdown_health_rx = shutdown_tx.subscribe();
        let mut shutdown_grpc_rx = shutdown_tx.subscribe();
        let shutdown_tx_for_signal = shutdown_tx.clone();
        tokio::spawn(async move {
            shutdown.await;
            // Receivers may already be gone if the servers errored
            // first; either way the broadcast send is best-effort.
            let _ = shutdown_tx_for_signal.send(());
        });

        let BoundServer {
            grpc,
            health_listener,
            health_addr,
            health_router,
        } = self;
        tracing::info!(
            address = %health_addr,
            "HTTP side-channel listening (/healthz, /readyz, /metrics)"
        );

        let health_task = tokio::spawn(async move {
            axum::serve(health_listener, health_router)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_health_rx.recv().await;
                })
                .await
                .map_err(ServerError::from)
        });

        // Run both halves to completion. If either errors out we still
        // wait for the other to drain — abandoning a running server
        // mid-shutdown corrupts in-flight connections.
        let grpc_result = grpc
            .serve_with_shutdown(async move {
                let _ = shutdown_grpc_rx.recv().await;
            })
            .await;
        if grpc_result.is_err() {
            let _ = shutdown_tx.send(());
        }
        let health_result = match health_task.await {
            Ok(r) => r,
            Err(join_err) => Err(ServerError::Io(std::io::Error::other(join_err.to_string()))),
        };

        grpc_result.and(health_result)
    }

    /// Serve both halves until SIGINT / SIGTERM arrives. The binary entry
    /// point ([`OssServer::run`] and `main`) reaches graceful shutdown through
    /// this.
    pub async fn serve(self) -> Result<(), ServerError> {
        self.serve_with_shutdown(shutdown_signal()).await
    }
}

/// Everything [`serve_grpc_chain`] needs to mount the Tonic chain: the bind
/// address, the Flight SQL context + tenant binding, the shared session store,
/// the optional trigger handles and engine session, and the resolved tier set.
///
/// Grouped into one options object (rather than a long positional argument list)
/// so callers name what they pass and the mount surface has one place to grow.
/// `OssServer` builds this from the engine session; test fixtures construct it
/// directly.
pub struct GrpcChain {
    /// Bind address for the combined gRPC + Flight SQL surface.
    pub addr: SocketAddr,
    /// Flight SQL session context.
    pub flight_ctx: SessionContext,
    /// Tenant binding the Flight SQL provider mutates per request.
    pub flight_binding: jammi_db::tenant_scope::TenantBinding,
    /// Session store shared between the `CatalogService` tenant trio (writers)
    /// and the engine-default `SessionIdTenantResolver` (reader).
    pub store: SessionStore,
    /// Trigger handles — `Some` iff the event tier is mounted.
    pub trigger: Option<crate::TriggerHandles>,
    /// Engine session backing the engine-layer services — `None` for the
    /// transport-only fixtures.
    pub engine: Option<Arc<InferenceSession>>,
    /// The tier set this chain mounts and advertises over `GetServerInfo`.
    pub tiers: TierSet,
    /// Shared metrics registry. The whole-server [`MetricsLayer`] holds it and
    /// drives the substrate counters / latency histogram from the request path;
    /// the Axum `/metrics` route reads the same registry to scrape it.
    pub metrics: Arc<MetricsRegistry>,
    /// The one tenant-binding resolver. Every engine service (and Flight SQL)
    /// binds its request's tenant through THIS resolver via the async
    /// [`crate::tenant_resolver_layer::TenantResolverLayer`] — the single binder,
    /// no interceptor path. Production and the OSS binary pass
    /// [`SessionIdTenantResolver::arc`] (the OSS-cooperative `jammi-session-id`
    /// default); a downstream composing the seam passes its own authenticating
    /// resolver to unify the composability seam with its BYO-auth seam. Only the
    /// services `assemble_grpc_chain` itself builds are bound by it — downstream
    /// services later [`AssembledChain::mount`]ed are NOT wrapped.
    pub tenant_resolver: Arc<dyn TenantResolver>,
}

/// The engine's fully-assembled gRPC chain, ready for a downstream to mount
/// additional services onto before serving.
///
/// Holds a [`tonic::service::Routes`] with the engine's services pre-added
/// (Flight SQL + `CatalogService` + the tier/engine services, including
/// `AuditService`) and any resource whose lifetime must span the serve loop (the
/// embedded training worker). A downstream chains [`Self::mount`] (un-gated) or
/// [`Self::mount_tenant_scoped`] (bound by the engine's single tenant resolver)
/// to add its own services beside the engine's, then [`Self::serve`]s — or
/// splits to compose one listener of its own: [`Self::into_layered_axum_router`]
/// is the safe default (the engine's transport stack pre-applied, ready for
/// [`axum::serve()`]), and
/// [`Self::into_axum_router`] is the expert, layer-free split for nesting under a
/// listener that already frames gRPC-web.
///
/// The transport layer stack (`accept_http1` + `MetricsLayer` +
/// `GrpcWebTrailersLayer` + `GrpcWebLayer`) is applied by [`Self::serve`], not
/// baked into the routes — see that method, [`Self::into_layered_axum_router`],
/// and [`Self::into_axum_router`] for the seam contract each path honours.
pub struct AssembledChain {
    addr: SocketAddr,
    routes: tonic::service::Routes,
    mounted: Vec<String>,
    // The metrics handle the `MetricsLayer` needs at serve time. Carried forward
    // because the layer stack is deferred to `serve` — the outermost layer
    // observes every request by method path.
    metrics: Arc<MetricsRegistry>,
    // The SAME `TenantResolverLayer` (holding the same `Arc<dyn TenantResolver>`)
    // that wraps every engine service in `assemble_grpc_chain`. Retained
    // (`#[derive(Clone)]`, a cheap `Arc` clone) so `mount_tenant_scoped` can wrap
    // a downstream service through it too — the single-binder invariant holds:
    // there is still exactly ONE resolver, never a second one forked off here.
    tenant_resolver_layer: TenantResolverLayer,
    // The embedded training worker the `train` tier owns, held RAII for the serve
    // loop. Owned by the chain (not the assemble frame) so it survives the
    // assemble→serve split; `serve` keeps it alive across the serve future and
    // `into_axum_router` hands it onward in [`ChainParts`]. `#[cfg]`-gated so a
    // serve-only build carries no worker field.
    #[cfg(feature = "train")]
    _train_worker: Option<jammi_ai::fine_tune::worker::EmbeddedWorker>,
}

/// The non-routing remainder of an [`AssembledChain`] after
/// [`AssembledChain::into_axum_router`] splits the routes off: the resolved bind
/// address, the mounted-service ledger (for the downstream's startup log), the
/// engine metrics handle (so a single-listener downstream can re-apply the
/// engine's [`MetricsLayer`] on its own listener), and the training-worker guard
/// the downstream must keep alive for the lifetime of its own serve loop.
pub struct ChainParts {
    pub addr: SocketAddr,
    pub mounted: Vec<String>,
    pub metrics: Arc<MetricsRegistry>,
    /// The embedded training worker guard. The downstream MUST hold this for the
    /// lifetime of its serve loop — dropping it stops the worker and submitted
    /// jobs stop running.
    #[cfg(feature = "train")]
    pub train_worker: Option<jammi_ai::fine_tune::worker::EmbeddedWorker>,
}

impl AssembledChain {
    /// Mount a downstream service beside the engine's, **un-gated by the engine's
    /// tenant resolver**. Delegates [`tonic::service::Routes::add_service`] (by
    /// value, chainable) and records the service's `NamedService::NAME` in the
    /// mounted ledger for the startup tracing line — so the ledger cannot drift
    /// from what is actually mounted. Generic: the engine names no consumer. The
    /// service inherits the transport layer stack [`Self::serve`] applies,
    /// exactly as the engine's own services do — but NOT the tenant-binding
    /// layer, so no `SessionTenant` extension is bound on its requests.
    ///
    /// Reach for this for a service that must run BEFORE a tenant is known — a
    /// pre-auth handshake (e.g. a login/bootstrap verb) that a caller reaches
    /// without a resolvable credential yet. Use [`Self::mount_tenant_scoped`]
    /// instead for a service that wants the engine's resolved tenant bound
    /// uniformly.
    pub fn mount<S>(mut self, svc: S) -> Self
    where
        S: tonic::codegen::Service<
                tonic::codegen::http::Request<tonic::body::Body>,
                Error = std::convert::Infallible,
            > + tonic::server::NamedService
            + Clone
            + Send
            + Sync
            + 'static,
        S::Response: axum::response::IntoResponse,
        S::Future: Send + 'static,
    {
        self.mounted.push(S::NAME.to_string());
        self.routes = self.routes.add_service(svc);
        self
    }

    /// Mount a downstream service beside the engine's, wrapped by the engine's
    /// SAME single [`TenantResolverLayer`] every engine service is wrapped with
    /// (retained on `self` from [`assemble_grpc_chain`] — there is still exactly
    /// ONE `Arc<dyn TenantResolver>`, never a second resolver forked off here).
    /// Per request, the wrapper resolves the tenant BEFORE the service's handler
    /// runs: `Ok(scope)` binds the `SessionTenant` extension the handler reads
    /// ([`TenantScope::Tenant`](crate::grpc::session::TenantScope::Tenant) →
    /// `Some`, [`TenantScope::Global`](crate::grpc::session::TenantScope::Global)
    /// → `None`) and calls it; `Err(status)` returns that status and the handler
    /// NEVER runs — see [`crate::tenant_resolver_layer`] for the full per-request
    /// contract. The wrapper forwards the inner service's `NamedService::NAME`,
    /// so the mounted ledger and the proto route both stay keyed to the
    /// service's own name, exactly as [`Self::mount`] records it.
    ///
    /// Reach for this so a downstream service gets `SessionTenant` bound
    /// uniformly with every engine service, dropping its own per-handler
    /// resolve. Use plain [`Self::mount`] instead for a service that must stay
    /// un-gated (e.g. a pre-auth login/bootstrap handshake reached before any
    /// tenant is known).
    ///
    /// **Do NOT** mount a service here that already binds its own
    /// `SessionTenant` internally (or resolves tenant identity some other way)
    /// — the two binds would race and the later one silently wins
    /// (last-writer-wins on the request extension), which is exactly the
    /// double-bind the engine's single-binder invariant exists to prevent. Wrap
    /// a service with this method at most once, and never alongside a
    /// self-resolving service.
    ///
    /// PRE-SPLIT ONLY: this method lives on [`AssembledChain`], before
    /// [`Self::into_axum_router`] / [`Self::into_layered_axum_router`] split the
    /// routes off. [`ChainParts`] does NOT carry the layer forward — a
    /// downstream that splits to compose its own listener re-applies the
    /// (already `pub`) [`TenantResolverLayer`] itself via [`tower::Layer::layer`]
    /// before nesting its service, exactly as this method does internally.
    pub fn mount_tenant_scoped<S, ResBody>(mut self, svc: S) -> Self
    where
        S: tonic::codegen::Service<
                tonic::codegen::http::Request<tonic::body::Body>,
                Response = tonic::codegen::http::Response<ResBody>,
                Error = std::convert::Infallible,
            > + tonic::server::NamedService
            + Clone
            + Send
            + Sync
            + 'static,
        S::Future: Send + 'static,
        ResBody: Default + 'static,
        tonic::codegen::http::Response<ResBody>: axum::response::IntoResponse,
    {
        let scoped = self.tenant_resolver_layer.clone().layer(svc);
        self.mounted.push(S::NAME.to_string());
        self.routes = self.routes.add_service(scoped);
        self
    }

    /// The bind address the engine resolved from config. The downstream serves here.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// The ledger of mounted service names, in mount order (engine's first, then
    /// any the downstream added via [`Self::mount`]). Read for a startup log; the
    /// ledger cannot drift from what is actually on the routes.
    pub fn mounted(&self) -> &[String] {
        &self.mounted
    }

    /// Bind the gRPC + Flight SQL listener eagerly and return a [`BoundChain`]
    /// holding it live. The listener is opened HERE, so [`BoundChain::addr`]
    /// reports the ACTUAL bound address — the real port even when the chain was
    /// assembled at an ephemeral `:0` — and the port stays held with no release
    /// window until [`BoundChain::serve_with_shutdown`] serves on the very same
    /// listener. A caller that needs the resolved port before serving (a test
    /// harness building a client, a downstream logging its startup address)
    /// reads it off the bound handle rather than pre-binding-and-dropping a
    /// throwaway listener.
    ///
    /// `nodelay` is set to match the [`tonic::transport::Server`] default the
    /// addr-based serve path applies, so the served connections behave
    /// identically to a chain served straight from a configured address.
    pub async fn bind(self) -> Result<BoundChain, ServerError> {
        let listener = TcpListener::bind(self.addr).await?;
        let incoming = TcpIncoming::from(listener).with_nodelay(Some(true));
        let addr = incoming.local_addr()?;
        Ok(BoundChain {
            incoming,
            addr,
            routes: self.routes,
            mounted: self.mounted,
            metrics: self.metrics,
            #[cfg(feature = "train")]
            _train_worker: self._train_worker,
        })
    }

    /// Serve the assembled chain (engine core + any downstream-mounted services)
    /// until `shutdown` resolves. A thin composition of [`Self::bind`] +
    /// [`BoundChain::serve_with_shutdown`] — binds the listener, then serves on
    /// it; the training-worker guard stays alive for the whole serve loop. The
    /// transport layer stack is applied by [`BoundChain::serve_with_shutdown`].
    pub async fn serve(
        self,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> Result<(), ServerError> {
        self.bind().await?.serve_with_shutdown(shutdown).await
    }

    /// Split into a plain, **LAYER-FREE** [`axum::Router`] (via
    /// [`tonic::service::Routes::into_axum_router`]) plus the [`ChainParts`]
    /// remainder — the EXPERT split, for a downstream that nests the engine's
    /// gRPC routes UNDER a listener that ALREADY carries its own gRPC-web layer
    /// stack. Most single-listener consumers want [`Self::into_layered_axum_router`]
    /// instead (the safe default — see below).
    ///
    /// SEAM CONTRACT: the returned router carries NO transport layers. The
    /// gRPC-web framing + trailer-repair + metrics layers are applied by
    /// [`Self::serve`] on the serve path, NOT baked into the routes. A downstream
    /// that serves this router on its OWN listener must therefore re-apply the
    /// full stack itself — [`GrpcWebLayer`] + [`GrpcWebTrailersLayer`] + the
    /// engine's [`MetricsLayer`] (via [`ChainParts::metrics`]) — or gRPC-web
    /// clients break: a trailers-only error response would miss the in-body
    /// trailer frame.
    ///
    /// PATH-SPECIFIC LAYERING: `accept_http1(true)` is a
    /// [`tonic::transport::Server`] builder method and applies ONLY on the
    /// [`Self::serve`] path — there is no `accept_http1` to call on the axum
    /// path; HTTP/1 is implicit in [`axum::serve()`]. On axum, re-apply the layers
    /// with `Router::layer` in inner→outer call order (axum runs the LAST
    /// `.layer` call as the outermost service, the inverse of the tonic builder),
    /// i.e. `.layer(GrpcWebLayer::new()).layer(GrpcWebTrailersLayer::new())
    /// .layer(MetricsLayer::new(metrics))`. This is exactly what
    /// [`Self::into_layered_axum_router`] does for you — reach for the layer-free
    /// split only when your outer listener already frames gRPC-web (re-applying
    /// here would double-frame).
    ///
    /// The router also carries a gRPC `unimplemented` fallback (from
    /// `Routes`' default), so a composing consumer must nest it under a path
    /// prefix or reconcile its own fallback, NOT blind-`.merge()` it.
    ///
    /// The downstream must hold [`ChainParts`] (specifically its training-worker
    /// guard) alive for the lifetime of its own serve loop.
    pub fn into_axum_router(self) -> (axum::Router, ChainParts) {
        let router = self.routes.into_axum_router();
        let parts = ChainParts {
            addr: self.addr,
            mounted: self.mounted,
            metrics: self.metrics,
            #[cfg(feature = "train")]
            train_worker: self._train_worker,
        };
        (router, parts)
    }

    /// Split into a **layered** [`axum::Router`] plus the [`ChainParts`]
    /// remainder — the SAFE DEFAULT for a downstream that composes ONE listener
    /// of its own. The returned router is ready to hand to [`axum::serve()`]
    /// DIRECTLY: it carries the engine's full transport contract, so the consumer
    /// re-applies nothing.
    ///
    /// CANONICAL LAYER STACK: this applies the SAME stack [`Self::serve`] applies
    /// — the whole-server [`MetricsLayer`] (outermost, observing every method
    /// path) wrapping [`GrpcWebTrailersLayer`] (the trailers-only error repair)
    /// wrapping [`GrpcWebLayer`] (gRPC-web framing) wrapping the routes. axum runs
    /// the LAST `.layer` call as the OUTERMOST service — the inverse of the tonic
    /// [`tonic::transport::Server`] builder, where the FIRST `.layer` is
    /// outermost — so the calls are ordered inner→outer here to land the exact
    /// same outermost→innermost stack `serve` builds. `accept_http1` has no axum
    /// analogue: HTTP/1 is implicit in [`axum::serve()`].
    ///
    /// ERGONOMIC GUARANTEE: the returned value is a plain `axum::Router` (state
    /// `()`, request body [`axum::body::Body`]) that [`axum::serve()`] accepts with
    /// no further ceremony. The layer stack rewrites the response body type; that
    /// normalization is resolved INTERNALLY (the layered routes are re-nested
    /// under a fresh [`axum::Router`]), so the consumer needs no
    /// `Router::<()>::new().merge(...)` re-nest of its own.
    ///
    /// Prefer this over [`Self::into_axum_router`] unless you are nesting under a
    /// listener that ALREADY frames gRPC-web — that expert path is layer-free
    /// precisely so it does not double-frame in that case.
    ///
    /// The downstream must hold [`ChainParts`] (specifically its training-worker
    /// guard) alive for the lifetime of its own serve loop.
    pub fn into_layered_axum_router(self) -> (axum::Router, ChainParts) {
        // The `MetricsLayer` holds a clone; the original moves into `ChainParts`
        // so the downstream can still scrape the same registry from its own
        // `/metrics` route.
        let metrics = Arc::clone(&self.metrics);
        // Apply the canonical stack in axum's inner→outer call order. axum runs
        // the last `.layer` as the outermost service, so ordering the calls
        // GrpcWebLayer → GrpcWebTrailersLayer → MetricsLayer reproduces `serve`'s
        // outermost→innermost stack: Metrics → GrpcWebTrailers → GrpcWebLayer →
        // routes.
        let layered = self
            .routes
            .into_axum_router()
            .layer(GrpcWebLayer::new())
            .layer(GrpcWebTrailersLayer::new())
            .layer(MetricsLayer::new(metrics));
        // Re-nest the layered routes under a fresh `Router` so the returned type
        // is a plain `axum::Router` whose request body is `axum::body::Body` —
        // the layer stack's response-body rewrite is absorbed here, and
        // `axum::serve` accepts the result directly with no consumer-side merge.
        let router = axum::Router::new().merge(layered);
        let parts = ChainParts {
            addr: self.addr,
            mounted: self.mounted,
            metrics: self.metrics,
            #[cfg(feature = "train")]
            train_worker: self._train_worker,
        };
        (router, parts)
    }
}

/// An [`AssembledChain`] whose gRPC + Flight SQL listener is bound and held
/// live, ready to serve. Returned by [`AssembledChain::bind`]. The listener is
/// open from bind through serve, so [`Self::addr`] reports the ACTUAL bound
/// address (the real port even for a `:0` assembly) and the port is never
/// observably free between resolving it and serving on it.
pub struct BoundChain {
    // The bound listener, wrapped as tonic's incoming stream with the same
    // `nodelay` the addr-based serve path applies. Held from bind through serve
    // — this is the whole point: zero release-then-rebind window.
    incoming: TcpIncoming,
    addr: SocketAddr,
    // The layer-free routes, carried forward so the transport layer stack is
    // still applied at serve time (G1) rather than baked in at bind.
    routes: tonic::service::Routes,
    mounted: Vec<String>,
    metrics: Arc<MetricsRegistry>,
    // The embedded training worker guard, held RAII across the serve loop — its
    // lifetime spans bind → serve, exactly as it did on `AssembledChain`.
    #[cfg(feature = "train")]
    _train_worker: Option<jammi_ai::fine_tune::worker::EmbeddedWorker>,
}

impl BoundChain {
    /// The ACTUAL address the listener is bound to. For a chain assembled at
    /// `:0`, this is the ephemeral port the kernel assigned — resolved at
    /// [`AssembledChain::bind`] and held live until serve.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// The ledger of mounted service names, in mount order. Read for a startup
    /// log; carried through the bind unchanged.
    pub fn mounted(&self) -> &[String] {
        &self.mounted
    }

    /// Serve the bound chain on its already-open listener until `shutdown`
    /// resolves. Consumes `self`, keeping the training-worker guard alive for
    /// the whole serve loop.
    ///
    /// The transport layers apply HERE, in this order: `accept_http1(true)` then
    /// `MetricsLayer` (outermost — observes every request by method path before
    /// routing) then `GrpcWebTrailersLayer` (wraps `GrpcWebLayer`, repairing the
    /// trailers-only error response into the in-body trailer frame a gRPC-web
    /// client requires) then `GrpcWebLayer`. Every service mounted via
    /// [`AssembledChain::mount`], engine or downstream, inherits gRPC-web framing
    /// + trailer repair with no per-service opt-in.
    pub async fn serve_with_shutdown(
        self,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> Result<(), ServerError> {
        tracing::info!(
            "gRPC chain ({}) listening on {}",
            self.mounted.join(" + "),
            self.addr
        );
        // The layer stack is deferred to here (G1): holding the post-layer
        // `Router<L>` would leak the concrete `Stack<…>` layer types into
        // `BoundChain`. `Routes` is the layer-free accumulation point;
        // `add_routes` attaches it behind the stack at serve time, then serves
        // on the pre-bound listener via `serve_with_incoming_shutdown`.
        let mut server = Server::builder()
            .accept_http1(true)
            .layer(MetricsLayer::new(self.metrics))
            .layer(GrpcWebTrailersLayer::new())
            .layer(GrpcWebLayer::new());
        server
            .add_routes(self.routes)
            .serve_with_incoming_shutdown(self.incoming, shutdown)
            .await
            .map_err(ServerError::from)
        // `self._train_worker` (train build) is dropped here, after the serve
        // future resolves — its RAII lifetime spans the whole serve loop.
    }
}

/// Assemble the engine's gRPC chain from `chain` **without serving it**, so a
/// downstream can [`AssembledChain::mount`] additional services onto the
/// engine's fully-assembled core chain before serving. This is the composability
/// seam.
///
/// **Always mounted** (the core tier + the Flight SQL transport): Flight SQL and
/// the control-plane `CatalogService` (its engine-free tenant trio +
/// `GetServerInfo` answer even when no engine is mounted; its catalog /
/// lifecycle verbs are backed by `engine` when present). When `engine` is
/// `Some`, the core data-plane services also mount: `EmbeddingService`,
/// `InferenceService`, `PipelineService`, `AuditService`. These are the
/// serve-path primitives every deployment needs.
///
/// **Mounted by tier** (only when `tiers` selected them):
/// - `EvalService` ← [`ServiceTier::Eval`]
/// - `TrainingService` ← [`ServiceTier::Train`] (and only when the `train`
///   feature is compiled in — the mount code itself is `#[cfg]`-gated)
/// - `TriggerService` ← [`ServiceTier::Event`], driven by `trigger` being
///   `Some` (the caller derives the handles iff the event tier is mounted)
///
/// `engine` and `trigger` are `Option` so the gRPC-Web / control-plane-only
/// fixtures (which construct no `InferenceSession`) can mount just the
/// transport + core handshake. The `tiers` argument is what the
/// `CatalogService.GetServerInfo` handshake advertises, so it must agree with
/// what is actually mounted — the caller is responsible for that agreement
/// (production goes through [`OssServer`], which derives both from one config).
///
/// The engine mounts NO `LifecycleService` — the `jammi.v1.lifecycle` contract
/// is answered by a platform server, not the OSS engine; an OSS server answers
/// `UNIMPLEMENTED` for those verbs.
pub fn assemble_grpc_chain(chain: GrpcChain) -> Result<AssembledChain, ServerError> {
    let GrpcChain {
        addr,
        flight_ctx,
        flight_binding,
        store,
        trigger,
        engine,
        tiers,
        metrics,
        tenant_resolver,
    } = chain;

    // Flight SQL — MUST-FIX 2: cover the `db.sql` lane through the SAME resolver
    // as the gRPC plane. The provider resolves each query's scope and binds it,
    // closing the cross-transport bypass where an authenticated gRPC plane would
    // still let Flight bind from an unauthenticated header (#220).
    let provider = TenantBoundProvider::new(
        flight_ctx.state(),
        flight_binding,
        Arc::clone(&tenant_resolver),
    );
    let flight = FlightSqlService::new_with_provider(Box::new(provider));
    let flight_svc = FlightServiceServer::new(flight);

    // The single binder. One `TenantResolverLayer` (holding the one resolver)
    // wraps every engine service uniformly — no branch, no separate interceptor,
    // so nothing can double-bind or clobber the resolved tenant. MUST-FIX 1 — the
    // wrapping applies only to the services THIS function builds; downstream
    // services mounted via `AssembledChain::mount` (a platform's own pre-auth
    // `Login`/`Bootstrap`, etc.) stay un-gated by the engine resolver.
    let resolver_layer = TenantResolverLayer::new(tenant_resolver);

    // Bind one engine service onto `routes` under the resolver layer. `$server`
    // is the bare `*ServiceServer::new(inner)`; the layer forwards its
    // `NamedService::NAME` so tonic routing keeps it.
    macro_rules! mount_engine {
        ($routes:expr, $mounted:expr, $name:literal, $server:expr) => {{
            $routes = $routes.add_service(resolver_layer.layer($server));
            $mounted.push($name.to_string());
        }};
    }

    // Accumulate the services layer-free on a `tonic::service::Routes`. The
    // transport layer stack is deferred to `AssembledChain::serve` (G1): holding
    // the post-`add_service` `Router<L>` would leak the concrete layer-stack type
    // into the seam and cannot grow in place (its `add_service` is by-value with
    // no `Default`). `Routes` is the composition point tonic provides for exactly
    // this — every service mounted onto it, engine or downstream, then inherits
    // the gRPC-web framing + trailer repair the serve path applies. Flight SQL is
    // NOT wrapped by the layer: its `db.sql` lane binds inside the
    // `TenantBoundProvider` (threaded with the same resolver above).
    let mut routes = tonic::service::Routes::new(flight_svc);
    let mut mounted = vec!["Flight SQL".to_string()];

    // The control plane: one `CatalogService` on the always-present core tier.
    // Its engine-free verbs (the tenant trio + `GetServerInfo`) ride the
    // `SessionStore` + `TierSet`, so it mounts even on an engine-light
    // deployment; its catalog / lifecycle verbs delegate to the shared engine
    // when one is present (`engine.clone()` here, with the original moved into
    // the engine-services block below).
    mount_engine!(
        routes,
        mounted,
        "CatalogService",
        CatalogServiceServer::new(CatalogServer::new(store, tiers.clone(), engine.clone()))
    );

    // Event tier: TriggerService. Driven by the caller having supplied handles
    // (it does so iff the event tier is mounted).
    if let Some(handles) = trigger {
        mount_engine!(
            routes,
            mounted,
            "TriggerService",
            TriggerServiceServer::new(TriggerServer::new(
                handles.topic_repo,
                handles.publisher,
                handles.subscriber,
            ))
        );
    }

    // The embedded training worker the `train` tier owns. Moved into the returned
    // `AssembledChain` so it outlives the assemble frame and spans the serve loop
    // (RAII). A serve-only build never sets it.
    #[cfg(feature = "train")]
    let mut train_worker: Option<jammi_ai::fine_tune::worker::EmbeddedWorker> = None;

    if let Some(session) = engine {
        // Core tier engine services: always mounted when an engine is present.
        mount_engine!(
            routes,
            mounted,
            "EmbeddingService",
            EmbeddingServiceServer::new(EmbeddingServer::new(Arc::clone(&session)))
        );
        mount_engine!(
            routes,
            mounted,
            "InferenceService",
            InferenceServiceServer::new(InferenceServer::new(Arc::clone(&session)))
        );
        mount_engine!(
            routes,
            mounted,
            "PipelineService",
            PipelineServiceServer::new(PipelineServer::new(Arc::clone(&session)))
        );
        mount_engine!(
            routes,
            mounted,
            "AuditService",
            AuditServiceServer::new(AuditServer::new(Arc::clone(&session)))
        );

        // Eval tier: EvalService.
        if tiers.contains(ServiceTier::Eval) {
            mount_engine!(
                routes,
                mounted,
                "EvalService",
                EvalServiceServer::new(EvalServer::new(Arc::clone(&session)))
            );
        }

        // Train tier: TrainingService (all three training kinds — fine-tune,
        // graph fine-tune, context-predictor). The mount code is `#[cfg]`-gated on
        // the `train` feature, so a serve-only build carries no training surface;
        // `TierSet::resolve` has already guaranteed the tier is not requested when
        // the feature is compiled out.
        #[cfg(feature = "train")]
        if tiers.contains(ServiceTier::Train) {
            // Start the worker that runs submitted jobs: a "GPU worker pool" is
            // just N processes claiming from the shared catalog, and the server
            // `train` tier runs one of them. `spawn` borrows `session` before it
            // is moved into `TrainingServer::new`; the worker is stored in
            // `AssembledChain` so it stops when the serve future resolves.
            train_worker = Some(jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(
                &session,
            )?);
            mount_engine!(
                routes,
                mounted,
                "TrainingService",
                TrainingServiceServer::new(TrainingServer::new(session))
            );
        }
    }

    Ok(AssembledChain {
        addr,
        routes,
        mounted,
        metrics,
        // The SAME layer `mount_engine!` wrapped every engine service with above
        // — retained so `AssembledChain::mount_tenant_scoped` can wrap a
        // downstream service through the identical single resolver.
        tenant_resolver_layer: resolver_layer,
        #[cfg(feature = "train")]
        _train_worker: train_worker,
    })
}

/// Build and serve the engine's gRPC chain on `chain.addr` in one call — the
/// OSS-only convenience for a caller serving straight from a configured address.
///
/// A thin composition of the seam: `assemble_grpc_chain(chain)?.serve(...)`,
/// which binds the listener eagerly ([`AssembledChain::bind`]) and serves on it.
/// A caller that needs the RESOLVED port before serving (an ephemeral `:0` bind)
/// drives [`assemble_grpc_chain`] → [`AssembledChain::bind`] →
/// [`BoundChain::serve_with_shutdown`] directly and reads the port off the bound
/// handle. Downstreams that mount their own services go through
/// [`assemble_grpc_chain`] + [`AssembledChain::mount`] instead.
pub async fn serve_grpc_chain(
    chain: GrpcChain,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), ServerError> {
    assemble_grpc_chain(chain)?.serve(shutdown).await
}

/// Install OS shutdown handlers and resolve when SIGINT or SIGTERM
/// arrives. Mirrors the existing `lib.rs` behaviour so the binary
/// shuts down on Ctrl+C and on `docker stop` (which sends SIGTERM).
async fn shutdown_signal() {
    let ctrl_c = async {
        match signal::ctrl_c().await {
            Ok(()) => {}
            Err(e) => tracing::error!("Failed to install Ctrl+C handler: {e}"),
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match signal::unix::signal(signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => {
                tracing::error!("Failed to install SIGTERM handler: {e}");
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }

    tracing::info!("Shutdown signal received, draining connections...");
}
