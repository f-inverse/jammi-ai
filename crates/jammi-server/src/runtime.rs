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
use tonic::transport::Server;
use tonic_web::GrpcWebLayer;

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
use crate::grpc::session::{SessionStore, TenantInterceptor};
#[cfg(feature = "train")]
use crate::grpc::training::TrainingServer;
use crate::grpc::trigger::TriggerServer;
use crate::grpc_web_trailers::GrpcWebTrailersLayer;
use crate::metrics_layer::MetricsLayer;
use crate::routes::health::{self, MetricsRegistry};
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
    /// configuration up front (parses both bind addresses, rejects
    /// matching health/flight ports), constructs the engine session
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

    /// Bind address the gRPC + Flight SQL surface listens on.
    pub fn flight_addr(&self) -> SocketAddr {
        self.flight_addr
    }

    /// Bind address the HTTP side-channel listens on.
    pub fn health_addr(&self) -> SocketAddr {
        self.health_addr
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

    /// Drive the server until SIGINT / SIGTERM arrives. Both the HTTP
    /// side-channel and the gRPC surface drain in parallel; the call
    /// returns when both have stopped accepting new connections and
    /// finished serving in-flight requests.
    pub async fn run(self) -> Result<(), ServerError> {
        self.run_with_shutdown(shutdown_signal()).await
    }

    /// Variant of [`Self::run`] that accepts a caller-provided
    /// shutdown future. Tests use this to drive deterministic
    /// teardown.
    pub async fn run_with_shutdown(
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

        let health_router = self.build_health_router();
        let health_listener = TcpListener::bind(self.health_addr).await?;
        tracing::info!(
            address = %self.health_addr,
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

        let grpc_future = self.build_and_serve_grpc(async move {
            let _ = shutdown_grpc_rx.recv().await;
        });

        // Run both halves to completion. If either errors out we still
        // wait for the other to drain — abandoning a running server
        // mid-shutdown corrupts in-flight connections.
        let grpc_result = grpc_future.await;
        if grpc_result.is_err() {
            let _ = shutdown_tx.send(());
        }
        let health_result = match health_task.await {
            Ok(r) => r,
            Err(join_err) => Err(ServerError::Io(std::io::Error::other(join_err.to_string()))),
        };

        grpc_result.and(health_result)
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

    async fn build_and_serve_grpc(
        &self,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> Result<(), ServerError> {
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
        serve_grpc_chain(
            GrpcChain {
                addr: self.flight_addr,
                flight_ctx: self.session.context().clone(),
                flight_binding: self.session.tenant_binding_arc(),
                store: self.session_store.clone(),
                trigger,
                engine: Some(Arc::clone(&self.session)),
                tiers: self.tiers.clone(),
                metrics: Arc::clone(&self.metrics),
            },
            shutdown,
        )
        .await
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
    /// Session store shared between every service via the tenant interceptor.
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
}

/// The engine's fully-assembled gRPC chain, ready for a downstream to mount
/// additional services onto before serving.
///
/// Holds a [`tonic::service::Routes`] with the engine's services pre-added
/// (Flight SQL + `CatalogService` + the tier/engine services, including
/// `AuditService`) and any resource whose lifetime must span the serve loop (the
/// embedded training worker). A downstream chains [`Self::mount`] to add its own
/// services beside the engine's, then [`Self::serve`]s — or splits via
/// [`Self::into_axum_router`] to compose one listener of its own.
///
/// The transport layer stack (`accept_http1` + `MetricsLayer` +
/// `GrpcWebTrailersLayer` + `GrpcWebLayer`) is applied by [`Self::serve`], not
/// baked into the routes — see that method and [`Self::into_axum_router`] for the
/// seam contract a single-listener consumer must honour.
pub struct AssembledChain {
    addr: SocketAddr,
    routes: tonic::service::Routes,
    mounted: Vec<String>,
    // The metrics handle the `MetricsLayer` needs at serve time. Carried forward
    // because the layer stack is deferred to `serve` — the outermost layer
    // observes every request by method path.
    metrics: Arc<MetricsRegistry>,
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
    /// Mount a downstream service beside the engine's. Delegates
    /// [`tonic::service::Routes::add_service`] (by value, chainable) and records
    /// the service's `NamedService::NAME` in the mounted ledger for the startup
    /// tracing line — so the ledger cannot drift from what is actually mounted.
    /// Generic: the engine names no consumer. The service inherits the transport
    /// layer stack [`Self::serve`] applies, exactly as the engine's own services
    /// do.
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

    /// Serve the assembled chain (engine core + any downstream-mounted services)
    /// until `shutdown` resolves. Consumes `self`, keeping the training-worker
    /// guard alive for the whole serve loop.
    ///
    /// The transport layers apply HERE, in this order: `accept_http1(true)` then
    /// `MetricsLayer` (outermost — observes every request by method path before
    /// routing) then `GrpcWebTrailersLayer` (wraps `GrpcWebLayer`, repairing the
    /// trailers-only error response into the in-body trailer frame a gRPC-web
    /// client requires) then `GrpcWebLayer`. Every service mounted via
    /// [`Self::mount`], engine or downstream, inherits gRPC-web framing + trailer
    /// repair with no per-service opt-in.
    pub async fn serve(
        self,
        shutdown: impl Future<Output = ()> + Send + 'static,
    ) -> Result<(), ServerError> {
        tracing::info!(
            "gRPC chain ({}) listening on {}",
            self.mounted.join(" + "),
            self.addr
        );
        // The layer stack is deferred to here (G1): `accept_http1` is a listener
        // property, and holding the post-layer `Router<L>` would leak the
        // concrete `Stack<…>` layer types into `AssembledChain`. `Routes` is the
        // layer-free accumulation point; `add_routes` attaches it behind the
        // stack at serve time.
        let mut server = Server::builder()
            .accept_http1(true)
            .layer(MetricsLayer::new(self.metrics))
            .layer(GrpcWebTrailersLayer::new())
            .layer(GrpcWebLayer::new());
        server
            .add_routes(self.routes)
            .serve_with_shutdown(self.addr, shutdown)
            .await
            .map_err(ServerError::from)
        // `self._train_worker` (train build) is dropped here, after the serve
        // future resolves — its RAII lifetime spans the whole serve loop.
    }

    /// Split into a plain [`axum::Router`] (via
    /// [`tonic::service::Routes::into_axum_router`]) plus the [`ChainParts`]
    /// remainder, for a downstream that composes ONE listener of its own (the
    /// gRPC routes nested beside its other HTTP routes).
    ///
    /// SEAM CONTRACT: the returned router is LAYER-FREE. The gRPC-web +
    /// trailer-repair layers are applied by [`Self::serve`], NOT baked into the
    /// routes — a downstream serving this router MUST re-apply
    /// `accept_http1(true)` + [`GrpcWebTrailersLayer`] + [`GrpcWebLayer`] (and the
    /// engine's [`MetricsLayer`], via [`ChainParts::metrics`]) at its own
    /// listener, or gRPC-web clients break: a trailers-only error response would
    /// miss the in-body trailer frame. `accept_http1` is a listener property, so
    /// the layers deliberately are NOT baked into the routes — baking them would
    /// double-frame a downstream that nests under its own grpc-web-layered
    /// listener.
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
    } = chain;
    let interceptor = TenantInterceptor::new(store.clone());

    let provider = TenantBoundProvider::new(flight_ctx.state(), flight_binding, store.clone());
    let flight = FlightSqlService::new_with_provider(Box::new(provider));
    let flight_svc = FlightServiceServer::new(flight);

    // The control plane: one `CatalogService` on the always-present core tier.
    // Its engine-free verbs (the tenant trio + `GetServerInfo`) ride the
    // `SessionStore` + `TierSet`, so it mounts even on an engine-light
    // deployment; its catalog / lifecycle verbs delegate to the shared engine
    // when one is present (`engine.clone()` here, with the original moved into
    // the engine-services block below).
    let catalog_svc = CatalogServiceServer::with_interceptor(
        CatalogServer::new(store, tiers.clone(), engine.clone()),
        interceptor.clone(),
    );

    // Accumulate the services layer-free on a `tonic::service::Routes`. The
    // transport layer stack is deferred to `AssembledChain::serve` (G1): holding
    // the post-`add_service` `Router<L>` would leak the concrete layer-stack type
    // into the seam and cannot grow in place (its `add_service` is by-value with
    // no `Default`). `Routes` is the composition point tonic provides for exactly
    // this — every service mounted onto it, engine or downstream, then inherits
    // the gRPC-web framing + trailer repair the serve path applies.
    let mut routes = tonic::service::Routes::new(flight_svc).add_service(catalog_svc);
    let mut mounted = vec!["Flight SQL".to_string(), "CatalogService".to_string()];

    // Event tier: TriggerService. Driven by the caller having supplied handles
    // (it does so iff the event tier is mounted).
    if let Some(handles) = trigger {
        let trigger_svc = TriggerServiceServer::with_interceptor(
            TriggerServer::new(handles.topic_repo, handles.publisher, handles.subscriber),
            interceptor.clone(),
        );
        routes = routes.add_service(trigger_svc);
        mounted.push("TriggerService".to_string());
    }

    // The embedded training worker the `train` tier owns. Moved into the returned
    // `AssembledChain` so it outlives the assemble frame and spans the serve loop
    // (RAII). A serve-only build never sets it.
    #[cfg(feature = "train")]
    let mut train_worker: Option<jammi_ai::fine_tune::worker::EmbeddedWorker> = None;

    if let Some(session) = engine {
        // Core tier engine services: always mounted when an engine is present.
        let embedding_svc = EmbeddingServiceServer::with_interceptor(
            EmbeddingServer::new(Arc::clone(&session)),
            interceptor.clone(),
        );
        routes = routes.add_service(embedding_svc);
        mounted.push("EmbeddingService".to_string());

        let inference_svc = InferenceServiceServer::with_interceptor(
            InferenceServer::new(Arc::clone(&session)),
            interceptor.clone(),
        );
        routes = routes.add_service(inference_svc);
        mounted.push("InferenceService".to_string());

        let pipeline_svc = PipelineServiceServer::with_interceptor(
            PipelineServer::new(Arc::clone(&session)),
            interceptor.clone(),
        );
        routes = routes.add_service(pipeline_svc);
        mounted.push("PipelineService".to_string());

        let audit_svc = AuditServiceServer::with_interceptor(
            AuditServer::new(Arc::clone(&session)),
            interceptor.clone(),
        );
        routes = routes.add_service(audit_svc);
        mounted.push("AuditService".to_string());

        // Eval tier: EvalService.
        if tiers.contains(ServiceTier::Eval) {
            let eval_svc = EvalServiceServer::with_interceptor(
                EvalServer::new(Arc::clone(&session)),
                interceptor.clone(),
            );
            routes = routes.add_service(eval_svc);
            mounted.push("EvalService".to_string());
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
            let training_svc =
                TrainingServiceServer::with_interceptor(TrainingServer::new(session), interceptor);
            routes = routes.add_service(training_svc);
            mounted.push("TrainingService".to_string());
        }
    }

    Ok(AssembledChain {
        addr,
        routes,
        mounted,
        metrics,
        #[cfg(feature = "train")]
        _train_worker: train_worker,
    })
}

/// Build and serve the engine's gRPC chain on `chain.addr` (the OSS-only path).
///
/// A thin composition of the D1 seam: `assemble_grpc_chain(chain)?.serve(...)`.
/// There is no parallel assembly logic to drift — every existing caller
/// ([`OssServer`]'s internal serve path, the test fixtures) keeps this exact
/// signature and behaviour. Downstreams that need to mount their own services go
/// through [`assemble_grpc_chain`] + [`AssembledChain::mount`] instead.
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
