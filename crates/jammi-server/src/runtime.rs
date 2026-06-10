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
}

/// Build and run the gRPC chain on `chain.addr`, mounting services per the
/// resolved [`TierSet`], and sharing the supplied `SessionStore` between every
/// service.
///
/// **Always mounted** (the core tier + the Flight SQL transport): Flight SQL and
/// the control-plane `CatalogService` (its engine-free tenant trio +
/// `GetServerInfo` answer even when no engine is mounted; its catalog /
/// lifecycle verbs are backed by `engine` when present). When `engine` is
/// `Some`, the core data-plane services also mount: `EmbeddingService`,
/// `InferenceService`, `AuditService`. These are the serve-path primitives every
/// deployment needs.
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
/// This is also the test-fixture entry-point. Both paths build the same Tonic
/// chain — there is no parallel API to drift.
pub async fn serve_grpc_chain(
    chain: GrpcChain,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), ServerError> {
    let GrpcChain {
        addr,
        flight_ctx,
        flight_binding,
        store,
        trigger,
        engine,
        tiers,
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

    // Layer order matters. `GrpcWebTrailersLayer` is added first, so in the
    // tower `ServiceBuilder` stack it wraps `GrpcWebLayer` and post-processes
    // the gRPC-Web-framed response: tonic encodes a unary handler error as a
    // trailers-only response (grpc-status in the HTTP headers, empty body) that
    // `GrpcWebLayer` passes through untouched, leaving a gRPC-Web client with no
    // in-body trailer frame to read. The repair layer rewrites that into the
    // in-body `0x80` trailer frame the gRPC-Web wire format requires. Raw gRPC
    // over HTTP/2 (the Rust `RemoteSession`) is unaffected — those responses are
    // not gRPC-Web and the layer skips them.
    let mut builder = Server::builder()
        .accept_http1(true)
        .layer(GrpcWebTrailersLayer::new())
        .layer(GrpcWebLayer::new())
        .add_service(flight_svc)
        .add_service(catalog_svc);

    let mut mounted = vec!["Flight SQL", "CatalogService"];

    // Event tier: TriggerService. Driven by the caller having supplied handles
    // (it does so iff the event tier is mounted).
    if let Some(handles) = trigger {
        let trigger_svc = TriggerServiceServer::with_interceptor(
            TriggerServer::new(handles.topic_repo, handles.publisher, handles.subscriber),
            interceptor.clone(),
        );
        builder = builder.add_service(trigger_svc);
        mounted.push("TriggerService");
    }

    // The embedded training worker the `train` tier owns. Held for the lifetime
    // of the serve loop (RAII): it is stopped when this future resolves on
    // shutdown, just like every other server-owned resource. A serve-only build
    // never sets it.
    #[cfg(feature = "train")]
    let mut _train_worker: Option<jammi_ai::fine_tune::worker::EmbeddedWorker> = None;

    if let Some(session) = engine {
        // Core tier engine services: always mounted when an engine is present.
        let embedding_svc = EmbeddingServiceServer::with_interceptor(
            EmbeddingServer::new(Arc::clone(&session)),
            interceptor.clone(),
        );
        builder = builder.add_service(embedding_svc);
        mounted.push("EmbeddingService");

        let inference_svc = InferenceServiceServer::with_interceptor(
            InferenceServer::new(Arc::clone(&session)),
            interceptor.clone(),
        );
        builder = builder.add_service(inference_svc);
        mounted.push("InferenceService");

        let pipeline_svc = PipelineServiceServer::with_interceptor(
            PipelineServer::new(Arc::clone(&session)),
            interceptor.clone(),
        );
        builder = builder.add_service(pipeline_svc);
        mounted.push("PipelineService");

        let audit_svc = AuditServiceServer::with_interceptor(
            AuditServer::new(Arc::clone(&session)),
            interceptor.clone(),
        );
        builder = builder.add_service(audit_svc);
        mounted.push("AuditService");

        // Eval tier: EvalService.
        if tiers.contains(ServiceTier::Eval) {
            let eval_svc = EvalServiceServer::with_interceptor(
                EvalServer::new(Arc::clone(&session)),
                interceptor.clone(),
            );
            builder = builder.add_service(eval_svc);
            mounted.push("EvalService");
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
            // `train` tier runs one of them. It stops when this future resolves
            // on shutdown (the guard drops with the function frame).
            _train_worker = Some(jammi_ai::fine_tune::worker::EmbeddedWorker::spawn(
                &session,
            )?);
            let training_svc =
                TrainingServiceServer::with_interceptor(TrainingServer::new(session), interceptor);
            builder = builder.add_service(training_svc);
            mounted.push("TrainingService");
        }
    }

    tracing::info!("gRPC chain ({}) listening on {addr}", mounted.join(" + "));
    builder
        .serve_with_shutdown(addr, shutdown)
        .await
        .map_err(ServerError::from)
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
