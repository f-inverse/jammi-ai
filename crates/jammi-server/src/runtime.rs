//! `OssServer` — the single orchestration entry-point for the OSS
//! `jammi-server` binary.
//!
//! One `OssServer` wires together:
//!
//! - the engine [`InferenceSession`] (catalog, mutable tables, broker)
//! - a [`SessionStore`] shared between Flight SQL and the gRPC services
//! - the Axum side-channel router (`/healthz`, `/readyz`, `/metrics`)
//! - one Tonic server hosting `FlightSqlService + SessionService +
//!   TriggerService` on a single port
//! - graceful shutdown wired to SIGINT/SIGTERM via a
//!   [`tokio::sync::broadcast`] so every component drains in parallel
//!
//! The structure is intentionally flat: no `runtime/` directory, no
//! per-component sub-modules. When a second binary materialises (e.g.
//! an enterprise edition) the same shape can be reused — the
//! orchestration is the engine of last resort and earns its keep by
//! being grep-able in one place.

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
use crate::grpc::embedding::EmbeddingServer;
use crate::grpc::proto::embedding::embedding_service_server::EmbeddingServiceServer;
use crate::grpc::proto::session::session_service_server::SessionServiceServer;
use crate::grpc::proto::trigger::trigger_service_server::TriggerServiceServer;
use crate::grpc::session::{SessionServer, SessionStore, TenantInterceptor};
use crate::grpc::trigger::TriggerServer;
use crate::routes::health::{self, MetricsRegistry};

/// Errors `OssServer::run` can surface to the binary's `main`.
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("config error: {0}")]
    Config(String),
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

        let flight_addr: SocketAddr = config.server.flight_listen.parse()?;
        let health_addr: SocketAddr = config.server.health_listen.parse()?;

        let session = Arc::new(InferenceSession::new(config).await?);
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
        let trigger = crate::TriggerHandles {
            topic_repo: self.session.topic_repo(),
            publisher: self.session.publisher(),
            subscriber: self.session.subscriber(),
        };
        serve_grpc_chain(
            self.flight_addr,
            self.session.context().clone(),
            self.session.tenant_binding_arc(),
            self.session_store.clone(),
            Some(trigger),
            Some(Arc::clone(&self.session)),
            shutdown,
        )
        .await
        .map_err(ServerError::from)
    }
}

/// Build and run the gRPC chain (Flight SQL + SessionService +
/// TriggerService + EmbeddingService) on `addr`, sharing the supplied
/// `SessionStore` between every service. Trigger handles and the embedding
/// session are optional — passing `None` keeps that surface unmounted, which
/// is what the gRPC-Web and `JammiSession`-only fixtures need (the embedding
/// service operates at the `InferenceSession` layer those fixtures don't
/// construct).
///
/// This is the test-fixture entry-point. Production code goes
/// through [`OssServer::run`] which derives every component from
/// the engine session itself. Both paths build the same Tonic
/// chain under the hood — there is no parallel API to drift.
pub async fn serve_grpc_chain(
    addr: SocketAddr,
    flight_ctx: SessionContext,
    flight_binding: jammi_db::tenant_scope::TenantBinding,
    store: SessionStore,
    trigger: Option<crate::TriggerHandles>,
    embedding: Option<Arc<InferenceSession>>,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), tonic::transport::Error> {
    let interceptor = TenantInterceptor::new(store.clone());

    let provider = TenantBoundProvider::new(flight_ctx.state(), flight_binding, store.clone());
    let flight = FlightSqlService::new_with_provider(Box::new(provider));
    let flight_svc = FlightServiceServer::new(flight);

    let session_svc =
        SessionServiceServer::with_interceptor(SessionServer::new(store), interceptor.clone());

    let mut builder = Server::builder()
        .accept_http1(true)
        .layer(GrpcWebLayer::new())
        .add_service(flight_svc)
        .add_service(session_svc);

    let mut mounted = vec!["Flight SQL", "SessionService"];
    if let Some(handles) = trigger {
        let trigger_svc = TriggerServiceServer::with_interceptor(
            TriggerServer::new(handles.topic_repo, handles.publisher, handles.subscriber),
            interceptor.clone(),
        );
        builder = builder.add_service(trigger_svc);
        mounted.push("TriggerService");
    }

    if let Some(session) = embedding {
        let embedding_svc =
            EmbeddingServiceServer::with_interceptor(EmbeddingServer::new(session), interceptor);
        builder = builder.add_service(embedding_svc);
        mounted.push("EmbeddingService");
    }

    tracing::info!("gRPC chain ({}) listening on {addr}", mounted.join(" + "));
    builder.serve_with_shutdown(addr, shutdown).await
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
