//! Library surface for the OSS `jammi-server` binary.
//!
//! The binary's orchestration entry-point is [`runtime::OssServer`].
//! The library also re-exports building blocks (`build_router`,
//! `flight::*`, `grpc::*`) so test fixtures and downstream binaries
//! (e.g. the `jammi` CLI's `serve` subcommand) can compose the same
//! pieces without reimplementing them.

pub mod error;
pub mod flight;
pub mod grpc;
pub mod grpc_web_trailers;
pub mod routes;
pub mod runtime;

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::routing::get;
use axum::Router;
use tokio::net::TcpListener;
use tokio::signal;

use jammi_db::catalog::topic_repo::TopicRepo;
use jammi_db::trigger::{Publisher, Subscriber};

use crate::error::fallback_handler;
use crate::routes::health::{self, MetricsRegistry};
use crate::runtime::ReadinessProbe;

/// Trigger-stream handles attached to the gRPC server. The caller
/// constructs these once per deployment (sharing one broker, publisher,
/// subscriber, and topic catalog repo across every connection).
///
/// Kept as a public struct because integration tests still wire trigger
/// handles manually for fixtures that need to drive a stubbed broker.
/// Production code goes through [`runtime::OssServer`] which derives
/// the same handles from the engine session.
pub struct TriggerHandles {
    pub topic_repo: Arc<TopicRepo>,
    pub publisher: Arc<Publisher>,
    pub subscriber: Arc<Subscriber>,
}

/// Build the side-channel router with `/healthz` only — no readiness
/// or metrics state attached. Callers that need the full surface
/// construct the router through [`runtime::OssServer`].
pub fn build_router() -> Router {
    Router::new()
        .route("/healthz", get(health::healthz))
        .fallback(fallback_handler)
}

/// Build the full side-channel router exposing `/healthz`, `/readyz`,
/// and `/metrics`. The readiness probe and metrics registry are passed
/// in as `Arc`s so test fixtures can substitute stubs.
pub fn build_health_router(
    readiness: Arc<ReadinessProbe>,
    metrics: Arc<MetricsRegistry>,
) -> Router {
    let readyz = Router::new()
        .route("/readyz", get(health::readyz))
        .with_state(readiness);
    let metrics = Router::new()
        .route("/metrics", get(health::metrics))
        .with_state(metrics);
    Router::new()
        .route("/healthz", get(health::healthz))
        .merge(readyz)
        .merge(metrics)
        .fallback(fallback_handler)
}

/// Start the health server with graceful shutdown on OS signals (Ctrl+C, SIGTERM).
///
/// Exposes only `/healthz` — for the full `/readyz` + `/metrics`
/// surface use [`runtime::OssServer::run`].
pub async fn serve(addr: SocketAddr) -> Result<(), std::io::Error> {
    serve_with_shutdown(addr, shutdown_signal()).await
}

/// Start the health server with a caller-provided shutdown future.
/// Useful for tests that need to trigger shutdown programmatically.
pub async fn serve_with_shutdown(
    addr: SocketAddr,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), std::io::Error> {
    let app = build_router();
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("Health server listening on {addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
}

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
