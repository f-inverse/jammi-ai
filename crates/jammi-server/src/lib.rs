pub mod error;
pub mod flight;
pub mod routes;
pub mod state;

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::routing::{delete, get, post};
use axum::Router;
use tokio::net::TcpListener;
use tokio::signal;

use crate::error::fallback_handler;
use crate::routes::{embeddings, fine_tune, health, models, sources};
use crate::state::AppState;

/// Build the axum router with all routes.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health::health))
        .route("/sources", get(sources::list_sources))
        .route("/sources", post(sources::add_source))
        .route("/sources/{id}", delete(sources::remove_source))
        .route("/models", get(models::list_models))
        .route("/models/preload", post(models::preload_model))
        .route(
            "/embeddings/generate",
            post(embeddings::generate_embeddings),
        )
        .route("/fine-tune", post(fine_tune::start_fine_tune))
        .route("/fine-tune", get(fine_tune::list_fine_tune_jobs))
        .fallback(fallback_handler)
        .with_state(state)
}

/// Start the HTTP server with graceful shutdown on OS signals (Ctrl+C, SIGTERM).
pub async fn serve(state: Arc<AppState>, addr: SocketAddr) -> Result<(), std::io::Error> {
    serve_with_shutdown(state, addr, shutdown_signal()).await
}

/// Start the HTTP server with a caller-provided shutdown future.
///
/// Useful for tests that need to trigger shutdown programmatically.
pub async fn serve_with_shutdown(
    state: Arc<AppState>,
    addr: SocketAddr,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), std::io::Error> {
    let app = build_router(state);
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("HTTP server listening on {addr}");
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
