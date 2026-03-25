pub mod error;
pub mod routes;
pub mod state;

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

/// Start the HTTP server with graceful shutdown.
pub async fn serve(state: Arc<AppState>, addr: SocketAddr) -> Result<(), std::io::Error> {
    let app = build_router(state);
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("HTTP server listening on {addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }

    tracing::info!("Shutdown signal received, draining connections...");
}
