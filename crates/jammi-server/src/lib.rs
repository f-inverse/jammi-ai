pub mod error;
pub mod flight;
pub mod grpc;
pub mod routes;

use std::future::Future;
use std::net::SocketAddr;

use axum::routing::get;
use axum::Router;
use tokio::net::TcpListener;
use tokio::signal;

use crate::error::fallback_handler;
use crate::routes::health;

/// Build the axum router with the health endpoint.
pub fn build_router() -> Router {
    Router::new()
        .route("/health", get(health::health))
        .fallback(fallback_handler)
}

/// Start the health server with graceful shutdown on OS signals (Ctrl+C, SIGTERM).
pub async fn serve(addr: SocketAddr) -> Result<(), std::io::Error> {
    serve_with_shutdown(addr, shutdown_signal()).await
}

/// Start the health server with a caller-provided shutdown future.
///
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

/// Start the gRPC server hosting [`grpc::session::SessionServer`] behind
/// the [`grpc::session::TenantInterceptor`]. Returns when `shutdown` resolves.
///
/// Flight SQL queries running through `flight::serve` should share the same
/// [`grpc::session::SessionStore`] so a tenant bound via `SessionService.SetTenant`
/// applies to Flight SQL queries on the same `jammi-session-id`.
pub async fn serve_grpc_with_shutdown(
    addr: SocketAddr,
    store: grpc::session::SessionStore,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), tonic::transport::Error> {
    use grpc::proto::session::session_service_server::SessionServiceServer;
    use grpc::session::{SessionServer, TenantInterceptor};

    let interceptor = TenantInterceptor::new(store.clone());
    let session_svc =
        SessionServiceServer::with_interceptor(SessionServer::new(store), interceptor);

    tracing::info!("gRPC server (SessionService) listening on {addr}");
    tonic::transport::Server::builder()
        .add_service(session_svc)
        .serve_with_shutdown(addr, shutdown)
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
