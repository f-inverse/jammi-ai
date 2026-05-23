pub mod error;
pub mod flight;
pub mod grpc;
pub mod routes;

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::routing::get;
use axum::Router;
use tokio::net::TcpListener;
use tokio::signal;

use jammi_engine::catalog::topic_repo::TopicRepo;
use jammi_engine::trigger::{Publisher, Subscriber};

use crate::error::fallback_handler;
use crate::routes::health;

/// Trigger-stream handles attached to the gRPC server. The caller
/// constructs these once per deployment (sharing one broker, publisher,
/// subscriber, and topic catalog repo across every connection) and passes
/// them to `serve_grpc_with_shutdown`. Omit by passing `None` to keep the
/// trigger surface unmounted.
pub struct TriggerHandles {
    pub topic_repo: Arc<TopicRepo>,
    pub publisher: Arc<Publisher>,
    pub subscriber: Arc<Subscriber>,
}

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
/// Native gRPC clients reach the surface over HTTP/2; browser callers reach
/// the same services over HTTP/1.1 via the gRPC-Web shim (`application/grpc-web+proto`).
///
/// Flight SQL queries running through `flight::serve` should share the same
/// [`grpc::session::SessionStore`] so a tenant bound via `SessionService.SetTenant`
/// applies to Flight SQL queries on the same `jammi-session-id`.
pub async fn serve_grpc_with_shutdown(
    addr: SocketAddr,
    store: grpc::session::SessionStore,
    trigger: Option<TriggerHandles>,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), tonic::transport::Error> {
    use grpc::proto::session::session_service_server::SessionServiceServer;
    use grpc::proto::trigger::trigger_service_server::TriggerServiceServer;
    use grpc::session::{SessionServer, TenantInterceptor};
    use grpc::trigger::TriggerServer;
    use tonic_web::GrpcWebLayer;

    let interceptor = TenantInterceptor::new(store.clone());
    let session_svc =
        SessionServiceServer::with_interceptor(SessionServer::new(store), interceptor.clone());

    let mut builder = tonic::transport::Server::builder()
        .accept_http1(true)
        .layer(GrpcWebLayer::new())
        .add_service(session_svc);
    if let Some(handles) = trigger {
        let trigger_svc = TriggerServiceServer::with_interceptor(
            TriggerServer::new(handles.topic_repo, handles.publisher, handles.subscriber),
            interceptor,
        );
        builder = builder.add_service(trigger_svc);
        tracing::info!("gRPC server (SessionService + TriggerService) listening on {addr}");
    } else {
        tracing::info!("gRPC server (SessionService) listening on {addr}");
    }

    builder.serve_with_shutdown(addr, shutdown).await
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
