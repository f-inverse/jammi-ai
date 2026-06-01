//! gRPC test helpers shared between `grpc_session`, `grpc_trigger`, and
//! `flight_tenant`. Each of those files previously carried its own copy of
//! the `with_session` interceptor closure, the `channel(addr)` constructor,
//! and the two well-known tenant UUIDs we use as test fixtures — three
//! near-identical copies that violated CLAUDE.md §DRY. Centralising them
//! here keeps the three test surfaces in lockstep and gives new tests one
//! obvious place to plug into.

use std::net::SocketAddr;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use jammi_ai::session::InferenceSession;
use jammi_db::TenantId;
use jammi_server::grpc::session::{SessionStore, SESSION_HEADER};
use jammi_test_utils::test_config;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::metadata::MetadataValue;
use tonic::transport::Channel;
use tonic::Request;

/// Well-known tenant UUIDs used as fixtures across the gRPC integration
/// tests. These are generic UUIDs not coupled to any downstream tenant
/// (jammi is the substrate; accurisk/lace/etc. live in product crates).
pub const TENANT_A: &str = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a";
pub const TENANT_B: &str = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9b";

/// Parse [`TENANT_A`] into a typed [`TenantId`]. Panics on programmer
/// error (the constant is a valid UUID by construction).
pub fn tenant_a() -> TenantId {
    TenantId::from_str(TENANT_A).expect("TENANT_A is a valid UUID")
}

/// Parse [`TENANT_B`] into a typed [`TenantId`].
pub fn tenant_b() -> TenantId {
    TenantId::from_str(TENANT_B).expect("TENANT_B is a valid UUID")
}

/// Build an HTTP/2 channel to an in-process Tonic server on `addr`. Used by
/// every gRPC test that needs to attach a client — the address is supplied
/// by the per-test fixture (typically backed by a `TcpListener` bound to
/// `127.0.0.1:0`).
pub async fn channel(addr: SocketAddr) -> Channel {
    Channel::from_shared(format!("http://{addr}"))
        .expect("channel uri")
        .connect()
        .await
        .expect("channel connect")
}

/// Guards that keep an in-process engine-backed gRPC server (and its catalog)
/// alive for the duration of a test. Dropping `shutdown` or letting it fall out
/// of scope tears the server down; `_dir` roots the engine's temp artifact dir.
pub struct EngineServer {
    pub addr: SocketAddr,
    pub shutdown: oneshot::Sender<()>,
    /// RAII guard: roots the engine's temp artifact dir for the server's
    /// lifetime and deletes it on drop. Held, never read.
    pub _dir: TempDir,
    pub handle: tokio::task::JoinHandle<()>,
}

/// Spin up an in-process gRPC server hosting the chain *with* the engine-backed
/// services (`EmbeddingService`, `InferenceService`, `EvalService`). These are
/// the only services that need a real `InferenceSession`; the trigger/session
/// surfaces are unmounted (`None`). Shared by the `grpc_inference` and
/// `grpc_eval` suites so they drive the same wiring the embedding suite does.
pub async fn start_engine_server() -> EngineServer {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = test_config(dir.path());
    let session = Arc::new(InferenceSession::new(cfg).await.expect("session"));

    let store = SessionStore::new();

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    drop(listener);

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let flight_ctx = session.context().clone();
    let binding = session.tenant_binding_arc();
    let handle = tokio::spawn(async move {
        jammi_server::runtime::serve_grpc_chain(
            addr,
            flight_ctx,
            binding,
            store,
            None,
            Some(session),
            async move {
                let _ = shutdown_rx.await;
            },
        )
        .await
        .expect("grpc server");
    });

    // Give the server a moment to bind.
    tokio::time::sleep(Duration::from_millis(50)).await;

    EngineServer {
        addr,
        shutdown: shutdown_tx,
        _dir: dir,
        handle,
    }
}

/// Build a request-extending interceptor closure that injects the
/// `jammi-session-id` header on every outgoing request. This is the test
/// counterpart to [`jammi_server::grpc::session::TenantInterceptor`]: the
/// server reads the header and binds the tenant; the test passes the same
/// session id on every call so the binding is observable.
pub fn with_session(
    session_id: &str,
) -> impl Fn(Request<()>) -> Result<Request<()>, tonic::Status> + Clone {
    let id: MetadataValue<_> = session_id.parse().expect("session-id ascii");
    move |mut req: Request<()>| {
        req.metadata_mut().insert(SESSION_HEADER, id.clone());
        Ok(req)
    }
}
