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
    /// The same `Arc<InferenceSession>` the server task drives. Shared so a
    /// test can wrap it in a `LocalSession` and assert a `RemoteSession` over
    /// the wire returns identical results / errors against the *same* engine.
    pub engine: Arc<InferenceSession>,
}

/// Spin up an in-process gRPC server hosting the chain *with* the engine-backed
/// services, mounting every compiled-in tier **except** the event tier (no
/// trigger handles). Shared by the `grpc_inference`, `grpc_eval`,
/// `grpc_introspection`, and `grpc_fine_tune` suites so they drive the same
/// wiring the embedding suite does.
pub async fn start_engine_server() -> EngineServer {
    // Every compiled-in optional tier except event — the engine-backed serve +
    // eval + (when compiled) train surface, without the trigger stream.
    let optional = jammi_server::tiers::ServiceTier::OPTIONAL
        .into_iter()
        .filter(|t| *t != jammi_server::tiers::ServiceTier::Event && t.compiled_in());
    let tiers = jammi_server::tiers::TierSet::resolve(optional).expect("non-event tiers resolve");
    start_engine_server_with_tiers(tiers).await
}

/// Like [`start_engine_server`] but also mounts the trigger handles (the event
/// tier), so the `TriggerService` (topics / publish / subscribe) is reachable
/// over the wire. Shared by the `RemoteSession` topic/subscribe/audit parity
/// tests, which drive those surfaces against the same engine a `LocalSession`
/// wraps.
pub async fn start_engine_server_with_trigger() -> EngineServer {
    start_engine_server_with_tiers(jammi_server::tiers::TierSet::all_compiled()).await
}

/// Spin up an in-process engine-backed gRPC server mounting exactly `tiers`.
/// The trigger handles (event tier) are derived from `tiers.contains(Event)`,
/// so what is mounted and what `GetServerInfo` advertises are one decision —
/// no way to construct a fixture whose handshake lies about its mount set.
/// Used by the tier-gating tests to stand up serve-only / serve+train / etc.
pub async fn start_engine_server_with_tiers(tiers: jammi_server::tiers::TierSet) -> EngineServer {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = test_config(dir.path());
    // `open` (not `new`) so the engine-backed server registers the compound
    // query SQL functions (`annotate`, …) on its context — the same shape the
    // production `OssServer` builds, and what the Flight SQL `annotate` test
    // exercises.
    let session = InferenceSession::open(cfg).await.expect("session");

    let store = SessionStore::new();
    let trigger = tiers
        .contains(jammi_server::tiers::ServiceTier::Event)
        .then(|| jammi_server::TriggerHandles {
            topic_repo: session.topic_repo(),
            publisher: session.publisher(),
            subscriber: session.subscriber(),
        });

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    drop(listener);

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let flight_ctx = session.context().clone();
    let binding = session.tenant_binding_arc();
    let engine = Arc::clone(&session);
    let handle = tokio::spawn(async move {
        jammi_server::runtime::serve_grpc_chain(
            jammi_server::runtime::GrpcChain {
                addr,
                flight_ctx,
                flight_binding: binding,
                store,
                trigger,
                engine: Some(session),
                tiers,
            },
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
        engine,
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
