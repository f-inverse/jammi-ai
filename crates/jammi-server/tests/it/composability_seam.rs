//! The S2 composability seam: `assemble_grpc_chain` → `AssembledChain`
//! (`mount` / `serve` / `into_axum_router`) → `ChainParts`.
//!
//! These cases prove a downstream can mount its own gRPC service beside the
//! engine's core chain on one listener (the `serve` path and the
//! single-listener `into_axum_router` path), that the mounted ledger records
//! it, and that the `jammi.v1.lifecycle` contract the engine mounts NO handler
//! for answers `UNIMPLEMENTED` on an OSS server. They also exercise the
//! authenticated-client seam (`SessionTransport::with_bearer`) end to end: the
//! bearer is stamped beside the session id, observed by a mounted service.
//!
//! The stub `LifecycleService` lives HERE, in the test — the engine ships no
//! implementation. It doubles as the "downstream mounts its own service" fixture
//! and, by echoing request metadata, as the `with_bearer` stamping oracle.

use std::net::SocketAddr;
use std::sync::Arc;

use jammi_admin::{CatalogClient, LifecycleClient};
use jammi_db::session::JammiSession;
use jammi_server::routes::health::MetricsRegistry;
use jammi_server::runtime::{assemble_grpc_chain, GrpcChain};
use jammi_server::tiers::TierSet;
use jammi_test_utils::test_config;
use jammi_wire::proto::lifecycle::lifecycle_service_server::{
    LifecycleService, LifecycleServiceServer,
};
use jammi_wire::proto::lifecycle::{
    ApplyLicenseRequest, ApplyLicenseResponse, BootstrapRequest, BootstrapResponse, LoginRequest,
    LoginResponse, PlatformStatus,
};
use jammi_wire::{SessionTransport, SESSION_HEADER};
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::transport::Endpoint;
use tonic::{Request, Response, Status};

/// Build a tonic [`Endpoint`] for a runtime-bound address (the ephemeral test
/// port is not `'static`, so the `&'static str` conversion cannot be used).
fn endpoint(addr: SocketAddr) -> Endpoint {
    Endpoint::from_shared(format!("http://{addr}")).expect("valid endpoint uri")
}

/// A stub `LifecycleService` that echoes back what it observed — used to prove a
/// downstream service mounts beside the engine's and that the client's headers
/// (session id + bearer) reach it. `status` echoes the `authorization` and
/// session-id metadata into two response fields so the caller can assert what
/// was stamped; the other verbs echo their request payloads.
#[derive(Clone, Default)]
struct EchoLifecycle;

#[tonic::async_trait]
impl LifecycleService for EchoLifecycle {
    async fn apply_license(
        &self,
        request: Request<ApplyLicenseRequest>,
    ) -> Result<Response<ApplyLicenseResponse>, Status> {
        let n = request.into_inner().signed_entitlement.len();
        Ok(Response::new(ApplyLicenseResponse {
            accepted: true,
            detail: format!("received {n} bytes"),
        }))
    }

    async fn bootstrap(
        &self,
        request: Request<BootstrapRequest>,
    ) -> Result<Response<BootstrapResponse>, Status> {
        let r = request.into_inner();
        Ok(Response::new(BootstrapResponse {
            dashboard_url: format!("https://dash/{}", r.admin_email),
            // Echo display name + bootstrap token so the client's forwarding
            // (and its `""` → absent mapping) is observable.
            first_login_credential: format!("{}|{}", r.admin_name, r.bootstrap_token),
        }))
    }

    async fn status(&self, request: Request<()>) -> Result<Response<PlatformStatus>, Status> {
        let md = request.metadata();
        let auth = md
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let sid = md
            .get(SESSION_HEADER)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        Ok(Response::new(PlatformStatus {
            bootstrapped: true,
            licensed: true,
            // Echo the stamped `authorization` and session id into two fields the
            // caller reads back — this is the `with_bearer` stamping oracle.
            license_state: auth,
            platform_version: sid,
        }))
    }

    async fn login(
        &self,
        request: Request<LoginRequest>,
    ) -> Result<Response<LoginResponse>, Status> {
        let email = request.into_inner().email;
        Ok(Response::new(LoginResponse {
            bearer: format!("bearer-for-{email}"),
            expires_at_micros: 0,
        }))
    }
}

/// Build a transport-only [`GrpcChain`] (no engine): Flight SQL + the
/// control-plane `CatalogService` mount, whose engine-free `GetServerInfo` +
/// tenant trio answer through the seam. Returns the chain plus the guards that
/// keep the backing session/catalog alive.
async fn transport_only_chain(addr: SocketAddr) -> (GrpcChain, TempDir, Arc<JammiSession>) {
    let dir = tempfile::tempdir().expect("tempdir");
    let session = Arc::new(
        JammiSession::new(test_config(dir.path()))
            .await
            .expect("session"),
    );
    let flight_ctx = session.context().clone();
    let flight_binding = session.tenant_binding_arc();
    let chain = GrpcChain {
        addr,
        flight_ctx,
        flight_binding,
        store: jammi_server::grpc::session::SessionStore::new(),
        trigger: None,
        engine: None,
        tiers: TierSet::resolve(std::iter::empty()).expect("core-only tier set resolves"),
        metrics: Arc::new(MetricsRegistry::new().unwrap()),
    };
    (chain, dir, session)
}

async fn bind_addr() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    drop(listener);
    addr
}

/// TEST #1 + #8 (positive) — a downstream mounts its own `LifecycleService`
/// beside the engine's `CatalogService` on one server; the engine core answers
/// through the seam, the mounted service answers, the ledger records it, and a
/// `with_bearer` client's session id + bearer both reach the mounted service.
#[tokio::test]
async fn downstream_mounts_a_service_beside_the_engine_and_both_answer() {
    let addr = bind_addr().await;
    let (chain, _dir, _session) = transport_only_chain(addr).await;

    let assembled = assemble_grpc_chain(chain)
        .expect("assemble")
        .mount(LifecycleServiceServer::new(EchoLifecycle));

    // The ledger picked up the mounted service's NamedService::NAME.
    assert!(
        assembled
            .mounted()
            .iter()
            .any(|m| m == "jammi.v1.lifecycle.LifecycleService"),
        "mounted ledger must record the downstream service: {:?}",
        assembled.mounted()
    );

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        assembled
            .serve(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("serve");
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // (a) The engine core still answers through the seam's layer stack.
    let catalog = CatalogClient::connect(endpoint(addr))
        .await
        .expect("catalog connect");
    let info = catalog.server_info().await.expect("server_info");
    assert!(
        !info.version.is_empty(),
        "engine core answers GetServerInfo"
    );

    // (b) The downstream-mounted service answers — and the `with_bearer`
    // transport stamped both the session id and the bearer, which the stub
    // echoes back.
    let transport = SessionTransport::connect(endpoint(addr))
        .await
        .expect("transport connect")
        .with_bearer("tok-123")
        .expect("bearer parses");
    let session_id = transport.session_id().to_string();
    let lifecycle = LifecycleClient::over(transport);
    let status = lifecycle.status().await.expect("status");
    assert_eq!(
        status.license_state, "Bearer tok-123",
        "the mounted service saw the stamped `authorization: Bearer` header"
    );
    assert_eq!(
        status.platform_version, session_id,
        "the mounted service saw the stamped session id alongside the bearer"
    );

    let _ = shutdown_tx.send(());
    let _ = handle.await;
}

/// TEST #7 — an OSS server (assembled with NO `LifecycleService` mounted)
/// answers `UNIMPLEMENTED` for the lifecycle contract. The doorway is defined in
/// the wire descriptor; the room is empty in OSS.
#[tokio::test]
async fn oss_server_answers_unimplemented_for_lifecycle() {
    let addr = bind_addr().await;
    let (chain, _dir, _session) = transport_only_chain(addr).await;

    // Note: no `.mount(...)` of a LifecycleService — exactly what the OSS engine
    // does.
    let assembled = assemble_grpc_chain(chain).expect("assemble");
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        assembled
            .serve(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("serve");
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let lifecycle = LifecycleClient::connect(endpoint(addr))
        .await
        .expect("connect");
    let err = lifecycle
        .status()
        .await
        .expect_err("OSS server must not implement LifecycleService");
    assert_eq!(
        err.code(),
        tonic::Code::Unimplemented,
        "an OSS server answers UNIMPLEMENTED for the lifecycle contract, got: {err:?}"
    );

    let _ = shutdown_tx.send(());
    let _ = handle.await;
}

/// TEST #6 — the candle-free `LifecycleClient` round-trips all four verbs
/// against the stub server, decoding each response shape. `apply_license`
/// surfaces `LicenseApplied` verbatim; `bootstrap` forwards `display_name` +
/// `bootstrap_token` and maps `""` to absent; `login` decodes a non-expiring
/// bearer (`expires_at_micros == 0`).
#[tokio::test]
async fn lifecycle_client_round_trips_the_four_verbs() {
    let addr = bind_addr().await;
    let (chain, _dir, _session) = transport_only_chain(addr).await;
    let assembled = assemble_grpc_chain(chain)
        .expect("assemble")
        .mount(LifecycleServiceServer::new(EchoLifecycle));
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        assembled
            .serve(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("serve");
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let client = LifecycleClient::connect(endpoint(addr))
        .await
        .expect("connect");

    let applied = client
        .apply_license(vec![1, 2, 3, 4])
        .await
        .expect("apply_license");
    assert!(applied.accepted);
    assert_eq!(applied.detail, "received 4 bytes");

    // A bootstrap with a display name and token: the stub echoes both back.
    let bootstrapped = client
        .bootstrap("admin@example.com", "pw", "Ada", "secret-token")
        .await
        .expect("bootstrap");
    assert_eq!(bootstrapped.dashboard_url, "https://dash/admin@example.com");
    assert_eq!(bootstrapped.first_login_credential, "Ada|secret-token");

    // Empty display name + token map to the proto3 default (absent on the wire):
    // the stub echoes empty strings, proving the client forwards `""` as absent
    // rather than a literal.
    let no_extras = client
        .bootstrap("op@example.com", "pw", "", "")
        .await
        .expect("bootstrap no extras");
    assert_eq!(no_extras.first_login_credential, "|");

    let bearer = client.login("user@example.com", "pw").await.expect("login");
    assert_eq!(bearer.bearer, "bearer-for-user@example.com");
    assert_eq!(
        bearer.expires_at_micros, 0,
        "a non-expiring bearer decodes to 0"
    );

    let _ = shutdown_tx.send(());
    let _ = handle.await;
}

/// TEST #8 (negative) — `with_bearer` is fallible: a non-ASCII token cannot
/// parse as gRPC ASCII metadata and surfaces as `JammiError::Config`, never a
/// panic or a silently dropped header.
#[tokio::test]
async fn with_bearer_rejects_a_non_ascii_token() {
    let addr = bind_addr().await;
    let (chain, _dir, _session) = transport_only_chain(addr).await;
    let assembled = assemble_grpc_chain(chain).expect("assemble");
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        assembled
            .serve(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("serve");
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let transport = SessionTransport::connect(endpoint(addr))
        .await
        .expect("connect");
    // A control character (newline) cannot appear in an HTTP/gRPC metadata
    // value, so the parse fails. `SessionTransport` is not `Debug`, so match
    // rather than `expect_err`.
    match transport.with_bearer("bad\nvalue") {
        Err(jammi_db::error::JammiError::Config(_)) => {}
        Err(other) => panic!("a bad bearer must surface as JammiError::Config, got {other:?}"),
        Ok(_) => panic!("a non-ascii bearer must be rejected, not silently accepted"),
    }

    let _ = shutdown_tx.send(());
    let _ = handle.await;
}

/// TEST #2 — the single-listener `into_axum_router` path: assemble, mount the
/// stub service, split into a layer-free `axum::Router` + `ChainParts`, re-apply
/// the transport layers test-side (the seam contract: the downstream owns them
/// on this path), nest the gRPC routes beside a plain HTTP route under one axum
/// listener, and assert BOTH a gRPC call (engine `GetServerInfo`) and the plain
/// HTTP route answer. `ChainParts` carries `addr` + the mounted ledger across
/// the split.
#[tokio::test]
async fn into_axum_router_composes_one_listener_with_a_plain_http_route() {
    use axum::routing::get;
    use tonic_web::GrpcWebLayer;

    let addr = bind_addr().await;
    let (chain, _dir, _session) = transport_only_chain(addr).await;
    let assembled = assemble_grpc_chain(chain)
        .expect("assemble")
        .mount(LifecycleServiceServer::new(EchoLifecycle));

    let (grpc_router, parts) = assembled.into_axum_router();
    // ChainParts carries the ledger + addr across the split.
    assert_eq!(parts.addr, addr);
    assert!(parts
        .mounted
        .iter()
        .any(|m| m == "jammi.v1.lifecycle.LifecycleService"));

    // Per the seam contract, the downstream re-applies the grpc-web + trailer
    // repair layers on its own listener (they are NOT baked into the routes).
    let grpc_router = grpc_router
        .layer(GrpcWebLayer::new())
        .layer(jammi_server::grpc_web_trailers::GrpcWebTrailersLayer::new());

    // One axum listener: the engine's gRPC routes plus a plain HTTP route the
    // downstream owns. The gRPC routes are keyed by `/<package>.<Service>/...`
    // paths, so a distinct top-level `/plain` route does not collide.
    let app = axum::Router::new()
        .route("/plain", get(|| async { "plain-http-ok" }))
        .merge(grpc_router);

    let listener = TcpListener::bind(addr).await.expect("rebind");
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("axum serve");
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // (a) The plain HTTP route answers.
    let body = reqwest::get(format!("http://{addr}/plain"))
        .await
        .expect("http get")
        .text()
        .await
        .expect("http body");
    assert_eq!(body, "plain-http-ok");

    // (b) A gRPC call to the engine core answers on the same listener.
    let catalog = CatalogClient::connect(endpoint(addr))
        .await
        .expect("catalog connect");
    let info = catalog
        .server_info()
        .await
        .expect("server_info over composed listener");
    assert!(!info.version.is_empty());

    // Keep the worker guard (if any) alive until after serving, per the contract.
    #[cfg(feature = "train")]
    let _keep = parts.train_worker;

    let _ = shutdown_tx.send(());
    let _ = handle.await;
}

/// Wrap a proto-encoded payload in a single gRPC-Web data frame: 1 flag byte
/// (`0x00` — not a trailer), a big-endian u32 length, then the payload. This is
/// the on-the-wire shape a browser gRPC-web client POSTs.
fn frame_grpc_web(payload: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(5 + payload.len());
    buf.push(0x00);
    buf.extend_from_slice(
        &u32::try_from(payload.len())
            .expect("payload fits in u32")
            .to_be_bytes(),
    );
    buf.extend_from_slice(payload);
    buf
}

/// Scan a gRPC-Web response body for its in-body trailer frame (flag `0x80`) and
/// return the decoded trailer text. Returns `None` if the body carries no
/// trailer frame — which for an error response is itself the failure the seam's
/// trailer-repair layer exists to prevent.
fn grpc_web_trailer_block(body: &[u8]) -> Option<String> {
    let mut cursor = 0;
    while cursor + 5 <= body.len() {
        let flag = body[cursor];
        let len = u32::from_be_bytes([
            body[cursor + 1],
            body[cursor + 2],
            body[cursor + 3],
            body[cursor + 4],
        ]) as usize;
        let start = cursor + 5;
        let end = start + len;
        assert!(end <= body.len(), "grpc-web frame extends past body");
        if flag & 0x80 != 0 {
            return Some(
                std::str::from_utf8(&body[start..end])
                    .expect("trailer block is utf-8")
                    .to_string(),
            );
        }
        cursor = end;
    }
    None
}

/// TEST — the single-listener `into_layered_axum_router` path: assemble an
/// engine-backed chain, mount the stub service, split into a LAYERED
/// `axum::Router` + `ChainParts`, and hand the router to `axum::serve`
/// DIRECTLY — no manual re-layering, no `Router::<()>::new().merge(...)`
/// re-nest. Over one listener this proves the helper is both CORRECT and
/// ERGONOMIC:
///
/// (i)  a gRPC verb answers through the pre-applied framing (native HTTP/2
///      `GetServerInfo`), and
/// (ii) the trailer-repair layer is live: a unary handler error
///      (`EmbeddingService.EncodeQuery` against a nonexistent model → an
///      `invalid_argument` Status) comes back to a gRPC-web client as an
///      in-body `0x80` trailer frame carrying `grpc-status: 3`, with the status
///      absent from the HTTP headers — mirroring the engine's `grpc_web`
///      trailer-repair oracle, but reached through the layered-router helper
///      rather than `serve`.
///
/// If the returned router needed a re-nest to make `axum::serve` accept it, this
/// test would not compile — that is the ergonomic half of the proof.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn into_layered_axum_router_serves_directly_with_grpc_web_trailer_repair() {
    use jammi_ai::session::InferenceSession;
    use jammi_server::grpc::proto::embedding::{
        encode_query_request::Input, EncodeQueryRequest, Modality,
    };
    use prost::Message as _;

    let addr = bind_addr().await;

    // An engine-backed chain so a real unary handler (`EncodeQuery`) can return
    // a structured error — the transport-only chain has no such data-plane verb.
    let dir = tempfile::tempdir().expect("tempdir");
    let session = InferenceSession::open(test_config(dir.path()))
        .await
        .expect("session");
    let flight_ctx = session.context().clone();
    let flight_binding = session.tenant_binding_arc();
    let chain = GrpcChain {
        addr,
        flight_ctx,
        flight_binding,
        store: jammi_server::grpc::session::SessionStore::new(),
        trigger: None,
        engine: Some(session),
        tiers: TierSet::resolve(std::iter::empty()).expect("core-only tier set resolves"),
        metrics: Arc::new(MetricsRegistry::new().unwrap()),
    };

    let assembled = assemble_grpc_chain(chain)
        .expect("assemble")
        .mount(LifecycleServiceServer::new(EchoLifecycle));

    // The whole point: the returned router is served DIRECTLY. No `.layer(...)`
    // re-application, no `Router::<()>::new().merge(...)` re-nest.
    let (router, parts) = assembled.into_layered_axum_router();
    assert_eq!(parts.addr, addr);
    assert!(parts
        .mounted
        .iter()
        .any(|m| m == "jammi.v1.lifecycle.LifecycleService"));

    let listener = TcpListener::bind(addr).await.expect("rebind");
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("axum serve");
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // (i) A gRPC verb answers through the pre-applied layer stack (native HTTP/2).
    let catalog = CatalogClient::connect(endpoint(addr))
        .await
        .expect("catalog connect");
    let info = catalog
        .server_info()
        .await
        .expect("server_info over layered listener");
    assert!(
        !info.version.is_empty(),
        "engine core answers GetServerInfo"
    );

    // (ii) The trailer-repair layer is live: an engine error reaches a gRPC-web
    // client as an in-body `0x80` trailer frame, not a trailers-only HTTP header
    // set with an empty body.
    let request_proto = EncodeQueryRequest {
        model_id: "local:/does/not/exist".into(),
        modality: Modality::Text as i32,
        input: Some(Input::Text("a query".into())),
    };
    let mut payload = Vec::new();
    request_proto.encode(&mut payload).expect("encode proto");
    let body = frame_grpc_web(&payload);

    let client = reqwest::Client::builder()
        .http1_only()
        .build()
        .expect("reqwest client");
    let response = client
        .post(format!(
            "http://{addr}/jammi.v1.embedding.EmbeddingService/EncodeQuery"
        ))
        .header("content-type", "application/grpc-web+proto")
        .header("accept", "application/grpc-web+proto")
        .header("x-grpc-web", "1")
        .header("jammi-session-id", "layered-grpc-web-error")
        .body(body)
        .send()
        .await
        .expect("grpc-web POST");

    // gRPC-web reports application errors as HTTP 200; the status rides the
    // in-body trailer frame, never a trailers-only HTTP header.
    assert_eq!(response.status(), 200, "gRPC-Web shim returns 200 OK");
    assert!(
        response.headers().get("grpc-status").is_none(),
        "grpc-status must be repaired into the in-body trailer frame, not left as an HTTP header"
    );

    let body_bytes = response.bytes().await.expect("response body");
    let trailers = grpc_web_trailer_block(&body_bytes)
        .expect("an error response must carry an in-body 0x80 trailer frame");
    assert!(
        trailers.contains("grpc-status: 3") || trailers.contains("grpc-status:3"),
        "the in-body trailer frame must carry the engine error's status (3 = invalid_argument), got {trailers:?}"
    );

    // Hold the worker guard (if any) alive until after serving, per the contract.
    #[cfg(feature = "train")]
    let _keep = parts.train_worker;

    let _ = shutdown_tx.send(());
    let _ = handle.await;
}
