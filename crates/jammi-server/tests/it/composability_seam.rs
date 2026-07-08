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
use jammi_db::TenantId;
use jammi_server::grpc::session::{TenantResolver, TenantScope};
use jammi_server::routes::health::MetricsRegistry;
use jammi_server::runtime::{assemble_grpc_chain, GrpcChain};
use jammi_server::tiers::TierSet;

use super::common::grpc::{tenant_a, tenant_b};
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
        tenant_resolver: jammi_server::grpc::session::SessionIdTenantResolver::arc(
            jammi_server::grpc::session::SessionStore::new(),
        ),
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
        tenant_resolver: jammi_server::grpc::session::SessionIdTenantResolver::arc(
            jammi_server::grpc::session::SessionStore::new(),
        ),
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

// ---------------------------------------------------------------------------
// The tenant-resolver seam: `assemble_grpc_chain` with `resolver: Some(...)`
// ---------------------------------------------------------------------------

/// A test [`TenantResolver`] standing in for a downstream's identity system: it
/// maps a fake bearer credential (`authorization: Bearer tenant-a|tenant-b`) to
/// the tenant it authorizes, and REJECTS a missing or unknown credential with
/// `UNAUTHENTICATED`. It returns `Tenant`/`Err` and NEVER `Global` — the
/// authenticating-resolver contract. It reads only the `authorization`
/// metadata; it never trusts `jammi-session-id` for identity.
struct BearerTenantResolver;

#[tonic::async_trait]
impl TenantResolver for BearerTenantResolver {
    async fn resolve(
        &self,
        metadata: &tonic::metadata::MetadataMap,
    ) -> Result<TenantScope, tonic::Status> {
        let token = metadata
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| tonic::Status::unauthenticated("missing bearer token"))?;
        match token {
            "Bearer tenant-a" => Ok(TenantScope::Tenant(tenant_a())),
            "Bearer tenant-b" => Ok(TenantScope::Tenant(tenant_b())),
            _ => Err(tonic::Status::unauthenticated("unknown credential")),
        }
    }
}

/// Write a one-row Parquet so a File source registers against a real, readable
/// path (registration validates the path).
fn write_probe_parquet(path: &std::path::Path) {
    use arrow::array::{ArrayRef, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;

    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![1_i64])) as ArrayRef],
    )
    .expect("batch");
    let file = std::fs::File::create(path).expect("create parquet");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("writer");
    writer.write(&batch).expect("write");
    writer.close().expect("close");
}

/// A catalog client that attaches `authorization: <cred>` on every request (or
/// nothing when `cred` is `None`) — the data-plane analogue of a consumer's
/// gateway stamping the caller's credential.
async fn catalog_client_with_cred(
    addr: SocketAddr,
    cred: Option<&'static str>,
) -> jammi_server::grpc::proto::catalog::catalog_service_client::CatalogServiceClient<
    tonic::service::interceptor::InterceptedService<
        tonic::transport::Channel,
        impl tonic::service::Interceptor,
    >,
> {
    use jammi_server::grpc::proto::catalog::catalog_service_client::CatalogServiceClient;
    let channel = tonic::transport::Channel::from_shared(format!("http://{addr}"))
        .expect("uri")
        .connect()
        .await
        .expect("connect");
    let header: Option<tonic::metadata::MetadataValue<_>> =
        cred.map(|c| c.parse().expect("cred is ascii"));
    let attach = move |mut req: Request<()>| {
        if let Some(h) = header.clone() {
            req.metadata_mut().insert("authorization", h);
        }
        Ok(req)
    };
    CatalogServiceClient::with_interceptor(channel, attach)
}

/// Pull the source ids a caller can see over the wire.
async fn list_source_ids_over_wire(
    client: &mut jammi_server::grpc::proto::catalog::catalog_service_client::CatalogServiceClient<
        tonic::service::interceptor::InterceptedService<
            tonic::transport::Channel,
            impl tonic::service::Interceptor,
        >,
    >,
) -> Result<Vec<String>, Status> {
    use jammi_server::grpc::proto::catalog::ListSourcesRequest;
    let resp = client.list_sources(ListSourcesRequest {}).await?;
    Ok(resp
        .into_inner()
        .sources
        .into_iter()
        .map(|s| s.source_id)
        .collect())
}

/// Run one Flight SQL `SELECT` scoped by the resolver credential and return the
/// `id` rows it observes.
async fn flight_select_ids(
    addr: SocketAddr,
    cred: &str,
) -> Result<Vec<i64>, arrow_flight::error::FlightError> {
    use arrow::array::Int64Array;
    use arrow_flight::sql::client::FlightSqlServiceClient;
    use futures::TryStreamExt;

    let channel = tonic::transport::Channel::from_shared(format!("http://{addr}"))
        .expect("uri")
        .connect()
        .await
        .expect("connect");
    let mut client = FlightSqlServiceClient::new(channel);
    client.set_header("authorization", cred);
    let info = client
        .execute(
            "SELECT id FROM mutable.public.widgets ORDER BY id".to_string(),
            None,
        )
        .await?;
    let endpoint = info.endpoint.first().cloned().expect("flight endpoint");
    let ticket = endpoint.ticket.expect("endpoint ticket");
    let mut stream = client.do_get(ticket).await?;
    let mut ids = Vec::new();
    while let Some(batch) = stream.try_next().await? {
        let col = batch
            .column_by_name("id")
            .expect("id column")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64 id column");
        for i in 0..col.len() {
            ids.push(col.value(i));
        }
    }
    Ok(ids)
}

/// THE SEAM PROOF — assemble an engine-backed chain with `resolver: Some(...)`
/// and assert, over ONE served listener and on BOTH transports (gRPC data-plane
/// + Flight `db.sql`):
///
/// (i)   an engine data-plane verb runs scoped to the RESOLVED tenant — tenant
///       A's credential lists A's source; the Flight lane returns A's row;
/// (ii)  CROSS-TENANT DENIAL — tenant A's data is invisible to tenant B's
///       credential (B lists only B's source; the Flight lane returns only B's
///       row);
/// (iii) a MISSING/invalid credential is `UNAUTHENTICATED` and the handler never
///       runs — for a gRPC verb AND the Flight `db.sql` lane — so the engine
///       plane no longer runs unscoped when a resolver is present.
///
/// The resolver REPLACES the stock `jammi-session-id` interceptor (single
/// binder): the tenant is the one the credential authorizes, never one a header
/// asserts.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resolver_seam_scopes_both_transports_and_rejects_missing_credential() {
    use jammi_ai::session::InferenceSession;
    use jammi_db::source::{FileFormat, SourceConnection, SourceType};
    use jammi_db::store::mutable::definition::{MutableTableDefinitionBuilder, MutableTableId};

    let addr = bind_addr().await;

    let dir = tempfile::tempdir().expect("tempdir");
    let session = InferenceSession::open(test_config(dir.path()))
        .await
        .expect("session");

    // gRPC data-plane fixture: one File source per tenant, each registered under
    // its own tenant scope so `list_sources` returns disjoint results per
    // authenticated caller.
    let register_source = |tenant: TenantId, id: &'static str| {
        let session = Arc::clone(&session);
        let dir = dir.path().to_path_buf();
        async move {
            let pq = dir.join(format!("{id}.parquet"));
            write_probe_parquet(&pq);
            let conn = SourceConnection {
                url: Some(format!("file://{}", pq.display())),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            };
            let local = jammi_ai::Session::new(Arc::clone(&session));
            session
                .with_tenant_scoped(tenant, |_scope| {
                    local.add_source(id, SourceType::File, conn)
                })
                .await
                .expect("tenant-scoped add_source");
        }
    };
    register_source(tenant_a(), "a_source").await;
    register_source(tenant_b(), "b_source").await;

    // Flight `db.sql` fixture: one mutable table addressed by both tenants; the
    // write-side tenant guard stamps each insert, so a scoped SELECT sees only
    // its own row.
    let def = MutableTableDefinitionBuilder::new(
        MutableTableId::new("widgets").unwrap(),
        Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "id",
            arrow_schema::DataType::Int64,
            false,
        )])),
    )
    .primary_key(vec!["id".into()])
    .build()
    .unwrap();
    session.create_mutable_table(def).await.unwrap();
    for (tenant, id) in [(tenant_a(), 1_i64), (tenant_b(), 2_i64)] {
        session
            .with_tenant_scoped(tenant, |scope| async move {
                scope
                    .sql(&format!(
                        "INSERT INTO mutable.public.widgets (id) VALUES ({id})"
                    ))
                    .await
                    .unwrap();
            })
            .await;
    }

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
        // The seam under test: a downstream supplies its own authenticating
        // resolver, which binds every engine service and the Flight lane through
        // the one tenant-binding mechanism.
        tenant_resolver: Arc::new(BearerTenantResolver),
    };

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

    // (i) + (ii) gRPC data-plane, scoped to the resolved tenant, cross-tenant
    // denial. A's credential lists A's source and NOT B's; B's the inverse.
    let mut client_a = catalog_client_with_cred(addr, Some("Bearer tenant-a")).await;
    let mut client_b = catalog_client_with_cred(addr, Some("Bearer tenant-b")).await;
    let sources_a = list_source_ids_over_wire(&mut client_a)
        .await
        .expect("a lists");
    let sources_b = list_source_ids_over_wire(&mut client_b)
        .await
        .expect("b lists");
    assert!(
        sources_a.contains(&"a_source".to_string()) && !sources_a.contains(&"b_source".to_string()),
        "tenant A's credential must scope the gRPC verb to A's source only; got {sources_a:?}"
    );
    assert!(
        sources_b.contains(&"b_source".to_string()) && !sources_b.contains(&"a_source".to_string()),
        "tenant B's credential must scope the gRPC verb to B's source only; got {sources_b:?}"
    );

    // (iii) gRPC — a MISSING credential is UNAUTHENTICATED; the handler never
    // runs (no unscoped fall-through).
    let mut anon = catalog_client_with_cred(addr, None).await;
    let err = list_source_ids_over_wire(&mut anon)
        .await
        .expect_err("a gRPC verb with no credential must be rejected");
    assert_eq!(
        err.code(),
        tonic::Code::Unauthenticated,
        "missing credential on the gRPC plane is Unauthenticated, got {:?}: {}",
        err.code(),
        err.message()
    );

    // (i) + (ii) Flight `db.sql`, scoped to the resolved tenant, cross-tenant
    // denial. A's credential returns only A's row; B's returns only B's.
    let ids_a = flight_select_ids(addr, "Bearer tenant-a")
        .await
        .expect("A flight query");
    let ids_b = flight_select_ids(addr, "Bearer tenant-b")
        .await
        .expect("B flight query");
    assert_eq!(
        ids_a,
        vec![1],
        "tenant A's Flight lane returns only A's row"
    );
    assert_eq!(
        ids_b,
        vec![2],
        "tenant B's Flight lane returns only B's row"
    );

    // (iii) Flight — a MISSING credential is rejected before any query binds, so
    // the `db.sql` lane never runs unscoped (the #220 cross-transport bypass is
    // closed).
    let flight_err = flight_select_ids(addr, "Bogus nonsense")
        .await
        .expect_err("an invalid credential must reject the Flight query");
    // The rejected status surfaces through the Flight client (its variant differs
    // by which leg fails — `execute` vs `do_get` — but the code is the resolver's
    // `Unauthenticated`). Assert on the carried status, not the wrapper variant.
    assert!(
        format!("{flight_err:?}").contains("Unauthenticated"),
        "an invalid credential on the Flight lane must be Unauthenticated, got {flight_err:?}"
    );

    let _ = shutdown_tx.send(());
    let _ = handle.await;
}
