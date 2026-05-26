//! gRPC-Web smoke test for `serve_grpc_with_shutdown`. Drives the gRPC
//! surface with a hand-built HTTP/1.1 client (reqwest) so the test
//! exercises the real `GrpcWebLayer` framing — the same path a browser
//! takes when it can't speak HTTP/2 + grpc framing directly. A native
//! HTTP/2 channel would skip the shim entirely; this test exists to
//! catch a regression where the shim is removed or stops accepting
//! HTTP/1.1.

use std::net::SocketAddr;
use std::str::FromStr;
use std::time::Duration;

use jammi_db::TenantId;
use jammi_server::grpc::proto::session::{SetTenantRequest, Tenant};
use jammi_server::grpc::session::{SessionId, SessionStore};
use prost::Message;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

use super::common::grpc::TENANT_A;

/// Spin up an in-process gRPC server hosting only `SessionService` (no
/// trigger handles — this test does not need them). Returns the bound
/// address, the shared store the server reads/writes, and the shutdown
/// + join handles.
async fn start_session_only_server() -> (
    SocketAddr,
    SessionStore,
    oneshot::Sender<()>,
    tokio::task::JoinHandle<()>,
) {
    let store = SessionStore::new();
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    drop(listener);

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let store_for_server = store.clone();
    let handle = tokio::spawn(async move {
        jammi_server::serve_grpc_with_shutdown(addr, store_for_server, None, async move {
            let _ = shutdown_rx.await;
        })
        .await
        .expect("grpc server");
    });

    // Give the server a moment to bind.
    tokio::time::sleep(Duration::from_millis(50)).await;

    (addr, store, shutdown_tx, handle)
}

/// Wrap a proto-encoded payload in a single gRPC-Web data frame:
/// 1 byte flag (0x00 — not a trailer), 4 bytes big-endian length, then
/// the payload. This is the on-the-wire shape the server's
/// `GrpcWebLayer` decodes before handing the request to Tonic.
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

/// Decode a gRPC-Web response body into `(data_frame, trailer_text)`.
/// The response is a concatenation of length-prefixed frames; the data
/// frame has flag 0x00 and the trailer frame has flag 0x80 with an
/// HTTP/1.1-style header block in the payload.
fn unframe_grpc_web(body: &[u8]) -> (Vec<u8>, String) {
    let mut data = Vec::new();
    let mut trailers = String::new();
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
        let payload = &body[start..end];
        if flag & 0x80 == 0 {
            data.extend_from_slice(payload);
        } else {
            trailers.push_str(std::str::from_utf8(payload).expect("trailer block is utf-8"));
        }
        cursor = end;
    }
    (data, trailers)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grpc_web_set_tenant_round_trip() {
    let (addr, store, shutdown, handle) = start_session_only_server().await;

    // Build the request body: a length-prefixed SetTenantRequest{ tenant }.
    let request_proto = SetTenantRequest {
        tenant: Some(Tenant {
            id: TENANT_A.to_string(),
        }),
    };
    let mut payload = Vec::new();
    request_proto.encode(&mut payload).expect("encode proto");
    let body = frame_grpc_web(&payload);

    // Issue the call as an HTTP/1.1 client. reqwest defaults to HTTP/1.1
    // for plaintext URLs (no ALPN), which is exactly what a browser
    // grpc-web client sends.
    let client = reqwest::Client::builder()
        .http1_only()
        .build()
        .expect("reqwest client");
    let session_id = "grpc-web-session";
    let response = client
        .post(format!(
            "http://{addr}/jammi.v1.session.SessionService/SetTenant"
        ))
        .header("content-type", "application/grpc-web+proto")
        .header("accept", "application/grpc-web+proto")
        .header("x-grpc-web", "1")
        .header("jammi-session-id", session_id)
        .body(body)
        .send()
        .await
        .expect("grpc-web POST");

    // The shim returns 200 + a grpc-web+proto framed body even on
    // application errors — the status code lives in the trailer block.
    assert_eq!(response.status(), 200, "expected 200 OK from gRPC-Web shim");
    assert_eq!(
        response.version(),
        reqwest::Version::HTTP_11,
        "gRPC-Web shim must serve the HTTP/1.1 client over HTTP/1.1"
    );
    let content_type = response
        .headers()
        .get("content-type")
        .expect("response has content-type")
        .to_str()
        .expect("content-type ascii");
    assert!(
        content_type.starts_with("application/grpc-web"),
        "expected grpc-web content-type, got {content_type:?}"
    );

    let body_bytes = response.bytes().await.expect("response body");
    let (data, trailers) = unframe_grpc_web(&body_bytes);
    assert!(
        trailers.contains("grpc-status: 0") || trailers.contains("grpc-status:0"),
        "expected grpc-status: 0 in trailer block, got {trailers:?}"
    );
    // SetTenant returns google.protobuf.Empty — the data frame is the
    // zero-length encoding of an empty message.
    assert!(
        data.is_empty(),
        "expected empty response body for SetTenant, got {} bytes",
        data.len()
    );

    // The interesting assertion: the server stored the binding. A
    // follow-up call (here, a direct store lookup using the same
    // session id the client sent) sees the tenant bound.
    let bound = store.get(&SessionId::new(session_id));
    assert_eq!(
        bound,
        Some(TenantId::from_str(TENANT_A).expect("TENANT_A is a uuid")),
        "session store must reflect the gRPC-Web SetTenant call"
    );

    let _ = shutdown.send(());
    let _ = handle.await;
}
