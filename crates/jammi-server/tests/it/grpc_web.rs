//! gRPC-Web smoke test for `serve_grpc_with_shutdown`. Drives the gRPC
//! surface with a hand-built HTTP/1.1 client (reqwest) so the test
//! exercises the real `GrpcWebLayer` framing — the same path a browser
//! takes when it can't speak HTTP/2 + grpc framing directly. A native
//! HTTP/2 channel would skip the shim entirely; this test exists to
//! catch a regression where the shim is removed or stops accepting
//! HTTP/1.1.

use std::net::SocketAddr;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use jammi_db::session::JammiSession;
use jammi_db::TenantId;
use jammi_server::grpc::proto::catalog::{SetTenantRequest, Tenant};
use jammi_server::grpc::session::{SessionId, SessionStore};
use jammi_test_utils::test_config;
use prost::Message;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

use super::common::grpc::TENANT_A;

/// Spin up an in-process gRPC server hosting `CatalogService` (no
/// trigger handles — this test does not need them). The Flight SQL
/// service is mounted on the same chain as in production so the
/// gRPC-Web routing assertion exercises the real binary's surface;
/// the test never issues a Flight RPC. Returns the bound address,
/// the shared store the server reads/writes, the shutdown + join
/// handles, and the `TempDir` keeping the session's catalog alive.
async fn start_session_only_server() -> (
    SocketAddr,
    SessionStore,
    oneshot::Sender<()>,
    tokio::task::JoinHandle<()>,
    TempDir,
) {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg = test_config(dir.path());
    let session = Arc::new(JammiSession::new(cfg).await.expect("session"));

    let store = SessionStore::new();
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    drop(listener);

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let store_for_server = store.clone();
    let flight_ctx = session.context().clone();
    let binding = session.tenant_binding_arc();
    let handle = tokio::spawn(async move {
        jammi_server::runtime::serve_grpc_chain(
            jammi_server::runtime::GrpcChain {
                addr,
                flight_ctx,
                flight_binding: binding,
                store: store_for_server,
                trigger: None,
                engine: None,
                tiers: jammi_server::tiers::TierSet::resolve(std::iter::empty())
                    .expect("core-only tier set resolves"),
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

    (addr, store, shutdown_tx, handle, dir)
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

/// The outcome of replaying `@connectrpc/connect-web`'s gRPC-Web *unary* reader
/// over a response body — the exact decision the TypeScript client reaches.
#[derive(Debug)]
enum ConnectWebUnary {
    /// `validateTrailer` found an error status (`findTrailerError` returned a
    /// `ConnectError`): the client surfaces this typed error. Carries the gRPC
    /// status code, the human message, and the per-detail type URLs.
    TypedError {
        code: i32,
        message: String,
        detail_type_urls: Vec<String>,
    },
    /// The reader fell through to `if (message === undefined) throw "missing
    /// message"` — i.e. no message frame AND `findTrailerError` found no error
    /// status. This is the residual-of-#51 failure: a trailers-only error whose
    /// `grpc-status-details-bin` lacks a `code`, so the client mis-reports
    /// `[unimplemented] "missing message"` instead of the real error.
    MissingMessage,
}

/// Replay `@connectrpc/connect-web`'s gRPC-Web unary response reader against a
/// raw response body, returning the decision the TS client reaches. A faithful
/// Rust port of `connect-web/src/grpc-web-transport.ts` `unary()` and
/// `connect/src/protocol-grpc/trailer-status.ts` `findTrailerError` (the
/// upstream source, pinned by the @connectrpc/connect-web version the edge's
/// `@f-inverse/jammi-client` wraps). The `GrpcWebTrailersLayer` strips
/// `grpc-status` from the HTTP headers into the in-body trailer frame, so the
/// terminal status the unary reader reads lives entirely in that frame's
/// `validateTrailer(trailer, …)` path — the HTTP headers carry no status and are
/// not consulted:
///
/// * It frames the body, treating `flag & 0x80` as the trailer frame and any
///   other frame as a message — so a trailers-only error leaves `message`
///   unset.
/// * `findTrailerError` *prefers* `grpc-status-details-bin` over the
///   `grpc-status` trailer: it decodes the header as a `google.rpc.Status` and,
///   if `status.code == 0`, treats the response as success. This is the exact
///   step #51's bare-`JammiErrorDetail` framing tripped — its `Any`-less,
///   `code`-less bytes decode to `code == 0`, so the client saw "success" and
///   then `missing message`. A spec-compliant `google.rpc.Status` envelope with
///   `code` set is what makes the client surface the typed error.
fn connect_web_unary_outcome(body: &[u8]) -> ConnectWebUnary {
    use jammi_server::grpc::proto::error::RpcStatus;
    use prost::Message as _;

    let (data, trailer_text) = unframe_grpc_web(body);
    let message_present = !data.is_empty();

    // `trailerParse`: CRLF-separated `name:value`, first colon splits, trim both.
    let mut grpc_status: Option<String> = None;
    let mut grpc_message: Option<String> = None;
    let mut status_details_bin: Option<String> = None;
    for line in trailer_text.split("\r\n") {
        if let Some(i) = line.find(':') {
            if i == 0 {
                continue;
            }
            let name = line[..i].trim().to_ascii_lowercase();
            let value = line[i + 1..].trim().to_string();
            match name.as_str() {
                "grpc-status" => grpc_status = Some(value),
                "grpc-message" => grpc_message = Some(value),
                "grpc-status-details-bin" => status_details_bin = Some(value),
                _ => {}
            }
        }
    }

    // `findTrailerError`: prefer the protobuf-encoded `google.rpc.Status`.
    if let Some(b64) = status_details_bin {
        // connect-web decodes base64 tolerant of missing padding (gRPC binary
        // header convention); replicate that before the `RpcStatus` decode.
        let bytes = base64_std_decode(&b64);
        let status = RpcStatus::decode(bytes.as_slice()).expect("decodes as google.rpc.Status");
        if status.code == 0 {
            // `findTrailerError` returns undefined → `validateTrailer` does not
            // throw → the unary reader hits `missing message`.
            return ConnectWebUnary::MissingMessage;
        }
        return ConnectWebUnary::TypedError {
            code: status.code,
            message: status.message,
            detail_type_urls: status.details.into_iter().map(|a| a.type_url).collect(),
        };
    }
    // Fallback: the `grpc-status` trailer.
    match grpc_status.as_deref() {
        None | Some("0") => {
            if message_present {
                ConnectWebUnary::TypedError {
                    code: 0,
                    message: grpc_message.unwrap_or_default(),
                    detail_type_urls: Vec::new(),
                }
            } else {
                ConnectWebUnary::MissingMessage
            }
        }
        Some(code) => ConnectWebUnary::TypedError {
            code: code.parse().expect("grpc-status is an integer"),
            message: grpc_message.unwrap_or_default(),
            detail_type_urls: Vec::new(),
        },
    }
}

/// Decode standard-alphabet base64 that may omit `=` padding — the gRPC
/// binary-header encoding tonic emits and connect-web's reader accepts. A
/// minimal dependency-free decoder so the interop test stays hermetic and pulls
/// no new crate just to mirror what the TS client does.
fn base64_std_decode(s: &str) -> Vec<u8> {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let val = |c: u8| -> u32 {
        ALPHABET
            .iter()
            .position(|&a| a == c)
            .unwrap_or_else(|| panic!("non-base64 byte {c:#x} in grpc-status-details-bin"))
            as u32
    };
    let symbols: Vec<u8> = s.bytes().filter(|&c| c != b'=').collect();
    let mut out = Vec::with_capacity(symbols.len() * 3 / 4);
    for chunk in symbols.chunks(4) {
        let mut acc = 0u32;
        for &c in chunk {
            acc = (acc << 6) | val(c);
        }
        // A 4-symbol chunk carries 3 bytes; 3 symbols carry 2; 2 symbols carry 1.
        let bits = chunk.len() * 6;
        acc <<= 24 - bits as u32;
        let nbytes = bits / 8;
        for i in 0..nbytes {
            out.push((acc >> (16 - i * 8)) as u8);
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grpc_web_set_tenant_round_trip() {
    let (addr, store, shutdown, handle, _dir) = start_session_only_server().await;

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
            "http://{addr}/jammi.v1.catalog.CatalogService/SetTenant"
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

/// A detail-bearing engine error returned by a unary handler MUST reach a
/// gRPC-Web client as a well-formed in-body `0x80` trailer frame carrying
/// `grpc-status` + a *spec-compliant* `grpc-status-details-bin`
/// (`google.rpc.Status`) — and `@connectrpc/connect-web`'s unary reader MUST
/// surface that as the real typed error, never `[unimplemented] "missing
/// message"`.
///
/// This is the real gRPC-Web ↔ live-tonic-web interop test the hermetic
/// substrate suite (which faked `fetch`) never had: it POSTs a real
/// `application/grpc-web+proto` request to an in-process `serve_grpc_chain`
/// that drives the *actual* `GrpcWebLayer` + `GrpcWebTrailersLayer` framing,
/// then replays connect-web's unary reader ([`connect_web_unary_outcome`]) over
/// the response and asserts the client surfaces the typed error.
///
/// This closes the gap #51 left. #51 moved the status into an in-body trailer
/// frame (so the framing assertions below pass), but its `grpc-status-details-bin`
/// carried a *bare* `JammiErrorDetail` proto — no top-level `code` field. The
/// gRPC rich-error contract requires that header to be a `google.rpc.Status`
/// whose `code` mirrors `grpc-status`; reading the bare proto as a
/// `google.rpc.Status`, connect-web's `findTrailerError` sees `code == 0`,
/// treats the error as success, and then throws `[unimplemented] "missing
/// message"`. So on the #51 framing [`connect_web_unary_outcome`] returns
/// [`ConnectWebUnary::MissingMessage`] and this test FAILS; with the
/// `google.rpc.Status` envelope (code set, typed detail packed as an `Any`) it
/// returns [`ConnectWebUnary::TypedError`] and the test PASSES.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grpc_web_error_carries_in_body_trailer_with_detail() {
    use jammi_server::grpc::proto::embedding::{
        encode_query_request::Input, EncodeQueryRequest, Modality,
    };

    let server = super::common::grpc::start_engine_server().await;

    // A nonexistent model drives the EmbeddingService `EncodeQuery` handler to
    // return `JammiError::Model` — an `invalid_argument` Status carrying the
    // structured `grpc-status-details-bin` detail (the detail-bearing path that
    // breaks Connect-ES on `main`).
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
            "http://{}/jammi.v1.embedding.EmbeddingService/EncodeQuery",
            server.addr
        ))
        .header("content-type", "application/grpc-web+proto")
        .header("accept", "application/grpc-web+proto")
        .header("x-grpc-web", "1")
        .header("jammi-session-id", "grpc-web-error")
        .body(body)
        .send()
        .await
        .expect("grpc-web POST");

    // gRPC-Web reports application errors as HTTP 200; the status lives in the
    // trailer frame, not the HTTP status line.
    assert_eq!(response.status(), 200, "gRPC-Web shim returns 200 OK");

    // The status must NOT be hoisted into the HTTP headers — a gRPC-Web client
    // reads it only from the in-body trailer frame. (On `main` it is in the
    // headers with an empty body, which is exactly the bug.)
    assert!(
        response.headers().get("grpc-status").is_none(),
        "grpc-status must not be a trailers-only HTTP header; it belongs in the in-body trailer frame"
    );

    let body_bytes = response.bytes().await.expect("response body");
    assert!(
        !body_bytes.is_empty(),
        "a gRPC-Web error response must carry an in-body trailer frame, not an empty body"
    );

    let (data, trailers) = unframe_grpc_web(&body_bytes);
    assert!(
        data.is_empty(),
        "an error response carries no data message, only the trailer frame; got {} data bytes",
        data.len()
    );
    assert!(
        trailers.contains("grpc-status: 3") || trailers.contains("grpc-status:3"),
        "the in-body trailer frame must carry the engine error's gRPC status (3 = invalid_argument), got {trailers:?}"
    );
    assert!(
        trailers.contains("grpc-message:"),
        "the in-body trailer frame must carry the human-readable grpc-message, got {trailers:?}"
    );
    assert!(
        trailers.contains("grpc-status-details-bin:"),
        "the in-body trailer frame must carry the structured jammi error detail (grpc-status-details-bin) so the typed-error contract survives gRPC-Web, got {trailers:?}"
    );

    // The gap #51 left: replay `@connectrpc/connect-web`'s unary reader over the
    // exact response bytes and assert it surfaces the typed error — not the
    // `[unimplemented] "missing message"` it reports on the #51 framing whose
    // `grpc-status-details-bin` lacks a `code`.
    let outcome = connect_web_unary_outcome(&body_bytes);
    match outcome {
        ConnectWebUnary::TypedError {
            code,
            message,
            detail_type_urls,
        } => {
            assert_eq!(
                code, 3,
                "connect-web must surface the engine's gRPC status (3 = invalid_argument), got {code}"
            );
            assert!(
                message.contains("/does/not/exist"),
                "connect-web must surface the real grpc-message, got {message:?}"
            );
            assert!(
                detail_type_urls
                    .iter()
                    .any(|u| u == "type.googleapis.com/jammi.v1.error.JammiErrorDetail"),
                "the google.rpc.Status envelope must carry the typed JammiErrorDetail as an Any so the contract survives, got {detail_type_urls:?}"
            );
        }
        ConnectWebUnary::MissingMessage => panic!(
            "connect-web reported `missing message` — the #51 residual: \
             grpc-status-details-bin must be a google.rpc.Status with `code` set, \
             not a bare JammiErrorDetail"
        ),
    }

    let _ = server.shutdown.send(());
    let _ = server.handle.await;
}
