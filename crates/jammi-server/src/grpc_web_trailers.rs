//! gRPC-Web trailers-only repair layer.
//!
//! tonic emits an error returned by a unary handler as a **Trailers-Only**
//! HTTP response: `grpc-status` / `grpc-message` / `grpc-status-details-bin`
//! live in the HTTP response *headers* and the body is empty (the
//! `status.into_http()` path reached by `Grpc::map_response`'s `t!` macro).
//! [`tonic_web::GrpcWebLayer`] only rewrites trailer *frames that already sit
//! in the body* into the in-body `0x80`-flagged frame a gRPC-Web client reads;
//! it never inspects the response headers, so a Trailers-Only response reaches
//! the client with an empty body and the status only in headers.
//!
//! The gRPC-Web protocol requires the status to ride in an in-body trailer
//! frame even for a trailers-only response (PROTOCOL-WEB.md: "trailers may be
//! sent together with response headers, with no message in the body" describes
//! the *wire-format* a compliant client must still read in-body). Connect-ES
//! enforces this strictly and has no opt-out, so without the in-body frame it
//! reports `ConnectError [unimplemented] "missing message"` for every
//! detail-bearing engine error — the structured [`jammi_ai::wire`] detail is
//! lost to the TypeScript / Workers edge.
//!
//! This layer only moves the status into the body; the *content* of the
//! `grpc-status-details-bin` trailer must independently be a spec-compliant
//! `google.rpc.Status` envelope (built in [`jammi_ai::wire`]) for a gRPC-web
//! client to surface the typed error rather than read `code == 0` and still
//! fail the unary read.
//!
//! This layer sits *outside* [`tonic_web::GrpcWebLayer`] (added first in the
//! `ServiceBuilder` chain so it post-processes the gRPC-Web-framed response).
//! When it sees a gRPC-Web response carrying `grpc-status` in the headers, it
//! moves the gRPC trailer headers into a single in-body `0x80` trailer frame
//! and strips them from the HTTP headers — producing exactly the framing a
//! gRPC-Web client decodes. The error detail (`grpc-status-details-bin`) rides
//! the frame intact, so the typed-error contract crosses the wire faithfully.
//!
//! It is a no-op for every other response:
//!
//! * non-gRPC-Web responses (raw gRPC over HTTP/2, Flight SQL, health) — the
//!   content-type gate skips them, so the raw-gRPC `RemoteSession` path is
//!   untouched (an HTTP/2 client reads the status from real trailers natively).
//! * gRPC-Web *data* responses and mid-stream errors — these already carry an
//!   in-body trailer frame from `GrpcWebLayer`; they have no `grpc-status`
//!   header, so the gate skips them.

use std::pin::Pin;
use std::task::{ready, Context, Poll};

use bytes::{BufMut, Bytes, BytesMut};
use http_body::{Frame, SizeHint};
use pin_project::pin_project;
use tonic::codegen::http::{header::CONTENT_TYPE, HeaderMap, HeaderName, Request, Response};
use tonic::codegen::{Body, Service};
use tonic::Status;
use tower::Layer;

/// gRPC-Web trailer frame flag byte (high bit set): marks the length-prefixed
/// frame that carries the HTTP/1-style trailer block, per PROTOCOL-WEB.md.
const GRPC_WEB_TRAILERS_BIT: u8 = 0x80;
/// 1 flag byte + 4 big-endian length bytes precede every gRPC-Web frame.
const FRAME_HEADER_SIZE: usize = 5;

/// Headers that describe the *HTTP framing* rather than gRPC trailer metadata.
/// They stay on the response and never enter the in-body trailer frame.
fn is_framing_header(name: &HeaderName) -> bool {
    name == CONTENT_TYPE || name == tonic::codegen::http::header::CONTENT_LENGTH
}

/// True when `headers` describe a gRPC-Web trailers-only response: a gRPC-Web
/// content-type carrying the terminal `grpc-status` in the HTTP headers. A
/// gRPC-Web *data* response (or a mid-stream error) carries its status in an
/// in-body trailer frame instead, so it has no `grpc-status` header and is not
/// matched here.
fn is_grpc_web_trailers_only(headers: &HeaderMap) -> bool {
    let is_grpc_web = headers
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with("application/grpc-web"));
    is_grpc_web && headers.contains_key(Status::GRPC_STATUS)
}

/// Build the gRPC-Web in-body trailer frame from the gRPC trailer headers.
/// Mirrors `tonic_web`'s own `make_trailers_frame`: a `0x80` flag, a
/// big-endian u32 length, then `name:value\r\n` for each gRPC trailer header
/// (framing headers excluded). The frame is the exact shape a gRPC-Web client
/// decodes as the response trailers.
fn trailers_frame(headers: &HeaderMap) -> Bytes {
    let mut block = Vec::new();
    for (name, value) in headers.iter() {
        if is_framing_header(name) {
            continue;
        }
        block.put_slice(name.as_ref());
        block.push(b':');
        block.put_slice(value.as_bytes());
        block.put_slice(b"\r\n");
    }

    let mut frame = BytesMut::with_capacity(FRAME_HEADER_SIZE + block.len());
    frame.put_u8(GRPC_WEB_TRAILERS_BIT);
    // A trailers-only response's header block is far below u32::MAX; the cast
    // is exact for any realistic status + detail payload.
    frame.put_u32(block.len() as u32);
    frame.put_slice(&block);
    frame.freeze()
}

/// [`Layer`] that installs [`GrpcWebTrailers`]. Add it to the tonic
/// `Server::builder()` chain *before* `GrpcWebLayer` so it post-processes the
/// gRPC-Web-framed response.
#[derive(Debug, Clone, Default)]
pub struct GrpcWebTrailersLayer;

impl GrpcWebTrailersLayer {
    pub fn new() -> Self {
        Self
    }
}

impl<S> Layer<S> for GrpcWebTrailersLayer {
    type Service = GrpcWebTrailers<S>;

    fn layer(&self, inner: S) -> Self::Service {
        GrpcWebTrailers { inner }
    }
}

/// Service that repairs gRPC-Web trailers-only responses (see module docs).
#[derive(Debug, Clone)]
pub struct GrpcWebTrailers<S> {
    inner: S,
}

impl<S, ReqBody, ResBody> Service<Request<ReqBody>> for GrpcWebTrailers<S>
where
    S: Service<Request<ReqBody>, Response = Response<ResBody>>,
    ResBody: Body<Data = Bytes>,
{
    type Response = Response<RepairedBody<ResBody>>;
    type Error = S::Error;
    type Future = ResponseFuture<S::Future>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<ReqBody>) -> Self::Future {
        ResponseFuture {
            inner: self.inner.call(req),
        }
    }
}

/// Future that rewrites a trailers-only gRPC-Web response into an in-body
/// trailer frame once the inner service resolves.
#[pin_project]
pub struct ResponseFuture<F> {
    #[pin]
    inner: F,
}

impl<F, ResBody, E> std::future::Future for ResponseFuture<F>
where
    F: std::future::Future<Output = Result<Response<ResBody>, E>>,
    ResBody: Body<Data = Bytes>,
{
    type Output = Result<Response<RepairedBody<ResBody>>, E>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let response = ready!(self.project().inner.poll(cx))?;
        Poll::Ready(Ok(repair(response)))
    }
}

/// Move the gRPC trailer headers of a trailers-only gRPC-Web response into a
/// single in-body trailer frame, leaving every other response untouched.
fn repair<B>(response: Response<B>) -> Response<RepairedBody<B>>
where
    B: Body<Data = Bytes>,
{
    let (mut parts, body) = response.into_parts();
    if !is_grpc_web_trailers_only(&parts.headers) {
        return Response::from_parts(parts, RepairedBody::passthrough(body));
    }

    let frame = trailers_frame(&parts.headers);
    // The status now lives in the in-body frame. Rebuild the header map with
    // only the framing headers so the client reads the terminal status from one
    // place — the trailer frame. `HeaderMap` has no `retain`; draining into a
    // fresh map is the total, allocation-bounded way to drop the gRPC trailer
    // headers (and any custom metadata) while keeping the framing ones.
    let mut kept = HeaderMap::with_capacity(parts.headers.len());
    for (name, value) in parts.headers.drain() {
        if let Some(name) = name {
            if is_framing_header(&name) {
                kept.insert(name, value);
            }
        }
    }
    // The body now carries exactly one frame; drop the now-stale length so
    // hyper frames the body from the stream rather than the `0` we received.
    kept.remove(tonic::codegen::http::header::CONTENT_LENGTH);
    parts.headers = kept;

    Response::from_parts(parts, RepairedBody::trailers_frame(frame))
}

/// Response body for [`GrpcWebTrailers`]. Either passes the inner body through
/// unchanged, or yields a single synthesized gRPC-Web trailer frame (for a
/// repaired trailers-only response — the original body was empty).
#[pin_project(project = RepairedBodyProj)]
pub enum RepairedBody<B> {
    Passthrough(#[pin] B),
    TrailersFrame(Option<Bytes>),
}

impl<B> RepairedBody<B> {
    fn passthrough(body: B) -> Self {
        RepairedBody::Passthrough(body)
    }

    fn trailers_frame(frame: Bytes) -> Self {
        RepairedBody::TrailersFrame(Some(frame))
    }
}

impl<B> Body for RepairedBody<B>
where
    B: Body<Data = Bytes>,
{
    type Data = Bytes;
    type Error = B::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        match self.project() {
            RepairedBodyProj::Passthrough(body) => body.poll_frame(cx),
            RepairedBodyProj::TrailersFrame(frame) => {
                Poll::Ready(frame.take().map(|f| Ok(Frame::data(f))))
            }
        }
    }

    fn is_end_stream(&self) -> bool {
        match self {
            RepairedBody::Passthrough(body) => body.is_end_stream(),
            RepairedBody::TrailersFrame(frame) => frame.is_none(),
        }
    }

    fn size_hint(&self) -> SizeHint {
        match self {
            RepairedBody::Passthrough(body) => body.size_hint(),
            RepairedBody::TrailersFrame(Some(frame)) => SizeHint::with_exact(frame.len() as u64),
            RepairedBody::TrailersFrame(None) => SizeHint::with_exact(0),
        }
    }
}
