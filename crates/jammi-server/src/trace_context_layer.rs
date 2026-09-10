//! Whole-server W3C `traceparent` continuation layer (#486).
//!
//! Sits beside [`crate::metrics_layer::MetricsLayer`] on the SAME combined
//! Flight SQL + gRPC chain (added at the same level — see
//! `runtime.rs`'s serve/axum paths — so it sees every request, engine or
//! downstream-mounted, before routing). Per request it opens ONE span
//! (`"grpc_request"`, carrying the method path) and continues the caller's
//! trace: [`jammi_ai::telemetry::set_parent_from_headers`] extracts the
//! incoming `traceparent`/`tracestate` pair from the request's HTTP headers
//! (the plain `http` crate `Request`, read here BEFORE tonic decodes gRPC
//! metadata from the same headers — reading it is non-destructive, so
//! tonic's own decode downstream is unaffected) and binds it as this span's
//! OpenTelemetry parent, so any span this process exports for the request —
//! this one, or a handler's own `#[tracing::instrument]` span opened while
//! it is entered — carries the SAME trace id the caller's proxy or gateway
//! already assigned. A request with no `traceparent` header extracts an
//! empty context, so the span starts a fresh trace, exactly as before this
//! layer existed.
//!
//! The span is entered around the ENTIRE inner future via
//! [`tracing::Instrument`], not just the synchronous `call()` invocation —
//! `tracing`'s span-entry is otherwise only current for the instant `call`
//! runs, long before the handler's own work (and its own child span) even
//! starts.
//!
//! When `jammi_ai`'s `telemetry-otlp` feature is off, continuation is a
//! no-op ([`jammi_ai::telemetry::set_parent_from_headers`] itself becomes
//! a trivial function under that build) — the span still opens (so
//! `#[tracing::instrument]` handler spans still nest under it identically),
//! it just carries no OTel parent to continue, which is moot: no exporter
//! exists to send it anywhere in that build either.

use tonic::codegen::http::{Request, Response};
use tonic::codegen::Service;
use tower::Layer;
use tracing::Instrument;

/// [`Layer`] that installs [`TraceContext`]. Add it to the tonic
/// `Server::builder()` chain (or the axum layer stack) beside
/// [`crate::metrics_layer::MetricsLayer`].
#[derive(Clone, Default)]
pub struct TraceContextLayer;

impl TraceContextLayer {
    pub fn new() -> Self {
        Self
    }
}

impl<S> Layer<S> for TraceContextLayer {
    type Service = TraceContext<S>;

    fn layer(&self, inner: S) -> Self::Service {
        TraceContext { inner }
    }
}

/// Service that opens one span per request, continuing the incoming W3C
/// trace context, and instruments the inner service's whole response future
/// with it (see module docs).
#[derive(Clone)]
pub struct TraceContext<S> {
    inner: S,
}

impl<S, ReqBody, ResBody> Service<Request<ReqBody>> for TraceContext<S>
where
    S: Service<Request<ReqBody>, Response = Response<ResBody>>,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = tracing::instrument::Instrumented<S::Future>;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<ReqBody>) -> Self::Future {
        let span = tracing::info_span!("grpc_request", path = %req.uri().path());
        // `jammi_ai::telemetry::set_parent_from_headers` exists only behind
        // `jammi-ai`'s `telemetry-otlp` feature. `jammi-server`'s own
        // same-named feature (default-on) forwards to it, but a
        // `--no-default-features` build of this crate must still compile —
        // the span still opens either way, it simply carries no OTel
        // parent to continue when this arm is compiled out (moot: no
        // exporter exists to send it anywhere in that build).
        #[cfg(feature = "telemetry-otlp")]
        jammi_ai::telemetry::set_parent_from_headers(&span, req.headers());
        self.inner.call(req).instrument(span)
    }
}
