//! Whole-server metrics layer.
//!
//! A single [`tower::Layer`] wraps the combined Flight SQL + gRPC chain (added
//! at the same level as the gRPC-Web layers, so it sees every request before
//! routing). Per request it inspects `req.uri().path()` and drives the four
//! substrate-level [`MetricsRegistry`] metrics from one place:
//!
//! * any `/jammi.v1.*` gRPC method increments `grpc_requests`;
//! * `/arrow.flight.protocol.FlightService/DoGet` increments `flight_queries`
//!   — DoGet is the data-fetch leg of a Flight SQL query, counted once at the
//!   HTTP layer (the Flight provider's `new_context` runs ≥2× per query, so
//!   counting there would over-count);
//! * `/jammi.v1.eval.EvalService/*` increments `eval_invocations`;
//! * `/jammi.v1.embedding.EmbeddingService/Search` additionally times the
//!   request future and records the elapsed end-to-end latency on
//!   `search_latency` when the response resolves.
//!
//! Counting at the whole-server layer (rather than per service / per
//! interceptor) keeps all four metrics live at one site with no threading into
//! the individual services — the layer holds the shared registry directly. The
//! `search_latency` observation spans deserialize + `Session::search` +
//! serialize, i.e. the user-meaningful end-to-end Search latency.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{ready, Context, Poll};
use std::time::Instant;

use pin_project::pin_project;
use tonic::codegen::http::{Request, Response};
use tonic::codegen::Service;
use tower::Layer;

use crate::routes::health::MetricsRegistry;

/// gRPC method-path prefix shared by every `jammi.v1.*` service.
const JAMMI_GRPC_PREFIX: &str = "/jammi.v1.";
/// Flight SQL data-fetch method path — the leg of a query that streams result
/// batches back, counted once per query.
const FLIGHT_DO_GET_PATH: &str = "/arrow.flight.protocol.FlightService/DoGet";
/// `EvalService` method-path prefix — any eval RPC.
const EVAL_PREFIX: &str = "/jammi.v1.eval.EvalService/";
/// The single embedding-search method path whose latency is observed.
const EMBEDDING_SEARCH_PATH: &str = "/jammi.v1.embedding.EmbeddingService/Search";

/// [`Layer`] that installs [`Metrics`]. Add it to the tonic `Server::builder()`
/// chain so it observes every request to the combined Flight + gRPC surface.
#[derive(Clone)]
pub struct MetricsLayer {
    registry: Arc<MetricsRegistry>,
}

impl MetricsLayer {
    pub fn new(registry: Arc<MetricsRegistry>) -> Self {
        Self { registry }
    }
}

impl<S> Layer<S> for MetricsLayer {
    type Service = Metrics<S>;

    fn layer(&self, inner: S) -> Self::Service {
        Metrics {
            inner,
            registry: Arc::clone(&self.registry),
        }
    }
}

/// Service that records the substrate-level metrics per request (see module
/// docs).
#[derive(Clone)]
pub struct Metrics<S> {
    inner: S,
    registry: Arc<MetricsRegistry>,
}

impl<S, ReqBody, ResBody> Service<Request<ReqBody>> for Metrics<S>
where
    S: Service<Request<ReqBody>, Response = Response<ResBody>>,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = MetricsFuture<S::Future>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<ReqBody>) -> Self::Future {
        let path = req.uri().path();

        if path.starts_with(JAMMI_GRPC_PREFIX) {
            self.registry.grpc_requests.inc();
        }
        if path == FLIGHT_DO_GET_PATH {
            self.registry.flight_queries.inc();
        }
        if path.starts_with(EVAL_PREFIX) {
            self.registry.eval_invocations.inc();
        }

        // Time the Search RPC end-to-end: start the clock here and observe on
        // the response future's completion. Every other path carries no timer.
        let timer =
            (path == EMBEDDING_SEARCH_PATH).then(|| (Instant::now(), Arc::clone(&self.registry)));

        MetricsFuture {
            inner: self.inner.call(req),
            timer,
        }
    }
}

/// Future that records the `search_latency` observation when the timed Search
/// response resolves, and otherwise forwards the inner response untouched.
#[pin_project]
pub struct MetricsFuture<F> {
    #[pin]
    inner: F,
    /// `Some` only for the Search path: the start instant plus the registry to
    /// observe on. The `Option` is `take`n on completion so the observation
    /// fires exactly once.
    timer: Option<(Instant, Arc<MetricsRegistry>)>,
}

impl<F, ResBody, E> std::future::Future for MetricsFuture<F>
where
    F: std::future::Future<Output = Result<Response<ResBody>, E>>,
{
    type Output = Result<Response<ResBody>, E>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        let response = ready!(this.inner.poll(cx));
        if let Some((start, registry)) = this.timer.take() {
            registry
                .search_latency
                .observe(start.elapsed().as_secs_f64());
        }
        Poll::Ready(response)
    }
}
