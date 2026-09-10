//! `[server.limits]` request-bounds enforcement — the tower layer stack that
//! refuses an inbound request at the edge, before any tenant-scoped catalog
//! read runs (so a refusal never leaks cross-tenant existence — see
//! [`crate::grpc::session`]'s per-verb tenant scoping for the read side of
//! that invariant), whenever it would exceed the deployment's configured
//! bounds ([`jammi_db::config::LimitsConfig`]).
//!
//! # Layer stack and ordering
//!
//! [`crate::runtime::BoundChain::serve_with_shutdown`] applies this module's
//! layers on the SAME tonic `Server::builder()` chain the existing gRPC-Web
//! framing layers use, positioned AFTER [`tonic_web::GrpcWebLayer`] (closer
//! to the routes) — outermost first:
//!
//! [`RefusalStatusLayer`] → [`GlobalConcurrencyLimitLayer`] →
//! [`PerConnectionLimitLayer`] → [`MethodClassLayer`] → routes
//!
//! Because [`crate::metrics_layer::MetricsLayer`] and the gRPC-Web layers sit
//! OUTSIDE (more outer than) every layer here, a refusal built by any layer
//! in this module is an ordinary [`tonic::Status`] HTTP response
//! (`status.into_http()` shape) and is therefore re-framed for a gRPC-Web
//! client by [`crate::grpc_web_trailers::GrpcWebTrailersLayer`] +
//! [`tonic_web::GrpcWebLayer`] on its way back out, with no gRPC-Web-specific
//! code needed in this module at all.
//!
//! # N4 — why NOT tonic's own `concurrency_limit_per_connection`/`load_shed`
//!
//! tonic's `Server::builder().concurrency_limit_per_connection(n)` and
//! `.load_shed(true)` are deliberately never used here. Per
//! `tonic-0.14.5/src/transport/server/mod.rs`'s `MakeSvc::call` (around
//! lines 1227–1250, invoked ONCE per accepted TCP connection): the value
//! passed to `Server::builder().layer(...)` (`self.inner`, i.e. the
//! already-fully-layered service — `MetricsLayer` + the gRPC-Web layers +
//! whatever this module adds) is wrapped by
//! `ServiceBuilder::new().layer(RecoverErrorLayer::new())
//! .option_layer(self.load_shed.then_some(LoadShedLayer::new()))
//! .option_layer(concurrency_limit.map(ConcurrencyLimitLayer::new))
//! .layer_fn(|s| GrpcTimeout::new(s, timeout)).service(svc)` — i.e. tonic's
//! OWN per-connection concurrency limit and load-shed sit OUTSIDE every
//! `.layer()` call a caller of `Server::builder()` makes, including
//! `MetricsLayer` and the gRPC-Web framing. A refusal from THOSE builder
//! knobs would therefore be uncounted (never reaches
//! [`RefusalStatusLayer`]) and never gRPC-Web-framed (never reaches
//! [`tonic_web::GrpcWebLayer`]) — silently invisible to both the metrics
//! surface and a gRPC-Web client. [`PerConnectionLimitLayer`] is instead an
//! ordinary user-space `.layer()` call, positioned INSIDE the existing
//! stack, so its refusals get both.
//!
//! # N5 — the `message_size` counting rule
//!
//! Message-size enforcement is NOT a layer in this module at all: every
//! mounted service is constructed with tonic's own
//! `.max_decoding_message_size(limits.max_message_bytes)` (see
//! `crate::runtime::assemble_grpc_chain`), and tonic's generated codec
//! itself refuses an inbound message that exceeds that cap — BELOW every
//! layer in this module, so no application code ever constructs that
//! specific refusal. Verified against the vendored source
//! (`tonic-0.14.5/src/codec/decode.rs:185–195`, `Decoder::decode`): that
//! rejection is `Status::out_of_range` (`OUT_OF_RANGE`, code 11) — NOT
//! `RESOURCE_EXHAUSTED` as an unverified reading of the wire semantics might
//! suggest; every OTHER budget this module itself enforces (in-flight,
//! per-connection, the two stream budgets) legitimately uses
//! `RESOURCE_EXHAUSTED`, since those genuinely are "the server is out of
//! capacity", whereas message-size is "this specific request violates a
//! fixed limit" — the distinction tonic's own codec already draws.
//! [`RefusalStatusLayer`] closes the loop: a response carrying
//! `grpc-status: OUT_OF_RANGE` with NO [`RefusedBound`] extension is counted
//! under [`MESSAGE_SIZE_LABEL`] — the only thing in this whole stack able to
//! produce that combination is tonic's own codec, since every
//! `RESOURCE_EXHAUSTED` or `DEADLINE_EXCEEDED` this module itself produces
//! always stamps a [`RefusedBound`] extension first.
//!
//! # Streaming-path exemption
//!
//! `JobService.WaitJob` and `TriggerService.Subscribe` are long-lived
//! server-streaming RPCs; [`is_streaming_path`] exempts them from the UNARY
//! bounds ([`jammi_db::config::LimitsConfig::max_in_flight`],
//! `max_in_flight_per_connection`, `request_timeout_secs`) and instead
//! governs them through [`jammi_db::config::LimitsConfig::wait_timeout_secs`]
//! and their own per-RPC stream budget (`max_subscriptions` / `max_job_waits`,
//! released when the stream ends or the connection drops — [`PermitBody`]).
//!
//! `wait_timeout_secs`, when configured, bounds a stream in one of three ways
//! depending on what the caller's `grpc-timeout` header declares — the SERVER
//! budget bounds the stream; the client imposes no deadline of its own by
//! default (see `jammi_client::wait_job`/`subscribe`):
//!
//! * a `grpc-timeout` header ABOVE the budget is refused at the edge, before
//!   the stream ever opens (`DEADLINE_EXCEEDED`, [`RefusedBound::Timeout`]) —
//!   the caller asked for more than the deployment allows.
//! * a `grpc-timeout` header WITHIN the budget is honoured as-is: no
//!   additional server-side deadline is layered on top of the caller's own
//!   declared one.
//! * NO `grpc-timeout` header at all (HTTP/2's own no-deadline default — the
//!   shape a header-less client, e.g. a Python-shaped one with no explicit
//!   timeout, sends) is NOT refused. Instead the configured budget itself
//!   becomes the stream's deadline: [`PermitBody::Deadlined`] races the
//!   response body against a timer armed for the budget, and once it elapses
//!   with the stream still open, closes it with a `DEADLINE_EXCEEDED` trailer
//!   — the stream ends at the budget, not at open, and never runs unbounded.
//!
//! [`is_streaming_path`] is a hardcoded two-path allowlist rather than a
//! path→class map derived from the compiled `FILE_DESCRIPTOR_SET` (the
//! literal PLAN-C §5 design): this codebase mounts exactly two
//! server-streaming RPCs today, so the derived map's only observable
//! behaviour over THIS binary is this same two-path set. This is a
//! documented, deliberate scope reduction — a third server-streaming RPC
//! added later needs this list extended by hand, and no test today catches
//! that omission.

use std::collections::HashMap;
use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{ready, Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use http_body::{Body, Frame, SizeHint};
use jammi_db::config::LimitsConfig;
use jammi_db::error::JammiError;
use pin_project::pin_project;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::Sleep;
use tonic::codegen::http::{HeaderMap, Request, Response};
use tonic::codegen::Service;
use tonic::transport::server::TcpConnectInfo;
use tonic::{Code, Status};
use tower::Layer;

use crate::routes::health::MetricsRegistry;

/// `JobService.WaitJob`'s wire path — server-streaming; see the module docs'
/// "Streaming-path exemption" section.
pub const WAIT_JOB_PATH: &str = "/jammi.v1.job.JobService/WaitJob";
/// `TriggerService.Subscribe`'s wire path — the SAME exemption as
/// [`WAIT_JOB_PATH`].
pub const SUBSCRIBE_PATH: &str = "/jammi.v1.trigger.TriggerService/Subscribe";

/// True for a path this stack treats as a long-lived server-streaming RPC.
/// See the module docs' "Streaming-path exemption" section.
pub fn is_streaming_path(path: &str) -> bool {
    path == WAIT_JOB_PATH || path == SUBSCRIBE_PATH
}

/// Which budget refused a request — attached to a synthesized refusal
/// [`Response`]'s extensions by every layer in this module except
/// [`RefusalStatusLayer`] itself. See the module docs' N5 section for the
/// one case with no extension (`message_size`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefusedBound {
    InFlight,
    InFlightPerConnection,
    Subscriptions,
    JobWaits,
    Timeout,
}

impl RefusedBound {
    /// The `jammi_grpc_refused_total{reason}` label this bound counts under.
    pub const fn label(self) -> &'static str {
        match self {
            RefusedBound::InFlight => "in_flight",
            RefusedBound::InFlightPerConnection => "in_flight_per_connection",
            RefusedBound::Subscriptions => "subscriptions",
            RefusedBound::JobWaits => "job_waits",
            RefusedBound::Timeout => "timeout",
        }
    }
}

/// The label [`RefusalStatusLayer`] counts an unlabeled `RESOURCE_EXHAUSTED`
/// response under — see the module docs' N5 section.
pub const MESSAGE_SIZE_LABEL: &str = "message_size";

/// Build a refused [`Response`]: a [`tonic::Status`] carrying `code` and
/// `message`, with the typed wire detail attached via
/// [`jammi_wire::attach_error_detail`] — reusing [`JammiError::Config`] to
/// carry the message. A limits refusal has no dedicated `JammiError` variant
/// of its own (widening that shared enum for a mechanism-only wire
/// annotation is outside this module's reach); the client-observable `code`,
/// `message`, and `bound` label are what a caller actually reads and matches
/// on, and those cross the wire faithfully either way. The response is then
/// stamped with `bound` so [`RefusalStatusLayer`] counts it under the right
/// label.
fn refuse<ResBody: Default>(code: Code, message: String, bound: RefusedBound) -> Response<ResBody> {
    let status = limits_status(code, message);
    let mut response = status.into_http::<ResBody>();
    response.extensions_mut().insert(bound);
    response
}

/// Build the [`Status`] a limits refusal or timeout carries, with the typed
/// wire detail attached exactly like every other refusal in this module
/// (reusing [`JammiError::Config`] — see [`refuse`]'s doc for why). Shared so
/// [`refuse`]'s edge refusal and [`PermitBody::Deadlined`]'s mid-stream
/// timeout trailer both carry the SAME typed annotation a decoding client
/// reads identically whichever path fired.
fn limits_status(code: Code, message: String) -> Status {
    let engine_err = JammiError::Config(message.clone());
    jammi_wire::attach_error_detail(code, message, &engine_err)
}

/// The message [`PermitBody::Deadlined`] closes a header-less stream with once
/// `server.limits.wait_timeout_secs` elapses — mirrors the edge-refusal
/// message's wording (`server.limits.wait_timeout_secs (Ns)`) so a client sees
/// the same budget cited whether it was refused up front (an over-budget
/// header) or timed out mid-stream (no header at all).
fn deadline_message(cap: Duration) -> String {
    format!(
        "server.limits.wait_timeout_secs ({}s) elapsed with no grpc-timeout \
         header bounding this stream — the configured budget became this \
         stream's deadline",
        cap.as_secs()
    )
}

// ─── RefusalStatusLayer ─────────────────────────────────────────────────

/// The single place `jammi_grpc_refused_total{reason}` is incremented. See
/// the module docs for its position in the layer stack and the counting
/// rule.
#[derive(Clone)]
pub struct RefusalStatusLayer {
    metrics: Arc<MetricsRegistry>,
}

impl RefusalStatusLayer {
    pub fn new(metrics: Arc<MetricsRegistry>) -> Self {
        Self { metrics }
    }
}

impl<S> Layer<S> for RefusalStatusLayer {
    type Service = RefusalStatus<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RefusalStatus {
            inner,
            metrics: Arc::clone(&self.metrics),
        }
    }
}

#[derive(Clone)]
pub struct RefusalStatus<S> {
    inner: S,
    metrics: Arc<MetricsRegistry>,
}

impl<S> tonic::server::NamedService for RefusalStatus<S>
where
    S: tonic::server::NamedService,
{
    const NAME: &'static str = S::NAME;
}

impl<S, ReqBody, ResBody> Service<Request<ReqBody>> for RefusalStatus<S>
where
    S: Service<Request<ReqBody>, Response = Response<ResBody>>,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = RefusalStatusFuture<S::Future>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<ReqBody>) -> Self::Future {
        RefusalStatusFuture {
            inner: self.inner.call(req),
            metrics: Arc::clone(&self.metrics),
        }
    }
}

#[pin_project]
pub struct RefusalStatusFuture<F> {
    #[pin]
    inner: F,
    metrics: Arc<MetricsRegistry>,
}

impl<F, ResBody, E> Future for RefusalStatusFuture<F>
where
    F: Future<Output = Result<Response<ResBody>, E>>,
{
    type Output = Result<Response<ResBody>, E>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        let response = ready!(this.inner.poll(cx));
        if let Ok(response) = &response {
            if let Some(bound) = response.extensions().get::<RefusedBound>() {
                this.metrics.record_refusal(bound.label());
            } else if let Some(status) = tonic::Status::from_header_map(response.headers()) {
                if status.code() == Code::OutOfRange {
                    this.metrics.record_refusal(MESSAGE_SIZE_LABEL);
                }
            }
        }
        Poll::Ready(response)
    }
}

// ─── GlobalConcurrencyLimitLayer ────────────────────────────────────────

/// Global cap on UNARY requests in flight across every connection
/// ([`LimitsConfig::max_in_flight`]). `0` means unbounded — no semaphore is
/// constructed, and every wrapped request passes straight through. A
/// streaming path ([`is_streaming_path`]) always bypasses this bound; see
/// the module docs.
#[derive(Clone)]
pub struct GlobalConcurrencyLimitLayer {
    semaphore: Option<Arc<Semaphore>>,
}

impl GlobalConcurrencyLimitLayer {
    pub fn new(max_in_flight: usize) -> Self {
        Self {
            semaphore: (max_in_flight != 0).then(|| Arc::new(Semaphore::new(max_in_flight))),
        }
    }
}

impl<S> Layer<S> for GlobalConcurrencyLimitLayer {
    type Service = GlobalConcurrencyLimit<S>;

    fn layer(&self, inner: S) -> Self::Service {
        GlobalConcurrencyLimit {
            inner,
            semaphore: self.semaphore.clone(),
        }
    }
}

#[derive(Clone)]
pub struct GlobalConcurrencyLimit<S> {
    inner: S,
    semaphore: Option<Arc<Semaphore>>,
}

impl<S> tonic::server::NamedService for GlobalConcurrencyLimit<S>
where
    S: tonic::server::NamedService,
{
    const NAME: &'static str = S::NAME;
}

impl<S, ReqBody, ResBody> Service<Request<ReqBody>> for GlobalConcurrencyLimit<S>
where
    S: Service<Request<ReqBody>, Response = Response<ResBody>> + Clone + Send + 'static,
    S::Future: Send + 'static,
    ReqBody: Send + 'static,
    ResBody: Default,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<ReqBody>) -> Self::Future {
        let bound_active = self.semaphore.is_some() && !is_streaming_path(req.uri().path());
        if !bound_active {
            let clone = self.inner.clone();
            let mut inner = std::mem::replace(&mut self.inner, clone);
            return Box::pin(async move { inner.call(req).await });
        }
        let sem = Arc::clone(self.semaphore.as_ref().expect("bound_active implies Some"));
        match sem.try_acquire_owned() {
            Ok(permit) => {
                let clone = self.inner.clone();
                let mut inner = std::mem::replace(&mut self.inner, clone);
                Box::pin(async move {
                    let response = inner.call(req).await;
                    drop(permit);
                    response
                })
            }
            Err(_) => Box::pin(async move {
                Ok(refuse(
                    Code::ResourceExhausted,
                    "server-wide in-flight request limit reached (server.limits.max_in_flight)"
                        .to_string(),
                    RefusedBound::InFlight,
                ))
            }),
        }
    }
}

// ─── PerConnectionLimitLayer ────────────────────────────────────────────

/// Cap on UNARY requests in flight on a SINGLE TCP connection
/// ([`LimitsConfig::max_in_flight_per_connection`]), keyed on the
/// connection's remote address read from tonic's own connect-info request
/// extension ([`TcpConnectInfo`], inserted unconditionally by tonic's
/// `MakeSvc` per accepted connection — see the module docs' N4 section for
/// why this is a user-space layer rather than tonic's own builder knob).
///
/// `0` means unbounded. A connection with no [`TcpConnectInfo`] extension
/// (a composability seam that did not go through tonic's own TCP accept
/// loop — e.g. the layer-free `into_axum_router` split nested under a
/// foreign listener) has no per-peer identity to key on, so this bound is a
/// no-op for that request rather than a spurious refusal.
///
/// The per-peer semaphore map is swept for fully-idle entries (available
/// permits == the configured limit, i.e. currently gating zero in-flight
/// requests) on every new-peer insert — bounded, amortised idle eviction
/// with no background sweep task.
#[derive(Clone)]
pub struct PerConnectionLimitLayer {
    limit: usize,
    peers: Arc<Mutex<HashMap<SocketAddr, Arc<Semaphore>>>>,
}

impl PerConnectionLimitLayer {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            peers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn semaphore_for(&self, addr: SocketAddr) -> Arc<Semaphore> {
        let mut peers = self.peers.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(sem) = peers.get(&addr) {
            return Arc::clone(sem);
        }
        // Idle eviction: drop every OTHER peer entry that is fully idle
        // before inserting this one, so the map cannot grow unbounded over
        // a long-lived server's connection churn.
        let limit = self.limit;
        peers.retain(|_, sem| sem.available_permits() != limit);
        let sem = Arc::new(Semaphore::new(self.limit));
        peers.insert(addr, Arc::clone(&sem));
        sem
    }
}

impl<S> Layer<S> for PerConnectionLimitLayer {
    type Service = PerConnectionLimit<S>;

    fn layer(&self, inner: S) -> Self::Service {
        PerConnectionLimit {
            inner,
            layer: self.clone(),
        }
    }
}

#[derive(Clone)]
pub struct PerConnectionLimit<S> {
    inner: S,
    layer: PerConnectionLimitLayer,
}

impl<S> tonic::server::NamedService for PerConnectionLimit<S>
where
    S: tonic::server::NamedService,
{
    const NAME: &'static str = S::NAME;
}

impl<S, ReqBody, ResBody> Service<Request<ReqBody>> for PerConnectionLimit<S>
where
    S: Service<Request<ReqBody>, Response = Response<ResBody>> + Clone + Send + 'static,
    S::Future: Send + 'static,
    ReqBody: Send + 'static,
    ResBody: Default,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<ReqBody>) -> Self::Future {
        let addr = if self.layer.limit == 0 || is_streaming_path(req.uri().path()) {
            None
        } else {
            req.extensions()
                .get::<TcpConnectInfo>()
                .and_then(|i| i.remote_addr())
        };

        let Some(addr) = addr else {
            let clone = self.inner.clone();
            let mut inner = std::mem::replace(&mut self.inner, clone);
            return Box::pin(async move { inner.call(req).await });
        };

        let sem = self.layer.semaphore_for(addr);
        match sem.try_acquire_owned() {
            Ok(permit) => {
                let clone = self.inner.clone();
                let mut inner = std::mem::replace(&mut self.inner, clone);
                Box::pin(async move {
                    let response = inner.call(req).await;
                    drop(permit);
                    response
                })
            }
            Err(_) => Box::pin(async move {
                Ok(refuse(
                    Code::ResourceExhausted,
                    "per-connection in-flight request limit reached \
                     (server.limits.max_in_flight_per_connection)"
                        .to_string(),
                    RefusedBound::InFlightPerConnection,
                ))
            }),
        }
    }
}

// ─── MethodClassLayer ────────────────────────────────────────────────────

/// The gRPC `grpc-timeout` header, standard across every implementation
/// (ASCII digits, 1–8 of them, then a unit char: `H`/`M`/`S`/`m`/`u`/`n`).
/// Parsing it locally (rather than depending on tonic's own, private
/// `GrpcTimeout`) is a small, bounded grammar.
fn parse_grpc_timeout(headers: &HeaderMap) -> Option<Duration> {
    let value = headers.get("grpc-timeout")?.to_str().ok()?;
    if value.is_empty() || value.len() > 9 {
        return None;
    }
    let (digits, unit) = value.split_at(value.len() - 1);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let amount: u64 = digits.parse().ok()?;
    let per_unit_nanos: u64 = match unit {
        "H" => 3_600_000_000_000,
        "M" => 60_000_000_000,
        "S" => 1_000_000_000,
        "m" => 1_000_000,
        "u" => 1_000,
        "n" => 1,
        _ => return None,
    };
    Some(Duration::from_nanos(amount.saturating_mul(per_unit_nanos)))
}

/// Unary `request_timeout_secs` and the two streaming budgets
/// (`max_subscriptions` / `max_job_waits`) plus `wait_timeout_secs`'s
/// edge-refusal, in one layer — see the module docs' "Streaming-path
/// exemption" section for why the two RPC classes are handled together
/// rather than as separate layers.
#[derive(Clone)]
pub struct MethodClassLayer {
    request_timeout: Option<Duration>,
    wait_timeout: Option<Duration>,
    subscriptions: Option<Arc<Semaphore>>,
    job_waits: Option<Arc<Semaphore>>,
}

impl MethodClassLayer {
    pub fn new(limits: &LimitsConfig) -> Self {
        Self {
            request_timeout: limits.request_timeout_secs.map(Duration::from_secs),
            wait_timeout: limits.wait_timeout_secs.map(Duration::from_secs),
            subscriptions: (limits.max_subscriptions != 0)
                .then(|| Arc::new(Semaphore::new(limits.max_subscriptions))),
            job_waits: (limits.max_job_waits != 0)
                .then(|| Arc::new(Semaphore::new(limits.max_job_waits))),
        }
    }
}

impl<S> Layer<S> for MethodClassLayer {
    type Service = MethodClass<S>;

    fn layer(&self, inner: S) -> Self::Service {
        MethodClass {
            inner,
            layer: self.clone(),
        }
    }
}

#[derive(Clone)]
pub struct MethodClass<S> {
    inner: S,
    layer: MethodClassLayer,
}

impl<S> tonic::server::NamedService for MethodClass<S>
where
    S: tonic::server::NamedService,
{
    const NAME: &'static str = S::NAME;
}

impl<S, ReqBody, ResBody> Service<Request<ReqBody>> for MethodClass<S>
where
    S: Service<Request<ReqBody>, Response = Response<ResBody>> + Clone + Send + 'static,
    S::Future: Send + 'static,
    ReqBody: Send + 'static,
    ResBody: Body<Data = Bytes> + Default + Send + 'static,
    ResBody::Error: Send,
{
    type Response = Response<PermitBody<ResBody>>;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<ReqBody>) -> Self::Future {
        let path = req.uri().path().to_string();
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);

        if is_streaming_path(&path) {
            // The server budget bounds this stream; the client imposes no
            // deadline of its own by default. Three arms — see the module
            // docs' "Streaming-path exemption" section:
            //   * a header ABOVE the budget is refused at the edge (unchanged);
            //   * a header WITHIN the budget is honoured as-is (no extra
            //     server-side deadline layered on top);
            //   * NO header at all is NOT refused — the budget itself becomes
            //     this stream's deadline (`deadline`, applied below via
            //     `PermitBody::Deadlined`).
            let mut deadline: Option<Duration> = None;
            if let Some(cap) = self.layer.wait_timeout {
                match parse_grpc_timeout(req.headers()) {
                    Some(requested) if requested > cap => {
                        return Box::pin(async move {
                            Ok(refuse::<ResBody>(
                                Code::DeadlineExceeded,
                                format!(
                                    "requested deadline exceeds server.limits.wait_timeout_secs \
                                     ({}s)",
                                    cap.as_secs()
                                ),
                                RefusedBound::Timeout,
                            )
                            .map(PermitBody::Passthrough))
                        });
                    }
                    Some(_) => {}
                    None => deadline = Some(cap),
                }
            }

            let is_wait_job = path == WAIT_JOB_PATH;
            let budget = if is_wait_job {
                self.layer.job_waits.clone()
            } else {
                self.layer.subscriptions.clone()
            };
            let bound = if is_wait_job {
                RefusedBound::JobWaits
            } else {
                RefusedBound::Subscriptions
            };
            let key = if is_wait_job {
                "max_job_waits"
            } else {
                "max_subscriptions"
            };

            return match budget {
                None => Box::pin(async move {
                    let response = inner.call(req).await?;
                    Ok(response.map(|body| PermitBody::new(body, None, deadline)))
                }),
                Some(sem) => match sem.try_acquire_owned() {
                    Ok(permit) => Box::pin(async move {
                        let response = inner.call(req).await?;
                        Ok(response.map(|body| PermitBody::new(body, Some(permit), deadline)))
                    }),
                    Err(_) => Box::pin(async move {
                        Ok(refuse::<ResBody>(
                            Code::ResourceExhausted,
                            format!("concurrent stream budget exhausted (server.limits.{key})"),
                            bound,
                        )
                        .map(PermitBody::Passthrough))
                    }),
                },
            };
        }

        // Unary path: `request_timeout_secs`, or a straight passthrough when
        // unset.
        match self.layer.request_timeout {
            None => Box::pin(async move {
                let response = inner.call(req).await?;
                Ok(response.map(PermitBody::Passthrough))
            }),
            Some(dur) => Box::pin(async move {
                match tokio::time::timeout(dur, inner.call(req)).await {
                    Ok(result) => {
                        let response = result?;
                        Ok(response.map(PermitBody::Passthrough))
                    }
                    Err(_) => Ok(refuse::<ResBody>(
                        Code::DeadlineExceeded,
                        format!(
                            "request exceeded server.limits.request_timeout_secs ({}s)",
                            dur.as_secs()
                        ),
                        RefusedBound::Timeout,
                    )
                    .map(PermitBody::Passthrough)),
                }
            }),
        }
    }
}

/// Response body for [`MethodClass`]. Three shapes:
///
/// * `Passthrough` — no permit, no deadline; the inner body unchanged.
/// * `Permitted` — holds an [`OwnedSemaphorePermit`] (a `max_subscriptions` /
///   `max_job_waits` stream-budget permit) released when the body reaches its
///   end frame OR is dropped (a client disconnect mid-stream, or the
///   connection dropping) — whichever happens first.
/// * `Deadlined` — the same permit release as `Permitted`, PLUS a
///   `server.limits.wait_timeout_secs` timer racing the inner body: if the
///   inner body has not reached its end frame once the timer fires, this body
///   synthesizes a `DEADLINE_EXCEEDED` trailer frame and ends the stream there
///   — the header-less-request arm of the module docs' "Streaming-path
///   exemption" section. Once fired, every subsequent poll returns `None`
///   (`fired`) — a body must not emit a frame after its trailers.
#[pin_project(project = PermitBodyProj)]
pub enum PermitBody<B> {
    Passthrough(#[pin] B),
    Permitted {
        #[pin]
        inner: B,
        permit: Option<OwnedSemaphorePermit>,
    },
    Deadlined {
        #[pin]
        inner: B,
        permit: Option<OwnedSemaphorePermit>,
        #[pin]
        sleep: Sleep,
        message: String,
        fired: bool,
    },
}

impl<B> PermitBody<B> {
    /// Build the right variant for a streaming response: no permit and no
    /// deadline is a bare passthrough; a `deadline` (the header-less-request
    /// arm) always produces `Deadlined`, with or without a permit; otherwise
    /// (`permit` alone) `Permitted`, matching the pre-existing shape.
    fn new(inner: B, permit: Option<OwnedSemaphorePermit>, deadline: Option<Duration>) -> Self {
        match deadline {
            Some(cap) => PermitBody::Deadlined {
                inner,
                permit,
                sleep: tokio::time::sleep(cap),
                message: deadline_message(cap),
                fired: false,
            },
            None => match permit {
                Some(permit) => PermitBody::Permitted {
                    inner,
                    permit: Some(permit),
                },
                None => PermitBody::Passthrough(inner),
            },
        }
    }
}

/// Needed so [`GlobalConcurrencyLimit`] / [`PerConnectionLimit`] can wrap
/// [`MethodClass`] (whose `Response` body is `PermitBody<B>`) under the same
/// `ResBody: Default` bound every early-refusal path in this module already
/// relies on.
impl<B: Default> Default for PermitBody<B> {
    fn default() -> Self {
        PermitBody::Passthrough(B::default())
    }
}

impl<B> Body for PermitBody<B>
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
            PermitBodyProj::Passthrough(body) => body.poll_frame(cx),
            PermitBodyProj::Permitted { inner, permit } => {
                let poll = inner.poll_frame(cx);
                if let Poll::Ready(None) = &poll {
                    // The stream reached its natural end: release the
                    // permit now rather than waiting for this body to drop.
                    permit.take();
                }
                poll
            }
            PermitBodyProj::Deadlined {
                inner,
                permit,
                sleep,
                message,
                fired,
            } => {
                if *fired {
                    return Poll::Ready(None);
                }
                // The inner body wins ties: a frame (including its natural
                // end) ready in the SAME poll the timer also fires is not a
                // timeout — the stream finished, full stop.
                match inner.poll_frame(cx) {
                    Poll::Ready(Some(frame)) => return Poll::Ready(Some(frame)),
                    Poll::Ready(None) => {
                        permit.take();
                        *fired = true;
                        return Poll::Ready(None);
                    }
                    Poll::Pending => {}
                }
                if sleep.poll(cx).is_ready() {
                    permit.take();
                    *fired = true;
                    let status = limits_status(Code::DeadlineExceeded, message.clone());
                    let mut headers = HeaderMap::new();
                    // The message is a fixed, ASCII format string (see
                    // `deadline_message`) — encoding it as a gRPC trailer
                    // cannot fail; an empty header map on the (unreachable)
                    // error path still ends the stream, just without the
                    // typed detail.
                    let _ = status.add_header(&mut headers);
                    return Poll::Ready(Some(Ok(Frame::trailers(headers))));
                }
                Poll::Pending
            }
        }
    }

    fn is_end_stream(&self) -> bool {
        match self {
            PermitBody::Passthrough(body) => body.is_end_stream(),
            PermitBody::Permitted { inner, .. } => inner.is_end_stream(),
            PermitBody::Deadlined { inner, fired, .. } => *fired || inner.is_end_stream(),
        }
    }

    fn size_hint(&self) -> SizeHint {
        match self {
            PermitBody::Passthrough(body) => body.size_hint(),
            PermitBody::Permitted { inner, .. } => inner.size_hint(),
            PermitBody::Deadlined { inner, .. } => inner.size_hint(),
        }
    }
}

#[cfg(test)]
mod tests {
    //! Deterministic proofs of every refusal path this module produces,
    //! exercised directly against the `tower::Service`/`Layer` contract each
    //! type implements — the SAME code path the live server runs.
    //! Concurrency bounds ([`GlobalConcurrencyLimit`], [`PerConnectionLimit`],
    //! and the streaming budgets on [`MethodClass`]) acquire their permit
    //! SYNCHRONOUSLY inside `Service::call`, before the returned future is
    //! ever polled — so calling a wrapped service N times without awaiting in
    //! between is a race-free way to prove "the (limit+1)-th caller is
    //! refused while the others are still in flight", with no sleep and no
    //! artificial delay hook needed anywhere in this suite.
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tokio::sync::Semaphore;
    use tonic::body::Body as TestBody;
    use tonic::codegen::http::Request as HttpRequest;
    use tonic::Code;

    use super::*;

    fn mk_req(path: &str) -> Request<()> {
        HttpRequest::builder().uri(path).body(()).unwrap()
    }

    fn mk_req_with_grpc_timeout(path: &str, value: &str) -> Request<()> {
        let mut req = mk_req(path);
        req.headers_mut()
            .insert("grpc-timeout", value.parse().unwrap());
        req
    }

    fn mk_req_from(path: &str, addr: std::net::SocketAddr) -> Request<()> {
        let mut req = mk_req(path);
        req.extensions_mut().insert(TcpConnectInfo {
            local_addr: None,
            remote_addr: Some(addr),
        });
        req
    }

    fn assert_refused<B>(resp: &Response<B>, bound: RefusedBound) {
        assert_eq!(
            resp.extensions().get::<RefusedBound>(),
            Some(&bound),
            "expected a {bound:?} refusal extension"
        );
        let status = tonic::Status::from_header_map(resp.headers())
            .expect("a refusal response must carry grpc-status headers");
        let expected_code = match bound {
            RefusedBound::Timeout => Code::DeadlineExceeded,
            _ => Code::ResourceExhausted,
        };
        assert_eq!(status.code(), expected_code);
    }

    fn assert_not_refused<B>(resp: &Response<B>) {
        assert!(
            resp.extensions().get::<RefusedBound>().is_none(),
            "response must not carry a RefusedBound extension"
        );
        assert!(
            resp.headers().get(tonic::Status::GRPC_STATUS).is_none(),
            "a passthrough success response must carry no grpc-status header"
        );
    }

    /// An inner service whose response future does not resolve until
    /// [`Semaphore::add_permits`] opens the gate — lets a test hold N "in-flight"
    /// requests open deterministically, without a sleep-based race.
    #[derive(Clone)]
    struct Held {
        gate: Arc<Semaphore>,
        entered: Arc<AtomicUsize>,
    }

    impl Service<Request<()>> for Held {
        type Response = Response<TestBody>;
        type Error = std::convert::Infallible;
        type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, _req: Request<()>) -> Self::Future {
            self.entered.fetch_add(1, Ordering::SeqCst);
            let gate = Arc::clone(&self.gate);
            Box::pin(async move {
                let _ = gate.acquire().await;
                Ok(Response::new(TestBody::default()))
            })
        }
    }

    /// An inner service that resolves immediately with an empty (already
    /// ended) success body — the "normal fast RPC" counterpart to [`Held`].
    #[derive(Clone)]
    struct Immediate;

    impl Service<Request<()>> for Immediate {
        type Response = Response<TestBody>;
        type Error = std::convert::Infallible;
        type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, _req: Request<()>) -> Self::Future {
            Box::pin(async move { Ok(Response::new(TestBody::default())) })
        }
    }

    /// A body that never emits a frame and never ends on its own -- proves
    /// [`PermitBody::Deadlined`] closes an otherwise-eternal stream once its
    /// OWN timer fires, with no reliance on the inner body's behaviour at all.
    #[derive(Default)]
    struct NeverEndingBody;

    impl Body for NeverEndingBody {
        type Data = Bytes;
        type Error = std::convert::Infallible;

        fn poll_frame(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
            Poll::Pending
        }
    }

    /// The `WaitJob`/`Subscribe` counterpart to [`Immediate`]: a stream that
    /// stays open forever on its own (see [`NeverEndingBody`]) -- the
    /// "genuinely live, no natural end" shape a real `WaitJob` on an
    /// unfinished job has.
    #[derive(Clone)]
    struct NeverEnding;

    impl Service<Request<()>> for NeverEnding {
        type Response = Response<NeverEndingBody>;
        type Error = std::convert::Infallible;
        type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, _req: Request<()>) -> Self::Future {
            Box::pin(async move { Ok(Response::new(NeverEndingBody)) })
        }
    }

    /// Drain every frame a [`Body`] yields, via `std::future::poll_fn` --
    /// polling `poll_frame` directly, exactly like a real HTTP/2 stack does.
    async fn drain_frames<B>(body: B) -> Vec<Frame<Bytes>>
    where
        B: Body<Data = Bytes>,
        B::Error: std::fmt::Debug,
    {
        let mut body = Box::pin(body);
        let mut frames = Vec::new();
        loop {
            match std::future::poll_fn(|cx| body.as_mut().poll_frame(cx)).await {
                Some(Ok(frame)) => frames.push(frame),
                Some(Err(e)) => panic!("body error: {e:?}"),
                None => break,
            }
        }
        frames
    }

    // ─── GlobalConcurrencyLimit ──────────────────────────────────────────

    #[tokio::test]
    async fn global_concurrency_limit_refuses_the_nth_plus_one_request_while_others_are_held() {
        let gate = Arc::new(Semaphore::new(0));
        let entered = Arc::new(AtomicUsize::new(0));
        let mut svc = GlobalConcurrencyLimitLayer::new(2).layer(Held {
            gate: Arc::clone(&gate),
            entered: Arc::clone(&entered),
        });

        let f1 = svc.call(mk_req("/jammi.v1.embedding.EmbeddingService/Search"));
        let f2 = svc.call(mk_req("/jammi.v1.embedding.EmbeddingService/Search"));
        // The 3rd caller must be refused SYNCHRONOUSLY -- `try_acquire_owned`
        // runs directly inside `call`, before ANY of these three futures is
        // ever polled (so `entered` -- which only increments once `Held`'s
        // future actually runs -- cannot yet reflect f1/f2 at this point;
        // checked below, after they resolve).
        let f3 = svc.call(mk_req("/jammi.v1.embedding.EmbeddingService/Search"));
        let r3 = f3.await.unwrap();
        assert_refused(&r3, RefusedBound::InFlight);

        gate.add_permits(1_000_000);
        let (r1, r2) = tokio::join!(f1, f2);
        assert_not_refused(&r1.unwrap());
        assert_not_refused(&r2.unwrap());
        // The refusal never touched the inner service -- exactly the 2
        // permitted calls reached it.
        assert_eq!(entered.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn global_concurrency_limit_zero_means_unbounded() {
        let gate = Arc::new(Semaphore::new(0));
        let entered = Arc::new(AtomicUsize::new(0));
        let mut svc = GlobalConcurrencyLimitLayer::new(0).layer(Held {
            gate: Arc::clone(&gate),
            entered: Arc::clone(&entered),
        });
        let futures: Vec<_> = (0..50)
            .map(|_| svc.call(mk_req("/jammi.v1.embedding.EmbeddingService/Search")))
            .collect();
        gate.add_permits(1_000_000);
        for f in futures {
            assert_not_refused(&f.await.unwrap());
        }
        assert_eq!(entered.load(Ordering::SeqCst), 50);
    }

    #[tokio::test]
    async fn global_concurrency_limit_bypasses_a_streaming_path() {
        let gate = Arc::new(Semaphore::new(0));
        let entered = Arc::new(AtomicUsize::new(0));
        let mut svc = GlobalConcurrencyLimitLayer::new(1).layer(Held {
            gate: Arc::clone(&gate),
            entered: Arc::clone(&entered),
        });
        // `max_in_flight` is a UNARY-only bound: every one of these opens
        // even though the limit is 1.
        let futures: Vec<_> = (0..5).map(|_| svc.call(mk_req(WAIT_JOB_PATH))).collect();
        gate.add_permits(1_000_000);
        for f in futures {
            assert_not_refused(&f.await.unwrap());
        }
        assert_eq!(entered.load(Ordering::SeqCst), 5);
    }

    // ─── PerConnectionLimit ──────────────────────────────────────────────

    #[tokio::test]
    async fn per_connection_limit_gates_independently_per_peer() {
        let gate = Arc::new(Semaphore::new(0));
        let entered = Arc::new(AtomicUsize::new(0));
        let mut svc = PerConnectionLimitLayer::new(1).layer(Held {
            gate: Arc::clone(&gate),
            entered: Arc::clone(&entered),
        });
        let peer_a: std::net::SocketAddr = "127.0.0.1:1".parse().unwrap();
        let peer_b: std::net::SocketAddr = "127.0.0.1:2".parse().unwrap();

        let a1 = svc.call(mk_req_from("/x/Y", peer_a));
        // Peer A's 2nd request, while A's 1st is still held: refused.
        let a2 = svc.call(mk_req_from("/x/Y", peer_a));
        let a2 = a2.await.unwrap();
        assert_refused(&a2, RefusedBound::InFlightPerConnection);

        // Peer B has its OWN budget -- untouched by peer A's exhaustion.
        let b1 = svc.call(mk_req_from("/x/Y", peer_b));
        gate.add_permits(1_000_000);
        let (a1, b1) = tokio::join!(a1, b1);
        assert_not_refused(&a1.unwrap());
        assert_not_refused(&b1.unwrap());
    }

    #[tokio::test]
    async fn per_connection_limit_with_no_connect_info_is_a_noop() {
        let gate = Arc::new(Semaphore::new(0));
        let entered = Arc::new(AtomicUsize::new(0));
        let mut svc = PerConnectionLimitLayer::new(1).layer(Held {
            gate: Arc::clone(&gate),
            entered: Arc::clone(&entered),
        });
        // No `TcpConnectInfo` extension on either request -- there is no
        // per-peer identity to key on, so this bound never refuses either.
        let f1 = svc.call(mk_req("/x/Y"));
        let f2 = svc.call(mk_req("/x/Y"));
        gate.add_permits(1_000_000);
        let (r1, r2) = tokio::join!(f1, f2);
        assert_not_refused(&r1.unwrap());
        assert_not_refused(&r2.unwrap());
    }

    // ─── MethodClass: unary timeout ─────────────────────────────────────

    #[tokio::test]
    async fn method_class_unary_timeout_refuses_with_deadline_exceeded() {
        let limits = LimitsConfig {
            request_timeout_secs: Some(1),
            ..LimitsConfig::default()
        };
        let mut svc = MethodClassLayer::new(&limits).layer(Held {
            gate: Arc::new(Semaphore::new(0)), // never fires -- the timeout wins
            entered: Arc::new(AtomicUsize::new(0)),
        });
        let resp = svc
            .call(mk_req("/jammi.v1.embedding.EmbeddingService/Search"))
            .await
            .unwrap();
        assert_refused(&resp, RefusedBound::Timeout);
    }

    #[tokio::test]
    async fn method_class_unary_no_timeout_configured_never_refuses() {
        let limits = LimitsConfig::default(); // request_timeout_secs: None
        let mut svc = MethodClassLayer::new(&limits).layer(Immediate);
        let resp = svc
            .call(mk_req("/jammi.v1.embedding.EmbeddingService/Search"))
            .await
            .unwrap();
        assert_not_refused(&resp);
    }

    // ─── MethodClass: wait_timeout_secs edge refusal ────────────────────

    #[tokio::test]
    async fn method_class_wait_timeout_refuses_an_over_budget_grpc_timeout_header() {
        let limits = LimitsConfig {
            wait_timeout_secs: Some(5),
            ..LimitsConfig::default()
        };
        let entered = Arc::new(AtomicUsize::new(0));
        let mut svc = MethodClassLayer::new(&limits).layer(Held {
            gate: Arc::new(Semaphore::new(0)),
            entered: Arc::clone(&entered),
        });
        let resp = svc
            .call(mk_req_with_grpc_timeout(WAIT_JOB_PATH, "10S"))
            .await
            .unwrap();
        assert_refused(&resp, RefusedBound::Timeout);
        // The inner service was never reached -- refused at the edge.
        assert_eq!(entered.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn method_class_wait_timeout_allows_a_within_budget_header() {
        let limits = LimitsConfig {
            wait_timeout_secs: Some(5),
            ..LimitsConfig::default()
        };
        let mut svc = MethodClassLayer::new(&limits).layer(Immediate);
        let resp = svc
            .call(mk_req_with_grpc_timeout(WAIT_JOB_PATH, "1S"))
            .await
            .unwrap();
        assert_not_refused(&resp);
    }

    /// RED before the reshape (#485 round 4): `mk_req` sends no `grpc-timeout`
    /// header at all (a header-less request, the shape a Python-shaped client
    /// with no explicit timeout sends); this used to be refused at the edge
    /// exactly like an over-budget header. The reshape: the SERVER budget
    /// bounds the stream instead -- this arm must NOT be refused at open, and
    /// the returned body must synthesize its own `DEADLINE_EXCEEDED` trailer
    /// once the budget elapses, never before it and never left open past it.
    #[tokio::test]
    async fn method_class_wait_timeout_with_no_header_is_not_refused_but_bounds_the_stream_as_a_deadline(
    ) {
        // Bypass `LimitsConfig`'s whole-second granularity for a fast,
        // deterministic unit test -- this `tests` submodule has field access
        // to `MethodClassLayer`'s private fields (same defining module tree).
        let budget = Duration::from_millis(60);
        let mut svc = MethodClassLayer {
            request_timeout: None,
            wait_timeout: Some(budget),
            subscriptions: None,
            job_waits: None,
        }
        .layer(NeverEnding);

        let started = std::time::Instant::now();
        let resp = svc.call(mk_req(WAIT_JOB_PATH)).await.unwrap();
        // NOT refused at open: the same success shape every passthrough
        // response has.
        assert_not_refused(&resp);
        assert!(
            started.elapsed() < budget,
            "opening the stream must not itself wait out the budget"
        );

        // `NeverEndingBody` never emits a frame on its own -- the ONLY frame
        // this body can ever produce is the deadline's synthesized trailer.
        let frames = drain_frames(resp.into_body()).await;
        assert_eq!(
            frames.len(),
            1,
            "exactly one synthesized DEADLINE_EXCEEDED trailer frame, then the stream ends"
        );
        let trailers = frames[0]
            .trailers_ref()
            .expect("the synthesized frame must be a trailers frame");
        let status = tonic::Status::from_header_map(trailers)
            .expect("the trailer frame must carry a decodable grpc-status");
        assert_eq!(status.code(), Code::DeadlineExceeded);
        assert!(
            started.elapsed() >= budget,
            "the deadline must not fire before the budget elapses"
        );
    }

    #[tokio::test]
    async fn method_class_wait_timeout_unset_never_refuses_regardless_of_header() {
        let limits = LimitsConfig::default(); // wait_timeout_secs: None
        let mut svc = MethodClassLayer::new(&limits).layer(Immediate);
        let resp = svc
            .call(mk_req_with_grpc_timeout(WAIT_JOB_PATH, "99999999H"))
            .await
            .unwrap();
        assert_not_refused(&resp);
    }

    // ─── MethodClass: stream budgets ─────────────────────────────────────

    #[tokio::test]
    async fn method_class_job_waits_budget_refuses_the_second_concurrent_wait_job_and_releases_on_drop(
    ) {
        let limits = LimitsConfig {
            max_job_waits: 1,
            ..LimitsConfig::default()
        };
        let mut svc = MethodClassLayer::new(&limits).layer(Immediate);

        let r1 = svc.call(mk_req(WAIT_JOB_PATH)).await.unwrap();
        assert_not_refused(&r1);

        // r1's body (and its held permit) is still alive: the 2nd concurrent
        // WaitJob is refused.
        let r2 = svc.call(mk_req(WAIT_JOB_PATH)).await.unwrap();
        assert_refused(&r2, RefusedBound::JobWaits);

        // Dropping r1's body releases its permit -- the module doc's
        // "released on end frame OR drop" contract, drop arm.
        drop(r1);
        let r3 = svc.call(mk_req(WAIT_JOB_PATH)).await.unwrap();
        assert_not_refused(&r3);
    }

    #[tokio::test]
    async fn method_class_subscriptions_budget_is_independent_of_job_waits() {
        let limits = LimitsConfig {
            max_job_waits: 1,
            max_subscriptions: 1,
            ..LimitsConfig::default()
        };
        let mut svc = MethodClassLayer::new(&limits).layer(Immediate);

        let wait1 = svc.call(mk_req(WAIT_JOB_PATH)).await.unwrap();
        assert_not_refused(&wait1);
        // A Subscribe stream draws from its OWN budget -- unaffected by
        // WaitJob's budget being exhausted.
        let sub1 = svc.call(mk_req(SUBSCRIBE_PATH)).await.unwrap();
        assert_not_refused(&sub1);

        let wait2 = svc.call(mk_req(WAIT_JOB_PATH)).await.unwrap();
        assert_refused(&wait2, RefusedBound::JobWaits);
        let sub2 = svc.call(mk_req(SUBSCRIBE_PATH)).await.unwrap();
        assert_refused(&sub2, RefusedBound::Subscriptions);
    }

    #[tokio::test]
    async fn method_class_zero_budget_means_unbounded() {
        let limits = LimitsConfig {
            max_job_waits: 0,
            ..LimitsConfig::default()
        };
        let mut svc = MethodClassLayer::new(&limits).layer(Immediate);
        for _ in 0..25 {
            let r = svc.call(mk_req(WAIT_JOB_PATH)).await.unwrap();
            assert_not_refused(&r);
            // Deliberately not dropped early -- proves the budget never
            // engages at all when configured to 0, not merely a large one.
            std::mem::forget(r);
        }
    }

    // ─── RefusalStatusLayer ──────────────────────────────────────────────

    #[tokio::test]
    async fn refusal_status_layer_counts_a_refused_bound_extension_under_its_label() {
        let metrics = Arc::new(MetricsRegistry::new().unwrap());
        struct AlwaysRefuses;
        impl Service<Request<()>> for AlwaysRefuses {
            type Response = Response<TestBody>;
            type Error = std::convert::Infallible;
            type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;
            fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
                Poll::Ready(Ok(()))
            }
            fn call(&mut self, _req: Request<()>) -> Self::Future {
                Box::pin(async move {
                    Ok(refuse::<TestBody>(
                        Code::ResourceExhausted,
                        "test refusal".to_string(),
                        RefusedBound::Subscriptions,
                    ))
                })
            }
        }
        let mut svc = RefusalStatusLayer::new(Arc::clone(&metrics)).layer(AlwaysRefuses);
        let _ = svc.call(mk_req("/x/Y")).await.unwrap();
        assert_eq!(
            metrics
                .grpc_refused
                .with_label_values(&["subscriptions"])
                .get(),
            1
        );
        assert_eq!(
            metrics.grpc_refused.with_label_values(&["job_waits"]).get(),
            0
        );
    }

    #[tokio::test]
    async fn refusal_status_layer_counts_an_unlabeled_out_of_range_as_message_size() {
        let metrics = Arc::new(MetricsRegistry::new().unwrap());
        struct RawCodecRefusal;
        impl Service<Request<()>> for RawCodecRefusal {
            type Response = Response<TestBody>;
            type Error = std::convert::Infallible;
            type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;
            fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
                Poll::Ready(Ok(()))
            }
            fn call(&mut self, _req: Request<()>) -> Self::Future {
                // Mirrors what tonic's own codec produces on this exact
                // vendored version (tonic-0.14.5/src/codec/decode.rs:185-195):
                // an OUT_OF_RANGE status response with NO RefusedBound
                // extension.
                Box::pin(async move {
                    Ok(
                        tonic::Status::out_of_range("decoded message length too large")
                            .into_http::<TestBody>(),
                    )
                })
            }
        }
        let mut svc = RefusalStatusLayer::new(Arc::clone(&metrics)).layer(RawCodecRefusal);
        let _ = svc.call(mk_req("/x/Y")).await.unwrap();
        assert_eq!(
            metrics
                .grpc_refused
                .with_label_values(&[MESSAGE_SIZE_LABEL])
                .get(),
            1
        );
    }

    #[tokio::test]
    async fn refusal_status_layer_does_not_count_a_normal_success_response() {
        let metrics = Arc::new(MetricsRegistry::new().unwrap());
        let mut svc = RefusalStatusLayer::new(Arc::clone(&metrics)).layer(Immediate);
        let _ = svc.call(mk_req("/x/Y")).await.unwrap();
        assert_eq!(
            metrics.grpc_refused.with_label_values(&["in_flight"]).get(),
            0
        );
        assert_eq!(
            metrics
                .grpc_refused
                .with_label_values(&[MESSAGE_SIZE_LABEL])
                .get(),
            0
        );
    }

    // ─── is_streaming_path / parse_grpc_timeout ─────────────────────────

    #[test]
    fn is_streaming_path_matches_exactly_the_two_known_streaming_rpcs() {
        assert!(is_streaming_path(WAIT_JOB_PATH));
        assert!(is_streaming_path(SUBSCRIBE_PATH));
        assert!(!is_streaming_path(
            "/jammi.v1.embedding.EmbeddingService/Search"
        ));
        assert!(!is_streaming_path("/jammi.v1.job.JobService/JobStatus"));
    }

    /// [`is_streaming_path`]'s hardcoded allowlist is a documented,
    /// deliberate scope reduction (module docs' "Streaming-path exemption"
    /// section) -- but nothing catches a THIRD server-streaming rpc landing
    /// without the list being extended by hand. This DERIVES the actual
    /// server-streaming method set from the compiled `jammi.v1`
    /// `FILE_DESCRIPTOR_SET` (`MethodDescriptorProto::server_streaming`) and
    /// asserts it equals the hardcoded set -- symmetric, so a stale entry (a
    /// removed rpc still allowlisted) fails just as loudly as a missing one.
    #[test]
    fn is_streaming_path_allowlist_matches_the_descriptor_derived_server_streaming_set() {
        use prost::Message;
        use prost_types::FileDescriptorSet;
        use std::collections::BTreeSet;

        const WIRE_PACKAGE_PREFIX: &str = "jammi.v1";
        let set = FileDescriptorSet::decode(jammi_wire::FILE_DESCRIPTOR_SET)
            .expect("the compiled jammi.v1 descriptor must decode");
        let mut derived = BTreeSet::new();
        for file in &set.file {
            let package = file.package();
            // The descriptor also carries `google.*` / `arrow.*` imports
            // (e.g. `arrow.flight.protocol.FlightService`, itself
            // server-streaming) -- only `jammi.v1.*` is this bind's scope,
            // matching `tenant_isolation_oracle.rs`'s `WIRE_PACKAGE_PREFIX`
            // convention.
            if package != WIRE_PACKAGE_PREFIX
                && !package.starts_with(&format!("{WIRE_PACKAGE_PREFIX}."))
            {
                continue;
            }
            for service in &file.service {
                for method in &service.method {
                    if method.server_streaming() {
                        derived.insert(format!("/{package}.{}/{}", service.name(), method.name()));
                    }
                }
            }
        }

        let hardcoded: BTreeSet<String> = [WAIT_JOB_PATH.to_string(), SUBSCRIBE_PATH.to_string()]
            .into_iter()
            .collect();

        assert_eq!(
            derived, hardcoded,
            "is_streaming_path's hardcoded allowlist has drifted from the descriptor-derived \
             server-streaming rpc set -- a new server-streaming rpc needs WAIT_JOB_PATH/\
             SUBSCRIBE_PATH's allowlist extended by hand (module docs' 'Streaming-path \
             exemption' section), or a removed one needs its stale entry dropped"
        );
    }

    #[test]
    fn parse_grpc_timeout_accepts_every_documented_unit() {
        let cases: &[(&str, Duration)] = &[
            ("10S", Duration::from_secs(10)),
            ("2M", Duration::from_secs(120)),
            ("1H", Duration::from_secs(3600)),
            ("500m", Duration::from_millis(500)),
            ("7u", Duration::from_micros(7)),
            ("9n", Duration::from_nanos(9)),
        ];
        for (raw, expected) in cases {
            let mut headers = HeaderMap::new();
            headers.insert("grpc-timeout", (*raw).parse().unwrap());
            assert_eq!(parse_grpc_timeout(&headers), Some(*expected), "{raw}");
        }
    }

    #[test]
    fn parse_grpc_timeout_rejects_malformed_or_absent_values() {
        assert_eq!(parse_grpc_timeout(&HeaderMap::new()), None);
        for bad in ["", "S", "10", "10X", "1234567890S"] {
            let mut headers = HeaderMap::new();
            if !bad.is_empty() {
                headers.insert("grpc-timeout", bad.parse().unwrap());
            }
            assert_eq!(
                parse_grpc_timeout(&headers),
                None,
                "expected None for {bad:?}"
            );
        }
    }
}
