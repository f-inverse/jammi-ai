//! HTTP side-channel endpoints — `/healthz`, `/readyz`, and `/metrics`.
//!
//! Liveness (`/healthz`) is a single, dependency-free probe: if the
//! process is up and the Axum router is serving, the answer is `200`.
//! Readiness (`/readyz`) goes one step further and pings the catalog
//! backend the engine session was built around — that's the substrate
//! resource Jammi can't serve without. Metrics (`/metrics`) emit a
//! Prometheus text-format snapshot of the registry the gRPC services
//! and Flight SQL layer feed counters into.
//!
//! The three handlers share no global state: each takes its dependency
//! through `axum::extract::State`, so test fixtures can wire stubbed
//! readiness probes and registries without touching a singleton.

use std::sync::Arc;

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use std::sync::{OnceLock, Weak};

use jammi_ai::fine_tune::worker::{LoopState, WorkerShared};
use jammi_db::catalog::lease_keeper::LeaseKeeper;
use prometheus::{
    Encoder, Gauge, Histogram, HistogramOpts, IntCounter, IntCounterVec, IntGauge, IntGaugeVec,
    Opts, Registry, TextEncoder,
};
use serde_json::{json, Value};

use crate::runtime::ReadinessProbe;

/// `GET /healthz` — liveness probe.
///
/// Returns `{"status": "ok", "version": "<workspace version>"}` without
/// touching any downstream dependency. A `200` here means the process
/// is alive; orchestration platforms use this to decide whether to
/// restart the container, not whether to route traffic to it.
pub async fn healthz() -> Json<Value> {
    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

/// `GET /readyz` — readiness probe.
///
/// Pings the catalog backend the engine session is bound to; on success
/// returns `200` with `{"status":"ready"}`, on failure returns `503`
/// with `{"status":"not_ready","detail":"<message>"}`. Use this for
/// load-balancer admission — a transient catalog outage should remove
/// the instance from rotation, not restart it.
pub async fn readyz(State(probe): State<Arc<ReadinessProbe>>) -> Response {
    match probe.check().await {
        Ok(()) => (StatusCode::OK, Json(json!({"status": "ready"}))).into_response(),
        Err(detail) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "not_ready",
                "detail": detail,
            })),
        )
            .into_response(),
    }
}

/// `GET /metrics` — Prometheus text-format snapshot.
///
/// Encodes the shared [`MetricsRegistry`] into the standard Prometheus
/// exposition format. The registry's lifetime is owned by the running
/// `OssServer`; the route only reads from it.
pub async fn metrics(State(registry): State<Arc<MetricsRegistry>>) -> Response {
    registry.refresh_gauges();
    let metric_families = registry.inner.gather();
    let mut buffer = Vec::new();
    let encoder = TextEncoder::new();
    if let Err(e) = encoder.encode(&metric_families, &mut buffer) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("metrics encode failure: {e}"),
        )
            .into_response();
    }
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, encoder.format_type())],
        buffer,
    )
        .into_response()
}

/// Prometheus registry plus the substrate-level counters and histogram
/// the gRPC services and Flight SQL layer increment. Held as `Arc` so
/// every Axum route handler and Tonic service can share one instance.
///
/// The counters are intentionally lite — gRPC requests, Flight queries,
/// eval invocations, and a search-latency histogram — matching the
/// SPEC-S5 §"Observability" line item.
pub struct MetricsRegistry {
    inner: Registry,
    pub grpc_requests: IntCounter,
    pub flight_queries: IntCounter,
    pub eval_invocations: IntCounter,
    pub search_latency: Histogram,
    /// Requests refused at the [`crate::limits`] edge (`[server.limits]`),
    /// labelled by `reason` — one of `message_size`, `in_flight`,
    /// `in_flight_per_connection`, `subscriptions`, `job_waits`, `timeout`.
    /// See [`Self::record_refusal`].
    pub grpc_refused: IntCounterVec,
    /// The worker gauge families — registered lazily by
    /// [`Self::attach_worker`] on a worker-enabled process, so a process
    /// with no claim loop omits the family entirely (absent, never 0).
    worker: OnceLock<WorkerGauges>,
    /// `jammi_lease_heartbeat_age_seconds` — registered by
    /// [`Self::attach_keeper`] on every process that has a session.
    keeper: OnceLock<KeeperGauge>,
}

/// The worker families and the `Weak` they are copied from on a scrape.
struct WorkerGauges {
    shared: Weak<WorkerShared>,
    jobs_queued: IntGaugeVec,
    jobs_running: IntGaugeVec,
    jobs_in_flight: IntGauge,
    claim_loop_up: IntGauge,
}

struct KeeperGauge {
    keeper: Arc<LeaseKeeper>,
    heartbeat_age: Gauge,
}

impl MetricsRegistry {
    /// Build a fresh registry with the four substrate-level metrics
    /// registered. Returns an error if any metric registration fails —
    /// in practice this only happens when names collide, so the caller
    /// should treat it as a startup-time fault, not a runtime concern.
    pub fn new() -> Result<Self, prometheus::Error> {
        let inner = Registry::new();

        let grpc_requests = IntCounter::new(
            "jammi_grpc_requests_total",
            "Total number of gRPC requests served across all jammi.v1 services.",
        )?;
        inner.register(Box::new(grpc_requests.clone()))?;

        let flight_queries = IntCounter::new(
            "jammi_flight_queries_total",
            "Total number of Flight SQL queries executed.",
        )?;
        inner.register(Box::new(flight_queries.clone()))?;

        let eval_invocations = IntCounter::new(
            "jammi_eval_invocations_total",
            "Total number of eval RPCs invoked.",
        )?;
        inner.register(Box::new(eval_invocations.clone()))?;

        let search_latency = Histogram::with_opts(
            HistogramOpts::new(
                "jammi_search_latency_seconds",
                "Vector-search request latency, in seconds.",
            )
            .buckets(vec![
                0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
            ]),
        )?;
        inner.register(Box::new(search_latency.clone()))?;

        let grpc_refused = IntCounterVec::new(
            Opts::new(
                "jammi_grpc_refused_total",
                "Total number of gRPC/Flight requests refused at the [server.limits] edge, \
                 labelled by refusal reason.",
            ),
            &["reason"],
        )?;
        inner.register(Box::new(grpc_refused.clone()))?;

        Ok(Self {
            inner,
            grpc_requests,
            flight_queries,
            eval_invocations,
            search_latency,
            grpc_refused,
            worker: OnceLock::new(),
            keeper: OnceLock::new(),
        })
    }

    /// Register the worker gauge families and bind them to a claim loop's
    /// shared state: `jammi_jobs_queued{kind}` / `jammi_jobs_running{kind}`
    /// (the sampler's last catalog snapshot — held `claimable = false` rows
    /// count as queued), `jammi_worker_jobs_in_flight` (loop-claimed jobs
    /// under a live hold; an inline `run_now` is never counted) and
    /// `jammi_worker_claim_loop_up` (1 while the loop reports `Running`; 0
    /// once it stopped, aborted, failed, or its state is gone). Called once,
    /// by the server after the worker guard is hoisted; a second call is a
    /// no-op. Never called on a worker-less process, so the family is
    /// absent there. A scrape copies the snapshot ([`Self::refresh_gauges`])
    /// and issues no catalog statement.
    pub fn attach_worker(&self, shared: Weak<WorkerShared>) -> Result<(), prometheus::Error> {
        if self.worker.get().is_some() {
            return Ok(());
        }
        let jobs_queued = IntGaugeVec::new(
            Opts::new(
                "jammi_jobs_queued",
                "Queued loop-claimable jobs (execution = 'queued') by kind, sampled from the \
                 catalog every [worker] metrics_sample_secs; a held (claimable = false) row \
                 counts as queued.",
            ),
            &["kind"],
        )?;
        let jobs_running = IntGaugeVec::new(
            Opts::new(
                "jammi_jobs_running",
                "Running loop-claimable jobs (execution = 'queued') by kind, sampled from the \
                 catalog every [worker] metrics_sample_secs.",
            ),
            &["kind"],
        )?;
        let jobs_in_flight = IntGauge::new(
            "jammi_worker_jobs_in_flight",
            "Loop-claimed jobs this process is running under a live lease hold (0 or 1); an \
             inline run_now in this process is never counted.",
        )?;
        let claim_loop_up = IntGauge::new(
            "jammi_worker_claim_loop_up",
            "1 while this process's claim loop task is running, 0 once it stopped, aborted or \
             failed.",
        )?;
        self.inner.register(Box::new(jobs_queued.clone()))?;
        self.inner.register(Box::new(jobs_running.clone()))?;
        self.inner.register(Box::new(jobs_in_flight.clone()))?;
        self.inner.register(Box::new(claim_loop_up.clone()))?;
        let _ = self.worker.set(WorkerGauges {
            shared,
            jobs_queued,
            jobs_running,
            jobs_in_flight,
            claim_loop_up,
        });
        Ok(())
    }

    /// Register `jammi_lease_heartbeat_age_seconds` — seconds since the
    /// process's lease keeper last completed a renewal pass — bound to the
    /// session's keeper. Every process has a keeper, so every server
    /// attaches one; a second call is a no-op.
    pub fn attach_keeper(&self, keeper: Arc<LeaseKeeper>) -> Result<(), prometheus::Error> {
        if self.keeper.get().is_some() {
            return Ok(());
        }
        let heartbeat_age = Gauge::new(
            "jammi_lease_heartbeat_age_seconds",
            "Seconds since this process's lease keeper last completed a renewal pass over \
             every lease it holds.",
        )?;
        self.inner.register(Box::new(heartbeat_age.clone()))?;
        let _ = self.keeper.set(KeeperGauge {
            keeper,
            heartbeat_age,
        });
        Ok(())
    }

    /// Copy the attached snapshots into the gauges — what a scrape does
    /// before `gather()`. Zero catalog statements: the worker sample is the
    /// sampler task's, the keeper age is an in-memory stamp.
    pub fn refresh_gauges(&self) {
        if let Some(w) = self.worker.get() {
            w.jobs_queued.reset();
            w.jobs_running.reset();
            match w.shared.upgrade() {
                Some(shared) => {
                    let sample = shared.sample();
                    for (kind, n) in sample.queued {
                        w.jobs_queued.with_label_values(&[&kind]).set(n);
                    }
                    for (kind, n) in sample.running {
                        w.jobs_running.with_label_values(&[&kind]).set(n);
                    }
                    w.jobs_in_flight.set(shared.in_flight() as i64);
                    w.claim_loop_up
                        .set((shared.loop_state() == LoopState::Running) as i64);
                }
                None => {
                    w.jobs_in_flight.set(0);
                    w.claim_loop_up.set(0);
                }
            }
        }
        if let Some(k) = self.keeper.get() {
            k.heartbeat_age
                .set(k.keeper.last_renewed_at().elapsed().as_secs_f64());
        }
    }

    /// Borrow the underlying `prometheus::Registry`. Tests that want to
    /// scrape metrics directly use this to call `.gather()`.
    pub fn inner(&self) -> &Registry {
        &self.inner
    }

    /// Increment `jammi_grpc_refused_total{reason}` by one. The single call
    /// site every [`crate::limits`] refusal path uses, so the label set
    /// (`RefusedBound::label`) and the counter's own labels cannot drift
    /// apart.
    pub fn record_refusal(&self, reason: &str) {
        self.grpc_refused.with_label_values(&[reason]).inc();
    }
}
