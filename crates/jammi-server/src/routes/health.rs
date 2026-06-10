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
use prometheus::{Encoder, Histogram, HistogramOpts, IntCounter, Registry, TextEncoder};
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
            "Total number of gRPC requests served (CatalogService + TriggerService).",
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

        Ok(Self {
            inner,
            grpc_requests,
            flight_queries,
            eval_invocations,
            search_latency,
        })
    }

    /// Borrow the underlying `prometheus::Registry`. Tests that want to
    /// scrape metrics directly use this to call `.gather()`.
    pub fn inner(&self) -> &Registry {
        &self.inner
    }
}
