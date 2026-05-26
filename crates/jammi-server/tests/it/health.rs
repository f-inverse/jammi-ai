//! Integration tests for the HTTP side-channel — `/healthz`, `/readyz`,
//! and `/metrics`. The full router is constructed via
//! `jammi_server::build_health_router` so the tests assert the same
//! surface the binary exposes.

use std::sync::Arc;

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use jammi_server::routes::health::MetricsRegistry;
use jammi_server::runtime::{ReadinessCheck, ReadinessProbe};
use tower::ServiceExt;

/// Always-ready stub used by happy-path readiness tests.
struct AlwaysReady;

#[async_trait]
impl ReadinessCheck for AlwaysReady {
    async fn check(&self) -> Result<(), String> {
        Ok(())
    }
}

/// Always-failing stub used by 503-path readiness tests.
struct AlwaysDown;

#[async_trait]
impl ReadinessCheck for AlwaysDown {
    async fn check(&self) -> Result<(), String> {
        Err("simulated catalog outage".into())
    }
}

fn router(readiness: Arc<dyn ReadinessCheck>) -> axum::Router {
    let probe = Arc::new(ReadinessProbe::new(readiness));
    let metrics = Arc::new(MetricsRegistry::new().expect("metrics registry"));
    jammi_server::build_health_router(probe, metrics)
}

#[tokio::test]
async fn healthz_returns_ok_with_version() {
    let app = router(Arc::new(AlwaysReady));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/healthz")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("oneshot");
    assert_eq!(resp.status(), StatusCode::OK);
    let body: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body bytes"),
    )
    .expect("json");
    assert_eq!(body["status"], "ok");
    assert!(
        body["version"].is_string(),
        "healthz body must carry the crate version, got {body:?}"
    );
}

#[tokio::test]
async fn readyz_returns_200_when_probe_succeeds() {
    let app = router(Arc::new(AlwaysReady));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/readyz")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("oneshot");
    assert_eq!(resp.status(), StatusCode::OK);
    let body: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body bytes"),
    )
    .expect("json");
    assert_eq!(body["status"], "ready");
}

#[tokio::test]
async fn readyz_returns_503_when_probe_fails() {
    let app = router(Arc::new(AlwaysDown));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/readyz")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("oneshot");
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("body bytes"),
    )
    .expect("json");
    assert_eq!(body["status"], "not_ready");
    assert_eq!(body["detail"], "simulated catalog outage");
}

#[tokio::test]
async fn metrics_returns_prometheus_text_format() {
    let app = router(Arc::new(AlwaysReady));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("oneshot");
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .expect("content-type")
        .to_str()
        .expect("ascii");
    assert!(
        ct.starts_with("text/plain"),
        "metrics content-type must be Prometheus text-format, got {ct}"
    );
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let text = std::str::from_utf8(&body).expect("utf8");
    for expected in [
        "jammi_grpc_requests_total",
        "jammi_flight_queries_total",
        "jammi_eval_invocations_total",
        "jammi_search_latency_seconds",
    ] {
        assert!(
            text.contains(expected),
            "metrics output must expose `{expected}`, got:\n{text}"
        );
    }
}

#[tokio::test]
async fn metrics_reflects_counter_increments() {
    let probe = Arc::new(ReadinessProbe::new(Arc::new(AlwaysReady)));
    let metrics = Arc::new(MetricsRegistry::new().expect("metrics registry"));

    metrics.grpc_requests.inc();
    metrics.grpc_requests.inc();
    metrics.flight_queries.inc();

    let app = jammi_server::build_health_router(probe, Arc::clone(&metrics));
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("oneshot");
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let text = std::str::from_utf8(&body).expect("utf8");
    assert!(
        text.contains("jammi_grpc_requests_total 2"),
        "expected grpc counter to read 2, got:\n{text}"
    );
    assert!(
        text.contains("jammi_flight_queries_total 1"),
        "expected flight counter to read 1, got:\n{text}"
    );
}
