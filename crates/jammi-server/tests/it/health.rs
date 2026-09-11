//! Integration tests for the HTTP side-channel — `/healthz`, `/readyz`,
//! and `/metrics`. The full router is constructed via
//! `jammi_server::build_health_router` so the tests assert the same
//! surface the binary exposes.

use std::sync::Arc;

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use jammi_server::routes::health::MetricsRegistry;
use jammi_server::runtime::{LivenessProbe, ReadinessCheck, ReadinessProbe};
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
    jammi_server::build_health_router(probe, metrics, Arc::new(LivenessProbe::always_healthy()))
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

    let app = jammi_server::build_health_router(
        probe,
        Arc::clone(&metrics),
        Arc::new(LivenessProbe::always_healthy()),
    );
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

// ---------------------------------------------------------------------------
// OPS (#482) — the worker gauges: `jammi_jobs_queued{kind}` /
// `jammi_jobs_running{kind}` sampled from the catalog every `[worker]
// metrics_sample_secs` on a dedicated task, `jammi_worker_jobs_in_flight`,
// `jammi_worker_claim_loop_up`, and `jammi_lease_heartbeat_age_seconds` —
// scraped off a full `OssServer`.
// ---------------------------------------------------------------------------

mod gauges {
    use std::sync::Arc;
    use std::time::Duration;

    use jammi_ai::fine_tune::spec::{TrainingCommon, TrainingSpec};
    use jammi_ai::fine_tune::worker::WorkerShared;
    use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod};
    use jammi_ai::jobs::{compute_test_hooks, ComputeSpec};
    use jammi_ai::model::ModelTask;
    use jammi_ai::session::InferenceSession;
    use jammi_db::config::{JammiConfig, WorkerKinds};
    use jammi_db::source::{FileFormat, SourceConnection, SourceType};
    use jammi_db::store::CachePolicy;
    use jammi_server::runtime::{OssServer, ServerError, ShutdownOutcome};
    use jammi_test_utils::{cookbook_fixture, fixture_url, test_config};
    use tokio::sync::watch;

    fn tiny_bert_model() -> String {
        format!("local:{}", cookbook_fixture("tiny_bert").display())
    }

    struct Served {
        session: Arc<InferenceSession>,
        health_addr: std::net::SocketAddr,
        shared: Option<std::sync::Weak<WorkerShared>>,
        drain_tx: watch::Sender<bool>,
        release_tx: watch::Sender<bool>,
        task: tokio::task::JoinHandle<Result<ShutdownOutcome, ServerError>>,
    }

    async fn serve(cfg: JammiConfig) -> Served {
        let server = OssServer::new(cfg).await.expect("OssServer::new");
        let session = server.session();
        let bound = server.bind().await.expect("bind");
        let health_addr = bound.health_addr();
        let shared = bound.worker_shared();
        let (drain_tx, drain_rx) = watch::channel(false);
        let (release_tx, release_rx) = watch::channel(false);
        let task = tokio::spawn(bound.serve_with_signals(drain_rx, release_rx));
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        while reqwest::get(format!("http://{health_addr}/healthz"))
            .await
            .is_err()
        {
            assert!(
                tokio::time::Instant::now() < deadline,
                "side-channel never up"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        Served {
            session,
            health_addr,
            shared,
            drain_tx,
            release_tx,
            task,
        }
    }

    impl Served {
        async fn release_and_finish(self) {
            let _ = self.drain_tx.send(true);
            let _ = self.release_tx.send(true);
            tokio::time::timeout(Duration::from_secs(60), self.task)
                .await
                .expect("the serve task ends")
                .unwrap()
                .unwrap();
        }
    }

    fn config(dir: &std::path::Path, sample_secs: u64, kinds: WorkerKinds) -> JammiConfig {
        let mut cfg = test_config(dir);
        cfg.server.health_listen = "127.0.0.1:0".into();
        cfg.server.flight_listen = "127.0.0.1:0".into();
        cfg.worker.metrics_sample_secs = sample_secs;
        cfg.worker.kinds = kinds;
        cfg
    }

    async fn scrape(health_addr: &std::net::SocketAddr) -> String {
        reqwest::get(format!("http://{health_addr}/metrics"))
            .await
            .expect("metrics GET")
            .text()
            .await
            .expect("metrics body")
    }

    fn sample(metrics: &str, name: &str) -> Option<f64> {
        metrics
            .lines()
            .find(|line| line.split_whitespace().next() == Some(name))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|v| v.parse::<f64>().ok())
    }

    async fn add_training(session: &InferenceSession) {
        session
            .add_source(
                "training",
                SourceType::File,
                SourceConnection {
                    url: Some(fixture_url("training_pairs.csv")),
                    format: Some(FileFormat::Csv),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
    }

    fn fine_tune(epochs: usize) -> jammi_ai::jobs::JobSpec {
        TrainingSpec::FineTune {
            source: "training".into(),
            columns: vec!["text_a".into(), "text_b".into(), "score".into()],
            method: FineTuneMethod::Lora,
            task: ModelTask::TextEmbedding,
            common: TrainingCommon {
                base_model: tiny_bert_model(),
                config: FineTuneConfig {
                    epochs,
                    batch_size: 8,
                    lora_rank: 4,
                    warmup_steps: 0,
                    ..Default::default()
                },
            },
        }
        .into()
    }

    /// Acceptance 6: `jammi_jobs_queued{kind="fine_tune"}` changes within
    /// one `metrics_sample_secs` of a submission (the worker claims only
    /// `embedding`, so the row stays queued). Base: the family does not
    /// exist.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn metrics_show_the_queued_count_change_within_one_tick_after_submit() {
        let dir = tempfile::TempDir::new().unwrap();
        let served = serve(config(
            dir.path(),
            1,
            WorkerKinds::Only(vec!["embedding".into()]),
        ))
        .await;
        add_training(&served.session).await;
        let before = scrape(&served.health_addr).await;
        assert_eq!(
            sample(&before, r#"jammi_jobs_queued{kind="fine_tune"}"#),
            None,
            "no fine_tune row yet:\n{before}"
        );
        served.session.enqueue(fine_tune(1), 0).await.unwrap();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
        loop {
            let m = scrape(&served.health_addr).await;
            if sample(&m, r#"jammi_jobs_queued{kind="fine_tune"}"#) == Some(1.0) {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "the queued gauge never moved within one tick:\n{m}"
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        served.release_and_finish().await;
    }

    /// A process with no claim loop omits the worker family (absent, never
    /// 0) while still exposing the keeper's heartbeat age.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn metrics_omit_worker_gauges_when_no_worker_is_attached() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut cfg = config(dir.path(), 1, WorkerKinds::default());
        cfg.worker.enabled = false;
        let served = serve(cfg).await;
        assert!(served.shared.is_none());
        let m = scrape(&served.health_addr).await;
        for family in [
            "jammi_jobs_queued",
            "jammi_jobs_running",
            "jammi_worker_jobs_in_flight",
            "jammi_worker_claim_loop_up",
        ] {
            assert!(
                !m.contains(family),
                "{family} must be absent on a worker-less process:\n{m}"
            );
        }
        assert!(
            sample(&m, "jammi_lease_heartbeat_age_seconds").is_some(),
            "the keeper gauge is present on every process:\n{m}"
        );
        served.release_and_finish().await;
    }

    /// A thousand scrapes issue zero catalog statements: with the sampler
    /// parked on a one-hour interval its sample count stays at 1.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_thousand_scrapes_issue_zero_catalog_statements() {
        let dir = tempfile::TempDir::new().unwrap();
        let served = serve(config(dir.path(), 3600, WorkerKinds::default())).await;
        let shared = served.shared.as_ref().unwrap().upgrade().unwrap();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        while shared.samples_taken() < 1 {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the sampler never ran"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let before = shared.samples_taken();
        for _ in 0..1000 {
            let m = scrape(&served.health_addr).await;
            assert!(sample(&m, "jammi_worker_claim_loop_up").is_some());
        }
        assert_eq!(
            shared.samples_taken(),
            before,
            "a scrape must never sample the catalog itself"
        );
        served.release_and_finish().await;
    }

    /// `jammi_worker_jobs_in_flight` is 1 while a loop-claimed job runs
    /// (and `jammi_jobs_running{kind}` follows within a tick), and 0 while
    /// an inline `run_now` is in flight — the inline claim is never the
    /// loop's.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn in_flight_gauge_is_one_during_a_loop_claimed_job_and_zero_during_run_now() {
        // Loop-claimed.
        let dir = tempfile::TempDir::new().unwrap();
        let served = serve(config(dir.path(), 1, WorkerKinds::default())).await;
        add_training(&served.session).await;
        let shared = served.shared.as_ref().unwrap().upgrade().unwrap();
        served.session.enqueue(fine_tune(20_000), 0).await.unwrap();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
        while shared.in_flight() != 1 {
            assert!(tokio::time::Instant::now() < deadline, "never in flight");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let m = scrape(&served.health_addr).await;
        assert_eq!(sample(&m, "jammi_worker_jobs_in_flight"), Some(1.0), "{m}");
        assert_eq!(sample(&m, "jammi_worker_claim_loop_up"), Some(1.0), "{m}");
        let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
        loop {
            let m = scrape(&served.health_addr).await;
            if sample(&m, r#"jammi_jobs_running{kind="fine_tune"}"#) == Some(1.0) {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "running gauge never moved:\n{m}"
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        served.release_and_finish().await;

        // Inline `run_now`, parked before dispatch.
        let dir = tempfile::TempDir::new().unwrap();
        let served = serve(config(dir.path(), 1, WorkerKinds::default())).await;
        served
            .session
            .add_source(
                "patents-gauge",
                SourceType::File,
                SourceConnection {
                    url: Some(fixture_url("patents.parquet")),
                    format: Some(FileFormat::Parquet),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let park = compute_test_hooks::arm(
            "patents-gauge",
            compute_test_hooks::ParkPoint::BeforeDispatch,
        );
        let runner = Arc::clone(&served.session);
        let run = tokio::spawn(async move {
            runner
                .run_now(ComputeSpec::Infer {
                    source_id: "patents-gauge".into(),
                    model_id: "local:/nonexistent/model/for-the-gauge-oracle".into(),
                    task: ModelTask::TextEmbedding,
                    content_columns: vec!["abstract".into()],
                    key_column: "id".into(),
                    cache: CachePolicy::Bypass,
                })
                .await
        });
        park.wait_parked().await;
        let m = scrape(&served.health_addr).await;
        assert_eq!(sample(&m, "jammi_worker_jobs_in_flight"), Some(0.0), "{m}");
        park.release();
        let _ = run.await.unwrap();
        served.release_and_finish().await;
    }
}
