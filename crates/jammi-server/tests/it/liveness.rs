//! OPS (#482): `/healthz` liveness — 503 within one heartbeat of the lease
//! keeper thread dying, 503 when the claim loop task panics, 200 while
//! draining (with `/readyz` 503) — off a full `OssServer`. Base: `/healthz`
//! is a stateless 200.

use std::sync::Arc;
use std::time::Duration;

use jammi_ai::fine_tune::spec::{TrainingCommon, TrainingSpec};
use jammi_ai::fine_tune::worker::loop_test_hooks;
use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod};
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::config::JammiConfig;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_server::runtime::{OssServer, ServerError, ShutdownOutcome};
use jammi_test_utils::{cookbook_fixture, fixture_url, test_config};
use tokio::sync::watch;

struct Served {
    session: Arc<InferenceSession>,
    health_addr: std::net::SocketAddr,
    drain_tx: watch::Sender<bool>,
    release_tx: watch::Sender<bool>,
    task: tokio::task::JoinHandle<Result<ShutdownOutcome, ServerError>>,
}

fn config(dir: &std::path::Path) -> JammiConfig {
    let mut cfg = test_config(dir);
    cfg.server.health_listen = "127.0.0.1:0".into();
    cfg.server.flight_listen = "127.0.0.1:0".into();
    cfg.lease.duration_secs = 3;
    cfg.lease.heartbeat_secs = 1;
    cfg
}

async fn serve(cfg: JammiConfig) -> Served {
    let server = OssServer::new(cfg).await.expect("OssServer::new");
    let session = server.session();
    let bound = server.bind().await.expect("bind");
    let health_addr = bound.health_addr();
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
        drain_tx,
        release_tx,
        task,
    }
}

async fn healthz(addr: &std::net::SocketAddr) -> (u16, serde_json::Value) {
    let resp = reqwest::get(format!("http://{addr}/healthz"))
        .await
        .expect("GET");
    let status = resp.status().as_u16();
    let body = resp.json::<serde_json::Value>().await.unwrap_or_default();
    (status, body)
}

async fn readyz_status(addr: &std::net::SocketAddr) -> (u16, serde_json::Value) {
    let resp = reqwest::get(format!("http://{addr}/readyz"))
        .await
        .expect("GET");
    let status = resp.status().as_u16();
    let body = resp.json::<serde_json::Value>().await.unwrap_or_default();
    (status, body)
}

async fn wait_healthz(
    addr: &std::net::SocketAddr,
    want: u16,
    bound: Duration,
) -> serde_json::Value {
    let deadline = tokio::time::Instant::now() + bound;
    loop {
        let (status, body) = healthz(addr).await;
        if status == want {
            return body;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "/healthz never reached {want}; last {status} {body}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Acceptance 7: 503 within one `heartbeat_secs` of the keeper thread
/// dying (`kill_thread_for_test` lands at its next tick), naming the keeper.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn healthz_flips_to_503_within_one_heartbeat_after_the_keeper_thread_dies() {
    let dir = tempfile::TempDir::new().unwrap();
    let served = serve(config(dir.path())).await;
    let (status, body) = healthz(&served.health_addr).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["lease_keeper"], true);
    assert_eq!(body["claim_loop"], "running");

    served.session.lease_keeper().kill_thread_for_test();
    let killed_at = tokio::time::Instant::now();
    // The kill lands at the next heartbeat tick (1 s); the flip is the
    // exit guard's, so it is visible within one heartbeat plus a quantum.
    let body = wait_healthz(&served.health_addr, 503, Duration::from_secs(3)).await;
    assert!(
        killed_at.elapsed() < Duration::from_secs(2),
        "the flip took {:?}, more than one heartbeat",
        killed_at.elapsed()
    );
    assert_eq!(body["status"], "unhealthy");
    assert_eq!(body["lease_keeper"], false);

    let _ = served.drain_tx.send(true);
    let _ = served.release_tx.send(true);
    let _ = tokio::time::timeout(Duration::from_secs(60), served.task).await;
}

/// A DRAIN is not a fault: `/healthz` stays 200 (loop `running`, a job in
/// flight) while `/readyz` says 503 "draining".
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn healthz_stays_200_while_draining() {
    let dir = tempfile::TempDir::new().unwrap();
    let served = serve(config(dir.path())).await;
    served
        .session
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
    let handle = served
        .session
        .enqueue(
            TrainingSpec::FineTune {
                source: "training".into(),
                columns: vec!["text_a".into(), "text_b".into(), "score".into()],
                method: FineTuneMethod::Lora,
                task: ModelTask::TextEmbedding,
                common: TrainingCommon {
                    base_model: format!("local:{}", cookbook_fixture("tiny_bert").display()),
                    config: FineTuneConfig {
                        epochs: 20_000,
                        batch_size: 8,
                        lora_rank: 4,
                        warmup_steps: 0,
                        ..Default::default()
                    },
                },
            }
            .into(),
            0,
        )
        .await
        .unwrap();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    while handle.status().await.unwrap() != "running" {
        assert!(tokio::time::Instant::now() < deadline);
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let _ = served.drain_tx.send(true);
    tokio::time::sleep(Duration::from_millis(500)).await;
    let (status, body) = healthz(&served.health_addr).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["claim_loop"], "running");
    let (ready, body) = readyz_status(&served.health_addr).await;
    assert_eq!(ready, 503, "{body}");
    assert_eq!(body["detail"], "draining");

    let _ = served.release_tx.send(true);
    let outcome = tokio::time::timeout(Duration::from_secs(30), served.task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(outcome, ShutdownOutcome::Released);
}

/// A panicking claim loop task is a fault: the exit guard reports
/// `Failed` and `/healthz` reads 503 with `claim_loop: "failed"` while the
/// keeper is still alive.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn healthz_503_when_the_claim_loop_task_panics() {
    let dir = tempfile::TempDir::new().unwrap();
    let served = serve(config(dir.path())).await;
    loop_test_hooks::arm_panic_at_next_tick(served.session.instance_id());
    let body = wait_healthz(&served.health_addr, 503, Duration::from_secs(10)).await;
    assert_eq!(body["status"], "unhealthy");
    assert_eq!(body["claim_loop"], "failed");
    assert_eq!(body["lease_keeper"], true);

    let _ = served.drain_tx.send(true);
    let outcome = tokio::time::timeout(Duration::from_secs(60), served.task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(
        matches!(
            outcome,
            ShutdownOutcome::Drained {
                worker_joined: false
            }
        ),
        "a failed loop has nothing to join: {outcome:?}"
    );
}
