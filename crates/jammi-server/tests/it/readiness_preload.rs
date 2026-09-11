//! OPS (#482): warm-before-ready — `[server] preload_models` is loaded
//! before `/readyz` reports ready, the claim loop is parked at its gate
//! (`workers.state = warming`) until every entry is cached, a failed entry
//! is a startup error, and a signal during the preload exits without
//! serving. Base: the list has no reader (`/readyz` 200 with a cold cache,
//! an unloadable entry serves anyway).

use std::sync::Arc;
use std::time::Duration;

use jammi_ai::fine_tune::spec::{TrainingCommon, TrainingSpec};
use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod};
use jammi_ai::model::cache::preload_test_hooks;
use jammi_ai::model::{ModelSource, ModelTask};
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::Catalog;
use jammi_db::config::{JammiConfig, PreloadEntry};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_server::grpc::proto::job::job_service_client::JobServiceClient;
use jammi_server::grpc::proto::job::ListWorkersRequest;
use jammi_server::runtime::{OssServer, ServerError, ShutdownOutcome};
use jammi_test_utils::{cookbook_fixture, fixture_url, test_config};
use tokio::sync::watch;

fn tiny_bert() -> String {
    format!("local:{}", cookbook_fixture("tiny_bert").display())
}

fn config(dir: &std::path::Path, preload: Vec<PreloadEntry>) -> JammiConfig {
    let mut cfg = test_config(dir);
    cfg.server.health_listen = "127.0.0.1:0".into();
    cfg.server.flight_listen = "127.0.0.1:0".into();
    cfg.server.preload_models = preload;
    cfg
}

struct Served {
    session: Arc<InferenceSession>,
    flight_addr: std::net::SocketAddr,
    health_addr: std::net::SocketAddr,
    drain_tx: watch::Sender<bool>,
    release_tx: watch::Sender<bool>,
    task: tokio::task::JoinHandle<Result<ShutdownOutcome, ServerError>>,
}

async fn serve(cfg: JammiConfig) -> Served {
    let server = OssServer::new(cfg).await.expect("OssServer::new");
    let session = server.session();
    let bound = server.bind().await.expect("bind");
    let flight_addr = bound.flight_addr();
    let health_addr = bound.health_addr();
    let (drain_tx, drain_rx) = watch::channel(false);
    let (release_tx, release_rx) = watch::channel(false);
    let task = tokio::spawn(bound.serve_with_signals(drain_rx, release_rx));
    // Wait for the side-channel — unless the serve task has already ended
    // (a preload startup error exits before anything is served).
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    while !task.is_finished()
        && reqwest::get(format!("http://{health_addr}/healthz"))
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
        flight_addr,
        health_addr,
        drain_tx,
        release_tx,
        task,
    }
}

async fn get(url: String) -> (u16, serde_json::Value) {
    let resp = reqwest::get(url).await.expect("GET");
    let status = resp.status().as_u16();
    let body = resp.json::<serde_json::Value>().await.unwrap_or_default();
    (status, body)
}

/// This process's `workers.state` as `ListWorkers` reports it over the wire
/// — the K4 half of the row read below.
///
/// Only callable once the process is SERVING: `serve_with_signals` binds the
/// gRPC listener at `bind` but starts accepting on it only after the preload
/// completes, so a connect during the warming window completes at TCP and
/// then waits forever for an HTTP/2 handshake nothing is there to answer.
/// `warming` is therefore a ROW fact (B4: the row is what another process
/// reads) — a peer replica sharing the catalog sees this one as `warming`
/// over ITS wire; this process cannot report its own.
async fn wire_worker_state(addr: std::net::SocketAddr) -> Option<String> {
    let channel = tonic::transport::Endpoint::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect()
        .await
        .expect("the gRPC listener answers during the preload");
    JobServiceClient::new(channel)
        .list_workers(ListWorkersRequest {})
        .await
        .expect("list_workers")
        .into_inner()
        .workers
        .into_iter()
        .next()
        .map(|w| w.state)
}

async fn worker_state(session: &InferenceSession) -> Option<String> {
    session
        .catalog()
        .list_workers()
        .await
        .unwrap()
        .into_iter()
        .next()
        .map(|w| w.state)
}

async fn wait_worker_state(session: &InferenceSession, want: &str) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        let state = worker_state(session).await;
        if state.as_deref() == Some(want) {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "workers.state never read {want}; last {state:?}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

fn one_epoch_fine_tune() -> jammi_ai::jobs::JobSpec {
    TrainingSpec::FineTune {
        source: "training".into(),
        columns: vec!["text_a".into(), "text_b".into(), "score".into()],
        method: FineTuneMethod::Lora,
        task: ModelTask::TextEmbedding,
        common: TrainingCommon {
            base_model: tiny_bert(),
            config: FineTuneConfig {
                epochs: 1,
                batch_size: 8,
                lora_rank: 4,
                warmup_steps: 0,
                ..Default::default()
            },
        },
    }
    .into()
}

/// Acceptance 8: `/readyz` is 503 "preloading 0/1" while the listed model
/// is parked before its load, `/healthz` is 200 throughout, the worker's
/// row reads `warming` and a queued job stays `queued`; once the preload
/// completes `/readyz` is 200, the row flips to `claiming` and the job
/// runs. Also `worker_does_not_claim_until_preload_completes`,
/// `healthz_is_200_during_preload`, and
/// `preload_entry_with_explicit_task_loads_with_that_task`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn readyz_is_503_until_preload_models_are_loaded_then_200() {
    let dir = tempfile::TempDir::new().unwrap();
    let source_key = ModelSource::parse(&tiny_bert()).to_string();
    let park = preload_test_hooks::arm(&source_key, preload_test_hooks::ParkPoint::BeforeLoad);
    let served = serve(config(
        dir.path(),
        vec![PreloadEntry {
            id: tiny_bert(),
            task: Some(ModelTask::TextEmbedding),
        }],
    ))
    .await;
    park.wait_parked().await;

    let (status, body) = get(format!("http://{}/readyz", served.health_addr)).await;
    assert_eq!(status, 503, "{body}");
    assert_eq!(body["detail"], "preloading 0/1");
    let (status, body) = get(format!("http://{}/healthz", served.health_addr)).await;
    assert_eq!(status, 200, "liveness is 200 during preload: {body}");
    // Acceptance 9's `warming` leg is asserted on the ROW: this process is
    // not accepting gRPC yet (see `wire_worker_state`), and the row is what
    // a peer's `ListWorkers` reads.
    wait_worker_state(&served.session, "warming").await;

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
        .enqueue(one_epoch_fine_tune(), 0)
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert_eq!(
        handle.status().await.unwrap(),
        "queued",
        "no job is claimed until the process is warm"
    );
    assert_eq!(
        worker_state(&served.session).await.as_deref(),
        Some("warming")
    );

    park.release();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    loop {
        let (status, _) = get(format!("http://{}/readyz", served.health_addr)).await;
        if status == 200 {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "/readyz never became 200"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    wait_worker_state(&served.session, "claiming").await;
    assert_eq!(
        wire_worker_state(served.flight_addr).await.as_deref(),
        Some("claiming"),
        "ListWorkers must report the same `claiming` the row carries"
    );
    tokio::time::timeout(Duration::from_secs(120), handle.wait())
        .await
        .expect("the job runs once warm")
        .expect("completes");
    // gRPC is served only after the preload.
    assert!(
        tonic::transport::Endpoint::from_shared(format!("http://{}", served.flight_addr))
            .unwrap()
            .connect()
            .await
            .is_ok()
    );

    let _ = served.drain_tx.send(true);
    let outcome = tokio::time::timeout(Duration::from_secs(60), served.task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(outcome, ShutdownOutcome::Drained { .. }));
}

/// An unloadable entry is a startup error: `serve` returns
/// `Err(ServerError::Preload)`, gRPC never served, the worker's row
/// deleted before the session closed. Base: the list has no reader, the
/// server serves.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn preload_of_an_unloadable_model_is_a_startup_error() {
    let dir = tempfile::TempDir::new().unwrap();
    let served = serve(config(
        dir.path(),
        vec![PreloadEntry {
            id: "local:/nonexistent/model/for-the-preload-oracle".into(),
            task: Some(ModelTask::TextEmbedding),
        }],
    ))
    .await;
    let outcome = tokio::time::timeout(Duration::from_secs(60), served.task)
        .await
        .unwrap()
        .unwrap();
    match outcome {
        Err(ServerError::Preload { id, reason }) => {
            assert!(id.contains("nonexistent"), "{id}");
            assert!(!reason.is_empty());
        }
        other => panic!("expected Err(Preload), got {other:?}"),
    }
    assert!(
        tonic::transport::Endpoint::from_shared(format!("http://{}", served.flight_addr))
            .unwrap()
            .connect()
            .await
            .is_err(),
        "gRPC must never have served"
    );
    let catalog = Catalog::open(dir.path()).await.unwrap();
    assert!(
        catalog.list_workers().await.unwrap().is_empty(),
        "the workers row is deleted on the preload exit"
    );
}

/// A bare id with no `models` row cannot resolve a task: a typed startup
/// error naming the fix.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn preload_of_a_bare_id_with_no_models_row_is_a_startup_error() {
    let dir = tempfile::TempDir::new().unwrap();
    let served = serve(config(
        dir.path(),
        vec![PreloadEntry {
            id: "no-such-model-row".into(),
            task: None,
        }],
    ))
    .await;
    let outcome = tokio::time::timeout(Duration::from_secs(60), served.task)
        .await
        .unwrap()
        .unwrap();
    match outcome {
        Err(ServerError::Preload { id, reason }) => {
            assert_eq!(id, "no-such-model-row");
            assert!(reason.contains("no models row"), "{reason}");
        }
        other => panic!("expected Err(Preload), got {other:?}"),
    }
}

/// A signal during the preload exits 0 (`Drained` / `Released`) without
/// ever serving gRPC; the worker's row is deleted and awaited before the
/// session closes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn signal_during_preload_exits_without_serving() {
    for release in [false, true] {
        let dir = tempfile::TempDir::new().unwrap();
        let source_key = ModelSource::parse(&tiny_bert()).to_string();
        let park = preload_test_hooks::arm(&source_key, preload_test_hooks::ParkPoint::BeforeLoad);
        let served = serve(config(
            dir.path(),
            vec![PreloadEntry {
                id: tiny_bert(),
                task: Some(ModelTask::TextEmbedding),
            }],
        ))
        .await;
        park.wait_parked().await;
        wait_worker_state(&served.session, "warming").await;
        if release {
            let _ = served.release_tx.send(true);
        } else {
            let _ = served.drain_tx.send(true);
        }
        let outcome = tokio::time::timeout(Duration::from_secs(60), served.task)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        if release {
            assert_eq!(outcome, ShutdownOutcome::Released);
        } else {
            assert_eq!(
                outcome,
                ShutdownOutcome::Drained {
                    worker_joined: true
                }
            );
        }
        assert!(
            tonic::transport::Endpoint::from_shared(format!("http://{}", served.flight_addr))
                .unwrap()
                .connect()
                .await
                .is_err(),
            "gRPC must never have served"
        );
        let catalog = Catalog::open(dir.path()).await.unwrap();
        assert!(
            catalog.list_workers().await.unwrap().is_empty(),
            "no phantom warming row survives the preload exit"
        );
        park.release();
    }
}

/// O4/W2: `ShutdownOutcome::Released` must come from a path that actually
/// issued the release mechanism — not merely from whichever signal happened
/// to preempt the preload. Arms the `ReleaseAt2e` rendezvous (the first
/// statement of `EmbeddedWorker::release_and_stop`'s 2e) before sending the
/// release signal: base called `stop_and_join` unconditionally on this arm,
/// so 2e is never reached and the rendezvous never fires, even though the
/// returned outcome already read `Released`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn release_signal_during_preload_actually_releases() {
    let dir = tempfile::TempDir::new().unwrap();
    let source_key = ModelSource::parse(&tiny_bert()).to_string();
    let load_park = preload_test_hooks::arm(&source_key, preload_test_hooks::ParkPoint::BeforeLoad);
    let served = serve(config(
        dir.path(),
        vec![PreloadEntry {
            id: tiny_bert(),
            task: Some(ModelTask::TextEmbedding),
        }],
    ))
    .await;
    load_park.wait_parked().await;
    wait_worker_state(&served.session, "warming").await;
    let instance_id = served.session.instance_id().to_string();
    let release_fired = jammi_ai::fine_tune::worker::loop_test_hooks::arm_rendezvous(
        &instance_id,
        jammi_ai::fine_tune::worker::loop_test_hooks::Rendezvous::ReleaseAt2e,
    );

    let _ = served.release_tx.send(true);

    tokio::time::timeout(Duration::from_secs(10), release_fired.wait_fired())
        .await
        .expect(
            "the preload-exit RELEASE arm must call release_and_stop (2e fires the rendezvous), \
             not stop_and_join",
        );
    let outcome = tokio::time::timeout(Duration::from_secs(60), served.task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(outcome, ShutdownOutcome::Released);
    load_park.release();
}
