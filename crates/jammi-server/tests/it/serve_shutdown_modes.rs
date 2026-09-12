//! OPS (#482): the two shutdown modes on the SERVER — `BoundServer::
//! serve_with_signals` driven by the drain / release watches (the exact
//! coupling the OS-signal watcher uses), plus the `jammi-server release`
//! subcommand against a real child process.
//!
//! Every oracle observes the catalog row and the loop's own state, never a
//! wall clock beyond the heartbeat bounds the design names. The server's
//! session is closed by the time `serve_with_signals` returns, so post-
//! shutdown rows are read through a fresh `Catalog` over the same directory.

use std::path::Path;
use std::sync::{Arc, Weak};
use std::time::Duration;

use futures::StreamExt;
use jammi_ai::fine_tune::worker::training_test_hooks;
use jammi_ai::jobs::compute_test_hooks;
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::jobs_repo::JobRecord;
use jammi_db::catalog::status::JobStatus;
use jammi_db::catalog::Catalog;
use jammi_db::config::JammiConfig;
use jammi_server::grpc::proto::catalog::catalog_service_client::CatalogServiceClient;
use jammi_server::grpc::proto::catalog::{
    AddSourceRequest, FileFormat, SourceConnection, SourceKind,
};
use jammi_server::grpc::proto::embedding::embedding_service_client::EmbeddingServiceClient;
use jammi_server::grpc::proto::embedding::{GenerateEmbeddingsRequest, Modality};
use jammi_server::grpc::proto::inference::ModelTask;
use jammi_server::grpc::proto::job::job_service_client::JobServiceClient;
use jammi_server::grpc::proto::job::submit_job_request::Spec;
use jammi_server::grpc::proto::job::{JobStatusRequest, ListWorkersRequest, SubmitJobRequest};
use jammi_server::grpc::proto::training::{FineTuneConfig, FineTuneMethod, FineTuneSpec};
use jammi_server::grpc::proto::trigger::trigger_service_client::TriggerServiceClient;
use jammi_server::grpc::proto::trigger::{SubscribeRequest, TopicName};
use jammi_server::runtime::{OssServer, ServerError, ShutdownOutcome};
use jammi_test_utils::{cookbook_fixture, fixture_url, test_config};
use tempfile::TempDir;
use tokio::sync::watch;
use tonic::transport::Channel;
use tonic::Code;

use super::common::grpc::channel;

fn tiny_bert_model_id() -> String {
    format!("local:{}", cookbook_fixture("tiny_bert").display())
}

/// Lease / heartbeat seconds for one server.
#[derive(Clone, Copy)]
struct Timing {
    lease: u64,
    heartbeat: u64,
}
const DEFAULT_TIMING: Timing = Timing {
    lease: 30,
    heartbeat: 10,
};
const FAST_TIMING: Timing = Timing {
    lease: 3,
    heartbeat: 1,
};

fn server_config(dir: &Path, timing: Timing, worker_enabled: bool) -> JammiConfig {
    let mut cfg = test_config(dir);
    cfg.server.health_listen = "127.0.0.1:0".to_string();
    cfg.server.flight_listen = "127.0.0.1:0".to_string();
    cfg.lease.duration_secs = timing.lease;
    cfg.lease.heartbeat_secs = timing.heartbeat;
    cfg.worker.enabled = worker_enabled;
    cfg
}

/// A full `OssServer` serving through `serve_with_signals`, with the two
/// signal watches in the test's hands.
struct Served {
    session: Arc<InferenceSession>,
    /// The server's own metrics registry — readable after the side-channel
    /// has stopped.
    metrics: Arc<jammi_server::routes::health::MetricsRegistry>,
    /// A weak handle on the embedded worker's shared state, captured BEFORE
    /// `serve_with_signals` takes ownership of the `EmbeddedWorker` guard —
    /// `WorkerShared` stays alive for the guard's whole lifetime (owned by
    /// both the guard and the loop task itself), so this stays upgradable
    /// for as long as `task` has not returned, letting a test observe
    /// [`jammi_ai::fine_tune::worker::LoopState`] directly rather than
    /// inferring it from elapsed wall-clock time (O1). `None` when the
    /// fixture disabled the worker.
    worker_shared: Option<Weak<jammi_ai::fine_tune::worker::WorkerShared>>,
    flight_addr: std::net::SocketAddr,
    health_addr: std::net::SocketAddr,
    drain_tx: watch::Sender<bool>,
    release_tx: watch::Sender<bool>,
    task: tokio::task::JoinHandle<Result<ShutdownOutcome, ServerError>>,
}

impl Served {
    fn drain(&self) {
        let _ = self.drain_tx.send(true);
    }
    fn release(&self) {
        let _ = self.release_tx.send(true);
    }
    async fn finish(self, bound: Duration) -> Result<ShutdownOutcome, ServerError> {
        tokio::time::timeout(bound, self.task)
            .await
            .expect("the serve task must end within the bound")
            .expect("the serve task must not panic")
    }

    /// Poll-until-predicate (never a blind sleep) for the loop task's OWN
    /// reported [`LoopState`](jammi_ai::fine_tune::worker::LoopState) to
    /// leave `Running` — O1's mechanism assertion: the loop task's actual
    /// termination, not merely that some bounded amount of wall-clock time
    /// has passed (a skipped-abort defect and a real abort both satisfy a
    /// wall-clock bound when the timeout backstop is one heartbeat; only
    /// the mechanism distinguishes them). Panics with the observed state
    /// (or "no worker") if `bound` elapses first — the RED signal at base,
    /// where the loop never leaves `Running` because nothing ever aborts
    /// it.
    async fn wait_loop_state_left_running(
        &self,
        bound: Duration,
    ) -> jammi_ai::fine_tune::worker::LoopState {
        let shared = self
            .worker_shared
            .as_ref()
            .expect("this fixture always spawns a worker")
            .clone();
        let deadline = tokio::time::Instant::now() + bound;
        loop {
            let Some(shared) = shared.upgrade() else {
                panic!(
                    "the worker's shared state vanished before its loop state was ever observed \
                     as non-Running -- the guard (and the loop task's own Arc) is gone without \
                     the loop ever reporting termination"
                );
            };
            let state = shared.loop_state();
            if state != jammi_ai::fine_tune::worker::LoopState::Running {
                return state;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "the loop task's own LoopState never left Running within {bound:?} -- it is \
                 still running (or detached), not aborted"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

async fn serve(dir: &Path, timing: Timing) -> Served {
    serve_with_config(server_config(dir, timing, true)).await
}

async fn serve_with_config(cfg: JammiConfig) -> Served {
    let server = OssServer::new(cfg).await.expect("OssServer::new");
    let session = server.session();
    let metrics = server.metrics();
    let bound = server.bind().await.expect("bind");
    let flight_addr = bound.flight_addr();
    let health_addr = bound.health_addr();
    // Captured BEFORE `serve_with_signals` consumes `bound` — see the
    // field doc on `Served::worker_shared`.
    let worker_shared = bound.worker_shared();
    let (drain_tx, drain_rx) = watch::channel(false);
    let (release_tx, release_rx) = watch::channel(false);
    let task = tokio::spawn(bound.serve_with_signals(drain_rx, release_rx));
    // The listeners are bound; wait until the side-channel actually answers
    // so a client's first connect never races the serve task's first poll.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        if reqwest::get(format!("http://{health_addr}/healthz"))
            .await
            .is_ok()
        {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the health side-channel never came up"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    Served {
        session,
        metrics,
        worker_shared,
        flight_addr,
        health_addr,
        drain_tx,
        release_tx,
        task,
    }
}

/// The current value of `jammi_grpc_refused_total{reason=<reason>}` in the
/// server's own registry.
fn refused(metrics: &jammi_server::routes::health::MetricsRegistry, reason: &str) -> f64 {
    metrics
        .inner()
        .gather()
        .into_iter()
        .filter(|f| f.name() == "jammi_grpc_refused_total")
        .flat_map(|f| f.get_metric().to_vec())
        .filter(|m| {
            m.get_label()
                .iter()
                .any(|l| l.name() == "reason" && l.value() == reason)
        })
        .map(|m| m.get_counter().get_value())
        .sum()
}

async fn add_source(ch: Channel, source_id: &str, file: &str, format: FileFormat) {
    CatalogServiceClient::new(ch)
        .add_source(AddSourceRequest {
            source_id: source_id.into(),
            source_kind: SourceKind::File as i32,
            connection: Some(SourceConnection {
                url: fixture_url(file),
                format: format as i32,
            }),
        })
        .await
        .expect("add_source");
}

async fn submit_fine_tune(ch: Channel, epochs: u32) -> String {
    JobServiceClient::new(ch)
        .submit_job(SubmitJobRequest {
            spec: Some(Spec::FineTune(FineTuneSpec {
                source: "training".into(),
                columns: vec!["text_a".into(), "text_b".into(), "score".into()],
                method: FineTuneMethod::Lora as i32,
                task: ModelTask::TextEmbedding as i32,
            })),
            base_model: tiny_bert_model_id(),
            config: Some(FineTuneConfig {
                epochs: Some(epochs),
                batch_size: Some(8),
                lora_rank: Some(4),
                warmup_steps: Some(0),
                ..Default::default()
            }),
            idempotency_key: String::new(),
        })
        .await
        .expect("submit_job")
        .into_inner()
        .job_id
}

async fn wait_status(catalog: &Catalog, job_id: &str, want: &str, bound: Duration) -> JobRecord {
    let deadline = tokio::time::Instant::now() + bound;
    loop {
        let row = catalog.get_job(job_id).await.expect("get_job");
        if row.status == want {
            return row;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "job {job_id} never reached {want}; last {row:?}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// The catalog of a server whose session has already closed.
async fn reopen(dir: &Path) -> Catalog {
    Catalog::open(dir).await.expect("reopen the catalog")
}

async fn get(url: String) -> (u16, serde_json::Value) {
    let resp = reqwest::get(url).await.expect("GET");
    let status = resp.status().as_u16();
    let body = resp.json::<serde_json::Value>().await.unwrap_or_default();
    (status, body)
}

/// `adapter.safetensors` under `root`, wherever the artifact store put it.
fn find_adapter(root: &Path) -> Option<std::path::PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).ok()?.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.file_name().is_some_and(|n| n == "adapter.safetensors")
                && !path.to_string_lossy().contains("_resume")
            {
                return Some(path);
            }
        }
    }
    None
}

async fn resume_epoch(session: &InferenceSession, job_id: &str) -> Option<u64> {
    let local = session
        .artifact_store()
        .fetch_resume_checkpoint(None, job_id)
        .await
        .unwrap()?;
    let state: serde_json::Value = serde_json::from_slice(
        &std::fs::read(local.dir().join("resume_state.json")).expect("resume_state.json"),
    )
    .unwrap();
    Some(
        state["last_completed_epoch"]
            .as_u64()
            .expect("integer last_completed_epoch"),
    )
}

/// Register an empty `events` topic so a `Subscribe` can idle on it.
async fn register_events_topic(session: &InferenceSession) {
    use arrow_schema::{DataType, Field, Schema};
    let topic = jammi_db::trigger::TopicDefinition {
        id: jammi_db::trigger::TopicId::new(),
        name: "events".into(),
        schema: Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)])),
        tenant: None,
        broker_metadata: Default::default(),
    };
    session
        .trigger_broker()
        .register_topic(&topic)
        .await
        .expect("broker register");
    session
        .topic_repo()
        .register_topic(&topic)
        .await
        .expect("catalog register");
}

async fn idle_subscribe(
    ch: Channel,
) -> tonic::Streaming<jammi_server::grpc::proto::trigger::SubscribedBatch> {
    TriggerServiceClient::new(ch)
        .subscribe(SubscribeRequest {
            topic: Some(TopicName {
                name: "events".into(),
            }),
            predicate: String::new(),
            from_offset: Some(0),
            tenant_id: String::new(),
            replay_only: false,
        })
        .await
        .expect("subscribe")
        .into_inner()
}

/// Drive an idle stream to its end: the terminal `Status` it closes with.
async fn stream_terminal(
    stream: &mut tonic::Streaming<jammi_server::grpc::proto::trigger::SubscribedBatch>,
    bound: Duration,
) -> tonic::Status {
    let deadline = tokio::time::Instant::now() + bound;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        match tokio::time::timeout(remaining, stream.next()).await {
            Ok(Some(Err(status))) => return status,
            Ok(Some(Ok(_))) => continue,
            Ok(None) => panic!("the idle stream ended without a terminal status"),
            Err(_) => panic!("the idle stream did not end within {bound:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// DRAIN
// ---------------------------------------------------------------------------

/// Acceptance 1: worker mid-job + DRAIN → the job lands `completed` under
/// the draining instance, attempt 1, with adapter bytes byte-identical to an
/// uninterrupted run; the serve task returns `Drained { worker_joined: true }`.
/// Base: the chain dropped the worker guard (abort) when serve returned and
/// the row was left `running`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sigterm_drains_the_in_flight_job_and_exits_drained() {
    // The uninterrupted control run.
    let control_dir = TempDir::new().unwrap();
    let control = serve(control_dir.path(), DEFAULT_TIMING).await;
    add_source(
        channel(control.flight_addr).await,
        "training",
        "training_pairs.csv",
        FileFormat::Csv,
    )
    .await;
    let control_job = submit_fine_tune(channel(control.flight_addr).await, 300).await;
    wait_status(
        control.session.catalog(),
        &control_job,
        &JobStatus::Completed.to_string(),
        Duration::from_secs(300),
    )
    .await;
    control.drain();
    assert!(matches!(
        control.finish(Duration::from_secs(60)).await.unwrap(),
        ShutdownOutcome::Drained { .. }
    ));
    let control_bytes =
        std::fs::read(find_adapter(control_dir.path()).expect("control adapter")).unwrap();

    // The drained run.
    let dir = TempDir::new().unwrap();
    let served = serve(dir.path(), DEFAULT_TIMING).await;
    add_source(
        channel(served.flight_addr).await,
        "training",
        "training_pairs.csv",
        FileFormat::Csv,
    )
    .await;
    let job_id = submit_fine_tune(channel(served.flight_addr).await, 300).await;
    wait_status(
        served.session.catalog(),
        &job_id,
        &JobStatus::Running.to_string(),
        Duration::from_secs(120),
    )
    .await;
    let instance_id = served.session.instance_id().to_string();
    served.drain();
    // The side-channel stays up through the drain and reports it.
    let (status, body) = get(format!("http://{}/readyz", served.health_addr)).await;
    assert_eq!(status, 503, "{body}");
    assert_eq!(body["detail"], "draining");
    let outcome = served.finish(Duration::from_secs(300)).await.unwrap();
    assert_eq!(
        outcome,
        ShutdownOutcome::Drained {
            worker_joined: true
        }
    );

    let catalog = reopen(dir.path()).await;
    let row = catalog.get_job(&job_id).await.unwrap();
    assert_eq!(row.status, JobStatus::Completed.to_string(), "{row:?}");
    assert_eq!(row.claimed_by.as_deref(), Some(instance_id.as_str()));
    assert_eq!((row.attempts, row.releases), (1, 0));
    assert!(
        catalog.list_workers().await.unwrap().is_empty(),
        "the joined loop's workers row is gone"
    );
    let drained_bytes = std::fs::read(find_adapter(dir.path()).expect("drained adapter")).unwrap();
    assert_eq!(
        drained_bytes, control_bytes,
        "a drained run's adapter must be byte-identical to an uninterrupted one"
    );
}

/// The gated join half must not stop the worker at t = 0: a job submitted
/// after serving began is claimed and completed by the still-live loop.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn worker_does_not_stop_before_drain_is_signalled() {
    let dir = TempDir::new().unwrap();
    let served = serve(dir.path(), DEFAULT_TIMING).await;
    tokio::time::sleep(Duration::from_millis(500)).await;
    add_source(
        channel(served.flight_addr).await,
        "training",
        "training_pairs.csv",
        FileFormat::Csv,
    )
    .await;
    let job_id = submit_fine_tune(channel(served.flight_addr).await, 1).await;
    wait_status(
        served.session.catalog(),
        &job_id,
        &JobStatus::Completed.to_string(),
        Duration::from_secs(120),
    )
    .await;
    served.drain();
    assert!(matches!(
        served.finish(Duration::from_secs(60)).await.unwrap(),
        ShutdownOutcome::Drained { .. }
    ));
}

/// Acceptance 5: an idle `Subscribe` stream is ended by the DRAIN with
/// `UNAVAILABLE` "server draining" within 5 s (base: the stream holds the
/// drain open forever) and the refusal is counted under `reason="draining"`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_idle_subscribe_stream_ends_with_unavailable_draining_on_sigterm() {
    let dir = TempDir::new().unwrap();
    let served = serve(dir.path(), DEFAULT_TIMING).await;
    register_events_topic(&served.session).await;
    let mut stream = idle_subscribe(channel(served.flight_addr).await).await;
    // Give the stream a moment to be genuinely open and idle.
    tokio::time::sleep(Duration::from_millis(300)).await;

    served.drain();
    let status = stream_terminal(&mut stream, Duration::from_secs(5)).await;
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert!(status.message().contains("server draining"), "{status:?}");
    assert_eq!(
        refused(&served.metrics, "draining"),
        1.0,
        "the drained stream is counted once"
    );
    assert!(matches!(
        served.finish(Duration::from_secs(60)).await.unwrap(),
        ShutdownOutcome::Drained { .. }
    ));
}

/// The RPC drain and the worker join run concurrently: with a long job in
/// flight, the idle stream is ended long before the job would finish (the
/// row is still `running` when the trailer arrives); a RELEASE then ends
/// the drain.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn drain_runs_rpc_drain_and_worker_join_concurrently() {
    let dir = TempDir::new().unwrap();
    let served = serve(dir.path(), FAST_TIMING).await;
    register_events_topic(&served.session).await;
    add_source(
        channel(served.flight_addr).await,
        "training",
        "training_pairs.csv",
        FileFormat::Csv,
    )
    .await;
    let job_id = submit_fine_tune(channel(served.flight_addr).await, 20_000).await;
    wait_status(
        served.session.catalog(),
        &job_id,
        &JobStatus::Running.to_string(),
        Duration::from_secs(120),
    )
    .await;
    let mut stream = idle_subscribe(channel(served.flight_addr).await).await;
    tokio::time::sleep(Duration::from_millis(300)).await;

    served.drain();
    let status = stream_terminal(&mut stream, Duration::from_secs(5)).await;
    assert_eq!(status.code(), Code::Unavailable);
    let row = served.session.catalog().get_job(&job_id).await.unwrap();
    assert_eq!(
        row.status,
        JobStatus::Running.to_string(),
        "the stream ended while the worker was still finishing its job"
    );
    assert!(
        !served.task.is_finished(),
        "the drain is still waiting on the worker"
    );

    served.release();
    let outcome = served.finish(Duration::from_secs(15)).await.unwrap();
    assert_eq!(outcome, ShutdownOutcome::Released);
}

// ---------------------------------------------------------------------------
// RELEASE
// ---------------------------------------------------------------------------

/// Acceptance 2: a second signal while draining releases within two
/// heartbeats — the row `running`/lease NULL/`releases 1` — and a fresh
/// process's `reclaim + claim_next` claims it at once with `attempts 2`.
///
/// R5 (round-2 REFINE, #482): O1 is made DETERMINISTIC by parking the loop
/// at `ParkPoint::BeforeHold` (the claim→hold prologue, after the claim's
/// own COMMIT, before the hold is registered) instead of racing a wall
/// clock against a real training run. Parked there, the loop cannot bail
/// COOPERATIVELY — the trainer never starts, so it can never reach an
/// epoch boundary and report `Stopped` — only an unconditional abort can
/// end it, on EITHER of two arms: DRAIN's `stop_and_join` takes the handle
/// and suspends on the parked loop's own terminal state; when RELEASE
/// preempts it, `tokio::select!` drops that suspended future, and
/// `TakenHandle::drop` restores the handle as `LoopTask::Abandoned` rather
/// than losing it to a bare detach (the F1 defect this state type closes);
/// `release_and_stop`'s 2e then finds `Abandoned`, `taken.reclaimed()` is
/// true, and aborts it unconditionally. Base (`cbd427b4`, prior to the
/// state-type fix): the same preemption instead loses the handle to a bare
/// `Option::take()` cancelled mid-await, so 2e's `if let Some(handle) =
/// self.take_handle()` finds `None` and skips its abort arm entirely — the
/// parked loop then runs on forever, detached, and its own `LoopState`
/// never leaves `Running` until this test's bound times out. (The prior
/// version of this oracle raced a real 20 000-epoch trainer against a
/// 300 ms `DRAIN`→`RELEASE` gap instead of a park, and — because the
/// trainer CAN legitimately bail cooperatively at its next epoch boundary
/// once 2b's keeper pass flips its hold `lost` before any abort lands —
/// flaked once in 33 paired runs asserting `Aborted` when the relay's own
/// recorded base run observed `Stopped`; that recorded run is what the
/// comment above described, not a fabricated narrative.)
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sigint_while_draining_releases_and_returns_released_within_two_heartbeats() {
    let dir = TempDir::new().unwrap();
    let served = serve(dir.path(), FAST_TIMING).await;
    add_source(
        channel(served.flight_addr).await,
        "training",
        "training_pairs.csv",
        FileFormat::Csv,
    )
    .await;
    let job_id = submit_fine_tune(channel(served.flight_addr).await, 20_000).await;
    // Armed immediately after submit returns, before the already-running
    // loop's NEXT poll tick (bounded below by `[worker] idle_poll_secs`,
    // default 1 s — the loop is already sleeping out its current tick with
    // an empty queue, so this always wins the race in practice).
    let park = jammi_ai::fine_tune::worker::loop_test_hooks::arm(
        &job_id,
        jammi_ai::fine_tune::worker::loop_test_hooks::ParkPoint::BeforeHold,
    );
    tokio::time::timeout(Duration::from_secs(30), park.wait_parked())
        .await
        .expect("the loop must reach the claim-to-hold prologue for this job");
    let instance_id = served.session.instance_id().to_string();

    served.drain();
    // Lets DRAIN's `gated_join` actually reach and suspend inside
    // `stop_and_join`'s `state_rx.wait_for(..)` before RELEASE preempts it
    // — the exact SIGTERM-then-SIGINT hazard F1 names.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let released_at = tokio::time::Instant::now();
    served.release();

    // O1's mechanism assertion: the loop task's OWN reported termination,
    // never elapsed wall-clock time alone.
    let loop_state = served
        .wait_loop_state_left_running(Duration::from_secs(FAST_TIMING.heartbeat + 3))
        .await;
    assert_eq!(
        loop_state,
        jammi_ai::fine_tune::worker::LoopState::Aborted,
        "the loop task must be ABORTED by the RELEASE that preempted the DRAIN's \
         stop_and_join, not left running detached"
    );

    let outcome = served
        .finish(Duration::from_secs(2 * FAST_TIMING.heartbeat + 8))
        .await
        .unwrap();
    assert_eq!(outcome, ShutdownOutcome::Released);
    let took = released_at.elapsed();

    let catalog = reopen(dir.path()).await;
    let row = catalog.get_job(&job_id).await.unwrap();
    assert_eq!(row.status, JobStatus::Running.to_string(), "{row:?}");
    assert_eq!(row.claimed_by.as_deref(), Some(instance_id.as_str()));
    assert!(row.lease_expires_at.is_none(), "{row:?}");
    assert_eq!((row.attempts, row.releases), (1, 1));
    assert!(
        took < Duration::from_secs(2 * FAST_TIMING.heartbeat + 3),
        "RELEASE took {took:?}, more than two heartbeats"
    );

    // A fresh process claims it at once — no lease window to wait out.
    assert_eq!(
        catalog
            .reclaim_expired_jobs(Duration::from_secs(FAST_TIMING.lease), 3)
            .await
            .unwrap(),
        1
    );
    let successor = catalog
        .claim_next("successor", &["fine_tune"], Duration::from_secs(30))
        .await
        .unwrap()
        .expect("claimable");
    assert_eq!(successor.job_id, job_id);
    assert_eq!(successor.attempts, 2);
}

/// A RELEASE preempts a DRAIN blocked on an in-flight unary: an inline
/// `GenerateEmbeddings` (`run_now`) parked before dispatch holds tonic's
/// graceful shutdown open; the release severs it and returns within two
/// heartbeats; the inline row is untouched (live lease, `releases 0`).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn release_preempts_a_drain_blocked_on_an_in_flight_unary() {
    let dir = TempDir::new().unwrap();
    let served = serve(dir.path(), FAST_TIMING).await;
    add_source(
        channel(served.flight_addr).await,
        "patents",
        "patents.parquet",
        FileFormat::Parquet,
    )
    .await;
    let park = compute_test_hooks::arm("patents", compute_test_hooks::ParkPoint::BeforeDispatch);
    let ch = channel(served.flight_addr).await;
    let mut unary = tokio::spawn(async move {
        EmbeddingServiceClient::new(ch)
            .generate_embeddings(GenerateEmbeddingsRequest {
                source_id: "patents".into(),
                model_id: tiny_bert_model_id(),
                columns: vec!["abstract".into()],
                key_column: "id".into(),
                modality: Modality::Text as i32,
                cache: jammi_wire::proto::inference::CachePolicy::Unspecified as i32,
            })
            .await
    });
    // The unary must PARK, not end: a unary that returns first names why.
    tokio::select! {
        () = park.wait_parked() => {}
        ended = &mut unary => panic!("the unary ended before parking: {ended:?}"),
    }
    let inline = served
        .session
        .catalog()
        .list_jobs()
        .await
        .unwrap()
        .into_iter()
        .find(|j| j.execution == "inline")
        .expect("the inline row exists while parked");

    served.drain();
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(
        !served.task.is_finished(),
        "the drain is blocked on the in-flight unary"
    );
    let released_at = tokio::time::Instant::now();
    served.release();
    let outcome = served
        .finish(Duration::from_secs(2 * FAST_TIMING.heartbeat + 8))
        .await
        .unwrap();
    assert_eq!(outcome, ShutdownOutcome::Released);
    assert!(released_at.elapsed() < Duration::from_secs(2 * FAST_TIMING.heartbeat + 3));
    // In-process, tonic's already-spawned connection task outlives the
    // dropped serve future (the binary's `process::exit(0)` is what ends it
    // in production): the parked unary is still alive here. Release it — its
    // run now fails against the closed catalog, so the client sees an error
    // and the row is left exactly as RELEASE left it.
    park.release();
    let result = tokio::time::timeout(Duration::from_secs(60), unary)
        .await
        .expect("the unary ends once unparked")
        .unwrap();
    assert!(
        result.is_err(),
        "the unary cannot finish against the released server's closed catalog"
    );

    let catalog = reopen(dir.path()).await;
    let row = catalog.get_job(&inline.job_id).await.unwrap();
    assert_eq!(row.status, JobStatus::Running.to_string());
    assert!(
        row.lease_expires_at.is_some(),
        "an inline row is never released"
    );
    assert_eq!(row.releases, 0);
}

/// The released server never finalizes the aborted job: after the detached
/// training thread has returned, the row is still `running`/lease NULL/
/// `claimed_by` the old instance, and the `_resume` manifest epoch is the
/// one read at release.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn released_server_never_finalizes_the_aborted_job() {
    let dir = TempDir::new().unwrap();
    let served = serve(dir.path(), FAST_TIMING).await;
    add_source(
        channel(served.flight_addr).await,
        "training",
        "training_pairs.csv",
        FileFormat::Csv,
    )
    .await;
    let job_id = submit_fine_tune(channel(served.flight_addr).await, 20_000).await;
    wait_status(
        served.session.catalog(),
        &job_id,
        &JobStatus::Running.to_string(),
        Duration::from_secs(120),
    )
    .await;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    while resume_epoch(&served.session, &job_id).await.is_none() {
        assert!(tokio::time::Instant::now() < deadline, "no bundle landed");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let instance_id = served.session.instance_id().to_string();
    let session = Arc::clone(&served.session);
    let threads_before = training_test_hooks::training_threads_finished();

    served.release();
    let outcome = served.finish(Duration::from_secs(15)).await.unwrap();
    assert_eq!(outcome, ShutdownOutcome::Released);
    let epoch_at_release = resume_epoch(&session, &job_id)
        .await
        .expect("bundle present");

    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    while training_test_hooks::training_threads_finished() <= threads_before {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the abandoned training thread never returned"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let catalog = reopen(dir.path()).await;
    let row = catalog.get_job(&job_id).await.unwrap();
    assert_eq!(row.status, JobStatus::Running.to_string(), "{row:?}");
    assert_eq!(row.claimed_by.as_deref(), Some(instance_id.as_str()));
    assert!(row.lease_expires_at.is_none());
    assert_eq!((row.attempts, row.releases), (1, 1));
    assert_eq!(
        resume_epoch(&session, &job_id).await,
        Some(epoch_at_release),
        "no bundle may land after the release"
    );
}

/// D3: the worker guard is hoisted onto `BoundServer` when `[worker]
/// enabled`, absent otherwise.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bound_server_holds_the_worker_guard_when_worker_enabled() {
    for enabled in [true, false] {
        let dir = TempDir::new().unwrap();
        let server = OssServer::new(server_config(dir.path(), DEFAULT_TIMING, enabled))
            .await
            .unwrap();
        let bound = server.bind().await.unwrap();
        assert_eq!(bound.has_worker(), enabled);
        let (drain_tx, drain_rx) = watch::channel(false);
        let (_release_tx, release_rx) = watch::channel(false);
        let task = tokio::spawn(bound.serve_with_signals(drain_rx, release_rx));
        tokio::time::sleep(Duration::from_millis(200)).await;
        let _ = drain_tx.send(true);
        let outcome = tokio::time::timeout(Duration::from_secs(30), task)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(
            outcome,
            ShutdownOutcome::Drained {
                worker_joined: enabled
            }
        );
    }
}

/// Acceptance 9 / K4: `ListWorkers` shows `state` equal to the row, and this
/// process's transitions arrive in order — `claiming` while the loop is
/// live, `draining` once a DRAIN begins with a job in flight — and the row
/// is gone after the release.
///
/// The `warming` leg is NOT here: a process is `warming` only while it has
/// models left to preload (`serve_with_signals` opens the session's worker
/// gate unconditionally the moment it has nothing to load, so a
/// hand-closed gate on an empty `preload_models` is reopened before any
/// client could look). `readiness_preload.rs` asserts `warming` — over the
/// same wire, against the same row — with a real preload parked.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn list_workers_reports_state_over_the_wire() {
    let dir = TempDir::new().unwrap();
    let served = serve(dir.path(), FAST_TIMING).await;

    async fn wire_state(
        addr: std::net::SocketAddr,
    ) -> Result<Option<String>, tonic::transport::Error> {
        let ch = tonic::transport::Endpoint::from_shared(format!("http://{addr}"))
            .unwrap()
            .connect()
            .await?;
        Ok(JobServiceClient::new(ch)
            .list_workers(ListWorkersRequest {})
            .await
            .expect("list_workers")
            .into_inner()
            .workers
            .into_iter()
            .next()
            .map(|w| w.state))
    }
    async fn row_state(session: &InferenceSession) -> Option<String> {
        session
            .catalog()
            .list_workers()
            .await
            .unwrap()
            .into_iter()
            .next()
            .map(|w| w.state)
    }
    /// `Err(transport error)` when the server stopped answering — the caller
    /// then surfaces the serve task's own outcome.
    async fn wait_state(served: &Served, want: &str) -> Result<(), tonic::transport::Error> {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        loop {
            let wire = wire_state(served.flight_addr).await?;
            let row = row_state(&served.session).await;
            if wire.as_deref() == Some(want) {
                assert_eq!(wire, row, "K4: the wire state equals the row");
                return Ok(());
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "never observed state {want}; wire {wire:?}, row {row:?}"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    if let Err(e) = wait_state(&served, "claiming").await {
        let outcome = served.finish(Duration::from_secs(30)).await;
        panic!("server stopped answering ({e}); serve task outcome: {outcome:?}");
    }

    add_source(
        channel(served.flight_addr).await,
        "training",
        "training_pairs.csv",
        FileFormat::Csv,
    )
    .await;
    let job_id = submit_fine_tune(channel(served.flight_addr).await, 20_000).await;
    wait_status(
        served.session.catalog(),
        &job_id,
        &JobStatus::Running.to_string(),
        Duration::from_secs(120),
    )
    .await;
    served.drain();
    // DRAIN closes the gRPC listener (tonic's graceful shutdown; D12), so a
    // fresh connection can never read `draining` over the wire — the row is
    // the truth another process reads (B4), and it is observed here.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        let row = row_state(&served.session).await;
        if row.as_deref() == Some("draining") {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "never observed state draining on the row; last {row:?}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    served.release();
    assert_eq!(
        served.finish(Duration::from_secs(15)).await.unwrap(),
        ShutdownOutcome::Released
    );
    assert!(
        reopen(dir.path())
            .await
            .list_workers()
            .await
            .unwrap()
            .is_empty(),
        "the released loop's row is deleted"
    );
}

// ---------------------------------------------------------------------------
// The binary: SIGTERM then `jammi-server release --pid`
// ---------------------------------------------------------------------------

const BIN: &str = env!("CARGO_BIN_EXE_jammi-server");

/// Acceptance 10: a real child `jammi-server` (fast lease) with a long job
/// running; SIGTERM (DRAIN) then `jammi-server release --pid <child>` sends
/// SIGINT; the child exits 0 within two heartbeats and its row reads
/// `running`/lease NULL/`releases 1`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn release_subcommand_sends_sigint_and_the_child_exits_zero_within_two_heartbeats() {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};

    let dir = TempDir::new().unwrap();
    let config_path = dir.path().join("jammi.toml");
    std::fs::write(
        &config_path,
        format!(
            "artifact_dir = \"{artifact_dir}\"\n\n\
             [gpu]\ndevice = -1\n\n\
             [inference]\nbatch_size = 8\n\n\
             [logging]\nlevel = \"info\"\n\n\
             [lease]\nduration_secs = {lease}\nheartbeat_secs = {heartbeat}\n\n\
             [server]\nhealth_listen = \"127.0.0.1:0\"\nflight_listen = \"127.0.0.1:0\"\n",
            artifact_dir = dir.path().display(),
            lease = FAST_TIMING.lease,
            heartbeat = FAST_TIMING.heartbeat,
        ),
    )
    .unwrap();
    let mut child = Command::new(BIN)
        .env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .env("HOME", dir.path())
        .args(["serve", "--config", config_path.to_str().unwrap()])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn jammi-server");
    let stdout = child.stdout.take().unwrap();
    let flight_addr: std::net::SocketAddr = tokio::task::spawn_blocking(move || {
        for line in BufReader::new(stdout).lines() {
            let line = line.unwrap();
            if let Some(rest) = line.strip_prefix("jammi-server listening flight=") {
                return rest.split_whitespace().next().unwrap().parse().unwrap();
            }
        }
        panic!("the child never announced its listening addresses");
    })
    .await
    .unwrap();

    add_source(
        channel(flight_addr).await,
        "training",
        "training_pairs.csv",
        FileFormat::Csv,
    )
    .await;
    let job_id = submit_fine_tune(channel(flight_addr).await, 20_000).await;
    let mut client = JobServiceClient::new(channel(flight_addr).await);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    loop {
        let status = client
            .job_status(JobStatusRequest {
                job_id: job_id.clone(),
            })
            .await
            .unwrap()
            .into_inner()
            .status;
        if status == "running" {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "never running: {status}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // DRAIN first (SIGTERM), then the actuator (SIGINT via the subcommand).
    let pid = child.id() as i32;
    // SAFETY: `kill(2)` on our own child's pid with a constant signal.
    assert_eq!(unsafe { libc::kill(pid, libc::SIGTERM) }, 0);
    tokio::time::sleep(Duration::from_millis(300)).await;
    let released_at = std::time::Instant::now();
    let status = Command::new(BIN)
        .env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .args(["release", "--pid", &pid.to_string()])
        .status()
        .expect("run jammi-server release");
    assert!(status.success(), "release must exit 0: {status}");

    let exit = tokio::task::spawn_blocking(move || child.wait())
        .await
        .unwrap()
        .expect("wait child");
    let took = released_at.elapsed();
    assert!(exit.success(), "the released server exits 0, got {exit}");
    assert!(
        took < Duration::from_secs(2 * FAST_TIMING.heartbeat + 5),
        "the child exited in {took:?}, more than two heartbeats"
    );

    let catalog = reopen(dir.path()).await;
    let row = catalog.get_job(&job_id).await.unwrap();
    assert_eq!(row.status, JobStatus::Running.to_string(), "{row:?}");
    assert!(row.lease_expires_at.is_none(), "{row:?}");
    assert_eq!((row.attempts, row.releases), (1, 1));
    assert!(catalog.list_workers().await.unwrap().is_empty());
}
