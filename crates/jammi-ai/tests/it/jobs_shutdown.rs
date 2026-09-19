//! The two shutdown modes on the LIBRARY — DRAIN
//! (`EmbeddedWorker::stop_and_join`) and RELEASE (`release_and_stop`) —
//! with every arm of the RELEASE mechanism pinned against the loop's own
//! rendezvous points (`loop_test_hooks`, `claim_test_hooks`, the compute and
//! materialization parks), never against a wall clock:
//!
//! * (a): a claim parked before COMMIT commits on unpark and self-releases;
//! * (a'): a stop during the idle sleep leaves the row `queued`;
//! * (b): the claim→hold prologue self-releases under `Releasing`;
//! * (c): the timeout arm's row is recovered by arm 1a, never `failed`;
//! * (d): `run_now` never changes `in_flight` and an inline
//!   materialization survives a RELEASE with its lease live;
//! * a compute job released mid-materialization resumes on the successor
//!   without a `BackOff`;
//! * a released training job is never finalized by the abandoned thread and
//!   its `_resume` manifest epoch never advances.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use jammi_ai::fine_tune::spec::{TrainingCommon, TrainingSpec};
use jammi_ai::fine_tune::worker::{
    loop_test_hooks, training_test_hooks, EmbeddedWorker, HoldReleaseOutcome, Holder, JobWorker,
    LoopState, ReleaseSweep, StopOutcome, WorkerPhase, WorkerShared,
};
use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod};
use jammi_ai::jobs::{compute_test_hooks, ComputeSpec, JobResult, JobSpec};
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::claim_test_hooks;
use jammi_db::catalog::lease_keeper::HoldRelease;
use jammi_db::catalog::status::JobStatus;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::mutable::test_hook::{arm as arm_materialization, MaterializationPoint};
use jammi_db::store::CachePolicy;

use crate::common;

fn tiny_bert_model() -> String {
    "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap()
}

/// Lease / heartbeat / idle-poll timing for one fixture, in seconds.
#[derive(Clone, Copy)]
struct Timing {
    lease: u64,
    heartbeat: u64,
    idle_poll: u64,
}

const DEFAULT_TIMING: Timing = Timing {
    lease: 30,
    heartbeat: 10,
    idle_poll: 1,
};

const FAST_TIMING: Timing = Timing {
    lease: 3,
    heartbeat: 1,
    idle_poll: 1,
};

/// A session over `dir` with the training and patents sources registered.
async fn session_at(dir: &std::path::Path, timing: Timing) -> Arc<InferenceSession> {
    let mut config = common::test_config(dir);
    config.lease.duration_secs = timing.lease;
    config.lease.heartbeat_secs = timing.heartbeat;
    config.worker.idle_poll_secs = timing.idle_poll;
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    for (name, file, format) in [
        ("training", "training_pairs.csv", FileFormat::Csv),
        ("patents", "patents.parquet", FileFormat::Parquet),
    ] {
        // A successor session over an existing catalog finds them registered.
        match session
            .add_source(
                name,
                SourceType::File,
                SourceConnection {
                    url: Some(common::fixture_url(file)),
                    format: Some(format),
                    ..Default::default()
                },
            )
            .await
        {
            Ok(()) => {}
            Err(e) if e.to_string().contains("already registered") => {}
            Err(e) => panic!("add_source({name}): {e}"),
        }
    }
    session
}

async fn session(timing: Timing) -> (Arc<InferenceSession>, tempfile::TempDir) {
    let dir = tempfile::TempDir::new().unwrap();
    let session = session_at(dir.path(), timing).await;
    (session, dir)
}

fn fine_tune(epochs: usize) -> JobSpec {
    TrainingSpec::FineTune {
        source: "training".to_string(),
        columns: vec![
            "text_a".to_string(),
            "text_b".to_string(),
            "score".to_string(),
        ],
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
            world_size: jammi_ai::fine_tune::spec::DEFAULT_WORLD_SIZE,
        },
        cache: jammi_db::store::CachePolicy::Bypass,
    }
    .into()
}

/// A compute kind that materializes a real table over the patents fixture.
fn embedding_spec(source: &str) -> ComputeSpec {
    ComputeSpec::Embedding {
        source_id: source.to_string(),
        model_id: tiny_bert_model(),
        columns: vec!["abstract".to_string()],
        key_column: "id".to_string(),
        modality: jammi_wire::request::Modality::Text,
        cache: CachePolicy::Bypass,
    }
}

/// A compute kind whose producer can never run (the model does not exist):
/// reaching the producer at all would itself be the failure.
fn never_dispatched_infer(source: &str) -> ComputeSpec {
    ComputeSpec::Infer {
        source_id: source.to_string(),
        model_id: "local:/nonexistent/model/for-the-shutdown-oracles".to_string(),
        task: ModelTask::TextEmbedding,
        content_columns: vec!["abstract".to_string()],
        key_column: "id".to_string(),
        cache: CachePolicy::Bypass,
    }
}

fn spawn_worker(session: &Arc<InferenceSession>) -> Arc<EmbeddedWorker> {
    Arc::new(
        EmbeddedWorker::spawn_worker(
            session,
            JobWorker::with_intervals(session, session.worker_intervals().unwrap()),
        )
        .unwrap(),
    )
}

fn shared_of(worker: &EmbeddedWorker) -> Arc<WorkerShared> {
    worker
        .shared()
        .upgrade()
        .expect("the loop's shared state lives as long as the guard")
}

/// Loop-claimed jobs running under a live hold, read off the host's slot
/// holder: `1` iff it is `JobRun` (a `ClaimProbe` — the claim round trip
/// or the claim→hold prologue — and a held gang `Rank` both read `0`).
fn in_flight(shared: &WorkerShared) -> usize {
    matches!(shared.admission().holder(), Holder::JobRun) as usize
}

async fn wait_in_flight(shared: &WorkerShared, want: usize) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    while in_flight(shared) != want {
        assert!(
            tokio::time::Instant::now() < deadline,
            "in_flight never reached {want} (now {}, holder {:?})",
            in_flight(shared),
            shared.admission().holder()
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

async fn wait_status(session: &InferenceSession, job_id: &str, want: &str, bound: Duration) {
    let deadline = tokio::time::Instant::now() + bound;
    loop {
        let row = session.catalog().get_job(job_id).await.unwrap();
        if row.status == want {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "job {job_id} never reached {want}; last {row:?}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// The epoch the job's durable `_resume` manifest currently names, or
/// `None` when no epoch boundary has landed a bundle yet.
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
            .expect("resume_state.json carries an integer last_completed_epoch"),
    )
}

/// Register the patents fixture under a per-test source name — the compute
/// park (`compute_test_hooks`) is keyed by source, so sibling tests in this
/// binary never take each other's arm.
async fn unique_patents(session: &InferenceSession) -> String {
    let name = format!("patents-{}", uuid::Uuid::new_v4().simple());
    session
        .add_source(
            &name,
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    name
}

// ---------------------------------------------------------------------------
// DRAIN
// ---------------------------------------------------------------------------

/// DRAIN mid-job: `stop_and_join` keeps the keeper alive and lets the
/// in-flight training run to its own terminal write — the job lands
/// `completed` under attempt 1, and the loop reports `Stopped`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stop_and_join_lets_the_epoch_bundle_land() {
    let (session, _dir) = session(DEFAULT_TIMING).await;
    let handle = session.enqueue(fine_tune(400), 0).await.unwrap();
    let worker = spawn_worker(&session);
    let shared = shared_of(&worker);
    wait_in_flight(&shared, 1).await;

    let outcome = tokio::time::timeout(Duration::from_secs(300), worker.stop_and_join())
        .await
        .expect("a generous backstop against a wedged or starved machine: DRAIN never returned")
        .unwrap();
    assert_eq!(outcome, StopOutcome::Joined);
    assert_eq!(shared.loop_state(), LoopState::Stopped);
    let row = session.catalog().get_job(&handle.job_id).await.unwrap();
    assert_eq!(row.status, JobStatus::Completed.to_string());
    assert_eq!(row.attempts, 1);
    assert_eq!(row.releases, 0);
    assert_eq!(row.claimed_by.as_deref(), Some(session.instance_id()));
    assert!(row.lease_expires_at.is_none());
    assert!(
        session.catalog().list_workers().await.unwrap().is_empty(),
        "the joined loop's workers row is deleted"
    );
}

/// An idle loop's sleep is `select!`ed against the stop watch: a DRAIN of an
/// idle worker returns at once, never after the full idle poll (30 s here;
/// base waits the poll out).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stop_and_join_returns_within_idle_poll_when_idle() {
    let (session, _dir) = session(Timing {
        idle_poll: 30,
        ..DEFAULT_TIMING
    })
    .await;
    let worker = spawn_worker(&session);
    // Let the loop take its first (empty) claim and enter the sleep.
    tokio::time::sleep(Duration::from_millis(500)).await;
    let outcome = tokio::time::timeout(Duration::from_secs(5), worker.stop_and_join())
        .await
        .expect("an idle worker stops within the bound, not after the 30 s poll")
        .unwrap();
    assert_eq!(outcome, StopOutcome::Joined);
    assert_eq!(shared_of(&worker).loop_state(), LoopState::Stopped);
}

// ---------------------------------------------------------------------------
// A cancelled `stop_and_join` must not detach the loop task
// ---------------------------------------------------------------------------

/// The embedded (non-server) arm. `stop_and_join` awaits the in-flight
/// job's own terminal `LoopState` for however long that job takes —
/// exactly the shape a `tokio::select!`/`tokio::time::timeout` can cancel
/// mid-await, the way `runtime.rs`'s RELEASE arm cancels a DRAIN's
/// `stop_and_join` when a second signal races it. Cancelling it here (via
/// `timeout`, the library-level equivalent of that `select!`) must NOT
/// detach the loop task: dropping the guard afterwards must still find it
/// and abort it — never a bare `None` with nothing left to do. `nothing
/// outlives session.close()`: closing the session after the abort does not
/// hang, and the row is left exactly where the abort left it (no detached
/// finalize lands afterward).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_the_guard_after_a_cancelled_stop_and_join_aborts_the_task() {
    let (session, _dir) = session(FAST_TIMING).await;
    let handle = session.enqueue(fine_tune(20_000), 0).await.unwrap();
    let worker = spawn_worker(&session);
    let shared = shared_of(&worker);
    wait_in_flight(&shared, 1).await;
    let threads_before = training_test_hooks::training_threads_finished();

    // Cancel `stop_and_join` long before a 20 000-epoch job could possibly
    // finish: its future is dropped while suspended awaiting the loop's
    // terminal state.
    let cancelled = tokio::time::timeout(Duration::from_millis(200), worker.stop_and_join()).await;
    assert!(
        cancelled.is_err(),
        "the job must still be running well past 200ms; stop_and_join must not have completed"
    );
    assert_eq!(
        shared.loop_state(),
        LoopState::Running,
        "the cancelled call alone must not touch the loop -- only Drop (or a later \
         release_and_stop) may abort it"
    );

    // The sole `Arc` on the guard — dropping it runs `EmbeddedWorker::drop`,
    // which must find `LoopTask::Abandoned` (not a lost, bare `None`) and
    // abort the task.
    assert_eq!(
        Arc::strong_count(&worker),
        1,
        "the test must hold the only strong reference for `drop` below to actually run"
    );
    drop(worker);

    let deadline = tokio::time::Instant::now() + Duration::from_secs(FAST_TIMING.heartbeat + 10);
    loop {
        if shared.loop_state() != LoopState::Running {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the loop task was never aborted -- it is still running, detached"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        shared.loop_state(),
        LoopState::Aborted,
        "Drop must ABORT a handle it reclaims from an abandoned stop attempt, not let it \
         return cooperatively"
    );

    // Wait for the abandoned training thread to actually RETURN
    // before reading the row -- the loop task's own `LoopState` (observed
    // above) says nothing about the `spawn_blocking` trainer it detached
    // from; reading the row immediately after the abort would assert "no
    // detached finalize lands" before the detached thread ever had the
    // chance to land one, which is vacuous. The same rendezvous is used at
    // `release_and_stop_leaves_running_with_null_lease_and_no_new_bundle`
    // in this file for exactly this reason.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(FAST_TIMING.heartbeat + 120);
    while training_test_hooks::training_threads_finished() <= threads_before {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the abandoned training thread never returned"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let row = session.catalog().get_job(&handle.job_id).await.unwrap();
    assert_eq!(
        row.status,
        JobStatus::Running.to_string(),
        "the abort leaves the row exactly where it was -- no detached finalize lands"
    );

    // Nothing outlives `session.close()`: closing does not hang on the
    // aborted loop's still-running `spawn_blocking` trainer.
    tokio::time::timeout(Duration::from_secs(30), session.close())
        .await
        .expect("session.close() must not hang on the aborted loop's detached trainer");
}

// ---------------------------------------------------------------------------
// RELEASE — in flight
// ---------------------------------------------------------------------------

/// RELEASE mid-epoch (the abort arm): the row is left `running` under this
/// instance with a NULL lease and `releases = 1`; the keeper flipped the
/// hold's `lost` before the abort so the detached training thread bails at
/// its next epoch boundary WITHOUT a bundle — the `_resume` manifest epoch
/// never advances past the value read at release — and never finalizes:
/// the row is still `running`/`claimed_by` this instance after the thread
/// has returned. Also the library's `release_does_not_advance_the_resume_
/// manifest_epoch` and `released_..._never_finalizes_the_aborted_job`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_and_stop_leaves_running_with_null_lease_and_no_new_bundle() {
    let (session, _dir) = session(FAST_TIMING).await;
    let handle = session.enqueue(fine_tune(20_000), 0).await.unwrap();
    // Armed BEFORE `spawn_worker` claims and starts the run, so the
    // trainer's fire (inside `save_resume_checkpoint`, the instant its
    // `put_resume_checkpoint` write lands) can never race ahead of the arm.
    let bundle_landed = loop_test_hooks::arm_observed(
        &handle.job_id,
        loop_test_hooks::Event::ResumeCheckpointWritten,
    );
    let worker = spawn_worker(&session);
    let shared = shared_of(&worker);
    wait_in_flight(&shared, 1).await;
    // At least one epoch boundary has landed a bundle, so there is an epoch
    // to compare. Waits on the REAL event -- the trainer's durable
    // `put_resume_checkpoint` write actually landing -- never on a
    // wall-clock guess at when one epoch's write might complete. A
    // generous 60s backstop against a wedged or starved machine.
    tokio::time::timeout(Duration::from_secs(60), bundle_landed.wait_fired())
        .await
        .expect(
            "a generous backstop against a wedged or starved machine: no resume bundle write \
             was ever observed",
        );
    assert!(
        resume_epoch(&session, &handle.job_id).await.is_some(),
        "the observed write must have left a readable bundle behind"
    );
    let threads_before = training_test_hooks::training_threads_finished();
    let claim_next_before_release = loop_test_hooks::claim_next_calls(session.instance_id());

    let report = tokio::time::timeout(Duration::from_secs(10), worker.release_and_stop())
        .await
        .expect("RELEASE is bounded by two heartbeats")
        .unwrap();
    assert_eq!(report.loop_state, LoopState::Aborted, "{report:?}");
    assert_eq!(
        loop_test_hooks::claim_next_calls(session.instance_id()),
        claim_next_before_release,
        "no claim_next after 2a on the in_flight>0/immediate-abort arm of 2e"
    );
    assert_eq!(
        report.holds,
        HoldReleaseOutcome::Observed(HoldRelease {
            released: 1,
            not_required: 0,
            failed: 0,
            attempted: 1,
        }),
        "{report:?}"
    );
    let epoch_at_release = resume_epoch(&session, &handle.job_id)
        .await
        .expect("the bundle is still there");
    let row = session.catalog().get_job(&handle.job_id).await.unwrap();
    assert_eq!(row.status, JobStatus::Running.to_string());
    assert_eq!(row.claimed_by.as_deref(), Some(session.instance_id()));
    assert!(row.lease_expires_at.is_none(), "{row:?}");
    assert_eq!((row.attempts, row.releases), (1, 1));
    assert!(
        session.catalog().list_workers().await.unwrap().is_empty(),
        "the released loop's workers row is deleted"
    );

    // The abandoned thread finishes (bails at its next boundary) … derived
    // from `FAST_TIMING.heartbeat`, mirroring this file's own sibling wait
    // above (`training_threads_finished` is a counter, not awaitable, so
    // this stays a derived poll rather than an event rendezvous).
    let deadline = tokio::time::Instant::now() + Duration::from_secs(FAST_TIMING.heartbeat + 120);
    while training_test_hooks::training_threads_finished() <= threads_before {
        assert!(
            tokio::time::Instant::now() < deadline,
            "derived from FAST_TIMING.heartbeat + 120: the abandoned training thread never \
             returned"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    // … and never finalized, never wrote another bundle.
    let after = session.catalog().get_job(&handle.job_id).await.unwrap();
    assert_eq!(after.status, JobStatus::Running.to_string());
    assert_eq!(after.claimed_by.as_deref(), Some(session.instance_id()));
    assert!(after.lease_expires_at.is_none());
    assert_eq!(
        resume_epoch(&session, &handle.job_id).await,
        Some(epoch_at_release),
        "no bundle may land after the lease was handed back"
    );
}

/// RELEASE during a compute materialization (job hold + `ResultTable` hold
/// registered, the writer parked inside `BuildingTable::finish`): both
/// leases are NULL afterwards (the jobs sweep and the jobs-linked building
/// sweep), and a fresh session's worker claims the job with `attempts = 2`,
/// its dispatch takes the claim-and-fail arm at once (never `BackOff` — it
/// completes well inside one lease window), clears the stale
/// `partial_result`, and runs to `completed` with its own `ready` table.
/// Base: the successor's own `create_result_table` CAS finds the stale
/// pointer and the job lands `failed` as `JobAttemptSuperseded`.
// The materialization park is one slot per point, process-wide: the tests
// that arm it are serialised against each other.
#[serial_test::serial(materialization_park)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_mid_materialization_resumes_on_the_successor_without_backoff() {
    let dir = tempfile::TempDir::new().unwrap();
    let session = session_at(dir.path(), DEFAULT_TIMING).await;
    let source = unique_patents(&session).await;
    let writer_id = session.result_store().writer_id().to_string();
    let parked = arm_materialization(MaterializationPoint::Materialization, &writer_id);
    let handle = session
        .enqueue(embedding_spec(&source).into(), 0)
        .await
        .unwrap();
    let worker = spawn_worker(&session);
    let shared = shared_of(&worker);
    parked
        .wait_parked()
        .await
        .expect("the loop's writer parks inside finish");
    assert_eq!(in_flight(&shared), 1);
    let mid = session.catalog().get_job(&handle.job_id).await.unwrap();
    let first_table = mid
        .partial_result
        .clone()
        .expect("the first attempt recorded its building table");
    let building = session
        .catalog()
        .get_result_table(&first_table)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(building.status, "building");
    assert!(building.lease_expires_at.is_some());

    let report = tokio::time::timeout(Duration::from_secs(30), worker.release_and_stop())
        .await
        .expect("RELEASE is bounded")
        .unwrap();
    assert_eq!(report.loop_state, LoopState::Aborted, "{report:?}");
    let row = session.catalog().get_job(&handle.job_id).await.unwrap();
    assert_eq!(row.status, JobStatus::Running.to_string());
    assert!(row.lease_expires_at.is_none());
    assert_eq!(row.releases, 1);
    let released = session
        .catalog()
        .get_result_table(&first_table)
        .await
        .unwrap()
        .unwrap();
    // The linked sweep NULLed the lease. The aborted future's own
    // `BuildingTable::drop` then marks the un-finished row `failed` under
    // the writer's CAS (its hold was not yet `lost` — the keeper flips that
    // at its next renewal), so the row reads `failed` (or, if the renewal
    // landed first, `building`); either way its lease is NULL and the
    // successor never backs off.
    assert!(
        matches!(released.status.as_str(), "building" | "failed"),
        "{released:?}"
    );
    assert_eq!(released.writer_id.as_deref(), Some(writer_id.as_str()));
    assert!(
        released.lease_expires_at.is_none(),
        "the linked building sweep NULLed the building lease"
    );
    parked.release();
    drop(worker);
    session.close().await;

    // The successor: a fresh session over the same catalog.
    let successor = session_at(dir.path(), DEFAULT_TIMING).await;
    let started = tokio::time::Instant::now();
    let worker2 = spawn_worker(&successor);
    wait_status(
        &successor,
        &handle.job_id,
        &JobStatus::Completed.to_string(),
        Duration::from_secs(25),
    )
    .await;
    assert!(
        started.elapsed() < Duration::from_secs(DEFAULT_TIMING.lease),
        "a BackOff would have waited out a full lease window"
    );
    let done = successor.catalog().get_job(&handle.job_id).await.unwrap();
    assert_eq!(done.attempts, 2);
    assert_eq!(done.claimed_by.as_deref(), Some(successor.instance_id()));
    let second_table = done.partial_result.clone().expect("the successor's table");
    assert_ne!(second_table, first_table);
    let ready = successor
        .catalog()
        .get_result_table(&second_table)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(ready.status, "ready");
    let result: JobResult = serde_json::from_str(done.result.as_deref().unwrap()).unwrap();
    assert!(matches!(result, JobResult::Table { table, .. } if table == second_table));
    worker2.stop_and_join().await.unwrap();
    successor.close().await;
}

// ---------------------------------------------------------------------------
// RELEASE — the four claim-window arms
// ---------------------------------------------------------------------------

/// (a): the loop is parked INSIDE `claim_next`, before COMMIT, with
/// `in_flight == 0`. RELEASE must not abort: it waits, and the claim —
/// unparked exactly when RELEASE reaches 2e (the `ReleaseAt2e` rendezvous,
/// never a wall clock: an unpark after the SQLite `database is locked` log
/// would deterministically take the timeout arm) — commits, runs into the
/// hold helper under `Releasing`, self-releases and never dispatches. The
/// row ends `running`/`claimed_by` me/lease NULL/`releases 1`/`attempts 1`;
/// the loop reports `Stopped`. On SQLite the parked transaction holds the
/// write lock, so the sweeps' statements log `database is locked` and are
/// tolerated (`sweep_one` reads `None`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_with_the_loop_paused_inside_claim_next_does_not_abort() {
    let (session, _dir) = session(DEFAULT_TIMING).await;
    let source = unique_patents(&session).await;
    let claim_park = claim_test_hooks::arm(
        session.instance_id(),
        claim_test_hooks::ParkPoint::ClaimBeforeCommit,
    );
    let dispatch_park =
        compute_test_hooks::arm(&source, compute_test_hooks::ParkPoint::BeforeDispatch);
    let handle = session
        .enqueue(never_dispatched_infer(&source).into(), 0)
        .await
        .unwrap();
    let worker = spawn_worker(&session);
    let shared = shared_of(&worker);
    claim_park.wait_parked().await;
    assert_eq!(in_flight(&shared), 0);
    // The claim already parked here is the ONE genuine race the mechanism
    // allows (its COMMIT precedes 2a); the counter must never grow past this
    // point -- release_and_stop has not even been called yet.
    let claim_next_before_release = loop_test_hooks::claim_next_calls(session.instance_id());

    let at_2e = loop_test_hooks::arm_rendezvous(
        session.instance_id(),
        loop_test_hooks::Rendezvous::ReleaseAt2e,
    );
    let releasing = {
        let worker = Arc::clone(&worker);
        tokio::spawn(async move { worker.release_and_stop().await })
    };
    at_2e.wait_fired().await;
    claim_park.release();
    let report = tokio::time::timeout(Duration::from_secs(60), releasing)
        .await
        .expect("RELEASE completes once the claim commits")
        .unwrap()
        .unwrap();
    assert_eq!(report.loop_state, LoopState::Stopped, "{report:?}");
    assert_eq!(report.sweep_two.jobs, Some(0), "{report:?}");
    assert_eq!(
        loop_test_hooks::claim_next_calls(session.instance_id()),
        claim_next_before_release,
        "no claim_next after 2a on the Running/cooperative-in-claim_next arm of 2e"
    );
    let row = session.catalog().get_job(&handle.job_id).await.unwrap();
    assert_eq!(row.status, JobStatus::Running.to_string());
    assert_eq!(row.claimed_by.as_deref(), Some(session.instance_id()));
    assert!(row.lease_expires_at.is_none(), "{row:?}");
    assert_eq!((row.attempts, row.releases), (1, 1));
    assert!(
        tokio::time::timeout(Duration::from_millis(200), dispatch_park.wait_parked())
            .await
            .is_err(),
        "the self-released claim must never reach the producer"
    );
}

/// (a'): the stop lands while the loop is in its idle sleep; the job is
/// submitted after the sleep began. The loop exits at its pre-claim check
/// (the sleep is interruptible) and the row is untouched: `queued`,
/// `attempts 0`, `releases 0`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_before_the_loop_reaches_claim_next_leaves_the_row_untouched() {
    let (session, _dir) = session(Timing {
        idle_poll: 5,
        ..DEFAULT_TIMING
    })
    .await;
    let worker = spawn_worker(&session);
    tokio::time::sleep(Duration::from_millis(500)).await;
    let handle = session.enqueue(fine_tune(20_000), 0).await.unwrap();
    let started = tokio::time::Instant::now();
    let report = worker.release_and_stop().await.unwrap();
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "the idle sleep must be interrupted, not waited out"
    );
    assert_eq!(report.loop_state, LoopState::Stopped, "{report:?}");
    let row = session.catalog().get_job(&handle.job_id).await.unwrap();
    assert_eq!(row.status, JobStatus::Queued.to_string());
    assert_eq!((row.attempts, row.releases), (0, 0));
    assert!(row.claimed_by.is_none());
}

/// (b): the loop is parked in the claim→hold prologue (after COMMIT,
/// before the hold). RELEASE waits (no abort), the prologue is unparked at
/// 2e, and the hold helper self-releases under `Releasing`: the row ends
/// `running`/lease NULL/`releases 1`, nothing dispatched (no training
/// thread, no `_resume` prefix). Both kinds.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_with_the_loop_paused_in_the_claim_to_hold_prologue_self_releases() {
    for kind in ["fine_tune", "compute"] {
        let (session, _dir) = session(DEFAULT_TIMING).await;
        let source = unique_patents(&session).await;
        let spec = if kind == "fine_tune" {
            fine_tune(20_000)
        } else {
            embedding_spec(&source).into()
        };
        let handle = session.enqueue(spec, 0).await.unwrap();
        let park = loop_test_hooks::arm(&handle.job_id, loop_test_hooks::ParkPoint::BeforeHold);
        let dispatch_park =
            compute_test_hooks::arm(&source, compute_test_hooks::ParkPoint::BeforeDispatch);
        let worker = spawn_worker(&session);
        let shared = shared_of(&worker);
        park.wait_parked().await;
        assert_eq!(
            in_flight(&shared),
            0,
            "{kind}: the hold is not registered yet"
        );
        // The claim already committed and parked here predates 2a by
        // construction; the counter must not grow past this point.
        let claim_next_before_release = loop_test_hooks::claim_next_calls(session.instance_id());

        let at_2e = loop_test_hooks::arm_rendezvous(
            session.instance_id(),
            loop_test_hooks::Rendezvous::ReleaseAt2e,
        );
        let releasing = {
            let worker = Arc::clone(&worker);
            tokio::spawn(async move { worker.release_and_stop().await })
        };
        at_2e.wait_fired().await;
        park.release();
        let report = tokio::time::timeout(Duration::from_secs(30), releasing)
            .await
            .expect("RELEASE completes once the prologue self-releases")
            .unwrap()
            .unwrap();
        assert_eq!(report.loop_state, LoopState::Stopped, "{kind}: {report:?}");
        assert_eq!(
            loop_test_hooks::claim_next_calls(session.instance_id()),
            claim_next_before_release,
            "{kind}: no claim_next after 2a on the Running/claim-to-hold-prologue arm of 2e"
        );
        let row = session.catalog().get_job(&handle.job_id).await.unwrap();
        assert_eq!(row.status, JobStatus::Running.to_string(), "{kind}");
        assert!(row.lease_expires_at.is_none(), "{kind}: {row:?}");
        assert_eq!((row.attempts, row.releases), (1, 1), "{kind}");
        // Nothing dispatched: a training thread would have retired the
        // submission-time `pending` acceleration marker at its probe, and
        // an epoch boundary would have landed a `_resume` bundle.
        assert!(
            row.acceleration_report
                .as_deref()
                .is_some_and(|r| r.contains("pending")),
            "{kind}: no training thread ran, got {:?}",
            row.acceleration_report
        );
        assert!(
            resume_epoch(&session, &handle.job_id).await.is_none(),
            "{kind}: no _resume prefix"
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(200), dispatch_park.wait_parked())
                .await
                .is_err(),
            "{kind}: no producer ran"
        );
        session.close().await;
    }
}

// ---------------------------------------------------------------------------
// The release barrier's blast radius
// equals the release's, not the releasing entry point's own worker.
// ---------------------------------------------------------------------------

/// The loop's OWN `EmbeddedWorker` never calls `release_and_stop` at
/// all here — `InferenceSession::release_job_leases` (the "no loop to stop"
/// sibling, the shape `runtime.rs`/`database.rs` use on
/// their `worker.is_none()` arm) is called directly while a REAL, live loop
/// on the SAME session is parked in the claim→hold prologue
/// (`register_job_hold_or_release`, before the hold is registered, phase
/// still `Running`). `release_job_leases` flips the session-owned
/// `HostAdmission` phase to `Releasing` (2a-equivalent) BEFORE its 2b/2c —
/// no `WorkerShared::stop` is ever touched, since `release_job_leases` does
/// not know the worker exists — so this proves the barrier the prologue
/// observes is the SHARED phase cell, not anything paired with a stop
/// signal on the specific `WorkerShared` that owns the parked loop: the
/// blast radius of a release equals the release's, regardless of which of
/// the two entry points (`EmbeddedWorker::release_and_stop` or
/// `InferenceSession::release_job_leases`) initiated it.
///
/// At the park, the claim has already committed (`execution = 'queued'`,
/// lease live) but the hold is not yet on the keeper, so 2b's per-hold pass
/// finds nothing of this job's (`attempted: 0`) and 2c's sweep is the
/// actual releaser (`sweep.jobs == Some(1)`). Once the park is released,
/// the prologue registers a (now-doomed) hold, reads the shared phase as
/// `Releasing`, and self-releases: the row is left `running`/lease
/// NULL/`releases 1`, and the producer never runs.
///
/// Mutation (executed): commenting out the
/// `if shared.phase() == WorkerPhase::Releasing` arm in
/// `register_job_hold_or_release` (`crates/jammi-ai/src/fine_tune/
/// worker.rs`) so the prologue always dispatches reds this test — the
/// `dispatch_park` wait that must time out instead resolves inside the
/// 200 ms bound, failing `"the self-released claim must never reach the
/// producer"`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_job_leases_reaches_a_live_foreign_loops_prologue_and_self_releases() {
    let (session, _dir) = session(DEFAULT_TIMING).await;
    let source = unique_patents(&session).await;
    let handle = session
        .enqueue(never_dispatched_infer(&source).into(), 0)
        .await
        .unwrap();
    let park = loop_test_hooks::arm(&handle.job_id, loop_test_hooks::ParkPoint::BeforeHold);
    let dispatch_park =
        compute_test_hooks::arm(&source, compute_test_hooks::ParkPoint::BeforeDispatch);
    let worker = spawn_worker(&session);
    let shared = shared_of(&worker);
    park.wait_parked().await;
    assert_eq!(in_flight(&shared), 0, "the hold is not registered yet");
    assert!(
        !shared.stop_requested(),
        "release_job_leases must never touch this worker's own WorkerShared::stop"
    );

    let (holds, sweep) =
        tokio::time::timeout(Duration::from_secs(10), session.release_job_leases())
            .await
            .expect("release_job_leases does not wait on any loop")
            .unwrap();
    assert_eq!(
        holds,
        HoldReleaseOutcome::Observed(HoldRelease {
            released: 0,
            not_required: 0,
            failed: 0,
            attempted: 0,
        }),
        "the keeper never held this job's lease at the time of the park: {holds:?}"
    );
    assert_eq!(
        sweep,
        ReleaseSweep {
            jobs: Some(1),
            building: Some(0),
        },
        "the sweep is the actual releaser: {sweep:?}"
    );
    assert_eq!(shared.admission().phase(), WorkerPhase::Releasing);

    park.release();
    assert!(
        tokio::time::timeout(Duration::from_millis(200), dispatch_park.wait_parked())
            .await
            .is_err(),
        "the self-released claim must never reach the producer"
    );
    let row = session.catalog().get_job(&handle.job_id).await.unwrap();
    assert_eq!(row.status, JobStatus::Running.to_string());
    assert_eq!(row.claimed_by.as_deref(), Some(session.instance_id()));
    assert!(row.lease_expires_at.is_none(), "{row:?}");
    assert_eq!((row.attempts, row.releases), (1, 1));

    worker.stop_and_join().await.unwrap();
    session.close().await;
}

/// The session's single claim-loop slot is STRUCTURAL — a second
/// `EmbeddedWorker::spawn_worker` on a session that already has one live is
/// refused with a typed [`JammiError::FineTune`], never a second claim
/// loop. The first worker's row/loop are entirely untouched by the refused
/// attempt. The slot is released by the FIRST guard's own `Drop`, not by
/// `stop_and_join` (which takes `&self` and leaves the guard itself alive)
/// — `stop_and_join` alone still refuses a successor spawn; only dropping
/// the guard admits one, on the SAME session.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_second_spawn_on_the_same_session_is_refused_structurally() {
    let (session, _dir) = session(DEFAULT_TIMING).await;
    let worker_a = spawn_worker(&session);
    let shared_a = shared_of(&worker_a);
    // The loop's own first statement upserts its `workers` row
    // asynchronously; wait for it so the row-count assertion below is not
    // racing that first write.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    while session.catalog().list_workers().await.unwrap().is_empty() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the first worker's `workers` row never appeared"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let err = match EmbeddedWorker::spawn_worker(
        &session,
        JobWorker::with_intervals(&session, session.worker_intervals().unwrap()),
    ) {
        Ok(_) => panic!("a second spawn on a session with a live loop must be refused"),
        Err(e) => e,
    };
    assert!(
        matches!(err, jammi_db::error::JammiError::FineTune(_)),
        "{err:?}"
    );
    // The refused attempt never touched the live worker's own state.
    assert_eq!(shared_a.loop_state(), LoopState::Running);
    assert_eq!(
        session.catalog().list_workers().await.unwrap().len(),
        1,
        "the refused spawn upserted no second `workers` row"
    );

    worker_a.stop_and_join().await.unwrap();
    assert_eq!(shared_a.loop_state(), LoopState::Stopped);
    // The loop task is joined, but the GUARD itself is still alive — the
    // slot is still held, so a spawn attempt here is still refused.
    assert!(
        matches!(
            EmbeddedWorker::spawn_worker(
                &session,
                JobWorker::with_intervals(&session, session.worker_intervals().unwrap()),
            ),
            Err(jammi_db::error::JammiError::FineTune(_))
        ),
        "stop_and_join alone (guard still alive) must not free the slot"
    );

    // Dropping the guard is the ONE release point: a successor spawn on the
    // SAME session, after the first guard is gone, is admitted — the slot
    // is a compare-and-set, not a once-per-session latch.
    drop(worker_a);
    let worker_b = spawn_worker(&session);
    worker_b.stop_and_join().await.unwrap();
    session.close().await;
}

/// The slot is held until `release_and_stop`
/// COMPLETES, not until the guard is dropped — a successor spawn is
/// admitted the instant `release_and_stop` returns, with the FIRST guard's
/// value still alive (not yet dropped). The successor belongs to a NEW
/// generation and is NOT stopped by the old RELEASE: its own `admits_claim`
/// reads `Running` (its epoch snapshot is taken AFTER the bump), so it
/// actually claims and runs a fresh job to completion — proving the reset
/// is not merely a phase flip nobody re-checks. The first guard's own later
/// `Drop` is a no-op (compare-and-set against a generation the successor
/// has already superseded) — it does not steal the successor's slot, and
/// the successor's row/loop are untouched afterward.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_and_stop_completing_frees_the_slot_for_a_successor_not_stopped_by_the_old_release()
{
    let (session, _dir) = session(DEFAULT_TIMING).await;
    let worker_a = spawn_worker(&session);
    let shared_a = shared_of(&worker_a);
    let report = worker_a.release_and_stop().await.unwrap();
    assert_eq!(report.loop_state, LoopState::Stopped, "{report:?}");

    // The FIRST guard (`worker_a`) is still alive here — never dropped —
    // yet a successor spawn is admitted.
    let worker_b = spawn_worker(&session);
    let shared_b = shared_of(&worker_b);
    assert_ne!(
        shared_a.instance_id(),
        "",
        "sanity: the old guard's shared state is still reachable"
    );

    // The successor is NOT stopped by the OLD release: it claims and runs a
    // fresh job to completion, exactly like a freshly-spawned worker on a
    // never-released session would.
    let handle = session.enqueue(fine_tune(1), 0).await.unwrap();
    wait_status(
        &session,
        &handle.job_id,
        &JobStatus::Completed.to_string(),
        Duration::from_secs(60),
    )
    .await;
    let row = session.catalog().get_job(&handle.job_id).await.unwrap();
    assert_eq!(row.claimed_by.as_deref(), Some(session.instance_id()));
    assert_eq!(row.attempts, 1, "{row:?}");
    assert_eq!(
        row.releases, 0,
        "the successor's own claim was never released: {row:?}"
    );

    // The first guard's `Drop` — its OWN release already ran inside
    // `release_and_stop`, so this must be a no-op: the successor's slot
    // (and its still-`Running` loop) is untouched.
    drop(worker_a);
    assert_eq!(
        shared_b.loop_state(),
        LoopState::Running,
        "the first guard's belated Drop must not touch the successor's loop"
    );
    // The slot is still HELD (by the successor's generation) — the first
    // guard's belated Drop must not have freed it: a third spawn attempt is
    // still refused. This is the compare-and-set half of the property: an
    // unconditional release in `Drop` would free the successor's slot out
    // from under it right here.
    assert!(
        matches!(
            EmbeddedWorker::spawn_worker(
                &session,
                JobWorker::with_intervals(&session, session.worker_intervals().unwrap()),
            ),
            Err(jammi_db::error::JammiError::FineTune(_))
        ),
        "the first guard's belated Drop must not free the successor's slot"
    );

    worker_b.stop_and_join().await.unwrap();
    session.close().await;
}

/// Safety: the loop's top-of-iteration gate refuses a new `claim_next` on the phase
/// read ALONE (`phase() != Running`), independent of `stop_requested()`.
/// Gate-direct: `WorkerShared::set_phase_for_test` (test-hooks only)
/// constructs the phase-flipped-without-a-stop shape no real
/// `release_and_stop`/`begin_drain` call can ever produce (both pair the
/// two in the same statement) — the ONLY way to prove the gate's phase arm
/// does the work on its own, deterministically, with no timing dependency.
/// Two queued rows are used (never one) so the property's widened quantifier
/// — no row's `attempts`/`releases`/`status`/`claimed_by` changes after the
/// phase flips — is witnessed over more than the single row a self-release
/// would otherwise leave touched.
///
/// A gate that read only `stop_requested()` (which this test never sets)
/// fails this as a LIVENESS failure, not a value mismatch: the loop
/// free-spins claiming and self-releasing forever and the bounded wait
/// below times out.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_gate_refuses_every_claim_when_phase_flips_without_a_stop() {
    let (session, _dir) = session(DEFAULT_TIMING).await;
    let handle_a = session.enqueue(fine_tune(1), 0).await.unwrap();
    let handle_b = session.enqueue(fine_tune(1), 0).await.unwrap();

    let job_worker = JobWorker::with_intervals(&session, session.worker_intervals().unwrap());
    // A fresh session's `HostAdmission` has seen no `begin_release` call
    // yet, so its release epoch is 0 — the gate-direct construction below
    // deliberately bypasses `try_claim_loop` entirely (no generation, no
    // slot), so there is no live epoch to read; this test is about the
    // PHASE read alone, never the epoch.
    let shared = WorkerShared::new(
        Arc::clone(session.host_admission()),
        session.instance_id().to_string(),
        0,
    );
    // The gate-direct construction: phase flipped, stop deliberately left
    // unset, BEFORE the loop ever takes its first iteration -- no race with
    // the loop's own startup sequence is possible, since the phase is fixed
    // before `run_until` is even called.
    shared.set_phase_for_test(WorkerPhase::Releasing);
    assert!(!shared.stop_requested());

    let claim_next_before = loop_test_hooks::claim_next_calls(session.instance_id());
    let mut state_rx = shared.state_receiver();
    let loop_task = {
        let shared = Arc::clone(&shared);
        tokio::spawn(async move { job_worker.run_until(shared).await })
    };

    let exited = tokio::time::timeout(
        Duration::from_secs(10),
        state_rx.wait_for(|s| *s != LoopState::Running),
    )
    .await;
    assert!(
        exited.is_ok(),
        "the loop must exit on the phase read alone -- stop is never requested in this test"
    );
    assert_eq!(
        shared.loop_state(),
        LoopState::Stopped,
        "{:?}",
        shared.loop_state()
    );
    loop_task.await.unwrap();

    assert_eq!(
        loop_test_hooks::claim_next_calls(session.instance_id()) - claim_next_before,
        0,
        "claim_next must never be called while phase != Running"
    );
    for handle in [&handle_a, &handle_b] {
        let row = session.catalog().get_job(&handle.job_id).await.unwrap();
        assert_eq!(row.status, JobStatus::Queued.to_string(), "{row:?}");
        assert_eq!((row.attempts, row.releases), (0, 0), "{row:?}");
        assert!(row.claimed_by.is_none(), "{row:?}");
    }
}

/// Wakeup/latency companion of the test above: every phase setter requests the stop in the SAME
/// statement pair as the phase flip — `begin_drain` (`:2417-2424`) and
/// `release_and_stop`'s 2a (`:2585`) — so exit latency after a flip is
/// bounded by the in-flight job, never by `idle_poll`. A check over the
/// two setters that exist, PLUS one other oracle that fails on
/// `release_and_stop`'s unpaired shape: reverting the 2a stop alone
/// (keeping the phase flip) leaves an idle-sleeping loop unwoken, so
/// `release_before_the_loop_reaches_claim_next_leaves_the_row_untouched`
/// (`:660`) waits out the idle poll instead of being interrupted and its
/// `elapsed() < 3 s` bound fails. `begin_drain`'s OWN pairing has no such
/// witness in this suite — `stop_and_join`, the only DRAIN entry point under
/// test, requests its own stop directly (`:2472-2473`) and never calls
/// `begin_drain`, so reverting `begin_drain`'s stop alone stays unobservable
/// from outside without the SAME gate-direct construction the test above uses
/// (measured: `stop_and_join_returns_within_idle_poll_when_idle`, `:271`,
/// stays green under that mutation).
///
/// Each setter's future is polled EXACTLY ONCE, by hand, off the runtime
/// with a no-op waker — the same technique
/// `a_release_racing_an_in_flight_drain_reads_stop_unwitnessed` uses above —
/// so the setter's synchronous prefix is proven to have run to completion
/// (phase flip AND stop request, both) before its first yield point, with
/// no timing assumption: everything before an `async fn`'s first `.await`
/// runs inside its very first `poll` call, whatever that call returns.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_phase_setter_pairs_the_stop_in_the_same_statement_group() {
    let (session, _dir) = session(DEFAULT_TIMING).await;
    let waker = std::task::Waker::noop();

    // `begin_drain`: `compare_exchange` then `request_stop`, both before its
    // first yield (the `set_worker_state` DB call).
    {
        let worker = spawn_worker(&session);
        let shared = shared_of(&worker);
        let mut drain_fut = std::pin::pin!(worker.begin_drain());
        let mut cx = std::task::Context::from_waker(waker);
        match drain_fut.as_mut().poll(&mut cx) {
            std::task::Poll::Pending => {
                assert_eq!(
                    shared.phase(),
                    WorkerPhase::Draining,
                    "{:?}",
                    shared.phase()
                );
                assert!(
                    shared.stop_requested(),
                    "begin_drain must pair the stop with the phase flip, in the same poll"
                );
                drain_fut.await;
            }
            std::task::Poll::Ready(()) => {
                assert_eq!(
                    shared.phase(),
                    WorkerPhase::Draining,
                    "{:?}",
                    shared.phase()
                );
                assert!(
                    shared.stop_requested(),
                    "begin_drain must pair the stop with the phase flip, in the same poll"
                );
            }
        }
        worker.release_and_stop().await.unwrap();
    }

    // `release_and_stop`'s 2a: `set_phase(Releasing)` then `request_stop`,
    // both before its first genuine yield (2b's keeper round trip).
    {
        let worker = spawn_worker(&session);
        let shared = shared_of(&worker);
        let mut release_fut = std::pin::pin!(worker.release_and_stop());
        let mut cx = std::task::Context::from_waker(waker);
        let report = match release_fut.as_mut().poll(&mut cx) {
            std::task::Poll::Pending => {
                assert_eq!(
                    shared.phase(),
                    WorkerPhase::Releasing,
                    "{:?}",
                    shared.phase()
                );
                assert!(
                    shared.stop_requested(),
                    "2a must pair the stop with the phase flip, in the same poll"
                );
                release_fut.await
            }
            std::task::Poll::Ready(r) => {
                assert_eq!(
                    shared.phase(),
                    WorkerPhase::Releasing,
                    "{:?}",
                    shared.phase()
                );
                assert!(
                    shared.stop_requested(),
                    "2a must pair the stop with the phase flip, in the same poll"
                );
                r
            }
        }
        .unwrap();
        assert_eq!(report.loop_state, LoopState::Stopped, "{report:?}");
    }
}

/// The loop's top-of-iteration gate is read ONCE, then
/// `reclaim_expired_jobs` is awaited, then `claim_next` runs, so a RELEASE
/// that lands during that reclaim round trip still finds the now-stale
/// top-of-iteration read `Running` and would initiate one claim if nothing
/// re-read the gate afterward. `WorkerShared::admits_claim` is re-read, with no `.await`
/// between that second read and `claim_next` itself, immediately after
/// `reclaim_expired_jobs` returns — this test parks the loop at exactly
/// that instant (`loop_test_hooks::arm_after_reclaim`, keyed by
/// `instance_id` since no job is claimed yet), runs a real RELEASE (2a's
/// phase-and-stop pair) to completion while the loop is parked there, then
/// releases the park.
///
/// Without the second read, the parked iteration, once released, falls
/// straight through to `claim_next` with no gate check in between —
/// `claim_next_calls` grows by one and the row is claimed-and-self-released
/// by `register_job_hold_or_release`'s `Releasing` arm (`(attempts,
/// releases) == (1, 1)`, `running`) — the SAME numeric signature as the
/// genuine, tolerated race, but reached through the reclaim window, not
/// through a claim whose COMMIT preceded 2a.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_landing_during_the_reclaim_window_is_caught_by_the_second_gate_read() {
    let (session, _dir) = session(DEFAULT_TIMING).await;
    let handle = session.enqueue(fine_tune(1), 0).await.unwrap();

    let park = loop_test_hooks::arm_after_reclaim(session.instance_id());
    let worker = spawn_worker(&session);
    park.wait_parked().await;
    let claim_next_before_release = loop_test_hooks::claim_next_calls(session.instance_id());

    let at_2e = loop_test_hooks::arm_rendezvous(
        session.instance_id(),
        loop_test_hooks::Rendezvous::ReleaseAt2e,
    );
    let releasing = {
        let worker = Arc::clone(&worker);
        tokio::spawn(async move { worker.release_and_stop().await })
    };
    at_2e.wait_fired().await;
    park.release();
    let report = tokio::time::timeout(Duration::from_secs(30), releasing)
        .await
        .expect("RELEASE completes once the parked iteration resolves")
        .unwrap()
        .unwrap();
    assert_eq!(report.loop_state, LoopState::Stopped, "{report:?}");
    assert_eq!(
        loop_test_hooks::claim_next_calls(session.instance_id()),
        claim_next_before_release,
        "no claim_next may be initiated once phase/stop flip during the reclaim window"
    );
    let row = session.catalog().get_job(&handle.job_id).await.unwrap();
    assert_eq!(row.status, JobStatus::Queued.to_string(), "{row:?}");
    assert_eq!((row.attempts, row.releases), (0, 0), "{row:?}");
    assert!(row.claimed_by.is_none(), "{row:?}");
}

/// (c): the prologue park is held PAST the heartbeat bound, so 2e's wait
/// times out and RELEASE takes its abort arm — the ONLY path where the loop
/// is aborted with `in_flight == 0`. The loop reports `Aborted`; no hold was
/// ever registered (`holds == Observed(HoldRelease::default())`); the row the abort left is the
/// honest one: the claim had COMMITTED before the park, so sweep #1 (2c,
/// which precedes 2e) already handed its lease back — `running` under this
/// instance, lease NULL, `releases 1`, `attempts 1` — and sweep #2 matched
/// 0 rows. Arm 1a recovers it on the very next reclaim (no lease window:
/// the lease is NULL) and a successor claims with `attempts 2`, never
/// `failed`, cap net 0.
///
/// The design's outcome (iii) row — the same abort with a LIVE lease and
/// `releases` unchanged — is not reachable through a park: it needs the
/// claim's COMMIT to land after sweep #2 (a backend-side race between a
/// flushed COMMIT and the dropped future), because any claim committed
/// before 2c is released by 2c. What this oracle pins is the arm's
/// invariant — the timeout abort never fails the job and never costs it an
/// attempt.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_timeout_arm_leaves_the_honest_row_recovered_by_arm_1a() {
    let (session, _dir) = session(FAST_TIMING).await;
    let handle = session.enqueue(fine_tune(20_000), 0).await.unwrap();
    let park = loop_test_hooks::arm(&handle.job_id, loop_test_hooks::ParkPoint::BeforeHold);
    let worker = spawn_worker(&session);
    park.wait_parked().await;
    let claim_next_before_release = loop_test_hooks::claim_next_calls(session.instance_id());
    // Positive control for the counter itself: the loop has already claimed
    // this row and parked in the claim→hold prologue, so `claim_next` must
    // already have been recorded at least once here — a dead counter (never
    // incremented) would make the reclaim-window oracle's `delta == 0`
    // assertion pass vacuously.
    assert!(
        claim_next_before_release >= 1,
        "the claim_next counter must be live before RELEASE runs: {claim_next_before_release}"
    );

    let started = tokio::time::Instant::now();
    let report = worker.release_and_stop().await.unwrap();
    assert!(
        started.elapsed() >= Duration::from_secs(FAST_TIMING.heartbeat),
        "the cooperative wait runs to its heartbeat bound first"
    );
    assert_eq!(report.loop_state, LoopState::Aborted, "{report:?}");
    assert_eq!(
        loop_test_hooks::claim_next_calls(session.instance_id()),
        claim_next_before_release,
        "no claim_next after 2a on the Running/timeout-abort arm of 2e"
    );
    assert_eq!(
        report.holds,
        HoldReleaseOutcome::Observed(HoldRelease::default()),
        "no hold was ever registered: {report:?}"
    );
    assert_eq!(report.sweep_one.jobs, Some(1), "{report:?}");
    assert_eq!(report.sweep_two.jobs, Some(0), "{report:?}");
    let row = session.catalog().get_job(&handle.job_id).await.unwrap();
    assert_eq!(row.status, JobStatus::Running.to_string());
    assert_eq!(row.claimed_by.as_deref(), Some(session.instance_id()));
    assert!(row.lease_expires_at.is_none(), "{row:?}");
    assert_eq!((row.attempts, row.releases), (1, 1));
    assert!(
        row.acceleration_report
            .as_deref()
            .is_some_and(|r| r.contains("pending")),
        "nothing dispatched: {:?}",
        row.acceleration_report
    );

    // Arm 1a recovers it at once.
    let lease = Duration::from_secs(FAST_TIMING.lease);
    assert_eq!(
        session
            .catalog()
            .reclaim_expired_jobs(lease, 3)
            .await
            .unwrap(),
        1
    );
    let requeued = session.catalog().get_job(&handle.job_id).await.unwrap();
    assert_eq!(requeued.status, JobStatus::Queued.to_string());
    let successor = session
        .catalog()
        .claim_next("successor", &["fine_tune"], lease)
        .await
        .unwrap()
        .expect("claimable by a successor");
    assert_eq!(successor.job_id, handle.job_id);
    assert_eq!((successor.attempts, successor.releases), (2, 1));
    assert_ne!(successor.status, JobStatus::Failed.to_string());
}

/// (d): an inline `run_now` (parked before dispatch) is not the loop's:
/// `in_flight` reads 0 throughout a concurrent RELEASE and the inline row
/// keeps its live lease and `releases 0`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_now_under_release_and_stop_does_not_change_in_flight() {
    let (session, _dir) = session(DEFAULT_TIMING).await;
    let source = unique_patents(&session).await;
    let worker = spawn_worker(&session);
    let shared = shared_of(&worker);
    let park = compute_test_hooks::arm(&source, compute_test_hooks::ParkPoint::BeforeDispatch);
    let runner = Arc::clone(&session);
    let spec = never_dispatched_infer(&source);
    let run = tokio::spawn(async move { runner.run_now(spec).await });
    park.wait_parked().await;
    assert_eq!(in_flight(&shared), 0, "an inline claim never counts");
    let inline = session
        .catalog()
        .list_jobs()
        .await
        .unwrap()
        .into_iter()
        .find(|j| j.execution == "inline")
        .expect("the inline row exists while parked");
    assert!(inline.lease_expires_at.is_some());

    let report = worker.release_and_stop().await.unwrap();
    assert_eq!(report.loop_state, LoopState::Stopped, "{report:?}");
    assert_eq!(
        report.holds,
        HoldReleaseOutcome::Observed(HoldRelease {
            released: 0,
            not_required: 1,
            failed: 0,
            attempted: 1,
        }),
        "the inline hold is registered but never released: {report:?}"
    );
    assert_eq!(in_flight(&shared), 0);
    let after = session.catalog().get_job(&inline.job_id).await.unwrap();
    assert_eq!(after.status, JobStatus::Running.to_string());
    assert!(
        after.lease_expires_at.is_some(),
        "the inline lease stays live"
    );
    assert_eq!(after.releases, 0);
    park.release();
    let _ = run.await.unwrap();
    assert_eq!(
        session
            .catalog()
            .get_job(&inline.job_id)
            .await
            .unwrap()
            .releases,
        0
    );
}

// ---------------------------------------------------------------------------
// The pair that actually differs
// ---------------------------------------------------------------------------

/// Comparing two calls that funnel into the SAME function
/// (`release_and_stop` on both arms) has zero true-positive capacity. The
/// library's claim is STATEMENT identity
/// between `EmbeddedWorker::release_and_stop` (2a-2c, 2e-2h -- 2d folded into 2a) and
/// `InferenceSession::release_job_leases` (2b+2c) -- the pair that
/// genuinely differs, since one drives a real loop task through 2a/2d-2h
/// around the shared 2b/2c core and the other is bare 2b+2c with no loop
/// at all. Pinned here on their REPORTS, never on row equality (no row
/// shape is common to both scenarios -- a loop-claimed `queued` job and a
/// worker-less inline job are not the same row to begin with, which is
/// exactly what a row-equality oracle got wrong).
///
/// Worker-owning arm: `release_and_stop`'s own `ReleaseReport` shows 2b
/// (the keeper's per-hold release) as the ACTUAL releaser -- `holds ==
/// Observed(HoldRelease { released: 1, .. })` -- so by the time 2c's sweep
/// (`sweep_two`, the final, unconditional one) runs, the row's lease is
/// already NULL and the sweep itself matches zero ADDITIONAL rows:
/// `sweep_two.jobs == Some(0)`. This is the same 2b-before-2c ordering
/// `release_job_leases` uses.
///
/// Worker-less arm: an inline `run_now` job's hold IS registered (`run_now`
/// holds every claim through the keeper) but never RELEASED by 2b
/// (`release_job_lease`'s own SQL carries `execution = 'queued'`, which an
/// inline row never satisfies, so it counts `not_required`) and never swept
/// by 2c either (the jobs sweep's same `execution = 'queued'` guard, and the
/// linked building sweep which only ranges over `queued` jobs) -- so
/// `release_job_leases` on a worker-less session with only an inline job in
/// flight releases nothing: `holds == Observed(HoldRelease { released: 0,
/// not_required: 1, failed: 0 })`, `sweep == ReleaseSweep { jobs: Some(0),
/// building: Some(0) }`, matching the doc's own "no-op by construction"
/// claim for the actual RELEASE effect (the row itself is untouched).
/// A second row, claimed by a DIFFERENT instance with a live lease, is
/// asserted completely untouched -- the call must never reach past its own
/// instance's rows.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_and_stop_report_matches_release_job_leases_on_the_pair_that_actually_differs() {
    // ----- worker-owning arm: release_and_stop's 2a-2c, 2e-2h (2d folded into 2a) -----
    let (owning_session, _owning_dir) = session(FAST_TIMING).await;
    let owning_handle = owning_session.enqueue(fine_tune(20_000), 0).await.unwrap();
    let worker = spawn_worker(&owning_session);
    let shared = shared_of(&worker);
    wait_in_flight(&shared, 1).await;

    let report = tokio::time::timeout(Duration::from_secs(10), worker.release_and_stop())
        .await
        .expect("RELEASE is bounded by two heartbeats")
        .unwrap();
    assert_eq!(
        report.holds,
        HoldReleaseOutcome::Observed(HoldRelease {
            released: 1,
            not_required: 0,
            failed: 0,
            attempted: 1,
        }),
        "2b's per-hold release is the actual releaser: {report:?}"
    );
    assert_eq!(
        report.sweep_two.jobs,
        Some(0),
        "2c's sweep must match zero ADDITIONAL rows once 2b already released the hold: \
         {report:?}"
    );
    let owning_row = owning_session
        .catalog()
        .get_job(&owning_handle.job_id)
        .await
        .unwrap();
    assert!(owning_row.lease_expires_at.is_none(), "{owning_row:?}");
    assert_eq!(owning_row.releases, 1, "{owning_row:?}");
    owning_session.close().await;

    // ----- worker-less arm: release_job_leases's bare 2b+2c -----
    let (leases_session, _leases_dir) = session(DEFAULT_TIMING).await;
    let source = unique_patents(&leases_session).await;
    let park = compute_test_hooks::arm(&source, compute_test_hooks::ParkPoint::BeforeDispatch);
    let runner = Arc::clone(&leases_session);
    let spec = never_dispatched_infer(&source);
    let run = tokio::spawn(async move { runner.run_now(spec).await });
    park.wait_parked().await;
    let inline = leases_session
        .catalog()
        .list_jobs()
        .await
        .unwrap()
        .into_iter()
        .find(|j| j.execution == "inline")
        .expect("the inline row exists while parked");
    assert!(inline.lease_expires_at.is_some());

    // A row claimed by a DIFFERENT instance -- never touched by this call.
    let other_handle = leases_session.enqueue(fine_tune(1), 0).await.unwrap();
    let other_lease = Duration::from_secs(3600);
    let other_claimed = leases_session
        .catalog()
        .claim_next("a-different-instance", &["fine_tune"], other_lease)
        .await
        .unwrap()
        .expect("claimed by a different instance");
    assert_ne!(
        other_claimed.claimed_by,
        Some(leases_session.instance_id().to_string())
    );

    let (holds, sweep) = leases_session.release_job_leases().await.unwrap();
    assert_eq!(
        holds,
        HoldReleaseOutcome::Observed(HoldRelease {
            released: 0,
            not_required: 1,
            failed: 0,
            attempted: 1,
        }),
        "an inline hold is registered but never released by 2b"
    );
    assert_eq!(
        sweep.jobs,
        Some(0),
        "the inline row is never swept: {sweep:?}"
    );
    assert_eq!(sweep.building, Some(0), "{sweep:?}");

    let inline_after = leases_session
        .catalog()
        .get_job(&inline.job_id)
        .await
        .unwrap();
    assert!(
        inline_after.lease_expires_at.is_some(),
        "the inline row's lease stays live"
    );
    assert_eq!(inline_after.releases, 0);

    let other_after = leases_session
        .catalog()
        .get_job(&other_handle.job_id)
        .await
        .unwrap();
    assert_eq!(
        other_after.lease_expires_at, other_claimed.lease_expires_at,
        "a different instance's live lease must be untouched: {other_after:?}"
    );
    assert_eq!(other_after.releases, 0, "{other_after:?}");
    assert_eq!(
        other_after.claimed_by,
        Some("a-different-instance".to_string()),
        "{other_after:?}"
    );

    park.release();
    let _ = run.await.unwrap();
    leases_session.close().await;
}

/// Producer-driven `stop_witnessed == false` arm: `stop_witnessed == false` requires BOTH
/// `stop_resolved == false` (this call's own `TakenHandle::take` lost the race — the handle
/// was already taken) AND `state_witnessed == false` (the terminal-state
/// watch fell back to the proxy read within one heartbeat). This is exactly
/// "any signal while draining is a RELEASE" (`deploy-server.md`): a DRAIN
/// (`stop_and_join`) already holds the handle and is waiting -- unbounded,
/// by design -- for the in-flight job to finish, so a RELEASE racing it
/// loses the handle and then times out waiting for a transition that the
/// still-running DRAIN has not produced.
///
/// BOTH halves are made deterministic, never a wall-clock/scheduler race:
/// the loop is parked mid-materialization with a REAL hold registered
/// (`in_flight == 1`), keyed on this session's own `writer_id`
/// (`jammi_db::store::mutable::test_hook`, the same mechanism and
/// `#[serial_test::serial(materialization_park)]` key
/// `release_mid_materialization_resumes_on_the_successor_without_backoff`
/// uses above) rather than `training_test_hooks::arm_pause_before_spawn_
/// blocking`, which parks a fine-tune run and this test's in-flight job is
/// a compute materialization. And
/// "the DRAIN wins the handle" is not left to whichever task the runtime
/// happens to schedule first -- `stop_and_join`'s future is polled EXACTLY
/// ONCE by hand, off the runtime, with a no-op waker. Its own body runs
/// `request_stop()` then `TakenHandle::take()` synchronously, with no
/// `.await` before the first one (`state_rx.wait_for(..).await`), so one
/// poll is guaranteed to execute the take and then return `Pending` — this
/// test's proof that the handle was actually taken, not an assumption about
/// scheduling order. This is not a literal `stop_witnessed: false`
/// construction; both booleans come from a real, running `EmbeddedWorker`.
#[serial_test::serial(materialization_park)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_release_racing_an_in_flight_drain_reads_stop_unwitnessed() {
    let (session, _dir) = session(FAST_TIMING).await;
    let source = unique_patents(&session).await;
    let writer_id = session.result_store().writer_id().to_string();
    let parked = arm_materialization(MaterializationPoint::Materialization, &writer_id);
    session
        .enqueue(embedding_spec(&source).into(), 0)
        .await
        .unwrap();
    let worker = spawn_worker(&session);
    let shared = shared_of(&worker);
    parked
        .wait_parked()
        .await
        .expect("the loop's writer parks inside finish");
    assert_eq!(
        in_flight(&shared),
        1,
        "a real hold must be registered before this test's race means anything"
    );

    // The DRAIN: unbounded by design (`stop_and_join` waits for the
    // in-flight job to finish). Polled exactly once, by hand, so the
    // handle is DETERMINISTICALLY taken before `release_and_stop` ever
    // runs -- exactly the state a genuine "SIGTERM then SIGINT before the
    // drain lands" sequence produces, without racing the scheduler for it.
    // Scoped in its own block: `std::pin::pin!`'s hidden local otherwise
    // outlives this function's tail, keeping `worker` borrowed past the
    // `drop(worker)` below.
    let report = {
        let mut drain_fut = std::pin::pin!(worker.stop_and_join());
        let waker = std::task::Waker::noop();
        let mut cx = std::task::Context::from_waker(waker);
        assert!(
            matches!(drain_fut.as_mut().poll(&mut cx), std::task::Poll::Pending),
            "stop_and_join's synchronous prefix (request_stop + TakenHandle::take) \
             must run to completion on its first poll and then suspend on the \
             still-parked loop's watch -- it must not resolve immediately"
        );

        tokio::time::timeout(Duration::from_secs(5), worker.release_and_stop())
            .await
            .expect(
                "a RELEASE that loses the handle race is bounded by one heartbeat, \
                 never by the DRAIN it lost to",
            )
            .unwrap()
        // `drain_fut` drops here, holding the taken handle right up to this
        // point (the parked materialization never resumes on its own) --
        // dropping it restores the loop task as `Abandoned` for
        // `EmbeddedWorker::drop` to reap.
    };
    assert!(
        !report.stop_witnessed,
        "a RELEASE that lost the handle race to a still-running DRAIN must read \
         stop UNWITNESSED, never fall back to a default `true`: {report:?}"
    );

    drop(worker);
    parked.release();
}

/// Producer-driven `HoldReleaseOutcome::Unobserved` on the SESSION arm
/// `runtime.rs`'s determinant table cites only `jammi-server`'s `liveness.rs`
/// `healthz_flips_to_503_...` test for this determinant, and that test drives the WORKER arm
/// (`EmbeddedWorker::release_and_stop`) only -- `InferenceSession::
/// release_job_leases`'s identical `Err => Unobserved` collapse has no
/// producer-driven oracle of its own. This drives it with the SAME
/// technique `liveness.rs` uses on the worker arm (`LeaseKeeper::
/// kill_thread_for_test`, a real dead keeper thread), applied to the
/// worker-less session path, never a literal `HoldReleaseOutcome`
/// construction.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_job_leases_is_unobserved_when_the_keeper_thread_is_dead() {
    let (session, _dir) = session(FAST_TIMING).await;
    assert!(session.lease_keeper().is_alive());
    session.lease_keeper().kill_thread_for_test();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while session.lease_keeper().is_alive() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the keeper thread never died within the bound"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let (holds, _sweep) = session.release_job_leases().await.unwrap();
    assert_eq!(
        holds,
        HoldReleaseOutcome::Unobserved,
        "a dead keeper thread must read as UNOBSERVED on the session arm too, \
         never a zero-count `Observed`: {holds:?}"
    );
}

/// Producer-driven `ReleaseSweep { jobs: None, building: Some(_) }`: every
/// other sweep test
/// that reaches an unconfirmed `ReleaseSweep` drops the `building` sweep
/// (a linked building table); none drives the `jobs` sweep statement's own
/// `Err` arm while `building` still confirms. This forces exactly that
/// asymmetric shape from a REAL producer -- dropping the `jobs.releases`
/// column (the same schema-level fault `lease_keeper.rs`'s
/// `release_job_holds_reports_failed_from_a_real_backend_fault` uses) fails
/// only `release_jobs_claimed_by`'s `UPDATE` (which writes that column);
/// `release_building_tables_of_claimant`'s `UPDATE` touches
/// `result_tables.lease_expires_at`, never `jobs.releases`, so it still
/// succeeds and confirms.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_and_stops_second_sweep_reports_jobs_none_building_some_from_a_real_fault() {
    use jammi_db::catalog::backend::{BackendImpl, TxOptions};
    use jammi_db::catalog::backend_sqlite::SqliteBackend;

    let (session, dir) = session(FAST_TIMING).await;
    session.enqueue(fine_tune(1), 0).await.unwrap();
    let worker = spawn_worker(&session);
    let shared = shared_of(&worker);
    wait_in_flight(&shared, 1).await;

    let fault_conn = BackendImpl::Sqlite(
        SqliteBackend::open(&dir.path().join("catalog.db"))
            .await
            .expect("second handle on the same catalog.db"),
    );
    fault_conn
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute("ALTER TABLE jobs DROP COLUMN releases", &[])
                    .await?;
                Ok(())
            })
        })
        .await
        .expect("inject the fault");

    let report = tokio::time::timeout(Duration::from_secs(10), worker.release_and_stop())
        .await
        .expect("RELEASE is bounded by two heartbeats")
        .unwrap();
    assert_eq!(
        report.sweep_two.jobs, None,
        "the jobs sweep statement itself must error under the real fault: {report:?}"
    );
    assert_eq!(
        report.sweep_two.building,
        Some(0),
        "the building sweep never touches the dropped column, so it must \
         still confirm: {report:?}"
    );
}

/// Producer-driven `ReleaseSweep { jobs: Some(_), building: None }` — the
/// mirror image of the test above. Beyond the COLUMN
/// `release_building_tables_of_claimant`'s `UPDATE` WRITES
/// (`result_tables.lease_expires_at`), that statement also NAMES a second
/// table it reads FROM, `result_tables` itself, and a fault on that table
/// is separable from the `jobs.releases` fault above because
/// `release_jobs_claimed_by` never touches `result_tables`. `ALTER TABLE
/// result_tables DROP COLUMN lease_expires_at` is rejected by the table's
/// own `idx_result_tables_lease` index, so this renames the table out from
/// under the statement instead — the same public `SqliteBackend::open` +
/// `CatalogBackend::transaction` surface the test above uses. No worker
/// and no enqueued job: `InferenceSession::release_job_leases` reaches
/// `release_sweep` directly, and the statement runs (and can fault) with
/// nothing claimed.
///
/// Named `one_sweep`, never `second_sweep`: `release_job_leases`
/// runs `release_sweep` exactly ONCE — there is no sweep #1 on this path to
/// be "second" after — so this drives `ReleaseSweep.building == None` on
/// the `SessionOnly` arm only. `EmbeddedWorker::release_and_stop`'s own
/// sweep #2 (2g) reaching `building == None` from a real backend fault has
/// no producer-driven oracle.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_job_leases_one_sweep_reports_building_none_jobs_some_from_a_real_fault() {
    use jammi_db::catalog::backend::{BackendImpl, TxOptions};
    use jammi_db::catalog::backend_sqlite::SqliteBackend;

    let (session, dir) = session(FAST_TIMING).await;

    let fault_conn = BackendImpl::Sqlite(
        SqliteBackend::open(&dir.path().join("catalog.db"))
            .await
            .expect("second handle on the same catalog.db"),
    );
    fault_conn
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "ALTER TABLE result_tables RENAME TO result_tables_gone",
                    &[],
                )
                .await?;
                Ok(())
            })
        })
        .await
        .expect("inject the fault");

    let (_holds, sweep) = session.release_job_leases().await.unwrap();
    assert_eq!(
        sweep.building, None,
        "the building sweep statement itself must error once its table is \
         renamed out from under it: {sweep:?}"
    );
    assert!(
        sweep.jobs.is_some(),
        "and the jobs sweep never touches result_tables, so it must still \
         confirm -- the mirror of the jobs-None/building-Some fault above: \
         {sweep:?}"
    );
}

/// (d)'s sibling: an inline `run_now` materialization (its `ResultTable`
/// hold on the shared keeper, adopted under the SAME `writer_id` the loop's
/// tables use, parked inside `finish`) survives a RELEASE: its building
/// lease stays live and is advanced by the keeper's next renewal, and after
/// the unpark the call completes with its table `ready`. A writer-scoped
/// sweep would have NULLed it.
#[serial_test::serial(materialization_park)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_now_materialization_survives_release_and_stop() {
    let (session, _dir) = session(FAST_TIMING).await;
    let source = unique_patents(&session).await;
    let writer_id = session.result_store().writer_id().to_string();
    let worker = spawn_worker(&session);
    let parked = arm_materialization(MaterializationPoint::Materialization, &writer_id);
    let runner = Arc::clone(&session);
    let spec = embedding_spec(&source);
    let run = tokio::spawn(async move { runner.run_now(spec).await });
    parked
        .wait_parked()
        .await
        .expect("run_now's writer parks inside finish");
    let inline = session
        .catalog()
        .list_jobs()
        .await
        .unwrap()
        .into_iter()
        .find(|j| j.execution == "inline")
        .expect("the inline row exists while parked");
    let table = inline.partial_result.clone().expect("its building table");
    let lease_before = session
        .catalog()
        .get_result_table(&table)
        .await
        .unwrap()
        .unwrap()
        .lease_expires_at
        .expect("a live building lease");

    let report = worker.release_and_stop().await.unwrap();
    assert_eq!(report.loop_state, LoopState::Stopped, "{report:?}");
    assert_eq!(report.sweep_one.building, Some(0), "{report:?}");
    let mid = session
        .catalog()
        .get_result_table(&table)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(mid.status, "building");
    assert!(
        mid.lease_expires_at.is_some(),
        "the inline building lease is untouched"
    );

    // The keeper keeps renewing it (the hold is not lost).
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    let renewed = session
        .catalog()
        .get_result_table(&table)
        .await
        .unwrap()
        .unwrap()
        .lease_expires_at
        .expect("still leased");
    assert!(renewed > lease_before, "the renewal keeps landing");

    parked.release();
    let result = run
        .await
        .unwrap()
        .expect("run_now completes after the unpark");
    assert!(matches!(&result, JobResult::Table { table: t, .. } if *t == table));
    assert_eq!(
        session
            .catalog()
            .get_result_table(&table)
            .await
            .unwrap()
            .unwrap()
            .status,
        "ready"
    );
    assert_eq!(
        session
            .catalog()
            .get_job(&inline.job_id)
            .await
            .unwrap()
            .status,
        JobStatus::Completed.to_string()
    );
}
