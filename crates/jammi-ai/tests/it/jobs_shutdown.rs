//! OPS (#482): the two shutdown modes on the LIBRARY — DRAIN
//! (`EmbeddedWorker::stop_and_join`) and RELEASE (`release_and_stop`) —
//! with every arm of the RELEASE mechanism pinned against the loop's own
//! rendezvous points (`loop_test_hooks`, `claim_test_hooks`, the compute and
//! materialization parks), never against a wall clock:
//!
//! * R5(a): a claim parked before COMMIT commits on unpark and self-releases;
//! * R5(a'): a stop during the idle sleep leaves the row `queued`;
//! * R5(b): the claim→hold prologue self-releases under `Releasing`;
//! * R5(c): the timeout arm's row is recovered by arm 1a, never `failed`;
//! * R5(d): `run_now` never changes `in_flight` and an inline
//!   materialization survives a RELEASE with its lease live;
//! * a compute job released mid-materialization resumes on the successor
//!   without a `BackOff` (the escape `esc-110` fix on the RELEASE path);
//! * a released training job is never finalized by the abandoned thread and
//!   its `_resume` manifest epoch never advances.

use std::sync::Arc;
use std::time::Duration;

use jammi_ai::fine_tune::spec::{TrainingCommon, TrainingSpec};
use jammi_ai::fine_tune::worker::{
    loop_test_hooks, training_test_hooks, EmbeddedWorker, JobWorker, LoopState, StopOutcome,
    WorkerShared,
};
use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod};
use jammi_ai::jobs::{compute_test_hooks, ComputeSpec, JobResult, JobSpec};
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::claim_test_hooks;
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
        },
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

async fn wait_in_flight(shared: &WorkerShared, want: usize) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    while shared.in_flight() != want {
        assert!(
            tokio::time::Instant::now() < deadline,
            "in_flight never reached {want} (now {})",
            shared.in_flight()
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
        .expect("a drain is bounded by the in-flight job's own duration")
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
// O2 — a cancelled `stop_and_join` must not detach the loop task
// ---------------------------------------------------------------------------

/// F1's embedded (non-server) arm. `stop_and_join` awaits the in-flight
/// job's own terminal `LoopState` for however long that job takes (D4) —
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
    let worker = spawn_worker(&session);
    let shared = shared_of(&worker);
    wait_in_flight(&shared, 1).await;
    // At least one epoch boundary has landed a bundle, so there is an epoch
    // to compare.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    while resume_epoch(&session, &handle.job_id).await.is_none() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "no resume bundle landed"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let threads_before = training_test_hooks::training_threads_finished();

    let report = tokio::time::timeout(Duration::from_secs(10), worker.release_and_stop())
        .await
        .expect("RELEASE is bounded by two heartbeats")
        .unwrap();
    assert_eq!(report.loop_state, LoopState::Aborted, "{report:?}");
    assert_eq!(report.holds_released, 1, "{report:?}");
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

    // The abandoned thread finishes (bails at its next boundary) …
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    while training_test_hooks::training_threads_finished() <= threads_before {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the abandoned training thread never returned"
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
    assert_eq!(shared.in_flight(), 1);
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

/// R5(a): the loop is parked INSIDE `claim_next`, before COMMIT, with
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
    assert_eq!(shared.in_flight(), 0);

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

/// R5(a'): the stop lands while the loop is in its idle sleep; the job is
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

/// R5(b): the loop is parked in the claim→hold prologue (after COMMIT,
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
            shared.in_flight(),
            0,
            "{kind}: the hold is not registered yet"
        );

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

/// R5(c): the prologue park is held PAST the heartbeat bound, so 2e's wait
/// times out and RELEASE takes its abort arm — the ONLY path where the loop
/// is aborted with `in_flight == 0`. The loop reports `Aborted`; no hold was
/// ever registered (`holds_released 0`); the row the abort left is the
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

    let started = tokio::time::Instant::now();
    let report = worker.release_and_stop().await.unwrap();
    assert!(
        started.elapsed() >= Duration::from_secs(FAST_TIMING.heartbeat),
        "the cooperative wait runs to its heartbeat bound first"
    );
    assert_eq!(report.loop_state, LoopState::Aborted, "{report:?}");
    assert_eq!(report.holds_released, 0, "{report:?}");
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

/// R5(d): an inline `run_now` (parked before dispatch) is not the loop's:
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
    assert_eq!(shared.in_flight(), 0, "an inline claim never counts");
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
        report.holds_released, 0,
        "the inline hold is never released"
    );
    assert_eq!(shared.in_flight(), 0);
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

/// R5(d)'s sibling: an inline `run_now` materialization (its `ResultTable`
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
