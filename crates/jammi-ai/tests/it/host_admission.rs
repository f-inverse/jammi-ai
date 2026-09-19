//! `HostAdmission` — this host's single job-slot holder cell and shutdown
//! phase, exercised at the cell and through
//! the REAL claim loop.
//!
//! The lattice: `Free` admits a rank; `ClaimProbe` is waited on for at most
//! one bound, then admits if freed or refuses; `JobRun` and another `Rank`
//! refuse at once; the SAME job at a GREATER attempt takes the slot from its
//! elder (whose guard then leaves the cell alone); an inline `run_now` is
//! outside the exclusion; and the claim loop moves the holder `Free →
//! ClaimProbe → JobRun → Free` around every claim, never claiming while a
//! rank is held (a peer is a fleet worker with a spare admission holder).

use std::sync::Arc;
use std::time::Duration;

use jammi_ai::fine_tune::worker::{
    loop_test_hooks, EmbeddedWorker, Holder, HolderBusy, HostAdmission, JobWorker, LoopState,
    WorkerPhase,
};
use jammi_ai::jobs::{compute_test_hooks, ComputeSpec};
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::instance::InstanceRegistration;
use jammi_db::catalog::status::JobStatus;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::CachePolicy;

use crate::common;

/// A bare admission cell, no session: the lattice at the cell.
fn cell() -> Arc<HostAdmission> {
    HostAdmission::new(Arc::new(InstanceRegistration::new(
        "cell-instance",
        None,
        None,
        None,
        None,
    )))
}

/// A session with fast timing (`lease 3s`, `heartbeat 1s`, `idle_poll 1s`)
/// and the `patents` fixture registered, for the loop-driven tests.
async fn session() -> (Arc<InferenceSession>, tempfile::TempDir) {
    let dir = tempfile::TempDir::new().unwrap();
    let mut config = common::test_config(dir.path());
    config.lease.duration_secs = 3;
    config.lease.heartbeat_secs = 1;
    config.worker.idle_poll_secs = 1;
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    (session, dir)
}

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

/// A compute spec whose dispatch parks (`compute_test_hooks::arm(..,
/// BeforeDispatch)`) — the same never-dispatched shape `jobs_shutdown.rs`
/// uses, so a job can be held inside its run without any model existing.
fn never_dispatched_infer(source: &str) -> ComputeSpec {
    ComputeSpec::Infer {
        source_id: source.to_string(),
        model_id: "local:/nonexistent/model/for-the-admission-oracles".to_string(),
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

async fn wait_holder(admission: &HostAdmission, pred: impl Fn(&Holder) -> bool, what: &str) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    while !pred(&admission.holder()) {
        assert!(
            tokio::time::Instant::now() < deadline,
            "holder never became {what} (now {:?})",
            admission.holder()
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

// ---------------------------------------------------------------------------
// The lattice at the cell (c2').
// ---------------------------------------------------------------------------

/// `Free` admits; the hold names the rank; dropping the hold frees the slot.
/// Mutation proof: a `RankHold::drop` that never resets the cell leaves the
/// final assertion reading `Rank{..}`.
#[tokio::test]
async fn free_admits_a_rank_and_dropping_the_hold_frees_the_slot() {
    let cell = cell();
    assert_eq!(cell.holder(), Holder::Free);
    let hold = cell.try_hold_rank("job-a", 1).expect("Free admits");
    assert_eq!(hold.job_id(), "job-a");
    assert_eq!(hold.attempt(), 1);
    assert_eq!(
        cell.holder(),
        Holder::Rank {
            job_id: "job-a".into(),
            attempt: 1
        }
    );
    drop(hold);
    assert_eq!(cell.holder(), Holder::Free);
}

/// `JobRun` and another `Rank` refuse AT ONCE — `admit_rank` returns well
/// inside its bound, naming the holder it found; nothing is waited on.
/// Mutation proof: waiting on `JobRun` the way a `ClaimProbe` is waited on
/// makes the elapsed assertion fail.
#[tokio::test]
async fn a_job_run_or_another_rank_refuses_at_once() {
    let cell = cell();
    let _elder = cell.try_hold_rank("job-a", 1).unwrap();
    let started = tokio::time::Instant::now();
    let busy = cell
        .admit_rank("job-b", 1, Duration::from_secs(5))
        .await
        .expect_err("another job's rank must be refused");
    assert_eq!(
        busy,
        HolderBusy::Rank {
            job_id: "job-a".into(),
            attempt: 1
        }
    );
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "a held Rank refuses at once, never after the bound"
    );
    drop(_elder);

    let _job = cell.hold_for_test(Holder::JobRun);
    let started = tokio::time::Instant::now();
    let busy = cell
        .admit_rank("job-b", 1, Duration::from_secs(5))
        .await
        .expect_err("a running loop job must be refused");
    assert_eq!(busy, HolderBusy::JobRun);
    assert!(started.elapsed() < Duration::from_secs(1));
}

/// The SAME job at an EQUAL attempt is a duplicate assignment of a held
/// session — refused; at a GREATER attempt it takes the slot in place, the
/// elder's guard then leaves the cell alone (the successor's hold survives
/// the elder's drop), and the successor's drop frees it. Mutation proof:
/// dropping the `held_attempt < attempt` arm refuses the successor;
/// dropping `RankHold::drop`'s identity check frees the slot under the
/// successor when the elder drops.
#[tokio::test]
async fn the_same_job_at_a_greater_attempt_takes_the_slot_and_the_elder_leaves_it() {
    let cell = cell();
    let elder = cell.try_hold_rank("job-a", 1).unwrap();
    assert_eq!(
        cell.try_hold_rank("job-a", 1)
            .expect_err("an equal attempt is a duplicate"),
        HolderBusy::Rank {
            job_id: "job-a".into(),
            attempt: 1
        }
    );
    let successor = cell
        .try_hold_rank("job-a", 2)
        .expect("a greater attempt of the same job takes the slot");
    assert_eq!(
        cell.holder(),
        Holder::Rank {
            job_id: "job-a".into(),
            attempt: 2
        }
    );
    drop(elder);
    assert_eq!(
        cell.holder(),
        Holder::Rank {
            job_id: "job-a".into(),
            attempt: 2
        },
        "the superseded elder's drop must not free the successor's slot"
    );
    drop(successor);
    assert_eq!(cell.holder(), Holder::Free);
}

/// A `ClaimProbe` is waited on: freed within the bound ⇒ admitted (well
/// before the bound elapses); still probing at the bound ⇒ refused
/// `ClaimProbe`, and only then. Mutation proof: refusing a probe at once
/// fails the first half; waiting past the bound fails the second half's
/// upper elapsed assertion; dropping the retry after a change fails the
/// first half (the freed slot is never re-tried).
#[tokio::test]
async fn a_claim_probe_is_waited_out_then_admitted_if_freed_or_refused_at_the_bound() {
    let cell = cell();
    let probe = cell.hold_for_test(Holder::ClaimProbe);
    let admit = {
        let cell = Arc::clone(&cell);
        tokio::spawn(async move { cell.admit_rank("job-a", 1, Duration::from_secs(3)).await })
    };
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(!admit.is_finished(), "must still be waiting on the probe");
    let started = tokio::time::Instant::now();
    drop(probe);
    let hold = admit
        .await
        .unwrap()
        .expect("freed within the bound: admitted");
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "admission follows the probe's release, not the bound"
    );
    assert_eq!(
        cell.holder(),
        Holder::Rank {
            job_id: "job-a".into(),
            attempt: 1
        }
    );
    drop(hold);

    let _probe = cell.hold_for_test(Holder::ClaimProbe);
    let bound = Duration::from_millis(400);
    let started = tokio::time::Instant::now();
    let busy = cell
        .admit_rank("job-a", 1, bound)
        .await
        .expect_err("a probe that never resolves refuses at the bound");
    let elapsed = started.elapsed();
    assert_eq!(busy, HolderBusy::ClaimProbe);
    assert!(elapsed >= bound, "refused before the bound: {elapsed:?}");
    assert!(
        elapsed < bound + Duration::from_secs(1),
        "refused long after the bound: {elapsed:?}"
    );
}

/// The phase cell: `begin_drain` moves `Running → Draining` and never
/// regresses a `Releasing`; `begin_release` wins from any phase; a
/// receiver observes the transition.
#[tokio::test]
async fn the_phase_cell_drains_once_and_release_wins() {
    let cell = cell();
    let mut rx = cell.phase_receiver();
    assert_eq!(cell.phase(), WorkerPhase::Running);
    assert!(cell.begin_drain());
    assert!(!cell.begin_drain(), "a second drain changes nothing");
    assert_eq!(cell.phase(), WorkerPhase::Draining);
    rx.wait_for(|p| *p != WorkerPhase::Running).await.unwrap();
    cell.begin_release().await;
    assert_eq!(cell.phase(), WorkerPhase::Releasing);
    assert!(
        !cell.begin_drain(),
        "a drain after a release never regresses it"
    );
    assert_eq!(cell.phase(), WorkerPhase::Releasing);
}

// ---------------------------------------------------------------------------
// The real claim loop.
// ---------------------------------------------------------------------------

/// The loop moves the holder `Free → ClaimProbe` (observed parked in the
/// claim→hold prologue, `ParkPoint::BeforeHold`) `→ JobRun` (observed
/// parked inside the run, before dispatch) `→ Free` (the run returned).
/// Mutation proof: flipping `ClaimProbe → JobRun` at `claim_next`'s `Some`
/// arm instead of at the hold site reads `JobRun` at the prologue park;
/// dropping the `ClaimGuard` reset leaves `JobRun` after the run.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_claim_loop_moves_the_holder_free_probe_run_free() {
    let (session, _dir) = session().await;
    let source = unique_patents(&session).await;
    let handle = session
        .enqueue(never_dispatched_infer(&source).into(), 0)
        .await
        .unwrap();
    let prologue = loop_test_hooks::arm(&handle.job_id, loop_test_hooks::ParkPoint::BeforeHold);
    let dispatch = compute_test_hooks::arm(&source, compute_test_hooks::ParkPoint::BeforeDispatch);
    let admission = Arc::clone(session.host_admission());
    assert_eq!(admission.holder(), Holder::Free);

    let worker = spawn_worker(&session);
    prologue.wait_parked().await;
    assert_eq!(
        admission.holder(),
        Holder::ClaimProbe,
        "the claim committed but the hold is not registered: still a probe"
    );
    prologue.release();
    dispatch.wait_parked().await;
    assert_eq!(
        admission.holder(),
        Holder::JobRun,
        "inside the run under a registered hold"
    );
    dispatch.release();
    wait_holder(&admission, |h| *h == Holder::Free, "Free after the run").await;
    let row = session.catalog().get_job(&handle.job_id).await.unwrap();
    assert!(
        row.is_terminal(),
        "the run reached a terminal state: {row:?}"
    );

    worker.stop_and_join().await.unwrap();
    session.close().await;
}

/// A peer never claims while it holds a rank: with a `Rank` held,
/// an idle loop skips every claim (`claim_next` is never called across two
/// idle polls, the queued job stays `queued`); the moment the hold drops,
/// the loop claims it. Mutation proof: removing the `probe_claim` gate lets
/// the loop claim beside the held rank — the queued row leaves `queued`
/// while the hold is still held.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_idle_loop_never_claims_while_a_rank_is_held() {
    let (session, _dir) = session().await;
    let source = unique_patents(&session).await;
    let admission = Arc::clone(session.host_admission());
    let worker = spawn_worker(&session);
    // Let the loop reach its idle sleep once before holding the slot, so
    // the hold is what the next iterations observe.
    let start_calls = loop_test_hooks::claim_next_calls(session.instance_id());
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    while loop_test_hooks::claim_next_calls(session.instance_id()) == start_calls {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the loop never claimed once"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    // The counter ticks while the loop still holds `ClaimProbe` (it is
    // recorded before `claim_next` returns), so a hold taken the instant it
    // moves races that probe and is refused `ClaimProbe`. Wait for the probe
    // to resolve to `Free` — the state the hold is meant to observe.
    wait_holder(
        &admission,
        |h| *h == Holder::Free,
        "Free after the first probe",
    )
    .await;
    let hold = admission.try_hold_rank("rank-job", 1).unwrap();
    // Any probe already past the gate resolves within one iteration.
    tokio::time::sleep(Duration::from_millis(1200)).await;
    let calls_at_hold = loop_test_hooks::claim_next_calls(session.instance_id());
    let handle = session
        .enqueue(never_dispatched_infer(&source).into(), 0)
        .await
        .unwrap();
    let dispatch = compute_test_hooks::arm(&source, compute_test_hooks::ParkPoint::BeforeDispatch);
    // Two idle polls (1s each) plus slack: the loop wakes twice and must
    // not claim either time.
    tokio::time::sleep(Duration::from_millis(2500)).await;
    assert_eq!(
        loop_test_hooks::claim_next_calls(session.instance_id()),
        calls_at_hold,
        "claim_next must not be called while a rank is held"
    );
    let row = session.catalog().get_job(&handle.job_id).await.unwrap();
    assert_eq!(row.status, JobStatus::Queued.to_string(), "{row:?}");
    assert_eq!(
        admission.holder(),
        Holder::Rank {
            job_id: "rank-job".into(),
            attempt: 1
        }
    );

    drop(hold);
    dispatch.wait_parked().await;
    assert_eq!(
        admission.holder(),
        Holder::JobRun,
        "freed: the loop claimed"
    );
    dispatch.release();
    wait_holder(&admission, |h| *h == Holder::Free, "Free after the run").await;
    worker.stop_and_join().await.unwrap();
    session.close().await;
}

/// An inline `run_now` is OUTSIDE the exclusion: parked before dispatch it
/// leaves the holder `Free`, and a rank is admitted beside it. Mutation
/// proof: an inline hold site that calls `job_running()` after a
/// `probe_claim()` reads `JobRun` here.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_inline_run_now_never_touches_the_holder() {
    let (session, _dir) = session().await;
    let source = unique_patents(&session).await;
    let admission = Arc::clone(session.host_admission());
    let park = compute_test_hooks::arm(&source, compute_test_hooks::ParkPoint::BeforeDispatch);
    let runner = Arc::clone(&session);
    let spec = never_dispatched_infer(&source);
    let run = tokio::spawn(async move { runner.run_now(spec).await });
    park.wait_parked().await;
    assert_eq!(
        admission.holder(),
        Holder::Free,
        "an inline claim never holds the slot"
    );
    let hold = admission
        .try_hold_rank("rank-beside-inline", 1)
        .expect("a rank is admitted beside an inline run");
    park.release();
    let _ = run.await.unwrap();
    assert_eq!(
        admission.holder(),
        Holder::Rank {
            job_id: "rank-beside-inline".into(),
            attempt: 1
        },
        "the inline run's end must not free a rank's hold"
    );
    drop(hold);
    session.close().await;
}

/// RELEASE reads the holder KIND: with a `Rank` held beside a
/// loop that is NOT idle — parked after `reclaim_expired_jobs`, before its
/// second gate read (`arm_after_reclaim`), the one place an idle-slot loop
/// can still be mid-round-trip when 2e runs — `release_and_stop` neither
/// aborts the loop nor touches the rank's hold: 2e waits one heartbeat, the
/// park is released 300 ms in, the loop's second gate read sees
/// `Releasing` and exits cooperatively (`Stopped`, witnessed); the phase
/// flips to `Releasing`, which is what ends the held rank's own session.
/// Mutation proof: treating a `Rank` like `JobRun` (abort now)
/// aborts the parked task before the park is ever released — the report
/// reads `Aborted`, not `Stopped`. An idle loop would exit at 2a's stop
/// before 2e runs and could not tell the two arms apart, which is why the
/// loop is parked here.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_and_stop_beside_a_held_rank_exits_cooperatively_and_flips_the_phase() {
    let (session, _dir) = session().await;
    let admission = Arc::clone(session.host_admission());
    let park = loop_test_hooks::arm_after_reclaim(session.instance_id());
    let worker = spawn_worker(&session);
    park.wait_parked().await;
    let hold = admission.try_hold_rank("rank-job", 1).unwrap();
    let mut phase = admission.phase_receiver();
    let releaser = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(300)).await;
        park.release();
    });
    let report = worker.release_and_stop().await.unwrap();
    releaser.await.unwrap();
    assert_eq!(
        report.loop_state,
        LoopState::Stopped,
        "a held Rank is not loop work: 2e waits for the cooperative exit, never aborts: {report:?}"
    );
    assert!(report.stop_witnessed, "{report:?}");
    phase
        .wait_for(|p| *p == WorkerPhase::Releasing)
        .await
        .unwrap();
    assert_eq!(
        admission.holder(),
        Holder::Rank {
            job_id: "rank-job".into(),
            attempt: 1
        },
        "RELEASE never writes the rank's hold; the rank's own session ends it"
    );
    drop(hold);
    session.close().await;
}
