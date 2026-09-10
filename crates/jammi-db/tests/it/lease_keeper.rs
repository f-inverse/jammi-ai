//! `LeaseKeeper` (N3): a dedicated OS thread that renews every held
//! lease from its OWN runtime and OWN catalog connection, immune to the
//! caller's main runtime being starved by CPU-bound work.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use jammi_db::catalog::jobs_repo::SubmitJobParams;
use jammi_db::catalog::lease_keeper::{LeaseKeeper, LeaseTarget};
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::status::JobExecution;
use jammi_db::catalog::Catalog;
use jammi_db::config::LeaseConfig;
use jammi_db::model_task::ModelTask;
use tempfile::tempdir;

/// Start a keeper that reopens a FRESH `Catalog` (its own connection pool, on
/// its own dedicated thread's own runtime) at `dir` every time it is invoked
/// — here exactly once, from inside the keeper thread.
async fn keeper_for(
    dir: PathBuf,
    intervals: jammi_db::catalog::lease::LeaseIntervals,
) -> Arc<LeaseKeeper> {
    LeaseKeeper::start(
        move || {
            let dir = dir.clone();
            Box::pin(async move { Catalog::open(&dir).await })
        },
        intervals,
    )
    .await
    .expect("the keeper connects to a local SQLite catalog within the lease window")
}

/// Short, valid lease timing for the test: `heartbeat * 2 < lease`.
fn fast_intervals() -> jammi_db::catalog::lease::LeaseIntervals {
    LeaseConfig {
        duration_secs: 3,
        heartbeat_secs: 1,
    }
    .intervals()
    .unwrap()
}

async fn seeded_catalog(dir: &std::path::Path) -> Catalog {
    let catalog = Catalog::open(dir).await.unwrap();
    catalog
        .register_model(RegisterModelParams {
            model_id: "keeper-base",
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: None,
            config_json: None,
        })
        .await
        .unwrap();
    catalog
}

/// N+1 CPU-bound tasks (no `.await` inside their loop) saturate an N-thread
/// runtime for a real wall-clock window; the keeper's hold for a
/// claimed job still renews `lease_expires_at` during that window — proving
/// it runs from a thread the busy runtime cannot starve.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn holds_renew_from_a_thread_while_the_callers_runtime_is_blocked() {
    const WORKER_THREADS: usize = 2;
    const BUSY_TASKS: usize = WORKER_THREADS + 1;
    let busy_for = Duration::from_millis(2_200); // > 2 heartbeat ticks at 1s

    let dir = tempdir().unwrap();
    let catalog = seeded_catalog(dir.path()).await;
    catalog
        .submit_job(SubmitJobParams {
            job_id: "kept-alive",
            kind: "fine_tune",
            execution: JobExecution::Queued,
            spec: "{}",
            model_ref: Some("keeper-base::1"),
            output_model_id: None,
            model_source: None,
            priority: 0,
        })
        .await
        .unwrap();
    let claimed = catalog
        .claim_next("instance-a", &["fine_tune"], Duration::from_secs(3))
        .await
        .unwrap()
        .expect("job claimed");
    let before = claimed
        .lease_expires_at
        .clone()
        .expect("lease stamped on claim");

    let intervals = fast_intervals();
    let keeper = keeper_for(dir.path().to_path_buf(), intervals).await;
    let hold = keeper.hold(LeaseTarget::Job {
        job_id: claimed.job_id.clone(),
        instance_id: "instance-a".to_string(),
        attempts: claimed.attempts,
    });

    // Saturate this runtime's worker threads with tasks that never yield —
    // the shape a CPU-bound inline job's blocking work takes. All BUSY_TASKS
    // spin against the SAME deadline, so together they occupy the runtime for
    // exactly `busy_for`, regardless of how many threads are actually free to
    // run them concurrently.
    let deadline = std::time::Instant::now() + busy_for;
    let mut handles = Vec::with_capacity(BUSY_TASKS);
    for _ in 0..BUSY_TASKS {
        handles.push(tokio::spawn(async move {
            while std::time::Instant::now() < deadline {
                // A tight, non-yielding spin — no `.await` point inside the
                // loop, so this task never releases its worker thread back
                // to the scheduler until the deadline passes.
                std::hint::black_box(0..1_000);
            }
        }));
    }
    // Joining does not depend on the (possibly-starved) timer driver: task
    // completion wakes its `JoinHandle` directly when the task itself
    // returns, which every one of the spawned tasks above does once its own
    // busy-loop condition goes false.
    for h in handles {
        h.await.expect("busy task panicked");
    }

    assert!(
        !hold.lost(),
        "the hold must not have been marked lost during the busy window"
    );
    let after = catalog
        .get_job("kept-alive")
        .await
        .unwrap()
        .lease_expires_at
        .expect("lease still stamped");
    assert!(
        after > before,
        "lease_expires_at must have advanced while the caller's runtime was fully blocked \
         (before = {before}, after = {after})"
    );
}

/// `lost()` flips to `true` once a peer's write makes the hold's own
/// renew CAS miss (here manufactured directly: a peer steals `claimed_by`,
/// exactly the shape a real reclaim leaves behind) — observed within one
/// heartbeat tick, with no action from the caller.
#[tokio::test]
async fn lost_flag_sets_after_a_peer_reclaims_the_held_job() {
    let dir = tempdir().unwrap();
    let catalog = seeded_catalog(dir.path()).await;
    catalog
        .submit_job(SubmitJobParams {
            job_id: "stolen",
            kind: "fine_tune",
            execution: JobExecution::Queued,
            spec: "{}",
            model_ref: Some("keeper-base::1"),
            output_model_id: None,
            model_source: None,
            priority: 0,
        })
        .await
        .unwrap();
    let claimed = catalog
        .claim_next("instance-a", &["fine_tune"], Duration::from_secs(3))
        .await
        .unwrap()
        .expect("job claimed");

    let intervals = fast_intervals();
    let keeper = keeper_for(dir.path().to_path_buf(), intervals).await;
    let hold = keeper.hold(LeaseTarget::Job {
        job_id: claimed.job_id.clone(),
        instance_id: "instance-a".to_string(),
        attempts: claimed.attempts,
    });
    assert!(!hold.lost(), "must not start lost");

    // A peer's reclaim, manufactured directly: steal `claimed_by` so this
    // hold's own `WHERE claimed_by = 'instance-a'` renew CAS matches
    // zero rows on the keeper's next tick — exactly the state a real
    // `reclaim_expired_jobs` requeue-then-reclaim would leave behind.
    use jammi_db::catalog::backend::TxOptions;
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE jobs SET claimed_by = 'peer-instance' WHERE job_id = 'stolen'",
                    &[],
                )
                .await
            })
        })
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(1_500)).await;
    assert!(
        hold.lost(),
        "the hold must observe the peer's steal within one heartbeat tick"
    );
}

/// A keeper whose connect factory never succeeds must not come back as a
/// handle whose holds all read "live": `start` retries inside the lease
/// window (more than one attempt lands) and then returns the typed
/// `Catalog` error. A `Config` error from the factory is permanent and
/// ends the retry at once — well inside the window.
#[tokio::test]
async fn start_returns_a_typed_error_when_the_connect_never_succeeds() {
    use std::sync::atomic::{AtomicU32, Ordering};

    let intervals = fast_intervals(); // lease 3 s, heartbeat 1 s
    let attempts = Arc::new(AtomicU32::new(0));
    let counted = Arc::clone(&attempts);
    let started = std::time::Instant::now();
    let err = LeaseKeeper::start(
        move || {
            counted.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {
                Err::<Catalog, _>(jammi_db::error::JammiError::Catalog(
                    "connection refused".into(),
                ))
            })
        },
        intervals,
    )
    .await
    .err()
    .expect("a keeper that never connects must not start");
    let elapsed = started.elapsed();
    assert!(
        matches!(&err, jammi_db::error::JammiError::Catalog(m) if m.contains("lease window") && m.contains("connection refused")),
        "expected the typed Catalog error naming the window and the last connect error, got {err:?}"
    );
    let n = attempts.load(Ordering::SeqCst);
    assert!(
        n > 1,
        "the connect must have been retried inside the window, got {n} attempt(s)"
    );
    assert!(
        elapsed >= Duration::from_secs(3) && elapsed < Duration::from_secs(8),
        "start must give up at the lease window, not before and not much after (took {elapsed:?})"
    );

    let started = std::time::Instant::now();
    let err = LeaseKeeper::start(
        || {
            Box::pin(async {
                Err::<Catalog, _>(jammi_db::error::JammiError::Config(
                    "bad catalog url".into(),
                ))
            })
        },
        intervals,
    )
    .await
    .err()
    .expect("a permanent Config error must fail start");
    assert!(
        matches!(&err, jammi_db::error::JammiError::Config(m) if m.contains("bad catalog url")),
        "a Config error is returned as-is, got {err:?}"
    );
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "a permanent error must not be retried up to the window"
    );
}

/// A keeper thread that dies AFTER `start` succeeded (here: a test hook
/// makes it panic at its next tick, the way a defect inside the renewal
/// loop would) must be visible to every hold it issued — a job hold and an
/// instance hold alike report `lost()`, the `lost_flag()` clone a training
/// loop polls flips too, and `is_alive()` reads `false` — all within the
/// lease window, with no action from the holder.
#[cfg(feature = "test-hooks")]
#[tokio::test]
async fn every_hold_reports_lost_when_the_keeper_thread_dies() {
    let dir = tempdir().unwrap();
    let catalog = seeded_catalog(dir.path()).await;
    catalog
        .submit_job(SubmitJobParams {
            job_id: "orphaned-by-death",
            kind: "fine_tune",
            execution: JobExecution::Queued,
            spec: "{}",
            model_ref: Some("keeper-base::1"),
            output_model_id: None,
            model_source: None,
            priority: 0,
        })
        .await
        .unwrap();
    let claimed = catalog
        .claim_next("instance-a", &["fine_tune"], Duration::from_secs(3))
        .await
        .unwrap()
        .expect("job claimed");
    catalog
        .upsert_instance("instance-a", None, None)
        .await
        .unwrap();

    let intervals = fast_intervals();
    let keeper = keeper_for(dir.path().to_path_buf(), intervals).await;
    assert!(keeper.is_alive(), "a keeper `start` returned is alive");
    let job_hold = keeper.hold(LeaseTarget::Job {
        job_id: claimed.job_id.clone(),
        instance_id: "instance-a".to_string(),
        attempts: claimed.attempts,
    });
    let instance_hold = keeper.hold(LeaseTarget::Instance("instance-a".to_string()));
    let polled_flag = job_hold.lost_flag();
    assert!(
        !job_hold.lost() && !instance_hold.lost(),
        "holds start live"
    );

    keeper.kill_thread_for_test();
    // Kill lands at the next heartbeat tick (1 s); wait well inside the
    // 3 s lease window so a flip observed here is the death arm, not the
    // "no renewal for a whole lease window" arm.
    tokio::time::sleep(Duration::from_millis(1_800)).await;

    assert!(!keeper.is_alive(), "the keeper must report its thread dead");
    assert!(
        job_hold.lost(),
        "a job hold must read lost once the keeper thread is dead"
    );
    assert!(
        instance_hold.lost(),
        "an instance hold must read lost once the keeper thread is dead"
    );
    assert!(
        polled_flag.load(std::sync::atomic::Ordering::SeqCst),
        "the lost_flag() clone a training loop polls must flip on keeper death too"
    );
}

/// `last_renewed_at()` on a hold advances once the keeper's renewal lands
/// on its row — the stamp the staleness arm of `lost()` is judged from.
#[tokio::test]
async fn a_holds_last_renewed_at_advances_with_each_landed_renewal() {
    let dir = tempdir().unwrap();
    let catalog = seeded_catalog(dir.path()).await;
    catalog
        .upsert_instance("instance-b", None, None)
        .await
        .unwrap();
    let intervals = fast_intervals();
    let keeper = keeper_for(dir.path().to_path_buf(), intervals).await;
    let hold = keeper.hold(LeaseTarget::Instance("instance-b".to_string()));
    let at_hold = hold.last_renewed_at();
    let keeper_pass_at_start = keeper.last_renewed_at();
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    assert!(
        hold.last_renewed_at() > at_hold,
        "one heartbeat tick later the hold's own renewal stamp has advanced"
    );
    assert!(
        keeper.last_renewed_at() > keeper_pass_at_start,
        "the keeper's pass stamp advances with each tick"
    );
    assert!(!hold.lost());
}
