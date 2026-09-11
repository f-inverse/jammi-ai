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

/// `shutdown_and_join` closes the keeper's OWN catalog connection —
/// independent of, and never released by, closing any OTHER handle on the
/// same directory (the keeper opens its own connection on its own
/// dedicated thread; see the module docs). Before this method existed the
/// only way to stop the keeper was `Drop`, which is flag-only and never
/// waits for the thread to exit — so this connection stayed open for as
/// long as the process ran, and for the SQLite backend that alone is
/// enough to keep the `unix-excl` VFS's process-scoped exclusive lock held,
/// refusing a successor even after every other handle had let go.
///
/// Proven with a FRESH `Catalog::open` on the same directory immediately
/// after `shutdown_and_join` returns: it must land well inside the 5 s
/// busy timeout a still-open keeper connection would force it to wait out
/// (`backend_sqlite`'s `busy_timeout(Duration::from_secs(5))`) — the same
/// distinction the Python
/// `test_close_hands_the_catalog_directory_to_a_successor_process` oracle
/// measures cross-process.
///
/// `shutdown_and_join`'s own timeout is generous (60 s), not the ~1 s a
/// `Notify`-woken thread normally takes: this suite runs hundreds of
/// SQLite pools open-and-close in one process, back to back, and the
/// actual OS-level `join()` this call performs (via `spawn_blocking`,
/// never `JoinHandle::is_finished` polling — see that method's own doc for
/// why) was measured to occasionally take 30+ real seconds under that
/// accumulated load, while still genuinely completing and genuinely
/// releasing the lock (proven by the reopen below landing well under its
/// own bound regardless). A single production process never reaches that
/// load shape — this generous window is a test-harness accommodation, not
/// evidence the mechanism itself is slow.
#[tokio::test]
async fn shutdown_and_join_releases_the_keepers_own_catalog_connection() {
    let dir = tempdir().unwrap();
    let intervals = fast_intervals();
    let keeper = keeper_for(dir.path().to_path_buf(), intervals).await;
    // A live hold, so the keeper is doing real renewal work over the
    // connection this test proves gets closed — not merely an idle thread
    // that happens to have one open.
    let _hold = keeper.hold(LeaseTarget::Instance("closing-probe".to_string()));

    keeper
        .shutdown_and_join(Duration::from_secs(60))
        .await
        .expect("the keeper thread must exit and close its connection within the window");
    assert!(
        !keeper.is_alive(),
        "shutdown_and_join must leave the keeper reporting dead"
    );

    let reopen_started = std::time::Instant::now();
    let reopened = Catalog::open(dir.path()).await.expect(
        "a fresh open on the same directory must succeed once the keeper's own \
         connection is closed",
    );
    let reopen_elapsed = reopen_started.elapsed();
    reopened.close().await;
    assert!(
        reopen_elapsed < Duration::from_secs(2),
        "the reopen took {reopen_elapsed:?} — the keeper's own connection must not still \
         be holding the process-exclusive lock (that would force this open to wait out \
         the 5 s busy timeout instead of landing almost immediately)"
    );

    // Idempotent: a second call finds no thread left to join.
    keeper
        .shutdown_and_join(Duration::from_secs(1))
        .await
        .expect("a second shutdown_and_join on an already-stopped keeper is Ok");
}

// ---------------------------------------------------------------------------
// OPS (#482) — the keeper under RELEASE: a Job hold registered after its row
// was released never re-arms the lease (the SQL `IS NOT NULL` guard, not a
// per-hold flag, is the guarantee), and `release_job_holds` releases
// `LeaseTarget::Job` holds only — an inline row's hold and a `ResultTable`
// hold are left alone.
// ---------------------------------------------------------------------------

macro_rules! skip_if_no_backend {
    ($backend:expr, $dir:expr) => {
        match jammi_test_utils::make_test_session($backend, $dir).await {
            Some(s) => s,
            None => {
                eprintln!("skipping {:?}: JAMMI_TEST_PG_URL unset", $backend);
                return;
            }
        }
    };
}

/// A hold registered AFTER the row's lease was released (the §3.4 2a helper
/// racing 2c's sweep, or a peer's stale registration) renews 0 rows: two
/// heartbeats later the lease is still NULL and the hold reads `lost`
/// through the zero-row renewal. Base: the heartbeat re-arms the NULLed
/// lease and the hold stays live.
#[test_case::test_case(jammi_db::catalog::backend::BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case::test_case(jammi_db::catalog::backend::BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_hold_registered_after_release_never_re_arms_the_lease(
    backend: jammi_db::catalog::backend::BackendKind,
) {
    let dir = tempdir().unwrap();
    let session = skip_if_no_backend!(backend, dir.path());
    let catalog = Arc::clone(session.catalog());
    let job_id = format!("rearm-{}", jammi_test_utils::unique_suffix());
    catalog
        .submit_job(SubmitJobParams {
            job_id: &job_id,
            kind: "embedding",
            execution: JobExecution::Queued,
            spec: "{}",
            model_ref: None,
            output_model_id: None,
            model_source: None,
            priority: 0,
        })
        .await
        .unwrap();
    let instance = format!("inst-{}", jammi_test_utils::unique_suffix());
    let claimed = catalog
        .claim_next(&instance, &["embedding"], Duration::from_secs(3))
        .await
        .unwrap()
        .expect("job claimed");
    assert_eq!(claimed.job_id, job_id);
    assert!(catalog
        .release_job_lease(&job_id, &instance, claimed.attempts)
        .await
        .unwrap());

    let keeper =
        crate::common::keeper_for_backend(backend, dir.path().to_path_buf(), fast_intervals())
            .await;
    let hold = keeper.hold(LeaseTarget::Job {
        job_id: job_id.clone(),
        instance_id: instance.clone(),
        attempts: claimed.attempts,
    });
    tokio::time::sleep(Duration::from_millis(2_200)).await;
    let row = catalog.get_job(&job_id).await.unwrap();
    assert!(
        row.lease_expires_at.is_none(),
        "two heartbeats after registering on a released row the lease is still NULL, \
         got {:?}",
        row.lease_expires_at
    );
    assert_eq!(row.releases, 1);
    assert!(
        hold.lost(),
        "the zero-row renewal flips the hold's lost flag"
    );
    drop(hold);
    keeper
        .shutdown_and_join(Duration::from_secs(10))
        .await
        .unwrap();
}

/// `release_job_holds` releases exactly the `LeaseTarget::Job` holds whose
/// rows are loop-claimed (`execution = 'queued'`): that row's lease goes
/// NULL and its hold reads lost; an inline row's Job hold is left alone
/// (`release_job_lease` returns `Ok(false)` on it) and keeps renewing; a
/// `ResultTable` hold is never touched by this call at all — its building
/// lease stays live and keeps renewing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_job_holds_flips_lost_and_skips_inline_holds() {
    use jammi_db::catalog::result_repo::{CreateResultTableParams, JobAttempt, ResultTableKind};

    let dir = tempdir().unwrap();
    let catalog = seeded_catalog(dir.path()).await;
    let job = |id: &'static str, execution: JobExecution| SubmitJobParams {
        job_id: id,
        kind: "fine_tune",
        execution,
        spec: "{}",
        model_ref: Some("keeper-base::1"),
        output_model_id: None,
        model_source: None,
        priority: 0,
    };
    catalog
        .submit_job(job("rjh-queued", JobExecution::Queued))
        .await
        .unwrap();
    catalog
        .submit_job(job("rjh-inline", JobExecution::Inline))
        .await
        .unwrap();
    let lease = Duration::from_secs(3);
    let queued = catalog
        .claim_next("me", &["fine_tune"], lease)
        .await
        .unwrap()
        .expect("queued claimed");
    let inline = catalog
        .claim_by_id("rjh-inline", "me", lease)
        .await
        .unwrap()
        .expect("inline claimed");
    catalog
        .create_result_table(CreateResultTableParams {
            table_name: "rjh_building",
            source_id: "src",
            model_id: "keeper-base",
            task: ModelTask::TextEmbedding,
            kind: ResultTableKind::Model,
            derived_from: None,
            parquet_path: "file:///tmp/rjh.parquet",
            dimensions: Some(4),
            key_column: None,
            text_columns: None,
            storage_precision: jammi_db::config::StoragePrecision::F32,
            oversample: 4,
            created_at: jammi_db::catalog::backend::now_sortable(),
            writer_id: Some("writer-rjh"),
            lease: Some(lease),
            job_attempt: Some(JobAttempt {
                job_id: "rjh-queued",
                instance_id: "me",
                attempts: queued.attempts,
            }),
        })
        .await
        .unwrap();

    let keeper = keeper_for(dir.path().to_path_buf(), fast_intervals()).await;
    let queued_hold = keeper.hold(LeaseTarget::Job {
        job_id: "rjh-queued".into(),
        instance_id: "me".into(),
        attempts: queued.attempts,
    });
    let inline_hold = keeper.hold(LeaseTarget::Job {
        job_id: "rjh-inline".into(),
        instance_id: "me".into(),
        attempts: inline.attempts,
    });
    let table_hold = keeper.hold(LeaseTarget::ResultTable {
        table: "rjh_building".into(),
        writer_id: "writer-rjh".into(),
    });
    let building_lease_before = catalog
        .get_result_table("rjh_building")
        .await
        .unwrap()
        .unwrap()
        .lease_expires_at
        .expect("building row leased");

    let released = keeper
        .release_job_holds(fast_intervals().heartbeat())
        .await
        .unwrap();
    assert_eq!(released, 1, "exactly the loop-claimed Job hold is released");

    let queued_row = catalog.get_job("rjh-queued").await.unwrap();
    assert!(queued_row.lease_expires_at.is_none());
    assert_eq!(queued_row.releases, 1);
    assert!(queued_hold.lost(), "the released hold reads lost at once");

    let inline_row = catalog.get_job("rjh-inline").await.unwrap();
    assert!(inline_row.lease_expires_at.is_some());
    assert_eq!(inline_row.releases, 0);
    assert!(!inline_hold.lost(), "an inline hold is untouched");
    assert!(!table_hold.lost(), "a ResultTable hold is untouched");
    assert!(catalog
        .get_result_table("rjh_building")
        .await
        .unwrap()
        .unwrap()
        .lease_expires_at
        .is_some());

    // Both untouched holds keep renewing across the next tick; the released
    // one is skipped and its lease stays NULL.
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    assert!(!inline_hold.lost());
    assert!(!table_hold.lost());
    let building_lease_after = catalog
        .get_result_table("rjh_building")
        .await
        .unwrap()
        .unwrap()
        .lease_expires_at
        .expect("building row still leased");
    assert!(
        building_lease_after > building_lease_before,
        "the ResultTable hold's renewal keeps landing"
    );
    assert!(catalog
        .get_job("rjh-queued")
        .await
        .unwrap()
        .lease_expires_at
        .is_none());

    drop(queued_hold);
    drop(inline_hold);
    drop(table_hold);
    keeper
        .shutdown_and_join(Duration::from_secs(10))
        .await
        .unwrap();
}
