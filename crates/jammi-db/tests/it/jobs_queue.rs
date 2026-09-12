//! Kind-agnostic job-queue primitives on the `jobs`/`instances`/`workers`
//! catalog tables (migration 029): atomic claim (queued and inline), lease
//! heartbeat, the full attempt guard, expired-lease/dead-instance reclaim,
//! and the esc-075 acceleration-report pending/retirement lifecycle.
//! Supersedes the former `training_jobs`-specific `fine_tune_queue.rs`,
//! whose claim/heartbeat/reclaim/finalize/acceleration-report coverage this
//! file carries forward against the generalised table — including the
//! atomic finish-with-model-registration-and-epoch-checkpoints machinery
//! [`Catalog::finish_job_with_model`] (the C1a design-gap this unit closes;
//! see the handoff doc's "Resolved" section).
//!
//! Every test is parameterised over [`BackendKind`] via `test_case` +
//! `cfg_attr`. The SQLite lane is always generated; the Postgres lane is
//! generated only when the `live-postgres-tests` feature is on, and skips at
//! runtime when `JAMMI_TEST_PG_URL` is unset. The Postgres lane exercises the
//! `FOR UPDATE SKIP LOCKED` claim path and the global expired-lease reclaim
//! scan that the SQLite serialized-UPDATE path cannot.
//!
//! The claim and reclaim queries scan `jobs` globally (not tenant- or
//! id-scoped). On the Postgres lane that table is shared across the whole
//! test run, so each test first clears it via [`reset_queue`]. CI's
//! `test-pg` job runs the Postgres lane with `--test-threads=1`, so the
//! reset-then-populate sequence is serialised and cannot race a sibling test.

use std::sync::Arc;
use std::time::Duration;

use jammi_db::catalog::backend::{BackendKind, SqlValue, TxOptions};
use jammi_db::catalog::jobs_repo::{
    EpochCheckpointRow, FinishJobParams, FinishJobWithModelParams, SubmitJobParams, WorkerState,
};
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::result_repo::{CreateResultTableParams, JobAttempt, ResultTableKind};
use jammi_db::catalog::status::{JobExecution, JobStatus};
use jammi_db::catalog::Catalog;
use jammi_db::config::StoragePrecision;
use jammi_db::model_task::ModelTask;
use jammi_test_utils::make_test_session;
use tempfile::tempdir;
use test_case::test_case;

/// SAFETY note: the Postgres lane returns `None` when `JAMMI_TEST_PG_URL`
/// is unset so the test can early-return rather than `#[ignore]`'ing
/// (CLAUDE.md forbids `#[ignore]`).
macro_rules! skip_if_no_backend {
    ($backend:expr, $dir:expr) => {
        match make_test_session($backend, $dir).await {
            Some(s) => s,
            None => {
                eprintln!("skipping {:?}: JAMMI_TEST_PG_URL unset", $backend);
                return;
            }
        }
    };
}

const KIND: &str = "fine_tune";
const KINDS: &[&str] = &[KIND];

/// A minimal queued job over the `q-base` model with the given id.
fn job_params(job_id: &str) -> SubmitJobParams<'_> {
    SubmitJobParams {
        job_id,
        kind: KIND,
        execution: JobExecution::Queued,
        spec: "{}",
        model_ref: Some("q-base::1"),
        output_model_id: None,
        model_source: None,
        priority: 0,
    }
}

/// A minimal INLINE job over the `q-base` model with the given id — never
/// selected by [`Catalog::claim_next`].
fn inline_job_params(job_id: &str) -> SubmitJobParams<'_> {
    SubmitJobParams {
        execution: JobExecution::Inline,
        ..job_params(job_id)
    }
}

/// A per-run unique suffix for a test's row ids. The Postgres lane shares
/// one catalog across runs, and `reset_queue` clears `jobs`/`instances`/
/// `workers` but NOT `models`/`result_tables` — so a fixed model or table
/// name (`jammi:fine-tuned:fz`, `rt-1-zombie-table`) would collide with
/// the previous run's leftover row on the second run against the same
/// database. Every test that registers a model or a result table names it
/// through this.
fn run_suffix() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..8].to_string()
}

/// Register the FK target model `q-base` once per test catalog.
async fn register_base_model(catalog: &Catalog) {
    catalog
        .register_model(RegisterModelParams {
            model_id: "q-base",
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
}

/// Clear every row from `jobs`/`instances`/`workers` so the global
/// claim/reclaim scans see only the rows this test creates. Needed because
/// the Postgres lane shares one catalog DB across the run; the SQLite lane
/// has a fresh tempdir per test but running the reset there too keeps both
/// lanes on one path. Run under `--test-threads=1` on the Postgres lane, so
/// it cannot race a sibling test.
async fn reset_queue(catalog: &Catalog) {
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute("DELETE FROM jobs", &[]).await?;
                tx.execute("DELETE FROM workers", &[]).await?;
                tx.execute("DELETE FROM instances", &[]).await?;
                Ok(())
            })
        })
        .await
        .unwrap();
}

/// Force `lease_expires_at = NULL` on one row via raw SQL — the state a
/// `running` row can never actually reach through this engine's own claim
/// path ([`Catalog::claim_next`] always stamps a lease), but a fixture must
/// still be able to MANUFACTURE it directly to pin the reclaim contract
/// against the state rather than against how a row got there.
async fn force_null_lease(catalog: &Catalog, job_id: &str) {
    let job_id = job_id.to_string();
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE jobs SET lease_expires_at = NULL WHERE job_id = $1",
                    &[SqlValue::TextOwned(job_id)],
                )
                .await
            })
        })
        .await
        .unwrap();
}

/// Set `priority`/`claimable` on one row via raw SQL — the claim-policy
/// columns are catalog data, so the fixture writes them as data rather than
/// through any typed setter (there is none; the engine only honors the
/// columns).
async fn set_claim_policy(catalog: &Catalog, job_id: &str, priority: i64, claimable: bool) {
    let job_id = job_id.to_string();
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE jobs SET priority = $1, claimable = $2 WHERE job_id = $3",
                    &[
                        SqlValue::Int(priority),
                        SqlValue::Bool(claimable),
                        SqlValue::TextOwned(job_id),
                    ],
                )
                .await
            })
        })
        .await
        .unwrap();
}

/// Force `instances.last_seen_at` into the past via raw SQL — the state a
/// dead process leaves behind (nothing heartbeats it any more).
async fn force_stale_instance(catalog: &Catalog, instance_id: &str, days_ago: i64) {
    let cutoff = (chrono::Utc::now() - chrono::Duration::days(days_ago))
        .format("%Y-%m-%dT%H:%M:%S%.9fZ")
        .to_string();
    let instance_id = instance_id.to_string();
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE instances SET last_seen_at = $1 WHERE instance_id = $2",
                    &[
                        SqlValue::TextOwned(cutoff),
                        SqlValue::TextOwned(instance_id),
                    ],
                )
                .await
            })
        })
        .await
        .unwrap();
}

/// Open a backend-parameterised catalog with the FK base model registered and
/// an empty queue. Returns `None` to signal the caller should skip (Postgres
/// without `JAMMI_TEST_PG_URL`).
macro_rules! queue_catalog {
    ($backend:expr, $dir:expr) => {{
        let session = skip_if_no_backend!($backend, $dir);
        let catalog = Arc::clone(session.catalog());
        reset_queue(&catalog).await;
        register_base_model(&catalog).await;
        (session, catalog)
    }};
}

/// Two concurrent claims against a single queued job run on separate tasks of
/// a multi-thread runtime: exactly one wins, the other sees an empty queue.
/// The winner's record is `running`, leased to it, and `attempts` is
/// incremented to one. Spawning the claims as distinct tasks (rather than
/// `tokio::join!`, which interleaves two futures on one task
/// deterministically) puts the Postgres `FOR UPDATE SKIP LOCKED` path under
/// real lock contention.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_worker_claim_exclusivity_grants_one_winner(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.submit_job(job_params("q-1")).await.unwrap();

    let c1 = Arc::clone(&catalog);
    let c2 = Arc::clone(&catalog);
    let lease = Duration::from_secs(30);
    let h1 = tokio::spawn(async move { c1.claim_next("worker-a", KINDS, lease).await.unwrap() });
    let h2 = tokio::spawn(async move { c2.claim_next("worker-b", KINDS, lease).await.unwrap() });
    let a = h1.await.unwrap();
    let b = h2.await.unwrap();

    let winners: Vec<_> = [a, b].into_iter().flatten().collect();
    assert_eq!(
        winners.len(),
        1,
        "exactly one concurrent claim must win the single queued job"
    );
    let claimed = &winners[0];
    assert_eq!(claimed.job_id, "q-1");
    assert_eq!(claimed.status, JobStatus::Running.to_string());
    assert!(
        matches!(claimed.claimed_by.as_deref(), Some("worker-a" | "worker-b")),
        "claimed_by must name the winning worker, got {:?}",
        claimed.claimed_by
    );
    assert!(claimed.lease_expires_at.is_some(), "lease must be stamped");
    assert_eq!(claimed.attempts, 1, "first claim sets attempts to 1");

    // The queue is now empty: a third claim returns None.
    let empty = catalog.claim_next("worker-c", KINDS, lease).await.unwrap();
    assert!(empty.is_none(), "no queued job remains after the claim");
}

/// An `execution = 'inline'` row is NEVER selected by `claim_next`, no matter
/// how it got to `queued` — the poll loop's own `WHERE execution = 'queued'`
/// predicate excludes it entirely. `claim_by_id` claims it directly.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn inline_rows_are_never_returned_by_claim_next(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.submit_job(inline_job_params("in-1")).await.unwrap();
    catalog.submit_job(job_params("q-1")).await.unwrap();

    let lease = Duration::from_secs(30);
    let claimed = catalog
        .claim_next("worker", KINDS, lease)
        .await
        .unwrap()
        .expect("the queued row is claimable");
    assert_eq!(
        claimed.job_id, "q-1",
        "claim_next must never select the inline row even though it is present and queued"
    );
    assert!(
        catalog
            .claim_next("worker", KINDS, lease)
            .await
            .unwrap()
            .is_none(),
        "the inline row must stay invisible to claim_next with no other queued row left"
    );

    // The inline row is still claimable — by id.
    let by_id = catalog
        .claim_by_id("in-1", "worker", lease)
        .await
        .unwrap()
        .expect("claim_by_id claims the inline row directly");
    assert_eq!(by_id.job_id, "in-1");
    assert_eq!(by_id.status, JobStatus::Running.to_string());
}

/// Claims hand out the oldest queued job first (FIFO by `created_at`).
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn claim_returns_oldest_queued_job_first(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.submit_job(job_params("old")).await.unwrap();
    // A distinct, strictly-later created_at so ORDER BY is unambiguous.
    tokio::time::sleep(Duration::from_millis(1100)).await;
    catalog.submit_job(job_params("new")).await.unwrap();

    let first = catalog
        .claim_next("w", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .expect("a job is queued");
    assert_eq!(first.job_id, "old", "oldest queued job is claimed first");
}

/// `priority` breaks the tie ahead of `created_at`.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn claim_honors_priority_over_created_at(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.submit_job(job_params("low")).await.unwrap();
    tokio::time::sleep(Duration::from_millis(1100)).await;
    catalog.submit_job(job_params("high")).await.unwrap();
    set_claim_policy(&catalog, "high", 10, true).await;

    let first = catalog
        .claim_next("w", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .expect("a job is queued");
    assert_eq!(
        first.job_id, "high",
        "the higher-priority job claims first despite being strictly younger"
    );

    let second = catalog
        .claim_next("w", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .expect("a job is queued");
    assert_eq!(
        second.job_id, "low",
        "the default-priority job claims once the higher-priority job is gone"
    );
}

/// A `claimable = FALSE` job is skipped by the claim, not errored, and stays
/// invisible until flipped back.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn claim_skips_a_held_job_without_erroring(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.submit_job(job_params("held")).await.unwrap();
    tokio::time::sleep(Duration::from_millis(1100)).await;
    catalog.submit_job(job_params("ready")).await.unwrap();
    set_claim_policy(&catalog, "held", 0, false).await;

    let first = catalog
        .claim_next("w", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .expect("the claimable job is queued");
    assert_eq!(
        first.job_id, "ready",
        "the older held job is skipped, not claimed and not errored"
    );

    let none = catalog
        .claim_next("w", KINDS, Duration::from_secs(30))
        .await
        .unwrap();
    assert!(
        none.is_none(),
        "the held job stays invisible to the claim while claimable = FALSE"
    );

    set_claim_policy(&catalog, "held", 0, true).await;
    let released = catalog
        .claim_next("w", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .expect("the released job is claimable again");
    assert_eq!(released.job_id, "held");
    assert_eq!(
        released.status,
        JobStatus::Running.to_string(),
        "the hold is released with no other status change along the way"
    );
}

/// A job re-queued by [`Catalog::reclaim_expired_jobs`] retains its
/// `priority`/`claimable` values and re-enters the claim ordering accordingly.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reclaimed_job_retains_priority_and_reenters_ordering(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.submit_job(job_params("sibling")).await.unwrap();
    tokio::time::sleep(Duration::from_millis(1100)).await;
    catalog.submit_job(job_params("hi")).await.unwrap();
    set_claim_policy(&catalog, "hi", 10, true).await;

    let claimed = catalog
        .claim_next("worker", KINDS, Duration::from_secs(0))
        .await
        .unwrap()
        .expect("the high-priority job claims first despite being younger");
    assert_eq!(claimed.job_id, "hi");
    let actioned = catalog
        .reclaim_expired_jobs(Duration::from_secs(0), 5)
        .await
        .unwrap();
    assert_eq!(actioned, 1, "the expired lease is re-queued");

    let next = catalog
        .claim_next("worker", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .expect("the re-queued job re-enters the ordering");
    assert_eq!(
        next.job_id, "hi",
        "reclaim preserves priority: the high-priority job claims again ahead of its \
         strictly-older sibling"
    );
}

/// Two concurrent claims against two jobs of differing priority each win a
/// distinct job.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_claim_composes_with_priority_ordering(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.submit_job(job_params("low")).await.unwrap();
    catalog.submit_job(job_params("high")).await.unwrap();
    set_claim_policy(&catalog, "high", 10, true).await;

    let c1 = Arc::clone(&catalog);
    let c2 = Arc::clone(&catalog);
    let lease = Duration::from_secs(30);
    let h1 = tokio::spawn(async move { c1.claim_next("worker-a", KINDS, lease).await.unwrap() });
    let h2 = tokio::spawn(async move { c2.claim_next("worker-b", KINDS, lease).await.unwrap() });
    let a = h1.await.unwrap();
    let b = h2.await.unwrap();

    let mut winners: Vec<String> = [a, b].into_iter().flatten().map(|r| r.job_id).collect();
    winners.sort();
    assert_eq!(
        winners,
        vec!["high".to_string(), "low".to_string()],
        "each concurrent claim wins a distinct job, spanning the priority ordering"
    );
}

/// Heartbeat renews the lease for the owning instance and refuses everyone
/// else, keyed on the FULL attempt guard (`claimed_by` AND `attempts`).
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn heartbeat_renews_for_owner_only(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.submit_job(job_params("hb")).await.unwrap();
    let claimed = catalog
        .claim_next("owner", KINDS, Duration::from_secs(5))
        .await
        .unwrap()
        .expect("job claimed");
    let first_lease = claimed.lease_expires_at.clone().unwrap();
    assert_eq!(claimed.attempts, 1);

    // A non-owner (wrong instance id) cannot renew the lease.
    let stolen = catalog
        .heartbeat_job("hb", "intruder", 1, Duration::from_secs(60))
        .await
        .unwrap();
    assert!(!stolen, "a non-owner must not renew the lease");

    // A stale attempt number cannot renew the lease either.
    let stale_attempt = catalog
        .heartbeat_job("hb", "owner", 99, Duration::from_secs(60))
        .await
        .unwrap();
    assert!(
        !stale_attempt,
        "a stale attempt number must not renew the lease"
    );

    // The owner renews with the correct attempt; the new deadline is later.
    tokio::time::sleep(Duration::from_millis(10)).await;
    let renewed = catalog
        .heartbeat_job("hb", "owner", 1, Duration::from_secs(120))
        .await
        .unwrap();
    assert!(renewed, "the owner renews its own lease");
    let after = catalog.get_job("hb").await.unwrap();
    assert!(
        after.lease_expires_at.as_deref().unwrap() > first_lease.as_str(),
        "renewed lease must extend past the original deadline"
    );

    // Once the job leaves `running`, even the owner cannot heartbeat it.
    let failed = catalog.fail_job("hb", "owner", 1, "boom").await.unwrap();
    assert!(failed, "the owner ends its own running job");
    let post_complete = catalog
        .heartbeat_job("hb", "owner", 1, Duration::from_secs(60))
        .await
        .unwrap();
    assert!(
        !post_complete,
        "a job that is not running cannot be heartbeat"
    );
}

/// The attempt guard also covers `finish_job` and `progress_job`: a
/// stale-attempt caller (a zombie of a prior attempt this job already moved
/// past via reclaim) matches zero rows on either.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn attempts_guard_covers_finish_and_progress(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.submit_job(job_params("ag")).await.unwrap();
    // Claim, expire (zero lease), and reclaim: attempts goes 0 -> 1 -> back to
    // `queued`. Claim again: attempts -> 2. The zombie from attempt 1 now
    // presents a stale `attempts = 1`.
    catalog
        .claim_next("zombie", KINDS, Duration::from_secs(0))
        .await
        .unwrap()
        .expect("first claim");
    catalog
        .reclaim_expired_jobs(Duration::from_secs(0), 5)
        .await
        .unwrap();
    let current = catalog
        .claim_next("live", KINDS, Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("second claim");
    assert_eq!(current.attempts, 2);

    let stale_progress = catalog
        .progress_job("ag", "zombie", 1, Some(10), None, None)
        .await
        .unwrap();
    assert!(
        !stale_progress,
        "the zombie's stale attempt must not record progress"
    );

    let stale_finish = catalog
        .finish_job(FinishJobParams {
            job_id: "ag",
            instance_id: "zombie",
            attempts: 1,
            result: "{}",
        })
        .await
        .unwrap();
    assert!(
        !stale_finish,
        "the zombie's stale attempt must not finish the job"
    );

    // The live claimant, at the current attempt, succeeds on both.
    assert!(catalog
        .progress_job("ag", "live", 2, Some(5), Some(10), Some("embedding"))
        .await
        .unwrap());
    assert!(catalog
        .finish_job(FinishJobParams {
            job_id: "ag",
            instance_id: "live",
            attempts: 2,
            result: "{}",
        })
        .await
        .unwrap());
    let done = catalog.get_job("ag").await.unwrap();
    assert_eq!(done.status, JobStatus::Completed.to_string());
}

/// An expired lease with attempts left re-queues the job; once attempts are
/// exhausted the job fails with the reason recorded.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reclaim_requeues_then_fails_when_attempts_exhausted(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.submit_job(job_params("rc")).await.unwrap();

    let claimed = catalog
        .claim_next("worker", KINDS, Duration::from_secs(0))
        .await
        .unwrap()
        .expect("job claimed");
    assert_eq!(claimed.attempts, 1);

    let actioned = catalog
        .reclaim_expired_jobs(Duration::from_secs(0), 2)
        .await
        .unwrap();
    assert_eq!(actioned, 1, "the one expired lease is re-queued");
    let requeued = catalog.get_job("rc").await.unwrap();
    assert_eq!(requeued.status, JobStatus::Queued.to_string());
    assert!(requeued.claimed_by.is_none(), "re-queue clears claimed_by");
    assert!(
        requeued.lease_expires_at.is_none(),
        "re-queue clears the lease deadline"
    );

    let reclaimed = catalog
        .claim_next("worker", KINDS, Duration::from_secs(0))
        .await
        .unwrap()
        .expect("re-queued job claimable");
    assert_eq!(reclaimed.attempts, 2);

    let actioned = catalog
        .reclaim_expired_jobs(Duration::from_secs(0), 2)
        .await
        .unwrap();
    assert_eq!(actioned, 1, "the exhausted lease is actioned once");
    let failed = catalog.get_job("rc").await.unwrap();
    assert_eq!(failed.status, JobStatus::Failed.to_string());
    assert!(
        failed.lease_expires_at.is_none(),
        "a failed job carries no live lease"
    );
    assert!(
        failed
            .error
            .as_deref()
            .is_some_and(|m| m.contains("lease expired")),
        "the failure records the lease-exhaustion reason, got {:?}",
        failed.error
    );
}

/// `fail_job` is a lease-guarded compare-and-set: only the instance that
/// still holds the lease (`claimed_by` + `attempts`) fails the job.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fail_is_a_lease_guarded_compare_and_set(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.submit_job(job_params("fl")).await.unwrap();

    catalog
        .claim_next("worker-a", KINDS, Duration::from_secs(0))
        .await
        .unwrap()
        .expect("worker-a claims");
    catalog
        .reclaim_expired_jobs(Duration::from_secs(0), 5)
        .await
        .unwrap();
    catalog
        .claim_next("worker-b", KINDS, Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("worker-b re-claims");

    // The stale worker-a (attempt 1) cannot fail the job worker-b (attempt
    // 2) now owns.
    let a_failed = catalog.fail_job("fl", "worker-a", 1, "boom").await.unwrap();
    assert!(
        !a_failed,
        "a worker that lost its lease cannot fail the job"
    );
    let after_a = catalog.get_job("fl").await.unwrap();
    assert_eq!(
        after_a.status,
        JobStatus::Running.to_string(),
        "the stale fail leaves the job running for its real owner"
    );

    let b_failed = catalog
        .fail_job("fl", "worker-b", 2, "real failure")
        .await
        .unwrap();
    assert!(b_failed, "the lease owner records the failure");
    let after_b = catalog.get_job("fl").await.unwrap();
    assert_eq!(after_b.status, JobStatus::Failed.to_string());
    assert_eq!(after_b.error.as_deref(), Some("real failure"));
}

/// A live (unexpired) lease is left untouched by reclaim.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reclaim_leaves_live_leases_untouched(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.submit_job(job_params("live")).await.unwrap();
    catalog
        .claim_next("worker", KINDS, Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("job claimed");

    let actioned = catalog
        .reclaim_expired_jobs(Duration::from_secs(3600), 5)
        .await
        .unwrap();
    assert_eq!(actioned, 0, "a live lease is not reclaimed");
    let still_running = catalog.get_job("live").await.unwrap();
    assert_eq!(still_running.status, JobStatus::Running.to_string());
}

/// PINS the claim side of the one-primitive reclaim rule ("absent or expired
/// is reclaimable"): [`Catalog::claim_next`] ALWAYS stamps `lease_expires_at`
/// on the row it hands back `running`.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn claim_always_stamps_a_non_null_lease(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.submit_job(job_params("stamped")).await.unwrap();
    let claimed = catalog
        .claim_next("worker", KINDS, Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("job claimed");
    assert_eq!(claimed.status, JobStatus::Running.to_string());
    assert!(
        claimed.lease_expires_at.is_some(),
        "every claim must stamp a non-NULL lease on the row it returns running"
    );

    let row = catalog.get_job("stamped").await.unwrap();
    assert_eq!(row.status, JobStatus::Running.to_string());
    assert!(
        row.lease_expires_at.is_some(),
        "the persisted row must carry the same non-NULL lease the claim returned"
    );
}

/// PINS the reclaim side of the same one-primitive rule: a `running` row
/// whose `lease_expires_at` is NULL is reclaimed exactly like an expired
/// (non-NULL, past) lease.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_running_row_with_a_null_lease_is_reclaimed(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.submit_job(job_params("nl")).await.unwrap();
    catalog
        .claim_next("worker", KINDS, Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("job claimed");
    force_null_lease(&catalog, "nl").await;

    let actioned = catalog
        .reclaim_expired_jobs(Duration::from_secs(3600), 5)
        .await
        .unwrap();
    assert_eq!(
        actioned, 1,
        "a NULL lease is reclaimed exactly like an expired one"
    );
    let requeued = catalog.get_job("nl").await.unwrap();
    assert_eq!(requeued.status, JobStatus::Queued.to_string());
}

// ─── liveness reclaim: inline-execution rows, gated on the owning instance ──

/// A stale (dead) owning `instances` row: the inline job is failed, never
/// requeued (there is no poll loop that could re-claim it).
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn inline_job_failed_when_owning_instance_is_stale(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog
        .submit_job(inline_job_params("dead-owner"))
        .await
        .unwrap();
    let lease = Duration::from_secs(3600);
    catalog
        .claim_by_id("dead-owner", "instance-dead", lease)
        .await
        .unwrap()
        .expect("inline job claimed");
    catalog
        .upsert_instance("instance-dead", None, None)
        .await
        .unwrap();
    force_stale_instance(&catalog, "instance-dead", 3650).await;

    let actioned = catalog.reclaim_expired_jobs(lease, 5).await.unwrap();
    assert_eq!(actioned, 1, "the inline job with a dead owner is actioned");
    let row = catalog.get_job("dead-owner").await.unwrap();
    assert_eq!(
        row.status,
        JobStatus::Failed.to_string(),
        "a stale-instance inline job is failed, never requeued"
    );
    assert!(
        row.error.as_deref().is_some_and(|m| m.contains("inline")),
        "got {:?}",
        row.error
    );
}

/// An ABSENT owning instance row (never upserted at all) is treated
/// identically to a stale one — "absent or stale" is one predicate.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn inline_job_failed_when_owning_instance_is_absent(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog
        .submit_job(inline_job_params("no-owner"))
        .await
        .unwrap();
    let lease = Duration::from_secs(3600);
    catalog
        .claim_by_id("no-owner", "instance-ghost", lease)
        .await
        .unwrap()
        .expect("inline job claimed");
    // No `upsert_instance` call at all: the instance row never existed.

    let actioned = catalog.reclaim_expired_jobs(lease, 5).await.unwrap();
    assert_eq!(actioned, 1);
    let row = catalog.get_job("no-owner").await.unwrap();
    assert_eq!(row.status, JobStatus::Failed.to_string());
}

/// A LIVE owning instance leaves the inline job entirely untouched.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn inline_job_untouched_when_owning_instance_is_live(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog
        .submit_job(inline_job_params("live-owner"))
        .await
        .unwrap();
    let lease = Duration::from_secs(3600);
    catalog
        .claim_by_id("live-owner", "instance-live", lease)
        .await
        .unwrap()
        .expect("inline job claimed");
    catalog
        .upsert_instance("instance-live", None, None)
        .await
        .unwrap();
    // Fresh `last_seen_at` (just upserted): well within the liveness margin.

    let actioned = catalog.reclaim_expired_jobs(lease, 5).await.unwrap();
    assert_eq!(
        actioned, 0,
        "a live owning instance means nothing is reclaimed"
    );
    let row = catalog.get_job("live-owner").await.unwrap();
    assert_eq!(row.status, JobStatus::Running.to_string());
}

/// Two processes sharing one operator LABEL are still two instances: the
/// `instances` row is keyed by the per-process id, never by the label, so
/// a dead process's inline job is failed by the inline-liveness reclaim
/// arm while a live peer carrying the SAME label (a fleet's stable
/// `JAMMI_WORKER_ID` across restarts — the replacement process, or a
/// sibling replica) keeps its own inline job untouched. The session-level
/// half of this contract — two sessions given one `JAMMI_WORKER_ID` mint
/// two ids and share the label — is pinned in `jammi-ai`'s
/// `instance_identity` suite; this is the catalog consequence.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dead_instances_inline_job_is_failed_while_a_live_peer_with_the_same_label_keeps_its_own(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());
    let lease = Duration::from_secs(3600);
    let label = "gpu-node-7";

    catalog
        .submit_job(inline_job_params("label-dead-job"))
        .await
        .unwrap();
    catalog
        .submit_job(inline_job_params("label-live-job"))
        .await
        .unwrap();
    catalog
        .claim_by_id("label-dead-job", "instance-dead-uuid", lease)
        .await
        .unwrap()
        .expect("the dead process's inline job was claimed while it lived");
    catalog
        .claim_by_id("label-live-job", "instance-live-uuid", lease)
        .await
        .unwrap()
        .expect("the live peer's inline job is claimed");
    catalog
        .upsert_instance("instance-dead-uuid", Some(label), None)
        .await
        .unwrap();
    catalog
        .upsert_instance("instance-live-uuid", Some(label), None)
        .await
        .unwrap();
    force_stale_instance(&catalog, "instance-dead-uuid", 3650).await;

    let actioned = catalog.reclaim_expired_jobs(lease, 5).await.unwrap();
    assert_eq!(
        actioned, 1,
        "exactly the dead instance's inline job is actioned; the label it shares with a \
         live peer must not keep it alive"
    );
    let dead = catalog.get_job("label-dead-job").await.unwrap();
    assert_eq!(dead.status, JobStatus::Failed.to_string());
    assert_eq!(dead.error.as_deref(), Some("inline executor died"));
    let live = catalog.get_job("label-live-job").await.unwrap();
    assert_eq!(
        live.status,
        JobStatus::Running.to_string(),
        "the live peer's own inline job is untouched"
    );
}

// ─── retention sweep ─────────────────────────────────────────────────────

/// `prune_jobs` deletes only TERMINAL rows past the retention window — a
/// young terminal row and any non-terminal row (however old) survive.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prune_jobs_deletes_only_terminal_rows_past_the_window(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog
        .submit_job(job_params("old-terminal"))
        .await
        .unwrap();
    catalog
        .submit_job(job_params("young-terminal"))
        .await
        .unwrap();
    catalog.submit_job(job_params("old-running")).await.unwrap();
    force_job_status_and_age(&catalog, "old-terminal", "completed", 31).await;
    force_job_status_and_age(&catalog, "young-terminal", "failed", 1).await;
    force_job_status_and_age(&catalog, "old-running", "running", 31).await;

    let deleted = catalog
        .prune_jobs(Duration::from_secs(30 * 86_400))
        .await
        .unwrap();
    assert_eq!(deleted, 1, "only the old terminal row is pruned");
    assert!(catalog.get_job("old-terminal").await.is_err());
    assert!(catalog.get_job("young-terminal").await.is_ok());
    assert!(catalog.get_job("old-running").await.is_ok());
}

// ─── F3: `submit_job_deduped`'s durable per-tenant idempotency key (migration
// 030) ───────────────────────────────────────────────────────────────────

/// RED before migration 030 + `submit_job_deduped`: there was no
/// `jobs.idempotency_key` column and no way to dedupe a retry durably (the
/// old dedupe lived in a process `HashMap`, forgotten across a restart and
/// racy under two concurrent identical submissions).
///
/// Sequential retry: a second `submit_job_deduped` call carrying the SAME
/// non-empty key as a still-known prior submission returns THAT prior job's
/// id, never its own — the row of record is unchanged, not duplicated.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn submit_job_deduped_sequential_retry_returns_the_same_job_id(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    let first = catalog
        .submit_job_deduped(job_params("dedupe-first"), Some("retry-key-1"))
        .await
        .unwrap();
    assert_eq!(first, "dedupe-first", "the first call's own id wins");

    let second = catalog
        .submit_job_deduped(job_params("dedupe-second"), Some("retry-key-1"))
        .await
        .unwrap();
    assert_eq!(
        second, first,
        "a retry carrying the same key must return the FIRST call's job id, \
         never submit a second row"
    );

    let all = catalog.list_jobs().await.unwrap();
    assert_eq!(all.iter().filter(|j| j.job_id == "dedupe-first").count(), 1);
    assert!(
        all.iter().all(|j| j.job_id != "dedupe-second"),
        "the second call's own job_id must never have been inserted"
    );
}

/// Concurrent retry: two `submit_job_deduped` calls racing the SAME key via
/// `tokio::join!` land exactly one row — the atomic `INSERT ... ON CONFLICT
/// ... DO NOTHING RETURNING` compare-and-set closes the lookup-then-insert
/// race window a separate reservation step could not.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn submit_job_deduped_concurrent_retry_produces_exactly_one_row(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    let cat_1 = Arc::clone(&catalog);
    let cat_2 = Arc::clone(&catalog);
    let (r1, r2) = tokio::join!(
        cat_1.submit_job_deduped(job_params("dedupe-race-a"), Some("race-key")),
        cat_2.submit_job_deduped(job_params("dedupe-race-b"), Some("race-key")),
    );
    let (id1, id2) = (r1.unwrap(), r2.unwrap());
    assert_eq!(
        id1, id2,
        "both concurrent racers must agree on ONE winning job id"
    );
    assert!(id1 == "dedupe-race-a" || id1 == "dedupe-race-b");

    let all = catalog.list_jobs().await.unwrap();
    let race_rows: Vec<_> = all
        .iter()
        .filter(|j| j.job_id == "dedupe-race-a" || j.job_id == "dedupe-race-b")
        .collect();
    assert_eq!(
        race_rows.len(),
        1,
        "exactly one row must exist for the raced key, never two: {race_rows:?}"
    );
}

/// A pruned row does not resurrect an identity for its key — there is no
/// separate map to resurrect. Submitting with a key, pruning that row away,
/// then retrying with the SAME key submits a genuinely fresh row rather than
/// erroring or reattaching to the deleted id.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn submit_job_deduped_after_the_row_is_pruned_submits_fresh(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    let first = catalog
        .submit_job_deduped(job_params("dedupe-pruned-first"), Some("prune-key"))
        .await
        .unwrap();
    assert_eq!(first, "dedupe-pruned-first");

    force_job_status_and_age(&catalog, &first, "completed", 31).await;
    let deleted = catalog
        .prune_jobs(Duration::from_secs(30 * 86_400))
        .await
        .unwrap();
    assert_eq!(deleted, 1, "the first row must be pruned away");
    assert!(catalog.get_job(&first).await.is_err());

    let second = catalog
        .submit_job_deduped(job_params("dedupe-pruned-second"), Some("prune-key"))
        .await
        .unwrap();
    assert_eq!(
        second, "dedupe-pruned-second",
        "the key's prior row is gone, so a retry submits fresh under its OWN id"
    );
}

/// Different tenants may reuse the same idempotency key: it is scoped by
/// `(tenant, key)`, not `key` alone.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn submit_job_deduped_different_tenants_may_reuse_a_key(backend: BackendKind) {
    use std::str::FromStr;

    use jammi_db::TenantId;

    let dir = tempdir().unwrap();
    let session = skip_if_no_backend!(backend, dir.path());
    let base = Arc::clone(session.catalog());
    reset_queue(&base).await;
    register_base_model(&base).await;

    let tenant_a = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e91").unwrap();
    let tenant_b = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e92").unwrap();
    let cat_a = base.pinned_to_tenant(Some(tenant_a));
    let cat_b = base.pinned_to_tenant(Some(tenant_b));

    let id_a = cat_a
        .submit_job_deduped(job_params("dedupe-tenant-a"), Some("shared-key"))
        .await
        .unwrap();
    let id_b = cat_b
        .submit_job_deduped(job_params("dedupe-tenant-b"), Some("shared-key"))
        .await
        .unwrap();
    assert_eq!(id_a, "dedupe-tenant-a");
    assert_eq!(
        id_b, "dedupe-tenant-b",
        "tenant B's submission under the SAME key must be its OWN fresh row, \
         never tenant A's"
    );
}

/// Pins `MAX_IDEMPOTENCY_KEY_BYTES`: an unbounded `idempotency_key` would be
/// accepted verbatim by SQLite (which has no index-row-size ceiling) but
/// fail on Postgres -- `index row size 5136 exceeds btree version 4 maximum
/// 2704` -- so the SAME input would diverge silently across backends absent
/// this bound. `submit_job_deduped` refuses a key one byte over the bound
/// identically on BOTH backends, with a typed `JammiError::Config` naming
/// the bound (never the key's own value).
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn submit_job_deduped_refuses_a_key_one_byte_over_the_bound(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    let oversize_key = "x".repeat(jammi_db::catalog::jobs_repo::MAX_IDEMPOTENCY_KEY_BYTES + 1);
    let err = catalog
        .submit_job_deduped(job_params("dedupe-oversize-key"), Some(&oversize_key))
        .await
        .expect_err("a key one byte over MAX_IDEMPOTENCY_KEY_BYTES must be refused");
    match err {
        jammi_db::error::JammiError::Config(message) => {
            assert!(
                message.contains("MAX_IDEMPOTENCY_KEY_BYTES"),
                "the refusal must name the bound, not the key's own value: {message}"
            );
            assert!(
                !message.contains(&oversize_key),
                "the refusal must never echo the oversize key's own value: {message}"
            );
        }
        other => panic!("expected JammiError::Config naming the bound, got {other:?}"),
    }
    assert!(
        catalog.list_jobs().await.unwrap().is_empty(),
        "a refused submission must never insert a row"
    );
}

/// A key exactly AT the bound is accepted identically on both backends --
/// the bound is inclusive, not off-by-one in either direction.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn submit_job_deduped_accepts_a_key_exactly_at_the_bound(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    let at_bound_key = "x".repeat(jammi_db::catalog::jobs_repo::MAX_IDEMPOTENCY_KEY_BYTES);
    let id = catalog
        .submit_job_deduped(job_params("dedupe-at-bound-key"), Some(&at_bound_key))
        .await
        .expect("a key exactly at MAX_IDEMPOTENCY_KEY_BYTES must be accepted");
    assert_eq!(id, "dedupe-at-bound-key");
}

async fn force_job_status_and_age(catalog: &Catalog, job_id: &str, status: &str, days_ago: i64) {
    let cutoff = (chrono::Utc::now() - chrono::Duration::days(days_ago))
        .format("%Y-%m-%dT%H:%M:%S%.9fZ")
        .to_string();
    let job_id = job_id.to_string();
    let status = status.to_string();
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE jobs SET status = $1, updated_at = $2 WHERE job_id = $3",
                    &[
                        SqlValue::TextOwned(status),
                        SqlValue::TextOwned(cutoff),
                        SqlValue::TextOwned(job_id),
                    ],
                )
                .await
            })
        })
        .await
        .unwrap();
}

// ---------------------------------------------------------------------------
// `Catalog::finish_job_with_model` — the atomic finish-with-model-
// registration-and-epoch-checkpoints machinery `training_repo::finalize_
// training_job` provided, reimplemented on the generalised `jobs` schema's
// full attempt guard (C1a design-gap resolution; see the handoff doc).
// ---------------------------------------------------------------------------

/// A minimal queued job over `q-base`, carrying `output_model_id` — set at
/// submit time on the new schema (unlike the removed `training_jobs`, where
/// it was written only by `finalize_training_job`).
fn job_params_with_output<'a>(job_id: &'a str, output_model_id: &'a str) -> SubmitJobParams<'a> {
    SubmitJobParams {
        output_model_id: Some(output_model_id),
        ..job_params(job_id)
    }
}

/// RED without the transaction: `finish_job_with_model` must be a single
/// attempt-guarded compare-and-set. A worker that lost its lease (a zombie)
/// must finish nothing — not the job status, not the output model's served
/// path, and not a single epoch-checkpoint row, even when it presents a
/// NON-EMPTY `epoch_checkpoints` slice — while the live owner's call commits
/// all three atomically. Splitting this into two transactions (finish the
/// job, then separately update the model) would let the loser's CAS-miss on
/// the job row still race a live owner's own model-row write, or would leave
/// a `completed` job with `artifact_path` still `None` on any crash between
/// the two statements — exactly the regression this method exists to close.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finish_job_with_model_is_an_attempt_guarded_compare_and_set(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());
    let job_id = format!("fz-{}", run_suffix());
    let model = format!("jammi:fine-tuned:{job_id}");
    let epoch_0 = format!("{model}:epoch_0");
    let epoch_1 = format!("{model}:epoch_1");

    catalog
        .submit_job(job_params_with_output(&job_id, &model))
        .await
        .unwrap();
    // Register the output model row with no served path: the served
    // `artifact_path` must be committed solely by the winning finish CAS, so
    // it starts NULL and only the live owner's finish sets it.
    catalog
        .register_model(RegisterModelParams {
            model_id: &model,
            version: 1,
            model_type: "fine-tuned",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: Some("q-base::1"),
            artifact_path: None,
            config_json: None,
        })
        .await
        .unwrap();

    // worker-a claims with a zero lease (immediately expired), then worker-b
    // reclaims it via the requeue path and re-claims — worker-b now owns it,
    // at attempt 2. worker-a's stale attempt is 1.
    let claimed_a = catalog
        .claim_next("worker-a", KINDS, Duration::from_secs(0))
        .await
        .unwrap()
        .expect("worker-a claims the job");
    assert_eq!(claimed_a.attempts, 1);
    let actioned = catalog
        .reclaim_expired_jobs(Duration::from_secs(0), 5)
        .await
        .unwrap();
    assert_eq!(actioned, 1, "the expired lease is re-queued");
    let owned_by_b = catalog
        .claim_next("worker-b", KINDS, Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("worker-b re-claims the requeued job");
    assert_eq!(owned_by_b.claimed_by.as_deref(), Some("worker-b"));
    assert_eq!(owned_by_b.attempts, 2);

    // worker-a (the stale owner, attempt=1) tries to finish: zero rows
    // match, so it does not finish and the job is untouched — exercised
    // with a NON-EMPTY `epoch_checkpoints`, not the vacuous empty slice the
    // winner call below uses.
    let loser_epoch_rows = [
        EpochCheckpointRow {
            model_id: &epoch_0,
            model_type: "fine-tuned",
            task: ModelTask::TextEmbedding,
            base_model_id: Some("q-base"),
            artifact_path: "file:///artifacts/fz/worker-a/1/checkpoints/epoch_0",
        },
        EpochCheckpointRow {
            model_id: &epoch_1,
            model_type: "fine-tuned",
            task: ModelTask::TextEmbedding,
            base_model_id: Some("q-base"),
            artifact_path: "file:///artifacts/fz/worker-a/1/checkpoints/epoch_1",
        },
    ];
    let a_finished = catalog
        .finish_job_with_model(FinishJobWithModelParams {
            job_id: &job_id,
            instance_id: "worker-a",
            attempts: 1,
            result: r#"{"k":1}"#,
            output_model_id: &model,
            output_model_version: 1,
            artifact_path: "file:///artifacts/fz/worker-a/1",
            epoch_checkpoints: &loser_epoch_rows,
        })
        .await
        .unwrap();
    assert!(
        !a_finished,
        "a worker whose stale attempt lost the race must not finish the job"
    );
    let after_a = catalog.get_job(&job_id).await.unwrap();
    assert_eq!(
        after_a.status,
        JobStatus::Running.to_string(),
        "the stale attempt's CAS leaves the job running"
    );
    let model_after_a = catalog
        .get_model(&model)
        .await
        .unwrap()
        .expect("the output model row exists");
    assert!(
        model_after_a.artifact_path.is_none(),
        "the stale attempt's failed CAS commits no served path; it stays NULL, got {:?}",
        model_after_a.artifact_path
    );
    for epoch_name in [&epoch_0, &epoch_1] {
        assert!(
            catalog.get_model(epoch_name).await.unwrap().is_none(),
            "a lease-lost attempt must never register an epoch-checkpoint row \
             ({epoch_name} must not exist)"
        );
    }

    // worker-b (the live owner, attempt=2) finishes: one row matches, the
    // job completes with the served path committed atomically.
    let b_finished = catalog
        .finish_job_with_model(FinishJobWithModelParams {
            job_id: &job_id,
            instance_id: "worker-b",
            attempts: 2,
            result: r#"{"completed_at":"2026-01-01T00:00:00Z"}"#,
            output_model_id: &model,
            output_model_version: 1,
            artifact_path: "file:///artifacts/fz/worker-b/2",
            epoch_checkpoints: &[],
        })
        .await
        .unwrap();
    assert!(b_finished, "the current attempt's owner finishes the job");
    let after_b = catalog.get_job(&job_id).await.unwrap();
    assert_eq!(after_b.status, JobStatus::Completed.to_string());
    let model_after_b = catalog
        .get_model(&model)
        .await
        .unwrap()
        .expect("the output model row exists");
    assert_eq!(
        model_after_b.artifact_path.as_deref(),
        Some("file:///artifacts/fz/worker-b/2"),
        "the winning finish CAS commits the live owner's prefix as the served path — \
         the sole writer of the committed pointer"
    );

    // A second finish by the same owner, same attempt is a no-op (status
    // is not running), so finish-with-model is not re-runnable once
    // terminal.
    let again = catalog
        .finish_job_with_model(FinishJobWithModelParams {
            job_id: &job_id,
            instance_id: "worker-b",
            attempts: 2,
            result: "{}",
            output_model_id: &model,
            output_model_version: 1,
            artifact_path: "file:///artifacts/fz/worker-b/2",
            epoch_checkpoints: &[],
        })
        .await
        .unwrap();
    assert!(!again, "a completed job cannot be finished again");
}

/// B5 hardening regression (unit 348), ported: the finish-with-model CAS's
/// model-row `UPDATE` must be scoped by `name AND version AND tenant`, never
/// `name` alone — three rows share the name "acme/tuned": (tenant-a, v1),
/// (tenant-a, v2), (tenant-b, v1) — and a tenant-a finish naming version 1
/// touches EXACTLY that row.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn finish_job_with_model_update_is_scoped_by_version_and_tenant(backend: BackendKind) {
    use std::str::FromStr;

    use jammi_db::TenantId;

    let dir = tempdir().unwrap();
    let session = skip_if_no_backend!(backend, dir.path());
    let base = Arc::clone(session.catalog());
    reset_queue(&base).await;

    let tenant_a = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap();
    let tenant_b = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9b").unwrap();
    let cat_a = base.pinned_to_tenant(Some(tenant_a));
    let cat_b = base.pinned_to_tenant(Some(tenant_b));

    for version in [1, 2] {
        cat_a
            .register_model(RegisterModelParams {
                model_id: "acme/tuned",
                version,
                model_type: "fine-tuned",
                backend: "candle",
                task: ModelTask::TextEmbedding,
                base_model_id: None,
                artifact_path: None,
                config_json: None,
            })
            .await
            .unwrap();
    }
    cat_b
        .register_model(RegisterModelParams {
            model_id: "acme/tuned",
            version: 1,
            model_type: "fine-tuned",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: None,
            config_json: None,
        })
        .await
        .unwrap();

    cat_a
        .register_model(RegisterModelParams {
            model_id: "acme/base",
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
    let base_pk = cat_a
        .get_model("acme/base")
        .await
        .unwrap()
        .expect("the base model row exists")
        .catalog_pk;
    cat_a
        .submit_job(SubmitJobParams {
            job_id: "vt-1",
            kind: KIND,
            execution: JobExecution::Queued,
            spec: "{}",
            model_ref: Some(&base_pk),
            output_model_id: Some("acme/tuned"),
            model_source: None,
            priority: 0,
        })
        .await
        .unwrap();
    cat_a
        .claim_next("worker-a", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .expect("the queued job is claimable");

    let finished = cat_a
        .finish_job_with_model(FinishJobWithModelParams {
            job_id: "vt-1",
            instance_id: "worker-a",
            attempts: 1,
            result: "{}",
            output_model_id: "acme/tuned",
            output_model_version: 1,
            artifact_path: "file:///artifacts/vt-1/worker-a/0",
            epoch_checkpoints: &[],
        })
        .await
        .unwrap();
    assert!(finished, "tenant-a's lease holder finishes the job");

    let a_v1 = cat_a
        .get_model_version("acme/tuned", 1)
        .await
        .unwrap()
        .expect("tenant-a v1 exists");
    assert_eq!(
        a_v1.artifact_path.as_deref(),
        Some("file:///artifacts/vt-1/worker-a/0"),
        "the finish's OWN (tenant, version) row is the one touched"
    );

    let a_v2 = cat_a
        .get_model_version("acme/tuned", 2)
        .await
        .unwrap()
        .expect("tenant-a v2 exists");
    assert!(
        a_v2.artifact_path.is_none(),
        "a DIFFERENT version sharing the same name must not be clobbered"
    );

    let b_v1 = cat_b
        .get_model_version("acme/tuned", 1)
        .await
        .unwrap()
        .expect("tenant-b v1 exists");
    assert!(
        b_v1.artifact_path.is_none(),
        "a DIFFERENT tenant's row sharing the same name must not be clobbered"
    );
}

/// A checkpoint row's catalog NAME occupied by
/// an unrelated, pre-existing model row must be skipped (never clobbered,
/// never shadow-inserted) while every OTHER retained epoch row registers
/// normally and the job still finishes successfully — checkpoints are
/// supplementary, never a reason to fail the job.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn finish_job_with_model_skips_a_name_occupied_epoch_checkpoint(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    // A pre-existing, unrelated row occupies the name the epoch_0 checkpoint
    // would otherwise claim.
    catalog
        .register_model(RegisterModelParams {
            model_id: "jammi:fine-tuned:oc:epoch_0",
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: Some("file:///pre-existing"),
            config_json: None,
        })
        .await
        .unwrap();
    catalog
        .register_model(RegisterModelParams {
            model_id: "jammi:fine-tuned:oc",
            version: 1,
            model_type: "fine-tuned",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: Some("q-base::1"),
            artifact_path: None,
            config_json: None,
        })
        .await
        .unwrap();
    catalog
        .submit_job(job_params_with_output("oc", "jammi:fine-tuned:oc"))
        .await
        .unwrap();
    catalog
        .claim_next("worker-a", KINDS, Duration::from_secs(30))
        .await
        .unwrap()
        .expect("job claimed");

    let epoch_rows = [
        EpochCheckpointRow {
            model_id: "jammi:fine-tuned:oc:epoch_0",
            model_type: "fine-tuned",
            task: ModelTask::TextEmbedding,
            base_model_id: Some("q-base"),
            artifact_path: "file:///artifacts/oc/worker-a/0/checkpoints/epoch_0",
        },
        EpochCheckpointRow {
            model_id: "jammi:fine-tuned:oc:epoch_1",
            model_type: "fine-tuned",
            task: ModelTask::TextEmbedding,
            base_model_id: Some("q-base"),
            artifact_path: "file:///artifacts/oc/worker-a/0/checkpoints/epoch_1",
        },
    ];
    let finished = catalog
        .finish_job_with_model(FinishJobWithModelParams {
            job_id: "oc",
            instance_id: "worker-a",
            attempts: 1,
            result: "{}",
            output_model_id: "jammi:fine-tuned:oc",
            output_model_version: 1,
            artifact_path: "file:///artifacts/oc/worker-a/0",
            epoch_checkpoints: &epoch_rows,
        })
        .await
        .unwrap();
    assert!(finished, "an occupied checkpoint name never fails the job");

    let occupied = catalog
        .get_model("jammi:fine-tuned:oc:epoch_0")
        .await
        .unwrap()
        .expect("the occupying row still exists");
    assert_eq!(
        occupied.artifact_path.as_deref(),
        Some("file:///pre-existing"),
        "the occupying row must not be clobbered by the skipped checkpoint"
    );
    let sibling = catalog
        .get_model("jammi:fine-tuned:oc:epoch_1")
        .await
        .unwrap()
        .expect("the non-colliding sibling checkpoint registers normally");
    assert_eq!(
        sibling.artifact_path.as_deref(),
        Some("file:///artifacts/oc/worker-a/0/checkpoints/epoch_1")
    );
}

// ---------------------------------------------------------------------------
// esc-075 — the tri-state `acceleration_report` lifecycle is closed AT THE
// CATALOG EDGE, ported onto the generalised `jobs` schema.
//
// `submit_job` stamps `{"state":"pending"}` = "submitted, no claimant has
// computed a determination YET". That sentence stops being true the moment
// the row reaches a TERMINAL status. Every terminal write (finish, fail, and
// both terminal reclaim arms) rewrites a still-pending marker inside its OWN
// UPDATE, never at N caller sites; the non-terminal requeue arm resets it,
// since a new attempt will re-probe.
// ---------------------------------------------------------------------------

const PENDING_MARKER: &str = r#"{"state":"pending"}"#;
const FAILED_BEFORE_PROBE: &str = r#"{"state":"undetermined","reason":"failed_before_probe"}"#;
const LEASE_EXPIRED_EXHAUSTED: &str =
    r#"{"state":"undetermined","reason":"lease_expired_attempts_exhausted"}"#;
const FINALIZED_WITHOUT_DETERMINATION: &str =
    r#"{"state":"undetermined","reason":"finalized_without_determination"}"#;
const INLINE_EXECUTOR_DIED: &str = r#"{"state":"undetermined","reason":"inline_executor_died"}"#;

/// Overwrite `acceleration_report` for `job_id` directly, bypassing every
/// lease guard — used only to construct a state the public API cannot reach
/// (a legacy `NULL`).
async fn set_acceleration_report_raw(catalog: &Catalog, job_id: &str, report: Option<&str>) {
    let job_id = job_id.to_string();
    let report = report.map(str::to_string);
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE jobs SET acceleration_report = $1 WHERE job_id = $2",
                    &[SqlValue::from(report), SqlValue::TextOwned(job_id)],
                )
                .await
            })
        })
        .await
        .unwrap();
}

/// [`Catalog::submit_job`] writes the explicit pending marker (never SQL
/// NULL); the claiming instance's [`Catalog::record_acceleration_report`]
/// overwrites it under a valid lease, pinning `attempts` to the exact
/// attempt the claim returned. A non-owner cannot write it.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn record_acceleration_report_writes_under_a_valid_lease(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.submit_job(job_params("ar-1")).await.unwrap();
    let submitted = catalog.get_job("ar-1").await.unwrap();
    assert_eq!(
        submitted.acceleration_report.as_deref(),
        Some(PENDING_MARKER),
        "submission writes the explicit pending marker, never NULL"
    );

    let claimed = catalog
        .claim_next("worker-a", KINDS, Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("job claimed");
    assert_eq!(claimed.attempts, 1);

    let report =
        r#"{"state":"determined","attempt":1,"device":"cuda:0","ops":{"layer_norm":"hit"}}"#;
    let wrote = catalog
        .record_acceleration_report("ar-1", "worker-a", claimed.attempts, report)
        .await
        .unwrap();
    assert!(wrote, "the current lease holder's write must land");

    let after = catalog.get_job("ar-1").await.unwrap();
    assert_eq!(
        after.acceleration_report.as_deref(),
        Some(report),
        "the determined report round-trips through parse_row verbatim"
    );

    let stolen = catalog
        .record_acceleration_report(
            "ar-1",
            "worker-b",
            claimed.attempts,
            "{\"state\":\"determined\"}",
        )
        .await
        .unwrap();
    assert!(!stolen, "a non-owner must not overwrite the report");
    let unchanged = catalog.get_job("ar-1").await.unwrap();
    assert_eq!(unchanged.acceleration_report.as_deref(), Some(report));
}

/// The mandatory `attempts` guard closes the zombie gap: an instance id can
/// recur (e.g. a stable `JAMMI_WORKER_ID` across process restarts), so a
/// reclaimed job re-claimed by an instance carrying the SAME id is
/// distinguished from its own zombie only by `attempts`.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn record_acceleration_report_rejects_a_zombies_stale_attempt(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.submit_job(job_params("ar-zombie")).await.unwrap();

    let first = catalog
        .claim_next("worker-a", KINDS, Duration::from_secs(0))
        .await
        .unwrap()
        .expect("worker-a claims attempt 1");
    assert_eq!(first.attempts, 1);

    let actioned = catalog
        .reclaim_expired_jobs(Duration::from_secs(0), 5)
        .await
        .unwrap();
    assert_eq!(actioned, 1, "the expired lease is re-queued");

    let second = catalog
        .claim_next("worker-a", KINDS, Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("worker-a re-claims as attempt 2");
    assert_eq!(second.attempts, 2);
    assert_eq!(second.claimed_by.as_deref(), Some("worker-a"));

    let zombie_report = r#"{"state":"determined","attempt":1,"stale":true}"#;
    let zombie_wrote = catalog
        .record_acceleration_report("ar-zombie", "worker-a", 1, zombie_report)
        .await
        .unwrap();
    assert!(
        !zombie_wrote,
        "a zombie presenting the OLD attempt must not write, even with the SAME \
         instance id and a currently-running status"
    );

    let current_report = r#"{"state":"determined","attempt":2,"stale":false}"#;
    let current_wrote = catalog
        .record_acceleration_report("ar-zombie", "worker-a", 2, current_report)
        .await
        .unwrap();
    assert!(current_wrote, "the current claimant's write must land");

    let late_zombie_wrote = catalog
        .record_acceleration_report("ar-zombie", "worker-a", 1, zombie_report)
        .await
        .unwrap();
    assert!(!late_zombie_wrote, "a late zombie write is still rejected");
    let after_late_zombie = catalog.get_job("ar-zombie").await.unwrap();
    assert_eq!(
        after_late_zombie.acceleration_report.as_deref(),
        Some(current_report),
        "the current claimant's report survives a late zombie write attempt"
    );
}

/// `acceleration_report` is not the `result` blob: `finish_job` overwrites
/// `status`/`result` but leaves `acceleration_report` untouched once it is
/// already a determination (not the pending marker).
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acceleration_report_survives_finish(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.submit_job(job_params("ar-fin")).await.unwrap();
    let claimed = catalog
        .claim_next("worker-a", KINDS, Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("job claimed");
    let report = r#"{"state":"determined","attempt":1,"device":"cuda:0"}"#;
    catalog
        .record_acceleration_report("ar-fin", "worker-a", claimed.attempts, report)
        .await
        .unwrap();

    let finished = catalog
        .finish_job(FinishJobParams {
            job_id: "ar-fin",
            instance_id: "worker-a",
            attempts: claimed.attempts,
            result: r#"{"completed_at":"2026-01-01T00:00:00Z"}"#,
        })
        .await
        .unwrap();
    assert!(finished, "the lease owner finishes the job");

    let after = catalog.get_job("ar-fin").await.unwrap();
    assert_eq!(after.status, JobStatus::Completed.to_string());
    assert_eq!(
        after.acceleration_report.as_deref(),
        Some(report),
        "finish must leave an already-determined acceleration_report INTACT"
    );
}

/// The terminal `fail` transition leaves an already-determined
/// `acceleration_report` untouched, the same as `finish`.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acceleration_report_survives_fail(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.submit_job(job_params("ar-fail")).await.unwrap();
    let claimed = catalog
        .claim_next("worker-a", KINDS, Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("job claimed");
    let report = r#"{"state":"determined","attempt":1,"device":"cpu"}"#;
    catalog
        .record_acceleration_report("ar-fail", "worker-a", claimed.attempts, report)
        .await
        .unwrap();

    let failed = catalog
        .fail_job("ar-fail", "worker-a", claimed.attempts, "boom")
        .await
        .unwrap();
    assert!(failed, "the lease owner records the failure");

    let after = catalog.get_job("ar-fail").await.unwrap();
    assert_eq!(after.status, JobStatus::Failed.to_string());
    assert_eq!(
        after.acceleration_report.as_deref(),
        Some(report),
        "fail must leave an already-determined acceleration_report INTACT"
    );
}

/// The requeue reclaim arm resets a dead attempt's report to the pending
/// marker (a new attempt will re-probe), while the terminal attempts-
/// exhausted arm preserves whatever determination the last attempt recorded
/// — the two arms move `acceleration_report` in OPPOSITE directions because
/// they mean opposite things about the job's future (#446 finding 1).
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acceleration_report_reclaim_arms_reset_on_requeue_and_persist_on_exhaustion(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.submit_job(job_params("ar-rc")).await.unwrap();
    let claimed = catalog
        .claim_next("worker-a", KINDS, Duration::from_secs(0))
        .await
        .unwrap()
        .expect("job claimed");
    let stale_report = r#"{"state":"determined","attempt":1,"stale":true}"#;
    let wrote = catalog
        .record_acceleration_report("ar-rc", "worker-a", claimed.attempts, stale_report)
        .await
        .unwrap();
    assert!(
        wrote,
        "the attempt-1 claimant writes its report before expiry"
    );

    let actioned = catalog
        .reclaim_expired_jobs(Duration::from_secs(0), 5)
        .await
        .unwrap();
    assert_eq!(actioned, 1, "the expired lease is re-queued");
    let requeued = catalog.get_job("ar-rc").await.unwrap();
    assert_eq!(requeued.status, JobStatus::Queued.to_string());
    assert_eq!(
        requeued.acceleration_report.as_deref(),
        Some(PENDING_MARKER),
        "the requeue arm resets the dead attempt's report to the pending marker"
    );

    let reclaimed = catalog
        .claim_next("worker-b", KINDS, Duration::from_secs(0))
        .await
        .unwrap()
        .expect("worker-b re-claims the requeued job");
    assert_eq!(reclaimed.attempts, 2);
    let fresh_report = r#"{"state":"determined","attempt":2,"stale":false}"#;
    catalog
        .record_acceleration_report("ar-rc", "worker-b", reclaimed.attempts, fresh_report)
        .await
        .unwrap();

    let actioned = catalog
        .reclaim_expired_jobs(Duration::from_secs(0), 2)
        .await
        .unwrap();
    assert_eq!(actioned, 1, "attempts (2) >= max (2): the job fails");
    let failed = catalog.get_job("ar-rc").await.unwrap();
    assert_eq!(failed.status, JobStatus::Failed.to_string());
    assert_eq!(
        failed.acceleration_report.as_deref(),
        Some(fresh_report),
        "the exhausted-attempts reclaim arm leaves acceleration_report untouched"
    );
}

/// (i) A job that fails between claim and the acceleration probe must not
/// keep the submission-time `pending` marker past its terminal `failed`
/// status: [`Catalog::fail_job`] rewrites `pending -> undetermined` with
/// reason `failed_before_probe` inside the same attempt-guarded UPDATE.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fail_rewrites_a_pending_report_to_undetermined(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog
        .submit_job(job_params("ar-f-pending"))
        .await
        .unwrap();
    let claimed = catalog
        .claim_next("worker-a", KINDS, Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("job claimed");
    let before = catalog.get_job("ar-f-pending").await.unwrap();
    assert_eq!(
        before.acceleration_report.as_deref(),
        Some(PENDING_MARKER),
        "precondition: the row still carries the submission-time pending marker"
    );

    let failed = catalog
        .fail_job(
            "ar-f-pending",
            "worker-a",
            claimed.attempts,
            "died before probe",
        )
        .await
        .unwrap();
    assert!(failed, "the lease owner records the failure");

    let after = catalog.get_job("ar-f-pending").await.unwrap();
    assert_eq!(after.status, JobStatus::Failed.to_string());
    assert_eq!(
        after.acceleration_report.as_deref(),
        Some(FAILED_BEFORE_PROBE),
        "a terminal `failed` row must never carry `pending`"
    );
}

/// (i, finish peer) [`Catalog::finish_job`] rewrites `pending ->
/// undetermined/finalized_without_determination` when nothing ever recorded
/// a determination — success is not evidence a determination exists.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finish_rewrites_a_pending_report_to_undetermined(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog
        .submit_job(job_params("ar-fz-pending"))
        .await
        .unwrap();
    let claimed = catalog
        .claim_next("worker-a", KINDS, Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("job claimed");

    let finished = catalog
        .finish_job(FinishJobParams {
            job_id: "ar-fz-pending",
            instance_id: "worker-a",
            attempts: claimed.attempts,
            result: "{}",
        })
        .await
        .unwrap();
    assert!(finished);

    let after = catalog.get_job("ar-fz-pending").await.unwrap();
    assert_eq!(after.status, JobStatus::Completed.to_string());
    assert_eq!(
        after.acceleration_report.as_deref(),
        Some(FINALIZED_WITHOUT_DETERMINATION),
        "a completed row that never recorded a determination must not carry `pending`"
    );
}

/// (ii) The rewrite is strictly `pending -> undetermined`: an already-
/// terminal report (`determined`, `not_applicable`, or a more specific
/// `undetermined`) is PRESERVED byte-for-byte on `fail`.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fail_preserves_an_already_terminal_report(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    let cases = [
        (
            "ar-f-det",
            r#"{"state":"determined","attempt":1,"device":"cuda:0"}"#,
        ),
        (
            "ar-f-na",
            r#"{"state":"not_applicable","reason":"context_predictor"}"#,
        ),
        (
            "ar-f-und",
            r#"{"state":"undetermined","reason":"failed_before_device_resolution"}"#,
        ),
    ];

    for (job_id, report) in cases {
        catalog.submit_job(job_params(job_id)).await.unwrap();
        let claimed = catalog
            .claim_next("worker-a", KINDS, Duration::from_secs(3600))
            .await
            .unwrap()
            .expect("job claimed");
        let wrote = catalog
            .record_acceleration_report(job_id, "worker-a", claimed.attempts, report)
            .await
            .unwrap();
        assert!(wrote, "the lease owner records its determination");

        let failed = catalog
            .fail_job(job_id, "worker-a", claimed.attempts, "boom")
            .await
            .unwrap();
        assert!(failed, "the lease owner records the failure");

        let after = catalog.get_job(job_id).await.unwrap();
        assert_eq!(after.status, JobStatus::Failed.to_string());
        assert_eq!(
            after.acceleration_report.as_deref(),
            Some(report),
            "{job_id}: an already-terminal report survives `fail` byte-for-byte"
        );
    }
}

/// (ii, degenerate) A legacy `NULL` reads back as "unknown". `fail` must NOT
/// fabricate a state for it: `NULL` is not `pending`, and the three-valued
/// `NULL = 'x'` comparison correctly falls through to the ELSE arm on both
/// backends.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fail_leaves_a_legacy_null_report_unknown(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.submit_job(job_params("ar-f-null")).await.unwrap();
    set_acceleration_report_raw(&catalog, "ar-f-null", None).await;
    let claimed = catalog
        .claim_next("worker-a", KINDS, Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("job claimed");

    let failed = catalog
        .fail_job("ar-f-null", "worker-a", claimed.attempts, "x")
        .await
        .unwrap();
    assert!(failed);

    let after = catalog.get_job("ar-f-null").await.unwrap();
    assert_eq!(after.status, JobStatus::Failed.to_string());
    assert_eq!(
        after.acceleration_report, None,
        "a legacy NULL stays NULL ('unknown') — the pending rewrite must never fabricate \
         a state for a row that never had one"
    );
}

/// (iii) The reclaim attempts-exhausted arm is the OTHER path to a terminal
/// `failed` status — reached with no live claimant at all. Same rule, its
/// own reason: `pending -> undetermined/lease_expired_attempts_exhausted`.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reclaim_exhausted_rewrites_a_pending_report_to_undetermined(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.submit_job(job_params("ar-rc-exh")).await.unwrap();
    catalog
        .claim_next("worker-a", KINDS, Duration::from_secs(0))
        .await
        .unwrap()
        .expect("job claimed");

    let actioned = catalog
        .reclaim_expired_jobs(Duration::from_secs(0), 1)
        .await
        .unwrap();
    assert_eq!(
        actioned, 1,
        "attempts (1) >= max (1): the job fails outright"
    );

    let after = catalog.get_job("ar-rc-exh").await.unwrap();
    assert_eq!(after.status, JobStatus::Failed.to_string());
    assert_eq!(
        after.acceleration_report.as_deref(),
        Some(LEASE_EXPIRED_EXHAUSTED),
        "the exhausted-attempts reclaim arm retires a still-pending report with its own reason"
    );
}

/// (iv, new arm) An `inline`-execution job's owning `instances` row going
/// stale/absent is ALSO a terminal `failed` transition with no live
/// claimant — the same esc-075 rule applies with its own distinct reason,
/// `inline_executor_died`, never conflated with the queued-execution
/// exhaustion reason above.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn inline_liveness_reclaim_rewrites_a_pending_report_to_undetermined(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog
        .submit_job(inline_job_params("ar-inline-dead"))
        .await
        .unwrap();
    // No `instances` row for "ghost" at all — the absent-owner sub-case.
    catalog
        .claim_by_id("ar-inline-dead", "ghost", Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("inline job claimed by id");

    let actioned = catalog
        .reclaim_expired_jobs(Duration::from_millis(1), 5)
        .await
        .unwrap();
    assert_eq!(actioned, 1, "the dead inline executor's job is failed");

    let after = catalog.get_job("ar-inline-dead").await.unwrap();
    assert_eq!(after.status, JobStatus::Failed.to_string());
    assert_eq!(
        after.acceleration_report.as_deref(),
        Some(INLINE_EXECUTOR_DIED),
        "the inline-liveness reclaim arm retires a still-pending report with its own, \
         distinct reason — never conflated with the queued-execution exhaustion reason"
    );
}

// ---------------------------------------------------------------------------
// esc-107 — `Catalog::create_result_table`'s `jobs.partial_result` CAS must
// carry the full `(job_id, claimed_by, attempts)` attempt guard, not
// `job_id` alone: a `job_id`-only predicate lets a zombie of a REQUEUED and
// RE-CLAIMED attempt still win the CAS, because the job genuinely IS
// `running` again — just under a LATER attempt the zombie never learned
// about.
// ---------------------------------------------------------------------------

fn result_table_params<'a>(
    table_name: &'a str,
    job_attempt: JobAttempt<'a>,
) -> CreateResultTableParams<'a> {
    CreateResultTableParams {
        table_name,
        source_id: "src",
        model_id: "q-base",
        task: ModelTask::TextEmbedding,
        kind: ResultTableKind::Model,
        derived_from: None,
        parquet_path: "file:///tmp/rt.parquet",
        dimensions: Some(4),
        key_column: None,
        text_columns: None,
        storage_precision: StoragePrecision::F32,
        oversample: 4,
        created_at: jammi_db::catalog::backend::now_sortable(),
        writer_id: None,
        lease: None,
        job_attempt: Some(job_attempt),
    }
}

/// RED against the `job_id`-only predicate this unit replaced: a zombie
/// presenting a SUPERSEDED attempt (its lease expired, the job was requeued
/// and re-claimed by a different instance, all while the zombie never
/// learned its lease was gone) must not win the `partial_result` CAS just
/// because the job is still `running` — it is running under a LATER
/// attempt. The live, current attempt's own call must be the one that
/// actually lands.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_result_table_cas_rejects_a_zombies_stale_attempt_across_a_reclaim(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());
    let job_id = format!("rt-1-{}", run_suffix());
    let zombie_table = format!("{job_id}-zombie-table");
    let live_table = format!("{job_id}-live-table");

    catalog.submit_job(job_params(&job_id)).await.unwrap();
    let claimed_a = catalog
        .claim_next("worker-a", KINDS, Duration::from_secs(0))
        .await
        .unwrap()
        .expect("worker-a claims the job");
    assert_eq!(claimed_a.attempts, 1);

    // worker-a's lease is already expired; reclaim requeues it, then
    // worker-b claims it fresh — the job is `running` again, but under
    // attempt 2, owned by worker-b. worker-a never observes any of this.
    let actioned = catalog
        .reclaim_expired_jobs(Duration::from_secs(0), 5)
        .await
        .unwrap();
    assert_eq!(actioned, 1, "the expired lease is re-queued");
    let claimed_b = catalog
        .claim_next("worker-b", KINDS, Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("worker-b re-claims the requeued job");
    assert_eq!(claimed_b.attempts, 2);

    // The zombie (worker-a, presenting its OLD attempt=1) tries to record
    // its own table as the job's partial_result. The job IS `running` —
    // just not under worker-a's attempt — so a `job_id`-only CAS would have
    // let this land; the full attempt guard must reject it.
    let zombie_result = catalog
        .create_result_table(result_table_params(
            &zombie_table,
            JobAttempt {
                job_id: &job_id,
                instance_id: "worker-a",
                attempts: 1,
            },
        ))
        .await;
    assert!(
        matches!(
            zombie_result,
            Err(jammi_db::error::JammiError::JobAttemptSuperseded { .. })
        ),
        "a zombie's stale attempt must be rejected as superseded, got {zombie_result:?}"
    );
    let after_zombie = catalog.get_job(&job_id).await.unwrap();
    assert_eq!(
        after_zombie.partial_result, None,
        "the zombie's rejected CAS must not record ANY partial_result — the row it tried \
         to insert must also be rolled back, not just left unlinked"
    );
    assert!(
        catalog
            .get_result_table(&zombie_table)
            .await
            .unwrap()
            .is_none(),
        "the zombie's `result_tables` row must have been rolled back with its CAS — \
         no orphan `{zombie_table}` row may survive the rejected transaction"
    );

    // The live, current attempt (worker-b, attempt=2) records its own table
    // successfully.
    catalog
        .create_result_table(result_table_params(
            &live_table,
            JobAttempt {
                job_id: &job_id,
                instance_id: "worker-b",
                attempts: 2,
            },
        ))
        .await
        .unwrap();
    let after_live = catalog.get_job(&job_id).await.unwrap();
    assert_eq!(
        after_live.partial_result.as_deref(),
        Some(live_table.as_str()),
        "the live attempt's own table is the one recorded as partial_result"
    );
}

// ---------------------------------------------------------------------------
// OPS (#482) — lease RELEASE on the jobs class: `release_job_lease` /
// `release_jobs_claimed_by` (the heartbeat CAS + `lease_expires_at = NULL,
// releases = releases + 1`, both carrying `AND lease_expires_at IS NOT NULL`),
// the release-aware reclaim cap (`attempts - releases`), the re-arm guard on
// `heartbeat_job`, `clear_partial_result`, the gauge sample query, and
// `workers.state`.
// ---------------------------------------------------------------------------

/// A compute-kind queued job (no FK model) with the given id.
fn compute_job_params(job_id: &str) -> SubmitJobParams<'_> {
    SubmitJobParams {
        kind: "embedding",
        model_ref: None,
        ..job_params(job_id)
    }
}

/// A released job (jobs lease NULL, `releases + 1`) is requeued by the very
/// next reclaim pass — with the FULL, unexpired lease window — instead of
/// waiting out `[lease] duration_secs`; a heartbeat can never re-arm the
/// released lease in between. Base: `reclaim_expired_jobs` matches 0 rows
/// until the lease deadline passes.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn released_job_is_requeued_by_the_next_reclaim_without_waiting_for_expiry(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());
    let lease = Duration::from_secs(3600);

    catalog.submit_job(job_params("rel")).await.unwrap();
    catalog.submit_job(job_params("live")).await.unwrap();
    let claimed = catalog
        .claim_next("me", KINDS, lease)
        .await
        .unwrap()
        .expect("job claimed");
    assert_eq!(claimed.job_id, "rel");
    assert_eq!(claimed.attempts, 1);
    assert_eq!(claimed.releases, 0, "a fresh claim has no releases");
    let live = catalog
        .claim_next("me", KINDS, lease)
        .await
        .unwrap()
        .expect("second job claimed");
    assert_eq!(live.job_id, "live");

    // Control: nothing is released yet, and the leases are hours away —
    // the reclaim sweep sees no expired row.
    let actioned = catalog.reclaim_expired_jobs(lease, 3).await.unwrap();
    assert_eq!(actioned, 0, "a live, unreleased lease is never reclaimed");

    let released = catalog.release_job_lease("rel", "me", 1).await.unwrap();
    assert!(released, "the owner releases its own live lease");
    let row = catalog.get_job("rel").await.unwrap();
    assert_eq!(row.status, JobStatus::Running.to_string());
    assert_eq!(row.claimed_by.as_deref(), Some("me"));
    assert!(row.lease_expires_at.is_none(), "release NULLs the lease");
    assert_eq!(row.releases, 1);
    assert_eq!(row.attempts, 1, "release never touches attempts");

    // The re-arm guard: the heartbeat CAS carries `lease_expires_at IS NOT
    // NULL`, so the old holder's next renewal matches zero rows.
    let renewed = catalog.heartbeat_job("rel", "me", 1, lease).await.unwrap();
    assert!(!renewed, "a heartbeat must never re-arm a released lease");
    assert!(
        catalog
            .get_job("rel")
            .await
            .unwrap()
            .lease_expires_at
            .is_none(),
        "the released lease stays NULL after the heartbeat"
    );

    // The next reclaim requeues it at once, under the full lease window.
    let actioned = catalog.reclaim_expired_jobs(lease, 3).await.unwrap();
    assert_eq!(actioned, 1, "exactly the released row is requeued");
    let requeued = catalog.get_job("rel").await.unwrap();
    assert_eq!(requeued.status, JobStatus::Queued.to_string());
    assert!(requeued.claimed_by.is_none());
    assert_eq!(
        requeued.releases, 1,
        "the release count survives the requeue"
    );
    assert_eq!(requeued.attempts, 1);
    let untouched = catalog.get_job("live").await.unwrap();
    assert_eq!(untouched.status, JobStatus::Running.to_string());
    assert!(untouched.lease_expires_at.is_some());
    assert_eq!(untouched.releases, 0);
}

/// R2 (round-2 REFINE, #482): the release write's own per-arm effect,
/// pinned as a before/after DELTA rather than a cross-arm row-equality
/// claim (the deleted `release_write_is_identical_on_library_and_server`
/// had zero true-positive capacity — both its arms funnelled into the
/// SAME `release_and_stop` call, and the design's own outcome (iii)
/// permits the two rows it compared to differ). `release_job_lease`'s SQL
/// (`jobs_repo.rs`) sets exactly three columns —
/// `lease_expires_at = NULL, releases = releases + 1, updated_at = $now`
/// — so this asserts that delta EXACTLY: every other `JobRecord` field is
/// byte-identical across the snapshot taken immediately before the
/// release and the one taken immediately after, field by field (a struct
/// comparison would need `PartialEq` on `JobRecord`, which this crate
/// does not derive — asserting field-by-field is also the more legible
/// failure on a mismatch). A second row, claimed by a DIFFERENT instance,
/// is asserted completely untouched (not merely its `releases`/lease) —
/// the release write must never touch a row it was not asked to release.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_job_lease_write_is_exactly_lease_null_releases_plus_one_and_timestamp(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());
    let lease = Duration::from_secs(3600);

    catalog.submit_job(job_params("delta")).await.unwrap();
    catalog.submit_job(job_params("other-owner")).await.unwrap();
    catalog
        .claim_next("me", KINDS, lease)
        .await
        .unwrap()
        .expect("the first job claimed by `me`");
    let other = catalog
        .claim_next("someone-else", KINDS, lease)
        .await
        .unwrap()
        .expect("the second job claimed by a DIFFERENT instance");
    assert_eq!(other.job_id, "other-owner");
    let other_before = catalog.get_job("other-owner").await.unwrap();

    // A timestamp column stored with second-or-finer resolution needs a
    // strictly later `now` to observe as "bumped" on some backends — a
    // fixed sleep here is not a race (this crate's own timestamp fixtures,
    // e.g. `claim_returns_oldest_queued_job_first` above, do the same).
    tokio::time::sleep(Duration::from_millis(1100)).await;

    let before = catalog.get_job("delta").await.unwrap();
    assert!(before.lease_expires_at.is_some(), "a fresh claim is leased");
    let released = catalog.release_job_lease("delta", "me", 1).await.unwrap();
    assert!(released, "the owner releases its own live lease");
    let after = catalog.get_job("delta").await.unwrap();

    // The three-field delta, exactly.
    assert!(after.lease_expires_at.is_none(), "{after:?}");
    assert_eq!(after.releases, before.releases + 1, "{after:?}");
    assert!(
        after.updated_at > before.updated_at,
        "updated_at must be bumped: before {:?}, after {:?}",
        before.updated_at,
        after.updated_at
    );

    // Every OTHER field byte-identical -- field by field (`JobRecord`
    // derives no `PartialEq`).
    assert_eq!(after.job_id, before.job_id);
    assert_eq!(after.kind, before.kind);
    assert_eq!(after.tenant_id, before.tenant_id);
    assert_eq!(after.status, before.status);
    assert_eq!(after.execution, before.execution);
    assert_eq!(after.spec, before.spec);
    assert_eq!(after.partial_result, before.partial_result);
    assert_eq!(after.result, before.result);
    assert_eq!(after.error, before.error);
    assert_eq!(after.progress_rows_done, before.progress_rows_done);
    assert_eq!(after.progress_rows_total, before.progress_rows_total);
    assert_eq!(after.progress_phase, before.progress_phase);
    assert_eq!(after.cancel_requested, before.cancel_requested);
    assert_eq!(after.model_ref, before.model_ref);
    assert_eq!(after.output_model_id, before.output_model_id);
    assert_eq!(after.model_source, before.model_source);
    assert_eq!(after.claimed_by, before.claimed_by);
    assert_eq!(
        after.attempts, before.attempts,
        "release never touches attempts"
    );
    assert_eq!(after.priority, before.priority);
    assert_eq!(after.claimable, before.claimable);
    assert_eq!(after.acceleration_report, before.acceleration_report);
    assert_eq!(after.created_at, before.created_at);

    // A row claimed by a DIFFERENT instance: completely untouched, not
    // merely its lease/`releases`.
    let other_after = catalog.get_job("other-owner").await.unwrap();
    assert_eq!(other_after.lease_expires_at, other_before.lease_expires_at);
    assert_eq!(other_after.releases, other_before.releases);
    assert_eq!(other_after.updated_at, other_before.updated_at);
    assert_eq!(other_after.claimed_by, other_before.claimed_by);
    assert_eq!(other_after.attempts, other_before.attempts);
    assert_eq!(other_after.status, other_before.status);
}

/// Every OTHER `JobRecord` field, byte-identical, field by field (no
/// `PartialEq` derive) — the shared assertion body for the sweep delta
/// oracle below, near-copied from
/// `release_job_lease_write_is_exactly_lease_null_releases_plus_one_and_timestamp`'s
/// inline comparison rather than factored out there too, so each oracle
/// stays independently readable.
fn assert_only_release_columns_changed(
    before: &jammi_db::catalog::jobs_repo::JobRecord,
    after: &jammi_db::catalog::jobs_repo::JobRecord,
) {
    assert!(after.lease_expires_at.is_none(), "{after:?}");
    assert_eq!(after.releases, before.releases + 1, "{after:?}");
    assert!(
        after.updated_at > before.updated_at,
        "updated_at must be bumped: before {:?}, after {:?}",
        before.updated_at,
        after.updated_at
    );
    assert_eq!(after.job_id, before.job_id);
    assert_eq!(after.kind, before.kind);
    assert_eq!(after.tenant_id, before.tenant_id);
    assert_eq!(after.status, before.status);
    assert_eq!(after.execution, before.execution);
    assert_eq!(after.spec, before.spec);
    assert_eq!(after.partial_result, before.partial_result);
    assert_eq!(after.result, before.result);
    assert_eq!(after.error, before.error);
    assert_eq!(after.progress_rows_done, before.progress_rows_done);
    assert_eq!(after.progress_rows_total, before.progress_rows_total);
    assert_eq!(after.progress_phase, before.progress_phase);
    assert_eq!(after.cancel_requested, before.cancel_requested);
    assert_eq!(after.model_ref, before.model_ref);
    assert_eq!(after.output_model_id, before.output_model_id);
    assert_eq!(after.model_source, before.model_source);
    assert_eq!(after.claimed_by, before.claimed_by);
    assert_eq!(
        after.attempts, before.attempts,
        "release never touches attempts"
    );
    assert_eq!(after.priority, before.priority);
    assert_eq!(after.claimable, before.claimable);
    assert_eq!(after.acceleration_report, before.acceleration_report);
    assert_eq!(after.created_at, before.created_at);
}

/// The sweep form's (`release_jobs_claimed_by`) OWN column-level delta,
/// exactly, on EVERY row it matches — its twin
/// `release_job_lease_write_is_exactly_lease_null_releases_plus_one_and_timestamp`
/// pins the same SQL (`jobs_repo.rs`: `lease_expires_at = NULL, releases =
/// releases + 1, updated_at = $now`) for the single-row statement, but until
/// now this statement — the ONLY writer on the worker-less RELEASE arm (no
/// loop, so `release_job_holds`'s per-hold pass never touches an inline
/// row's `Job` hold either) — had only a row-COUNT oracle
/// (`a_double_release_increments_releases_once` et al.), never a
/// column-level one. Two rows claimed by `me` are swept in one call; an
/// inline row claimed by `me` (never matches `execution = 'queued'`) and a
/// row claimed by a DIFFERENT instance are both asserted COMPLETELY
/// untouched, not merely their lease/`releases`.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_jobs_claimed_by_write_is_exactly_lease_null_releases_plus_one_and_timestamp(
    backend: BackendKind,
) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());
    let lease = Duration::from_secs(3600);

    catalog.submit_job(job_params("sweep-a")).await.unwrap();
    catalog.submit_job(job_params("sweep-b")).await.unwrap();
    catalog
        .submit_job(inline_job_params("sweep-inline"))
        .await
        .unwrap();
    catalog.submit_job(job_params("sweep-other")).await.unwrap();

    catalog
        .claim_next("me", KINDS, lease)
        .await
        .unwrap()
        .expect("sweep-a claimed by me");
    catalog
        .claim_next("me", KINDS, lease)
        .await
        .unwrap()
        .expect("sweep-b claimed by me");
    catalog
        .claim_by_id("sweep-inline", "me", lease)
        .await
        .unwrap()
        .expect("sweep-inline claimed by me");
    catalog
        .claim_next("someone-else", KINDS, lease)
        .await
        .unwrap()
        .expect("sweep-other claimed by a DIFFERENT instance");

    // A timestamp column stored with second-or-finer resolution needs a
    // strictly later `now` to observe as "bumped" — see the twin oracle's
    // identical comment.
    tokio::time::sleep(Duration::from_millis(1100)).await;

    let a_before = catalog.get_job("sweep-a").await.unwrap();
    let b_before = catalog.get_job("sweep-b").await.unwrap();
    let inline_before = catalog.get_job("sweep-inline").await.unwrap();
    let other_before = catalog.get_job("sweep-other").await.unwrap();
    assert!(a_before.lease_expires_at.is_some());
    assert!(b_before.lease_expires_at.is_some());

    let swept = catalog.release_jobs_claimed_by("me").await.unwrap();
    assert_eq!(swept, 2, "exactly the two queued rows claimed by `me`");

    let a_after = catalog.get_job("sweep-a").await.unwrap();
    let b_after = catalog.get_job("sweep-b").await.unwrap();
    assert_only_release_columns_changed(&a_before, &a_after);
    assert_only_release_columns_changed(&b_before, &b_after);

    // The inline row claimed by the SAME instance: completely untouched
    // (`execution = 'queued'` never matches it).
    let inline_after = catalog.get_job("sweep-inline").await.unwrap();
    assert_eq!(
        inline_after.lease_expires_at,
        inline_before.lease_expires_at
    );
    assert_eq!(inline_after.releases, inline_before.releases);
    assert_eq!(inline_after.updated_at, inline_before.updated_at);
    assert_eq!(inline_after.claimed_by, inline_before.claimed_by);
    assert_eq!(inline_after.attempts, inline_before.attempts);
    assert_eq!(inline_after.status, inline_before.status);

    // A row claimed by a DIFFERENT instance: completely untouched.
    let other_after = catalog.get_job("sweep-other").await.unwrap();
    assert_eq!(other_after.lease_expires_at, other_before.lease_expires_at);
    assert_eq!(other_after.releases, other_before.releases);
    assert_eq!(other_after.updated_at, other_before.updated_at);
    assert_eq!(other_after.claimed_by, other_before.claimed_by);
    assert_eq!(other_after.attempts, other_before.attempts);
    assert_eq!(other_after.status, other_before.status);

    // Idempotent: a second sweep matches neither row again.
    assert_eq!(catalog.release_jobs_claimed_by("me").await.unwrap(), 0);
}

/// The reclaim cap compares `attempts - releases` against `MAX_ATTEMPTS`
/// (3): a deploy storm of three releases leaves the job claimable at
/// `attempts 4, releases 3`; three genuine expiries fail it; 3 claims / 2
/// releases with the third lease expired is requeued, not failed.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reclaim_cap_counts_attempts_minus_releases(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());
    const MAX: u32 = 3;
    let lease = Duration::from_secs(3600);

    // Three releases: attempts 4, releases 3, still claimable.
    catalog.submit_job(job_params("storm")).await.unwrap();
    for round in 1..=3u32 {
        let claimed = catalog
            .claim_next("me", KINDS, lease)
            .await
            .unwrap()
            .expect("storm claimed");
        assert_eq!(claimed.attempts, round);
        assert!(catalog
            .release_job_lease("storm", "me", round)
            .await
            .unwrap());
        assert_eq!(catalog.reclaim_expired_jobs(lease, MAX).await.unwrap(), 1);
        let row = catalog.get_job("storm").await.unwrap();
        assert_eq!(
            row.status,
            JobStatus::Queued.to_string(),
            "release {round} must leave the job claimable, never failed"
        );
        assert_eq!(row.releases, round);
    }
    let fourth = catalog
        .claim_next("me", KINDS, lease)
        .await
        .unwrap()
        .expect("still claimable after three releases");
    assert_eq!((fourth.attempts, fourth.releases), (4, 3));
    // Park it terminal so the sweeps below never see it.
    assert!(catalog
        .finish_job(FinishJobParams {
            job_id: "storm",
            instance_id: "me",
            attempts: 4,
            result: "{}",
        })
        .await
        .unwrap());

    // Three genuine expiries (no releases) still fail the job.
    catalog.submit_job(job_params("exp")).await.unwrap();
    for round in 1..=2u32 {
        let claimed = catalog
            .claim_next("me", KINDS, Duration::from_secs(0))
            .await
            .unwrap()
            .expect("exp claimed");
        assert_eq!(claimed.attempts, round);
        assert_eq!(
            catalog
                .reclaim_expired_jobs(Duration::from_secs(0), MAX)
                .await
                .unwrap(),
            1
        );
        assert_eq!(
            catalog.get_job("exp").await.unwrap().status,
            JobStatus::Queued.to_string()
        );
    }
    let third = catalog
        .claim_next("me", KINDS, Duration::from_secs(0))
        .await
        .unwrap()
        .expect("exp claimed a third time");
    assert_eq!(third.attempts, 3);
    assert_eq!(
        catalog
            .reclaim_expired_jobs(Duration::from_secs(0), MAX)
            .await
            .unwrap(),
        1
    );
    let failed = catalog.get_job("exp").await.unwrap();
    assert_eq!(failed.status, JobStatus::Failed.to_string());
    assert_eq!((failed.attempts, failed.releases), (3, 0));

    // 3 claims / 2 releases, the third lease expired: requeued (3 - 2 < 3).
    catalog.submit_job(job_params("mix")).await.unwrap();
    for round in 1..=2u32 {
        let claimed = catalog
            .claim_next("me", KINDS, lease)
            .await
            .unwrap()
            .expect("mix claimed");
        assert_eq!(claimed.attempts, round);
        assert!(catalog.release_job_lease("mix", "me", round).await.unwrap());
        assert_eq!(catalog.reclaim_expired_jobs(lease, MAX).await.unwrap(), 1);
    }
    let third = catalog
        .claim_next("me", KINDS, Duration::from_secs(0))
        .await
        .unwrap()
        .expect("mix claimed a third time");
    assert_eq!((third.attempts, third.releases), (3, 2));
    assert_eq!(
        catalog
            .reclaim_expired_jobs(Duration::from_secs(0), MAX)
            .await
            .unwrap(),
        1
    );
    let mixed = catalog.get_job("mix").await.unwrap();
    assert_eq!(
        mixed.status,
        JobStatus::Queued.to_string(),
        "attempts - releases = 1 < 3: requeued, never failed"
    );
    assert_eq!((mixed.attempts, mixed.releases), (3, 2));
}

/// Every release statement is idempotent by `lease_expires_at IS NOT NULL`:
/// a second `release_job_lease` matches zero rows and `releases` stays 1; a
/// `release_jobs_claimed_by` sweep after either matches 0 rows; and an
/// inline row claimed by the same instance is never released by either
/// (both carry `execution = 'queued'`).
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_double_release_increments_releases_once(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());
    let lease = Duration::from_secs(3600);

    catalog.submit_job(job_params("dbl")).await.unwrap();
    catalog
        .submit_job(inline_job_params("dbl-inline"))
        .await
        .unwrap();
    let claimed = catalog
        .claim_next("me", KINDS, lease)
        .await
        .unwrap()
        .expect("dbl claimed");
    let inline = catalog
        .claim_by_id("dbl-inline", "me", lease)
        .await
        .unwrap()
        .expect("inline claimed");
    assert!(inline.lease_expires_at.is_some());

    assert!(catalog
        .release_job_lease("dbl", "me", claimed.attempts)
        .await
        .unwrap());
    assert!(
        !catalog
            .release_job_lease("dbl", "me", claimed.attempts)
            .await
            .unwrap(),
        "the second release matches zero rows"
    );
    assert_eq!(catalog.get_job("dbl").await.unwrap().releases, 1);
    assert_eq!(
        catalog.release_jobs_claimed_by("me").await.unwrap(),
        0,
        "a sweep after a per-row release matches nothing"
    );

    // The inline row: neither statement touches it.
    assert!(
        !catalog
            .release_job_lease("dbl-inline", "me", inline.attempts)
            .await
            .unwrap(),
        "an inline row is never released"
    );
    let inline_after = catalog.get_job("dbl-inline").await.unwrap();
    assert!(inline_after.lease_expires_at.is_some());
    assert_eq!(inline_after.releases, 0);

    // The sweep form on a fresh claim: exactly one row, then zero.
    catalog.submit_job(job_params("dbl-2")).await.unwrap();
    catalog
        .claim_next("me", KINDS, lease)
        .await
        .unwrap()
        .expect("dbl-2 claimed");
    assert_eq!(catalog.release_jobs_claimed_by("me").await.unwrap(), 1);
    assert_eq!(catalog.release_jobs_claimed_by("me").await.unwrap(), 0);
    assert_eq!(catalog.get_job("dbl-2").await.unwrap().releases, 1);
    assert!(
        catalog
            .get_job("dbl-inline")
            .await
            .unwrap()
            .lease_expires_at
            .is_some(),
        "the sweep leaves the inline row's lease live"
    );
}

/// `clear_partial_result` is the attempt-guarded inverse of
/// `create_result_table`'s `partial_result` CAS: after the successor's
/// claim-and-fail arm clears the column, the SAME attempt's own
/// `create_result_table` records its fresh table instead of landing
/// `JobAttemptSuperseded` (the pre-existing defect the 68 README records,
/// escape `esc-110`). A stale attempt or a different table name clears
/// nothing.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn clear_partial_result_lets_the_next_attempt_record_its_own_table(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());
    let suffix = run_suffix();
    let t1 = format!("cpr_t1_{suffix}");
    let t2 = format!("cpr_t2_{suffix}");

    catalog.submit_job(compute_job_params("cpr")).await.unwrap();
    let claimed = catalog
        .claim_next("me", &["embedding"], Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("cpr claimed");
    let attempt = JobAttempt {
        job_id: "cpr",
        instance_id: "me",
        attempts: claimed.attempts,
    };
    catalog
        .create_result_table(result_table_params(&t1, attempt))
        .await
        .unwrap();
    assert_eq!(
        catalog
            .get_job("cpr")
            .await
            .unwrap()
            .partial_result
            .as_deref(),
        Some(t1.as_str())
    );

    // Guards: a stale attempt, or the wrong table, clears nothing.
    assert!(!catalog
        .clear_partial_result("cpr", "me", claimed.attempts + 1, &t1)
        .await
        .unwrap());
    assert!(!catalog
        .clear_partial_result("cpr", "me", claimed.attempts, &t2)
        .await
        .unwrap());
    assert_eq!(
        catalog
            .get_job("cpr")
            .await
            .unwrap()
            .partial_result
            .as_deref(),
        Some(t1.as_str()),
        "a guarded miss leaves the column"
    );

    assert!(catalog
        .clear_partial_result("cpr", "me", claimed.attempts, &t1)
        .await
        .unwrap());
    assert!(catalog
        .get_job("cpr")
        .await
        .unwrap()
        .partial_result
        .is_none());
    // Idempotent: a second clear matches zero rows.
    assert!(!catalog
        .clear_partial_result("cpr", "me", claimed.attempts, &t1)
        .await
        .unwrap());

    // The same attempt now records its own fresh table.
    catalog
        .create_result_table(result_table_params(&t2, attempt))
        .await
        .expect("the cleared column lets the attempt record its next table");
    assert_eq!(
        catalog
            .get_job("cpr")
            .await
            .unwrap()
            .partial_result
            .as_deref(),
        Some(t2.as_str())
    );
}

/// The finalize CAS is `claimed_by / status / attempts`, never the lease
/// (`jobs_repo.rs` finish/fail): a released row can still be finished by
/// its old holder — the named library-only divergence (D19), pinned here as
/// a fact rather than left implicit.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finalize_cas_still_matches_a_released_lease(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());
    let lease = Duration::from_secs(3600);

    // `finish_job` (the compute finalize).
    catalog
        .submit_job(compute_job_params("fin-c"))
        .await
        .unwrap();
    let c = catalog
        .claim_next("me", &["embedding"], lease)
        .await
        .unwrap()
        .expect("fin-c claimed");
    assert!(catalog
        .release_job_lease("fin-c", "me", c.attempts)
        .await
        .unwrap());
    assert!(
        catalog
            .finish_job(FinishJobParams {
                job_id: "fin-c",
                instance_id: "me",
                attempts: c.attempts,
                result: "{}",
            })
            .await
            .unwrap(),
        "the finalize CAS ignores the lease"
    );
    let done = catalog.get_job("fin-c").await.unwrap();
    assert_eq!(done.status, JobStatus::Completed.to_string());
    assert_eq!(done.releases, 1);

    // `finish_job_with_model` (the training finalize).
    let model = format!("jammi:fine-tuned:fin-{}", run_suffix());
    catalog
        .register_model(RegisterModelParams {
            model_id: &model,
            version: 1,
            model_type: "fine-tuned",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: Some("q-base::1"),
            artifact_path: None,
            config_json: None,
        })
        .await
        .unwrap();
    catalog
        .submit_job(SubmitJobParams {
            output_model_id: Some(&model),
            ..job_params("fin-t")
        })
        .await
        .unwrap();
    let t = catalog
        .claim_next("me", KINDS, lease)
        .await
        .unwrap()
        .expect("fin-t claimed");
    assert!(catalog
        .release_job_lease("fin-t", "me", t.attempts)
        .await
        .unwrap());
    assert!(
        catalog
            .finish_job_with_model(FinishJobWithModelParams {
                job_id: "fin-t",
                instance_id: "me",
                attempts: t.attempts,
                result: "{}",
                output_model_id: &model,
                output_model_version: 1,
                artifact_path: "file:///artifacts/fin-t/me/1",
                epoch_checkpoints: &[],
            })
            .await
            .unwrap(),
        "finish_job_with_model's CAS ignores the lease too"
    );
    assert_eq!(
        catalog.get_job("fin-t").await.unwrap().status,
        JobStatus::Completed.to_string()
    );
}

/// The gauge sample query counts `execution = 'queued'` rows in `queued` /
/// `running` per kind — inline and terminal rows excluded, a held
/// (`claimable = false`) row still counted as queued — and returns them
/// sorted in Rust.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn count_jobs_by_kind_status_matches_row_counts(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());
    let lease = Duration::from_secs(3600);

    assert!(
        catalog
            .count_jobs_by_kind_status()
            .await
            .unwrap()
            .is_empty(),
        "an empty queue samples to no rows"
    );

    catalog.submit_job(job_params("cnt-ft-1")).await.unwrap();
    catalog.submit_job(job_params("cnt-ft-2")).await.unwrap();
    catalog.submit_job(job_params("cnt-ft-held")).await.unwrap();
    set_claim_policy(&catalog, "cnt-ft-held", 0, false).await;
    catalog
        .submit_job(compute_job_params("cnt-emb"))
        .await
        .unwrap();
    catalog
        .submit_job(inline_job_params("cnt-inline"))
        .await
        .unwrap();
    catalog.submit_job(job_params("cnt-done")).await.unwrap();

    // cnt-ft-1 running; cnt-done completed; cnt-inline claimed inline.
    let running = catalog
        .claim_next("me", KINDS, lease)
        .await
        .unwrap()
        .expect("cnt-ft-1 claimed");
    assert_eq!(running.job_id, "cnt-ft-1");
    let done = catalog
        .claim_next("me", KINDS, lease)
        .await
        .unwrap()
        .expect("cnt-ft-2 claimed");
    assert_eq!(done.job_id, "cnt-ft-2");
    assert!(catalog
        .finish_job(FinishJobParams {
            job_id: "cnt-ft-2",
            instance_id: "me",
            attempts: 1,
            result: "{}",
        })
        .await
        .unwrap());
    catalog
        .claim_by_id("cnt-inline", "me", lease)
        .await
        .unwrap()
        .expect("inline claimed");

    let counts = catalog.count_jobs_by_kind_status().await.unwrap();
    assert_eq!(
        counts,
        vec![
            ("embedding".to_string(), "queued".to_string(), 1),
            // cnt-done (queued) + cnt-ft-held (queued, claimable = false).
            ("fine_tune".to_string(), "queued".to_string(), 2),
            ("fine_tune".to_string(), "running".to_string(), 1),
        ],
        "queued/running counts per kind over execution = 'queued' rows only"
    );
}

/// `workers.state` round-trips through `upsert_worker` / `set_worker_state`
/// / `list_workers` in the `warming -> claiming -> draining` order the loop
/// task writes it; a state write on an absent row matches nothing.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn set_worker_state_round_trips_through_list_workers(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());
    let state_of = |workers: Vec<jammi_db::catalog::jobs_repo::WorkerRecord>| {
        workers
            .into_iter()
            .find(|w| w.instance_id == "w-state")
            .map(|w| w.state)
    };

    catalog
        .upsert_instance("w-state", Some("lbl"), Some("host"))
        .await
        .unwrap();
    catalog
        .upsert_worker("w-state", "all", WorkerState::Warming)
        .await
        .unwrap();
    assert_eq!(
        state_of(catalog.list_workers().await.unwrap()).as_deref(),
        Some("warming")
    );
    assert!(catalog
        .set_worker_state("w-state", WorkerState::Claiming)
        .await
        .unwrap());
    assert_eq!(
        state_of(catalog.list_workers().await.unwrap()).as_deref(),
        Some("claiming")
    );
    assert!(catalog
        .set_worker_state("w-state", WorkerState::Draining)
        .await
        .unwrap());
    assert_eq!(
        state_of(catalog.list_workers().await.unwrap()).as_deref(),
        Some("draining")
    );
    // A re-upsert (a restarted loop on the same instance) resets the state.
    catalog
        .upsert_worker("w-state", "fine_tune", WorkerState::Warming)
        .await
        .unwrap();
    let again = catalog
        .list_workers()
        .await
        .unwrap()
        .into_iter()
        .find(|w| w.instance_id == "w-state")
        .expect("row present");
    assert_eq!(
        (again.state.as_str(), again.kinds.as_str()),
        ("warming", "fine_tune")
    );
    assert!(
        !catalog
            .set_worker_state("w-absent", WorkerState::Claiming)
            .await
            .unwrap(),
        "no row, no state write"
    );
    assert!(catalog.delete_worker("w-state").await.unwrap());
}
