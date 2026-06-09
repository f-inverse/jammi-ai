//! Lease-based training-job queue primitives on the `training_jobs` catalog
//! table: atomic claim, lease heartbeat, and expired-lease reclaim.
//!
//! Every test is parameterised over [`BackendKind`] via `test_case` +
//! `cfg_attr`. The SQLite lane is always generated; the Postgres lane is
//! generated only when the `live-postgres-tests` feature is on, and skips at
//! runtime when `JAMMI_TEST_PG_URL` is unset. The Postgres lane exercises the
//! `FOR UPDATE SKIP LOCKED` claim path and the global expired-lease reclaim
//! scan that the SQLite serialized-UPDATE path cannot.
//!
//! The claim and reclaim queries scan `training_jobs` globally (they are not
//! tenant- or id-scoped — a worker takes the oldest queued job across the whole
//! table, and reclaim sweeps every expired lease). On the Postgres lane that
//! single table is shared across the whole test run, so each test first clears
//! it via [`reset_queue`] to start from a known-empty queue. CI's `test-pg` job
//! runs the Postgres lane with `--test-threads=1`, so the reset-then-populate
//! sequence is serialised and cannot race a sibling test.

use std::sync::Arc;
use std::time::Duration;

use jammi_db::catalog::backend::{BackendKind, TxOptions};
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::status::TrainingJobStatus;
use jammi_db::catalog::training_repo::CreateTrainingJobParams;
use jammi_db::catalog::Catalog;
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

/// A minimal queued training job over the `q-base` model with the given id.
fn job_params(job_id: &str) -> CreateTrainingJobParams<'_> {
    CreateTrainingJobParams {
        job_id,
        base_model_id: "q-base::1",
        training_source: "src.csv",
        loss_type: "contrastive",
        hyperparams: "{}",
        kind: "fine_tune",
        training_spec: "{}",
    }
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

/// Clear every row from `training_jobs` so the global claim/reclaim scans see
/// only the rows this test creates. Needed because the Postgres lane shares one
/// catalog DB across the run; the SQLite lane has a fresh tempdir per test but
/// running the reset there too keeps both lanes on one path. Run under
/// `--test-threads=1` on the Postgres lane, so it cannot race a sibling test.
async fn reset_queue(catalog: &Catalog) {
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move { tx.execute("DELETE FROM training_jobs", &[]).await })
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

/// Two concurrent claims against a single queued job run on separate tasks of a
/// multi-thread runtime: exactly one wins, the other sees an empty queue. The
/// winner's record is `running`, leased to it, and `attempts` is incremented to
/// 1. Spawning the claims as distinct tasks (rather than `tokio::join!`, which
/// interleaves two futures on one task deterministically) puts the Postgres
/// `FOR UPDATE SKIP LOCKED` path under real lock contention.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_claim_grants_one_winner(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog
        .create_training_job(job_params("q-1"))
        .await
        .unwrap();

    let c1 = Arc::clone(&catalog);
    let c2 = Arc::clone(&catalog);
    let lease = Duration::from_secs(30);
    let h1 =
        tokio::spawn(async move { c1.claim_next_training_job("worker-a", lease).await.unwrap() });
    let h2 =
        tokio::spawn(async move { c2.claim_next_training_job("worker-b", lease).await.unwrap() });
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
    assert_eq!(claimed.status, TrainingJobStatus::Running.to_string());
    assert!(
        matches!(claimed.claimed_by.as_deref(), Some("worker-a" | "worker-b")),
        "claimed_by must name the winning worker, got {:?}",
        claimed.claimed_by
    );
    assert!(claimed.lease_expires_at.is_some(), "lease must be stamped");
    assert_eq!(claimed.attempts, 1, "first claim sets attempts to 1");

    // The queue is now empty: a third claim returns None.
    let empty = catalog
        .claim_next_training_job("worker-c", lease)
        .await
        .unwrap();
    assert!(empty.is_none(), "no queued job remains after the claim");
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

    catalog
        .create_training_job(job_params("old"))
        .await
        .unwrap();
    // A distinct, strictly-later created_at so ORDER BY is unambiguous.
    tokio::time::sleep(Duration::from_millis(1100)).await;
    catalog
        .create_training_job(job_params("new"))
        .await
        .unwrap();

    let first = catalog
        .claim_next_training_job("w", Duration::from_secs(30))
        .await
        .unwrap()
        .expect("a job is queued");
    assert_eq!(first.job_id, "old", "oldest queued job is claimed first");
}

/// Heartbeat renews the lease for the owning worker and refuses everyone else.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn heartbeat_renews_for_owner_only(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.create_training_job(job_params("hb")).await.unwrap();
    let claimed = catalog
        .claim_next_training_job("owner", Duration::from_secs(5))
        .await
        .unwrap()
        .expect("job claimed");
    let first_lease = claimed.lease_expires_at.clone().unwrap();

    // A non-owner cannot renew the lease.
    let stolen = catalog
        .heartbeat_training_job("hb", "intruder", Duration::from_secs(60))
        .await
        .unwrap();
    assert!(!stolen, "a non-owner must not renew the lease");

    // The owner renews; the new deadline is later than the original.
    tokio::time::sleep(Duration::from_millis(10)).await;
    let renewed = catalog
        .heartbeat_training_job("hb", "owner", Duration::from_secs(120))
        .await
        .unwrap();
    assert!(renewed, "the owner renews its own lease");
    let after = catalog.get_training_job("hb").await.unwrap();
    assert!(
        after.lease_expires_at.as_deref().unwrap() > first_lease.as_str(),
        "renewed lease must extend past the original deadline"
    );

    // Once the job leaves `running`, even the owner cannot heartbeat it.
    catalog
        .update_training_status("hb", TrainingJobStatus::Completed, None)
        .await
        .unwrap();
    let post_complete = catalog
        .heartbeat_training_job("hb", "owner", Duration::from_secs(60))
        .await
        .unwrap();
    assert!(
        !post_complete,
        "a job that is no longer running cannot be heartbeat"
    );
}

/// An expired lease with attempts left re-queues the job (clearing the lease);
/// once attempts are exhausted the job fails with the reason recorded.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reclaim_requeues_then_fails_when_attempts_exhausted(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.create_training_job(job_params("rc")).await.unwrap();

    // Claim with a zero lease so it is already expired by the time reclaim
    // runs — a deterministic forced expiry, no sleep needed.
    let claimed = catalog
        .claim_next_training_job("worker", Duration::from_secs(0))
        .await
        .unwrap()
        .expect("job claimed");
    assert_eq!(claimed.attempts, 1);

    // attempts (1) < max (2): the job is re-queued and its lease cleared.
    let actioned = catalog.reclaim_expired_training_jobs(2).await.unwrap();
    assert_eq!(actioned, 1, "the one expired lease is re-queued");
    let requeued = catalog.get_training_job("rc").await.unwrap();
    assert_eq!(requeued.status, TrainingJobStatus::Queued.to_string());
    assert!(requeued.claimed_by.is_none(), "re-queue clears claimed_by");
    assert!(
        requeued.lease_expires_at.is_none(),
        "re-queue clears the lease deadline"
    );

    // Claim again (attempts -> 2), expire again, reclaim with max=2: now
    // attempts (2) >= max (2), so the job fails with the reason recorded.
    let reclaimed = catalog
        .claim_next_training_job("worker", Duration::from_secs(0))
        .await
        .unwrap()
        .expect("re-queued job claimable");
    assert_eq!(reclaimed.attempts, 2);

    let actioned = catalog.reclaim_expired_training_jobs(2).await.unwrap();
    assert_eq!(actioned, 1, "the exhausted lease is actioned once");
    let failed = catalog.get_training_job("rc").await.unwrap();
    assert_eq!(failed.status, TrainingJobStatus::Failed.to_string());
    assert!(
        failed.lease_expires_at.is_none(),
        "a failed job carries no live lease"
    );
    assert!(
        failed
            .error_message
            .as_deref()
            .is_some_and(|m| m.contains("lease expired")),
        "the failure records the lease-exhaustion reason, got {:?}",
        failed.error_message
    );
}

/// The terminal finalize is a lease-guarded compare-and-set: only the worker
/// that still holds the lease (`claimed_by` + `running`) finalizes the job. A
/// worker whose lease was reclaimed by another matches zero rows and does not
/// write the output model or flip the status — the guard that stops two workers
/// from both finalizing one job.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finalize_is_a_lease_guarded_compare_and_set(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.create_training_job(job_params("fz")).await.unwrap();
    // Register the output model row with no served path: the served
    // `artifact_path` must be committed solely by the winning finalize CAS, so
    // it starts NULL and only the live owner's finalize sets it.
    catalog
        .register_model(RegisterModelParams {
            model_id: "jammi:fine-tuned:fz",
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
    // reclaims it via the requeue path and re-claims — worker-b now owns it.
    catalog
        .claim_next_training_job("worker-a", Duration::from_secs(0))
        .await
        .unwrap()
        .expect("worker-a claims the job");
    let actioned = catalog.reclaim_expired_training_jobs(5).await.unwrap();
    assert_eq!(actioned, 1, "the expired lease is re-queued");
    let owned_by_b = catalog
        .claim_next_training_job("worker-b", Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("worker-b re-claims the requeued job");
    assert_eq!(owned_by_b.claimed_by.as_deref(), Some("worker-b"));

    // worker-a (the stale owner) tries to finalize: zero rows match, so it does
    // not finalize and the job is untouched.
    let a_finalized = catalog
        .finalize_training_job(
            "fz",
            "worker-a",
            "jammi:fine-tuned:fz",
            "file:///artifacts/fz/worker-a/2",
            Some(r#"{"k":1}"#),
        )
        .await
        .unwrap();
    assert!(
        !a_finalized,
        "a worker that lost its lease must not finalize the job"
    );
    let after_a = catalog.get_training_job("fz").await.unwrap();
    assert_eq!(
        after_a.status,
        TrainingJobStatus::Running.to_string(),
        "the stale worker's CAS leaves the job running"
    );
    assert!(
        after_a.output_model_id.is_none(),
        "the stale worker writes no output model"
    );
    let model_after_a = catalog
        .get_model("jammi:fine-tuned:fz")
        .await
        .unwrap()
        .expect("the output model row exists");
    assert!(
        model_after_a.artifact_path.is_none(),
        "the stale worker's failed CAS commits no served path; it stays NULL, \
         got {:?}",
        model_after_a.artifact_path
    );

    // worker-b (the live owner) finalizes: one row matches, the job completes
    // with the output model and the metrics recorded.
    let b_finalized = catalog
        .finalize_training_job(
            "fz",
            "worker-b",
            "jammi:fine-tuned:fz",
            "file:///artifacts/fz/worker-b/3",
            Some(r#"{"completed_at":"2026-01-01T00:00:00Z"}"#),
        )
        .await
        .unwrap();
    assert!(b_finalized, "the lease owner finalizes the job");
    let after_b = catalog.get_training_job("fz").await.unwrap();
    assert_eq!(after_b.status, TrainingJobStatus::Completed.to_string());
    assert_eq!(
        after_b.output_model_id.as_deref(),
        Some("jammi:fine-tuned:fz")
    );
    assert_eq!(
        after_b.completed_at.as_deref(),
        Some("2026-01-01T00:00:00Z"),
        "the finalize records the run metrics"
    );
    let model_after_b = catalog
        .get_model("jammi:fine-tuned:fz")
        .await
        .unwrap()
        .expect("the output model row exists");
    assert_eq!(
        model_after_b.artifact_path.as_deref(),
        Some("file:///artifacts/fz/worker-b/3"),
        "the winning finalize CAS commits the live owner's prefix as the served \
         path — the sole writer of the committed pointer"
    );

    // A second finalize by the same owner is now a no-op (status is no longer
    // running), so finalize is not re-runnable once terminal.
    let again = catalog
        .finalize_training_job(
            "fz",
            "worker-b",
            "jammi:fine-tuned:fz",
            "file:///artifacts/fz/worker-b/3",
            None,
        )
        .await
        .unwrap();
    assert!(!again, "a completed job cannot be finalized again");
}

/// The terminal failure write is lease-guarded the same way as finalize: only
/// the worker that still holds the lease can stamp `failed`. A stale worker
/// cannot mark `failed` a job the re-claiming worker is running (which would
/// otherwise block that worker's finalize).
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fail_is_a_lease_guarded_compare_and_set(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog.create_training_job(job_params("fl")).await.unwrap();

    // worker-a claims (zero lease → expires), worker-b reclaims and owns it.
    catalog
        .claim_next_training_job("worker-a", Duration::from_secs(0))
        .await
        .unwrap()
        .expect("worker-a claims");
    catalog.reclaim_expired_training_jobs(5).await.unwrap();
    catalog
        .claim_next_training_job("worker-b", Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("worker-b re-claims");

    // The stale worker-a cannot fail the job worker-b now owns.
    let a_failed = catalog
        .fail_training_job("fl", "worker-a", Some(r#"{"error_message":"boom"}"#))
        .await
        .unwrap();
    assert!(
        !a_failed,
        "a worker that lost its lease cannot fail the job"
    );
    let after_a = catalog.get_training_job("fl").await.unwrap();
    assert_eq!(
        after_a.status,
        TrainingJobStatus::Running.to_string(),
        "the stale fail leaves the job running for its real owner"
    );

    // The live owner-b records the failure.
    let b_failed = catalog
        .fail_training_job(
            "fl",
            "worker-b",
            Some(r#"{"error_message":"real failure"}"#),
        )
        .await
        .unwrap();
    assert!(b_failed, "the lease owner records the failure");
    let after_b = catalog.get_training_job("fl").await.unwrap();
    assert_eq!(after_b.status, TrainingJobStatus::Failed.to_string());
    assert_eq!(after_b.error_message.as_deref(), Some("real failure"));
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

    catalog
        .create_training_job(job_params("live"))
        .await
        .unwrap();
    catalog
        .claim_next_training_job("worker", Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("job claimed");

    let actioned = catalog.reclaim_expired_training_jobs(5).await.unwrap();
    assert_eq!(actioned, 0, "a live lease is not reclaimed");
    let still_running = catalog.get_training_job("live").await.unwrap();
    assert_eq!(still_running.status, TrainingJobStatus::Running.to_string());
}
