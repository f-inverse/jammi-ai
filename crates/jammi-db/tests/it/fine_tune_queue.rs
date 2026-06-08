//! Lease-based training-job queue primitives on the `training_jobs` catalog
//! table: atomic claim, lease heartbeat, and expired-lease reclaim.

use std::sync::Arc;
use std::time::Duration;

use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::status::TrainingJobStatus;
use jammi_db::catalog::training_repo::CreateTrainingJobParams;
use jammi_db::catalog::Catalog;
use jammi_db::model_task::ModelTask;
use tempfile::tempdir;

/// A minimal queued fine-tune job over the `q-base` model with the given id.
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

/// Two concurrent claims against a single queued job: exactly one wins; the
/// loser sees an empty queue. The winner's record is `running`, leased to it,
/// and `attempts` is incremented to 1.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_claim_grants_one_winner() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());
    register_base_model(&catalog).await;

    catalog
        .create_training_job(job_params("q-1"))
        .await
        .unwrap();

    let c1 = Arc::clone(&catalog);
    let c2 = Arc::clone(&catalog);
    let lease = Duration::from_secs(30);
    let (a, b) = tokio::join!(
        async move { c1.claim_next_training_job("worker-a", lease).await.unwrap() },
        async move { c2.claim_next_training_job("worker-b", lease).await.unwrap() },
    );

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
#[tokio::test]
async fn claim_returns_oldest_queued_job_first() {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).await.unwrap();
    register_base_model(&catalog).await;

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
#[tokio::test]
async fn heartbeat_renews_for_owner_only() {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).await.unwrap();
    register_base_model(&catalog).await;

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
#[tokio::test]
async fn reclaim_requeues_then_fails_when_attempts_exhausted() {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).await.unwrap();
    register_base_model(&catalog).await;

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
#[tokio::test]
async fn finalize_is_a_lease_guarded_compare_and_set() {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).await.unwrap();
    register_base_model(&catalog).await;

    catalog.create_training_job(job_params("fz")).await.unwrap();

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
        .finalize_training_job("fz", "worker-a", "jammi:fine-tuned:fz", Some(r#"{"k":1}"#))
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

    // worker-b (the live owner) finalizes: one row matches, the job completes
    // with the output model and the metrics recorded.
    let b_finalized = catalog
        .finalize_training_job(
            "fz",
            "worker-b",
            "jammi:fine-tuned:fz",
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

    // A second finalize by the same owner is now a no-op (status is no longer
    // running), so finalize is not re-runnable once terminal.
    let again = catalog
        .finalize_training_job("fz", "worker-b", "jammi:fine-tuned:fz", None)
        .await
        .unwrap();
    assert!(!again, "a completed job cannot be finalized again");
}

/// The terminal failure write is lease-guarded the same way as finalize: only
/// the worker that still holds the lease can stamp `failed`. A stale worker
/// cannot mark `failed` a job the re-claiming worker is running (which would
/// otherwise block that worker's finalize).
#[tokio::test]
async fn fail_is_a_lease_guarded_compare_and_set() {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).await.unwrap();
    register_base_model(&catalog).await;

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
#[tokio::test]
async fn reclaim_leaves_live_leases_untouched() {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).await.unwrap();
    register_base_model(&catalog).await;

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
