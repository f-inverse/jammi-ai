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

use jammi_db::catalog::backend::{BackendKind, SqlValue, TxOptions};
use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::status::TrainingJobStatus;
use jammi_db::catalog::training_repo::{CreateTrainingJobParams, FinalizeTrainingJobParams};
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

/// Set `priority`/`claimable` on one row via raw SQL, mirroring [`reset_queue`]
/// — the claim-policy columns are catalog data, so the fixture writes them as
/// data rather than through any typed setter (there is none; the engine only
/// honors the columns).
async fn set_claim_policy(catalog: &Catalog, job_id: &str, priority: i64, claimable: bool) {
    let job_id = job_id.to_string();
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE training_jobs SET priority = $1, claimable = $2 WHERE job_id = $3",
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
/// one. Spawning the claims as distinct tasks (rather than `tokio::join!`, which
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

/// `priority` breaks the tie ahead of `created_at`: a younger high-priority job
/// claims before an older default-priority job, and the default-priority job
/// still claims after it.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn claim_honors_priority_over_created_at(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog
        .create_training_job(job_params("low"))
        .await
        .unwrap();
    // A distinct, strictly-later created_at so the FIFO tiebreak would (if
    // priority did not win) pick "low" first.
    tokio::time::sleep(Duration::from_millis(1100)).await;
    catalog
        .create_training_job(job_params("high"))
        .await
        .unwrap();
    set_claim_policy(&catalog, "high", 10, true).await;

    let first = catalog
        .claim_next_training_job("w", Duration::from_secs(30))
        .await
        .unwrap()
        .expect("a job is queued");
    assert_eq!(
        first.job_id, "high",
        "the higher-priority job claims first despite being strictly younger"
    );

    let second = catalog
        .claim_next_training_job("w", Duration::from_secs(30))
        .await
        .unwrap()
        .expect("a job is queued");
    assert_eq!(
        second.job_id, "low",
        "the default-priority job claims once the higher-priority job is gone"
    );
}

/// A `claimable = FALSE` job is skipped by the claim, not errored, and stays
/// invisible until flipped back — with no status change either way.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn claim_skips_a_held_job_without_erroring(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog
        .create_training_job(job_params("held"))
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(1100)).await;
    catalog
        .create_training_job(job_params("ready"))
        .await
        .unwrap();
    set_claim_policy(&catalog, "held", 0, false).await;

    let first = catalog
        .claim_next_training_job("w", Duration::from_secs(30))
        .await
        .unwrap()
        .expect("the claimable job is queued");
    assert_eq!(
        first.job_id, "ready",
        "the older held job is skipped, not claimed and not errored"
    );

    let none = catalog
        .claim_next_training_job("w", Duration::from_secs(30))
        .await
        .unwrap();
    assert!(
        none.is_none(),
        "the held job stays invisible to the claim while claimable = FALSE"
    );

    set_claim_policy(&catalog, "held", 0, true).await;
    let released = catalog
        .claim_next_training_job("w", Duration::from_secs(30))
        .await
        .unwrap()
        .expect("the released job is claimable again");
    assert_eq!(released.job_id, "held");
    assert_eq!(
        released.status,
        TrainingJobStatus::Running.to_string(),
        "the hold is released with no other status change along the way"
    );
}

/// A job re-queued by [`Catalog::reclaim_expired_training_jobs`] retains its
/// `priority`/`claimable` values (the row is untouched by reclaim, only its
/// lease fields are cleared) and re-enters the claim ordering accordingly.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reclaimed_job_retains_priority_and_reenters_ordering(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    // "sibling" is the older, default-priority row; "hi" is strictly younger.
    // Only `priority` — never plain `created_at` FIFO — can put "hi" ahead:
    // this is what makes the final assertion discriminate a reclaim that
    // preserves priority from one that silently resets it.
    catalog
        .create_training_job(job_params("sibling"))
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(1100)).await;
    catalog.create_training_job(job_params("hi")).await.unwrap();
    set_claim_policy(&catalog, "hi", 10, true).await;

    // Claim "hi" with a zero lease so it is already expired by the time
    // reclaim runs, then reclaim re-queues it.
    let claimed = catalog
        .claim_next_training_job("worker", Duration::from_secs(0))
        .await
        .unwrap()
        .expect("the high-priority job claims first despite being younger");
    assert_eq!(claimed.job_id, "hi");
    let actioned = catalog.reclaim_expired_training_jobs(5).await.unwrap();
    assert_eq!(actioned, 1, "the expired lease is re-queued");

    // Under contention with the older, default-priority sibling, the
    // re-queued job's priority still wins. If reclaim had reset priority to
    // the default, plain `created_at` FIFO would pick the older "sibling"
    // instead — that is the failure this assertion catches.
    let next = catalog
        .claim_next_training_job("worker", Duration::from_secs(30))
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
/// distinct job: `FOR UPDATE SKIP LOCKED` composes with the new
/// `ORDER BY priority DESC, created_at` — one worker gets the high-priority
/// job, the other gets the remaining job, never the same one and never none
/// while a claimable job is still queued.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_claim_composes_with_priority_ordering(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog
        .create_training_job(job_params("low"))
        .await
        .unwrap();
    catalog
        .create_training_job(job_params("high"))
        .await
        .unwrap();
    set_claim_policy(&catalog, "high", 10, true).await;

    let c1 = Arc::clone(&catalog);
    let c2 = Arc::clone(&catalog);
    let lease = Duration::from_secs(30);
    let h1 =
        tokio::spawn(async move { c1.claim_next_training_job("worker-a", lease).await.unwrap() });
    let h2 =
        tokio::spawn(async move { c2.claim_next_training_job("worker-b", lease).await.unwrap() });
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
    // The owner drives it out of `running` through the lease-guarded terminal
    // transition.
    let failed = catalog
        .fail_training_job("hb", "owner", None)
        .await
        .unwrap();
    assert!(failed, "the owner ends its own running job");
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
    // not finalize and the job is untouched. F5(b): the doc's central claim —
    // "a zombie/lease-lost worker can never register an epoch-checkpoint
    // row" — is exercised here with a NON-EMPTY `epoch_checkpoints`, not the
    // vacuous empty slice the rest of this test's finalize calls use.
    let loser_epoch_rows = [
        jammi_db::catalog::training_repo::EpochCheckpointRow {
            model_id: "jammi:fine-tuned:fz:epoch_0",
            model_type: "fine-tuned",
            task: ModelTask::TextEmbedding,
            base_model_id: Some("q-base"),
            artifact_path: "file:///artifacts/fz/worker-a/2/checkpoints/epoch_0",
        },
        jammi_db::catalog::training_repo::EpochCheckpointRow {
            model_id: "jammi:fine-tuned:fz:epoch_1",
            model_type: "fine-tuned",
            task: ModelTask::TextEmbedding,
            base_model_id: Some("q-base"),
            artifact_path: "file:///artifacts/fz/worker-a/2/checkpoints/epoch_1",
        },
    ];
    let a_finalized = catalog
        .finalize_training_job(FinalizeTrainingJobParams {
            job_id: "fz",
            worker_id: "worker-a",
            output_model_id: "jammi:fine-tuned:fz",
            output_model_version: 1,
            artifact_path: "file:///artifacts/fz/worker-a/2",
            metrics: Some(r#"{"k":1}"#),
            epoch_checkpoints: &loser_epoch_rows,
        })
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
    // ZERO epoch-checkpoint rows registered for the loser's attempt — the
    // insert sits inside the SAME `if job_updated == 1` guard the served-path
    // UPDATE does, so a CAS that matched zero job rows never reaches it.
    for epoch_name in ["jammi:fine-tuned:fz:epoch_0", "jammi:fine-tuned:fz:epoch_1"] {
        assert!(
            catalog.get_model(epoch_name).await.unwrap().is_none(),
            "a lease-lost worker must never register an epoch-checkpoint row \
             ({epoch_name} must not exist)"
        );
    }

    // worker-b (the live owner) finalizes: one row matches, the job completes
    // with the output model and the metrics recorded.
    let b_finalized = catalog
        .finalize_training_job(FinalizeTrainingJobParams {
            job_id: "fz",
            worker_id: "worker-b",
            output_model_id: "jammi:fine-tuned:fz",
            output_model_version: 1,
            artifact_path: "file:///artifacts/fz/worker-b/3",
            metrics: Some(r#"{"completed_at":"2026-01-01T00:00:00Z"}"#),
            epoch_checkpoints: &[],
        })
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
        .finalize_training_job(FinalizeTrainingJobParams {
            job_id: "fz",
            worker_id: "worker-b",
            output_model_id: "jammi:fine-tuned:fz",
            output_model_version: 1,
            artifact_path: "file:///artifacts/fz/worker-b/3",
            metrics: None,
            epoch_checkpoints: &[],
        })
        .await
        .unwrap();
    assert!(!again, "a completed job cannot be finalized again");
}

/// B5 hardening regression (unit 348): before this contract item, the finalize
/// CAS's model-row `UPDATE` matched `WHERE name = $output_model_id` alone — no
/// `version`, no tenant predicate. Two rows sharing that bare NAME (a second
/// VERSION of the same tenant's model, or a DIFFERENT TENANT's row that happens
/// to carry the same name) would BOTH have been clobbered by one worker's
/// `artifact_path`. This pins the fix: three rows share the name "acme/tuned" —
/// (tenant-a, v1), (tenant-a, v2), (tenant-b, v1) — and a tenant-a finalize
/// naming version 1 touches EXACTLY that row.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn finalize_model_update_is_scoped_by_version_and_tenant(backend: BackendKind) {
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

    // Three rows sharing the name "acme/tuned": tenant-a v1, tenant-a v2,
    // tenant-b v1. None carries an artifact_path yet.
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

    // A base model + a queued job under tenant-a, whose output_model_id names
    // the shared "acme/tuned" name.
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
        .create_training_job(CreateTrainingJobParams {
            job_id: "vt-1",
            base_model_id: &base_pk,
            training_source: "src.csv",
            loss_type: "contrastive",
            hyperparams: "{}",
            kind: "fine_tune",
            training_spec: "{}",
        })
        .await
        .unwrap();
    cat_a
        .claim_next_training_job("worker-a", Duration::from_secs(30))
        .await
        .unwrap()
        .expect("the queued job is claimable");

    let finalized = cat_a
        .finalize_training_job(FinalizeTrainingJobParams {
            job_id: "vt-1",
            worker_id: "worker-a",
            output_model_id: "acme/tuned",
            output_model_version: 1,
            artifact_path: "file:///artifacts/vt-1/worker-a/0",
            metrics: None,
            epoch_checkpoints: &[],
        })
        .await
        .unwrap();
    assert!(finalized, "tenant-a's lease holder finalizes the job");

    let a_v1 = cat_a
        .get_model_version("acme/tuned", 1)
        .await
        .unwrap()
        .expect("tenant-a v1 exists");
    assert_eq!(
        a_v1.artifact_path.as_deref(),
        Some("file:///artifacts/vt-1/worker-a/0"),
        "the finalize's OWN (tenant, version) row is the one touched"
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

/// F5(a) (unit 348 audit): the finalize-CAS winner registers ALL retained
/// epoch-checkpoint rows tenant-stamped to the job's own tenant, and those
/// rows are invisible under a DIFFERENT tenant's scope — the same isolation
/// every other tenant-scoped catalog row gets, exercised through both
/// `get_model` (a single-row resolve) and `list_models` (the registry-wide
/// projection).
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn finalize_registers_epoch_checkpoints_tenant_stamped_and_isolated(backend: BackendKind) {
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
        .create_training_job(CreateTrainingJobParams {
            job_id: "ft-tenant-1",
            base_model_id: &base_pk,
            training_source: "src.csv",
            loss_type: "contrastive",
            hyperparams: "{}",
            kind: "fine_tune",
            training_spec: "{}",
        })
        .await
        .unwrap();
    cat_a
        .claim_next_training_job("worker-a", Duration::from_secs(30))
        .await
        .unwrap()
        .expect("the queued job is claimable");

    let epoch_names: Vec<String> = (0..3)
        .map(|n| format!("jammi:fine-tuned:ft-tenant-1:epoch_{n}"))
        .collect();
    let epoch_paths: Vec<String> = (0..3)
        .map(|n| format!("file:///ft-tenant-1/worker-a/0/checkpoints/epoch_{n}"))
        .collect();
    let epoch_rows: Vec<jammi_db::catalog::training_repo::EpochCheckpointRow<'_>> = epoch_names
        .iter()
        .zip(epoch_paths.iter())
        .map(
            |(name, path)| jammi_db::catalog::training_repo::EpochCheckpointRow {
                model_id: name,
                model_type: "fine-tuned",
                task: ModelTask::TextEmbedding,
                base_model_id: Some("acme/base"),
                artifact_path: path,
            },
        )
        .collect();

    let finalized = cat_a
        .finalize_training_job(
            jammi_db::catalog::training_repo::FinalizeTrainingJobParams {
                job_id: "ft-tenant-1",
                worker_id: "worker-a",
                output_model_id: "jammi:fine-tuned:ft-tenant-1",
                output_model_version: 1,
                artifact_path: "file:///ft-tenant-1/worker-a/0",
                metrics: None,
                epoch_checkpoints: &epoch_rows,
            },
        )
        .await
        .unwrap();
    assert!(finalized, "tenant-a's lease holder finalizes the job");

    // All N=3 rows are registered and resolve under tenant-a.
    for name in &epoch_names {
        let row = cat_a
            .get_model(name)
            .await
            .unwrap()
            .unwrap_or_else(|| panic!("{name} must be registered under tenant-a"));
        assert_eq!(row.status, "checkpoint");
    }

    // NONE of them are visible under tenant-b's scope — `get_model` and
    // `list_models` both.
    for name in &epoch_names {
        assert!(
            cat_b.get_model(name).await.unwrap().is_none(),
            "{name} must be invisible under a different tenant's scope"
        );
    }
    let tenant_a_names: std::collections::HashSet<String> = cat_a
        .list_models()
        .await
        .unwrap()
        .into_iter()
        .map(|m| m.model_id)
        .collect();
    for name in &epoch_names {
        assert!(
            tenant_a_names.contains(name),
            "list_models under tenant-a must include {name}"
        );
    }
    let tenant_b_names: std::collections::HashSet<String> = cat_b
        .list_models()
        .await
        .unwrap()
        .into_iter()
        .map(|m| m.model_id)
        .collect();
    for name in &epoch_names {
        assert!(
            !tenant_b_names.contains(name),
            "list_models under tenant-b must NOT include {name}"
        );
    }
}

/// F4 (unit 348 audit): a checkpoint row's catalog NAME can be occupied by an
/// unrelated, pre-existing model row at ANY version — not just the
/// checkpoint's own hardcoded version 1, which a bare `ON CONFLICT(model_id)`
/// would only catch on an EXACT PK match and would otherwise silently no-op
/// with no diagnostic. The finalize CAS must: skip registering that ONE
/// checkpoint row (never clobber the occupying row, never insert a shadowed
/// duplicate), `tracing::warn!` naming the occupied catalog name and the
/// skipped artifact prefix, register every OTHER retained epoch row
/// normally, and still finalize the job successfully — checkpoints are
/// supplementary, never a reason to fail the job.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test]
async fn finalize_skips_a_name_occupied_epoch_checkpoint_and_warns(backend: BackendKind) {
    use std::sync::Mutex;

    use tracing::subscriber::DefaultGuard;
    use tracing_subscriber::fmt::MakeWriter;

    #[derive(Clone)]
    struct BufferWriter(Arc<Mutex<Vec<u8>>>);
    impl std::io::Write for BufferWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl<'w> MakeWriter<'w> for BufferWriter {
        type Writer = BufferWriter;
        fn make_writer(&'w self) -> Self::Writer {
            self.clone()
        }
    }

    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog
        .create_training_job(job_params("epc-1"))
        .await
        .unwrap();
    catalog
        .claim_next_training_job("worker-a", Duration::from_secs(30))
        .await
        .unwrap()
        .expect("the queued job is claimable");

    // A pre-existing, unrelated user model that happens to occupy the exact
    // name the epoch_0 checkpoint would register under — at version 7, a
    // DIFFERENT version than the checkpoint's hardcoded 1 (so a bare
    // `ON CONFLICT(model_id)` — keyed on the FULL pk including version —
    // would never even fire; the two rows would coexist as distinct PKs,
    // and `get_model`'s `ORDER BY version DESC` would silently shadow
    // whichever row has the lower version).
    let occupied_name = "jammi:fine-tuned:epc-1:epoch_0";
    catalog
        .register_model(RegisterModelParams {
            model_id: occupied_name,
            version: 7,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: Some("file:///pre-existing/user/artifact"),
            config_json: None,
        })
        .await
        .unwrap();

    let epoch0_prefix = "file:///epc-1/worker-a/0/checkpoints/epoch_0";
    let epoch1_prefix = "file:///epc-1/worker-a/0/checkpoints/epoch_1";
    let epoch_rows = [
        jammi_db::catalog::training_repo::EpochCheckpointRow {
            model_id: occupied_name,
            model_type: "fine-tuned",
            task: ModelTask::TextEmbedding,
            base_model_id: Some("q-base"),
            artifact_path: epoch0_prefix,
        },
        jammi_db::catalog::training_repo::EpochCheckpointRow {
            model_id: "jammi:fine-tuned:epc-1:epoch_1",
            model_type: "fine-tuned",
            task: ModelTask::TextEmbedding,
            base_model_id: Some("q-base"),
            artifact_path: epoch1_prefix,
        },
    ];

    let buffer = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::fmt()
        .with_writer(BufferWriter(buffer.clone()))
        .with_ansi(false)
        .finish();
    let _guard: DefaultGuard = tracing::subscriber::set_default(subscriber);

    let finalized = catalog
        .finalize_training_job(
            jammi_db::catalog::training_repo::FinalizeTrainingJobParams {
                job_id: "epc-1",
                worker_id: "worker-a",
                output_model_id: "jammi:fine-tuned:epc-1",
                output_model_version: 1,
                artifact_path: "file:///epc-1/worker-a/0",
                metrics: None,
                epoch_checkpoints: &epoch_rows,
            },
        )
        .await
        .unwrap();
    assert!(
        finalized,
        "finalize succeeds even though one checkpoint row's name is occupied — \
         checkpoints are supplementary, never a reason to fail the job"
    );

    // The pre-existing user row is UNTOUCHED — same version, same
    // artifact_path — and is still exactly what `get_model` (the mechanism
    // `describe_model` resolves through) resolves for that name.
    let resolved = catalog
        .get_model(occupied_name)
        .await
        .unwrap()
        .expect("the occupied name still resolves");
    assert_eq!(
        resolved.version, 7,
        "the pre-existing row's version must be untouched"
    );
    assert_eq!(
        resolved.artifact_path.as_deref(),
        Some("file:///pre-existing/user/artifact"),
        "the pre-existing row's artifact_path must be untouched — never clobbered by the \
         checkpoint, and no shadowing duplicate inserted alongside it"
    );

    // The OTHER retained epoch row (no name collision) registered normally —
    // one collision must not block the rest.
    let epoch1_row = catalog
        .get_model("jammi:fine-tuned:epc-1:epoch_1")
        .await
        .unwrap()
        .expect("the non-colliding epoch row registers normally");
    assert_eq!(epoch1_row.status, "checkpoint");
    assert_eq!(epoch1_row.artifact_path.as_deref(), Some(epoch1_prefix));

    // The warning names both the occupied catalog name and the skipped
    // artifact prefix — never a silent no-op.
    let logs = String::from_utf8(buffer.lock().unwrap().clone()).expect("utf-8 logs");
    assert!(
        logs.contains(occupied_name) && logs.contains(epoch0_prefix),
        "the warning must name the occupied catalog name and the skipped artifact prefix; \
         captured logs:\n{logs}"
    );
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
