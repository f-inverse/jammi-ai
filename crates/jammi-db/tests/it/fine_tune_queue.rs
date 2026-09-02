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

/// [`Catalog::create_training_job`] writes the explicit `{"state":"pending"}`
/// marker (never SQL NULL) at submission — the one payload shape the catalog
/// itself writes; the claiming worker's
/// [`Catalog::record_acceleration_report`] overwrites it under a valid lease
/// with its own producer-owned payload, pinning `attempts` to the exact
/// attempt the claim returned.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn record_acceleration_report_writes_under_a_valid_lease(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog
        .create_training_job(job_params("ar-1"))
        .await
        .unwrap();
    let submitted = catalog.get_training_job("ar-1").await.unwrap();
    assert_eq!(
        submitted.acceleration_report.as_deref(),
        Some(r#"{"state":"pending"}"#),
        "submission writes the explicit pending marker, never NULL"
    );

    let claimed = catalog
        .claim_next_training_job("worker-a", Duration::from_secs(3600))
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

    let after = catalog.get_training_job("ar-1").await.unwrap();
    assert_eq!(
        after.acceleration_report.as_deref(),
        Some(report),
        "the determined report round-trips through parse_row verbatim"
    );

    // A non-owner (right attempt, wrong worker) cannot write it.
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
    let unchanged = catalog.get_training_job("ar-1").await.unwrap();
    assert_eq!(unchanged.acceleration_report.as_deref(), Some(report));
}

/// The mandatory `attempts` guard closes the zombie gap: `JAMMI_WORKER_ID` is
/// stable across process restarts, so a reclaimed job re-claimed by a worker
/// carrying the SAME worker id (a restarted process, or — as modeled here —
/// the exact identity a real restart preserves) is distinguished from its own
/// zombie only by `attempts`. After a reclaim bumps `attempts`, a write
/// presenting the OLD attempt must NOT land (the report stays whatever it was
/// before the stale write), while a write presenting the CURRENT attempt does.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn record_acceleration_report_rejects_a_zombies_stale_attempt(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog
        .create_training_job(job_params("ar-zombie"))
        .await
        .unwrap();

    // worker-a's first attempt claims with a zero lease (immediately expired).
    let first = catalog
        .claim_next_training_job("worker-a", Duration::from_secs(0))
        .await
        .unwrap()
        .expect("worker-a claims attempt 1");
    assert_eq!(first.attempts, 1);

    // The lease expires and reclaim re-queues the job.
    let actioned = catalog.reclaim_expired_training_jobs(5).await.unwrap();
    assert_eq!(actioned, 1, "the expired lease is re-queued");

    // worker-a re-claims — SAME worker id (JAMMI_WORKER_ID is stable across
    // restarts), but a NEW attempt. This is the current, live claimant.
    let second = catalog
        .claim_next_training_job("worker-a", Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("worker-a re-claims as attempt 2");
    assert_eq!(second.attempts, 2);
    assert_eq!(second.claimed_by.as_deref(), Some("worker-a"));

    // The zombie: worker-a's FIRST process, still holding attempt=1 in memory,
    // computes and tries to write its (now-stale) report. `claimed_by` and
    // `status` alone would match (same worker id, still running) — only the
    // `attempts` guard distinguishes it from the live claimant.
    let zombie_report = r#"{"state":"determined","attempt":1,"stale":true}"#;
    let zombie_wrote = catalog
        .record_acceleration_report("ar-zombie", "worker-a", 1, zombie_report)
        .await
        .unwrap();
    assert!(
        !zombie_wrote,
        "a zombie presenting the OLD attempt must not write, even with the SAME \
         worker id and a currently-running status"
    );
    let after_zombie = catalog.get_training_job("ar-zombie").await.unwrap();
    assert_ne!(
        after_zombie.acceleration_report.as_deref(),
        Some(zombie_report),
        "the zombie's stale report must never land"
    );

    // The current claimant (attempt=2) writes successfully.
    let current_report = r#"{"state":"determined","attempt":2,"stale":false}"#;
    let current_wrote = catalog
        .record_acceleration_report("ar-zombie", "worker-a", 2, current_report)
        .await
        .unwrap();
    assert!(current_wrote, "the current claimant's write must land");
    let after_current = catalog.get_training_job("ar-zombie").await.unwrap();
    assert_eq!(
        after_current.acceleration_report.as_deref(),
        Some(current_report),
        "the current claimant's report is what the row carries"
    );

    // A late zombie write (still presenting attempt=1) after the current
    // claimant has already written must also be rejected, and must not
    // clobber the current claimant's now-recorded report.
    let late_zombie_wrote = catalog
        .record_acceleration_report("ar-zombie", "worker-a", 1, zombie_report)
        .await
        .unwrap();
    assert!(!late_zombie_wrote, "a late zombie write is still rejected");
    let after_late_zombie = catalog.get_training_job("ar-zombie").await.unwrap();
    assert_eq!(
        after_late_zombie.acceleration_report.as_deref(),
        Some(current_report),
        "the current claimant's report survives a late zombie write attempt"
    );
}

/// `acceleration_report` is not the `metrics` blob: the terminal `finalize`
/// transition overwrites `metrics`/`status`/`output_model_id` but leaves
/// `acceleration_report` untouched.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acceleration_report_survives_finalize(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog
        .create_training_job(job_params("ar-fin"))
        .await
        .unwrap();
    catalog
        .register_model(RegisterModelParams {
            model_id: "jammi:fine-tuned:ar-fin",
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

    let claimed = catalog
        .claim_next_training_job("worker-a", Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("job claimed");
    let report = r#"{"state":"determined","attempt":1,"device":"cuda:0"}"#;
    catalog
        .record_acceleration_report("ar-fin", "worker-a", claimed.attempts, report)
        .await
        .unwrap();

    let finalized = catalog
        .finalize_training_job(FinalizeTrainingJobParams {
            job_id: "ar-fin",
            worker_id: "worker-a",
            output_model_id: "jammi:fine-tuned:ar-fin",
            output_model_version: 1,
            artifact_path: "file:///artifacts/ar-fin/worker-a/0",
            metrics: Some(r#"{"completed_at":"2026-01-01T00:00:00Z"}"#),
            epoch_checkpoints: &[],
        })
        .await
        .unwrap();
    assert!(finalized, "the lease owner finalizes the job");

    let after = catalog.get_training_job("ar-fin").await.unwrap();
    assert_eq!(after.status, TrainingJobStatus::Completed.to_string());
    assert_eq!(
        after.acceleration_report.as_deref(),
        Some(report),
        "finalize must leave acceleration_report INTACT — it is not the metrics blob"
    );
}

/// The terminal `fail` transition leaves `acceleration_report` untouched, the
/// same as `finalize`.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acceleration_report_survives_fail(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog
        .create_training_job(job_params("ar-fail"))
        .await
        .unwrap();
    let claimed = catalog
        .claim_next_training_job("worker-a", Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("job claimed");
    let report = r#"{"state":"determined","attempt":1,"device":"cpu"}"#;
    catalog
        .record_acceleration_report("ar-fail", "worker-a", claimed.attempts, report)
        .await
        .unwrap();

    let failed = catalog
        .fail_training_job("ar-fail", "worker-a", Some(r#"{"error_message":"boom"}"#))
        .await
        .unwrap();
    assert!(failed, "the lease owner records the failure");

    let after = catalog.get_training_job("ar-fail").await.unwrap();
    assert_eq!(after.status, TrainingJobStatus::Failed.to_string());
    assert_eq!(
        after.acceleration_report.as_deref(),
        Some(report),
        "fail must leave acceleration_report INTACT — it is not the metrics blob"
    );
}

/// The two reclaim arms compose across a full requeue → re-claim → exhaust
/// lifecycle, and they move `acceleration_report` in OPPOSITE directions
/// because they mean opposite things about the job's future (#446 finding 1):
/// the re-queue arm resets the dead attempt's report to the pending marker
/// (a new attempt will re-probe), while the terminal attempts-exhausted arm
/// preserves whatever determination the last attempt did record.
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

    catalog
        .create_training_job(job_params("ar-rc"))
        .await
        .unwrap();
    let claimed = catalog
        .claim_next_training_job("worker-a", Duration::from_secs(0))
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

    // Requeue arm: the lease is already expired, so reclaim re-queues the job.
    let actioned = catalog.reclaim_expired_training_jobs(5).await.unwrap();
    assert_eq!(actioned, 1, "the expired lease is re-queued");
    let requeued = catalog.get_training_job("ar-rc").await.unwrap();
    assert_eq!(requeued.status, TrainingJobStatus::Queued.to_string());
    assert_eq!(
        requeued.acceleration_report.as_deref(),
        Some(r#"{"state":"pending"}"#),
        "the requeue arm of reclaim resets the dead attempt's report to the pending \
         marker — the row is queued for a NEW attempt that will re-probe, so the stale \
         determination must not survive onto it"
    );

    // The new claimant re-claims with a zero lease too (immediately expired),
    // so the next reclaim sweep can act on it directly without a live lease
    // blocking the claim), and overwrites the stale report with its own.
    let reclaimed = catalog
        .claim_next_training_job("worker-b", Duration::from_secs(0))
        .await
        .unwrap()
        .expect("worker-b re-claims the requeued job");
    assert_eq!(reclaimed.attempts, 2);
    let fresh_report = r#"{"state":"determined","attempt":2,"stale":false}"#;
    catalog
        .record_acceleration_report("ar-rc", "worker-b", reclaimed.attempts, fresh_report)
        .await
        .unwrap();
    let after_overwrite = catalog.get_training_job("ar-rc").await.unwrap();
    assert_eq!(
        after_overwrite.acceleration_report.as_deref(),
        Some(fresh_report),
        "the new claimant's write replaces the stale report"
    );

    // Exhausted-attempts arm: worker-b's lease (already expired) is reclaimed
    // with a max that is already met by its `attempts = 2`, driving the job
    // straight to `failed` (the exhausted branch, not requeue) — the report
    // is left exactly as the current claimant wrote it.
    let actioned = catalog.reclaim_expired_training_jobs(2).await.unwrap();
    assert_eq!(actioned, 1, "attempts (2) >= max (2): the job fails");
    let failed = catalog.get_training_job("ar-rc").await.unwrap();
    assert_eq!(failed.status, TrainingJobStatus::Failed.to_string());
    assert_eq!(
        failed.acceleration_report.as_deref(),
        Some(fresh_report),
        "the exhausted-attempts reclaim arm leaves acceleration_report untouched too"
    );
}

// ---------------------------------------------------------------------------
// esc-075 / #446 finding 1 — the tri-state `acceleration_report` lifecycle is
// closed AT THE CATALOG EDGE.
//
// `create_training_job` stamps `{"state":"pending"}` = "submitted, no claimant
// has computed a determination YET". That sentence stops being true the moment
// the row reaches a TERMINAL status: a job that failed between claim and the
// acceleration probe would otherwise carry `pending` forever, a state the
// contract has no reading for. The three transitions below make that
// unrepresentable by construction — inside the same lease-guarded / reclaim
// UPDATE, not at N worker call sites (any of which can be skipped, or added
// later without the marker).
//
// Every test asserts the marker BYTES (the vocabulary embedded and remote
// surfaces read) and the degenerate arms: an already-terminal report is
// PRESERVED (never re-stamped), and a legacy pre-migration-026 `NULL`
// ("unknown") is never fabricated into a state it does not have.
// ---------------------------------------------------------------------------

/// The exact marker bytes the catalog substitutes for a still-`pending` report
/// on the lease-guarded terminal-failure write.
const FAILED_BEFORE_PROBE: &str = r#"{"state":"undetermined","reason":"failed_before_probe"}"#;

/// The exact marker bytes the catalog substitutes for a still-`pending` report
/// on the reclaim attempts-exhausted arm.
const LEASE_EXPIRED_EXHAUSTED: &str =
    r#"{"state":"undetermined","reason":"lease_expired_attempts_exhausted"}"#;

/// The submission-time marker, repeated here as the literal bytes the wire
/// surfaces read (not re-exported from the crate — the test asserts the
/// vocabulary independently of the constant that produces it).
const PENDING_MARKER: &str = r#"{"state":"pending"}"#;

/// Overwrite `acceleration_report` for `job_id` directly, bypassing every
/// lease guard — used only to construct a state the public API cannot reach
/// (a pre-migration-026 legacy `NULL`).
async fn set_acceleration_report_raw(catalog: &Catalog, job_id: &str, report: Option<&str>) {
    let job_id = job_id.to_string();
    let report = report.map(str::to_string);
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE training_jobs SET acceleration_report = $1 WHERE job_id = $2",
                    &[SqlValue::from(report), SqlValue::TextOwned(job_id)],
                )
                .await
            })
        })
        .await
        .unwrap();
}

/// (i) A job that fails between claim and the acceleration probe must not keep
/// the submission-time `pending` marker past its terminal `failed` status:
/// [`Catalog::fail_training_job`] rewrites `pending → undetermined` with reason
/// `failed_before_probe` INSIDE the same lease-guarded UPDATE. No worker call
/// site participates — this is the property that makes the unmarked
/// `record_failed` sites in the worker unable to leak a pending-forever row.
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
        .create_training_job(job_params("ar-f-pending"))
        .await
        .unwrap();
    let claimed = catalog
        .claim_next_training_job("worker-a", Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("job claimed");
    assert_eq!(claimed.attempts, 1);
    // The probe never runs: no `record_acceleration_report` call at all.
    let before = catalog.get_training_job("ar-f-pending").await.unwrap();
    assert_eq!(
        before.acceleration_report.as_deref(),
        Some(PENDING_MARKER),
        "precondition: the row still carries the submission-time pending marker"
    );

    let failed = catalog
        .fail_training_job(
            "ar-f-pending",
            "worker-a",
            Some(r#"{"error_message":"died before the probe"}"#),
        )
        .await
        .unwrap();
    assert!(failed, "the lease owner records the failure");

    let after = catalog.get_training_job("ar-f-pending").await.unwrap();
    assert_eq!(after.status, TrainingJobStatus::Failed.to_string());
    assert_eq!(
        after.acceleration_report.as_deref(),
        Some(FAILED_BEFORE_PROBE),
        "a terminal `failed` row must never carry `pending` — the catalog edge rewrites it \
         to the self-describing undetermined marker in the SAME UPDATE"
    );
}

/// (ii) The rewrite is strictly `pending → undetermined`: an already-terminal
/// report (`determined`, `not_applicable`, or an `undetermined` with the
/// worker's OWN more specific reason) is PRESERVED byte-for-byte. A blanket
/// "stamp undetermined on fail" would destroy the measured determination of
/// every job that failed AFTER a successful probe — the exact regression this
/// arm pins.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fail_preserves_an_already_terminal_report(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    // Every non-pending payload shape a producer may have already written,
    // including one that is ALREADY `undetermined` but with a more specific
    // reason the catalog must not coarsen to `failed_before_probe`.
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
        catalog
            .create_training_job(job_params(job_id))
            .await
            .unwrap();
        let claimed = catalog
            .claim_next_training_job("worker-a", Duration::from_secs(3600))
            .await
            .unwrap()
            .expect("job claimed");
        assert_eq!(
            claimed.job_id, job_id,
            "claims run oldest-first, one job at a time"
        );
        let wrote = catalog
            .record_acceleration_report(job_id, "worker-a", claimed.attempts, report)
            .await
            .unwrap();
        assert!(wrote, "the lease owner records its determination");

        let failed = catalog
            .fail_training_job(job_id, "worker-a", Some(r#"{"error_message":"boom"}"#))
            .await
            .unwrap();
        assert!(failed, "the lease owner records the failure");

        let after = catalog.get_training_job(job_id).await.unwrap();
        assert_eq!(after.status, TrainingJobStatus::Failed.to_string());
        assert_eq!(
            after.acceleration_report.as_deref(),
            Some(report),
            "{job_id}: an already-terminal report survives `fail` byte-for-byte — the \
             rewrite fires ONLY on the pending marker"
        );
    }
}

/// (ii, degenerate) A legacy pre-migration-026 row reads back SQL `NULL` —
/// "unknown", an honest absence of information. `fail` must NOT fabricate a
/// state for it: `NULL` is not `pending`, and the three-valued `NULL = 'x'`
/// comparison correctly falls through to the ELSE arm on both backends.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fail_leaves_a_legacy_null_report_unknown(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog
        .create_training_job(job_params("ar-f-null"))
        .await
        .unwrap();
    set_acceleration_report_raw(&catalog, "ar-f-null", None).await;
    catalog
        .claim_next_training_job("worker-a", Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("job claimed");

    let failed = catalog
        .fail_training_job("ar-f-null", "worker-a", Some(r#"{"error_message":"x"}"#))
        .await
        .unwrap();
    assert!(failed);

    let after = catalog.get_training_job("ar-f-null").await.unwrap();
    assert_eq!(after.status, TrainingJobStatus::Failed.to_string());
    assert_eq!(
        after.acceleration_report, None,
        "a legacy NULL stays NULL ('unknown') — the pending rewrite must never fabricate \
         a state for a row that never had one"
    );
}

/// (iii) The reclaim attempts-exhausted arm is the OTHER path to a terminal
/// `failed` status — reached with no worker in the loop at all (the claimant
/// is dead; nothing on that side can ever compensate). Same rule, its own
/// reason: `pending → undetermined/lease_expired_attempts_exhausted`.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reclaim_exhaustion_rewrites_a_pending_report_to_undetermined(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog
        .create_training_job(job_params("ar-x-pending"))
        .await
        .unwrap();
    // Zero lease: already expired when reclaim runs. attempts -> 1.
    let claimed = catalog
        .claim_next_training_job("worker-a", Duration::from_secs(0))
        .await
        .unwrap()
        .expect("job claimed");
    assert_eq!(claimed.attempts, 1);

    // max_attempts = 1, so attempts (1) >= max (1): the exhausted arm, not
    // requeue.
    let actioned = catalog.reclaim_expired_training_jobs(1).await.unwrap();
    assert_eq!(actioned, 1, "attempts (1) >= max (1): the job fails");

    let after = catalog.get_training_job("ar-x-pending").await.unwrap();
    assert_eq!(after.status, TrainingJobStatus::Failed.to_string());
    assert_eq!(
        after.acceleration_report.as_deref(),
        Some(LEASE_EXPIRED_EXHAUSTED),
        "the exhausted-attempts reclaim arm drives the row terminal, so it must also \
         retire the pending marker — with the reason that names WHY no determination exists"
    );
}

/// (iv) The exhausted arm's rewrite is `pending`-only too: a determination the
/// dead attempt DID record before its lease expired is the last true thing
/// known about the job, and survives the terminal reclaim byte-for-byte.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reclaim_exhaustion_preserves_a_determined_report(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog
        .create_training_job(job_params("ar-x-det"))
        .await
        .unwrap();
    let claimed = catalog
        .claim_next_training_job("worker-a", Duration::from_secs(0))
        .await
        .unwrap()
        .expect("job claimed");
    let report = r#"{"state":"determined","attempt":1,"device":"cpu"}"#;
    let wrote = catalog
        .record_acceleration_report("ar-x-det", "worker-a", claimed.attempts, report)
        .await
        .unwrap();
    assert!(wrote, "the claimant probed before its lease expired");

    let actioned = catalog.reclaim_expired_training_jobs(1).await.unwrap();
    assert_eq!(actioned, 1, "attempts (1) >= max (1): the job fails");

    let after = catalog.get_training_job("ar-x-det").await.unwrap();
    assert_eq!(after.status, TrainingJobStatus::Failed.to_string());
    assert_eq!(
        after.acceleration_report.as_deref(),
        Some(report),
        "a determination recorded by the dead attempt survives the terminal reclaim"
    );
}

/// (v) The requeue arm is NOT terminal — it hands the job back to the queue for
/// a NEW attempt that will re-probe. A `determined` report from the dead
/// attempt describes the hardware/config THAT attempt saw, and must not survive
/// onto a `queued` row where every reader would attribute it to the job's
/// current (not-yet-started) attempt. Reclaim resets it to the submission-time
/// pending marker — the same state a freshly-created queued job carries.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reclaim_requeue_resets_the_report_to_pending(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog
        .create_training_job(job_params("ar-rq"))
        .await
        .unwrap();
    let claimed = catalog
        .claim_next_training_job("worker-a", Duration::from_secs(0))
        .await
        .unwrap()
        .expect("job claimed");
    let stale = r#"{"state":"determined","attempt":1,"device":"cuda:0"}"#;
    let wrote = catalog
        .record_acceleration_report("ar-rq", "worker-a", claimed.attempts, stale)
        .await
        .unwrap();
    assert!(wrote);

    // max_attempts = 5 > attempts (1): the requeue arm.
    let actioned = catalog.reclaim_expired_training_jobs(5).await.unwrap();
    assert_eq!(actioned, 1, "the expired lease is re-queued");

    let requeued = catalog.get_training_job("ar-rq").await.unwrap();
    assert_eq!(requeued.status, TrainingJobStatus::Queued.to_string());
    assert_eq!(
        requeued.acceleration_report.as_deref(),
        Some(PENDING_MARKER),
        "a re-queued job is awaiting a fresh determination — the dead attempt's report \
         must not survive onto a queued row"
    );

    // And the next claimant's own report lands over the reset marker as usual.
    let second = catalog
        .claim_next_training_job("worker-b", Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("worker-b re-claims");
    assert_eq!(second.attempts, 2);
    assert_eq!(
        second.acceleration_report.as_deref(),
        Some(PENDING_MARKER),
        "the claim itself does not resurrect the dead attempt's report"
    );
    let fresh = r#"{"state":"determined","attempt":2,"device":"cpu"}"#;
    catalog
        .record_acceleration_report("ar-rq", "worker-b", 2, fresh)
        .await
        .unwrap();
    assert_eq!(
        catalog
            .get_training_job("ar-rq")
            .await
            .unwrap()
            .acceleration_report
            .as_deref(),
        Some(fresh)
    );
}

/// (vi) Control — the rewrite fires ONLY on the three terminal writes (and the
/// requeue reset), and only against the `pending` marker. A job that probes and
/// then succeeds keeps its determination through every transition in its
/// lifecycle: submission, claim, `mark_training_running`, heartbeat, a live
/// (unexpired) lease surviving a reclaim sweep, and the terminal `finalize`. If
/// this control ever goes red, a CASE has leaked past the pending marker into a
/// payload it must not touch.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_job_that_never_fails_keeps_its_report_untouched(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog
        .create_training_job(job_params("ar-ok"))
        .await
        .unwrap();
    catalog
        .register_model(RegisterModelParams {
            model_id: "jammi:fine-tuned:ar-ok",
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

    // Submitted, unclaimed: pending, and a reclaim sweep does not see it.
    assert_eq!(
        catalog
            .get_training_job("ar-ok")
            .await
            .unwrap()
            .acceleration_report
            .as_deref(),
        Some(PENDING_MARKER)
    );
    assert_eq!(
        catalog.reclaim_expired_training_jobs(5).await.unwrap(),
        0,
        "a queued job has no lease to reclaim"
    );
    assert_eq!(
        catalog
            .get_training_job("ar-ok")
            .await
            .unwrap()
            .acceleration_report
            .as_deref(),
        Some(PENDING_MARKER),
        "a no-op reclaim sweep must not reset an unclaimed job's marker"
    );

    // Claimed under a LIVE lease, still pre-probe: pending survives the claim,
    // a run-start metrics stamp, a heartbeat, and a reclaim sweep that finds
    // the lease unexpired.
    let claimed = catalog
        .claim_next_training_job("worker-a", Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("job claimed");
    assert_eq!(claimed.acceleration_report.as_deref(), Some(PENDING_MARKER));
    assert!(catalog
        .mark_training_running(
            "ar-ok",
            "worker-a",
            Some(r#"{"started_at":"2026-01-01T00:00:00Z"}"#)
        )
        .await
        .unwrap());
    assert!(catalog
        .heartbeat_training_job("ar-ok", "worker-a", Duration::from_secs(3600))
        .await
        .unwrap());
    assert_eq!(
        catalog.reclaim_expired_training_jobs(5).await.unwrap(),
        0,
        "a live lease is not reclaimed"
    );
    assert_eq!(
        catalog
            .get_training_job("ar-ok")
            .await
            .unwrap()
            .acceleration_report
            .as_deref(),
        Some(PENDING_MARKER),
        "no non-terminal transition retires the pending marker"
    );

    // The probe lands, then the job completes: the determination is the row's
    // final, untouched state.
    let report = r#"{"state":"determined","attempt":1,"device":"cuda:0"}"#;
    assert!(catalog
        .record_acceleration_report("ar-ok", "worker-a", claimed.attempts, report)
        .await
        .unwrap());
    assert!(catalog
        .finalize_training_job(FinalizeTrainingJobParams {
            job_id: "ar-ok",
            worker_id: "worker-a",
            output_model_id: "jammi:fine-tuned:ar-ok",
            output_model_version: 1,
            artifact_path: "file:///artifacts/ar-ok/worker-a/0",
            metrics: Some(r#"{"completed_at":"2026-01-01T00:01:00Z"}"#),
            epoch_checkpoints: &[],
        })
        .await
        .unwrap());

    let after = catalog.get_training_job("ar-ok").await.unwrap();
    assert_eq!(after.status, TrainingJobStatus::Completed.to_string());
    assert_eq!(
        after.acceleration_report.as_deref(),
        Some(report),
        "the happy path's determination is untouched by every write in the lifecycle"
    );
}

/// The exact marker bytes the catalog substitutes for a still-`pending` report
/// on the lease-guarded terminal-success write.
const FINALIZED_WITHOUT_DETERMINATION: &str =
    r#"{"state":"undetermined","reason":"finalized_without_determination"}"#;

/// Register the output model `finalize_training_job` commits the served path
/// onto, for `job_id`'s conventional output id.
async fn register_output_model(catalog: &Catalog, job_id: &str) -> String {
    let model_id = format!("jammi:fine-tuned:{job_id}");
    catalog
        .register_model(RegisterModelParams {
            model_id: &model_id,
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
    model_id
}

/// (vii) `finalize` is the THIRD terminal writer, and success is not evidence
/// of a determination: the worker's acceleration-report persist step is
/// best-effort (a lost lease guard or a catalog error is swallowed, and the run
/// proceeds to publish-and-finalize), so a job can reach terminal `completed`
/// having never recorded one. `pending` on a `completed` row is the same
/// unreadable state the failure paths retire — so the same lease-guarded
/// UPDATE that stamps `completed` retires it, with the reason that names WHY:
/// the job finished, but nothing ever determined its acceleration.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finalize_rewrites_a_pending_report_to_undetermined(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog
        .create_training_job(job_params("ar-fin-pending"))
        .await
        .unwrap();
    let output_model_id = register_output_model(&catalog, "ar-fin-pending").await;
    let claimed = catalog
        .claim_next_training_job("worker-a", Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("job claimed");
    assert_eq!(claimed.attempts, 1);
    // The probe's persist step never landed: no `record_acceleration_report`
    // call at all, exactly as a swallowed lease-guard miss or catalog error
    // leaves the row.
    let before = catalog.get_training_job("ar-fin-pending").await.unwrap();
    assert_eq!(
        before.acceleration_report.as_deref(),
        Some(PENDING_MARKER),
        "precondition: the row still carries the submission-time pending marker"
    );

    let finalized = catalog
        .finalize_training_job(FinalizeTrainingJobParams {
            job_id: "ar-fin-pending",
            worker_id: "worker-a",
            output_model_id: &output_model_id,
            output_model_version: 1,
            artifact_path: "file:///artifacts/ar-fin-pending/worker-a/0",
            metrics: Some(r#"{"completed_at":"2026-01-01T00:00:00Z"}"#),
            epoch_checkpoints: &[],
        })
        .await
        .unwrap();
    assert!(finalized, "the lease owner finalizes the job");

    let after = catalog.get_training_job("ar-fin-pending").await.unwrap();
    assert_eq!(after.status, TrainingJobStatus::Completed.to_string());
    assert_eq!(
        after.acceleration_report.as_deref(),
        Some(FINALIZED_WITHOUT_DETERMINATION),
        "a terminal `completed` row must never carry `pending` — finalize is a TERMINAL \
         write, so it retires the marker in the SAME lease-guarded UPDATE"
    );
}

/// (viii) `finalize`'s rewrite is strictly `pending`-valued, exactly like the
/// failure paths': every already-terminal payload a producer may have written —
/// `determined`, `not_applicable`, and an `undetermined` carrying the worker's
/// OWN more specific reason — survives byte-for-byte. A blanket "stamp
/// undetermined on finalize" would destroy the measured determination of every
/// successful run, which is the regression this arm pins.
/// `acceleration_report_survives_finalize` is the single-payload control for
/// the same property.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finalize_preserves_an_already_terminal_report(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    let cases = [
        (
            "ar-fin-det",
            r#"{"state":"determined","attempt":1,"device":"cuda:0"}"#,
        ),
        (
            "ar-fin-na",
            r#"{"state":"not_applicable","reason":"context_predictor"}"#,
        ),
        (
            "ar-fin-und",
            r#"{"state":"undetermined","reason":"failed_before_device_resolution"}"#,
        ),
    ];

    for (job_id, report) in cases {
        catalog
            .create_training_job(job_params(job_id))
            .await
            .unwrap();
        let output_model_id = register_output_model(&catalog, job_id).await;
        let claimed = catalog
            .claim_next_training_job("worker-a", Duration::from_secs(3600))
            .await
            .unwrap()
            .expect("job claimed");
        assert_eq!(
            claimed.job_id, job_id,
            "claims run oldest-first, one job at a time"
        );
        let wrote = catalog
            .record_acceleration_report(job_id, "worker-a", claimed.attempts, report)
            .await
            .unwrap();
        assert!(wrote, "the lease owner records its determination");

        let finalized = catalog
            .finalize_training_job(FinalizeTrainingJobParams {
                job_id,
                worker_id: "worker-a",
                output_model_id: &output_model_id,
                output_model_version: 1,
                artifact_path: "file:///artifacts/x/worker-a/0",
                metrics: Some(r#"{"completed_at":"2026-01-01T00:00:00Z"}"#),
                epoch_checkpoints: &[],
            })
            .await
            .unwrap();
        assert!(finalized, "the lease owner finalizes the job");

        let after = catalog.get_training_job(job_id).await.unwrap();
        assert_eq!(after.status, TrainingJobStatus::Completed.to_string());
        assert_eq!(
            after.acceleration_report.as_deref(),
            Some(report),
            "{job_id}: an already-terminal report survives `finalize` byte-for-byte — the \
             rewrite fires ONLY on the pending marker"
        );
    }
}

/// (ix, degenerate) A legacy pre-migration-026 row reads back SQL `NULL` —
/// "unknown". `finalize` must not fabricate a state for it either: the
/// three-valued `NULL = 'x'` comparison falls through to the `ELSE` arm on both
/// backends, so `NULL` stays `NULL`.
#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(
    feature = "live-postgres-tests",
    test_case(BackendKind::Postgres ; "postgres")
)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finalize_leaves_a_legacy_null_report_unknown(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let (_session, catalog) = queue_catalog!(backend, dir.path());

    catalog
        .create_training_job(job_params("ar-fin-null"))
        .await
        .unwrap();
    let output_model_id = register_output_model(&catalog, "ar-fin-null").await;
    set_acceleration_report_raw(&catalog, "ar-fin-null", None).await;
    catalog
        .claim_next_training_job("worker-a", Duration::from_secs(3600))
        .await
        .unwrap()
        .expect("job claimed");

    let finalized = catalog
        .finalize_training_job(FinalizeTrainingJobParams {
            job_id: "ar-fin-null",
            worker_id: "worker-a",
            output_model_id: &output_model_id,
            output_model_version: 1,
            artifact_path: "file:///artifacts/ar-fin-null/worker-a/0",
            metrics: None,
            epoch_checkpoints: &[],
        })
        .await
        .unwrap();
    assert!(finalized);

    let after = catalog.get_training_job("ar-fin-null").await.unwrap();
    assert_eq!(after.status, TrainingJobStatus::Completed.to_string());
    assert_eq!(
        after.acceleration_report, None,
        "a legacy NULL stays NULL ('unknown') — finalize's pending rewrite must never \
         fabricate a state for a row that never had one"
    );
}
