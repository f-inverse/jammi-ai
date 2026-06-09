//! Property 1 — exactly-one-claim.
//!
//! One queued job, N worker processes racing to claim it from the shared
//! Postgres catalog. The claim is a `SELECT … FOR UPDATE SKIP LOCKED` so exactly
//! one worker wins; the others skip the locked row and find nothing. The proof
//! is in the *terminal* state: the job completes exactly once, names a single
//! `output_model_id`, that model is registered exactly once, and the
//! `claimed_by` recorded is one of the spawned workers' ids — never a
//! double-finalize, never two model rows for the one job.

use std::time::Duration;

use crate::harness::{self, Backends, Fleet, JobSize};

const TEST: &str = "exactly_one_claim";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn one_job_n_workers_exactly_one_wins() {
    let Some(backends) = Backends::from_env_or_skip(TEST) else {
        return;
    };
    let result_root = backends.unique_result_root(TEST);
    let (session, _dir) = harness::harness_session(&backends, &result_root).await;

    // Register a per-test-unique source and submit ONE queued job before any
    // worker can claim. The unique name keeps this test's source registration
    // from colliding with a prior test's row on the shared persistent catalog.
    let source = harness::unique_source_name(TEST);
    harness::register_training_source(&session, &source).await;
    let (job_id, expected_model) =
        harness::submit_fine_tune(&session, &source, JobSize::Quick).await;

    // Spawn a fleet of 4 workers that all race to claim the single job.
    let mut fleet = Fleet::spawn(&backends, &result_root, 4);
    let worker_ids: Vec<String> = fleet.worker_ids().into_iter().map(str::to_string).collect();

    // Poll for the terminal `completed` state. Only the lease-guarded finalize
    // CAS by the sole claimer flips the job to `completed`; a worker that did not
    // win the claim never reaches finalize. `await_job` dumps every worker's
    // config + log (and the final job row) and panics loudly if the fleet dies or
    // stalls, so a CI-only failure is diagnosable rather than a bare timeout.
    let record = harness::await_job(
        &mut fleet,
        &session,
        &job_id,
        None,
        "the single job reaches `completed` under one of the racing workers",
        |r| r.status == "completed",
    )
    .await;

    // Exactly one claimer, and it is one of the spawned workers.
    let claimed_by = record
        .claimed_by
        .as_deref()
        .expect("a completed job records its claimer");
    assert!(
        worker_ids.iter().any(|w| w == claimed_by),
        "claimed_by {claimed_by:?} must be one of the spawned workers {worker_ids:?}"
    );

    // Exactly one output model id, matching the deterministic id the submit minted.
    assert_eq!(
        record.output_model_id.as_deref(),
        Some(expected_model.as_str()),
        "the completed job names exactly the deterministic output model id"
    );

    // The finalize ran once: a single attempt, no reclaim churn (no worker
    // crashed, so the winner's first attempt finalized). The CAS is the sole
    // finalizer, so a second worker reaching finalize would have matched zero
    // rows — there is no second `completed` write to observe.
    assert_eq!(
        record.attempts, 1,
        "an uncontested win finalizes on the first attempt (no reclaim)"
    );

    // Exactly one model row for that id across the whole catalog — no
    // double-registration from two workers both believing they won.
    let models = session.catalog().list_models().await.unwrap();
    let matching = models
        .iter()
        .filter(|m| m.model_id == expected_model)
        .count();
    assert_eq!(
        matching, 1,
        "exactly one model row is registered for the winning job, got {matching}"
    );

    // The committed served pointer is set (the finalize CAS wrote it) and roots
    // under this run's MinIO prefix — the winner's artifact, on the shared bucket.
    let model = models
        .iter()
        .find(|m| m.model_id == expected_model)
        .unwrap();
    let artifact = model
        .artifact_path
        .as_deref()
        .expect("the finalize CAS commits the served artifact_path");
    assert!(
        artifact.starts_with(&result_root),
        "the committed artifact {artifact:?} roots under this run's MinIO prefix {result_root:?}"
    );

    drop(fleet);
    // A brief grace so the SIGKILL'd children are reaped before the test exits;
    // not load-bearing for the assertions (all already made), just tidy teardown.
    tokio::time::sleep(Duration::from_millis(100)).await;
}
