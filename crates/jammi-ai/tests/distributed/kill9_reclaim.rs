//! Property 2 — kill-9 reclaim.
//!
//! A worker claims a job under a short (3 s) lease, then is SIGKILL'd mid-run —
//! the harshest failure: no graceful shutdown, no terminal write, the lease
//! simply stops being renewed. A surviving worker's `reclaim_expired_training_jobs`
//! observes the expired lease, re-queues the job (bumping `attempts`), re-claims
//! it, and runs it to `completed`. The job's model is recorded exactly once,
//! under the *reclaiming* worker — proving the fleet recovers a crashed worker's
//! in-flight job without a double-finalize.

use crate::harness::{self, Backends, Fleet, JobSize};

const TEST: &str = "kill9_reclaim";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn killed_worker_job_is_reclaimed_and_completed_once() {
    let Some(backends) = Backends::from_env_or_skip(TEST) else {
        return;
    };
    let result_root = backends.unique_result_root(TEST);
    let (session, _dir) = harness::harness_session(&backends, &result_root).await;

    harness::register_training_source(&session, "training").await;
    // Crashable: the run must reliably still be `running` when we detect the
    // claimer and SIGKILL it, so the crash lands mid-flight, not after finish.
    let (job_id, expected_model) =
        harness::submit_fine_tune(&session, "training", JobSize::Crashable).await;

    // Two workers: whichever claims first will be killed; the other reclaims.
    let mut fleet = Fleet::spawn(&backends, &result_root, 2);

    // Detect the claimer: poll until the job is `running` and stamped with a
    // `claimed_by`. The run is long enough (6 epochs over the tiny model) to
    // outlive this detection + the kill, so we crash a worker genuinely mid-job.
    let first_claimer =
        harness::poll_until(harness::TERMINAL_TIMEOUT, harness::POLL_INTERVAL, || {
            let session = &session;
            let job_id = &job_id;
            async move {
                let r = session.catalog().get_training_job(job_id).await.ok()?;
                (r.status == "running")
                    .then(|| r.claimed_by.clone())
                    .flatten()
            }
        })
        .await
        .expect("a worker claims the job and marks it running before the lease window closes");

    // SIGKILL the claimer mid-run — no terminal write, the lease just dies.
    assert!(
        fleet.kill9(&first_claimer),
        "the detected claimer {first_claimer:?} is one of the spawned workers"
    );

    // Poll until a DIFFERENT worker has reclaimed and completed the job. Reclaim
    // re-queues the expired lease (so a re-run bumps `attempts` to ≥ 2) and the
    // surviving worker finalizes — the committed `claimed_by` is the reclaimer,
    // not the corpse.
    let record = harness::poll_until(harness::TERMINAL_TIMEOUT, harness::POLL_INTERVAL, || {
        let session = &session;
        let job_id = &job_id;
        async move {
            let r = session.catalog().get_training_job(job_id).await.ok()?;
            (r.status == "completed").then_some(r)
        }
    })
    .await
    .expect("the killed worker's job is reclaimed by a survivor and completed");

    let final_claimer = record
        .claimed_by
        .as_deref()
        .expect("the completed job records its (reclaiming) claimer");
    assert_ne!(
        final_claimer, first_claimer,
        "the reclaiming worker must differ from the killed claimer"
    );
    assert!(
        record.attempts >= 2,
        "reclaim re-queues the crashed attempt, so the completed run is attempt ≥ 2, got {}",
        record.attempts
    );

    // The model is recorded exactly once, with the deterministic id, and exactly
    // one model row exists — the crash + reclaim never double-finalizes.
    assert_eq!(
        record.output_model_id.as_deref(),
        Some(expected_model.as_str()),
        "the reclaimed job names the same deterministic output model id"
    );
    let models = session.catalog().list_models().await.unwrap();
    let matching = models
        .iter()
        .filter(|m| m.model_id == expected_model)
        .count();
    assert_eq!(
        matching, 1,
        "exactly one model row after crash + reclaim, got {matching}"
    );

    drop(fleet);
}
