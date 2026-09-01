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

/// `Backends::from_env_or_skip`, upgraded to a hard failure when
/// `JAMMI_REQUIRE_DISTRIBUTED` is set — the live-distributed/NATS-
/// availability lane's own require-gate: the pod session that is SUPPOSED
/// to have the shared Postgres + MinIO backends configured (CI's
/// distributed workflow) treats an unconfigured lane as a failure, never a
/// silent skip. Duplicated identically across this family's four live-
/// distributed test binaries (`artifact_crash_window.rs`,
/// `cross_tenant_isolation.rs`, `exactly_one_claim.rs`, `kill9_reclaim.rs`)
/// rather than shared through `harness.rs`: `Backends::from_env_or_skip` is
/// an ASSOCIATED fn always called qualified (`Backends::from_env_or_skip
/// (..)`), never as a BARE call `check_kernel_oracles.py`'s KO-7 dominance
/// check can credit, and gating is per-file by construction (never
/// cross-file by name alone) — the same small-duplication idiom
/// `cuda_device` carries across `crates/jammi-kernels/tests/{cuda_parity,
/// flash_smoke,flash_op_oracles,flash_torch_parity}.rs`, applied here to a
/// live-backend availability probe instead of a hardware one.
///
/// The nested (not `&&`-collapsed) `if`s below are deliberate: KO-7's
/// registry verifier requires the INNER `if`'s condition to be EXACTLY the
/// `JAMMI_REQUIRE_*` env-read call, with no leading/trailing condition.
#[allow(clippy::collapsible_if)]
fn required_backends(test: &str) -> Option<Backends> {
    let backends = Backends::from_env_or_skip(test);
    if backends.is_none() {
        if std::env::var_os("JAMMI_REQUIRE_DISTRIBUTED").is_some() {
            panic!(
                "{test}: JAMMI_REQUIRE_DISTRIBUTED is set but the distributed lane's shared \
                 backends are unconfigured — a silent skip is not acceptable here"
            );
        }
    }
    backends
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn killed_worker_job_is_reclaimed_and_completed_once() {
    let Some(backends) = required_backends(TEST) else {
        return;
    };
    let result_root = backends.unique_result_root(TEST);
    let (session, _dir) = harness::harness_session(&backends, &result_root).await;

    // Per-test-unique source so this test's registration never collides with a
    // prior test's row on the shared persistent catalog.
    let source = harness::unique_source_name(TEST);
    harness::register_training_source(&session, &source).await;
    // Crashable: the run must reliably still be `running` when we detect the
    // claimer and SIGKILL it, so the crash lands mid-flight, not after finish.
    let (job_id, expected_model) =
        harness::submit_fine_tune(&session, &source, JobSize::Crashable).await;

    // Two workers: whichever claims first will be killed; the other reclaims.
    let mut fleet = Fleet::spawn(&backends, &result_root, 2);

    // Detect the claimer: poll until the job is `running` and stamped with a
    // `claimed_by`. The run is long enough (6 epochs over the tiny model) to
    // outlive this detection + the kill, so we crash a worker genuinely mid-job.
    // `await_job` dumps the fleet's configs/logs + the final job row and panics
    // loudly if a worker dies on its own or the claim never lands.
    let first_claimer = harness::await_job(
        &mut fleet,
        &session,
        &job_id,
        None,
        "a worker claims the job and marks it running before the lease window closes",
        |r| r.status == "running" && r.claimed_by.is_some(),
    )
    .await
    .claimed_by
    .expect("a running job records its claimer");

    // SIGKILL the claimer mid-run — no terminal write, the lease just dies.
    assert!(
        fleet.kill9(&first_claimer),
        "the detected claimer {first_claimer:?} is one of the spawned workers"
    );

    // Poll until a DIFFERENT worker has reclaimed and completed the job. Reclaim
    // re-queues the expired lease (so a re-run bumps `attempts` to ≥ 2) and the
    // surviving worker finalizes — the committed `claimed_by` is the reclaimer,
    // not the corpse.
    let record = harness::await_job(
        &mut fleet,
        &session,
        &job_id,
        None,
        "the killed worker's job is reclaimed by a survivor and completed",
        |r| r.status == "completed",
    )
    .await;

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
