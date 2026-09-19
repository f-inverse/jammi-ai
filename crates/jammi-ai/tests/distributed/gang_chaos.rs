//! The process-level chaos rows: a two-rank `Peer` gang
//! across REAL `jammi-server` processes over the shared Postgres catalog and
//! MinIO root, one process SIGKILLed mid-run.
//!
//! Every spawned worker is gang-capable (`[server] peer_bind`/
//! `peer_advertise`, `[distributed] max_world_size = 2`, `harness::
//! worker_toml`), so a `world_size = 2` job is claimed by one process (the
//! coordinator, rank 0) which lists and dials one member (rank 1). The
//! fleet has three processes: whichever two form the first gang, a third is
//! left to join the successor gang.
//!
//! - **SIGKILL a peer** (the member, rank 1): the coordinator's round fails
//!   on the dropped stream, the attempt is retired with no terminal write
//!   and its lease left to expire (a rank failure spends the
//!   attempt), a survivor requeues and re-claims it, a NEW gang runs it
//!   from the last epoch checkpoint to `completed`: one model row, the
//!   resume checkpoint reaped by the finalize winner.
//! - **SIGKILL the coordinator**: the member's held session ends on the
//!   dropped stream (its slot frees), the lease expires, a survivor
//!   requeues and re-claims, a new gang completes it — the same via the
//!   lease.
//!
//! Advisory in the distributed lane (`distributed.yml`'s chaos leg), like
//! the other SIGKILL rows: timing-sensitive by nature, and dependent on the
//! server's rank body for a cross-process gang to complete at all.

use jammi_db::catalog::jobs_repo::WorkerState;

use jammi_test_utils::DistributedBackends;

use crate::harness::{self, Fleet, JobSize};

const TEST_PEER: &str = "gang_chaos_peer";
const TEST_COORDINATOR: &str = "gang_chaos_coordinator";

/// The member a coordinator assigns rank 1 to, computed exactly as the
/// coordinator body does (`assign_ranks` over the listing sorted by
/// `instance_id` bytes): the first `claiming` `fine_tune` worker that is
/// not the coordinator itself.
async fn rank_1_of(session: &jammi_ai::session::InferenceSession, coordinator: &str) -> String {
    let mut candidates: Vec<String> = session
        .catalog()
        .list_workers()
        .await
        .unwrap()
        .into_iter()
        .filter(|w| {
            w.instance_id != coordinator
                && w.state == WorkerState::Claiming.to_string()
                && w.kinds.split(',').any(|k| k.trim() == "fine_tune")
        })
        .map(|w| w.instance_id)
        .collect();
    candidates.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
    candidates
        .into_iter()
        .next()
        .expect("a second fleet member exists to be rank 1")
}

/// The job completed under a new gang from the crashed attempt: exactly one
/// model row with the deterministic id, at least one attempt spent (the
/// retired one — `attempts - releases >= 2`), never the reclaim cap, and
/// the job-level resume checkpoint reaped by the finalize winner.
async fn assert_completed_by_a_new_gang(
    session: &std::sync::Arc<jammi_ai::session::InferenceSession>,
    record: &jammi_db::catalog::jobs_repo::JobRecord,
    expected_model: &str,
) {
    assert!(
        record.attempts - record.releases >= 2,
        "the crashed attempt was spent (never released) and a successor's claim ran the job: \
         attempts={} releases={}",
        record.attempts,
        record.releases
    );
    assert_eq!(
        record.output_model_id.as_deref(),
        Some(expected_model),
        "the reclaimed job names the same deterministic output model id"
    );
    let models = session.catalog().list_models().await.unwrap();
    assert_eq!(
        models
            .iter()
            .filter(|m| m.model_id == expected_model)
            .count(),
        1,
        "exactly one model row after the crash and the new gang"
    );
    assert!(
        session
            .artifact_store()
            .fetch_resume_checkpoint(None, &record.job_id)
            .await
            .unwrap()
            .is_none(),
        "the finalize winner reaps the job-level resume checkpoint"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn killed_peer_job_is_reclaimed_and_completed_by_a_new_gang() {
    let backends = DistributedBackends::from_env();
    let result_root = backends.unique_result_root(TEST_PEER);
    let (session, _dir) = harness::harness_session(&backends, &result_root).await;
    let source = harness::unique_source_name(TEST_PEER);
    harness::register_training_source(&session, &source).await;
    let (job_id, expected_model) =
        harness::submit_gang_fine_tune(&session, &source, JobSize::Crashable).await;

    let mut fleet = Fleet::spawn(&backends, &result_root, 3);
    let coordinator = harness::await_job(
        &mut fleet,
        &session,
        &job_id,
        None,
        "a worker claims the two-rank job and marks it running",
        |r| {
            r.status == jammi_db::catalog::status::JobStatus::Running.to_string()
                && r.claimed_by.is_some()
        },
    )
    .await
    .claimed_by
    .expect("a running job records its claimer");

    // Rank 1 is the coordinator's first listed member; the crash lands once
    // the gang is observed mid-run.
    let member = rank_1_of(&session, &coordinator).await;
    harness::await_mid_run(&mut fleet, &session, &job_id).await;
    let member_label = harness::label_of(&session, &member).await;
    assert!(
        fleet.kill9(&member_label),
        "rank 1 {member:?} (label {member_label:?}) is one of the spawned workers"
    );

    let record = harness::await_job(
        &mut fleet,
        &session,
        &job_id,
        None,
        "the killed peer's job is retired, requeued after the lease and completed by a new gang",
        |r| r.status == jammi_db::catalog::status::JobStatus::Completed.to_string(),
    )
    .await;
    assert_completed_by_a_new_gang(&session, &record, &expected_model).await;
    drop(fleet);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn killed_coordinator_job_is_reclaimed_and_completed_by_a_new_gang() {
    let backends = DistributedBackends::from_env();
    let result_root = backends.unique_result_root(TEST_COORDINATOR);
    let (session, _dir) = harness::harness_session(&backends, &result_root).await;
    let source = harness::unique_source_name(TEST_COORDINATOR);
    harness::register_training_source(&session, &source).await;
    let (job_id, expected_model) =
        harness::submit_gang_fine_tune(&session, &source, JobSize::Crashable).await;

    let mut fleet = Fleet::spawn(&backends, &result_root, 3);
    let coordinator = harness::await_job(
        &mut fleet,
        &session,
        &job_id,
        None,
        "a worker claims the two-rank job and marks it running",
        |r| {
            r.status == jammi_db::catalog::status::JobStatus::Running.to_string()
                && r.claimed_by.is_some()
        },
    )
    .await
    .claimed_by
    .expect("a running job records its claimer");
    harness::await_mid_run(&mut fleet, &session, &job_id).await;
    let coordinator_label = harness::label_of(&session, &coordinator).await;
    assert!(
        fleet.kill9(&coordinator_label),
        "the coordinator {coordinator:?} (label {coordinator_label:?}) is one of the spawned \
         workers"
    );

    let record = harness::await_job(
        &mut fleet,
        &session,
        &job_id,
        None,
        "the killed coordinator's job expires its lease, is requeued and completed by a new gang",
        |r| r.status == jammi_db::catalog::status::JobStatus::Completed.to_string(),
    )
    .await;
    assert_ne!(
        record.claimed_by.as_deref(),
        Some(coordinator.as_str()),
        "a survivor, never the corpse, completed the job"
    );
    assert_completed_by_a_new_gang(&session, &record, &expected_model).await;
    drop(fleet);
}
