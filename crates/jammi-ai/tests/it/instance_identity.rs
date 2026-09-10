//! Process identity: `instances.instance_id` / `jobs.claimed_by` is a
//! per-process UUID minted at session construction; `JAMMI_WORKER_ID` is
//! only the row's `label`. The catalog consequence — a dead instance's
//! inline job is reclaimed even when a live peer carries the same label —
//! is pinned in `jammi-db`'s `jobs_queue` suite; this file pins the
//! session-level half and the `workers` row lifecycle around a claim loop.

use std::sync::Arc;
use std::time::Duration;

use jammi_ai::fine_tune::worker::EmbeddedWorker;
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::jobs_repo::WorkerRecord;

use crate::common;

const LABEL: &str = "shared-label-7";

async fn session() -> (Arc<InferenceSession>, tempfile::TempDir) {
    let dir = tempfile::TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    (session, dir)
}

/// Poll `list_workers` until `pred` holds (the `workers` upsert on spawn and
/// the delete on drop both ride detached tasks).
async fn await_workers(
    session: &InferenceSession,
    what: &str,
    pred: impl Fn(&[WorkerRecord]) -> bool,
) -> Vec<WorkerRecord> {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let rows = session.catalog().list_workers().await.unwrap();
        if pred(&rows) {
            return rows;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "timed out awaiting: {what}; workers = {rows:?}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Two sessions constructed under ONE `JAMMI_WORKER_ID` are two instances:
/// their ids differ, neither id IS the label, and each process's `workers`
/// row (once it runs a claim loop) shows the shared label beside its own
/// id. A stop (graceful) and a drop (abort) each remove the process's
/// `workers` row, so a stopped claimant is no longer listed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_sessions_given_one_worker_id_mint_distinct_ids_and_share_the_label() {
    std::env::set_var("JAMMI_WORKER_ID", LABEL);
    let (a, _dir_a) = session().await;
    let (b, _dir_b) = session().await;
    std::env::remove_var("JAMMI_WORKER_ID");

    assert_ne!(
        a.instance_id(),
        b.instance_id(),
        "two processes sharing a label must not share an identity"
    );
    assert_ne!(a.instance_id(), LABEL, "the label is never the identity");
    assert!(
        uuid::Uuid::parse_str(a.instance_id()).is_ok(),
        "the identity is a minted UUID, got {:?}",
        a.instance_id()
    );

    let worker_a = EmbeddedWorker::spawn(&a).unwrap();
    let worker_b = EmbeddedWorker::spawn(&b).unwrap();
    let rows_a = await_workers(&a, "session a's workers row", |rows| rows.len() == 1).await;
    assert_eq!(rows_a[0].instance_id, a.instance_id());
    assert_eq!(
        rows_a[0].label.as_deref(),
        Some(LABEL),
        "the label rides the instances row"
    );
    let rows_b = await_workers(&b, "session b's workers row", |rows| rows.len() == 1).await;
    assert_eq!(rows_b[0].instance_id, b.instance_id());
    assert_eq!(rows_b[0].label.as_deref(), Some(LABEL));

    worker_a.stop_and_join().await.unwrap();
    assert!(
        a.catalog().list_workers().await.unwrap().is_empty(),
        "a gracefully stopped claim loop deletes its workers row before returning"
    );
    drop(worker_b);
    await_workers(&b, "session b's workers row deleted on drop", |rows| {
        rows.is_empty()
    })
    .await;
}
