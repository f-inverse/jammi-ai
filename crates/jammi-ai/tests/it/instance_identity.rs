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
/// `workers` row, so a stopped claimant is absent from the listing.
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

/// `InferenceSession::close()` releases every catalog connection the
/// session holds — including the lease keeper's (N3) OWN connection, opened
/// independently on its own dedicated thread at construction
/// ([`InferenceSession::new`]) and never touched by closing the session's
/// shared pool alone.
///
/// Before `close()` shut the keeper down and joined it, that connection
/// stayed open for as long as the keeper thread ran (unbounded past a plain
/// `Drop`, since `LeaseKeeper::drop` is flag-only and never waits for the
/// thread to exit) — so for the SQLite backend it alone kept the
/// `unix-excl` VFS's process-scoped exclusive lock held, and a successor
/// process opening the SAME catalog directory was refused within the 5 s
/// busy timeout even though every other handle had let go.
///
/// Proven here by timing a fresh [`jammi_db::catalog::Catalog::open`] on the
/// SAME directory immediately after `close()` returns: in isolation this
/// lands in single-digit milliseconds (the connection is genuinely gone,
/// not merely flagged), and the bound below is a generous multiple of that
/// — chosen so it is comfortably inside the 5 s busy timeout a still-open
/// keeper connection would force the reopen to wait out, without being a
/// false negative under this binary's own accumulated load (hundreds of
/// `InferenceSession`s across this suite each leave an un-joined keeper OS
/// thread behind on `Drop`, exactly like `Drop`'s own doc describes).
/// Landing inside the bound is only possible once every connection in the
/// process this session opened — the shared pool AND the keeper's own —
/// has actually released the file.
#[tokio::test]
async fn close_releases_the_lease_keepers_own_connection_so_a_fresh_open_is_fast() {
    let (session, dir) = session().await;
    // The session is doing real work over its keeper-held instance lease —
    // not an idle keeper that happens to have a connection open.
    assert!(session.lease_keeper().is_alive());

    session.close().await;
    assert!(
        !session.lease_keeper().is_alive(),
        "close() must leave the lease keeper reporting dead"
    );

    let started = std::time::Instant::now();
    let reopened = jammi_db::catalog::Catalog::open(dir.path())
        .await
        .expect("a fresh open on the same directory must succeed once close() has returned");
    let elapsed = started.elapsed();
    reopened.close().await;
    assert!(
        elapsed < Duration::from_secs(2),
        "the reopen took {elapsed:?} — a connection this process still held (the lease \
         keeper's own, in particular) would force it to wait out the 5 s busy timeout \
         instead of landing almost immediately"
    );
}
