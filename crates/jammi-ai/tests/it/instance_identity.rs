//! Process identity: `instances.instance_id` / `jobs.claimed_by` is a
//! per-process UUID minted at session construction; `JAMMI_WORKER_ID` is
//! only the row's `label`. The catalog consequence — a dead instance's
//! inline job is reclaimed even when a live peer carries the same label —
//! is pinned in `jammi-db`'s `jobs_queue` suite; this file pins the
//! session-level half and the `workers` row lifecycle around a claim loop.

use std::sync::Arc;
use std::time::Duration;

use jammi_ai::fine_tune::worker::{EmbeddedWorker, COMPILED_KINDS};
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::backend::{SqlValue, TxOptions};
use jammi_db::catalog::instance::{CanonicalRoot, GangListing, InstanceRegistration};
use jammi_db::catalog::jobs_repo::WorkerRecord;
use jammi_db::catalog::lease::{instance_liveness_margin, instance_prune_window};
use jammi_db::catalog::Catalog;
use jammi_db::error::JammiError;

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

// ---------------------------------------------------------------------------
// U5b-1a c3 — `[server] peer_advertise` threaded through
// `InstanceRegistration::from_config`, and the `JobWorker`/`EmbeddedWorker`
// ownership of the registration's worker half (CONTRACT-U5b-1a §8).
// ---------------------------------------------------------------------------

/// `(peer_addr, result_root)` as the `instances` row actually carries them —
/// read by raw SQL rather than through `peer_addr_of`/`list_gang_members`
/// (both apply freshness/kind/root filters that would leave "is this NULL?"
/// ambiguous with "was there no row at all?"). `None` means no row.
async fn instance_columns(
    catalog: &Catalog,
    instance_id: &str,
) -> Option<(Option<String>, Option<String>)> {
    let instance_id = instance_id.to_string();
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.query_opt(
                    "SELECT peer_addr, result_root FROM instances WHERE instance_id = $1",
                    &[SqlValue::TextOwned(instance_id)],
                    |row| {
                        Ok((
                            row.try_get::<String>("peer_addr")?,
                            row.try_get::<String>("result_root")?,
                        ))
                    },
                )
                .await
            })
        })
        .await
        .unwrap()
}

/// The total row count of `instances` — used to assert a row was NEVER
/// written (a failed `from_config` check must fail session open before any
/// write, §8 B3).
async fn instances_row_count(catalog: &Catalog) -> i64 {
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.query_opt("SELECT COUNT(*) AS n FROM instances", &[], |row| {
                    row.get::<i64>("n")
                })
                .await
            })
        })
        .await
        .unwrap()
        .unwrap_or(0)
}

/// Force `instances.last_seen_at` into the past — mirrors
/// `jammi-db`'s `gang_instance_freshness.rs`/`gang_membership.rs` helper of
/// the same name.
async fn force_stale_instance(catalog: &Catalog, instance_id: &str, ago: Duration) {
    let cutoff = (chrono::Utc::now() - chrono::Duration::from_std(ago).unwrap())
        .format("%Y-%m-%dT%H:%M:%S%.9fZ")
        .to_string();
    let instance_id = instance_id.to_string();
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "UPDATE instances SET last_seen_at = $1 WHERE instance_id = $2",
                    &[
                        SqlValue::TextOwned(cutoff),
                        SqlValue::TextOwned(instance_id),
                    ],
                )
                .await
            })
        })
        .await
        .unwrap();
}

/// Force-delete an `instances` row (`workers` cascades with it) — mirrors
/// `jammi-db`'s own helper of the same name: the state a stale sweep, or a
/// process that lost its row during a transient outage, leaves behind.
async fn force_delete_instance(catalog: &Catalog, instance_id: &str) {
    let instance_id = instance_id.to_string();
    catalog
        .backend_arc()
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "DELETE FROM instances WHERE instance_id = $1",
                    &[SqlValue::TextOwned(instance_id)],
                )
                .await
            })
        })
        .await
        .unwrap();
}

/// A config with a fast lease (`duration_secs = 3`, `heartbeat_secs = 1` —
/// the minimum whole-second pair honouring `heartbeat * 2 < duration`) so a
/// keeper pass and the liveness margin/prune window are observable within a
/// test's own timeout, plus `[server] peer_bind`/`peer_advertise` set to a
/// distinct loopback port pair.
fn fast_peer_config(dir: &std::path::Path, port: u16) -> jammi_db::config::JammiConfig {
    let mut config = common::test_config(dir);
    config.lease.duration_secs = 3;
    config.lease.heartbeat_secs = 1;
    config.server.peer_bind = Some(format!("0.0.0.0:{port}"));
    config.server.peer_advertise = Some(format!("127.0.0.1:{port}"));
    config
}

/// Poll until `catalog.list_gang_members` returns `instance_id` as a member,
/// or panic past `deadline`.
async fn wait_until_gang_member(
    catalog: &Catalog,
    instance_id: &str,
    kind: &str,
    canonical_root: &CanonicalRoot,
    lease: Duration,
) {
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    loop {
        let members = catalog
            .list_gang_members(GangListing {
                kind,
                self_instance: "not-a-real-instance-id",
                canonical_root,
                lease,
            })
            .await
            .unwrap();
        if members.iter().any(|m| m.instance_id == instance_id) {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for '{instance_id}' to become a gang member; members = {members:?}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// A library config (no `[server] peer_advertise`) writes `NULL` for both
/// `peer_addr` and `result_root` — the non-member shape.
#[tokio::test]
async fn library_config_without_peer_advertise_writes_null_membership_columns() {
    let (session, _dir) = session().await;
    let row = instance_columns(session.catalog(), session.instance_id())
        .await
        .expect("a session's own row must exist");
    assert_eq!(row, (None, None), "a library config must write NULL/NULL");
}

/// `[server] peer_advertise` set, `[storage] result_root` UNSET: through the
/// REAL `InferenceSession::open` construction (never a direct db write) the
/// row carries a non-NULL `peer_addr`/`result_root`, and `result_root` is
/// exactly `canonical_result_root()`'s own value (`canon(artifact_dir)/jammi_db`).
#[tokio::test]
async fn peer_advertise_set_result_root_unset_produces_a_nonnull_row_via_open() {
    let dir = tempfile::TempDir::new().unwrap();
    let config = fast_peer_config(dir.path(), 19101);
    let expected_root = config
        .canonical_result_root()
        .unwrap()
        .expect("peer_advertise is set")
        .as_str()
        .to_string();

    let session = InferenceSession::open(config).await.unwrap();
    let row = instance_columns(session.catalog(), session.instance_id())
        .await
        .expect("the row must exist");
    assert_eq!(row.0.as_deref(), Some("127.0.0.1:19101"));
    assert_eq!(row.1.as_deref(), Some(expected_root.as_str()));
}

/// `[server] peer_advertise` set, `[storage] result_root` SET (to an
/// existing directory): through `InferenceSession::open_with_placement`
/// the row carries a non-NULL `peer_addr`/`result_root`, with `result_root`
/// exactly the canonicalized `result_root` (no `jammi_db` leaf appended —
/// the B2 erratum arm).
#[tokio::test]
async fn peer_advertise_set_result_root_set_produces_a_nonnull_row_via_open_with_placement() {
    let dir = tempfile::TempDir::new().unwrap();
    let root_dir = tempfile::TempDir::new().unwrap();
    let mut config = fast_peer_config(dir.path(), 19102);
    config.storage.result_root = Some(root_dir.path().to_string_lossy().into_owned());
    let expected_root = config
        .canonical_result_root()
        .unwrap()
        .expect("peer_advertise is set")
        .as_str()
        .to_string();

    let session =
        InferenceSession::open_with_placement(config, Arc::new(jammi_db::index::AllLocal))
            .await
            .unwrap();
    let row = instance_columns(session.catalog(), session.instance_id())
        .await
        .expect("the row must exist");
    assert_eq!(row.0.as_deref(), Some("127.0.0.1:19102"));
    assert_eq!(row.1.as_deref(), Some(expected_root.as_str()));
}

/// `peer_advertise` without `peer_bind` fails session open with a typed
/// error naming BOTH keys — before any row is written.
#[tokio::test]
async fn peer_advertise_without_peer_bind_fails_open_naming_both_keys() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut config = common::test_config(dir.path());
    config.server.peer_advertise = Some("127.0.0.1:19103".to_string());

    let err = match InferenceSession::new(config).await {
        Ok(_) => panic!("expected session open to fail"),
        Err(e) => e,
    };
    assert!(matches!(err, JammiError::Config(_)), "{err:?}");
    let msg = err.to_string();
    assert!(
        msg.contains("peer_advertise") && msg.contains("peer_bind"),
        "error must name both keys: {msg}"
    );
}

/// F1 fix (the two enforcement points reaching the SAME verdict on one
/// config): a MISSING (non-existent, but well-formed and absolute)
/// `result_root` anchor with `peer_advertise` set is CREATED, never
/// refused — `InstanceRegistration::from_config` MATERIALIZES it
/// (idempotent with `JammiSession`'s own `create_dir_all` of `artifact_dir`,
/// and with `ResultStore`'s later one of the SAME `result_root` path), the
/// same way `JammiConfig::load_from` now accepts a missing anchor at load
/// (`jammi-db`'s `config::tests::
/// load_from_accepts_a_fresh_missing_anchor_when_peer_advertise_is_set`).
/// Session open succeeds and writes a non-NULL `instances` row.
#[tokio::test]
async fn missing_result_root_anchor_is_created_and_session_open_succeeds() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut config = fast_peer_config(dir.path(), 19104);
    let missing = dir.path().join("does-not-exist");
    assert!(!missing.exists());
    config.storage.result_root = Some(missing.to_string_lossy().into_owned());

    let session = InferenceSession::new(config)
        .await
        .expect("a missing (creatable) result_root anchor must be accepted, not refused");
    assert!(
        missing.is_dir(),
        "from_config must have materialized the missing anchor"
    );
    let row = instance_columns(session.catalog(), session.instance_id())
        .await
        .expect("the row must exist");
    assert!(row.0.is_some(), "peer_addr must be non-NULL: {row:?}");
    assert!(row.1.is_some(), "result_root must be non-NULL: {row:?}");
}

/// The F1 oracle, stated directly: `JammiConfig::load_from` on a config
/// whose `artifact_dir` anchor does not exist yet (the fresh-host case)
/// SUCCEEDS (the pure `MembershipConfig::validate` never reads the
/// filesystem), AND `InferenceSession::open` on THAT SAME config writes the
/// non-NULL member row — the two enforcement points agree, closing the gap
/// where `load_from` used to be STRICTER (it called the impure, existence-
/// checking `from_config` directly) than the session backstop (which
/// silently accepted the same fresh anchor because `JammiSession::new`'s own
/// catalog open had already `create_dir_all`'d it first).
#[tokio::test]
async fn load_from_and_session_open_agree_on_a_fresh_artifact_dir() {
    let base = tempfile::TempDir::new().unwrap();
    let missing = base.path().join("does-not-exist-yet");
    assert!(!missing.exists());
    let port = 19108u16;
    let toml_path = base.path().join("jammi.toml");
    // The `[artifact_dir]`-relative TOML file lives OUTSIDE `missing` (in
    // `base`, which the tempdir already created), while `artifact_dir`
    // itself names `missing` — genuinely absent until materialized.
    std::fs::write(
        &toml_path,
        format!(
            "artifact_dir = {:?}\n[server]\npeer_bind = \"0.0.0.0:{port}\"\n\
             peer_advertise = \"127.0.0.1:{port}\"\n[lease]\nduration_secs = 3\n\
             heartbeat_secs = 1\n",
            missing.to_str().unwrap()
        ),
    )
    .unwrap();

    // The REAL public loader — `JammiConfig::load_from`, the exact sequence
    // a production process runs — succeeds on the fresh anchor and creates
    // nothing.
    let config = jammi_db::config::JammiConfig::load_from(Some(&toml_path), std::iter::empty())
        .expect("load_from must accept a fresh, absolute, well-formed anchor");
    assert!(
        !missing.exists(),
        "load_from is PURE: it must never create a directory as a side effect of loading"
    );

    // The SAME config, opened as a real session: the materializing
    // backstop creates the anchor and writes a non-NULL member row — the
    // two enforcement points agree.
    let session = InferenceSession::new(config).await.unwrap();
    assert!(missing.is_dir(), "session open must materialize the anchor");
    let row = instance_columns(session.catalog(), session.instance_id())
        .await
        .expect("the row must exist");
    assert!(row.0.is_some(), "peer_addr must be non-NULL: {row:?}");
    assert!(row.1.is_some(), "result_root must be non-NULL: {row:?}");
}

/// A `result_root` anchor that exists but is a FILE, not a directory, still
/// fails session open with a typed error naming the key, and the
/// `instances` row is NEVER written — the ONE case a missing/creatable
/// anchor can never be confused with.
#[tokio::test]
async fn file_result_root_anchor_fails_open_and_writes_no_row() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut config = fast_peer_config(dir.path(), 19109);
    let file_path = dir.path().join("not-a-dir");
    std::fs::write(&file_path, b"x").unwrap();
    config.storage.result_root = Some(file_path.to_string_lossy().into_owned());

    let err = match InferenceSession::new(config).await {
        Ok(_) => panic!("expected session open to fail"),
        Err(e) => e,
    };
    assert!(matches!(err, JammiError::Config(_)), "{err:?}");
    assert!(
        err.to_string().contains("result_root") && err.to_string().contains("directory"),
        "error must name the offending key: {err}"
    );

    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    let catalog = loop {
        match Catalog::open(dir.path()).await {
            Ok(c) => break c,
            Err(e) => {
                assert!(
                    std::time::Instant::now() < deadline,
                    "timed out reopening the catalog after the failed session: {e}"
                );
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    };
    assert_eq!(
        instances_row_count(&catalog).await,
        0,
        "the failed `from_config` check must run before any `instances` write"
    );
    catalog.close().await;
}

/// P-M4 (ai-level, §8 B1 restated): a session with `[worker] enabled` and
/// `peer_advertise` set is a `list_gang_members` member (`kinds` + `state ==
/// claiming`) once its claim loop warms up. Force-deleting its `instances`
/// row and waiting one real `LeaseKeeper` pass (this session's OWN keeper —
/// no hand-built one) reregisters the WHOLE tuple: the process is a member
/// again, with `kinds`/`state` byte-identical to before the delete.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_gang_member_survives_a_forced_instance_delete_after_one_keeper_pass() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut config = fast_peer_config(dir.path(), 19105);
    config.worker.enabled = true;
    let lease = config.lease.intervals().unwrap().lease();
    let canonical_root = config.canonical_result_root().unwrap().unwrap();
    let kind = COMPILED_KINDS[0];

    let session = InferenceSession::open(config).await.unwrap();
    let worker = EmbeddedWorker::spawn(&session).unwrap();

    wait_until_gang_member(
        session.catalog(),
        session.instance_id(),
        kind,
        &canonical_root,
        lease,
    )
    .await;
    let before = session
        .catalog()
        .list_workers()
        .await
        .unwrap()
        .into_iter()
        .find(|w| w.instance_id == session.instance_id())
        .expect("this process's own workers row");
    assert_eq!(before.state, "claiming");

    force_delete_instance(session.catalog(), session.instance_id()).await;
    // Wait comfortably past one heartbeat tick (1s) for the session's own
    // keeper to notice the missed touch and reregister the whole tuple.
    tokio::time::sleep(Duration::from_secs(3)).await;

    wait_until_gang_member(
        session.catalog(),
        session.instance_id(),
        kind,
        &canonical_root,
        lease,
    )
    .await;
    let after = session
        .catalog()
        .list_workers()
        .await
        .unwrap()
        .into_iter()
        .find(|w| w.instance_id == session.instance_id())
        .expect("the reregistered workers row");
    assert_eq!(after.kinds, before.kinds, "kinds must be byte-identical");
    assert_eq!(after.state, before.state, "state must be byte-identical");

    worker.stop_and_join().await.unwrap();
}

/// A drained worker (its claim loop stopped, the registration's worker cell
/// cleared) is NOT resurrected as a gang member after a forced delete: the
/// keeper reregisters the `instances` row (the process itself is alive) but
/// never re-inserts a `workers` row, so `list_gang_members`'s INNER JOIN
/// still excludes it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_drained_worker_is_not_resurrected_as_a_member_after_a_forced_delete() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut config = fast_peer_config(dir.path(), 19106);
    config.worker.enabled = true;
    let lease = config.lease.intervals().unwrap().lease();
    let canonical_root = config.canonical_result_root().unwrap().unwrap();
    let kind = COMPILED_KINDS[0];

    let session = InferenceSession::open(config).await.unwrap();
    let worker = EmbeddedWorker::spawn(&session).unwrap();
    wait_until_gang_member(
        session.catalog(),
        session.instance_id(),
        kind,
        &canonical_root,
        lease,
    )
    .await;

    // A graceful stop clears the registration's worker cell BEFORE deleting
    // the `workers` row (§8 B1) — the process itself (its `instances` row)
    // stays.
    worker.stop_and_join().await.unwrap();
    assert!(session.catalog().list_workers().await.unwrap().is_empty());

    force_delete_instance(session.catalog(), session.instance_id()).await;
    tokio::time::sleep(Duration::from_secs(3)).await;

    // The `instances` row itself is back (the process is still alive)...
    assert!(
        session
            .catalog()
            .fresh_instance(session.instance_id(), lease)
            .await
            .unwrap(),
        "the instances row must reregister — the process itself never stopped"
    );
    // ...but it is never a gang member again: no `workers` row rides along.
    let members = session
        .catalog()
        .list_gang_members(GangListing {
            kind,
            self_instance: "not-a-real-instance-id",
            canonical_root: &canonical_root,
            lease,
        })
        .await
        .unwrap();
    assert!(
        !members
            .iter()
            .any(|m| m.instance_id == session.instance_id()),
        "a drained worker must never resurface as a gang member: {members:?}"
    );
}

/// The prune-window property through the REAL construction sweep: a foreign
/// `instances` row stale in `(margin, window]` (`instance_liveness_margin`
/// `<` `instance_prune_window`, both over the SAME lease) survives a second
/// session's boot-time `prune_instances` call — only the RIGHT function
/// (`instance_prune_window`, 3·lease), never the old literal
/// `saturating_mul(2)` (which equals the margin itself), can leave such a
/// row standing.
#[tokio::test]
async fn a_row_stale_in_the_margin_to_window_gap_survives_a_boot_sweep() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut config = fast_peer_config(dir.path(), 19107);
    // A3: `fast_peer_config`'s default 3 s lease left only ~1.5 s of real
    // wall-clock slack between seeding the row and the boot sweep actually
    // running (`InstanceRegistration::from_config`, the lease keeper start,
    // the result store build/recover, the Hub source, …) — comfortably
    // exceeded under load, flaking this test RED with no defect present.
    // A longer lease widens the (margin, window] gap proportionally
    // (`window - margin == lease`), and biasing `ago` a QUARTER of the gap
    // past `margin` (rather than the midpoint) maximises the slack before
    // `window` while staying safely past `margin` itself.
    config.lease.duration_secs = 9;
    let lease = config.lease.intervals().unwrap().lease();
    let margin = instance_liveness_margin(lease);
    let window = instance_prune_window(lease);
    assert!(
        margin < window,
        "the window must be strictly beyond the margin"
    );

    // Seed a foreign row (no live process) directly, backdated to
    // `margin + (window - margin) / 4` — stale under the margin, but with
    // generous slack before `window` (with lease = 9 s: margin = 18 s,
    // window = 27 s, ago = 20.25 s, ~6.75 s of slack).
    let catalog = Catalog::open(dir.path()).await.unwrap();
    let foreign = InstanceRegistration::new("foreign-instance", None, None, None, None);
    catalog.upsert_instance(&foreign).await.unwrap();
    let ago = margin + (window - margin) / 4;
    force_stale_instance(&catalog, "foreign-instance", ago).await;
    catalog.close().await;

    // A second session's construction sweep prunes with
    // `instance_prune_window(lease)` (session.rs), not the liveness margin
    // alone.
    let session = InferenceSession::open(config).await.unwrap();
    let row = instance_columns(session.catalog(), "foreign-instance").await;
    assert!(
        row.is_some(),
        "a row stale in (margin, window] must survive the boot sweep"
    );
}
