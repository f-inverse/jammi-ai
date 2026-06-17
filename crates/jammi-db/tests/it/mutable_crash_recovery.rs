//! Crash-consistency of the mutable-table substrate under `SIGKILL`. Two
//! families of test, both driven by the same self-respawn + `SIGKILL` harness:
//!
//! 1. Atomic multi-row write (SPEC-02 §11 exit criterion #2): a partial
//!    `insert_batch` transaction rolls back wholesale.
//! 2. Lifecycle crash-consistency (H4 §4.3 T3): register / register_topic /
//!    drop_table / drop_topic each run as ONE backend transaction spanning both
//!    the catalog row and the storage `CREATE TABLE`/`DROP TABLE` (the mutable
//!    storage tables live in the catalog's own database). A crash leaves either
//!    NOTHING or EVERYTHING — never a torn half (INV-4: a `mutable_tables` row
//!    ⇔ its storage table; a `topics` row ⇔ its backing row + storage table).
//!
//! Mechanics (shared):
//! - Parent spawns the test binary recursively with `JAMMI_TEST_CRASH_CHILD=1`,
//!   pointing at a shared tempdir and a [`test_hook`]-driven ready file.
//! - Child opens a `JammiSession` and drives the op under test. A test-hook
//!   checkpoint fires mid-transaction — for the insert test once the per-call
//!   row counter crosses the threshold; for the lifecycle tests at the op's
//!   commit boundary (every statement issued, nothing durable). The hook writes
//!   the ready file and parks forever on a notifier no one signals.
//! - Parent polls for the ready file (proof the checkpoint fired), `SIGKILL`s
//!   the child's pid, awaits exit, opens a fresh session on the same tempdir,
//!   and asserts the recovered state. The in-flight transaction died with the
//!   child, so SQLite WAL recovery rolls it back.
//!
//! Gated behind `feature = "test-hooks"` because the hooks are only compiled
//! under that feature. The non-feature build of the engine has no checkpoint
//! behaviour at all.
//!
//! Note: the crash-injection harness runs on SQLite only. `SIGKILL` mid-write
//! exercises the catalog backend's recovery (SQLite WAL rollback), and the
//! single-transaction lifecycle ops rely on DDL-in-transaction, which both
//! SQLite (`BEGIN IMMEDIATE`) and Postgres (transactional DDL) provide. The
//! Postgres side of single-transaction register/drop is covered by the
//! Postgres-gated `mutable_tables` / `trigger` suites (CI's "Test (Postgres)"
//! lane via `live-postgres-tests` + `JAMMI_TEST_PG_URL`), which drive the same
//! `register` / `drop_table` / `register_topic` / `drop_topic` code paths; what
//! is SQLite-specific here is the `SIGKILL`-during-transaction recovery harness,
//! not the atomicity boundary under test.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use jammi_db::catalog::backend::{BackendError, TxOptions};
use jammi_db::session::JammiSession;
use jammi_db::store::mutable::definition::{MutableTableDefinitionBuilder, MutableTableId};
use jammi_db::store::mutable::test_hook::{
    CHECKPOINT_AFTER_ENV, LIFECYCLE_CHECKPOINT_ENV, READY_FILE_ENV,
};
use jammi_db::trigger::ids::TopicId;
use jammi_db::trigger::topic::TopicDefinition;

use crate::common;

const CHILD_MARKER_ENV: &str = "JAMMI_TEST_CRASH_CHILD";
const ARTIFACT_DIR_ENV: &str = "JAMMI_TEST_ARTIFACT_DIR";
const TABLE_NAME: &str = "crash_target";

/// The mutable table a lifecycle workload creates / drops.
const LIFECYCLE_TABLE: &str = "lifecycle_target";
/// The topic a register_topic / drop_topic workload creates / drops.
const LIFECYCLE_TOPIC: &str = "events.lifecycle";
/// Fixed topic id so the child and the parent's post-restart assertion derive
/// the SAME backing-table name (`__topic_<uuid>`). A fresh `TopicId::new()`
/// would differ between the two processes.
const LIFECYCLE_TOPIC_ID: &str = "11111111-1111-4111-8111-111111111111";

fn crash_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, false),
    ]))
}

fn build_batch(start: i64, len: i64) -> RecordBatch {
    let ids = Int64Array::from_iter_values(start..(start + len));
    let payloads: Vec<String> = (start..(start + len)).map(|i| format!("r{i}")).collect();
    let payloads = StringArray::from_iter_values(payloads);
    RecordBatch::try_new(crash_schema(), vec![Arc::new(ids), Arc::new(payloads)]).unwrap()
}

async fn child_workload() {
    let dir = std::env::var(ARTIFACT_DIR_ENV).expect("child needs artifact dir");
    let dir = PathBuf::from(dir);
    let config = common::test_config(&dir);
    let session = JammiSession::new(config).await.expect("child session");

    let id = MutableTableId::new(TABLE_NAME).unwrap();
    let def = MutableTableDefinitionBuilder::new(id.clone(), crash_schema())
        .primary_key(vec!["id".into()])
        .build()
        .unwrap();
    session.create_mutable_table(def).await.unwrap();

    // Two 50-row batches inside one transaction. The hook fires after the
    // first `insert_batch` returns (the per-call counter reaches 50). The
    // hook writes the ready file and parks; the second `insert_batch` is
    // never reached; the transaction never commits.
    let backend = session.catalog().backend_arc();
    let registry = session.mutable_tables_arc();
    let _ = backend
        .transaction(TxOptions::default(), move |tx| {
            let registry = Arc::clone(&registry);
            let id = id.clone();
            Box::pin(async move {
                let b1 = build_batch(0, 50);
                let b2 = build_batch(50, 50);
                registry
                    .insert_batch(tx, &id, &b1)
                    .await
                    .map_err(|e| jammi_db::BackendError::Execution(e.to_string()))?;
                registry
                    .insert_batch(tx, &id, &b2)
                    .await
                    .map_err(|e| jammi_db::BackendError::Execution(e.to_string()))?;
                Ok::<(), jammi_db::BackendError>(())
            })
        })
        .await;

    unreachable!("hook parks the child; SIGKILL is the only exit");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mutable_partial_insert_rolls_back_under_sigkill() {
    if std::env::var(CHILD_MARKER_ENV).is_ok() {
        child_workload().await;
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let ready_file = dir.path().join("ready");
    let exe = std::env::current_exe().expect("current_exe for child spawn");

    let mut child = tokio::process::Command::new(&exe)
        .args([
            "--exact",
            "--nocapture",
            "mutable_crash_recovery::mutable_partial_insert_rolls_back_under_sigkill",
        ])
        .env(CHILD_MARKER_ENV, "1")
        .env(ARTIFACT_DIR_ENV, dir.path())
        .env(READY_FILE_ENV, &ready_file)
        .env(CHECKPOINT_AFTER_ENV, "50")
        .spawn()
        .expect("spawn child test process");

    let pid = child.id().expect("child pid available") as i32;

    // Poll for ready-file appearance. 30 s upper bound covers `cargo test`'s
    // first-time compile/sccache warmup on cold runners; production CI is
    // sub-second.
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if tokio::fs::try_exists(&ready_file)
            .await
            .expect("try_exists ready file")
        {
            break;
        }
        if Instant::now() > deadline {
            let _ = child.kill().await;
            panic!("child never reached the checkpoint within 30s");
        }
        if let Ok(Some(status)) = child.try_wait() {
            panic!("child exited before checkpoint: {status:?}");
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // SAFETY: `pid` is the child we just spawned and stored above; `SIGKILL`
    // is unconditional and the call is synchronous. No allocator or thread
    // state is involved.
    let rc = unsafe { libc::kill(pid, libc::SIGKILL) };
    assert_eq!(
        rc,
        0,
        "libc::kill failed: {}",
        std::io::Error::last_os_error()
    );

    let _ = child.wait().await;

    // Fresh session on the same artifact dir. SQLite's WAL recovery rolls
    // back the in-flight transaction. The mutable-table storage table must
    // exist (the `CREATE TABLE` committed before the INSERT transaction
    // began) but contain zero rows.
    let restart = JammiSession::new(common::test_config(dir.path()))
        .await
        .expect("restart session");
    let batches = restart
        .sql(&format!(
            "SELECT COUNT(*) AS n FROM mutable.public.{TABLE_NAME}"
        ))
        .await
        .expect("count after crash");
    let total = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("count returns Int64")
        .value(0);
    assert_eq!(
        total, 0,
        "post-SIGKILL restart must see zero rows — the INSERT transaction never committed",
    );
}

// ---------------------------------------------------------------------------
// H4 §4.3 T3 — crash-consistency of the mutable-table + topic LIFECYCLE.
//
// Each register/drop lifecycle op now runs as ONE backend transaction spanning
// both the catalog row and the storage `CREATE TABLE`/`DROP TABLE` (the mutable
// storage tables live in the catalog's own database). The engine fires
// `maybe_signal_lifecycle(op)` at that transaction's commit boundary — every
// statement issued, nothing durable — so a `SIGKILL` while the child parks
// there proves the op is all-or-nothing. After restart we assert INV-4:
//
//   * a `mutable_tables` row  ⇔  its storage table exists
//   * a `topics` row          ⇔  its backing `mutable_tables` row + storage table
//
// Because the op is one transaction, the crash leaves either NOTHING
// (register rolled back) or EVERYTHING intact (drop rolled back) — never a torn
// half. The kill firing at the right boundary is proved by the ready-file: the
// child only writes it from inside the matching op's commit-boundary hook, so
// the parent reaching the kill means the checkpoint was hit (the test panics if
// the child exits before writing it).
// ---------------------------------------------------------------------------

fn lifecycle_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("kind", DataType::Utf8, false),
    ]))
}

fn lifecycle_table_def() -> jammi_db::store::mutable::definition::MutableTableDefinition {
    MutableTableDefinitionBuilder::new(
        MutableTableId::new(LIFECYCLE_TABLE).unwrap(),
        lifecycle_schema(),
    )
    .primary_key(vec!["id".into()])
    .build()
    .unwrap()
}

fn lifecycle_topic_def() -> TopicDefinition {
    use std::str::FromStr;
    TopicDefinition {
        id: TopicId::from_str(LIFECYCLE_TOPIC_ID).unwrap(),
        name: LIFECYCLE_TOPIC.to_string(),
        schema: lifecycle_schema(),
        tenant: None,
        broker_metadata: BTreeMap::new(),
    }
}

/// Whether a storage table named `table` physically exists in the catalog
/// database. Backend-agnostic: a `SELECT … LIMIT 0` succeeds iff the table is
/// present, and surfaces a missing-table error otherwise. Used to assert INV-4
/// independently of the catalog row.
async fn storage_table_exists(session: &JammiSession, table: &str) -> bool {
    let sql = format!("SELECT 1 FROM \"{table}\" LIMIT 0");
    session
        .catalog()
        .backend_arc()
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            move |tx| {
                let sql = sql.clone();
                Box::pin(async move {
                    tx.query(&sql, &[], |_row| Ok::<(), BackendError>(()))
                        .await?;
                    Ok::<(), BackendError>(())
                })
            },
        )
        .await
        .is_ok()
}

/// Whether the `topics` catalog row named `name` exists (unscoped lookup).
async fn topic_row_exists(session: &JammiSession, name: &str) -> bool {
    session
        .topic_repo()
        .lookup_by_name(name, None)
        .await
        .expect("topic lookup")
        .is_some()
}

/// Child side of a lifecycle crash test. Opens a session, optionally
/// pre-creates committed state, then runs the op under test. The op's
/// commit-boundary hook (keyed by [`LIFECYCLE_CHECKPOINT_ENV`]) fires and parks
/// the child; `SIGKILL` is the only exit, so the op never commits.
async fn lifecycle_child_workload(op: &str) {
    let dir = std::env::var(ARTIFACT_DIR_ENV).expect("child needs artifact dir");
    let dir = PathBuf::from(dir);
    let session = JammiSession::new(common::test_config(&dir))
        .await
        .expect("child session");

    match op {
        "register" => {
            // Crash mid-register: nothing was committed before.
            session
                .mutable_tables()
                .register(lifecycle_table_def())
                .await
                .unwrap();
        }
        "register_topic" => {
            session
                .topic_repo()
                .register_topic(&lifecycle_topic_def())
                .await
                .unwrap();
        }
        "drop_table" => {
            // Commit the table first (no checkpoint matches "register"), then
            // crash during the drop.
            session
                .mutable_tables()
                .register(lifecycle_table_def())
                .await
                .unwrap();
            let id = MutableTableId::new(LIFECYCLE_TABLE).unwrap();
            session.mutable_tables().drop_table(&id).await.unwrap();
        }
        "drop_topic" => {
            let topic = lifecycle_topic_def();
            session.topic_repo().register_topic(&topic).await.unwrap();
            session
                .topic_repo()
                .drop_topic(topic.id, None)
                .await
                .unwrap();
        }
        other => panic!("unknown lifecycle op {other}"),
    }

    unreachable!("hook parks the child; SIGKILL is the only exit");
}

/// Parent side: spawn the named test recursively as a child that performs `op`,
/// wait for the commit-boundary checkpoint, SIGKILL, then restart and run
/// `assert_post` against the recovered state.
async fn run_lifecycle_crash<F, Fut>(test_name: &str, op: &str, assert_post: F)
where
    F: FnOnce(JammiSession) -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let dir = tempfile::tempdir().unwrap();
    let ready_file = dir.path().join("ready");
    let exe = std::env::current_exe().expect("current_exe for child spawn");

    let mut child = tokio::process::Command::new(&exe)
        .args(["--exact", "--nocapture", test_name])
        .env(CHILD_MARKER_ENV, "1")
        .env(ARTIFACT_DIR_ENV, dir.path())
        .env(READY_FILE_ENV, &ready_file)
        .env(LIFECYCLE_CHECKPOINT_ENV, op)
        .spawn()
        .expect("spawn child test process");

    let pid = child.id().expect("child pid available") as i32;

    // Poll for the ready file. It is written ONLY from the matching op's
    // commit-boundary hook, so its appearance is proof the checkpoint fired.
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if tokio::fs::try_exists(&ready_file)
            .await
            .expect("try_exists ready file")
        {
            break;
        }
        if Instant::now() > deadline {
            let _ = child.kill().await;
            panic!("child never reached the {op} checkpoint within 30s");
        }
        if let Ok(Some(status)) = child.try_wait() {
            panic!("child exited before the {op} checkpoint: {status:?}");
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // SAFETY: `pid` is the child we just spawned; `SIGKILL` is unconditional and
    // synchronous. No allocator or thread state is touched.
    let rc = unsafe { libc::kill(pid, libc::SIGKILL) };
    assert_eq!(
        rc,
        0,
        "libc::kill failed: {}",
        std::io::Error::last_os_error()
    );
    let _ = child.wait().await;

    // Restart on the same artifact dir; assert the recovered state honours
    // INV-4 with no torn half.
    let restart = JammiSession::new(common::test_config(dir.path()))
        .await
        .expect("restart session");
    assert_post(restart).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_table_crash_leaves_nothing() {
    const NAME: &str = "mutable_crash_recovery::register_table_crash_leaves_nothing";
    if std::env::var(CHILD_MARKER_ENV).is_ok() {
        lifecycle_child_workload("register").await;
        return;
    }
    run_lifecycle_crash(NAME, "register", |restart| async move {
        // INV-4: no catalog row, and (⇔) no storage table. The whole
        // single-transaction register rolled back.
        let id = MutableTableId::new(LIFECYCLE_TABLE).unwrap();
        let row = restart.mutable_tables().get(&id).await.expect("get");
        assert!(
            row.is_none(),
            "crash mid-register must leave NO catalog row (rolled back)"
        );
        assert!(
            !storage_table_exists(&restart, LIFECYCLE_TABLE).await,
            "crash mid-register must leave NO storage table (rolled back)"
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_topic_crash_leaves_nothing() {
    const NAME: &str = "mutable_crash_recovery::register_topic_crash_leaves_nothing";
    if std::env::var(CHILD_MARKER_ENV).is_ok() {
        lifecycle_child_workload("register_topic").await;
        return;
    }
    run_lifecycle_crash(NAME, "register_topic", |restart| async move {
        // INV-4: no topic row, no backing catalog row, no backing storage
        // table. The single transaction (backing row + CREATE TABLE + topics
        // row) rolled back wholesale.
        assert!(
            !topic_row_exists(&restart, LIFECYCLE_TOPIC).await,
            "crash mid-register_topic must leave NO topics row"
        );
        let backing = lifecycle_topic_def().backing_table_name();
        let backing_id = MutableTableId::new(backing.clone()).unwrap();
        assert!(
            restart
                .mutable_tables()
                .get(&backing_id)
                .await
                .expect("get backing")
                .is_none(),
            "crash mid-register_topic must leave NO backing catalog row"
        );
        assert!(
            !storage_table_exists(&restart, &backing).await,
            "crash mid-register_topic must leave NO backing storage table"
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drop_table_crash_leaves_everything() {
    const NAME: &str = "mutable_crash_recovery::drop_table_crash_leaves_everything";
    if std::env::var(CHILD_MARKER_ENV).is_ok() {
        lifecycle_child_workload("drop_table").await;
        return;
    }
    run_lifecycle_crash(NAME, "drop_table", |restart| async move {
        // INV-4: the (committed) table survives intact — both catalog row and
        // storage table — because the single-transaction drop rolled back.
        let id = MutableTableId::new(LIFECYCLE_TABLE).unwrap();
        assert!(
            restart
                .mutable_tables()
                .get(&id)
                .await
                .expect("get")
                .is_some(),
            "crash mid-drop must leave the catalog row intact (rolled back)"
        );
        assert!(
            storage_table_exists(&restart, LIFECYCLE_TABLE).await,
            "crash mid-drop must leave the storage table intact (rolled back)"
        );
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn drop_topic_crash_leaves_everything() {
    const NAME: &str = "mutable_crash_recovery::drop_topic_crash_leaves_everything";
    if std::env::var(CHILD_MARKER_ENV).is_ok() {
        lifecycle_child_workload("drop_topic").await;
        return;
    }
    run_lifecycle_crash(NAME, "drop_topic", |restart| async move {
        // INV-4: topic row + backing catalog row + backing storage table all
        // survive — the single-transaction drop_topic rolled back wholesale.
        assert!(
            topic_row_exists(&restart, LIFECYCLE_TOPIC).await,
            "crash mid-drop_topic must leave the topics row intact (rolled back)"
        );
        let backing = lifecycle_topic_def().backing_table_name();
        let backing_id = MutableTableId::new(backing.clone()).unwrap();
        assert!(
            restart
                .mutable_tables()
                .get(&backing_id)
                .await
                .expect("get backing")
                .is_some(),
            "crash mid-drop_topic must leave the backing catalog row intact"
        );
        assert!(
            storage_table_exists(&restart, &backing).await,
            "crash mid-drop_topic must leave the backing storage table intact"
        );
    })
    .await;
}
