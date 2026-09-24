//! Crash-consistency of the mutable-table substrate under `SIGKILL`:
//!
//! 1. Atomic multi-row write: a partial `insert_batch` transaction rolls back
//!    wholesale.
//! 2. Lifecycle: register / register_topic / drop_table / drop_topic each run as
//!    ONE backend transaction spanning the catalog row and the storage
//!    `CREATE TABLE`/`DROP TABLE` (the mutable storage tables live in the
//!    catalog's own database). A crash leaves either nothing or everything —
//!    never a torn half: a `mutable_tables` row exists iff its storage table
//!    does, and a `topics` row iff its backing row and storage table do.
//!
//! Each parent test runs a child test in its own process
//! ([`common::kill_child_at_checkpoint`]). The child opens a session and drives
//! the operation until a test-hook checkpoint parks it mid-transaction — for the
//! insert once the per-call row counter crosses a threshold, for a lifecycle op
//! at its commit boundary (every statement issued, nothing durable). The parent
//! `SIGKILL`s it, opens a fresh session on the same directory, and asserts the
//! recovered state: the in-flight transaction died with the child, so SQLite's
//! WAL recovery rolls it back.
//!
//! The harness runs on SQLite only and is compiled under `test-hooks`. The
//! single-transaction boundary itself relies on transactional DDL, which
//! Postgres also provides; the `mutable_tables` and `trigger` suites drive the
//! same operations on Postgres under `live-postgres-tests`.

use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use jammi_db::catalog::backend::{BackendError, TxOptions};
use jammi_db::session::JammiSession;
use jammi_db::store::mutable::definition::{MutableTableDefinitionBuilder, MutableTableId};
use jammi_db::store::mutable::test_hook::{CHECKPOINT_AFTER_ENV, LIFECYCLE_CHECKPOINT_ENV};
use jammi_db::trigger::ids::TopicId;
use jammi_db::trigger::topic::TopicDefinition;

use crate::common;

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
    let dir = std::env::var(common::ARTIFACT_DIR_ENV).expect("child needs artifact dir");
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
    backend
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
        .await
        .expect("the transaction runs until the hook parks it");

    unreachable!("hook parks the child; SIGKILL is the only exit");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "child process of mutable_partial_insert_rolls_back_under_sigkill"]
async fn mutable_partial_insert_child() {
    child_workload().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mutable_partial_insert_rolls_back_under_sigkill() {
    let dir = tempfile::tempdir().unwrap();
    common::kill_child_at_checkpoint(
        "mutable_crash_recovery::mutable_partial_insert_child",
        dir.path(),
        (CHECKPOINT_AFTER_ENV, "50"),
    )
    .await;

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
    }
}

/// Whether a storage table named `table` physically exists in the catalog
/// database. Backend-agnostic: a `SELECT … LIMIT 0` succeeds iff the table is
/// present, and surfaces a missing-table error otherwise. Checks the storage side
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

/// Child side of every lifecycle crash test: the op under test is the one
/// [`LIFECYCLE_CHECKPOINT_ENV`] names. Opens a session, optionally pre-creates
/// committed state, then runs the op; its commit-boundary hook parks the child,
/// and `SIGKILL` is the only exit, so the op never commits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "child process of the lifecycle crash tests"]
async fn lifecycle_crash_child() {
    let op = std::env::var(LIFECYCLE_CHECKPOINT_ENV).expect("the parent names the op");
    let dir = std::env::var(common::ARTIFACT_DIR_ENV).expect("child needs artifact dir");
    let dir = PathBuf::from(dir);
    let session = JammiSession::new(common::test_config(&dir))
        .await
        .expect("child session");

    match op.as_str() {
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

/// Runs [`lifecycle_crash_child`] with `op`, kills it at the op's commit
/// boundary, then restarts and runs `assert_post` against the recovered state.
async fn run_lifecycle_crash<F, Fut>(op: &str, assert_post: F)
where
    F: FnOnce(JammiSession) -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let dir = tempfile::tempdir().unwrap();
    common::kill_child_at_checkpoint(
        "mutable_crash_recovery::lifecycle_crash_child",
        dir.path(),
        (LIFECYCLE_CHECKPOINT_ENV, op),
    )
    .await;
    let restart = JammiSession::new(common::test_config(dir.path()))
        .await
        .expect("restart session");
    assert_post(restart).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn register_table_crash_leaves_nothing() {
    run_lifecycle_crash("register", |restart| async move {
        // no catalog row, and (⇔) no storage table. The whole
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
    run_lifecycle_crash("register_topic", |restart| async move {
        // no topic row, no backing catalog row, no backing storage
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
    run_lifecycle_crash("drop_table", |restart| async move {
        // the (committed) table survives intact — both catalog row and
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
    run_lifecycle_crash("drop_topic", |restart| async move {
        // topic row + backing catalog row + backing storage table all
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
