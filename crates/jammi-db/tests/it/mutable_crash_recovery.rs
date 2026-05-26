//! Phase 2 SPEC-02 §11 exit criterion #2 — atomic multi-row write under
//! crash. Justifies the "either every step lands or nothing lands" contract
//! in SPEC-02 §3.5.
//!
//! Mechanics:
//! - Parent spawns the test binary recursively with `JAMMI_TEST_CRASH_CHILD=1`,
//!   pointing at a shared tempdir and the [`test_hook`]-driven ready file.
//! - Child opens a `JammiSession`, registers a mutable table, and calls
//!   `insert_batch` twice (50 rows each) inside one transaction. The hook
//!   fires once the per-call row counter crosses 50, writes the ready file,
//!   then parks forever on a notifier no one signals.
//! - Parent polls for the ready file, sends `SIGKILL` to the child's pid,
//!   awaits exit, then opens a fresh session against the same tempdir and
//!   asserts the table is empty. The first insert's rows were never
//!   committed (the transaction died with the child), so SQLite WAL
//!   recovery rolls back to the pre-transaction state.
//!
//! Gated behind `feature = "test-hooks"` because the hook is only compiled
//! under that feature. The non-feature build of the engine has no
//! checkpoint behaviour at all.
//!
//! Note: not parameterized over `BackendKind`. The atomic-rollback contract
//! is a property of the catalog backend (rusqlite + sqlx Postgres both
//! deliver RAII rollback on connection drop); covering it once on the
//! default deployment proves the engine's usage is correct. A future
//! follow-up can parameterize if a Postgres-only regression appears.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use jammi_db::catalog::backend::TxOptions;
use jammi_db::session::JammiSession;
use jammi_db::store::mutable::definition::{MutableTableDefinitionBuilder, MutableTableId};
use jammi_db::store::mutable::test_hook::{CHECKPOINT_AFTER_ENV, READY_FILE_ENV};

use crate::common;

const CHILD_MARKER_ENV: &str = "JAMMI_TEST_CRASH_CHILD";
const ARTIFACT_DIR_ENV: &str = "JAMMI_TEST_ARTIFACT_DIR";
const TABLE_NAME: &str = "crash_target";

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
    assert_eq!(rc, 0, "libc::kill returned errno {}", unsafe {
        *libc::__error()
    });

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
