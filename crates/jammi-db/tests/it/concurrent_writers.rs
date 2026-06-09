//! Regression coverage for the SQLite write-transaction deadlock.
//!
//! Two write transactions that each *read then write* the catalog must
//! serialise on SQLite's write lock, not deadlock. Under WAL, opening every
//! transaction `BEGIN DEFERRED` lets two such transactions both take a read
//! snapshot and then race to upgrade to a writer — a `SQLITE_BUSY_SNAPSHOT`
//! conflict that `busy_timeout` provably cannot break (waiting never resolves a
//! snapshot-upgrade race), surfacing as `database is locked`. A write
//! transaction must therefore take the write lock at BEGIN time
//! (`BEGIN IMMEDIATE`), which is what [`TxOptions`] with `read_only = false`
//! now guarantees on the SQLite backend.
//!
//! This is the catalog mechanics behind the embedded training worker's polling
//! loop (`claim`/`heartbeat`/`reclaim`, all write transactions) racing a
//! foreground catalog write such as `add_channel_columns`. The test drives the
//! same read-then-write transaction shape from multiple tasks against one
//! SQLite catalog and asserts every transaction commits without a lock error.

use std::sync::Arc;

use jammi_db::catalog::backend::{BackendError, SqlValue, TxOptions};
use jammi_db::catalog::Catalog;
use tempfile::tempdir;

/// One read-then-write transaction against a single-row scratch table: read the
/// current counter, then write it back incremented. This is the minimal shape
/// that deadlocks two concurrent `BEGIN DEFERRED` transactions on WAL — both
/// snapshot-read the row, then both try to upgrade to a writer. With the
/// IMMEDIATE write-lock fix the second transaction blocks on `busy_timeout` and
/// proceeds once the first commits.
async fn read_then_write(catalog: &Catalog) -> Result<(), BackendError> {
    let backend = catalog.backend_arc();
    backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                let current = tx
                    .query_opt("SELECT n FROM deadlock_probe WHERE id = 1", &[], |row| {
                        row.get::<i64>("n")
                    })
                    .await?
                    .unwrap_or(0);
                tx.execute(
                    "UPDATE deadlock_probe SET n = $1 WHERE id = 1",
                    &[SqlValue::Int(current + 1)],
                )
                .await?;
                Ok(())
            })
        })
        .await
}

/// Concurrent read-then-write catalog transactions must serialise, not
/// deadlock. Many tasks each run a tight loop of [`read_then_write`] against one
/// SQLite catalog on a multi-thread runtime — the same contention the embedded
/// training worker's polling loop puts on the catalog when a foreground write
/// lands at the same moment.
///
/// Without the write-lock-at-BEGIN fix this reliably fails with `(code: 5)
/// database is locked` (`SQLITE_BUSY_SNAPSHOT`), which `busy_timeout` cannot
/// resolve. With the fix every transaction commits.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_read_then_write_does_not_deadlock() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());

    let backend = catalog.backend_arc();
    backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "CREATE TABLE deadlock_probe (id INTEGER PRIMARY KEY, n INTEGER NOT NULL)",
                    &[],
                )
                .await?;
                tx.execute("INSERT INTO deadlock_probe (id, n) VALUES (1, 0)", &[])
                    .await?;
                Ok(())
            })
        })
        .await
        .unwrap();

    const WRITERS: usize = 8;
    const ITERS: usize = 60;

    let mut handles = Vec::with_capacity(WRITERS);
    for _ in 0..WRITERS {
        let catalog = Arc::clone(&catalog);
        handles.push(tokio::spawn(async move {
            for _ in 0..ITERS {
                read_then_write(&catalog).await?;
            }
            Ok::<(), BackendError>(())
        }));
    }

    for handle in handles {
        handle
            .await
            .expect("writer task panicked")
            .expect("concurrent read-then-write transaction failed (database is locked?)");
    }

    let backend = catalog.backend_arc();
    let total = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query_opt("SELECT n FROM deadlock_probe WHERE id = 1", &[], |row| {
                        row.get::<i64>("n")
                    })
                    .await
                })
            },
        )
        .await
        .unwrap()
        .unwrap();

    // Every increment committed: writers serialised on the write lock rather
    // than losing updates or deadlocking.
    assert_eq!(total, (WRITERS * ITERS) as i64);
}
