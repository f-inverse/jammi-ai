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
use std::time::Duration;

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

/// A write transaction whose future is cancelled between `BEGIN IMMEDIATE` and
/// `COMMIT` must release the write lock and return its pooled connection clean.
///
/// A write transaction holds the SQLite WAL write lock from `BEGIN IMMEDIATE`
/// until it commits or rolls back. If the transaction is driven by a raw BEGIN
/// on a bare pooled connection, a future cancelled mid-flight (the training
/// worker aborts its heartbeat/handle, dropping in-flight catalog writes)
/// returns the connection to the pool *still inside an open transaction*,
/// holding the write lock — poisoning it: the next checkout of that connection
/// fails `cannot start a transaction within a transaction`.
///
/// Opening through sqlx's `Transaction` (which rolls back on drop) makes
/// cancellation safe: the dropped future releases the lock and the connection
/// returns clean. This test cancels a write transaction enough times to cycle
/// through every connection in the 8-slot pool, then asserts a fresh write
/// transaction still succeeds and that no cancelled write leaked a commit.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelled_write_transaction_does_not_poison_pool() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());

    let backend = catalog.backend_arc();
    backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "CREATE TABLE cancel_probe (id INTEGER PRIMARY KEY, n INTEGER NOT NULL)",
                    &[],
                )
                .await?;
                tx.execute("INSERT INTO cancel_probe (id, n) VALUES (1, 0)", &[])
                    .await?;
                Ok(())
            })
        })
        .await
        .unwrap();

    // Cancel more transactions than the pool has connections (8) so every
    // pooled connection is checked out, BEGIN-IMMEDIATE'd, written to, then
    // dropped before COMMIT — and later reused for the fresh transaction below.
    //
    // On the poisoning path the cancellation manifests two ways across the
    // loop: a healthy connection times out mid-transaction (the future is
    // dropped past the deadline), while a reused-but-poisoned connection
    // short-circuits — its `BEGIN IMMEDIATE` fails `cannot start a transaction
    // within a transaction` before the parking sleep, so the future returns an
    // error fast instead of timing out. Either way the transaction did not
    // commit; the proof that the pool stays usable is the fresh transaction
    // below, not the per-iteration outcome.
    const CANCELLATIONS: usize = 16;
    for _ in 0..CANCELLATIONS {
        let backend = catalog.backend_arc();
        let outcome = tokio::time::timeout(
            Duration::from_millis(20),
            backend.transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    // Write inside the transaction (takes effect under the
                    // IMMEDIATE write lock)...
                    tx.execute("UPDATE cancel_probe SET n = n + 1 WHERE id = 1", &[])
                        .await?;
                    // ...then park past the outer deadline so the future is
                    // dropped after the write but before COMMIT.
                    tokio::time::sleep(Duration::from_secs(3600)).await;
                    Ok(())
                })
            }),
        )
        .await;
        // The transaction must never report success: it is always cut off
        // before COMMIT — either by the outer timeout (`Err(Elapsed)`) or, on a
        // reused poisoned connection, by its own BEGIN failing (`Ok(Err(_))`).
        assert!(
            !matches!(outcome, Ok(Ok(()))),
            "a cancelled write transaction should never commit"
        );
    }

    // The pool is not poisoned: a fresh write transaction reusing the cycled
    // connections opens and commits cleanly. Before the fix this fails with
    // `cannot start a transaction within a transaction`.
    let backend = catalog.backend_arc();
    backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute("UPDATE cancel_probe SET n = 100 WHERE id = 1", &[])
                    .await?;
                Ok(())
            })
        })
        .await
        .expect("fresh write transaction after cancellations (pool poisoned?)");

    // The cancelled writes rolled back; only the committed fresh write stuck.
    let backend = catalog.backend_arc();
    let n = backend
        .transaction(
            TxOptions {
                read_only: true,
                ..Default::default()
            },
            |tx| {
                Box::pin(async move {
                    tx.query_opt("SELECT n FROM cancel_probe WHERE id = 1", &[], |row| {
                        row.get::<i64>("n")
                    })
                    .await
                })
            },
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(n, 100, "a cancelled write transaction leaked a commit");
}

/// A write transaction whose future is cancelled *during `BEGIN IMMEDIATE`
/// itself* — after the connection's worker has issued the BEGIN and bumped its
/// transaction depth, but before the sqlx `Transaction` (whose drop guard rolls
/// back) is constructed — must not poison its pooled connection.
///
/// This is the narrow-window sibling of
/// [`cancelled_write_transaction_does_not_poison_pool`], which cancels *after*
/// `BEGIN` returns (where the drop guard already exists). Here the cancellation
/// lands inside the begin: `Pool::begin_with` runs the `BEGIN` statement and
/// only then wraps the connection in a `Transaction`. A future dropped in that
/// gap leaves the connection in the pool still inside a transaction, holding the
/// WAL write lock with no guard to release it — so its next checkout fails
/// `InvalidSavePointStatement` (a custom `BEGIN` is illegal at depth > 0) and
/// every other writer starves on the leaked lock (`database is locked`). The
/// backend closes the window by driving the begin on a detached task, so a
/// cancelled caller drops only the join handle while the begin still completes
/// into a fully-guarded `Transaction` that rolls back on drop.
///
/// Many short-deadline `timeout`s race steady writers against one catalog on a
/// multi-thread runtime — the deadlines are spread across microseconds so some
/// land squarely in the begin window. After each round a fresh write must still
/// commit; before the fix this panics within a few rounds.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn begin_window_cancellation_does_not_poison_pool() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());

    let backend = catalog.backend_arc();
    backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(
                    "CREATE TABLE begin_probe (id INTEGER PRIMARY KEY, n INTEGER NOT NULL)",
                    &[],
                )
                .await?;
                tx.execute("INSERT INTO begin_probe (id, n) VALUES (1, 0)", &[])
                    .await?;
                Ok(())
            })
        })
        .await
        .unwrap();

    const ROUNDS: usize = 200;
    for round in 0..ROUNDS {
        let mut handles = Vec::new();

        // Steady writers: contend for the IMMEDIATE write lock so a cancelled
        // begin lands while the worker is mid-BEGIN.
        for _ in 0..6 {
            let catalog = Arc::clone(&catalog);
            handles.push(tokio::spawn(async move {
                let backend = catalog.backend_arc();
                let _ = backend
                    .transaction(TxOptions::default(), |tx| {
                        Box::pin(async move {
                            tx.execute("UPDATE begin_probe SET n = n + 1 WHERE id = 1", &[])
                                .await?;
                            Ok(())
                        })
                    })
                    .await;
            }));
        }

        // Cancelled writers at micro-deadlines spread across the begin window.
        for k in 0..6 {
            let catalog = Arc::clone(&catalog);
            let micros = 1 + (k as u64) * 37 + (round as u64 % 13);
            handles.push(tokio::spawn(async move {
                let backend = catalog.backend_arc();
                let _ = tokio::time::timeout(
                    Duration::from_micros(micros),
                    backend.transaction(TxOptions::default(), |tx| {
                        Box::pin(async move {
                            tx.execute("UPDATE begin_probe SET n = n + 1 WHERE id = 1", &[])
                                .await?;
                            Ok(())
                        })
                    }),
                )
                .await;
            }));
        }

        for handle in handles {
            handle.await.expect("writer task panicked");
        }

        // The pool survived the round: a fresh write opens and commits. Before
        // the fix this fails with `InvalidSavePointStatement` or `database is
        // locked` once a poisoned connection is recycled.
        backend
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute("UPDATE begin_probe SET n = n WHERE id = 1", &[])
                        .await?;
                    Ok(())
                })
            })
            .await
            .unwrap_or_else(|e| {
                panic!("round {round}: pool poisoned by begin-window cancel: {e:?}")
            });
    }
}
