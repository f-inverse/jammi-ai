//! SQLite implementation of [`CatalogBackend`] backed by `sqlx::SqlitePool`.

use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use sqlx::sqlite::{
    SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions, SqliteSynchronous,
};
use sqlx::ConnectOptions;

use super::backend::{classify, BackendError, BackendKind, CatalogBackend, Transaction, TxOptions};

/// SQLite-backed catalog. Wraps a connection pool with WAL mode + 5 s busy
/// timeout, matching the original `r2d2_sqlite`-based behaviour.
pub struct SqliteBackend {
    pool: SqlitePool,
}

impl SqliteBackend {
    /// Open (or create) the catalog database at `path`.
    pub async fn open(path: &Path) -> Result<Arc<Self>, BackendError> {
        let opts = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .busy_timeout(Duration::from_secs(5))
            .synchronous(SqliteSynchronous::Normal)
            .foreign_keys(true)
            .log_statements(tracing::log::LevelFilter::Trace);

        let pool = SqlitePoolOptions::new()
            .max_connections(8)
            .connect_with(opts)
            .await
            .map_err(classify)?;

        Ok(Arc::new(Self { pool }))
    }
}

impl CatalogBackend for SqliteBackend {
    fn transaction<'a, F, R>(
        &'a self,
        opts: TxOptions,
        f: F,
    ) -> Pin<Box<dyn Future<Output = Result<R, BackendError>> + Send + 'a>>
    where
        F: for<'tx> FnOnce(
                &'tx mut Transaction<'tx>,
            )
                -> Pin<Box<dyn Future<Output = Result<R, BackendError>> + Send + 'tx>>
            + Send
            + 'a,
        R: Send + 'a,
    {
        Box::pin(async move {
            // SQLite has no SET TRANSACTION ISOLATION LEVEL; isolation is fixed
            // by the journal mode (WAL gives snapshot reads). The write/read
            // distinction is carried entirely by the BEGIN mode, which `sqlx`'s
            // default `Pool::begin` (always DEFERRED) cannot express — so we
            // open the transaction through `Pool::begin_with`, which runs our
            // custom BEGIN yet still yields a sqlx `Transaction` that rolls back
            // on drop/cancel.
            //
            // A write transaction MUST take the database write lock at BEGIN
            // time (`BEGIN IMMEDIATE`): under WAL, two DEFERRED transactions
            // that each read then upgrade to a write deadlock with
            // SQLITE_BUSY_SNAPSHOT, which `busy_timeout` cannot break (waiting
            // never resolves a snapshot-upgrade conflict). IMMEDIATE makes
            // concurrent writers serialise on `busy_timeout` instead. A
            // read-only transaction stays DEFERRED so reads take a snapshot
            // without serialising against each other or against writers.
            let begin = if opts.read_only {
                "BEGIN DEFERRED"
            } else {
                "BEGIN IMMEDIATE"
            };
            let _ = (opts.isolation, opts.read_only);

            // The BEGIN itself must be uncancellable. `Pool::begin_with` issues
            // the `BEGIN` statement and only then constructs the sqlx
            // `Transaction` whose drop guard rolls back. If the caller's future
            // is dropped *while that begin is in flight* — after the worker has
            // run `BEGIN IMMEDIATE` and bumped its per-connection transaction
            // depth, but before the `Transaction` exists — there is no guard to
            // roll it back: the pooled connection returns to the pool still
            // inside a transaction, holding the WAL write lock. Its next checkout
            // then fails (`InvalidSavePointStatement`, because a custom `BEGIN`
            // is illegal at depth > 0), and every other writer starves on the
            // leaked write lock (`database is locked`). Running the begin on a
            // detached task closes the window: a cancelled caller drops only the
            // `JoinHandle`, the task still drives the begin to a fully-formed
            // `Transaction`, and that `Transaction` then drops through its own
            // guard — rolling back and returning the connection clean.
            let pool = self.pool.clone();
            let mut tx = tokio::spawn(async move { pool.begin_with(begin).await })
                .await
                .map_err(|join| {
                    BackendError::Unavailable(format!("transaction begin task failed: {join}"))
                })?
                .map_err(classify)?;

            // Scope wrapper so its borrow of `tx` ends before we move `tx`
            // into commit/rollback. The HRTB on `f` borrows wrapper for its
            // entire lifetime, so wrapper must drop before tx moves.
            let outcome = {
                let mut wrapper = Transaction::new_sqlite(&mut tx);
                f(&mut wrapper).await
            };

            match outcome {
                Ok(value) => {
                    tx.commit().await.map_err(classify)?;
                    Ok(value)
                }
                Err(err) => {
                    let _ = tx.rollback().await;
                    Err(err)
                }
            }
        })
    }

    fn migrate(&self) -> Pin<Box<dyn Future<Output = Result<(), BackendError>> + Send + '_>> {
        Box::pin(async move { super::migrations::run(self).await })
    }

    fn ping(&self) -> Pin<Box<dyn Future<Output = Result<(), BackendError>> + Send + '_>> {
        Box::pin(async move {
            sqlx::query("SELECT 1")
                .execute(&self.pool)
                .await
                .map_err(classify)?;
            Ok(())
        })
    }

    fn backend_kind(&self) -> BackendKind {
        BackendKind::Sqlite
    }
}
