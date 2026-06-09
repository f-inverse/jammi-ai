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
            // `Pool::begin` (always DEFERRED) cannot express — so we acquire a
            // pooled connection and drive BEGIN/COMMIT/ROLLBACK ourselves.
            //
            // A write transaction MUST take the database write lock at BEGIN
            // time (`BEGIN IMMEDIATE`): under WAL, two DEFERRED transactions
            // that each read then upgrade to a write deadlock with
            // SQLITE_BUSY_SNAPSHOT, which `busy_timeout` cannot break (waiting
            // never resolves a snapshot-upgrade conflict). IMMEDIATE makes
            // concurrent writers serialise on `busy_timeout` instead. A
            // read-only transaction stays DEFERRED so reads take a snapshot
            // without serialising against each other or against writers.
            let _ = opts.isolation;
            let begin = if opts.read_only {
                "BEGIN DEFERRED"
            } else {
                "BEGIN IMMEDIATE"
            };

            let mut conn = self.pool.acquire().await.map_err(classify)?;
            sqlx::query(begin)
                .execute(&mut *conn)
                .await
                .map_err(classify)?;

            // Scope wrapper so its borrow of `conn` ends before we issue
            // COMMIT/ROLLBACK. The HRTB on `f` makes the future borrow wrapper
            // for `'tx = lifetime of wrapper`, so wrapper must drop first.
            let outcome = {
                let mut wrapper = Transaction::new_sqlite(&mut conn);
                f(&mut wrapper).await
            };

            match outcome {
                Ok(value) => {
                    sqlx::query("COMMIT")
                        .execute(&mut *conn)
                        .await
                        .map_err(classify)?;
                    Ok(value)
                }
                Err(err) => {
                    let _ = sqlx::query("ROLLBACK").execute(&mut *conn).await;
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
