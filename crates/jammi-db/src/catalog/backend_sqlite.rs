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
            let mut tx = self.pool.begin().await.map_err(classify)?;

            // SQLite has no SET TRANSACTION ISOLATION LEVEL. WAL mode gives
            // snapshot isolation for readers; stronger isolation requires
            // BEGIN IMMEDIATE which sqlx's Pool::begin doesn't expose. For
            // RepeatableRead/Serializable on SQLite we accept the default
            // BEGIN DEFERRED — sufficient for the catalog's workload. The
            // read_only flag is advisory on SQLite.
            let _ = (opts.isolation, opts.read_only);

            // Scope wrapper so its borrow of `tx` ends before we move `tx`
            // into commit/rollback. The HRTB on `f` makes the future borrow
            // wrapper for `'tx = lifetime of wrapper`, so wrapper must drop.
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
