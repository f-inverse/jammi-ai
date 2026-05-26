//! Postgres implementation of [`CatalogBackend`] backed by `sqlx::PgPool`.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use sqlx::postgres::{PgConnectOptions, PgPool, PgPoolOptions};

use super::backend::{
    classify, BackendError, BackendKind, CatalogBackend, IsolationLevel, Transaction, TxOptions,
};

/// Postgres-backed catalog. Wraps `sqlx::PgPool`.
pub struct PostgresBackend {
    pool: PgPool,
}

impl PostgresBackend {
    /// Open the catalog database described by `url` with explicit pool
    /// options.
    ///
    /// `pool_size` becomes the pool's `max_connections`; `max_lifetime_secs`
    /// — when `Some` — sets `max_lifetime` on connections so deployments
    /// behind a connection-recycling proxy (PgBouncer, RDS Proxy) avoid
    /// hot-spotting one long-lived connection.
    pub async fn open_with_options(
        url: &str,
        pool_size: u32,
        max_lifetime_secs: Option<u32>,
    ) -> Result<Arc<Self>, BackendError> {
        let opts: PgConnectOptions = url.parse().map_err(classify)?;
        let mut builder = PgPoolOptions::new().max_connections(pool_size);
        if let Some(secs) = max_lifetime_secs {
            builder = builder.max_lifetime(Duration::from_secs(secs as u64));
        }
        let pool = builder.connect_with(opts).await.map_err(classify)?;
        Ok(Arc::new(Self { pool }))
    }
}

impl CatalogBackend for PostgresBackend {
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

            // Postgres allows SET TRANSACTION as the first statement after BEGIN.
            let iso_sql = match opts.isolation {
                IsolationLevel::ReadCommitted => "SET TRANSACTION ISOLATION LEVEL READ COMMITTED",
                IsolationLevel::RepeatableRead => "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ",
                IsolationLevel::Serializable => "SET TRANSACTION ISOLATION LEVEL SERIALIZABLE",
            };
            sqlx::query(iso_sql)
                .execute(&mut *tx)
                .await
                .map_err(classify)?;
            if opts.read_only {
                sqlx::query("SET TRANSACTION READ ONLY")
                    .execute(&mut *tx)
                    .await
                    .map_err(classify)?;
            }

            // Scope wrapper so its borrow of `tx` ends before we move `tx`
            // into commit/rollback. The HRTB on `f` borrows wrapper for its
            // entire lifetime, so wrapper must drop before tx moves.
            let outcome = {
                let mut wrapper = Transaction::new_postgres(&mut tx);
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
        BackendKind::Postgres
    }
}
