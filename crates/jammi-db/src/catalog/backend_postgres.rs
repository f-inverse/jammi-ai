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
    /// The pool's configured `max_connections`, retained since `sqlx::Pool`
    /// exposes live connection counts but not the configured ceiling.
    pool_size: u32,
    /// The park on this pool's connection returns (see
    /// [`super::pool_test_hooks`]).
    #[cfg(feature = "test-hooks")]
    return_park: Arc<super::pool_test_hooks::ReturnPark>,
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
        let opts = pg_connect_options(url).map_err(classify)?;
        let mut builder = PgPoolOptions::new().max_connections(pool_size);
        if let Some(secs) = max_lifetime_secs {
            builder = builder.max_lifetime(Duration::from_secs(secs as u64));
        }
        #[cfg(feature = "test-hooks")]
        let return_park = Arc::new(super::pool_test_hooks::ReturnPark::default());
        #[cfg(feature = "test-hooks")]
        let builder = super::pool_test_hooks::install(builder, &return_park);
        let pool = builder.connect_with(opts).await.map_err(classify)?;
        Ok(Arc::new(Self {
            pool,
            pool_size,
            #[cfg(feature = "test-hooks")]
            return_park,
        }))
    }

    /// The park on this pool's connection returns (see
    /// [`super::pool_test_hooks`]).
    #[cfg(feature = "test-hooks")]
    pub fn return_park(&self) -> &Arc<super::pool_test_hooks::ReturnPark> {
        &self.return_park
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

    /// Close the pool and wait for every connection to shut down. Postgres
    /// holds no local file locks, so this is a courtesy to the server's
    /// connection budget rather than a correctness requirement — but the
    /// contract is the same one the SQLite backend needs, so the seam is
    /// uniform.
    fn close(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async move {
            super::backend::close_pool_and_drain(&self.pool, BackendKind::Postgres).await
        })
    }

    fn backend_kind(&self) -> BackendKind {
        BackendKind::Postgres
    }

    fn pool_size(&self) -> u32 {
        self.pool_size
    }
}

impl PostgresBackend {
    /// The raw connection pool, for [`super::backend::BackendImpl::
    /// query_untransacted`] — a read issued directly against the pool
    /// (sqlx picks an idle connection, runs the statement standalone, and
    /// returns it), never wrapped in an explicit `BEGIN`/`SET
    /// TRANSACTION ...`/`COMMIT` the way [`CatalogBackend::transaction`]
    /// always pays for (measured: 4 extra round trips for a single
    /// read-only `SELECT`). `pub(crate)`: only the backend-agnostic
    /// dispatcher in `backend.rs` calls this.
    pub(crate) fn pool(&self) -> &PgPool {
        &self.pool
    }
}

/// The connect options a Postgres URL names, read in libpq's connection-URI
/// form (<https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING-URIS>):
/// `postgresql://[userspec@][hostspec][/dbname][?paramspec]`. The one parser
/// for every Postgres URL the engine takes through sqlx (the catalog, the
/// trigger broker).
///
/// sqlx reads the URL with the WHATWG URL grammar, which refuses credentials
/// beside an empty host — the form libpq uses for a Unix-socket connection
/// (`postgresql://user:@/db?host=/run/postgresql`: the host is empty and the
/// `host` parameter names the socket directory). libpq defines the `user` and
/// `password` parameters as equivalent to the userinfo, so a URL whose host
/// is empty is read with its userinfo moved into them; every other URL is
/// read as written. sqlx's own parameter handling reads `host=/dir` as the
/// socket directory.
pub(crate) fn pg_connect_options(url: &str) -> Result<PgConnectOptions, sqlx::Error> {
    socket_form_with_parameters(url)
        .as_deref()
        .unwrap_or(url)
        .parse()
}

/// `url` with its userinfo moved into `user` / `password` parameters, when
/// its authority carries a userinfo and an empty host; `None` otherwise.
fn socket_form_with_parameters(url: &str) -> Option<String> {
    let (scheme, rest) = url.split_once("://")?;
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let (authority, tail) = rest.split_at(authority_end);
    let (userinfo, host) = authority.rsplit_once('@')?;
    if !host.is_empty() {
        return None;
    }
    let (user, password) = match userinfo.split_once(':') {
        Some((user, password)) => (user, Some(password)),
        None => (userinfo, None),
    };
    // The userinfo is percent-encoded; a query value is form-encoded, where a
    // literal `+` reads as a space, so a `+` is carried as `%2B`.
    let parameter = |name: &str, value: &str| format!("{name}={}", value.replace('+', "%2B"));
    let moved = [
        (!user.is_empty()).then(|| parameter("user", user)),
        password
            .filter(|p| !p.is_empty())
            .map(|p| parameter("password", p)),
    ];
    let (path, query) = match tail.split_once('?') {
        Some((path, query)) => (path, Some(query.to_string())),
        None => (tail, None),
    };
    let query = query
        .into_iter()
        .chain(moved.into_iter().flatten())
        .collect::<Vec<_>>()
        .join("&");
    Some(if query.is_empty() {
        format!("{scheme}://{path}")
    } else {
        format!("{scheme}://{path}?{query}")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// libpq's Unix-socket URI — credentials beside an empty host, the
    /// socket directory as the `host` parameter — reads as the socket, the
    /// user and the database.
    #[test]
    fn a_socket_uri_with_credentials_reads_as_libpq_reads_it() {
        let opts = pg_connect_options("postgresql://postgres:@/jammi?host=/tmp/pg_sock").unwrap();
        assert_eq!(
            opts.get_socket().map(|p| p.to_string_lossy().into_owned()),
            Some("/tmp/pg_sock".to_string())
        );
        assert_eq!(opts.get_username(), "postgres");
        assert_eq!(opts.get_database(), Some("jammi"));
    }

    /// A password beside an empty host moves with its percent-encoding
    /// intact, a literal `+` included.
    #[test]
    fn a_moved_password_keeps_its_encoding() {
        assert_eq!(
            socket_form_with_parameters("postgres://app:p%40ss+w@/db?host=/run/pg").as_deref(),
            Some("postgres:///db?host=/run/pg&user=app&password=p%40ss%2Bw")
        );
        assert!(pg_connect_options("postgres://app:p%40ss+w@/db?host=/run/pg").is_ok());
    }

    /// A URL with a host, or with no userinfo, is read exactly as written.
    #[test]
    fn a_host_or_a_bare_socket_uri_is_read_as_written() {
        assert_eq!(
            socket_form_with_parameters("postgres://u:p@localhost:5433/db"),
            None
        );
        assert_eq!(socket_form_with_parameters("postgresql:///db?host=/tmp"), None);
        let opts = pg_connect_options("postgres://u:p@localhost:5433/db").unwrap();
        assert_eq!(opts.get_host(), "localhost");
        assert_eq!(opts.get_port(), 5433);
        assert_eq!(opts.get_username(), "u");
        let opts = pg_connect_options("postgresql:///db?host=/tmp&user=u").unwrap();
        assert_eq!(opts.get_username(), "u");
        assert!(opts.get_socket().is_some());
    }
}
