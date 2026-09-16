//! Backend-agnostic transactional surface for the catalog.
//!
//! The catalog and every repo speak through [`CatalogBackend`]. Backend
//! dialect (SQLite vs Postgres) lives behind the trait edge; engine code sees
//! one trait, one [`Transaction`], one [`BackendError`] taxonomy.
//!
//! Transactions are closure-passing: the caller hands a closure to
//! [`CatalogBackend::transaction`]; the backend opens a transaction, invokes
//! the closure with a `&mut Transaction<'_>`, commits on `Ok(_)`, rolls back
//! on `Err(_)`. The `Transaction<'_>` lifetime is bound to the closure's
//! stack frame so it cannot leak.
//!
//! Parameter binding flows through the engine-owned [`SqlValue`] enum: every
//! call site assembles `&[SqlValue<'_>]` and the backend impl translates to
//! the driver's native representation. Reads use the [`Row`] handle's
//! `get::<T>("name")` API, which delegates to backend-specific row types.

use std::future::Future;
use std::pin::Pin;

use thiserror::Error;

use crate::tenant::TenantId;

/// Backend-agnostic transactional surface.
///
/// Implemented by [`super::backend_sqlite::SqliteBackend`] and
/// [`super::backend_postgres::PostgresBackend`]. The generic `transaction`
/// method is not `dyn`-compatible (Rust object-safety rules disallow generic
/// methods in trait objects), so the engine stores backends behind the
/// concrete [`BackendImpl`] enum, not `Arc<dyn CatalogBackend>`. The trait
/// still serves as the contract for generic helpers (e.g. the catalog's
/// migration runner) and for backend implementations to honor.
pub trait CatalogBackend: Send + Sync {
    /// Run `f` inside one backend transaction. Commits on `Ok(_)`, rolls back
    /// on `Err(_)`. The `&mut Transaction<'_>` cannot escape `f`.
    ///
    /// An implementation may spawn a detached task internally: the SQLite
    /// backend opens its uncancellable `BEGIN` on `tokio::spawn(...).await`. So
    /// this future must be *driven by a live runtime that can poll that
    /// spawned task* — never blocked on with `Handle::block_on` from inside a
    /// runtime worker thread, which would pin the worker on the join handle
    /// while the spawned begin starves. Awaiting it normally (multi- or
    /// single-thread runtime) is fine.
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
        R: Send + 'a;

    /// Apply pending migrations to bring the catalog to the latest schema.
    /// Idempotent.
    fn migrate(&self) -> Pin<Box<dyn Future<Output = Result<(), BackendError>> + Send + '_>>;

    /// Cheap reachability test. Issued by health-endpoint consumers; never
    /// takes a lock and never opens a transaction. Implementations run
    /// `SELECT 1` against the connection pool and surface pool failures as
    /// [`BackendError::Unavailable`] via [`classify`].
    fn ping(&self) -> Pin<Box<dyn Future<Output = Result<(), BackendError>> + Send + '_>>;

    /// Close the connection pool and wait for every connection to be returned
    /// and shut down. **Idempotent**, and the only deterministic release
    /// point.
    ///
    /// Dropping a catalog handle is NOT a release: `sqlx` hands a returned
    /// connection to a background task to close, so after `drop(catalog)` the
    /// pool's connections — and, for SQLite, the OS file locks and the
    /// `catalog.db-wal` sidecar they keep alive — outlive the drop for an
    /// unbounded time (measured: still held after blocking indefinitely,
    /// released after a single 50 ms await). Under the process-exclusive VFS
    /// the SQLite backend uses (see
    /// [`super::backend_sqlite`]) that matters to correctness and not just to
    /// tidiness: the exclusive lock is what refuses a second process, so a
    /// process that seeds a catalog and then hands the directory to another
    /// process MUST await this before spawning it.
    ///
    /// Awaiting this future waits for outstanding checkouts, so a caller that
    /// holds a live `Transaction` across it will wait for that transaction to
    /// finish.
    fn close(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;

    /// Backend identity for telemetry and dialect-conditional code paths.
    fn backend_kind(&self) -> BackendKind;

    /// The connection pool's `max_connections`. Used to size the concurrent
    /// tail-replay semaphore owned by
    /// [`crate::source::mutable::MutableTableRegistry`] (its `replay_permits`
    /// field) — sized `pool_size - 2` (min 1) — so tail replays can never
    /// starve publishers of pool connections on either backend (SQLite's pool
    /// is a hardcoded 8, see `backend_sqlite.rs`'s `open`).
    fn pool_size(&self) -> u32;
}

/// Dynamic-dispatch wrapper over the concrete backend implementations. Used
/// in place of `Arc<dyn CatalogBackend>` (which is impossible due to
/// `transaction`'s generic parameters).
pub enum BackendImpl {
    Sqlite(std::sync::Arc<super::backend_sqlite::SqliteBackend>),
    Postgres(std::sync::Arc<super::backend_postgres::PostgresBackend>),
}

impl BackendImpl {
    /// Run `f` inside one backend transaction. Dispatches to the concrete
    /// backend's `transaction` method. Same contract as
    /// [`CatalogBackend::transaction`].
    pub fn transaction<'a, F, R>(
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
        match self {
            BackendImpl::Sqlite(b) => b.transaction(opts, f),
            BackendImpl::Postgres(b) => b.transaction(opts, f),
        }
    }

    pub fn migrate(&self) -> Pin<Box<dyn Future<Output = Result<(), BackendError>> + Send + '_>> {
        match self {
            BackendImpl::Sqlite(b) => b.migrate(),
            BackendImpl::Postgres(b) => b.migrate(),
        }
    }

    /// Dispatch [`CatalogBackend::ping`] to the inner backend.
    pub fn ping(&self) -> Pin<Box<dyn Future<Output = Result<(), BackendError>> + Send + '_>> {
        match self {
            BackendImpl::Sqlite(b) => b.ping(),
            BackendImpl::Postgres(b) => b.ping(),
        }
    }

    /// Dispatch [`CatalogBackend::close`] to the inner backend. Same
    /// contract: idempotent, and the only deterministic release point for the
    /// pool's connections and (for SQLite) the file locks they hold.
    pub fn close(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        match self {
            BackendImpl::Sqlite(b) => b.close(),
            BackendImpl::Postgres(b) => b.close(),
        }
    }

    /// Construct a [`BackendImpl::Sqlite`] by opening (or creating) the
    /// catalog DB at `path`. Migrations are *not* run here — call
    /// [`BackendImpl::migrate`] after wiring.
    pub async fn sqlite_from_path(path: &std::path::Path) -> Result<Self, BackendError> {
        let sqlite = super::backend_sqlite::SqliteBackend::open(path).await?;
        Ok(Self::Sqlite(sqlite))
    }

    /// Construct a [`BackendImpl::Postgres`] from a connection URL and pool
    /// options. Migrations are *not* run here — call
    /// [`BackendImpl::migrate`] after wiring.
    pub async fn postgres_from_url(
        url: &str,
        pool_size: u32,
        max_lifetime_secs: Option<u32>,
    ) -> Result<Self, BackendError> {
        let pg = super::backend_postgres::PostgresBackend::open_with_options(
            url,
            pool_size,
            max_lifetime_secs,
        )
        .await?;
        Ok(Self::Postgres(pg))
    }

    pub fn backend_kind(&self) -> BackendKind {
        match self {
            BackendImpl::Sqlite(b) => b.backend_kind(),
            BackendImpl::Postgres(b) => b.backend_kind(),
        }
    }

    /// Dispatch [`CatalogBackend::pool_size`] to the inner backend.
    pub fn pool_size(&self) -> u32 {
        match self {
            BackendImpl::Sqlite(b) => b.pool_size(),
            BackendImpl::Postgres(b) => b.pool_size(),
        }
    }

    /// The park on this backend's connection returns (see
    /// [`super::pool_test_hooks`]).
    #[cfg(feature = "test-hooks")]
    pub fn return_park(&self) -> &std::sync::Arc<super::pool_test_hooks::ReturnPark> {
        match self {
            BackendImpl::Sqlite(b) => b.return_park(),
            BackendImpl::Postgres(b) => b.return_park(),
        }
    }
}

/// Options applied to a transaction at `BEGIN` time.
#[derive(Debug, Clone, Copy)]
pub struct TxOptions {
    pub isolation: IsolationLevel,
    pub read_only: bool,
}

impl Default for TxOptions {
    fn default() -> Self {
        Self {
            isolation: IsolationLevel::ReadCommitted,
            read_only: false,
        }
    }
}

/// SQL isolation level, mapped to the appropriate backend statement.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum IsolationLevel {
    /// Postgres `READ COMMITTED`; SQLite implicit snapshot in WAL mode.
    #[default]
    ReadCommitted,
    /// Postgres `REPEATABLE READ`; SQLite stricter than default.
    RepeatableRead,
    /// Postgres `SERIALIZABLE`; SQLite stricter than default.
    Serializable,
}

/// Which backend this `CatalogBackend` wraps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Sqlite,
    Postgres,
}

/// Lifetime-scoped transactional handle handed to a [`CatalogBackend::transaction`]
/// closure. Holds a borrowed reference to the backend's connection (the
/// transaction itself is owned by the backend's `transaction` method).
pub struct Transaction<'tx> {
    pub(crate) inner: TxInner<'tx>,
    tenant: Option<TenantId>,
}

pub(crate) enum TxInner<'tx> {
    Sqlite(&'tx mut sqlx::SqliteConnection),
    Postgres(&'tx mut sqlx::PgConnection),
}

impl<'tx> Transaction<'tx> {
    pub(crate) fn new_sqlite(conn: &'tx mut sqlx::SqliteConnection) -> Self {
        Self {
            inner: TxInner::Sqlite(conn),
            tenant: None,
        }
    }

    pub(crate) fn new_postgres(conn: &'tx mut sqlx::PgConnection) -> Self {
        Self {
            inner: TxInner::Postgres(conn),
            tenant: None,
        }
    }

    /// Execute a statement that returns no rows. Returns the affected row count.
    pub async fn execute(
        &mut self,
        stmt: &str,
        params: &[SqlValue<'_>],
    ) -> Result<u64, BackendError> {
        match &mut self.inner {
            TxInner::Sqlite(c) => {
                let mut q = sqlx::query(stmt);
                for p in params {
                    q = bind_sqlite(q, p);
                }
                let res = q.execute(&mut **c).await.map_err(classify)?;
                Ok(res.rows_affected())
            }
            TxInner::Postgres(c) => {
                let mut q = sqlx::query(stmt);
                for v in params {
                    q = bind_postgres(q, v);
                }
                let res = q.execute(&mut **c).await.map_err(classify)?;
                Ok(res.rows_affected())
            }
        }
    }

    /// Execute a query and map each row.
    pub async fn query<F, R>(
        &mut self,
        stmt: &str,
        params: &[SqlValue<'_>],
        mut row_mapper: F,
    ) -> Result<Vec<R>, BackendError>
    where
        F: FnMut(&Row<'_>) -> Result<R, BackendError>,
    {
        match &mut self.inner {
            TxInner::Sqlite(c) => {
                let mut q = sqlx::query(stmt);
                for p in params {
                    q = bind_sqlite(q, p);
                }
                let rows = q.fetch_all(&mut **c).await.map_err(classify)?;
                rows.iter()
                    .map(|r| {
                        row_mapper(&Row {
                            inner: RowInner::Sqlite(r),
                        })
                    })
                    .collect()
            }
            TxInner::Postgres(c) => {
                let mut q = sqlx::query(stmt);
                for v in params {
                    q = bind_postgres(q, v);
                }
                let rows = q.fetch_all(&mut **c).await.map_err(classify)?;
                rows.iter()
                    .map(|r| {
                        row_mapper(&Row {
                            inner: RowInner::Postgres(r),
                        })
                    })
                    .collect()
            }
        }
    }

    /// Execute a query expected to return at most one row.
    pub async fn query_opt<F, R>(
        &mut self,
        stmt: &str,
        params: &[SqlValue<'_>],
        row_mapper: F,
    ) -> Result<Option<R>, BackendError>
    where
        F: FnMut(&Row<'_>) -> Result<R, BackendError>,
    {
        let mut rows = self.query(stmt, params, row_mapper).await?;
        Ok(rows.pop())
    }

    /// Bind a tenant for this transaction. Read by [`Self::assert_tenant_matches`]
    /// to enforce the write-side guard described in SPEC-03 §7.
    pub fn set_tenant(&mut self, tenant: Option<TenantId>) {
        self.tenant = tenant;
    }

    pub fn tenant(&self) -> Option<TenantId> {
        self.tenant
    }

    /// Assert that `row_tenant` matches the transaction's bound tenant.
    /// Returns [`BackendError::TenantMismatch`] otherwise. This is the
    /// defence-in-depth write-side guard: every code path that emits a
    /// tenant-aware row should call this before issuing the `INSERT` /
    /// `UPDATE` so the engine never persists a row whose tenant disagrees
    /// with the session that produced it.
    pub fn assert_tenant_matches(
        &self,
        row_tenant: Option<TenantId>,
        table: &str,
    ) -> Result<(), BackendError> {
        if row_tenant == self.tenant {
            Ok(())
        } else {
            Err(BackendError::TenantMismatch {
                table: table.to_string(),
                expected: self.tenant,
                got: row_tenant,
            })
        }
    }
}

/// The SQL type a bound `NULL` must carry so a dynamically-typed backend
/// (SQLite) and a statically-typed one (Postgres) bind the same column kind
/// whether a given cell is null or not — Postgres rejects a text null bound
/// into a non-text column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqlNullType {
    Bool,
    Int,
    Float,
    Text,
    Bytes,
}

/// A full-resolution, lexicographically-sortable creation-timestamp string,
/// identical byte-for-byte whichever backend a catalog row lands on.
///
/// Exists because a SQL-side `CURRENT_TIMESTAMP` default is not a portable
/// ordering key: SQLite renders it at second resolution with no offset,
/// Postgres at microsecond resolution with one, so two backends fed the same
/// migration DDL populate a `created_at` column with values of different
/// resolution and shape. Nanosecond precision here means two rows an
/// application creates in the same call, or even the same second, still
/// resolve by plain string order — an `ORDER BY created_at DESC` query never
/// needs a backend-specific tiebreak column (SQLite's `rowid` has no
/// Postgres analog, and even Postgres's own `ctid` is not creation-ordered
/// after a `VACUUM`).
///
/// Callers stamp this once at row-creation time and bind it as an ordinary
/// `TEXT` value — the column stays a plain string on both backends, so
/// nothing downstream needs to parse it as a timestamp to sort on it.
pub fn now_sortable() -> String {
    chrono::Utc::now()
        .format("%Y-%m-%dT%H:%M:%S%.9fZ")
        .to_string()
}

/// Engine-owned parameter value. Backend impls translate to driver-native
/// types in `bind_sqlite` / `bind_postgres`.
#[derive(Debug, Clone)]
pub enum SqlValue<'v> {
    Null(SqlNullType),
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(&'v str),
    TextOwned(String),
    Bytes(&'v [u8]),
    BytesOwned(Vec<u8>),
    Uuid(uuid::Uuid),
    Json(serde_json::Value),
    Timestamp(chrono::DateTime<chrono::Utc>),
}

impl<'v> From<&'v str> for SqlValue<'v> {
    fn from(v: &'v str) -> Self {
        SqlValue::Text(v)
    }
}
impl From<String> for SqlValue<'static> {
    fn from(v: String) -> Self {
        SqlValue::TextOwned(v)
    }
}
impl From<i32> for SqlValue<'_> {
    fn from(v: i32) -> Self {
        SqlValue::Int(v as i64)
    }
}
impl From<i64> for SqlValue<'_> {
    fn from(v: i64) -> Self {
        SqlValue::Int(v)
    }
}
impl From<u32> for SqlValue<'_> {
    fn from(v: u32) -> Self {
        SqlValue::Int(v as i64)
    }
}
impl From<u64> for SqlValue<'_> {
    fn from(v: u64) -> Self {
        SqlValue::Int(v as i64)
    }
}
impl From<bool> for SqlValue<'_> {
    fn from(v: bool) -> Self {
        SqlValue::Bool(v)
    }
}
impl From<f64> for SqlValue<'_> {
    fn from(v: f64) -> Self {
        SqlValue::Float(v)
    }
}
impl From<uuid::Uuid> for SqlValue<'_> {
    fn from(v: uuid::Uuid) -> Self {
        SqlValue::Uuid(v)
    }
}
impl From<serde_json::Value> for SqlValue<'_> {
    fn from(v: serde_json::Value) -> Self {
        SqlValue::Json(v)
    }
}
impl From<chrono::DateTime<chrono::Utc>> for SqlValue<'_> {
    fn from(v: chrono::DateTime<chrono::Utc>) -> Self {
        SqlValue::Timestamp(v)
    }
}
/// Declares the [`SqlNullType`] a bound `NULL` must carry when `Option<T>`
/// is `None` — the null must bind the same SQL type `T`'s `Into<SqlValue>`
/// impl produces for `Some`, so a Postgres column sees one consistent type
/// across null and non-null rows.
pub trait HasSqlNull {
    const NULL_TYPE: SqlNullType;
}

impl HasSqlNull for String {
    const NULL_TYPE: SqlNullType = SqlNullType::Text;
}
impl HasSqlNull for &str {
    const NULL_TYPE: SqlNullType = SqlNullType::Text;
}
impl HasSqlNull for i64 {
    const NULL_TYPE: SqlNullType = SqlNullType::Int;
}
impl HasSqlNull for i32 {
    const NULL_TYPE: SqlNullType = SqlNullType::Int;
}
impl HasSqlNull for f64 {
    const NULL_TYPE: SqlNullType = SqlNullType::Float;
}
impl HasSqlNull for bool {
    const NULL_TYPE: SqlNullType = SqlNullType::Bool;
}
impl<'v, T> From<Option<T>> for SqlValue<'v>
where
    T: Into<SqlValue<'v>> + HasSqlNull,
{
    fn from(v: Option<T>) -> Self {
        v.map(Into::into).unwrap_or(SqlValue::Null(T::NULL_TYPE))
    }
}

/// Row handle handed to `row_mapper`. Column lookup is by name.
pub struct Row<'r> {
    inner: RowInner<'r>,
}

enum RowInner<'r> {
    Sqlite(&'r sqlx::sqlite::SqliteRow),
    Postgres(&'r sqlx::postgres::PgRow),
}

impl<'r> Row<'r> {
    /// Get a non-null column value, decoding to `T` via [`FromSqlValue`].
    pub fn get<T: FromSqlValue>(&self, name: &str) -> Result<T, BackendError> {
        match self.inner {
            RowInner::Sqlite(r) => T::from_sqlite_row(r, name),
            RowInner::Postgres(r) => T::from_postgres_row(r, name),
        }
    }

    /// Get a nullable column value. `Ok(None)` if the column is NULL.
    pub fn try_get<T: FromSqlValue>(&self, name: &str) -> Result<Option<T>, BackendError> {
        match self.inner {
            RowInner::Sqlite(r) => T::from_sqlite_row_opt(r, name),
            RowInner::Postgres(r) => T::from_postgres_row_opt(r, name),
        }
    }
}

/// Read-path conversion trait. Implementations decode a column value from the
/// backend-native row representation.
pub trait FromSqlValue: Sized {
    fn from_sqlite_row(row: &sqlx::sqlite::SqliteRow, name: &str) -> Result<Self, BackendError>;
    fn from_postgres_row(row: &sqlx::postgres::PgRow, name: &str) -> Result<Self, BackendError>;
    fn from_sqlite_row_opt(
        row: &sqlx::sqlite::SqliteRow,
        name: &str,
    ) -> Result<Option<Self>, BackendError>;
    fn from_postgres_row_opt(
        row: &sqlx::postgres::PgRow,
        name: &str,
    ) -> Result<Option<Self>, BackendError>;
}

macro_rules! impl_from_sql_primitive {
    ($t:ty) => {
        impl FromSqlValue for $t {
            fn from_sqlite_row(
                row: &sqlx::sqlite::SqliteRow,
                name: &str,
            ) -> Result<Self, BackendError> {
                use sqlx::Row as _;
                row.try_get::<$t, _>(name).map_err(classify)
            }
            fn from_postgres_row(
                row: &sqlx::postgres::PgRow,
                name: &str,
            ) -> Result<Self, BackendError> {
                use sqlx::Row as _;
                row.try_get::<$t, _>(name).map_err(classify)
            }
            fn from_sqlite_row_opt(
                row: &sqlx::sqlite::SqliteRow,
                name: &str,
            ) -> Result<Option<Self>, BackendError> {
                use sqlx::Row as _;
                row.try_get::<Option<$t>, _>(name).map_err(classify)
            }
            fn from_postgres_row_opt(
                row: &sqlx::postgres::PgRow,
                name: &str,
            ) -> Result<Option<Self>, BackendError> {
                use sqlx::Row as _;
                row.try_get::<Option<$t>, _>(name).map_err(classify)
            }
        }
    };
}

impl_from_sql_primitive!(String);
impl_from_sql_primitive!(i64);
impl_from_sql_primitive!(i32);
// `i16` decodes SMALLINT/INT2 columns (Postgres rejects an `i32` bind
// against INT2; sqlx's SQLite driver also has a native `i16` mapping).
// Needed by the storage-typed replay decode path (Int8/Int16 columns).
impl_from_sql_primitive!(i16);
impl_from_sql_primitive!(bool);
impl_from_sql_primitive!(f64);
// `f32` decodes REAL/FLOAT4 columns — `f64` rejects them on Postgres.
// Needed by the storage-typed replay decode path (Float32 columns).
impl_from_sql_primitive!(f32);
impl_from_sql_primitive!(Vec<u8>);

impl FromSqlValue for uuid::Uuid {
    fn from_sqlite_row(row: &sqlx::sqlite::SqliteRow, name: &str) -> Result<Self, BackendError> {
        use sqlx::Row as _;
        let s: String = row.try_get(name).map_err(classify)?;
        uuid::Uuid::parse_str(&s).map_err(|e| BackendError::TypeConversion {
            column: name.to_string(),
            detail: e.to_string(),
        })
    }
    fn from_postgres_row(row: &sqlx::postgres::PgRow, name: &str) -> Result<Self, BackendError> {
        use sqlx::Row as _;
        row.try_get(name).map_err(classify)
    }
    fn from_sqlite_row_opt(
        row: &sqlx::sqlite::SqliteRow,
        name: &str,
    ) -> Result<Option<Self>, BackendError> {
        use sqlx::Row as _;
        let raw: Option<String> = row.try_get(name).map_err(classify)?;
        raw.map(|s| {
            uuid::Uuid::parse_str(&s).map_err(|e| BackendError::TypeConversion {
                column: name.to_string(),
                detail: e.to_string(),
            })
        })
        .transpose()
    }
    fn from_postgres_row_opt(
        row: &sqlx::postgres::PgRow,
        name: &str,
    ) -> Result<Option<Self>, BackendError> {
        use sqlx::Row as _;
        row.try_get(name).map_err(classify)
    }
}

impl FromSqlValue for chrono::DateTime<chrono::Utc> {
    fn from_sqlite_row(row: &sqlx::sqlite::SqliteRow, name: &str) -> Result<Self, BackendError> {
        use sqlx::Row as _;
        row.try_get::<chrono::NaiveDateTime, _>(name)
            .map(|ndt| chrono::DateTime::from_naive_utc_and_offset(ndt, chrono::Utc))
            .map_err(classify)
    }
    fn from_postgres_row(row: &sqlx::postgres::PgRow, name: &str) -> Result<Self, BackendError> {
        use sqlx::Row as _;
        row.try_get(name).map_err(classify)
    }
    fn from_sqlite_row_opt(
        row: &sqlx::sqlite::SqliteRow,
        name: &str,
    ) -> Result<Option<Self>, BackendError> {
        use sqlx::Row as _;
        let raw: Option<chrono::NaiveDateTime> = row.try_get(name).map_err(classify)?;
        Ok(raw.map(|ndt| chrono::DateTime::from_naive_utc_and_offset(ndt, chrono::Utc)))
    }
    fn from_postgres_row_opt(
        row: &sqlx::postgres::PgRow,
        name: &str,
    ) -> Result<Option<Self>, BackendError> {
        use sqlx::Row as _;
        row.try_get(name).map_err(classify)
    }
}

impl FromSqlValue for serde_json::Value {
    fn from_sqlite_row(row: &sqlx::sqlite::SqliteRow, name: &str) -> Result<Self, BackendError> {
        use sqlx::Row as _;
        let s: String = row.try_get(name).map_err(classify)?;
        serde_json::from_str(&s).map_err(|e| BackendError::TypeConversion {
            column: name.to_string(),
            detail: e.to_string(),
        })
    }
    fn from_postgres_row(row: &sqlx::postgres::PgRow, name: &str) -> Result<Self, BackendError> {
        use sqlx::Row as _;
        row.try_get(name).map_err(classify)
    }
    fn from_sqlite_row_opt(
        row: &sqlx::sqlite::SqliteRow,
        name: &str,
    ) -> Result<Option<Self>, BackendError> {
        use sqlx::Row as _;
        let raw: Option<String> = row.try_get(name).map_err(classify)?;
        raw.map(|s| {
            serde_json::from_str(&s).map_err(|e| BackendError::TypeConversion {
                column: name.to_string(),
                detail: e.to_string(),
            })
        })
        .transpose()
    }
    fn from_postgres_row_opt(
        row: &sqlx::postgres::PgRow,
        name: &str,
    ) -> Result<Option<Self>, BackendError> {
        use sqlx::Row as _;
        row.try_get(name).map_err(classify)
    }
}

/// Backend-agnostic error taxonomy. Variants are populated by [`classify`]
/// from raw `sqlx::Error`.
#[derive(Debug, Error)]
pub enum BackendError {
    #[error("backend execution failure: {0}")]
    Execution(String),
    #[error("constraint violation on {table}: {detail}")]
    Constraint { table: String, detail: String },
    #[error("backend unavailable: {0}")]
    Unavailable(String),
    #[error("transaction retry required: {0}")]
    Retry(String),
    #[error("migration failure: {0}")]
    Migration(String),
    #[error("type conversion failure on column {column}: {detail}")]
    TypeConversion { column: String, detail: String },
    #[error("tenant mismatch writing {table}: session={expected:?}, row={got:?}")]
    TenantMismatch {
        table: String,
        expected: Option<TenantId>,
        got: Option<TenantId>,
    },
    /// A transaction-internal refusal sentinel: return this from a
    /// [`CatalogBackend::transaction`] closure to force a ROLLBACK of every
    /// write the closure already issued, naming the row that made the whole
    /// batch unsafe to commit. `.await?`-ing the transaction call would
    /// otherwise fold this into the generic [`crate::error::JammiError::BackendDriver`]
    /// arm; a caller that needs the typed refusal (e.g.
    /// [`crate::catalog::Catalog::delete_result_tables_for_source`]'s
    /// `SourceBusy`) matches on `Err(BackendError::Busy(_))` BEFORE applying
    /// `?`/`.into()` to the transaction's result.
    #[error("busy: {0}")]
    Busy(String),
    #[error("sqlx backend error: {0}")]
    Sqlx(#[from] sqlx::Error),
}

/// Ceiling on the post-`close` drain. Reaching it means a connection never
/// finished closing; the drain gives up and warns rather than hanging the
/// caller's shutdown forever.
pub(crate) const CLOSE_DRAIN_CEILING: std::time::Duration = std::time::Duration::from_secs(30);

/// One step of a shutdown wait.
///
/// Deliberately not `yield_now` in a tight loop: the work being waited for is
/// a *spawned* connection-return task plus a driver thread's `sqlite3_close`,
/// and a `yield_now` spin from inside `block_on` pins the calling core and
/// starves exactly that work (measured: a close that resolves in ~3 ms when
/// waited on politely took ~2 s when spun on). Deliberately not
/// `tokio::time::sleep` either: that requires the runtime's time driver, which
/// an embedding host is free not to enable, and a shutdown path must not panic
/// on a `Builder::new_current_thread()` runtime. Sleeping on a blocking-pool
/// thread needs neither.
pub(crate) async fn shutdown_wait_step() {
    let _ = tokio::task::spawn_blocking(|| {
        std::thread::sleep(std::time::Duration::from_millis(1));
    })
    .await;
}

/// Close `pool` and wait until it holds **no** live connections.
///
/// `sqlx`'s own `Pool::close` is not that barrier, for two reasons.
///
/// First, a `PoolConnection` returns itself to the pool from a task spawned
/// in its `Drop`, so a connection dropped shortly before the close can still
/// be mid-return when `close()` resolves; the driver-level close then lands
/// afterwards. For SQLite that is the difference between a release point and a
/// race: the final `sqlite3_close` is what drops the process-exclusive lock and
/// deletes `catalog.db-wal`, and it was measured landing *after*
/// `Pool::close()` returned roughly one open/close cycle in seven.
///
/// Second — and this is why the wait below re-runs `close` rather than only
/// polling `size()` — one `Pool::close` pass can END with a connection it will
/// never close. The pass (sqlx 0.8, `PoolInner::close`) sets the pool's closed
/// flag, then repeatedly sweeps the idle queue and waits on the pool's permits,
/// returning once it holds every permit — i.e. once no connection is checked
/// out. A return task that read the closed flag as *unset* (it began before
/// the pass and is spending its round trip in the on-release liveness ping)
/// then pushes its connection onto the idle queue and only THEN gives its
/// permit back: the pass's last sweep ran before that push, the permit arrives
/// after it, and the pass returns with the connection idle inside a closed
/// pool. Nothing sweeps a closed pool's idle queue on its own, so `size()`
/// stays at one until the pool is dropped. On a server DRAIN the last catalog
/// write before the session closes is exactly such a drop (the worker join's
/// `workers` row delete), and its return's ping round trip is the window —
/// sub-millisecond on a loopback, milliseconds on a container network — so the
/// barrier waited out its whole ceiling whenever a loaded host lost the race
/// (the compose smoke's `docker compose restart`: 3 ms on three runs, 30 s on
/// the fourth).
///
/// `Pool::close` may be called again, and every pass sweeps the idle queue
/// afresh, so the barrier is a LOOP of close passes: each closes everything
/// idle and waits for everything checked out, and the loop ends when the
/// pool's accounting reads empty. A connection that lands idle after one
/// pass's sweep is closed by the next; a return that reads the flag as set
/// closes its own connection and decrements `size()` itself; a connection
/// checked out for good blocks the pass, as it always did. Reaching the
/// ceiling therefore means a driver-level close that does not complete, and
/// the warning names the consequence per backend.
///
/// `Pool::size()` is decremented by each connection's `DecrementSizeGuard`,
/// which is dropped only after that connection's driver-level close has
/// completed, so `size() == 0` covers every connection the pool's accounting
/// knows about. It was measured **not sufficient on its own** for SQLite —
/// the `-wal` still outlived it by ~3 ms once every few cycles — so the SQLite
/// backend follows this with a wait on SQLite's own release evidence. The loop
/// waits through [`shutdown_wait_step`], which leaves the runtime free to run
/// the return tasks it is waiting on.
pub(crate) async fn close_pool_and_drain<DB: sqlx::Database>(
    pool: &sqlx::Pool<DB>,
    kind: BackendKind,
) {
    let deadline = std::time::Instant::now() + CLOSE_DRAIN_CEILING;
    loop {
        pool.close().await;
        if pool.size() == 0 {
            return;
        }
        if std::time::Instant::now() >= deadline {
            warn_close_ceiling(pool, kind);
            return;
        }
        shutdown_wait_step().await;
    }
}

/// The ceiling warning, with the consequence stated for the backend at hand:
/// a SQLite connection that never closed still holds the catalog file (the
/// process-exclusive lock and the `-wal` with it); a Postgres one is a pooled
/// server connection this process keeps open until it exits.
fn warn_close_ceiling<DB: sqlx::Database>(pool: &sqlx::Pool<DB>, kind: BackendKind) {
    let consequence = match kind {
        BackendKind::Sqlite => "the backing file may still be held",
        BackendKind::Postgres => "a pooled server connection stays open until this process exits",
    };
    tracing::warn!(
        remaining = pool.size(),
        ceiling_secs = CLOSE_DRAIN_CEILING.as_secs(),
        "catalog pool close timed out with connections still open; {consequence}"
    );
}

/// Classify a raw `sqlx::Error` into the engine-owned [`BackendError`]
/// taxonomy. Constraint and retry detection rely on backend-specific
/// `DatabaseError` flags exposed by sqlx.
pub fn classify(err: sqlx::Error) -> BackendError {
    use sqlx::Error::*;
    match &err {
        Database(db_err) if db_err.is_unique_violation() => BackendError::Constraint {
            table: db_err.table().unwrap_or("<unknown>").to_string(),
            detail: db_err.message().to_string(),
        },
        Database(db_err) if db_err.code().as_deref() == Some("40001") => {
            BackendError::Retry(db_err.message().to_string())
        }
        PoolTimedOut | PoolClosed => BackendError::Unavailable(err.to_string()),
        _ => BackendError::Sqlx(err),
    }
}

fn bind_sqlite<'q>(
    q: sqlx::query::Query<'q, sqlx::Sqlite, sqlx::sqlite::SqliteArguments<'q>>,
    v: &'q SqlValue<'_>,
) -> sqlx::query::Query<'q, sqlx::Sqlite, sqlx::sqlite::SqliteArguments<'q>> {
    match v {
        SqlValue::Null(SqlNullType::Bool) => q.bind(Option::<bool>::None),
        SqlValue::Null(SqlNullType::Int) => q.bind(Option::<i64>::None),
        SqlValue::Null(SqlNullType::Float) => q.bind(Option::<f64>::None),
        SqlValue::Null(SqlNullType::Text) => q.bind(Option::<String>::None),
        SqlValue::Null(SqlNullType::Bytes) => q.bind(Option::<Vec<u8>>::None),
        SqlValue::Bool(b) => q.bind(*b),
        SqlValue::Int(i) => q.bind(*i),
        SqlValue::Float(f) => q.bind(*f),
        SqlValue::Text(s) => q.bind(s.to_string()),
        SqlValue::TextOwned(s) => q.bind(s.clone()),
        SqlValue::Bytes(b) => q.bind(b.to_vec()),
        SqlValue::BytesOwned(b) => q.bind(b.clone()),
        SqlValue::Uuid(u) => q.bind(u.to_string()),
        SqlValue::Json(j) => q.bind(j.to_string()),
        SqlValue::Timestamp(t) => q.bind(t.naive_utc()),
    }
}

fn bind_postgres<'q>(
    q: sqlx::query::Query<'q, sqlx::Postgres, sqlx::postgres::PgArguments>,
    v: &'q SqlValue<'_>,
) -> sqlx::query::Query<'q, sqlx::Postgres, sqlx::postgres::PgArguments> {
    match v {
        SqlValue::Null(SqlNullType::Bool) => q.bind(Option::<bool>::None),
        SqlValue::Null(SqlNullType::Int) => q.bind(Option::<i64>::None),
        SqlValue::Null(SqlNullType::Float) => q.bind(Option::<f64>::None),
        SqlValue::Null(SqlNullType::Text) => q.bind(Option::<String>::None),
        SqlValue::Null(SqlNullType::Bytes) => q.bind(Option::<Vec<u8>>::None),
        SqlValue::Bool(b) => q.bind(*b),
        SqlValue::Int(i) => q.bind(*i),
        SqlValue::Float(f) => q.bind(*f),
        SqlValue::Text(s) => q.bind(s.to_string()),
        SqlValue::TextOwned(s) => q.bind(s.clone()),
        SqlValue::Bytes(b) => q.bind(b.to_vec()),
        SqlValue::BytesOwned(b) => q.bind(b.clone()),
        SqlValue::Uuid(u) => q.bind(*u),
        SqlValue::Json(j) => q.bind(j.clone()),
        SqlValue::Timestamp(t) => q.bind(*t),
    }
}

#[cfg(test)]
mod close_barrier_tests {
    //! [`close_pool_and_drain`] against the race its documentation names: a
    //! connection whose return began before the close and lands on the idle
    //! queue after the close's final sweep. The return is held open inside
    //! `after_release` (the slot the `test-hooks` park uses too) and the close
    //! starts only once that return has announced it is there, so the
    //! interleaving is fixed, never raced. Both backends, since the pool is
    //! `sqlx`'s on both.

    use std::time::{Duration, Instant};

    use tokio::sync::mpsc;

    use super::{close_pool_and_drain, BackendKind};

    /// How long a return is held open. Far above the joins a real close
    /// waits behind (milliseconds), far below `CLOSE_DRAIN_CEILING`.
    const RETURN_PARK: Duration = Duration::from_millis(300);
    /// A close that waits out the ceiling on the leaked connection takes
    /// `CLOSE_DRAIN_CEILING`; one that sweeps it takes about `RETURN_PARK`.
    const BOUND: Duration = Duration::from_secs(5);

    /// Every return announces itself on `entered` as it begins its park.
    fn park_returns<DB: sqlx::Database>(
        options: sqlx::pool::PoolOptions<DB>,
        entered: mpsc::UnboundedSender<()>,
    ) -> sqlx::pool::PoolOptions<DB> {
        options.after_release(move |_conn, _meta| {
            let entered = entered.clone();
            Box::pin(async move {
                let _ = entered.send(());
                tokio::time::sleep(RETURN_PARK).await;
                Ok(true)
            })
        })
    }

    /// Check one connection out and drop it, and start the close only once
    /// that return is inside its park — the closed flag read as unset, the
    /// idle push still ahead. (The pool's connect-time validation connection
    /// is released to the idle queue directly, never through the return
    /// task, so the first parked return is this one.) Returns the close's
    /// wall clock; asserts the pool's accounting reads empty afterwards.
    async fn close_against_one_in_flight_return<DB: sqlx::Database>(
        pool: &sqlx::Pool<DB>,
        entered: &mut mpsc::UnboundedReceiver<()>,
        kind: BackendKind,
    ) -> Duration {
        let conn = pool.acquire().await.expect("acquire");
        drop(conn);
        entered
            .recv()
            .await
            .expect("the dropped connection's return parks");
        let started = Instant::now();
        close_pool_and_drain(pool, kind).await;
        let elapsed = started.elapsed();
        assert_eq!(
            pool.size(),
            0,
            "the barrier returned with connections still counted (idle = {})",
            pool.num_idle()
        );
        elapsed
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sqlite_barrier_closes_a_connection_returned_during_the_close() {
        let dir = tempfile::tempdir().unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let pool = park_returns(
            sqlx::sqlite::SqlitePoolOptions::new().max_connections(8),
            tx,
        )
        .connect_with(
            sqlx::sqlite::SqliteConnectOptions::new()
                .filename(dir.path().join("catalog.db"))
                .create_if_missing(true),
        )
        .await
        .expect("open sqlite pool");
        let elapsed = close_against_one_in_flight_return(&pool, &mut rx, BackendKind::Sqlite).await;
        assert!(
            elapsed < BOUND,
            "the close took {elapsed:?}: it waited out the ceiling on a connection it never swept"
        );
    }

    /// Live: requires `JAMMI_TEST_PG_URL`; skips (never `#[ignore]`)
    /// otherwise.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn postgres_barrier_closes_a_connection_returned_during_the_close() {
        let Some(url) = jammi_test_utils::pg_url_for_tests() else {
            eprintln!(
                "skipping postgres_barrier_closes_a_connection_returned_during_the_close: \
                 JAMMI_TEST_PG_URL unset"
            );
            return;
        };
        let (tx, mut rx) = mpsc::unbounded_channel();
        let pool = park_returns(sqlx::postgres::PgPoolOptions::new().max_connections(8), tx)
            .connect_with(
                url.parse::<sqlx::postgres::PgConnectOptions>()
                    .expect("pg url"),
            )
            .await
            .expect("open postgres pool");
        let elapsed =
            close_against_one_in_flight_return(&pool, &mut rx, BackendKind::Postgres).await;
        assert!(
            elapsed < BOUND,
            "the close took {elapsed:?}: it waited out the ceiling on a connection it never swept"
        );
    }
}
