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

    /// Backend identity for telemetry and dialect-conditional code paths.
    fn backend_kind(&self) -> BackendKind;
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

/// Engine-owned parameter value. Backend impls translate to driver-native
/// types in `bind_sqlite` / `bind_postgres`.
#[derive(Debug, Clone)]
pub enum SqlValue<'v> {
    Null,
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
impl<'v, T> From<Option<T>> for SqlValue<'v>
where
    T: Into<SqlValue<'v>>,
{
    fn from(v: Option<T>) -> Self {
        v.map(Into::into).unwrap_or(SqlValue::Null)
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
impl_from_sql_primitive!(bool);
impl_from_sql_primitive!(f64);
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
    #[error("sqlx backend error: {0}")]
    Sqlx(#[from] sqlx::Error),
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
        SqlValue::Null => q.bind(Option::<String>::None),
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
        SqlValue::Null => q.bind(Option::<String>::None),
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
