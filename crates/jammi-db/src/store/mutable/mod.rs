//! Mutable companion tables.
//!
//! A mutable companion table is a catalog-registered relation that lives in
//! the same backend database as the catalog (SQLite by default; Postgres in
//! shared deployments), supports transactional `INSERT` / `REPLACE INTO` /
//! `UPDATE` / `DELETE` through DataFusion DML, and federates with Parquet
//! result tables and external sources in one query plan.

pub mod definition;
pub mod postgres;
pub mod provider;
pub mod sink;
pub mod sqlite;
#[cfg(feature = "test-hooks")]
pub mod test_hook;

pub use definition::{
    MutableIndexDef, MutableTableDefinition, MutableTableDefinitionBuilder, MutableTableError,
    MutableTableId,
};

use crate::catalog::backend::BackendImpl;
use crate::sql::quote_ident;
use crate::tenant::TenantId;

/// Backend-specific surface for mutable-table DDL/DML rendering.
///
/// Implementations are pure renderers — no I/O. Execution flows through the
/// associated `BackendImpl` returned by [`MutableBackend::catalog_backend`].
pub trait MutableBackend: Send + Sync {
    /// `CREATE TABLE` statement for `def`. The implicit `tenant_id TEXT`
    /// column is always emitted, per the engine's tenant-identifier
    /// discipline (see
    /// `docs/guide/src/philosophy.md#the-one-rule-everything-else-follows-from`).
    fn create_table_ddl(&self, def: &MutableTableDefinition) -> String;

    /// `CREATE INDEX` statement for one secondary index.
    fn create_index_ddl(&self, def: &MutableTableDefinition, idx: &MutableIndexDef) -> String;

    /// `DROP TABLE` statement. Backend-specific CASCADE semantics.
    fn drop_table_ddl(&self, def: &MutableTableDefinition) -> String;

    /// The most parameters one statement may bind on this backend. A write
    /// wider than this is split into statements that each fit, inside the
    /// same transaction, so it stays one atomic unit.
    fn max_bind_params(&self) -> usize;

    /// Multi-row `INSERT` statement with parameter placeholders. `n_rows` controls
    /// how many rows of `VALUES (…),(…),…` are emitted; total parameter count is
    /// `n_rows * (columns.len() + 1)` (the +1 is the implicit `tenant_id`).
    fn insert_dml(&self, def: &MutableTableDefinition, columns: &[&str], n_rows: usize) -> String;

    /// `DELETE` of the rows whose primary key is one of `n_keys` key tuples,
    /// restricted to the rows `owned` selects. The key tuples bind as
    /// `$1 … $(n_keys * primary_key.len())`, row-major. The row-value `IN`
    /// renders identically on SQLite (3.15+) and Postgres.
    fn delete_keys_dml(&self, def: &MutableTableDefinition, n_keys: usize, owned: &str) -> String {
        let width = def.primary_key.len();
        let key = def
            .primary_key
            .iter()
            .map(|c| quote_ident(c))
            .collect::<Vec<_>>()
            .join(", ");
        let tuples = (0..n_keys)
            .map(|r| {
                let slots = (1..=width)
                    .map(|i| format!("${}", r * width + i))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("({slots})")
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "DELETE FROM {} WHERE ({key}) IN ({tuples}) AND {owned}",
            quote_ident(def.id.as_str())
        )
    }

    /// `SELECT` statement for the `TableProvider::scan` path.
    fn scan_dml(
        &self,
        def: &MutableTableDefinition,
        projection: &[&str],
        predicate: Option<&str>,
        limit: Option<usize>,
    ) -> String;

    /// The matching catalog backend (used to open transactions).
    fn catalog_backend(&self) -> &BackendImpl;
}

/// The rows a session reads: its own and the global (`tenant_id IS NULL`)
/// ones. A session bound to no tenant reads only the global rows.
pub(crate) fn visible_rows(tenant: Option<TenantId>) -> String {
    match tenant {
        Some(t) => format!("(\"tenant_id\" = '{t}' OR \"tenant_id\" IS NULL)"),
        None => owned_rows(None),
    }
}

/// The rows a session may rewrite: only its own. A tenant reads the global
/// rows but never updates or deletes them, so a rewrite can never move a
/// row across tenants.
pub(crate) fn owned_rows(tenant: Option<TenantId>) -> String {
    match tenant {
        Some(t) => format!("\"tenant_id\" = '{t}'"),
        None => "\"tenant_id\" IS NULL".to_string(),
    }
}
