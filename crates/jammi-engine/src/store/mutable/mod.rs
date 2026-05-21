//! Mutable companion tables.
//!
//! A mutable companion table is a catalog-registered relation that lives in
//! the same backend database as the catalog (SQLite by default; Postgres in
//! shared deployments), supports transactional `INSERT` / `UPDATE` / `DELETE`
//! through DataFusion DML, and federates with Parquet result tables and
//! external sources in one query plan.
//!
//! See `docs/plans/cp9-substrate-primitives/SPEC-02-mutable-tables.md`.

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

/// Backend-specific surface for mutable-table DDL/DML rendering.
///
/// Implementations are pure renderers — no I/O. Execution flows through the
/// associated [`BackendImpl`] returned by [`MutableBackend::catalog_backend`].
pub trait MutableBackend: Send + Sync {
    /// `CREATE TABLE` statement for `def`. The implicit `tenant_id TEXT`
    /// column is always emitted (per ADR-00).
    fn create_table_ddl(&self, def: &MutableTableDefinition) -> String;

    /// `CREATE INDEX` statement for one secondary index.
    fn create_index_ddl(&self, def: &MutableTableDefinition, idx: &MutableIndexDef) -> String;

    /// `DROP TABLE` statement. Backend-specific CASCADE semantics.
    fn drop_table_ddl(&self, def: &MutableTableDefinition) -> String;

    /// Multi-row `INSERT` statement with parameter placeholders. `n_rows` controls
    /// how many rows of `VALUES (…),(…),…` are emitted; total parameter count is
    /// `n_rows * (columns.len() + 1)` (the +1 is the implicit `tenant_id`).
    fn insert_dml(&self, def: &MutableTableDefinition, columns: &[&str], n_rows: usize) -> String;

    /// `UPDATE … SET … WHERE` statement.
    fn update_dml(
        &self,
        def: &MutableTableDefinition,
        set_columns: &[&str],
        where_predicate: &str,
    ) -> String;

    /// `DELETE FROM … WHERE` statement.
    fn delete_dml(&self, def: &MutableTableDefinition, where_predicate: &str) -> String;

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
