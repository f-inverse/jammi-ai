//! Migration runner. Applies the SQL constants from [`super::schema`] in
//! order, tracking which have been applied in an `applied_migrations` ledger.
//! Backend-agnostic: works through [`CatalogBackend`].

use super::backend::{BackendError, CatalogBackend, SqlValue, TxOptions};
use super::schema;

/// Ordered list of migrations. Each entry's first element is the name
/// recorded in `applied_migrations`; the second is the SQL DDL.
///
/// Entries are append-only. Once a migration name has been published it must
/// never be renamed, reordered, or have its SQL changed — doing so would
/// break upgrades from existing catalogs.
const MIGRATIONS: &[(&str, &str)] = &[
    ("001_core_tables", schema::MIGRATION_001_CORE_TABLES),
    ("002_result_tables", schema::MIGRATION_002_RESULT_TABLES),
    ("003_eval_columns", schema::MIGRATION_003_EVAL_COLUMNS),
    (
        "004_drop_embedding_sets",
        schema::MIGRATION_004_DROP_EMBEDDING_SETS,
    ),
    ("005_tenant_scope", schema::MIGRATION_005_TENANT_SCOPE),
    ("006_channel_columns", schema::MIGRATION_006_CHANNEL_COLUMNS),
    ("007_mutable_tables", schema::MIGRATION_007_MUTABLE_TABLES),
];

const APPLIED_MIGRATIONS_DDL: &str = r#"
CREATE TABLE IF NOT EXISTS applied_migrations (
    name        TEXT PRIMARY KEY,
    applied_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"#;

/// Apply all pending migrations. Idempotent.
pub(crate) async fn run<B: CatalogBackend + ?Sized>(backend: &B) -> Result<(), BackendError> {
    backend
        .transaction(TxOptions::default(), |tx| {
            Box::pin(async move {
                tx.execute(APPLIED_MIGRATIONS_DDL, &[]).await?;
                let applied: Vec<String> = tx
                    .query("SELECT name FROM applied_migrations", &[], |row| {
                        row.get::<String>("name")
                    })
                    .await?;
                let applied_set: std::collections::HashSet<&str> =
                    applied.iter().map(String::as_str).collect();

                for (name, ddl) in MIGRATIONS {
                    if applied_set.contains(name) {
                        continue;
                    }
                    // Migration SQL constants contain multiple statements
                    // separated by `;`. sqlx's `execute()` for SQLite only
                    // runs the first statement; we split and run each in
                    // turn, inside the same transaction.
                    for stmt in split_statements(ddl) {
                        tx.execute(&stmt, &[]).await?;
                    }
                    tx.execute(
                        "INSERT INTO applied_migrations (name) VALUES ($1)",
                        &[SqlValue::Text(name)],
                    )
                    .await?;
                }
                Ok(())
            })
        })
        .await
}

/// Split a multi-statement SQL string on `;` boundaries, ignoring empty
/// trailing fragments and stripping leading whitespace. SQL comments are not
/// removed (the migration constants don't contain them).
fn split_statements(sql: &str) -> Vec<String> {
    sql.split(';')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}
