//! Migration runner. Applies the SQL constants from [`super::schema`] in
//! order, tracking which have been applied in an `applied_migrations` ledger.
//! Backend-agnostic: works through [`CatalogBackend`].

use super::backend::{BackendError, CatalogBackend, SqlValue, TxOptions};
use super::schema;

/// Ordered list of migrations. Each entry's first element is the name
/// recorded in `applied_migrations`; the second is the SQL DDL.
///
/// Entries are append-only: a new migration is appended; names are never
/// renamed or reordered. The SQL itself may be edited only when the change
/// is invisible to the resulting schema — e.g. swapping a backend-specific
/// `DEFAULT` expression for a portable one that produces the same column
/// type and constraint set. Any change that alters the schema shape (new
/// column, dropped column, different constraint) belongs in a new migration.
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
    (
        "008_mutable_order_column",
        schema::MIGRATION_008_MUTABLE_ORDER_COLUMN,
    ),
    ("009_topics", schema::MIGRATION_009_TOPICS),
    (
        "010_rename_source_type_local_to_file",
        schema::MIGRATION_010_RENAME_SOURCE_TYPE_LOCAL_TO_FILE,
    ),
    ("011_eval_per_query", schema::MIGRATION_011_EVAL_PER_QUERY),
    (
        "012_topics_tenant_unique",
        schema::MIGRATION_012_TOPICS_TENANT_UNIQUE,
    ),
    (
        "013_result_table_kind",
        schema::MIGRATION_013_RESULT_TABLE_KIND,
    ),
    ("014_bm25_channel", schema::MIGRATION_014_BM25_CHANNEL),
    (
        "015_fine_tune_job_queue",
        schema::MIGRATION_015_FINE_TUNE_JOB_QUEUE,
    ),
    (
        "016_rename_training_jobs",
        schema::MIGRATION_016_RENAME_TRAINING_JOBS,
    ),
    (
        "017_model_artifact_path_column",
        schema::MIGRATION_017_MODEL_ARTIFACT_PATH_COLUMN,
    ),
    (
        "018_eval_runs_model_id_nullable",
        schema::MIGRATION_018_EVAL_RUNS_MODEL_ID_NULLABLE,
    ),
    (
        "019_normalize_model_status",
        schema::MIGRATION_019_NORMALIZE_MODEL_STATUS,
    ),
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
