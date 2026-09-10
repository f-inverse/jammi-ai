//! Migration runner. Applies the SQL constants from [`super::schema`] in
//! order, tracking which have been applied in an `applied_migrations` ledger.
//! Backend-agnostic: works through [`CatalogBackend`].

use super::backend::{BackendError, BackendKind, CatalogBackend, SqlValue, TxOptions};
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
    (
        "020_channel_tenant_scope",
        schema::MIGRATION_020_CHANNEL_TENANT_SCOPE,
    ),
    (
        "021_materialization_contract",
        schema::MIGRATION_021_MATERIALIZATION_CONTRACT,
    ),
    (
        "022_definition_hash_index",
        schema::MIGRATION_022_DEFINITION_HASH_INDEX,
    ),
    (
        "023_storage_precision",
        schema::MIGRATION_023_STORAGE_PRECISION,
    ),
    ("024_claim_policy", schema::MIGRATION_024_CLAIM_POLICY),
    ("025_index_segments", schema::MIGRATION_025_INDEX_SEGMENTS),
    (
        "026_acceleration_report",
        schema::MIGRATION_026_ACCELERATION_REPORT,
    ),
    (
        "027_result_table_lease",
        schema::MIGRATION_027_RESULT_TABLE_LEASE,
    ),
    (
        "028_topics_next_offset",
        schema::MIGRATION_028_TOPICS_NEXT_OFFSET,
    ),
    (
        "029_jobs_instances_workers",
        schema::MIGRATION_029_JOBS_INSTANCES_WORKERS,
    ),
];

const APPLIED_MIGRATIONS_DDL: &str = r#"
CREATE TABLE IF NOT EXISTS applied_migrations (
    name        TEXT PRIMARY KEY,
    applied_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"#;

/// Key of the transaction-scoped Postgres advisory lock [`run`] takes before
/// it touches the `applied_migrations` ledger (`0x6a61_6d6d_695f_6d69` is the
/// ASCII bytes of `jammi_mi`).
///
/// Why it exists: the runner reads the ledger and then executes
/// **non-idempotent** DDL (`CREATE TABLE result_tables`, no `IF NOT EXISTS`)
/// inside one `READ COMMITTED` transaction. Postgres gives two transactions on
/// one database no mutual exclusion across that read-then-DDL window, so two
/// fresh replicas booting together both saw an empty ledger and the loser
/// failed with SQLSTATE `42P07` (`relation "..." already exists`) or `23505`
/// on the ledger primary key -- escape-ledger row
/// `esc-093-postgres-migrations-race-without-cross-process-lock` (issue #479).
/// `SELECT pg_advisory_xact_lock($1)` with this key is the runner's first
/// statement on Postgres, so the ledger read happens after the lock by
/// construction and the loser re-reads a complete ledger once the winner
/// commits.
///
/// Advisory locks are scoped to one database, so the key needs no database-name
/// hashing; the `_xact_` flavour is released on commit **or** rollback, which
/// keeps it correct under PgBouncer transaction pooling (Postgres docs
/// section 13.3.5 "Advisory Locks" and section 9.28.10). The DDL stays
/// non-idempotent on purpose: the lock is the mechanism and
/// `tests/it/migrations.rs::concurrent_migrate_on_fresh_postgres_is_safe`
/// proves it; an `IF NOT EXISTS` sprinkle would hide a second racer's partial
/// schema instead of serialising it.
///
/// SQLite needs nothing here: its backend opens every write transaction
/// `BEGIN IMMEDIATE` under a 5 s `busy_timeout`, so the whole runner is already
/// one serialised writer.
pub(crate) const JAMMI_MIGRATION_LOCK_KEY: i64 = 0x6a61_6d6d_695f_6d69;

/// Apply all pending migrations. Idempotent, and safe to run concurrently
/// from several processes against one catalog.
///
/// One transaction does everything: on Postgres it first takes the
/// transaction-scoped advisory lock keyed by [`JAMMI_MIGRATION_LOCK_KEY`]
/// (see there for the race it closes), then creates the `applied_migrations`
/// ledger if absent, reads it, and applies every entry of `MIGRATIONS` the
/// ledger does not name, recording each as it goes. A concurrent caller on
/// Postgres blocks on the advisory lock until the first commits or rolls
/// back, then reads the ledger the winner left and applies nothing. On SQLite
/// the backend's `BEGIN IMMEDIATE` write transaction is the same serialiser.
///
/// Under `feature = "test-hooks"` a rendezvous point sits between the ledger
/// read and the first DDL statement
/// (`store::mutable::test_hook::maybe_signal_migration_ledger_read`) so a test
/// can pin the interleaving the lock must survive.
pub(crate) async fn run<B: CatalogBackend + ?Sized>(backend: &B) -> Result<(), BackendError> {
    let kind = backend.backend_kind();
    backend
        .transaction(TxOptions::default(), move |tx| {
            Box::pin(async move {
                if kind == BackendKind::Postgres {
                    // Must be the first statement: everything below reads or
                    // writes state a concurrent runner would race on.
                    tx.execute(
                        "SELECT pg_advisory_xact_lock($1)",
                        &[SqlValue::Int(JAMMI_MIGRATION_LOCK_KEY)],
                    )
                    .await?;
                }
                tx.execute(APPLIED_MIGRATIONS_DDL, &[]).await?;
                let applied: Vec<String> = tx
                    .query("SELECT name FROM applied_migrations", &[], |row| {
                        row.get::<String>("name")
                    })
                    .await?;
                let applied_set: std::collections::HashSet<&str> =
                    applied.iter().map(String::as_str).collect();

                // The ledger has been read and no DDL has run: the window a
                // second runner must not be allowed to share.
                #[cfg(feature = "test-hooks")]
                crate::store::mutable::test_hook::maybe_signal_migration_ledger_read(kind).await;

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
