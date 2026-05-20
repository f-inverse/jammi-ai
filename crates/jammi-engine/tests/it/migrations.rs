use jammi_engine::catalog::Catalog;
use rusqlite::Connection;
use tempfile::tempdir;

#[test]
fn migration_005_adds_tenant_id_to_every_table() {
    let dir = tempdir().unwrap();
    let _catalog = Catalog::open(dir.path()).unwrap();

    let conn = Connection::open(dir.path().join("catalog.db")).unwrap();
    for table in [
        "sources",
        "models",
        "fine_tune_jobs",
        "eval_runs",
        "result_tables",
        "evidence_channels",
    ] {
        let columns: Vec<String> = conn
            .prepare(&format!("SELECT name FROM pragma_table_info('{table}')"))
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert!(
            columns.iter().any(|c| c == "tenant_id"),
            "table '{table}' must have a tenant_id column after migration 005; \
             got columns: {columns:?}"
        );
    }
}

#[test]
fn migration_005_creates_tenant_index_per_table() {
    let dir = tempdir().unwrap();
    let _catalog = Catalog::open(dir.path()).unwrap();

    let conn = Connection::open(dir.path().join("catalog.db")).unwrap();
    for (table, idx) in [
        ("sources", "idx_sources_tenant"),
        ("models", "idx_models_tenant"),
        ("fine_tune_jobs", "idx_fine_tune_jobs_tenant"),
        ("eval_runs", "idx_eval_runs_tenant"),
        ("result_tables", "idx_result_tables_tenant"),
        ("evidence_channels", "idx_evidence_channels_tenant"),
    ] {
        let exists: bool = conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?1 AND tbl_name=?2",
                rusqlite::params![idx, table],
                |row| row.get::<_, i64>(0).map(|_| true),
            )
            .unwrap_or(false);
        assert!(
            exists,
            "index '{idx}' on '{table}' must exist after migration 005"
        );
    }
}

#[test]
fn migration_005_back_fills_existing_rows_to_null() {
    let dir = tempdir().unwrap();
    let _catalog = Catalog::open(dir.path()).unwrap();
    let conn = Connection::open(dir.path().join("catalog.db")).unwrap();

    // evidence_channels has two seeded rows from MIGRATION_001. Both must
    // have tenant_id = NULL after MIGRATION_005 runs against a greenfield db.
    let nulls: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM evidence_channels WHERE tenant_id IS NULL",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        nulls, 2,
        "both seeded evidence_channels rows must have tenant_id NULL"
    );
}

#[test]
fn migrations_are_idempotent_across_reopens() {
    let dir = tempdir().unwrap();
    let _c1 = Catalog::open(dir.path()).unwrap();
    drop(_c1);
    let _c2 = Catalog::open(dir.path()).unwrap();
    // Re-open is the contract: no panic, no error.
}
