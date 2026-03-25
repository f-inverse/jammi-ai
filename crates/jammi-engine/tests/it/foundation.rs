use jammi_engine::catalog::status::ResultTableStatus;
use jammi_engine::catalog::Catalog;
use tempfile::tempdir;

#[test]
fn catalog_opens_and_creates_tables() {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).unwrap();

    // Exercise public APIs that transitively verify each table exists
    assert!(catalog.list_sources().unwrap().is_empty());
    assert!(catalog.list_models().unwrap().is_empty());
    assert!(catalog
        .list_result_tables_by_status(ResultTableStatus::Ready)
        .unwrap()
        .is_empty());
    assert!(!catalog.evidence_channel_names().unwrap().is_empty());
}

#[test]
fn catalog_migrations_idempotent() {
    let dir = tempdir().unwrap();
    let _catalog1 = Catalog::open(dir.path()).unwrap();
    drop(_catalog1);
    let catalog2 = Catalog::open(dir.path());
    assert!(catalog2.is_ok(), "Opening catalog twice should not fail");
}
