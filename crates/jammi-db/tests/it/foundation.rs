use jammi_db::catalog::status::ResultTableStatus;
use jammi_db::catalog::Catalog;
use tempfile::tempdir;

#[tokio::test]
async fn catalog_opens_and_creates_tables() {
    let dir = tempdir().unwrap();
    let catalog = Catalog::open(dir.path()).await.unwrap();

    assert!(catalog.list_sources().await.unwrap().is_empty());
    assert!(catalog.list_models().await.unwrap().is_empty());
    assert!(catalog
        .list_result_tables_by_status(ResultTableStatus::Ready)
        .await
        .unwrap()
        .is_empty());
    assert!(!catalog.channels().list().await.unwrap().is_empty());
}

#[tokio::test]
async fn catalog_migrations_idempotent() {
    let dir = tempdir().unwrap();
    let catalog1 = Catalog::open(dir.path()).await.unwrap();
    drop(catalog1);
    let catalog2 = Catalog::open(dir.path()).await;
    assert!(catalog2.is_ok(), "Opening catalog twice should not fail");
}
