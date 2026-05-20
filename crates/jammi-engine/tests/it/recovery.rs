use std::sync::Arc;

use jammi_engine::catalog::model_repo::RegisterModelParams;
use jammi_engine::catalog::status::{FineTuneJobStatus, ResultTableStatus};
use jammi_engine::catalog::Catalog;
use jammi_engine::store::ResultStore;
use tempfile::tempdir;

/// Crash recovery: stale Building result tables → Failed, stale Running
/// fine-tune jobs → Failed.
#[tokio::test]
async fn crash_recovery_cleans_up_stale_result_tables_and_fine_tune_jobs() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());

    catalog
        .register_model(RegisterModelParams {
            model_id: "test-model",
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: "text_embedding",
            ..Default::default()
        })
        .await
        .unwrap();

    let result_store = ResultStore::new(dir.path(), Arc::clone(&catalog)).unwrap();
    let table_info = result_store
        .create_table("src1", "text_embedding", "test-model", None, None, None)
        .await
        .unwrap();

    catalog
        .create_fine_tune_job(
            "ft-crash-1",
            "test-model::1",
            "pairs.csv",
            "contrastive",
            "{}",
        )
        .await
        .unwrap();
    catalog
        .update_fine_tune_status("ft-crash-1", FineTuneJobStatus::Running, None)
        .await
        .unwrap();

    let building = catalog
        .list_result_tables_by_status(ResultTableStatus::Building)
        .await
        .unwrap();
    assert_eq!(building.len(), 1);

    let job = catalog.get_fine_tune_job("ft-crash-1").await.unwrap();
    assert_eq!(job.status, FineTuneJobStatus::Running.to_string());

    result_store.recover().await.unwrap();
    let cleaned = catalog.cleanup_stale_fine_tune_jobs().await.unwrap();
    assert_eq!(cleaned, 1);

    let building_after = catalog
        .list_result_tables_by_status(ResultTableStatus::Building)
        .await
        .unwrap();
    assert!(building_after.is_empty());

    let table = catalog
        .get_result_table(&table_info.table_name)
        .await
        .unwrap()
        .expect("table should still exist");
    assert_eq!(table.status, ResultTableStatus::Failed.to_string());

    let job_after = catalog.get_fine_tune_job("ft-crash-1").await.unwrap();
    assert_eq!(job_after.status, FineTuneJobStatus::Failed.to_string());
}
