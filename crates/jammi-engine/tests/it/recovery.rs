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
    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());

    // Register a model (FK constraint for both result_tables and fine_tune_jobs)
    catalog
        .register_model(RegisterModelParams {
            model_id: "test-model",
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: "embedding",
            ..Default::default()
        })
        .unwrap();

    // Create a result table (status = Building via default, no Parquet file on disk)
    let result_store = ResultStore::new(dir.path(), Arc::clone(&catalog)).unwrap();
    let table_info = result_store
        .create_table("src1", "embedding", "test-model", None, None, None)
        .unwrap();

    // Create a fine-tune job and transition to Running (simulates crashed training)
    catalog
        .create_fine_tune_job(
            "ft-crash-1",
            "test-model::1",
            "pairs.csv",
            "contrastive",
            "{}",
        )
        .unwrap();
    catalog
        .update_fine_tune_status("ft-crash-1", FineTuneJobStatus::Running, None)
        .unwrap();

    // Verify stale state
    let building = catalog
        .list_result_tables_by_status(ResultTableStatus::Building)
        .unwrap();
    assert_eq!(
        building.len(),
        1,
        "Should have one Building table before recovery"
    );

    let job = catalog.get_fine_tune_job("ft-crash-1").unwrap();
    assert_eq!(job.status, FineTuneJobStatus::Running.to_string());

    // Run recovery
    result_store.recover().await.unwrap();
    let cleaned = catalog.cleanup_stale_fine_tune_jobs().unwrap();
    assert_eq!(
        cleaned, 1,
        "Should have cleaned up one running fine-tune job"
    );

    // Verify: no more Building tables
    let building_after = catalog
        .list_result_tables_by_status(ResultTableStatus::Building)
        .unwrap();
    assert!(
        building_after.is_empty(),
        "No Building tables should remain"
    );

    // Verify: crashed table → Failed (no Parquet file existed)
    let table = catalog
        .get_result_table(&table_info.table_name)
        .unwrap()
        .expect("table should still exist");
    assert_eq!(
        table.status,
        ResultTableStatus::Failed.to_string(),
        "Crashed table should be marked Failed"
    );

    // Verify: crashed job → Failed
    let job_after = catalog.get_fine_tune_job("ft-crash-1").unwrap();
    assert_eq!(
        job_after.status,
        FineTuneJobStatus::Failed.to_string(),
        "Crashed fine-tune job should be Failed"
    );
}
