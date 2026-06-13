use std::sync::Arc;

use jammi_db::catalog::model_repo::RegisterModelParams;
use jammi_db::catalog::status::ResultTableStatus;
use jammi_db::catalog::Catalog;
use jammi_db::config::AnnIndexConfig;
use jammi_db::model_task::ModelTask;
use jammi_db::store::ResultStore;
use tempfile::tempdir;

/// Crash recovery: a stale `Building` result table is reconciled to `Failed` at
/// startup. (Orphaned training jobs are recovered by the worker's lease-based
/// `reclaim_expired_training_jobs`, not a startup sweep — that path is exercised
/// in the engine crate's worker tests.)
#[tokio::test]
async fn crash_recovery_cleans_up_stale_result_tables() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).await.unwrap());

    catalog
        .register_model(RegisterModelParams {
            model_id: "test-model",
            version: 1,
            model_type: "embedding",
            backend: "candle",
            task: ModelTask::TextEmbedding,
            base_model_id: None,
            artifact_path: None,
            config_json: None,
        })
        .await
        .unwrap();

    let result_store =
        ResultStore::new(dir.path(), Arc::clone(&catalog), AnnIndexConfig::default()).unwrap();
    let table_info = result_store
        .create_table(
            "src1",
            ModelTask::TextEmbedding,
            jammi_db::catalog::result_repo::ResultTableKind::Model,
            None,
            "test-model",
            None,
            None,
            None,
        )
        .await
        .unwrap();

    let building = catalog
        .list_result_tables_by_status(ResultTableStatus::Building)
        .await
        .unwrap();
    assert_eq!(building.len(), 1);

    result_store.recover().await.unwrap();

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
}
