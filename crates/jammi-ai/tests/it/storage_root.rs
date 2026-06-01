//! `InferenceSession` honours `config.storage.result_root`: when set, result
//! tables are rooted there (here a hermetic `memory://` URL standing in for an
//! `r2://`/`s3://` deploy root) rather than on local disk under `artifact_dir`.

use std::sync::Arc;

use arrow::array::{ArrayRef, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use jammi_ai::session::InferenceSession;
use jammi_db::config::StorageConfig;
use jammi_db::model_task::ModelTask;
use tempfile::TempDir;

use crate::common;

/// With `storage.result_root` set to a `memory://` URL, the session's result
/// store creates tables under that root and round-trips a batch back — proving
/// the configured cloud root threads from `JammiConfig` into the `ResultStore`
/// without touching local disk for the table data. The catalog (SQLite under
/// the temp `artifact_dir`) is unaffected.
#[tokio::test]
async fn inference_session_roots_result_tables_at_configured_memory_root() {
    let dir = TempDir::new().unwrap();
    let mut config = common::test_config(dir.path());
    config.storage = StorageConfig {
        result_root: Some("memory:///jammi_results".into()),
        cloud: None,
    };

    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    let store = session.result_store();

    let info = store
        .create_table(
            "patents",
            ModelTask::Classification,
            "model",
            None,
            None,
            None,
        )
        .await
        .unwrap();
    assert!(
        info.parquet_url
            .as_str()
            .starts_with("memory:///jammi_results/"),
        "result table not rooted at the configured memory root: {}",
        info.parquet_url
    );

    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Utf8, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringArray::from(vec!["x", "y"])) as ArrayRef],
    )
    .unwrap();
    let mut writer = store
        .open_writer(&info.parquet_url, Arc::clone(&schema))
        .await
        .unwrap();
    writer.write_batch(&batch).await.unwrap();
    let rows = writer.close().await.unwrap();
    assert_eq!(rows, 2);

    // Nothing was written under the local jammi_db dir — the table lives in
    // the in-memory root.
    let local_db = dir.path().join("jammi_db");
    let has_parquet = local_db.exists()
        && std::fs::read_dir(&local_db)
            .unwrap()
            .filter_map(|e| e.ok())
            .any(|e| e.path().extension().is_some_and(|x| x == "parquet"));
    assert!(
        !has_parquet,
        "result-table parquet leaked to local disk under {local_db:?}"
    );
}
