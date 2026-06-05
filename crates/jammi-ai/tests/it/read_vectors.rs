//! `InferenceSession::read_vectors` — typed read of a `FixedSizeList<Float32>`
//! column from an embedding result table, exercised through the AI-aware
//! session wrapper. Hermetic: a tempdir-backed parquet file is written
//! through the engine's `ObjectParquetWriter`, registered as a result table
//! via the catalog the `InferenceSession` shares, then read back through
//! the forwarder.

use std::sync::Arc;

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field};
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::result_repo::CreateResultTableParams;
use jammi_db::catalog::status::ResultTableStatus;
use jammi_db::model_task::ModelTask;
use jammi_db::storage::{JammiObjectStore, ObjectParquetWriter, StorageRegistry, StorageUrl};
use jammi_db::store::schema::embedding_table_schema;
use tempfile::tempdir;

use crate::common;

#[tokio::test]
async fn inference_session_read_vectors_forwards_to_jammi_session() {
    let dir = tempdir().unwrap();
    let config = common::test_config(dir.path());
    let session = InferenceSession::new(config).await.unwrap();

    let dim = 4_i32;
    let schema = embedding_table_schema(dim as usize);
    let rows: Vec<Vec<f32>> = vec![
        vec![0.1, 0.2, 0.3, 0.4],
        vec![-1.0, 0.0, 1.0, 2.5],
        vec![0.5, 2.71, 1.41, 0.0],
        vec![f32::MIN_POSITIVE, 1.0, -1.0, f32::EPSILON],
    ];
    let n = rows.len();
    let row_ids: Vec<String> = (0..n).map(|i| format!("r{i}")).collect();
    let source_ids = vec!["src".to_string(); n];
    let model_ids = vec!["model".to_string(); n];

    let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    let values = Arc::new(Float32Array::from(flat));
    let item_field = Arc::new(Field::new("item", DataType::Float32, false));
    let vectors = FixedSizeListArray::try_new(item_field, dim, values, None).unwrap();

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(row_ids)) as ArrayRef,
            Arc::new(StringArray::from(source_ids)),
            Arc::new(StringArray::from(model_ids)),
            Arc::new(vectors),
        ],
    )
    .unwrap();

    let parquet_path = dir.path().join("embeddings.parquet");
    let url = StorageUrl::parse(parquet_path.to_str().unwrap()).unwrap();
    let registry = StorageRegistry::new();
    let driver = registry.driver_for(&url, None).unwrap();
    let handle = JammiObjectStore::new(driver, url.clone());
    let mut writer = ObjectParquetWriter::open(&handle, Arc::clone(&schema))
        .await
        .unwrap();
    writer.write_batch(&batch).await.unwrap();
    writer.close().await.unwrap();

    let table_name = "embeddings_inference_session";
    session
        .catalog()
        .create_result_table(CreateResultTableParams {
            table_name,
            source_id: "src",
            model_id: "model",
            task: ModelTask::TextEmbedding,
            kind: jammi_db::catalog::result_repo::ResultTableKind::Model,
            derived_from: None,
            parquet_path: url.as_str(),
            index_path: None,
            dimensions: Some(dim),
            key_column: None,
            text_columns: None,
        })
        .await
        .unwrap();
    session
        .catalog()
        .update_result_table_status(table_name, ResultTableStatus::Ready, n)
        .await
        .unwrap();

    let record = session
        .catalog()
        .get_result_table(table_name)
        .await
        .unwrap()
        .expect("result table just created");

    let read = session.read_vectors(&record).await.unwrap();
    assert_eq!(read.len(), n);
    for (got, expected) in read.iter().zip(rows.iter()) {
        assert_eq!(got, expected);
    }
}
