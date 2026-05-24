//! `JammiSession::read_vectors` — typed read of a `FixedSizeList<Float32>`
//! column from an embedding result table. Hermetic: a tempdir-backed
//! parquet file is written through the engine's `ObjectParquetWriter`,
//! registered as a result table in the catalog, then read back through the
//! session API.

use std::sync::Arc;

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use jammi_engine::catalog::result_repo::CreateResultTableParams;
use jammi_engine::catalog::status::ResultTableStatus;
use jammi_engine::error::JammiError;
use jammi_engine::storage::{JammiObjectStore, ObjectParquetWriter, StorageRegistry, StorageUrl};
use jammi_engine::store::schema::embedding_table_schema;
use jammi_test_utils::make_test_session;
use tempfile::tempdir;

/// Build the four input rows used by both happy and negative paths.
fn input_vectors() -> Vec<Vec<f32>> {
    vec![
        vec![0.1, 0.2, 0.3, 0.4],
        vec![-1.0, 0.0, 1.0, 2.5],
        vec![0.5, 2.71, 1.41, 0.0],
        vec![f32::MIN_POSITIVE, 1.0, -1.0, f32::EPSILON],
    ]
}

/// Build a `FixedSizeList<Float32>` of the given inner length from a flat
/// `Vec<Vec<f32>>`.
fn fixed_size_list_from(rows: &[Vec<f32>], dim: i32) -> FixedSizeListArray {
    let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    let values = Arc::new(Float32Array::from(flat));
    let field = Arc::new(Field::new("item", DataType::Float32, false));
    FixedSizeListArray::try_new(field, dim, values, None).unwrap()
}

#[tokio::test]
async fn read_vectors_returns_input_rows_byte_for_byte() {
    let dir = tempdir().unwrap();
    let session = make_test_session(
        jammi_engine::catalog::backend::BackendKind::Sqlite,
        dir.path(),
    )
    .await
    .expect("sqlite session");

    let dim = 4_i32;
    let schema = embedding_table_schema(dim as usize);
    let rows = input_vectors();
    let n = rows.len();
    let row_ids: Vec<String> = (0..n).map(|i| format!("r{i}")).collect();
    let source = vec!["src".to_string(); n];
    let model = vec!["model".to_string(); n];
    let vectors = fixed_size_list_from(&rows, dim);
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(row_ids)) as ArrayRef,
            Arc::new(StringArray::from(source)),
            Arc::new(StringArray::from(model)),
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

    let table_name = "embeddings_unit";
    session
        .catalog()
        .create_result_table(CreateResultTableParams {
            table_name,
            source_id: "src",
            model_id: "model",
            task: "text_embedding",
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
        .unwrap();
    let read = session.read_vectors(&record).await.unwrap();
    assert_eq!(read.len(), n);
    for (got, expected) in read.iter().zip(rows.iter()) {
        assert_eq!(got, expected);
    }
}

#[tokio::test]
async fn read_vectors_surfaces_typed_schema_error_on_wrong_column_shape() {
    // The parquet at the registered table's URL carries a `vector` column
    // typed Utf8, not FixedSizeList<Float32>. `read_vectors` must surface a
    // `JammiError::Schema` with the actual shape populated — proves callers
    // see a typed signal instead of the panic-on-downcast the OSS path used
    // to emit when consumers reached straight at the parquet.
    let dir = tempdir().unwrap();
    let session = make_test_session(
        jammi_engine::catalog::backend::BackendKind::Sqlite,
        dir.path(),
    )
    .await
    .expect("sqlite session");

    let wrong_schema = Arc::new(Schema::new(vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("vector", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&wrong_schema),
        vec![
            Arc::new(StringArray::from(vec!["r0"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["not-a-vector"])),
        ],
    )
    .unwrap();

    let parquet_path = dir.path().join("wrong_shape.parquet");
    let url = StorageUrl::parse(parquet_path.to_str().unwrap()).unwrap();
    let registry = StorageRegistry::new();
    let driver = registry.driver_for(&url, None).unwrap();
    let handle = JammiObjectStore::new(driver, url.clone());
    let mut writer = ObjectParquetWriter::open(&handle, Arc::clone(&wrong_schema))
        .await
        .unwrap();
    writer.write_batch(&batch).await.unwrap();
    writer.close().await.unwrap();

    let table_name = "wrong_shape";
    session
        .catalog()
        .create_result_table(CreateResultTableParams {
            table_name,
            source_id: "src",
            model_id: "model",
            task: "text_embedding",
            parquet_path: url.as_str(),
            index_path: None,
            dimensions: Some(4),
            key_column: None,
            text_columns: None,
        })
        .await
        .unwrap();
    session
        .catalog()
        .update_result_table_status(table_name, ResultTableStatus::Ready, 1)
        .await
        .unwrap();

    let record = session
        .catalog()
        .get_result_table(table_name)
        .await
        .unwrap()
        .unwrap();
    let err = session.read_vectors(&record).await.unwrap_err();
    match err {
        JammiError::Schema {
            table,
            column,
            expected,
            actual,
        } => {
            assert_eq!(table, table_name);
            assert_eq!(column, "vector");
            assert_eq!(expected, "FixedSizeList<Float32>");
            assert!(
                !actual.is_empty() && actual != "missing",
                "actual should describe Utf8 column shape, got {actual:?}"
            );
        }
        other => panic!("expected JammiError::Schema, got {other:?}"),
    }
}
