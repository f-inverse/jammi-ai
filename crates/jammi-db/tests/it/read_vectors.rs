//! `ResultStore::read_vectors` — typed read of a `FixedSizeList<Float32>`
//! column from a pinned embedding result table. Hermetic: a tempdir-backed
//! parquet file is written through the engine's `ObjectParquetWriter`,
//! registered as a result table in the catalog, then pinned and read back
//! through the store.

use std::sync::Arc;

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use jammi_db::catalog::backend::BackendKind;
use jammi_db::catalog::result_repo::CreateResultTableParams;
use jammi_db::catalog::status::ResultTableStatus;
use jammi_db::error::JammiError;
use jammi_db::storage::{ObjectParquetWriter, StorageRegistry, StorageUrl};
use jammi_db::store::schema::embedding_table_schema;
use jammi_db::ModelTask;
use jammi_test_utils::{make_test_session, unique_suffix};
use tempfile::tempdir;
use test_case::test_case;

use crate::common;

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

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn read_vectors_returns_input_rows_byte_for_byte(backend: BackendKind) {
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;

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
            jammi_db::store::content_hash::null_hash_column(n),
        ],
    )
    .unwrap();

    let parquet_path = dir.path().join("embeddings.parquet");
    let url = StorageUrl::parse(parquet_path.to_str().unwrap()).unwrap();
    let registry = StorageRegistry::new();
    let handle = registry.handle_for(&url, None).unwrap();
    let mut writer = ObjectParquetWriter::open(&handle, Arc::clone(&schema))
        .await
        .unwrap();
    writer.write_batch(&batch).await.unwrap();
    writer.close().await.unwrap();

    let table_name = format!("embeddings_unit_{}", unique_suffix());
    let table_name = table_name.as_str();
    session
        .catalog()
        .create_result_table(CreateResultTableParams {
            writer_id: None,
            lease: None,
            table_name,
            source_id: "src",
            model_id: "model",
            task: ModelTask::TextEmbedding,
            kind: jammi_db::catalog::result_repo::ResultTableKind::Model,
            derived_from: None,
            parquet_path: url.as_str(),
            dimensions: Some(dim),
            key_column: None,
            text_columns: None,
            storage_precision: jammi_db::config::StoragePrecision::F32,
            oversample: 4,
            created_at: jammi_db::catalog::lease::canonical_stamp_now(),
            job_attempt: None,
            replaces: None,
        })
        .await
        .unwrap();
    session
        .catalog()
        .update_result_table_status(table_name, ResultTableStatus::Ready, n)
        .await
        .unwrap();

    let store = common::store_over(dir.path(), session.catalog());
    let pin = common::pin(&store, table_name).await;
    let read = store.read_vectors(session.context(), &pin).await.unwrap();
    assert_eq!(read.len(), n);
    for (got, expected) in read.iter().zip(rows.iter()) {
        assert_eq!(got, expected);
    }
}

#[test_case(BackendKind::Sqlite ; "sqlite")]
#[cfg_attr(feature = "live-postgres-tests", test_case(BackendKind::Postgres ; "postgres"))]
#[tokio::test]
async fn read_vectors_surfaces_typed_engine_fault_on_wrong_column_shape(backend: BackendKind) {
    // The parquet at the registered table's URL carries a `vector` column
    // typed Utf8, not FixedSizeList<Float32>. This is the TABLE's own stored
    // artifact — the corruption is never the caller's fault, so
    // `read_vectors` must surface `JammiError::IncompatibleFormat` (never
    // the caller class, `JammiError::Schema`, for this engine-owned defect)
    // with the actual shape populated — callers see a typed,
    // correctly-attributed signal, never a panic on downcast.
    let dir = tempdir().unwrap();
    let session = make_test_session(backend, dir.path()).await;

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
    let handle = registry.handle_for(&url, None).unwrap();
    let mut writer = ObjectParquetWriter::open(&handle, Arc::clone(&wrong_schema))
        .await
        .unwrap();
    writer.write_batch(&batch).await.unwrap();
    writer.close().await.unwrap();

    let table_name = format!("wrong_shape_{}", unique_suffix());
    let table_name = table_name.as_str();
    session
        .catalog()
        .create_result_table(CreateResultTableParams {
            writer_id: None,
            lease: None,
            table_name,
            source_id: "src",
            model_id: "model",
            task: ModelTask::TextEmbedding,
            kind: jammi_db::catalog::result_repo::ResultTableKind::Model,
            derived_from: None,
            parquet_path: url.as_str(),
            dimensions: Some(4),
            key_column: None,
            text_columns: None,
            storage_precision: jammi_db::config::StoragePrecision::F32,
            oversample: 4,
            created_at: jammi_db::catalog::lease::canonical_stamp_now(),
            job_attempt: None,
            replaces: None,
        })
        .await
        .unwrap();
    session
        .catalog()
        .update_result_table_status(table_name, ResultTableStatus::Ready, 1)
        .await
        .unwrap();

    let store = common::store_over(dir.path(), session.catalog());
    let pin = common::pin(&store, table_name).await;
    let err = store
        .read_vectors(session.context(), &pin)
        .await
        .unwrap_err();
    match err {
        JammiError::IncompatibleFormat {
            artifact,
            found,
            supported,
        } => {
            assert!(artifact.contains(table_name) && artifact.contains("vector"));
            assert_eq!(supported, "FixedSizeList<Float32>");
            assert!(
                !found.is_empty() && found != "missing",
                "found should describe Utf8 column shape, got {found:?}"
            );
        }
        other => panic!("expected JammiError::IncompatibleFormat, got {other:?}"),
    }
}
