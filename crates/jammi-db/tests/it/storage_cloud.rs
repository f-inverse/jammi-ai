//! Live cloud-backend round-trip tests.
//!
//! Each test is compiled only under its per-cloud Cargo feature
//! (`live-s3-tests`, `live-gcs-tests`, `live-azure-tests`); on the hermetic
//! `cargo test` lane none is on, so this module is empty and makes no network
//! call.
//!
//! Each test reads its bucket from the environment and fails naming the
//! variable when it is unset:
//!   - `JAMMI_TEST_S3_BUCKET`     — `s3://bucket/prefix`
//!   - `JAMMI_TEST_GCS_BUCKET`    — `gs://bucket/prefix`
//!   - `JAMMI_TEST_AZURE_BUCKET`  — `azure://container/prefix`
//!
//! Plus the usual SDK credentials in env (AWS_*, GOOGLE_APPLICATION_*,
//! AZURE_*).

#![cfg(any(
    feature = "live-s3-tests",
    feature = "live-gcs-tests",
    feature = "live-azure-tests"
))]

use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use jammi_db::storage::{
    reader::{count_parquet_rows, is_valid_parquet},
    ObjectParquetWriter, StorageRegistry, StorageUrl,
};

fn three_col_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("score", DataType::Float32, false),
    ]));
    RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef,
            Arc::new(Float32Array::from(vec![0.1, 0.2, 0.3])) as ArrayRef,
        ],
    )
    .unwrap()
}

async fn round_trip_under(url: StorageUrl) {
    let registry = StorageRegistry::new();
    let handle = registry.handle_for(&url, None).unwrap();
    let batch = three_col_batch();
    let schema = batch.schema();
    let mut w = ObjectParquetWriter::open(&handle, schema).await.unwrap();
    w.write_batch(&batch).await.unwrap();
    let rows = w.close().await.unwrap();
    assert_eq!(rows, 3);
    assert!(is_valid_parquet(&handle).await.unwrap());
    assert_eq!(count_parquet_rows(&handle).await.unwrap(), 3);

    // Best-effort cleanup so re-runs don't accumulate orphan objects.
    let path = handle.data_path().unwrap();
    handle.vanish_for_test(&path).await.unwrap();
}

#[cfg(feature = "live-s3-tests")]
#[tokio::test]
async fn s3_parquet_round_trip() {
    let base = jammi_test_resources::env("JAMMI_TEST_S3_BUCKET");
    let key = format!(
        "{}/jammi-storage-test-{}.parquet",
        base.trim_end_matches('/'),
        uuid::Uuid::new_v4().simple()
    );
    let url = StorageUrl::parse(&key).expect("S3 URL parses");
    round_trip_under(url).await;
}

#[cfg(feature = "live-gcs-tests")]
#[tokio::test]
async fn gcs_parquet_round_trip() {
    let base = jammi_test_resources::env("JAMMI_TEST_GCS_BUCKET");
    let key = format!(
        "{}/jammi-storage-test-{}.parquet",
        base.trim_end_matches('/'),
        uuid::Uuid::new_v4().simple()
    );
    let url = StorageUrl::parse(&key).expect("GCS URL parses");
    round_trip_under(url).await;
}

#[cfg(feature = "live-azure-tests")]
#[tokio::test]
async fn azure_parquet_round_trip() {
    let base = jammi_test_resources::env("JAMMI_TEST_AZURE_BUCKET");
    let key = format!(
        "{}/jammi-storage-test-{}.parquet",
        base.trim_end_matches('/'),
        uuid::Uuid::new_v4().simple()
    );
    let url = StorageUrl::parse(&key).expect("Azure URL parses");
    round_trip_under(url).await;
}
