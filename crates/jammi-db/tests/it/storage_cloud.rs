//! Live cloud-backend round-trip tests.
//!
//! Each test is gated behind a per-cloud Cargo feature (`live-s3-tests`,
//! `live-gcs-tests`, `live-azure-tests`) and a `JAMMI_TEST_*_BUCKET`
//! environment variable. On the hermetic `cargo test` lane these
//! features are off so the tests never compile, never run, and never
//! make a network call.
//!
//! Required env vars (set by CI's `test-live-cloud` job):
//!   - `JAMMI_TEST_S3_BUCKET`     — `s3://bucket/prefix`
//!   - `JAMMI_TEST_GCS_BUCKET`    — `gs://bucket/prefix`
//!   - `JAMMI_TEST_AZURE_BUCKET`  — `azure://container/prefix`
//!
//! Plus the usual SDK credentials in env (AWS_*, GOOGLE_APPLICATION_*,
//! AZURE_*). Tests that find the variable unset early-return with a
//! `tracing::warn` so CI logs flag the skip — `#[ignore]` is forbidden
//! by CLAUDE.md.

#[cfg(any(
    feature = "live-s3-tests",
    feature = "live-gcs-tests",
    feature = "live-azure-tests"
))]
use std::sync::Arc;

#[cfg(any(
    feature = "live-s3-tests",
    feature = "live-gcs-tests",
    feature = "live-azure-tests"
))]
use arrow::array::{ArrayRef, Float32Array, StringArray};
#[cfg(any(
    feature = "live-s3-tests",
    feature = "live-gcs-tests",
    feature = "live-azure-tests"
))]
use arrow::datatypes::{DataType, Field, Schema};
#[cfg(any(
    feature = "live-s3-tests",
    feature = "live-gcs-tests",
    feature = "live-azure-tests"
))]
use arrow::record_batch::RecordBatch;
#[cfg(any(
    feature = "live-s3-tests",
    feature = "live-gcs-tests",
    feature = "live-azure-tests"
))]
use jammi_db::storage::{
    reader::{count_parquet_rows, is_valid_parquet},
    JammiObjectStore, ObjectParquetWriter, StorageRegistry, StorageUrl,
};

#[cfg(any(
    feature = "live-s3-tests",
    feature = "live-gcs-tests",
    feature = "live-azure-tests"
))]
fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.is_empty())
}

#[cfg(any(
    feature = "live-s3-tests",
    feature = "live-gcs-tests",
    feature = "live-azure-tests"
))]
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

#[cfg(any(
    feature = "live-s3-tests",
    feature = "live-gcs-tests",
    feature = "live-azure-tests"
))]
async fn round_trip_under(url: StorageUrl) {
    let registry = StorageRegistry::new();
    let driver = registry.driver_for(&url, None).unwrap();
    let handle = JammiObjectStore::new(driver, url.clone());
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
    handle.delete_if_exists(&path).await.unwrap();
}

#[cfg(feature = "live-s3-tests")]
#[tokio::test]
async fn s3_parquet_round_trip() {
    let Some(base) = env("JAMMI_TEST_S3_BUCKET") else {
        tracing::warn!("JAMMI_TEST_S3_BUCKET unset; skipping live S3 test");
        return;
    };
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
    let Some(base) = env("JAMMI_TEST_GCS_BUCKET") else {
        tracing::warn!("JAMMI_TEST_GCS_BUCKET unset; skipping live GCS test");
        return;
    };
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
    let Some(base) = env("JAMMI_TEST_AZURE_BUCKET") else {
        tracing::warn!("JAMMI_TEST_AZURE_BUCKET unset; skipping live Azure test");
        return;
    };
    let key = format!(
        "{}/jammi-storage-test-{}.parquet",
        base.trim_end_matches('/'),
        uuid::Uuid::new_v4().simple()
    );
    let url = StorageUrl::parse(&key).expect("Azure URL parses");
    round_trip_under(url).await;
}

#[cfg(not(any(
    feature = "live-s3-tests",
    feature = "live-gcs-tests",
    feature = "live-azure-tests"
)))]
#[test]
fn live_cloud_tests_compile_check_only() {
    // The cloud round-trip tests live behind per-cloud Cargo features.
    // This stub exists so the module still has at least one symbol when
    // none of those features is active — keeping `mod storage_cloud;` in
    // main.rs from emitting an "empty module" warning on the default
    // `cargo test` lane.
}
