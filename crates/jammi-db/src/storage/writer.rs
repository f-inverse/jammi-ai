//! Async Parquet writer that targets any `object_store::ObjectStore` —
//! local disk, S3, GCS, Azure, in-memory test driver.

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use parquet::arrow::async_writer::ParquetObjectWriter;
use parquet::arrow::AsyncArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

use super::error::StorageError;
use super::object_store_handle::JammiObjectStore;

/// The row-group row count every writer uses in a release build, and the
/// default under `feature = "test-hooks"` when the override env var is
/// unset or unparsable.
const DEFAULT_MAX_ROW_GROUP_ROWS: usize = 65_536;

/// Environment variable that overrides the writer's row-group row count.
///
/// Only read under `feature = "test-hooks"`; a release build has no such
/// knob (see [`max_row_group_row_count`]).
#[cfg(feature = "test-hooks")]
pub const ROW_GROUP_ROWS_ENV: &str = "JAMMI_TEST_ROW_GROUP_ROWS";

/// Row-group row count for a newly opened writer.
///
/// Fixed at [`DEFAULT_MAX_ROW_GROUP_ROWS`] in a release build — the on-disk
/// layout every reader depends on. Under `feature = "test-hooks"`,
/// `JAMMI_TEST_ROW_GROUP_ROWS` overrides it (falling back to the default
/// when unset, empty, non-numeric, or zero) so a fixture of a few hundred
/// rows can be split across several row groups — reproducing the
/// multi-row-group class the streaming loader's oracles need without
/// writing 65 536+ rows on every test run.
#[cfg(feature = "test-hooks")]
fn max_row_group_row_count() -> usize {
    std::env::var(ROW_GROUP_ROWS_ENV)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_MAX_ROW_GROUP_ROWS)
}

/// Row-group row count for a newly opened writer: always the fixed default
/// outside `feature = "test-hooks"`.
#[cfg(not(feature = "test-hooks"))]
fn max_row_group_row_count() -> usize {
    DEFAULT_MAX_ROW_GROUP_ROWS
}

/// Writes Arrow `RecordBatch`es to a Parquet file using the object-store
/// backend the handle was constructed with.
///
/// Produces the same on-the-wire bytes as the previous sync `ArrowWriter`
/// path — ZSTD compression, 64K row groups — so existing readers keep
/// working unchanged.
pub struct ObjectParquetWriter {
    writer: AsyncArrowWriter<ParquetObjectWriter>,
    row_count: usize,
    path: String,
}

impl ObjectParquetWriter {
    /// Open a new writer at the handle's data path.
    pub async fn open(handle: &JammiObjectStore, schema: SchemaRef) -> Result<Self, StorageError> {
        let path = handle.data_path()?;
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::default()))
            .set_max_row_group_row_count(Some(max_row_group_row_count()))
            .build();
        let inner = ParquetObjectWriter::new(handle.driver(), path.clone());
        let writer = AsyncArrowWriter::try_new(inner, schema, Some(props)).map_err(|e| {
            StorageError::layout(path.to_string(), format!("Parquet writer init: {e}"))
        })?;
        Ok(Self {
            writer,
            row_count: 0,
            path: path.to_string(),
        })
    }

    /// Append a batch to the file.
    pub async fn write_batch(&mut self, batch: &RecordBatch) -> Result<(), StorageError> {
        self.writer
            .write(batch)
            .await
            .map_err(|e| StorageError::layout(self.path.clone(), format!("Parquet write: {e}")))?;
        self.row_count += batch.num_rows();
        Ok(())
    }

    /// Flush and close the writer, returning the total row count.
    pub async fn close(self) -> Result<usize, StorageError> {
        let count = self.row_count;
        let path = self.path.clone();
        self.writer
            .close()
            .await
            .map_err(|e| StorageError::layout(path, format!("Parquet close: {e}")))?;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Float32Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::storage::{JammiObjectStore, StorageRegistry, StorageUrl};

    fn three_col_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("score", DataType::Float32, true),
        ]))
    }

    #[tokio::test]
    async fn round_trip_through_memory_driver() {
        let registry = StorageRegistry::new();
        let url = StorageUrl::memory("benchmarks/snapshot.parquet");
        let driver = registry.driver_for(&url, None).unwrap();
        let handle = JammiObjectStore::new(driver, url);

        let schema = three_col_schema();
        let mut writer = ObjectParquetWriter::open(&handle, Arc::clone(&schema))
            .await
            .unwrap();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![Some("a"), Some("b"), None])),
                Arc::new(Float32Array::from(vec![Some(0.1), None, Some(0.3)])),
            ],
        )
        .unwrap();
        writer.write_batch(&batch).await.unwrap();
        let rows = writer.close().await.unwrap();
        assert_eq!(rows, 3);

        // Sanity: the bytes are now readable back through the same handle.
        let bytes = handle
            .get_bytes(&handle.data_path().unwrap())
            .await
            .unwrap();
        assert!(!bytes.is_empty());
    }

    /// Outside `feature = "test-hooks"` there is no knob at all: the function
    /// always returns the fixed release-build default.
    #[test]
    fn default_row_group_row_count_is_65_536() {
        assert_eq!(max_row_group_row_count(), 65_536);
    }

    #[cfg(feature = "test-hooks")]
    mod test_hooks_gated {
        use super::*;
        use std::sync::OnceLock;
        use tokio::sync::Mutex;

        fn single_int_col_schema() -> SchemaRef {
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]))
        }

        async fn num_row_groups(handle: &JammiObjectStore) -> usize {
            let bytes = handle
                .get_bytes(&handle.data_path().unwrap())
                .await
                .unwrap();
            let builder =
                parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(bytes)
                    .unwrap();
            builder.metadata().num_row_groups()
        }

        // `JAMMI_TEST_ROW_GROUP_ROWS` is process-global; serialize the tests
        // that mutate it with an async-aware mutex so the guard can be held
        // across `.await` without tripping `clippy::await_holding_lock`, and a
        // panicking test never poisons the lock for the rest of the suite
        // (mirrors `crates/jammi-db/tests/it/audit.rs`'s `env_lock`).
        fn env_lock() -> &'static Mutex<()> {
            static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
            LOCK.get_or_init(|| Mutex::new(()))
        }

        /// With the knob unset, `test-hooks` still falls back to the release
        /// default — the feature alone does not change behaviour.
        #[tokio::test]
        async fn unset_override_falls_back_to_default() {
            let _g = env_lock().lock().await;
            std::env::remove_var(ROW_GROUP_ROWS_ENV);
            assert_eq!(max_row_group_row_count(), 65_536);
        }

        /// A zero or non-numeric override is refused (falls back to the
        /// default) rather than handed to the Parquet writer, which would
        /// otherwise reject `Some(0)` at `open()` time.
        #[tokio::test]
        async fn zero_and_non_numeric_overrides_fall_back_to_default() {
            let _g = env_lock().lock().await;
            for bad in ["0", "not-a-number", ""] {
                std::env::set_var(ROW_GROUP_ROWS_ENV, bad);
                assert_eq!(max_row_group_row_count(), 65_536, "override = {bad:?}");
            }
            std::env::remove_var(ROW_GROUP_ROWS_ENV);
        }

        /// The property this fold exists for: a 200-row write with the knob
        /// at 64 lands in exactly 4 row groups (64, 64, 64, 8), read back
        /// from the Parquet footer.
        #[tokio::test]
        async fn override_splits_small_fixture_into_multiple_row_groups() {
            let _g = env_lock().lock().await;
            std::env::set_var(ROW_GROUP_ROWS_ENV, "64");

            let registry = StorageRegistry::new();
            let url = StorageUrl::memory("test-hooks/row-groups.parquet");
            let driver = registry.driver_for(&url, None).unwrap();
            let handle = JammiObjectStore::new(driver, url);

            let schema = single_int_col_schema();
            let mut writer = ObjectParquetWriter::open(&handle, Arc::clone(&schema))
                .await
                .unwrap();
            let ids: Vec<i64> = (0..200).collect();
            let batch =
                RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(ids))])
                    .unwrap();
            writer.write_batch(&batch).await.unwrap();
            let rows = writer.close().await.unwrap();
            assert_eq!(rows, 200);

            let groups = num_row_groups(&handle).await;
            std::env::remove_var(ROW_GROUP_ROWS_ENV);
            assert_eq!(groups, 4, "expected ceil(200/64) = 4 row groups");
        }
    }
}
