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
            .set_max_row_group_size(65_536)
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
}
