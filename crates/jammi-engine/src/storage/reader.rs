//! Async Parquet reader / metadata probe targeting any
//! `object_store::ObjectStore`. Mirrors the writer surface — same backend,
//! same path semantics.

use bytes::Bytes;
use futures::TryStreamExt;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::async_reader::{ParquetObjectReader, ParquetRecordBatchStreamBuilder};

use super::error::StorageError;
use super::object_store_handle::JammiObjectStore;

/// True if the object at the handle's data path is a syntactically valid
/// Parquet file (readable footer).
///
/// Always reads the file fully because the validator is reused for tiny
/// catalog-row recovery probes. Don't call on multi-GB files.
pub async fn is_valid_parquet(handle: &JammiObjectStore) -> Result<bool, StorageError> {
    let path = handle.data_path()?;
    if !handle.exists(&path).await? {
        return Ok(false);
    }
    let bytes = handle.get_bytes(&path).await?;
    Ok(ParquetRecordBatchReaderBuilder::try_new(bytes).is_ok())
}

/// Row count for a Parquet file by reading its footer metadata.
pub async fn count_parquet_rows(handle: &JammiObjectStore) -> Result<usize, StorageError> {
    let path = handle.data_path()?;
    let bytes = handle.get_bytes(&path).await?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).map_err(|e| {
        StorageError::layout(path.to_string(), format!("Parquet metadata read: {e}"))
    })?;
    Ok(builder.metadata().file_metadata().num_rows() as usize)
}

/// Read every byte of the underlying object into memory. Used by the
/// sidecar-index loader, which then hands the bytes to USearch via a
/// temp-file shim.
pub async fn read_all(handle: &JammiObjectStore) -> Result<Bytes, StorageError> {
    let path = handle.data_path()?;
    handle.get_bytes(&path).await
}

/// Stream Arrow batches from the object's Parquet file. Used by the
/// recovery path to rebuild the ANN index after a crash without loading
/// the full file into memory.
pub async fn read_all_record_batches(
    handle: &JammiObjectStore,
) -> Result<Vec<arrow::array::RecordBatch>, StorageError> {
    let path = handle.data_path()?;
    let reader = ParquetObjectReader::new(handle.driver(), path.clone());
    let stream = ParquetRecordBatchStreamBuilder::new(reader)
        .await
        .map_err(|e| {
            StorageError::layout(path.to_string(), format!("Parquet reader builder: {e}"))
        })?
        .build()
        .map_err(|e| StorageError::layout(path.to_string(), format!("Parquet stream build: {e}")))?;
    stream
        .try_collect()
        .await
        .map_err(|e| StorageError::layout(path.to_string(), format!("Parquet batch read: {e}")))
}
