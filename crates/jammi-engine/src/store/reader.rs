use std::path::Path;

use datafusion::datasource::file_format::options::ParquetReadOptions;
use datafusion::prelude::SessionContext;

use crate::error::{JammiError, Result};

/// Check whether a file is a valid Parquet file (has a readable footer).
pub fn is_valid_parquet(path: &str) -> bool {
    let Ok(file) = std::fs::File::open(path) else {
        return false;
    };
    parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file).is_ok()
}

/// Count the number of rows in a Parquet file by reading its metadata.
pub fn count_parquet_rows(path: &str) -> Result<usize> {
    let file = std::fs::File::open(path)?;
    let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| JammiError::Other(format!("Parquet metadata read: {e}")))?;
    Ok(builder.metadata().file_metadata().num_rows() as usize)
}

/// Register a Parquet file as a DataFusion table under the name `jammi.{name}`.
pub async fn register_parquet_table(ctx: &SessionContext, name: &str, path: &Path) -> Result<()> {
    let table_ref = format!("jammi.{name}");
    let path_str = path
        .to_str()
        .ok_or_else(|| JammiError::Other("Non-UTF8 path".into()))?;
    ctx.register_parquet(&table_ref, path_str, ParquetReadOptions::default())
        .await?;
    Ok(())
}
