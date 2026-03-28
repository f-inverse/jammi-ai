use std::path::Path;
use std::sync::Arc;

use datafusion::catalog::MemorySchemaProvider;
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
///
/// Ensures the `jammi` schema exists in the default catalog before registration.
pub async fn register_parquet_table(ctx: &SessionContext, name: &str, path: &Path) -> Result<()> {
    // Ensure the "jammi" schema exists in the default catalog.
    // DataFusion does not auto-create schemas for register_parquet.
    let default_catalog_name = ctx.state().config_options().catalog.default_catalog.clone();
    let default_catalog = ctx
        .catalog(&default_catalog_name)
        .ok_or_else(|| JammiError::Other("Default catalog not found".into()))?;
    if default_catalog.schema("jammi").is_none() {
        let _ = default_catalog.register_schema("jammi", Arc::new(MemorySchemaProvider::new()));
    }

    let table_ref = format!("jammi.{name}");
    let path_str = path
        .to_str()
        .ok_or_else(|| JammiError::Other("Non-UTF8 path".into()))?;
    ctx.register_parquet(&table_ref, path_str, ParquetReadOptions::default())
        .await?;
    Ok(())
}
