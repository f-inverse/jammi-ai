use std::sync::Arc;

use datafusion::catalog::Session;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::datasource::TableProvider;

use super::FileFormat;
use crate::error::{JammiError, Result};

/// Create a ListingTable for a local file path.
pub async fn create_listing_table(
    path: &str,
    format: &FileFormat,
    file_extension: Option<&str>,
    session: &dyn Session,
) -> Result<Arc<dyn TableProvider>> {
    let table_url = ListingTableUrl::parse(path)?;

    let (df_format, default_ext): (
        Arc<dyn datafusion::datasource::file_format::FileFormat>,
        &str,
    ) = match format {
        FileFormat::Parquet => (
            Arc::new(datafusion::datasource::file_format::parquet::ParquetFormat::default()),
            ".parquet",
        ),
        FileFormat::Csv => (
            Arc::new(datafusion::datasource::file_format::csv::CsvFormat::default()),
            ".csv",
        ),
        FileFormat::Json => (
            Arc::new(datafusion::datasource::file_format::json::JsonFormat::default()),
            ".json",
        ),
        FileFormat::Avro => return Err(JammiError::Config("Avro not yet supported".into())),
    };

    let ext = file_extension.unwrap_or(default_ext);
    let options = ListingOptions::new(df_format).with_file_extension(ext);
    let config = ListingTableConfig::new(table_url)
        .with_listing_options(options)
        .infer_schema(session)
        .await?;
    let table = ListingTable::try_new(config)?;
    Ok(Arc::new(table))
}
