//! `File` source driver: file-shaped data (Parquet / CSV / JSON) read
//! through any [`StorageUrl`]-addressable backend.
//!
//! The driver is scheme-agnostic because DataFusion's `ListingTable`
//! accepts any URL the embedded `object_store` registry recognises — the
//! engine registers the same drivers it built via [`crate::storage`] so
//! cloud schemes work end-to-end without DataFusion having to know about
//! Jammi's [`StorageRegistry`].

use std::sync::Arc;

use datafusion::catalog::Session;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::datasource::TableProvider;
use datafusion::prelude::SessionContext;
use futures::TryStreamExt;

use super::FileFormat;
use crate::error::{JammiError, Result};
use crate::storage::{CloudConfig, StorageRegistry, StorageUrl};

/// Build a DataFusion [`ListingTable`] for the given storage URL.
///
/// The URL is validated through [`StorageUrl`] first so unsupported
/// schemes / malformed inputs return a typed `StorageError` rather than
/// a deep `DataFusionError`. The matching `object_store` driver is
/// registered on `ctx`'s `RuntimeEnv` so DataFusion's own listing logic
/// can list and read the file.
pub async fn create_listing_table(
    ctx: &SessionContext,
    registry: &StorageRegistry,
    url: &StorageUrl,
    format: &FileFormat,
    file_extension: Option<&str>,
    cloud: Option<&CloudConfig>,
    session: &dyn Session,
) -> Result<Arc<dyn TableProvider>> {
    let driver = registry.driver_for(url, cloud)?;
    register_driver_for_url(ctx, url, Arc::clone(&driver))?;

    let table_url = ListingTableUrl::parse(url.as_str())?;

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
        FileFormat::JsonLines => (
            // DataFusion's `JsonFormat` is already line-delimited, so this
            // shares the same reader as `FileFormat::Json` — only the default
            // directory-listing extension differs.
            Arc::new(datafusion::datasource::file_format::json::JsonFormat::default()),
            ".jsonl",
        ),
        FileFormat::Avro => return Err(JammiError::Config("Avro not yet supported".into())),
    };

    let ext = file_extension.unwrap_or(default_ext);

    // A zero-match listing infers an empty (column-less, row-less) schema
    // SILENTLY — DataFusion's `infer_schema` succeeds over an empty file list,
    // registering a table nobody asked for. Fail loudly instead, naming the
    // extension filter and the url, BEFORE schema inference runs: a `.jsonl`
    // corpus registered as `json` (default `.json` glob), a `.json` corpus
    // registered as `jsonl` (default `.jsonl` glob), an `.ndjson`-named corpus
    // under either default extension, a `.JSONL` case mismatch, and a
    // `.jsonl.gz` corpus all resolve to zero matches here and are rejected —
    // this check runs for every format arm (parquet/csv/json/jsonl), not just
    // one, since it sits above the per-format branch.
    let mut matches = table_url
        .list_all_files(session, driver.as_ref(), ext)
        .await?;
    if matches.try_next().await?.is_none() {
        return Err(JammiError::Config(format!(
            "no files matched extension '{ext}' under '{url}' — the source \
             would register with no schema and no rows"
        )));
    }
    drop(matches);

    let options = ListingOptions::new(df_format).with_file_extension(ext);
    let config = ListingTableConfig::new(table_url)
        .with_listing_options(options)
        .infer_schema(session)
        .await?;
    let table = ListingTable::try_new(config)?;
    Ok(Arc::new(table))
}

/// Register the driver we built ourselves with DataFusion's runtime so its
/// `ListingTableUrl` resolves the same backend on every read.
///
/// `file://` and `memory://` are already known by DataFusion's defaults;
/// only cloud schemes need explicit registration.
fn register_driver_for_url(
    ctx: &SessionContext,
    url: &StorageUrl,
    driver: Arc<dyn object_store::ObjectStore>,
) -> Result<()> {
    use crate::storage::Scheme;
    match url.scheme() {
        Scheme::File | Scheme::Memory => return Ok(()),
        _ => {}
    }
    let parsed = ::url::Url::parse(url.as_str())
        .map_err(|e| JammiError::Config(format!("Storage URL '{url}' did not re-parse: {e}")))?;
    ctx.runtime_env().register_object_store(&parsed, driver);
    Ok(())
}
