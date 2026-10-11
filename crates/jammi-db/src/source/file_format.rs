//! `File` source driver: file-shaped data (Parquet / CSV / JSON / JSONL) read
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
use futures::{future, TryStreamExt};
use object_store::ObjectMeta;

use super::FileFormat;
use crate::error::{JammiError, Result};
use crate::storage::{CloudConfig, StorageRegistry, StorageUrl};

/// List every object under `table_url` whose path matches `ext` (DataFusion's
/// own extension + glob predicate) AND whose size is nonzero.
///
/// This is the EXACT predicate `ListingOptions::infer_schema` applies
/// internally (`datafusion-catalog-listing`'s `options.rs` drops any
/// zero-size object before merging a schema — an empty file cannot affect
/// the schema and may error on read attempts). A 0-byte file (`touch
/// events.jsonl`, an interrupted export) lists but carries no schema, so it
/// must not count as a match signal any more than a truly absent file would.
///
/// [`create_listing_table`] calls this ONCE and reuses the returned list both
/// to decide whether a corpus has any usable content (the zero-match guard)
/// and to infer the schema (`FileFormat::infer_schema` below) — a single
/// listing site, so the guard's predicate and the schema-inference predicate
/// can never re-diverge the way they did before this filter existed here.
async fn list_matching_files(
    session: &dyn Session,
    store: &dyn object_store::ObjectStore,
    table_url: &ListingTableUrl,
    ext: &str,
) -> Result<Vec<ObjectMeta>> {
    let files = table_url
        .list_all_files(session, store, ext)
        .await?
        .try_filter(|meta| future::ready(meta.size > 0))
        .try_collect()
        .await?;
    Ok(files)
}

/// Build a DataFusion [`ListingTable`] for the given storage URL.
///
/// The URL is validated through [`StorageUrl`] first so unsupported
/// schemes / malformed inputs return a typed `StorageError` rather than
/// a deep `DataFusionError`. The matching `object_store` driver is
/// registered on `session`'s `RuntimeEnv` so DataFusion's own listing logic
/// can list and read the file.
///
/// A listing that matches zero USABLE files (after the size-`> 0` filter
/// above) is a typed `JammiError::Config` naming the extension(s) tried and
/// the url, never a silently registered column-less, row-less table
/// (DataFusion's schema inference succeeds over an empty file list) — this
/// applies to every format (parquet/csv/json/jsonl), not just one.
///
/// [`FileFormat::JsonLines`] additionally gets an ADAPTIVE default extension
/// when the caller has not overridden `file_extension` explicitly: `.jsonl`
/// is tried first, and only when it has zero matches is `.ndjson` tried as a
/// fallback — `.jsonl` always wins over `.ndjson` when a directory holds both
/// (the `.ndjson` files are then silently excluded, the ordinary "wrong
/// extension" listing semantic every other format already has, not a new
/// one). This is the only reachable path onto an `.ndjson` corpus: neither
/// the wire `SourceConnection` message nor the CLI's `--format` flag exposes
/// a `file_extension` override, so without this, `.ndjson`-named files could
/// never be named by the format token alone. An explicit `file_extension`
/// override (Rust-API-only today) is honoured literally, with no adaptive
/// fallback.
///
/// The adaptive resolution runs ONCE, here, at whatever moment the caller
/// invokes this function — it is NOT re-derived from directory contents on
/// every call, because directory contents can change between calls (a
/// `.jsonl` file added to a corpus that previously resolved to `.ndjson`
/// would otherwise flip which rows a later build serves). The returned
/// `Option<String>` is `Some(ext)` exactly when the adaptive path ran
/// (`FileFormat::JsonLines` with no `file_extension` override) — DERIVED
/// from the same `ext` the listing actually served, `adaptive.then(|| ext…)`,
/// so the served extension and the persisted pin can never spell different
/// values. The caller (`JammiSession::add_source`) persists it into the
/// `SourceConnection` it writes to the catalog, so every build from that
/// row — on this process or any other replica resolving the source —
/// passes an explicit `file_extension` and takes the non-adaptive branch
/// below, resolved once and pinned forever (mirrors
/// [`super::SourceConnection::tenant_column`]'s persist-so-every-build-
/// replays-it pattern). `None` for every other format, and for `JsonLines`
/// with an explicit override already in force.
///
/// An explicit `file_extension` override must be non-empty and start with
/// `.` — a typed `JammiError::Config`, not a listing that silently matches
/// nothing (or, worse, matches every file if a caller's off-by-one produced
/// an extension-less glob).
pub async fn create_listing_table(
    registry: &StorageRegistry,
    url: &StorageUrl,
    format: &FileFormat,
    file_extension: Option<&str>,
    cloud: Option<&CloudConfig>,
    session: &dyn Session,
) -> Result<(Arc<dyn TableProvider>, Option<String>)> {
    if let Some(ext) = file_extension {
        if ext.is_empty() || !ext.starts_with('.') {
            return Err(JammiError::InvalidArgument {
                argument: "file_extension".into(),
                reason: format!(
                    "file_extension override '{ext}' must be non-empty and start with '.' (e.g. \
                 '.jsonl')"
                ),
            });
        }
    }

    let driver = registry.driver_for(url, cloud)?;
    crate::storage::read_view::register_read_view(session.runtime_env(), url, Arc::clone(&driver))?;

    let table_url = ListingTableUrl::parse(url.as_str())?;

    let df_format: Arc<dyn datafusion::datasource::file_format::FileFormat> = match format {
        FileFormat::Parquet => {
            Arc::new(datafusion::datasource::file_format::parquet::ParquetFormat::default())
        }
        FileFormat::Csv => Arc::new(datafusion::datasource::file_format::csv::CsvFormat::default()),
        // DataFusion's `JsonFormat` reads line-delimited JSON, so the two
        // JSON formats share it and differ only in the files they list.
        FileFormat::Json | FileFormat::JsonLines => {
            Arc::new(datafusion::datasource::file_format::json::JsonFormat::default())
        }
    };

    // The extensions the listing tries, in order: the caller's override
    // alone, or the format's own ([`FileFormat::extensions`]). The first that
    // matches any file is the one served.
    let candidates: Vec<&str> = match file_extension {
        Some(ext) => vec![ext],
        None => format.extensions().to_vec(),
    };
    let mut served = None;
    for candidate in &candidates {
        let files = list_matching_files(session, driver.as_ref(), &table_url, candidate).await?;
        if !files.is_empty() {
            served = Some((*candidate, files));
            break;
        }
    }
    let (ext, files) = served.ok_or_else(|| JammiError::InvalidArgument {
        argument: "url".into(),
        reason: format!(
            "no files matched extension {} under '{url}' — the source would register with no \
             schema and no rows",
            candidates.join(" or ")
        ),
    })?;
    // A choice among several extensions is pinned: the extension served is
    // the one every later build from the catalog row lists, so a file of the
    // other spelling added later never flips which files the source serves.
    let resolved_for_persistence = (candidates.len() > 1).then(|| ext.to_string());

    let schema = df_format.infer_schema(session, &driver, &files).await?;

    let options = ListingOptions::new(df_format).with_file_extension(ext);
    let config = ListingTableConfig::new(table_url)
        .with_listing_options(options)
        .with_schema(schema);
    let table = ListingTable::try_new(config)?;
    Ok((Arc::new(table), resolved_for_persistence))
}
