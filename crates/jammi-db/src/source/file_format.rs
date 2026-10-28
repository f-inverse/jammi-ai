//! `File` source driver: file-shaped data (Parquet / CSV / JSON / JSONL) read
//! through any [`StorageUrl`]-addressable backend.
//!
//! The driver is scheme-agnostic because DataFusion's `ListingTable`
//! accepts any URL the embedded `object_store` registry recognises — the
//! engine registers the same drivers it built via [`crate::storage`] so
//! cloud schemes work end-to-end without DataFusion having to know about
//! Jammi's [`StorageRegistry`].

use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::{Field, Schema, SchemaRef};
use datafusion::catalog::Session;
use datafusion::datasource::file_format::csv::CsvFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::datasource::TableProvider;
use futures::{future, stream, StreamExt, TryStreamExt};
use object_store::ObjectMeta;

use super::FileFormat;
use crate::error::{JammiError, Result};
use crate::storage::{CloudConfig, DynObjectStore, StorageRegistry, StorageUrl};

/// A DataFusion file reader, which infers a file's schema and scans it.
type Reader = Arc<dyn datafusion::datasource::file_format::FileFormat>;

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
///
/// A listing of several files registers as ONE table, its files the parts
/// of it (#755): files that do not agree on that table's columns are a typed
/// `JammiError::InvalidArgument` naming two of them (see `disagreement`),
/// never one table of the union of their columns.
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

    let df_format: Reader = match format {
        FileFormat::Parquet => {
            Arc::new(datafusion::datasource::file_format::parquet::ParquetFormat::default())
        }
        FileFormat::Csv => Arc::new(csv_format()),
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

    let parts = part_schemas(format, &df_format, session, &driver, &table_url, &files).await?;
    if let Some(why) = disagreement(format, &parts) {
        return Err(JammiError::InvalidArgument {
            argument: "url".into(),
            reason: format!(
                "the files under '{url}' are not parts of one table: {why}; register each file \
                 as its own source to read them as separate tables"
            ),
        });
    }
    let schema = df_format.infer_schema(session, &driver, &files).await?;
    let schema = nullable_where_absent(&schema, &parts);

    let options = ListingOptions::new(df_format).with_file_extension(ext);
    let config = ListingTableConfig::new(table_url)
        .with_listing_options(options)
        .with_schema(schema);
    let table = ListingTable::try_new(config)?;
    Ok((Arc::new(table), resolved_for_persistence))
}

/// The reader a CSV source's files are read with.
fn csv_format() -> CsvFormat {
    CsvFormat::default()
}

/// What a directory's Parquet parts must agree on.
const PARQUET_PARTS: &str = "A directory is read as one table, so every file's columns must be \
                             in its widest file, with the same types";

/// What a directory's CSV parts must agree on.
const CSV_PARTS: &str = "CSV columns are read by position, so every file in a directory must \
                         have the same header, in the same order";

/// Each of a listing's `files` with the schema its format reads from that
/// file alone, named by its path under `table_url`, in path order (the order
/// DataFusion merges a directory's schema in): the parts [`disagreement`]
/// checks. Empty when there is nothing to check: a single file, or JSON.
async fn part_schemas(
    format: &FileFormat,
    reader: &Reader,
    session: &dyn Session,
    store: &DynObjectStore,
    table_url: &ListingTableUrl,
    files: &[ObjectMeta],
) -> Result<Vec<(String, SchemaRef)>> {
    if files.len() < 2 {
        return Ok(Vec::new());
    }
    let reader: Reader = match format {
        // The footers DataFusion's own inference then reads from its cache.
        FileFormat::Parquet => Arc::clone(reader),
        // A CSV part's header alone: no record is read to infer a type from.
        FileFormat::Csv => Arc::new(csv_format().with_schema_infer_max_rec(0)),
        FileFormat::Json | FileFormat::JsonLines => return Ok(Vec::new()),
    };
    let reader = &reader;
    let mut in_order: Vec<&ObjectMeta> = files.iter().collect();
    in_order.sort_by(|a, b| a.location.cmp(&b.location));
    stream::iter(in_order)
        .map(|file| async move {
            let schema = reader
                .infer_schema(session, store, std::slice::from_ref(file))
                .await?;
            Ok::<_, JammiError>((under(table_url, file), schema))
        })
        // Boxed, as DataFusion's own inference boxes this stream: unboxed,
        // its borrowing futures fail the callers' `Send` bound
        // (https://github.com/rust-lang/rust/issues/64552).
        .boxed()
        // As many files at once as DataFusion's own inference reads.
        .buffered(session.config_options().execution.meta_fetch_concurrency)
        .try_collect()
        .await
}

/// `file`'s path under the listing's URL, the name an error gives it.
fn under(table_url: &ListingTableUrl, file: &ObjectMeta) -> String {
    match file.location.prefix_match(table_url.prefix()) {
        Some(segments) => segments
            .map(|segment| segment.as_ref().to_string())
            .collect::<Vec<_>>()
            .join("/"),
        None => file.location.to_string(),
    }
}

/// Where a directory's `parts` fail to form one table, if anywhere.
///
/// A directory registers as one table read across all its files, so they
/// must agree on that table's columns: DataFusion would otherwise read them
/// as the union of every file's columns, NULL wherever a file lacks one — a
/// folder of unrelated exports registered as one table of meaningless rows.
/// What agreeing means follows how each format's reader finds a file's
/// columns:
///
/// - Parquet finds them by name, so a part may lack some of the table's
///   columns (a later partition that added one, the earlier rows reading it
///   as NULL). Every file's columns must be in the directory's widest file,
///   each of a type that merges with that file's, as DataFusion merges the
///   table's schema.
/// - CSV finds them by position, so every file's header must name the same
///   columns in the same order: a reordered header would read its values
///   under the wrong columns.
/// - A JSON record carries its own keys, so a JSON directory is the union of
///   its records' keys, as one file of sparse records already is. Unchecked.
///
/// Every build of a source checks its listing, so a file added after
/// registration is refused the next time a process builds the source.
fn disagreement(format: &FileFormat, parts: &[(String, SchemaRef)]) -> Option<String> {
    match format {
        FileFormat::Parquet => parquet_disagreement(parts),
        FileFormat::Csv => csv_disagreement(parts),
        FileFormat::Json | FileFormat::JsonLines => None,
    }
}

/// Where Parquet `parts` fail to nest in the widest of them, if anywhere.
fn parquet_disagreement(parts: &[(String, SchemaRef)]) -> Option<String> {
    // `min_by_key` keeps the first of equals: the first widest file in path order.
    let (widest, table) = parts
        .iter()
        .min_by_key(|(_, schema)| Reverse(schema.fields().len()))?;
    let table: HashMap<&str, &Field> = table
        .fields()
        .iter()
        .map(|field| (field.name().as_str(), field.as_ref()))
        .collect();
    for (file, schema) in parts {
        let mut absent = Vec::new();
        for field in schema.fields() {
            let Some(&wide) = table.get(field.name().as_str()) else {
                absent.push(field.name().as_str());
                continue;
            };
            if wide.data_type() != field.data_type() && wide.clone().try_merge(field).is_err() {
                return Some(format!(
                    "'{file}' has '{}' as {} where '{widest}', the directory's widest file, has \
                     {}. {PARQUET_PARTS}",
                    field.name(),
                    field.data_type(),
                    wide.data_type()
                ));
            }
        }
        if !absent.is_empty() {
            return Some(format!(
                "'{file}' has {}, which '{widest}', the directory's widest file, does not. \
                 {PARQUET_PARTS}",
                quoted(&absent)
            ));
        }
    }
    None
}

/// Where CSV `parts` fail to share the first one's header, if anywhere.
fn csv_disagreement(parts: &[(String, SchemaRef)]) -> Option<String> {
    let names = |schema: &SchemaRef| -> Vec<String> {
        schema.fields().iter().map(|f| f.name().clone()).collect()
    };
    let (first, header) = parts.first()?;
    let header = names(header);
    parts[1..].iter().find_map(|(file, schema)| {
        let found = names(schema);
        let why = match found
            .iter()
            .zip(&header)
            .position(|(theirs, ours)| theirs != ours)
        {
            Some(at) => format!(
                "'{file}' has '{}' as column {} where '{first}' has '{}'",
                found[at],
                at + 1,
                header[at]
            ),
            None if found.len() != header.len() => format!(
                "'{file}' has {} columns where '{first}' has {}",
                found.len(),
                header.len()
            ),
            None => return None,
        };
        Some(format!("{why}. {CSV_PARTS}"))
    })
}

/// `names`, quoted: the first ten of them, and how many more.
fn quoted(names: &[&str]) -> String {
    let mut shown: Vec<String> = names
        .iter()
        .take(10)
        .map(|name| format!("'{name}'"))
        .collect();
    if names.len() > 10 {
        shown.push(format!("{} more", names.len() - 10));
    }
    shown.join(", ")
}

/// `schema` with every column some part lacks made nullable: a part's rows
/// read a column it lacks as NULL, where DataFusion's merge, seeing only the
/// files that hold the column, keeps their non-nullable declaration.
fn nullable_where_absent(schema: &Schema, parts: &[(String, SchemaRef)]) -> SchemaRef {
    let mut held: HashMap<&str, usize> = HashMap::new();
    for (_, part) in parts {
        for field in part.fields() {
            *held.entry(field.name().as_str()).or_default() += 1;
        }
    }
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|field| {
            let lacked = held.get(field.name().as_str()).copied().unwrap_or(0) < parts.len();
            field
                .as_ref()
                .clone()
                .with_nullable(field.is_nullable() || lacked)
        })
        .collect();
    Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()))
}
