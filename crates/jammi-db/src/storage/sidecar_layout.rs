//! Sidecar-index round-trip helpers.
//!
//! A result table at `<root>/<table>.parquet` may carry a *sidecar bundle*: a
//! set of sibling objects that hold an out-of-band index. Which siblings exist
//! is a function of the table's [`SidecarKind`] — the kind owns its extension
//! set in [`sidecar_extensions`], so writer / reader / cleanup all discover the
//! same files and a new kind adds one registry entry rather than editing shared
//! control flow.
//!
//! The shipped ANN kind ([`SidecarKind::Ann`]) carries three siblings, plus a
//! fourth for a quantized-precision graph and a fifth for a `Binary` one:
//!
//! - `<root>/<table>.usearch`        — serialised USearch graph
//! - `<root>/<table>.rowmap`         — row-id mapping (Jammi-owned format)
//! - `<root>/<table>.manifest.json`  — version, dimensions, count, backend
//! - `<root>/<table>.rawf32`         — raw-`f32` rescore companion (quantized
//!   precision only; absent — and simply skipped — for an `F32` graph)
//! - `<root>/<table>.threshold`      — per-dimension threshold τ (`Binary`
//!   precision only; absent — and simply skipped — for every other precision)
//!
//! USearch's `save` / `load` are file-path-based (FFI), so for non-`file://`
//! backends we materialise the bundle through a tempfile, then push the
//! bytes to / pull from the object store via the [`JammiObjectStore`]
//! handle.

use std::path::Path;

use crate::error::Result;
use crate::index::sidecar::SidecarIndex;

use super::object_store_handle::JammiObjectStore;
use super::url::Scheme;

/// The kind of sidecar bundle a result table carries.
///
/// A table's kind declares which sidecar extensions sit beside its Parquet
/// object. Each variant owns its extension set in [`sidecar_extensions`], so a
/// new derived-table shape is one variant plus one registry arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidecarKind {
    /// Approximate-nearest-neighbour table: a USearch graph plus its row-id
    /// map and manifest.
    Ann,
    /// Lexical (BM25) table: a tantivy inverted index serialised as a
    /// `.tantivy` directory beside the Parquet object. The lexical peer of
    /// [`SidecarKind::Ann`].
    Lexical,
    /// A table that carries no sidecar bundle (e.g. a plain derived/edge
    /// table whose state lives entirely in its Parquet object).
    None,
}

/// The sidecar extensions a [`SidecarKind`] carries, in a stable order.
///
/// This is the single registry the layout consults: writer, reader, and
/// cleanup all enumerate a kind's siblings through here, so they never drift
/// and a new kind is one match arm rather than an edit to every loop. Every
/// extension here is optional at the file level — the remote save/load
/// helpers and [`delete_sidecar`] all skip a sibling that doesn't exist,
/// which is how `rawf32` (quantized-only) coexists with the three that every
/// `Ann` graph carries.
pub fn sidecar_extensions(kind: SidecarKind) -> &'static [&'static str] {
    match kind {
        SidecarKind::Ann => &[
            "usearch",
            "rowmap",
            "manifest.json",
            crate::index::sidecar::RESCORE_COMPANION_EXTENSION,
            crate::index::sidecar::THRESHOLD_COMPANION_EXTENSION,
        ],
        SidecarKind::Lexical => &["tantivy"],
        SidecarKind::None => &[],
    }
}

/// Persist a built [`SidecarIndex`] beside `handle`'s data object.
///
/// For `file://` schemes USearch writes directly to the destination path —
/// no intermediate copy. For cloud schemes we serialise into a tempdir
/// first, then upload each file by extension.
pub async fn save_sidecar(handle: &JammiObjectStore, index: &SidecarIndex) -> Result<()> {
    match handle.scheme() {
        Scheme::File => save_sidecar_local(handle, index),
        _ => save_sidecar_remote(handle, index).await,
    }
}

/// The sidecar extensions a `ready` row of kind `kind`, precision
/// `precision`, and `row_count` rows is REQUIRED to carry right now — the
/// subset of [`sidecar_extensions`] that is not merely optional-if-present
/// but a completeness contract `reconcile`'s row→object arm verifies with a
/// live `exists()` per extension before it will leave the row `ready`.
///
/// [`SidecarKind::Ann`] always requires `usearch`/`rowmap`/`manifest.json`
/// once the table has ANY row (`row_count == 0` requires none of them — a
/// zero-row embedding table never grew a segment, so nothing to require);
/// the quantized-precision rescore companion
/// (`crate::index::sidecar::RESCORE_COMPANION_EXTENSION`) is additionally
/// required for a non-[`crate::config::StoragePrecision::F32`] non-[`crate::config::StoragePrecision::Binary`]
/// graph, and the per-dimension threshold companion
/// (`crate::index::sidecar::THRESHOLD_COMPANION_EXTENSION`) for a
/// [`crate::config::StoragePrecision::Binary`] one — mirroring exactly which siblings
/// [`save_sidecar`] actually writes for that precision.
pub fn required_sidecar_extensions(
    kind: SidecarKind,
    precision: crate::config::StoragePrecision,
    row_count: usize,
) -> Vec<&'static str> {
    use crate::config::StoragePrecision;
    match kind {
        SidecarKind::Ann => {
            if row_count == 0 {
                return Vec::new();
            }
            let mut exts = vec!["usearch", "rowmap", "manifest.json"];
            match precision {
                StoragePrecision::F32 => {}
                StoragePrecision::Binary => {
                    exts.push(crate::index::sidecar::THRESHOLD_COMPANION_EXTENSION)
                }
                StoragePrecision::F16 | StoragePrecision::Int8 => {
                    exts.push(crate::index::sidecar::RESCORE_COMPANION_EXTENSION)
                }
            }
            exts
        }
        SidecarKind::Lexical => vec!["tantivy"],
        SidecarKind::None => Vec::new(),
    }
}

/// Best-effort cleanup: delete every sidecar sibling a `kind` carries.
pub async fn delete_sidecar(handle: &JammiObjectStore, kind: SidecarKind) -> Result<()> {
    for ext in sidecar_extensions(kind) {
        let path = handle.sibling_path(ext)?;
        handle.delete_if_exists(&path).await?;
    }
    Ok(())
}

fn save_sidecar_local(handle: &JammiObjectStore, index: &SidecarIndex) -> Result<()> {
    let base = local_base_path(handle)?;
    index.save(&base)?;
    Ok(())
}

async fn save_sidecar_remote(handle: &JammiObjectStore, index: &SidecarIndex) -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let stem = tmp.path().join("sidecar");
    index.save(&stem)?;

    for ext in sidecar_extensions(SidecarKind::Ann) {
        let local_path = stem.with_extension(ext);
        if !local_path.exists() {
            continue;
        }
        let bytes = std::fs::read(&local_path)?;
        let remote = handle.sibling_path(ext)?;
        handle.put_bytes(&remote, bytes.into()).await?;
    }
    Ok(())
}

/// Resolve the on-disk stem for a `file://` handle. Strips the trailing
/// extension off the data path — `SidecarIndex` does `with_extension()`
/// internally to derive the sibling paths. Shared with
/// [`crate::storage::index_cache`], the load-side counterpart to `save_sidecar`.
pub(crate) fn local_base_path(handle: &JammiObjectStore) -> Result<std::path::PathBuf> {
    let raw = handle.url().path();
    let path = Path::new(raw);
    Ok(path.with_extension(""))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AnnIndexConfig, StoragePrecision};
    use crate::index::VectorIndex;
    use crate::storage::{JammiObjectStore, StorageRegistry, StorageUrl};

    fn build_small_index() -> SidecarIndex {
        let mut idx =
            SidecarIndex::new(4, &AnnIndexConfig::default(), StoragePrecision::F32).unwrap();
        idx.add("row-a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add("row-b", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        idx.add("row-c", &[0.0, 0.0, 1.0, 0.0]).unwrap();
        idx.build().unwrap();
        idx
    }

    #[tokio::test]
    async fn save_then_delete_round_trips_the_bundle() {
        // The write + cleanup halves of the layout. The load half is the
        // content-addressed segment cache's, covered in `storage::index_cache`.
        let registry = StorageRegistry::new();
        let url = StorageUrl::memory("snapshots/2026/data.parquet");
        let driver = registry.driver_for(&url, None).unwrap();
        let handle = JammiObjectStore::new(driver, url);

        save_sidecar(&handle, &build_small_index()).await.unwrap();
        let usearch = handle.sibling_path("usearch").unwrap();
        assert!(handle.exists(&usearch).await.unwrap());

        delete_sidecar(&handle, SidecarKind::Ann).await.unwrap();
        assert!(!handle.exists(&usearch).await.unwrap());
    }

    #[test]
    fn ann_kind_carries_todays_five_extensions() {
        assert_eq!(
            sidecar_extensions(SidecarKind::Ann),
            ["usearch", "rowmap", "manifest.json", "rawf32", "threshold"],
        );
    }

    #[test]
    fn lexical_kind_carries_the_tantivy_sibling() {
        assert_eq!(sidecar_extensions(SidecarKind::Lexical), ["tantivy"]);
    }

    #[test]
    fn none_kind_carries_no_extensions() {
        assert!(sidecar_extensions(SidecarKind::None).is_empty());
    }

    #[test]
    fn required_extensions_are_empty_for_a_zero_row_table() {
        use crate::config::StoragePrecision;
        assert!(required_sidecar_extensions(SidecarKind::Ann, StoragePrecision::F32, 0).is_empty());
    }

    #[test]
    fn required_extensions_f32_never_needs_a_rescore_companion() {
        use crate::config::StoragePrecision;
        assert_eq!(
            required_sidecar_extensions(SidecarKind::Ann, StoragePrecision::F32, 10),
            ["usearch", "rowmap", "manifest.json"],
        );
    }

    #[test]
    fn required_extensions_quantized_needs_the_rescore_companion() {
        use crate::config::StoragePrecision;
        assert_eq!(
            required_sidecar_extensions(SidecarKind::Ann, StoragePrecision::Int8, 10),
            ["usearch", "rowmap", "manifest.json", "rawf32"],
        );
    }

    #[test]
    fn required_extensions_binary_needs_the_threshold_companion() {
        use crate::config::StoragePrecision;
        assert_eq!(
            required_sidecar_extensions(SidecarKind::Ann, StoragePrecision::Binary, 10),
            ["usearch", "rowmap", "manifest.json", "threshold"],
        );
    }
}
