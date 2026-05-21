//! Sidecar-index round-trip helpers.
//!
//! A sidecar index for a result table at `<root>/<table>.parquet` lives at
//! three sibling objects:
//!
//! - `<root>/<table>.usearch`        — serialised USearch graph
//! - `<root>/<table>.rowmap`         — row-id mapping (Jammi-owned format)
//! - `<root>/<table>.manifest.json`  — version, dimensions, count, backend
//!
//! USearch's `save` / `load` are file-path-based (FFI), so for non-`file://`
//! backends we materialise the bundle through a tempfile, then push the
//! bytes to / pull from the object store via the [`JammiObjectStore`]
//! handle.

use std::path::Path;

use crate::error::Result;
use crate::index::sidecar::SidecarIndex;

use super::error::StorageError;
use super::object_store_handle::JammiObjectStore;
use super::url::Scheme;

/// Names of the three extensions that make up a sidecar bundle. Kept in
/// one place so writer / reader / cleanup never drift.
pub const SIDECAR_EXTENSIONS: [&str; 3] = ["usearch", "rowmap", "manifest.json"];

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

/// Load a sidecar bundle into a [`SidecarIndex`].
///
/// For `file://` schemes USearch reads the destination path directly. For
/// cloud schemes we download into a tempdir, then load from there.
pub async fn load_sidecar(handle: &JammiObjectStore) -> Result<SidecarIndex> {
    match handle.scheme() {
        Scheme::File => load_sidecar_local(handle),
        _ => load_sidecar_remote(handle).await,
    }
}

/// Best-effort cleanup: delete every sidecar sibling that exists.
pub async fn delete_sidecar(handle: &JammiObjectStore) -> Result<()> {
    for ext in SIDECAR_EXTENSIONS {
        let path = handle.sibling_path(ext).map_err(map_storage)?;
        handle
            .delete_if_exists(&path)
            .await
            .map_err(map_storage)?;
    }
    Ok(())
}

fn save_sidecar_local(handle: &JammiObjectStore, index: &SidecarIndex) -> Result<()> {
    let base = local_base_path(handle)?;
    index.save(&base)?;
    Ok(())
}

fn load_sidecar_local(handle: &JammiObjectStore) -> Result<SidecarIndex> {
    let base = local_base_path(handle)?;
    SidecarIndex::load(&base)
}

async fn save_sidecar_remote(handle: &JammiObjectStore, index: &SidecarIndex) -> Result<()> {
    let tmp = tempfile::tempdir().map_err(|e| {
        crate::error::JammiError::Other(format!("sidecar tempdir create: {e}"))
    })?;
    let stem = tmp.path().join("sidecar");
    index.save(&stem)?;

    for ext in SIDECAR_EXTENSIONS {
        let local_path = stem.with_extension(ext);
        if !local_path.exists() {
            continue;
        }
        let bytes = std::fs::read(&local_path)?;
        let remote = handle.sibling_path(ext).map_err(map_storage)?;
        handle
            .put_bytes(&remote, bytes.into())
            .await
            .map_err(map_storage)?;
    }
    Ok(())
}

async fn load_sidecar_remote(handle: &JammiObjectStore) -> Result<SidecarIndex> {
    let tmp = tempfile::tempdir().map_err(|e| {
        crate::error::JammiError::Other(format!("sidecar tempdir create: {e}"))
    })?;
    let stem = tmp.path().join("sidecar");

    for ext in SIDECAR_EXTENSIONS {
        let remote = handle.sibling_path(ext).map_err(map_storage)?;
        if !handle.exists(&remote).await.map_err(map_storage)? {
            continue;
        }
        let bytes = handle.get_bytes(&remote).await.map_err(map_storage)?;
        std::fs::write(stem.with_extension(ext), &bytes)?;
    }

    SidecarIndex::load(&stem)
}

/// Resolve the on-disk stem for a `file://` handle. Strips the `.parquet`
/// extension off the data path — `SidecarIndex` does `with_extension()`
/// internally to derive the three sibling paths.
fn local_base_path(handle: &JammiObjectStore) -> Result<std::path::PathBuf> {
    let raw = handle.url().path();
    let path = Path::new(raw);
    Ok(path.with_extension(""))
}

fn map_storage(e: StorageError) -> crate::error::JammiError {
    crate::error::JammiError::Other(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::VectorIndex;
    use crate::storage::{JammiObjectStore, StorageRegistry, StorageUrl};

    fn build_small_index() -> SidecarIndex {
        let mut idx = SidecarIndex::new(4).unwrap();
        idx.add("row-a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add("row-b", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        idx.add("row-c", &[0.0, 0.0, 1.0, 0.0]).unwrap();
        idx.build().unwrap();
        idx
    }

    #[tokio::test]
    async fn sidecar_round_trip_memory() {
        let registry = StorageRegistry::new();
        let url = StorageUrl::memory("snapshots/2026/data.parquet");
        let driver = registry.driver_for(&url, None).unwrap();
        let handle = JammiObjectStore::new(driver, url);

        let index = build_small_index();
        save_sidecar(&handle, &index).await.unwrap();

        let loaded = load_sidecar(&handle).await.unwrap();
        let hits = loaded.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(hits.first().map(|(id, _)| id.as_str()), Some("row-a"));

        delete_sidecar(&handle).await.unwrap();
        let path = handle.sibling_path("usearch").unwrap();
        assert!(!handle.exists(&path).await.unwrap());
    }
}
