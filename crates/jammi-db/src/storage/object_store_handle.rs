//! High-level wrapper around `Arc<dyn ObjectStore>` carrying the URL it was
//! built from. The handle is the read/write surface every Jammi component
//! (result writer, sidecar layout, ANN index loader) calls into.

use std::sync::Arc;

use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures::TryStreamExt;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;

use super::builder::DynObjectStore;
use super::error::StorageError;
use super::url::{Scheme, StorageUrl};

/// The two states a [`JammiObjectStore::delete_if_exists`] call can end in —
/// deliberately NOT collapsed into a bare `Result<(), StorageError>`, because
/// a 404 and an actual removal are different facts a caller may need to act
/// on differently (most sharply: `store::reconcile`'s byte-accounting, which
/// must credit `bytes_reclaimed` only for a key THIS call actually removed,
/// never one that was already gone when the delete ran — see esc-484's
/// vanish-window defect, where collapsing the two let a race credit bytes
/// that were never freed by the pass reporting them).
///
/// Both variants are **driver-reported**, not independently verified: they
/// reflect only what the underlying `object_store` driver's `delete` call
/// returned, never a fresh existence check. Drivers whose delete is
/// idempotent never distinguish the two — the AWS driver used for both
/// `s3://` and `r2://` roots issues a bare `DELETE` and returns success on a
/// 204, which S3 answers for a key that does not exist, so `Absent` is
/// unreachable on those roots and an already-gone key is reported `Deleted`
/// (open as esc-103 in `.jammi/escapes.jsonl`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeleteOutcome {
    /// The driver's `delete` call returned success. On a driver whose
    /// delete is idempotent (`s3://`/`r2://`) this does NOT prove an object
    /// was actually removed — a key that was already absent is reported
    /// `Deleted` too (esc-103).
    Deleted,
    /// The driver's `delete` call surfaced a not-found error. Only drivers
    /// that report deletes of missing keys as an error reach this variant
    /// (the local filesystem driver does); the `s3://`/`r2://` AWS driver
    /// never does, so `Absent` is unreachable there (esc-103).
    Absent,
}

/// One object a `JammiObjectStore::list` enumeration found under a prefix —
/// the engine's own shape (never `object_store::ObjectMeta` directly), so a
/// caller (only [`crate::store::reconcile`] today — see the never-`LIST`
/// hot-path rule below) depends on exactly the three fields reconcile needs,
/// not the full upstream metadata surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectMeta {
    /// The object's full path, relative to the driver's own root.
    pub path: ObjectPath,
    /// Size in bytes.
    pub size: u64,
    /// Last-modified time, as reported by the backend.
    pub last_modified: DateTime<Utc>,
}

/// A constructed object-store driver bound to the URL that produced it.
///
/// Cloning is cheap (`Arc<dyn ObjectStore>` clone) and intentional — the
/// same handle is shared across the writer, reader, and sidecar-layout
/// helpers so they all hit the same driver instance.
#[derive(Clone)]
pub struct JammiObjectStore {
    driver: DynObjectStore,
    url: StorageUrl,
}

impl JammiObjectStore {
    /// Construct a handle from a previously-built driver and the URL it
    /// was opened against.
    pub fn new(driver: DynObjectStore, url: StorageUrl) -> Self {
        Self { driver, url }
    }

    /// The URL this handle was opened against.
    pub fn url(&self) -> &StorageUrl {
        &self.url
    }

    /// The scheme of the URL — convenient when callers branch on local
    /// vs cloud (e.g. ANN-index loaders that need a temp-file copy for
    /// cloud schemes).
    pub fn scheme(&self) -> Scheme {
        self.url.scheme()
    }

    /// Underlying `Arc<dyn ObjectStore>`. Exposed for the writer / reader
    /// helpers; user code should never reach for it directly.
    pub fn driver(&self) -> Arc<dyn ObjectStore> {
        Arc::clone(&self.driver)
    }

    /// Path component of the handle's URL, parsed as an `object_store::Path`.
    /// This is what the writer / reader pass to `driver.put / driver.get`.
    pub fn data_path(&self) -> Result<ObjectPath, StorageError> {
        Self::parse_path(&self.url, self.url.path())
    }

    /// Sibling path next to the data path: `data.parquet` → `data.<ext>`.
    ///
    /// Used by [`crate::storage::sidecar_layout`] to derive `.usearch`,
    /// `.rowmap`, `.manifest.json` paths from the Parquet path.
    pub fn sibling_path(&self, ext: &str) -> Result<ObjectPath, StorageError> {
        let base = self.url.path();
        let (stem, _) = base
            .rsplit_once('.')
            .ok_or_else(|| StorageError::layout(base, "no file extension to swap"))?;
        let candidate = format!("{stem}.{ext}");
        Self::parse_path(&self.url, &candidate)
    }

    /// Path under the same directory as the handle's data path.
    pub fn child_path(&self, name: &str) -> Result<ObjectPath, StorageError> {
        let base = self.url.path();
        let parent = base.rsplit_once('/').map(|(p, _)| p).unwrap_or("");
        let candidate = if parent.is_empty() {
            name.to_string()
        } else {
            format!("{parent}/{name}")
        };
        Self::parse_path(&self.url, &candidate)
    }

    /// Convenience: write `bytes` to `path` on the underlying driver.
    pub async fn put_bytes(&self, path: &ObjectPath, bytes: Bytes) -> Result<(), StorageError> {
        self.driver
            .put(path, bytes.into())
            .await
            .map_err(|e| StorageError::io(path.to_string(), e))?;
        Ok(())
    }

    /// Convenience: read `path` fully into memory.
    pub async fn get_bytes(&self, path: &ObjectPath) -> Result<Bytes, StorageError> {
        let result = self
            .driver
            .get(path)
            .await
            .map_err(|e| StorageError::io(path.to_string(), e))?;
        result
            .bytes()
            .await
            .map_err(|e| StorageError::io(path.to_string(), e))
    }

    /// Convenience: delete `path` if it exists (404 is *not* an error —
    /// matches the engine's "best-effort cleanup" contract). Returns which
    /// of the two the driver reported ([`DeleteOutcome`]) rather than
    /// collapsing both into a bare success: a caller that credits bytes or
    /// keys against this deletion (`store::reconcile`'s accounting) must be
    /// able to tell "this call removed the object" from "it was already
    /// gone" — see [`DeleteOutcome`]'s own doc comment for why that
    /// distinction is only as good as the driver underneath (unreachable
    /// `Absent` on `s3://`/`r2://`, esc-103).
    pub async fn delete_if_exists(&self, path: &ObjectPath) -> Result<DeleteOutcome, StorageError> {
        match self.driver.delete(path).await {
            Ok(()) => Ok(DeleteOutcome::Deleted),
            Err(object_store::Error::NotFound { .. }) => Ok(DeleteOutcome::Absent),
            Err(e) => Err(StorageError::io(path.to_string(), e)),
        }
    }

    /// List every object under `prefix` (recursive — object stores have no
    /// directory concept, so this is a flat enumeration of every key sharing
    /// the prefix), sorted by path.
    ///
    /// `pub(crate)`: the ONLY caller is
    /// [`crate::store::reconcile`](crate::store::reconcile). Every hot
    /// read/write path in this engine resolves a key it already knows (a
    /// catalog row's `parquet_path`, a manifest's listed entry) rather than
    /// discovering keys by listing — object-store `LIST` is the operation
    /// most backends serve slowest and least consistently (S3's `LIST` is
    /// eventually consistent under some storage classes; GCS and Azure both
    /// rate-limit it far more aggressively than `GET`/`HEAD`/`PUT`). Reconcile
    /// is the one deliberately out-of-band maintenance pass that is allowed to
    /// pay that cost; nothing else in the crate may call this method.
    pub(crate) async fn list(&self, prefix: &ObjectPath) -> Result<Vec<ObjectMeta>, StorageError> {
        let mut metas: Vec<ObjectMeta> = self
            .driver
            .list(Some(prefix))
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| StorageError::io(prefix.to_string(), e))?
            .into_iter()
            .map(|m| ObjectMeta {
                path: m.location,
                size: m.size,
                last_modified: m.last_modified,
            })
            .collect();
        metas.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(metas)
    }

    /// True if `path` exists in the underlying store.
    pub async fn exists(&self, path: &ObjectPath) -> Result<bool, StorageError> {
        match self.driver.head(path).await {
            Ok(_) => Ok(true),
            Err(object_store::Error::NotFound { .. }) => Ok(false),
            Err(e) => Err(StorageError::io(path.to_string(), e)),
        }
    }

    fn parse_path(url: &StorageUrl, raw: &str) -> Result<ObjectPath, StorageError> {
        // For cloud schemes the first path segment is the bucket — the
        // driver was bound to that bucket at build time so we strip it
        // before handing the key to `object_store::Path::parse`.
        let key = match url.scheme() {
            Scheme::File | Scheme::Memory => raw.trim_start_matches('/').to_string(),
            _ => raw
                .split_once('/')
                .map(|(_, rest)| rest.to_string())
                .unwrap_or_default(),
        };
        ObjectPath::parse(&key).map_err(|e| StorageError::layout(raw, e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::StorageRegistry;

    #[tokio::test]
    async fn round_trip_via_memory() {
        let registry = StorageRegistry::new();
        let url = StorageUrl::memory("benchmarks/2026.parquet");
        let driver = registry.driver_for(&url, None).unwrap();
        let handle = JammiObjectStore::new(driver, url);

        let path = handle.data_path().unwrap();
        let payload = Bytes::from_static(b"hello world");
        handle.put_bytes(&path, payload.clone()).await.unwrap();
        let read = handle.get_bytes(&path).await.unwrap();
        assert_eq!(read, payload);

        assert!(handle.exists(&path).await.unwrap());
        handle.delete_if_exists(&path).await.unwrap();
        assert!(!handle.exists(&path).await.unwrap());
    }

    #[test]
    fn sibling_path_swaps_extension() {
        let registry = StorageRegistry::new();
        let url = StorageUrl::memory("benchmarks/data.parquet");
        let driver = registry.driver_for(&url, None).unwrap();
        let handle = JammiObjectStore::new(driver, url);
        let sibling = handle.sibling_path("usearch").unwrap();
        assert!(sibling.to_string().ends_with("data.usearch"));
    }

    #[tokio::test]
    async fn list_enumerates_every_object_under_a_prefix_sorted() {
        let registry = StorageRegistry::new();
        let url = StorageUrl::memory("root");
        let driver = registry.driver_for(&url, None).unwrap();
        let handle = JammiObjectStore::new(driver, url);

        for name in ["b/two.parquet", "a/one.parquet", "a/one.manifest.json"] {
            let path = ObjectPath::parse(name).unwrap();
            handle
                .put_bytes(&path, Bytes::from_static(b"x"))
                .await
                .unwrap();
        }
        let listed = handle.list(&ObjectPath::parse("").unwrap()).await.unwrap();
        let names: Vec<String> = listed.iter().map(|m| m.path.to_string()).collect();
        assert_eq!(
            names,
            vec!["a/one.manifest.json", "a/one.parquet", "b/two.parquet"],
            "list must be sorted and enumerate every key under the prefix"
        );
        assert!(listed.iter().all(|m| m.size == 1));
    }

    #[test]
    fn child_path_appends_to_dir() {
        let registry = StorageRegistry::new();
        let url = StorageUrl::memory("snapshots/2026/data.parquet");
        let driver = registry.driver_for(&url, None).unwrap();
        let handle = JammiObjectStore::new(driver, url);
        let child = handle.child_path("manifest.json").unwrap();
        assert!(child.to_string().ends_with("snapshots/2026/manifest.json"));
    }
}
