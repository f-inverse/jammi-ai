//! High-level wrapper around `Arc<dyn ObjectStore>` carrying the URL it was
//! built from. The handle is the read/write surface every Jammi component
//! (result writer, sidecar layout, ANN index loader) calls into.

use std::sync::Arc;

use bytes::Bytes;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;

use super::builder::DynObjectStore;
use super::error::StorageError;
use super::url::{Scheme, StorageUrl};

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
    /// matches the engine's "best-effort cleanup" contract).
    pub async fn delete_if_exists(&self, path: &ObjectPath) -> Result<(), StorageError> {
        match self.driver.delete(path).await {
            Ok(()) => Ok(()),
            Err(object_store::Error::NotFound { .. }) => Ok(()),
            Err(e) => Err(StorageError::io(path.to_string(), e)),
        }
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
