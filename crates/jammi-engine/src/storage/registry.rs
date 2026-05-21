//! Process-wide cache of constructed `ObjectStore` drivers.
//!
//! Two URLs pointing at the same `(scheme, bucket)` share a single driver
//! instance so we don't repeatedly re-validate credentials / re-open
//! connection pools on every request.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::builder::{build_object_store, DynObjectStore};
use super::config::CloudConfig;
use super::error::StorageError;
use super::url::{Scheme, StorageUrl};

/// Cache key — driver identity is determined by the scheme plus the first
/// path segment (bucket for cloud schemes, root for `file`/`memory`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DriverKey {
    scheme: Scheme,
    root: String,
}

impl DriverKey {
    fn from_url(url: &StorageUrl) -> Self {
        let scheme = url.scheme();
        let root = match scheme {
            Scheme::File | Scheme::Memory => String::new(),
            _ => url
                .path()
                .split('/')
                .next()
                .unwrap_or_default()
                .to_string(),
        };
        Self { scheme, root }
    }
}

/// Holds the per-`(scheme, root)` driver cache. Cheap to clone (interior
/// `Arc<Mutex<…>>`), so callers can stash one on each session without
/// fighting borrow checking.
#[derive(Clone, Default)]
pub struct StorageRegistry {
    inner: Arc<Mutex<HashMap<DriverKey, DynObjectStore>>>,
}

impl StorageRegistry {
    /// Build an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolve (or construct and cache) the driver for `url`.
    ///
    /// `config` is consulted on the first construction for `(scheme, root)`;
    /// subsequent calls for the same key return the cached driver and
    /// ignore `config`. Callers needing distinct credentials per call
    /// should use distinct buckets.
    pub fn driver_for(
        &self,
        url: &StorageUrl,
        config: Option<&CloudConfig>,
    ) -> Result<DynObjectStore, StorageError> {
        let key = DriverKey::from_url(url);
        let mut guard = self.inner.lock().expect("storage registry mutex poisoned");
        if let Some(existing) = guard.get(&key) {
            return Ok(Arc::clone(existing));
        }
        let driver = build_object_store(url, config)?;
        guard.insert(key, Arc::clone(&driver));
        Ok(driver)
    }

    /// Drop the cached driver for `url`. The next `driver_for` call
    /// reconstructs the driver — used by tests that need to swap a
    /// credential set for the same bucket.
    pub fn evict(&self, url: &StorageUrl) {
        let key = DriverKey::from_url(url);
        let mut guard = self.inner.lock().expect("storage registry mutex poisoned");
        guard.remove(&key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caches_drivers_per_root() {
        let r = StorageRegistry::new();
        let a = StorageUrl::memory("benchmarks/2026.parquet");
        let b = StorageUrl::memory("benchmarks/2027.parquet");

        let d1 = r.driver_for(&a, None).unwrap();
        let d2 = r.driver_for(&b, None).unwrap();
        // Both memory URLs share the empty-root key → same driver.
        assert!(Arc::ptr_eq(&d1, &d2));
    }

    #[test]
    fn evict_clears_cache() {
        let r = StorageRegistry::new();
        let u = StorageUrl::memory("snapshots/x");
        let d1 = r.driver_for(&u, None).unwrap();
        r.evict(&u);
        let d2 = r.driver_for(&u, None).unwrap();
        assert!(!Arc::ptr_eq(&d1, &d2));
    }
}
