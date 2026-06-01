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
            _ => url.path().split('/').next().unwrap_or_default().to_string(),
        };
        Self { scheme, root }
    }
}

/// Holds the per-`(scheme, root)` driver cache. Cheap to clone (interior
/// `Arc<Mutex<…>>`), so callers can stash one on each session without
/// fighting borrow checking.
///
/// A registry may carry a **default** [`CloudConfig`] — the deploy-wide
/// credentials from `[storage.cloud]`. Every `driver_for(url, None)` call
/// falls back to it, so a session built with a default cloud config resolves
/// both its result root and a wire `AddSource("r2://…")` (whose
/// `SourceConnection` carries no inline credentials) without each call site
/// re-threading the config.
#[derive(Clone, Default)]
pub struct StorageRegistry {
    inner: Arc<Mutex<HashMap<DriverKey, DynObjectStore>>>,
    default_cloud: Option<Arc<CloudConfig>>,
}

impl StorageRegistry {
    /// Build an empty registry with no default cloud config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a registry whose `driver_for(url, None)` calls fall back to
    /// `default_cloud` — the deploy-wide `[storage.cloud]` credentials.
    pub fn with_default_cloud(default_cloud: Option<CloudConfig>) -> Self {
        Self {
            inner: Arc::default(),
            default_cloud: default_cloud.map(Arc::new),
        }
    }

    /// The default cloud config this registry falls back to, if any.
    pub fn default_cloud(&self) -> Option<&CloudConfig> {
        self.default_cloud.as_deref()
    }

    /// Resolve (or construct and cache) the driver for `url`.
    ///
    /// `config` is consulted on the first construction for `(scheme, root)`;
    /// when it is `None`, the registry's default cloud config (if any) is
    /// used instead. Subsequent calls for the same key return the cached
    /// driver and ignore `config`. Callers needing distinct credentials per
    /// call should use distinct buckets.
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
        let effective = config.or(self.default_cloud.as_deref());
        let driver = build_object_store(url, effective)?;
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

    #[test]
    fn default_cloud_is_exposed() {
        use super::super::config::{CloudConfig, R2Config};
        let cfg = CloudConfig::R2(R2Config {
            account_id: Some("acct".into()),
            ..Default::default()
        });
        let r = StorageRegistry::with_default_cloud(Some(cfg));
        assert!(matches!(r.default_cloud(), Some(CloudConfig::R2(_))));

        let empty = StorageRegistry::new();
        assert!(empty.default_cloud().is_none());
    }

    // R2's builder requires an R2Config; with `None` passed to `driver_for`
    // the registry must fall back to its default cloud config to build the
    // driver. Without the fallback this would error with "R2 requires an
    // R2Config". Gated on the driver feature so it compiles only when R2 is
    // built in; no network — `AmazonS3Builder::build()` only constructs the
    // client.
    #[cfg(feature = "storage-r2")]
    #[test]
    fn driver_for_falls_back_to_default_cloud() {
        use super::super::config::{CloudConfig, R2Config};
        let cfg = CloudConfig::R2(R2Config {
            account_id: Some("abc123".into()),
            access_key_id: Some("k".into()),
            secret_access_key: Some("s".into()),
            ..Default::default()
        });
        let r = StorageRegistry::with_default_cloud(Some(cfg));
        let url = StorageUrl::parse("r2://archives/x").unwrap();
        // `None` here: the registry supplies the default cloud config.
        let store = r
            .driver_for(&url, None)
            .expect("r2 driver builds via default cloud");
        let _ = format!("{store}");
    }
}
