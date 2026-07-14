//! Content-addressed local cache for ANN index **segments**.
//!
//! A table's ANN index is a set of immutable segments, each a
//! [`SidecarIndex`] bundle beside the
//! result Parquet. This cache is the single load path a segment takes on the way
//! into a [`SegmentedIndex`](crate::index::SegmentedIndex): it materialises a
//! remote segment bundle into a local directory USearch can read, keyed so that
//! the same immutable segment is fetched from object storage at most once per
//! process lifetime.
//!
//! It mirrors the model-artifact cache ([`crate::store::artifact`]): download
//! into a sibling tempdir, then atomically rename into a content-addressed cache
//! directory so a concurrent load never observes a partial bundle; a `file://`
//! bundle short-circuits to an in-place load with no copy; there is no eviction
//! (a segment is immutable, so a cached bundle is never stale).
//!
//! ## Manifest-addressed key
//!
//! The cache key is the sha256 of the segment's `.manifest.json` **bytes**. The
//! manifest carries a write-time `created_at`, so the key is *write*-addressed,
//! not a pure content-address — but invalidation is still sound: a segment is
//! immutable, so its manifest bytes are stable, so every load of it hashes the
//! same key and hits the same cached directory; a rebuilt segment writes a fresh
//! `created_at`, hence a new key and a correct re-fetch. (This is why the cache
//! keys on the manifest bytes directly rather than on any producing-descriptor
//! definition hash: a segment carries no producing descriptor, and the manifest
//! bytes are the artifact that is actually on disk.)
//!
//! ## Precision-derived sibling set
//!
//! Which siblings a segment carries is a function of its storage precision, read
//! from the manifest's `scalar_kind`:
//!
//! - every precision:      `.usearch` / `.rowmap` / `.manifest.json` (3)
//! - `F16` / `Int8`:       `+ .rawf32` (4)
//! - `Binary`:             `+ .rawf32` `+ .threshold` (5)
//!
//! The set is derived, not a fixed list, so a common `Int8` / `F16` segment does
//! not `GET` a nonexistent `.threshold` (a 404), and a `Binary` segment does not
//! miss the τ companion its load requires.

use std::path::PathBuf;

use bytes::Bytes;

use crate::config::{AnnIndexConfig, StoragePrecision};
use crate::error::{JammiError, Result};
use crate::index::sidecar::{
    SidecarIndex, RESCORE_COMPANION_EXTENSION, THRESHOLD_COMPANION_EXTENSION,
};
use crate::storage::sidecar_layout::local_base_path;
use crate::storage::{sha256_hex, JammiObjectStore, Scheme, StorageRegistry, StorageUrl};

/// The local file stem every cached bundle is written under (`segment.usearch`,
/// `segment.rowmap`, …). Fixed, because the cache directory is already keyed by
/// the manifest hash — the stem inside it carries no further identity.
const CACHE_STEM: &str = "segment";

/// The manifest sibling extension — fetched first (it is the cache key) and
/// reused verbatim when materialising the bundle.
const MANIFEST_EXT: &str = "manifest.json";

/// The just-scalar-kind view of a segment manifest, decoded to derive the
/// precision-specific sibling set before the bundle is fetched. Only
/// `scalar_kind` is read; a manifest missing it is a hard `serde` error, the
/// same strictness [`SidecarIndex::load`] applies to the full header.
#[derive(serde::Deserialize)]
struct ScalarKindHeader {
    scalar_kind: StoragePrecision,
}

/// The sidecar siblings a segment at `precision` carries, in a stable order —
/// the precision-derived (not fixed) set the cache downloads.
fn segment_siblings(precision: StoragePrecision) -> Vec<&'static str> {
    let mut exts = vec!["usearch", "rowmap", MANIFEST_EXT];
    if precision.needs_rescore() {
        exts.push(RESCORE_COMPANION_EXTENSION);
    }
    if precision == StoragePrecision::Binary {
        exts.push(THRESHOLD_COMPANION_EXTENSION);
    }
    exts
}

/// Materialises remote ANN index segments into a content-addressed local cache,
/// over the same [`StorageRegistry`] the result store uses. Constructed once at
/// [`crate::store::ResultStore`] construction from the session artifact
/// directory.
pub struct SegmentIndexCache {
    registry: StorageRegistry,
    cache_root: PathBuf,
}

impl SegmentIndexCache {
    /// Construct a segment cache rooted at `cache_root`, sharing `registry` with
    /// the engine session so cloud credentials are registered once. The
    /// directory is created so the first fetch's atomic rename has a parent.
    pub fn new(registry: StorageRegistry, cache_root: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&cache_root)?;
        Ok(Self {
            registry,
            cache_root,
        })
    }

    /// Load the segment bundle at `index_url` into a [`SidecarIndex`], applying
    /// the query-time HNSW knob from `ann` and strict-checking
    /// `expected_precision` (the catalog row's persisted precision) against the
    /// bundle's own stamped precision.
    ///
    /// A `file://` bundle loads in place. A remote bundle is fetched into the
    /// content-addressed cache (a hit skips every download) and loaded from
    /// there.
    pub async fn load_segment(
        &self,
        index_url: &StorageUrl,
        ann: &AnnIndexConfig,
        expected_precision: StoragePrecision,
    ) -> Result<SidecarIndex> {
        let driver = self.registry.driver_for(index_url, None)?;
        let handle = JammiObjectStore::new(driver, index_url.clone());

        if handle.scheme() == Scheme::File {
            // The bundle already lives on a local path USearch can open — no
            // cache entry, no copy.
            let base = local_base_path(&handle)?;
            return SidecarIndex::load(&base, ann, expected_precision);
        }

        // Remote: address by the manifest bytes. Fetch it first — it is both the
        // cache key and one of the siblings the bundle needs.
        let manifest_remote = handle.sibling_path(MANIFEST_EXT)?;
        let manifest_bytes: Bytes = handle.get_bytes(&manifest_remote).await?;
        let key = sha256_hex(&manifest_bytes);
        let cache_dir = self.cache_root.join(&key);
        let base = cache_dir.join(CACHE_STEM);

        // Cache hit: a prior load's atomic rename published this dir, so it is
        // complete by construction.
        if cache_dir.is_dir() {
            return SidecarIndex::load(&base, ann, expected_precision);
        }

        let header: ScalarKindHeader = serde_json::from_slice(&manifest_bytes)?;
        let siblings = segment_siblings(header.scalar_kind);

        // Download into a sibling tempdir, reusing the manifest bytes already in
        // hand, then atomically rename into the cache.
        let tmp = tempfile::tempdir_in(&self.cache_root)?;
        let tmp_stem = tmp.path().join(CACHE_STEM);
        for ext in siblings {
            let bytes: Bytes = if ext == MANIFEST_EXT {
                manifest_bytes.clone()
            } else {
                let remote = handle.sibling_path(ext)?;
                handle.get_bytes(&remote).await?
            };
            std::fs::write(tmp_stem.with_extension(ext), &bytes)?;
        }

        // Atomic publish. A concurrent load may have won the race and renamed an
        // identical (same-key, therefore byte-identical) bundle into place — a
        // benign loss; keep the existing dir.
        match std::fs::rename(tmp.path(), &cache_dir) {
            Ok(()) => {}
            Err(_) if cache_dir.is_dir() => {}
            Err(e) => return Err(JammiError::Io(e)),
        }

        SidecarIndex::load(&base, ann, expected_precision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::sidecar::SidecarIndex;
    use crate::index::VectorIndex;
    use crate::storage::sidecar_layout::save_sidecar;

    fn build_small_index(precision: StoragePrecision) -> SidecarIndex {
        let mut idx = SidecarIndex::new(4, &AnnIndexConfig::default(), precision).unwrap();
        idx.add("row-a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add("row-b", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        idx.add("row-c", &[0.0, 0.0, 1.0, 0.0]).unwrap();
        idx.build().unwrap();
        idx
    }

    /// A cache and the registry it shares — the same registry must back both the
    /// cache and any save handle, since each `StorageRegistry`'s `memory://`
    /// store is independent.
    fn cache() -> (tempfile::TempDir, StorageRegistry, SegmentIndexCache) {
        let dir = tempfile::tempdir().unwrap();
        let registry = StorageRegistry::new();
        let cache =
            SegmentIndexCache::new(registry.clone(), dir.path().join("index_cache")).unwrap();
        (dir, registry, cache)
    }

    #[test]
    fn siblings_are_precision_derived_three_tiers() {
        assert_eq!(
            segment_siblings(StoragePrecision::F32),
            ["usearch", "rowmap", "manifest.json"]
        );
        assert_eq!(
            segment_siblings(StoragePrecision::Int8),
            ["usearch", "rowmap", "manifest.json", "rawf32"]
        );
        assert_eq!(
            segment_siblings(StoragePrecision::F16),
            ["usearch", "rowmap", "manifest.json", "rawf32"]
        );
        assert_eq!(
            segment_siblings(StoragePrecision::Binary),
            ["usearch", "rowmap", "manifest.json", "rawf32", "threshold"]
        );
    }

    #[tokio::test]
    async fn remote_hit_skips_download_after_first_fetch() {
        let (_dir, registry, cache) = cache();
        let url = StorageUrl::memory("segs/tbl__seg0.idx");
        let driver = registry.driver_for(&url, None).unwrap();
        let handle = JammiObjectStore::new(driver, url.clone());

        save_sidecar(&handle, &build_small_index(StoragePrecision::F32))
            .await
            .unwrap();

        // First load populates the content-addressed cache dir.
        let first = cache
            .load_segment(&url, &AnnIndexConfig::default(), StoragePrecision::F32)
            .await
            .unwrap();
        assert_eq!(first.len(), 3);

        // Delete the remote graph/rowmap, keep the manifest (the key). A second
        // load that re-downloaded would fail; a cache hit serves from disk.
        handle
            .delete_if_exists(&handle.sibling_path("usearch").unwrap())
            .await
            .unwrap();
        handle
            .delete_if_exists(&handle.sibling_path("rowmap").unwrap())
            .await
            .unwrap();
        let second = cache
            .load_segment(&url, &AnnIndexConfig::default(), StoragePrecision::F32)
            .await
            .unwrap();
        let hits = second.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(hits.first().map(|(id, _)| id.as_str()), Some("row-a"));
    }

    #[tokio::test]
    async fn changed_manifest_refetches_under_a_new_key() {
        let (_dir, registry, cache) = cache();
        let url = StorageUrl::memory("segs/tbl__seg1.idx");
        let driver = registry.driver_for(&url, None).unwrap();
        let handle = JammiObjectStore::new(driver, url.clone());

        save_sidecar(&handle, &build_small_index(StoragePrecision::F32))
            .await
            .unwrap();
        cache
            .load_segment(&url, &AnnIndexConfig::default(), StoragePrecision::F32)
            .await
            .unwrap();

        // Rebuild with a different row set → a new manifest (fresh created_at +
        // count) → a new key → a correct re-fetch (not the stale cached dir).
        let mut rebuilt =
            SidecarIndex::new(4, &AnnIndexConfig::default(), StoragePrecision::F32).unwrap();
        rebuilt.add("row-z", &[0.0, 0.0, 0.0, 1.0]).unwrap();
        rebuilt.build().unwrap();
        save_sidecar(&handle, &rebuilt).await.unwrap();

        let reloaded = cache
            .load_segment(&url, &AnnIndexConfig::default(), StoragePrecision::F32)
            .await
            .unwrap();
        let hits = reloaded.search(&[0.0, 0.0, 0.0, 1.0], 1).unwrap();
        assert_eq!(hits.first().map(|(id, _)| id.as_str()), Some("row-z"));
    }

    #[tokio::test]
    async fn file_scheme_loads_in_place_without_a_cache_entry() {
        let root = tempfile::tempdir().unwrap();
        let cache_dir = tempfile::tempdir().unwrap();
        let cache =
            SegmentIndexCache::new(StorageRegistry::new(), cache_dir.path().to_path_buf()).unwrap();
        let registry = StorageRegistry::new();
        let url = StorageUrl::parse(root.path().join("tbl__seg0.idx").to_str().unwrap()).unwrap();
        let driver = registry.driver_for(&url, None).unwrap();
        let handle = JammiObjectStore::new(driver, url.clone());

        save_sidecar(&handle, &build_small_index(StoragePrecision::F32))
            .await
            .unwrap();
        let loaded = cache
            .load_segment(&url, &AnnIndexConfig::default(), StoragePrecision::F32)
            .await
            .unwrap();
        assert_eq!(loaded.len(), 3);
        // Nothing was written under the cache root — the file:// bundle loaded in
        // place.
        assert!(std::fs::read_dir(cache_dir.path())
            .unwrap()
            .next()
            .is_none());
    }
}
