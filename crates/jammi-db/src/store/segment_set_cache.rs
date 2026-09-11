//! Per-process cache of loaded segment sets, keyed `(table_name, version)`.
//!
//! A version's SEMANTIC content is immutable once ready (its manifest is
//! written before the publish CAS and its number is never reused), so a set
//! loaded for `(table, Some(N))` is sound for the process lifetime; a
//! never-refreshed table's base set is frozen too (a base segment insert
//! requires the table `building`), so it caches under `(table, None)`.
//! Eviction is by table: on `bind_result_table`, on publish, on table delete.
//! Version manifests are cached beside the sets (small JSON, immutable).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use crate::index::segment::SegmentedIndex;
use crate::store::deletes::DeletionMask;
use crate::store::version::VersionManifest;

/// A table version's loaded, masked segment set.
pub struct LoadedSegmentSet {
    pub index: Arc<SegmentedIndex>,
    pub mask: Arc<DeletionMask>,
}

/// `(table_name, version)` — `None` is a never-refreshed table's base set.
type SetKey = (String, Option<i64>);

#[derive(Default)]
pub struct SegmentSetCache {
    sets: Mutex<HashMap<SetKey, Arc<LoadedSegmentSet>>>,
    manifests: Mutex<HashMap<(String, i64), Arc<VersionManifest>>>,
}

impl std::fmt::Debug for SegmentSetCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SegmentSetCache").finish()
    }
}

impl SegmentSetCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, table: &str, version: Option<i64>) -> Option<Arc<LoadedSegmentSet>> {
        self.sets
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&(table.to_string(), version))
            .cloned()
    }

    pub fn insert(&self, table: &str, version: Option<i64>, set: Arc<LoadedSegmentSet>) {
        self.sets
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert((table.to_string(), version), set);
    }

    pub fn get_manifest(&self, table: &str, version: i64) -> Option<Arc<VersionManifest>> {
        self.manifests
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&(table.to_string(), version))
            .cloned()
    }

    pub fn insert_manifest(&self, table: &str, version: i64, manifest: Arc<VersionManifest>) {
        self.manifests
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert((table.to_string(), version), manifest);
    }

    /// Drop every entry (sets and manifests) of `table`.
    pub fn evict_table(&self, table: &str) {
        self.sets
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .retain(|(t, _), _| t != table);
        self.manifests
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .retain(|(t, _), _| t != table);
    }
}
