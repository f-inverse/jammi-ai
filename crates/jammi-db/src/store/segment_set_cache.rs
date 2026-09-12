//! Per-process cache of loaded segment sets, keyed `(table_name, version)`.
//!
//! A version's SEMANTIC content is immutable once ready (its manifest is
//! written before the publish CAS and its number is never reused), so a set
//! loaded for `(table, Some(N))` is sound for the process lifetime; a
//! never-refreshed table's base set is frozen too (a base segment insert
//! requires the table `building`), so it caches under `(table, None)`.
//! Eviction is by table: on `bind_result_table`, on publish, on table delete.
//! Version manifests are cached beside the sets (small JSON, immutable).
//!
//! The inferred base [`arrow::datatypes::SchemaRef`] and loaded
//! [`DeletionMask`] `crate::store::ResultStore::build_masked_provider` needs
//! per call are cached here too, under the same `(table, version)` key: both
//! are per-version-IMMUTABLE IO products (a fragment's schema and a
//! version's deletion mask never change once the version is `ready`), so
//! re-reading them on every `build_masked_provider` call — an object-store
//! list plus a Parquet footer read for the schema, an object read for the
//! mask — buys nothing but cost. This is the M3 cost fix; it is never a
//! substitute for [`crate::store::PinnedSource`]'s correctness fix, which
//! closes a different bug (a straddled resolve, not a cache miss). The
//! `Arc<dyn datafusion::datasource::TableProvider>` itself is deliberately
//! NOT cached here: `build_result_table_provider` registers the fragment
//! URL's object store on the `SessionContext` it is passed, so a provider
//! built for one session and reused under another could scan without that
//! registration ever having run.

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
    /// The inferred base schema of a masked provider's fragments, keyed
    /// `(table, version)`. See the module doc for why this is safe to cache
    /// forever (until [`Self::evict_table`]) rather than per-call.
    masked_schemas: Mutex<HashMap<(String, i64), arrow::datatypes::SchemaRef>>,
    /// A version's loaded [`DeletionMask`], keyed `(table, version)`. Same
    /// immutability argument as `masked_schemas`.
    masked_masks: Mutex<HashMap<(String, i64), Arc<DeletionMask>>>,
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

    /// The cached inferred base schema for `(table, version)`'s masked
    /// provider, if this process has already built one.
    pub fn get_masked_schema(
        &self,
        table: &str,
        version: i64,
    ) -> Option<arrow::datatypes::SchemaRef> {
        self.masked_schemas
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&(table.to_string(), version))
            .cloned()
    }

    pub fn insert_masked_schema(
        &self,
        table: &str,
        version: i64,
        schema: arrow::datatypes::SchemaRef,
    ) {
        self.masked_schemas
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert((table.to_string(), version), schema);
    }

    /// The cached [`DeletionMask`] for `(table, version)`, if this process
    /// has already loaded one.
    pub fn get_masked_mask(&self, table: &str, version: i64) -> Option<Arc<DeletionMask>> {
        self.masked_masks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&(table.to_string(), version))
            .cloned()
    }

    pub fn insert_masked_mask(&self, table: &str, version: i64, mask: Arc<DeletionMask>) {
        self.masked_masks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert((table.to_string(), version), mask);
    }

    /// Drop every entry (sets, manifests, and the masked-provider schema/mask
    /// memo) of `table`.
    pub fn evict_table(&self, table: &str) {
        self.sets
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .retain(|(t, _), _| t != table);
        self.manifests
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .retain(|(t, _), _| t != table);
        self.masked_schemas
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .retain(|(t, _), _| t != table);
        self.masked_masks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .retain(|(t, _), _| t != table);
    }
}
