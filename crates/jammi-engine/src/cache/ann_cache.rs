use arrow::array::RecordBatch;
use moka::future::Cache;

/// Cache key: (source_id, table_name, query_hash, k).
/// query_hash is a fast hash of the f32 query vector via seahash.
#[derive(Hash, Eq, PartialEq, Clone)]
struct AnnCacheKey {
    source_id: String,
    table_name: String,
    query_hash: u64,
    k: usize,
}

impl AnnCacheKey {
    fn new(source_id: &str, table_name: &str, query: &[f32], k: usize) -> Self {
        let bytes: &[u8] = bytemuck::cast_slice(query);
        let query_hash = seahash::hash(bytes);
        Self {
            source_id: source_id.into(),
            table_name: table_name.into(),
            query_hash,
            k,
        }
    }
}

/// ANN search result cache backed by moka.
pub struct AnnCache {
    entries: Cache<AnnCacheKey, Vec<RecordBatch>>,
}

impl AnnCache {
    pub fn new(max_entries: u64) -> Self {
        Self {
            entries: Cache::builder()
                .max_capacity(max_entries)
                .support_invalidation_closures()
                .build(),
        }
    }

    pub async fn get(
        &self,
        source_id: &str,
        table_name: &str,
        query: &[f32],
        k: usize,
    ) -> Option<Vec<RecordBatch>> {
        let key = AnnCacheKey::new(source_id, table_name, query, k);
        self.entries.get(&key).await
    }

    pub async fn put(
        &self,
        source_id: &str,
        table_name: &str,
        query: &[f32],
        k: usize,
        batches: Vec<RecordBatch>,
    ) {
        let key = AnnCacheKey::new(source_id, table_name, query, k);
        self.entries.insert(key, batches).await;
    }

    /// Invalidate all entries for a source. Called when new embeddings are generated.
    pub fn invalidate_source(&self, source_id: &str) {
        let sid = source_id.to_string();
        self.entries
            .invalidate_entries_if(move |k, _v| k.source_id == sid)
            .expect("invalidate_entries_if should not fail");
    }

    /// Force moka to process pending invalidation predicates.
    /// Normally moka processes these lazily; call this in tests after
    /// `invalidate_source` to ensure entries are evicted immediately.
    pub async fn run_pending_tasks(&self) {
        self.entries.run_pending_tasks().await;
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.entries.entry_count(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: u64,
}
