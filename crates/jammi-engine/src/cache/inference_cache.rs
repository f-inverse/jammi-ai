use arrow::array::RecordBatch;
use moka::future::Cache;

/// Cache key: (model_id, task, content_hash).
/// Model version is embedded in model_id (e.g. "model-a::1"), so stale
/// entries naturally miss when the model changes.
#[derive(Hash, Eq, PartialEq, Clone)]
struct InferenceCacheKey {
    model_id: String,
    task: String,
    content_hash: u64,
}

/// Inference result cache backed by moka.
pub struct InferenceCache {
    entries: Cache<InferenceCacheKey, Vec<RecordBatch>>,
}

impl InferenceCache {
    pub fn new(max_entries: u64) -> Self {
        Self {
            entries: Cache::builder().max_capacity(max_entries).build(),
        }
    }

    pub async fn get(&self, model_id: &str, task: &str, content: &str) -> Option<Vec<RecordBatch>> {
        let key = InferenceCacheKey {
            model_id: model_id.into(),
            task: task.into(),
            content_hash: seahash::hash(content.as_bytes()),
        };
        self.entries.get(&key).await
    }

    pub async fn put(&self, model_id: &str, task: &str, content: &str, batches: Vec<RecordBatch>) {
        let key = InferenceCacheKey {
            model_id: model_id.into(),
            task: task.into(),
            content_hash: seahash::hash(content.as_bytes()),
        };
        self.entries.insert(key, batches).await;
    }

    pub fn stats(&self) -> super::ann_cache::CacheStats {
        super::ann_cache::CacheStats {
            entries: self.entries.entry_count(),
        }
    }
}
