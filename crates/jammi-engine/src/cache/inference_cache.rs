use arrow::array::RecordBatch;
use moka::future::Cache;

use crate::model_task::ModelTask;

/// Cache key: (model_id, task, content_hash).
/// Model version is embedded in model_id (e.g. "model-a::1"), so stale
/// entries naturally miss when the model changes.
#[derive(Hash, Eq, PartialEq, Clone)]
struct InferenceCacheKey {
    model_id: String,
    task: ModelTask,
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

    pub async fn get(
        &self,
        model_id: &str,
        task: ModelTask,
        content: &str,
    ) -> Option<Vec<RecordBatch>> {
        let key = InferenceCacheKey {
            model_id: model_id.into(),
            task,
            content_hash: seahash::hash(content.as_bytes()),
        };
        self.entries.get(&key).await
    }

    pub async fn put(
        &self,
        model_id: &str,
        task: ModelTask,
        content: &str,
        batches: Vec<RecordBatch>,
    ) {
        let key = InferenceCacheKey {
            model_id: model_id.into(),
            task,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Two cache keys built from the same `ModelTask` variant hit; keys built
    /// from different variants miss. Catches regressions where the key
    /// migrated to a string would silently equate "text_embedding" entries
    /// across model versions or vice versa.
    #[tokio::test]
    async fn typed_task_key_equality_is_by_variant() {
        let cache = InferenceCache::new(16);
        cache
            .put("m::1", ModelTask::TextEmbedding, "hello", vec![])
            .await;

        assert!(
            cache
                .get("m::1", ModelTask::TextEmbedding, "hello")
                .await
                .is_some(),
            "same (model, task, content) should hit"
        );
        assert!(
            cache
                .get("m::1", ModelTask::Classification, "hello")
                .await
                .is_none(),
            "different task variant should miss even with identical model+content"
        );
        assert!(
            cache
                .get("m::2", ModelTask::TextEmbedding, "hello")
                .await
                .is_none(),
            "different model version should miss"
        );
    }
}
