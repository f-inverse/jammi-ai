use jammi_db::cache::ann_cache::AnnCache;

use std::sync::Arc;

// ─── ANN cache: hit, miss, invalidation, concurrent access ───────────────────
//
// One cache instance exercises miss → put → hit → key variation → invalidation
// → concurrent stress. All share the same cache, testing the full lifecycle.

#[tokio::test]
async fn ann_cache_lifecycle_and_concurrency() {
    let cache = Arc::new(AnnCache::new(1000));
    let query = vec![0.5_f32; 32];

    // Miss on first lookup
    assert!(
        cache
            .get("patents", "emb_table_1", &query, 10)
            .await
            .is_none(),
        "First lookup should miss"
    );

    // Put then hit
    cache
        .put("patents", "emb_table_1", &query, 10, vec![])
        .await;
    assert!(
        cache
            .get("patents", "emb_table_1", &query, 10)
            .await
            .is_some(),
        "Second lookup should hit"
    );

    // Different k or source → miss (distinct cache keys)
    assert!(
        cache
            .get("patents", "emb_table_1", &query, 5)
            .await
            .is_none(),
        "Different k should miss"
    );
    assert!(
        cache
            .get("other_source", "emb_table_1", &query, 10)
            .await
            .is_none(),
        "Different source should miss"
    );

    // Invalidate source → entries gone
    cache.invalidate_source("patents").unwrap();
    cache.run_pending_tasks().await;
    assert!(
        cache
            .get("patents", "emb_table_1", &query, 10)
            .await
            .is_none(),
        "After invalidation should miss"
    );
    assert_eq!(cache.stats().entries, 0);

    // Concurrent stress: 20 tasks doing mixed get/put/invalidate — no panics
    let mut handles = Vec::new();
    for i in 0..20 {
        let cache = Arc::clone(&cache);
        handles.push(tokio::spawn(async move {
            let query = vec![i as f32; 32];
            cache.put("src", "tbl", &query, 10, vec![]).await;
            cache.get("src", "tbl", &query, 10).await;
            if i % 5 == 0 {
                cache.invalidate_source("src").unwrap();
            }
        }));
    }
    for h in handles {
        h.await.unwrap();
    }
}
