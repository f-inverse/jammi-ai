use jammi_engine::cache::ann_cache::AnnCache;
use jammi_engine::cache::inference_cache::InferenceCache;
use jammi_engine::error::JammiError;
use jammi_engine::source::retry::{retry_with_backoff, RetryConfig};

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

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
            let _ = cache.get("src", "tbl", &query, 10).await;
            if i % 5 == 0 {
                cache.invalidate_source("src").unwrap();
            }
        }));
    }
    for h in handles {
        h.await.unwrap();
    }
}

// ─── Inference cache: task+model versioning ──────────────────────────────────

#[tokio::test]
async fn inference_cache_hit_and_model_version_miss() {
    let cache = InferenceCache::new(100);

    cache
        .put("model-a::1", "embedding", "hello world", vec![])
        .await;

    // Hit with same key
    assert!(
        cache
            .get("model-a::1", "embedding", "hello world")
            .await
            .is_some(),
        "Same model+task+content should hit"
    );

    // Miss with different model version (simulates model update)
    assert!(
        cache
            .get("model-a::2", "embedding", "hello world")
            .await
            .is_none(),
        "Different model version should miss"
    );

    // Miss with different task
    assert!(
        cache
            .get("model-a::1", "classification", "hello world")
            .await
            .is_none(),
        "Different task should miss"
    );
}

// ─── Retry: backoff behavior, attempt count, exhaustion, timing ──────────────
//
// One test exercises: partial failure → success with exact attempt count,
// full exhaustion with source-naming error format, and exponential timing.
// Each retry scenario needs its own config/counter but no expensive setup.

#[tokio::test]
async fn retry_backoff_count_exhaustion_and_timing() {
    // --- Partial failure: 2 failures then success = 3 total attempts ---
    let attempt_count = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&attempt_count);

    let config = RetryConfig {
        max_attempts: 4,
        initial_backoff: Duration::from_millis(10),
        max_backoff: Duration::from_millis(100),
        backoff_multiplier: 2.0,
        jitter: false,
    };

    let result = retry_with_backoff("test-source", &config, || {
        let counter = Arc::clone(&counter);
        async move {
            let attempt = counter.fetch_add(1, Ordering::SeqCst);
            if attempt < 2 {
                Err(JammiError::Source {
                    source_id: "test-source".into(),
                    message: "connection refused".into(),
                })
            } else {
                Ok(42)
            }
        }
    })
    .await;
    assert_eq!(result.unwrap(), 42);
    assert_eq!(
        attempt_count.load(Ordering::SeqCst),
        3,
        "Should have tried 3 times (2 failures + 1 success)"
    );

    // --- Exhaustion: all attempts fail, error names the source ---
    let exhaust_config = RetryConfig {
        max_attempts: 3,
        initial_backoff: Duration::from_millis(10),
        max_backoff: Duration::from_millis(50),
        backoff_multiplier: 2.0,
        jitter: false,
    };

    let result: Result<(), _> = retry_with_backoff("flaky_postgres", &exhaust_config, || async {
        Err::<(), _>(JammiError::Source {
            source_id: "flaky_postgres".into(),
            message: "connection refused".into(),
        })
    })
    .await;
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("flaky_postgres"),
        "Error should name source: {msg}"
    );
    assert!(
        msg.contains("3 attempts"),
        "Error should mention attempt count: {msg}"
    );

    // --- Timing: exponential delays (50 + 100 + 200 = 350ms minimum) ---
    let timing_config = RetryConfig {
        max_attempts: 4,
        initial_backoff: Duration::from_millis(50),
        max_backoff: Duration::from_secs(1),
        backoff_multiplier: 2.0,
        jitter: false,
    };

    let start = Instant::now();
    let _: Result<(), _> = retry_with_backoff("timer-test", &timing_config, || async {
        Err::<(), _>(JammiError::Source {
            source_id: "timer-test".into(),
            message: "timeout".into(),
        })
    })
    .await;
    assert!(
        start.elapsed() >= Duration::from_millis(300),
        "Exponential backoff should take >=300ms for 3 retries, took {:?}",
        start.elapsed(),
    );
}

// ─── RetryConfig validation ──────────────────────────────────────────────────

#[test]
fn retry_config_validation() {
    assert!(RetryConfig::default().validate().is_ok());

    let bad_attempts = RetryConfig {
        max_attempts: 0,
        ..Default::default()
    };
    assert!(bad_attempts.validate().is_err());

    let bad_multiplier = RetryConfig {
        backoff_multiplier: 0.5,
        ..Default::default()
    };
    assert!(bad_multiplier.validate().is_err());
}
