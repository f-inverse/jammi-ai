pub use jammi_test_utils::*;

use std::sync::Arc;

use jammi_db::store::ArtifactStore;
use jammi_numerics::retrieval::AggregateMetrics;

/// The ANN index-segment bundle base URLs of `table_name`, in segment order —
/// the load-side handle for tests that inspect a table's on-disk sidecar bundle
/// now that a table's index is a set of segments rather than one `index_path`.
pub async fn segment_index_urls(
    session: &jammi_ai::session::InferenceSession,
    table_name: &str,
) -> Vec<String> {
    session
        .catalog()
        .list_index_segments(table_name)
        .await
        .unwrap()
        .into_iter()
        .map(|s| s.index_path)
        .collect()
}

/// The single segment's bundle base URL for a freshly-embedded table (one embed
/// pass writes exactly one segment). Panics unless there is exactly one.
pub async fn segment0_index_url(
    session: &jammi_ai::session::InferenceSession,
    table_name: &str,
) -> String {
    let mut urls = segment_index_urls(session, table_name).await;
    assert_eq!(
        urls.len(),
        1,
        "expected exactly one index segment for '{table_name}'"
    );
    urls.pop().unwrap()
}

/// Build an [`ArtifactStore`] rooted at a hermetic `memory://` URL with a fresh
/// local fetch cache, for resolver-level unit tests that construct a
/// `ModelResolver` directly (rather than through a full `InferenceSession`). The
/// cache dir leaks for the test binary's lifetime — acceptable in a test.
pub fn test_artifact_store() -> Arc<ArtifactStore> {
    let cache = tempfile::tempdir().unwrap().keep();
    Arc::new(
        ArtifactStore::with_root(
            jammi_db::storage::StorageUrl::memory("test-artifacts"),
            jammi_db::storage::StorageRegistry::new(),
            cache,
        )
        .unwrap(),
    )
}

/// Return the four aggregate retrieval metrics paired with their wire-format
/// snake_case names. Tests that need to iterate over every metric (range
/// checks, baseline-vs-candidate diffs, determinism comparisons) consume this
/// instead of indexing fields by string — the array literal makes adding a
/// new metric to [`AggregateMetrics`] a single-file edit that the compiler
/// guides exhaustively.
pub fn aggregate_named_metrics(agg: &AggregateMetrics) -> [(&'static str, f64); 4] {
    [
        ("recall_at_k", agg.recall_at_k),
        ("precision_at_k", agg.precision_at_k),
        ("mrr", agg.mrr),
        ("ndcg", agg.ndcg),
    ]
}
