pub use jammi_test_utils::*;

use std::sync::Arc;

use jammi_db::store::ArtifactStore;
use jammi_numerics::retrieval::AggregateMetrics;

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
