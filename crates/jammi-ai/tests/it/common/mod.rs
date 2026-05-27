pub use jammi_test_utils::*;

use jammi_numerics::retrieval::AggregateMetrics;

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
