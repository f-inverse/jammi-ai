//! Typed reports for the eval API.
//!
//! Every eval entry point (`eval_embeddings`, `eval_inference`, `eval_compare`)
//! returns one of these structs. The shape carries both an aggregate (mean
//! across queries / records) and per-query / per-record arrays. The per-query
//! data is what sample-based statistical rules (Welch's t, Mann-Whitney U)
//! consume at gate time; the aggregate is what the catalog persists for
//! historical reporting.

use std::collections::BTreeMap;

use jammi_numerics::classification::ClassificationResult;
use jammi_numerics::ner::{Entity, NerMetrics};
use jammi_numerics::retrieval::{AggregateMetrics, QueryMetrics};
// The bootstrap / rank-sum kernels are pulled in only where significance is
// *computed* — the local engine's `eval_compare`. The significance *types*
// below stay transport-neutral so the thin wire client round-trips them.
#[cfg(feature = "local")]
use jammi_numerics::stats::{bootstrap_ci, mann_whitney_u, Interval};
use serde::{Deserialize, Serialize};

/// Result of one `eval_embeddings` invocation. Carries the mean over all
/// queries (`aggregate.recall_at_k`, etc.) and the per-query records that
/// sample-based statistical rules consume at gate time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingEvalReport {
    /// The catalog id of the recorded run. The same id keys the persisted
    /// per-query rows (`_jammi_eval_per_query`), so a caller can read them
    /// back via `eval_per_query(eval_run_id)` (spec J9). Every embedding eval
    /// records a run, so this is always a valid, re-readable run id.
    #[serde(default)]
    pub eval_run_id: String,
    /// Mean metrics across all queries.
    pub aggregate: AggregateMetrics,
    /// One record per query, in golden-set order.
    pub per_query: Vec<PerQueryRecord>,
}

/// A single query's metrics plus its `query_id` join key.
///
/// `metrics` (the historical single-cutoff [`QueryMetrics`]) is unchanged so
/// existing consumers keep reading `metrics.recall` / `metrics.mrr` / etc.
/// The J9 additions — `recall_at_ks` (Recall@{1,3,5,10}), `distance` (the
/// top-1 result's score), and opaque `cohorts` tags — are additive fields that
/// older callers can ignore. These are the fields persisted to
/// `_jammi_eval_per_query`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerQueryRecord {
    /// `query_id` from the golden source (the join key).
    pub query_id: String,
    pub metrics: QueryMetrics,
    /// Recall at multiple cutoffs as `(k, recall@k)` pairs, in ascending k.
    #[serde(default)]
    pub recall_at_ks: Vec<(usize, f64)>,
    /// The top-1 retrieved result's score (distance / similarity), or `0.0`
    /// when the query returned no results.
    #[serde(default)]
    pub distance: f64,
    /// Opaque per-query cohort tags supplied at eval time (`{}` when none).
    /// The substrate never interprets these — declaration/validation is a
    /// downstream consumer's concern.
    #[serde(default)]
    pub cohorts: BTreeMap<String, String>,
}

/// The cutoffs at which per-query Recall@k is recorded and persisted (spec J9).
pub const PER_QUERY_RECALL_KS: [usize; 4] = [1, 3, 5, 10];

/// Result of one `eval_inference` invocation. The aggregate variant matches
/// the task kind.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceEvalReport {
    pub aggregate: InferenceAggregate,
    pub per_record: Vec<PerRecordPrediction>,
}

/// Aggregate metrics for `eval_inference`, tagged by task kind.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "task", rename_all = "snake_case")]
pub enum InferenceAggregate {
    Classification(ClassificationResult),
    Ner(NerMetrics),
}

/// One predicted / gold prediction pair from `eval_inference`, tagged by
/// task kind so the per-record array carries task-shaped payloads instead
/// of a single string-pair shape.
///
/// Classification carries the predicted/gold label strings; NER carries
/// the predicted/gold entity-span sets so downstream consumers can compute
/// per-record precision/recall without re-decoding the JSON payload the
/// NER inference adapter wrote.
///
/// Wire shape mirrors [`InferenceAggregate`]: a serde-tagged enum with
/// `"task": "classification"` or `"task": "ner"` discriminating the
/// variant, so both wire formats round-trip through the same
/// `serializable_to_pydict` conversion used by every other eval response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "task", rename_all = "snake_case")]
pub enum PerRecordPrediction {
    Classification {
        record_id: String,
        predicted: String,
        gold: String,
    },
    Ner {
        record_id: String,
        predicted: Vec<Entity>,
        gold: Vec<Entity>,
    },
}

/// Result of one `eval_compare` invocation across multiple embedding tables.
/// The first table in `per_table` is the baseline; all subsequent entries
/// carry their `delta` relative to that baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareEvalReport {
    pub per_table: Vec<TableEvalReport>,
}

/// One table's embedding eval plus its delta relative to the baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableEvalReport {
    pub table_name: String,
    pub embedding_eval: EmbeddingEvalReport,
    /// Aggregate-metric deltas relative to the first table in `per_table`.
    /// `None` for the baseline itself.
    pub delta: Option<AggregateDelta>,
}

/// Per-metric deltas (absolute + relative) between two aggregate scores, plus
/// the distribution-free paired significance of each delta.
///
/// `significance` is computed from the two tables' per-query metric arrays
/// (paired by `query_id`), not from the aggregates: it is `None` only when the
/// two runs share no query in common, so there is nothing to pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateDelta {
    pub recall_at_k: MetricDelta,
    pub precision_at_k: MetricDelta,
    pub mrr: MetricDelta,
    pub ndcg: MetricDelta,
    /// Paired significance of each metric delta, or `None` when the baseline
    /// and treatment runs share no `query_id` to pair on.
    #[serde(default)]
    pub significance: Option<DeltaSignificance>,
}

/// Single-metric delta: `absolute = model - baseline`, `relative = absolute /
/// baseline` (or `0.0` when the baseline is zero).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MetricDelta {
    pub absolute: f64,
    pub relative: f64,
}

/// Distribution-free paired significance for each metric in an
/// [`AggregateDelta`], in the same metric layout as the delta itself.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DeltaSignificance {
    pub recall_at_k: MetricSignificance,
    pub precision_at_k: MetricSignificance,
    pub mrr: MetricSignificance,
    pub ndcg: MetricSignificance,
}

/// Distribution-free significance of one metric's delta, computed from the
/// paired per-query values.
///
/// - `ci` is a percentile bootstrap confidence interval on the mean paired
///   difference (`treatment − baseline`); a CI that excludes zero is the
///   resampling analogue of "the delta is significant".
/// - `p_value` is the two-tailed Mann–Whitney U p-value comparing the
///   baseline and treatment per-query distributions — distribution-free and
///   robust to the bounded, tie-heavy shape of retrieval metrics.
///
/// Both are deterministic: the bootstrap runs under [`BOOTSTRAP_SEED`] and a
/// fixed iteration count, so identical inputs yield identical output.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MetricSignificance {
    pub p_value: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
}

/// Bootstrap resamples for the paired confidence interval. Fixed so the CI is
/// a deterministic function of its inputs.
#[cfg(feature = "local")]
pub const BOOTSTRAP_ITERATIONS: usize = 10_000;

/// Two-tailed significance level for the bootstrap CI (a 95% interval).
#[cfg(feature = "local")]
pub const BOOTSTRAP_ALPHA: f64 = 0.05;

/// Pinned RNG seed for the paired bootstrap. The seed is part of the
/// significance contract — it is what makes the CI reproducible across runs.
#[cfg(feature = "local")]
pub const BOOTSTRAP_SEED: u64 = 0x6a616d6d695f7031; // "jammi_p1"

/// One per-query metric value paired between baseline and treatment.
#[cfg(feature = "local")]
struct PairedMetric {
    baseline: f64,
    treatment: f64,
}

/// Pair two tables' per-query records by `query_id`, returning the paired
/// values for one metric (selected by `extract`) in ascending `query_id`
/// order. Queries present in only one table are dropped — significance is a
/// paired test and an unpaired query carries no difference to resample.
#[cfg(feature = "local")]
fn paired_metric<F>(
    baseline: &[PerQueryRecord],
    treatment: &[PerQueryRecord],
    extract: F,
) -> Vec<PairedMetric>
where
    F: Fn(&QueryMetrics) -> f64,
{
    let treatment_by_id: BTreeMap<&str, &QueryMetrics> = treatment
        .iter()
        .map(|r| (r.query_id.as_str(), &r.metrics))
        .collect();
    baseline
        .iter()
        .filter_map(|b| {
            treatment_by_id
                .get(b.query_id.as_str())
                .map(|t| PairedMetric {
                    baseline: extract(&b.metrics),
                    treatment: extract(t),
                })
        })
        .collect()
}

/// Compute the distribution-free significance of one metric's paired deltas:
/// a bootstrap CI on the mean paired difference plus a Mann–Whitney U p-value
/// between the baseline and treatment distributions. Returns `None` when the
/// pairing is empty (no shared `query_id`).
#[cfg(feature = "local")]
fn metric_significance(paired: &[PairedMetric]) -> Option<MetricSignificance> {
    if paired.is_empty() {
        return None;
    }
    let differences: Vec<f64> = paired.iter().map(|p| p.treatment - p.baseline).collect();
    let baseline: Vec<f64> = paired.iter().map(|p| p.baseline).collect();
    let treatment: Vec<f64> = paired.iter().map(|p| p.treatment).collect();

    let Interval { lower, upper } = bootstrap_ci(
        &differences,
        |resample| resample.iter().sum::<f64>() / resample.len() as f64,
        BOOTSTRAP_ITERATIONS,
        BOOTSTRAP_ALPHA,
        BOOTSTRAP_SEED,
    )
    .ok()?;
    let p_value = mann_whitney_u(&baseline, &treatment).ok()?.p_value;

    Some(MetricSignificance {
        p_value,
        ci_lower: lower,
        ci_upper: upper,
    })
}

/// Compute paired significance for all four metrics between a baseline and a
/// treatment run's per-query records. Returns `None` when the two runs share
/// no `query_id` (nothing to pair).
#[cfg(feature = "local")]
pub(crate) fn delta_significance(
    baseline: &[PerQueryRecord],
    treatment: &[PerQueryRecord],
) -> Option<DeltaSignificance> {
    Some(DeltaSignificance {
        recall_at_k: metric_significance(&paired_metric(baseline, treatment, |m| m.recall))?,
        precision_at_k: metric_significance(&paired_metric(baseline, treatment, |m| m.precision))?,
        mrr: metric_significance(&paired_metric(baseline, treatment, |m| m.mrr))?,
        ndcg: metric_significance(&paired_metric(baseline, treatment, |m| m.ndcg))?,
    })
}

#[cfg(all(test, feature = "local"))]
mod significance_tests {
    use super::*;

    /// Build a per-query record carrying only the four scalar metrics under
    /// test; the other fields are irrelevant to significance.
    fn record(query_id: &str, recall: f64, precision: f64, mrr: f64, ndcg: f64) -> PerQueryRecord {
        PerQueryRecord {
            query_id: query_id.to_string(),
            metrics: QueryMetrics {
                recall,
                precision,
                mrr,
                ndcg,
            },
            recall_at_ks: Vec::new(),
            distance: 0.0,
            cohorts: BTreeMap::new(),
        }
    }

    /// A query set where the treatment dominates the baseline on every query
    /// and metric — used to exercise the "significant improvement" path.
    fn improving_pair() -> (Vec<PerQueryRecord>, Vec<PerQueryRecord>) {
        let baseline: Vec<PerQueryRecord> = (0..20)
            .map(|i| record(&format!("q{i}"), 0.2, 0.2, 0.2, 0.2))
            .collect();
        let treatment: Vec<PerQueryRecord> = (0..20)
            .map(|i| record(&format!("q{i}"), 0.8, 0.8, 0.8, 0.8))
            .collect();
        (baseline, treatment)
    }

    #[test]
    fn pairs_only_shared_query_ids() {
        let baseline = vec![
            record("a", 0.1, 0.1, 0.1, 0.1),
            record("b", 0.2, 0.2, 0.2, 0.2),
        ];
        let treatment = vec![
            record("b", 0.5, 0.5, 0.5, 0.5),
            record("c", 0.9, 0.9, 0.9, 0.9),
        ];
        let paired = paired_metric(&baseline, &treatment, |m| m.recall);
        assert_eq!(paired.len(), 1, "only query 'b' is shared");
        assert_eq!(paired[0].baseline, 0.2);
        assert_eq!(paired[0].treatment, 0.5);
    }

    #[test]
    fn no_shared_queries_yields_none() {
        let baseline = vec![record("a", 0.1, 0.1, 0.1, 0.1)];
        let treatment = vec![record("z", 0.9, 0.9, 0.9, 0.9)];
        assert!(delta_significance(&baseline, &treatment).is_none());
    }

    #[test]
    fn deterministic_under_pinned_seed() {
        let (baseline, treatment) = improving_pair();
        let first = delta_significance(&baseline, &treatment).expect("paired");
        let second = delta_significance(&baseline, &treatment).expect("paired");
        // Bit-for-bit identical: the bootstrap is seeded and the U test is
        // closed-form, so repeated calls on the same input never diverge.
        assert_eq!(first.recall_at_k.p_value, second.recall_at_k.p_value);
        assert_eq!(first.recall_at_k.ci_lower, second.recall_at_k.ci_lower);
        assert_eq!(first.recall_at_k.ci_upper, second.recall_at_k.ci_upper);
        assert_eq!(first.ndcg.ci_upper, second.ndcg.ci_upper);
    }

    #[test]
    fn improvement_is_significant() {
        let (baseline, treatment) = improving_pair();
        let sig = delta_significance(&baseline, &treatment).expect("paired");
        // A uniform +0.6 lift on every query: the paired-difference CI sits
        // entirely above zero and the rank-sum p-value is tiny.
        assert!(
            sig.recall_at_k.ci_lower > 0.0,
            "CI lower bound should exclude zero: {}",
            sig.recall_at_k.ci_lower
        );
        assert!(
            sig.recall_at_k.p_value < 0.01,
            "p-value should be small for a clear lift: {}",
            sig.recall_at_k.p_value
        );
    }

    #[test]
    fn identical_runs_ci_brackets_zero() {
        // Baseline == treatment: every paired difference is exactly zero, so
        // the bootstrap CI collapses to [0, 0] and the distributions are
        // indistinguishable (p ≈ 1).
        let baseline: Vec<PerQueryRecord> = (0..20)
            .map(|i| record(&format!("q{i}"), 0.5, 0.5, 0.5, 0.5))
            .collect();
        let treatment = baseline.clone();
        let sig = delta_significance(&baseline, &treatment).expect("paired");
        assert_eq!(sig.mrr.ci_lower, 0.0);
        assert_eq!(sig.mrr.ci_upper, 0.0);
        assert!(
            sig.mrr.p_value > 0.99,
            "identical distributions should be indistinguishable: {}",
            sig.mrr.p_value
        );
    }
}
