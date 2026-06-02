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
use serde::{Deserialize, Serialize};

/// Result of one `eval_embeddings` invocation. Carries the mean over all
/// queries (`aggregate.recall_at_k`, etc.) and the per-query records that
/// sample-based statistical rules consume at gate time.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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

/// Per-metric deltas (absolute + relative) between two aggregate scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateDelta {
    pub recall_at_k: MetricDelta,
    pub precision_at_k: MetricDelta,
    pub mrr: MetricDelta,
    pub ndcg: MetricDelta,
}

/// Single-metric delta: `absolute = model - baseline`, `relative = absolute /
/// baseline` (or `0.0` when the baseline is zero).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct MetricDelta {
    pub absolute: f64,
    pub relative: f64,
}
