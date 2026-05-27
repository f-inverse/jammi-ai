//! Typed reports for the eval API.
//!
//! Every eval entry point (`eval_embeddings`, `eval_inference`, `eval_compare`)
//! returns one of these structs. The shape carries both an aggregate (mean
//! across queries / records) and per-query / per-record arrays. The per-query
//! data is what sample-based statistical rules (Welch's t, Mann-Whitney U)
//! consume at gate time; the aggregate is what the catalog persists for
//! historical reporting.

use jammi_numerics::classification::ClassificationResult;
use jammi_numerics::ner::{Entity, NerMetrics};
use jammi_numerics::retrieval::{AggregateMetrics, QueryMetrics};
use serde::{Deserialize, Serialize};

/// Result of one `eval_embeddings` invocation. Carries the mean over all
/// queries (`aggregate.recall_at_k`, etc.) and the per-query records that
/// sample-based statistical rules consume at gate time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingEvalReport {
    /// Mean metrics across all queries.
    pub aggregate: AggregateMetrics,
    /// One record per query, in golden-set order.
    pub per_query: Vec<PerQueryRecord>,
}

/// A single query's metrics plus its `query_id` join key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerQueryRecord {
    /// `query_id` from the golden source (the join key).
    pub query_id: String,
    pub metrics: QueryMetrics,
}

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
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MetricDelta {
    pub absolute: f64,
    pub relative: f64,
}
