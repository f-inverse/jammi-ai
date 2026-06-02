//! `EvalService` proto↔domain conversions.
//!
//! The response messages mirror the engine's report structs
//! ([`EmbeddingEvalReport`], [`InferenceEvalReport`], [`CompareEvalReport`], the
//! catalog's [`PerQueryEvalRecord`], and the `jammi-numerics` metric structs)
//! field for field, so each encode is an infallible `From`. The request side
//! decodes the eval task ([`EvalTask`]) and rebuilds the cohort-tag map; the
//! substrate never interprets the tags.

use std::collections::{BTreeMap, HashMap};

use jammi_db::catalog::eval_repo::PerQueryEvalRecord;
use jammi_numerics::classification::{ClassMetrics, ClassificationResult};
use jammi_numerics::ner::{Entity, NerMetrics, TypeMetrics};
use jammi_numerics::retrieval::{AggregateMetrics, QueryMetrics};
use tonic::Status;

use crate::eval::report::{
    AggregateDelta, CompareEvalReport, EmbeddingEvalReport, InferenceAggregate,
    InferenceEvalReport, MetricDelta, PerQueryRecord, PerRecordPrediction, TableEvalReport,
};
use crate::eval::EvalTask;
use crate::wire::proto::eval as pb;

/// Map the proto [`EvalTask`] discriminant onto the engine's [`EvalTask`]. An
/// unspecified or unknown task is rejected — a request that names no task is a
/// client error. `EvalTask` is local to this crate, so the decode is a direct
/// `TryFrom<i32>`.
impl TryFrom<i32> for EvalTaskFromWire {
    type Error = Status;

    fn try_from(task: i32) -> Result<Self, Self::Error> {
        match pb::EvalTask::try_from(task) {
            Ok(pb::EvalTask::Classification) => Ok(EvalTaskFromWire(EvalTask::Classification)),
            Ok(pb::EvalTask::Ner) => Ok(EvalTaskFromWire(EvalTask::Ner)),
            Ok(pb::EvalTask::Unspecified) | Err(_) => {
                Err(Status::invalid_argument("task must be specified"))
            }
        }
    }
}

/// Carries the decoded engine [`EvalTask`]. A bare `impl TryFrom<i32> for
/// EvalTask` collides with `core`'s blanket `TryFrom<T> for T` (the compiler
/// cannot prove `EvalTask != i32` across crates), so the conversion lands on
/// this local marker; the handler unwraps `.0`.
pub struct EvalTaskFromWire(pub EvalTask);

/// Encode the engine's [`EvalTask`] onto the wire enum — the inverse of the
/// [`EvalTaskFromWire`] decode, for the [`crate::RemoteSession`] send side.
/// Total: both inference tasks map to a concrete wire variant (the engine type
/// has no unspecified state).
pub fn eval_task_to_proto(task: EvalTask) -> pb::EvalTask {
    match task {
        EvalTask::Classification => pb::EvalTask::Classification,
        EvalTask::Ner => pb::EvalTask::Ner,
    }
}

/// Rebuild the engine's `query_id → {key: value}` cohort map from the proto
/// `map<string, CohortTags>`. The substrate never interprets these tags.
pub fn cohorts_from_proto(
    cohorts: HashMap<String, pb::CohortTags>,
) -> HashMap<String, BTreeMap<String, String>> {
    cohorts
        .into_iter()
        .map(|(query_id, tags)| (query_id, tags.tags.into_iter().collect()))
        .collect()
}

/// Encode the engine's `query_id → {key: value}` cohort map onto the proto
/// `map<string, CohortTags>` — the inverse of [`cohorts_from_proto`], for the
/// [`crate::RemoteSession`] send side. The substrate never interprets the tags.
pub fn cohorts_to_proto(
    cohorts: &HashMap<String, BTreeMap<String, String>>,
) -> HashMap<String, pb::CohortTags> {
    cohorts
        .iter()
        .map(|(query_id, tags)| {
            (
                query_id.clone(),
                pb::CohortTags {
                    tags: tags.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                },
            )
        })
        .collect()
}

// ─── Report → proto ──────────────────────────────────────────────────────────

impl From<&AggregateMetrics> for pb::AggregateMetrics {
    fn from(a: &AggregateMetrics) -> Self {
        pb::AggregateMetrics {
            recall_at_k: a.recall_at_k,
            precision_at_k: a.precision_at_k,
            mrr: a.mrr,
            ndcg: a.ndcg,
        }
    }
}

impl From<&QueryMetrics> for pb::QueryMetrics {
    fn from(m: &QueryMetrics) -> Self {
        pb::QueryMetrics {
            recall: m.recall,
            precision: m.precision,
            mrr: m.mrr,
            ndcg: m.ndcg,
        }
    }
}

impl From<PerQueryRecord> for pb::PerQueryRecord {
    fn from(r: PerQueryRecord) -> Self {
        pb::PerQueryRecord {
            query_id: r.query_id,
            metrics: Some((&r.metrics).into()),
            recall_at_ks: r
                .recall_at_ks
                .into_iter()
                .map(|(k, recall)| pb::RecallAtK {
                    k: k as u32,
                    recall,
                })
                .collect(),
            distance: r.distance,
            cohorts: r.cohorts.into_iter().collect(),
        }
    }
}

impl From<EmbeddingEvalReport> for pb::EmbeddingEvalReport {
    fn from(report: EmbeddingEvalReport) -> Self {
        pb::EmbeddingEvalReport {
            eval_run_id: report.eval_run_id,
            aggregate: Some((&report.aggregate).into()),
            per_query: report.per_query.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<PerQueryEvalRecord> for pb::PerQueryEvalRecord {
    fn from(rec: PerQueryEvalRecord) -> Self {
        pb::PerQueryEvalRecord {
            eval_run_id: rec.eval_run_id,
            query_id: rec.query_id,
            cohorts_json: rec.cohorts_json,
            metrics_json: rec.metrics_json,
        }
    }
}

impl From<&ClassMetrics> for pb::ClassMetrics {
    fn from(m: &ClassMetrics) -> Self {
        pb::ClassMetrics {
            precision: m.precision,
            recall: m.recall,
            f1: m.f1,
        }
    }
}

impl From<&ClassificationResult> for pb::ClassificationResult {
    fn from(c: &ClassificationResult) -> Self {
        pb::ClassificationResult {
            accuracy: c.accuracy,
            f1: c.f1,
            per_class: c
                .per_class
                .iter()
                .map(|(k, v)| (k.clone(), v.into()))
                .collect(),
        }
    }
}

impl From<&TypeMetrics> for pb::TypeMetrics {
    fn from(m: &TypeMetrics) -> Self {
        pb::TypeMetrics {
            precision: m.precision,
            recall: m.recall,
            f1: m.f1,
            support: m.support as u64,
        }
    }
}

impl From<&NerMetrics> for pb::NerMetrics {
    fn from(n: &NerMetrics) -> Self {
        pb::NerMetrics {
            precision: n.precision,
            recall: n.recall,
            f1: n.f1,
            per_type: n
                .per_type
                .iter()
                .map(|(k, v)| (k.clone(), v.into()))
                .collect(),
        }
    }
}

impl From<&Entity> for pb::Entity {
    fn from(e: &Entity) -> Self {
        pb::Entity {
            label: e.label.clone(),
            start: e.start as u64,
            end: e.end as u64,
            text: e.text.clone(),
            confidence: e.confidence,
        }
    }
}

impl From<&InferenceAggregate> for pb::InferenceAggregate {
    fn from(a: &InferenceAggregate) -> Self {
        let aggregate = match a {
            InferenceAggregate::Classification(c) => {
                pb::inference_aggregate::Aggregate::Classification(c.into())
            }
            InferenceAggregate::Ner(n) => pb::inference_aggregate::Aggregate::Ner(n.into()),
        };
        pb::InferenceAggregate {
            aggregate: Some(aggregate),
        }
    }
}

impl From<PerRecordPrediction> for pb::PerRecordPrediction {
    fn from(p: PerRecordPrediction) -> Self {
        use pb::per_record_prediction as wire;
        let prediction = match p {
            PerRecordPrediction::Classification {
                record_id,
                predicted,
                gold,
            } => wire::Prediction::Classification(wire::Classification {
                record_id,
                predicted,
                gold,
            }),
            PerRecordPrediction::Ner {
                record_id,
                predicted,
                gold,
            } => wire::Prediction::Ner(wire::Ner {
                record_id,
                predicted: predicted.iter().map(Into::into).collect(),
                gold: gold.iter().map(Into::into).collect(),
            }),
        };
        pb::PerRecordPrediction {
            prediction: Some(prediction),
        }
    }
}

impl From<InferenceEvalReport> for pb::InferenceEvalReport {
    fn from(report: InferenceEvalReport) -> Self {
        pb::InferenceEvalReport {
            aggregate: Some((&report.aggregate).into()),
            per_record: report.per_record.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<&MetricDelta> for pb::MetricDelta {
    fn from(d: &MetricDelta) -> Self {
        pb::MetricDelta {
            absolute: d.absolute,
            relative: d.relative,
        }
    }
}

impl From<&AggregateDelta> for pb::AggregateDelta {
    fn from(d: &AggregateDelta) -> Self {
        pb::AggregateDelta {
            recall_at_k: Some((&d.recall_at_k).into()),
            precision_at_k: Some((&d.precision_at_k).into()),
            mrr: Some((&d.mrr).into()),
            ndcg: Some((&d.ndcg).into()),
        }
    }
}

impl From<TableEvalReport> for pb::TableEvalReport {
    fn from(t: TableEvalReport) -> Self {
        pb::TableEvalReport {
            table_name: t.table_name,
            embedding_eval: Some(t.embedding_eval.into()),
            delta: t.delta.as_ref().map(Into::into),
        }
    }
}

impl From<CompareEvalReport> for pb::CompareEvalReport {
    fn from(report: CompareEvalReport) -> Self {
        pb::CompareEvalReport {
            per_table: report.per_table.into_iter().map(Into::into).collect(),
        }
    }
}

// ─── proto → report (the RemoteSession decode side) ──────────────────────────
//
// The inverse of the encodes above. A remote client reads a report message off
// the wire and rebuilds the engine report struct so `Session::Remote` returns
// the identical type `Session::Local` does. Each decode mirrors its encode
// field for field; the report structs and the metric structs are local crates,
// so these `From` impls are orphan-rule-clean without a newtype. A `None` on a
// required nested message (a malformed payload) rebuilds the zero value for that
// metric block rather than panicking — decode stays total.

impl From<pb::AggregateMetrics> for AggregateMetrics {
    fn from(a: pb::AggregateMetrics) -> Self {
        AggregateMetrics {
            recall_at_k: a.recall_at_k,
            precision_at_k: a.precision_at_k,
            mrr: a.mrr,
            ndcg: a.ndcg,
        }
    }
}

impl From<pb::QueryMetrics> for QueryMetrics {
    fn from(m: pb::QueryMetrics) -> Self {
        QueryMetrics {
            recall: m.recall,
            precision: m.precision,
            mrr: m.mrr,
            ndcg: m.ndcg,
        }
    }
}

impl From<pb::PerQueryRecord> for PerQueryRecord {
    fn from(r: pb::PerQueryRecord) -> Self {
        PerQueryRecord {
            query_id: r.query_id,
            metrics: r.metrics.map(Into::into).unwrap_or_default(),
            recall_at_ks: r
                .recall_at_ks
                .into_iter()
                .map(|p| (p.k as usize, p.recall))
                .collect(),
            distance: r.distance,
            cohorts: r.cohorts.into_iter().collect(),
        }
    }
}

impl From<pb::EmbeddingEvalReport> for EmbeddingEvalReport {
    fn from(report: pb::EmbeddingEvalReport) -> Self {
        EmbeddingEvalReport {
            eval_run_id: report.eval_run_id,
            aggregate: report.aggregate.map(Into::into).unwrap_or_default(),
            per_query: report.per_query.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<pb::PerQueryEvalRecord> for PerQueryEvalRecord {
    fn from(rec: pb::PerQueryEvalRecord) -> Self {
        PerQueryEvalRecord {
            eval_run_id: rec.eval_run_id,
            query_id: rec.query_id,
            cohorts_json: rec.cohorts_json,
            metrics_json: rec.metrics_json,
        }
    }
}

impl From<pb::ClassMetrics> for ClassMetrics {
    fn from(m: pb::ClassMetrics) -> Self {
        ClassMetrics {
            precision: m.precision,
            recall: m.recall,
            f1: m.f1,
        }
    }
}

impl From<pb::ClassificationResult> for ClassificationResult {
    fn from(c: pb::ClassificationResult) -> Self {
        ClassificationResult {
            accuracy: c.accuracy,
            f1: c.f1,
            per_class: c
                .per_class
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect(),
        }
    }
}

impl From<pb::TypeMetrics> for TypeMetrics {
    fn from(m: pb::TypeMetrics) -> Self {
        TypeMetrics {
            precision: m.precision,
            recall: m.recall,
            f1: m.f1,
            support: m.support as usize,
        }
    }
}

impl From<pb::NerMetrics> for NerMetrics {
    fn from(n: pb::NerMetrics) -> Self {
        NerMetrics {
            precision: n.precision,
            recall: n.recall,
            f1: n.f1,
            per_type: n.per_type.into_iter().map(|(k, v)| (k, v.into())).collect(),
        }
    }
}

impl From<pb::Entity> for Entity {
    fn from(e: pb::Entity) -> Self {
        Entity {
            label: e.label,
            start: e.start as usize,
            end: e.end as usize,
            text: e.text,
            confidence: e.confidence,
        }
    }
}

impl From<pb::InferenceAggregate> for InferenceAggregate {
    fn from(a: pb::InferenceAggregate) -> Self {
        use pb::inference_aggregate::Aggregate;
        match a.aggregate {
            Some(Aggregate::Classification(c)) => InferenceAggregate::Classification(c.into()),
            Some(Aggregate::Ner(n)) => InferenceAggregate::Ner(n.into()),
            // A report with no aggregate is a malformed payload; the engine
            // never emits one. Rebuild an empty classification result rather
            // than panic — decode is total.
            None => InferenceAggregate::Classification(ClassificationResult::default()),
        }
    }
}

impl From<pb::PerRecordPrediction> for PerRecordPrediction {
    fn from(p: pb::PerRecordPrediction) -> Self {
        use pb::per_record_prediction::Prediction;
        match p.prediction {
            Some(Prediction::Classification(c)) => PerRecordPrediction::Classification {
                record_id: c.record_id,
                predicted: c.predicted,
                gold: c.gold,
            },
            Some(Prediction::Ner(n)) => PerRecordPrediction::Ner {
                record_id: n.record_id,
                predicted: n.predicted.into_iter().map(Into::into).collect(),
                gold: n.gold.into_iter().map(Into::into).collect(),
            },
            // A prediction with no variant is a malformed payload; the engine
            // never emits one. Rebuild an empty classification record.
            None => PerRecordPrediction::Classification {
                record_id: String::new(),
                predicted: String::new(),
                gold: String::new(),
            },
        }
    }
}

impl From<pb::InferenceEvalReport> for InferenceEvalReport {
    fn from(report: pb::InferenceEvalReport) -> Self {
        InferenceEvalReport {
            aggregate: report.aggregate.map(Into::into).unwrap_or_else(|| {
                InferenceAggregate::Classification(ClassificationResult::default())
            }),
            per_record: report.per_record.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<pb::MetricDelta> for MetricDelta {
    fn from(d: pb::MetricDelta) -> Self {
        MetricDelta {
            absolute: d.absolute,
            relative: d.relative,
        }
    }
}

impl From<pb::AggregateDelta> for AggregateDelta {
    fn from(d: pb::AggregateDelta) -> Self {
        AggregateDelta {
            recall_at_k: d.recall_at_k.map(Into::into).unwrap_or_default(),
            precision_at_k: d.precision_at_k.map(Into::into).unwrap_or_default(),
            mrr: d.mrr.map(Into::into).unwrap_or_default(),
            ndcg: d.ndcg.map(Into::into).unwrap_or_default(),
        }
    }
}

impl From<pb::TableEvalReport> for TableEvalReport {
    fn from(t: pb::TableEvalReport) -> Self {
        TableEvalReport {
            table_name: t.table_name,
            embedding_eval: t.embedding_eval.map(Into::into).unwrap_or_default(),
            delta: t.delta.map(Into::into),
        }
    }
}

impl From<pb::CompareEvalReport> for CompareEvalReport {
    fn from(report: pb::CompareEvalReport) -> Self {
        CompareEvalReport {
            per_table: report.per_table.into_iter().map(Into::into).collect(),
        }
    }
}
