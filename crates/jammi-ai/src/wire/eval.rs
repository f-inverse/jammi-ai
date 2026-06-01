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
