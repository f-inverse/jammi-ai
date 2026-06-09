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
use jammi_db::error::JammiError;
use jammi_numerics::classification::{ClassMetrics, ClassificationResult};
use jammi_numerics::ner::{Entity, NerMetrics, TypeMetrics};
use jammi_numerics::retrieval::{AggregateMetrics, QueryMetrics};
use tonic::Status;

use crate::eval::report::{
    AggregateDelta, CalibrationAggregate, CalibrationEvalReport, CohortCalibration,
    CompareEvalReport, DeltaSignificance, EmbeddingEvalReport, InferenceAggregate,
    InferenceEvalReport, MetricDelta, MetricSignificance, PerQueryRecord, PerRecordCalibration,
    PerRecordPrediction, TableEvalReport,
};
use crate::eval::{EvalCalibrationShape, EvalTask};
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

/// Map the proto [`pb::CalibrationShape`] discriminant onto the engine's
/// [`EvalCalibrationShape`]. An unspecified or unknown shape is rejected — a
/// request that names no predictive shape cannot select the columns or scoring
/// family, so it is a client error rather than a silent default.
pub fn calibration_shape_from_proto(shape: i32) -> Result<EvalCalibrationShape, Status> {
    match pb::CalibrationShape::try_from(shape) {
        Ok(pb::CalibrationShape::Gaussian) => Ok(EvalCalibrationShape::Gaussian),
        Ok(pb::CalibrationShape::Sample) => Ok(EvalCalibrationShape::Sample),
        Ok(pb::CalibrationShape::Unspecified) | Err(_) => Err(Status::invalid_argument(
            "calibration shape must be GAUSSIAN or SAMPLE",
        )),
    }
}

/// Encode the engine's [`EvalCalibrationShape`] onto the wire enum — the inverse
/// of [`calibration_shape_from_proto`], for the [`crate::RemoteSession`] send
/// side. Total: every engine shape maps to a concrete wire variant.
pub fn calibration_shape_to_proto(shape: EvalCalibrationShape) -> pb::CalibrationShape {
    match shape {
        EvalCalibrationShape::Gaussian => pb::CalibrationShape::Gaussian,
        EvalCalibrationShape::Sample => pb::CalibrationShape::Sample,
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

impl From<&MetricSignificance> for pb::MetricSignificance {
    fn from(s: &MetricSignificance) -> Self {
        pb::MetricSignificance {
            p_value: s.p_value,
            ci_lower: s.ci_lower,
            ci_upper: s.ci_upper,
        }
    }
}

impl From<&DeltaSignificance> for pb::DeltaSignificance {
    fn from(s: &DeltaSignificance) -> Self {
        pb::DeltaSignificance {
            recall_at_k: Some((&s.recall_at_k).into()),
            precision_at_k: Some((&s.precision_at_k).into()),
            mrr: Some((&s.mrr).into()),
            ndcg: Some((&s.ndcg).into()),
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
            significance: d.significance.as_ref().map(Into::into),
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

impl From<&CalibrationAggregate> for pb::CalibrationAggregate {
    fn from(a: &CalibrationAggregate) -> Self {
        pb::CalibrationAggregate {
            n: a.n as u64,
            crps: a.crps,
            nll: a.nll,
            adaptive_ece: a.adaptive_ece,
            sharpness: a.sharpness,
            coverage: a.coverage,
        }
    }
}

impl From<CohortCalibration> for pb::CohortCalibration {
    fn from(c: CohortCalibration) -> Self {
        pb::CohortCalibration {
            key: c.key,
            value: c.value,
            n: c.n as u64,
            crps: c.crps,
            crps_ci_lower: c.crps_ci_lower,
            crps_ci_upper: c.crps_ci_upper,
            coverage: c.coverage,
        }
    }
}

impl From<PerRecordCalibration> for pb::PerRecordCalibration {
    fn from(r: PerRecordCalibration) -> Self {
        pb::PerRecordCalibration {
            record_id: r.record_id,
            crps: r.crps,
            nll: r.nll,
            pit: r.pit,
            covered: r.covered,
            interval_width: r.interval_width,
            cohorts: r.cohorts.into_iter().collect(),
        }
    }
}

impl From<CalibrationEvalReport> for pb::CalibrationEvalReport {
    fn from(report: CalibrationEvalReport) -> Self {
        pb::CalibrationEvalReport {
            eval_run_id: report.eval_run_id,
            aggregate: Some((&report.aggregate).into()),
            per_cohort: report.per_cohort.into_iter().map(Into::into).collect(),
            per_record: report.per_record.into_iter().map(Into::into).collect(),
        }
    }
}

// ─── proto → report (the RemoteSession decode side) ──────────────────────────
//
// The inverse of the encodes above. A remote client reads a report message off
// the wire and rebuilds the engine report struct so `Session::Remote` returns
// the identical type `Session::Local` does. Each decode mirrors its encode
// field for field; the report structs and the metric structs are local crates,
// so these impls are orphan-rule-clean without a newtype.
//
// A field that proto3 represents as a nested *message* but the engine domain
// requires (a metric block the server always populates) decodes through a
// fallible `TryFrom`: a `None` there is only reachable on a corrupt or
// incompatible wire payload, and fabricating the zero value (recall = 0.0, …)
// would hand a consumer plausible-wrong result data it cannot distinguish from
// a real score. At the network boundary the right idiom is to propagate the
// error as a value — [`malformed`] builds a [`JammiError::Eval`] naming the
// missing field, and the [`crate::RemoteSession`] eval verb returns it as the
// call's `Err`. Scalar proto3 primitives keep their proto-default (0 / "");
// only required nested messages reject. Genuinely-optional fields
// ([`TableEvalReport::delta`], `None` for the compare baseline) stay `Option`.

/// Build the decode error for a required nested message that arrived `None`.
fn malformed(field: &str) -> JammiError {
    JammiError::Eval(format!("malformed eval report: missing {field}"))
}

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

impl TryFrom<pb::PerQueryRecord> for PerQueryRecord {
    type Error = JammiError;

    fn try_from(r: pb::PerQueryRecord) -> Result<Self, Self::Error> {
        Ok(PerQueryRecord {
            query_id: r.query_id,
            metrics: r
                .metrics
                .map(Into::into)
                .ok_or_else(|| malformed("PerQueryRecord.metrics"))?,
            recall_at_ks: r
                .recall_at_ks
                .into_iter()
                .map(|p| (p.k as usize, p.recall))
                .collect(),
            distance: r.distance,
            cohorts: r.cohorts.into_iter().collect(),
        })
    }
}

impl TryFrom<pb::EmbeddingEvalReport> for EmbeddingEvalReport {
    type Error = JammiError;

    fn try_from(report: pb::EmbeddingEvalReport) -> Result<Self, Self::Error> {
        Ok(EmbeddingEvalReport {
            eval_run_id: report.eval_run_id,
            aggregate: report
                .aggregate
                .map(Into::into)
                .ok_or_else(|| malformed("EmbeddingEvalReport.aggregate"))?,
            per_query: report
                .per_query
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<_, _>>()?,
        })
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

impl TryFrom<pb::InferenceAggregate> for InferenceAggregate {
    type Error = JammiError;

    fn try_from(a: pb::InferenceAggregate) -> Result<Self, Self::Error> {
        use pb::inference_aggregate::Aggregate;
        match a.aggregate {
            Some(Aggregate::Classification(c)) => Ok(InferenceAggregate::Classification(c.into())),
            Some(Aggregate::Ner(n)) => Ok(InferenceAggregate::Ner(n.into())),
            // An aggregate with no variant is only reachable on a corrupt
            // payload; the engine always tags one. Reject rather than fabricate
            // an empty result a gate would read as a real score.
            None => Err(malformed("InferenceAggregate.aggregate")),
        }
    }
}

impl TryFrom<pb::PerRecordPrediction> for PerRecordPrediction {
    type Error = JammiError;

    fn try_from(p: pb::PerRecordPrediction) -> Result<Self, Self::Error> {
        use pb::per_record_prediction::Prediction;
        match p.prediction {
            Some(Prediction::Classification(c)) => Ok(PerRecordPrediction::Classification {
                record_id: c.record_id,
                predicted: c.predicted,
                gold: c.gold,
            }),
            Some(Prediction::Ner(n)) => Ok(PerRecordPrediction::Ner {
                record_id: n.record_id,
                predicted: n.predicted.into_iter().map(Into::into).collect(),
                gold: n.gold.into_iter().map(Into::into).collect(),
            }),
            // A prediction with no variant is only reachable on a corrupt
            // payload; the engine always tags one.
            None => Err(malformed("PerRecordPrediction.prediction")),
        }
    }
}

impl TryFrom<pb::InferenceEvalReport> for InferenceEvalReport {
    type Error = JammiError;

    fn try_from(report: pb::InferenceEvalReport) -> Result<Self, Self::Error> {
        Ok(InferenceEvalReport {
            aggregate: report
                .aggregate
                .ok_or_else(|| malformed("InferenceEvalReport.aggregate"))?
                .try_into()?,
            per_record: report
                .per_record
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<_, _>>()?,
        })
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

impl From<pb::MetricSignificance> for MetricSignificance {
    fn from(s: pb::MetricSignificance) -> Self {
        MetricSignificance {
            p_value: s.p_value,
            ci_lower: s.ci_lower,
            ci_upper: s.ci_upper,
        }
    }
}

impl TryFrom<pb::DeltaSignificance> for DeltaSignificance {
    type Error = JammiError;

    fn try_from(s: pb::DeltaSignificance) -> Result<Self, Self::Error> {
        Ok(DeltaSignificance {
            recall_at_k: s
                .recall_at_k
                .map(Into::into)
                .ok_or_else(|| malformed("DeltaSignificance.recall_at_k"))?,
            precision_at_k: s
                .precision_at_k
                .map(Into::into)
                .ok_or_else(|| malformed("DeltaSignificance.precision_at_k"))?,
            mrr: s
                .mrr
                .map(Into::into)
                .ok_or_else(|| malformed("DeltaSignificance.mrr"))?,
            ndcg: s
                .ndcg
                .map(Into::into)
                .ok_or_else(|| malformed("DeltaSignificance.ndcg"))?,
        })
    }
}

impl TryFrom<pb::AggregateDelta> for AggregateDelta {
    type Error = JammiError;

    fn try_from(d: pb::AggregateDelta) -> Result<Self, Self::Error> {
        Ok(AggregateDelta {
            recall_at_k: d
                .recall_at_k
                .map(Into::into)
                .ok_or_else(|| malformed("AggregateDelta.recall_at_k"))?,
            precision_at_k: d
                .precision_at_k
                .map(Into::into)
                .ok_or_else(|| malformed("AggregateDelta.precision_at_k"))?,
            mrr: d
                .mrr
                .map(Into::into)
                .ok_or_else(|| malformed("AggregateDelta.mrr"))?,
            ndcg: d
                .ndcg
                .map(Into::into)
                .ok_or_else(|| malformed("AggregateDelta.ndcg"))?,
            // `significance` is genuinely optional: `None` is "no shared query
            // to pair on", not a corrupt payload.
            significance: d.significance.map(TryInto::try_into).transpose()?,
        })
    }
}

impl TryFrom<pb::TableEvalReport> for TableEvalReport {
    type Error = JammiError;

    fn try_from(t: pb::TableEvalReport) -> Result<Self, Self::Error> {
        Ok(TableEvalReport {
            table_name: t.table_name,
            embedding_eval: t
                .embedding_eval
                .ok_or_else(|| malformed("TableEvalReport.embedding_eval"))?
                .try_into()?,
            // `delta` is genuinely optional: `None` is the compare baseline,
            // not a corrupt payload.
            delta: t.delta.map(TryInto::try_into).transpose()?,
        })
    }
}

impl TryFrom<pb::CompareEvalReport> for CompareEvalReport {
    type Error = JammiError;

    fn try_from(report: pb::CompareEvalReport) -> Result<Self, Self::Error> {
        Ok(CompareEvalReport {
            per_table: report
                .per_table
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<_, _>>()?,
        })
    }
}

impl From<pb::CalibrationAggregate> for CalibrationAggregate {
    fn from(a: pb::CalibrationAggregate) -> Self {
        CalibrationAggregate {
            n: a.n as usize,
            crps: a.crps,
            nll: a.nll,
            adaptive_ece: a.adaptive_ece,
            sharpness: a.sharpness,
            coverage: a.coverage,
        }
    }
}

impl From<pb::CohortCalibration> for CohortCalibration {
    fn from(c: pb::CohortCalibration) -> Self {
        CohortCalibration {
            key: c.key,
            value: c.value,
            n: c.n as usize,
            crps: c.crps,
            crps_ci_lower: c.crps_ci_lower,
            crps_ci_upper: c.crps_ci_upper,
            coverage: c.coverage,
        }
    }
}

impl From<pb::PerRecordCalibration> for PerRecordCalibration {
    fn from(r: pb::PerRecordCalibration) -> Self {
        PerRecordCalibration {
            record_id: r.record_id,
            crps: r.crps,
            nll: r.nll,
            pit: r.pit,
            covered: r.covered,
            interval_width: r.interval_width,
            cohorts: r.cohorts.into_iter().collect(),
        }
    }
}

impl TryFrom<pb::CalibrationEvalReport> for CalibrationEvalReport {
    type Error = JammiError;

    fn try_from(report: pb::CalibrationEvalReport) -> Result<Self, Self::Error> {
        Ok(CalibrationEvalReport {
            eval_run_id: report.eval_run_id,
            // The engine always populates the aggregate block; a `None` here is
            // only reachable on a corrupt payload, and a fabricated zero report
            // would read as a real (perfect) calibration score.
            aggregate: report
                .aggregate
                .map(Into::into)
                .ok_or_else(|| malformed("CalibrationEvalReport.aggregate"))?,
            per_cohort: report.per_cohort.into_iter().map(Into::into).collect(),
            per_record: report.per_record.into_iter().map(Into::into).collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    //! Decode-side rejection of corrupt payloads. The engine always populates
    //! every required nested metric block, so a `None` there is only reachable
    //! on a corrupt/incompatible wire payload: the decode must reject it as an
    //! `Err`, never fabricate a zeroed report a consumer would read as a real
    //! score. The happy path (every block present) is exercised end-to-end by
    //! the `grpc_remote_compute` round-trip integration test.

    use super::*;

    fn populated_aggregate() -> pb::AggregateMetrics {
        pb::AggregateMetrics {
            recall_at_k: 0.8,
            precision_at_k: 0.7,
            mrr: 0.6,
            ndcg: 0.9,
        }
    }

    fn populated_query_metrics() -> pb::QueryMetrics {
        pb::QueryMetrics {
            recall: 1.0,
            precision: 1.0,
            mrr: 1.0,
            ndcg: 1.0,
        }
    }

    fn assert_missing(err: JammiError, field: &str) {
        match err {
            JammiError::Eval(msg) => assert_eq!(
                msg,
                format!("malformed eval report: missing {field}"),
                "decode error names the missing field"
            ),
            other => panic!("expected JammiError::Eval, got {other:?}"),
        }
    }

    #[test]
    fn per_query_record_missing_metrics_is_rejected() {
        let wire = pb::PerQueryRecord {
            query_id: "q1".into(),
            metrics: None,
            recall_at_ks: vec![],
            distance: 0.0,
            cohorts: Default::default(),
        };
        assert_missing(
            PerQueryRecord::try_from(wire).expect_err("missing metrics must reject"),
            "PerQueryRecord.metrics",
        );
    }

    #[test]
    fn embedding_report_missing_aggregate_is_rejected() {
        let wire = pb::EmbeddingEvalReport {
            eval_run_id: "run-1".into(),
            aggregate: None,
            per_query: vec![],
        };
        assert_missing(
            EmbeddingEvalReport::try_from(wire).expect_err("missing aggregate must reject"),
            "EmbeddingEvalReport.aggregate",
        );
    }

    #[test]
    fn embedding_report_with_aggregate_decodes() {
        let wire = pb::EmbeddingEvalReport {
            eval_run_id: "run-1".into(),
            aggregate: Some(populated_aggregate()),
            per_query: vec![pb::PerQueryRecord {
                query_id: "q1".into(),
                metrics: Some(populated_query_metrics()),
                recall_at_ks: vec![pb::RecallAtK { k: 1, recall: 1.0 }],
                distance: 0.42,
                cohorts: Default::default(),
            }],
        };
        let report = EmbeddingEvalReport::try_from(wire).expect("a fully populated report decodes");
        assert_eq!(report.aggregate.recall_at_k, 0.8);
        assert_eq!(report.per_query.len(), 1);
        assert_eq!(report.per_query[0].metrics.recall, 1.0);
    }

    #[test]
    fn inference_aggregate_with_no_variant_is_rejected() {
        let wire = pb::InferenceAggregate { aggregate: None };
        assert_missing(
            InferenceAggregate::try_from(wire).expect_err("untagged aggregate must reject"),
            "InferenceAggregate.aggregate",
        );
    }

    #[test]
    fn aggregate_delta_missing_metric_block_is_rejected() {
        let wire = pb::AggregateDelta {
            recall_at_k: Some(pb::MetricDelta {
                absolute: 0.1,
                relative: 0.2,
            }),
            precision_at_k: None,
            mrr: None,
            ndcg: None,
            significance: None,
        };
        assert_missing(
            AggregateDelta::try_from(wire).expect_err("missing precision delta must reject"),
            "AggregateDelta.precision_at_k",
        );
    }

    #[test]
    fn delta_significance_missing_metric_block_is_rejected() {
        let wire = pb::DeltaSignificance {
            recall_at_k: Some(pb::MetricSignificance {
                p_value: 0.04,
                ci_lower: 0.01,
                ci_upper: 0.2,
            }),
            precision_at_k: None,
            mrr: None,
            ndcg: None,
        };
        assert_missing(
            DeltaSignificance::try_from(wire).expect_err("missing precision significance rejects"),
            "DeltaSignificance.precision_at_k",
        );
    }

    #[test]
    fn aggregate_delta_significance_round_trips() {
        // A populated `significance` block survives encode → decode unchanged,
        // and the genuinely-optional `None` decodes back to `None`.
        let block = pb::MetricSignificance {
            p_value: 0.03,
            ci_lower: 0.05,
            ci_upper: 0.4,
        };
        let domain = AggregateDelta {
            recall_at_k: MetricDelta {
                absolute: 0.1,
                relative: 0.2,
            },
            precision_at_k: MetricDelta {
                absolute: 0.1,
                relative: 0.2,
            },
            mrr: MetricDelta {
                absolute: 0.1,
                relative: 0.2,
            },
            ndcg: MetricDelta {
                absolute: 0.1,
                relative: 0.2,
            },
            significance: Some(DeltaSignificance {
                recall_at_k: MetricSignificance::from(block),
                precision_at_k: MetricSignificance::from(block),
                mrr: MetricSignificance::from(block),
                ndcg: MetricSignificance::from(block),
            }),
        };
        let wire: pb::AggregateDelta = (&domain).into();
        let back = AggregateDelta::try_from(wire).expect("round-trip");
        let sig = back.significance.expect("significance present");
        assert_eq!(sig.recall_at_k.p_value, 0.03);
        assert_eq!(sig.ndcg.ci_upper, 0.4);
    }

    #[test]
    fn table_report_missing_embedding_eval_is_rejected() {
        let wire = pb::TableEvalReport {
            table_name: "patents".into(),
            embedding_eval: None,
            delta: None,
        };
        assert_missing(
            TableEvalReport::try_from(wire).expect_err("missing embedding_eval must reject"),
            "TableEvalReport.embedding_eval",
        );
    }

    #[test]
    fn table_report_baseline_keeps_optional_delta_none() {
        // `delta = None` is the compare baseline, a genuinely-optional field —
        // it decodes to `None`, not an error, when the required block is present.
        let wire = pb::TableEvalReport {
            table_name: "patents".into(),
            embedding_eval: Some(pb::EmbeddingEvalReport {
                eval_run_id: "run-1".into(),
                aggregate: Some(populated_aggregate()),
                per_query: vec![],
            }),
            delta: None,
        };
        let report = TableEvalReport::try_from(wire).expect("baseline table decodes");
        assert!(report.delta.is_none(), "baseline carries no delta");
    }

    #[test]
    fn corrupt_inner_table_fails_the_whole_compare_decode() {
        // A corrupt block nested inside `per_table` propagates out as the
        // compare decode's error rather than yielding a zeroed table entry.
        let wire = pb::CompareEvalReport {
            per_table: vec![pb::TableEvalReport {
                table_name: "patents".into(),
                embedding_eval: None,
                delta: None,
            }],
        };
        assert_missing(
            CompareEvalReport::try_from(wire).expect_err("corrupt inner table must reject"),
            "TableEvalReport.embedding_eval",
        );
    }
}
