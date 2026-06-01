//! `EvalService` gRPC implementation.
//!
//! Four verbs land on the wire: `EvalEmbeddings`, `EvalPerQuery`,
//! `EvalInference`, `EvalCompare`. Each is a thin adapter over the
//! transport-agnostic [`Session`]/[`LocalSession`] abstraction (never raw
//! [`InferenceSession`] calls): proto in, one `Session::eval_*` call, proto
//! out. The service reimplements no metric or retrieval logic.
//!
//! The response messages mirror the Rust report structs the abstraction
//! returns — [`jammi_ai::eval::EmbeddingEvalReport`],
//! [`jammi_ai::eval::InferenceEvalReport`],
//! [`jammi_ai::eval::CompareEvalReport`], and the catalog's
//! [`jammi_db::catalog::eval_repo::PerQueryEvalRecord`] — field for field, with
//! typed fields rather than opaque JSON (except the per-query persistence
//! record, whose `cohorts`/`metrics` columns are JSON-object strings by storage
//! shape and are carried verbatim).
//!
//! Tenant scope is read from the request's [`crate::grpc::session::
//! SessionTenant`] extension (set upstream by the shared `TenantInterceptor`)
//! and applied via `with_tenant_scoped`, matching every other engine-backed
//! gRPC surface.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use jammi_ai::eval::report::{
    AggregateDelta, CompareEvalReport, EmbeddingEvalReport, InferenceAggregate,
    InferenceEvalReport, MetricDelta, PerQueryRecord, PerRecordPrediction, TableEvalReport,
};
use jammi_ai::eval::EvalTask;
use jammi_ai::session::InferenceSession;
use jammi_ai::{LocalSession, Session};
use jammi_db::catalog::eval_repo::PerQueryEvalRecord;
use jammi_numerics::classification::{ClassMetrics, ClassificationResult};
use jammi_numerics::ner::{Entity, NerMetrics, TypeMetrics};
use jammi_numerics::retrieval::{AggregateMetrics, QueryMetrics};
use tonic::{Request, Response, Status};

use crate::grpc::proto::eval as pb;
use crate::grpc::proto::eval::eval_service_server::EvalService;
use crate::grpc::wire::{map_engine_error, require_nonempty, scoped, session_tenant};

/// Server-side handler for the eval gRPC surface. Holds a shared engine session
/// it wraps in a [`LocalSession`] per call to reach the unified transport
/// surface.
pub struct EvalServer {
    session: Arc<InferenceSession>,
}

impl EvalServer {
    pub fn new(session: Arc<InferenceSession>) -> Self {
        Self { session }
    }

    /// A [`Session`] over the shared engine; see [`crate::grpc::inference`] for
    /// the tenant-scope wiring rationale.
    fn local(&self) -> Session {
        Session::Local(LocalSession::new(Arc::clone(&self.session)))
    }
}

#[tonic::async_trait]
impl EvalService for EvalServer {
    async fn eval_embeddings(
        &self,
        request: Request<pb::EvalEmbeddingsRequest>,
    ) -> Result<Response<pb::EmbeddingEvalReport>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        require_nonempty(&req.source_id, "source_id")?;
        require_nonempty(&req.golden_source, "golden_source")?;
        let embedding_table = optional_str(&req.embedding_table);
        let cohorts = cohorts_from_proto(req.cohorts);
        let session = self.local();

        let report = scoped(&self.session, tenant, || {
            session.eval_embeddings(
                &req.source_id,
                embedding_table,
                &req.golden_source,
                req.k as usize,
                &cohorts,
            )
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(embedding_report_to_proto(report)))
    }

    async fn eval_per_query(
        &self,
        request: Request<pb::EvalPerQueryRequest>,
    ) -> Result<Response<pb::EvalPerQueryResponse>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        require_nonempty(&req.eval_run_id, "eval_run_id")?;
        let session = self.local();

        let records = scoped(&self.session, tenant, || {
            session.eval_per_query(&req.eval_run_id)
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(pb::EvalPerQueryResponse {
            records: records.into_iter().map(per_query_record_to_proto).collect(),
        }))
    }

    async fn eval_inference(
        &self,
        request: Request<pb::EvalInferenceRequest>,
    ) -> Result<Response<pb::InferenceEvalReport>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        require_nonempty(&req.model_id, "model_id")?;
        require_nonempty(&req.source_id, "source_id")?;
        require_nonempty(&req.golden_source, "golden_source")?;
        require_nonempty(&req.label_column, "label_column")?;
        if req.columns.is_empty() {
            return Err(Status::invalid_argument("columns is required"));
        }
        let task = eval_task_from_proto(req.task)?;
        let session = self.local();

        let report = scoped(&self.session, tenant, || {
            session.eval_inference(
                &req.model_id,
                &req.source_id,
                &req.columns,
                task,
                &req.golden_source,
                &req.label_column,
            )
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(inference_report_to_proto(report)))
    }

    async fn eval_compare(
        &self,
        request: Request<pb::EvalCompareRequest>,
    ) -> Result<Response<pb::CompareEvalReport>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        require_nonempty(&req.source_id, "source_id")?;
        require_nonempty(&req.golden_source, "golden_source")?;
        if req.embedding_tables.len() < 2 {
            return Err(Status::invalid_argument(
                "embedding_tables requires at least two tables",
            ));
        }
        let session = self.local();

        let report = scoped(&self.session, tenant, || {
            session.eval_compare(
                &req.embedding_tables,
                &req.source_id,
                &req.golden_source,
                req.k as usize,
            )
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(compare_report_to_proto(report)))
    }
}

/// `""` → `None`, a non-empty string → `Some(&str)`. Mirrors the engine's
/// `Option<&str>` "use the most recent table" sentinel for `embedding_table`.
fn optional_str(s: &str) -> Option<&str> {
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// Map the proto [`EvalTask`] onto the engine's [`EvalTask`]. An unspecified
/// task is rejected — a request that names no task is a client error.
fn eval_task_from_proto(task: i32) -> Result<EvalTask, Status> {
    match pb::EvalTask::try_from(task) {
        Ok(pb::EvalTask::Classification) => Ok(EvalTask::Classification),
        Ok(pb::EvalTask::Ner) => Ok(EvalTask::Ner),
        Ok(pb::EvalTask::Unspecified) | Err(_) => {
            Err(Status::invalid_argument("task must be specified"))
        }
    }
}

/// Rebuild the engine's `query_id → {key: value}` cohort map from the proto
/// `map<string, CohortTags>`. The substrate never interprets these tags.
fn cohorts_from_proto(
    cohorts: HashMap<String, pb::CohortTags>,
) -> HashMap<String, BTreeMap<String, String>> {
    cohorts
        .into_iter()
        .map(|(query_id, tags)| (query_id, tags.tags.into_iter().collect()))
        .collect()
}

// ─── Report → proto ──────────────────────────────────────────────────────────

fn aggregate_to_proto(a: &AggregateMetrics) -> pb::AggregateMetrics {
    pb::AggregateMetrics {
        recall_at_k: a.recall_at_k,
        precision_at_k: a.precision_at_k,
        mrr: a.mrr,
        ndcg: a.ndcg,
    }
}

fn query_metrics_to_proto(m: &QueryMetrics) -> pb::QueryMetrics {
    pb::QueryMetrics {
        recall: m.recall,
        precision: m.precision,
        mrr: m.mrr,
        ndcg: m.ndcg,
    }
}

fn per_query_to_proto(r: PerQueryRecord) -> pb::PerQueryRecord {
    pb::PerQueryRecord {
        query_id: r.query_id,
        metrics: Some(query_metrics_to_proto(&r.metrics)),
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

fn embedding_report_to_proto(report: EmbeddingEvalReport) -> pb::EmbeddingEvalReport {
    pb::EmbeddingEvalReport {
        eval_run_id: report.eval_run_id,
        aggregate: Some(aggregate_to_proto(&report.aggregate)),
        per_query: report
            .per_query
            .into_iter()
            .map(per_query_to_proto)
            .collect(),
    }
}

fn per_query_record_to_proto(rec: PerQueryEvalRecord) -> pb::PerQueryEvalRecord {
    pb::PerQueryEvalRecord {
        eval_run_id: rec.eval_run_id,
        query_id: rec.query_id,
        cohorts_json: rec.cohorts_json,
        metrics_json: rec.metrics_json,
    }
}

fn class_metrics_to_proto(m: &ClassMetrics) -> pb::ClassMetrics {
    pb::ClassMetrics {
        precision: m.precision,
        recall: m.recall,
        f1: m.f1,
    }
}

fn classification_to_proto(c: &ClassificationResult) -> pb::ClassificationResult {
    pb::ClassificationResult {
        accuracy: c.accuracy,
        f1: c.f1,
        per_class: c
            .per_class
            .iter()
            .map(|(k, v)| (k.clone(), class_metrics_to_proto(v)))
            .collect(),
    }
}

fn type_metrics_to_proto(m: &TypeMetrics) -> pb::TypeMetrics {
    pb::TypeMetrics {
        precision: m.precision,
        recall: m.recall,
        f1: m.f1,
        support: m.support as u64,
    }
}

fn ner_metrics_to_proto(n: &NerMetrics) -> pb::NerMetrics {
    pb::NerMetrics {
        precision: n.precision,
        recall: n.recall,
        f1: n.f1,
        per_type: n
            .per_type
            .iter()
            .map(|(k, v)| (k.clone(), type_metrics_to_proto(v)))
            .collect(),
    }
}

fn entity_to_proto(e: &Entity) -> pb::Entity {
    pb::Entity {
        label: e.label.clone(),
        start: e.start as u64,
        end: e.end as u64,
        text: e.text.clone(),
        confidence: e.confidence,
    }
}

fn inference_aggregate_to_proto(a: &InferenceAggregate) -> pb::InferenceAggregate {
    let aggregate = match a {
        InferenceAggregate::Classification(c) => {
            pb::inference_aggregate::Aggregate::Classification(classification_to_proto(c))
        }
        InferenceAggregate::Ner(n) => {
            pb::inference_aggregate::Aggregate::Ner(ner_metrics_to_proto(n))
        }
    };
    pb::InferenceAggregate {
        aggregate: Some(aggregate),
    }
}

fn per_record_to_proto(p: PerRecordPrediction) -> pb::PerRecordPrediction {
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
            predicted: predicted.iter().map(entity_to_proto).collect(),
            gold: gold.iter().map(entity_to_proto).collect(),
        }),
    };
    pb::PerRecordPrediction {
        prediction: Some(prediction),
    }
}

fn inference_report_to_proto(report: InferenceEvalReport) -> pb::InferenceEvalReport {
    pb::InferenceEvalReport {
        aggregate: Some(inference_aggregate_to_proto(&report.aggregate)),
        per_record: report
            .per_record
            .into_iter()
            .map(per_record_to_proto)
            .collect(),
    }
}

fn metric_delta_to_proto(d: &MetricDelta) -> pb::MetricDelta {
    pb::MetricDelta {
        absolute: d.absolute,
        relative: d.relative,
    }
}

fn aggregate_delta_to_proto(d: &AggregateDelta) -> pb::AggregateDelta {
    pb::AggregateDelta {
        recall_at_k: Some(metric_delta_to_proto(&d.recall_at_k)),
        precision_at_k: Some(metric_delta_to_proto(&d.precision_at_k)),
        mrr: Some(metric_delta_to_proto(&d.mrr)),
        ndcg: Some(metric_delta_to_proto(&d.ndcg)),
    }
}

fn table_report_to_proto(t: TableEvalReport) -> pb::TableEvalReport {
    pb::TableEvalReport {
        table_name: t.table_name,
        embedding_eval: Some(embedding_report_to_proto(t.embedding_eval)),
        delta: t.delta.as_ref().map(aggregate_delta_to_proto),
    }
}

fn compare_report_to_proto(report: CompareEvalReport) -> pb::CompareEvalReport {
    pb::CompareEvalReport {
        per_table: report
            .per_table
            .into_iter()
            .map(table_report_to_proto)
            .collect(),
    }
}
