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

use std::sync::Arc;

use jammi_ai::session::InferenceSession;
use jammi_ai::wire::{cohorts_from_proto, EvalTaskFromWire};
use jammi_ai::{LocalSession, Session};
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

        Ok(Response::new(report.into()))
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
            records: records.into_iter().map(Into::into).collect(),
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
        let task = EvalTaskFromWire::try_from(req.task)?.0;
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

        Ok(Response::new(report.into()))
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

        Ok(Response::new(report.into()))
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
