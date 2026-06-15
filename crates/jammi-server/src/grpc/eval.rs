//! `EvalService` gRPC implementation.
//!
//! Five verbs land on the wire: `EvalEmbeddings`, `EvalPerQuery`,
//! `EvalInference`, `EvalCompare`, `EvalCalibration`. Each is a thin adapter
//! over the engine session: proto in, one engine eval call inside the request's
//! tenant scope, proto out. The service reimplements no metric or retrieval
//! logic. The retrieval/inference/compare verbs route through the
//! transport-agnostic [`Session`] abstraction;
//! `EvalCalibration` calls the [`InferenceSession`] verb directly (the
//! calibration runner is not on the unified transport surface).
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
use jammi_ai::Session;
use tonic::{Request, Response, Status};

use crate::grpc::proto::eval as pb;
use crate::grpc::proto::eval::eval_service_server::EvalService;
use crate::grpc::wire::{map_engine_error, scoped, session_tenant_traced};

/// Server-side handler for the eval gRPC surface. Holds a shared engine session
/// it wraps in a [`Session`] per call to reach the unified transport
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
        Session::new(Arc::clone(&self.session))
    }
}

#[tonic::async_trait]
impl EvalService for EvalServer {
    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn eval_embeddings(
        &self,
        request: Request<pb::EvalEmbeddingsRequest>,
    ) -> Result<Response<pb::EmbeddingEvalReport>, Status> {
        let tenant = session_tenant_traced(&request);
        // Decode through the shared `jammi_ai::wire` seam — the same decode the
        // embedded binding's `_eval_embeddings_proto` drives — so both transports
        // validate and submit an identical request.
        let args = jammi_ai::wire::eval_embeddings_from_proto(request.into_inner())?;
        let session = self.local();

        let report = scoped(&self.session, tenant, || {
            session.eval_embeddings(
                &args.source_id,
                args.embedding_table.as_deref(),
                &args.golden_source,
                args.k,
                &args.cohorts,
            )
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(report.into()))
    }

    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn eval_per_query(
        &self,
        request: Request<pb::EvalPerQueryRequest>,
    ) -> Result<Response<pb::EvalPerQueryResponse>, Status> {
        let tenant = session_tenant_traced(&request);
        // Decode through the shared `jammi_ai::wire` seam — the same decode the
        // embedded binding's `_eval_per_query_proto` drives — so both transports
        // validate and submit an identical request.
        let eval_run_id = jammi_ai::wire::eval_per_query_from_proto(request.into_inner())?;
        let session = self.local();

        let records = scoped(&self.session, tenant, || {
            session.eval_per_query(&eval_run_id)
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(pb::EvalPerQueryResponse {
            records: records.into_iter().map(Into::into).collect(),
        }))
    }

    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn eval_inference(
        &self,
        request: Request<pb::EvalInferenceRequest>,
    ) -> Result<Response<pb::InferenceEvalReport>, Status> {
        let tenant = session_tenant_traced(&request);
        // Decode through the shared `jammi_ai::wire` seam — the same decode the
        // embedded binding's `_eval_inference_proto` drives — so both transports
        // validate and submit an identical request.
        let args = jammi_ai::wire::eval_inference_from_proto(request.into_inner())?;
        let session = self.local();

        let report = scoped(&self.session, tenant, || {
            session.eval_inference(
                &args.model_id,
                &args.source_id,
                &args.columns,
                args.task,
                &args.golden_source,
                &args.label_column,
            )
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(report.into()))
    }

    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn eval_compare(
        &self,
        request: Request<pb::EvalCompareRequest>,
    ) -> Result<Response<pb::CompareEvalReport>, Status> {
        let tenant = session_tenant_traced(&request);
        // Decode through the shared `jammi_ai::wire` seam — the same decode the
        // embedded binding's `_eval_compare_proto` drives — so both transports
        // validate and submit an identical request.
        let args = jammi_ai::wire::eval_compare_from_proto(request.into_inner())?;
        let session = self.local();

        let report = scoped(&self.session, tenant, || {
            session.eval_compare(
                &args.embedding_tables,
                &args.source_id,
                &args.golden_source,
                args.k,
            )
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(report.into()))
    }

    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn eval_calibration(
        &self,
        request: Request<pb::EvalCalibrationRequest>,
    ) -> Result<Response<pb::CalibrationEvalReport>, Status> {
        let tenant = session_tenant_traced(&request);
        // Decode through the shared `jammi_ai::wire` seam — the same decode the
        // embedded binding's `_eval_calibration_proto` drives — so both transports
        // validate and submit an identical request.
        let args = jammi_ai::wire::eval_calibration_from_proto(request.into_inner())?;

        let report = scoped(&self.session, tenant, || async {
            self.session
                .eval_calibration(
                    &args.source_id,
                    &args.golden_source,
                    args.shape,
                    &args.cohorts,
                )
                .await
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(report.into()))
    }
}
