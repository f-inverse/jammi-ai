//! `PipelineService` gRPC implementation.
//!
//! Three verbs land on the wire: `BuildNeighborGraph`, `PropagateEmbeddings`,
//! and `AssembleContext`. Each is a thin adapter over the engine
//! [`InferenceSession`]: proto in, one engine call inside the request's tenant
//! scope, proto out. The service reimplements no retrieval, graph, or
//! aggregation logic.
//!
//! The two graph-build verbs return the engine's result-table record as the
//! shared [`jammi.v1.embedding.ResultTable`] (the same handle `GenerateEmbeddings`
//! returns) — the compute stays server-side and the client reads the table via
//! SQL. `AssembleContext` returns its pooled context vector and carried metadata
//! inline: the vector as IEEE-754 `float` (bit-exact for the engine's
//! `Vec<f32>`), the hydrated value rows as one Arrow IPC stream.
//!
//! Tenant scope is read from the request's [`crate::grpc::session::
//! SessionTenant`] extension (set upstream by the shared `TenantInterceptor`)
//! and applied via [`scoped`], matching every other engine-backed gRPC surface.
//! `build_neighbor_graph` self-scopes internally too; calling it inside [`scoped`]
//! is an idempotent same-tenant re-scope, not a special case.

use std::sync::Arc;

use jammi_ai::session::InferenceSession;
use jammi_ai::wire::{
    asof_join_from_proto, assemble_context_request_from_proto, assemble_context_to_proto,
    build_neighbor_graph_from_proto, propagate_request_from_proto, recompute_from_proto,
    recompute_report_to_proto,
};
use jammi_db::error::JammiError;
use tonic::{Request, Response, Status};

use crate::grpc::proto::embedding::ResultTable;
use crate::grpc::proto::pipeline::pipeline_service_server::PipelineService;
use crate::grpc::proto::pipeline::{
    AsofJoinRequest, AssembleContextRequest, AssembleContextResponse, BuildNeighborGraphRequest,
    PropagateEmbeddingsRequest, RecomputeReport as ProtoRecomputeReport, RecomputeRequest,
};
use crate::grpc::wire::{map_engine_error, scoped, session_tenant_traced};

/// Server-side handler for the pipeline gRPC surface. Holds the shared engine
/// session it drives directly inside the request's tenant scope.
pub struct PipelineServer {
    session: Arc<InferenceSession>,
}

impl PipelineServer {
    pub fn new(session: Arc<InferenceSession>) -> Self {
        Self { session }
    }
}

#[tonic::async_trait]
impl PipelineService for PipelineServer {
    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn build_neighbor_graph(
        &self,
        request: Request<BuildNeighborGraphRequest>,
    ) -> Result<Response<ResultTable>, Status> {
        let tenant = session_tenant_traced(&request);
        let args = build_neighbor_graph_from_proto(request.into_inner())?;

        let (record, outcome) = scoped(&self.session, tenant, || async {
            self.session
                .build_neighbor_graph(
                    &args.source_id,
                    args.embedding_table.as_deref(),
                    &args.params,
                    args.cache,
                )
                .await
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(jammi_wire::result_table_with_outcome(
            record,
            jammi_ai::wire::cache_outcome_to_proto(&outcome),
        )))
    }

    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn propagate_embeddings(
        &self,
        request: Request<PropagateEmbeddingsRequest>,
    ) -> Result<Response<ResultTable>, Status> {
        let tenant = session_tenant_traced(&request);
        let (req, cache) = propagate_request_from_proto(request.into_inner())?;

        let (record, outcome) = scoped(&self.session, tenant, || async {
            self.session.propagate_embeddings(&req, cache).await
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(jammi_wire::result_table_with_outcome(
            record,
            jammi_ai::wire::cache_outcome_to_proto(&outcome),
        )))
    }

    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn assemble_context(
        &self,
        request: Request<AssembleContextRequest>,
    ) -> Result<Response<AssembleContextResponse>, Status> {
        let tenant = session_tenant_traced(&request);
        let req = assemble_context_request_from_proto(request.into_inner())?;

        let context = scoped(&self.session, tenant, || async {
            self.session.assemble_context(&req).await
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(assemble_context_to_proto(context)?))
    }

    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn asof_join(
        &self,
        request: Request<AsofJoinRequest>,
    ) -> Result<Response<ResultTable>, Status> {
        let tenant = session_tenant_traced(&request);
        let args = asof_join_from_proto(request.into_inner())?;

        let record = scoped(&self.session, tenant, || async {
            self.session
                .asof_join(&args.spine, &args.facts, &args.spec)
                .await
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(record.into()))
    }

    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn recompute(
        &self,
        request: Request<RecomputeRequest>,
    ) -> Result<Response<ProtoRecomputeReport>, Status> {
        let tenant = session_tenant_traced(&request);
        let args = recompute_from_proto(request.into_inner())?;

        // Resolve the target table and recompute inside the request's tenant
        // scope: the catalog read is tenant-filtered, so a peer that names another
        // tenant's table resolves no row and the recompute refuses — the same
        // boundary `verify_materialization` / `staleness` ride.
        let report = scoped(&self.session, tenant, || async {
            let record = self
                .session
                .catalog()
                .get_result_table(&args.table)
                .await?
                .ok_or_else(|| {
                    JammiError::Catalog(format!("Result table '{}' not found", args.table))
                })?;
            self.session.recompute(&record, args.cascade).await
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(recompute_report_to_proto(report)))
    }
}
