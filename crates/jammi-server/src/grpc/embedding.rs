//! `EmbeddingService` gRPC implementation.
//!
//! Each verb is a thin wire adapter over the transport-agnostic
//! [`Session`] abstraction (never raw [`InferenceSession`]
//! calls): proto in, one `Session` method, proto out.
//!
//! * `GenerateEmbeddings` — scan a source's `columns`, run the modality's
//!   tower, persist one vector per row (peer of `Session::generate_embeddings`,
//!   keyed by `Modality`).
//! * `EncodeQuery` — encode a single query into one vector with the modality's
//!   tower (peer of `Session::encode_query`).
//! * `Search` — nearest-neighbor search over a source's embedding table, by a
//!   precomputed vector or an existing row's key (peer of the abstraction's
//!   flat `Session::search`). This is the embedding-consumption verb on the
//!   gRPC-web transport edge runtimes reach; it adds no new consumption model.
//!
//! These are the data-plane compute verbs; source registration and model
//! introspection — the control-plane catalog surface this lane reads — live on
//! [`CatalogService`](crate::grpc::catalog).
//!
//! The abstraction dispatches each `Modality` onto the engine's concrete
//! tower method; this module reimplements no embedding logic. Modality and
//! input are validated at the wire edge: an unspecified modality and a
//! text/bytes-vs-modality mismatch are rejected with `invalid_argument`.
//!
//! Tenant scope is read from the request's [`SessionTenant`] extension (set
//! upstream by the async tenant-binding layer, [`crate::tenant_resolver_layer`]) and applied to the
//! call via [`crate::grpc::wire::scoped`] — the same task-local the engine the
//! [`Session`] wraps observes — matching how the Flight SQL and Trigger
//! surfaces resolve their tenant.
//!
//! [`SessionTenant`]: crate::grpc::session::SessionTenant

use std::sync::Arc;

use jammi_ai::session::InferenceSession;
use jammi_ai::Session;
use jammi_wire::result_rows_to_proto;
use tonic::{Request, Response, Status};

use crate::grpc::proto::embedding::embedding_service_server::EmbeddingService;
use crate::grpc::proto::embedding::{
    CompactEmbeddingsRequest, EncodeQueryRequest, EncodeQueryResponse, ExpireVersionsRequest,
    ExpiryReport, GenerateEmbeddingsRequest, ImportEmbeddingsRequest, LexicalSearchRequest,
    RefreshEmbeddingsRequest, RefreshReport, ResultTable, SearchRequest, SearchResponse,
};
use crate::grpc::wire::{map_engine_error, scoped, session_tenant_traced};

/// Server-side handler for the embedding gRPC surface. Holds a shared engine
/// session it wraps in a [`Session`] per call to reach the unified
/// transport surface.
pub struct EmbeddingServer {
    session: Arc<InferenceSession>,
}

impl EmbeddingServer {
    pub fn new(session: Arc<InferenceSession>) -> Self {
        Self { session }
    }

    /// A [`Session`] over the shared engine. Wrapping is an `Arc` clone; the
    /// resulting `Session` delegates to the same engine, so a tenant scope
    /// installed by [`scoped`] (a task-local on this task) is observed by the
    /// call made through it.
    fn local(&self) -> Session {
        Session::new(Arc::clone(&self.session))
    }
}

#[tonic::async_trait]
impl EmbeddingService for EmbeddingServer {
    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn generate_embeddings(
        &self,
        request: Request<GenerateEmbeddingsRequest>,
    ) -> Result<Response<ResultTable>, Status> {
        let tenant = session_tenant_traced(&request);
        // Decode through the shared `jammi_ai::wire` seam — the same decode the
        // embedded binding's `_generate_embeddings_proto` drives — so both
        // transports validate and submit an identical request.
        let request = jammi_ai::wire::generate_embeddings_from_proto(request.into_inner())?;
        let session = self.local();

        let (record, outcome) = scoped(&self.session, tenant, || {
            session.generate_embeddings(request)
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(jammi_wire::result_table_with_outcome(
            record,
            jammi_wire::cache_outcome_to_proto(&outcome),
        )))
    }

    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn import_embeddings(
        &self,
        request: Request<ImportEmbeddingsRequest>,
    ) -> Result<Response<ResultTable>, Status> {
        let tenant = session_tenant_traced(&request);
        // Decode through the shared `jammi_ai::wire` seam — the same decode the
        // embedded binding's `_import_embeddings_proto` drives — so both
        // transports validate and submit an identical request.
        let args = jammi_ai::wire::import_embeddings_from_proto(request.into_inner())?;
        let session = self.local();

        let record = scoped(&self.session, tenant, || {
            session.import_embeddings(
                &args.source_id,
                &args.model_id,
                &args.vectors_url,
                &args.key_column,
                &args.text_columns,
                args.dimensions,
            )
        })
        .await
        .map_err(map_engine_error)?;

        // A fresh import always computes (no producer cache); carry the honest
        // `COMPUTED` outcome so the wire shape matches `GenerateEmbeddings`.
        Ok(Response::new(jammi_wire::result_table_with_outcome(
            record,
            jammi_wire::cache_outcome_to_proto(&jammi_db::store::CacheOutcome::Computed),
        )))
    }

    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn encode_query(
        &self,
        request: Request<EncodeQueryRequest>,
    ) -> Result<Response<EncodeQueryResponse>, Status> {
        let tenant = session_tenant_traced(&request);
        // Decode through the shared `jammi_ai::wire` seam — the same decode the
        // embedded binding's `_encode_query_proto` drives — so both transports
        // validate and submit an identical request.
        let args = jammi_ai::wire::encode_query_from_proto(request.into_inner())?;
        let session = self.local();

        let embedding = scoped(&self.session, tenant, || {
            session.encode_query(&args.model_id, args.input, args.modality, args.dimensions)
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(EncodeQueryResponse { embedding }))
    }

    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn refresh_embeddings(
        &self,
        request: Request<RefreshEmbeddingsRequest>,
    ) -> Result<Response<RefreshReport>, Status> {
        let tenant = session_tenant_traced(&request);
        // The same decode seam the embedded binding's `_refresh_embeddings_proto`
        // drives, so both transports validate one request shape.
        let args = jammi_ai::wire::refresh_embeddings_from_proto(request.into_inner())?;
        let session = self.local();
        let report = scoped(&self.session, tenant, || {
            session.refresh_embeddings(&args.table, args.options)
        })
        .await
        .map_err(map_engine_error)?;
        Ok(Response::new(
            jammi_wire::embedding_refresh::refresh_report_to_proto(&report),
        ))
    }

    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn compact_embeddings(
        &self,
        request: Request<CompactEmbeddingsRequest>,
    ) -> Result<Response<RefreshReport>, Status> {
        let tenant = session_tenant_traced(&request);
        let table = jammi_ai::wire::compact_embeddings_from_proto(request.into_inner())?;
        let session = self.local();
        let report = scoped(&self.session, tenant, || session.compact_embeddings(&table))
            .await
            .map_err(map_engine_error)?;
        Ok(Response::new(
            jammi_wire::embedding_refresh::refresh_report_to_proto(&report),
        ))
    }

    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn expire_versions(
        &self,
        request: Request<ExpireVersionsRequest>,
    ) -> Result<Response<ExpiryReport>, Status> {
        let tenant = session_tenant_traced(&request);
        let args = jammi_ai::wire::expire_versions_from_proto(request.into_inner())?;
        let session = self.local();
        let report = scoped(&self.session, tenant, || {
            session.expire_versions(&args.table, args.before)
        })
        .await
        .map_err(map_engine_error)?;
        Ok(Response::new(
            jammi_wire::embedding_refresh::expiry_report_to_proto(&report),
        ))
    }

    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let tenant = session_tenant_traced(&request);
        // Decode the request through the shared `jammi_ai::wire` seam — the same
        // decode the embedded binding's `_search_proto` drives — and return the
        // engine's hydrated rows as they are, so both transports hand the caller
        // one result.
        let request = jammi_ai::wire::search_from_proto(request.into_inner())?;
        let session = self.local();
        let batches = scoped(&self.session, tenant, || session.search(request))
            .await
            .map_err(map_engine_error)?;
        Ok(Response::new(SearchResponse {
            result: Some(result_rows_to_proto(&batches)?),
        }))
    }

    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn lexical_search(
        &self,
        request: Request<LexicalSearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let tenant = session_tenant_traced(&request);
        let request = jammi_ai::wire::lexical_search_from_proto(request.into_inner())?;
        let session = self.local();
        let batches = scoped(&self.session, tenant, || session.lexical_search(request))
            .await
            .map_err(map_engine_error)?;
        Ok(Response::new(SearchResponse {
            result: Some(result_rows_to_proto(&batches)?),
        }))
    }
}
