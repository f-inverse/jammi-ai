//! `EmbeddingService` gRPC implementation.
//!
//! Each verb is a thin wire adapter over the transport-agnostic
//! [`Session`] abstraction (never raw [`InferenceSession`]
//! calls): proto in, one `Session` method, proto out.
//!
//! * `GenerateEmbeddings` — scan a source's `columns`, run the modality's
//!   tower, persist one vector per row (peer of `Session::generate_embeddings`,
//!   keyed by [`Modality`]).
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
//! The abstraction dispatches each [`Modality`] onto the engine's concrete
//! tower method; this module reimplements no embedding logic. Modality and
//! input are validated at the wire edge: an unspecified modality and a
//! text/bytes-vs-modality mismatch are rejected with `invalid_argument`.
//!
//! Tenant scope is read from the request's [`SessionTenant`] extension (set
//! upstream by [`crate::grpc::session::TenantInterceptor`]) and applied to the
//! call via [`crate::grpc::wire::scoped`] — the same task-local the engine the
//! [`Session`] wraps observes — matching how the Flight SQL and Trigger
//! surfaces resolve their tenant.
//!
//! [`SessionTenant`]: crate::grpc::session::SessionTenant

use std::sync::Arc;

use jammi_ai::session::InferenceSession;
use jammi_ai::Session;
use tonic::{Request, Response, Status};

use std::collections::HashMap;

use arrow::array::{Array, Float32Array, RecordBatch, StringArray};
use arrow::util::display::{ArrayFormatter, FormatOptions};

use crate::grpc::proto::embedding::embedding_service_server::EmbeddingService;
use crate::grpc::proto::embedding::{
    EncodeQueryRequest, EncodeQueryResponse, GenerateEmbeddingsRequest, ResultTable, SearchHit,
    SearchRequest, SearchResponse,
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
        let args = jammi_ai::wire::generate_embeddings_from_proto(request.into_inner())?;
        let session = self.local();

        let record = scoped(&self.session, tenant, || {
            session.generate_embeddings(
                &args.source_id,
                &args.model_id,
                &args.columns,
                &args.key_column,
                args.modality,
            )
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(ResultTable::from(record)))
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
            session.encode_query(&args.model_id, args.input, args.modality)
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(EncodeQueryResponse { embedding }))
    }

    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let tenant = session_tenant_traced(&request);
        // Decode the request through the shared `jammi_ai::wire` seam — the same
        // decode the embedded binding's `_search_proto` drives — so both
        // transports validate and submit an identical request. Only the request
        // is collapsed: the response is transport-specific (wire hits here, Arrow
        // in the embedded binding), so the hit projection stays in this handler.
        let mut request = jammi_ai::wire::search_from_proto(request.into_inner())?;
        // The client's `select` is what each hit's `columns` map carries; keep it
        // before expanding the engine projection.
        let select = std::mem::take(&mut request.select);
        // The abstraction projects exactly the requested columns; the handler
        // needs `_row_id` + `similarity` for every hit's key and score, so add
        // them when a non-empty select would otherwise drop them. An empty select
        // keeps every hydrated column (key + score included).
        request.select = search_select(&select);
        let session = self.local();

        let batches = scoped(&self.session, tenant, || session.search(request))
            .await
            .map_err(map_engine_error)?;

        let hits = batches_to_hits(&batches, &select)?;
        Ok(Response::new(SearchResponse { hits }))
    }
}

/// The projection the abstraction's `search` runs for a client `select`. An
/// empty `select` projects nothing (all hydrated columns survive, so key and
/// score are present). A non-empty `select` projects the requested columns
/// **plus** `_row_id` and `similarity` — the handler always needs those to
/// build each hit's key and score, even when the client did not list them.
fn search_select(select: &[String]) -> Vec<String> {
    if select.is_empty() {
        return Vec::new();
    }
    let mut columns: Vec<String> = vec!["_row_id".to_string(), "similarity".to_string()];
    for name in select {
        if name != "_row_id" && name != "similarity" {
            columns.push(name.clone());
        }
    }
    columns
}

/// Map each result row to a [`SearchHit`]: `_row_id` → key, `similarity` →
/// score, and each requested `select` column stringified into `columns`.
///
/// `select` columns are read from the projected batch via the type-general
/// Arrow formatter, so any scalar column the engine returns is carried on the
/// wire without a per-dtype branch here.
fn batches_to_hits(batches: &[RecordBatch], select: &[String]) -> Result<Vec<SearchHit>, Status> {
    let mut hits = Vec::new();
    let format = FormatOptions::default();
    for batch in batches {
        let keys = column_as::<StringArray>(batch, "_row_id")?;
        let scores = column_as::<Float32Array>(batch, "similarity")?;
        let formatters: Vec<(String, ArrayFormatter)> = select
            .iter()
            .map(|name| {
                let array = batch.column_by_name(name).ok_or_else(|| {
                    Status::invalid_argument(format!("select column '{name}' not in results"))
                })?;
                let formatter = ArrayFormatter::try_new(array.as_ref(), &format)
                    .map_err(|e| Status::internal(format!("format column '{name}': {e}")))?;
                Ok((name.clone(), formatter))
            })
            .collect::<Result<_, Status>>()?;

        for row in 0..batch.num_rows() {
            let columns: HashMap<String, String> = formatters
                .iter()
                .map(|(name, fmt)| (name.clone(), fmt.value(row).to_string()))
                .collect();
            hits.push(SearchHit {
                key: keys.value(row).to_string(),
                score: scores.value(row),
                columns,
            });
        }
    }
    Ok(hits)
}

/// Downcast a named column to a concrete Arrow array, mapping a missing or
/// wrong-typed column to an internal [`Status`] (the search plan owns these
/// columns, so a mismatch is a server-side invariant break, not a bad input).
fn column_as<'a, A: Array + 'static>(batch: &'a RecordBatch, name: &str) -> Result<&'a A, Status> {
    batch
        .column_by_name(name)
        .ok_or_else(|| Status::internal(format!("search result missing '{name}' column")))?
        .as_any()
        .downcast_ref::<A>()
        .ok_or_else(|| {
            Status::internal(format!("search result '{name}' column has unexpected type"))
        })
}
