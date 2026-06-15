//! `InferenceService` gRPC implementation.
//!
//! One verb lands on the wire: `Infer`. Like the embedding verbs, it is a thin
//! adapter over the transport-agnostic [`Session`] abstraction
//! (never raw [`InferenceSession`] calls): proto in, one `Session::infer` call,
//! proto out. The service reimplements no scan or forward logic.
//!
//! `Session::infer` returns `Vec<RecordBatch>`; the handler carries them as one
//! Arrow IPC stream in the response's `ArrowBatch` — the same Flight-IPC
//! pairing `TriggerService` uses, encoded through the shared
//! [`jammi_wire::infer_result_to_proto`] helper.
//!
//! Tenant scope is read from the request's [`crate::grpc::session::
//! SessionTenant`] extension (set upstream by the shared `TenantInterceptor`)
//! and applied via `with_tenant_scoped`, matching every other engine-backed
//! gRPC surface.

use std::sync::Arc;

use jammi_ai::pipeline::context_predictor::{
    ContextServeOptions, ContextServeSource, PredictionWithProvenance,
};
use jammi_ai::pipeline::context_set::HybridMerge;
use jammi_ai::session::InferenceSession;
use jammi_ai::wire::{edge_gather_from_proto, predicted_distribution_to_proto};
use jammi_ai::Session;
use jammi_wire::infer_result_to_proto;
use tonic::{Request, Response, Status};

use crate::grpc::proto::inference::inference_service_server::InferenceService;
use crate::grpc::proto::inference::{InferRequest, InferResponse, PredictRequest, PredictResponse};
use crate::grpc::wire::{map_engine_error, require_nonempty, scoped, session_tenant_traced};

/// Server-side handler for the inference gRPC surface. Holds a shared engine
/// session it wraps in a [`Session`] per call to reach the unified
/// transport surface.
pub struct InferenceServer {
    session: Arc<InferenceSession>,
}

impl InferenceServer {
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
impl InferenceService for InferenceServer {
    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn infer(
        &self,
        request: Request<InferRequest>,
    ) -> Result<Response<InferResponse>, Status> {
        let tenant = session_tenant_traced(&request);
        // Decode through the shared `jammi_ai::wire` seam — the same decode the
        // embedded binding's `_infer_proto` drives — so both transports validate
        // and submit an identical request.
        let args = jammi_ai::wire::infer_from_proto(request.into_inner())?;
        let session = self.local();

        let batches = scoped(&self.session, tenant, || {
            session.infer(
                &args.source_id,
                &args.model,
                args.task,
                &args.columns,
                &args.key_column,
            )
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(InferResponse {
            result: Some(infer_result_to_proto(batches)?),
        }))
    }

    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn predict(
        &self,
        request: Request<PredictRequest>,
    ) -> Result<Response<PredictResponse>, Status> {
        let tenant = session_tenant_traced(&request);
        let req = request.into_inner();
        require_nonempty(&req.model_id, "model_id")?;
        require_nonempty(&req.source, "source")?;
        require_nonempty(&req.target_key, "target_key")?;

        // Reconstruct the live-context source the embed binding builds: an absent
        // edge gather is ANN-only; a gather with no `hybrid_ann_k` is declared-edge
        // only; a gather with `hybrid_ann_k` is the union of the two.
        let gather = edge_gather_from_proto(req.edges)?;
        let serve_source = match (gather, req.hybrid_ann_k) {
            (None, _) => ContextServeSource::Ann,
            (Some(edges), None) => ContextServeSource::Edges(edges),
            (Some(edges), Some(ann_k)) => ContextServeSource::Hybrid {
                ann_k: ann_k as usize,
                edges,
                merge: HybridMerge::Union,
            },
        };
        let options = ContextServeOptions {
            source: serve_source,
            split: req.split,
        };

        let prediction: PredictionWithProvenance = scoped(&self.session, tenant, || async {
            let served = self
                .session
                .load_context_predictor(&req.model_id, &req.source, options)
                .await?;
            self.session
                .predict_with_context_predictor_provenanced(&served, &req.target_key)
                .await
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(PredictResponse {
            distribution: Some(predicted_distribution_to_proto(&prediction.distribution)),
            source: context_source_tag(prediction.source).to_string(),
            context_ref: prediction.context_keys,
        }))
    }
}

/// The string tag for a context's assembly fact ("ann" / "edges" / "hybrid"),
/// surfaced on a prediction so a remote consumer can see how the context was
/// assembled — the same vocabulary the embed binding's dict exposes.
fn context_source_tag(kind: jammi_ai::pipeline::context_set::ContextSourceKind) -> &'static str {
    use jammi_ai::pipeline::context_set::ContextSourceKind;
    match kind {
        ContextSourceKind::Ann => "ann",
        ContextSourceKind::Edges => "edges",
        ContextSourceKind::Hybrid => "hybrid",
    }
}
