//! `InferenceService` gRPC implementation.
//!
//! One verb lands on the wire: `Infer`. Like the embedding verbs, it is a thin
//! adapter over the transport-agnostic [`Session`]/[`LocalSession`] abstraction
//! (never raw [`InferenceSession`] calls): proto in, one `Session::infer` call,
//! proto out. The service reimplements no scan or forward logic.
//!
//! `Session::infer` returns `Vec<RecordBatch>`; the handler carries them as one
//! Arrow IPC stream in the response's `ArrowBatch` — the same Flight-IPC
//! pairing `TriggerService` uses, encoded through the shared
//! [`crate::grpc::wire::encode_ipc_stream`] helper.
//!
//! Tenant scope is read from the request's [`crate::grpc::session::
//! SessionTenant`] extension (set upstream by the shared `TenantInterceptor`)
//! and applied via `with_tenant_scoped`, matching every other engine-backed
//! gRPC surface.

use std::sync::Arc;

use jammi_ai::session::InferenceSession;
use jammi_ai::{LocalSession, Session};
use tonic::{Request, Response, Status};

use crate::grpc::proto::inference::inference_service_server::InferenceService;
use crate::grpc::proto::inference::{InferRequest, InferResponse};
use crate::grpc::proto::trigger::ArrowBatch;
use crate::grpc::wire::{
    encode_ipc_stream, map_engine_error, model_task_from_proto, require_nonempty, scoped,
    session_tenant,
};

/// Server-side handler for the inference gRPC surface. Holds a shared engine
/// session it wraps in a [`LocalSession`] per call to reach the unified
/// transport surface.
pub struct InferenceServer {
    session: Arc<InferenceSession>,
}

impl InferenceServer {
    pub fn new(session: Arc<InferenceSession>) -> Self {
        Self { session }
    }

    /// A [`Session`] over the shared engine. Wrapping is an `Arc` clone; the
    /// resulting `LocalSession` delegates to the same engine, so a tenant scope
    /// installed by [`scoped`] (a task-local on this task) is observed by the
    /// call made through it.
    fn local(&self) -> Session {
        Session::Local(LocalSession::new(Arc::clone(&self.session)))
    }
}

#[tonic::async_trait]
impl InferenceService for InferenceServer {
    async fn infer(
        &self,
        request: Request<InferRequest>,
    ) -> Result<Response<InferResponse>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        require_nonempty(&req.source_id, "source_id")?;
        require_nonempty(&req.model_id, "model_id")?;
        require_nonempty(&req.key_column, "key_column")?;
        if req.columns.is_empty() {
            return Err(Status::invalid_argument("columns is required"));
        }
        let task = model_task_from_proto(req.task)?;
        let session = self.local();

        let batches = scoped(&self.session, tenant, || {
            session.infer(
                &req.source_id,
                &req.model_id,
                task,
                &req.columns,
                &req.key_column,
            )
        })
        .await
        .map_err(map_engine_error)?;

        // Carry the result rows as one self-describing IPC stream. An empty
        // result (empty source) has no schema to encode, so it round-trips as
        // an empty `ArrowBatch`.
        let result = match batches.first() {
            Some(first) => {
                let body = encode_ipc_stream(&first.schema(), &batches)?;
                ArrowBatch {
                    data_header: Vec::new(),
                    data_body: body,
                    app_metadata: Vec::new(),
                }
            }
            None => ArrowBatch::default(),
        };

        Ok(Response::new(InferResponse {
            result: Some(result),
        }))
    }
}
