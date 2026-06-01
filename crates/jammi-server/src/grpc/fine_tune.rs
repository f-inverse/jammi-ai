//! `FineTuneService` gRPC implementation.
//!
//! Two verbs land on the wire: `StartFineTune` and `FineTuneStatus`. Like every
//! other engine-backed service, they are thin adapters over the
//! transport-agnostic [`Session`]/[`LocalSession`] abstraction (never raw
//! [`InferenceSession`] calls): proto in, one `Session::fine_tune` /
//! `Session::fine_tune_status` call, proto out. The service reimplements no
//! training, LoRA, or scan logic.
//!
//! `Session::fine_tune` returns a `FineTuneJobId`; the handler carries its
//! string into the response and the client polls `FineTuneStatus` by that id
//! until a terminal state. There is no progress stream — the abstraction
//! exposes none.
//!
//! The request's `FineTuneConfig` mirrors the engine's `FineTuneConfig` field
//! for field; the shared `TryFrom<pb::FineTuneConfig>` conversion in
//! [`jammi_ai::wire`] maps it (and its nested loss / enum types) onto the engine
//! struct, leaving the engine's defaults in place for any field a client did not
//! set (an absent `config` message → the engine default entirely). `task`
//! reuses `jammi.v1.inference.ModelTask` via the shared
//! [`jammi_ai::wire::model_task_from_proto`].
//!
//! Tenant scope is read from the request's [`crate::grpc::session::
//! SessionTenant`] extension (set upstream by the shared `TenantInterceptor`)
//! and applied via `with_tenant_scoped`, matching every other engine-backed
//! gRPC surface.

use std::sync::Arc;

use jammi_ai::fine_tune::FineTuneConfig;
use jammi_ai::local_session::FineTuneJobId;
use jammi_ai::session::InferenceSession;
use jammi_ai::wire::{method_from_proto, model_task_from_proto};
use jammi_ai::{LocalSession, Session};
use tonic::{Request, Response, Status};

use crate::grpc::proto::fine_tune as pb;
use crate::grpc::proto::fine_tune::fine_tune_service_server::FineTuneService;
use crate::grpc::wire::{map_engine_error, require_nonempty, scoped, session_tenant};

/// Server-side handler for the fine-tune gRPC surface. Holds a shared engine
/// session it wraps in a [`LocalSession`] per call to reach the unified
/// transport surface.
pub struct FineTuneServer {
    session: Arc<InferenceSession>,
}

impl FineTuneServer {
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
impl FineTuneService for FineTuneServer {
    async fn start_fine_tune(
        &self,
        request: Request<pb::StartFineTuneRequest>,
    ) -> Result<Response<pb::StartFineTuneResponse>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        require_nonempty(&req.source_id, "source_id")?;
        require_nonempty(&req.base_model, "base_model")?;
        if req.columns.is_empty() {
            return Err(Status::invalid_argument("columns is required"));
        }
        let method = method_from_proto(req.method)?;
        let task = model_task_from_proto(req.task)?;
        let config = req.config.map(FineTuneConfig::try_from).transpose()?;
        let session = self.local();

        let job_id = scoped(&self.session, tenant, || {
            session.fine_tune(
                &req.source_id,
                &req.base_model,
                &req.columns,
                method,
                task,
                config,
            )
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(pb::StartFineTuneResponse {
            job_id: job_id.0,
        }))
    }

    async fn fine_tune_status(
        &self,
        request: Request<pb::FineTuneStatusRequest>,
    ) -> Result<Response<pb::FineTuneStatusResponse>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        require_nonempty(&req.job_id, "job_id")?;
        let id = FineTuneJobId(req.job_id);
        let session = self.local();

        let status = scoped(&self.session, tenant, || session.fine_tune_status(&id))
            .await
            .map_err(map_engine_error)?;

        Ok(Response::new(pb::FineTuneStatusResponse { status }))
    }
}
