//! `TrainingService` gRPC implementation.
//!
//! Two verbs land on the wire: `StartTraining` and `TrainingStatus`. The service
//! serves all three engine training kinds (`fine_tune`, `graph_fine_tune`,
//! `context_predictor`) behind one verb: `StartTraining` carries a full
//! `TrainingSpec` oneof, the handler decodes it to the engine `TrainingSpec` via
//! the shared [`jammi_ai::wire`] conversion, and dispatches to the matching
//! engine submit verb — returning the durable job id and the deterministic
//! output model id. The service reimplements no training, LoRA, sampling, or scan
//! logic.
//!
//! Each engine submit returns a `TrainingJob` (job id + output model id); the
//! handler carries both into the response. `TrainingStatus` reads the job record
//! back and returns its status, output model id, and — when the job failed — the
//! error message, so a remote `wait()` can retrieve the result and a failure
//! reason. There is no progress stream — the abstraction exposes none.
//!
//! Tenant scope is read from the request's [`crate::grpc::session::
//! SessionTenant`] extension (set upstream by the shared `TenantInterceptor`)
//! and applied via [`scoped`], matching every other engine-backed gRPC surface.

use std::sync::Arc;

use jammi_ai::fine_tune::spec::TrainingSpec;
use jammi_ai::fine_tune::training_job::TrainingJob;
use jammi_ai::session::InferenceSession;
use jammi_ai::wire::training_spec_from_proto;
use tonic::{Request, Response, Status};

use crate::grpc::proto::training as pb;
use crate::grpc::proto::training::training_service_server::TrainingService;
use crate::grpc::wire::{map_engine_error, require_nonempty, scoped, session_tenant};

/// Server-side handler for the training gRPC surface. Holds the shared engine
/// session it submits jobs against and reads job records back from.
pub struct TrainingServer {
    session: Arc<InferenceSession>,
}

impl TrainingServer {
    pub fn new(session: Arc<InferenceSession>) -> Self {
        Self { session }
    }

    /// Dispatch a decoded engine [`TrainingSpec`] to the matching submit verb,
    /// returning the durable [`TrainingJob`] handle. The spec variant is the
    /// engine's source of truth for which verb produced the job, so the match
    /// arms re-destructure it back into the verb's positional arguments — the
    /// inverse of the engine's own spec construction.
    async fn submit(&self, spec: TrainingSpec) -> Result<TrainingJob, jammi_db::error::JammiError> {
        match spec {
            TrainingSpec::FineTune {
                source,
                columns,
                method,
                task,
                common,
            } => {
                self.session
                    .fine_tune(
                        &source,
                        &common.base_model,
                        &columns,
                        method,
                        task,
                        Some(common.config),
                    )
                    .await
            }
            TrainingSpec::GraphFineTune {
                sources,
                sample_config,
                common,
            } => {
                self.session
                    .fine_tune_graph(
                        &sources,
                        &common.base_model,
                        sample_config,
                        Some(common.config),
                    )
                    .await
            }
            TrainingSpec::ContextPredictor {
                source,
                predictor_spec,
            } => {
                self.session
                    .train_context_predictor(&source, &predictor_spec)
                    .await
            }
        }
    }
}

#[tonic::async_trait]
impl TrainingService for TrainingServer {
    async fn start_training(
        &self,
        request: Request<pb::StartTrainingRequest>,
    ) -> Result<Response<pb::StartTrainingResponse>, Status> {
        let tenant = session_tenant(&request);
        let spec = training_spec_from_proto(request.into_inner())?;

        let job = scoped(&self.session, tenant, || self.submit(spec))
            .await
            .map_err(map_engine_error)?;

        Ok(Response::new(pb::StartTrainingResponse {
            job_id: job.job_id,
            model_id: job.model_id,
        }))
    }

    async fn training_status(
        &self,
        request: Request<pb::TrainingStatusRequest>,
    ) -> Result<Response<pb::TrainingStatusResponse>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        require_nonempty(&req.job_id, "job_id")?;

        let record = scoped(&self.session, tenant, || {
            self.session.catalog().get_training_job(&req.job_id)
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(pb::TrainingStatusResponse {
            status: record.status,
            // The output model id is set once the job completes; a queued /
            // running job carries the empty string here.
            model_id: record.output_model_id.unwrap_or_default(),
            // The failure message is surfaced only on a failed job; empty
            // otherwise so a remote `wait()` reads it exactly when status is
            // "failed".
            error: record.error_message.unwrap_or_default(),
        }))
    }
}
