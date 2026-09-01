//! `TrainingService` gRPC implementation.
//!
//! Three verbs land on the wire: `StartTraining`, `TrainingStatus`, and
//! `ListTrainingJobs`. The service
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
//! reason. `ListTrainingJobs` reads the same lifecycle projection for every job
//! visible to the session tenant, most recent first — a listing of
//! `TrainingStatus` answers plus each job's submit-time identity, never a
//! progress surface. There is no progress stream — the abstraction exposes
//! none, and the engine persists run metrics only at job finalization, so a
//! mid-run row has no metric to read.
//!
//! Tenant scope is read from the request's [`crate::grpc::session::
//! SessionTenant`] extension (set upstream by the async tenant-binding layer)
//! and applied via [`scoped`], matching every other engine-backed gRPC surface.

use std::sync::Arc;

use jammi_ai::fine_tune::spec::TrainingSpec;
use jammi_ai::fine_tune::training_job::TrainingJob;
use jammi_ai::session::InferenceSession;
use jammi_ai::wire::training_spec_from_proto;
use tonic::{Request, Response, Status};

use crate::grpc::proto::training as pb;
use crate::grpc::proto::training::training_service_server::TrainingService;
use crate::grpc::wire::{map_engine_error, require_nonempty, scoped, session_tenant_traced};

/// Server-side handler for the training gRPC surface. Holds the shared engine
/// session it submits jobs against and reads job records back from.
pub struct TrainingServer {
    session: Arc<InferenceSession>,
}

impl TrainingServer {
    pub fn new(session: Arc<InferenceSession>) -> Self {
        Self { session }
    }

    /// Submit a decoded engine [`TrainingSpec`] on the request's session,
    /// returning the durable [`TrainingJob`] handle. Delegates to the shared
    /// [`InferenceSession::run_training_spec`] seam — the same dispatch the
    /// embedded binding drives — so both transports submit an identical job from
    /// an identical decode.
    async fn submit(&self, spec: TrainingSpec) -> Result<TrainingJob, jammi_db::error::JammiError> {
        self.session.run_training_spec(spec).await
    }
}

#[tonic::async_trait]
impl TrainingService for TrainingServer {
    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn start_training(
        &self,
        request: Request<pb::StartTrainingRequest>,
    ) -> Result<Response<pb::StartTrainingResponse>, Status> {
        let tenant = session_tenant_traced(&request);
        let spec = training_spec_from_proto(request.into_inner())?;

        let job = scoped(&self.session, tenant, || self.submit(spec))
            .await
            .map_err(map_engine_error)?;

        Ok(Response::new(pb::StartTrainingResponse {
            job_id: job.job_id,
            model_id: job.model_id,
        }))
    }

    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn training_status(
        &self,
        request: Request<pb::TrainingStatusRequest>,
    ) -> Result<Response<pb::TrainingStatusResponse>, Status> {
        let tenant = session_tenant_traced(&request);
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
            // The catalog's `training_jobs.metrics` column, relayed verbatim —
            // the SAME raw JSON blob the embedded `TrainingJob.metrics()` reads
            // off this record, with the same absent-until-recorded semantics
            // (`None` here maps to `optional` unset, matching field presence
            // rather than an empty string).
            metrics_json: record.metrics,
            // The catalog's `training_jobs.acceleration_report` column, relayed
            // verbatim — the SAME opaque, self-describing JSON blob the
            // embedded surface's catalog record read returns off this row
            // (esc-075). `None` (SQL `NULL`, a pre-migration-026 row) maps to
            // `optional` unset here, never an empty string; any present blob
            // (whose `"state"` vocabulary is owned by the producer, not this
            // handler) passes through byte-for-byte, matching `metrics_json`'s
            // presence contract.
            acceleration_report_json: record.acceleration_report,
        }))
    }

    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn list_training_jobs(
        &self,
        request: Request<pb::ListTrainingJobsRequest>,
    ) -> Result<Response<pb::ListTrainingJobsResponse>, Status> {
        let tenant = session_tenant_traced(&request);

        let records = scoped(&self.session, tenant, || {
            self.session.catalog().list_training_jobs()
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(pb::ListTrainingJobsResponse {
            jobs: records
                .into_iter()
                .map(|record| pb::TrainingJobSummary {
                    job_id: record.job_id,
                    kind: record.kind,
                    status: record.status,
                    base_model_id: record.base_model_id,
                    // Empty until the job completes, matching `TrainingStatus`.
                    output_model_id: record.output_model_id.unwrap_or_default(),
                    created_at: record.created_at,
                    // Non-empty exactly when status is "failed", matching
                    // `TrainingStatus`.
                    error: record.error_message.unwrap_or_default(),
                })
                .collect(),
        }))
    }
}
