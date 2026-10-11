//! `JobService` gRPC implementation: the durable job queue on the wire.
//!
//! `SubmitJob` submits a training job (the three training kinds of its
//! oneof); `JobStatus` and the resumable server-streaming `WaitJob` read any
//! job by id; `ListJobs`, `CancelJob`, `ListWorkers` and `PruneJobs` complete
//! the set. Every compute verb is synchronous on its own service (e.g.
//! `PipelineService::BuildNeighborGraph`) and answers with its own response,
//! byte-identical to what a `jobs` row of the same kind resolves to — both
//! run through the one `execute_compute` dispatcher (`jammi_ai::jobs`).
//!
//! Tenant scope is read from the request's [`crate::grpc::session::
//! SessionTenant`] extension (set upstream by the async tenant-binding layer)
//! and applied via [`scoped`], matching every other engine-backed gRPC
//! surface. A job row carries the submitting tenant (`Catalog::get_job`'s own
//! `WHERE tenant_id = $t OR tenant_id IS NULL` gate); a peer naming an
//! unresolvable job id gets `NOT_FOUND`, never `PERMISSION_DENIED` — this
//! handler never distinguishes "absent" from "exists under another tenant",
//! matching every other tenant-scoped read in this codebase.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use futures::Stream;
use jammi_ai::fine_tune::spec::TrainingSpec;
use jammi_ai::jobs::{JobHandle, JobResult as EngineJobResult};
use jammi_ai::session::InferenceSession;
use jammi_ai::wire::training_spec_from_proto;
use jammi_db::catalog::jobs_repo::JobRecord;
use jammi_db::error::JammiError;
use jammi_db::TenantId;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use crate::grpc::proto::job as pb;
use crate::grpc::proto::job::job_service_server::JobService;
use crate::grpc::wire::{map_engine_error, require_nonempty, scoped, session_tenant_traced};

/// `WaitJob`'s server-side poll interval.
const WAIT_JOB_POLL: Duration = Duration::from_millis(100);
/// Bounded so a slow/blocked receiver applies backpressure rather than
/// buffering unboundedly; small because a frame is produced at most once per
/// [`WAIT_JOB_POLL`] tick.
const WAIT_JOB_BUFFER: usize = 8;

/// Server-side handler for the `JobService` gRPC surface. Holds the shared
/// engine session it submits jobs against and reads job records back from.
pub struct JobServer {
    session: Arc<InferenceSession>,
}

impl JobServer {
    pub fn new(session: Arc<InferenceSession>) -> Self {
        Self { session }
    }

    /// Submit a decoded engine [`TrainingSpec`] on the request's session,
    /// returning the durable [`JobHandle`] — the caller's own on a
    /// fresh submission, or the pre-existing job's handle when
    /// `idempotency_key` is non-empty and collides with a still-known prior
    /// submission (migration 030's durable per-tenant dedupe; see
    /// [`InferenceSession::run_training_spec_deduped`]). Delegates to that
    /// seam — the same dispatch the embedded binding drives (with an empty
    /// key, which never dedupes) — so both transports submit an identical
    /// job from an identical decode.
    async fn submit(
        &self,
        spec: TrainingSpec,
        idempotency_key: &str,
    ) -> Result<JobHandle, JammiError> {
        let key = (!idempotency_key.is_empty()).then_some(idempotency_key);
        self.session.run_training_spec_deduped(spec, key).await
    }
}

#[tonic::async_trait]
impl JobService for JobServer {
    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn submit_job(
        &self,
        request: Request<pb::SubmitJobRequest>,
    ) -> Result<Response<pb::SubmitJobResponse>, Status> {
        let tenant = session_tenant_traced(&request);
        let req = request.into_inner();
        let idempotency_key = req.idempotency_key.clone();

        // Refused at the edge, before the spec ever decodes: a typed
        // `InvalidArgument` naming the bound, never the (potentially
        // sensitive) key value itself.
        // `Catalog::submit_job_deduped` asserts the same bound again
        // (defence in depth) for every OTHER caller of that entry point.
        if idempotency_key.len() > jammi_db::catalog::jobs_repo::MAX_IDEMPOTENCY_KEY_BYTES {
            return Err(map_engine_error(JammiError::InvalidArgument {
                argument: "idempotency_key".into(),
                reason: format!(
                    "exceeds the maximum length of {} bytes",
                    jammi_db::catalog::jobs_repo::MAX_IDEMPOTENCY_KEY_BYTES
                ),
            }));
        }

        let spec = training_spec_from_proto(req)?;

        // Durable per-tenant dedupe (migration 030) lives inside the engine
        // seam this delegates to — `submit` reads back the winning row's own
        // handle when `idempotency_key` collides with a still-known prior
        // submission, so there is no separate memory-map lookup here (and
        // nothing for a process restart or a pruned row to forget).
        let job = scoped(&self.session, tenant, || {
            self.submit(spec, &idempotency_key)
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(pb::SubmitJobResponse {
            job_id: job.job_id().to_string(),
            kind: jammi_wire::job_kind_to_proto(job.kind()) as i32,
            output_model_id: job.output_model_id().map(str::to_string),
        }))
    }

    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn job_status(
        &self,
        request: Request<pb::JobStatusRequest>,
    ) -> Result<Response<pb::JobStatusResponse>, Status> {
        let tenant = session_tenant_traced(&request);
        let req = request.into_inner();
        require_nonempty(&req.job_id, "job_id")?;

        let record = read_job(&self.session, tenant, &req.job_id).await?;
        Ok(Response::new(job_status_response_from_record(&record)?))
    }

    type WaitJobStream = Pin<Box<dyn Stream<Item = Result<pb::JobEvent, Status>> + Send + 'static>>;

    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn wait_job(
        &self,
        request: Request<pb::JobHandle>,
    ) -> Result<Response<Self::WaitJobStream>, Status> {
        let tenant = session_tenant_traced(&request);
        let req = request.into_inner();
        require_nonempty(&req.job_id, "job_id")?;

        // Validate the id resolves before opening the stream, so a bad id
        // fails the unary-shaped call rather than the first stream frame.
        let first = read_job(&self.session, tenant, &req.job_id).await?;

        let (tx, rx) = mpsc::channel::<Result<pb::JobEvent, Status>>(WAIT_JOB_BUFFER);
        let session = Arc::clone(&self.session);
        let job_id = req.job_id.clone();
        tokio::spawn(async move {
            let mut next = Ok(first);
            loop {
                let (frame, last) = match next {
                    Ok(record) if record.status.is_terminal() => (
                        job_status_response_from_record(&record).map(|done| pb::JobEvent {
                            event: Some(pb::job_event::Event::Done(Box::new(done))),
                        }),
                        true,
                    ),
                    Ok(record) => (
                        Ok(pb::JobEvent {
                            event: Some(pb::job_event::Event::Progress(
                                jammi_wire::job_progress_to_proto(record.progress),
                            )),
                        }),
                        false,
                    ),
                    Err(status) => (Err(status), true),
                };
                // A send fails only when the receiver dropped — a client
                // disconnect. The job itself is untouched; a reconnect
                // resumes the wait.
                if tx.send(frame).await.is_err() || last {
                    return;
                }
                tokio::time::sleep(WAIT_JOB_POLL).await;
                next = read_job(&session, tenant, &job_id).await;
            }
        });

        Ok(Response::new(
            Box::pin(ReceiverStream::new(rx)) as Self::WaitJobStream
        ))
    }

    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn list_jobs(
        &self,
        request: Request<pb::ListJobsRequest>,
    ) -> Result<Response<pb::ListJobsResponse>, Status> {
        let tenant = session_tenant_traced(&request);

        let records = scoped(&self.session, tenant, || self.session.catalog().list_jobs())
            .await
            .map_err(map_engine_error)?;

        Ok(Response::new(pb::ListJobsResponse {
            jobs: records.into_iter().map(job_summary_from_record).collect(),
        }))
    }

    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn cancel_job(
        &self,
        request: Request<pb::CancelJobRequest>,
    ) -> Result<Response<pb::CancelJobResponse>, Status> {
        let tenant = session_tenant_traced(&request);
        let req = request.into_inner();
        require_nonempty(&req.job_id, "job_id")?;

        let cancelled = scoped(&self.session, tenant, || {
            self.session.catalog().cancel_request(&req.job_id)
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(pb::CancelJobResponse { cancelled }))
    }

    #[tracing::instrument(skip(self, _request))]
    async fn list_workers(
        &self,
        _request: Request<pb::ListWorkersRequest>,
    ) -> Result<Response<pb::ListWorkersResponse>, Status> {
        let records = self
            .session
            .catalog()
            .list_workers()
            .await
            .map_err(map_engine_error)?;

        Ok(Response::new(pb::ListWorkersResponse {
            workers: records
                .into_iter()
                .map(|w| pb::WorkerSummary {
                    instance_id: w.instance_id,
                    label: w.label,
                    host: w.host,
                    kinds: jammi_wire::job_kinds_to_proto(&w.kinds),
                    started_at: w.started_at,
                    last_seen_at: w.last_seen_at,
                    state: jammi_wire::worker_state_to_proto(w.state) as i32,
                    devices: w
                        .devices
                        .into_iter()
                        .map(|d| pb::DeviceFact {
                            kind: jammi_wire::device_kind_to_proto(d.kind) as i32,
                            ordinal: d.ordinal,
                        })
                        .collect(),
                })
                .collect(),
        }))
    }

    /// Tenant-scoped: this RPC sweeps only the CALLER's own terminal rows,
    /// through the SAME [`scoped`] path every other `JobService` RPC uses —
    /// no verb on this wire surface is exempt from row-scoped tenant
    /// isolation (family D/L). This is DISTINCT from the process's
    /// construction-time sweep (`InferenceSession::wrap`, not an RPC), which
    /// stays global-by-design (it runs once at boot, before any request is
    /// scoped, under an explicit admin bypass) — see that call site's doc.
    #[tracing::instrument(skip(self, request), fields(tenant_id = tracing::field::Empty))]
    async fn prune_jobs(
        &self,
        request: Request<pb::PruneJobsRequest>,
    ) -> Result<Response<pb::PruneJobsResponse>, Status> {
        let tenant = session_tenant_traced(&request);
        let retention = self.session.inner_config().jobs.retention();
        let jobs_deleted = scoped(&self.session, tenant, || {
            self.session.catalog().prune_jobs(retention)
        })
        .await
        .map_err(map_engine_error)? as u64;
        Ok(Response::new(pb::PruneJobsResponse { jobs_deleted }))
    }
}

/// Read a job row by id under the request's tenant scope, applying
/// row-scoped on-read reclaim before returning it. An absent
/// or cross-tenant id is the only failure `Catalog::get_job` raises
/// (`JammiError::Catalog`) — mapped to `NOT_FOUND` (never
/// `PERMISSION_DENIED`, matching every other tenant-scoped read in this
/// codebase) while still attaching the faithful structured error detail
/// (`jammi_wire::attach_error_detail`, the same mechanism
/// [`map_engine_error`] uses) so a remote caller reconstructs the exact
/// `JammiError::Catalog` variant the embedded surface would raise for
/// the same lookup — a `Status::not_found` built from a bare string
/// would instead decode as the message-only `JammiError::Remote` fallback.
async fn read_job(
    session: &Arc<InferenceSession>,
    tenant: Option<TenantId>,
    job_id: &str,
) -> Result<JobRecord, Status> {
    let record = scoped(session, tenant, || session.catalog().get_job(job_id))
        .await
        .map_err(|e| {
            let message = e.to_string();
            jammi_wire::attach_error_detail(tonic::Code::NotFound, message, &e)
        })?;
    match scoped(session, tenant, || session.reclaim_job_on_read(&record)).await {
        Ok(Some(reclaimed)) => Ok(reclaimed),
        Ok(None) => Ok(record),
        Err(e) => Err(map_engine_error(e)),
    }
}

/// Build a [`pb::JobStatusResponse`] from a job row: status, kind, progress,
/// failure message, the output model id the submit stamped (training kinds
/// only), the tagged terminal result (once completed), and the relayed
/// acceleration report blob.
fn job_status_response_from_record(record: &JobRecord) -> Result<pb::JobStatusResponse, Status> {
    let result = EngineJobResult::recorded(record)
        .map_err(map_engine_error)?
        .map(|parsed| match parsed {
            EngineJobResult::Model {
                model_id,
                artifact_path,
                metrics,
                cache_outcome,
            } => pb::job_status_response::Result::Model(pb::ModelResult {
                model_id,
                artifact_path,
                metrics_json: metrics,
                cache_outcome: Some(jammi_wire::cache_outcome_to_proto(&cache_outcome)),
            }),
            EngineJobResult::Table {
                table,
                cache_outcome,
            } => pb::job_status_response::Result::Table(pb::TableResult {
                table,
                cache_outcome: Some(jammi_wire::cache_outcome_to_proto(&cache_outcome)),
            }),
        });

    Ok(pb::JobStatusResponse {
        status: jammi_wire::job_status_to_proto(record.status) as i32,
        kind: jammi_wire::job_kind_to_proto(record.kind) as i32,
        progress: Some(jammi_wire::job_progress_to_proto(record.progress)),
        error: record.error.clone(),
        output_model_id: record.output_model_id.clone(),
        result,
        acceleration_report_json: record.acceleration_report.clone(),
    })
}

/// Build a [`pb::JobSummary`] from a job row — the lifecycle projection a
/// listing reads. Its `output_model_id` is the model the job produced
/// ([`JobRecord::produced_model_id`]), absent until a training kind completes.
fn job_summary_from_record(record: JobRecord) -> pb::JobSummary {
    let output_model_id = record.produced_model_id().map(str::to_string);
    pb::JobSummary {
        job_id: record.job_id,
        kind: jammi_wire::job_kind_to_proto(record.kind) as i32,
        status: jammi_wire::job_status_to_proto(record.status) as i32,
        base_model_id: record.model_ref,
        output_model_id,
        created_at: record.created_at,
        error: record.error,
        claimed_by: record.claimed_by,
        ranks: record.ranks,
    }
}
