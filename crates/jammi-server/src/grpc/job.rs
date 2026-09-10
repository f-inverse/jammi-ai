//! `JobService` gRPC implementation.
//!
//! `JobService` replaces `TrainingService` (PLAN-C §3): `SubmitJob` folds
//! `StartTraining`, `JobStatus` folds `TrainingStatus`, `ListJobs` folds
//! `ListTrainingJobs`, and the service adds `WaitJob` (a resumable
//! server-streaming wait) and `CancelJob`/`ListWorkers`, generalising the
//! training-only queue into the kind-agnostic durable job abstraction every
//! compute verb submits through internally (`jammi_ai::jobs`,
//! `InferenceSession::run_now`). This handler carries only the three
//! training-kind `SubmitJob` variants directly (the oneof); every other
//! compute verb is synchronous on the wire TODAY — it keeps its own
//! dedicated request AND response message on its own service (e.g.
//! `PipelineService::BuildNeighborGraph`) rather than going through
//! `SubmitJobResponse`, so this service's `JobStatus`/`WaitJob` never has a
//! compute-kind row to answer for over the wire (K4: the synchronous
//! response is still byte-identical to what a `jobs` row of the same kind
//! would resolve to, since both paths run through the same
//! `execute_compute` dispatcher).
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
use jammi_ai::fine_tune::training_job::{resolve_model_id, TrainingJob};
use jammi_ai::jobs::JobResult as EngineJobResult;
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

/// `WaitJob`'s server-side poll interval (PLAN-C §3).
const WAIT_JOB_POLL: Duration = Duration::from_millis(100);
/// Bounded so a slow/blocked receiver applies backpressure rather than
/// buffering unboundedly; small because a frame is produced at most once per
/// [`WAIT_JOB_POLL`] tick.
const WAIT_JOB_BUFFER: usize = 8;

/// The three engine training kinds `SubmitJob` accepts — the only kinds
/// `resolve_model_id` and `run_training_spec` know how to handle. Shared by
/// every place this handler must tell a training row from a compute row.
fn is_training_kind(kind: &str) -> bool {
    matches!(kind, "fine_tune" | "graph_fine_tune" | "context_predictor")
}

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
    /// returning the durable [`TrainingJob`] handle — the caller's own on a
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
    ) -> Result<TrainingJob, JammiError> {
        let key = (!idempotency_key.is_empty()).then_some(idempotency_key);
        self.session.run_training_spec_deduped(spec, key).await
    }

    /// Read a job row by id under the request's tenant scope, applying
    /// row-scoped on-read reclaim (PLAN-C §2) before returning it. An absent
    /// or cross-tenant id is the only failure `Catalog::get_job` raises
    /// (`JammiError::Catalog`) — mapped to `NOT_FOUND` (never
    /// `PERMISSION_DENIED`, matching every other tenant-scoped read in this
    /// codebase) while still attaching the faithful structured error detail
    /// (`jammi_wire::attach_error_detail`, the same mechanism
    /// [`map_engine_error`] uses) so a remote caller reconstructs the exact
    /// `JammiError::Catalog` variant the embedded surface would raise for
    /// the same lookup — a `Status::not_found` built from a bare string
    /// would instead decode as the lossy `JammiError::Other` fallback.
    async fn read_job(&self, tenant: Option<TenantId>, job_id: &str) -> Result<JobRecord, Status> {
        let record = scoped(&self.session, tenant, || {
            self.session.catalog().get_job(job_id)
        })
        .await
        .map_err(|e| {
            let message = e.to_string();
            jammi_wire::attach_error_detail(tonic::Code::NotFound, message, &e)
        })?;
        match scoped(&self.session, tenant, || {
            self.session.reclaim_job_on_read(&record)
        })
        .await
        {
            Ok(Some(reclaimed)) => Ok(reclaimed),
            Ok(None) => Ok(record),
            Err(e) => Err(map_engine_error(e)),
        }
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
            let message = format!(
                "idempotency_key exceeds the maximum length \
                 (MAX_IDEMPOTENCY_KEY_BYTES = {} bytes)",
                jammi_db::catalog::jobs_repo::MAX_IDEMPOTENCY_KEY_BYTES
            );
            let engine_err = JammiError::Config(message.clone());
            return Err(jammi_wire::attach_error_detail(
                tonic::Code::InvalidArgument,
                message,
                &engine_err,
            ));
        }

        let spec = training_spec_from_proto(req)?;
        let kind = spec.kind().to_string();

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
            job_id: job.job_id,
            kind,
            output_model_id: job.model_id,
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

        let record = self.read_job(tenant, &req.job_id).await?;
        Ok(Response::new(job_status_response_from_record(
            &req.job_id,
            &record,
        )?))
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
        let first = self.read_job(tenant, &req.job_id).await?;

        let (tx, rx) = mpsc::channel::<Result<pb::JobEvent, Status>>(WAIT_JOB_BUFFER);
        let session = Arc::clone(&self.session);
        let job_id = req.job_id.clone();
        tokio::spawn(async move {
            let mut record = first;
            loop {
                if jammi_db::catalog::status::JobStatus::Completed.to_string() == record.status
                    || jammi_db::catalog::status::JobStatus::Failed.to_string() == record.status
                {
                    let event = job_status_response_from_record(&job_id, &record).map(|done| {
                        pb::JobEvent {
                            event: Some(pb::job_event::Event::Done(done)),
                        }
                    });
                    let _ = tx.send(event).await;
                    return;
                }

                let progress = pb::JobProgress {
                    rows_done: record.progress_rows_done.map(|v| v as u64),
                    rows_total: record.progress_rows_total.map(|v| v as u64),
                    phase: record.progress_phase.clone().unwrap_or_default(),
                };
                if tx
                    .send(Ok(pb::JobEvent {
                        event: Some(pb::job_event::Event::Progress(progress)),
                    }))
                    .await
                    .is_err()
                {
                    // The receiver dropped — a client disconnect. The job
                    // itself is untouched; a reconnect resumes the wait.
                    return;
                }

                tokio::time::sleep(WAIT_JOB_POLL).await;

                record = match scoped(&session, tenant, || session.catalog().get_job(&job_id)).await
                {
                    Ok(r) => r,
                    Err(e) => {
                        let message = e.to_string();
                        let status =
                            jammi_wire::attach_error_detail(tonic::Code::NotFound, message, &e);
                        let _ = tx.send(Err(status)).await;
                        return;
                    }
                };
                record =
                    match scoped(&session, tenant, || session.reclaim_job_on_read(&record)).await {
                        Ok(Some(reclaimed)) => reclaimed,
                        Ok(None) => record,
                        Err(e) => {
                            let _ = tx.send(Err(map_engine_error(e))).await;
                            return;
                        }
                    };
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
                    label: w.label.unwrap_or_default(),
                    host: w.host.unwrap_or_default(),
                    kinds: w.kinds,
                    started_at: w.started_at,
                    last_seen_at: w.last_seen_at,
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

/// Build a [`pb::JobStatusResponse`] from a job row: status, kind, progress,
/// failure message, the resolved output model id (training kinds only), the
/// tagged terminal result (once completed), and the relayed acceleration
/// report blob.
fn job_status_response_from_record(
    job_id: &str,
    record: &JobRecord,
) -> Result<pb::JobStatusResponse, Status> {
    let output_model_id = if is_training_kind(&record.kind) {
        resolve_model_id(job_id, record).map_err(map_engine_error)?
    } else {
        String::new()
    };

    let result = match record.result.as_deref() {
        Some(raw) if record.status == "completed" => {
            let parsed: EngineJobResult = serde_json::from_str(raw).map_err(|e| {
                Status::internal(format!(
                    "job '{job_id}' recorded an unparseable result: {e}"
                ))
            })?;
            Some(match parsed {
                EngineJobResult::Model {
                    model_id,
                    artifact_path,
                    metrics,
                } => pb::job_status_response::Result::Model(pb::ModelResult {
                    model_id,
                    artifact_path,
                    metrics_json: metrics,
                }),
                EngineJobResult::Table {
                    table,
                    cache_outcome,
                } => pb::job_status_response::Result::Table(pb::TableResult {
                    table,
                    cache_outcome,
                }),
            })
        }
        _ => None,
    };

    Ok(pb::JobStatusResponse {
        status: record.status.clone(),
        kind: record.kind.clone(),
        progress: Some(pb::JobProgress {
            rows_done: record.progress_rows_done.map(|v| v as u64),
            rows_total: record.progress_rows_total.map(|v| v as u64),
            phase: record.progress_phase.clone().unwrap_or_default(),
        }),
        error: record.error.clone().unwrap_or_default(),
        output_model_id,
        result,
        acceleration_report_json: record.acceleration_report.clone(),
    })
}

/// Build a [`pb::JobSummary`] from a job row — the same lifecycle projection
/// [`job_status_response_from_record`] reads, but relaying `output_model_id`
/// verbatim (the catalog column, empty until stamped) rather than resolving
/// it: this listing answers "has the output row landed yet", not "what will
/// this job's model be called" (mirrors the former `ListTrainingJobs`
/// contract, generalised to every kind).
fn job_summary_from_record(record: JobRecord) -> pb::JobSummary {
    pb::JobSummary {
        job_id: record.job_id,
        kind: record.kind,
        status: record.status,
        base_model_id: record.model_ref.unwrap_or_default(),
        output_model_id: record.output_model_id.unwrap_or_default(),
        created_at: record.created_at,
        error: record.error.unwrap_or_default(),
    }
}
