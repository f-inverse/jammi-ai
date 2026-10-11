//! The wire form of the job queue's vocabulary: a job's status, kind and
//! progress, and a worker's state and devices. Encode is total; decode refuses
//! an `UNSPECIFIED` value where the engine always states one, so a peer that
//! sent nothing is never read as a default.

use jammi_datafusion::ComputeDeviceKind;
use jammi_db::catalog::instance::DeviceFact;
use jammi_db::catalog::jobs_repo::{JobProgress, WorkerState};
use jammi_db::catalog::status::{JobKind, JobKinds, JobPhase, JobStatus};
use jammi_db::store::CacheOutcome;
use tonic::Status;

use crate::cache_outcome_from_proto;
use crate::proto::job as pb;

/// The one mapping between an engine vocabulary and its wire enum: `to` is
/// total, and `from` refuses the wire's `UNSPECIFIED` arm, naming what the
/// peer left unsaid.
macro_rules! wire_vocabulary {
    ($what:literal, $to:ident, $from:ident, $engine:ident, $wire:ident, { $($e:ident <=> $w:ident),+ $(,)? }) => {
        #[doc = concat!("Encode a ", $what, ".")]
        pub fn $to(value: $engine) -> pb::$wire {
            match value {
                $($engine::$e => pb::$wire::$w,)+
            }
        }

        #[doc = concat!("Decode a ", $what, "; an unspecified one is a peer fault.")]
        pub fn $from(value: pb::$wire) -> Result<$engine, Status> {
            match value {
                $(pb::$wire::$w => Ok($engine::$e),)+
                pb::$wire::Unspecified => {
                    Err(Status::internal(concat!("the peer sent no ", $what)))
                }
            }
        }
    };
}

wire_vocabulary!("job status", job_status_to_proto, job_status_from_proto, JobStatus, JobStatus, {
    Queued <=> Queued,
    Running <=> Running,
    Completed <=> Completed,
    Failed <=> Failed,
    Cancelled <=> Cancelled,
});

wire_vocabulary!("job kind", job_kind_to_proto, job_kind_from_proto, JobKind, JobKind, {
    FineTune <=> FineTune,
    GraphFineTune <=> GraphFineTune,
    ContextPredictor <=> ContextPredictor,
    NeighborGraph <=> NeighborGraph,
    Propagate <=> Propagate,
    GraphStructure <=> GraphStructure,
    AsofJoin <=> AsofJoin,
    LexicalIndex <=> LexicalIndex,
    Embedding <=> Embedding,
    Infer <=> Infer,
});

wire_vocabulary!("job phase", job_phase_to_proto, job_phase_from_proto, JobPhase, JobPhase, {
    Assembling <=> Assembling,
    Training <=> Training,
    Computing <=> Computing,
    Publishing <=> Publishing,
});

wire_vocabulary!("worker state", worker_state_to_proto, worker_state_from_proto, WorkerState, WorkerState, {
    Warming <=> Warming,
    Claiming <=> Claiming,
    Draining <=> Draining,
});

wire_vocabulary!("device kind", device_kind_to_proto, device_kind_from_proto, ComputeDeviceKind, DeviceKind, {
    Cpu <=> Cpu,
    Cuda <=> Cuda,
    Metal <=> Metal,
});

/// Encode a worker's kind set.
pub fn job_kinds_to_proto(kinds: &JobKinds) -> Vec<i32> {
    kinds
        .as_slice()
        .iter()
        .map(|&kind| job_kind_to_proto(kind) as i32)
        .collect()
}

/// Decode a worker's kind set.
pub fn job_kinds_from_proto(kinds: &[i32]) -> Result<JobKinds, Status> {
    kinds
        .iter()
        .map(|&raw| {
            pb::JobKind::try_from(raw)
                .map_err(|_| Status::internal(format!("the peer sent an unknown job kind {raw}")))
                .and_then(job_kind_from_proto)
        })
        .collect::<Result<_, _>>()
        .map(JobKinds::new)
}

/// Encode a job's progress; `None` (nothing recorded yet) is the unspecified
/// phase with no rows.
pub fn job_progress_to_proto(progress: Option<JobProgress>) -> pb::JobProgress {
    match progress {
        Some(progress) => pb::JobProgress {
            done: progress.done,
            total: progress.total,
            phase: job_phase_to_proto(progress.phase) as i32,
        },
        None => pb::JobProgress::default(),
    }
}

/// Decode a job's progress: the unspecified phase is "nothing recorded yet".
pub fn job_progress_from_proto(progress: pb::JobProgress) -> Result<Option<JobProgress>, Status> {
    match progress.phase() {
        pb::JobPhase::Unspecified => Ok(None),
        phase => Ok(Some(JobProgress {
            phase: job_phase_from_proto(phase)?,
            done: progress.done,
            total: progress.total,
        })),
    }
}

/// One job's lifecycle status, decoded from a `JobStatus` response — what
/// either client's `job_status` returns.
#[derive(Debug, Clone)]
pub struct JobStatusInfo {
    /// The job's lifecycle status.
    pub status: JobStatus,
    /// The kind this job submitted under.
    pub kind: JobKind,
    /// The id the model of a training kind registers under, stamped at
    /// submission; absent for a compute kind.
    pub output_model_id: Option<String>,
    /// The model this job produced, named by its terminal result: present once
    /// a training kind completes.
    pub produced_model_id: Option<String>,
    /// The result table this job produced: present once a compute kind
    /// completes.
    pub produced_table: Option<String>,
    /// Whether the completed job computed its result or reused an identical
    /// one; absent until it completes.
    pub cache_outcome: Option<CacheOutcome>,
    /// The failure message, present once the job fails.
    pub error: Option<String>,
    /// Run metrics recorded for a training kind, as the opaque JSON blob text
    /// nested inside the wire's `JobStatus.model.metrics_json` —
    /// the SAME blob the embedded `Job.metrics()` reads. `None` for a
    /// job with no run record yet, a compute kind, or a still-running job;
    /// this control-plane read never decodes or re-encodes the blob, so it
    /// stays byte-identical to the wire field. Schema documented at the
    /// trainer, not here.
    pub metrics_json: Option<String>,
    /// GPU-acceleration determination for this job, as the opaque,
    /// self-describing JSON blob text the wire's
    /// `JobStatus.acceleration_report_json` carries — the SAME
    /// catalog `jobs.acceleration_report` column the embedded record read
    /// returns. `None` for a row whose column is SQL `NULL`;
    /// otherwise a `"state"`-keyed object whose vocabulary is owned by the
    /// payload's producer (e.g. `"pending"` before a determination exists,
    /// `"determined"` once one does) and documented there, not enumerated
    /// here. This control-plane read never decodes or re-encodes the blob, so
    /// it stays byte-identical to the wire field.
    pub acceleration_report_json: Option<String>,
}

/// Decode a `JobStatus` response. The terminal result's fields are present
/// only for its arm: a training kind's model, metrics and outcome, a compute
/// kind's table and outcome; none before the job completes.
pub fn job_status_info_from_proto(
    resp: pb::JobStatusResponse,
) -> std::result::Result<JobStatusInfo, tonic::Status> {
    use pb::job_status_response::Result as WireResult;
    let status = job_status_from_proto(resp.status())?;
    let kind = job_kind_from_proto(resp.kind())?;
    let (produced_model_id, metrics_json, produced_table, cache_outcome) = match resp.result {
        Some(WireResult::Model(m)) => (
            Some(m.model_id),
            m.metrics_json,
            None,
            Some(cache_outcome_from_proto(m.cache_outcome)?),
        ),
        Some(WireResult::Table(t)) => (
            None,
            None,
            Some(t.table),
            Some(cache_outcome_from_proto(t.cache_outcome)?),
        ),
        None => (None, None, None, None),
    };
    Ok(JobStatusInfo {
        status,
        kind,
        output_model_id: resp.output_model_id,
        produced_model_id,
        produced_table,
        cache_outcome,
        error: resp.error,
        metrics_json,
        acceleration_report_json: resp.acceleration_report_json,
    })
}

/// Decode one `ListJobs` row.
pub fn job_summary_from_proto(j: pb::JobSummary) -> std::result::Result<JobSummary, tonic::Status> {
    Ok(JobSummary {
        kind: job_kind_from_proto(j.kind())?,
        status: job_status_from_proto(j.status())?,
        base_model_id: j.base_model_id,
        output_model_id: j.output_model_id,
        created_at: j.created_at,
        error: j.error,
        claimed_by: j.claimed_by,
        ranks: j.ranks,
        job_id: j.job_id,
    })
}

/// Decode one `ListWorkers` row.
pub fn worker_summary_from_proto(
    w: pb::WorkerSummary,
) -> std::result::Result<WorkerSummary, tonic::Status> {
    Ok(WorkerSummary {
        kinds: job_kinds_from_proto(&w.kinds)?,
        state: worker_state_from_proto(w.state())?,
        devices: w
            .devices
            .iter()
            .map(|d| {
                Ok(DeviceFact {
                    kind: device_kind_from_proto(d.kind())?,
                    ordinal: d.ordinal,
                })
            })
            .collect::<std::result::Result<_, tonic::Status>>()?,
        instance_id: w.instance_id,
        label: w.label,
        host: w.host,
        started_at: w.started_at,
        last_seen_at: w.last_seen_at,
    })
}

/// One `ListJobs` row: a job's lifecycle projection plus its submit-time
/// identity, for every job kind.
#[derive(Debug, Clone)]
pub struct JobSummary {
    /// Server-assigned job id — the `job_status` key.
    pub job_id: String,
    /// The kind this job submitted under.
    pub kind: JobKind,
    /// The job's lifecycle status.
    pub status: JobStatus,
    /// The base model a training kind trains from — the catalog's registered
    /// model id. Absent for a compute kind.
    pub base_model_id: Option<String>,
    /// The model this job produced: present once a training kind completes.
    pub output_model_id: Option<String>,
    /// Job creation time, as recorded by the catalog (UTC text timestamp).
    pub created_at: String,
    /// The failure message, present once the job fails.
    pub error: Option<String>,
    /// The instance holding (or last holding) the job's claim; absent while
    /// queued.
    pub claimed_by: Option<String>,
    /// The instance that ran each rank of the latest training attempt, in
    /// rank order; empty for a compute job and until an attempt records it.
    pub ranks: Vec<String>,
}

/// One engine process currently running the claim loop, as `ListWorkers`
/// reports it.
#[derive(Debug, Clone)]
pub struct WorkerSummary {
    /// The process's minted-at-construction uuid.
    pub instance_id: String,
    /// `JAMMI_WORKER_ID`, non-unique — an operator-chosen label; absent when
    /// unset.
    pub label: Option<String>,
    /// The machine the process runs on, as its OS names it.
    pub host: Option<String>,
    /// The job kinds this process claims.
    pub kinds: JobKinds,
    /// The claim loop's lifecycle state.
    pub state: WorkerState,
    pub started_at: String,
    pub last_seen_at: String,
    /// This worker's device inventory — a `ListWorkers` mirror only (never
    /// the placement policy's authority; see
    /// `pb::WorkerSummary::devices`'s doc).
    pub devices: Vec<DeviceFact>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_engine_value_round_trips_and_unspecified_is_refused() {
        for &status in JobStatus::ALL {
            assert_eq!(
                job_status_from_proto(job_status_to_proto(status)).unwrap(),
                status
            );
        }
        for &kind in JobKind::ALL {
            assert_eq!(job_kind_from_proto(job_kind_to_proto(kind)).unwrap(), kind);
        }
        for &phase in JobPhase::ALL {
            assert_eq!(
                job_phase_from_proto(job_phase_to_proto(phase)).unwrap(),
                phase
            );
        }
        for state in [
            WorkerState::Warming,
            WorkerState::Claiming,
            WorkerState::Draining,
        ] {
            assert_eq!(
                worker_state_from_proto(worker_state_to_proto(state)).unwrap(),
                state
            );
        }
        assert_eq!(
            job_status_from_proto(pb::JobStatus::Unspecified)
                .unwrap_err()
                .code(),
            tonic::Code::Internal
        );
        assert_eq!(
            job_kinds_from_proto(&job_kinds_to_proto(&JobKinds::all())).unwrap(),
            JobKinds::all()
        );
        assert!(job_kinds_from_proto(&[99]).is_err());
    }

    #[test]
    fn progress_round_trips_and_nothing_recorded_is_the_unspecified_phase() {
        let progress = JobProgress {
            phase: JobPhase::Training,
            done: Some(120),
            total: Some(640),
        };
        assert_eq!(
            job_progress_from_proto(job_progress_to_proto(Some(progress))).unwrap(),
            Some(progress)
        );
        assert_eq!(
            job_progress_from_proto(job_progress_to_proto(None)).unwrap(),
            None
        );
    }
}

#[cfg(test)]
mod decoded_tests {
    use super::*;
    use pb::{job_status_response::Result as WireResult, JobStatusResponse, ModelResult};

    /// The present arm: a completed training job's wire response carries a
    /// `model` result with a metrics blob, and the control-plane read relays
    /// it verbatim (no decode, no re-encode) — byte-identical to the wire
    /// text.
    #[test]
    fn metrics_json_present_arm_carries_the_wire_blob_verbatim() {
        let resp = JobStatusResponse {
            status: pb::JobStatus::Completed as i32,
            kind: pb::JobKind::FineTune as i32,
            progress: None,
            output_model_id: Some("jammi:fine-tuned:abc".to_string()),
            error: None,
            result: Some(WireResult::Model(ModelResult {
                model_id: "jammi:fine-tuned:abc".to_string(),
                artifact_path: "file:///artifacts/abc".to_string(),
                metrics_json: Some(r#"{"best_loss":0.1,"train_loss_curve":[[0,0.2]]}"#.to_string()),
                cache_outcome: Some(crate::cache_outcome_to_proto(
                    &jammi_db::store::CacheOutcome::Computed,
                )),
            })),
            acceleration_report_json: Some(r#"{"state":"determined","fa2_f16":true}"#.to_string()),
        };
        let info = job_status_info_from_proto(resp).unwrap();
        assert_eq!(
            info.metrics_json.as_deref(),
            Some(r#"{"best_loss":0.1,"train_loss_curve":[[0,0.2]]}"#)
        );
    }

    /// The absent arm: a queued job's wire response carries no result yet
    /// (field presence, not an empty string) — the control-plane read stays
    /// `None`, never inventing an empty-object placeholder.
    #[test]
    fn metrics_json_absent_arm_stays_none() {
        let resp = JobStatusResponse {
            status: pb::JobStatus::Queued as i32,
            kind: pb::JobKind::FineTune as i32,
            progress: None,
            output_model_id: None,
            error: None,
            result: None,
            acceleration_report_json: Some(r#"{"state":"pending"}"#.to_string()),
        };
        let info = job_status_info_from_proto(resp).unwrap();
        assert_eq!(info.metrics_json, None);
    }

    /// A compute kind's terminal `table` result carries no metrics — the
    /// control-plane read stays `None` rather than misreading the other
    /// oneof arm.
    #[test]
    fn table_result_arm_carries_no_metrics() {
        let resp = JobStatusResponse {
            status: pb::JobStatus::Completed as i32,
            kind: pb::JobKind::Embedding as i32,
            progress: None,
            output_model_id: None,
            error: None,
            result: Some(WireResult::Table(pb::TableResult {
                table: "jammi.embeddings_1".to_string(),
                cache_outcome: Some(crate::cache_outcome_to_proto(
                    &jammi_db::store::CacheOutcome::Computed,
                )),
            })),
            acceleration_report_json: None,
        };
        let info = job_status_info_from_proto(resp).unwrap();
        assert_eq!(info.metrics_json, None);
    }

    /// The determined arm: a claimed job's wire response carries
    /// the claiming worker's determination, and the control-plane read
    /// relays it verbatim (no decode, no re-encode) — byte-identical to the
    /// wire text. Mirrors `metrics_json_present_arm_carries_the_wire_blob_verbatim`.
    #[test]
    fn acceleration_report_json_determined_arm_carries_the_wire_blob_verbatim() {
        let resp = JobStatusResponse {
            status: pb::JobStatus::Running as i32,
            kind: pb::JobKind::FineTune as i32,
            progress: None,
            output_model_id: None,
            error: None,
            result: None,
            acceleration_report_json: Some(
                r#"{"state":"determined","fa2_f16":true,"reason":"sm_90 capable"}"#.to_string(),
            ),
        };
        let info = job_status_info_from_proto(resp).unwrap();
        assert_eq!(
            info.acceleration_report_json.as_deref(),
            Some(r#"{"state":"determined","fa2_f16":true,"reason":"sm_90 capable"}"#)
        );
    }

    /// The pending arm: a freshly submitted, unclaimed job's wire
    /// response carries the explicit pending marker, never `None` or an
    /// empty string.
    #[test]
    fn acceleration_report_json_pending_arm_carries_the_explicit_marker() {
        let resp = JobStatusResponse {
            status: pb::JobStatus::Queued as i32,
            kind: pb::JobKind::FineTune as i32,
            progress: None,
            output_model_id: None,
            error: None,
            result: None,
            acceleration_report_json: Some(r#"{"state":"pending"}"#.to_string()),
        };
        let info = job_status_info_from_proto(resp).unwrap();
        assert_eq!(
            info.acceleration_report_json.as_deref(),
            Some(r#"{"state":"pending"}"#)
        );
    }

    /// The absent arm: a row whose `acceleration_report` column is SQL
    /// `NULL` carries no wire field (field presence,
    /// not an empty string) — the control-plane read stays `None`, never
    /// inventing a fabricated tri-state value. Mirrors
    /// `metrics_json_absent_arm_stays_none`.
    #[test]
    fn acceleration_report_json_absent_arm_stays_none() {
        let resp = JobStatusResponse {
            status: pb::JobStatus::Completed as i32,
            kind: pb::JobKind::FineTune as i32,
            progress: None,
            output_model_id: Some("jammi:fine-tuned:legacy".to_string()),
            error: None,
            result: Some(WireResult::Model(ModelResult {
                model_id: "jammi:fine-tuned:legacy".to_string(),
                artifact_path: "file:///artifacts/legacy".to_string(),
                metrics_json: Some(r#"{"best_loss":0.2}"#.to_string()),
                cache_outcome: Some(crate::cache_outcome_to_proto(
                    &jammi_db::store::CacheOutcome::Computed,
                )),
            })),
            acceleration_report_json: None,
        };
        let info = job_status_info_from_proto(resp).unwrap();
        assert_eq!(info.acceleration_report_json, None);
    }
}
