//! The wire form of the job queue's vocabulary: a job's status, kind and
//! progress, and a worker's state and devices. Encode is total; decode refuses
//! an `UNSPECIFIED` value where the engine always states one, so a peer that
//! sent nothing is never read as a default.

use jammi_datafusion::ComputeDeviceKind;
use jammi_db::catalog::jobs_repo::{JobProgress, WorkerState};
use jammi_db::catalog::status::{JobKind, JobKinds, JobPhase, JobStatus};
use tonic::Status;

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
