//! The training stage's wire form: what a `PhysicalExtensionCodec` writes
//! for a [`TrainingExec`] and rebuilds the node from in another process.
//! Only the job crosses; the decoded node binds to the decoding process's
//! own [`TrainingRunner`].

use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;
use prost::Message;

use crate::device::ComputeDeviceKind;
use crate::error::{Error, Result};
use crate::training::exec::{TrainingExec, TrainingJob, TrainingRunner};

/// The stage's wire package (`jammi.training.v1`, compiled by this crate's
/// `build.rs`). Public so a codec's own tests can construct a descriptor
/// the encoder never would and prove the decode refuses it.
#[allow(clippy::all, dead_code, missing_docs)]
pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/jammi.training.v1.rs"));
}

/// `job` as its wire descriptor.
pub fn job_to_proto(job: &TrainingJob) -> pb::TrainingExecNode {
    pb::TrainingExecNode {
        job_id: job.job_id.clone(),
        attempt: job.attempt,
        submitter: job.submitter.clone(),
        device_kind: job.device_kind.wire_str().to_string(),
        claimed_at: job.claimed_at.to_rfc3339(),
    }
}

/// The job a wire descriptor carries.
pub fn job_from_proto(msg: pb::TrainingExecNode) -> Result<TrainingJob> {
    Ok(TrainingJob {
        device_kind: ComputeDeviceKind::parse(&msg.device_kind)
            .ok_or_else(|| Error::UnknownDeviceKind(msg.device_kind.clone()))?,
        claimed_at: chrono::DateTime::parse_from_rfc3339(&msg.claimed_at)
            .map_err(|e| Error::Decode(format!("TrainingExecNode: claimed_at: {e}")))?
            .with_timezone(&chrono::Utc),
        job_id: msg.job_id,
        attempt: msg.attempt,
        submitter: msg.submitter,
    })
}

/// Encode a [`TrainingExec`]'s job into `buf`.
pub fn encode_training(exec: &TrainingExec, buf: &mut Vec<u8>) -> Result<()> {
    job_to_proto(exec.job())
        .encode(buf)
        .map_err(|e| Error::Decode(e.to_string()))
}

/// Rebuild a [`TrainingExec`] from `body`, bound to `runner`.
pub fn decode_training(
    body: &[u8],
    runner: Arc<dyn TrainingRunner>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let msg = pb::TrainingExecNode::decode(body).map_err(|e| Error::Decode(e.to_string()))?;
    Ok(Arc::new(TrainingExec::new(job_from_proto(msg)?, runner)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::exec::NoTrainingRunner;

    /// A job round-trips its wire form field for field.
    #[test]
    fn a_job_round_trips_its_wire_form() {
        let job = TrainingJob {
            job_id: "job-1".into(),
            attempt: 3,
            submitter: "instance-a".into(),
            device_kind: ComputeDeviceKind::Cuda,
            claimed_at: chrono::DateTime::parse_from_rfc3339("2026-09-23T12:34:56.789Z")
                .unwrap()
                .with_timezone(&chrono::Utc),
        };
        let mut buf = Vec::new();
        encode_training(
            &TrainingExec::new(job.clone(), Arc::new(NoTrainingRunner)),
            &mut buf,
        )
        .unwrap();
        let decoded = decode_training(&buf, Arc::new(NoTrainingRunner)).unwrap();
        let decoded = decoded
            .downcast_ref::<TrainingExec>()
            .expect("the decode rebuilds a TrainingExec");
        assert_eq!(decoded.job(), &job);
    }

    /// A descriptor naming no device kind the crate has is refused typed.
    #[test]
    fn an_unknown_device_kind_is_refused_typed() {
        let mut msg = job_to_proto(&TrainingJob {
            job_id: "job-1".into(),
            attempt: 1,
            submitter: "instance-a".into(),
            device_kind: ComputeDeviceKind::Cpu,
            claimed_at: chrono::Utc::now(),
        });
        msg.device_kind = "tpu".into();
        let buf = msg.encode_to_vec();
        let Err(err) = decode_training(&buf, Arc::new(NoTrainingRunner)) else {
            panic!("an unknown device kind must refuse");
        };
        assert!(
            matches!(err, Error::UnknownDeviceKind(ref kind) if kind == "tpu"),
            "{err}"
        );
    }
}
