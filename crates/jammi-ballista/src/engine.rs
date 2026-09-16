//! `JammiExecutionEngine` — wraps Ballista's [`DefaultExecutionEngine`] and
//! adds jammi's own per-stage duties before delegating (contract
//! `feat_500-wave4` §2.2, refined by §9 B3 and by the LANE pressure-round
//! correction below).
//!
//! K7 device pinning: `InferenceExec::device_kind()` is a required
//! constructor argument, so every `InferenceExec` — decoded or in-process —
//! names a concrete kind (`InferenceExecBuilder::new`, `inference_exec.rs`;
//! the codec never invents or rewrites it, `codec.rs`); `GangExec`'s
//! descriptor carries the same kind, stamped by the submitter
//! (`GangDescriptor::device_kind`, `jammi_ai::fine_tune::worker::
//! JobWorker::submit_placed`). A stage whose `InferenceExec`/`GangExec`
//! names a kind different from THIS executor's own `InferenceSession::
//! compute_device().kind()` is refused typed, never silently run on the
//! wrong device — [`stage_device_kind`] is the ONE predicate this engine's
//! K7 check, `placement::DevicePlacement`'s binding eligibility, and
//! `client::submit_physical_plan`'s pre-submission refusal all read: the
//! required kind is the PLAN's own (KIND MATCH), never "does any GPU exist
//! anywhere" — the earlier `stage_is_gpu_bound` predicate this pass
//! replaced would have refused a `GangExec` unconditionally on an all-CPU
//! cluster (a `GangExec` is never itself CUDA/Metal-shaped; the kind is a
//! property of its DESCRIPTOR, not of the node type) and would have let a
//! CPU-kind `InferenceExec` bind to a `[worker]`-disabled executor
//! reporting no devices at all.
//!
//! The other duty (README r41): a stage whose plan contains a `GangExec`
//! must be single-partition — one gang mechanism, never a multi-partition
//! fan-out of the coordinator body. Refused typed, naming the partition
//! count found.

use std::sync::Arc;

use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionConfig;

use ballista_core::JobId;
use ballista_executor::execution_engine::{
    DefaultExecutionEngine, ExecutionEngine, QueryStageExecutor,
};

use jammi_ai::operator::gang_exec::GangExec;
use jammi_ai::operator::inference_exec::InferenceExec;
use jammi_ai::session::InferenceSession;
use jammi_db::store::manifest::ComputeDeviceKind;

/// Wraps [`DefaultExecutionEngine`], holding the executor process's own
/// session for the K7 device-kind refusal.
pub struct JammiExecutionEngine {
    session: Arc<InferenceSession>,
    inner: DefaultExecutionEngine,
}

impl JammiExecutionEngine {
    /// Wrap the executor's session. `inner` is Ballista's own default engine
    /// (shuffle reader rewrite + writer wrap unchanged, `work_dir` local —
    /// D2 condition 3 stands, no object-store shuffle in v1).
    pub fn new(session: Arc<InferenceSession>) -> Self {
        Self {
            session,
            inner: DefaultExecutionEngine::new(),
        }
    }
}

/// `ComputeDeviceKind`'s canonical wire spelling in a `compute_executors.
/// devices`/`workers.devices` `DeviceFact.kind` string. The ONE mapping
/// [`crate::placement::DevicePlacement`]'s binding eligibility and
/// [`crate::client::submit_physical_plan`]'s pre-submission refusal both
/// read against a registered executor's device inventory (never a second,
/// independently-drifting copy of this match).
pub fn device_kind_wire_str(kind: ComputeDeviceKind) -> &'static str {
    match kind {
        ComputeDeviceKind::Cpu => "cpu",
        ComputeDeviceKind::Cuda => "cuda",
        ComputeDeviceKind::Metal => "metal",
    }
}

/// Whether `plan` contains a `GangExec` anywhere in its tree — a leaf node
/// (zero children), so a depth-first search over `.children()` finds it
/// regardless of the shuffle-writer wrapping the scheduler always applies.
fn contains_gang(plan: &Arc<dyn ExecutionPlan>) -> bool {
    if plan.downcast_ref::<GangExec>().is_some() {
        return true;
    }
    plan.children().into_iter().any(contains_gang)
}

/// The device kind `plan` REQUIRES, if any: the first `GangExec`'s
/// (`descriptor().device_kind`) or `InferenceExec`'s (`device_kind()`)
/// stamped kind found in the tree, depth-first; `None` for a plan carrying
/// neither (a plain scan/shuffle stage, which no device predicate
/// constrains). This is the ONE predicate `JammiExecutionEngine`'s K7
/// check, `placement::DevicePlacement`'s binding eligibility, and
/// `client::submit_physical_plan`'s pre-submission refusal all read — KIND
/// MATCH, never "is this GPU-shaped": a `GangExec` carries whatever kind
/// its submitter stamped (CPU included), so this predicate is `Some` for
/// EVERY gang stage and EVERY inference stage, not only a GPU-bound one.
pub fn stage_device_kind(plan: &Arc<dyn ExecutionPlan>) -> Option<ComputeDeviceKind> {
    if let Some(exec) = plan.downcast_ref::<GangExec>() {
        return Some(exec.descriptor().device_kind);
    }
    if let Some(exec) = plan.downcast_ref::<InferenceExec>() {
        return Some(exec.device_kind());
    }
    plan.children().into_iter().find_map(stage_device_kind)
}

impl ExecutionEngine for JammiExecutionEngine {
    fn create_query_stage_exec(
        &self,
        job_id: JobId,
        stage_id: usize,
        partition_id: usize,
        plan: Arc<dyn ExecutionPlan>,
        work_dir: &str,
        config: &SessionConfig,
    ) -> DfResult<Arc<dyn QueryStageExecutor>> {
        let own_kind = self.session.compute_device().kind();
        if let Some(required_kind) = stage_device_kind(&plan) {
            if required_kind != own_kind {
                return Err(DataFusionError::Execution(format!(
                    "jammi-ballista K7: stage {stage_id} of job {job_id} requires device_kind \
                     {required_kind:?} (an InferenceExec or GangExec descriptor), this executor \
                     runs {own_kind:?} — refused, never silently run on the wrong device"
                )));
            }
        }
        if contains_gang(&plan) {
            let partitions = plan.properties().output_partitioning().partition_count();
            if partitions != 1 {
                return Err(DataFusionError::Execution(format!(
                    "jammi-ballista: stage {stage_id} of job {job_id} contains a GangExec but \
                     has {partitions} partitions — one gang mechanism, never a multi-partition \
                     fan-out of the coordinator body (README r41)"
                )));
            }
        }
        self.inner
            .create_query_stage_exec(job_id, stage_id, partition_id, plan, work_dir, config)
    }
}
