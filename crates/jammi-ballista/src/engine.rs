//! `JammiExecutionEngine` — wraps Ballista's [`DefaultExecutionEngine`] and
//! adds jammi's own per-stage duties before delegating (contract
//! `feat_500-wave4` §2.2, refined by §9 B3).
//!
//! K7 device pinning: `InferenceExec::device_kind()` is a required
//! constructor argument, so every `InferenceExec` — decoded or in-process —
//! names a concrete kind (`InferenceExecBuilder::new`, `inference_exec.rs`;
//! the codec never invents or rewrites it, `codec.rs`). A stage whose
//! `InferenceExec` names a kind different from THIS executor's own
//! `InferenceSession::compute_device().kind()` is refused typed, never
//! silently run on the wrong device.
//!
//! The other duty (README r41): a stage whose plan contains a `GangExec`
//! must be single-partition — one gang mechanism, never a multi-partition
//! fan-out of the coordinator body. Refused typed, naming the partition
//! count found.

use std::sync::Arc;

use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionConfig;

use ballista_core::JobId;
use ballista_executor::execution_engine::{
    DefaultExecutionEngine, ExecutionEngine, QueryStageExecutor,
};

use jammi_ai::operator::inference_exec::InferenceExec;
use jammi_ai::session::InferenceSession;

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

/// The first `InferenceExec` device-kind mismatch found in `plan` against
/// `own_kind`, if any: `Some(descriptor_kind)`.
fn first_device_kind_mismatch(
    plan: &Arc<dyn ExecutionPlan>,
    own_kind: jammi_db::store::manifest::ComputeDeviceKind,
) -> DfResult<Option<jammi_db::store::manifest::ComputeDeviceKind>> {
    let mut mismatch = None;
    plan.apply(|node| {
        if let Some(exec) = node.downcast_ref::<InferenceExec>() {
            let kind = exec.device_kind();
            if kind != own_kind {
                mismatch = Some(kind);
                return Ok(TreeNodeRecursion::Stop);
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(mismatch)
}

/// Whether `plan` contains a `GangExec` anywhere in its tree — a leaf node
/// (zero children), so a depth-first search over `.children()` finds it
/// regardless of the shuffle-writer wrapping the scheduler always applies.
fn contains_gang(plan: &Arc<dyn ExecutionPlan>) -> bool {
    if plan
        .downcast_ref::<jammi_ai::operator::gang_exec::GangExec>()
        .is_some()
    {
        return true;
    }
    plan.children().into_iter().any(contains_gang)
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
        if let Some(descriptor_kind) = first_device_kind_mismatch(&plan, own_kind)? {
            return Err(DataFusionError::Execution(format!(
                "jammi-ballista K7: stage {stage_id} of job {job_id} names InferenceExec \
                 device_kind {descriptor_kind:?}, this executor runs {own_kind:?} — refused, \
                 never silently run on the wrong device"
            )));
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
