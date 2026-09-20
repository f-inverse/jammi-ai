//! `JammiExecutionEngine` — wraps Ballista's [`DefaultExecutionEngine`] and
//! adds jammi's own per-stage duties before delegating.
//!
//! Device pinning: `InferenceSpec::device_kind` is a required field, so
//! every `InferenceExec` — decoded or in-process — names a concrete kind
//! (`inference_exec.rs`; the codec never invents or rewrites it,
//! `codec.rs`); `GangExec`'s
//! descriptor carries the same kind, stamped by the submitter
//! (`GangDescriptor::device_kind`, `jammi_ai::fine_tune::worker::
//! JobWorker::submit_placed`). A stage whose `InferenceExec`/`GangExec`
//! names a kind different from THIS executor's own `InferenceSession::
//! compute_device().kind()` is refused typed, never silently run on the
//! wrong device — [`required_device_kind`] is the ONE predicate this engine's
//! device-kind check, `placement::DevicePlacement`'s binding eligibility,
//! and `client::submit_physical_plan`'s pre-submission refusal all read: the
//! required kind is the PLAN's own (KIND MATCH), never "does any GPU exist
//! anywhere". A "GPU-bound stage" predicate would refuse a `GangExec`
//! unconditionally on an all-CPU cluster (a `GangExec` is never itself
//! CUDA/Metal-shaped; the kind is a property of its DESCRIPTOR, not of the
//! node type) and would let a CPU-kind `InferenceExec` bind to a
//! `[worker]`-disabled executor reporting no devices at all.
//!
//! The second duty: a stage whose plan contains a `GangExec`
//! must be single-partition — one gang mechanism, never a multi-partition
//! fan-out of the coordinator body. Refused typed, naming the partition
//! count found.
//!
//! The third: a stage's failure leaves this executor typed. Ballista carries
//! a task's failure from here to the client as text alone (its `Debug`
//! rendering into `FailedTask.error`, then `FailedJob.error`, then the
//! client's `Execution` message), so this engine places `TaskErrorEnvelopeExec`
//! under the stage's shuffle writer: a failure the engine's own classifier
//! types (`JammiError::from(DataFusionError)`: a plan node's
//! `External(JammiError)`, a `ResourcesExhausted`, an object-store
//! not-found) is wrapped in [`TaskErrorEnvelope`], whose text is the error's
//! wire encoding beside its message, and `client::submit_physical_plan`
//! decodes it back. A failure the classifier leaves foreign crosses as the
//! string it is.

use std::sync::Arc;

use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion::prelude::SessionConfig;
use futures::TryStreamExt;

use ballista_core::JobId;
use ballista_executor::execution_engine::{
    DefaultExecutionEngine, ExecutionEngine, QueryStageExecutor,
};

use jammi_ai::operator::gang_exec::GangExec;
use jammi_ai::operator::inference_exec::InferenceExec;
use jammi_ai::session::InferenceSession;
use jammi_db::error::JammiError;
use jammi_db::store::manifest::ComputeDeviceKind;
use jammi_wire::TaskErrorEnvelope;

/// Wraps [`DefaultExecutionEngine`], holding the executor process's own
/// session for the device-kind refusal.
pub struct JammiExecutionEngine {
    session: Arc<InferenceSession>,
    inner: DefaultExecutionEngine,
}

impl JammiExecutionEngine {
    /// Wrap the executor's session. `inner` is Ballista's own default engine
    /// (shuffle reader rewrite + writer wrap unchanged, `work_dir` local —
    /// no object-store shuffle in v1).
    pub fn new(session: Arc<InferenceSession>) -> Self {
        Self {
            session,
            inner: DefaultExecutionEngine::new(),
        }
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
/// constrains). This is the ONE predicate `JammiExecutionEngine`'s
/// device-kind check, `placement::DevicePlacement`'s binding eligibility, and
/// `client::submit_physical_plan`'s pre-submission refusal all read — KIND
/// MATCH, never "is this GPU-shaped": a `GangExec` carries whatever kind
/// its submitter stamped (CPU included), so this predicate is `Some` for
/// EVERY gang stage and EVERY inference stage, not only a GPU-bound one.
pub fn required_device_kind(plan: &Arc<dyn ExecutionPlan>) -> Option<ComputeDeviceKind> {
    if let Some(exec) = plan.downcast_ref::<GangExec>() {
        return Some(exec.descriptor().device_kind);
    }
    if let Some(exec) = plan.downcast_ref::<InferenceExec>() {
        return Some(exec.spec().device_kind);
    }
    plan.children().into_iter().find_map(required_device_kind)
}

/// A stage's failure in the form that leaves this executor: a failure the
/// engine's classifier types is wrapped in a [`TaskErrorEnvelope`] (its
/// text is what Ballista copies from hop to hop); a foreign one is handed
/// back as it was — the classifier's `Arc` is its own and unshared, so the
/// original error is recovered whole.
pub fn envelope_task_error(e: DataFusionError) -> DataFusionError {
    match JammiError::from(e) {
        JammiError::DataFusion(foreign) => {
            Arc::try_unwrap(foreign).unwrap_or_else(DataFusionError::Shared)
        }
        typed => DataFusionError::External(Box::new(TaskErrorEnvelope::new(typed))),
    }
}

/// The plan node this engine places between a stage's shuffle writer and
/// the stage's own plan: every error the child raises — at `execute` or
/// from its stream — passes through [`envelope_task_error`] before the
/// writer sees it, because the writer renders a stream error's `Debug` into
/// an `Execution` string on its single-partition path, where the type would
/// be lost. Pass-through in every other respect (the child's schema,
/// properties, partitioning, order). Never serialized: added on the executor
/// after decode.
#[derive(Debug)]
struct TaskErrorEnvelopeExec {
    child: Arc<dyn ExecutionPlan>,
}

impl DisplayAs for TaskErrorEnvelopeExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "TaskErrorEnvelopeExec")
    }
}

impl ExecutionPlan for TaskErrorEnvelopeExec {
    fn name(&self) -> &str {
        "TaskErrorEnvelopeExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.child.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.child]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let [child] = <[Arc<dyn ExecutionPlan>; 1]>::try_from(children).map_err(|children| {
            DataFusionError::Internal(format!(
                "TaskErrorEnvelopeExec has exactly one child, got {}",
                children.len()
            ))
        })?;
        Ok(Arc::new(Self { child }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        let stream = self
            .child
            .execute(partition, context)
            .map_err(envelope_task_error)?;
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            stream.schema(),
            stream.map_err(envelope_task_error),
        )))
    }
}

/// `plan` with its shuffle writer's child wrapped in [`TaskErrorEnvelopeExec`]
/// — the scheduler roots every stage in a writer, so the wrap goes one level
/// down, under it.
fn envelope_stage_failures(plan: Arc<dyn ExecutionPlan>) -> DfResult<Arc<dyn ExecutionPlan>> {
    let children = plan
        .children()
        .into_iter()
        .map(|child| {
            Arc::new(TaskErrorEnvelopeExec {
                child: Arc::clone(child),
            }) as Arc<dyn ExecutionPlan>
        })
        .collect();
    plan.with_new_children(children)
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
        if let Some(required_kind) = required_device_kind(&plan) {
            if required_kind != own_kind {
                return Err(DataFusionError::Execution(format!(
                    "jammi-ballista: stage {stage_id} of job {job_id} requires device_kind \
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
                     fan-out of the coordinator body"
                )));
            }
        }
        let plan = envelope_stage_failures(plan)?;
        self.inner
            .create_query_stage_exec(job_id, stage_id, partition_id, plan, work_dir, config)
    }
}

#[cfg(test)]
mod envelope_tests {
    use super::*;

    /// A plan node's typed refusal, wrapped however the optimizer wrapped
    /// it, leaves the executor as an envelope whose `Display` decodes back
    /// to the same variant and fields.
    #[test]
    fn a_typed_stage_failure_leaves_as_a_decodable_envelope() {
        let raised = DataFusionError::Context(
            "InferenceExec".into(),
            Box::new(DataFusionError::External(Box::new(
                JammiError::InvalidKey {
                    column: "id".into(),
                    null_count: 1,
                },
            ))),
        );
        let text = envelope_task_error(raised).to_string();
        match TaskErrorEnvelope::extract(&text) {
            Ok(Some(JammiError::InvalidKey { column, null_count })) => {
                assert_eq!(column, "id");
                assert_eq!(null_count, 1);
            }
            other => panic!("expected the enveloped InvalidKey, got {other:?} in {text}"),
        }
    }

    /// A failure the classifier leaves foreign crosses as the string it is:
    /// no marker, the same `Display`.
    #[test]
    fn a_foreign_stage_failure_crosses_unchanged() {
        let raised = DataFusionError::Plan("shuffle file group 3 missing".into());
        let before = raised.to_string();
        let text = envelope_task_error(raised).to_string();
        assert_eq!(text, before);
        assert!(matches!(TaskErrorEnvelope::extract(&text), Ok(None)));
    }
}
