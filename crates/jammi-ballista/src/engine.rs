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
//! Both refusals are decided by [`stage_refusal`] before the stage exists —
//! `JammiError::DeviceKindUnheld` (the plan's kind, this executor's own as
//! the one kind held) and `JammiError::GangFanOut` (the descriptor's job,
//! the partition count) — and leave the executor through the SAME envelope
//! a running stage's failure does: Ballista renders a stage-creation error
//! with `Debug` into the task's `FailedTask.error` (non-retryable, so the
//! job fails on the first attempt), and the envelope's text is what that
//! rendering carries. On a cluster whose scheduler binds by KIND MATCH
//! (`placement::DevicePlacement`) neither refusal is reachable from a
//! well-formed submission — they are the executor's own guard against a
//! stage bound past that match or a plan fanned out past its planner.
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

use jammi_ai::operator::gang_exec::{GangDescriptor, GangExec};
use jammi_ai::operator::inference_exec::InferenceExec;
use jammi_ai::session::InferenceSession;
use jammi_db::compute_plane::PlanRequirements;
use jammi_db::error::JammiError;
use jammi_db::store::manifest::ComputeDeviceKind;
use jammi_wire::TaskErrorEnvelope;

/// Wraps [`DefaultExecutionEngine`], holding the executor process's own
/// session for [`stage_refusal`]'s device-kind check.
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

/// The `GangExec`'s own descriptor found anywhere in `plan`, if any — a
/// `GangExec` is a leaf (zero children, `crate::codec`'s decode never wraps
/// it), so a plain depth-first search over `.children()` finds it wherever
/// the stage's shuffle-writer wrapping placed it. `descriptor().job_id` is
/// jammi's OWN fine-tune catalog job id — DISTINCT from Ballista's
/// internally-minted submission `JobId` (unrelated id spaces: a real
/// Ballista submission never gives them the same value, only a hermetic
/// test fixture that deliberately aliases them would). The ONE lookup this
/// engine's fan-out refusal and `placement::DevicePlacement`'s submitter
/// exclusion and re-launch guard all read.
pub(crate) fn gang_descriptor_of(plan: &Arc<dyn ExecutionPlan>) -> Option<&GangDescriptor> {
    if let Some(exec) = plan.downcast_ref::<GangExec>() {
        return Some(exec.descriptor());
    }
    plan.children().into_iter().find_map(gang_descriptor_of)
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

/// What `plan` asks of the executor that holds it, read off its own nodes:
/// the kind it requires ([`required_device_kind`]) and, for a plan
/// carrying a gang, the gang's own submitter as the executor it must not
/// land on (`placement::DevicePlacement`'s submitter exclusion, read from
/// the same descriptor). The one reader the submit client's admission and
/// the binder's eligibility share.
pub fn plan_requirements(plan: &Arc<dyn ExecutionPlan>) -> PlanRequirements {
    PlanRequirements {
        device_kind: required_device_kind(plan),
        excluded_executor: gang_descriptor_of(plan).map(|d| d.submitter.clone()),
    }
}

/// Why THIS executor cannot create a stage over `plan`, decided before the
/// stage exists, or `None` when it can (module doc): the plan's required
/// kind ([`required_device_kind`]) is not `own_kind`, or the plan carries a
/// gang (`gang_descriptor_of`) at other than one output partition. A plan
/// requiring no kind and carrying no gang is never refused here.
pub(crate) fn stage_refusal(
    own_kind: ComputeDeviceKind,
    plan: &Arc<dyn ExecutionPlan>,
) -> Option<JammiError> {
    if let Some(required) = required_device_kind(plan) {
        if required != own_kind {
            return Some(JammiError::DeviceKindUnheld {
                required,
                held: vec![own_kind],
            });
        }
    }
    let descriptor = gang_descriptor_of(plan)?;
    let partitions = plan.properties().output_partitioning().partition_count();
    (partitions != 1).then(|| JammiError::GangFanOut {
        job_id: descriptor.job_id.clone(),
        partitions: partitions as u64,
    })
}

/// `typed` in the form that leaves this executor: an `External` over its
/// [`TaskErrorEnvelope`], whose text is what Ballista copies from hop to
/// hop.
fn enveloped(typed: JammiError) -> DataFusionError {
    DataFusionError::External(Box::new(TaskErrorEnvelope::new(typed)))
}

/// A stage's failure in the form that leaves this executor: a failure the
/// engine's classifier types is enveloped (`enveloped`); a foreign one is handed
/// back as it was — the classifier's `Arc` is its own and unshared, so the
/// original error is recovered whole.
pub(crate) fn envelope_task_error(e: DataFusionError) -> DataFusionError {
    match JammiError::from(e) {
        JammiError::DataFusion(foreign) => {
            Arc::try_unwrap(foreign).unwrap_or_else(DataFusionError::Shared)
        }
        typed => enveloped(typed),
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
        if let Some(refusal) = stage_refusal(self.session.compute_device().kind(), &plan) {
            tracing::warn!(
                job_id = %job_id,
                stage_id,
                partition_id,
                error = %refusal,
                "jammi-ballista: stage refused before it ran"
            );
            return Err(enveloped(refusal));
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

#[cfg(test)]
mod refusal_tests {
    use super::*;
    use datafusion::physical_plan::union::UnionExec;

    fn gang(kind: ComputeDeviceKind) -> Arc<dyn ExecutionPlan> {
        Arc::new(GangExec::new(GangDescriptor {
            job_id: "job-7".to_string(),
            attempt: 0,
            world: 2,
            submitter: "submitter".to_string(),
            device_kind: kind,
        }))
    }

    /// The string Ballista renders a stage-creation failure into
    /// (`FailedTask::from`'s `Debug` of the `BallistaError`, then the
    /// scheduler's and the client's prefixes), read back as the typed
    /// error.
    fn restored(refusal: JammiError) -> JammiError {
        let leaving = enveloped(refusal);
        let arrived = format!(
            "Job 7bY2 failed: Job failed due to stage 1 failed: Task failed due to runtime \
             execution error: DataFusionError({leaving:?})\n"
        );
        TaskErrorEnvelope::extract(&arrived)
            .expect("a well-formed envelope decodes")
            .expect("the envelope is present")
    }

    /// A stage whose plan stamps another device kind is refused before it
    /// runs, as `DeviceKindUnheld` naming the plan's kind and this
    /// executor's own as the one kind held — and restores as the same
    /// after Ballista's rendering.
    #[test]
    fn a_stage_of_another_device_kind_is_refused_typed_before_it_runs() {
        let refusal = stage_refusal(ComputeDeviceKind::Cpu, &gang(ComputeDeviceKind::Cuda))
            .expect("a CUDA-stamped gang is refused on a CPU executor");
        match restored(refusal) {
            JammiError::DeviceKindUnheld { required, held } => {
                assert_eq!(required, ComputeDeviceKind::Cuda);
                assert_eq!(held, vec![ComputeDeviceKind::Cpu]);
            }
            other => panic!("expected DeviceKindUnheld, got {other:?}"),
        }
        assert!(
            stage_refusal(ComputeDeviceKind::Cpu, &gang(ComputeDeviceKind::Cpu)).is_none(),
            "a gang stamped this executor's own kind, at one partition, is held"
        );
    }

    /// A stage carrying a gang at more than one output partition is refused
    /// before it runs, as `GangFanOut` naming the descriptor's job and the
    /// partition count — and restores as the same after Ballista's
    /// rendering. Two gang leaves under a union is the smallest such plan:
    /// the union's output partitioning is the sum of its inputs'.
    #[test]
    fn a_fanned_out_gang_stage_is_refused_typed_before_it_runs() {
        let fanned_out = UnionExec::try_new(vec![
            gang(ComputeDeviceKind::Cpu),
            gang(ComputeDeviceKind::Cpu),
        ])
        .expect("a union of two gang leaves plans");
        assert_eq!(
            fanned_out
                .properties()
                .output_partitioning()
                .partition_count(),
            2
        );
        let refusal = stage_refusal(ComputeDeviceKind::Cpu, &fanned_out)
            .expect("a two-partition gang stage is refused");
        match restored(refusal) {
            JammiError::GangFanOut { job_id, partitions } => {
                assert_eq!(job_id, "job-7");
                assert_eq!(partitions, 2);
            }
            other => panic!("expected GangFanOut, got {other:?}"),
        }
    }

    /// A plan requiring no device kind and carrying no gang is never
    /// refused by this executor, whatever its own kind.
    #[test]
    fn a_plan_with_no_device_requirement_and_no_gang_is_held() {
        let plain: Arc<dyn ExecutionPlan> =
            Arc::new(datafusion::physical_plan::empty::EmptyExec::new(Arc::new(
                arrow::datatypes::Schema::empty(),
            )));
        assert!(stage_refusal(ComputeDeviceKind::Cuda, &plain).is_none());
        assert!(stage_refusal(ComputeDeviceKind::Cpu, &plain).is_none());
    }
}
