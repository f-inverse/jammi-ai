//! The submit client: the two verbs the client role's `ComputePlane`
//! makes, [`unheld`] (the admission: can a live executor hold this plan?
//! — [`unheld_by`], a pure predicate over the plan's requirements and the
//! live inventory) and [`place`] (the submission itself); the two are
//! separate so a caller that must run an unheld plan somewhere else can
//! tell a refusal from a submission's failure. [`submit_physical_plan`] is
//! the two in sequence, refusing typed.
//!
//! A placed task's failure reaches this client as the string Ballista
//! copied from hop to hop (`DataFusionError::Execution("Job {id} failed: …")`).
//! When that string carries the [`TaskErrorEnvelope`] the executor's engine
//! wrote (`engine::envelope_task_error`), this seam hands its caller the
//! typed `JammiError` back as `DataFusionError::External` — the leaf the
//! engine's classifier (`JammiError::from(DataFusionError)`) already
//! restores, so a placed refusal classifies exactly as the in-process one.

use std::collections::BTreeSet;
use std::sync::Arc;

use datafusion::error::DataFusionError;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::ExecutionPlan;
use futures::TryStreamExt;

use ballista_core::config::BallistaConfig as BallistaClientConfig;
use ballista_core::execution_plans::execute_physical_plan;
use ballista_core::extension::SessionConfigExt;

use datafusion_proto::protobuf::PhysicalPlanNode;

use jammi_ai::session::InferenceSession;
use jammi_db::catalog::compute_repo::ComputeExecutorRecord;
use jammi_db::compute_plane::{PlanRequirements, Unheld};
use jammi_db::error::JammiError;
use jammi_db::store::manifest::ComputeDeviceKind;
use jammi_wire::TaskErrorEnvelope;

use crate::codec::JammiCodec;
use crate::engine::plan_requirements;
use crate::error::{Error, Result};

/// The client half of the envelope: an `Execution` message carrying a
/// [`TaskErrorEnvelope`] becomes `External` over the decoded `JammiError`;
/// one carrying a stale or malformed envelope becomes `External` over the
/// typed decode refusal; any other error is handed back as it is.
pub fn restore_task_error(e: DataFusionError) -> DataFusionError {
    let DataFusionError::Execution(message) = &e else {
        return e;
    };
    match TaskErrorEnvelope::extract(message) {
        Ok(Some(typed)) => DataFusionError::External(Box::new(typed)),
        Ok(None) => e,
        Err(refused) => DataFusionError::External(Box::new(JammiError::from(refused))),
    }
}

/// Why no executor in `live` can hold a plan with `requirements`, or
/// `None` when one can — the admission, a pure predicate over the plan's
/// requirements and the live inventory. In order: no live executor lists
/// THE EXACT kind the plan requires (`placement::lists_kind`, KIND MATCH —
/// CPU is a kind too; the refusal names every kind the live executors do
/// list, none when none is live); a plan requiring no kind finds no live
/// executor at all; every eligible executor is one the plan excludes (a
/// gang's own submitter). The same rules `placement::DevicePlacement`
/// binds by, so a plan admitted here is one the binder can bind.
pub fn unheld_by(
    requirements: &PlanRequirements,
    live: &[&ComputeExecutorRecord],
) -> Option<Unheld> {
    let of_kind = live
        .iter()
        .filter(|r| {
            requirements
                .device_kind
                .is_none_or(|required| crate::placement::lists_kind(&r.devices, required))
        })
        .collect::<Vec<_>>();
    if let (Some(required), true) = (requirements.device_kind, of_kind.is_empty()) {
        let held = live
            .iter()
            .flat_map(|r| r.devices.iter())
            .filter_map(|d| ComputeDeviceKind::parse(&d.kind))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        return Some(Unheld::NoExecutorOfKind { required, held });
    }
    if live.is_empty() {
        return Some(Unheld::NoLiveExecutor);
    }
    match &requirements.excluded_executor {
        Some(submitter) if of_kind.iter().all(|r| &r.executor_id == submitter) => {
            Some(Unheld::OnlyTheSubmitter {
                submitter: submitter.clone(),
            })
        }
        _ => None,
    }
}

/// Why the cluster cannot hold `plan` right now, or `None` when it can —
/// [`unheld_by`] over the plan's own requirements (`engine::
/// plan_requirements`) and the LIVE executors (`cluster::executor_is_live`,
/// the predicate the scheduler's binder applies: a row a killed executor
/// left behind admits nothing), decided BEFORE submitting, so a plan no
/// executor can bind is never parked unschedulable on the scheduler. Read
/// from `session`'s own catalog — the same shared store the scheduler's
/// `DevicePlacement` reads from — so a submitter always sees the inventory
/// the placement decision itself will.
pub async fn unheld(
    session: &InferenceSession,
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Option<Unheld>> {
    let rows = session
        .catalog()
        .list_compute_executors()
        .await
        .map_err(Error::Catalog)?;
    let now = chrono::Utc::now();
    let live = rows
        .iter()
        .filter(|r| crate::cluster::executor_is_live(r, now))
        .collect::<Vec<_>>();
    Ok(unheld_by(&plan_requirements(plan), &live))
}

/// Submit `plan` to the scheduler at `scheduler_url` (`http://host:port`)
/// through [`JammiCodec`], returning the collected stream — the submission
/// alone; [`unheld`] is the caller's admission. `session`'s own
/// `SessionConfig` is upgraded for Ballista (`SessionConfigExt::
/// upgrade_for_ballista`) so the scheduler resolves the same UDFs/session
/// options the submitter's plan was built under.
///
/// The job's failure surfaces from this call itself (Ballista awaits the
/// job's terminal status before handing back the stream) and a partition
/// fetch's from the stream; both pass through [`restore_task_error`].
pub async fn place(
    session: &Arc<InferenceSession>,
    scheduler_url: &str,
    plan: Arc<dyn ExecutionPlan>,
) -> Result<SendableRecordBatchStream> {
    let codec = JammiCodec::new(session);
    let session_config = session.context().copied_config().upgrade_for_ballista();
    let session_id = session.context().session_id();
    let schema = plan.schema();
    let stream = execute_physical_plan::<PhysicalPlanNode>(
        scheduler_url.to_string(),
        &BallistaClientConfig::default(),
        plan,
        &codec,
        session_id,
        session_config,
    )
    .await
    .map_err(restore_task_error)?;
    Ok(Box::pin(RecordBatchStreamAdapter::new(
        schema,
        stream.map_err(restore_task_error),
    )))
}

/// [`unheld`] then [`place`]: submit `plan` to the scheduler at
/// `scheduler_url`, refusing typed ([`Error::Unheld`]) before submitting
/// when no live executor can hold it.
pub async fn submit_physical_plan(
    session: &Arc<InferenceSession>,
    scheduler_url: &str,
    plan: Arc<dyn ExecutionPlan>,
) -> Result<SendableRecordBatchStream> {
    if let Some(why) = unheld(session, &plan).await? {
        return Err(Error::Unheld(why));
    }
    place(session, scheduler_url, plan).await
}

#[cfg(test)]
mod tests {
    use super::*;

    use jammi_db::catalog::instance::DeviceFact;
    use jammi_db::catalog::status::ComputeExecutorStatus;

    fn executor(id: &str, kinds: &[&str]) -> ComputeExecutorRecord {
        ComputeExecutorRecord {
            executor_id: id.to_string(),
            instance_id: id.to_string(),
            host: "127.0.0.1".to_string(),
            port: 0,
            grpc_port: 0,
            task_slots: 1,
            available_slots: 1,
            status: ComputeExecutorStatus::Active,
            heartbeat_at: jammi_db::catalog::lease::canonical_stamp_now(),
            metadata: String::new(),
            devices: kinds
                .iter()
                .map(|kind| DeviceFact {
                    kind: kind.to_string(),
                    ordinal: 0,
                })
                .collect(),
        }
    }

    /// The admission table: an empty inventory, a kind no live executor
    /// lists, an inventory whose only eligible executor is the plan's own
    /// submitter, and the admitted shapes — a plan requiring nothing, a
    /// plan whose kind a peer lists, a gang with a peer of its kind.
    #[test]
    fn unheld_by_reads_the_requirements_against_the_live_inventory() {
        let cpu_peer = executor("peer", &["cpu"]);
        let cuda_self = executor("self", &["cuda"]);
        let cuda_peer = executor("other", &["cuda"]);
        let none = PlanRequirements::default();
        let cuda = PlanRequirements {
            device_kind: Some(ComputeDeviceKind::Cuda),
            excluded_executor: None,
        };
        let cuda_gang = PlanRequirements {
            device_kind: Some(ComputeDeviceKind::Cuda),
            excluded_executor: Some("self".to_string()),
        };

        assert_eq!(unheld_by(&none, &[]), Some(Unheld::NoLiveExecutor));
        assert_eq!(
            unheld_by(&cuda, &[]),
            Some(Unheld::NoExecutorOfKind {
                required: ComputeDeviceKind::Cuda,
                held: vec![],
            })
        );
        assert_eq!(unheld_by(&none, &[&cpu_peer]), None);
        assert_eq!(
            unheld_by(&cuda, &[&cpu_peer]),
            Some(Unheld::NoExecutorOfKind {
                required: ComputeDeviceKind::Cuda,
                held: vec![ComputeDeviceKind::Cpu],
            })
        );
        assert_eq!(unheld_by(&cuda, &[&cpu_peer, &cuda_self]), None);
        assert_eq!(
            unheld_by(&cuda_gang, &[&cpu_peer, &cuda_self]),
            Some(Unheld::OnlyTheSubmitter {
                submitter: "self".to_string(),
            })
        );
        assert_eq!(unheld_by(&cuda_gang, &[&cuda_self, &cuda_peer]), None);
    }

    /// The string a placed refusal reaches this client as, once every hop
    /// has prefixed it, classifies as the same variant and fields the
    /// in-process path raises.
    #[test]
    fn a_placed_refusal_classifies_as_the_in_process_one() {
        let envelope = TaskErrorEnvelope::new(JammiError::SourceNotFound {
            source_id: "patents".into(),
        });
        let arrived = DataFusionError::Execution(format!(
            "Job 7bY2 failed: Job failed due to stage 1 failed: Task failed due to runtime \
             execution error: DataFusionError(External({envelope:?}))\n"
        ));
        match JammiError::from(restore_task_error(arrived)) {
            JammiError::SourceNotFound { source_id } => assert_eq!(source_id, "patents"),
            other => panic!("expected SourceNotFound, got {other:?}"),
        }
    }

    /// A foreign failure string is handed back as it arrived; a stale
    /// envelope is the typed decode refusal, never the string.
    #[test]
    fn a_foreign_failure_passes_and_a_stale_envelope_is_refused_typed() {
        let foreign = DataFusionError::Execution("Job 7bY2 failed: no alive executors".into());
        assert!(matches!(
            restore_task_error(foreign),
            DataFusionError::Execution(m) if m == "Job 7bY2 failed: no alive executors"
        ));
        let stale = DataFusionError::Execution(
            "Job 7bY2 failed: External error: jammi-error:2:00:Source not found: patents".into(),
        );
        match JammiError::from(restore_task_error(stale)) {
            JammiError::IncompatibleFormat { found, .. } => assert_eq!(found, "2"),
            other => panic!("expected IncompatibleFormat, got {other:?}"),
        }
    }
}
