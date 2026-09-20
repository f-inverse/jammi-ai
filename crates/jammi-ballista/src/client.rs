//! `submit_physical_plan` — the seam a scheduler-role process's
//! `PlacedGangSubmitter` (jammi-ai's session seam, installed by the
//! scheduler role) calls to place a plan on the cluster instead of running
//! it in-process.
//!
//! A placed task's failure reaches this client as the string Ballista
//! copied from hop to hop (`DataFusionError::Execution("Job {id} failed: …")`).
//! When that string carries the [`TaskErrorEnvelope`] the executor's engine
//! wrote (`engine::envelope_task_error`), this seam hands its caller the
//! typed `JammiError` back as `DataFusionError::External` — the leaf the
//! engine's classifier (`JammiError::from(DataFusionError)`) already
//! restores, so a placed refusal classifies exactly as the in-process one.

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
use jammi_db::error::JammiError;
use jammi_wire::TaskErrorEnvelope;

use crate::codec::JammiCodec;
use crate::engine::required_device_kind;
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

/// Submit `plan` to the scheduler at `scheduler_url` (`http://host:port`)
/// through [`JammiCodec`], returning the collected stream. `session`'s own
/// `SessionConfig` is upgraded for Ballista (`SessionConfigExt::
/// upgrade_for_ballista`) so the scheduler resolves the same UDFs/session
/// options the submitter's plan was built under.
///
/// The KIND MATCH refusal:
/// when `plan` requires a device kind ([`required_device_kind`] — the SAME
/// predicate `placement::DevicePlacement` uses — a `GangExec`'s stamped
/// kind or an `InferenceExec`'s, CPU included) and NO registered compute
/// executor lists THAT EXACT kind, this call refuses typed, naming the
/// kind, BEFORE submitting rather than parking the plan unschedulable on
/// the scheduler. Read from `session`'s own catalog — the same shared
/// store the scheduler's `DevicePlacement` reads from — so a submitter
/// always sees the same device inventory the placement decision itself
/// will.
///
/// The job's failure surfaces from this call itself (Ballista awaits the
/// job's terminal status before handing back the stream) and a partition
/// fetch's from the stream; both pass through [`restore_task_error`].
pub async fn submit_physical_plan(
    session: &Arc<InferenceSession>,
    scheduler_url: &str,
    plan: Arc<dyn ExecutionPlan>,
) -> Result<SendableRecordBatchStream> {
    if let Some(required_kind) = required_device_kind(&plan) {
        // LIVE executors only (`cluster::executor_is_live`, the same
        // predicate the scheduler's binder applies): a row a killed GPU
        // executor left behind must not admit a plan the binder can never
        // place — that would park it, the outcome this refusal exists to
        // prevent.
        let rows = session
            .catalog()
            .list_compute_executors()
            .await
            .map_err(Error::Catalog)?;
        let now = chrono::Utc::now();
        let wire = crate::engine::device_kind_wire_str(required_kind);
        let has_match = rows.iter().any(|r| {
            crate::cluster::executor_is_live(r, now) && r.devices.iter().any(|d| d.kind == wire)
        });
        if !has_match {
            return Err(Error::Config(format!(
                "jammi-ballista: this plan requires device_kind {required_kind:?} but no \
                 live registered compute executor lists a {wire} device — refused before \
                 submitting, never parked unschedulable"
            )));
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

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
