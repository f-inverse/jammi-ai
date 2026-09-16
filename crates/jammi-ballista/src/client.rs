//! `submit_physical_plan` — the seam a scheduler-role process's
//! `PlacedGangSubmitter` (jammi-ai's session seam, installed by the
//! scheduler role) calls to place a plan on the cluster instead of running
//! it in-process.

use std::sync::Arc;

use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::ExecutionPlan;

use ballista_core::config::BallistaConfig as BallistaClientConfig;
use ballista_core::execution_plans::execute_physical_plan;
use ballista_core::extension::SessionConfigExt;

use datafusion_proto::protobuf::PhysicalPlanNode;

use jammi_ai::session::InferenceSession;

use crate::codec::JammiCodec;
use crate::engine::stage_device_kind;
use crate::error::{Error, Result};

/// Submit `plan` to the scheduler at `scheduler_url` (`http://host:port`)
/// through [`JammiCodec`], returning the collected stream. `session`'s own
/// `SessionConfig` is upgraded for Ballista (`SessionConfigExt::
/// upgrade_for_ballista`) so the scheduler resolves the same UDFs/session
/// options the submitter's plan was built under.
///
/// The KIND MATCH refusal (contract §3, LANE pressure-round correction):
/// when `plan` requires a device kind ([`stage_device_kind`] — the SAME
/// predicate `placement::DevicePlacement` uses — a `GangExec`'s stamped
/// kind or an `InferenceExec`'s, CPU included) and NO registered compute
/// executor lists THAT EXACT kind, this call refuses typed, naming the
/// kind, BEFORE submitting rather than parking the plan unschedulable on
/// the scheduler. Read from `session`'s own catalog — the same shared
/// store the scheduler's `DevicePlacement` reads from — so a submitter
/// always sees the same device inventory the placement decision itself
/// will.
pub async fn submit_physical_plan(
    session: &Arc<InferenceSession>,
    scheduler_url: &str,
    plan: Arc<dyn ExecutionPlan>,
) -> Result<SendableRecordBatchStream> {
    if let Some(required_kind) = stage_device_kind(&plan) {
        let devices = session
            .catalog()
            .list_compute_executor_devices()
            .await
            .map_err(Error::Catalog)?;
        let wire = crate::engine::device_kind_wire_str(required_kind);
        let has_match = devices
            .iter()
            .any(|(_, ds)| ds.iter().any(|d| d.kind == wire));
        if !has_match {
            return Err(Error::Config(format!(
                "jammi-ballista: this plan requires device_kind {required_kind:?} but no \
                 registered compute executor lists a {wire} device — refused before \
                 submitting, never parked unschedulable"
            )));
        }
    }
    let codec = JammiCodec::new(session);
    let session_config = session.context().copied_config().upgrade_for_ballista();
    let session_id = session.context().session_id();
    let stream = execute_physical_plan::<PhysicalPlanNode>(
        scheduler_url.to_string(),
        &BallistaClientConfig::default(),
        plan,
        &codec,
        session_id,
        session_config,
    )
    .await?;
    Ok(stream)
}
