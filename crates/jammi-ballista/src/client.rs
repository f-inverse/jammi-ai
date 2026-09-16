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
use crate::engine::stage_is_gpu_bound;
use crate::error::{Error, Result};

/// Submit `plan` to the scheduler at `scheduler_url` (`http://host:port`)
/// through [`JammiCodec`], returning the collected stream. `session`'s own
/// `SessionConfig` is upgraded for Ballista (`SessionConfigExt::
/// upgrade_for_ballista`) so the scheduler resolves the same UDFs/session
/// options the submitter's plan was built under.
///
/// The device-less refusal (contract §3): when `plan` is GPU-bound
/// ([`stage_is_gpu_bound`] — the SAME predicate `placement::DevicePlacement`
/// uses) and NO registered compute executor lists a `cuda`/`metal` device,
/// this call refuses typed BEFORE submitting rather than parking the plan
/// unschedulable on the scheduler. Read from `session`'s own catalog — the
/// same shared store the scheduler's `DevicePlacement` reads from — so a
/// submitter always sees the same device inventory the placement decision
/// itself will.
pub async fn submit_physical_plan(
    session: &Arc<InferenceSession>,
    scheduler_url: &str,
    plan: Arc<dyn ExecutionPlan>,
) -> Result<SendableRecordBatchStream> {
    if stage_is_gpu_bound(&plan) {
        let devices = session
            .catalog()
            .list_compute_executor_devices()
            .await
            .map_err(Error::Catalog)?;
        let has_gpu = devices
            .iter()
            .any(|(_, ds)| ds.iter().any(|d| d.kind == "cuda" || d.kind == "metal"));
        if !has_gpu {
            return Err(Error::Config(
                "jammi-ballista: this plan is GPU-bound but no registered compute executor \
                 lists a cuda/metal device — refused before submitting, never parked \
                 unschedulable"
                    .to_string(),
            ));
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
