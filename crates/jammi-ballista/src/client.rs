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
use crate::error::Result;

/// Submit `plan` to the scheduler at `scheduler_url` (`http://host:port`)
/// through [`JammiCodec`], returning the collected stream. `session`'s own
/// `SessionConfig` is upgraded for Ballista (`SessionConfigExt::
/// upgrade_for_ballista`) so the scheduler resolves the same UDFs/session
/// options the submitter's plan was built under.
pub async fn submit_physical_plan(
    session: &Arc<InferenceSession>,
    scheduler_url: &str,
    plan: Arc<dyn ExecutionPlan>,
) -> Result<SendableRecordBatchStream> {
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
