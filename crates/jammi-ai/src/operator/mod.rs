pub mod ann_search_exec;
pub mod inference_exec;
pub mod key_check_exec;
pub mod numbered_input_exec;
pub mod placed_attempt_exec;
pub mod row_cost_exec;

use std::sync::Arc;

use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties};

/// `plan` over one partition: a stock coalesce, left out where it would be
/// the identity.
pub(crate) fn single_partition(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    if plan.output_partitioning().partition_count() > 1 {
        Arc::new(CoalescePartitionsExec::new(plan))
    } else {
        plan
    }
}
