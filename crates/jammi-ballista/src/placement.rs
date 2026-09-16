//! `PlacementPolicy` — the `TaskDistributionPolicy::Custom` jammi installs
//! on every scheduler role, from U8a on (contract `feat_500-wave4` §9 B1).
//!
//! Round-robin over executor slots, with ONE exclusion: a stage whose plan
//! contains a `GangExec` is never bound to the executor whose id equals the
//! descriptor's own `submitter` instance id — the submitter's own host holds
//! `Holder::JobRun` (soon `Awaiting`, §9 B2) for the whole await, so binding
//! the gang task back to it would deadlock the placed run against itself.
//!
//! U8b extends this SAME policy with the device predicate
//! (`DevicePlacement`) rather than replacing it.
//!
//! `bind_task_round_robin` (Ballista's own default-policy round robin) is
//! `pub(crate)` to `ballista-scheduler` and not reachable from here, so this
//! is jammi's own re-implementation of the same algorithm (slots sorted
//! descending by capacity, tasks fanned out round-robin), over the same
//! PUBLIC types the trait's own signature already requires
//! (`AvailableTaskSlots`, `JobInfoCache`, `BoundTask`, the `ExecutionGraph`
//! trait's `fetch_running_stage`, `TaskDescription`, `create_task_info` —
//! confirmed public in `ballista-scheduler-54.1.0/src/{cluster/mod.rs,
//! state/{execution_graph.rs,execution_stage.rs,task_manager.rs}}`).

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::error::Result as DfResult;
use datafusion::physical_plan::ExecutionPlan;

use ballista_core::serde::protobuf::{job_status, AvailableTaskSlots};
use ballista_core::serde::scheduler::PartitionId;
use ballista_core::JobId;
use ballista_scheduler::cluster::{BoundTask, DistributionPolicy};
use ballista_scheduler::state::execution_graph::{create_task_info, TaskDescription};
use ballista_scheduler::state::task_manager::JobInfoCache;

/// The submitter instance id a stage's `GangExec` must never be bound to, if
/// the stage contains one — a `GangExec` is a leaf (zero children,
/// `crate::codec`'s decode never wraps it), so a plain depth-first search
/// over `.children()` finds it wherever the stage's shuffle-writer wrapping
/// placed it.
fn gang_submitter_of(plan: &Arc<dyn ExecutionPlan>) -> Option<String> {
    if let Some(exec) = plan.downcast_ref::<jammi_ai::operator::gang_exec::GangExec>() {
        return Some(exec.descriptor().submitter.clone());
    }
    plan.children().into_iter().find_map(gang_submitter_of)
}

/// Round-robin task placement excluding a gang's own submitter (see module
/// doc). `name()` distinguishes it in scheduler logs/metrics from Ballista's
/// own built-in policies.
#[derive(Debug, Clone, Default)]
pub struct PlacementPolicy;

#[async_trait]
impl DistributionPolicy for PlacementPolicy {
    async fn bind_tasks(
        &self,
        mut slots: Vec<&mut AvailableTaskSlots>,
        running_jobs: Arc<HashMap<JobId, JobInfoCache>>,
    ) -> DfResult<Vec<BoundTask>> {
        let mut bound: Vec<BoundTask> = Vec::new();
        let mut total_slots: u32 = slots.iter().map(|s| s.slots).sum();
        if total_slots == 0 || slots.is_empty() {
            return Ok(bound);
        }
        // Descending by capacity, same shape as Ballista's own round robin.
        slots.sort_by(|a, b| b.slots.cmp(&a.slots));

        let mut idx = 0usize;
        for (job_id, job_info) in running_jobs.iter() {
            if !matches!(job_info.status, Some(job_status::Status::Running(_))) {
                continue;
            }
            let mut graph = job_info.execution_graph.write().await;
            let session_id = graph.session_id().to_string();
            let mut black_list: Vec<usize> = Vec::new();
            while let Some((stage, task_id_gen)) = graph.fetch_running_stage(&black_list) {
                let exclude = gang_submitter_of(&stage.plan);
                let runnable_partitions: Vec<usize> = stage
                    .task_infos
                    .iter()
                    .enumerate()
                    .filter(|(_, info)| info.is_none())
                    .map(|(p, _)| p)
                    .collect();
                if runnable_partitions.is_empty() {
                    black_list.push(stage.stage_id);
                    continue;
                }
                let mut placed_any = false;
                for partition_id in runnable_partitions {
                    // Advance to the next slot with capacity, skipping the
                    // excluded executor; give up on this task if every slot
                    // is either full or excluded.
                    let mut skipped = 0usize;
                    loop {
                        if skipped >= slots.len() {
                            break;
                        }
                        if idx >= slots.len() {
                            idx = 0;
                        }
                        let is_excluded =
                            exclude.as_deref() == Some(slots[idx].executor_id.as_str());
                        if slots[idx].slots == 0 || is_excluded {
                            idx += 1;
                            skipped += 1;
                            continue;
                        }
                        break;
                    }
                    if skipped >= slots.len() {
                        // No eligible (non-excluded, non-full) slot exists
                        // right now for this task; leave it unbound this
                        // round rather than mis-place it.
                        continue;
                    }
                    let executor_id = slots[idx].executor_id.clone();
                    let task_id = *task_id_gen;
                    *task_id_gen += 1;
                    stage.task_infos[partition_id] =
                        Some(create_task_info(executor_id.clone(), task_id));
                    let partition = PartitionId {
                        job_id: job_id.to_owned(),
                        stage_id: stage.stage_id,
                        partition_id,
                    };
                    bound.push((
                        executor_id,
                        TaskDescription {
                            session_id: session_id.clone(),
                            partition,
                            stage_attempt_num: stage.stage_attempt_num,
                            task_id,
                            task_attempt: stage.task_failure_numbers[partition_id],
                            plan: stage.plan.clone(),
                            session_config: stage.session_config.clone(),
                        },
                    ));
                    slots[idx].slots -= 1;
                    total_slots -= 1;
                    idx += 1;
                    placed_any = true;
                    if total_slots == 0 {
                        return Ok(bound);
                    }
                }
                if !placed_any {
                    black_list.push(stage.stage_id);
                }
            }
        }
        Ok(bound)
    }

    fn name(&self) -> &str {
        "jammi-round-robin"
    }
}
