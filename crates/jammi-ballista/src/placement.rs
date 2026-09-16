//! `DevicePlacement` — the `TaskDistributionPolicy::Custom` jammi installs
//! on every scheduler role (contract `feat_500-wave4.md` §3 and §9 B1;
//! UNITS §U8a introduced the submitter exclusion, U8b EXTENDS the SAME
//! policy with the device predicate and the re-launch guard rather than
//! adding a second one).
//!
//! Round-robin over executor slots, with three refinements over plain
//! round-robin:
//!
//! 1. A stage whose plan contains a `GangExec` is never bound to the
//!    executor whose id equals the descriptor's own `submitter` instance id
//!    — the submitter's own host holds a rank/job admission for the whole
//!    await (contract §9 B1/B2), so binding the gang task back to it would
//!    deadlock the placed run against itself.
//! 2. KIND MATCH (contract §3, LANE pressure-round correction): a stage
//!    whose plan carries a required device kind — a `GangExec` (its
//!    descriptor's own stamped `device_kind`, CPU included) or an
//!    `InferenceExec` (its `device_kind()`) —
//!    [`crate::engine::stage_device_kind`], the ONE predicate this policy
//!    and `client::submit_physical_plan`'s pre-submission refusal both use
//!    — binds only to an executor whose OWN registration lists THAT EXACT
//!    kind (`compute_executors.devices`,
//!    [`jammi_db::catalog::Catalog::list_compute_executor_devices`] — never
//!    a join through `workers.instance_id`, see that method's doc). A stage
//!    carrying no device kind (a plain scan/shuffle stage) is unconstrained
//!    by this refinement. This replaced an earlier "is this GPU-bound"
//!    predicate that would have refused every `GangExec` on an all-CPU
//!    cluster (a gang's kind is a property of its DESCRIPTOR, not of the
//!    node type — a CPU-stamped gang must bind to a CPU executor, exactly
//!    as a CPU-stamped `InferenceExec` does).
//! 3. A `GangExec` stage whose job row is already `claimed_by` an executor
//!    OTHER than this stage's own submitter is never bound to ANY slot —
//!    the bind-time half of the re-launch guard (contract §2.4's `
//!    transfer_claim` is the other half): Ballista's own reset-on-
//!    `ExecutorLost` can re-offer a gang's task for binding after the
//!    original launch already transferred the claim, and this predicate
//!    refuses that second launch before it ever reaches an executor. A job
//!    row this policy cannot read (deleted mid-round, or a genuine fault)
//!    is folded into the SAME refusal — never bind a gang task whose
//!    ownership cannot be verified.
//!
//! The slot CAS ([`jammi_db::catalog::Catalog::bind_compute_slots`]) happens
//! per candidate BEFORE the graph's task info is stamped (contract §9 A10):
//! a lost CAS — the catalog's committed `available_slots` is the truth,
//! never this call's in-memory snapshot — leaves that slot's local count at
//! zero for the rest of this call and the task UNSTAMPED, tried again on the
//! executor's OWN `bind_schedulable_tasks` retry.
//!
//! `bind_task_round_robin`/`bind_task_bias` (Ballista's own built-in
//! policies) are `pub(crate)` to `ballista-scheduler` and not reachable from
//! here. [`DevicePlacement`] is jammi's own re-implementation of that
//! algorithm, extended as above. `bind_round_robin` below is a SEPARATE,
//! much smaller re-implementation with none of these three refinements —
//! `CatalogClusterState::bind_schedulable_tasks`'s fallback for the
//! `TaskDistributionPolicy::Bias`/`RoundRobin` built-in arms, which
//! `jammi-server` never selects (there is no knob: `roles::host_scheduler`
//! always installs `Custom(DevicePlacement)`) and which, unlike `Custom`,
//! never reserves a catalog slot — kept only so `ClusterState`'s generic
//! contract is complete and testable, never safe to reach from
//! configuration under a catalog two schedulers share.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::physical_plan::ExecutionPlan;

use ballista_core::serde::protobuf::{job_status, AvailableTaskSlots};
use ballista_core::serde::scheduler::PartitionId;
use ballista_core::JobId;
use ballista_scheduler::cluster::{BoundTask, DistributionPolicy};
use ballista_scheduler::state::execution_graph::{create_task_info, TaskDescription};
use ballista_scheduler::state::task_manager::JobInfoCache;

use jammi_db::catalog::Catalog;

/// The `GangExec`'s own descriptor found anywhere in `plan`, if any — a
/// `GangExec` is a leaf (zero children, `crate::codec`'s decode never wraps
/// it), so a plain depth-first search over `.children()` finds it wherever
/// the stage's shuffle-writer wrapping placed it. `descriptor().job_id` is
/// jammi's OWN fine-tune catalog job id — DISTINCT from the `JobId` key
/// `bind_tasks`' own `running_jobs` map uses, which is Ballista's
/// internally-minted submission id (unrelated id spaces: a real Ballista
/// submission never gives them the same value, only a hermetic test
/// fixture that deliberately aliases them would) — refinement 3's re-launch
/// guard below reads THIS job_id, never the outer loop's.
fn gang_descriptor_of(
    plan: &Arc<dyn ExecutionPlan>,
) -> Option<jammi_ai::operator::gang_exec::GangDescriptor> {
    if let Some(exec) = plan.downcast_ref::<jammi_ai::operator::gang_exec::GangExec>() {
        return Some(exec.descriptor().clone());
    }
    plan.children().into_iter().find_map(gang_descriptor_of)
}

/// jammi's shipped scheduler task-distribution policy (see the module doc).
/// `name()` distinguishes it in scheduler logs/metrics from Ballista's own
/// built-in policies.
#[derive(Clone)]
pub struct DevicePlacement {
    catalog: Arc<Catalog>,
}

impl DevicePlacement {
    pub fn new(catalog: Arc<Catalog>) -> Self {
        Self { catalog }
    }
}

impl std::fmt::Debug for DevicePlacement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DevicePlacement").finish()
    }
}

/// Whether `devices` lists `kind` — the KIND MATCH refinement 2 needs
/// (module doc), never "any GPU exists" or "any device exists".
fn lists_kind(
    devices: &[jammi_db::catalog::instance::DeviceFact],
    kind: jammi_db::store::manifest::ComputeDeviceKind,
) -> bool {
    let wire = crate::engine::device_kind_wire_str(kind);
    devices.iter().any(|d| d.kind == wire)
}

#[async_trait]
impl DistributionPolicy for DevicePlacement {
    async fn bind_tasks(
        &self,
        mut slots: Vec<&mut AvailableTaskSlots>,
        running_jobs: Arc<HashMap<JobId, JobInfoCache>>,
    ) -> DfResult<Vec<BoundTask>> {
        let mut bound: Vec<BoundTask> = Vec::new();
        if slots.is_empty() {
            return Ok(bound);
        }

        // Refinement 2's sole authority, read once per call (module doc).
        let executor_devices: HashMap<String, Vec<jammi_db::catalog::instance::DeviceFact>> = self
            .catalog
            .list_compute_executor_devices()
            .await
            .map_err(|e| {
                DataFusionError::Execution(format!(
                    "jammi-ballista DevicePlacement: device read: {e}"
                ))
            })?
            .into_iter()
            .collect();

        // Refinement 3's row-read cache, keyed by the FINE-TUNE catalog
        // job_id (`GangDescriptor::job_id`, read lazily below — never
        // Ballista's own `JobId`, see `gang_descriptor_of`'s doc) so a
        // job with multiple runnable gang stages across rounds reads its
        // row at most once per `bind_tasks` call. `None` (unclaimed,
        // should not occur for a `running` row but is not itself a fault)
        // never triggers the guard; an unreadable row is folded into
        // "claimed by someone unverifiable" via a sentinel that can never
        // equal a real submitter id.
        let mut claim_of: HashMap<String, Option<String>> = HashMap::new();

        // Descending by capacity, same shape Ballista's own round robin uses.
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
                let gang_descriptor = gang_descriptor_of(&stage.plan);
                if let Some(descriptor) = &gang_descriptor {
                    let claim = match claim_of.get(&descriptor.job_id) {
                        Some(cached) => cached.clone(),
                        None => {
                            let row = jammi_db::tenant_scope::TenantBinding::admin_scope(async {
                                self.catalog.get_job(&descriptor.job_id).await
                            })
                            .await;
                            let claim = row
                                .map(|r| r.claimed_by)
                                .unwrap_or_else(|_| Some(String::new()));
                            claim_of.insert(descriptor.job_id.clone(), claim.clone());
                            claim
                        }
                    };
                    if let Some(claimant) = &claim {
                        if claimant != &descriptor.submitter {
                            tracing::warn!(
                                job_id = %descriptor.job_id,
                                ballista_job_id = %job_id,
                                stage_id = stage.stage_id,
                                submitter = %descriptor.submitter,
                                claimed_by = %claimant,
                                "jammi-ballista DevicePlacement: refusing to bind a GangExec \
                                 whose job row is already claimed by another instance — the \
                                 re-launch guard (contract §2.4)"
                            );
                            black_list.push(stage.stage_id);
                            continue;
                        }
                    }
                }
                let gang_submitter = gang_descriptor.as_ref().map(|d| d.submitter.clone());
                let required_kind = crate::engine::stage_device_kind(&stage.plan);
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
                    let mut attempts = 0usize;
                    while attempts < slots.len() {
                        if idx >= slots.len() {
                            idx = 0;
                        }
                        let executor_id = slots[idx].executor_id.clone();
                        let eligible = slots[idx].slots > 0
                            && gang_submitter.as_deref() != Some(executor_id.as_str())
                            && match required_kind {
                                None => true,
                                Some(kind) => executor_devices
                                    .get(&executor_id)
                                    .map(|ds| lists_kind(ds, kind))
                                    .unwrap_or(false),
                            };
                        if !eligible {
                            idx += 1;
                            attempts += 1;
                            continue;
                        }
                        // Refinements' CAS (module doc): the catalog's slot
                        // count is the truth, never this call's snapshot.
                        let won = self
                            .catalog
                            .bind_compute_slots(&executor_id, 1)
                            .await
                            .map_err(|e| {
                                DataFusionError::Execution(format!(
                                    "jammi-ballista DevicePlacement: slot CAS: {e}"
                                ))
                            })?;
                        if !won {
                            slots[idx].slots = 0;
                            idx += 1;
                            attempts += 1;
                            continue;
                        }
                        let task_id = *task_id_gen;
                        *task_id_gen += 1;
                        stage.task_infos[partition_id] =
                            Some(create_task_info(executor_id.clone(), task_id));
                        let partition = PartitionId {
                            job_id: job_id.to_owned(),
                            stage_id: stage.stage_id,
                            partition_id,
                        };
                        // The distributed lane's per-executor task-binding
                        // determinant (contract §2.5, acceptance (a3)): the
                        // Ballista client API exposes no `job_id`/task
                        // attribution to a `submit_physical_plan` caller, so
                        // the lane's own oracle (`tests/distributed/main.rs`)
                        // greps the scheduler process's OWN log for this line
                        // naming which registered executor each stage/
                        // partition bound to, rather than fabricating a
                        // weaker check.
                        tracing::info!(
                            job_id = %job_id,
                            stage_id = stage.stage_id,
                            partition_id,
                            executor_id = %executor_id,
                            gang_submitter = ?gang_submitter,
                            required_kind = ?required_kind,
                            "jammi-ballista DevicePlacement: bound task"
                        );
                        bound.push((
                            executor_id.clone(),
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
                        idx += 1;
                        placed_any = true;
                        break;
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
        "jammi-device-placement"
    }
}

/// The `Bias`/`RoundRobin` fallback — see the module doc for why this is a
/// separate, deliberately unrefined re-implementation.
pub(crate) async fn bind_round_robin(
    mut slots: Vec<&mut AvailableTaskSlots>,
    running_jobs: Arc<HashMap<JobId, JobInfoCache>>,
) -> Vec<BoundTask> {
    let mut bound: Vec<BoundTask> = Vec::new();
    let mut total_slots: u32 = slots.iter().map(|s| s.slots).sum();
    if total_slots == 0 || slots.is_empty() {
        return bound;
    }
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
                if idx >= slots.len() {
                    idx = 0;
                }
                if slots[idx].slots == 0 {
                    black_list.push(stage.stage_id);
                    break;
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
                    return bound;
                }
            }
            if !placed_any {
                black_list.push(stage.stage_id);
            }
        }
    }
    bound
}
