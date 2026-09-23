//! `DevicePlacement` — the `TaskDistributionPolicy::Custom` jammi installs
//! on every scheduler role: one policy carrying the submitter exclusion,
//! the device predicate, and the re-launch guard.
//!
//! Round-robin over executor slots, with three refinements over plain
//! round-robin:
//!
//! 1. A stage whose plan contains a `PlacedAttemptExec` is never bound to the
//!    executor whose id equals the descriptor's own `submitter` instance id
//!    — the submitter's own host holds a rank/job admission for the whole
//!    await, so binding the attempt's task back to it would
//!    deadlock the placed run against itself.
//! 2. KIND MATCH: a stage
//!    whose plan carries a required device kind — a `PlacedAttemptExec` (its
//!    descriptor's own stamped `device_kind`, CPU included) or an
//!    `InferenceExec` (its `spec().device_kind`) —
//!    [`crate::engine::required_device_kind`], the ONE predicate this policy
//!    and `client::submit_physical_plan`'s pre-submission refusal both use
//!    — binds only to an executor whose OWN registration lists THAT EXACT
//!    kind (`compute_executors.devices`,
//!    [`jammi_db::catalog::Catalog::list_compute_executor_devices`] — never
//!    a join through `workers.instance_id`, see that method's doc). A stage
//!    carrying no device kind (a plain scan/shuffle stage) is unconstrained
//!    by this refinement. The predicate is KIND MATCH, never "is this
//!    GPU-bound": a placed attempt's kind is a property of its DESCRIPTOR, not of the
//!    node type — a CPU-stamped attempt must bind to a CPU executor, exactly
//!    as a CPU-stamped `InferenceExec` does.
//! 3. A `PlacedAttemptExec` stage whose job row is already `claimed_by` an executor
//!    OTHER than this stage's own submitter is never bound to ANY slot —
//!    the bind-time half of the re-launch guard (`transfer_claim` is the
//!    other half): Ballista's own reset-on-
//!    `ExecutorLost` can re-offer an attempt's task for binding after the
//!    original launch already transferred the claim, and this predicate
//!    refuses that second launch before it ever reaches an executor. A job
//!    row this policy cannot read (deleted mid-round, or a genuine fault)
//!    is folded into the SAME refusal — never bind an attempt's task whose
//!    ownership cannot be verified.
//!
//! The slot CAS ([`jammi_db::catalog::Catalog::bind_compute_slots`]) happens
//! per candidate BEFORE the graph's task info is stamped:
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

use ballista_core::serde::protobuf::{job_status, AvailableTaskSlots};
use ballista_core::serde::scheduler::PartitionId;
use ballista_core::JobId;
use ballista_scheduler::cluster::{BoundTask, DistributionPolicy};
use ballista_scheduler::state::execution_graph::{create_task_info, TaskDescription};
use ballista_scheduler::state::task_manager::JobInfoCache;

use jammi_db::catalog::Catalog;

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

/// A catalog fault in a bind round, in the form the `DistributionPolicy`
/// contract returns: `External` over the typed engine error, so a reader of
/// the scheduler's log sees the fault as it was (a `Catalog`, a
/// `BackendDriver`), never a stringly re-spelling of it.
fn catalog_fault(e: jammi_db::error::JammiError) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

/// The line [`DevicePlacement`] writes at the one call site that binds a
/// task to an executor, naming the job, stage, partition and executor as
/// fields — the determinant a distributed oracle greps the scheduler's log
/// for.
pub const BOUND_TASK_LOG: &str = "jammi-ballista DevicePlacement: bound task";

/// Whether `devices` lists `kind` — the KIND MATCH refinement 2 needs
/// (module doc), never "any GPU exists" or "any device exists". The ONE
/// predicate this binder and `client::unheld`'s pre-submission refusal
/// read against a registered executor's device inventory.
pub(crate) fn lists_kind(
    devices: &[jammi_db::catalog::instance::DeviceFact],
    kind: jammi_datafusion::ComputeDeviceKind,
) -> bool {
    let wire = kind.wire_str();
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

        // Refinement 2's sole authority, read once per call (module doc). A
        // catalog fault here, as at the slot CAS below, leaves this call as
        // the typed engine error it is: the scheduler's event loop logs the
        // bind round's failure and offers again on its next revive, so no
        // job fails on a round the catalog could not serve.
        let executor_devices: HashMap<String, Vec<jammi_db::catalog::instance::DeviceFact>> = self
            .catalog
            .list_compute_executor_devices()
            .await
            .map_err(catalog_fault)?
            .into_iter()
            .collect();

        // Refinement 3's row-read cache, keyed by the TRAINING job's catalog
        // job_id (`PlacedAttempt::job_id`, read lazily below — never
        // Ballista's own `JobId`, see `placed_attempt_of`'s doc) so a
        // job with multiple runnable placed-attempt stages across rounds reads its
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
                // Refinement 3's re-launch guard reads the descriptor's OWN
                // training job id (`crate::engine::placed_attempt_of`'s
                // doc), never the outer loop's Ballista `JobId`.
                let placed_attempt = crate::engine::placed_attempt_of(&stage.plan).cloned();
                if let Some(descriptor) = &placed_attempt {
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
                                "jammi-ballista DevicePlacement: refusing to bind a PlacedAttemptExec \
                                 whose job row is already claimed by another instance — the \
                                 re-launch guard"
                            );
                            black_list.push(stage.stage_id);
                            continue;
                        }
                    }
                }
                let attempt_submitter = placed_attempt.as_ref().map(|d| d.submitter.clone());
                let required_kind = crate::engine::required_device_kind(&stage.plan);
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
                            && attempt_submitter.as_deref() != Some(executor_id.as_str())
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
                            .map_err(catalog_fault)?;
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
                        // determinant: the
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
                            attempt_submitter = ?attempt_submitter,
                            required_kind = ?required_kind,
                            "{BOUND_TASK_LOG}"
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
