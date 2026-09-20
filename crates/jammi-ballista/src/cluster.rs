//! `CatalogClusterState` / `CatalogJobState` — the catalog-backed
//! `ballista_scheduler::cluster::{ClusterState, JobState}` implementations.
//! Every table verb these two types call is generic, distributor-neutral
//! CRUD in `jammi_db::catalog::compute_repo` — this module is the ONLY place
//! Ballista's own `ClusterState`/`JobState` vocabulary meets it.
//!
//! **Execution graphs are never persisted** (Ballista 54.1 has no
//! execution-graph serialisation). `compute_jobs` mirrors OWNERSHIP and
//! STATUS text only. A scheduler restart therefore keeps executor
//! registrations and job status rows readable but starts with an
//! EMPTY in-memory graph map: [`CatalogJobState::get_execution_graph`]
//! answers `None` for a job this process's memory never built, even though
//! [`CatalogJobState::get_job_status`] still answers the row's status — the
//! new scheduler's own event loop tolerates that split (it is never asked to
//! resume a graph it does not have; the job is instead re-run through
//! jammi's own reclaim, never revived by Ballista).
//!
//! **An executor's loss is the typed failure of the jobs bound to it.**
//! Every plan jammi places roots in a leased row or a claimed gang, which
//! a relaunched stage can never take again (the row moved on under the
//! lost holder, the claim moved with it), so Ballista's stage relaunch —
//! its answer to a lost executor — can only end in that refusal, and only
//! once something else revives the scheduler's offers. The catalog-backed
//! state therefore fails the jobs itself, at the loss: [`ClusterState::
//! remove_executor`] is what Ballista's scheduler awaits before it posts
//! its own `ExecutorLost` event, and `PlacedJobs::fail_bound_to` runs
//! inside it — the failure reaches the submitter typed
//! ([`jammi_db::error::JammiError::ExecutorLost`]) and the job's cancel is
//! queued on the scheduler's one FIFO event loop ahead of the loss, so no
//! stage of a failed job is ever reset for relaunch.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex as StdMutex, OnceLock, RwLock as StdRwLock, Weak};

use async_trait::async_trait;
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::RwLock as AsyncRwLock;

use ballista_core::error::{BallistaError, Result as BallistaResult};
use ballista_core::serde::protobuf::{
    executor_status, job_status, ExecutorHeartbeat, ExecutorStatus, FailedJob, JobStatus,
    QueuedJob, RunningJob, SuccessfulJob,
};
use ballista_core::serde::scheduler::{ExecutorData, ExecutorMetadata};
use ballista_core::{ConfigProducer, JobId, JobStatusSubscriber};

use ballista_scheduler::cluster::event::ClusterEventSender;
use ballista_scheduler::cluster::{
    BoundTask, ClusterState, ClusterStateEvent, ClusterStateEventStream, ExecutorSlot, JobState,
    JobStateEvent, JobStateEventStream,
};
use ballista_scheduler::config::TaskDistributionPolicy;
use ballista_scheduler::scheduler_server::{SchedulerServer, SessionBuilder};
use ballista_scheduler::state::execution_graph::ExecutionGraphBox;
use ballista_scheduler::state::session_manager::create_datafusion_context;
use ballista_scheduler::state::task_manager::JobInfoCache;

use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_proto::protobuf::{LogicalPlanNode, PhysicalPlanNode};

use jammi_db::catalog::compute_repo::{ComputeExecutorRecord, ComputeJobRecord};
use jammi_db::catalog::status::ComputeExecutorStatus;
use jammi_db::catalog::Catalog;
use jammi_db::error::JammiError;
use jammi_wire::TaskErrorEnvelope;

use crate::placement::bind_round_robin;

/// The line the scheduler logs, once per job, as it fails a placed job
/// whose executor was lost. Carries `executor_id` and `job_id` as fields.
pub const EXECUTOR_LOST_LOG: &str = "executor lost: its placed job fails typed";

/// The scheduler's placed jobs, as the cluster state reaches them: what
/// `roles::host_scheduler` installs on [`CatalogClusterState`] once the
/// scheduler exists (the state is built before it). Held weakly — the
/// scheduler holds the state, so a strong handle here would keep a stopped
/// scheduler alive — and a role that has stopped has no job left to fail.
pub struct PlacedJobs {
    scheduler: Weak<SchedulerServer<LogicalPlanNode, PhysicalPlanNode>>,
    job_state: Arc<dyn JobState>,
}

impl PlacedJobs {
    pub(crate) fn new(
        scheduler: &Arc<SchedulerServer<LogicalPlanNode, PhysicalPlanNode>>,
        job_state: Arc<dyn JobState>,
    ) -> Self {
        Self {
            scheduler: Arc::downgrade(scheduler),
            job_state,
        }
    }

    /// Fail every running job with a task on `executor_id`, typed as
    /// [`JammiError::ExecutorLost`] naming both, and cancel it on the
    /// plane. The failure is written into the job's own graph as the
    /// [`TaskErrorEnvelope`] a failed task's error carries, then saved
    /// through the [`JobState`] — the seam that persists the job's status
    /// and hands it to the submitter's status stream — so the submitter's
    /// `client::restore_task_error` reads it back as the typed error. The
    /// cancel then retires the job on the scheduler (its graph leaves the
    /// running set, its other tasks are cancelled); the status the cancel
    /// saves afterwards reaches a subscriber that has already read the
    /// typed failure ahead of it.
    ///
    /// Ordering guarantee: called from [`ClusterState::remove_executor`],
    /// which Ballista's `SchedulerServer::remove_executor` awaits BEFORE it
    /// posts `ExecutorLost` to the scheduler's one FIFO event loop. The
    /// cancel posted here is therefore queued ahead of the loss, and the
    /// loss's reset — which relaunches a lost job's stages — finds no graph
    /// of a failed job to reset.
    async fn fail_bound_to(&self, executor_id: &str) {
        let Some(scheduler) = self.scheduler.upgrade() else {
            return;
        };
        let running = scheduler.state.task_manager.get_running_job_cache();
        for (job_id, info) in running.iter() {
            let saved = {
                let mut graph = info.execution_graph.write().await;
                let bound_here =
                    matches!(graph.status().status, Some(job_status::Status::Running(_)))
                        && graph
                            .running_tasks()
                            .iter()
                            .any(|task| task.executor_id == executor_id);
                if !bound_here {
                    continue;
                }
                tracing::warn!(executor_id, job_id = %job_id, "{EXECUTOR_LOST_LOG}");
                let lost = JammiError::ExecutorLost {
                    executor_id: executor_id.to_string(),
                    job_id: job_id.to_string(),
                };
                graph.fail_job(TaskErrorEnvelope::new(lost).to_string());
                self.job_state.save_job(job_id, &graph).await
            };
            if let Err(e) = saved {
                tracing::error!(
                    job_id = %job_id,
                    error = %e,
                    "the lost job's failure could not be saved; its submitter waits on the cancel"
                );
            }
            if let Err(e) = scheduler.cancel_job(job_id.clone()).await {
                tracing::error!(job_id = %job_id, error = %e, "the lost job could not be cancelled");
            }
        }
    }
}

/// The one mapping between Ballista's executor status and the catalog's:
/// `Active`/`Terminating`/`Dead` are the same three states on both sides.
/// A heartbeat that carries no status, or Ballista's `Unknown`, claims
/// nothing about the executor's lifecycle and is refused typed — a
/// `ballista-executor` process always reports one of the three, so a
/// status-less heartbeat is a foreign sender, not a state to record.
fn catalog_status(heartbeat: &ExecutorHeartbeat) -> BallistaResult<ComputeExecutorStatus> {
    match heartbeat.status.as_ref().and_then(|s| s.status.as_ref()) {
        Some(executor_status::Status::Active(_)) => Ok(ComputeExecutorStatus::Active),
        Some(executor_status::Status::Terminating(_)) => Ok(ComputeExecutorStatus::Terminating),
        Some(executor_status::Status::Dead(_)) => Ok(ComputeExecutorStatus::Dead),
        Some(executor_status::Status::Unknown(_)) | None => Err(BallistaError::Internal(format!(
            "jammi-ballista: heartbeat from executor {} carries no status",
            heartbeat.executor_id
        ))),
    }
}

/// Ballista's own status for a catalog state — the inverse of
/// [`catalog_status`], for seeding the heartbeat cache from the rows.
fn ballista_status(status: ComputeExecutorStatus) -> executor_status::Status {
    match status {
        ComputeExecutorStatus::Active => executor_status::Status::Active(String::default()),
        ComputeExecutorStatus::Terminating => {
            executor_status::Status::Terminating(String::default())
        }
        ComputeExecutorStatus::Dead => executor_status::Status::Dead(String::default()),
    }
}

/// Map a catalog/backend failure into the `BallistaError` every `ClusterState`/
/// `JobState` method must return — these traits are Ballista's, fixed by the
/// crate, so there is no ambient `From` the way `jammi-ballista::error::Error`
/// gets one.
fn ballista_err(e: jammi_db::error::JammiError) -> BallistaError {
    BallistaError::Internal(format!("jammi-ballista: catalog error: {e}"))
}

/// How long a registered executor's last heartbeat may lie in the past
/// before the catalog stops treating the row as a LIVE executor: DERIVED
/// from Ballista's own scheduler liveness notion (`SchedulerConfig::
/// default().executor_timeout_seconds`, the window after which
/// `ballista-scheduler` expires an executor it has stopped hearing from —
/// the default `roles::scheduler_config` keeps), never a second literal,
/// so "live to placement" and "live to Ballista" are one definition and
/// cannot drift apart.
pub fn executor_liveness_window() -> chrono::Duration {
    static WINDOW: std::sync::OnceLock<chrono::Duration> = std::sync::OnceLock::new();
    *WINDOW.get_or_init(|| {
        let secs = ballista_scheduler::config::SchedulerConfig::default().executor_timeout_seconds;
        chrono::Duration::seconds(
            i64::try_from(secs).expect("Ballista's default executor_timeout_seconds fits i64"),
        )
    })
}

/// The ONE liveness predicate every read that decides on executors shares
/// (`bind_schedulable_tasks`, `client::submit_physical_plan`'s device-kind
/// refusal): the row's `status` is `Active` (a `Terminating` row —
/// `roles::ExecutorRole::begin_drain` — or a `Dead` one is not) AND its
/// `heartbeat_at` lies within [`executor_liveness_window`] of `now`. A row
/// left behind by a process that never ran its graceful `remove_executor`
/// (SIGKILL, a crashed pod) therefore stops counting after the window, and
/// a test row written with a stale timestamp never counts at all. A
/// `heartbeat_at` this predicate cannot parse is not live (the row-fact
/// rule: the row is wrong, not the read).
pub fn executor_is_live(
    rec: &jammi_db::catalog::compute_repo::ComputeExecutorRecord,
    now: chrono::DateTime<chrono::Utc>,
) -> bool {
    if rec.status != ComputeExecutorStatus::Active {
        return false;
    }
    match chrono::DateTime::parse_from_rfc3339(&rec.heartbeat_at) {
        Ok(hb) => {
            now.signed_duration_since(hb.with_timezone(&chrono::Utc)) <= executor_liveness_window()
        }
        Err(_) => false,
    }
}

/// The live executor rows: every `compute_executors` row `catalog` holds
/// that [`executor_is_live`] admits now — the one read the submit client's
/// admission and the scheduler's binder share.
pub async fn live_executors(
    catalog: &Catalog,
) -> jammi_db::error::Result<Vec<jammi_db::catalog::compute_repo::ComputeExecutorRecord>> {
    let now = chrono::Utc::now();
    Ok(catalog
        .list_compute_executors()
        .await?
        .into_iter()
        .filter(|r| executor_is_live(r, now))
        .collect())
}

/// The catalog-backed [`ClusterState`]. Registrations, slots, and heartbeats
/// live in `compute_executors`; the ONE thing kept only in this
/// process's memory is the executor-heartbeat CACHE the trait's own
/// `executor_heartbeats`/`get_executor_heartbeat` require to be synchronous.
///
/// **Heartbeat cache staleness.** The cache is seeded from the catalog's
/// committed `heartbeat_at`/`status` at [`Self::init`] and refreshed
/// write-through by every [`Self::save_executor_heartbeat`] THIS process
/// handles. A second scheduler process sharing the same catalog never
/// receives the first scheduler's executors'
/// heartbeats directly — its cache reflects only what its OWN `init` read
/// plus whatever heartbeats land on IT — so `executor_heartbeats()` on a
/// standby scheduler can read stale relative to the catalog's own row. The
/// consequence for `expire_dead_executors` (which reads this cache, never
/// the catalog): a standby scheduler's liveness view of an executor it does
/// not itself serve is only as fresh as its last `init`, an inherent
/// property of the active/standby split, not a bug this cache should paper
/// over with an unbounded background poll.
pub struct CatalogClusterState {
    catalog: Arc<Catalog>,
    heartbeats: StdRwLock<HashMap<String, ExecutorHeartbeat>>,
    cluster_event_sender: ClusterEventSender<ClusterStateEvent>,
    placed_jobs: OnceLock<PlacedJobs>,
}

impl CatalogClusterState {
    pub fn new(catalog: Arc<Catalog>) -> Self {
        Self {
            catalog,
            heartbeats: StdRwLock::new(HashMap::new()),
            cluster_event_sender: ClusterEventSender::new(256),
            placed_jobs: OnceLock::new(),
        }
    }

    /// Install the scheduler's placed jobs, so a removed executor fails
    /// the jobs bound to it (`PlacedJobs::fail_bound_to`). Write-once:
    /// the state serves one scheduler, and a second install keeps the
    /// first.
    pub(crate) fn install_placed_jobs(&self, jobs: PlacedJobs) {
        if self.placed_jobs.set(jobs).is_err() {
            tracing::warn!("a scheduler's placed jobs are already installed on this cluster state");
        }
    }

    fn cache_heartbeat(&self, hb: ExecutorHeartbeat) {
        self.heartbeats
            .write()
            .expect("heartbeat cache lock poisoned")
            .insert(hb.executor_id.clone(), hb);
    }
}

fn record_to_executor_metadata(rec: &ComputeExecutorRecord) -> ExecutorMetadata {
    ExecutorMetadata {
        id: rec.executor_id.clone(),
        host: rec.host.clone(),
        port: rec.port,
        grpc_port: rec.grpc_port,
        specification: ballista_core::serde::scheduler::ExecutorSpecification {
            task_slots: rec.task_slots,
        },
        os_info: Default::default(),
    }
}

#[async_trait]
impl ClusterState for CatalogClusterState {
    async fn init(&self) -> BallistaResult<()> {
        let rows = self
            .catalog
            .list_compute_executors()
            .await
            .map_err(ballista_err)?;
        let mut cache = self
            .heartbeats
            .write()
            .expect("heartbeat cache lock poisoned");
        for rec in rows {
            let status = ballista_status(rec.status);
            cache.insert(
                rec.executor_id.clone(),
                ExecutorHeartbeat {
                    executor_id: rec.executor_id,
                    // Best-effort: the catalog's `heartbeat_at` is an opaque
                    // sortable TEXT, not a Unix-seconds integer — `init`
                    // seeds the cache with "now" rather than mis-decoding a
                    // shape it cannot invert, which only matters for
                    // `expire_dead_executors`' very first sweep after a
                    // restart (see this type's own doc on staleness).
                    timestamp: unix_seconds_now(),
                    metrics: vec![],
                    status: Some(ExecutorStatus {
                        status: Some(status),
                    }),
                    peak_proc_physical_memory: 0,
                    peak_proc_virtual_memory: 0,
                },
            );
        }
        Ok(())
    }

    async fn bind_schedulable_tasks(
        &self,
        distribution: TaskDistributionPolicy,
        active_jobs: Arc<HashMap<JobId, JobInfoCache>>,
        executors: Option<HashSet<String>>,
    ) -> BallistaResult<Vec<BoundTask>> {
        let rows = live_executors(&self.catalog).await.map_err(ballista_err)?;
        let mut slots: Vec<ballista_core::serde::protobuf::AvailableTaskSlots> = rows
            .into_iter()
            .filter(|r| {
                r.available_slots > 0
                    && executors
                        .as_ref()
                        .map(|e| e.contains(&r.executor_id))
                        .unwrap_or(true)
            })
            .map(|r| ballista_core::serde::protobuf::AvailableTaskSlots {
                executor_id: r.executor_id,
                slots: r.available_slots,
            })
            .collect();
        let slot_refs: Vec<&mut ballista_core::serde::protobuf::AvailableTaskSlots> =
            slots.iter_mut().collect();

        let bound = match distribution {
            // Neither built-in is ever selected by `jammi-server` (there is
            // no knob: `roles::host_scheduler` always installs
            // `Custom(DevicePlacement)`) — these two arms
            // exist only so `ClusterState`'s generic contract is complete
            // and testable; unlike `Custom`, they never reserve a catalog
            // slot (no CAS), so they are not safe under a shared catalog and
            // must never be reachable from configuration (`BallistaConfig`
            // carries no distribution knob).
            TaskDistributionPolicy::Bias | TaskDistributionPolicy::RoundRobin => {
                bind_round_robin(slot_refs, active_jobs).await
            }
            TaskDistributionPolicy::Custom(ref policy) => {
                policy.bind_tasks(slot_refs, active_jobs).await?
            }
        };
        Ok(bound)
    }

    async fn unbind_tasks(&self, executor_slots: Vec<ExecutorSlot>) -> BallistaResult<()> {
        let mut increments: HashMap<String, i64> = HashMap::new();
        for (executor_id, n) in executor_slots {
            *increments.entry(executor_id).or_insert(0) += i64::from(n);
        }
        // Ballista's scheduler removes an executor whose task launch failed
        // (`remove_executor` deletes its row) and THEN unbinds the slots it
        // had reserved on it. Slots on a row that no longer exists have
        // nothing to return to: they are dropped here, not refused, so the
        // rest of the batch (a live executor's slots) is still returned.
        // Refusing would fail the whole unbind with "adjust_compute_slots:
        // no row for executor_id" on exactly that ordering.
        let registered: std::collections::HashSet<String> = self
            .catalog
            .list_compute_executors()
            .await
            .map_err(ballista_err)?
            .into_iter()
            .map(|r| r.executor_id)
            .collect();
        let deltas: Vec<(&str, i64)> = increments
            .iter()
            .filter(|(id, _)| {
                let live = registered.contains(id.as_str());
                if !live {
                    tracing::debug!(
                        executor_id = %id,
                        "unbind_tasks: executor row already removed; its slots are dropped"
                    );
                }
                live
            })
            .map(|(id, n)| (id.as_str(), *n))
            .collect();
        if deltas.is_empty() {
            return Ok(());
        }
        self.catalog
            .adjust_compute_slots(&deltas)
            .await
            .map_err(ballista_err)
    }

    async fn register_executor(
        &self,
        metadata: ExecutorMetadata,
        spec: ExecutorData,
    ) -> BallistaResult<()> {
        // Devices: `ClusterState::register_executor`'s signature (fixed by
        // Ballista) carries no device field, and this method may be
        // registering a REMOTE executor process this scheduler has no other
        // channel to — so a device claim can only come from a row THAT
        // EXECUTOR'S OWN process already wrote. `roles::host_executor` calls
        // `Catalog::upsert_compute_executor` directly, over the SAME shared
        // catalog, right after its own registration completes (see that
        // function's doc) — deterministically the LAST write of its own
        // startup sequence, so it always wins regardless of whether this
        // registration hook runs before or after it. Preserve whatever is
        // already there (empty if none) rather than wiping it here.
        let existing_devices = self
            .catalog
            .get_compute_executor(&metadata.id)
            .await
            .map_err(ballista_err)?
            .map(|r| r.devices)
            .unwrap_or_default();
        let now = jammi_db::catalog::lease::canonical_stamp_now();
        let rec = ComputeExecutorRecord {
            executor_id: metadata.id.clone(),
            instance_id: metadata.id.clone(),
            host: metadata.host.clone(),
            port: metadata.port,
            grpc_port: metadata.grpc_port,
            task_slots: spec.total_task_slots,
            available_slots: spec.available_task_slots,
            status: ComputeExecutorStatus::Active,
            heartbeat_at: now,
            metadata: String::new(),
            devices: existing_devices,
        };
        self.catalog
            .upsert_compute_executor(&rec)
            .await
            .map_err(ballista_err)?;
        self.cache_heartbeat(ExecutorHeartbeat {
            executor_id: metadata.id.clone(),
            timestamp: unix_seconds_now(),
            metrics: vec![],
            status: Some(ExecutorStatus {
                status: Some(ballista_status(ComputeExecutorStatus::Active)),
            }),
            peak_proc_physical_memory: 0,
            peak_proc_virtual_memory: 0,
        });
        self.cluster_event_sender
            .send(&ClusterStateEvent::RegisteredExecutor {
                executor_id: metadata.id,
            });
        Ok(())
    }

    async fn save_executor_metadata(&self, metadata: ExecutorMetadata) -> BallistaResult<()> {
        // Metadata-only refresh (host/ports/slots): preserve `available_slots`
        // and `devices` from the existing row — this is never the verb that
        // adjusts capacity or a device claim.
        let existing = self
            .catalog
            .get_compute_executor(&metadata.id)
            .await
            .map_err(ballista_err)?;
        let (available_slots, devices, status, heartbeat_at) = match existing {
            Some(r) => (r.available_slots, r.devices, r.status, r.heartbeat_at),
            None => (
                metadata.specification.task_slots,
                Vec::new(),
                ComputeExecutorStatus::Active,
                jammi_db::catalog::lease::canonical_stamp_now(),
            ),
        };
        let rec = ComputeExecutorRecord {
            executor_id: metadata.id.clone(),
            instance_id: metadata.id.clone(),
            host: metadata.host,
            port: metadata.port,
            grpc_port: metadata.grpc_port,
            task_slots: metadata.specification.task_slots,
            available_slots,
            status,
            heartbeat_at,
            metadata: String::new(),
            devices,
        };
        self.catalog
            .upsert_compute_executor(&rec)
            .await
            .map_err(ballista_err)?;
        self.cluster_event_sender
            .send(&ClusterStateEvent::RegisteredExecutor {
                executor_id: metadata.id,
            });
        Ok(())
    }

    async fn get_executor_metadata(&self, executor_id: &str) -> BallistaResult<ExecutorMetadata> {
        self.catalog
            .get_compute_executor(executor_id)
            .await
            .map_err(ballista_err)?
            .map(|r| record_to_executor_metadata(&r))
            .ok_or_else(|| BallistaError::Internal(format!("no executor with id {executor_id}")))
    }

    async fn registered_executor_metadata(&self) -> Vec<ExecutorMetadata> {
        self.catalog
            .list_compute_executors()
            .await
            .unwrap_or_default()
            .iter()
            .map(record_to_executor_metadata)
            .collect()
    }

    /// The catalog write is monotone in the executor's lifecycle
    /// (`Catalog::record_compute_heartbeat`): an `Active` report that lands
    /// after the row read `Terminating` — the executor's periodic heartbeat
    /// built before its drain began — refreshes `heartbeat_at` and leaves
    /// the row draining, so `executor_is_live` never reads a draining
    /// executor as bindable again.
    async fn save_executor_heartbeat(&self, heartbeat: ExecutorHeartbeat) -> BallistaResult<()> {
        let status = catalog_status(&heartbeat)?;
        let updated = self
            .catalog
            .record_compute_heartbeat(
                &heartbeat.executor_id,
                status,
                &jammi_db::catalog::lease::canonical_stamp_now(),
            )
            .await
            .map_err(ballista_err)?;
        if !updated {
            // A heartbeat for an executor whose row has not been registered
            // yet (a race between the registration RPC and its own
            // heartbeat loop): this is a ROW FACT, not a fault — the cache
            // still gets the report so `executor_heartbeats()` is not blind
            // to it.
            tracing::warn!(
                executor_id = %heartbeat.executor_id,
                "heartbeat for an unregistered compute executor"
            );
        }
        self.cache_heartbeat(heartbeat);
        Ok(())
    }

    /// The executor's row leaves the catalog first — no reader admits it
    /// from here on — then the jobs bound to it fail typed
    /// (`PlacedJobs::fail_bound_to`, whose ordering against Ballista's
    /// own `ExecutorLost` this method's caller guarantees), then the
    /// heartbeat cache and the event follow.
    async fn remove_executor(&self, executor_id: &str) -> BallistaResult<()> {
        self.catalog
            .remove_compute_executor(executor_id)
            .await
            .map_err(ballista_err)?;
        if let Some(jobs) = self.placed_jobs.get() {
            jobs.fail_bound_to(executor_id).await;
        }
        self.heartbeats
            .write()
            .expect("heartbeat cache lock poisoned")
            .remove(executor_id);
        self.cluster_event_sender
            .send(&ClusterStateEvent::RemovedExecutor {
                executor_id: executor_id.to_string(),
            });
        Ok(())
    }

    fn executor_heartbeats(&self) -> HashMap<String, ExecutorHeartbeat> {
        self.heartbeats
            .read()
            .expect("heartbeat cache lock poisoned")
            .clone()
    }

    fn get_executor_heartbeat(&self, executor_id: &str) -> Option<ExecutorHeartbeat> {
        self.heartbeats
            .read()
            .expect("heartbeat cache lock poisoned")
            .get(executor_id)
            .cloned()
    }

    async fn cluster_state_events(&self) -> BallistaResult<ClusterStateEventStream> {
        Ok(Box::pin(self.cluster_event_sender.subscribe()))
    }
}

fn unix_seconds_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The catalog-backed [`JobState`]: ownership and status text live in
/// `compute_jobs`; the execution graph — which Ballista 54.1 cannot
/// serialize — lives ONLY in this process's own memory, exactly
/// like [`ballista_scheduler::cluster::memory::InMemoryJobState`]'s own
/// `completed_jobs`/`running_jobs` maps.
///
/// `accept_job` (the pre-planning "queued" bookkeeping) is kept PURELY
/// in-memory, never touching the catalog — the same shape
/// `InMemoryJobState::accept_job` itself uses (it is a `DashMap` insert with
/// no persistence at all); a job that fails before `submit_job` is recorded
/// by `fail_unscheduled_job` (async, so it CAN reach the catalog). The
/// trait's `accept_job` is a SYNC method, and `InMemoryJobState`'s own
/// `accept_job` (`ballista-scheduler-54.1.0/src/cluster/memory.rs:483-488`)
/// never persists either; this type mirrors that behavior.
pub struct CatalogJobState {
    catalog: Arc<Catalog>,
    scheduler: String,
    graphs: AsyncRwLock<HashMap<JobId, ExecutionGraphBox>>,
    queued: StdMutex<HashMap<JobId, (String, u64)>>,
    subscribers: StdMutex<HashMap<JobId, JobStatusSubscriber>>,
    session_builder: SessionBuilder,
    config_producer: ConfigProducer,
    job_event_sender: ClusterEventSender<JobStateEvent>,
}

impl CatalogJobState {
    pub fn new(
        catalog: Arc<Catalog>,
        scheduler: impl Into<String>,
        session_builder: SessionBuilder,
        config_producer: ConfigProducer,
    ) -> Self {
        Self {
            catalog,
            scheduler: scheduler.into(),
            graphs: AsyncRwLock::new(HashMap::new()),
            queued: StdMutex::new(HashMap::new()),
            subscribers: StdMutex::new(HashMap::new()),
            session_builder,
            config_producer,
            job_event_sender: ClusterEventSender::new(256),
        }
    }

    fn status_text(status: &job_status::Status) -> &'static str {
        match status {
            job_status::Status::Queued(_) => "queued",
            job_status::Status::Running(_) => "running",
            job_status::Status::Successful(_) => "successful",
            job_status::Status::Failed(_) => "failed",
        }
    }

    fn notify(&self, job_id: &JobId, status: JobStatus) {
        let subscriber = self
            .subscribers
            .lock()
            .expect("subscribers lock poisoned")
            .get(job_id)
            .cloned();
        if let Some(sub) = subscriber {
            if matches!(sub.try_send(status), Err(TrySendError::Full(_))) {
                tracing::error!(
                    job_id = %job_id,
                    "job status subscriber is blocked; the update is dropped, never retried"
                );
            }
        }
    }
}

/// Reconstruct a minimal [`JobStatus`] from a catalog row when this
/// process's memory has no graph for the job (a restarted scheduler
/// answers `get_job_status` from the row even though
/// `get_execution_graph` is `None`). The inner message carries only what the
/// row itself has — `queued_at` decodes the decimal string
/// [`CatalogJobState::submit_job`] stamped, `started_at`/partition/error
/// detail are not reconstructable from a status-only row and are left at
/// their zero/empty default (this limitation is why
/// `ballista-scheduler`'s REST introspection surface stays off).
fn record_to_job_status(rec: &ComputeJobRecord) -> JobStatus {
    let queued_at: u64 = rec.queued_at.parse().unwrap_or(0);
    let status = match rec.status.as_str() {
        "queued" => job_status::Status::Queued(QueuedJob { queued_at }),
        "successful" => job_status::Status::Successful(SuccessfulJob {
            partition_location: vec![],
            queued_at,
            started_at: 0,
            ended_at: 0,
        }),
        "failed" => job_status::Status::Failed(FailedJob {
            error: String::new(),
            queued_at,
            started_at: 0,
            ended_at: 0,
        }),
        _ => job_status::Status::Running(RunningJob {
            queued_at,
            started_at: 0,
            scheduler: rec.owner.clone(),
        }),
    };
    JobStatus {
        job_id: rec.job_id.clone(),
        job_name: String::new(),
        status: Some(status),
    }
}

#[async_trait]
impl JobState for CatalogJobState {
    fn accept_job(&self, job_id: &JobId, job_name: &str, queued_at: u64) -> BallistaResult<()> {
        self.queued
            .lock()
            .expect("queued-jobs lock poisoned")
            .insert(job_id.clone(), (job_name.to_string(), queued_at));
        Ok(())
    }

    fn pending_job_number(&self) -> usize {
        self.queued.lock().expect("queued-jobs lock poisoned").len()
    }

    async fn submit_job(
        &self,
        job_id: JobId,
        graph: &ExecutionGraphBox,
        subscriber: Option<JobStatusSubscriber>,
    ) -> BallistaResult<()> {
        let queued_at = self
            .queued
            .lock()
            .expect("queued-jobs lock poisoned")
            .remove(&job_id)
            .ok_or_else(|| {
                BallistaError::Internal(format!(
                    "failed to submit job {job_id}: not found in queued jobs"
                ))
            })?
            .1;
        if let Some(sub) = subscriber {
            self.subscribers
                .lock()
                .expect("subscribers lock poisoned")
                .insert(job_id.clone(), sub);
        }
        let rec = ComputeJobRecord {
            job_id: job_id.to_string(),
            owner: self.scheduler.clone(),
            status: "running".to_string(),
            queued_at: queued_at.to_string(),
            updated_at: jammi_db::catalog::lease::canonical_stamp_now(),
        };
        self.catalog
            .put_compute_job(&rec)
            .await
            .map_err(ballista_err)?;
        self.graphs
            .write()
            .await
            .insert(job_id.clone(), graph.cloned());
        self.job_event_sender.send(&JobStateEvent::JobAcquired {
            job_id,
            owner: self.scheduler.clone(),
        });
        Ok(())
    }

    async fn get_jobs(&self) -> BallistaResult<HashSet<JobId>> {
        Ok(self
            .catalog
            .list_compute_jobs()
            .await
            .map_err(ballista_err)?
            .into_iter()
            .filter(|r| r.owner == self.scheduler)
            .map(|r| JobId::from(r.job_id))
            .collect())
    }

    async fn get_all_jobs(&self) -> BallistaResult<HashSet<JobId>> {
        Ok(self
            .catalog
            .list_compute_jobs()
            .await
            .map_err(ballista_err)?
            .into_iter()
            .map(|r| JobId::from(r.job_id))
            .collect())
    }

    async fn get_job_status(&self, job_id: &JobId) -> BallistaResult<Option<JobStatus>> {
        if let Some((job_name, queued_at)) = self
            .queued
            .lock()
            .expect("queued-jobs lock poisoned")
            .get(job_id)
        {
            return Ok(Some(JobStatus {
                job_id: job_id.to_string(),
                job_name: job_name.clone(),
                status: Some(job_status::Status::Queued(QueuedJob {
                    queued_at: *queued_at,
                })),
            }));
        }
        let job_id_s = job_id.to_string();
        let row = jammi_db::tenant_scope::TenantBinding::admin_scope(async {
            self.catalog.get_compute_job(&job_id_s).await
        })
        .await
        .map_err(ballista_err)?;
        Ok(row.as_ref().map(record_to_job_status))
    }

    async fn get_execution_graph(
        &self,
        job_id: &JobId,
    ) -> BallistaResult<Option<ExecutionGraphBox>> {
        Ok(self.graphs.read().await.get(job_id).map(|g| g.cloned()))
    }

    async fn save_job(&self, job_id: &JobId, graph: &ExecutionGraphBox) -> BallistaResult<()> {
        let status = graph.status().clone();
        let Some(inner) = status.status.as_ref() else {
            return Err(BallistaError::Internal(format!(
                "save_job: job {job_id} has no status"
            )));
        };
        let rec = ComputeJobRecord {
            job_id: job_id.to_string(),
            owner: self.scheduler.clone(),
            status: Self::status_text(inner).to_string(),
            queued_at: self
                .catalog
                .get_compute_job(job_id.as_ref())
                .await
                .map_err(ballista_err)?
                .map(|r| r.queued_at)
                .unwrap_or_default(),
            updated_at: jammi_db::catalog::lease::canonical_stamp_now(),
        };
        self.catalog
            .put_compute_job(&rec)
            .await
            .map_err(ballista_err)?;
        self.graphs
            .write()
            .await
            .insert(job_id.clone(), graph.cloned());
        self.notify(job_id, status.clone());
        self.job_event_sender.send(&JobStateEvent::JobUpdated {
            job_id: job_id.clone(),
            status,
        });
        Ok(())
    }

    async fn fail_unscheduled_job(&self, job_id: &JobId, reason: String) -> BallistaResult<()> {
        let queued_at = self
            .queued
            .lock()
            .expect("queued-jobs lock poisoned")
            .remove(job_id)
            .ok_or_else(|| {
                BallistaError::Internal(format!(
                    "could not fail unscheduled job {job_id}: not found in queued jobs"
                ))
            })?
            .1;
        let rec = ComputeJobRecord {
            job_id: job_id.to_string(),
            owner: self.scheduler.clone(),
            status: "failed".to_string(),
            queued_at: queued_at.to_string(),
            updated_at: jammi_db::catalog::lease::canonical_stamp_now(),
        };
        self.catalog
            .put_compute_job(&rec)
            .await
            .map_err(ballista_err)?;
        let _ = reason;
        Ok(())
    }

    async fn remove_job(&self, job_id: &JobId) -> BallistaResult<()> {
        self.catalog
            .delete_compute_job(job_id.as_ref())
            .await
            .map_err(ballista_err)?;
        self.graphs.write().await.remove(job_id);
        self.subscribers
            .lock()
            .expect("subscribers lock poisoned")
            .remove(job_id);
        Ok(())
    }

    async fn try_acquire_job(&self, job_id: &JobId) -> BallistaResult<Option<ExecutionGraphBox>> {
        // Active/standby: "acquire" only ever succeeds against
        // a graph THIS process's own memory already holds — there is no
        // serialized graph to revive from another scheduler's ownership, so
        // acquiring a job this process never itself submitted always misses,
        // even when the catalog row's `owner` names this scheduler (a stale
        // ownership record left by a process that has since restarted).
        let graph = self.graphs.read().await.get(job_id).map(|g| g.cloned());
        let Some(graph) = graph else {
            return Ok(None);
        };
        let job_id_s = job_id.to_string();
        let owner = jammi_db::tenant_scope::TenantBinding::admin_scope(async {
            self.catalog.get_compute_job(&job_id_s).await
        })
        .await
        .map_err(ballista_err)?
        .map(|r| r.owner);
        if owner.as_deref() == Some(self.scheduler.as_str()) {
            Ok(Some(graph))
        } else {
            Ok(None)
        }
    }

    async fn job_state_events(&self) -> BallistaResult<JobStateEventStream> {
        Ok(Box::pin(self.job_event_sender.subscribe()))
    }

    async fn create_or_update_session(
        &self,
        session_id: &str,
        config: &SessionConfig,
    ) -> BallistaResult<Arc<SessionContext>> {
        self.job_event_sender.send(&JobStateEvent::SessionAccessed {
            session_id: session_id.to_string(),
        });
        Ok(create_datafusion_context(
            config,
            self.session_builder.clone(),
        )?)
    }

    async fn remove_session(&self, session_id: &str) -> BallistaResult<()> {
        self.job_event_sender.send(&JobStateEvent::SessionRemoved {
            session_id: session_id.to_string(),
        });
        Ok(())
    }

    fn produce_config(&self) -> SessionConfig {
        (self.config_producer)()
    }
}
