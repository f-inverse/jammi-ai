//! The fleets a leg runs on: spawned on this host from a `jammi-server`
//! binary through `jammi_test_utils::fleet` — the same facility the
//! distributed lane spawns its fleets through — or joined where it already
//! runs across hosts. Every fleet shares one Postgres catalog and one
//! S3-class store, which this process reads as an observer (a session with
//! no worker) to find who claimed a job, what it published, and where.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use jammi_ai::session::InferenceSession;
use jammi_ballista::placement::BOUND_TASK_LOG;
use jammi_db::catalog::jobs_repo::{JobRecord, WorkerRecord};
use jammi_db::catalog::status::JobStatus;
use jammi_db::config::{
    CatalogConfig, DistributedConfig, GpuConfig, JammiConfig, LeaseConfig, StorageConfig,
    WorkerConfig,
};
use jammi_db::store::SINK_WRITE_LOG;
use jammi_test_utils::fleet::{
    BallistaRole, Fleet, ProcSpec, ShapeDRole, WorkerRole, MAX_WORLD_SIZE,
};
use jammi_test_utils::DistributedBackends;

use super::PlaneParams;
use crate::leg::RanOn;

/// A fleet member's role, as the leg names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemberRole {
    /// `placed`: the claimant that hosts the scheduler and the client role,
    /// places every attempt and hosts no executor.
    Submitter,
    /// `placed`: the executor a placed attempt runs on.
    Executor,
    Scheduler,
    Query,
    Compute,
}

impl MemberRole {
    pub fn as_str(self) -> &'static str {
        match self {
            MemberRole::Submitter => "submitter",
            MemberRole::Executor => "executor",
            MemberRole::Scheduler => "scheduler",
            MemberRole::Query => "query",
            MemberRole::Compute => "compute",
        }
    }
}

/// One fleet member by label and role.
#[derive(Debug, Clone)]
pub struct Member {
    pub label: String,
    pub role: MemberRole,
}

/// A running fleet and the observer that reads its catalog and store.
pub struct RunningFleet {
    /// The processes this leg spawned; none when the fleet was joined.
    fleet: Option<Fleet>,
    members: Vec<Member>,
    pub session: Arc<InferenceSession>,
    /// The gRPC address of the query tier a shape-d job is submitted
    /// through.
    pub query_addr: Option<String>,
}

/// The generous terminal-state timeout — cold boot + Postgres connect +
/// migrate + a tiny CPU LoRA fine-tune + publish + finalize.
pub const TERMINAL_TIMEOUT: Duration = Duration::from_secs(300);
const POLL_INTERVAL: Duration = Duration::from_millis(250);

/// The line the submitter writes when a claimed attempt is handed off to
/// the executor it was placed on.
pub const HANDED_OFF_LOG: &str = "run_placed_attempt: submitter HandedOff";
/// The line a sink writes when it places its materialization on the compute
/// plane, and the one it writes when the plane cannot hold it.
pub const SINK_PLACED_LOG: &str = "materialization placed on the compute plane";
pub const SINK_LOCAL_LOG: &str = "materialization runs in this process";

impl RunningFleet {
    /// The backends every fleet shares, from the environment
    /// (`JAMMI_TEST_PG_URL`, `JAMMI_TEST_S3_ENDPOINT`, `JAMMI_TEST_S3_BUCKET`,
    /// `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`).
    fn backends() -> DistributedBackends {
        DistributedBackends::from_env()
    }

    /// The `placed` fleet: a submitter that claims jobs of `kinds` and
    /// places them, and one executor that holds what is placed and claims
    /// nothing of its own. Both on `device` (the executor's CUDA ordinal,
    /// or `-1`): an attempt is placed only on an executor of its claimant's
    /// device kind.
    pub async fn spawn_placed(
        plane: &PlaneParams,
        leg: &str,
        device: i32,
        kinds: &'static [&'static str],
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let backends = Self::backends();
        let result_root = backends.unique_result_root(leg);
        let scheduler_port = jammi_test_utils::free_port();
        let specs = vec![
            ProcSpec::fresh_on(
                BallistaRole::SchedulerAndClient { scheduler_port },
                WorkerRole {
                    enabled: true,
                    kinds: Some(kinds),
                    idle_poll_secs: 1,
                },
                device,
            ),
            ProcSpec::fresh_on(
                BallistaRole::Executor { scheduler_port },
                WorkerRole {
                    enabled: true,
                    kinds: Some(&[]),
                    idle_poll_secs: 1,
                },
                device,
            ),
        ];
        let roles = [MemberRole::Submitter, MemberRole::Executor];
        Self::spawn(plane, backends, result_root, specs, &roles, None).await
    }

    /// The `shape-d` fleet on this host: the deployed topology's three role
    /// configs, one compute process on `device`.
    pub async fn spawn_shape_d(
        plane: &PlaneParams,
        leg: &str,
        device: i32,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let backends = Self::backends();
        let result_root = backends.unique_result_root(leg);
        let scheduler_port = jammi_test_utils::free_port();
        let specs = vec![
            ProcSpec::shape_d(ShapeDRole::Scheduler, scheduler_port, -1),
            ProcSpec::shape_d(ShapeDRole::Query, scheduler_port, -1),
            ProcSpec::shape_d(ShapeDRole::Compute, scheduler_port, device),
        ];
        let query_addr = format!("127.0.0.1:{}", specs[1].flight_port());
        let roles = [
            MemberRole::Scheduler,
            MemberRole::Query,
            MemberRole::Compute,
        ];
        Self::spawn(
            plane,
            backends,
            result_root,
            specs,
            &roles,
            Some(query_addr),
        )
        .await
    }

    /// A shape-d fleet already running (`--query-addr`): nothing is
    /// spawned; the job is submitted through the query tier and observed
    /// through the shared backends. Its members are the workers registered
    /// in the catalog, so a compute process is one whose worker claims
    /// training kinds.
    pub async fn join_shape_d(
        query_addr: &str,
        leg: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let backends = Self::backends();
        let result_root = backends.unique_result_root(leg);
        let session = observer_session(&backends, &result_root).await?;
        let members = session
            .catalog()
            .list_workers()
            .await?
            .into_iter()
            .map(|w| Member {
                label: w.label.clone().unwrap_or_else(|| w.instance_id.clone()),
                role: if w.kinds.contains("fine_tune") {
                    MemberRole::Compute
                } else {
                    MemberRole::Query
                },
            })
            .collect();
        Ok(Self {
            fleet: None,
            members,
            session,
            query_addr: Some(query_addr.to_string()),
        })
    }

    async fn spawn(
        plane: &PlaneParams,
        backends: DistributedBackends,
        result_root: String,
        specs: Vec<ProcSpec>,
        roles: &[MemberRole],
        query_addr: Option<String>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let exe = plane.server_bin.clone().ok_or_else(|| {
            format!(
                "this rung spawns a fleet on this host: pass --server-bin <jammi-server> ({} \
                 were given)",
                plane.summary()
            )
        })?;
        if !exe.is_file() {
            return Err(format!("--server-bin {} is not a file", exe.display()).into());
        }
        let repo_root = plane
            .repo_root
            .clone()
            .unwrap_or_else(jammi_test_utils::workspace_root);
        let fleet = Fleet::spawn(&exe, &repo_root, &backends, &result_root, specs);
        let members = roles
            .iter()
            .enumerate()
            .map(|(index, &role)| Member {
                label: fleet.label(index).to_string(),
                role,
            })
            .collect();
        let session = observer_session(&backends, &result_root).await?;
        let mut running = Self {
            fleet: Some(fleet),
            members,
            session,
            query_addr,
        };
        running.await_workers_registered().await?;
        Ok(running)
    }

    pub fn member(&self, role: MemberRole) -> Option<&Member> {
        self.members.iter().find(|m| m.role == role)
    }

    /// The worker row of `label`, once it exists.
    pub async fn worker_of_label(
        &self,
        label: &str,
    ) -> Result<WorkerRecord, Box<dyn std::error::Error + Send + Sync>> {
        self.session
            .catalog()
            .list_workers()
            .await?
            .into_iter()
            .find(|w| w.label.as_deref() == Some(label))
            .ok_or_else(|| format!("no worker row for label {label:?}").into())
    }

    /// Every spawned member that runs a worker has its `workers` row —
    /// the row its label is tied to an instance id through.
    async fn await_workers_registered(
        &mut self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let Some(fleet) = &self.fleet else {
            return Ok(());
        };
        let labels: Vec<String> = (0..self.members.len())
            .filter(|&i| fleet.spec(i).worker_enabled())
            .map(|i| fleet.label(i).to_string())
            .collect();
        let deadline = Instant::now() + Duration::from_secs(90);
        loop {
            let workers = self.session.catalog().list_workers().await?;
            if labels
                .iter()
                .all(|l| workers.iter().any(|w| w.label.as_deref() == Some(l)))
            {
                return Ok(());
            }
            self.check_alive("waiting for the fleet's workers rows")?;
            if Instant::now() >= deadline {
                self.diagnostics("the fleet's workers rows never appeared");
                return Err(format!("timed out waiting for workers rows for {labels:?}").into());
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }

    /// Poll the catalog for `job_id` until `want` holds — failing the
    /// moment a member exits, the row settles on a terminal status `want`
    /// rejects, or the timeout passes.
    pub async fn await_job(
        &mut self,
        job_id: &str,
        what: &str,
        mut want: impl FnMut(&JobRecord) -> bool,
    ) -> Result<JobRecord, Box<dyn std::error::Error + Send + Sync>> {
        let catalog = self.session.catalog().pinned_to_tenant(None);
        let deadline = Instant::now() + TERMINAL_TIMEOUT;
        loop {
            if let Ok(record) = catalog.get_job(job_id).await {
                if want(&record) {
                    return Ok(record);
                }
                if record.is_terminal() {
                    self.diagnostics(&format!(
                        "job {job_id} settled on status={:?} error={:?}, not: {what}",
                        record.status, record.error
                    ));
                    return Err(format!(
                        "job {job_id} reached a terminal status ({:?}) that is not: {what} \
                         (error: {:?})",
                        record.status, record.error
                    )
                    .into());
                }
            }
            self.check_alive(what)?;
            if Instant::now() >= deadline {
                self.diagnostics(&format!("timed out after {TERMINAL_TIMEOUT:?}: {what}"));
                return Err(format!("timed out after {TERMINAL_TIMEOUT:?}: {what}").into());
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }

    /// The first line of `label`'s log containing `needle`, polling until
    /// it lands (a line a member writes after the catalog fact already
    /// awaited is never a single read).
    pub async fn await_log_line(
        &mut self,
        label: &str,
        needle: &str,
        what: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let deadline = Instant::now() + TERMINAL_TIMEOUT;
        loop {
            if let Some(line) = self.log_line(label, needle) {
                return Ok(line);
            }
            self.check_alive(what)?;
            if Instant::now() >= deadline {
                self.diagnostics(&format!("timed out awaiting the log line: {what}"));
                return Err(format!("{label}'s log never showed: {what}").into());
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }

    /// The first line of `label`'s log containing `needle`, if any yet.
    /// `None` for a joined fleet, whose logs are not this process's to read.
    pub fn log_line(&self, label: &str, needle: &str) -> Option<String> {
        let fleet = self.fleet.as_ref()?;
        fleet
            .log_contents(label)
            .lines()
            .find(|line| line.contains(needle))
            .map(str::to_string)
    }

    /// Whether the logs are this process's to read.
    pub fn has_logs(&self) -> bool {
        self.fleet.is_some()
    }

    fn check_alive(&mut self, what: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let Some(fleet) = &mut self.fleet else {
            return Ok(());
        };
        if let Some((worker, status)) = fleet.first_unexpected_exit() {
            fleet.dump_diagnostics(&format!("worker {worker} exited ({status}) during: {what}"));
            return Err(format!("fleet member {worker} exited ({status}) during: {what}").into());
        }
        Ok(())
    }

    pub fn diagnostics(&self, context: &str) {
        if let Some(fleet) = &self.fleet {
            fleet.dump_diagnostics(context);
        }
    }

    /// Where `claimed_by` ran, as one of this fleet's members — refused
    /// when it is not one, or is a member of a role that must never train
    /// (`must_not`).
    pub async fn ran_on(
        &self,
        claimed_by: &str,
        must_not: &[MemberRole],
    ) -> Result<RanOn, Box<dyn std::error::Error + Send + Sync>> {
        let workers = self.session.catalog().list_workers().await?;
        let worker = workers
            .iter()
            .find(|w| w.instance_id == claimed_by)
            .ok_or_else(|| {
                format!("claimant {claimed_by} has no workers row: not a member of this fleet")
            })?;
        let member = self
            .members
            .iter()
            .find(|m| worker.label.as_deref() == Some(m.label.as_str()))
            .ok_or_else(|| {
                format!(
                    "claimant {claimed_by} (label {:?}) is not a member of this fleet: {:?}",
                    worker.label,
                    self.members.iter().map(|m| &m.label).collect::<Vec<_>>()
                )
            })?;
        if must_not.contains(&member.role) {
            return Err(format!(
                "the job trained on the {} process ({}), which this rung requires it never \
                 does: the leg would claim a placement that did not happen",
                member.role.as_str(),
                member.label
            )
            .into());
        }
        Ok(RanOn {
            instance_id: claimed_by.to_string(),
            label: worker.label.clone(),
            host: worker.host.clone(),
            role: member.role.as_str().to_string(),
            evidence: Vec::new(),
        })
    }

    /// The label of the member in `role`.
    fn label_of(
        &self,
        role: MemberRole,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        self.member(role)
            .map(|m| m.label.clone())
            .ok_or_else(|| format!("the fleet has no {} process", role.as_str()).into())
    }

    /// The completed job `job_id`, or the reason it is not.
    pub async fn await_completed(
        &mut self,
        job_id: &str,
        what: &str,
    ) -> Result<JobRecord, Box<dyn std::error::Error + Send + Sync>> {
        self.await_job(job_id, what, |r| {
            r.status == JobStatus::Completed.to_string()
        })
        .await
    }

    /// Where the completed job `record` ran, as the catalog says: the member
    /// holding its claim, never one of `must_not`, with the row itself as
    /// the first evidence.
    async fn claim_ran_on(
        &self,
        record: &JobRecord,
        must_not: &[MemberRole],
    ) -> Result<RanOn, Box<dyn std::error::Error + Send + Sync>> {
        let claimed_by = record
            .claimed_by
            .as_deref()
            .ok_or("the completed job names no claimant")?;
        let mut ran_on = self.ran_on(claimed_by, must_not).await?;
        ran_on.evidence.push(format!(
            "jobs.claimed_by={claimed_by} attempts={} releases={}",
            record.attempts, record.releases
        ));
        Ok(ran_on)
    }

    /// Where a training job placed by the submitter ran: the job completed,
    /// its claim transferred to the executor (never the submitter), proven
    /// by the submitter's hand-off line and the scheduler's binding.
    pub async fn placed_training_ran_on(
        &mut self,
        job_id: &str,
    ) -> Result<(JobRecord, RanOn), Box<dyn std::error::Error + Send + Sync>> {
        let record = self
            .await_completed(job_id, "the placed job completes")
            .await?;
        let mut ran_on = self.claim_ran_on(&record, &[MemberRole::Submitter]).await?;
        let submitter = self.label_of(MemberRole::Submitter)?;
        ran_on.evidence.push(
            self.await_log_line(&submitter, HANDED_OFF_LOG, "the submitter's hand-off line")
                .await?,
        );
        ran_on.evidence.push(
            self.await_log_line(&submitter, BOUND_TASK_LOG, "the scheduler's task binding")
                .await?,
        );
        Ok((record, ran_on))
    }

    /// Where a training job on the shape-d fleet ran: the job completed,
    /// claimed by a compute process, never the query tier or the scheduler
    /// — the catalog's claim, and that process's own line on the job where
    /// it wrote one and the logs are this process's to read.
    pub async fn shape_d_training_ran_on(
        &mut self,
        job_id: &str,
    ) -> Result<(JobRecord, RanOn), Box<dyn std::error::Error + Send + Sync>> {
        let record = self
            .await_completed(job_id, "the shape-d job completes")
            .await?;
        let mut ran_on = self
            .claim_ran_on(&record, &[MemberRole::Query, MemberRole::Scheduler])
            .await?;
        if let Some(line) = ran_on
            .label
            .as_deref()
            .and_then(|compute| self.log_line(compute, job_id))
        {
            ran_on.evidence.push(line);
        }
        Ok((record, ran_on))
    }

    /// Where the sink that committed `table_name` ran: the member in
    /// `writer`, proven by `placer`'s placement line, `scheduler`'s binding
    /// and the writer's own sink write — refused when the placer ran the
    /// materialization itself. On a joined fleet, whose logs are not this
    /// process's, the writer's identity alone.
    pub async fn placed_sink_ran_on(
        &mut self,
        table_name: &str,
        placer: MemberRole,
        scheduler: MemberRole,
        writer: MemberRole,
    ) -> Result<RanOn, Box<dyn std::error::Error + Send + Sync>> {
        let writer_label = self.label_of(writer)?;
        let worker = self.worker_of_label(&writer_label).await.ok();
        let mut ran_on = RanOn {
            instance_id: worker
                .as_ref()
                .map(|w| w.instance_id.clone())
                .unwrap_or_else(|| writer_label.clone()),
            label: Some(writer_label.clone()),
            host: worker.and_then(|w| w.host),
            role: writer.as_str().to_string(),
            evidence: Vec::new(),
        };
        if !self.has_logs() {
            return Ok(ran_on);
        }
        let placer_label = self.label_of(placer)?;
        let scheduler_label = self.label_of(scheduler)?;
        if let Some(local) = self.log_line(&placer_label, SINK_LOCAL_LOG) {
            return Err(format!(
                "the {} process ran the materialization itself, so this serve was not placed: \
                 {local}",
                placer.as_str()
            )
            .into());
        }
        ran_on.evidence.push(
            self.await_log_line(
                &placer_label,
                SINK_PLACED_LOG,
                "the placer's placement line",
            )
            .await?,
        );
        ran_on.evidence.push(
            self.await_log_line(
                &scheduler_label,
                BOUND_TASK_LOG,
                "the scheduler binding the sink's task",
            )
            .await?,
        );
        let write = self
            .await_log_line(&writer_label, SINK_WRITE_LOG, "the writer's sink write")
            .await?;
        if !write.contains(table_name) {
            let own = self.log_line(&writer_label, table_name).ok_or_else(|| {
                format!(
                    "the {} process wrote a table, but never {table_name}: {write}",
                    writer.as_str()
                )
            })?;
            ran_on.evidence.push(own);
        }
        ran_on.evidence.push(write);
        if let Some(line) = self.log_line(&placer_label, SINK_WRITE_LOG) {
            if line.contains(table_name) {
                return Err(format!(
                    "the {} process wrote {table_name} itself: {line}",
                    placer.as_str()
                )
                .into());
            }
        }
        Ok(ran_on)
    }

    /// A `file://` URL every member on this host can read, under `dir`.
    pub fn local_url(path: &Path) -> String {
        format!("file://{}", path.display())
    }
}

/// The observer: a session over the fleet's backends with no worker, which
/// submits and reads and never claims.
pub async fn observer_session(
    backends: &DistributedBackends,
    result_root: &str,
) -> Result<Arc<InferenceSession>, Box<dyn std::error::Error + Send + Sync>> {
    let dir = tempfile::tempdir()?;
    let artifact_dir: PathBuf = dir.path().to_path_buf();
    // The observer's scratch outlives the session: its fetch cache is where
    // published bundles are verified into.
    std::mem::forget(dir);
    let config = JammiConfig {
        artifact_dir,
        gpu: GpuConfig {
            device: -1,
            ..Default::default()
        },
        catalog: CatalogConfig::Postgres {
            url: backends.pg_url.clone().into(),
            pool_size: 8,
            max_lifetime_secs: None,
        },
        storage: StorageConfig {
            result_root: Some(result_root.to_string()),
            cloud: Some(backends.cloud()),
        },
        lease: LeaseConfig {
            duration_secs: 3,
            heartbeat_secs: 1,
        },
        worker: WorkerConfig {
            enabled: false,
            idle_poll_secs: 1,
            ..Default::default()
        },
        distributed: DistributedConfig {
            max_world_size: MAX_WORLD_SIZE,
        },
        ..Default::default()
    };
    Ok(InferenceSession::open(config).await?)
}
