//! A fleet of `jammi-server` processes on one host: rendered role configs,
//! spawned binaries, captured logs, and the diagnostics a failure prints.
//! The distributed lanes and the ladder's `placed`/`shape-d` legs launch
//! their fleets through this one facility, so a fleet a test proves against
//! is the fleet a leg measures.
//!
//! A process is either [`ProcConfig::Rendered`] — a config this module
//! writes from a [`BallistaRole`] and a [`WorkerRole`], the ad-hoc fleets
//! the lanes build to isolate one mechanism — or [`ProcConfig::ShapeD`]: one
//! of the deployed topology's committed role configs
//! (`deploy/kubernetes/overlays/shape-d/jammi-<role>.toml`), run as
//! written, with the values that differ per box — catalog, store, devices,
//! listeners, the scheduler's address — layered over it through the
//! `JAMMI_<PATH>` environment exactly as the deployment layers them
//! ([`ShapeDRole::env`]). A process on another host is started the same
//! way with the same variables, so a fleet across hosts is the same fleet.

use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use tempfile::TempDir;

use jammi_datafusion::ComputeDeviceKind;

use crate::DistributedBackends;

const LEASE_SECS: u64 = 3;
const HEARTBEAT_SECS: u64 = 1;
const IDLE_POLL_SECS: u64 = 1;
const RANK_TIMEOUT_SECS: u64 = 10;
/// `[distributed] max_world_size` every rendered process carries; the jobs
/// the lanes submit use `world_size = 2`.
pub const MAX_WORLD_SIZE: u32 = 3;

/// The audit master key every spawned process is given.
pub const TEST_AUDIT_MASTER_KEY: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

/// The `jammi-server` binary a test process's own build produced, in the
/// SAME `CARGO_TARGET_DIR` this test process was built into (`cargo build
/// -p jammi-server --bin jammi-server --features storage-s3`).
pub fn jammi_server_binary() -> PathBuf {
    let test_exe = std::env::current_exe().expect("current_exe for binary resolution");
    let profile_dir = test_exe
        .parent()
        .and_then(Path::parent)
        .expect("test exe under {profile}/deps/");
    let bin = profile_dir.join(if cfg!(windows) {
        "jammi-server.exe"
    } else {
        "jammi-server"
    });
    assert!(
        bin.is_file(),
        "`jammi-server` binary not found at {}. Build it first: `cargo build -p jammi-server \
         --bin jammi-server --features storage-s3` into this same CARGO_TARGET_DIR.",
        bin.display()
    );
    bin
}

/// This process's Ballista roles, if any: `SchedulerAndExecutor` renders
/// `[ballista.scheduler]`, `[ballista.executor]` and `[ballista.client]`
/// (the process names itself, so the training attempts it claims are
/// placed); `SchedulerAndClient` renders `[ballista.scheduler]` and
/// `[ballista.client]` (a claimant that places every attempt and hosts no
/// executor, so none ever trains on it while a live executor can hold it);
/// `Scheduler` renders `[ballista.scheduler]` only (a scheduler that runs
/// no task itself, so no placed task ever lands on the process the plane
/// lives in); `Executor` renders `[ballista.executor]` only (pointed at
/// `scheduler_port`); `Client` renders `[ballista.client]` only (a query
/// tier whose materializations go to the scheduler at `scheduler_port`);
/// `None` renders no `[ballista]` section at all (the plain, unplaced
/// comparison fleet).
#[derive(Clone, Copy, Debug)]
pub enum BallistaRole {
    SchedulerAndExecutor { scheduler_port: u16 },
    SchedulerAndClient { scheduler_port: u16 },
    Scheduler { scheduler_port: u16 },
    Executor { scheduler_port: u16 },
    Client { scheduler_port: u16 },
    None,
}

impl BallistaRole {
    /// Whether a process of this role registers a `compute_executors` row:
    /// the roles that render `[ballista.executor]`. A client-role process
    /// submits and hosts no executor; an unplaced one has no plane at all.
    pub fn hosts_executor(self) -> bool {
        matches!(
            self,
            BallistaRole::SchedulerAndExecutor { .. } | BallistaRole::Executor { .. }
        )
    }
}

/// This process's `[worker]` shape.
#[derive(Clone, Copy, Debug)]
pub struct WorkerRole {
    pub enabled: bool,
    /// `None` = `kinds = "all"`; `Some(ks)` = exactly `ks` — `Some(&[])` is a
    /// fleet member that claims nothing.
    pub kinds: Option<&'static [&'static str]>,
    pub idle_poll_secs: u64,
}

impl Default for WorkerRole {
    fn default() -> Self {
        Self {
            enabled: true,
            kinds: None,
            idle_poll_secs: IDLE_POLL_SECS,
        }
    }
}

/// The deployed topology's three roles, by the committed config each runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeDRole {
    /// `jammi-scheduler.toml`: the one Ballista scheduler; claims nothing,
    /// runs no task.
    Scheduler,
    /// `jammi-query.toml`: the query tier, a Ballista client; the surface a
    /// job is submitted through; claims nothing.
    Query,
    /// `jammi-compute.toml`: an executor beside a worker that claims every
    /// training kind and trains it on this process's devices.
    Compute,
}

impl ShapeDRole {
    pub fn as_str(self) -> &'static str {
        match self {
            ShapeDRole::Scheduler => "scheduler",
            ShapeDRole::Query => "query",
            ShapeDRole::Compute => "compute",
        }
    }

    /// The committed config this role runs, under the repository root.
    pub fn config_path(self, repo_root: &Path) -> PathBuf {
        repo_root
            .join("deploy/kubernetes/overlays/shape-d")
            .join(format!("jammi-{}.toml", self.as_str()))
    }

    /// The `JAMMI_<PATH>` variables that place this role on one box: the
    /// backends every role shares and the listeners and addresses that
    /// differ per process — the same layer the deployment's Secret and
    /// downward-API `env` provide, so the committed file is run as written.
    /// Every role of one fleet is placed with the same `compute_device`:
    /// the compute role trains on it, and the query role names its kind as
    /// the kind a model's plan is placed onto.
    pub fn env(self, place: &ShapeDPlace<'_>) -> Vec<(String, String)> {
        let b = place.backends;
        let mut env = vec![
            (
                "JAMMI_ARTIFACT_DIR".to_string(),
                place.artifact_dir.display().to_string(),
            ),
            ("JAMMI_CATALOG__POSTGRES__URL".to_string(), b.pg_url.clone()),
            (
                "JAMMI_CATALOG__POSTGRES__POOL_SIZE".to_string(),
                "8".to_string(),
            ),
            (
                "JAMMI_STORAGE__RESULT_ROOT".to_string(),
                place.result_root.to_string(),
            ),
            (
                "JAMMI_STORAGE__CLOUD__S3__REGION".to_string(),
                b.region.clone(),
            ),
            (
                "JAMMI_STORAGE__CLOUD__S3__ENDPOINT".to_string(),
                b.s3_endpoint.clone(),
            ),
            (
                "JAMMI_STORAGE__CLOUD__S3__ALLOW_HTTP".to_string(),
                b.allows_http().to_string(),
            ),
            (
                "JAMMI_LEASE__DURATION_SECS".to_string(),
                LEASE_SECS.to_string(),
            ),
            (
                "JAMMI_LEASE__HEARTBEAT_SECS".to_string(),
                HEARTBEAT_SECS.to_string(),
            ),
            (
                "JAMMI_SERVER__FLIGHT_LISTEN".to_string(),
                format!("{}:{}", place.bind_host, place.ports.flight),
            ),
            (
                "JAMMI_SERVER__HEALTH_LISTEN".to_string(),
                format!("{}:{}", place.bind_host, place.ports.health),
            ),
        ];
        match self {
            // The scheduler and query roles compute nothing: no device.
            ShapeDRole::Scheduler => env.extend([
                ("JAMMI_GPU__DEVICE".to_string(), "-1".to_string()),
                (
                    "JAMMI_BALLISTA__SCHEDULER__BIND".to_string(),
                    format!("0.0.0.0:{}", place.ports.scheduler),
                ),
                (
                    "JAMMI_BALLISTA__SCHEDULER__ADVERTISE_HOST".to_string(),
                    place.advertise_host.to_string(),
                ),
            ]),
            ShapeDRole::Query => env.extend([
                ("JAMMI_GPU__DEVICE".to_string(), "-1".to_string()),
                (
                    "JAMMI_BALLISTA__CLIENT__SCHEDULER_ADDRESS".to_string(),
                    place.scheduler_address.to_string(),
                ),
                (
                    "JAMMI_BALLISTA__CLIENT__DEVICE_KIND".to_string(),
                    place.compute_kind().wire_str().to_string(),
                ),
            ]),
            ShapeDRole::Compute => env.extend([
                (
                    "JAMMI_BALLISTA__EXECUTOR__SCHEDULER_ADDRESS".to_string(),
                    place.scheduler_address.to_string(),
                ),
                (
                    "JAMMI_BALLISTA__EXECUTOR__BIND".to_string(),
                    format!("{}:{}", place.bind_host, place.ports.exec_bind),
                ),
                (
                    "JAMMI_BALLISTA__EXECUTOR__GRPC_BIND".to_string(),
                    format!("{}:{}", place.bind_host, place.ports.exec_grpc),
                ),
                (
                    "JAMMI_BALLISTA__EXECUTOR__ADVERTISE_HOST".to_string(),
                    place.advertise_host.to_string(),
                ),
                (
                    "JAMMI_SERVER__PEER_BIND".to_string(),
                    format!("{}:{}", place.bind_host, place.ports.peer),
                ),
                (
                    "JAMMI_SERVER__PEER_ADVERTISE".to_string(),
                    format!("{}:{}", place.advertise_host, place.ports.peer),
                ),
                (
                    "JAMMI_GPU__DEVICE".to_string(),
                    place.compute_device.to_string(),
                ),
                (
                    "JAMMI_GPU__DEVICES".to_string(),
                    format!("[{}]", place.compute_device),
                ),
                ("JAMMI_WORKER__LOCAL_RANKS".to_string(), "1".to_string()),
                (
                    "JAMMI_WORKER__IDLE_POLL_SECS".to_string(),
                    IDLE_POLL_SECS.to_string(),
                ),
                (
                    "JAMMI_WORKER__RANK_TIMEOUT_SECS".to_string(),
                    RANK_TIMEOUT_SECS.to_string(),
                ),
            ]),
        }
        env
    }
}

/// The values that place one shape-d role on one box.
#[derive(Clone, Copy)]
pub struct ShapeDPlace<'a> {
    pub backends: &'a DistributedBackends,
    pub result_root: &'a str,
    pub artifact_dir: &'a Path,
    /// The interface the process binds (`127.0.0.1` on one host, `0.0.0.0`
    /// across hosts).
    pub bind_host: &'a str,
    /// The name other processes dial this one by.
    pub advertise_host: &'a str,
    /// `host:port` of the fleet's scheduler.
    pub scheduler_address: &'a str,
    pub ports: Ports,
    /// The CUDA ordinal the fleet's compute tier trains on, or `-1` for the
    /// CPU — one fact of the fleet, the same for every role placed in it.
    pub compute_device: i32,
}

impl ShapeDPlace<'_> {
    /// The device kind the fleet's compute tier holds.
    pub fn compute_kind(&self) -> ComputeDeviceKind {
        if self.compute_device < 0 {
            ComputeDeviceKind::Cpu
        } else {
            ComputeDeviceKind::Cuda
        }
    }
}

/// One process's listener ports.
#[derive(Clone, Copy, Debug)]
pub struct Ports {
    pub flight: u16,
    pub health: u16,
    pub peer: u16,
    /// Meaningful on a scheduler-hosting process.
    pub scheduler: u16,
    /// Meaningful on an executor-hosting process.
    pub exec_bind: u16,
    pub exec_grpc: u16,
}

impl Ports {
    /// Six free loopback ports.
    pub fn fresh() -> Self {
        Self {
            flight: crate::free_port(),
            health: crate::free_port(),
            peer: crate::free_port(),
            scheduler: crate::free_port(),
            exec_bind: crate::free_port(),
            exec_grpc: crate::free_port(),
        }
    }
}

/// How one process is configured.
#[derive(Clone, Copy, Debug)]
pub enum ProcConfig {
    /// A config rendered from the two roles, on `device` (a CUDA ordinal,
    /// or `-1` for CPU).
    Rendered {
        ballista: BallistaRole,
        worker: WorkerRole,
        device: i32,
    },
    /// The deployed topology's committed config for `role`, layered over
    /// through the environment; `scheduler_port` is the fleet's scheduler,
    /// `compute_device` the ordinal the fleet's compute tier trains on.
    ShapeD {
        role: ShapeDRole,
        scheduler_port: u16,
        compute_device: i32,
    },
}

/// One process's full listener/role spec.
#[derive(Clone, Copy, Debug)]
pub struct ProcSpec {
    pub ports: Ports,
    pub config: ProcConfig,
}

impl ProcSpec {
    /// A rendered process on fresh ports, on CPU.
    pub fn fresh(ballista: BallistaRole, worker: WorkerRole) -> Self {
        Self::fresh_on(ballista, worker, -1)
    }

    /// A rendered process on fresh ports, on `device` (a CUDA ordinal, or
    /// `-1` for CPU).
    pub fn fresh_on(ballista: BallistaRole, worker: WorkerRole, device: i32) -> Self {
        Self {
            ports: Ports::fresh(),
            config: ProcConfig::Rendered {
                ballista,
                worker,
                device,
            },
        }
    }

    /// A shape-d role on fresh ports. `scheduler_port` is the fleet's one
    /// scheduler port: the scheduler role binds it, every other role dials
    /// it; `compute_device` is the fleet's compute tier's CUDA ordinal (or
    /// `-1`), the same for every role of the fleet.
    pub fn shape_d(role: ShapeDRole, scheduler_port: u16, compute_device: i32) -> Self {
        let mut ports = Ports::fresh();
        ports.scheduler = scheduler_port;
        Self {
            ports,
            config: ProcConfig::ShapeD {
                role,
                scheduler_port,
                compute_device,
            },
        }
    }

    pub fn flight_port(&self) -> u16 {
        self.ports.flight
    }

    /// Whether this process hosts an executor.
    pub fn hosts_executor(&self) -> bool {
        match self.config {
            ProcConfig::Rendered { ballista, .. } => ballista.hosts_executor(),
            ProcConfig::ShapeD { role, .. } => role == ShapeDRole::Compute,
        }
    }

    /// Whether this process runs a `[worker]`.
    pub fn worker_enabled(&self) -> bool {
        match self.config {
            ProcConfig::Rendered { worker, .. } => worker.enabled,
            ProcConfig::ShapeD { role, .. } => role == ShapeDRole::Compute,
        }
    }
}

fn render_toml(
    backends: &DistributedBackends,
    result_root: &str,
    artifact_dir: &str,
    ports: Ports,
    ballista: BallistaRole,
    worker: WorkerRole,
    device: i32,
) -> String {
    let allow_http = backends.allows_http();
    let kinds = match worker.kinds {
        Some(kinds) => format!(
            "kinds = [{}]",
            kinds
                .iter()
                .map(|k| format!("\"{k}\""))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        None => "kinds = \"all\"".to_string(),
    };
    let mut out = format!(
        r#"
artifact_dir = "{artifact_dir}"

[gpu]
device = {device}

[catalog.postgres]
url = "{pg_url}"
pool_size = 8

[storage]
result_root = "{result_root}"

[storage.cloud.s3]
region = "{region}"
endpoint = "{s3_endpoint}"
allow_http = {allow_http}

[lease]
duration_secs = {LEASE_SECS}
heartbeat_secs = {HEARTBEAT_SECS}

[worker]
enabled = {enabled}
{kinds}
idle_poll_secs = {idle_poll_secs}
local_ranks = 1
rank_timeout_secs = {RANK_TIMEOUT_SECS}

[distributed]
max_world_size = {MAX_WORLD_SIZE}

[server]
flight_listen = "127.0.0.1:{flight_port}"
health_listen = "127.0.0.1:{health_port}"
peer_bind = "127.0.0.1:{peer_port}"
peer_advertise = "127.0.0.1:{peer_port}"
services = []
"#,
        pg_url = backends.pg_url,
        region = backends.region,
        s3_endpoint = backends.s3_endpoint,
        enabled = worker.enabled,
        idle_poll_secs = worker.idle_poll_secs,
        flight_port = ports.flight,
        health_port = ports.health,
        peer_port = ports.peer,
    );

    // A scheduler is bound the way a deployment binds it (every interface)
    // and advertised by a dialable host, so every placed task's status
    // report exercises the advertised name.
    let scheduler_section = |scheduler_port: u16| {
        format!(
            "\n[ballista.scheduler]\nbind = \"0.0.0.0:{scheduler_port}\"\n\
             advertise_host = \"127.0.0.1\"\n"
        )
    };
    let client_section = |scheduler_port: u16| {
        format!("\n[ballista.client]\nscheduler_address = \"127.0.0.1:{scheduler_port}\"\n")
    };
    let executor_section = |scheduler_port: u16| {
        format!(
            "\n[ballista.executor]\nscheduler_address = \"127.0.0.1:{scheduler_port}\"\n\
             bind = \"127.0.0.1:{}\"\ngrpc_bind = \"127.0.0.1:{}\"\n\
             advertise_host = \"127.0.0.1\"\ntask_slots = 1\n",
            ports.exec_bind, ports.exec_grpc,
        )
    };
    match ballista {
        BallistaRole::None => {}
        BallistaRole::Scheduler { scheduler_port } => {
            out.push_str(&scheduler_section(scheduler_port));
        }
        BallistaRole::SchedulerAndClient { scheduler_port } => {
            out.push_str(&scheduler_section(scheduler_port));
            out.push_str(&client_section(scheduler_port));
        }
        BallistaRole::SchedulerAndExecutor { scheduler_port } => {
            out.push_str(&scheduler_section(scheduler_port));
            out.push_str(&executor_section(scheduler_port));
            out.push_str(&client_section(scheduler_port));
        }
        BallistaRole::Executor { scheduler_port } => {
            out.push_str(&executor_section(scheduler_port));
        }
        BallistaRole::Client { scheduler_port } => {
            out.push_str(&client_section(scheduler_port));
        }
    }
    out
}

/// One spawned `jammi-server` process and the scratch dir backing its
/// config + log. Killed on drop via the owning [`Fleet`].
pub struct WorkerProc {
    pub label: String,
    child: Child,
    /// The effective config: the rendered file, or the committed file's
    /// path and the environment layered over it.
    effective_config: String,
    log_path: PathBuf,
    _scratch: TempDir,
    spec: ProcSpec,
}

/// A fleet of processes on this host.
pub struct Fleet {
    exe: PathBuf,
    repo_root: PathBuf,
    workers: Vec<WorkerProc>,
    run_id: String,
}

impl Fleet {
    /// Spawn one process per `specs[i]` from `exe`, labelled
    /// `lane-{run_id}-{i+1}` — a per-`Fleet` unique run id (never a fixed
    /// `lane-1`/`lane-2`/…): the shared Postgres catalog's `workers` row
    /// lookup is by LABEL, and a fixed label would collide with a PRIOR
    /// run's still-present row for the same label (Postgres persists across
    /// the whole live lane, unlike SQLite's per-test isolation) — a stale
    /// row would silently resolve to the wrong instance. A shape-d process
    /// runs the committed config under `repo_root`.
    pub fn spawn(
        exe: &Path,
        repo_root: &Path,
        backends: &DistributedBackends,
        result_root: &str,
        specs: Vec<ProcSpec>,
    ) -> Self {
        let run_id = uuid::Uuid::new_v4().simple().to_string()[..8].to_string();
        let workers = specs
            .into_iter()
            .enumerate()
            .map(|(i, spec)| {
                spawn_one(
                    exe,
                    repo_root,
                    backends,
                    result_root,
                    &format!("lane-{run_id}-{}", i + 1),
                    spec,
                )
            })
            .collect();
        Self {
            exe: exe.to_path_buf(),
            repo_root: repo_root.to_path_buf(),
            workers,
            run_id,
        }
    }

    pub fn worker_labels(&self) -> Vec<&str> {
        self.workers.iter().map(|w| w.label.as_str()).collect()
    }

    /// The labels of the members whose role hosts an executor, in spawn
    /// order — the set that registers with the compute plane. Every one of
    /// them runs a `[worker]`: a member is known to the shared catalog by
    /// its label only through its `workers` row, so an executor whose
    /// worker is disabled has an executor registration no test can tie
    /// back to it — refused here, never a 60 s timeout.
    pub fn executor_labels(&self) -> Vec<&str> {
        self.workers
            .iter()
            .filter(|w| w.spec.hosts_executor())
            .inspect(|w| {
                assert!(
                    w.spec.worker_enabled(),
                    "executor-hosting member {} runs no worker: its label resolves to no \
                     instance id; give it a worker of a kind the test never enqueues",
                    w.label
                )
            })
            .map(|w| w.label.as_str())
            .collect()
    }

    /// The `i`-th spawned worker's label (0-indexed), in spawn order.
    pub fn label(&self, i: usize) -> &str {
        self.workers[i].label.as_str()
    }

    /// The spec the `i`-th process was spawned from.
    pub fn spec(&self, i: usize) -> &ProcSpec {
        &self.workers[i].spec
    }

    /// The Flight SQL address of the worker labelled `label`.
    pub fn flight_addr(&self, label: &str) -> std::net::SocketAddr {
        let w = self.worker(label);
        std::net::SocketAddr::from(([127, 0, 0, 1], w.spec.ports.flight))
    }

    fn worker(&self, label: &str) -> &WorkerProc {
        self.workers
            .iter()
            .find(|w| w.label == label)
            .unwrap_or_else(|| panic!("no worker labelled {label:?}"))
    }

    /// Spawn ONE more process into this already-running fleet, labelled
    /// with the SAME run id (`lane-{run_id}-{n}`, `n` continuing the
    /// existing sequence) — a LATE-joining process, added only after the
    /// earlier processes' own claim/placement race has already resolved,
    /// so it plays no part in that race. Returns the new process's own
    /// index (for [`Fleet::label`]).
    pub fn spawn_more(
        &mut self,
        backends: &DistributedBackends,
        result_root: &str,
        spec: ProcSpec,
    ) -> usize {
        let idx = self.workers.len();
        let label = format!("lane-{}-{}", self.run_id, idx + 1);
        self.workers.push(spawn_one(
            &self.exe,
            &self.repo_root,
            backends,
            result_root,
            &label,
            spec,
        ));
        idx
    }

    /// The captured stdout+stderr log of the worker labelled `label`, read
    /// fresh (the process may still be writing it).
    pub fn log_contents(&self, label: &str) -> String {
        std::fs::read_to_string(&self.worker(label).log_path).unwrap_or_default()
    }

    pub fn kill9(&mut self, label: &str) -> bool {
        let Some(w) = self.workers.iter_mut().find(|w| w.label == label) else {
            return false;
        };
        sigkill(&mut w.child);
        true
    }

    /// Spawn a REPLACEMENT process at the SAME index, with the SAME spec
    /// (same ports — a fixed `scheduler.bind` rebinds once the killed
    /// process's listener is released). The replacement is a freshly-minted
    /// instance (a new `instances` row): `InferenceSession::instance_id` is
    /// minted at session construction, never externally supplied, so a
    /// killed-then-respawned process cannot keep the OLD instance id — a
    /// scheduler-restart test needs only the OTHER executors' registrations
    /// and the job status rows to survive, which the shared catalog carries
    /// regardless of the replacement's own identity.
    pub fn respawn(&mut self, backends: &DistributedBackends, result_root: &str, label: &str) {
        let idx = self
            .workers
            .iter()
            .position(|w| w.label == label)
            .unwrap_or_else(|| panic!("no worker labelled {label:?} to respawn"));
        let spec = self.workers[idx].spec;
        // The old listener may take a moment to release after SIGKILL.
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let probe = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                spawn_one(
                    &self.exe,
                    &self.repo_root,
                    backends,
                    result_root,
                    label,
                    spec,
                )
            }));
            match probe {
                Ok(w) => {
                    self.workers[idx] = w;
                    return;
                }
                Err(e) => {
                    if Instant::now() >= deadline {
                        std::panic::resume_unwind(e);
                    }
                    std::thread::sleep(Duration::from_millis(200));
                }
            }
        }
    }

    /// The first member that has exited other than by SIGKILL, with its
    /// status.
    pub fn first_unexpected_exit(&mut self) -> Option<(String, std::process::ExitStatus)> {
        for w in &mut self.workers {
            if let Ok(Some(status)) = w.child.try_wait() {
                if status.signal() == Some(libc::SIGKILL) {
                    continue;
                }
                return Some((w.label.clone(), status));
            }
        }
        None
    }

    pub fn dump_diagnostics(&self, context: &str) {
        eprintln!("\n========== fleet diagnostics: {context} ==========");
        for w in &self.workers {
            eprintln!("\n----- worker {} -----", w.label);
            eprintln!("[effective config]\n{}", w.effective_config);
            match std::fs::read_to_string(&w.log_path) {
                Ok(log) if log.trim().is_empty() => {
                    eprintln!("[worker stdout+stderr] <empty> ({})", w.log_path.display())
                }
                Ok(log) => eprintln!("[worker stdout+stderr {}]\n{log}", w.log_path.display()),
                Err(e) => eprintln!(
                    "[worker stdout+stderr] <unreadable: {e}> ({})",
                    w.log_path.display()
                ),
            }
        }
        eprintln!("========== end diagnostics: {context} ==========\n");
    }
}

impl Drop for Fleet {
    fn drop(&mut self) {
        for w in &mut self.workers {
            sigkill(&mut w.child);
            let _ = w.child.wait();
        }
    }
}

fn sigkill(child: &mut Child) {
    let pid = child.id() as libc::pid_t;
    // SAFETY: `pid` is a child this process spawned; SIGKILL is unconditional
    // and synchronous. A racing exit makes the call a no-op (ESRCH).
    unsafe {
        libc::kill(pid, libc::SIGKILL);
    }
}

/// The variables every spawned process inherits from this one, by name:
/// the kernel arm a leg is run under must reach the process that trains.
const PASSED_THROUGH: &[&str] = &["JAMMI_KERNELS_DISABLE", "RUST_LOG", "CUDA_VISIBLE_DEVICES"];

fn spawn_one(
    exe: &Path,
    repo_root: &Path,
    backends: &DistributedBackends,
    result_root: &str,
    label: &str,
    spec: ProcSpec,
) -> WorkerProc {
    let scratch = TempDir::new().expect("worker scratch dir");
    let artifact_dir = scratch.path().join("artifacts");
    std::fs::create_dir_all(&artifact_dir).expect("worker artifact_dir");

    let (config_path, env, effective_config) = match spec.config {
        ProcConfig::Rendered {
            ballista,
            worker,
            device,
        } => {
            let config_toml = render_toml(
                backends,
                result_root,
                artifact_dir.to_str().expect("utf8 artifact_dir"),
                spec.ports,
                ballista,
                worker,
                device,
            );
            let config_path = scratch.path().join("jammi.toml");
            std::fs::write(&config_path, &config_toml).expect("write worker config");
            (config_path, Vec::new(), config_toml)
        }
        ProcConfig::ShapeD {
            role,
            scheduler_port,
            compute_device,
        } => {
            let config_path = role.config_path(repo_root);
            assert!(
                config_path.is_file(),
                "the shape-d {} role's committed config is not at {}",
                role.as_str(),
                config_path.display()
            );
            let scheduler_address = format!("127.0.0.1:{scheduler_port}");
            let env = role.env(&ShapeDPlace {
                backends,
                result_root,
                artifact_dir: &artifact_dir,
                bind_host: "127.0.0.1",
                advertise_host: "127.0.0.1",
                scheduler_address: &scheduler_address,
                ports: spec.ports,
                compute_device,
            });
            let effective = std::iter::once(format!("--config {}", config_path.display()))
                .chain(env.iter().map(|(k, v)| format!("{k}={v}")))
                .collect::<Vec<_>>()
                .join("\n");
            (config_path, env, effective)
        }
    };

    let log_path = scratch.path().join("worker.log");
    let log = std::fs::File::create(&log_path).expect("worker log file");
    let log_err = log.try_clone().expect("clone worker log fd");

    let mut command = Command::new(exe);
    command
        .arg("--config")
        .arg(&config_path)
        .env("JAMMI_WORKER_ID", label)
        .env("AWS_ACCESS_KEY_ID", &backends.access_key_id)
        .env("AWS_SECRET_ACCESS_KEY", &backends.secret_access_key)
        .env("AWS_REGION", &backends.region)
        .env("JAMMI_AUDIT_MASTER_KEY", TEST_AUDIT_MASTER_KEY)
        .envs(env)
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(log_err));
    for name in PASSED_THROUGH {
        if let Ok(value) = std::env::var(name) {
            command.env(name, value);
        }
    }
    let child = command
        .spawn()
        .unwrap_or_else(|e| panic!("spawn `jammi-server` worker {label}: {e}"));

    WorkerProc {
        label: label.to_string(),
        child,
        effective_config,
        log_path,
        _scratch: scratch,
        spec,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backends() -> DistributedBackends {
        DistributedBackends {
            pg_url: "postgres://u:p@db:5432/jammi".into(),
            s3_endpoint: "http://store:9000".into(),
            s3_bucket: "jammi-dist".into(),
            access_key_id: "k".into(),
            secret_access_key: "s".into(),
            region: "us-east-1".into(),
        }
    }

    fn place<'a>(backends: &'a DistributedBackends, artifact_dir: &'a Path) -> ShapeDPlace<'a> {
        ShapeDPlace {
            backends,
            result_root: "s3://jammi-dist/legs",
            artifact_dir,
            bind_host: "0.0.0.0",
            advertise_host: "compute-0.fleet",
            scheduler_address: "scheduler.fleet:50050",
            ports: Ports {
                flight: 8815,
                health: 8080,
                peer: 9000,
                scheduler: 50050,
                exec_bind: 50051,
                exec_grpc: 50052,
            },
            compute_device: 1,
        }
    }

    fn value<'a>(env: &'a [(String, String)], key: &str) -> Option<&'a str> {
        env.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    /// Every role dials the one scheduler by its address; the roles that
    /// compute nothing carry no device, and the compute role carries the
    /// device it trains on and the listeners it advertises.
    #[test]
    fn shape_d_roles_layer_the_box_over_the_committed_config() {
        let backends = backends();
        let dir = Path::new("/var/lib/jammi");
        let scheduler = ShapeDRole::Scheduler.env(&place(&backends, dir));
        let query = ShapeDRole::Query.env(&place(&backends, dir));
        let compute = ShapeDRole::Compute.env(&place(&backends, dir));
        for env in [&scheduler, &query, &compute] {
            assert_eq!(
                value(env, "JAMMI_CATALOG__POSTGRES__URL"),
                Some("postgres://u:p@db:5432/jammi")
            );
            assert_eq!(
                value(env, "JAMMI_STORAGE__RESULT_ROOT"),
                Some("s3://jammi-dist/legs")
            );
            assert_eq!(
                value(env, "JAMMI_STORAGE__CLOUD__S3__ALLOW_HTTP"),
                Some("true")
            );
        }
        assert_eq!(
            value(&scheduler, "JAMMI_BALLISTA__SCHEDULER__BIND"),
            Some("0.0.0.0:50050")
        );
        assert_eq!(value(&scheduler, "JAMMI_GPU__DEVICE"), Some("-1"));
        assert_eq!(
            value(&query, "JAMMI_BALLISTA__CLIENT__SCHEDULER_ADDRESS"),
            Some("scheduler.fleet:50050")
        );
        assert_eq!(value(&query, "JAMMI_GPU__DEVICE"), Some("-1"));
        assert_eq!(
            value(&query, "JAMMI_BALLISTA__CLIENT__DEVICE_KIND"),
            Some("cuda"),
            "the query tier places a model's plan onto the compute tier's kind"
        );
        assert_eq!(
            value(&compute, "JAMMI_BALLISTA__EXECUTOR__SCHEDULER_ADDRESS"),
            Some("scheduler.fleet:50050")
        );
        assert_eq!(
            value(&compute, "JAMMI_BALLISTA__EXECUTOR__ADVERTISE_HOST"),
            Some("compute-0.fleet")
        );
        assert_eq!(
            value(&compute, "JAMMI_SERVER__PEER_ADVERTISE"),
            Some("compute-0.fleet:9000")
        );
        assert_eq!(value(&compute, "JAMMI_GPU__DEVICE"), Some("1"));
        assert_eq!(value(&compute, "JAMMI_GPU__DEVICES"), Some("[1]"));
        assert_eq!(value(&compute, "JAMMI_WORKER__LOCAL_RANKS"), Some("1"));
        assert!(value(&scheduler, "JAMMI_BALLISTA__EXECUTOR__BIND").is_none());
        assert!(value(&query, "JAMMI_BALLISTA__SCHEDULER__BIND").is_none());
    }

    /// A fleet whose compute tier trains on the CPU has a query tier that
    /// places a model's plan onto the CPU: the committed query config names
    /// the deployment's CUDA tier, and the box it runs on is layered over it.
    #[test]
    fn a_cpu_fleets_query_tier_names_the_cpu() {
        let backends = backends();
        let cpu_fleet = ShapeDPlace {
            compute_device: -1,
            ..place(&backends, Path::new("/var/lib/jammi"))
        };
        let query = ShapeDRole::Query.env(&cpu_fleet);
        assert_eq!(
            value(&query, "JAMMI_BALLISTA__CLIENT__DEVICE_KIND"),
            Some("cpu")
        );
        let compute = ShapeDRole::Compute.env(&cpu_fleet);
        assert_eq!(value(&compute, "JAMMI_GPU__DEVICE"), Some("-1"));
    }

    /// The config each role runs is the deployment's own file.
    #[test]
    fn shape_d_roles_run_the_committed_configs() {
        let root = crate::workspace_root();
        for role in [
            ShapeDRole::Scheduler,
            ShapeDRole::Query,
            ShapeDRole::Compute,
        ] {
            let path = role.config_path(&root);
            assert!(path.is_file(), "{} is missing", path.display());
            assert!(path.ends_with(format!("shape-d/jammi-{}.toml", role.as_str())));
        }
    }
}
