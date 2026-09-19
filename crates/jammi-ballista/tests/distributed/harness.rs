//! The three(+)-process Ballista lane's harness: spawning real `jammi-server`
//! binaries with `[ballista]` roles configured (one hosting the scheduler +
//! an executor, the rest hosting executors only), plus the fine-tune
//! submission helpers (`add_training_source`, `submit_gang_fine_tune`,
//! `JobSize`, `await_job`, `unique_source_name`,
//! `training_pairs_url`, `tiny_bert_model`, `label_of`) ported from
//! `crates/jammi-ai/tests/distributed/harness.rs`.
//!
//! `jammi-ai`'s harness is a private test module of a DIFFERENT crate,
//! unreachable from here, so this is a reduced copy of it; the shared part
//! belongs in `jammi-test-utils`.

use std::net::TcpListener;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use jammi_ai::fine_tune::spec::{TrainingCommon, TrainingSpec};
use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod};
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::config::{
    CatalogConfig, DistributedConfig, InferenceConfig, JammiConfig, LeaseConfig, StorageConfig,
    WorkerConfig,
};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::store::CachePolicy;
use jammi_test_utils::DistributedBackends;
use tempfile::TempDir;

const LEASE_SECS: u64 = 3;
const HEARTBEAT_SECS: u64 = 1;
const IDLE_POLL_SECS: u64 = 1;
const RANK_TIMEOUT_SECS: u64 = 10;
/// `[distributed] max_world_size` every ballista-fleet process renders; the
/// jobs this lane submits use `world_size = 2`.
const MAX_WORLD_SIZE: u32 = 3;

/// Locate the `jammi-server` binary this workspace's `cargo build -p
/// jammi-server --bin jammi-server --features storage-s3` produced, in the
/// SAME `CARGO_TARGET_DIR` this test process itself was built into.
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

/// An unused TCP port on localhost for a listener a SPAWNED process binds
/// later. Never `bind(:0)`-then-release: that hands out a port from the
/// kernel's ephemeral range, the same range every outgoing `connect()` this
/// test process makes (Postgres, MinIO) draws its local port from, so the
/// released port can be taken by a client socket before the child binds it
/// (CI run 35134806942, lane-1: "failed to bind OSS server listeners:
/// Address already in use"). Ports come from a range BELOW every platform's
/// ephemeral floor (Linux 32768, macOS 49152), verified bindable at pick
/// time, and never handed out twice by this process.
pub fn free_port() -> u16 {
    use std::collections::HashSet;
    use std::hash::{BuildHasher, Hasher};
    use std::sync::Mutex;
    static HANDED_OUT: Mutex<Option<HashSet<u16>>> = Mutex::new(None);
    const LO: u32 = 20_000;
    const SPAN: u32 = 12_000;
    let mut guard = HANDED_OUT.lock().expect("port ledger lock poisoned");
    let handed = guard.get_or_insert_with(HashSet::new);
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    );
    let mut cursor = (h.finish() % u64::from(SPAN)) as u32;
    for _ in 0..SPAN {
        let port = (LO + cursor) as u16;
        cursor = (cursor + 1) % SPAN;
        if handed.contains(&port) {
            continue;
        }
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            handed.insert(port);
            return port;
        }
    }
    panic!(
        "no bindable port in {LO}..{} for the lane's fleet",
        LO + SPAN
    );
}

const TEST_AUDIT_MASTER_KEY: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

/// This process's Ballista role, if any: `SchedulerAndExecutor` renders both
/// `[ballista] scheduler_bind` and `[ballista.executor]`; `Executor` renders
/// `[ballista.executor]` only (pointed at `scheduler_port`); `None` renders
/// no `[ballista]` section at all (the plain, unplaced comparison fleet).
#[derive(Clone, Copy)]
pub enum BallistaRole {
    SchedulerAndExecutor { scheduler_port: u16 },
    Executor { scheduler_port: u16 },
    None,
}

/// This process's fine-tune-facing `[worker]` shape.
#[derive(Clone, Copy)]
pub struct WorkerRole {
    pub enabled: bool,
    /// `None` = `kinds = "all"`; `Some(k)` = `kinds = [k]`.
    pub kind: Option<&'static str>,
    pub idle_poll_secs: u64,
}

impl Default for WorkerRole {
    fn default() -> Self {
        Self {
            enabled: true,
            kind: None,
            idle_poll_secs: IDLE_POLL_SECS,
        }
    }
}

/// One process's full listener/role spec.
#[derive(Clone, Copy)]
pub struct ProcSpec {
    pub flight_port: u16,
    pub health_port: u16,
    pub peer_port: u16,
    pub ballista: BallistaRole,
    /// Only meaningful when `ballista` carries an executor role: this
    /// process's own Flight-shuffle/gRPC-task listener ports.
    pub exec_bind_port: u16,
    pub exec_grpc_port: u16,
    pub worker: WorkerRole,
}

impl ProcSpec {
    pub fn fresh(ballista: BallistaRole, worker: WorkerRole) -> Self {
        Self {
            flight_port: free_port(),
            health_port: free_port(),
            peer_port: free_port(),
            ballista,
            exec_bind_port: free_port(),
            exec_grpc_port: free_port(),
            worker,
        }
    }
}

fn render_toml(
    backends: &DistributedBackends,
    result_root: &str,
    artifact_dir: &str,
    spec: &ProcSpec,
) -> String {
    let allow_http = backends.allows_http();
    let kinds = match spec.worker.kind {
        Some(k) => format!("kinds = [\"{k}\"]"),
        None => "kinds = \"all\"".to_string(),
    };
    let mut out = format!(
        r#"
artifact_dir = "{artifact_dir}"

[gpu]
device = -1

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
        enabled = spec.worker.enabled,
        idle_poll_secs = spec.worker.idle_poll_secs,
        flight_port = spec.flight_port,
        health_port = spec.health_port,
        peer_port = spec.peer_port,
    );

    match spec.ballista {
        BallistaRole::None => {}
        BallistaRole::SchedulerAndExecutor { scheduler_port } => {
            out.push_str(&format!(
                "\n[ballista]\nscheduler_bind = \"127.0.0.1:{scheduler_port}\"\n"
            ));
            out.push_str(&format!(
                "\n[ballista.executor]\nscheduler_address = \"127.0.0.1:{scheduler_port}\"\n\
                 bind = \"127.0.0.1:{}\"\ngrpc_bind = \"127.0.0.1:{}\"\n\
                 advertise_host = \"127.0.0.1\"\ntask_slots = 1\n",
                spec.exec_bind_port, spec.exec_grpc_port,
            ));
        }
        BallistaRole::Executor { scheduler_port } => {
            out.push_str(&format!(
                "\n[ballista.executor]\nscheduler_address = \"127.0.0.1:{scheduler_port}\"\n\
                 bind = \"127.0.0.1:{}\"\ngrpc_bind = \"127.0.0.1:{}\"\n\
                 advertise_host = \"127.0.0.1\"\ntask_slots = 1\n",
                spec.exec_bind_port, spec.exec_grpc_port,
            ));
        }
    }
    out
}

/// One spawned `jammi-server` process and the scratch dir backing its
/// config + log. Killed on drop via the owning [`Fleet`].
pub struct WorkerProc {
    pub label: String,
    child: Child,
    config_toml: String,
    log_path: PathBuf,
    _scratch: TempDir,
    spec: ProcSpec,
}

pub struct Fleet {
    workers: Vec<WorkerProc>,
    run_id: String,
}

impl Fleet {
    /// Spawn one process per `specs[i]`, labelled `lane-{run_id}-{i+1}` — a
    /// per-`Fleet` unique run id (never a fixed `lane-1`/`lane-2`/…): the
    /// shared Postgres catalog's `workers` row lookup is by LABEL
    /// (`harness::label_of`/`instance_id_of_label`), and a fixed label
    /// would collide with a PRIOR test run's still-present row for the same
    /// label (Postgres persists across the whole live-lane process, unlike
    /// SQLite's per-test-fixture isolation) — a stale row would silently
    /// resolve to the wrong instance.
    pub fn spawn(backends: &DistributedBackends, result_root: &str, specs: Vec<ProcSpec>) -> Self {
        let exe = jammi_server_binary();
        let run_id = uuid::Uuid::new_v4().simple().to_string()[..8].to_string();
        let workers = specs
            .into_iter()
            .enumerate()
            .map(|(i, spec)| {
                spawn_one(
                    &exe,
                    backends,
                    result_root,
                    &format!("lane-{run_id}-{}", i + 1),
                    spec,
                )
            })
            .collect();
        Self { workers, run_id }
    }

    pub fn worker_labels(&self) -> Vec<&str> {
        self.workers.iter().map(|w| w.label.as_str()).collect()
    }

    /// The `i`-th spawned worker's label (0-indexed), in spawn order.
    pub fn label(&self, i: usize) -> &str {
        self.workers[i].label.as_str()
    }

    /// Spawn ONE more process into this already-running fleet, labelled
    /// with the SAME run id (`lane-{run_id}-{n}`, `n` continuing the
    /// existing sequence) — a LATE-joining process (e.g. the kill test's independent
    /// reclaimer), added only after the earlier processes' own claim/
    /// placement race has already resolved, so it plays no part in that
    /// race. Returns the new process's own index (for `Fleet::label`).
    pub fn spawn_more(
        &mut self,
        backends: &DistributedBackends,
        result_root: &str,
        spec: ProcSpec,
    ) -> usize {
        let exe = jammi_server_binary();
        let idx = self.workers.len();
        let label = format!("lane-{}-{}", self.run_id, idx + 1);
        self.workers
            .push(spawn_one(&exe, backends, result_root, &label, spec));
        idx
    }

    /// The captured stdout+stderr log of the worker labelled `label`, read
    /// fresh (the process may still be writing it).
    pub fn log_contents(&self, label: &str) -> String {
        let w = self
            .workers
            .iter()
            .find(|w| w.label == label)
            .unwrap_or_else(|| panic!("no worker labelled {label:?}"));
        std::fs::read_to_string(&w.log_path).unwrap_or_default()
    }

    pub fn kill9(&mut self, label: &str) -> bool {
        let Some(w) = self.workers.iter_mut().find(|w| w.label == label) else {
            return false;
        };
        sigkill(&mut w.child);
        true
    }

    /// Spawn a REPLACEMENT process at the SAME index, with the SAME spec
    /// (same ports — a fixed `scheduler_bind` rebinds once the killed
    /// process's listener is released). The replacement is a freshly-minted
    /// instance (a new `instances` row): `InferenceSession::instance_id` is
    /// minted at session construction, never externally supplied, so a
    /// killed-then-respawned process cannot keep the OLD instance id — the
    /// scheduler-restart test needs only the OTHER executors' registrations
    /// and the job status rows to survive, which the shared catalog carries
    /// regardless of the replacement's own identity.
    pub fn respawn(&mut self, backends: &DistributedBackends, result_root: &str, label: &str) {
        let exe = jammi_server_binary();
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
                spawn_one(&exe, backends, result_root, label, spec)
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

    fn first_unexpected_exit(&mut self) -> Option<(String, std::process::ExitStatus)> {
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
        eprintln!("\n========== distributed ballista lane diagnostics: {context} ==========");
        for w in &self.workers {
            eprintln!("\n----- worker {} -----", w.label);
            eprintln!("[effective jammi.toml]\n{}", w.config_toml);
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

fn spawn_one(
    exe: &Path,
    backends: &DistributedBackends,
    result_root: &str,
    label: &str,
    spec: ProcSpec,
) -> WorkerProc {
    let scratch = TempDir::new().expect("worker scratch dir");
    let artifact_dir = scratch.path().join("artifacts");
    std::fs::create_dir_all(&artifact_dir).expect("worker artifact_dir");

    let config_path = scratch.path().join("jammi.toml");
    let config_toml = render_toml(
        backends,
        result_root,
        artifact_dir.to_str().expect("utf8 artifact_dir"),
        &spec,
    );
    std::fs::write(&config_path, &config_toml).expect("write worker config");

    let log_path = scratch.path().join("worker.log");
    let log = std::fs::File::create(&log_path).expect("worker log file");
    let log_err = log.try_clone().expect("clone worker log fd");

    let child = Command::new(exe)
        .arg("--config")
        .arg(&config_path)
        .env("JAMMI_WORKER_ID", label)
        .env("AWS_ACCESS_KEY_ID", &backends.access_key_id)
        .env("AWS_SECRET_ACCESS_KEY", &backends.secret_access_key)
        .env("AWS_REGION", &backends.region)
        .env("JAMMI_AUDIT_MASTER_KEY", TEST_AUDIT_MASTER_KEY)
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(log_err))
        .spawn()
        .unwrap_or_else(|e| panic!("spawn `jammi-server` worker {label}: {e}"));

    WorkerProc {
        label: label.to_string(),
        child,
        config_toml,
        log_path,
        _scratch: scratch,
        spec,
    }
}

/// The generous terminal-state timeout — cold boot + Postgres connect +
/// migrate + a tiny CPU LoRA fine-tune + publish to MinIO + finalize, under a
/// 3s lease with reclaim on a crash.
pub const TERMINAL_TIMEOUT: Duration = Duration::from_secs(150);
pub const POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Poll the shared catalog for `job_id` until `want(&record)` holds. Fails
/// loudly (fleet diagnostics + final job row) on an early unexpected worker
/// exit or the timeout, never silently waiting it out. Also fails loudly the
/// moment the row reaches a TERMINAL status (`JobRecord::is_terminal`, the
/// ONE terminality predicate) `want` does not accept: a terminal
/// row never mutates further, so a fixture polling for one specific literal
/// status fails immediately naming the status it actually settled on,
/// rather than burning the rest of [`TERMINAL_TIMEOUT`] on a DIFFERENT
/// terminal status instead.
pub async fn await_job(
    fleet: &mut Fleet,
    session: &Arc<InferenceSession>,
    job_id: &str,
    label: &str,
    mut want: impl FnMut(&jammi_db::catalog::jobs_repo::JobRecord) -> bool,
) -> jammi_db::catalog::jobs_repo::JobRecord {
    let catalog = session.catalog().pinned_to_tenant(None);
    let deadline = Instant::now() + TERMINAL_TIMEOUT;
    loop {
        if let Ok(record) = catalog.get_job(job_id).await {
            if want(&record) {
                return record;
            }
            if record.is_terminal() {
                fleet.dump_diagnostics(&format!(
                    "job {job_id} settled on status={:?} error={:?}, which does not satisfy: \
                     {label}",
                    record.status, record.error
                ));
                panic!(
                    "distributed ballista lane: job {job_id} reached a TERMINAL status ({:?}) \
                     that does not satisfy: {label} (error: {:?}) — a terminal row never \
                     mutates further, so waiting out the remaining timeout cannot help.",
                    record.status, record.error
                );
            }
        }
        if let Some((worker, status)) = fleet.first_unexpected_exit() {
            fleet.dump_diagnostics(&format!(
                "worker {worker} exited unexpectedly ({status}) while awaiting: {label}"
            ));
            panic!(
                "distributed ballista lane: worker {worker} exited unexpectedly ({status}) \
                 before the fleet could satisfy: {label}."
            );
        }
        if Instant::now() >= deadline {
            fleet.dump_diagnostics(&format!("timed out after {TERMINAL_TIMEOUT:?}: {label}"));
            panic!("distributed ballista lane: timed out after {TERMINAL_TIMEOUT:?}: {label}.");
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

/// Poll the captured log of the worker labelled `label` until it contains
/// `needle`, returning the log. A log line a worker writes AFTER the catalog
/// fact a test already awaited (the placed submitter's `HandedOff` line
/// lands only once the gang's result stream has drained, which is after the
/// coordinator committed `completed`) is never a single read: CI run
/// 35127543679 read the submitter's log 240 ms after the job finished and
/// found the line absent. Fails loudly (fleet diagnostics) on an early
/// unexpected worker exit or the timeout.
pub async fn await_log_contains(
    fleet: &mut Fleet,
    label: &str,
    needle: &str,
    what: &str,
) -> String {
    let deadline = Instant::now() + TERMINAL_TIMEOUT;
    loop {
        let log = fleet.log_contents(label);
        if log.contains(needle) {
            return log;
        }
        if let Some((worker, status)) = fleet.first_unexpected_exit() {
            fleet.dump_diagnostics(&format!(
                "worker {worker} exited unexpectedly ({status}) while awaiting the log line: {what}"
            ));
            panic!(
                "distributed ballista lane: worker {worker} exited unexpectedly ({status}) \
                 before {label}'s log showed: {what}."
            );
        }
        if Instant::now() >= deadline {
            fleet.dump_diagnostics(&format!(
                "timed out after {TERMINAL_TIMEOUT:?} awaiting the log line: {what}"
            ));
            panic!(
                "distributed ballista lane: timed out after {TERMINAL_TIMEOUT:?}: {label}'s log \
                 never showed: {what}; log:\n{log}"
            );
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

/// Poll for a plain condition over the catalog with no `Fleet` diagnostics
/// (used where no fleet failure is expected to interrupt the wait, e.g. a
/// `submit_physical_plan` result already resolved).
pub async fn await_condition(timeout: Duration, mut predicate: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if predicate() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    false
}

/// Build the harness's own observer session against the shared Postgres +
/// MinIO, rooted at `result_root`. `[worker] enabled = false`: it only
/// submits and observes.
pub async fn harness_session(
    backends: &DistributedBackends,
    result_root: &str,
) -> (Arc<InferenceSession>, TempDir) {
    harness_session_with(backends, result_root, InferenceConfig::default()).await
}

/// [`harness_session`] planning its inference with `inference` — the fan-out
/// and chunk size a submitted plan carries to the executors.
pub async fn harness_session_with(
    backends: &DistributedBackends,
    result_root: &str,
    inference: InferenceConfig,
) -> (Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().expect("harness artifact_dir");
    let config = JammiConfig {
        inference,
        artifact_dir: dir.path().to_path_buf(),
        gpu: jammi_db::config::GpuConfig {
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
            duration_secs: LEASE_SECS,
            heartbeat_secs: HEARTBEAT_SECS,
        },
        worker: WorkerConfig {
            enabled: false,
            idle_poll_secs: IDLE_POLL_SECS,
            ..Default::default()
        },
        distributed: DistributedConfig {
            max_world_size: MAX_WORLD_SIZE,
        },
        ..Default::default()
    };
    let session = InferenceSession::open(config)
        .await
        .expect("harness session connects to shared Postgres + MinIO");
    (session, dir)
}

/// The `JAMMI_WORKER_ID` label of the fleet member whose minted instance id
/// is `instance_id`.
pub async fn label_of(session: &InferenceSession, instance_id: &str) -> String {
    let workers = session.catalog().list_workers().await.unwrap();
    workers
        .iter()
        .find(|w| w.instance_id == instance_id)
        .and_then(|w| w.label.clone())
        .unwrap_or_else(|| {
            panic!(
                "claimed_by {instance_id:?} is not a labelled fleet member; workers = {workers:?}"
            )
        })
}

pub fn training_pairs_url() -> String {
    jammi_test_utils::fixture_url("training_pairs.csv")
}

pub fn tiny_bert_model() -> String {
    format!(
        "local:{}",
        jammi_test_utils::cookbook_fixture("tiny_bert")
            .to_str()
            .expect("utf8 tiny_bert path")
    )
}

pub fn unique_source_name(role: &str) -> String {
    format!("{role}-{}", uuid::Uuid::new_v4().simple())
}

pub async fn add_training_source(session: &Arc<InferenceSession>, name: &str) {
    session
        .add_source(
            name,
            SourceType::File,
            SourceConnection {
                url: Some(training_pairs_url()),
                format: Some(FileFormat::Csv),
                ..Default::default()
            },
        )
        .await
        .unwrap_or_else(|e| panic!("register training source {name}: {e}"));
}

#[derive(Clone, Copy)]
pub enum JobSize {
    Quick,
    Crashable,
}

impl JobSize {
    fn epochs(self) -> usize {
        match self {
            JobSize::Quick => 3,
            // The placed path (a working Ballista fleet, unlike a plain
            // in-process claim) resolves the placement round-trip almost
            // instantly on localhost — 60 epochs of `tiny_bert` LoRA
            // complete inside the harness's own detect+kill window (the
            // kill test then reads a single-entry `claimed_by` sequence:
            // the job finished before the kill landed). The kill
            // lands ~1 s after the placed claim, so the job must outlive
            // that; the SUCCESSOR then runs the whole job again from
            // scratch and must finish inside the reclaim wait. Measured
            // rates: ~15 epochs/s locally, ~5 epochs/s on a CI runner
            // (900 epochs reach about epoch 734 at the 150 s deadline).
            // 450 epochs keeps the job crashable (≥ 30 s at
            // the fast rate) and completes in ~90 s at the slow one.
            JobSize::Crashable => 450,
        }
    }
}

fn lane_fine_tune_config(size: JobSize) -> FineTuneConfig {
    FineTuneConfig {
        epochs: size.epochs(),
        batch_size: 8,
        lora_rank: 4,
        warmup_steps: 0,
        ..Default::default()
    }
}

/// Submit one durable `world_size`-rank LoRA fine-tune over `source`.
/// Returns `(job_id, output_model_id)`.
pub async fn submit_gang_fine_tune(
    session: &Arc<InferenceSession>,
    source: &str,
    size: JobSize,
    world_size: u32,
) -> (String, String) {
    let job = session
        .run_training_spec(TrainingSpec::FineTune {
            source: source.to_string(),
            columns: vec![
                "text_a".to_string(),
                "text_b".to_string(),
                "score".to_string(),
            ],
            method: FineTuneMethod::Lora,
            task: ModelTask::TextEmbedding,
            common: TrainingCommon {
                base_model: tiny_bert_model(),
                config: lane_fine_tune_config(size),
                world_size,
            },
            cache: CachePolicy::Bypass,
        })
        .await
        .expect("submit a queued gang fine-tune job to the shared catalog");
    (job.job_id.clone(), job.model_id().to_string())
}

/// A 2-file parquet directory source, disjoint keys — the SAME shape
/// `crates/jammi-ai/tests/it/pipeline.rs::write_two_file_source` uses for
/// its own `build_embedding_plan` oracle, ported here so the scan stage has
/// `partition_count >= 2`.
pub fn write_two_file_source(dir: &Path) -> String {
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;

    let src_dir = dir.join("two_files");
    std::fs::create_dir_all(&src_dir).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("text", DataType::Utf8, false),
    ]));
    // 4 rows/file, disjoint id ranges. DataFusion's file-group builder
    // coalesces files below `datafusion.optimizer.repartition_file_min_size`
    // (10 MiB default) into ONE partition regardless of `target_partitions`
    // — every caller that needs `partition_count >= 2` out of this small a
    // source must first lower that threshold on its OWN
    // session (`SET datafusion.optimizer.repartition_file_min_size = 1`),
    // never inflate file size to work around it.
    const ROWS_PER_FILE: i64 = 4;
    for f in 0..2i64 {
        let base = f * ROWS_PER_FILE;
        let ids: Vec<i64> = (base..base + ROWS_PER_FILE).collect();
        let texts: Vec<String> = (0..ROWS_PER_FILE)
            .map(|i| format!("file {f} row {i} about topic {}", (base + i) % 37))
            .collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(texts)),
            ],
        )
        .unwrap();
        let file = std::fs::File::create(src_dir.join(format!("part{f}.parquet"))).unwrap();
        let mut w = ArrowWriter::try_new(file, schema.clone(), None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }
    format!("file://{}", src_dir.display())
}
