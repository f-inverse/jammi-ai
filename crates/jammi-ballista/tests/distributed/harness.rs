//! The three(+)-process Ballista lane's harness: spawning real `jammi-server`
//! binaries with `[ballista]` roles configured (one hosting the scheduler +
//! an executor, the rest hosting executors only), plus the fine-tune
//! submission helpers (`register_training_source`, `submit_gang_fine_tune`,
//! `submit_fine_tune`, `JobSize`, `await_job`, `unique_source_name`,
//! `training_pairs_url`, `tiny_bert_model`, `label_of`) ported from
//! `crates/jammi-ai/tests/distributed/harness.rs`.
//!
//! COMMON.md's own instruction: `jammi-ai`'s harness is a private test module
//! of a DIFFERENT crate, unreachable from here, so a reduced copy is the
//! honest shape — named here as a copy, a candidate for the lead's
//! consolidation to lift into `jammi-test-utils`.

#![allow(dead_code)]

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
    CatalogConfig, DistributedConfig, JammiConfig, LeaseConfig, StorageConfig, WorkerConfig,
};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::storage::{CloudConfig, S3Config};
use jammi_db::store::CachePolicy;
use tempfile::TempDir;

/// The live backends this lane needs: Postgres, an S3-compatible object
/// store (MinIO), and the S3 credentials the spawned children authenticate
/// with. `detect()` returns the first missing variable's name, rather than
/// silently running a degraded lane.
pub struct Backends {
    pub pg_url: String,
    pub s3_endpoint: String,
    pub s3_bucket: String,
    pub access_key_id: String,
    pub secret_access_key: String,
    pub region: String,
}

impl Backends {
    pub fn detect() -> Result<Self, String> {
        fn var(name: &str) -> Option<String> {
            std::env::var(name).ok().filter(|s| !s.is_empty())
        }
        let need = |name: &str| var(name).ok_or_else(|| format!("{name} is unset"));
        Ok(Self {
            pg_url: need("JAMMI_TEST_PG_URL")?,
            s3_endpoint: need("JAMMI_TEST_S3_ENDPOINT")?,
            s3_bucket: need("JAMMI_TEST_S3_BUCKET")?,
            access_key_id: need("AWS_ACCESS_KEY_ID")?,
            secret_access_key: need("AWS_SECRET_ACCESS_KEY")?,
            region: var("AWS_REGION").unwrap_or_else(|| "us-east-1".to_string()),
        })
    }

    /// A unique `s3://bucket/prefix` for this test run.
    pub fn unique_result_root(&self, test: &str) -> String {
        format!(
            "s3://{}/dist-ballista-{}-{}",
            self.s3_bucket,
            test,
            uuid::Uuid::new_v4().simple()
        )
    }

    fn cloud(&self) -> CloudConfig {
        CloudConfig::S3(S3Config {
            region: Some(self.region.clone()),
            endpoint: Some(self.s3_endpoint.clone()),
            access_key_id: Some(self.access_key_id.clone()),
            secret_access_key: Some(self.secret_access_key.clone().into()),
            session_token: None,
            allow_http: self.s3_endpoint.starts_with("http://"),
        })
    }
}

/// `Backends::detect`, upgraded to a hard failure when
/// `JAMMI_REQUIRE_DISTRIBUTED` is set — this lane's own require-gate (the
/// same idiom `crates/jammi-ai/tests/distributed/{kill9_reclaim,gang_chaos}.rs`
/// each carry per-file). `None` is still returned for an ad-hoc local run
/// with no backends configured and no require flag.
#[allow(clippy::collapsible_if)]
pub fn required_backends(test: &str) -> Option<Backends> {
    let backends = Backends::detect();
    match backends {
        Ok(b) => Some(b),
        Err(reason) => {
            if std::env::var_os("JAMMI_REQUIRE_DISTRIBUTED").is_some() {
                panic!(
                    "{test}: JAMMI_REQUIRE_DISTRIBUTED is set but the distributed lane's shared \
                     backends are unconfigured ({reason}) — a silent skip is not acceptable here"
                );
            }
            eprintln!("SKIPPED {test}: {reason}");
            None
        }
    }
}

const LEASE_SECS: u64 = 3;
const HEARTBEAT_SECS: u64 = 1;
const IDLE_POLL_SECS: u64 = 1;
const RANK_TIMEOUT_SECS: u64 = 10;
/// `[distributed] max_world_size` every ballista-fleet process renders
/// (LANE brief item 1); the jobs this lane submits use `world_size = 2`.
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

/// An ephemeral, unused TCP port on localhost (bind-then-release).
pub fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind an ephemeral port")
        .local_addr()
        .unwrap()
        .port()
}

const TEST_AUDIT_MASTER_KEY: &str =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

/// This process's Ballista role, if any: `SchedulerAndExecutor` renders both
/// `[ballista] scheduler_bind` and `[ballista.executor]`; `Executor` renders
/// `[ballista.executor]` only (pointed at `scheduler_port`);
/// `SchedulerOnly` renders `[ballista] scheduler_bind` only (b2's second
/// scheduler); `None` renders no `[ballista]` section at all (the plain,
/// wave-3-path comparison fleet).
#[derive(Clone, Copy)]
pub enum BallistaRole {
    SchedulerAndExecutor { scheduler_port: u16 },
    Executor { scheduler_port: u16 },
    SchedulerOnly { scheduler_port: u16 },
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
    backends: &Backends,
    result_root: &str,
    artifact_dir: &str,
    spec: &ProcSpec,
) -> String {
    let allow_http = backends.s3_endpoint.starts_with("http://");
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
        BallistaRole::SchedulerOnly { scheduler_port } => {
            out.push_str(&format!(
                "\n[ballista]\nscheduler_bind = \"127.0.0.1:{scheduler_port}\"\n"
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

impl WorkerProc {
    /// The `http://host:port` this process's Ballista scheduler listens on,
    /// if it hosts one.
    pub fn scheduler_url(&self) -> Option<String> {
        match self.spec.ballista {
            BallistaRole::SchedulerAndExecutor { scheduler_port }
            | BallistaRole::SchedulerOnly { scheduler_port } => {
                Some(format!("http://127.0.0.1:{scheduler_port}"))
            }
            _ => None,
        }
    }
}

pub struct Fleet {
    workers: Vec<WorkerProc>,
    backends_for_respawn: (String, String), // (pg_url snapshot unused placeholder)
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
    pub fn spawn(backends: &Backends, result_root: &str, specs: Vec<ProcSpec>) -> Self {
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
        Self {
            workers,
            backends_for_respawn: (String::new(), String::new()),
        }
    }

    pub fn worker_labels(&self) -> Vec<&str> {
        self.workers.iter().map(|w| w.label.as_str()).collect()
    }

    /// The `i`-th spawned worker's label (0-indexed), in spawn order.
    pub fn label(&self, i: usize) -> &str {
        self.workers[i].label.as_str()
    }

    pub fn scheduler_url_of(&self, label: &str) -> String {
        self.workers
            .iter()
            .find(|w| w.label == label)
            .and_then(|w| w.scheduler_url())
            .unwrap_or_else(|| panic!("worker {label:?} hosts no Ballista scheduler"))
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
    /// process's listener is released). Used by (b1). The replacement is a
    /// freshly-minted instance (a new `instances` row): `instance_id` is
    /// minted at session construction, never externally supplied
    /// (`crates/jammi-ai/src/session.rs:464`), so a killed-then-respawned
    /// process cannot literally keep the OLD instance id — (b1)'s own
    /// assertion (contract acceptance list) only needs the OTHER executors'
    /// registrations and the job status rows to survive, which the shared
    /// catalog carries regardless of the replacement's own identity.
    pub fn respawn(&mut self, backends: &Backends, result_root: &str, label: &str) {
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
    backends: &Backends,
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
/// exit or the timeout, never silently waiting it out.
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
pub async fn harness_session(backends: &Backends, result_root: &str) -> (Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().expect("harness artifact_dir");
    let config = JammiConfig {
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

pub async fn register_training_source(session: &Arc<InferenceSession>, name: &str) {
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
            JobSize::Crashable => 60,
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

pub async fn submit_fine_tune(
    session: &Arc<InferenceSession>,
    source: &str,
    size: JobSize,
) -> (String, String) {
    let job = session
        .fine_tune(
            source,
            &tiny_bert_model(),
            &[
                "text_a".to_string(),
                "text_b".to_string(),
                "score".to_string(),
            ],
            FineTuneMethod::Lora,
            ModelTask::TextEmbedding,
            Some(lane_fine_tune_config(size)),
        )
        .await
        .expect("submit queued fine-tune job to shared catalog");
    (job.job_id.clone(), job.model_id().to_string())
}

/// A 2-file parquet directory source, disjoint keys — the SAME shape
/// `crates/jammi-ai/tests/it/pipeline.rs::write_two_file_source` uses for
/// its own `build_embedding_plan` oracle, ported here so the scan stage has
/// `partition_count >= 2` (contract §2.5).
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
    // source (oracle (a3)) must first lower that threshold on its OWN
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
