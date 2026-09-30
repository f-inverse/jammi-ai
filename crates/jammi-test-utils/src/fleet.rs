//! A fleet of `jammi-server` processes on one host: spawned binaries,
//! captured logs, and the diagnostics a failure prints. The ladder's
//! `shape-d` legs launch their fleets through this facility.
//!
//! Every process runs one of the deployed topology's committed role configs
//! ([`ShapeDRole::config_path`]) as written, with
//! the values that differ per box — catalog, store, devices, listeners —
//! layered over it through the `JAMMI_<PATH>` environment exactly as the
//! deployment layers them ([`ShapeDRole::env`]). A process on another host
//! is started the same way with the same variables, so a fleet across hosts
//! is the same fleet.

use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use tempfile::TempDir;

use crate::DistributedBackends;

const LEASE_SECS: u64 = 3;
const HEARTBEAT_SECS: u64 = 1;
const IDLE_POLL_SECS: u64 = 1;
const RANK_TIMEOUT_SECS: u64 = 10;
/// `[distributed] max_world_size` an observer of a fleet carries.
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

/// The deployed topology's two roles, by the committed config each runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeDRole {
    /// The base query tier's `jammi.toml`: the surface a job is submitted
    /// through; claims nothing.
    Query,
    /// `jammi-compute.toml`: a worker that claims every job kind and runs it
    /// on this process's devices.
    Compute,
}

impl ShapeDRole {
    pub fn as_str(self) -> &'static str {
        match self {
            ShapeDRole::Query => "query",
            ShapeDRole::Compute => "compute",
        }
    }

    /// The committed config this role runs, under the repository root: the
    /// base query tier's, or the Shape D overlay's compute tier's.
    pub fn config_path(self, repo_root: &Path) -> PathBuf {
        let kube = repo_root.join("deploy/kubernetes");
        match self {
            ShapeDRole::Query => kube.join("base/jammi.toml"),
            ShapeDRole::Compute => kube.join("overlays/shape-d/jammi-compute.toml"),
        }
    }

    /// The `JAMMI_<PATH>` variables that place this role on one box: the
    /// backends every role shares and the listeners and addresses that
    /// differ per process — the same layer the deployment's Secret and
    /// downward-API `env` provide, so the committed file is run as written.
    /// The compute role runs its jobs on `compute_device`.
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
            // The query role computes nothing: no device.
            ShapeDRole::Query => env.push(("JAMMI_GPU__DEVICE".to_string(), "-1".to_string())),
            ShapeDRole::Compute => env.extend([
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
    pub ports: Ports,
    /// The CUDA ordinal the fleet's compute tier runs on, or `-1` for the
    /// CPU — one fact of the fleet.
    pub compute_device: i32,
}

/// One process's listener ports.
#[derive(Clone, Copy, Debug)]
pub struct Ports {
    pub flight: u16,
    pub health: u16,
    pub peer: u16,
}

impl Ports {
    /// Three free loopback ports.
    pub fn fresh() -> Self {
        Self {
            flight: crate::free_port(),
            health: crate::free_port(),
            peer: crate::free_port(),
        }
    }
}

/// One process's listeners and role.
#[derive(Clone, Copy, Debug)]
pub struct ProcSpec {
    pub ports: Ports,
    pub role: ShapeDRole,
    /// The CUDA ordinal the fleet's compute tier runs on, or `-1`.
    pub compute_device: i32,
}

impl ProcSpec {
    /// A shape-d role on fresh ports; `compute_device` is the fleet's
    /// compute tier's CUDA ordinal (or `-1`), the same for every role of the
    /// fleet.
    pub fn shape_d(role: ShapeDRole, compute_device: i32) -> Self {
        Self {
            ports: Ports::fresh(),
            role,
            compute_device,
        }
    }

    pub fn flight_port(&self) -> u16 {
        self.ports.flight
    }

    /// Whether this process runs a `[worker]`.
    pub fn worker_enabled(&self) -> bool {
        self.role == ShapeDRole::Compute
    }
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
    workers: Vec<WorkerProc>,
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
        Self { workers }
    }

    pub fn worker_labels(&self) -> Vec<&str> {
        self.workers.iter().map(|w| w.label.as_str()).collect()
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

    /// The captured stdout+stderr log of the worker labelled `label`, read
    /// fresh (the process may still be writing it).
    pub fn log_contents(&self, label: &str) -> String {
        std::fs::read_to_string(&self.worker(label).log_path).unwrap_or_default()
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

    let config_path = spec.role.config_path(repo_root);
    assert!(
        config_path.is_file(),
        "the shape-d {} role's committed config is not at {}",
        spec.role.as_str(),
        config_path.display()
    );
    let env = spec.role.env(&ShapeDPlace {
        backends,
        result_root,
        artifact_dir: &artifact_dir,
        bind_host: "127.0.0.1",
        advertise_host: "127.0.0.1",
        ports: spec.ports,
        compute_device: spec.compute_device,
    });
    let effective_config = std::iter::once(format!("--config {}", config_path.display()))
        .chain(env.iter().map(|(k, v)| format!("{k}={v}")))
        .collect::<Vec<_>>()
        .join(
            "
",
        );

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
            ports: Ports {
                flight: 8815,
                health: 8080,
                peer: 9000,
            },
            compute_device: 1,
        }
    }

    fn value<'a>(env: &'a [(String, String)], key: &str) -> Option<&'a str> {
        env.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    /// Both roles share the fleet's backends; the query role carries no
    /// device, and the compute role carries the device it runs on and the
    /// peer listener it advertises.
    #[test]
    fn shape_d_roles_layer_the_box_over_the_committed_config() {
        let backends = backends();
        let dir = Path::new("/var/lib/jammi");
        let query = ShapeDRole::Query.env(&place(&backends, dir));
        let compute = ShapeDRole::Compute.env(&place(&backends, dir));
        for env in [&query, &compute] {
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
        assert_eq!(value(&query, "JAMMI_GPU__DEVICE"), Some("-1"));
        assert!(value(&query, "JAMMI_SERVER__PEER_BIND").is_none());
        assert_eq!(
            value(&compute, "JAMMI_SERVER__PEER_ADVERTISE"),
            Some("compute-0.fleet:9000")
        );
        assert_eq!(value(&compute, "JAMMI_GPU__DEVICE"), Some("1"));
        assert_eq!(value(&compute, "JAMMI_GPU__DEVICES"), Some("[1]"));
        assert_eq!(value(&compute, "JAMMI_WORKER__LOCAL_RANKS"), Some("1"));
    }

    /// The config each role runs is the deployment's own file.
    #[test]
    fn shape_d_roles_run_the_committed_configs() {
        let root = crate::workspace_root();
        for role in [ShapeDRole::Query, ShapeDRole::Compute] {
            let path = role.config_path(&root);
            assert!(path.is_file(), "{} is missing", path.display());
        }
    }
}
