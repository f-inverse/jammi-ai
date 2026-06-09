//! Shared machinery for the distributed-validation lane: backend discovery, a
//! shared-backend [`InferenceSession`] for the harness itself, a [`Fleet`] of
//! spawned `jammi serve` worker processes with RAII teardown, and a
//! fixed-sleep-free catalog poller.
//!
//! The harness is the *submitter and observer*; the spawned child processes are
//! the *workers*. The harness session is built with no train tier and no
//! embedded worker, so it never claims a job itself — it only writes queued rows
//! (via `fine_tune`) and polls the shared catalog for the children's terminal
//! writes. This mirrors the production split where a submitting client and the
//! GPU worker fleet are different processes against one catalog.

use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::Arc;
use std::time::{Duration, Instant};

use jammi_ai::fine_tune::{FineTuneConfig, FineTuneMethod};
use jammi_ai::model::ModelTask;
use jammi_ai::session::InferenceSession;
use jammi_db::config::{CatalogConfig, JammiConfig, StorageConfig, TrainingConfig};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::storage::{CloudConfig, S3Config};
use tempfile::TempDir;

/// The shared backends the lane needs, discovered from the environment. Absent
/// any one of them the whole lane is unconfigured and every test skips.
pub struct Backends {
    /// Shared Postgres catalog URL (`JAMMI_TEST_PG_URL`).
    pub pg_url: String,
    /// MinIO S3 endpoint (`JAMMI_TEST_S3_ENDPOINT`, e.g. `http://127.0.0.1:9000`).
    pub s3_endpoint: String,
    /// Pre-created bucket the lane roots artifacts under (`JAMMI_TEST_S3_BUCKET`).
    pub s3_bucket: String,
    /// MinIO access key (`AWS_ACCESS_KEY_ID`).
    pub access_key_id: String,
    /// MinIO secret key (`AWS_SECRET_ACCESS_KEY`).
    pub secret_access_key: String,
    /// Region (`AWS_REGION`); defaults to `us-east-1` for MinIO.
    pub region: String,
}

impl Backends {
    /// Discover the shared backends, or `None` (with a `tracing::warn` naming the
    /// first missing variable) when the lane is unconfigured. A test calls this
    /// first and early-returns on `None` — the no-`#[ignore]` skip discipline.
    pub fn from_env_or_skip(test: &str) -> Option<Self> {
        fn var(name: &str) -> Option<String> {
            std::env::var(name).ok().filter(|s| !s.is_empty())
        }
        let missing = |name: &str| -> Option<String> {
            match var(name) {
                Some(v) => Some(v),
                None => {
                    tracing::warn!(
                        test,
                        var = name,
                        "distributed lane unconfigured ({name} unset); skipping"
                    );
                    None
                }
            }
        };
        Some(Self {
            pg_url: missing("JAMMI_TEST_PG_URL")?,
            s3_endpoint: missing("JAMMI_TEST_S3_ENDPOINT")?,
            s3_bucket: missing("JAMMI_TEST_S3_BUCKET")?,
            access_key_id: missing("AWS_ACCESS_KEY_ID")?,
            secret_access_key: missing("AWS_SECRET_ACCESS_KEY")?,
            region: var("AWS_REGION").unwrap_or_else(|| "us-east-1".to_string()),
        })
    }

    /// A unique `s3://bucket/prefix` for this test run, so concurrent runs (or a
    /// re-run after a flake) never collide on the shared bucket. The prefix
    /// carries the test name for log-archaeology on a failure.
    pub fn unique_result_root(&self, test: &str) -> String {
        format!(
            "s3://{}/dist-{}-{}",
            self.s3_bucket,
            test,
            uuid::Uuid::new_v4().simple()
        )
    }

    /// The S3 [`CloudConfig`] threaded to every object-store driver: the MinIO
    /// endpoint, the test creds, `allow_http` (MinIO speaks plain HTTP locally),
    /// and the region. The same credentials the spawned children inherit via
    /// their `AWS_*` env, so the harness reads exactly what the workers wrote.
    fn cloud(&self) -> CloudConfig {
        CloudConfig::S3(S3Config {
            region: Some(self.region.clone()),
            endpoint: Some(self.s3_endpoint.clone()),
            access_key_id: Some(self.access_key_id.clone()),
            secret_access_key: Some(self.secret_access_key.clone()),
            session_token: None,
            allow_http: self.s3_endpoint.starts_with("http://"),
        })
    }
}

/// The validated short worker timing the lane drives: a 3 s lease, 1 s
/// heartbeat, 1 s idle poll. The heartbeat clears the strict `heartbeat*2 <
/// lease` margin (`2 < 3`), so a live worker renews well inside the lease while
/// reclaim of a *dead* worker's job happens within roughly one poll + lease (a
/// few seconds) — fast enough for the kill-9 reclaim assertion, slow enough that
/// a live short run never spuriously loses its lease.
const LEASE_SECS: u64 = 3;
const HEARTBEAT_SECS: u64 = 1;
const IDLE_POLL_SECS: u64 = 1;

/// Build the harness's own session against the shared Postgres + MinIO,
/// rooted at `result_root`. It mounts no train tier and spawns no worker, so it
/// only submits queued jobs and observes — the spawned children do the claiming.
///
/// Returns the session plus the [`TempDir`] backing its local fetch cache /
/// artifact_dir, which must outlive the session.
pub async fn harness_session(
    backends: &Backends,
    result_root: &str,
) -> (Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().expect("harness artifact_dir");
    let config = shared_config(backends, result_root, dir.path());
    let session = InferenceSession::open(config)
        .await
        .expect("harness session connects to shared Postgres + MinIO");
    (session, dir)
}

/// The config the harness session and the spawned workers share: the same
/// Postgres catalog, the same MinIO-backed `result_root`, the same short worker
/// timing. `artifact_dir` is per-process local scratch (fetch cache, logs); the
/// durable state lives entirely in the shared catalog + object store.
fn shared_config(backends: &Backends, result_root: &str, artifact_dir: &Path) -> JammiConfig {
    JammiConfig {
        artifact_dir: artifact_dir.to_path_buf(),
        // CPU-only: the lane validates orchestration/durability, not kernels.
        gpu: jammi_db::config::GpuConfig {
            device: -1,
            ..Default::default()
        },
        catalog: CatalogConfig::Postgres {
            url: backends.pg_url.clone(),
            pool_size: 8,
            max_lifetime_secs: None,
        },
        storage: StorageConfig {
            result_root: Some(result_root.to_string()),
            cloud: Some(backends.cloud()),
        },
        training: TrainingConfig {
            lease_duration_secs: LEASE_SECS,
            heartbeat_interval_secs: HEARTBEAT_SECS,
            idle_poll_secs: IDLE_POLL_SECS,
        },
        ..Default::default()
    }
}

/// One spawned `jammi serve` worker process and the scratch dir backing its
/// config + log. Killed on drop via the owning [`Fleet`].
struct WorkerProc {
    worker_id: String,
    child: Child,
    /// The effective `jammi.toml` this worker was launched with. Held verbatim
    /// so a failure dump shows exactly the catalog / storage / tier / timing the
    /// worker was configured with — no re-deriving it from the scratch dir.
    config_toml: String,
    /// Path to the worker's captured stdout+stderr log under its scratch dir.
    /// Surfaced on a failure so a CI-only lane can see the worker's own view
    /// (device selection, claim attempts, the publish error, a panic).
    log_path: PathBuf,
    /// Per-worker scratch (its `jammi.toml`, artifact_dir, stdout/stderr log).
    /// Kept so it outlives the child and survives until [`Fleet`] teardown.
    _scratch: TempDir,
}

/// An RAII fleet of spawned `jammi serve` worker processes against the shared
/// catalog + object store. Dropping the fleet SIGKILLs every still-running child
/// (even on a test panic, via `Drop`), so a failed assertion never leaks a
/// worker process holding a lease on the shared catalog.
pub struct Fleet {
    workers: Vec<WorkerProc>,
}

impl Fleet {
    /// Spawn `n` `jammi serve` workers, each with a distinct `JAMMI_WORKER_ID`
    /// (`worker-1`..`worker-n`), distinct gRPC + health ports, the shared
    /// catalog + `result_root`, the short worker timing, and the train tier
    /// mounted. The MinIO credentials are passed through the child env so the
    /// worker's S3 driver authenticates exactly as the harness session does.
    pub fn spawn(backends: &Backends, result_root: &str, n: usize) -> Self {
        let exe = jammi_serve_binary();
        let workers = (1..=n)
            .map(|i| spawn_worker(&exe, backends, result_root, &format!("worker-{i}"), i))
            .collect();
        Self { workers }
    }

    /// The seeded ids of the spawned workers, in spawn order — `worker-1`..`-n`.
    /// Property assertions match `claimed_by` against these.
    pub fn worker_ids(&self) -> Vec<&str> {
        self.workers.iter().map(|w| w.worker_id.as_str()).collect()
    }

    /// SIGKILL exactly one worker by its seeded id, returning whether it was
    /// found and signalled. Used by the kill-9 reclaim and artifact-crash-window
    /// properties to crash a *specific* claimer mid-job.
    pub fn kill9(&mut self, worker_id: &str) -> bool {
        let Some(w) = self.workers.iter_mut().find(|w| w.worker_id == worker_id) else {
            return false;
        };
        sigkill(&mut w.child);
        true
    }

    /// The id and exit status of the first worker that has exited on its own —
    /// i.e. NOT one a test deliberately `kill9`'d. A worker we crashed exits with
    /// a SIGKILL-shaped status (`signal() == Some(SIGKILL)`); any other exit (a
    /// config rejection, a panic, a missing-feature `SchemeNotEnabled`, a clean
    /// `0`) means the worker died for a reason the lane did not intend, and the
    /// poll should fail loudly with that worker's log rather than wait out the
    /// full terminal timeout. `None` means every worker is still running (or was
    /// intentionally killed).
    ///
    /// `try_wait` is non-blocking; a still-running child returns `Ok(None)` and
    /// is skipped. A reaped child (we already `wait`'d it) also yields no status.
    fn first_unexpected_exit(&mut self) -> Option<(String, std::process::ExitStatus)> {
        for w in &mut self.workers {
            if let Ok(Some(status)) = w.child.try_wait() {
                if status.signal() == Some(libc::SIGKILL) {
                    // A worker a property deliberately crashed — expected.
                    continue;
                }
                return Some((w.worker_id.clone(), status));
            }
        }
        None
    }

    /// Dump every worker's effective config and captured stdout+stderr to the
    /// test's stderr, tagged with `context`. The CI-only diagnosability surface:
    /// this is the sole place a failed dispatch can see the workers' own view, so
    /// it dumps unconditionally (a passing run never calls it). Best-effort — a
    /// log we cannot read is reported as such rather than masking the original
    /// failure.
    pub fn dump_diagnostics(&self, context: &str) {
        eprintln!("\n========== distributed lane diagnostics: {context} ==========");
        eprintln!("fleet of {} worker process(es)", self.workers.len());
        for w in &self.workers {
            eprintln!("\n----- worker {} -----", w.worker_id);
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
    /// Best-effort SIGKILL + reap of every child, so a panic in a property test
    /// never leaks a worker process. `kill`/`wait` errors are ignored — a child
    /// that already exited (e.g. one we `kill9`'d earlier) is already gone.
    fn drop(&mut self) {
        for w in &mut self.workers {
            sigkill(&mut w.child);
            let _ = w.child.wait();
        }
    }
}

/// SIGKILL a child by pid. We send the signal explicitly (rather than
/// `Child::kill`, which would also work on Unix) so the call site reads as the
/// deliberate "crash a worker" action the crash-window properties model, and so
/// a child we already killed is a harmless no-op (ESRCH) rather than a panic.
fn sigkill(child: &mut Child) {
    let pid = child.id() as libc::pid_t;
    // SAFETY: `pid` is a child this process spawned; SIGKILL is unconditional and
    // synchronous, touching no allocator or thread state. A racing exit makes the
    // call a no-op (ESRCH), which we intentionally ignore.
    unsafe {
        libc::kill(pid, libc::SIGKILL);
    }
}

/// Spawn one worker process. The worker is configured entirely through a
/// per-process `jammi.toml` (catalog, storage, training timing, the train tier
/// and its distinct ports) plus the `JAMMI_WORKER_ID` seed and the `AWS_*`
/// credentials in its environment. stdout+stderr are redirected to a per-worker
/// log under its scratch dir so a CI failure can surface the worker's view.
fn spawn_worker(
    exe: &Path,
    backends: &Backends,
    result_root: &str,
    worker_id: &str,
    index: usize,
) -> WorkerProc {
    let scratch = TempDir::new().expect("worker scratch dir");
    let artifact_dir = scratch.path().join("artifacts");
    std::fs::create_dir_all(&artifact_dir).expect("worker artifact_dir");

    // Distinct ports per worker so N servers coexist on one host. The flight
    // (gRPC) port is the worker's wire surface; the health port serves /readyz.
    let flight_port = 50100 + index;
    let health_port = 50200 + index;

    let config_path = scratch.path().join("jammi.toml");
    let config_toml = worker_toml(
        result_root,
        &backends.pg_url,
        &backends.s3_endpoint,
        &backends.region,
        artifact_dir.to_str().expect("utf8 artifact_dir"),
        flight_port,
        health_port,
    );
    std::fs::write(&config_path, &config_toml).expect("write worker config");

    let log_path = scratch.path().join("worker.log");
    let log = std::fs::File::create(&log_path).expect("worker log file");
    let log_err = log.try_clone().expect("clone worker log fd");

    let child = Command::new(exe)
        .arg("--config")
        .arg(&config_path)
        .arg("serve")
        .env("JAMMI_WORKER_ID", worker_id)
        // The S3 driver authenticates from these (MinIO creds). The harness
        // session reads the same root with the same creds, so write-on-worker /
        // read-on-harness round-trips over real MinIO.
        .env("AWS_ACCESS_KEY_ID", &backends.access_key_id)
        .env("AWS_SECRET_ACCESS_KEY", &backends.secret_access_key)
        .env("AWS_REGION", &backends.region)
        // An audit master key is required for the engine's audit sign path; a
        // fixed test key keeps every worker's signer consistent.
        .env("JAMMI_AUDIT_MASTER_KEY", "distributed-lane-test-key")
        .stdout(log)
        .stderr(log_err)
        // Own process group so a stray signal to the harness never propagates to
        // a worker we mean to kill deterministically (and vice versa).
        .process_group(0)
        .spawn()
        .unwrap_or_else(|e| panic!("spawn `jammi serve` worker {worker_id}: {e}"));

    WorkerProc {
        worker_id: worker_id.to_string(),
        child,
        config_toml,
        log_path,
        _scratch: scratch,
    }
}

/// Render a worker's `jammi.toml`. The S3 secrets are deliberately absent — they
/// arrive as `AWS_*` env on the child — so the rendered file carries no
/// credential. `allow_http` lets the S3 driver talk plain HTTP to MinIO.
fn worker_toml(
    result_root: &str,
    pg_url: &str,
    s3_endpoint: &str,
    region: &str,
    artifact_dir: &str,
    flight_port: usize,
    health_port: usize,
) -> String {
    let allow_http = s3_endpoint.starts_with("http://");
    format!(
        r#"
artifact_dir = "{artifact_dir}"

[gpu]
device = -1

[catalog]
kind = "postgres"
url = "{pg_url}"
pool_size = 8

[storage]
result_root = "{result_root}"

[storage.cloud]
kind = "s3"
region = "{region}"
endpoint = "{s3_endpoint}"
allow_http = {allow_http}

[training]
lease_duration_secs = {LEASE_SECS}
heartbeat_interval_secs = {HEARTBEAT_SECS}
idle_poll_secs = {IDLE_POLL_SECS}

[server]
# Distinct per-worker ports so N servers coexist on one host.
flight_listen = "127.0.0.1:{flight_port}"
health_listen = "127.0.0.1:{health_port}"
# Mount core + the train tier so the worker claims and runs submitted jobs.
services = ["train"]
"#
    )
}

/// Resolve the `jammi` server binary the lane spawns. The test binary lives in
/// `{target}/{profile}/deps/`, and `cargo build -p jammi-cli` places `jammi`
/// alongside it in `{target}/{profile}/`. We walk up from the running test exe
/// to that sibling — robust to a custom `CARGO_TARGET_DIR` (the lane sets one)
/// without depending on `CARGO_BIN_EXE_*`, which Cargo only sets for binaries in
/// the *same* package as the test.
fn jammi_serve_binary() -> PathBuf {
    let test_exe = std::env::current_exe().expect("current_exe for binary resolution");
    // .../{profile}/deps/distributed-<hash>  →  .../{profile}/jammi
    let profile_dir = test_exe
        .parent() // deps/
        .and_then(Path::parent) // {profile}/
        .expect("test exe under {profile}/deps/");
    let bin = profile_dir.join(if cfg!(windows) { "jammi.exe" } else { "jammi" });
    assert!(
        bin.is_file(),
        "`jammi` server binary not found at {}. The distributed lane requires it: \
         `cargo build -p jammi-cli` before running this harness.",
        bin.display()
    );
    bin
}

/// The generous terminal-state timeout: spawned workers must boot (process
/// start + Postgres connect + migrate + tier mount), poll, claim, run a tiny
/// CPU LoRA fine-tune, publish to MinIO, and finalize — all under a 3 s lease
/// with reclaim on a crash. 120 s comfortably covers a cold CI runner while
/// still failing fast on a genuinely stuck fleet.
pub const TERMINAL_TIMEOUT: Duration = Duration::from_secs(120);

/// Tight poll interval for [`await_job`] — 250 ms keeps the harness responsive
/// to a terminal write (and to an early worker exit) without hammering the
/// shared catalog.
pub const POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Poll the shared catalog for `job_id` until `want(&record)` holds, returning
/// the matching record. This is the lane's single observation primitive, and it
/// fails *loudly and diagnosably* — the discipline a CI-only lane needs:
///
/// - **Early worker exit:** each tick checks the fleet for a worker that exited
///   on its own (a config rejection, a panic, a `SchemeNotEnabled` publish
///   failure, a clean `0` — anything other than a deliberate `kill9`). The
///   instant one is seen, the helper dumps full fleet diagnostics and panics
///   naming that worker — never silently waiting out the 120 s timeout on a
///   fleet that is already dead.
/// - **Timeout:** if `want` never holds within [`TERMINAL_TIMEOUT`], the helper
///   dumps full fleet diagnostics (every worker's effective config + captured
///   stdout/stderr) and the job's FINAL catalog row (status / claimed_by /
///   attempts / output_model_id / error_message), then panics with `label`.
///
/// `label` describes the awaited condition (e.g. `"job reaches completed"`) so
/// the panic message and the diagnostics header read as one story.
pub async fn await_job(
    fleet: &mut Fleet,
    session: &Arc<InferenceSession>,
    job_id: &str,
    tenant: Option<jammi_db::TenantId>,
    label: &str,
    mut want: impl FnMut(&jammi_db::catalog::training_repo::TrainingJobRecord) -> bool,
) -> jammi_db::catalog::training_repo::TrainingJobRecord {
    let catalog = session.catalog().pinned_to_tenant(tenant);
    let deadline = Instant::now() + TERMINAL_TIMEOUT;
    loop {
        if let Ok(record) = catalog.get_training_job(job_id).await {
            if want(&record) {
                return record;
            }
        }

        // A worker that died unprompted will never satisfy `want`; fail now with
        // its log rather than wait out the full terminal timeout.
        if let Some((worker, status)) = fleet.first_unexpected_exit() {
            fleet.dump_diagnostics(&format!(
                "worker {worker} exited unexpectedly ({status}) while awaiting: {label}"
            ));
            panic!(
                "distributed lane: worker {worker} exited unexpectedly ({status}) before \
                 the fleet could satisfy: {label}. See the dumped worker config + log above."
            );
        }

        if Instant::now() >= deadline {
            fleet.dump_diagnostics(&format!(
                "timed out after {TERMINAL_TIMEOUT:?} awaiting: {label}"
            ));
            dump_final_job_row(session, job_id, tenant, label).await;
            panic!(
                "distributed lane: timed out after {TERMINAL_TIMEOUT:?} awaiting: {label}. \
                 See the dumped worker configs/logs and final job row above."
            );
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

/// Dump the job's final catalog row to the test's stderr — the submitter's view
/// of where the job got stuck (e.g. `status="failed"` with an
/// `error_message` naming a `SchemeNotEnabled` publish failure). Pairs with
/// [`Fleet::dump_diagnostics`] (the workers' view) to make a failed dispatch
/// fully diagnosable from the CI log alone.
async fn dump_final_job_row(
    session: &Arc<InferenceSession>,
    job_id: &str,
    tenant: Option<jammi_db::TenantId>,
    label: &str,
) {
    eprintln!("\n----- final catalog row for job {job_id} (awaiting: {label}) -----");
    match session
        .catalog()
        .pinned_to_tenant(tenant)
        .get_training_job(job_id)
        .await
    {
        Ok(r) => eprintln!(
            "status={:?} claimed_by={:?} attempts={} output_model_id={:?} \
             tenant_id={:?} lease_expires_at={:?} error_message={:?}",
            r.status,
            r.claimed_by,
            r.attempts,
            r.output_model_id,
            r.tenant_id,
            r.lease_expires_at,
            r.error_message,
        ),
        Err(e) => eprintln!("<job row unreadable: {e}>"),
    }
}

/// Path to the lane's standard text fine-tune fixture (`training_pairs.csv`,
/// columns `text_a,text_b,score`) — a generic, consumer-free synthetic triplet
/// table shared with the `it` fine-tune tests.
pub fn training_pairs_url() -> String {
    jammi_test_utils::fixture_url("training_pairs.csv")
}

/// `local:`-prefixed path to the hermetic `tiny_bert` encoder fixture — the same
/// consumer-free 32-dim model the `it` fine-tune tests train, so the worker
/// loads real weights with no network.
pub fn tiny_bert_model() -> String {
    format!(
        "local:{}",
        jammi_test_utils::cookbook_fixture("tiny_bert")
            .to_str()
            .expect("utf8 tiny_bert path")
    )
}

/// A source name unique to this test run, carrying the test name for
/// log-archaeology and a fresh UUID for collision-freedom. The lane runs every
/// property sequentially against ONE shared, persistent Postgres catalog whose
/// `sources` table keys on a global `source_id` PRIMARY KEY, so a fixed name
/// would collide with a prior test's still-present row (`Source already
/// registered`). A per-test-unique name is the catalog-namespace isolation that
/// lets sequential tests share the persistent catalog without ever colliding.
///
/// `role` distinguishes multiple sources within one test (e.g. the two tenants'
/// private sources in cross-tenant isolation); a test with a single source
/// passes its own name.
pub fn unique_source_name(role: &str) -> String {
    format!("{role}-{}", uuid::Uuid::new_v4().simple())
}

/// Register the `training_pairs.csv` triplet source on `session` under `name`.
/// Source registration is catalog state, so a source the harness registers is
/// visible to the spawned workers when they reconstruct the job's loader from
/// its persisted spec. `name` must be unique to this test run (see
/// [`unique_source_name`]) so a registration never collides with a prior test's
/// row on the shared persistent catalog.
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

/// How long the submitted fine-tune should run, which sets the crash window the
/// property needs. The job's *duration* is the load-bearing variable: a property
/// that crashes the claimer mid-run needs the run to reliably outlast the
/// harness's detect-then-kill window; a property that only watches for a clean
/// terminal state wants the fastest valid run.
#[derive(Clone, Copy)]
pub enum JobSize {
    /// A minimal valid run — completes as fast as a real `tiny_bert` LoRA fit
    /// allows. For properties that race only on the *claim* / terminal state
    /// (exactly-one-claim, cross-tenant isolation), where run length is noise.
    Quick,
    /// A deliberately longer run (more epochs) so the job is reliably still
    /// `running` when the harness detects the claimer and SIGKILLs it — the
    /// crash must land *inside* the run, not after it finished. Still completes
    /// well inside [`TERMINAL_TIMEOUT`] once reclaimed.
    Crashable,
}

impl JobSize {
    /// Epoch count for this size. `Crashable` runs many epochs so the wall-clock
    /// run spans seconds (comfortably longer than the sub-second detect+kill
    /// window); `Quick` runs few so the lane's claim/terminal assertions resolve
    /// fast. Both are real fits over the tiny model — neither is a no-op.
    fn epochs(self) -> usize {
        match self {
            JobSize::Quick => 3,
            JobSize::Crashable => 60,
        }
    }
}

/// A real LoRA fine-tune config over `tiny_bert` and the triplet source, sized
/// per [`JobSize`]. The `text_a,text_b,score` columns match `training_pairs.csv`.
fn lane_fine_tune_config(size: JobSize) -> FineTuneConfig {
    FineTuneConfig {
        epochs: size.epochs(),
        batch_size: 8,
        lora_rank: 4,
        warmup_steps: 0,
        ..Default::default()
    }
}

/// Submit one durable LoRA fine-tune job over `source` on `session`, returning
/// its `(job_id, output_model_id)`. The submit only writes a `queued` row — no
/// worker runs on the harness session — so the spawned fleet does the claiming.
/// The job's spec is fully self-describing (source + columns), so any worker can
/// reconstruct and run it from the catalog alone. `size` sets the run length the
/// calling property needs (see [`JobSize`]).
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
