//! The three-process Ballista lane's harness: spawning real `jammi-server`
//! binaries with `[ballista]` roles configured, one hosting the scheduler
//! (+ an executor), two hosting executors only.
//!
//! Shape mirrors `crates/jammi-ai/tests/distributed/harness.rs`'s
//! `Backends`/`Fleet`/`spawn_worker`/`worker_toml`/`jammi_server_binary`/
//! `await_job` (COMMON.md's own instruction: that harness is a private test
//! module of a different crate, so a copy is the honest shape here — named
//! in this crate's contract as a copy, a candidate for the lead's
//! consolidation to lift into `jammi-test-utils`). Reduced to what THIS
//! crate's lane exercises: no fine-tune/gang submission helpers (those stay
//! in `jammi-ai`'s own harness; the GANG unit's gang-through-Ballista
//! oracles are a named follow-up here, see this crate's contract file).

// The helpers below are the follow-up commit's starting point (see the
// module doc): `main.rs`'s one test detects backends and skips before
// calling most of them in THIS pass, which would otherwise be a dead-code
// build failure under `-D warnings`.
#![allow(dead_code)]

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

/// The live backends this lane needs: Postgres (`JAMMI_TEST_PG_URL`) and an
/// S3-compatible object store (`JAMMI_TEST_S3_ENDPOINT` / `_BUCKET`, MinIO
/// in CI/dev — `docker run ... minio/minio`, per this crate's brief).
/// `detect()` returns `None` naming what is missing, rather than silently
/// running a degraded lane.
pub struct Backends {
    pub pg_url: String,
    pub s3_endpoint: String,
    pub s3_bucket: String,
}

impl Backends {
    pub fn detect() -> Result<Self, String> {
        let pg_url = std::env::var("JAMMI_TEST_PG_URL")
            .map_err(|_| "JAMMI_TEST_PG_URL is unset".to_string())?;
        let s3_endpoint = std::env::var("JAMMI_TEST_S3_ENDPOINT")
            .map_err(|_| "JAMMI_TEST_S3_ENDPOINT is unset".to_string())?;
        let s3_bucket = std::env::var("JAMMI_TEST_S3_BUCKET")
            .map_err(|_| "JAMMI_TEST_S3_BUCKET is unset".to_string())?;
        Ok(Self {
            pg_url,
            s3_endpoint,
            s3_bucket,
        })
    }
}

/// Locate the `jammi-server` binary this workspace's `cargo build -p
/// jammi-server --bin jammi-server --features storage-s3` produced, in the
/// SAME `CARGO_TARGET_DIR` this test process itself was built into.
pub fn jammi_server_binary() -> PathBuf {
    let mut path = std::env::current_exe()
        .expect("test binary has a path")
        .to_path_buf();
    // `<target>/<profile>/deps/<test-binary>` -> `<target>/<profile>/jammi-server`
    path.pop(); // deps
    path.pop(); // profile dir stays
    path.push("jammi-server");
    path
}

/// An ephemeral, unused TCP port on localhost — the same "probe, then
/// release" pattern `roles.rs::host_executor` uses for an unset `grpc_bind`
/// port, applied here to allocate NON-colliding ports across three
/// processes before writing each one's config file.
pub fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind an ephemeral port")
        .local_addr()
        .unwrap()
        .port()
}

/// One `jammi-server` process's `[ballista]` role config, rendered to TOML.
pub struct RoleConfig {
    pub scheduler_bind: Option<String>,
    pub executor_scheduler_address: Option<String>,
    pub executor_bind: Option<String>,
    pub executor_grpc_bind: Option<String>,
}

pub fn worker_toml(
    backends: &Backends,
    artifact_dir: &std::path::Path,
    role: &RoleConfig,
) -> String {
    let mut out = format!(
        "artifact_dir = {artifact_dir:?}\n\
         [catalog.postgres]\nurl = {pg_url:?}\n\
         [storage]\nresult_root = \"s3://{bucket}/jammi-ballista-it\"\n\
         [storage.cloud.s3]\nendpoint = {endpoint:?}\n\
         [worker]\nenabled = false\n\
         [ballista]\n",
        artifact_dir = artifact_dir.to_string_lossy(),
        pg_url = backends.pg_url,
        bucket = backends.s3_bucket,
        endpoint = backends.s3_endpoint,
    );
    if let Some(bind) = &role.scheduler_bind {
        out.push_str(&format!("scheduler_bind = {bind:?}\n"));
    }
    if let Some(addr) = &role.executor_scheduler_address {
        out.push_str("[ballista.executor]\n");
        out.push_str(&format!("scheduler_address = {addr:?}\n"));
        out.push_str(&format!(
            "bind = {:?}\n",
            role.executor_bind.as_deref().unwrap_or("127.0.0.1:0")
        ));
        out.push_str(&format!(
            "grpc_bind = {:?}\n",
            role.executor_grpc_bind.as_deref().unwrap_or("127.0.0.1:0")
        ));
        out.push_str("advertise_host = \"127.0.0.1\"\n");
        out.push_str("task_slots = 1\n");
    }
    out
}

/// A spawned `jammi-server` process, killed on drop.
pub struct WorkerProc {
    child: Child,
}

impl Drop for WorkerProc {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Spawn `jammi-server` with `config_path`, `health_listen`/`flight_listen`
/// on fresh ephemeral ports (via `JAMMI_SERVER__*` env, the same override
/// layer `JammiConfig::load_from` reads).
pub fn spawn_worker(config_path: &std::path::Path) -> WorkerProc {
    let child = Command::new(jammi_server_binary())
        .env("JAMMI_CONFIG", config_path)
        .env(
            "JAMMI_SERVER__HEALTH_LISTEN",
            format!("127.0.0.1:{}", free_port()),
        )
        .env(
            "JAMMI_SERVER__FLIGHT_LISTEN",
            format!("127.0.0.1:{}", free_port()),
        )
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn jammi-server");
    WorkerProc { child }
}

/// Poll `predicate` until it is `true` or `timeout` elapses.
pub async fn await_condition(timeout: Duration, mut predicate: impl FnMut() -> bool) -> bool {
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if predicate() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    false
}
