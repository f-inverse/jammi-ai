//! A hermetic `jammi-server` subprocess the CLI integration tests run against.
//!
//! The `jammi` CLI is a strict gRPC client, so every CLI integration test needs
//! a live server to talk to. [`TestServer::spawn`] boots a `jammi-server`
//! subprocess with the default in-process backends (SQLite catalog + in-memory
//! broker) pointed at a per-test `TempDir`, on ephemeral ports, and waits for
//! its `/readyz` probe before returning. Dropping the server SIGKILLs the child
//! (RAII), so a failed assertion never leaks a server process.
//!
//! **The server owns its catalog exclusively.** The SQLite catalog is
//! single-process (`docs/guide/src/catalog-and-broker.md`): for as long as the
//! subprocess is live it holds `<dir>/catalog.db`, and a second process opening
//! that file is out of contract — the backend refuses it with an
//! `SQLITE_BUSY`-class `BackendError`, and before that refusal existed the
//! sharing silently corrupted the WAL. A test that needs catalog state the CLI
//! itself has no write path for therefore seeds it BEFORE the server exists,
//! via [`TestServer::spawn_with_scratch`]; a test that needs to read the
//! catalog directly stops the server first (drop the [`TestServer`]). While the
//! server is live, the only supported way in or out is its public surface —
//! the CLI binary or the gRPC client.
//!
//! The server binary is resolved by walking up from the running test executable
//! to its sibling under `{target}/{profile}/jammi-server` — the same trick the
//! distributed harness uses, robust to a custom `CARGO_TARGET_DIR` without
//! depending on `CARGO_BIN_EXE_*` (which Cargo only sets for binaries in the
//! test's own package).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use assert_cmd::Command as AssertCommand;
use tempfile::TempDir;

/// A running `jammi-server` the CLI tests target over `--target`.
pub struct TestServer {
    child: Child,
    flight_port: u16,
    _scratch: TempDir,
    // The background thread draining the child's stdout. Held so it lives as long
    // as the server; it exits on its own when the child is killed (EOF) on drop.
    _stdout_drain: thread::JoinHandle<()>,
}

impl TestServer {
    /// Spawn a server with default backends on ephemeral (`:0`) ports and block
    /// until `/readyz` returns 200 (or panic after a generous timeout). The child
    /// binds the ports itself and announces the ACTUAL ports it got on stdout —
    /// the harness reads that banner rather than pre-picking a port and dropping
    /// it (which would open a release-then-rebind race). The catalog and artifact
    /// state live under a fresh per-test `TempDir`.
    ///
    /// Use [`TestServer::spawn_with_scratch`] instead when the test needs
    /// catalog rows seeded before the server comes up.
    pub fn spawn() -> Self {
        Self::spawn_with_scratch(TempDir::new().expect("tempdir for server scratch"))
    }

    /// Like [`TestServer::spawn`], but on a scratch directory the caller has
    /// already prepared — the seam for a test that must seed catalog state the
    /// CLI itself has no write path for (e.g. a training-job fixture;
    /// submission is SDK-only, never the CLI).
    ///
    /// The catalog is single-process, so seeding happens strictly BEFORE the
    /// server exists: open a `jammi_db::catalog::Catalog` on the directory,
    /// insert the fixture rows, drop the catalog and settle the close with
    /// [`await_catalog_released`], then hand the directory here. Opening a
    /// second catalog on this directory while the returned handle is live is
    /// out of contract (see the module doc) — read through the CLI or the gRPC
    /// client instead, or drop the handle first.
    ///
    /// This constructor verifies that precondition rather than trusting it: it
    /// panics up front if this process still holds the catalog, instead of
    /// racing the server onto a file it cannot take exclusively and leaving a
    /// `/readyz` timeout to explain it.
    pub fn spawn_with_scratch(scratch: TempDir) -> Self {
        assert!(
            !catalog_sidecars_present(scratch.path()),
            "the catalog at {} is still open in this process — the server subprocess cannot \
             take it exclusively. Drop the seeding `Catalog` and `await_catalog_released` \
             before spawning.",
            scratch.path().join("catalog.db").display()
        );
        let exe = server_binary();

        let mut child = Command::new(&exe)
            .env("JAMMI_ARTIFACT_DIR", scratch.path())
            .env("JAMMI_SERVER__FLIGHT_LISTEN", "127.0.0.1:0")
            .env("JAMMI_SERVER__HEALTH_LISTEN", "127.0.0.1:0")
            // A fixed audit master key keeps the server's audit signer happy
            // without a per-test secret.
            .env("JAMMI_AUDIT_MASTER_KEY", "cli-it-test-key")
            .env_remove("JAMMI_CONFIG")
            .stdout(Stdio::piped())
            .spawn()
            .unwrap_or_else(|e| panic!("spawn jammi-server at {}: {e}", exe.display()));

        // Read the child's stdout for its startup banner (the ACTUAL bound
        // ports), then keep draining so a full pipe never blocks the child's own
        // logging. The banner is a fixed line the server prints before serving:
        // `jammi-server listening flight=<addr> health=<addr>`.
        let stdout = child.stdout.take().expect("child stdout is piped");
        let (tx, rx) = mpsc::channel::<(u16, u16)>();
        let drain = thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut tx = Some(tx);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) | Err(_) => break, // EOF (child exited) or read error
                    Ok(_) => {
                        if let Some(sender) = tx.as_ref() {
                            if let Some(ports) = parse_listening_banner(&line) {
                                let _ = sender.send(ports);
                                tx = None; // reported once; keep draining stdout
                            }
                        }
                    }
                }
            }
        });

        let (flight_port, health_port) = rx.recv_timeout(Duration::from_secs(30)).expect(
            "jammi-server did not announce its listening ports on stdout within 30s \
             (did it fail to bind?)",
        );

        let server = Self {
            child,
            flight_port,
            _scratch: scratch,
            _stdout_drain: drain,
        };
        server.wait_ready(health_port);
        server
    }

    /// The `--target` URL a CLI invocation should use to reach this server.
    pub fn target(&self) -> String {
        format!("grpc://127.0.0.1:{}", self.flight_port)
    }

    /// A `jammi` CLI command pre-pointed at this server's `--target`.
    pub fn cli(&self) -> AssertCommand {
        let mut cmd = AssertCommand::cargo_bin("jammi").expect("jammi-cli binary built");
        cmd.args(["--target", &self.target()])
            .env_remove("JAMMI_CONFIG");
        cmd
    }

    fn wait_ready(&self, health_port: u16) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline {
            if http_ok(health_port, "/readyz") {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        panic!("jammi-server did not become ready on health port {health_port} within 30s");
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Wait until no connection from THIS process holds `<dir>/catalog.db`, so a
/// [`TestServer`] can then take the file exclusively.
///
/// **Dropping a `Catalog` is not by itself a release.** `sqlx` hands a pooled
/// connection back — and, on a closed pool, shuts it down — from a spawned
/// task, so the close only happens once the runtime is driven again; a drop
/// followed by blocking the runtime thread leaves the file held indefinitely
/// (measured: the `-wal`/`-shm` sidecars survive a 4s blocking wait, and
/// vanish within one 50ms `await`). That is why this is `async` and why it
/// yields between polls rather than sleeping the thread.
///
/// The barrier is observable, not a fixed sleep: when the last connection to a
/// WAL database closes cleanly, SQLite checkpoints into the main file and
/// removes both sidecars. Their absence is the proof.
///
/// A directory with no `catalog.db` at all has nothing to release and returns
/// immediately.
pub async fn await_catalog_released(dir: &Path) {
    if !dir.join("catalog.db").exists() {
        return;
    }
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if !catalog_sidecars_present(dir) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!(
        "catalog at {} still has live WAL sidecars after 30s — this process has not closed \
         its own connections. Drop the seeding `Catalog` (and every clone of its backend \
         `Arc`) first, and make sure the runtime is free to drive the close.",
        dir.join("catalog.db").display()
    );
}

/// Whether `<dir>/catalog.db` has live `-wal`/`-shm` sidecars — i.e. some
/// connection still has the WAL database open. See [`await_catalog_released`].
fn catalog_sidecars_present(dir: &Path) -> bool {
    dir.join("catalog.db-wal").exists() || dir.join("catalog.db-shm").exists()
}

/// Parse the server's fixed startup banner — `jammi-server listening
/// flight=<addr> health=<addr>` — into `(flight_port, health_port)`. Returns
/// `None` for any other line (ordinary log output), so the caller can scan the
/// child's stdout stream and pick out the one banner line.
fn parse_listening_banner(line: &str) -> Option<(u16, u16)> {
    let rest = line.trim().strip_prefix("jammi-server listening ")?;
    let mut flight = None;
    let mut health = None;
    for token in rest.split_whitespace() {
        if let Some(addr) = token.strip_prefix("flight=") {
            flight = addr.rsplit(':').next().and_then(|p| p.parse().ok());
        } else if let Some(addr) = token.strip_prefix("health=") {
            health = addr.rsplit(':').next().and_then(|p| p.parse().ok());
        }
    }
    Some((flight?, health?))
}

/// Issue a minimal HTTP/1.1 GET and report whether the response status line is
/// `200`. Keeping the probe dependency-free avoids pulling an HTTP client into
/// the CLI test crate just to poll `/readyz`.
fn http_ok(port: u16, path: &str) -> bool {
    let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) else {
        return false;
    };
    let req = format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
    if stream.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let mut buf = String::new();
    if stream.read_to_string(&mut buf).is_err() {
        return false;
    }
    buf.lines()
        .next()
        .map(|status| status.contains("200"))
        .unwrap_or(false)
}

/// Resolve the `jammi-server` binary built alongside the test executable.
fn server_binary() -> PathBuf {
    let test_exe = std::env::current_exe().expect("current_exe for binary resolution");
    // .../{profile}/deps/it-<hash>  →  .../{profile}/jammi-server
    let profile_dir = test_exe
        .parent() // deps/
        .and_then(Path::parent) // {profile}/
        .expect("test exe under {profile}/deps/");
    let bin = profile_dir.join(if cfg!(windows) {
        "jammi-server.exe"
    } else {
        "jammi-server"
    });
    assert!(
        bin.is_file(),
        "`jammi-server` binary not found at {}. The CLI integration tests \
         require it: `cargo build -p jammi-server` before running them.",
        bin.display()
    );
    bin
}
