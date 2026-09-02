//! A hermetic `jammi-server` subprocess the CLI integration tests run against.
//!
//! The `jammi` CLI is a strict gRPC client, so every CLI integration test needs
//! a live server to talk to. [`TestServer::spawn`] boots a `jammi-server`
//! subprocess with the default in-process backends (SQLite catalog + in-memory
//! broker) pointed at a per-test `TempDir`, on ephemeral ports, and waits for
//! its `/readyz` probe before returning. Dropping the server SIGKILLs the child
//! (RAII), so a failed assertion never leaks a server process.
//!
//! # The server owns its catalog exclusively
//!
//! The SQLite catalog is single-process, and since the `unix-excl` seam
//! (`jammi_db::catalog::backend_sqlite`, `docs/guide/src/catalog-and-broker.md`)
//! that contract is a *mechanism*, not a convention: for as long as any
//! connection in one process holds `<dir>/catalog.db`, that process owns an
//! exclusive lock over the file, and a second process opening it waits out the
//! 5 s busy timeout and is then refused with a typed `SQLITE_BUSY`-class
//! `BackendError::Unavailable`. So a test that needs catalog state the CLI
//! itself has no write path for seeds it BEFORE the server exists, via
//! [`TestServer::spawn_with_scratch`]; a test that needs to read the catalog
//! directly stops the server first (drop the [`TestServer`]). While the server
//! is live, the only supported way in or out is its public surface — the CLI
//! binary or the gRPC client.
//!
//! # Handing the catalog over is an awaited event, not an observed file
//!
//! Dropping the seeding `jammi_db::catalog::Catalog` does not hand the
//! directory over: `sqlx` closes a returned connection from a background task,
//! so the exclusive lock outlives the drop by an unbounded interval. The
//! release point is `Catalog::close(self).await`, which closes the pool,
//! drains it, and waits on SQLite's own evidence of release. **That await is
//! the handoff** — after it returns, the directory is the server's to take, and
//! nothing here needs to look at the filesystem to find that out.
//!
//! An earlier version of this harness polled `catalog.db-wal` / `catalog.db-shm`
//! for exactly that purpose. Under the seam that barrier is unsound in both
//! directions, as the `jammi-db` owner measured:
//!
//! - `catalog.db-shm` **never exists at all** — `unix-excl` keeps the wal-index
//!   in heap memory and opens no `-shm`, so its absence proves nothing.
//! - `catalog.db-wal` can legitimately **survive a completed release**: SQLite
//!   deletes it only when the last close's PASSIVE checkpoint completes, and
//!   that checkpoint may decline (observed on 1 run in 40 of this module). A
//!   `-wal` poll can therefore wait for a deletion that is never coming.
//!
//! And the failure a premature poll caused is no longer a silent race: it is
//! now a hard typed refusal that fails the server's startup after five seconds.
//! [`TestServer::try_spawn_with_scratch`] surfaces exactly that shape, and the
//! `train` module drives it as a negative control.
//!
//! The server binary is resolved by walking up from the running test executable
//! to its sibling under `{target}/{profile}/jammi-server` — the same trick the
//! distributed harness uses, robust to a custom `CARGO_TARGET_DIR` without
//! depending on `CARGO_BIN_EXE_*` (which Cargo only sets for binaries in the
//! test's own package).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use assert_cmd::Command as AssertCommand;
use tempfile::TempDir;

/// How long to wait for the child's `jammi-server listening …` banner. Only
/// reached when the child neither announces nor exits; a child that dies
/// during startup closes its stdout and is detected immediately.
const BANNER_TIMEOUT: Duration = Duration::from_secs(30);

/// How long to wait for `/readyz` to return 200 once the ports are known.
const READY_TIMEOUT: Duration = Duration::from_secs(30);

/// A running `jammi-server` the CLI tests target over `--target`.
pub struct TestServer {
    child: Child,
    flight_port: u16,
    _scratch: TempDir,
    // The background threads draining the child's stdout/stderr. Held so they
    // live as long as the server; each exits on its own when the child is
    // killed (EOF) on drop.
    _drains: Drains,
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
    /// The catalog is single-process and the contract is enforced, so seeding
    /// happens strictly BEFORE the server exists: open a
    /// `jammi_db::catalog::Catalog` on the directory, insert the fixture rows,
    /// then `Catalog::close(self).await` — awaiting that close IS the handoff
    /// (see the module doc) — and only then hand the directory here. Opening a
    /// second catalog on this directory while the returned handle is live is
    /// out of contract: read through the CLI or the gRPC client instead, or
    /// drop the handle first.
    ///
    /// # Panics
    ///
    /// Panics, quoting the child's captured stdout and stderr, if the server
    /// never becomes ready — which is what a caller who skipped the close will
    /// see, since the child is then refused the catalog and exits. Use
    /// [`TestServer::try_spawn_with_scratch`] to assert on that failure instead
    /// of aborting on it.
    pub fn spawn_with_scratch(scratch: TempDir) -> Self {
        match Self::try_spawn_with_scratch(scratch) {
            Ok(server) => server,
            Err(failure) => panic!("{failure}"),
        }
    }

    /// [`TestServer::spawn_with_scratch`] as a `Result`: the startup failure is
    /// a value, carrying the child's merged stdout/stderr and how it ended, so
    /// a test can assert on WHY the server did not come up.
    ///
    /// `Err` distinguishes the two shapes a startup failure can take, which is
    /// the whole point of returning it rather than panicking:
    /// [`ServerStartupFailure::exit`] is `Some(status)` when the child exited
    /// on its own (the refusal shape: a typed error, logged, then a non-zero
    /// exit) and `None` when it was still running at the deadline and this
    /// harness killed it (a hang or a retry loop — never an acceptable
    /// response to out-of-contract input).
    pub fn try_spawn_with_scratch(scratch: TempDir) -> Result<Self, ServerStartupFailure> {
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
            .stderr(Stdio::piped())
            .spawn()
            .unwrap_or_else(|e| panic!("spawn jammi-server at {}: {e}", exe.display()));

        // Read the child's stdout for its startup banner (the ACTUAL bound
        // ports), and keep draining both pipes so a full pipe never blocks the
        // child's own logging. Every line of both streams is also accumulated,
        // so a startup failure can be explained by what the child actually
        // said: the server logs through `tracing` to STDOUT and prints only
        // config-load failures on stderr, so a diagnosis that read one stream
        // would miss half the failures. The banner is a fixed line the server
        // prints before serving:
        // `jammi-server listening flight=<addr> health=<addr>`.
        let logs = Arc::new(Mutex::new(String::new()));
        let stdout = child.stdout.take().expect("child stdout is piped");
        let stderr = child.stderr.take().expect("child stderr is piped");
        let (tx, rx) = mpsc::channel::<(u16, u16)>();
        let mut tx = Some(tx);
        let drains = Drains {
            stdout: drain_pipe(stdout, "[stdout] ", Arc::clone(&logs), move |line| {
                if let Some(sender) = tx.as_ref() {
                    if let Some(ports) = parse_listening_banner(line) {
                        let _ = sender.send(ports);
                        tx = None; // reported once; keep draining stdout
                    }
                }
            }),
            stderr: drain_pipe(stderr, "[stderr] ", Arc::clone(&logs), |_| {}),
        };

        let (flight_port, health_port) = match rx.recv_timeout(BANNER_TIMEOUT) {
            Ok(ports) => ports,
            // The sender lives in the stdout drain, so a disconnect means the
            // child closed stdout without ever announcing — it is exiting.
            Err(RecvTimeoutError::Disconnected) => {
                return Err(ServerStartupFailure::collect(
                    child, true, drains, &logs, scratch,
                ));
            }
            Err(RecvTimeoutError::Timeout) => {
                return Err(ServerStartupFailure::collect(
                    child, false, drains, &logs, scratch,
                ));
            }
        };

        if !wait_ready(health_port, READY_TIMEOUT) {
            return Err(ServerStartupFailure::collect(
                child, false, drains, &logs, scratch,
            ));
        }

        Ok(Self {
            child,
            flight_port,
            _scratch: scratch,
            _drains: drains,
        })
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
}

impl Drop for TestServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The two threads draining a child's pipes, held together so a startup
/// failure can join both before reading the log they wrote into.
struct Drains {
    stdout: thread::JoinHandle<()>,
    stderr: thread::JoinHandle<()>,
}

impl Drains {
    /// Wait for both drains to finish. They finish on EOF, i.e. once the child
    /// has exited (or been killed) and its pipes are closed — so joining is
    /// what makes the accumulated log complete rather than truncated.
    fn join(self) {
        let _ = self.stdout.join();
        let _ = self.stderr.join();
    }
}

/// A `jammi-server` subprocess that never reached `/readyz`, as a value.
///
/// Returned by [`TestServer::try_spawn_with_scratch`]. Holds the child's
/// merged stdout/stderr and how the child ended, and keeps the scratch
/// directory alive for the lifetime of the failure so the assertions can still
/// look at it.
pub struct ServerStartupFailure {
    /// `Some` if the child exited on its own; `None` if it was still running
    /// at the deadline and had to be killed.
    exit: Option<ExitStatus>,
    logs: String,
    _scratch: TempDir,
}

impl ServerStartupFailure {
    /// Everything the child wrote to stdout and stderr before it ended, in
    /// arrival order, each line tagged with its stream.
    pub fn logs(&self) -> &str {
        &self.logs
    }

    /// How the child ended: `Some(status)` when it exited on its own,
    /// `None` when it was still running at the deadline and this harness
    /// killed it.
    pub fn exit(&self) -> Option<ExitStatus> {
        self.exit
    }

    /// Reap the child, join the drains so the log is complete, and package it.
    ///
    /// `exited_on_own` is `true` only when the caller already has proof the
    /// child is on its way out (its stdout hit EOF); then this blocks on
    /// `wait` for the real status. Otherwise the child is still running as far
    /// as anyone knows: its status is re-checked once, and if it is genuinely
    /// still alive it is killed and `exit` stays `None` — a killed child's
    /// status would say nothing about how the server responded.
    fn collect(
        mut child: Child,
        exited_on_own: bool,
        drains: Drains,
        logs: &Arc<Mutex<String>>,
        scratch: TempDir,
    ) -> Self {
        let exit = if exited_on_own {
            child.wait().ok()
        } else {
            let observed = child.try_wait().ok().flatten();
            if observed.is_none() {
                let _ = child.kill();
                let _ = child.wait();
            }
            observed
        };
        drains.join();
        let logs = logs.lock().map(|g| g.clone()).unwrap_or_default();
        Self {
            exit,
            logs,
            _scratch: scratch,
        }
    }
}

impl std::fmt::Display for ServerStartupFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.exit {
            Some(status) => write!(
                f,
                "jammi-server exited during startup ({status}) without becoming ready"
            )?,
            None => write!(
                f,
                "jammi-server never became ready and was still running at the deadline \
                 (killed by the harness)"
            )?,
        }
        write!(f, "; child output:\n{}", self.logs)
    }
}

impl std::fmt::Debug for ServerStartupFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

/// Drain one of the child's pipes on a background thread: hand each line to
/// `on_line`, then append it to `sink` tagged with `tag`. Returns on EOF or
/// read error, which is what makes the thread joinable once the child is gone.
fn drain_pipe<R: Read + Send + 'static>(
    pipe: R,
    tag: &'static str,
    sink: Arc<Mutex<String>>,
    mut on_line: impl FnMut(&str) + Send + 'static,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut reader = BufReader::new(pipe);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => break, // EOF (child exited) or read error
                Ok(_) => {
                    on_line(&line);
                    if let Ok(mut buf) = sink.lock() {
                        buf.push_str(tag);
                        buf.push_str(&line);
                        if !line.ends_with('\n') {
                            buf.push('\n');
                        }
                    }
                }
            }
        }
    })
}

/// Poll `/readyz` on `health_port` until it returns 200, or `timeout` expires.
fn wait_ready(health_port: u16, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if http_ok(health_port, "/readyz") {
            return true;
        }
        thread::sleep(Duration::from_millis(100));
    }
    false
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
