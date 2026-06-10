//! A hermetic `jammi-server` subprocess the CLI integration tests run against.
//!
//! The `jammi` CLI is a strict gRPC client, so every CLI integration test needs
//! a live server to talk to. [`TestServer::spawn`] boots a `jammi-server`
//! subprocess with the default in-process backends (SQLite catalog + in-memory
//! broker) pointed at a per-test `TempDir`, on ephemeral ports, and waits for
//! its `/readyz` probe before returning. Dropping the server SIGKILLs the child
//! (RAII), so a failed assertion never leaks a server process.
//!
//! The server binary is resolved by walking up from the running test executable
//! to its sibling under `{target}/{profile}/jammi-server` — the same trick the
//! distributed harness uses, robust to a custom `CARGO_TARGET_DIR` without
//! depending on `CARGO_BIN_EXE_*` (which Cargo only sets for binaries in the
//! test's own package).

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use assert_cmd::Command as AssertCommand;
use tempfile::TempDir;

/// A running `jammi-server` the CLI tests target over `--target`.
pub struct TestServer {
    child: Child,
    flight_port: u16,
    _scratch: TempDir,
}

impl TestServer {
    /// Spawn a server with default backends on ephemeral ports and block until
    /// `/readyz` returns 200 (or panic after a generous timeout). The catalog
    /// and artifact state live under a per-test `TempDir`.
    pub fn spawn() -> Self {
        let scratch = TempDir::new().expect("tempdir for server scratch");
        let flight_port = free_port();
        let health_port = free_port();
        let exe = server_binary();

        let child = Command::new(&exe)
            .env("JAMMI_ARTIFACT_DIR", scratch.path())
            .env(
                "JAMMI_SERVER__FLIGHT_LISTEN",
                format!("127.0.0.1:{flight_port}"),
            )
            .env(
                "JAMMI_SERVER__HEALTH_LISTEN",
                format!("127.0.0.1:{health_port}"),
            )
            // A fixed audit master key keeps the server's audit signer happy
            // without a per-test secret.
            .env("JAMMI_AUDIT_MASTER_KEY", "cli-it-test-key")
            .env_remove("JAMMI_CONFIG")
            .spawn()
            .unwrap_or_else(|e| panic!("spawn jammi-server at {}: {e}", exe.display()));

        let server = Self {
            child,
            flight_port,
            _scratch: scratch,
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

/// Bind a free ephemeral port and release it, returning the number. There is a
/// small race between release and the server's bind, but on a CI host the
/// kernel does not immediately re-hand the same port, so this is reliable for a
/// test harness.
fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().expect("local addr").port()
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
