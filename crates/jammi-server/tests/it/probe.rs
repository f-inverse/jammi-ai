//! `jammi_server::probe` against a real listener.
//!
//! Drives the probe's HTTP client against an actual `/readyz` endpoint —
//! `lib.rs`'s [`jammi_server::build_health_router`] mounted on a live
//! loopback socket, so the in-process legs prove the wire round trip
//! (status code, body) rather than a stubbed transport. The subprocess
//! legs below go one step further and cross the actual process boundary,
//! proving `main.rs`'s `ExitCode`/stderr mapping — the in-process `check`
//! calls above only prove `jammi_server::probe::check`'s `Result`, never
//! the exit code a supervisor actually observes.

use std::io::Write;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use axum::body::{Body, Bytes};
use axum::http::{header, StatusCode};
use axum::routing::get;
use axum::Router;
use jammi_server::routes::health::MetricsRegistry;
use jammi_server::runtime::{ReadinessCheck, ReadinessProbe};
use tokio::process::Command;

/// Flips between `Ok` and `Err` so one listener can serve both the 200 and
/// the 503 leg of the oracle.
struct Toggle(Arc<AtomicBool>);

#[async_trait]
impl ReadinessCheck for Toggle {
    async fn check(&self) -> Result<(), String> {
        if self.0.load(Ordering::SeqCst) {
            Ok(())
        } else {
            Err("simulated catalog outage".into())
        }
    }
}

const TIMEOUT: Duration = Duration::from_secs(5);

/// Path to the built `jammi-server` binary — available to a crate's own
/// integration tests when the crate has a `[[bin]]` target.
const BIN: &str = env!("CARGO_BIN_EXE_jammi-server");

/// A loopback address nothing in this suite — or any non-root process on
/// either CI platform this suite runs on (GitHub Actions' Linux and macOS
/// runners are both non-root) — can ever be listening on: binding TCP port 1
/// requires root / `CAP_NET_BIND_SERVICE`, so a connection attempt here
/// refuses deterministically.
///
/// This replaces a bind-`127.0.0.1:0`-then-drop pattern, which resolves an
/// ephemeral port, releases it, and dials the now-bare address — a window
/// during which `cargo test`'s default multi-threaded harness (every
/// `#[tokio::test]` runs concurrently) can let a DIFFERENT test's listener
/// claim that exact just-freed port before this test's probe connects,
/// turning "connection refused" into a real (and wrong) response. Of the two
/// options for closing that window — shrink it by closing the listener
/// immediately before probing, or dial an address nothing can ever bind —
/// this is the latter and the less racy: the first only narrows the race,
/// it does not remove it.
const UNREACHABLE_ADDR: &str = "127.0.0.1:1";

/// Build a `jammi-server probe` command with a hermetic environment: only
/// `PATH` is carried through from the harness process, everything else
/// (including any ambient `JAMMI_AUDIT_MASTER_KEY`, `JAMMI_CONFIG`, or other
/// `JAMMI_*` override already set in the CI runner's or a developer's own
/// shell) is cleared. Every subprocess leg in this file resolves its target
/// from an explicit `--url` or an explicit, already-existing `--config`
/// path (see `crates/jammi-db/src/config/mod.rs`'s
/// `resolve_config_path_in`, which returns an explicit path immediately
/// when it exists, never consulting `JAMMI_CONFIG` or a platform config
/// directory) — so none of them need `HOME` or `TMPDIR` in the child's
/// environment on either macOS or Linux; a future subprocess test that
/// relies on default (no `--config`, no `--url`) resolution would need to
/// add them back.
fn probe_command() -> Command {
    let mut cmd = Command::new(BIN);
    cmd.env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default());
    cmd
}

#[tokio::test]
async fn check_is_ok_only_on_200() {
    let ready = Arc::new(AtomicBool::new(true));
    let readiness = Arc::new(ReadinessProbe::new(Arc::new(Toggle(ready.clone()))));
    let metrics = Arc::new(MetricsRegistry::new().expect("metrics registry"));
    let app = jammi_server::build_health_router(readiness, metrics);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("axum::serve");
    });
    let url = format!("http://{addr}/readyz");

    // 200 leg.
    jammi_server::probe::check(&url, TIMEOUT)
        .await
        .expect("a ready /readyz must check Ok");

    // 503 leg, same listener.
    ready.store(false, Ordering::SeqCst);
    let err = jammi_server::probe::check(&url, TIMEOUT)
        .await
        .expect_err("a not-ready /readyz must check Err");
    assert!(
        err.contains("503") || err.to_lowercase().contains("service unavailable"),
        "failure message should carry the response status, got: {err}"
    );
}

#[tokio::test]
async fn check_fails_with_a_connection_error_on_an_unreachable_port() {
    // Dial a reserved port nothing can be listening on (see
    // UNREACHABLE_ADDR's doc comment) — deterministically refused, no
    // bind-then-drop race with a concurrent test thread.
    let url = format!("http://{UNREACHABLE_ADDR}/readyz");

    let err = jammi_server::probe::check(&url, TIMEOUT)
        .await
        .expect_err("an unreachable port must check Err");
    assert!(
        !err.is_empty(),
        "connection failure must carry a non-empty error message"
    );
}

#[tokio::test]
async fn check_does_not_follow_a_redirect() {
    // A `/readyz` that answers 302 with a Location to a healthy endpoint
    // must still fail the probe — a 3xx is a failure like any other
    // non-200 status, never something the client follows on the probe's
    // behalf.
    let addr = spawn_redirecting_readyz().await;
    let url = format!("http://{addr}/readyz");

    let err = jammi_server::probe::check(&url, TIMEOUT)
        .await
        .expect_err("a redirecting /readyz must check Err, never be followed");
    assert!(
        err.contains("302") || err.to_lowercase().contains("found"),
        "failure message should carry the redirect status, got: {err}"
    );
}

/// Spawn a bare readyz-only listener on an ephemeral loopback port,
/// returning its address and a background task handle. The task is
/// dropped (and the listener with it) when the caller is done.
async fn spawn_readyz(status: StatusCode, body: &'static str) -> std::net::SocketAddr {
    let app = Router::new().route("/readyz", get(move || async move { (status, body) }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("axum::serve");
    });
    addr
}

/// Spawn a listener whose `/readyz` answers exactly `302 Found` with a
/// `Location` pointing at a `/final` route that itself answers `200` — the
/// divergence-prone shape a redirect-following client would silently pass
/// through as ready.
async fn spawn_redirecting_readyz() -> std::net::SocketAddr {
    let app = Router::new()
        .route(
            "/readyz",
            get(|| async { (StatusCode::FOUND, [(header::LOCATION, "/final")]) }),
        )
        .route("/final", get(|| async { (StatusCode::OK, "ok") }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("axum::serve");
    });
    addr
}

/// Spawn a listener whose `/readyz` answers `503` with a body streamed as
/// `chunk_count` chunks of `chunk`, one `delay` apart — far larger in total
/// than [`jammi_server::probe::MAX_BODY_BYTES`] and, thanks to `delay`, far
/// slower to fully drain than the cap should ever require waiting for.
async fn spawn_slow_oversized_readyz(
    chunk: Bytes,
    chunk_count: usize,
    delay: Duration,
) -> std::net::SocketAddr {
    let app = Router::new().route(
        "/readyz",
        get(move || {
            let chunk = chunk.clone();
            async move {
                let stream = futures::stream::unfold(0usize, move |sent| {
                    let chunk = chunk.clone();
                    async move {
                        if sent >= chunk_count {
                            None
                        } else {
                            tokio::time::sleep(delay).await;
                            Some((Ok::<_, std::io::Error>(chunk.clone()), sent + 1))
                        }
                    }
                });
                (StatusCode::SERVICE_UNAVAILABLE, Body::from_stream(stream))
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("axum::serve");
    });
    addr
}

/// The body cap bounds RESIDENT MEMORY, not just the echoed string: a
/// misbehaving `/readyz` streaming a body far larger than
/// [`jammi_server::probe::MAX_BODY_BYTES`], slowly (one small chunk every
/// 20ms — fully draining the configured 4000 chunks would take ~80s), must
/// not make the probe wait for the whole thing. `check` reads via
/// `Response::chunk()` and stops the instant the cap is reached, so this
/// completes in a handful of chunk-intervals, not anywhere near 80s — the
/// timing assertion is what distinguishes "streamed and stopped early" from
/// the pre-fix `Response::bytes()` shape, which buffers the entire body
/// before any cap is applied and would need the full drain to return at
/// all.
#[tokio::test]
async fn probe_caps_a_slow_oversized_body_without_buffering_the_whole_thing() {
    let chunk = Bytes::from(vec![b'x'; 256]);
    // 4000 * 256 B ~= 1 MiB total, far past the 4 KiB cap; 4000 * 20ms = 80s
    // to fully drain — the probe must return in a small fraction of that.
    let addr = spawn_slow_oversized_readyz(chunk, 4000, Duration::from_millis(20)).await;
    let url = format!("http://{addr}/readyz");

    let start = std::time::Instant::now();
    let err = jammi_server::probe::check(&url, Duration::from_secs(60))
        .await
        .expect_err("a 503 must check Err");
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_secs(5),
        "the probe must stop reading once the cap is reached rather than wait for the full \
         (deliberately slow, oversized) body; elapsed {elapsed:?}, expected well under the \
         ~80s a full drain would take"
    );
    assert!(
        err.contains("[truncated]"),
        "a capped body must carry the truncation marker, got: {err}"
    );
    let echoed_x_count = err.matches('x').count();
    assert!(
        echoed_x_count <= jammi_server::probe::MAX_BODY_BYTES,
        "the echoed body must never exceed the cap, got {echoed_x_count} bytes of body, cap is \
         {}",
        jammi_server::probe::MAX_BODY_BYTES
    );
}

/// A large but immediately-available (non-streamed) failure body is still
/// truncated to the cap with the marker appended — the ordinary case the
/// slow/streamed oracle above complements.
#[tokio::test]
async fn check_truncates_a_large_failure_body() {
    let body = "x".repeat(jammi_server::probe::MAX_BODY_BYTES + 500);
    let addr = spawn_readyz(
        StatusCode::SERVICE_UNAVAILABLE,
        Box::leak(body.into_boxed_str()),
    )
    .await;
    let url = format!("http://{addr}/readyz");

    let err = jammi_server::probe::check(&url, TIMEOUT)
        .await
        .expect_err("a 503 must check Err");
    assert!(
        err.contains("[truncated]"),
        "an oversized body must be truncated with the marker, got: {err}"
    );
    let echoed_x_count = err.matches('x').count();
    assert!(
        echoed_x_count <= jammi_server::probe::MAX_BODY_BYTES,
        "echoed body must never exceed the cap, got {echoed_x_count}"
    );
}

#[tokio::test]
async fn subprocess_probe_exits_zero_on_200() {
    let addr = spawn_readyz(axum::http::StatusCode::OK, "ready").await;
    let output = probe_command()
        .args(["probe", "--url", &format!("http://{addr}/readyz")])
        .output()
        .await
        .expect("run jammi-server probe");
    assert!(
        output.status.success(),
        "expected exit 0, got {:?}, stderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

#[tokio::test]
async fn subprocess_probe_exits_one_on_503_with_status_on_stderr() {
    let addr = spawn_readyz(axum::http::StatusCode::SERVICE_UNAVAILABLE, "not ready yet").await;
    let output = probe_command()
        .args(["probe", "--url", &format!("http://{addr}/readyz")])
        .output()
        .await
        .expect("run jammi-server probe");
    assert_eq!(output.status.code(), Some(1), "expected exit 1");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("503") || stderr.to_lowercase().contains("service unavailable"),
        "stderr should carry the status, got: {stderr}"
    );
}

#[tokio::test]
async fn subprocess_probe_exits_one_on_connection_refused_with_error_text_on_stderr() {
    // See UNREACHABLE_ADDR's doc comment: a reserved port, not a
    // bind-then-drop ephemeral one, so this is race-free under a
    // multi-threaded harness.
    let output = probe_command()
        .args([
            "probe",
            "--url",
            &format!("http://{UNREACHABLE_ADDR}/readyz"),
        ])
        .output()
        .await
        .expect("run jammi-server probe");
    assert_eq!(output.status.code(), Some(1), "expected exit 1");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.trim().is_empty(),
        "connection refusal must print an error to stderr"
    );
}

#[tokio::test]
async fn subprocess_probe_exits_one_on_a_redirect() {
    let addr = spawn_redirecting_readyz().await;

    let output = probe_command()
        .args(["probe", "--url", &format!("http://{addr}/readyz")])
        .output()
        .await
        .expect("run jammi-server probe");
    assert_eq!(
        output.status.code(),
        Some(1),
        "a 302 -> 200 endpoint must not be followed into an exit-0 probe"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("302") || stderr.to_lowercase().contains("found"),
        "stderr should carry the redirect status, got: {stderr}"
    );
}

#[tokio::test]
async fn subprocess_probe_with_config_targets_the_configured_health_listen() {
    let addr = spawn_readyz(axum::http::StatusCode::OK, "ready").await;

    let mut config_file = tempfile::NamedTempFile::new().expect("create temp config");
    writeln!(config_file, "[server]").expect("write config");
    writeln!(config_file, "health_listen = \"{addr}\"").expect("write config");
    config_file.flush().expect("flush config");

    let output = probe_command()
        .args([
            "probe",
            "--config",
            config_file.path().to_str().expect("utf8 temp path"),
        ])
        .stdin(Stdio::null())
        .output()
        .await
        .expect("run jammi-server probe --config");
    assert!(
        output.status.success(),
        "probe --config must resolve health_listen from the file and exit 0, got {:?}, stderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}
