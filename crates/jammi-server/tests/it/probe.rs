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
use std::net::TcpListener as StdTcpListener;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
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
    // Resolve a free ephemeral port, then drop the listener without ever
    // serving on it — nothing is listening there, so a probe against it
    // must fail via a transport (connection) error, not a status check.
    let std_listener = StdTcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
    let addr = std_listener.local_addr().expect("local addr");
    drop(std_listener);
    let url = format!("http://{addr}/readyz");

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

#[tokio::test]
async fn subprocess_probe_exits_zero_on_200() {
    let addr = spawn_readyz(axum::http::StatusCode::OK, "ready").await;
    let output = Command::new(BIN)
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
    let output = Command::new(BIN)
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
    let std_listener = StdTcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
    let addr = std_listener.local_addr().expect("local addr");
    drop(std_listener);

    let output = Command::new(BIN)
        .args(["probe", "--url", &format!("http://{addr}/readyz")])
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

    let output = Command::new(BIN)
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

    let output = Command::new(BIN)
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
