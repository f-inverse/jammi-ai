//! `jammi_server::probe` against a real listener.
//!
//! Drives the probe's HTTP client against an actual `/readyz` endpoint —
//! same router `serve` mounts — over a live loopback socket, so the
//! oracle proves the wire round trip (status code, body) rather than a
//! stubbed transport.

use std::net::TcpListener as StdTcpListener;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use jammi_server::routes::health::MetricsRegistry;
use jammi_server::runtime::{ReadinessCheck, ReadinessProbe};

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

#[tokio::test]
async fn probe_exits_zero_on_200_and_one_on_503() {
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
        .expect("a ready /readyz must probe as success (exit 0)");

    // 503 leg, same listener.
    ready.store(false, Ordering::SeqCst);
    let err = jammi_server::probe::check(&url, TIMEOUT)
        .await
        .expect_err("a not-ready /readyz must probe as failure (exit 1)");
    assert!(
        err.contains("503") || err.to_lowercase().contains("service unavailable"),
        "failure message should carry the response status, got: {err}"
    );
}

#[tokio::test]
async fn probe_fails_with_a_connection_error_on_an_unreachable_port() {
    // Resolve a free ephemeral port, then drop the listener without ever
    // serving on it — nothing is listening there, so a probe against it
    // must fail via a transport (connection) error, not a status check.
    let std_listener = StdTcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
    let addr = std_listener.local_addr().expect("local addr");
    drop(std_listener);
    let url = format!("http://{addr}/readyz");

    let err = jammi_server::probe::check(&url, TIMEOUT)
        .await
        .expect_err("an unreachable port must fail the probe (exit 1)");
    assert!(
        !err.is_empty(),
        "connection failure must carry a non-empty error message"
    );
}
