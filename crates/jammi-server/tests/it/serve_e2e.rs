//! End-to-end test for `OssServer::run` — boot the full binary in
//! process with a temporary SQLite config, run a Flight SQL roundtrip
//! against `:flight_listen`, hit `/healthz` on `:health_listen`, then
//! drive a graceful shutdown.

use std::time::Duration;

use arrow_flight::sql::client::FlightSqlServiceClient;
use futures::TryStreamExt;
use jammi_db::config::JammiConfig;
use jammi_server::runtime::OssServer;
use jammi_test_utils::test_config;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::transport::Channel;

/// Acquire two distinct loopback ports for health + flight by binding
/// `127.0.0.1:0` twice and immediately dropping the listeners. The
/// kernel issues unique ephemeral ports, so the bound addresses are
/// safe to hand to the OSS server seconds later — the race window is
/// the same one every other multi-listener integration test relies on.
async fn pick_two_ports() -> (String, String) {
    let l1 = TcpListener::bind("127.0.0.1:0").await.expect("bind 1");
    let l2 = TcpListener::bind("127.0.0.1:0").await.expect("bind 2");
    let a1 = l1.local_addr().expect("addr 1");
    let a2 = l2.local_addr().expect("addr 2");
    drop(l1);
    drop(l2);
    (a1.to_string(), a2.to_string())
}

fn test_oss_config(artifact_dir: &std::path::Path, health: &str, flight: &str) -> JammiConfig {
    let mut cfg = test_config(artifact_dir);
    cfg.server.health_listen = health.to_string();
    cfg.server.flight_listen = flight.to_string();
    cfg
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn oss_server_serves_healthz_and_flight_sql_round_trip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (health_addr, flight_addr) = pick_two_ports().await;
    let cfg = test_oss_config(dir.path(), &health_addr, &flight_addr);

    let server = OssServer::new(cfg).await.expect("OssServer::new");
    let server_health_addr = server.health_addr();
    let server_flight_addr = server.flight_addr();

    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let server_task = tokio::spawn(async move {
        server
            .run_with_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
    });

    // Wait for both listeners to bind. The HTTP side-channel and the
    // Tonic chain bind in parallel; 200ms is comfortably above the
    // measured worst-case bind latency on CI hardware.
    tokio::time::sleep(Duration::from_millis(200)).await;

    // /healthz round-trip.
    let healthz = reqwest::get(format!("http://{server_health_addr}/healthz"))
        .await
        .expect("healthz GET")
        .json::<serde_json::Value>()
        .await
        .expect("healthz json");
    assert_eq!(healthz["status"], "ok");

    // /readyz round-trip — the catalog probe runs against the live
    // SQLite catalog the test config provisioned.
    let readyz = reqwest::get(format!("http://{server_health_addr}/readyz"))
        .await
        .expect("readyz GET");
    assert_eq!(
        readyz.status(),
        200,
        "readyz must be 200 against live SQLite"
    );

    // /metrics returns the Prometheus text snapshot.
    let metrics_text = reqwest::get(format!("http://{server_health_addr}/metrics"))
        .await
        .expect("metrics GET")
        .text()
        .await
        .expect("metrics body");
    assert!(
        metrics_text.contains("jammi_grpc_requests_total"),
        "metrics body must enumerate the OSS counters; got:\n{metrics_text}"
    );

    // Flight SQL round-trip — connect to the gRPC chain and run a
    // trivial SELECT.
    let channel = Channel::from_shared(format!("http://{server_flight_addr}"))
        .expect("channel uri")
        .connect()
        .await
        .expect("connect");
    let mut client = FlightSqlServiceClient::new(channel);
    let info = client
        .execute("SELECT 1 AS one".to_string(), None)
        .await
        .expect("flight execute");
    let endpoint = info
        .endpoint
        .first()
        .cloned()
        .expect("flight endpoint must be present");
    let ticket = endpoint.ticket.expect("endpoint ticket");
    let mut stream = client.do_get(ticket).await.expect("do_get");
    let mut total_rows: usize = 0;
    while let Some(batch) = stream.try_next().await.expect("flight stream") {
        total_rows += batch.num_rows();
    }
    assert!(
        total_rows > 0,
        "flight SQL roundtrip must return at least one row"
    );

    // Trigger graceful shutdown and wait for the server task to drain.
    shutdown_tx.send(()).expect("shutdown send");
    let result = tokio::time::timeout(Duration::from_secs(5), server_task)
        .await
        .expect("server shutdown within 5s")
        .expect("server task panic");
    assert!(result.is_ok(), "server exit status: {result:?}");
}
