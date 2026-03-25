use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

// ─── Health and 404 fallback ────────────────────────────────────────────────

#[tokio::test]
async fn health_and_fallback() {
    let app = jammi_server::build_router();

    // Health returns 200 with "status" field
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(body["status"], "ok");

    // 404 fallback has "error" and "code" fields
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/nonexistent")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let body: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert!(body["error"].is_string());
    assert!(body["code"].is_string());
}

// ─── Config validation ──────────────────────────────────────────────────────

#[test]
fn server_config_validation() {
    use jammi_engine::config::ServerConfig;

    // Valid default
    assert!(ServerConfig::default().validate().is_ok());

    // Invalid health_listen address
    let bad = ServerConfig {
        health_listen: "not-an-address".into(),
        ..Default::default()
    };
    let err = bad.validate().unwrap_err().to_string();
    assert!(
        err.contains("health_listen"),
        "Error should mention 'health_listen': {err}"
    );

    // Same address for health and Flight
    let same = ServerConfig {
        health_listen: "0.0.0.0:8080".into(),
        flight_listen: "0.0.0.0:8080".into(),
        ..Default::default()
    };
    assert!(
        same.validate().is_err(),
        "Same health_listen and flight_listen should fail"
    );
}

// ─── Concurrent health checks ───────────────────────────────────────────────

#[tokio::test]
async fn concurrent_health_checks() {
    let mut handles = Vec::new();
    for _ in 0..20 {
        let app = jammi_server::build_router();
        handles.push(tokio::spawn(async move {
            let resp = app
                .oneshot(
                    Request::builder()
                        .uri("/health")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            resp.status()
        }));
    }

    for h in handles {
        assert_eq!(h.await.unwrap(), StatusCode::OK);
    }
}

// ─── Graceful shutdown ──────────────────────────────────────────────────────

#[tokio::test]
async fn graceful_shutdown_completes_cleanly() {
    // Bind to a random port.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);

    // Oneshot channel acts as our shutdown trigger.
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let shutdown = async move {
        let _ = rx.await;
    };

    let server_handle =
        tokio::spawn(async move { jammi_server::serve_with_shutdown(addr, shutdown).await });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Confirm the server is up.
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .expect("Health request should succeed while server is running");
    assert_eq!(resp.status(), 200);

    // Trigger shutdown.
    tx.send(()).expect("Shutdown signal should send");

    let result = tokio::time::timeout(std::time::Duration::from_secs(5), server_handle)
        .await
        .expect("Server should shut down within 5 seconds")
        .expect("Server task should not panic");
    assert!(result.is_ok(), "Server should exit with Ok(())");
}
