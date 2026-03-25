use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use crate::common;

// ─── Health, error schema, source CRUD ───────────────────────────────────────
//
// One test app exercises: health returns 200, error responses have both
// "error" and "code" fields, and source CRUD lifecycle (create/list/delete).

#[tokio::test]
async fn health_and_error_schema_and_source_crud() {
    let dir = tempfile::tempdir().unwrap();
    let state = common::test_app_state(dir.path()).await;
    let app = jammi_server::build_router(Arc::clone(&state));

    // --- Health returns 200 with "status" field ---
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

    // --- 404 fallback has "error" and "code" fields ---
    let resp = app
        .clone()
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
    assert!(
        body["error"].is_string(),
        "Error response must have 'error' string field"
    );
    assert!(
        body["code"].is_string(),
        "Error response must have 'code' string field"
    );
    assert!(!body["error"].as_str().unwrap().is_empty());
    assert!(!body["code"].as_str().unwrap().is_empty());

    // --- Source CRUD: POST creates, GET lists, DELETE removes ---
    // POST /sources → 201
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/sources")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "source_id": "test_src",
                        "source_type": "local",
                        "url": "file:///nonexistent.parquet",
                        "format": "parquet"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    // Local source with nonexistent path may fail at table creation.
    // The important thing is the server doesn't panic.
    let status = resp.status();
    assert!(
        status == StatusCode::CREATED || status == StatusCode::INTERNAL_SERVER_ERROR,
        "POST /sources should return 201 or a structured error, got {status}"
    );

    // GET /sources → 200 with "sources" array
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/sources")
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
    assert!(
        body["sources"].is_array(),
        "GET /sources should return sources array"
    );

    // DELETE /sources/:id → 204 (or 500 if source didn't register due to missing file)
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/sources/test_src")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        resp.status() == StatusCode::NO_CONTENT
            || resp.status() == StatusCode::INTERNAL_SERVER_ERROR,
        "DELETE /sources/:id should return 204 or structured error, got {}",
        resp.status()
    );

    // After deletion, GET /sources should not contain 'test_src'.
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/sources")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(
        &axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    let sources = body["sources"].as_array().unwrap();
    assert!(
        !sources
            .iter()
            .any(|s| s["source_id"].as_str() == Some("test_src")),
        "Deleted source should not appear in list"
    );

    // --- Malformed JSON → structured error ---
    let app = jammi_server::build_router(Arc::clone(&state));
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/sources")
                .header("content-type", "application/json")
                .body(Body::from("not json at all"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        resp.status().is_client_error(),
        "Malformed JSON should return 4xx, got {}",
        resp.status()
    );

    // Missing required fields
    let app = jammi_server::build_router(Arc::clone(&state));
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/sources")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"source_type": "local"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        resp.status().is_client_error(),
        "Missing source_id should return 4xx, got {}",
        resp.status()
    );

    // Invalid enum value
    let app = jammi_server::build_router(Arc::clone(&state));
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/sources")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"source_id": "x", "source_type": "invalid_type"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        resp.status().is_client_error(),
        "Invalid source_type should return 4xx, got {}",
        resp.status()
    );
}

// ─── Config validation ───────────────────────────────────────────────────────

#[test]
fn server_config_validation() {
    use jammi_engine::config::ServerConfig;

    // Valid default
    assert!(ServerConfig::default().validate().is_ok());

    // Invalid listen address
    let bad = ServerConfig {
        listen: "not-an-address".into(),
        ..Default::default()
    };
    let err = bad.validate().unwrap_err().to_string();
    assert!(
        err.contains("listen"),
        "Error should mention 'listen': {err}"
    );

    // Same address for HTTP and Flight
    let same = ServerConfig {
        listen: "0.0.0.0:8080".into(),
        flight_listen: "0.0.0.0:8080".into(),
        ..Default::default()
    };
    assert!(
        same.validate().is_err(),
        "Same listen and flight_listen should fail"
    );
}

// ─── Concurrent health checks ────────────────────────────────────────────────

#[tokio::test]
async fn concurrent_health_checks() {
    let dir = tempfile::tempdir().unwrap();
    let state = common::test_app_state(dir.path()).await;

    let mut handles = Vec::new();
    for _ in 0..20 {
        let app = jammi_server::build_router(Arc::clone(&state));
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

// ─── CP5b UAT 7: Graceful shutdown ──────────────────────────────────────────

#[tokio::test]
async fn graceful_shutdown_completes_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    let state = common::test_app_state(dir.path()).await;

    // Bind to a random port.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener); // Release the port so serve_with_shutdown can bind it.

    // Oneshot channel acts as our shutdown trigger.
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let shutdown = async move {
        let _ = rx.await;
    };

    // Start the server in a background task.
    let server_handle =
        tokio::spawn(async move { jammi_server::serve_with_shutdown(state, addr, shutdown).await });

    // Give the server a moment to start listening.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Confirm the server is up by hitting /health.
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .expect("Health request should succeed while server is running");
    assert_eq!(resp.status(), 200);

    // Trigger shutdown.
    tx.send(()).expect("Shutdown signal should send");

    // Server should exit cleanly.
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), server_handle)
        .await
        .expect("Server should shut down within 5 seconds")
        .expect("Server task should not panic");
    assert!(result.is_ok(), "Server should exit with Ok(())");
}

// ─── Concurrent source mutations ─────────────────────────────────────────────

#[tokio::test]
async fn concurrent_source_mutations_are_safe() {
    let dir = tempfile::tempdir().unwrap();
    let state = common::test_app_state(dir.path()).await;

    // Bind to random port
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);

    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let shutdown = async move {
        let _ = rx.await;
    };

    let server_state = Arc::clone(&state);
    let server_handle = tokio::spawn(async move {
        jammi_server::serve_with_shutdown(server_state, addr, shutdown).await
    });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // 10 concurrent POST /sources requests with unique source_ids
    let client = reqwest::Client::new();
    let mut handles = Vec::new();
    for i in 0..10 {
        let c = client.clone();
        let url = format!("http://{addr}/sources");
        handles.push(tokio::spawn(async move {
            c.post(&url)
                .json(&serde_json::json!({
                    "source_id": format!("src_{i}"),
                    "source_type": "local",
                    "url": format!("file:///nonexistent_{i}.parquet"),
                    "format": "parquet"
                }))
                .send()
                .await
        }));
    }

    // All requests must complete (no deadlocks, no panics)
    for h in handles {
        let resp = h.await.unwrap().unwrap();
        let status = resp.status().as_u16();
        assert!(
            status == 201 || (400..=599).contains(&status),
            "Response should be 201 or structured error, got {status}"
        );
    }

    tx.send(()).unwrap();
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), server_handle)
        .await
        .unwrap()
        .unwrap();
    assert!(result.is_ok());
}
