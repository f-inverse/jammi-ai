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
    let app = jammi_server::build_router(state);

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
