use std::time::Duration;

use jammi_ai::model::backend::http::HttpBackend;
use jammi_ai::model::ModelTask;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ─── HTTP backend: embedding and error handling ─────────────────────────────

#[tokio::test]
async fn http_backend_embedding_and_errors() {
    let server = MockServer::start().await;

    // Mock /v1/embeddings → returns 2 embeddings of dim 3
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [
                { "embedding": [0.1, 0.2, 0.3] },
                { "embedding": [0.4, 0.5, 0.6] }
            ]
        })))
        .mount(&server)
        .await;

    let backend = HttpBackend::new(Duration::from_secs(5)).unwrap();
    let base_url = server.uri();

    // --- Embedding request: 2 inputs → 2 vectors of dim 3 ---
    let result = backend
        .forward(
            &base_url,
            &["hello".into(), "world".into()],
            "test-model",
            ModelTask::Embedding,
        )
        .await
        .unwrap();

    assert_eq!(
        result.float_outputs.len(),
        2,
        "Should have 2 embedding vectors"
    );
    assert_eq!(
        result.float_outputs[0].len(),
        3,
        "Each vector should have dim 3"
    );
    assert_eq!(result.shapes[0], (2, 3));
    assert!(
        result.row_status.iter().all(|&s| s),
        "All rows should succeed"
    );

    // --- Non-embedding task returns error ---
    let result = backend
        .forward(
            &base_url,
            &["test".into()],
            "test-model",
            ModelTask::Classification,
        )
        .await;
    match result {
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("only supports embedding"),
                "Error should explain the limitation: {msg}"
            );
        }
        Ok(_) => panic!("Non-embedding tasks should return an error"),
    }

    // --- Server error: 500 → graceful error ---
    let error_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
        .mount(&error_server)
        .await;

    let result = backend
        .forward(
            &error_server.uri(),
            &["test".into()],
            "test-model",
            ModelTask::Embedding,
        )
        .await;
    match result {
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("500"),
                "Error should mention status code: {msg}"
            );
        }
        Ok(_) => panic!("500 response should return an error"),
    }
}
