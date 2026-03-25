use std::time::Duration;

use jammi_ai::model::backend::http::HttpBackend;
use jammi_ai::model::backend::vllm::VllmBackend;
use jammi_ai::model::{BackendType, ModelTask};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ─── HTTP backend: embedding, chat, error handling ───────────────────────────
//
// One wiremock server exercises: embedding request → vector output,
// chat request → text output, server error → graceful failure,
// and correct endpoint routing (/v1/chat/completions, not /v1/completions).

#[tokio::test]
async fn http_backend_embedding_and_chat_and_errors() {
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

    // Mock /v1/chat/completions → returns a summary
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "choices": [{
                "message": { "role": "assistant", "content": "This is a summary." }
            }]
        })))
        .mount(&server)
        .await;

    let backend = HttpBackend::new(Duration::from_secs(5));
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

    // --- Chat request: 1 input → 1 text output ---
    let result = backend
        .forward(
            &base_url,
            &["Summarize this document.".into()],
            "test-model",
            ModelTask::Summarization,
        )
        .await
        .unwrap();

    assert_eq!(result.string_outputs.len(), 1, "Should have 1 output head");
    assert_eq!(result.string_outputs[0].len(), 1, "Should have 1 row");
    assert_eq!(result.string_outputs[0][0], "This is a summary.");
    assert!(result.row_status[0], "Row should succeed");

    // --- Verify /v1/chat/completions is used (not /v1/completions) ---
    // If the backend had used /v1/completions, wiremock would return 404
    // and the result would be an error. The success above proves correct routing.

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

// ─── Backend type enum: serde round-trip, Vllm + Http variants ───────────────

#[test]
fn contract_backend_type_has_vllm_and_http_variants() {
    // Verify serde round-trip for all variants
    let variants = [
        (BackendType::Candle, "\"candle\""),
        (BackendType::Ort, "\"ort\""),
        (BackendType::Vllm, "\"vllm\""),
        (BackendType::Http, "\"http\""),
    ];
    for (variant, expected_json) in variants {
        let json = serde_json::to_string(&variant).unwrap();
        assert_eq!(json, expected_json);
        let roundtrip: BackendType = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip, variant);
    }
}

// ─── vLLM: failure modes ─────────────────────────────────────────────────────

#[test]
fn vllm_check_installed_returns_result() {
    // On CI without vllm, this should return an error (not panic)
    let result = VllmBackend::check_vllm_installed();
    match result {
        Ok(()) => {} // vllm installed — fine
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("vllm") && msg.contains("not found"),
                "Error should mention vllm not found: {msg}"
            );
        }
    }
}

#[tokio::test]
async fn vllm_health_timeout_returns_error() {
    // Start a mock server that returns 503 on /health (simulating a server
    // that never becomes ready)
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&server)
        .await;

    // Extract port from the mock server URI
    let port: u16 = server.uri().rsplit(':').next().unwrap().parse().unwrap();

    // Use a short timeout (3s) for testing
    let result = VllmBackend::wait_for_health(port, "test-model", Duration::from_secs(3)).await;

    assert!(result.is_err(), "Should timeout waiting for health");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("failed to start") || msg.contains("timeout"),
        "Error should mention timeout: {msg}"
    );
}
