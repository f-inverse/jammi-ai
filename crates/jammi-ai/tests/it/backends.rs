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
            ModelTask::TextEmbedding,
        )
        .await
        .unwrap();

    // `BackendOutput`'s row-major invariant: ONE flattened `[rows, dim]`
    // buffer in `float_outputs[0]`, never one `Vec` per row.
    assert_eq!(
        result.float_outputs.len(),
        1,
        "one flattened output head, not one Vec per row"
    );
    assert_eq!(
        result.float_outputs[0].len(),
        6,
        "2 rows * dim 3, flattened row-major"
    );
    assert_eq!(result.shapes[0], (2, 3));
    assert!(
        result.row_status.iter().all(|&s| s),
        "All rows should succeed"
    );
    assert_eq!(
        result.single_row_or_err(0).unwrap(),
        &[0.1_f32, 0.2, 0.3],
        "row 0 must read back its own embedding, in row-major order"
    );
    assert_eq!(
        result.single_row_or_err(1).unwrap(),
        &[0.4_f32, 0.5, 0.6],
        "row 1 must read back its own embedding, not row 0's"
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
            ModelTask::TextEmbedding,
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

/// A response whose individual rows are ragged but whose TOTAL element count
/// still equals `rows * dim` (row 0's width) must be refused by name, not
/// spliced. Three rows of width 2, 1, 3 sum to 6 == 3 * 2 — a check that only
/// verifies the aggregate sum (`flat.len() == rows * dim`) would accept this
/// and silently splice row 2's first element into row 1's slice.
#[tokio::test]
async fn http_backend_refuses_a_ragged_response_row_width() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [
                { "embedding": [0.1, 0.2] },
                { "embedding": [0.3] },
                { "embedding": [0.4, 0.5, 0.6] }
            ]
        })))
        .mount(&server)
        .await;

    let backend = HttpBackend::new(Duration::from_secs(5)).unwrap();
    let err = backend
        .forward(
            &server.uri(),
            &["a".into(), "b".into(), "c".into()],
            "test-model",
            ModelTask::TextEmbedding,
        )
        .await
        .expect_err(
            "a per-row width that disagrees with row 0's, even when the total sum matches \
             rows*dim, must be a typed refusal, not a spliced row",
        );
    let msg = err.to_string();
    assert!(msg.contains("row 1"), "must name the offending row: {msg}");
    assert!(
        msg.contains('1') && msg.contains('2'),
        "must name both the row's own width (1) and the expected width (2): {msg}"
    );
}

/// A response with FEWER rows than inputs must be a named refusal — the
/// count-mismatch check must fire before any per-row width check runs (there
/// is no row 2 to widen-check against). Verified by temporarily removing the
/// `response.data.len() != inputs.len()` check: this test goes RED (the
/// mismatched response is silently accepted, `dim` reads off a response
/// row that does not correspond to the caller's 3rd input, rather than the
/// `Err` asserted below).
#[tokio::test]
async fn http_backend_refuses_fewer_response_rows_than_inputs() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [
                { "embedding": [0.1, 0.2] }
            ]
        })))
        .mount(&server)
        .await;

    let backend = HttpBackend::new(Duration::from_secs(5)).unwrap();
    let err = backend
        .forward(
            &server.uri(),
            &["a".into(), "b".into(), "c".into()],
            "test-model",
            ModelTask::TextEmbedding,
        )
        .await
        .expect_err("fewer response rows than inputs must be a typed refusal");
    let msg = err.to_string();
    assert!(msg.contains('1') && msg.contains('3'), "got: {msg}");
}

/// The peer of the above: MORE response rows than inputs must be refused the
/// same way, never silently truncated to the caller's input count. Verified
/// by temporarily removing the count-mismatch check: this test goes RED (the
/// extra row is silently accepted instead of the `Err` asserted below).
#[tokio::test]
async fn http_backend_refuses_more_response_rows_than_inputs() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [
                { "embedding": [0.1, 0.2] },
                { "embedding": [0.3, 0.4] },
                { "embedding": [0.5, 0.6] }
            ]
        })))
        .mount(&server)
        .await;

    let backend = HttpBackend::new(Duration::from_secs(5)).unwrap();
    let err = backend
        .forward(
            &server.uri(),
            &["a".into(), "b".into()],
            "test-model",
            ModelTask::TextEmbedding,
        )
        .await
        .expect_err("more response rows than inputs must be a typed refusal");
    let msg = err.to_string();
    assert!(msg.contains('3') && msg.contains('2'), "got: {msg}");
}

/// Advisory #421-frontend-follow-on-4: an empty batch of INPUTS (`&[]`) must
/// send NO request at all and return `Ok` with `BackendOutput`'s shared
/// empty-batch shape `(0, 0)` — the SAME shape `CandleModel::forward_embedding`
/// / `forward_image_embedding` / `forward_audio_embedding` return for
/// `num_rows == 0` (see `crates/jammi-ai/src/inference/adapter/mod.rs`'s
/// `BackendOutput` doc, "The empty-batch shape: `(0, 0)`"), never the old
/// `Err("HTTP embedding request needs at least one input")`. No `Mock` is
/// mounted on `server` here — if `forward` sent a request anyway, wiremock's
/// own unmatched-request panic (or, at minimum, a non-2xx-driven `Err`
/// instead of the `Ok` asserted below) would fail this test; `received_
/// requests()` asserts the same fact directly and by name rather than
/// relying on that panic alone.
#[tokio::test]
async fn http_backend_empty_batch_matches_embedded_shape_and_sends_no_request() {
    let server = MockServer::start().await;

    let backend = HttpBackend::new(Duration::from_secs(5)).unwrap();
    let result = backend
        .forward(&server.uri(), &[], "test-model", ModelTask::TextEmbedding)
        .await
        .expect("an empty input batch must succeed with the shared empty-batch shape");

    assert_eq!(
        result.float_outputs,
        vec![Vec::<f32>::new()],
        "one (empty) float head, matching CandleModel's own empty-batch float_outputs"
    );
    assert!(result.string_outputs.is_empty());
    assert!(result.row_status.is_empty());
    assert!(result.row_errors.is_empty());
    assert_eq!(
        result.shapes,
        vec![(0, 0)],
        "must match CandleModel's own (0, 0) empty-batch shape, not a fabricated dim"
    );

    let requests = server
        .received_requests()
        .await
        .expect("wiremock request recording must be enabled by default");
    assert!(
        requests.is_empty(),
        "an empty input batch must send no request at all, got: {requests:?}"
    );
}

/// A ZERO-WIDTH row 0 (`embedding: []`) carries no real embedding at all.
/// The HTTP layer derives `dim` from row 0's own width and hands the whole
/// buffer to `BackendOutput::single_head`, which refuses a zero-dim head by
/// name — this test drives that refusal end to end through the HTTP path.
/// Verified by temporarily reverting `single_head`'s `dim == 0` check: this
/// test goes RED (`Ok` with a vacuous zero-width embedding instead of the
/// `Err` asserted below).
#[tokio::test]
async fn http_backend_refuses_a_zero_width_row_zero() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [
                { "embedding": Vec::<f32>::new() }
            ]
        })))
        .mount(&server)
        .await;

    let backend = HttpBackend::new(Duration::from_secs(5)).unwrap();
    let err = backend
        .forward(
            &server.uri(),
            &["a".into()],
            "test-model",
            ModelTask::TextEmbedding,
        )
        .await
        .expect_err("a zero-width row 0 must be a typed refusal, never a vacuous embedding");
    assert!(err.to_string().contains("dim"), "got: {err}");
}
