//! HTTP backend for OpenAI-compatible embedding endpoints.
//!
//! Routes embedding tasks to `POST /v1/embeddings`.

use std::time::Duration;

use jammi_db::error::{JammiError, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::inference::adapter::BackendOutput;
use crate::model::ModelTask;

/// HTTP backend that forwards inference to an OpenAI-compatible endpoint.
pub struct HttpBackend {
    client: Client,
}

impl HttpBackend {
    /// Create a new HTTP backend with the given request timeout.
    pub fn new(timeout: Duration) -> Result<Self> {
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| JammiError::Backend(format!("Failed to build HTTP client: {e}")))?;
        Ok(Self { client })
    }

    /// Forward inference to the remote endpoint.
    ///
    /// Only embedding tasks are supported via `POST {base_url}/v1/embeddings`.
    pub async fn forward(
        &self,
        base_url: &str,
        inputs: &[String],
        model_id: &str,
        task: ModelTask,
    ) -> Result<BackendOutput> {
        match task {
            ModelTask::TextEmbedding => self.forward_embeddings(base_url, inputs, model_id).await,
            other => Err(JammiError::Backend(format!(
                "HTTP backend only supports embedding task, got {other}"
            ))),
        }
    }

    async fn forward_embeddings(
        &self,
        base_url: &str,
        inputs: &[String],
        model_id: &str,
    ) -> Result<BackendOutput> {
        let url = format!("{}/v1/embeddings", base_url.trim_end_matches('/'));
        let body = EmbeddingRequest {
            input: inputs.to_vec(),
            model: model_id.to_string(),
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| JammiError::Backend(format!("HTTP embedding request failed: {e}")))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(JammiError::Backend(format!(
                "HTTP embedding request returned {status}: {body}"
            )));
        }

        let response: EmbeddingResponse = resp
            .json()
            .await
            .map_err(|e| JammiError::Backend(format!("Failed to parse embedding response: {e}")))?;

        if response.data.len() != inputs.len() {
            return Err(JammiError::Backend(format!(
                "HTTP embedding response returned {} vector(s), expected one per input ({})",
                response.data.len(),
                inputs.len()
            )));
        }
        let n = response.data.len();
        if n == 0 {
            return Err(JammiError::Backend(
                "HTTP embedding request needs at least one input".into(),
            ));
        }
        let dim = response.data[0].embedding.len();
        // `BackendOutput`'s row-major invariant (see its doc): output head 0
        // is ONE flattened `[n, dim]` buffer, never one `Vec` per row.
        // `single_head` enforces the AGGREGATE count (`flat.len() == n *
        // dim`) at construction; that alone would still accept a response
        // whose individual rows are ragged but happen to sum to `n * dim`
        // (e.g. row 0 short by one value, a later row long by one) — such a
        // response would splice a later row's tail into an earlier row's
        // slice.
        //
        // `row_widths::validate` must run to completion BEFORE any buffer is
        // allocated. A ragged response where row 0 is wide and every later
        // row is empty (e.g. n = 10^6, dim = 10^6) would otherwise size
        // `Vec::with_capacity(n * dim)` off row 0 alone, requesting an
        // arbitrarily large allocation before row 1 -- the row that proves
        // the response is ragged -- is ever inspected. This is enforced at
        // the TYPE level, not by call-site discipline: `row_widths::flatten`
        // is the only way to build the buffer, and it takes a
        // `row_widths::Validated`, whose fields are private to the
        // `row_widths` submodule -- `forward_embeddings` (this function,
        // outside that submodule) cannot construct one by field literal, only
        // by calling `row_widths::validate`. A reorder that tried to flatten
        // before validating is a compile error here, not a runtime property
        // a test has to police.
        let validated = row_widths::validate(&response.data, dim)?;
        let flat = row_widths::flatten(validated, n)?;
        BackendOutput::single_head(flat, n, dim, vec![true; n], vec![String::new(); n])
    }
}

/// Row-width validation as a type: constructing a [`row_widths::Validated`]
/// is the only way to obtain the proof [`row_widths::flatten`] requires, and
/// [`row_widths::validate`] is the only function in the crate that can build
/// one -- its fields are private to this submodule, so `http`'s own code
/// (including `forward_embeddings`) cannot fabricate one by field literal.
mod row_widths {
    use super::EmbeddingData;
    use jammi_db::error::{JammiError, Result};

    /// Proof that every row in the wrapped slice has already been checked
    /// against a common width by [`validate`] -- the only function that can
    /// construct one, since its fields are private to this module.
    /// [`flatten`] requires this type rather than a bare `&[EmbeddingData]`
    /// so building the flat buffer without first validating row widths is a
    /// compile error, not a call-order convention.
    #[derive(Debug)]
    pub(super) struct Validated<'a> {
        data: &'a [EmbeddingData],
        dim: usize,
    }

    /// Validate that every row in `data` has exactly `dim` elements (row 0's
    /// own width), refusing by row index and both widths on the first
    /// disagreement.
    ///
    /// Split out as its own allocation-free pass so callers can validate
    /// BEFORE sizing a `rows * dim` buffer off `dim` alone: a ragged
    /// response (row 0 wide, later rows short, empty, or long) must be
    /// refused on the row that disagrees, never after an allocation request
    /// sized by an unvalidated `dim`. Returns a [`Validated`] rather than
    /// `()` so [`flatten`] can require proof of this check at the type
    /// level.
    pub(super) fn validate(data: &[EmbeddingData], dim: usize) -> Result<Validated<'_>> {
        for (i, d) in data.iter().enumerate() {
            if d.embedding.len() != dim {
                return Err(JammiError::Backend(format!(
                    "HTTP embedding response row {i} has width {}, expected {dim} (row 0's width)",
                    d.embedding.len()
                )));
            }
        }
        Ok(Validated { data, dim })
    }

    /// Flatten already-validated rows into one row-major `[n, dim]` buffer.
    ///
    /// Takes a [`Validated`], which only [`validate`] can construct, so this
    /// can never run ahead of validation -- there is no `&[EmbeddingData]`
    /// overload to reorder into. `n` is the row count `dim` is checked
    /// against (`response.data.len()`, validated separately against
    /// `inputs.len()` by the caller); the two are threaded separately
    /// because `Validated` proves per-row width agreement, not the row count
    /// itself.
    pub(super) fn flatten(validated: Validated<'_>, n: usize) -> Result<Vec<f32>> {
        let dim = validated.dim;
        let capacity = n.checked_mul(dim).ok_or_else(|| {
            JammiError::Backend(format!(
                "HTTP embedding response: rows*dim overflows (rows={n}, dim={dim})"
            ))
        })?;
        let mut flat = Vec::with_capacity(capacity);
        for d in validated.data {
            flat.extend_from_slice(&d.embedding);
        }
        Ok(flat)
    }
}

// ─── Request/Response types (OpenAI-compatible) ──────────────────────────────

#[derive(Serialize)]
struct EmbeddingRequest {
    input: Vec<String>,
    model: String,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ragged response where row 0 is wide (dim 1_000_000) and every
    /// subsequent row is empty implies an aggregate `n * dim` of 10^12
    /// elements (~4 TB of `f32`) if `dim` were trusted past row 0.
    /// `row_widths::validate` is a pure, allocation-free pass: it must
    /// refuse on row 1 -- the first row that disagrees with row 0's width
    /// -- without ever sizing a buffer off the unvalidated `dim`.
    ///
    /// Verified by deleting the width-mismatch check inside
    /// `row_widths::validate` (having it return `Ok(..)` unconditionally):
    /// this test's `unwrap_err()` then panics, since row 1 no longer
    /// produces an `Err`. (This test calls `row_widths::validate` directly,
    /// so a mutation confined to `forward_embeddings` -- e.g. how it uses
    /// the result -- cannot be observed here; see the separate
    /// `forward_embeddings_*` test below for that boundary.)
    #[test]
    fn validate_row_widths_refuses_a_ragged_row_before_any_large_allocation() {
        let dim = 1_000_000;
        let mut data = vec![EmbeddingData {
            embedding: vec![0.0_f32; dim],
        }];
        data.extend((1..1_000_000).map(|_| EmbeddingData {
            embedding: Vec::new(),
        }));

        let err = row_widths::validate(&data, dim).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("row 1"), "must name the offending row: {msg}");
        assert!(
            msg.contains("width 0") && msg.contains("expected 1000000"),
            "must name both the row's own width (0) and row 0's width (1000000): {msg}"
        );
    }

    #[test]
    fn validate_row_widths_accepts_a_uniform_response() {
        let data = vec![
            EmbeddingData {
                embedding: vec![0.1, 0.2],
            },
            EmbeddingData {
                embedding: vec![0.3, 0.4],
            },
        ];
        assert!(row_widths::validate(&data, 2).is_ok());
    }

    /// End-to-end presence oracle: a SMALL ragged response (row 0 width 2,
    /// row 1 width 1 -- deliberately too small to ever panic or exhaust
    /// memory on either code path) driven through `forward_embeddings`
    /// itself -- `HttpBackend`'s own private method, reached here via a
    /// wiremock-backed HTTP round trip -- rather than by calling
    /// `row_widths::validate` directly. This proves the check actually fires
    /// on a real response body decoded off the wire, not only in
    /// `row_widths::validate`'s own unit tests above.
    ///
    /// `row_widths::validate`'s row/width message (`"row {i} has width {},
    /// expected {dim}"`) is the ONLY place in `forward_embeddings` that can
    /// produce that exact text, so asserting it on `forward_embeddings`'s own
    /// `Err` proves validation ran and its result reached the caller
    /// unchanged — no output is produced either way, since this is an
    /// error path.
    ///
    /// Ordering (validate-before-flatten) is a separate property from
    /// presence, and it is NOT this test's oracle: it is enforced at the
    /// type level -- `row_widths::flatten` requires a `row_widths::Validated`,
    /// whose fields are private to the `row_widths` submodule, so nothing
    /// outside it (including `forward_embeddings`) can fabricate one without
    /// calling `row_widths::validate`. There is no reachable reordering to
    /// red this test against (a reorder attempt is a compile error, not a
    /// runtime behavior a test can observe).
    /// Verified by deleting the `row_widths::validate(...)?` line in
    /// `forward_embeddings` and building `flat` directly from
    /// `response.data` in its place (bypassing `row_widths` entirely, since
    /// there is no other way to obtain a buffer without going through it):
    /// this test goes RED. `single_head` then refuses on the AGGREGATE
    /// mismatch instead ("flat buffer has 3 value(s), expected rows*dim
    /// (2*2)"), which names neither "row 1" nor "width 1".
    #[tokio::test]
    async fn forward_embeddings_refuses_a_ragged_response_naming_row_and_widths() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/embeddings"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "data": [
                        {"embedding": [0.1, 0.2]},
                        {"embedding": [0.3]},
                    ]
                })),
            )
            .mount(&server)
            .await;

        let backend = HttpBackend::new(Duration::from_secs(5)).unwrap();
        let inputs = vec!["a".to_string(), "b".to_string()];
        let err = backend
            .forward_embeddings(&server.uri(), &inputs, "model")
            .await
            .expect_err("a ragged response must never resolve to a BackendOutput");
        let msg = err.to_string();
        assert!(msg.contains("row 1"), "must name the offending row: {msg}");
        assert!(
            msg.contains("width 1") && msg.contains("expected 2"),
            "must name both the row's own width (1) and row 0's width (2): {msg}"
        );
    }
}
