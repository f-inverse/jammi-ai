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
        // `validate_row_widths` runs to completion BEFORE any buffer is
        // allocated. A ragged response where row 0 is wide and every later
        // row is empty (e.g. n = 10^6, dim = 10^6) would otherwise size
        // `Vec::with_capacity(n * dim)` off row 0 alone, requesting an
        // arbitrarily large allocation before row 1 -- the row that proves
        // the response is ragged -- is ever inspected. Validating first
        // means the refusal fires on row 1's width, never on an allocation
        // request.
        validate_row_widths(&response.data, dim)?;
        let capacity = n.checked_mul(dim).ok_or_else(|| {
            JammiError::Backend(format!(
                "HTTP embedding response: rows*dim overflows (rows={n}, dim={dim})"
            ))
        })?;
        let mut flat = Vec::with_capacity(capacity);
        for d in &response.data {
            flat.extend_from_slice(&d.embedding);
        }
        BackendOutput::single_head(flat, n, dim, vec![true; n], vec![String::new(); n])
    }
}

/// Validate that every row in `data` has exactly `dim` elements (row 0's own
/// width), refusing by row index and both widths on the first disagreement.
///
/// Split out as its own allocation-free pass so callers can validate BEFORE
/// sizing a `rows * dim` buffer off `dim` alone: a ragged response (row 0
/// wide, later rows short, empty, or long) must be refused on the row that
/// disagrees, never after an allocation request sized by an unvalidated
/// `dim`.
fn validate_row_widths(data: &[EmbeddingData], dim: usize) -> Result<()> {
    for (i, d) in data.iter().enumerate() {
        if d.embedding.len() != dim {
            return Err(JammiError::Backend(format!(
                "HTTP embedding response row {i} has width {}, expected {dim} (row 0's width)",
                d.embedding.len()
            )));
        }
    }
    Ok(())
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

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ragged response where row 0 is wide (dim 1_000_000) and every
    /// subsequent row is empty implies an aggregate `n * dim` of 10^12
    /// elements (~4 TB of `f32`) if `dim` were trusted past row 0.
    /// `validate_row_widths` is a pure, allocation-free pass: it must
    /// refuse on row 1 -- the first row that disagrees with row 0's width
    /// -- without ever sizing a buffer off the unvalidated `dim`.
    ///
    /// Verified by reverting the split (folding `validate_row_widths` back
    /// into a loop that calls `Vec::with_capacity(n * dim)` before
    /// checking any row, as in the pre-fold code): this test's caller in
    /// `forward_embeddings` would then request a ~4 TB allocation before
    /// row 1 is ever inspected, aborting the process rather than returning
    /// the `Err` asserted here.
    #[test]
    fn validate_row_widths_refuses_a_ragged_row_before_any_large_allocation() {
        let dim = 1_000_000;
        let mut data = vec![EmbeddingData {
            embedding: vec![0.0_f32; dim],
        }];
        data.extend((1..1_000_000).map(|_| EmbeddingData {
            embedding: Vec::new(),
        }));

        let err = validate_row_widths(&data, dim).unwrap_err();
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
        assert!(validate_row_widths(&data, 2).is_ok());
    }
}
