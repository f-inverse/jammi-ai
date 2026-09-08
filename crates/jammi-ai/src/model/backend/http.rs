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

        let dim = response
            .data
            .first()
            .map(|d| d.embedding.len())
            .unwrap_or(0);
        let n = response.data.len();
        // `BackendOutput`'s row-major invariant (see its doc): output head 0
        // is ONE flattened `[n, dim]` buffer, never one `Vec` per row. Build
        // through `BackendOutput::single_head` so a row whose embedding width
        // disagrees with the first row's fails the CONSTRUCTOR, rather than
        // silently misaligning every later row's slice.
        let mut flat = Vec::with_capacity(n * dim);
        for d in &response.data {
            flat.extend_from_slice(&d.embedding);
        }
        BackendOutput::single_head(flat, n, dim, vec![true; n], vec![String::new(); n])
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

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}
