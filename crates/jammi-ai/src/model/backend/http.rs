//! HTTP backend for OpenAI-compatible inference endpoints.
//!
//! Routes embedding tasks to `POST /v1/embeddings` and text generation
//! tasks (summarization, classification, text generation) to
//! `POST /v1/chat/completions`.

use std::time::Duration;

use jammi_engine::error::{JammiError, Result};
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
    pub fn new(timeout: Duration) -> Self {
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .expect("Failed to build HTTP client");
        Self { client }
    }

    /// Forward inference to the remote endpoint.
    ///
    /// - Embedding tasks → `POST {base_url}/v1/embeddings`
    /// - Other tasks → `POST {base_url}/v1/chat/completions`
    pub async fn forward(
        &self,
        base_url: &str,
        inputs: &[String],
        model_id: &str,
        task: ModelTask,
    ) -> Result<BackendOutput> {
        match task {
            ModelTask::Embedding => self.forward_embeddings(base_url, inputs, model_id).await,
            _ => self.forward_chat(base_url, inputs, model_id, task).await,
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
        let float_outputs: Vec<Vec<f32>> = response.data.into_iter().map(|d| d.embedding).collect();
        let n = float_outputs.len();

        Ok(BackendOutput {
            float_outputs,
            string_outputs: Vec::new(),
            row_status: vec![true; n],
            row_errors: vec![String::new(); n],
            shapes: vec![(n, dim)],
        })
    }

    async fn forward_chat(
        &self,
        base_url: &str,
        inputs: &[String],
        model_id: &str,
        _task: ModelTask,
    ) -> Result<BackendOutput> {
        let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
        let mut string_outputs = Vec::new();
        let mut row_status = Vec::new();
        let mut row_errors = Vec::new();

        for input in inputs {
            let body = ChatCompletionRequest {
                model: model_id.to_string(),
                messages: vec![ChatMessage {
                    role: "user".to_string(),
                    content: input.clone(),
                }],
                max_tokens: Some(256),
            };

            match self.client.post(&url).json(&body).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if !status.is_success() {
                        let body = resp.text().await.unwrap_or_default();
                        row_status.push(false);
                        row_errors.push(format!("{status}: {body}"));
                        string_outputs.push(String::new());
                        continue;
                    }
                    match resp.json::<ChatCompletionResponse>().await {
                        Ok(chat_resp) => {
                            let text = chat_resp
                                .choices
                                .into_iter()
                                .next()
                                .map(|c| c.message.content)
                                .unwrap_or_default();
                            string_outputs.push(text);
                            row_status.push(true);
                            row_errors.push(String::new());
                        }
                        Err(e) => {
                            row_status.push(false);
                            row_errors.push(format!("Parse error: {e}"));
                            string_outputs.push(String::new());
                        }
                    }
                }
                Err(e) => {
                    row_status.push(false);
                    row_errors.push(format!("Request error: {e}"));
                    string_outputs.push(String::new());
                }
            }
        }

        let n = string_outputs.len();
        Ok(BackendOutput {
            float_outputs: Vec::new(),
            string_outputs: vec![string_outputs],
            row_status,
            row_errors,
            shapes: vec![(n, 0)],
        })
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

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}
