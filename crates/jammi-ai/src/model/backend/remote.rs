//! A model served at a remote endpoint: the deployment declares it
//! (`[models.remote.<name>]`), a plan names it `remote:<name>`, and its
//! forwards are requests.
//!
//! The endpoint is the device its forwards run on: it holds no memory this
//! process budgets, so its admission is the declaration's `max_in_flight`,
//! and a chunk of rows is one request. The endpoint tokenizes, so a row
//! costs one and the chunk budget bounds a request in rows.

use std::time::Duration;

use arrow::array::ArrayRef;
use jammi_datafusion::{BackendOutput, ModelTask};
use jammi_db::config::{RemoteModelConfig, RemoteProtocol};
use jammi_db::error::{JammiError, Result};
use jammi_db::store::manifest::RemoteRun;
use jammi_numerics::ShapeLadder;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, RETRY_AFTER};
use serde::{Deserialize, Serialize};

use super::rows::{embedding_output, TextRows};

/// A declared remote model, ready to send requests: its declaration, with
/// the credentials resolved into headers the client marks sensitive.
pub struct RemoteModel {
    name: String,
    run: RemoteRun,
    client: reqwest::Client,
    headers: HeaderMap,
    max_retries: u32,
}

impl std::fmt::Debug for RemoteModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The headers carry credentials; they are never printed.
        f.debug_struct("RemoteModel")
            .field("name", &self.name)
            .field("run", &self.run)
            .finish_non_exhaustive()
    }
}

/// One request's rows: the texts that carry input, and every row's status.
pub struct PreparedRequest {
    valid: Vec<usize>,
    texts: Vec<String>,
    row_status: Vec<bool>,
    row_errors: Vec<String>,
}

impl RemoteModel {
    /// The model `name` declares: its credentials resolved, its client
    /// built. A credential that cannot be read, or that is not a valid
    /// header value, is refused naming the header — never its value.
    pub fn from_config(name: &str, config: &RemoteModelConfig) -> Result<Self> {
        let headers = config
            .headers
            .iter()
            .map(|(header, source)| {
                let key = HeaderName::from_bytes(header.as_bytes()).map_err(|e| {
                    JammiError::Config(format!("models.remote.{name}.headers.{header}: {e}"))
                })?;
                let mut value =
                    HeaderValue::from_str(source.resolve()?.expose()).map_err(|_| {
                        JammiError::Config(format!(
                            "models.remote.{name}.headers.{header}: the value is not a valid \
                             header value"
                        ))
                    })?;
                value.set_sensitive(true);
                Ok((key, value))
            })
            .collect::<Result<HeaderMap>>()?;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs.get()))
            .build()
            .map_err(|e| JammiError::Backend(format!("remote:{name}: HTTP client: {e}")))?;
        Ok(Self {
            name: name.to_string(),
            run: RemoteRun {
                protocol: config.protocol,
                url: config.url.clone(),
                model: config.model.clone(),
                dimensions: config.dimensions.get() as u64,
                revision: config.revision.clone(),
            },
            client,
            headers,
            max_retries: config.max_retries,
        })
    }

    /// The run a materialization records for this model.
    pub fn run(&self) -> &RemoteRun {
        &self.run
    }

    /// The width of every vector the model returns.
    pub fn dimensions(&self) -> usize {
        self.run.dimensions as usize
    }

    /// Refuse a task the protocol does not carry.
    pub(crate) fn check_task(&self, task: ModelTask) -> Result<()> {
        match (self.run.protocol, task) {
            (RemoteProtocol::OpenaiEmbeddings, ModelTask::TextEmbedding) => Ok(()),
            (protocol, task) => Err(JammiError::Model {
                model_id: format!("remote:{}", self.name),
                message: format!(
                    "the {protocol:?} protocol carries text embedding only; it cannot run \
                     {task}"
                ),
            }),
        }
    }

    /// Every row costs one: the endpoint tokenizes, so the engine bounds a
    /// request in rows. A row with no input costs nothing.
    pub fn row_costs(&self, content: &[ArrayRef], task: ModelTask) -> Result<Vec<u32>> {
        self.check_task(task)?;
        Ok(TextRows::of(content)?
            .row_status
            .into_iter()
            .map(u32::from)
            .collect())
    }

    /// A request has no padded axis.
    pub fn shape_ladder(&self, task: ModelTask) -> Result<ShapeLadder> {
        self.check_task(task)?;
        Ok(ShapeLadder::fixed())
    }

    /// The rows of one request.
    pub fn prepare(&self, content: &[ArrayRef], task: ModelTask) -> Result<PreparedRequest> {
        self.check_task(task)?;
        let rows = TextRows::of(content)?;
        let texts = rows.valid_texts().into_iter().map(str::to_owned).collect();
        Ok(PreparedRequest {
            valid: rows.valid,
            texts,
            row_status: rows.row_status,
            row_errors: rows.row_errors,
        })
    }

    /// Send the request and lay the vectors back out at their rows. A chunk
    /// with no row carrying input sends nothing.
    pub async fn forward(&self, request: PreparedRequest) -> Result<BackendOutput> {
        let PreparedRequest {
            valid,
            texts,
            row_status,
            row_errors,
        } = request;
        let vectors = if texts.is_empty() {
            Vec::new()
        } else {
            self.embed(texts).await?
        };
        let rows: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
        embedding_output(&valid, row_status, row_errors, self.dimensions(), &rows)
            .map_err(|e| self.refusal(&e.to_string()))
    }

    /// The vectors of `texts`, in their order, retrying what the endpoint
    /// refuses as retryable.
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let body = EmbeddingRequest {
            input: &texts,
            model: &self.run.model,
        };
        let mut attempt = 0;
        loop {
            let sent = self
                .client
                .post(&self.run.url)
                .headers(self.headers.clone())
                .json(&body)
                .send()
                .await;
            let retry = match sent {
                Ok(response) if response.status().is_success() => {
                    let parsed: EmbeddingResponse = response
                        .json()
                        .await
                        .map_err(|e| self.refusal(&format!("unreadable response: {e}")))?;
                    return self.in_order(parsed, texts.len());
                }
                Ok(response) => {
                    let status = response.status();
                    let after = retry_after(response.headers());
                    if !(status.as_u16() == 429 || status.is_server_error()) {
                        let body = response
                            .text()
                            .await
                            .unwrap_or_else(|e| format!("(body unreadable: {e})"));
                        return Err(self.refusal(&format!("{status}: {}", truncated(&body))));
                    }
                    (format!("{status}"), after)
                }
                Err(e) if e.is_timeout() || e.is_connect() => (e.without_url().to_string(), None),
                Err(e) => return Err(self.refusal(&e.without_url().to_string())),
            };
            if attempt >= self.max_retries {
                return Err(self.refusal(&format!("{} after {} attempt(s)", retry.0, attempt + 1)));
            }
            tokio::time::sleep(retry.1.unwrap_or_else(|| backoff(attempt))).await;
            attempt += 1;
        }
    }

    /// The response's vectors placed by their `index`, every input covered
    /// exactly once — a missing, repeated or out-of-range index is refused
    /// naming it, never paired with the wrong row by position.
    fn in_order(&self, response: EmbeddingResponse, inputs: usize) -> Result<Vec<Vec<f32>>> {
        let mut placed: Vec<Option<Vec<f32>>> = vec![None; inputs];
        for datum in response.data {
            let slot = placed.get_mut(datum.index).ok_or_else(|| {
                self.refusal(&format!(
                    "index {} is outside the {inputs} input(s) sent",
                    datum.index
                ))
            })?;
            if slot.replace(datum.embedding).is_some() {
                return Err(self.refusal(&format!("index {} answered twice", datum.index)));
            }
        }
        placed
            .into_iter()
            .enumerate()
            .map(|(i, vector)| {
                vector.ok_or_else(|| self.refusal(&format!("no vector for input {i}")))
            })
            .collect()
    }

    /// A refusal naming this model and its endpoint — never a header.
    fn refusal(&self, reason: &str) -> JammiError {
        JammiError::Backend(format!(
            "remote:{} at {}: {reason}",
            self.name, self.run.url
        ))
    }
}

/// The delay the endpoint asks for, in whole seconds.
fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    headers
        .get(RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse()
        .ok()
        .map(Duration::from_secs)
}

/// 250 ms, doubling per attempt, capped at 8 s.
fn backoff(attempt: u32) -> Duration {
    Duration::from_millis(250u64.saturating_mul(1 << attempt.min(5)))
}

/// A response body short enough to put in an error.
fn truncated(body: &str) -> &str {
    let end = body
        .char_indices()
        .nth(512)
        .map_or(body.len(), |(at, _)| at);
    &body[..end]
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    input: &'a [String],
    model: &'a str,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingDatum>,
}

#[derive(Deserialize)]
struct EmbeddingDatum {
    index: usize,
    embedding: Vec<f32>,
}
