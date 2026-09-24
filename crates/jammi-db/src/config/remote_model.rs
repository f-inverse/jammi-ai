//! `[models.remote.<name>]` — a model served at a remote endpoint, declared
//! by the deployment and referenced as `remote:<name>`.
//!
//! The declaration lives in configuration, not the catalog: it carries the
//! endpoint's credentials, and credentials belong where [`SecretSource`]
//! already lives. Everything a run's outputs depend on — the protocol, the
//! endpoint, the remote model's name, its output width and the revision the
//! operator pins — is what a materialization records for the run.

use std::collections::BTreeMap;
use std::num::{NonZeroU64, NonZeroUsize};

use serde::{Deserialize, Serialize};

use super::secret::SecretSource;
use crate::error::{JammiError, Result};

/// The wire protocol a remote endpoint speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RemoteProtocol {
    /// The OpenAI-compatible embeddings request (`{"input": [..], "model":
    /// ..}` answered by `{"data": [{"index", "embedding"}]}`), served by
    /// hosted APIs and by self-hosted inference servers alike.
    OpenaiEmbeddings,
}

/// One remote model's declaration.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteModelConfig {
    /// The protocol the endpoint speaks.
    pub protocol: RemoteProtocol,
    /// The full URL requests are sent to (for the OpenAI protocol, the
    /// embeddings route itself, e.g. `https://host/v1/embeddings`).
    pub url: String,
    /// The model name the endpoint is asked for.
    pub model: String,
    /// The width of every vector the model returns. A response of any other
    /// width is refused by row.
    pub dimensions: NonZeroUsize,
    /// The operator's pin of what the name serves (a provider's model
    /// snapshot, a deployment tag). No endpoint exposes a digest of its
    /// weights, so this pin — with the endpoint and name — is what tells
    /// two runs apart: change it when the model behind the name changes.
    pub revision: String,
    /// Headers sent with every request (an `Authorization` bearer token, an
    /// API-key header), each inline or `{ file = "…" }`. Never logged, never
    /// recorded.
    #[serde(default)]
    pub headers: BTreeMap<String, SecretSource>,
    /// Seconds one request may take, connection included. Default: 60.
    #[serde(default = "RemoteModelConfig::default_timeout_secs")]
    pub timeout_secs: NonZeroU64,
    /// The most requests in flight to this model at once, across every plan
    /// in the process. Default: 4.
    #[serde(default = "RemoteModelConfig::default_max_in_flight")]
    pub max_in_flight: NonZeroUsize,
    /// How many times a request refused as retryable (429, 5xx, a timeout
    /// or a dropped connection) is sent again, after the delay the endpoint
    /// names in `Retry-After` or an exponential backoff. Default: 2.
    #[serde(default = "RemoteModelConfig::default_max_retries")]
    pub max_retries: u32,
}

impl RemoteModelConfig {
    fn default_timeout_secs() -> NonZeroU64 {
        const SIXTY: NonZeroU64 = match NonZeroU64::new(60) {
            Some(n) => n,
            None => panic!("60 is non-zero"),
        };
        SIXTY
    }

    fn default_max_in_flight() -> NonZeroUsize {
        const FOUR: NonZeroUsize = match NonZeroUsize::new(4) {
            Some(n) => n,
            None => panic!("4 is non-zero"),
        };
        FOUR
    }

    fn default_max_retries() -> u32 {
        2
    }

    /// Refuse a declaration no request could be built from, naming the key
    /// under `[models.remote.<name>]`.
    pub fn validate(&self, name: &str) -> Result<()> {
        let key = |field: &str| format!("models.remote.{name}.{field}");
        if name.is_empty() || name.contains(char::is_whitespace) {
            return Err(JammiError::Config(format!(
                "models.remote: {name:?} is not a model name (non-empty, no whitespace)"
            )));
        }
        super::http_url(&key("url"), &self.url)?;
        for (field, value) in [("model", &self.model), ("revision", &self.revision)] {
            if value.trim().is_empty() {
                return Err(JammiError::Config(format!(
                    "{} must not be empty",
                    key(field)
                )));
            }
        }
        if let Some(header) = self.headers.keys().find(|h| !is_header_name(h)) {
            return Err(JammiError::Config(format!(
                "{}: {header:?} is not an HTTP header name",
                key("headers")
            )));
        }
        Ok(())
    }
}

/// An RFC 9110 field name: one or more token characters.
fn is_header_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b))
}
