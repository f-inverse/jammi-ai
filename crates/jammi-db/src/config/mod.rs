use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

use serde::{Deserialize, Deserializer, Serialize};

pub use crate::catalog::lease::LeaseIntervals;
use crate::error::{JammiError, Result};
use crate::storage::{AzureConfig, CloudConfig, GcsConfig, R2Config, S3Config};

mod env_map;
mod layers;
pub mod secret;
#[cfg(test)]
mod tests;

pub use secret::{Secret, SecretSource};

use layers::Node;

// ─── Config-layer enums ─────────────────────────────────────────────────────

/// Backend selection strategy for model inference.
///
/// `Auto` defers to the model resolver; concrete variants force a specific backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendSelection {
    Auto,
    Candle,
    Ort,
    Http,
}

impl fmt::Display for BackendSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Candle => write!(f, "candle"),
            Self::Ort => write!(f, "ort"),
            Self::Http => write!(f, "http"),
        }
    }
}

impl FromStr for BackendSelection {
    type Err = JammiError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "candle" => Ok(Self::Candle),
            "ort" => Ok(Self::Ort),
            "http" => Ok(Self::Http),
            other => Err(JammiError::Config(format!(
                "Unknown backend '{other}'. Expected: auto, candle, ort, http"
            ))),
        }
    }
}

/// Distance metric for ANN indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistanceMetric {
    Cosine,
    L2,
}

impl fmt::Display for DistanceMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cosine => write!(f, "cosine"),
            Self::L2 => write!(f, "l2"),
        }
    }
}

impl FromStr for DistanceMetric {
    type Err = JammiError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "cosine" => Ok(Self::Cosine),
            "l2" => Ok(Self::L2),
            other => Err(JammiError::Config(format!(
                "Unknown distance metric '{other}'. Expected: cosine, l2"
            ))),
        }
    }
}

/// ANN index type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexType {
    IvfHnswSq,
}

impl fmt::Display for IndexType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IvfHnswSq => write!(f, "ivf_hnsw_sq"),
        }
    }
}

impl FromStr for IndexType {
    type Err = JammiError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "ivf_hnsw_sq" => Ok(Self::IvfHnswSq),
            other => Err(JammiError::Config(format!(
                "Unknown index type '{other}'. Expected: ivf_hnsw_sq"
            ))),
        }
    }
}

/// Log output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    Text,
    Json,
}

impl fmt::Display for LogFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text => write!(f, "text"),
            Self::Json => write!(f, "json"),
        }
    }
}

impl FromStr for LogFormat {
    type Err = JammiError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "text" => Ok(Self::Text),
            "json" => Ok(Self::Json),
            other => Err(JammiError::Config(format!(
                "Unknown log format '{other}'. Expected: text, json"
            ))),
        }
    }
}

// ─── Config structs ─────────────────────────────────────────────────────────

/// Top-level configuration for the Jammi AI engine.
///
/// Load from a TOML file via [`JammiConfig::load`], with environment variable overrides.
///
/// # Catalog and broker selection
///
/// ```toml
/// artifact_dir = "/var/lib/jammi"
///
/// [catalog.postgres]
/// url = "${POSTGRES_URL}?sslmode=verify-full&sslrootcert=/etc/ssl/certs/ca-certificates.crt"
/// pool_size = 16
/// max_lifetime_secs = 1800
///
/// [broker.jet_stream]
/// url = "nats://${NATS_HOST}:4222"
/// retention_seconds = 604800
/// credentials = { file = "/var/run/secrets/nats.creds" }
/// ```
///
/// `catalog.postgres.url` should carry `?sslmode=verify-full` (plus
/// `sslrootcert=` naming a CA bundle, unless the server's certificate chains
/// to a public root) rather than the sslx default `prefer` — `sqlx` verifies
/// against the **webpki** root store, not the OS trust store, so a private CA
/// needs `sslrootcert=` even on a host that otherwise trusts it system-wide.
/// `sslmode=require` upgrades the connection to TLS but never verifies the
/// server's certificate — it defeats a MITM only when the network path is
/// already trusted, which is not the assumption behind reaching outside the
/// process. See the guide's "Catalog Backend and Trigger Broker" page.
///
/// # Secrets
///
/// Secret-valued fields (`catalog.postgres.url`, `broker.jet_stream.url`,
/// `broker.jet_stream.credentials`, `inference.http.headers` values,
/// `storage.cloud.*`'s credential fields) are
/// typed [`Secret`]: they accept either the value inline or `{ file = "…" }`
/// naming a file that holds it, resolve at load, and render as `Secret(***)`
/// in every `Debug` — so a `{:?}` of the whole config carries no secret. See
/// [`secret`] for the rules.
///
/// # Environment variable overrides
///
/// Every field above is overridable: `JAMMI_<PATH>` (segments joined by
/// `__`, e.g. `JAMMI_CATALOG__POSTGRES__URL`) always resolves as config, and
/// a bare `JAMMI_<FIELD>` (no `__`) resolves as config iff `FIELD` exactly
/// names one of `JammiConfig`'s own top-level fields — every other
/// `JAMMI_*` name is a runtime knob outside this layer and is left alone. An
/// unknown section, an unknown key, or a value outside a field's domain is a
/// load-time error naming the offending variable — never a silent drop.
/// [`JammiConfig::load_from`] is the layering entry point;
/// [`JammiConfig::load`] is `load_from` against the real process
/// environment.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct JammiConfig {
    /// Root directory for all persisted artifacts (catalog DB, model weights, indices).
    /// Default: platform-specific data directory, or `.jammi` as fallback.
    pub artifact_dir: PathBuf,
    /// DataFusion query-engine settings.
    pub engine: EngineConfig,
    /// GPU device and memory settings.
    pub gpu: GpuConfig,
    /// Model inference defaults (backend, batching, concurrency).
    pub inference: InferenceConfig,
    /// Embedding index defaults (distance metric, index type).
    pub embedding: EmbeddingConfig,
    /// Fine-tuning hyperparameter defaults.
    pub fine_tuning: FineTuningConfig,
    /// The one lease timing every leased catalog row shares: how long a claim
    /// (a training job, a building result table) is owned before it is
    /// reclaimable, and how often its holder renews it.
    pub lease: LeaseConfig,
    /// Worker-loop settings: whether this process claims `jobs` rows at all,
    /// which kinds it claims, and how often an idle worker polls for work.
    pub worker: WorkerConfig,
    /// Job retention: how long a terminal `jobs` row survives the sweep.
    pub jobs: JobsConfig,
    /// Cache layer settings (ANN cache, embedding cache).
    pub cache: CacheConfig,
    /// HTTP and Arrow Flight server bind addresses.
    pub server: ServerConfig,
    /// Tracing/logging configuration.
    pub logging: LoggingConfig,
    /// Vendor-neutral OTLP trace export: collector endpoint, request headers,
    /// `service.name`, and sample ratio. Built into an exporter by
    /// `jammi_ai::telemetry::otlp_layer` (behind the `telemetry-otlp` cargo
    /// feature); `jammi-db` carries only the raw, typed section.
    pub observability: ObservabilityConfig,
    /// Catalog backend selection. Default: SQLite under `artifact_dir`.
    pub catalog: CatalogConfig,
    /// Trigger broker selection. Default: in-process [`crate::trigger::InMemoryBroker`].
    pub broker: BrokerConfig,
    /// Audit signing-key source. Default: env-backed
    /// [`crate::audit::EnvSigningKeyStore`] reading `JAMMI_AUDIT_MASTER_KEY`.
    pub signing_key: SigningKeyConfig,
    /// Object-storage selection for result tables and cloud sources. Default:
    /// empty — result tables live on local disk under `artifact_dir` and
    /// `r2://`/`s3://`/`gs://`/`azure://` sources resolve via the SDK's
    /// default credential chain.
    pub storage: StorageConfig,
    /// Model source (Hugging Face Hub cache root, endpoint, token, offline
    /// mode). Built once into a `HubSource` at the `jammi-ai` session
    /// choke point; `jammi-db` only carries the raw, typed selection.
    pub models: ModelsConfig,
}

/// Object-storage configuration for Jammi-owned result tables and for
/// resolving cloud data sources.
///
/// Both fields are optional. When `result_root` is unset, result tables
/// (Parquet + USearch sidecar indexes) live on local disk under
/// `{artifact_dir}/jammi_db/` — today's behaviour. When set, it is a storage
/// URL (`r2://bucket/prefix`, `s3://bucket/prefix`, `gs://…`, `azure://…`,
/// or a local `file:///…`) the session roots every new result table under.
///
/// `cloud` carries the driver credentials. It is the **default** cloud config
/// the session threads to every object-store driver it builds — for the result
/// root *and* for a wire `AddSource("r2://…")` whose `SourceConnection` carries
/// no inline credentials. Secrets are not required here: the S3/R2 drivers read
/// `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_ENDPOINT_URL` from the
/// process environment (`AmazonS3Builder::from_env`), so a deployer can supply
/// only the non-secret bits in the TOML and inject the access key + secret as
/// container env vars.
///
/// # TOML — Cloudflare R2 result tables
///
/// ```toml
/// [storage]
/// result_root = "r2://jammi-results/prod"
///
/// [storage.cloud.r2]
/// account_id = "abc123def456"
/// # access_key_id / secret_access_key come from the environment:
/// #   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
/// ```
///
/// # TOML — S3 with an explicit region, secrets from env
///
/// ```toml
/// [storage]
/// result_root = "s3://jammi-results/prod"
///
/// [storage.cloud.s3]
/// region = "us-east-1"
/// ```
///
/// `[storage.cloud]` is externally tagged by cloud provider — `s3`, `r2`,
/// `gcs`, or `azure` — rather than a `kind` key inside one shared table
/// (H2/H16); a bare `storage.cloud = "s3"` also selects a variant with its
/// per-field defaults.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StorageConfig {
    /// Storage URL the session roots result tables under. `None` → local
    /// `{artifact_dir}/jammi_db/`.
    pub result_root: Option<String>,
    /// Default per-cloud driver credentials. Threaded to every driver the
    /// session builds for the result root and for cloud sources. `None` → the
    /// SDK default credential chain (env vars, instance profile, …).
    ///
    /// Deserializes through the config-only, externally-tagged
    /// `CloudSection` wrapper (never [`CloudConfig`]'s own internally
    /// tagged, non-`deny_unknown_fields` shape — that shape stays exactly as
    /// persisted in `sources.options` rows) and converts; see
    /// `cloud_via_section`.
    #[serde(deserialize_with = "cloud_via_section")]
    pub cloud: Option<CloudConfig>,
}

/// Config-only, externally tagged mirror of [`CloudConfig`]'s four
/// variants, used solely to deserialize `[storage.cloud]` — see
/// [`StorageConfig::cloud`] and [`cloud_via_section`]. `CloudConfig` itself
/// (the shape persisted in a `sources.options` row) is untouched and stays
/// without `deny_unknown_fields`, so an old row with an unknown future key
/// still reloads (H2/H17).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum CloudSection {
    S3(S3Section),
    R2(R2Section),
    Gcs(GcsSection),
    Azure(AzureSection),
}

/// Config-side mirror of [`crate::storage::S3Config`]. `secret_access_key`
/// and `session_token` are [`Secret`]-typed (H9); `access_key_id` is not.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct S3Section {
    region: Option<String>,
    endpoint: Option<String>,
    access_key_id: Option<String>,
    secret_access_key: Option<Secret>,
    session_token: Option<Secret>,
    #[serde(default)]
    allow_http: bool,
}

/// Config-side mirror of [`crate::storage::R2Config`].
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct R2Section {
    account_id: Option<String>,
    endpoint: Option<String>,
    access_key_id: Option<String>,
    secret_access_key: Option<Secret>,
    #[serde(default)]
    allow_http: bool,
}

/// Config-side mirror of [`crate::storage::GcsConfig`], collapsed to a
/// single `service_account` field (inline JSON, or `{ file = "…" }`
/// naming a service-account JSON file) that replaces the persisted shape's
/// separate `service_account_json`/`service_account_path` — see
/// [`cloud_via_section`] for how it maps onto them.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct GcsSection {
    service_account: Option<Secret>,
}

/// Config-side mirror of [`crate::storage::AzureConfig`].
/// `account_key`/`sas_token`/`client_secret` are [`Secret`]-typed (H9).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AzureSection {
    account_name: Option<String>,
    account_key: Option<Secret>,
    sas_token: Option<Secret>,
    tenant_id: Option<String>,
    client_id: Option<String>,
    client_secret: Option<Secret>,
}

impl From<CloudSection> for CloudConfig {
    fn from(section: CloudSection) -> Self {
        match section {
            CloudSection::S3(c) => CloudConfig::S3(S3Config {
                region: c.region,
                endpoint: c.endpoint,
                access_key_id: c.access_key_id,
                secret_access_key: c.secret_access_key,
                session_token: c.session_token,
                allow_http: c.allow_http,
            }),
            CloudSection::R2(c) => CloudConfig::R2(R2Config {
                account_id: c.account_id,
                endpoint: c.endpoint,
                access_key_id: c.access_key_id,
                secret_access_key: c.secret_access_key,
                allow_http: c.allow_http,
            }),
            CloudSection::Gcs(c) => CloudConfig::Gcs(GcsConfig {
                service_account_json: c.service_account,
                service_account_path: None,
            }),
            CloudSection::Azure(c) => CloudConfig::Azure(AzureConfig {
                account_name: c.account_name,
                account_key: c.account_key,
                sas_token: c.sas_token,
                tenant_id: c.tenant_id,
                client_id: c.client_id,
                client_secret: c.client_secret,
            }),
        }
    }
}

/// `StorageConfig::cloud`'s `deserialize_with`: deserialize an
/// `Option<CloudSection>` (the config-only, externally-tagged,
/// `deny_unknown_fields` shape) and convert to `Option<CloudConfig>` (H2).
/// There is no separate "wire struct" — `StorageConfig`'s derived
/// `Deserialize` is the sole config-layer path for `cloud`, so a `kind =`
/// path can never drift from this one.
fn cloud_via_section<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<CloudConfig>, D::Error> {
    Ok(Option::<CloudSection>::deserialize(deserializer)?.map(CloudConfig::from))
}

/// Catalog backend selection. The catalog and the mutable companion tables
/// share this backend.
///
/// Externally tagged: the variant name is a TOML table (or a bare string for
/// a variant with no required fields), never a `kind` key inside one shared
/// `[catalog]` table — so `[catalog.postgres]` and `[catalog.sqlite]` cannot
/// both be present without naming that conflict.
///
/// # TOML
///
/// ```toml
/// [catalog.sqlite]
/// # path = "/var/lib/jammi/catalog.db"   # optional override
/// ```
///
/// ```toml
/// [catalog.postgres]
/// url = "${POSTGRES_URL}?sslmode=verify-full&sslrootcert=/etc/ssl/certs/ca-certificates.crt"
/// pool_size = 16
/// max_lifetime_secs = 1800
/// ```
///
/// ```toml
/// [catalog.postgres]
/// url = { file = "/run/secrets/postgres-url" }
/// ```
///
/// `url` carries the `sslmode`/`sslrootcert` query parameters like any other
/// part of the connection string — see [`JammiConfig`]'s "Catalog and broker
/// selection" section for why `verify-full` (not the sqlx default `prefer`,
/// and not the encrypted-but-unverified `require`) is the right value for a
/// deployment that leaves its own trusted network.
///
/// A bare string also selects a variant with no required fields:
/// `catalog = "sqlite"`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum CatalogConfig {
    /// SQLite under `artifact_dir`. The laptop / dev default.
    Sqlite {
        /// Override the catalog DB path. Defaults to
        /// `{artifact_dir}/catalog.db` when omitted.
        #[serde(default)]
        path: Option<PathBuf>,
    },
    /// Postgres (or compatible) catalog. Used for SaaS deployments and
    /// self-hosted production.
    Postgres {
        /// Connection URL, e.g.
        /// `postgres://user:pass@host:5432/jammi?sslmode=verify-full&sslrootcert=/etc/ssl/certs/ca-certificates.crt`.
        /// A [`Secret`]: inline or `{ file = "…" }`; never printed.
        url: Secret,
        /// `sqlx::PgPool` `max_connections`. Default: 8.
        #[serde(default = "default_pool_size")]
        pool_size: u32,
        /// Optional `sqlx::PgPool` `max_lifetime` in seconds. `None`
        /// leaves the pool default in effect.
        #[serde(default)]
        max_lifetime_secs: Option<u32>,
    },
}

impl Default for CatalogConfig {
    fn default() -> Self {
        Self::Sqlite { path: None }
    }
}

fn default_pool_size() -> u32 {
    8
}

/// Trigger broker selection.
///
/// Externally tagged, like [`CatalogConfig`].
///
/// # TOML
///
/// ```toml
/// broker = "in_memory"
/// ```
///
/// ```toml
/// [broker.jet_stream]
/// url = "nats://${NATS_HOST}:4222"
/// retention_seconds = 604800
/// credentials = { file = "/var/run/secrets/nats.creds" }
/// ```
#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum BrokerConfig {
    /// In-process broker. Default; matches the laptop / dev workflow.
    #[default]
    InMemory,
    /// JetStream (NATS). Requires the `jetstream-broker` cargo feature on
    /// `jammi-db`; building a session whose config selects `JetStream`
    /// without the feature returns [`crate::error::JammiError::Config`].
    JetStream {
        /// NATS server URL, e.g. `nats://nats.svc:4222`. A [`Secret`]: NATS
        /// URLs carry userinfo/token auth inline (`nats://user:pass@host`),
        /// the same class of leak `catalog.postgres.url` guards against —
        /// inline or `{ file = "…" }`; never printed.
        url: Secret,
        /// Default per-stream retention in seconds. Per-topic
        /// `broker_metadata.retention_seconds` overrides this value.
        /// Default: 7 days (604 800).
        #[serde(default = "default_retention_secs")]
        retention_seconds: u64,
        /// Optional NATS credentials — the **contents** of a `.creds` file
        /// (user JWT + NKEY seed), as a [`Secret`]: inline, or
        /// `{ file = "/run/secrets/nats.creds" }` to read the file at load.
        /// When unset the broker connects anonymously.
        #[serde(default)]
        credentials: Option<Secret>,
    },
    /// Postgres `LISTEN`/`NOTIFY` wake-up transport
    /// ([`crate::trigger::PostgresBroker`]) — a topology change, not a
    /// cargo feature: `sqlx`'s `postgres` feature is unconditional in the
    /// workspace, so this variant always compiles in.
    ///
    /// This driver carries no data of its own — the topic's mutable backing
    /// table is the authoritative log, and every replica's `url` MUST point
    /// at the SAME Postgres database (`NOTIFY` is scoped to one instance; a
    /// replica listening on a different database silently degrades to
    /// `idle_poll`-only delivery, never data loss — see
    /// `crate::trigger::postgres`'s module docs for the full connection-cost
    /// and pool-budget accounting).
    ///
    /// # TOML
    ///
    /// ```toml
    /// [broker.postgres]
    /// # url = "postgres://user:pass@host:5432/jammi"   # optional; defaults
    /// #                                                 # to `catalog.postgres.url`
    /// idle_poll_secs = 5
    /// ```
    Postgres {
        /// Connection URL. Defaults to `[catalog.postgres].url` when unset
        /// AND the catalog itself is Postgres; a SQLite catalog with no
        /// explicit `url` here is a typed [`crate::error::JammiError::Config`]
        /// naming both keys (there is no default to fall back to). A
        /// [`Secret`]: inline or `{ file = "…" }`; never printed. An
        /// explicit `url` pointing at a Postgres database other than the
        /// catalog's is allowed — this driver carries no data, so only the
        /// NOTIFY channel needs to be shared across replicas.
        #[serde(default)]
        url: Option<Secret>,
        /// Idle-tick interval: how often every topic is woken regardless of
        /// NOTIFY traffic, bounding how long a lost notification (a listener
        /// reconnect window, a notify-queue overflow) can go undetected.
        /// Must be `>= 1`; `0` is a typed [`crate::error::JammiError::Config`].
        /// Default: 5.
        #[serde(default = "default_idle_poll_secs")]
        idle_poll_secs: u64,
    },
}

fn default_retention_secs() -> u64 {
    7 * 24 * 60 * 60
}

fn default_idle_poll_secs() -> u64 {
    5
}

/// Audit signing-key source selection.
///
/// Selects the [`crate::audit::SigningKeyStore`] the session uses to obtain the
/// audit HMAC master key. The self-host seam: a future deployment holding its
/// master key in a key-management service selects a new variant here without
/// touching the signing path.
///
/// # TOML
///
/// ```toml
/// signing_key = "env"
/// ```
///
/// ```toml
/// [signing_key.file]
/// path = "/run/secrets/jammi-audit-master-key"
/// ```
#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum SigningKeyConfig {
    /// Read the master key from `JAMMI_AUDIT_MASTER_KEY` via
    /// [`crate::audit::EnvSigningKeyStore`]. The default.
    #[default]
    Env,
    /// Read the master key from a file via
    /// [`crate::audit::FileSigningKeyStore`]: the same 64-hex-character key
    /// the env store expects, one trailing newline tolerated. The file is
    /// read at each signing-key request, not at config load, so a rotated
    /// mount is picked up without a restart — and a missing file is the same
    /// `MasterKey` error an unset variable is.
    File {
        /// Path of the file holding the hex-encoded 32-byte master key.
        path: PathBuf,
    },
}

/// DataFusion query-engine settings.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngineConfig {
    /// Number of DataFusion execution threads. Default: available CPU count.
    pub execution_threads: usize,
    /// Maximum memory for the query engine (e.g., `"75%"` or `"4GB"`). Default: `"75%"`.
    pub memory_limit: String,
    /// Maximum rows per DataFusion batch. Default: 8192.
    pub batch_size: usize,
}

/// GPU device and memory settings.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GpuConfig {
    /// CUDA device ordinal. Default: 0.
    pub device: i32,
    /// GPU memory limit (e.g., `"auto"` or `"8GB"`). Default: `"auto"`.
    pub memory_limit: String,
    /// Fraction of GPU memory to allocate (0.0 - 1.0). Default: 0.9.
    pub memory_fraction: f64,
    /// Require a usable GPU: when `true`, refuse to fall back to CPU and fail
    /// fast if the requested device is unavailable. Default: `false` (degrade
    /// to CPU with a warning).
    pub require_gpu: bool,
    /// Global default inference compute precision. A per-model
    /// `compute_precision` declared in the model's `config.json` overrides
    /// this; both default to `F32`. `BF16` is a GPU-tier value: the candle
    /// load boundary admits it on a CUDA device of compute capability >= 8.0
    /// (Ampere+) and fails loud on a lower-capability or non-CUDA device.
    /// Default: `F32`.
    pub compute_precision: jammi_numerics::ComputePrecision,
}

/// Model inference defaults.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InferenceConfig {
    /// Backend selection strategy. Default: `Auto`.
    pub default_backend: BackendSelection,
    /// Maximum requests per inference batch. Default: 32.
    pub batch_size: usize,
    /// Seconds to wait before flushing an incomplete batch. Default: 300.
    pub batch_timeout_secs: u64,
    /// Maximum number of models held in memory simultaneously. 0 means unlimited. Default: 0.
    pub max_loaded_models: usize,
    /// HTTP backend configuration (for remote inference endpoints).
    pub http: HttpConfig,
}

/// HTTP backend configuration for remote inference endpoints.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HttpConfig {
    /// Request timeout in seconds. Default: 60.
    pub timeout_secs: u64,
    /// Extra HTTP headers sent with every inference request. Header values
    /// are [`Secret`]s (an `Authorization` bearer token is the common case):
    /// each accepts the value inline or `{ file = "…" }`, and none is ever
    /// printed. Ordered so the request header set is deterministic.
    pub headers: BTreeMap<String, Secret>,
}

/// The precision the ANN sidecar index quantizes its stored vectors to.
///
/// This is a **storage** concept, orthogonal to
/// [`jammi_numerics::ComputePrecision`] (which names the dtype a model's
/// matmuls run at): `StoragePrecision` names the dtype the HNSW graph's own
/// vectors are quantized to on disk / in RAM, independent of how those
/// vectors were computed. The Parquet result table always holds `f32` — this
/// enum governs only the sidecar accelerator's memory footprint.
///
/// `F32` keeps the index's own vectors exact, so a search is single-stage. A
/// quantized precision (`F16`, `Int8`) shrinks the in-RAM graph but makes the
/// index's own stored vectors lossy, so a search over one is a two-stage
/// retrieve-then-rescore: the quantized graph proposes an oversampled
/// candidate set, and the exact `f32` rescore companion
/// ([`crate::index::sidecar::SidecarIndex::get_exact`]) re-ranks it.
///
/// `Binary` is USearch's `B1` scalar kind, paired with Hamming distance (bit
/// count) rather than cosine: each dimension is packed down to a single sign
/// bit (see [`crate::index::sidecar`]'s shared pack function), so its search
/// stage and rescore-candidate width behave differently enough from the other
/// three precisions to need their own defaults — see
/// [`Self::default_oversample`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StoragePrecision {
    /// Full-precision `f32` vectors in the index — exact, single-stage search.
    #[default]
    F32,
    /// 16-bit half-precision index vectors — quantized, rescored.
    F16,
    /// 8-bit signed-integer index vectors — quantized, rescored. USearch's
    /// `I8` scalar kind (linear per-vector affine quantization).
    Int8,
    /// 1-bit sign-quantized index vectors, searched by Hamming distance —
    /// quantized, rescored. USearch's `B1` scalar kind (one packed bit per
    /// dimension, `sign(v - τ)` against a per-dimension threshold τ fit from
    /// the corpus — see [`crate::index::sidecar`]'s `ThresholdKind`).
    Binary,
}

impl StoragePrecision {
    /// The USearch scalar-quantization kind this precision maps onto — the
    /// sole place the storage-precision vocabulary touches USearch's own
    /// `ScalarKind` naming.
    pub fn to_scalar_kind(self) -> usearch::ScalarKind {
        match self {
            Self::F32 => usearch::ScalarKind::F32,
            Self::F16 => usearch::ScalarKind::F16,
            Self::Int8 => usearch::ScalarKind::I8,
            Self::Binary => usearch::ScalarKind::B1,
        }
    }

    /// Whether a table at this precision needs the raw-`f32` rescore
    /// companion: `true` for every quantized precision (the index's own
    /// stored vectors are lossy), `false` for `F32` (the index already holds
    /// the exact vectors, so a second retrieval stage would only re-derive
    /// what the first stage already returned).
    pub fn needs_rescore(self) -> bool {
        !matches!(self, Self::F32)
    }

    /// The oversample multiplier a NEW table at this precision stamps onto
    /// its catalog row when the deployment has left
    /// [`AnnIndexConfig::oversample`] unset (`None`).
    ///
    /// `Binary`'s single-bit-per-dimension Hamming coarse stage needs a much
    /// wider candidate pool than the other three precisions' cosine-ranked
    /// stage to recover comparable recall — the Wave 1.5 go/no-go spike
    /// measured recall@1/10 = 0.94/0.992 at oversample `32` (vs. 0.59 recall@1
    /// at the shared default of `4`), so `Binary` gets its own wider default
    /// while `F32`/`F16`/`Int8` keep `4`.
    pub fn default_oversample(self) -> usize {
        match self {
            Self::Binary => 32,
            Self::F32 | Self::F16 | Self::Int8 => DEFAULT_OVERSAMPLE,
        }
    }
}

impl fmt::Display for StoragePrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::Int8 => write!(f, "int8"),
            Self::Binary => write!(f, "binary"),
        }
    }
}

impl FromStr for StoragePrecision {
    type Err = JammiError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "int8" => Ok(Self::Int8),
            "binary" => Ok(Self::Binary),
            other => Err(JammiError::Config(format!(
                "unknown storage precision '{other}'. Expected: f32, f16, int8, binary"
            ))),
        }
    }
}

/// Default of [`AnnIndexConfig::oversample`] — retrieve `k * 4` candidates
/// from the quantized graph before rescoring down to `k` exact results.
const DEFAULT_OVERSAMPLE: usize = 4;

/// HNSW graph-tuning knobs for the ANN sidecar index — the universal
/// recall-vs-cost dials of a hierarchical navigable small-world graph, named
/// for the HNSW primitive and independent of the backing index library.
///
/// `0` on any of the three HNSW fields means "use the index backend's
/// built-in default"; `connectivity` and `build_expansion` are fixed when the
/// graph is constructed, `search_expansion` is a query-time dial applied to a
/// loaded index. `storage_precision` and `oversample` are not HNSW knobs but
/// live here as the deployment-wide defaults every newly-created embedding
/// table's catalog row is stamped with at creation — see
/// [`crate::catalog::result_repo::ResultTableRecord::storage_precision`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnnIndexConfig {
    /// Maximum connections per graph node (HNSW *M*). Higher trades a larger
    /// index and slower build for better recall. `0` = backend default.
    pub connectivity: usize,
    /// Candidate-list width during graph construction (HNSW *ef_construction*).
    /// Higher trades slower build for a better-quality graph. `0` = backend default.
    pub build_expansion: usize,
    /// Candidate-list width during search (HNSW *ef_search*). Higher trades
    /// slower queries for better recall; mutable on a loaded index.
    /// `0` = backend default.
    pub search_expansion: usize,
    /// The precision new embedding tables' sidecar index is built at.
    /// Default: `F32`.
    pub storage_precision: StoragePrecision,
    /// How many candidates a quantized-index search retrieves per requested
    /// result before rescoring down to `k` exact matches (`k * oversample`).
    /// Irrelevant at `F32` (single-stage, no rescore).
    ///
    /// `None` (the default) means the deployment has not configured this
    /// knob — [`Self::effective_oversample_for`] falls back to the
    /// precision's own [`StoragePrecision::default_oversample`]. `Some(v)` is
    /// an explicit deployment choice and is honored *verbatim* for every
    /// precision, including `Some(4)` on a `Binary` deployment — an explicit
    /// override is never silently widened back to the precision default. A
    /// `Some(0)` (or any value below `1`) is clamped to `1` at
    /// [`Self::effective_oversample`] / [`Self::effective_oversample_for`] —
    /// an oversample below `1` would retrieve fewer candidates than the
    /// request asks for.
    pub oversample: Option<usize>,
}

impl AnnIndexConfig {
    /// [`Self::oversample`] resolved against the shared `DEFAULT_OVERSAMPLE`
    /// when unset, then clamped to at least `1` — the domain-valid multiplier
    /// a quantized-index retrieve stage actually uses when no per-precision
    /// context is available (the pre-migration-023 fallback path). A
    /// misconfigured `0` must never shrink the candidate set below the
    /// requested `k`.
    pub fn effective_oversample(&self) -> usize {
        self.oversample.unwrap_or(DEFAULT_OVERSAMPLE).max(1)
    }

    /// The oversample a NEW table at `precision` stamps onto its catalog row:
    /// the deployment's explicit [`Self::oversample`] when set — honored
    /// verbatim for every precision, including `Some(4)` on `Binary` — clamped
    /// to at least `1`; otherwise `precision`'s own
    /// [`StoragePrecision::default_oversample`], so an untouched deployment
    /// config stamps `32` for a `Binary` table and `4` for the other three.
    pub fn effective_oversample_for(&self, precision: StoragePrecision) -> usize {
        self.oversample
            .unwrap_or_else(|| precision.default_oversample())
            .max(1)
    }

    /// Resolve the retrieve→rescore oversample multiplier a single search
    /// actually uses: a per-request `request_override` wins, then the table's
    /// own stamped default (`ResultTableRecord::oversample`, passed as
    /// `table_default`) — never today's deployment config, mirroring how
    /// `storage_precision` is resolved — falling back to this deployment's
    /// [`Self::effective_oversample`] only for a table with no stamped oversample
    /// column. Clamped to at least `1`, so a `0` override or a misconfigured `0`
    /// default never shrinks the candidate set below `k`.
    pub fn resolve_oversample(
        &self,
        request_override: Option<usize>,
        table_default: Option<usize>,
    ) -> usize {
        request_override
            .or(table_default)
            .unwrap_or_else(|| self.effective_oversample())
            .max(1)
    }
}

/// Embedding index defaults.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EmbeddingConfig {
    /// Distance metric for ANN indices. Default: `Cosine`.
    pub default_distance_metric: DistanceMetric,
    /// ANN index type. Default: `IvfHnswSq`.
    pub default_index_type: IndexType,
    /// Rows between index checkpoint writes. Default: 1000.
    pub checkpoint_interval: usize,
    /// HNSW graph-tuning knobs for the ANN sidecar index.
    pub ann: AnnIndexConfig,
}

/// Fine-tuning hyperparameter defaults.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FineTuningConfig {
    /// LoRA adapter rank. Default: 8.
    pub default_lora_rank: usize,
    /// Initial learning rate. Default: 0.0002.
    pub default_learning_rate: f64,
    /// Number of training epochs. Default: 3.
    pub default_epochs: usize,
    /// Training batch size. Default: 8.
    pub default_batch_size: usize,
    /// Fraction of an epoch between checkpoint saves. Default: 0.1.
    pub checkpoint_fraction: f64,
}

/// The one lease timing every leased catalog row shares — a claimed training
/// job and a `building` result table are both owned under a lease their holder
/// heartbeats, and both are reclaimed by a sweep once it expires.
///
/// Both values are seconds. Defaults reproduce the engine's built-in timing
/// (30 s lease, 10 s heartbeat), so a config without a `[lease]` section behaves
/// identically to one that omits it. Short values let a deployment (or a test)
/// drive lease-expiry and reclaim quickly.
///
/// # TOML
///
/// ```toml
/// [lease]
/// duration_secs = 30
/// heartbeat_secs = 10
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LeaseConfig {
    /// How long a claim owns its row before it is considered orphaned and
    /// reclaimable. Default: 30.
    pub duration_secs: u64,
    /// How often the holder renews the lease. Must leave a real margin under
    /// `duration_secs` (see [`Self::intervals`]) so a single missed beat does
    /// not drop a live holder's lease. Default: 10.
    pub heartbeat_secs: u64,
}

impl Default for LeaseConfig {
    fn default() -> Self {
        Self {
            duration_secs: 30,
            heartbeat_secs: 10,
        }
    }
}

impl LeaseConfig {
    /// Resolve the typed [`LeaseIntervals`] this timing implies, enforcing the
    /// lease invariants:
    ///
    /// - both values non-zero;
    /// - `heartbeat_secs * 2 < duration_secs` — the heartbeat must leave a real
    ///   margin under the lease (a live holder beats at least twice within one
    ///   lease, so a single missed beat still leaves one in-window renewal that
    ///   lands strictly before the lease expires — never coincident with
    ///   expiry, which would race a reclaim).
    ///
    /// Returns [`JammiError::Config`] with a clear message on a violation. The
    /// engine never silently clamps a bad value, and no operator-supplied `u64`
    /// can overflow the margin check — an absurd heartbeat is rejected, not
    /// wrapped.
    pub fn intervals(&self) -> Result<LeaseIntervals> {
        use std::time::Duration;

        if self.heartbeat_secs == 0 {
            return Err(JammiError::Config(
                "lease.heartbeat_secs must be > 0".into(),
            ));
        }
        if self.duration_secs == 0 {
            return Err(JammiError::Config("lease.duration_secs must be > 0".into()));
        }
        // Overflow-safe strict margin: `heartbeat * 2 < lease`. The doubled
        // heartbeat overflowing `u64` is itself a rejection (any such value
        // dwarfs any finite lease), so the multiply never wraps or panics.
        if self
            .heartbeat_secs
            .checked_mul(2)
            .is_none_or(|hb2| hb2 >= self.duration_secs)
        {
            return Err(JammiError::Config(format!(
                "lease.heartbeat_secs ({}) must be strictly under half of \
                 lease.duration_secs ({}): the heartbeat must leave a real margin \
                 under the lease so a live holder renews strictly before the lease expires, \
                 or its claim is spuriously reclaimed mid-flight",
                self.heartbeat_secs, self.duration_secs
            )));
        }
        Ok(LeaseIntervals::new_validated(
            Duration::from_secs(self.duration_secs),
            Duration::from_secs(self.heartbeat_secs),
        ))
    }
}

/// Worker-loop settings: whether this process runs the claim loop at all,
/// which job kinds it claims, and — when it does — how often an idle worker
/// polls for new work. The lease a claim is held under and the heartbeat that
/// renews it are the deployment's one [`LeaseConfig`], shared with every
/// other leased row.
///
/// Replaces the former `[training] run_worker`/`idle_poll_secs`: the
/// claim loop is not training-specific — a process opts into claiming
/// any kind-agnostic `jobs` row, training or compute, and `kinds` selects
/// which. Every key `[training]` carried was worker-related, so the section
/// is gone entirely rather than left holding nothing.
///
/// Defaults reproduce the engine's built-in timing (1 s idle poll) with the
/// claim loop on and every kind claimed, so a config without a `[worker]`
/// section behaves identically to one that omits it.
///
/// The section rejects unknown keys. In particular the lease keys live only
/// under `[lease]` (`duration_secs` / `heartbeat_secs`); a TOML naming them
/// under `[worker]` is a hard load error, never a silent fall-back to the
/// default timing.
///
/// # TOML
///
/// ```toml
/// [worker]
/// enabled = true
/// kinds = "all"
/// idle_poll_secs = 1
/// metrics_sample_secs = 5
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WorkerConfig {
    /// Whether THIS process runs the claim loop at all. Default: `true`.
    ///
    /// `true` — the process opens the catalog and drives the loop: it claims
    /// queued jobs of `kinds`, renews the lease while they run, and reclaims
    /// leases that expired under a dead claimant.
    ///
    /// `false` — the process still mounts and serves the submission surface
    /// and still accepts submissions, but never claims. Submitted jobs stay
    /// `queued` until some process configured with `enabled = true` opens
    /// the catalog. On a single-process catalog (SQLite) that means this
    /// process must close the catalog first; a multi-process catalog
    /// (Postgres) can run both concurrently.
    ///
    /// This is a runtime/driver setting, not a build feature and not a separate
    /// code path: whatever arm decides whether to spawn a claim loop reads this
    /// same key, so a deployment that serves over the wire and one that runs
    /// in-process answer the question identically. Turning it off never
    /// removes the submission surface — only the claiming.
    ///
    /// The poll below describes the loop this flag gates; it is validated by
    /// [`Self::worker_intervals`] regardless of `enabled`, so a config that
    /// switches the loop on later cannot smuggle in bad timing.
    pub enabled: bool,
    /// Which job kinds this worker claims. `"all"` (the default) claims every
    /// kind compiled into the binary; an explicit list claims only those
    /// named. This layer carries the raw selection only — `jammi-ai` owns
    /// the kind vocabulary and validates it against the compiled set at
    /// startup, the same "raw tokens here, validated downstream" split
    /// [`ServiceSelection`] uses for `[server].services`.
    pub kinds: WorkerKinds,
    /// How often an idle worker polls for a queued job (and reclaims expired
    /// leases). Must be non-zero — a zero poll is a busy-loop. Default: 1.
    pub idle_poll_secs: u64,
    /// How often a worker-enabled process samples the queue gauges
    /// (`jammi_jobs_queued{kind}` / `jammi_jobs_running{kind}`) from the
    /// catalog — one `GROUP BY kind, status` statement per tick, on a
    /// dedicated task, never on a `/metrics` scrape and never on the claim
    /// loop (a scrape storm must not become a catalog storm). Must be
    /// `>= 1`. Default: 5.
    pub metrics_sample_secs: u64,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            // Default on: an unconfigured deployment is a whole one — it both
            // accepts jobs and works them. Opting out is the explicit act.
            enabled: true,
            kinds: WorkerKinds::default(),
            idle_poll_secs: 1,
            metrics_sample_secs: 5,
        }
    }
}

/// The worker's job-kind selection: `All` (the default) claims every kind
/// compiled into this binary; `Only` claims exactly the named kinds (e.g.
/// `[]` for a claim loop that runs but claims nothing — equivalent to
/// `enabled = false` for the poll loop's own effect, but distinguishable in
/// `ListWorkers`). Deserializes through the identical hand-written grammar
/// [`ServiceSelection`] uses (H7: `"all"`, a comma-separated list, or a TOML
/// array) — mapped 1:1 rather than duplicating the visitor, since the two
/// selections share exactly the same shape and differ only in vocabulary
/// (service tiers vs. job kinds, each validated by its own owning layer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerKinds {
    /// Every kind compiled into this binary.
    All(AllSentinel),
    /// Exactly these kinds.
    Only(Vec<String>),
}

impl Default for WorkerKinds {
    fn default() -> Self {
        WorkerKinds::All(AllSentinel::All)
    }
}

impl<'de> Deserialize<'de> for WorkerKinds {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Ok(match ServiceSelection::deserialize(deserializer)? {
            ServiceSelection::All(sentinel) => WorkerKinds::All(sentinel),
            ServiceSelection::Only(tokens) => WorkerKinds::Only(tokens),
        })
    }
}

/// The validated, typed worker timing the claim loop drives itself with.
///
/// `lease` and `heartbeat` are the deployment's [`LeaseIntervals`] — the single
/// source of truth for the lease window, so the worker's renew always targets
/// the same deadline the reclaim path compares against. The constructor
/// [`WorkerConfig::worker_intervals`] is the only way to build one, so the
/// margin and non-zero-poll invariants hold by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerIntervals {
    /// The lease window — how long a claim owns a job before it is reclaimable.
    pub lease: std::time::Duration,
    /// The lease-renewal interval, safely inside `lease`.
    pub heartbeat: std::time::Duration,
    /// The idle-worker poll interval.
    pub idle_poll: std::time::Duration,
}

impl WorkerConfig {
    /// Resolve the typed [`WorkerIntervals`] this timing implies over the
    /// deployment's validated `lease`, enforcing the worker's own invariant:
    /// `idle_poll_secs >= 1` — a zero idle poll is a busy-loop. The lease
    /// margin was already enforced by [`LeaseConfig::intervals`].
    ///
    /// Returns [`JammiError::Config`] with a clear message on a violation. The
    /// engine never silently clamps a bad value.
    pub fn worker_intervals(&self, lease: LeaseIntervals) -> Result<WorkerIntervals> {
        use std::time::Duration;

        if self.idle_poll_secs == 0 {
            return Err(JammiError::Config(
                "worker.idle_poll_secs must be > 0 (a zero poll is a busy-loop)".into(),
            ));
        }
        if self.metrics_sample_secs == 0 {
            return Err(JammiError::Config(
                "worker.metrics_sample_secs must be >= 1 (a zero interval is a busy-loop)".into(),
            ));
        }
        Ok(WorkerIntervals {
            lease: lease.lease(),
            heartbeat: lease.heartbeat(),
            idle_poll: Duration::from_secs(self.idle_poll_secs),
        })
    }
}

/// Job retention (N9): how long a TERMINAL `jobs` row (`completed` /
/// `failed`) keeps blocking `delete_model` and survives the retention sweep
/// (`prune_jobs`) before it is eligible for deletion. A non-terminal job
/// blocks `delete_model` indefinitely and is never pruned, regardless of age.
///
/// # TOML
///
/// ```toml
/// [jobs]
/// retention_days = 30
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct JobsConfig {
    /// How many days a terminal job row survives before the sweep may delete
    /// it (and before `delete_model`'s referential scan stops counting it as
    /// a blocking reference). Default: 30.
    pub retention_days: u32,
}

impl Default for JobsConfig {
    fn default() -> Self {
        Self { retention_days: 30 }
    }
}

impl JobsConfig {
    /// The largest `retention_days` accepted at load: ten years. Above it
    /// the value is almost certainly a units mistake (seconds or hours
    /// typed into a days field), and it is also where `days * 86_400` as a
    /// timestamp offset stops being a duration any catalog backend can
    /// subtract from `now` without overflow.
    pub const MAX_RETENTION_DAYS: u32 = 3650;

    /// Validate the retention window: `0` is allowed (a terminal job is
    /// prunable, and stops blocking `delete_model`, as soon as it is
    /// terminal); anything above [`Self::MAX_RETENTION_DAYS`] is a typed
    /// [`JammiError::Config`] naming the field, refused at load rather than
    /// surfacing as a clock overflow in the first retention sweep.
    pub fn validate(&self) -> Result<()> {
        if self.retention_days > Self::MAX_RETENTION_DAYS {
            return Err(JammiError::Config(format!(
                "[jobs] retention_days = {} exceeds the {}-day cap",
                self.retention_days,
                Self::MAX_RETENTION_DAYS
            )));
        }
        Ok(())
    }

    /// The retention window as a [`Duration`] — the one conversion every
    /// sweep and reference scan shares.
    pub fn retention(&self) -> Duration {
        Duration::from_secs(u64::from(self.retention_days) * 86_400)
    }
}

/// Cache layer settings.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CacheConfig {
    /// Enable the ANN query result cache. Default: true.
    pub ann_cache_enabled: bool,
    /// Maximum entries in the ANN cache. Default: 10000.
    pub ann_cache_max_entries: usize,
    /// Enable the embedding vector cache. Default: true.
    pub embedding_cache_enabled: bool,
    /// Maximum size of the embedding cache (e.g., `"1GB"`). Default: `"1GB"`.
    pub embedding_cache_size: String,
}

/// Arrow Flight SQL and health-probe server bind addresses.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    /// Health probe listen address. Default: `"0.0.0.0:8080"`.
    pub health_listen: String,
    /// Arrow Flight listen address. Default: `"0.0.0.0:8081"`.
    pub flight_listen: String,
    /// Models to load into the cache at server startup, BEFORE `/readyz`
    /// reports ready and before this process's claim loop claims anything
    /// (warm-before-ready). Each entry is a bare model id string, whose
    /// task is resolved from the catalog's `models` row at the server's
    /// startup edge, or a `{ id, task }` table naming the task explicitly
    /// (a `local:` path has no row). A listed model that cannot load, a
    /// bare id with no `models` row, or an unknown task token is a startup
    /// error: the server exits non-zero instead of serving. Default: `[]`.
    pub preload_models: Vec<PreloadEntry>,
    /// Optional gRPC service tiers this deployment mounts, beyond the always-on
    /// core tier. Tokens are `"event"`, `"eval"` (the `jammi-server`
    /// service-tier mechanism owns their meaning and validation; this layer
    /// only carries the raw selection so the engine config stays free of
    /// server-tier types). An empty list means serve-only (core only); the
    /// default mounts every tier (all-in-one). A token naming an unknown tier
    /// is a startup error surfaced by the server. Whether this process runs
    /// jobs is `[worker] enabled`, not a tier.
    pub services: ServiceSelection,
    /// Request-bounds and refusal-policy limits for the combined gRPC +
    /// Flight SQL surface. See [`LimitsConfig`].
    pub limits: LimitsConfig,
    /// The INTERNAL peer listener for beyond-one-node retrieval: the address
    /// this replica serves `jammi.v1.peer.PeerService` on, to other replicas
    /// of the same deployment. `None` (the default) = no third listener =
    /// single node; a replica is a segment owner iff this is set. Validated
    /// like `health_listen` / `flight_listen`: parseable, and distinct from
    /// both at a fixed port (`:0` never collides). Served outside the tenant
    /// layer (I-PEER): every client of it is a jammi coordinator.
    pub peer_bind: Option<String>,
    /// MARGINAL-LOAD ADMISSION per query, in bytes: the maximum estimated
    /// bytes ONE query may load locally for segments it does not own, when
    /// their owners are unreachable (the last rung of the placed-search
    /// failure ladder). Unset (the default) = unbounded. NOT a memory cap: the
    /// segment cache never evicts, earlier queries' loads are invisible to the
    /// check, and concurrent queries admit independently, so peak heap is
    /// concurrency × budget. Read by the result store; a library embedder sets
    /// it through the same config. `Some(0)` is refused.
    pub peer_local_load_bytes: Option<u64>,
}

/// The optional service-tier selection for a server deployment. `All` (the
/// default) mounts every optional tier; `Only` mounts core plus
/// exactly the named optional tiers. Kept as raw tokens here so `jammi-db` does
/// not depend on `jammi-server`'s tier vocabulary — the server resolves and
/// validates them.
///
/// Hand-written [`Deserialize`] (not `#[serde(untagged)]`) so the three
/// natural TOML/env forms all parse with one shared, case-sensitive grammar
/// (H7) — `services` is the one field in this config whose env spelling is
/// `all|tier,tier` rather than TOML syntax:
///
/// ```toml
/// services = "all"             # all-in-one (the default)
/// services = "event,eval"      # comma list (also accepted in TOML)
/// services = ["event", "eval"] # core + these optional tiers
/// services = []                # serve-only (core only)
/// ```
///
/// `JAMMI_SERVER__SERVICES=all` or `JAMMI_SERVER__SERVICES=event,eval` use
/// the same comma-list grammar. Exactly `"all"` (case-sensitive — `"ALL"` is
/// a one-token tier list, rejected by the server as an unknown tier name)
/// selects [`AllSentinel::All`]; empty tokens in a comma list are filtered,
/// so `""` and `","` both mean serve-only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceSelection {
    /// The `"all"` sentinel: mount core plus every optional tier compiled into
    /// this binary (all-in-one). The default deployment shape.
    All(AllSentinel),
    /// Mount core plus exactly these optional tiers (e.g. `["event"]` for an
    /// event box, or `[]` for serve-only).
    Only(Vec<String>),
}

/// The `"all"` literal, as its own one-variant marker type so
/// [`ServiceSelection::All`] cannot be constructed with anything else.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllSentinel {
    All,
}

impl Default for ServiceSelection {
    fn default() -> Self {
        ServiceSelection::All(AllSentinel::All)
    }
}

impl<'de> Deserialize<'de> for ServiceSelection {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct ServiceSelectionVisitor;

        impl<'de> serde::de::Visitor<'de> for ServiceSelectionVisitor {
            type Value = ServiceSelection;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(
                    "\"all\", a comma-separated tier list (e.g. \"event,eval\"), or an array of tier tokens",
                )
            }

            fn visit_str<E: serde::de::Error>(
                self,
                value: &str,
            ) -> std::result::Result<Self::Value, E> {
                // Trim surrounding whitespace before anything else: a
                // trailing newline from a secrets file or a heredoc-sourced
                // env var is not part of the value, the same way every other
                // env/file-backed value in this config tolerates it. H7:
                // case-sensitive otherwise, like every other value in this
                // config — `"ALL"` is NOT the sentinel.
                let value = value.trim();
                if value == "all" {
                    return Ok(ServiceSelection::All(AllSentinel::All));
                }
                Ok(ServiceSelection::Only(
                    value
                        .split(',')
                        .map(str::trim)
                        .filter(|t| !t.is_empty())
                        .map(str::to_string)
                        .collect(),
                ))
            }

            fn visit_string<E: serde::de::Error>(
                self,
                value: String,
            ) -> std::result::Result<Self::Value, E> {
                self.visit_str(&value)
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                // Same rule as `visit_str`'s comma-list arm: trim each
                // token and filter empties, so an array token sourced from
                // a templated value (a trailing/leading newline or blank
                // entry) is treated the same way a comma-list spelling of
                // the identical intent already is.
                let mut tokens = Vec::new();
                while let Some(token) = seq.next_element::<String>()? {
                    let token = token.trim();
                    if !token.is_empty() {
                        tokens.push(token.to_string());
                    }
                }
                Ok(ServiceSelection::Only(tokens))
            }
        }

        deserializer.deserialize_any(ServiceSelectionVisitor)
    }
}

/// Request-bounds and refusal-policy limits for the combined gRPC + Flight
/// SQL surface: inbound message size, in-flight request concurrency (global
/// and per TCP connection), an optional per-unary-request timeout, and the
/// two long-lived-stream budgets (`TriggerService.Subscribe`,
/// `JobService.WaitJob`). A request that would exceed any of these is
/// refused at the edge — before a tenant-scoped catalog read ever runs, so a
/// refusal leaks nothing about cross-tenant existence — with a typed gRPC
/// status and a `jammi_grpc_refused_total{reason}` counter increment. See
/// `jammi_server::limits` (the crate that owns the tower layer stack
/// enforcing this) for the wire contract.
///
/// # TOML
///
/// ```toml
/// [server.limits]
/// max_message_bytes = 67108864
/// max_in_flight = 256
/// max_in_flight_per_connection = 64
/// request_timeout_secs = 30
/// wait_timeout_secs = 300
/// max_subscriptions = 256
/// max_job_waits = 1024
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LimitsConfig {
    /// Maximum size, in bytes, of a single INBOUND gRPC/Flight message this
    /// server will decode — enforced per mounted service via tonic's own
    /// `max_decoding_message_size`. There is no outbound cap: a large result
    /// set is never truncated. Must be `> 0`. Default: 64 MiB (67108864).
    pub max_message_bytes: u64,
    /// Global cap on the number of unary requests this process serves
    /// concurrently, across every connection. `0` means unbounded — not
    /// "refuse everything". Default: 256.
    pub max_in_flight: usize,
    /// Cap on the number of unary requests a SINGLE TCP connection may have
    /// in flight concurrently. `0` means unbounded. When both this and
    /// `max_in_flight` are non-zero (bounded), this must be `<=
    /// max_in_flight` — a single connection is never permitted a larger
    /// budget than the process-wide one. Default: 64.
    pub max_in_flight_per_connection: usize,
    /// Maximum wall-clock duration a unary request may run before this
    /// server cancels it and returns `DEADLINE_EXCEEDED`. `None` (the
    /// default — the key absent or explicitly unset) means no
    /// server-imposed timeout. Unary methods only: the two server-streaming
    /// RPCs (`TriggerService.Subscribe`, `JobService.WaitJob`) are governed
    /// by `wait_timeout_secs` and the stream budgets below instead, never
    /// this key. `Some(0)` is rejected at load — a zero timeout would refuse
    /// every request instantly, never the intent of setting this key.
    pub request_timeout_secs: Option<u64>,
    /// Bounds a `TriggerService.Subscribe` or `JobService.WaitJob` stream.
    /// The SERVER budget bounds the stream; the client imposes no deadline of
    /// its own by default (`jammi_client::DataClient::wait_job`/`subscribe`
    /// send no `grpc-timeout` header). Three arms:
    ///
    /// * a `grpc-timeout` header ABOVE this budget is refused at the edge —
    ///   before the stream ever opens — with `DEADLINE_EXCEEDED`.
    /// * a `grpc-timeout` header WITHIN this budget is ENFORCED by the
    ///   server itself, at the caller's own declared deadline: the stream
    ///   ends with `DEADLINE_EXCEEDED` once that (shorter) duration elapses,
    ///   never the wider budget (`jammi_server::limits::PermitBody::Deadlined`).
    ///   This is deliberate, not merely "honoured as-is": tonic's own
    ///   `GrpcTimeout` middleware races only the service future, never a
    ///   streaming response body already returned, so a caller that declared
    ///   a deadline and then ignored it would otherwise hold this stream's
    ///   permit forever.
    /// * NO `grpc-timeout` header at all (HTTP/2's own no-deadline default —
    ///   the shape a header-less, e.g. Python-shaped, client sends) is NOT
    ///   refused: this budget itself becomes the stream's deadline, ending
    ///   it with `DEADLINE_EXCEEDED` once it elapses, wherever the stream
    ///   then stands (`jammi_server::limits::PermitBody::Deadlined`).
    ///
    /// `None` (the default) means no cap: a stream runs until terminal
    /// (`WaitJob`) or indefinitely (`Subscribe`), regardless of what a caller
    /// does or does not declare. `Some(0)` is rejected at load, for the same
    /// reason as `request_timeout_secs`.
    pub wait_timeout_secs: Option<u64>,
    /// Cap on the number of concurrently open `TriggerService.Subscribe`
    /// streams this process serves. `0` means unbounded. Default: 256.
    pub max_subscriptions: usize,
    /// Cap on the number of concurrently open `JobService.WaitJob` streams
    /// this process serves. `0` means unbounded. Default: 1024.
    pub max_job_waits: usize,
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            max_message_bytes: 64 * 1024 * 1024,
            max_in_flight: 256,
            max_in_flight_per_connection: 64,
            request_timeout_secs: None,
            wait_timeout_secs: None,
            max_subscriptions: 256,
            max_job_waits: 1024,
        }
    }
}

impl LimitsConfig {
    /// Validate the domain of every knob, naming the offending key in the
    /// error. `0` is a valid, meaningful value for the four concurrency/
    /// budget knobs (`max_in_flight`, `max_in_flight_per_connection`,
    /// `max_subscriptions`, `max_job_waits`) — it means unbounded — so only
    /// the CROSS-knob relation (`max_in_flight_per_connection <=
    /// max_in_flight`, when both are bounded) and the two knobs where `0`
    /// (or an explicit zero timeout) has no sane reading
    /// (`max_message_bytes`, `request_timeout_secs`, `wait_timeout_secs`)
    /// are rejected. Negative values and TOML integers that overflow the
    /// unsigned field types are already refused by `serde`/`toml` below this
    /// call, naming the same key, before this method ever runs.
    pub fn validate(&self) -> Result<()> {
        if self.max_message_bytes == 0 {
            return Err(JammiError::Config(
                "server.limits.max_message_bytes must be > 0".into(),
            ));
        }
        if self.max_in_flight_per_connection != 0
            && self.max_in_flight != 0
            && self.max_in_flight_per_connection > self.max_in_flight
        {
            return Err(JammiError::Config(format!(
                "server.limits.max_in_flight_per_connection ({}) must be <= \
                 server.limits.max_in_flight ({}) when both are bounded",
                self.max_in_flight_per_connection, self.max_in_flight
            )));
        }
        if self.request_timeout_secs == Some(0) {
            return Err(JammiError::Config(
                "server.limits.request_timeout_secs must be > 0 when set".into(),
            ));
        }
        if self.wait_timeout_secs == Some(0) {
            return Err(JammiError::Config(
                "server.limits.wait_timeout_secs must be > 0 when set".into(),
            ));
        }
        Ok(())
    }
}

/// One `[server] preload_models` entry: `"id"` (task from the `models`
/// row) or `{ id = "…", task = "text_embedding" }` (explicit task —
/// required for a `local:` path, which has no row). Deserialized from either
/// shape by hand so the layer's typed-error contract holds under
/// `deny_unknown_fields`: an unknown key or task token is a load-time
/// `JammiError::Config` naming it, never a silent default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreloadEntry {
    /// The model id (`local:<path>`, an HF repo id, or a catalog model name).
    pub id: String,
    /// The task to load under; `None` = resolve from the `models` row.
    pub task: Option<crate::model_task::ModelTask>,
}

impl<'de> Deserialize<'de> for PreloadEntry {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct PreloadEntryVisitor;

        impl<'de> serde::de::Visitor<'de> for PreloadEntryVisitor {
            type Value = PreloadEntry;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a model id string or { id, task }")
            }

            fn visit_str<E: serde::de::Error>(
                self,
                v: &str,
            ) -> std::result::Result<PreloadEntry, E> {
                if v.trim().is_empty() {
                    return Err(E::custom("preload_models: a model id must not be empty"));
                }
                Ok(PreloadEntry {
                    id: v.to_string(),
                    task: None,
                })
            }

            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<PreloadEntry, M::Error> {
                let mut id: Option<String> = None;
                let mut task: Option<crate::model_task::ModelTask> = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "id" => {
                            if id.is_some() {
                                return Err(serde::de::Error::duplicate_field("id"));
                            }
                            id = Some(map.next_value()?);
                        }
                        "task" => {
                            if task.is_some() {
                                return Err(serde::de::Error::duplicate_field("task"));
                            }
                            let token: String = map.next_value()?;
                            task = Some(
                                crate::model_task::ModelTask::try_from_db_str(&token).map_err(
                                    |e| {
                                        serde::de::Error::custom(format!(
                                            "preload_models: unknown task `{token}`: {e}"
                                        ))
                                    },
                                )?,
                            );
                        }
                        other => {
                            return Err(serde::de::Error::unknown_field(other, &["id", "task"]));
                        }
                    }
                }
                let id = id.ok_or_else(|| serde::de::Error::missing_field("id"))?;
                if id.trim().is_empty() {
                    return Err(serde::de::Error::custom(
                        "preload_models: a model id must not be empty",
                    ));
                }
                Ok(PreloadEntry { id, task })
            }
        }

        deserializer.deserialize_any(PreloadEntryVisitor)
    }
}

impl ServerConfig {
    /// Validate server configuration.
    pub fn validate(&self) -> Result<()> {
        use std::net::SocketAddr;

        let health: SocketAddr = self.health_listen.parse().map_err(|e| {
            crate::error::JammiError::Config(format!(
                "Invalid health_listen address '{}': {e}",
                self.health_listen
            ))
        })?;
        let flight: SocketAddr = self.flight_listen.parse().map_err(|e| {
            crate::error::JammiError::Config(format!(
                "Invalid flight_listen address '{}': {e}",
                self.flight_listen
            ))
        })?;
        // Two surfaces must not bind the same concrete address. An ephemeral
        // (`:0`) request never collides — the kernel assigns each bind a distinct
        // free port — so identical `:0` addresses are allowed; only identical
        // FIXED addresses would land both surfaces on one port.
        if health == flight && health.port() != 0 {
            return Err(crate::error::JammiError::Config(
                "health_listen and flight_listen must be different addresses".into(),
            ));
        }
        // The third listener, when set, joins the same fixed-address rule
        // against BOTH of the others (a 3-way check).
        if let Some(raw) = &self.peer_bind {
            let peer: SocketAddr = raw.parse().map_err(|e| {
                crate::error::JammiError::Config(format!("Invalid peer_bind address '{raw}': {e}"))
            })?;
            if peer.port() != 0 {
                if peer == flight {
                    return Err(crate::error::JammiError::Config(
                        "peer_bind and flight_listen must be different addresses".into(),
                    ));
                }
                if peer == health {
                    return Err(crate::error::JammiError::Config(
                        "peer_bind and health_listen must be different addresses".into(),
                    ));
                }
            }
        }
        if self.peer_local_load_bytes == Some(0) {
            return Err(crate::error::JammiError::Config(
                "server.peer_local_load_bytes must be > 0 when set (unset = unbounded)".into(),
            ));
        }
        self.limits.validate()?;
        Ok(())
    }
}

/// Tracing/logging configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoggingConfig {
    /// Log level filter (e.g., `"info"`, `"debug"`, `"warn"`). Default: `"info"`.
    pub level: String,
    /// Output format. Default: `Text`.
    pub format: LogFormat,
}

/// Vendor-neutral OTLP trace export (#486): where to send spans, the request
/// headers the exporter attaches, the resource's `service.name`, and the
/// fraction of traces to keep.
///
/// `jammi-db` carries this raw, typed section only — the exporter itself
/// (`jammi_ai::telemetry::otlp_layer`, gated behind the `telemetry-otlp`
/// cargo feature) lives in `jammi-ai`, mirroring [`ModelsConfig::hub_token`]'s
/// split (H4): a header value stays an unresolved [`SecretSource`] here
/// rather than an eagerly-read [`Secret`], because resolving it is the
/// exporter's job at the point it actually builds the gRPC metadata a
/// resolved value never needs to exist before that.
///
/// # TOML
///
/// ```toml
/// [observability]
/// otlp_endpoint = "http://localhost:4317"
/// service_name = "jammi"
/// sample_ratio = 1.0
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ObservabilityConfig {
    /// The OTLP/gRPC collector endpoint spans export to (e.g.
    /// `http://localhost:4317`). `None` (the default) means: build no
    /// exporter and open no connection — a process with no configured
    /// endpoint attempts zero network egress for tracing, regardless of
    /// whether the `telemetry-otlp` feature is compiled in.
    pub otlp_endpoint: Option<String>,
    /// Request headers the exporter attaches to every export call (e.g. an
    /// auth token a collector requires). Each value is an unresolved
    /// [`SecretSource`] — read at the point the exporter builds the gRPC
    /// metadata, not at config load — so a value is never eagerly resolved
    /// (and never logged: [`SecretSource`]'s own `Debug` redacts it, and
    /// this map holds sources, never resolved [`Secret`] text). Default:
    /// empty.
    pub otlp_headers: BTreeMap<String, SecretSource>,
    /// `service.name` resource attribute stamped on every exported span.
    /// Default: `"jammi"`.
    pub service_name: String,
    /// Fraction of traces kept by the parent-based ratio sampler, in
    /// `[0.0, 1.0]`. `1.0` (the default) samples everything; `0.0` samples
    /// nothing (but the exporter still installs — set `otlp_endpoint` to
    /// `None` instead to skip the exporter entirely). A root span always
    /// defers to this ratio; a span with a sampled remote parent is always
    /// kept, honouring the incoming decision (the "parent-based" half).
    pub sample_ratio: f64,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            otlp_endpoint: None,
            otlp_headers: BTreeMap::new(),
            service_name: "jammi".into(),
            sample_ratio: 1.0,
        }
    }
}

impl ObservabilityConfig {
    /// Validate the domain of every knob that can be checked without
    /// resolving a header (headers stay unresolved here — see the struct
    /// docs): `sample_ratio` must be a FINITE value within `[0.0, 1.0]`
    /// (`RangeInclusive::contains`'s `<=`/`>=` comparisons are false against
    /// a NaN operand on either side, so a NaN ratio already falls into this
    /// branch — it is never silently treated as in-range); a configured
    /// `otlp_endpoint` must parse as a URL with an `http`/`https` scheme and
    /// a host, naming the offending value rather than surfacing as an opaque
    /// exporter-construction failure deep in `jammi_ai::telemetry::otlp_layer`.
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.sample_ratio) {
            return Err(JammiError::Config(format!(
                "observability.sample_ratio = {} must be within [0.0, 1.0]",
                self.sample_ratio
            )));
        }
        if let Some(endpoint) = &self.otlp_endpoint {
            let parsed = url::Url::parse(endpoint).map_err(|e| {
                JammiError::Config(format!(
                    "observability.otlp_endpoint '{endpoint}' is not a valid URL: {e}"
                ))
            })?;
            if !matches!(parsed.scheme(), "http" | "https") {
                return Err(JammiError::Config(format!(
                    "observability.otlp_endpoint '{endpoint}' must use the http or https \
                     scheme, got '{}'",
                    parsed.scheme()
                )));
            }
            if parsed.host_str().is_none() {
                return Err(JammiError::Config(format!(
                    "observability.otlp_endpoint '{endpoint}' must name a host"
                )));
            }
        }
        Ok(())
    }
}

/// Model source: where the Hugging Face Hub cache lives, which endpoint and
/// token it talks to, and whether the process may reach the network at all.
///
/// `jammi-db` only carries this raw, typed selection — the `jammi-ai`
/// session choke point builds it into one `HubSource` (root resolution,
/// `ApiBuilder::from_cache`, endpoint/token fallback chain) exactly once per
/// session.
///
/// # TOML
///
/// ```toml
/// [models]
/// hub_endpoint = "https://huggingface.co"
/// hub_cache_dir = "/var/cache/jammi"
/// hub_token = { file = "/run/secrets/hf-token" }
/// offline = false
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelsConfig {
    /// Hub API endpoint. `None` → config default, then a non-empty,
    /// TRIMMED `HF_ENDPOINT` (a present-but-empty `HF_ENDPOINT` is treated
    /// the same as unset — see `jammi-ai`'s `model::hub` module docs,
    /// "empty values are absent, and every value is trimmed"), then the
    /// Hub's own default.
    pub hub_endpoint: Option<String>,
    /// Root directory the Hub cache lives under (a `hub/` subdirectory is
    /// appended). `None` → a non-empty `HF_HUB_CACHE` (used AS the cache
    /// root directly, nothing appended, matching `huggingface_hub`'s own
    /// convention), then a non-empty `HF_HOME` (`hub/` appended), then
    /// `directories::BaseDirs::home_dir()/.cache/huggingface` (`hub/`
    /// appended). A present-but-empty `HF_HUB_CACHE`/`HF_HOME` is treated
    /// the same as unset, never as a literal empty/CWD-relative root.
    pub hub_cache_dir: Option<PathBuf>,
    /// Hub bearer token. Kept as an unresolved [`SecretSource`] — not
    /// eagerly resolved into a [`Secret`] at config load — because the
    /// fallback chain (a non-empty `HF_TOKEN`, then a non-empty
    /// `HUGGING_FACE_HUB_TOKEN` — `huggingface_hub`'s own live legacy alias,
    /// `utils/_auth.py:145-147` — then the token file) is read at the
    /// `jammi-ai` session choke point, not here (H4). The token FILE is
    /// `HF_TOKEN_PATH`, when non-empty, naming the file directly (matching
    /// `huggingface_hub`'s own `HF_TOKEN_PATH`, `constants.py:247-254`);
    /// otherwise `<HF_HOME>/token` — `HF_HOME` resolved on its own,
    /// independently of whichever tier won `hub_cache_dir`'s own precedence
    /// above — never derived from the cache root itself (a
    /// `HF_HUB_CACHE`-driven cache root has no `hub` path component to pop
    /// the way a `<HF_HOME>/hub`-shaped root does).
    pub hub_token: Option<SecretSource>,
    /// Refuse every network fetch: a model loads only from `local:` or an
    /// already-resolved catalog row. `Some(_)` wins outright over the
    /// `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` environment variables — set
    /// `Some(false)` explicitly (a literal `offline = false` in the TOML) to
    /// force online even when one of them is set in the process
    /// environment; an OMITTED field (`None`, the `#[serde(default)]`
    /// value) falls back to a non-empty `HF_HUB_OFFLINE`, then — only when
    /// `HF_HUB_OFFLINE` is itself unset OR present-but-empty (treated
    /// identically — see `jammi-ai`'s `model::hub` module docs, "empty
    /// values are absent, and every value is trimmed"; a WHITESPACE-only
    /// `HF_HUB_OFFLINE` is one case where this crate's own reading
    /// disclosably diverges from `huggingface_hub`'s, in the safe
    /// direction — see that same section) — to `TRANSFORMERS_OFFLINE` (`huggingface_hub`'s
    /// own alias for this variable), then to `false`. `Option<bool>`, not a
    /// plain `bool`, is what makes "explicitly set to false" distinguishable
    /// from "never mentioned" — the same reason the other three fields
    /// above are already `Option`-typed. See `jammi-ai`'s
    /// `model::hub::HubSource::from_config` (a downstream crate — not
    /// linkable from here) for the resolution this drives and the accepted
    /// `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` truthy values
    /// (`huggingface_hub`'s own `ENV_VARS_TRUE_VALUES`: `"1"`, `"on"`,
    /// `"yes"`, `"true"`, case-insensitively).
    pub offline: Option<bool>,
}

// --- Defaults ---

fn default_artifact_dir() -> PathBuf {
    directories::ProjectDirs::from("ai", "jammi", "jammi")
        .map(|d| d.data_local_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from(".jammi"))
}

impl Default for JammiConfig {
    fn default() -> Self {
        Self {
            artifact_dir: default_artifact_dir(),
            engine: EngineConfig::default(),
            gpu: GpuConfig::default(),
            inference: InferenceConfig::default(),
            embedding: EmbeddingConfig::default(),
            fine_tuning: FineTuningConfig::default(),
            lease: LeaseConfig::default(),
            worker: WorkerConfig::default(),
            jobs: JobsConfig::default(),
            cache: CacheConfig::default(),
            server: ServerConfig::default(),
            logging: LoggingConfig::default(),
            observability: ObservabilityConfig::default(),
            catalog: CatalogConfig::default(),
            broker: BrokerConfig::default(),
            signing_key: SigningKeyConfig::default(),
            storage: StorageConfig::default(),
            models: ModelsConfig::default(),
        }
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            execution_threads: num_cpus(),
            memory_limit: "75%".into(),
            batch_size: 8192,
        }
    }
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device: 0,
            memory_limit: "auto".into(),
            memory_fraction: 0.9,
            require_gpu: false,
            compute_precision: jammi_numerics::ComputePrecision::F32,
        }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            default_backend: BackendSelection::Auto,
            batch_size: 32,
            batch_timeout_secs: 300,
            max_loaded_models: 0,
            http: HttpConfig::default(),
        }
    }
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            timeout_secs: 60,
            headers: BTreeMap::new(),
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            default_distance_metric: DistanceMetric::Cosine,
            default_index_type: IndexType::IvfHnswSq,
            checkpoint_interval: 1000,
            ann: AnnIndexConfig::default(),
        }
    }
}

impl Default for FineTuningConfig {
    fn default() -> Self {
        Self {
            default_lora_rank: 8,
            default_learning_rate: 0.0002,
            default_epochs: 3,
            default_batch_size: 8,
            checkpoint_fraction: 0.1,
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            ann_cache_enabled: true,
            ann_cache_max_entries: 10000,
            embedding_cache_enabled: true,
            embedding_cache_size: "1GB".into(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            health_listen: "0.0.0.0:8080".into(),
            flight_listen: "0.0.0.0:8081".into(),
            preload_models: Vec::new(),
            services: ServiceSelection::default(),
            limits: LimitsConfig::default(),
            peer_bind: None,
            peer_local_load_bytes: None,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".into(),
            format: LogFormat::Text,
        }
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

// --- Loading ---

/// The filesystem roots [`resolve_config_path_in`] probes, injected so a test
/// can point every step at a tempdir instead of the real `/etc` or platform
/// config directory (H15).
pub(crate) struct ConfigRoots {
    /// The "current directory" root — production passes `.`.
    pub cwd: PathBuf,
    /// The system-wide config directory — production passes `/etc/jammi`.
    pub etc_dir: PathBuf,
    /// The platform per-user config directory
    /// (`directories::ProjectDirs::config_dir()`), when the platform exposes
    /// one.
    pub platform_dir: Option<PathBuf>,
}

impl ConfigRoots {
    fn production() -> Self {
        Self {
            cwd: PathBuf::from("."),
            etc_dir: PathBuf::from("/etc/jammi"),
            platform_dir: directories::ProjectDirs::from("ai", "jammi", "jammi")
                .map(|d| d.config_dir().to_path_buf()),
        }
    }
}

/// Resolve the config file path (H6 order): an explicit path that exists,
/// then `env["JAMMI_CONFIG"]`, then `{roots.cwd}/jammi.toml`, then
/// `{roots.etc_dir}/jammi.toml`, then `{roots.platform_dir}/config.toml`.
/// `env` is the same map [`JammiConfig::load_from`]/[`JammiConfig::parse_from`]
/// were handed — this function reads no process env of its own.
pub(crate) fn resolve_config_path_in(
    explicit: Option<&Path>,
    roots: &ConfigRoots,
    env: &BTreeMap<String, String>,
) -> Option<PathBuf> {
    if let Some(p) = explicit {
        if p.exists() {
            return Some(p.to_path_buf());
        }
    }
    if let Some(env_path) = env.get("JAMMI_CONFIG") {
        let p = PathBuf::from(env_path);
        if p.exists() {
            return Some(p);
        }
    }
    let cwd = roots.cwd.join("jammi.toml");
    if cwd.exists() {
        return Some(cwd);
    }
    let etc = roots.etc_dir.join("jammi.toml");
    if etc.exists() {
        return Some(etc);
    }
    if let Some(dir) = &roots.platform_dir {
        let p = dir.join("config.toml");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Turn a `serde_path_to_error` failure over the [`Node`] tree into a
/// [`JammiError::Config`] that names both the offending struct path (R7) and
/// every `JAMMI_*` variable whose path has that same prefix — the
/// provenance rule: the variable(s) named come from the merged tree itself,
/// never guessed independently of it.
fn describe_deserialize_error(
    err: serde_path_to_error::Error<layers::NodeError>,
    env_vars_present: &[String],
) -> JammiError {
    let path = err.path().to_string();
    let inner = err.into_inner().0;
    if path.is_empty() || path == "." {
        return JammiError::Config(inner);
    }
    let env_prefix = format!("JAMMI_{}", path.replace('.', "__").to_uppercase());
    let named: Vec<&str> = env_vars_present
        .iter()
        .map(String::as_str)
        .filter(|v| v.starts_with(&env_prefix) || env_prefix.starts_with(v))
        .collect();
    if named.is_empty() {
        JammiError::Config(format!("{path}: {inner}"))
    } else {
        JammiError::Config(format!("{path}: {inner} (env: {})", named.join(", ")))
    }
}

impl JammiConfig {
    /// Load configuration the production way: resolve the file (explicit
    /// path, `JAMMI_CONFIG`, `./jammi.toml`, `/etc/jammi/jammi.toml`, the
    /// platform config dir — `resolve_config_path_in`) against the real
    /// process environment and filesystem roots, then [`Self::load_from`].
    pub fn load(path: Option<&Path>) -> Result<Self> {
        Self::load_from(path, std::env::vars())
    }

    /// The real implementation: resolve `file` against the production
    /// filesystem roots using `env` (never `std::env` directly — `env` is
    /// the single source both `JAMMI_CONFIG` resolution and every `JAMMI_*`
    /// override read from), read it if found, [`Self::parse_from`] it, then
    /// run post-load validation (`storage.cloud.validate()`, the training
    /// worker-interval invariants).
    pub fn load_from(
        file: Option<&Path>,
        env: impl IntoIterator<Item = (String, String)>,
    ) -> Result<Self> {
        let env_map: BTreeMap<String, String> = env.into_iter().collect();
        let roots = ConfigRoots::production();
        let path = resolve_config_path_in(file, &roots, &env_map);
        let contents = match &path {
            Some(p) => std::fs::read_to_string(p)?,
            None => String::new(),
        };
        let config = Self::parse_from(&contents, env_map)?;
        // Catch a partial cloud-credential set (e.g. an R2 `access_key_id`
        // without its `secret_access_key`) at load time rather than deep
        // inside the first object-store request.
        if let Some(cloud) = &config.storage.cloud {
            cloud.validate()?;
        }
        // Reject a lease timing that violates the heartbeat margin, or a
        // worker poll that is a busy-loop, at load time rather than at
        // worker spawn, deep in a server startup.
        let lease = config.lease.intervals()?;
        config.worker.worker_intervals(lease)?;
        // Reject a retention window past the cap at load, not in the first
        // sweep's timestamp arithmetic.
        config.jobs.validate()?;
        // Reject an out-of-domain `[server.limits]` knob (a zero
        // `max_message_bytes`, a per-connection budget over the global one, a
        // zero timeout) at load time, naming the offending key, rather than
        // at server startup deep inside `OssServer::new`.
        config.server.limits.validate()?;
        // Reject an out-of-domain `[observability]` knob (a `sample_ratio`
        // outside `[0.0, 1.0]`, including NaN, or a malformed/non-http(s)
        // `otlp_endpoint`) at load time, naming the offending key, rather
        // than at the first `jammi_ai::telemetry::otlp_layer` call.
        config.observability.validate()?;
        Ok(config)
    }

    /// The parse-only core: interpolate `${VAR}` from `env`,
    /// parse the TOML file layer, build the `JAMMI_*` env layer from the
    /// SAME `env` map (T5's namespace rule — `env_map::build_env_layer`),
    /// deep-merge the two (`layers::merge` — H1), and deserialize the
    /// whole typed [`JammiConfig`] from the merged tree in one pass via
    /// `serde_path_to_error` (R7: every error names the struct path and the
    /// variable(s) at it). No post-load validation — that is
    /// [`Self::load_from`]'s job, so a content oracle can call this directly
    /// without also pinning `storage.cloud.validate()`/worker-interval
    /// behaviour.
    pub fn parse_from(
        toml_src: &str,
        env: impl IntoIterator<Item = (String, String)>,
    ) -> Result<Self> {
        let env_map: BTreeMap<String, String> = env.into_iter().collect();
        let interpolated = interpolate_env_vars(toml_src, |name| env_map.get(name).cloned())?;
        // `interpolated` is the file text AFTER `${VAR}` expansion — an
        // unquoted `url = ${POSTGRES_URL}` puts the secret straight into
        // this text, so a parse error here must never render via
        // `Display`/`to_string()` (which quotes a code frame of the
        // offending source line verbatim). `layers::describe_toml_error`
        // renders `message()` plus a safe line:column locator instead.
        let file_value: toml::Value = interpolated.parse().map_err(|e: toml::de::Error| {
            JammiError::Config(format!(
                "invalid TOML: {}",
                layers::describe_toml_error(&interpolated, &e)
            ))
        })?;
        let file_node = Node::from_toml(file_value);
        let env_node =
            env_map::build_env_layer(env_map.iter().map(|(k, v)| (k.clone(), v.clone())))
                .map_err(|e| JammiError::Config(e.0))?;
        let mut env_vars_present = Vec::new();
        env_node.vars(&mut env_vars_present);
        let merged = layers::merge(file_node, env_node);
        serde_path_to_error::deserialize(merged)
            .map_err(|e| describe_deserialize_error(e, &env_vars_present))
    }
}

/// Substitute `${VAR}` patterns in `input`, resolved through `lookup` (H8) —
/// never `std::env` directly, so the loader's env-reading is a single,
/// explicit seam.
///
/// Rules:
/// - `${NAME}` is replaced by `lookup(NAME)`. A name must start with
///   `[A-Za-z_]` and continue with `[A-Za-z0-9_]`.
/// - A missing variable (`lookup` returns `None`) returns
///   [`JammiError::Config`]. The loader does **not** silently substitute an
///   empty string — that is a common source of "deployed config has empty
///   Postgres URL" outages.
/// - `$$` escapes a literal `$`.
/// - A bare `$` not followed by `$` or `{` is preserved verbatim (lets the
///   raw `$` in a TOML password slip through unchanged).
/// - An unterminated `${` returns [`JammiError::Config`].
/// - Interpolation is one-pass and not recursive: the value of `${X}` is not
///   re-scanned.
pub fn interpolate_env_vars(
    input: &str,
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<String> {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b != b'$' {
            // `bytes[i]` came from `input.as_bytes()` and indexing is at a
            // char boundary at this point: every previous step either copied
            // exactly one ASCII byte (`$`, `{`, `}`, or a variable-name
            // character) or copied a whole UTF-8 substring from `input` via
            // `&input[..]`. The non-ASCII branch below preserves boundaries.
            if b < 0x80 {
                out.push(b as char);
                i += 1;
            } else {
                // Non-ASCII byte: scan to the next ASCII char or `$` and
                // copy the slice in one go so we never split a code point.
                let start = i;
                while i < bytes.len() && bytes[i] >= 0x80 {
                    i += 1;
                }
                out.push_str(&input[start..i]);
            }
            continue;
        }

        // We saw a `$`. Peek the next byte.
        let next = bytes.get(i + 1).copied();
        match next {
            Some(b'$') => {
                out.push('$');
                i += 2;
            }
            Some(b'{') => {
                let name_start = i + 2;
                let close = bytes[name_start..]
                    .iter()
                    .position(|&c| c == b'}')
                    .map(|off| name_start + off)
                    .ok_or_else(|| {
                        JammiError::Config(format!(
                            "Unterminated env-var reference `${{` at offset {i}"
                        ))
                    })?;
                let name = &input[name_start..close];
                if !is_valid_env_name(name) {
                    return Err(JammiError::Config(format!(
                        "Invalid env-var name `${{{name}}}` at offset {i}: \
                         names must match [A-Za-z_][A-Za-z0-9_]*"
                    )));
                }
                let value = lookup(name).ok_or_else(|| {
                    JammiError::Config(format!("Env var `{name}` referenced by config is not set"))
                })?;
                out.push_str(&value);
                i = close + 1;
            }
            // A bare `$` (end of input or followed by anything else): keep
            // the literal `$` so escaped passwords containing one `$` do
            // not trip the loader.
            _ => {
                out.push('$');
                i += 1;
            }
        }
    }
    Ok(out)
}

fn is_valid_env_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Parse a boolean environment-variable override. Accepts `true`/`false` and
/// the numeric spellings `1`/`0`, case-insensitively and with surrounding
/// whitespace trimmed — the four spellings a shell, a container manifest, or
/// an orchestrator template actually emits. Kept for the pinned unit tests
/// below; the production path (any `bool`-typed field, e.g.
/// `training.run_worker`) reaches the identical rule through
/// [`layers::parse_lenient_bool`], which this wraps.
#[cfg(test)]
fn parse_env_bool(var: &str, raw: &str) -> Result<bool> {
    layers::parse_lenient_bool(raw).ok_or_else(|| {
        JammiError::Config(format!(
            "Invalid boolean '{raw}' for {var}. Expected: true, false, 1, 0"
        ))
    })
}
