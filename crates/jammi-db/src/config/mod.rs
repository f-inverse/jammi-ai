use std::collections::BTreeMap;
use std::fmt;
use std::num::NonZeroUsize;

use jammi_numerics::ChunkBudget;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

use serde::{Deserialize, Deserializer, Serialize};

pub use crate::catalog::lease::LeaseIntervals;
use crate::error::{JammiError, Result};
use crate::storage::{AzureConfig, CloudConfig, GcsConfig, R2Config, S3Config};

mod env_map;
pub mod host_memory;
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

/// Which collective a multi-rank worker reduces gradients over.
///
/// This is *configuration*, not a build feature: the same binary answers
/// `auto`, `nccl` and `cpu`, and no variant selects a code path at compile
/// time. The vocabulary is fixed here; the layer that owns communicators
/// resolves `Auto` against what the process can actually reach (a CUDA
/// build with visible devices, or the host reduction otherwise) and refuses
/// [`Self::Nccl`] on a build without CUDA — that refusal is a runtime
/// predicate at session open, never a `#[cfg]` in this crate, which cannot
/// see another crate's features.
///
/// # TOML
///
/// ```toml
/// [worker]
/// collective = "auto"   # auto | nccl | cpu
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CollectiveSelection {
    /// Pick the best collective this process can actually reach: NCCL on a
    /// CUDA build with enough visible devices, the host reduction
    /// otherwise. Degrades rather than refuses. Default.
    #[default]
    Auto,
    /// Require NCCL. A build without CUDA refuses to open rather than
    /// silently reducing on the host at a fraction of the throughput the
    /// deployment asked for.
    Nccl,
    /// Require the host reduction, even where NCCL is available — the
    /// deterministic, device-free path a hermetic run and a debugging
    /// session want.
    Cpu,
}

impl fmt::Display for CollectiveSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Nccl => write!(f, "nccl"),
            Self::Cpu => write!(f, "cpu"),
        }
    }
}

impl FromStr for CollectiveSelection {
    type Err = JammiError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "nccl" => Ok(Self::Nccl),
            "cpu" => Ok(Self::Cpu),
            other => Err(JammiError::Config(format!(
                "Unknown collective '{other}'. Expected: auto, nccl, cpu"
            ))),
        }
    }
}

impl CollectiveSelection {
    /// Whether this selection can only be honoured by a CUDA build.
    ///
    /// The `#[cfg]`-free half of the "`collective = \"nccl\"` on a build
    /// without CUDA is refused" rule: this crate states the *predicate* over
    /// the parsed value, and the layer that knows its own build features
    /// (the session, at open) applies it and raises the typed refusal.
    /// [`Self::Auto`] is `false` — it degrades to the host reduction — and
    /// so is [`Self::Cpu`], which never wants a device.
    pub fn requires_cuda(self) -> bool {
        matches!(self, Self::Nccl)
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
    /// The widest `Peer` gang any coordinator on this deployment may admit
    /// — loads independently of `[worker]`'s own per-host rank count; see
    /// [`DistributedConfig`].
    pub distributed: DistributedConfig,
    /// Job retention: how long a terminal `jobs` row survives the sweep.
    pub jobs: JobsConfig,
    /// Cache layer settings (ANN cache, embedding cache).
    pub cache: CacheConfig,
    /// HTTP and Arrow Flight server bind addresses.
    pub server: ServerConfig,
    /// Whether this process hosts a Ballista scheduler and/or executor
    /// role for the distributed compute plane. Default: neither role
    /// (today's process, byte-for-byte). See [`BallistaConfig`].
    pub ballista: BallistaConfig,
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
/// `gcs`, or `azure` — rather than a `kind` key inside one shared table;
/// a bare `storage.cloud = "s3"` also selects a variant with its
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
/// still reloads.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum CloudSection {
    S3(S3Section),
    R2(R2Section),
    Gcs(GcsSection),
    Azure(AzureSection),
}

/// Config-side mirror of [`crate::storage::S3Config`]. `secret_access_key`
/// and `session_token` are [`Secret`]-typed; `access_key_id` is not.
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
/// `account_key`/`sas_token`/`client_secret` are [`Secret`]-typed.
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
/// `deny_unknown_fields` shape) and convert to `Option<CloudConfig>`.
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
    /// The engine's CPU parallelism budget, the one setting that bounds all of
    /// it: DataFusion's partitions, the forwards a CPU device admits at once,
    /// and the process-wide pool candle's CPU math and the media front ends
    /// run on (which a binary sizes from this at startup; a process embedding
    /// the engine as a library owns that pool itself). Default: the CPU count
    /// the OS reports — set it where a container is allotted fewer cores than
    /// it can see.
    pub execution_threads: std::num::NonZeroUsize,
    /// Maximum memory for the query engine: `"<n>%"` (1-100) of host
    /// physical memory, `"<n>GB"`/`"<n>MB"`/`"<n>KB"` (binary units), or
    /// `"<n>"` (bytes). Default: `"75%"`. Parsed by
    /// [`Self::memory_limit_bytes`] — see its doc for the full grammar and
    /// refusals.
    pub memory_limit: String,
    /// Maximum rows per DataFusion batch. Default: 8192.
    pub batch_size: usize,
}

impl EngineConfig {
    /// Below this, a resolved `memory_limit` is refused at load: 64 MiB is small enough that
    /// DataFusion's own long-lived pool consumers (a `SortPreservingMergeExec`'s per-partition
    /// reservation, an external sorter's spill buffer) would be refused on the very first
    /// non-trivial query, before the setting ever bounds the workload it exists to bound.
    pub const MEMORY_LIMIT_FLOOR_BYTES: u64 = 64 * 1024 * 1024;

    /// Parse `[engine] memory_limit` into bytes — the ONE reader of the
    /// field; every consumer of the byte value (the session's
    /// [`crate::memory_pool::ActiveSpillPool`]) calls this,
    /// never the raw string.
    ///
    /// # Grammar
    ///
    /// - `"<n>%"`, `1 <= n <= 100`: that percentage of
    ///   [`host_memory::total_physical_memory_bytes`] (a Linux cgroup
    ///   ceiling honoured when it is lower than the host total and
    ///   readable), read once per call — not cached, so a caller that wants
    ///   ONE resolved value for a whole session's lifetime calls this once
    ///   and keeps the `u64`, the same discipline
    ///   `crate::session::JammiSession::build` follows.
    /// - `"<n>GB"` / `"<n>MB"` / `"<n>KB"`: `n` binary (1024-based) units.
    /// - `"<n>"`: `n` bytes, unadorned.
    ///
    /// # Refusals
    ///
    /// Every arm is a typed [`JammiError::Config`] naming the key and the
    /// configured value:
    ///
    /// - a percentage outside `1..=100`;
    /// - a form matching none of the three shapes above (an empty string, a
    ///   decimal, a stray unit with no digits, an unrecognised suffix, a
    ///   negative number);
    /// - a resolved value below [`Self::MEMORY_LIMIT_FLOOR_BYTES`] — the
    ///   floor also named in the message, so `"007"` (7 bytes, a
    ///   `[engine]` config typo for `"7%"` or similar) is refused rather than
    ///   silently building a 7-byte pool no query could ever run under.
    pub fn memory_limit_bytes(&self) -> Result<u64> {
        let raw = self.memory_limit.trim();
        let bytes = if let Some(pct) = raw.strip_suffix('%') {
            let pct: u64 = pct
                .parse()
                .map_err(|_| Self::memory_limit_grammar_error(&self.memory_limit))?;
            if !(1..=100).contains(&pct) {
                return Err(JammiError::Config(format!(
                    "[engine] memory_limit = {:?}: a percentage must be between 1 and 100",
                    self.memory_limit
                )));
            }
            let total = host_memory::total_physical_memory_bytes()?;
            total.saturating_mul(pct) / 100
        } else if let Some(n) = raw.strip_suffix("GB") {
            Self::parse_binary_unit(n, &self.memory_limit, 1024 * 1024 * 1024)?
        } else if let Some(n) = raw.strip_suffix("MB") {
            Self::parse_binary_unit(n, &self.memory_limit, 1024 * 1024)?
        } else if let Some(n) = raw.strip_suffix("KB") {
            Self::parse_binary_unit(n, &self.memory_limit, 1024)?
        } else {
            raw.parse::<u64>()
                .map_err(|_| Self::memory_limit_grammar_error(&self.memory_limit))?
        };
        if bytes < Self::MEMORY_LIMIT_FLOOR_BYTES {
            return Err(JammiError::Config(format!(
                "[engine] memory_limit = {:?} resolves to {bytes} byte(s), below the {} MiB \
                 floor (a smaller pool would refuse DataFusion's own long-lived reservations \
                 before it ever bounds a query)",
                self.memory_limit,
                Self::MEMORY_LIMIT_FLOOR_BYTES / (1024 * 1024)
            )));
        }
        Ok(bytes)
    }

    fn parse_binary_unit(digits: &str, raw: &str, unit: u64) -> Result<u64> {
        let n: u64 = digits
            .parse()
            .map_err(|_| Self::memory_limit_grammar_error(raw))?;
        Ok(n.saturating_mul(unit))
    }

    fn memory_limit_grammar_error(raw: &str) -> JammiError {
        JammiError::Config(format!(
            "[engine] memory_limit = {raw:?} is not a valid form: use \"<n>%\" (1-100), \
             \"<n>GB\"/\"<n>MB\"/\"<n>KB\", or \"<n>\" (bytes)"
        ))
    }
}

/// GPU device and memory settings.
///
/// `device` is the deployment's PRIMARY device and stays the single answer to
/// "which device does a one-device process run on". `devices` is the optional
/// plural: the ordered list a multi-rank worker places rank `i` on. The two
/// are one setting seen at two arities, so they are kept in agreement by
/// [`GpuConfig::validate`] rather than left to drift — a list whose first
/// entry is not `device` is a load error, never a silent second opinion about
/// the primary.
///
/// # TOML
///
/// ```toml
/// [gpu]
/// device = 0
/// devices = [0, 1]   # optional; unset = the one-device list `[device]`
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GpuConfig {
    /// CUDA device ordinal, and the PRIMARY of [`Self::devices`]. `-1` is the
    /// CPU. Default: 0.
    pub device: i32,
    /// The ordered device list a multi-rank worker places its ranks on: rank
    /// `i` runs on `devices[i]`.
    ///
    /// `None` — the key is absent — is NOT the empty list: it means "no
    /// plural was configured", and [`Self::device_list`] resolves it to the
    /// one-device list `[device]`, which is exactly the single-device
    /// deployment every reader of `device` already assumes. An explicit
    /// `devices = []` is a different statement — "run on no device at all" —
    /// and is refused at load, so the absent case can never be confused with
    /// a configured-empty one.
    ///
    /// The list is stored raw (unresolved) so a `GpuConfig` built in code
    /// with `..Default::default()` cannot hold a plural that contradicts the
    /// `device` its author set; the resolution lives in
    /// [`Self::device_list`], the one place both arities are reconciled.
    pub devices: Option<Vec<i32>>,
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

impl GpuConfig {
    /// The CPU device ordinal: `-1`, the value every backend already reads as
    /// "no CUDA device, run on the host".
    pub const CPU_DEVICE: i32 = -1;

    /// The resolved device list: the configured plural when one was given,
    /// and otherwise the one-device list `[device]`.
    ///
    /// This is the ONE reconciliation of the two arities — a caller never
    /// reads [`Self::devices`] directly to decide where to place work, so an
    /// absent plural and a one-device plural are indistinguishable downstream
    /// (they mean the same deployment), while an absent plural and an
    /// explicit empty one are not (the empty one does not load).
    pub fn device_list(&self) -> Vec<i32> {
        match &self.devices {
            Some(devices) => devices.clone(),
            None => vec![self.device],
        }
    }

    /// Validate the device configuration at load, refusing every list that
    /// names something no rank can be placed on. Returns a typed
    /// [`JammiError::Config`] naming the offending key.
    ///
    /// Two rules are about the plural itself:
    ///
    /// - an explicit `devices = []` — a deployment with nowhere to run, and
    ///   a different statement from omitting the key;
    /// - a first entry that is not `device` — the plural and the primary
    ///   disagree about which device is rank 0's, and guessing one of them is
    ///   how a "CPU-pinned" session ends up on a GPU.
    ///
    /// The rest are about the RESOLVED list ([`Self::device_list`]), so the
    /// same configuration gets the same verdict at either arity — `device =
    /// -5` and `devices = [-5]` describe one deployment and are refused
    /// alike:
    ///
    /// - an ordinal below [`Self::CPU_DEVICE`] — not a device;
    /// - a repeated ordinal — two ranks on one device is a placement mistake,
    ///   and it makes the `local_ranks <= devices` bound meaningless;
    /// - the CPU (`-1`) listed alongside real ordinals — one gang runs on one
    ///   kind of device, and a mixed list has no collective that spans it.
    pub fn validate(&self) -> Result<()> {
        if let Some(devices) = &self.devices {
            if devices.is_empty() {
                return Err(JammiError::Config(
                    "[gpu] devices must not be empty (omit the key for the single-device default)"
                        .into(),
                ));
            }
            if devices[0] != self.device {
                return Err(JammiError::Config(format!(
                    "[gpu] devices = {:?} disagrees with device = {}: the first entry is rank \
                     0's device and must equal `device` (set `device = {}` or list it first)",
                    devices, self.device, devices[0]
                )));
            }
        }
        let devices = self.device_list();
        // Name the key the value was actually written under, so the message
        // points at the line the operator has to edit.
        let key = |i: usize| match self.devices {
            Some(_) => format!("devices[{i}]"),
            None => "device".to_string(),
        };
        for (i, ordinal) in devices.iter().enumerate() {
            if *ordinal < Self::CPU_DEVICE {
                return Err(JammiError::Config(format!(
                    "[gpu] {} = {ordinal} is not a device ordinal ({} is the CPU and the \
                     smallest accepted value)",
                    key(i),
                    Self::CPU_DEVICE
                )));
            }
            if devices[..i].contains(ordinal) {
                return Err(JammiError::Config(format!(
                    "[gpu] devices = {devices:?} repeats device {ordinal}: one rank per device"
                )));
            }
        }
        if devices.len() > 1 && devices.contains(&Self::CPU_DEVICE) {
            return Err(JammiError::Config(format!(
                "[gpu] devices = {devices:?} mixes the CPU ({}) with device ordinals: a gang \
                 runs on one kind of device",
                Self::CPU_DEVICE
            )));
        }
        Ok(())
    }
}

/// Model inference defaults.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InferenceConfig {
    /// Backend selection strategy. Default: `Auto`.
    pub default_backend: BackendSelection,
    /// The most rows one model forward takes. A forward chunk is cut from
    /// the input ordered by row cost (token count, then key) under this cap
    /// and [`Self::batch_tokens`], whatever the fan-out. Default: 32.
    pub batch_size: usize,
    /// The most padded tokens one model forward takes: the rows of a chunk
    /// times the width they are padded to. A budget in tokens is what bounds
    /// a forward's activation memory — a row cap alone under-fills the
    /// device on short rows and over-fills it on long ones. A row longer
    /// than the budget still forwards alone. Default: 16384 (32 rows of a
    /// 512-token encoder).
    pub batch_tokens: usize,
    /// Seconds to wait before flushing an incomplete batch. Default: 300.
    pub batch_timeout_secs: u64,
    /// Maximum number of models held in memory simultaneously. 0 means unlimited. Default: 0.
    pub max_loaded_models: usize,
    /// The inference fan-out: how many partitions of one plan forward chunks
    /// concurrently — threads of one process, or tasks of a cluster when the
    /// plan is submitted to one. Written bytes are identical at every value:
    /// the rows a model forwards together are decided by the row costs and
    /// the chunk budget alone. Default: 1.
    pub partitions: usize,
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
    /// wider candidate pool than the other three precisions' cosine-ranked stage to recover
    /// comparable recall — measured recall@1/10 = 0.94/0.992 at oversample `32` (vs. 0.59 recall@1
    /// at the shared default of `4`), so `Binary` gets its own wider default while
    /// `F32`/`F16`/`Int8` keep `4`.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
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
    /// Rows per ANN segment of a written embedding table: the segments are
    /// consecutive runs of the table's rows at this budget, each built on
    /// its own thread as its rows are written, and a query fans out over
    /// them. Smaller segments build sooner and more in parallel; larger
    /// ones cost a query fewer graph searches. Default: 4096.
    pub index_segment_rows: NonZeroUsize,
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
/// local_ranks = 1
/// collective = "auto"
/// rank_timeout_secs = 120
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
    /// How many ranks THIS HOST places on its own `[gpu] devices` for a job
    /// it runs entirely in-process — one rank per device, rank `i` on
    /// `[gpu] devices[i]`. Must be `>= 1` (`1`, the default, is the
    /// single-rank deployment: no gang, no collective) and never more than
    /// the configured device count, both enforced by
    /// [`WorkerConfig::topology`] at load.
    ///
    /// Orthogonal to `[distributed]
    /// max_world_size` (the widest `Peer` gang across FLEET members a
    /// coordinator on this deployment may accept) and to the per-job
    /// `world_size` in `TrainingCommon` (identity-relevant, checked against
    /// `[distributed] max_world_size` at submit) — the three knobs load
    /// independently, with no cross-check between any pair.
    /// The claiming worker decides the layout from the job's `world_size`
    /// and this value alone: within it, every rank runs in-process over a
    /// `Local` gang; beyond it, this process is rank 0 of a `Peer` gang.
    pub local_ranks: u32,
    /// Which collective a multi-rank worker reduces over. Default: `auto`.
    /// Configuration, not a build feature — see [`CollectiveSelection`].
    pub collective: CollectiveSelection,
    /// How long a rank waits on its peers at a gang boundary before the wait
    /// is a failure. Must be `> 0` — a zero deadline expires before any peer
    /// can answer, turning every gang into an immediate failure. Default:
    /// 120.
    pub rank_timeout_secs: u64,
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
            // One rank on the primary device: an unconfigured deployment is
            // the single-process one it has always been, with no gang and no
            // collective to resolve.
            local_ranks: 1,
            collective: CollectiveSelection::Auto,
            rank_timeout_secs: 120,
        }
    }
}

/// The worker's job-kind selection: `All` (the default) claims every kind
/// compiled into this binary; `Only` claims exactly the named kinds (e.g.
/// `[]` for a claim loop that runs but claims nothing — equivalent to
/// `enabled = false` for the poll loop's own effect, but distinguishable in
/// `ListWorkers`). Deserializes through the identical hand-written grammar
/// [`ServiceSelection`] uses (`"all"`, a comma-separated list, or a TOML
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

    /// Resolve the validated [`WorkerTopology`] this `[worker]` section
    /// implies over `gpu`'s devices, refusing at load every width no
    /// placement exists for.
    ///
    /// The refusals, each typed ([`JammiError::Config`]) and naming its key:
    ///
    /// - `local_ranks == 0` — a deployment with no rank cannot run anything,
    ///   and `0` is not "unset" (the unset value is the default `1`);
    /// - `local_ranks > devices` — there is no device for the last rank, and
    ///   the alternative to refusing is two ranks silently sharing one;
    /// - `rank_timeout_secs == 0` — a deadline that has already passed.
    ///
    /// `gpu`'s own domain rules ([`GpuConfig::validate`]) are checked first,
    /// so the device count this bounds `local_ranks` against is a count of
    /// distinct, placeable devices.
    pub fn topology(&self, gpu: &GpuConfig) -> Result<WorkerTopology> {
        gpu.validate()?;
        let devices = gpu.device_list();
        if self.local_ranks == 0 {
            return Err(JammiError::Config(
                "[worker] local_ranks must be >= 1 (1 is the single-rank deployment; 0 has no \
                 rank to run on)"
                    .into(),
            ));
        }
        if self.local_ranks as usize > devices.len() {
            return Err(JammiError::Config(format!(
                "[worker] local_ranks = {} exceeds the {} configured device(s) {:?}: one rank \
                 per device, so list more in `[gpu] devices` or lower `local_ranks`",
                self.local_ranks,
                devices.len(),
                devices
            )));
        }
        if self.rank_timeout_secs == 0 {
            return Err(JammiError::Config(
                "[worker] rank_timeout_secs must be > 0 (a zero deadline expires before any \
                 peer can answer)"
                    .into(),
            ));
        }
        Ok(WorkerTopology {
            local_ranks: self.local_ranks,
            devices,
            collective: self.collective,
            rank_timeout: Duration::from_secs(self.rank_timeout_secs),
        })
    }
}

/// The validated rank topology a worker places work with: how many ranks this
/// deployment runs, which device each of them gets, which collective they
/// reduce over, and how long a rank waits at a gang boundary.
///
/// [`WorkerConfig::topology`] is the only constructor, so every instance has
/// already cleared the bounds: `local_ranks >= 1`, `local_ranks <=
/// devices.len()`, the devices distinct and placeable, and a non-zero
/// timeout. The fields are private for the same reason — the bounds hold for
/// the lifetime of the value, not just at the moment it was built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerTopology {
    local_ranks: u32,
    devices: Vec<i32>,
    collective: CollectiveSelection,
    rank_timeout: Duration,
}

impl WorkerTopology {
    /// How many ranks this deployment runs. Always `>= 1`.
    pub fn local_ranks(&self) -> u32 {
        self.local_ranks
    }

    /// Every configured device, in order — the full set a process opens
    /// per-device resources (a scheduler, a model cache) for, which may be
    /// wider than the gang.
    pub fn devices(&self) -> &[i32] {
        &self.devices
    }

    /// The devices the ranks of a gang occupy: the first
    /// [`Self::local_ranks`] entries of [`Self::devices`]. Never longer than
    /// the gang, so a caller cannot spawn a rank onto a device no rank owns.
    pub fn rank_devices(&self) -> &[i32] {
        &self.devices[..self.local_ranks as usize]
    }

    /// The device rank `rank` runs on, or `None` when `rank` is not a rank of
    /// this topology (`rank >= local_ranks`) — an out-of-range rank has no
    /// device, and saying so is not the same as handing back the primary.
    pub fn device_for_rank(&self, rank: u32) -> Option<i32> {
        (rank < self.local_ranks).then(|| self.devices[rank as usize])
    }

    /// The configured collective. `Auto` is still unresolved here: this layer
    /// records what was asked for, and the layer that owns communicators
    /// decides what `Auto` becomes.
    pub fn collective(&self) -> CollectiveSelection {
        self.collective
    }

    /// How long a rank waits on its peers at a gang boundary. Always `> 0`.
    pub fn rank_timeout(&self) -> Duration {
        self.rank_timeout
    }

    /// Whether this topology has more than one rank — the one question that
    /// decides whether a collective is needed at all.
    pub fn is_distributed(&self) -> bool {
        self.local_ranks > 1
    }
}

/// The widest `Peer` gang any coordinator on this deployment may accept —
/// bounds a job's per-job `world_size` (`TrainingCommon`, checked at
/// submit against [`Self::max_world_size`] by the submit edge) ACROSS FLEET
/// MEMBERS. Orthogonal to [`WorkerConfig::local_ranks`],
/// which bounds how many ranks THIS HOST
/// places on its own `[gpu] devices` for a job it runs entirely in-process:
/// the two knobs load independently, with no cross-check between them — a
/// deployment can
/// set one without the other, and this crate never reads
/// [`WorkerConfig::local_ranks`] while validating this section or vice
/// versa.
///
/// This crate only loads and validates the knob; the submit-time check
/// against a job's own `world_size` is the coordinator body's
/// (`fine_tune/spec.rs`), not built here.
///
/// # TOML
///
/// ```toml
/// [distributed]
/// max_world_size = 1
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DistributedConfig {
    /// The widest gang a coordinator on this deployment may admit. Must be
    /// `>= 1` (`1`, the default, is the single-rank deployment: no fleet
    /// gang is ever admitted). `0` is refused at load — it is not "unset",
    /// since the unset value is the default `1`.
    pub max_world_size: u32,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        // One rank: an unconfigured deployment admits no fleet gang at
        // all, matching `[worker]`'s own single-rank default.
        Self { max_world_size: 1 }
    }
}

impl DistributedConfig {
    /// Validate `max_world_size`: refuses `0` (a deployment that admits no
    /// rank at all cannot be the widest bound anything is checked against)
    /// at load time, naming the key, rather than surfacing as a confusing
    /// "every job refused" symptom the first time a job is submitted.
    pub fn validate(&self) -> Result<()> {
        if self.max_world_size == 0 {
            return Err(JammiError::Config(
                "[distributed] max_world_size must be >= 1 (1 is the single-rank deployment; \
                 0 admits no gang at all)"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// Job retention: how long a TERMINAL `jobs` row (`completed` /
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

/// Which [`crate::index::SegmentPlacement`] a session builds:
/// [`Self::Local`] ([`crate::index::AllLocal`], the default — every segment
/// is this process's own, a single node regardless of what else is
/// configured) or [`Self::Rendezvous`] ([`crate::index::RendezvousPlacement`]
/// over the live `instances` ring — beyond-one-node retrieval). `rendezvous`
/// with no `[server] peer_advertise` is refused, by name, at the ONE
/// membership choke point
/// ([`crate::catalog::instance::MembershipConfig::validate`]) both
/// [`crate::config::JammiConfig::load_from`] and
/// [`crate::catalog::instance::InstanceRegistration::from_config`] call —
/// never here, and never in [`ServerConfig::validate`], which cannot see
/// whether a segment placement even wants a member row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementMode {
    /// Every segment is this process's own — [`crate::index::AllLocal`].
    #[default]
    Local,
    /// Beyond-one-node retrieval over the live `instances` ring —
    /// [`crate::index::RendezvousPlacement`]. Requires `[server]
    /// peer_advertise`.
    Rendezvous,
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
    /// The address OTHER replicas dial THIS process's `peer_bind` listener
    /// at — usually a load-balancer-free, directly-routable `host:port`
    /// (`peer_bind` itself is commonly `0.0.0.0:PORT`, unusable as a dial
    /// target). `None` (the default) = this process never advertises a gang
    /// membership row: its `instances.peer_addr`/`result_root` columns stay
    /// `NULL` regardless of whether `peer_bind` is set. Requires `peer_bind`
    /// to be set too (refused at the ONE membership choke point,
    /// [`crate::catalog::instance::InstanceRegistration::from_config`] —
    /// naming BOTH keys); parses as a
    /// [`PeerAddr`](crate::catalog::instance::PeerAddr).
    ///
    /// # TOML
    ///
    /// ```toml
    /// [server]
    /// peer_bind = "0.0.0.0:9000"
    /// peer_advertise = "10.0.4.7:9000"
    /// ```
    pub peer_advertise: Option<String>,
    /// MARGINAL-LOAD ADMISSION per query, in bytes: the maximum estimated
    /// bytes ONE query may load locally for segments it does not own, when
    /// their owners are unreachable (the last rung of the placed-search
    /// failure ladder). Unset (the default) = unbounded. NOT a memory cap: the
    /// segment cache never evicts, earlier queries' loads are invisible to the
    /// check, and concurrent queries admit independently, so peak heap is
    /// concurrency × budget. Read by the result store; a library embedder sets
    /// it through the same config. `Some(0)` is refused.
    pub peer_local_load_bytes: Option<u64>,
    /// Which [`crate::index::SegmentPlacement`] this session builds. Default
    /// [`PlacementMode::Local`] — every existing deployment's behaviour is
    /// unchanged. `"rendezvous"` requires `peer_advertise` to be set too
    /// (refused at the membership choke point — see [`PlacementMode`]'s doc).
    ///
    /// # TOML
    ///
    /// ```toml
    /// [server]
    /// placement = "rendezvous"
    /// peer_bind = "0.0.0.0:9000"
    /// peer_advertise = "10.0.4.7:9000"
    /// ```
    pub placement: PlacementMode,
}

/// The optional service-tier selection for a server deployment. `All` (the
/// default) mounts every optional tier; `Only` mounts core plus
/// exactly the named optional tiers. Kept as raw tokens here so `jammi-db` does
/// not depend on `jammi-server`'s tier vocabulary — the server resolves and
/// validates them.
///
/// Hand-written [`Deserialize`] (not `#[serde(untagged)]`) so the three
/// natural TOML/env forms all parse with one shared, case-sensitive grammar
/// — `services` is the one field in this config whose env spelling is
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
                // env/file-backed value in this config tolerates it.
                // Case-sensitive otherwise, like every other value in this
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
    pub task: Option<crate::ModelTask>,
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
                let mut task: Option<crate::ModelTask> = None;
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
                            task = Some(crate::ModelTask::parse(&token).map_err(|e| {
                                serde::de::Error::custom(format!(
                                    "preload_models: unknown task `{token}`: {e}"
                                ))
                            })?);
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

/// Two of the six `[server]`/`[ballista]` fixed listener addresses collide
/// iff their ports are equal and non-zero AND (their hosts are equal, OR
/// either host is UNSPECIFIED — `0.0.0.0` / `[::]` binds every local
/// interface, so it always overlaps a host-specific bind on the same port,
/// and the two unspecified wildcards of different families, `0.0.0.0` and
/// `[::]`, overlap each other on a dual-stack listener the same way). An
/// ephemeral (`:0`) address never collides with anything — the kernel
/// assigns each bind a distinct free port, so two `:0` addresses (even the
/// identical host) are never a collision. The ONE definition
/// [`ServerConfig::validate`]'s three-way check and [`BallistaConfig::validate`]'s
/// six-way check both apply, so the two call sites can never diverge on what
/// "collide" means.
pub(crate) fn addresses_collide(a: std::net::SocketAddr, b: std::net::SocketAddr) -> bool {
    a.port() != 0
        && a.port() == b.port()
        && (a.ip() == b.ip() || a.ip().is_unspecified() || b.ip().is_unspecified())
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
        // Two surfaces must not bind an address whose port collides —
        // [`addresses_collide`]: equal ports, non-zero, on equal or
        // either-unspecified hosts. An ephemeral (`:0`) request never
        // collides — the kernel assigns each bind a distinct free port.
        if addresses_collide(health, flight) {
            return Err(crate::error::JammiError::Config(
                "health_listen and flight_listen must be different addresses".into(),
            ));
        }
        // The third listener, when set, joins the same collision rule
        // against BOTH of the others (a 3-way check).
        if let Some(raw) = &self.peer_bind {
            let peer: SocketAddr = raw.parse().map_err(|e| {
                crate::error::JammiError::Config(format!("Invalid peer_bind address '{raw}': {e}"))
            })?;
            if addresses_collide(peer, flight) {
                return Err(crate::error::JammiError::Config(
                    "peer_bind and flight_listen must be different addresses".into(),
                ));
            }
            if addresses_collide(peer, health) {
                return Err(crate::error::JammiError::Config(
                    "peer_bind and health_listen must be different addresses".into(),
                ));
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

/// `[ballista]`: which of the three compute-plane roles this process holds
/// — a scheduler, an executor, a client — in any combination. Unset (the
/// default) means none (roles are config, never a cargo feature). A
/// scheduler and an executor on one process is the single-node cluster; a
/// client is a process whose batch statements run on the cluster.
///
/// # TOML
///
/// ```toml
/// [ballista.scheduler]
/// bind = "0.0.0.0:50050"                    # Some = host a scheduler
/// advertise_host = "10.0.4.7"               # default: the bind host
///
/// [ballista.executor]
/// scheduler_address = "10.0.4.7:50050"      # Some = host an executor
/// bind = "0.0.0.0:50051"                    # shuffle (Arrow Flight) listener
/// grpc_bind = "0.0.0.0:50052"               # task (gRPC) listener
/// advertise_host = "10.0.4.8"               # default: the bind host
/// work_dir = "/var/lib/jammi/shuffle"       # default: a fresh temp dir
/// task_slots = 1                            # >= 1
///
/// [ballista.client]
/// scheduler_address = "10.0.4.7:50050"      # Some = a client of that scheduler
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BallistaConfig {
    /// This process hosts a Ballista scheduler iff `Some`. `None` (the
    /// default) means no scheduler role.
    pub scheduler: Option<BallistaSchedulerConfig>,
    /// This process hosts a Ballista executor iff `Some`. `None` (the
    /// default) means no executor role.
    pub executor: Option<BallistaExecutorConfig>,
    /// This process is a client of a Ballista scheduler iff `Some`: a
    /// statement whose plan is a materialization runs on that scheduler's
    /// executors. `None` (the default) means every statement runs in this
    /// process.
    pub client: Option<BallistaClientConfig>,
}

/// `[ballista.client]`: the scheduler a client-role process submits its
/// materializations to. Named explicitly even on a process that hosts the
/// scheduler itself: hosting a scheduler makes a process the cluster's
/// binder, which says nothing about where that process's own statements
/// run — the two are separate roles, held separately.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BallistaClientConfig {
    /// The scheduler this client submits to: `host:port` (a `SocketAddr`,
    /// or a DNS name and port — the Kubernetes case). Required: the unset
    /// default (empty) is refused by [`BallistaConfig::validate`].
    pub scheduler_address: String,
    /// The device kind this client places its models' plans onto — a
    /// deployment fact, not this process's hardware: a CPU query tier
    /// placing onto a GPU compute tier names `"cuda"`. Unset, a plan
    /// requires the kind of this process's own compute device. A plan no
    /// live executor holds runs in this process whatever it names, and its
    /// table records the device it ran on.
    pub device_kind: Option<crate::store::manifest::ComputeDeviceKind>,
}

/// `[ballista.scheduler]`: a scheduler role's listener and the host
/// executors dial it back at.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BallistaSchedulerConfig {
    /// This scheduler's gRPC listener. Default: `"0.0.0.0:50050"`.
    pub bind: String,
    /// The host executors dial to report a placed task's status back to
    /// this scheduler — stamped, with `bind`'s port, into every task it
    /// places. `None` (the default) means the `bind` host.
    pub advertise_host: Option<String>,
}

impl Default for BallistaSchedulerConfig {
    fn default() -> Self {
        Self {
            bind: "0.0.0.0:50050".into(),
            advertise_host: None,
        }
    }
}

impl BallistaSchedulerConfig {
    /// The host an executor dials to reach this scheduler: see
    /// `advertised_host`.
    pub fn advertised_host(&self) -> Result<String> {
        advertised_host(
            "ballista.scheduler",
            &self.bind,
            self.advertise_host.as_deref(),
        )
    }
}

/// The host a peer dials to reach the listener a role binds at `bind`:
/// `advertise_host` when set, otherwise `bind`'s own host. A bind on an
/// unspecified host (`0.0.0.0`/`::`) accepts on every interface but names
/// none — it unwraps to a real address only on the *dialling* peer's side
/// of a connection this process accepted, never on this process's own —
/// so such a bind must advertise a host or it is refused, naming `table`'s
/// keys. One rule for both roles: the scheduler's name rides in every task
/// it places (the executor's status-report target) and the executor's in
/// its registration (the scheduler's task-push target).
fn advertised_host(table: &str, bind: &str, advertise_host: Option<&str>) -> Result<String> {
    let addr: std::net::SocketAddr = bind
        .parse()
        .map_err(|e| JammiError::Config(format!("Invalid {table}.bind address '{bind}': {e}")))?;
    match advertise_host {
        Some(host) => Ok(host.to_string()),
        None if addr.ip().is_unspecified() => Err(JammiError::Config(format!(
            "{table}.advertise_host must be set when {table}.bind '{bind}' has an \
             unspecified host (0.0.0.0/::)"
        ))),
        None => Ok(addr.ip().to_string()),
    }
}

/// `[ballista.executor]`: an executor role's listeners, the scheduler it
/// registers with, and its task-slot capacity.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BallistaExecutorConfig {
    /// The scheduler this executor registers with and takes tasks from:
    /// `host:port` (a `SocketAddr`, or a DNS name and port — the
    /// Kubernetes case). Required: the unset default (empty) is refused by
    /// [`BallistaConfig::validate`].
    pub scheduler_address: String,
    /// This executor's Arrow Flight (shuffle) listener. Default:
    /// `"0.0.0.0:50051"`.
    pub bind: String,
    /// This executor's gRPC (task) listener. Default: `"0.0.0.0:50052"`.
    pub grpc_bind: String,
    /// The host other executors and the scheduler dial to reach this
    /// executor. `None` (the default) means the `bind` host.
    pub advertise_host: Option<String>,
    /// Local directory Ballista's shuffle writer stages files under.
    /// `None` (the default) means a fresh temporary directory per process
    /// (no object-store shuffle in v1).
    pub work_dir: Option<PathBuf>,
    /// Concurrent task slots this executor offers the scheduler. Must be
    /// `>= 1`. Default: 1.
    pub task_slots: u32,
}

impl Default for BallistaExecutorConfig {
    fn default() -> Self {
        Self {
            scheduler_address: String::new(),
            bind: "0.0.0.0:50051".into(),
            grpc_bind: "0.0.0.0:50052".into(),
            advertise_host: None,
            work_dir: None,
            task_slots: 1,
        }
    }
}

impl BallistaExecutorConfig {
    /// The host the scheduler (and other executors) dial to reach this
    /// executor: see `advertised_host`.
    pub fn advertised_host(&self) -> Result<String> {
        advertised_host(
            "ballista.executor",
            &self.bind,
            self.advertise_host.as_deref(),
        )
    }
}

impl BallistaConfig {
    /// Whether this process hosts a Ballista scheduler role.
    pub fn hosts_scheduler(&self) -> bool {
        self.scheduler.is_some()
    }

    /// Whether this process hosts a Ballista executor role.
    pub fn hosts_executor(&self) -> bool {
        self.executor.is_some()
    }

    /// Whether this process is a client of a Ballista scheduler.
    pub fn hosts_client(&self) -> bool {
        self.client.is_some()
    }

    /// Validate `config`'s `[ballista]` section: a CROSS-SECTION check, the
    /// same shape as
    /// [`crate::catalog::instance::MembershipConfig::validate`] — ONE
    /// `&JammiConfig` in, never `self` plus a separately-threaded
    /// `&ServerConfig` (a second read into the same config), because the
    /// six-address collision rule below needs both `config.ballista` and
    /// `config.server` at once. Every configured bind address parses;
    /// `executor.scheduler_address` and `client.scheduler_address` each
    /// parse as a validated `host:port` DIAL target
    /// ([`crate::catalog::instance::PeerAddr`] — hostnames are the
    /// Kubernetes case, so this is never restricted to a `SocketAddr`, and
    /// a `:0` scheduler address is refused the same way `PeerAddr` refuses
    /// one for any dial target; a dial target binds nothing, so the client
    /// role joins no collision check); a FIXED-port collision among
    /// `scheduler.bind`, `executor.bind`, `executor.grpc_bind`,
    /// `server.health_listen`, `server.flight_listen`, `server.peer_bind`
    /// is refused naming BOTH keys (an ephemeral `:0` never collides — each
    /// resolves to a distinct kernel-assigned port, the same rule
    /// [`ServerConfig::validate`] applies to its own three listeners);
    /// `executor.task_slots == 0` and `executor.work_dir = Some("")` are
    /// refused.
    ///
    /// Called by [`JammiConfig::load_from`]. [`ServerConfig::validate`]
    /// stays the owner of its OWN three addresses' domain validity and is
    /// called from `jammi_server::runtime::OssServer::new`, never from
    /// `load_from` — so a `JammiConfig` built by struct literal (or by
    /// `parse_from` alone) and handed straight to `OssServer::new`, the way
    /// most `jammi-server` integration tests do, never runs THIS function
    /// either. Hosting the roles this section describes is `OssServer::
    /// new`'s own job, so that constructor also calls
    /// `BallistaConfig::validate(&config)` right after
    /// `config.server.validate()` — the same second-call-site shape
    /// [`crate::catalog::instance::MembershipConfig::validate`] has at
    /// `InstanceRegistration::from_config`, for the same "struct-literal
    /// config skips load_from" reason.
    pub fn validate(config: &JammiConfig) -> Result<()> {
        use std::net::SocketAddr;

        let ballista = &config.ballista;
        let server = &config.server;

        // Every currently-configured FIXED listener this section and
        // `server` own, named, so a collision names both keys. `server`'s
        // own three are re-parsed here rather than threaded through as
        // already-parsed `SocketAddr`s; an unparseable one is
        // `ServerConfig::validate`'s own refusal, not this function's, so
        // it is silently skipped here (`.ok()`) — leaving the other
        // listeners checked against each other exactly as if it were
        // absent, never a spurious ballista-side error about a server key
        // this function does not own.
        let mut fixed: Vec<(&'static str, SocketAddr)> = Vec::new();

        if let Some(scheduler) = &ballista.scheduler {
            let addr: SocketAddr = scheduler.bind.parse().map_err(|e| {
                JammiError::Config(format!(
                    "Invalid ballista.scheduler.bind address '{}': {e}",
                    scheduler.bind
                ))
            })?;
            fixed.push(("ballista.scheduler.bind", addr));
            // An executor reports every placed task's status back to the
            // scheduler's advertised name; an unspecified bind host must
            // therefore advertise one.
            scheduler.advertised_host()?;
        }

        if let Some(executor) = &ballista.executor {
            let bind: SocketAddr = executor.bind.parse().map_err(|e| {
                JammiError::Config(format!(
                    "Invalid ballista.executor.bind address '{}': {e}",
                    executor.bind
                ))
            })?;
            fixed.push(("ballista.executor.bind", bind));

            let grpc_bind: SocketAddr = executor.grpc_bind.parse().map_err(|e| {
                JammiError::Config(format!(
                    "Invalid ballista.executor.grpc_bind address '{}': {e}",
                    executor.grpc_bind
                ))
            })?;
            fixed.push(("ballista.executor.grpc_bind", grpc_bind));

            crate::catalog::instance::PeerAddr::parse(&executor.scheduler_address).map_err(
                |e| JammiError::Config(format!("Invalid ballista.executor.scheduler_address: {e}")),
            )?;

            if executor.task_slots == 0 {
                return Err(JammiError::Config(
                    "ballista.executor.task_slots must be >= 1".into(),
                ));
            }

            if let Some(dir) = &executor.work_dir {
                if dir.as_os_str().is_empty() {
                    return Err(JammiError::Config(
                        "ballista.executor.work_dir must not be empty when set".into(),
                    ));
                }
            }

            // The scheduler dials the executor's flight/task ports back
            // (registration + task push); an unspecified `bind` host must
            // therefore advertise one.
            executor.advertised_host()?;
        }

        if let Some(client) = &ballista.client {
            crate::catalog::instance::PeerAddr::parse(&client.scheduler_address).map_err(|e| {
                JammiError::Config(format!("Invalid ballista.client.scheduler_address: {e}"))
            })?;
        }

        if let Ok(addr) = server.health_listen.parse::<SocketAddr>() {
            fixed.push(("server.health_listen", addr));
        }
        if let Ok(addr) = server.flight_listen.parse::<SocketAddr>() {
            fixed.push(("server.flight_listen", addr));
        }
        if let Some(raw) = &server.peer_bind {
            if let Ok(addr) = raw.parse::<SocketAddr>() {
                fixed.push(("server.peer_bind", addr));
            }
        }

        for i in 0..fixed.len() {
            for j in (i + 1)..fixed.len() {
                let (name_a, addr_a) = fixed[i];
                let (name_b, addr_b) = fixed[j];
                if addresses_collide(addr_a, addr_b) {
                    return Err(JammiError::Config(format!(
                        "{name_a} ({addr_a}) and {name_b} ({addr_b}) must not bind colliding \
                         addresses (equal ports on equal, or either unspecified, hosts)"
                    )));
                }
            }
        }

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

/// Vendor-neutral OTLP trace export: where to send spans, the request
/// headers the exporter attaches, the resource's `service.name`, and the
/// fraction of traces to keep.
///
/// `jammi-db` carries this raw, typed section only — the exporter itself
/// (`jammi_ai::telemetry::otlp_layer`, gated behind the `telemetry-otlp`
/// cargo feature) lives in `jammi-ai`, mirroring [`ModelsConfig::hub_token`]'s
/// split: a header value stays an unresolved [`SecretSource`] here
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
    /// `jammi-ai` session choke point, not here. The token FILE is
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
            distributed: DistributedConfig::default(),
            jobs: JobsConfig::default(),
            cache: CacheConfig::default(),
            server: ServerConfig::default(),
            ballista: BallistaConfig::default(),
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
            // Unset, NOT `[0]`: a `GpuConfig { device: -1, ..default() }`
            // must resolve to the CPU, which it does only while the plural
            // stays absent and `device_list()` derives it from `device`.
            devices: None,
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
            batch_tokens: 16384,
            batch_timeout_secs: 300,
            max_loaded_models: 0,
            partitions: 1,
            http: HttpConfig::default(),
        }
    }
}

impl InferenceConfig {
    /// `partitions` above this is refused — not a compute limit (forwards
    /// are admitted elsewhere), but a RESIDENCY one: each partition holds a
    /// runner, an exchange channel and a chunk being gathered, so an unbounded
    /// value is an unbounded number of resident runners per query. `1024` is
    /// generous relative to any real deployment's device or core count while
    /// still refusing an obviously-mistyped value (a raw row count, a byte
    /// size).
    pub const MAX_PARTITIONS: usize = 1024;

    /// `batch_size` and `batch_tokens` as the chunk budget a plan is built
    /// with. `0` is refused by name for either: a budget of nothing forwards
    /// nothing.
    pub fn chunk_budget(&self) -> Result<ChunkBudget> {
        let non_zero = |name: &str, value: usize| {
            NonZeroUsize::new(value).ok_or_else(|| {
                JammiError::Config(format!(
                    "[inference] {name} must be >= 1 (0 is refused, never silently treated as 1)"
                ))
            })
        };
        Ok(ChunkBudget {
            rows: non_zero("batch_size", self.batch_size)?,
            tokens: non_zero("batch_tokens", self.batch_tokens)?,
        })
    }

    /// `partitions` as the non-zero fan-out a plan is built with. `0` is
    /// refused by name — a config author who wrote `0` meant something, and
    /// silently getting `1` back is not it — and so is a value above
    /// [`Self::MAX_PARTITIONS`].
    pub fn fan_out(&self) -> Result<NonZeroUsize> {
        let partitions = NonZeroUsize::new(self.partitions).ok_or_else(|| {
            JammiError::Config(
                "[inference] partitions must be >= 1 (0 is refused, never silently treated as 1)"
                    .into(),
            )
        })?;
        if partitions.get() > Self::MAX_PARTITIONS {
            return Err(JammiError::Config(format!(
                "[inference] partitions = {} exceeds the maximum {} — partitions is a count of \
                 RESIDENT runners held open by one query, not a compute knob",
                self.partitions,
                Self::MAX_PARTITIONS
            )));
        }
        Ok(partitions)
    }

    /// Refuse a chunk budget or `partitions` no plan can be built with.
    ///
    /// Called by [`JammiConfig::load_from`]. A `JammiConfig` built by
    /// struct literal (or by `parse_from` alone) and handed straight to an
    /// `InferenceSession` constructor never runs `load_from`, so the second
    /// call site is `jammi_ai::session::InferenceSession::wrap_with`, the
    /// universal constructor funnel that also covers
    /// [`crate::catalog::instance::MembershipConfig::validate`] for the
    /// identical reason.
    pub fn validate(&self) -> Result<()> {
        self.chunk_budget()?;
        self.fan_out()?;
        Ok(())
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
            index_segment_rows: NonZeroUsize::new(4096).expect("a positive segment budget"),
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
            peer_advertise: None,
            peer_local_load_bytes: None,
            placement: PlacementMode::default(),
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

fn num_cpus() -> std::num::NonZeroUsize {
    std::thread::available_parallelism().unwrap_or(std::num::NonZeroUsize::MIN)
}

// --- Loading ---

/// The filesystem roots [`resolve_config_path_in`] probes, injected so a test
/// can point every step at a tempdir instead of the real `/etc` or platform
/// config directory.
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

/// Resolve the config file path, in order: an explicit path that exists,
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
/// [`JammiError::Config`] that names both the offending struct path and
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
    /// The result-table root this deployment resolves to, VERBATIM: the
    /// explicit `[storage] result_root` when set, else `{artifact_dir}/
    /// jammi_db` — the SAME derivation `jammi_db::store::ResultStore::new`'s
    /// local-root arm performs (`artifact_dir.join("jammi_db")`), the ONE
    /// place that join happens so nothing downstream re-derives it
    /// independently. This string is exactly what a gang member's
    /// `instances.result_root` column carries
    /// ([`crate::catalog::instance::InstanceRegistration::from_config`]) —
    /// no scheme folding, no symlink resolution, no reinterpretation of any
    /// kind.
    ///
    /// # Errors
    ///
    /// [`JammiError::Config`] naming `artifact_dir` when its joined
    /// `{artifact_dir}/jammi_db` path is not valid UTF-8 — never a silent
    /// lossy fold (`Path::to_string_lossy`'s replacement-character
    /// substitution), since that fold could make two genuinely different
    /// paths compare equal downstream.
    pub fn resolved_result_root(&self) -> Result<String> {
        match &self.storage.result_root {
            Some(root) => Ok(root.clone()),
            None => {
                let joined = self.artifact_dir.join("jammi_db");
                joined.to_str().map(str::to_string).ok_or_else(|| {
                    JammiError::Config(format!(
                        "artifact_dir '{}' is not valid UTF-8",
                        self.artifact_dir.display()
                    ))
                })
            }
        }
    }

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
    /// run post-load validation (`storage.cloud.validate()`, the worker
    /// interval and rank-topology invariants, the section validators).
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
        // Reject a device list nothing can be placed on (empty, repeated, a
        // plural that disagrees with the primary — `GpuConfig::validate`,
        // which `topology` runs first, so the device rules are checked here
        // exactly once) and a rank width no placement exists for (zero,
        // wider than the devices, a zero gang deadline) at load, naming the
        // offending key — never at the first gang boundary, half-way into a
        // training run.
        config.worker.topology(&config.gpu)?;
        // Reject a `[distributed] max_world_size = 0` at load time, naming
        // the key — never cross-checked against `[worker]`'s own topology
        // (the two knobs load independently).
        config.distributed.validate()?;
        // Reject a retention window past the cap at load, not in the first
        // sweep's timestamp arithmetic.
        config.jobs.validate()?;
        // Reject an out-of-domain `[server.limits]` knob (a zero
        // `max_message_bytes`, a per-connection budget over the global one, a
        // zero timeout) at load time, naming the offending key, rather than
        // at server startup deep inside `OssServer::new`.
        config.server.limits.validate()?;
        // Reject an out-of-domain `[ballista]` knob (an unparseable bind, a
        // fixed-port collision with itself or with `[server]`'s three
        // listeners, a zero `task_slots`, an empty `work_dir`) at load
        // time, naming the offending key. `BallistaConfig::validate` is a
        // cross-section check (its own doc names the SECOND call site
        // `OssServer::new` must also make, for a struct-literal config that
        // skips `load_from` entirely).
        BallistaConfig::validate(&config)?;
        // Reject an out-of-domain `[observability]` knob (a `sample_ratio`
        // outside `[0.0, 1.0]`, including NaN, or a malformed/non-http(s)
        // `otlp_endpoint`) at load time, naming the offending key, rather
        // than at the first `jammi_ai::telemetry::otlp_layer` call.
        config.observability.validate()?;
        // Reject an out-of-grammar `[engine] memory_limit` (an unparseable
        // form, an out-of-range percentage, or a resolved value below the
        // floor) at load time, naming the key — rather than at the first
        // session build, deep inside `JammiSession::build`'s memory-pool
        // construction. The resolved value itself is discarded here; every
        // real consumer re-resolves through this same reader.
        config.engine.memory_limit_bytes()?;
        // Reject a `[server] peer_advertise` that cannot resolve a valid
        // gang-membership shape (an unset `peer_bind`, or an unparseable
        // address) at load time, naming the offending key — rather than
        // only surfacing deep inside `InferenceSession::wrap_with`'s own
        // registration call. `MembershipConfig::validate` performs no
        // filesystem access and no interpretation of the result root at
        // all — the row carries `resolved_result_root()` verbatim;
        // `InstanceRegistration::from_config`, which `wrap_with`
        // calls (every `InferenceSession` constructor funnels through it),
        // is the ONLY other caller, so a struct-literal config that skips
        // `load_from` entirely is still covered there. The `Option` is
        // discarded; this call is for its early-failure side effect only.
        let _ = crate::catalog::instance::MembershipConfig::validate(&config)?;
        // Refuse an `[inference]` fan-out or batch size no plan can be built
        // with, at load time and by name. The second call site (a
        // struct-literal `JammiConfig` that skips this function entirely) is
        // named on `InferenceConfig::validate`'s own doc.
        config.inference.validate()?;
        Ok(config)
    }

    /// The parse-only core: interpolate `${VAR}` from `env`,
    /// parse the TOML file layer, build the `JAMMI_*` env layer from the
    /// SAME `env` map (the namespace rule — `env_map::build_env_layer`),
    /// deep-merge the two (`layers::merge`), and deserialize the
    /// whole typed [`JammiConfig`] from the merged tree in one pass via
    /// `serde_path_to_error` (every error names the struct path and the
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

/// Substitute `${VAR}` patterns in `input`, resolved through `lookup` —
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
