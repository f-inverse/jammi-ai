use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde::Deserialize;

use crate::error::{JammiError, Result};
use crate::storage::CloudConfig;

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
/// [catalog]
/// kind = "postgres"
/// url = "${POSTGRES_URL}"
/// pool_size = 16
/// max_lifetime_secs = 1800
///
/// [broker]
/// kind = "jet_stream"
/// url = "nats://${NATS_HOST}:4222"
/// retention_seconds = 604800
/// credentials_path = "/var/run/secrets/nats.creds"
/// ```
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
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
    /// Cache layer settings (ANN cache, embedding cache).
    pub cache: CacheConfig,
    /// HTTP and Arrow Flight server bind addresses.
    pub server: ServerConfig,
    /// Tracing/logging configuration.
    pub logging: LoggingConfig,
    /// Catalog backend selection. Default: SQLite under `artifact_dir`.
    pub catalog: CatalogConfig,
    /// Trigger broker selection. Default: in-process [`crate::trigger::InMemoryBroker`].
    pub broker: BrokerConfig,
    /// Object-storage selection for result tables and cloud sources. Default:
    /// empty — result tables live on local disk under `artifact_dir` and
    /// `r2://`/`s3://`/`gs://`/`azure://` sources resolve via the SDK's
    /// default credential chain.
    pub storage: StorageConfig,
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
/// [storage.cloud]
/// kind = "r2"
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
/// [storage.cloud]
/// kind = "s3"
/// region = "us-east-1"
/// ```
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    /// Storage URL the session roots result tables under. `None` → local
    /// `{artifact_dir}/jammi_db/`.
    pub result_root: Option<String>,
    /// Default per-cloud driver credentials. Threaded to every driver the
    /// session builds for the result root and for cloud sources. `None` → the
    /// SDK default credential chain (env vars, instance profile, …).
    pub cloud: Option<CloudConfig>,
}

/// Catalog backend selection. The catalog and the mutable companion tables
/// share this backend.
///
/// # TOML
///
/// ```toml
/// [catalog]
/// kind = "sqlite"
/// # path = "/var/lib/jammi/catalog.db"   # optional override
/// ```
///
/// ```toml
/// [catalog]
/// kind = "postgres"
/// url = "${POSTGRES_URL}"
/// pool_size = 16
/// max_lifetime_secs = 1800
/// ```
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
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
        /// Connection URL, e.g. `postgres://user:pass@host:5432/jammi`.
        url: String,
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
/// # TOML
///
/// ```toml
/// [broker]
/// kind = "in_memory"
/// ```
///
/// ```toml
/// [broker]
/// kind = "jet_stream"
/// url = "nats://${NATS_HOST}:4222"
/// retention_seconds = 604800
/// credentials_path = "/var/run/secrets/nats.creds"
/// ```
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BrokerConfig {
    /// In-process broker. Default; matches the laptop / dev workflow.
    InMemory,
    /// JetStream (NATS). Requires the `jetstream-broker` cargo feature on
    /// `jammi-db`; building a session whose config selects `JetStream`
    /// without the feature returns [`crate::error::JammiError::Config`].
    JetStream {
        /// NATS server URL, e.g. `nats://nats.svc:4222`.
        url: String,
        /// Default per-stream retention in seconds. Per-topic
        /// `broker_metadata.retention_seconds` overrides this value.
        /// Default: 7 days (604 800).
        #[serde(default = "default_retention_secs")]
        retention_seconds: u64,
        /// Optional path to a NATS `.creds` file. When unset the broker
        /// connects anonymously.
        #[serde(default)]
        credentials_path: Option<PathBuf>,
    },
}

impl Default for BrokerConfig {
    fn default() -> Self {
        Self::InMemory
    }
}

fn default_retention_secs() -> u64 {
    7 * 24 * 60 * 60
}

/// DataFusion query-engine settings.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
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
#[serde(default)]
pub struct GpuConfig {
    /// CUDA device ordinal. Default: 0.
    pub device: i32,
    /// GPU memory limit (e.g., `"auto"` or `"8GB"`). Default: `"auto"`.
    pub memory_limit: String,
    /// Fraction of GPU memory to allocate (0.0 - 1.0). Default: 0.9.
    pub memory_fraction: f64,
}

/// Model inference defaults.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
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
#[serde(default)]
pub struct HttpConfig {
    /// Request timeout in seconds. Default: 60.
    pub timeout_secs: u64,
    /// Extra HTTP headers sent with every inference request.
    pub headers: std::collections::HashMap<String, String>,
}

/// Embedding index defaults.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    /// Distance metric for ANN indices. Default: `Cosine`.
    pub default_distance_metric: DistanceMetric,
    /// ANN index type. Default: `IvfHnswSq`.
    pub default_index_type: IndexType,
    /// Rows between index checkpoint writes. Default: 1000.
    pub checkpoint_interval: usize,
}

/// Fine-tuning hyperparameter defaults.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
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

/// Cache layer settings.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
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
#[serde(default)]
pub struct ServerConfig {
    /// Health probe listen address. Default: `"0.0.0.0:8080"`.
    pub health_listen: String,
    /// Arrow Flight listen address. Default: `"0.0.0.0:8081"`.
    pub flight_listen: String,
    /// Model IDs to preload into memory at server startup.
    pub preload_models: Vec<String>,
}

impl ServerConfig {
    /// Validate server configuration.
    pub fn validate(&self) -> Result<()> {
        use std::net::SocketAddr;

        let _: SocketAddr = self.health_listen.parse().map_err(|e| {
            crate::error::JammiError::Config(format!(
                "Invalid health_listen address '{}': {e}",
                self.health_listen
            ))
        })?;
        let _: SocketAddr = self.flight_listen.parse().map_err(|e| {
            crate::error::JammiError::Config(format!(
                "Invalid flight_listen address '{}': {e}",
                self.flight_listen
            ))
        })?;
        if self.health_listen == self.flight_listen {
            return Err(crate::error::JammiError::Config(
                "health_listen and flight_listen must be different addresses".into(),
            ));
        }
        Ok(())
    }
}

/// Tracing/logging configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LoggingConfig {
    /// Log level filter (e.g., `"info"`, `"debug"`, `"warn"`). Default: `"info"`.
    pub level: String,
    /// Output format. Default: `Text`.
    pub format: LogFormat,
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
            cache: CacheConfig::default(),
            server: ServerConfig::default(),
            logging: LoggingConfig::default(),
            catalog: CatalogConfig::default(),
            broker: BrokerConfig::default(),
            storage: StorageConfig::default(),
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
            headers: std::collections::HashMap::new(),
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            default_distance_metric: DistanceMetric::Cosine,
            default_index_type: IndexType::IvfHnswSq,
            checkpoint_interval: 1000,
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

impl JammiConfig {
    /// Load configuration from a TOML file (resolved via explicit path, `JAMMI_CONFIG` env,
    /// `./jammi.toml`, or platform config dir) and apply environment variable overrides.
    ///
    /// Before TOML parsing the loader runs [`interpolate_env_vars`] on the
    /// raw file contents: `${NAME}` is substituted with the value of
    /// `std::env::var("NAME")`, `$$` escapes a literal `$`, and a missing
    /// variable is a hard error (no silent empty substitution). See
    /// [`interpolate_env_vars`] for the full rules.
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let config_path = Self::resolve_config_path(path);
        let mut config: Self = match config_path {
            Some(p) => {
                let contents = std::fs::read_to_string(&p)?;
                let interpolated = interpolate_env_vars(&contents)?;
                toml::from_str(&interpolated)?
            }
            None => Self::default(),
        };
        config.apply_env_overrides();
        // Catch a partial cloud-credential set (e.g. an R2 `access_key_id`
        // without its `secret_access_key`) at load time rather than deep
        // inside the first object-store request.
        if let Some(cloud) = &config.storage.cloud {
            cloud.validate()?;
        }
        Ok(config)
    }

    fn resolve_config_path(explicit: Option<&Path>) -> Option<PathBuf> {
        if let Some(p) = explicit {
            if p.exists() {
                return Some(p.to_path_buf());
            }
        }

        if let Ok(env_path) = std::env::var("JAMMI_CONFIG") {
            let p = PathBuf::from(env_path);
            if p.exists() {
                return Some(p);
            }
        }

        let cwd = PathBuf::from("jammi.toml");
        if cwd.exists() {
            return Some(cwd);
        }

        let home = directories::ProjectDirs::from("ai", "jammi", "jammi")
            .map(|d| d.config_dir().join("config.toml"));
        if let Some(p) = home {
            if p.exists() {
                return Some(p);
            }
        }

        None
    }

    fn apply_env_overrides(&mut self) {
        if let Ok(v) = std::env::var("JAMMI_ARTIFACT_DIR") {
            self.artifact_dir = PathBuf::from(v);
        }

        // Engine
        if let Ok(v) = std::env::var("JAMMI_ENGINE__EXECUTION_THREADS") {
            if let Ok(n) = v.parse() {
                self.engine.execution_threads = n;
            }
        }
        if let Ok(v) = std::env::var("JAMMI_ENGINE__BATCH_SIZE") {
            if let Ok(n) = v.parse() {
                self.engine.batch_size = n;
            }
        }
        if let Ok(v) = std::env::var("JAMMI_ENGINE__MEMORY_LIMIT") {
            self.engine.memory_limit = v;
        }

        // GPU
        if let Ok(v) = std::env::var("JAMMI_GPU__DEVICE") {
            if let Ok(n) = v.parse() {
                self.gpu.device = n;
            }
        }
        if let Ok(v) = std::env::var("JAMMI_GPU__MEMORY_LIMIT") {
            self.gpu.memory_limit = v;
        }
        if let Ok(v) = std::env::var("JAMMI_GPU__MEMORY_FRACTION") {
            if let Ok(n) = v.parse() {
                self.gpu.memory_fraction = n;
            }
        }

        // Inference
        if let Ok(v) = std::env::var("JAMMI_INFERENCE__DEFAULT_BACKEND") {
            match v.parse() {
                Ok(b) => self.inference.default_backend = b,
                Err(e) => tracing::warn!("Ignoring invalid JAMMI_INFERENCE__DEFAULT_BACKEND: {e}"),
            }
        }
        if let Ok(v) = std::env::var("JAMMI_INFERENCE__BATCH_SIZE") {
            if let Ok(n) = v.parse() {
                self.inference.batch_size = n;
            }
        }
        if let Ok(v) = std::env::var("JAMMI_INFERENCE__BATCH_TIMEOUT_SECS") {
            if let Ok(n) = v.parse() {
                self.inference.batch_timeout_secs = n;
            }
        }
        if let Ok(v) = std::env::var("JAMMI_INFERENCE__MAX_LOADED_MODELS") {
            if let Ok(n) = v.parse() {
                self.inference.max_loaded_models = n;
            }
        }

        // Logging
        if let Ok(v) = std::env::var("JAMMI_LOGGING__LEVEL") {
            self.logging.level = v;
        }
        if let Ok(v) = std::env::var("JAMMI_LOGGING__FORMAT") {
            match v.parse() {
                Ok(f) => self.logging.format = f,
                Err(e) => tracing::warn!("Ignoring invalid JAMMI_LOGGING__FORMAT: {e}"),
            }
        }

        // Server
        if let Ok(v) = std::env::var("JAMMI_SERVER__HEALTH_LISTEN") {
            self.server.health_listen = v;
        }
        if let Ok(v) = std::env::var("JAMMI_SERVER__FLIGHT_LISTEN") {
            self.server.flight_listen = v;
        }
    }
}

/// Substitute `${VAR}` patterns in `input` from the process environment.
///
/// Rules:
/// - `${NAME}` is replaced by the value of `std::env::var("NAME")`. A name
///   must start with `[A-Za-z_]` and continue with `[A-Za-z0-9_]`.
/// - A missing variable returns [`JammiError::Config`]. The loader does
///   **not** silently substitute an empty string — that is a common source of
///   "deployed config has empty Postgres URL" outages.
/// - `$$` escapes a literal `$`.
/// - A bare `$` not followed by `$` or `{` is preserved verbatim (lets the
///   raw `$` in a TOML password slip through unchanged).
/// - An unterminated `${` returns [`JammiError::Config`].
/// - Interpolation is one-pass and not recursive: the value of `${X}` is not
///   re-scanned.
pub fn interpolate_env_vars(input: &str) -> Result<String> {
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
                let value = std::env::var(name).map_err(|_| {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_config_round_trip_sqlite_default() {
        let toml_src = r#"
            [catalog]
            kind = "sqlite"
        "#;
        let cfg: JammiConfig = toml::from_str(toml_src).unwrap();
        assert_eq!(cfg.catalog, CatalogConfig::Sqlite { path: None });
    }

    #[test]
    fn catalog_config_round_trip_sqlite_with_path() {
        let toml_src = r#"
            [catalog]
            kind = "sqlite"
            path = "/srv/jammi/catalog.db"
        "#;
        let cfg: JammiConfig = toml::from_str(toml_src).unwrap();
        assert_eq!(
            cfg.catalog,
            CatalogConfig::Sqlite {
                path: Some(PathBuf::from("/srv/jammi/catalog.db"))
            }
        );
    }

    #[test]
    fn catalog_config_round_trip_postgres() {
        let toml_src = r#"
            [catalog]
            kind = "postgres"
            url = "postgres://u:p@h/db"
            pool_size = 16
            max_lifetime_secs = 1800
        "#;
        let cfg: JammiConfig = toml::from_str(toml_src).unwrap();
        assert_eq!(
            cfg.catalog,
            CatalogConfig::Postgres {
                url: "postgres://u:p@h/db".into(),
                pool_size: 16,
                max_lifetime_secs: Some(1800),
            }
        );
    }

    #[test]
    fn catalog_config_postgres_defaults() {
        let toml_src = r#"
            [catalog]
            kind = "postgres"
            url = "postgres://u:p@h/db"
        "#;
        let cfg: JammiConfig = toml::from_str(toml_src).unwrap();
        assert_eq!(
            cfg.catalog,
            CatalogConfig::Postgres {
                url: "postgres://u:p@h/db".into(),
                pool_size: 8,
                max_lifetime_secs: None,
            }
        );
    }

    #[test]
    fn broker_config_round_trip_in_memory() {
        let toml_src = r#"
            [broker]
            kind = "in_memory"
        "#;
        let cfg: JammiConfig = toml::from_str(toml_src).unwrap();
        assert_eq!(cfg.broker, BrokerConfig::InMemory);
    }

    #[test]
    fn broker_config_round_trip_jetstream() {
        let toml_src = r#"
            [broker]
            kind = "jet_stream"
            url = "nats://nats.svc:4222"
            retention_seconds = 86400
            credentials_path = "/run/secrets/nats.creds"
        "#;
        let cfg: JammiConfig = toml::from_str(toml_src).unwrap();
        assert_eq!(
            cfg.broker,
            BrokerConfig::JetStream {
                url: "nats://nats.svc:4222".into(),
                retention_seconds: 86400,
                credentials_path: Some(PathBuf::from("/run/secrets/nats.creds")),
            }
        );
    }

    #[test]
    fn broker_config_jetstream_defaults() {
        let toml_src = r#"
            [broker]
            kind = "jet_stream"
            url = "nats://nats.svc:4222"
        "#;
        let cfg: JammiConfig = toml::from_str(toml_src).unwrap();
        assert_eq!(
            cfg.broker,
            BrokerConfig::JetStream {
                url: "nats://nats.svc:4222".into(),
                retention_seconds: 7 * 24 * 60 * 60,
                credentials_path: None,
            }
        );
    }

    #[test]
    fn jammi_config_default_uses_sqlite_and_in_memory() {
        let cfg = JammiConfig::default();
        assert_eq!(cfg.catalog, CatalogConfig::Sqlite { path: None });
        assert_eq!(cfg.broker, BrokerConfig::InMemory);
    }

    #[test]
    fn storage_config_default_is_local() {
        let cfg = JammiConfig::default();
        assert!(cfg.storage.result_root.is_none());
        assert!(cfg.storage.cloud.is_none());
    }

    #[test]
    fn storage_config_round_trip_r2() {
        // Secrets (access_key_id / secret_access_key) are deliberately absent:
        // they come from the container's AWS_* env vars at driver-build time.
        let toml_src = r#"
            [storage]
            result_root = "r2://jammi-results/prod"

            [storage.cloud]
            kind = "r2"
            account_id = "abc123def456"
        "#;
        let cfg: JammiConfig = toml::from_str(toml_src).unwrap();
        assert_eq!(
            cfg.storage.result_root.as_deref(),
            Some("r2://jammi-results/prod")
        );
        match cfg.storage.cloud {
            Some(CloudConfig::R2(r2)) => {
                assert_eq!(r2.account_id.as_deref(), Some("abc123def456"));
                assert!(r2.access_key_id.is_none());
                assert!(r2.secret_access_key.is_none());
            }
            other => panic!("expected R2 cloud config, got {other:?}"),
        }
    }

    #[test]
    fn storage_config_round_trip_s3() {
        let toml_src = r#"
            [storage]
            result_root = "s3://jammi-results/prod"

            [storage.cloud]
            kind = "s3"
            region = "us-east-1"
        "#;
        let cfg: JammiConfig = toml::from_str(toml_src).unwrap();
        assert_eq!(
            cfg.storage.result_root.as_deref(),
            Some("s3://jammi-results/prod")
        );
        match cfg.storage.cloud {
            Some(CloudConfig::S3(s3)) => {
                assert_eq!(s3.region.as_deref(), Some("us-east-1"));
                assert!(s3.access_key_id.is_none());
            }
            other => panic!("expected S3 cloud config, got {other:?}"),
        }
    }

    #[test]
    fn load_rejects_partial_r2_credentials() {
        // account_id + access_key_id but no secret — the fail-closed
        // CloudConfig::validate must reject this at load time.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("jammi.toml");
        std::fs::write(
            &path,
            r#"
                [storage]
                result_root = "r2://bucket/prefix"

                [storage.cloud]
                kind = "r2"
                account_id = "abc"
                access_key_id = "only-the-key"
            "#,
        )
        .unwrap();
        let err = JammiConfig::load(Some(&path)).unwrap_err();
        assert!(
            matches!(err, JammiError::Storage(_)),
            "expected a Storage validation error, got {err:?}"
        );
    }

    #[test]
    fn interpolate_env_vars_happy_path() {
        // Parallel tests would collide on a shared env var; the test name is
        // baked into the var name to keep each test's view independent.
        std::env::set_var("JAMMI_TEST_INTERP_HAPPY", "from-env");
        let out = interpolate_env_vars("url = \"${JAMMI_TEST_INTERP_HAPPY}\"").unwrap();
        assert_eq!(out, "url = \"from-env\"");
        std::env::remove_var("JAMMI_TEST_INTERP_HAPPY");
    }

    #[test]
    fn interpolate_env_vars_missing_is_typed_error() {
        // Use a unique name to dodge a parallel-test race that sets it.
        std::env::remove_var("JAMMI_TEST_INTERP_DEFINITELY_NOT_SET");
        let err =
            interpolate_env_vars("url = \"${JAMMI_TEST_INTERP_DEFINITELY_NOT_SET}\"").unwrap_err();
        match err {
            JammiError::Config(msg) => {
                assert!(
                    msg.contains("JAMMI_TEST_INTERP_DEFINITELY_NOT_SET"),
                    "msg = {msg}"
                );
            }
            other => panic!("expected JammiError::Config, got {other:?}"),
        }
    }

    #[test]
    fn interpolate_env_vars_escape_double_dollar() {
        let out = interpolate_env_vars("password = \"$$secret$$\"").unwrap();
        assert_eq!(out, "password = \"$secret$\"");
    }

    #[test]
    fn interpolate_env_vars_unterminated_brace_errors() {
        let err = interpolate_env_vars("url = \"${UNCLOSED\"").unwrap_err();
        assert!(matches!(err, JammiError::Config(_)), "{err:?}");
    }

    #[test]
    fn interpolate_env_vars_bare_dollar_preserved() {
        let out = interpolate_env_vars("hint = \"price is $5\"").unwrap();
        assert_eq!(out, "hint = \"price is $5\"");
    }

    #[test]
    fn interpolate_env_vars_invalid_name_errors() {
        let err = interpolate_env_vars("url = \"${1bad}\"").unwrap_err();
        match err {
            JammiError::Config(msg) => assert!(msg.contains("Invalid env-var name"), "{msg}"),
            other => panic!("expected JammiError::Config, got {other:?}"),
        }
    }

    #[test]
    fn load_interpolates_before_parse() {
        std::env::set_var("JAMMI_TEST_LOAD_URL", "postgres://u:p@h/db");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("jammi.toml");
        std::fs::write(
            &path,
            r#"
                [catalog]
                kind = "postgres"
                url = "${JAMMI_TEST_LOAD_URL}"
                pool_size = 4
            "#,
        )
        .unwrap();
        let cfg = JammiConfig::load(Some(&path)).unwrap();
        assert_eq!(
            cfg.catalog,
            CatalogConfig::Postgres {
                url: "postgres://u:p@h/db".into(),
                pool_size: 4,
                max_lifetime_secs: None,
            }
        );
        std::env::remove_var("JAMMI_TEST_LOAD_URL");
    }
}
