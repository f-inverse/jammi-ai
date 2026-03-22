use serde::Deserialize;
use std::path::{Path, PathBuf};

use crate::error::Result;

/// Top-level configuration for the Jammi AI platform.
///
/// Load from a TOML file via [`JammiConfig::load`], with environment variable overrides.
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
    /// Backend selection strategy (e.g., `"auto"`, `"candle"`, `"vllm"`). Default: `"auto"`.
    pub default_backend: String,
    /// Maximum requests per inference batch. Default: 32.
    pub batch_size: usize,
    /// Seconds to wait before flushing an incomplete batch. Default: 300.
    pub batch_timeout_secs: u64,
    /// Maximum number of models held in memory simultaneously. 0 means unlimited. Default: 0.
    pub max_loaded_models: usize,
    /// vLLM-specific backend configuration.
    pub vllm: VllmConfig,
    /// HTTP backend configuration (for remote inference endpoints).
    pub http: HttpConfig,
}

/// vLLM backend connection settings.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct VllmConfig {
    /// vLLM server hostname. Default: None (auto-detect).
    pub host: Option<String>,
    /// vLLM server port. Default: None (auto-detect).
    pub port: Option<u16>,
    /// Additional CLI arguments passed to the vLLM server process.
    pub extra_args: Vec<String>,
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
    /// Distance metric for ANN indices (e.g., `"cosine"`, `"l2"`). Default: `"cosine"`.
    pub default_distance_metric: String,
    /// ANN index type (e.g., `"ivf_hnsw_sq"`). Default: `"ivf_hnsw_sq"`.
    pub default_index_type: String,
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

/// HTTP and Arrow Flight server bind addresses.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    /// HTTP API listen address. Default: `"0.0.0.0:8080"`.
    pub listen: String,
    /// Arrow Flight listen address. Default: `"0.0.0.0:8081"`.
    pub flight_listen: String,
    /// Model IDs to preload into memory at server startup.
    pub preload_models: Vec<String>,
}

/// Tracing/logging configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LoggingConfig {
    /// Log level filter (e.g., `"info"`, `"debug"`, `"warn"`). Default: `"info"`.
    pub level: String,
    /// Output format: `"text"` or `"json"`. Default: `"text"`.
    pub format: String,
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
            default_backend: "auto".into(),
            batch_size: 32,
            batch_timeout_secs: 300,
            max_loaded_models: 0,
            vllm: VllmConfig::default(),
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
            default_distance_metric: "cosine".into(),
            default_index_type: "ivf_hnsw_sq".into(),
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
            listen: "0.0.0.0:8080".into(),
            flight_listen: "0.0.0.0:8081".into(),
            preload_models: Vec::new(),
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".into(),
            format: "text".into(),
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
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let config_path = Self::resolve_config_path(path);
        let mut config: Self = match config_path {
            Some(p) => {
                let contents = std::fs::read_to_string(&p)?;
                toml::from_str(&contents)?
            }
            None => Self::default(),
        };
        config.apply_env_overrides();
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
            self.inference.default_backend = v;
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
            self.logging.format = v;
        }

        // Server
        if let Ok(v) = std::env::var("JAMMI_SERVER__LISTEN") {
            self.server.listen = v;
        }
        if let Ok(v) = std::env::var("JAMMI_SERVER__FLIGHT_LISTEN") {
            self.server.flight_listen = v;
        }
    }
}
