use serde::Deserialize;
use std::path::{Path, PathBuf};

use crate::error::Result;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct JammiConfig {
    pub artifact_dir: PathBuf,
    pub engine: EngineConfig,
    pub gpu: GpuConfig,
    pub inference: InferenceConfig,
    pub embedding: EmbeddingConfig,
    pub fine_tuning: FineTuningConfig,
    pub cache: CacheConfig,
    pub server: ServerConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EngineConfig {
    pub execution_threads: usize,
    pub memory_limit: String,
    pub batch_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct GpuConfig {
    pub device: i32,
    pub memory_limit: String,
    pub memory_fraction: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct InferenceConfig {
    pub default_backend: String,
    pub batch_size: usize,
    pub batch_timeout_secs: u64,
    pub max_loaded_models: usize,
    pub vllm: VllmConfig,
    pub http: HttpConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct VllmConfig {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub extra_args: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct HttpConfig {
    pub timeout_secs: u64,
    pub headers: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    pub default_distance_metric: String,
    pub default_index_type: String,
    pub checkpoint_interval: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct FineTuningConfig {
    pub default_lora_rank: usize,
    pub default_learning_rate: f64,
    pub default_epochs: usize,
    pub default_batch_size: usize,
    pub checkpoint_fraction: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CacheConfig {
    pub ann_cache_enabled: bool,
    pub ann_cache_max_entries: usize,
    pub embedding_cache_enabled: bool,
    pub embedding_cache_size: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub listen: String,
    pub flight_listen: String,
    pub preload_models: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LoggingConfig {
    pub level: String,
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
    /// Load config from file + env vars.
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
