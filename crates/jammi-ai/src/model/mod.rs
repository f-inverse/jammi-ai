pub mod backend;
pub mod cache;
pub mod resolver;
pub mod tokenizer;

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use arrow::array::ArrayRef;
use backend::candle::CandleModel;
use backend::ort::OrtModel;
use jammi_engine::error::{JammiError, Result};
use serde::{Deserialize, Serialize};

use crate::inference::adapter::BackendOutput;

/// Unique identifier for a loaded model, used as cache key.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ModelId(pub String);

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Explicit model source — the user declares where to load from, no fallback.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModelSource {
    /// A HuggingFace Hub repository (e.g. `"sentence-transformers/all-MiniLM-L6-v2"`).
    HuggingFace(String),
    /// A local directory containing model files (config.json + weights).
    Local(PathBuf),
}

impl ModelSource {
    /// Create a HuggingFace Hub source.
    pub fn hf(repo_id: impl Into<String>) -> Self {
        Self::HuggingFace(repo_id.into())
    }

    /// Create a local filesystem source.
    pub fn local(path: impl Into<PathBuf>) -> Self {
        Self::Local(path.into())
    }

    /// Parse a user-provided model ID string into a ModelSource.
    /// `"local:/path/to/model"` → `Local(path)`, anything else → `HuggingFace(id)`.
    pub fn parse(id: &str) -> Self {
        if let Some(path) = id.strip_prefix("local:") {
            Self::Local(std::path::PathBuf::from(path))
        } else {
            Self::HuggingFace(id.to_string())
        }
    }

    /// Reconstruct a ModelSource from a canonical name (as stored in result_tables).
    /// Absolute paths that exist on disk → Local, everything else → HuggingFace.
    pub fn from_canonical(canonical_name: &str) -> Self {
        let path = std::path::Path::new(canonical_name);
        if path.is_absolute() && path.exists() {
            Self::Local(path.to_path_buf())
        } else {
            Self::HuggingFace(canonical_name.to_string())
        }
    }
}

impl std::fmt::Display for ModelSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HuggingFace(repo_id) => write!(f, "{repo_id}"),
            Self::Local(path) => write!(f, "{}", path.display()),
        }
    }
}

/// Construct the catalog PK for a model: `"{canonical_name}::{version}"`.
/// The canonical name is `ModelSource::to_string()` output.
pub fn to_catalog_pk(canonical_name: &str, version: i32) -> String {
    format!("{canonical_name}::{version}")
}

impl From<&ModelSource> for ModelId {
    fn from(source: &ModelSource) -> Self {
        ModelId(source.to_string())
    }
}

/// Which backend to use for this model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendType {
    /// Candle — native Rust inference via safetensors weights.
    Candle,
    /// ONNX Runtime — cross-platform inference via ONNX models.
    Ort,
    /// HTTP — remote model endpoint (REST/gRPC).
    Http,
}

/// What task this model performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelTask {
    /// Produce dense vector representations of input text.
    TextEmbedding,
    /// Produce dense vector representations of input images.
    ImageEmbedding,
    /// Assign a label and confidence score to input text.
    Classification,
    /// Extract named entities (person, org, location, etc.) from text.
    Ner,
}

impl std::fmt::Display for ModelTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TextEmbedding => write!(f, "text_embedding"),
            Self::ImageEmbedding => write!(f, "image_embedding"),
            Self::Classification => write!(f, "classification"),
            Self::Ner => write!(f, "ner"),
        }
    }
}

impl std::str::FromStr for ModelTask {
    type Err = jammi_engine::error::JammiError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "text_embedding" => Ok(Self::TextEmbedding),
            "image_embedding" => Ok(Self::ImageEmbedding),
            "classification" => Ok(Self::Classification),
            "ner" => Ok(Self::Ner),
            other => Err(jammi_engine::error::JammiError::Other(format!(
                "Unknown model task '{other}'. Expected: text_embedding, image_embedding, classification, ner"
            ))),
        }
    }
}

/// A resolved model — files located, backend determined, NOT yet loaded.
pub struct ResolvedModel {
    /// HuggingFace or local identifier for this model.
    pub model_id: ModelId,
    /// Selected inference backend.
    pub backend: BackendType,
    /// ML task this model performs.
    pub task: ModelTask,
    /// Path to the model's `config.json`.
    pub config_path: std::path::PathBuf,
    /// Paths to weight files (safetensors shards or ONNX).
    pub weights_paths: Vec<std::path::PathBuf>,
    /// Path to `tokenizer.json`, if present.
    pub tokenizer_path: Option<std::path::PathBuf>,
    /// Parsed contents of `config.json`.
    pub model_config: serde_json::Value,
    /// Parent model ID for fine-tuned variants.
    pub base_model_id: Option<ModelId>,
    /// Path to LoRA adapter directory (for fine-tuned models).
    pub adapter_path: Option<std::path::PathBuf>,
    /// Estimated GPU memory in bytes (sum of weight file sizes).
    pub estimated_memory: usize,
}

/// Model architecture dimensions used for memory estimation and output sizing.
#[derive(Debug, Clone)]
pub struct ModelDimensions {
    /// Size of the hidden representation (embedding dimension).
    pub hidden_size: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads per layer.
    pub num_attention_heads: usize,
    /// Feed-forward intermediate layer size.
    pub intermediate_size: usize,
}

impl ModelDimensions {
    /// Parse from HuggingFace config.json or OpenCLIP open_clip_config.json.
    pub fn from_config(config: &serde_json::Value) -> Option<Self> {
        // Standard text model format (BERT, ModernBERT, etc.)
        if let Some(hidden_size) = config.get("hidden_size").and_then(|v| v.as_u64()) {
            let hidden_size = hidden_size as usize;
            let num_layers = config
                .get("num_hidden_layers")
                .or(config.get("num_layers"))?
                .as_u64()? as usize;
            let num_attention_heads = config["num_attention_heads"].as_u64()? as usize;
            let intermediate_size = config
                .get("intermediate_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(hidden_size as u64 * 4) as usize;
            return Some(Self {
                hidden_size,
                num_layers,
                num_attention_heads,
                intermediate_size,
            });
        }

        // OpenCLIP format: model_cfg.vision_cfg with embed_dim at top level
        if let Some(model_cfg) = config.get("model_cfg") {
            let vision_cfg = model_cfg.get("vision_cfg")?;
            let embed_dim = model_cfg.get("embed_dim").and_then(|v| v.as_u64())? as usize;
            let width = vision_cfg.get("width").and_then(|v| v.as_u64())? as usize;
            let num_layers = vision_cfg.get("layers").and_then(|v| v.as_u64())? as usize;
            // heads may be absent in OpenCLIP configs — default to width/64 (ViT convention)
            let num_attention_heads = vision_cfg
                .get("heads")
                .and_then(|v| v.as_u64())
                .unwrap_or((width / 64) as u64) as usize;
            let mlp_ratio = vision_cfg
                .get("mlp_ratio")
                .and_then(|v| v.as_f64())
                .unwrap_or(4.0);
            let intermediate_size = (width as f64 * mlp_ratio) as usize;
            return Some(Self {
                hidden_size: embed_dim,
                num_layers,
                num_attention_heads,
                intermediate_size,
            });
        }

        None
    }

    /// Peak activation memory for one inference batch (encoder-only, no gradients).
    pub fn estimate_activation_memory(&self, batch_size: usize, seq_len: usize) -> usize {
        let bytes_per_elem = 4; // f32
        let attention_scores =
            batch_size * self.num_attention_heads * seq_len * seq_len * bytes_per_elem;
        let ffn_intermediate = batch_size * seq_len * self.intermediate_size * bytes_per_elem;
        attention_scores.max(ffn_intermediate)
    }
}

/// A model loaded into memory, ready for inference.
pub enum LoadedModel {
    /// Loaded via the Candle backend (safetensors weights).
    Candle(Box<CandleModel>),
    /// Loaded via the ORT backend (ONNX weights).
    Ort(OrtModel),
}

impl LoadedModel {
    /// Estimate GPU memory for one inference batch.
    pub fn estimate_batch_memory(&self, batch_size: usize, seq_len: usize) -> usize {
        match self {
            LoadedModel::Candle(m) => m.dimensions.estimate_activation_memory(batch_size, seq_len),
            LoadedModel::Ort(m) => m.dimensions.estimate_activation_memory(batch_size, seq_len),
        }
    }

    /// Return the embedding dimension (hidden size), if known.
    pub fn embedding_dim(&self) -> Option<usize> {
        match self {
            LoadedModel::Candle(m) => Some(m.dimensions.hidden_size),
            LoadedModel::Ort(m) => Some(m.dimensions.hidden_size),
        }
    }

    /// Run forward pass on Arrow content columns. Returns raw output.
    pub fn forward(&self, content: &[ArrayRef], task: ModelTask) -> Result<BackendOutput> {
        match self {
            LoadedModel::Candle(m) => m.forward(content, task),
            LoadedModel::Ort(_) => Err(JammiError::Inference(
                "ORT forward pass not available in this build".into(),
            )),
        }
    }
}

/// RAII guard that decrements ref count on drop.
pub struct ModelGuard {
    /// Shared handle to the loaded model.
    pub model: Arc<LoadedModel>,
    ref_count: Arc<AtomicUsize>,
}

impl Drop for ModelGuard {
    fn drop(&mut self) {
        self.ref_count.fetch_sub(1, Ordering::Release);
    }
}
