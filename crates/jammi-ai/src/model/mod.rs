pub mod backend;
pub mod cache;
pub mod resolver;
pub mod tokenizer;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use arrow::array::ArrayRef;
use backend::candle::CandleModel;
use backend::ort::OrtModel;
use jammi_engine::error::{JammiError, Result};
use serde::{Deserialize, Serialize};

use crate::inference::adapter::BackendOutput;

/// Unique identifier for a loaded model.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ModelId(pub String);

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
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
    /// vLLM — high-throughput serving for large language models.
    Vllm,
    /// HTTP — remote model endpoint (REST/gRPC).
    Http,
}

/// What task this model performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelTask {
    /// Produce dense vector representations of input text.
    Embedding,
    /// Assign a label and confidence score to input text.
    Classification,
    /// Generate a condensed summary of input text.
    Summarization,
    /// Detect and localize objects in images.
    ObjectDetection,
    /// Extract named entities (person, org, location, etc.) from text.
    Ner,
    /// Generate continuation text from a prompt.
    TextGeneration,
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
    /// Parse from HuggingFace config.json.
    pub fn from_config(config: &serde_json::Value) -> Option<Self> {
        let hidden_size = config["hidden_size"].as_u64()? as usize;
        let num_layers = config
            .get("num_hidden_layers")
            .or(config.get("num_layers"))?
            .as_u64()? as usize;
        let num_attention_heads = config["num_attention_heads"].as_u64()? as usize;
        let intermediate_size = config
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(hidden_size as u64 * 4) as usize;
        Some(Self {
            hidden_size,
            num_layers,
            num_attention_heads,
            intermediate_size,
        })
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
