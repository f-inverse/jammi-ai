pub mod backend;
pub mod cache;
pub mod resolver;
pub mod tokenizer;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use backend::candle::CandleModel;
use backend::ort::OrtModel;
use serde::{Deserialize, Serialize};

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
    Candle,
    Ort,
    Vllm,
    Http,
}

/// What task this model performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelTask {
    Embedding,
    Classification,
    Summarization,
    ObjectDetection,
    Ner,
    TextGeneration,
}

/// A resolved model — files located, backend determined, NOT yet loaded.
pub struct ResolvedModel {
    pub model_id: ModelId,
    pub backend: BackendType,
    pub task: ModelTask,
    pub config_path: std::path::PathBuf,
    pub weights_paths: Vec<std::path::PathBuf>,
    pub tokenizer_path: Option<std::path::PathBuf>,
    pub model_config: serde_json::Value,
    pub base_model_id: Option<ModelId>,
    pub estimated_memory: usize,
}

/// Model configuration for memory estimation.
#[derive(Debug, Clone)]
pub struct ModelDimensions {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
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
    Candle(CandleModel),
    Ort(OrtModel),
}

impl LoadedModel {
    pub fn estimate_batch_memory(&self, batch_size: usize, seq_len: usize) -> usize {
        match self {
            LoadedModel::Candle(m) => m.dimensions.estimate_activation_memory(batch_size, seq_len),
            LoadedModel::Ort(m) => m.dimensions.estimate_activation_memory(batch_size, seq_len),
        }
    }

    pub fn embedding_dim(&self) -> Option<usize> {
        match self {
            LoadedModel::Candle(m) => Some(m.dimensions.hidden_size),
            LoadedModel::Ort(m) => Some(m.dimensions.hidden_size),
        }
    }
}

/// RAII guard that decrements ref count on drop.
pub struct ModelGuard {
    pub model: Arc<LoadedModel>,
    ref_count: Arc<AtomicUsize>,
}

impl Drop for ModelGuard {
    fn drop(&mut self) {
        self.ref_count.fetch_sub(1, Ordering::Release);
    }
}
