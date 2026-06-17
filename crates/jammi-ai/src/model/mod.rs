pub mod backend;
pub mod cache;
pub mod clip_bpe;
pub mod resolver;
pub mod tokenizer;

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use arrow::array::ArrayRef;
use backend::candle::CandleModel;
use backend::ort::OrtModel;
use jammi_db::error::{JammiError, Result};
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
    ///
    /// - `"local:/path/to/model"` → `Local(path)`
    /// - `"hf://owner/repo"` → `HuggingFace("owner/repo")` (strips `hf://`)
    /// - `"owner/repo"` → `HuggingFace("owner/repo")`
    pub fn parse(id: &str) -> Self {
        if let Some(path) = id.strip_prefix("local:") {
            Self::Local(std::path::PathBuf::from(path))
        } else if let Some(repo_id) = id.strip_prefix("hf://") {
            Self::HuggingFace(repo_id.to_string())
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
///
/// Re-exported from `jammi_db` so the engine — which owns the catalog
/// tables that persist this — and `jammi_ai` agree on the variant set and
/// on-disk spelling without `jammi_db` depending on `jammi_ai`.
pub use jammi_db::ModelTask;

/// Where the tokenizer for a resolved model lives, and what shape it is.
///
/// Most checkpoints carry an HF-converted `tokenizer.json`; stock OpenCLIP
/// repos instead ship the legacy gzipped BPE vocab. The resolver picks
/// whichever is present and the loader dispatches on the variant.
#[derive(Debug, Clone)]
pub enum TokenizerSource {
    /// HuggingFace-shape `tokenizer.json` (works for BERT-family, ModernBERT,
    /// DistilBERT, and OpenCLIP repos that ship a pre-converted file).
    HuggingFaceJson(std::path::PathBuf),
    /// OpenCLIP-native `bpe_simple_vocab_16e6.txt.gz` — built directly into a
    /// BPE tokenizer at load time, no HF pre-conversion required.
    OpenClipBpe(std::path::PathBuf),
}

impl TokenizerSource {
    /// Filesystem path of the tokenizer artifact.
    pub fn path(&self) -> &std::path::Path {
        match self {
            Self::HuggingFaceJson(p) | Self::OpenClipBpe(p) => p,
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
    /// Tokenizer source (HF JSON or OpenCLIP BPE), if present.
    pub tokenizer: Option<TokenizerSource>,
    /// Parsed contents of `config.json`.
    pub model_config: serde_json::Value,
    /// Parsed contents of `preprocessor_config.json`, if present. Carries the
    /// feature-extractor geometry (CLAP fusion front-end: sample rate, FFT
    /// window, hop, mel-filter band, max length) the audio path needs so the
    /// bytes-to-spectrogram transform is config-driven, not hardcoded.
    pub preprocessor_config: Option<serde_json::Value>,
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
        // HF-CLAP audio tower (`ClapAudioModelWithProjection`): top-level
        // `clap_audio_model` config (or a nested `audio_config` block under a
        // top-level `ClapConfig`). Its embedding dimensionality is
        // `projection_dim`; `num_attention_heads`/`depths` are per-stage arrays,
        // so the standard scalar-`num_attention_heads` text branch cannot parse
        // it — detect it first off `model_type`.
        if let Some(dims) = Self::from_hf_clap_config(config) {
            return Some(dims);
        }

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

    /// Parse the HF-CLAP audio-tower geometry (`ClapAudioModelWithProjection`),
    /// returning `None` for any non-CLAP config.
    ///
    /// Accepts both the flat `clap_audio_model` config and a top-level
    /// `ClapConfig` carrying a nested `audio_config`. The reported
    /// `hidden_size` is the tower's output embedding dimensionality
    /// (`projection_dim`, the shared cross-modal latent); `num_layers` is the
    /// number of hierarchical Swin stages (`depths.len()`); attention heads and
    /// the intermediate FFN size are taken from the final stage (the widest),
    /// which bounds the per-batch activation footprint.
    fn from_hf_clap_config(config: &serde_json::Value) -> Option<Self> {
        let audio = config.get("audio_config").unwrap_or(config);
        if audio.get("model_type").and_then(|v| v.as_str()) != Some("clap_audio_model") {
            return None;
        }
        let projection_dim = config
            .get("projection_dim")
            .or_else(|| audio.get("projection_dim"))
            .and_then(|v| v.as_u64())? as usize;
        let final_width = audio.get("hidden_size").and_then(|v| v.as_u64())? as usize;
        let depths = audio.get("depths").and_then(|v| v.as_array())?;
        let num_layers = depths.len();
        let num_attention_heads = audio
            .get("num_attention_heads")
            .and_then(|v| v.as_array())
            .and_then(|h| h.last())
            .and_then(|v| v.as_u64())? as usize;
        let mlp_ratio = audio
            .get("mlp_ratio")
            .and_then(|v| v.as_f64())
            .unwrap_or(4.0);
        let intermediate_size = (final_width as f64 * mlp_ratio) as usize;
        Some(Self {
            hidden_size: projection_dim,
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

    /// Output dimensionality of the model's embedding head, if known.
    ///
    /// For BERT-family encoders this is the transformer's `hidden_size`.
    /// For OpenCLIP-family models (vision and text towers) this is the
    /// projected shared-latent `embed_dim` — the dimension that vectors
    /// emitted by `generate_text_embeddings`, `generate_image_embeddings`,
    /// `encode_text_query`, and `encode_image_query` carry, and the
    /// dimension that cross-modal cosine similarity is computed in. It is
    /// not the per-tower hidden `width`; the in-tower hidden size is
    /// projected through `visual.proj` / `text_projection` before the
    /// embedding is exposed.
    pub fn embedding_dim(&self) -> Option<usize> {
        match self {
            LoadedModel::Candle(m) => Some(m.dimensions.hidden_size),
            LoadedModel::Ort(m) => Some(m.dimensions.hidden_size),
        }
    }

    /// The persisted predictive-distribution form of a reloaded regression head
    /// (`Gaussian` or `Quantile { levels }`), or `None` for a non-regression
    /// model. Serving reads this to select the `Infer` output adapter so a
    /// quantile-trained head is served as quantile points, never silently
    /// mis-decoded as a Gaussian `(mean, std)`. The ORT backend has no
    /// regression head, so it always reports `None`.
    pub fn regression_form(&self) -> Option<&crate::inference::adapter::DistributionForm> {
        match self {
            LoadedModel::Candle(m) => m.regression_form(),
            LoadedModel::Ort(_) => None,
        }
    }

    /// The persisted scaler's σ_y for a reloaded regression head, or `None` for a
    /// non-regression / no-scaler / ORT model. Serving reads this to scale a
    /// Gaussian head's served σ from the z-space the loss trained (σ_z ≈ 1) back
    /// to raw units (`σ_y·σ_z`) — the σ-axis half of the de-standardise contract
    /// (the mean/quantile axes carry σ_y in the backend's affine).
    pub fn regression_std_scale(&self) -> Option<f32> {
        match self {
            LoadedModel::Candle(m) => m.regression_std_scale(),
            LoadedModel::Ort(_) => None,
        }
    }

    /// TEST-ONLY non-vacuity seam: zero a loaded regression head's trained LoRA
    /// `B` factor so it regresses to its zero-initialised base and emits the
    /// scaler offset `μ_y` for every input (the untrained-head behaviour). No-op
    /// for a non-regression / ORT model. Used by the regression-surface tests to
    /// prove their group-separation assertion collapses to ≈0 when the head
    /// carries no learned signal. See
    /// [`super::backend::candle::CandleModel::zero_distribution_head_for_test`].
    #[doc(hidden)]
    pub fn zero_distribution_head_for_test(&mut self) {
        match self {
            LoadedModel::Candle(m) => m.zero_distribution_head_for_test(),
            LoadedModel::Ort(_) => {}
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
