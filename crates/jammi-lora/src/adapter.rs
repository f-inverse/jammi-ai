//! Persisted adapter metadata (`adapter_config.json` contents).

use std::collections::HashMap;

#[cfg(feature = "candle")]
use candle_core::DType;
use serde::{Deserialize, Serialize};

use crate::config::LoraBuildConfig;

/// Dtype used for the frozen backbone at training time.
///
/// `BF16` halves backbone memory with negligible impact on training dynamics
/// because the backbone weights are frozen. The trainable LoRA A and B
/// matrices always stay in F32.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackboneDtype {
    /// Full precision — maximally compatible.
    #[default]
    F32,
    /// BFloat16 — recommended for CUDA/Metal; cuts backbone VRAM by ~half.
    BF16,
    /// Half-precision float — compatible with most CUDA devices.
    F16,
}

/// The `BackboneDtype -> candle DType` mapping is the one place this config enum
/// touches candle, so it rides the `candle` feature; the config-vocabulary build
/// keeps the enum without the tensor stack.
#[cfg(feature = "candle")]
impl From<BackboneDtype> for DType {
    fn from(d: BackboneDtype) -> Self {
        match d {
            BackboneDtype::F32 => DType::F32,
            BackboneDtype::BF16 => DType::BF16,
            BackboneDtype::F16 => DType::F16,
        }
    }
}

/// Metadata describing a LoRA adapter injected into an encoder's internal
/// attention/FFN linears.
///
/// Persisted as JSON alongside `adapter.safetensors`. Discrimination between
/// different *kinds* of adapters (e.g. an external projection head vs. these
/// internal adapters) is a concern for the caller; this struct describes the
/// internal-adapter case only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterConfig {
    /// Backbone model family: e.g. `"bert"`, `"distilbert"`, `"modernbert"`.
    pub model_type: String,
    /// Default LoRA rank used at training time.
    pub lora_rank: usize,
    /// LoRA scaling factor used at training time.
    pub lora_alpha: f64,
    /// Whether RSLoRA-style `alpha / sqrt(rank)` scaling was used.
    pub use_rslora: bool,
    /// Module-name suffixes that received a LoRA adapter.
    pub target_modules: Vec<String>,
    /// Optional restriction of LoRA injection to specific layer indices.
    #[serde(default)]
    pub layers_to_transform: Option<Vec<usize>>,
    /// Per-module rank overrides keyed by module-name substring.
    #[serde(default)]
    pub rank_pattern: HashMap<String, usize>,
    /// Dtype used for the frozen backbone at training time. Defaults to F32.
    #[serde(default)]
    pub backbone_dtype: BackboneDtype,
}

impl AdapterConfig {
    /// Snapshot an `AdapterConfig` from a build-time `LoraBuildConfig`.
    ///
    /// Run-time-only fields (`lora_dropout`, `init_mode`) are intentionally
    /// not persisted — they affect training behaviour but do not change the
    /// shape or semantics of the loaded adapter weights.
    pub fn from_build(
        model_type: &str,
        lora: &LoraBuildConfig<'_>,
        backbone_dtype: BackboneDtype,
    ) -> Self {
        Self {
            model_type: model_type.into(),
            lora_rank: lora.lora_rank,
            lora_alpha: lora.lora_alpha,
            use_rslora: lora.use_rslora,
            target_modules: lora.target_modules.to_vec(),
            layers_to_transform: lora.layers_to_transform.clone(),
            rank_pattern: lora.rank_pattern.clone(),
            backbone_dtype,
        }
    }
}
