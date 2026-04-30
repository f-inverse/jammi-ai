//! PEFT-style deep LoRA: selectively apply LoRA adapters to named linear layers
//! inside the encoder stack rather than only to an external projection.
//!
//! Architecture overview
//! ─────────────────────
//! `DeepLoraEncoder` (trait) — forward pass + named weight export
//! `DeepLoraModel`           — owns a `Box<dyn DeepLoraEncoder>` + config metadata
//!
//! Concrete encoder implementations:
//!   `bert.rs`       — BERT / RoBERTa / CamemBERT / XLM-RoBERTa
//!   `distilbert.rs` — DistilBERT
//!   `modernbert.rs` — ModernBERT

pub mod bert;
pub mod distilbert;
pub mod modernbert;

use std::collections::HashMap;
use std::path::Path;

use candle_core::Tensor;
use jammi_engine::error::{JammiError, Result};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// Core trait
// ─────────────────────────────────────────────────────────────────────────────

/// An encoder that runs a forward pass through (a subset of) transformer layers,
/// with LoRA adapters injected into the named linear projections.
pub trait DeepLoraEncoder: Send + Sync {
    /// Run a forward pass and return mean-pooled, L2-normalised embeddings.
    ///
    /// `input_ids`      — `[batch, seq_len]` u32 token IDs  
    /// `attention_mask` — `[batch, seq_len]` u32 mask (1 = real token)
    fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor>;

    /// Return references to all trainable LoRA parameters (A and B matrices).
    fn trainable_params(&self) -> Vec<&Tensor>;

    /// Return all trainable tensors as a name→tensor map, with tensors moved to
    /// CPU.  Used when saving adapter weights to disk.
    fn named_trainable_weights(&self) -> Result<HashMap<String, Tensor>>;

    /// Set training vs evaluation mode on every LoRA layer inside the encoder.
    ///
    /// In evaluation mode (`false`) LoRA dropout is disabled, ensuring
    /// deterministic outputs during validation and inference.
    fn set_training(&mut self, training: bool);

    /// Load LoRA A/B tensors from a name→tensor map (as produced by
    /// `named_trainable_weights`) into the encoder's LoRA layers in place.
    /// Used to restore checkpoints during training.
    fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Adapter config (written to disk alongside adapter weights)
// ─────────────────────────────────────────────────────────────────────────────

/// Metadata persisted in `adapter_config.json` for every deep-LoRA run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepLoraAdapterConfig {
    pub adapter_type: String,
    pub model_type: String,
    pub lora_rank: usize,
    pub lora_alpha: f64,
    pub use_rslora: bool,
    pub target_modules: Vec<String>,
    #[serde(default)]
    pub layers_to_transform: Option<Vec<usize>>,
    #[serde(default)]
    pub rank_pattern: HashMap<String, usize>,
    /// Dtype used for the frozen backbone at training time. Defaults to "f32"
    /// for configs written before this field was added.
    #[serde(default)]
    pub backbone_dtype: super::BackboneDtype,
}

impl DeepLoraAdapterConfig {
    pub fn new_deep_lora(
        model_type: &str,
        lora_rank: usize,
        lora_alpha: f64,
        use_rslora: bool,
        target_modules: Vec<String>,
        layers_to_transform: Option<Vec<usize>>,
        rank_pattern: HashMap<String, usize>,
        backbone_dtype: super::BackboneDtype,
    ) -> Self {
        Self {
            adapter_type: "deep_lora".into(),
            model_type: model_type.into(),
            lora_rank,
            lora_alpha,
            use_rslora,
            target_modules,
            layers_to_transform,
            rank_pattern,
            backbone_dtype,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Wrapper struct
// ─────────────────────────────────────────────────────────────────────────────

/// Wraps a concrete `DeepLoraEncoder` alongside the adapter metadata needed
/// for serialisation and inference re-loading.
pub struct DeepLoraModel {
    pub encoder: Box<dyn DeepLoraEncoder>,
    pub config: DeepLoraAdapterConfig,
}

impl DeepLoraModel {
    pub fn new(encoder: Box<dyn DeepLoraEncoder>, config: DeepLoraAdapterConfig) -> Self {
        Self { encoder, config }
    }

    /// Forward pass delegated to the inner encoder.
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        self.encoder.forward(input_ids, attention_mask)
    }

    /// All trainable LoRA parameters for the optimizer.
    pub fn trainable_params(&self) -> Vec<&Tensor> {
        self.encoder.trainable_params()
    }

    /// Set training vs evaluation mode on every LoRA layer inside the encoder.
    pub fn set_training(&mut self, training: bool) {
        self.encoder.set_training(training);
    }

    /// Restore LoRA weights from a checkpoint safetensors file into the
    /// encoder's LoRA layers in place.
    pub fn load_from_checkpoint(
        &mut self,
        path: &Path,
        device: &candle_core::Device,
    ) -> Result<()> {
        let weights = candle_core::safetensors::load(path, device)
            .map_err(|e| JammiError::FineTune(format!("Load deep LoRA checkpoint: {e}")))?;
        self.encoder.load_weights(&weights)
    }

    // ── Persistence ──────────────────────────────────────────────────────────

    /// Save LoRA A/B matrices as `adapter.safetensors` and write
    /// `adapter_config.json` into `dir`.
    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;

        let weights = self.encoder.named_trainable_weights()?;
        candle_core::safetensors::save(&weights, &dir.join("adapter.safetensors"))
            .map_err(|e| JammiError::FineTune(format!("Save deep_lora safetensors: {e}")))?;

        let cfg_json = serde_json::to_string_pretty(&self.config)
            .map_err(|e| JammiError::FineTune(format!("Serialize adapter_config: {e}")))?;
        std::fs::write(dir.join("adapter_config.json"), cfg_json)?;

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Module-name matching helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Decide whether a linear layer at `layer_idx` with the given `module_name`
/// suffix should receive a LoRA adapter.
///
/// Rules (mirroring PEFT):
/// 1. If `target_modules` contains `"all-linear"`, always return true.
/// 2. Otherwise, `module_name` must equal or end with one of the target strings.
/// 3. If `layers_to_transform` is `Some(indices)`, `layer_idx` must be in the list.
pub fn should_apply_lora(
    module_name: &str,
    target_modules: &[String],
    layer_idx: usize,
    layers_to_transform: &Option<Vec<usize>>,
) -> bool {
    if let Some(indices) = layers_to_transform {
        if !indices.contains(&layer_idx) {
            return false;
        }
    }

    if target_modules.iter().any(|t| t == "all-linear") {
        return true;
    }
    target_modules
        .iter()
        .any(|t| module_name == t.as_str() || module_name.ends_with(t.as_str()))
}

/// Look up the effective LoRA rank for a module name, consulting `rank_pattern`
/// first and falling back to `default_rank`.
pub fn effective_rank(
    module_name: &str,
    default_rank: usize,
    rank_pattern: &HashMap<String, usize>,
) -> usize {
    rank_pattern
        .iter()
        .find(|(k, _)| module_name.contains(k.as_str()))
        .map(|(_, &r)| r)
        .unwrap_or(default_rank)
}
