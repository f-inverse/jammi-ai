//! Closed-enum dispatch over the two ways jammi-ai trains and persists a
//! LoRA adapter:
//!
//! - [`TrainingTarget::ProjectionHead`] — LoRA wraps a projection head added
//!   on top of a frozen base model's pooled embedding output. The base model
//!   produces embeddings via its own (frozen) forward; the head shifts them.
//!   Works with any [`crate::model::LoadedModel`] backbone.
//! - [`TrainingTarget::EncoderAdapters`] — LoRA is injected into the encoder's
//!   internal attention/FFN linears via [`jammi_encoders::AnyEncoder`].
//!   Requires a jammi-encoders-supported architecture (BERT family).
//!
//! The two variants share no on-disk discriminator at the safetensors level —
//! the persisted [`SavedAdapter`] enum (serde-tagged via `adapter_type`) is
//! the type-level switch.

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use jammi_encoders::AnyEncoder;
use jammi_engine::error::{JammiError, Result};
use jammi_lora::AdapterConfig;
use serde::{Deserialize, Serialize};

use super::lora::LoraModel;

/// What a [`crate::fine_tune::trainer::TrainingLoop`] is training.
///
/// The `EncoderAdapters` variant payload is boxed because [`AnyEncoder`] is
/// substantially larger than a [`LoraModel`] — keeping the variants on a
/// similar size footprint is what [`clippy::large_enum_variant`] is asking
/// for and is the right thing here, not a band-aid.
pub enum TrainingTarget {
    /// LoRA wraps a projection head sitting on top of a frozen base model's
    /// pooled output. The head is one or two [`jammi_lora::LoraLinear`]
    /// layers (projection-only for embeddings; projection + classifier for
    /// classification / NER).
    ProjectionHead { head: LoraModel },
    /// LoRA is injected into the encoder's internal attention/FFN linears.
    EncoderAdapters(Box<EncoderAdaptersTarget>),
}

/// Payload of [`TrainingTarget::EncoderAdapters`].
pub struct EncoderAdaptersTarget {
    /// The LoRA-injected encoder used both during training and as the
    /// blueprint inference must rebuild from the saved tensors.
    pub encoder: AnyEncoder,
    /// Metadata persisted alongside the trained A/B weights so inference
    /// can rebuild the encoder with the adapter installed.
    pub adapter_cfg: AdapterConfig,
}

impl TrainingTarget {
    /// References to every trainable tensor in this target.
    pub fn trainable_params(&self) -> Vec<&Tensor> {
        match self {
            Self::ProjectionHead { head } => head.trainable_params(),
            Self::EncoderAdapters(state) => state.encoder.trainable_params(),
        }
    }

    /// Toggle training mode. Used to disable LoRA dropout during validation
    /// passes; no-op for variants whose layers don't use dropout.
    pub fn set_training(&mut self, training: bool) {
        match self {
            Self::ProjectionHead { head } => {
                for (_, layer) in head.layers.iter_mut() {
                    layer.set_training(training);
                }
            }
            Self::EncoderAdapters(state) => state.encoder.set_training(training),
        }
    }

    /// Collect all trainable A/B tensors as a CPU-side `HashMap` ready for
    /// safetensors serialisation. Keys follow the convention each variant
    /// uses to label its trainable sites.
    pub fn named_trainable_weights(&self) -> Result<HashMap<String, Tensor>> {
        match self {
            Self::ProjectionHead { head } => {
                let mut out = HashMap::new();
                for (name, layer) in &head.layers {
                    out.insert(
                        format!("{name}.lora_a"),
                        layer.lora_a.to_device(&Device::Cpu).map_err(|e| {
                            JammiError::FineTune(format!("Move lora_a to CPU: {e}"))
                        })?,
                    );
                    out.insert(
                        format!("{name}.lora_b"),
                        layer.lora_b.to_device(&Device::Cpu).map_err(|e| {
                            JammiError::FineTune(format!("Move lora_b to CPU: {e}"))
                        })?,
                    );
                }
                Ok(out)
            }
            Self::EncoderAdapters(state) => state
                .encoder
                .named_trainable_weights()
                .map_err(|e| JammiError::FineTune(format!("Encoder trainable weights: {e}"))),
        }
    }

    /// Apply a previously-saved `HashMap<key, tensor>` back into the target's
    /// trainable sites. Used to restore the best checkpoint at end of
    /// training.
    pub fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        match self {
            Self::ProjectionHead { head } => {
                for (name, layer) in head.layers.iter_mut() {
                    if let Some(a) = weights.get(&format!("{name}.lora_a")) {
                        layer.lora_a = a.clone();
                    }
                    if let Some(b) = weights.get(&format!("{name}.lora_b")) {
                        layer.lora_b = b.clone();
                    }
                }
                Ok(())
            }
            Self::EncoderAdapters(state) => state
                .encoder
                .load_weights(weights)
                .map_err(|e| JammiError::FineTune(format!("Encoder load weights: {e}"))),
        }
    }

    /// Build the persisted metadata for this target.
    pub fn saved_adapter(&self, config: &super::FineTuneConfig) -> SavedAdapter {
        match self {
            Self::ProjectionHead { head } => SavedAdapter::ProjectionHead(ProjectionHeadConfig {
                lora_rank: config.lora_rank,
                lora_alpha: config.lora_alpha,
                head_layers: head.layers.iter().map(|(name, _)| name.clone()).collect(),
            }),
            Self::EncoderAdapters(state) => {
                SavedAdapter::EncoderAdapters(Box::new(state.adapter_cfg.clone()))
            }
        }
    }
}

/// Persisted metadata serialised into `adapter_config.json` for both adapter
/// flavours jammi-ai produces.
///
/// The `adapter_type` JSON field discriminates which variant a saved adapter
/// is, replacing the implicit "presence/absence of config.json" discriminator
/// the previous projection-only format relied on.
///
/// `EncoderAdapters` boxes its payload because [`AdapterConfig`] is
/// substantially larger than [`ProjectionHeadConfig`] (multiple `Vec`s,
/// a `HashMap`, and a string), and unboxed it would inflate every
/// `SavedAdapter` value to the larger size regardless of variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "adapter_type", rename_all = "snake_case")]
pub enum SavedAdapter {
    /// Adapter is a projection head on top of a frozen base model. The
    /// `adapter.safetensors` keys are `{head_layer}.lora_a` /
    /// `{head_layer}.lora_b` for every entry in `head_layers`.
    ProjectionHead(ProjectionHeadConfig),
    /// Adapter is a set of LoRA A/B tensors injected into an encoder's
    /// attention/FFN linears.
    EncoderAdapters(Box<AdapterConfig>),
}

/// Metadata for a projection-head adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionHeadConfig {
    /// LoRA rank used at training time.
    pub lora_rank: usize,
    /// LoRA scaling factor used at training time.
    pub lora_alpha: f64,
    /// Names of the head layers, in order. Each name keys two safetensors
    /// tensors: `{name}.lora_a` and `{name}.lora_b`. Typical shapes:
    /// - embeddings: `["projection"]`
    /// - classification: `["projection", "classifier"]`
    /// - NER: `["projection", "token_classifier"]`
    pub head_layers: Vec<String>,
}
