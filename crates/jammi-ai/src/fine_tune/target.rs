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
use jammi_db::error::{JammiError, Result};
use jammi_encoders::AnyEncoder;
use jammi_lora::AdapterConfig;
use serde::{Deserialize, Serialize};

use super::lora::LoraModel;
use super::regression_loss::TargetScaler;

/// What a [`crate::fine_tune::trainer::TrainingLoop`] is training.
///
/// The `EncoderAdapters` variant payload is boxed because [`AnyEncoder`] is
/// substantially larger than a [`LoraModel`] — keeping the variants on a
/// similar size footprint is what `clippy::large_enum_variant` is asking
/// for and is the right thing here, not a band-aid.
pub enum TrainingTarget {
    /// LoRA wraps a projection head sitting on top of a frozen base model's
    /// pooled output. The head is one or two [`jammi_lora::LoraLinear`]
    /// layers: a single `projection` layer for embedding fine-tunes, or a
    /// `projection` + `classifier` pair for classification / NER.
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
    ///
    /// `target_scaler` is the regression head's de-standardising affine, present
    /// only for a `ModelTask::Regression` projection head. It is persisted in the
    /// adapter config so a reloaded head de-standardises identically — the head's
    /// raw output is the one transform shared by training and serving.
    pub(crate) fn saved_adapter(
        &self,
        config: &super::FineTuneConfig,
        target_scaler: Option<TargetScaler>,
    ) -> SavedAdapter {
        match self {
            Self::ProjectionHead { head } => SavedAdapter::ProjectionHead(ProjectionHeadConfig {
                lora_rank: config.lora_rank,
                lora_alpha: config.lora_alpha,
                head_layers: head.layers.iter().map(|(name, _)| name.clone()).collect(),
                target_scaler,
            }),
            Self::EncoderAdapters(state) => {
                SavedAdapter::EncoderAdapters(Box::new(state.adapter_cfg.clone()))
            }
        }
    }
}

/// Persisted metadata serialised into `adapter_config.json` for both adapter
/// flavours jammi-ai produces. The `adapter_type` JSON field tags which
/// variant a saved adapter is.
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
    /// De-standardising affine for a regression head: `Some(μ_y, σ_y)` for a
    /// `ModelTask::Regression` head, `None` for every other projection head.
    /// Serving rebuilds it so the reloaded head emits the same raw, outcome-unit
    /// distribution parameters the trained head did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) target_scaler: Option<TargetScaler>,
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};
    use tempfile::tempdir;

    use super::super::regression_loss::TargetScaler;
    use super::{ProjectionHeadConfig, SavedAdapter};

    /// A regression head's de-standardising affine must survive the
    /// save → reload round-trip through the adapter directory, so a served head
    /// emits the target offset rather than the zero-init z-space value.
    ///
    /// This guards scaler persistence end to end: build a `ProjectionHeadConfig`
    /// carrying a non-trivial `TargetScaler` (μ_y ≈ 2017, σ_y ≈ 2 — the
    /// calendar-year shape), persist it with the real `save_adapter` writer,
    /// reload it with the real `load_adapter` reader, then apply the reloaded
    /// scaler's `destandardize_gaussian` — exactly what serving's
    /// `forward_regression` does — to a ZERO z-space head. The served mean must
    /// come back at μ_y (≈2017), NOT 0; a dropped scaler would serve ≈0.
    #[test]
    fn regression_target_scaler_survives_adapter_round_trip() {
        let device = Device::Cpu;
        // Calendar years 2014..=2020 — the high-offset, low-variance shape.
        let years = Tensor::from_vec(
            vec![2014.0f32, 2015.0, 2016.0, 2017.0, 2018.0, 2019.0, 2020.0],
            (7,),
            &device,
        )
        .unwrap();
        let true_mean = 2017.0f32;
        let scaler = TargetScaler::from_targets(&years).unwrap();

        let saved = SavedAdapter::ProjectionHead(ProjectionHeadConfig {
            lora_rank: 4,
            lora_alpha: 8.0,
            head_layers: vec!["projection".into(), "distribution".into()],
            target_scaler: Some(scaler),
        });

        // A two-tensor weights map so `save_adapter` writes a valid safetensors.
        let mut weights = std::collections::HashMap::new();
        weights.insert(
            "distribution.lora_a".to_string(),
            Tensor::zeros((4, 8), candle_core::DType::F32, &device).unwrap(),
        );
        weights.insert(
            "distribution.lora_b".to_string(),
            Tensor::zeros((2, 4), candle_core::DType::F32, &device).unwrap(),
        );

        let dir = tempdir().unwrap();
        jammi_lora::save_adapter(dir.path(), &weights, &saved).unwrap();
        let (reloaded, _tensors): (SavedAdapter, _) =
            jammi_lora::load_adapter(dir.path(), &device).unwrap();

        let reloaded_scaler = match reloaded {
            SavedAdapter::ProjectionHead(cfg) => cfg
                .target_scaler
                .expect("the reloaded regression head must carry its persisted target scaler"),
            SavedAdapter::EncoderAdapters(_) => panic!("round-trip changed the adapter variant"),
        };

        // The served head: a ZERO z-space Gaussian head (mean column 0, raw_std 0),
        // de-standardised exactly as `forward_regression` does on serve.
        let z_head = Tensor::zeros((1, 2), candle_core::DType::F32, &device).unwrap();
        let served = reloaded_scaler.destandardize_gaussian(&z_head).unwrap();
        let served_mean = served.to_vec2::<f32>().unwrap()[0][0];

        assert!(
            (served_mean - true_mean).abs() < 1.0,
            "reloaded scaler must de-standardise the served mean to μ_y ≈ {true_mean}, \
             got {served_mean} (a dropped scaler would serve ≈0)"
        );
        assert!(
            served_mean.abs() > 100.0,
            "served mean {served_mean} collapsed toward the zero-init z-space value — \
             the persisted de-standardisation was lost"
        );
    }
}
