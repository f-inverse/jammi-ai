//! Projection-head assemblies for
//! [`crate::fine_tune::target::TrainingTarget::ProjectionHead`]: a single
//! projection layer for embedding fine-tunes, or a projection plus a
//! classifier head for classification / NER. All three builders return a
//! [`LoraModel`] — a flat sequence of named [`jammi_lora::LoraLinear`]
//! layers that the trainer consumes uniformly.

use candle_core::{DType, Tensor};
use candle_nn::{Linear, VarBuilder};
use jammi_engine::error::{JammiError, Result};
use jammi_lora::LoraLinear;

/// A sequence of named LoRA layers — the trainable state of a
/// [`TrainingTarget::ProjectionHead`] target.
///
/// [`TrainingTarget::ProjectionHead`]: crate::fine_tune::target::TrainingTarget::ProjectionHead
pub struct LoraModel {
    /// LoRA layers keyed by their name (e.g. `"projection"`, `"classifier"`).
    pub layers: Vec<(String, LoraLinear)>,
}

impl LoraModel {
    /// Collect all trainable parameters across all LoRA layers.
    pub fn trainable_params(&self) -> Vec<&Tensor> {
        self.layers
            .iter()
            .flat_map(|(_, layer)| layer.trainable_params())
            .collect()
    }
}

/// Build a projection-plus-classifier head for classification fine-tunes.
///
/// Layer 0 (`projection`): LoRA-wrapped identity, `hidden → hidden`.
/// Layer 1 (`classifier`): LoRA-wrapped zeros, `hidden → num_classes`.
pub fn build_classification_head(
    hidden_size: usize,
    num_classes: usize,
    config: &super::FineTuneConfig,
    vb: &VarBuilder,
) -> Result<LoraModel> {
    // Layer 0: projection (identity base, same as embedding)
    let proj_base = Tensor::eye(hidden_size, DType::F32, vb.device())
        .map_err(|e| JammiError::FineTune(format!("Projection identity: {e}")))?;
    let proj_linear = Linear::new(proj_base, None);
    let projection = LoraLinear::new_simple(
        proj_linear,
        config.lora_rank,
        config.lora_alpha,
        &vb.pp("projection"),
    )
    .map_err(|e| JammiError::FineTune(format!("Projection LoRA: {e}")))?;

    // Layer 1: classifier (zeros base, trained from scratch via LoRA)
    let cls_base = Tensor::zeros((num_classes, hidden_size), DType::F32, vb.device())
        .map_err(|e| JammiError::FineTune(format!("Classifier zeros: {e}")))?;
    let cls_linear = Linear::new(cls_base, None);
    let classifier = LoraLinear::new_simple(
        cls_linear,
        config.lora_rank,
        config.lora_alpha,
        &vb.pp("classifier"),
    )
    .map_err(|e| JammiError::FineTune(format!("Classifier LoRA: {e}")))?;

    Ok(LoraModel {
        layers: vec![
            ("projection".to_string(), projection),
            ("classifier".to_string(), classifier),
        ],
    })
}

/// Build a projection-plus-token-classifier head for NER fine-tunes.
///
/// Layer 0 (`projection`): LoRA-wrapped identity, `hidden → hidden`, per token.
/// Layer 1 (`token_classifier`): LoRA-wrapped zeros, `hidden → num_labels`, per token.
pub fn build_ner_head(
    hidden_size: usize,
    num_labels: usize,
    config: &super::FineTuneConfig,
    vb: &VarBuilder,
) -> Result<LoraModel> {
    // Layer 0: projection (identity base, same as embedding/classification)
    let proj_base = Tensor::eye(hidden_size, DType::F32, vb.device())
        .map_err(|e| JammiError::FineTune(format!("NER projection identity: {e}")))?;
    let proj_linear = Linear::new(proj_base, None);
    let projection = LoraLinear::new_simple(
        proj_linear,
        config.lora_rank,
        config.lora_alpha,
        &vb.pp("projection"),
    )
    .map_err(|e| JammiError::FineTune(format!("NER projection LoRA: {e}")))?;

    // Layer 1: token classifier (zeros base, trained from scratch via LoRA)
    let cls_base = Tensor::zeros((num_labels, hidden_size), DType::F32, vb.device())
        .map_err(|e| JammiError::FineTune(format!("NER classifier zeros: {e}")))?;
    let cls_linear = Linear::new(cls_base, None);
    let classifier = LoraLinear::new_simple(
        cls_linear,
        config.lora_rank,
        config.lora_alpha,
        &vb.pp("token_classifier"),
    )
    .map_err(|e| JammiError::FineTune(format!("NER classifier LoRA: {e}")))?;

    Ok(LoraModel {
        layers: vec![
            ("projection".to_string(), projection),
            ("token_classifier".to_string(), classifier),
        ],
    })
}

/// Build a single-layer head for embedding fine-tunes.
///
/// One `projection` LoRA layer wrapping an identity weight
/// (`hidden → hidden`). At init, `B = 0` and the layer acts as identity,
/// so the fine-tuned model starts producing the same embeddings as the
/// frozen base.
pub fn build_projection_head(
    hidden_size: usize,
    config: &super::FineTuneConfig,
    vb: &VarBuilder,
) -> Result<LoraModel> {
    let base_weight = Tensor::eye(hidden_size, DType::F32, vb.device())
        .map_err(|e| JammiError::FineTune(format!("Identity weight: {e}")))?;
    let base = Linear::new(base_weight, None);
    let lora = LoraLinear::new_simple(
        base,
        config.lora_rank,
        config.lora_alpha,
        &vb.pp("projection"),
    )
    .map_err(|e| JammiError::FineTune(format!("Projection LoRA: {e}")))?;
    Ok(LoraModel {
        layers: vec![("projection".into(), lora)],
    })
}
