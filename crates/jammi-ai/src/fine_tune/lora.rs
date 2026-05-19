//! LoRA model assembly and adapter persistence helpers built on top of
//! [`jammi_lora`].
//!
//! The single-layer LoRA primitive (`LoraLinear`), the init-mode enum, and the
//! `MaybeLoraLinear` wrapper now live in `jammi-lora`. This module provides the
//! jammi-ai-specific assemblies — projection-only / classification / NER
//! adapter stacks — and the safetensors save/load helpers keyed by the per-job
//! layer names jammi-ai uses.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, VarBuilder};
use jammi_engine::error::{JammiError, Result};
use jammi_lora::LoraLinear;

/// A collection of LoRA layers applied to a model.
pub struct LoraModel {
    /// LoRA layers keyed by their path in the model (e.g. "encoder.layer.0.attention.query").
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

/// Save LoRA adapter weights (only A and B matrices) as safetensors.
pub fn save_lora_weights(model: &LoraModel, path: &Path) -> Result<()> {
    let mut tensors = HashMap::new();
    for (name, layer) in &model.layers {
        tensors.insert(
            format!("{name}.lora_a"),
            layer
                .lora_a
                .to_device(&Device::Cpu)
                .map_err(|e| JammiError::FineTune(format!("Save A: {e}")))?,
        );
        tensors.insert(
            format!("{name}.lora_b"),
            layer
                .lora_b
                .to_device(&Device::Cpu)
                .map_err(|e| JammiError::FineTune(format!("Save B: {e}")))?,
        );
    }
    candle_core::safetensors::save(&tensors, path)
        .map_err(|e| JammiError::FineTune(format!("Save safetensors: {e}")))
}

/// Load LoRA adapter weights from a safetensors file.
pub fn load_lora_weights(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    candle_core::safetensors::load(path, device)
        .map_err(|e| JammiError::FineTune(format!("Load safetensors: {e}")))
}

/// Apply loaded adapter weights to an existing LoRA model.
pub fn apply_loaded_weights(
    model: &mut LoraModel,
    weights: &HashMap<String, Tensor>,
) -> Result<()> {
    for (name, layer) in &mut model.layers {
        if let Some(a) = weights.get(&format!("{name}.lora_a")) {
            layer.lora_a = a.clone();
        }
        if let Some(b) = weights.get(&format!("{name}.lora_b")) {
            layer.lora_b = b.clone();
        }
    }
    Ok(())
}

/// Build a LoRA model for classification: projection + classification head.
///
/// Layer 0: "projection" -- LoRA-adapted identity (hidden_size -> hidden_size)
/// Layer 1: "classifier" -- LoRA-adapted zeros->random (hidden_size -> num_classes)
pub fn build_lora_classification(
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

/// Build a LoRA model for NER: projection + token classifier.
///
/// Layer 0: "projection" -- LoRA-adapted identity (hidden_size -> hidden_size), applied per token
/// Layer 1: "token_classifier" -- LoRA-adapted zeros->random (hidden_size -> num_labels), per token
pub fn build_lora_ner(
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

/// Build a LoRA projection layer for embedding fine-tuning.
/// Creates an identity base weight (hidden_size x hidden_size) wrapped with LoRA.
/// At init (B=0), the projection is identity — model starts producing the same embeddings.
pub fn build_lora_projection(
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
