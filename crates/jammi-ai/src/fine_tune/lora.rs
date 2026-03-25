//! LoRA (Low-Rank Adaptation) layer implementation.
//!
//! Replaces `W·x` with `(W + B·A)·x` where A is `rank×in_features` and B is
//! `out_features×rank`. At initialization B is zeros, so the LoRA layer starts
//! as an identity transform over the base weight.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Init, Linear, Module, VarBuilder};
use jammi_engine::error::{JammiError, Result};

/// A single LoRA-augmented linear layer: frozen base + trainable A, B matrices.
pub struct LoraLinear {
    base: Linear,
    /// A matrix: (rank × in_features), initialized with kaiming uniform.
    pub lora_a: Tensor,
    /// B matrix: (out_features × rank), initialized to zeros.
    pub lora_b: Tensor,
    alpha: f64,
    rank: usize,
}

impl LoraLinear {
    /// Wrap a frozen `Linear` layer with LoRA adapters.
    ///
    /// `rank` controls the number of low-rank dimensions. `alpha` scales the
    /// LoRA contribution: `output = base(x) + (alpha/rank) * x @ A^T @ B^T`.
    pub fn new(base: Linear, rank: usize, alpha: f64, vb: &VarBuilder) -> Result<Self> {
        let in_features = base
            .weight()
            .dim(1)
            .map_err(|e| JammiError::FineTune(format!("LoRA dim: {e}")))?;
        let out_features = base
            .weight()
            .dim(0)
            .map_err(|e| JammiError::FineTune(format!("LoRA dim: {e}")))?;

        let lora_a = vb
            .get_with_hints(
                (rank, in_features),
                "lora_a",
                Init::Kaiming {
                    dist: candle_nn::init::NormalOrUniform::Uniform,
                    fan: candle_nn::init::FanInOut::FanIn,
                    non_linearity: candle_nn::init::NonLinearity::Linear,
                },
            )
            .map_err(|e| JammiError::FineTune(format!("LoRA A init: {e}")))?;

        let lora_b = vb
            .get_with_hints((out_features, rank), "lora_b", Init::Const(0.0))
            .map_err(|e| JammiError::FineTune(format!("LoRA B init: {e}")))?;

        Ok(Self {
            base,
            lora_a,
            lora_b,
            alpha,
            rank,
        })
    }

    /// Forward pass: `base(x) + scaling * (x @ A^T @ B^T)`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let base_out = self
            .base
            .forward(x)
            .map_err(|e| JammiError::FineTune(format!("LoRA base forward: {e}")))?;
        let scaling = self.alpha / self.rank as f64;

        let lora_out = x
            .matmul(
                &self
                    .lora_a
                    .t()
                    .map_err(|e| JammiError::FineTune(format!("LoRA A^T: {e}")))?,
            )
            .map_err(|e| JammiError::FineTune(format!("LoRA x@A^T: {e}")))?
            .matmul(
                &self
                    .lora_b
                    .t()
                    .map_err(|e| JammiError::FineTune(format!("LoRA B^T: {e}")))?,
            )
            .map_err(|e| JammiError::FineTune(format!("LoRA xA^T@B^T: {e}")))?;

        let scaled = (&lora_out * scaling)
            .map_err(|e| JammiError::FineTune(format!("LoRA scaling: {e}")))?;

        (&base_out + &scaled).map_err(|e| JammiError::FineTune(format!("LoRA add: {e}")))
    }

    /// Reconstruct a LoraLinear from pre-loaded weights (for inference with saved adapters).
    pub fn from_loaded(base: Linear, lora_a: Tensor, lora_b: Tensor, alpha: f64) -> Self {
        let rank = lora_a.dims()[0];
        Self {
            base,
            lora_a,
            lora_b,
            alpha,
            rank,
        }
    }

    /// Return references to the two trainable parameter tensors (A and B).
    pub fn trainable_params(&self) -> Vec<&Tensor> {
        vec![&self.lora_a, &self.lora_b]
    }
}

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
    let lora = LoraLinear::new(
        base,
        config.lora_rank,
        config.lora_alpha,
        &vb.pp("projection"),
    )?;
    Ok(LoraModel {
        layers: vec![("projection".into(), lora)],
    })
}
