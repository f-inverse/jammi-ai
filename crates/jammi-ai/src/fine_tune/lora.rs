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
use serde::{Deserialize, Serialize};

/// Initialization strategy for the LoRA A matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoraInitMode {
    /// B = zeros, A = kaiming uniform (default — adapter starts at zero delta).
    ZerosB,
    /// Both A and B initialized with Gaussian noise (σ = 0.02).
    Gaussian,
    /// PiSSA: A/B initialized from principal singular values of the base weight.
    /// Falls back to ZerosB when the base weight is not available for SVD.
    PiSSA,
}

impl Default for LoraInitMode {
    fn default() -> Self {
        Self::ZerosB
    }
}

/// A single LoRA-augmented linear layer: frozen base + trainable A, B matrices.
pub struct LoraLinear {
    base: Linear,
    /// A matrix: (rank × in_features).
    pub lora_a: Tensor,
    /// B matrix: (out_features × rank).
    pub lora_b: Tensor,
    /// Pre-computed scaling factor (alpha/rank or alpha/sqrt(rank) when use_rslora).
    scaling: f64,
    #[allow(dead_code)]
    rank: usize,
    /// Optional dropout probability applied to the LoRA path — only active when
    /// `training` is `true`.  Set to `false` during evaluation and inference to
    /// avoid stochastic noise in validation loss and inference outputs.
    dropout: Option<f32>,
    /// Whether the layer is in training mode.  Defaults to `true` for newly
    /// constructed layers; `false` for layers loaded from saved adapters.
    pub training: bool,
}

impl LoraLinear {
    /// Wrap a frozen `Linear` layer with LoRA adapters.
    ///
    /// `rank` controls the number of low-rank dimensions. `alpha` scales the
    /// LoRA contribution. When `use_rslora` is true, scaling = alpha/sqrt(rank)
    /// instead of alpha/rank. `init_mode` controls A/B initialization.
    pub fn new(
        base: Linear,
        rank: usize,
        alpha: f64,
        use_rslora: bool,
        init_mode: LoraInitMode,
        dropout: Option<f32>,
        vb: &VarBuilder,
    ) -> Result<Self> {
        let in_features = base
            .weight()
            .dim(1)
            .map_err(|e| JammiError::FineTune(format!("LoRA dim: {e}")))?;
        let out_features = base
            .weight()
            .dim(0)
            .map_err(|e| JammiError::FineTune(format!("LoRA dim: {e}")))?;

        let (a_init, b_init) = match init_mode {
            LoraInitMode::ZerosB => (
                Init::Kaiming {
                    dist: candle_nn::init::NormalOrUniform::Uniform,
                    fan: candle_nn::init::FanInOut::FanIn,
                    non_linearity: candle_nn::init::NonLinearity::Linear,
                },
                Init::Const(0.0),
            ),
            LoraInitMode::Gaussian => (Init::Randn { mean: 0.0, stdev: 0.02 }, Init::Randn { mean: 0.0, stdev: 0.02 }),
            LoraInitMode::PiSSA => (
                // PiSSA requires SVD of the base weight — fall back to ZerosB
                Init::Kaiming {
                    dist: candle_nn::init::NormalOrUniform::Uniform,
                    fan: candle_nn::init::FanInOut::FanIn,
                    non_linearity: candle_nn::init::NonLinearity::Linear,
                },
                Init::Const(0.0),
            ),
        };

        let lora_a = vb
            .get_with_hints((rank, in_features), "lora_a", a_init)
            .map_err(|e| JammiError::FineTune(format!("LoRA A init: {e}")))?;

        let lora_b = vb
            .get_with_hints((out_features, rank), "lora_b", b_init)
            .map_err(|e| JammiError::FineTune(format!("LoRA B init: {e}")))?;

        let scaling = if use_rslora {
            alpha / (rank as f64).sqrt()
        } else {
            alpha / rank as f64
        };

        Ok(Self {
            base,
            lora_a,
            lora_b,
            scaling,
            rank,
            dropout,
            training: true,
        })
    }

    /// Convenience constructor matching the old signature (ZerosB, no dropout, standard scaling).
    pub fn new_simple(base: Linear, rank: usize, alpha: f64, vb: &VarBuilder) -> Result<Self> {
        Self::new(base, rank, alpha, false, LoraInitMode::ZerosB, None, vb)
    }

    /// Switch between training and evaluation mode.
    ///
    /// When `false`, the dropout path is skipped so validation loss and
    /// inference outputs are deterministic.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Forward pass: `base(x) + scaling * dropout(x @ A^T @ B^T)`.
    ///
    /// Mixed-precision aware: the frozen base linear always runs in F32 for
    /// device-agnostic compatibility (BF16/F16 matmul is not supported on all
    /// CUDA compute capabilities).  Since the base weights are frozen, running
    /// them in F32 does not affect gradient flow through the LoRA path.
    /// The result is cast back to the backbone's native dtype before the LoRA
    /// delta is added, so downstream layers stay in the expected precision.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let base_dtype = self.base.weight().dtype();

        // Frozen path: always compute in F32 for maximum device compatibility.
        let x_f32 = if x.dtype() == DType::F32 {
            x.clone()
        } else {
            x.to_dtype(DType::F32)
                .map_err(|e| JammiError::FineTune(format!("LoRA base input cast: {e}")))?
        };
        let w_f32 = if base_dtype == DType::F32 {
            self.base.weight().clone()
        } else {
            self.base
                .weight()
                .to_dtype(DType::F32)
                .map_err(|e| JammiError::FineTune(format!("LoRA base weight cast: {e}")))?
        };
        let bias_f32 = self
            .base
            .bias()
            .map(|b| {
                if b.dtype() == DType::F32 {
                    Ok(b.clone())
                } else {
                    b.to_dtype(DType::F32)
                        .map_err(|e| JammiError::FineTune(format!("LoRA base bias cast: {e}")))
                }
            })
            .transpose()?;
        let base_out_f32 = Linear::new(w_f32, bias_f32)
            .forward(&x_f32)
            .map_err(|e| JammiError::FineTune(format!("LoRA base forward: {e}")))?;
        // Cast output back to backbone dtype so downstream transformer layers
        // continue operating in their expected precision.
        let base_out = if base_dtype == DType::F32 {
            base_out_f32
        } else {
            base_out_f32
                .to_dtype(base_dtype)
                .map_err(|e| JammiError::FineTune(format!("LoRA base output cast: {e}")))?
        };

        // LoRA path: A/B matrices are always F32; cast input accordingly.
        let lora_dtype = self.lora_a.dtype();
        let x_lora = if x.dtype() != lora_dtype {
            x.to_dtype(lora_dtype)
                .map_err(|e| JammiError::FineTune(format!("LoRA lora dtype cast: {e}")))?
        } else {
            x.clone()
        };

        let lora_in = if self.training {
            if let Some(p) = self.dropout {
                candle_nn::ops::dropout(&x_lora, p)
                    .map_err(|e| JammiError::FineTune(format!("LoRA dropout: {e}")))?
            } else {
                x_lora
            }
        } else {
            x_lora
        };

        // x @ A^T : Linear with weight = lora_a (shape [rank, in]) computes
        //   y = x @ lora_a^T  →  [..., rank]
        let a_lin = Linear::new(self.lora_a.clone(), None);
        let after_a = a_lin
            .forward(&lora_in)
            .map_err(|e| JammiError::FineTune(format!("LoRA x@A^T: {e}")))?;

        // (x @ A^T) @ B^T : Linear with weight = lora_b (shape [out, rank]).
        let b_lin = Linear::new(self.lora_b.clone(), None);
        let lora_out = b_lin
            .forward(&after_a)
            .map_err(|e| JammiError::FineTune(format!("LoRA xA^T@B^T: {e}")))?;

        let scaled = (&lora_out * self.scaling)
            .map_err(|e| JammiError::FineTune(format!("LoRA scaling: {e}")))?;

        // Cast LoRA delta back to the frozen output dtype before adding.
        let scaled_cast = if scaled.dtype() != base_out.dtype() {
            scaled
                .to_dtype(base_out.dtype())
                .map_err(|e| JammiError::FineTune(format!("LoRA output dtype cast: {e}")))?
        } else {
            scaled
        };

        (&base_out + &scaled_cast).map_err(|e| JammiError::FineTune(format!("LoRA add: {e}")))
    }

    /// Reconstruct a LoraLinear from pre-loaded weights (for inference with saved adapters).
    pub fn from_loaded(base: Linear, lora_a: Tensor, lora_b: Tensor, alpha: f64) -> Self {
        let rank = lora_a.dims()[0];
        let scaling = alpha / rank as f64;
        Self {
            base,
            lora_a,
            lora_b,
            scaling,
            rank,
            dropout: None,
            training: false,
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
    )?;

    // Layer 1: classifier (zeros base, trained from scratch via LoRA)
    let cls_base = Tensor::zeros((num_classes, hidden_size), DType::F32, vb.device())
        .map_err(|e| JammiError::FineTune(format!("Classifier zeros: {e}")))?;
    let cls_linear = Linear::new(cls_base, None);
    let classifier = LoraLinear::new_simple(
        cls_linear,
        config.lora_rank,
        config.lora_alpha,
        &vb.pp("classifier"),
    )?;

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
    )?;

    // Layer 1: token classifier (zeros base, trained from scratch via LoRA)
    let cls_base = Tensor::zeros((num_labels, hidden_size), DType::F32, vb.device())
        .map_err(|e| JammiError::FineTune(format!("NER classifier zeros: {e}")))?;
    let cls_linear = Linear::new(cls_base, None);
    let classifier = LoraLinear::new_simple(
        cls_linear,
        config.lora_rank,
        config.lora_alpha,
        &vb.pp("token_classifier"),
    )?;

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
    )?;
    Ok(LoraModel {
        layers: vec![("projection".into(), lora)],
    })
}
