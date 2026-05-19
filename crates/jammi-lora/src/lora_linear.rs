//! Single LoRA-augmented linear layer: frozen base + trainable A and B matrices.

use candle_core::{DType, Tensor};
use candle_nn::{Init, Linear, Module, VarBuilder};

use crate::error::LoraError;
use crate::init::LoraInitMode;

/// A linear layer wrapped with a LoRA adapter.
///
/// The base weight is treated as frozen. The output is
/// `base(x) + scaling * dropout(x @ A^T @ B^T)`.
pub struct LoraLinear {
    base: Linear,
    /// LoRA A matrix with shape `(rank, in_features)`.
    pub lora_a: Tensor,
    /// LoRA B matrix with shape `(out_features, rank)`.
    pub lora_b: Tensor,
    /// Pre-computed scaling factor (`alpha / rank` or `alpha / sqrt(rank)`).
    scaling: f64,
    /// Optional dropout probability applied to the LoRA path while training.
    dropout: Option<f32>,
    /// Whether the layer is currently in training mode.
    training: bool,
}

impl LoraLinear {
    /// Wrap a frozen `Linear` layer with a LoRA adapter.
    ///
    /// `rank` is the low-rank dimension. `alpha` scales the LoRA contribution.
    /// With `use_rslora`, the scaling becomes `alpha / sqrt(rank)` instead of
    /// `alpha / rank`. `init_mode` selects how the A and B tensors are seeded.
    pub fn new(
        base: Linear,
        rank: usize,
        alpha: f64,
        use_rslora: bool,
        init_mode: LoraInitMode,
        dropout: Option<f32>,
        vb: &VarBuilder,
    ) -> Result<Self, LoraError> {
        if rank == 0 {
            return Err(LoraError::Config("LoRA rank must be > 0".into()));
        }
        let in_features = base.weight().dim(1)?;
        let out_features = base.weight().dim(0)?;

        let (a_init, b_init) = match init_mode {
            LoraInitMode::ZerosB => (
                Init::Kaiming {
                    dist: candle_nn::init::NormalOrUniform::Uniform,
                    fan: candle_nn::init::FanInOut::FanIn,
                    non_linearity: candle_nn::init::NonLinearity::Linear,
                },
                Init::Const(0.0),
            ),
            LoraInitMode::Gaussian => (
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.02,
                },
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.02,
                },
            ),
        };

        let lora_a = vb.get_with_hints((rank, in_features), "lora_a", a_init)?;
        let lora_b = vb.get_with_hints((out_features, rank), "lora_b", b_init)?;

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
            dropout,
            training: true,
        })
    }

    /// Convenience constructor: `ZerosB` init, no dropout, vanilla `alpha/rank`
    /// scaling.
    pub fn new_simple(
        base: Linear,
        rank: usize,
        alpha: f64,
        vb: &VarBuilder,
    ) -> Result<Self, LoraError> {
        Self::new(base, rank, alpha, false, LoraInitMode::ZerosB, None, vb)
    }

    /// Reconstruct a `LoraLinear` from tensors already loaded from disk.
    ///
    /// Scaling is derived as `alpha / rank` where rank is inferred from
    /// `lora_a.dims()[0]`. RSLoRA scaling is intentionally not represented
    /// here because callers reconstructing from disk always know the
    /// effective scaling implied by the saved adapter.
    pub fn from_loaded(base: Linear, lora_a: Tensor, lora_b: Tensor, alpha: f64) -> Self {
        let rank = lora_a.dims()[0];
        let scaling = alpha / rank as f64;
        Self {
            base,
            lora_a,
            lora_b,
            scaling,
            dropout: None,
            training: false,
        }
    }

    /// Toggle training mode. When `false`, dropout in the LoRA path is skipped
    /// so validation loss and inference outputs are deterministic.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Forward: `base(x) + scaling * dropout(x @ A^T @ B^T)`.
    ///
    /// The frozen base path runs in F32 for device-agnostic matmul support;
    /// the result is cast back to the backbone dtype before the LoRA delta is
    /// added so downstream layers stay in their expected precision.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, LoraError> {
        let base_dtype = self.base.weight().dtype();

        let x_f32 = if x.dtype() == DType::F32 {
            x.clone()
        } else {
            x.to_dtype(DType::F32)?
        };
        let w_f32 = if base_dtype == DType::F32 {
            self.base.weight().clone()
        } else {
            self.base.weight().to_dtype(DType::F32)?
        };
        let bias_f32 = self
            .base
            .bias()
            .map(|b| {
                if b.dtype() == DType::F32 {
                    Ok::<_, candle_core::Error>(b.clone())
                } else {
                    b.to_dtype(DType::F32)
                }
            })
            .transpose()?;
        let base_out_f32 = Linear::new(w_f32, bias_f32).forward(&x_f32)?;
        let base_out = if base_dtype == DType::F32 {
            base_out_f32
        } else {
            base_out_f32.to_dtype(base_dtype)?
        };

        let lora_dtype = self.lora_a.dtype();
        let x_lora = if x.dtype() != lora_dtype {
            x.to_dtype(lora_dtype)?
        } else {
            x.clone()
        };

        let lora_in = if self.training {
            if let Some(p) = self.dropout {
                candle_nn::ops::dropout(&x_lora, p)?
            } else {
                x_lora
            }
        } else {
            x_lora
        };

        let a_lin = Linear::new(self.lora_a.clone(), None);
        let after_a = a_lin.forward(&lora_in)?;
        let b_lin = Linear::new(self.lora_b.clone(), None);
        let lora_out = b_lin.forward(&after_a)?;

        let scaled = (&lora_out * self.scaling)?;
        let scaled_cast = if scaled.dtype() != base_out.dtype() {
            scaled.to_dtype(base_out.dtype())?
        } else {
            scaled
        };

        Ok((&base_out + &scaled_cast)?)
    }

    /// References to the two trainable LoRA parameter tensors.
    pub fn trainable_params(&self) -> Vec<&Tensor> {
        vec![&self.lora_a, &self.lora_b]
    }
}
