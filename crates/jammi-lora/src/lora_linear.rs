//! Single LoRA-augmented linear layer: frozen base + trainable A and B matrices.

use std::sync::Mutex;

use candle_core::{DType, Tensor};
use candle_nn::{Init, Linear, Module, VarBuilder, VarMap};

use crate::error::LoraError;
use crate::init::LoraInitMode;
use crate::seeded::{
    gaussian_fill, kaiming_uniform_fill, seed_for_param, DropoutStream, SplitMix64,
};

/// Overwrite the storage of the `Var` already registered at `name` in `varmap`
/// with `value`, reaching it through the shared `&VarMap` (no `&mut` needed
/// because `Var::set` is `&self`). Fails if no such `Var` exists — the caller
/// must have registered it (via `get_with_hints`) first.
fn set_var(varmap: &VarMap, name: &str, value: &Tensor) -> Result<(), LoraError> {
    let data = varmap.data().lock().map_err(|_| {
        LoraError::Config(format!("seeded init: VarMap mutex poisoned setting {name}"))
    })?;
    let var = data.get(name).ok_or_else(|| {
        LoraError::Config(format!("seeded init: {name} not registered in VarMap"))
    })?;
    var.set(value)
        .map_err(|e| LoraError::Config(format!("seeded init set {name}: {e}")))
}

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
    /// Run-owned, seeded dropout stream. `Some` exactly when `dropout > 0`.
    /// Interior-mutable because the `Module`-style `forward(&self, …)` advances
    /// the mask stream; a `Mutex` (not `RefCell`) keeps `LoraLinear: Sync`, which
    /// the model holds across threads. The trainer drives forwards
    /// single-threaded and in a deterministic order, so the lock is uncontended
    /// and the k-th training forward through this layer always consumes the k-th
    /// mask.
    dropout_stream: Option<Mutex<DropoutStream>>,
    /// Whether the layer is currently in training mode.
    training: bool,
}

impl LoraLinear {
    /// Wrap a frozen `Linear` layer with a LoRA adapter.
    ///
    /// `rank` is the low-rank dimension. `alpha` scales the LoRA contribution.
    /// With `use_rslora`, the scaling becomes `alpha / sqrt(rank)` instead of
    /// `alpha / rank`. `init_mode` selects how the A and B tensors are seeded.
    ///
    /// `seed` makes the A/B init and the dropout mask a pure function of the run
    /// seed and the parameter's fully-qualified name (`{vb.prefix()}.lora_a` /
    /// `…lora_b`): the draws come from a jammi-owned `SplitMix64`, **not**
    /// candle's unseedable global RNG. Each parameter's stream is keyed by name
    /// (not by `VarMap`/construction order), so the same seed yields
    /// byte-identical adapters run-to-run and across processes. The seeded
    /// tensors are still registered as trainable `Var`s in `varmap` — candle's
    /// own `Init` first allocates the `Var` (with a deterministic placeholder so
    /// no RNG is touched), then the seeded values are written into it in place.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base: Linear,
        rank: usize,
        alpha: f64,
        use_rslora: bool,
        init_mode: LoraInitMode,
        dropout: Option<f32>,
        seed: u64,
        varmap: &VarMap,
        vb: &VarBuilder,
    ) -> Result<Self, LoraError> {
        if rank == 0 {
            return Err(LoraError::Config("LoRA rank must be > 0".into()));
        }
        let in_features = base.weight().dim(1)?;
        let out_features = base.weight().dim(0)?;
        let device = vb.device().clone();

        // Fetch (or, in the training path, allocate + register) the A/B tensors.
        // `Init::Const(0.0)` is deterministic so candle's RNG is never invoked.
        // In the TRAINING path the `VarBuilder` is VarMap-backed: this registers
        // fresh trainable `Var`s, which we then overwrite with the seeded draw.
        // In the INFERENCE path the `VarBuilder` is mmaped-safetensors-backed:
        // `get_with_hints` returns the SAVED adapter tensors and nothing is in
        // the (dummy) VarMap — so the seeded fill is correctly skipped and the
        // loaded weights stand.
        let lora_a = vb.get_with_hints((rank, in_features), "lora_a", Init::Const(0.0))?;
        let lora_b = vb.get_with_hints((out_features, rank), "lora_b", Init::Const(0.0))?;

        // Fully-qualified parameter names — the stable per-parameter draw key.
        // Built exactly as candle's `VarBuilder::path` joins them (no leading
        // dot when the prefix is empty) so they match the registered `Var` keys.
        let prefix = vb.prefix();
        let qualify = |leaf: &str| {
            if prefix.is_empty() {
                leaf.to_string()
            } else {
                format!("{prefix}.{leaf}")
            }
        };
        let a_name = qualify("lora_a");
        let b_name = qualify("lora_b");

        // Only seed-init the parameters that were just registered as trainable
        // `Var`s in `varmap`. If they are absent, this is the load-from-adapter
        // inference path and the values `get_with_hints` returned are the saved
        // weights, which must not be perturbed.
        let registered = {
            let data = varmap
                .data()
                .lock()
                .map_err(|_| LoraError::Config("seeded init: VarMap mutex poisoned".into()))?;
            data.contains_key(&a_name) && data.contains_key(&b_name)
        };

        if registered {
            let (a_values, b_values): (Vec<f32>, Vec<f32>) = match init_mode {
                LoraInitMode::ZerosB => {
                    // A: Kaiming-uniform over fan_in = in_features. B: zeros.
                    let mut rng = SplitMix64::new(seed_for_param(seed, &a_name));
                    let a = kaiming_uniform_fill(&mut rng, rank * in_features, in_features);
                    let b = vec![0.0_f32; out_features * rank];
                    (a, b)
                }
                LoraInitMode::Gaussian => {
                    // Both A and B ~ Normal(0, 0.02), independent name-keyed streams.
                    let mut rng_a = SplitMix64::new(seed_for_param(seed, &a_name));
                    let mut rng_b = SplitMix64::new(seed_for_param(seed, &b_name));
                    let a = gaussian_fill(&mut rng_a, rank * in_features, 0.02);
                    let b = gaussian_fill(&mut rng_b, out_features * rank, 0.02);
                    (a, b)
                }
            };

            let a_tensor = Tensor::from_vec(a_values, (rank, in_features), &device)?;
            let b_tensor = Tensor::from_vec(b_values, (out_features, rank), &device)?;
            // Overwrite the just-registered `Var`s' storage in place. `Var::set`
            // takes `&self`, so we reach it through the shared `VarMap` (which the
            // `VarBuilder` registered into) without needing `&mut` — the `Var`
            // identity the optimiser collects via `all_vars()` is preserved.
            set_var(varmap, &a_name, &a_tensor)?;
            set_var(varmap, &b_name, &b_tensor)?;
        }

        let scaling = if use_rslora {
            alpha / (rank as f64).sqrt()
        } else {
            alpha / rank as f64
        };

        let dropout_stream = dropout
            .filter(|p| *p > 0.0)
            .map(|_| Mutex::new(DropoutStream::new(seed, &vb.prefix())));

        Ok(Self {
            base,
            lora_a,
            lora_b,
            scaling,
            dropout,
            dropout_stream,
            training: true,
        })
    }

    /// Convenience constructor: `ZerosB` init, no dropout, vanilla `alpha/rank`
    /// scaling, seeded from `seed`.
    pub fn new_simple(
        base: Linear,
        rank: usize,
        alpha: f64,
        seed: u64,
        varmap: &VarMap,
        vb: &VarBuilder,
    ) -> Result<Self, LoraError> {
        Self::new(
            base,
            rank,
            alpha,
            false,
            LoraInitMode::ZerosB,
            None,
            seed,
            varmap,
            vb,
        )
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
            dropout_stream: None,
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
            match (self.dropout, &self.dropout_stream) {
                (Some(p), Some(stream)) if p > 0.0 => {
                    // Seeded inverted-dropout: draw a Bernoulli mask from the
                    // run-owned stream (NOT candle's unseedable `ops::dropout`)
                    // and apply it. The stream advances one block of draws per
                    // training forward, so the mask is a pure function of the
                    // seed and the forward's position in the deterministic
                    // training order.
                    let mask_vals = stream
                        .lock()
                        .map_err(|_| LoraError::Config("dropout stream mutex poisoned".into()))?
                        .draw_mask(x_lora.elem_count(), p);
                    let mask = Tensor::from_vec(mask_vals, x_lora.shape(), x_lora.device())?
                        .to_dtype(x_lora.dtype())?;
                    (&x_lora * &mask)?
                }
                _ => x_lora,
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
