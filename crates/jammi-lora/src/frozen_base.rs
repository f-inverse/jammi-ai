//! `FrozenBase`: the closed enum naming what a frozen base weight — either
//! a `MaybeLoraLinear::Frozen`'s own layer, or a `LoraLinear`'s base — is
//! stored as. Two arms today: a plain dense `candle_nn::Linear` (every
//! existing safetensors-loaded path, unchanged), or a GGUF-quantized weight
//! ([`QuantizedLinear`], wrapping a candle `QTensor` behind
//! `jammi_kernels::ops::quant_matmul_grad` — the always-differentiable op,
//! never candle's own `QMatMul`/`apply_op1_no_bwd`, see that op's own
//! module doc).
//!
//! # Why a closed enum, not a trait object
//!
//! Matches this crate's existing `MaybeLoraLinear` convention (module doc
//! there): a `dyn Module`-style trait object would erase the compile-time
//! guarantee that every base-weight consumer handles BOTH storage formats —
//! a closed `match` fails to compile the moment a THIRD arm is added
//! without updating every consumer, exactly the property this crate's
//! existing `MaybeLoraLinear`/`FrozenBase` pairing already relies on.
//!
//! # The six base-weight consumers this type threads through
//!
//! (Named explicitly so a reviewer can check each is actually handled, not
//! merely "the enum exists somewhere":)
//! 1. `wrapper.rs`'s `MaybeLoraLinear::forward`'s `Frozen` arm — now
//!    `base.forward(x)`, i.e. [`FrozenBase::forward`] below (Dense's own
//!    dtype-cast-then-forward preserved byte-for-byte inside that method;
//!    Quantized routes through [`QuantizedLinear::forward`]'s F32 rule).
//! 2. `lora_linear.rs`'s `LoraLinear::new`/`new_with_base` `in_features`/
//!    `out_features` derivation — now [`FrozenBase::in_features`]/
//!    [`FrozenBase::out_features`].
//! 3. `lora_linear.rs`'s `frozen_weight_gate` call site in `new_with_base`
//!    — now [`FrozenBase::dweight_needed`]: `Dense` routes through the
//!    existing three-way `frozen_weight_gate` classification unchanged;
//!    `Quantized` is CONSTANT `false` (see that method's own doc for why
//!    this is a structural fact, not a narrowed check).
//! 4. `lora_linear.rs`'s SAME `frozen_weight_gate` call site, in
//!    `from_loaded_with_base` — the other constructor, same
//!    [`FrozenBase::dweight_needed`] call.
//! 5. `lora_linear.rs`'s eval-mode and eager-fallback base-dtype casts —
//!    both now call [`FrozenBase::forward`] (the same method wrapper.rs's
//!    `Frozen` arm uses — one definition of "how does this base weight
//!    turn `x` into an output", not three copies that could drift).
//! 6. `lora_linear.rs`'s `has_bias`/admission-predicate site — stays
//!    Dense-specific by construction: it lives ONLY inside
//!    `LoraLinear::forward`'s `FrozenBase::Dense(..)` match arm (the fused
//!    `LowRankResidualLinear` kernel's own domain requires a dense `Tensor`
//!    weight argument — see its module doc — so admission is never even
//!    evaluated for a `Quantized` base), reading `base_linear.bias()`
//!    directly rather than through a shared accessor.

use std::sync::Arc;

use candle_core::quantized::QTensor;
use candle_core::{DType, Tensor};
use candle_nn::{Linear, Module};
use jammi_kernels::ops::quant_matmul_grad;

use crate::error::LoraError;
use crate::lora_linear::frozen_weight_gate;

/// A frozen GGUF-quantized linear layer: `Arc<QTensor>` (`[out_features,
/// in_features]`, the same convention `candle_nn::Linear`'s own weight
/// uses) plus an optional dense `f32`-or-narrower bias.
///
/// # The uniform F32 activation rule (K2: no unsupported dtype can reach candle)
///
/// [`Self::forward`] ALWAYS casts `x` to `F32` before calling
/// `quant_matmul_grad`, adds the bias (also cast to `F32`), and casts the
/// result back to `x`'s ORIGINAL dtype — one rule, every device, rather
/// than a per-backend dtype table. This is deliberately the WIDEST-common-
/// denominator rule, not the loosest-per-backend one: candle's own
/// `QTensor::cpu_fwd` accepts `F32` OR `F16` on CPU (see
/// `jammi_kernels::ops::quant_matmul_grad`'s own module doc), but its Metal
/// arm `assert_eq!(storage.dtype(), DType::F32)`s internally — a PANIC on
/// anything else, not a typed error (that op's own module doc explains why
/// it does not add a typed guard in front of a candle-internal panic).
/// Casting to `F32` unconditionally, on every device, is therefore the ONE
/// choice that can never reach that panic: `F32` is accepted everywhere
/// candle's own quantized matmul runs at all (CPU: yes; Metal: the only
/// dtype that does not panic; CUDA: supported). A per-device "F16 is
/// slightly cheaper on CPU" optimization would reintroduce exactly the
/// device-dependent dtype table this rule exists to avoid.
#[derive(Debug)]
pub struct QuantizedLinear {
    weight: Arc<QTensor>,
    bias: Option<Tensor>,
}

impl QuantizedLinear {
    /// `weight` must be rank 2, `[out_features, in_features]` (family D —
    /// checked here, in ADDITION to `QTensor::cpu_fwd`'s own `dims2()?`
    /// check at dispatch time, so a malformed weight is refused at
    /// CONSTRUCTION with a clear message rather than only at the first
    /// forward). `bias`, when present, must be `[out_features]`.
    pub fn new(weight: Arc<QTensor>, bias: Option<Tensor>) -> Result<Self, LoraError> {
        let dims = weight.shape().dims();
        if dims.len() != 2 {
            return Err(LoraError::Config(format!(
                "QuantizedLinear: weight must be rank 2 [out_features, in_features], got {dims:?}"
            )));
        }
        let out_features = dims[0];
        if let Some(b) = &bias {
            if b.dims() != [out_features] {
                return Err(LoraError::Config(format!(
                    "QuantizedLinear: bias shape {:?} must be [out_features] = [{out_features}]",
                    b.dims()
                )));
            }
        }
        Ok(Self { weight, bias })
    }

    /// `weight.shape()`'s second (last) dimension — see [`Self::new`]'s
    /// domain doc for why this is always rank 2, `[out, in]`.
    pub fn in_features(&self) -> usize {
        self.weight.shape().dims()[1]
    }

    /// `weight.shape()`'s first dimension.
    pub fn out_features(&self) -> usize {
        self.weight.shape().dims()[0]
    }

    /// The optional dense bias, in its ORIGINAL (construction-time) dtype —
    /// [`Self::forward`] casts it to `F32` internally, once, rather than
    /// mutating this field.
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// The underlying quantized weight — read access only; a
    /// `QuantizedLinear` is immutable after construction (matching
    /// `candle_nn::Linear`'s own immutable-after-construction convention).
    pub fn weight(&self) -> &Arc<QTensor> {
        &self.weight
    }

    /// The uniform F32 activation rule — see this type's own doc.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, LoraError> {
        let orig_dtype = x.dtype();
        let x_f32 = if orig_dtype == DType::F32 {
            x.clone()
        } else {
            x.to_dtype(DType::F32)?
        };
        let y = quant_matmul_grad(&x_f32, self.weight.clone())?;
        let y = match &self.bias {
            Some(b) => {
                let b_f32 = if b.dtype() == DType::F32 {
                    b.clone()
                } else {
                    b.to_dtype(DType::F32)?
                };
                y.broadcast_add(&b_f32)?
            }
            None => y,
        };
        Ok(if orig_dtype == DType::F32 {
            y
        } else {
            y.to_dtype(orig_dtype)?
        })
    }
}

/// See the module doc. `Dense` is every existing safetensors-loaded path,
/// byte-unchanged; `Quantized` is the new GGUF weight-source arm.
#[derive(Debug)]
pub enum FrozenBase {
    /// A plain frozen dense linear layer — every path that exists today.
    Dense(Linear),
    /// A frozen GGUF-quantized linear layer.
    Quantized(QuantizedLinear),
}

impl FrozenBase {
    /// `in_features` — module doc consumer 2.
    pub fn in_features(&self) -> Result<usize, LoraError> {
        Ok(match self {
            Self::Dense(l) => l.weight().dim(1)?,
            Self::Quantized(q) => q.in_features(),
        })
    }

    /// `out_features` — module doc consumer 2.
    pub fn out_features(&self) -> Result<usize, LoraError> {
        Ok(match self {
            Self::Dense(l) => l.weight().dim(0)?,
            Self::Quantized(q) => q.out_features(),
        })
    }

    /// Whether the fused `LowRankResidualLinear` site's `bwd` must compute
    /// and return `Some(dW)` for this base weight — module doc consumers
    /// 3/4. `Dense` routes through `frozen_weight_gate`'s existing
    /// three-way (`true` leaf / trainable `Var` / ambiguous-tracked-refusal)
    /// classification, unchanged.
    ///
    /// `Quantized` is CONSTANT `false` — a structural fact, not a narrowed
    /// check: `candle_core::quantized::QTensor` has no `is_variable`
    /// accessor and is never constructed FROM a `Var` anywhere in this
    /// workspace (a `Var` wraps an ordinary `f32`/`f16`/`bf16` `Tensor`;
    /// `QTensor` is a structurally distinct type holding block-quantized
    /// bytes) — the "trainable Var base" and "ambiguous tracked-non-Var"
    /// cases `frozen_weight_gate` exists to classify for `Dense` have no
    /// reachable `Quantized` analogue at all. A typed error here (mirroring
    /// `frozen_weight_gate`'s own ambiguous-case refusal) would therefore
    /// be DEAD CODE: no production call path in this workspace can ever
    /// construct a `QuantizedLinear` whose weight is anything other than a
    /// true frozen `Arc<QTensor>` leaf. Documented here, not guarded with
    /// an unreachable typed error, per this crate's own K2 discipline
    /// (a typed refusal for a genuinely unreachable case reads as evidence
    /// of a real hazard where none exists). Asserted by
    /// `dweight_needed_is_constant_false_for_a_quantized_base` in
    /// `lora_linear.rs`'s test module.
    pub fn dweight_needed(&self) -> Result<bool, LoraError> {
        match self {
            Self::Dense(l) => frozen_weight_gate(l.weight()),
            Self::Quantized(_) => Ok(false),
        }
    }

    /// `forward` — module doc consumers 1 and 5: `Dense`'s cast-to-weight-
    /// dtype-then-forward is PRESERVED BYTE-FOR-BYTE from every prior
    /// release (the exact composition `wrapper.rs`'s `MaybeLoraLinear::
    /// forward`'s `Frozen` arm used inline before this type existed);
    /// `Quantized` routes through [`QuantizedLinear::forward`]'s uniform F32
    /// rule.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, LoraError> {
        match self {
            Self::Dense(l) => {
                let w_dtype = l.weight().dtype();
                let x_cast = if x.dtype() != w_dtype {
                    x.to_dtype(w_dtype)?
                } else {
                    x.clone()
                };
                Ok(l.forward(&x_cast)?)
            }
            Self::Quantized(q) => q.forward(x),
        }
    }
}

impl From<Linear> for FrozenBase {
    /// The Dense-only wrapping every EXISTING construction path uses
    /// unchanged (module doc: `LoraLinear::new`/`new_simple`/`from_loaded`,
    /// and every `MaybeLoraLinear::Frozen(..)` call site in
    /// `jammi-encoders`, wrap a plain `Linear` through this).
    fn from(l: Linear) -> Self {
        Self::Dense(l)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType;
    use candle_core::Device;

    fn make_quantized(out_f: usize, in_f: usize) -> Arc<QTensor> {
        let device = Device::Cpu;
        let w_v: Vec<f32> = (0..out_f * in_f)
            .map(|i| ((i as f64) * 0.041 + 0.3).sin() as f32)
            .collect();
        let w = Tensor::from_vec(w_v, (out_f, in_f), &device).unwrap();
        Arc::new(QTensor::quantize(&w, GgmlDType::Q8_0).unwrap())
    }

    #[test]
    fn quantized_linear_rejects_non_rank2_weight() {
        let device = Device::Cpu;
        let w = Tensor::zeros(32, DType::F32, &device).unwrap();
        let q = QTensor::quantize(&w, GgmlDType::Q8_0).unwrap();
        let err = QuantizedLinear::new(Arc::new(q), None).unwrap_err();
        assert!(matches!(err, LoraError::Config(_)));
    }

    #[test]
    fn quantized_linear_rejects_mismatched_bias_shape() {
        let device = Device::Cpu;
        let weight = make_quantized(4, 32);
        let bad_bias = Tensor::zeros(3, DType::F32, &device).unwrap(); // out_features is 4
        let err = QuantizedLinear::new(weight, Some(bad_bias)).unwrap_err();
        assert!(matches!(err, LoraError::Config(_)));
    }

    #[test]
    fn frozen_base_accessors_agree_between_dense_and_quantized() {
        let device = Device::Cpu;
        let (out_f, in_f) = (4usize, 32usize);
        let weight = make_quantized(out_f, in_f);
        let ql = QuantizedLinear::new(weight, None).unwrap();
        let quantized = FrozenBase::Quantized(ql);
        assert_eq!(quantized.in_features().unwrap(), in_f);
        assert_eq!(quantized.out_features().unwrap(), out_f);

        let dense_w = Tensor::zeros((out_f, in_f), DType::F32, &device).unwrap();
        let dense = FrozenBase::Dense(Linear::new(dense_w, None));
        assert_eq!(dense.in_features().unwrap(), in_f);
        assert_eq!(dense.out_features().unwrap(), out_f);
    }

    /// `dweight_needed` is CONSTANT `false` for a `Quantized` base —
    /// mechanism pin for [`FrozenBase::dweight_needed`]'s own doc.
    #[test]
    fn dweight_needed_is_constant_false_for_a_quantized_base() {
        let weight = make_quantized(4, 32);
        let ql = QuantizedLinear::new(weight, None).unwrap();
        let quantized = FrozenBase::Quantized(ql);
        assert!(!quantized.dweight_needed().unwrap());
    }

    /// The uniform F32 activation rule (module doc): a non-`F32` input is
    /// cast to `F32`, matmul'd, and cast back to the ORIGINAL dtype.
    #[test]
    fn quantized_linear_forward_round_trips_a_non_f32_input_dtype() {
        let device = Device::Cpu;
        let (out_f, in_f, rows) = (4usize, 32usize, 2usize);
        let weight = make_quantized(out_f, in_f);
        let ql = QuantizedLinear::new(weight, None).unwrap();
        let x_v: Vec<f32> = (0..rows * in_f)
            .map(|i| ((i as f64) * 0.017 + 1.0).cos() as f32)
            .collect();
        let x_f16 = Tensor::from_vec(x_v, (rows, in_f), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let y = ql.forward(&x_f16).unwrap();
        assert_eq!(
            y.dtype(),
            DType::F16,
            "output must round-trip to x's own dtype"
        );
        assert_eq!(y.dims(), &[rows, out_f]);
    }

    /// A bias is added in `F32`, then rounded back once — same "one round
    /// point" doctrine `jammi_kernels::ops::quant_matmul_grad`'s own module
    /// doc states for the matmul itself.
    #[test]
    fn quantized_linear_forward_adds_the_bias() {
        let device = Device::Cpu;
        let (out_f, in_f, rows) = (3usize, 32usize, 1usize);
        let weight = make_quantized(out_f, in_f);
        let bias = Tensor::from_vec(vec![10.0f32, 20.0, 30.0], out_f, &device).unwrap();
        let ql = QuantizedLinear::new(weight.clone(), Some(bias.clone())).unwrap();
        let ql_no_bias = QuantizedLinear::new(weight, None).unwrap();
        let x_v: Vec<f32> = (0..rows * in_f)
            .map(|i| ((i as f64) * 0.023 + 0.5).sin() as f32)
            .collect();
        let x = Tensor::from_vec(x_v, (rows, in_f), &device).unwrap();

        let y_with_bias = ql
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let y_without_bias = ql_no_bias
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let bias_v = bias.to_vec1::<f32>().unwrap();
        for i in 0..out_f {
            let diff = (y_with_bias[i] - (y_without_bias[i] + bias_v[i])).abs();
            assert!(diff < 1e-5, "index {i}: diff {diff}");
        }
    }
}
