//! LayerNorm whose backward is well-defined.
//!
//! In eval mode, delegates to candle's fused `crate::ops::layer_norm` for parity
//! with `candle_nn::LayerNorm`'s fast path. In training mode, composes the same
//! math out of primitive ops whose `bwd` is implemented, so gradient propagates
//! through to upstream trainable parameters. The two paths are algebraically
//! equivalent; FP rounding differs by ~1 ULP per accumulation.
//!
//! The fast path is only entered when `bias.is_some()` and the input is
//! contiguous, matching `candle_nn::LayerNorm`'s own entry conditions.
//!
//! ## The bias-free training path: `jammi_kernels::ops::LayerNormFused`
//!
//! A THIRD path exists, gated on `bias.is_none() && training`: every
//! ModernBERT LayerNorm (`ModernBertConfig` cannot even express a biased
//! LayerNorm — no `norm_bias` field exists) dispatches to the fused
//! CUDA/CPU kernel instead of the `~12`-op eager composition below, when
//! the fused kernel's own domain holds (`x`'s device is CPU or CUDA —
//! `LayerNormFused` has no `metal_fwd`, and candle's default `metal_fwd`
//! ERRORS rather than falling back, so a Metal tensor is refused by this
//! predicate rather than reaching `apply2` and hard-erroring; dtype
//! F32/BF16 matching between `x` and `weight`; both contiguous; `hidden`
//! within the kernel's ceiling). Outside that domain — or on the
//! `parity-test`/BERT/DistilBERT paths, which are BIASED and so never
//! reach this arm at all — `slow()` runs exactly as before. This is a K2
//! "validate, don't silently degrade" admission check: the fused/eager
//! decision is recorded ([`LN_DISPATCH_COUNTERS`]) and a failed predicate
//! either falls back with a log-once WARN or, in `Strict` mode
//! ([`admission_mode`]), errors instead of silently falling back.
//!
//! Eval (`training == false`) NEVER reaches the fused arm regardless of
//! `bias` — the match below only adds a NEW arm for `(None, true)`; every
//! other `(bias, training)` combination is byte-for-byte the same code
//! path this file had before the fused kernel existed. Eval/serving
//! numerics are therefore bit-identical before/after this change (see
//! this module's own
//! `tests::eval_mode_forward_is_bit_identical_regardless_of_fused_eligibility`).
//!
//! `dgamma_needed` is `self.weight.is_variable()`, evaluated fresh on
//! every fused-path call — NOT a hardcoded `false`. `is_variable()` is
//! unsound as a general "does this need a gradient" predicate (see
//! `jammi_kernels::ops::layer_norm`'s module doc: it is two-state over a
//! three-state lattice, and cannot tell a true external constant apart
//! from an INTERMEDIATE on a path to a `Var`) — but that hazard is
//! ONE-DIRECTIONAL and does not apply to `weight` here. `weight` is a
//! `LayerNorm`'s own leaf module parameter — loaded straight from a
//! `VarBuilder` with no upstream op — never an intermediate produced by
//! composing other tensors, so the only two real states are "is a `Var`"
//! (today: never, in this crate — only LoRA A/B are trainable; a future
//! trainable-gamma mode would make this `true`) and "is a true frozen
//! leaf" (today's actual state, `VarBuilder::from_mmaped_safetensors` —
//! see `modernbert.rs`). `is_variable() == true` is therefore a SUFFICIENT
//! (not merely convenient) condition here: if `weight` somehow were an
//! intermediate despite never being constructed that way, `is_variable()
//! == false` would make this `false`, and if that later turned out to be
//! the wrong call, candle's own backward walk panics loudly (`grad not
//! populated`, `backprop.rs:175`) rather than silently training a
//! grad-less parameter — a safe failure mode, not a silent-wrong one.

use std::sync::LazyLock;

use candle_core::{DType, Tensor, D};
use candle_nn::{Init, VarBuilder};
use jammi_kernels::admission::{
    admission_mode, admit, counters_for, device_is_supported, DispatchCounters, DispatchOutcome,
};
use jammi_kernels::ops::{apply2, LayerNormFused, MAX_HIDDEN};

use crate::error::EncoderError;

/// Per-op fused/eager dispatch counts for the bias-free training
/// LayerNorm, read from `jammi_kernels::admission`'s op-keyed registry
/// (`counters_for`) rather than a directly-owned `static DispatchCounters`
/// — this crate's C2-C5 four ops (this one plus RoPE/softmax/GeGLU in
/// `crate::modernbert`) were the registry's pre-existing hand-declared
/// statics; migrating them here is what makes the registry the SOLE
/// source of dispatch counters crate-wide (`jammi-lora`'s LoRA-site ops
/// already used it from the start — see `jammi_kernels::admission`'s
/// module doc). A `LazyLock`, not a plain `fn`, so `LN_DISPATCH_COUNTERS`
/// stays a `static` item: `crate::ln_dispatch_snapshot` (`lib.rs`, shared
/// class, not touched by this migration) calls
/// `layer_norm::LN_DISPATCH_COUNTERS.snapshot()` — a bare path followed by
/// a method call — which keeps compiling unchanged against a `LazyLock`
/// (auto-deref resolves `.snapshot()` through it to
/// `DispatchCounters::snapshot`) but would NOT compile against a renamed
/// function (`LN_DISPATCH_COUNTERS().snapshot()` is a different call
/// shape). This static itself is `pub` but lives inside a crate-private
/// module (`mod layer_norm;` in `lib.rs`) — unnameable from outside this
/// crate; `crate::ln_dispatch_snapshot` is the actual public read API a
/// durable job record or a bench report uses.
pub static LN_DISPATCH_COUNTERS: LazyLock<&'static DispatchCounters> =
    LazyLock::new(|| counters_for("layer_norm_fused"));

/// The fused kernel's domain, checked at the call site (family D / K2):
/// `x` and `weight` live on a device [`device_is_supported`] accepts,
/// share a dtype the kernel implements (F32 or BF16), both are
/// contiguous (`LayerNormFused` refuses a strided view rather than risk
/// misreading the row grouping — see its module doc), `weight` is rank-1
/// matching `x`'s last dimension, and that dimension is within the
/// kernel's `MAX_HIDDEN` ceiling (a conservative validated bound, not a
/// hardware limit — see `MAX_HIDDEN`'s own doc). Returns the aggregate
/// predicate and the name of whichever check is the reason (the first
/// one evaluated, or a fixed "domain_ok" name when everything holds) —
/// the failing name is what a Fallback-mode log line or a Strict-mode
/// error names.
fn fused_admission_predicate(x: &Tensor, weight: &Tensor) -> (bool, &'static str) {
    if !device_is_supported(x.device()) {
        return (false, "device_is_cpu_or_cuda");
    }
    if x.dtype() != weight.dtype() || !matches!(x.dtype(), DType::F32 | DType::BF16) {
        return (false, "dtype_f32_or_bf16_matching_between_x_and_weight");
    }
    if !x.is_contiguous() {
        return (false, "x_contiguous");
    }
    if !weight.is_contiguous() {
        return (false, "weight_contiguous");
    }
    let Some(&hidden) = x.dims().last() else {
        return (false, "x_rank_at_least_1");
    };
    if weight.dims() != [hidden] {
        return (false, "weight_rank1_matches_x_last_dim");
    }
    if hidden == 0 || hidden > MAX_HIDDEN {
        return (false, "hidden_within_kernel_max_hidden");
    }
    (true, "domain_ok")
}

/// Layer normalisation over the last dimension with optional affine bias.
pub struct LayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    eps: f64,
    training: bool,
}

impl LayerNorm {
    /// Load a LayerNorm under `vb`'s current prefix. `weight` and (when
    /// `with_bias` is true) `bias` are read from the safetensors layout
    /// expected at that prefix; if absent, they are initialised to ones and
    /// zeros respectively.
    pub fn new(
        hidden_size: usize,
        eps: f64,
        with_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self, EncoderError> {
        let weight = vb.get_with_hints(hidden_size, "weight", Init::Const(1.0))?;
        let bias = with_bias
            .then(|| vb.get_with_hints(hidden_size, "bias", Init::Const(0.0)))
            .transpose()?;
        Ok(Self {
            weight,
            bias,
            eps,
            training: false,
        })
    }

    /// Switch between the fused eval forward and the gradient-carrying training
    /// forward.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// `[..., hidden] -> [..., hidden]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        match (&self.bias, self.training) {
            (Some(bias), false) if x.is_contiguous() => Ok(candle_nn::ops::layer_norm(
                x,
                &self.weight,
                bias,
                self.eps as f32,
            )?),
            (None, true) => self.forward_fused_or_fallback(x),
            _ => self.slow(x),
        }
    }

    /// The bias-free, training-mode arm: dispatches to
    /// [`LayerNormFused`] when its domain holds, else falls back to
    /// [`Self::slow`] (recording which happened either way). See this
    /// module's doc for the full design and why `dgamma_needed` is
    /// `self.weight.is_variable()`, not a hardcoded `false`.
    fn forward_fused_or_fallback(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        let (holds, predicate) = fused_admission_predicate(x, &self.weight);
        let outcome = admit(
            admission_mode(),
            "layer_norm_fused",
            predicate,
            holds,
            *LN_DISPATCH_COUNTERS,
        )?;
        match outcome {
            DispatchOutcome::Fused => Ok(apply2(
                x,
                &self.weight,
                LayerNormFused::new(self.eps, self.weight.is_variable()),
            )?),
            DispatchOutcome::Eager => self.slow(x),
        }
    }

    /// `y = xhat * gamma [+ beta]`, matching torch's `layer_norm_cuda`,
    /// PINNED to torch 2.13.0
    /// (`aten/src/ATen/native/cuda/layer_norm_kernel.cu`'s
    /// `vectorized_layer_norm_kernel_impl`: "Computation is performed in
    /// T_ACC, X is cast to T_ACC and result is implicitly cast to T" —
    /// `out = gamma * (rstd * (x - mean)) + beta`, gamma AND beta both
    /// applied in the f32 accumulator before the SINGLE implicit cast to
    /// the output dtype) and jammi's own fused CUDA kernel
    /// (`cuda/layer_norm.cu:124`: `yr[i] =
    /// __float2bfloat16(xhat * __bfloat162float(gamma[i]))`, one
    /// `__float2bfloat16` call, at the end). `mean`/`variance`/`xhat`/the
    /// affine are ALL computed in `internal_dtype` (f32 whenever `x_dtype`
    /// is F16/BF16); `weight`/`bias` are upcast to `internal_dtype` for the
    /// affine rather than mixing dtypes, and the whole result is cast to
    /// `x_dtype` exactly once, at the very end.
    ///
    /// Previously this rounded `xhat` to `x_dtype` BEFORE multiplying by
    /// `weight` (and, when biased, added `bias` as a further `x_dtype`
    /// op) — two-to-three rounding points instead of one. A measured,
    /// non-vacuous divergence at production shape (`hidden=1024`,
    /// `batch=2`, `seq` in `{128, 512}`) is the RED control in
    /// `tests::layer_norm_slow_matches_truth_at_production_shape_seq128`/
    /// `_seq512` — see those tests' own printed mismatch counts for a
    /// reproducible figure (no number is hardcoded here; the committed
    /// test is the producer). Every served bias-free (ModernBERT)
    /// LayerNorm output on an F16/BF16 backbone — training-eager fallback
    /// AND (through this same function) any `training=true` call that
    /// misses the fused kernel's admission domain — therefore changes at
    /// the ULP level; F32-backbone serving is UNCHANGED
    /// (`internal_dtype == x_dtype` there, so every `to_dtype` call below
    /// is a same-dtype no-op). eval (`training == false`) is UNCHANGED by
    /// this fix in call SHAPE (still routes to `candle_nn::ops::layer_norm`
    /// for the biased fast path, never `slow()`, when biased+eval+contiguous),
    /// but the biased eval fast path itself was never affected by this
    /// defect (candle's fused `layer_norm` already rounds once) — this
    /// fix changes ONLY `slow()`'s own output, i.e. the bias-free
    /// training-mode paths and any biased non-fast-path call.
    ///
    /// Domain check (K2): `weight`'s (and, when biased, `bias`'s) dtype
    /// must match `x`'s own dtype — mirroring `fused_admission_predicate`'s
    /// own `dtype_f32_or_bf16_matching_between_x_and_weight` check above.
    /// Before this check existed, a caller passing a mismatched-dtype
    /// weight got candle's own `broadcast_mul` dtype-mismatch error (the
    /// pre-fix code multiplied at `x_dtype` directly); the internal-dtype
    /// upcast this fix introduces (`weight.to_dtype(internal_dtype)`)
    /// would otherwise silently accept ANY weight dtype and produce a
    /// confident wrong number instead — a real domain-widening
    /// regression the fix must not introduce. See
    /// `tests::slow_refuses_a_dtype_mismatched_weight_instead_of_silently_upcasting`.
    fn slow(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        let x_dtype = x.dtype();
        if self.weight.dtype() != x_dtype {
            return Err(EncoderError::Config(format!(
                "LayerNorm::slow: weight dtype {:?} does not match x dtype {:?} -- refusing \
                 rather than silently upcasting a mismatched-dtype weight into `internal_dtype` \
                 (mirrors `fused_admission_predicate`'s own \
                 `dtype_f32_or_bf16_matching_between_x_and_weight` check)",
                self.weight.dtype(),
                x_dtype
            )));
        }
        if let Some(b) = &self.bias {
            if b.dtype() != x_dtype {
                return Err(EncoderError::Config(format!(
                    "LayerNorm::slow: bias dtype {:?} does not match x dtype {:?} -- same \
                     domain-validity refusal as the weight check above",
                    b.dtype(),
                    x_dtype
                )));
            }
        }
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden = x.dim(D::Minus1)?;
        let x_internal = x.to_dtype(internal_dtype)?;
        let mean = (x_internal.sum_keepdim(D::Minus1)? / hidden as f64)?;
        let centered = x_internal.broadcast_sub(&mean)?;
        let variance = (centered.sqr()?.sum_keepdim(D::Minus1)? / hidden as f64)?;
        let normalized = centered.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let weight_internal = self.weight.to_dtype(internal_dtype)?;
        let scaled_internal = normalized.broadcast_mul(&weight_internal)?;
        let out_internal = match &self.bias {
            None => scaled_internal,
            Some(b) => scaled_internal.broadcast_add(&b.to_dtype(internal_dtype)?)?,
        };
        Ok(out_internal.to_dtype(x_dtype)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};
    use half::bf16;

    fn bias_free_ln(weight: Tensor, eps: f64, training: bool) -> LayerNorm {
        LayerNorm {
            weight,
            bias: None,
            eps,
            training,
        }
    }

    /// The positive half of the device clause: CPU must satisfy it (every
    /// other test in this file relies on that implicitly; this pins it
    /// explicitly as its own assertion on the predicate's return value,
    /// not just "the forward call happened to succeed").
    #[test]
    fn fused_admission_predicate_accepts_cpu_device() {
        let device = Device::Cpu;
        let hidden = 4;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, hidden), &device).unwrap();
        let weight = Tensor::from_slice(&[1.0f32; 4], (hidden,), &device).unwrap();
        let (holds, predicate) = fused_admission_predicate(&x, &weight);
        assert!(holds, "CPU must satisfy the device clause: {predicate}");
    }

    /// The NEGATIVE half: a Metal device must be REJECTED. This IS
    /// hermetically testable with no `metal` feature on this crate at
    /// all: `candle_core` re-exports a `MetalDevice` type at its crate
    /// root regardless of whether ITS `metal` feature is on — the real
    /// backend's type when it is, and a public, zero-field dummy-backend
    /// unit struct (`pub struct MetalDevice;`, `dummy_metal_backend.rs`)
    /// when it is off — so `Device::Metal(MetalDevice)` is constructible
    /// today, unconditionally, with a bare unit-struct literal. This
    /// crate has no `metal` feature to gate on (declaring an empty one
    /// just to `#[cfg]` against it would be a phantom feature, and
    /// `cfg(feature = "metal")` on an undeclared feature trips rustc's
    /// `unexpected_cfgs` lint under `-D warnings`), so this test is
    /// unconditional: it exercises the dummy backend's zero-field
    /// `MetalDevice`, which is what candle-core actually compiles here.
    /// If this crate ever gains a real `metal` feature, `MetalDevice`
    /// becomes the real (non-unit) backend type and THIS specific
    /// construction stops compiling — a loud compile error flagging the
    /// exact test that needs replacing with a real Metal device/ordinal,
    /// not a silently-stale green test.
    #[test]
    fn device_is_supported_rejects_metal() {
        let metal = Device::Metal(candle_core::MetalDevice);
        assert!(
            !device_is_supported(&metal),
            "Metal must be rejected: LayerNormFused has no metal_fwd, and \
             candle's default metal_fwd errors rather than falling back"
        );
    }

    /// The eval-path bit-identity requirement: a `(bias.is_none(),
    /// training == false)` forward must be UNCHANGED by the fused
    /// kernel's existence, even on an input/weight pair that WOULD
    /// satisfy the fused admission domain if `training` were `true`
    /// (bf16, contiguous, `hidden` well within `MAX_HIDDEN`) — proving
    /// eval never reaches [`LayerNorm::forward_fused_or_fallback`]
    /// because the `match` in `forward` structurally routes it to
    /// `slow()`, not merely because this particular fixture happens to
    /// fail the domain check.
    #[test]
    fn eval_mode_forward_is_bit_identical_regardless_of_fused_eligibility() {
        let device = Device::Cpu;
        let hidden = 8;
        let xv: Vec<bf16> = (0..hidden)
            .map(|i| bf16::from_f32(i as f32 * 0.37 - 1.2))
            .collect();
        let gv: Vec<bf16> = (0..hidden)
            .map(|i| bf16::from_f32(1.0 + i as f32 * 0.05))
            .collect();
        let x = Tensor::from_slice(&xv, (1, hidden), &device).unwrap();
        let weight = Tensor::from_slice(&gv, (hidden,), &device).unwrap();
        assert!(x.is_contiguous());
        assert!(weight.is_contiguous());
        let (holds, _) = fused_admission_predicate(&x, &weight);
        assert!(
            holds,
            "fixture must satisfy the fused domain — the test proves eval \
             skips it anyway, not that the fixture happens to be ineligible"
        );

        let mut ln = bias_free_ln(weight, 1e-5, false);
        let before: Vec<Vec<bf16>> = ln.forward(&x).unwrap().to_vec2().unwrap();

        // Exercise the fused arm (this binary now has one) without
        // changing the eval call itself.
        ln.set_training(true);
        let _ = ln.forward(&x).unwrap();
        ln.set_training(false);

        let after: Vec<Vec<bf16>> = ln.forward(&x).unwrap().to_vec2().unwrap();
        assert_eq!(
            before, after,
            "eval-mode (training=false) forward must be byte-identical \
             before and after the fused kernel exists"
        );

        // And it is exactly `slow()` — eval's real, unchanged code path.
        let via_slow: Vec<Vec<bf16>> = ln.slow(&x).unwrap().to_vec2().unwrap();
        assert_eq!(before, via_slow);
    }

    /// The biased path (BERT/DistilBERT) never reaches the fused arm:
    /// `(Some(bias), _)` never matches `(None, true)`, structurally, for
    /// ANY value of `training` — `forward`'s match has exactly one arm
    /// that calls [`Self::forward_fused_or_fallback`], and it requires
    /// `bias` to be `None`. This test pins the OBSERVABLE consequence:
    /// a biased LayerNorm's output (both training and eval) is exactly
    /// the eager composition's, matching the pre-existing behavior this
    /// commit does not touch. (A global dispatch-counter assertion is
    /// deliberately NOT used here — `LN_DISPATCH_COUNTERS` is one
    /// process-wide static shared with every other test in this binary,
    /// so a snapshot-delta check would be racy under `cargo test`'s
    /// default parallel execution; the "assert their call sites don't
    /// switch" claim is instead a property of `forward`'s match itself,
    /// exercised here via its actual output.)
    #[test]
    fn biased_layer_norm_output_is_unaffected_in_training_and_eval() {
        let device = Device::Cpu;
        let hidden = 8;
        let weight = Tensor::from_slice(&[1.3f32; 8], (hidden,), &device).unwrap();
        let bias = Tensor::from_slice(&[0.2f32; 8], (hidden,), &device).unwrap();
        let x = Tensor::from_slice(
            &[0.5f32, -1.0, 2.0, 0.25, -0.5, 1.5, -2.0, 0.75],
            (1, hidden),
            &device,
        )
        .unwrap();

        let mut ln = LayerNorm {
            weight: weight.clone(),
            bias: Some(bias.clone()),
            eps: 1e-5,
            training: true,
        };
        let out_training: Vec<f32> = ln
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let expected_training: Vec<f32> = ln
            .slow(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(out_training, expected_training);

        ln.set_training(false);
        let out_eval: Vec<f32> = ln
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let expected_eval: Vec<f32> = candle_nn::ops::layer_norm(&x, &weight, &bias, 1e-5)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(out_eval, expected_eval);
    }

    /// Oracle 2 at the encoder level (per the fused-kernels plan's scope
    /// 7b, applied to the ACTUAL `slow()` this crate ships — the leaf
    /// `jammi-kernels` crate reproduces this composition in its own
    /// hermetic tests instead, since it cannot depend on this crate; see
    /// `jammi_kernels`' `tests/layer_norm_oracles.rs`). Compares the real
    /// dispatch path (`forward` with `bias.is_none() && training`)
    /// against `slow()` on the identical input, fwd AND bwd.
    #[test]
    fn fused_training_path_matches_slow_within_tolerance_fwd_and_bwd() {
        let device = Device::Cpu;
        let hidden = 8;
        let rows = 3;
        let xv: Vec<f32> = (0..rows * hidden)
            .map(|i| (i as f32 * 0.31 - 1.5).sin() * 3.0)
            .collect();
        let gv: Vec<f32> = (0..hidden).map(|i| 0.8 + i as f32 * 0.1).collect();

        let x_fused =
            Var::from_tensor(&Tensor::from_slice(&xv, (rows, hidden), &device).unwrap()).unwrap();
        let w_fused =
            Var::from_tensor(&Tensor::from_slice(&gv, (hidden,), &device).unwrap()).unwrap();
        let mut ln_fused = bias_free_ln(w_fused.as_tensor().clone(), 1e-5, true);
        ln_fused.training = true;

        let (holds, predicate) = fused_admission_predicate(x_fused.as_tensor(), &ln_fused.weight);
        assert!(holds, "fixture must be fused-eligible: {predicate}");
        // `LN_DISPATCH_COUNTERS` is one process-wide static shared with
        // every other test in this binary — under parallel test
        // execution an exact before+1 delta would be racy, so this only
        // asserts monotonic increase (other concurrent tests can only
        // add to it, never subtract).
        let before = LN_DISPATCH_COUNTERS.snapshot();
        let out_fused = ln_fused.forward(&x_fused).unwrap();
        let after = LN_DISPATCH_COUNTERS.snapshot();
        assert!(
            after.fused > before.fused,
            "this fixture must actually dispatch the fused kernel, not fall back \
             (before={before:?}, after={after:?})"
        );

        let x_eager =
            Var::from_tensor(&Tensor::from_slice(&xv, (rows, hidden), &device).unwrap()).unwrap();
        let w_eager =
            Var::from_tensor(&Tensor::from_slice(&gv, (hidden,), &device).unwrap()).unwrap();
        let ln_eager = bias_free_ln(w_eager.as_tensor().clone(), 1e-5, true);
        let out_eager = ln_eager.slow(&x_eager).unwrap();

        let vf: Vec<f32> = out_fused.flatten_all().unwrap().to_vec1().unwrap();
        let ve: Vec<f32> = out_eager.flatten_all().unwrap().to_vec1().unwrap();
        for (i, (f, e)) in vf.iter().zip(ve.iter()).enumerate() {
            assert!((f - e).abs() < 1e-4, "fwd[{i}]: fused {f} vs slow() {e}");
        }

        let grads_fused = out_fused.backward().unwrap();
        let grads_eager = out_eager.backward().unwrap();
        let dxf: Vec<f32> = grads_fused
            .get(&x_fused)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let dxe: Vec<f32> = grads_eager
            .get(&x_eager)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, (f, e)) in dxf.iter().zip(dxe.iter()).enumerate() {
            assert!((f - e).abs() < 1e-3, "dx[{i}]: fused {f} vs slow() {e}");
        }

        // `w_fused`/`w_eager` are both real `Var`s (trainable parameters
        // in this fixture) — `self.weight.is_variable()` must therefore
        // have set `dgamma_needed = true` on the fused call, and
        // `dgamma`'s slot must be populated and match the eager
        // composition's gradient for `gamma`. Before the
        // `is_variable()`-driven fix, this fixture was constructing an
        // UNSOUND state (a trainable `Var` gamma paired with a hardcoded
        // `dgamma_needed = false`): `grads_fused.get(&w_fused)` would
        // have been `None` here — no panic, just a silently missing
        // gradient a real AdamW step would skip (`backprop.rs:674-677`).
        let dgf: Vec<f32> = grads_fused
            .get(&w_fused)
            .expect(
                "dgamma_needed must be true for a trainable Var gamma \
                 (self.weight.is_variable()) — this must not be None",
            )
            .to_vec1()
            .unwrap();
        let dge: Vec<f32> = grads_eager.get(&w_eager).unwrap().to_vec1().unwrap();
        for (i, (f, e)) in dgf.iter().zip(dge.iter()).enumerate() {
            assert!((f - e).abs() < 1e-3, "dgamma[{i}]: fused {f} vs slow() {e}");
        }
    }

    /// The domain-widening regression check (K2): a BF16 `x` paired with
    /// an F32 `weight` must be REFUSED, not silently upcast into
    /// `internal_dtype` and rounded down to a confident wrong bf16
    /// number. This is the exact mismatch
    /// `fused_admission_predicate`'s own
    /// `dtype_f32_or_bf16_matching_between_x_and_weight` check refuses on
    /// the fused path; `slow()` must refuse it too.
    #[test]
    fn slow_refuses_a_dtype_mismatched_weight_instead_of_silently_upcasting() {
        let device = Device::Cpu;
        let hidden = 8;
        let xv: Vec<bf16> = (0..hidden)
            .map(|i| bf16::from_f32(i as f32 * 0.1))
            .collect();
        let x = Tensor::from_slice(&xv, (1, hidden), &device).unwrap();
        let weight = Tensor::from_slice(&[1.0f32; 8], (hidden,), &device).unwrap();
        let ln = bias_free_ln(weight, 1e-5, true);
        let err = ln
            .slow(&x)
            .expect_err("mismatched weight/x dtype must error, not silently compute");
        assert!(
            matches!(err, EncoderError::Config(_)),
            "expected a Config error naming the dtype mismatch, got {err:?}"
        );
    }

    /// Deterministic LCG walk producing PRODUCTION-AMPLITUDE f32 values in
    /// `[-half_width, half_width)`, tracked by its literal seed/multiplier/
    /// increment (not RNG-crate state) — the same convention
    /// `crate::test_support::deterministic_fill_varmap` uses at a
    /// narrower range, widened here so the bf16-rounded fixture spans
    /// several bf16 ULP steps and actually exercises a rounding-placement
    /// difference rather than a range where every rounding decision lands
    /// the same way regardless of where the cast sits.
    fn lcg_fixture(mut state: u32, n: usize, half_width: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let unit = (state >> 8) as f32 / (1u32 << 24) as f32; // [0, 1)
                (unit - 0.5) * 2.0 * half_width
            })
            .collect()
    }

    /// A from-scratch (no candle tensor ops, no `jammi_kernels` import)
    /// f32-accumulated, ascending-index, two-pass reference for the
    /// bias-free LayerNorm epilogue, rounded to bf16 EXACTLY ONCE at the
    /// end — the "torch placement" `slow()`'s doc pins to torch 2.13.0.
    /// Independently re-derived, not imported, from
    /// `jammi_kernels::ops::layer_norm`'s own private
    /// `mean_var_f32`/`ln_fwd_row_bf16` (this crate cannot import that
    /// private fn anyway) — the SAME fixed fold order (family J), so a
    /// bug shared by both implementations would not silently cancel.
    ///
    /// This fold order is NOT guaranteed to bit-match candle's own
    /// `Tensor::sum_keepdim` at production `hidden`: `sum_keepdim`'s CPU
    /// backend uses a SIMD-lane partial-sum reduction on targets where
    /// `neon`/`avx2`/`simd128` is enabled (candle-core 0.11.0's
    /// `cpu/mod.rs::vec_sum`), a DIFFERENT (still IEEE-754-correct, just
    /// differently associated) fold order than this function's plain
    /// left-to-right accumulation. That is a real, small,
    /// reduction-order-only divergence at production width — see
    /// `REDUCTION_ORDER_BUDGET_FRACTION`'s doc — orthogonal to the
    /// rounding-PLACEMENT defect `slow()`'s fix addresses.
    fn scalar_layer_norm_truth_bf16(
        x: &[bf16],
        gamma: &[bf16],
        hidden: usize,
        eps: f64,
    ) -> Vec<bf16> {
        let rows = x.len() / hidden;
        let eps = eps as f32;
        let mut out = Vec::with_capacity(x.len());
        for r in 0..rows {
            let row = &x[r * hidden..(r + 1) * hidden];
            let mut sum = 0f32;
            for v in row {
                sum += v.to_f32();
            }
            let mean = sum / hidden as f32;
            let mut sumsq = 0f32;
            for v in row {
                let d = v.to_f32() - mean;
                sumsq += d * d;
            }
            let variance = sumsq / hidden as f32;
            let invvar = 1.0 / (variance + eps).sqrt();
            for i in 0..hidden {
                let xhat = (row[i].to_f32() - mean) * invvar;
                out.push(bf16::from_f32(xhat * gamma[i].to_f32()));
            }
        }
        out
    }

    /// The PRE-FIX formula this commit removes: round `xhat` to bf16
    /// BEFORE multiplying by `gamma` — `bf16(bf16(xhat) * gamma)`, two
    /// rounding points instead of one. A deliberately WRONG
    /// reimplementation kept ONLY as this oracle's non-vacuity control:
    /// proves the fixture actually exercises the rounding-placement
    /// difference (mismatches against the truth on a stated,
    /// asserted-positive count), not a fixture that happens to round the
    /// same way regardless of where the cast sits.
    fn scalar_layer_norm_double_round_bf16(
        x: &[bf16],
        gamma: &[bf16],
        hidden: usize,
        eps: f64,
    ) -> Vec<bf16> {
        let rows = x.len() / hidden;
        let eps = eps as f32;
        let mut out = Vec::with_capacity(x.len());
        for r in 0..rows {
            let row = &x[r * hidden..(r + 1) * hidden];
            let mut sum = 0f32;
            for v in row {
                sum += v.to_f32();
            }
            let mean = sum / hidden as f32;
            let mut sumsq = 0f32;
            for v in row {
                let d = v.to_f32() - mean;
                sumsq += d * d;
            }
            let variance = sumsq / hidden as f32;
            let invvar = 1.0 / (variance + eps).sqrt();
            for i in 0..hidden {
                let xhat = (row[i].to_f32() - mean) * invvar;
                let xhat_bf16 = bf16::from_f32(xhat); // ROUND #1 (the pre-fix defect).
                out.push(bf16::from_f32(xhat_bf16.to_f32() * gamma[i].to_f32()));
                // ROUND #2.
            }
        }
        out
    }

    /// The ONLY source of disagreement between `slow()` (candle
    /// `Tensor::sum_keepdim`, SIMD-lane reduction on this crate's own
    /// dev/CI targets) and [`scalar_layer_norm_truth_bf16`] (ascending-
    /// index fold) that survives the one-rounding fix is reduction-ORDER
    /// noise in the mean/variance sums straddling a bf16 rounding
    /// boundary for a handful of elements — not the rounding-PLACEMENT
    /// bug. `2%` of the row count is a deliberately conservative ceiling
    /// over that mechanism (measured, on this crate's own dev target, to
    /// be a small fraction of a percent — see the printed count in
    /// `layer_norm_slow_matches_truth_at_production_shape`'s test
    /// output — this is headroom over that measurement, not a value
    /// tightened to it, so a different libm/SIMD width on another CI
    /// runner does not flake this test) and two orders of magnitude
    /// under the double-rounding control's own measured count (this same
    /// test asserts the control's count is far larger).
    const REDUCTION_ORDER_BUDGET_FRACTION: f64 = 0.02;

    /// Biting oracle (family F: measured live against an independently-
    /// derived reference, not a same-code tautology) at PRODUCTION
    /// shape — `hidden=1024`, `rows = batch * seq` for `batch=2`, `seq in
    /// {128, 512}` — calling the REAL `LayerNorm::slow` (not a
    /// reimplementation of it): `jammi-kernels` is a leaf crate and
    /// cannot reach this function at all (see that crate's
    /// `tests/layer_norm_oracles.rs` module doc), so THIS is the only
    /// place in the workspace that can exercise `slow()`'s actual
    /// dispatch against an independent numeric truth.
    ///
    /// Reverting this file's production `slow()` hunk (restoring the
    /// pre-fix two-round `normalized.to_dtype(x_dtype)?.broadcast_mul(&weight)`
    /// form) turns this test RED: `slow()`'s output then matches
    /// [`scalar_layer_norm_double_round_bf16`] almost everywhere instead
    /// of the truth reference, so `mismatch_vs_truth` blows past
    /// `REDUCTION_ORDER_BUDGET_FRACTION`'s budget.
    fn layer_norm_slow_matches_truth_at_production_shape(rows: usize, hidden: usize, seed: u32) {
        let device = Device::Cpu;
        let eps = 1e-5f64;
        let n = rows * hidden;

        let xf = lcg_fixture(seed, n, 24.0);
        let gf = lcg_fixture(seed.wrapping_add(0x9E37_79B9), hidden, 2.0);
        let x_bf16: Vec<bf16> = xf.iter().map(|&v| bf16::from_f32(v)).collect();
        let g_bf16: Vec<bf16> = gf.iter().map(|&v| bf16::from_f32(v)).collect();
        assert!(
            x_bf16.iter().all(|v| v.to_f32().is_finite()),
            "fixture x must be finite before any bit compare"
        );
        assert!(
            g_bf16.iter().all(|v| v.to_f32().is_finite()),
            "fixture gamma must be finite before any bit compare"
        );

        let x = Tensor::from_slice(&x_bf16, (rows, hidden), &device).unwrap();
        let weight = Tensor::from_slice(&g_bf16, (hidden,), &device).unwrap();
        let ln = bias_free_ln(weight.clone(), eps, true);

        let slow_out: Vec<bf16> = ln
            .slow(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let fused_out: Vec<bf16> = apply2(&x, &weight, LayerNormFused::new(eps, false))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            slow_out.iter().all(|v| v.to_f32().is_finite()),
            "slow() output must be finite before any bit compare"
        );
        assert!(
            fused_out.iter().all(|v| v.to_f32().is_finite()),
            "fused output must be finite before any bit compare"
        );

        let truth = scalar_layer_norm_truth_bf16(&x_bf16, &g_bf16, hidden, eps);
        assert!(
            truth.iter().all(|v| v.to_f32().is_finite()),
            "truth output must be finite before any bit compare"
        );

        // The fused CPU arm runs the SAME ascending-scalar algorithm this
        // truth reference does (independently re-derived, not imported) —
        // no candle-tensor-op reduction is involved on either side, so
        // this one IS bit-exact, unconditionally.
        assert_eq!(
            fused_out, truth,
            "LayerNormFused's CPU arm must be bit-exact vs the scalar truth (same fixed fold order)"
        );

        let mismatch_vs_truth = slow_out
            .iter()
            .zip(truth.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let budget = ((n as f64) * REDUCTION_ORDER_BUDGET_FRACTION).ceil() as usize;
        println!(
            "layer_norm_slow_matches_truth_at_production_shape(rows={rows}, hidden={hidden}): \
             slow() vs truth mismatches = {mismatch_vs_truth}/{n} (budget {budget})"
        );
        assert!(
            mismatch_vs_truth <= budget,
            "slow() diverged from the f32-round-once truth on {mismatch_vs_truth}/{n} \
             elements, past the {budget}-element reduction-order budget — this is the \
             rounding-PLACEMENT regression the fix restores, not reduction-order noise"
        );

        // RED CONTROL (non-vacuity): the pre-fix double-rounding formula
        // must differ from truth on a stated, ASSERTED-POSITIVE count —
        // proving the fixture actually exercises the rounding-placement
        // difference, and that its magnitude swamps the reduction-order
        // budget above (so the two mechanisms are told apart, not
        // conflated).
        let double_round = scalar_layer_norm_double_round_bf16(&x_bf16, &g_bf16, hidden, eps);
        assert!(
            double_round.iter().all(|v| v.to_f32().is_finite()),
            "double-round control output must be finite before any bit compare"
        );
        let mismatch_double_round = double_round
            .iter()
            .zip(truth.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!(
            "layer_norm_slow_matches_truth_at_production_shape(rows={rows}, hidden={hidden}): \
             double-round control vs truth mismatches = {mismatch_double_round}/{n}"
        );
        assert!(
            mismatch_double_round > 0,
            "RED control is vacuous: the double-rounding formula matched the truth on every \
             element (mismatch count 0) — this fixture does not exercise the \
             rounding-placement difference at all"
        );
        assert!(
            mismatch_double_round > budget * 5,
            "RED control's divergence ({mismatch_double_round}) must swamp the \
             reduction-order budget ({budget}) by a wide margin, or it is not actually \
             distinguishing the rounding-placement bug from ordinary reduction-order noise"
        );
    }

    #[test]
    fn layer_norm_slow_matches_truth_at_production_shape_seq128() {
        // batch=2, seq=128, hidden=1024 -> rows=256.
        layer_norm_slow_matches_truth_at_production_shape(2 * 128, 1024, 0xC0FF_EE01);
    }

    #[test]
    fn layer_norm_slow_matches_truth_at_production_shape_seq512() {
        // batch=2, seq=512, hidden=1024 -> rows=1024.
        layer_norm_slow_matches_truth_at_production_shape(2 * 512, 1024, 0xC0FF_EE02);
    }

    #[test]
    fn strict_mode_errors_instead_of_falling_back_on_a_failed_predicate() {
        // SAFETY (test-only): env var mutation is racy across threads in
        // general, but `admission_mode()` memoizes into a `OnceLock` the
        // first time it is called in this PROCESS — this test's value
        // only takes effect if it runs before anything else calls
        // `admission_mode()`. `cargo test`'s default per-test-thread
        // model makes ordering non-deterministic across the WHOLE binary,
        // so this test instead calls `jammi_kernels::admission::admit`
        // directly with an explicit `Strict` mode, exercising the exact
        // same code `forward_fused_or_fallback` runs without depending on
        // the env-var memoization's timing.
        use jammi_kernels::admission::{admit, AdmissionMode};
        let counters = jammi_kernels::admission::DispatchCounters::new();
        let err = admit(
            AdmissionMode::Strict,
            "layer_norm_fused",
            "x_contiguous",
            false,
            &counters,
        )
        .expect_err("a failed predicate in Strict mode must error");
        assert!(matches!(
            err,
            jammi_kernels::error::KernelError::StrictModeFallback {
                op: "layer_norm_fused",
                predicate: "x_contiguous"
            }
        ));
    }

    /// The admission/counter key this crate dispatches a fused path under
    /// is a call-site literal, independent of the kernel op's `name()` by
    /// construction (`jammi_kernels::admission::counters_for`'s doc): an
    /// admission key names a consumer's fused PATH, which may compose
    /// several ops, so it can legitimately differ from any one
    /// `CustomOp`'s name (the LoRA consumer keys `"lora_linear_fused"`
    /// over the op named `"low_rank_residual_linear"`). Where this crate
    /// keys a path by the op's own name — layer-norm and softmax — that
    /// coincidence is what lets a counters snapshot be read side by side
    /// with the op's error payloads, so it is pinned here without a third
    /// literal: the registry entry each `*_DISPATCH_COUNTERS` resolves to
    /// must be the very entry `counters_for(op.name())` resolves to
    /// (`counters_for` hands back the same `&'static` for the same key).
    /// The `admit(..)` call sites' `op` argument is the same key by
    /// convention but has no read-back API (it feeds a log-once WARN and
    /// the `StrictModeFallback` payload), so it stays pinned only by the
    /// strict-mode tests' literal matches above.
    #[test]
    fn dispatch_counter_keys_agree_with_the_kernel_ops_names() {
        use candle_core::CustomOp2;
        use jammi_kernels::admission::counters_for;
        use jammi_kernels::ops::{LayerNormFused, SoftmaxLastDimFused};
        assert!(
            std::ptr::eq(
                counters_for(LayerNormFused::new(1e-5, false).name()),
                *LN_DISPATCH_COUNTERS
            ),
            "LN_DISPATCH_COUNTERS is keyed by a literal that drifted from LayerNormFused::name()"
        );
        assert!(
            std::ptr::eq(
                counters_for(SoftmaxLastDimFused::default().name()),
                *crate::modernbert::SOFTMAX_DISPATCH_COUNTERS
            ),
            "SOFTMAX_DISPATCH_COUNTERS is keyed by a literal that drifted from \
             SoftmaxLastDimFused::name()"
        );
    }
}
