//! Fused erf-based GELU forward + backward — `GeluErfFused`, a `CustomOp1`
//! that reproduces `candle_core::Tensor::gelu_erf`'s own per-dtype
//! arithmetic bit-for-bit (CPU F32; CUDA F32/BF16/F16), never a
//! re-derivation with its own precision.
//!
//! A generic Tensor-API primitive (family L: this crate names no consumer);
//! its real call site is `jammi-encoders`' training-mode GELU seam
//! (`activations::gelu_erf`, wired at BERT/DistilBERT's `Intermediate`/`Ffn`
//! sites — a companion branch of this same contract).
//!
//! ## Three cdf formulations in this crate — pin which one this op tracks
//!
//! `GeluErfFused` is this crate's THIRD, independent "compute a Gaussian
//! CDF for a GELU-shaped activation" implementation, and each one
//! deliberately tracks a DIFFERENT upstream reference rather than sharing
//! one "the" formula:
//!
//! 1. **candle CPU (`statrs`-derived erf polynomial).** `candle_core`'s own
//!    `GeluErf::f32` (`op.rs:1010-1012`) computes
//!    `(crate::cpu::erf::erf_f32(v * FRAC_1_SQRT_2) + 1.) * 0.5 * v`, where
//!    `crate::cpu::erf::erf_f32` is a CRATE-PRIVATE wrapper (not
//!    re-exported by candle-core) around a rational-polynomial erf
//!    approximation. This op's OWN CPU F32 arm ([`gelu_erf_cdf_f32`] below)
//!    reproduces that EXACT formula via [`libm::erff`] instead — the same
//!    substitution `ops::geglu`'s module doc documents and justifies (this
//!    crate depends on `libm` directly precisely because candle does not
//!    re-export its own erf), giving bit-identical CPU F32 forward output
//!    to `Tensor::gelu_erf` (pinned by
//!    [`tests::cpu_f32_forward_bit_identical_to_candle_gelu_erf`] and the
//!    leaf-crate oracle `tests/gelu_erf_oracles.rs`).
//! 2. **candle CUDA (`normcdff`).** candle-kernels' `ugelu_erf_{f32,bf16,f16}`
//!    (`unary.cu:31-34,118,174`; `cuda_utils.cuh:140-195`) computes
//!    `x * normcdfg(x)` using CUDA's own hardware `normcdff`/`normcdf`
//!    intrinsic — a DIFFERENT numerical routine from `erf_f32`'s polynomial
//!    (both approximate the same mathematical CDF, to different rounding).
//!    This op's CUDA arm (`crate::cuda::gelu_erf`) reproduces THIS formula
//!    exactly, including candle's own bf16/f16 DOUBLE-ROUNDING shape (see
//!    "16-bit CUDA forward" below) — this op exists specifically to be the
//!    fused, single-kernel equivalent of candle's own eager CUDA dispatch,
//!    so it tracks CUDA's reference, not CPU's, on that device.
//! 3. **GeGLU (`erff`, `kernels-community`-derived).** [`crate::ops::geglu`]'s
//!    own `gelu_erf_f32` computes `x * 0.5 * (1 + erff(x/sqrt(2)))` via
//!    [`libm::erff`] directly (not through candle's polynomial at all) —
//!    chosen there to match the upstream HF/`kernels-community`
//!    `gelu_and_mul` "fp32 opmath" reference, a DIFFERENT design goal from
//!    this op's "track candle's own eager arm bit-for-bit" goal. See that
//!    module's doc for the full rationale; the two ops' CPU F32 answers
//!    happen to be numerically very close (same underlying `erff` call,
//!    same algebraic identity `Phi(x) = 0.5*(1+erf(x/sqrt(2)))`) but are
//!    NOT the same code path and are not asserted bit-identical to each
//!    other anywhere in this crate.
//!
//! **Why this op tracks candle's own formulas (1) and (2), not GeGLU's
//! (3):** this op's whole purpose is to let `jammi-encoders`' training arm
//! dispatch a fused kernel that is numerically INDISTINGUISHABLE from the
//! eager `x.gelu_erf()?` call it replaces (unlike GeGLU, which replaces a
//! DIFFERENT eager composition — `narrow`+`narrow`+`gelu_erf`+`mul` — that
//! never runs standalone `gelu_erf` on the packed tensor at all, so has no
//! "track this call bit-for-bit" target to begin with). A caller comparing
//! `GeluErfFused` against `Tensor::gelu_erf()?` therefore expects (and
//! gets) bit-identical CPU F32, and the documented double-rounding-exact
//! CUDA behaviour below — not GeGLU's own, differently-motivated tolerance.
//!
//! ## Forward CUDA — match candle's dtype-native rounding EXACTLY
//!
//! - **F32**: `x * normcdff(x)` — one rounding, no intermediate at all.
//! - **BF16/F16 (double rounding, matching candle's `ugelu_erf_{bf16,f16}`
//!   bit-for-bit)**: `cdf16 = round16(normcdff(f32(x)))` (**ROUND 1**,
//!   candle-kernels' `normcdfg(bf16/half)` — `cuda_utils.cuh:174,195` —
//!   rounds the CDF to the 16-bit dtype BEFORE the multiply, not after);
//!   `out = round16(f32(x) * f32(cdf16))` (**ROUND 2** — candle's own
//!   `x * normcdfg(x)` uses the 16-bit type's native `operator*`, i.e.
//!   `__hmul`/bf16's own multiply, which correctly rounds the EXACT product
//!   of two values that are themselves exactly representable in f32 — bit-
//!   identical to computing the product in f32 and rounding once). This is
//!   a real, disclosed TWO-rounding-point regime (like `GegluFused`'s own
//!   forward — see that module's doc's "bf16 boundary-rounding" section for
//!   the general shape), chosen here not by a design decision but because
//!   it is EXACTLY what the eager arm this op fuses away already does —
//!   `tests/cuda_parity.rs`'s CUDA legs assert `==` (bit-exact, not merely
//!   within tolerance) against `Tensor::gelu_erf()?` on CUDA at every
//!   admitted dtype.
//!
//! ## Forward CPU — F32 only
//!
//! `(erf_f32(x * FRAC_1_SQRT_2) + 1.) * 0.5 * x`, via [`libm::erff`] — see
//! formulation (1) above. BF16/F16 have NO CPU arm (`UnsupportedDTypeForOp`,
//! a typed refusal, not a silent fallback): candle's own CPU BF16/F16
//! `GeluErf` arms compute in **f64** (`bf16::from_f64(Self::f64(v.to_f64()))`,
//! `op.rs:1003-1008`) — a THIRD, wider-precision path this op does not
//! reproduce on CPU (unlike CUDA, where matching candle's own dtype-native
//! rounding is exactly this op's job) — so admitting CPU BF16/F16 here
//! would silently diverge from `Tensor::gelu_erf()?` in a way this op's own
//! "track candle bit-for-bit" contract cannot honor without a THIRD CPU
//! code path solely to reproduce an f64 detour. `jammi-encoders`' admission
//! predicate treats this refusal as a counted eager fallback, never a hard
//! error outside `Strict` mode.
//!
//! ## Backward — ONE kernel, `dx = dy * (Phi(x) + x*phi(x))`
//!
//! Standard GELU(erf) derivative: `Phi(x) = 0.5*(1+erf(x/sqrt(2)))` the
//! Gaussian CDF, `phi(x) = kBeta * exp(-x^2/2)` the Gaussian PDF,
//! `kBeta = 1/sqrt(2*pi)`. `Phi` uses the erf form on CPU (formulation (1)
//! above — the SAME `erf_f32` this op's own forward already computes) and
//! `normcdff`/`normcdf` on CUDA (formulation (2)) — the SAME per-device cdf
//! this op's own forward uses, so forward and backward never disagree on
//! which Gaussian-CDF routine represents `x`'s own dtype/device. `phi`'s
//! normalizing constant, `kBeta = 0.3989422804014327f`
//! (`M_2_SQRTPI * M_SQRT1_2 * 0.5 == 1/sqrt(2*pi)`), is the exact ATen
//! `kBeta` constant `ops::geglu`'s own backward derivation cites and uses —
//! see that module's doc for the ATen citation; [`GELU_ERF_ALPHA_F32`]/
//! [`GELU_ERF_BETA_F32`] below are this file's OWN copies (not shared with
//! `ops::geglu`'s identically-valued constants — each op's `.rs`/`.cu` file
//! is a self-contained translation unit in this crate's convention, the
//! same "duplicate rather than share" idiom `geglu_f16.cu`'s module doc
//! states for its own reasons). Every term is computed in f32 and rounded
//! ONCE at the store (bf16/f16 upcast at load, matching this crate's usual
//! "f32-accumulate, round once" backward convention — `ops::layer_norm`'s
//! `LayerNormBwdDx`, `ops::softmax`'s `SoftmaxBwdDScores`,
//! `ops::geglu`'s `GegluBwdDWiOut`).
//!
//! `bwd` keys its single gradient slot on `arg.track_op()` (esc-053's
//! class fix), NOT `is_variable()` alone: `arg` may be an INTERMEDIATE on a
//! path to a `Var` (`is_variable() == false`, `track_op() == true`) the
//! same way any op's argument can (`ops`'s module doc, "`is_variable()` is
//! NOT a 'does this need a gradient?' gate"), and `arg.track_op() == false`
//! is the STRUCTURAL condition under which no downstream `Var` can possibly
//! need this gradient at all — the SAME sanctioned exception shape
//! `RopeFused::bwd`'s `dcos`/`dsin` checks document (`ops::rope`'s module
//! doc, "`bwd`: RoPE with the sign of `sin` flipped"), narrowed to this
//! op's single differentiable argument. `bwd` returns `None` ONLY in that
//! one case; every tracked `arg` gets a real, computed gradient — never a
//! hardcoded `None` for a merely-not-yet-reached call site.
//!
//! Tape shape: `x.gelu_erf()?` is ALREADY a single forward tape node
//! (candle's own `Op::Unary(_, GeluErf)`), so — unlike `GegluFused`'s
//! narrow+narrow+mul collapse — this op's win is not a FORWARD node-count
//! reduction (`sorted_nodes()` on `GeluErfFused`'s own output is the same
//! `[x, out]`, length 2, as eager's). The win is that candle's backward for
//! `Op::Unary(_, GeluErf)` evaluates its closed-form derivative as a chain
//! of ~11 SEPARATE `Tensor` ops (`backprop.rs:624-633`: `sqr`, `neg`,
//! `affine`, `exp`, `affine`, `mul`, `affine`, `erf`, `affine`, `affine`,
//! `add`, `mul`) — each a real allocation/launch, even though none of them
//! is itself pushed onto the FORWARD graph's `sorted_nodes()` (they belong
//! to a throwaway sub-graph `backward()` never re-differentiates) — while
//! `GeluErfFused::bwd` computes the identical closed form in ONE internal
//! `CustomOp2` ([`GeluErfBwdDx`]) node.
//! [`tests::bwd_helper_is_one_node_vs_the_same_closed_form_built_from_ordinary_tensor_ops`]
//! demonstrates this the only way `sorted_nodes()` CAN show it: by building
//! that same closed form explicitly as an ordinary forward `Tensor`
//! composition (candle's own backward algebra, reproduced verbatim) and
//! comparing ITS node count against `GeluErfBwdDx`'s single node.
//!
//! ## Domain (family D / K2)
//!
//! `x` must be fully contiguous ([`candle_core::Layout::contiguous_offsets`],
//! the same idiom every other op in this crate uses — this op's kernel has
//! no per-row structure at all (pure elementwise, unlike `GegluFused`'s
//! gate/up split or `LayerNormFused`'s row reduction), so the ONLY reason
//! contiguity is required is the same one every other op states: a
//! raw-pointer kernel needs a flat linear index) and NON-EMPTY: unlike
//! several sibling ops (`GegluFused`'s `last == 0`, `LayerNormFused`'s
//! `hidden == 0`), a zero-element `x` is a TYPED REFUSAL here, not a
//! degenerate no-op — this op's real call site (`jammi-encoders`' GELU
//! seam) never hands it an empty tensor, and refusing loudly is simpler and
//! strictly safer than adding a second "build an empty output of the right
//! dtype" code path this op has no live caller to exercise (`tests::
//! empty_input_is_a_typed_refusal_not_a_silent_no_op` pins the refusal).
//! CPU supports F32 only; CUDA supports F32, BF16, F16 — any other dtype
//! (on either device) is `Error::UnsupportedDTypeForOp`. No CUDA-specific
//! width ceiling beyond the crate-wide `u32::MAX` element-count guard
//! (`ops::launch_domain::check_elem_count_fits_u32`): this op's kernels
//! have no per-row shared-memory footprint.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, CustomOp2, Error, Layout, Result, Shape, Tensor};

/// `1/sqrt(2)` — this file's own copy of the same ATen `kAlpha` constant
/// `ops::geglu::GELU_ALPHA_F32` documents (see this module's doc, "backward"
/// section, for why each op's `.rs` keeps its own copy rather than sharing
/// one).
const GELU_ERF_ALPHA_F32: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// `1/sqrt(2*pi)` — this file's own copy of the same ATen `kBeta` constant
/// `ops::geglu::GELU_BETA_F32` documents.
const GELU_ERF_BETA_F32: f32 =
    std::f32::consts::FRAC_2_SQRT_PI * std::f32::consts::FRAC_1_SQRT_2 * 0.5;

/// `Phi(x) = 0.5*(1+erf(x/sqrt(2)))`, the standard-normal CDF, via
/// [`libm::erff`] — bit-identical to `candle_core`'s own (crate-private)
/// `crate::cpu::erf::erf_f32`-based `GeluErf::f32` formula. See the module
/// doc's "three cdf formulations" section, item 1.
fn gelu_erf_cdf_f32(x: f32) -> f32 {
    (libm::erff(x * GELU_ERF_ALPHA_F32) + 1.0) * 0.5
}

/// `gelu_erf(x) = x * Phi(x)` — matches `Tensor::gelu_erf`'s CPU F32 arm
/// exactly (same formula, same underlying erf routine).
fn gelu_erf_fwd_f32(x: f32) -> f32 {
    x * gelu_erf_cdf_f32(x)
}

/// `d/dx gelu_erf(x) = Phi(x) + x*phi(x)`, `phi(x) = kBeta*exp(-x^2/2)` —
/// see the module doc's "backward" section for the ATen citation.
fn gelu_erf_grad_f32(x: f32) -> f32 {
    let cdf = gelu_erf_cdf_f32(x);
    let pdf = GELU_ERF_BETA_F32 * (-0.5 * x * x).exp();
    cdf + x * pdf
}

/// Validates the domain every arm of this op shares (see the module doc's
/// "domain" section) and returns the contiguous `[o1, o2)` byte-offset-free
/// element range `l`'s storage occupies. `pub(crate)`: `crate::cuda::gelu_erf`
/// shares this exact check rather than re-deriving it — the same
/// "one definition per op's domain check" convention `ops::geglu::geglu_dims`
/// documents for its own reasons.
pub(crate) fn check_domain(l: &Layout, op: &'static str) -> Result<(usize, usize)> {
    if l.shape().elem_count() == 0 {
        return Err(Error::Msg(format!(
            "{op}: input has 0 elements — refused as a domain violation rather than \
             treated as a degenerate no-op (this op's real call site never hands it an \
             empty tensor, and a second 'build an empty output' code path would carry \
             maintenance cost with no live caller to exercise it)."
        )));
    }
    l.contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op })
}

/// Fused erf-based GELU forward. See the module doc for the full design.
#[derive(Debug, Clone, Copy, Default)]
pub struct GeluErfFused;

impl super::sealed::Sealed for GeluErfFused {}

impl CustomOp1 for GeluErfFused {
    fn name(&self) -> &'static str {
        "gelu_erf_fused"
    }

    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout) -> Result<(CpuStorage, Shape)> {
        let (o1, o2) = check_domain(l1, self.name())?;
        match s1 {
            CpuStorage::F32(x) => {
                let out: Vec<f32> = x[o1..o2].iter().map(|&v| gelu_erf_fwd_f32(v)).collect();
                Ok((CpuStorage::F32(out), l1.shape().clone()))
            }
            s => Err(Error::UnsupportedDTypeForOp(s.dtype(), self.name())),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        crate::cuda::gelu_erf::cuda_fwd(s1, l1)
    }

    /// See the module doc's "backward" section: `dx`'s slot is keyed on
    /// `arg.track_op()` (esc-053), not `is_variable()` alone.
    fn bwd(&self, arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        if !arg.track_op() {
            return Ok(None);
        }
        let dx = super::apply2(arg, grad_res, GeluErfBwdDx)?;
        Ok(Some(dx))
    }
}

/// `GeluErfFused`'s internal backward helper producing `dx` in ONE kernel
/// launch. Not exported — invoked only from [`GeluErfFused::bwd`] via
/// [`super::apply2`].
#[derive(Debug, Clone, Copy)]
struct GeluErfBwdDx;

impl super::sealed::Sealed for GeluErfBwdDx {}

impl CustomOp2 for GeluErfBwdDx {
    fn name(&self) -> &'static str {
        "gelu_erf_fused_bwd_dx"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        if l1.dims() != l2.dims() {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: l1.shape().clone(),
                rhs: l2.shape().clone(),
                op: self.name(),
            });
        }
        if s1.dtype() != s2.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: self.name(),
            });
        }
        let (o1, o2) = check_domain(l1, self.name())?;
        let (d1, d2) = l2
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: self.name() })?;
        match (s1, s2) {
            (CpuStorage::F32(x), CpuStorage::F32(dy)) => {
                let out: Vec<f32> = x[o1..o2]
                    .iter()
                    .zip(dy[d1..d2].iter())
                    .map(|(&xv, &dyv)| dyv * gelu_erf_grad_f32(xv))
                    .collect();
                Ok((CpuStorage::F32(out), l1.shape().clone()))
            }
            (s1, _) => Err(Error::UnsupportedDTypeForOp(s1.dtype(), self.name())),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        crate::cuda::gelu_erf::cuda_bwd_dx(s1, l1, s2, l2)
    }

    // No `bwd` override: this helper's own second-order gradient is never
    // requested by any call site in this crate or its consumers — mirroring
    // `GegluBwdDWiOut`'s/`LayerNormBwdDx`'s identical notes.
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    fn fused(x: &Tensor) -> Result<Tensor> {
        crate::ops::apply1(x, GeluErfFused)
    }

    /// Sign-mixed, production-amplitude grid (family D boundary/degenerate
    /// oracle: negative, zero, small-positive, large-positive, and a
    /// production-scale tail) — bit-identical to `Tensor::gelu_erf()?` on
    /// CPU F32. The leaf-crate integration suite (`tests/gelu_erf_oracles.rs`)
    /// repeats this at a larger, non-toy width.
    #[test]
    fn cpu_f32_forward_bit_identical_to_candle_gelu_erf() {
        let device = Device::Cpu;
        let v: [f32; 10] = [-6.0, -3.0, -1.0, -0.3, -0.0001, 0.0, 0.0001, 0.5, 2.0, 9.0];
        let x = Tensor::from_slice(&v, (v.len(),), &device).unwrap();
        let fused_out: Vec<f32> = fused(&x).unwrap().to_vec1().unwrap();
        let eager_out: Vec<f32> = x.gelu_erf().unwrap().to_vec1().unwrap();
        for (i, (&f, &e)) in fused_out.iter().zip(eager_out.iter()).enumerate() {
            assert!(
                f.to_bits() == e.to_bits(),
                "elem[{i}]: fused {f} (0x{:08x}) vs eager {e} (0x{:08x}) must be bit-identical",
                f.to_bits(),
                e.to_bits()
            );
        }
    }

    #[test]
    fn empty_input_is_a_typed_refusal_not_a_silent_no_op() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[] as &[f32], (0,), &device).unwrap();
        let err = fused(&x).expect_err("an empty input must be refused, not silently accepted");
        assert!(matches!(err, Error::Msg(_)));
    }

    #[test]
    fn non_contiguous_input_is_refused_not_silently_misread() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), &device)
            .unwrap()
            .t()
            .unwrap();
        assert!(!x.is_contiguous());
        let err = fused(&x).expect_err("a non-contiguous input must be refused");
        assert!(matches!(err, Error::RequiresContiguous { .. }));
    }

    #[test]
    fn cpu_bf16_is_a_typed_refusal_not_a_silent_fallback() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, 2.0], (2,), &device)
            .unwrap()
            .to_dtype(candle_core::DType::BF16)
            .unwrap();
        let err = fused(&x).expect_err("CPU BF16 has no arm — it must be a typed refusal");
        assert!(matches!(
            err,
            Error::UnsupportedDTypeForOp(candle_core::DType::BF16, _)
        ));
    }

    #[test]
    fn cpu_f16_is_a_typed_refusal_not_a_silent_fallback() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, 2.0], (2,), &device)
            .unwrap()
            .to_dtype(candle_core::DType::F16)
            .unwrap();
        let err = fused(&x).expect_err("CPU F16 has no arm — it must be a typed refusal");
        assert!(matches!(
            err,
            Error::UnsupportedDTypeForOp(candle_core::DType::F16, _)
        ));
    }

    #[test]
    fn unsupported_dtype_is_a_typed_refusal() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1u32, 2], (2,), &device).unwrap();
        let err = fused(&x).expect_err("U32 has no gelu_erf_fused CPU arm");
        assert!(matches!(
            err,
            Error::UnsupportedDTypeForOp(candle_core::DType::U32, _)
        ));
    }

    /// `bwd` vs. central finite differences, spanning negative/near-zero/
    /// positive/tail `x` — the region where `Phi(x) + x*phi(x)`'s two terms
    /// partially cancel (crossing zero near `x = -0.75`) and a sign error
    /// is likeliest to hide.
    #[test]
    fn gradcheck_dx_spans_negative_near_zero_and_tail_values() {
        let device = Device::Cpu;
        let x0: [f32; 7] = [-4.0, -0.7517915, -0.1, 0.0, 0.2, 1.5, 5.0];

        let x = Var::from_tensor(&Tensor::from_slice(&x0, (x0.len(),), &device).unwrap()).unwrap();
        let out = fused(&x).unwrap();
        let grads = out.sum_all().unwrap().backward().unwrap();
        let dx: Vec<f32> = grads.get(&x).unwrap().to_vec1().unwrap();

        let sum_fwd = |x: &Tensor| -> f64 {
            fused(x)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap() as f64
        };
        let fd_eps = 2e-3f32;
        let tol = 5e-2f64;
        for i in 0..x0.len() {
            let mut xp = x0;
            xp[i] += fd_eps;
            let mut xm = x0;
            xm[i] -= fd_eps;
            let xp_t = Tensor::from_slice(&xp, (x0.len(),), &device).unwrap();
            let xm_t = Tensor::from_slice(&xm, (x0.len(),), &device).unwrap();
            let numeric = (sum_fwd(&xp_t) - sum_fwd(&xm_t)) / (2.0 * fd_eps as f64);
            assert!(
                (numeric - dx[i] as f64).abs() < tol,
                "dx[{i}]: numeric {numeric} vs analytic {} (x0={x0:?})",
                dx[i]
            );
        }
    }

    /// `bwd` returns `None` for an untracked `arg` (esc-053's class: this
    /// is the sanctioned exception, gated on the STRUCTURAL `track_op()`
    /// predicate, not on `is_variable()` alone).
    #[test]
    fn bwd_returns_none_for_an_untracked_arg() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, -2.0, 0.5], (3,), &device).unwrap();
        assert!(!x.track_op());
        let op = GeluErfFused;
        let res = fused(&x).unwrap();
        let grad_res = Tensor::ones_like(&res).unwrap();
        let d = op.bwd(&x, &res, &grad_res).unwrap();
        assert!(
            d.is_none(),
            "an untracked arg must yield None, not a computed-but-unused gradient"
        );
    }

    /// Chain-rule-through-an-intermediate regression: `x` is an
    /// INTERMEDIATE (`is_variable() == false`, `track_op() == true`) on a
    /// path to a `Var` — the exact predicate hole `track_op()` (not
    /// `is_variable()`) closes.
    #[test]
    fn bwd_chains_through_an_intermediate_non_variable_x() {
        let device = Device::Cpu;
        let w0: [f32; 4] = [0.5, -1.0, 2.0, -0.25];
        let w = Var::from_tensor(&Tensor::from_slice(&w0, (4,), &device).unwrap()).unwrap();
        let x = w.affine(1.5, 0.2).unwrap();
        assert!(!x.is_variable() && x.track_op());
        let out = fused(&x).unwrap();
        let grads = out.sum_all().unwrap().backward().unwrap(); // must not panic
        let dw: Vec<f32> = grads.get(&w).unwrap().to_vec1().unwrap();
        assert!(
            dw.iter().any(|&g| g != 0.0),
            "gradient must be nonzero: {dw:?}"
        );
    }

    /// `x.gelu_erf()?`'s own forward tape is ALREADY one node
    /// (`[x, out]`, length 2) — this op's forward does not reduce that
    /// (module doc's "tape shape" section). Pinned here so the claim is
    /// measured, not merely asserted in prose.
    #[test]
    fn fused_forward_tape_matches_eagers_own_single_node_shape() {
        let device = Device::Cpu;
        let x0: [f32; 4] = [1.0, -2.0, 0.5, 3.0];

        let w_fused = Var::from_tensor(&Tensor::from_slice(&x0, (4,), &device).unwrap()).unwrap();
        let fused_nodes = fused(&w_fused).unwrap().sorted_nodes().len();

        let w_eager = Var::from_tensor(&Tensor::from_slice(&x0, (4,), &device).unwrap()).unwrap();
        let eager_nodes = w_eager.gelu_erf().unwrap().sorted_nodes().len();

        assert_eq!(
            fused_nodes, 2,
            "[x, out] -- the leaf Var plus one CustomOp1 node"
        );
        assert_eq!(
            eager_nodes, fused_nodes,
            "eager's own `Op::Unary(_, GeluErf)` is already a single forward node too -- \
             this op's win is in the BACKWARD launch count, not the forward tape shape"
        );
    }

    /// Reproduces candle's OWN `Op::Unary(_, GeluErf)` backward algebra
    /// (`backprop.rs:624-633`) verbatim, but as an explicit FORWARD
    /// `Tensor` composition rather than inside `backward()`'s own walk —
    /// i.e. what a hand-rolled (non-fused) implementation of this exact
    /// closed-form gradient would have to build. Uses candle's own
    /// truncated `0.398942` literal deliberately (this is a structural
    /// "how many nodes does this formula need" comparison, not a numerics
    /// oracle — [`gradcheck_dx_spans_negative_near_zero_and_tail_values`]
    /// and the leaf-crate `tests/gelu_erf_oracles.rs` already cover
    /// numerics).
    fn backprop_style_gelu_erf_grad(arg: &Tensor) -> Result<Tensor> {
        let neg_half_square = (arg.sqr()?.neg()? / 2.)?;
        let scaled_exp_arg = (0.398942 * neg_half_square.exp()? * arg)?;
        let arg_scaled_sqrt = (arg / 2f64.sqrt())?;
        let erf_scaled_sqrt = (0.5 * arg_scaled_sqrt.erf()?)?;
        0.5 + scaled_exp_arg + erf_scaled_sqrt
    }

    /// The tape-node-count oracle the module doc's "tape shape" section
    /// promises: `GeluErfBwdDx` (ONE `CustomOp2` node) vs. the SAME closed
    /// form built from ordinary, separately-tracked `Tensor` ops.
    #[test]
    fn bwd_helper_is_one_node_vs_the_same_closed_form_built_from_ordinary_tensor_ops() {
        let device = Device::Cpu;
        let x0: [f32; 4] = [1.0, -2.0, 0.5, 3.0];
        let dy0: [f32; 4] = [0.5, 1.0, -0.5, 2.0];

        let x_f = Var::from_tensor(&Tensor::from_slice(&x0, (4,), &device).unwrap()).unwrap();
        let dy_f = Tensor::from_slice(&dy0, (4,), &device).unwrap();
        let dx_f = crate::ops::apply2(&x_f, &dy_f, GeluErfBwdDx).unwrap();
        let fused_nodes = dx_f.sorted_nodes().len();

        let x_e = Var::from_tensor(&Tensor::from_slice(&x0, (4,), &device).unwrap()).unwrap();
        let dy_e = Tensor::from_slice(&dy0, (4,), &device).unwrap();
        let grad_e = backprop_style_gelu_erf_grad(&x_e).unwrap();
        let dx_e = (&dy_e * &grad_e).unwrap();
        let eager_nodes = dx_e.sorted_nodes().len();

        assert_eq!(
            fused_nodes, 2,
            "[x, dx] -- the leaf Var plus one CustomOp2 node"
        );
        assert!(
            fused_nodes < eager_nodes,
            "the fused backward helper must retain FEWER tape nodes than the same closed \
             form built from ordinary Tensor ops: fused={fused_nodes} eager={eager_nodes}"
        );
        assert!(
            eager_nodes >= 8,
            "sanity: the hand-composed closed form must actually retain multiple nodes for \
             this comparison to mean anything, got {eager_nodes}"
        );
    }
}
