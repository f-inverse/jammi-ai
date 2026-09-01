//! Fused, bias-free LayerNorm forward + backward.
//!
//! `y = ((x - mean) * invvar) * gamma`, reduced over the LAST dimension
//! ("hidden"); `x` is `[rows, hidden]` (any leading batch shape, flattened
//! to `rows`), `gamma` is `[hidden]`. This is the bias-free variant only —
//! BERT/DistilBERT's biased LayerNorm keeps its existing (unfused) path;
//! see `jammi-encoders`' call site for why.
//!
//! ## No save-for-backward (candle 0.11)
//!
//! `CustomOp2::fwd` returns exactly one `(Storage, Shape)` — there is no
//! channel to stash `mean`/`invvar` for `bwd` to read back later (unlike,
//! say, PyTorch's `ctx.save_for_backward`). `bwd` therefore RECOMPUTES
//! `mean`/`invvar` from `x` (one extra read of `x`, budgeted per the
//! fused-kernels plan) rather than caching anything in the op itself —
//! which would also violate the `Copy`/stateless requirement every op in
//! this crate is held to (see `ops`'s module doc).
//!
//! ## Three kernels, one call site
//!
//! - [`LayerNormFused`] (`CustomOp2`: `x`, `gamma`) — the forward. Its
//!   `bwd` does not compose ordinary `Tensor` ops (the way `Axpy`'s does);
//!   it dispatches into two more `KernelOp`s so the expensive per-element
//!   work (the `dx` recompute) is a genuine fused kernel on CUDA, not a
//!   handful of broadcasted candle ops:
//!   - `LayerNormBwdDx` (`CustomOp3`: `x`, `gamma`, `grad_output`) → `dx`.
//!     ONE kernel launch: recompute `mean`/`invvar` from `x`, the two
//!     per-row reduction scalars (Apex/ATen canonical), then `dx` — all in
//!     the same launch (a two-phase bwd would double LN backward launches
//!     post-fusion, which the fused-kernels plan explicitly rejects).
//!   - `LayerNormBwdDgamma` (`CustomOp2`: `x`, `grad_output`) → `dgamma`
//!     (shape `[hidden]`, summed over rows) — needs no `gamma` input at
//!     all (`dgamma_i = sum_rows(grad_output_i * xhat_i)`), only invoked
//!     when the call site says it needs to be.
//!
//! ## The `dgamma` skip: construction data, evaluated from `is_variable()`
//!
//! `LayerNormFused::dgamma_needed` is FROZEN INTO the `Copy` op instance
//! at construction, before `apply2` ever runs — the op itself still
//! never inspects any tensor's state at call time, so it stays exactly
//! as stateless as every other op in this crate (see `ops`'s module
//! doc's `Copy` discussion). What changed from this crate's original
//! design (and the fused-kernels plan's original wording, "construction
//! data, not a runtime predicate") is WHAT THE CALL SITE PASSES IN:
//! `jammi-encoders`' call site now sets it to `self.weight.is_variable()`,
//! re-evaluated on every call, rather than a single hardcoded `false`
//! chosen once and never revisited.
//!
//! That deviation is DELIBERATE and sound, not a relapse into the
//! `is_variable()` hazard `Axpy`'s doc warns about (`is_variable()`
//! cannot tell a true external constant apart from an INTERMEDIATE on a
//! path to a `Var`, and gating `bwd`'s OWN return value on it — the
//! design `Axpy` replaced — reproduces that chain-rule break). The
//! difference here is WHAT is being tested and WHEN: `dx`'s slot is
//! still ALWAYS `Some(dx)` regardless of `x`'s `is_variable()` status —
//! `LayerNormFused::bwd` never gates `dx` on anything, exactly like
//! `Axpy`. Only `gamma`'s slot uses `is_variable()`, and only because a
//! `LayerNorm`'s `gamma` is structurally a LEAF MODULE PARAMETER — loaded
//! straight from a `VarBuilder`, never produced by composing other
//! tensors — so the INTERMEDIATE-on-a-path-to-a-`Var` case that makes
//! `is_variable()` ambiguous for an arbitrary tensor simply cannot arise
//! for it. The only two real states left are "is a `Var`" (trainable)
//! and "is a true frozen leaf", and `is_variable() == true` is a
//! SUFFICIENT (not merely convenient) signal for the former. A first
//! version of this call site hardcoded `dgamma_needed = false`
//! reasoning "gamma is always frozen today, only LoRA A/B train" — sound
//! as of that commit, but a SILENT-WRONG landmine for the future: candle's
//! own backward walk skips accumulating into a `None` slot with no error
//! (`backprop.rs`'s `Op::CustomOp2` arm only calls `grads.or_insert` when
//! `bwd` returned `Some`), so a later trainable-gamma mode would have
//! silently never trained `gamma` — no panic, no error, just a parameter
//! an optimizer step quietly skips. Re-deriving the flag from
//! `is_variable()` on every call closes that hole structurally instead of
//! relying on whoever adds trainable-gamma support later to remember to
//! flip a hardcoded bool. Here there is no ambiguity for `dx`'s slot —
//! `LayerNormFused::bwd` ALWAYS returns `Some(dx)` for `x`, exactly like
//! `Axpy`'s bwd always returns `Some` for both slots, regardless of
//! whether `x` happens to be an intermediate. If `gamma` somehow WERE an
//! intermediate despite never being constructed that way, `is_variable()
//! == false` would make this op emit `None` for it, and if that turned
//! out to be the wrong call, candle's own backward walk panics loudly
//! (`grad not populated`, `backprop.rs:175`) rather than silently
//! training a grad-less parameter — a safe failure mode, not a
//! silent-wrong one.
//!
//! ## Domain (family D)
//!
//! `x` and `gamma` must be fully contiguous (`contiguous_offsets()`,
//! honoring a nonzero `start_offset` from a narrowed-but-contiguous view —
//! the same idiom `Axpy`'s CUDA arm uses, and for the same reason: a
//! raw-pointer kernel has no flat linear index for a strided view). This
//! is a real domain restriction (arbitrary strides are NOT walked here,
//! unlike `Axpy`'s CPU arm), deliberately: LayerNorm's per-row reduction
//! needs a well-defined `[rows, hidden]` grouping, and the actual call
//! site (encoder activations) is always contiguous — a non-contiguous
//! input is exactly the kind of case the call site's own admission check
//! (dtype/shape/contiguity/capability) is supposed to catch and fall back
//! on, not something this op should try to generalize into and risk
//! getting the row-grouping wrong for. `gamma` must be rank-1 with length
//! equal to `x`'s last dimension. CPU supports F32, BF16 (bias-free
//! LayerNorm's real training dtype), and F16; no F64 leg. F32/BF16 stay
//! device-uniform with CUDA (unlike `Axpy`, which is more generically
//! typed), since the profiled workload (ModernBERT-large, bf16) never
//! needs F64 here and keeping the domain device-uniform avoids a
//! CPU-passes/CUDA-refuses split with no oracle covering it. **F16 is a
//! disclosed, TEMPORARY exception to that device-uniformity rule**: the
//! CPU F16 arm (`ln_fwd_f16`/`ln_bwd_dx_f16`/`ln_bwd_dgamma_f16` below)
//! exists to serve as the independent-reference arm the f16 oracle suites
//! need (`docs/maintainer/cuda-kernel-guide.md`'s per-op f16
//! reference-regime table names this op's regime as f32-internal,
//! round-once, matching `jammi_encoders::layer_norm::LayerNorm::slow`'s
//! F16 upcast at `jammi-encoders/src/layer_norm.rs:353-370`); no CUDA F16
//! dispatch arm exists yet, and the admission predicate is not widened to
//! F16 until one does (K2's no-Hold-without-dispatch rule), so this
//! asymmetry is covered by an oracle (this op's own CPU tests below), not
//! a silent gap.
//! `hidden == 0` degenerates to an empty output (nothing to normalize) —
//! this makes `rows = elem_count / hidden` safe without a separate empty
//! fast-path, since `elem_count == 0` follows fromm `hidden == 0` (a
//! zero-length dimension implies zero elements) and the same is true in
//! reverse whenever the last dim is genuinely 0.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp2, CustomOp3, Error, Layout, Result, Shape, Tensor};
use half::{bf16, f16};

use super::empty_like;

/// The largest `hidden` (last-dim) size the CUDA kernels accept.
///
/// This is a conservative, VALIDATED ceiling — NOT a hardware constraint.
/// The block-per-row reduction's shared-memory scratch is `O(block_dim)`
/// floats (a few hundred bytes, `LN_BLOCK` wide — see
/// `cuda/layer_norm.cu`'s `block_reduce_sum`), not `O(hidden)`, and every
/// per-row pass is a grid-stride loop with no upper bound on `hidden` for
/// CORRECTNESS: a row ten times this size would still compute the right
/// answer, one kernel launch, same shared-memory footprint. The refusal
/// above `MAX_HIDDEN` exists because the kernel's numerics and
/// performance are validated up to here — ModernBERT-large's hidden=1024
/// with 8x headroom over the profiled workload — and no further; raising
/// it later is a matter of extending oracle coverage to larger rows, not
/// lifting a real ceiling this kernel design imposes. Enforced only on
/// the CUDA arm (`crate::cuda::layer_norm`). The CPU arm has no such
/// ceiling (see the module doc) but re-exports this constant so a call
/// site can apply ONE domain check that holds regardless of which device
/// a tensor happens to be on.
pub const MAX_HIDDEN: usize = 8192;

/// Bias-free LayerNorm forward. See the module doc for the full design.
#[derive(Debug, Clone, Copy)]
pub struct LayerNormFused {
    pub eps: f64,
    /// Whether `bwd` should compute and return `Some(dgamma)` for the
    /// `gamma` slot. Frozen into this `Copy` instance by the call site
    /// before `apply2` runs (the op itself never inspects any tensor's
    /// state) — see the module doc's "`dgamma` skip" section for why
    /// `jammi-encoders`' call site sets this to `gamma.is_variable()`
    /// re-evaluated per call, not a single hardcoded value, and why that
    /// is sound specifically for a leaf module parameter.
    pub dgamma_needed: bool,
}

impl LayerNormFused {
    pub fn new(eps: f64, dgamma_needed: bool) -> Self {
        Self { eps, dgamma_needed }
    }
}

impl super::sealed::Sealed for LayerNormFused {}

/// Checks that `l2` (gamma) is rank-1 with length equal to `l1`'s
/// (`x`'s) last dimension, returning that length ("hidden"). Shared by
/// every op in this file — `LayerNormFused`'s forward and both backward
/// helpers apply the identical `x`-vs-`gamma` shape rule. `pub(crate)`:
/// `crate::cuda::layer_norm` imports this exact check rather than
/// re-deriving it (the same "shared, not duplicated" choice
/// `ops::softmax::softmax_dims` and `ops::rope::rope_dims` make).
pub(crate) fn hidden_of(l1: &Layout, l2: &Layout, op: &'static str) -> Result<usize> {
    let dims = l1.dims();
    let hidden = *dims.last().ok_or_else(|| {
        Error::Msg(format!(
            "{op}: input must have rank >= 1 to define a last (hidden) dimension"
        ))
    })?;
    if l2.dims() != [hidden] {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l1.shape().clone(),
            rhs: l2.shape().clone(),
            op,
        });
    }
    Ok(hidden)
}

impl CustomOp2 for LayerNormFused {
    fn name(&self) -> &'static str {
        "layer_norm_fused"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let hidden = hidden_of(l1, l2, self.name())?;
        if hidden == 0 {
            return empty_like(s1, s2, l1, self.name());
        }
        let rows = l1.shape().elem_count() / hidden;
        let (o1, o2) = l1
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: self.name() })?;
        let (g1, g2) = l2
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: self.name() })?;
        match (s1, s2) {
            (CpuStorage::F32(x), CpuStorage::F32(g)) => {
                let out = ln_fwd_f32(&x[o1..o2], &g[g1..g2], rows, hidden, self.eps as f32);
                Ok((CpuStorage::F32(out), l1.shape().clone()))
            }
            (CpuStorage::BF16(x), CpuStorage::BF16(g)) => {
                let out = ln_fwd_bf16(&x[o1..o2], &g[g1..g2], rows, hidden, self.eps as f32);
                Ok((CpuStorage::BF16(out), l1.shape().clone()))
            }
            (CpuStorage::F16(x), CpuStorage::F16(g)) => {
                let out = ln_fwd_f16(&x[o1..o2], &g[g1..g2], rows, hidden, self.eps as f32);
                Ok((CpuStorage::F16(out), l1.shape().clone()))
            }
            (s1, s2) if s1.dtype() != s2.dtype() => Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: self.name(),
            }),
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
        crate::cuda::layer_norm::cuda_fwd(self.eps, s1, l1, s2, l2)
    }

    /// See the module doc's "no save-for-backward" and "construction
    /// data" sections. `dx`'s slot is ALWAYS `Some` — `x` may be an
    /// intermediate on a path to a `Var` (`Tensor::is_variable() ==
    /// false` does not mean "no gradient needed", see `Axpy`'s doc on
    /// this exact hazard) — only `gamma`'s slot is ever `None`, and only
    /// when `self.dgamma_needed` says so.
    fn bwd(
        &self,
        arg1: &Tensor,
        arg2: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let dx = super::apply3(arg1, arg2, grad_res, LayerNormBwdDx { eps: self.eps })?;
        let dgamma = if self.dgamma_needed {
            Some(super::apply2(
                arg1,
                grad_res,
                LayerNormBwdDgamma { eps: self.eps },
            )?)
        } else {
            None
        };
        Ok((Some(dx), dgamma))
    }
}

/// `LayerNormFused`'s internal backward helper producing `dx`. Not
/// exported — only ever invoked from [`LayerNormFused::bwd`] via
/// [`super::apply3`]. See the module doc for why this is a `CustomOp3`
/// (needs `x`, `gamma`, AND `grad_output`) rather than composed from
/// ordinary `Tensor` ops.
#[derive(Debug, Clone, Copy)]
struct LayerNormBwdDx {
    eps: f64,
}

impl super::sealed::Sealed for LayerNormBwdDx {}

impl CustomOp3 for LayerNormBwdDx {
    fn name(&self) -> &'static str {
        "layer_norm_fused_bwd_dx"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let hidden = hidden_of(l1, l2, self.name())?;
        if l3.dims() != l1.dims() {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: l1.shape().clone(),
                rhs: l3.shape().clone(),
                op: self.name(),
            });
        }
        if s1.dtype() != s3.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s3.dtype(),
                op: self.name(),
            });
        }
        if hidden == 0 {
            return empty_like(s1, s2, l1, self.name());
        }
        let rows = l1.shape().elem_count() / hidden;
        let (o1, o2) = l1
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: self.name() })?;
        let (g1, g2) = l2
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: self.name() })?;
        let (d1, d2) = l3
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: self.name() })?;
        match (s1, s2, s3) {
            (CpuStorage::F32(x), CpuStorage::F32(g), CpuStorage::F32(dy)) => {
                let dx = ln_bwd_dx_f32(
                    &x[o1..o2],
                    &g[g1..g2],
                    &dy[d1..d2],
                    rows,
                    hidden,
                    self.eps as f32,
                );
                Ok((CpuStorage::F32(dx), l1.shape().clone()))
            }
            (CpuStorage::BF16(x), CpuStorage::BF16(g), CpuStorage::BF16(dy)) => {
                let dx = ln_bwd_dx_bf16(
                    &x[o1..o2],
                    &g[g1..g2],
                    &dy[d1..d2],
                    rows,
                    hidden,
                    self.eps as f32,
                );
                Ok((CpuStorage::BF16(dx), l1.shape().clone()))
            }
            (CpuStorage::F16(x), CpuStorage::F16(g), CpuStorage::F16(dy)) => {
                let dx = ln_bwd_dx_f16(
                    &x[o1..o2],
                    &g[g1..g2],
                    &dy[d1..d2],
                    rows,
                    hidden,
                    self.eps as f32,
                );
                Ok((CpuStorage::F16(dx), l1.shape().clone()))
            }
            (s1, s2, _) if s1.dtype() != s2.dtype() => Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: self.name(),
            }),
            (s1, _, _) => Err(Error::UnsupportedDTypeForOp(s1.dtype(), self.name())),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
        s3: &candle_core::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        crate::cuda::layer_norm::cuda_bwd_dx(self.eps, s1, l1, s2, l2, s3, l3)
    }

    // No `bwd` override: this helper's own gradient (a second-order
    // derivative of the fused LayerNorm) is never requested by any call
    // site in this crate or its consumers — the default `CustomOp3::bwd`
    // (`Err(BackwardNotSupported)`) is the correct refusal if anything
    // ever tried.
}

/// `LayerNormFused`'s internal backward helper producing `dgamma`. Not
/// exported — only ever invoked from [`LayerNormFused::bwd`], and only
/// when `dgamma_needed` is true. Needs no `gamma` input at all:
/// `dgamma_i = sum_rows(grad_output_i * xhat_i)`, and `xhat` is
/// recomputed from `x` alone.
#[derive(Debug, Clone, Copy)]
struct LayerNormBwdDgamma {
    eps: f64,
}

impl super::sealed::Sealed for LayerNormBwdDgamma {}

impl CustomOp2 for LayerNormBwdDgamma {
    fn name(&self) -> &'static str {
        "layer_norm_fused_bwd_dgamma"
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
        let dims = l1.dims();
        let hidden = *dims.last().ok_or_else(|| {
            Error::Msg(format!(
                "{}: input must have rank >= 1 to define a last (hidden) dimension",
                self.name()
            ))
        })?;
        if hidden == 0 {
            return match (s1, s2) {
                (CpuStorage::F32(_), CpuStorage::F32(_)) => {
                    Ok((CpuStorage::F32(Vec::new()), Shape::from(0usize)))
                }
                (CpuStorage::BF16(_), CpuStorage::BF16(_)) => {
                    Ok((CpuStorage::BF16(Vec::new()), Shape::from(0usize)))
                }
                (CpuStorage::F16(_), CpuStorage::F16(_)) => {
                    Ok((CpuStorage::F16(Vec::new()), Shape::from(0usize)))
                }
                (s1, s2) if s1.dtype() != s2.dtype() => Err(Error::DTypeMismatchBinaryOp {
                    lhs: s1.dtype(),
                    rhs: s2.dtype(),
                    op: self.name(),
                }),
                (s1, _) => Err(Error::UnsupportedDTypeForOp(s1.dtype(), self.name())),
            };
        }
        let rows = l1.shape().elem_count() / hidden;
        let (o1, o2) = l1
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: self.name() })?;
        let (d1, d2) = l2
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: self.name() })?;
        match (s1, s2) {
            (CpuStorage::F32(x), CpuStorage::F32(dy)) => {
                let dg = ln_bwd_dgamma_f32(&x[o1..o2], &dy[d1..d2], rows, hidden, self.eps as f32);
                Ok((CpuStorage::F32(dg), Shape::from(hidden)))
            }
            (CpuStorage::BF16(x), CpuStorage::BF16(dy)) => {
                let dg = ln_bwd_dgamma_bf16(&x[o1..o2], &dy[d1..d2], rows, hidden, self.eps as f32);
                Ok((CpuStorage::BF16(dg), Shape::from(hidden)))
            }
            (CpuStorage::F16(x), CpuStorage::F16(dy)) => {
                let dg = ln_bwd_dgamma_f16(&x[o1..o2], &dy[d1..d2], rows, hidden, self.eps as f32);
                Ok((CpuStorage::F16(dg), Shape::from(hidden)))
            }
            (s1, s2) if s1.dtype() != s2.dtype() => Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: self.name(),
            }),
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
        crate::cuda::layer_norm::cuda_bwd_dgamma(self.eps, s1, l1, s2, l2)
    }

    // No `bwd` override — see `LayerNormBwdDx`'s identical note.
}

// -----------------------------------------------------------------------
// CPU math. Fixed fold order throughout (family J): every reduction below
// walks its row in plain ascending index order, so a given `(x, gamma)`
// (or `(x, gamma, dy)`) pair always yields the same output bit-for-bit —
// no parallel/unordered accumulation on this path.
// -----------------------------------------------------------------------

fn mean_var_f32(row: &[f32], hidden: usize) -> (f32, f32) {
    let mut sum = 0f32;
    for &v in row {
        sum += v;
    }
    let mean = sum / hidden as f32;
    let mut sumsq = 0f32;
    for &v in row {
        let d = v - mean;
        sumsq += d * d;
    }
    (mean, sumsq / hidden as f32)
}

/// BF16 row mean/variance, accumulated in f32 — same two-pass, ascending-
/// index fold order as [`mean_var_f32`] (family J: one fixed reduction
/// order, not "whatever the loop happened to do"), so every bf16 call site
/// below (`ln_fwd_row_bf16`, `ln_bwd_dx_row_bf16`, `ln_bwd_dgamma_bf16`)
/// computes the identical numeric sequence the CUDA kernel's own
/// f32-accumulate row-stats pass does — the bf16-CPU-vs-CUDA parity
/// contract the layer_norm oracle suite checks byte-for-byte.
fn mean_var_bf16(row: &[bf16], hidden: usize) -> (f32, f32) {
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
    (mean, sumsq / hidden as f32)
}

fn ln_fwd_row_f32(x: &[f32], gamma: &[f32], eps: f32, out: &mut [f32]) {
    let hidden = x.len();
    let (mean, var) = mean_var_f32(x, hidden);
    let invvar = 1.0 / (var + eps).sqrt();
    for i in 0..hidden {
        let xhat = (x[i] - mean) * invvar;
        out[i] = xhat * gamma[i];
    }
}

fn ln_fwd_f32(x: &[f32], gamma: &[f32], rows: usize, hidden: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; rows * hidden];
    for r in 0..rows {
        let lo = r * hidden;
        let hi = lo + hidden;
        ln_fwd_row_f32(&x[lo..hi], gamma, eps, &mut out[lo..hi]);
    }
    out
}

/// BF16 accumulates in f32 (mean/var/xhat), rounding to bf16 exactly once
/// on the way out — the same accumulation semantics documented on
/// `ops::axpy`'s BF16 arm, and what the CUDA kernel does too.
fn ln_fwd_row_bf16(x: &[bf16], gamma: &[bf16], eps: f32, out: &mut [bf16]) {
    let hidden = x.len();
    let (mean, var) = mean_var_bf16(x, hidden);
    let invvar = 1.0 / (var + eps).sqrt();
    for i in 0..hidden {
        let xhat = (x[i].to_f32() - mean) * invvar;
        out[i] = bf16::from_f32(xhat * gamma[i].to_f32());
    }
}

fn ln_fwd_bf16(x: &[bf16], gamma: &[bf16], rows: usize, hidden: usize, eps: f32) -> Vec<bf16> {
    let mut out = vec![bf16::ZERO; rows * hidden];
    for r in 0..rows {
        let lo = r * hidden;
        let hi = lo + hidden;
        ln_fwd_row_bf16(&x[lo..hi], gamma, eps, &mut out[lo..hi]);
    }
    out
}

/// F16 row mean/variance, accumulated in f32 — [`mean_var_bf16`]'s exact
/// twin, substituting `half::f16`. This op's f32-internal regime (see the
/// module doc and `docs/maintainer/cuda-kernel-guide.md`'s per-op table)
/// matches `jammi_encoders::layer_norm::LayerNorm::slow`'s F16 upcast.
fn mean_var_f16(row: &[f16], hidden: usize) -> (f32, f32) {
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
    (mean, sumsq / hidden as f32)
}

/// F32-accumulate, round-to-f16 once — [`ln_fwd_row_bf16`]'s exact twin.
fn ln_fwd_row_f16(x: &[f16], gamma: &[f16], eps: f32, out: &mut [f16]) {
    let hidden = x.len();
    let (mean, var) = mean_var_f16(x, hidden);
    let invvar = 1.0 / (var + eps).sqrt();
    for i in 0..hidden {
        let xhat = (x[i].to_f32() - mean) * invvar;
        out[i] = f16::from_f32(xhat * gamma[i].to_f32());
    }
}

fn ln_fwd_f16(x: &[f16], gamma: &[f16], rows: usize, hidden: usize, eps: f32) -> Vec<f16> {
    let mut out = vec![f16::ZERO; rows * hidden];
    for r in 0..rows {
        let lo = r * hidden;
        let hi = lo + hidden;
        ln_fwd_row_f16(&x[lo..hi], gamma, eps, &mut out[lo..hi]);
    }
    out
}

/// The Apex/ATen canonical `dx`, computed row-by-row in f32:
/// `xhat = (x-mean)*invvar`; `t = dy*gamma`;
/// `dx = (t - mean_row(t) - xhat * mean_row(t*xhat)) * invvar`.
fn ln_bwd_dx_row_f32(x: &[f32], gamma: &[f32], dy: &[f32], eps: f32, dx: &mut [f32]) {
    let hidden = x.len();
    let (mean, var) = mean_var_f32(x, hidden);
    let invvar = 1.0 / (var + eps).sqrt();

    let mut sum_t = 0f32;
    let mut sum_t_xhat = 0f32;
    for i in 0..hidden {
        let xhat = (x[i] - mean) * invvar;
        let t = dy[i] * gamma[i];
        sum_t += t;
        sum_t_xhat += t * xhat;
    }
    let mean_t = sum_t / hidden as f32;
    let mean_t_xhat = sum_t_xhat / hidden as f32;

    for i in 0..hidden {
        let xhat = (x[i] - mean) * invvar;
        let t = dy[i] * gamma[i];
        dx[i] = (t - mean_t - xhat * mean_t_xhat) * invvar;
    }
}

fn ln_bwd_dx_f32(
    x: &[f32],
    gamma: &[f32],
    dy: &[f32],
    rows: usize,
    hidden: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; rows * hidden];
    for r in 0..rows {
        let lo = r * hidden;
        let hi = lo + hidden;
        ln_bwd_dx_row_f32(&x[lo..hi], gamma, &dy[lo..hi], eps, &mut out[lo..hi]);
    }
    out
}

fn ln_bwd_dx_row_bf16(x: &[bf16], gamma: &[bf16], dy: &[bf16], eps: f32, dx: &mut [bf16]) {
    let hidden = x.len();
    let (mean, var) = mean_var_bf16(x, hidden);
    let invvar = 1.0 / (var + eps).sqrt();

    let mut sum_t = 0f32;
    let mut sum_t_xhat = 0f32;
    for i in 0..hidden {
        let xhat = (x[i].to_f32() - mean) * invvar;
        let t = dy[i].to_f32() * gamma[i].to_f32();
        sum_t += t;
        sum_t_xhat += t * xhat;
    }
    let mean_t = sum_t / hidden as f32;
    let mean_t_xhat = sum_t_xhat / hidden as f32;

    for i in 0..hidden {
        let xhat = (x[i].to_f32() - mean) * invvar;
        let t = dy[i].to_f32() * gamma[i].to_f32();
        dx[i] = bf16::from_f32((t - mean_t - xhat * mean_t_xhat) * invvar);
    }
}

fn ln_bwd_dx_bf16(
    x: &[bf16],
    gamma: &[bf16],
    dy: &[bf16],
    rows: usize,
    hidden: usize,
    eps: f32,
) -> Vec<bf16> {
    let mut out = vec![bf16::ZERO; rows * hidden];
    for r in 0..rows {
        let lo = r * hidden;
        let hi = lo + hidden;
        ln_bwd_dx_row_bf16(&x[lo..hi], gamma, &dy[lo..hi], eps, &mut out[lo..hi]);
    }
    out
}

/// [`ln_bwd_dx_row_bf16`]'s exact twin, substituting `half::f16`.
fn ln_bwd_dx_row_f16(x: &[f16], gamma: &[f16], dy: &[f16], eps: f32, dx: &mut [f16]) {
    let hidden = x.len();
    let (mean, var) = mean_var_f16(x, hidden);
    let invvar = 1.0 / (var + eps).sqrt();

    let mut sum_t = 0f32;
    let mut sum_t_xhat = 0f32;
    for i in 0..hidden {
        let xhat = (x[i].to_f32() - mean) * invvar;
        let t = dy[i].to_f32() * gamma[i].to_f32();
        sum_t += t;
        sum_t_xhat += t * xhat;
    }
    let mean_t = sum_t / hidden as f32;
    let mean_t_xhat = sum_t_xhat / hidden as f32;

    for i in 0..hidden {
        let xhat = (x[i].to_f32() - mean) * invvar;
        let t = dy[i].to_f32() * gamma[i].to_f32();
        dx[i] = f16::from_f32((t - mean_t - xhat * mean_t_xhat) * invvar);
    }
}

fn ln_bwd_dx_f16(
    x: &[f16],
    gamma: &[f16],
    dy: &[f16],
    rows: usize,
    hidden: usize,
    eps: f32,
) -> Vec<f16> {
    let mut out = vec![f16::ZERO; rows * hidden];
    for r in 0..rows {
        let lo = r * hidden;
        let hi = lo + hidden;
        ln_bwd_dx_row_f16(&x[lo..hi], gamma, &dy[lo..hi], eps, &mut out[lo..hi]);
    }
    out
}

/// `dgamma_i = sum_rows(dy_i * xhat_i)` — fixed fold order: rows walked
/// `0..rows` in ascending order, accumulating into `dgamma[i]` each time
/// (family J: the same input always folds in the same order).
fn ln_bwd_dgamma_f32(x: &[f32], dy: &[f32], rows: usize, hidden: usize, eps: f32) -> Vec<f32> {
    let mut dgamma = vec![0f32; hidden];
    for r in 0..rows {
        let lo = r * hidden;
        let hi = lo + hidden;
        let xr = &x[lo..hi];
        let dyr = &dy[lo..hi];
        let (mean, var) = mean_var_f32(xr, hidden);
        let invvar = 1.0 / (var + eps).sqrt();
        for i in 0..hidden {
            let xhat = (xr[i] - mean) * invvar;
            dgamma[i] += dyr[i] * xhat;
        }
    }
    dgamma
}

fn ln_bwd_dgamma_bf16(x: &[bf16], dy: &[bf16], rows: usize, hidden: usize, eps: f32) -> Vec<bf16> {
    let mut acc = vec![0f32; hidden];
    for r in 0..rows {
        let lo = r * hidden;
        let hi = lo + hidden;
        let xr = &x[lo..hi];
        let dyr = &dy[lo..hi];
        let (mean, var) = mean_var_bf16(xr, hidden);
        let invvar = 1.0 / (var + eps).sqrt();
        for i in 0..hidden {
            let xhat = (xr[i].to_f32() - mean) * invvar;
            acc[i] += dyr[i].to_f32() * xhat;
        }
    }
    // One rounding to bf16 at the very end, matching the crate's
    // f32-accumulate-round-once convention.
    acc.into_iter().map(bf16::from_f32).collect()
}

/// [`ln_bwd_dgamma_bf16`]'s exact twin, substituting `half::f16`.
fn ln_bwd_dgamma_f16(x: &[f16], dy: &[f16], rows: usize, hidden: usize, eps: f32) -> Vec<f16> {
    let mut acc = vec![0f32; hidden];
    for r in 0..rows {
        let lo = r * hidden;
        let hi = lo + hidden;
        let xr = &x[lo..hi];
        let dyr = &dy[lo..hi];
        let (mean, var) = mean_var_f16(xr, hidden);
        let invvar = 1.0 / (var + eps).sqrt();
        for i in 0..hidden {
            let xhat = (xr[i].to_f32() - mean) * invvar;
            acc[i] += dyr[i].to_f32() * xhat;
        }
    }
    acc.into_iter().map(f16::from_f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn ln(eps: f64, dgamma_needed: bool, x: &Tensor, gamma: &Tensor) -> Result<Tensor> {
        crate::ops::apply2(x, gamma, LayerNormFused::new(eps, dgamma_needed))
    }

    /// Hand-computed reference for a single row: `hidden = 4`,
    /// `x = [1, 2, 3, 4]` -> `mean = 2.5`, `var = 1.25`.
    #[test]
    fn cpu_fwd_f32_matches_hand_computed_values() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let gamma = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], (4,), &device).unwrap();
        let out = ln(1e-5, false, &x, &gamma)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let mean = 2.5f32;
        let var = 1.25f32;
        let invvar = 1.0 / (var + 1e-5f32).sqrt();
        let expected: Vec<f32> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .map(|&v| (v - mean) * invvar)
            .collect();
        for (o, e) in out[0].iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-5, "{o} vs {e}");
        }
    }

    #[test]
    fn gamma_scales_the_normalized_output() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let gamma = Tensor::from_slice(&[2.0f32, 2.0, 2.0, 2.0], (4,), &device).unwrap();
        let out_scaled = ln(1e-5, false, &x, &gamma)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let ones = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], (4,), &device).unwrap();
        let out_unit = ln(1e-5, false, &x, &ones)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for (s, u) in out_scaled[0].iter().zip(out_unit[0].iter()) {
            assert!((s - 2.0 * u).abs() < 1e-5);
        }
    }

    #[test]
    fn empty_batch_is_a_no_op_not_an_error() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[] as &[f32], (0, 4), &device).unwrap();
        let gamma = Tensor::from_slice(&[1.0f32; 4], (4,), &device).unwrap();
        let out = ln(1e-5, false, &x, &gamma)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn zero_hidden_is_a_no_op_not_a_division_by_zero_panic() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[] as &[f32], (3, 0), &device).unwrap();
        let gamma = Tensor::from_slice(&[] as &[f32], (0,), &device).unwrap();
        let out = ln(1e-5, false, &x, &gamma)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|row| row.is_empty()));
    }

    #[test]
    fn gamma_shape_mismatch_is_refused_not_broadcast() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let gamma = Tensor::from_slice(&[1.0f32, 1.0, 1.0], (3,), &device).unwrap();
        let err = ln(1e-5, false, &x, &gamma).expect_err("hidden-size mismatch must be refused");
        assert!(matches!(err, Error::ShapeMismatchBinaryOp { .. }));
    }

    #[test]
    fn dtype_mismatch_between_x_and_gamma_is_refused() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let gamma = Tensor::from_slice(&[bf16::from_f32(1.0); 4], (4,), &device).unwrap();
        let err = ln(1e-5, false, &x, &gamma).expect_err("dtype mismatch must be refused");
        assert!(matches!(err, Error::DTypeMismatchBinaryOp { .. }));
    }

    #[test]
    fn non_contiguous_x_is_refused_not_silently_misread() {
        let device = Device::Cpu;
        // A transposed [4, 2] -> [2, 4] view: contiguous along the WRONG
        // axis for row-major [rows, hidden] grouping.
        let x = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            (4, 2),
            &device,
        )
        .unwrap()
        .t()
        .unwrap();
        assert!(!x.is_contiguous());
        let gamma = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], (4,), &device).unwrap();
        let err = ln(1e-5, false, &x, &gamma).expect_err("non-contiguous x must be refused");
        assert!(matches!(err, Error::RequiresContiguous { .. }));
    }

    #[test]
    fn bf16_forward_matches_f32_accumulation_rounded_once() {
        let device = Device::Cpu;
        let xv = [1.0f32, 2.0, 3.0, 4.0];
        let xb: Vec<bf16> = xv.iter().map(|&v| bf16::from_f32(v)).collect();
        let gb = [bf16::from_f32(1.0); 4];
        let x = Tensor::from_slice(&xb, (1, 4), &device).unwrap();
        let gamma = Tensor::from_slice(&gb, (4,), &device).unwrap();
        let out: Vec<bf16> = ln(1e-5, false, &x, &gamma)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Independently-computed reference in f64, matching the op's
        // documented f32-accumulate-round-once semantics: mean/var over
        // the bf16-rounded inputs, one final rounding to bf16.
        let xf: Vec<f64> = xb.iter().map(|v| v.to_f32() as f64).collect();
        let mean: f64 = xf.iter().sum::<f64>() / xf.len() as f64;
        let var: f64 = xf.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / xf.len() as f64;
        let invvar = 1.0 / (var + 1e-5).sqrt();
        let expected: Vec<f32> = xf.iter().map(|&v| ((v - mean) * invvar) as f32).collect();

        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o.to_f32() - e).abs() < 1e-2, "{o} vs {e}");
        }
    }

    /// F16 forward's exact twin of `bf16_forward_matches_f32_accumulation_
    /// rounded_once` above — the independent, higher-precision (f64)
    /// reference this op's own module doc and the per-op f16
    /// reference-regime table (`docs/maintainer/cuda-kernel-guide.md`)
    /// pin: mean/var over the f16-rounded inputs, one final rounding to
    /// f16.
    #[test]
    fn f16_forward_matches_f32_accumulation_rounded_once() {
        use half::f16;
        let device = Device::Cpu;
        let xv = [1.0f32, 2.0, 3.0, 4.0];
        let xh: Vec<f16> = xv.iter().map(|&v| f16::from_f32(v)).collect();
        let gh = [f16::from_f32(1.0); 4];
        let x = Tensor::from_slice(&xh, (1, 4), &device).unwrap();
        let gamma = Tensor::from_slice(&gh, (4,), &device).unwrap();
        let out: Vec<f16> = ln(1e-5, false, &x, &gamma)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let xf: Vec<f64> = xh.iter().map(|v| v.to_f32() as f64).collect();
        let mean: f64 = xf.iter().sum::<f64>() / xf.len() as f64;
        let var: f64 = xf.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / xf.len() as f64;
        let invvar = 1.0 / (var + 1e-5).sqrt();
        let expected: Vec<f32> = xf.iter().map(|&v| ((v - mean) * invvar) as f32).collect();

        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o.to_f32() - e).abs() < 1e-2, "{o} vs {e}");
        }
    }

    // -----------------------------------------------------------------
    // `LayerNormFused::cpu_fwd`'s dtype-mismatch guard (its `(s1, s2) if
    // s1.dtype() != s2.dtype()` match arm), the SAME-unsupported-dtype
    // cell (MUT-1: that guard forced `true` survived). The "!=" false-arm (a real
    // mismatch, e.g. F32 vs BF16) is already `dtype_mismatch_between_
    // x_and_gamma_is_refused` above; that test alone does not kill the
    // guard-forced-`true` mutant, because BOTH the real code and the
    // `true` mutant return `DTypeMismatchBinaryOp` on a real mismatch.
    // Only a SAME (equal), UNSUPPORTED dtype on both operands (F64 here
    // — not F32, not BF16, not F16: this crate's D2 f16-oracle work added
    // a real F16 CPU arm, so F16 stopped being a same-unsupported-dtype
    // witness and F64 replaces it, verified to still fall through to the
    // same `_ => UnsupportedDTypeForOp` arm) tells them apart: real code
    // falls through to `UnsupportedDTypeForOp`; the `true` mutant reports
    // `DTypeMismatchBinaryOp` instead even though the two dtypes agree.
    // -----------------------------------------------------------------
    #[test]
    fn same_unsupported_dtype_is_unsupported_not_a_false_mismatch() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f64, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let gamma = Tensor::from_slice(&[1.0f64; 4], (4,), &device).unwrap();
        let err = ln(1e-5, false, &x, &gamma)
            .expect_err("F64/F64 is a real dtype (equal on both sides) this op does not implement");
        match err {
            Error::UnsupportedDTypeForOp(dtype, _) => assert_eq!(dtype, DType::F64),
            other => panic!("expected UnsupportedDTypeForOp(F64, _), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // `LayerNormBwdDx::cpu_fwd`'s own dtype guard (`s1` = x, `s2` =
    // gamma; MUT-1 mutations of its `(s1, s2, _) if s1.dtype() !=
    // s2.dtype()` match arm: guard forced `true`, forced `false`, and
    // `!=` -> `==`). No existing test calls this
    // internal helper directly (it is normally reached only through
    // `LayerNormFused::bwd`, which candle's autograd invokes with
    // matching dtypes by construction) — both cells below are new.
    // -----------------------------------------------------------------
    fn ln_bwd_dx(eps: f64, x: &Tensor, gamma: &Tensor, dy: &Tensor) -> Result<Tensor> {
        crate::ops::apply3(x, gamma, dy, LayerNormBwdDx { eps })
    }

    /// Real mismatch cell (x != gamma dtype, dy matches x so
    /// `LayerNormBwdDx::cpu_fwd`'s earlier `if s1.dtype() != s3.dtype()`
    /// check is not what fires here):
    /// kills the guard-forced-`false` and `!=`->`==` mutants, which would
    /// otherwise report `UnsupportedDTypeForOp` instead of the correct
    /// `DTypeMismatchBinaryOp{lhs, rhs}`.
    #[test]
    fn bwd_dx_dtype_mismatch_between_x_and_gamma_is_refused() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let gamma = Tensor::from_slice(&[bf16::from_f32(1.0); 4], (4,), &device).unwrap();
        let dy = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], (1, 4), &device).unwrap();
        let err =
            ln_bwd_dx(1e-5, &x, &gamma, &dy).expect_err("x/gamma dtype mismatch must be refused");
        match err {
            Error::DTypeMismatchBinaryOp { lhs, rhs, .. } => {
                assert_eq!(lhs, DType::F32);
                assert_eq!(rhs, DType::BF16);
            }
            other => panic!("expected DTypeMismatchBinaryOp{{F32, BF16}}, got {other:?}"),
        }
    }

    /// Same-unsupported-dtype cell (x == gamma == dy == F64, per the same
    /// F16-now-real-arm retarget as `same_unsupported_dtype_is_unsupported_
    /// not_a_false_mismatch` above): kills the guard-forced-`true` mutant,
    /// which would otherwise report `DTypeMismatchBinaryOp` for two
    /// operands that actually agree.
    #[test]
    fn bwd_dx_same_unsupported_dtype_is_unsupported_not_a_false_mismatch() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f64, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let gamma = Tensor::from_slice(&[1.0f64; 4], (4,), &device).unwrap();
        let dy = Tensor::from_slice(&[1.0f64; 4], (1, 4), &device).unwrap();
        let err = ln_bwd_dx(1e-5, &x, &gamma, &dy)
            .expect_err("F64/F64/F64 is a real, equal-dtype triple this op does not implement");
        match err {
            Error::UnsupportedDTypeForOp(dtype, _) => assert_eq!(dtype, DType::F64),
            other => panic!("expected UnsupportedDTypeForOp(F64, _), got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // `LayerNormBwdDgamma::cpu_fwd` has TWO copies of this same guard —
    // the `(s1, s2) if s1.dtype() != s2.dtype()` arm of the `hidden == 0`
    // fast path's `match (s1, s2)` and the identical arm of the general
    // `hidden > 0` path's `match (s1, s2)` — each with its own `true`/
    // `false`/`!=`->`==` survivors. Both cells (mismatch, same-
    // unsupported) are needed at BOTH hidden==0 and hidden>0, four
    // tests total.
    // -----------------------------------------------------------------
    fn ln_bwd_dgamma(eps: f64, x: &Tensor, dy: &Tensor) -> Result<Tensor> {
        crate::ops::apply2(x, dy, LayerNormBwdDgamma { eps })
    }

    #[test]
    fn bwd_dgamma_hidden_zero_dtype_mismatch_is_refused() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[] as &[f32], (3, 0), &device).unwrap();
        let dy = Tensor::from_slice(&[] as &[bf16], (3, 0), &device).unwrap();
        let err = ln_bwd_dgamma(1e-5, &x, &dy).expect_err("x/dy dtype mismatch must be refused");
        match err {
            Error::DTypeMismatchBinaryOp { lhs, rhs, .. } => {
                assert_eq!(lhs, DType::F32);
                assert_eq!(rhs, DType::BF16);
            }
            other => panic!("expected DTypeMismatchBinaryOp{{F32, BF16}}, got {other:?}"),
        }
    }

    #[test]
    fn bwd_dgamma_hidden_zero_same_unsupported_dtype_is_unsupported_not_a_false_mismatch() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[] as &[f64], (3, 0), &device).unwrap();
        let dy = Tensor::from_slice(&[] as &[f64], (3, 0), &device).unwrap();
        let err = ln_bwd_dgamma(1e-5, &x, &dy)
            .expect_err("F64/F64 (equal, hidden == 0) is not implemented by this op");
        match err {
            Error::UnsupportedDTypeForOp(dtype, _) => assert_eq!(dtype, DType::F64),
            other => panic!("expected UnsupportedDTypeForOp(F64, _), got {other:?}"),
        }
    }

    #[test]
    fn bwd_dgamma_hidden_nonzero_dtype_mismatch_is_refused() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let dy = Tensor::from_slice(&[bf16::from_f32(1.0); 4], (1, 4), &device).unwrap();
        let err = ln_bwd_dgamma(1e-5, &x, &dy).expect_err("x/dy dtype mismatch must be refused");
        match err {
            Error::DTypeMismatchBinaryOp { lhs, rhs, .. } => {
                assert_eq!(lhs, DType::F32);
                assert_eq!(rhs, DType::BF16);
            }
            other => panic!("expected DTypeMismatchBinaryOp{{F32, BF16}}, got {other:?}"),
        }
    }

    #[test]
    fn bwd_dgamma_hidden_nonzero_same_unsupported_dtype_is_unsupported_not_a_false_mismatch() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f64, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let dy = Tensor::from_slice(&[1.0f64; 4], (1, 4), &device).unwrap();
        let err = ln_bwd_dgamma(1e-5, &x, &dy)
            .expect_err("F64/F64 (equal, hidden > 0) is not implemented by this op");
        match err {
            Error::UnsupportedDTypeForOp(dtype, _) => assert_eq!(dtype, DType::F64),
            other => panic!("expected UnsupportedDTypeForOp(F64, _), got {other:?}"),
        }
    }

    /// `rows == 0, hidden != 0` (an empty batch over a non-empty hidden
    /// dim) is a DIFFERENT domain point than `hidden == 0`: `cpu_fwd`'s
    /// `hidden == 0` early-return is not taken here (`hidden` is `4`), so
    /// this fixture instead falls through to `ln_bwd_dgamma_f32` with
    /// `rows = 0` — `vec![0f32; hidden]`, all-zero and `[hidden]`-shaped,
    /// NOT the `[0]`-shaped output the `hidden == 0` branch returns.
    /// Pinning this shape+content here is what
    /// `crate::cuda::layer_norm::cuda_bwd_dgamma`'s `n == 0` fast path
    /// (the CUDA glue's OWN parallel branch) is checked against — see
    /// this crate's `tests/empty_non_contiguous_admission_class_oracle.rs`
    /// for the CUDA-gated twin over a real device.
    #[test]
    fn bwd_dgamma_zero_rows_hidden_nonzero_is_hidden_shaped_all_zero_not_zero_length() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[] as &[f32], (0, 4), &device).unwrap();
        let dy = Tensor::from_slice(&[] as &[f32], (0, 4), &device).unwrap();
        let out = ln_bwd_dgamma(1e-5, &x, &dy).unwrap();
        assert_eq!(
            out.dims(),
            &[4],
            "rows == 0, hidden != 0 must still be [hidden]-shaped, not [0]-shaped"
        );
        let dgamma: Vec<f32> = out.to_vec1().unwrap();
        assert!(
            dgamma.iter().all(|&d| d == 0.0),
            "summing zero rows must give exactly zero at every position, got {dgamma:?}"
        );
    }

    // -----------------------------------------------------------------
    // The four `+`->`-` eps-sign-flip survivors in the row kernels
    // (`ln_fwd_row_bf16`, `ln_bwd_dx_row_f32`, `ln_bwd_dgamma_f32`,
    // `ln_bwd_dgamma_bf16`), all of the shape `invvar = 1.0 / (var +
    // eps).sqrt()`. Rule 2 of the contract
    // (tolerance failures): derive a bound so tight the flip cannot hide
    // inside it, rather than shrinking `eps` relative to a fixed abs
    // tolerance.
    //
    // The fixture below makes the bound EXACT (zero slack), not merely
    // tight: every element of a row is the SAME value, so `var` is
    // EXACTLY `0.0` in f32 (mean == every element, so `x[i] - mean ==
    // 0.0` exactly, and `0.0 * invvar == 0.0` for ANY finite `invvar`,
    // regardless of its magnitude). That makes the correct output
    // EXACTLY zero — a real, sign-independent guarantee, since `var +
    // eps = eps > 0` is always a valid domain point for `sqrt`.
    //
    // The `+`->`-` mutation instead computes `sqrt(var - eps) =
    // sqrt(-eps)`, which is NaN for any `eps > 0` in this degenerate-
    // variance regime — `bound/max|signal| = 0/0`: not "within
    // tolerance", but a hard finite-vs-non-finite divergence (family F:
    // every element of a `0.0 * NaN = NaN` product), the strongest
    // possible measurement of an eps-sign flip. A non-degenerate `eps` (
    // 1e-2, far from f32's own ULP at this magnitude) rules out this
    // being a rounding-noise artifact.
    // -----------------------------------------------------------------
    #[test]
    fn constant_row_forward_is_exactly_zero_pinning_the_eps_sign_bf16() {
        let device = Device::Cpu;
        let xb = [bf16::from_f32(3.0); 4];
        let gb = [bf16::from_f32(1.0); 4];
        let x = Tensor::from_slice(&xb, (1, 4), &device).unwrap();
        let gamma = Tensor::from_slice(&gb, (4,), &device).unwrap();
        let out: Vec<bf16> = ln(1e-2, false, &x, &gamma)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for o in &out {
            assert_eq!(
                o.to_f32(),
                0.0,
                "a constant row must normalize to EXACTLY zero regardless of eps's \
                 magnitude — a non-finite value here means `var + eps` was computed as \
                 `var - eps` (sqrt of a negative number)"
            );
        }
    }

    #[test]
    fn constant_row_backward_dx_is_exactly_zero_pinning_the_eps_sign_f32() {
        // `dy`/`gamma` also held CONSTANT across the row: `t = dy*gamma`
        // is then the same value at every position, so `mean_t == t` and
        // `dx[i] = (t - mean_t - xhat*mean_t_xhat) * invvar` collapses to
        // `(0 - 0*mean_t_xhat) * invvar == 0` for ANY finite `invvar` —
        // exactly the same "value times finite is exact, value times NaN
        // is not" measurement as the forward case above.
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[3.0f32, 3.0, 3.0, 3.0], (1, 4), &device).unwrap();
        let gamma = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], (4,), &device).unwrap();
        let dy = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], (1, 4), &device).unwrap();
        let dx: Vec<f32> = ln_bwd_dx(1e-2, &x, &gamma, &dy)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(
            dx,
            vec![0.0f32; 4],
            "a constant x row (with constant dy*gamma) must give an EXACTLY zero dx \
             regardless of eps's magnitude — a non-finite value here means `var + eps` \
             was computed as `var - eps`"
        );
    }

    #[test]
    fn constant_row_backward_dgamma_is_exactly_zero_pinning_the_eps_sign_f32() {
        // `dgamma_i = sum_rows(dy_i * xhat_i)`; a constant x row makes
        // `xhat_i == 0.0` exactly for every `i`, so `dgamma[i] += dy[i] *
        // 0.0 == 0.0` for ANY finite `invvar` (`dy` need not be constant
        // here — only `x`'s row does).
        let device = Device::Cpu;
        let x = Tensor::from_slice(
            &[3.0f32, 3.0, 3.0, 3.0, -2.0, -2.0, -2.0, -2.0],
            (2, 4),
            &device,
        )
        .unwrap();
        let dy = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0],
            (2, 4),
            &device,
        )
        .unwrap();
        let dgamma: Vec<f32> = ln_bwd_dgamma(1e-2, &x, &dy)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(
            dgamma,
            vec![0.0f32; 4],
            "every row constant (independently) must give an EXACTLY zero dgamma \
             regardless of eps's magnitude — a non-finite value here means `var + eps` \
             was computed as `var - eps`"
        );
    }

    #[test]
    fn constant_row_backward_dgamma_is_exactly_zero_pinning_the_eps_sign_bf16() {
        let device = Device::Cpu;
        let xb = [bf16::from_f32(3.0); 4];
        let dyb: Vec<bf16> = [1.0f32, 2.0, 3.0, 4.0].map(bf16::from_f32).to_vec();
        let x = Tensor::from_slice(&xb, (1, 4), &device).unwrap();
        let dy = Tensor::from_slice(&dyb, (1, 4), &device).unwrap();
        let dgamma: Vec<bf16> = ln_bwd_dgamma(1e-2, &x, &dy)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for d in &dgamma {
            assert_eq!(
                d.to_f32(),
                0.0,
                "a constant x row must give an EXACTLY zero dgamma regardless of eps's \
                 magnitude — a non-finite value here means `var + eps` was computed as \
                 `var - eps`"
            );
        }
    }

    /// Cosmetic `name()` survivors (this file's three ops). What the
    /// snapshot pins: `name()` is the `op` payload of every typed refusal
    /// an op here raises on its CPU arm (`hidden_of(.., self.name())`,
    /// `RequiresContiguous { op: self.name() }`, `DTypeMismatchBinaryOp {
    /// op: self.name(), .. }`, `UnsupportedDTypeForOp(_, self.name())`)
    /// and of candle's own `BackwardNotSupported { op: self.name() }` —
    /// the diagnostic name a user matches error messages on, so a rename
    /// silently changes every one of them. It is NOT an admission/counter
    /// key: those are a consumer's own dispatch-site literals (`admit(..,
    /// "<key>", ..)` / `counters_for("<key>")` — see
    /// `crate::admission::counters_for`'s doc), independent of `name()`
    /// by construction. ONE snapshot per op pins the exact string rather
    /// than leaving `fn name` free to drift to any other non-empty value.
    #[test]
    fn every_ops_name_in_this_file_is_pinned() {
        assert_eq!(LayerNormFused::new(1e-5, false).name(), "layer_norm_fused");
        assert_eq!(
            LayerNormBwdDx { eps: 1e-5 }.name(),
            "layer_norm_fused_bwd_dx"
        );
        assert_eq!(
            LayerNormBwdDgamma { eps: 1e-5 }.name(),
            "layer_norm_fused_bwd_dgamma"
        );
    }

    /// The CUDA arm fills the SAME `op` payload field from a fn-local
    /// `const OP` literal in `cuda/layer_norm.rs` (a second copy of each
    /// name — that arm is a free function without `&self`). A user
    /// matching on the `op` payload must see one name per op on both
    /// devices, so each copy is pinned to `name()`. `cuda` is
    /// `#[cfg(feature = "cuda")]`-gated in `lib.rs`, so on a CPU-only
    /// build the CUDA source TEXT is the only observable form of that
    /// arm; the count assertion keeps the pin non-vacuous (a renamed or
    /// added `const OP` cannot hide behind the per-name `contains`).
    #[test]
    fn cuda_arm_op_literal_agrees_with_name_for_every_op_in_this_file() {
        let names = [
            LayerNormFused::new(1e-5, false).name(),
            LayerNormBwdDx { eps: 1e-5 }.name(),
            LayerNormBwdDgamma { eps: 1e-5 }.name(),
        ];
        let cuda_src = include_str!("../cuda/layer_norm.rs");
        assert_eq!(
            cuda_src.matches("const OP: &str = \"").count(),
            names.len(),
            "cuda/layer_norm.rs must declare exactly one `const OP` per op in this file"
        );
        for name in names {
            assert!(
                cuda_src.contains(&format!("const OP: &str = \"{name}\";")),
                "cuda/layer_norm.rs has no `const OP` equal to `{name}`"
            );
        }
    }
}
