//! Fused GeGLU (gated GELU) forward + backward.
//!
//! `out = gelu(gate) * up`, where `wi_out` (the packed `Wi` projection's
//! output, `[<leading axes>, 2*intermediate]`) splits into `gate =
//! wi_out[..., ..intermediate]` (the FIRST half) and `up = wi_out[...,
//! intermediate..]` (the SECOND half). This replaces the `[narrow, narrow,
//! gelu_erf, mul]` tail `jammi-encoders`' ModernBERT MLP call site composes
//! today:
//!
//! ```text
//! // modernbert.rs (ModernBertMlp::forward's eval arm, and
//! // geglu_apply_training's eager-fallback arm — both run this same
//! // composition; only the training arm ever dispatches to this op)
//! let up_gate = self.wi.forward(&normed)?;
//! let intermediate = up_gate.dim(D::Minus1)? / 2;
//! let gate = up_gate.narrow(D::Minus1, 0, intermediate)?;
//! let up = up_gate.narrow(D::Minus1, intermediate, intermediate)?;
//! let act = (gate.gelu_erf()? * up)?;
//! ```
//!
//! [`GegluFused`] is a `CustomOp1` over the WHOLE `wi_out` (not two
//! pre-narrowed tensors) — the split itself happens inside the kernel, so
//! the fused path reads `wi_out` once and writes only the half-width
//! output, rather than materializing `gate`, `up`, `gelu(gate)`, and the
//! product as four separate tape-resident tensors (see "memory note"
//! below).
//!
//! ## Split convention (family D: pin which half is which)
//!
//! `gate` is the FIRST half (`wi_out[..., ..intermediate]`), `up` is the
//! SECOND (`wi_out[..., intermediate..]`) — this is NOT a free choice this
//! op invents; it is dictated by `modernbert.rs`'s own `narrow` calls
//! (`gate = narrow(0, intermediate)`, `up = narrow(intermediate,
//! intermediate)`, quoted above) and by the packed `Wi` weight layout those
//! narrows read from. A caller that got this backwards would silently
//! gate on the WRONG projection half — [`tests::swapped_halves_gives_a_
//! different_and_wrong_answer`] pins a hand-computable case proving the
//! two conventions disagree (a "swap fails loudly" oracle, in the sense
//! that a caller comparing against the correct convention would see a
//! clear mismatch, not a silent near-miss).
//!
//! ## `GeluVariant`: construction data, not a silent convention (C4's
//! audited policy-as-construction-data rule)
//!
//! ModernBERT uses the ERF-based GELU (`gate.gelu_erf()?`, never the
//! tanh approximation) — confirmed directly at the call site quoted
//! above. A future caller with a DIFFERENT GeGLU variant (tanh-approximate
//! GELU, as some other architectures use) must never silently get the
//! wrong activation from this op: [`GeluVariant`] is CONSTRUCTION DATA on
//! the `Copy` [`GegluFused`] instance (the same shape
//! `SoftmaxLastDimFused::fully_masked` / `LayerNormFused::dgamma_needed`
//! take), and [`GeluVariant::Tanh`] is a TYPED REFUSAL
//! ([`check_variant`], consulted at the top of every `cpu_fwd`/`cuda_fwd`/
//! the internal backward helper) rather than an implementation — this op
//! ships exactly the variant its one real caller needs, and a caller that
//! asks for the other gets a loud `candle_core::Error`, never a silently
//! wrong number. [`GeluVariant::default()`] is [`GeluVariant::Erf`].
//!
//! ## bf16 boundary-rounding: Option A, round-before-multiply (C4's
//! audited bf16 boundary-rounding rule)
//!
//! The eager composition above is TWO separate candle ops — `gate.gelu_erf()?`
//! (one op, one output materialized) then `* up` (a second op) — not one
//! fused expression. Primary-source research (2026-08-23) on the upstream
//! HuggingFace ModernBERT reference and the HF `kernels-community`
//! `gelu_and_mul` fused kernel confirms this is the intended semantics
//! upstream too: `gelu_and_mul`'s `activation_kernels.cu` computes the
//! activation and casts to the storage dtype INSIDE `gelu_kernel` before
//! the elementwise multiply runs as a separate step — i.e. the activation
//! is rounded to the storage dtype BEFORE the multiply, not kept in a wider
//! internal type across both steps. This op reproduces that ordering
//! (DECIDED: option A from the fused-kernels plan) rather than computing
//! `gelu(gate) * up` entirely in f32 and rounding only once at the end:
//!
//! - F32: `act = gelu_erf_f32(gate)`; `out = act * up` — no intermediate
//!   rounding exists at all (one dtype throughout), so this is simply the
//!   ordinary two-op computation with no rounding-order question.
//! - BF16: `act_f32 = gelu_erf_f32(gate.to_f32())`; `act_bf16 =
//!   bf16::from_f32(act_f32)` (**ROUND 1** — matches `gelu_kernel`'s
//!   cast-to-storage-dtype-before-the-multiply step, and matches what a
//!   real two-op eager composition materializes for `gate.gelu_erf()?` as
//!   its own output tensor); `out = bf16::from_f32(act_bf16.to_f32() *
//!   up.to_f32())` (**ROUND 2** — matches `half::bf16`'s own `Mul` impl,
//!   `bf16::from_f32(a.to_f32()*b.to_f32())`, which is exactly what `act *
//!   up` computes once `act` is itself a materialized bf16 tensor). Two
//!   genuine rounding points, deliberately, matching the two-op eager
//!   shape rather than this crate's usual "accumulate in f32 throughout,
//!   round once" convention — because here the THING BEING MATCHED (the
//!   upstream two-op reference) itself rounds twice; collapsing to one
//!   rounding would be a DIFFERENT (over-precise-relative-to-eager)
//!   computation, not a correctness improvement.
//!
//! `gelu_erf_f32` itself: `x * 0.5 * (1 + erf(x/sqrt(2)))`, computed via
//! [`libm::erff`] — the SAME function `candle_core`'s own `GeluErf::f32`
//! impl calls internally (`crate::cpu::erf::erf_f32`, `op.rs`), which
//! candle-core does not re-export publicly (see this crate's `Cargo.toml`
//! for why `libm` is a direct dependency here rather than a re-derived,
//! different-precision erf). Note this is NOT bit-identical to candle's
//! own eager `gelu_erf()?` on a BF16 tensor specifically: candle's `bf16`
//! arm of `GeluErf` computes in **f64** (`bf16::from_f64(Self::f64(v.to_f64()))`,
//! `op.rs`), not f32 — a real, disclosed precision difference (this op
//! computes the activation in F32, matching every other fused op in this
//! crate's stated "f32-accumulate" convention, and matching the upstream
//! HF/`kernels-community` reference's own "fp32 opmath" more closely than
//! candle's f64 arm does). The fused-vs-eager BF16 forward oracle
//! therefore states a measured, non-vacuous tolerance against candle's
//! ACTUAL two-op composition, not a bit-exact assertion — the same "own
//! tolerance oracle" doctrine every other fused op's training arm in this
//! crate follows.
//!
//! ## No save-for-backward (candle 0.11): `bwd` recomputes from `wi_out`
//!
//! `CustomOp1::fwd` returns exactly one `(Storage, Shape)` — there is no
//! channel to stash the forward's intermediate `gelu(gate)` for `bwd` to
//! read back later. `bwd` therefore RECOMPUTES `gelu(gate)` (and its
//! derivative) from `wi_out` — the op's own argument, already tape-resident
//! (it is `Wi`'s matmul output, needed anyway) — rather than caching
//! anything in the op itself, which would also violate the `Copy`/
//! stateless requirement every op in this crate is held to (see `ops`'s
//! module doc). The whole backward is ONE internal kernel
//! ([`GegluBwdDWiOut`], a `CustomOp2` over `(wi_out, dy)`) writing BOTH
//! halves of `dwi_out` in a single launch — not two separate kernels for
//! `d_gate` and `d_up` — matching `LayerNormBwdDx`'s and `SoftmaxBwdDScores`'s
//! "one launch, not two" precedent.
//!
//! ## Backward derivation
//!
//! `gelu_erf(x) = x * Phi(x)`, `Phi(x) = 0.5*(1+erf(x/sqrt(2)))` the
//! standard-normal CDF. Its derivative:
//!
//! ```text
//! d/dx gelu_erf(x) = Phi(x) + x*phi(x)
//! ```
//!
//! where `phi(x) = (1/sqrt(2*pi)) * exp(-x^2/2)` is the standard-normal
//! PDF — the exact formula (and constant names) ATen's CPU `gelu_backward`
//! kernel uses in its erf ("none"-approximate) mode
//! (`ActivationGeluKernel.cu`/`Activation.cpp`'s `GeluBackward`): `cdf +
//! x*pdf` with `kAlpha = M_SQRT1_2` (`1/sqrt(2)`) and `kBeta =
//! M_2_SQRTPI * M_SQRT1_2 * 0.5` (`(2/sqrt(pi)) * (1/sqrt(2)) * 0.5 ==
//! 1/sqrt(2*pi)`, the PDF's normalizing constant) — [`GELU_ALPHA_F32`] and
//! [`GELU_BETA_F32`] below are exactly those two constants, reused
//! identically by the forward CDF and the backward derivative so both use
//! the SAME rounded value of each. With `out = gelu_erf(gate) * up` and
//! `dy = d(loss)/d(out)`:
//!
//! ```text
//! d_gate = dy * up * gelu_erf'(gate)   =  dy * up * (Phi(gate) + gate*phi(gate))
//! d_up   = dy * gelu_erf(gate)         =  dy * gate * Phi(gate)
//! ```
//!
//! [`tests::gradcheck_dwi_out_f32`] (this file) and the leaf-crate
//! integration suite (`tests/geglu_oracles.rs`) pin this against central
//! finite differences, including negative `gate` values, near-zero
//! `gate`, and `|gate| > 3` tails — the region where `gelu_erf'`'s two
//! terms partially cancel and a sign error is likeliest to hide.
//!
//! **This round-once F32 backward is not merely "this crate's house
//! convention" — it is STRICTLY CLOSER to the ATen/PyTorch reference than
//! candle-eager is.** ATen's own `gelu_backward` (erf mode) evaluates the
//! SAME closed form (`cdf + x*pdf`) in one fp32 pass and rounds once on
//! store — roughly 2-3 total roundings end to end, the same shape this
//! kernel takes. candle-eager's ~6-separately-rounded-op cascade (below)
//! is the outlier relative to that reference, not this kernel. This is
//! why C8's loss-quality acceptance measures fused-vs-REFERENCE (the
//! `jammi-bench/reference` torch harness) rather than fused-vs-candle-
//! eager on this specific op: candle-eager is not the more-correct
//! baseline here to converge toward.
//!
//! **The F32 backward can never be bit-exact vs. candle-eager BY
//! CONSTRUCTION, independent of any bf16 rounding question**: candle's
//! own `Op::Unary(_, GeluErf)` gradient (`backprop.rs:628`) hardcodes the
//! PDF-normalizing constant as the TRUNCATED literal `0.398942`, not the
//! full-precision `1/sqrt(2*pi) = 0.3989422804...` [`GELU_BETA_F32`]
//! below computes — a systematic ~7.0e-7 RELATIVE offset in the `pdf`
//! term on every element, present even when every OTHER source of
//! divergence (dtype, op-count, evaluation order) is held equal. This
//! quantitatively explains the measured F32 backward residual: this
//! crate's own production-width measurement
//! (`tests::bwd_matches_eager_at_production_width_f32`,
//! `geglu_oracles.rs`) is `~1.9e-6`, the SAME order of magnitude as this
//! ~7e-7-relative offset scaled by the fixture's own value range — not a
//! bug, and not reducible to zero without candle changing its own
//! (independently rounded) constant.
//!
//! **BF16 backward rounding — a disclosed, deliberate divergence from
//! eager's own multi-op cascade**: eager's backward walks candle's
//! hand-written `Op::Unary(_, GeluErf)` gradient
//! (`candle_core::backprop`: `0.5 + 0.398942*exp(-x^2/2)*x +
//! 0.5*erf(x/sqrt(2))`, `backprop.rs`) as roughly half a dozen SEPARATE
//! `Tensor` ops chained together (`sqr`, `neg`, `exp`, a scaled `erf`,
//! several affines/adds), each one rounding its own bf16 output
//! independently — a genuinely different rounding path from a single
//! closed-form kernel evaluation. Reproducing that op-by-op cascade would
//! buy no correctness (it is the identical closed form, just computed in
//! more separately-rounded steps) at real implementation cost, so
//! [`GegluBwdDWiOut`]'s BF16 arm instead follows this crate's USUAL bf16
//! backward convention (`LayerNormBwdDx`, `SoftmaxBwdDScores`):
//! accumulate `d_gate`/`d_up` in F32 throughout, rounding EACH to bf16
//! exactly once at the very end. This is a MEASURED, non-vacuous
//! divergence (a RELATIVE-tolerance-with-an-absolute-floor oracle in
//! `tests/geglu_oracles.rs`,
//! `eager_vs_fused_bf16_bwd_diverges_and_stays_within_the_stated_
//! tolerance_at_production_width` — not a raw bit-diff/ULP count, because
//! this op's own value range can sit arbitrarily close to zero, where a
//! bf16 bit-pattern diff is a degenerate metric; see the forward oracle's
//! identical reasoning), not an assumed-zero one — forward's two-
//! rounding-point design above and backward's one-rounding-point design
//! here are DIFFERENT, disclosed choices, not an inconsistency: forward
//! is chosen to match the upstream two-op reference's OWN rounding;
//! backward has no such "the reference literally does this" argument
//! against candle-eager specifically (eager's own backward cascade is an
//! artifact of candle's autodiff decomposing one closed-form gradient
//! into several ops, not a deliberate reference design) — but it DOES
//! match the ATen reference's own round-once shape more closely than
//! eager's cascade does, per the point above.
//!
//! ## Memory note (the actual win this commit targets)
//!
//! Eager retains, on the backward tape, at `[<leading>, intermediate]`
//! shape: `gate` (the narrow view — cheap, a view not a copy), `up`
//! (likewise), `gelu(gate)` (`gate.gelu_erf()?`'s own output — a REAL
//! materialized tensor), and the final product `act * up`'s own node.
//! Backward through `Op::Unary(_, GeluErf)` additionally recomputes
//! several MORE `[<leading>, intermediate]`-shaped intermediates internally
//! (`sqr`, `exp`, `erf`, per `backprop.rs`, quoted above) — none of which
//! are retained from forward (candle recomputes them during the backward
//! walk itself), but each is a fresh allocation during backward regardless.
//! The fused path retains only `wi_out` itself (already tape-resident — it
//! is `Wi`'s matmul output, needed for LoRA's own backward through `Wi`
//! too) and the half-width `out`; `bwd` recomputes `gelu(gate)` and its
//! derivative from `wi_out` inside ONE kernel launch with no additional
//! tape-visible allocation. At ModernBERT-large's shape (`intermediate =
//! 2624` per HuggingFace's published `answerdotai/ModernBERT-large`
//! `config.json`; `hidden = 1024`), this removes several
//! `[batch, seq, 2624]`-shaped retained/recomputed tensors per layer, on
//! top of collapsing what was 4 graph nodes (two narrows, one `gelu_erf`,
//! one `mul`) into 1 forward node.
//!
//! ## Domain (family D)
//!
//! `wi_out` must be fully contiguous ([`candle_core::Layout::contiguous_offsets`],
//! the same idiom every other op in this crate uses, and for the same
//! reason: a raw-pointer kernel has no flat linear index for a strided
//! view — this op's real call site is always contiguous, being a matmul's
//! direct output). CPU supports F32 and BF16 (this crate's real training
//! dtypes), plus F16 (`geglu_fwd_f16`/`geglu_bwd_f16` below). Campaign
//! #443 W2b added the matching CUDA F16 dispatch arm
//! (`crate::cuda::geglu`'s `DType::F16` arms, backed by the SEPARATE
//! `cuda/geglu_f16.cu` translation unit — see that file's module doc for
//! why it duplicates rather than shares code with the F32/BF16 kernels),
//! so `jammi-encoders`' admission predicate is now widened to F16 too
//! (K2's no-Hold-without-dispatch rule); see
//! `docs/maintainer/cuda-kernel-guide.md`'s per-op f16 reference-regime
//! table. The last dimension must be EVEN (it packs two equal halves) —
//! an ODD last dimension is a structural domain violation (there is no way
//! to split it into equal `gate`/`up` halves), refused with a typed error,
//! not silently truncated or padded. A last dimension of exactly `0`
//! degenerates to an empty output (nothing to compute) — the same
//! "zero-length last dim implies zero elements" argument
//! `LayerNormFused::hidden == 0` documents; this is checked BEFORE the
//! even/odd check (0 is technically even, but the degenerate-empty case is
//! handled as its own fast path rather than falling through the ordinary
//! arithmetic, mirroring every other op's `last == 0` / `hidden == 0`
//! precedent). No CUDA-specific width ceiling exists (unlike
//! `LayerNormFused::MAX_HIDDEN` / `SoftmaxLastDimFused::MAX_LAST_DIM`):
//! this op's CUDA kernels are purely elementwise (no per-row block
//! reduction, hence no shared-memory footprint that scales with
//! `intermediate`), so the CUDA arm's only numeric domain constraint is
//! the same `elements <= u32::MAX` guard every op's CUDA glue in this
//! crate states (the launch grid and the kernel's own indices are 32-bit).

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, CustomOp2, Error, Layout, Result, Shape, Tensor};
use half::{bf16, f16};

/// `1/sqrt(2)` — ATen's `kAlpha` (`ActivationGeluKernel.cu`'s erf-mode
/// `gelu_backward`), reused for both the forward CDF and the backward
/// derivative so both use the identical rounded constant.
const GELU_ALPHA_F32: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// `(2/sqrt(pi)) * (1/sqrt(2)) * 0.5 == 1/sqrt(2*pi)` — ATen's `kBeta`
/// (`ActivationGeluKernel.cu`), the standard-normal PDF's normalizing
/// constant.
const GELU_BETA_F32: f32 = std::f32::consts::FRAC_2_SQRT_PI * std::f32::consts::FRAC_1_SQRT_2 * 0.5;

/// `gelu_erf(x) = x * Phi(x)`, `Phi(x) = 0.5*(1+erf(x/sqrt(2)))` — matches
/// `candle_core`'s own `GeluErf::f32` formula
/// (`(erf_f32(v*FRAC_1_SQRT_2)+1.)*0.5*v`) exactly, via the same
/// [`libm::erff`] candle's own (non-public) `crate::cpu::erf::erf_f32`
/// wraps. See the module doc's "bf16 boundary-rounding" section for why
/// this crate depends on `libm` directly.
fn gelu_erf_f32(x: f32) -> f32 {
    (libm::erff(x * GELU_ALPHA_F32) + 1.0) * 0.5 * x
}

/// `d/dx gelu_erf(x) = Phi(x) + x*phi(x)` — see the module doc's
/// "backward derivation" section for the ATen citation. Returns
/// `(gelu_erf(x), gelu_erf'(x))` together since both share `cdf`.
fn gelu_erf_and_grad_f32(x: f32) -> (f32, f32) {
    let cdf = (libm::erff(x * GELU_ALPHA_F32) + 1.0) * 0.5;
    let pdf = GELU_BETA_F32 * (-0.5 * x * x).exp();
    (x * cdf, cdf + x * pdf)
}

/// Which GELU formula [`GegluFused`]'s forward/backward implement. See the
/// module doc's "`GeluVariant`: construction data" section — this is
/// CONSTRUCTION DATA on the `Copy` op, never a runtime predicate the op
/// evaluates against any tensor's own state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeluVariant {
    /// The erf-based GELU (`candle_core::Tensor::gelu_erf`) — what
    /// ModernBERT's MLP call site actually uses. Implemented.
    Erf,
    /// The tanh approximation (`candle_core::Tensor::gelu`). NOT
    /// implemented by this op: a typed refusal (`check_variant`), not a
    /// silently-computed (and silently wrong, for a caller that actually
    /// wanted the tanh formula) alternative activation.
    Tanh,
}

impl Default for GeluVariant {
    /// [`GeluVariant::Erf`] — the only variant this crate's one real
    /// caller (ModernBERT's MLP) needs. A future caller wanting the tanh
    /// approximation must construct [`GeluVariant::Tanh`] explicitly and
    /// gets a loud, typed refusal rather than a silently-erf-computed
    /// result.
    fn default() -> Self {
        GeluVariant::Erf
    }
}

/// Refuses [`GeluVariant::Tanh`] with a typed error naming the op — see
/// the module doc. `pub(crate)`: `crate::cuda::geglu` calls this too, so
/// both the CPU and CUDA arms (and the internal backward helper) apply
/// the identical check rather than each re-deriving it.
pub(crate) fn check_variant(variant: GeluVariant, op: &'static str) -> Result<()> {
    match variant {
        GeluVariant::Erf => Ok(()),
        GeluVariant::Tanh => Err(Error::Msg(format!(
            "{op}: GeluVariant::Tanh is not implemented — a typed refusal, not a \
             silently-wrong activation. ModernBERT (and every call site this crate \
             ships) uses GeluVariant::Erf; a caller that genuinely needs the tanh \
             approximation must not construct this op with Tanh until it is \
             implemented."
        ))),
    }
}

/// Validates `wi_out`'s domain shared by every arm of this op (see the
/// module doc's "domain" section) and returns `(rows, intermediate)`:
/// `intermediate` is HALF of `wi_out`'s last dimension, `rows =
/// wi_out.elem_count() / (2*intermediate)` (or `(0, 0)` when
/// `intermediate == 0` — the caller checks that and does not divide by
/// it). `pub(crate)`: `crate::cuda::geglu` shares this exact check rather
/// than re-deriving it (the same "shared, not duplicated" choice
/// `ops::softmax::softmax_dims` documents for its own reasons).
pub(crate) fn geglu_dims(l: &Layout, op: &'static str) -> Result<(usize, usize)> {
    let dims = l.dims();
    let Some(&last) = dims.last() else {
        return Err(Error::Msg(format!(
            "{op}: input must have rank >= 1 to define a last (gate+up packed) dimension"
        )));
    };
    if last == 0 {
        return Ok((0, 0));
    }
    if !last.is_multiple_of(2) {
        return Err(Error::Msg(format!(
            "{op}: last dim {last} is odd — wi_out must pack gate and up into two \
             EQUAL halves (gate = wi_out[..,..last/2], up = wi_out[..,last/2..]); an \
             odd width cannot split evenly. This is a structural domain violation, \
             not a validated-coverage ceiling."
        )));
    }
    let intermediate = last / 2;
    Ok((dims.iter().product::<usize>() / last, intermediate))
}

/// `wi_out`'s shape with its last dimension replaced by `intermediate`
/// (half of `wi_out`'s own last dimension) — the shape [`GegluFused`]'s
/// forward output (and [`GegluBwdDWiOut`]'s `dy` argument) must have.
/// `pub(crate)`: shared with `crate::cuda::geglu`.
pub(crate) fn output_shape(l: &Layout, intermediate: usize) -> Shape {
    let mut dims = l.dims().to_vec();
    if let Some(last) = dims.last_mut() {
        *last = intermediate;
    }
    Shape::from(dims)
}

/// Fused GeGLU forward. See the module doc for the full design.
#[derive(Debug, Clone, Copy, Default)]
pub struct GegluFused {
    /// See [`GeluVariant`]'s doc. Construction data, never inspected
    /// against any tensor's runtime state.
    pub variant: GeluVariant,
}

impl GegluFused {
    pub fn new(variant: GeluVariant) -> Self {
        Self { variant }
    }
}

impl super::sealed::Sealed for GegluFused {}

impl CustomOp1 for GegluFused {
    fn name(&self) -> &'static str {
        "geglu_fused"
    }

    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout) -> Result<(CpuStorage, Shape)> {
        check_variant(self.variant, self.name())?;
        let (rows, intermediate) = geglu_dims(l1, self.name())?;
        let out_shape = output_shape(l1, intermediate);
        if intermediate == 0 {
            return match s1 {
                CpuStorage::F32(_) => Ok((CpuStorage::F32(Vec::new()), out_shape)),
                CpuStorage::BF16(_) => Ok((CpuStorage::BF16(Vec::new()), out_shape)),
                CpuStorage::F16(_) => Ok((CpuStorage::F16(Vec::new()), out_shape)),
                s => Err(Error::UnsupportedDTypeForOp(s.dtype(), self.name())),
            };
        }
        let (o1, o2) = l1
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: self.name() })?;
        match s1 {
            CpuStorage::F32(x) => {
                let out = geglu_fwd_f32(&x[o1..o2], rows, intermediate);
                Ok((CpuStorage::F32(out), out_shape))
            }
            CpuStorage::BF16(x) => {
                let out = geglu_fwd_bf16(&x[o1..o2], rows, intermediate);
                Ok((CpuStorage::BF16(out), out_shape))
            }
            CpuStorage::F16(x) => {
                let out = geglu_fwd_f16(&x[o1..o2], rows, intermediate);
                Ok((CpuStorage::F16(out), out_shape))
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
        crate::cuda::geglu::cuda_fwd(self.variant, s1, l1)
    }

    /// See the module doc's "no save-for-backward" section. `dwi_out`'s
    /// slot is ALWAYS `Some` — `arg` (`wi_out`) may be an INTERMEDIATE on
    /// a path to a `Var` (`Tensor::is_variable() == false` does not mean
    /// "no gradient needed" — the same hazard `Axpy`'s doc, and every
    /// other op in this crate, document), so this never gates on it.
    fn bwd(&self, arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        let dwi_out = super::apply2(
            arg,
            grad_res,
            GegluBwdDWiOut {
                variant: self.variant,
            },
        )?;
        Ok(Some(dwi_out))
    }
}

/// `GegluFused`'s internal backward helper producing `dwi_out` (full
/// width — both `d_gate` and `d_up`, written in ONE kernel launch). Not
/// exported — only ever invoked from [`GegluFused::bwd`] via
/// [`super::apply2`]. See the module doc for the derivation and the bf16
/// rounding disclosure.
#[derive(Debug, Clone, Copy)]
struct GegluBwdDWiOut {
    variant: GeluVariant,
}

impl super::sealed::Sealed for GegluBwdDWiOut {}

impl CustomOp2 for GegluBwdDWiOut {
    fn name(&self) -> &'static str {
        "geglu_fused_bwd_dwi_out"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        check_variant(self.variant, self.name())?;
        let (rows, intermediate) = geglu_dims(l1, self.name())?;
        let expected_dy = output_shape(l1, intermediate);
        if l2.dims() != expected_dy.dims() {
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
        if intermediate == 0 {
            return match s1 {
                CpuStorage::F32(_) => Ok((CpuStorage::F32(Vec::new()), l1.shape().clone())),
                CpuStorage::BF16(_) => Ok((CpuStorage::BF16(Vec::new()), l1.shape().clone())),
                CpuStorage::F16(_) => Ok((CpuStorage::F16(Vec::new()), l1.shape().clone())),
                s => Err(Error::UnsupportedDTypeForOp(s.dtype(), self.name())),
            };
        }
        let (o1, o2) = l1
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: self.name() })?;
        let (d1, d2) = l2
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: self.name() })?;
        match (s1, s2) {
            (CpuStorage::F32(x), CpuStorage::F32(dy)) => {
                let out = geglu_bwd_f32(&x[o1..o2], &dy[d1..d2], rows, intermediate);
                Ok((CpuStorage::F32(out), l1.shape().clone()))
            }
            (CpuStorage::BF16(x), CpuStorage::BF16(dy)) => {
                let out = geglu_bwd_bf16(&x[o1..o2], &dy[d1..d2], rows, intermediate);
                Ok((CpuStorage::BF16(out), l1.shape().clone()))
            }
            (CpuStorage::F16(x), CpuStorage::F16(dy)) => {
                let out = geglu_bwd_f16(&x[o1..o2], &dy[d1..d2], rows, intermediate);
                Ok((CpuStorage::F16(out), l1.shape().clone()))
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
        crate::cuda::geglu::cuda_bwd_dwi_out(self.variant, s1, l1, s2, l2)
    }

    // No `bwd` override: this helper's own second-order gradient is never
    // requested by any call site in this crate or its consumers — the
    // default `CustomOp2::bwd` (`Err(BackwardNotSupported)`) is the
    // correct refusal if anything ever tried, mirroring
    // `LayerNormBwdDx`'s / `SoftmaxBwdDScores`'s identical notes.
}

// -----------------------------------------------------------------------
// CPU math. Fixed fold order throughout (family J): every row is computed
// independently and every column within a row is computed in plain
// ascending index order — no reduction, so no fold-order question beyond
// "iterate ascending", but stated for consistency with every other op's
// determinism section.
// -----------------------------------------------------------------------

fn geglu_fwd_row_f32(row: &[f32], intermediate: usize, out: &mut [f32]) {
    for i in 0..intermediate {
        let gate = row[i];
        let up = row[intermediate + i];
        out[i] = gelu_erf_f32(gate) * up;
    }
}

fn geglu_fwd_f32(wi_out: &[f32], rows: usize, intermediate: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * intermediate];
    for r in 0..rows {
        let row = &wi_out[r * 2 * intermediate..(r + 1) * 2 * intermediate];
        let outr = &mut out[r * intermediate..(r + 1) * intermediate];
        geglu_fwd_row_f32(row, intermediate, outr);
    }
    out
}

/// BF16: rounds the activation to BF16 immediately (**ROUND 1**), then
/// multiplies in f32 and rounds ONCE more (**ROUND 2**) — see the module
/// doc's "bf16 boundary-rounding" section for why this is two genuine
/// rounding points, matching the upstream two-op reference, rather than
/// this crate's usual single-rounding convention.
fn geglu_fwd_row_bf16(row: &[bf16], intermediate: usize, out: &mut [bf16]) {
    for i in 0..intermediate {
        let gate = row[i].to_f32();
        let up = row[intermediate + i].to_f32();
        let act_bf16 = bf16::from_f32(gelu_erf_f32(gate)); // ROUND 1
        out[i] = bf16::from_f32(act_bf16.to_f32() * up); // ROUND 2
    }
}

fn geglu_fwd_bf16(wi_out: &[bf16], rows: usize, intermediate: usize) -> Vec<bf16> {
    let mut out = vec![bf16::ZERO; rows * intermediate];
    for r in 0..rows {
        let row = &wi_out[r * 2 * intermediate..(r + 1) * 2 * intermediate];
        let outr = &mut out[r * intermediate..(r + 1) * intermediate];
        geglu_fwd_row_bf16(row, intermediate, outr);
    }
    out
}

/// [`geglu_fwd_row_bf16`]'s exact twin, substituting `half::f16` — the
/// same two-rounding-point regime (candle's own F16 `GeluErf::f16` arm
/// ALSO computes in f64, mirroring its `bf16` arm exactly — see
/// `candle-core`'s `op.rs`), recorded in the per-op f16 reference-regime
/// table (`docs/maintainer/cuda-kernel-guide.md`).
fn geglu_fwd_row_f16(row: &[f16], intermediate: usize, out: &mut [f16]) {
    for i in 0..intermediate {
        let gate = row[i].to_f32();
        let up = row[intermediate + i].to_f32();
        let act_f16 = f16::from_f32(gelu_erf_f32(gate)); // ROUND 1
        out[i] = f16::from_f32(act_f16.to_f32() * up); // ROUND 2
    }
}

fn geglu_fwd_f16(wi_out: &[f16], rows: usize, intermediate: usize) -> Vec<f16> {
    let mut out = vec![f16::ZERO; rows * intermediate];
    for r in 0..rows {
        let row = &wi_out[r * 2 * intermediate..(r + 1) * 2 * intermediate];
        let outr = &mut out[r * intermediate..(r + 1) * intermediate];
        geglu_fwd_row_f16(row, intermediate, outr);
    }
    out
}

/// `d_gate = dy*up*gelu_erf'(gate)`, `d_up = dy*gelu_erf(gate)` — see the
/// module doc's "backward derivation" section.
fn geglu_bwd_row_f32(row: &[f32], dy: &[f32], intermediate: usize, dwi: &mut [f32]) {
    for i in 0..intermediate {
        let gate = row[i];
        let up = row[intermediate + i];
        let dyi = dy[i];
        let (gelu_val, gelu_deriv) = gelu_erf_and_grad_f32(gate);
        dwi[i] = dyi * up * gelu_deriv;
        dwi[intermediate + i] = dyi * gelu_val;
    }
}

fn geglu_bwd_f32(wi_out: &[f32], dy: &[f32], rows: usize, intermediate: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * 2 * intermediate];
    for r in 0..rows {
        let row = &wi_out[r * 2 * intermediate..(r + 1) * 2 * intermediate];
        let dyr = &dy[r * intermediate..(r + 1) * intermediate];
        let dwr = &mut out[r * 2 * intermediate..(r + 1) * 2 * intermediate];
        geglu_bwd_row_f32(row, dyr, intermediate, dwr);
    }
    out
}

/// BF16: f32-accumulate throughout, rounding EACH of `d_gate`/`d_up` to
/// bf16 exactly once at the very end — see the module doc's "BF16
/// backward rounding" section for why this deliberately does NOT mirror
/// eager's own multi-op rounding cascade.
fn geglu_bwd_row_bf16(row: &[bf16], dy: &[bf16], intermediate: usize, dwi: &mut [bf16]) {
    for i in 0..intermediate {
        let gate = row[i].to_f32();
        let up = row[intermediate + i].to_f32();
        let dyi = dy[i].to_f32();
        let (gelu_val, gelu_deriv) = gelu_erf_and_grad_f32(gate);
        dwi[i] = bf16::from_f32(dyi * up * gelu_deriv);
        dwi[intermediate + i] = bf16::from_f32(dyi * gelu_val);
    }
}

fn geglu_bwd_bf16(wi_out: &[bf16], dy: &[bf16], rows: usize, intermediate: usize) -> Vec<bf16> {
    let mut out = vec![bf16::ZERO; rows * 2 * intermediate];
    for r in 0..rows {
        let row = &wi_out[r * 2 * intermediate..(r + 1) * 2 * intermediate];
        let dyr = &dy[r * intermediate..(r + 1) * intermediate];
        let dwr = &mut out[r * 2 * intermediate..(r + 1) * 2 * intermediate];
        geglu_bwd_row_bf16(row, dyr, intermediate, dwr);
    }
    out
}

/// [`geglu_bwd_row_bf16`]'s exact twin, substituting `half::f16`.
fn geglu_bwd_row_f16(row: &[f16], dy: &[f16], intermediate: usize, dwi: &mut [f16]) {
    for i in 0..intermediate {
        let gate = row[i].to_f32();
        let up = row[intermediate + i].to_f32();
        let dyi = dy[i].to_f32();
        let (gelu_val, gelu_deriv) = gelu_erf_and_grad_f32(gate);
        dwi[i] = f16::from_f32(dyi * up * gelu_deriv);
        dwi[intermediate + i] = f16::from_f32(dyi * gelu_val);
    }
}

fn geglu_bwd_f16(wi_out: &[f16], dy: &[f16], rows: usize, intermediate: usize) -> Vec<f16> {
    let mut out = vec![f16::ZERO; rows * 2 * intermediate];
    for r in 0..rows {
        let row = &wi_out[r * 2 * intermediate..(r + 1) * 2 * intermediate];
        let dyr = &dy[r * intermediate..(r + 1) * intermediate];
        let dwr = &mut out[r * 2 * intermediate..(r + 1) * 2 * intermediate];
        geglu_bwd_row_f16(row, dyr, intermediate, dwr);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn fused(variant: GeluVariant, wi_out: &Tensor) -> Result<Tensor> {
        // Through `ops::apply1` (requires `T: KernelOp`), not
        // `Tensor::apply_op1` directly — the enforcement point `ops`'s
        // `KernelOp` bound exists for.
        crate::ops::apply1(wi_out, GegluFused::new(variant))
    }

    fn eager(wi_out: &Tensor) -> Result<Tensor> {
        let intermediate = wi_out.dim(candle_core::D::Minus1)? / 2;
        let gate = wi_out.narrow(candle_core::D::Minus1, 0, intermediate)?;
        let up = wi_out.narrow(candle_core::D::Minus1, intermediate, intermediate)?;
        (gate.gelu_erf()? * up)?.contiguous()
    }

    #[test]
    fn cpu_fwd_f32_matches_hand_computed_values() {
        let device = Device::Cpu;
        // gate = [0.0, 1.0], up = [2.0, 3.0]; gelu_erf(0) = 0 exactly.
        let wi = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0], (1, 4), &device).unwrap();
        let out = fused(GeluVariant::Erf, &wi)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let g1 = gelu_erf_f32(1.0f32);
        assert!(
            (out[0] - 0.0).abs() < 1e-6,
            "gelu_erf(0)*2 must be 0: {out:?}"
        );
        assert!(
            (out[1] - g1 * 3.0).abs() < 1e-6,
            "{out:?} vs gelu_erf(1)*3={}",
            g1 * 3.0
        );
    }

    #[test]
    fn cpu_fwd_matches_eager_narrow_gelu_erf_mul_composition() {
        let device = Device::Cpu;
        let v: [f32; 8] = [-2.0, -0.5, 0.0, 0.3, 1.0, 2.5, -1.5, 4.0];
        let wi = Tensor::from_slice(&v, (2, 4), &device).unwrap();
        let out_fused: Vec<f32> = fused(GeluVariant::Erf, &wi)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let out_eager: Vec<f32> = eager(&wi)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, (f, e)) in out_fused.iter().zip(out_eager.iter()).enumerate() {
            assert!((f - e).abs() < 1e-5, "elem[{i}]: fused {f} vs eager {e}");
        }
    }

    /// Split-convention oracle (family D): swapping which half is `gate`
    /// vs `up` must give a DIFFERENT answer on an asymmetric fixture — a
    /// caller that got the convention backwards would silently gate on
    /// the wrong projection half.
    #[test]
    fn swapped_halves_gives_a_different_and_wrong_answer() {
        let device = Device::Cpu;
        // Asymmetric: first half != second half, and neither gelu_erf(x)
        // nor the product is symmetric under swapping which is "gate".
        let v: [f32; 4] = [3.0, -1.0, 0.1, 5.0];
        let wi = Tensor::from_slice(&v, (1, 4), &device).unwrap();
        let correct: Vec<f32> = fused(GeluVariant::Erf, &wi)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // The swapped convention: up as gate, gate as up.
        let swapped = Tensor::from_slice(&[v[2], v[3], v[0], v[1]], (1, 4), &device).unwrap();
        let wrong: Vec<f32> = fused(GeluVariant::Erf, &swapped)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        assert!(
            correct
                .iter()
                .zip(wrong.iter())
                .any(|(c, w)| (c - w).abs() > 1e-3),
            "swapping which half is gate/up must change the answer: {correct:?} vs {wrong:?}"
        );
    }

    #[test]
    fn tanh_variant_is_a_typed_refusal_not_a_silent_computation() {
        let device = Device::Cpu;
        let wi = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let err = fused(GeluVariant::Tanh, &wi).expect_err("Tanh must be refused, not computed");
        assert!(matches!(err, Error::Msg(_)));
    }

    #[test]
    fn empty_last_dim_is_a_no_op_not_an_error() {
        let device = Device::Cpu;
        let wi = Tensor::from_slice(&[] as &[f32], (3, 0), &device).unwrap();
        let out = fused(GeluVariant::Erf, &wi).unwrap();
        assert_eq!(out.dims(), &[3, 0]);
    }

    #[test]
    fn odd_last_dim_is_refused_not_silently_misread() {
        let device = Device::Cpu;
        let wi = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (1, 3), &device).unwrap();
        let err = fused(GeluVariant::Erf, &wi).expect_err("odd last dim must be refused");
        assert!(matches!(err, Error::Msg(_)));
    }

    #[test]
    fn non_contiguous_wi_out_is_refused_not_silently_misread() {
        let device = Device::Cpu;
        // A transposed view: contiguous along the WRONG axis.
        let wi = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            (4, 2),
            &device,
        )
        .unwrap()
        .t()
        .unwrap();
        assert!(!wi.is_contiguous());
        let err = fused(GeluVariant::Erf, &wi).expect_err("non-contiguous wi_out must be refused");
        assert!(matches!(err, Error::RequiresContiguous { .. }));
    }

    #[test]
    fn bf16_forward_matches_f32_reference_within_a_small_tolerance() {
        let device = Device::Cpu;
        let v: [f32; 4] = [1.5, -2.0, 0.25, 3.0];
        let vb: Vec<bf16> = v.iter().map(|&x| bf16::from_f32(x)).collect();
        let wi = Tensor::from_slice(&vb, (1, 4), &device).unwrap();
        let out: Vec<bf16> = fused(GeluVariant::Erf, &wi)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        // Independent f64 reference computed from the SAME (already
        // bf16-rounded) inputs, matching this op's documented two-round
        // semantics (round the activation, then round the product).
        // intermediate = 2: gate = wi_out[..,0..2], up = wi_out[..,2..4],
        // so element 0's gate/up are wi_out[0] / wi_out[2].
        let gate = vb[0].to_f64();
        let up = vb[2].to_f64();
        let cdf = 0.5 * (1.0 + libm::erf(gate * std::f64::consts::FRAC_1_SQRT_2));
        let act = bf16::from_f64(gate * cdf);
        let expected = bf16::from_f64(act.to_f64() * up);
        assert!(
            (out[0].to_f32() - expected.to_f32()).abs() < 1e-2,
            "{:?} vs {:?}",
            out[0],
            expected
        );
    }

    /// F16's exact twin of `bf16_forward_matches_f32_reference_within_a_
    /// small_tolerance` above.
    #[test]
    fn f16_forward_matches_f32_reference_within_a_small_tolerance() {
        let device = Device::Cpu;
        let v: [f32; 4] = [1.5, -2.0, 0.25, 3.0];
        let vh: Vec<f16> = v.iter().map(|&x| f16::from_f32(x)).collect();
        let wi = Tensor::from_slice(&vh, (1, 4), &device).unwrap();
        let out: Vec<f16> = fused(GeluVariant::Erf, &wi)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let gate = vh[0].to_f64();
        let up = vh[2].to_f64();
        let cdf = 0.5 * (1.0 + libm::erf(gate * std::f64::consts::FRAC_1_SQRT_2));
        let act = f16::from_f64(gate * cdf);
        let expected = f16::from_f64(act.to_f64() * up);
        assert!(
            (out[0].to_f32() - expected.to_f32()).abs() < 1e-2,
            "{:?} vs {:?}",
            out[0],
            expected
        );
    }
}
