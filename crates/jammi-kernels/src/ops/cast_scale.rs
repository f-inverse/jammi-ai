//! Two cast-boundary fusions for [`super::LowRankResidualLinear::bwd`]'s
//! epilogue and residual-add sites (the "cast-boundary lever", Wave 1
//! (e)/(f): B1 is the epilogue's `to_dtype(F32)` + `.affine(scale, 0.0)`
//! pair fused as [`CastScaleBf16F32`]; B3 is the residual-add's
//! `to_dtype(bf16)` + `Tensor::add` pair fused as [`CastAddBf16`] — see
//! each type's own doc below for its exact expression and traffic model).
//! Both ops are generic Tensor-API primitives (family L: this crate names
//! no consumer); `LowRankResidualLinear::bwd` is their first caller, not a
//! fixed contract.
//!
//! ## [`CastScaleBf16F32`] — B1's `to_dtype(F32)` + `.affine(scale, 0.0)`
//!
//! `out_f32 = f32(x_bf16) * scale + 0.0` in ONE kernel, replacing candle's
//! own two-launch chain: `cast_bf16_f32` (`candle-kernels-0.11.0/src/
//! cast.cu`'s `cast_<__nv_bfloat16, float>` — `out[i] = inp[i]`, an exact
//! widening via `__nv_bfloat16`'s implicit `operator float()`, no rounding)
//! THEN `affine_f32` (`candle-kernels-0.11.0/src/affine.cu:45`,
//! `AFFINE_OP(float, affine_f32, x * mul + add)`, with `mul`/`add` computed
//! Rust-side as `T::from_f64(self.0)` — for `T = f32` that is `scaling as
//! f32` verbatim, `candle-core-0.11.0/src/dtype.rs:235`; confirmed at BOTH
//! call sites that build the kernel's launch args, `cpu_backend/mod.rs:313`
//! (`let mul = T::from_f64(self.0)`) and `cuda_backend/mod.rs:158`
//! (`barg!(builder, T::from_f64(self.0))`) — the CPU and CUDA arms compute
//! the SAME `f32` constant the SAME way). This kernel reproduces that exact
//! two-step expression bit-for-bit — `x_f32 = (float)x_bf16` (exact) then
//! `x_f32 * (scale as f32) + 0.0f32` — with the unscaled f32 intermediate's
//! HBM round-trip removed rather than any arithmetic changed. The `+ 0.0f`
//! term is REQUIRED (not an optimizable no-op): it is what makes this
//! bit-identical to `affine_f32`'s own `x * mul + add` for the signed-zero
//! case (`-0.0 * scale + 0.0 == +0.0` in IEEE 754 round-to-nearest, whereas
//! `-0.0 * scale` alone stays `-0.0` when `scale > 0` — see the RED control
//! in this file's own tests).
//!
//! Traffic: old = cast(read 2B + write 4B) + affine(read 4B + write 4B) =
//! 14 B/elem; new = read 2B + write 4B = 6 B/elem (design study §1's (e),
//! credit 8 B/elem).
//!
//! Domain (family D): input MUST be `BF16` — this op's own name states the
//! one dtype pair it fuses; a caller with an already-`F32` `grad_res` has
//! nothing to fuse (candle's `affine_f32` alone is already the minimal
//! single kernel for that case) and should call `Tensor::affine` directly,
//! never routing an `F32` input through this op. `cpu_fwd` refuses any
//! other input dtype with a typed `Error::UnsupportedDTypeForOp` (never a
//! silent reinterpretation). The CUDA arm additionally REQUIRES contiguous
//! storage (`Error::RequiresContiguous`, never a silent misread of a
//! strided view) — matching every other elementwise op in this crate
//! ([`super::Axpy`]/[`super::ScaledCastAdd`]'s own documented CUDA domain).
//! See "Contiguity at the call site" below for why `bwd` never actually
//! reaches this refusal in production.
//!
//! ## [`CastAddBf16`] — B3's `to_dtype(bf16)` + the `dx_base + d_x_lora` add
//!
//! `out_bf16 = base_bf16 + round_to_bf16(f32val)`, replacing candle's own
//! two-launch chain: `cast_f32_bf16` (`cast.cu`'s `cast_<float,
//! __nv_bfloat16>` — `out[i] = inp[i]`, an implicit `__nv_bfloat16(float)`
//! constructor, round-to-nearest-even) THEN `badd_bf16`
//! (`candle-kernels-0.11.0/src/binary.cu:5`, `BINARY_OP(__nv_bfloat16,
//! badd_bf16, x + y)` — literally the C++ `operator+` on two `__nv_bfloat16`
//! operands). This kernel's CUDA arm uses the IDENTICAL expression
//! `base[i] + delta` (not a widen-to-f32-and-round emulation of its own),
//! so the compiled SASS for the add step is whatever nvcc emits for `x + y`
//! on `__nv_bfloat16` at THIS crate's pinned `sm_80` target — the same
//! question `binary.cu:5` answers for candle's own kernel, by construction
//! rather than by re-deriving what the hardware intrinsic does.
//!
//! The rounding order the task requires — round the f32 intermediate to
//! bf16 IN-REGISTER first (`__float2bfloat16`), THEN add in native bf16 —
//! is exactly what the two-argument order encodes: `f32val` is rounded to
//! `delta: __nv_bfloat16` before it ever reaches the `+`.
//!
//! **Why this rounding order, not accumulate-then-round-once — the PEFT
//! citation trail.** PEFT's `Linear.forward` (installed `peft==0.20.0`,
//! `peft/tuners/lora/layer.py:1056`) casts its input UP before the adapter
//! GEMMs: `x = self._cast_input_dtype(x, lora_A.weight.dtype)`.
//! `_cast_input_dtype` (`peft/tuners/tuners_utils.py:1777-1792`, verified at
//! the installed source this session) is a plain `x.to(dtype=dtype)` when
//! the dtypes differ (`:1790-1792`) — an ordinary torch `.to()` cast, not a
//! custom op. `dx` (this op's `f32val` argument, `d_xd` in
//! `LowRankResidualLinear::bwd`) is the gradient flowing BACKWARD through
//! that SAME forward cast, so its backward is `.to()`'s own backward
//! (`aten::_to_copy`) — torch's `_to_copy_backward` casts the upstream f32
//! gradient DOWN to the input's original (`bf16`) dtype BEFORE autograd
//! accumulates it against the base branch's own (already-`bf16`) gradient
//! contribution at the shared leaf, per `_to_copy`'s standard "backward of
//! a cast is a cast of the gradient" rule (`tools/autograd/derivatives.yaml`
//! / `torch/csrc/autograd/FunctionsManual.cpp`'s `_to_copy_backward`) — i.e.
//! round-then-add on the `dx` path, exactly what this kernel's argument
//! order preserves. UNVERIFIED-AT-SOURCE-THIS-SESSION: no torch
//! installation or vendored `derivatives.yaml`/`FunctionsManual.cpp` was
//! reachable from this environment (checked: no `torch` Python module, no
//! `derivatives.yaml` on this filesystem) — the `_to_copy_backward`
//! citation is stated from the standard "cast backward is a cast" autograd
//! convention documented in torch's own public docs and PEFT's own
//! matching forward-cast shape, NOT confirmed by reading the C++ source
//! directly this session. A future reader with pod/torch access should
//! confirm `derivatives.yaml`'s `_to_copy: self: _to_copy_backward(grad,
//! self.options())` line before treating this as source-verified rather
//! than convention-inferred.
//!
//! **CPU arm and the double-rounding argument.** The CPU arm cannot call an
//! `operator+`; it must decide what "native bf16 add" means in software.
//! `half::bf16`'s own `Add` impl (`half-2.7.1/src/bfloat.rs:998-1004`,
//! `Self::from_f32(Self::to_f32(self) + Self::to_f32(rhs))`) — the SAME
//! implementation candle's CPU `badd_bf16` arm dispatches through
//! (`candle-core-0.11.0/src/op.rs:456`, `bin_op!(Add, add, |v1, v2| v1 +
//! v2, ..)`, generic over `T: WithDType`, `T = bf16` resolving to this
//! exact `+`) — widens both operands to `f32`, adds in `f32`, and rounds
//! back to `bf16` ONCE. This is provably equal to a genuine single-rounding
//! narrow-precision add for operands that are THEMSELVES `bf16`-precision
//! (as both `base` and `delta` are here, `delta` having just been rounded
//! down from `f32val`): the classical double-rounding-safety bound requires
//! the intermediate format to carry at least `2 * (narrow mantissa bits) +
//! 2` bits beyond what the narrow format needs to represent the exact sum
//! without ambiguity; `f32`'s 24-bit (23 stored + implicit) significand
//! against `bf16`'s 8-bit (7 stored + implicit) significand is a 16-bit
//! margin, far past that bound, so `round_bf16(round_f32(a + b)) ==
//! round_bf16(a + b)` for any two `bf16`-representable `a`, `b` — no
//! double-rounding divergence is possible at this precision gap. The CPU
//! arm therefore reproduces `half::bf16::Add` (hence candle's `badd_bf16`)
//! bit-for-bit using the widen-add-round idiom, matching this crate's own
//! established precedent for the identical situation
//! ([`super::ScaledCastAdd`]'s `(BF16, F32)` CPU arm, whose own module doc
//! makes the same claim and is backed by an on-device oracle,
//! `tests/scaled_cast_add_oracles.rs`).
//!
//! **A real, discovered divergence this file's own tests route around:**
//! candle-core-0.11.0's CPU `Tensor::add` for `BF16`, on any host compiled
//! with `target_feature = "neon"` (this crate's own arm64 dev/CI hosts
//! included) WITHOUT the separate `target_feature = "bf16"` dot-product
//! path, dispatches through `VecOps::vec_add` -> `CurrentCpuBF16`
//! (`candle-core-0.11.0/src/cpu/neon.rs`'s first `inner` module) for every
//! element a `STEP = 32`-wide vectorized loop covers; its `vec_store`
//! narrows the `f32` SIMD sum back to `bf16` via `vshrn_n_u32::<16>` — a
//! bare mantissa TRUNCATION (round toward zero), NOT round-to-nearest-even
//! — only the `n % 32` scalar "leftover" elements take the correctly-
//! rounded `half::bf16::Add` path this file's own arm matches. This means
//! a literal `Tensor::add` at production width (`n >> 32`) is NOT a
//! portable oracle (family J: its bit pattern depends on the HOST
//! compiler's `target_feature` flags, not just the math) and is NOT what
//! this op's own correctness target is anyway — CUDA's `badd_bf16` is a
//! correctly-rounded hardware bf16 add, which this crate's round-to-
//! nearest CPU arm matches and candle's own NEON-truncating CPU path does
//! not. This file's tests therefore compare the production-amplitude
//! fixture against a HAND-COMPUTED round-to-nearest reference (portable,
//! and the mathematically-correct target), with a separate small-`n` test
//! (`n = 8`, under `STEP`) proving bit-exactness against a REAL
//! `Tensor::add` call on the scalar (round-to-nearest) path.
//!
//! Traffic: old = cast(read 4B + write 2B) + badd(read 2B + read 2B + write
//! 2B) = 12 B/elem; new = read `f32val` 4B + read `base` 2B + write `out`
//! 2B = 8 B/elem (design study §1's (f), credit 4 B/elem).
//!
//! Domain (family D): `base` MUST be `BF16`, `f32val` MUST be `F32`, same
//! shape (not broadcasting). The `base_dtype == F32` case in
//! `LowRankResidualLinear::bwd` (`dx = dx_base + d_x_lora`, both already
//! `F32`) is already a single candle `badd_f32` launch with nothing to
//! fuse, and is NOT this op's domain — the call site keeps using a plain
//! `Tensor::add` for it. The CUDA arm additionally REQUIRES contiguous
//! storage for both operands, same as [`CastScaleBf16F32`] above.
//!
//! ## Contiguity at the call site — why the CUDA refusal is defense in
//! ## depth, not a live production path
//!
//! Neither op's CUDA arm falls back to the two-kernel eager chain on a
//! non-contiguous input the way a CALL SITE'S OWN admission predicate can
//! (e.g. `LoraLinear::forward`'s `lora_linear_admission_predicate`) — it
//! returns `Error::RequiresContiguous`, matching every other elementwise op
//! in this crate ([`super::Axpy`]/[`super::ScaledCastAdd`]'s documented
//! CUDA domain: "additionally requires contiguous storage"). `bwd` itself
//! has no eager alternative to fall back TO at this point (unlike the
//! outer `LowRankResidualLinear` site, which decided fused-vs-eager BEFORE
//! `bwd` ever ran) — a `RequiresContiguous` here would fail the whole
//! backward pass, not degrade gracefully, so it matters that this path is
//! never actually reachable in production: `CastScaleBf16F32`'s only
//! argument is `grad_res`, the raw upstream gradient `CustomOp3::bwd`
//! receives — candle's own `GradStore::or_insert`/`insert`
//! (`backprop.rs`) accumulates gradients via `zeros_like` + `add`, both of
//! which allocate fresh, contiguous storage, so `grad_res` is contiguous
//! by construction every time `bwd` is invoked through candle's normal
//! `Tensor::backward` walk. `CastAddBf16`'s two arguments are
//! `dx_base_2d` (a FRESH `Tensor::matmul` output — `BackendStorage::matmul`
//! always allocates a new contiguous buffer, `cuda_backend/mod.rs`'s
//! `dev.alloc::<T>(elem_count)` calls) and `d_x_lora_f32_2d` (either that
//! same fresh-GEMM shape or [`super::DropoutFused`]'s own kernel output,
//! also freshly allocated) — neither is ever a narrowed/transposed VIEW.
//! `RequiresContiguous` is therefore reachable only if a future change to
//! `bwd` threads a genuinely non-contiguous tensor through one of these two
//! slots; kept as a typed refusal (not a panic, not silently misread
//! strides) rather than trusted away, per this crate's usual "an op trusts
//! no caller for its own domain" doctrine — but not exercised by any
//! oracle in this file, since production never reaches it.
//!
//! Both ops dispatch through [`crate::admission::admit`] with their own
//! [`crate::admission::DispatchCounters`] key (`op.name()`), so
//! `JAMMI_KERNELS_DISABLE=cast_scale_bf16_f32` / `cast_add_bf16` force the
//! ORIGINAL two-kernel eager composition back on for a same-build forced
//! A/B, and a zero-dispatch run is observable (never silently green).

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp1, CustomOp2, DType, Error, Layout, Result, Shape, Tensor};
use half::bf16;

use crate::layout_walk::StridedOffsets;

/// `out = f32(x) * scale + 0.0`, `x` required `BF16`. See the module doc
/// for why the fixed `+ 0.0` term is load-bearing (signed-zero identity,
/// matching `affine_f32`'s own expression) and why this op's domain is
/// `BF16`-only rather than accepting `F32` too (nothing to fuse there).
///
/// STATELESS BY CONSTRUCTION (`Copy`, see `ops`'s module doc): `scale` is
/// construction data, matching [`super::Axpy`]/[`super::ScaledCastAdd`]'s
/// own `f64`-typed scalar field.
#[derive(Debug, Clone, Copy)]
pub struct CastScaleBf16F32 {
    pub scale: f64,
}

impl CastScaleBf16F32 {
    pub fn new(scale: f64) -> Self {
        Self { scale }
    }
}

impl super::sealed::Sealed for CastScaleBf16F32 {}

impl CustomOp1 for CastScaleBf16F32 {
    fn name(&self) -> &'static str {
        "cast_scale_bf16_f32"
    }

    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout) -> Result<(CpuStorage, Shape)> {
        match s1 {
            CpuStorage::BF16(x) => {
                let out = cast_scale_bf16_f32(self.scale, x, l1);
                Ok((CpuStorage::F32(out), l1.shape().clone()))
            }
            s1 => Err(Error::UnsupportedDTypeForOp(s1.dtype(), self.name())),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        crate::cuda::cast_scale::cuda_fwd_cast_scale_bf16_f32(self.scale, s1, l1)
    }

    /// `d_x = cast_to(x.dtype())(grad_res.affine(scale, 0.0))` — the chain
    /// rule through the (straight-through, matching
    /// [`super::ScaledCastAdd::bwd`]'s convention) round and the scalar
    /// multiply/widening-cast pair this op's forward composes. Dead in
    /// `LowRankResidualLinear::bwd`'s own usage today (its `grad_res`
    /// argument is an untracked upstream-gradient tensor, never itself
    /// re-differentiated — see that op's own module doc, "Tensor-level"),
    /// implemented anyway for the same reason
    /// [`super::Axpy::bwd`]/[`super::ScaledCastAdd::bwd`] always return
    /// `Some`: this is a generic primitive (family L), not guaranteed to
    /// stay behind an untracked call site forever.
    fn bwd(&self, arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        let d = grad_res.affine(self.scale, 0.0)?;
        let d = if arg.dtype() == d.dtype() {
            d
        } else {
            d.to_dtype(arg.dtype())?
        };
        Ok(Some(d))
    }
}

/// Fixed fold order (family J): `StridedOffsets` walked in the same
/// sequence for a given layout every time.
fn cast_scale_bf16_f32(scale: f64, x: &[bf16], lx: &Layout) -> Vec<f32> {
    let scale = scale as f32;
    StridedOffsets::from_layout(lx)
        .map(|ix| x[ix].to_f32() * scale + 0.0f32)
        .collect()
}

/// `out = base + round_to_bf16(f32val)`, `base` required `BF16`, `f32val`
/// required `F32`, identical shape (not broadcasting). See the module doc
/// for the rounding-order requirement (round f32val to bf16 in-register
/// FIRST, then add) and the double-rounding-safety argument backing the CPU
/// arm's widen-add-round idiom.
///
/// STATELESS BY CONSTRUCTION (`Copy`): no construction data at all — unlike
/// [`CastScaleBf16F32`]/[`super::Axpy`], this op has no scalar field, only
/// the fixed cast-then-add expression.
#[derive(Debug, Clone, Copy, Default)]
pub struct CastAddBf16;

impl CastAddBf16 {
    pub fn new() -> Self {
        Self
    }
}

impl super::sealed::Sealed for CastAddBf16 {}

impl CustomOp2 for CastAddBf16 {
    fn name(&self) -> &'static str {
        "cast_add_bf16"
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
        match (s1, s2) {
            (CpuStorage::BF16(base), CpuStorage::F32(f32val)) => {
                let out = cast_add_bf16(base, l1, f32val, l2);
                Ok((CpuStorage::BF16(out), l1.shape().clone()))
            }
            (s1, s2) => {
                if s1.dtype() != DType::BF16 {
                    Err(Error::UnsupportedDTypeForOp(s1.dtype(), self.name()))
                } else {
                    Err(Error::UnsupportedDTypeForOp(s2.dtype(), self.name()))
                }
            }
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
        crate::cuda::cast_scale::cuda_fwd_cast_add_bf16(s1, l1, s2, l2)
    }

    /// `d_base = dy` (identity — `base` enters the sum unrounded, matching
    /// [`super::Axpy::bwd`]'s "the add is linear with unit coefficient"
    /// convention); `d_f32val = cast_to(F32)(dy)` — the chain rule through
    /// the (straight-through) round, matching
    /// [`super::ScaledCastAdd::bwd`]'s `d_lora` cast. Dead in
    /// `LowRankResidualLinear::bwd`'s own usage today for the same reason
    /// [`CastScaleBf16F32::bwd`]'s doc states; implemented for completeness
    /// as a generic primitive (family L).
    fn bwd(
        &self,
        _base: &Tensor,
        f32val: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let d_base = grad_res.clone();
        let d_f32val = if grad_res.dtype() == f32val.dtype() {
            grad_res.clone()
        } else {
            grad_res.to_dtype(f32val.dtype())?
        };
        Ok((Some(d_base), Some(d_f32val)))
    }
}

/// Fixed fold order (family J), and the double-rounding-safety argument the
/// module doc states: widen both `bf16` operands to `f32`, add, round once
/// — bit-identical to a genuine single-rounding narrow add at this
/// precision gap, and to `half::bf16::Add` (hence candle's own CPU
/// `badd_bf16`) by construction, since that is exactly what `half::bf16`'s
/// `Add` impl does.
fn cast_add_bf16(base: &[bf16], lb: &Layout, f32val: &[f32], lf: &Layout) -> Vec<bf16> {
    StridedOffsets::from_layout(lb)
        .zip(StridedOffsets::from_layout(lf))
        .map(|(ib, il)| {
            let delta = bf16::from_f32(f32val[il]);
            base[ib] + delta
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn cast_scale(scale: f64, x: &Tensor) -> Result<Tensor> {
        crate::ops::apply1(x, CastScaleBf16F32::new(scale))
    }

    fn cast_add(base: &Tensor, f32val: &Tensor) -> Result<Tensor> {
        crate::ops::apply2(base, f32val, CastAddBf16::new())
    }

    // ---- CastScaleBf16F32 ----

    #[test]
    fn cast_scale_op_name_is_pinned() {
        assert_eq!(CastScaleBf16F32::new(1.0).name(), "cast_scale_bf16_f32");
    }

    #[test]
    fn cast_scale_bwd_matches_a_hand_computed_straight_through_gradient() {
        // Dead in `LowRankResidualLinear::bwd`'s own usage (see this op's
        // `bwd` doc), but implemented for completeness as a generic
        // primitive — exercised directly here (not through `apply1`+
        // `backward()`, since `grad_res` in the real call site is never
        // itself tracked) to close the mutation surface `bwd` otherwise
        // leaves untested (MUT-1 discipline: every function this crate
        // ships gets a covering test, not only the ones a current call
        // site happens to reach).
        let device = Device::Cpu;
        let arg = Tensor::from_slice(&[bf16::from_f32(1.0), bf16::from_f32(-2.0)], (2,), &device)
            .unwrap();
        let res = Tensor::from_slice(&[0.0f32, 0.0], (2,), &device).unwrap();
        let grad_res = Tensor::from_slice(&[4.0f32, -8.0], (2,), &device).unwrap();
        let op = CastScaleBf16F32::new(0.5);
        let d = CustomOp1::bwd(&op, &arg, &res, &grad_res)
            .unwrap()
            .expect("bwd must return Some");
        assert_eq!(
            d.dtype(),
            DType::BF16,
            "d must be cast back to arg's own dtype"
        );
        let got: Vec<bf16> = d.to_vec1().unwrap();
        // grad_res.affine(0.5, 0.0) = [2.0, -4.0], cast to bf16 (exact at
        // these magnitudes).
        assert_eq!(got, [bf16::from_f32(2.0), bf16::from_f32(-4.0)]);
    }

    #[test]
    fn cast_scale_bwd_is_identity_when_arg_is_already_f32() {
        // Isolates the `arg.dtype() == d.dtype()` branch (the `==` -> `!=`
        // mutation): when `arg` is `F32` (matching `d`'s own dtype, since
        // this op's forward always produces `F32`), `bwd` must return `d`
        // UNCHANGED (no `to_dtype` round-trip), not a cast that this
        // fixture's exact-integer values would otherwise hide.
        let device = Device::Cpu;
        let arg = Tensor::from_slice(&[1.0f32, -2.0], (2,), &device).unwrap();
        let res = Tensor::from_slice(&[0.0f32, 0.0], (2,), &device).unwrap();
        let grad_res = Tensor::from_slice(&[4.0f32, -8.0], (2,), &device).unwrap();
        let op = CastScaleBf16F32::new(0.5);
        let d = CustomOp1::bwd(&op, &arg, &res, &grad_res)
            .unwrap()
            .expect("bwd must return Some");
        assert_eq!(d.dtype(), DType::F32);
        let got: Vec<f32> = d.to_vec1().unwrap();
        assert_eq!(got, [2.0, -4.0]);
    }

    #[test]
    fn cpu_fwd_matches_hand_computed_values() {
        let device = Device::Cpu;
        let xv = [
            bf16::from_f32(1.5),
            bf16::from_f32(-2.25),
            bf16::from_f32(0.0),
        ];
        let x = Tensor::from_slice(&xv, (3,), &device).unwrap();
        let out = cast_scale(2.0, &x).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(out.as_slice(), [3.0f32, -4.5, 0.0]);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn bit_identical_to_the_eager_two_kernel_chain_at_production_amplitude() {
        // Production width (>= 4096 elements, guide §3.4) and amplitude
        // spanning the range this op's real caller sees: base-residual-like
        // magnitudes up to ~6.7e3 (esc-045's own layer-18 residual
        // magnitude, the real checkpoint measurement this fixture's
        // amplitude is sized from), exact zeros, negative zeros, and
        // subnormals.
        let device = Device::Cpu;
        let n = 4096usize;
        let mut xv: Vec<bf16> = (0..n)
            .map(|i| {
                let v = ((i as f32 * 0.017).sin()) * 6700.0;
                bf16::from_f32(v)
            })
            .collect();
        // Force a few boundary values explicitly (family D: an oracle keyed
        // on production shape AND amplitude, not decoration).
        xv[0] = bf16::from_f32(0.0);
        xv[1] = bf16::from_f32(-0.0);
        xv[2] = bf16::from_bits(0x0001); // smallest positive subnormal
        xv[3] = bf16::from_bits(0x8001); // smallest negative subnormal
        xv[4] = bf16::from_f32(f32::MIN_POSITIVE);

        let x = Tensor::from_slice(&xv, (n,), &device).unwrap();
        let scale = 0.11048_f64; // alpha/rank-shaped, non-power-of-two.

        let fused = cast_scale(scale, &x).unwrap().to_vec1::<f32>().unwrap();
        let eager = x
            .to_dtype(DType::F32)
            .unwrap()
            .affine(scale, 0.0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        // Finiteness-affirmative FIRST (guide §3.7): count non-finite
        // elements in both arms before any bit comparison — a NaN must
        // fail, never silently pass a `!=` check.
        let fused_finite = fused.iter().filter(|v| v.is_finite()).count();
        let eager_finite = eager.iter().filter(|v| v.is_finite()).count();
        assert_eq!(fused_finite, n, "fused arm produced a non-finite element");
        assert_eq!(eager_finite, n, "eager arm produced a non-finite element");

        for i in 0..n {
            assert!(
                fused[i].to_bits() == eager[i].to_bits(),
                "index {i}: fused {} (bits {:#010x}) != eager {} (bits {:#010x})",
                fused[i],
                fused[i].to_bits(),
                eager[i],
                eager[i].to_bits()
            );
        }
    }

    #[test]
    fn red_control_a_missing_plus_zero_diverges_on_negative_zero() {
        // Non-vacuity (guide §3.7/§3.8, family F): a deliberately WRONG
        // expression (`x * scale` without `+ 0.0f`) must diverge from the
        // real kernel on an input containing -0.0 — proving the assertion
        // above is not vacuously true regardless of the `+ 0.0` term.
        let scale = 3.0_f32;
        let neg_zero = bf16::from_f32(-0.0);
        let correct = neg_zero.to_f32() * scale + 0.0f32;
        let wrong = neg_zero.to_f32() * scale; // the RED control expression
        assert_eq!(correct.to_bits(), 0f32.to_bits(), "correct must be +0.0");
        assert_eq!(
            wrong.to_bits(),
            (-0.0f32).to_bits(),
            "wrong must stay -0.0 (the divergence this control proves is real)"
        );
        assert_ne!(
            correct.to_bits(),
            wrong.to_bits(),
            "RED control did not diverge — the +0.0 oracle would be vacuous"
        );
    }

    #[test]
    fn empty_tensor_is_a_no_op_not_an_error() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[] as &[bf16], (0,), &device).unwrap();
        let out = cast_scale(3.0, &x).unwrap().to_vec1::<f32>().unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn f32_input_is_refused_with_a_typed_error() {
        // This op's domain is BF16-only (see module doc: an F32 input has
        // nothing to fuse) — never a silent reinterpretation of F32 bytes.
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, 2.0], (2,), &device).unwrap();
        let err = cast_scale(1.0, &x).expect_err("F32 input must be refused, not silently cast");
        assert!(matches!(err, Error::UnsupportedDTypeForOp(DType::F32, ..)));
    }

    #[test]
    fn non_contiguous_view_is_still_correct_on_cpu() {
        let device = Device::Cpu;
        let xv: Vec<bf16> = (1..=6).map(|i| bf16::from_f32(i as f32)).collect();
        let x = Tensor::from_slice(&xv, (2, 3), &device)
            .unwrap()
            .t()
            .unwrap();
        assert!(!x.is_contiguous());
        let out = cast_scale(2.0, &x).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(out, vec![vec![2.0, 8.0], vec![4.0, 10.0], vec![6.0, 12.0]]);
    }

    #[test]
    fn dispatch_growth_across_a_depth_sweep_stays_bounded_by_r1() {
        // Guide §3.2: key the oracle on GROWTH across a depth sweep, keyed
        // against the SAME run's own r(1), never an absolute ULP constant.
        // Each "layer" repeats the identical fused-vs-eager comparison at
        // an INDEPENDENT random-ish input (deterministic fixture); since
        // this op is bit-exact per element, r(L) is exactly 0 for every L
        // (there is no accumulation to grow), which is itself the growth
        // bound holding with C = 0 (the strongest possible case, not a
        // vacuous check — r(1) is computed from a genuinely nonzero
        // reference sum, not skipped).
        let device = Device::Cpu;
        let mut r1 = None;
        for depth in [1usize, 4, 8, 28] {
            let n = 512usize;
            let xv: Vec<bf16> = (0..n)
                .map(|i| bf16::from_f32(((i * (depth + 1)) as f32 * 0.031).cos() * 100.0))
                .collect();
            let x = Tensor::from_slice(&xv, (n,), &device).unwrap();
            let scale = 0.0625_f64;
            let fused = cast_scale(scale, &x).unwrap().to_vec1::<f32>().unwrap();
            let eager = x
                .to_dtype(DType::F32)
                .unwrap()
                .affine(scale, 0.0)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let mut abs_diff = 0.0f64;
            let mut abs_ref = 0.0f64;
            for i in 0..n {
                abs_diff += f64::from((fused[i] - eager[i]).abs());
                abs_ref += f64::from(eager[i].abs());
            }
            assert!(
                abs_ref > 0.0,
                "degenerate reference sum — not a real oracle"
            );
            let r = abs_diff / abs_ref;
            assert!(
                r.is_finite() && r == 0.0,
                "depth {depth}: r={r}, expected bit-exact 0.0"
            );
            if depth == 1 {
                r1 = Some(r);
            } else {
                let r1 = r1.expect("r(1) computed first");
                // No absolute ULP floor (guide §3.8): the bound is r(1)
                // itself, not an added constant — this op is exactly
                // bit-exact, so r(1) == 0.0 and the bound is r <= 0.0.
                assert!(r <= r1, "depth {depth}: r={r} grew past r(1)={r1}");
            }
        }
    }

    // ---- CastAddBf16 ----

    #[test]
    fn cast_add_op_name_is_pinned() {
        assert_eq!(CastAddBf16::new().name(), "cast_add_bf16");
    }

    #[test]
    fn cast_add_bwd_matches_a_hand_computed_straight_through_gradient() {
        // Dead in `LowRankResidualLinear::bwd`'s own usage (see this op's
        // `bwd` doc); exercised directly for the same MUT-1 reason
        // `cast_scale_bwd_matches_a_hand_computed_straight_through_gradient`
        // states.
        let device = Device::Cpu;
        let base = Tensor::from_slice(&[bf16::from_f32(1.0), bf16::from_f32(-2.0)], (2,), &device)
            .unwrap();
        let f32val = Tensor::from_slice(&[0.5f32, 3.25], (2,), &device).unwrap();
        let res =
            Tensor::from_slice(&[bf16::from_f32(0.0), bf16::from_f32(0.0)], (2,), &device).unwrap();
        let grad_res =
            Tensor::from_slice(&[bf16::from_f32(4.0), bf16::from_f32(-8.0)], (2,), &device)
                .unwrap();
        let op = CastAddBf16::new();
        let (d_base, d_f32val) = CustomOp2::bwd(&op, &base, &f32val, &res, &grad_res).unwrap();
        let d_base = d_base.expect("d_base must be Some");
        let d_f32val = d_f32val.expect("d_f32val must be Some");
        assert_eq!(
            d_base.dtype(),
            DType::BF16,
            "d_base is identity — base's own dtype"
        );
        assert_eq!(
            d_f32val.dtype(),
            DType::F32,
            "d_f32val is cast to f32val's own dtype"
        );
        let got_base: Vec<bf16> = d_base.to_vec1().unwrap();
        let got_f32val: Vec<f32> = d_f32val.to_vec1().unwrap();
        assert_eq!(got_base, [bf16::from_f32(4.0), bf16::from_f32(-8.0)]);
        assert_eq!(got_f32val, [4.0, -8.0]);
    }

    #[test]
    fn cast_add_bwd_is_identity_when_grad_res_already_matches_f32val_dtype() {
        // Isolates the `grad_res.dtype() == f32val.dtype()` branch (the
        // `==` -> `!=` mutation): both `f32val` and `grad_res` are `F32`
        // here, so `d_f32val` must be `grad_res` UNCHANGED (no `to_dtype`
        // round-trip).
        let device = Device::Cpu;
        let base = Tensor::from_slice(&[bf16::from_f32(1.0), bf16::from_f32(-2.0)], (2,), &device)
            .unwrap();
        let f32val = Tensor::from_slice(&[0.5f32, 3.25], (2,), &device).unwrap();
        let res = base.clone();
        // A `CustomOp2::bwd` signature requires `grad_res` to match `res`'s
        // OWN dtype in the real candle call graph, but this op's `bwd` is
        // exercised directly (white-box) here — an `F32` `grad_res` is
        // exactly the branch under test, matching the doc's own "chain
        // rule through the round" derivation for a caller that reaches
        // this op with an already-`F32` upstream gradient.
        let grad_res = Tensor::from_slice(&[4.0f32, -8.0], (2,), &device).unwrap();
        let op = CastAddBf16::new();
        let (_d_base, d_f32val) = CustomOp2::bwd(&op, &base, &f32val, &res, &grad_res).unwrap();
        let d_f32val = d_f32val.expect("d_f32val must be Some");
        assert_eq!(d_f32val.dtype(), DType::F32);
        let got: Vec<f32> = d_f32val.to_vec1().unwrap();
        assert_eq!(got, [4.0, -8.0]);
    }

    #[test]
    fn cast_add_dtype_error_reports_the_actual_offending_argument() {
        // Discriminates the `s1.dtype() != DType::BF16` branch (the `!=`
        // -> `==` mutation): `base` (s1) is the invalid one (`F32`, not
        // `BF16`) while `f32val` (s2) is ALSO invalid but at a DIFFERENT
        // dtype (`BF16`, not `F32`) — a mutated `==` would report `s2`'s
        // dtype (`BF16`) instead of `s1`'s own (`F32`), distinguishably
        // wrong (unlike a same-dtype fixture, where both branches report
        // the same value and the mutation is invisible).
        let device = Device::Cpu;
        let base = Tensor::from_slice(&[1.0f32, 2.0], (2,), &device).unwrap();
        let f32val =
            Tensor::from_slice(&[bf16::from_f32(1.0), bf16::from_f32(2.0)], (2,), &device).unwrap();
        let err = cast_add(&base, &f32val)
            .expect_err("both dtypes invalid — must refuse, reporting base's own dtype");
        match err {
            Error::UnsupportedDTypeForOp(dtype, op) => {
                assert_eq!(
                    dtype,
                    DType::F32,
                    "must report base's (s1's) own offending dtype"
                );
                assert_eq!(op, "cast_add_bf16");
            }
            other => {
                panic!("expected UnsupportedDTypeForOp(F32, \"cast_add_bf16\"), got {other:?}")
            }
        }
    }

    #[test]
    fn cast_add_cpu_fwd_matches_hand_computed_values() {
        let device = Device::Cpu;
        let base = Tensor::from_slice(&[bf16::from_f32(1.0), bf16::from_f32(-2.0)], (2,), &device)
            .unwrap();
        let f32val = Tensor::from_slice(&[0.5f32, 3.25], (2,), &device).unwrap();
        let out = cast_add(&base, &f32val).unwrap().to_vec1::<bf16>().unwrap();
        let expected = [
            bf16::from_f32(1.0) + bf16::from_f32(0.5),
            bf16::from_f32(-2.0) + bf16::from_f32(3.25),
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn cast_add_bit_identical_to_the_eager_two_kernel_chain_at_production_amplitude() {
        let device = Device::Cpu;
        let n = 4096usize;
        let mut base_v: Vec<bf16> = (0..n)
            .map(|i| bf16::from_f32(((i as f32 * 0.013).cos()) * 6700.0))
            .collect();
        let mut f32_v: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.029).sin()) * 3.7).collect();
        base_v[0] = bf16::from_f32(0.0);
        base_v[1] = bf16::from_f32(-0.0);
        f32_v[0] = 0.0;
        f32_v[1] = -0.0;
        f32_v[2] = f32::MIN_POSITIVE;
        f32_v[3] = -f32::MIN_POSITIVE;
        f32_v[4] = f32::from_bits(1); // smallest positive f32 subnormal.

        let base = Tensor::from_slice(&base_v, (n,), &device).unwrap();
        let f32val = Tensor::from_slice(&f32_v, (n,), &device).unwrap();

        let fused = cast_add(&base, &f32val).unwrap().to_vec1::<bf16>().unwrap();
        // "Eager" here is the PORTABLE mathematical definition of `cast to
        // bf16, then add` — round-to-nearest-even at both steps — NOT a
        // literal `Tensor::add` call. Deliberately: candle-core-0.11.0's
        // CPU `Tensor::add` for `BF16` on an ARM host with
        // `target_feature = "neon"` (this crate's own CI/dev hosts
        // included) dispatches through `VecOps::vec_add`'s
        // `CurrentCpuBF16` path (`candle-core-0.11.0/src/cpu/neon.rs`'s
        // `inner` module gated `#[cfg(not(target_feature = "bf16"))]`,
        // `vec_store`), which narrows each `f32` sum back to `bf16` via
        // `vshrn_n_u32::<16>` — a bare mantissa TRUNCATION (round toward
        // zero), not round-to-nearest-even — for every element the
        // `STEP = 32`-wide vectorized loop covers (only genuinely
        // "leftover" elements, `n % 32`, fall through to the scalar
        // `half::bf16::Add` path, which IS round-to-nearest). At this
        // test's production width (`n = 4096`, `>> 32`), comparing against
        // a literal `Tensor::add` would make the "reference" depend on
        // this HOST'S compile-time `target_feature` flags — not a portable
        // oracle (family J), and not what production cares about anyway:
        // the real target is CUDA's `badd_bf16` (a hardware bf16 add,
        // correctly-rounded per IEEE 754), which this hand-computed
        // round-to-nearest reference matches, and candle's own NEON
        // truncation does not. A SEPARATE small-`n` test below (`n = 8`,
        // under `STEP`) exercises the ACTUAL `Tensor::add` composition
        // end-to-end, safely on the scalar (round-to-nearest) path.
        let eager: Vec<bf16> = base_v
            .iter()
            .zip(f32_v.iter())
            .map(|(&b, &f)| b + bf16::from_f32(f))
            .collect();

        let fused_finite = fused.iter().filter(|v| v.to_f32().is_finite()).count();
        let eager_finite = eager.iter().filter(|v| v.to_f32().is_finite()).count();
        assert_eq!(fused_finite, n, "fused arm produced a non-finite element");
        assert_eq!(eager_finite, n, "eager arm produced a non-finite element");

        for i in 0..n {
            assert!(
                fused[i].to_bits() == eager[i].to_bits(),
                "index {i}: fused {} (bits {:#06x}) != eager {} (bits {:#06x})",
                fused[i],
                fused[i].to_bits(),
                eager[i],
                eager[i].to_bits()
            );
        }
    }

    #[test]
    fn cast_add_bit_identical_to_a_real_tensor_add_composition_below_the_simd_step() {
        // The companion to the production-amplitude test above: `n = 8`
        // stays safely under `CurrentCpuBF16::STEP = 32`
        // (`candle-core-0.11.0/src/cpu/neon.rs`), so `Tensor::add`'s own
        // dispatch NEVER enters the vectorized (truncating) loop — every
        // element takes the scalar `half::bf16::Add` leftover path, which
        // IS round-to-nearest-even. This exercises the ACTUAL two-kernel
        // eager composition (`to_dtype` then a real `Tensor::add` call),
        // not the hand-derived reference the larger test uses.
        let device = Device::Cpu;
        let base_v: Vec<bf16> = (0..8)
            .map(|i| bf16::from_f32(((i as f32) * 0.7 - 2.1).cos() * 40.0))
            .collect();
        let f32_v: Vec<f32> = (0..8).map(|i| ((i as f32) * 1.3).sin() * 2.5).collect();

        let base = Tensor::from_slice(&base_v, (8,), &device).unwrap();
        let f32val = Tensor::from_slice(&f32_v, (8,), &device).unwrap();

        let fused = cast_add(&base, &f32val).unwrap().to_vec1::<bf16>().unwrap();
        let eager = (&base + &f32val.to_dtype(DType::BF16).unwrap())
            .unwrap()
            .to_vec1::<bf16>()
            .unwrap();

        for i in 0..8 {
            assert!(
                fused[i].to_bits() == eager[i].to_bits(),
                "index {i}: fused {} (bits {:#06x}) != eager {} (bits {:#06x})",
                fused[i],
                fused[i].to_bits(),
                eager[i],
                eager[i].to_bits()
            );
        }
    }

    #[test]
    fn red_control_rounding_after_the_add_instead_of_before_diverges() {
        // Non-vacuity: a deliberately mis-ordered reference (accumulate in
        // f32 THEN round once, instead of rounding f32val to bf16 FIRST)
        // must diverge from the real kernel on values that land exactly
        // between two representable bf16 values only when combined in one
        // order vs the other.
        let base = bf16::from_f32(1.0);
        // Just above the round-to-even halfway point between bf16(0.0) and
        // bf16(2^-8) = 0.00390625: rounds f32val UP to 0.00390625 alone,
        // but the SUM `1.0 + f32val` (computed at full f32 precision, i.e.
        // NOT pre-rounded) is close enough to 1.0 that round-to-nearest
        // picks the bf16 ABOVE 1.0 (1.0078125) instead — a genuine
        // divergence between "round f32val first, then add" and
        // "accumulate in f32, round once" (verified by an independent
        // Python re-derivation of IEEE 754 round-to-nearest-even at this
        // exact value, not by trial and error against this crate's own
        // code).
        let f32val = 0.003_907_204_f32;
        let correct = base + bf16::from_f32(f32val); // round f32val to bf16 FIRST, then add.
        let wrong = bf16::from_f32(base.to_f32() + f32val); // accumulate-then-round-once.
        assert_ne!(
            correct.to_bits(),
            wrong.to_bits(),
            "RED control did not diverge — the rounding-order oracle would be vacuous"
        );
    }

    #[test]
    fn cast_add_empty_tensor_is_a_no_op_not_an_error() {
        let device = Device::Cpu;
        let base = Tensor::from_slice(&[] as &[bf16], (0,), &device).unwrap();
        let f32val = Tensor::from_slice(&[] as &[f32], (0,), &device).unwrap();
        let out = cast_add(&base, &f32val).unwrap().to_vec1::<bf16>().unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn shape_mismatch_is_refused_not_broadcast() {
        let device = Device::Cpu;
        let base =
            Tensor::from_slice(&[bf16::from_f32(1.0), bf16::from_f32(2.0)], (2,), &device).unwrap();
        let f32val = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (3,), &device).unwrap();
        let err =
            cast_add(&base, &f32val).expect_err("mismatched shapes must not silently broadcast");
        assert!(matches!(err, Error::ShapeMismatchBinaryOp { .. }));
    }

    #[test]
    fn f32_base_is_refused_with_a_typed_error() {
        let device = Device::Cpu;
        let base = Tensor::from_slice(&[1.0f32, 2.0], (2,), &device).unwrap();
        let f32val = Tensor::from_slice(&[1.0f32, 2.0], (2,), &device).unwrap();
        let err =
            cast_add(&base, &f32val).expect_err("F32 base must be refused, not treated as BF16");
        assert!(matches!(err, Error::UnsupportedDTypeForOp(DType::F32, ..)));
    }

    #[test]
    fn cast_add_non_contiguous_view_is_still_correct_on_cpu() {
        let device = Device::Cpu;
        let base_v: Vec<bf16> = (1..=6).map(|i| bf16::from_f32(i as f32)).collect();
        let base = Tensor::from_slice(&base_v, (2, 3), &device)
            .unwrap()
            .t()
            .unwrap();
        assert!(!base.is_contiguous());
        let f32val = Tensor::from_slice(&[0.0f32; 6], (3, 2), &device).unwrap();
        let out = cast_add(&base, &f32val).unwrap().to_vec2::<bf16>().unwrap();
        assert_eq!(
            out,
            vec![
                vec![bf16::from_f32(1.0), bf16::from_f32(4.0)],
                vec![bf16::from_f32(2.0), bf16::from_f32(5.0)],
                vec![bf16::from_f32(3.0), bf16::from_f32(6.0)],
            ]
        );
    }
}
