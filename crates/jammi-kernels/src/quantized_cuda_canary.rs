//! The quantized-CUDA load-time canary (issue #434): a known-answer check
//! run lazily before the FIRST quantized CUDA matmul this process performs
//! — the engine-side guard for the shipped `candle-kernels` 0.11 cu12
//! wheel's own packaging defect (proven on a live H100 pod; see issue
//! #434's root-cause comment for the full localization matrix), not merely
//! the CI image that happened to reproduce it.
//!
//! ## The failure class this guards against
//!
//! `candle-kernels` 0.11 compiles its quantized fast-path kernels
//! (`fast_mmvq`/`fast_mmq`, `QCudaStorage::fwd`'s FIRST-tried arm,
//! `candle-core` 0.11.0 `quantized/cuda.rs:846-860`) as SINGLE-ARCH SASS —
//! no PTX — selected at BUILD time by `CUDA_COMPUTE_CAP`. A cap-80 build's
//! cubins load on sm_86/sm_89 (same major, forward-compatible) but CANNOT
//! load on sm_90 (H100): `cuobjdump` on the shipped `cu12` wheel showed 15
//! cubins, all `sm_80`. The raw `<<<...>>>` launchers in candle-kernels
//! never call `cudaGetLastError` — a launch that fails with `209`
//! (`cudaErrorNoKernelImageForDevice`) fails SILENTLY, and the caller then
//! reads back an uninitialized `dev.alloc` buffer: deterministic-per-
//! allocation garbage (`O(10)`-`O(1e38)` observed on the pod, reproduced to
//! the digit against CI's own numbers), not an `Err`.
//!
//! `candle_core::quantized::cuda::set_force_dmmv(true)` routes
//! `QCudaStorage::fwd` around BOTH fast arms entirely (`cuda.rs:853`) onto
//! the legacy, PTX-JIT'd dequantize-then-matvec path — proven correct on
//! the mismatched build (issue #434: measured `0.0288`, ordinary
//! quantization error, vs `18.6` garbage on the broken fast path). Every
//! OTHER kernel this workspace ships (dense matmul, dequantize,
//! `jammi-kernels`'s own fused ops) is PTX-JIT'd already and unaffected by
//! any of this — this module exists ONLY to gate the quantized fast path.
//!
//! ## What this guard can, and cannot, detect
//!
//! It proves ONE known-answer quantized-CUDA matmul, on ONE device, at
//! whatever moment the first production call reaches it (this is a
//! load-time canary, not a per-call check — see "cache granularity"
//! below). It CANNOT: detect a device that develops a fault AFTER passing;
//! distinguish "SASS arch mismatch" from any OTHER cause that would make
//! this exact known-answer case wrong (a candle upstream regression in the
//! fast kernels would ALSO be caught here, indistinguishably, and routed
//! to the same DMMV fallback — the conservative, correct response either
//! way: K2, refusal/fallback beats a confident wrong number); or prove
//! that EVERY shape/dtype this workspace's quantized matmul ever sees is
//! safe — only that THIS ONE fixed case is. The true sm_90-cap_80-mismatch
//! scenario this guard was written against was proven end-to-end on a live
//! H100 pod per issue #434's root-cause comment; this crate's own
//! CUDA-gated test (`quantized_cuda_canary_passes_on_a_healthy_build_and_device`,
//! `tests/cuda_parity.rs`) proves only that the canary's OWN mechanism
//! (construct the fixed case, dispatch, compare, classify) runs and passes
//! on whatever CI's own arch-matched prove lane provides — it does not and
//! cannot reproduce the mismatch itself hermetically (that needs a
//! genuinely arch-mismatched cubin, a CI/build-matrix concern and a
//! separate remediation wave; see issue #434).
//!
//! ## Cache granularity: per-process, not per-device
//!
//! `set_force_dmmv` is a `candle_core`-internal PROCESS-GLOBAL
//! `AtomicBool` (`quantized/cuda.rs:22-27`), not a per-device switch — so
//! this guard's own verdict is cached per-PROCESS ([`std::sync::OnceLock`]),
//! not per-device-ordinal: once any CUDA device on this process is found to
//! need the DMMV fallback, that fallback is (unavoidably, given candle's
//! own global flag) in effect for every OTHER CUDA device this same
//! process ever touches too. Forcing the proven-correct slow path onto a
//! device that might otherwise have passed its own canary is the safe
//! default under this constraint (never the reverse), not a bug to work
//! around here.
//!
//! ## Zero per-call overhead after the first
//!
//! [`crate::quantized_cuda_canary::ensure_quantized_cuda_admitted`] is the
//! entry `ops::quant_matmul_grad` (`crate::ops::quant_matmul_grad`) calls
//! before every dispatch; after the first CUDA call in the process, this
//! degrades to one `OnceLock` read (no kernel launch, no host/device copy).
//!
//! ## Why the canary calls `ops::apply_stateful1`/`QuantMatMulGrad`
//! directly, never the public `quant_matmul_grad` helper
//!
//! `quant_matmul_grad` itself calls
//! [`crate::quantized_cuda_canary::ensure_quantized_cuda_admitted`] FIRST
//! (the wiring this module exists for) — so a canary that called
//! `quant_matmul_grad` to run its own known-answer forward would recurse
//! into this same check (and, on the very first call, into a `OnceLock`
//! still being initialized: `std::sync::OnceLock::get_or_init` panics on
//! reentrant initialization). The canary instead constructs
//! [`crate::ops::QuantMatMulGrad`] and applies it via
//! [`crate::ops::apply_stateful1`] directly — the exact same `CustomOp1`
//! `quant_matmul_grad` itself applies, one layer below the wrapper that
//! calls this module, so the kernel entry under test is byte-identical
//! without the recursion.
//!
//! ## The known-answer case and its bound (family F: measured, not assumed)
//!
//! One `Q8_0` block (`GgmlDType::Q8_0`'s own block size — the smallest
//! shape `QTensor::quantize` accepts at all, family D): a `[1, 32]` weight
//! alternating `+1.0`/`-1.0` (exact `Q8_0` amplitude `1.0`, no rounding
//! ambiguity at construction) dotted against a fixed `cos`-fixture
//! activation of amplitude `<= 1.0` (identical shape to this crate's own
//! `cuda_parity.rs` fixtures). The reference CPU value at this fixture is
//! `~0.664` — genuinely `O(1)`, not accidentally huge or accidentally
//! zero.
//!
//! The ordinary CPU-vs-CUDA divergence on a HEALTHY device is bounded
//! analytically (same derivation this crate's own `cuda_parity.rs` uses
//! for its `q8_1_activation_quant_bound`): the CUDA fast kernels
//! re-quantize the activation to `Q8_1` before the dot product (a step
//! `cpu_fwd` never performs), a per-element rounding error of at most
//! `0.5 * activation_amplitude / 127`, propagated through a `k`-deep dot
//! product against a weight of amplitude `weight_amplitude`:
//! `2.0 * k * weight_amplitude * 0.5 * activation_amplitude / 127`. At this
//! fixture's own `k = 32`, `weight_amplitude = 1.0`,
//! `activation_amplitude <= 1.0`: `~0.252` — squarely `O(0.5)`, matching
//! this crate's own measured `Q8_0` forward-parity residual elsewhere
//! (`ops::quant_matmul_grad`'s own CPU test module: `0.0288` at a larger,
//! multi-block shape).
//!
//! `CANARY_BOUND` is `1.0` — roughly 4x the analytic ordinary-noise ceiling
//! above, while sitting orders of magnitude below any garbage value issue
//! #434 actually observed (`O(10)`-`O(1e38)`, uninitialized device memory
//! read back after a silently-failed kernel launch). A case whose
//! reference value and tolerance are both `O(1)` and whose failure mode is
//! `O(10)` or larger separates decisively — no headroom tuning is needed
//! to tell the two apart. (`CANARY_BOUND` is a plain code span here, not
//! an intra-doc link: it lives behind `#[cfg(feature = "cuda")]`, while
//! this module doc is compiled unconditionally — a link would break the
//! default, non-`cuda` rustdoc build.)
use candle_core::Device;

use crate::error::{KernelError, Result};

#[cfg(feature = "cuda")]
use std::borrow::Cow;
#[cfg(feature = "cuda")]
use std::sync::{Arc, OnceLock};

#[cfg(feature = "cuda")]
use candle_core::quantized::{GgmlDType, QStorage, QTensor};
#[cfg(feature = "cuda")]
use candle_core::Tensor;

/// One `Q8_0` block — see the module doc's "known-answer case" section.
#[cfg(feature = "cuda")]
const CANARY_IN_FEATURES: usize = 32;
#[cfg(feature = "cuda")]
const CANARY_OUT_FEATURES: usize = 1;
#[cfg(feature = "cuda")]
const CANARY_ROWS: usize = 1;

/// See the module doc's bound derivation.
#[cfg(feature = "cuda")]
const CANARY_BOUND: f32 = 1.0;

/// The canary's own verdict — already-resolved, so [`decide`] (the
/// decision core every call site's outcome flows through) is testable with
/// literal, deterministic inputs, independent of any real device. Mirrors
/// `crate::admission::admit_inner`'s own "decision core... testable with
/// literal inputs" split.
///
/// `LegacyDmmvFallback`/`Refused` are constructed ONLY by [`run_canary`]
/// (`#[cfg(feature = "cuda")]`) in production; without this crate's own
/// `cuda` feature compiled in, [`ensure_quantized_cuda_admitted`]'s
/// fallback arm constructs `FastKernelsTrusted` only, and this crate's own
/// unit tests (below) are the sole other constructor of every variant —
/// `#[cfg_attr(not(feature = "cuda"), allow(dead_code))]` reflects that
/// honestly (those two variants are genuinely unreachable from non-test
/// code in a non-`cuda` build) rather than suppressing dead-code analysis
/// crate-wide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
enum CanaryVerdict {
    /// The fast kernels (`fast_mmvq`/`fast_mmq`) produced a correct
    /// known-answer result on this device. No global `candle_core` state
    /// was touched.
    FastKernelsTrusted,
    /// The fast kernels failed the canary; `set_force_dmmv(true)` was
    /// called and the legacy DMMV path re-verified correct.
    LegacyDmmvFallback,
    /// BOTH the fast kernels and the DMMV fallback failed the canary.
    Refused,
}

/// [`ensure_quantized_cuda_admitted`]'s decision core: `verdict` is already
/// resolved (by [`canary_verdict`], from a real device query) rather than
/// computed here, so this function is unit-testable with literal
/// [`CanaryVerdict`] values, in every feature configuration (unlike the
/// device-touching functions below, this one never needs the `cuda`
/// feature or a real CUDA device to exercise).
fn decide(verdict: CanaryVerdict) -> Result<()> {
    match verdict {
        CanaryVerdict::FastKernelsTrusted | CanaryVerdict::LegacyDmmvFallback => Ok(()),
        CanaryVerdict::Refused => Err(KernelError::QuantizedCudaCanaryFailed),
    }
}

/// Computed exactly once per process (see the module doc's "cache
/// granularity" section for why per-process, not per-device-ordinal, is
/// the correct granularity given `set_force_dmmv`'s own global scope).
#[cfg(feature = "cuda")]
fn canary_verdict(device: &Device) -> CanaryVerdict {
    static VERDICT: OnceLock<CanaryVerdict> = OnceLock::new();
    *VERDICT.get_or_init(|| run_canary(device))
}

/// Runs the known-answer case; on failure, logs ONCE (this function itself
/// only ever runs once per process — see [`canary_verdict`]'s own
/// `OnceLock`), flips `set_force_dmmv(true)`, and re-runs the case on the
/// legacy path before giving up.
#[cfg(feature = "cuda")]
fn run_canary(device: &Device) -> CanaryVerdict {
    if canary_case_passes(device) {
        return CanaryVerdict::FastKernelsTrusted;
    }
    tracing::warn!(
        "quantized fast-path kernels (fast_mmvq/fast_mmq) cannot execute correctly on this \
         device -- arch-mismatched single-arch SASS build (see issue #434); falling back to \
         the legacy PTX-JIT'd DMMV path: correct, slower"
    );
    candle_core::quantized::cuda::set_force_dmmv(true);
    if canary_case_passes(device) {
        CanaryVerdict::LegacyDmmvFallback
    } else {
        CanaryVerdict::Refused
    }
}

#[cfg(feature = "cuda")]
fn canary_case_passes(device: &Device) -> bool {
    match canary_max_abs_diff(device) {
        Ok(diff) => diff.is_finite() && diff < CANARY_BOUND,
        Err(_) => false,
    }
}

/// Builds the known-answer case, runs it on CPU (the reference — CPU
/// `cpu_fwd` is proven correct independently by `ops::quant_matmul_grad`'s
/// own CPU test module) and on `device` (the SAME quantized bytes, uploaded
/// via `QStorage::from_data` — the identical host-to-device upload
/// `candle_core::quantized::ggml_file::qtensor_from_ggml`'s own
/// `cuda::load_quantized` arm performs, no re-quantization on either side),
/// and returns the max absolute elementwise difference. See the module
/// doc's "why not the public `quant_matmul_grad` helper" section for why
/// this calls `apply_stateful1`/`QuantMatMulGrad` directly.
#[cfg(feature = "cuda")]
fn canary_max_abs_diff(device: &Device) -> candle_core::Result<f32> {
    let w_v: Vec<f32> = (0..CANARY_OUT_FEATURES * CANARY_IN_FEATURES)
        .map(|i| if i % 2 == 0 { 1.0f32 } else { -1.0f32 })
        .collect();
    let x_v: Vec<f32> = (0..CANARY_ROWS * CANARY_IN_FEATURES)
        .map(|i| ((i as f64) * 0.091 + 1.0).cos() as f32)
        .collect();

    let cpu = Device::Cpu;
    let w_cpu_tensor = Tensor::from_vec(w_v, (CANARY_OUT_FEATURES, CANARY_IN_FEATURES), &cpu)?;
    let wq_cpu = QTensor::quantize(&w_cpu_tensor, GgmlDType::Q8_0)?;
    let bytes = wq_cpu.data()?.into_owned();
    let cuda_storage = QStorage::from_data(Cow::Owned(bytes), device, GgmlDType::Q8_0)?;
    let wq_cuda = QTensor::new(cuda_storage, wq_cpu.shape().clone())?;

    let x_cpu = Tensor::from_vec(x_v.clone(), (CANARY_ROWS, CANARY_IN_FEATURES), &cpu)?;
    let x_cuda = Tensor::from_vec(x_v, (CANARY_ROWS, CANARY_IN_FEATURES), device)?;

    let y_cpu = crate::ops::apply_stateful1(
        &x_cpu.contiguous()?,
        crate::ops::QuantMatMulGrad::new(Arc::new(wq_cpu)),
    )?;
    let y_cuda = crate::ops::apply_stateful1(
        &x_cuda.contiguous()?,
        crate::ops::QuantMatMulGrad::new(Arc::new(wq_cuda)),
    )?;

    let cpu_v = y_cpu.flatten_all()?.to_vec1::<f32>()?;
    let cuda_v = y_cuda.flatten_all()?.to_vec1::<f32>()?;

    let mut max_abs_diff = 0f32;
    for (c, g) in cpu_v.iter().zip(cuda_v.iter()) {
        max_abs_diff = max_abs_diff.max((c - g).abs());
    }
    Ok(max_abs_diff)
}

/// The engine guard's own entry point: `crate::ops::quant_matmul_grad`
/// calls this BEFORE every dispatch (see the module doc's "zero per-call
/// overhead" section), so no production quantized matmul can reach a CUDA
/// device's fast kernels ahead of this check.
///
/// A no-op (`Ok(())`, `device` unused) on CPU/Metal, and — without this
/// crate's own `cuda` feature compiled in — on every device: the failure
/// class this guards against is CUDA-only (module doc), and this crate's
/// `cuda` feature forwarding `candle-core/cuda` (`Cargo.toml`) is what
/// makes `candle_core::quantized::cuda::set_force_dmmv` a public path at
/// all — see `crate::admission::device_is_supported`'s own doc for the
/// identical `cfg!(feature = "cuda")` reasoning this crate follows
/// throughout.
pub fn ensure_quantized_cuda_admitted(device: &Device) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if !matches!(device, Device::Cuda(_)) {
            return Ok(());
        }
        decide(canary_verdict(device))
    }
    #[cfg(not(feature = "cuda"))]
    {
        // No quantized-CUDA capability exists to gate at all without this
        // crate's own `cuda` feature compiled in (module doc) -- routed
        // through the SAME decision core as the real check (rather than a
        // bare `Ok(())`) so `decide`/`CanaryVerdict` stay live code in
        // EVERY feature configuration, not merely under `cuda` + `test`.
        let _ = device;
        decide(CanaryVerdict::FastKernelsTrusted)
    }
}

/// Test/introspection-only (mirrors `ops::bwd_gradient_gemm_layouts`'s own
/// "mechanism pin" convention, `ops/mod.rs`): whether this process's
/// quantized-CUDA canary settled on the fast kernels (as opposed to the
/// legacy DMMV fallback) for `device`. Runs (and caches, via
/// `canary_verdict`'s own `OnceLock`) the canary if it has not already
/// run this process. `tests/cuda_parity.rs`'s
/// `quantized_cuda_canary_passes_on_a_healthy_build_and_device` is the
/// sole caller.
#[doc(hidden)]
#[cfg(feature = "cuda")]
pub fn quantized_cuda_canary_used_fast_kernels_for_test(device: &Device) -> bool {
    matches!(canary_verdict(device), CanaryVerdict::FastKernelsTrusted)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`decide`] admits on [`CanaryVerdict::FastKernelsTrusted`] — no
    /// device, no `cuda` feature required (module doc: `decide` is the
    /// device-independent decision core).
    #[test]
    fn decide_admits_on_fast_kernels_trusted() {
        assert!(decide(CanaryVerdict::FastKernelsTrusted).is_ok());
    }

    /// [`decide`] admits on [`CanaryVerdict::LegacyDmmvFallback`] — the
    /// fallback engaging is a SUCCESSFUL outcome (a proven-correct, slower
    /// path), not an error.
    #[test]
    fn decide_admits_on_legacy_dmmv_fallback() {
        assert!(decide(CanaryVerdict::LegacyDmmvFallback).is_ok());
    }

    /// [`decide`] refuses on [`CanaryVerdict::Refused`] with the typed
    /// [`KernelError::QuantizedCudaCanaryFailed`] — refusal beats a
    /// confident wrong number (K2), never a silent `Ok`.
    #[test]
    fn decide_refuses_when_both_paths_fail() {
        let err = decide(CanaryVerdict::Refused).unwrap_err();
        assert!(matches!(err, KernelError::QuantizedCudaCanaryFailed));
    }

    /// [`ensure_quantized_cuda_admitted`] is a total no-op on a CPU device
    /// — the failure class this module guards against does not apply
    /// there (module doc). Runs in every feature configuration: this
    /// assertion does not require the `cuda` feature at all, since the
    /// CPU early-return sits OUTSIDE the `#[cfg(feature = "cuda")]` split
    /// only when that feature is off (see that function's own two `cfg`
    /// arms) — with the feature ON, the same CPU device still hits the
    /// `!matches!(device, Device::Cuda(_))` early return inside the first
    /// arm. Either way, `Ok(())` for `Device::Cpu`.
    #[test]
    fn ensure_quantized_cuda_admitted_is_a_no_op_on_cpu() {
        assert!(ensure_quantized_cuda_admitted(&Device::Cpu).is_ok());
    }
}
