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
//! quantization error, vs `18.6` garbage on the broken fast path — the same
//! `0.0288` magnitude is independently reproduced in-tree, see
//! `forward_parity_against_dense_dequantized_reference_q8_0_q4_0_q4k`'s own
//! measured `Q8_0` residual). Every OTHER kernel this workspace ships
//! (dense matmul, dequantize, `jammi-kernels`'s own fused ops) is PTX-JIT'd
//! already and unaffected by any of this — this module exists ONLY to gate
//! the quantized fast path.
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
//! ## Cache granularity: per-device VERDICT, per-process REMEDIATION
//!
//! `canary_verdict` runs (and caches) the known-answer case ONCE PER
//! CUDA ORDINAL, keyed in a `Mutex<HashMap<usize, CanaryVerdict>>` by
//! `canary_device_ordinal` — a `FastKernelsTrusted` verdict on ordinal 0
//! never admits ordinal 1 unchecked; a multi-GPU process with one
//! arch-matched and one arch-mismatched device gets each device's OWN
//! known-answer run. (An earlier version of this guard cached a single
//! process-global verdict regardless of ordinal — a real defect, not a
//! documented tradeoff: it let one passing device silently vouch for every
//! other CUDA device in the process. Fixed as part of issue #434's
//! remediation.)
//!
//! The one thing that IS still process-global, unavoidably, is the
//! REMEDIATION `set_force_dmmv` itself applies: it is a `candle_core`-
//! internal PROCESS-GLOBAL `AtomicBool` (`quantized/cuda.rs:22-27`), not a
//! per-device switch. So once ANY ordinal's canary run calls
//! `set_force_dmmv(true)`, every OTHER CUDA device this same process
//! touches afterward — including one whose own canary already passed and
//! is cached `FastKernelsTrusted` — actually executes on the legacy DMMV
//! path from that point on too, regardless of what its own cached verdict
//! says (the cached verdict only gates admission; it does not, and cannot,
//! reach back into candle's own dispatch to force the fast path). This
//! granularity mismatch (per-device diagnosis, per-process remediation) is
//! a real, unavoidable consequence of `set_force_dmmv`'s own global scope,
//! not a bug this guard can close from its own side — and it is the SAFE
//! direction to be wrong in: forcing the proven-correct slow path onto a
//! device that might otherwise have passed its own canary, never the
//! reverse.
//!
//! ## Zero per-call overhead after the first
//!
//! [`crate::quantized_cuda_canary::ensure_quantized_cuda_admitted`] is the
//! entry `ops::quant_matmul_grad` (`crate::ops::quant_matmul_grad`) calls
//! before every dispatch; after the first CUDA call FOR A GIVEN ORDINAL in
//! the process, this degrades to one table lookup (a `Mutex` lock + a
//! `HashMap` get keyed on that ordinal — no kernel launch, no host/device
//! copy).
//!
//! ## Why the canary calls `ops::apply_stateful1`/`QuantMatMulGrad`
//! directly, never the public `quant_matmul_grad` helper
//!
//! `quant_matmul_grad` itself calls
//! [`crate::quantized_cuda_canary::ensure_quantized_cuda_admitted`] FIRST
//! (the wiring this module exists for) — so a canary that called
//! `quant_matmul_grad` to run its own known-answer forward would recurse
//! into this same check (and, on the very first call for a given ordinal,
//! into UNBOUNDED recursion: `canary_verdict`'s own table has no entry
//! for that ordinal yet — filling it in is the very thing this call is
//! doing — so the recursive call would run the canary again, and again,
//! never reaching a base case). The canary instead constructs
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
//! `activation_amplitude <= 1.0`: `~0.252` (no-producer: analytically
//! derived from the rounding-error formula stated in this same paragraph,
//! not a measured run) — squarely `O(0.5)`, matching this crate's own
//! measured `Q8_0` forward-parity residual elsewhere, see
//! `forward_parity_against_dense_dequantized_reference_q8_0_q4_0_q4k`'s own
//! `0.0288` at a larger, multi-block shape.
//!
//! `CANARY_BOUND` is `0.3` — roughly 1.2x the `~0.252` (no-producer:
//! analytically derived, same rounding-error formula as above, not a
//! measured run) ordinary-noise ceiling above (enough margin to absorb the
//! fixture's own `f32` rounding without false-failing a healthy device),
//! while sitting decisively BELOW this fixture's own known-answer
//! magnitude (`~0.664`, see
//! `canary_diff_passes_rejects_the_all_zeros_failed_launch_signature`'s own
//! pinned fixture) and orders of magnitude below any garbage value issue
//! #434 actually observed (`O(10)`-`O(1e38)`, uninitialized device memory
//! read back after a silently-failed kernel launch). This is the fix for a
//! real defect an earlier version of this guard had: `CANARY_BOUND == 1.0`
//! admitted an ALL-ZEROS output — `|0.664 - 0| == 0.664 < 1.0` — the
//! canonical signature of a failed launch reading back a zeroed (rather
//! than uninitialized-garbage) allocation, which this fixture's own
//! magnitude happens to be small enough to hide behind a bound sized only
//! against the garbage-value ceiling and not against the known answer
//! itself. `0.3 < 0.664` closes that gap: a zeroed buffer now fails
//! decisively, proven by `canary_case_outcome`'s own decision-core test
//! (see
//! `canary_case_outcome_reports_a_known_answer_disagreement_as_ok_false`)
//! constructing exactly that state (`diff == 0.664`, the zero-output
//! signature) without needing a device at all. `CANARY_BOUND` is not
//! feature-gated (unlike the fixture-shape constants below it): the
//! decision core that reads it (`canary_diff_passes`) is deliberately
//! compiled and testable in every feature configuration, mirroring
//! `decide`'s own "testable with literal inputs, independent of any real
//! device" split.
use candle_core::Device;

use crate::error::{KernelError, Result};

#[cfg(feature = "cuda")]
use std::borrow::Cow;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex, OnceLock};

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

/// See the module doc's bound derivation. NOT feature-gated (unlike the
/// fixture-shape constants above): [`canary_diff_passes`], the decision
/// core that reads this, is deliberately compiled and testable in every
/// feature configuration. `#[cfg_attr(not(feature = "cuda"), allow(dead_code))]`
/// mirrors [`CanaryVerdict`]'s own honest annotation just below: without
/// the `cuda` feature, only this crate's own unit tests read it.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const CANARY_BOUND: f32 = 0.3;

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

/// [`canary_case_passes`]'s decision core: `diff` (a max absolute
/// elementwise difference, already computed) is finite and strictly below
/// [`CANARY_BOUND`]. Pure and device-independent — see the module doc's
/// `CANARY_BOUND` paragraph for why `0.3` decisively separates ordinary
/// quantization noise (`~0.252`) from BOTH this fixture's own known-answer
/// magnitude (`~0.664`, the boundary an all-zeros failed-launch readback
/// used to hide behind at the old `1.0` bound) and any larger garbage
/// value a genuinely arch-mismatched launch produces.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn canary_diff_passes(diff: f32) -> bool {
    diff.is_finite() && diff < CANARY_BOUND
}

/// [`canary_case_passes`]'s other decision-core half: given the `Result`
/// [`canary_max_abs_diff`] already produced (`Ok(diff)`, or an infra `Err`
/// this function itself never constructs — device OOM, allocation failure,
/// a transient driver error), decides the case's outcome WITHOUT touching a
/// device. `Ok(true)` is a known-answer match, `Ok(false)` is a
/// successfully-computed, finite known-answer DISAGREEMENT (the real
/// "arch-mismatched fast kernels" signal — see the module doc's failure
/// class section), and `Err(_)` is an infra failure that PROPAGATES
/// unchanged rather than being folded into either outcome: [`run_canary`]
/// must never flip `set_force_dmmv` or settle on [`CanaryVerdict::Refused`]
/// on account of a transient error that says nothing about whether the fast
/// kernels themselves are correct. Literal `Result<f32>` inputs make all
/// three outcomes independently testable below with no device or `cuda`
/// feature required.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn canary_case_outcome(diff: candle_core::Result<f32>) -> candle_core::Result<bool> {
    diff.map(canary_diff_passes)
}

/// This CUDA device's ordinal — the key [`canary_verdict`]'s per-device
/// table uses (module doc's "cache granularity" section).
/// `ensure_quantized_cuda_admitted`'s `cfg(feature = "cuda")` arm only ever
/// calls [`canary_verdict`] after matching `Device::Cuda(_)`, so the
/// `Cpu`/`Metal` arm below is unreachable in production today; it exists
/// so this fn is TOTAL over every `DeviceLocation` variant (never a
/// partial match relying on `unreachable!()`, which would turn a future,
/// legitimate non-CUDA call site here into a new panic surface — this
/// guard's own job is to be conservative, not to add one) rather than
/// because ordinal `0` is a meaningful fallback for a non-CUDA device.
#[cfg(feature = "cuda")]
fn canary_device_ordinal(device: &Device) -> usize {
    match device.location() {
        candle_core::DeviceLocation::Cuda { gpu_id } => gpu_id,
        candle_core::DeviceLocation::Cpu | candle_core::DeviceLocation::Metal { .. } => 0,
    }
}

/// Computed once PER CUDA ORDINAL (module doc's "cache granularity"
/// section — this used to be a single process-global `OnceLock`, which let
/// ordinal 0's verdict silently vouch for every other CUDA device in the
/// process; fixed as part of issue #434's remediation). An infra `Err` from
/// [`run_canary`] is NEVER cached here — it propagates straight to the
/// caller (see [`canary_case_outcome`]'s own doc) and the NEXT call for
/// this ordinal retries from scratch, exactly as if nothing had run yet.
#[cfg(feature = "cuda")]
fn canary_verdict(device: &Device) -> candle_core::Result<CanaryVerdict> {
    static VERDICTS: OnceLock<Mutex<HashMap<usize, CanaryVerdict>>> = OnceLock::new();
    let ordinal = canary_device_ordinal(device);
    let table = VERDICTS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(v) = table
        .lock()
        .expect("quantized-CUDA canary verdict table mutex poisoned")
        .get(&ordinal)
    {
        return Ok(*v);
    }
    // Runs OUTSIDE the lock: `run_canary` dispatches real device work
    // (kernel launches, host<->device copies) that must never happen while
    // holding this table's mutex.
    let verdict = run_canary(device)?;
    Ok(*table
        .lock()
        .expect("quantized-CUDA canary verdict table mutex poisoned")
        .entry(ordinal)
        .or_insert(verdict))
}

/// Runs the known-answer case; on failure, logs ONCE PER ORDINAL (this
/// function itself only ever runs once per ordinal that reaches a
/// non-`Err` outcome — see [`canary_verdict`]'s own table), flips
/// `set_force_dmmv(true)`, and re-runs the case on the legacy path before
/// giving up. An infra `Err` from either [`canary_case_passes`] call
/// propagates immediately: neither arm of this fn ever flips
/// `set_force_dmmv` or returns [`CanaryVerdict::Refused`] on account of a
/// call that could not run at all, only on account of one that RAN and
/// disagreed.
#[cfg(feature = "cuda")]
fn run_canary(device: &Device) -> candle_core::Result<CanaryVerdict> {
    if canary_case_passes(device)? {
        return Ok(CanaryVerdict::FastKernelsTrusted);
    }
    tracing::warn!(
        "quantized fast-path kernels (fast_mmvq/fast_mmq) cannot execute correctly on this \
         device -- arch-mismatched single-arch SASS build (see issue #434); falling back to \
         the legacy PTX-JIT'd DMMV path: correct, slower"
    );
    candle_core::quantized::cuda::set_force_dmmv(true);
    if canary_case_passes(device)? {
        Ok(CanaryVerdict::LegacyDmmvFallback)
    } else {
        Ok(CanaryVerdict::Refused)
    }
}

#[cfg(feature = "cuda")]
fn canary_case_passes(device: &Device) -> candle_core::Result<bool> {
    canary_case_outcome(canary_max_abs_diff(device))
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
///
/// Returns `candle_core::Result<()>`, not this crate's own
/// [`crate::error::Result`], DELIBERATELY: two distinct failure kinds can
/// reach a caller here, and only one of them is this module's own typed
/// refusal. A [`KernelError::QuantizedCudaCanaryFailed`] (both the fast
/// kernels AND the DMMV fallback disagreed with the known answer) is
/// preserved, TYPED, through the crate-wide
/// [`impl From<KernelError> for candle_core::Error`](crate::error) (wrapping
/// via `Error::Cuda`'s own `Box<dyn std::error::Error + Send + Sync>`
/// payload) — `Error::Cuda` downcasts (`std::error::Error::downcast_ref`)
/// back to `KernelError` at any downstream call site that wants to match it
/// specifically, rather than collapsing to an untyped
/// `Error::Msg(e.to_string())` a caller can only pattern-match by string.
/// `ops::low_rank_residual_linear::admit_cast_boundary` is this crate's
/// OTHER production call site for the same conversion — see that fn's own
/// doc. An infra error from `canary_verdict` (device OOM, allocation
/// failure, a transient driver error — see `canary_case_outcome`'s own doc)
/// propagates completely unchanged: it is not this guard's own refusal at
/// all, so it is never re-wrapped.
pub fn ensure_quantized_cuda_admitted(device: &Device) -> candle_core::Result<()> {
    #[cfg(feature = "cuda")]
    {
        if !matches!(device, Device::Cuda(_)) {
            return Ok(());
        }
        let verdict = canary_verdict(device)?;
        decide(verdict).map_err(candle_core::Error::from)
    }
    #[cfg(not(feature = "cuda"))]
    {
        // No quantized-CUDA capability exists to gate at all without this
        // crate's own `cuda` feature compiled in (module doc) -- routed
        // through the SAME decision core as the real check (rather than a
        // bare `Ok(())`) so `decide`/`CanaryVerdict` stay live code in
        // EVERY feature configuration, not merely under `cuda` + `test`.
        let _ = device;
        decide(CanaryVerdict::FastKernelsTrusted).map_err(candle_core::Error::from)
    }
}

/// Test/introspection-only (mirrors `ops::bwd_gradient_gemm_layouts`'s own
/// "mechanism pin" convention, `ops/mod.rs`): whether this process's
/// quantized-CUDA canary settled on the fast kernels (as opposed to the
/// legacy DMMV fallback) for `device`. Runs (and caches, via
/// `canary_verdict`'s own per-ordinal table) the canary if it has not
/// already run for this device's own ordinal. `tests/cuda_parity.rs`'s
/// `quantized_cuda_canary_passes_on_a_healthy_build_and_device` is the
/// sole caller — it always calls `quant_matmul_grad` first (which must
/// itself succeed), so an infra `Err` reaching this fn's own
/// `canary_verdict` call here would already have surfaced there; folding
/// it to `false` here is therefore never observed to mask a real infra
/// failure in practice.
#[doc(hidden)]
#[cfg(feature = "cuda")]
pub fn quantized_cuda_canary_used_fast_kernels_for_test(device: &Device) -> bool {
    matches!(
        canary_verdict(device),
        Ok(CanaryVerdict::FastKernelsTrusted)
    )
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

    /// The missing downcast oracle (audit advisory 4): `decide`'s typed
    /// `KernelError`, wrapped through
    /// [`impl From<KernelError> for candle_core::Error`](crate::error) —
    /// the SAME conversion `ensure_quantized_cuda_admitted` and
    /// `ops::low_rank_residual_linear::admit_cast_boundary` both apply at
    /// their own production call sites — downcasts (`std::error::Error::
    /// downcast_ref`) back to the ORIGINAL `KernelError` value on the other
    /// side of `candle_core::Error::Cuda`'s `Box<dyn std::error::Error +
    /// Send + Sync>` payload. Proven for BOTH `KernelError` variants that
    /// production code actually constructs through this channel:
    /// `QuantizedCudaCanaryFailed` (this module's own refusal, produced by
    /// `decide`) and `StrictModeFallback` (`crate::admission`'s typed
    /// refusal STRICT-mode callers exist to match) — a single test with two
    /// independent round-trips, not two near-duplicate tests, since both
    /// assert the identical mechanism against different `KernelError`
    /// payloads.
    #[test]
    fn typed_kernel_error_round_trips_through_the_from_impl_via_downcast() {
        let wrapped: candle_core::Error = decide(CanaryVerdict::Refused).unwrap_err().into();
        let candle_core::Error::Cuda(boxed) = &wrapped else {
            panic!("expected From<KernelError> to wrap via Error::Cuda, got {wrapped:?}");
        };
        let downcast = boxed
            .downcast_ref::<KernelError>()
            .expect("Error::Cuda's boxed payload must downcast back to KernelError");
        assert!(matches!(downcast, KernelError::QuantizedCudaCanaryFailed));

        let strict_fallback = KernelError::StrictModeFallback {
            op: "low_rank_residual_linear",
            predicate: "grad_res_is_bf16_a_fusable_two_kernel_chain",
        };
        let wrapped_strict: candle_core::Error = strict_fallback.into();
        let candle_core::Error::Cuda(boxed_strict) = &wrapped_strict else {
            panic!("expected From<KernelError> to wrap via Error::Cuda, got {wrapped_strict:?}");
        };
        let downcast_strict = boxed_strict
            .downcast_ref::<KernelError>()
            .expect("Error::Cuda's boxed payload must downcast back to KernelError");
        assert!(matches!(
            downcast_strict,
            KernelError::StrictModeFallback {
                op: "low_rank_residual_linear",
                predicate: "grad_res_is_bf16_a_fusable_two_kernel_chain",
            }
        ));
    }

    /// [`canary_diff_passes`] admits the fixture's own analytic
    /// ordinary-noise ceiling (`~0.252`, no-producer: analytically derived,
    /// module doc's bound derivation — not a measured run).
    #[test]
    fn canary_diff_passes_admits_ordinary_quantization_noise() {
        assert!(canary_diff_passes(0.252));
    }

    /// [`canary_diff_passes`] REJECTS the all-zeros failed-launch signature
    /// this guard's old `CANARY_BOUND == 1.0` used to admit: a zeroed
    /// device readback against this fixture's own known-answer magnitude
    /// (`~0.664`) yields `diff == 0.664` — the exact zero-output state the
    /// module doc's `CANARY_BOUND` paragraph names.
    #[test]
    fn canary_diff_passes_rejects_the_all_zeros_failed_launch_signature() {
        assert!(!canary_diff_passes(0.664));
    }

    /// [`canary_diff_passes`] rejects non-finite input outright — `NaN <
    /// CANARY_BOUND` is `false` in IEEE-754 either way, but the explicit
    /// `is_finite()` guard makes that fact load-bearing rather than
    /// incidental (family F: a negative control must fail on every bad
    /// path, non-finite included, never merely rely on a comparison that
    /// happens to already reject it).
    #[test]
    fn canary_diff_passes_rejects_non_finite() {
        assert!(!canary_diff_passes(f32::NAN));
        assert!(!canary_diff_passes(f32::INFINITY));
    }

    /// [`canary_case_outcome`]'s first of three decision-core outcomes: a
    /// successfully-computed, in-bound diff is `Ok(true)`.
    #[test]
    fn canary_case_outcome_admits_a_known_answer_match() {
        assert!(canary_case_outcome(Ok(0.01)).unwrap());
    }

    /// Second outcome: a successfully-computed, OUT-of-bound diff is
    /// `Ok(false)` — the zero-output state again, this time through the
    /// SAME decision core [`canary_case_passes`] itself calls in
    /// production, not just through [`canary_diff_passes`] directly.
    #[test]
    fn canary_case_outcome_reports_a_known_answer_disagreement_as_ok_false() {
        assert!(!canary_case_outcome(Ok(0.664)).unwrap());
    }

    /// Third outcome: an infra `Err` (device OOM, allocation failure, a
    /// transient driver error — never a computed disagreement) PROPAGATES
    /// unchanged, rather than being folded into `Ok(false)`. Conflating
    /// these two would let a transient error permanently flip
    /// `set_force_dmmv` or refuse a healthy device outright (see
    /// `canary_case_outcome`'s own doc) — this test pins that a
    /// synthetic infra error survives the decision core as the SAME
    /// error, not a laundered boolean.
    #[test]
    fn canary_case_outcome_propagates_infra_errors_without_flipping_or_refusing() {
        let err = canary_case_outcome(Err(candle_core::Error::Msg(
            "synthetic infra failure: device OOM".to_string(),
        )))
        .expect_err("an infra Err must propagate, never silently read as Ok(false)");
        match err {
            candle_core::Error::Msg(msg) => assert_eq!(msg, "synthetic infra failure: device OOM"),
            other => panic!("expected the SAME Msg error to propagate unchanged, got {other:?}"),
        }
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
