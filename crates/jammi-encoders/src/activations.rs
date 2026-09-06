//! Activation functions shared by more than one tower's MLP.

use std::sync::LazyLock;

use candle_core::{DType, Tensor};
use jammi_kernels::admission::{
    admission_mode, admit, counters_for, device_is_supported, DispatchCounters, DispatchOutcome,
};

use crate::error::EncoderError;

/// QuickGelu activation: `x * sigmoid(1.702 * x)`. OpenCLIP uses this in
/// both the text ([`crate::clip_text`]) and vision
/// ([`crate::open_clip_vision`]) tower MLPs (not the standard erf-based
/// GELU); the single shared implementation both towers call.
pub(crate) fn quick_gelu(xs: &Tensor) -> Result<Tensor, EncoderError> {
    Ok((xs * candle_nn::ops::sigmoid(&(xs * 1.702f64)?)?)?)
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused GELU-erf seam (issue #463)
// ─────────────────────────────────────────────────────────────────────────────

/// Fused/eager dispatch counters for the `gelu_erf` seam, read from the
/// registry (`counters_for`) — mirroring `crate::layer_norm::LN_DISPATCH_COUNTERS`
/// / `crate::attention_cascade::SOFTMAX_DISPATCH_COUNTERS`. `pub(crate)`
/// (not `pub`) — a durable job record or a bench report reads it through a
/// `crate::gelu_dispatch_snapshot`-shaped API once one exists (not wired in
/// this unit — see [`gelu_erf`]'s own doc for what IS wired).
pub(crate) static GELU_DISPATCH_COUNTERS: LazyLock<&'static DispatchCounters> =
    LazyLock::new(|| counters_for("gelu_erf_fused"));

/// The fused GELU-erf kernel's domain, checked at the call site (family D /
/// K2): `x`'s device is one [`device_is_supported`] accepts, its dtype is
/// `F32` on either device or additionally `BF16`/`F16` on CUDA (matching
/// `jammi_kernels::ops::GeluErfFused`'s own per-device forward domain — F32
/// only on CPU, F32/BF16/F16 on CUDA — see that op's module doc), `x` is
/// contiguous (the op refuses a strided view, the same idiom every fused op
/// in this crate follows), and `x` is non-empty (an empty tensor has no
/// domain to fuse over; the eager `Tensor::gelu_erf` handles it identically
/// either way, so refusing it here just makes the choice a counted decline
/// rather than a dispatch to an op that would do no useful work).
fn gelu_admission_predicate(x: &Tensor) -> (bool, &'static str) {
    if !device_is_supported(x.device()) {
        return (false, "device_is_cpu_or_cuda");
    }
    let dtype_ok = if x.device().is_cuda() {
        matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16)
    } else {
        matches!(x.dtype(), DType::F32)
    };
    if !dtype_ok {
        return (
            false,
            if x.device().is_cuda() {
                "dtype_f32_bf16_or_f16_on_cuda"
            } else {
                "dtype_f32_only_on_cpu"
            },
        );
    }
    if !x.is_contiguous() {
        return (false, "contiguous");
    }
    if x.elem_count() == 0 {
        return (false, "non_empty");
    }
    (true, "domain_ok")
}

/// `TODO(lead-merge)`: the ONE line to flip once `jammi_kernels::ops::GeluErfFused`
/// (issue #463, `jammi-kernels`, `feat/463-gelu-erf-op`) lands on this
/// branch's base. Today this adapter calls the unchanged eager
/// `Tensor::gelu_erf` on the arm [`gelu_erf`]'s admission already counts as
/// `fused` — the counter/admission/predicate machinery above is real
/// already, so every counter test this unit adds passes unchanged the
/// moment the lead flips this one line to
/// `Ok(jammi_kernels::ops::apply1(x, jammi_kernels::ops::GeluErfFused::default())?)`
/// after merging the kernels branch first (see the contract's #462/#463
/// consolidation note).
fn dispatch_gelu_erf_fused(x: &Tensor) -> Result<Tensor, EncoderError> {
    // TODO(lead-merge): Ok(jammi_kernels::ops::apply1(x, jammi_kernels::ops::GeluErfFused::default())?)
    Ok(x.gelu_erf()?)
}

/// The GELU-erf seam: `training == false` calls the unchanged
/// `x.gelu_erf()` directly (no admission machinery at all — eval's output
/// is exactly what it always was, byte-for-byte); `training == true` admits
/// `gelu_erf_fused` on [`gelu_admission_predicate`]'s domain and dispatches
/// to [`dispatch_gelu_erf_fused`] (today an eager-calling adapter — see that
/// function's own doc) or the same unchanged eager call, recording which
/// happened either way. Wired at `bert.rs:192`
/// (`BertIntermediate::forward`) and `distilbert.rs:142`
/// (`DistilBertFfn::forward`) only.
///
/// **Not wired** (recorded here, per plan v2 R5', rather than silently
/// excluded): `crate::htsat_audio`'s two GELU sites (`SwinBlock`,
/// `ClapAudioProjection`) have no training flag at all and
/// `HtsatAudio::set_training` never reaches the projection — wiring them
/// means two new propagation edges with no oracle and nothing trains HTSAT
/// in this unit, a #421-adjacent gap tracked separately. `crate::context`'s
/// GELU site has no train/eval split. The GeGLU eager reference arm
/// (`crate::modernbert::geglu_apply_training`'s own `gate.gelu_erf()?`
/// call) and `quick_gelu` (above) are architecturally different
/// activations, out of this seam's scope entirely.
pub(crate) fn gelu_erf(x: &Tensor, training: bool) -> Result<Tensor, EncoderError> {
    if !training {
        return Ok(x.gelu_erf()?);
    }
    let (holds, predicate) = gelu_admission_predicate(x);
    let outcome = admit(
        admission_mode(),
        "gelu_erf_fused",
        predicate,
        holds,
        *GELU_DISPATCH_COUNTERS,
    )?;
    match outcome {
        DispatchOutcome::Fused => dispatch_gelu_erf_fused(x),
        DispatchOutcome::Eager => Ok(x.gelu_erf()?),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn gelu_erf_eval_matches_plain_gelu_erf_bit_identical() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[-2.0f32, -0.5, 0.0, 0.5, 2.0], (5,), &device).unwrap();
        let got = gelu_erf(&x, false).unwrap().to_vec1::<f32>().unwrap();
        let want = x.gelu_erf().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(
            got, want,
            "eval must be byte-for-byte the plain gelu_erf call"
        );
    }

    #[test]
    fn gelu_admission_predicate_accepts_contiguous_f32_cpu() {
        let device = Device::Cpu;
        let x = Tensor::zeros((2, 4), DType::F32, &device).unwrap();
        let (holds, predicate) = gelu_admission_predicate(&x);
        assert!(holds, "predicate={predicate}");
    }

    #[test]
    fn gelu_admission_predicate_rejects_bf16_on_cpu() {
        let device = Device::Cpu;
        let x = Tensor::zeros((2, 4), DType::BF16, &device).unwrap();
        let (holds, predicate) = gelu_admission_predicate(&x);
        assert!(!holds);
        assert_eq!(predicate, "dtype_f32_only_on_cpu");
    }

    #[test]
    fn gelu_admission_predicate_rejects_non_contiguous() {
        let device = Device::Cpu;
        let x = Tensor::zeros((4, 4), DType::F32, &device)
            .unwrap()
            .t()
            .unwrap();
        let (holds, predicate) = gelu_admission_predicate(&x);
        assert!(!holds);
        assert_eq!(predicate, "contiguous");
    }

    #[test]
    fn gelu_admission_predicate_rejects_empty() {
        let device = Device::Cpu;
        let x = Tensor::zeros((0, 4), DType::F32, &device).unwrap();
        let (holds, predicate) = gelu_admission_predicate(&x);
        assert!(!holds);
        assert_eq!(predicate, "non_empty");
    }

    /// Training's fused arm is a COUNTED dispatch today, even though the
    /// adapter still computes the eager value underneath (see
    /// `dispatch_gelu_erf_fused`'s doc) — the counter proof this unit's
    /// two-sided tests rely on is real now, only the numeric kernel is
    /// pending the lead's merge.
    #[test]
    fn gelu_erf_training_admits_fused_on_a_supported_shape_and_matches_eager_value() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[-2.0f32, -0.5, 0.0, 0.5, 2.0], (5,), &device).unwrap();
        let before = GELU_DISPATCH_COUNTERS.snapshot();
        let got = gelu_erf(&x, true).unwrap().to_vec1::<f32>().unwrap();
        let after = GELU_DISPATCH_COUNTERS.snapshot();
        let want = x.gelu_erf().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got, want);
        assert!(
            after.fused > before.fused,
            "a supported F32/contiguous/non-empty CPU tensor must dispatch fused"
        );
    }
}
