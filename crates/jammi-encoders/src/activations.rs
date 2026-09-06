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

/// Dispatches to the real fused kernel, `jammi_kernels::ops::GeluErfFused`
/// (issue #463) via [`jammi_kernels::ops::apply1`] — the same `CustomOp1`
/// adapter idiom every other fused op in this crate's seams uses (mirroring
/// `attention_cascade`'s `AttentionBlockFused`/`softmax`'s `SoftmaxFused`
/// call sites). On CPU F32 this op's forward is bit-identical to
/// `Tensor::gelu_erf` (the op's own module doc, "three cdf formulations",
/// item 1) — the fused arm never diverges numerically from the eager call
/// it replaces on the domain [`gelu_admission_predicate`] admits; only the
/// kernel launch changes. Backward matches candle's own `Op::Unary(_,
/// GeluErf)` composition within `jammi_kernels::ops::gelu_erf`'s documented
/// condition-aware bound (see that op's module doc's "backward" section and
/// its own exported `jammi_kernels::ops::gelu_erf::{COND_AWARE_TOL,
/// COND_AWARE_ABS_FLOOR}` constants, which this crate's own gradcheck
/// oracle imports rather than hand-copying — audit round item 4).
fn dispatch_gelu_erf_fused(x: &Tensor) -> Result<Tensor, EncoderError> {
    // `GeluErfFused` is a unit struct — clippy's `default_constructed_unit_structs`
    // prefers the bare value over `GeluErfFused::default()` (both are
    // identical; `Default` is only derived so callers who prefer that
    // spelling elsewhere in this crate's ecosystem have it available).
    Ok(jammi_kernels::ops::apply1(
        x,
        jammi_kernels::ops::GeluErfFused,
    )?)
}

/// The GELU-erf seam: `training == false` calls the unchanged
/// `x.gelu_erf()` directly (no admission machinery at all — eval's output
/// is exactly what it always was, byte-for-byte); `training == true` admits
/// `gelu_erf_fused` on [`gelu_admission_predicate`]'s domain and dispatches
/// to [`dispatch_gelu_erf_fused`] (the real `GeluErfFused` `CustomOp1` — see
/// that function's own doc) or the same unchanged eager call, recording
/// which happened either way. Wired at `bert.rs:296`
/// (`BertIntermediate::forward`'s `activations::gelu_erf(&hidden,
/// training)`) and `distilbert.rs:213` (`DistilBertFfn::forward`'s
/// `activations::gelu_erf(&mid, training)`) only — as of this fix round
/// (item 6) both receive `training` as a call-chain PARAMETER sourced
/// from `Bert::training`/`DistilBert::training`, not a per-sub-struct
/// stored copy, so this seam always sees the SAME value the encoder's own
/// `forward_hidden` dispatched on (these two line numbers are NOT tracked
/// by `ci/scripts/perf/check_citations.py`, which only resolves
/// `finetune_step.rs`/`grad_oracle.rs` under `jammi-bench` — a future edit
/// to either call site's surrounding code can silently drift this
/// citation; verify by name if in doubt).
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

    /// Strict mode on a refused domain (family K2, audit round item 7): a
    /// CPU `BF16` input fails [`gelu_admission_predicate`]'s
    /// `dtype_f32_only_on_cpu` check, so under `Strict` `gelu_erf`'s own
    /// `admit()` call must return `KernelError::StrictModeFallback`
    /// instead of silently falling back to the eager `gelu_erf()` call —
    /// mirroring `crate::bert::tests::bert_strict_mode_on_a_refused_domain_is_a_typed_error_in_a_fresh_process`'s
    /// shape exactly, including the fresh-child-process isolation
    /// `JAMMI_KERNELS_STRICT`'s process-wide `OnceLock` requires.
    #[test]
    fn gelu_erf_strict_mode_on_a_refused_domain_is_a_typed_error_in_a_fresh_process() {
        let exe = std::env::current_exe().expect("test binary path");
        let output = std::process::Command::new(exe)
            .args([
                "activations::tests::strict_mode_child_process_body",
                "--exact",
                "--nocapture",
                "--ignored",
            ])
            .env("JAMMI_KERNELS_STRICT", "1")
            .output()
            .expect("spawn child test binary");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "child process assertion failed: stdout={stdout}\nstderr={}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            stdout.contains("1 passed"),
            "the child process must have actually run (and passed) exactly one test -- \
             stdout={stdout}"
        );
    }

    /// Only meaningful inside the child process the test above spawns.
    /// `#[ignore]`d so the NORMAL (non-Strict) test run never executes it
    /// directly — only the `--ignored --exact` child-process invocation
    /// does.
    #[test]
    #[ignore]
    fn strict_mode_child_process_body() {
        let device = Device::Cpu;
        let x = Tensor::zeros((2, 4), DType::BF16, &device).unwrap();
        let before = GELU_DISPATCH_COUNTERS.snapshot();
        let err = gelu_erf(&x, true)
            .expect_err("CPU BF16 under Strict must be a typed refusal, not a silent eager");
        let after = GELU_DISPATCH_COUNTERS.snapshot();
        let msg = err.to_string();
        assert!(
            msg.contains("gelu_erf_fused") && msg.contains("dtype_f32_only_on_cpu"),
            "expected a StrictModeFallback naming the refused op/predicate: {msg}"
        );
        assert_eq!(
            after.fused, before.fused,
            "the fused counter must stay UNTOUCHED under Strict -- a Strict refusal never \
             dispatches fused, it errors instead of falling back"
        );
    }

    /// Training's fused arm is a COUNTED dispatch through the real
    /// `GeluErfFused` `CustomOp1` (see [`dispatch_gelu_erf_fused`]'s own
    /// doc) — CPU F32 is bit-identical to the eager call it replaces, so
    /// `got == want` here is a genuine numeric proof, not merely a counter
    /// proof.
    #[test]
    fn gelu_erf_training_admits_fused_on_a_supported_shape_and_matches_eager_value() {
        // Two-sided under the crate-shared counter lock (audit round item
        // 8): this test reads the SAME process-wide `gelu_erf_fused`
        // registry every other GELU/attention counter test in this crate
        // reads, so it must serialize against them the same way
        // `bert`/`distilbert`/`modernbert`'s own counter tests do.
        let _guard = crate::attention_cascade::ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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
        assert_eq!(
            after.eager, before.eager,
            "a supported shape must NOT also bump the eager count -- only the fused arm ran"
        );
    }

    /// The crate's own two-sided GELU counter oracle, now genuinely
    /// exercising the real fused kernel (fix-round item 1): `gelu_erf(x,
    /// true)` on a head64-shaped (`h=2, d=64`), sign-mixed
    /// production-amplitude fixture must (a) bump `gelu_fused_dispatches`,
    /// and (b) produce hidden states BIT-IDENTICAL to the unchanged eager
    /// `x.gelu_erf()` call — CPU F32 is `GeluErfFused`'s own documented
    /// contract (`jammi_kernels::ops::gelu_erf`'s module doc, "three cdf
    /// formulations", item 1: this op's CPU F32 arm is bit-identical to
    /// `Tensor::gelu_erf`), not merely a tolerance match. Both arms called
    /// directly — the seam's own fused dispatch (through `gelu_erf`,
    /// admission included) and the plain eager call — the way the crate's
    /// other tolerance oracles (e.g. `bert::tests::
    /// bert_head64_fused_attention_matches_eager_composition_within_tolerance`)
    /// isolate the eager arm without touching any process-wide admission
    /// switch (`JAMMI_KERNELS_DISABLE`). Two-sided under the crate-shared
    /// counter lock (audit round item 8).
    #[test]
    fn gelu_erf_training_fused_forward_is_bit_identical_to_eager_on_head64_fixture() {
        let _guard = crate::attention_cascade::ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        // head64-shaped fixture: b=2, s=5, h=2, d=64 attention output width
        // (hd=128), fed through a 4x FFN blow-up (intermediate=512) — the
        // real `BertIntermediate`/`DistilBertFfn` shape this seam is wired
        // at (`bert.rs`'s `BertIntermediate::forward`, `distilbert.rs`'s
        // `DistilBertFfn::forward` — see `gelu_erf`'s own doc for the
        // current line numbers).
        let (b, s, h, d) = (2usize, 5usize, 2usize, 64usize);
        let intermediate = 4 * h * d;
        let n = b * s * intermediate;
        let v: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32; // 0..1
                (t * 2.0 - 1.0) * 20.0 // sign-mixed, spanning +-20 (production amplitude)
            })
            .collect();
        let x = Tensor::from_slice(&v, (b, s, intermediate), &device).unwrap();

        let before = GELU_DISPATCH_COUNTERS.snapshot();
        let fused_out: Vec<f32> = gelu_erf(&x, true)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let after = GELU_DISPATCH_COUNTERS.snapshot();
        assert!(
            after.fused > before.fused,
            "a supported F32/contiguous/non-empty CPU tensor must dispatch fused"
        );
        assert_eq!(
            after.eager, before.eager,
            "a supported shape must NOT also bump the eager count -- only the fused arm ran"
        );

        let eager_direct: Vec<f32> = x
            .gelu_erf()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(fused_out.len(), eager_direct.len());
        for (i, (&fv, &ev)) in fused_out.iter().zip(eager_direct.iter()).enumerate() {
            assert!(
                fv.to_bits() == ev.to_bits(),
                "elem[{i}]: seam's fused-arm output {fv} (0x{:08x}) vs direct eager {ev} \
                 (0x{:08x}) must be bit-identical on CPU F32 (GeluErfFused's own contract)",
                fv.to_bits(),
                ev.to_bits()
            );
        }
    }

    /// This file's own re-derivation of `jammi_kernels::ops::gelu_erf`'s
    /// documented backward CONDITION-AWARE bound formula, now IMPORTING
    /// that op's own `COND_AWARE_TOL`/`COND_AWARE_ABS_FLOOR` constants
    /// (audit round item 4: these used to be hand-copied `f64` literals
    /// here, a duplication the kernels crate's own audit round closed by
    /// exporting them as `pub const` — see `gelu_erf.rs`'s own doc for
    /// both constants' derivation; this file no longer has a second,
    /// independently-drifting copy of either number). `jammi-encoders` has
    /// no `libm` dependency of its own, so `Phi(x)` is computed via
    /// `candle_core::Tensor::erf` (the same standard-normal-CDF identity
    /// `Phi(x) = 0.5*(1+erf(x/sqrt(2)))`) rather than a direct `libm::erf`
    /// call — numerically equivalent for this bound's purpose (an f32
    /// `erf` evaluation is well within the bound's own floor).
    fn gelu_erf_cond_aware_bound_batch(device: &Device, xs: &[f32]) -> Vec<f64> {
        use jammi_kernels::ops::gelu_erf::{COND_AWARE_ABS_FLOOR, COND_AWARE_TOL};
        let scaled: Vec<f32> = xs
            .iter()
            .map(|&x| x * std::f32::consts::FRAC_1_SQRT_2)
            .collect();
        let erf_v: Vec<f32> = Tensor::from_slice(&scaled, (scaled.len(),), device)
            .unwrap()
            .erf()
            .unwrap()
            .to_vec1()
            .unwrap();
        xs.iter()
            .zip(erf_v.iter())
            .map(|(&x, &e)| {
                let phi = 0.5 * (1.0 + e as f64);
                let pdf = std::f64::consts::FRAC_2_SQRT_PI
                    * std::f64::consts::FRAC_1_SQRT_2
                    * 0.5
                    * (-0.5 * x as f64 * x as f64).exp();
                let x_phi = x as f64 * pdf;
                COND_AWARE_TOL * (phi.abs() + x_phi.abs()) + COND_AWARE_ABS_FLOOR
            })
            .collect()
    }

    /// Backward through a LoRA-targeted `intermediate.dense`/`lin1` site
    /// (contract fix-round item 1): [`Tensor::backward`] on the fused arm's
    /// output must reach the LoRA `A`/`B` gradients within
    /// `jammi_kernels::ops::gelu_erf`'s own documented condition-aware
    /// bound, not merely finite/non-zero. Deliberately minimal shape
    /// (`in_features = rank = 1`, one row) so the propagated bound on
    /// `dA`/`dB` is EXACT closed-form arithmetic rather than a derived
    /// operator-norm approximation: with `hidden[o] = base[o] + scaling *
    /// x0 * a * b[o]`, `d(hidden[o])/da = scaling * x0 * b[o]` and
    /// `d(hidden[o])/d(b[o]) = scaling * x0 * a`, so the per-element bound
    /// on `|dHidden_fused[o] - dHidden_eager[o]| <=
    /// cond_aware_bound(hidden[o])` (this op's own oracle) propagates
    /// LINEARLY to `|dA_fused - dA_eager| <= scaling*|x0| * sum_o
    /// cond_aware_bound(hidden[o]) * |b[o]|` and `|dB_fused[o] -
    /// dB_eager[o]| <= scaling*|x0|*|a| * cond_aware_bound(hidden[o])` —
    /// both computed directly below (no autodiff replica needed at this
    /// shape). A 2x safety margin sits on top of the ideal-linear
    /// propagation to absorb the LoRA linear's own (unrelated,
    /// floating-point-only) rounding noise between the two independent
    /// `backward()` calls; `hidden` itself is shared bit-for-bit between
    /// both arms (the SAME `Tensor`, built once), so any measured
    /// divergence is a real fused-vs-eager GELU numeric difference
    /// reaching the LoRA gradient, never a fixture difference.
    #[test]
    fn gelu_erf_training_fused_backward_matches_eager_on_lora_targeted_dense_within_condition_aware_bound(
    ) {
        use candle_nn::{Linear, VarBuilder, VarMap};
        use jammi_lora::{FrozenBase, LoraInitMode, LoraLinear};

        // Two-sided under the crate-shared counter lock (audit round item
        // 8) — same rationale as the sibling tests above.
        let _guard = crate::attention_cascade::ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let hd = 6usize; // "intermediate.dense"/"lin1" out_features stand-in.
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Sign-mixed base, including the backward's own documented
        // crossing point `x* = -0.7517915` (`jammi_kernels::ops::gelu_erf`'s
        // module doc / `ops::gelu_erf`'s test grid) and a boundary `0.0`.
        let base_v: Vec<f32> = vec![-6.0, -3.0, -0.7517915, 0.0, 2.0, 6.0];
        let base = Linear::new(Tensor::from_vec(base_v, (hd, 1), &device).unwrap(), None);
        let lin = LoraLinear::new_with_base(
            FrozenBase::Dense(base),
            1,
            4.0,
            false,
            LoraInitMode::Gaussian,
            None,
            37,
            &varmap,
            &vb.pp("intermediate_dense"),
        )
        .unwrap();
        let scaling = lin.scaling();

        let x0 = 6.5f32;
        let x_in = Tensor::from_vec(vec![x0], (1, 1), &device).unwrap();
        let hidden = lin.forward(&x_in).unwrap(); // [1, hd], shared bit-for-bit by both arms.
        let hidden_v: Vec<f32> = hidden.flatten_all().unwrap().to_vec1().unwrap();

        let params = lin.trainable_params(); // [lora_a, lora_b] (`LoraLinear::trainable_params`'s own order).
        let a_var = params[0];
        let b_var = params[1];
        let a_val: f32 = a_var.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        let b_val: Vec<f32> = b_var.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Fused arm: through the real seam, admission included.
        let before = GELU_DISPATCH_COUNTERS.snapshot();
        let out_fused = gelu_erf(&hidden, true).unwrap();
        let after = GELU_DISPATCH_COUNTERS.snapshot();
        assert!(
            after.fused > before.fused,
            "must dispatch fused at this shape"
        );
        assert_eq!(
            after.eager, before.eager,
            "a supported shape must NOT also bump the eager count -- only the fused arm ran"
        );
        let grads_fused = out_fused
            .sum_all()
            .unwrap()
            .backward()
            .expect("backward through the fused GELU seam");

        // Eager arm: the unchanged, direct `Tensor::gelu_erf()` call on the
        // SAME `hidden` tensor built above.
        let grads_eager = hidden
            .gelu_erf()
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .expect("backward through the eager gelu_erf call");

        let bounds = gelu_erf_cond_aware_bound_batch(&device, &hidden_v);
        const MARGIN: f64 = 2.0;
        const EPS: f64 = 1e-12;

        let da_bound: f64 = bounds
            .iter()
            .zip(b_val.iter())
            .map(|(&bd, &bo)| bd * (bo as f64).abs())
            .sum::<f64>()
            * scaling.abs()
            * (x0 as f64).abs();
        let da_fused: f32 = grads_fused
            .get(a_var)
            .expect("no fused-arm gradient reached lora_a")
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        let da_eager: f32 = grads_eager
            .get(a_var)
            .expect("no eager-arm gradient reached lora_a")
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        let delta_da = (da_fused as f64 - da_eager as f64).abs();
        assert!(
            delta_da.is_finite() && delta_da <= MARGIN * da_bound + EPS,
            "lora_a: fused {da_fused} vs eager {da_eager}, |Δ|={delta_da:e} exceeds \
             {MARGIN}x the propagated condition-aware bound {da_bound:e}"
        );

        let db_fused: Vec<f32> = grads_fused
            .get(b_var)
            .expect("no fused-arm gradient reached lora_b")
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let db_eager: Vec<f32> = grads_eager
            .get(b_var)
            .expect("no eager-arm gradient reached lora_b")
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (o, &bd) in bounds.iter().enumerate() {
            let db_bound_o = bd * scaling.abs() * (x0 as f64).abs() * (a_val as f64).abs();
            let delta = (db_fused[o] as f64 - db_eager[o] as f64).abs();
            assert!(
                delta.is_finite() && delta <= MARGIN * db_bound_o + EPS,
                "lora_b[{o}] (hidden={}): fused {} vs eager {}, |Δ|={delta:e} exceeds \
                 {MARGIN}x the propagated condition-aware bound {db_bound_o:e}",
                hidden_v[o],
                db_fused[o],
                db_eager[o]
            );
        }
    }
}
