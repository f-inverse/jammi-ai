//! `LoraLinear`'s fused epilogue (`jammi_kernels::ops::ScaledCastAdd`) —
//! integration-level oracles on top of `jammi-kernels`' own CPU-hermetic
//! kernel oracles (`crates/jammi-kernels/tests/scaled_cast_add_oracles.rs`),
//! proving the WIRING (admission, training-gate, counters), not the kernel
//! math again.
//!
//! 1. `fused_epilogue_matches_manual_eager_reconstruction_bit_exactly` —
//!    the end-to-end oracle: `LoraLinear::forward` (which, on CPU with an
//!    F32 backbone, satisfies the fused kernel's admission domain and
//!    dispatches through it) reproduces a manually-reconstructed eager
//!    `[mul, cast, add]` composition bit-for-bit — proving the wiring
//!    routes through the SAME base_out/lora_out the eager path would have
//!    used, with no other divergence introduced.
//! 2. `eval_mode_never_dispatches_the_fused_kernel` — the training-only
//!    gate: `training == false` must never touch the dispatch counters,
//!    mirroring `jammi_encoders::layer_norm`'s own eval-mode bit-identity
//!    test.
//! 3. `training_mode_on_a_supported_dtype_dispatches_fused_and_is_counted`
//!    / `training_mode_on_an_unsupported_dtype_falls_back_and_is_counted`
//!    — the positive/negative dispatch-counter proof for the esc-031
//!    golden's two branches (F32 base -> fused; F16 base -> eager
//!    fallback, matching `backbone_precision_parity`'s existing dtype
//!    split).
//! 4. `esc_031_golden_holds_through_the_fused_path_with_dispatch_proof` —
//!    the esc-031 golden (`Lora(W) == Frozen(W)` at `lora_b == 0`),
//!    re-run here with an explicit assertion that the FUSED kernel (not a
//!    fallback) is what produced the agreement, so the golden is not
//!    accidentally green only because the fused path never engaged.
//!
//! **Migrated (whole-site fusion commit):** items 3 and 4 originally
//! asserted on `lora_epilogue_dispatch_snapshot` (the single-op epilogue
//! counter). `LoraLinear::forward`'s training arm now routes through
//! `jammi_kernels::ops::LowRankResidualLinear` — the WHOLE site, not just the
//! epilogue — so these three tests now assert on
//! `lora_linear_fused_dispatch_snapshot` instead, and
//! `training_mode_on_a_supported_dtype_dispatches_fused_and_is_counted`
//! additionally pins that `lora_epilogue`'s own counter stays untouched by
//! a fused-site dispatch (documented, not silently left stale).

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap};
use jammi_lora::{
    lora_epilogue_dispatch_snapshot, lora_linear_fused_dispatch_snapshot, LoraInitMode, LoraLinear,
    MaybeLoraLinear,
};

fn cpu() -> Device {
    Device::Cpu
}

/// Deterministic, non-degenerate base weight — same construction every
/// call, so building it twice (once for the `LoraLinear` under test, once
/// for the manual eager reconstruction) yields bit-identical tensors.
fn build_base(in_features: usize, out_features: usize, device: &Device, dtype: DType) -> Linear {
    let mut row = Vec::with_capacity(in_features * out_features);
    for i in 0..out_features {
        for j in 0..in_features {
            row.push(((i * 7 + j * 3) as f32).sin());
        }
    }
    let w = Tensor::from_vec(row, (out_features, in_features), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    Linear::new(w, None)
}

fn rand_input(device: &Device) -> Tensor {
    Tensor::randn(0f32, 1.0, (2, 5, 8), device).unwrap()
}

/// Same weight VALUES as [`build_base`], but with an EXACTLY-ZERO bias —
/// `bias.is_some()` alone is enough to force `LoraLinear::forward`'s
/// eager-fallback arm (see `lora_linear_admission_predicate`'s
/// `base_has_no_bias` check), while a zero bias adds NOTHING numerically
/// to the base output — so this is the "same math, forced-eager" fixture
/// the fused-vs-eager cross-arm oracles below need: any output difference
/// between a fused-arm and an eager-arm instance built with this vs
/// [`build_base`] is attributable ONLY to which arm dispatched, never to
/// the bias term itself.
fn build_base_with_zero_bias(
    in_features: usize,
    out_features: usize,
    device: &Device,
    dtype: DType,
) -> Linear {
    let base = build_base(in_features, out_features, device, dtype);
    let bias = Tensor::zeros((out_features,), dtype, device).unwrap();
    Linear::new(base.weight().clone(), Some(bias))
}

#[test]
fn fused_epilogue_matches_manual_eager_reconstruction_bit_exactly() {
    let device = cpu();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let base_for_lora = build_base(8, 16, &device, DType::F32);
    let base_for_eager = build_base(8, 16, &device, DType::F32);
    let x = rand_input(&device);
    let alpha = 8.0;
    let rank = 4;
    let scaling = alpha / rank as f64;

    let lora = LoraLinear::new(
        base_for_lora,
        rank,
        alpha,
        false,
        LoraInitMode::Gaussian,
        None,
        7,
        &varmap,
        &vb,
    )
    .unwrap();

    // On CPU with an F32 backbone the fused kernel's admission domain
    // holds (device CPU, both slots F32, both contiguous, shapes equal),
    // so `lora.forward` dispatches through `ScaledCastAdd`.
    let fused_out = lora.forward(&x).unwrap();

    // Manual eager reconstruction using the SAME lora_a/lora_b the fused
    // call used (public fields) and an independently-built, bit-identical
    // base weight — the exact `[mul, cast, add]` composition
    // `LoraLinear::forward` used to run unconditionally.
    let base_out = base_for_eager.forward(&x).unwrap();
    let a_lin = Linear::new(lora.lora_a.clone(), None);
    let after_a = a_lin.forward(&x).unwrap();
    let b_lin = Linear::new(lora.lora_b.clone(), None);
    let lora_out = b_lin.forward(&after_a).unwrap();
    let scaled = (&lora_out * scaling).unwrap();
    let manual = (&base_out + &scaled).unwrap();

    let fused_v = fused_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let manual_v = manual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(
        fused_v, manual_v,
        "the fused epilogue must reproduce the eager composition bit-for-bit at F32"
    );
}

/// The training-only gate: `LoraLinear::forward` structurally returns
/// through [`eager_epilogue`](jammi_lora) BEFORE ever calling `admit`
/// when `!self.training` (see the `if !self.training { return ... }`
/// early return in `forward`) — so eval mode can never dispatch the fused
/// kernel, regardless of dtype/device eligibility. This is proven by
/// output bit-identity against a manually-reconstructed eager composition
/// (the same technique as
/// `fused_epilogue_matches_manual_eager_reconstruction_bit_exactly`)
/// rather than a dispatch-counter delta: the counters are process-wide and
/// shared with every OTHER test in this binary running concurrently
/// (`cargo test`'s default thread-per-test model), so an exact
/// before/after equality on them would be racy — the same reason
/// `jammi_encoders::modernbert`'s own dispatch-counter oracles use
/// `>` (this fixture's call increments) rather than `==` (nothing else
/// ran) deltas.
#[test]
fn eval_mode_never_dispatches_the_fused_kernel() {
    let device = cpu();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let base_for_lora = build_base(8, 16, &device, DType::F32);
    let base_for_eager = build_base(8, 16, &device, DType::F32);
    let x = rand_input(&device);
    let alpha = 8.0;
    let rank = 4;
    let scaling = alpha / rank as f64;

    let mut lora = LoraLinear::new(
        base_for_lora,
        rank,
        alpha,
        false,
        LoraInitMode::Gaussian,
        None,
        11,
        &varmap,
        &vb,
    )
    .unwrap();
    lora.set_training(false);

    let eval_out = lora.forward(&x).unwrap();

    let base_out = base_for_eager.forward(&x).unwrap();
    let a_lin = Linear::new(lora.lora_a.clone(), None);
    let after_a = a_lin.forward(&x).unwrap();
    let b_lin = Linear::new(lora.lora_b.clone(), None);
    let lora_out = b_lin.forward(&after_a).unwrap();
    let scaled = (&lora_out * scaling).unwrap();
    let manual_eager = (&base_out + &scaled).unwrap();

    assert_eq!(
        eval_out.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        manual_eager
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        "eval mode must be bit-identical to the eager composition"
    );
}

#[test]
fn training_mode_on_a_supported_dtype_dispatches_fused_and_is_counted() {
    let device = cpu();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    // F32 backbone: within `ScaledCastAdd`'s CPU domain.
    let base = build_base(8, 16, &device, DType::F32);
    let x = rand_input(&device);

    let lora = LoraLinear::new(
        base,
        4,
        8.0,
        false,
        LoraInitMode::Gaussian,
        None,
        13,
        &varmap,
        &vb,
    )
    .unwrap();

    // `>` rather than `== +1`: the counters are process-wide and shared
    // with every other test in this binary running concurrently — see
    // `eval_mode_never_dispatches_the_fused_kernel`'s doc for why an exact
    // delta would be racy.
    let before = lora_linear_fused_dispatch_snapshot();
    // `lora_epilogue`'s OWN counter, unchanged by this forward: the P2
    // fused LoRA site never calls `ScaledCastAdd` through `admit` (it
    // reuses `ScaledCastAdd::cpu_fwd` directly, internally — see
    // `jammi_lora::lora_epilogue_counters`'s doc). A fused-dispatch
    // forward must NOT move `lora_epilogue`'s snapshot at all.
    let epilogue_before = lora_epilogue_dispatch_snapshot();
    let _ = lora.forward(&x).unwrap();
    let after = lora_linear_fused_dispatch_snapshot();
    let epilogue_after = lora_epilogue_dispatch_snapshot();
    assert!(
        after.fused > before.fused,
        "an F32-backbone training forward must dispatch the fused kernel \
         (before={before:?}, after={after:?})"
    );
    assert_eq!(
        epilogue_after, epilogue_before,
        "lora_epilogue's own counter must be untouched by a fused LoRA-site \
         dispatch (before={epilogue_before:?}, after={epilogue_after:?})"
    );
}

#[test]
fn training_mode_on_an_unsupported_dtype_falls_back_and_is_counted() {
    let device = cpu();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    // F16 backbone: candle's CPU matmul accepts F16 (unlike BF16), but
    // `ScaledCastAdd`'s CPU forward implements F32/BF16 only — the
    // admission predicate must refuse this and fall back.
    let base = build_base(8, 16, &device, DType::F16);
    let x = rand_input(&device);

    let lora = LoraLinear::new(
        base,
        4,
        8.0,
        false,
        LoraInitMode::Gaussian,
        None,
        17,
        &varmap,
        &vb,
    )
    .unwrap();

    let before = lora_linear_fused_dispatch_snapshot();
    let _ = lora.forward(&x).unwrap();
    let after = lora_linear_fused_dispatch_snapshot();
    assert!(
        after.eager > before.eager,
        "an F16-backbone training forward must fall back to eager \
         (before={before:?}, after={after:?})"
    );
}

/// esc-031's golden (`Lora(W) == Frozen(W)` at `lora_b == 0`), re-run with
/// an explicit dispatch-counter proof that the FUSED kernel — not a
/// silent fallback — produced the agreement, so a future admission-
/// predicate regression that always falls back could not hide behind this
/// test still being green.
#[test]
fn esc_031_golden_holds_through_the_fused_path_with_dispatch_proof() {
    let device = cpu();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let w = build_base(8, 16, &device, DType::F32);
    let w_for_frozen = build_base(8, 16, &device, DType::F32);
    let x = rand_input(&device);

    let lora = LoraLinear::new(
        w,
        4,
        8.0,
        false,
        LoraInitMode::ZerosB,
        None,
        19,
        &varmap,
        &vb,
    )
    .unwrap();
    let frozen = MaybeLoraLinear::Frozen(w_for_frozen);

    let before = lora_linear_fused_dispatch_snapshot();
    let lora_out = lora.forward(&x).unwrap();
    let after = lora_linear_fused_dispatch_snapshot();
    assert!(
        after.fused > before.fused,
        "the golden must be exercised THROUGH the fused kernel, not a fallback \
         (before={before:?}, after={after:?})"
    );

    let frozen_out = frozen.forward(&x).unwrap();
    assert_eq!(
        lora_out.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        frozen_out.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        "with lora_b == 0 the fused epilogue adds cast(0*scaling) == 0: Lora(W) must \
         be bit-identical to Frozen(W)"
    );
}

/// esc-033's determinism/resume oracles, restated against the PRODUCTION
/// `LoraLinear::forward` path (an adversarial mutation audit finding: the
/// prior oracle suite only exercised `DropoutMasks::apply` directly, never
/// `forward` itself — a corrupted `forward_idx` reservation at the actual
/// call site, or a divergence between the fused-site key and the
/// eager-fallback key, could have slipped through). `dispatch_fused`
/// selects which arm dispatches (`true`: F32 backbone, no bias -> fused;
/// `false`: a bias-carrying base -> eager fallback), so this proves the
/// SAME resume invariant on BOTH arms independently.
fn resume_reproduces_the_uninterrupted_dropout_stream(dispatch_fused: bool) {
    let device = cpu();
    const N: usize = 6;
    const K: u64 = 2;

    let build = |seed: u64, varmap: &VarMap, vb: &VarBuilder| -> LoraLinear {
        let (in_features, out_features) = (8, 16);
        let mut row = Vec::with_capacity(in_features * out_features);
        for i in 0..out_features {
            for j in 0..in_features {
                row.push(((i * 7 + j * 3) as f32).sin());
            }
        }
        let w = Tensor::from_vec(row, (out_features, in_features), &device).unwrap();
        let base = if dispatch_fused {
            Linear::new(w, None)
        } else {
            let bias = Tensor::zeros((out_features,), DType::F32, &device).unwrap();
            Linear::new(w, Some(bias))
        };
        LoraLinear::new(
            base,
            4,
            8.0,
            false,
            LoraInitMode::Gaussian,
            Some(0.3),
            seed,
            varmap,
            vb,
        )
        .unwrap()
    };
    let x = Tensor::ones((2, 5, 8), DType::F32, &device).unwrap();

    // Uninterrupted reference: N forwards, every output recorded.
    let ref_varmap = VarMap::new();
    let ref_vb = VarBuilder::from_varmap(&ref_varmap, DType::F32, &device);
    let reference = build(321, &ref_varmap, &ref_vb);
    let before = lora_linear_fused_dispatch_snapshot();
    let mut ref_outputs = Vec::with_capacity(N);
    for _ in 0..N {
        ref_outputs.push(
            reference
                .forward(&x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
        );
    }
    let after = lora_linear_fused_dispatch_snapshot();
    if dispatch_fused {
        assert!(
            after.fused > before.fused,
            "expected the fused arm to dispatch"
        );
    } else {
        assert!(
            after.eager > before.eager,
            "expected the eager arm to dispatch"
        );
    }

    // The "crashed" run: a separate instance, only the first K forwards.
    let interrupted_varmap = VarMap::new();
    let interrupted_vb = VarBuilder::from_varmap(&interrupted_varmap, DType::F32, &device);
    let interrupted = build(321, &interrupted_varmap, &interrupted_vb);
    for _ in 0..K {
        interrupted.forward(&x).unwrap();
    }
    let pos = interrupted.dropout_position().unwrap().unwrap();
    assert_eq!(pos, K, "dropout_position must count PRODUCTION forwards");

    // The resumed run: a FRESH instance restored to that position,
    // continuing for every remaining forward.
    let resumed_varmap = VarMap::new();
    let resumed_vb = VarBuilder::from_varmap(&resumed_varmap, DType::F32, &device);
    let resumed = build(321, &resumed_varmap, &resumed_vb);
    resumed.restore_dropout_position(pos).unwrap();
    let mut resumed_outputs = Vec::with_capacity(N - K as usize);
    for _ in 0..(N - K as usize) {
        resumed_outputs.push(
            resumed
                .forward(&x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
        );
    }

    for i in 0..(N - K as usize) {
        assert_eq!(
            resumed_outputs[i],
            ref_outputs[K as usize + i],
            "post-restore production forward {i} diverged from the uninterrupted run \
             (dispatch_fused={dispatch_fused})"
        );
    }
}

#[test]
fn fused_arm_production_path_resume_reproduces_the_uninterrupted_dropout_stream() {
    resume_reproduces_the_uninterrupted_dropout_stream(true);
}

#[test]
fn eager_arm_production_path_resume_reproduces_the_uninterrupted_dropout_stream() {
    resume_reproduces_the_uninterrupted_dropout_stream(false);
}

/// The negative control proving the oracle above has teeth (esc-033's
/// anti-relaxation clause, restated at the production `LoraLinear::forward`
/// level): restoring to `K + 1` instead of `K` must NOT reproduce the
/// uninterrupted run's continuation.
#[test]
fn fused_arm_production_path_would_catch_an_off_by_one_resume_position() {
    let device = cpu();
    const N: usize = 5;
    const K: u64 = 2;

    let build = |seed: u64, varmap: &VarMap, vb: &VarBuilder| -> LoraLinear {
        let base = build_base(8, 16, &device, DType::F32);
        LoraLinear::new(
            base,
            4,
            8.0,
            false,
            LoraInitMode::Gaussian,
            Some(0.3),
            seed,
            varmap,
            vb,
        )
        .unwrap()
    };
    let x = Tensor::ones((2, 5, 8), DType::F32, &device).unwrap();

    let ref_varmap = VarMap::new();
    let ref_vb = VarBuilder::from_varmap(&ref_varmap, DType::F32, &device);
    let reference = build(654, &ref_varmap, &ref_vb);
    let mut ref_outputs = Vec::with_capacity(N);
    for _ in 0..N {
        ref_outputs.push(
            reference
                .forward(&x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
        );
    }

    let off_by_one_varmap = VarMap::new();
    let off_by_one_vb = VarBuilder::from_varmap(&off_by_one_varmap, DType::F32, &device);
    let off_by_one = build(654, &off_by_one_varmap, &off_by_one_vb);
    off_by_one.restore_dropout_position(K + 1).unwrap(); // the injected bug: should be K.
    let wrong_output = off_by_one
        .forward(&x)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    assert_ne!(
        wrong_output, ref_outputs[K as usize],
        "an off-by-one resume position must NOT reproduce the correct continuation — \
         if it did, the positive oracle above would be vacuous"
    );
}

/// Production-level companion to `jammi_lora::seeded::tests::
/// restore_is_position_independent`: restoring is an ASSIGNMENT (not a
/// replay), so it must be exactly as fast and exactly as correct at a
/// position no replay loop could reach in test time. Two independently
/// restored `LoraLinear`s at the SAME huge position must produce
/// bit-identical `forward` output.
#[test]
fn resume_at_a_huge_position_is_correct_at_the_production_path() {
    let device = cpu();
    let base_a = build_base(8, 16, &device, DType::F32);
    let base_b = build_base(8, 16, &device, DType::F32);
    let x = Tensor::ones((2, 5, 8), DType::F32, &device).unwrap();
    // Within u32::MAX (the Philox counter's forward-index ceiling) but far
    // beyond anything a replay loop could reach in test time.
    let far = 2_000_000_000u64;

    let varmap_a = VarMap::new();
    let vb_a = VarBuilder::from_varmap(&varmap_a, DType::F32, &device);
    let lora_a = LoraLinear::new(
        base_a,
        4,
        8.0,
        false,
        LoraInitMode::Gaussian,
        Some(0.3),
        9,
        &varmap_a,
        &vb_a,
    )
    .unwrap();
    lora_a.restore_dropout_position(far).unwrap();

    let varmap_b = VarMap::new();
    let vb_b = VarBuilder::from_varmap(&varmap_b, DType::F32, &device);
    let lora_b = LoraLinear::new(
        base_b,
        4,
        8.0,
        false,
        LoraInitMode::Gaussian,
        Some(0.3),
        9,
        &varmap_b,
        &vb_b,
    )
    .unwrap();
    lora_b.restore_dropout_position(far).unwrap();

    let out_a = lora_a.forward(&x).unwrap();
    let out_b = lora_b.forward(&x).unwrap();
    assert_eq!(
        out_a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        out_b.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        "two independently-restored LoraLinears at the same huge position must \
         draw the identical dropout mask and produce bit-identical output"
    );
    assert_eq!(lora_a.dropout_position().unwrap().unwrap(), far + 1);
}

/// Production-level companion to `jammi_lora::seeded::tests::
/// forward_counter_overflow_is_a_typed_refusal_not_a_silent_wrap`: a
/// forward counter that would overflow the Philox counter's 32-bit slot
/// must surface as a typed `Err` from `LoraLinear::forward` itself (the
/// actual call site), not silently wrap into a REUSED (and therefore
/// wrongly correlated) counter value.
#[test]
fn resume_past_the_forward_counter_ceiling_is_a_typed_refusal_at_the_production_path() {
    let device = cpu();
    let base = build_base(8, 16, &device, DType::F32);
    let x = Tensor::ones((2, 5, 8), DType::F32, &device).unwrap();

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let lora = LoraLinear::new(
        base,
        4,
        8.0,
        false,
        LoraInitMode::Gaussian,
        Some(0.3),
        1,
        &varmap,
        &vb,
    )
    .unwrap();
    lora.restore_dropout_position(u32::MAX as u64 + 1).unwrap();

    assert!(
        lora.forward(&x).is_err(),
        "a forward index beyond u32::MAX must be refused by LoraLinear::forward itself, \
         not silently wrapped"
    );
}

/// (Round-2 audit finding B1): every prior resume oracle in this file
/// compares an arm against ITSELF (fused-vs-fused, or eager-vs-eager) —
/// none of them ever cross-compares the FUSED arm's actual dropout draw
/// against the EAGER arm's, so a divergence introduced ONLY on one arm's
/// key (e.g. a `forward_idx` off-by-one applied to just the fused
/// construction call) would survive the whole suite. This test closes
/// that gap directly: SAME `ResumeState` (both fresh, position 0), SAME
/// seed/prefix/config (hence identical seeded `A`/`B` AND identical
/// `layer_id`), SAME input — one instance forced fused (bias-free base),
/// one forced eager (a `build_base_with_zero_bias` base, which changes
/// NOTHING numerically) — their outputs must be BIT-IDENTICAL. This is
/// the strongest form of "the same key produces the same result" this
/// crate can state: not merely that resuming reproduces a FIXED arm's own
/// earlier run, but that the two DIFFERENT arms of the SAME logical
/// forward never diverge.
#[test]
fn fused_and_eager_arms_draw_the_bit_identical_dropout_stream_at_the_same_resume_state() {
    let device = cpu();
    let (in_features, out_features, rank) = (8usize, 16usize, 4usize);
    let seed = 999u64;
    let x_v: Vec<f32> = (0..2 * 5 * in_features)
        .map(|i| ((i as f32) * 0.23).sin())
        .collect();
    let x = Tensor::from_slice(&x_v, (2, 5, in_features), &device).unwrap();

    let varmap_f = VarMap::new();
    let vb_f = VarBuilder::from_varmap(&varmap_f, DType::F32, &device);
    let base_f = build_base(in_features, out_features, &device, DType::F32);
    let lora_f = LoraLinear::new(
        base_f,
        rank,
        8.0,
        false,
        LoraInitMode::Gaussian,
        Some(0.3),
        seed,
        &varmap_f,
        &vb_f,
    )
    .unwrap();

    let varmap_e = VarMap::new();
    let vb_e = VarBuilder::from_varmap(&varmap_e, DType::F32, &device);
    let base_e = build_base_with_zero_bias(in_features, out_features, &device, DType::F32);
    let lora_e = LoraLinear::new(
        base_e,
        rank,
        8.0,
        false,
        LoraInitMode::Gaussian,
        Some(0.3),
        seed,
        &varmap_e,
        &vb_e,
    )
    .unwrap();

    let before_f = lora_linear_fused_dispatch_snapshot();
    let y_f = lora_f.forward(&x).unwrap();
    let after_f = lora_linear_fused_dispatch_snapshot();
    assert!(
        after_f.fused > before_f.fused,
        "the bias-free fixture must actually dispatch the fused arm"
    );

    let before_e = lora_linear_fused_dispatch_snapshot();
    let y_e = lora_e.forward(&x).unwrap();
    let after_e = lora_linear_fused_dispatch_snapshot();
    assert!(
        after_e.eager > before_e.eager,
        "the zero-bias fixture must actually dispatch the eager-fallback arm"
    );

    assert_eq!(
        y_f.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        y_e.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        "same ResumeState, same seed/prefix/config, same input: the fused and \
         eager-fallback arms must draw the BIT-IDENTICAL dropout stream and \
         produce bit-identical output"
    );
}

/// The negative control proving the cross-arm oracle above has teeth: two
/// instances with DIFFERENT seeds (hence different `run_seed`, so a
/// different Philox key on both the seeded-init AND the dropout draw)
/// must NOT produce the same output. If they did, the positive oracle
/// above would be vacuously insensitive to a divergent key — exactly the
/// class of bug (one arm's key silently diverging from the other's) B1
/// exists to catch.
#[test]
fn fused_and_eager_arms_with_different_seeds_do_not_coincidentally_match() {
    let device = cpu();
    let (in_features, out_features, rank) = (8usize, 16usize, 4usize);
    let x_v: Vec<f32> = (0..2 * 5 * in_features)
        .map(|i| ((i as f32) * 0.23).sin())
        .collect();
    let x = Tensor::from_slice(&x_v, (2, 5, in_features), &device).unwrap();

    let varmap_f = VarMap::new();
    let vb_f = VarBuilder::from_varmap(&varmap_f, DType::F32, &device);
    let base_f = build_base(in_features, out_features, &device, DType::F32);
    let lora_f = LoraLinear::new(
        base_f,
        rank,
        8.0,
        false,
        LoraInitMode::Gaussian,
        Some(0.3),
        999,
        &varmap_f,
        &vb_f,
    )
    .unwrap();

    let varmap_e = VarMap::new();
    let vb_e = VarBuilder::from_varmap(&varmap_e, DType::F32, &device);
    let base_e = build_base_with_zero_bias(in_features, out_features, &device, DType::F32);
    let lora_e = LoraLinear::new(
        base_e,
        rank,
        8.0,
        false,
        LoraInitMode::Gaussian,
        Some(0.3),
        1000, // DIFFERENT seed — the injected divergence.
        &varmap_e,
        &vb_e,
    )
    .unwrap();

    let y_f = lora_f.forward(&x).unwrap();
    let y_e = lora_e.forward(&x).unwrap();
    assert_ne!(
        y_f.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        y_e.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        "a divergent key between the two arms must NOT coincidentally produce the \
         same output — if it did, the positive cross-arm oracle above would be vacuous"
    );
}

/// (Round-2 audit finding B2): the module doc's tape-node-reduction claim
/// was never MEASURED against the actual PRODUCTION `LoraLinear::forward`
/// path (only against a from-scratch reconstruction in
/// `jammi_kernels::ops::low_rank_residual_linear`'s own test suite). This
/// measures it directly, at the shape/config the doc's headline
/// describes: rank-3 `x`, `F32`, `dropout = 0.3` (a `Var`-tracked
/// intermediate — the dropout DOES add tracked nodes, unlike the
/// dropout-less measurement in `jammi_kernels::ops`), a frozen (plain,
/// non-`Var`) `w`. `Tensor::sorted_nodes()` is candle's own PUBLIC
/// topological-sort-for-backward API — the exact list `Tensor::backward`
/// walks and `GradStore::or_insert` allocates a full-size `zeros_like` +
/// `add` for.
#[test]
fn production_path_retains_fewer_tape_nodes_fused_vs_eager_fallback() {
    let device = cpu();
    let (in_features, out_features, rank) = (8usize, 16usize, 4usize);

    // FUSED arm: bias-free base.
    let varmap_f = VarMap::new();
    let vb_f = VarBuilder::from_varmap(&varmap_f, DType::F32, &device);
    let base_f = build_base(in_features, out_features, &device, DType::F32);
    let lora_f = LoraLinear::new(
        base_f,
        rank,
        8.0,
        false,
        LoraInitMode::Gaussian,
        Some(0.3),
        7,
        &varmap_f,
        &vb_f,
    )
    .unwrap();
    // `x` itself is a PLAIN (untracked) tensor here, matching the site's
    // own domain (a real call site's incoming activation is typically NOT
    // itself a bare leaf `Var`) — this isolates the LoRA SITE's own node
    // contribution (`A`/`B` and the site's own op(s)) from whatever `x`'s
    // upstream graph would separately contribute.
    let x_f = rand_input(&device);
    let before_f = lora_linear_fused_dispatch_snapshot();
    let y_f = lora_f.forward(&x_f).unwrap();
    let after_f = lora_linear_fused_dispatch_snapshot();
    assert!(
        after_f.fused > before_f.fused,
        "must actually dispatch fused"
    );
    let nodes_fused = y_f.sorted_nodes().len();

    // EAGER-FALLBACK arm: zero-bias base forces fallback, same shapes/config.
    let varmap_e = VarMap::new();
    let vb_e = VarBuilder::from_varmap(&varmap_e, DType::F32, &device);
    let base_e = build_base_with_zero_bias(in_features, out_features, &device, DType::F32);
    let lora_e = LoraLinear::new(
        base_e,
        rank,
        8.0,
        false,
        LoraInitMode::Gaussian,
        Some(0.3),
        7,
        &varmap_e,
        &vb_e,
    )
    .unwrap();
    let x_e = rand_input(&device);
    let before_e = lora_linear_fused_dispatch_snapshot();
    let y_e = lora_e.forward(&x_e).unwrap();
    let after_e = lora_linear_fused_dispatch_snapshot();
    assert!(
        after_e.eager > before_e.eager,
        "must actually dispatch eager"
    );
    let nodes_eager = y_e.sorted_nodes().len();

    assert!(
        nodes_fused < nodes_eager,
        "the fused arm must retain FEWER tape nodes than the eager-fallback arm: \
         fused={nodes_fused} eager={nodes_eager}"
    );
    // Pin the MEASURED constants directly (not just "fewer than").
    assert_eq!(nodes_fused, 5, "measured production-path FUSED node count");
    assert_eq!(nodes_eager, 11, "measured production-path EAGER node count");
}
