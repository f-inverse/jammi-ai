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
//!    / `training_mode_on_an_f16_backbone_dispatches_fused_and_is_counted`
//!    / `training_mode_on_a_mismatched_dtype_pair_falls_back_and_is_counted`
//!    — the positive/negative dispatch-counter proof across the admission
//!    predicate's real dtype domain (`F32`/`BF16`/`F16` all fuse when `x`
//!    and the base weight MATCH; a mismatched pair falls back regardless
//!    of either individual dtype's own support — campaign #443 D1 widened
//!    both the fused kernel's own domain and this call-site predicate to
//!    admit `F16`; esc-076 found the predicate widening had been missed).
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
    lora_epilogue_dispatch_snapshot, lora_linear_fused_dispatch_snapshot, FrozenBase, LoraInitMode,
    LoraLinear, MaybeLoraLinear,
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

/// A fixed, deterministic, SMALL-INTEGER `f32` fixture (`{-4, .., 4}`) —
/// the SAME discipline `jammi_kernels::ops::low_rank_residual_linear`'s own
/// `exact_fixture` and `lora_linear_oracles.rs`'s copy of it use: every
/// partial sum this composition's GEMMs form from these values stays a
/// small exact integer, so a `assert_eq!` (bit-exact) claim across the
/// fused CustomOp3 path and a manually-reconstructed `[mul, cast, add]`
/// eager composition is architecture-independent BY CONSTRUCTION — those
/// two code paths legitimately hand `gemm` DIFFERENT operand stride
/// patterns for the mathematically identical reduction (the fused path
/// packs `lora_a`/`lora_b` into one row-packed `ab` buffer; the eager path
/// calls three independent `Linear::forward`s), and `gemm`'s own
/// summation-order choice can depend on that — a real, EXPECTED 1-`f32`-
/// ULP divergence this test used to hit on non-integer (`Tensor::randn`/
/// `sin`-fixture) values, NOT a wiring bug (see
/// `ops::low_rank_residual_linear`'s own module doc and
/// `lora_linear_oracles.rs`'s "oracle contract" section for the same
/// citation). This test's OWN stated purpose is proving the WIRING (same
/// `base_out`/`lora_out`, no extra divergence) — exact-integer values let
/// it keep the bit-exact assertion that purpose calls for.
fn exact_fixture(n: usize, phase: i64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = (i as i64 * 7 + phase * 13).rem_euclid(9);
            (v - 4) as f32
        })
        .collect()
}

#[test]
fn fused_epilogue_matches_manual_eager_reconstruction_bit_exactly() {
    let device = cpu();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let (in_features, out_features, rank) = (8usize, 16usize, 4usize);
    let w_v = exact_fixture(out_features * in_features, 2);
    let w_for_lora = Tensor::from_slice(&w_v, (out_features, in_features), &device).unwrap();
    let w_for_eager = w_for_lora.clone();
    let base_for_lora = Linear::new(w_for_lora, None);
    let base_for_eager = Linear::new(w_for_eager, None);
    let x_v = exact_fixture(2 * 5 * in_features, 1);
    let x = Tensor::from_slice(&x_v, (2, 5, in_features), &device).unwrap();
    let alpha = 8.0;
    let scaling = alpha / rank as f64; // 2.0, exact in binary.

    let mut lora = LoraLinear::new(
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
    // Overwrite the seeded-random draw with the SAME exact-integer
    // discipline (`forward` reads these public fields directly — see
    // `LoraLinear::forward`'s `self.lora_a`/`self.lora_b` uses — so both
    // the fused and the manual-eager reconstruction below see the
    // identical values).
    let a_v = exact_fixture(rank * in_features, 3);
    let b_v = exact_fixture(out_features * rank, 4);
    lora.lora_a = Tensor::from_slice(&a_v, (rank, in_features), &device).unwrap();
    lora.lora_b = Tensor::from_slice(&b_v, (out_features, rank), &device).unwrap();

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
fn training_mode_on_a_mismatched_dtype_pair_falls_back_and_is_counted() {
    let device = cpu();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    // `x` (`rand_input`) is always `F32`; the base weight here is `F16` —
    // a MISMATCHED `(x, w)` pair, which `lora_linear_admission_predicate`
    // must refuse regardless of whether either individual dtype is, on
    // its own, in the op's supported set (campaign #443 D1 widened that
    // set — and this predicate — to admit `F16` too, but only a genuine
    // `(F16, F16)` pair; see
    // `training_mode_on_an_f16_backbone_dispatches_fused_and_is_counted`
    // for that positive case). Named for what this fixture actually is
    // (a mismatch), not "F16 is categorically unsupported" — the doc this
    // replaced was stale the moment `F16` support landed.
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
        "a mismatched (x=F32, w=F16) training forward must fall back to eager \
         (before={before:?}, after={after:?})"
    );
}

/// The positive counterpart to
/// `training_mode_on_a_mismatched_dtype_pair_falls_back_and_is_counted`:
/// a GENUINE `(F16, F16)` pair (both `x` and the base weight) must
/// dispatch the fused kernel, exactly like the existing `F32`/`BF16` cells
/// — campaign #443 D1 widened `LowRankResidualLinear`'s (and
/// `ScaledCastAdd`'s CPU epilogue's) own domain to `F16` end to end, and
/// this predicate must actually admit it rather than leaving an `F16`
/// backbone permanently eager-only (esc-075/esc-076's own triage row: this
/// call site's `F16` gap left it silently, permanently eager; the pod's
/// own 17360-fused/0-eager trace, reproduced by this test's own
/// counter-delta assertion below, is the measured proof this widening
/// closes it). This is NOT a claimed fix for the separate `s512`
/// held-out-eval OOM esc-076 also tracked — that OOM reproduced on BOTH
/// `bf16` and `f16` `alloff` legs alike and was pinned to, and fixed at, a
/// different call site entirely (`evaluate_held_out`'s eval-batch
/// bucket-up, `be1450ae`); no causal link between the two is asserted
/// here.
#[test]
fn training_mode_on_an_f16_backbone_dispatches_fused_and_is_counted() {
    let device = cpu();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let base = build_base(8, 16, &device, DType::F16);
    let x = rand_input(&device).to_dtype(DType::F16).unwrap();

    let lora = LoraLinear::new(
        base,
        4,
        8.0,
        false,
        LoraInitMode::Gaussian,
        None,
        23,
        &varmap,
        &vb,
    )
    .unwrap();

    let before = lora_linear_fused_dispatch_snapshot();
    let epilogue_before = lora_epilogue_dispatch_snapshot();
    let _ = lora.forward(&x).unwrap();
    let after = lora_linear_fused_dispatch_snapshot();
    let epilogue_after = lora_epilogue_dispatch_snapshot();
    assert!(
        after.fused > before.fused,
        "a genuine (F16, F16) training forward must dispatch the fused kernel \
         (before={before:?}, after={after:?})"
    );
    assert_eq!(
        epilogue_after, epilogue_before,
        "lora_epilogue's own counter must be untouched by a fused LoRA-site \
         dispatch (before={epilogue_before:?}, after={epilogue_after:?})"
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
    let frozen = MaybeLoraLinear::Frozen(FrozenBase::Dense(w_for_frozen));

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
    // Exact-integer `x`/base-weight/`lora_a`/`lora_b` (see this file's
    // `exact_fixture` doc) AND `p = 0.5` (so the inverted-dropout scale
    // `1/(1-p) == 2.0` is itself exact in binary): this test's OWN claim
    // is that the FUSED and EAGER arms draw the bit-identical dropout
    // STREAM, not a re-litigation of GEMM summation-order parity (already
    // covered by `jammi-kernels`' own oracles) — exact-integer values
    // through every rounding point (including dropout's own scale)
    // isolate that claim from a legitimate, expected 1-ULP divergence the
    // fused (row-packed `ab`) and eager (three independent `Linear`
    // calls) code paths can otherwise hand `gemm` via different operand
    // stride patterns.
    let x_v = exact_fixture(2 * 5 * in_features, 21);
    let x = Tensor::from_slice(&x_v, (2, 5, in_features), &device).unwrap();
    let w_v = exact_fixture(out_features * in_features, 22);
    let a_v = exact_fixture(rank * in_features, 23);
    let b_v = exact_fixture(out_features * rank, 24);
    let p = 0.5f32;

    let varmap_f = VarMap::new();
    let vb_f = VarBuilder::from_varmap(&varmap_f, DType::F32, &device);
    let base_f = Linear::new(
        Tensor::from_slice(&w_v, (out_features, in_features), &device).unwrap(),
        None,
    );
    let mut lora_f = LoraLinear::new(
        base_f,
        rank,
        8.0,
        false,
        LoraInitMode::Gaussian,
        Some(p),
        seed,
        &varmap_f,
        &vb_f,
    )
    .unwrap();
    lora_f.lora_a = Tensor::from_slice(&a_v, (rank, in_features), &device).unwrap();
    lora_f.lora_b = Tensor::from_slice(&b_v, (out_features, rank), &device).unwrap();

    let varmap_e = VarMap::new();
    let vb_e = VarBuilder::from_varmap(&varmap_e, DType::F32, &device);
    let w_for_eager = Tensor::from_slice(&w_v, (out_features, in_features), &device).unwrap();
    let zero_bias = Tensor::zeros((out_features,), DType::F32, &device).unwrap();
    let base_e = Linear::new(w_for_eager, Some(zero_bias));
    let mut lora_e = LoraLinear::new(
        base_e,
        rank,
        8.0,
        false,
        LoraInitMode::Gaussian,
        Some(p),
        seed,
        &varmap_e,
        &vb_e,
    )
    .unwrap();
    lora_e.lora_a = Tensor::from_slice(&a_v, (rank, in_features), &device).unwrap();
    lora_e.lora_b = Tensor::from_slice(&b_v, (out_features, rank), &device).unwrap();

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

/// (Round-2 audit A3): the fused arm's `self.scaling as f32` (a plain Rust
/// narrowing cast) vs the eager arm's `lora_out * self.scaling` (an `f64`
/// scalar multiplied onto an `F32` tensor via candle's `Affine` CPU
/// kernel) — MEASURED, not assumed equal, at an rsLoRA scaling that is
/// genuinely irrational in binary: `alpha = 8, rank = 8` ->
/// `scaling = 8 / sqrt(8) = 2.8284271247461903`, which has no exact `f32`
/// representation either way.
///
/// Reading candle-core 0.11.0's own CPU `Affine` kernel
/// (`cpu_backend/mod.rs`, `impl Map1 for Affine`: `let mul =
/// T::from_f64(self.0);` then `v * mul` in `T`'s own arithmetic) shows it
/// narrows the `f64` scaling constant to `T` (here `f32`) FIRST, via
/// `f32::from_f64` — the SAME round-to-nearest `f64`->`f32` narrowing
/// Rust's `as f32` performs. So the fused arm's `self.scaling as f32` and
/// the eager arm's `T::from_f64(self.scaling)` narrow to the IDENTICAL
/// `f32` constant, and both then multiply an `f32` tensor by that SAME
/// constant — no divergence should exist at this ONE rounding point. This
/// test proves that directly (exact-integer `x`/`w`/`lora_a`/`lora_b` so
/// the ONLY non-exact value anywhere in the computation is `scaling`
/// itself), rather than leaving it assumed from reading the source alone.
#[test]
fn rslora_irrational_scaling_agrees_between_fused_and_eager_arms_at_f32() {
    let device = cpu();
    let (in_features, out_features, rank) = (8usize, 16usize, 8usize);
    let alpha = 8.0;
    let use_rslora = true; // scaling = alpha / sqrt(rank) = 2.8284271247461903, irrational in binary.

    let x_v = exact_fixture(2 * 5 * in_features, 41);
    let x = Tensor::from_slice(&x_v, (2, 5, in_features), &device).unwrap();
    let w_v = exact_fixture(out_features * in_features, 42);
    let a_v = exact_fixture(rank * in_features, 43);
    let b_v = exact_fixture(out_features * rank, 44);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let base_for_lora = Linear::new(
        Tensor::from_slice(&w_v, (out_features, in_features), &device).unwrap(),
        None,
    );
    let mut lora = LoraLinear::new(
        base_for_lora,
        rank,
        alpha,
        use_rslora,
        LoraInitMode::Gaussian,
        None,
        13,
        &varmap,
        &vb,
    )
    .unwrap();
    lora.lora_a = Tensor::from_slice(&a_v, (rank, in_features), &device).unwrap();
    lora.lora_b = Tensor::from_slice(&b_v, (out_features, rank), &device).unwrap();

    // Fused arm: bias-free F32 base on CPU, training — dispatches through
    // `LowRankResidualLinear`'s `self.scaling as f32`.
    let fused_out = lora.forward(&x).unwrap();

    // Eager arm: the SAME `eager_epilogue` formula `forward` itself uses
    // in eval mode (`lora_out * self.scaling`, `scaling: f64`), built by
    // hand here so it runs even though `lora` is in training mode.
    let base_for_eager = Linear::new(
        Tensor::from_slice(&w_v, (out_features, in_features), &device).unwrap(),
        None,
    );
    let base_out = base_for_eager.forward(&x).unwrap();
    let a_lin = Linear::new(lora.lora_a.clone(), None);
    let after_a = a_lin.forward(&x).unwrap();
    let b_lin = Linear::new(lora.lora_b.clone(), None);
    let lora_out = b_lin.forward(&after_a).unwrap();
    // `lora.scaling` is a private field (this is an external integration
    // test); recompute it via the SAME documented formula
    // (`alpha / sqrt(rank)` for rsLoRA) rather than reach for it.
    let scaling = alpha / (rank as f64).sqrt();
    let scaled = (&lora_out * scaling).unwrap();
    let eager_out = (&base_out + &scaled).unwrap();

    let fused_v = fused_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let eager_v = eager_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(
        fused_v, eager_v,
        "rsLoRA's irrational scaling must narrow to the IDENTICAL f32 constant \
         on both arms (self.scaling as f32 == candle Affine's T::from_f64(self.scaling)) \
         — a real divergence here would mean the fused epilogue silently uses a \
         DIFFERENT scaling than the documented eager formula"
    );
}

/// esc-031's quantized twin golden (issue #351): `Lora(Wq) == Frozen(Wq)`
/// at `lora_b == 0`, mirroring `esc_031_golden_holds_through_the_fused_path_
/// with_dispatch_proof`'s Dense golden above — but the PROOF shape is the
/// opposite one: `LoraLinear::forward`'s own doc states a `Quantized` base
/// NEVER touches `lora_linear_fused_counters()` at all (neither `Fused` nor
/// `Eager` — the fused kernel's domain requires a dense weight `Tensor`
/// argument, so a quantized base is never even offered to it), so the
/// dispatch-counter proof here is that BOTH counts stay UNCHANGED across
/// the forward, not that one of them increased.
mod esc_031_quantized_twin {
    use std::sync::Arc;

    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};
    use jammi_lora::{FrozenBase, LoraInitMode, LoraLinear, MaybeLoraLinear, QuantizedLinear};

    const IN: usize = 32; // Q8_0's block size — the domain [`QTensor::quantize`] requires.
    const OUT: usize = 4;
    const BATCH: usize = 3;

    /// Deterministic, non-degenerate `[OUT, IN]` weight, quantized to
    /// `Q8_0` — called TWICE per test (once for the `Frozen` arm, once for
    /// the `Lora` arm) so the two arms hold INDEPENDENTLY quantized, but
    /// content-identical, `QTensor`s (mirroring `build_base`'s own "same
    /// construction every call" doc above).
    fn quantized_weight() -> Arc<QTensor> {
        let device = Device::Cpu;
        let data: Vec<f32> = (0..OUT * IN)
            .map(|i| ((i % 17) as f32 - 8.0) / 6.0)
            .collect();
        let w = Tensor::from_vec(data, (OUT, IN), &device).unwrap();
        Arc::new(QTensor::quantize(&w, GgmlDType::Q8_0).unwrap())
    }

    fn input(device: &Device) -> Tensor {
        let data: Vec<f32> = (0..BATCH * IN)
            .map(|i| ((i % 13) as f32 - 6.0) / 5.0)
            .collect();
        Tensor::from_vec(data, (BATCH, IN), device).unwrap()
    }

    fn finite_count(t: &Tensor) -> usize {
        t.flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .filter(|v| v.is_finite())
            .count()
    }

    fn spread(t: &Tensor) -> f32 {
        let v = t
            .flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        v.iter().fold(f32::MIN, |a, b| a.max(*b)) - v.iter().fold(f32::MAX, |a, b| a.min(*b))
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        a.iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn lora_wq_equals_frozen_wq_at_lora_b_zero() {
        let device = Device::Cpu;
        let x = input(&device);

        let frozen = MaybeLoraLinear::Frozen(FrozenBase::Quantized(
            QuantizedLinear::new(quantized_weight(), None).unwrap(),
        ));

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let lora_base =
            FrozenBase::Quantized(QuantizedLinear::new(quantized_weight(), None).unwrap());
        let lora_inner = LoraLinear::new_with_base(
            lora_base,
            4,
            8.0,
            false,
            LoraInitMode::ZerosB,
            None,
            31,
            &varmap,
            &vb,
        )
        .unwrap();
        let lora = MaybeLoraLinear::Lora(lora_inner);

        // The zero-delta premise, asserted BEFORE the comparison — without
        // this, agreement below would be measuring "the LoRA contribution
        // happens to be small", not the golden itself.
        let MaybeLoraLinear::Lora(inner) = &lora else {
            unreachable!("constructed as Lora")
        };
        let b_sum = inner
            .lora_b
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(b_sum, 0.0, "ZerosB must leave lora_b exactly zero");

        // The "Quantized never touches the fused-site dispatch counters"
        // claim is NOT re-checked here via a before/after snapshot: this
        // integration-test BINARY runs every `#[test]` in this file
        // CONCURRENTLY by default, and several sibling tests here
        // (`training_mode_on_a_supported_dtype_dispatches_fused_and_is_counted`
        // and friends) deliberately increment the SAME process-global
        // counter — an exact before/after equality assertion would be
        // flaky under that concurrency (a sibling test's own Dense
        // dispatch could land between this test's two snapshots). The
        // counter-untouched claim is instead pinned, isolation-safe,
        // by `lora_linear.rs`'s own lib unit test
        // `quantized_base_forward_never_touches_the_fused_dispatch_counters`
        // (no other test in that binary calls `LoraLinear::forward` at
        // all, so a before/after equality assertion there is race-free).
        let fo = frozen.forward(&x).unwrap();
        let lo = lora.forward(&x).unwrap();

        // Non-vacuity: NaN fails every comparison bound in both directions,
        // and a zeroed matmul would make any two outputs trivially agree.
        assert_eq!(
            finite_count(&fo),
            fo.elem_count(),
            "frozen output non-finite"
        );
        assert_eq!(finite_count(&lo), lo.elem_count(), "lora output non-finite");
        assert!(spread(&fo) > 0.0, "frozen output is constant");
        assert!(spread(&lo) > 0.0, "lora output is constant");

        let delta = max_abs_diff(&fo, &lo);
        assert_eq!(
            delta, 0.0,
            "with lora_b == 0 the LoRA arm is the frozen arm: Lora(Wq) must be \
             bit-identical to Frozen(Wq), got delta={delta}"
        );
    }

    /// Positive control: the harness above must actually be able to
    /// DISCRIMINATE — with a NONZERO `lora_b` (Gaussian init), the two arms
    /// must genuinely differ, proving `lora_wq_equals_frozen_wq_at_lora_b_
    /// zero`'s `delta == 0.0` reading is a real agreement, not an artifact
    /// of a fixture that always reads `delta == 0.0` regardless of
    /// `lora_b`'s value.
    #[test]
    fn the_harness_discriminates_a_nonzero_lora_b() {
        let device = Device::Cpu;
        let x = input(&device);

        let frozen = MaybeLoraLinear::Frozen(FrozenBase::Quantized(
            QuantizedLinear::new(quantized_weight(), None).unwrap(),
        ));

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let lora_base =
            FrozenBase::Quantized(QuantizedLinear::new(quantized_weight(), None).unwrap());
        let lora_inner = LoraLinear::new_with_base(
            lora_base,
            4,
            8.0,
            false,
            LoraInitMode::Gaussian,
            None,
            31,
            &varmap,
            &vb,
        )
        .unwrap();
        let lora = MaybeLoraLinear::Lora(lora_inner);

        let fo = frozen.forward(&x).unwrap();
        let lo = lora.forward(&x).unwrap();
        let delta = max_abs_diff(&fo, &lo);
        assert!(
            delta > 1e-4,
            "control void: Gaussian-initialized lora_b must produce a genuinely \
             different output from the frozen arm, got delta={delta}"
        );
    }
}
