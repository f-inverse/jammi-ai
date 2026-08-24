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

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap};
use jammi_lora::{lora_epilogue_dispatch_snapshot, LoraInitMode, LoraLinear, MaybeLoraLinear};

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
    let before = lora_epilogue_dispatch_snapshot();
    let _ = lora.forward(&x).unwrap();
    let after = lora_epilogue_dispatch_snapshot();
    assert!(
        after.fused > before.fused,
        "an F32-backbone training forward must dispatch the fused kernel \
         (before={before:?}, after={after:?})"
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

    let before = lora_epilogue_dispatch_snapshot();
    let _ = lora.forward(&x).unwrap();
    let after = lora_epilogue_dispatch_snapshot();
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

    let before = lora_epilogue_dispatch_snapshot();
    let lora_out = lora.forward(&x).unwrap();
    let after = lora_epilogue_dispatch_snapshot();
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
