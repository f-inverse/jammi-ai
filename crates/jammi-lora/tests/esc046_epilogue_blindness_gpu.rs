//! esc-046 (GH#374) leg (1) BLINDNESS control, from
//! `.jammi/escapes.jsonl`'s `esc-046-lora-epilogue-rounds-delta-before-add`
//! row: the same-build forced-arm A/B (`JAMMI_KERNELS_DISABLE=
//! lora_linear_fused`, contract K-aux) is structurally BLIND to a rounding
//! defect that BOTH the fused (`jammi_kernels::ops::LowRankResidualLinear`,
//! whose epilogue reuses `ScaledCastAdd`) and eager
//! (`LoraLinear`'s own `eager_epilogue`, `lora_linear.rs:78-91`) arms carry
//! identically — forcing the fused kernel off and reading the eager
//! fallback instead proves NOTHING about which ORDER either arm rounds in,
//! because both arms always agreed (bit-identical) whether or not the bug
//! was present. esc-046's control clause (1) requires this leg to read
//! GREEN (bit-identical) both PRE-fix (the historical, buggy state) and
//! POST-fix (today's state, both arms fixed together) — a RED reading here
//! would mean the fix only touched one of the two arms, exactly the
//! regression this leg exists to catch.
//!
//! DEDICATED FILE, single test: `JAMMI_KERNELS_DISABLE` mutates process
//! environment via `std::env::set_var`, which would race every OTHER test
//! in the same binary if this lived alongside them (see
//! `crates/jammi-bench/tests/finetune_step_kernel_disable.rs`'s own note on
//! why ITS legs spawn a subprocess instead — `jammi-lora` has no bin target
//! to spawn, so this file's isolation instead comes from being cargo's own
//! unit of test-binary granularity: one file, one test, one process, no
//! peer test to race).
//!
//! CUDA-gated, not CPU: `LowRankResidualLinear`'s CPU arm cannot even ADMIT
//! a `BF16` backbone (candle-core 0.11's CPU matmul has no `BF16` impl —
//! see that op's own "CPU `BF16` matmul" module-doc section) — and `BF16`
//! is exactly the dtype esc-046's rounding-order bug requires to be
//! observable at all (an `F32` epilogue has no rounding anywhere, any
//! ordering is bit-identical trivially). This leg is therefore only
//! meaningful on CUDA.

#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap};
use jammi_lora::{LoraInitMode, LoraLinear};

/// Deterministic, non-degenerate `f32`-then-cast fixture spanning a real
/// amplitude range (not a toy small-integer one — esc-046's own bug is
/// amplitude-dependent: one bf16 ULP is `1.0` at `|base|~100`, `32` at
/// ModernBERT-large's own layer-18 residual magnitude, `-6688`), built the
/// same way every time so building it twice yields bit-identical tensors.
fn wide_fixture(n: usize, phase: i64, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = (i as i64 * 7 + phase * 13).rem_euclid(2000) - 1000;
            (v as f32 / 1000.0) * scale
        })
        .collect()
}

#[test]
fn fused_vs_eager_forced_arm_ab_is_bit_identical_both_pre_and_post_fix_bf16_cuda() {
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("esc046 blindness leg: skipping — no CUDA device available");
        return;
    };

    let (in_features, out_features, rank) = (64usize, 128usize, 16usize);
    let alpha = 32.0;
    let scaling = alpha / rank as f64;

    let w_v = wide_fixture(out_features * in_features, 2, 300.0);
    let w = Tensor::from_slice(&w_v, (out_features, in_features), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let x_v = wide_fixture(2 * 5 * in_features, 1, 3.0);
    let x = Tensor::from_slice(&x_v, (2, 5, in_features), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let build = |bias: bool| -> LoraLinear {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::BF16, &device);
        let base_bias = bias.then(|| Tensor::zeros((out_features,), DType::BF16, &device).unwrap());
        let base = Linear::new(w.clone(), base_bias);
        let mut lora = LoraLinear::new(
            base,
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
        // Overwrite the seeded-random draw with a shared deterministic
        // fixture (public fields, per `fused_epilogue.rs`'s established
        // precedent) so BOTH `LoraLinear` instances below use IDENTICAL
        // `lora_a`/`lora_b`.
        let a_v = wide_fixture(rank * in_features, 3, 5.0);
        let b_v = wide_fixture(out_features * rank, 4, 5.0);
        lora.lora_a = Tensor::from_slice(&a_v, (rank, in_features), &device).unwrap();
        lora.lora_b = Tensor::from_slice(&b_v, (out_features, rank), &device).unwrap();
        lora
    };

    // `bias: false` -> admission-eligible for the fused whole-site kernel
    // (`base_has_no_bias` domain check, `lora_linear_admission_predicate`).
    let mut lora_fused_arm = build(false);
    lora_fused_arm.set_training(true);

    std::env::set_var("JAMMI_KERNELS_DISABLE", "lora_linear_fused");
    let mut lora_eager_arm = build(false);
    lora_eager_arm.set_training(true);
    let eager_out = lora_eager_arm.forward(&x).unwrap();
    std::env::remove_var("JAMMI_KERNELS_DISABLE");

    let fused_out = lora_fused_arm.forward(&x).unwrap();

    let fused_v: Vec<half::bf16> = fused_out
        .flatten_all()
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_vec1()
        .unwrap();
    let eager_v: Vec<half::bf16> = eager_out
        .flatten_all()
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_vec1()
        .unwrap();

    assert_eq!(fused_v.len(), eager_v.len(), "output length mismatch");
    // Finiteness-affirmative (clause 4) before the bit-identity compare.
    for (i, (&f, &e)) in fused_v.iter().zip(eager_v.iter()).enumerate() {
        assert!(
            f.to_f32().is_finite() && e.to_f32().is_finite(),
            "index {i}: a non-finite value slipped through (fused={f:?}, eager={e:?})"
        );
    }
    let mismatches: Vec<usize> = fused_v
        .iter()
        .zip(eager_v.iter())
        .enumerate()
        .filter(|(_, (f, e))| f.to_bits() != e.to_bits())
        .map(|(i, _)| i)
        .collect();
    assert!(
        mismatches.is_empty(),
        "esc-046 leg (1) BLINDNESS violated: the fused arm (`LowRankResidualLinear`) and the \
         forced-eager arm (`JAMMI_KERNELS_DISABLE=lora_linear_fused` -> `eager_epilogue`) \
         disagree on {}/{} elements — they must round IDENTICALLY (both arms were fixed \
         together in the same esc-046 change; a fix that touched only one arm reads RED here, \
         not GREEN). First mismatch at index {}: fused={:?} eager={:?}",
        mismatches.len(),
        fused_v.len(),
        mismatches[0],
        fused_v[mismatches[0]],
        eager_v[mismatches[0]],
    );
}
