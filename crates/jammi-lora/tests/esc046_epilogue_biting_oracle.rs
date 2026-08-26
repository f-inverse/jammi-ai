//! esc-046 (GH#374) — the production-width, REAL-DISPATCH biting oracle.
//! `.jammi/escapes.jsonl`'s
//! `esc-046-lora-forward-epilogue-rounds-the-delta-before-the-add-vs-peft`
//! row's leg (1) BLINDNESS (a same-build fused-vs-eager A/B is structurally
//! blind to a defect BOTH arms carry identically) and controls (a) POWER OF
//! THE COMPARISON, (b) F32-TRUTH DIRECTION, (c) NON-VACUITY AND A
//! DISCRIMINATING FIXTURE, (d) NON-FINITE COUNTS AS MISMATCH, (e) THE
//! ORACLE LIVES IN THE CRATE THAT OWNS THE ARM, and (f) BOTH ARMS ASSERTED
//! SEPARATELY vs the reference.
//!
//! Companions, NOT duplicated here:
//!   - `crates/jammi-kernels/tests/scaled_cast_add_peft_rounding.rs` — the
//!     CPU-hermetic oracle for [`jammi_kernels::ops::ScaledCastAdd`]
//!     directly (the crate that owns the fused kernel), legs (2)/(3)/(4)
//!     and control (b).
//!   - `jammi_lora::lora_linear::eager_epilogue_tests` (this crate's own
//!     `src/lora_linear.rs`, `#[cfg(test)] mod`) — `eager_epilogue` itself,
//!     CPU-hermetic, called directly (not through `LoraLinear::forward`'s
//!     dispatch).
//!
//! ## Why this file is CUDA-only, not CPU-hermetic
//!
//! `lora_linear_admission_predicate`'s fused-kernel domain admits
//! `(x.dtype(), w.dtype()) == (BF16, BF16)` (see that predicate's own
//! doc) — but candle-core 0.11.0's CPU `MatMul::f` (`src/cpu_backend/
//! mod.rs`'s gemm-backed `impl Map2 for MatMul`, no `mkl`/`accelerate`
//! feature enabled anywhere in this workspace) admits only `F16 | F32 |
//! F64` for its `T::DTYPE` match and refuses `BF16` with a typed
//! `Error::UnsupportedDTypeForOp(BF16, "matmul")` BEFORE the `gemm` crate
//! is ever reached — see
//! `crates/jammi-kernels/src/ops/low_rank_residual_linear.rs`'s own
//! `bf16_base_on_cpu_is_a_typed_error_not_a_panic_or_wrong_number` test
//! for the exact citation. This is NOT specific to the fused kernel:
//! `candle_nn::Linear::forward` (the EAGER arm's own `self.base.forward`
//! call) issues the identical `x.matmul(w.t())` and hits the SAME refusal
//! — so on CPU, EITHER arm's `base_out = base.forward(x)` step fails
//! before ever reaching either arm's epilogue, for a `BF16` base. A
//! CPU-hermetic "biting oracle" that actually exercises the bf16
//! rounding-boundary regime through the REAL, full `LoraLinear::forward`
//! pipeline (both arms) is therefore not constructible at all — this is a
//! genuine candle CPU limitation (disclosed, not a defect in this fix),
//! not something a different fixture choice can route around. `cuBLAS`
//! supports `BF16` GEMMs natively, so this file runs for real only on
//! CUDA — `#![cfg(feature = "cuda")]` (never compiled by a plain
//! `cargo test -p jammi-lora`) plus a runtime `cuda_device()` probe (the
//! same `JAMMI_REQUIRE_CUDA`-panics-rather-than-skips discipline
//! `crates/jammi-kernels/tests/cuda_parity.rs`'s own `cuda_device()`
//! uses, so a broken device acquisition on the pod session this file's
//! own landing proof runs under reads as FAILED, not as a silently
//! skipped GREEN).
//!
//! THIS file is the one that drives the REAL `LoraLinear::forward` dispatch
//! (both the fused arm, via [`jammi_kernels::ops::LowRankResidualLinear`],
//! and the eager arm, via eval mode's unconditional `eager_epilogue` call)
//! with a nonzero [`jammi_lora::lora_linear_fused_dispatch_snapshot`] proof
//! — never a re-implementation of either arm's math inside this file. Both
//! arms are compared SEPARATELY against an independently-built PEFT
//! reference (real `candle_nn::Linear::forward` GEMMs + candle's own
//! `Tensor` arithmetic and `to_dtype` casts — never a copy of either arm's
//! own logic), so a defect either arm carries identically (leg (1)
//! BLINDNESS) is still caught.
//!
//! ## The fixture
//!
//! `in_features = 64`, `out_features = 4096`, `rank = 16`, `x` shape
//! `(32, 64)` (`32 * 4096 = 131072` output elements — production width,
//! `>= 4096`). `BF16` base weight and `x` (the domain
//! `lora_linear_admission_predicate` admits to the fused kernel — see that
//! function's own doc: `(x.dtype(), w.dtype())` must be `(F32,F32)` or
//! `(BF16,BF16)`; `lora_a`/`lora_b` stay `F32` regardless, matching every
//! production call site in this workspace today). Weight/input amplitudes
//! are tuned (deterministic trig fixture — the same idiom
//! `crates/jammi-kernels/src/ops/cast_scale.rs`'s own production-amplitude
//! tests use, family L: no untracked external generator) so the real GEMM
//! output lands in the `|base_out| ~ 100` regime esc-046's own lead-measured
//! reproduction used, and the LoRA delta lands in the `~3` regime that
//! measurably crosses bf16 rounding boundaries at that amplitude.

#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap};
use jammi_lora::{lora_linear_fused_dispatch_snapshot, LoraInitMode, LoraLinear};

const IN_FEATURES: usize = 64;
const OUT_FEATURES: usize = 4096;
const RANK: usize = 16;
const ROWS: usize = 32; // x: (ROWS, IN_FEATURES) -> ROWS * OUT_FEATURES = 131072 output elements.
const ALPHA: f64 = 32.0; // scaling = ALPHA / RANK = 2.0, matching esc-046's own fixture.
const MIN_DISCRIMINATING: usize = 20;

fn cuda_device() -> Option<Device> {
    match Device::new_cuda(0) {
        Ok(d) => Some(d),
        Err(e) => {
            if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
                panic!(
                    "esc046_epilogue_biting_oracle: JAMMI_REQUIRE_CUDA is set but no CUDA \
                     device could be acquired — this is a landing proof, a silent skip here is \
                     not acceptable: {e}"
                );
            }
            eprintln!("esc046_epilogue_biting_oracle: skipping — no CUDA device available ({e})");
            None
        }
    }
}

/// Deterministic trig fixture, scaled so the real GEMM output
/// (`base @ x^T`) lands near `|base_out| ~ 100` — `in_features = 64`
/// contributes a `sqrt(64) = 8` central-limit factor, so weight amplitude
/// `~6.5` and `x` amplitude `~2` (`8 * 6.5 * 2 ~= 104`) targets that
/// regime. Values are NOT bf16-exact (a "wide, non-tidy" fixture — the
/// non-vacuity partner every bit-exactness oracle in this tree uses, see
/// `scaled_cast_add_oracles.rs`'s own fixture doc).
fn build_base_weight(device: &Device) -> Tensor {
    let mut v = Vec::with_capacity(OUT_FEATURES * IN_FEATURES);
    for i in 0..OUT_FEATURES {
        for j in 0..IN_FEATURES {
            let idx = (i * IN_FEATURES + j) as f32;
            v.push(((idx * 0.0137).sin()) * 6.5);
        }
    }
    Tensor::from_slice(&v, (OUT_FEATURES, IN_FEATURES), device).unwrap()
}

fn build_x(device: &Device) -> Tensor {
    let mut v = Vec::with_capacity(ROWS * IN_FEATURES);
    for i in 0..ROWS {
        for j in 0..IN_FEATURES {
            let idx = (i * IN_FEATURES + j) as f32;
            v.push(((idx * 0.0311).cos()) * 2.0);
        }
    }
    Tensor::from_slice(&v, (ROWS, IN_FEATURES), device).unwrap()
}

/// `lora_a`/`lora_b` amplitudes tuned so `B(A(x)) * scaling` lands near
/// `~3` (esc-046's own delta regime): `after_a = A @ x^T` has dimension
/// `RANK`, `lora_out = B @ after_a^T` has dimension `OUT_FEATURES` — two
/// more `sqrt` central-limit factors (`sqrt(IN_FEATURES) = 8`,
/// `sqrt(RANK) = 4`) on top of the two weight amplitudes and `scaling`.
fn build_lora_ab(device: &Device) -> (Tensor, Tensor) {
    let mut a = Vec::with_capacity(RANK * IN_FEATURES);
    for i in 0..RANK {
        for j in 0..IN_FEATURES {
            let idx = (i * IN_FEATURES + j) as f32;
            a.push(((idx * 0.0421).sin()) * 0.35);
        }
    }
    let mut b = Vec::with_capacity(OUT_FEATURES * RANK);
    for i in 0..OUT_FEATURES {
        for j in 0..RANK {
            let idx = (i * RANK + j) as f32;
            b.push(((idx * 0.0577).cos()) * 0.18);
        }
    }
    let a_t = Tensor::from_slice(&a, (RANK, IN_FEATURES), device).unwrap();
    let b_t = Tensor::from_slice(&b, (OUT_FEATURES, RANK), device).unwrap();
    (a_t, b_t)
}

/// Widening a `BF16` tensor to `F32` (`Tensor::to_dtype`) is exact — see
/// `eager_epilogue_tests::widen_to_f32`'s own doc (`src/lora_linear.rs`)
/// for why comparing the widened `f32` values is equivalent to comparing
/// the underlying `bf16` bit patterns directly.
fn widen_to_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

struct Fixture {
    base_out_bf16: Tensor, // real GEMM, BF16
    lora_out_f32: Tensor,  // real GEMM, F32
    scaling: f64,
}

/// Builds the independent PEFT reference inputs via REAL
/// `candle_nn::Linear::forward` GEMMs on the SAME `base_weight`/`x`/
/// `lora_a`/`lora_b` a `LoraLinear` under test is constructed from — never
/// a re-implementation of `eager_epilogue`'s or `ScaledCastAdd`'s own
/// logic.
fn reference_fixture(
    base_weight: &Tensor,
    x_bf16: &Tensor,
    lora_a: &Tensor,
    lora_b: &Tensor,
    scaling: f64,
) -> Fixture {
    let base_lin = Linear::new(base_weight.clone(), None);
    let base_out_bf16 = base_lin.forward(x_bf16).unwrap();

    let x_f32 = x_bf16.to_dtype(DType::F32).unwrap();
    let a_lin = Linear::new(lora_a.clone(), None);
    let after_a = a_lin.forward(&x_f32).unwrap();
    let b_lin = Linear::new(lora_b.clone(), None);
    let lora_out_f32 = b_lin.forward(&after_a).unwrap();

    Fixture {
        base_out_bf16,
        lora_out_f32,
        scaling,
    }
}

impl Fixture {
    /// PEFT-ordered truth: widen `base_out` to `f32` (lossless), add the
    /// already-`f32`-scaled `lora_out`, round to `bf16` ONCE — via
    /// candle's own (trusted, generic) `Tensor` arithmetic and `to_dtype`,
    /// matching `peft/tuners/lora/layer.py`'s `Linear.forward` (see
    /// `ops/scaled_cast_add.rs`'s module doc for the source quote).
    fn peft_truth(&self) -> Tensor {
        let scaled = (&self.lora_out_f32 * self.scaling).unwrap();
        (self.base_out_bf16.to_dtype(DType::F32).unwrap() + &scaled)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
    }

    /// The REJECTED, pre-esc-046 formula: round the scaled delta to
    /// `bf16` FIRST, then add-and-round again.
    fn mis_ordered(&self) -> Tensor {
        let scaled = (&self.lora_out_f32 * self.scaling).unwrap();
        let scaled_rounded_first = scaled
            .to_dtype(DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        (self.base_out_bf16.to_dtype(DType::F32).unwrap() + &scaled_rounded_first)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
    }
}

#[test]
fn fused_and_eager_arms_both_match_the_peft_reference_at_production_width_esc046() {
    let Some(device) = cuda_device() else {
        return;
    };
    let base_weight = build_base_weight(&device).to_dtype(DType::BF16).unwrap();
    let x = build_x(&device).to_dtype(DType::BF16).unwrap();
    let (lora_a, lora_b) = build_lora_ab(&device);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mut lora = LoraLinear::new(
        Linear::new(base_weight.clone(), None),
        RANK,
        ALPHA,
        false,
        LoraInitMode::Gaussian,
        None,
        99,
        &varmap,
        &vb,
    )
    .unwrap();
    // Overwrite the seeded-random draw with this test's own deterministic
    // fixture (public fields — `forward` reads them directly).
    lora.lora_a = lora_a.clone();
    lora.lora_b = lora_b.clone();

    // The independent PEFT reference, built from real GEMMs on the SAME
    // base_weight/x/lora_a/lora_b — never a copy of either arm's own
    // logic (control (e)).
    let reference = reference_fixture(&base_weight, &x, &lora_a, &lora_b, lora.scaling());
    let peft_truth = reference.peft_truth();
    let mis_ordered = reference.mis_ordered();

    // --- FUSED ARM: real dispatch, with DispatchCounters provenance ---
    let before = lora_linear_fused_dispatch_snapshot();
    let fused_out = lora.forward(&x).unwrap();
    let after = lora_linear_fused_dispatch_snapshot();
    assert!(
        after.fused > before.fused,
        "control (e)/dispatch provenance: the fused arm must actually have dispatched \
         (before={before:?}, after={after:?}) — a BF16/BF16 domain forward that fell back to \
         eager would make this leg's 'fused arm' claim false"
    );

    // --- EAGER ARM: eval mode always takes `eager_epilogue` unconditionally
    // (see `LoraLinear::forward`'s own doc: "Eval/serving: always the
    // eager composition, unconditionally") — the real production eager
    // path, not a re-implementation.
    lora.set_training(false);
    let eager_out = lora.forward(&x).unwrap();

    let truth_v = widen_to_f32(&peft_truth);
    let mis_v = widen_to_f32(&mis_ordered);
    let fused_v = widen_to_f32(&fused_out);
    let eager_v = widen_to_f32(&eager_out);
    let base_out_v = widen_to_f32(&reference.base_out_bf16);
    let lora_out_v: Vec<f32> = reference
        .lora_out_f32
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let n = truth_v.len();
    assert_eq!(n, ROWS * OUT_FEATURES);
    assert_eq!(fused_v.len(), n);
    assert_eq!(eager_v.len(), n);

    // Control (d) NON-FINITE COUNTS AS MISMATCH, written affirmatively,
    // before any comparison.
    for i in 0..n {
        assert!(
            truth_v[i].is_finite()
                && mis_v[i].is_finite()
                && fused_v[i].is_finite()
                && eager_v[i].is_finite()
                && base_out_v[i].is_finite()
                && lora_out_v[i].is_finite(),
            "index {i}: a non-finite value slipped through"
        );
    }

    // Control (c) NON-VACUITY AND A DISCRIMINATING FIXTURE: the two
    // candidate formulas must genuinely separate on this REAL,
    // GEMM-produced fixture.
    let discriminating = (0..n).filter(|&i| truth_v[i] != mis_v[i]).count();
    assert!(
        discriminating >= MIN_DISCRIMINATING,
        "fixture is not discriminating: only {discriminating}/{n} elements separate the \
         once-rounded (PEFT) formula from the round-then-add (rejected) one — this fixture \
         would read GREEN on a broken build regardless of which arm dispatched"
    );

    // Control (f) BOTH ARMS ASSERTED SEPARATELY vs the reference, and the
    // DEFECT leg (post-fix: GREEN, raw bit pattern via the exact
    // bf16-widened-to-f32 comparison, never a tolerance).
    let fused_mismatches: Vec<usize> = (0..n).filter(|&i| fused_v[i] != truth_v[i]).collect();
    assert!(
        fused_mismatches.is_empty(),
        "the FUSED arm does NOT match PEFT's rounding order on {}/{n} elements (esc-046) — \
         first mismatch idx={} base_out={} lora_out={} fused={:?} peft_truth={:?}",
        fused_mismatches.len(),
        fused_mismatches[0],
        base_out_v[fused_mismatches[0]],
        lora_out_v[fused_mismatches[0]],
        fused_v[fused_mismatches[0]],
        truth_v[fused_mismatches[0]],
    );
    let eager_mismatches: Vec<usize> = (0..n).filter(|&i| eager_v[i] != truth_v[i]).collect();
    assert!(
        eager_mismatches.is_empty(),
        "the EAGER arm does NOT match PEFT's rounding order on {}/{n} elements (esc-046) — \
         first mismatch idx={} base_out={} lora_out={} eager={:?} peft_truth={:?}",
        eager_mismatches.len(),
        eager_mismatches[0],
        base_out_v[eager_mismatches[0]],
        lora_out_v[eager_mismatches[0]],
        eager_v[eager_mismatches[0]],
        truth_v[eager_mismatches[0]],
    );

    // Leg (1) BLINDNESS: fused and eager must ALSO be bit-identical to
    // EACH OTHER (the same-build A/B this whole class of defect defeats
    // when both arms carry it identically) — asserted here as a sanity
    // check that this fixture reproduces the wiring both arms share, not
    // as a substitute for the PEFT-reference comparisons above.
    assert_eq!(
        fused_v, eager_v,
        "the fused and eager arms must agree with each other (both correctly matching PEFT) — \
         a same-build A/B alone could not have distinguished a shared defect from a shared fix, \
         which is exactly why this test compares BOTH arms against the independent PEFT \
         reference above, not merely against each other"
    );

    // Control (a) POWER OF THE COMPARISON: the rejected model must
    // genuinely diverge from the REAL dispatched output (re-derived here
    // from `fused_v`, not merely from the two reference formulas above).
    let fused_vs_mis = (0..n).filter(|&i| fused_v[i] != mis_v[i]).count();
    assert!(
        fused_vs_mis >= MIN_DISCRIMINATING,
        "control (a) void: the real fused dispatch output and the rejected round-before-add \
         model must diverge on >= {MIN_DISCRIMINATING} elements for a RED-on-old-code reading \
         to mean anything; measured {fused_vs_mis}"
    );

    // Control (b) F32-TRUTH DIRECTION: on exactly the elements where the
    // two candidate formulas disagree, the once-rounded (produced) value
    // must be no farther from f64 truth (built from `base_out`'s OWN
    // already-bf16-rounded value plus the exact `f32` scaled lora delta)
    // than the round-then-add (rejected) value is, strict on at least
    // one.
    let mut strict_improvements = 0usize;
    let mut violations = 0usize;
    for i in 0..n {
        if truth_v[i] == mis_v[i] {
            continue;
        }
        let scaled_delta_f64 = f64::from(lora_out_v[i]) * lora.scaling();
        let truth_f64 = f64::from(base_out_v[i]) + scaled_delta_f64;
        let once_err = (f64::from(truth_v[i]) - truth_f64).abs();
        let old_err = (f64::from(mis_v[i]) - truth_f64).abs();
        if once_err > old_err + 1e-6 {
            violations += 1;
        }
        if once_err + 1e-6 < old_err {
            strict_improvements += 1;
        }
    }
    assert_eq!(
        violations, 0,
        "control (b) violated: on {violations} differing elements the once-rounded value is \
         FARTHER from f64 truth than the round-then-add value"
    );
    assert!(
        strict_improvements >= 1,
        "control (b) is vacuous: the once-rounded value must be STRICTLY closer to f64 truth \
         than the round-then-add value on at least one differing element"
    );
}
