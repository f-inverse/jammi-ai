//! CPU-hermetic oracles for `ScaledCastAdd` — the fused LoRA-site epilogue
//! (`out = base + cast(lora * scaling)`) C6 replaces the eager `[mul, cast,
//! add]` composition with.
//!
//!   1. `gradcheck_*` — bwd vs. central finite differences (f32, two
//!      scaling values so a sign error in either grad slot cannot hide).
//!      `ScaledCastAdd`'s CPU forward implements F32/BF16 storage only (no
//!      F64 arm — the op's whole reason to exist is bf16-boundary
//!      rounding, which F64 cannot exercise), unlike `Axpy`'s f64 leg.
//!   2. `fused_vs_eager_*` — fwd AND bwd vs. the eager `[mul, cast, add]`
//!      composition `LoraLinear::forward` actually runs today. `F32`/`F32`
//!      and `BF16`(base)/`F32`(lora) — the two dtype combinations
//!      `jammi-lora`'s admission predicate actually reaches — are asserted
//!      BIT-EXACT (`assert_eq!`), not merely within a tolerance: see this
//!      op's own module doc for why its rounding model was chosen to
//!      reproduce eager's round-before-add sequence rather than diverge
//!      from it the way `Axpy` deliberately does.
//!   3. `bwd_chains_through_an_intermediate_*` — the chain-rule oracle
//!      through an intermediate (non-`Var`) input, mirroring `Axpy`'s
//!      `oracles.rs`.
//!   4. `f32_base_bf16_lora_*` / `bf16_base_bf16_lora_*` — the
//!      UNREACHABLE-today `(F32, BF16)` and `(BF16, BF16)` combinations
//!      (`ScaledCastAdd` accepts them; `jammi-lora`'s admission predicate
//!      never dispatches them, since `lora_a`/`lora_b` are always `F32` in
//!      this workspace — see the op's own module doc) do NOT reproduce
//!      eager bit-for-bit, unlike the two reachable combinations oracle 2
//!      covers. Measured and bounded here (relative-with-floor, the C4/C5
//!      `bf16_close` pattern), not silently assumed equal just because the
//!      crate is publishable and a future caller could reach this
//!      combination.
//!
//! Statelessness is enforced structurally (`Copy`), same argument as
//! `oracles.rs`'s own doc — no runtime "interleaving oracle" here either.

use candle_core::{DType, Device, Tensor, Var};
use half::bf16;
use jammi_kernels::ops::{apply2, ScaledCastAdd};

fn fused_fwd(scaling: f64, base: &Tensor, lora: &Tensor) -> candle_core::Result<Tensor> {
    apply2(base, lora, ScaledCastAdd::new(scaling))
}

/// The eager `[mul, cast, add]` composition `LoraLinear::forward` actually
/// runs (see `crates/jammi-lora/src/lora_linear.rs`): `scaled = lora *
/// scaling` (in `lora`'s own dtype), cast to `base`'s dtype IF DIFFERENT,
/// then add.
fn eager_fwd(scaling: f64, base: &Tensor, lora: &Tensor) -> candle_core::Result<Tensor> {
    let scaled = (lora * scaling)?;
    let scaled_cast = if scaled.dtype() != base.dtype() {
        scaled.to_dtype(base.dtype())?
    } else {
        scaled
    };
    base + scaled_cast
}

// ---------------------------------------------------------------------
// Oracle 1: gradcheck vs. central finite differences
// ---------------------------------------------------------------------

fn gradcheck_f32(scaling: f64, eps: f32, tol: f64) {
    let device = Device::Cpu;
    let base0: [f32; 6] = [-2.0, -0.75, -0.1, 0.3, 1.2, 4.0];
    let lora0: [f32; 6] = [0.5, -1.5, 2.25, -0.4, 3.0, -0.2];

    let base = Var::from_tensor(&Tensor::from_slice(&base0, (6,), &device).unwrap()).unwrap();
    let lora = Var::from_tensor(&Tensor::from_slice(&lora0, (6,), &device).unwrap()).unwrap();

    let out = fused_fwd(scaling, &base, &lora).unwrap();
    let grads = out.backward().unwrap();
    let d_base: Vec<f32> = grads.get(&base).unwrap().to_vec1().unwrap();
    let d_lora: Vec<f32> = grads.get(&lora).unwrap().to_vec1().unwrap();

    let sum_fwd = |base: &Tensor, lora: &Tensor| -> f64 {
        fused_fwd(scaling, base, lora)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap() as f64
    };

    for i in 0..base0.len() {
        let mut bp = base0;
        bp[i] += eps;
        let mut bm = base0;
        bm[i] -= eps;
        let bp_t = Tensor::from_slice(&bp, (6,), &device).unwrap();
        let bm_t = Tensor::from_slice(&bm, (6,), &device).unwrap();
        let numeric = (sum_fwd(&bp_t, &lora) - sum_fwd(&bm_t, &lora)) / (2.0 * eps as f64);
        assert!(
            (numeric - d_base[i] as f64).abs() < tol,
            "d_base[{i}]: numeric {numeric} vs analytic {}",
            d_base[i]
        );
    }
    for i in 0..lora0.len() {
        let mut lp = lora0;
        lp[i] += eps;
        let mut lm = lora0;
        lm[i] -= eps;
        let lp_t = Tensor::from_slice(&lp, (6,), &device).unwrap();
        let lm_t = Tensor::from_slice(&lm, (6,), &device).unwrap();
        let numeric = (sum_fwd(&base, &lp_t) - sum_fwd(&base, &lm_t)) / (2.0 * eps as f64);
        assert!(
            (numeric - d_lora[i] as f64).abs() < tol,
            "d_lora[{i}]: numeric {numeric} vs analytic {}",
            d_lora[i]
        );
    }
}

#[test]
fn gradcheck_scaled_cast_add_bwd_f32() {
    gradcheck_f32(1.75, 1e-3, 5e-2);
}

#[test]
fn gradcheck_scaled_cast_add_bwd_f32_negative_scaling() {
    // A second scaling value (negative, non-unit) — the closed form is
    // `d_base = 1`, `d_lora = scaling`, so a positive-only fixture could
    // hide a sign error the way a single alpha could for Axpy.
    gradcheck_f32(-0.4, 1e-3, 5e-2);
}

// ---------------------------------------------------------------------
// Oracle 2: fused vs. eager `[mul, cast, add]` — BIT-EXACT at the
// reachable dtype combinations (see module doc).
// ---------------------------------------------------------------------

#[test]
fn fused_vs_eager_f32_f32_fwd_and_bwd_are_bit_exact() {
    let device = Device::Cpu;
    let scaling = 2.5_f64;
    let bv = [1.0f32, -2.0, 3.5, -0.25];
    let lv = [0.5f32, 1.5, -3.0, 4.0];

    let b_fused = Var::from_tensor(&Tensor::from_slice(&bv, (4,), &device).unwrap()).unwrap();
    let l_fused = Var::from_tensor(&Tensor::from_slice(&lv, (4,), &device).unwrap()).unwrap();
    let b_eager = Var::from_tensor(&Tensor::from_slice(&bv, (4,), &device).unwrap()).unwrap();
    let l_eager = Var::from_tensor(&Tensor::from_slice(&lv, (4,), &device).unwrap()).unwrap();

    let out_fused = fused_fwd(scaling, &b_fused, &l_fused).unwrap();
    let out_eager = eager_fwd(scaling, &b_eager, &l_eager).unwrap();
    assert_eq!(
        out_fused.to_vec1::<f32>().unwrap(),
        out_eager.to_vec1::<f32>().unwrap()
    );

    let grads_fused = out_fused.backward().unwrap();
    let grads_eager = out_eager.backward().unwrap();
    assert_eq!(
        grads_fused.get(&b_fused).unwrap().to_vec1::<f32>().unwrap(),
        grads_eager.get(&b_eager).unwrap().to_vec1::<f32>().unwrap(),
    );
    assert_eq!(
        grads_fused.get(&l_fused).unwrap().to_vec1::<f32>().unwrap(),
        grads_eager.get(&l_eager).unwrap().to_vec1::<f32>().unwrap(),
    );
}

/// The REJECTED alternative rounding model this op's module doc argues
/// against: accumulate the whole expression in `f32` and round ONCE at the
/// end (`Axpy`'s own precedent), rather than rounding the scaled delta to
/// `base`'s dtype FIRST and only then adding-and-rounding (what this op
/// actually implements). Used ONLY by the discrimination proof below, to
/// show a fixture exists where the two models produce DIFFERENT bf16
/// results — i.e. a regression from round-before-add to f32-accumulate
/// would be caught, not silently passed.
fn f32_accumulate_round_once(base: f32, lora: f32, scaling: f32) -> bf16 {
    bf16::from_f32(base + lora * scaling)
}

/// A wide, non-tidy fixture (values NOT chosen for bf16-exactness) — the
/// non-vacuity partner to the bit-exact assertion below. Element index 5
/// (`base = 1.0078125`, `lora = 22.508249282836914`, `scaling = 0.1`) is
/// the DISCRIMINATING element, verified by hand below: it is not enough
/// for a fixture to merely use "untidy" values — round-before-add and
/// f32-accumulate must land on OPPOSITE sides of a bf16 rounding boundary
/// for the same input, or a regression to the rejected model would pass
/// this test vacuously (the earlier version of this fixture had exactly
/// that defect: every element's rounding error was too small relative to
/// its magnitude to ever cross a rounding boundary — `|scaled delta|
/// ~= 0.2` against `|base| ~= 18.5` cannot move the sum's bf16 ULP
/// (`~0.0625` there), so round-before-add and f32-accumulate always
/// agreed on those five elements regardless of which model this op
/// actually implemented).
///
/// Hand-verified model values for element 5, at `f32`/`f64` precision:
/// `delta_f32 = 22.508249282836914 * 0.1 = 2.2508249282836914`.
/// Round-before-add (this op's actual model): round `delta_f32` to bf16
/// FIRST — `2.2508249282836914` is closest to the bf16 grid point `2.25`
/// (ULP `2^-6 = 0.015625` at this magnitude) — then add
/// `1.0078125 + 2.25 = 3.2578125` and round THAT sum to bf16: `3.2578125`
/// sits EXACTLY halfway between the grid points `3.25` and `3.265625`, so
/// round-to-nearest-even picks `3.25` (`208 * 2^-6`, even). Result: `3.25`.
/// f32-accumulate (the rejected model): sum first in `f32` —
/// `1.0078125 + 2.2508249282836914 = 3.2586374282836914` — then round
/// ONCE: this is closer to `3.265625` (`209 * 2^-6`) than to `3.25`.
/// Result: `3.265625`. The two models disagree by exactly one bf16 ULP —
/// asserted below via [`f32_accumulate_round_once`], not merely argued in
/// this comment.
// `22.508249282836914` is the exact decimal expansion of one specific f32
// bit pattern, verified by hand against candle's actual rounding (see the
// discrimination proof below) — kept at full precision rather than
// clippy's own suggested truncation (`22.508_25`) so nothing here risks
// silently landing on a DIFFERENT f32 value than the one this test's
// documented hand computation is actually about.
#[allow(clippy::excessive_precision)]
#[test]
fn fused_vs_eager_bf16_base_f32_lora_fwd_and_bwd_are_bit_exact_on_a_divergent_fixture() {
    let device = Device::Cpu;
    let scaling = 0.1_f64; // not exactly representable in bf16 or f32
    let bv: Vec<bf16> = [
        -18.5f32, -18.5, -18.5, -17.75, 12.375,
        1.0078125, // element 5: the discriminating case, see this test's doc
    ]
    .iter()
    .map(|&v| bf16::from_f32(v))
    .collect();
    let lv = [
        -2.015625f32,
        1.703125,
        2.234375,
        -2.015625,
        0.001,
        22.508249282836914,
    ];

    let base_fused = Var::from_tensor(&Tensor::from_slice(&bv, (6,), &device).unwrap()).unwrap();
    let lora_fused = Var::from_tensor(&Tensor::from_slice(&lv, (6,), &device).unwrap()).unwrap();
    let base_eager = Var::from_tensor(&Tensor::from_slice(&bv, (6,), &device).unwrap()).unwrap();
    let lora_eager = Var::from_tensor(&Tensor::from_slice(&lv, (6,), &device).unwrap()).unwrap();

    let out_fused = fused_fwd(scaling, &base_fused, &lora_fused).unwrap();
    let out_eager = eager_fwd(scaling, &base_eager, &lora_eager).unwrap();
    let fused_v: Vec<bf16> = out_fused.to_vec1().unwrap();
    let eager_v: Vec<bf16> = out_eager.to_vec1().unwrap();
    assert_eq!(out_fused.dtype(), DType::BF16);
    assert_eq!(fused_v, eager_v, "fwd must be bit-exact, not merely close");

    // The discrimination proof: this fixture is chosen so element 5's
    // rounding error crosses a bf16 rounding boundary, making the
    // assertion below non-vacuous — a regression to f32-accumulate would
    // fail it, not silently pass. Element 5's fused/eager result must
    // equal the round-before-add model's hand-computed value (`3.25`) and
    // must DIFFER from the rejected f32-accumulate model's value
    // (`3.265625`).
    let discriminating_idx = 5;
    let round_before_add = fused_v[discriminating_idx];
    let round_once = f32_accumulate_round_once(
        bv[discriminating_idx].to_f32(),
        lv[discriminating_idx],
        scaling as f32,
    );
    assert_eq!(
        round_before_add,
        bf16::from_f32(3.25),
        "round-before-add (this op's actual model) must equal the hand-computed 3.25"
    );
    assert_eq!(
        round_once,
        bf16::from_f32(3.265625),
        "f32-accumulate (the rejected model) must equal the hand-computed 3.265625"
    );
    assert_ne!(
        round_before_add, round_once,
        "the fixture must be genuinely discriminating: a regression from round-before-add \
         to f32-accumulate must change element {discriminating_idx}'s result, not agree with it"
    );

    let grads_fused = out_fused.backward().unwrap();
    let grads_eager = out_eager.backward().unwrap();
    let d_base_fused: Vec<bf16> = grads_fused.get(&base_fused).unwrap().to_vec1().unwrap();
    let d_base_eager: Vec<bf16> = grads_eager.get(&base_eager).unwrap().to_vec1().unwrap();
    assert_eq!(d_base_fused, d_base_eager);
    let d_lora_fused: Vec<f32> = grads_fused.get(&lora_fused).unwrap().to_vec1().unwrap();
    let d_lora_eager: Vec<f32> = grads_eager.get(&lora_eager).unwrap().to_vec1().unwrap();
    assert_eq!(d_lora_fused, d_lora_eager);
}

/// esc-031's own premise at the kernel level: `lora_b == 0` means
/// `lora_out == 0`, so `scaled = 0`, and `out` must be bit-identical to
/// `base` (the Frozen arm) for the reachable BF16/F32 combination too, not
/// only the F32/F32 combination the file-level test above already covers.
#[test]
fn fused_vs_eager_bf16_zero_lora_matches_frozen_arm_bit_exactly() {
    let device = Device::Cpu;
    let scaling = 8.0 / 4.0_f64; // a real, non-degenerate LoRA scaling
    let bv: Vec<bf16> = [1.0f32, -2.0, 3.5, -0.25]
        .iter()
        .map(|&v| bf16::from_f32(v))
        .collect();
    let base = Tensor::from_slice(&bv, (4,), &device).unwrap();
    let lora = Tensor::zeros((4,), DType::F32, &device).unwrap();

    let out = fused_fwd(scaling, &base, &lora).unwrap();
    assert_eq!(
        out.to_vec1::<bf16>().unwrap(),
        base.to_vec1::<bf16>().unwrap(),
        "zero LoRA delta must reproduce the frozen base bit-for-bit"
    );
}

// ---------------------------------------------------------------------
// Oracle 3: bwd always returns Some — chain-rule through an intermediate
// ---------------------------------------------------------------------

#[test]
fn bwd_chains_through_an_intermediate_non_variable_node() {
    let device = Device::Cpu;
    let w =
        Var::from_tensor(&Tensor::from_slice(&[1.0f32, 2.0, 3.0], (3,), &device).unwrap()).unwrap();
    // `lora` is an INTERMEDIATE (`w.affine(2, 0)`), not itself a `Var`.
    let lora = w.affine(2.0, 0.0).unwrap();
    assert!(!lora.is_variable());
    let base =
        Var::from_tensor(&Tensor::from_slice(&[10.0f32, 20.0, 30.0], (3,), &device).unwrap())
            .unwrap();

    // out = base + 3*(2*w) = base + 6*w
    let out = fused_fwd(3.0, &base, &lora).unwrap();
    let grads = out.backward().unwrap(); // must not panic

    let dw: Vec<f32> = grads.get(&w).unwrap().to_vec1().unwrap();
    assert_eq!(
        dw,
        vec![6.0, 6.0, 6.0],
        "d(sum(base + 6*w))/dw == 6, by the chain rule"
    );
    let d_base: Vec<f32> = grads.get(&base).unwrap().to_vec1().unwrap();
    assert_eq!(d_base, vec![1.0, 1.0, 1.0]);
}

#[test]
fn bwd_populates_a_grad_for_a_true_frozen_leaf_input_too_and_it_is_harmless() {
    let device = Device::Cpu;
    let frozen_lora = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (3,), &device).unwrap();
    assert!(!frozen_lora.is_variable());
    let base =
        Var::from_tensor(&Tensor::from_slice(&[10.0f32, 20.0, 30.0], (3,), &device).unwrap())
            .unwrap();

    let out = fused_fwd(2.0, &base, &frozen_lora).unwrap();
    let grads = out.backward().unwrap();
    assert_eq!(
        grads.get(&frozen_lora).unwrap().to_vec1::<f32>().unwrap(),
        vec![2.0, 2.0, 2.0]
    );
    assert_eq!(
        grads.get(&base).unwrap().to_vec1::<f32>().unwrap(),
        vec![1.0, 1.0, 1.0]
    );
}

// ---------------------------------------------------------------------
// Oracle 4: the UNREACHABLE-today (F32, BF16) / (BF16, BF16) combinations
// diverge from eager — measured, bounded, not assumed.
// ---------------------------------------------------------------------

/// Root cause (traced to candle-core 0.11.0's own CPU `Affine` impl,
/// `cpu_backend/mod.rs`'s `impl Map1 for Affine`: `let mul =
/// T::from_f64(self.0);` then `v * mul + add` in `T`'s own arithmetic):
/// eager's `lora_out * scaling` rounds the SCALING CONSTANT to `lora_out`'s
/// own storage dtype FIRST, then multiplies in that dtype's arithmetic.
/// When `lora_out` is `BF16`, that is an EXTRA bf16 rounding of `scaling`
/// itself that this kernel never performs (`ScaledCastAdd` always widens
/// `lora` to `f32` and multiplies by `scaling as f32` — see
/// `crate::ops::scaled_cast_add`'s CPU arms). When `lora_out` is `F32`
/// (the two combinations oracle 2 covers), `T::from_f64` targets `F32`,
/// which is exactly what this kernel already does — no divergence there.
/// So the divergence class is keyed on LORA'S dtype being `BF16`, not on
/// `base`'s dtype: both (`F32`,`BF16`) and (`BF16`,`BF16`) inherit it.
fn f32_base_bf16_lora_diffs(scaling: f64, basev: &[f32], lorav: &[f32]) -> Vec<(f64, f64)> {
    let device = Device::Cpu;
    let n = basev.len();
    let lora_bf16: Vec<bf16> = lorav.iter().map(|&v| bf16::from_f32(v)).collect();

    let base = Tensor::from_slice(basev, (n,), &device).unwrap();
    let lora = Tensor::from_slice(&lora_bf16, (n,), &device).unwrap();

    let fused: Vec<f32> = fused_fwd(scaling, &base, &lora).unwrap().to_vec1().unwrap();
    let eager: Vec<f32> = eager_fwd(scaling, &base, &lora).unwrap().to_vec1().unwrap();

    fused
        .iter()
        .zip(eager.iter())
        .map(|(&f, &e)| {
            let diff = (f as f64 - e as f64).abs();
            let magnitude = (f.abs() as f64).max(e.abs() as f64);
            (diff, magnitude)
        })
        .collect()
}

/// Relative-with-floor bound for the (`F32` base, `BF16` lora) and
/// (`BF16`, `BF16`) divergences — the C4/C5 `bf16_close` pattern (see
/// `geglu_oracles.rs`'s `BF16_REL_TOL`/`BF16_ABS_FLOOR` derivation): a pure
/// relative bound cannot describe a near-zero-crossing element (magnitude
/// near 0 on one or both sides), so an additive absolute floor covers
/// that class separately from the relative term, which covers ordinary
/// non-trivial-magnitude divergence.
///
/// Verified directly (not just argued): swept 5 non-bf16-exact `scaling`
/// values (`0.1`, `1.3/16`, `8/3`, `-2.2522`, `0.0265625`) against 2000
/// deterministic synthetic `(base, lora)` pairs each (10,000 points,
/// `base` amplitude ~50, `lora` amplitude ~30 — comparable magnitudes, so
/// the delta is never negligible relative to the sum the way the C6 audit
/// found the ORIGINAL version of this file's bit-exactness fixture to be
/// vacuous). Two divergence classes measured:
///
/// - Ordinary (large-magnitude) divergence: worst observed `0.333` at
///   `scaling = 8/3` (not exactly representable in bf16), magnitude
///   `~83-129` — `0.333/83 ~= 0.4%` relative.
/// - Near-zero-crossing divergence (small magnitude, `scaling = -2.2522`):
///   worst observed absolute diff `0.174` at magnitude `0.82` (`~21%`
///   relative — exactly the case a pure relative bound cannot cover: the
///   scaling constant's own bf16 rounding, root-caused above, shifts
///   WHICH SIDE of zero a small element lands on).
///
/// `FLOOR = 2^-2 = 0.25` covers every near-zero-crossing element measured
/// (worst `0.174`, `~1.4x` headroom) with room to spare below the worst
/// ordinary-divergence element (`0.333`) too. For elements whose diff
/// exceeds the floor, the RESIDUAL relative requirement — `(diff -
/// FLOOR) / magnitude`, maximized over every such element in the sweep —
/// measures `0.52%`; `REL = 2^-6 = 1.5625%` keeps `~3x` headroom over
/// that.
const F32_BASE_BF16_LORA_REL_TOL: f64 = 0.015625; // 2^-6
const F32_BASE_BF16_LORA_ABS_FLOOR: f64 = 0.25; // 2^-2

/// `true` iff `(diff, magnitude)` is within the stated relative-with-floor
/// bound: `diff <= FLOOR` (near-zero-crossing class, floor alone) OR
/// `(diff - FLOOR) / magnitude <= REL` (ordinary class, floor-then-relative).
fn within_f32_base_bf16_lora_bound(diff: f64, magnitude: f64) -> bool {
    diff <= F32_BASE_BF16_LORA_ABS_FLOOR
        || (diff - F32_BASE_BF16_LORA_ABS_FLOOR) / magnitude.max(1e-12)
            <= F32_BASE_BF16_LORA_REL_TOL
}

#[test]
fn f32_base_bf16_lora_diverges_and_stays_within_the_measured_relative_tolerance() {
    let scalings: [f64; 5] = [0.1, 1.3 / 16.0, 8.0 / 3.0, -2.2522, 0.0265625];
    let mut any_nonzero = false;
    for &scaling in scalings.iter() {
        let n = 2000usize;
        let basev: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.013 - 13.0).sin() * 50.0)
            .collect();
        let lorav: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.037 + 2.0).cos() * 30.0)
            .collect();
        for (diff, magnitude) in f32_base_bf16_lora_diffs(scaling, &basev, &lorav) {
            if diff > 0.0 {
                any_nonzero = true;
            }
            assert!(
                within_f32_base_bf16_lora_bound(diff, magnitude),
                "diff {diff} at magnitude {magnitude} (scaling {scaling}) exceeds the \
                 stated relative-with-floor bound (FLOOR={F32_BASE_BF16_LORA_ABS_FLOOR}, \
                 REL={F32_BASE_BF16_LORA_REL_TOL})"
            );
        }
    }
    assert!(
        any_nonzero,
        "expected the (F32, BF16) combination to genuinely diverge from eager \
         somewhere in this sweep — the tolerance above would otherwise be unexercised"
    );
}

/// The (`BF16`, `BF16`) combination inherits the SAME root cause (lora's
/// dtype is `BF16` either way) — asserted here with the same bound, on a
/// fixture that also makes `base` itself `BF16` (exercising the op's
/// fourth and last CPU dtype-combination arm, `scaled_cast_add_bf16_bf16`,
/// which no other test in this file reaches directly).
#[test]
fn bf16_base_bf16_lora_diverges_and_stays_within_the_measured_relative_tolerance() {
    let device = Device::Cpu;
    let scaling = 0.1_f64;
    let n = 2000usize;
    let basev: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.013 - 13.0).sin() * 50.0)
        .collect();
    let lorav: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.037 + 2.0).cos() * 30.0)
        .collect();
    let base_bf16: Vec<bf16> = basev.iter().map(|&v| bf16::from_f32(v)).collect();
    let lora_bf16: Vec<bf16> = lorav.iter().map(|&v| bf16::from_f32(v)).collect();

    let base = Tensor::from_slice(&base_bf16, (n,), &device).unwrap();
    let lora = Tensor::from_slice(&lora_bf16, (n,), &device).unwrap();
    let fused: Vec<f32> = fused_fwd(scaling, &base, &lora)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap();
    let eager: Vec<f32> = eager_fwd(scaling, &base, &lora)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap();

    let mut any_nonzero = false;
    for (&f, &e) in fused.iter().zip(eager.iter()) {
        let diff = (f as f64 - e as f64).abs();
        let magnitude = (f.abs() as f64).max(e.abs() as f64);
        if diff > 0.0 {
            any_nonzero = true;
        }
        assert!(
            within_f32_base_bf16_lora_bound(diff, magnitude),
            "diff {diff} at magnitude {magnitude} exceeds the stated relative-with-floor \
             bound (FLOOR={F32_BASE_BF16_LORA_ABS_FLOOR}, REL={F32_BASE_BF16_LORA_REL_TOL})"
        );
    }
    assert!(
        any_nonzero,
        "expected the (BF16, BF16) combination to genuinely diverge from eager \
         somewhere in this fixture"
    );
}
