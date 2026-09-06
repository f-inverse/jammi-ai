//! CPU-hermetic oracles for `GeluErfFused` — the same rigor-chain pattern
//! `tests/geglu_oracles.rs` establishes, for a `CustomOp1` whose forward is
//! designed to be bit-identical to `Tensor::gelu_erf` on CPU F32 (see
//! `jammi_kernels::ops::gelu_erf`'s module doc).
//!
//! Most of this op's oracle suite (typed refusals, `track_op()`-gated
//! `bwd`, chain-rule-through-an-intermediate, the tape-node-count
//! mechanism) lives as unit tests inside `src/ops/gelu_erf.rs` itself,
//! mirroring `layer_norm`'s split. This file carries the oracles that need
//! a REAL `Tensor::backward()` walk through candle's own `Op::Unary(_,
//! GeluErf)` composition compared against the fused op, at production
//! amplitude/width, plus the KO-1 forced-defect control:
//!
//!   1. `forward_*` — bit-identity to `Tensor::gelu_erf` F32 over a
//!      sign-mixed, production-amplitude grid.
//!   2. `backward_*` — `GeluErfFused`'s backward vs. candle's OWN
//!      `Op::Unary(_, GeluErf)` gradient (`backprop.rs:624-633`'s 12-op
//!      composition, exercised through a real `.backward()` call, never a
//!      hand-rolled stand-in) within the CONDITION-AWARE bound `|Δ| <=
//!      tol*(|Phi(x)|+|x*phi(x)|) + floor` — a plain relative bound is
//!      meaningless here because `Phi(x)+x*phi(x)` (the true derivative)
//!      crosses zero near `x = -0.7517915` (see `ops::gelu_erf`'s module
//!      doc), and a relative-with-no-floor bound is ALSO insufficient on
//!      the negative tail, where `Phi(x) -> 0` (unlike the positive tail,
//!      where `Phi(x) -> 1`) so the bound's own denominator shrinks toward
//!      zero while candle's OWN backward formula computes that tiny
//!      result as a difference of two O(1) quantities —
//!      [`backward_matches_candles_own_composition_within_the_condition_aware_bound`]
//!      below is the LIVE producer for the measured floor this collapse
//!      needs: it computes the max ADDITIVE excess of `|Δ|` over the
//!      relative term alone, and its argmax `x`, and asserts `1.5x`
//!      headroom against `COND_AWARE_ABS_FLOOR` (`jammi_kernels::ops::
//!      gelu_erf`'s own doc on that constant cites this exact mechanism).
//!   3. `ko1_*` — the producer-injected control (`KO-1`): a hand-built
//!      "backward with the `x*phi(x)` term dropped" must FAIL the SAME
//!      bound somewhere on the pre-registered grid `x in +-[0.05, 4]`
//!      (no-producer: a pre-registered design choice, not a measurement —
//!      see [`ko1_dropping_the_x_phi_term_fails_the_condition_aware_bound_on_the_registered_grid`]
//!      below for the exact grid construction), as a MAX-OVER-GRID
//!      assertion (a per-point ratio is degenerate near `x = 0`, where
//!      `x*phi(x) -> 0`).
//!   4. `tape_node_count_*` — the production-width twin of the op module's
//!      own tape-shape oracle.
//!   5. `track_op_*` — a tracked, non-`Var` intermediate (`is_variable() ==
//!      false`, `track_op() == true`) yields the IDENTICAL numeric gradient
//!      a direct `Var` does, through the identity map's Jacobian.

use candle_core::{Device, Tensor, Var};
use jammi_kernels::ops::gelu_erf::{COND_AWARE_ABS_FLOOR, COND_AWARE_TOL};
use jammi_kernels::ops::{apply1, GeluErfFused};

fn fused(x: &Tensor) -> candle_core::Result<Tensor> {
    apply1(x, GeluErfFused)
}

/// `Phi(x) = 0.5*(1+erf(x/sqrt(2)))`, the standard-normal CDF, computed at
/// F64 precision via [`libm::erf`] — this file's OWN independent ground
/// truth (family F: a numpy-first-shaped reference, never re-derived from
/// this crate's own f32 kernel code under test).
fn phi_f64(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x * std::f64::consts::FRAC_1_SQRT_2))
}

/// `x*phi(x)`, `phi(x) = (1/sqrt(2*pi))*exp(-x^2/2)` the standard-normal
/// PDF — this file's own F64 ground truth for the second term of
/// `gelu_erf'(x) = Phi(x) + x*phi(x)`.
fn x_phi_f64(x: f64) -> f64 {
    let pdf = std::f64::consts::FRAC_2_SQRT_PI
        * std::f64::consts::FRAC_1_SQRT_2
        * 0.5
        * (-0.5 * x * x).exp();
    x * pdf
}

// `COND_AWARE_TOL`/`COND_AWARE_ABS_FLOOR` themselves are NOT redefined
// here: they are this crate's own design constants (`pub const` on
// `jammi_kernels::ops::gelu_erf`, imported above), and this file's own
// `backward_matches_candles_own_composition_within_the_condition_aware_bound`
// test below is the LIVE producer their doc comments (on the canonical
// definitions) cite for the floor's `1.5x`-headroom derivation — see that
// doc for the full mechanism.

/// `tol*(|Phi(x)|+|x*phi(x)|) + floor` — RESULT-relative alone would be
/// meaningless at the crossing (`Phi(x)+x*phi(x) == 0` there, the reason
/// the relative term sums the ABSOLUTE value of each half rather than
/// their possibly-cancelling sum), and relative-with-no-floor is ALSO
/// insufficient on the negative tail for the reason `COND_AWARE_ABS_FLOOR`
/// documents.
fn cond_aware_bound(x: f64) -> f64 {
    COND_AWARE_TOL * (phi_f64(x).abs() + x_phi_f64(x).abs()) + COND_AWARE_ABS_FLOOR
}

// ---------------------------------------------------------------------
// Oracle 1: forward bit-identity, sign-mixed production-amplitude grid.
// ---------------------------------------------------------------------

#[test]
fn forward_bit_identical_to_candle_gelu_erf_at_production_amplitude() {
    let device = Device::Cpu;
    let n = 4096usize;
    // Sign-mixed, spanning +-20 -- wider than the census's observed
    // production `max|qkv| ~ 9-18` (`docs/maintainer/cuda-kernel-guide.md`
    // §3.4), with the full range densely swept (not merely the endpoints).
    let v: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1) as f32; // 0..1
            (t * 2.0 - 1.0) * 20.0 // -20..20
        })
        .collect();
    let x = Tensor::from_slice(&v, (n,), &device).unwrap();
    let fused_out: Vec<f32> = fused(&x).unwrap().to_vec1().unwrap();
    let eager_out: Vec<f32> = x.gelu_erf().unwrap().to_vec1().unwrap();
    assert_eq!(fused_out.len(), n);
    for (i, (&f, &e)) in fused_out.iter().zip(eager_out.iter()).enumerate() {
        assert!(
            f.to_bits() == e.to_bits(),
            "elem[{i}] (x={}): fused {f} (0x{:08x}) vs eager {e} (0x{:08x}) must be \
             bit-identical",
            v[i],
            f.to_bits(),
            e.to_bits()
        );
    }
}

// ---------------------------------------------------------------------
// Oracle 2: backward vs. candle's real Op::Unary(_, GeluErf) composition.
// ---------------------------------------------------------------------

#[test]
fn backward_matches_candles_own_composition_within_the_condition_aware_bound() {
    let device = Device::Cpu;
    let n = 801usize; // step 0.02 over [-8, 8].
    let v: Vec<f32> = (0..n).map(|i| -8.0 + i as f32 * 0.02).collect();

    let x_fused = Var::from_tensor(&Tensor::from_slice(&v, (n,), &device).unwrap()).unwrap();
    let dx_fused: Vec<f32> = fused(&x_fused)
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap()
        .get(&x_fused)
        .unwrap()
        .to_vec1()
        .unwrap();

    // The REAL eager composition -- candle's own `Op::Unary(_, GeluErf)`
    // gradient, walked through a genuine `.backward()` call, never a
    // hand-rolled stand-in.
    let x_eager = Var::from_tensor(&Tensor::from_slice(&v, (n,), &device).unwrap()).unwrap();
    let dx_eager: Vec<f32> = x_eager
        .gelu_erf()
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap()
        .get(&x_eager)
        .unwrap()
        .to_vec1()
        .unwrap();

    let mut max_ratio = 0.0f64;
    // The floor's own derivation mechanism, made REAL (KO-4): the max
    // ADDITIVE excess of `|diff|` over the RELATIVE term alone (no floor
    // added) -- the exact residual `COND_AWARE_ABS_FLOOR` exists to cover
    // -- and its argmax `x`. This is the quantity `COND_AWARE_ABS_FLOOR`'s
    // own doc (`jammi_kernels::ops::gelu_erf`) cites as "measured": it must
    // have a LIVE producer here, not a bare, unverifiable number in prose.
    let mut max_additive_excess = f64::NEG_INFINITY;
    let mut worst_excess_x = 0.0f64;
    for (i, ((&xv, &df), &de)) in v
        .iter()
        .zip(dx_fused.iter())
        .zip(dx_eager.iter())
        .enumerate()
    {
        let diff = (df as f64 - de as f64).abs();
        let relative_term =
            COND_AWARE_TOL * (phi_f64(xv as f64).abs() + x_phi_f64(xv as f64).abs());
        let bound = relative_term + COND_AWARE_ABS_FLOOR;
        // Affirmative comparison (KO-2/3.7): a non-finite `diff` must FAIL,
        // never read as a vacuous pass.
        assert!(
            diff.is_finite() && diff <= bound,
            "x[{i}]={xv}: |fused {df} - eager {de}| = {diff} exceeds the condition-aware \
             bound {bound} (tol={COND_AWARE_TOL} * (|Phi|+|x*phi|) + {COND_AWARE_ABS_FLOOR})"
        );
        if bound > 0.0 {
            max_ratio = f64::max(max_ratio, diff / bound);
        }
        let additive_excess = diff - relative_term;
        if additive_excess > max_additive_excess {
            max_additive_excess = additive_excess;
            worst_excess_x = xv as f64;
        }
    }
    // Disclosure, not a pass/fail line (the assertion above is that):
    // printed by this test so the measured worst-case ratio has a real
    // producer, never a bare, unverifiable number in prose (KO-4).
    println!(
        "backward_matches_candles_own_composition: measured max diff/bound ratio over \
         [-8,8] step 0.02 = {max_ratio}"
    );
    // The SAME KO-4 discipline, for the floor's own derivation specifically:
    // print the exact quantity `COND_AWARE_ABS_FLOOR`'s doc cites, and its
    // argmax, on EVERY run.
    println!(
        "backward_matches_candles_own_composition: measured max additive excess (|diff| - \
         relative-term-only) over [-8,8] step 0.02 = {max_additive_excess} at x = \
         {worst_excess_x}"
    );
    assert!(
        max_additive_excess.is_finite() && max_additive_excess * 1.5 <= COND_AWARE_ABS_FLOOR,
        "COND_AWARE_ABS_FLOOR ({COND_AWARE_ABS_FLOOR}) must retain >= 1.5x headroom over the \
         measured max additive excess ({max_additive_excess} at x={worst_excess_x}) -- if this \
         fails, the floor's own derivation (a fixed 1.5x multiple of a measured worst case) is \
         stale and must be re-measured from this run's own printed value, never silently \
         loosened"
    );
}

// ---------------------------------------------------------------------
// Oracle 3 (KO-1): dropping the x*phi(x) term must fail the SAME bound.
// ---------------------------------------------------------------------

#[test]
fn ko1_dropping_the_x_phi_term_fails_the_condition_aware_bound_on_the_registered_grid() {
    // Pre-registered grid: x in +-[0.05, 4] -- deliberately excludes a
    // neighborhood of 0 (where x*phi(x) -> 0, so the dropped term itself
    // vanishes and a max-over-grid check would be degenerate there) and
    // stops at 4 (well short of where phi's own exponential tail makes the
    // dropped term negligible relative to the bound's own tiny scale).
    let mut xs: Vec<f64> = Vec::new();
    let mut t = 0.05f64;
    while t <= 4.0 + 1e-9 {
        xs.push(t);
        xs.push(-t);
        t += 0.01;
    }
    assert!(xs.len() > 100, "the registered grid must be non-trivial");

    let device = Device::Cpu;
    let v: Vec<f32> = xs.iter().map(|&x| x as f32).collect();
    let n = v.len();

    // The CORRECT reference: candle's own real backward, exactly as
    // `backward_matches_candles_own_composition_within_the_condition_aware_bound`
    // uses it.
    let x_eager = Var::from_tensor(&Tensor::from_slice(&v, (n,), &device).unwrap()).unwrap();
    let dx_correct: Vec<f32> = x_eager
        .gelu_erf()
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap()
        .get(&x_eager)
        .unwrap()
        .to_vec1()
        .unwrap();

    // The KO-1 forced defect: `dx_broken(x) = Phi(x)` alone, with the
    // `x*phi(x)` term dropped entirely -- built independently here (this
    // file's own `phi_f64`), never by reaching into `GeluErfBwdDx`'s
    // private implementation.
    let dx_broken: Vec<f64> = xs.iter().map(|&x| phi_f64(x)).collect();

    let mut max_excess = f64::NEG_INFINITY;
    let mut worst: (f64, f64, f64) = (0.0, 0.0, 0.0); // (x, diff, bound)
    for (i, &x) in xs.iter().enumerate() {
        let diff = (dx_broken[i] - dx_correct[i] as f64).abs();
        let bound = cond_aware_bound(x);
        let excess = diff - bound;
        if excess > max_excess {
            max_excess = excess;
            worst = (x, diff, bound);
        }
    }
    let (worst_x, worst_diff, worst_bound) = worst;
    // Disclosure with a real producer (KO-4): the exact margin is MEASURED
    // right here, not asserted from an external, unverifiable number.
    println!(
        "ko1 control: worst point x={worst_x}, |dropped-x*phi composition - correct| = \
         {worst_diff}, bound = {worst_bound}, excess over bound = {max_excess}"
    );
    assert!(
        max_excess.is_finite() && max_excess > 0.0,
        "KO-1 control must FAIL the condition-aware bound SOMEWHERE on the registered grid \
         (measured max excess over bound = {max_excess} <= 0 -- the control is not \
         discriminating, i.e. it would pass as if it were correct)"
    );
}

// ---------------------------------------------------------------------
// Oracle 4: tape-node count, production width.
// ---------------------------------------------------------------------

#[test]
fn tape_node_count_matches_the_op_modules_toy_width_oracle_at_production_width() {
    let device = Device::Cpu;
    let n = 2624usize; // ModernBERT-large's real intermediate_size.
    let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin() * 6.0).collect();

    let w_fused = Var::from_tensor(&Tensor::from_slice(&v, (n,), &device).unwrap()).unwrap();
    let fused_nodes = fused(&w_fused).unwrap().sorted_nodes().len();

    let w_eager = Var::from_tensor(&Tensor::from_slice(&v, (n,), &device).unwrap()).unwrap();
    let eager_nodes = w_eager.gelu_erf().unwrap().sorted_nodes().len();

    assert_eq!(fused_nodes, 2, "[x, out] regardless of width");
    assert_eq!(
        eager_nodes, fused_nodes,
        "eager's own forward tape is already a single node too, at any width"
    );
}

// ---------------------------------------------------------------------
// Oracle 5: track_op() (tracked non-Var intermediate) vs. a direct Var.
// ---------------------------------------------------------------------

#[test]
fn track_op_intermediate_and_var_produce_identical_gradients() {
    let device = Device::Cpu;
    let v: [f32; 6] = [-3.0, -0.5, 0.0, 0.5, 2.0, 5.0];
    let n = v.len();

    // Direct Var.
    let w_direct = Var::from_tensor(&Tensor::from_slice(&v, (n,), &device).unwrap()).unwrap();
    let dx_direct: Vec<f32> = fused(&w_direct)
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap()
        .get(&w_direct)
        .unwrap()
        .to_vec1()
        .unwrap();

    // Tracked, non-Var intermediate: the IDENTITY affine (`*1 + 0`) on a
    // `Var` -- `is_variable() == false`, `track_op() == true` -- so the
    // gradient w.r.t. the ORIGINAL `w` must be numerically identical to
    // the direct-`Var` case (the identity map's Jacobian is 1).
    let w_indirect = Var::from_tensor(&Tensor::from_slice(&v, (n,), &device).unwrap()).unwrap();
    let x_indirect = w_indirect.affine(1.0, 0.0).unwrap();
    assert!(
        !x_indirect.is_variable() && x_indirect.track_op(),
        "fixture invariant: x_indirect must be a tracked non-Var intermediate"
    );
    let dw_indirect: Vec<f32> = fused(&x_indirect)
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap()
        .get(&w_indirect)
        .unwrap()
        .to_vec1()
        .unwrap();

    for (i, (&d, &e)) in dx_direct.iter().zip(dw_indirect.iter()).enumerate() {
        assert!(
            (d - e).abs() < 1e-6,
            "elem[{i}]: Var-direct gradient {d} vs tracked-intermediate gradient {e} must be \
             identical (identity-map Jacobian)"
        );
    }
}
