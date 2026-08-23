//! CPU-hermetic oracles for `Axpy`, the proof op establishing the
//! CustomOp2 + build-infrastructure pattern every later fused op copies.
//!
//!   1. `gradcheck_*` — bwd vs. central finite differences (f32, f64).
//!   2. `fused_vs_eager_*` — fwd AND bwd vs. candle's own affine+add
//!      composition.
//!   3. `bwd_*` — `Axpy::bwd` always returns `Some` for both gradient
//!      slots (never gates on `Tensor::is_variable()`), including the
//!      regression case that gating broke: an INTERMEDIATE tensor on a
//!      path to a `Var`, where `is_variable() == false` but candle's own
//!      backward walk still requires a populated gradient for it.
//!
//! Statelessness is enforced STRUCTURALLY (every op is required to be
//! `Copy` — see `jammi_kernels::ops`'s module doc for why), not by a
//! runtime "interleaving oracle": `CustomOp2::apply_op2` takes the op BY
//! VALUE, so a fresh instance already backs every call regardless of the
//! op's own properties — a test that ran two forwards before either
//! backward could never actually discriminate a save-state-in-the-op-
//! struct bug under that API, so it doesn't exist here as a fake proof.
//!
//! The CUDA↔CPU parity leg lives in `tests/cuda_parity.rs` (feature-gated,
//! runs only where a CUDA device exists) — not here; this file is
//! deliberately runnable with no GPU and no network.

use candle_core::{DType, Device, Tensor, Var};
use half::bf16;
use jammi_kernels::ops::{apply2, Axpy};

fn axpy_fwd(alpha: f64, x: &Tensor, y: &Tensor) -> candle_core::Result<Tensor> {
    // Through `ops::apply2` (requires `T: KernelOp`), not
    // `Tensor::apply_op2` directly — this is the enforcement point
    // `ops`'s `KernelOp` bound exists for, exercised here rather than
    // left purely aspirational.
    apply2(x, y, Axpy::new(alpha))
}

fn eager_fwd(alpha: f64, x: &Tensor, y: &Tensor) -> candle_core::Result<Tensor> {
    x.affine(alpha, 0.0)?.add(y)
}

// ---------------------------------------------------------------------
// Oracle 1: gradcheck vs. central finite differences (f32, f64)
// ---------------------------------------------------------------------

/// `sum(alpha*x + y)`'s finite-difference derivative w.r.t. `x[i]` should
/// equal `alpha`, and w.r.t. `y[i]` should equal `1` — trivial in closed
/// form, but this exercises the SAME machinery (`apply_op2`, `.backward()`,
/// `GradStore::get`) every later, less-trivial fused op's gradcheck reuses,
/// with genuine per-element perturbation rather than asserting the closed
/// form directly.
fn gradcheck_f32(alpha: f64, eps: f32, tol: f64) {
    let device = Device::Cpu;
    let x0: [f32; 6] = [-2.0, -0.75, -0.1, 0.3, 1.2, 4.0];
    let y0: [f32; 6] = [0.5, -1.5, 2.25, -0.4, 3.0, -0.2];

    let x = Var::from_tensor(&Tensor::from_slice(&x0, (6,), &device).unwrap()).unwrap();
    let y = Var::from_tensor(&Tensor::from_slice(&y0, (6,), &device).unwrap()).unwrap();

    let out = axpy_fwd(alpha, &x, &y).unwrap();
    let grads = out.backward().unwrap();
    let dx: Vec<f32> = grads.get(&x).unwrap().to_vec1().unwrap();
    let dy: Vec<f32> = grads.get(&y).unwrap().to_vec1().unwrap();

    let sum_fwd = |x: &Tensor, y: &Tensor| -> f64 {
        axpy_fwd(alpha, x, y)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap() as f64
    };

    for i in 0..x0.len() {
        let mut xp = x0;
        xp[i] += eps;
        let mut xm = x0;
        xm[i] -= eps;
        let xp_t = Tensor::from_slice(&xp, (6,), &device).unwrap();
        let xm_t = Tensor::from_slice(&xm, (6,), &device).unwrap();
        let numeric = (sum_fwd(&xp_t, &y) - sum_fwd(&xm_t, &y)) / (2.0 * eps as f64);
        assert!(
            (numeric - dx[i] as f64).abs() < tol,
            "dx[{i}]: numeric {numeric} vs analytic {}",
            dx[i]
        );
    }
    for i in 0..y0.len() {
        let mut yp = y0;
        yp[i] += eps;
        let mut ym = y0;
        ym[i] -= eps;
        let yp_t = Tensor::from_slice(&yp, (6,), &device).unwrap();
        let ym_t = Tensor::from_slice(&ym, (6,), &device).unwrap();
        let numeric = (sum_fwd(&x, &yp_t) - sum_fwd(&x, &ym_t)) / (2.0 * eps as f64);
        assert!(
            (numeric - dy[i] as f64).abs() < tol,
            "dy[{i}]: numeric {numeric} vs analytic {}",
            dy[i]
        );
    }
}

fn gradcheck_f64(alpha: f64, eps: f64, tol: f64) {
    let device = Device::Cpu;
    let x0: [f64; 6] = [-2.0, -0.75, -0.1, 0.3, 1.2, 4.0];
    let y0: [f64; 6] = [0.5, -1.5, 2.25, -0.4, 3.0, -0.2];

    let x = Var::from_tensor(&Tensor::from_slice(&x0, (6,), &device).unwrap()).unwrap();
    let y = Var::from_tensor(&Tensor::from_slice(&y0, (6,), &device).unwrap()).unwrap();

    let out = axpy_fwd(alpha, &x, &y).unwrap();
    let grads = out.backward().unwrap();
    let dx: Vec<f64> = grads.get(&x).unwrap().to_vec1().unwrap();
    let dy: Vec<f64> = grads.get(&y).unwrap().to_vec1().unwrap();

    let sum_fwd = |x: &Tensor, y: &Tensor| -> f64 {
        axpy_fwd(alpha, x, y)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f64>()
            .unwrap()
    };

    for i in 0..x0.len() {
        let mut xp = x0;
        xp[i] += eps;
        let mut xm = x0;
        xm[i] -= eps;
        let xp_t = Tensor::from_slice(&xp, (6,), &device).unwrap();
        let xm_t = Tensor::from_slice(&xm, (6,), &device).unwrap();
        let numeric = (sum_fwd(&xp_t, &y) - sum_fwd(&xm_t, &y)) / (2.0 * eps);
        assert!(
            (numeric - dx[i]).abs() < tol,
            "dx[{i}]: numeric {numeric} vs analytic {}",
            dx[i]
        );
    }
    for i in 0..y0.len() {
        let mut yp = y0;
        yp[i] += eps;
        let mut ym = y0;
        ym[i] -= eps;
        let yp_t = Tensor::from_slice(&yp, (6,), &device).unwrap();
        let ym_t = Tensor::from_slice(&ym, (6,), &device).unwrap();
        let numeric = (sum_fwd(&x, &yp_t) - sum_fwd(&x, &ym_t)) / (2.0 * eps);
        assert!(
            (numeric - dy[i]).abs() < tol,
            "dy[{i}]: numeric {numeric} vs analytic {}",
            dy[i]
        );
    }
}

#[test]
fn gradcheck_axpy_bwd_f32() {
    // f32 storage limits how small eps can go before subtraction noise
    // dominates; 1e-3 / tol 5e-2 is the standard f32 central-difference
    // regime.
    gradcheck_f32(1.75, 1e-3, 5e-2);
}

#[test]
fn gradcheck_axpy_bwd_f64() {
    // f64 affords a much tighter eps and tolerance — this is the precise
    // leg of the oracle.
    gradcheck_f64(1.75, 1e-6, 1e-6);
}

// ---------------------------------------------------------------------
// Oracle 2: fused vs. eager (candle's own affine + add composition)
// ---------------------------------------------------------------------

#[test]
fn fused_vs_eager_f32_fwd_and_bwd_match_exactly() {
    // f32: both paths do one multiply then one add in the same order
    // (`alpha*x` then `+y`), so the two roundings coincide bit-for-bit —
    // exact-match tolerance (0 ULP), not a stated epsilon.
    let device = Device::Cpu;
    let alpha = 2.5_f64;
    let xv = [1.0f32, -2.0, 3.5, -0.25];
    let yv = [0.5f32, 1.5, -3.0, 4.0];

    let x_fused = Var::from_tensor(&Tensor::from_slice(&xv, (4,), &device).unwrap()).unwrap();
    let y_fused = Var::from_tensor(&Tensor::from_slice(&yv, (4,), &device).unwrap()).unwrap();
    let x_eager = Var::from_tensor(&Tensor::from_slice(&xv, (4,), &device).unwrap()).unwrap();
    let y_eager = Var::from_tensor(&Tensor::from_slice(&yv, (4,), &device).unwrap()).unwrap();

    let out_fused = axpy_fwd(alpha, &x_fused, &y_fused).unwrap();
    let out_eager = eager_fwd(alpha, &x_eager, &y_eager).unwrap();
    assert_eq!(
        out_fused.to_vec1::<f32>().unwrap(),
        out_eager.to_vec1::<f32>().unwrap()
    );

    let grads_fused = out_fused.backward().unwrap();
    let grads_eager = out_eager.backward().unwrap();
    assert_eq!(
        grads_fused.get(&x_fused).unwrap().to_vec1::<f32>().unwrap(),
        grads_eager.get(&x_eager).unwrap().to_vec1::<f32>().unwrap(),
    );
    assert_eq!(
        grads_fused.get(&y_fused).unwrap().to_vec1::<f32>().unwrap(),
        grads_eager.get(&y_eager).unwrap().to_vec1::<f32>().unwrap(),
    );
}

/// Exact bf16 bit distance (as a signed integer over the raw `u16` bit
/// patterns) — precise, unlike a float-subtraction-based ULP approximation.
fn bf16_bit_diff(a: bf16, b: bf16) -> i32 {
    a.to_bits() as i32 - b.to_bits() as i32
}

/// The stated fused-vs-eager bf16 tolerance: 2 ULP. The fused CPU kernel
/// accumulates in f32 then rounds to bf16 once (matching the CUDA kernel's
/// accumulation semantics — scope decision 4 of the fused-kernels plan);
/// candle's own `affine`+`add` composition multiplies/adds directly in
/// bf16, with an extra intermediate rounding. The two are not expected to
/// bit-match in general — this is a real, documented divergence in
/// rounding path, not the op's tolerance being loose out of laziness.
const BF16_ULP_TOL: i32 = 2;

fn fused_vs_eager_bf16_bit_diffs(alpha: f64, xv: &[f32], yv: &[f32]) -> Vec<i32> {
    let device = Device::Cpu;
    let xb: Vec<bf16> = xv.iter().map(|&v| bf16::from_f32(v)).collect();
    let yb: Vec<bf16> = yv.iter().map(|&v| bf16::from_f32(v)).collect();
    let n = xb.len();

    let x = Tensor::from_slice(&xb, (n,), &device).unwrap();
    let y = Tensor::from_slice(&yb, (n,), &device).unwrap();

    let fused: Vec<bf16> = axpy_fwd(alpha, &x, &y).unwrap().to_vec1().unwrap();
    let eager: Vec<bf16> = eager_fwd(alpha, &x, &y)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_vec1()
        .unwrap();

    fused
        .iter()
        .zip(eager.iter())
        .map(|(&f, &e)| bf16_bit_diff(f, e))
        .collect()
}

#[test]
fn fused_vs_eager_bf16_fwd_is_within_the_stated_tolerance_on_an_exact_fixture() {
    // This particular fixture happens to be bf16-exact at every
    // intermediate step (every product/sum is exactly representable), so
    // the measured gap here is 0 — it does NOT exercise the 2-ULP
    // tolerance (see the divergent fixture below for that). Kept as a
    // "the two paths agree when there is nothing to disagree about" case.
    let diffs =
        fused_vs_eager_bf16_bit_diffs(1.25, &[1.0f32, -2.0, 3.5, -0.25], &[0.5f32, 1.5, -3.0, 4.0]);
    for (i, d) in diffs.iter().enumerate() {
        assert!(
            d.abs() <= BF16_ULP_TOL,
            "element {i}: bit diff {d} exceeds the stated {BF16_ULP_TOL}-ULP tolerance"
        );
    }
}

#[test]
fn fused_vs_eager_bf16_tolerance_is_measured_not_vacuous() {
    // A fixture chosen (by brute-force search over a small grid) so the
    // f32-accumulate-round-once path and the round-at-each-step eager path
    // GENUINELY diverge: alpha = 0.1 is not exactly representable in bf16,
    // and these particular products need more than bf16's 8 significant
    // bits to round identically both ways. This is what makes the 2-ULP
    // bound in the test above a MEASURED tolerance rather than an asserted
    // one that happens never to be exercised.
    let diffs = fused_vs_eager_bf16_bit_diffs(
        0.1,
        &[-18.5f32, -18.5, -18.5, -17.75],
        &[-2.015625f32, 1.703125, 2.234375, -2.015625],
    );
    assert!(
        diffs.iter().any(|&d| d != 0),
        "expected fixture to diverge (measured diffs: {diffs:?}) — the tolerance \
         is not being exercised"
    );
    for (i, d) in diffs.iter().enumerate() {
        assert!(
            d.abs() <= BF16_ULP_TOL,
            "element {i}: bit diff {d} exceeds the stated {BF16_ULP_TOL}-ULP tolerance"
        );
    }
}

// ---------------------------------------------------------------------
// Oracle 3: bwd always returns Some — is_variable() cannot safely gate it
// ---------------------------------------------------------------------

#[test]
fn bwd_chains_through_an_intermediate_non_variable_node() {
    // `x` is NOT itself a `Var` — it is an INTERMEDIATE (`w.affine(2, 0)`)
    // on a path to a `Var` (`w`). `Tensor::is_variable()` cannot tell this
    // apart from a true external constant; gating `bwd` on it (this crate's
    // previous design) returned `None` for `x`'s slot and panicked here —
    // "candle internal error - grad not populated" (backprop.rs:174) —
    // because candle's own backward walk still requires a gradient for
    // `x` once its turn comes up, having already marked it `track_grad =
    // true` (`sorted_nodes`, backprop.rs:47-158) since it leads to `w`.
    let device = Device::Cpu;
    let w =
        Var::from_tensor(&Tensor::from_slice(&[1.0f32, 2.0, 3.0], (3,), &device).unwrap()).unwrap();
    let x = w.affine(2.0, 0.0).unwrap();
    assert!(
        !x.is_variable(),
        "x must be the is_variable()==false case under test"
    );
    let y = Var::from_tensor(&Tensor::from_slice(&[10.0f32, 20.0, 30.0], (3,), &device).unwrap())
        .unwrap();

    // out = 3 * (2*w) + y = 6*w + y
    let out = axpy_fwd(3.0, &x, &y).unwrap();
    let grads = out.backward().unwrap(); // must not panic

    let dw: Vec<f32> = grads.get(&w).unwrap().to_vec1().unwrap();
    assert_eq!(
        dw,
        vec![6.0, 6.0, 6.0],
        "d(sum(6*w + y))/dw == 6, by the chain rule"
    );
    let dy: Vec<f32> = grads.get(&y).unwrap().to_vec1().unwrap();
    assert_eq!(dy, vec![1.0, 1.0, 1.0]);

    // And it matches the eager composition's gradient exactly (same ops,
    // same intermediate, same fold order).
    let w2 =
        Var::from_tensor(&Tensor::from_slice(&[1.0f32, 2.0, 3.0], (3,), &device).unwrap()).unwrap();
    let x2 = w2.affine(2.0, 0.0).unwrap();
    let y2 = Var::from_tensor(&Tensor::from_slice(&[10.0f32, 20.0, 30.0], (3,), &device).unwrap())
        .unwrap();
    let grads2 = eager_fwd(3.0, &x2, &y2).unwrap().backward().unwrap();
    assert_eq!(dw, grads2.get(&w2).unwrap().to_vec1::<f32>().unwrap());
}

#[test]
fn bwd_populates_a_grad_for_a_true_frozen_leaf_input_too_and_it_is_harmless() {
    // Since `is_variable()` cannot safely distinguish a true external leaf
    // from an intermediate (the case above), `Axpy::bwd` no longer
    // special-cases either argument: both grad slots are always `Some`,
    // even when an input is a genuinely frozen constant with no upstream
    // op. The extra `GradStore` entry this creates is simply never
    // required or consumed — candle's own backward walk never pushes a
    // `track_grad == false` node onto `sorted_nodes` — so it is observable
    // (present, and correct if queried) but not a correctness hazard.
    let device = Device::Cpu;
    let frozen = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (3,), &device).unwrap();
    assert!(!frozen.is_variable());
    let y = Var::from_tensor(&Tensor::from_slice(&[10.0f32, 20.0, 30.0], (3,), &device).unwrap())
        .unwrap();

    let out = axpy_fwd(2.0, &frozen, &y).unwrap();
    let grads = out.backward().unwrap();
    assert_eq!(
        grads.get(&frozen).unwrap().to_vec1::<f32>().unwrap(),
        vec![2.0, 2.0, 2.0]
    );
    assert_eq!(
        grads.get(&y).unwrap().to_vec1::<f32>().unwrap(),
        vec![1.0, 1.0, 1.0]
    );
}

#[test]
fn bwd_is_never_called_when_neither_input_leads_to_a_variable() {
    // When NEITHER input is (or leads to) a `Var`, candle's own pre-pass
    // (`Tensor::sorted_nodes`'s `track_grad`, backprop.rs:47-158) decides
    // the whole `Axpy` node itself needs no gradient and never pushes it
    // onto `sorted_nodes` — so `Axpy::bwd` is never invoked at all here,
    // one layer above anything this op's `bwd` body controls. `.backward()`
    // must still succeed (a degenerate-but-valid, all-constants graph is
    // not an error), and both slots are absent — not because `bwd`
    // returned `None` (compare to `bwd_populates_a_grad_for_a_true_frozen_
    // leaf_input_too...`, where ONE Var present makes candle call `bwd`
    // and it returns `Some` for both, including the still-frozen one).
    let device = Device::Cpu;
    let x = Tensor::from_slice(&[1.0f32], (1,), &device).unwrap();
    let y = Tensor::from_slice(&[2.0f32], (1,), &device).unwrap();
    let out = axpy_fwd(1.0, &x, &y).unwrap();
    let grads = out.backward().unwrap();
    assert!(grads.get(&x).is_none());
    assert!(grads.get(&y).is_none());
}
