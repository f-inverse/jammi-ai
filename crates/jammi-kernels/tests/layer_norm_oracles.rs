//! CPU-hermetic oracles for `LayerNormFused` — the same rigor-chain
//! pattern `tests/oracles.rs` establishes for `Axpy`, extended for a
//! `CustomOp2` whose backward is NOT ordinary `Tensor` composition (it
//! dispatches into two more `KernelOp`s — see
//! `jammi_kernels::ops::layer_norm`'s module doc).
//!
//!   1. `gradcheck_*` — `bwd` vs. central finite differences (`dx` AND
//!      `dgamma`).
//!   2. `eager_vs_fused_*` — fwd+bwd vs. a hand-written composition of
//!      ordinary candle ops that computes the SAME bias-free LayerNorm
//!      (mean/center/variance/normalize/cast-back/scale — the same shape
//!      `jammi-encoders::layer_norm::slow()`'s bias-free arm has), stated
//!      tolerance derived from the f32-accumulation rounding model. This
//!      crate is a LEAF (no `jammi-*` deps — see its module doc / the
//!      fused-kernels plan's scope decision 12), so it cannot import
//!      `slow()` itself; the actual "against the real `slow()`" oracle
//!      (calling that exact function) lives in `jammi-encoders`'
//!      `tests/it` suite, where `slow()` is reachable. The composition
//!      reproduced here is the same math for the SAME reason axpy's
//!      `eager_fwd` reproduces `affine`+`add` rather than importing
//!      anything — a leaf-crate-clean "what the eager path computes"
//!      fixture.
//!   3. `dgamma_needed_true_through_an_intermediate_*` /
//!      `dgamma_needed_false_on_a_frozen_leaf_*` — the construction-data
//!      regression oracle: `dgamma_needed=true` through an INTERMEDIATE
//!      `x` (an `w.affine` chain, not a raw `Var`) still produces a
//!      correct `dx` (and `dgamma`) matching the eager composition;
//!      `dgamma_needed=false` on a genuinely frozen (non-`Var`) `gamma`
//!      neither panics nor ever populates a gradient for it.
//!
//! The CUDA↔CPU parity leg (fwd + both bwd outputs, contiguous/narrowed/
//! empty/multi-row, hidden 1024 and non-1024, bf16+f32) lives in
//! `tests/cuda_parity.rs`, gated the same way `Axpy`'s is.

use candle_core::{DType, Device, Tensor, Var, D};
use half::bf16;
use jammi_kernels::ops::{apply2, LayerNormFused};

fn fused(eps: f64, dgamma_needed: bool, x: &Tensor, gamma: &Tensor) -> candle_core::Result<Tensor> {
    apply2(x, gamma, LayerNormFused::new(eps, dgamma_needed))
}

/// The bias-free eager composition `jammi-encoders::layer_norm::slow()`
/// runs when `bias.is_none()`: f32-internal mean/center/variance/
/// normalize, cast BACK to `x`'s own dtype, THEN multiply by `gamma` (in
/// `x`'s dtype, not f32) — reproduced here candle-op-for-candle-op so the
/// comparison is against the actual composition, not a re-derived
/// closed form.
fn eager(eps: f64, x: &Tensor, gamma: &Tensor) -> candle_core::Result<Tensor> {
    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let hidden = x.dim(D::Minus1)?;
    let x_internal = x.to_dtype(internal_dtype)?;
    let mean = (x_internal.sum_keepdim(D::Minus1)? / hidden as f64)?;
    let centered = x_internal.broadcast_sub(&mean)?;
    let variance = (centered.sqr()?.sum_keepdim(D::Minus1)? / hidden as f64)?;
    let normalized = centered.broadcast_div(&(variance + eps)?.sqrt()?)?;
    normalized.to_dtype(x_dtype)?.broadcast_mul(gamma)
}

// ---------------------------------------------------------------------
// Oracle 1: gradcheck vs. central finite differences (dx AND dgamma)
// ---------------------------------------------------------------------

#[test]
fn gradcheck_dx_f32() {
    let device = Device::Cpu;
    let x0: [f32; 8] = [-2.0, -0.75, -0.1, 0.3, 1.2, 4.0, 0.6, -1.3];
    let gamma0: [f32; 4] = [1.5, 0.5, -0.75, 2.0];
    let eps = 1e-5;
    let hidden = gamma0.len();
    let rows = x0.len() / hidden;

    let x = Var::from_tensor(&Tensor::from_slice(&x0, (rows, hidden), &device).unwrap()).unwrap();
    let gamma = Tensor::from_slice(&gamma0, (hidden,), &device).unwrap();

    let out = fused(eps, false, &x, &gamma).unwrap();
    let grads = out.backward().unwrap();
    let dx: Vec<f32> = grads
        .get(&x)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let sum_fwd = |x: &Tensor| -> f64 {
        fused(eps, false, x, &gamma)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap() as f64
    };

    let fd_eps = 2e-3f32;
    let tol = 5e-2f64;
    for i in 0..x0.len() {
        let mut xp = x0;
        xp[i] += fd_eps;
        let mut xm = x0;
        xm[i] -= fd_eps;
        let xp_t = Tensor::from_slice(&xp, (rows, hidden), &device).unwrap();
        let xm_t = Tensor::from_slice(&xm, (rows, hidden), &device).unwrap();
        let numeric = (sum_fwd(&xp_t) - sum_fwd(&xm_t)) / (2.0 * fd_eps as f64);
        assert!(
            (numeric - dx[i] as f64).abs() < tol,
            "dx[{i}]: numeric {numeric} vs analytic {}",
            dx[i]
        );
    }
}

#[test]
fn gradcheck_dgamma_f32() {
    let device = Device::Cpu;
    let x0: [f32; 8] = [-2.0, -0.75, -0.1, 0.3, 1.2, 4.0, 0.6, -1.3];
    let gamma0: [f32; 4] = [1.5, 0.5, -0.75, 2.0];
    let eps = 1e-5;
    let hidden = gamma0.len();
    let rows = x0.len() / hidden;

    let x = Tensor::from_slice(&x0, (rows, hidden), &device).unwrap();
    let gamma =
        Var::from_tensor(&Tensor::from_slice(&gamma0, (hidden,), &device).unwrap()).unwrap();

    let out = fused(eps, true, &x, &gamma).unwrap();
    let grads = out.backward().unwrap();
    let dgamma: Vec<f32> = grads.get(&gamma).unwrap().to_vec1().unwrap();

    let sum_fwd = |gamma: &Tensor| -> f64 {
        fused(eps, true, &x, gamma)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap() as f64
    };

    let fd_eps = 2e-3f32;
    let tol = 5e-2f64;
    for i in 0..gamma0.len() {
        let mut gp = gamma0;
        gp[i] += fd_eps;
        let mut gm = gamma0;
        gm[i] -= fd_eps;
        let gp_t = Tensor::from_slice(&gp, (hidden,), &device).unwrap();
        let gm_t = Tensor::from_slice(&gm, (hidden,), &device).unwrap();
        let numeric = (sum_fwd(&gp_t) - sum_fwd(&gm_t)) / (2.0 * fd_eps as f64);
        assert!(
            (numeric - dgamma[i] as f64).abs() < tol,
            "dgamma[{i}]: numeric {numeric} vs analytic {}",
            dgamma[i]
        );
    }
}

// ---------------------------------------------------------------------
// Oracle 2: fused vs. eager (candle composition), fwd AND bwd
// ---------------------------------------------------------------------

#[test]
fn eager_vs_fused_f32_fwd_and_bwd_match_within_stated_tolerance() {
    let device = Device::Cpu;
    let x0: [f32; 8] = [1.0, -2.0, 3.5, -0.25, 0.1, 2.2, -1.1, 0.75];
    let gamma0: [f32; 4] = [1.25, -0.5, 2.0, 0.75];
    let eps = 1e-5;
    let hidden = gamma0.len();
    let rows = x0.len() / hidden;

    let x_f = Var::from_tensor(&Tensor::from_slice(&x0, (rows, hidden), &device).unwrap()).unwrap();
    let g_f = Var::from_tensor(&Tensor::from_slice(&gamma0, (hidden,), &device).unwrap()).unwrap();
    let x_e = Var::from_tensor(&Tensor::from_slice(&x0, (rows, hidden), &device).unwrap()).unwrap();
    let g_e = Var::from_tensor(&Tensor::from_slice(&gamma0, (hidden,), &device).unwrap()).unwrap();

    let out_f = fused(eps, true, &x_f, &g_f).unwrap();
    let out_e = eager(eps, &x_e, &g_e).unwrap();
    let vf: Vec<f32> = out_f.flatten_all().unwrap().to_vec1().unwrap();
    let ve: Vec<f32> = out_e.flatten_all().unwrap().to_vec1().unwrap();
    // f32-throughout (no bf16 rounding involved on this fixture): the two
    // compositions should match tightly, though not necessarily to 0 ULP
    // (different op sequencing can still round differently at the last
    // bit) — a small stated absolute tolerance.
    for (i, (f, e)) in vf.iter().zip(ve.iter()).enumerate() {
        assert!((f - e).abs() < 1e-4, "fwd[{i}]: fused {f} vs eager {e}");
    }

    let grads_f = out_f.backward().unwrap();
    let grads_e = out_e.backward().unwrap();
    let dxf: Vec<f32> = grads_f
        .get(&x_f)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let dxe: Vec<f32> = grads_e
        .get(&x_e)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    for (i, (f, e)) in dxf.iter().zip(dxe.iter()).enumerate() {
        assert!((f - e).abs() < 1e-3, "dx[{i}]: fused {f} vs eager {e}");
    }
    let dgf: Vec<f32> = grads_f.get(&g_f).unwrap().to_vec1().unwrap();
    let dge: Vec<f32> = grads_e.get(&g_e).unwrap().to_vec1().unwrap();
    for (i, (f, e)) in dgf.iter().zip(dge.iter()).enumerate() {
        assert!((f - e).abs() < 1e-3, "dgamma[{i}]: fused {f} vs eager {e}");
    }
}

/// bf16: the eager composition rounds `xhat` to bf16 BEFORE multiplying
/// by `gamma` (`normalized.to_dtype(x_dtype)?.broadcast_mul(&weight)` in
/// `slow()`); the fused kernel multiplies `xhat * gamma` in f32 and
/// rounds ONCE at the very end. These are genuinely different rounding
/// paths — a measured, non-vacuous divergence, not a laziness-driven
/// tolerance. `BF16_ULP_TOL` mirrors `tests/oracles.rs`'s axpy constant.
const BF16_ULP_TOL: i32 = 4;

fn bf16_bit_diff(a: bf16, b: bf16) -> i32 {
    a.to_bits() as i32 - b.to_bits() as i32
}

#[test]
fn eager_vs_fused_bf16_fwd_diverges_and_stays_within_the_stated_ulp_tolerance() {
    let device = Device::Cpu;
    // Chosen so the two bf16 rounding paths (round-xhat-then-multiply vs.
    // multiply-in-f32-then-round-once) genuinely disagree, not just a
    // fixture that happens to land exactly either way.
    let x0: [f32; 8] = [-18.5, -18.5, -18.5, -17.75, 3.375, -4.125, 9.0625, -2.5];
    let gamma0: [f32; 4] = [0.1, 1.703125, -2.015625, 2.234375];
    let eps = 1e-5;
    let hidden = gamma0.len();
    let rows = x0.len() / hidden;

    let xb: Vec<bf16> = x0.iter().map(|&v| bf16::from_f32(v)).collect();
    let gb: Vec<bf16> = gamma0.iter().map(|&v| bf16::from_f32(v)).collect();
    let x = Tensor::from_slice(&xb, (rows, hidden), &device).unwrap();
    let gamma = Tensor::from_slice(&gb, (hidden,), &device).unwrap();

    let fused_out: Vec<bf16> = fused(eps, false, &x, &gamma)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let eager_out: Vec<bf16> = eager(eps, &x, &gamma)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let diffs: Vec<i32> = fused_out
        .iter()
        .zip(eager_out.iter())
        .map(|(&f, &e)| bf16_bit_diff(f, e))
        .collect();
    assert!(
        diffs.iter().any(|&d| d != 0),
        "expected fixture to diverge (measured diffs: {diffs:?}) — the \
         tolerance is not being exercised"
    );
    for (i, d) in diffs.iter().enumerate() {
        assert!(
            d.abs() <= BF16_ULP_TOL,
            "element {i}: bit diff {d} exceeds the stated {BF16_ULP_TOL}-ULP tolerance"
        );
    }
}

/// The backward analog of the forward divergence oracle above: the SAME
/// rounding-path mechanism (eager rounds `xhat` to bf16 before
/// multiplying by `gamma`; fused multiplies in f32 and rounds once) also
/// makes `dx`/`dgamma` bf16-dtype-specific — an f32-tensor `dx`/`dgamma`
/// gradcheck (the earlier f32 oracles in this file) cannot bound this,
/// because f32 tensors never exercise the bf16 rounding step at all. This
/// is a measured, non-vacuous divergence on BOTH `dx` and `dgamma`
/// (each asserted `any(!= 0)`, the same non-vacuous-control shape as the
/// forward oracle), with both kept within the stated `BF16_ULP_TOL`.
///
/// On `BF16_ULP_TOL` (shared with the forward oracle above): backward
/// does strictly more bf16-sensitive arithmetic per element than forward
/// (two per-row reductions — `mean_row(t)` and `mean_row(t*xhat)` — plus
/// the final `(t - mean_t - xhat*mean_t_xhat) * invvar` multiply-add,
/// versus forward's single `xhat * gamma`), so a 2x-forward's-budget
/// allowance is the a priori argument for a wider bound here. The
/// MEASURED max on this fixture (printed below) is only 1 ULP for both
/// `dx` and `dgamma` — the same as the forward oracle's own measured max
/// on its fixture — so `BF16_ULP_TOL = 4` is a deliberately conservative
/// ceiling on top of that, not a tight derivation from either argument;
/// it is sized to tolerate a worse fixture or a different libm's
/// rounding on another platform without flaking, and the measured value
/// is what actually establishes how tight it is today.
#[test]
fn eager_vs_fused_bf16_bwd_diverges_and_stays_within_the_stated_ulp_tolerance() {
    let device = Device::Cpu;
    let x0: [f32; 8] = [-18.5, -18.5, -18.5, -17.75, 3.375, -4.125, 9.0625, -2.5];
    let gamma0: [f32; 4] = [0.1, 1.703125, -2.015625, 2.234375];
    // A NON-uniform loss weight: `Tensor::backward()` seeds the output's
    // gradient with `ones_like` (candle-core's `backprop.rs:168`), and
    // `1.0 * gamma` rounds identically in bf16 and f32 for ANY `gamma`
    // (multiplying by exactly 1 loses no precision either way) — that
    // would make `dy * gamma` trivially agree regardless of dtype and
    // mask the very rounding-order mechanism this oracle exists to
    // measure. Weighting the output before summing makes the effective
    // `dy` at the LayerNorm's own output this non-trivial, bf16-awkward
    // vector instead of a uniform 1.
    let w0: [f32; 8] = [
        1.703125, -2.015625, 2.234375, 0.1, -18.5, -17.75, 3.375, 9.0625,
    ];
    let eps = 1e-5;
    let hidden = gamma0.len();
    let rows = x0.len() / hidden;

    let xb: Vec<bf16> = x0.iter().map(|&v| bf16::from_f32(v)).collect();
    let gb: Vec<bf16> = gamma0.iter().map(|&v| bf16::from_f32(v)).collect();
    let wb: Vec<bf16> = w0.iter().map(|&v| bf16::from_f32(v)).collect();

    let x_f = Var::from_tensor(&Tensor::from_slice(&xb, (rows, hidden), &device).unwrap()).unwrap();
    let g_f = Var::from_tensor(&Tensor::from_slice(&gb, (hidden,), &device).unwrap()).unwrap();
    let x_e = Var::from_tensor(&Tensor::from_slice(&xb, (rows, hidden), &device).unwrap()).unwrap();
    let g_e = Var::from_tensor(&Tensor::from_slice(&gb, (hidden,), &device).unwrap()).unwrap();
    let w_f = Tensor::from_slice(&wb, (rows, hidden), &device).unwrap();
    let w_e = Tensor::from_slice(&wb, (rows, hidden), &device).unwrap();

    let out_f = fused(eps, true, &x_f, &g_f).unwrap();
    let out_e = eager(eps, &x_e, &g_e).unwrap();
    let loss_f = (&out_f * &w_f).unwrap().sum_all().unwrap();
    let loss_e = (&out_e * &w_e).unwrap().sum_all().unwrap();
    let grads_f = loss_f.backward().unwrap();
    let grads_e = loss_e.backward().unwrap();

    let dxf: Vec<bf16> = grads_f
        .get(&x_f)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let dxe: Vec<bf16> = grads_e
        .get(&x_e)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let dx_diffs: Vec<i32> = dxf
        .iter()
        .zip(dxe.iter())
        .map(|(&f, &e)| bf16_bit_diff(f, e))
        .collect();
    assert!(
        dx_diffs.iter().any(|&d| d != 0),
        "expected dx fixture to diverge (measured diffs: {dx_diffs:?}) — the \
         tolerance is not being exercised"
    );
    let dx_max = dx_diffs.iter().map(|d| d.abs()).max().unwrap_or(0);
    println!("eager_vs_fused_bf16_bwd: measured max |dx bit-diff| = {dx_max}");
    for (i, d) in dx_diffs.iter().enumerate() {
        assert!(
            d.abs() <= BF16_ULP_TOL,
            "dx element {i}: bit diff {d} exceeds the stated {BF16_ULP_TOL}-ULP tolerance"
        );
    }

    let dgf: Vec<bf16> = grads_f.get(&g_f).unwrap().to_vec1().unwrap();
    let dge: Vec<bf16> = grads_e.get(&g_e).unwrap().to_vec1().unwrap();
    let dg_diffs: Vec<i32> = dgf
        .iter()
        .zip(dge.iter())
        .map(|(&f, &e)| bf16_bit_diff(f, e))
        .collect();
    // Same non-vacuity control as dx above (and the forward oracle): a
    // fixture that never actually exercised the tolerance would make the
    // bound below an unfalsifiable no-op.
    assert!(
        dg_diffs.iter().any(|&d| d != 0),
        "expected dgamma fixture to diverge (measured diffs: {dg_diffs:?}) — \
         the tolerance is not being exercised"
    );
    let dg_max = dg_diffs.iter().map(|d| d.abs()).max().unwrap_or(0);
    println!("eager_vs_fused_bf16_bwd: measured max |dgamma bit-diff| = {dg_max}");
    for (i, d) in dg_diffs.iter().enumerate() {
        assert!(
            d.abs() <= BF16_ULP_TOL,
            "dgamma element {i}: bit diff {d} exceeds the stated {BF16_ULP_TOL}-ULP tolerance"
        );
    }
}

// ---------------------------------------------------------------------
// Oracle 3: the construction-data dgamma mechanism
// ---------------------------------------------------------------------

#[test]
fn dgamma_needed_true_through_an_intermediate_x_matches_eager() {
    // `x` is an INTERMEDIATE (`w.affine(2, 0)`) on a path to a `Var`
    // (`w`) — `is_variable() == false`, exactly the case `Axpy`'s own
    // regression test exercises (see `tests/oracles.rs`). `dx`'s slot
    // must still be populated (LayerNormFused::bwd never gates it on
    // `is_variable()`), and with `dgamma_needed=true`, `gamma`'s slot
    // (a genuine `Var` here) must also match the eager gradient.
    let device = Device::Cpu;
    let w0: [f32; 8] = [0.5, -1.0, 2.0, 0.25, -0.5, 1.5, -2.0, 0.75];
    let gamma0: [f32; 4] = [1.0, 2.0, -1.0, 0.5];
    let eps = 1e-5;
    let hidden = gamma0.len();
    let rows = w0.len() / hidden;

    let w = Var::from_tensor(&Tensor::from_slice(&w0, (rows, hidden), &device).unwrap()).unwrap();
    let x = w.affine(2.0, 0.0).unwrap();
    assert!(
        !x.is_variable(),
        "x must be the is_variable()==false case under test"
    );
    let gamma =
        Var::from_tensor(&Tensor::from_slice(&gamma0, (hidden,), &device).unwrap()).unwrap();

    let out = fused(eps, true, &x, &gamma).unwrap();
    let grads = out.backward().unwrap(); // must not panic
    let dw: Vec<f32> = grads
        .get(&w)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let dgamma: Vec<f32> = grads.get(&gamma).unwrap().to_vec1().unwrap();

    // Same graph shape, through the eager composition.
    let w2 = Var::from_tensor(&Tensor::from_slice(&w0, (rows, hidden), &device).unwrap()).unwrap();
    let x2 = w2.affine(2.0, 0.0).unwrap();
    let gamma2 =
        Var::from_tensor(&Tensor::from_slice(&gamma0, (hidden,), &device).unwrap()).unwrap();
    let grads2 = eager(eps, &x2, &gamma2).unwrap().backward().unwrap();
    let dw2: Vec<f32> = grads2
        .get(&w2)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let dgamma2: Vec<f32> = grads2.get(&gamma2).unwrap().to_vec1().unwrap();

    for (i, (a, b)) in dw.iter().zip(dw2.iter()).enumerate() {
        assert!((a - b).abs() < 1e-3, "dw[{i}]: fused {a} vs eager {b}");
    }
    for (i, (a, b)) in dgamma.iter().zip(dgamma2.iter()).enumerate() {
        assert!((a - b).abs() < 1e-3, "dgamma[{i}]: fused {a} vs eager {b}");
    }
}

#[test]
fn dgamma_needed_false_on_a_frozen_leaf_gamma_neither_panics_nor_emits_a_gamma_grad() {
    // `gamma` here is a genuinely frozen leaf: a plain `Tensor`, never
    // wrapped in `Var`, with no upstream op — exactly the shape
    // `jammi-encoders`' `LayerNorm::weight` has today (loaded straight
    // from a frozen backbone `VarBuilder`; only LoRA A/B are trainable
    // `Var`s). `x` IS a `Var` so the whole node still needs a gradient
    // (bwd genuinely gets called), but `dgamma_needed=false` must mean
    // `gamma`'s slot comes back `None` — no crash, no populated grad.
    let device = Device::Cpu;
    let x0: [f32; 8] = [0.5, -1.0, 2.0, 0.25, -0.5, 1.5, -2.0, 0.75];
    let gamma0: [f32; 4] = [1.0, 2.0, -1.0, 0.5];
    let eps = 1e-5;
    let hidden = gamma0.len();
    let rows = x0.len() / hidden;

    let x = Var::from_tensor(&Tensor::from_slice(&x0, (rows, hidden), &device).unwrap()).unwrap();
    let gamma = Tensor::from_slice(&gamma0, (hidden,), &device).unwrap();
    assert!(!gamma.is_variable());

    let out = fused(eps, false, &x, &gamma).unwrap();
    let grads = out.backward().unwrap(); // must not panic
    assert!(grads.get(&x).is_some(), "dx must still be populated");
    assert!(
        grads.get(&gamma).is_none(),
        "dgamma_needed=false must mean gamma's gradient is never populated"
    );
}
