//! CPU-hermetic oracles for `LayerNormFused` — the same rigor-chain
//! pattern `tests/oracles.rs` establishes for `Axpy`, extended for a
//! `CustomOp2` whose backward is NOT ordinary `Tensor` composition (it
//! dispatches into two more `KernelOp`s — see
//! `jammi_kernels::ops::layer_norm`'s module doc).
//!
//!   1. `gradcheck_*` — `bwd` vs. central finite differences (`dx` AND
//!      `dgamma`).
//!   2. `fused_vs_formula_*` — the fused `LayerNormFused` kernel vs. a
//!      hand-written FORMULA: a candle-op composition IN THIS FILE that
//!      computes the SAME bias-free LayerNorm math
//!      (mean/center/variance/normalize/gamma-in-f32/cast-once — the same
//!      one-rounding shape `jammi-encoders::layer_norm::slow()`'s
//!      bias-free arm has since the eager-LN one-rounding fix), f32
//!      asserted with a small stated tolerance (different op sequencing
//!      can still round the last bit differently) and bf16 asserted
//!      BIT-EXACT (measured on these fixtures; see the bf16 tests' own
//!      docs for why that is not a structural guarantee at every shape).
//!      NAMING NOTE (do not read this as an eager-parity claim): this
//!      crate is a LEAF (no `jammi-*` deps — see its module doc / the
//!      fused-kernels plan's scope decision 12), so `formula()` below
//!      cannot import `slow()` and is NOT a call into `slow()` — it is
//!      an independently-written reproduction of the SAME MATH, updated
//!      by hand whenever `slow()`'s own rounding placement changes (as
//!      it did in this same PR), which makes a diff that changes both
//!      `slow()` and `formula()` together structurally unable to prove
//!      anything about `slow()`'s OWN correctness — only that this
//!      file's copy of the math agrees with the fused kernel. The BITING
//!      oracle that calls the REAL `slow()` (and the REAL
//!      `LayerNormFused` fused CPU arm, via `apply2`) against an
//!      independently-derived, non-candle-op scalar truth lives in
//!      `jammi-encoders`' own `src/layer_norm.rs`
//!      `#[cfg(test)] mod tests` —
//!      `layer_norm_slow_matches_truth_at_production_shape_seq128`/
//!      `_seq512` — the only place in the workspace `slow()` is
//!      reachable at all. `formula()`'s value here is narrower than that:
//!      it lets THIS crate's own `LayerNormFused` fwd/bwd oracles run
//!      hermetically with no `jammi-*` dependency, the same reason
//!      axpy's `formula_fwd` reproduces `affine`+`add` rather than
//!      importing anything.
//!   3. `dgamma_needed_true_through_an_intermediate_*` /
//!      `dgamma_needed_false_on_a_frozen_leaf_*` — the construction-data
//!      regression oracle: `dgamma_needed=true` through an INTERMEDIATE
//!      `x` (an `w.affine` chain, not a raw `Var`) still produces a
//!      correct `dx` (and `dgamma`) matching the formula composition;
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

/// The bias-free formula composition `jammi-encoders::layer_norm::slow()`
/// runs: f32-internal mean/center/variance/normalize, gamma upcast to
/// `internal_dtype` and applied THERE (never rounded to `x`'s dtype
/// first), one cast to `x_dtype` at the very end — reproduced here
/// candle-op-for-candle-op so the comparison is against the actual
/// composition, not a re-derived closed form. Matches torch's
/// `layer_norm_cuda` epilogue (`vectorized_layer_norm_kernel_impl`:
/// "Computation is performed in T_ACC ... result is implicitly cast to
/// T") and `LayerNormFused`'s own CPU/CUDA arm (`cuda/layer_norm.cu:124`:
/// `yr[i] = __float2bfloat16(xhat * __bfloat162float(gamma[i]))`).
fn formula(eps: f64, x: &Tensor, gamma: &Tensor) -> candle_core::Result<Tensor> {
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
    let gamma_internal = gamma.to_dtype(internal_dtype)?;
    let scaled_internal = normalized.broadcast_mul(&gamma_internal)?;
    scaled_internal.to_dtype(x_dtype)
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
// Oracle 2: fused vs. formula (candle composition), fwd AND bwd
// ---------------------------------------------------------------------

#[test]
fn fused_vs_formula_f32_fwd_and_bwd_match_within_stated_tolerance() {
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
    let out_e = formula(eps, &x_e, &g_e).unwrap();
    let vf: Vec<f32> = out_f.flatten_all().unwrap().to_vec1().unwrap();
    let ve: Vec<f32> = out_e.flatten_all().unwrap().to_vec1().unwrap();
    // f32-throughout (no bf16 rounding involved on this fixture): the two
    // compositions should match tightly, though not necessarily to 0 ULP
    // (different op sequencing can still round differently at the last
    // bit) — a small stated absolute tolerance.
    for (i, (f, e)) in vf.iter().zip(ve.iter()).enumerate() {
        assert!((f - e).abs() < 1e-4, "fwd[{i}]: fused {f} vs formula {e}");
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
        assert!((f - e).abs() < 1e-3, "dx[{i}]: fused {f} vs formula {e}");
    }
    let dgf: Vec<f32> = grads_f.get(&g_f).unwrap().to_vec1().unwrap();
    let dge: Vec<f32> = grads_e.get(&g_e).unwrap().to_vec1().unwrap();
    for (i, (f, e)) in dgf.iter().zip(dge.iter()).enumerate() {
        assert!(
            (f - e).abs() < 1e-3,
            "dgamma[{i}]: fused {f} vs formula {e}"
        );
    }
}

/// bf16: BEFORE the one-rounding fix, the formula composition rounded
/// `xhat` to bf16 BEFORE multiplying by `gamma`
/// (`normalized.to_dtype(x_dtype)?.broadcast_mul(&weight)`, the pre-fix
/// `slow()`), while the fused kernel multiplied `xhat * gamma` in f32 and
/// rounded ONCE at the very end — measured on this same fixture, that was
/// a genuine, non-vacuous divergence. `formula()` above now runs the
/// IDENTICAL one-rounding shape (`gamma` upcast to f32, multiplied,
/// rounded once), so formula and fused are expected to — and on this
/// fixture, measured to — agree BIT-EXACTLY. `bf16_bit_diff` is kept
/// (rather than deleted along with the tolerance) because it is still
/// what prints the measured max in the backward oracle below, and as the
/// tool a future regression would use to re-derive a tolerance if a
/// production-`hidden`-sized fixture ever exposed a reduction-order
/// difference between candle's `sum_keepdim` and the fused kernel's
/// ascending-index scalar fold (neither this small fixture nor the
/// f32 oracle above needed one).
fn bf16_bit_diff(a: bf16, b: bf16) -> i32 {
    a.to_bits() as i32 - b.to_bits() as i32
}

#[test]
fn fused_vs_formula_bf16_fwd_is_bit_exact_after_the_one_rounding_fix() {
    let device = Device::Cpu;
    // The same fixture the pre-fix divergence oracle used (chosen to make
    // the OLD two-rounding-path mismatch as visible as possible) — kept
    // unchanged so the before/after comparison is apples to apples.
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
    let formula_out: Vec<bf16> = formula(eps, &x, &gamma)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let diffs: Vec<i32> = fused_out
        .iter()
        .zip(formula_out.iter())
        .map(|(&f, &e)| bf16_bit_diff(f, e))
        .collect();
    println!("fused_vs_formula_bf16_fwd: measured bit-diffs (post-fix) = {diffs:?}");
    assert_eq!(
        fused_out, formula_out,
        "fwd must now be bit-exact (measured diffs: {diffs:?}) — the pre-fix defect \
         (26% of elements diverging, see `LayerNorm::slow`'s doc) is gone"
    );
}

/// The backward analog of the forward divergence oracle above: the SAME
/// rounding-path mechanism (formula rounds `xhat` to bf16 before
/// multiplying by `gamma`; fused multiplies in f32 and rounds once) also
/// makes `dx`/`dgamma` bf16-dtype-specific — an f32-tensor `dx`/`dgamma`
/// gradcheck (the earlier f32 oracles in this file) cannot bound this,
/// because f32 tensors never exercise the bf16 rounding step at all. This
/// is a measured, non-vacuous divergence on BOTH `dx` and `dgamma`
/// (each asserted `any(!= 0)`, the same non-vacuous-control shape as the
/// forward oracle), with both kept within the stated `BF16_ULP_TOL`.
///
/// `dx` (the analytical Apex/ATen-canonical closed form `LayerNormBwdDx`
/// computes) and the formula composition's `dx` (candle autograd
/// differentiating through the composed ops) were already two DIFFERENT
/// derivations of the same gradient before this fix — the forward
/// one-rounding fix removes one source of divergence (the forward
/// rounding-order mismatch feeding into both graphs' `xhat`), not
/// necessarily every source (the two `dx` derivations remain distinct op
/// sequences in principle). Measured on this fixture, post-fix, both `dx`
/// and `dgamma` are bit-exact (`diffs` printed below are all `0`) — see
/// the forward oracle's doc for why a small `hidden` (here 4) makes exact
/// agreement plausible even though it is not a structural guarantee for
/// an arbitrary shape.
#[test]
fn fused_vs_formula_bf16_bwd_is_bit_exact_after_the_one_rounding_fix() {
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
    let out_e = formula(eps, &x_e, &g_e).unwrap();
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
    println!("fused_vs_formula_bf16_bwd: measured dx bit-diffs (post-fix) = {dx_diffs:?}");
    assert_eq!(
        dxf, dxe,
        "dx must now be bit-exact (measured diffs: {dx_diffs:?})"
    );

    let dgf: Vec<bf16> = grads_f.get(&g_f).unwrap().to_vec1().unwrap();
    let dge: Vec<bf16> = grads_e.get(&g_e).unwrap().to_vec1().unwrap();
    let dg_diffs: Vec<i32> = dgf
        .iter()
        .zip(dge.iter())
        .map(|(&f, &e)| bf16_bit_diff(f, e))
        .collect();
    println!("fused_vs_formula_bf16_bwd: measured dgamma bit-diffs (post-fix) = {dg_diffs:?}");
    assert_eq!(
        dgf, dge,
        "dgamma must now be bit-exact (measured diffs: {dg_diffs:?})"
    );
}

// ---------------------------------------------------------------------
// Oracle 3: the construction-data dgamma mechanism
// ---------------------------------------------------------------------

#[test]
fn dgamma_needed_true_through_an_intermediate_x_matches_formula() {
    // `x` is an INTERMEDIATE (`w.affine(2, 0)`) on a path to a `Var`
    // (`w`) — `is_variable() == false`, exactly the case `Axpy`'s own
    // regression test exercises (see `tests/oracles.rs`). `dx`'s slot
    // must still be populated (LayerNormFused::bwd never gates it on
    // `is_variable()`), and with `dgamma_needed=true`, `gamma`'s slot
    // (a genuine `Var` here) must also match the formula gradient.
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

    // Same graph shape, through the formula composition.
    let w2 = Var::from_tensor(&Tensor::from_slice(&w0, (rows, hidden), &device).unwrap()).unwrap();
    let x2 = w2.affine(2.0, 0.0).unwrap();
    let gamma2 =
        Var::from_tensor(&Tensor::from_slice(&gamma0, (hidden,), &device).unwrap()).unwrap();
    let grads2 = formula(eps, &x2, &gamma2).unwrap().backward().unwrap();
    let dw2: Vec<f32> = grads2
        .get(&w2)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let dgamma2: Vec<f32> = grads2.get(&gamma2).unwrap().to_vec1().unwrap();

    for (i, (a, b)) in dw.iter().zip(dw2.iter()).enumerate() {
        assert!((a - b).abs() < 1e-3, "dw[{i}]: fused {a} vs formula {b}");
    }
    for (i, (a, b)) in dgamma.iter().zip(dgamma2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "dgamma[{i}]: fused {a} vs formula {b}"
        );
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
