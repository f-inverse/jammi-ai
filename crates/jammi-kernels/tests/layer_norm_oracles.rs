//! CPU-hermetic oracles for `LayerNormFused` — this crate's usual
//! rigor chain (gradcheck, fused-vs-independent-formula, and the
//! chain-rule-through-an-intermediate regression), for a
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
//!      every other `*_oracles.rs` file here reimplements its op's
//!      reference formula rather than importing one.
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
//! `tests/cuda_parity.rs`, gated the same way every other op's is.

use candle_core::{DType, Device, Tensor, Var, D};
use half::{bf16, f16};
use jammi_kernels::ops::{apply2, apply3, LayerNormBiasedFused, LayerNormFused};

fn fused(eps: f64, dgamma_needed: bool, x: &Tensor, gamma: &Tensor) -> candle_core::Result<Tensor> {
    apply2(x, gamma, LayerNormFused::new(eps, dgamma_needed))
}

/// #460 (C-LN): the bias-carrying sibling of [`fused`] above.
fn fused_biased(
    eps: f64,
    dgamma_needed: bool,
    dbeta_needed: bool,
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
) -> candle_core::Result<Tensor> {
    apply3(
        x,
        gamma,
        beta,
        LayerNormBiasedFused::new(eps, dgamma_needed, dbeta_needed),
    )
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
///
/// `rstd` is computed as a RECIPROCAL followed by a multiply, not a
/// division, matching torch's `rsqrt(var+eps)` placement
/// (`aten/src/ATen/native/cuda/layer_norm_kernel.cu:278`;
/// `aten/src/ATen/native/cpu/layer_norm_kernel.cpp` likewise on the CPU
/// arm) and `jammi-encoders::layer_norm::slow()`'s own `rstd` line (see
/// that function's doc for the full citation and the F32-precision
/// discriminator that proves the placement is load-bearing). Division and
/// multiply-by-reciprocal are not bit-identical in floating point (the
/// reciprocal is itself a rounded value), so a division-form `formula()`
/// would silently diverge from both `LayerNormFused`'s actual arm and the
/// real `slow()` this leaf crate cannot import — the bf16 bit-exact
/// checks below only mean anything because this reproduction takes the
/// same path.
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
    let rstd = (variance + eps)?.sqrt()?.recip()?;
    let normalized = centered.broadcast_mul(&rstd)?;
    let gamma_internal = gamma.to_dtype(internal_dtype)?;
    let scaled_internal = normalized.broadcast_mul(&gamma_internal)?;
    scaled_internal.to_dtype(x_dtype)
}

/// #460 (C-LN): [`formula`]'s bias-carrying twin — `beta` upcast to
/// `internal_dtype` and added THERE (never rounded to `x`'s dtype first),
/// matching `jammi-encoders::layer_norm::LayerNorm::slow`'s biased arm and
/// `LayerNormBiasedFused`'s own CPU/CUDA epilogue (`xhat * gamma + beta`,
/// one rounding at the very end).
fn formula_biased(
    eps: f64,
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
) -> candle_core::Result<Tensor> {
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
    let rstd = (variance + eps)?.sqrt()?.recip()?;
    let normalized = centered.broadcast_mul(&rstd)?;
    let gamma_internal = gamma.to_dtype(internal_dtype)?;
    let scaled_internal = normalized.broadcast_mul(&gamma_internal)?;
    let beta_internal = beta.to_dtype(internal_dtype)?;
    let out_internal = scaled_internal.broadcast_add(&beta_internal)?;
    out_internal.to_dtype(x_dtype)
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
        "fwd must now be bit-exact (measured diffs: {diffs:?}) — the pre-fix double-rounding \
         defect (see `LayerNorm::slow`'s doc) is gone"
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
    // (`w`) — `is_variable() == false`, exactly the hazard
    // `jammi_kernels::ops`'s module doc names. `dx`'s slot
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

// ---------------------------------------------------------------------
// #460 (C-LN): `LayerNormBiasedFused` oracles — the same three-oracle
// shape as the bias-free op above, plus `dbeta`.
// ---------------------------------------------------------------------

#[test]
fn gradcheck_dbeta_f32() {
    let device = Device::Cpu;
    let x0: [f32; 8] = [-2.0, -0.75, -0.1, 0.3, 1.2, 4.0, 0.6, -1.3];
    let gamma0: [f32; 4] = [1.5, 0.5, -0.75, 2.0];
    let beta0: [f32; 4] = [0.2, -0.3, 0.5, -0.1];
    let eps = 1e-5;
    let hidden = gamma0.len();
    let rows = x0.len() / hidden;

    let x = Tensor::from_slice(&x0, (rows, hidden), &device).unwrap();
    let gamma = Tensor::from_slice(&gamma0, (hidden,), &device).unwrap();
    let beta = Var::from_tensor(&Tensor::from_slice(&beta0, (hidden,), &device).unwrap()).unwrap();

    // A non-uniform loss weight, matching `fused_vs_formula_bf16_bwd_is_
    // bit_exact_after_the_one_rounding_fix`'s rationale: `backward()`
    // seeds an all-ones upstream gradient, which would make every
    // `dbeta_i` trivially equal `rows` regardless of a real bug in the
    // reduction's per-element wiring (a permutation of columns would
    // still sum to the same total). Weighting breaks that degeneracy.
    let w0: [f32; 8] = [0.3, -1.1, 2.2, 0.7, -0.4, 1.6, -2.3, 0.9];
    let w = Tensor::from_slice(&w0, (rows, hidden), &device).unwrap();

    let out = fused_biased(eps, false, true, &x, &gamma, beta.as_tensor()).unwrap();
    let loss = (&out * &w).unwrap().sum_all().unwrap();
    let grads = loss.backward().unwrap();
    let dbeta: Vec<f32> = grads.get(&beta).unwrap().to_vec1().unwrap();

    let sum_fwd = |beta: &Tensor| -> f64 {
        (fused_biased(eps, false, false, &x, &gamma, beta).unwrap() * &w)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap() as f64
    };

    let fd_eps = 2e-3f32;
    let tol = 5e-2f64;
    for i in 0..beta0.len() {
        let mut bp = beta0;
        bp[i] += fd_eps;
        let mut bm = beta0;
        bm[i] -= fd_eps;
        let bp_t = Tensor::from_slice(&bp, (hidden,), &device).unwrap();
        let bm_t = Tensor::from_slice(&bm, (hidden,), &device).unwrap();
        let numeric = (sum_fwd(&bp_t) - sum_fwd(&bm_t)) / (2.0 * fd_eps as f64);
        assert!(
            (numeric - dbeta[i] as f64).abs() < tol,
            "dbeta[{i}]: numeric {numeric} vs analytic {}",
            dbeta[i]
        );
    }
}

#[test]
fn fused_vs_formula_biased_f32_fwd_and_bwd_match_within_stated_tolerance() {
    let device = Device::Cpu;
    let x0: [f32; 8] = [1.0, -2.0, 3.5, -0.25, 0.1, 2.2, -1.1, 0.75];
    let gamma0: [f32; 4] = [1.25, -0.5, 2.0, 0.75];
    let beta0: [f32; 4] = [0.3, -0.2, 0.1, -0.4];
    let eps = 1e-5;
    let hidden = gamma0.len();
    let rows = x0.len() / hidden;

    let x_f = Var::from_tensor(&Tensor::from_slice(&x0, (rows, hidden), &device).unwrap()).unwrap();
    let g_f = Var::from_tensor(&Tensor::from_slice(&gamma0, (hidden,), &device).unwrap()).unwrap();
    let b_f = Var::from_tensor(&Tensor::from_slice(&beta0, (hidden,), &device).unwrap()).unwrap();
    let x_e = Var::from_tensor(&Tensor::from_slice(&x0, (rows, hidden), &device).unwrap()).unwrap();
    let g_e = Var::from_tensor(&Tensor::from_slice(&gamma0, (hidden,), &device).unwrap()).unwrap();
    let b_e = Var::from_tensor(&Tensor::from_slice(&beta0, (hidden,), &device).unwrap()).unwrap();

    let out_f = fused_biased(eps, true, true, &x_f, &g_f, b_f.as_tensor()).unwrap();
    let out_e = formula_biased(eps, &x_e, &g_e, b_e.as_tensor()).unwrap();
    let vf: Vec<f32> = out_f.flatten_all().unwrap().to_vec1().unwrap();
    let ve: Vec<f32> = out_e.flatten_all().unwrap().to_vec1().unwrap();
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
    let dbf: Vec<f32> = grads_f.get(&b_f).unwrap().to_vec1().unwrap();
    let dbe: Vec<f32> = grads_e.get(&b_e).unwrap().to_vec1().unwrap();
    for (i, (f, e)) in dbf.iter().zip(dbe.iter()).enumerate() {
        assert!((f - e).abs() < 1e-3, "dbeta[{i}]: fused {f} vs formula {e}");
    }
}

/// A deterministic (LCG-seeded) fixture generator, reused across the
/// production-width tests below — no external RNG dependency, and the
/// same seed always yields the same fixture (family J: reproducible
/// numerics).
fn lcg_f32(seed: &mut u32, half_width: f32) -> f32 {
    *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
    let u = (*seed >> 8) as f32 / (1u32 << 24) as f32; // [0, 1)
    (u * 2.0 - 1.0) * half_width
}

/// Production-width (rule 3.4 of the kernel guide): `hidden` in
/// `{768, 1024}` (BERT-base/ModernBERT-large), non-uniform `dy`, `beta`
/// present. bf16 bound derived over `|xhat*gamma| + |beta|` — beta can
/// CANCEL the scaled term, so the bound sums the two magnitudes rather
/// than assuming the output's own magnitude bounds the error (the same
/// derivation `ops::layer_norm`'s own `bf16_forward_biased_matches_f32_
/// accumulation_rounded_once` unit test uses, at production scale here).
#[test]
fn bf16_biased_fwd_bwd_bound_at_production_width() {
    let device = Device::Cpu;
    for &hidden in &[768usize, 1024usize] {
        let rows = 4;
        let mut seed = 0xC0FFEEu32 ^ (hidden as u32);
        let xv: Vec<f32> = (0..rows * hidden)
            .map(|_| lcg_f32(&mut seed, 6.0))
            .collect();
        let gv: Vec<f32> = (0..hidden).map(|_| 0.5 + lcg_f32(&mut seed, 1.0)).collect();
        let bv: Vec<f32> = (0..hidden).map(|_| lcg_f32(&mut seed, 1.5)).collect();
        let dyv: Vec<f32> = (0..rows * hidden)
            .map(|_| lcg_f32(&mut seed, 3.0))
            .collect();

        let xb: Vec<bf16> = xv.iter().map(|&v| bf16::from_f32(v)).collect();
        let gb: Vec<bf16> = gv.iter().map(|&v| bf16::from_f32(v)).collect();
        let bb: Vec<bf16> = bv.iter().map(|&v| bf16::from_f32(v)).collect();
        let dyb: Vec<bf16> = dyv.iter().map(|&v| bf16::from_f32(v)).collect();

        let x_f =
            Var::from_tensor(&Tensor::from_slice(&xb, (rows, hidden), &device).unwrap()).unwrap();
        let g_f = Var::from_tensor(&Tensor::from_slice(&gb, (hidden,), &device).unwrap()).unwrap();
        let b_f = Var::from_tensor(&Tensor::from_slice(&bb, (hidden,), &device).unwrap()).unwrap();
        let dy = Tensor::from_slice(&dyb, (rows, hidden), &device).unwrap();

        let out = fused_biased(
            1e-5,
            true,
            true,
            x_f.as_tensor(),
            g_f.as_tensor(),
            b_f.as_tensor(),
        )
        .unwrap();
        let out_v: Vec<bf16> = out.flatten_all().unwrap().to_vec1().unwrap();

        // Independent f64 reference: mean/var over the bf16-rounded
        // inputs, gamma/beta applied in f64, one final rounding to bf16.
        let mut max_rel: f32 = 0.0;
        for r in 0..rows {
            let row: Vec<f64> = xb[r * hidden..(r + 1) * hidden]
                .iter()
                .map(|v| v.to_f32() as f64)
                .collect();
            let mean: f64 = row.iter().sum::<f64>() / hidden as f64;
            let var: f64 = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / hidden as f64;
            let invvar = 1.0 / (var + 1e-5).sqrt();
            for c in 0..hidden {
                let xhat = (row[c] - mean) * invvar;
                let g = gb[c].to_f32() as f64;
                let b = bb[c].to_f32() as f64;
                let expected = (xhat * g + b) as f32;
                let got = out_v[r * hidden + c].to_f32();
                let bound = ((xhat * g).abs() as f32 + b.abs() as f32) * 2e-2 + 1e-2;
                assert!(
                    got.is_finite() && (got - expected).abs() < bound,
                    "hidden={hidden} row={r} col={c}: got {got} vs expected {expected} \
                     (bound {bound})"
                );
                if expected != 0.0 {
                    max_rel = max_rel.max((got - expected).abs() / expected.abs());
                }
            }
        }
        println!(
            "bf16_biased_fwd_bwd_bound_at_production_width: hidden={hidden} max_rel={max_rel}"
        );

        let loss = (&out * &dy).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        assert!(grads.get(&x_f).is_some());
        assert!(grads.get(&g_f).is_some());
        let dbeta: Vec<bf16> = grads.get(&b_f).unwrap().to_vec1().unwrap();
        assert!(
            dbeta.iter().any(|v| v.to_f32() != 0.0),
            "hidden={hidden}: dbeta must be a live (non-vacuous) signal, not all-zero"
        );
    }
}

/// [`bf16_biased_fwd_bwd_bound_at_production_width`]'s F16 twin. F16 has
/// MORE mantissa bits than bf16 (10 vs 7), so this bound must be TIGHTER
/// than bf16's, not wider: bf16's own coefficients (`2e-2`/`1e-2`) are
/// derived from a half-bf16-ulp relative error (`2^-8 ≈ 3.9e-3`) scaled up
/// by this op's own n-term reduction slack (the same "round-once epilogue"
/// derivation `ops::layer_norm`'s `bf16_forward_biased_matches_f32_
/// accumulation_rounded_once` unit test uses). F16's half-ulp relative
/// error is `2^-11 ≈ 4.9e-4` — exactly `1/8` of bf16's `2^-8` (11 vs 8
/// mantissa+implicit bits, `2^(11-8) = 8`) — so applying the SAME
/// reduction-slack multiplier to the smaller per-element error and scaling
/// bf16's own coefficients down by that same `1/8` ulp ratio gives F16's
/// bound: `2e-2 / 8 = 2.5e-3`, `1e-2 / 8 = 1.25e-3`. Reports `max_rel`
/// exactly like the bf16 twin so the slack is visible.
#[test]
fn f16_biased_fwd_bwd_bound_at_production_width() {
    let device = Device::Cpu;
    for &hidden in &[768usize, 1024usize] {
        let rows = 4;
        let mut seed = 0xF16Bu32 ^ (hidden as u32);
        let xv: Vec<f32> = (0..rows * hidden)
            .map(|_| lcg_f32(&mut seed, 4.0))
            .collect();
        let gv: Vec<f32> = (0..hidden).map(|_| 0.5 + lcg_f32(&mut seed, 0.8)).collect();
        let bv: Vec<f32> = (0..hidden).map(|_| lcg_f32(&mut seed, 1.0)).collect();
        let dyv: Vec<f32> = (0..rows * hidden)
            .map(|_| lcg_f32(&mut seed, 2.0))
            .collect();

        let xh: Vec<f16> = xv.iter().map(|&v| f16::from_f32(v)).collect();
        let gh: Vec<f16> = gv.iter().map(|&v| f16::from_f32(v)).collect();
        let bh: Vec<f16> = bv.iter().map(|&v| f16::from_f32(v)).collect();
        let dyh: Vec<f16> = dyv.iter().map(|&v| f16::from_f32(v)).collect();

        let x_f =
            Var::from_tensor(&Tensor::from_slice(&xh, (rows, hidden), &device).unwrap()).unwrap();
        let g_f = Var::from_tensor(&Tensor::from_slice(&gh, (hidden,), &device).unwrap()).unwrap();
        let b_f = Var::from_tensor(&Tensor::from_slice(&bh, (hidden,), &device).unwrap()).unwrap();
        let dy = Tensor::from_slice(&dyh, (rows, hidden), &device).unwrap();

        let out = fused_biased(
            1e-5,
            true,
            true,
            x_f.as_tensor(),
            g_f.as_tensor(),
            b_f.as_tensor(),
        )
        .unwrap();
        let out_v: Vec<f16> = out.flatten_all().unwrap().to_vec1().unwrap();

        let mut max_rel: f32 = 0.0;
        for r in 0..rows {
            let row: Vec<f64> = xh[r * hidden..(r + 1) * hidden]
                .iter()
                .map(|v| v.to_f32() as f64)
                .collect();
            let mean: f64 = row.iter().sum::<f64>() / hidden as f64;
            let var: f64 = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / hidden as f64;
            let invvar = 1.0 / (var + 1e-5).sqrt();
            for c in 0..hidden {
                let xhat = (row[c] - mean) * invvar;
                let g = gh[c].to_f32() as f64;
                let b = bh[c].to_f32() as f64;
                let expected = (xhat * g + b) as f32;
                let got = out_v[r * hidden + c].to_f32();
                let bound = ((xhat * g).abs() as f32 + b.abs() as f32) * 2.5e-3 + 1.25e-3;
                assert!(
                    got.is_finite() && (got - expected).abs() < bound,
                    "hidden={hidden} row={r} col={c}: got {got} vs expected {expected} \
                     (bound {bound})"
                );
                if expected != 0.0 {
                    max_rel = max_rel.max((got - expected).abs() / expected.abs());
                }
            }
        }
        println!("f16_biased_fwd_bwd_bound_at_production_width: hidden={hidden} max_rel={max_rel}");

        let loss = (&out * &dy).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        assert!(grads.get(&x_f).is_some());
        assert!(grads.get(&g_f).is_some());
        let dbeta: Vec<f16> = grads.get(&b_f).unwrap().to_vec1().unwrap();
        assert!(
            dbeta.iter().any(|v| v.to_f32() != 0.0),
            "hidden={hidden}: dbeta must be a live (non-vacuous) signal, not all-zero"
        );
    }
}
