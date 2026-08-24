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
use jammi_kernels::ops::{apply2, Axpy, FullyMaskedPolicy, SoftmaxLastDimFused};

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

// ---------------------------------------------------------------------
// SoftmaxLastDimFused: `scale` semantics — folding `1/sqrt(head_dim)`
// into this op, replacing the `Op::Affine` node
// `ModernBertAttention::forward`'s training arm used to retain — see
// `jammi_kernels::ops::softmax`'s module doc's "scale semantics" section.
//
// The comparison target throughout is `fused-with-scale` vs. `(candle's
// own Tensor::affine) THEN fused-no-scale` — the EXACT two-op composition
// `scale` replaces, using THIS crate's own reduction kernel on both
// sides, not `candle_nn::ops::softmax` (whose own fold order need not
// match this op's — that comparison already lives in `ops/softmax.rs`'s
// module tests, at its own stated tolerance, and is orthogonal to
// whether `scale` itself is applied correctly).
// ---------------------------------------------------------------------

fn softmax_scaled(
    scores: &Tensor,
    mask: &Tensor,
    policy: FullyMaskedPolicy,
    scale: f32,
) -> candle_core::Result<Tensor> {
    // `.with_scale` validates `scale` (family D — see its own doc); every
    // fixture in this file passes a genuine finite positive scale, so
    // `.expect` here is a test-fixture assumption, not a silent unwrap of a
    // real fallible path. The domain refusal itself is exercised directly
    // by `ops::softmax::tests::with_scale_refuses_zero_negative_nan_and_infinite`
    // (in `crates/jammi-kernels/src/ops/softmax.rs`'s own `mod tests`, not
    // in this file).
    apply2(
        scores,
        mask,
        SoftmaxLastDimFused::new(policy)
            .with_scale(scale)
            .expect("test fixture scale must be finite and > 0.0"),
    )
}

fn softmax_unscaled(
    scores: &Tensor,
    mask: &Tensor,
    policy: FullyMaskedPolicy,
) -> candle_core::Result<Tensor> {
    apply2(scores, mask, SoftmaxLastDimFused::new(policy))
}

fn scale_scores_fixture(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 * 0.013 - 5.0).sin() * 2.0)
        .collect()
}

/// Mask shape `[batch, 1, seq, seq]` — the local-attention COMBINED-mask
/// shape `ModernBertAttention::forward`'s training arm actually builds
/// (padding mask + sliding-window band, pre-summed before reaching this
/// op). Values are drawn from the REAL ModernBERT mask alphabet: `0.0`
/// (unmasked), `-10_000.0` (one `MASKED_LOGIT`), `-20_000.0` (two
/// `MASKED_LOGIT`s summed — the local-layer padding+band combination).
/// The LAST query row of every batch element is forced FULLY masked
/// (alternating `-10_000.0`/`-20_000.0`, never `0.0`), exercising
/// `FullyMaskedPolicy::Zeros`'s short-circuit at production width.
fn scale_mask_fixture(batch: usize, seq: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; batch * seq * seq];
    for b in 0..batch {
        for q in 0..seq {
            for k in 0..seq {
                let idx = (b * seq + q) * seq + k;
                mask[idx] = if q == seq - 1 {
                    if k % 2 == 0 {
                        -10_000.0
                    } else {
                        -20_000.0
                    }
                } else {
                    match (q * 7 + k * 3 + b) % 5 {
                        0 => -20_000.0,
                        1 => -10_000.0,
                        _ => 0.0,
                    }
                };
            }
        }
    }
    mask
}

/// A REL-POS-BIAS-shaped mask: small, continuous, NEVER `0.0` and NEVER
/// near `MASKED_LOGIT` magnitude, unlike [`scale_mask_fixture`]'s real
/// ModernBERT alphabet `{0.0, -10_000.0, -20_000.0}`. This exists to close
/// a specific gap that alphabet cannot: at `mask = 0.0`, adding it is a
/// round-identity regardless of how precisely the scaled score was rounded
/// beforehand (`x + 0.0 == x` at every precision); at `mask ≈
/// MASKED_LOGIT`, the add ANNIHILATES the scaled score (BF16's ULP near
/// `10_000` dwarfs any real score), which also erases any precision
/// difference in how the score itself was rounded. Neither case can
/// observe WHERE the BF16 kernel's intermediate rounding happens
/// (`softmax_row_bf16`'s `scaled = bf16::from_f32(scores[i] * scale_bf)`,
/// rounded to BF16 BEFORE the mask add — see the module doc's "scale
/// semantics" section) — an oracle built only from that alphabet would
/// pass identically whether or not that intermediate rounding step were
/// silently dropped (i.e. the product kept in F32 until AFTER the mask
/// add). Values in `[-0.5, 0.5]` are the same order of magnitude as the
/// scaled scores themselves, so the mask-add's OWN rounding is where a
/// dropped intermediate-rounding regression would actually show up.
fn small_bias_mask_fixture(batch: usize, seq: usize) -> Vec<f32> {
    (0..batch * seq * seq)
        .map(|i| (i as f32 * 0.037 - 3.0).sin() * 0.5)
        .collect()
}

/// `1.0 / sqrt(64)` — ModernBERT-large's REAL `head_dim` (`num_heads =
/// 16`, `hidden = 1024`). `0.125` is an exact power of two: representable
/// EXACTLY (no rounding at all) in F32, F64, AND BF16 alike, so this
/// fixture's bit-exactness is a MATHEMATICAL guarantee, not a lucky
/// empirical coincidence — the `f64 -> f32 -> bf16` double-rounding path
/// this op's `scale` field takes and the `f64 -> bf16` single-rounding
/// path `Tensor::affine`'s own `T::from_f64` takes both round the SAME
/// exact value to the SAME bits.
const PRODUCTION_SCALE: f64 = 0.125;

/// `1.0 / sqrt(128)` — the OTHER ModernBERT-class `head_dim` value
/// (`head_dim = hidden_size / num_attention_heads`, which need not land on
/// `64` for every config this crate's admission predicate must still
/// accept). UNLIKE [`PRODUCTION_SCALE`], `128` is a power of two but
/// `sqrt(128)` and therefore `1/sqrt(128)` is IRRATIONAL — genuinely
/// rounded at every precision, not a lucky exact-representation coincidence
/// — so this exercises the bounded-double-rounding class
/// [`PRODUCTION_SCALE`]'s bit-exactness does not, at REAL production
/// tensor widths (unlike the small `head_dim = 48` toy fixture below).
fn head_dim_128_scale() -> f64 {
    1.0 / 128.0f64.sqrt()
}

fn f32_scale_bit_exact_vs_affine_then_unscaled(batch: usize, heads: usize, seq: usize, scale: f64) {
    let device = Device::Cpu;
    let sv = scale_scores_fixture(batch * heads * seq * seq);
    let mv = scale_mask_fixture(batch, seq);
    let scores = Tensor::from_slice(&sv, (batch, heads, seq, seq), &device).unwrap();
    let mask = Tensor::from_slice(&mv, (batch, 1, seq, seq), &device).unwrap();

    let fused_scaled =
        softmax_scaled(&scores, &mask, FullyMaskedPolicy::Zeros, scale as f32).unwrap();
    // The two-op composition `scale` replaces: candle's own
    // `Tensor::affine` (the SAME op `ModernBertAttention::forward` used
    // to call via `scores / sqrt(head_dim)`) THEN this op with `scale =
    // 1.0` (its default).
    let affined = scores.affine(scale, 0.0).unwrap();
    let affine_then_unscaled = softmax_unscaled(&affined, &mask, FullyMaskedPolicy::Zeros).unwrap();

    let a: Vec<f32> = fused_scaled.flatten_all().unwrap().to_vec1().unwrap();
    let b: Vec<f32> = affine_then_unscaled
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(
        a.iter().any(|v| v.is_finite() && *v != 0.0),
        "fixture must be non-degenerate"
    );
    // F32 bit-exactness (unlike the BF16 leg below) does NOT depend on
    // `scale` being an exact power of two: `SoftmaxLastDimFused::scale` is
    // populated by the SAME `f64 -> f32` cast `Tensor::affine`'s own F32
    // branch performs (see `ops::softmax`'s module doc), so this holds for
    // ANY finite `scale` -- exercised here at BOTH `PRODUCTION_SCALE`
    // (power-of-two) and `head_dim_128_scale()` (irrational) call sites.
    assert_eq!(
        a, b,
        "F32: fused-with-scale must be BIT-EXACT vs affine-then-fused-no-scale \
         (both run the SAME reduction kernel, differing only in when the scale \
         multiply happens — two ordinary f32 ops, same order, on both sides)"
    );
}

#[test]
fn softmax_scale_f32_bit_exact_vs_affine_then_unscaled_production_width_2_16_128_128() {
    f32_scale_bit_exact_vs_affine_then_unscaled(2, 16, 128, PRODUCTION_SCALE);
}

#[test]
fn softmax_scale_f32_bit_exact_vs_affine_then_unscaled_production_width_1_16_512_512() {
    f32_scale_bit_exact_vs_affine_then_unscaled(1, 16, 512, PRODUCTION_SCALE);
}

/// `head_dim = 128` leg, F32: still BIT-EXACT (see the helper's own doc for
/// why F32 exactness does not depend on `scale` being a power of two).
#[test]
fn softmax_scale_f32_bit_exact_vs_affine_then_unscaled_head_dim_128_production_width_2_16_128_128()
{
    f32_scale_bit_exact_vs_affine_then_unscaled(2, 16, 128, head_dim_128_scale());
}

fn bf16_scale_max_ulp_diff_vs_affine_then_unscaled(
    batch: usize,
    heads: usize,
    seq: usize,
    scale: f64,
) -> i32 {
    let device = Device::Cpu;
    let sv = scale_scores_fixture(batch * heads * seq * seq);
    let mv = scale_mask_fixture(batch, seq);
    let sb: Vec<bf16> = sv.iter().map(|&v| bf16::from_f32(v)).collect();
    let mb: Vec<bf16> = mv.iter().map(|&v| bf16::from_f32(v)).collect();
    let scores = Tensor::from_slice(&sb, (batch, heads, seq, seq), &device).unwrap();
    let mask = Tensor::from_slice(&mb, (batch, 1, seq, seq), &device).unwrap();

    let fused_scaled =
        softmax_scaled(&scores, &mask, FullyMaskedPolicy::Zeros, scale as f32).unwrap();
    // `scores.affine(scale, 0.0)` on a BF16 tensor calls candle's OWN
    // `Affine<bf16>` branch (`mul = bf16::from_f64(scale)`, direct
    // f64->bf16, single rounding) -- the TRUE eager rounding path this
    // op's own `scale` field must reproduce (see the module doc).
    let affined = scores.affine(scale, 0.0).unwrap();
    let affine_then_unscaled = softmax_unscaled(&affined, &mask, FullyMaskedPolicy::Zeros).unwrap();

    let a: Vec<bf16> = fused_scaled.flatten_all().unwrap().to_vec1().unwrap();
    let b: Vec<bf16> = affine_then_unscaled
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(
        a.iter().any(|v| v.to_f32().abs() > 1e-3),
        "fixture must be non-degenerate"
    );
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x.to_bits() as i32 - y.to_bits() as i32).abs())
        .max()
        .unwrap_or(0)
}

/// MEASURED, not assumed: at ModernBERT-large's REAL `head_dim = 64`
/// (`scale = 0.125`, an exact power of two — see [`PRODUCTION_SCALE`]'s
/// doc), the `f64 -> f32 -> bf16` double-rounding path this op's `scale`
/// field takes and the `f64 -> bf16` single-rounding path `Tensor::affine`
/// takes land on the IDENTICAL bit pattern (no rounding occurs on either
/// path for an exact power of two), so this op is BIT-EXACT vs. the
/// two-op composition at BF16 too, at production width. This is a
/// MATHEMATICAL guarantee for THIS scale value, not a general BF16
/// claim — see the `head_dim = 128` legs below (and `ops::softmax`'s
/// module doc) for the double-rounding class this op discloses for a
/// non-exact scale.
#[test]
fn softmax_scale_bf16_bit_exact_vs_affine_then_unscaled_production_width_2_16_128_128() {
    let max_ulp = bf16_scale_max_ulp_diff_vs_affine_then_unscaled(2, 16, 128, PRODUCTION_SCALE);
    eprintln!("MEASURED max bf16 ULP diff (scale rounding vs affine), seq=128: {max_ulp}");
    assert_eq!(
        max_ulp, 0,
        "expected BIT-EXACT at head_dim=64 (scale=0.125 is an exact power of two, \
         representable exactly in F32/F64/BF16 alike) -- measured max ULP diff {max_ulp}"
    );
}

#[test]
fn softmax_scale_bf16_bit_exact_vs_affine_then_unscaled_production_width_1_16_512_512() {
    let max_ulp = bf16_scale_max_ulp_diff_vs_affine_then_unscaled(1, 16, 512, PRODUCTION_SCALE);
    eprintln!("MEASURED max bf16 ULP diff (scale rounding vs affine), seq=512: {max_ulp}");
    assert_eq!(
        max_ulp, 0,
        "expected BIT-EXACT at head_dim=64 (scale=0.125 is an exact power of two, \
         representable exactly in F32/F64/BF16 alike) -- measured max ULP diff {max_ulp}"
    );
}

/// `head_dim = 128` leg, BF16, at PRODUCTION tensor widths (unlike the
/// small toy `head_dim = 48` fixture below). `1/sqrt(128)`'s F32 bit
/// pattern is `0x3db5_04f3` — nowhere near a BF16 rounding tie (low 16
/// bits `0x04f3`, far from the `0x8000` halfway point), so this leg does
/// NOT exercise the `f64 -> f32 -> bf16` double-rounding vs. `f64 ->
/// bf16` gap the module doc discloses (`128` is not one of the 4 known
/// mismatch values — see the module doc's `scale_constant_bf16_max_1_ulp_across_head_dim_1_to_20000`
/// sweep, and its separate exact-tie sibling test, for that). What this leg DOES
/// measure: this op's BF16 kernel is bit-exact vs. affine-then-unscaled
/// at this specific `head_dim`, at production tensor width — MEASURED,
/// not assumed, still bounded `<= 1` ULP by the assertion (not `== 0`)
/// so a genuine divergence at some other `head_dim` is not silently
/// masked.
#[test]
fn softmax_scale_bf16_head_dim_128_bounded_vs_affine_then_unscaled_production_width_2_16_128_128() {
    let max_ulp = bf16_scale_max_ulp_diff_vs_affine_then_unscaled(2, 16, 128, head_dim_128_scale());
    eprintln!("MEASURED max bf16 ULP diff (head_dim=128, non-power-of-two), seq=128: {max_ulp}");
    assert!(
        max_ulp <= 1,
        "double-rounding gap larger than the theoretical 1-ULP worst case: {max_ulp}"
    );
}

#[test]
fn softmax_scale_bf16_head_dim_128_bounded_vs_affine_then_unscaled_production_width_1_16_512_512() {
    let max_ulp = bf16_scale_max_ulp_diff_vs_affine_then_unscaled(1, 16, 512, head_dim_128_scale());
    eprintln!("MEASURED max bf16 ULP diff (head_dim=128, non-power-of-two), seq=512: {max_ulp}");
    assert!(
        max_ulp <= 1,
        "double-rounding gap larger than the theoretical 1-ULP worst case: {max_ulp}"
    );
}

/// A NON-power-of-two scale, disclosed honestly rather than swept under
/// the ModernBERT-only fixture above: `head_dim = 48`, `scale =
/// 1/sqrt(48)`, F32 bit pattern `0x3e13_cd3a` — like `head_dim = 128`
/// above, NOT near a BF16 rounding tie (low 16 bits `0xcd3a`), so this
/// leg does NOT exercise the double-rounding gap the module doc names
/// either (see that doc's sweep test and its two genuine-tie legs for
/// what does). MEASURED, not assumed — this op is NOT claimed bit-exact
/// here; the point is pinning this op's kernel against the
/// affine-then-unscaled composition for at least one non-power-of-two,
/// non-ModernBERT-production value, not just the power-of-two production
/// one.
#[test]
fn softmax_scale_bf16_non_power_of_two_head_dim_is_measured_not_assumed() {
    let device = Device::Cpu;
    let scale = 1.0 / (48.0f64).sqrt();
    let seq = 8;
    let sv = scale_scores_fixture(2 * seq * seq);
    let mv = scale_mask_fixture(1, seq);
    let sb: Vec<bf16> = sv.iter().map(|&v| bf16::from_f32(v)).collect();
    let mb: Vec<bf16> = mv.iter().map(|&v| bf16::from_f32(v)).collect();
    let scores = Tensor::from_slice(&sb, (1, 2, seq, seq), &device).unwrap();
    let mask = Tensor::from_slice(&mb, (1, 1, seq, seq), &device).unwrap();

    let fused_scaled =
        softmax_scaled(&scores, &mask, FullyMaskedPolicy::Zeros, scale as f32).unwrap();
    let affined = scores.affine(scale, 0.0).unwrap();
    let affine_then_unscaled = softmax_unscaled(&affined, &mask, FullyMaskedPolicy::Zeros).unwrap();

    let a: Vec<bf16> = fused_scaled.flatten_all().unwrap().to_vec1().unwrap();
    let b: Vec<bf16> = affine_then_unscaled
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let max_ulp = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x.to_bits() as i32 - y.to_bits() as i32).abs())
        .max()
        .unwrap_or(0);
    eprintln!("MEASURED max bf16 ULP diff (scale=1/sqrt(48), non-power-of-two): {max_ulp}");
    // A generous bound: this is a MEASURED double-rounding class, not a
    // correctness bug -- 1 ULP is the theoretical worst case for a single
    // extra rounding step, so anything larger would indicate a REAL bug,
    // not just the disclosed double-rounding gap.
    assert!(
        max_ulp <= 1,
        "double-rounding gap larger than the theoretical 1-ULP worst case: {max_ulp}"
    );
}

/// Closes a real gap in every OTHER BF16 leg above: every one of them uses
/// [`scale_mask_fixture`]'s real ModernBERT alphabet (`{0.0, -10_000.0,
/// -20_000.0}`), and at BOTH ends of that alphabet the mask add cannot
/// observe WHERE the intermediate rounding happens (`0.0` is a
/// round-identity; `MASKED_LOGIT` magnitude annihilates the scaled score
/// regardless of how precisely it was rounded beforehand) — see
/// [`small_bias_mask_fixture`]'s doc. If `softmax_row_bf16`'s
/// load-bearing intermediate rounding (`scaled =
/// bf16::from_f32(scores[i].to_f32() * scale_bf.to_f32())`, rounded to
/// BF16 BEFORE the mask add, not kept in F32 until after it) were
/// silently dropped, every OTHER BF16 oracle in this file would still
/// pass — this is the one that would not.
///
/// Uses `FullyMaskedPolicy::Propagate` (not `Zeros`): the small, signed
/// `[-0.5, 0.5]` mask fixture is not the real masking alphabet, so a row
/// that happens to land all-negative is not "fully masked" in the
/// production sense — `Zeros`'s short-circuit would misfire on it and
/// mask the very rounding behavior this test exists to observe. Uses
/// [`head_dim_128_scale`] (irrational), not [`PRODUCTION_SCALE`]: `0.125`
/// is an exact power of two, so `scores[i] * 0.125` is an EXACT BF16
/// operation (a power-of-two multiply only shifts the exponent) with no
/// rounding to relocate in the first place — this test needs a `scale`
/// whose multiply genuinely rounds.
///
/// RED-verified: temporarily changing `softmax_row_bf16` to compute
/// `bf16::from_f32(scores[i].to_f32() * scale_bf.to_f32() + mask[i].to_f32())`
/// directly (keeping the product in F32 across the mask add, i.e.
/// dropping the intermediate rounding this test defends) makes this
/// assertion fail on this fixture, while every other test in this file
/// (including the other BF16 legs above) still passes.
#[test]
fn softmax_scale_bf16_small_additive_mask_bit_exact_vs_affine_then_unscaled() {
    let device = Device::Cpu;
    let batch = 2;
    let heads = 16;
    let seq = 128;
    let scale = head_dim_128_scale();
    let sv = scale_scores_fixture(batch * heads * seq * seq);
    let mv = small_bias_mask_fixture(batch, seq);
    let sb: Vec<bf16> = sv.iter().map(|&v| bf16::from_f32(v)).collect();
    let mb: Vec<bf16> = mv.iter().map(|&v| bf16::from_f32(v)).collect();
    let scores = Tensor::from_slice(&sb, (batch, heads, seq, seq), &device).unwrap();
    let mask = Tensor::from_slice(&mb, (batch, 1, seq, seq), &device).unwrap();

    let fused_scaled =
        softmax_scaled(&scores, &mask, FullyMaskedPolicy::Propagate, scale as f32).unwrap();
    let affined = scores.affine(scale, 0.0).unwrap();
    let affine_then_unscaled =
        softmax_unscaled(&affined, &mask, FullyMaskedPolicy::Propagate).unwrap();

    let a: Vec<bf16> = fused_scaled.flatten_all().unwrap().to_vec1().unwrap();
    let b: Vec<bf16> = affine_then_unscaled
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(
        a.iter().any(|v| v.to_f32().abs() > 1e-3),
        "fixture must be non-degenerate"
    );
    assert_eq!(
        a, b,
        "BF16 small-additive-mask leg: fused-with-scale must be BIT-EXACT vs \
         affine-then-fused-no-scale — a mismatch here means the intermediate \
         rounding point (`scaled = bf16::from_f32(scores[i] * scale_bf)`, rounded \
         BEFORE the mask add) moved"
    );
}

/// `d(y)/d(scores) = d(y)/d(pre_softmax) * scale` (chain rule through
/// `pre_softmax = scale * scores + mask`) — verifies `bwd` actually
/// multiplies by `scale`, with a NON-UNIFORM `dy` seed (a uniform seed
/// would make this identically zero for every softmax row, since
/// `sum(y) == 1` — see `jammi-encoders`' `modernbert.rs` test of the same
/// shape for why that would be a VACUOUS check, family F).
#[test]
fn softmax_scale_bwd_multiplies_raw_dscores_by_scale() {
    let device = Device::Cpu;
    let scale = PRODUCTION_SCALE;
    let s0: [f32; 6] = [0.3, -1.2, 2.0, 0.1, -0.5, 1.7];
    let m0: [f32; 6] = [0.0, -10_000.0, 0.0, -20_000.0, 0.0, 0.0];
    let dy0: [f32; 6] = [0.5, -1.0, 2.0, 0.25, -0.75, 1.5];

    let scores = Var::from_tensor(&Tensor::from_slice(&s0, (2, 3), &device).unwrap()).unwrap();
    let mask = Tensor::from_slice(&m0, (1, 3), &device).unwrap();
    let dy = Tensor::from_slice(&dy0, (2, 3), &device).unwrap();

    let out = softmax_scaled(&scores, &mask, FullyMaskedPolicy::Propagate, scale as f32).unwrap();
    let loss = (&out * &dy).unwrap().sum_all().unwrap();
    let dscores: Vec<f32> = loss
        .backward()
        .unwrap()
        .get(&scores)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    // `d(y)/d(pre_softmax)`, from an INDEPENDENT graph rooted at the
    // already-scaled scores (matching the F32 arm's OWN `scores[i] *
    // scale` step exactly, so the two sides differ ONLY by the `* scale`
    // this test is checking for -- not by an unrelated rounding path).
    let s0_scaled: Vec<f32> = s0.iter().map(|&v| v * scale as f32).collect();
    let scaled_scores =
        Var::from_tensor(&Tensor::from_slice(&s0_scaled, (2, 3), &device).unwrap()).unwrap();
    let out_pre = softmax_unscaled(&scaled_scores, &mask, FullyMaskedPolicy::Propagate).unwrap();
    let loss_pre = (&out_pre * &dy).unwrap().sum_all().unwrap();
    let d_pre_softmax: Vec<f32> = loss_pre
        .backward()
        .unwrap()
        .get(&scaled_scores)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    assert!(
        dscores.iter().any(|v| v.abs() > 1e-4),
        "gradient must be measured-nonzero"
    );
    for (i, (d, dp)) in dscores.iter().zip(d_pre_softmax.iter()).enumerate() {
        let expected = scale as f32 * dp;
        assert!(
            (d - expected).abs() < 1e-5,
            "dscores[{i}] = {d} vs scale * d_pre_softmax = {expected} (scale={scale})"
        );
    }
}

/// Companion: `dmask` (when `mask` IS a `Var`) uses the RAW, UNSCALED
/// `d(y)/d(pre_softmax)` -- `d(pre_softmax)/d(mask) == 1`, never `scale`.
/// `y` itself depends on `scale` (`pre_softmax = scale*scores + mask`
/// evaluates at a DIFFERENT point for a different `scale`), so `dmask`
/// evaluated at a DIFFERENT `scale` is NOT expected to match numerically
/// (a naive "same `scores`, different `scale`, dmask must match" claim
/// would be mathematically wrong) -- the CORRECT claim, verified here, is
/// that `dmask` from the scaled run matches `dmask` from an INDEPENDENT
/// unscaled reference run evaluated AT THE SAME pre-softmax point
/// (pre-multiplied scores, `scale = 1.0`) -- i.e. `dmask` is exactly
/// `mask_grad(d_pre_softmax)`, never further multiplied by `scale`, using
/// the SAME "compare against an independent graph at the identical
/// pre-softmax point" construction `softmax_scale_bwd_multiplies_raw_dscores_by_scale`
/// uses for `dscores`.
#[test]
fn softmax_scale_dmask_uses_unscaled_gradient_not_scaled() {
    let device = Device::Cpu;
    let sv: [f32; 8] = [1.0, -1.0, 2.0, 0.5, -0.3, 1.7, 0.2, -2.1];
    let mv: [f32; 4] = [0.1, -0.2, 0.3, -0.1];
    let dyv: [f32; 8] = [0.4, -0.9, 1.1, 0.6, -1.3, 0.2, -0.5, 0.8];
    let scale = PRODUCTION_SCALE;
    let scores = Tensor::from_slice(&sv, (2, 4), &device).unwrap();
    // A NON-UNIFORM `dy` seed -- `Tensor::backward()`'s implicit all-ones
    // seed would make `dscores` (and therefore `dmask`) IDENTICALLY zero
    // regardless of `scale` (`sum(y) == 1` for every softmax row), which
    // would make this comparison VACUOUS (family F): both sides would be
    // trivially-equal zeros, proving nothing about `mask_grad`'s actual
    // scale-independence.
    let dy = Tensor::from_slice(&dyv, (2, 4), &device).unwrap();

    let mask_scaled = Var::from_tensor(&Tensor::from_slice(&mv, (1, 4), &device).unwrap()).unwrap();
    let out_scaled = softmax_scaled(
        &scores,
        &mask_scaled,
        FullyMaskedPolicy::Propagate,
        scale as f32,
    )
    .unwrap();
    let loss_scaled = (&out_scaled * &dy).unwrap().sum_all().unwrap();
    let dmask_scaled: Vec<f32> = loss_scaled
        .backward()
        .unwrap()
        .get(&mask_scaled)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    // The reference: an INDEPENDENT graph rooted at the ALREADY-scaled
    // scores (matching `pre_softmax = scale*scores` exactly), run through
    // the UNSCALED op (`scale = 1.0`) -- computes the IDENTICAL
    // pre-softmax value (hence the IDENTICAL `y`) the scaled run above
    // internally reaches, so `dmask` from this reference is directly
    // comparable, unlike a reference at the RAW (un-multiplied) `scores`
    // (a different pre-softmax point entirely).
    let sv_scaled: Vec<f32> = sv.iter().map(|&v| v * scale as f32).collect();
    let scores_pre = Tensor::from_slice(&sv_scaled, (2, 4), &device).unwrap();
    let mask_ref = Var::from_tensor(&Tensor::from_slice(&mv, (1, 4), &device).unwrap()).unwrap();
    let out_ref = softmax_unscaled(&scores_pre, &mask_ref, FullyMaskedPolicy::Propagate).unwrap();
    let loss_ref = (&out_ref * &dy).unwrap().sum_all().unwrap();
    let dmask_ref: Vec<f32> = loss_ref
        .backward()
        .unwrap()
        .get(&mask_ref)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    assert!(
        dmask_scaled.iter().any(|v| v.abs() > 1e-4),
        "gradient must be measured-nonzero"
    );
    assert_eq!(
        dmask_scaled, dmask_ref,
        "dmask must be IDENTICAL to the unscaled reference evaluated at the SAME \
         pre_softmax point -- dmask never gets the scale factor"
    );
}
