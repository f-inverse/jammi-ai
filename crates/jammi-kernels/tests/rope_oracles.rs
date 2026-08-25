//! CPU-hermetic oracles for `RopeFused` — the fused-vs-eager and
//! chain-rule rigor-chain `tests/layer_norm_oracles.rs` establishes,
//! adapted for RoPE's specific `bwd` mechanism (reusing the SAME
//! `KernelOp` with `sin` negated, rather than a dedicated second kernel —
//! see `jammi_kernels::ops::rope`'s module doc).
//!
//!   1. `gradcheck_dx_*` — `bwd` vs. central finite differences.
//!   2. `eager_vs_fused_*` — fwd+bwd vs. a hand-written composition of
//!      ordinary candle ops that reproduces the EXACT eager RoPE
//!      composition `RotaryEmbedding::apply` uses in `jammi-encoders`
//!      (ModernBERT): `narrow`/`neg`/`Tensor::cat`/broadcast-mul/add. This
//!      crate is a LEAF (no `jammi-*` deps), so it cannot import that
//!      function itself; the actual "against the real `apply()`" oracle
//!      lives in `jammi-encoders`'s own test suite, where that function is
//!      reachable. F32 is asserted BIT-EXACT (`assert_eq!`, not a
//!      tolerance) — the rounding model predicts it (identical IEEE-754
//!      op sequence, no `fmad` contraction on the CPU arm) and the
//!      measured result confirms it; the crate's derive-the-tolerance
//!      doctrine means a tolerance is stated only where the model
//!      predicts real divergence, never as a default hedge. BF16 is
//!      measured-nonzero within `BF16_ULP_TOL` — a deliberately
//!      conservative CEILING (the measured max on these fixtures is
//!      printed, and is smaller than the stated bound; see the constant's
//!      own doc), inherited from `LayerNormFused`'s identical bf16
//!      oracle for the same reason: bf16's own two-rounding-path
//!      divergence (see `eager()`'s doc) is real and worth a headroom
//!      margin, not a number to shrink to the observed minimum.
//!   3. `chain_rule_through_an_intermediate_x` — the same
//!      `is_variable() == false` intermediate hazard `Axpy`'s regression
//!      test exercises, at a 4D (batch/heads/seq/head_dim) shape matching
//!      the real call site.
//!   4. `negated_sin_bwd_matches_a_hand_computed_sign_flip` — a
//!      hand-computable 4-element case pinning the sign convention (which
//!      half of `rotate_half` gets the negation) so a sign flip fails with
//!      a readable message, independent of `ops::rope`'s own unit tests.
//!
//! The CUDA↔CPU parity leg (fwd + bwd, contiguous/narrowed/empty, head_dim
//! 64 and a non-power-of-two even head_dim, bf16+f32) lives in
//! `tests/cuda_parity.rs`, gated the same way `Axpy`'s/`LayerNormFused`'s
//! are.

use candle_core::{Device, Tensor, Var, D};
use half::bf16;
use jammi_kernels::ops::{apply3, RopeFused};

fn fused(negate_sin: bool, x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
    apply3(x, cos, sin, RopeFused::new(negate_sin))
}

/// The EXACT eager composition `RotaryEmbedding::apply` runs (see
/// `jammi-encoders/src/modernbert.rs`): `x` is `[batch, heads, seq,
/// head_dim]`, `cos`/`sin` are pre-broadcast to `[1, 1, seq, head_dim]`
/// (this fixture builds them directly in that shape, mirroring the
/// call site's cached/unsqueezed tables — table hoisting does not change
/// this math, only where the cast/unsqueeze happen once).
fn eager(x: &Tensor, cos_b: &Tensor, sin_b: &Tensor) -> candle_core::Result<Tensor> {
    let head_dim = x.dim(D::Minus1)?;
    let half = head_dim / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let neg_x2 = (x2 * -1.0f64)?;
    let rot_half = Tensor::cat(&[&neg_x2, &x1], D::Minus1)?;
    let cos_part = x.broadcast_mul(cos_b)?;
    let sin_part = rot_half.broadcast_mul(sin_b)?;
    (cos_part + sin_part)?.contiguous()
}

/// Builds a deterministic `[period, hidden]` table pair from a fixed seed
/// (real angle-derived values, not arbitrary constants) with the SAME
/// column-duplication `RotaryEmbedding::new` bakes in (`cos[.., i] ==
/// cos[.., i + half]`) — this op's domain premise (module doc).
fn table(period: usize, hidden: usize, theta_base: f64) -> (Vec<f32>, Vec<f32>) {
    let half = hidden / 2;
    let mut cos = vec![0f32; period * hidden];
    let mut sin = vec![0f32; period * hidden];
    for pos in 0..period {
        for half_pass in 0..2 {
            for i in 0..half {
                let theta = (pos as f64) * theta_base.powf(-2.0 * i as f64 / hidden as f64);
                let idx = pos * hidden + half_pass * half + i;
                cos[idx] = theta.cos() as f32;
                sin[idx] = theta.sin() as f32;
            }
        }
    }
    (cos, sin)
}

#[test]
fn gradcheck_dx_f32_vs_central_finite_differences_4d_shape() {
    let device = Device::Cpu;
    let (batch, heads, seq, hidden) = (2, 2, 3, 8);
    let n = batch * heads * seq * hidden;
    let x0: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.19 - 2.0).sin() * 2.5)
        .collect();
    let (cos_v, sin_v) = table(seq, hidden, 100.0);

    let x =
        Var::from_tensor(&Tensor::from_slice(&x0, (batch, heads, seq, hidden), &device).unwrap())
            .unwrap();
    let cos = Tensor::from_slice(&cos_v, (1, 1, seq, hidden), &device).unwrap();
    let sin = Tensor::from_slice(&sin_v, (1, 1, seq, hidden), &device).unwrap();

    let out = fused(false, &x, &cos, &sin).unwrap();
    let grads = out.backward().unwrap();
    let dx: Vec<f32> = grads
        .get(&x)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let sum_fwd = |x: &Tensor| -> f64 {
        fused(false, x, &cos, &sin)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap() as f64
    };

    let fd_eps = 2e-3f32;
    let tol = 5e-2f64;
    for i in 0..n {
        let mut xp = x0.clone();
        xp[i] += fd_eps;
        let mut xm = x0.clone();
        xm[i] -= fd_eps;
        let xp_t = Tensor::from_slice(&xp, (batch, heads, seq, hidden), &device).unwrap();
        let xm_t = Tensor::from_slice(&xm, (batch, heads, seq, hidden), &device).unwrap();
        let numeric = (sum_fwd(&xp_t) - sum_fwd(&xm_t)) / (2.0 * fd_eps as f64);
        assert!(
            (numeric - dx[i] as f64).abs() < tol,
            "dx[{i}]: numeric {numeric} vs analytic {}",
            dx[i]
        );
    }
}

#[test]
fn eager_vs_fused_f32_fwd_and_bwd_are_bit_exact() {
    let device = Device::Cpu;
    let (batch, heads, seq, hidden) = (2, 3, 4, 8);
    let n = batch * heads * seq * hidden;
    let x0: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.11 - 1.0).cos() * 3.0)
        .collect();
    let (cos_v, sin_v) = table(seq, hidden, 10_000.0);

    let x_f =
        Var::from_tensor(&Tensor::from_slice(&x0, (batch, heads, seq, hidden), &device).unwrap())
            .unwrap();
    let x_e =
        Var::from_tensor(&Tensor::from_slice(&x0, (batch, heads, seq, hidden), &device).unwrap())
            .unwrap();
    let cos = Tensor::from_slice(&cos_v, (1, 1, seq, hidden), &device).unwrap();
    let sin = Tensor::from_slice(&sin_v, (1, 1, seq, hidden), &device).unwrap();

    let out_f = fused(false, &x_f, &cos, &sin).unwrap();
    let out_e = eager(&x_e, &cos, &sin).unwrap();
    let vf: Vec<f32> = out_f.flatten_all().unwrap().to_vec1().unwrap();
    let ve: Vec<f32> = out_e.flatten_all().unwrap().to_vec1().unwrap();
    // BIT-EXACT, not a tolerance: the rounding model predicts it. Both
    // paths compute the identical IEEE-754 op sequence per element —
    // `round(x[col]*cos[col]) + round(rotate_half(x)[col]*sin[col])`, one
    // rounded multiply per term then one rounded add — because
    // multiplying by `sign == 1.0` (the fused kernel's identity case) and
    // `rotate_half`'s `-1.0` negation are BOTH exact IEEE-754 operations
    // (negation and multiplication by 1.0 introduce no rounding), and
    // `cos[j] == cos[j+half]` / `sin[j] == sin[j+half]` are the SAME `f32`
    // bit pattern by table construction (`RotaryEmbedding::new` computes
    // each angle's `cos()`/`sin()` once in `f64` and writes the identical
    // rounded `f32` to both columns), not merely equal values. This CPU
    // build applies no `fmad` contraction (that is an nvcc-only default
    // this crate's CUDA build declines to force off globally — see
    // `build.rs`'s PINNED FLAGS comment — irrelevant here since this is
    // the CPU arm), so there is no other source of divergence. `assert_eq!`
    // is therefore the correct assertion, not a loosely-related tolerance
    // check — a real divergence here would mean the kernel's op grouping
    // does not actually match the eager composition's, which is exactly
    // the class of bug this oracle exists to catch.
    assert_eq!(vf, ve, "fwd must be bit-exact, not merely close");

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
    // Same bit-exactness argument extends to `dx`: the fused kernel's
    // `negate_sin=true` reuse computes each `dx` element via the same two
    // rounded multiplies (`dy[j]*cos[j]`, `dy[j+half]*sin[j]`) as the
    // eager composition's autograd-derived backward, with only EXACT
    // negations and a commutative final add differing in between.
    assert_eq!(dxf, dxe, "dx must be bit-exact, not merely close");
}

/// bf16 fwd: the eager composition rounds intermediate products
/// (`x*cos`, `rotate_half(x)*sin`, the sum) to bf16 at each op boundary;
/// the fused kernel accumulates the whole elementwise expression in f32
/// and rounds ONCE. A measured, non-vacuous divergence — NOT the f32
/// case above, where the rounding model predicts (and the measurement
/// confirms) bit-exactness; bf16's own per-op rounding genuinely differs
/// between the two paths, so a tolerance is the honest assertion here.
///
/// `BF16_ULP_TOL = 4` is an INHERITED-CONSERVATIVE ceiling (the same
/// value and the same status `LayerNormFused`'s identical constant has,
/// not independently re-derived here): both fwd and bwd tests below
/// PRINT their own measured maximum unconditionally (`println!`), and on
/// these fixtures that measured max is smaller than `4` — the constant is
/// a deliberate headroom margin over the measured value, sized to
/// tolerate a worse fixture or a different libm's rounding on another
/// platform without flaking, not a number tightened to today's
/// measurement.
const BF16_ULP_TOL: i32 = 4;

fn bf16_bit_diff(a: bf16, b: bf16) -> i32 {
    a.to_bits() as i32 - b.to_bits() as i32
}

#[test]
fn eager_vs_fused_bf16_fwd_diverges_and_stays_within_the_stated_ulp_tolerance() {
    let device = Device::Cpu;
    let (batch, heads, seq, hidden) = (1, 2, 4, 8);
    let n = batch * heads * seq * hidden;
    let x0: Vec<f32> = (0..n)
        .map(|i| [-18.5, -17.75, 9.0625, -2.5, 3.375, -4.125, 12.5, -6.25][i % 8])
        .collect();
    let (cos_v, sin_v) = table(seq, hidden, 10_000.0);
    let xb: Vec<bf16> = x0.iter().map(|&v| bf16::from_f32(v)).collect();
    let cb: Vec<bf16> = cos_v.iter().map(|&v| bf16::from_f32(v)).collect();
    let sb: Vec<bf16> = sin_v.iter().map(|&v| bf16::from_f32(v)).collect();

    let x = Tensor::from_slice(&xb, (batch, heads, seq, hidden), &device).unwrap();
    let cos = Tensor::from_slice(&cb, (1, 1, seq, hidden), &device).unwrap();
    let sin = Tensor::from_slice(&sb, (1, 1, seq, hidden), &device).unwrap();

    let fused_out: Vec<bf16> = fused(false, &x, &cos, &sin)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let eager_out: Vec<bf16> = eager(&x, &cos, &sin)
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
    let max_diff = diffs.iter().map(|d| d.abs()).max().unwrap_or(0);
    println!("eager_vs_fused_bf16_fwd: measured max |bit-diff| = {max_diff}");
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

/// Backward analog: `Tensor::backward()`'s ones-seed cancels bf16
/// rounding divergence (per the fused-kernels plan's C2 lesson — `1.0 *
/// anything` rounds identically regardless of dtype), so this weights the
/// output with a non-uniform, bf16-awkward vector before summing, making
/// the effective `dy` genuinely non-trivial.
#[test]
fn eager_vs_fused_bf16_bwd_diverges_and_stays_within_the_stated_ulp_tolerance() {
    let device = Device::Cpu;
    let (batch, heads, seq, hidden) = (1, 2, 4, 8);
    let n = batch * heads * seq * hidden;
    let x0: Vec<f32> = (0..n)
        .map(|i| [-18.5, -17.75, 9.0625, -2.5, 3.375, -4.125, 12.5, -6.25][i % 8])
        .collect();
    let w0: Vec<f32> = (0..n)
        .map(|i| [3.375, -4.125, 12.5, -6.25, -18.5, -17.75, 9.0625, -2.5][i % 8])
        .collect();
    let (cos_v, sin_v) = table(seq, hidden, 10_000.0);
    let xb: Vec<bf16> = x0.iter().map(|&v| bf16::from_f32(v)).collect();
    let wb: Vec<bf16> = w0.iter().map(|&v| bf16::from_f32(v)).collect();
    let cb: Vec<bf16> = cos_v.iter().map(|&v| bf16::from_f32(v)).collect();
    let sb: Vec<bf16> = sin_v.iter().map(|&v| bf16::from_f32(v)).collect();

    let x_f =
        Var::from_tensor(&Tensor::from_slice(&xb, (batch, heads, seq, hidden), &device).unwrap())
            .unwrap();
    let x_e =
        Var::from_tensor(&Tensor::from_slice(&xb, (batch, heads, seq, hidden), &device).unwrap())
            .unwrap();
    let cos = Tensor::from_slice(&cb, (1, 1, seq, hidden), &device).unwrap();
    let sin = Tensor::from_slice(&sb, (1, 1, seq, hidden), &device).unwrap();
    let w_f = Tensor::from_slice(&wb, (batch, heads, seq, hidden), &device).unwrap();
    let w_e = Tensor::from_slice(&wb, (batch, heads, seq, hidden), &device).unwrap();

    let out_f = fused(false, &x_f, &cos, &sin).unwrap();
    let out_e = eager(&x_e, &cos, &sin).unwrap();
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
    let diffs: Vec<i32> = dxf
        .iter()
        .zip(dxe.iter())
        .map(|(&f, &e)| bf16_bit_diff(f, e))
        .collect();
    let max_diff = diffs.iter().map(|d| d.abs()).max().unwrap_or(0);
    println!("eager_vs_fused_bf16_bwd: measured max |dx bit-diff| = {max_diff}");
    assert!(
        diffs.iter().any(|&d| d != 0),
        "expected dx fixture to diverge (measured diffs: {diffs:?}) — the \
         tolerance is not being exercised"
    );
    for (i, d) in diffs.iter().enumerate() {
        assert!(
            d.abs() <= BF16_ULP_TOL,
            "dx element {i}: bit diff {d} exceeds the stated {BF16_ULP_TOL}-ULP tolerance"
        );
    }
}

/// Chain-rule oracle: `x` is an INTERMEDIATE (`w.affine(2, 0)`) on a path
/// to a `Var` — `is_variable() == false`, the exact hazard `Axpy`'s own
/// regression test exercises — at the real 4D call-site shape.
#[test]
fn chain_rule_through_an_intermediate_x() {
    let device = Device::Cpu;
    let (batch, heads, seq, hidden) = (2, 2, 3, 8);
    let n = batch * heads * seq * hidden;
    let w0: Vec<f32> = (0..n).map(|i| (i as f32 * 0.23 - 1.5).sin()).collect();
    let (cos_v, sin_v) = table(seq, hidden, 500.0);

    let w =
        Var::from_tensor(&Tensor::from_slice(&w0, (batch, heads, seq, hidden), &device).unwrap())
            .unwrap();
    let x = w.affine(2.0, 0.0).unwrap();
    assert!(
        !x.is_variable(),
        "x must be the is_variable()==false case under test"
    );
    let cos = Tensor::from_slice(&cos_v, (1, 1, seq, hidden), &device).unwrap();
    let sin = Tensor::from_slice(&sin_v, (1, 1, seq, hidden), &device).unwrap();

    let out = fused(false, &x, &cos, &sin).unwrap();
    let grads = out.backward().unwrap(); // must not panic
    let dw: Vec<f32> = grads
        .get(&w)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let x2: Vec<f32> = w0.iter().map(|&v| 2.0 * v).collect();
    let x_direct =
        Var::from_tensor(&Tensor::from_slice(&x2, (batch, heads, seq, hidden), &device).unwrap())
            .unwrap();
    let out2 = fused(false, &x_direct, &cos, &sin).unwrap();
    let grads2 = out2.backward().unwrap();
    let dx: Vec<f32> = grads2
        .get(&x_direct)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    for (i, (a, b)) in dw.iter().zip(dx.iter()).enumerate() {
        assert!((a - 2.0 * b).abs() < 1e-4, "dw[{i}]: {a} vs 2*{b}");
    }
}

/// The negated-sin bwd identity, at a hand-computable 4-element case
/// (`period = 1`): `bwd` must equal applying THIS op with `(cos, -sin)`
/// to `dy` — a readable failure if the sign convention (which half of
/// `rotate_half` gets the negation) is ever flipped by accident,
/// independent of `ops::rope`'s own internal unit test of the same fact.
#[test]
fn negated_sin_bwd_matches_a_hand_computed_sign_flip() {
    let device = Device::Cpu;
    let c = 0.6f32;
    let s = 0.8f32; // c^2+s^2=1, a genuine rotation angle
    let x0 = [1.0f32, 2.0, 3.0, 4.0];
    let x = Var::from_tensor(&Tensor::from_slice(&x0, (1, 1, 1, 4), &device).unwrap()).unwrap();
    let cos = Tensor::from_slice(&[c, c, c, c], (1, 1, 1, 4), &device).unwrap();
    let sin = Tensor::from_slice(&[s, s, s, s], (1, 1, 1, 4), &device).unwrap();

    // Non-uniform loss weight so `Tensor::backward()`'s ones-seed does not
    // trivially collapse the identity being tested.
    let w0 = [0.3f32, -1.7, 2.2, 0.5];
    let w = Tensor::from_slice(&w0, (1, 1, 1, 4), &device).unwrap();

    let out = fused(false, &x, &cos, &sin).unwrap();
    let loss = (&out * &w).unwrap().sum_all().unwrap();
    let grads = loss.backward().unwrap();
    let dx: Vec<f32> = grads
        .get(&x)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    // dy (before the elementwise mul by w, i.e. the gradient at `out`) is
    // exactly `w` here (d(sum(out*w))/d(out) = w). The identity: dx =
    // rope_fwd(dy, cos, -sin).
    let neg_sin = (&sin * -1.0f64).unwrap();
    let dy = w.clone();
    let expected: Vec<f32> = fused(false, &dy, &cos, &neg_sin)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    for (i, (a, b)) in dx.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "sign-convention mismatch at dx[{i}]: bwd gave {a}, rope_fwd(dy, cos, -sin) gives {b} \
             — the rotate-half negation half may have flipped"
        );
    }

    // And the SAME identity is exposed directly via `negate_sin=true`.
    let via_negate_sin: Vec<f32> = fused(true, &dy, &cos, &sin)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    for (i, (a, b)) in expected.iter().zip(via_negate_sin.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "negate_sin=true must equal rope_fwd(dy, cos, -sin) at [{i}]: {a} vs {b}"
        );
    }
}

/// esc-045 round 5, E1: `x`/`grad_res` at PRODUCTION amplitude (guide
/// §3.4) — `hidden = 64` (ModernBERT-large's real `head_dim`), `seq =
/// 128`, with two coupled "massive activation" channel pairs (`5`/`37`,
/// `21`/`53` — both `col` and `col+half`, so the rotate-half coupling this
/// op's math depends on is exercised at the extreme amplitude, not just a
/// single stray channel) forced to `|x| ~ 6688`, matching round 4's
/// measured `max|qkv| ~ 6688` at the layer-18 massive-activation onset
/// (`5d3716a`'s commit message: "one layer past the |x|~6688 massive-
/// activation onset at layer 18"). Every other channel stays at the
/// moderate `O(1)`-`O(10)` amplitude this file's other fixtures already
/// cover. `grad_res` reuses the same massive-channel pattern with an
/// independent salt (a downstream gradient that has itself passed through
/// amplified paths is not a moderate one; it is not literally `x` reused).
fn massive_amplitude_fixture(total_rows: usize, hidden: usize, salt: f32) -> Vec<f32> {
    let massive_cols = [5usize, 37, 21, 53];
    (0..total_rows * hidden)
        .map(|idx| {
            let row = idx / hidden;
            let col = idx % hidden;
            let base = ((row * hidden + col) as f32 * 0.173 + salt).sin() * 8.0;
            if massive_cols.contains(&col) {
                let sign = if (row + col).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                };
                sign * (6688.0 + base)
            } else {
                base
            }
        })
        .collect()
}

/// Independent numpy/f64-first TRUTH (family F) for the forward pass —
/// the raw per-column rotation algebra derived directly from `out_j =
/// x_j*cos_j + rotate_half(x)_j*sin_j`, NOT a call through `fused`/`eager`
/// at any dtype, so this oracle cannot share a bug with either arm under
/// test. `theta` is recomputed from `(seq_idx, theta_base)` rather than
/// read back from the f32 `cos`/`sin` tables the arms under test consume,
/// for the same reason.
fn truth_fwd_f64(
    x: &[f32],
    theta_base: f64,
    total_rows: usize,
    period: usize,
    hidden: usize,
) -> Vec<f64> {
    let half = hidden / 2;
    let mut out = vec![0f64; total_rows * hidden];
    for r in 0..total_rows {
        let seq_idx = r % period;
        for col in 0..hidden {
            let i = col % half;
            let theta = (seq_idx as f64) * theta_base.powf(-2.0 * i as f64 / hidden as f64);
            let (c, s) = (theta.cos(), theta.sin());
            let xv = x[r * hidden + col] as f64;
            let rh = if col < half {
                -(x[r * hidden + col + half] as f64)
            } else {
                x[r * hidden + col - half] as f64
            };
            out[r * hidden + col] = xv * c + rh * s;
        }
    }
    out
}

/// Independent numpy/f64-first TRUTH for `dx`, derived by hand from the
/// module doc's Jacobian-transpose identity (`ops::rope`'s "`bwd`: RoPE
/// with the sign of `sin` flipped" section): for `col < half`, `dx_col =
/// dy_col*cos_col + dy_{col+half}*sin_col`; for `col >= half`, `dx_col =
/// dy_col*cos_col - dy_{col-half}*sin_col` — worked from `out_col`'s own
/// dependence on `x_col` AND `x_{col +/- half}` (each column of `x`
/// contributes to exactly two output columns under this op's pairing), not
/// by calling `fused`'s `bwd` or `eager`'s autograd.
fn truth_bwd_f64(
    dy: &[f32],
    theta_base: f64,
    total_rows: usize,
    period: usize,
    hidden: usize,
) -> Vec<f64> {
    let half = hidden / 2;
    let mut out = vec![0f64; total_rows * hidden];
    for r in 0..total_rows {
        let seq_idx = r % period;
        for col in 0..hidden {
            let i = col % half;
            let theta = (seq_idx as f64) * theta_base.powf(-2.0 * i as f64 / hidden as f64);
            let (c, s) = (theta.cos(), theta.sin());
            let dyv = dy[r * hidden + col] as f64;
            let (pair, sign) = if col < half {
                (col + half, 1.0)
            } else {
                (col - half, -1.0)
            };
            let dy_pair = dy[r * hidden + pair] as f64;
            out[r * hidden + col] = dyv * c + sign * dy_pair * s;
        }
    }
    out
}

/// `(L1 relative error, L2 norm ratio)` of a `bf16` arm against an `f64`
/// truth vector — ascending-index fold order (family J), the same fixed
/// order the op's own module doc pins for its kernel math.
fn relerr_and_norm_ratio(measured: &[bf16], truth: &[f64]) -> (f64, f64) {
    let mut abs_diff_sum = 0f64;
    let mut truth_abs_sum = 0f64;
    let mut measured_sq_sum = 0f64;
    let mut truth_sq_sum = 0f64;
    for (&m, &t) in measured.iter().zip(truth.iter()) {
        let mf = m.to_f32() as f64;
        abs_diff_sum += (mf - t).abs();
        truth_abs_sum += t.abs();
        measured_sq_sum += mf * mf;
        truth_sq_sum += t * t;
    }
    let relerr = abs_diff_sum / truth_abs_sum;
    let norm_ratio = measured_sq_sum.sqrt() / truth_sq_sum.sqrt();
    (relerr, norm_ratio)
}

/// esc-045 round 5, E1 (decides H1 vs H2 — see the round-5 dispatch): at
/// PRODUCTION amplitude (`massive_amplitude_fixture`), is the FUSED bf16
/// kernel itself further from an independent f64 truth than the EAGER
/// candle-op composition is, for fwd AND for `dx` separately? If fused's
/// relerr is comparable to eager's (both near bf16's own precision floor,
/// ~0.4%), the isolated op is not the defect (H2: the divergence measured
/// against jammi's own eager arm in round 4 must be introduced or
/// amplified downstream of this op, not inside it). If fused's relerr is
/// ORDERS OF MAGNITUDE larger than eager's, that is H1 — a real kernel
/// defect at this amplitude, not a rounding-model artifact.
#[test]
fn eager_vs_fused_bf16_fwd_and_bwd_at_production_amplitude_vs_independent_f64_truth() {
    let device = Device::Cpu;
    let (heads, seq, hidden) = (4usize, 128usize, 64usize);
    let total_rows = heads * seq;
    let theta_base = 10_000.0f64;

    let x0 = massive_amplitude_fixture(total_rows, hidden, 0.0);
    let dy0 = massive_amplitude_fixture(total_rows, hidden, 100.0);
    assert!(
        x0.iter().any(|&v| v.abs() > 6000.0),
        "fixture must actually reach production amplitude"
    );

    let (cos_v, sin_v) = table(seq, hidden, theta_base);
    let xb: Vec<bf16> = x0.iter().map(|&v| bf16::from_f32(v)).collect();
    let cb: Vec<bf16> = cos_v.iter().map(|&v| bf16::from_f32(v)).collect();
    let sb: Vec<bf16> = sin_v.iter().map(|&v| bf16::from_f32(v)).collect();
    let dyb: Vec<bf16> = dy0.iter().map(|&v| bf16::from_f32(v)).collect();

    let x_fused =
        Var::from_tensor(&Tensor::from_slice(&xb, (1, heads, seq, hidden), &device).unwrap())
            .unwrap();
    let x_eager =
        Var::from_tensor(&Tensor::from_slice(&xb, (1, heads, seq, hidden), &device).unwrap())
            .unwrap();
    let cos_bf = Tensor::from_slice(&cb, (1, 1, seq, hidden), &device).unwrap();
    let sin_bf = Tensor::from_slice(&sb, (1, 1, seq, hidden), &device).unwrap();

    let out_fused = fused(false, &x_fused, &cos_bf, &sin_bf).unwrap();
    let out_eager = eager(&x_eager, &cos_bf, &sin_bf).unwrap();
    let fwd_fused: Vec<bf16> = out_fused.flatten_all().unwrap().to_vec1().unwrap();
    let fwd_eager: Vec<bf16> = out_eager.flatten_all().unwrap().to_vec1().unwrap();

    let truth_fwd = truth_fwd_f64(&x0, theta_base, total_rows, seq, hidden);
    let (fused_relerr_fwd, fused_norm_ratio_fwd) = relerr_and_norm_ratio(&fwd_fused, &truth_fwd);
    let (eager_relerr_fwd, eager_norm_ratio_fwd) = relerr_and_norm_ratio(&fwd_eager, &truth_fwd);
    println!(
        "[E1 fwd] fused: relerr={fused_relerr_fwd:.6e} norm_ratio={fused_norm_ratio_fwd:.6} | \
         eager: relerr={eager_relerr_fwd:.6e} norm_ratio={eager_norm_ratio_fwd:.6}"
    );

    let w_fused = Tensor::from_slice(&dyb, (1, heads, seq, hidden), &device).unwrap();
    let w_eager = Tensor::from_slice(&dyb, (1, heads, seq, hidden), &device).unwrap();
    let loss_fused = (&out_fused * &w_fused).unwrap().sum_all().unwrap();
    let loss_eager = (&out_eager * &w_eager).unwrap().sum_all().unwrap();
    let dx_fused: Vec<bf16> = loss_fused
        .backward()
        .unwrap()
        .get(&x_fused)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let dx_eager: Vec<bf16> = loss_eager
        .backward()
        .unwrap()
        .get(&x_eager)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let truth_dx = truth_bwd_f64(&dy0, theta_base, total_rows, seq, hidden);
    let (fused_relerr_bwd, fused_norm_ratio_bwd) = relerr_and_norm_ratio(&dx_fused, &truth_dx);
    let (eager_relerr_bwd, eager_norm_ratio_bwd) = relerr_and_norm_ratio(&dx_eager, &truth_dx);
    println!(
        "[E1 bwd] fused: relerr={fused_relerr_bwd:.6e} norm_ratio={fused_norm_ratio_bwd:.6} | \
         eager: relerr={eager_relerr_bwd:.6e} norm_ratio={eager_norm_ratio_bwd:.6}"
    );

    // Non-finite negative control (§3.7): a NaN massive-activation channel
    // must propagate to a NaN output, never silently compare as passing.
    let mut x_nan = x0.clone();
    x_nan[5] = f32::NAN;
    let xb_nan: Vec<bf16> = x_nan.iter().map(|&v| bf16::from_f32(v)).collect();
    let x_nan_t = Tensor::from_slice(&xb_nan, (1, heads, seq, hidden), &device).unwrap();
    let out_nan: Vec<bf16> = fused(false, &x_nan_t, &cos_bf, &sin_bf)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(
        out_nan[5].is_nan(),
        "NaN in a massive-activation channel must produce a NaN output"
    );

    // §3.3: agreement is not accuracy — anchor both arms to the
    // independent f64 truth and require the fused arm to be NO FURTHER
    // from it than the eager arm, within a small factor covering ordinary
    // fixture-to-fixture noise. `1e-2` absolute floor covers the case
    // where eager's relerr measures near-zero (guards a 0/near-0
    // denominator blowing the ratio up spuriously).
    let fwd_ratio_ok = fused_relerr_fwd <= (2.0 * eager_relerr_fwd).max(fused_relerr_fwd.min(1e-2));
    let bwd_ratio_ok = fused_relerr_bwd <= (2.0 * eager_relerr_bwd).max(fused_relerr_bwd.min(1e-2));
    assert!(
        fused_relerr_fwd.is_finite() && fwd_ratio_ok,
        "fwd: fused arm ({fused_relerr_fwd:.6e}) is materially further from f64 truth than the \
         eager arm ({eager_relerr_fwd:.6e}) at production amplitude — this is H1, a real kernel \
         defect, not ordinary bf16 rounding"
    );
    assert!(
        fused_relerr_bwd.is_finite() && bwd_ratio_ok,
        "bwd: fused arm ({fused_relerr_bwd:.6e}) is materially further from f64 truth than the \
         eager arm ({eager_relerr_bwd:.6e}) at production amplitude — this is H1, a real kernel \
         defect, not ordinary bf16 rounding"
    );
    // Both arms must also be within an ordinary bf16-precision band of
    // truth in absolute terms (not just relative to each other) — bf16's
    // machine epsilon is 2^-8 ~ 0.39%; 2% is a deliberate headroom margin
    // (not tightened to the measured minimum) over that, the same
    // inherited-conservative-ceiling status `BF16_ULP_TOL` documents.
    assert!(
        fused_relerr_fwd < 0.02 && eager_relerr_fwd < 0.02,
        "fwd relerr exceeds the bf16-precision band even before comparing the two arms: \
         fused={fused_relerr_fwd:.6e} eager={eager_relerr_fwd:.6e}"
    );
    assert!(
        fused_relerr_bwd < 0.02 && eager_relerr_bwd < 0.02,
        "bwd relerr exceeds the bf16-precision band even before comparing the two arms: \
         fused={fused_relerr_bwd:.6e} eager={eager_relerr_bwd:.6e}"
    );
}

/// esc-045 round 5, E1's RED control — pinned here as an automated
/// assertion. The round-5 cuda-runs artifact's own `red_control` field
/// (`.jammi/escapes.jsonl`'s esc-045 row) reports it as a MANUAL,
/// un-committed check: "Negating the rotate-half sign convention in
/// `rope_fwd_row_bf16` (reverted after the check) drove fwd relerr to
/// 4.183690e-1 — confirms the oracle discriminates a real defect, not a
/// vacuous pass." A duplicated reference (never a runtime hook into the
/// production kernel: `RopeFused::negate_sin` is a DIFFERENT, already
/// production-legitimate flag — it flips the WHOLE `sin` term's sign for
/// `bwd`'s reuse of the fwd kernel, not which half of `rotate_half` gets
/// negated — see `ops::rope`'s module doc) that swaps `rotate_half`'s
/// sign convention (`[x2, -x1]` in place of the correct `[-x2, x1]`) on
/// the SAME production-amplitude fixture, at bf16 precision, measured
/// against the SAME independent f64 truth
/// [`eager_vs_fused_bf16_fwd_and_bwd_at_production_amplitude_vs_independent_f64_truth`]
/// anchors to.
fn buggy_wrong_rotate_half_sign_fwd_bf16(
    x: &[f32],
    theta_base: f64,
    total_rows: usize,
    period: usize,
    hidden: usize,
) -> Vec<bf16> {
    let half = hidden / 2;
    let mut out = vec![bf16::ZERO; total_rows * hidden];
    for r in 0..total_rows {
        let seq_idx = r % period;
        for col in 0..hidden {
            let i = col % half;
            let theta = (seq_idx as f64) * theta_base.powf(-2.0 * i as f64 / hidden as f64);
            let (c, s) = (theta.cos() as f32, theta.sin() as f32);
            let xv = x[r * hidden + col];
            // WRONG sign convention — the exact reverse of `truth_fwd_f64`'s
            // `rh` (which mirrors the real `rotate_half`'s `[-x2, x1]`
            // pairing): `col < half` reads the POSITIVE partner and
            // `col >= half` reads the NEGATED one.
            let rh = if col < half {
                x[r * hidden + col + half]
            } else {
                -x[r * hidden + col - half]
            };
            out[r * hidden + col] = bf16::from_f32(xv * c + rh * s);
        }
    }
    out
}

#[test]
fn wrong_rotate_half_sign_convention_is_caught_relerr_over_0_1() {
    let (heads, seq, hidden) = (4usize, 128usize, 64usize);
    let total_rows = heads * seq;
    let theta_base = 10_000.0f64;

    let x0 = massive_amplitude_fixture(total_rows, hidden, 0.0);
    assert!(
        x0.iter().any(|&v| v.abs() > 6000.0),
        "fixture must actually reach production amplitude"
    );
    let truth_fwd = truth_fwd_f64(&x0, theta_base, total_rows, seq, hidden);
    let buggy = buggy_wrong_rotate_half_sign_fwd_bf16(&x0, theta_base, total_rows, seq, hidden);

    // Finiteness-affirmative (guide §3.7) before the discriminating compare.
    for (i, v) in buggy.iter().enumerate() {
        assert!(
            v.to_f32().is_finite(),
            "index {i}: a non-finite value slipped through the buggy reference (v={v:?})"
        );
    }

    let (relerr, _norm_ratio) = relerr_and_norm_ratio(&buggy, &truth_fwd);
    println!("[E1 RED control] wrong-sign relerr={relerr:.6e}");
    assert!(
        relerr.is_finite() && relerr > 0.1,
        "RED CONTROL did not fire: a wrong rotate_half sign convention must diverge sharply \
         from the independent f64 truth (round 5's manual check measured 4.18e-1; `> 0.1` pins \
         that this oracle's fixture and tolerance band actually discriminate a real \
         sign-convention defect, not vacuously agree with everything) — got relerr={relerr:.6e}"
    );
}

#[test]
fn f32_dtype_forward_matches_double_precision_reference() {
    // A numpy/f64-first reference at a hand-verifiable head_dim=2 (a
    // single rotated pair), asserting the op against an independently
    // computed value rather than merely against itself.
    let device = Device::Cpu;
    let theta = 0.7f64;
    let (c64, s64) = (theta.cos(), theta.sin());
    let x0 = [2.0f64, -1.0];
    let expected = [
        (x0[0] * c64 - x0[1] * s64) as f32,
        (x0[1] * c64 + x0[0] * s64) as f32,
    ];
    let x = Tensor::from_slice(&[x0[0] as f32, x0[1] as f32], (1, 2), &device).unwrap();
    let cos = Tensor::from_slice(&[c64 as f32, c64 as f32], (1, 2), &device).unwrap();
    let sin = Tensor::from_slice(&[s64 as f32, s64 as f32], (1, 2), &device).unwrap();
    let out: Vec<f32> = fused(false, &x, &cos, &sin)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    for (o, e) in out.iter().zip(expected.iter()) {
        assert!((o - e).abs() < 1e-6, "{o} vs {e}");
    }
    // Non-finite negative control: NaN must propagate, not silently
    // compare as "passing" (`NaN > c` is false — a naive comparison would
    // let this through).
    let x_nan = Tensor::from_slice(&[f32::NAN, -1.0f32], (1, 2), &device).unwrap();
    let out_nan: Vec<f32> = fused(false, &x_nan, &cos, &sin)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(
        out_nan[0].is_nan(),
        "NaN input must produce a NaN output, not a silently-finite wrong number"
    );
}
