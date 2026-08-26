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
//!      predicts real divergence, never as a default hedge. BF16 is ALSO
//!      asserted BIT-EXACT since the RoPE one-rounding fix (`eager()` now
//!      upcasts every operand to f32 before the rotation and rounds once
//!      at the end, the same shape `RopeFused` itself computes) — measured
//!      on these fixtures; see the bf16 tests' own docs for why that is
//!      not a structural guarantee at every shape.
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

use candle_core::{DType, Device, Tensor, Var, D};
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
/// this math, only where the cast/unsqueeze happen once). Matches HF's
/// `apply_rotary_pos_emb` (`q.float() * cos + rotate_half(q.float()) *
/// sin`, then `.to(original_dtype)` once): every operand is upcast to
/// `internal_dtype` (f32 whenever the input dtype is F16/BF16) before the
/// multiply/add, and the result is cast to the input dtype exactly once
/// at the end — the same one-rounding shape `RopeFused` itself already
/// computes (`cuda/rope.cu:62`). For F32 inputs `internal_dtype == F32`,
/// so every cast here is a same-dtype no-op and this degenerates to the
/// original (already bit-exact) composition.
fn eager(x: &Tensor, cos_b: &Tensor, sin_b: &Tensor) -> candle_core::Result<Tensor> {
    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let head_dim = x.dim(D::Minus1)?;
    let half = head_dim / 2;
    let x_internal = x.to_dtype(internal_dtype)?;
    let cos_internal = cos_b.to_dtype(internal_dtype)?;
    let sin_internal = sin_b.to_dtype(internal_dtype)?;
    let x1 = x_internal.narrow(D::Minus1, 0, half)?;
    let x2 = x_internal.narrow(D::Minus1, half, half)?;
    let neg_x2 = (x2 * -1.0f64)?;
    let rot_half = Tensor::cat(&[&neg_x2, &x1], D::Minus1)?;
    let cos_part = x_internal.broadcast_mul(&cos_internal)?;
    let sin_part = rot_half.broadcast_mul(&sin_internal)?;
    let out_internal = (cos_part + sin_part)?;
    out_internal.to_dtype(x_dtype)?.contiguous()
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

/// bf16 fwd: BEFORE the one-rounding fix, the eager composition rounded
/// intermediate products (`x*cos`, `rotate_half(x)*sin`, the sum) to
/// bf16 at each op boundary while the fused kernel accumulated the whole
/// elementwise expression in f32 and rounded ONCE — measured on this
/// same fixture, that was a genuine, non-vacuous divergence. `eager()`
/// above now runs the IDENTICAL one-rounding shape (every operand
/// upcast to f32, one cast back at the end), so eager and fused are
/// expected to — and on this fixture, measured to — agree BIT-EXACTLY,
/// the same outcome the LN oracle's identical fix produced. `bf16_bit_diff`
/// is kept for the same reason `layer_norm_oracles.rs` keeps its own copy:
/// it is the tool a future regression (e.g. a production-`head_dim`-sized
/// fixture exposing a reduction-order difference) would use to re-derive
/// a tolerance.
fn bf16_bit_diff(a: bf16, b: bf16) -> i32 {
    a.to_bits() as i32 - b.to_bits() as i32
}

#[test]
fn eager_vs_fused_bf16_fwd_is_bit_exact_after_the_one_rounding_fix() {
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
    println!("eager_vs_fused_bf16_fwd: measured bit-diffs (post-fix) = {diffs:?}");
    assert_eq!(
        fused_out, eager_out,
        "fwd must now be bit-exact (measured diffs: {diffs:?}) — the pre-fix defect \
         (23% of elements diverging, see `RotaryEmbedding::apply`'s doc) is gone"
    );
}

/// Backward analog: `Tensor::backward()`'s ones-seed cancels bf16
/// rounding divergence (per the fused-kernels plan's C2 lesson — `1.0 *
/// anything` rounds identically regardless of dtype), so this weights the
/// output with a non-uniform, bf16-awkward vector before summing, making
/// the effective `dy` genuinely non-trivial. Measured on this fixture,
/// post-fix, `dx` is bit-exact — see the forward oracle's doc for why
/// that is not a structural guarantee at every shape.
#[test]
fn eager_vs_fused_bf16_bwd_is_bit_exact_after_the_one_rounding_fix() {
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
    println!("eager_vs_fused_bf16_bwd: measured dx bit-diffs (post-fix) = {diffs:?}");
    assert_eq!(
        dxf, dxe,
        "dx must now be bit-exact (measured diffs: {diffs:?})"
    );
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
