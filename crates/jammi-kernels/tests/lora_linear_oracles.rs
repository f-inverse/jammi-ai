//! `LowRankResidualLinear` production-width oracles: CPU-hermetic, `F32`
//! (candle-core 0.11.0's CPU backend has no `BF16` matmul without
//! `mkl`/`accelerate` — see `ops::low_rank_residual_linear`'s module doc; the `BF16`
//! leg lives in `cuda_parity.rs`, pod-run).
//!
//! `crates/jammi-kernels/src/ops/lora_linear.rs`'s own `#[cfg(test)]`
//! module already covers the kernel's small-dimension math (closed-form
//! reference, bit-exact-vs-manual-composition, gradcheck, domain
//! refusals). This file adds what that module deliberately does NOT: the
//! shapes the #352 profile actually measured
//! (`[24, 128, 1024] -> {3072, 5248}`, ModernBERT-large's Wqkv/Wo/Wi
//! widths) run end to end against the CURRENT eager composition
//! (`LoraLinear::forward`'s training-arm math, reproduced here with plain
//! `Tensor` ops so this file needs no `jammi-lora` dependency — a leaf
//! crate names no consumer, family L), plus a boundary set (empty rows,
//! a single row, a rank-2 pooled head) family D calls for.
//!
//! ## The oracle contract, honestly stated
//!
//! An A100 pod run found two of these tests' `tol == 0.0` assertions
//! failing by exactly 1 `f32` ULP on x86/AVX-512 (they passed reliably on
//! Apple/NEON, where this suite was originally authored). The cause is
//! NOT a correctness bug: the fused op and the "eager" reference below
//! can legitimately hand `Tensor::matmul` DIFFERENT operand stride
//! patterns for the mathematically-identical GEMM, and `gemm`'s internal
//! kernel/blocking selection — and hence its floating-point SUMMATION
//! ORDER — can depend on that stride pattern and on the host architecture
//! (`rows == 1` additionally selects a gemv-style path). Floating-point
//! addition is not associative, so a different summation order can land
//! on a different (still correctly-rounded) result in the last bit. This
//! is expected GEMM behaviour, not a bug: NVIDIA's own cuBLAS docs
//! ("Results reproducibility") state plainly that different summation
//! orders — across library versions, hardware, or problem sizes — can
//! produce bitwise-different results. Bit-exactness across a GEMM was
//! never something an eager reference legitimately offered; only the
//! ROUNDING POINTS (the epilogue's cast, the base/rank GEMMs' final
//! store) are a real parity claim (see the bf16-boundary-rounding-parity
//! rule: parity is about WHERE a comparison rounds, not that every
//! intermediate reduction sums in the same order).
//!
//! This file therefore draws the line honestly rather than loosening
//! `tol` on the EXISTING (sine-valued) fixtures to paper over it:
//!
//! - **Bit-exact legs** (`tol == 0.0`) now use [`exact_fixture`]:
//!   small-integer `f32` values (`{-4, .., 4}`) chosen so every partial
//!   sum this op's GEMMs form stays a SMALL EXACT INTEGER, well under
//!   `f32`'s 24-bit mantissa's exact range at every width this file
//!   exercises (even ModernBERT-large's `in=1024`). An exact-integer sum
//!   is IDENTICAL regardless of which order it is added in — no rounding
//!   ever occurs, so the comparison is bit-exact on ANY architecture BY
//!   CONSTRUCTION, not by luck. Every `scale`/dropout-`p` combination on
//!   these legs is also chosen to be exactly representable in binary
//!   (`0.5`, `1.0`, `2.0`, …) so the epilogue's multiply introduces no
//!   rounding of its own either.
//! - **One random-valued leg** (production_width_wqkv, DUPLICATED with a
//!   realistic sine-valued fixture) asserts within a tolerance DERIVED
//!   from the standard `n`-term `f32` dot-product reordering-error bound
//!   (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
//!   Theorem 4.2: a recursively-summed `n`-term sum deviates from the
//!   true sum by at most `(n-1) * u * sum(|x_i|)`, `u = 2^-24`; two
//!   different valid summation orders can therefore differ from EACH
//!   OTHER by up to twice that) — see [`derived_dot_product_tolerance`]'s
//!   doc for the exact formula, computed from the fixture's own KNOWN
//!   magnitude bound and this op's largest reduction depth, not tuned
//!   post hoc to make a failing assertion pass.
//! - The FD gradchecks (`production_width_backward_matches_candle_
//!   autograd_of_the_current_composition`) are unchanged: they were
//!   already a `1e-3` tolerance, never a `tol == 0.0` claim.

use candle_core::{DType, Device, Result, Tensor, Var};
use jammi_kernels::ops::{DropoutKey, LowRankResidualLinear};

/// The eager composition `LoraLinear::forward`'s training arm builds
/// today, reproduced directly over plain tensors (no dropout): `base_out
/// + scale * (x @ A^T @ B^T)`, `A` `[r, in]`, `B` `[out, r]`.
fn eager_forward(x: &Tensor, w: &Tensor, a: &Tensor, b: &Tensor, scale: f64) -> Result<Tensor> {
    let base_out = x.matmul(&w.t()?)?;
    let after_a = x.matmul(&a.t()?)?;
    let lora_out = after_a.matmul(&b.t()?)?;
    let scaled = (&lora_out * scale)?;
    &base_out + &scaled
}

/// Packs `a`/`b` into `ab`'s row-packed `[in + out, rank]` layout — see
/// `jammi_kernels::ops::low_rank_residual_linear`'s module doc, "the packed-`ab` GEMM
/// eligibility problem". `a.t()` is a non-contiguous VIEW; `Tensor::cat`'s
/// dim-0 path handles that via each arg's own `Layout` (no `.contiguous()`
/// call needed first — unlike the column-packed layout this replaced,
/// which needed one for `B^T`).
fn pack_ab(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    Tensor::cat(&[&a.t()?, b], 0)
}

fn fused_forward(x: &Tensor, w: &Tensor, ab: &Tensor, op: LowRankResidualLinear) -> Result<Tensor> {
    x.apply_op3(w, ab, op)
}

/// A fixed, deterministic (not `Tensor::randn`-seeded — this crate does
/// not depend on `rand`) f32 fixture, values in a modest range
/// (`|v| <= 0.3`). Realistic/non-integer — see the module doc's "oracle
/// contract" section for why this is now paired with a DERIVED tolerance
/// rather than `tol == 0.0`.
fn fixture(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (phase + i as f32 * 0.017).sin() * 0.3)
        .collect()
}

/// A fixed, deterministic, SMALL-INTEGER `f32` fixture (`{-4, .., 4}`),
/// used for every `tol == 0.0` (bit-exact) leg in this file — see the
/// module doc's "oracle contract" section for why exact-integer values
/// make bit-exactness architecture-independent rather than an accident of
/// one CPU's `gemm` kernel selection.
fn exact_fixture(n: usize, phase: i64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = (i as i64 * 7 + phase * 13).rem_euclid(9);
            (v - 4) as f32
        })
        .collect()
}

/// A defensible, DERIVED (not tuned-to-pass) absolute tolerance for an
/// `n`-term `f32` dot product compared across two possibly-different valid
/// summation orders — see the module doc's "oracle contract" section for
/// the Higham citation this pulls from. `max_term_magnitude` is the
/// caller-supplied bound on `|x_i * y_i|` for a single product term (known
/// from the fixture's own construction, e.g. `0.3 * 0.3` for two
/// `fixture()`-drawn factors); `safety_factor` covers (a) this being a
/// worst-case, not typical-case, bound, and (b) this op's LoRA branch
/// chaining TWO such reductions (`h = xd @ A^T` then `delta = h @ B^T`),
/// whose errors compound rather than being independent draws.
fn derived_dot_product_tolerance(n: usize, max_term_magnitude: f32, safety_factor: f32) -> f32 {
    let u = 2f32.powi(-24); // f32 unit roundoff.
    safety_factor * 2.0 * (n as f32).powi(2) * u * max_term_magnitude
}

fn assert_close(got: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        assert!(
            diff <= tol,
            "{label}[{i}]: got {g} expected {e} diff {diff} (tol {tol})"
        );
    }
}

/// Wqkv-shaped: `in=1024, out=3072` (3x hidden, the fused QKV projection),
/// `B_eff*S = 24*128 = 3072` rows, `rank=16` — the #352 profile's own
/// numbers.
#[test]
fn production_width_wqkv_forward_matches_the_eager_composition() {
    let device = Device::Cpu;
    let (rows, inf, outf, r) = (24 * 128, 1024usize, 3072usize, 16usize);
    let scale = 8.0 / (r as f64);

    let x = Tensor::from_slice(&exact_fixture(rows * inf, 1), (rows, inf), &device).unwrap();
    let w = Tensor::from_slice(&exact_fixture(outf * inf, 2), (outf, inf), &device).unwrap();
    let a = Tensor::from_slice(&exact_fixture(r * inf, 3), (r, inf), &device).unwrap();
    let b = Tensor::from_slice(&exact_fixture(outf * r, 4), (outf, r), &device).unwrap();
    let ab = pack_ab(&a, &b).unwrap();

    let op = LowRankResidualLinear::new(scale as f32, inf, outf, r, None, false).unwrap();
    let fused = fused_forward(&x, &w, &ab, op).unwrap();
    let eager = eager_forward(&x, &w, &a, &b, scale).unwrap();

    let fused_v: Vec<f32> = fused.flatten_all().unwrap().to_vec1().unwrap();
    let eager_v: Vec<f32> = eager.flatten_all().unwrap().to_vec1().unwrap();
    assert_close(&fused_v, &eager_v, 0.0, "wqkv_forward");
}

/// The honest, random-valued counterpart to the exact-integer leg above:
/// realistic (non-integer) production values via [`fixture`], compared
/// within a tolerance DERIVED from the `n`-term `f32` dot-product
/// reordering-error bound — see the module doc's "oracle contract"
/// section. This is what proves the crate has NOT simply swept precision
/// under the rug by only ever testing exact-integer inputs.
#[test]
fn production_width_wqkv_forward_matches_the_eager_composition_within_a_derived_ulp_bound() {
    let device = Device::Cpu;
    let (rows, inf, outf, r) = (24 * 128, 1024usize, 3072usize, 16usize);
    let scale = 8.0 / (r as f64);

    let x = Tensor::from_slice(&fixture(rows * inf, 0.15), (rows, inf), &device).unwrap();
    let w = Tensor::from_slice(&fixture(outf * inf, 0.25), (outf, inf), &device).unwrap();
    let a = Tensor::from_slice(&fixture(r * inf, 0.35), (r, inf), &device).unwrap();
    let b = Tensor::from_slice(&fixture(outf * r, 0.45), (outf, r), &device).unwrap();
    let ab = pack_ab(&a, &b).unwrap();

    let op = LowRankResidualLinear::new(scale as f32, inf, outf, r, None, false).unwrap();
    let fused = fused_forward(&x, &w, &ab, op).unwrap();
    let eager = eager_forward(&x, &w, &a, &b, scale).unwrap();

    // `fixture()` bounds every element by 0.3, so a two-factor product term
    // is bounded by 0.3*0.3 = 0.09; `inf` (1024) is the largest reduction
    // depth in this op's GEMM chain; `safety_factor = 4.0` covers the
    // two-hop LoRA branch's compounded rounding (see
    // `derived_dot_product_tolerance`'s doc).
    let tol = derived_dot_product_tolerance(inf, 0.09, 4.0);
    let fused_v: Vec<f32> = fused.flatten_all().unwrap().to_vec1().unwrap();
    let eager_v: Vec<f32> = eager.flatten_all().unwrap().to_vec1().unwrap();
    assert_close(&fused_v, &eager_v, tol, "wqkv_forward_ulp_bound");
}

/// Wo-shaped: `in=3072/? -> here in=1024,out=1024` GeGLU's `Wi` (packed,
/// `out=5248` for ModernBERT-large's intermediate*2) is the SECOND shape
/// the #352 profile names; exercised here at `in=1024, out=5248`.
#[test]
fn production_width_wi_forward_matches_the_eager_composition() {
    let device = Device::Cpu;
    let (rows, inf, outf, r) = (24 * 128, 1024usize, 5248usize, 16usize);
    let scale = 8.0 / (r as f64);

    let x = Tensor::from_slice(&exact_fixture(rows * inf, 5), (rows, inf), &device).unwrap();
    let w = Tensor::from_slice(&exact_fixture(outf * inf, 6), (outf, inf), &device).unwrap();
    let a = Tensor::from_slice(&exact_fixture(r * inf, 7), (r, inf), &device).unwrap();
    let b = Tensor::from_slice(&exact_fixture(outf * r, 8), (outf, r), &device).unwrap();
    let ab = pack_ab(&a, &b).unwrap();

    let op = LowRankResidualLinear::new(scale as f32, inf, outf, r, None, false).unwrap();
    let fused = fused_forward(&x, &w, &ab, op).unwrap();
    let eager = eager_forward(&x, &w, &a, &b, scale).unwrap();

    let fused_v: Vec<f32> = fused.flatten_all().unwrap().to_vec1().unwrap();
    let eager_v: Vec<f32> = eager.flatten_all().unwrap().to_vec1().unwrap();
    assert_close(&fused_v, &eager_v, 0.0, "wi_forward");
}

/// The rank-3 call shape (`[B, S, in]`, ModernBERT's real activation
/// shape — `LoraLinear::forward` never sees a pre-flattened `[B*S, in]`
/// tensor) with production-width `in`/`out`, at a smaller `B*S` so the
/// test stays fast; the flattening logic is dimension-count-generic (see
/// `LowRankResidualLinear::flatten_x`), so this is the shape-handling oracle,
/// not a second numeric one.
#[test]
fn production_width_rank3_forward_matches_the_eager_composition() {
    let device = Device::Cpu;
    let (b_dim, s_dim, inf, outf, r) = (4usize, 32usize, 1024usize, 3072usize, 16usize);
    let rows = b_dim * s_dim;
    let scale = 8.0 / (r as f64);

    let x =
        Tensor::from_slice(&exact_fixture(rows * inf, 11), (b_dim, s_dim, inf), &device).unwrap();
    let w = Tensor::from_slice(&exact_fixture(outf * inf, 12), (outf, inf), &device).unwrap();
    let a = Tensor::from_slice(&exact_fixture(r * inf, 13), (r, inf), &device).unwrap();
    let b = Tensor::from_slice(&exact_fixture(outf * r, 14), (outf, r), &device).unwrap();
    let ab = pack_ab(&a, &b).unwrap();

    let op = LowRankResidualLinear::new(scale as f32, inf, outf, r, None, false).unwrap();
    let fused = fused_forward(&x, &w, &ab, op).unwrap();
    assert_eq!(fused.dims(), &[b_dim, s_dim, outf]);

    let x_2d = x.reshape((rows, inf)).unwrap();
    let eager = eager_forward(&x_2d, &w, &a, &b, scale).unwrap();

    let fused_v: Vec<f32> = fused.flatten_all().unwrap().to_vec1().unwrap();
    let eager_v: Vec<f32> = eager.flatten_all().unwrap().to_vec1().unwrap();
    assert_close(&fused_v, &eager_v, 0.0, "rank3_forward");
}

/// Backward at production width, vs candle's own autograd walk of the
/// CURRENT eager composition — the node-count reduction this op exists
/// for must not change the GRADIENT VALUES, only the tape shape. Compares
/// `dx`/`dA`/`dB`/`dW` (with `dweight_needed=true`) against
/// `candle_core::Tensor::backward()`'s own result on an equivalent
/// (unfused) graph, with a NON-UNIFORM upstream gradient (`sum(out * dy)`
/// with `dy` a deterministic sine pattern, never all-ones).
#[test]
fn production_width_backward_matches_candle_autograd_of_the_current_composition() {
    let device = Device::Cpu;
    let (rows, inf, outf, r) = (256usize, 1024usize, 3072usize, 16usize);
    let scale = 8.0 / (r as f64);

    let x_v = fixture(rows * inf, 0.13);
    let w_v = fixture(outf * inf, 0.23);
    let a_v = fixture(r * inf, 0.33);
    let b_v = fixture(outf * r, 0.43);
    let dy_v: Vec<f32> = (0..rows * outf).map(|i| (i as f32 * 0.071).cos()).collect();

    // Fused side.
    let x_fused =
        Var::from_tensor(&Tensor::from_slice(&x_v, (rows, inf), &device).unwrap()).unwrap();
    let w_fused =
        Var::from_tensor(&Tensor::from_slice(&w_v, (outf, inf), &device).unwrap()).unwrap();
    let a_fused = Var::from_tensor(&Tensor::from_slice(&a_v, (r, inf), &device).unwrap()).unwrap();
    let b_fused = Var::from_tensor(&Tensor::from_slice(&b_v, (outf, r), &device).unwrap()).unwrap();
    let ab_fused = pack_ab(a_fused.as_tensor(), b_fused.as_tensor()).unwrap();
    let dy = Tensor::from_slice(&dy_v, (rows, outf), &device).unwrap();

    let op = LowRankResidualLinear::new(scale as f32, inf, outf, r, None, true).unwrap();
    let out_fused = x_fused
        .as_tensor()
        .apply_op3(w_fused.as_tensor(), &ab_fused, op)
        .unwrap();
    let loss_fused = (&out_fused * &dy).unwrap().sum_all().unwrap();
    let grads_fused = loss_fused.backward().unwrap();

    // Eager side: an independent set of leaves with the SAME values,
    // through candle's own autograd over the unfused composition.
    let x_eager =
        Var::from_tensor(&Tensor::from_slice(&x_v, (rows, inf), &device).unwrap()).unwrap();
    let w_eager =
        Var::from_tensor(&Tensor::from_slice(&w_v, (outf, inf), &device).unwrap()).unwrap();
    let a_eager = Var::from_tensor(&Tensor::from_slice(&a_v, (r, inf), &device).unwrap()).unwrap();
    let b_eager = Var::from_tensor(&Tensor::from_slice(&b_v, (outf, r), &device).unwrap()).unwrap();
    let out_eager = eager_forward(
        x_eager.as_tensor(),
        w_eager.as_tensor(),
        a_eager.as_tensor(),
        b_eager.as_tensor(),
        scale,
    )
    .unwrap();
    let loss_eager = (&out_eager * &dy).unwrap().sum_all().unwrap();
    let grads_eager = loss_eager.backward().unwrap();

    let tol = 1e-3f32;
    assert_close(
        &grads_fused
            .get(&x_fused)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        &grads_eager
            .get(&x_eager)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        tol,
        "dx",
    );
    assert_close(
        &grads_fused
            .get(&w_fused)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        &grads_eager
            .get(&w_eager)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        tol,
        "dw",
    );
    assert_close(
        &grads_fused
            .get(&a_fused)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        &grads_eager
            .get(&a_eager)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        tol,
        "da",
    );
    assert_close(
        &grads_fused
            .get(&b_fused)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        &grads_eager
            .get(&b_eager)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        tol,
        "db",
    );
}

/// The dropout-arm counterpart to
/// `production_width_backward_matches_candle_autograd_of_the_current_composition`:
/// SAME comparison, but with `dropout: Some(key)` on the fused side and
/// the identical `DropoutFused` key applied by hand to the LoRA branch's
/// input on the eager side (never the base — dropout only ever touches
/// the LoRA path, see the op's own module doc). Closes a real coverage
/// gap an adversarial mutation audit found: NEITHER an unmasked
/// `d_xd` in `bwd` (dropping the gradient re-mask entirely) NOR skipping
/// the dropout re-application in the recomputed `xd` moved this file's
/// prior (dropout-less) backward oracle — this test is what actually
/// exercises that code path's gradient correctness.
#[test]
fn production_width_backward_matches_candle_autograd_of_the_current_composition_with_dropout() {
    let device = Device::Cpu;
    let (rows, inf, outf, r) = (256usize, 1024usize, 3072usize, 16usize);
    let scale = 8.0 / (r as f64);
    let key = DropoutKey {
        seed: 555,
        layer_id: 9,
        forward_idx: 3,
        p: 0.2,
    };

    let x_v = fixture(rows * inf, 0.17);
    let w_v = fixture(outf * inf, 0.27);
    let a_v = fixture(r * inf, 0.37);
    let b_v = fixture(outf * r, 0.47);
    let dy_v: Vec<f32> = (0..rows * outf).map(|i| (i as f32 * 0.091).cos()).collect();

    // Fused side.
    let x_fused =
        Var::from_tensor(&Tensor::from_slice(&x_v, (rows, inf), &device).unwrap()).unwrap();
    let w_fused =
        Var::from_tensor(&Tensor::from_slice(&w_v, (outf, inf), &device).unwrap()).unwrap();
    let a_fused = Var::from_tensor(&Tensor::from_slice(&a_v, (r, inf), &device).unwrap()).unwrap();
    let b_fused = Var::from_tensor(&Tensor::from_slice(&b_v, (outf, r), &device).unwrap()).unwrap();
    let ab_fused = pack_ab(a_fused.as_tensor(), b_fused.as_tensor()).unwrap();
    let dy = Tensor::from_slice(&dy_v, (rows, outf), &device).unwrap();

    let op = LowRankResidualLinear::new(scale as f32, inf, outf, r, Some(key), true).unwrap();
    let out_fused = x_fused
        .as_tensor()
        .apply_op3(w_fused.as_tensor(), &ab_fused, op)
        .unwrap();
    let loss_fused = (&out_fused * &dy).unwrap().sum_all().unwrap();
    let grads_fused = loss_fused.backward().unwrap();

    // Eager side: an independent set of leaves with the SAME values, the
    // SAME dropout key applied by hand to the LoRA branch's input only,
    // through candle's own autograd over the unfused composition.
    let x_eager =
        Var::from_tensor(&Tensor::from_slice(&x_v, (rows, inf), &device).unwrap()).unwrap();
    let w_eager =
        Var::from_tensor(&Tensor::from_slice(&w_v, (outf, inf), &device).unwrap()).unwrap();
    let a_eager = Var::from_tensor(&Tensor::from_slice(&a_v, (r, inf), &device).unwrap()).unwrap();
    let b_eager = Var::from_tensor(&Tensor::from_slice(&b_v, (outf, r), &device).unwrap()).unwrap();
    let dropout_op =
        jammi_kernels::ops::DropoutFused::new(key.seed, key.layer_id, key.forward_idx, key.p)
            .unwrap();
    let xd_eager = jammi_kernels::ops::apply1(x_eager.as_tensor(), dropout_op).unwrap();
    let base_out = x_eager.as_tensor().matmul(&w_eager.t().unwrap()).unwrap();
    let after_a = xd_eager.matmul(&a_eager.t().unwrap()).unwrap();
    let lora_out = after_a.matmul(&b_eager.t().unwrap()).unwrap();
    let scaled = (&lora_out * scale).unwrap();
    let out_eager = (&base_out + &scaled).unwrap();
    let loss_eager = (&out_eager * &dy).unwrap().sum_all().unwrap();
    let grads_eager = loss_eager.backward().unwrap();

    let tol = 1e-3f32;
    assert_close(
        &grads_fused
            .get(&x_fused)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        &grads_eager
            .get(&x_eager)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        tol,
        "dx_with_dropout",
    );
    assert_close(
        &grads_fused
            .get(&w_fused)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        &grads_eager
            .get(&w_eager)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        tol,
        "dw_with_dropout",
    );
    assert_close(
        &grads_fused
            .get(&a_fused)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        &grads_eager
            .get(&a_eager)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        tol,
        "da_with_dropout",
    );
    assert_close(
        &grads_fused
            .get(&b_fused)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        &grads_eager
            .get(&b_eager)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        tol,
        "db_with_dropout",
    );
}

/// A rank-2 pooled classification head (`fine_tune/lora.rs`'s shape):
/// `in=1024` (a ModernBERT-large-width pooled vector), `out=2` (a binary
/// classification head), `rank=8`.
#[test]
fn rank2_pooled_classification_head_matches_the_eager_composition() {
    let device = Device::Cpu;
    let (rows, inf, outf, r) = (16usize, 1024usize, 2usize, 8usize);
    let scale = 8.0 / (r as f64);

    let x = Tensor::from_slice(&exact_fixture(rows * inf, 9), (rows, inf), &device).unwrap();
    let w = Tensor::from_slice(&exact_fixture(outf * inf, 11), (outf, inf), &device).unwrap();
    let a = Tensor::from_slice(&exact_fixture(r * inf, 13), (r, inf), &device).unwrap();
    let b = Tensor::from_slice(&exact_fixture(outf * r, 15), (outf, r), &device).unwrap();
    let ab = pack_ab(&a, &b).unwrap();

    let op = LowRankResidualLinear::new(scale as f32, inf, outf, r, None, false).unwrap();
    let fused = fused_forward(&x, &w, &ab, op).unwrap();
    assert_eq!(fused.dims(), &[rows, outf]);
    let eager = eager_forward(&x, &w, &a, &b, scale).unwrap();

    assert_close(
        &fused.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        &eager.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        0.0,
        "rank2_head",
    );
}

/// Boundary (family D): a single-row `x` (`M == 1`) — the degenerate GEMM
/// dimension every `bmnk` derivation must still handle correctly (not
/// merely "happens to work" for `M > 1`).
#[test]
fn single_row_boundary_matches_the_eager_composition() {
    let device = Device::Cpu;
    let (inf, outf, r) = (32usize, 24usize, 4usize);
    let scale = 8.0 / (r as f64);

    let x = Tensor::from_slice(&exact_fixture(inf, 21), (1, inf), &device).unwrap();
    let w = Tensor::from_slice(&exact_fixture(outf * inf, 22), (outf, inf), &device).unwrap();
    let a = Tensor::from_slice(&exact_fixture(r * inf, 23), (r, inf), &device).unwrap();
    let b = Tensor::from_slice(&exact_fixture(outf * r, 24), (outf, r), &device).unwrap();
    let ab = pack_ab(&a, &b).unwrap();

    let op = LowRankResidualLinear::new(scale as f32, inf, outf, r, None, false).unwrap();
    let fused = fused_forward(&x, &w, &ab, op).unwrap();
    let eager = eager_forward(&x, &w, &a, &b, scale).unwrap();

    assert_close(
        &fused.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        &eager.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        0.0,
        "single_row",
    );
}

/// Boundary (family D): an empty `x` (`M == 0`, e.g. a zero-length
/// leading batch dim) must be a no-op output of the right shape, not a
/// panic or an illegal-shaped GEMM.
#[test]
fn empty_rows_boundary_produces_an_empty_output_not_a_panic() {
    let device = Device::Cpu;
    let (inf, outf, r) = (8usize, 6usize, 2usize);
    let x = Tensor::from_slice(&[] as &[f32], (0, inf), &device).unwrap();
    let w = Tensor::from_slice(&fixture(outf * inf, 3.0), (outf, inf), &device).unwrap();
    let a = Tensor::from_slice(&fixture(r * inf, 3.1), (r, inf), &device).unwrap();
    let b = Tensor::from_slice(&fixture(outf * r, 3.2), (outf, r), &device).unwrap();
    let ab = pack_ab(&a, &b).unwrap();

    let op = LowRankResidualLinear::new(1.0, inf, outf, r, None, false).unwrap();
    let fused = fused_forward(&x, &w, &ab, op).unwrap();
    assert_eq!(fused.dims(), &[0, outf]);
}

/// Dropout at production width: the fused kernel's internal draw must
/// match `DropoutFused` applied directly with the SAME key — the same
/// determinism proof `ops::low_rank_residual_linear`'s own unit test makes at small
/// scale, repeated here at the width the #352 profile measured so a
/// scale-dependent indexing bug (e.g. an `i as u32` truncation somewhere
/// in the flatten path) cannot hide in a small fixture. `p = 0.5` (so the
/// inverted-dropout scale `1/(1-p) == 2.0` is exact in binary) and
/// exact-integer value fixtures, matching this file's `tol == 0.0` legs
/// elsewhere — see the module doc's "oracle contract" section.
#[test]
fn production_width_dropout_matches_dropout_fused_applied_directly() {
    let device = Device::Cpu;
    let (rows, inf, outf, r) = (24 * 128, 1024usize, 3072usize, 16usize);
    let scale = 8.0 / (r as f64);
    let key = DropoutKey {
        seed: 4242,
        layer_id: 3,
        forward_idx: 1,
        p: 0.5,
    };

    let x = Tensor::from_slice(&exact_fixture(rows * inf, 31), (rows, inf), &device).unwrap();
    let w = Tensor::from_slice(&exact_fixture(outf * inf, 32), (outf, inf), &device).unwrap();
    let a = Tensor::from_slice(&exact_fixture(r * inf, 33), (r, inf), &device).unwrap();
    let b = Tensor::from_slice(&exact_fixture(outf * r, 34), (outf, r), &device).unwrap();
    let ab = pack_ab(&a, &b).unwrap();

    let op = LowRankResidualLinear::new(scale as f32, inf, outf, r, Some(key), false).unwrap();
    let fused = fused_forward(&x, &w, &ab, op).unwrap();

    // `base` must use `x` UNDROPPED (only the LoRA branch is dropped),
    // so this is built manually rather than via `eager_forward` (whose
    // `base_out` and LoRA branch both read the same first argument).
    let dropout_op =
        jammi_kernels::ops::DropoutFused::new(key.seed, key.layer_id, key.forward_idx, key.p)
            .unwrap();
    let xd = jammi_kernels::ops::apply1(&x, dropout_op).unwrap();
    let base_out = x.matmul(&w.t().unwrap()).unwrap();
    let after_a = xd.matmul(&a.t().unwrap()).unwrap();
    let lora_out = after_a.matmul(&b.t().unwrap()).unwrap();
    let scaled = (&lora_out * scale).unwrap();
    let expected = (&base_out + &scaled).unwrap();

    assert_close(
        &fused.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        &expected.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        0.0,
        "production_width_dropout",
    );
}

/// `F16` (an unrelated reduced dtype, NOT `BF16`) reaching the fused op's
/// `w`/`x` slot is a typed domain refusal, not a silent misinterpretation
/// — this op's dtype domain is `(F32, F32)` / `(BF16, BF16)` only.
#[test]
fn f16_base_is_a_typed_domain_refusal() {
    let device = Device::Cpu;
    let (rows, inf, outf, r) = (4usize, 8usize, 6usize, 2usize);
    let x = Tensor::from_slice(&fixture(rows * inf, 6.0), (rows, inf), &device)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let w = Tensor::from_slice(&fixture(outf * inf, 6.1), (outf, inf), &device)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let a = Tensor::from_slice(&fixture(r * inf, 6.2), (r, inf), &device).unwrap();
    let b = Tensor::from_slice(&fixture(outf * r, 6.3), (outf, r), &device).unwrap();
    let ab = pack_ab(&a, &b).unwrap();

    let op = LowRankResidualLinear::new(1.0, inf, outf, r, None, false).unwrap();
    assert!(fused_forward(&x, &w, &ab, op).is_err());
}
