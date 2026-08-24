//! `LoraLinearFused` production-width oracles: CPU-hermetic, `F32`
//! (candle-core 0.11.0's CPU backend has no `BF16` matmul without
//! `mkl`/`accelerate` — see `ops::lora_linear`'s module doc; the `BF16`
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

use candle_core::{DType, Device, Result, Tensor, Var};
use jammi_kernels::ops::{DropoutKey, LoraLinearFused};

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

fn pack_ab(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let bt = b.t()?.contiguous()?;
    Tensor::cat(&[a, &bt], 1)
}

fn fused_forward(x: &Tensor, w: &Tensor, ab: &Tensor, op: LoraLinearFused) -> Result<Tensor> {
    x.apply_op3(w, ab, op)
}

/// A fixed, deterministic (not `Tensor::randn`-seeded — this crate does
/// not depend on `rand`) f32 fixture, values in a modest range.
fn fixture(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (phase + i as f32 * 0.017).sin() * 0.3)
        .collect()
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

    let x = Tensor::from_slice(&fixture(rows * inf, 0.1), (rows, inf), &device).unwrap();
    let w = Tensor::from_slice(&fixture(outf * inf, 0.2), (outf, inf), &device).unwrap();
    let a = Tensor::from_slice(&fixture(r * inf, 0.3), (r, inf), &device).unwrap();
    let b = Tensor::from_slice(&fixture(outf * r, 0.4), (outf, r), &device).unwrap();
    let ab = pack_ab(&a, &b).unwrap();

    let op = LoraLinearFused::new(scale as f32, inf, outf, r, None, false).unwrap();
    let fused = fused_forward(&x, &w, &ab, op).unwrap();
    let eager = eager_forward(&x, &w, &a, &b, scale).unwrap();

    let fused_v: Vec<f32> = fused.flatten_all().unwrap().to_vec1().unwrap();
    let eager_v: Vec<f32> = eager.flatten_all().unwrap().to_vec1().unwrap();
    assert_close(&fused_v, &eager_v, 0.0, "wqkv_forward");
}

/// Wo-shaped: `in=3072/? -> here in=1024,out=1024` GeGLU's `Wi` (packed,
/// `out=5248` for ModernBERT-large's intermediate*2) is the SECOND shape
/// the #352 profile names; exercised here at `in=1024, out=5248`.
#[test]
fn production_width_wi_forward_matches_the_eager_composition() {
    let device = Device::Cpu;
    let (rows, inf, outf, r) = (24 * 128, 1024usize, 5248usize, 16usize);
    let scale = 8.0 / (r as f64);

    let x = Tensor::from_slice(&fixture(rows * inf, 0.5), (rows, inf), &device).unwrap();
    let w = Tensor::from_slice(&fixture(outf * inf, 0.6), (outf, inf), &device).unwrap();
    let a = Tensor::from_slice(&fixture(r * inf, 0.7), (r, inf), &device).unwrap();
    let b = Tensor::from_slice(&fixture(outf * r, 0.8), (outf, r), &device).unwrap();
    let ab = pack_ab(&a, &b).unwrap();

    let op = LoraLinearFused::new(scale as f32, inf, outf, r, None, false).unwrap();
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
/// `LoraLinearFused::flatten_x`), so this is the shape-handling oracle,
/// not a second numeric one.
#[test]
fn production_width_rank3_forward_matches_the_eager_composition() {
    let device = Device::Cpu;
    let (b_dim, s_dim, inf, outf, r) = (4usize, 32usize, 1024usize, 3072usize, 16usize);
    let rows = b_dim * s_dim;
    let scale = 8.0 / (r as f64);

    let x = Tensor::from_slice(&fixture(rows * inf, 0.11), (b_dim, s_dim, inf), &device).unwrap();
    let w = Tensor::from_slice(&fixture(outf * inf, 0.21), (outf, inf), &device).unwrap();
    let a = Tensor::from_slice(&fixture(r * inf, 0.31), (r, inf), &device).unwrap();
    let b = Tensor::from_slice(&fixture(outf * r, 0.41), (outf, r), &device).unwrap();
    let ab = pack_ab(&a, &b).unwrap();

    let op = LoraLinearFused::new(scale as f32, inf, outf, r, None, false).unwrap();
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

    let op = LoraLinearFused::new(scale as f32, inf, outf, r, None, true).unwrap();
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

/// A rank-2 pooled classification head (`fine_tune/lora.rs`'s shape):
/// `in=1024` (a ModernBERT-large-width pooled vector), `out=2` (a binary
/// classification head), `rank=8`.
#[test]
fn rank2_pooled_classification_head_matches_the_eager_composition() {
    let device = Device::Cpu;
    let (rows, inf, outf, r) = (16usize, 1024usize, 2usize, 8usize);
    let scale = 8.0 / (r as f64);

    let x = Tensor::from_slice(&fixture(rows * inf, 0.9), (rows, inf), &device).unwrap();
    let w = Tensor::from_slice(&fixture(outf * inf, 1.1), (outf, inf), &device).unwrap();
    let a = Tensor::from_slice(&fixture(r * inf, 1.3), (r, inf), &device).unwrap();
    let b = Tensor::from_slice(&fixture(outf * r, 1.5), (outf, r), &device).unwrap();
    let ab = pack_ab(&a, &b).unwrap();

    let op = LoraLinearFused::new(scale as f32, inf, outf, r, None, false).unwrap();
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

    let x = Tensor::from_slice(&fixture(inf, 2.1), (1, inf), &device).unwrap();
    let w = Tensor::from_slice(&fixture(outf * inf, 2.2), (outf, inf), &device).unwrap();
    let a = Tensor::from_slice(&fixture(r * inf, 2.3), (r, inf), &device).unwrap();
    let b = Tensor::from_slice(&fixture(outf * r, 2.4), (outf, r), &device).unwrap();
    let ab = pack_ab(&a, &b).unwrap();

    let op = LoraLinearFused::new(scale as f32, inf, outf, r, None, false).unwrap();
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

    let op = LoraLinearFused::new(1.0, inf, outf, r, None, false).unwrap();
    let fused = fused_forward(&x, &w, &ab, op).unwrap();
    assert_eq!(fused.dims(), &[0, outf]);
}

/// Dropout at production width: the fused kernel's internal draw must
/// match `DropoutFused` applied directly with the SAME key — the same
/// determinism proof `ops::lora_linear`'s own unit test makes at small
/// scale, repeated here at the width the #352 profile measured so a
/// scale-dependent indexing bug (e.g. an `i as u32` truncation somewhere
/// in the flatten path) cannot hide in a small fixture.
#[test]
fn production_width_dropout_matches_dropout_fused_applied_directly() {
    let device = Device::Cpu;
    let (rows, inf, outf, r) = (24 * 128, 1024usize, 3072usize, 16usize);
    let scale = 8.0 / (r as f64);
    let key = DropoutKey {
        seed: 4242,
        layer_id: 3,
        forward_idx: 1,
        p: 0.05,
    };

    let x = Tensor::from_slice(&fixture(rows * inf, 5.0), (rows, inf), &device).unwrap();
    let w = Tensor::from_slice(&fixture(outf * inf, 5.1), (outf, inf), &device).unwrap();
    let a = Tensor::from_slice(&fixture(r * inf, 5.2), (r, inf), &device).unwrap();
    let b = Tensor::from_slice(&fixture(outf * r, 5.3), (outf, r), &device).unwrap();
    let ab = pack_ab(&a, &b).unwrap();

    let op = LoraLinearFused::new(scale as f32, inf, outf, r, Some(key), false).unwrap();
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

    let op = LoraLinearFused::new(1.0, inf, outf, r, None, false).unwrap();
    assert!(fused_forward(&x, &w, &ab, op).is_err());
}
