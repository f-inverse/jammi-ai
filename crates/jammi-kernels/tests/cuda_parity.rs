//! CPU↔CUDA parity oracle for `Axpy` — this is the landing proof the
//! lead's pod session runs, not a smoke test.
//!
//! Compiles and links ONLY with the `cuda` feature (`required-features =
//! ["cuda"]` in `Cargo.toml` — a plain `cargo test -p jammi-kernels` never
//! even builds this file). At runtime, a machine that compiled with the
//! feature but has no physical GPU attached (a real shape: a build image
//! with the CUDA toolkit but no driver) is treated as "skip", not "fail" —
//! `Device::new_cuda(0)` erroring is the signal — UNLESS the environment
//! variable `JAMMI_REQUIRE_CUDA` is set (the pod session that is this
//! file's actual landing proof sets it), in which case a device-
//! acquisition failure PANICS with the underlying error instead of
//! returning. Without that distinction, a broken device acquisition on the
//! pod would silently read as 5 skipped tests rather than 5 failed ones —
//! exactly the "fell back/skipped everywhere and it read as green" failure
//! mode `admission::AdmissionMode::Strict` exists to prevent, reproduced
//! here in the one file whose entire job is to be that failure mode's
//! landing proof.
//!
//! Covers exactly the divergence-prone classes this crate's own review
//! found bugs in: a contiguous case, a NARROWED tensor with a nonzero
//! `start_offset` (the missing-offset bug: the CUDA arm used to read the
//! base buffer's first `n` elements instead of the tensor's real range),
//! an empty tensor (the illegal `(0, 1, 1)` launch grid), a size that is
//! an exact multiple of the launch's 1024-thread block, a size that is
//! NOT a multiple of it (exercises the kernel's `if (i < n)` bounds check
//! on a partial last block), both supported dtypes (f32, bf16), and both
//! forward AND backward.
#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor, Var, D};
use half::bf16;
use jammi_kernels::ops::{
    apply1, apply2, apply3, Axpy, DropoutFused, DropoutKey, FullyMaskedPolicy, GegluFused,
    GeluVariant, LayerNormFused, LowRankResidualLinear, PhiloxKatProbe, RopeFused, ScaledCastAdd,
    SoftmaxLastDimFused,
};

fn axpy(alpha: f64, x: &Tensor, y: &Tensor) -> candle_core::Result<Tensor> {
    apply2(x, y, Axpy::new(alpha))
}

/// F32 forward/backward CPU-vs-CUDA absolute tolerance. nvcc's default
/// `--fmad=true` may contract `alpha*x+y` into a single-rounding hardware
/// FMA instruction on the GPU, differing from the CPU's two separately-
/// rounded operations (`alpha*x` then `+y`) by up to ~1 ULP — expected,
/// not a bug (this build pins `-use_fast_math` OFF, but fmad contraction
/// is a distinct nvcc default, not implied by that flag). This bound is
/// generous relative to that gap while still tight enough to catch a real
/// error (wrong offset, reversed operand, wrong dtype cast).
const F32_TOL: f64 = 1e-4;

fn cuda_device() -> Option<Device> {
    match Device::new_cuda(0) {
        Ok(d) => Some(d),
        Err(e) => {
            if std::env::var_os("JAMMI_REQUIRE_CUDA").is_some() {
                panic!(
                    "cuda_parity: JAMMI_REQUIRE_CUDA is set but no CUDA device could be \
                     acquired — this is the landing proof, a silent skip here is not \
                     acceptable: {e}"
                );
            }
            eprintln!("cuda_parity: skipping — no CUDA device available ({e})");
            None
        }
    }
}

/// A fixed, deterministic f32 fixture of length `n`, values in a modest
/// range so f32/bf16 rounding stays representative rather than degenerate.
fn fixture(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (seed + i as f32 * 0.37).sin() * 10.0)
        .collect()
}

/// A fixed, deterministic, SMALL-INTEGER `f32` fixture (`{-4, .., 4}`) —
/// mirrors `ops::low_rank_residual_linear`'s own `exact_fixture` and
/// `lora_linear_oracles.rs`'s copy of it: every partial sum this op's
/// GEMMs form from these values stays a SMALL EXACT INTEGER, well under
/// `f32`'s 24-bit mantissa's exact range even at this file's largest
/// reduction depth (`inf = 1024`, worst-case term `4*4=16`, worst-case sum
/// `1024*16 = 16384 << 2^24`). An exact-integer sum is IDENTICAL
/// regardless of summation order — so a `diff == 0.0` (bit-exact) claim
/// across CPU and CUDA is architecture/library-independent BY
/// CONSTRUCTION, not an accident of one `gemm` kernel's blocking choice
/// (both `gemm_reduced_precision_f32()` and `gemm_reduced_precision_bf16()`
/// default to `false` in candle-core 0.11.0 — see [`BF16_U`]'s doc for how
/// that was verified — so neither arm truncates to `tf32`/reduced `bf16`
/// precision here).
fn exact_fixture(n: usize, phase: i64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = (i as i64 * 7 + phase * 13).rem_euclid(9);
            (v - 4) as f32
        })
        .collect()
}

fn assert_parity_f32(cuda: &Device, alpha: f64, xv: &[f32], yv: &[f32]) {
    let cpu = Device::Cpu;
    let n = xv.len();

    let x_cpu = Var::from_tensor(&Tensor::from_slice(xv, (n,), &cpu).unwrap()).unwrap();
    let y_cpu = Var::from_tensor(&Tensor::from_slice(yv, (n,), &cpu).unwrap()).unwrap();
    let out_cpu = axpy(alpha, &x_cpu, &y_cpu).unwrap();
    let grads_cpu = out_cpu.backward().unwrap();

    let x_gpu = Var::from_tensor(&Tensor::from_slice(xv, (n,), cuda).unwrap()).unwrap();
    let y_gpu = Var::from_tensor(&Tensor::from_slice(yv, (n,), cuda).unwrap()).unwrap();
    let out_gpu = axpy(alpha, &x_gpu, &y_gpu).unwrap();
    let grads_gpu = out_gpu.backward().unwrap();

    let out_cpu_v: Vec<f32> = out_cpu.to_vec1().unwrap();
    let out_gpu_v: Vec<f32> = out_gpu.to_device(&cpu).unwrap().to_vec1().unwrap();
    assert_eq!(out_cpu_v.len(), n);
    // A short (or empty) `out_gpu_v` would make the `zip` below stop early
    // and pass VACUOUSLY — asserting the length explicitly is what turns
    // "the kernel silently produced fewer elements than requested" into a
    // failure instead of a no-op comparison.
    assert_eq!(
        out_gpu_v.len(),
        n,
        "GPU forward output length mismatch (got {}, expected {n})",
        out_gpu_v.len()
    );
    for (i, (c, g)) in out_cpu_v.iter().zip(out_gpu_v.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "fwd[{i}]: cpu {c} vs cuda {g} (alpha={alpha}, n={n})"
        );
    }

    let dx_cpu: Vec<f32> = grads_cpu.get(&x_cpu).unwrap().to_vec1().unwrap();
    let dx_gpu: Vec<f32> = grads_gpu
        .get(&x_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_vec1()
        .unwrap();
    let dy_cpu: Vec<f32> = grads_cpu.get(&y_cpu).unwrap().to_vec1().unwrap();
    let dy_gpu: Vec<f32> = grads_gpu
        .get(&y_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_vec1()
        .unwrap();
    // Same vacuous-pass hazard as the forward output above, for all four
    // gradient vectors.
    assert_eq!(dx_cpu.len(), n);
    assert_eq!(
        dx_gpu.len(),
        n,
        "dx GPU length mismatch (got {}, expected {n})",
        dx_gpu.len()
    );
    assert_eq!(dy_cpu.len(), n);
    assert_eq!(
        dy_gpu.len(),
        n,
        "dy GPU length mismatch (got {}, expected {n})",
        dy_gpu.len()
    );
    for (i, (c, g)) in dx_cpu.iter().zip(dx_gpu.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "dx[{i}]: cpu {c} vs cuda {g}"
        );
    }
    for (i, (c, g)) in dy_cpu.iter().zip(dy_gpu.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "dy[{i}]: cpu {c} vs cuda {g}"
        );
    }
}

fn assert_parity_bf16(cuda: &Device, alpha: f64, xv: &[f32], yv: &[f32]) {
    let cpu = Device::Cpu;
    let n = xv.len();
    let xb: Vec<bf16> = xv.iter().map(|&v| bf16::from_f32(v)).collect();
    let yb: Vec<bf16> = yv.iter().map(|&v| bf16::from_f32(v)).collect();

    let x_cpu = Tensor::from_slice(&xb, (n,), &cpu).unwrap();
    let y_cpu = Tensor::from_slice(&yb, (n,), &cpu).unwrap();
    let out_cpu: Vec<f32> = axpy(alpha, &x_cpu, &y_cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap();

    let x_gpu = Tensor::from_slice(&xb, (n,), cuda).unwrap();
    let y_gpu = Tensor::from_slice(&yb, (n,), cuda).unwrap();
    let out_gpu: Vec<f32> = axpy(alpha, &x_gpu, &y_gpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_vec1()
        .unwrap();

    assert_eq!(out_cpu.len(), n);
    assert_eq!(
        out_gpu.len(),
        n,
        "bf16 GPU forward output length mismatch (got {}, expected {n})",
        out_gpu.len()
    );

    // Same accumulation semantics on both devices (f32-accumulate, round
    // to bf16 once) — this should be tighter than the CPU fused-vs-eager
    // bf16 bound (`tests/oracles.rs`), which compares two DIFFERENT
    // rounding paths; here both paths are the same kernel semantics on
    // different hardware, so fmad-class ~1-ULP-at-bf16 differences are
    // the only expected source of divergence.
    for (i, (c, g)) in out_cpu.iter().zip(out_gpu.iter()).enumerate() {
        let ulp = 2.0f32.powi(-7) * c.abs().max(*g).max(1.0);
        assert!(
            (c - g).abs() <= 2.0 * ulp,
            "bf16 fwd[{i}]: cpu {c} vs cuda {g}"
        );
    }
}

#[test]
fn parity_contiguous_small() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let x = fixture(8, 1.0);
    let y = fixture(8, 2.0);
    assert_parity_f32(&cuda, 1.75, &x, &y);
    assert_parity_bf16(&cuda, 1.75, &x, &y);
}

#[test]
fn parity_narrowed_with_nonzero_offset() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    // Build a [3, 8] tensor and narrow to the middle row: contiguous but
    // start_offset == 8 (nonzero) — exactly the case the CUDA arm used to
    // get wrong (it ignored `start_offset` and read the base buffer's
    // first `n` elements instead of the narrowed row's own data).
    let base_x = fixture(24, 3.0);
    let base_y = fixture(24, 4.0);
    let cpu = Device::Cpu;

    let xt_cpu = Tensor::from_slice(&base_x, (3, 8), &cpu)
        .unwrap()
        .narrow(0, 1, 1)
        .unwrap()
        .flatten_all()
        .unwrap();
    let yt_cpu = Tensor::from_slice(&base_y, (3, 8), &cpu)
        .unwrap()
        .narrow(0, 1, 1)
        .unwrap()
        .flatten_all()
        .unwrap();
    assert!(xt_cpu.is_contiguous());
    assert_ne!(xt_cpu.layout().start_offset(), 0);

    let xt_gpu = Tensor::from_slice(&base_x, (3, 8), &cuda)
        .unwrap()
        .narrow(0, 1, 1)
        .unwrap()
        .flatten_all()
        .unwrap();
    let yt_gpu = Tensor::from_slice(&base_y, (3, 8), &cuda)
        .unwrap()
        .narrow(0, 1, 1)
        .unwrap()
        .flatten_all()
        .unwrap();
    assert!(xt_gpu.is_contiguous());
    assert_ne!(xt_gpu.layout().start_offset(), 0);

    let out_cpu: Vec<f32> = axpy(2.0, &xt_cpu, &yt_cpu).unwrap().to_vec1().unwrap();
    let out_gpu: Vec<f32> = axpy(2.0, &xt_gpu, &yt_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_vec1()
        .unwrap();
    // Assert lengths BEFORE the zip below: a short (or empty) `out_gpu` —
    // e.g. if the offset fix regressed and the kernel launched over the
    // wrong (possibly zero) element count — would otherwise make the zip
    // stop early and the per-element comparison pass vacuously.
    assert_eq!(out_cpu.len(), 8);
    assert_eq!(
        out_gpu.len(),
        8,
        "narrowed GPU forward output length mismatch (got {}, expected 8)",
        out_gpu.len()
    );
    for (i, (c, g)) in out_cpu.iter().zip(out_gpu.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "narrowed fwd[{i}]: cpu {c} vs cuda {g}"
        );
    }
    // And both must equal the middle row's own values, not the base
    // buffer's first 8 elements (the exact bug this test exists for).
    let expected: Vec<f32> = base_x[8..16]
        .iter()
        .zip(base_y[8..16].iter())
        .map(|(&x, &y)| 2.0 * x + y)
        .collect();
    for (i, (g, e)) in out_gpu.iter().zip(expected.iter()).enumerate() {
        assert!(
            ((*g - *e).abs() as f64) <= F32_TOL,
            "narrowed fwd[{i}] vs hand-computed: cuda {g} vs {e}"
        );
    }
}

#[test]
fn parity_empty() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let cpu = Device::Cpu;
    let x_cpu = Tensor::from_slice(&[] as &[f32], (0,), &cpu).unwrap();
    let y_cpu = Tensor::from_slice(&[] as &[f32], (0,), &cpu).unwrap();
    let x_gpu = Tensor::from_slice(&[] as &[f32], (0,), &cuda).unwrap();
    let y_gpu = Tensor::from_slice(&[] as &[f32], (0,), &cuda).unwrap();

    let out_cpu: Vec<f32> = axpy(1.0, &x_cpu, &y_cpu).unwrap().to_vec1().unwrap();
    // This must not attempt an illegal (0, 1, 1) launch grid.
    let out_gpu: Vec<f32> = axpy(1.0, &x_gpu, &y_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(out_cpu.is_empty());
    assert!(out_gpu.is_empty());
}

#[test]
fn parity_multi_block_exact_multiple_of_block_size() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    // 4096 == 4 * 1024 (the launch's block_dim) — a clean multi-block
    // boundary, every block fully occupied.
    let x = fixture(4096, 5.0);
    let y = fixture(4096, 6.0);
    assert_parity_f32(&cuda, 0.5, &x, &y);
}

#[test]
fn parity_multi_block_not_a_multiple_of_block_size() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    // 2000 spans 2 blocks (grid = ceil(2000/1024) = 2) with a PARTIAL last
    // block — exercises the kernel's `if (i < n)` bounds check for threads
    // past the real element count.
    let x = fixture(2000, 7.0);
    let y = fixture(2000, 8.0);
    assert_parity_f32(&cuda, -2.25, &x, &y);
    assert_parity_bf16(&cuda, -2.25, &x, &y);
}

// =======================================================================
// LayerNormFused CPU<->CUDA parity: fwd + BOTH backward outputs (dx,
// dgamma). Covers the same divergence-prone classes as Axpy's suite above
// (contiguous, narrowed-with-nonzero-offset, empty, a block-size boundary)
// PLUS the LN-specific dimensions this op actually varies over: hidden
// 1024 (ModernBERT-large) and a non-1024 hidden, and multiple rows per
// launch (one CUDA thread block per row).
// =======================================================================

fn ln_forward(
    eps: f64,
    dgamma_needed: bool,
    x: &Tensor,
    gamma: &Tensor,
) -> candle_core::Result<Tensor> {
    apply2(x, gamma, LayerNormFused::new(eps, dgamma_needed))
}

fn assert_ln_parity_f32(
    cuda: &Device,
    eps: f64,
    rows: usize,
    hidden: usize,
    xv: &[f32],
    gv: &[f32],
) {
    let cpu = Device::Cpu;
    let n = rows * hidden;

    let x_cpu = Var::from_tensor(&Tensor::from_slice(xv, (rows, hidden), &cpu).unwrap()).unwrap();
    let g_cpu = Var::from_tensor(&Tensor::from_slice(gv, (hidden,), &cpu).unwrap()).unwrap();
    let out_cpu = ln_forward(eps, true, &x_cpu, &g_cpu).unwrap();
    let grads_cpu = out_cpu.backward().unwrap();

    let x_gpu = Var::from_tensor(&Tensor::from_slice(xv, (rows, hidden), cuda).unwrap()).unwrap();
    let g_gpu = Var::from_tensor(&Tensor::from_slice(gv, (hidden,), cuda).unwrap()).unwrap();
    let out_gpu = ln_forward(eps, true, &x_gpu, &g_gpu).unwrap();
    let grads_gpu = out_gpu.backward().unwrap();

    let out_cpu_v: Vec<f32> = out_cpu.flatten_all().unwrap().to_vec1().unwrap();
    let out_gpu_v: Vec<f32> = out_gpu
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(out_cpu_v.len(), n);
    assert_eq!(out_gpu_v.len(), n, "LN GPU fwd length mismatch");
    for (i, (c, g)) in out_cpu_v.iter().zip(out_gpu_v.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "ln fwd[{i}]: cpu {c} vs cuda {g} (rows={rows}, hidden={hidden})"
        );
    }

    let dx_cpu: Vec<f32> = grads_cpu
        .get(&x_cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let dx_gpu: Vec<f32> = grads_gpu
        .get(&x_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(dx_cpu.len(), n);
    assert_eq!(dx_gpu.len(), n, "LN GPU dx length mismatch");
    for (i, (c, g)) in dx_cpu.iter().zip(dx_gpu.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "ln dx[{i}]: cpu {c} vs cuda {g}"
        );
    }

    let dg_cpu: Vec<f32> = grads_cpu.get(&g_cpu).unwrap().to_vec1().unwrap();
    let dg_gpu: Vec<f32> = grads_gpu
        .get(&g_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(dg_cpu.len(), hidden);
    assert_eq!(dg_gpu.len(), hidden, "LN GPU dgamma length mismatch");
    for (i, (c, g)) in dg_cpu.iter().zip(dg_gpu.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "ln dgamma[{i}]: cpu {c} vs cuda {g}"
        );
    }
}

fn assert_ln_parity_bf16(
    cuda: &Device,
    eps: f64,
    rows: usize,
    hidden: usize,
    xv: &[f32],
    gv: &[f32],
) {
    let cpu = Device::Cpu;
    let n = rows * hidden;
    let xb: Vec<bf16> = xv.iter().map(|&v| bf16::from_f32(v)).collect();
    let gb: Vec<bf16> = gv.iter().map(|&v| bf16::from_f32(v)).collect();

    let x_cpu = Var::from_tensor(&Tensor::from_slice(&xb, (rows, hidden), &cpu).unwrap()).unwrap();
    let g_cpu = Var::from_tensor(&Tensor::from_slice(&gb, (hidden,), &cpu).unwrap()).unwrap();
    let out_cpu = ln_forward(eps, true, &x_cpu, &g_cpu).unwrap();
    let grads_cpu = out_cpu.backward().unwrap();

    let x_gpu = Var::from_tensor(&Tensor::from_slice(&xb, (rows, hidden), cuda).unwrap()).unwrap();
    let g_gpu = Var::from_tensor(&Tensor::from_slice(&gb, (hidden,), cuda).unwrap()).unwrap();
    let out_gpu = ln_forward(eps, true, &x_gpu, &g_gpu).unwrap();
    let grads_gpu = out_gpu.backward().unwrap();

    let to_f32 = |t: &Tensor| -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    };
    // Same accumulation semantics on both devices (f32-accumulate, round
    // to bf16 once) — an fmad-class ~1-ULP-at-bf16 bound, same rationale
    // as `assert_parity_bf16` above.
    let bf16_bound = |c: f32, g: f32| 2.0 * 2.0f32.powi(-7) * c.abs().max(g).max(1.0);

    let out_cpu_v = to_f32(&out_cpu);
    let out_gpu_v = to_f32(&out_gpu.to_device(&cpu).unwrap());
    assert_eq!(out_cpu_v.len(), n);
    assert_eq!(out_gpu_v.len(), n, "LN bf16 GPU fwd length mismatch");
    for (i, (c, g)) in out_cpu_v.iter().zip(out_gpu_v.iter()).enumerate() {
        assert!(
            (c - g).abs() <= bf16_bound(*c, *g),
            "ln bf16 fwd[{i}]: cpu {c} vs cuda {g}"
        );
    }

    let dx_cpu_v = to_f32(&grads_cpu.get(&x_cpu).unwrap().clone());
    let dx_gpu_v = to_f32(&grads_gpu.get(&x_gpu).unwrap().to_device(&cpu).unwrap());
    assert_eq!(dx_cpu_v.len(), n);
    assert_eq!(dx_gpu_v.len(), n, "LN bf16 GPU dx length mismatch");
    for (i, (c, g)) in dx_cpu_v.iter().zip(dx_gpu_v.iter()).enumerate() {
        assert!(
            (c - g).abs() <= bf16_bound(*c, *g),
            "ln bf16 dx[{i}]: cpu {c} vs cuda {g}"
        );
    }

    let dg_cpu_v = to_f32(&grads_cpu.get(&g_cpu).unwrap().clone());
    let dg_gpu_v = to_f32(&grads_gpu.get(&g_gpu).unwrap().to_device(&cpu).unwrap());
    assert_eq!(dg_cpu_v.len(), hidden);
    assert_eq!(dg_gpu_v.len(), hidden, "LN bf16 GPU dgamma length mismatch");
    for (i, (c, g)) in dg_cpu_v.iter().zip(dg_gpu_v.iter()).enumerate() {
        assert!(
            (c - g).abs() <= bf16_bound(*c, *g),
            "ln bf16 dgamma[{i}]: cpu {c} vs cuda {g}"
        );
    }
}

#[test]
fn ln_parity_contiguous_hidden_1024_modernbert_shape() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let rows = 4;
    let hidden = 1024;
    let x = fixture(rows * hidden, 1.0);
    let g = fixture(hidden, 2.0);
    assert_ln_parity_f32(&cuda, 1e-5, rows, hidden, &x, &g);
    assert_ln_parity_bf16(&cuda, 1e-5, rows, hidden, &x, &g);
}

#[test]
fn ln_parity_contiguous_non_1024_hidden() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    // Not a power of two, not a multiple of the kernel's LN_BLOCK (256) —
    // exercises the grid-stride tail within a row.
    let rows = 3;
    let hidden = 300;
    let x = fixture(rows * hidden, 3.0);
    let g = fixture(hidden, 4.0);
    assert_ln_parity_f32(&cuda, 1e-5, rows, hidden, &x, &g);
    assert_ln_parity_bf16(&cuda, 1e-5, rows, hidden, &x, &g);
}

#[test]
fn ln_parity_narrowed_with_nonzero_offset() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    // A [3, rows, hidden] tensor narrowed to its middle "batch" slab: the
    // resulting [rows, hidden] view is contiguous but has a nonzero
    // `start_offset` — the same class of bug `Axpy`'s CUDA arm had
    // (reading the base buffer's first elements instead of the tensor's
    // real range) reproduced for LN's own `contiguous_offsets()` slicing.
    let rows = 2;
    let hidden = 16;
    let base_x = fixture(3 * rows * hidden, 5.0);
    let g = fixture(hidden, 6.0);
    let cpu = Device::Cpu;

    let x_cpu = Tensor::from_slice(&base_x, (3, rows, hidden), &cpu)
        .unwrap()
        .narrow(0, 1, 1)
        .unwrap()
        .flatten(0, 1)
        .unwrap();
    assert!(x_cpu.is_contiguous());
    assert_ne!(x_cpu.layout().start_offset(), 0);

    let x_gpu = Tensor::from_slice(&base_x, (3, rows, hidden), &cuda)
        .unwrap()
        .narrow(0, 1, 1)
        .unwrap()
        .flatten(0, 1)
        .unwrap();
    let g_cpu = Tensor::from_slice(&g, (hidden,), &cpu).unwrap();
    let g_gpu = Tensor::from_slice(&g, (hidden,), &cuda).unwrap();

    let out_cpu: Vec<f32> = ln_forward(1e-5, false, &x_cpu, &g_cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let out_gpu: Vec<f32> = ln_forward(1e-5, false, &x_gpu, &g_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(out_cpu.len(), rows * hidden);
    assert_eq!(
        out_gpu.len(),
        rows * hidden,
        "narrowed LN GPU fwd length mismatch"
    );
    for (i, (c, g)) in out_cpu.iter().zip(out_gpu.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "narrowed ln fwd[{i}]: cpu {c} vs cuda {g}"
        );
    }
    // And matches hand-computed LayerNorm over the middle slab's own
    // data, not the base buffer's first `rows*hidden` elements.
    let expected_slab = &base_x[rows * hidden..2 * rows * hidden];
    for r in 0..rows {
        let row = &expected_slab[r * hidden..(r + 1) * hidden];
        let mean: f32 = row.iter().sum::<f32>() / hidden as f32;
        let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / hidden as f32;
        let invvar = 1.0 / (var + 1e-5f32).sqrt();
        for i in 0..hidden {
            let expected = (row[i] - mean) * invvar * g[i];
            let got = out_gpu[r * hidden + i];
            assert!(
                (got - expected).abs() <= F32_TOL as f32,
                "narrowed ln fwd[{r},{i}] vs hand-computed: cuda {got} vs {expected}"
            );
        }
    }
}

#[test]
fn ln_parity_empty_batch() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let cpu = Device::Cpu;
    let hidden = 8;
    let x_cpu = Tensor::from_slice(&[] as &[f32], (0, hidden), &cpu).unwrap();
    let x_gpu = Tensor::from_slice(&[] as &[f32], (0, hidden), &cuda).unwrap();
    let g = fixture(hidden, 9.0);
    let g_cpu = Tensor::from_slice(&g, (hidden,), &cpu).unwrap();
    let g_gpu = Tensor::from_slice(&g, (hidden,), &cuda).unwrap();

    let out_cpu: Vec<f32> = ln_forward(1e-5, false, &x_cpu, &g_cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    // Must not attempt an illegal zero-block launch.
    let out_gpu: Vec<f32> = ln_forward(1e-5, false, &x_gpu, &g_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(out_cpu.is_empty());
    assert!(out_gpu.is_empty());
}

// =======================================================================
// RopeFused CPU<->CUDA parity: fwd + bwd (dx). Covers the same
// divergence-prone classes as Axpy's/LayerNormFused's suites above
// (contiguous, narrowed-with-nonzero-offset, empty, a block-size
// boundary) PLUS the RoPE-specific dimension this op varies over:
// head_dim 64 (ModernBERT-large's) and a non-power-of-two EVEN head_dim
// (exercises the grid-stride tail and the `col < half` branch's bounds).
// =======================================================================

fn rope(negate_sin: bool, x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
    apply3(x, cos, sin, RopeFused::new(negate_sin))
}

/// A deterministic `[period, hidden]` RoPE table with the SAME
/// column-duplication `RotaryEmbedding::new` bakes in (`jammi-encoders`) —
/// this op's domain premise.
fn rope_table(period: usize, hidden: usize, theta_base: f64) -> Vec<f32> {
    let half = hidden / 2;
    let mut out = vec![0f32; period * hidden];
    for pos in 0..period {
        for half_pass in 0..2 {
            for i in 0..half {
                let theta = (pos as f64) * theta_base.powf(-2.0 * i as f64 / hidden as f64);
                out[pos * hidden + half_pass * half + i] = theta.cos() as f32;
            }
        }
    }
    out
}

fn rope_sin_table(period: usize, hidden: usize, theta_base: f64) -> Vec<f32> {
    let half = hidden / 2;
    let mut out = vec![0f32; period * hidden];
    for pos in 0..period {
        for half_pass in 0..2 {
            for i in 0..half {
                let theta = (pos as f64) * theta_base.powf(-2.0 * i as f64 / hidden as f64);
                out[pos * hidden + half_pass * half + i] = theta.sin() as f32;
            }
        }
    }
    out
}

fn assert_rope_parity_f32(cuda: &Device, batch: usize, seq: usize, hidden: usize, xv: &[f32]) {
    let cpu = Device::Cpu;
    let n = xv.len();
    let cos_v = rope_table(seq, hidden, 10_000.0);
    let sin_v = rope_sin_table(seq, hidden, 10_000.0);

    let x_cpu =
        Var::from_tensor(&Tensor::from_slice(xv, (batch, 1, seq, hidden), &cpu).unwrap()).unwrap();
    let cos_cpu = Tensor::from_slice(&cos_v, (1, 1, seq, hidden), &cpu).unwrap();
    let sin_cpu = Tensor::from_slice(&sin_v, (1, 1, seq, hidden), &cpu).unwrap();
    let out_cpu = rope(false, &x_cpu, &cos_cpu, &sin_cpu).unwrap();
    let grads_cpu = out_cpu.backward().unwrap();

    let x_gpu =
        Var::from_tensor(&Tensor::from_slice(xv, (batch, 1, seq, hidden), cuda).unwrap()).unwrap();
    let cos_gpu = Tensor::from_slice(&cos_v, (1, 1, seq, hidden), cuda).unwrap();
    let sin_gpu = Tensor::from_slice(&sin_v, (1, 1, seq, hidden), cuda).unwrap();
    let out_gpu = rope(false, &x_gpu, &cos_gpu, &sin_gpu).unwrap();
    let grads_gpu = out_gpu.backward().unwrap();

    let out_cpu_v: Vec<f32> = out_cpu.flatten_all().unwrap().to_vec1().unwrap();
    let out_gpu_v: Vec<f32> = out_gpu
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(out_cpu_v.len(), n);
    assert_eq!(out_gpu_v.len(), n, "rope GPU fwd length mismatch");
    for (i, (c, g)) in out_cpu_v.iter().zip(out_gpu_v.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "rope fwd[{i}]: cpu {c} vs cuda {g} (batch={batch}, seq={seq}, hidden={hidden})"
        );
    }

    let dx_cpu: Vec<f32> = grads_cpu
        .get(&x_cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let dx_gpu: Vec<f32> = grads_gpu
        .get(&x_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(dx_cpu.len(), n);
    assert_eq!(dx_gpu.len(), n, "rope GPU dx length mismatch");
    for (i, (c, g)) in dx_cpu.iter().zip(dx_gpu.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "rope dx[{i}]: cpu {c} vs cuda {g}"
        );
    }
}

fn assert_rope_parity_bf16(cuda: &Device, batch: usize, seq: usize, hidden: usize, xv: &[f32]) {
    let cpu = Device::Cpu;
    let n = xv.len();
    let cos_v = rope_table(seq, hidden, 10_000.0);
    let sin_v = rope_sin_table(seq, hidden, 10_000.0);
    let xb: Vec<bf16> = xv.iter().map(|&v| bf16::from_f32(v)).collect();
    let cb: Vec<bf16> = cos_v.iter().map(|&v| bf16::from_f32(v)).collect();
    let sb: Vec<bf16> = sin_v.iter().map(|&v| bf16::from_f32(v)).collect();

    let x_cpu =
        Var::from_tensor(&Tensor::from_slice(&xb, (batch, 1, seq, hidden), &cpu).unwrap()).unwrap();
    let cos_cpu = Tensor::from_slice(&cb, (1, 1, seq, hidden), &cpu).unwrap();
    let sin_cpu = Tensor::from_slice(&sb, (1, 1, seq, hidden), &cpu).unwrap();
    let out_cpu = rope(false, &x_cpu, &cos_cpu, &sin_cpu).unwrap();
    let grads_cpu = out_cpu.backward().unwrap();

    let x_gpu =
        Var::from_tensor(&Tensor::from_slice(&xb, (batch, 1, seq, hidden), cuda).unwrap()).unwrap();
    let cos_gpu = Tensor::from_slice(&cb, (1, 1, seq, hidden), cuda).unwrap();
    let sin_gpu = Tensor::from_slice(&sb, (1, 1, seq, hidden), cuda).unwrap();
    let out_gpu = rope(false, &x_gpu, &cos_gpu, &sin_gpu).unwrap();
    let grads_gpu = out_gpu.backward().unwrap();

    let to_f32 = |t: &Tensor| -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    };
    let bf16_bound = |c: f32, g: f32| 2.0 * 2.0f32.powi(-7) * c.abs().max(g).max(1.0);

    let out_cpu_v = to_f32(&out_cpu);
    let out_gpu_v = to_f32(&out_gpu.to_device(&cpu).unwrap());
    assert_eq!(out_cpu_v.len(), n);
    assert_eq!(out_gpu_v.len(), n, "rope bf16 GPU fwd length mismatch");
    for (i, (c, g)) in out_cpu_v.iter().zip(out_gpu_v.iter()).enumerate() {
        assert!(
            (c - g).abs() <= bf16_bound(*c, *g),
            "rope bf16 fwd[{i}]: cpu {c} vs cuda {g}"
        );
    }

    let dx_cpu_v = to_f32(&grads_cpu.get(&x_cpu).unwrap().clone());
    let dx_gpu_v = to_f32(&grads_gpu.get(&x_gpu).unwrap().to_device(&cpu).unwrap());
    assert_eq!(dx_cpu_v.len(), n);
    assert_eq!(dx_gpu_v.len(), n, "rope bf16 GPU dx length mismatch");
    for (i, (c, g)) in dx_cpu_v.iter().zip(dx_gpu_v.iter()).enumerate() {
        assert!(
            (c - g).abs() <= bf16_bound(*c, *g),
            "rope bf16 dx[{i}]: cpu {c} vs cuda {g}"
        );
    }
}

#[test]
fn rope_parity_contiguous_head_dim_64_modernbert_large_shape() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let batch = 2;
    let seq = 6;
    let hidden = 64;
    let x = fixture(batch * seq * hidden, 1.0);
    assert_rope_parity_f32(&cuda, batch, seq, hidden, &x);
    assert_rope_parity_bf16(&cuda, batch, seq, hidden, &x);
}

#[test]
fn rope_parity_non_power_of_two_even_head_dim() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    // 20 is even (a valid rotate-half split) but not a power of two and
    // not a multiple of any convenient block size — exercises the
    // grid-stride tail and the `col < half` boundary.
    let batch = 3;
    let seq = 5;
    let hidden = 20;
    let x = fixture(batch * seq * hidden, 2.0);
    assert_rope_parity_f32(&cuda, batch, seq, hidden, &x);
    assert_rope_parity_bf16(&cuda, batch, seq, hidden, &x);
}

#[test]
fn rope_parity_narrowed_with_nonzero_offset() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    // A [3, batch, seq, hidden] tensor narrowed to its middle "batch"
    // slab: contiguous but nonzero `start_offset` — the same class of bug
    // `Axpy`'s/`LayerNormFused`'s CUDA arms had (reading the base buffer's
    // first elements instead of the tensor's real range).
    let batch = 2;
    let seq = 4;
    let hidden = 16;
    let base_x = fixture(3 * batch * seq * hidden, 3.0);
    let cos_v = rope_table(seq, hidden, 10_000.0);
    let sin_v = rope_sin_table(seq, hidden, 10_000.0);
    let cpu = Device::Cpu;

    let x_cpu = Tensor::from_slice(&base_x, (3, batch, seq, hidden), &cpu)
        .unwrap()
        .narrow(0, 1, 1)
        .unwrap()
        .flatten(0, 1)
        .unwrap();
    assert!(x_cpu.is_contiguous());
    assert_ne!(x_cpu.layout().start_offset(), 0);

    let x_gpu = Tensor::from_slice(&base_x, (3, batch, seq, hidden), &cuda)
        .unwrap()
        .narrow(0, 1, 1)
        .unwrap()
        .flatten(0, 1)
        .unwrap();
    let cos_cpu = Tensor::from_slice(&cos_v, (1, 1, seq, hidden), &cpu).unwrap();
    let sin_cpu = Tensor::from_slice(&sin_v, (1, 1, seq, hidden), &cpu).unwrap();
    let cos_gpu = Tensor::from_slice(&cos_v, (1, 1, seq, hidden), &cuda).unwrap();
    let sin_gpu = Tensor::from_slice(&sin_v, (1, 1, seq, hidden), &cuda).unwrap();

    let out_cpu: Vec<f32> = rope(false, &x_cpu, &cos_cpu, &sin_cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let out_gpu: Vec<f32> = rope(false, &x_gpu, &cos_gpu, &sin_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(out_cpu.len(), batch * seq * hidden);
    assert_eq!(
        out_gpu.len(),
        batch * seq * hidden,
        "narrowed rope GPU fwd length mismatch"
    );
    for (i, (c, g)) in out_cpu.iter().zip(out_gpu.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "narrowed rope fwd[{i}]: cpu {c} vs cuda {g}"
        );
    }
}

#[test]
fn rope_parity_empty_batch() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let cpu = Device::Cpu;
    let hidden = 8;
    let seq = 4;
    let x_cpu = Tensor::from_slice(&[] as &[f32], (0, 1, seq, hidden), &cpu).unwrap();
    let x_gpu = Tensor::from_slice(&[] as &[f32], (0, 1, seq, hidden), &cuda).unwrap();
    let cos_v = rope_table(seq, hidden, 10_000.0);
    let sin_v = rope_sin_table(seq, hidden, 10_000.0);
    let cos_cpu = Tensor::from_slice(&cos_v, (1, 1, seq, hidden), &cpu).unwrap();
    let sin_cpu = Tensor::from_slice(&sin_v, (1, 1, seq, hidden), &cpu).unwrap();
    let cos_gpu = Tensor::from_slice(&cos_v, (1, 1, seq, hidden), &cuda).unwrap();
    let sin_gpu = Tensor::from_slice(&sin_v, (1, 1, seq, hidden), &cuda).unwrap();

    let out_cpu: Vec<f32> = rope(false, &x_cpu, &cos_cpu, &sin_cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    // Must not attempt an illegal zero-block launch.
    let out_gpu: Vec<f32> = rope(false, &x_gpu, &cos_gpu, &sin_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(out_cpu.is_empty());
    assert!(out_gpu.is_empty());
}

/// Safe-softmax parity: a FULLY masked row (mask alone has no exact `0.0`
/// entry) must output ZEROS identically on CPU and CUDA, f32 and bf16 —
/// see `ops/softmax.rs`'s module doc's "fully-masked row: safe-softmax
/// zeros" section. Backward must also be exactly zero (falls out of the
/// existing `(dy - sum(dy*y)) * y` formula with `y == 0`).
#[test]
fn softmax_parity_fully_masked_row_is_zero_on_both_devices() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let cpu = Device::Cpu;
    let last = 8;
    let sv = fixture(last, 1.0);
    let mv = vec![-10_000.0f32; last];

    let s_cpu = Var::from_tensor(&Tensor::from_slice(&sv, (1, last), &cpu).unwrap()).unwrap();
    let m_cpu = Tensor::from_slice(&mv, (1, last), &cpu).unwrap();
    let out_cpu = softmax_with_policy(&s_cpu, &m_cpu, FullyMaskedPolicy::Zeros).unwrap();

    let s_gpu = Var::from_tensor(&Tensor::from_slice(&sv, (1, last), &cuda).unwrap()).unwrap();
    let m_gpu = Tensor::from_slice(&mv, (1, last), &cuda).unwrap();
    let out_gpu = softmax_with_policy(&s_gpu, &m_gpu, FullyMaskedPolicy::Zeros).unwrap();

    let out_cpu_v: Vec<f32> = out_cpu.flatten_all().unwrap().to_vec1().unwrap();
    let out_gpu_v: Vec<f32> = out_gpu
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(out_cpu_v, vec![0.0f32; last], "CPU must output zeros");
    assert_eq!(out_gpu_v, vec![0.0f32; last], "CUDA must output zeros");

    let dy_seed = fixture(last, 2.0);
    let dy_seed_cpu = Tensor::from_slice(&dy_seed, (1, last), &cpu).unwrap();
    let dy_seed_gpu = Tensor::from_slice(&dy_seed, (1, last), &cuda).unwrap();
    let loss_cpu = (&out_cpu * &dy_seed_cpu).unwrap().sum_all().unwrap();
    let loss_gpu = (&out_gpu * &dy_seed_gpu).unwrap().sum_all().unwrap();
    let dx_cpu: Vec<f32> = loss_cpu
        .backward()
        .unwrap()
        .get(&s_cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let dx_gpu: Vec<f32> = loss_gpu
        .backward()
        .unwrap()
        .get(&s_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(dx_cpu, vec![0.0f32; last], "CPU dscores must be zero");
    assert_eq!(dx_gpu, vec![0.0f32; last], "CUDA dscores must be zero");
}

// =======================================================================
// SoftmaxLastDimFused CPU<->CUDA parity: fwd + bwd (dscores). Covers the
// same divergence-prone classes as the suites above (contiguous, narrowed-
// with-nonzero-offset, empty, a block-size boundary) PLUS the dimensions
// this op actually varies over: a long row (seq 512 class, exercising the
// grid-stride reduction over many more elements than one block width) and
// a non-power-of-two last dim.
// =======================================================================

fn softmax(scores: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
    apply2(scores, mask, SoftmaxLastDimFused::default())
}

fn softmax_with_policy(
    scores: &Tensor,
    mask: &Tensor,
    policy: FullyMaskedPolicy,
) -> candle_core::Result<Tensor> {
    apply2(scores, mask, SoftmaxLastDimFused::new(policy))
}

fn eager_softmax(scores: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
    candle_nn::ops::softmax(&scores.broadcast_add(mask)?, D::Minus1)
}

/// A deterministic additive-mask fixture: `0.0` at most positions, a
/// finite large-negative (`-10_000.0`, matching `jammi_encoders::mask`'s
/// real `MASKED_LOGIT`) at every third position along the last axis.
fn mask_fixture(last: usize) -> Vec<f32> {
    (0..last)
        .map(|i| if i % 3 == 0 { -10_000.0 } else { 0.0 })
        .collect()
}

fn assert_softmax_parity_f32(cuda: &Device, rows: usize, last: usize, sv: &[f32]) {
    let cpu = Device::Cpu;
    let n = rows * last;
    let mv = mask_fixture(last);

    let s_cpu = Var::from_tensor(&Tensor::from_slice(sv, (rows, last), &cpu).unwrap()).unwrap();
    let m_cpu = Tensor::from_slice(&mv, (1, last), &cpu).unwrap();
    let out_cpu = softmax(&s_cpu, &m_cpu).unwrap();

    let s_gpu = Var::from_tensor(&Tensor::from_slice(sv, (rows, last), cuda).unwrap()).unwrap();
    let m_gpu = Tensor::from_slice(&mv, (1, last), cuda).unwrap();
    let out_gpu = softmax(&s_gpu, &m_gpu).unwrap();

    let out_cpu_v: Vec<f32> = out_cpu.flatten_all().unwrap().to_vec1().unwrap();
    let out_gpu_v: Vec<f32> = out_gpu
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(out_cpu_v.len(), n);
    assert_eq!(out_gpu_v.len(), n, "softmax GPU fwd length mismatch");
    for (i, (c, g)) in out_cpu_v.iter().zip(out_gpu_v.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "softmax fwd[{i}]: cpu {c} vs cuda {g} (rows={rows}, last={last})"
        );
    }

    // Non-uniform dy (family F): `Tensor::backward()`'s implicit all-ones
    // seed makes `dscores = (dy - sum(dy*y)) * y` IDENTICALLY zero for
    // every softmax row (`sum(y) == 1`), so a parity check seeded that way
    // would pass VACUOUSLY even with `bwd` badly broken (e.g. `* scale`
    // deleted). Seed with a fixed non-uniform `dy` instead, matching
    // `softmax_parity_fully_masked_row_is_zero_on_both_devices`'s own idiom.
    let dy_seed = fixture(n, 5.0);
    let dy_cpu = Tensor::from_slice(&dy_seed, (rows, last), &cpu).unwrap();
    let dy_gpu = Tensor::from_slice(&dy_seed, (rows, last), cuda).unwrap();
    let loss_cpu = (&out_cpu * &dy_cpu).unwrap().sum_all().unwrap();
    let loss_gpu = (&out_gpu * &dy_gpu).unwrap().sum_all().unwrap();
    let grads_cpu = loss_cpu.backward().unwrap();
    let grads_gpu = loss_gpu.backward().unwrap();

    let dx_cpu: Vec<f32> = grads_cpu
        .get(&s_cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let dx_gpu: Vec<f32> = grads_gpu
        .get(&s_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(dx_cpu.len(), n);
    assert_eq!(dx_gpu.len(), n, "softmax GPU dscores length mismatch");
    // Non-vacuity (family F): pin that the CPU reference itself is
    // measurably nonzero -- otherwise the parity loop below would pass
    // trivially without proving `bwd` ran the real formula.
    let dx_cpu_norm: f64 = dx_cpu
        .iter()
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>()
        .sqrt();
    assert!(
        dx_cpu_norm > 1e-3,
        "CPU dscores must be measured-nonzero (norm {dx_cpu_norm}) -- a vacuous \
         all-zero reference would make the parity check below prove nothing"
    );
    for (i, (c, g)) in dx_cpu.iter().zip(dx_gpu.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "softmax dscores[{i}]: cpu {c} vs cuda {g}"
        );
    }
}

fn assert_softmax_parity_bf16(cuda: &Device, rows: usize, last: usize, sv: &[f32]) {
    let cpu = Device::Cpu;
    let n = rows * last;
    let mv = mask_fixture(last);
    let sb: Vec<bf16> = sv.iter().map(|&v| bf16::from_f32(v)).collect();
    let mb: Vec<bf16> = mv.iter().map(|&v| bf16::from_f32(v)).collect();

    let s_cpu = Var::from_tensor(&Tensor::from_slice(&sb, (rows, last), &cpu).unwrap()).unwrap();
    let m_cpu = Tensor::from_slice(&mb, (1, last), &cpu).unwrap();
    let out_cpu = softmax(&s_cpu, &m_cpu).unwrap();

    let s_gpu = Var::from_tensor(&Tensor::from_slice(&sb, (rows, last), cuda).unwrap()).unwrap();
    let m_gpu = Tensor::from_slice(&mb, (1, last), cuda).unwrap();
    let out_gpu = softmax(&s_gpu, &m_gpu).unwrap();

    let to_f32 = |t: &Tensor| -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    };
    let bf16_bound = |c: f32, g: f32| 2.0 * 2.0f32.powi(-7) * c.abs().max(g).max(1.0);

    let out_cpu_v = to_f32(&out_cpu);
    let out_gpu_v = to_f32(&out_gpu.to_device(&cpu).unwrap());
    assert_eq!(out_cpu_v.len(), n);
    assert_eq!(out_gpu_v.len(), n, "softmax bf16 GPU fwd length mismatch");
    for (i, (c, g)) in out_cpu_v.iter().zip(out_gpu_v.iter()).enumerate() {
        assert!(
            (c - g).abs() <= bf16_bound(*c, *g),
            "softmax bf16 fwd[{i}]: cpu {c} vs cuda {g}"
        );
    }

    // Non-uniform dy (family F): see `assert_softmax_parity_f32`'s
    // identical note -- an implicit all-ones seed makes `dscores`
    // identically zero for a softmax row, which would pass vacuously.
    let dy_seed_f = fixture(n, 5.0);
    let dy_seed_b: Vec<bf16> = dy_seed_f.iter().map(|&v| bf16::from_f32(v)).collect();
    let dy_cpu = Tensor::from_slice(&dy_seed_b, (rows, last), &cpu).unwrap();
    let dy_gpu = Tensor::from_slice(&dy_seed_b, (rows, last), cuda).unwrap();
    let loss_cpu = (&out_cpu * &dy_cpu).unwrap().sum_all().unwrap();
    let loss_gpu = (&out_gpu * &dy_gpu).unwrap().sum_all().unwrap();
    let grads_cpu = loss_cpu.backward().unwrap();
    let grads_gpu = loss_gpu.backward().unwrap();

    let dx_cpu_v = to_f32(&grads_cpu.get(&s_cpu).unwrap().clone());
    let dx_gpu_v = to_f32(&grads_gpu.get(&s_gpu).unwrap().to_device(&cpu).unwrap());
    assert_eq!(dx_cpu_v.len(), n);
    assert_eq!(
        dx_gpu_v.len(),
        n,
        "softmax bf16 GPU dscores length mismatch"
    );
    // Non-vacuity (family F): pin that the CPU reference itself is
    // measurably nonzero before trusting the parity loop below.
    let dx_cpu_norm: f64 = dx_cpu_v
        .iter()
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>()
        .sqrt();
    assert!(
        dx_cpu_norm > 1e-3,
        "CPU dscores must be measured-nonzero (norm {dx_cpu_norm}) -- a vacuous \
         all-zero reference would make the parity check below prove nothing"
    );
    for (i, (c, g)) in dx_cpu_v.iter().zip(dx_gpu_v.iter()).enumerate() {
        assert!(
            (c - g).abs() <= bf16_bound(*c, *g),
            "softmax bf16 dscores[{i}]: cpu {c} vs cuda {g}"
        );
    }
}

#[test]
fn softmax_parity_contiguous_small() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let rows = 4;
    let last = 8;
    let sv = fixture(rows * last, 1.0);
    assert_softmax_parity_f32(&cuda, rows, last, &sv);
    assert_softmax_parity_bf16(&cuda, rows, last, &sv);
}

#[test]
fn softmax_parity_long_row_seq_512_class() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    // seq=512: ModernBERT's quadratic-regime shape, exercising the
    // grid-stride reduction over many more than one block width (256).
    let rows = 2;
    let last = 512;
    let sv = fixture(rows * last, 2.0);
    assert_softmax_parity_f32(&cuda, rows, last, &sv);
    assert_softmax_parity_bf16(&cuda, rows, last, &sv);
}

#[test]
fn softmax_parity_non_power_of_two_last_dim() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    // 300 is not a power of two and not a multiple of SM_BLOCK (256) --
    // exercises the grid-stride tail within a row.
    let rows = 3;
    let last = 300;
    let sv = fixture(rows * last, 3.0);
    assert_softmax_parity_f32(&cuda, rows, last, &sv);
    assert_softmax_parity_bf16(&cuda, rows, last, &sv);
}

#[test]
fn softmax_parity_narrowed_with_nonzero_offset() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    // A [3, rows, last] tensor narrowed to its middle "batch" slab: the
    // resulting [rows, last] view is contiguous but has a nonzero
    // `start_offset` -- the same class of bug this crate's other CUDA arms
    // had (reading the base buffer's first elements instead of the
    // tensor's real range).
    let rows = 2;
    let last = 16;
    let base = fixture(3 * rows * last, 4.0);
    let mv = mask_fixture(last);
    let cpu = Device::Cpu;

    let s_cpu = Tensor::from_slice(&base, (3, rows, last), &cpu)
        .unwrap()
        .narrow(0, 1, 1)
        .unwrap()
        .flatten(0, 1)
        .unwrap();
    assert!(s_cpu.is_contiguous());
    assert_ne!(s_cpu.layout().start_offset(), 0);

    let s_gpu = Tensor::from_slice(&base, (3, rows, last), &cuda)
        .unwrap()
        .narrow(0, 1, 1)
        .unwrap()
        .flatten(0, 1)
        .unwrap();
    let m_cpu = Tensor::from_slice(&mv, (1, last), &cpu).unwrap();
    let m_gpu = Tensor::from_slice(&mv, (1, last), &cuda).unwrap();

    let out_cpu: Vec<f32> = softmax(&s_cpu, &m_cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let out_gpu: Vec<f32> = softmax(&s_gpu, &m_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(out_cpu.len(), rows * last);
    assert_eq!(
        out_gpu.len(),
        rows * last,
        "narrowed softmax GPU fwd length mismatch"
    );
    for (i, (c, g)) in out_cpu.iter().zip(out_gpu.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "narrowed softmax fwd[{i}]: cpu {c} vs cuda {g}"
        );
    }
    // And matches the eager composition on the middle slab's own data, not
    // the base buffer's first `rows*last` elements.
    let expected_slab_scores =
        Tensor::from_slice(&base[rows * last..2 * rows * last], (rows, last), &cpu).unwrap();
    let expected = eager_softmax(&expected_slab_scores, &m_cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (i, (g, e)) in out_gpu.iter().zip(expected.iter()).enumerate() {
        assert!(
            ((*g - *e).abs() as f64) <= F32_TOL,
            "narrowed softmax fwd[{i}] vs eager: cuda {g} vs {e}"
        );
    }
}

#[test]
fn softmax_parity_empty_batch() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let cpu = Device::Cpu;
    let last = 8;
    let s_cpu = Tensor::from_slice(&[] as &[f32], (0, last), &cpu).unwrap();
    let s_gpu = Tensor::from_slice(&[] as &[f32], (0, last), &cuda).unwrap();
    let mv = mask_fixture(last);
    let m_cpu = Tensor::from_slice(&mv, (1, last), &cpu).unwrap();
    let m_gpu = Tensor::from_slice(&mv, (1, last), &cuda).unwrap();

    let out_cpu: Vec<f32> = softmax(&s_cpu, &m_cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    // Must not attempt an illegal zero-block launch.
    let out_gpu: Vec<f32> = softmax(&s_gpu, &m_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(out_cpu.is_empty());
    assert!(out_gpu.is_empty());
}

// =======================================================================
// SoftmaxLastDimFused CPU<->CUDA parity: `scale` semantics — folding
// `1/sqrt(head_dim)` into this op; see `ops/softmax.rs`'s module doc's
// "scale semantics" section. `scale =
// 0.125` (`1/sqrt(64)`, ModernBERT-large's REAL `head_dim`) throughout —
// an exact power of two, so the CPU<->CUDA comparison below is not
// confounded by the `scale` field's own rounding (already proven exact
// at this value in `tests/oracles.rs`'s CPU-hermetic legs); these legs
// are specifically about the CUDA kernel's OWN `scale * scores[i] +
// mask[i]` (F32) / softmax.cu's bf16_mul_rounded (softmax.cu:83) plus
// bf16_add_rounded (softmax.cu:101) (BF16) arithmetic matching the CPU
// arm's — contiguous, narrowed-with-nonzero-
// offset, and BOTH production `seq` classes (128, 512).
// =======================================================================

const HEAD_DIM_64_SCALE: f32 = 0.125;

fn softmax_scale(scores: &Tensor, mask: &Tensor, scale: f32) -> candle_core::Result<Tensor> {
    // `.with_scale` validates `scale` (family D — see its own doc) and
    // returns `KernelError`; every fixture in this file passes a genuine
    // finite positive scale, so `.expect` here is a test-fixture
    // assumption, not a silent unwrap of a real fallible path.
    apply2(
        scores,
        mask,
        SoftmaxLastDimFused::default()
            .with_scale(scale)
            .expect("test fixture scale must be finite and > 0.0"),
    )
}

fn assert_softmax_scale_parity_f32(
    cuda: &Device,
    rows: usize,
    last: usize,
    sv: &[f32],
    scale: f32,
) {
    let cpu = Device::Cpu;
    let n = rows * last;
    let mv = mask_fixture(last);

    let s_cpu = Var::from_tensor(&Tensor::from_slice(sv, (rows, last), &cpu).unwrap()).unwrap();
    let m_cpu = Tensor::from_slice(&mv, (1, last), &cpu).unwrap();
    let out_cpu = softmax_scale(&s_cpu, &m_cpu, scale).unwrap();

    let s_gpu = Var::from_tensor(&Tensor::from_slice(sv, (rows, last), cuda).unwrap()).unwrap();
    let m_gpu = Tensor::from_slice(&mv, (1, last), cuda).unwrap();
    let out_gpu = softmax_scale(&s_gpu, &m_gpu, scale).unwrap();

    let out_cpu_v: Vec<f32> = out_cpu.flatten_all().unwrap().to_vec1().unwrap();
    let out_gpu_v: Vec<f32> = out_gpu
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(out_cpu_v.len(), n);
    assert_eq!(out_gpu_v.len(), n, "softmax scale GPU fwd length mismatch");
    for (i, (c, g)) in out_cpu_v.iter().zip(out_gpu_v.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "softmax scale fwd[{i}]: cpu {c} vs cuda {g} (rows={rows}, last={last}, scale={scale})"
        );
    }

    // Non-uniform dy (family F): `Tensor::backward()`'s implicit all-ones
    // seed makes `dscores = (dy - sum(dy*y)) * y` IDENTICALLY zero for
    // every softmax row (`sum(y) == 1`) -- with a uniform seed, `dscores`
    // would be an uninformative all-zero on BOTH devices, exercising
    // nothing about `SoftmaxBwdDScores`'s CPU/CUDA numeric coupling.
    // Note what this parity check can and cannot catch: `bwd`'s `* scale`
    // multiply (`d_pre_softmax.affine(self.scale as f64, 0.0)`) is ONE
    // Rust code path shared by both devices -- deleting it would move the
    // CPU and CUDA arms of THIS check identically, so this comparison
    // cannot detect that regression regardless of the `dy` seed. That
    // check instead lives in `tests/oracles.rs`'s
    // `softmax_scale_bwd_multiplies_raw_dscores_by_scale`, which compares
    // against an INDEPENDENT reference graph rather than CPU-vs-CUDA. What
    // a non-uniform seed on THIS check buys is a measurably-nonzero
    // `dscores` that exercises `SoftmaxBwdDScores`'s own forward-y-coupled
    // arithmetic (`(dy - sum(dy*y)) * y`) matching between the CPU and
    // CUDA kernel implementations -- seed with a fixed non-uniform `dy`.
    let dy_seed = fixture(n, 6.0);
    let dy_cpu = Tensor::from_slice(&dy_seed, (rows, last), &cpu).unwrap();
    let dy_gpu = Tensor::from_slice(&dy_seed, (rows, last), cuda).unwrap();
    let loss_cpu = (&out_cpu * &dy_cpu).unwrap().sum_all().unwrap();
    let loss_gpu = (&out_gpu * &dy_gpu).unwrap().sum_all().unwrap();
    let grads_cpu = loss_cpu.backward().unwrap();
    let grads_gpu = loss_gpu.backward().unwrap();

    let dx_cpu: Vec<f32> = grads_cpu
        .get(&s_cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let dx_gpu: Vec<f32> = grads_gpu
        .get(&s_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(dx_cpu.len(), n);
    assert_eq!(dx_gpu.len(), n, "softmax scale GPU dscores length mismatch");
    // Non-vacuity (family F): pin that the CPU reference itself is
    // measurably nonzero -- otherwise the parity loop below proves
    // nothing about the CUDA kernel's own `scale` multiply.
    let dx_cpu_norm: f64 = dx_cpu
        .iter()
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>()
        .sqrt();
    assert!(
        dx_cpu_norm > 1e-3,
        "CPU dscores must be measured-nonzero (norm {dx_cpu_norm}) -- a vacuous \
         all-zero reference would make the parity check below prove nothing"
    );
    for (i, (c, g)) in dx_cpu.iter().zip(dx_gpu.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "softmax scale dscores[{i}]: cpu {c} vs cuda {g}"
        );
    }
}

fn assert_softmax_scale_parity_bf16(
    cuda: &Device,
    rows: usize,
    last: usize,
    sv: &[f32],
    scale: f32,
) {
    let cpu = Device::Cpu;
    let n = rows * last;
    let mv = mask_fixture(last);
    let sb: Vec<bf16> = sv.iter().map(|&v| bf16::from_f32(v)).collect();
    let mb: Vec<bf16> = mv.iter().map(|&v| bf16::from_f32(v)).collect();

    let s_cpu = Var::from_tensor(&Tensor::from_slice(&sb, (rows, last), &cpu).unwrap()).unwrap();
    let m_cpu = Tensor::from_slice(&mb, (1, last), &cpu).unwrap();
    let out_cpu = softmax_scale(&s_cpu, &m_cpu, scale).unwrap();

    let s_gpu = Var::from_tensor(&Tensor::from_slice(&sb, (rows, last), cuda).unwrap()).unwrap();
    let m_gpu = Tensor::from_slice(&mb, (1, last), cuda).unwrap();
    let out_gpu = softmax_scale(&s_gpu, &m_gpu, scale).unwrap();

    let to_f32 = |t: &Tensor| -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    };
    let bf16_bound = |c: f32, g: f32| 2.0 * 2.0f32.powi(-7) * c.abs().max(g).max(1.0);

    let out_cpu_v = to_f32(&out_cpu);
    let out_gpu_v = to_f32(&out_gpu.to_device(&cpu).unwrap());
    assert_eq!(out_cpu_v.len(), n);
    assert_eq!(
        out_gpu_v.len(),
        n,
        "softmax scale bf16 GPU fwd length mismatch"
    );
    for (i, (c, g)) in out_cpu_v.iter().zip(out_gpu_v.iter()).enumerate() {
        assert!(
            (c - g).abs() <= bf16_bound(*c, *g),
            "softmax scale bf16 fwd[{i}]: cpu {c} vs cuda {g}"
        );
    }

    // Non-uniform dy (family F): see `assert_softmax_scale_parity_f32`'s
    // identical note -- this exercises `SoftmaxBwdDScores`'s forward-y
    // coupling on a measurably-nonzero `dscores`, not the shared `* scale`
    // multiply (which a CPU<->CUDA comparison cannot catch either way;
    // that check lives in `tests/oracles.rs`'s
    // `softmax_scale_bwd_multiplies_raw_dscores_by_scale`).
    let dy_seed_f = fixture(n, 6.0);
    let dy_seed_b: Vec<bf16> = dy_seed_f.iter().map(|&v| bf16::from_f32(v)).collect();
    let dy_cpu = Tensor::from_slice(&dy_seed_b, (rows, last), &cpu).unwrap();
    let dy_gpu = Tensor::from_slice(&dy_seed_b, (rows, last), cuda).unwrap();
    let loss_cpu = (&out_cpu * &dy_cpu).unwrap().sum_all().unwrap();
    let loss_gpu = (&out_gpu * &dy_gpu).unwrap().sum_all().unwrap();
    let grads_cpu = loss_cpu.backward().unwrap();
    let grads_gpu = loss_gpu.backward().unwrap();

    let dx_cpu_v = to_f32(&grads_cpu.get(&s_cpu).unwrap().clone());
    let dx_gpu_v = to_f32(&grads_gpu.get(&s_gpu).unwrap().to_device(&cpu).unwrap());
    assert_eq!(dx_cpu_v.len(), n);
    assert_eq!(
        dx_gpu_v.len(),
        n,
        "softmax scale bf16 GPU dscores length mismatch"
    );
    // Non-vacuity (family F): pin that the CPU reference itself is
    // measurably nonzero before trusting the parity loop below.
    let dx_cpu_norm: f64 = dx_cpu_v
        .iter()
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>()
        .sqrt();
    assert!(
        dx_cpu_norm > 1e-3,
        "CPU dscores must be measured-nonzero (norm {dx_cpu_norm}) -- a vacuous \
         all-zero reference would make the parity check below prove nothing"
    );
    for (i, (c, g)) in dx_cpu_v.iter().zip(dx_gpu_v.iter()).enumerate() {
        assert!(
            (c - g).abs() <= bf16_bound(*c, *g),
            "softmax scale bf16 dscores[{i}]: cpu {c} vs cuda {g}"
        );
    }
}

#[test]
fn softmax_scale_parity_contiguous_small() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let rows = 4;
    let last = 8;
    let sv = fixture(rows * last, 1.0);
    assert_softmax_scale_parity_f32(&cuda, rows, last, &sv, HEAD_DIM_64_SCALE);
    assert_softmax_scale_parity_bf16(&cuda, rows, last, &sv, HEAD_DIM_64_SCALE);
}

/// `seq = 128` — ModernBERT-large's small-`seq` production class.
#[test]
fn softmax_scale_parity_head_dim_64_seq_128_class() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let rows = 4; // e.g. batch*heads collapsed onto the leading axis.
    let last = 128;
    let sv = fixture(rows * last, 2.0);
    assert_softmax_scale_parity_f32(&cuda, rows, last, &sv, HEAD_DIM_64_SCALE);
    assert_softmax_scale_parity_bf16(&cuda, rows, last, &sv, HEAD_DIM_64_SCALE);
}

/// `seq = 512` — the quadratic-regime `seq` class the fused-kernels plan
/// names as this program's actual memory target; exercises the
/// grid-stride reduction over many more than one block width (256).
#[test]
fn softmax_scale_parity_head_dim_64_seq_512_class() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let rows = 2;
    let last = 512;
    let sv = fixture(rows * last, 3.0);
    assert_softmax_scale_parity_f32(&cuda, rows, last, &sv, HEAD_DIM_64_SCALE);
    assert_softmax_scale_parity_bf16(&cuda, rows, last, &sv, HEAD_DIM_64_SCALE);
}

#[test]
fn softmax_scale_parity_narrowed_with_nonzero_offset() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    // A [3, rows, last] tensor narrowed to its middle "batch" slab (the
    // SAME class of bug this crate's other CUDA arms had — see
    // `softmax_parity_narrowed_with_nonzero_offset`'s identical
    // construction — now with `scale` threaded through).
    let rows = 2;
    let last = 16;
    let base = fixture(3 * rows * last, 4.0);
    let mv = mask_fixture(last);
    let cpu = Device::Cpu;

    let s_cpu = Tensor::from_slice(&base, (3, rows, last), &cpu)
        .unwrap()
        .narrow(0, 1, 1)
        .unwrap()
        .flatten(0, 1)
        .unwrap();
    assert!(s_cpu.is_contiguous());
    assert_ne!(s_cpu.layout().start_offset(), 0);

    let s_gpu = Tensor::from_slice(&base, (3, rows, last), &cuda)
        .unwrap()
        .narrow(0, 1, 1)
        .unwrap()
        .flatten(0, 1)
        .unwrap();
    let m_cpu = Tensor::from_slice(&mv, (1, last), &cpu).unwrap();
    let m_gpu = Tensor::from_slice(&mv, (1, last), &cuda).unwrap();

    let out_cpu: Vec<f32> = softmax_scale(&s_cpu, &m_cpu, HEAD_DIM_64_SCALE)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let out_gpu: Vec<f32> = softmax_scale(&s_gpu, &m_gpu, HEAD_DIM_64_SCALE)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(out_cpu.len(), rows * last);
    assert_eq!(
        out_gpu.len(),
        rows * last,
        "narrowed softmax scale GPU fwd length mismatch"
    );
    for (i, (c, g)) in out_cpu.iter().zip(out_gpu.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "narrowed softmax scale fwd[{i}]: cpu {c} vs cuda {g}"
        );
    }
    // And matches the eager `affine(scale, 0.0)` + `broadcast_add` +
    // `softmax` composition on the middle slab's own data (the two-op
    // composition `scale` replaces), not the base buffer's first
    // `rows*last` elements.
    let expected_slab_scores =
        Tensor::from_slice(&base[rows * last..2 * rows * last], (rows, last), &cpu).unwrap();
    let expected_pre = expected_slab_scores
        .affine(HEAD_DIM_64_SCALE as f64, 0.0)
        .unwrap();
    let expected = eager_softmax(&expected_pre, &m_cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (i, (g, e)) in out_gpu.iter().zip(expected.iter()).enumerate() {
        assert!(
            ((*g - *e).abs() as f64) <= F32_TOL,
            "narrowed softmax scale fwd[{i}] vs eager: cuda {g} vs {e}"
        );
    }
}

// =======================================================================
// SoftmaxLastDimFused: CUDA-only, SAME-DEVICE BF16 bit-exactness for the
// pre-mask-add rounding point (softmax.cu's bf16_mul_rounded, softmax.cu:83,
// used at softmax.cu:272/280/287). This is NOT a CPU<->CUDA parity check (every
// leg above is) -- it compares fused-with-scale against
// affine-then-fused-no-scale ENTIRELY ON CUDA, mirroring
// `tests/oracles.rs`'s CPU-hermetic
// `softmax_scale_bf16_small_additive_mask_bit_exact_vs_affine_then_unscaled`.
// It exists because NEITHER of this crate's other CUDA-BF16 checks can
// catch a regression in that rounding point: `assert_softmax_scale_parity_bf16`
// above is a CPU<->CUDA comparison at a 2-BF16-ULP bound wide enough to
// swallow exactly the single-extra-rounding-step this test defends
// against, and every mask fixture used by the suites above is the real
// ModernBERT alphabet `{0.0, -10_000.0, -20_000.0}`, which cannot observe
// the rounding point at all (round-identity at `0.0`, annihilation near
// `MASKED_LOGIT` -- see `small_bias_mask_fixture`'s own doc, identical
// reasoning to its CPU-oracle sibling). Concretely, this catches removing
// the pre-mask-add rounding from `softmax.cu` -- i.e. computing
// `scores[i] * scale + mask[i]` as one un-rounded step instead of
// `bf16_mul_rounded(scores[i], scale)` FIRST, rounding to BF16, THEN
// adding `mask[i]` and rounding again.
//
// Only the head_dim=128 leg below (`scale = 1/sqrt(128)`, not exactly
// representable in BF16 or F32) actually discriminates that mutation:
// multiplying a BF16 value by an exact power of two (the head_dim=64
// leg's `scale = 0.125 = 1/sqrt(64)`, ModernBERT-large's real production
// value) only shifts the exponent, so `bf16_mul_rounded(s, 0.125)` is
// exact and produces the IDENTICAL bit pattern whether or not the
// intermediate round-after-multiply step runs -- the head_dim=64 leg
// still proves CUDA/CPU-independent same-device bit-exactness at the real
// production scale, but a mutant that drops the intermediate rounding
// entirely passes it unchanged. Both legs are gated by
// `JAMMI_REQUIRE_CUDA` like every other test in this file (via
// `cuda_device()`'s early return).
// =======================================================================

/// A REL-POS-BIAS-shaped mask: small, continuous, NEVER `0.0` and NEVER
/// near `MASKED_LOGIT` magnitude — see this section's own doc and
/// `tests/oracles.rs`'s identical `small_bias_mask_fixture` for why the
/// real ModernBERT alphabet cannot exercise the rounding point this test
/// defends.
fn small_bias_mask_fixture(batch: usize, seq: usize) -> Vec<f32> {
    (0..batch * seq * seq)
        .map(|i| (i as f32 * 0.037 - 3.0).sin() * 0.5)
        .collect()
}

fn assert_softmax_scale_bf16_bit_exact_same_device_cuda(
    cuda: &Device,
    batch: usize,
    heads: usize,
    seq: usize,
    scale: f64,
) {
    let sv = fixture(batch * heads * seq * seq, 5.0);
    let mv = small_bias_mask_fixture(batch, seq);
    let sb: Vec<bf16> = sv.iter().map(|&v| bf16::from_f32(v)).collect();
    let mb: Vec<bf16> = mv.iter().map(|&v| bf16::from_f32(v)).collect();
    let scores = Tensor::from_slice(&sb, (batch, heads, seq, seq), cuda).unwrap();
    let mask = Tensor::from_slice(&mb, (batch, 1, seq, seq), cuda).unwrap();

    // `FullyMaskedPolicy::Propagate` (`softmax_scale`'s default via
    // `SoftmaxLastDimFused::default()`), NOT `Zeros`: the small, signed
    // `[-0.5, 0.5]` mask fixture is not the real masking alphabet, so a
    // row that happens to land all-negative is not "fully masked" in the
    // production sense -- `Zeros`'s short-circuit would misfire on it,
    // exactly as `tests/oracles.rs`'s CPU sibling documents.
    let fused_scaled = softmax_scale(&scores, &mask, scale as f32).unwrap();
    let affined = scores.affine(scale, 0.0).unwrap();
    let affine_then_unscaled = softmax_scale(&affined, &mask, 1.0).unwrap();

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
        "CUDA BF16 small-additive-mask leg (head_dim scale={scale}): fused-with-scale \
         must be BIT-EXACT vs affine-then-fused-no-scale, entirely on CUDA -- a \
         mismatch here means softmax.cu's pre-mask-add rounding point moved"
    );
}

#[test]
fn softmax_scale_bf16_small_additive_mask_bit_exact_same_device_head_dim_64() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    assert_softmax_scale_bf16_bit_exact_same_device_cuda(
        &cuda,
        2,
        16,
        128,
        HEAD_DIM_64_SCALE as f64,
    );
}

#[test]
fn softmax_scale_bf16_small_additive_mask_bit_exact_same_device_head_dim_128() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let scale = 1.0 / 128.0f64.sqrt();
    assert_softmax_scale_bf16_bit_exact_same_device_cuda(&cuda, 2, 16, 128, scale);
}

// =======================================================================
// GegluFused CPU<->CUDA parity: fwd + bwd (dwi_out). Covers the same
// divergence-prone classes as the suites above (contiguous, narrowed-with-
// nonzero-offset, empty, a block-size boundary, a non-power-of-two width)
// PLUS ModernBERT-large's actual production width (intermediate=2624),
// since this op's CUDA kernel is purely elementwise (grid-stride over
// n_out = rows*intermediate) rather than a per-row block reduction.
// =======================================================================

fn geglu(wi_out: &Tensor) -> candle_core::Result<Tensor> {
    apply1(wi_out, GegluFused::new(GeluVariant::Erf))
}

fn assert_geglu_parity_f32(cuda: &Device, rows: usize, intermediate: usize, wv: &[f32]) {
    let cpu = Device::Cpu;
    let n_out = rows * intermediate;

    let wi_cpu =
        Var::from_tensor(&Tensor::from_slice(wv, (rows, 2 * intermediate), &cpu).unwrap()).unwrap();
    let out_cpu = geglu(&wi_cpu).unwrap();
    let grads_cpu = out_cpu.backward().unwrap();

    let wi_gpu =
        Var::from_tensor(&Tensor::from_slice(wv, (rows, 2 * intermediate), cuda).unwrap()).unwrap();
    let out_gpu = geglu(&wi_gpu).unwrap();
    let grads_gpu = out_gpu.backward().unwrap();

    let out_cpu_v: Vec<f32> = out_cpu.flatten_all().unwrap().to_vec1().unwrap();
    let out_gpu_v: Vec<f32> = out_gpu
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(out_cpu_v.len(), n_out);
    assert_eq!(out_gpu_v.len(), n_out, "geglu GPU fwd length mismatch");
    for (i, (c, g)) in out_cpu_v.iter().zip(out_gpu_v.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "geglu fwd[{i}]: cpu {c} vs cuda {g} (rows={rows}, intermediate={intermediate})"
        );
    }

    let dwi_cpu: Vec<f32> = grads_cpu
        .get(&wi_cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let dwi_gpu: Vec<f32> = grads_gpu
        .get(&wi_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(dwi_cpu.len(), rows * 2 * intermediate);
    assert_eq!(
        dwi_gpu.len(),
        rows * 2 * intermediate,
        "geglu GPU dwi_out length mismatch"
    );
    for (i, (c, g)) in dwi_cpu.iter().zip(dwi_gpu.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "geglu dwi_out[{i}]: cpu {c} vs cuda {g}"
        );
    }
}

fn assert_geglu_parity_bf16(cuda: &Device, rows: usize, intermediate: usize, wv: &[f32]) {
    let cpu = Device::Cpu;
    let n_out = rows * intermediate;
    let wb: Vec<bf16> = wv.iter().map(|&v| bf16::from_f32(v)).collect();

    let wi_cpu =
        Var::from_tensor(&Tensor::from_slice(&wb, (rows, 2 * intermediate), &cpu).unwrap())
            .unwrap();
    let out_cpu = geglu(&wi_cpu).unwrap();
    let grads_cpu = out_cpu.backward().unwrap();

    let wi_gpu =
        Var::from_tensor(&Tensor::from_slice(&wb, (rows, 2 * intermediate), cuda).unwrap())
            .unwrap();
    let out_gpu = geglu(&wi_gpu).unwrap();
    let grads_gpu = out_gpu.backward().unwrap();

    let to_f32 = |t: &Tensor| -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    };
    // The SAME relative-with-floor-at-1.0 bound `assert_softmax_parity_bf16`
    // / the LayerNorm/RoPE parity helpers use (`REL = 2*2^-7 = 2^-6`,
    // floored at magnitude `1.0`) — NOT the wider relative-with-large-
    // absolute-floor bound `geglu_oracles.rs`'s CPU-only fused-vs-EAGER
    // oracle needs (`BF16_ABS_FLOOR = 2^-5`, sized for eager's rounding
    // CASCADE occasionally underflowing an intermediate to exact bf16
    // zero — a mechanism that cannot arise HERE, since both sides of THIS
    // comparison run the IDENTICAL fused single-rounding kernel, just on
    // different hardware/compilers). CPU<->CUDA divergence is bounded by
    // ordinary rounding-order/`--fmad=true` contraction (`build.rs`'s
    // documented tolerance doctrine), the same class every other op's
    // parity leg in this file already bounds with this exact formula.
    let bf16_bound = |c: f32, g: f32| 2.0 * 2.0f32.powi(-7) * c.abs().max(g).max(1.0);

    let out_cpu_v = to_f32(&out_cpu);
    let out_gpu_v = to_f32(&out_gpu.to_device(&cpu).unwrap());
    assert_eq!(out_cpu_v.len(), n_out);
    assert_eq!(out_gpu_v.len(), n_out, "geglu bf16 GPU fwd length mismatch");
    for (i, (c, g)) in out_cpu_v.iter().zip(out_gpu_v.iter()).enumerate() {
        assert!(
            (c - g).abs() <= bf16_bound(*c, *g),
            "geglu bf16 fwd[{i}]: cpu {c} vs cuda {g}"
        );
    }

    let dwi_cpu_v = to_f32(&grads_cpu.get(&wi_cpu).unwrap().clone());
    let dwi_gpu_v = to_f32(&grads_gpu.get(&wi_gpu).unwrap().to_device(&cpu).unwrap());
    assert_eq!(dwi_cpu_v.len(), rows * 2 * intermediate);
    assert_eq!(
        dwi_gpu_v.len(),
        rows * 2 * intermediate,
        "geglu bf16 GPU dwi_out length mismatch"
    );
    for (i, (c, g)) in dwi_cpu_v.iter().zip(dwi_gpu_v.iter()).enumerate() {
        assert!(
            (c - g).abs() <= bf16_bound(*c, *g),
            "geglu bf16 dwi_out[{i}]: cpu {c} vs cuda {g}"
        );
    }
}

#[test]
fn geglu_parity_contiguous_small() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let rows = 4;
    let intermediate = 8;
    let wv = fixture(rows * 2 * intermediate, 1.0);
    assert_geglu_parity_f32(&cuda, rows, intermediate, &wv);
    assert_geglu_parity_bf16(&cuda, rows, intermediate, &wv);
}

#[test]
fn geglu_parity_production_width_modernbert_large() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    // ModernBERT-large's real `intermediate_size` (HuggingFace's published
    // `answerdotai/ModernBERT-large` `config.json`) — also comfortably
    // multi-block for the 256-wide grid-stride kernel.
    let rows = 2;
    let intermediate = 2624;
    let wv = fixture(rows * 2 * intermediate, 2.0);
    assert_geglu_parity_f32(&cuda, rows, intermediate, &wv);
    assert_geglu_parity_bf16(&cuda, rows, intermediate, &wv);
}

#[test]
fn geglu_parity_non_power_of_two_intermediate() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    // 300 is not a power of two and not a multiple of GEGLU_BLOCK (256) --
    // exercises the grid-stride tail.
    let rows = 3;
    let intermediate = 300;
    let wv = fixture(rows * 2 * intermediate, 3.0);
    assert_geglu_parity_f32(&cuda, rows, intermediate, &wv);
    assert_geglu_parity_bf16(&cuda, rows, intermediate, &wv);
}

#[test]
fn geglu_parity_multi_block_exact_multiple_of_block_size() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    // n_out = rows*intermediate = 2*512 = 1024 = exactly 4 * GEGLU_BLOCK
    // (256).
    let rows = 2;
    let intermediate = 512;
    let wv = fixture(rows * 2 * intermediate, 4.0);
    assert_geglu_parity_f32(&cuda, rows, intermediate, &wv);
}

#[test]
fn geglu_parity_narrowed_with_nonzero_offset() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    // A [3, rows, 2*intermediate] tensor narrowed to its middle "batch"
    // slab: the resulting view is contiguous but has a nonzero
    // `start_offset` -- the same class of bug this crate's other CUDA arms
    // had (reading the base buffer's first elements instead of the
    // tensor's real range).
    let rows = 2;
    let intermediate = 16;
    let base = fixture(3 * rows * 2 * intermediate, 4.0);
    let cpu = Device::Cpu;

    let wi_cpu = Tensor::from_slice(&base, (3, rows, 2 * intermediate), &cpu)
        .unwrap()
        .narrow(0, 1, 1)
        .unwrap()
        .flatten(0, 1)
        .unwrap();
    assert!(wi_cpu.is_contiguous());
    assert_ne!(wi_cpu.layout().start_offset(), 0);

    let wi_gpu = Tensor::from_slice(&base, (3, rows, 2 * intermediate), &cuda)
        .unwrap()
        .narrow(0, 1, 1)
        .unwrap()
        .flatten(0, 1)
        .unwrap();

    let out_cpu: Vec<f32> = geglu(&wi_cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let out_gpu: Vec<f32> = geglu(&wi_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(out_cpu.len(), rows * intermediate);
    assert_eq!(
        out_gpu.len(),
        rows * intermediate,
        "narrowed geglu GPU fwd length mismatch"
    );
    for (i, (c, g)) in out_cpu.iter().zip(out_gpu.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "narrowed geglu fwd[{i}]: cpu {c} vs cuda {g}"
        );
    }
    // And matches the eager (narrow+narrow+gelu_erf+mul) composition on
    // the middle slab's own data, not the base buffer's first
    // `rows*2*intermediate` elements.
    let expected_slab = Tensor::from_slice(
        &base[rows * 2 * intermediate..2 * rows * 2 * intermediate],
        (rows, 2 * intermediate),
        &cpu,
    )
    .unwrap();
    let gate = expected_slab.narrow(D::Minus1, 0, intermediate).unwrap();
    let up = expected_slab
        .narrow(D::Minus1, intermediate, intermediate)
        .unwrap();
    let expected = (gate.gelu_erf().unwrap() * up)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for (i, (g, e)) in out_gpu.iter().zip(expected.iter()).enumerate() {
        assert!(
            ((*g - *e).abs() as f64) <= F32_TOL,
            "narrowed geglu fwd[{i}] vs eager: cuda {g} vs {e}"
        );
    }
}

#[test]
fn geglu_parity_empty_last_dim() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let cpu = Device::Cpu;
    let rows = 3;
    let wi_cpu = Tensor::from_slice(&[] as &[f32], (rows, 0), &cpu).unwrap();
    let wi_gpu = Tensor::from_slice(&[] as &[f32], (rows, 0), &cuda).unwrap();

    let out_cpu: Vec<f32> = geglu(&wi_cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    // Must not attempt an illegal zero-block launch.
    let out_gpu: Vec<f32> = geglu(&wi_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(out_cpu.is_empty());
    assert!(out_gpu.is_empty());
}

// =======================================================================
// ScaledCastAdd CPU<->CUDA parity: fwd + BOTH backward outputs (d_base,
// d_lora), at the two dtype combinations `jammi-lora`'s admission
// predicate actually reaches (`F32`/`F32` and `BF16` base /`F32` lora).
// Covers the same divergence-prone classes as Axpy's suite above
// (contiguous, narrowed-with-nonzero-offset, empty, a block-size boundary).
// =======================================================================

fn scaled_cast_add(scaling: f64, base: &Tensor, lora: &Tensor) -> candle_core::Result<Tensor> {
    apply2(base, lora, ScaledCastAdd::new(scaling))
}

fn assert_scaled_cast_add_parity_f32_f32(
    cuda: &Device,
    scaling: f64,
    basev: &[f32],
    loraev: &[f32],
) {
    let cpu = Device::Cpu;
    let n = basev.len();

    let base_cpu = Var::from_tensor(&Tensor::from_slice(basev, (n,), &cpu).unwrap()).unwrap();
    let lora_cpu = Var::from_tensor(&Tensor::from_slice(loraev, (n,), &cpu).unwrap()).unwrap();
    let out_cpu = scaled_cast_add(scaling, &base_cpu, &lora_cpu).unwrap();
    let grads_cpu = out_cpu.backward().unwrap();

    let base_gpu = Var::from_tensor(&Tensor::from_slice(basev, (n,), cuda).unwrap()).unwrap();
    let lora_gpu = Var::from_tensor(&Tensor::from_slice(loraev, (n,), cuda).unwrap()).unwrap();
    let out_gpu = scaled_cast_add(scaling, &base_gpu, &lora_gpu).unwrap();
    let grads_gpu = out_gpu.backward().unwrap();

    let out_cpu_v: Vec<f32> = out_cpu.to_vec1().unwrap();
    let out_gpu_v: Vec<f32> = out_gpu.to_device(&cpu).unwrap().to_vec1().unwrap();
    assert_eq!(out_cpu_v.len(), n);
    assert_eq!(
        out_gpu_v.len(),
        n,
        "GPU forward output length mismatch (got {}, expected {n})",
        out_gpu_v.len()
    );
    for (i, (c, g)) in out_cpu_v.iter().zip(out_gpu_v.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "fwd[{i}]: cpu {c} vs cuda {g} (scaling={scaling}, n={n})"
        );
    }

    let d_base_cpu: Vec<f32> = grads_cpu.get(&base_cpu).unwrap().to_vec1().unwrap();
    let d_base_gpu: Vec<f32> = grads_gpu
        .get(&base_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_vec1()
        .unwrap();
    let d_lora_cpu: Vec<f32> = grads_cpu.get(&lora_cpu).unwrap().to_vec1().unwrap();
    let d_lora_gpu: Vec<f32> = grads_gpu
        .get(&lora_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(d_base_cpu.len(), n);
    assert_eq!(d_base_gpu.len(), n, "d_base GPU length mismatch");
    assert_eq!(d_lora_cpu.len(), n);
    assert_eq!(d_lora_gpu.len(), n, "d_lora GPU length mismatch");
    for (i, (c, g)) in d_base_cpu.iter().zip(d_base_gpu.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "d_base[{i}]: cpu {c} vs cuda {g}"
        );
    }
    for (i, (c, g)) in d_lora_cpu.iter().zip(d_lora_gpu.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "d_lora[{i}]: cpu {c} vs cuda {g}"
        );
    }
}

/// The `BF16` base / `F32` lora combination — the one the fine-tune bench
/// (bf16 backbone) actually dispatches through.
fn assert_scaled_cast_add_parity_bf16_base(
    cuda: &Device,
    scaling: f64,
    basev: &[f32],
    loraev: &[f32],
) {
    let cpu = Device::Cpu;
    let n = basev.len();
    let base_bf16: Vec<bf16> = basev.iter().map(|&v| bf16::from_f32(v)).collect();

    let base_cpu = Tensor::from_slice(&base_bf16, (n,), &cpu).unwrap();
    let lora_cpu = Tensor::from_slice(loraev, (n,), &cpu).unwrap();
    let out_cpu: Vec<f32> = scaled_cast_add(scaling, &base_cpu, &lora_cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap();

    let base_gpu = Tensor::from_slice(&base_bf16, (n,), cuda).unwrap();
    let lora_gpu = Tensor::from_slice(loraev, (n,), cuda).unwrap();
    let out_gpu: Vec<f32> = scaled_cast_add(scaling, &base_gpu, &lora_gpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_vec1()
        .unwrap();

    assert_eq!(out_cpu.len(), n);
    assert_eq!(
        out_gpu.len(),
        n,
        "bf16-base GPU forward output length mismatch (got {}, expected {n})",
        out_gpu.len()
    );
    // Same accumulation semantics on both devices (round-to-bf16-then-add,
    // per this op's module doc) — fmad-class ~1-ULP-at-bf16 differences
    // are the only expected source of divergence between CPU and CUDA.
    for (i, (c, g)) in out_cpu.iter().zip(out_gpu.iter()).enumerate() {
        let ulp = 2.0f32.powi(-7) * c.abs().max(*g).max(1.0);
        assert!(
            (c - g).abs() <= 2.0 * ulp,
            "bf16-base fwd[{i}]: cpu {c} vs cuda {g}"
        );
    }
}

#[test]
fn scaled_cast_add_parity_contiguous_small() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let base = fixture(8, 1.0);
    let lora = fixture(8, 2.0);
    assert_scaled_cast_add_parity_f32_f32(&cuda, 1.75, &base, &lora);
    assert_scaled_cast_add_parity_bf16_base(&cuda, 1.75, &base, &lora);
}

#[test]
fn scaled_cast_add_parity_narrowed_with_nonzero_offset() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    // Build a [3, 8] tensor and narrow to the middle row — the missing-
    // offset class this crate's own review found in `Axpy`'s CUDA arm.
    let base_all = fixture(24, 3.0);
    let lora_all = fixture(24, 4.0);
    let cpu = Device::Cpu;

    let base_cpu = Tensor::from_slice(&base_all, (3, 8), &cpu)
        .unwrap()
        .narrow(0, 1, 1)
        .unwrap()
        .flatten_all()
        .unwrap();
    let lora_cpu = Tensor::from_slice(&lora_all, (3, 8), &cpu)
        .unwrap()
        .narrow(0, 1, 1)
        .unwrap()
        .flatten_all()
        .unwrap();
    assert!(base_cpu.is_contiguous());
    assert_ne!(base_cpu.layout().start_offset(), 0);

    let base_gpu = Tensor::from_slice(&base_all, (3, 8), &cuda)
        .unwrap()
        .narrow(0, 1, 1)
        .unwrap()
        .flatten_all()
        .unwrap();
    let lora_gpu = Tensor::from_slice(&lora_all, (3, 8), &cuda)
        .unwrap()
        .narrow(0, 1, 1)
        .unwrap()
        .flatten_all()
        .unwrap();
    assert!(base_gpu.is_contiguous());
    assert_ne!(base_gpu.layout().start_offset(), 0);

    let out_cpu: Vec<f32> = scaled_cast_add(2.0, &base_cpu, &lora_cpu)
        .unwrap()
        .to_vec1()
        .unwrap();
    let out_gpu: Vec<f32> = scaled_cast_add(2.0, &base_gpu, &lora_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(out_cpu.len(), 8);
    assert_eq!(
        out_gpu.len(),
        8,
        "narrowed GPU forward output length mismatch (got {}, expected 8)",
        out_gpu.len()
    );
    for (i, (c, g)) in out_cpu.iter().zip(out_gpu.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "narrowed fwd[{i}]: cpu {c} vs cuda {g}"
        );
    }
    let expected: Vec<f32> = base_all[8..16]
        .iter()
        .zip(lora_all[8..16].iter())
        .map(|(&b, &l)| b + 2.0 * l)
        .collect();
    for (i, (g, e)) in out_gpu.iter().zip(expected.iter()).enumerate() {
        assert!(
            ((*g - *e).abs() as f64) <= F32_TOL,
            "narrowed fwd[{i}] vs hand-computed: cuda {g} vs {e}"
        );
    }
}

#[test]
fn scaled_cast_add_parity_empty() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let cpu = Device::Cpu;
    let base_cpu = Tensor::from_slice(&[] as &[f32], (0,), &cpu).unwrap();
    let lora_cpu = Tensor::from_slice(&[] as &[f32], (0,), &cpu).unwrap();
    let base_gpu = Tensor::from_slice(&[] as &[f32], (0,), &cuda).unwrap();
    let lora_gpu = Tensor::from_slice(&[] as &[f32], (0,), &cuda).unwrap();

    let out_cpu: Vec<f32> = scaled_cast_add(1.0, &base_cpu, &lora_cpu)
        .unwrap()
        .to_vec1()
        .unwrap();
    // Must not attempt an illegal (0, 1, 1) launch grid.
    let out_gpu: Vec<f32> = scaled_cast_add(1.0, &base_gpu, &lora_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(out_cpu.is_empty());
    assert!(out_gpu.is_empty());
}

#[test]
fn scaled_cast_add_parity_multi_block_exact_multiple_of_block_size() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let base = fixture(4096, 5.0);
    let lora = fixture(4096, 6.0);
    assert_scaled_cast_add_parity_f32_f32(&cuda, 0.5, &base, &lora);
}

#[test]
fn scaled_cast_add_parity_multi_block_not_a_multiple_of_block_size() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let base = fixture(2000, 7.0);
    let lora = fixture(2000, 8.0);
    assert_scaled_cast_add_parity_f32_f32(&cuda, -2.25, &base, &lora);
    assert_scaled_cast_add_parity_bf16_base(&cuda, -2.25, &base, &lora);
}

// =======================================================================
// Philox4x32-10 KAT-on-CUDA: Random123's published known-answer test
// vectors run through `PhiloxKatProbe`'s CUDA arm (`dropout.cu`'s
// `philox_kat` device function) and asserted bit-identical to the exact
// same vectors `jammi_kernels::philox`'s own CPU tests assert. THIS is
// the proof the C7 contract requires: the Rust CPU port and the CUDA
// device function compute the identical `u32` stream, not merely "both
// happen to compile" — see `crate::philox`'s module doc (embedded via
// `jammi_kernels::philox`, re-exercised here) and `ops::PhiloxKatProbe`'s
// doc for why this goes through the ordinary `apply1` dispatch path
// rather than a bespoke raw-buffer-download API.
// =======================================================================

#[test]
fn philox_kat_vectors_match_on_cuda() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let cpu = Device::Cpu;
    let dummy = Tensor::from_slice(&[0.0f32], (1,), &cuda).unwrap();
    // The three vectors are `jammi_kernels::philox::tests`'s own —
    // Random123 `tests/kat_vectors`, `philox4x32 10` (fetched 2026-08-24
    // from `DEShawResearch/random123@main`).
    let vectors: [([u32; 4], [u32; 2], [u32; 4]); 3] = [
        (
            [0, 0, 0, 0],
            [0, 0],
            [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8],
        ),
        (
            [0xffffffff; 4],
            [0xffffffff, 0xffffffff],
            [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd],
        ),
        (
            [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
            [0xa4093822, 0x299f31d0],
            [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1],
        ),
    ];
    for (ctr, key, expected) in vectors {
        let op = PhiloxKatProbe::new(ctr, key);
        let out: Vec<u32> = apply1(&dummy, op)
            .unwrap()
            .to_device(&cpu)
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(
            out,
            expected.to_vec(),
            "philox4x32-10 KAT mismatch on CUDA for ctr={ctr:?} key={key:?}"
        );
    }
}

// =======================================================================
// DropoutFused CPU<->CUDA parity: the KEEP/DROP decision (an INTEGER
// comparison over the identical Philox stream, just proven bit-identical
// above) must match exactly, and the applied scale (`__fmul_rn` on CUDA,
// a lone `f32 * f32` on CPU — the SAME single IEEE-754 round-to-nearest
// operation) must match within this file's ordinary `F32_TOL`/bf16-ULP
// bounds (an fmad-contraction-class gap CANNOT occur here: there is no
// neighboring add for either side's multiply to fuse with). Covers the
// same divergence-prone classes as every other op's suite in this file
// (contiguous, narrowed-with-nonzero-offset, empty) PLUS this op's own
// domain edge: p == 0.0 must be a bit-exact no-op on BOTH devices.
// =======================================================================

fn dropout(
    seed: u64,
    layer_id: u32,
    forward_idx: u32,
    p: f32,
    x: &Tensor,
) -> candle_core::Result<Tensor> {
    let op = DropoutFused::new(seed, layer_id, forward_idx, p)?;
    apply1(x, op)
}

/// Decision + applied-value parity, fwd AND bwd, for a fixed
/// `(seed, layer_id, forward_idx, p)` over a large tensor — oracle 2 (CPU
/// mask == CUDA mask bit-for-bit at the DECISION level) plus the applied
/// scale's own tolerance.
fn assert_dropout_parity_f32(
    cuda: &Device,
    seed: u64,
    layer_id: u32,
    forward_idx: u32,
    p: f32,
    xv: &[f32],
) {
    let cpu = Device::Cpu;
    let n = xv.len();

    let x_cpu = Var::from_tensor(&Tensor::from_slice(xv, (n,), &cpu).unwrap()).unwrap();
    let out_cpu = dropout(seed, layer_id, forward_idx, p, &x_cpu).unwrap();
    let grads_cpu = out_cpu.backward().unwrap();

    let x_gpu = Var::from_tensor(&Tensor::from_slice(xv, (n,), cuda).unwrap()).unwrap();
    let out_gpu = dropout(seed, layer_id, forward_idx, p, &x_gpu).unwrap();
    let grads_gpu = out_gpu.backward().unwrap();

    let out_cpu_v: Vec<f32> = out_cpu.to_vec1().unwrap();
    let out_gpu_v: Vec<f32> = out_gpu.to_device(&cpu).unwrap().to_vec1().unwrap();
    assert_eq!(out_cpu_v.len(), n);
    assert_eq!(out_gpu_v.len(), n, "dropout GPU fwd length mismatch");
    for (i, (c, g)) in out_cpu_v.iter().zip(out_gpu_v.iter()).enumerate() {
        // The DECISION must match exactly: a kept-on-one-device,
        // dropped-on-the-other element is a hard failure, not a tolerance
        // question (the whole point of the shared Philox stream + integer
        // threshold, proven above).
        assert_eq!(
            *c == 0.0,
            *g == 0.0,
            "dropout fwd[{i}]: KEEP/DROP decision disagrees, cpu {c} vs cuda {g}"
        );
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "dropout fwd[{i}]: cpu {c} vs cuda {g}"
        );
    }

    let dx_cpu: Vec<f32> = grads_cpu.get(&x_cpu).unwrap().to_vec1().unwrap();
    let dx_gpu: Vec<f32> = grads_gpu
        .get(&x_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(dx_cpu.len(), n);
    assert_eq!(dx_gpu.len(), n, "dropout GPU dx length mismatch");
    for (i, (c, g)) in dx_cpu.iter().zip(dx_gpu.iter()).enumerate() {
        assert_eq!(
            *c == 0.0,
            *g == 0.0,
            "dropout dx[{i}]: KEEP/DROP decision disagrees on the regenerated backward, \
             cpu {c} vs cuda {g}"
        );
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "dropout dx[{i}]: cpu {c} vs cuda {g}"
        );
    }
}

#[test]
fn dropout_parity_contiguous_small() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let x = fixture(2048, 1.0);
    assert_dropout_parity_f32(&cuda, 4242, 7, 3, 0.05, &x);
}

#[test]
fn dropout_parity_p_zero_is_bit_exact_no_op_on_both_devices() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let cpu = Device::Cpu;
    let x = fixture(512, 2.0);
    let x_cpu = Tensor::from_slice(&x, (512,), &cpu).unwrap();
    let x_gpu = Tensor::from_slice(&x, (512,), &cuda).unwrap();
    let out_cpu: Vec<f32> = dropout(1, 0, 0, 0.0, &x_cpu).unwrap().to_vec1().unwrap();
    let out_gpu: Vec<f32> = dropout(1, 0, 0, 0.0, &x_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(out_cpu, x, "p=0.0 must be a bit-exact no-op on CPU");
    assert_eq!(out_gpu, x, "p=0.0 must be a bit-exact no-op on CUDA");
}

#[test]
fn dropout_parity_narrowed_with_nonzero_offset() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let cpu = Device::Cpu;
    let base = fixture(3 * 256, 3.0);

    let x_cpu = Tensor::from_slice(&base, (3, 256), &cpu)
        .unwrap()
        .narrow(0, 1, 1)
        .unwrap()
        .flatten_all()
        .unwrap();
    assert!(x_cpu.is_contiguous());
    assert_ne!(x_cpu.layout().start_offset(), 0);
    let x_gpu = Tensor::from_slice(&base, (3, 256), &cuda)
        .unwrap()
        .narrow(0, 1, 1)
        .unwrap()
        .flatten_all()
        .unwrap();

    let out_cpu: Vec<f32> = dropout(99, 2, 1, 0.3, &x_cpu).unwrap().to_vec1().unwrap();
    let out_gpu: Vec<f32> = dropout(99, 2, 1, 0.3, &x_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(out_cpu.len(), 256);
    assert_eq!(
        out_gpu.len(),
        256,
        "narrowed dropout GPU fwd length mismatch"
    );
    for (i, (c, g)) in out_cpu.iter().zip(out_gpu.iter()).enumerate() {
        assert_eq!(
            *c == 0.0,
            *g == 0.0,
            "narrowed dropout fwd[{i}]: KEEP/DROP decision disagrees"
        );
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "narrowed dropout fwd[{i}]: cpu {c} vs cuda {g}"
        );
    }
}

#[test]
fn dropout_parity_empty() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let cpu = Device::Cpu;
    let x_cpu = Tensor::from_slice(&[] as &[f32], (0,), &cpu).unwrap();
    let x_gpu = Tensor::from_slice(&[] as &[f32], (0,), &cuda).unwrap();
    let out_cpu: Vec<f32> = dropout(1, 0, 0, 0.3, &x_cpu).unwrap().to_vec1().unwrap();
    // Must not attempt an illegal zero-block launch.
    let out_gpu: Vec<f32> = dropout(1, 0, 0, 0.3, &x_gpu)
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(out_cpu.is_empty());
    assert!(out_gpu.is_empty());
}

#[test]
fn dropout_parity_multi_block_not_a_multiple_of_block_size() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    // 2000 spans multiple 256-thread blocks with a partial last block.
    let x = fixture(2000, 9.0);
    assert_dropout_parity_f32(&cuda, 7, 1, 12, 0.4, &x);
}

// =======================================================================
// LowRankResidualLinear CPU<->CUDA parity: forward + all three backward outputs
// (dx, dw, dab), at the two dtype combinations `jammi-lora`'s admission
// predicate reaches ((F32,F32,F32) and (BF16,BF16,F32)), plus a dropout
// leg and a narrowed-with-nonzero-offset `x`. `BF16` here is the ONLY
// place this op's base-matmul dtype is actually exercised end to end —
// candle-core 0.11.0's CPU backend has no `BF16` matmul (see
// `ops::low_rank_residual_linear`'s module doc), so the "CPU" side of this comparison
// runs the base GEMM at `F32` (a bit-exact cast-up of the same `BF16`
// input) while the CUDA side runs the REAL `BF16` cuBLAS path — this is a
// weaker parity claim than the other bf16 legs above (comparing two
// DIFFERENT dtype executions, not the same dtype on two devices), stated
// explicitly rather than silently reused from the `f32_f32` helper's
// shape.
// =======================================================================

/// `f32` unit roundoff, `2^-24` (Higham, *Accuracy and Stability of
/// Numerical Algorithms*, 2nd ed., Theorem 4.2 — see
/// `lora_linear_oracles.rs`'s `derived_dot_product_tolerance` doc for the
/// full citation this mirrors; generalized here to a `unit_roundoff`
/// parameter so [`higham_bound`] covers both this constant and
/// [`BF16_U`]).
const F32_U: f64 = 1.0 / 16_777_216.0; // 2^-24

/// `bf16` unit roundoff, `2^-8` (7 explicit mantissa bits + the implicit
/// leading one — `half::bf16`'s own layout). Used ONLY to bound the ONE
/// place this op's CUDA arm actually rounds a value TO `bf16` (the base
/// GEMM's own tensor-core output store, and the epilogue's cast of the
/// scaled LoRA delta) — never as an accumulation-order bound: the base
/// GEMM's OWN accumulation happens at `f32` precision on tensor cores
/// regardless of `bf16` storage, per `CUBLAS_COMPUTE_32F` — candle-core
/// 0.11.0's `cuda_backend::gemm_strided_batched_bf16` (registry source,
/// `~/.cargo/registry/src/.../candle-core-0.11.0/src/cuda_backend/
/// mod.rs:2685`) defaults `gemm_reduced_precision_bf16()` to `false`
/// (`MM_BF16_REDUCED_PRECISION: AtomicBool = AtomicBool::new(false)`,
/// never set `true` anywhere in this workspace — verified by reading
/// both), i.e. the FULL-precision `f32`-accumulate compute type, not the
/// further-reduced `CUBLAS_COMPUTE_32F_FAST_16BF` mode.
const BF16_U: f64 = 1.0 / 256.0; // 2^-8

/// Higham (2002) Theorem 4.2: a recursively-summed `n`-term sum deviates
/// from the true sum by at most `(n-1) * u * sum(|x_i|)`; bounding
/// `sum(|x_i|) <= n * max_term_magnitude` gives the `n^2` shape below. Two
/// INDEPENDENTLY-chosen valid summation orders (cuBLAS's tensor-core
/// tiling vs the CPU `gemm` crate's own blocking) can therefore differ
/// from EACH OTHER by up to twice one order's own worst-case drift from
/// the infinite-precision sum — the leading `2.0`. Mirrors
/// `lora_linear_oracles.rs`'s `derived_dot_product_tolerance` exactly
/// (same citation, same shape).
fn higham_bound(n: usize, max_term_magnitude: f64, unit_roundoff: f64) -> f64 {
    2.0 * (n as f64).powi(2) * unit_roundoff * max_term_magnitude
}

/// A single "round to `bf16`" bound: `bf16` has 8 significant bits
/// (7 explicit mantissa bits + the implicit leading one), so round-to-
/// nearest storage of a value `v` introduces at most `0.5 * 2^-7 * |v|`
/// == `BF16_U * |v|` relative error. Doubled here as an explicit safety
/// margin over that textbook half-ULP bound: this test does not control
/// cuBLAS's own internal reduction-tree implementation closely enough to
/// prove it hits the textbook bound EXACTLY, only that it runs at
/// `CUBLAS_COMPUTE_32F` (see [`BF16_U`]'s own doc for how that was
/// verified) rather than a further-reduced mode.
fn bf16_round_bound(value: f64) -> f64 {
    2.0 * BF16_U * value.abs()
}

/// A conservative, DERIVED (not tuned-to-pass) absolute tolerance
/// covering every CPU<->CUDA `f32` comparison this section's tests make
/// (`fwd`, `dx`, `dw`, `da`, `db`): each is the output of one or two
/// chained GEMM reductions over this op's own dimensions (`rows`, `inf`,
/// `outf`; `rank` is always this op's smallest dimension, so it never
/// drives the bound) — see this file's leading `LowRankResidualLinear`
/// section comment for the full forward/backward reduction-depth
/// enumeration (`dx`'s `d_x_lora` branch alone chains TWO GEMMs: `g =
/// d_lora @ B` over `outf`, then `d_xd = g @ A` over `rank`; `dw`/`da`/
/// `db` each reduce over `rows`). Rather than deriving five separate
/// per-output bounds, this uses the LARGEST reduction depth appearing
/// ANYWHERE in that graph (`max(rows, inf, outf)`) and a `chain_factor`
/// of `3.0`: one contribution for the output's own reduction, plus up to
/// two chained upstream reductions whose own summation-order error
/// propagates through a further multiply-reduce (no chain in this op's
/// backward is longer than two GEMMs deep). `amplitude` is derived from
/// the ACTUAL fixture data passed in (the max absolute value across every
/// operand slice), not hardcoded, so this stays correct if a caller's
/// fixture amplitude ever changes. Looser than a per-output derivation
/// would be, but still tight enough to catch a real bug: a dropped or
/// mis-indexed term produces an error on the order of a SINGLE term's own
/// magnitude (`amplitude^2`), orders of magnitude above this bound at any
/// `n > 1` — the EXACT-integer legs below close the gap this looseness
/// leaves for a bug small enough to hide under it.
fn lora_linear_parity_tolerance(rows: usize, inf: usize, outf: usize, values: &[&[f32]]) -> f64 {
    let amplitude = values
        .iter()
        .flat_map(|s| s.iter())
        .fold(0.0f64, |acc, &v| acc.max(f64::from(v.abs())));
    let n = rows.max(inf).max(outf);
    let max_term_magnitude = amplitude * amplitude;
    let chain_factor = 3.0;
    chain_factor * higham_bound(n, max_term_magnitude, F32_U)
}

/// Packs `a`/`b` into `ab`'s row-packed `[in + out, rank]` layout — see
/// `jammi_kernels::ops::low_rank_residual_linear`'s module doc, "the packed-`ab` GEMM
/// eligibility problem". No `.contiguous()` call needed before packing
/// (`Tensor::cat`'s dim-0 path handles `a.t()`'s non-contiguous view via
/// each arg's own `Layout`) — unlike the column-packed layout this
/// replaced, which needed one for `B^T`.
fn pack_ab(a: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    Tensor::cat(&[&a.t()?, b], 0)
}

/// Bundles [`lora_linear`]'s construction data into one value — the
/// small-params-struct fix for `clippy::too_many_arguments` (no `#[allow]`
/// per the fix contract) rather than a 9-positional-argument function.
#[derive(Clone, Copy)]
struct LoraLinearParams {
    scale: f32,
    inf: usize,
    outf: usize,
    r: usize,
    dropout: Option<DropoutKey>,
    dweight_needed: bool,
}

fn lora_linear(
    x: &Tensor,
    w: &Tensor,
    ab: &Tensor,
    p: LoraLinearParams,
) -> candle_core::Result<Tensor> {
    let op = LowRankResidualLinear::new(p.scale, p.inf, p.outf, p.r, p.dropout, p.dweight_needed)?;
    x.apply_op3(w, ab, op)
}

#[allow(clippy::too_many_arguments)]
fn assert_lora_linear_parity_f32(
    cuda: &Device,
    rows: usize,
    inf: usize,
    outf: usize,
    r: usize,
    scale: f32,
    xv: &[f32],
    wv: &[f32],
    av: &[f32],
    bv: &[f32],
) {
    let cpu = Device::Cpu;

    let x_cpu = Var::from_tensor(&Tensor::from_slice(xv, (rows, inf), &cpu).unwrap()).unwrap();
    let w_cpu = Var::from_tensor(&Tensor::from_slice(wv, (outf, inf), &cpu).unwrap()).unwrap();
    let a_cpu = Var::from_tensor(&Tensor::from_slice(av, (r, inf), &cpu).unwrap()).unwrap();
    let b_cpu = Var::from_tensor(&Tensor::from_slice(bv, (outf, r), &cpu).unwrap()).unwrap();
    let ab_cpu = pack_ab(a_cpu.as_tensor(), b_cpu.as_tensor()).unwrap();
    let out_cpu = lora_linear(
        x_cpu.as_tensor(),
        w_cpu.as_tensor(),
        &ab_cpu,
        LoraLinearParams {
            scale,
            inf,
            outf,
            r,
            dropout: None,
            dweight_needed: true,
        },
    )
    .unwrap();
    let grads_cpu = out_cpu.sum_all().unwrap().backward().unwrap();

    let x_gpu = Var::from_tensor(&Tensor::from_slice(xv, (rows, inf), cuda).unwrap()).unwrap();
    let w_gpu = Var::from_tensor(&Tensor::from_slice(wv, (outf, inf), cuda).unwrap()).unwrap();
    let a_gpu = Var::from_tensor(&Tensor::from_slice(av, (r, inf), cuda).unwrap()).unwrap();
    let b_gpu = Var::from_tensor(&Tensor::from_slice(bv, (outf, r), cuda).unwrap()).unwrap();
    let ab_gpu = pack_ab(a_gpu.as_tensor(), b_gpu.as_tensor()).unwrap();
    let out_gpu = lora_linear(
        x_gpu.as_tensor(),
        w_gpu.as_tensor(),
        &ab_gpu,
        LoraLinearParams {
            scale,
            inf,
            outf,
            r,
            dropout: None,
            dweight_needed: true,
        },
    )
    .unwrap();
    let grads_gpu = out_gpu.sum_all().unwrap().backward().unwrap();

    // A single DERIVED (Higham Thm 4.2, see `lora_linear_parity_tolerance`'s
    // own doc) tolerance covers `fwd` and every backward output below: all
    // five are chained-GEMM outputs over this call's own `(rows, inf,
    // outf)`, and the fixture's amplitude is read directly off the actual
    // data (not hardcoded), so this stays correct across every caller of
    // this function.
    let tol = lora_linear_parity_tolerance(rows, inf, outf, &[xv, wv, av, bv]);

    let out_cpu_v: Vec<f32> = out_cpu.flatten_all().unwrap().to_vec1().unwrap();
    let out_gpu_v: Vec<f32> = out_gpu
        .flatten_all()
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(out_cpu_v.len(), rows * outf);
    assert_eq!(out_gpu_v.len(), rows * outf, "GPU forward length mismatch");
    for (i, (c, g)) in out_cpu_v.iter().zip(out_gpu_v.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= tol,
            "fwd[{i}]: cpu {c} vs cuda {g} (tol {tol})"
        );
    }

    let check = |name: &str, cpu_t: &Tensor, gpu_t: &Tensor| {
        let c: Vec<f32> = cpu_t.flatten_all().unwrap().to_vec1().unwrap();
        let g: Vec<f32> = gpu_t
            .flatten_all()
            .unwrap()
            .to_device(&cpu)
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(c.len(), g.len(), "{name}: length mismatch");
        for (i, (cv, gv)) in c.iter().zip(g.iter()).enumerate() {
            assert!(
                ((*cv - *gv).abs() as f64) <= tol,
                "{name}[{i}]: cpu {cv} vs cuda {gv} (tol {tol})"
            );
        }
    };
    check(
        "dx",
        grads_cpu.get(&x_cpu).unwrap(),
        grads_gpu.get(&x_gpu).unwrap(),
    );
    check(
        "dw",
        grads_cpu.get(&w_cpu).unwrap(),
        grads_gpu.get(&w_gpu).unwrap(),
    );
    check(
        "da",
        grads_cpu.get(&a_cpu).unwrap(),
        grads_gpu.get(&a_gpu).unwrap(),
    );
    check(
        "db",
        grads_cpu.get(&b_cpu).unwrap(),
        grads_gpu.get(&b_gpu).unwrap(),
    );
}

#[test]
fn lora_linear_parity_f32_contiguous_small() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let (rows, inf, outf, r) = (6usize, 5usize, 7usize, 3usize);
    let x = fixture(rows * inf, 1.0);
    let w = fixture(outf * inf, 2.0);
    let a = fixture(r * inf, 3.0);
    let b = fixture(outf * r, 4.0);
    assert_lora_linear_parity_f32(&cuda, rows, inf, outf, r, 1.6, &x, &w, &a, &b);
}

#[test]
fn lora_linear_parity_f32_production_width_wqkv() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let (rows, inf, outf, r) = (24 * 128, 1024usize, 3072usize, 16usize);
    let x = fixture(rows * inf, 0.1);
    let w = fixture(outf * inf, 0.2);
    let a = fixture(r * inf, 0.3);
    let b = fixture(outf * r, 0.4);
    assert_lora_linear_parity_f32(&cuda, rows, inf, outf, r, 0.5, &x, &w, &a, &b);
}

#[test]
fn lora_linear_parity_f32_narrowed_with_nonzero_offset() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let cpu = Device::Cpu;
    let (rows, inf, outf, r) = (4usize, 6usize, 5usize, 2usize);
    // Build a [3*rows, inf] tensor and narrow to the middle `rows` block —
    // the missing-offset class this crate's own review found in `Axpy`'s
    // CUDA arm, exercised on this op's `x` argument.
    let x_all = fixture(3 * rows * inf, 5.0);
    let w = fixture(outf * inf, 6.0);
    let a = fixture(r * inf, 7.0);
    let b = fixture(outf * r, 8.0);
    let ab_cpu = pack_ab(
        &Tensor::from_slice(&a, (r, inf), &cpu).unwrap(),
        &Tensor::from_slice(&b, (outf, r), &cpu).unwrap(),
    )
    .unwrap();
    let ab_gpu = pack_ab(
        &Tensor::from_slice(&a, (r, inf), &cuda).unwrap(),
        &Tensor::from_slice(&b, (outf, r), &cuda).unwrap(),
    )
    .unwrap();
    let w_cpu = Tensor::from_slice(&w, (outf, inf), &cpu).unwrap();
    let w_gpu = Tensor::from_slice(&w, (outf, inf), &cuda).unwrap();

    let x_cpu_full = Tensor::from_slice(&x_all, (3 * rows, inf), &cpu).unwrap();
    let x_cpu = x_cpu_full.narrow(0, rows, rows).unwrap();
    assert!(!x_cpu.is_contiguous() || x_cpu.layout().start_offset() != 0);
    let x_gpu_full = Tensor::from_slice(&x_all, (3 * rows, inf), &cuda).unwrap();
    let x_gpu = x_gpu_full.narrow(0, rows, rows).unwrap();

    let params = |scale: f32| LoraLinearParams {
        scale,
        inf,
        outf,
        r,
        dropout: None,
        dweight_needed: false,
    };
    let out_cpu = lora_linear(&x_cpu, &w_cpu, &ab_cpu, params(1.2)).unwrap();
    let out_gpu = lora_linear(&x_gpu, &w_gpu, &ab_gpu, params(1.2)).unwrap();

    let tol = lora_linear_parity_tolerance(rows, inf, outf, &[&x_all, &w, &a, &b]);
    let out_cpu_v: Vec<f32> = out_cpu.flatten_all().unwrap().to_vec1().unwrap();
    let out_gpu_v: Vec<f32> = out_gpu
        .flatten_all()
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(out_gpu_v.len(), rows * outf, "GPU forward length mismatch");
    for (i, (c, g)) in out_cpu_v.iter().zip(out_gpu_v.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= tol,
            "narrowed fwd[{i}]: cpu {c} vs cuda {g} (tol {tol})"
        );
    }
}

/// A GENUINELY non-contiguous `x` (a transposed VIEW, not merely a
/// nonzero-offset contiguous narrow — see
/// `lora_linear_parity_f32_narrowed_with_nonzero_offset` for that class):
/// `ops::low_rank_residual_linear::materialize_contiguous_if_needed` must produce the
/// SAME CUDA result as it does on CPU, at the SAME (transposed) input —
/// the CUDA-only leg the fix contract calls for, since a strided
/// materialize-copy's `to_dtype`-based gather is a DIFFERENT code path per
/// device (`CudaStorage::to_dtype`'s own cast kernel vs the CPU backend's).
#[test]
fn lora_linear_parity_f32_transposed_x_is_materialized_identically_on_both_devices() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let cpu = Device::Cpu;
    let (inf, outf, r) = (6usize, 5usize, 2usize);
    let rows = 4usize;
    let x_t_v = fixture(inf * rows, 13.0); // stored as [inf, rows], then transposed to [rows, inf].
    let w = fixture(outf * inf, 14.0);
    let a = fixture(r * inf, 15.0);
    let b = fixture(outf * r, 16.0);

    let ab_cpu = pack_ab(
        &Tensor::from_slice(&a, (r, inf), &cpu).unwrap(),
        &Tensor::from_slice(&b, (outf, r), &cpu).unwrap(),
    )
    .unwrap();
    let ab_gpu = pack_ab(
        &Tensor::from_slice(&a, (r, inf), &cuda).unwrap(),
        &Tensor::from_slice(&b, (outf, r), &cuda).unwrap(),
    )
    .unwrap();
    let w_cpu = Tensor::from_slice(&w, (outf, inf), &cpu).unwrap();
    let w_gpu = Tensor::from_slice(&w, (outf, inf), &cuda).unwrap();

    let x_cpu = Tensor::from_slice(&x_t_v, (inf, rows), &cpu)
        .unwrap()
        .t()
        .unwrap();
    assert!(!x_cpu.is_contiguous());
    let x_gpu = Tensor::from_slice(&x_t_v, (inf, rows), &cuda)
        .unwrap()
        .t()
        .unwrap();
    assert!(!x_gpu.is_contiguous());

    let params = LoraLinearParams {
        scale: 0.9,
        inf,
        outf,
        r,
        dropout: None,
        dweight_needed: false,
    };
    let out_cpu = lora_linear(&x_cpu, &w_cpu, &ab_cpu, params).unwrap();
    let out_gpu = lora_linear(&x_gpu, &w_gpu, &ab_gpu, params).unwrap();

    let tol = lora_linear_parity_tolerance(rows, inf, outf, &[&x_t_v, &w, &a, &b]);
    let out_cpu_v: Vec<f32> = out_cpu.flatten_all().unwrap().to_vec1().unwrap();
    let out_gpu_v: Vec<f32> = out_gpu
        .flatten_all()
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(out_gpu_v.len(), rows * outf, "GPU forward length mismatch");
    for (i, (c, g)) in out_cpu_v.iter().zip(out_gpu_v.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= tol,
            "transposed-x fwd[{i}]: cpu {c} vs cuda {g} (tol {tol})"
        );
    }
}

#[test]
fn lora_linear_parity_f32_dropout_matches_across_devices() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let cpu = Device::Cpu;
    let (rows, inf, outf, r) = (8usize, 6usize, 5usize, 2usize);
    let key = DropoutKey {
        seed: 123,
        layer_id: 4,
        forward_idx: 2,
        p: 0.3,
    };
    let x = fixture(rows * inf, 9.0);
    let w = fixture(outf * inf, 10.0);
    let a = fixture(r * inf, 11.0);
    let b = fixture(outf * r, 12.0);

    let x_cpu = Tensor::from_slice(&x, (rows, inf), &cpu).unwrap();
    let w_cpu = Tensor::from_slice(&w, (outf, inf), &cpu).unwrap();
    let ab_cpu = pack_ab(
        &Tensor::from_slice(&a, (r, inf), &cpu).unwrap(),
        &Tensor::from_slice(&b, (outf, r), &cpu).unwrap(),
    )
    .unwrap();
    let params = LoraLinearParams {
        scale: 0.7,
        inf,
        outf,
        r,
        dropout: Some(key),
        dweight_needed: false,
    };
    let out_cpu = lora_linear(&x_cpu, &w_cpu, &ab_cpu, params).unwrap();

    let x_gpu = Tensor::from_slice(&x, (rows, inf), &cuda).unwrap();
    let w_gpu = Tensor::from_slice(&w, (outf, inf), &cuda).unwrap();
    let ab_gpu = pack_ab(
        &Tensor::from_slice(&a, (r, inf), &cuda).unwrap(),
        &Tensor::from_slice(&b, (outf, r), &cuda).unwrap(),
    )
    .unwrap();
    let out_gpu = lora_linear(&x_gpu, &w_gpu, &ab_gpu, params).unwrap();

    // Inverted dropout scales a KEPT element by `1/(1-p)` before it ever
    // reaches the LoRA branch's own GEMMs — that inflates the effective
    // per-term magnitude feeding `higham_bound` by the same factor, so the
    // amplitude this derives from `x`'s RAW (pre-dropout) values is scaled
    // up here rather than silently under-counting it.
    let dropout_inv_scale = 1.0 / (1.0 - f64::from(key.p));
    let x_scaled: Vec<f32> = x.iter().map(|&v| v * dropout_inv_scale as f32).collect();
    let tol = lora_linear_parity_tolerance(rows, inf, outf, &[&x_scaled, &w, &a, &b]);
    let out_cpu_v: Vec<f32> = out_cpu.flatten_all().unwrap().to_vec1().unwrap();
    let out_gpu_v: Vec<f32> = out_gpu
        .flatten_all()
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(out_gpu_v.len(), rows * outf, "GPU forward length mismatch");
    for (i, (c, g)) in out_cpu_v.iter().zip(out_gpu_v.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= tol,
            "dropout fwd[{i}]: cpu {c} vs cuda {g} (tol {tol}) — the SAME (seed, layer_id, \
             forward_idx) key must draw the identical mask on both devices \
             (Philox is a pure function of the counter, not device RNG state)"
        );
    }
}

/// `BF16` base on CUDA (the real production dtype pair) — the CPU side
/// runs the base matmul at `F32` (candle-core 0.11.0 has no CPU `BF16`
/// matmul; see this file's own leading comment on this section and
/// `ops::low_rank_residual_linear`'s module doc), so this compares CUDA's REAL `BF16`
/// path against an `F32`-CPU reference built from the SAME bit-pattern
/// values (each `BF16` input is round-tripped through `bf16::from_f32`
/// before either device runs, so both sides start from IDENTICAL
/// bf16-quantized numbers).
///
/// ## Why the bound is keyed to `base`/`delta`'s OWN magnitude, not `out`'s
///
/// A standalone on-device probe (a plain `x_bf16.matmul(&w_bf16.t())`,
/// bypassing this op entirely) against an `f64` "gold" reference built from
/// the SAME bf16-rounded operands measured the CUDA `base` GEMM's own
/// error as CORRECTLY-ROUNDED `bf16` storage behaviour: at this test's
/// `inf = 1024`, `base`'s own magnitude is in the tens of thousands (the
/// ±10-amplitude fixture, summed over 1024 terms), so `bf16`'s 8
/// significant bits put a SINGLE round-to-nearest store's worst case at
/// ~128 absolute (`half_ulp(45000) == 128`) — exactly what the probe
/// measured (max `|cuda - gold| == 128.2`, `cuda_vs_gold[191] == 107.5`),
/// confirming `gemm_reduced_precision_bf16() == false` (full `f32`
/// accumulate, see [`BF16_U`]'s doc) and NOT a functional bug.
///
/// This fixture's `base` and `scale * delta` happen to nearly CANCEL
/// (`base[191] ~= -45163`, `delta_scaled[191] ~= +47549`, `out[191] ~=
/// 2385`) — family K: a bound proportional to `out`'s own (cancelled, much
/// smaller) magnitude is the WRONG shape of bound for a computation this
/// close to catastrophic cancellation; `base`'s ~128 absolute error, which
/// is tiny relative to `base`'s own ~45000 magnitude, becomes a spurious
/// ~5% "error" once measured against the cancelled ~2385 output instead.
/// This bounds each term by ITS OWN magnitude via [`bf16_round_bound`]
/// instead, computed from `base`/`delta` pieces built SEPARATELY below (not
/// from `out_cpu` alone).
#[test]
fn lora_linear_parity_bf16_base_production_width() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let cpu = Device::Cpu;
    let (rows, inf, outf, r) = (256usize, 1024usize, 3072usize, 16usize);
    let scale = 0.5f32;

    let xv = fixture(rows * inf, 1.0);
    let wv = fixture(outf * inf, 2.0);
    let av = fixture(r * inf, 3.0);
    let bv = fixture(outf * r, 4.0);
    let x_bf16: Vec<bf16> = xv.iter().map(|&v| bf16::from_f32(v)).collect();
    let w_bf16: Vec<bf16> = wv.iter().map(|&v| bf16::from_f32(v)).collect();
    // Reconstruct the exact f32 values BF16 quantization produced, so the
    // CPU F32 reference and the CUDA BF16 run start from identical numbers.
    let x_requantized: Vec<f32> = x_bf16.iter().map(|v| v.to_f32()).collect();
    let w_requantized: Vec<f32> = w_bf16.iter().map(|v| v.to_f32()).collect();

    let x_cpu = Tensor::from_slice(&x_requantized, (rows, inf), &cpu).unwrap();
    let w_cpu = Tensor::from_slice(&w_requantized, (outf, inf), &cpu).unwrap();
    let a_cpu = Tensor::from_slice(&av, (r, inf), &cpu).unwrap();
    let b_cpu = Tensor::from_slice(&bv, (outf, r), &cpu).unwrap();
    let ab_cpu = pack_ab(&a_cpu, &b_cpu).unwrap();
    let params = LoraLinearParams {
        scale,
        inf,
        outf,
        r,
        dropout: None,
        dweight_needed: false,
    };
    let out_cpu = lora_linear(&x_cpu, &w_cpu, &ab_cpu, params).unwrap();

    // The SAME two pieces `out_cpu` sums, kept separate so the bound below
    // can be sized to EACH term's own magnitude rather than their (possibly
    // near-cancelling) combined value — see this test's own doc.
    let base_only_cpu: Vec<f32> = x_cpu
        .matmul(&w_cpu.t().unwrap())
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let delta_scaled_cpu: Vec<f32> = x_cpu
        .matmul(&a_cpu.t().unwrap())
        .unwrap()
        .matmul(&b_cpu.t().unwrap())
        .unwrap()
        .affine(f64::from(scale), 0.0)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let x_gpu = Tensor::from_slice(&x_bf16, (rows, inf), &cuda).unwrap();
    let w_gpu = Tensor::from_slice(&w_bf16, (outf, inf), &cuda).unwrap();
    let ab_gpu = pack_ab(
        &Tensor::from_slice(&av, (r, inf), &cuda).unwrap(),
        &Tensor::from_slice(&bv, (outf, r), &cuda).unwrap(),
    )
    .unwrap();
    let out_gpu = lora_linear(&x_gpu, &w_gpu, &ab_gpu, params).unwrap();

    let out_cpu_v: Vec<f32> = out_cpu.flatten_all().unwrap().to_vec1().unwrap();
    let out_gpu_v: Vec<f32> = out_gpu
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(out_gpu_v.len(), rows * outf, "GPU forward length mismatch");
    // A small floor covers the `f32`-only parts of this computation (the
    // LoRA GEMMs themselves, and the base GEMM's own negligible `f32`
    // summation-order noise — measured ~0.008 by the same standalone probe
    // cited in this test's doc).
    let abs_floor = 1e-1f64;
    for (i, (c, g)) in out_cpu_v.iter().zip(out_gpu_v.iter()).enumerate() {
        let bound = bf16_round_bound(f64::from(base_only_cpu[i]))
            + bf16_round_bound(f64::from(delta_scaled_cpu[i]))
            + abs_floor;
        assert!(
            f64::from(*c - *g).abs() <= bound,
            "bf16-base fwd[{i}]: cpu(f32 ref) {c} vs cuda(bf16) {g} (bound {bound}, \
             base_only {} delta_scaled {})",
            base_only_cpu[i],
            delta_scaled_cpu[i]
        );
    }
}

/// The bit-exact counterpart the loose, DERIVED tolerance above cannot
/// stand in for: [`lora_linear_parity_tolerance`]'s bound is sized to
/// catch a real bug (an error on the order of a single term's own
/// magnitude), but a bug small enough to hide under it — a half-term drop,
/// a single row read from the wrong offset — needs an assertion with NO
/// slack at all. [`exact_fixture`]'s small-integer values make that
/// possible at PRODUCTION width: every intermediate sum here stays a
/// small exact integer, so CPU and CUDA MUST agree bit-for-bit (`tol ==
/// 0.0`) regardless of which valid summation order either `gemm`
/// implementation picks — see `exact_fixture`'s own doc.
#[test]
fn lora_linear_parity_f32_exact_integer_fixture_is_bit_exact() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let (rows, inf, outf, r) = (24usize, 1024usize, 3072usize, 16usize);
    let x = exact_fixture(rows * inf, 1);
    let w = exact_fixture(outf * inf, 2);
    let a = exact_fixture(r * inf, 3);
    let b = exact_fixture(outf * r, 4);
    // scale = 2.0: exact in binary, so the epilogue's own multiply
    // introduces no rounding of its own either.
    assert_lora_linear_parity_f32_bit_exact(&cuda, rows, inf, outf, r, 2.0, &x, &w, &a, &b);
}

/// Bit-exact (`tol == 0.0`) sibling of [`assert_lora_linear_parity_f32`],
/// for [`exact_fixture`]-only callers — kept as a SEPARATE function (not a
/// `tol` parameter on the loose one) so which legs claim bit-exactness is
/// visible at every call site, not buried in an argument.
#[allow(clippy::too_many_arguments)]
fn assert_lora_linear_parity_f32_bit_exact(
    cuda: &Device,
    rows: usize,
    inf: usize,
    outf: usize,
    r: usize,
    scale: f32,
    xv: &[f32],
    wv: &[f32],
    av: &[f32],
    bv: &[f32],
) {
    let cpu = Device::Cpu;

    let x_cpu = Var::from_tensor(&Tensor::from_slice(xv, (rows, inf), &cpu).unwrap()).unwrap();
    let w_cpu = Var::from_tensor(&Tensor::from_slice(wv, (outf, inf), &cpu).unwrap()).unwrap();
    let a_cpu = Var::from_tensor(&Tensor::from_slice(av, (r, inf), &cpu).unwrap()).unwrap();
    let b_cpu = Var::from_tensor(&Tensor::from_slice(bv, (outf, r), &cpu).unwrap()).unwrap();
    let ab_cpu = pack_ab(a_cpu.as_tensor(), b_cpu.as_tensor()).unwrap();
    let params = LoraLinearParams {
        scale,
        inf,
        outf,
        r,
        dropout: None,
        dweight_needed: true,
    };
    let out_cpu = lora_linear(x_cpu.as_tensor(), w_cpu.as_tensor(), &ab_cpu, params).unwrap();
    let grads_cpu = out_cpu.sum_all().unwrap().backward().unwrap();

    let x_gpu = Var::from_tensor(&Tensor::from_slice(xv, (rows, inf), cuda).unwrap()).unwrap();
    let w_gpu = Var::from_tensor(&Tensor::from_slice(wv, (outf, inf), cuda).unwrap()).unwrap();
    let a_gpu = Var::from_tensor(&Tensor::from_slice(av, (r, inf), cuda).unwrap()).unwrap();
    let b_gpu = Var::from_tensor(&Tensor::from_slice(bv, (outf, r), cuda).unwrap()).unwrap();
    let ab_gpu = pack_ab(a_gpu.as_tensor(), b_gpu.as_tensor()).unwrap();
    let out_gpu = lora_linear(x_gpu.as_tensor(), w_gpu.as_tensor(), &ab_gpu, params).unwrap();
    let grads_gpu = out_gpu.sum_all().unwrap().backward().unwrap();

    let check = |name: &str, cpu_t: &Tensor, gpu_t: &Tensor| {
        let c: Vec<f32> = cpu_t.flatten_all().unwrap().to_vec1().unwrap();
        let g: Vec<f32> = gpu_t
            .flatten_all()
            .unwrap()
            .to_device(&cpu)
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(
            c, g,
            "{name}: exact-integer fixture must be bit-exact CPU vs CUDA"
        );
    };
    check(
        "fwd",
        &out_cpu.flatten_all().unwrap(),
        &out_gpu.flatten_all().unwrap(),
    );
    check(
        "dx",
        grads_cpu.get(&x_cpu).unwrap(),
        grads_gpu.get(&x_gpu).unwrap(),
    );
    check(
        "dw",
        grads_cpu.get(&w_cpu).unwrap(),
        grads_gpu.get(&w_gpu).unwrap(),
    );
    check(
        "da",
        grads_cpu.get(&a_cpu).unwrap(),
        grads_gpu.get(&a_gpu).unwrap(),
    );
    check(
        "db",
        grads_cpu.get(&b_cpu).unwrap(),
        grads_gpu.get(&b_gpu).unwrap(),
    );
}

/// The `bf16` counterpart to
/// `lora_linear_parity_f32_exact_integer_fixture_is_bit_exact`: small
/// integer `x`/`w` (exactly representable in `bf16`, since `bf16` exactly
/// represents every integer in `[-256, 256]`) at production `inf`, so the
/// true (infinite-precision) base sum is EXACTLY representable in `f32`
/// (`1024 * 4*4 = 16384 << 2^24`) and the CUDA base GEMM's tensor-core
/// `f32` accumulation is therefore EXACT up to the ONE unavoidable
/// rounding point this leg cannot eliminate: the final store of that exact
/// sum down to `bf16`. The expected reference is built with that SAME
/// single rounding (`bf16::from_f32(exact_sum)`), and the epilogue's own
/// two rounding points (`ScaledCastAdd`'s "round delta to `bf16` first,
/// then add-and-round-once" — see `ops::scaled_cast_add`'s module doc) are
/// reproduced by hand with the exact same formula
/// `scaled_cast_add_bf16_f32` uses. This is BIT-EXACT (not a tolerance
/// leg): a half-term drop, a wrong row, or a transposed operand would
/// almost certainly miss this exact reference, unlike the loose
/// term-magnitude bound `lora_linear_parity_bf16_base_production_width`
/// uses for its realistic (non-integer) fixture.
#[test]
fn lora_linear_parity_bf16_exact_integer_fixture_is_bit_exact() {
    let Some(cuda) = cuda_device() else {
        return;
    };
    let cpu = Device::Cpu;
    let (rows, inf, outf, r) = (8usize, 1024usize, 12usize, 4usize);
    let scale = 2.0f32; // exact in binary.

    let xv = exact_fixture(rows * inf, 1);
    let wv = exact_fixture(outf * inf, 2);
    let av = exact_fixture(r * inf, 3);
    let bv = exact_fixture(outf * r, 4);
    let x_bf16: Vec<bf16> = xv.iter().map(|&v| bf16::from_f32(v)).collect();
    let w_bf16: Vec<bf16> = wv.iter().map(|&v| bf16::from_f32(v)).collect();

    // The exact (f64, then losslessly narrowed to f32 — every partial sum
    // stays a small exact integer) base and delta pieces, computed
    // independently of candle's own GEMM so this reference shares no code
    // path with the op under test (family F).
    let mut base_bf16_expected = vec![bf16::from_f32(0.0); rows * outf];
    let mut delta_f32_exact = vec![0.0f32; rows * outf];
    for i in 0..rows {
        for o in 0..outf {
            let mut base_acc = 0i64;
            for k in 0..inf {
                base_acc += (xv[i * inf + k] as i64) * (wv[o * inf + k] as i64);
            }
            base_bf16_expected[i * outf + o] = bf16::from_f32(base_acc as f32);

            let mut delta_acc = 0i64;
            for j in 0..r {
                let mut h_acc = 0i64;
                for k in 0..inf {
                    h_acc += (xv[i * inf + k] as i64) * (av[j * inf + k] as i64);
                }
                delta_acc += h_acc * (bv[o * r + j] as i64);
            }
            delta_f32_exact[i * outf + o] = delta_acc as f32;
        }
    }
    let expected: Vec<bf16> = base_bf16_expected
        .iter()
        .zip(delta_f32_exact.iter())
        .map(|(&base, &delta)| {
            // Mirrors `scaled_cast_add_bf16_f32` exactly (round delta to
            // bf16 first, then add-and-round-once).
            let delta_bf16 = bf16::from_f32(delta * scale);
            bf16::from_f32(base.to_f32() + delta_bf16.to_f32())
        })
        .collect();

    let x_gpu = Tensor::from_slice(&x_bf16, (rows, inf), &cuda).unwrap();
    let w_gpu = Tensor::from_slice(&w_bf16, (outf, inf), &cuda).unwrap();
    let ab_gpu = pack_ab(
        &Tensor::from_slice(&av, (r, inf), &cuda).unwrap(),
        &Tensor::from_slice(&bv, (outf, r), &cuda).unwrap(),
    )
    .unwrap();
    let params = LoraLinearParams {
        scale,
        inf,
        outf,
        r,
        dropout: None,
        dweight_needed: false,
    };
    let out_gpu = lora_linear(&x_gpu, &w_gpu, &ab_gpu, params).unwrap();
    let got: Vec<bf16> = out_gpu
        .flatten_all()
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_vec1()
        .unwrap();

    assert_eq!(
        got, expected,
        "exact-integer bf16 fixture must be bit-exact against a reference built from the \
         SAME single rounding point the op's own epilogue documents"
    );
}
