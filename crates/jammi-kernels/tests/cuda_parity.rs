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
    apply2, apply3, Axpy, FullyMaskedPolicy, LayerNormFused, RopeFused, SoftmaxLastDimFused,
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
    let grads_cpu = out_cpu.backward().unwrap();

    let s_gpu = Var::from_tensor(&Tensor::from_slice(sv, (rows, last), cuda).unwrap()).unwrap();
    let m_gpu = Tensor::from_slice(&mv, (1, last), cuda).unwrap();
    let out_gpu = softmax(&s_gpu, &m_gpu).unwrap();
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
    assert_eq!(out_gpu_v.len(), n, "softmax GPU fwd length mismatch");
    for (i, (c, g)) in out_cpu_v.iter().zip(out_gpu_v.iter()).enumerate() {
        assert!(
            ((*c - *g).abs() as f64) <= F32_TOL,
            "softmax fwd[{i}]: cpu {c} vs cuda {g} (rows={rows}, last={last})"
        );
    }

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
    let grads_cpu = out_cpu.backward().unwrap();

    let s_gpu = Var::from_tensor(&Tensor::from_slice(&sb, (rows, last), cuda).unwrap()).unwrap();
    let m_gpu = Tensor::from_slice(&mb, (1, last), cuda).unwrap();
    let out_gpu = softmax(&s_gpu, &m_gpu).unwrap();
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
    assert_eq!(out_gpu_v.len(), n, "softmax bf16 GPU fwd length mismatch");
    for (i, (c, g)) in out_cpu_v.iter().zip(out_gpu_v.iter()).enumerate() {
        assert!(
            (c - g).abs() <= bf16_bound(*c, *g),
            "softmax bf16 fwd[{i}]: cpu {c} vs cuda {g}"
        );
    }

    let dx_cpu_v = to_f32(&grads_cpu.get(&s_cpu).unwrap().clone());
    let dx_gpu_v = to_f32(&grads_gpu.get(&s_gpu).unwrap().to_device(&cpu).unwrap());
    assert_eq!(dx_cpu_v.len(), n);
    assert_eq!(
        dx_gpu_v.len(),
        n,
        "softmax bf16 GPU dscores length mismatch"
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
