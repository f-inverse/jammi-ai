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

use candle_core::{DType, Device, Tensor, Var};
use half::bf16;
use jammi_kernels::ops::{apply2, Axpy};

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
