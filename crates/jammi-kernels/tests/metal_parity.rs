//! CPU↔Metal parity oracles for the two `CustomOp`s a QLoRA training
//! forward on Apple Silicon actually reaches — the landing proof for issue
//! #433 (LoRA/QLoRA training on Metal dying at the shipped default
//! `lora_dropout = 0.05`, because `ops::DropoutFused` had no `metal_fwd`
//! and candle's default `metal_fwd` is a typed `Err`, not a fallback).
//!
//! Compiles and links ONLY with the `metal` feature
//! (`required-features = ["metal"]` in `Cargo.toml` — a plain `cargo test
//! -p jammi-kernels` never even builds this file), mirroring
//! `cuda_parity.rs`'s own gating convention. At runtime, a machine that
//! compiled with the feature but has no physical Metal device (or is not
//! on macOS) is treated as "skip", not "fail" — `Device::new_metal(0)`
//! erroring, OR PANICKING, is the signal — UNLESS `JAMMI_REQUIRE_METAL` is
//! set, in which case a device-acquisition failure PANICS instead of
//! returning, for the exact reason `cuda_parity.rs`'s own doc gives:
//! without that distinction a broken device acquisition on a machine that
//! is SUPPOSED to have a GPU would silently read as skipped tests rather
//! than failed ones.
//!
//! `metal_device_or_skip`'s acquisition is wrapped in
//! `std::panic::catch_unwind`: on at least one real GH `macos-14` runner,
//! `Device::new_metal(0)` does not merely return `Err` on a missing/broken
//! device — an `objc2` class lookup inside candle-metal-kernels'
//! `residency_set.rs:18` (`MTLResidencySetDescriptor`) can PANIC instead,
//! a probe-time failure mode a bare `Result` cannot model. Catching that
//! panic here is sound: the probe owns no lock and mutates no shared state
//! before failing, so unwinding out of it leaves nothing poisoned to clean
//! up — unlike catching a panic across a held mutex guard or a
//! half-mutated `static`. Both failure shapes (a returned `Err`, or a
//! caught panic) fold into the SAME skip/require message and the SAME
//! `Option<Device>` lattice below, so every call site downstream of this
//! fn sees one uniform "skip or proceed" decision regardless of which way
//! acquisition actually failed.
//!
//! ## `ops::DropoutFused` (this file's primary subject)
//!
//! `DropoutFused::metal_fwd` is a device-scoped deterministic HOST
//! fallback (no Metal Philox compute kernel exists in this crate — see
//! `ops::dropout`'s module doc for why that shape was judged
//! disproportionate to this defect): it downloads the Metal input to the
//! host and runs the SAME `dropout_f32`/`dropout_bf16` functions
//! `cpu_fwd` does. That makes this suite's job narrower than
//! `cuda_parity.rs`'s (which proves independent CPU and CUDA
//! implementations agree): here the two code paths ARE the same code, so
//! this suite instead proves the DISPATCH is wired correctly end to end
//! (the right storage got downloaded, sliced at the right offsets,
//! computed, and re-uploaded) — i.e. it is a wiring/plumbing oracle, not
//! an independent-implementation cross-check.
//!
//! ## `ops::QuantMatMulGrad` (the OTHER op a QLoRA forward reaches on
//! Metal, already `metal_fwd`-complete before this file existed)
//!
//! `FrozenBase::Quantized` (`jammi-lora`) ALWAYS routes its forward
//! through `quant_matmul_grad`, on EVERY device — unlike the fused LoRA
//! epilogue site, a Quantized base never falls back to a Dense-only fused
//! kernel, so this op's own Metal correctness is EQUALLY load-bearing for
//! issue #433's QLoRA-on-Metal repro as `DropoutFused`'s is. Its
//! `metal_fwd` already delegated to `QTensor::metal_fwd` (candle-core's
//! own quantized Metal kernel) before this fix; its `bwd` dequantizes `W`
//! and runs an ordinary `Tensor::matmul` — device-generic candle ops, no
//! per-device branch of its own. Neither half had a Metal-specific test
//! anywhere in this crate before this file — added here alongside the
//! dropout oracles because both were exercised (and needed) by the SAME
//! real-hardware landing run (`cargo test -p jammi-ai --test
//! metal_quantized_gpu --features metal,local`) this fix's investigation
//! used to confirm QLoRA training data flow end to end on Metal.

#![cfg(feature = "metal")]

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor, Var};
use half::bf16;
use jammi_kernels::ops::{apply1, quant_matmul_grad, DropoutFused};
use std::sync::Arc;

/// A panic payload folded to a human-readable string — `std::panic::
/// catch_unwind`'s error type is `Box<dyn std::any::Any + Send>`, which
/// carries no `Display`/`Debug` of its own; `panic!("{msg}")`/`panic!(msg)`
/// (the two shapes `std`'s own panic machinery ever constructs) box either
/// a `&'static str` or an owned `String`, so those two downcasts cover
/// every REAL panic payload this process can produce.
fn panic_payload_to_string(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

/// Acquire a Metal device, or `None` to skip — unless `JAMMI_REQUIRE_METAL`
/// is set, in which case a failure PANICS. Mirrors `cuda_parity.rs`'s
/// `cuda_device_or_skip`, widened (module doc above) to fold BOTH a
/// returned `Err` AND a caught panic from `Device::new_metal(0)` into the
/// same `Result<Device, String>` before deciding skip-vs-require, so the
/// decision itself (the `if`/`panic!` below) is written exactly once.
fn metal_device_or_skip() -> Option<Device> {
    let outcome: Result<Device, String> = match std::panic::catch_unwind(|| Device::new_metal(0)) {
        Ok(Ok(d)) => Ok(d),
        Ok(Err(e)) => Err(e.to_string()),
        Err(payload) => Err(format!(
            "Device::new_metal(0) panicked: {}",
            panic_payload_to_string(payload.as_ref())
        )),
    };
    match outcome {
        Ok(d) => Some(d),
        Err(msg) => {
            if std::env::var_os("JAMMI_REQUIRE_METAL").is_some() {
                panic!("JAMMI_REQUIRE_METAL is set but no Metal device is available: {msg}");
            }
            eprintln!("metal_parity: skipping — no Metal device available: {msg}");
            None
        }
    }
}

fn dropout(seed: u64, layer_id: u32, forward_idx: u32, p: f32, x: &Tensor) -> Tensor {
    let op = DropoutFused::new(seed, layer_id, forward_idx, p).unwrap();
    apply1(x, op).unwrap()
}

/// F32: several `(seed, layer_id, forward_idx, p)` combinations, each
/// checked for byte-exact CPU/Metal agreement — not just one fixture,
/// since a wrong offset or a wrong dtype branch could plausibly cancel out
/// for one particular size/seed.
#[test]
fn dropout_f32_mask_matches_cpu_across_several_configs() {
    let Some(metal) = metal_device_or_skip() else {
        return;
    };
    let cpu = Device::Cpu;
    let configs: [(u64, u32, u32, f32, usize); 4] = [
        (1, 0, 0, 0.05, 1), // the shipped LoRA default, at the smallest possible size
        (11, 3, 5, 0.3, 1000),
        (4242, 7, 1, 0.5, 100_000),
        (u64::MAX, u32::MAX, u32::MAX, 0.99, 777),
    ];
    for (seed, layer_id, forward_idx, p, n) in configs {
        let v: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.001).collect();
        let x_cpu = Tensor::from_slice(&v, (n,), &cpu).unwrap();
        let x_metal = Tensor::from_slice(&v, (n,), &metal).unwrap();
        let out_cpu: Vec<f32> = dropout(seed, layer_id, forward_idx, p, &x_cpu)
            .to_vec1()
            .unwrap();
        let out_metal: Vec<f32> = dropout(seed, layer_id, forward_idx, p, &x_metal)
            .to_vec1()
            .unwrap();
        assert_eq!(
            out_cpu, out_metal,
            "seed={seed} layer_id={layer_id} forward_idx={forward_idx} p={p} n={n}: \
             Metal mask must be byte-identical to CPU's"
        );
    }
}

/// BF16: the other production activation dtype — same byte-exactness
/// requirement, through the `bf16::from_f32` single-rounding path.
#[test]
fn dropout_bf16_mask_matches_cpu() {
    let Some(metal) = metal_device_or_skip() else {
        return;
    };
    let cpu = Device::Cpu;
    let n = 5000usize;
    let v: Vec<bf16> = (0..n)
        .map(|i| bf16::from_f32(1.0 + i as f32 * 0.01))
        .collect();
    let x_cpu = Tensor::from_slice(&v, (n,), &cpu).unwrap();
    let x_metal = Tensor::from_slice(&v, (n,), &metal).unwrap();
    let out_cpu: Vec<f32> = dropout(99, 2, 4, 0.4, &x_cpu)
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap();
    let out_metal: Vec<f32> = dropout(99, 2, 4, 0.4, &x_metal)
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(
        out_cpu, out_metal,
        "BF16 Metal mask must be byte-identical to CPU's"
    );
}

/// Backward: `dx` computed through a Metal `Var` must match `dx` computed
/// through the identical CPU `Var` — proving `bwd`'s regenerated decision
/// (which re-dispatches through the SAME `apply1`, hence the SAME
/// `metal_fwd`) round-trips correctly, not just forward.
#[test]
fn dropout_backward_dx_matches_cpu() {
    let Some(metal) = metal_device_or_skip() else {
        return;
    };
    let cpu = Device::Cpu;
    let n = 5000usize;
    let xv: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.001).collect();
    let dy_v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();

    let dx_on = |device: &Device| -> Vec<f32> {
        let x = Var::from_tensor(&Tensor::from_slice(&xv, (n,), device).unwrap()).unwrap();
        let op = DropoutFused::new(99, 2, 4, 0.4).unwrap();
        let y = apply1(x.as_tensor(), op).unwrap();
        let dy = Tensor::from_slice(&dy_v, (n,), device).unwrap();
        let loss = (&y * &dy).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        grads.get(&x).unwrap().to_vec1().unwrap()
    };

    let dx_cpu = dx_on(&cpu);
    let dx_metal = dx_on(&metal);
    assert_eq!(
        dx_cpu, dx_metal,
        "backward dx through a Metal Var must match the identical CPU Var's dx"
    );
}

/// Domain oracle (family D), replayed on Metal: an empty tensor is a
/// no-op, not an error, on this device too.
///
/// Built via `Tensor::zeros` rather than `Tensor::from_slice(&[], ..)`:
/// candle-metal-kernels' `newBufferWithBytes:length:0` (the path
/// `from_slice`/`to_device` take, `storage_from_slice`/
/// `storage_from_cpu_storage`) fails to allocate a zero-BYTE Metal buffer
/// on this host (`Metal error Failed to create metal resource: Buffer`) —
/// a pre-existing candle-core Metal-backend quirk unrelated to this op
/// (`Tensor::zeros`'s `with_size_for` path allocates the SAME zero-element
/// buffer successfully), reproducible with a bare
/// `Tensor::from_slice(&[] as &[f32], (0,), &metal_device)` outside this
/// op entirely. Using `zeros` here keeps this test proving THIS op's
/// domain behavior, not re-litigating an unrelated candle-core allocator
/// gap.
#[test]
fn dropout_empty_tensor_is_a_no_op_on_metal() {
    let Some(metal) = metal_device_or_skip() else {
        return;
    };
    let x = Tensor::zeros((0,), DType::F32, &metal).unwrap();
    let out: Vec<f32> = dropout(1, 0, 0, 0.3, &x).to_vec1().unwrap();
    assert!(out.is_empty());
}

/// Domain oracle (family D), replayed on Metal: a non-contiguous view is
/// refused with the SAME typed error `cpu_fwd` returns, not silently
/// misread through the raw-buffer download.
#[test]
fn dropout_non_contiguous_view_is_refused_on_metal() {
    let Some(metal) = metal_device_or_skip() else {
        return;
    };
    let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &metal)
        .unwrap()
        .t()
        .unwrap();
    assert!(!x.is_contiguous());
    let op = DropoutFused::new(1, 0, 0, 0.3).unwrap();
    let err = apply1(&x, op).expect_err("non-contiguous Metal input must be refused");
    assert!(matches!(err, candle_core::Error::RequiresContiguous { .. }));
}

/// Domain oracle (family D), replayed on Metal, pinning the boundary WHERE
/// the empty-fast-path and the non-contiguous refusal above could
/// disagree: a ZERO-ELEMENT, NON-CONTIGUOUS view. `ops::dropout`'s
/// `metal_fwd` used to check `elem_count() == 0` BEFORE calling
/// `contiguous_offsets()`, so an empty-but-non-contiguous layout took the
/// empty fast path silently instead of hitting the SAME
/// `RequiresContiguous` refusal `cpu_fwd` gives it — the two domains
/// disagreed exactly at this one corner. `Tensor::zeros((0, 3), ..)`
/// transposed to `(3, 0)`: `candle_core::Shape::is_contiguous` resets its
/// running stride accumulator to `0` at the zero-sized dim, so the
/// FOLLOWING dim (`3`, stride `1` post-transpose) fails the `stride ==
/// acc` check — genuinely non-contiguous by candle's own definition, even
/// though it holds no elements. Built via `zeros` (not `from_slice`), same
/// reason as `dropout_empty_tensor_is_a_no_op_on_metal` above.
#[test]
fn dropout_empty_non_contiguous_view_is_refused_on_metal() {
    let Some(metal) = metal_device_or_skip() else {
        return;
    };
    let x = Tensor::zeros((0, 3), DType::F32, &metal)
        .unwrap()
        .t()
        .unwrap();
    assert_eq!(x.elem_count(), 0);
    assert!(
        !x.is_contiguous(),
        "a (0, 3) tensor transposed to (3, 0) must read as non-contiguous"
    );
    let op = DropoutFused::new(1, 0, 0, 0.3).unwrap();
    let err = apply1(&x, op).expect_err("an empty, non-contiguous Metal input must be refused");
    assert!(matches!(err, candle_core::Error::RequiresContiguous { .. }));
}

/// Domain oracle (family D) pinning byte-parity on the ONE input where
/// this file's own module doc's "downloads the input's raw backing buffer
/// ... slices `[o1..o2]` itself" design could plausibly diverge: an
/// `o1 > 0` CONTIGUOUS view, built via `Tensor::narrow` on a larger
/// backing buffer. `dropout_f32`/`dropout_bf16`'s KEEP/DROP decision is a
/// function of the LOCAL element index within the sliced window (`0..len`,
/// `dropout_f32`'s own `x.iter().enumerate()`), so a wrong slice window —
/// e.g. reading from the downloaded buffer's absolute position `0` instead
/// of `o1` — would apply the SAME mask decisions to the WRONG underlying
/// VALUES rather than erroring, invisibly to every other fixture in this
/// file (every one of which narrows nothing, so `o1 == 0` there and the
/// "slice from `o1`" and "slice from `0`" bugs would coincide). This
/// fixture's values are strictly increasing, so a wrong window changes the
/// output vector, not just its mask.
#[test]
fn dropout_offset_narrowed_view_matches_cpu() {
    let Some(metal) = metal_device_or_skip() else {
        return;
    };
    let cpu = Device::Cpu;
    let n = 20usize;
    let v: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.001).collect();
    let x_cpu = Tensor::from_slice(&v, (n,), &cpu)
        .unwrap()
        .narrow(0, 7, 10)
        .unwrap();
    let x_metal = Tensor::from_slice(&v, (n,), &metal)
        .unwrap()
        .narrow(0, 7, 10)
        .unwrap();
    assert!(x_cpu.is_contiguous());
    assert!(x_metal.is_contiguous());
    let out_cpu: Vec<f32> = dropout(3, 1, 2, 0.4, &x_cpu).to_vec1().unwrap();
    let out_metal: Vec<f32> = dropout(3, 1, 2, 0.4, &x_metal).to_vec1().unwrap();
    assert_eq!(
        out_cpu, out_metal,
        "an offset-narrowed (o1 > 0) contiguous Metal view must byte-match the identical CPU view"
    );
}

/// `ops::QuantMatMulGrad`'s backward (`dx = dy @ dequantize(W)`) on Metal
/// must match CPU's, for the same quantized weight and the same input —
/// the OTHER op a QLoRA training forward on Metal depends on end to end
/// (see this file's module doc). `QTensor::quantize_onto` builds the
/// SAME quantized bytes directly on each target device (no cross-device
/// copy of the quantized weight itself involved), isolating this
/// assertion to `bwd`'s own dequantize-then-matmul math.
#[test]
fn quant_matmul_grad_backward_dx_matches_cpu() {
    let Some(metal) = metal_device_or_skip() else {
        return;
    };
    let cpu = Device::Cpu;
    let (out_f, in_f, rows) = (5usize, 64usize, 3usize);
    let w_f32: Vec<f32> = (0..out_f * in_f)
        .map(|i| ((i as f64) * 0.037 + 0.25).sin() as f32)
        .collect();
    let w_cpu = Tensor::from_vec(w_f32, (out_f, in_f), &cpu).unwrap();
    let x_v: Vec<f32> = (0..rows * in_f)
        .map(|i| ((i as f64) * 0.091 + 1.0).cos() as f32)
        .collect();

    let dx_on = |device: &Device| -> Vec<f32> {
        let wq = Arc::new(QTensor::quantize_onto(&w_cpu, GgmlDType::Q8_0, device).unwrap());
        let x = Var::from_tensor(&Tensor::from_vec(x_v.clone(), (rows, in_f), device).unwrap())
            .unwrap();
        let y = quant_matmul_grad(x.as_tensor(), wq).unwrap();
        let loss = y.sum_all().unwrap();
        let grads = loss.backward().unwrap();
        grads
            .get(&x)
            .expect("quant_matmul_grad's bwd must never return None (module doc)")
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };

    let dx_cpu = dx_on(&cpu);
    let dx_metal = dx_on(&metal);
    assert_eq!(
        dx_cpu, dx_metal,
        "quant_matmul_grad backward dx on Metal must match CPU's for the same quantized weight"
    );
}
