use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::{CudaStorage, DType, Error, Layout, Result, Shape};
use half::bf16;

use super::PTX_AXPY;

/// The kernel module name PTX functions are loaded under
/// (`CudaDevice::get_or_load_custom_func`'s module cache key) — arbitrary,
/// but stable and unique to this op so a second op's module never collides.
const MODULE_NAME: &str = "jammi_kernels_axpy";

pub(crate) fn cuda_fwd(
    alpha: f64,
    s1: &CudaStorage,
    l1: &Layout,
    s2: &CudaStorage,
    l2: &Layout,
) -> Result<(CudaStorage, Shape)> {
    if l1.dims() != l2.dims() {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l1.shape().clone(),
            rhs: l2.shape().clone(),
            op: "axpy",
        });
    }

    let device = s1.device().clone();
    let n = l1.shape().elem_count();
    let shape = l1.shape().clone();

    // n == 0: `LaunchConfig::for_num_elems(0)` yields grid_dim (0, 1, 1) —
    // an illegal launch. Match the CPU arm's documented no-op contract
    // (`ops::axpy`'s `empty_tensor_is_a_no_op_not_an_error`) instead of
    // ever reaching the launch below.
    if n == 0 {
        return match (s1.dtype(), s2.dtype()) {
            (DType::F32, DType::F32) => {
                let out = unsafe { device.alloc::<f32>(0) }?;
                Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
            }
            (DType::BF16, DType::BF16) => {
                let out = unsafe { device.alloc::<bf16>(0) }?;
                Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
            }
            (lhs, rhs) if lhs != rhs => Err(Error::DTypeMismatchBinaryOp {
                lhs,
                rhs,
                op: "axpy",
            }),
            (dtype, _) => Err(Error::UnsupportedDTypeForOp(dtype, "axpy")),
        };
    }

    // K2 (refuse, don't compute past the domain): the launch grid below is
    // built from `n as u32`, and the kernel's own bounds check
    // (`if (i < n)`) is over that same (would-be-truncated) `n`. Above
    // `u32::MAX` the cast silently wraps, under-launching relative to the
    // real element count and leaving the allocation's tail uninitialized —
    // a confident wrong answer, not a crash. Refuse explicitly instead.
    if n > u32::MAX as usize {
        return Err(Error::Msg(format!(
            "axpy: {n} elements exceeds u32::MAX; the CUDA launch grid and \
             the kernel's bounds check are both 32-bit"
        )));
    }

    // `is_contiguous()` alone is NOT sufficient: it does not imply
    // `start_offset == 0` (candle's own doc on `Layout::is_contiguous`:
    // "does not imply that the start offset is 0 or that there are no
    // extra elements at the end of the storage") — a `narrow`'d-but-still-
    // contiguous view can have a nonzero offset into a LARGER base buffer.
    // `as_cuda_slice::<T>()` returns the WHOLE base `CudaSlice`, so reading
    // it from element 0 (this function's previous behavior) reads the
    // base buffer's first `n` elements instead of this tensor's actual
    // `[start_offset, start_offset + n)` range — reproduced in
    // `tests/cuda_parity.rs::parity_narrowed_with_nonzero_offset`.
    // `contiguous_offsets()` gives the real `[o1, o2)` range; slicing to it
    // is candle's own idiom for this exact situation
    // (`cuda_backend/mod.rs`'s `IndexAdd` CUDA impl does the same
    // `contiguous_offsets()` -> `slice(o1..o2)` on both of its tensor
    // args).
    let (o1_x, o2_x) = l1
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: "axpy" })?;
    let (o1_y, o2_y) = l2
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: "axpy" })?;

    let cfg = LaunchConfig::for_num_elems(n as u32);

    match (s1.dtype(), s2.dtype()) {
        (DType::F32, DType::F32) => {
            let x = s1.as_cuda_slice::<f32>()?.slice(o1_x..o2_x);
            let y = s2.as_cuda_slice::<f32>()?.slice(o1_y..o2_y);
            let func = device.get_or_load_custom_func("axpy_f32", MODULE_NAME, PTX_AXPY)?;
            let out = unsafe { device.alloc::<f32>(n) }?;
            let alpha_f32 = alpha as f32;
            let mut builder = func.builder();
            builder.arg(&alpha_f32);
            builder.arg(&x);
            builder.arg(&y);
            builder.arg(&out);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        (DType::BF16, DType::BF16) => {
            let x = s1.as_cuda_slice::<bf16>()?.slice(o1_x..o2_x);
            let y = s2.as_cuda_slice::<bf16>()?.slice(o1_y..o2_y);
            let func = device.get_or_load_custom_func("axpy_bf16", MODULE_NAME, PTX_AXPY)?;
            let out = unsafe { device.alloc::<bf16>(n) }?;
            let alpha_f32 = alpha as f32;
            let mut builder = func.builder();
            builder.arg(&alpha_f32);
            builder.arg(&x);
            builder.arg(&y);
            builder.arg(&out);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        (lhs, rhs) if lhs != rhs => Err(Error::DTypeMismatchBinaryOp {
            lhs,
            rhs,
            op: "axpy",
        }),
        (dtype, _) => Err(Error::UnsupportedDTypeForOp(dtype, "axpy")),
    }
}
