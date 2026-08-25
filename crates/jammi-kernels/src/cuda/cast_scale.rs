//! CUDA glue for [`crate::ops::CastScaleBf16F32`] /
//! [`crate::ops::CastAddBf16`] — see `../ops/cast_scale.rs`'s module doc for
//! the traffic model and rounding-order derivations both kernels below
//! implement.

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::PushKernelArg;
use candle_core::{CudaStorage, DType, Error, Layout, Result, Shape};
use half::bf16;

use super::PTX_CAST_SCALE;

/// See `../axpy.rs`'s identical constant for the module-name rationale.
const MODULE_NAME: &str = "jammi_kernels_cast_scale";

pub(crate) fn cuda_fwd_cast_scale_bf16_f32(
    scale: f64,
    s1: &CudaStorage,
    l1: &Layout,
) -> Result<(CudaStorage, Shape)> {
    const OP: &str = "cast_scale_bf16_f32";
    if s1.dtype() != DType::BF16 {
        return Err(Error::UnsupportedDTypeForOp(s1.dtype(), OP));
    }

    let device = s1.device().clone();
    let shape = l1.shape().clone();
    let n = shape.elem_count();
    let scale_f32 = scale as f32;

    // n == 0: match every other op's CUDA glue in this crate — an explicit
    // empty allocation, never a `LaunchConfig::for_num_elems(0)` launch
    // (grid_dim (0, 1, 1) is illegal).
    if n == 0 {
        return Ok((super::alloc_empty(&device, DType::F32, OP)?, shape));
    }

    super::check_elem_count_fits_u32(OP, n)?;

    let (o1, o2) = l1
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;
    let cfg = super::elemwise_launch_config(n as u32);

    let x = s1.as_cuda_slice::<bf16>()?.slice(o1..o2);
    let func =
        device.get_or_load_custom_func("cast_scale_bf16_f32", MODULE_NAME, PTX_CAST_SCALE)?;
    let out = unsafe { device.alloc::<f32>(n) }?;
    let mut builder = func.builder();
    builder.arg(&scale_f32);
    builder.arg(&x);
    builder.arg(&out);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
    Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
}

pub(crate) fn cuda_fwd_cast_add_bf16(
    s1: &CudaStorage,
    l1: &Layout,
    s2: &CudaStorage,
    l2: &Layout,
) -> Result<(CudaStorage, Shape)> {
    const OP: &str = "cast_add_bf16";
    if l1.dims() != l2.dims() {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l1.shape().clone(),
            rhs: l2.shape().clone(),
            op: OP,
        });
    }
    if s1.dtype() != DType::BF16 {
        return Err(Error::UnsupportedDTypeForOp(s1.dtype(), OP));
    }
    if s2.dtype() != DType::F32 {
        return Err(Error::UnsupportedDTypeForOp(s2.dtype(), OP));
    }

    let device = s1.device().clone();
    let shape = l1.shape().clone();
    let n = shape.elem_count();

    if n == 0 {
        return Ok((super::alloc_empty(&device, DType::BF16, OP)?, shape));
    }

    super::check_elem_count_fits_u32(OP, n)?;

    let (o1_base, o2_base) = l1
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;
    let (o1_f32, o2_f32) = l2
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;
    let cfg = super::elemwise_launch_config(n as u32);

    let base = s1.as_cuda_slice::<bf16>()?.slice(o1_base..o2_base);
    let f32val = s2.as_cuda_slice::<f32>()?.slice(o1_f32..o2_f32);
    let func = device.get_or_load_custom_func("cast_add_bf16", MODULE_NAME, PTX_CAST_SCALE)?;
    let out = unsafe { device.alloc::<bf16>(n) }?;
    let mut builder = func.builder();
    builder.arg(&base);
    builder.arg(&f32val);
    builder.arg(&out);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
    Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
}
