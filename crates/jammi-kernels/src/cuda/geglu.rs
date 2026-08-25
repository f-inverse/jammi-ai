use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::{CudaStorage, DType, Error, Layout, Result, Shape};
use half::bf16;

use super::PTX_GEGLU;
use crate::ops::geglu::{check_variant, geglu_dims, output_shape, GeluVariant};

/// See `../axpy.rs`'s identical constant for the module-name rationale —
/// arbitrary but stable and unique to this op's PTX module.
const MODULE_NAME: &str = "jammi_kernels_geglu";

/// Grid-stride block size (see `geglu.cu`'s module doc for why this op is
/// purely elementwise, with no per-row block reduction).
const GEGLU_BLOCK: u32 = 256;

/// A conservative 1-D grid cap; the kernel's own grid-stride loop covers
/// any `n_out` beyond `GEGLU_BLOCK * GEGLU_MAX_GRID` correctly (unlike
/// `Axpy`'s single-pass `if (i < n)` kernel, this one does not need the
/// grid to cover `n_out` in a single pass).
const GEGLU_MAX_GRID: u32 = 65_535;

fn launch_config(n: u32) -> LaunchConfig {
    let blocks = n.div_ceil(GEGLU_BLOCK).clamp(1, GEGLU_MAX_GRID);
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (GEGLU_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub(crate) fn cuda_fwd(
    variant: GeluVariant,
    s1: &CudaStorage,
    l1: &Layout,
) -> Result<(CudaStorage, Shape)> {
    const OP: &str = "geglu_fused";
    check_variant(variant, OP)?;
    let (rows, intermediate) = geglu_dims(l1, OP)?;
    let out_shape = output_shape(l1, intermediate);
    let device = s1.device().clone();

    if intermediate == 0 {
        return Ok((super::alloc_empty(&device, s1.dtype(), OP)?, out_shape));
    }
    let n_out = rows * intermediate;
    super::check_elem_count_fits_u32(OP, n_out)?;
    let (o1, o2) = l1
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;
    let cfg = launch_config(n_out as u32);
    let intermediate_u32 = intermediate as u32;
    let n_out_u32 = n_out as u32;

    match s1.dtype() {
        DType::F32 => {
            let x = s1.as_cuda_slice::<f32>()?.slice(o1..o2);
            let func = device.get_or_load_custom_func("geglu_fwd_f32", MODULE_NAME, PTX_GEGLU)?;
            let out = unsafe { device.alloc::<f32>(n_out) }?;
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&out);
            builder.arg(&intermediate_u32);
            builder.arg(&n_out_u32);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), out_shape))
        }
        DType::BF16 => {
            let x = s1.as_cuda_slice::<bf16>()?.slice(o1..o2);
            let func = device.get_or_load_custom_func("geglu_fwd_bf16", MODULE_NAME, PTX_GEGLU)?;
            let out = unsafe { device.alloc::<bf16>(n_out) }?;
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&out);
            builder.arg(&intermediate_u32);
            builder.arg(&n_out_u32);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), out_shape))
        }
        dtype => Err(Error::UnsupportedDTypeForOp(dtype, OP)),
    }
}

pub(crate) fn cuda_bwd_dwi_out(
    variant: GeluVariant,
    s1: &CudaStorage,
    l1: &Layout,
    s2: &CudaStorage,
    l2: &Layout,
) -> Result<(CudaStorage, Shape)> {
    const OP: &str = "geglu_fused_bwd_dwi_out";
    check_variant(variant, OP)?;
    let (rows, intermediate) = geglu_dims(l1, OP)?;
    let expected_dy = output_shape(l1, intermediate);
    if l2.dims() != expected_dy.dims() {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l1.shape().clone(),
            rhs: l2.shape().clone(),
            op: OP,
        });
    }
    if s1.dtype() != s2.dtype() {
        return Err(Error::DTypeMismatchBinaryOp {
            lhs: s1.dtype(),
            rhs: s2.dtype(),
            op: OP,
        });
    }
    let device = s1.device().clone();
    let wi_shape = l1.shape().clone();

    if intermediate == 0 {
        return Ok((super::alloc_empty(&device, s1.dtype(), OP)?, wi_shape));
    }
    let n_out = rows * intermediate;
    super::check_elem_count_fits_u32(OP, n_out)?;
    let n_full = rows * 2 * intermediate;
    let (o1, o2) = l1
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;
    let (d1, d2) = l2
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;
    let cfg = launch_config(n_out as u32);
    let intermediate_u32 = intermediate as u32;
    let n_out_u32 = n_out as u32;

    match s1.dtype() {
        DType::F32 => {
            let x = s1.as_cuda_slice::<f32>()?.slice(o1..o2);
            let dy = s2.as_cuda_slice::<f32>()?.slice(d1..d2);
            let func =
                device.get_or_load_custom_func("geglu_bwd_dwi_out_f32", MODULE_NAME, PTX_GEGLU)?;
            let out = unsafe { device.alloc::<f32>(n_full) }?;
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&dy);
            builder.arg(&out);
            builder.arg(&intermediate_u32);
            builder.arg(&n_out_u32);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), wi_shape))
        }
        DType::BF16 => {
            let x = s1.as_cuda_slice::<bf16>()?.slice(o1..o2);
            let dy = s2.as_cuda_slice::<bf16>()?.slice(d1..d2);
            let func =
                device.get_or_load_custom_func("geglu_bwd_dwi_out_bf16", MODULE_NAME, PTX_GEGLU)?;
            let out = unsafe { device.alloc::<bf16>(n_full) }?;
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&dy);
            builder.arg(&out);
            builder.arg(&intermediate_u32);
            builder.arg(&n_out_u32);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), wi_shape))
        }
        dtype => Err(Error::UnsupportedDTypeForOp(dtype, OP)),
    }
}
