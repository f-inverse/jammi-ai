use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::PushKernelArg;
use candle_core::{CudaStorage, DType, Error, Layout, Result, Shape};
use half::{bf16, f16};

use super::{PTX_GELU_ERF, PTX_GELU_ERF_F16};
use crate::ops::gelu_erf::check_domain;

/// See `crate::cuda`'s module doc for the module-name rationale — arbitrary
/// but stable and unique to this op's PTX module.
const MODULE_NAME: &str = "jammi_kernels_gelu_erf";

/// The F16 arm's OWN PTX module name — `gelu_erf_f16.cu` is a SEPARATE
/// translation unit (see that file's module doc), so it needs a distinct
/// module name from [`MODULE_NAME`].
const MODULE_NAME_F16: &str = "jammi_kernels_gelu_erf_f16";

pub(crate) fn cuda_fwd(s1: &CudaStorage, l1: &Layout) -> Result<(CudaStorage, Shape)> {
    const OP: &str = "gelu_erf_fused";
    let (o1, o2) = check_domain(l1, OP)?;
    let n = l1.shape().elem_count();
    super::check_elem_count_fits_u32(OP, n)?;
    let device = s1.device().clone();
    let shape = l1.shape().clone();
    let cfg = super::elemwise_launch_config(n as u32);

    match s1.dtype() {
        DType::F32 => {
            let x = s1.as_cuda_slice::<f32>()?.slice(o1..o2);
            let func =
                device.get_or_load_custom_func("gelu_erf_fwd_f32", MODULE_NAME, PTX_GELU_ERF)?;
            let out = unsafe { device.alloc::<f32>(n) }?;
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&out);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        DType::BF16 => {
            let x = s1.as_cuda_slice::<bf16>()?.slice(o1..o2);
            let func =
                device.get_or_load_custom_func("gelu_erf_fwd_bf16", MODULE_NAME, PTX_GELU_ERF)?;
            let out = unsafe { device.alloc::<bf16>(n) }?;
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&out);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        DType::F16 => {
            let x = s1.as_cuda_slice::<f16>()?.slice(o1..o2);
            let func = device.get_or_load_custom_func(
                "gelu_erf_fwd_f16",
                MODULE_NAME_F16,
                PTX_GELU_ERF_F16,
            )?;
            let out = unsafe { device.alloc::<f16>(n) }?;
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&out);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        dtype => Err(Error::UnsupportedDTypeForOp(dtype, OP)),
    }
}

pub(crate) fn cuda_bwd_dx(
    s1: &CudaStorage,
    l1: &Layout,
    s2: &CudaStorage,
    l2: &Layout,
) -> Result<(CudaStorage, Shape)> {
    const OP: &str = "gelu_erf_fused_bwd_dx";
    if l1.dims() != l2.dims() {
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
    let (o1, o2) = check_domain(l1, OP)?;
    let (d1, d2) = l2
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;
    let n = l1.shape().elem_count();
    super::check_elem_count_fits_u32(OP, n)?;
    let device = s1.device().clone();
    let shape = l1.shape().clone();
    let cfg = super::elemwise_launch_config(n as u32);

    match s1.dtype() {
        DType::F32 => {
            let x = s1.as_cuda_slice::<f32>()?.slice(o1..o2);
            let dy = s2.as_cuda_slice::<f32>()?.slice(d1..d2);
            let func =
                device.get_or_load_custom_func("gelu_erf_bwd_dx_f32", MODULE_NAME, PTX_GELU_ERF)?;
            let out = unsafe { device.alloc::<f32>(n) }?;
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&dy);
            builder.arg(&out);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        DType::BF16 => {
            let x = s1.as_cuda_slice::<bf16>()?.slice(o1..o2);
            let dy = s2.as_cuda_slice::<bf16>()?.slice(d1..d2);
            let func = device.get_or_load_custom_func(
                "gelu_erf_bwd_dx_bf16",
                MODULE_NAME,
                PTX_GELU_ERF,
            )?;
            let out = unsafe { device.alloc::<bf16>(n) }?;
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&dy);
            builder.arg(&out);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        DType::F16 => {
            let x = s1.as_cuda_slice::<f16>()?.slice(o1..o2);
            let dy = s2.as_cuda_slice::<f16>()?.slice(d1..d2);
            let func = device.get_or_load_custom_func(
                "gelu_erf_bwd_dx_f16",
                MODULE_NAME_F16,
                PTX_GELU_ERF_F16,
            )?;
            let out = unsafe { device.alloc::<f16>(n) }?;
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&dy);
            builder.arg(&out);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        dtype => Err(Error::UnsupportedDTypeForOp(dtype, OP)),
    }
}
