use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::PushKernelArg;
use candle_core::{CudaStorage, DType, Error, Layout, Result, Shape};
use half::bf16;

use super::PTX_SCALED_CAST_ADD;

/// The kernel module name PTX functions are loaded under — arbitrary, but
/// stable and unique to this op (mirrors `crate::cuda::axpy::MODULE_NAME`).
const MODULE_NAME: &str = "jammi_kernels_scaled_cast_add";

pub(crate) fn cuda_fwd(
    scaling: f64,
    s1: &CudaStorage,
    l1: &Layout,
    s2: &CudaStorage,
    l2: &Layout,
) -> Result<(CudaStorage, Shape)> {
    if l1.dims() != l2.dims() {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l1.shape().clone(),
            rhs: l2.shape().clone(),
            op: "scaled_cast_add",
        });
    }

    let device = s1.device().clone();
    let n = l1.shape().elem_count();
    let shape = l1.shape().clone();
    let scaling_f32 = scaling as f32;

    // n == 0: `LaunchConfig::for_num_elems(0)` yields grid_dim (0, 1, 1) —
    // an illegal launch. Match the CPU arm's documented no-op contract
    // (`ops::scaled_cast_add`'s `empty_tensor_is_a_no_op_not_an_error`)
    // instead of ever reaching the launch below.
    if n == 0 {
        // Output dtype follows `base_dtype` (`s1`) regardless of
        // `lora_dtype` (`s2`) — the same rule the non-empty match below
        // encodes across its four dtype-pair arms. Each tensor's dtype is
        // validated independently (base first, matching the original
        // match's arm order) before deferring the actual empty-alloc
        // dispatch to the shared helper.
        if !matches!(s1.dtype(), DType::F32 | DType::BF16) {
            return Err(Error::UnsupportedDTypeForOp(s1.dtype(), "scaled_cast_add"));
        }
        if !matches!(s2.dtype(), DType::F32 | DType::BF16) {
            return Err(Error::UnsupportedDTypeForOp(s2.dtype(), "scaled_cast_add"));
        }
        return Ok((
            super::alloc_empty(&device, s1.dtype(), "scaled_cast_add")?,
            shape,
        ));
    }

    // K2 (refuse, don't compute past the domain): see `crate::cuda::axpy`'s
    // identical comment — the launch grid and the kernel's own bounds
    // check are both 32-bit.
    super::check_elem_count_fits_u32("scaled_cast_add", n)?;

    // Same `contiguous_offsets()` requirement (and the same nonzero-
    // start_offset rationale) as `crate::cuda::axpy::cuda_fwd`.
    let (o1_base, o2_base) = l1.contiguous_offsets().ok_or(Error::RequiresContiguous {
        op: "scaled_cast_add",
    })?;
    let (o1_lora, o2_lora) = l2.contiguous_offsets().ok_or(Error::RequiresContiguous {
        op: "scaled_cast_add",
    })?;

    let cfg = super::elemwise_launch_config(n as u32);

    match (s1.dtype(), s2.dtype()) {
        (DType::F32, DType::F32) => {
            let base = s1.as_cuda_slice::<f32>()?.slice(o1_base..o2_base);
            let lora = s2.as_cuda_slice::<f32>()?.slice(o1_lora..o2_lora);
            let func = device.get_or_load_custom_func(
                "scaled_cast_add_f32_f32",
                MODULE_NAME,
                PTX_SCALED_CAST_ADD,
            )?;
            let out = unsafe { device.alloc::<f32>(n) }?;
            let mut builder = func.builder();
            builder.arg(&scaling_f32);
            builder.arg(&base);
            builder.arg(&lora);
            builder.arg(&out);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        (DType::F32, DType::BF16) => {
            let base = s1.as_cuda_slice::<f32>()?.slice(o1_base..o2_base);
            let lora = s2.as_cuda_slice::<bf16>()?.slice(o1_lora..o2_lora);
            let func = device.get_or_load_custom_func(
                "scaled_cast_add_f32_bf16",
                MODULE_NAME,
                PTX_SCALED_CAST_ADD,
            )?;
            let out = unsafe { device.alloc::<f32>(n) }?;
            let mut builder = func.builder();
            builder.arg(&scaling_f32);
            builder.arg(&base);
            builder.arg(&lora);
            builder.arg(&out);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        (DType::BF16, DType::F32) => {
            let base = s1.as_cuda_slice::<bf16>()?.slice(o1_base..o2_base);
            let lora = s2.as_cuda_slice::<f32>()?.slice(o1_lora..o2_lora);
            let func = device.get_or_load_custom_func(
                "scaled_cast_add_bf16_f32",
                MODULE_NAME,
                PTX_SCALED_CAST_ADD,
            )?;
            let out = unsafe { device.alloc::<bf16>(n) }?;
            let mut builder = func.builder();
            builder.arg(&scaling_f32);
            builder.arg(&base);
            builder.arg(&lora);
            builder.arg(&out);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        (DType::BF16, DType::BF16) => {
            let base = s1.as_cuda_slice::<bf16>()?.slice(o1_base..o2_base);
            let lora = s2.as_cuda_slice::<bf16>()?.slice(o1_lora..o2_lora);
            let func = device.get_or_load_custom_func(
                "scaled_cast_add_bf16_bf16",
                MODULE_NAME,
                PTX_SCALED_CAST_ADD,
            )?;
            let out = unsafe { device.alloc::<bf16>(n) }?;
            let mut builder = func.builder();
            builder.arg(&scaling_f32);
            builder.arg(&base);
            builder.arg(&lora);
            builder.arg(&out);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        (base_dtype, _) if !matches!(base_dtype, DType::F32 | DType::BF16) => {
            Err(Error::UnsupportedDTypeForOp(base_dtype, "scaled_cast_add"))
        }
        (_, lora_dtype) => Err(Error::UnsupportedDTypeForOp(lora_dtype, "scaled_cast_add")),
    }
}
