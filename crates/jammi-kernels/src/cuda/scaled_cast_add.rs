use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::PushKernelArg;
use candle_core::{CudaStorage, DType, Error, Layout, Result, Shape};
use half::{bf16, f16};

use super::{PTX_SCALED_CAST_ADD, PTX_SCALED_CAST_ADD_F16};

/// The kernel module name PTX functions are loaded under — arbitrary, but
/// stable and unique to this op (see `crate::cuda`'s module doc).
const MODULE_NAME: &str = "jammi_kernels_scaled_cast_add";

/// The F16 arms' OWN PTX module name (campaign #443 W2c) —
/// `scaled_cast_add_f16.cu` is a SEPARATE translation unit (see that file's
/// module doc), so it needs a distinct module name from [`MODULE_NAME`].
const MODULE_NAME_F16: &str = "jammi_kernels_scaled_cast_add_f16";

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

    // Contiguity is checked FIRST, unconditionally -- even before the
    // `n == 0` fast path below -- so this arm's own domain (`RequiresContiguous`,
    // the same nonzero-start_offset rationale `crate::cuda`'s module doc gives)
    // applies identically to an empty tensor rather than being silently
    // ADMITTED through the fast path (`cpu_fwd` itself never requires
    // contiguity -- it walks `StridedOffsets` -- so this is the CUDA arm's
    // OWN self-consistency, matching every other elementwise op's
    // documented CUDA domain in this crate). `o1_base`/`o2_base`/
    // `o1_lora`/`o2_lora` are unused by the `n == 0` branch itself --
    // computed here only so the domain check runs in the same place for
    // both branches.
    let (o1_base, o2_base) = l1.contiguous_offsets().ok_or(Error::RequiresContiguous {
        op: "scaled_cast_add",
    })?;
    let (o1_lora, o2_lora) = l2.contiguous_offsets().ok_or(Error::RequiresContiguous {
        op: "scaled_cast_add",
    })?;

    // n == 0: `LaunchConfig::for_num_elems(0)` yields grid_dim (0, 1, 1) —
    // an illegal launch. Match the CPU arm's documented no-op contract
    // (`ops::scaled_cast_add`'s `empty_tensor_is_a_no_op_not_an_error`)
    // instead of ever reaching the launch below.
    if n == 0 {
        // Output dtype follows `base_dtype` (`s1`) regardless of
        // `lora_dtype` (`s2`) — the same rule the non-empty match below
        // encodes across its seven dtype-pair arms. The PAIR (not each
        // dtype checked independently) is validated against the EXACT
        // combinations the non-empty match dispatches: `BF16`+`F16` and
        // `F16`+`BF16` are each independently a "known" dtype for this op
        // on ONE operand, but neither pair is a supported COMBINATION (no
        // such kernel exists) — checking each side separately would
        // silently ADMIT that unsupported pair at `n == 0` while the
        // non-empty match refuses it, exactly the shape-dependent dtype
        // domain split campaign #443's D1 audit named for `alloc_empty`'s
        // callers (family D). Matching the pair directly here closes it.
        match (s1.dtype(), s2.dtype()) {
            (DType::F32, DType::F32)
            | (DType::F32, DType::BF16)
            | (DType::BF16, DType::F32)
            | (DType::BF16, DType::BF16)
            | (DType::F16, DType::F32)
            | (DType::F32, DType::F16)
            | (DType::F16, DType::F16) => {}
            (base_dtype, _) if !matches!(base_dtype, DType::F32 | DType::BF16 | DType::F16) => {
                return Err(Error::UnsupportedDTypeForOp(base_dtype, "scaled_cast_add"));
            }
            (_, lora_dtype) => {
                return Err(Error::UnsupportedDTypeForOp(lora_dtype, "scaled_cast_add"));
            }
        }
        return Ok((
            super::alloc_empty(&device, s1.dtype(), "scaled_cast_add")?,
            shape,
        ));
    }

    // K2 (refuse, don't compute past the domain): see
    // `crate::ops::launch_domain::check_elem_count_fits_u32`'s own doc —
    // the launch grid and the kernel's own bounds check are both 32-bit.
    super::check_elem_count_fits_u32("scaled_cast_add", n)?;

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
        (DType::F16, DType::F32) => {
            let base = s1.as_cuda_slice::<f16>()?.slice(o1_base..o2_base);
            let lora = s2.as_cuda_slice::<f32>()?.slice(o1_lora..o2_lora);
            let func = device.get_or_load_custom_func(
                "scaled_cast_add_f16_f32",
                MODULE_NAME_F16,
                PTX_SCALED_CAST_ADD_F16,
            )?;
            let out = unsafe { device.alloc::<f16>(n) }?;
            let mut builder = func.builder();
            builder.arg(&scaling_f32);
            builder.arg(&base);
            builder.arg(&lora);
            builder.arg(&out);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        (DType::F32, DType::F16) => {
            let base = s1.as_cuda_slice::<f32>()?.slice(o1_base..o2_base);
            let lora = s2.as_cuda_slice::<f16>()?.slice(o1_lora..o2_lora);
            let func = device.get_or_load_custom_func(
                "scaled_cast_add_f32_f16",
                MODULE_NAME_F16,
                PTX_SCALED_CAST_ADD_F16,
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
        (DType::F16, DType::F16) => {
            let base = s1.as_cuda_slice::<f16>()?.slice(o1_base..o2_base);
            let lora = s2.as_cuda_slice::<f16>()?.slice(o1_lora..o2_lora);
            let func = device.get_or_load_custom_func(
                "scaled_cast_add_f16_f16",
                MODULE_NAME_F16,
                PTX_SCALED_CAST_ADD_F16,
            )?;
            let out = unsafe { device.alloc::<f16>(n) }?;
            let mut builder = func.builder();
            builder.arg(&scaling_f32);
            builder.arg(&base);
            builder.arg(&lora);
            builder.arg(&out);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        (base_dtype, _) if !matches!(base_dtype, DType::F32 | DType::BF16 | DType::F16) => {
            Err(Error::UnsupportedDTypeForOp(base_dtype, "scaled_cast_add"))
        }
        (_, lora_dtype) => Err(Error::UnsupportedDTypeForOp(lora_dtype, "scaled_cast_add")),
    }
}
