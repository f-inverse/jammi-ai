use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::PushKernelArg;
use candle_core::{CudaStorage, DType, Error, Layout, Result, Shape};
use half::{bf16, f16};

use super::{PTX_ROPE, PTX_ROPE_F16};
use crate::ops::rope::rope_dims;
use crate::ops::MAX_HEAD_DIM;

/// See `../axpy.rs`'s identical constant for the module-name rationale —
/// arbitrary but stable and unique to this op's PTX module.
const MODULE_NAME: &str = "jammi_kernels_rope";

/// The F16 arm's OWN PTX module name — `rope_f16.cu` is a SEPARATE
/// translation unit (see that file's module doc), so it needs a distinct
/// module name from [`MODULE_NAME`].
const MODULE_NAME_F16: &str = "jammi_kernels_rope_f16";

/// `hidden > MAX_HEAD_DIM`: refused above a conservative, VALIDATED
/// ceiling — NOT a hardware limit (see `ops::MAX_HEAD_DIM`'s doc). `n >
/// u32::MAX`: the kernel's own indices (`hidden`, `period` arguments) and
/// the launch grid are 32-bit, exactly the guard `axpy.rs`/`layer_norm.rs`
/// document for the same reason.
fn check_cuda_domain(op: &'static str, n: usize, hidden: usize) -> Result<()> {
    if hidden > MAX_HEAD_DIM {
        return Err(Error::Msg(format!(
            "{op}: head_dim={hidden} exceeds the CUDA kernel's MAX_HEAD_DIM={MAX_HEAD_DIM} \
             (a conservative validated ceiling, not a hardware limit — see \
             ops::MAX_HEAD_DIM's doc); the CPU arm has no such ceiling"
        )));
    }
    super::check_elem_count_fits_u32(op, n)
}

pub(crate) fn cuda_fwd(
    negate_sin: bool,
    s1: &CudaStorage,
    l1: &Layout,
    s2: &CudaStorage,
    l2: &Layout,
    s3: &CudaStorage,
    l3: &Layout,
) -> Result<(CudaStorage, Shape)> {
    const OP: &str = "rope_fused";
    let (hidden, period, _total_rows) = rope_dims(l1, l2, l3, OP)?;
    let shape = l1.shape().clone();
    let device = s1.device().clone();
    let n = l1.shape().elem_count();

    if s1.dtype() != s2.dtype() || s1.dtype() != s3.dtype() {
        return Err(Error::DTypeMismatchBinaryOp {
            lhs: s1.dtype(),
            rhs: s2.dtype(),
            op: OP,
        });
    }

    // `hidden == 0` is checked on its own — matching `cpu_fwd`'s domain
    // exactly: `ops::rope::RopeFused::cpu_fwd` exempts contiguity ONLY
    // when `hidden == 0` (its own early `empty_like` return, BEFORE
    // `contiguous_offsets()`), never for a broader `n == 0`. The
    // dtype-agreement check above already guarantees `s1.dtype() ==
    // s2.dtype() == s3.dtype()` here.
    if hidden == 0 {
        return Ok((super::alloc_empty(&device, s1.dtype(), OP)?, shape));
    }

    // Contiguity is checked NEXT, before the (now `hidden != 0`) `n == 0`
    // fast path below — a `total_rows == 0` (`n == 0` with `hidden != 0`)
    // empty-but-non-contiguous layout still falls through `cpu_fwd`'s OWN
    // `contiguous_offsets()` calls (only `hidden == 0`, handled above,
    // skips them there), so this arm must refuse the same layout rather
    // than silently admitting it through a combined `hidden == 0 || n ==
    // 0` fast path — the exact class of divergence this fix closes.
    let (x1, x2) = l1
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;
    let (c1, c2) = l2
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;
    let (s_1, s_2) = l3
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;

    if n == 0 {
        return Ok((super::alloc_empty(&device, s1.dtype(), OP)?, shape));
    }
    check_cuda_domain(OP, n, hidden)?;

    let cfg = super::elemwise_launch_config(n as u32);
    let hidden_u32 = hidden as u32;
    let period_u32 = period as u32;
    let sign: f32 = if negate_sin { -1.0 } else { 1.0 };

    match s1.dtype() {
        DType::F32 => {
            let x = s1.as_cuda_slice::<f32>()?.slice(x1..x2);
            let cos = s2.as_cuda_slice::<f32>()?.slice(c1..c2);
            let sin = s3.as_cuda_slice::<f32>()?.slice(s_1..s_2);
            let func = device.get_or_load_custom_func("rope_fwd_f32", MODULE_NAME, PTX_ROPE)?;
            let out = unsafe { device.alloc::<f32>(n) }?;
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&cos);
            builder.arg(&sin);
            builder.arg(&out);
            builder.arg(&hidden_u32);
            builder.arg(&period_u32);
            builder.arg(&sign);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        DType::BF16 => {
            let x = s1.as_cuda_slice::<bf16>()?.slice(x1..x2);
            let cos = s2.as_cuda_slice::<bf16>()?.slice(c1..c2);
            let sin = s3.as_cuda_slice::<bf16>()?.slice(s_1..s_2);
            let func = device.get_or_load_custom_func("rope_fwd_bf16", MODULE_NAME, PTX_ROPE)?;
            let out = unsafe { device.alloc::<bf16>(n) }?;
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&cos);
            builder.arg(&sin);
            builder.arg(&out);
            builder.arg(&hidden_u32);
            builder.arg(&period_u32);
            builder.arg(&sign);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        DType::F16 => {
            let x = s1.as_cuda_slice::<f16>()?.slice(x1..x2);
            let cos = s2.as_cuda_slice::<f16>()?.slice(c1..c2);
            let sin = s3.as_cuda_slice::<f16>()?.slice(s_1..s_2);
            let func =
                device.get_or_load_custom_func("rope_fwd_f16", MODULE_NAME_F16, PTX_ROPE_F16)?;
            let out = unsafe { device.alloc::<f16>(n) }?;
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&cos);
            builder.arg(&sin);
            builder.arg(&out);
            builder.arg(&hidden_u32);
            builder.arg(&period_u32);
            builder.arg(&sign);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        dtype => Err(Error::UnsupportedDTypeForOp(dtype, OP)),
    }
}
