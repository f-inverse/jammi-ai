use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::{CudaStorage, DType, Error, Layout, Result, Shape};
use half::bf16;

use super::PTX_ROPE;
use crate::ops::MAX_HEAD_DIM;

/// See `../axpy.rs`'s identical constant for the module-name rationale —
/// arbitrary but stable and unique to this op's PTX module.
const MODULE_NAME: &str = "jammi_kernels_rope";

fn rope_dims(
    l1: &Layout,
    l2: &Layout,
    l3: &Layout,
    op: &'static str,
) -> Result<(usize, usize, usize)> {
    let hidden = *l1.dims().last().ok_or_else(|| {
        Error::Msg(format!(
            "{op}: input must have rank >= 1 to define a last (head_dim) dimension"
        ))
    })?;
    if l2.dims() != l3.dims() {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l2.shape().clone(),
            rhs: l3.shape().clone(),
            op,
        });
    }
    let Some(&cos_last) = l2.dims().last() else {
        return Err(Error::Msg(format!(
            "{op}: cos/sin must have rank >= 1 to define a last (head_dim) dimension"
        )));
    };
    if cos_last != hidden {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l1.shape().clone(),
            rhs: l2.shape().clone(),
            op,
        });
    }
    if hidden == 0 {
        return Ok((0, 0, 0));
    }
    if !hidden.is_multiple_of(2) {
        return Err(Error::Msg(format!(
            "{op}: head_dim={hidden} must be even — rotate-half splits it into two \
             equal halves"
        )));
    }
    let cos_elems = l2.shape().elem_count();
    if cos_elems == 0 || !cos_elems.is_multiple_of(hidden) {
        return Err(Error::Msg(format!(
            "{op}: cos/sin element count {cos_elems} is not a positive multiple of \
             head_dim={hidden}"
        )));
    }
    let period = cos_elems / hidden;
    // SUFFICIENT check, matching `ops::rope::rope_dims`'s identical
    // reasoning exactly (see that function's doc): `total_rows % period
    // == 0` alone is not enough — the axis immediately before `hidden`
    // must equal `period` (unless `period == 1`, a one-row table that
    // broadcasts safely over any axis size — `row % 1 == 0` always), or
    // `row % period` (this kernel's own indexing, `rope.cu`) walks the
    // wrong axis and silently misreads the table.
    let x_dims = l1.dims();
    let axis_before_hidden = if x_dims.len() >= 2 {
        x_dims[x_dims.len() - 2]
    } else {
        1
    };
    if period != 1 && axis_before_hidden != period {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l1.shape().clone(),
            rhs: l2.shape().clone(),
            op,
        });
    }
    let x_elems = l1.shape().elem_count();
    let total_rows = x_elems / hidden;
    Ok((hidden, period, total_rows))
}

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
    if n > u32::MAX as usize {
        return Err(Error::Msg(format!(
            "{op}: {n} elements exceeds u32::MAX; the CUDA launch grid and the \
             kernel's indices are both 32-bit"
        )));
    }
    Ok(())
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

    if hidden == 0 || n == 0 {
        return match s1.dtype() {
            DType::F32 => {
                let out = unsafe { device.alloc::<f32>(0) }?;
                Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
            }
            DType::BF16 => {
                let out = unsafe { device.alloc::<bf16>(0) }?;
                Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
            }
            dtype => Err(Error::UnsupportedDTypeForOp(dtype, OP)),
        };
    }
    check_cuda_domain(OP, n, hidden)?;

    let (x1, x2) = l1
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;
    let (c1, c2) = l2
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;
    let (s_1, s_2) = l3
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
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
        dtype => Err(Error::UnsupportedDTypeForOp(dtype, OP)),
    }
}
