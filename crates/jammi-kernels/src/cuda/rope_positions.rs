use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::PushKernelArg;
use candle_core::{CudaStorage, DType, Error, Layout, Result, Shape};
use half::bf16;

use super::PTX_ROPE_POSITIONS;
use crate::ops::rope_positions::{rope_positions_dims, PositionArm};
use crate::ops::MAX_HEAD_DIM;

/// See `../axpy.rs`'s identical constant for the module-name rationale.
const MODULE_NAME: &str = "jammi_kernels_rope_positions";

/// Same shape as `../rope.rs`'s `check_cuda_domain`: `d > MAX_HEAD_DIM`
/// (a validated-coverage ceiling, not a hardware one) and `n >
/// u32::MAX` (the kernel's own 32-bit indices/launch grid).
fn check_cuda_domain(op: &'static str, n: usize, d: usize) -> Result<()> {
    if d > MAX_HEAD_DIM {
        return Err(Error::Msg(format!(
            "{op}: head_dim={d} exceeds the CUDA kernel's MAX_HEAD_DIM={MAX_HEAD_DIM} \
             (a conservative validated ceiling, not a hardware limit); the CPU arm has no \
             such ceiling"
        )));
    }
    super::check_elem_count_fits_u32(op, n)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cuda_fwd(
    seq: usize,
    negate_sin: bool,
    arm: PositionArm,
    s1: &CudaStorage,
    l1: &Layout,
    s2: &CudaStorage,
    l2: &Layout,
    s3: &CudaStorage,
    l3: &Layout,
) -> Result<(CudaStorage, Shape)> {
    const OP: &str = "rope_positions_fused";
    let (total, h, d) = rope_positions_dims(l1, l2, l3, seq, arm)?;
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

    // `d == 0` is checked on its own — matching `cpu_fwd`'s domain
    // exactly: `ops::rope_positions::RopePositionsFused::cpu_fwd` exempts
    // contiguity ONLY when `d == 0` (its own early `empty_like` return,
    // BEFORE `contiguous_offsets()`), never for a broader `n == 0`.
    if d == 0 {
        return Ok((super::alloc_empty(&device, s1.dtype(), OP)?, shape));
    }

    // Contiguity is checked NEXT, before the (now `d != 0`) `n == 0` fast
    // path below — a `total == 0` (`n == 0` with `d != 0`)
    // empty-but-non-contiguous layout still falls through `cpu_fwd`'s OWN
    // `contiguous_offsets()` calls (only `d == 0`, handled above, skips
    // them there), so this arm must refuse the same layout rather than
    // silently admitting it through a combined `d == 0 || n == 0` fast
    // path — the exact class of divergence this fix closes.
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
    check_cuda_domain(OP, n, d)?;

    let cfg = super::elemwise_launch_config(n as u32);
    let h_u32 = h as u32;
    let d_u32 = d as u32;
    let seq_u32 = seq as u32;
    let sign: f32 = if negate_sin { -1.0 } else { 1.0 };
    let _ = total; // carried for readability/doc parity with rope_positions_dims's tuple

    match s1.dtype() {
        DType::F32 => {
            let qkv = s1.as_cuda_slice::<f32>()?.slice(x1..x2);
            let cos = s2.as_cuda_slice::<f32>()?.slice(c1..c2);
            let sin = s3.as_cuda_slice::<f32>()?.slice(s_1..s_2);
            let func = device.get_or_load_custom_func(
                "rope_positions_fwd_f32",
                MODULE_NAME,
                PTX_ROPE_POSITIONS,
            )?;
            let out = unsafe { device.alloc::<f32>(n) }?;
            let mut builder = func.builder();
            builder.arg(&qkv);
            builder.arg(&cos);
            builder.arg(&sin);
            builder.arg(&out);
            builder.arg(&h_u32);
            builder.arg(&d_u32);
            builder.arg(&seq_u32);
            builder.arg(&sign);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        DType::BF16 => {
            let qkv = s1.as_cuda_slice::<bf16>()?.slice(x1..x2);
            let cos = s2.as_cuda_slice::<bf16>()?.slice(c1..c2);
            let sin = s3.as_cuda_slice::<bf16>()?.slice(s_1..s_2);
            let func = device.get_or_load_custom_func(
                "rope_positions_fwd_bf16",
                MODULE_NAME,
                PTX_ROPE_POSITIONS,
            )?;
            let out = unsafe { device.alloc::<bf16>(n) }?;
            let mut builder = func.builder();
            builder.arg(&qkv);
            builder.arg(&cos);
            builder.arg(&sin);
            builder.arg(&out);
            builder.arg(&h_u32);
            builder.arg(&d_u32);
            builder.arg(&seq_u32);
            builder.arg(&sign);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        dtype => Err(Error::UnsupportedDTypeForOp(dtype, OP)),
    }
}
