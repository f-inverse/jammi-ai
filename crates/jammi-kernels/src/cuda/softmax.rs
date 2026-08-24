use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::{CudaStorage, DType, Error, Layout, Result, Shape};
use half::bf16;

use super::PTX_SOFTMAX;
use crate::ops::softmax::{softmax_dims, FullyMaskedPolicy, MAX_LAST_DIM, MAX_RANK};

/// `FullyMaskedPolicy` as the `u32` flag the CUDA kernels take (`1` =
/// `Zeros`, `0` = `Propagate`) — construction data threaded through from
/// `SoftmaxLastDimFused::fully_masked`, the SAME value the CPU arm's
/// `softmax_row_f32`/`softmax_row_bf16` receive, never a runtime
/// per-tensor predicate.
fn policy_flag(policy: FullyMaskedPolicy) -> u32 {
    match policy {
        FullyMaskedPolicy::Zeros => 1,
        FullyMaskedPolicy::Propagate => 0,
    }
}

/// See `../../cuda/axpy.rs`'s identical constant for the module-name
/// rationale — arbitrary but stable and unique to this op's PTX module.
const MODULE_NAME: &str = "jammi_kernels_softmax";

/// One CUDA thread block per row; must match `SM_BLOCK` in `softmax.cu`.
const SM_BLOCK: u32 = 256;

/// `last > MAX_LAST_DIM`: refused above a conservative, VALIDATED ceiling —
/// NOT a hardware limit (see `ops::softmax::MAX_LAST_DIM`'s doc). `n >
/// u32::MAX`: the launch grid and the kernel's own indices are 32-bit,
/// exactly the guard `axpy.rs`/`layer_norm.rs` document for the same
/// reason.
fn check_last_and_n(op: &'static str, n: usize, last: usize) -> Result<()> {
    if last > MAX_LAST_DIM {
        return Err(Error::Msg(format!(
            "{op}: last={last} exceeds the CUDA kernel's MAX_LAST_DIM={MAX_LAST_DIM} \
             (a conservative validated ceiling, not a hardware limit — see \
             ops::softmax::MAX_LAST_DIM's doc); the CPU arm has no such ceiling"
        )));
    }
    super::check_elem_count_fits_u32(op, n)
}

/// `rank > MAX_RANK`: refused because the CUDA kernel's mask-broadcast
/// index takes exactly THREE leading-axis-size scalar arguments per
/// tensor — a REAL implementation constraint, unlike `MAX_LAST_DIM` (see
/// `ops::softmax::MAX_RANK`'s doc).
fn check_rank(op: &'static str, rank: usize) -> Result<()> {
    if rank > MAX_RANK {
        return Err(Error::Msg(format!(
            "{op}: rank={rank} exceeds the CUDA kernel's MAX_RANK={MAX_RANK} (a real \
             fixed-arity constraint of the mask-broadcast index, not a validated-coverage \
             ceiling — see ops::softmax::MAX_RANK's doc); the CPU arm has no such ceiling"
        )));
    }
    Ok(())
}

/// Packs `lead` (a tensor's own leading-axis sizes, length `<= 3` —
/// callers check `rank <= MAX_RANK` first) into the kernel's fixed
/// 3-scalar signature, left-padding with `1` when `lead` has fewer than
/// three entries: a virtual size-1 axis is a true no-op in the kernel's
/// decompose/re-ravel arithmetic (`row % 1 == 0`, `row / 1 == row`), so no
/// separate "how many axes are real" parameter is needed.
fn lead_dims3(lead: &[usize]) -> [u32; 3] {
    let mut out = [1u32; 3];
    let n = lead.len();
    for (i, &d) in lead.iter().enumerate() {
        out[3 - n + i] = d as u32;
    }
    out
}

pub(crate) fn cuda_fwd(
    s1: &CudaStorage,
    l1: &Layout,
    s2: &CudaStorage,
    l2: &Layout,
    fully_masked: FullyMaskedPolicy,
) -> Result<(CudaStorage, Shape)> {
    const OP: &str = "softmax_last_dim_fused";
    let (rows, last) = softmax_dims(l1, l2, OP)?;
    let shape = l1.shape().clone();
    let device = s1.device().clone();
    let n = l1.shape().elem_count();

    if last == 0 || n == 0 {
        if s1.dtype() != s2.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: OP,
            });
        }
        return Ok((super::alloc_empty(&device, s1.dtype(), OP)?, shape));
    }
    let rank = l1.dims().len();
    check_rank(OP, rank)?;
    check_last_and_n(OP, n, last)?;

    let (o1, o2) = l1
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;
    let (m1, m2) = l2
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;
    let s_lead = lead_dims3(&l1.dims()[..rank - 1]);
    let m_lead = lead_dims3(&l2.dims()[..rank - 1]);

    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (SM_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let last_u32 = last as u32;
    let policy_u32 = policy_flag(fully_masked);

    match (s1.dtype(), s2.dtype()) {
        (DType::F32, DType::F32) => {
            let sc = s1.as_cuda_slice::<f32>()?.slice(o1..o2);
            let mk = s2.as_cuda_slice::<f32>()?.slice(m1..m2);
            let func =
                device.get_or_load_custom_func("softmax_fwd_f32", MODULE_NAME, PTX_SOFTMAX)?;
            let out = unsafe { device.alloc::<f32>(n) }?;
            let mut builder = func.builder();
            builder.arg(&sc);
            builder.arg(&mk);
            builder.arg(&out);
            builder.arg(&last_u32);
            builder.arg(&s_lead[0]);
            builder.arg(&s_lead[1]);
            builder.arg(&s_lead[2]);
            builder.arg(&m_lead[0]);
            builder.arg(&m_lead[1]);
            builder.arg(&m_lead[2]);
            builder.arg(&policy_u32);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        (DType::BF16, DType::BF16) => {
            let sc = s1.as_cuda_slice::<bf16>()?.slice(o1..o2);
            let mk = s2.as_cuda_slice::<bf16>()?.slice(m1..m2);
            let func =
                device.get_or_load_custom_func("softmax_fwd_bf16", MODULE_NAME, PTX_SOFTMAX)?;
            let out = unsafe { device.alloc::<bf16>(n) }?;
            let mut builder = func.builder();
            builder.arg(&sc);
            builder.arg(&mk);
            builder.arg(&out);
            builder.arg(&last_u32);
            builder.arg(&s_lead[0]);
            builder.arg(&s_lead[1]);
            builder.arg(&s_lead[2]);
            builder.arg(&m_lead[0]);
            builder.arg(&m_lead[1]);
            builder.arg(&m_lead[2]);
            builder.arg(&policy_u32);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        (lhs, rhs) if lhs != rhs => Err(Error::DTypeMismatchBinaryOp { lhs, rhs, op: OP }),
        (dtype, _) => Err(Error::UnsupportedDTypeForOp(dtype, OP)),
    }
}

pub(crate) fn cuda_bwd_dscores(
    s1: &CudaStorage,
    l1: &Layout,
    s2: &CudaStorage,
    l2: &Layout,
) -> Result<(CudaStorage, Shape)> {
    const OP: &str = "softmax_last_dim_fused_bwd_dscores";
    if l1.dims() != l2.dims() {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l1.shape().clone(),
            rhs: l2.shape().clone(),
            op: OP,
        });
    }
    let dims = l1.dims();
    let last = *dims.last().ok_or_else(|| {
        Error::Msg(format!(
            "{OP}: input must have rank >= 1 to define a last (reduction) dimension"
        ))
    })?;
    let shape = l1.shape().clone();
    let device = s1.device().clone();
    let n = l1.shape().elem_count();

    if last == 0 || n == 0 {
        if s1.dtype() != s2.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: OP,
            });
        }
        return Ok((super::alloc_empty(&device, s1.dtype(), OP)?, shape));
    }
    check_last_and_n(OP, n, last)?;
    let rows = n / last;

    let (o1, o2) = l1
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;
    let (d1, d2) = l2
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;

    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (SM_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let last_u32 = last as u32;

    match (s1.dtype(), s2.dtype()) {
        (DType::F32, DType::F32) => {
            let y = s1.as_cuda_slice::<f32>()?.slice(o1..o2);
            let dy = s2.as_cuda_slice::<f32>()?.slice(d1..d2);
            let func = device.get_or_load_custom_func(
                "softmax_bwd_dscores_f32",
                MODULE_NAME,
                PTX_SOFTMAX,
            )?;
            let out = unsafe { device.alloc::<f32>(n) }?;
            let mut builder = func.builder();
            builder.arg(&y);
            builder.arg(&dy);
            builder.arg(&out);
            builder.arg(&last_u32);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        (DType::BF16, DType::BF16) => {
            let y = s1.as_cuda_slice::<bf16>()?.slice(o1..o2);
            let dy = s2.as_cuda_slice::<bf16>()?.slice(d1..d2);
            let func = device.get_or_load_custom_func(
                "softmax_bwd_dscores_bf16",
                MODULE_NAME,
                PTX_SOFTMAX,
            )?;
            let out = unsafe { device.alloc::<bf16>(n) }?;
            let mut builder = func.builder();
            builder.arg(&y);
            builder.arg(&dy);
            builder.arg(&out);
            builder.arg(&last_u32);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        (lhs, rhs) if lhs != rhs => Err(Error::DTypeMismatchBinaryOp { lhs, rhs, op: OP }),
        (dtype, _) => Err(Error::UnsupportedDTypeForOp(dtype, OP)),
    }
}
