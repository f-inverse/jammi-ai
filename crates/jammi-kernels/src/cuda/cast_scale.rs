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

    // Contiguity is checked FIRST, unconditionally -- even before the
    // `n == 0` fast path below -- so a non-contiguous VIEW is refused the
    // same way whether or not it happens to be empty, matching this op's
    // own documented CUDA domain (module doc: "The CUDA arm additionally
    // REQUIRES contiguous storage"; `cpu_fwd` itself never requires
    // contiguity -- it walks `StridedOffsets` -- so this is the CUDA arm's
    // OWN self-consistency, not a match to a `cpu_fwd` requirement). `o1`/
    // `o2` are unused by the `n == 0` branch itself -- computed here only
    // so the domain check runs in the same place for both branches.
    let (o1, o2) = l1
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;

    // n == 0: match every other op's CUDA glue in this crate — an explicit
    // empty allocation, never a `LaunchConfig::for_num_elems(0)` launch
    // (grid_dim (0, 1, 1) is illegal).
    if n == 0 {
        return Ok((super::alloc_empty(&device, DType::F32, OP)?, shape));
    }

    super::check_elem_count_fits_u32(OP, n)?;

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

/// Launches the IDENTICAL kernel [`cuda_fwd_cast_scale_bf16_f32`] does,
/// but writes into an EXISTING, caller-owned output storage instead of
/// allocating a fresh one every call. Test-only scaffolding (no
/// `admit`/`DispatchCounters` involvement, not part of this crate's op
/// dispatch surface) — see `../ops/cast_scale.rs`'s `cast_scale_bf16_f32_into`
/// (the `#[doc(hidden)] pub` entry point `tests/cuda_parity.rs`'s isolated-
/// timing harness calls) for why this exists: cudarc has no caching
/// allocator, so `cuda_fwd_cast_scale_bf16_f32`'s own `device.alloc::<f32>(n)`
/// (a fresh `cuMemAlloc`, freed via `Drop` at the end of every timed
/// iteration in the wrapper-including-alloc path) dominates wall-clock at
/// this op's 151 MB production output width — measured directly rather
/// than assumed, see that test's own printed numbers.
pub(crate) fn cuda_launch_cast_scale_bf16_f32_into(
    scale: f64,
    s1: &CudaStorage,
    l1: &Layout,
    out: &CudaStorage,
    l_out: &Layout,
) -> Result<()> {
    const OP: &str = "cast_scale_bf16_f32";
    if s1.dtype() != DType::BF16 {
        return Err(Error::UnsupportedDTypeForOp(s1.dtype(), OP));
    }
    if out.dtype() != DType::F32 {
        return Err(Error::UnsupportedDTypeForOp(out.dtype(), OP));
    }
    if l1.shape().elem_count() != l_out.shape().elem_count() {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l1.shape().clone(),
            rhs: l_out.shape().clone(),
            op: OP,
        });
    }

    let device = s1.device().clone();
    let n = l1.shape().elem_count();
    let scale_f32 = scale as f32;

    // Contiguity checked before the `n == 0` fast path, matching this
    // file's own domain -- see `cuda_fwd_cast_scale_bf16_f32`'s identical
    // comment above.
    let (o1, o2) = l1
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;
    let (oo1, oo2) = l_out
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;

    if n == 0 {
        return Ok(());
    }

    super::check_elem_count_fits_u32(OP, n)?;

    let cfg = super::elemwise_launch_config(n as u32);

    let x = s1.as_cuda_slice::<bf16>()?.slice(o1..o2);
    let out_slice = out.as_cuda_slice::<f32>()?.slice(oo1..oo2);
    let func =
        device.get_or_load_custom_func("cast_scale_bf16_f32", MODULE_NAME, PTX_CAST_SCALE)?;
    let mut builder = func.builder();
    builder.arg(&scale_f32);
    builder.arg(&x);
    builder.arg(&out_slice);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
    Ok(())
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

    // Contiguity checked before the `n == 0` fast path -- see
    // `cuda_fwd_cast_scale_bf16_f32`'s identical comment above.
    let (o1_base, o2_base) = l1
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;
    let (o1_f32, o2_f32) = l2
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;

    if n == 0 {
        return Ok((super::alloc_empty(&device, DType::BF16, OP)?, shape));
    }

    super::check_elem_count_fits_u32(OP, n)?;

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

/// The `cast_add_bf16` counterpart of [`cuda_launch_cast_scale_bf16_f32_into`]
/// — same rationale (test-only preallocated-output scaffolding), same
/// kernel, writing into an existing output storage instead of allocating
/// a fresh one. `cast_add_bf16`'s output is 25 MB at production width
/// (vs `cast_scale_bf16_f32`'s 151 MB), so the allocator cost this
/// isolates is much smaller here — provided anyway so both ops in this
/// file report the SAME two numbers (kernel-only vs wrapper-including-
/// alloc), rather than leaving a reader to wonder why only one does.
pub(crate) fn cuda_launch_cast_add_bf16_into(
    s1: &CudaStorage,
    l1: &Layout,
    s2: &CudaStorage,
    l2: &Layout,
    out: &CudaStorage,
    l_out: &Layout,
) -> Result<()> {
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
    if out.dtype() != DType::BF16 {
        return Err(Error::UnsupportedDTypeForOp(out.dtype(), OP));
    }
    if l1.shape().elem_count() != l_out.shape().elem_count() {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l1.shape().clone(),
            rhs: l_out.shape().clone(),
            op: OP,
        });
    }

    let device = s1.device().clone();
    let n = l1.shape().elem_count();

    // Contiguity checked before the `n == 0` fast path -- see
    // `cuda_fwd_cast_scale_bf16_f32`'s identical comment above.
    let (o1_base, o2_base) = l1
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;
    let (o1_f32, o2_f32) = l2
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;
    let (oo1, oo2) = l_out
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;

    if n == 0 {
        return Ok(());
    }

    super::check_elem_count_fits_u32(OP, n)?;

    let cfg = super::elemwise_launch_config(n as u32);

    let base = s1.as_cuda_slice::<bf16>()?.slice(o1_base..o2_base);
    let f32val = s2.as_cuda_slice::<f32>()?.slice(o1_f32..o2_f32);
    let out_slice = out.as_cuda_slice::<bf16>()?.slice(oo1..oo2);
    let func = device.get_or_load_custom_func("cast_add_bf16", MODULE_NAME, PTX_CAST_SCALE)?;
    let mut builder = func.builder();
    builder.arg(&base);
    builder.arg(&f32val);
    builder.arg(&out_slice);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
    Ok(())
}
