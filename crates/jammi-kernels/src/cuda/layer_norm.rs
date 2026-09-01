use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::{CudaStorage, DType, Error, Layout, Result, Shape};
use half::bf16;

use super::PTX_LAYER_NORM;
use crate::ops::layer_norm::hidden_of;
use crate::ops::MAX_HIDDEN;

/// See `../../cuda/axpy.rs`'s identical constant for the module-name
/// rationale — arbitrary but stable and unique to this op's PTX module.
const MODULE_NAME: &str = "jammi_kernels_layer_norm";

/// One CUDA thread block per row; must match `LN_BLOCK` in `layer_norm.cu`.
const LN_BLOCK: u32 = 256;

/// `hidden > MAX_HIDDEN`: refused above a conservative, VALIDATED ceiling
/// — NOT a hardware limit (see `ops::MAX_HIDDEN`'s doc: the block-per-row
/// reduction's shared memory is `O(blockDim.x)`, not `O(hidden)`, so
/// nothing about the launch itself breaks above this value; the refusal
/// is about validated numerics/performance coverage, not a real ceiling
/// this kernel design imposes). `n > u32::MAX`: the launch grid and the
/// kernel's own indices are 32-bit, exactly the guard `axpy.rs` documents
/// for the same reason.
fn check_cuda_domain(op: &'static str, n: usize, hidden: usize) -> Result<()> {
    if hidden > MAX_HIDDEN {
        return Err(Error::Msg(format!(
            "{op}: hidden={hidden} exceeds the CUDA kernel's MAX_HIDDEN={MAX_HIDDEN} \
             (a conservative validated ceiling, not a hardware limit — see \
             ops::MAX_HIDDEN's doc); the CPU arm has no such ceiling"
        )));
    }
    super::check_elem_count_fits_u32(op, n)
}

pub(crate) fn cuda_fwd(
    eps: f64,
    s1: &CudaStorage,
    l1: &Layout,
    s2: &CudaStorage,
    l2: &Layout,
) -> Result<(CudaStorage, Shape)> {
    const OP: &str = "layer_norm_fused";
    let hidden = hidden_of(l1, l2, OP)?;
    let shape = l1.shape().clone();
    let device = s1.device().clone();
    let n = l1.shape().elem_count();

    // `hidden == 0` is checked FIRST, on its own — matching `cpu_fwd`'s
    // domain exactly: `ops::layer_norm::LayerNormFused::cpu_fwd` exempts
    // contiguity ONLY when `hidden == 0` (its own early `empty_like`
    // return, BEFORE `contiguous_offsets()`), never for a broader `n == 0`.
    if hidden == 0 {
        if s1.dtype() != s2.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: OP,
            });
        }
        return Ok((super::alloc_empty(&device, s1.dtype(), OP)?, shape));
    }

    // Contiguity is checked NEXT, before the (now `hidden != 0`) `n == 0`
    // fast path below — a `rows == 0` (`n == 0` with `hidden != 0`)
    // empty-but-non-contiguous layout still falls through `cpu_fwd`'s OWN
    // `contiguous_offsets()` call (only `hidden == 0`, handled above,
    // skips it there), so this arm must refuse the same layout rather than
    // silently admitting it through a combined `hidden == 0 || n == 0`
    // fast path — the exact class of divergence this fix closes.
    let (o1, o2) = l1
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;
    let (g1, g2) = l2
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;

    if n == 0 {
        if s1.dtype() != s2.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: OP,
            });
        }
        return Ok((super::alloc_empty(&device, s1.dtype(), OP)?, shape));
    }
    check_cuda_domain(OP, n, hidden)?;
    let rows = n / hidden;

    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (LN_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let hidden_u32 = hidden as u32;
    let eps_f32 = eps as f32;

    match (s1.dtype(), s2.dtype()) {
        (DType::F32, DType::F32) => {
            let x = s1.as_cuda_slice::<f32>()?.slice(o1..o2);
            let g = s2.as_cuda_slice::<f32>()?.slice(g1..g2);
            let func = device.get_or_load_custom_func(
                "layer_norm_fwd_f32",
                MODULE_NAME,
                PTX_LAYER_NORM,
            )?;
            let out = unsafe { device.alloc::<f32>(n) }?;
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&g);
            builder.arg(&out);
            builder.arg(&hidden_u32);
            builder.arg(&eps_f32);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        (DType::BF16, DType::BF16) => {
            let x = s1.as_cuda_slice::<bf16>()?.slice(o1..o2);
            let g = s2.as_cuda_slice::<bf16>()?.slice(g1..g2);
            let func = device.get_or_load_custom_func(
                "layer_norm_fwd_bf16",
                MODULE_NAME,
                PTX_LAYER_NORM,
            )?;
            let out = unsafe { device.alloc::<bf16>(n) }?;
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&g);
            builder.arg(&out);
            builder.arg(&hidden_u32);
            builder.arg(&eps_f32);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        (lhs, rhs) if lhs != rhs => Err(Error::DTypeMismatchBinaryOp { lhs, rhs, op: OP }),
        (dtype, _) => Err(Error::UnsupportedDTypeForOp(dtype, OP)),
    }
}

pub(crate) fn cuda_bwd_dx(
    eps: f64,
    s1: &CudaStorage,
    l1: &Layout,
    s2: &CudaStorage,
    l2: &Layout,
    s3: &CudaStorage,
    l3: &Layout,
) -> Result<(CudaStorage, Shape)> {
    const OP: &str = "layer_norm_fused_bwd_dx";
    let hidden = hidden_of(l1, l2, OP)?;
    if l3.dims() != l1.dims() {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l1.shape().clone(),
            rhs: l3.shape().clone(),
            op: OP,
        });
    }
    if s1.dtype() != s3.dtype() {
        return Err(Error::DTypeMismatchBinaryOp {
            lhs: s1.dtype(),
            rhs: s3.dtype(),
            op: OP,
        });
    }
    let shape = l1.shape().clone();
    let device = s1.device().clone();
    let n = l1.shape().elem_count();

    // See `cuda_fwd`'s identical comment above: `hidden == 0` is checked
    // on its own, matching `ops::layer_norm::LayerNormBwdDx::cpu_fwd`'s
    // domain (its own early `empty_like` return is gated on `hidden == 0`
    // alone, before `contiguous_offsets()`).
    if hidden == 0 {
        if s1.dtype() != s2.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: OP,
            });
        }
        return Ok((super::alloc_empty(&device, s1.dtype(), OP)?, shape));
    }

    // Contiguity checked before the (now `hidden != 0`) `n == 0` fast path
    // — see `cuda_fwd`'s identical comment above.
    let (o1, o2) = l1
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;
    let (g1, g2) = l2
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;
    let (d1, d2) = l3
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;

    if n == 0 {
        if s1.dtype() != s2.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: OP,
            });
        }
        return Ok((super::alloc_empty(&device, s1.dtype(), OP)?, shape));
    }
    check_cuda_domain(OP, n, hidden)?;
    let rows = n / hidden;

    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (LN_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let hidden_u32 = hidden as u32;
    let eps_f32 = eps as f32;

    match (s1.dtype(), s2.dtype()) {
        (DType::F32, DType::F32) => {
            let x = s1.as_cuda_slice::<f32>()?.slice(o1..o2);
            let g = s2.as_cuda_slice::<f32>()?.slice(g1..g2);
            let dy = s3.as_cuda_slice::<f32>()?.slice(d1..d2);
            let func = device.get_or_load_custom_func(
                "layer_norm_bwd_dx_f32",
                MODULE_NAME,
                PTX_LAYER_NORM,
            )?;
            let out = unsafe { device.alloc::<f32>(n) }?;
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&g);
            builder.arg(&dy);
            builder.arg(&out);
            builder.arg(&hidden_u32);
            builder.arg(&eps_f32);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        (DType::BF16, DType::BF16) => {
            let x = s1.as_cuda_slice::<bf16>()?.slice(o1..o2);
            let g = s2.as_cuda_slice::<bf16>()?.slice(g1..g2);
            let dy = s3.as_cuda_slice::<bf16>()?.slice(d1..d2);
            let func = device.get_or_load_custom_func(
                "layer_norm_bwd_dx_bf16",
                MODULE_NAME,
                PTX_LAYER_NORM,
            )?;
            let out = unsafe { device.alloc::<bf16>(n) }?;
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&g);
            builder.arg(&dy);
            builder.arg(&out);
            builder.arg(&hidden_u32);
            builder.arg(&eps_f32);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        (lhs, rhs) if lhs != rhs => Err(Error::DTypeMismatchBinaryOp { lhs, rhs, op: OP }),
        (dtype, _) => Err(Error::UnsupportedDTypeForOp(dtype, OP)),
    }
}

pub(crate) fn cuda_bwd_dgamma(
    eps: f64,
    s1: &CudaStorage,
    l1: &Layout,
    s2: &CudaStorage,
    l2: &Layout,
) -> Result<(CudaStorage, Shape)> {
    const OP: &str = "layer_norm_fused_bwd_dgamma";
    if l1.dims() != l2.dims() {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l1.shape().clone(),
            rhs: l2.shape().clone(),
            op: OP,
        });
    }
    let dims = l1.dims();
    let hidden = *dims.last().ok_or_else(|| {
        Error::Msg(format!(
            "{OP}: input must have rank >= 1 to define a last (hidden) dimension"
        ))
    })?;
    let device = s1.device().clone();
    let n = l1.shape().elem_count();

    // See `cuda_fwd`'s identical comment above: `hidden == 0` is checked
    // on its own, matching `ops::layer_norm`'s internal `LayerNormBwdDgamma
    // ::cpu_fwd` domain (its own early-return is gated on `hidden == 0`
    // alone, before `contiguous_offsets()`). This arm's `Shape::from(
    // 0usize)` is correct ONLY here, where `hidden` (the output's own
    // length) is itself `0` — see the `n == 0` fast path below for the
    // `rows == 0, hidden != 0` case, which must NOT reuse this shape.
    if hidden == 0 {
        if s1.dtype() != s2.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: OP,
            });
        }
        return Ok((
            super::alloc_empty(&device, s1.dtype(), OP)?,
            Shape::from(0usize),
        ));
    }

    // Contiguity checked before the (now `hidden != 0`) `n == 0` fast path
    // — see `cuda_fwd`'s identical comment above.
    let (o1, o2) = l1
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;
    let (d1, d2) = l2
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;

    // `rows == 0` here (`n == 0` with `hidden != 0`, since the `hidden ==
    // 0` case already returned above): `dgamma` is a sum over rows, so
    // zero rows still sum to a `[hidden]`-shaped, all-zero output — NOT
    // `alloc_empty`'s `[0]`-shaped buffer, which would silently produce a
    // wrong-shaped `dgamma` on CUDA relative to `cpu_fwd`'s own `rows ==
    // 0` path (`ln_bwd_dgamma_f32`'s `vec![0f32; hidden]`, `Shape::from(
    // hidden)`). See `super::alloc_zeros`'s doc for why this needs its own
    // helper rather than reusing `alloc_empty`.
    if n == 0 {
        if s1.dtype() != s2.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: OP,
            });
        }
        return Ok((
            super::alloc_zeros(&device, s1.dtype(), hidden, OP)?,
            Shape::from(hidden),
        ));
    }
    check_cuda_domain(OP, n, hidden)?;
    let rows = n / hidden;

    // Pass 1 (row_cfg): one block per row, caching that row's mean/invvar
    // once. Pass 2 (col_cfg): column-tiled, grid-stride over `hidden`.
    // Two launches, O(rows*hidden) total work — NOT the O(rows*hidden^2)
    // shape a single recompute-per-column-per-row kernel would have (see
    // layer_norm.cu's module doc).
    let row_cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (LN_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let col_block = LN_BLOCK;
    let col_grid = (hidden as u32).div_ceil(col_block).max(1);
    let col_cfg = LaunchConfig {
        grid_dim: (col_grid, 1, 1),
        block_dim: (col_block, 1, 1),
        shared_mem_bytes: 0,
    };
    let rows_u32 = rows as u32;
    let hidden_u32 = hidden as u32;
    let eps_f32 = eps as f32;

    match (s1.dtype(), s2.dtype()) {
        (DType::F32, DType::F32) => {
            let x = s1.as_cuda_slice::<f32>()?.slice(o1..o2);
            let dy = s2.as_cuda_slice::<f32>()?.slice(d1..d2);

            let mean = unsafe { device.alloc::<f32>(rows) }?;
            let invvar = unsafe { device.alloc::<f32>(rows) }?;
            let stats_func = device.get_or_load_custom_func(
                "layer_norm_row_stats_f32",
                MODULE_NAME,
                PTX_LAYER_NORM,
            )?;
            let mut stats_builder = stats_func.builder();
            stats_builder.arg(&x);
            stats_builder.arg(&mean);
            stats_builder.arg(&invvar);
            stats_builder.arg(&hidden_u32);
            stats_builder.arg(&eps_f32);
            unsafe { stats_builder.launch(row_cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;

            let dgamma_func = device.get_or_load_custom_func(
                "layer_norm_bwd_dgamma_f32",
                MODULE_NAME,
                PTX_LAYER_NORM,
            )?;
            let out = unsafe { device.alloc::<f32>(hidden) }?;
            let mut dg_builder = dgamma_func.builder();
            dg_builder.arg(&x);
            dg_builder.arg(&dy);
            dg_builder.arg(&mean);
            dg_builder.arg(&invvar);
            dg_builder.arg(&out);
            dg_builder.arg(&rows_u32);
            dg_builder.arg(&hidden_u32);
            unsafe { dg_builder.launch(col_cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((
                CudaStorage::wrap_cuda_slice(out, device),
                Shape::from(hidden),
            ))
        }
        (DType::BF16, DType::BF16) => {
            let x = s1.as_cuda_slice::<bf16>()?.slice(o1..o2);
            let dy = s2.as_cuda_slice::<bf16>()?.slice(d1..d2);

            let mean = unsafe { device.alloc::<f32>(rows) }?;
            let invvar = unsafe { device.alloc::<f32>(rows) }?;
            let stats_func = device.get_or_load_custom_func(
                "layer_norm_row_stats_bf16",
                MODULE_NAME,
                PTX_LAYER_NORM,
            )?;
            let mut stats_builder = stats_func.builder();
            stats_builder.arg(&x);
            stats_builder.arg(&mean);
            stats_builder.arg(&invvar);
            stats_builder.arg(&hidden_u32);
            stats_builder.arg(&eps_f32);
            unsafe { stats_builder.launch(row_cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;

            let dgamma_func = device.get_or_load_custom_func(
                "layer_norm_bwd_dgamma_bf16",
                MODULE_NAME,
                PTX_LAYER_NORM,
            )?;
            // Accumulate in F32 (column-tiled, no atomics — see
            // layer_norm.cu's module doc), then round to bf16 once via a
            // tiny elementwise cast kernel, matching the crate's
            // f32-accumulate-round-once convention and keeping this op's
            // OUTPUT dtype identical to the CPU arm's (bf16 in, bf16 out)
            // regardless of device.
            let f32_scratch = unsafe { device.alloc::<f32>(hidden) }?;
            let mut dg_builder = dgamma_func.builder();
            dg_builder.arg(&x);
            dg_builder.arg(&dy);
            dg_builder.arg(&mean);
            dg_builder.arg(&invvar);
            dg_builder.arg(&f32_scratch);
            dg_builder.arg(&rows_u32);
            dg_builder.arg(&hidden_u32);
            unsafe { dg_builder.launch(col_cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;

            let cast_func = device.get_or_load_custom_func(
                "layer_norm_cast_f32_to_bf16",
                MODULE_NAME,
                PTX_LAYER_NORM,
            )?;
            let out = unsafe { device.alloc::<bf16>(hidden) }?;
            let cast_cfg = LaunchConfig::for_num_elems(hidden as u32);
            let mut cast_builder = cast_func.builder();
            cast_builder.arg(&f32_scratch);
            cast_builder.arg(&out);
            cast_builder.arg(&hidden_u32);
            unsafe { cast_builder.launch(cast_cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((
                CudaStorage::wrap_cuda_slice(out, device),
                Shape::from(hidden),
            ))
        }
        (lhs, rhs) if lhs != rhs => Err(Error::DTypeMismatchBinaryOp { lhs, rhs, op: OP }),
        (dtype, _) => Err(Error::UnsupportedDTypeForOp(dtype, OP)),
    }
}
