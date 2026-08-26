//! CUDA glue, compiled only when the `cuda` feature is active. `build.rs`
//! compiles `axpy.cu` to PTX at `OUT_DIR/axpy.ptx`; embedded here via
//! `include_str!` and loaded at runtime through candle's public
//! `CudaDevice::get_or_load_custom_func`.

use candle_core::cuda_backend::cudarc::driver::LaunchConfig;
use candle_core::{CudaDevice, CudaStorage, DType, Error, Result};
use half::bf16;

pub(crate) mod adamw_step;
pub(crate) mod attention_block;
pub(crate) mod axpy;
pub(crate) mod cast_scale;
pub(crate) mod dropout;
pub(crate) mod geglu;
pub(crate) mod layer_norm;
pub(crate) mod low_rank_residual_linear;
pub(crate) mod rope;
pub(crate) mod rope_positions;
pub(crate) mod scaled_cast_add;
pub(crate) mod softmax;

pub(crate) const PTX_ADAMW_STEP: &str = include_str!(concat!(env!("OUT_DIR"), "/adamw_step.ptx"));
pub(crate) const PTX_AXPY: &str = include_str!(concat!(env!("OUT_DIR"), "/axpy.ptx"));
pub(crate) const PTX_CAST_SCALE: &str = include_str!(concat!(env!("OUT_DIR"), "/cast_scale.ptx"));
pub(crate) const PTX_DROPOUT: &str = include_str!(concat!(env!("OUT_DIR"), "/dropout.ptx"));
pub(crate) const PTX_GEGLU: &str = include_str!(concat!(env!("OUT_DIR"), "/geglu.ptx"));
pub(crate) const PTX_LAYER_NORM: &str = include_str!(concat!(env!("OUT_DIR"), "/layer_norm.ptx"));
pub(crate) const PTX_ROPE: &str = include_str!(concat!(env!("OUT_DIR"), "/rope.ptx"));
pub(crate) const PTX_ROPE_POSITIONS: &str =
    include_str!(concat!(env!("OUT_DIR"), "/rope_positions.ptx"));
pub(crate) const PTX_SCALED_CAST_ADD: &str =
    include_str!(concat!(env!("OUT_DIR"), "/scaled_cast_add.ptx"));
pub(crate) const PTX_SOFTMAX: &str = include_str!(concat!(env!("OUT_DIR"), "/softmax.ptx"));

/// `n > u32::MAX`: every op's CUDA launch grid and its kernel's own
/// indices are 32-bit, so an element count above `u32::MAX` would silently
/// truncate via `as u32` (under-launching, leaving the allocation's tail
/// uninitialized) rather than fail loudly (family D / K2) — refused here
/// instead. Shared by every op's own domain check: `geglu::check_n` is a
/// direct pass-through; `layer_norm`/`rope`'s combined `check_cuda_domain`
/// and `softmax`'s combined `check_last_and_n` call this for the
/// `u32::MAX` half of their check, alongside their own op-specific
/// ceiling (`MAX_HIDDEN`/`MAX_HEAD_DIM`/`MAX_LAST_DIM`).
pub(crate) fn check_elem_count_fits_u32(op: &'static str, n: usize) -> Result<()> {
    if n > u32::MAX as usize {
        return Err(Error::Msg(format!(
            "{op}: {n} elements exceeds u32::MAX; the CUDA launch grid and \
             the kernel's indices are both 32-bit"
        )));
    }
    Ok(())
}

/// The grid-stride, one-thread-per-element launch config the crate's
/// single-pass elementwise kernels (`Axpy`, `ScaledCastAdd`, `RopeFused`)
/// all use — the kernel's own `if (i < n)` bounds check covers `n` in one
/// launch, so `LaunchConfig::for_num_elems` alone is sufficient.
/// `GegluFused`'s own `launch_config` (`cuda/geglu.rs`) is DELIBERATELY
/// separate: a block-capped, grid-strided config for a kernel whose loop
/// body walks PAST the launch grid — a different shape, not unified here.
pub(crate) fn elemwise_launch_config(n: u32) -> LaunchConfig {
    LaunchConfig::for_num_elems(n)
}

/// Wrap a freshly allocated, zero-length device buffer of `dtype` as a
/// `CudaStorage` — the degenerate "0 output elements" fast path every op's
/// `cuda_fwd` (and backward helper) takes identically: F32/BF16 are this
/// crate's two production dtypes, anything else a typed refusal. The
/// caller validates cross-tensor dtype agreement FIRST (this only handles
/// the single already-agreed-on `dtype`, exactly like every inlined match
/// this replaces did).
pub(crate) fn alloc_empty(
    device: &CudaDevice,
    dtype: DType,
    op: &'static str,
) -> Result<CudaStorage> {
    match dtype {
        DType::F32 => {
            let out = unsafe { device.alloc::<f32>(0) }?;
            Ok(CudaStorage::wrap_cuda_slice(out, device.clone()))
        }
        DType::BF16 => {
            let out = unsafe { device.alloc::<bf16>(0) }?;
            Ok(CudaStorage::wrap_cuda_slice(out, device.clone()))
        }
        dtype => Err(Error::UnsupportedDTypeForOp(dtype, op)),
    }
}
