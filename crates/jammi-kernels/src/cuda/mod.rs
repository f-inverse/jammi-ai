//! CUDA glue, compiled only when the `cuda` feature is active. `build.rs`
//! compiles `axpy.cu` to PTX at `OUT_DIR/axpy.ptx`; embedded here via
//! `include_str!` and loaded at runtime through candle's public
//! `CudaDevice::get_or_load_custom_func`.

use candle_core::cuda_backend::cudarc::driver::LaunchConfig;
use candle_core::{CudaDevice, CudaStorage, DType, Error, Result};
use half::{bf16, f16};

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
/// F16 monomorphic arm (campaign #443 W2c) — see `axpy_f16.cu`'s module doc
/// for why this is a SEPARATE `.cu` file rather than a widened `axpy.cu`.
pub(crate) const PTX_AXPY_F16: &str = include_str!(concat!(env!("OUT_DIR"), "/axpy_f16.ptx"));
pub(crate) const PTX_CAST_SCALE: &str = include_str!(concat!(env!("OUT_DIR"), "/cast_scale.ptx"));
/// F16 monomorphic arm (campaign #443 W2c) — see `cast_scale_f16.cu`'s
/// module doc. Backs the NEW `CastScaleF16F32`/`CastAddF16` types
/// (`ops::cast_scale`), not a widened match arm on the existing
/// BF16-monomorphic `CastScaleBf16F32`/`CastAddBf16`.
pub(crate) const PTX_CAST_SCALE_F16: &str =
    include_str!(concat!(env!("OUT_DIR"), "/cast_scale_f16.ptx"));
pub(crate) const PTX_DROPOUT: &str = include_str!(concat!(env!("OUT_DIR"), "/dropout.ptx"));
/// F16 monomorphic arm (campaign #443 W2c) — see `dropout_f16.cu`'s module
/// doc; carries its own Philox4x32-10 device functions (no shared `.cuh`,
/// per the W2b idiom this file continues).
pub(crate) const PTX_DROPOUT_F16: &str = include_str!(concat!(env!("OUT_DIR"), "/dropout_f16.ptx"));
pub(crate) const PTX_GEGLU: &str = include_str!(concat!(env!("OUT_DIR"), "/geglu.ptx"));
pub(crate) const PTX_LAYER_NORM: &str = include_str!(concat!(env!("OUT_DIR"), "/layer_norm.ptx"));
/// F16 monomorphic arm — see `layer_norm_f16.cu`'s module doc for why this
/// is a SEPARATE `.cu` file (and thus a separate PTX module) rather than
/// a widened `layer_norm.cu`.
pub(crate) const PTX_LAYER_NORM_F16: &str =
    include_str!(concat!(env!("OUT_DIR"), "/layer_norm_f16.ptx"));
pub(crate) const PTX_ROPE: &str = include_str!(concat!(env!("OUT_DIR"), "/rope.ptx"));
/// F16 monomorphic arm — see `rope_f16.cu`'s module doc.
pub(crate) const PTX_ROPE_F16: &str = include_str!(concat!(env!("OUT_DIR"), "/rope_f16.ptx"));
pub(crate) const PTX_ROPE_POSITIONS: &str =
    include_str!(concat!(env!("OUT_DIR"), "/rope_positions.ptx"));
/// F16 monomorphic arm (campaign #443 W2c) — see `rope_positions_f16.cu`'s
/// module doc; carries its own copy of `rope_common.cuh`'s `rope_rotate`
/// (no shared `.cuh`, per `rope_f16.cu`'s identical W2b precedent).
pub(crate) const PTX_ROPE_POSITIONS_F16: &str =
    include_str!(concat!(env!("OUT_DIR"), "/rope_positions_f16.ptx"));
pub(crate) const PTX_SCALED_CAST_ADD: &str =
    include_str!(concat!(env!("OUT_DIR"), "/scaled_cast_add.ptx"));
/// F16 monomorphic arm (campaign #443 W2c) — see `scaled_cast_add_f16.cu`'s
/// module doc. Three combinations (`F16`+`F32`, `F32`+`F16`, `F16`+`F16`),
/// mirroring the existing four-combo `F32`/`BF16` matrix's own split.
pub(crate) const PTX_SCALED_CAST_ADD_F16: &str =
    include_str!(concat!(env!("OUT_DIR"), "/scaled_cast_add_f16.ptx"));
pub(crate) const PTX_SOFTMAX: &str = include_str!(concat!(env!("OUT_DIR"), "/softmax.ptx"));
/// F16 monomorphic arm — see `softmax_f16.cu`'s module doc.
pub(crate) const PTX_SOFTMAX_F16: &str = include_str!(concat!(env!("OUT_DIR"), "/softmax_f16.ptx"));
pub(crate) const PTX_GEGLU_F16: &str = include_str!(concat!(env!("OUT_DIR"), "/geglu_f16.ptx"));

/// The element-count ceiling every dispatch below refuses above, and the
/// grid-stride launch geometry `geglu`'s `launch_config` builds — RE-
/// EXPORTED from `crate::ops::launch_domain`, never re-implemented here.
///
/// They are defined in `ops` because `mod cuda` is `#[cfg(feature =
/// "cuda")]` and they are pure arithmetic/domain facts (no device, no PTX,
/// no launch): defining them here would mean their unit tests only ever
/// compiled on a CUDA-feature build. See `ops::launch_domain`'s own module
/// doc for the full indexing contract (campaign #446, finding 4: 64-bit
/// in-kernel index arithmetic, 32-bit kernel scalar parameters, and why
/// each half needs the other).
pub(crate) use crate::ops::launch_domain::{
    check_elem_count_fits_u32, geglu_grid_blocks, GEGLU_BLOCK,
};

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
/// `cuda_fwd` (and backward helper) takes identically: F32/BF16/F16 are
/// this crate's production dtypes (F16 added in campaign #443 W2b for
/// `layer_norm`/`softmax`/`geglu`/`rope`, extended in W2c to
/// `rope_positions`/`axpy`/`dropout`/`scaled_cast_add` — every op with a
/// compiled F16 dispatch arm; a caller for any OTHER op never reaches this
/// arm with `DType::F16` because ITS OWN dtype match fails first — see
/// `tests::empty_f16_is_refused_for_an_op_with_no_f16_dispatch_arm` below
/// for `cast_scale_bf16_f32`/`cast_add_bf16`, the two ops this crate ships
/// WITHOUT an F16 dispatch arm as of W2c), anything else a typed refusal.
/// The caller validates cross-tensor dtype agreement FIRST (this only
/// handles the single already-agreed-on `dtype`, exactly like every
/// inlined match this replaces did).
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
        DType::F16 => {
            let out = unsafe { device.alloc::<f16>(0) }?;
            Ok(CudaStorage::wrap_cuda_slice(out, device.clone()))
        }
        dtype => Err(Error::UnsupportedDTypeForOp(dtype, op)),
    }
}

/// Wrap a freshly allocated, zero-FILLED device buffer of `dtype` and
/// length `len` as a `CudaStorage` — for the `rows == 0, <reduction axis>
/// != 0` degenerate fast path a summed-over-rows op (e.g.
/// `layer_norm::cuda_bwd_dgamma`) must take: unlike [`alloc_empty`]'s
/// zero-LENGTH buffer (correct only when the whole output is empty), a
/// sum over zero rows still produces a `[hidden]`-shaped, all-zero
/// output — the exact shape `layer_norm::ops`'s CPU reference
/// (`ln_bwd_dgamma_f32`'s `vec![0f32; hidden]`) returns for the same
/// input, so returning `alloc_empty`'s `[0]`-shaped buffer here instead
/// would be a cross-arm shape divergence (family D).
pub(crate) fn alloc_zeros(
    device: &CudaDevice,
    dtype: DType,
    len: usize,
    op: &'static str,
) -> Result<CudaStorage> {
    match dtype {
        DType::F32 => {
            let out = device.alloc_zeros::<f32>(len)?;
            Ok(CudaStorage::wrap_cuda_slice(out, device.clone()))
        }
        DType::BF16 => {
            let out = device.alloc_zeros::<bf16>(len)?;
            Ok(CudaStorage::wrap_cuda_slice(out, device.clone()))
        }
        DType::F16 => {
            let out = device.alloc_zeros::<f16>(len)?;
            Ok(CudaStorage::wrap_cuda_slice(out, device.clone()))
        }
        dtype => Err(Error::UnsupportedDTypeForOp(dtype, op)),
    }
}
