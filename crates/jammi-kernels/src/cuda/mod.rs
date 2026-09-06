//! CUDA glue, compiled only when the `cuda` feature is active. `build.rs`
//! compiles every `src/cuda/*.cu` to PTX (e.g. `layer_norm.cu` to
//! `OUT_DIR/layer_norm.ptx`); embedded here via `include_str!` and loaded
//! at runtime through candle's public
//! `CudaDevice::get_or_load_custom_func`.
//!
//! ## Two conventions every op's glue in here shares
//!
//! **PTX module names.** Each op loads its PTX under a `MODULE_NAME`
//! constant (`CudaDevice::get_or_load_custom_func`'s module cache key).
//! The exact string is arbitrary; what matters is that it is stable and
//! unique to that op, so a second op's module never collides with it. An
//! op with a separate monomorphic F16 translation unit carries a second,
//! equally distinct `MODULE_NAME_F16`.
//!
//! **Contiguity is checked FIRST, and `is_contiguous()` alone is NOT
//! sufficient.** Every `cuda_fwd` here resolves its buffers with
//! `Layout::contiguous_offsets()` before its `n == 0` fast path, so a
//! non-contiguous VIEW is refused with `Error::RequiresContiguous`
//! whether or not it happens to be empty (the CPU arms never require
//! contiguity — they walk `StridedOffsets` — so this is the CUDA arm's
//! OWN self-consistency, not a match to a `cpu_fwd` requirement).
//! `is_contiguous()` does not imply `start_offset == 0` (candle's own doc
//! on `Layout::is_contiguous`: "does not imply that the start offset is 0
//! or that there are no extra elements at the end of the storage") — a
//! `narrow`'d-but-still-contiguous view can have a nonzero offset into a
//! LARGER base buffer. `as_cuda_slice::<T>()` returns the WHOLE base
//! `CudaSlice`, so reading it from element 0 would read the base buffer's
//! first `n` elements instead of this tensor's actual `[start_offset,
//! start_offset + n)` range — a confident wrong answer with no error.
//! `contiguous_offsets()` gives the real `[o1, o2)` range, and slicing to
//! it is candle's own idiom for this exact situation
//! (`cuda_backend/mod.rs`'s `IndexAdd` CUDA impl does the same
//! `contiguous_offsets()` -> `slice(o1..o2)` on both of its tensor args).
//! `tests/cuda_parity.rs`'s nonzero-start-offset parity legs pin it.

use candle_core::cuda_backend::cudarc::driver::LaunchConfig;
use candle_core::{CudaDevice, CudaStorage, DType, Error, Result};
use half::{bf16, f16};

pub(crate) mod adamw_step;
pub(crate) mod attention_block;
pub(crate) mod cast_scale;
pub(crate) mod dropout;
pub(crate) mod geglu;
pub(crate) mod gelu_erf;
pub(crate) mod layer_norm;
pub(crate) mod low_rank_residual_linear;
pub(crate) mod rope;
pub(crate) mod rope_positions;
pub(crate) mod scaled_cast_add;
pub(crate) mod softmax;

pub(crate) const PTX_ADAMW_STEP: &str = include_str!(concat!(env!("OUT_DIR"), "/adamw_step.ptx"));
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
pub(crate) const PTX_GELU_ERF: &str = include_str!(concat!(env!("OUT_DIR"), "/gelu_erf.ptx"));
/// F16 monomorphic arm — see `gelu_erf_f16.cu`'s module doc.
pub(crate) const PTX_GELU_ERF_F16: &str =
    include_str!(concat!(env!("OUT_DIR"), "/gelu_erf_f16.ptx"));
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
/// single-pass elementwise kernels (`ScaledCastAdd`, `RopeFused`)
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
/// `rope_positions`/`dropout`/`scaled_cast_add` — every op with a
/// compiled F16 dispatch arm), anything else a typed refusal.
///
/// The two ops this crate ships WITHOUT an F16 dispatch arm
/// (`cast_scale_bf16_f32`, `cast_add_bf16`) cannot reach this function's
/// `DType::F16` arm at all, and the reason is STRUCTURAL rather than
/// test-pinned (a prior revision of this comment cited a
/// `tests::empty_f16_is_refused_for_an_op_with_no_f16_dispatch_arm` that
/// has never existed in this tree — campaign #446, finding 14): their
/// CUDA glue (`cast_scale::cuda_fwd_cast_scale_bf16_f32`,
/// `cuda_fwd_cast_add_bf16`) refuses any non-BF16 input with
/// `UnsupportedDTypeForOp` at its FIRST statement, before the `n == 0`
/// fast path, and each then passes a LITERAL `DType::F32`/`DType::BF16`
/// here — never `s1.dtype()`. That refusal IS test-pinned, on the CUDA
/// arm and at BOTH `n == 0` and `n > 0`, by
/// `tests/cuda_parity.rs`'s
/// `cast_scale_bf16_f32_and_cast_add_bf16_refuse_f16_both_empty_and_nonempty`
/// — the test the missing citation was presumably reaching for.
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
