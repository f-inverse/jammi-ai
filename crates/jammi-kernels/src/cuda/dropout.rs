use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::{CudaStorage, DType, Error, Layout, Result, Shape};
use half::bf16;

use super::PTX_DROPOUT;
use crate::ops::DropoutFused;

/// See `../axpy.rs`'s identical constant for the module-name rationale.
const MODULE_NAME: &str = "jammi_kernels_dropout";

/// Grid-stride block size — same choice as `geglu.cu`/`GEGLU_BLOCK`: this
/// kernel is purely elementwise (no per-row reduction, no shared memory),
/// so there is nothing block-size-sensitive about the choice beyond
/// "occupy the device reasonably".
const DROPOUT_BLOCK: u32 = 256;

/// A conservative 1-D grid cap; the kernel's own grid-stride loop (written
/// in `unsigned long long`, NOT `unsigned int`) covers any `n` beyond
/// `DROPOUT_BLOCK * DROPOUT_MAX_GRID` correctly — unlike `Axpy`'s
/// single-pass kernel, this op has no `u32::MAX` element-count ceiling at
/// all (see this module's `cuda_fwd`, which therefore does NOT check
/// `n > u32::MAX` the way every other op's CUDA glue in this crate does).
const DROPOUT_MAX_GRID: u32 = 65_535;

fn launch_config(n: u64) -> LaunchConfig {
    let blocks_needed = n.div_ceil(u64::from(DROPOUT_BLOCK));
    let blocks = u32::try_from(blocks_needed)
        .unwrap_or(u32::MAX)
        .clamp(1, DROPOUT_MAX_GRID);
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (DROPOUT_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub(crate) fn cuda_fwd(
    params: &DropoutFused,
    s1: &CudaStorage,
    l1: &Layout,
) -> Result<(CudaStorage, Shape)> {
    const OP: &str = "dropout_fused";
    let device = s1.device().clone();
    let shape = l1.shape().clone();
    let n = l1.shape().elem_count();

    // n == 0: `launch_config(0)` would still clamp to at least 1 block —
    // matching every other op's CUDA glue in this crate, avoid the launch
    // entirely and return an explicitly empty output instead.
    if n == 0 {
        return Ok((super::alloc_empty(&device, s1.dtype(), OP)?, shape));
    }

    let (o1, o2) = l1
        .contiguous_offsets()
        .ok_or(Error::RequiresContiguous { op: OP })?;
    let cfg = launch_config(n as u64);
    let (seed, layer_id, forward_idx, threshold, scale) = params.cuda_launch_args();
    let n_u64 = n as u64;

    match s1.dtype() {
        DType::F32 => {
            let x = s1.as_cuda_slice::<f32>()?.slice(o1..o2);
            let func =
                device.get_or_load_custom_func("dropout_fwd_f32", MODULE_NAME, PTX_DROPOUT)?;
            let out = unsafe { device.alloc::<f32>(n) }?;
            let mut builder = func.builder();
            builder.arg(&seed);
            builder.arg(&layer_id);
            builder.arg(&forward_idx);
            builder.arg(&threshold);
            builder.arg(&scale);
            builder.arg(&x);
            builder.arg(&out);
            builder.arg(&n_u64);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        DType::BF16 => {
            let x = s1.as_cuda_slice::<bf16>()?.slice(o1..o2);
            let func =
                device.get_or_load_custom_func("dropout_fwd_bf16", MODULE_NAME, PTX_DROPOUT)?;
            let out = unsafe { device.alloc::<bf16>(n) }?;
            let mut builder = func.builder();
            builder.arg(&seed);
            builder.arg(&layer_id);
            builder.arg(&forward_idx);
            builder.arg(&threshold);
            builder.arg(&scale);
            builder.arg(&x);
            builder.arg(&out);
            builder.arg(&n_u64);
            unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
            Ok((CudaStorage::wrap_cuda_slice(out, device), shape))
        }
        dtype => Err(Error::UnsupportedDTypeForOp(dtype, OP)),
    }
}

/// TEST-SUPPORT ONLY: launches `dropout.cu`'s `philox_kat` — a single
/// thread, single block kernel that runs `philox4x32_10(counter, key)`
/// device-side and writes its 4 output words. See
/// `crate::ops::PhiloxKatProbe`'s doc: this is the CUDA half of the
/// shared Random123 known-answer-test proof, reached through the SAME
/// `apply1` dispatch path as every real op in this crate rather than a
/// bespoke raw-buffer-download API.
pub(crate) fn cuda_philox_kat(
    counter: [u32; 4],
    key: [u32; 2],
    device: &candle_core::CudaDevice,
) -> Result<(CudaStorage, Shape)> {
    let device = device.clone();
    let func = device.get_or_load_custom_func("philox_kat", MODULE_NAME, PTX_DROPOUT)?;
    let out = unsafe { device.alloc::<u32>(4) }?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(&counter[0]);
    builder.arg(&counter[1]);
    builder.arg(&counter[2]);
    builder.arg(&counter[3]);
    builder.arg(&key[0]);
    builder.arg(&key[1]);
    builder.arg(&out);
    unsafe { builder.launch(cfg) }.map_err(|e| Error::Cuda(Box::new(e)))?;
    Ok((CudaStorage::wrap_cuda_slice(out, device), Shape::from((4,))))
}
