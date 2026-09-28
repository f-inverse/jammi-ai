//! Opening a CUDA device with the GEMM numerics every jammi computation
//! assumes.
//!
//! candle issues each `bf16`/`f16` matmul with an `f32` compute type. cuBLAS's
//! default math mode still lets a split-K kernel reduce its partial sums in
//! the OUTPUT type, rounding each partial to `bf16`/`f16` before the final
//! sum. Whether a given matmul takes that path is cuBLAS's per-card heuristic
//! keyed on the operand shape: on an RTX 4090 or RTX 6000 Ada a `[m, 32] x
//! [32, 64]` product splits K sixteen ways from `m = 17`, and not at all at
//! `m = 528`, while an L40S or L4 never splits either shape. A row's result
//! then depends on the batch it was computed in and on the card it ran on.
//! Holding every reduction to the compute type makes the split paths round
//! once, like the unsplit ones.

use candle_core::{Device, Result};

/// Opens CUDA device `ordinal` with cuBLAS's reductions held to the compute
/// type (see the module doc). Every CUDA device jammi computes on is opened
/// here, so no path runs on a handle left at cuBLAS's default math mode.
///
/// # Errors
/// When the device cannot be opened — including in a build without candle's
/// CUDA backend — or cuBLAS refuses the math mode.
pub fn open_cuda(ordinal: usize) -> Result<Device> {
    let device = Device::new_cuda(ordinal)?;
    #[cfg(feature = "cuda")]
    hold_reductions_to_compute_type(&device)?;
    Ok(device)
}

#[cfg(feature = "cuda")]
fn hold_reductions_to_compute_type(device: &Device) -> Result<()> {
    use candle_core::cuda_backend::cudarc::cublas::sys;
    use candle_core::cuda_backend::WrapErr;

    let blas = device.as_cuda_device()?.cublas_handle();
    // SAFETY: `blas` is candle's live handle for this device, kept alive by
    // the `Arc` for the duration of the call; setting its math mode is a
    // plain handle property write.
    unsafe {
        sys::cublasSetMathMode(
            *blas.handle(),
            sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
        )
    }
    .result()
    .w()
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "live-gpu-tests")]
    mod gpu {
        use crate::device::open_cuda;
        use candle_core::cuda_backend::cudarc::cublas::sys;

        #[test]
        fn open_cuda_leaves_the_cublas_handle_disallowing_reduced_precision_reduction() {
            let device = open_cuda(0).expect("this test needs CUDA device 0");
            let blas = device.as_cuda_device().unwrap().cublas_handle();
            let mut mode = sys::cublasMath_t::CUBLAS_DEFAULT_MATH;
            // SAFETY: reads the math mode of candle's live handle for this
            // device into a local.
            let status = unsafe { sys::cublasGetMathMode(*blas.handle(), &mut mode) };
            assert_eq!(status, sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
            assert_eq!(
                mode,
                sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION
            );
        }
    }
}
