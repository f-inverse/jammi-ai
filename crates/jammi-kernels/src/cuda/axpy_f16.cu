// axpy_f16.cu — y' = alpha * x + y, elementwise, F16 monomorphic arm.
// Compiled to PTX only when the `cuda` feature is active (see ../../build.rs);
// the pinned build flags (sm_80 baseline, no -use_fast_math) live there, not
// here.
//
// DELIBERATE DUPLICATION (campaign #443 W2b/W2c contract) — see
// `layer_norm_f16.cu`'s identical note: this is a SEPARATE translation unit
// from `axpy.cu` (a separate PTX module, `PTX_AXPY_F16` in `../mod.rs`) so
// that file's own git history stays byte-untouched — provable by `git diff`.
// No shared `.cuh`.
//
// Domain and accumulation regime are IDENTICAL to `axpy.cu`'s module doc,
// substituting `__half` for `__nv_bfloat16`: f32-accumulate, ONE rounding to
// f16 on the way out — matching `ops/axpy.rs`'s `axpy_f16` CPU reference arm
// exactly (per the per-op f16 reference-regime table,
// `docs/maintainer/cuda-kernel-guide.md` §3.10).
#include <cuda_fp16.h>
#include <cstddef>

extern "C" __global__ void axpy_f16(
    const float alpha,
    const __half *x,
    const __half *y,
    __half *out,
    const size_t n
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float xv = __half2float(x[i]);
        float yv = __half2float(y[i]);
        out[i] = __float2half(alpha * xv + yv);
    }
}
