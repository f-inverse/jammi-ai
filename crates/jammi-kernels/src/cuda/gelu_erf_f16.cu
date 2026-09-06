// gelu_erf_f16.cu -- fused erf-based GELU forward + backward, F16
// monomorphic arm. Compiled to PTX only when the `cuda` feature is active
// (see ../../build.rs); the pinned build flags (sm_80 baseline, no
// -use_fast_math) live there, not here.
//
// DELIBERATE DUPLICATION (campaign #443 W2b convention -- see
// `geglu_f16.cu`'s/`layer_norm_f16.cu`'s identical note): a SEPARATE
// translation unit from `gelu_erf.cu`, with its own `#include
// <cuda_fp16.h>` and its own `gelu_erf_pdf` helper -- NOT a shared `.cuh`.
//
// Domain, the single-pass elementwise indexing, and the forward/backward
// formulas are IDENTICAL to `gelu_erf.cu`'s module doc, substituting
// `__half` for `__nv_bfloat16` (`__hmul`/`__float2half`/`__half2float` in
// place of the bf16 intrinsics) -- the exact same two-rounding-point
// forward regime and one-rounding backward regime, matching candle-kernels'
// `ugelu_erf_f16` (`unary.cu:174`; `cuda_utils.cuh:174`) bit-for-bit.
#include <cuda_fp16.h>
#include <cstddef>

#define GELU_ERF_KBETA 0.3989422804014327f

__device__ __forceinline__ float gelu_erf_pdf(float x) {
    return GELU_ERF_KBETA * expf(-0.5f * x * x);
}

// ---------------------------------------------------------------------
// Forward.
// ---------------------------------------------------------------------

extern "C" __global__ void gelu_erf_fwd_f16(
    const __half* x, __half* out, const size_t n
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        __half xh = x[i];
        float cdf_f32 = normcdff(__half2float(xh));
        __half cdf16 = __float2half(cdf_f32); // ROUND 1
        out[i] = __hmul(xh, cdf16); // ROUND 2
    }
}

// ---------------------------------------------------------------------
// Backward: dx = dy * (Phi(x) + x*phi(x)).
// ---------------------------------------------------------------------

extern "C" __global__ void gelu_erf_bwd_dx_f16(
    const __half* x, const __half* dy, __half* dx, const size_t n
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float xv = __half2float(x[i]);
        float dyv = __half2float(dy[i]);
        float cdf = normcdff(xv);
        float pdf = gelu_erf_pdf(xv);
        dx[i] = __float2half(dyv * (cdf + xv * pdf));
    }
}
