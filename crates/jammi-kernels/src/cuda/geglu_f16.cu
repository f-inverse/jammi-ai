// geglu_f16.cu — fused GeGLU (gated GELU) forward + backward, F16
// monomorphic arm. Compiled to PTX only when the `cuda` feature is active
// (see ../../build.rs); the pinned build flags (sm_80 baseline, no
// -use_fast_math) live there, not here.
//
// DELIBERATE DUPLICATION (campaign #443 W2b contract) — see
// `layer_norm_f16.cu`'s identical note: this is a SEPARATE translation
// unit from `geglu.cu`, with its own copy of the `gelu_erf_cdf`/
// `gelu_erf_pdf` helpers and its own `#include <cuda_fp16.h>` — NOT a
// shared `.cuh`. `geglu.cu` is byte-untouched.
//
// Domain and the purely-elementwise indexing are IDENTICAL to `geglu.cu`'s
// module doc. Per the per-op f16 reference-regime table
// (`docs/maintainer/cuda-kernel-guide.md` §3.10), this op is DTYPE-NATIVE,
// TWO rounding points in forward (round the activation to f16 immediately
// — ROUND 1 — matching the upstream two-op reference's own
// materialize-then-multiply ordering, then multiply in f32 and round ONCE
// more on the way out — ROUND 2), and TWO independent rounding points in
// backward (`d_gate`/`d_up`, each f32-accumulated and rounded to f16
// exactly once) — the exact same regime as the existing BF16 arm,
// substituting the narrower 16-bit type.
#include <cuda_fp16.h>
#include <cstddef>

#define GEGLU_KALPHA 0.70710678118654752440f
#define GEGLU_KBETA 0.3989422804014327f

__device__ __forceinline__ float gelu_erf_cdf(float x) {
    return (erff(x * GEGLU_KALPHA) + 1.0f) * 0.5f;
}

__device__ __forceinline__ float gelu_erf_pdf(float x) {
    return GEGLU_KBETA * expf(-0.5f * x * x);
}

// ---------------------------------------------------------------------
// Forward: out = gelu_erf(gate) * up.
// ---------------------------------------------------------------------

extern "C" __global__ void geglu_fwd_f16(
    const __half* wi_out, __half* out,
    const unsigned int intermediate, const unsigned int n_out
) {
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_out;
         idx += blockDim.x * gridDim.x) {
        unsigned int row = idx / intermediate;
        unsigned int col = idx % intermediate;
        size_t base = (size_t)row * 2 * (size_t)intermediate;
        float gate = __half2float(wi_out[base + col]);
        float up = __half2float(wi_out[base + intermediate + col]);
        float act_f32 = gate * gelu_erf_cdf(gate);
        __half act_f16 = __float2half(act_f32); // ROUND 1
        float out_f32 = __half2float(act_f16) * up;
        out[idx] = __float2half(out_f32); // ROUND 2
    }
}

// ---------------------------------------------------------------------
// Backward: writes BOTH halves of dwi_out in one launch.
// ---------------------------------------------------------------------

extern "C" __global__ void geglu_bwd_dwi_out_f16(
    const __half* wi_out, const __half* dy, __half* dwi_out,
    const unsigned int intermediate, const unsigned int n_out
) {
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_out;
         idx += blockDim.x * gridDim.x) {
        unsigned int row = idx / intermediate;
        unsigned int col = idx % intermediate;
        size_t base = (size_t)row * 2 * (size_t)intermediate;
        float gate = __half2float(wi_out[base + col]);
        float up = __half2float(wi_out[base + intermediate + col]);
        float dyi = __half2float(dy[idx]);
        float cdf = gelu_erf_cdf(gate);
        float pdf = gelu_erf_pdf(gate);
        float gelu_val = gate * cdf;
        float gelu_deriv = cdf + gate * pdf;
        dwi_out[base + col] = __float2half(dyi * up * gelu_deriv);
        dwi_out[base + intermediate + col] = __float2half(dyi * gelu_val);
    }
}
