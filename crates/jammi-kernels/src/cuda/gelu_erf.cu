// gelu_erf.cu -- fused erf-based GELU forward + backward, F32 + BF16.
// Compiled to PTX only when the `cuda` feature is active (see
// ../../build.rs); the pinned build flags (sm_80 baseline, no
// -use_fast_math) live there, not here.
//
// Domain: `x` is contiguous, non-empty (the Rust glue in ../cuda/gelu_erf.rs
// validates both before any launch; this file assumes a flat linear index
// and does not re-validate). Purely elementwise, single-pass launch (`i <
// n`, one thread per element, `LaunchConfig::for_num_elems` from the host
// side) -- no per-row reduction, matching `scaled_cast_add.cu`'s /
// `rope.cu`'s shape rather than `geglu.cu`'s grid-stride loop (this op has
// no packed-halves indexing to walk PAST the launch grid for).
//
// Forward tracks candle-kernels' OWN `ugelu_erf_{f32,bf16}` bit-for-bit
// (`unary.cu:31-34,118`; `cuda_utils.cuh:140,174`) -- see
// `../ops/gelu_erf.rs`'s module doc, "three cdf formulations", item 2:
//   F32:  out = x * normcdff(x)                              -- one rounding.
//   BF16: cdf16 = round16(normcdff(f32(x)))                  -- ROUND 1
//         out   = round16(f32(x) * f32(cdf16))  (== __hmul)  -- ROUND 2
// `__hmul` on two values that are each exactly representable in f32
// correctly rounds their EXACT product -- bit-identical to computing the
// product in f32 and rounding once, which is why this matches candle's own
// `x * normcdfg(x)` (native `__nv_bfloat16::operator*`) exactly.
//
// Backward: ONE kernel, `dx = dy * (Phi(x) + x*phi(x))`, `Phi = normcdff`
// (matching this file's own forward CDF routine), `phi(x) = kBeta *
// expf(-0.5*x*x)`, `kBeta = 1/sqrt(2*pi)` (ATen's constant -- see
// `../ops/gelu_erf.rs`'s module doc, "backward" section). f32 math
// throughout, one rounding at the store; bf16 upcasts at load.
#include <cuda_bf16.h>
#include <cstddef>

#define GELU_ERF_KBETA 0.3989422804014327f

__device__ __forceinline__ float gelu_erf_pdf(float x) {
    return GELU_ERF_KBETA * expf(-0.5f * x * x);
}

// ---------------------------------------------------------------------
// Forward.
// ---------------------------------------------------------------------

extern "C" __global__ void gelu_erf_fwd_f32(
    const float* x, float* out, const size_t n
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float xv = x[i];
        out[i] = xv * normcdff(xv);
    }
}

extern "C" __global__ void gelu_erf_fwd_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* out, const size_t n
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        __nv_bfloat16 xb = x[i];
        float cdf_f32 = normcdff(__bfloat162float(xb));
        __nv_bfloat16 cdf16 = __float2bfloat16(cdf_f32); // ROUND 1
        out[i] = __hmul(xb, cdf16); // ROUND 2 -- bit-identical to candle's `x * normcdfg(x)`.
    }
}

// ---------------------------------------------------------------------
// Backward: dx = dy * (Phi(x) + x*phi(x)).
// ---------------------------------------------------------------------

extern "C" __global__ void gelu_erf_bwd_dx_f32(
    const float* x, const float* dy, float* dx, const size_t n
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float xv = x[i];
        float cdf = normcdff(xv);
        float pdf = gelu_erf_pdf(xv);
        dx[i] = dy[i] * (cdf + xv * pdf);
    }
}

extern "C" __global__ void gelu_erf_bwd_dx_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* dy, __nv_bfloat16* dx, const size_t n
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float xv = __bfloat162float(x[i]);
        float dyv = __bfloat162float(dy[i]);
        float cdf = normcdff(xv);
        float pdf = gelu_erf_pdf(xv);
        dx[i] = __float2bfloat16(dyv * (cdf + xv * pdf));
    }
}
