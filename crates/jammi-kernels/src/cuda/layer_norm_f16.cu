// layer_norm_f16.cu — fused, bias-free LayerNorm forward + backward, F16
// monomorphic arm. Compiled to PTX only when the `cuda` feature is active
// (see ../../build.rs); the pinned build flags (sm_80 baseline, no
// -use_fast_math) live there, not here.
//
// DELIBERATE DUPLICATION (campaign #443 W2b contract): this file is a
// SEPARATE translation unit from `layer_norm.cu` (the existing F32/BF16
// kernels), carrying its OWN copy of every file-scope `__device__` helper
// and its own `#include <cuda_fp16.h>` — NOT a shared `.cuh` (`build.rs`'s
// own comment on `rope_common.cuh` documents the non-tracked-header
// staleness hazard: `bindgen_cuda` does not track `#include`d header
// dependencies, so a header-only edit would not trigger recompilation of
// every `.cu` that includes it). Keeping this file wholly separate also
// makes `layer_norm.cu`'s own byte-identity (this campaign's audit
// assertion — `git diff` on it must be empty) trivially provable: nothing
// in this file ever touches that one.
//
// Domain, block/launch shape, and the two-pass `dgamma` design are
// IDENTICAL to `layer_norm.cu`'s module doc — see that file for the full
// design rationale. This file only substitutes `__half` for
// `__nv_bfloat16` and matches the per-op f16 reference-regime table
// (`docs/maintainer/cuda-kernel-guide.md` §3.10): f32-internal
// (mean/var/xhat accumulate in f32), ONE rounding to f16 on the way out —
// the exact same regime as the existing BF16 arms, substituting the
// narrower 16-bit type.
#include <cuda_fp16.h>
#include <cstddef>

#define LN_BLOCK 256

__device__ __forceinline__ float block_reduce_sum(float val, float* scratch) {
    int tid = threadIdx.x;
    scratch[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    float total = scratch[0];
    __syncthreads();
    return total;
}

// ---------------------------------------------------------------------
// Forward: y = ((x - mean) * invvar) * gamma. f32 accumulation
// throughout; one rounding to f16.
// ---------------------------------------------------------------------

extern "C" __global__ void layer_norm_fwd_f16(
    const __half* x, const __half* gamma, __half* y,
    const unsigned int hidden, const float eps
) {
    __shared__ float scratch[LN_BLOCK];
    size_t row = blockIdx.x;
    const __half* xr = x + row * (size_t)hidden;
    __half* yr = y + row * (size_t)hidden;

    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        sum += __half2float(xr[i]);
    }
    sum = block_reduce_sum(sum, scratch);
    float mean = sum / (float)hidden;

    float sumsq = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float d = __half2float(xr[i]) - mean;
        sumsq += d * d;
    }
    sumsq = block_reduce_sum(sumsq, scratch);
    float invvar = rsqrtf(sumsq / (float)hidden + eps);

    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float xhat = (__half2float(xr[i]) - mean) * invvar;
        yr[i] = __float2half(xhat * __half2float(gamma[i]));
    }
}

// ---------------------------------------------------------------------
// Backward dx: recompute mean/invvar, then the two per-row scalars, then
// dx — one kernel launch, three grid-stride passes over the row.
// ---------------------------------------------------------------------

extern "C" __global__ void layer_norm_bwd_dx_f16(
    const __half* x, const __half* gamma, const __half* dy,
    __half* dx, const unsigned int hidden, const float eps
) {
    __shared__ float scratch[LN_BLOCK];
    size_t row = blockIdx.x;
    const __half* xr = x + row * (size_t)hidden;
    const __half* dyr = dy + row * (size_t)hidden;
    __half* dxr = dx + row * (size_t)hidden;

    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        sum += __half2float(xr[i]);
    }
    sum = block_reduce_sum(sum, scratch);
    float mean = sum / (float)hidden;

    float sumsq = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float d = __half2float(xr[i]) - mean;
        sumsq += d * d;
    }
    sumsq = block_reduce_sum(sumsq, scratch);
    float invvar = rsqrtf(sumsq / (float)hidden + eps);

    float sum_t = 0.0f;
    float sum_t_xhat = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float xhat = (__half2float(xr[i]) - mean) * invvar;
        float t = __half2float(dyr[i]) * __half2float(gamma[i]);
        sum_t += t;
        sum_t_xhat += t * xhat;
    }
    sum_t = block_reduce_sum(sum_t, scratch);
    sum_t_xhat = block_reduce_sum(sum_t_xhat, scratch);
    float mean_t = sum_t / (float)hidden;
    float mean_t_xhat = sum_t_xhat / (float)hidden;

    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float xhat = (__half2float(xr[i]) - mean) * invvar;
        float t = __half2float(dyr[i]) * __half2float(gamma[i]);
        dxr[i] = __float2half((t - mean_t - xhat * mean_t_xhat) * invvar);
    }
}

// ---------------------------------------------------------------------
// Backward dgamma, pass 1: per-row mean/invvar, cached once per row.
// ---------------------------------------------------------------------

extern "C" __global__ void layer_norm_row_stats_f16(
    const __half* x, float* mean_out, float* invvar_out,
    const unsigned int hidden, const float eps
) {
    __shared__ float scratch[LN_BLOCK];
    size_t row = blockIdx.x;
    const __half* xr = x + row * (size_t)hidden;

    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        sum += __half2float(xr[i]);
    }
    sum = block_reduce_sum(sum, scratch);
    float mean = sum / (float)hidden;

    float sumsq = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float d = __half2float(xr[i]) - mean;
        sumsq += d * d;
    }
    sumsq = block_reduce_sum(sumsq, scratch);
    float invvar = rsqrtf(sumsq / (float)hidden + eps);

    if (threadIdx.x == 0) {
        mean_out[row] = mean;
        invvar_out[row] = invvar;
    }
}

// ---------------------------------------------------------------------
// Backward dgamma, pass 2: column-tiled accumulation across rows, reading
// pass 1's cached per-row `mean`/`invvar`. Accumulates into an F32
// SCRATCH buffer (never f16 directly), matching the BF16 arm's identical
// convention — `dg_builder` rounds once, at the very end, to f16.
// ---------------------------------------------------------------------

extern "C" __global__ void layer_norm_bwd_dgamma_f16(
    const __half* x, const __half* dy, const float* mean,
    const float* invvar, float* dgamma_f32, const unsigned int rows,
    const unsigned int hidden
) {
    for (size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x; col < hidden;
         col += (size_t)blockDim.x * gridDim.x) {
        float acc = 0.0f;
        for (unsigned int r = 0; r < rows; r++) {
            float xhat =
                (__half2float(x[(size_t)r * hidden + col]) - mean[r]) * invvar[r];
            acc += __half2float(dy[(size_t)r * hidden + col]) * xhat;
        }
        dgamma_f32[col] = acc;
    }
}

// Elementwise F32 -> F16 rounding, used only to finish the f16 dgamma
// path above (one rounding, at the very end, off the reduction's own
// accumulation).
extern "C" __global__ void layer_norm_cast_f32_to_f16(
    const float* src, __half* dst, const unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2half(src[i]);
    }
}

// ---------------------------------------------------------------------
// #460 (C-LN): bias-carrying forward, F16. APPEND-ONLY — see
// `layer_norm.cu`'s identical comment above this block's F32/BF16 twin for
// the full design rationale (ATen citation, `--fmad=true` form, the
// `template <bool HAS_BETA>` shared-row-body shape). Every kernel ABOVE
// this comment in THIS file is byte-for-byte unchanged by this addition.
// ---------------------------------------------------------------------

template <bool HAS_BETA>
__device__ __forceinline__ void ln_fwd_row_body_f16(
    const __half* xr, const __half* gamma, const __half* beta, __half* yr,
    const unsigned int hidden, const float eps, float* scratch
) {
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        sum += __half2float(xr[i]);
    }
    sum = block_reduce_sum(sum, scratch);
    float mean = sum / (float)hidden;

    float sumsq = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float d = __half2float(xr[i]) - mean;
        sumsq += d * d;
    }
    sumsq = block_reduce_sum(sumsq, scratch);
    float invvar = rsqrtf(sumsq / (float)hidden + eps);

    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float xhat = (__half2float(xr[i]) - mean) * invvar;
        float scaled = xhat * __half2float(gamma[i]);
        float outv = HAS_BETA ? (scaled + __half2float(beta[i])) : scaled;
        yr[i] = __float2half(outv);
    }
}

extern "C" __global__ void layer_norm_fwd_f16_biased(
    const __half* x, const __half* gamma, const __half* beta, __half* y,
    const unsigned int hidden, const float eps
) {
    __shared__ float scratch[LN_BLOCK];
    size_t row = blockIdx.x;
    ln_fwd_row_body_f16<true>(
        x + row * (size_t)hidden, gamma, beta, y + row * (size_t)hidden,
        hidden, eps, scratch
    );
}
