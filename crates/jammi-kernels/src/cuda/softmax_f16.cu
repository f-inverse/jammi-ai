// softmax_f16.cu — fused masked softmax-last-dim forward + backward
// (output-only bwd), F16 monomorphic arm. Compiled to PTX only when the
// `cuda` feature is active (see ../../build.rs); the pinned build flags
// (sm_80 baseline, no -use_fast_math) live there, not here.
//
// DELIBERATE DUPLICATION (campaign #443 W2b contract) — see
// `layer_norm_f16.cu`'s identical note: this is a SEPARATE translation
// unit from `softmax.cu`, with its own copies of every file-scope
// `__device__` helper (including its own f16 analogs of
// `bf16_mul_rounded`/`bf16_add_rounded`) and its own `#include
// <cuda_fp16.h>` — NOT a shared `.cuh`. `softmax.cu` is byte-untouched.
//
// Domain, block/launch shape, and the mask-broadcast indexing are
// IDENTICAL to `softmax.cu`'s module doc. Per the per-op f16
// reference-regime table (`docs/maintainer/cuda-kernel-guide.md` §3.10),
// this op is DTYPE-NATIVE at two points — the scale-multiply and the
// mask-add each round to f16 immediately, matching `half::f16`'s own
// `Mul`/`Add` impls and `candle_nn::ops::softmax`'s native f16
// `broadcast_add` — every step AFTER the mask add (max/exp/sum/normalize)
// stays f32, exactly like the BF16 arm's identical convention.
#include <cuda_fp16.h>
#include <cstddef>

#define SM_BLOCK 256

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

__device__ __forceinline__ float block_reduce_max(float val, float* scratch) {
    int tid = threadIdx.x;
    scratch[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }
    float total = scratch[0];
    __syncthreads();
    return total;
}

// F16 analog of `softmax.cu`'s `bf16_mul_rounded`: rounds `s * scale_f16`
// to F16 immediately, matching `half::f16`'s own `Mul` impl (round-trip
// through f32, round once).
__device__ __forceinline__ __half f16_mul_rounded(__half s, __half scale_f16) {
    return __float2half(__half2float(s) * __half2float(scale_f16));
}

// F16 analog of `softmax.cu`'s `bf16_add_rounded`: rounds `scores_i +
// mask_i` to F16 immediately, matching `candle_nn::ops::softmax`'s native
// F16 `broadcast_add` (which rounds at exactly this step).
__device__ __forceinline__ float f16_add_rounded(__half a, __half b) {
    float sum = __half2float(a) + __half2float(b);
    return __half2float(__float2half(sum));
}

// Identical to `softmax.cu`'s `mask_row_offset` — see that file's doc.
__device__ __forceinline__ size_t mask_row_offset(
    size_t row,
    unsigned int s0, unsigned int s1, unsigned int s2,
    unsigned int m0, unsigned int m1, unsigned int m2
) {
    unsigned int idx2 = (unsigned int)(row % s2);
    size_t rem = row / s2;
    unsigned int idx1 = (unsigned int)(rem % s1);
    rem = rem / s1;
    unsigned int idx0 = (unsigned int)(rem % s0);
    unsigned int mi0 = (m0 == 1) ? 0 : idx0;
    unsigned int mi1 = (m1 == 1) ? 0 : idx1;
    unsigned int mi2 = (m2 == 1) ? 0 : idx2;
    return ((size_t)mi0 * m1 + mi1) * m2 + mi2;
}

// Identical to `softmax.cu`'s `mask_row_is_fully_masked` — see that
// file's doc.
__device__ __forceinline__ bool mask_row_is_fully_masked(float mask_max) {
    return mask_max < 0.0f;
}

// ---------------------------------------------------------------------
// Forward: y = softmax(scores + mask, last dim).
// ---------------------------------------------------------------------

extern "C" __global__ void softmax_fwd_f16(
    const __half* scores, const __half* mask, __half* y,
    const unsigned int last,
    const unsigned int s0, const unsigned int s1, const unsigned int s2,
    const unsigned int m0, const unsigned int m1, const unsigned int m2,
    const unsigned int zero_on_fully_masked,
    const float scale
) {
    __shared__ float scratch[SM_BLOCK];
    size_t row = blockIdx.x;
    size_t mrow = mask_row_offset(row, s0, s1, s2, m0, m1, m2);
    const __half* sr = scores + row * (size_t)last;
    const __half* mr = mask + mrow * (size_t)last;
    __half* yr = y + row * (size_t)last;
    const __half scale_f16 = __float2half(scale);

    if (zero_on_fully_masked) {
        float mask_max = -INFINITY;
        for (unsigned int i = threadIdx.x; i < last; i += blockDim.x) {
            mask_max = fmaxf(mask_max, __half2float(mr[i]));
        }
        mask_max = block_reduce_max(mask_max, scratch);
        if (mask_row_is_fully_masked(mask_max)) {
            for (unsigned int i = threadIdx.x; i < last; i += blockDim.x) {
                yr[i] = __float2half(0.0f);
            }
            return;
        }
    }

    float maxv = -INFINITY;
    for (unsigned int i = threadIdx.x; i < last; i += blockDim.x) {
        __half scaled = f16_mul_rounded(sr[i], scale_f16);
        float v = f16_add_rounded(scaled, mr[i]);
        maxv = fmaxf(maxv, v);
    }
    maxv = block_reduce_max(maxv, scratch);

    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < last; i += blockDim.x) {
        __half scaled = f16_mul_rounded(sr[i], scale_f16);
        float v = f16_add_rounded(scaled, mr[i]);
        sum += expf(v - maxv);
    }
    sum = block_reduce_sum(sum, scratch);

    for (unsigned int i = threadIdx.x; i < last; i += blockDim.x) {
        __half scaled = f16_mul_rounded(sr[i], scale_f16);
        float v = f16_add_rounded(scaled, mr[i]);
        float e = expf(v - maxv);
        yr[i] = __float2half(e / sum);
    }
}

// ---------------------------------------------------------------------
// Backward: dscores = (dy - dot(dy, y)) * y.
// ---------------------------------------------------------------------

extern "C" __global__ void softmax_bwd_dscores_f16(
    const __half* y, const __half* dy, __half* dscores, const unsigned int last
) {
    __shared__ float scratch[SM_BLOCK];
    size_t row = blockIdx.x;
    const __half* yr = y + row * (size_t)last;
    const __half* dyr = dy + row * (size_t)last;
    __half* dsr = dscores + row * (size_t)last;

    float dot = 0.0f;
    for (unsigned int i = threadIdx.x; i < last; i += blockDim.x) {
        dot += __half2float(dyr[i]) * __half2float(yr[i]);
    }
    dot = block_reduce_sum(dot, scratch);

    for (unsigned int i = threadIdx.x; i < last; i += blockDim.x) {
        float yv = __half2float(yr[i]);
        float dyv = __half2float(dyr[i]);
        dsr[i] = __float2half((dyv - dot) * yv);
    }
}
