// rope_f16.cu — fused rotate-half RoPE forward (and, via `sign`, the same
// kernel reused for backward — see `../ops/rope.rs`'s module doc), F16
// monomorphic arm. Compiled to PTX only when the `cuda` feature is active
// (see ../../build.rs); the pinned build flags (sm_80 baseline, no
// -use_fast_math) live there, not here.
//
// DELIBERATE DUPLICATION (campaign #443 W2b contract) — see
// `layer_norm_f16.cu`'s identical note. UNLIKE `rope.cu` (which shares its
// `rope_rotate` device function with `rope_positions.cu` via
// `rope_common.cuh`), this file carries its OWN copy of `rope_rotate`
// rather than including that header: the contract's "no shared `.cuh` for
// the new f16 files" rule applies even where the existing bf16 sibling
// itself uses one. `rope.cu`/`rope_common.cuh` are byte-untouched.
//
// Domain and the period-modulo indexing are IDENTICAL to `rope.cu`'s
// module doc. Per the per-op f16 reference-regime table
// (`docs/maintainer/cuda-kernel-guide.md` §3.10), this op is f32-internal
// (accumulate in f32, matching `layer_norm`'s BF16 arms), ONE rounding to
// f16 on the way out — the exact same regime as the existing BF16 arm,
// substituting the narrower 16-bit type.
#include <cuda_fp16.h>
#include <cstddef>

// Own copy of `rope_common.cuh`'s `rope_rotate` — see this file's module
// doc for why this is duplicated rather than `#include`d.
__device__ __forceinline__ float rope_rotate_f16(
    const float value, const float partner, const float c, const float s, const float sign
) {
    return value * c + partner * s * sign;
}

extern "C" __global__ void rope_fwd_f16(
    const __half* x, const __half* cos_t, const __half* sin_t,
    __half* out, const unsigned int hidden, const unsigned int period,
    const float sign, const size_t n
) {
    const unsigned int half = hidden / 2;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)blockDim.x * gridDim.x) {
        unsigned int col = (unsigned int)(i % hidden);
        size_t row = i / hidden;
        unsigned int seq_idx = (unsigned int)(row % period);
        size_t row_base = row * (size_t)hidden;
        size_t table_base = (size_t)seq_idx * hidden;
        float xv = __half2float(x[i]);
        float rh = (col < half) ? -__half2float(x[row_base + col + half])
                                 : __half2float(x[row_base + col - half]);
        float c = __half2float(cos_t[table_base + col]);
        float s = __half2float(sin_t[table_base + col]);
        out[i] = __float2half(rope_rotate_f16(xv, rh, c, s, sign));
    }
}
