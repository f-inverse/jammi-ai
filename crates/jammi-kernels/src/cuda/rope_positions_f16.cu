// rope_positions_f16.cu — RoPE rotate-half on the FA2-packed
// `[total, 3, h, d]` `qkv` buffer, F16 monomorphic arm. Compiled to PTX only
// when the `cuda` feature is active (see ../../build.rs); the pinned build
// flags (sm_80 baseline, no -use_fast_math) live there, not here.
//
// DELIBERATE DUPLICATION (campaign #443 W2b/W2c contract) — see
// `layer_norm_f16.cu`'s identical note. This is a SEPARATE translation unit
// from `rope_positions.cu` (a separate PTX module, `PTX_ROPE_POSITIONS_F16`
// in `../mod.rs`), so that file (and the `rope_common.cuh` header it
// shares with `rope.cu`) stay byte-untouched — provable by `git diff`. This
// file carries its OWN copy of `rope_rotate` (mirroring `rope_f16.cu`'s
// identical choice) rather than `#include`-ing `rope_common.cuh`: no shared
// `.cuh` for the new f16 files, even where the bf16 sibling uses one.
//
// Domain, indexing (V slot pass-through, dense/ragged dual-arm reuse via
// `seq`) and the accumulation regime are all IDENTICAL to
// `rope_positions.cu`'s module doc, substituting `__half` for
// `__nv_bfloat16`: f32-accumulate, ONE rounding to f16 on the way out
// (matching `ops/rope_positions.rs`'s `rope_positions_fwd_f16` CPU
// reference arm and `rope_f16.cu`'s own regime for the SAME per-element
// expression, per the per-op f16 reference-regime table,
// `docs/maintainer/cuda-kernel-guide.md` §3.10).
#include <cuda_fp16.h>
#include <cstddef>

// Own copy of `rope_common.cuh`'s `rope_rotate` — see this file's module
// doc for why this is duplicated rather than `#include`d.
__device__ __forceinline__ float rope_rotate_positions_f16(
    const float value, const float partner, const float c, const float s, const float sign
) {
    return value * c + partner * s * sign;
}

extern "C" __global__ void rope_positions_fwd_f16(
    const __half* qkv, const __half* cos_t, const __half* sin_t, __half* out,
    const unsigned int h, const unsigned int d, const unsigned int seq,
    const float sign, const size_t n
) {
    const unsigned int half = d / 2;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)blockDim.x * gridDim.x) {
        unsigned int d_idx = (unsigned int)(i % d);
        size_t row = i / d;               // row = token*3*h + slot*h + h_idx
        size_t row2 = row / h;
        unsigned int slot = (unsigned int)(row2 % 3);
        if (slot == 2) {
            out[i] = qkv[i];
            continue;
        }
        size_t token = row2 / 3;
        unsigned int seq_idx = (unsigned int)(token % seq);
        size_t row_base = i - d_idx;
        size_t table_base = (size_t)seq_idx * d;
        float xv = __half2float(qkv[i]);
        float rh = (d_idx < half) ? -__half2float(qkv[row_base + d_idx + half])
                                  : __half2float(qkv[row_base + d_idx - half]);
        float c = __half2float(cos_t[table_base + d_idx]);
        float s = __half2float(sin_t[table_base + d_idx]);
        out[i] = __float2half(rope_rotate_positions_f16(xv, rh, c, s, sign));
    }
}
