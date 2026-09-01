// dropout_f16.cu — device-side counter-based dropout, F16 monomorphic arm:
// y = x * mask * scale, mask computed in-kernel from Philox4x32-10 and never
// materialized. Compiled to PTX only when the `cuda` feature is active (see
// ../../build.rs); the pinned build flags (sm_80 baseline, no
// -use_fast_math) live there, not here.
//
// DELIBERATE DUPLICATION (campaign #443 W2b/W2c contract) — see
// `layer_norm_f16.cu`'s identical note. This is a SEPARATE translation unit
// from `dropout.cu` (a separate PTX module, `PTX_DROPOUT_F16` in
// `../mod.rs`), so `dropout.cu` stays byte-untouched — provable by
// `git diff`. It carries its OWN copy of the Philox4x32-10 device functions
// (ported from Random123, D. E. Shaw Research, BSD-3-Clause — see
// `dropout.cu`'s module doc for the full citation, identical here) rather
// than `#include`-ing anything: no shared `.cuh`.
//
// Domain: contiguous input only, F16. The Rust glue (../cuda/dropout.rs)
// slices to `contiguous_offsets()` before any launch; this file assumes a
// flat linear index and does not itself re-validate contiguity or dtype.
//
// Regime (per the per-op f16 reference-regime table,
// `docs/maintainer/cuda-kernel-guide.md` §3.10): dtype-independent KEEP/DROP
// decision (the Philox draw is a pure function of position, never of the
// tensor's value or dtype) + f32-internal scale multiply on a KEPT element,
// ONE rounding point (a DROPPED element is an exact zero, no rounding at
// all) — matching `ops/dropout.rs`'s `dropout_f16` CPU reference arm
// exactly. `__fmul_rn` (explicit round-to-nearest, never fused into a
// neighboring add — there is none here) is the same pinning `dropout.cu`'s
// F32/BF16 kernels use, for the identical reason (see that file's module
// doc).
#include <cuda_fp16.h>
#include <cstddef>

#define PHILOX_M0 0xD2511F53u
#define PHILOX_M1 0xCD9E8D57u
#define PHILOX_W0 0x9E3779B9u
#define PHILOX_W1 0xBB67AE85u

__device__ __forceinline__ void philox4x32_round_f16(unsigned int ctr[4], const unsigned int key[2]) {
    unsigned int hi0 = __umulhi(PHILOX_M0, ctr[0]);
    unsigned int lo0 = PHILOX_M0 * ctr[0];
    unsigned int hi1 = __umulhi(PHILOX_M1, ctr[2]);
    unsigned int lo1 = PHILOX_M1 * ctr[2];
    unsigned int new0 = hi1 ^ ctr[1] ^ key[0];
    unsigned int new2 = hi0 ^ ctr[3] ^ key[1];
    ctr[0] = new0;
    ctr[1] = lo1;
    ctr[2] = new2;
    ctr[3] = lo0;
}

__device__ __forceinline__ void philox4x32_bumpkey_f16(unsigned int key[2]) {
    key[0] += PHILOX_W0;
    key[1] += PHILOX_W1;
}

__device__ __forceinline__ void philox4x32_10_f16(unsigned int ctr[4], unsigned int key[2]) {
    philox4x32_round_f16(ctr, key);
    for (int r = 1; r < 10; ++r) {
        philox4x32_bumpkey_f16(key);
        philox4x32_round_f16(ctr, key);
    }
}

__device__ __forceinline__ unsigned int philox_draw_f16(
    unsigned long long seed,
    unsigned int layer_id,
    unsigned int forward_idx,
    unsigned long long element_index
) {
    unsigned int key[2] = {
        static_cast<unsigned int>(seed),
        static_cast<unsigned int>(seed >> 32)
    };
    unsigned int ctr[4] = {
        layer_id,
        forward_idx,
        static_cast<unsigned int>(element_index),
        static_cast<unsigned int>(element_index >> 32)
    };
    philox4x32_10_f16(ctr, key);
    return ctr[0];
}

extern "C" __global__ void dropout_fwd_f16(
    const unsigned long long seed,
    const unsigned int layer_id,
    const unsigned int forward_idx,
    const unsigned long long threshold,
    const float scale,
    const __half *x,
    __half *out,
    const unsigned long long n
) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)blockDim.x * gridDim.x) {
        unsigned int draw = philox_draw_f16(seed, layer_id, forward_idx, i);
        if ((unsigned long long)draw < threshold) {
            float xv = __half2float(x[i]);
            out[i] = __float2half(__fmul_rn(xv, scale));
        } else {
            out[i] = __float2half(0.0f);
        }
    }
}
