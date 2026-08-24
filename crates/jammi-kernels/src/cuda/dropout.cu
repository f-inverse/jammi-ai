// dropout.cu — device-side counter-based dropout: y = x * mask * scale,
// mask computed in-kernel from Philox4x32-10 and never materialized.
// Compiled to PTX only when the `cuda` feature is active (see
// ../../build.rs); the pinned build flags (sm_80 baseline, no
// -use_fast_math) live there, not here.
//
// Philox4x32-10 ported from Random123 (D. E. Shaw Research,
// https://github.com/DEShawResearch/random123, BSD-3-Clause — see that
// repository's LICENSE; NOT curand's headers, which ship their own Philox
// under an EULA this build never reads or reuses). This is a SECOND,
// textually independent implementation of the SAME published algorithm as
// ../philox.rs's Rust port — there is no shared source between a .rs file
// and a .cu file compiled by nvcc. `philox_kat` below (a minimal test-only
// entry point, not used by the dropout kernels themselves) is what
// ../../tests/cuda_parity.rs's `philox_kat_vectors_match_on_cuda` calls to
// prove this device function reproduces Random123's published
// known-answer test vectors bit-for-bit, identically to ../philox.rs's own
// KAT tests — that pair of tests is the actual proof the two
// implementations compute the same function, not merely "both compile".
//
// Domain: contiguous input only (a raw-pointer kernel has no flat linear
// index for a strided view — the Rust glue in ../cuda/dropout.rs slices to
// `contiguous_offsets()` before any launch); F32 and BF16. This file
// assumes a flat linear index (the tensor's LOGICAL index, which coincides
// with the storage index exactly because the input is contiguous) and
// does not itself re-validate contiguity or dtype.
#include <cuda_bf16.h>
#include <cstddef>

#define PHILOX_M0 0xD2511F53u
#define PHILOX_M1 0xCD9E8D57u
#define PHILOX_W0 0x9E3779B9u
#define PHILOX_W1 0xBB67AE85u

// One `_philox4x32round` (philox.h): (hi0,lo0) = mulhilo32(M0, ctr[0]),
// (hi1,lo1) = mulhilo32(M1, ctr[2]), output
// {hi1^ctr[1]^key[0], lo1, hi0^ctr[3]^key[1], lo0}. `__umulhi` is CUDA's
// exact 32x32->64 high-half intrinsic (matching Rust's
// `((a as u64 * b as u64) >> 32) as u32` exactly — an EXACT integer
// operation, no rounding question, unlike the final scale multiply below).
__device__ __forceinline__ void philox4x32_round(unsigned int ctr[4], const unsigned int key[2]) {
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

// `_philox4x32bumpkey`: key[0] += W0; key[1] += W1 (unsigned add: defined
// wraparound, matching Rust's `wrapping_add`).
__device__ __forceinline__ void philox4x32_bumpkey(unsigned int key[2]) {
    key[0] += PHILOX_W0;
    key[1] += PHILOX_W1;
}

// philox4x32-10(counter, key) -> 4 words, written into `ctr` in place.
// Round 1 runs with the UNBUMPED key; key is bumped before each of rounds
// 2..=10 (9 bumps for 10 rounds) — see ../philox.rs's module doc for why
// this exact ordering is load-bearing (it is what the KAT vectors pin).
__device__ __forceinline__ void philox4x32_10(unsigned int ctr[4], unsigned int key[2]) {
    philox4x32_round(ctr, key);
    for (int r = 1; r < 10; ++r) {
        philox4x32_bumpkey(key);
        philox4x32_round(ctr, key);
    }
}

// The counter mapping (../philox.rs's `philox_draw`, quoted identically
// here): key = (seed lo, seed hi); counter = (layer_id, forward_idx,
// element_index lo, element_index hi). Returns Philox's first output word
// as "the draw" — see ../philox.rs's module doc for why only 1 of 4 words
// is used (memory-bound kernel, simplicity over throughput).
__device__ __forceinline__ unsigned int philox_draw(
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
    philox4x32_10(ctr, key);
    return ctr[0];
}

// ---------------------------------------------------------------------
// Test-only entry point: writes philox4x32_10(ctr, key) for a SINGLE
// (ctr, key) pair (one thread, one block) into `out[4]`. Not used by the
// dropout kernels below (which inline the same device function directly);
// this exists solely so ../../tests/cuda_parity.rs can load it and compare
// against Random123's published KAT vectors, proving THIS device function
// (not just the .rs port) reproduces them.
// ---------------------------------------------------------------------
extern "C" __global__ void philox_kat(
    const unsigned int ctr0, const unsigned int ctr1,
    const unsigned int ctr2, const unsigned int ctr3,
    const unsigned int key0, const unsigned int key1,
    unsigned int *out
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned int ctr[4] = {ctr0, ctr1, ctr2, ctr3};
        unsigned int key[2] = {key0, key1};
        philox4x32_10(ctr, key);
        out[0] = ctr[0];
        out[1] = ctr[1];
        out[2] = ctr[2];
        out[3] = ctr[3];
    }
}

// ---------------------------------------------------------------------
// Dropout forward: y = KEEP ? x * scale : 0, one thread per element, a
// grid-stride loop over `n` (this tensor's elem_count == the flat range
// `contiguous_offsets()` sliced the Rust glue to). `threshold`/`scale` are
// the SAME `u64`/`f32` ../ops/dropout.rs's `DropoutFused::new` computes
// host-side, in f64, ONCE — never recomputed here. The comparison runs in
// `unsigned long long` (`u64`) so `threshold == 2^32` (p == 0.0) never
// wraps, exactly matching the CPU arm.
//
// `__fmul_rn` (explicitly round-to-nearest, never fused into a
// neighboring add — there IS no neighboring add in this lone multiply) is
// what pins this kernel's applied value bit-for-bit against plain Rust
// `f32 * f32` on the CPU arm — see ../ops/dropout.rs's module doc.
// ---------------------------------------------------------------------

extern "C" __global__ void dropout_fwd_f32(
    const unsigned long long seed,
    const unsigned int layer_id,
    const unsigned int forward_idx,
    const unsigned long long threshold,
    const float scale,
    const float *x,
    float *out,
    const unsigned long long n
) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)blockDim.x * gridDim.x) {
        unsigned int draw = philox_draw(seed, layer_id, forward_idx, i);
        if ((unsigned long long)draw < threshold) {
            out[i] = __fmul_rn(x[i], scale);
        } else {
            out[i] = 0.0f;
        }
    }
}

extern "C" __global__ void dropout_fwd_bf16(
    const unsigned long long seed,
    const unsigned int layer_id,
    const unsigned int forward_idx,
    const unsigned long long threshold,
    const float scale,
    const __nv_bfloat16 *x,
    __nv_bfloat16 *out,
    const unsigned long long n
) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)blockDim.x * gridDim.x) {
        unsigned int draw = philox_draw(seed, layer_id, forward_idx, i);
        if ((unsigned long long)draw < threshold) {
            float xv = __bfloat162float(x[i]);
            out[i] = __float2bfloat16(__fmul_rn(xv, scale));
        } else {
            out[i] = __float2bfloat16(0.0f);
        }
    }
}
