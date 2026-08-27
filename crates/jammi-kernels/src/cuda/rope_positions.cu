// rope_positions.cu — RoPE rotate-half on the FA2-packed `[total, 3, h, d]`
// `qkv` buffer (P6 Stage B B3-dense). Q/K slots (0, 1) get the SAME
// per-element expression `rope.cu` uses (shared via `rope_common.cuh`,
// see that header's doc); the V slot (2) passes through unchanged — this
// buffer is handed to `flash_attention_varlen` as ONE tensor, so a valid
// V channel must exist in the output even though RoPE never touches it.
//
// Domain: `qkv` is `[total, 3, h, d]`, row-major contiguous, bf16 or f32
// (validated by the Rust glue, `../ops/rope_positions.rs`'s
// `rope_positions_dims`); `cos`/`sin` are `[period, d]` with `period ==
// seq` (or `period == 1`, DENSE arm only), `d` even. `seq` is this call's
// modulo base: `token = row / (3*h)`; `position = token % seq` — the
// modulo form stated in the P6 Stage B v5 contract §3.6 ("position =
// token % s for dense").
//
// This SAME kernel serves BOTH the DENSE and the RAGGED (M1a — varlen
// positions) arms without any change here: the GENERAL varlen mechanism,
// `positions[r] = r - cu[seq(r)]` (a per-row lookup TABLE rather than one
// shared `seq`), is implemented on the RUST side as a pre-gather —
// `../ops/rope_positions.rs`'s `gather_ragged_tables` index-selects the
// caller's base `cos`/`sin` into a per-row `[total, d]` table BEFORE this
// kernel ever launches, and the launch is made with `seq` set to `total`
// (the gathered table's own row count). `token % seq` with `seq == total`
// degenerates to `token % total == token` for every `token < total` — the
// row index itself — so `position = token % seq`'s closed form, unchanged
// here, already computes the right answer once the table has been
// pre-gathered: no per-row lookup TABLE indexing was ever needed INSIDE
// this kernel. See `../ops/rope_positions.rs`'s module doc, "The ragged
// arm" section, for the Rust-side arm discriminant (`PositionArm`) that
// selects which GUARDS apply to a given call — this kernel's own
// per-element math has no arm-specific branch at all.
//
// ONE THREAD PER OUTPUT ELEMENT (`n = total*3*h*d`), grid-stride, no
// per-row reduction — same shape as `rope.cu`'s own launch.
#include <cuda_bf16.h>
#include <cstddef>
#include "rope_common.cuh"

extern "C" __global__ void rope_positions_fwd_f32(
    const float* qkv, const float* cos_t, const float* sin_t, float* out,
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
        float xv = qkv[i];
        float rh = (d_idx < half) ? -qkv[row_base + d_idx + half] : qkv[row_base + d_idx - half];
        float c = cos_t[table_base + d_idx];
        float s = sin_t[table_base + d_idx];
        out[i] = rope_rotate(xv, rh, c, s, sign);
    }
}

extern "C" __global__ void rope_positions_fwd_bf16(
    const __nv_bfloat16* qkv, const __nv_bfloat16* cos_t, const __nv_bfloat16* sin_t,
    __nv_bfloat16* out, const unsigned int h, const unsigned int d, const unsigned int seq,
    const float sign, const size_t n
) {
    const unsigned int half = d / 2;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)blockDim.x * gridDim.x) {
        unsigned int d_idx = (unsigned int)(i % d);
        size_t row = i / d;
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
        float xv = __bfloat162float(qkv[i]);
        float rh = (d_idx < half) ? -__bfloat162float(qkv[row_base + d_idx + half])
                                   : __bfloat162float(qkv[row_base + d_idx - half]);
        float c = __bfloat162float(cos_t[table_base + d_idx]);
        float s = __bfloat162float(sin_t[table_base + d_idx]);
        out[i] = __float2bfloat16(rope_rotate(xv, rh, c, s, sign));
    }
}
