// rope.cu — fused rotate-half RoPE forward (and, via `sign`, the same
// kernel reused for backward — see `../ops/rope.rs`'s module doc).
// Compiled to PTX only when the `cuda` feature is active (see
// ../../build.rs); the pinned build flags (sm_80 baseline, no
// -use_fast_math) live there, not here.
//
// Domain: `x` is `[<any leading dims>, hidden]`, row-major contiguous (a
// possibly narrowed-but-contiguous nonzero-offset view is handled by the
// Rust glue slicing to `contiguous_offsets()` before the launch); `cos`/
// `sin` are `[period, hidden]` contiguous, `hidden` even, `total_rows =
// x_elem_count / hidden` an exact multiple of `period` — all enforced by
// the Rust glue (../rope.rs) before any launch, per ../ops/rope.rs's
// `rope_dims`.
//
// ONE THREAD PER OUTPUT ELEMENT, grid-stride (no per-row reduction of any
// kind — this op is purely elementwise, unlike LayerNorm — so there is no
// shared-memory scratch and no ceiling this kernel's own resource usage
// would impose; `MAX_HEAD_DIM` in ../ops/rope.rs is a validated-coverage
// ceiling, not a hardware one).
#include <cuda_bf16.h>
#include <cstddef>

extern "C" __global__ void rope_fwd_f32(
    const float* x, const float* cos_t, const float* sin_t, float* out,
    const unsigned int hidden, const unsigned int period, const float sign,
    const size_t n
) {
    const unsigned int half = hidden / 2;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)blockDim.x * gridDim.x) {
        unsigned int col = (unsigned int)(i % hidden);
        size_t row = i / hidden;
        unsigned int seq_idx = (unsigned int)(row % period);
        size_t row_base = row * (size_t)hidden;
        size_t table_base = (size_t)seq_idx * hidden;
        float xv = x[i];
        float rh = (col < half) ? -x[row_base + col + half] : x[row_base + col - half];
        float c = cos_t[table_base + col];
        float s = sin_t[table_base + col];
        out[i] = xv * c + rh * s * sign;
    }
}

extern "C" __global__ void rope_fwd_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* cos_t, const __nv_bfloat16* sin_t,
    __nv_bfloat16* out, const unsigned int hidden, const unsigned int period,
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
        float xv = __bfloat162float(x[i]);
        float rh = (col < half) ? -__bfloat162float(x[row_base + col + half])
                                 : __bfloat162float(x[row_base + col - half]);
        float c = __bfloat162float(cos_t[table_base + col]);
        float s = __bfloat162float(sin_t[table_base + col]);
        out[i] = __float2bfloat16(xv * c + rh * s * sign);
    }
}
