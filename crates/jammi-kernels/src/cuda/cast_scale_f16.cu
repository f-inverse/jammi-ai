// cast_scale_f16.cu — F16 monomorphic arms for `crate::ops::CastScaleF16F32`
// / `crate::ops::CastAddF16` — the F16 analogs of `cast_scale.cu`'s
// BF16-only `cast_scale_bf16_f32`/`cast_add_bf16` (see
// `../ops/cast_scale.rs`'s module doc for why these are NEW, independent
// types rather than widened match arms: `CastScaleBf16F32`/`CastAddBf16`
// are domain-restricted to BF16 by construction). Compiled to PTX only
// when the `cuda` feature is active (see ../../build.rs); the pinned build
// flags (sm_80 baseline, no -use_fast_math) live there, not here.
//
// DELIBERATE DUPLICATION (campaign #443 W2b/W2c contract) — see
// `layer_norm_f16.cu`'s identical note. This is a SEPARATE translation unit
// from `cast_scale.cu` (a separate PTX module, `PTX_CAST_SCALE_F16` in
// `../mod.rs`), so that file stays byte-untouched — provable by `git diff`.
// No shared `.cuh`.
//
// `cast_scale_f16_f32`: out = f32(x) * scale + 0.0, x required F16 — the
// `+ 0.0` term is REQUIRED (signed-zero identity), matching
// `cast_scale_bf16_f32`'s own expression exactly, substituting `__half`.
//
// `cast_add_f16`: out = base + round_to_f16(f32val), base required F16,
// f32val required F32 — `f32val` is rounded to `__half` FIRST
// (`__float2half`), then added natively in `__half`, matching
// `cast_add_bf16`'s rounding-order contract exactly, substituting `__half`.
#include <cuda_fp16.h>
#include <cstddef>

extern "C" __global__ void cast_scale_f16_f32(
    const float scale,
    const __half *x,
    float *out,
    const size_t n
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __half2float(x[i]) * scale + 0.0f;
    }
}

extern "C" __global__ void cast_add_f16(
    const __half *base,
    const float *f32val,
    __half *out,
    const size_t n
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        __half delta = __float2half(f32val[i]);
        out[i] = __hadd(base[i], delta);
    }
}
