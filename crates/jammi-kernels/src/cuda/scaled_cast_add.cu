// scaled_cast_add.cu — out = base + round(lora * scaling), elementwise,
// where `round` is a cast to `base`'s own dtype. Compiled to PTX only when
// the `cuda` feature is active (see ../../build.rs); the pinned build flags
// (sm_80 baseline, no -use_fast_math) live there, not here.
//
// Domain: contiguous, identically-shaped inputs only, `base`/`lora` each
// independently f32 or bf16 (four kernels below, one per combination). The
// Rust glue (../cuda/scaled_cast_add.rs) checks shape/dtype/contiguity
// before a launch; these kernels assume a flat linear index and do not
// re-validate.
//
// Rounding model (matches the CPU arm in ../ops/scaled_cast_add.rs and its
// module doc): the scaled delta is rounded to `base`'s dtype FIRST (a
// distinct step, matching PEFT's `.to_dtype(base_out.dtype())`), THEN
// added — two round points for a bf16 `base`, reproducing the eager
// `[mul, cast, add]` composition's own rounding path rather than
// accumulating everything in f32 and rounding once (the `Axpy` precedent
// this kernel deliberately does NOT follow — see the module doc).
#include <cuda_bf16.h>
#include <cstddef>

extern "C" __global__ void scaled_cast_add_f32_f32(
    const float scaling,
    const float *base,
    const float *lora,
    float *out,
    const size_t n
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = base[i] + lora[i] * scaling;
    }
}

extern "C" __global__ void scaled_cast_add_f32_bf16(
    const float scaling,
    const float *base,
    const __nv_bfloat16 *lora,
    float *out,
    const size_t n
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float lv = __bfloat162float(lora[i]);
        out[i] = base[i] + lv * scaling;
    }
}

extern "C" __global__ void scaled_cast_add_bf16_f32(
    const float scaling,
    const __nv_bfloat16 *base,
    const float *lora,
    __nv_bfloat16 *out,
    const size_t n
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        // Round point 1: the scaled delta, rounded to bf16 (matches
        // eager's explicit `.to_dtype(base_out.dtype())`).
        __nv_bfloat16 delta = __float2bfloat16(lora[i] * scaling);
        // Round point 2: the add itself, promote-compute-round-once.
        float bv = __bfloat162float(base[i]);
        float dv = __bfloat162float(delta);
        out[i] = __float2bfloat16(bv + dv);
    }
}

extern "C" __global__ void scaled_cast_add_bf16_bf16(
    const float scaling,
    const __nv_bfloat16 *base,
    const __nv_bfloat16 *lora,
    __nv_bfloat16 *out,
    const size_t n
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float lv = __bfloat162float(lora[i]);
        __nv_bfloat16 delta = __float2bfloat16(lv * scaling);
        float bv = __bfloat162float(base[i]);
        float dv = __bfloat162float(delta);
        out[i] = __float2bfloat16(bv + dv);
    }
}
