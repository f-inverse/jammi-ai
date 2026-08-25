// scaled_cast_add.cu — out = round(base + lora * scaling), elementwise,
// where `round` is a single cast to `base`'s own dtype AFTER the add.
// Compiled to PTX only when the `cuda` feature is active (see ../../build.rs);
// the pinned build flags (sm_80 baseline, no -use_fast_math) live there, not
// here.
//
// Domain: contiguous, identically-shaped inputs only, `base`/`lora` each
// independently f32 or bf16 (four kernels below, one per combination). The
// Rust glue (../cuda/scaled_cast_add.rs) checks shape/dtype/contiguity
// before a launch; these kernels assume a flat linear index and do not
// re-validate.
//
// Rounding model (esc-046, GH#374; matches the CPU arm in
// ../ops/scaled_cast_add.rs and its module doc): `base` widens to `f32`
// (lossless), adds the already-`f32`-scaled `lora`, rounds ONCE to `base`'s
// dtype. Matches PEFT's `Linear.forward` (`peft/tuners/lora/layer.py`
// 1044-1069, `v0.20.0`): torch's `+` promotes a bf16 `result` to the
// delta's `f32` dtype (no rounding lost on `result`'s side), adds in f32,
// and only THEN casts back down once via `.to(torch_result_dtype)` — ONE
// round point, not two. An earlier revision of this kernel rounded the
// scaled delta to bf16 FIRST (an extra round point never present in the
// PEFT reference); `Axpy`'s "f32-accumulate, round once" precedent is the
// one this op now follows too, not an exception to it.
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
        // ONE round point (esc-046): `base` widens to f32, adds the
        // already-f32-scaled `lora`, rounds once on store — no
        // intermediate bf16-rounded `delta`.
        float bv = __bfloat162float(base[i]);
        out[i] = __float2bfloat16(bv + lora[i] * scaling);
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
        float bv = __bfloat162float(base[i]);
        float lv = __bfloat162float(lora[i]);
        out[i] = __float2bfloat16(bv + lv * scaling);
    }
}
