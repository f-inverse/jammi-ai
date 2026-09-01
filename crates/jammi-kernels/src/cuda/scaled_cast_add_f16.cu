// scaled_cast_add_f16.cu — out = round(base + lora * scaling), elementwise,
// F16 monomorphic arms. Compiled to PTX only when the `cuda` feature is
// active (see ../../build.rs); the pinned build flags (sm_80 baseline, no
// -use_fast_math) live there, not here.
//
// DELIBERATE DUPLICATION (campaign #443 W2b/W2c contract) — see
// `layer_norm_f16.cu`'s identical note. This is a SEPARATE translation unit
// from `scaled_cast_add.cu` (a separate PTX module,
// `PTX_SCALED_CAST_ADD_F16` in `../mod.rs`), so that file stays
// byte-untouched — provable by `git diff`. No shared `.cuh`.
//
// Three combinations (base, lora) in {F16, F32} minus the all-F32 case
// (already covered by `scaled_cast_add_f32_f32` in `scaled_cast_add.cu`):
// `F16`+`F32`, `F32`+`F16`, `F16`+`F16` — mirroring `ops/scaled_cast_add.rs`'s
// own CPU-side split (`scaled_cast_add_f16_f32`/`scaled_cast_add_f32_f16`/
// `scaled_cast_add_f16_f16`). Output dtype follows `base`'s dtype, exactly
// like the existing F32/BF16 matrix.
//
// Rounding model (esc-046, GH#374; matches `scaled_cast_add.cu`'s own module
// doc and `ops/scaled_cast_add.rs`'s CPU arms): `base` widens to `f32`
// (lossless), adds the already-`f32`-scaled `lora`, rounds ONCE to `base`'s
// dtype. `__fmul_rn`/`__fadd_rn` keep the multiply and the add as two
// SEPARATELY rounded f32 operations (never let nvcc's `--fmad=true` default
// fuse them into one `fma.rn.f32`), for the identical reason
// `scaled_cast_add.cu`'s module doc states at length.
#include <cuda_fp16.h>
#include <cstddef>

extern "C" __global__ void scaled_cast_add_f16_f32(
    const float scaling,
    const __half *base,
    const float *lora,
    __half *out,
    const size_t n
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float bv = __half2float(base[i]);
        out[i] = __float2half(__fadd_rn(bv, __fmul_rn(lora[i], scaling)));
    }
}

extern "C" __global__ void scaled_cast_add_f32_f16(
    const float scaling,
    const float *base,
    const __half *lora,
    float *out,
    const size_t n
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float lv = __half2float(lora[i]);
        out[i] = __fadd_rn(base[i], __fmul_rn(lv, scaling));
    }
}

extern "C" __global__ void scaled_cast_add_f16_f16(
    const float scaling,
    const __half *base,
    const __half *lora,
    __half *out,
    const size_t n
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float bv = __half2float(base[i]);
        float lv = __half2float(lora[i]);
        out[i] = __float2half(__fadd_rn(bv, __fmul_rn(lv, scaling)));
    }
}
