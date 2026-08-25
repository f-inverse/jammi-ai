// cast_scale.cu — the two cast-boundary fusions for
// LowRankResidualLinear::bwd (design study v2's Wave 1 (e)/(f)). Compiled
// to PTX only when the `cuda` feature is active (see ../../build.rs); the
// pinned build flags (sm_80 baseline, no -use_fast_math) live there, not
// here.
//
// Domain: contiguous inputs only, `n` fits u32. The Rust glue
// (../cast_scale.rs) checks shape/dtype/contiguity before a launch; these
// kernels assume a flat linear index and do not re-validate.
#include <cuda_bf16.h>
#include <cstddef>

// out = f32(x) * scale + 0.0f, elementwise. Replaces candle's own
// `cast_bf16_f32` (candle-kernels-0.11.0/src/cast.cu, `out[i] = inp[i]`, an
// exact widening) THEN `affine_f32` (candle-kernels-0.11.0/src/
// affine.cu:45, `x * mul + add`) with ONE kernel, ONE HBM round-trip
// instead of two. `__bfloat162float` is the exact (lossless) widening
// candle's own cast uses (an implicit `operator float()` on
// `__nv_bfloat16`, bit-identical to this explicit call). The `+ 0.0f` is
// REQUIRED for signed-zero identity with `affine_f32`'s own expression —
// see ../ops/cast_scale.rs's module doc for the RED control this guards.
extern "C" __global__ void cast_scale_bf16_f32(
    const float scale,
    const __nv_bfloat16 *x,
    float *out,
    const size_t n
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __bfloat162float(x[i]) * scale + 0.0f;
    }
}

// out = base + round_to_bf16(f32val), elementwise. Replaces candle's own
// `cast_f32_bf16` (cast.cu, `out[i] = inp[i]`, an implicit
// `__nv_bfloat16(float)` constructor — round-to-nearest-even) THEN
// `badd_bf16` (candle-kernels-0.11.0/src/binary.cu:5, `x + y` on two
// `__nv_bfloat16` operands) with ONE kernel. The rounding order the task
// requires — round `f32val` to bf16 IN-REGISTER first, THEN add — is
// exactly the two-step body below: `delta` is a genuine `__nv_bfloat16`
// value (rounded via `__float2bfloat16`, candle's own cast idiom) before
// the `+`. The add itself uses the LITERAL expression `base[i] + delta`
// (not a widen-to-f32-and-round emulation) so the compiled SASS for this
// step is whatever nvcc emits for `x + y` on `__nv_bfloat16` at this
// crate's pinned sm_80 target — the identical expression candle's own
// `badd_bf16` compiles, by construction rather than by re-deriving what
// the hardware intrinsic does.
extern "C" __global__ void cast_add_bf16(
    const __nv_bfloat16 *base,
    const float *f32val,
    __nv_bfloat16 *out,
    const size_t n
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        __nv_bfloat16 delta = __float2bfloat16(f32val[i]);
        out[i] = base[i] + delta;
    }
}
