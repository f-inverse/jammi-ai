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
// 1044-1069, `v0.20.0`, re-read at source on pod a100e 2026-08-26): torch's
// `+` promotes a bf16 `result` to the delta's `f32` dtype (no rounding lost
// on `result`'s side), adds in f32, and only THEN casts back down once via
// `.to(torch_result_dtype)` — ONE round point, not two. An earlier revision
// of this kernel rounded the scaled delta to bf16 FIRST (an extra round
// point never present in the PEFT reference); this crate's
// "f32-accumulate, round once" convention (see `../ops/mod.rs` and the
// per-op f16 reference-regime table in
// `docs/maintainer/cuda-kernel-guide.md`) is the one this op now follows
// too, not an exception to it.
//
// esc-046 audit round 2 (finding 2): PEFT computes the scaled delta
// (`lora_B(...) * scaling`) as its OWN separate kernel launch (one f32
// rounding, stored), then `result + delta` as a SECOND separate launch
// (another f32 rounding) — two separately-rounded f32 operations, never
// fused. `build.rs`'s own PINNED FLAGS comment states nvcc's `--fmad=true`
// default (on regardless of `-use_fast_math`, never globally disabled
// here) may contract an expression shaped like `base + lora * scaling`
// into a single hardware `fma.rn.f32` — ONE rounding of the true product
// before the add, not two — and prescribes exactly the fix for a kernel
// that needs bit-exact parity on one specific expression: explicitly
// rounded intrinsics (`__fmul_rn` / `__fadd_rn`), not a global
// `--fmad=false`. An earlier revision of this file used plain `+`/`*`
// operators here; nvcc 12.6.85 (`--ptx -arch=compute_80`, this crate's
// pinned target) contracted `bv + lora[i] * scaling` into `fma.rn.f32` in
// the bf16_f32 kernel specifically (confirmed by compiling this file to
// PTX and reading the emitted SASS-adjacent PTX directly) — measured to
// diverge from the separately-rounded (PEFT-faithful) reference on
// 1/131072 elements (no-producer: the audit's one-off probe, uncommitted)
// at a NON-DYADIC scaling (`sqrt(3)`; every dyadic scaling tested,
// including this op's own committed fixtures, is 0/131072 (no-producer:
// derived, not measured — a power-of-two scaling factor makes the product
// exact in f32 regardless of fusion), so a dyadic scaling has ZERO power
// to detect this class). All
// four kernels below share the identical `base + lora * scaling` shape
// (the f32-output kernels included — torch's f32 add-of-a-scaled-delta
// path is ALSO mul-round-then-add-round, two separate launches, never
// fused), so all four are pinned here with `__fmul_rn`/`__fadd_rn`, not
// just the one the audit's probe happened to compile.
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
        out[i] = __fadd_rn(base[i], __fmul_rn(lora[i], scaling));
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
        out[i] = __fadd_rn(base[i], __fmul_rn(lv, scaling));
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
        // intermediate bf16-rounded `delta`. `__fmul_rn`/`__fadd_rn`
        // (round 2 fix) keep the multiply and the add as two SEPARATELY
        // rounded f32 operations, matching PEFT's own two-launch
        // execution instead of letting nvcc fuse them into one `fma.rn.f32`.
        float bv = __bfloat162float(base[i]);
        out[i] = __float2bfloat16(__fadd_rn(bv, __fmul_rn(lora[i], scaling)));
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
        out[i] = __float2bfloat16(__fadd_rn(bv, __fmul_rn(lv, scaling)));
    }
}
