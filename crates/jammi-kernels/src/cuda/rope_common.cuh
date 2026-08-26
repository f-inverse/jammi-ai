// rope_common.cuh — the ONE per-element rotate-half expression shared by
// `rope.cu` (period-modulo indexing, `RopeFused`) and `rope_positions.cu`
// (packed-`[total,3,h,d]`-buffer indexing with the SAME math, `crate::
// ops::RopePositionsFused` — P6 Stage B B3-dense). `#include`d by both
// `.cu` translation units (each compiled to its own PTX module by
// `bindgen_cuda`'s `src/cuda/*.cu` glob — there is no device-code linking
// across separate PTX modules, so a shared `__device__` function must
// live in a header both files `#include`, not in either `.cu` file
// itself).
//
// value, partner, c, s, sign -> the SAME 2x2 rotation
// `RopeFused`'s own module doc derives (`../ops/rope.rs`): forward is
// `out = x*cos + rotate_half(x)*sin`; backward reuses this identical
// expression with `sign` negated (no permutation of the upstream
// gradient needed) — see that doc for the full derivation. `partner` is
// the caller's own `rotate_half(x)` value at this position (already
// negated for the `col < half` half by the caller, per `rope.cu`'s
// existing convention) — this function does not re-derive `partner`,
// only the final blend, so both kernels' indexing math (period-modulo vs
// packed-buffer-decode) stays entirely in the caller.
#ifndef JAMMI_ROPE_COMMON_CUH
#define JAMMI_ROPE_COMMON_CUH

__device__ __forceinline__ float rope_rotate(
    const float value, const float partner, const float c, const float s, const float sign
) {
    return value * c + partner * s * sign;
}

#endif // JAMMI_ROPE_COMMON_CUH
