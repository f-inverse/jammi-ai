// geglu.cu — fused GeGLU (gated GELU) forward + backward.
// Compiled to PTX only when the `cuda` feature is active (see
// ../../build.rs); the pinned build flags (sm_80 baseline, no
// -use_fast_math) live there, not here.
//
// Domain: `wi_out` is `[<leading axes>, 2*intermediate]`, row-major
// contiguous (a possibly narrowed-but-contiguous nonzero-offset view is
// handled by the Rust glue slicing to `contiguous_offsets()` before the
// launch — this file assumes a flat linear index and does not itself
// re-validate contiguity or the even-last-dim domain, both enforced by
// `../ops/geglu.rs`/`../cuda/geglu.rs` before any launch).
//
// Purely ELEMENTWISE (no per-row reduction, unlike layer_norm.cu/
// softmax.cu): a grid-stride loop over `n_out = rows*intermediate`
// (the HALF-width output element count). For output element `idx`:
//   row = idx / intermediate; col = idx % intermediate
//   gate = wi_out[row*2*intermediate + col]
//   up   = wi_out[row*2*intermediate + intermediate + col]
//
// INDEXING CONTRACT (campaign #446, finding 4 — see
// `../ops/launch_domain.rs`'s module doc for the whole rule and the CPU
// lane that enforces it): `idx`, `row` and the stride are `size_t`, while
// `intermediate`/`n_out` stay 32-bit `unsigned int` PARAMETERS. This
// asymmetry is deliberate. These loops used to declare `unsigned int idx`;
// the stride is `GEGLU_BLOCK * GEGLU_MAX_GRID == 16'776'960`, so a 32-bit
// lane could only ever hold values congruent to its own start modulo
// `gcd(stride, 2^32) == 256`. Above `n_out == UINT_MAX - 255` the exit
// window `[n_out, 2^32)` is narrower than 256 and cannot contain one
// value of every residue class, so every lane whose class is missing
// spins FOREVER (at `n_out == UINT_MAX`, 255 of every 256 threads); just
// below that window the loop escapes only after re-walking its whole
// orbit (65'537 visits where 257 was the job). A 64-bit induction
// variable makes both impossible for EVERY `n_out` the host admits. The parameters stay 32-bit because the Rust
// glue pushes them by value (`builder.arg(&n_out_u32)`): widening one
// side without the other would make this kernel read 8 bytes where 4 were
// pushed. `cuda/geglu.rs`'s `check_elem_count_fits_u32` is the single
// place that bounds `n_out` to what a 32-bit parameter can carry.
//
// `kAlpha`/`kBeta` mirror ATen's `ActivationGeluKernel.cu` erf-mode
// `gelu_backward` constants exactly (see `../ops/geglu.rs`'s module doc):
// `kAlpha = 1/sqrt(2)`, `kBeta = (2/sqrt(pi))*(1/sqrt(2))*0.5 ==
// 1/sqrt(2*pi)`. `erff`/`expf` (no fast-math) match the CPU arm's
// `libm::erff`-based formula to within ordinary cross-platform libm ULP
// differences — this build's own tolerance doctrine (`build.rs`), not a
// bit-exact claim.
#include <cuda_bf16.h>
#include <cstddef>

#define GEGLU_KALPHA 0.70710678118654752440f
#define GEGLU_KBETA 0.3989422804014327f

__device__ __forceinline__ float gelu_erf_cdf(float x) {
    return (erff(x * GEGLU_KALPHA) + 1.0f) * 0.5f;
}

__device__ __forceinline__ float gelu_erf_pdf(float x) {
    return GEGLU_KBETA * expf(-0.5f * x * x);
}

// ---------------------------------------------------------------------
// Forward: out = gelu_erf(gate) * up.
// ---------------------------------------------------------------------

extern "C" __global__ void geglu_fwd_f32(
    const float* wi_out, float* out,
    const unsigned int intermediate, const unsigned int n_out
) {
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n_out;
         idx += (size_t)blockDim.x * gridDim.x) {
        size_t row = idx / intermediate;
        unsigned int col = (unsigned int)(idx % intermediate);
        size_t base = row * 2 * (size_t)intermediate;
        float gate = wi_out[base + col];
        float up = wi_out[base + intermediate + col];
        out[idx] = gate * gelu_erf_cdf(gate) * up;
    }
}

// BF16: rounds the activation to bf16 immediately (ROUND 1, matching the
// upstream two-op reference's own materialize-then-multiply ordering —
// see `../ops/geglu.rs`'s module doc), then multiplies in f32 and rounds
// ONCE more on the way out (ROUND 2, matching `half::bf16`'s own `Mul`
// semantics).
extern "C" __global__ void geglu_fwd_bf16(
    const __nv_bfloat16* wi_out, __nv_bfloat16* out,
    const unsigned int intermediate, const unsigned int n_out
) {
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n_out;
         idx += (size_t)blockDim.x * gridDim.x) {
        size_t row = idx / intermediate;
        unsigned int col = (unsigned int)(idx % intermediate);
        size_t base = row * 2 * (size_t)intermediate;
        float gate = __bfloat162float(wi_out[base + col]);
        float up = __bfloat162float(wi_out[base + intermediate + col]);
        float act_f32 = gate * gelu_erf_cdf(gate);
        __nv_bfloat16 act_bf16 = __float2bfloat16(act_f32); // ROUND 1
        float out_f32 = __bfloat162float(act_bf16) * up;
        out[idx] = __float2bfloat16(out_f32); // ROUND 2
    }
}

// ---------------------------------------------------------------------
// Backward: writes BOTH halves of dwi_out in one launch.
//   d_gate = dy * up * gelu_erf'(gate) = dy * up * (cdf(gate) + gate*pdf(gate))
//   d_up   = dy * gelu_erf(gate)       = dy * gate * cdf(gate)
// ---------------------------------------------------------------------

extern "C" __global__ void geglu_bwd_dwi_out_f32(
    const float* wi_out, const float* dy, float* dwi_out,
    const unsigned int intermediate, const unsigned int n_out
) {
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n_out;
         idx += (size_t)blockDim.x * gridDim.x) {
        size_t row = idx / intermediate;
        unsigned int col = (unsigned int)(idx % intermediate);
        size_t base = row * 2 * (size_t)intermediate;
        float gate = wi_out[base + col];
        float up = wi_out[base + intermediate + col];
        float dyi = dy[idx];
        float cdf = gelu_erf_cdf(gate);
        float pdf = gelu_erf_pdf(gate);
        float gelu_val = gate * cdf;
        float gelu_deriv = cdf + gate * pdf;
        dwi_out[base + col] = dyi * up * gelu_deriv;
        dwi_out[base + intermediate + col] = dyi * gelu_val;
    }
}

// BF16: f32-accumulate throughout, rounding EACH of d_gate/d_up to bf16
// exactly once at the very end — this crate's usual bf16-backward
// convention (see `../ops/geglu.rs`'s module doc for why this
// deliberately does NOT mirror eager's own multi-op rounding cascade).
extern "C" __global__ void geglu_bwd_dwi_out_bf16(
    const __nv_bfloat16* wi_out, const __nv_bfloat16* dy, __nv_bfloat16* dwi_out,
    const unsigned int intermediate, const unsigned int n_out
) {
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n_out;
         idx += (size_t)blockDim.x * gridDim.x) {
        size_t row = idx / intermediate;
        unsigned int col = (unsigned int)(idx % intermediate);
        size_t base = row * 2 * (size_t)intermediate;
        float gate = __bfloat162float(wi_out[base + col]);
        float up = __bfloat162float(wi_out[base + intermediate + col]);
        float dyi = __bfloat162float(dy[idx]);
        float cdf = gelu_erf_cdf(gate);
        float pdf = gelu_erf_pdf(gate);
        float gelu_val = gate * cdf;
        float gelu_deriv = cdf + gate * pdf;
        dwi_out[base + col] = __float2bfloat16(dyi * up * gelu_deriv);
        dwi_out[base + intermediate + col] = __float2bfloat16(dyi * gelu_val);
    }
}
