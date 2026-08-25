// adamw_step.cu — the two in-place kernels behind the multi-tensor AdamW
// lever. Compiled to PTX only when the `cuda` feature is active (see
// ../../build.rs); the pinned build flags (sm_80 baseline, no
// -use_fast_math) live there, not here.
//
// Both kernels mutate their FIRST buffer in place (no output allocation —
// this is the whole point: candle's `InplaceOp2`/`InplaceOp3` give a
// writable device pointer directly, so there is no `Var::set` D2D memcpy
// afterward — see ../ops/adamw_step.rs's module doc).
//
// Domain: contiguous, identically-shaped F32 buffers only. The Rust glue
// (../cuda/adamw_step.rs) checks shape/dtype/contiguity before a launch;
// these kernels assume a flat linear index and do not themselves
// re-validate.
//
// Rounding: plain arithmetic (`*`, `+`, `-`, `/`, `sqrtf`), matching the CPU
// arm's operation order exactly. nvcc's `--fmad=true` default (on
// regardless of `-use_fast_math`, which stays off) may still contract an
// `a*b+c`-shaped sub-expression into a single-rounding FMA — this crate's
// CPU arm is bit-exact against the eager chain, but the CUDA arm is a
// TOLERANCE claim, not bit-exact, exactly like every other fused op in this
// crate (see `tests/cuda_parity.rs`'s `F32_TOL`) — see
// `../ops/adamw_step.rs`'s module doc and the design study
// (`scratchpad/design-multi-tensor-adamw.md`) for why forcing
// `--fmad=false` here would not by itself make this bit-identical to
// candle's own (separately compiled) eager CUDA kernels.
#include <cstddef>

extern "C" __global__ void adamw_moment_update_f32(
    float *m,
    const float *g,
    const float beta,
    const float one_minus_beta,
    const int square_grad,
    const size_t n
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float gv = g[i];
        if (square_grad) {
            gv = gv * gv;
        }
        m[i] = beta * m[i] + one_minus_beta * gv;
    }
}

extern "C" __global__ void adamw_theta_update_f32(
    float *theta,
    const float *m,
    const float *v,
    const float one_minus_lr_lambda,
    const float scale_m,
    const float scale_v,
    const float eps,
    const float lr,
    const size_t n
) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float m_hat = m[i] * scale_m;
        float v_hat = v[i] * scale_v;
        float denom = sqrtf(v_hat) + eps;
        float adjusted_grad = m_hat / denom;
        theta[i] = theta[i] * one_minus_lr_lambda - adjusted_grad * lr;
    }
}
