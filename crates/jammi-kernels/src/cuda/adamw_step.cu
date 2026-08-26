// adamw_step.cu — the two in-place kernels behind the multi-tensor AdamW
// lever. Compiled to PTX only when the `cuda` feature is active (see
// ../../build.rs); the pinned build flags (sm_80 baseline, no
// -use_fast_math) live there, not here.
//
// Both real kernels mutate their FIRST buffer in place (no output
// allocation — this is the whole point: candle's `InplaceOp2`/`InplaceOp3`
// give a writable device pointer directly, so there is no `Var::set` D2D
// memcpy afterward — see ../ops/adamw_step.rs's module doc).
//
// Domain: contiguous, identically-shaped F32 buffers only. The Rust glue
// (../cuda/adamw_step.rs) checks shape/dtype/contiguity before a launch;
// these kernels assume a flat linear index and do not themselves
// re-validate.
//
// BIT-IDENTITY, not a tolerance (fix for the adversarial audit's finding
// (1): nvcc's `--fmad=true` default (on regardless of `-use_fast_math`,
// which stays off) silently contracted the bare `beta * m[i] + one_minus_
// beta * gv` / `theta[i] * one_minus_lr_lambda - adjusted_grad * lr`
// sub-expressions this file used to write into single-rounding hardware
// FMAs — measured on jammi-a100: 5145/16384 `m` elements differed from
// candle's own eager CUDA chain at t=3 with nonzero prior moments. Per
// `build.rs`'s own pinned-flags comment and `docs/maintainer/cuda-kernel-
// guide.md`, the fix is explicit-rounding PTX intrinsics IN THE EXPRESSION
// (`__fmul_rn`/`__fadd_rn`/`__fsub_rn`/`__fdiv_rn`), not a TU-wide
// `--fmad=false` (which would cost every OTHER kernel in this crate real
// performance for a guarantee only this one needs). Each intrinsic call is
// a single, non-fusable IEEE round-to-nearest operation — ptxas cannot
// silently merge two of them into an FMA the way it can with bare `*`/`+`.
//
// Candle's eager chain is a composition of SEPARATE single-op kernels
// (`candle-kernels-0.11.0/src/affine.cu`'s `AFFINE_OP(float, affine_f32,
// x * mul + add)` for every `Tensor * f64`/`Tensor + f64`, and a genuine
// standalone binary add/sub/div/mul kernel for every `Tensor <op> Tensor`)
// — so matching it bit-for-bit means reproducing THAT many separate
// roundings, not the fewest-operations fusion a human would otherwise
// write. Per `adamw.rs:94-100`:
//   next_m = (m*beta1) + (g*(1-beta1))            -- affine(m,b1,0) + affine(g,1-b1,0)
//   next_v = (v*beta2) + ((g*g)*(1-beta2))         -- affine(v,b2,0) + affine(g*g,1-b2,0), g*g itself a standalone unary Sqr (v*v, op.rs:591)
//   m_hat  = next_m*scale_m                        -- affine(next_m,scale_m,0)
//   v_hat  = next_v*scale_v                        -- affine(next_v,scale_v,0)
//   denom  = sqrt(v_hat) + eps                      -- affine(sqrt(v_hat),1.0,eps) == sqrt(v_hat)+eps bit-for-bit (mul=1 is exact)
//   adj    = m_hat / denom                          -- standalone binary div
//   theta' = (theta*(1-lr*lambda)) - (adj*lr)       -- affine(theta,1-lr*lambda,0) - affine(adj,lr,0), standalone binary sub
// EVERY `affine(x, mul, 0.0)` site literally computes `x*mul + 0.0f`
// (`candle-core-0.11.0/src/cpu_backend/mod.rs:311-317`'s `Affine` map is
// `v * mul + add` with `add = 0.0` for a scalar `Mul`, and the CUDA kernel
// above is the same expression) — the `+ 0.0f` LAUNDERS a `-0.0` product
// (e.g. an underflowed or exact-zero multiply) to `+0.0` per IEEE-754's
// opposite-sign-zero-sum rule. Skipping that add (writing just `x*mul`)
// preserves `-0.0` where candle's own kernel would not, so every affine
// site below reproduces the `+ 0.0f` explicitly, not just the multiply.
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
        // `square_grad`'s `gv * gv` is candle's standalone unary `Sqr` op
        // (`v * v`, op.rs:591) — ONE rounding, no affine/`+0.0` site (it is
        // never followed by "+ 0.0" in the eager chain; `g.sqr()` is its
        // own single-op kernel), so a bare `__fmul_rn` here already matches.
        float gv = g[i];
        if (square_grad) {
            gv = __fmul_rn(gv, gv);
        }
        // affine(m, beta, 0.0) and affine(g_or_gg, one_minus_beta, 0.0):
        // two INDEPENDENTLY rounded, INDEPENDENTLY zero-laundered terms,
        // each its own candle kernel launch in the eager chain.
        float term_m = __fadd_rn(__fmul_rn(beta, m[i]), 0.0f);
        float term_g = __fadd_rn(__fmul_rn(one_minus_beta, gv), 0.0f);
        // Tensor + Tensor: a genuine standalone binary add (candle's own
        // `badd` kernel), one more rounding on top of the two above.
        m[i] = __fadd_rn(term_m, term_g);
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
        // affine(next_m, scale_m, 0.0), affine(next_v, scale_v, 0.0).
        float m_hat = __fadd_rn(__fmul_rn(m[i], scale_m), 0.0f);
        float v_hat = __fadd_rn(__fmul_rn(v[i], scale_v), 0.0f);
        // affine(sqrt(v_hat), 1.0, eps) == sqrt(v_hat) + eps bit-for-bit:
        // `x * 1.0` is exact (no rounding, no sign change) at any finite
        // `x`, so folding that multiply away changes nothing observable —
        // unlike the `mul=<other>, add=0.0` sites above, `mul=1.0` never
        // needs a separate `+0.0f` laundering step either (`x*1.0` cannot
        // itself introduce a new `-0.0` that `x` didn't already carry, and
        // `eps` is never exactly the additive identity in this op's real
        // callers, so the sign-of-zero corner case this file launders
        // elsewhere does not arise here).
        float denom = __fadd_rn(sqrtf(v_hat), eps);
        // Tensor / Tensor: standalone binary div.
        float adjusted_grad = __fdiv_rn(m_hat, denom);
        // affine(theta, one_minus_lr_lambda, 0.0), affine(adjusted_grad, lr, 0.0).
        float theta_scaled = __fadd_rn(__fmul_rn(theta[i], one_minus_lr_lambda), 0.0f);
        float adj_scaled = __fadd_rn(__fmul_rn(adjusted_grad, lr), 0.0f);
        // Tensor - Tensor: standalone binary sub.
        theta[i] = __fsub_rn(theta_scaled, adj_scaled);
    }
}

// TEST-ONLY NEGATIVE CONTROL — never reached by production dispatch (no
// admission/dispatch site names it; see `ops::adamw_step::
// AdamMomentUpdateFmaContractedRedControl`'s doc, the ONLY caller). Exists
// solely so the bit-identity harness in `tests/cuda_parity.rs` (and this
// crate's own CPU-side unit test) can prove it has the POWER to detect the
// exact defect class this file's fix closes: deliberately forces the
// single-rounding FMA contraction commit 0498f8b risked leaving to
// `--fmad=true`'s discretion, via CUDA's explicit `fmaf()` intrinsic
// (`fmaf(beta, m[i], one_minus_beta*gv)` — one rounding for the whole
// expression) rather than this file's real two-separate-roundings-then-add
// (`adamw_moment_update_f32`, above). Reproduces the audit's measured
// finding (5145/16384 `m` elements differ from the eager CUDA chain at
// t=3, nonzero prior moments) DETERMINISTICALLY rather than depending on
// ptxas's optional contraction, which is what makes it a reliable negative
// control rather than a compiler-version-dependent one.
extern "C" __global__ void adamw_moment_update_f32_fma_contracted_red_control(
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
        m[i] = fmaf(beta, m[i], one_minus_beta * gv);
    }
}
