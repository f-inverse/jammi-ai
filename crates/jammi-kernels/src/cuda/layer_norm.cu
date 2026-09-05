// layer_norm.cu — fused, bias-free LayerNorm forward + backward. Compiled
// to PTX only when the `cuda` feature is active (see ../../build.rs); the
// pinned build flags (sm_80 baseline, no -use_fast_math) live there, not
// here.
//
// Domain: `x` is `[rows, hidden]`, row-major contiguous (a possibly
// narrowed-but-contiguous nonzero-offset view is handled by the Rust glue
// slicing to `contiguous_offsets()` before the launch — this file assumes
// a flat `row*hidden + col` index and does not itself re-validate
// contiguity); `gamma` is `[hidden]` contiguous. `hidden <= 8192` and
// `rows*hidden <= u32::MAX` are enforced by the Rust glue
// (../layer_norm.rs) before any launch.
//
// ONE THREAD BLOCK PER ROW for forward and backward-dx. Every per-row
// reduction (mean, variance, and both backward-dx reduction scalars) is a
// block-wide shared-memory tree reduction over `LN_BLOCK` threads — the
// scratch is O(blockDim.x) floats (a few hundred bytes, not KB), NOT
// O(hidden): `MAX_HIDDEN` (../ops/layer_norm.rs) is therefore a
// conservative validated ceiling, not a hardware limit this shared-memory
// footprint would impose — a grid-stride loop over `hidden` within each
// per-row pass has no correctness ceiling on `hidden` at all. `dx`'s
// backward is ONE kernel launch: recompute mean/invvar from `x`, the two
// per-row scalars (Apex/ATen canonical), and `dx` itself, all in the same
// launch — a two-phase bwd (recompute in one launch, dx in another) would
// double LN backward launches post-fusion.
//
// `dgamma` (only launched when the call site says it needs it — see
// `LayerNormFused::dgamma_needed`'s "construction data" doc; reachable
// whenever gamma is a genuine trainable `Var`, which the CPU/GPU parity
// this file backs must hold for too) is TWO kernels, not one, to stay
// O(rows*hidden) total rather than O(rows*hidden^2):
//   1. `layer_norm_row_stats_{f32,bf16}` — ONE block per row (the exact
//      same block-wide reduction as forward/dx above), writing that row's
//      `mean`/`invvar` ONCE into small `[rows]`-length buffers.
//   2. `layer_norm_bwd_dgamma_{f32,bf16}` — column-tiled: each thread
//      owns one or more hidden columns (grid-stride across
//      `gridDim.x * blockDim.x`) and walks every row accumulating
//      `dy[row,col] * xhat[row,col]`, reading the CACHED `mean[row]` /
//      `invvar[row]` from pass 1 (O(1) per element) instead of
//      recomputing a full-row reduction per column per row (the
//      O(rows*hidden^2) shape this design replaces). `atomicAdd` is not
//      used here — dgamma columns are partitioned across threads
//      (grid-stride), never shared, so no thread ever writes another
//      thread's output element.
#include <cuda_bf16.h>
#include <cstddef>

#define LN_BLOCK 256

__device__ __forceinline__ float block_reduce_sum(float val, float* scratch) {
    int tid = threadIdx.x;
    scratch[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    float total = scratch[0];
    __syncthreads();
    return total;
}

// ---------------------------------------------------------------------
// Forward: y = ((x - mean) * invvar) * gamma. f32 accumulation
// throughout; one rounding to the output dtype.
// ---------------------------------------------------------------------

extern "C" __global__ void layer_norm_fwd_f32(
    const float* x, const float* gamma, float* y,
    const unsigned int hidden, const float eps
) {
    __shared__ float scratch[LN_BLOCK];
    size_t row = blockIdx.x;
    const float* xr = x + row * (size_t)hidden;
    float* yr = y + row * (size_t)hidden;

    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) sum += xr[i];
    sum = block_reduce_sum(sum, scratch);
    float mean = sum / (float)hidden;

    float sumsq = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float d = xr[i] - mean;
        sumsq += d * d;
    }
    sumsq = block_reduce_sum(sumsq, scratch);
    float invvar = rsqrtf(sumsq / (float)hidden + eps);

    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float xhat = (xr[i] - mean) * invvar;
        yr[i] = xhat * gamma[i];
    }
}

extern "C" __global__ void layer_norm_fwd_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* gamma, __nv_bfloat16* y,
    const unsigned int hidden, const float eps
) {
    __shared__ float scratch[LN_BLOCK];
    size_t row = blockIdx.x;
    const __nv_bfloat16* xr = x + row * (size_t)hidden;
    __nv_bfloat16* yr = y + row * (size_t)hidden;

    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        sum += __bfloat162float(xr[i]);
    }
    sum = block_reduce_sum(sum, scratch);
    float mean = sum / (float)hidden;

    float sumsq = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float d = __bfloat162float(xr[i]) - mean;
        sumsq += d * d;
    }
    sumsq = block_reduce_sum(sumsq, scratch);
    float invvar = rsqrtf(sumsq / (float)hidden + eps);

    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float xhat = (__bfloat162float(xr[i]) - mean) * invvar;
        yr[i] = __float2bfloat16(xhat * __bfloat162float(gamma[i]));
    }
}

// ---------------------------------------------------------------------
// Backward dx: recompute mean/invvar, then the two per-row scalars, then
// dx — one kernel launch, three grid-stride passes over the row.
// ---------------------------------------------------------------------

extern "C" __global__ void layer_norm_bwd_dx_f32(
    const float* x, const float* gamma, const float* dy, float* dx,
    const unsigned int hidden, const float eps
) {
    __shared__ float scratch[LN_BLOCK];
    size_t row = blockIdx.x;
    const float* xr = x + row * (size_t)hidden;
    const float* dyr = dy + row * (size_t)hidden;
    float* dxr = dx + row * (size_t)hidden;

    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) sum += xr[i];
    sum = block_reduce_sum(sum, scratch);
    float mean = sum / (float)hidden;

    float sumsq = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float d = xr[i] - mean;
        sumsq += d * d;
    }
    sumsq = block_reduce_sum(sumsq, scratch);
    float invvar = rsqrtf(sumsq / (float)hidden + eps);

    float sum_t = 0.0f;
    float sum_t_xhat = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float xhat = (xr[i] - mean) * invvar;
        float t = dyr[i] * gamma[i];
        sum_t += t;
        sum_t_xhat += t * xhat;
    }
    sum_t = block_reduce_sum(sum_t, scratch);
    sum_t_xhat = block_reduce_sum(sum_t_xhat, scratch);
    float mean_t = sum_t / (float)hidden;
    float mean_t_xhat = sum_t_xhat / (float)hidden;

    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float xhat = (xr[i] - mean) * invvar;
        float t = dyr[i] * gamma[i];
        dxr[i] = (t - mean_t - xhat * mean_t_xhat) * invvar;
    }
}

extern "C" __global__ void layer_norm_bwd_dx_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* gamma, const __nv_bfloat16* dy,
    __nv_bfloat16* dx, const unsigned int hidden, const float eps
) {
    __shared__ float scratch[LN_BLOCK];
    size_t row = blockIdx.x;
    const __nv_bfloat16* xr = x + row * (size_t)hidden;
    const __nv_bfloat16* dyr = dy + row * (size_t)hidden;
    __nv_bfloat16* dxr = dx + row * (size_t)hidden;

    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        sum += __bfloat162float(xr[i]);
    }
    sum = block_reduce_sum(sum, scratch);
    float mean = sum / (float)hidden;

    float sumsq = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float d = __bfloat162float(xr[i]) - mean;
        sumsq += d * d;
    }
    sumsq = block_reduce_sum(sumsq, scratch);
    float invvar = rsqrtf(sumsq / (float)hidden + eps);

    float sum_t = 0.0f;
    float sum_t_xhat = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float xhat = (__bfloat162float(xr[i]) - mean) * invvar;
        float t = __bfloat162float(dyr[i]) * __bfloat162float(gamma[i]);
        sum_t += t;
        sum_t_xhat += t * xhat;
    }
    sum_t = block_reduce_sum(sum_t, scratch);
    sum_t_xhat = block_reduce_sum(sum_t_xhat, scratch);
    float mean_t = sum_t / (float)hidden;
    float mean_t_xhat = sum_t_xhat / (float)hidden;

    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float xhat = (__bfloat162float(xr[i]) - mean) * invvar;
        float t = __bfloat162float(dyr[i]) * __bfloat162float(gamma[i]);
        dxr[i] = __float2bfloat16((t - mean_t - xhat * mean_t_xhat) * invvar);
    }
}

// ---------------------------------------------------------------------
// Backward dgamma, pass 1: per-row mean/invvar, cached once per row.
// Identical block-wide reduction shape to forward/dx above — ONE block
// per row, writing `mean_out[row]` / `invvar_out[row]` exactly once
// (only `threadIdx.x == 0` writes, after every thread's contribution to
// the block reduction has already landed in `scratch[0]`).
// ---------------------------------------------------------------------

extern "C" __global__ void layer_norm_row_stats_f32(
    const float* x, float* mean_out, float* invvar_out,
    const unsigned int hidden, const float eps
) {
    __shared__ float scratch[LN_BLOCK];
    size_t row = blockIdx.x;
    const float* xr = x + row * (size_t)hidden;

    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) sum += xr[i];
    sum = block_reduce_sum(sum, scratch);
    float mean = sum / (float)hidden;

    float sumsq = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float d = xr[i] - mean;
        sumsq += d * d;
    }
    sumsq = block_reduce_sum(sumsq, scratch);
    float invvar = rsqrtf(sumsq / (float)hidden + eps);

    if (threadIdx.x == 0) {
        mean_out[row] = mean;
        invvar_out[row] = invvar;
    }
}

extern "C" __global__ void layer_norm_row_stats_bf16(
    const __nv_bfloat16* x, float* mean_out, float* invvar_out,
    const unsigned int hidden, const float eps
) {
    __shared__ float scratch[LN_BLOCK];
    size_t row = blockIdx.x;
    const __nv_bfloat16* xr = x + row * (size_t)hidden;

    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        sum += __bfloat162float(xr[i]);
    }
    sum = block_reduce_sum(sum, scratch);
    float mean = sum / (float)hidden;

    float sumsq = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float d = __bfloat162float(xr[i]) - mean;
        sumsq += d * d;
    }
    sumsq = block_reduce_sum(sumsq, scratch);
    float invvar = rsqrtf(sumsq / (float)hidden + eps);

    if (threadIdx.x == 0) {
        mean_out[row] = mean;
        invvar_out[row] = invvar;
    }
}

// ---------------------------------------------------------------------
// Backward dgamma, pass 2: column-tiled accumulation across rows, reading
// pass 1's cached per-row `mean`/`invvar` (O(1) per element visited —
// this is what keeps the whole dgamma path O(rows*hidden) rather than
// O(rows*hidden^2), unlike a design that recomputed each row's stats
// once per column visiting it). `dgamma_col = sum_rows(dy[row,col] *
// xhat[row,col])`. Each thread owns one or more columns (grid-stride);
// `atomicAdd` is not used — columns are partitioned across threads, never
// shared, so no thread ever writes another thread's output element.
//
// `col` is `size_t`, per the crate-wide INDEXING CONTRACT for grid-stride
// loops (campaign #446 finding 4; stated in full in `geglu.cu`'s module
// doc and `../ops/launch_domain.rs`'s). These three dgamma loops were
// never REACHABLY vulnerable — `hidden` is bounded by `ops::MAX_HIDDEN`
// (8192) at the host edge, far below any 32-bit wrap — but the contract is
// a property of the loop SHAPE, not of one op's ceiling: a lexical scan
// (`launch_domain::tests::every_grid_stride_loop_in_a_cuda_source_is_64_bit`)
// cannot distinguish "bounded elsewhere" from "unbounded", so every
// grid-stride loop in this directory is 64-bit and the rule stays
// mechanically checkable.
// ---------------------------------------------------------------------

extern "C" __global__ void layer_norm_bwd_dgamma_f32(
    const float* x, const float* dy, const float* mean, const float* invvar,
    float* dgamma, const unsigned int rows, const unsigned int hidden
) {
    for (size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x; col < hidden;
         col += (size_t)blockDim.x * gridDim.x) {
        float acc = 0.0f;
        for (unsigned int r = 0; r < rows; r++) {
            float xhat = (x[(size_t)r * hidden + col] - mean[r]) * invvar[r];
            acc += dy[(size_t)r * hidden + col] * xhat;
        }
        dgamma[col] = acc;
    }
}

// Same computation, bf16 I/O with f32 math; accumulates into an F32
// SCRATCH buffer (never bf16 directly — see ../layer_norm.rs's glue for
// why: this keeps the accumulation in the same f32-throughout convention
// as every other bf16 arm in this crate, then the glue rounds once, at
// the very end, to bf16).
extern "C" __global__ void layer_norm_bwd_dgamma_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* dy, const float* mean,
    const float* invvar, float* dgamma_f32, const unsigned int rows,
    const unsigned int hidden
) {
    for (size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x; col < hidden;
         col += (size_t)blockDim.x * gridDim.x) {
        float acc = 0.0f;
        for (unsigned int r = 0; r < rows; r++) {
            float xhat =
                (__bfloat162float(x[(size_t)r * hidden + col]) - mean[r]) * invvar[r];
            acc += __bfloat162float(dy[(size_t)r * hidden + col]) * xhat;
        }
        dgamma_f32[col] = acc;
    }
}

// Elementwise F32 -> BF16 rounding, used only to finish the bf16 dgamma
// path above (one rounding, at the very end, off the reduction's own
// accumulation).
extern "C" __global__ void layer_norm_cast_f32_to_bf16(
    const float* src, __nv_bfloat16* dst, const unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2bfloat16(src[i]);
    }
}

// ---------------------------------------------------------------------
// #460 (C-LN): bias-carrying forward, F32/BF16. APPEND-ONLY from here —
// every kernel ABOVE this comment is byte-for-byte unchanged by this
// addition (a `git diff` restricted to the lines above this block, at the
// #460 unit's tip vs its base, is empty); the bias-free symbols
// (`layer_norm_fwd_f32`/`layer_norm_fwd_bf16`) are therefore bit-identical
// by construction, not merely "not intended to change" — nothing below
// this line is reachable from them.
//
// `y = ((x - mean) * invvar) * gamma + beta`, matching ATen's
// `vectorized_layer_norm_kernel_impl`/`LayerNormForwardCUDAKernel` (torch
// v2.8.0 `aten/src/ATen/native/cuda/layer_norm_kernel.cu:93-112`, pinned by
// `jammi_encoders::layer_norm::LayerNorm::slow`'s own doc citation):
// "Computation is performed in T_ACC ... result is implicitly cast to T" —
// gamma AND beta both applied in the f32 accumulator, ONE cast to the
// output dtype at the very end. `--fmad=true` is on for this crate's build
// (see `../../build.rs`), so this is written as the literal
// `xhat * gamma[i] + beta[i]` expression (an FMA-eligible form), never a
// `+ 0.0f` sentinel that would defeat fusion into the mul.
//
// Each per-dtype row body below is its OWN `template <bool HAS_BETA>`
// `__device__ __forceinline__` definition, specialised at COMPILE TIME
// (never a runtime null-pointer branch) — NOT shared with the pre-existing
// bias-free row body above this comment block. Because the bias-free
// kernels above are byte-untouched (this block's own opening claim), their
// mean/var reduction is a SEPARATE, textually duplicated copy from this
// template's `HAS_BETA = false` arithmetic path — an accepted drift
// surface: a future change to one row-math body (e.g. a different
// reduction order) will NOT automatically propagate to the other, and
// nothing here enforces they stay in sync beyond this file's own review
// and the CPU<->CUDA parity suite (`cuda_parity.rs`) exercising both. This
// duplication is the direct, deliberate cost of the bit-identity-by-
// construction guarantee this block's opening comment makes: keeping the
// pre-existing kernel bytes untouched (provable via `git diff`) requires
// NOT refactoring it into a shared template the new kernel also
// instantiates. Only the `HAS_BETA = true` instantiation of this NEW
// template is ever emitted as a kernel below — `LayerNormBiasedFused`
// (`../ops/layer_norm.rs`) is a `CustomOp3` with a REQUIRED (non-nullable)
// `beta` tensor, so a `HAS_BETA = false` instantiation has no caller
// today; the template stays generic (not hand-monomorphised to `true`) so
// a future nullable-beta caller costs no kernel-body rewrite, only a new
// `extern "C"` wrapper — it would NOT, by itself, deduplicate the two row
// bodies, since the bias-free kernel above would still need to be
// rewritten to call this template instead of its own inline arithmetic.
// ---------------------------------------------------------------------

template <bool HAS_BETA>
__device__ __forceinline__ void ln_fwd_row_body_f32(
    const float* xr, const float* gamma, const float* beta, float* yr,
    const unsigned int hidden, const float eps, float* scratch
) {
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) sum += xr[i];
    sum = block_reduce_sum(sum, scratch);
    float mean = sum / (float)hidden;

    float sumsq = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float d = xr[i] - mean;
        sumsq += d * d;
    }
    sumsq = block_reduce_sum(sumsq, scratch);
    float invvar = rsqrtf(sumsq / (float)hidden + eps);

    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float xhat = (xr[i] - mean) * invvar;
        yr[i] = HAS_BETA ? (xhat * gamma[i] + beta[i]) : (xhat * gamma[i]);
    }
}

extern "C" __global__ void layer_norm_fwd_f32_biased(
    const float* x, const float* gamma, const float* beta, float* y,
    const unsigned int hidden, const float eps
) {
    __shared__ float scratch[LN_BLOCK];
    size_t row = blockIdx.x;
    ln_fwd_row_body_f32<true>(
        x + row * (size_t)hidden, gamma, beta, y + row * (size_t)hidden,
        hidden, eps, scratch
    );
}

template <bool HAS_BETA>
__device__ __forceinline__ void ln_fwd_row_body_bf16(
    const __nv_bfloat16* xr, const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta, __nv_bfloat16* yr, const unsigned int hidden,
    const float eps, float* scratch
) {
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        sum += __bfloat162float(xr[i]);
    }
    sum = block_reduce_sum(sum, scratch);
    float mean = sum / (float)hidden;

    float sumsq = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float d = __bfloat162float(xr[i]) - mean;
        sumsq += d * d;
    }
    sumsq = block_reduce_sum(sumsq, scratch);
    float invvar = rsqrtf(sumsq / (float)hidden + eps);

    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float xhat = (__bfloat162float(xr[i]) - mean) * invvar;
        float scaled = xhat * __bfloat162float(gamma[i]);
        float outv = HAS_BETA ? (scaled + __bfloat162float(beta[i])) : scaled;
        yr[i] = __float2bfloat16(outv);
    }
}

extern "C" __global__ void layer_norm_fwd_bf16_biased(
    const __nv_bfloat16* x, const __nv_bfloat16* gamma,
    const __nv_bfloat16* beta, __nv_bfloat16* y, const unsigned int hidden,
    const float eps
) {
    __shared__ float scratch[LN_BLOCK];
    size_t row = blockIdx.x;
    ln_fwd_row_body_bf16<true>(
        x + row * (size_t)hidden, gamma, beta, y + row * (size_t)hidden,
        hidden, eps, scratch
    );
}
