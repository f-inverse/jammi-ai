// softmax.cu — fused masked softmax-last-dim forward + backward
// (output-only bwd). Compiled to PTX only when the `cuda` feature is
// active (see ../../build.rs); the pinned build flags (sm_80 baseline, no
// -use_fast_math) live there, not here.
//
// Domain: `scores` is `[<leading axes>, last]`, row-major contiguous (a
// possibly narrowed-but-contiguous nonzero-offset view is handled by the
// Rust glue slicing to `contiguous_offsets()` before the launch — this
// file assumes a flat `row*last + col` index and does not itself
// re-validate contiguity); `mask` likewise, per its own (possibly
// broadcast) leading-axis sizes. `last <= MAX_LAST_DIM` (4096),
// `rank <= MAX_RANK` (4, i.e. at most 3 leading axes), and
// `rows*last <= u32::MAX` are enforced by the Rust glue
// (../ops/softmax.rs, ../cuda/softmax.rs) before any launch.
//
// ONE THREAD BLOCK PER ROW for forward and backward. Every per-row
// reduction (max, sum) is a block-wide shared-memory tree reduction over
// `SM_BLOCK` threads — the same `block_reduce_sum` shape `layer_norm.cu`
// already ships and this repository's reviewers have already audited (see
// `ops::softmax`'s module doc for why this op takes the classic
// multi-pass route rather than an online single-pass rescaling
// recurrence); `block_reduce_max` is its `fmaxf` analogue.
//
// Forward reads `mask` through a 3-leading-axis broadcast index
// (`mask_row_offset`): each tensor's up-to-three leading-axis sizes are
// passed as individual scalar arguments, left-padded with `1` when the
// tensor's real leading-axis count is smaller than three (a virtual
// size-1 axis is a true no-op in both the decompose and re-ravel steps
// below, so no separate "how many axes are real" argument is needed).
//
// Backward (`softmax_bwd_dscores_*`) needs only `y` and `dy` — no mask,
// no broadcast index at all (the standard softmax backward identity is
// output-only): `dscores = (dy - dot(dy, y)) * y`. The `scale`
// multiplicative factor (see `../ops/softmax.rs`'s "scale semantics"
// module-doc section) is applied to `scores` BEFORE the mask add in the
// FORWARD kernels below; it does NOT appear here, because
// `SoftmaxLastDimFused::bwd` multiplies this kernel's raw
// `d(y)/d(pre_softmax)` output by `scale` at the Rust/Tensor level
// (a plain `Tensor::affine` call, not a further CUDA launch this file
// owns) to produce the gradient flowing back into `scores`.
#include <cuda_bf16.h>
#include <cstddef>

#define SM_BLOCK 256

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

__device__ __forceinline__ float block_reduce_max(float val, float* scratch) {
    int tid = threadIdx.x;
    scratch[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }
    float total = scratch[0];
    __syncthreads();
    return total;
}

// Rounds `s * scale_bf` to BF16 immediately, matching `half::bf16`'s own
// `Mul` impl (round-trip through f32, round once) — see
// `../ops/softmax.rs`'s module doc's "scale semantics" -> "reproducing
// candle's own affine rounding point" section: this is the SAME rounding
// `Affine<bf16>::f` performs for the `v * mul` step, `mul` here being
// `scale_bf` (itself already rounded to bf16 once, per-launch, matching
// `Affine`'s own `mul = T::from_f64(self.0)`).
__device__ __forceinline__ __nv_bfloat16 bf16_mul_rounded(__nv_bfloat16 s, __nv_bfloat16 scale_bf) {
    return __float2bfloat16(__bfloat162float(s) * __bfloat162float(scale_bf));
}

// Rounds `scores_i + mask_i` to BF16 immediately, matching
// `candle_nn::ops::softmax`'s native BF16 `broadcast_add` (which rounds at
// exactly this step, since it never upcasts) — see `../ops/softmax.rs`'s
// module doc's "bf16 mask-add rounding" section: this ANNIHILATES a real
// (small) score against the real `MASKED_LOGIT` magnitude (`-10_000.0`) at
// a MASKED position, matching the upstream HuggingFace ModernBERT
// reference's own BF16-native mask add rather than a strictly-more-
// precise (but wrong-relative-to-that-reference) F32-throughout add. Only
// used on rows that reach the main reduction below (a FULLY-masked row is
// short-circuited to zero before this is ever called — see
// `mask_row_is_fully_masked` and the module doc's "fully-masked row"
// section). The ONE place the bf16 forward kernel deviates from F32
// accumulation; every step after this (max/exp/sum/normalize) still
// accumulates in f32.
__device__ __forceinline__ float bf16_add_rounded(__nv_bfloat16 a, __nv_bfloat16 b) {
    float sum = __bfloat162float(a) + __bfloat162float(b);
    return __bfloat162float(__float2bfloat16(sum));
}

// Maps a flattened `row` (0-indexed over `scores`'s up-to-three leading
// axes, row-major, `s2` fastest-varying) to the flat row index into
// `mask`'s own leading-axis space, substituting index `0` on every axis
// where `mask`'s size (`m0`/`m1`/`m2`) is `1` — the exact same full,
// exact multi-index unravel/ravel `ops/softmax.rs`'s `mask_row_offset`
// documents and performs on the CPU arm.
__device__ __forceinline__ size_t mask_row_offset(
    size_t row,
    unsigned int s0, unsigned int s1, unsigned int s2,
    unsigned int m0, unsigned int m1, unsigned int m2
) {
    unsigned int idx2 = (unsigned int)(row % s2);
    size_t rem = row / s2;
    unsigned int idx1 = (unsigned int)(rem % s1);
    rem = rem / s1;
    unsigned int idx0 = (unsigned int)(rem % s0);
    unsigned int mi0 = (m0 == 1) ? 0 : idx0;
    unsigned int mi1 = (m1 == 1) ? 0 : idx1;
    unsigned int mi2 = (m2 == 1) ? 0 : idx2;
    return ((size_t)mi0 * m1 + mi1) * m2 + mi2;
}

// Safe softmax (see `../ops/softmax.rs`'s module doc's "fully-masked row"
// section, and PyTorch's `_safe_softmax` / FlashAttention-2's `softmax.h`
// zero-on-empty-row precedent this op deliberately follows instead of
// `candle_nn::ops::softmax`'s own NaN-or-uniform behavior there): a row is
// fully masked iff its OWN mask values (read alone, never `scores+mask`)
// contain no exact `0.0` — this crate's additive-mask convention uses
// `0.0` as the "unmasked" identity and any value `< 0.0` as "some degree
// of masking" (see `../ops/softmax.rs`'s `row_is_fully_masked`, the exact
// same rule, on the CPU arm). Every thread in the block computes the SAME
// `mask_max` after `block_reduce_max` (its scratch-buffer broadcast), so
// every thread takes the SAME branch below — no partial-block divergence.
__device__ __forceinline__ bool mask_row_is_fully_masked(float mask_max) {
    return mask_max < 0.0f;
}

// ---------------------------------------------------------------------
// Forward: y = softmax(scores + mask, last dim). F32 accumulation
// throughout (row max and row sum both in f32). Under
// `zero_on_fully_masked` (construction data, `FullyMaskedPolicy::Zeros`
// in the Rust glue) ONLY, a fully-masked row (detected from `mask` alone)
// short-circuits to an all-zero output before ever reading `scores` for
// the add; under `Propagate` (`0`) this never happens and the kernel
// reproduces `candle_nn::ops::softmax` exactly on every row.
// ---------------------------------------------------------------------

extern "C" __global__ void softmax_fwd_f32(
    const float* scores, const float* mask, float* y,
    const unsigned int last,
    const unsigned int s0, const unsigned int s1, const unsigned int s2,
    const unsigned int m0, const unsigned int m1, const unsigned int m2,
    const unsigned int zero_on_fully_masked,
    const float scale
) {
    __shared__ float scratch[SM_BLOCK];
    size_t row = blockIdx.x;
    size_t mrow = mask_row_offset(row, s0, s1, s2, m0, m1, m2);
    const float* sr = scores + row * (size_t)last;
    const float* mr = mask + mrow * (size_t)last;
    float* yr = y + row * (size_t)last;

    // `zero_on_fully_masked` is `FullyMaskedPolicy` as a `u32` (Rust glue's
    // `policy_flag`) — construction data threaded through unchanged, never
    // a runtime predicate this kernel evaluates on its own. `Propagate`
    // (`0`) skips this entire block and falls through to the ordinary
    // computation below unconditionally, reproducing eager EXACTLY
    // (including `NaN` on an all-`-inf` row) — see `ops/softmax.rs`'s
    // module doc for why this is NOT a universal rule this kernel applies
    // unconditionally.
    if (zero_on_fully_masked) {
        float mask_max = -INFINITY;
        for (unsigned int i = threadIdx.x; i < last; i += blockDim.x) {
            mask_max = fmaxf(mask_max, mr[i]);
        }
        mask_max = block_reduce_max(mask_max, scratch);
        if (mask_row_is_fully_masked(mask_max)) {
            for (unsigned int i = threadIdx.x; i < last; i += blockDim.x) {
                yr[i] = 0.0f;
            }
            return;
        }
    }

    // `scale` then the mask add, as TWO separate f32 operations (matching
    // candle's own `scores.affine(scale, 0.0)` followed by
    // `broadcast_add(mask)` for `T = f32` — see `../ops/softmax.rs`'s
    // module doc's "scale semantics" section). `--fmad` contraction is
    // accepted here within this op's stated CUDA tolerance, per this
    // crate's existing build-flags doctrine (`../../build.rs`) — not
    // pinned away, unlike the CPU arm's Rust code which never auto-fuses.
    float maxv = -INFINITY;
    for (unsigned int i = threadIdx.x; i < last; i += blockDim.x) {
        float v = sr[i] * scale + mr[i];
        maxv = fmaxf(maxv, v);
    }
    maxv = block_reduce_max(maxv, scratch);

    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < last; i += blockDim.x) {
        float v = sr[i] * scale + mr[i];
        float e = expf(v - maxv);
        yr[i] = e;
        sum += e;
    }
    sum = block_reduce_sum(sum, scratch);

    for (unsigned int i = threadIdx.x; i < last; i += blockDim.x) {
        yr[i] = yr[i] / sum;
    }
}

// BF16: rounds `scores+mask` to bf16 immediately (`bf16_add_rounded`, see
// its own doc), then accumulates the REST of the reduction in f32,
// rounding to bf16 exactly once at the very end. Rather than storing an
// f32-precision intermediate into the bf16 output buffer as scratch
// (which would round TWICE — once storing it, once after the final
// divide — breaking this crate's f32-accumulate-round-once convention),
// this RECOMPUTES `bf16_add_rounded` and `expf` a second time in the
// normalize pass: one extra pair of global bf16 reads and one extra
// `expf` call per element, bandwidth/compute-cheap relative to the
// O(last^2)-class memory problem this op actually targets (see
// `ops::softmax`'s module doc).
extern "C" __global__ void softmax_fwd_bf16(
    const __nv_bfloat16* scores, const __nv_bfloat16* mask, __nv_bfloat16* y,
    const unsigned int last,
    const unsigned int s0, const unsigned int s1, const unsigned int s2,
    const unsigned int m0, const unsigned int m1, const unsigned int m2,
    const unsigned int zero_on_fully_masked,
    const float scale
) {
    __shared__ float scratch[SM_BLOCK];
    size_t row = blockIdx.x;
    size_t mrow = mask_row_offset(row, s0, s1, s2, m0, m1, m2);
    const __nv_bfloat16* sr = scores + row * (size_t)last;
    const __nv_bfloat16* mr = mask + mrow * (size_t)last;
    __nv_bfloat16* yr = y + row * (size_t)last;
    // Rounded ONCE per launch (matching `Affine<bf16>::f`'s own `mul =
    // T::from_f64(self.0)` — a single conversion of the scale CONSTANT,
    // not a per-element re-conversion; see `../ops/softmax.rs`'s matching
    // `softmax_fwd_bf16`'s identical `scale_bf` hoist).
    const __nv_bfloat16 scale_bf = __float2bfloat16(scale);

    // Safe softmax first (see `softmax_fwd_f32`'s identical check and the
    // module doc), under `zero_on_fully_masked` ONLY: short-circuit a
    // fully-masked row to zero before the BF16-rounded add below is ever
    // reached. `Propagate` (`0`) never takes this branch, so a fully-
    // masked row falls through to the BF16-rounded-add path below
    // unconditionally, reproducing eager's own annihilated-uniform output
    // exactly.
    if (zero_on_fully_masked) {
        float mask_max = -INFINITY;
        for (unsigned int i = threadIdx.x; i < last; i += blockDim.x) {
            mask_max = fmaxf(mask_max, __bfloat162float(mr[i]));
        }
        mask_max = block_reduce_max(mask_max, scratch);
        if (mask_row_is_fully_masked(mask_max)) {
            for (unsigned int i = threadIdx.x; i < last; i += blockDim.x) {
                yr[i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

    float maxv = -INFINITY;
    for (unsigned int i = threadIdx.x; i < last; i += blockDim.x) {
        __nv_bfloat16 scaled = bf16_mul_rounded(sr[i], scale_bf);
        float v = bf16_add_rounded(scaled, mr[i]);
        maxv = fmaxf(maxv, v);
    }
    maxv = block_reduce_max(maxv, scratch);

    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < last; i += blockDim.x) {
        __nv_bfloat16 scaled = bf16_mul_rounded(sr[i], scale_bf);
        float v = bf16_add_rounded(scaled, mr[i]);
        sum += expf(v - maxv);
    }
    sum = block_reduce_sum(sum, scratch);

    for (unsigned int i = threadIdx.x; i < last; i += blockDim.x) {
        __nv_bfloat16 scaled = bf16_mul_rounded(sr[i], scale_bf);
        float v = bf16_add_rounded(scaled, mr[i]);
        float e = expf(v - maxv);
        yr[i] = __float2bfloat16(e / sum);
    }
}

// ---------------------------------------------------------------------
// Backward: dscores = (dy - dot(dy, y)) * y — output-only, no mask/scores
// input, no broadcast index. One block per row, one reduction pass.
// ---------------------------------------------------------------------

extern "C" __global__ void softmax_bwd_dscores_f32(
    const float* y, const float* dy, float* dscores, const unsigned int last
) {
    __shared__ float scratch[SM_BLOCK];
    size_t row = blockIdx.x;
    const float* yr = y + row * (size_t)last;
    const float* dyr = dy + row * (size_t)last;
    float* dsr = dscores + row * (size_t)last;

    float dot = 0.0f;
    for (unsigned int i = threadIdx.x; i < last; i += blockDim.x) {
        dot += dyr[i] * yr[i];
    }
    dot = block_reduce_sum(dot, scratch);

    for (unsigned int i = threadIdx.x; i < last; i += blockDim.x) {
        dsr[i] = (dyr[i] - dot) * yr[i];
    }
}

// esc-045 round 3 (GH#374): matches ATen's own bf16 softmax backward
// rounding placement — `aten/src/ATen/native/cuda/SoftMax.cu`'s
// `softmax_backward_cuda_out` (v2.13.0) computes `Tensor tmp = grad *
// output;` as a SEPARATE elementwise kernel FIRST (`mul_kernel_cuda`,
// `opmath_t = float`, cast to `BFloat16` ONCE per element when it stores
// `tmp`), so the per-element product is genuinely BF16-rounded BEFORE the
// row reduction runs; `cunn_SoftMaxBackward` then sums THAT already-rounded
// `tmp` in f32, and its epilogue reads `tmp_i` (not the raw `dy_i` — per
// the kernel's own "gradOutput that we get here is really gradOutput *
// output" comment) for the final subtract-and-round. See
// `../ops/softmax.rs`'s `dscores_row_bf16` doc for the full derivation and
// `tests/softmax_bwd_dscores_aten_rounding.rs` for the op-level oracle. A
// prior revision of this kernel accumulated the UNROUNDED product directly
// — more precise per term, but not what torch does; `bf16_tmp` below is
// recomputed (not cached) in the second pass, matching this file's
// existing "read the same input pointers, don't stage an extra buffer"
// idiom — one extra `__float2bfloat16`/`__bfloat162float` round-trip per
// element, bandwidth-neutral.
extern "C" __global__ void softmax_bwd_dscores_bf16(
    const __nv_bfloat16* y, const __nv_bfloat16* dy, __nv_bfloat16* dscores,
    const unsigned int last
) {
    __shared__ float scratch[SM_BLOCK];
    size_t row = blockIdx.x;
    const __nv_bfloat16* yr = y + row * (size_t)last;
    const __nv_bfloat16* dyr = dy + row * (size_t)last;
    __nv_bfloat16* dsr = dscores + row * (size_t)last;

    float dot = 0.0f;
    for (unsigned int i = threadIdx.x; i < last; i += blockDim.x) {
        __nv_bfloat16 tmp = __float2bfloat16(__bfloat162float(dyr[i]) * __bfloat162float(yr[i]));
        dot += __bfloat162float(tmp);
    }
    dot = block_reduce_sum(dot, scratch);

    for (unsigned int i = threadIdx.x; i < last; i += blockDim.x) {
        float yv = __bfloat162float(yr[i]);
        __nv_bfloat16 tmp = __float2bfloat16(__bfloat162float(dyr[i]) * yv);
        dsr[i] = __float2bfloat16(__bfloat162float(tmp) - yv * dot);
    }
}
