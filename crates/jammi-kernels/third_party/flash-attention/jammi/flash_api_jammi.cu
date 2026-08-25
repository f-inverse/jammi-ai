/*
 * flash_api_jammi.cu — jammi's torch-free wrapper over the vendored
 * FlashAttention-2 hdim64/bf16/sm80 non-causal kernels. See
 * flash_api_jammi.h for the ABI and the layout it serves.
 *
 * Every `params.*` assignment below cites the upstream line in
 * csrc/flash_attn/flash_api.cpp (tag v2.8.3.post1, a8aa52b1) it mirrors.
 * Upstream's `set_params_fprop` (flash_api.cpp:26-159) and
 * `set_params_dgrad` (flash_api.cpp:161-241) take `at::Tensor`s and read
 * strides off them; here the strides are the packed layout's constants.
 */
#include "flash_api_jammi.h"

#include <cmath>
#include <cstdint>
#include <cstring>

#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "flash.h"

namespace FLASH_NAMESPACE {
// The two explicit specialisations defined in
// flash_{fwd,bwd}_hdim64_bf16_sm80.cu. Declared here so the calls below
// bind to those definitions rather than implicitly instantiating the
// (definition-less) primary templates from flash.h:189-192.
template <>
void run_mha_fwd_<cutlass::bfloat16_t, 64, false>(Flash_fwd_params &params, cudaStream_t stream);
template <>
void run_mha_bwd_<cutlass::bfloat16_t, 64, false>(Flash_bwd_params &params, cudaStream_t stream);
}  // namespace FLASH_NAMESPACE

namespace {

constexpr int32_t kHeadDim = 64;
// flash_api.cpp:647 / 1051: round_multiple(head_size, head_size <= 128 ? 32 : 64) == 64 for 64.
constexpr int32_t kHeadDimRounded = 64;
// flash_api.cpp:1100,1106,1113: "128 is the max block size on the seqlen_q dimension".
constexpr int32_t kSeqlenQPad = 128;

// flash_api.cpp:646 / 1050.
inline int32_t round_multiple(int32_t x, int32_t m) { return (x + m - 1) / m * m; }

// Every dimension product the kernels form must fit the 32-bit offsets
// they index with (flash.h:165-166). Checked in 64-bit before any product
// is formed in 32-bit.
inline bool fits_i32(int64_t v) { return v >= 0 && v <= INT32_MAX; }

int32_t query_device(int *device, int *num_sms, int *cc_major) {
    cudaError_t e = cudaGetDevice(device);
    if (e != cudaSuccess) return JAMMI_FLASH_ERR_CUDA;
    if (num_sms != nullptr) {
        // hardware_info.h:37-41 get_num_sm — without its exit(1).
        e = cudaDeviceGetAttribute(num_sms, cudaDevAttrMultiProcessorCount, *device);
        if (e != cudaSuccess) return JAMMI_FLASH_ERR_CUDA;
    }
    if (cc_major != nullptr) {
        // hardware_info.h:30-35 get_compute_capability — without its exit(1).
        e = cudaDeviceGetAttribute(cc_major, cudaDevAttrComputeCapabilityMajor, *device);
        if (e != cudaSuccess) return JAMMI_FLASH_ERR_CUDA;
    }
    return JAMMI_FLASH_OK;
}

// `flash_bwd_kernel.h`'s dropout setup reads `params.rng_state[0]` and
// `[1]` UNCONDITIONALLY, at function scope — NOT inside `if constexpr
// (Is_dropout)` (only the further uses at :535/:541/:546 are guarded).
// Upstream always points `rng_state` at a live 2-element device tensor,
// unconditionally, in the forward (flash_api.cpp:725) and always forwards
// it in the backward (flash_api.cpp:1171-1172); leaving it NULL here (as
// this wrapper originally did) is a null pointer dereferenced by every
// backward launch — it happened to pass because nvcc dead-code-eliminated
// the load when `Is_dropout` is false at compile time (this build's
// FLASHATTENTION_DISABLE_DROPOUT), which is a compiler behaviour, not a
// language guarantee. Fixed with a lazily-allocated, zeroed, 2-element
// scratch buffer, reused for the process's lifetime (never freed, like a
// static): its VALUES are never read past that unconditional line (dropout
// is compiled out), only the pointer must be dereferenceable.
uint64_t *rng_state_scratch() {
    static uint64_t *ptr = nullptr;
    if (ptr == nullptr) {
        uint64_t *p = nullptr;
        if (cudaMalloc(&p, 2 * sizeof(uint64_t)) != cudaSuccess) return nullptr;
        if (cudaMemset(p, 0, 2 * sizeof(uint64_t)) != cudaSuccess) return nullptr;
        ptr = p;
    }
    return ptr;
}

// flash_api.cpp:1115 (the deterministic dq_accum split count) — identical
// to the grid the kernel launches with, flash_bwd_launch_template.h:78-81.
int32_t dq_accum_splits(int32_t batch, int32_t num_heads, int32_t deterministic, int32_t *out) {
    if (!deterministic) {
        *out = 1;
        return JAMMI_FLASH_OK;
    }
    int device = 0, num_sms = 0;
    int32_t st = query_device(&device, &num_sms, nullptr);
    if (st != JAMMI_FLASH_OK) return st;
    const int64_t bh = int64_t(batch) * int64_t(num_heads);
    *out = int32_t((int64_t(num_sms) + bh - 1) / bh);
    return JAMMI_FLASH_OK;
}

// The checks common to fwd and bwd, in the order upstream performs them
// where an order exists. `window_size_*` are normalised IN PLACE exactly as
// flash_api.cpp does before `set_params_fprop`.
int32_t check_common(int32_t total_q, int32_t batch, int32_t num_heads, int32_t head_dim,
                     int32_t max_seqlen, float softmax_scale, float p_dropout,
                     int32_t *window_size_left, int32_t *window_size_right) {
    // Only the hdim64 specialisations are linked (see the declarations above).
    if (head_dim != kHeadDim) return JAMMI_FLASH_ERR_HEAD_DIM;
    // flash_api.cpp:603 / 1044 `batch_size > 0`; the rest pin the domain
    // this ABI serves (the max_seqlen == 0 arm, flash_api.cpp:737-744, is
    // refused rather than emulated).
    if (batch <= 0 || num_heads <= 0 || total_q <= 0 || max_seqlen <= 0) {
        return JAMMI_FLASH_ERR_DIMS;
    }
    if (max_seqlen > total_q) return JAMMI_FLASH_ERR_DIMS;
    // 32-bit offset budget (flash.h:165-166): the largest offsets the
    // kernels form are into dq_accum, (total_q + 128*B) * H * d_rounded,
    // and into qkv, total_q * 3 * H * d.
    const int64_t rows_padded = int64_t(total_q) + int64_t(kSeqlenQPad) * int64_t(batch);
    if (!fits_i32(rows_padded * int64_t(num_heads) * int64_t(kHeadDimRounded))) {
        return JAMMI_FLASH_ERR_DIMS;
    }
    if (!fits_i32(int64_t(total_q) * 3 * int64_t(num_heads) * int64_t(kHeadDim))) {
        return JAMMI_FLASH_ERR_DIMS;
    }
    // flash_api.cpp:132 `p_dropout < 1.f` and :133-135 (DISABLE_DROPOUT ⇒
    // `p_dropout == 0.0f`). Any non-zero value — including NaN, which
    // compares unequal to everything — is refused.
    if (!(p_dropout == 0.0f)) return JAMMI_FLASH_ERR_DROPOUT_UNSUPPORTED;
    // This crate's own rule for a multiplicative scale (family D): finite
    // and > 0. `!(x > 0)` also catches NaN.
    if (!(softmax_scale > 0.0f) || !std::isfinite(softmax_scale)) return JAMMI_FLASH_ERR_SCALE;
    // -1 is the only negative value with a meaning (unbounded).
    if (*window_size_left < -1 || *window_size_right < -1) return JAMMI_FLASH_ERR_WINDOW;
    // flash_api.cpp:608-609 / 1055-1056: a window at least as wide as the
    // longest sequence is the unbounded window.
    if (*window_size_left >= max_seqlen) *window_size_left = -1;
    if (*window_size_right >= max_seqlen) *window_size_right = -1;
    // flash_api.cpp:139 `is_causal = window_size_left < 0 && window_size_right == 0`
    // — that selects the causal instantiation, which is not linked.
    if (*window_size_left < 0 && *window_size_right == 0) return JAMMI_FLASH_ERR_CAUSAL_UNSUPPORTED;
    return JAMMI_FLASH_OK;
}

int32_t check_device() {
    int device = 0, cc_major = 0;
    int32_t st = query_device(&device, nullptr, &cc_major);
    if (st != JAMMI_FLASH_OK) return st;
    // flash_api.cpp:540-542 / 1007-1009 "FlashAttention only supports Ampere GPUs or newer."
    if (cc_major < 8) return JAMMI_FLASH_ERR_COMPUTE_CAPABILITY;
    return JAMMI_FLASH_OK;
}

// set_params_fprop, flash_api.cpp:26-159, for the packed varlen layout.
// `window_size_*` must already be normalised by check_common.
void fill_fprop(FLASH_NAMESPACE::Flash_fwd_params &params, const void *qkv, void *o,
                float *softmax_lse, const int32_t *cu_seqlens, int32_t total_q, int32_t batch,
                int32_t num_heads, int32_t max_seqlen, float softmax_scale,
                int32_t window_size_left, int32_t window_size_right) {
    using index_t = FLASH_NAMESPACE::Qkv_params::index_t;
    // flash_api.cpp:56 "Reset the parameters": every field not assigned
    // below is zero/NULL — p_ptr, seqused_k, leftpad_k, block_table,
    // knew/vnew, rotary, alibi_slopes_ptr, rng_state, philox_args,
    // num_splits (=> the non-split kernel, flash_api.cpp:247), and the
    // *_batch_stride fields (unused when cu_seqlens_q != nullptr,
    // flash_api.cpp:75-84; block_info.h:29-31 takes the sum_s_q branch).
    params = {};

    // flash_api.cpp:58.
    params.is_bf16 = true;

    // flash_api.cpp:61-63 with the packed layout: q/k/v are the three
    // [H, D] slabs of each [3, H, D] row.
    const index_t hd = index_t(num_heads) * kHeadDim;
    const cutlass::bfloat16_t *qkv_t = static_cast<const cutlass::bfloat16_t *>(qkv);
    params.q_ptr = const_cast<cutlass::bfloat16_t *>(qkv_t);
    params.k_ptr = const_cast<cutlass::bfloat16_t *>(qkv_t + hd);
    params.v_ptr = const_cast<cutlass::bfloat16_t *>(qkv_t + 2 * hd);
    // flash_api.cpp:64-70 "All stride are in elements": stride(-3) of a
    // [total_q, 3, H, D] view narrowed on dim 1 is 3*H*D; stride(-2) is D.
    params.q_row_stride = 3 * hd;
    params.k_row_stride = 3 * hd;
    params.v_row_stride = 3 * hd;
    params.q_head_stride = kHeadDim;
    params.k_head_stride = kHeadDim;
    params.v_head_stride = kHeadDim;
    // flash_api.cpp:71-73: o is contiguous [total_q, H, D].
    params.o_ptr = o;
    params.o_row_stride = hd;
    params.o_head_stride = kHeadDim;

    // flash_api.cpp:86-88: one cumulative array serves both q and k (the
    // packed layout has total_k == total_q); seqused_k is NULL.
    params.cu_seqlens_q = const_cast<int *>(cu_seqlens);
    params.cu_seqlens_k = const_cast<int *>(cu_seqlens);
    params.seqused_k = nullptr;

    // flash_api.cpp:91 (no return_softmax: p is NULL) and :94.
    params.p_ptr = nullptr;
    params.softmax_lse_ptr = softmax_lse;

    // flash_api.cpp:97-106 with mha_varlen_fwd's arguments
    // (flash_api.cpp:671-675): b = batch, seqlen_q = seqlen_k = max_seqlen,
    // the rounded lengths are round_multiple(max_seqlen, 128)
    // (flash_api.cpp:648-649), h = h_k = num_heads, d = d_rounded = 64.
    params.b = batch;
    params.h = num_heads;
    params.h_k = num_heads;
    params.h_h_k_ratio = 1;
    params.seqlen_q = max_seqlen;
    params.seqlen_k = max_seqlen;
    params.seqlen_q_rounded = round_multiple(max_seqlen, 128);
    params.seqlen_k_rounded = round_multiple(max_seqlen, 128);
    params.d = kHeadDim;
    params.d_rounded = kHeadDimRounded;

    // flash_api.cpp:116-121 (softcap == 0 branch). `softmax_scale *
    // M_LOG2E` is a float*double product narrowed to float, as upstream.
    params.softcap = 0.0f;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    // flash_api.cpp:124-131 with p_dropout == 0 (keep-probability 1).
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.0f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;

    // flash_api.cpp:139 (false after check_common) and :141-144.
    params.is_causal = false;
    if (window_size_left < 0 && window_size_right >= 0) window_size_left = max_seqlen;
    if (window_size_left >= 0 && window_size_right < 0) window_size_right = max_seqlen;
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    // flash_api.cpp:151.
    params.is_seqlens_k_cumulative = true;

    // flash_api.cpp:157-158 with mha_varlen_fwd's arguments
    // (flash_api.cpp:687-688): seqlenq_ngroups_swapped is false because
    // h == h_k (flash_api.cpp:592), unpadded_lse is true.
    params.unpadded_lse = true;
    params.seqlenq_ngroups_swapped = false;

    // flash_api.cpp:689 (fwd) / :1161 (bwd).
    params.total_q = total_q;
    // flash_api.cpp:697 (`!paged_KV ? 1`).
    params.page_block_size = 1;
    // `rng_state_scratch` above: a live, zeroed, 2-element device
    // buffer, not NULL — the backward kernel dereferences this
    // unconditionally regardless of whether dropout is compiled in. If the
    // allocation ever fails this is left NULL and the caller (both entry
    // points below) checks for that and returns `JAMMI_FLASH_ERR_CUDA`
    // before launching.
    params.rng_state = rng_state_scratch();
}

}  // namespace

extern "C" {

const char *jammi_flash_strerror(int32_t status) {
    switch (status) {
        case JAMMI_FLASH_OK: return "ok";
        case JAMMI_FLASH_ERR_NULL_POINTER: return "a required device pointer is NULL";
        case JAMMI_FLASH_ERR_HEAD_DIM: return "head_dim must be 64 (only the hdim64 kernels are compiled)";
        case JAMMI_FLASH_ERR_DIMS: return "batch, num_heads, total_q and max_seqlen must be > 0, max_seqlen <= total_q, and every offset must fit int32";
        case JAMMI_FLASH_ERR_DROPOUT_UNSUPPORTED: return "p_dropout must be 0.0: this build compiles dropout out (FLASHATTENTION_DISABLE_DROPOUT) and a non-zero probability would be silently ignored";
        case JAMMI_FLASH_ERR_CAUSAL_UNSUPPORTED: return "window_size (left < 0, right == 0) selects the causal kernel, which is not compiled";
        case JAMMI_FLASH_ERR_SCALE: return "softmax_scale must be finite and > 0";
        case JAMMI_FLASH_ERR_BUFFER_LEN: return "a buffer element count differs from the count its shape implies";
        case JAMMI_FLASH_ERR_COMPUTE_CAPABILITY: return "FlashAttention only supports Ampere GPUs or newer (compute capability >= 8.0)";
        case JAMMI_FLASH_ERR_CUDA: return "a CUDA runtime call failed";
        case JAMMI_FLASH_ERR_SPLIT_KERNEL: return "internal: num_splits > 1 would select the split-KV kernel, which is not compiled";
        case JAMMI_FLASH_ERR_ABI: return "args struct size mismatch between the Rust mirror and flash_api_jammi.h";
        case JAMMI_FLASH_ERR_DQ_ACCUM_SPLITS: return "dq_accum_splits differs from the split count the backward kernel uses on this device";
        case JAMMI_FLASH_ERR_WINDOW: return "window_size_left/right must be -1 (unbounded) or >= 0";
        default: return "unknown jammi_flash status";
    }
}

size_t jammi_flash_sizeof_fwd_args(void) { return sizeof(jammi_flash_varlen_fwd_args); }
size_t jammi_flash_sizeof_bwd_args(void) { return sizeof(jammi_flash_varlen_bwd_args); }

int32_t jammi_flash_num_sms(int32_t *out_num_sms) {
    if (out_num_sms == nullptr) return JAMMI_FLASH_ERR_NULL_POINTER;
    int device = 0, num_sms = 0;
    int32_t st = query_device(&device, &num_sms, nullptr);
    if (st != JAMMI_FLASH_OK) return st;
    *out_num_sms = num_sms;
    return JAMMI_FLASH_OK;
}

int32_t jammi_flash_dq_accum_splits(int32_t batch, int32_t num_heads, int32_t deterministic,
                                    int32_t *out_splits) {
    if (out_splits == nullptr) return JAMMI_FLASH_ERR_NULL_POINTER;
    if (batch <= 0 || num_heads <= 0) return JAMMI_FLASH_ERR_DIMS;
    return dq_accum_splits(batch, num_heads, deterministic, out_splits);
}

int32_t jammi_flash_varlen_fwd(const jammi_flash_varlen_fwd_args *a) {
    if (a == nullptr) return JAMMI_FLASH_ERR_NULL_POINTER;
    if (a->struct_size != int32_t(sizeof(jammi_flash_varlen_fwd_args))) return JAMMI_FLASH_ERR_ABI;
    if (a->qkv == nullptr || a->o == nullptr || a->softmax_lse == nullptr || a->cu_seqlens == nullptr) {
        return JAMMI_FLASH_ERR_NULL_POINTER;
    }
    int32_t window_size_left = a->window_size_left;
    int32_t window_size_right = a->window_size_right;
    int32_t st = check_common(a->total_q, a->batch, a->num_heads, a->head_dim, a->max_seqlen,
                              a->softmax_scale, a->p_dropout, &window_size_left, &window_size_right);
    if (st != JAMMI_FLASH_OK) return st;
    // Buffer element counts, exact (flash_api.cpp:611-623 CHECK_SHAPE, :652
    // softmax_lse {num_heads, total_q}).
    const int64_t hd = int64_t(a->num_heads) * kHeadDim;
    if (a->qkv_len != int64_t(a->total_q) * 3 * hd) return JAMMI_FLASH_ERR_BUFFER_LEN;
    if (a->o_len != int64_t(a->total_q) * hd) return JAMMI_FLASH_ERR_BUFFER_LEN;
    if (a->softmax_lse_len != int64_t(a->num_heads) * int64_t(a->total_q)) return JAMMI_FLASH_ERR_BUFFER_LEN;
    if (a->cu_seqlens_len != int64_t(a->batch) + 1) return JAMMI_FLASH_ERR_BUFFER_LEN;
    st = check_device();
    if (st != JAMMI_FLASH_OK) return st;

    FLASH_NAMESPACE::Flash_fwd_params params;
    fill_fprop(params, a->qkv, a->o, a->softmax_lse, a->cu_seqlens, a->total_q, a->batch,
               a->num_heads, a->max_seqlen, a->softmax_scale, window_size_left, window_size_right);
    // the rng_state scratch allocation failed.
    if (params.rng_state == nullptr) return JAMMI_FLASH_ERR_CUDA;
    // The recompute-soundness invariant: the split-KV forward
    // (flash_api.cpp:247-251, `num_splits > 1 || force_split_kernel`) is
    // never selected. `params = {}` made num_splits 0; assert it anyway so
    // a future edit that sets it cannot pass silently — and the split
    // kernel's TU is not compiled into this library at all.
    if (params.num_splits > 1) return JAMMI_FLASH_ERR_SPLIT_KERNEL;

    // Clear any stale (non-sticky) error so the post-launch check below
    // reports THIS launch, not an earlier caller's.
    (void)cudaGetLastError();
    // flash_api.cpp:738-739 `run_mha_fwd(params, stream, /*force_split_kernel=*/paged_KV=false)`
    // → flash_api.cpp:248 `run_mha_fwd_<elem_type, kHeadDim, Is_causal>` with
    // <bf16, 64, false>.
    FLASH_NAMESPACE::run_mha_fwd_<cutlass::bfloat16_t, 64, false>(params, static_cast<cudaStream_t>(a->stream));
    if (cudaGetLastError() != cudaSuccess) return JAMMI_FLASH_ERR_CUDA;
    return JAMMI_FLASH_OK;
}

int32_t jammi_flash_varlen_bwd(const jammi_flash_varlen_bwd_args *a) {
    if (a == nullptr) return JAMMI_FLASH_ERR_NULL_POINTER;
    if (a->struct_size != int32_t(sizeof(jammi_flash_varlen_bwd_args))) return JAMMI_FLASH_ERR_ABI;
    if (a->qkv == nullptr || a->o == nullptr || a->softmax_lse == nullptr || a->d_o == nullptr ||
        a->d_qkv == nullptr || a->softmax_d == nullptr || a->dq_accum == nullptr ||
        a->cu_seqlens == nullptr) {
        return JAMMI_FLASH_ERR_NULL_POINTER;
    }
    int32_t window_size_left = a->window_size_left;
    int32_t window_size_right = a->window_size_right;
    int32_t st = check_common(a->total_q, a->batch, a->num_heads, a->head_dim, a->max_seqlen,
                              a->softmax_scale, a->p_dropout, &window_size_left, &window_size_right);
    if (st != JAMMI_FLASH_OK) return st;
    // flash_api.cpp:1115 / flash_bwd_launch_template.h:78-81: the split
    // count is a device property; the caller's allocation must match.
    const int32_t deterministic = a->deterministic != 0;
    int32_t nsplits = 0;
    st = dq_accum_splits(a->batch, a->num_heads, deterministic, &nsplits);
    if (st != JAMMI_FLASH_OK) return st;
    if (a->dq_accum_splits != nsplits) return JAMMI_FLASH_ERR_DQ_ACCUM_SPLITS;
    // Buffer element counts, exact (flash_api.cpp:1058-1064 CHECK_SHAPE,
    // :1100 softmax_d {num_heads, total_q + 128*batch}, :1113/1116
    // dq_accum {[nsplits,] total_q + 128*batch, num_heads, head_size_rounded}).
    const int64_t hd = int64_t(a->num_heads) * kHeadDim;
    const int64_t rows_padded = int64_t(a->total_q) + int64_t(kSeqlenQPad) * int64_t(a->batch);
    if (a->qkv_len != int64_t(a->total_q) * 3 * hd) return JAMMI_FLASH_ERR_BUFFER_LEN;
    if (a->o_len != int64_t(a->total_q) * hd) return JAMMI_FLASH_ERR_BUFFER_LEN;
    if (a->softmax_lse_len != int64_t(a->num_heads) * int64_t(a->total_q)) return JAMMI_FLASH_ERR_BUFFER_LEN;
    if (a->d_o_len != int64_t(a->total_q) * hd) return JAMMI_FLASH_ERR_BUFFER_LEN;
    if (a->d_qkv_len != int64_t(a->total_q) * 3 * hd) return JAMMI_FLASH_ERR_BUFFER_LEN;
    if (a->softmax_d_len != int64_t(a->num_heads) * rows_padded) return JAMMI_FLASH_ERR_BUFFER_LEN;
    if (a->dq_accum_len != int64_t(nsplits) * rows_padded * int64_t(a->num_heads) * kHeadDimRounded) {
        return JAMMI_FLASH_ERR_BUFFER_LEN;
    }
    if (a->cu_seqlens_len != int64_t(a->batch) + 1) return JAMMI_FLASH_ERR_BUFFER_LEN;
    st = check_device();
    if (st != JAMMI_FLASH_OK) return st;

    using index_t = FLASH_NAMESPACE::Qkv_params::index_t;
    FLASH_NAMESPACE::Flash_bwd_params params;
    // set_params_dgrad, flash_api.cpp:161-241, begins with set_params_fprop
    // (flash_api.cpp:196-210) — same arguments as the forward, o = the
    // forward's output (flash_api.cpp:1144).
    fill_fprop(params, a->qkv, const_cast<void *>(a->o), const_cast<float *>(a->softmax_lse),
               a->cu_seqlens, a->total_q, a->batch, a->num_heads, a->max_seqlen, a->softmax_scale,
               window_size_left, window_size_right);
    // the rng_state scratch allocation failed — the backward kernel
    // dereferences `params.rng_state[0]`/`[1]` unconditionally
    // (`flash_bwd_kernel.h:446`), so a NULL here would be a null pointer
    // dereference in device code, not a clean refusal.
    if (params.rng_state == nullptr) return JAMMI_FLASH_ERR_CUDA;

    // flash_api.cpp:213-215: d_o is contiguous [total_q, H, D].
    params.do_ptr = const_cast<void *>(a->d_o);
    params.do_row_stride = index_t(a->num_heads) * kHeadDim;
    params.do_head_stride = kHeadDim;
    // flash_api.cpp:216-224 with dq/dk/dv = the three slabs of the packed
    // d_qkv (upstream's qkvpacked path hands `dqkv[:, 0/1/2]` views in;
    // their stride(-3) is 3*H*D and stride(-2) is D).
    cutlass::bfloat16_t *dqkv_t = static_cast<cutlass::bfloat16_t *>(a->d_qkv);
    params.dq_ptr = dqkv_t;
    params.dk_ptr = dqkv_t + index_t(a->num_heads) * kHeadDim;
    params.dv_ptr = dqkv_t + 2 * index_t(a->num_heads) * kHeadDim;
    params.dq_row_stride = 3 * index_t(a->num_heads) * kHeadDim;
    params.dk_row_stride = 3 * index_t(a->num_heads) * kHeadDim;
    params.dv_row_stride = 3 * index_t(a->num_heads) * kHeadDim;
    params.dq_head_stride = kHeadDim;
    params.dk_head_stride = kHeadDim;
    params.dv_head_stride = kHeadDim;
    // flash_api.cpp:226-231: *_batch_stride are unused with cu_seqlens
    // (left zero by `params = {}`).

    // flash_api.cpp:233-235 with mha_varlen_bwd's arguments
    // (flash_api.cpp:1148-1150): dq_accum set (loop == true), dk/dv accum NULL.
    params.dq_accum_ptr = a->dq_accum;
    params.dk_accum_ptr = nullptr;
    params.dv_accum_ptr = nullptr;
    // flash_api.cpp:238 / :1152.
    params.dsoftmax_sum = a->softmax_d;
    // flash_api.cpp:240 / :1158.
    params.deterministic = deterministic != 0;
    // flash_api.cpp:1160: `!deterministic ? 0 : dq_accum.stride(0)` of the
    // [nsplits, total_q + 128*batch, num_heads, head_size_rounded] tensor.
    params.dq_accum_split_stride =
        !deterministic ? 0 : index_t(rows_padded) * index_t(a->num_heads) * kHeadDimRounded;
    // flash_api.cpp:1161.
    params.total_q = a->total_q;
    // flash_api.cpp:1171-1180: rng_state / philox_args only matter under
    // dropout (compiled out); both stay NULL/zero from `params = {}`.

    if (params.num_splits > 1) return JAMMI_FLASH_ERR_SPLIT_KERNEL;

    (void)cudaGetLastError();
    // flash_api.cpp:1163,1185 `run_mha_bwd(params, stream)` →
    // flash_api.cpp:761 `run_mha_bwd_<elem_type, kHeadDim, Is_causal>` with
    // <bf16, 64, false>.
    FLASH_NAMESPACE::run_mha_bwd_<cutlass::bfloat16_t, 64, false>(params, static_cast<cudaStream_t>(a->stream));
    if (cudaGetLastError() != cudaSuccess) return JAMMI_FLASH_ERR_CUDA;
    return JAMMI_FLASH_OK;
}

}  // extern "C"
