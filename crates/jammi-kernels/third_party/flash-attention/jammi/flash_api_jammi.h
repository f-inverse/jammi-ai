/*
 * flash_api_jammi.h — torch-free C ABI over the vendored FlashAttention-2
 * hdim64 / bf16-or-fp16 / sm80 / non-causal forward + backward kernels
 * (`third_party/flash-attention/src/flash_{fwd,bwd}_hdim64_{bf16,fp16}_sm80.cu`
 * — campaign #443 D2 adds the fp16 pair alongside the original bf16 one;
 * `dtype` below (0 = bf16, 1 = fp16) selects which explicit specialisation
 * `flash_api_jammi.cu` calls).
 *
 * This file is jammi's own (not upstream). It exists so the Rust side
 * (`crates/jammi-kernels/src/flash/mod.rs`) never has to see
 * `Flash_fwd_params` / `Flash_bwd_params`: every field of those structs is
 * filled in `flash_api_jammi.cu` exactly as upstream
 * `csrc/flash_attn/flash_api.cpp` fills them (`set_params_fprop`,
 * `set_params_dgrad`, `mha_varlen_fwd`, `mha_varlen_bwd`), with each field
 * cited to its upstream line.
 *
 * Layout served (the qkv-PACKED varlen case, upstream
 * `flash_attn_varlen_qkvpacked_func`):
 *   qkv    bf16 [total_q, 3, H, 64]   q/k/v are strided views: row stride
 *                                     3*H*64, head stride 64, base offsets
 *                                     0 / H*64 / 2*H*64 elements.
 *   o      bf16 [total_q, H, 64]      contiguous.
 *   lse    f32  [H, total_q]          `unpadded_lse` layout
 *                                     (flash_api.cpp:652,688).
 *   d_o    bf16 [total_q, H, 64]      contiguous.
 *   d_qkv  bf16 [total_q, 3, H, 64]   dq/dk/dv written in place with the
 *                                     packed strides (flash_api.cpp:216-224).
 *   softmax_d  f32 [H, total_q + 128*B]                 scratch
 *                                     (flash_api.cpp:1100).
 *   dq_accum   f32 [nsplits, total_q + 128*B, H, 64]    scratch
 *                                     (flash_api.cpp:1112-1117); nsplits = 1
 *                                     when !deterministic (allocated
 *                                     uninitialised, the kernel clears it),
 *                                     ceil(num_SM / (B*H)) when deterministic
 *                                     (MUST be zeroed by the caller — the
 *                                     kernel does not clear it in that mode,
 *                                     flash_bwd_launch_template.h:84-88).
 *
 * Every `TORCH_CHECK` that applies to this layout is a returned status code
 * from the `jammi_flash_status` table below; nothing is silently ignored.
 * `p_dropout != 0` is a HARD ERROR (the build defines
 * FLASHATTENTION_DISABLE_DROPOUT, under which upstream's own check
 * flash_api.cpp:133-135 is the only thing standing between the caller and a
 * kernel that silently ignores the probability — and a recompute-in-backward
 * caller relies on the forward being a pure function of its inputs).
 *
 * Buffer element counts (`*_len`) are part of the contract and are checked
 * for EXACT equality with the shape-derived count: a caller that hands a
 * bigger or smaller buffer than the layout above has a shape bug, and the
 * kernels index by shape, not by buffer length.
 *
 * All calls are asynchronous on `stream`; no call synchronises, and no call
 * reads device memory on the host (cu_seqlens is consumed only by the
 * kernels — `total_q` and `max_seqlen` are host-side inputs).
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes. Every value has a static message: `jammi_flash_strerror`. */
enum jammi_flash_status {
    JAMMI_FLASH_OK = 0,
    /* A required pointer (buffer, cu_seqlens, stream may be NULL = legacy
     * default stream, but the buffers may not) is NULL. */
    JAMMI_FLASH_ERR_NULL_POINTER = 1,
    /* head_dim != 64 — only `run_mha_{fwd,bwd}_<bf16, 64, false>` is
     * compiled into libjammi_flash.a (HEADDIM_SWITCH is not linked). */
    JAMMI_FLASH_ERR_HEAD_DIM = 2,
    /* batch, num_heads, total_q or max_seqlen is <= 0 (flash_api.cpp:603
     * `batch_size > 0`; the max_seqlen == 0 arm at flash_api.cpp:737-744 /
     * 1184-1191 is "zero the outputs", which this ABI refuses instead of
     * emulating), or a shape product overflows int32 (the kernels index
     * with 32-bit offsets, flash.h:165-166). */
    JAMMI_FLASH_ERR_DIMS = 3,
    /* p_dropout != 0.0f (flash_api.cpp:133-135 under
     * FLASHATTENTION_DISABLE_DROPOUT). */
    JAMMI_FLASH_ERR_DROPOUT_UNSUPPORTED = 4,
    /* The (window_size_left, window_size_right) pair would select
     * `is_causal` (left < 0 && right == 0, flash_api.cpp:139) — the causal
     * template instantiation is not compiled into this library. */
    JAMMI_FLASH_ERR_CAUSAL_UNSUPPORTED = 5,
    /* softmax_scale is not finite or is <= 0 (this crate's own family-D
     * rule for a multiplicative scale; upstream does not check). */
    JAMMI_FLASH_ERR_SCALE = 6,
    /* A `*_len` differs from the element count the shape implies. */
    JAMMI_FLASH_ERR_BUFFER_LEN = 7,
    /* The current device's compute capability major < 8
     * (flash_api.cpp:540-542 / 1007-1009). */
    JAMMI_FLASH_ERR_COMPUTE_CAPABILITY = 8,
    /* A CUDA runtime query (cudaGetDevice / cudaDeviceGetAttribute) or the
     * post-launch cudaGetLastError reported an error. */
    JAMMI_FLASH_ERR_CUDA = 9,
    /* Internal invariant: params.num_splits > 1 after zero-init — would
     * mean the split-KV forward (flash_api.cpp:247-251) was requested; that
     * path is neither compiled nor reachable from this ABI. */
    JAMMI_FLASH_ERR_SPLIT_KERNEL = 10,
    /* `struct_size` in the args struct != sizeof the C struct: the Rust
     * `#[repr(C)]` mirror and this header have drifted. */
    JAMMI_FLASH_ERR_ABI = 11,
    /* `dq_accum_splits` (the caller's allocated split count) != the value
     * the kernel will use (1, or ceil(num_SM / (B*H)) when deterministic —
     * flash_api.cpp:1115, flash_bwd_launch_template.h:78-81). */
    JAMMI_FLASH_ERR_DQ_ACCUM_SPLITS = 12,
    /* `window_size_left` / `window_size_right` below -1 (the only negative
     * value with a meaning is -1 = unbounded). */
    JAMMI_FLASH_ERR_WINDOW = 13,
    /* `dtype` is neither 0 (bf16) nor 1 (fp16) — campaign #443 D2. */
    JAMMI_FLASH_ERR_DTYPE = 14,
};

/* Element dtype selector for `dtype` below (campaign #443 D2): selects
 * which of the two compiled explicit specialisations
 * (`run_mha_{fwd,bwd}_<cutlass::bfloat16_t, 64, false>` /
 * `run_mha_{fwd,bwd}_<cutlass::half_t, 64, false>`) `flash_api_jammi.cu`
 * calls. Every buffer's element type (`qkv`/`o`/`d_o`/`d_qkv`) is this
 * dtype; `softmax_lse`/`softmax_d`/`dq_accum` stay `f32` regardless. */
enum jammi_flash_dtype {
    JAMMI_FLASH_DTYPE_BF16 = 0,
    JAMMI_FLASH_DTYPE_FP16 = 1,
};

/* Forward arguments. Field order: 8-byte fields first (pointers, i64),
 * then 4-byte fields, so the C and Rust layouts carry no interior padding
 * that could differ between the two. `struct_size` MUST be set to
 * sizeof(jammi_flash_varlen_fwd_args) by the caller. `dtype` (campaign
 * #443 D2) is a 4-byte field, placed with the other 4-byte fields — its
 * addition does not change any OTHER field's offset, only appends one. */
typedef struct jammi_flash_varlen_fwd_args {
    const void *qkv;            /* dtype [total_q, 3, num_heads, 64] */
    void *o;                    /* dtype [total_q, num_heads, 64] */
    float *softmax_lse;         /* f32  [num_heads, total_q] */
    const int32_t *cu_seqlens;  /* i32  [batch + 1], device, cumulative */
    void *stream;               /* cudaStream_t (CUstream); NULL = default */
    int64_t qkv_len;            /* element counts of the buffers above */
    int64_t o_len;
    int64_t softmax_lse_len;
    int64_t cu_seqlens_len;
    int32_t struct_size;        /* sizeof(*this), ABI guard */
    int32_t total_q;            /* sum of the batch's sequence lengths */
    int32_t batch;
    int32_t num_heads;
    int32_t head_dim;           /* must be 64 */
    int32_t max_seqlen;         /* max over the batch's sequence lengths */
    int32_t window_size_left;   /* -1 = unbounded; else keys >= q - left */
    int32_t window_size_right;  /* -1 = unbounded; else keys <= q + right */
    float softmax_scale;
    float p_dropout;            /* must be 0.0f */
    int32_t dtype;              /* jammi_flash_dtype: 0 = bf16, 1 = fp16 */
} jammi_flash_varlen_fwd_args;

/* Backward arguments. Same ordering rule as the forward struct. */
typedef struct jammi_flash_varlen_bwd_args {
    const void *qkv;            /* dtype [total_q, 3, num_heads, 64] */
    const void *o;              /* dtype [total_q, num_heads, 64] (the fwd output) */
    const float *softmax_lse;   /* f32  [num_heads, total_q] (the fwd lse) */
    const void *d_o;            /* dtype [total_q, num_heads, 64] */
    void *d_qkv;                /* dtype [total_q, 3, num_heads, 64], written */
    float *softmax_d;           /* f32  [num_heads, total_q + 128*batch], scratch */
    float *dq_accum;            /* f32  [dq_accum_splits, total_q + 128*batch, num_heads, 64], scratch */
    const int32_t *cu_seqlens;  /* i32  [batch + 1] */
    void *stream;
    int64_t qkv_len;
    int64_t o_len;
    int64_t softmax_lse_len;
    int64_t d_o_len;
    int64_t d_qkv_len;
    int64_t softmax_d_len;
    int64_t dq_accum_len;
    int64_t cu_seqlens_len;
    int32_t struct_size;
    int32_t total_q;
    int32_t batch;
    int32_t num_heads;
    int32_t head_dim;
    int32_t max_seqlen;
    int32_t window_size_left;
    int32_t window_size_right;
    float softmax_scale;
    float p_dropout;            /* must be 0.0f */
    int32_t deterministic;      /* 0 / 1 */
    int32_t dq_accum_splits;    /* the split count dq_accum was allocated with */
    int32_t dtype;              /* jammi_flash_dtype: 0 = bf16, 1 = fp16 */
} jammi_flash_varlen_bwd_args;

/* Runs the forward. Writes `o` and `softmax_lse`. Asynchronous on `stream`. */
int32_t jammi_flash_varlen_fwd(const jammi_flash_varlen_fwd_args *args);

/* Runs the backward. Writes `d_qkv`; clobbers `softmax_d` and `dq_accum`.
 * When `deterministic`, `dq_accum` must be all-zero on entry. Asynchronous. */
int32_t jammi_flash_varlen_bwd(const jammi_flash_varlen_bwd_args *args);

/* The dq_accum split count the backward will use for (batch, num_heads,
 * deterministic) on the CURRENT device: 1 when !deterministic, else
 * ceil(num_SM / (batch * num_heads)) (flash_api.cpp:1115). Returns a status. */
int32_t jammi_flash_dq_accum_splits(int32_t batch, int32_t num_heads,
                                    int32_t deterministic, int32_t *out_splits);

/* The current device's multiprocessor count (cudaDevAttrMultiProcessorCount). */
int32_t jammi_flash_num_sms(int32_t *out_num_sms);

/* Static message for a status code; never NULL (unknown codes map to a
 * fixed "unknown" string). */
const char *jammi_flash_strerror(int32_t status);

/* sizeof the two args structs as compiled into the library — the Rust side
 * compares these against its `#[repr(C)]` mirrors. */
size_t jammi_flash_sizeof_fwd_args(void);
size_t jammi_flash_sizeof_bwd_args(void);

#ifdef __cplusplus
} /* extern "C" */
#endif
