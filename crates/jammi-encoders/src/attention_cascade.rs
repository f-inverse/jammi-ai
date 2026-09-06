//! The per-layer fused-attention cascade — `flash (caller-supplied) →
//! mem_efficient_attention → attention_block_fused → eager composition
//! (with `softmax_last_dim_fused` admission)` — shared by every encoder
//! whose attention arm wants it, extracted from `crate::modernbert` (R1'/R2'
//! of `plan-cattn-cmlp-v2.md`, issue #462).
//!
//! ## What moved here, and what did not
//!
//! Every PURE predicate and every per-layer numeric composition
//! (`attention_block_admission_predicate`, `mem_efficient_attention_predicate`,
//! `softmax_admission_predicate` + `softmax_apply_training`,
//! [`FusedAttentionMasks`]/[`TrainingMaskInputs`], `forward_memeff_attention`,
//! `forward_eager_training_attention_composition`, and the top-level
//! [`training_attention_cascade`] dispatcher) moved verbatim (numerics
//! unchanged) from `crate::modernbert`. The flash TRANSPORT protocol —
//! `FlashDecision`/`CompactedBatch` themselves, the dense/ragged/padded
//! FlashAttention-2 call sites, `unpad_rows`/`repad_rows`, and
//! `decide_flash_admission` — stays in `crate::modernbert`: it is an
//! encoder-BOUNDARY concern (a whole-forward compaction decision, not a
//! per-layer arm), and cannot be delivered through a per-layer seam (R1'
//! ruling). This module only ever *reports* the flash decision a caller
//! hands it (`admit_cascade("attention_block_flash", ..)`, so the counter
//! fires for every caller, including one that never transports) and, when
//! that decision is `Fused`, delegates to the caller's own transport via
//! [`training_attention_cascade`]'s `on_flash_fused` callback.
//!
//! ## `RopeCtx`: representing "this caller has no RoPE at all"
//!
//! [`AttentionBlockFused`]/[`MemEfficientAttention`] are `CustomOp3`s: a
//! `rope_pack` tensor argument is REQUIRED at every call, whether or not
//! `rope` (construction data, a `bool`) is `true` — when it is `false` the
//! argument is present but never read (see either op's own module doc's
//! "rope_pack ... only when rope == true" section). A caller with no
//! positional-embedding table at all (BERT/DistilBERT — absolute position
//! embeddings are baked into the embeddings sum, never applied per layer)
//! therefore still needs SOME tensor to hand the fused arm, even though it
//! is never read. [`RopeCtx`] represents this exactly: `enabled: false`
//! callers set `pack` to a per-module cached placeholder (allocated ONCE,
//! at construction, never read — see `bert::BertSelfAttention`'s own
//! `rope_placeholder` field), never re-derived per forward. This is a
//! narrower, type-safe restatement of plan v2's "`rope: None` with a
//! per-module cached `[2,1,1,64]` placeholder": the cascade's own functions
//! take `&RopeCtx` (never `Option<&RopeCtx>`) so the placeholder-vs-no-op
//! choice is always explicit at the call site, and the CustomOp3 arity
//! constraint above is satisfied by construction rather than by an
//! `Option::unwrap_or` at the one call site that would need it.
//!
//! ## The QKV cat bridge: a real, measured cost on BOTH arms
//!
//! BERT/DistilBERT project `q`/`k`/`v` through three SEPARATE linears
//! (unlike ModernBERT's single fused `Wqkv`), so every training-mode call
//! into this cascade bridges them with `Tensor::cat(&[q, k, v], D::Minus1)`
//! first — `[B, S, 3*hidden]`. At `batch=8, seq=512, hidden=768` (BERT-base
//! shape, `f32`) that materialises `8 * 512 * 3 * 768 * 4` bytes = **37.75
//! MB** on the forward pass, and candle's own `Op::Cat` backward
//! (`backprop.rs`: one `narrow` + one `zeros_like` `or_insert` + one add,
//! PER ARG) allocates and zero-fills THREE separate `[B, S, 768]` buffers
//! on the backward pass — the same 37.75 MB again, split three ways. This
//! cost rides on BOTH the fused arm (which reshapes the cat'd `qkv` into
//! `[B, S, 3, h, d]` before calling the op) and the eager arm (which
//! `narrow`s the SAME cat'd pack back apart) — so an A/B measurement of
//! this seam at the block site prices the fused kernel's OWN win net of
//! nothing hidden in the bridge. The bridge is otherwise value-preserving
//! (cat-then-narrow is exact; `zeros_like` plus an add is exact), so it
//! introduces no rounding of its own — only allocation and copy cost. A
//! packed-QKV LoRA target (BERT/DistilBERT never expose one today) would
//! be the cheaper future bridge; out of scope for this unit.
//!
//! ## The `Propagate` policy's bf16 divergence on a fully-masked row
//!
//! BERT/DistilBERT declare [`FullyMaskedPolicy::Propagate`] (not
//! ModernBERT's `Zeros`) — see `bert::BertSelfAttention::forward_training`'s
//! doc for why: it is the exact-arithmetic-equivalent, at `F32`, of main's
//! own eager `softmax(scores/scale + mask)` on an all-padding row (a
//! genuine input class — see `crate::mask::sliding_window_mask`'s doc,
//! though BERT/DistilBERT never build a sliding-window band themselves;
//! the row still arises from an all-padding QUERY position). At `BF16`,
//! however, `Propagate` is NOT bit-identical between the eager and fused
//! arms on that one row class: [`crate::mask::MASKED_LOGIT`] is `-10_000.0`
//! (`mask.rs`'s own constant); eager's mask-ADD happens directly at the
//! backbone's `BF16` precision (8 significand bits, ULP `64` at magnitude
//! `1e4`), so the tiny per-position score differences a fully-masked row's
//! raw `q @ kᵀ` still carries are ABSORBED — every position rounds to the
//! SAME `BF16` value once `-10_000.0` is added, and the row's softmax comes
//! out uniform (the same "annihilated-uniform in BF16" behaviour
//! `crate::modernbert`'s own softmax doc records for eval on this exact
//! row class). The FUSED op computes its internal mask-add in `F32`
//! (`AttentionBlockFused`/`SoftmaxLastDimFused`'s own per-dtype rounding
//! contract) before rounding once to the backbone dtype at the very end —
//! so the same tiny per-position differences survive there, and the row is
//! NOT perfectly uniform. Both are internally consistent with their own
//! arithmetic; they simply disagree with each other, by construction, at
//! `BF16` on this one row class — disclosed here the same way
//! `crate::modernbert`'s own softmax doc discloses its `Zeros` divergence
//! from eval. `F32` (this unit's own tolerance oracles, and every fixture
//! this crate ships) is unaffected: `internal_dtype == backbone_dtype`
//! there, so both arms add the exact same `F32` bits.

use std::sync::LazyLock;

use candle_core::{DType, Device, Tensor, D};
use jammi_kernels::admission::{
    admission_mode, admit, admit_cascade, cascade_counters_for, counters_for, device_is_supported,
    CascadeOutcome, DispatchCounters, DispatchOutcome, PredicateOutcome,
};
use jammi_kernels::ops::{
    apply2, apply3, mem_efficient_attention, AttentionBlockFused, FullyMaskedPolicy,
    MemEfficientAttention, SoftmaxLastDimFused, ATTENTION_BLOCK_HEAD_DIM, ATTENTION_BLOCK_MAX_SEQ,
    MAX_LAST_DIM, MAX_RANK, MEM_EFFICIENT_MAX_SEQ, MEM_EFFICIENT_MIN_CHUNK,
};

use crate::error::EncoderError;
use crate::modernbert::{CompactedBatch, FlashDecision};

/// The key-chunk width [`forward_memeff_attention`] hands
/// [`MemEfficientAttention::new`] — moved verbatim from `crate::modernbert`
/// (see that constant's prior doc, `modernbert.rs` git history, for the
/// "floor-respecting default, not a measured crossover point" disclosure
/// this move does not change).
const MEM_EFFICIENT_CHUNK: usize = 1024;

/// Pins [`MEM_EFFICIENT_CHUNK`] above [`MEM_EFFICIENT_MIN_CHUNK`]'s launch-
/// count floor at compile time — moved verbatim alongside the constant it
/// guards.
const _: () = assert!(MEM_EFFICIENT_CHUNK >= MEM_EFFICIENT_MIN_CHUNK);

/// A local-attention layer's sliding-window half-width, bundled with the
/// "is this layer local at all" fact `is_local: bool` used to be a
/// separately-carried, independently-must-agree field. `Some` replaces
/// `is_local == true` (ModernBERT's local layers); `None` replaces
/// `is_local == false` (ModernBERT's global layers, and EVERY BERT/
/// DistilBERT layer — neither architecture has a sliding-window concept).
#[derive(Debug, Clone, Copy)]
pub(crate) struct LocalWindow {
    pub(crate) half_window: usize,
}

/// [`RopeCtx::apply`]'s function-pointer type, named so its own field
/// declaration reads as one bounded-complexity type rather than the raw
/// `Option<&dyn Fn(..) -> Result<..>>` spelled out inline (clippy's
/// `type_complexity` lint).
type RopeApplyFn<'a> = dyn Fn(&Tensor) -> Result<Tensor, EncoderError> + 'a;

/// The per-layer RoPE context the cascade's fused/eager arms need — see
/// this module's doc for why `enabled: false` still carries a `pack`
/// tensor rather than making the whole context `Option`.
pub(crate) struct RopeCtx<'a> {
    /// `[2, 1, 1, seq_max, head_dim]` (or a smaller, unread placeholder
    /// when `!enabled`) — [`AttentionBlockFused`]/[`MemEfficientAttention`]'s
    /// shared `rope_pack` argument.
    pub(crate) pack: &'a Tensor,
    /// Whether RoPE is actually applied: the fused ops' own `rope: bool`
    /// construction data, and whether the eager composition below rotates
    /// Q/K at all.
    pub(crate) enabled: bool,
    /// The eager per-tensor rotation, consulted only when `enabled`.
    /// `None` whenever `!enabled` (BERT/DistilBERT: no RoPE, so no
    /// rotation function exists to call).
    pub(crate) apply: Option<&'a RopeApplyFn<'a>>,
}

fn apply_rope(rope: &RopeCtx<'_>, x: &Tensor) -> Result<Tensor, EncoderError> {
    if rope.enabled {
        match rope.apply {
            Some(f) => f(x),
            None => Ok(x.clone()),
        }
    } else {
        Ok(x.clone())
    }
}

/// Fused/eager dispatch counters for the shared attention-block cascade —
/// keyed identically to `crate::modernbert::ATTENTION_BLOCK_DISPATCH_COUNTERS`
/// (both resolve to the SAME `&'static DispatchCounters` via
/// `jammi_kernels::admission::counters_for`'s op-keyed registry — see that
/// function's own doc). A separate `LazyLock` here, rather than importing
/// ModernBERT's, is what lets `crate::modernbert`'s own tests keep reading
/// their copy UNMODIFIED while this module's callers (BERT, DistilBERT)
/// read theirs.
pub(crate) static ATTENTION_BLOCK_DISPATCH_COUNTERS: LazyLock<&'static DispatchCounters> =
    LazyLock::new(|| counters_for("attention_block_fused"));

/// Same rationale as [`ATTENTION_BLOCK_DISPATCH_COUNTERS`], for
/// `softmax_last_dim_fused` / `crate::modernbert::SOFTMAX_DISPATCH_COUNTERS`.
pub(crate) static SOFTMAX_DISPATCH_COUNTERS: LazyLock<&'static DispatchCounters> =
    LazyLock::new(|| counters_for("softmax_last_dim_fused"));

/// Test-only serialization for two-sided (`fused` advanced AND `eager`
/// unchanged) assertions against the process-wide dispatch/cascade counter
/// registry this module's functions read (`ATTENTION_BLOCK_DISPATCH_COUNTERS`,
/// `SOFTMAX_DISPATCH_COUNTERS`, and the `attention_block_flash`/
/// `mem_efficient_attention` cascade counters `cascade_counters_for`
/// resolves): promoted here from `crate::modernbert`'s own (module-private)
/// `mod tests::ATTENTION_BLOCK_COUNTER_TEST_LOCK` (issue #462, R2') so
/// `crate::bert`'s and `crate::distilbert`'s own unit tests — which read the
/// SAME process-wide counters through this crate-shared cascade, in the
/// SAME `cargo test --lib` binary — can serialize against ModernBERT's
/// counter tests too, mirroring `crate::layer_norm::DISPATCH_COUNTER_TEST_LOCK`'s
/// identical crate-visible-promotion shape. `crate::modernbert`'s own `mod
/// tests` re-imports this exact static under its original bare name, so
/// every one of its 30+ existing `ATTENTION_BLOCK_COUNTER_TEST_LOCK.lock()`
/// call sites keeps compiling and passing unmodified — a path-only change.
#[cfg(test)]
pub(crate) static ATTENTION_BLOCK_COUNTER_TEST_LOCK: std::sync::Mutex<()> =
    std::sync::Mutex::new(());

/// The fused whole-attention-block kernel's domain, checked at the call
/// site (family D / K2) — moved verbatim from
/// `crate::modernbert::attention_block_admission_predicate`: `qkv`'s device
/// is one [`device_is_supported`] accepts, `qkv`/`extended_mask` share a
/// dtype the kernel implements PER-DEVICE (`F32` on either device; `BF16`
/// or `F16` admitted ONLY on CUDA), `qkv` is contiguous, `head_dim` is
/// exactly [`ATTENTION_BLOCK_HEAD_DIM`], `seq` is nonzero and within
/// [`ATTENTION_BLOCK_MAX_SEQ`], `extended_mask` (the padding mask ALONE) is
/// contiguous and shaped `[batch|1, 1, 1, seq]`, and — on a local layer
/// only — `local_mask` (the per-forward padding-plus-band sum) is present,
/// contiguous, and shaped `[batch|1, 1, seq, seq]`.
pub(crate) fn attention_block_admission_predicate(
    qkv: &Tensor,
    seq: usize,
    _h: usize,
    d: usize,
    extended_mask: &Tensor,
    is_local: bool,
    local_mask: Option<&Tensor>,
) -> (bool, &'static str) {
    if !device_is_supported(qkv.device()) {
        return (false, "device_is_cpu_or_cuda");
    }
    let dtype_ok = if qkv.dtype() != extended_mask.dtype() {
        false
    } else if qkv.device().is_cuda() {
        matches!(qkv.dtype(), DType::F32 | DType::BF16 | DType::F16)
    } else {
        matches!(qkv.dtype(), DType::F32)
    };
    if !dtype_ok {
        return (
            false,
            if qkv.device().is_cuda() {
                "dtype_f32_bf16_or_f16_matching_between_qkv_and_mask_on_cuda"
            } else {
                "dtype_f32_matching_between_qkv_and_mask_on_cpu"
            },
        );
    }
    if !qkv.is_contiguous() {
        return (false, "qkv_contiguous");
    }
    if d != ATTENTION_BLOCK_HEAD_DIM {
        return (false, "head_dim_is_attention_block_fixed_head_dim");
    }
    if seq == 0 || seq > ATTENTION_BLOCK_MAX_SEQ {
        return (false, "seq_within_attention_block_max_seq");
    }
    if !extended_mask.is_contiguous() {
        return (false, "mask_contiguous");
    }
    let m_dims = extended_mask.dims();
    if m_dims.len() != 4 || m_dims[1] != 1 || m_dims[2] != 1 || m_dims[3] != seq {
        return (false, "mask_shape_batch_or_one_1_1_seq");
    }
    if is_local {
        let Some(local) = local_mask else {
            return (false, "local_mask_present");
        };
        if !local.is_contiguous() {
            return (false, "local_mask_contiguous");
        }
        let l_dims = local.dims();
        if l_dims.len() != 4
            || (l_dims[0] != 1 && l_dims[0] != m_dims[0])
            || l_dims[1] != 1
            || l_dims[2] != seq
            || l_dims[3] != seq
        {
            return (false, "local_mask_shape_batch_or_one_1_seq_seq");
        }
    }
    (true, "domain_ok")
}

/// The memory-efficient (chunked) attention arm's domain — moved verbatim
/// from `crate::modernbert::mem_efficient_attention_predicate`. `DomainMiss`
/// when `flash` already holds (the flash cascade owns this call, memeff is
/// not even consulted), when `seq` is within [`ATTENTION_BLOCK_MAX_SEQ`]
/// (the block arm handles it), or when `seq` exceeds
/// [`MEM_EFFICIENT_MAX_SEQ`] (the op's own validated ceiling); `CapabilityMiss`
/// on an unsupported device or a device-dishonest dtype.
pub(crate) fn mem_efficient_attention_predicate(
    device: &Device,
    dtype: DType,
    seq: usize,
    flash: &FlashDecision,
) -> (PredicateOutcome, &'static str) {
    if matches!(flash, FlashDecision::Fused(_)) {
        return (PredicateOutcome::DomainMiss, "flash_admission_holds");
    }
    if !device_is_supported(device) {
        return (PredicateOutcome::CapabilityMiss, "device_is_cpu_or_cuda");
    }
    let dtype_ok = if device.is_cuda() {
        matches!(dtype, DType::F32 | DType::BF16 | DType::F16)
    } else {
        matches!(dtype, DType::F32)
    };
    if !dtype_ok {
        return (
            PredicateOutcome::DomainMiss,
            if device.is_cuda() {
                "dtype_f32_bf16_or_f16_on_cuda"
            } else {
                "dtype_f32_only_on_cpu"
            },
        );
    }
    if seq <= ATTENTION_BLOCK_MAX_SEQ {
        return (
            PredicateOutcome::DomainMiss,
            "seq_within_attention_block_max_seq",
        );
    }
    if seq > MEM_EFFICIENT_MAX_SEQ {
        return (
            PredicateOutcome::DomainMiss,
            "seq_within_mem_efficient_max_seq",
        );
    }
    (PredicateOutcome::Holds, "domain_ok")
}

/// The fused masked-softmax kernel's domain — moved verbatim from
/// `crate::modernbert::softmax_admission_predicate`.
pub(crate) fn softmax_admission_predicate(
    scores: &Tensor,
    mask: &Tensor,
    scores_divisor: f64,
) -> (bool, &'static str) {
    if !device_is_supported(scores.device()) {
        return (false, "device_is_cpu_or_cuda");
    }
    if scores.dtype() != mask.dtype()
        || !matches!(scores.dtype(), DType::F32 | DType::BF16 | DType::F16)
    {
        return (
            false,
            "dtype_f32_bf16_or_f16_matching_between_scores_and_mask",
        );
    }
    if !scores.is_contiguous() {
        return (false, "scores_contiguous");
    }
    if !mask.is_contiguous() {
        return (false, "mask_contiguous");
    }
    let rank = scores.dims().len();
    if rank == 0 || rank > MAX_RANK {
        return (false, "rank_within_kernel_max_rank");
    }
    let last = *scores.dims().last().unwrap_or(&0);
    if last == 0 || last > MAX_LAST_DIM {
        return (false, "last_dim_within_kernel_max_last_dim");
    }
    if !jammi_kernels::ops::mask_broadcast_class_holds(scores, mask) {
        return (false, "mask_broadcast_class");
    }
    let scale_mul = (1.0 / scores_divisor) as f32;
    if !(scale_mul.is_finite() && scale_mul > 0.0) {
        return (false, "scale_finite_positive");
    }
    (true, "domain_ok")
}

/// The additive masks the FUSED whole-attention-block arm consumes, built
/// ONCE per training forward by the caller (`crate::modernbert::ModernBert::
/// forward_hidden_inner`, `crate::bert::Bert::forward_hidden`,
/// `crate::distilbert::DistilBert::forward_hidden`) — never per layer.
/// Moved verbatim from `crate::modernbert::FusedAttentionMasks`; see that
/// struct's prior doc (git history) for the launch-count and rounding-order
/// disclosures this move does not change.
pub(crate) struct FusedAttentionMasks {
    /// `[batch, 1, 1, seq]` in the backbone dtype — the padding mask
    /// alone, what a GLOBAL layer's fused arm passes as `mask`. BERT/
    /// DistilBERT (no local layers at all) only ever populate this field.
    pub(crate) global: Tensor,
    /// `[batch, 1, seq, seq]` in the backbone dtype — padding plus the
    /// sliding-window band. `None` iff the caller has no local layer
    /// (every BERT/DistilBERT caller, and an all-global ModernBERT).
    pub(crate) local: Option<Tensor>,
}

impl FusedAttentionMasks {
    /// `extended_f32` is the `[batch, 1, 1, seq]` padding mask
    /// (`crate::mask::extended_attention_mask`'s output), `local_band_f32`
    /// is `crate::mask::sliding_window_mask`'s `[1, 1, seq, seq]` band (or
    /// `None` for a caller with no local layer), and `dtype` is the
    /// backbone dtype the attention `qkv` will carry.
    pub(crate) fn build(
        extended_f32: &Tensor,
        local_band_f32: Option<&Tensor>,
        dtype: DType,
    ) -> Result<Self, EncoderError> {
        let global = extended_f32.to_dtype(dtype)?;
        let local = match local_band_f32 {
            Some(band) => Some(extended_f32.broadcast_add(band)?.to_dtype(dtype)?),
            None => None,
        };
        Ok(Self { global, local })
    }
}

/// The three mask inputs [`training_attention_cascade`] takes, bundled —
/// moved verbatim from `crate::modernbert::TrainingMaskInputs`.
pub(crate) struct TrainingMaskInputs<'a> {
    pub(crate) extended: &'a Tensor,
    pub(crate) local_band: Option<&'a Tensor>,
    pub(crate) fused: Option<&'a FusedAttentionMasks>,
}

/// Dispatches to [`SoftmaxLastDimFused`] when its domain holds, else falls
/// back to the eager `(scores / scale).broadcast_add(mask)` plus
/// `candle_nn::ops::softmax` composition — moved verbatim from
/// `crate::modernbert::softmax_apply_training`, parameterized by `policy`
/// (plan v2 R3': the fully-masked policy is declared once per caller at the
/// seam edge and applies to every arm of the cascade for that caller,
/// including this one).
pub(crate) fn softmax_apply_training(
    scores: &Tensor,
    mask: &Tensor,
    scores_divisor: f64,
    policy: FullyMaskedPolicy,
) -> Result<Tensor, EncoderError> {
    let (holds, predicate) = softmax_admission_predicate(scores, mask, scores_divisor);
    let outcome = admit(
        admission_mode(),
        "softmax_last_dim_fused",
        predicate,
        holds,
        *SOFTMAX_DISPATCH_COUNTERS,
    )?;
    match outcome {
        DispatchOutcome::Fused => Ok(apply2(
            scores,
            mask,
            SoftmaxLastDimFused::new(policy).with_scale((1.0 / scores_divisor) as f32)?,
        )?),
        DispatchOutcome::Eager => Ok(candle_nn::ops::softmax(
            &(scores / scores_divisor)?.broadcast_add(mask)?,
            D::Minus1,
        )?),
    }
}

/// The memory-efficient (chunked) attention arm — moved verbatim from
/// `crate::modernbert::ModernBertAttention::forward_memeff_attention`,
/// parameterized by `rope`/`half_window`/`policy` in place of `self.rope`/
/// `self.half_window`/the hardcoded `FullyMaskedPolicy::Zeros`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_memeff_attention(
    qkv: &Tensor,
    batch: usize,
    seq: usize,
    h: usize,
    d: usize,
    extended_mask_f32: &Tensor,
    rope: &RopeCtx<'_>,
    half_window: Option<usize>,
    policy: FullyMaskedPolicy,
) -> Result<Tensor, EncoderError> {
    let qkv5 = qkv.reshape((batch, seq, 3, h, d))?;
    let key_mask = extended_mask_f32.to_dtype(qkv.dtype())?;
    let op = MemEfficientAttention::new(
        1.0 / (d as f32).sqrt(),
        policy,
        rope.enabled,
        half_window,
        MEM_EFFICIENT_CHUNK,
    )
    .map_err(|e| EncoderError::Config(format!("mem_efficient_attention: {e}")))?;
    mem_efficient_attention(&qkv5, rope.pack, &key_mask, op)
        .map_err(|e| EncoderError::Config(format!("mem_efficient_attention: {e}")))
}

/// TODAY'S exact training-arm eager composition — moved verbatim from
/// `crate::modernbert::ModernBertAttention::forward_eager_training_attention_composition`,
/// parameterized by `rope`/`window`/`policy`/`training` in place of `self.rope`/
/// `self.is_local`/the hardcoded `FullyMaskedPolicy::Zeros`/`self.training`.
/// `training` keeps BOTH of the original method's branches reachable
/// (the fused-softmax training branch AND the plain eval-style
/// two-sequential-adds branch): every real caller only ever reaches this
/// function through [`training_attention_cascade`] with `training: true`,
/// but `crate::modernbert`'s own unit tests call the ModernBERT method
/// wrapper directly with either value — moving the branch changes nothing
/// about which branch a given caller reaches.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_eager_training_attention_composition(
    qkv: &Tensor,
    batch: usize,
    seq: usize,
    h: usize,
    d: usize,
    extended_mask: &Tensor,
    local_band: Option<&Tensor>,
    rope: &RopeCtx<'_>,
    window: Option<LocalWindow>,
    policy: FullyMaskedPolicy,
    training: bool,
) -> Result<Tensor, EncoderError> {
    let q = qkv
        .narrow(D::Minus1, 0, h * d)?
        .reshape((batch, seq, h, d))?
        .transpose(1, 2)?;
    let k = qkv
        .narrow(D::Minus1, h * d, h * d)?
        .reshape((batch, seq, h, d))?
        .transpose(1, 2)?;
    let v = qkv
        .narrow(D::Minus1, 2 * h * d, h * d)?
        .reshape((batch, seq, h, d))?
        .transpose(1, 2)?;

    let q = apply_rope(rope, &q)?;
    let k = apply_rope(rope, &k)?;

    let scale = (d as f64).sqrt();
    // UNSCALED when training (folded into `softmax_apply_training`'s own
    // `scale` instead) — see that function's doc.
    let raw_scores = crate::contiguous_matmul(&q, &k.transpose(D::Minus1, D::Minus2)?)?;
    let extended_mask = extended_mask.to_dtype(raw_scores.dtype())?;

    let attn = if training {
        let mask = match (window.is_some(), local_band) {
            (true, Some(band)) => {
                extended_mask.broadcast_add(&band.to_dtype(raw_scores.dtype())?)?
            }
            (true, None) => {
                return Err(EncoderError::Config(
                    "local-attention layer reached without a sliding-window band".into(),
                ))
            }
            (false, _) => extended_mask,
        };
        softmax_apply_training(&raw_scores, &mask, scale, policy)?
    } else {
        let scores = (&raw_scores / scale)?;
        let scores = scores.broadcast_add(&extended_mask)?;
        let scores = match (window.is_some(), local_band) {
            (true, Some(band)) => scores.broadcast_add(&band.to_dtype(scores.dtype())?)?,
            (true, None) => {
                return Err(EncoderError::Config(
                    "local-attention layer reached without a sliding-window band".into(),
                ))
            }
            (false, _) => scores,
        };
        candle_nn::ops::softmax(&scores, D::Minus1)?
    };

    Ok(crate::contiguous_matmul(&attn, &v)?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((batch, seq, h * d))?)
}

/// The shared per-layer cascade — moved verbatim (numerics unchanged) from
/// `crate::modernbert::ModernBertAttention::forward_training_attention`'s
/// body, parameterized by `rope`/`window`/`policy`/`flash` in place of
/// `self.rope`/`self.is_local`+`self.half_window`/the hardcoded
/// `FullyMaskedPolicy::Zeros`/the method's own `flash: &FlashDecision`
/// argument (already a parameter before this move).
///
/// `on_flash_fused` is the ONE piece of this cascade that stays
/// caller-specific: when the flash cascade admits `Fused`, dense/ragged
/// FlashAttention-2 transport is an encoder-boundary protocol this module
/// never owns (R1' ruling — see this module's doc). ModernBERT's own
/// wrapper method passes its real `forward_flash_dense_attention`; BERT/
/// DistilBERT never reach this closure at all, because they always supply
/// `flash: &FlashDecision::Declined { .. }` — `admit_cascade` can only ever
/// report `CascadeOutcome::Fused` for a `flash` whose own
/// [`FlashDecision::outcome`] is `PredicateOutcome::Holds`, which
/// `FlashDecision::Declined` can never be.
#[allow(clippy::too_many_arguments)]
pub(crate) fn training_attention_cascade(
    qkv: &Tensor,
    batch: usize,
    seq: usize,
    h: usize,
    d: usize,
    masks: TrainingMaskInputs<'_>,
    flash: &FlashDecision,
    rope: &RopeCtx<'_>,
    window: Option<LocalWindow>,
    policy: FullyMaskedPolicy,
    on_flash_fused: impl FnOnce(&CompactedBatch) -> Result<Tensor, EncoderError>,
) -> Result<Tensor, EncoderError> {
    // Flash cascade: reported here for EVERY caller (contract shared
    // vocabulary — "never silent"), even one (BERT/DistilBERT) whose own
    // `flash` is always `Declined { CapabilityMiss, "flash_transport_not_wired" }`.
    let flash_dispatch = admit_cascade(
        admission_mode(),
        "attention_block_flash",
        flash.reason(),
        flash.outcome(),
        true,
        cascade_counters_for("attention_block_flash"),
    )?;
    if flash_dispatch == CascadeOutcome::Fused {
        let admission = match flash {
            FlashDecision::Fused(batch) => batch,
            FlashDecision::Declined { outcome, reason } => {
                return Err(EncoderError::Config(format!(
                    "attention_block_flash dispatched Fused but flash itself is \
                     Declined(outcome={outcome:?}, reason={reason}) -- admit_cascade and \
                     FlashDecision disagree"
                )))
            }
        };
        return on_flash_fused(admission);
    }

    // Memeff cascade: consulted BEFORE the block arm's own `admit()` (never
    // through it) — see `mem_efficient_attention_predicate`'s doc.
    let (memeff_outcome, memeff_reason) =
        mem_efficient_attention_predicate(qkv.device(), qkv.dtype(), seq, flash);
    let memeff_dispatch = admit_cascade(
        admission_mode(),
        "mem_efficient_attention",
        memeff_reason,
        memeff_outcome,
        true,
        cascade_counters_for("mem_efficient_attention"),
    )?;
    if memeff_dispatch == CascadeOutcome::Fused {
        return forward_memeff_attention(
            qkv,
            batch,
            seq,
            h,
            d,
            masks.extended,
            rope,
            window.map(|w| w.half_window),
            policy,
        );
    }

    if window.is_some() && masks.local_band.is_none() {
        return Err(EncoderError::Config(
            "local-attention layer reached without a sliding-window band".into(),
        ));
    }
    let Some(fused) = masks.fused else {
        return Err(EncoderError::Config(
            "training-mode attention fell through to the block/eager arm without the \
             per-forward fused masks -- the caller builds them once per forward whenever \
             memeff will not handle it (mem_efficient_attention_predicate declined here too); \
             a direct caller in training mode must supply them on this path"
                .into(),
        ));
    };

    let (holds, predicate) = attention_block_admission_predicate(
        qkv,
        seq,
        h,
        d,
        &fused.global,
        window.is_some(),
        fused.local.as_ref(),
    );
    let outcome = admit(
        admission_mode(),
        "attention_block_fused",
        predicate,
        holds,
        *ATTENTION_BLOCK_DISPATCH_COUNTERS,
    )?;
    match outcome {
        DispatchOutcome::Fused => {
            let qkv5 = qkv.reshape((batch, seq, 3, h, d))?;
            let mask = match (window.is_some(), fused.local.as_ref()) {
                (true, Some(local)) => local,
                (true, None) => {
                    return Err(EncoderError::Config(
                        "local-attention layer reached without a combined fused mask".into(),
                    ))
                }
                (false, _) => &fused.global,
            };
            let op = AttentionBlockFused::new(1.0 / (d as f32).sqrt(), policy, rope.enabled)?;
            Ok(apply3(&qkv5, rope.pack, mask, op)?)
        }
        DispatchOutcome::Eager => forward_eager_training_attention_composition(
            qkv,
            batch,
            seq,
            h,
            d,
            masks.extended,
            masks.local_band,
            rope,
            window,
            policy,
            true,
        ),
    }
}
