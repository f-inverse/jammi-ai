//! The per-layer fused-attention cascade — `flash (caller-supplied) →
//! mem_efficient_attention → attention_block_fused → eager composition
//! (with `softmax_last_dim_fused` admission)` — shared by every encoder
//! whose attention arm wants it, extracted from `crate::modernbert` (issue
//! #462): every pure predicate and per-layer numeric composition moves
//! verbatim into this module, while the flash TRANSPORT protocol (dense/
//! ragged FlashAttention-2 call sites, `unpad_rows`/`repad_rows`,
//! `decide_flash_admission`) stays an encoder-boundary concern owned by
//! `crate::modernbert` — see "What moved here, and what did not" below for
//! the exact line.
//!
//! ## What moved here, and what did not
//!
//! Every PURE predicate and every per-layer numeric composition
//! (`attention_block_admission_predicate`, `mem_efficient_attention_predicate`,
//! `softmax_admission_predicate` + `softmax_apply_training`,
//! [`FusedAttentionMasks`]/[`TrainingMaskInputs`], `forward_memeff_attention`,
//! `forward_eager_training_attention_composition`, and the top-level
//! [`training_attention_cascade`] dispatcher) moved verbatim (numerics
//! unchanged) from `crate::modernbert`, along with the flash cascade's own
//! OUTCOME vocabulary ([`FlashDecision`]/[`CompactedBatch`] — audit round
//! item 10: this generic vocabulary lives with the generic seam even
//! though only `crate::modernbert` constructs a real
//! [`FlashDecision::Fused`] today). The flash TRANSPORT protocol itself —
//! the dense/ragged/padded FlashAttention-2 call sites, `unpad_rows`/
//! `repad_rows`, and `decide_flash_admission` (the code that actually
//! POPULATES a `CompactedBatch`) — stays in `crate::modernbert`: it is an
//! encoder-BOUNDARY concern (a whole-forward compaction decision, not a
//! per-layer arm), and cannot be delivered through a per-layer seam (R1'
//! ruling). This module only ever *reports* the flash decision a caller
//! hands it (`admit_cascade("attention_block_flash", ..)`, so the counter
//! fires for every caller, including one that never transports) and, when
//! that decision is `Fused`, delegates to the caller's own transport via
//! [`training_attention_cascade`]'s `on_flash_fused` callback.
//!
//! ## `RopeCtx`: representing "this caller has no RoPE at all", and laziness
//!
//! [`AttentionBlockFused`]/[`MemEfficientAttention`] are `CustomOp3`s: a
//! `rope_pack` tensor argument is REQUIRED at every call, whether or not
//! `rope` (construction data, a `bool`) is `true` — when it is `false` the
//! argument is present but never read (see either op's own module doc's
//! "rope_pack ... only when rope == true" section). A caller with no
//! positional-embedding table at all (BERT/DistilBERT — absolute position
//! embeddings are baked into the embeddings sum, never applied per layer)
//! therefore still needs SOME tensor to hand the fused arm, even though it
//! is never read. [`RopeCtx::Disabled`] represents this exactly: its
//! `placeholder` is a per-module cached tensor (allocated ONCE, at
//! construction, never read — see `bert::BertSelfAttention`'s own
//! `rope_placeholder` field), never re-derived per forward. This is a
//! narrower, type-safe restatement of plan v2's "`rope: None` with a
//! per-module cached `[2,1,1,64]` placeholder": the cascade's own functions
//! take `&RopeCtx` (never `Option<&RopeCtx>`) so the placeholder-vs-no-op
//! choice is always explicit at the call site, and the CustomOp3 arity
//! constraint above is satisfied by construction rather than by an
//! `Option::unwrap_or` at the one call site that would need it.
//!
//! [`RopeCtx::Enabled`]'s `pack` field is a PROVIDER (`&dyn Fn() -> ..`),
//! not an already-materialized `&Tensor` (audit round item 5): main's own
//! placement called `RotaryEmbedding::cached_rope_pack` ONLY inside the
//! memeff arm and inside the block-fused arm's `DispatchOutcome::Fused`
//! branch — never for a flash dispatch, never for the eager arm (which
//! rotates Q/K directly via [`RopeCtx::Enabled`]'s `apply`, no pack tensor
//! involved at all). Since a `RopeCtx` must be built BEFORE calling
//! [`training_attention_cascade`] (whose flash check runs first and may
//! return before `rope` is ever consulted), an eagerly-materialized `pack`
//! field would force that computation — and any error or lock it can
//! raise — onto every dispatch, including the ones that never asked for
//! it. The provider shape restores main's exact laziness: [`RopeCtx::pack`]
//! is called only from [`forward_memeff_attention`] and the block-fused
//! arm, in the same order main did.
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

/// The once-per-forward flash-cascade decision (contract v4 §3.2), decided
/// ONCE by a caller's own whole-forward entry point (mirroring
/// [`FusedAttentionMasks`]) and threaded per layer into
/// [`training_attention_cascade`]. Owns the compacted batch's row
/// `lengths` and the `[total]` unpad gather indices, but deliberately NOT
/// a constructed `jammi_kernels::flash::CuSeqlens`: that type is
/// feature-gated behind `jammi-kernels`'s `flash-attn` (not forwarded by
/// this crate's `Cargo.toml`), and `CuSeqlens::from_lengths` is cheap
/// enough to construct on demand, once, at the real flash call site —
/// holding a `CuSeqlens` across a whole forward buys nothing and would
/// force a premature feature dependency. Moved here from `crate::modernbert`
/// (issue #462 fix round, item 10): the flash cascade's OUTCOME vocabulary
/// is generic to the shared seam even though only `crate::modernbert`
/// constructs a real [`FlashDecision::Fused`] today (BERT/DistilBERT
/// always supply [`FlashDecision::Declined`]) — the fields stay
/// `pub(crate)` so `crate::modernbert`'s own flash TRANSPORT (which stays
/// there — an encoder-boundary concern, see this module's doc) can still
/// construct and read them directly.
pub(crate) struct CompactedBatch {
    /// One length per batch element, `lengths[b] <= seq`. Consumed by
    /// `crate::modernbert::ModernBertAttention::forward_flash_dense_attention`
    /// (`CuSeqlens::from_lengths`) and
    /// `crate::modernbert::ModernBertAttention::forward_flash_ragged_attention`
    /// (`flash_attention_varlen_with_rope_ragged`'s own `lengths`
    /// parameter) under the `flash-attn` feature; on a plain build the
    /// field is only read by `crate::modernbert`'s own tests, so
    /// `#[allow(dead_code)]` stays even though it is no longer
    /// unconditionally dead.
    #[allow(dead_code)]
    pub(crate) lengths: Vec<usize>,
    /// `[total]` gather indices into the flattened `[batch * seq]` row
    /// axis — every REAL (non-pad) row, batch-then-seq order. Consumed by
    /// `crate::modernbert::ModernBert::forward_hidden_with_lengths`'s
    /// encoder-boundary transport (P6 Stage B B3-padded, contract v4 §3.5):
    /// `unpad_rows` once before layer 0, `repad_rows` once after the last
    /// layer — the DENSE arm never reads this field (dense skips
    /// compaction entirely, see [`Self::is_dense`]/
    /// `crate::modernbert::decide_flash_admission`'s doc).
    pub(crate) gather_indices: Tensor,
    /// Same status as `lengths`: the padded/ragged arm's own production
    /// consumer, `#[allow(dead_code)]` on a plain (non-`flash-attn`) build.
    #[allow(dead_code)]
    pub(crate) total: usize,
    /// The batch's own (padded) sequence length — `mask.dim(1)` at the
    /// point `crate::modernbert::decide_flash_admission` decided this
    /// batch. NOT `lengths.iter().max()`: a shorter `seq` would silently
    /// narrow the RoPE table
    /// `crate::modernbert::ModernBertAttention::forward_flash_ragged_attention`
    /// gathers from below what a full-length row actually needs. Same
    /// `#[allow(dead_code)]` status as `lengths`/`total`.
    #[allow(dead_code)]
    pub(crate) seq: usize,
    /// `lengths.iter().all(|&l| l == seq)` (contract v4 delta 4's
    /// discriminator — NEVER `total == batch * seq`, which a genuinely
    /// padded-but-numerically-coincidental batch could also satisfy),
    /// computed ONCE at construction (see
    /// `crate::modernbert::build_flash_forward_decision`) rather than
    /// re-derived at each of this field's several call sites
    /// ([`FlashDecision::reason`], `crate::modernbert::ModernBertAttention::forward`'s
    /// transport branch, `crate::modernbert::ModernBert::forward_hidden_with_lengths`'s
    /// own transport decision) — one source of truth for a fact three
    /// different call sites need to agree on.
    pub(crate) is_dense: bool,
}

/// The full once-per-forward flash-cascade decision, decided ONCE by a
/// caller's whole-forward entry point (mirroring [`FusedAttentionMasks`])
/// and threaded per layer into [`training_attention_cascade`] — every
/// LAYER's own `admit_cascade` call reports against it (contract v4 §3.2:
/// "the counters are per-dispatch, not per-forward" — this type is what
/// makes that per-layer call cheap: no layer re-derives the outcome/reason).
///
/// Two variants, not a `CompactedBatch`/`outcome`/`reason` struct with the
/// first field optional: the prior shape let `outcome == Holds` and
/// `admission == None` be constructed simultaneously — an invalid state a
/// caller's own dispatch code had to guard with a RUNTIME `ok_or_else` (a
/// string-message fallback for a state the type itself should have
/// refused to represent). This enum makes that state a COMPILE ERROR
/// instead: [`Self::Fused`] always carries its [`CompactedBatch`],
/// [`Self::Declined`] never does — no runtime check stands between "the
/// cascade decided Fused" and "a `CompactedBatch` exists".
///
/// [`Self::outcome`]/[`Self::reason`] recover the [`PredicateOutcome`] /
/// reason string every `admit_cascade` call site still needs (`Fused`
/// always reports `Holds`, and — contract v4 delta 4 — a PER-VARIANT
/// truthful reason: `"domain_ok_dense"` when [`CompactedBatch::is_dense`],
/// `"domain_ok_padded"` otherwise; see [`Self::reason`]), so callers built
/// around the old struct's two bare fields keep exactly the same call
/// shape.
pub(crate) enum FlashDecision {
    /// The batch is flash-eligible — `attention_block_flash` dispatches
    /// `Fused`. DENSE (`admission.is_dense`,
    /// `crate::modernbert::ModernBertAttention::forward_flash_dense_attention`
    /// runs, hidden stays `[batch, seq, hidden]`, no transport) or
    /// genuinely PADDED (`!admission.is_dense`,
    /// `crate::modernbert::ModernBertAttention::forward_flash_ragged_attention`
    /// runs, `crate::modernbert::ModernBert::forward_hidden_with_lengths`
    /// unpads hidden to `[total, hidden]` once before layer 0 and repads
    /// it once after the last — P6 Stage B B3-padded, contract v4 §3.1's
    /// item 1). Either way this variant always carries its
    /// `CompactedBatch` — see `crate::modernbert::build_flash_forward_decision`,
    /// the only constructor.
    Fused(CompactedBatch),
    /// The cascade declines — `outcome`/`reason` are whatever
    /// `crate::modernbert::flash_admission_predicate` (or
    /// `crate::modernbert::decide_flash_admission`'s own `op_disabled`
    /// short-circuit) determined. NEVER carries a `CompactedBatch`: that
    /// batch is out of THIS decision's scope once declined, so
    /// `crate::modernbert::ModernBert::forward_hidden_with_lengths` never
    /// transports and every layer runs the padded `[batch, seq, hidden]`
    /// block/eager arm exactly as before this seam existed.
    Declined {
        outcome: PredicateOutcome,
        reason: &'static str,
    },
}

impl FlashDecision {
    /// The [`PredicateOutcome`] every `admit_cascade` call site reports —
    /// `Holds` for [`Self::Fused`] (the only outcome that variant can mean),
    /// whatever [`Self::Declined`] itself carries otherwise.
    pub(crate) fn outcome(&self) -> PredicateOutcome {
        match self {
            FlashDecision::Fused(_) => PredicateOutcome::Holds,
            FlashDecision::Declined { outcome, .. } => *outcome,
        }
    }

    /// The reason string every `admit_cascade` call site reports —
    /// per-variant truthful for [`Self::Fused`] (contract v4 delta 4:
    /// `"domain_ok_dense"` must not describe a padded dispatch, and vice
    /// versa): `"domain_ok_dense"` when the carried `CompactedBatch` is
    /// dense, `"domain_ok_padded"` when it is genuinely padded. Whatever
    /// [`Self::Declined`] itself carries otherwise.
    pub(crate) fn reason(&self) -> &'static str {
        match self {
            FlashDecision::Fused(batch) if batch.is_dense => "domain_ok_dense",
            FlashDecision::Fused(_) => "domain_ok_padded",
            FlashDecision::Declined { reason, .. } => reason,
        }
    }
}

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

/// A local-attention layer's marker — `Some(LocalWindow)` replaces
/// `is_local == true` (ModernBERT's local layers); `None` replaces
/// `is_local == false` (ModernBERT's global layers, and EVERY BERT/
/// DistilBERT layer — neither architecture has a sliding-window concept).
/// A fieldless marker, not a `half_window: usize` carrier (audit round
/// item 3): the actual window WIDTH the block/eager arms need lives in
/// the already-built band tensor
/// ([`TrainingMaskInputs::local_band`]/[`FusedAttentionMasks::local`])
/// those arms consume, never in this type — [`forward_memeff_attention`]'s
/// own `half_window: Option<usize>` parameter is a SEPARATE,
/// independently-sourced value (see [`training_attention_cascade`]'s doc
/// for why the two must never be derived from each other).
#[derive(Debug, Clone, Copy)]
pub(crate) struct LocalWindow;

/// [`RopeCtx::Enabled`]'s eager-rotation function-pointer type, named so
/// its own field declaration reads as one bounded-complexity type rather
/// than the raw `&dyn Fn(..) -> Result<..>` spelled out inline (clippy's
/// `type_complexity` lint).
type RopeApplyFn<'a> = dyn Fn(&Tensor) -> Result<Tensor, EncoderError> + 'a;

/// [`RopeCtx::Enabled`]'s rope-PACK-materializing function-pointer type —
/// a PROVIDER, not an already-materialized `&'a Tensor` (audit round item
/// 5): the pack is a real per-forward computation
/// (`RotaryEmbedding::cached_rope_pack`, memoized per dtype but still a
/// lock + possible first-call `Tensor::stack`), and main's own placement
/// computed it ONLY inside the arm that actually consumes it (the memeff
/// arm, and the block-fused arm's `DispatchOutcome::Fused` branch) —
/// never for a flash dispatch and never for the eager arm. A caller must
/// still build a `RopeCtx` BEFORE calling [`training_attention_cascade`]
/// (whose flash check runs first and may return before ever touching
/// `rope`), so `pack` cannot be an already-materialized `&Tensor` without
/// forcing that materialization ahead of the flash check every time.
/// Wrapping it as a zero-argument provider keeps construction itself
/// free — the provider is only ever CALLED from [`forward_memeff_attention`]
/// and the block-fused arm inside [`training_attention_cascade`], exactly
/// where main called `cached_rope_pack` — while [`apply_rope`] (the eager
/// arm's own consumer) never calls it at all.
type RopePackFn<'a> = dyn Fn() -> Result<Tensor, EncoderError> + 'a;

/// The per-layer RoPE context the cascade's fused/eager arms need. Two
/// variants, not a `pack`/`enabled`/`apply` struct with `apply: Option`
/// (audit round item 2): the prior shape let `enabled: true, apply: None`
/// be constructed simultaneously — an invalid state [`apply_rope`] had to
/// silently resolve to "no rotation" at runtime rather than the type
/// itself refusing to represent it (`crate::modernbert`'s own
/// `forward_memeff_attention` test wrapper constructed exactly this state
/// before this fix, since that arm never calls [`apply_rope`] at all and
/// so never noticed). This enum makes the state a COMPILE ERROR instead:
/// [`Self::Enabled`] always carries its `apply` function alongside `pack`,
/// [`Self::Disabled`] carries neither.
pub(crate) enum RopeCtx<'a> {
    /// No RoPE at all (BERT/DistilBERT: absolute position embeddings are
    /// summed into the input once, never applied per layer). `placeholder`
    /// is a per-module cached tensor — allocated ONCE, at construction,
    /// never read by the fused ops (`rope: false` there) and never
    /// re-derived per forward; see [`Self::pack`].
    Disabled { placeholder: &'a Tensor },
    /// RoPE is genuinely applied. `pack` is called ONLY from
    /// [`forward_memeff_attention`] and [`training_attention_cascade`]'s
    /// block-fused arm (see [`RopePackFn`]'s doc for why it is a provider,
    /// not an already-materialized tensor); `apply` rotates a Q/K tensor
    /// in the eager composition, consulted only there.
    Enabled {
        pack: &'a RopePackFn<'a>,
        apply: &'a RopeApplyFn<'a>,
    },
}

impl<'a> RopeCtx<'a> {
    /// The op-facing rotary pack: [`Self::Disabled`]'s placeholder
    /// (infallible, a cheap `Tensor::clone` — candle's `Tensor` shares its
    /// underlying storage `Arc`, so this never copies data) or
    /// [`Self::Enabled`]'s provider, CALLED here (see [`RopePackFn`]'s
    /// doc for why laziness matters: calling this from the flash arm or
    /// the eager arm would materialize/error on the pack where main never
    /// did).
    fn pack(&self) -> Result<Tensor, EncoderError> {
        match self {
            RopeCtx::Disabled { placeholder } => Ok((*placeholder).clone()),
            RopeCtx::Enabled { pack, .. } => pack(),
        }
    }

    /// The fused ops' own `rope: bool` construction data.
    fn enabled(&self) -> bool {
        matches!(self, RopeCtx::Enabled { .. })
    }
}

fn apply_rope(rope: &RopeCtx<'_>, x: &Tensor) -> Result<Tensor, EncoderError> {
    match rope {
        RopeCtx::Disabled { .. } => Ok(x.clone()),
        RopeCtx::Enabled { apply, .. } => apply(x),
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
/// `self.half_window`/the hardcoded `FullyMaskedPolicy::Zeros`. `rope.pack()`
/// is called here, matching main's own placement exactly (audit round item
/// 5) — see [`RopePackFn`]'s doc.
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
    let rope_pack = rope.pack()?;
    let key_mask = extended_mask_f32.to_dtype(qkv.dtype())?;
    let op = MemEfficientAttention::new(
        1.0 / (d as f32).sqrt(),
        policy,
        rope.enabled(),
        half_window,
        MEM_EFFICIENT_CHUNK,
    )
    .map_err(|e| EncoderError::Config(format!("mem_efficient_attention: {e}")))?;
    mem_efficient_attention(&qkv5, &rope_pack, &key_mask, op)
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
/// body, parameterized by `rope`/`window`/`half_window`/`policy`/`flash` in
/// place of `self.rope`/`self.is_local`+`self.half_window`/the hardcoded
/// `FullyMaskedPolicy::Zeros`/the method's own `flash: &FlashDecision`
/// argument (already a parameter before this move).
///
/// `window` and `half_window` are TWO SEPARATE parameters, not one
/// (audit round item 3): `window: Option<LocalWindow>` is consulted ONLY
/// as `is_some()` — the block/eager arms' "is this layer local" fact,
/// which gates whether they combine in the sliding-window band `masks`
/// already carries. `half_window: Option<usize>` is the RAW scalar
/// [`forward_memeff_attention`] needs, and it is passed straight through
/// UNTOUCHED to that call below — never derived from `window` (main's own
/// `forward_memeff_attention` read `self.half_window` directly, with no
/// `is_local` coupling and no `unwrap_or` substitution; a caller that
/// re-derived `half_window` from `window.map(|w| w.half_window)` here
/// would silently turn "half_window is `None`" into "half_window is
/// `Some(0)`" whenever `window.is_some()`, a confident-wrong scalar,
/// or silently drop a real `half_window` whenever `window` and
/// `half_window` briefly disagree). Every real caller keeps them in
/// lockstep at construction (see `crate::modernbert`'s own attention
/// constructor's doc for where `is_local`/`half_window` are set together),
/// but this cascade never assumes that lockstep to derive one from the
/// other.
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
    half_window: Option<usize>,
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
            half_window,
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
            // `rope.pack()` is called HERE, inside the block-fused arm,
            // matching main's own placement exactly (audit round item 5)
            // — see `RopePackFn`'s doc: neither the flash arm above nor
            // the eager arm below ever materializes or errors on it.
            let rope_pack = rope.pack()?;
            let mask = match (window.is_some(), fused.local.as_ref()) {
                (true, Some(local)) => local,
                (true, None) => {
                    return Err(EncoderError::Config(
                        "local-attention layer reached without a combined fused mask".into(),
                    ))
                }
                (false, _) => &fused.global,
            };
            let op = AttentionBlockFused::new(1.0 / (d as f32).sqrt(), policy, rope.enabled())?;
            Ok(apply3(&qkv5, &rope_pack, mask, op)?)
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
