//! Fused masked softmax-last-dim forward + backward, with the additive
//! mask ADD folded in.
//!
//! `y = softmax(scores + mask, last dim)` — reduced over the LAST
//! dimension; `scores` is `[<any leading dims>, last]` (any rank >= 1,
//! `last` the softmax/reduction axis), `mask` is an ADDITIVE bias
//! broadcasting onto `scores` per the "supported mask broadcast class"
//! below. This replaces the `[broadcast_add(mask) -> softmax]` tail
//! `jammi-encoders`' ModernBERT attention call site composes today
//! (`candle_nn::ops::softmax(&scores.broadcast_add(&mask)?, D::Minus1)`,
//! see `modernbert.rs`'s `ModernBertAttention::forward`).
//!
//! ## Why fold the mask in at all (the memory lever, not just fewer ops)
//!
//! ModernBERT's attention scores are `[batch, heads, seq, seq]` —
//! quadratic in `seq`, and the single largest retained-tape tensor in the
//! whole forward (~1.4GB at seq 128, ~22GB at seq 1024, batch 8 — the
//! fused-kernels plan's profile). The eager composition
//! (`candle_nn::ops::softmax`: `max_keepdim`, `broadcast_sub`, `exp`,
//! `sum_keepdim`, `broadcast_div`) retains EVERY one of those intermediates
//! at `[batch, heads, seq, seq]` on the backward tape, because each is a
//! plain `Tensor` op with its own `Op::*` graph node. [`SoftmaxLastDimFused`]
//! collapses that whole chain into ONE graph node (`Op::CustomOp2`), and its
//! `bwd` (see below) needs only `y` itself (already tape-resident — the
//! V-matmul needs it too) and `grad_res` — no `[B,H,S,S]`-shaped intermediate
//! survives the forward pass at all. `tests::fused_softmax_retains_fewer_tape_nodes_than_eager`
//! measures this directly via `Tensor::sorted_nodes()` (candle's own public
//! topological-sort-for-backward API): the real VRAM number is the lead's
//! pod A/B, but the NODE-COUNT reduction this claim rests on is measured
//! here, live, on CPU.
//!
//! ## The supported mask broadcast class (family D)
//!
//! `scores` and `mask` must have the SAME RANK (no implicit NumPy-style
//! leading-dimension padding — an equal-rank requirement checked directly,
//! not inferred). The LAST axis (the reduction axis) must match EXACTLY
//! between the two (`mask`'s last-axis value is a real per-key additive
//! bias, not something this op ever broadcasts away) — this is
//! deliberately narrower than a fully general broadcast rule, and it is
//! sufficient for every call site this crate ships. Every OTHER axis (all
//! but the last) may be either `1` (broadcasts: `mask`'s single value along
//! that axis applies to every position `scores` has there) or exactly equal
//! to `scores`'s corresponding axis. This covers BOTH shapes ModernBERT's
//! attention call site actually produces after this commit's call-site
//! restructuring (see `modernbert.rs`'s `ModernBertAttention::forward`):
//! the padding mask alone, `[batch, 1, 1, seq]` (broadcasts over `heads`
//! AND the query-row axis), and the padding-mask-plus-sliding-band SUM
//! precomputed by the call site BEFORE reaching this op, `[batch, 1, seq,
//! seq]` (broadcasts only over `heads`) — both against `scores`'s `[batch,
//! heads, seq, seq]`. If a caller's mask shape falls outside this class
//! (different rank, or a leading axis that is neither `1` nor equal), this
//! op REFUSES (`Error::ShapeMismatchBinaryOp`) rather than guessing at a
//! broadcast — the call site's own admission predicate is what turns that
//! refusal into a counted eager fallback (K2), matching the RoPE/LayerNorm
//! precedent's "validate, don't silently degrade" doctrine.
//!
//! Unlike [`crate::ops::rope`]'s single `period` scalar (sound there only
//! because RoPE's callers broadcast over exactly ONE axis, and `rope_dims`'s
//! own doc derives the extra "axis immediately before hidden" condition
//! needed to make a single modulus safe), this op's mask can broadcast over
//! ANY SUBSET of the leading axes independently. [`mask_row_offset`]
//! therefore does not attempt to reduce that to a scalar modulus at all: it
//! performs a full, exact multi-index unravel (`row` against `scores`'s own
//! leading-axis shape) followed by a re-ravel into `mask`'s leading-axis
//! shape (substituting index `0` on every axis where `mask`'s size is `1`).
//! This is provably correct for the general N-axis case by construction,
//! not a shortcut whose soundness depends on an extra side condition the
//! way RoPE's `period` trick does.
//!
//! ## Why the mask is folded in BEFORE this op runs, not by it
//!
//! ModernBERT's attention adds up to TWO additive masks onto `scores`
//! today: the padding mask (always) and, for a local-attention layer, the
//! sliding-window band (`crate::mask::sliding_window_mask` in
//! `jammi-encoders`). This op is a `CustomOp2` — TWO tensor arguments,
//! `scores` and ONE mask — so a local layer's call site combines its two
//! masks into a single small tensor (`extended_mask.broadcast_add(&band)`,
//! at most `[batch, 1, seq, seq]` — NEVER `[batch, heads, seq, seq]`, since
//! neither mask ever carries a `heads` axis) BEFORE calling this op, on the
//! TRAINING arm only (see `modernbert.rs`'s doc for why eval's numeric path
//! is untouched: floating-point addition is not associative, so combining
//! the two masks first is not bit-identical to the eager composition's
//! sequential `broadcast_add`s, even though it is algebraically
//! equivalent — exactly the same "different-but-equivalent, own tolerance
//! oracle" shape this crate's other fused ops take on their training arm).
//!
//! ## Extreme-value domain (family D)
//!
//! This op's forward is `max`-then-`exp`-then-normalize, the SAME shape
//! `candle_nn::ops::softmax` composes (`max_keepdim`, `broadcast_sub`,
//! `exp`, `sum_keepdim`, `broadcast_div`), for every row that has at least
//! one attendable (unmasked) position:
//! - A row containing one (or more, but not all) masked entries (`mask[i] =
//!   -inf`, or the real finite `MASKED_LOGIT`): the masked position
//!   contributes `~0` to the row's normalization (`exp(-inf - max) = 0`
//!   exactly; a finite deeply-negative entry underflows to `~0` after the
//!   exponential), matching eager.
//! - Large positive `scores` values: no overflow, exactly because of the
//!   max-subtraction (`exp` never sees an argument larger than `0`) — the
//!   textbook reason any softmax implementation subtracts the row max
//!   first, reproduced here rather than skipped for a false simplicity
//!   gain.
//!
//! ## Fully-masked row: safe-softmax zeros, an INTENTIONAL divergence from `candle_nn::ops::softmax`
//!
//! CORRECTED (an audit finding — the previous wording here, inherited from
//! a since-corrected claim in `jammi_encoders::mask`, was FALSE):
//! `jammi-encoders`' actual call site DOES construct a fully-masked row in
//! production once padding is present. `sliding_window_mask`'s band ALONE
//! always keeps a query's own diagonal in-window, but a query that is
//! ITSELF a pad token has that diagonal KEY also masked by the padding
//! mask (which depends only on key position, not query position — see
//! `jammi_encoders::mask::extended_attention_mask`'s doc). A trailing pad
//! run longer than the local layer's `half_window` makes EVERY key in a
//! deep pad query's window a pad key too, so band+padding together produce
//! a genuinely fully-masked row — routine for a short sequence padded to a
//! fixed length in one of ModernBERT's local-attention layers
//! (`local_attention = 128` → `half_window = 64`; any batch element with
//! more than ~64 trailing pads hits this), not a hypothetical.
//!
//! What should this op DO on such a row? Two candidate references
//! disagree, and this op deliberately follows the one that is NOT this
//! crate's own eager composition:
//! - `candle_nn::ops::softmax` computes NATIVELY in the tensor's own dtype
//!   throughout (`broadcast_add`, `max_keepdim`, `broadcast_sub`, `exp`,
//!   `sum_keepdim`, `broadcast_div` — no internal F32 upcast). On a row
//!   whose every entry is `-inf`, that produces `NaN` (`max = -inf`,
//!   `diff = (-inf) - (-inf) = NaN`). On a row whose every entry is the
//!   REAL finite `MASKED_LOGIT = -10_000.0`: in BF16, whose ULP near
//!   magnitude `10_000` is ≈`64` — far larger than any real `scores[i]`
//!   (`O(1)`–`O(10)`) — `bf16(scores[i] - 10_000.0)` rounds to the SAME
//!   value regardless of `i`, ANNIHILATING the score; the resulting
//!   uniform post-add row makes `candle_nn::ops::softmax`'s BF16 output a
//!   UNIFORM `1/n` distribution. In F32 the same row does NOT annihilate
//!   (F32's ULP near `10_000` is ≈`0.001`), so eager's F32 output there is
//!   `≈softmax(scores)`, not uniform — three DIFFERENT eager behaviors
//!   (NaN / uniform / near-normal) depending on dtype and sentinel, none
//!   of them a considered design for "no attendable key exists here".
//! - Production attention kernels treat this case explicitly:
//!   PyTorch's `_safe_softmax` (`aten/src/ATen/native/transformers/attention.cpp`,
//!   the composite/math SDPA backend: a row where every position is masked
//!   is detected and its output forced to all-zero rather than computing
//!   `NaN`) and FlashAttention-2's online-softmax (`softmax.h`: the
//!   reciprocal row-sum used to normalize is forced to `1` whenever the
//!   accumulated sum is `0`, and the unnormalized accumulator for such a
//!   row is `0`, i.e. the output is `0`, with the row's log-sum-exp
//!   reported as `+inf` to flag it) both converge on the SAME answer:
//!   **zero**, not `NaN`, and not a uniform distribution.
//!
//! This op CAN follow the production-kernel convention instead of
//! `candle_nn::ops::softmax`'s — but only when its caller EXPLICITLY asks
//! for it via [`FullyMaskedPolicy::Zeros`] (see that type's own doc for
//! why this is CONSTRUCTION DATA, not an unconditional behavior this op
//! imposes on every caller): `SoftmaxLastDimFused { fully_masked:
//! FullyMaskedPolicy::Zeros }` outputs ALL ZEROS on a fully-masked row,
//! for every dtype and every masking convention (`-inf` or finite);
//! `SoftmaxLastDimFused::default()` (`FullyMaskedPolicy::Propagate`)
//! reproduces `candle_nn::ops::softmax` EXACTLY there instead (dtype-and-
//! sentinel-dependent `NaN`/uniform/near-normal, per the bullets above).
//! `ModernBertAttention`'s training arm is the ONE call site in this
//! crate that constructs `Zeros` — an INTENTIONAL, disclosed divergence
//! from the eager composition it otherwise replaces, justified by that
//! call site's own masking convention (below) — the fused-vs-eager
//! oracles for `Zeros` below RECORD eager's (dtype-dependent,
//! NaN-or-uniform) output as the KNOWN-DIVERGENT baseline on this one
//! input class, rather than asserting equality to it; the oracles for
//! `Propagate` assert equality to eager, unconditionally, on the same
//! fixtures.
//!
//! **Detection** ([`row_is_fully_masked`], consulted ONLY under `Zeros`):
//! this crate's additive-mask convention (`jammi_encoders::mask`) uses
//! `0.0` (exactly) as the "unmasked" identity value and any value `< 0.0`
//! (`MASKED_LOGIT`, `2 * MASKED_LOGIT` when two masks compose, or `-inf`)
//! as "some degree of masking" — never a mix of positive values and this
//! convention. A row is fully masked iff its OWN mask values (not `scores
//! + mask`) contain no exact `0.0`, i.e. `max_i mask[i] < 0.0`. This is
//! checked on `mask` ALONE, before ever reading `scores` — exact for both
//! real call-site shapes (`[batch,1,1,seq]` padding-only and
//! `[batch,1,seq,seq]` padding-plus-band) — but it is NOT a universal
//! truth about additive masking: a caller using a uniformly-negative
//! additive bias that is not a masking sentinel at all (e.g. a
//! distance-decay attention bias, never exactly `0.0` anywhere by design)
//! would have EVERY row misclassified as "fully masked" under this rule.
//! That is exactly why [`FullyMaskedPolicy`] exists as an explicit,
//! caller-asserted opt-in rather than an unconditional behavior: a caller
//! whose convention does not satisfy this premise simply never requests
//! `Zeros`, and gets `Propagate`'s eager-exact behavior instead.
//!
//! **Inert downstream in BOTH directions, verified**: a fully-masked row
//! only arises at a query position that is ITSELF padding (see above).
//! FORWARD: `jammi_encoders::pooling`'s `mean_pool`/`weighted_mean_pool`
//! (`pooling.rs`'s `hidden.broadcast_mul(&mask...)`, keyed off the REAL
//! `[batch, seq]` attention mask, not this op's internal one) discard a
//! pad position's hidden state PROVIDED it is FINITE (`0.0 * v == 0` for
//! any finite `v`) — a `NaN` there is NOT discarded by a multiply (`0.0 *
//! NaN == NaN` in IEEE754) and would poison the whole pooled sum instead;
//! `max_pool` (`where_cond` substituting a sentinel there) is the one that
//! genuinely survives `NaN`, since it SELECTS rather than multiplies. Both
//! `FullyMaskedPolicy` variants this section discusses leave a FINITE
//! value at this row (`Zeros`: exact `0.0`; `Propagate` on the REAL
//! finite-`MASKED_LOGIT` convention: a finite annihilated-uniform or
//! near-normal result, never `NaN`), so this distinction does not bite in
//! practice for either policy on ModernBERT's actual masking convention —
//! it would only matter for a hypothetical `Propagate` caller using the
//! synthetic all-`-inf` convention feeding into `mean_pool`, which is
//! exactly why `jammi_encoders::mask::MASKED_LOGIT` is finite, not `-inf`
//! (see that constant's own doc). BACKWARD: because pooling ZEROES the
//! pad position's contribution to the pooled loss, the gradient flowing
//! BACK INTO that softmax row (`dy`) is exactly `0.0` too, so `dscores =
//! (dy - sum(dy*y)) * y` becomes `(0 - sum(0*y)) * y`. For `Zeros` (`y ==
//! 0`, always finite) this is exactly `0`. For `Propagate` on the REAL
//! finite-`MASKED_LOGIT` shape (`y` is finite — never `NaN`, since
//! `MASKED_LOGIT` never produces one), `sum(0*y) == 0` exactly and `0 * y
//! == 0` exactly too (a finite value times `0` is `0` in IEEE754) — so
//! `Propagate` ALSO yields an exactly-zero gradient for this row on
//! ModernBERT's actual masking convention: the two policies' TRAINING
//! DYNAMICS through a pad-query row are IDENTICAL here, not merely close.
//! This equivalence does NOT extend to the synthetic all-`-inf`
//! convention under `Propagate`: there `y` is `NaN` (see the bullets
//! above), and IEEE754 gives `0 * NaN == NaN`, so `dy == 0` does NOT save
//! the product — `Propagate` on an all-`-inf` row poisons its OWN
//! gradient with `NaN` regardless of `dy`, exactly as `candle_nn::ops::softmax`
//! itself would. This is not a gap in this op's behavior; it is `Propagate`
//! faithfully reproducing what `-inf` masking does in eager, `NaN`
//! propagation included.
//!
//! **Backward falls out for free under `Zeros`**: `dscores = (dy -
//! sum(dy*y)) * y` with `y` (this row) `== 0` gives `dscores = (dy - 0) *
//! 0 = 0` — no special case is needed in [`SoftmaxBwdDScores`] at all,
//! under either policy; `bwd` does not even read `self.fully_masked`.
//!
//! For the PARTIALLY-masked case (some, not all, positions masked;
//! independent of `FullyMaskedPolicy`, which only gates the FULLY-masked
//! branch), the BF16-native mask-add rounding [`softmax_row_bf16`]
//! performs (matching `candle_nn::ops::softmax`'s own BF16-native
//! `broadcast_add`) remains correct and is KEPT: primary-source research
//! against the upstream HuggingFace ModernBERT reference
//! (`modeling_modernbert.py` + `masking_utils`'s eager mask path)
//! confirms it ALSO adds its (BF16-typed) mask in BF16 before an F32
//! softmax — the same BF16-native add-then-round this op's BF16 arm
//! reproduces, not a divergence from that specific reference. Only the
//! FULLY-masked case (which the true HF reference never constructs at
//! all — its padding mask is KEY-only, so a pad QUERY row still attends
//! to every REAL key; a fully-masked row is specific to THIS crate's own
//! band+padding combination) is where `FullyMaskedPolicy` matters at all.
//!
//! This op is still tested against genuine `-inf` inputs too (it is a
//! generic primitive per family L — it names no consumer, and a future
//! caller may not share ModernBERT's masking convention at all): the
//! extreme-value tests below exercise the synthetic all-`-inf` shape, the
//! real finite-`MASKED_LOGIT` all-masked-row shape (F32 and BF16, the
//! latter at the production reduction width — see the algorithm section
//! below), and a single (not all) `-inf`/`MASKED_LOGIT` position
//! separately — under BOTH `FullyMaskedPolicy` variants.
//!
//! ## Algorithm choice: classic multi-pass, not online single-pass
//!
//! The fused-kernels plan offers a choice ("online single-pass max+sum
//! (Milakov-Gimelshein) or classic two-pass — your choice, f32
//! accumulation, justify in the doc"). This op takes the classic,
//! multi-pass route (max, then exp+sum, then normalize — three passes for
//! BF16 specifically, see below) for two reasons: (1) it reuses the EXACT
//! block-wide reduction primitive (`block_reduce_sum`/`block_reduce_max`,
//! one block per row, grid-stride within the row) this crate's fused
//! LayerNorm already ships and this repository's reviewers have already
//! audited, rather than introducing a new running-max/running-sum
//! rescaling recurrence with its own correctness surface; (2) THIS op's
//! actual memory win is the BACKWARD tape retention (see above) and the
//! elimination of the `[B,H,S,S]`-shaped eager intermediates, not the
//! forward pass's own instruction count — an extra grid-stride pass over
//! one row (bandwidth-bound, `O(last)`, not `O(last^2)`) is cheap next to
//! the `S^2`-class memory problem this commit actually targets, so trading
//! a more complex single-pass recurrence for a simpler, already-audited
//! multi-pass shape is the right tradeoff here. F32 accumulation
//! throughout (row max and row sum both accumulate in `f32`, matching
//! every other op in this crate); BF16 rounds to BF16 exactly once, at the
//! very end (see `softmax_row_bf16` and the CUDA kernel's identical
//! discipline — the CUDA arm cannot cheaply cache an f32-precision
//! intermediate in a register across the sum pass and the normalize pass
//! without either a `last`-sized scratch buffer or a third grid-stride pass
//! that RECOMPUTES `exp` from `scores`/`mask` a second time; it takes the
//! latter — one extra `expf` and one extra pair of global reads per
//! element, bandwidth/compute-cheap, disclosed honestly — specifically to
//! avoid storing-then-rereading an already-BF16-rounded intermediate,
//! which would round TWICE and break this crate's "f32-accumulate,
//! round-once" convention).
//!
//! One MORE pass than the count above, disclosed honestly: the
//! safe-softmax check (previous section) reads `mask` ALONE in its own
//! grid-stride pass (one more `block_reduce_max`) before the main
//! reduction ever starts, on EVERY row, not only fully-masked ones — an
//! `O(last)` mask-only read, cheap next to the row's other passes and
//! negligible next to the `S^2`-class memory problem this op targets, but
//! a real, additional cost this doc does not hide.
//!
//! ## `bwd`: needs ONLY the output (`res` = `y`) and `grad_res`
//!
//! `dscores = (dy - sum(dy * y, last)) * y` — the standard softmax
//! backward identity, and it needs no `scores`, no `mask`, and no
//! recomputation of the forward at all: just `y` (already resident — the
//! attention V-matmul reads it right after) and `dy`. Implemented as ONE
//! internal kernel, [`SoftmaxBwdDScores`] (`CustomOp2`: `y`, `dy`), reused
//! via `super::apply2` from [`SoftmaxLastDimFused::bwd`] — the same
//! internal-helper-`KernelOp` shape `LayerNormFused::bwd` and `RopeFused`'s
//! reused-with-negated-sin trick both use, rather than composing ordinary
//! `Tensor` ops for this (which would reintroduce exactly the retained-
//! intermediate cost this whole op exists to remove).
//!
//! `dscores`'s slot is ALWAYS `Some` (the same "may be an intermediate on a
//! path to a `Var`" rule `LayerNormFused`'s `dx` and `RopeFused`'s `dx`
//! slots document — `is_variable() == false` does NOT mean "no gradient
//! needed"). `mask`'s slot uses `mask.is_variable()` directly, exactly like
//! `RopeFused`'s `dcos`/`dsin` (no separate construction-data flag is
//! needed the way `LayerNormFused::dgamma_needed` needs one, because `bwd`
//! already receives `mask` itself as an argument — nothing to freeze ahead
//! of time). In every call site this crate ships, `mask` is a true external
//! constant (built by `broadcast_add`ing `Tensor`s that are never wrapped
//! in `Var`), so `dmask` is `None` in practice; [`mask_grad`] computes a
//! REAL gradient via ordinary `Tensor` composition (sum over exactly the
//! axes `mask` broadcast, then reshape) for the case a future caller DOES
//! make it trainable — the same "correctness over micro-optimization, this
//! path is provably dead today" choice `RopeFused::bwd`'s `dcos`/`dsin` and
//! `Axpy::bwd` both make.
//!
//! ## esc-037 disposition
//!
//! esc-037 (`.jammi/escapes.jsonl`) names TWO backward-truncating APIs:
//! `candle_nn::ops::softmax_last_dim` (`apply_op1_no_bwd`) at
//! `htsat_audio.rs:632`, `clip_text.rs:150`, and `open_clip_vit.rs:169`;
//! and `QMatMul` (`candle-core`'s `quantized/mod.rs:1023`), the natural
//! entry point for quantized fine-tuning. [`SoftmaxLastDimFused`] is a
//! DIFFERENT operator entirely — a `CustomOp2` with a REAL `bwd` (this
//! module), dispatched via `super::apply2` (never `apply_op2_no_bwd`) —
//! and it is wired at a DIFFERENT call site (ModernBERT's attention
//! softmax, which today uses `candle_nn::ops::softmax`, an ordinary
//! composed-and-differentiable function, not `softmax_last_dim`). This
//! commit therefore does not make esc-037's named observable impossible
//! on ANY of the paths it names — none of the three `softmax_last_dim`
//! call sites is touched, `QMatMul` is untouched entirely, and none of
//! them reaches this op. `closes_escape` is NOT claimed; esc-037 remains
//! fully open for BOTH the `softmax_last_dim` call sites AND `QMatMul`.
//! (This op's OWN existence is also not a new instance of esc-037's
//! hazard class: it never uses `apply_op2_no_bwd`, so nothing upstream of
//! it silently loses its gradient the way esc-037 describes.)
//!
//! ## Domain, continued: dtype / contiguity / rank
//!
//! CPU supports F32 and BF16 (this crate's real training dtypes, matching
//! every other fused op here). Both `scores` and `mask` must be fully
//! contiguous (`contiguous_offsets()`, the same idiom as every other op in
//! this crate, and for the same reason: a raw-pointer kernel has no flat
//! linear index for a strided view). `last == 0` degenerates to an empty
//! output (nothing to normalize over) — the same "zero-length last dim
//! implies zero elements" argument `LayerNormFused`'s `hidden == 0` case
//! documents.
//!
//! [`MAX_LAST_DIM`] mirrors `LayerNormFused::MAX_HIDDEN` / `RopeFused::MAX_HEAD_DIM`
//! exactly: a conservative, VALIDATED ceiling (oracle coverage stops here —
//! ModernBERT-large's `seq` class, 128/512, well inside it), NOT a hardware
//! limit, enforced ONLY on the CUDA arm (the block-wide reduction's shared
//! memory is `O(blockDim.x)`, not `O(last)`, exactly like LayerNorm's). The
//! CPU arm has no such ceiling. [`MAX_RANK`] is DIFFERENT IN KIND from
//! those two: it is a REAL implementation constraint of the CUDA kernel's
//! fixed-arity signature (three leading-axis-size scalar arguments per
//! tensor, padded with `1` on the left when `scores`'s actual leading-axis
//! count is smaller — see `crate::cuda::softmax`), not merely "untested
//! above here". The CPU arm's [`mask_row_offset`] is a `Vec`-based general
//! unravel/ravel with no such ceiling, so `MAX_RANK` is, like
//! `MAX_LAST_DIM`, enforced only on the CUDA arm — the call site applies
//! both uniformly across devices (the LayerNorm/RoPE admission-predicate
//! precedent) so a real training run never depends on which device
//! happens to accept a wider domain.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp2, Error, Layout, Result, Shape, Tensor};
use half::bf16;

/// The largest `last` (reduction-axis) size the CUDA kernel accepts.
///
/// A conservative, VALIDATED ceiling — NOT a hardware constraint, the same
/// status `LayerNormFused::MAX_HIDDEN` / `RopeFused::MAX_HEAD_DIM` document:
/// the block-wide reduction below is `O(blockDim.x)` shared memory, not
/// `O(last)`, and the grid-stride loop over `last` has no correctness bound.
/// ModernBERT-large's profiled `seq` classes (128, 512) sit well inside this
/// with room to spare. Enforced only on the CUDA arm
/// (`crate::cuda::softmax`); the CPU arm has no such ceiling but re-exports
/// this constant so a call site can apply ONE domain check regardless of
/// device.
pub const MAX_LAST_DIM: usize = 4096;

/// The largest total rank (`scores.rank()`, equivalently `mask.rank()` —
/// the domain requires them equal) the CUDA kernel accepts.
///
/// UNLIKE [`MAX_LAST_DIM`], this is a REAL implementation constraint, not
/// merely a validated-coverage ceiling: the CUDA glue (`crate::cuda::softmax`)
/// passes each tensor's leading-axis sizes to the kernel as THREE fixed
/// scalar arguments (padded with `1` on the left when the actual leading-
/// axis count is smaller than three), so a rank above `4` (more than three
/// leading axes plus the reduction axis) has nowhere to go in that fixed
/// signature. `4` is exactly ModernBERT attention's own shape (`[batch,
/// heads, seq, seq]`, three leading axes) — zero headroom is needed for
/// this crate's actual workload. The CPU arm's `mask_row_offset` is a
/// general `Vec`-based unravel/ravel with no such ceiling, so — mirroring
/// `MAX_LAST_DIM` — this is enforced only on the CUDA arm; a call site
/// applies it uniformly across devices anyway (see the module doc).
pub const MAX_RANK: usize = 4;

/// Validates the `(scores, mask)` domain shared by every arm of this op —
/// see the module doc's "supported mask broadcast class" section. Returns
/// `(rows, last)`: `last` is the shared reduction-axis size, `rows =
/// scores.elem_count() / last` (or `(0, 0)` when `last == 0`, signalling
/// the empty fast path — the caller checks that and does not divide by it).
pub(crate) fn softmax_dims(
    l_scores: &Layout,
    l_mask: &Layout,
    op: &'static str,
) -> Result<(usize, usize)> {
    let s_dims = l_scores.dims();
    let m_dims = l_mask.dims();
    if s_dims.len() != m_dims.len() {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l_scores.shape().clone(),
            rhs: l_mask.shape().clone(),
            op,
        });
    }
    let Some(&last) = s_dims.last() else {
        return Err(Error::Msg(format!(
            "{op}: input must have rank >= 1 to define a last (reduction) dimension"
        )));
    };
    // Rank equality above guarantees `m_dims` is also non-empty here.
    if m_dims[m_dims.len() - 1] != last {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l_scores.shape().clone(),
            rhs: l_mask.shape().clone(),
            op,
        });
    }
    let lead = s_dims.len() - 1;
    for axis in 0..lead {
        let (sd, md) = (s_dims[axis], m_dims[axis]);
        if md != 1 && md != sd {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: l_scores.shape().clone(),
                rhs: l_mask.shape().clone(),
                op,
            });
        }
    }
    if last == 0 {
        return Ok((0, 0));
    }
    Ok((s_dims.iter().product::<usize>() / last, last))
}

/// Whether `mask` is within `scores`'s supported broadcast class (the
/// module doc's "supported mask broadcast class" section) — the EXACT
/// same check `cpu_fwd`/`cuda_fwd` apply internally (this calls
/// `softmax_dims` directly, not a re-derived copy of its logic).
///
/// `pub`, unlike `softmax_dims` itself: a call site's own admission
/// predicate (e.g. `jammi_encoders::modernbert::softmax_admission_predicate`)
/// uses this to check the broadcast class BEFORE ever calling [`super::apply2`]
/// with this op, so a shape outside the class becomes a COUNTED eager
/// fallback (K2's "validate, don't silently degrade" doctrine) at the call
/// site, rather than a `candle_core::Error` surfacing from inside the op
/// on the training arm. The op's own internal check is NOT removed or
/// weakened by this — it remains the correct defense for any caller that
/// invokes `apply2` directly (every hermetic unit test in this module,
/// and any future caller that does not go through an admission predicate
/// at all).
pub fn mask_broadcast_class_holds(scores: &Tensor, mask: &Tensor) -> bool {
    softmax_dims(
        scores.layout(),
        mask.layout(),
        "softmax_last_dim_fused_admission_check",
    )
    .is_ok()
}

/// Maps a flattened row index `row` (0-indexed over `scores`'s leading
/// axes, `0..rows`, row-major — the LAST of `s_lead` varies fastest) to the
/// corresponding flat row index into `mask`'s OWN leading-axis space,
/// honoring broadcasting (an axis where `m_lead`'s size is `1` always maps
/// to index `0`). See the module doc for why this is a full, exact
/// multi-index unravel/ravel rather than a single-modulus shortcut.
/// `s_lead.len() == m_lead.len()` is a precondition (`softmax_dims` already
/// enforces the equal-rank domain that guarantees it).
fn mask_row_offset(row: usize, s_lead: &[usize], m_lead: &[usize]) -> usize {
    let rank = s_lead.len();
    let mut idx = vec![0usize; rank];
    let mut rem = row;
    for axis in (0..rank).rev() {
        let d = s_lead[axis].max(1);
        idx[axis] = rem % d;
        rem /= d;
    }
    let mut flat = 0usize;
    for axis in 0..rank {
        let i = if m_lead[axis] == 1 { 0 } else { idx[axis] };
        flat = flat * m_lead[axis].max(1) + i;
    }
    flat
}

fn empty_like(
    s1: &CpuStorage,
    s2: &CpuStorage,
    l1: &Layout,
    op: &'static str,
) -> Result<(CpuStorage, Shape)> {
    match (s1, s2) {
        (CpuStorage::F32(_), CpuStorage::F32(_)) => {
            Ok((CpuStorage::F32(Vec::new()), l1.shape().clone()))
        }
        (CpuStorage::BF16(_), CpuStorage::BF16(_)) => {
            Ok((CpuStorage::BF16(Vec::new()), l1.shape().clone()))
        }
        (s1, s2) if s1.dtype() != s2.dtype() => Err(Error::DTypeMismatchBinaryOp {
            lhs: s1.dtype(),
            rhs: s2.dtype(),
            op,
        }),
        (s1, _) => Err(Error::UnsupportedDTypeForOp(s1.dtype(), op)),
    }
}

/// Policy for [`SoftmaxLastDimFused`]'s behavior on a row where EVERY
/// position is masked (see the module doc's "fully-masked row" section).
/// CONSTRUCTION DATA — frozen into the `Copy` op instance before `apply2`
/// ever runs, the SAME shape `LayerNormFused::dgamma_needed` uses (see its
/// doc): the op itself never inspects any tensor's runtime state to make
/// this choice, the CALLER decides at construction time, explicitly.
///
/// This exists because `row_is_fully_masked`'s detection rule (`max
/// mask < 0.0` means "no attendable key") is a REAL, documented domain
/// restriction, not a universal truth about additive masking: a caller
/// using a uniformly-negative additive bias that is NOT a masking
/// sentinel at all (a distance-decay attention bias, say, that is never
/// exactly `0.0` anywhere by design) would have every one of its rows
/// misclassified as "fully masked" under that rule. Every OTHER domain
/// restriction in this crate is CHECKED AND REFUSED (a shape outside the
/// broadcast class, a dtype this op does not implement); this one, if it
/// were unconditional, would instead SILENTLY REWRITE THE OUTPUT for a
/// caller whose masking convention does not match the assumption — the
/// one behavior family D exists to rule out. Making it construction data
/// closes that gap: a generic caller gets [`FullyMaskedPolicy::Propagate`]
/// (this op's `Default`) unless it explicitly asks for
/// [`FullyMaskedPolicy::Zeros`], asserting its own masking convention
/// satisfies the rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FullyMaskedPolicy {
    /// Match `candle_nn::ops::softmax` EXACTLY on a fully-masked row:
    /// `NaN` for an `-inf` sentinel, a dtype-and-magnitude-dependent
    /// uniform-or-near-normal result for a finite one (see the module
    /// doc). Correct for ANY additive-bias convention, including ones
    /// `row_is_fully_masked`'s rule cannot safely classify. THE DEFAULT
    /// — see [`FullyMaskedPolicy::default`]'s doc for why.
    Propagate,
    /// The production-attention-kernel convention (PyTorch SDPA's
    /// `_safe_softmax`, FlashAttention-2's zero-on-empty-row): output ALL
    /// ZEROS. Sound ONLY when the caller's OWN masking convention
    /// guarantees `mask[i] == 0.0` exactly means "unmasked" and any
    /// `mask[i] < 0.0` means "masked" — `jammi_encoders::mask`'s
    /// convention, and the only one any call site in this crate ships
    /// actually opts into this policy for
    /// (`modernbert.rs`'s `softmax_apply_training`, with the citations
    /// its own doc carries for why that premise holds there).
    Zeros,
}

impl Default for FullyMaskedPolicy {
    /// [`FullyMaskedPolicy::Propagate`] — the conservative, generically-
    /// correct default for a crate-owned-by-nobody primitive (family L:
    /// this op names no consumer). A caller that never opted into the
    /// production-kernel zero-output behavior gets candle-eager's OWN
    /// output on a fully-masked row, never a silent behavior change this
    /// op invented on its own initiative. `ModernBertAttention`'s training
    /// arm is the ONE call site in this crate that opts into
    /// [`FullyMaskedPolicy::Zeros`] instead, explicitly, with its own
    /// domain premise stated at the call site
    /// (`modernbert.rs`'s `softmax_apply_training`).
    fn default() -> Self {
        FullyMaskedPolicy::Propagate
    }
}

/// Fused masked softmax-last-dim forward. See the module doc for the full
/// design.
#[derive(Debug, Clone, Copy, Default)]
pub struct SoftmaxLastDimFused {
    /// See [`FullyMaskedPolicy`]'s doc. Construction data, never inspected
    /// at runtime against any tensor's own state. `derive(Default)` here
    /// resolves to `FullyMaskedPolicy::default()` (`Propagate`) via that
    /// type's own `Default` impl — the SAME conservative default, just
    /// derived rather than hand-written (clippy's `derivable_impls`).
    pub fully_masked: FullyMaskedPolicy,
}

impl SoftmaxLastDimFused {
    pub fn new(fully_masked: FullyMaskedPolicy) -> Self {
        Self { fully_masked }
    }
}

impl super::sealed::Sealed for SoftmaxLastDimFused {}

impl CustomOp2 for SoftmaxLastDimFused {
    fn name(&self) -> &'static str {
        "softmax_last_dim_fused"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (rows, last) = softmax_dims(l1, l2, self.name())?;
        if last == 0 {
            return empty_like(s1, s2, l1, self.name());
        }
        let (o1, o2) = l1
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: self.name() })?;
        let (m1, m2) = l2
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: self.name() })?;
        let rank = l1.dims().len();
        let s_lead = &l1.dims()[..rank - 1];
        let m_lead = &l2.dims()[..rank - 1];
        match (s1, s2) {
            (CpuStorage::F32(sc), CpuStorage::F32(mk)) => {
                let out = softmax_fwd_f32(
                    &sc[o1..o2],
                    &mk[m1..m2],
                    rows,
                    last,
                    s_lead,
                    m_lead,
                    self.fully_masked,
                );
                Ok((CpuStorage::F32(out), l1.shape().clone()))
            }
            (CpuStorage::BF16(sc), CpuStorage::BF16(mk)) => {
                let out = softmax_fwd_bf16(
                    &sc[o1..o2],
                    &mk[m1..m2],
                    rows,
                    last,
                    s_lead,
                    m_lead,
                    self.fully_masked,
                );
                Ok((CpuStorage::BF16(out), l1.shape().clone()))
            }
            (s1, s2) if s1.dtype() != s2.dtype() => Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: self.name(),
            }),
            (s1, _) => Err(Error::UnsupportedDTypeForOp(s1.dtype(), self.name())),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        crate::cuda::softmax::cuda_fwd(s1, l1, s2, l2, self.fully_masked)
    }

    /// See the module doc's "`bwd`: needs ONLY the output" section.
    /// `dscores`'s slot is ALWAYS `Some`; `mask`'s slot follows
    /// `mask.is_variable()`.
    fn bwd(
        &self,
        _scores: &Tensor,
        mask: &Tensor,
        res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let dscores = super::apply2(res, grad_res, SoftmaxBwdDScores)?;
        let dmask = if mask.is_variable() {
            Some(mask_grad(&dscores, mask.shape())?)
        } else {
            None
        };
        Ok((Some(dscores), dmask))
    }
}

/// `d(out)/d(mask)` summed over exactly the axes `mask` broadcast over
/// (where `mask`'s own size was `1` but `dscores`'s — i.e. `scores`'s — was
/// not), then reshaped to `mask`'s own shape. Ordinary `Tensor`
/// composition (`sum_keepdim`), not a further fused kernel — deliberately,
/// since `mask` is never a `Var` in any call site this crate ships (see the
/// module doc); this exists so a future caller that DOES make it trainable
/// gets a correct gradient rather than a silently-`None` one, mirroring
/// `crate::ops::rope`'s `rope_grad_table` exactly.
fn mask_grad(dscores: &Tensor, mask_shape: &Shape) -> Result<Tensor> {
    let mut out = dscores.clone();
    for (axis, (&d, &m)) in dscores.dims().iter().zip(mask_shape.dims()).enumerate() {
        if m == 1 && d != 1 {
            out = out.sum_keepdim(axis)?;
        }
    }
    out.reshape(mask_shape.clone())
}

/// `SoftmaxLastDimFused`'s internal backward helper producing `dscores`.
/// Not exported — only ever invoked from [`SoftmaxLastDimFused::bwd`] via
/// [`super::apply2`]. `CustomOp2` over `(y, dy)` — see the module doc for
/// why this needs no `scores`/`mask` input at all.
#[derive(Debug, Clone, Copy)]
struct SoftmaxBwdDScores;

impl super::sealed::Sealed for SoftmaxBwdDScores {}

impl CustomOp2 for SoftmaxBwdDScores {
    fn name(&self) -> &'static str {
        "softmax_last_dim_fused_bwd_dscores"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        if l1.dims() != l2.dims() {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: l1.shape().clone(),
                rhs: l2.shape().clone(),
                op: self.name(),
            });
        }
        let dims = l1.dims();
        let last = *dims.last().ok_or_else(|| {
            Error::Msg(format!(
                "{}: input must have rank >= 1 to define a last (reduction) dimension",
                self.name()
            ))
        })?;
        if last == 0 {
            return empty_like(s1, s2, l1, self.name());
        }
        let rows = dims.iter().product::<usize>() / last;
        let (o1, o2) = l1
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: self.name() })?;
        let (d1, d2) = l2
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: self.name() })?;
        match (s1, s2) {
            (CpuStorage::F32(y), CpuStorage::F32(dy)) => {
                let out = dscores_f32(&y[o1..o2], &dy[d1..d2], rows, last);
                Ok((CpuStorage::F32(out), l1.shape().clone()))
            }
            (CpuStorage::BF16(y), CpuStorage::BF16(dy)) => {
                let out = dscores_bf16(&y[o1..o2], &dy[d1..d2], rows, last);
                Ok((CpuStorage::BF16(out), l1.shape().clone()))
            }
            (s1, s2) if s1.dtype() != s2.dtype() => Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: self.name(),
            }),
            (s1, _) => Err(Error::UnsupportedDTypeForOp(s1.dtype(), self.name())),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        crate::cuda::softmax::cuda_bwd_dscores(s1, l1, s2, l2)
    }

    // No `bwd` override: this helper's own second-order gradient is never
    // requested by any call site in this crate or its consumers — the
    // default `CustomOp2::bwd` (`Err(BackwardNotSupported)`) is the correct
    // refusal if anything ever tried, mirroring `LayerNormBwdDx`'s and
    // `LayerNormBwdDgamma`'s identical notes.
}

// -----------------------------------------------------------------------
// CPU math. Fixed fold order throughout (family J): every reduction below
// walks its row in plain ascending index order, so a given input always
// yields the same output bit-for-bit — no parallel/unordered accumulation
// on this path.
// -----------------------------------------------------------------------

/// Whether `mask` (a single row's slice) marks EVERY position as masked —
/// the "safe softmax" trigger. See the module doc's "fully-masked row"
/// section: this crate's additive-mask convention uses `0.0` (exactly) as
/// the "unmasked" identity and any value `< 0.0` as "some degree of
/// masking" (never a mix of positive and non-`0.0`-negative values in any
/// call site this crate ships) — a row where NO position holds the exact
/// `0.0` identity has no attendable key at all.
fn row_is_fully_masked(mask: &[f32]) -> bool {
    mask.iter().cloned().fold(f32::NEG_INFINITY, f32::max) < 0.0
}

fn softmax_row_f32(scores: &[f32], mask: &[f32], out: &mut [f32], policy: FullyMaskedPolicy) {
    let last = scores.len();
    // Safe softmax (module doc): under `FullyMaskedPolicy::Zeros` ONLY, a
    // fully-masked row outputs ZEROS, matching PyTorch SDPA's
    // `_safe_softmax` / FlashAttention-2's zero-on-empty-row convention —
    // an INTENTIONAL, documented divergence from `candle_nn::ops::softmax`'s
    // own uniform-1/n-or-NaN behavior there. `Propagate` skips this branch
    // entirely and always falls through to the ordinary computation below,
    // reproducing eager EXACTLY (including `NaN` on an all-`-inf` row).
    if policy == FullyMaskedPolicy::Zeros && row_is_fully_masked(mask) {
        out.fill(0.0);
        return;
    }
    let mut max = f32::NEG_INFINITY;
    for i in 0..last {
        let v = scores[i] + mask[i];
        if v > max {
            max = v;
        }
    }
    let mut sum = 0f32;
    for i in 0..last {
        let v = scores[i] + mask[i];
        let e = (v - max).exp();
        out[i] = e;
        sum += e;
    }
    for v in out.iter_mut() {
        *v /= sum;
    }
}

fn softmax_fwd_f32(
    scores: &[f32],
    mask: &[f32],
    rows: usize,
    last: usize,
    s_lead: &[usize],
    m_lead: &[usize],
    policy: FullyMaskedPolicy,
) -> Vec<f32> {
    let mut out = vec![0f32; rows * last];
    for r in 0..rows {
        let mrow = mask_row_offset(r, s_lead, m_lead);
        let sr = &scores[r * last..(r + 1) * last];
        let mr = &mask[mrow * last..(mrow + 1) * last];
        let outr = &mut out[r * last..(r + 1) * last];
        softmax_row_f32(sr, mr, outr, policy);
    }
    out
}

/// BF16 accumulates in f32 (row max, the `exp` values, and the row sum all
/// stay in f32 via a temporary `Vec<f32>` — cheap on CPU, no shared-memory
/// constraint the way the CUDA arm has), rounding to bf16 exactly once on
/// the way out.
fn softmax_row_bf16(scores: &[bf16], mask: &[bf16], out: &mut [bf16], policy: FullyMaskedPolicy) {
    let last = scores.len();
    // Safe softmax first (module doc), under `FullyMaskedPolicy::Zeros`
    // ONLY: a fully-masked row outputs ZEROS regardless of dtype, so the
    // BF16-annihilation question below never even arises for this row — it
    // is short-circuited before the add. `Propagate` never takes this
    // branch, so a fully-masked row falls through to the BF16-rounded-add
    // path below unconditionally, reproducing eager's own annihilated-
    // uniform output exactly.
    if policy == FullyMaskedPolicy::Zeros {
        let mask_row_max = mask
            .iter()
            .map(|m| m.to_f32())
            .fold(f32::NEG_INFINITY, f32::max);
        if mask_row_max < 0.0 {
            out.fill(bf16::ZERO);
            return;
        }
    }
    // The ONE deliberate deviation from this crate's "F32-accumulate-
    // throughout" convention (see the module doc's "bf16 mask-add
    // rounding" section), for every row that reaches this point (i.e. is
    // NOT fully masked, but may still have SOME masked positions):
    // `scores[i] + mask[i]` is rounded to BF16 IMMEDIATELY, matching
    // `candle_nn::ops::softmax`'s native BF16 `broadcast_add` (which
    // rounds at that exact step) — NOT computed and kept in F32. At the
    // real `MASKED_LOGIT` magnitude (`-10_000.0`) this rounding
    // ANNIHILATES any real `scores[i]` (BF16's ULP there, ≈64, far exceeds
    // a real score's own magnitude) at a MASKED position, matching
    // eager's own BF16-native rounding there rather than a strictly-more-
    // precise (but wrong-relative-to-eager) result. Every step AFTER this
    // one (max/exp/sum/normalize) still accumulates in F32, matching the
    // crate's usual convention.
    let mut v = vec![0f32; last];
    let mut max = f32::NEG_INFINITY;
    for i in 0..last {
        let vi = bf16::from_f32(scores[i].to_f32() + mask[i].to_f32()).to_f32();
        v[i] = vi;
        if vi > max {
            max = vi;
        }
    }
    let mut exps = vec![0f32; last];
    let mut sum = 0f32;
    for i in 0..last {
        let e = (v[i] - max).exp();
        exps[i] = e;
        sum += e;
    }
    for i in 0..last {
        out[i] = bf16::from_f32(exps[i] / sum);
    }
}

fn softmax_fwd_bf16(
    scores: &[bf16],
    mask: &[bf16],
    rows: usize,
    last: usize,
    s_lead: &[usize],
    m_lead: &[usize],
    policy: FullyMaskedPolicy,
) -> Vec<bf16> {
    let mut out = vec![bf16::ZERO; rows * last];
    for r in 0..rows {
        let mrow = mask_row_offset(r, s_lead, m_lead);
        let sr = &scores[r * last..(r + 1) * last];
        let mr = &mask[mrow * last..(mrow + 1) * last];
        let outr = &mut out[r * last..(r + 1) * last];
        softmax_row_bf16(sr, mr, outr, policy);
    }
    out
}

/// `dscores_row = (dy - dot(dy, y)) * y` — the standard softmax backward
/// identity, needing only `y` and `dy`. Fixed fold order (family J):
/// `dot` accumulates over the row in ascending index order.
fn dscores_row_f32(y: &[f32], dy: &[f32], out: &mut [f32]) {
    let mut dot = 0f32;
    for i in 0..y.len() {
        dot += dy[i] * y[i];
    }
    for i in 0..y.len() {
        out[i] = (dy[i] - dot) * y[i];
    }
}

fn dscores_f32(y: &[f32], dy: &[f32], rows: usize, last: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * last];
    for r in 0..rows {
        let yr = &y[r * last..(r + 1) * last];
        let dyr = &dy[r * last..(r + 1) * last];
        let outr = &mut out[r * last..(r + 1) * last];
        dscores_row_f32(yr, dyr, outr);
    }
    out
}

fn dscores_row_bf16(y: &[bf16], dy: &[bf16], out: &mut [bf16]) {
    let mut dot = 0f32;
    for i in 0..y.len() {
        dot += dy[i].to_f32() * y[i].to_f32();
    }
    for i in 0..y.len() {
        out[i] = bf16::from_f32((dy[i].to_f32() - dot) * y[i].to_f32());
    }
}

fn dscores_bf16(y: &[bf16], dy: &[bf16], rows: usize, last: usize) -> Vec<bf16> {
    let mut out = vec![bf16::ZERO; rows * last];
    for r in 0..rows {
        let yr = &y[r * last..(r + 1) * last];
        let dyr = &dy[r * last..(r + 1) * last];
        let outr = &mut out[r * last..(r + 1) * last];
        dscores_row_bf16(yr, dyr, outr);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    /// `SoftmaxLastDimFused::default()` (`FullyMaskedPolicy::Propagate`) —
    /// every test using this helper is either not exercising a
    /// fully-masked row at all (where the two policies are identical) or
    /// is explicitly testing `Propagate`'s eager-matching behavior there;
    /// tests of `FullyMaskedPolicy::Zeros` use [`fused_with_policy`]
    /// directly instead.
    fn fused(scores: &Tensor, mask: &Tensor) -> Result<Tensor> {
        crate::ops::apply2(scores, mask, SoftmaxLastDimFused::default())
    }

    fn fused_with_policy(
        scores: &Tensor,
        mask: &Tensor,
        policy: FullyMaskedPolicy,
    ) -> Result<Tensor> {
        crate::ops::apply2(scores, mask, SoftmaxLastDimFused::new(policy))
    }

    fn eager(scores: &Tensor, mask: &Tensor) -> Result<Tensor> {
        candle_nn::ops::softmax(&scores.broadcast_add(mask)?, candle_core::D::Minus1)
    }

    // -------------------------------------------------------------------
    // Oracle 4: row-sum invariant.
    // -------------------------------------------------------------------
    #[test]
    fn row_sums_to_one_within_tolerance() {
        let device = Device::Cpu;
        let scores = Tensor::from_slice(
            &[1.0f32, 2.0, -3.0, 0.5, -1.0, 4.0, 0.0, 0.25],
            (2, 4),
            &device,
        )
        .unwrap();
        let mask = Tensor::from_slice(&[0.0f32; 4], (1, 4), &device).unwrap();
        let out = fused(&scores, &mask).unwrap().to_vec2::<f32>().unwrap();
        for row in out {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row sum {sum} != 1");
        }
    }

    #[test]
    fn cpu_fwd_matches_hand_computed_values() {
        let device = Device::Cpu;
        let scores = Tensor::from_slice(&[1.0f32, 2.0, 3.0], (1, 3), &device).unwrap();
        let mask = Tensor::from_slice(&[0.0f32, 0.0, 0.0], (1, 3), &device).unwrap();
        let out = fused(&scores, &mask)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let m = 3.0f32;
        let exps = [1.0f32 - m, 2.0 - m, 3.0 - m].map(f32::exp);
        let sum: f32 = exps.iter().sum();
        let expected: Vec<f32> = exps.iter().map(|e| e / sum).collect();
        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-6, "{o} vs {e}");
        }
    }

    #[test]
    fn mask_add_is_folded_in_matching_broadcast_add_then_softmax() {
        let device = Device::Cpu;
        let scores = Tensor::from_slice(&[1.0f32, 5.0, -2.0, 0.5], (1, 4), &device).unwrap();
        let mask =
            Tensor::from_slice(&[0.0f32, -10_000.0, 0.0, -10_000.0], (1, 4), &device).unwrap();
        let got = fused(&scores, &mask)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let want = eager(&scores, &mask)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-6, "{g} vs {w}");
        }
    }

    // -------------------------------------------------------------------
    // Oracle 5: extreme-value domain tests.
    // -------------------------------------------------------------------
    #[test]
    fn a_single_masked_position_contributes_exactly_zero() {
        let device = Device::Cpu;
        let scores = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let mask =
            Tensor::from_slice(&[0.0f32, f32::NEG_INFINITY, 0.0, 0.0], (1, 4), &device).unwrap();
        let out = fused(&scores, &mask)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(out[1], 0.0, "a -inf-masked position must contribute 0");
        let sum: f32 = out.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "remaining positions still sum to 1: {sum}"
        );
        // Matches eager exactly at this fixture.
        let want = eager(&scores, &mask)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (g, w) in out.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-6, "{g} vs {w}");
        }
    }

    /// Audit BLOCK finding, resolved via `FullyMaskedPolicy::Zeros`: a
    /// fully-masked row (synthetic `-inf` convention here) outputs ZEROS
    /// (safe-softmax — see the module doc's "fully-masked row" section),
    /// an INTENTIONAL divergence from `candle_nn::ops::softmax`'s own
    /// `NaN` there. Eager's `NaN` is recorded as the known-divergent
    /// baseline, not matched.
    #[test]
    fn an_all_masked_row_under_zeros_policy_outputs_zeros_diverging_from_eagers_nan_intentionally()
    {
        let device = Device::Cpu;
        let scores = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let mask = Tensor::from_slice(&[f32::NEG_INFINITY; 4], (1, 4), &device).unwrap();
        let out = fused_with_policy(&scores, &mask, FullyMaskedPolicy::Zeros)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(
            out,
            vec![0.0f32; 4],
            "all-masked row must be all zeros under FullyMaskedPolicy::Zeros: {out:?}"
        );
        let want = eager(&scores, &mask)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            want.iter().all(|v| v.is_nan()),
            "sanity: eager's own composition must be all-NaN here -- the KNOWN-DIVERGENT \
             baseline Zeros deliberately does not match, per the module doc: {want:?}"
        );
    }

    /// Companion to the test above: `FullyMaskedPolicy::Propagate` (this
    /// op's `Default`) must instead MATCH eager's `NaN` exactly on the
    /// identical fixture -- the conservative, generically-correct default
    /// this crate ships when a caller does not opt into `Zeros`.
    #[test]
    fn an_all_masked_row_under_propagate_policy_matches_eagers_nan_exactly() {
        let device = Device::Cpu;
        let scores = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let mask = Tensor::from_slice(&[f32::NEG_INFINITY; 4], (1, 4), &device).unwrap();
        let out = fused_with_policy(&scores, &mask, FullyMaskedPolicy::Propagate)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            out.iter().all(|v| v.is_nan()),
            "Propagate must match eager's NaN on an all-masked row: {out:?}"
        );
        // `fused()` (the default-constructed helper) must agree.
        let out_default = fused(&scores, &mask)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(out_default.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn large_positive_scores_do_not_overflow() {
        let device = Device::Cpu;
        // exp(1e4) alone would overflow f32 without max-subtraction.
        let scores =
            Tensor::from_slice(&[1.0e4f32, 1.0e4 + 1.0, 1.0e4 - 5.0], (1, 3), &device).unwrap();
        let mask = Tensor::from_slice(&[0.0f32; 3], (1, 3), &device).unwrap();
        let out = fused(&scores, &mask)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            out.iter().all(|v| v.is_finite()),
            "must not overflow: {out:?}"
        );
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "row sum {sum} != 1");
    }

    #[test]
    fn finite_masked_logit_matches_the_real_call_sites_convention() {
        // jammi_encoders::mask::MASKED_LOGIT is -10_000.0, a FINITE large
        // negative, never -inf, per that module's own doc (sliding windows
        // keep every query's own diagonal, so no real row is ever fully
        // masked). This op's domain does not special-case that value at
        // all -- it is just a very negative finite score, handled by the
        // ordinary max-then-exp path, and it must still exactly match
        // eager.
        let device = Device::Cpu;
        const MASKED_LOGIT: f32 = -10_000.0;
        let scores = Tensor::from_slice(&[0.5f32, -0.2, 1.3, 0.0], (1, 4), &device).unwrap();
        let mask = Tensor::from_slice(&[0.0f32, MASKED_LOGIT, 0.0, MASKED_LOGIT], (1, 4), &device)
            .unwrap();
        let got = fused(&scores, &mask)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let want = eager(&scores, &mask)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-6, "{g} vs {w}");
        }
        assert!(got.iter().all(|v| v.is_finite()));
    }

    // -------------------------------------------------------------------
    // Domain / degenerate-input refusals.
    // -------------------------------------------------------------------
    #[test]
    fn empty_batch_is_a_no_op_not_an_error() {
        let device = Device::Cpu;
        let scores = Tensor::from_slice(&[] as &[f32], (0, 4), &device).unwrap();
        let mask = Tensor::from_slice(&[0.0f32; 4], (1, 4), &device).unwrap();
        let out = fused(&scores, &mask).unwrap().to_vec2::<f32>().unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn zero_last_dim_is_a_no_op_not_a_division_by_zero_panic() {
        let device = Device::Cpu;
        let scores = Tensor::from_slice(&[] as &[f32], (3, 0), &device).unwrap();
        let mask = Tensor::from_slice(&[] as &[f32], (1, 0), &device).unwrap();
        let out = fused(&scores, &mask).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|row| row.is_empty()));
    }

    #[test]
    fn rank_mismatch_between_scores_and_mask_is_refused() {
        let device = Device::Cpu;
        let scores = Tensor::from_slice(&[1.0f32; 8], (2, 1, 4), &device).unwrap();
        let mask = Tensor::from_slice(&[0.0f32; 4], (1, 4), &device).unwrap();
        let err = fused(&scores, &mask).expect_err("rank mismatch must be refused, not padded");
        assert!(matches!(err, Error::ShapeMismatchBinaryOp { .. }));
    }

    #[test]
    fn last_dim_mismatch_is_refused_not_broadcast() {
        let device = Device::Cpu;
        let scores = Tensor::from_slice(&[1.0f32; 8], (2, 4), &device).unwrap();
        let mask = Tensor::from_slice(&[0.0f32; 3], (1, 3), &device).unwrap();
        let err = fused(&scores, &mask).expect_err("last-dim mismatch must be refused");
        assert!(matches!(err, Error::ShapeMismatchBinaryOp { .. }));
    }

    #[test]
    fn a_leading_axis_neither_one_nor_equal_is_refused_not_silently_misindexed() {
        // scores: [3, 2, 4]; mask: [2, 2, 4] -- axis 0 is neither `1` nor
        // equal to scores' `3`. Accepting this would force a guess at which
        // rows to serve from a too-short mask; refused instead (the same
        // "provably correct, not merely necessary" discipline RoPE's
        // `rope_dims` documents).
        let device = Device::Cpu;
        let scores = Tensor::from_slice(&[0.0f32; 3 * 2 * 4], (3, 2, 4), &device).unwrap();
        let mask = Tensor::from_slice(&[0.0f32; 2 * 2 * 4], (2, 2, 4), &device).unwrap();
        let err = fused(&scores, &mask)
            .expect_err("a leading axis that is neither 1 nor equal must be refused");
        assert!(matches!(err, Error::ShapeMismatchBinaryOp { .. }));
    }

    #[test]
    fn dtype_mismatch_is_refused() {
        let device = Device::Cpu;
        let scores = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let mask = Tensor::from_slice(&[bf16::from_f32(0.0); 4], (1, 4), &device).unwrap();
        let err = fused(&scores, &mask).expect_err("dtype mismatch must be refused");
        assert!(matches!(err, Error::DTypeMismatchBinaryOp { .. }));
    }

    #[test]
    fn non_contiguous_scores_is_refused_not_silently_misread() {
        let device = Device::Cpu;
        let scores = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            (4, 2),
            &device,
        )
        .unwrap()
        .t()
        .unwrap();
        assert!(!scores.is_contiguous());
        let mask = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0], (1, 4), &device).unwrap();
        let err = fused(&scores, &mask).expect_err("non-contiguous scores must be refused");
        assert!(matches!(err, Error::RequiresContiguous { .. }));
    }

    #[test]
    fn bf16_forward_matches_f32_accumulation_rounded_once() {
        let device = Device::Cpu;
        let sv = [1.0f32, 2.0, -3.0, 0.5];
        let sb: Vec<bf16> = sv.iter().map(|&v| bf16::from_f32(v)).collect();
        let mb = [bf16::ZERO; 4];
        let scores = Tensor::from_slice(&sb, (1, 4), &device).unwrap();
        let mask = Tensor::from_slice(&mb, (1, 4), &device).unwrap();
        let out: Vec<bf16> = fused(&scores, &mask)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let xf: Vec<f64> = sb.iter().map(|v| v.to_f32() as f64).collect();
        let max = xf.iter().cloned().fold(f64::MIN, f64::max);
        let exps: Vec<f64> = xf.iter().map(|&v| (v - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        let expected: Vec<f32> = exps.iter().map(|&e| (e / sum) as f32).collect();

        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o.to_f32() - e).abs() < 1e-2, "{o} vs {e}");
        }
    }

    // -------------------------------------------------------------------
    // Oracle 1: CPU gradcheck (f64 reference) for dscores vs central
    // finite differences.
    // -------------------------------------------------------------------
    #[test]
    fn gradcheck_dscores_vs_central_finite_differences() {
        let device = Device::Cpu;
        let s0: [f32; 6] = [0.3, -1.2, 2.0, 0.1, -0.5, 1.7];
        let m0: [f32; 6] = [0.0, 0.0, -10_000.0, 0.0, 0.0, 0.0];
        let scores = Var::from_tensor(&Tensor::from_slice(&s0, (2, 3), &device).unwrap()).unwrap();
        let mask = Tensor::from_slice(&m0, (1, 3), &device).unwrap();

        let out = fused(&scores, &mask).unwrap();
        // Non-uniform seed (not `Tensor::backward`'s implicit all-ones):
        // dot with a fixed non-uniform weight tensor so every component of
        // `dy` differs, exercising the full `(dy - dot(dy,y)) * y` formula
        // rather than the degenerate all-ones case.
        let dy_seed =
            Tensor::from_slice(&[0.5f32, -1.0, 2.0, 0.25, -0.75, 1.5], (2, 3), &device).unwrap();
        let loss = (&out * &dy_seed).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let dscores: Vec<f32> = grads
            .get(&scores)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let loss_fwd = |s: &Tensor| -> f64 {
            let y = fused(s, &mask).unwrap();
            (&y * &dy_seed)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap() as f64
        };

        let fd_eps = 1e-3f32;
        let tol = 5e-2f64;
        for i in 0..s0.len() {
            let mut sp = s0;
            sp[i] += fd_eps;
            let mut sm = s0;
            sm[i] -= fd_eps;
            let sp_t = Tensor::from_slice(&sp, (2, 3), &device).unwrap();
            let sm_t = Tensor::from_slice(&sm, (2, 3), &device).unwrap();
            let numeric = (loss_fwd(&sp_t) - loss_fwd(&sm_t)) / (2.0 * fd_eps as f64);
            assert!(
                (numeric - dscores[i] as f64).abs() < tol,
                "dscores[{i}]: numeric {numeric} vs analytic {}",
                dscores[i]
            );
        }
    }

    // -------------------------------------------------------------------
    // Oracle 2: fused-vs-eager fwd+bwd, f32.
    // -------------------------------------------------------------------
    #[test]
    fn fused_matches_eager_fwd_and_bwd_f32() {
        let device = Device::Cpu;
        let sv: Vec<f32> = (0..24)
            .map(|i| (i as f32 * 0.37 - 4.0).sin() * 3.0)
            .collect();
        let mv: Vec<f32> = (0..8)
            .map(|i| if i % 3 == 0 { -10_000.0 } else { 0.0 })
            .collect();

        let s_fused = Var::from_tensor(&Tensor::from_slice(&sv, (3, 8), &device).unwrap()).unwrap();
        let mask_fused = Tensor::from_slice(&mv, (1, 8), &device).unwrap();
        let out_fused = fused(&s_fused, &mask_fused).unwrap();

        let s_eager = Var::from_tensor(&Tensor::from_slice(&sv, (3, 8), &device).unwrap()).unwrap();
        let mask_eager = Tensor::from_slice(&mv, (1, 8), &device).unwrap();
        let out_eager = eager(&s_eager, &mask_eager).unwrap();

        let vf: Vec<f32> = out_fused.flatten_all().unwrap().to_vec1().unwrap();
        let ve: Vec<f32> = out_eager.flatten_all().unwrap().to_vec1().unwrap();
        for (i, (f, e)) in vf.iter().zip(ve.iter()).enumerate() {
            assert!((f - e).abs() < 1e-5, "fwd[{i}]: fused {f} vs eager {e}");
        }

        // Non-uniform seed dy.
        let dy_seed = Tensor::from_slice(
            &(0..24)
                .map(|i| (i as f32 * 0.11 - 1.0).cos())
                .collect::<Vec<f32>>(),
            (3, 8),
            &device,
        )
        .unwrap();
        let loss_fused = (&out_fused * &dy_seed).unwrap().sum_all().unwrap();
        let loss_eager = (&out_eager * &dy_seed).unwrap().sum_all().unwrap();
        let grads_fused = loss_fused.backward().unwrap();
        let grads_eager = loss_eager.backward().unwrap();
        let dxf: Vec<f32> = grads_fused
            .get(&s_fused)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let dxe: Vec<f32> = grads_eager
            .get(&s_eager)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, (f, e)) in dxf.iter().zip(dxe.iter()).enumerate() {
            assert!((f - e).abs() < 1e-3, "dscores[{i}]: fused {f} vs eager {e}");
        }
    }

    #[test]
    fn fused_matches_eager_bf16_measured_nonzero_with_non_uniform_dy() {
        let device = Device::Cpu;
        let sv: Vec<f32> = (0..16)
            .map(|i| (i as f32 * 0.29 - 2.0).sin() * 2.0)
            .collect();
        let mv: Vec<f32> = (0..4)
            .map(|i| if i == 1 { -10_000.0 } else { 0.0 })
            .collect();
        let sb: Vec<bf16> = sv.iter().map(|&v| bf16::from_f32(v)).collect();
        let mb: Vec<bf16> = mv.iter().map(|&v| bf16::from_f32(v)).collect();

        let s_fused = Var::from_tensor(&Tensor::from_slice(&sb, (4, 4), &device).unwrap()).unwrap();
        let mask_fused = Tensor::from_slice(&mb, (1, 4), &device).unwrap();
        let out_fused = fused(&s_fused, &mask_fused).unwrap();

        let s_eager = Var::from_tensor(&Tensor::from_slice(&sb, (4, 4), &device).unwrap()).unwrap();
        let mask_eager = Tensor::from_slice(&mb, (1, 4), &device).unwrap();
        let out_eager = eager(&s_eager, &mask_eager).unwrap();

        let to_f32 = |t: &Tensor| -> Vec<f32> {
            t.to_dtype(candle_core::DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap()
        };
        let vf = to_f32(&out_fused);
        let ve = to_f32(&out_eager);
        // Measured-nonzero: pin that the fixture actually produces
        // non-trivial output (a vacuous all-zero comparison would pass
        // trivially and prove nothing).
        assert!(
            vf.iter().any(|v| v.abs() > 1e-3),
            "fixture must be non-degenerate"
        );
        for (i, (f, e)) in vf.iter().zip(ve.iter()).enumerate() {
            assert!(
                (f - e).abs() < 5e-2,
                "bf16 fwd[{i}]: fused {f} vs eager {e}"
            );
        }

        let dy_seed_f: Vec<f32> = (0..16).map(|i| (i as f32 * 0.17 - 1.3).cos()).collect();
        let dy_seed_b: Vec<bf16> = dy_seed_f.iter().map(|&v| bf16::from_f32(v)).collect();
        let dy_seed = Tensor::from_slice(&dy_seed_b, (4, 4), &device).unwrap();
        let loss_fused = (&out_fused * &dy_seed).unwrap().sum_all().unwrap();
        let loss_eager = (&out_eager * &dy_seed).unwrap().sum_all().unwrap();
        let grads_fused = loss_fused.backward().unwrap();
        let grads_eager = loss_eager.backward().unwrap();
        let dxf = to_f32(&grads_fused.get(&s_fused).unwrap().clone());
        let dxe = to_f32(&grads_eager.get(&s_eager).unwrap().clone());
        assert!(
            dxf.iter().any(|v| v.abs() > 1e-4),
            "gradient must be measured-nonzero"
        );
        for (i, (f, e)) in dxf.iter().zip(dxe.iter()).enumerate() {
            assert!(
                (f - e).abs() < 0.15,
                "bf16 dscores[{i}]: fused {f} vs eager {e}"
            );
        }
    }

    /// Audit BLOCK finding, RESOLVED via `FullyMaskedPolicy::Zeros`: a
    /// FULLY masked row at the REAL convention (`MASKED_LOGIT = -10_000.0`,
    /// finite — not the synthetic `-inf` shape the tests above cover),
    /// mirroring the padding-query construction that reaches this in
    /// production (see `jammi_encoders::mask::sliding_window_mask`'s
    /// corrected doc: a pad query whose entire window lies in the pad
    /// region has every key masked this way). `candle_nn::ops::softmax` on
    /// a BF16 tensor computes NATIVELY in BF16 (no internal F32 upcast), so
    /// BF16's coarse ULP near magnitude `10_000` (~64) ANNIHILATES any real
    /// score there, and eager's reference output on this row is UNIFORM
    /// `1/last`. `Zeros` does NOT reproduce that (see the module doc's
    /// "fully-masked row" section). Eager's uniform output is recorded
    /// here as the KNOWN-DIVERGENT baseline.
    #[test]
    fn bf16_fully_masked_row_under_zeros_policy_outputs_zeros_diverging_from_eagers_uniform_output()
    {
        let device = Device::Cpu;
        let last = 8;
        let sv: Vec<f32> = (0..last)
            .map(|i| (i as f32 * 0.37 - 2.0).sin() * 5.0)
            .collect();
        let mv: Vec<f32> = vec![-10_000.0f32; last];
        let sb: Vec<bf16> = sv.iter().map(|&v| bf16::from_f32(v)).collect();
        let mb: Vec<bf16> = mv.iter().map(|&v| bf16::from_f32(v)).collect();
        let scores = Tensor::from_slice(&sb, (1, last), &device).unwrap();
        let mask = Tensor::from_slice(&mb, (1, last), &device).unwrap();

        let to_f32 = |t: &Tensor| -> Vec<f32> {
            t.to_dtype(candle_core::DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap()
        };
        let got = to_f32(&fused_with_policy(&scores, &mask, FullyMaskedPolicy::Zeros).unwrap());
        let want = to_f32(&eager(&scores, &mask).unwrap());

        assert_eq!(
            got,
            vec![0.0f32; last],
            "all-masked row must be all zeros under FullyMaskedPolicy::Zeros: {got:?}"
        );
        // Sanity (non-vacuity): eager's OWN reference must actually be
        // (near-)uniform here -- confirming this is a genuine divergence
        // being deliberately not matched, not a vacuous comparison against
        // a reference that happened to also be zero.
        let uniform = 1.0f32 / last as f32;
        for (i, w) in want.iter().enumerate() {
            assert!(
                (w - uniform).abs() < 1e-2,
                "eager[{i}] = {w}, expected ~uniform {uniform} (bf16's ULP near \
                 -10_000 must annihilate the real score) -- if this fails, eager's \
                 own reference stopped being annihilated and this oracle needs revisiting"
            );
        }
    }

    /// Companion: `FullyMaskedPolicy::Propagate` MUST reproduce eager's
    /// annihilated-uniform output on the identical fixture (the BLOCK
    /// fix's original behavior, now correctly scoped to `Propagate` rather
    /// than being this op's unconditional default).
    #[test]
    fn bf16_fully_masked_row_under_propagate_policy_matches_eagers_uniform_output() {
        let device = Device::Cpu;
        let last = 8;
        let sv: Vec<f32> = (0..last)
            .map(|i| (i as f32 * 0.37 - 2.0).sin() * 5.0)
            .collect();
        let mv: Vec<f32> = vec![-10_000.0f32; last];
        let sb: Vec<bf16> = sv.iter().map(|&v| bf16::from_f32(v)).collect();
        let mb: Vec<bf16> = mv.iter().map(|&v| bf16::from_f32(v)).collect();
        let scores = Tensor::from_slice(&sb, (1, last), &device).unwrap();
        let mask = Tensor::from_slice(&mb, (1, last), &device).unwrap();

        let to_f32 = |t: &Tensor| -> Vec<f32> {
            t.to_dtype(candle_core::DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap()
        };
        let got = to_f32(&fused_with_policy(&scores, &mask, FullyMaskedPolicy::Propagate).unwrap());
        let want = to_f32(&eager(&scores, &mask).unwrap());
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-2, "propagate[{i}] = {g} vs eager {w}");
        }
    }

    /// Audit advisory 2, addressed: the bf16 fused-vs-eager bound above was
    /// measured at `last = 4`; the divergence mechanism (bf16-native vs
    /// f32-accumulated reduction) grows with the reduction width. This
    /// measures the REAL bound at ModernBERT-large's production widths
    /// (`seq = 128`, and the quadratic-regime `seq = 512`), fwd AND bwd,
    /// with a partially-masked (not fully-masked) row so this exercises
    /// the general reduction-width scaling, not the fully-masked
    /// annihilation case above.
    fn bf16_width_case(last: usize, tol_fwd: f32, tol_bwd: f32) {
        let device = Device::Cpu;
        let rows = 2;
        let sv: Vec<f32> = (0..rows * last)
            .map(|i| (i as f32 * 0.071 - 3.0).sin() * 2.0)
            .collect();
        let mv: Vec<f32> = (0..last)
            .map(|i| if i % 7 == 0 { -10_000.0 } else { 0.0 })
            .collect();
        let sb: Vec<bf16> = sv.iter().map(|&v| bf16::from_f32(v)).collect();
        let mb: Vec<bf16> = mv.iter().map(|&v| bf16::from_f32(v)).collect();

        let s_fused =
            Var::from_tensor(&Tensor::from_slice(&sb, (rows, last), &device).unwrap()).unwrap();
        let mask_fused = Tensor::from_slice(&mb, (1, last), &device).unwrap();
        let out_fused = fused(&s_fused, &mask_fused).unwrap();

        let s_eager =
            Var::from_tensor(&Tensor::from_slice(&sb, (rows, last), &device).unwrap()).unwrap();
        let mask_eager = Tensor::from_slice(&mb, (1, last), &device).unwrap();
        let out_eager = eager(&s_eager, &mask_eager).unwrap();

        let to_f32 = |t: &Tensor| -> Vec<f32> {
            t.to_dtype(candle_core::DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap()
        };
        let vf = to_f32(&out_fused);
        let ve = to_f32(&out_eager);
        assert!(
            vf.iter().any(|v| v.abs() > 1e-3),
            "fixture must be non-degenerate (last={last})"
        );
        let max_fwd = vf
            .iter()
            .zip(ve.iter())
            .map(|(f, e)| (f - e).abs())
            .fold(0.0f32, f32::max);
        eprintln!("MEASURED max|delta| bf16 fwd (last={last}) = {max_fwd:e}");
        for (i, (f, e)) in vf.iter().zip(ve.iter()).enumerate() {
            assert!(
                (f - e).abs() < tol_fwd,
                "bf16 fwd[{i}] (last={last}): fused {f} vs eager {e}"
            );
        }

        let dy_seed_f: Vec<f32> = (0..rows * last)
            .map(|i| (i as f32 * 0.113 - 1.7).cos())
            .collect();
        let dy_seed_b: Vec<bf16> = dy_seed_f.iter().map(|&v| bf16::from_f32(v)).collect();
        let dy_seed = Tensor::from_slice(&dy_seed_b, (rows, last), &device).unwrap();
        let loss_fused = (&out_fused * &dy_seed).unwrap().sum_all().unwrap();
        let loss_eager = (&out_eager * &dy_seed).unwrap().sum_all().unwrap();
        let grads_fused = loss_fused.backward().unwrap();
        let grads_eager = loss_eager.backward().unwrap();
        let dxf = to_f32(&grads_fused.get(&s_fused).unwrap().clone());
        let dxe = to_f32(&grads_eager.get(&s_eager).unwrap().clone());
        assert!(
            dxf.iter().any(|v| v.abs() > 1e-4),
            "gradient must be measured-nonzero (last={last})"
        );
        let max_bwd = dxf
            .iter()
            .zip(dxe.iter())
            .map(|(f, e)| (f - e).abs())
            .fold(0.0f32, f32::max);
        eprintln!("MEASURED max|delta| bf16 bwd (last={last}) = {max_bwd:e}");
        for (i, (f, e)) in dxf.iter().zip(dxe.iter()).enumerate() {
            assert!(
                (f - e).abs() < tol_bwd,
                "bf16 dscores[{i}] (last={last}): fused {f} vs eager {e}"
            );
        }
    }

    /// MEASURED at `last = 128` (ModernBERT-large's profiled `seq`; see the
    /// commit message for the `--nocapture` transcript this bound is taken
    /// from): max|delta| fwd = `1.2207e-3`, bwd = `1.1108e-2`. Bounds below
    /// carry roughly 2.5x margin over the measured value, not a
    /// hypothesized scaling law: the audit named "the divergence mechanism
    /// grows with reduction width" as a real concern worth measuring, and
    /// having measured it at three widths (`4`, `128`, `512`) the growth is
    /// NOT monotonic in this fixture (bwd: `4` → `1.953e-3`, `128` →
    /// `1.1108e-2`, `512` → `4.425e-3`) — width clearly matters (both `128`
    /// and `512` exceed the `last = 4` bound by 2-6x), but the exact
    /// relationship depends on the specific score/mask/`dy` fixture, not a
    /// clean power law this test could assert instead of measuring.
    #[test]
    fn fused_matches_eager_bf16_at_production_width_128() {
        bf16_width_case(128, 3e-3, 2.8e-2);
    }

    /// MEASURED at `last = 512` (the quadratic-regime `seq` the
    /// fused-kernels plan names as this commit's actual target): max|delta|
    /// fwd = `3.9673e-4`, bwd = `4.425e-3`. See `..._128`'s doc for why
    /// these bounds are measured-plus-margin, not a scaling formula.
    #[test]
    fn fused_matches_eager_bf16_at_production_width_512() {
        bf16_width_case(512, 1.2e-3, 1.1e-2);
    }

    // -------------------------------------------------------------------
    // Oracle 3: chain-rule oracle through an intermediate `scores` tensor.
    // -------------------------------------------------------------------
    #[test]
    fn bwd_chains_through_an_intermediate_non_variable_scores() {
        let device = Device::Cpu;
        let w0: [f32; 6] = [0.5, -1.0, 2.0, 0.25, -0.3, 1.1];
        let m0: [f32; 3] = [0.0, -10_000.0, 0.0];
        let w = Var::from_tensor(&Tensor::from_slice(&w0, (2, 3), &device).unwrap()).unwrap();
        let scores = w.affine(2.0, 0.0).unwrap();
        assert!(!scores.is_variable());
        let mask = Tensor::from_slice(&m0, (1, 3), &device).unwrap();

        let out = fused(&scores, &mask).unwrap();
        let grads = out.backward().unwrap(); // must not panic
        let dw: Vec<f32> = grads
            .get(&w)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let scores_direct =
            Var::from_tensor(&Tensor::from_slice(&w0.map(|v| 2.0 * v), (2, 3), &device).unwrap())
                .unwrap();
        let out2 = fused(&scores_direct, &mask).unwrap();
        let grads2 = out2.backward().unwrap();
        let dscores: Vec<f32> = grads2
            .get(&scores_direct)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, (a, b)) in dw.iter().zip(dscores.iter()).enumerate() {
            assert!((a - 2.0 * b).abs() < 1e-4, "dw[{i}]: {a} vs 2*{b}");
        }
    }

    #[test]
    fn mask_gradient_is_none_when_not_a_variable() {
        let device = Device::Cpu;
        let scores = Var::from_tensor(
            &Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap(),
        )
        .unwrap();
        let mask = Tensor::from_slice(&[0.0f32; 4], (1, 4), &device).unwrap();
        assert!(!mask.is_variable());
        let out = fused(&scores, &mask).unwrap();
        let grads = out.backward().unwrap();
        assert!(grads.get(&scores).is_some());
        assert!(grads.get(&mask).is_none());
    }

    #[test]
    fn mask_gradient_is_populated_and_correct_when_a_variable() {
        let device = Device::Cpu;
        let sv: [f32; 8] = [1.0, -1.0, 2.0, 0.5, -0.3, 1.7, 0.2, -2.1];
        let mv: [f32; 4] = [0.1, -0.2, 0.3, -0.1];
        let scores = Tensor::from_slice(&sv, (2, 4), &device).unwrap();
        let mask = Var::from_tensor(&Tensor::from_slice(&mv, (1, 4), &device).unwrap()).unwrap();

        let out = fused(&scores, &mask).unwrap();
        let grads = out.backward().unwrap();
        let dmask: Vec<f32> = grads
            .get(&mask)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Reference via the eager composition's own (real, candle-native)
        // gradient for the broadcast mask add.
        let mask_eager =
            Var::from_tensor(&Tensor::from_slice(&mv, (1, 4), &device).unwrap()).unwrap();
        let out_eager = eager(&scores, &mask_eager).unwrap();
        let grads_eager = out_eager.backward().unwrap();
        let dmask_eager: Vec<f32> = grads_eager
            .get(&mask_eager)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, (f, e)) in dmask.iter().zip(dmask_eager.iter()).enumerate() {
            assert!((f - e).abs() < 1e-3, "dmask[{i}]: fused {f} vs eager {e}");
        }
    }

    /// Verifies, rather than merely asserting in the module doc, that a
    /// fully-masked row's ZERO output propagates to a ZERO gradient with
    /// NO special-casing needed in `SoftmaxBwdDScores`: `dscores = (dy -
    /// sum(dy*y)) * y` with `y == 0` gives `(dy - 0) * 0 == 0` for every
    /// element, purely from the EXISTING formula reading `res` (the
    /// forward's own zero output) and `grad_res`.
    #[test]
    fn fully_masked_row_backward_is_zero_falling_out_of_the_existing_formula() {
        let device = Device::Cpu;
        let last = 4;
        let scores = Var::from_tensor(
            &Tensor::from_slice(&[1.0f32, -2.0, 3.0, -0.5], (1, last), &device).unwrap(),
        )
        .unwrap();
        let mask = Tensor::from_slice(&[f32::NEG_INFINITY; 4], (1, last), &device).unwrap();

        let out = fused_with_policy(&scores, &mask, FullyMaskedPolicy::Zeros).unwrap();
        let out_v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(
            out_v,
            vec![0.0f32; last],
            "sanity: forward must be zero here"
        );

        // Non-uniform, non-zero seed dy -- if backward secretly depended on
        // `dy` alone (a bug that ignored `y`), a non-zero `dy` would leak
        // through into a non-zero gradient.
        let dy_seed = Tensor::from_slice(&[0.7f32, -1.3, 2.1, 0.4], (1, last), &device).unwrap();
        let loss = (&out * &dy_seed).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let dscores: Vec<f32> = grads
            .get(&scores)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(
            dscores,
            vec![0.0f32; last],
            "a fully-masked row's gradient must be exactly zero: {dscores:?}"
        );
    }

    /// Audit advisory 1, verified (not merely asserted in the module doc's
    /// "Inert downstream" section): on the REAL finite-`MASKED_LOGIT`
    /// convention, `dy == 0` (exactly what pooling's own backward produces
    /// for a pad position) drives `dscores` to exactly zero under BOTH
    /// `FullyMaskedPolicy` variants -- the training dynamics through a
    /// fully-masked row are IDENTICAL between the fused/`Zeros` path and
    /// the eager-fallback/`Propagate` path, not merely close.
    #[test]
    fn fully_masked_row_backward_is_zero_under_both_policies_given_pooling_style_zero_dy() {
        let device = Device::Cpu;
        let last = 4;
        let scores_v = [1.0f32, -2.0, 3.0, -0.5];
        let mask_v = [-10_000.0f32; 4];

        for policy in [FullyMaskedPolicy::Zeros, FullyMaskedPolicy::Propagate] {
            let scores =
                Var::from_tensor(&Tensor::from_slice(&scores_v, (1, last), &device).unwrap())
                    .unwrap();
            let mask = Tensor::from_slice(&mask_v, (1, last), &device).unwrap();
            let out = fused_with_policy(&scores, &mask, policy).unwrap();
            let out_v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
            assert!(
                out_v.iter().all(|v| v.is_finite()),
                "policy {policy:?}: y must be finite on the real finite-MASKED_LOGIT \
                 convention, got {out_v:?}"
            );

            // `dy == 0` everywhere, exactly -- the pooling-zeroes-a-pad-row shape.
            let dy = Tensor::from_slice(&[0.0f32; 4], (1, last), &device).unwrap();
            let loss = (&out * &dy).unwrap().sum_all().unwrap();
            let grads = loss.backward().unwrap();
            let dscores: Vec<f32> = grads
                .get(&scores)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(
                dscores,
                vec![0.0f32; last],
                "policy {policy:?}: dscores must be exactly zero when dy==0 on the real \
                 finite-MASKED_LOGIT convention, got {dscores:?}"
            );
        }
    }

    /// The documented caveat to the equivalence above: on the SYNTHETIC
    /// all-`-inf` convention (never used by the real call site — see the
    /// module doc), `Propagate`'s `NaN`-valued `y` poisons its own
    /// gradient regardless of `dy` (`0 * NaN == NaN` in IEEE754), so the
    /// "identical training dynamics" claim does NOT extend to `-inf`
    /// masking under `Propagate`. This is `Propagate` faithfully
    /// reproducing eager's own `NaN` propagation, not a bug -- verified
    /// here so the equivalence claim above is not overclaimed as universal.
    #[test]
    fn propagate_policy_all_inf_row_backward_is_nan_even_with_zero_dy_unlike_the_finite_case() {
        let device = Device::Cpu;
        let last = 4;
        let scores = Var::from_tensor(
            &Tensor::from_slice(&[1.0f32, -2.0, 3.0, -0.5], (1, last), &device).unwrap(),
        )
        .unwrap();
        let mask = Tensor::from_slice(&[f32::NEG_INFINITY; 4], (1, last), &device).unwrap();
        let out = fused_with_policy(&scores, &mask, FullyMaskedPolicy::Propagate).unwrap();
        let out_v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            out_v.iter().all(|v| v.is_nan()),
            "sanity: y must be NaN on an all-inf row under Propagate: {out_v:?}"
        );

        let dy = Tensor::from_slice(&[0.0f32; 4], (1, last), &device).unwrap();
        let loss = (&out * &dy).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let dscores: Vec<f32> = grads
            .get(&scores)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            dscores.iter().all(|v| v.is_nan()),
            "Propagate + all-inf must poison its own gradient with NaN even though \
             dy==0 (0*NaN==NaN in IEEE754): {dscores:?}"
        );
    }

    #[test]
    fn period_style_broadcast_over_batch_and_heads_matches_eager() {
        // scores: [batch=2, heads=3, q=2, k=4]; mask: [batch=2, heads=1,
        // q=1, k=4] -- broadcasts over heads AND the query-row axis, the
        // ModernBERT padding-mask shape (`extended_attention_mask`).
        let device = Device::Cpu;
        let batch = 2;
        let heads = 3;
        let q = 2;
        let k = 4;
        let sv: Vec<f32> = (0..batch * heads * q * k)
            .map(|i| (i as f32 * 0.23 - 3.0).sin() * 2.0)
            .collect();
        let mv: Vec<f32> = (0..batch * k)
            .map(|i| if i % k == 1 { -10_000.0 } else { 0.0 })
            .collect();
        let scores = Tensor::from_slice(&sv, (batch, heads, q, k), &device).unwrap();
        let mask = Tensor::from_slice(&mv, (batch, 1, 1, k), &device).unwrap();

        let got = fused(&scores, &mask)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let want = eager(&scores, &mask)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-5, "elem[{i}]: fused {g} vs eager {w}");
        }
    }

    #[test]
    fn broadcast_over_heads_only_matches_eager_the_local_layer_combined_mask_shape() {
        // scores: [batch=2, heads=2, q=3, k=3]; mask: [batch=2, heads=1,
        // q=3, k=3] -- broadcasts ONLY over heads, the shape the call site
        // builds by pre-summing the padding mask and the sliding-window
        // band for a local-attention layer BEFORE calling this op.
        let device = Device::Cpu;
        let batch = 2;
        let heads = 2;
        let q = 3;
        let k = 3;
        let sv: Vec<f32> = (0..batch * heads * q * k)
            .map(|i| (i as f32 * 0.19 + 1.0).cos() * 1.5)
            .collect();
        let mv: Vec<f32> = (0..batch * q * k)
            .map(|i| {
                if (i % k) as i64 - ((i / k) % q) as i64 == 0 {
                    0.0
                } else {
                    -10_000.0
                }
            })
            .collect();
        let scores = Tensor::from_slice(&sv, (batch, heads, q, k), &device).unwrap();
        let mask = Tensor::from_slice(&mv, (batch, 1, q, k), &device).unwrap();

        let got = fused(&scores, &mask)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let want = eager(&scores, &mask)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-5, "elem[{i}]: fused {g} vs eager {w}");
        }
    }

    // -------------------------------------------------------------------
    // Oracle 8 (MEMORY ORACLE): the fused path retains FEWER tape nodes
    // than eager for the same computation. `Tensor::sorted_nodes()` is
    // candle's own PUBLIC topological-sort-for-backward API (the exact
    // list `Tensor::backward` walks) -- a node appears in it iff it is
    // needed for gradient computation, so this is a direct, honest count
    // of what backward keeps resident, not a proxy or an estimate.
    // -------------------------------------------------------------------
    #[test]
    fn fused_softmax_retains_fewer_tape_nodes_than_eager() {
        let device = Device::Cpu;
        let sv: Vec<f32> = (0..12).map(|i| (i as f32 * 0.3 - 1.0).sin()).collect();
        let mv = [0.0f32; 4];

        let s_fused = Var::from_tensor(&Tensor::from_slice(&sv, (3, 4), &device).unwrap()).unwrap();
        let mask_fused = Tensor::from_slice(&mv, (1, 4), &device).unwrap();
        let y_fused = fused(&s_fused, &mask_fused).unwrap();

        let s_eager = Var::from_tensor(&Tensor::from_slice(&sv, (3, 4), &device).unwrap()).unwrap();
        let mask_eager = Tensor::from_slice(&mv, (1, 4), &device).unwrap();
        let y_eager = eager(&s_eager, &mask_eager).unwrap();

        let fused_nodes = y_fused.sorted_nodes().len();
        let eager_nodes = y_eager.sorted_nodes().len();
        assert!(
            fused_nodes < eager_nodes,
            "fused must retain FEWER tape nodes than eager: fused={fused_nodes} eager={eager_nodes}"
        );
        // Pin the exact fused count too: `scores` (the leaf `Var`) plus the
        // ONE `CustomOp2` node itself -- nothing else survives, because
        // `mask` is not a `Var` here (matching every real call site) and
        // is therefore never pushed onto `sorted_nodes` at all (candle
        // only pushes nodes that participate in gradient tracking).
        assert_eq!(
            fused_nodes, 2,
            "fused tape should be exactly [scores, y] -- no [B,H,S,S]-shaped intermediate \
             survives forward"
        );
        // The eager composition chains through several ordinary `Tensor`
        // ops (broadcast_add, max_keepdim, broadcast_sub, exp, sum_keepdim,
        // broadcast_div), each its own retained node.
        assert!(
            eager_nodes >= 5,
            "sanity: the eager composition must actually retain multiple nodes for this \
             comparison to mean anything, got {eager_nodes}"
        );
    }
}
