//! Fused scaled-dot-product attention block: rotary embedding, `QKᵀ`,
//! an additive mask (the caller's own pre-combined padding-plus-band
//! tensor — see "the window is construction data at the call site" below),
//! softmax, then `PV`, composed inside ONE `CustomOp3`
//! tape node instead of the ~10-op eager chain (`RoPE` twice, a matmul, a
//! scale, up to two mask adds, a softmax, a second matmul, a transpose and
//! a reshape) `jammi-encoders`' ModernBERT attention call site builds today.
//! Generic primitive (family L): this crate names no consumer — the
//! doc below cites ModernBERT's own shapes/values only to explain the
//! numeric choices this op makes, never as a dependency.
//!
//! ## Tier 0: a COMPOSED interior, not a hand-written fused kernel
//!
//! Both device arms reuse this crate's EXISTING primitives at the storage
//! level rather than a new monolithic kernel: `BackendStorage::matmul`
//! (the same cuBLAS/`gemm`-crate call `candle_nn::Linear`/`Tensor::matmul`
//! already issue) for `QKᵀ` and `PV`, and this op's own row-math (CPU) or
//! the EXISTING `RopeFused`/`SoftmaxLastDimFused` kernels (CUDA, invoked
//! directly via their own `CustomOp3::cuda_fwd`/`CustomOp2::cuda_fwd`, the
//! same reuse idiom `LowRankResidualLinear`'s CUDA glue documents for
//! `DropoutFused`/`ScaledCastAdd`) for RoPE and the masked softmax. The WIN
//! is that all of it happens inside ONE `Op::CustomOp3` node: candle's
//! backward tape retains nothing `[batch, heads, seq, seq]`-shaped between
//! forward and backward (the single largest class of retained activation
//! in ModernBERT's training step), where the eager composition retains
//! several.
//!
//! ## The `rope_pack` argument: packing `cos`+`sin` into `CustomOp3`'s
//! third slot
//!
//! `CustomOp3` takes exactly three tensor arguments; this op's contract
//! needs FOUR conceptually independent tensors on the RoPE side alone
//! (`qkv`, `cos`, `sin`, `mask`). Rather than inventing a new
//! representation for the RoPE table, `rope_pack` is `Tensor::stack(&[cos,
//! sin], 0)` of [`RopeFused`]'s OWN `[1, 1, S_max, head_dim]` cos/sin
//! tables — the exact same values `RotaryEmbedding::cached_tables`
//! produces, packed along a new leading axis of size 2 (`rope_pack[0] ==
//! cos`, `rope_pack[1] == sin`) purely to fit `CustomOp3`'s arity. This
//! introduces no new numeric representation and no rounding of its own
//! (a `stack` is a pure memory copy of the SAME bytes); it is a resolved
//! interpretation of the op contract's literal 3-argument constraint, not
//! a numeric design choice — see this crate's hand-off notes for the
//! disclosure.
//!
//! ## Fixed domain: `head_dim == 64` (family D)
//!
//! Unlike every other op in this crate, this one pins `head_dim` to
//! exactly `HEAD_DIM` (`64`) rather than accepting any positive even
//! width. This is load-bearing for the scale fold below, not an arbitrary
//! restriction: `scale = 1 / sqrt(head_dim) = 0.125`, an EXACT power of
//! two, representable without rounding in every float format this op
//! supports. Multiplying `Q` by an exact power of two BEFORE the `QKᵀ`
//! matmul (rather than dividing the `[seq, seq]`-shaped SCORE matrix by
//! `sqrt(head_dim)` AFTER, as the eager composition does) is bit-exact to
//! that post-divide precisely because scaling one GEMM operand by an exact
//! power of two commutes exactly with both the multiply-accumulate chain
//! and the final division (no mantissa rounding is introduced by an
//! exponent-only shift, provided no overflow/underflow — never a concern
//! at this magnitude). Folding the scale into `Q` (`[batch, heads, seq,
//! head_dim]`-sized, not `[batch, heads, seq, seq]`-sized) is the "fold"
//! this crate's P1 commit established for the same reason: one
//! elementwise pass over the SMALLER tensor replaces one over the
//! quadratic one. A generic `head_dim` would not preserve this
//! bit-exactness in general (`1/sqrt(d)` is irrational for most `d`), so
//! this op refuses any other width rather than silently losing the
//! guarantee.
//!
//! ## The window is construction data at the CALL SITE, not this op (family D)
//!
//! This op has no `window`/`half_window` field and computes no band
//! predicate of its own. Whatever band a local-attention caller wants is
//! the caller's construction data: the VALUE contract this op relies on
//! is only that `mask[b, q, k]` is `0.0` for an attendable key and a
//! value `< 0.0` at the [`WINDOW_MASKED_VALUE`] magnitude for a masked
//! one — how the caller builds, caches, or combines the padding and band
//! terms into that value is outside this op's knowledge (family L: this
//! crate names no consumer, so no consumer's cache is a premise here; the
//! sentinel-equality pin lives on the consumer side, in its own mask
//! module). The caller passes this op ONE already-combined additive
//! mask, exactly the shape
//! [`super::SoftmaxLastDimFused`] already accepts (`[batch|1, 1, seq|1,
//! seq]` — see that op's module doc's "supported mask broadcast class").
//! Earlier revisions of this op re-derived the SAME band predicate three
//! times more (once per CPU-forward row, once in a CUDA scratch-mask
//! builder, once in `bwd`'s recompute) on top of the encoder's own copy —
//! four copies of `i.abs_diff(k) <= half_window` for one predicate. This
//! design removes all three: the op reads whatever additive value the
//! caller's combined mask carries at `(b, q, k)` and never asks whether
//! that value came from padding, a window band, or both — it does not
//! need to know, since [`WINDOW_MASKED_VALUE`] (`-10_000.0`) and
//! `jammi_encoders::mask::MASKED_LOGIT` are the SAME numeric sentinel (see
//! the encoder-side test pinning that equality), so a doubly-masked key
//! (`padded AND out-of-window`, `-20_000.0`) still reads as "masked, not
//! attendable" under the exact same `< 0.0` rule
//! [`super::softmax::row_is_fully_masked`] already applies.
//!
//! A row `(b, q)` is fully masked (every key masked) iff `max_k
//! combined_mask[b, q, k] < 0.0` — [`FullyMaskedPolicy`] governs what
//! happens there, identically to [`super::SoftmaxLastDimFused`] (this op
//! reuses that exact policy type and, for the CPU arm, that exact row
//! math — [`super::softmax::softmax_row_f32`] directly).
//!
//! `mask`'s domain therefore widened from the padding-only `[batch|1, 1,
//! 1, seq]` shape earlier revisions required to `[batch|1, 1, seq|1,
//! seq]` — the `seq|1` third axis lets a global layer keep passing the
//! narrower padding-only shape (no query-row broadcast needed) while a
//! local layer passes the wider padding-plus-band sum. See
//! [`check_mask`].
//!
//! ## Domain (family D)
//!
//! `qkv`: rank 5 `[batch, seq, 3, heads, head_dim]`, contiguous, dtype
//! `F32` (CPU and CUDA) or `BF16`/`F16` (CUDA only — candle-core 0.11's CPU
//! backend has no `BF16`/`F16` `MatMul` impl, the SAME pre-existing
//! limitation `LowRankResidualLinear`'s module doc discloses; this op's CPU
//! domain therefore accepts `F32` only, refusing `BF16`/`F16` with a typed
//! `UnsupportedDTypeForOp` rather than reaching a confusing failure three
//! calls deep inside a matmul). `F16` (campaign #443 D1) is admitted on
//! CUDA on exactly the same basis as `BF16`: this op has no `.cu` kernel of
//! its own (`crate::cuda::attention_block`'s module doc), so its dtype
//! domain is the INTERSECTION of what candle's own generic storage ops
//! support and what its two composed sub-kernels ([`RopeFused`],
//! [`super::SoftmaxLastDimFused`]) dispatch for — both now compile real
//! `F16` arms (campaign #443 W2b). `head_dim` must be exactly `HEAD_DIM`.
//! `seq` must be `<= MAX_SEQ`. `rope_pack` (when `rope == true`): rank
//! 5 `[2, 1, 1, seq_max, head_dim]`, `seq_max >= seq`, contiguous, same
//! dtype as `qkv`. `mask`: rank 4 `[batch|1, 1, seq|1, seq]`, contiguous,
//! same dtype as `qkv` — the SAME broadcast class
//! [`super::SoftmaxLastDimFused`] accepts, since the caller may pass
//! either the padding mask alone (global layer) or the padding-plus-band
//! sum (local layer) — see the "window is construction data at the call
//! site" section above. `b == 0 || seq == 0 || heads == 0` is in-domain
//! and yields an empty `[batch, seq, heads * head_dim]` output: the CPU
//! arm computes it through its general path (zero-trip loops, zero-extent
//! GEMMs — no separate fast path, see `cpu_fwd`); the CUDA arm returns an
//! empty allocation before touching cuBLAS. The domain checks above apply
//! to empty inputs exactly as to non-empty ones.
//!
//! ### `BF16` validated-coverage ceiling: `|qkv| <= 1` for the fused-vs-eager bound
//!
//! Same status as [`MAX_SEQ`]'s own doc: a VALIDATED-COVERAGE ceiling, not
//! a hardware or type limit — the op ACCEPTS any finite `BF16` `qkv` on
//! CUDA, but `tests/cuda_parity.rs`'s `Bf16LegBounds` derived-bound
//! framework (this crate's ONLY non-vacuous fused-vs-eager guarantee at
//! `BF16`) is proven only up to `|qkv| <= 1`: the softmax-flip term in
//! that bound is `ulp(S_max) * V_max / 2`, and `ulp` grows with the
//! VALUE — at the fixture's `|qkv| <= 1` amplitude `S_max` is `O(1)`-`O(4)`
//! and the bound is a few percent of signal; at `|qkv| = 10` `S_max` is
//! `O(400)`, `ulp(S_max)` is `O(2)`, and the resulting bound EXCEEDS the
//! gradient signal itself — no non-vacuous bound exists there, so no test
//! in this crate asserts one. This is a REAL gap, not a hypothetical one:
//! `tests::fused_attention_block_matches_eager_lora_gradients_at_production_seq_on_head64`
//! (`jammi-encoders`, `src/modernbert.rs`) measured `max|qkv| ≈ 9`–`18`
//! (one outlier at `1.775e1`) on the REAL ModernBERT-large checkpoint at
//! `batch=8, seq=512`, `seed=42`, sha `8922094aa35d381d108420fefe82cba122bf6ebb`
//! (`JAMMI_DEBUG_QKV_AMP=1` on the `Wqkv` output feeding
//! `forward_training_attention`'s fused arm) — an order of magnitude past
//! this ceiling. The op still admits and runs correctly there (this op's
//! own defect history was a GEMM-operand-form determinism issue — see
//! "GEMM operand form is a determinism concern" above — NOT an
//! amplitude-domain violation), but this crate makes NO derived-bound
//! CLAIM at that amplitude: the encoder-level oracle above compares fused
//! vs eager through the REAL call site at `F32` (this op's CPU-only dtype)
//! specifically because no bf16 bound at production amplitude exists to
//! assert. A future CUDA-side production-amplitude bf16 bound would need
//! either a tighter softmax-flip model or accepting a much larger
//! constant.
//!
//! ## `bwd`: ordinary `Tensor` composition, reusing this crate's own ops
//!
//! Candle has no save-for-backward channel (the same constraint
//! [`super::LayerNormFused`]'s and [`super::RopeFused`]'s own `bwd`
//! methods document), so `bwd` recomputes the rotated `Q`/`K`, the raw
//! scores, and the softmax output `P` from `qkv`/`rope_pack`/`mask` —
//! calling [`super::apply3`] with [`super::RopeFused`] and
//! [`super::apply2`] with [`super::SoftmaxLastDimFused`] DIRECTLY, rather
//! than a second hand-written kernel. This is the same pattern
//! [`super::RopeFused::bwd`] and [`super::SoftmaxLastDimFused::bwd`]
//! already use (composing ordinary `Tensor` ops, including calls into EACH
//! OTHER's `apply*` entry points, inside their own `bwd`), just with a
//! longer chain — and, unlike them, it runs DETACHED (next section), which
//! is what makes the "no retained `[batch, heads, seq, seq]` activation"
//! claim true rather than merely stated. `bwd` forces
//! `.contiguous()` on every GEMM operand that is not either fully
//! row-major OR a single transposed view of one (`gemm_config`'s admissible
//! shapes — see `LowRankResidualLinear`'s module doc for the citation and
//! the on-device failure that motivated checking this explicitly); the
//! `is_gemm_operand_admissible` test below proves this holds for every
//! operand `bwd` actually builds, at both a boundary and a production-scale
//! rank, off the real `Layout` each carries (device-independent, mirroring
//! `LowRankResidualLinear`'s own precedent).
//!
//! ### `bwd`'s six GEMMs: which recompute a forward GEMM, which are new,
//! and which operand form each targets
//!
//! `fwd` issues exactly two GEMMs: `scores = q_scaled · kᵀ` (`k`'s
//! transpose is a VIEW — cuBLAS `OP_T`, no materialize) and `ctx = p · v`
//! (both row-major — `OP_N`/`OP_N`). `bwd` reissues ONE of those two
//! (`scores`, to recompute `P`) and adds four NEW gradient GEMMs, each
//! issued through [`matmul_grad_lhs`]/[`matmul_grad_rhs`] — the ONE
//! definition of "the gradient GEMM of `A@B`" this op shares with
//! candle's own generic `Op::Matmul` backward (see those functions' own
//! doc), so a gradient GEMM's operand form is chosen once, not re-derived
//! independently at each call site:
//!
//! | `bwd` GEMM | recompute of a fwd GEMM? | operand form target |
//! |---|---|---|
//! | `scores = q_scaled · k_rotᵀ` | YES — `fwd`'s `QKᵀ`, same shape | matches `fwd`'s OWN operand form (`k_rot` transpose VIEW, no `.contiguous()`) — see "fwd/bwd GEMM shape match" below |
//! | `dv = matmul_grad_rhs(p, dctx)` | no — new | `p` transpose VIEW (already contiguous, no extra materialize) |
//! | `dp = matmul_grad_lhs(dctx, v_c)` | no — new | `v_c` transpose VIEW (already contiguous) |
//! | `dqs = matmul_grad_lhs(ds, kt_contig)` | no — new | `kt_contig`, a MATERIALIZED transpose of `k_rot` — NOT a view of it |
//! | `dkr = matmul_grad_rhs(q_scaled, ds)` then `.transpose()` | no — new | `q_scaled` transpose VIEW (already contiguous); the trailing `.transpose()` is a view too |
//!
//! (`ctx` itself — `fwd`'s SECOND GEMM — is never recomputed in `bwd`: the
//! op contract's "no packed `[O|L]` output" design means `bwd` only ever
//! needs `dctx`, supplied by the caller as `grad_res`, never `ctx` itself.)
//!
//! `dv`/`dp` reach production's own operand form for free: `p` and `v_c`
//! are ALREADY naturally contiguous on both the fused and the production
//! eager path (`crate::contiguous_matmul`'s `.contiguous()` on an
//! already-contiguous tensor is a no-op clone), so
//! `matmul_grad_rhs`/`matmul_grad_lhs` see the identical operand shapes
//! either way. `dqs`/`dkr` do NOT get that for free: production's forward
//! for the `scores` GEMM (`crate::contiguous_matmul`,
//! `crates/jammi-encoders/src/lib.rs:139-141`, called from
//! `modernbert.rs:1016`) materializes BOTH operands, unlike `fwd`'s own
//! view-based `scores` GEMM this op recomputes above — so `bwd` builds a
//! SEPARATE materialized `kt_contig` specifically for the `dqs`/`dkr`
//! gradient GEMMs (never for the `scores` recompute, which keeps `fwd`'s
//! own view form — see "fwd/bwd GEMM shape match" below for why those two
//! targets are legitimately different GEMMs with legitimately different
//! operand-form rules), so `matmul_grad_lhs`/`matmul_grad_rhs` differentiate
//! the SAME `Op::Matmul(q_scaled, kt_contig)` shape production's own
//! autograd does, byte-for-byte, not merely value-for-value.
//!
//! ### GEMM operand form is a determinism concern, not just admissibility
//!
//! cuBLAS's bf16 strided-batched GEMM picks its internal accumulation
//! order from the operand's `(rows, cols, row_stride, col_stride)`, and a
//! materialized-contiguous operand and a transposed-view operand of the
//! SAME logical matrix can drive cuBLAS to different blocking, producing
//! a different bf16-rounded result for that one GEMM. Any per-GEMM
//! difference this causes is small (a few ULPs — `Bf16LegBounds` in
//! `tests/cuda_parity.rs`) but SYSTEMATIC (a fixed function of the two
//! operand forms, not i.i.d. noise): it does not average out across a
//! training step or a layer stack. A prior round of this fix dropped
//! `.contiguous()` from `dv`/`dp`/`dkr` on the theory that candle's
//! `Op::Matmul` backward "never materializes"; that is true of `dv`/`dp`
//! (whose forward operands are already contiguous either way) but false
//! of `dqs`/`dkr`, whose forward (`scores`) IS materialized in
//! production — a mismatch the prior round's own oracles could not see:
//! a CPU/F32 encoder-level comparison runs at a dtype/device pair with no
//! cuBLAS bf16 blocking sensitivity at all, and this file's bf16
//! derived-bound legs compare a SINGLE `bwd` call, whose own
//! `7 * bf16_ulp(dqkv_max)` floor swallows a divergence this small by
//! construction (a single call cannot show that the divergence is
//! SYSTEMATIC rather than ordinary bf16 rounding noise — only compounding
//! it across a step can). `matmul_grad_lhs`/`matmul_grad_rhs` plus
//! `kt_contig` (previous section) close the gap STRUCTURALLY: the fused
//! and production GEMMs are the same call by definition now, not merely
//! measured close —
//! `tests::attention_block_bwd_dqs_dkr_gemm_layouts_match_production_
//! orientation_cuda` (`tests/cuda_parity.rs`, via
//! [`bwd_gradient_gemm_layouts`]) proves it directly: it captures each
//! gradient GEMM's operand `Layout` FROM `bwd_core`'s own code (not a
//! fixture reconstructed separately from it) and asserts two STRUCTURAL
//! properties that flip specifically between the pre-round-4 operand
//! forms and this round's fix (that test's own doc has the exact
//! stride/shape claims), so a future re-introduction of a bare
//! `k_rot`/`q_scaled` operand in place of `kt_contig` fails immediately,
//! structurally, without needing bf16 noise to average out first.
//!
//! This op's numeric backstop — for the ORIGINAL divergence this round
//! exists to guard against, not for the `dqs`/`dkr` change specifically
//! (see the next paragraph) — lives one crate up:
//! `jammi_encoders::modernbert::tests::attention_block_fused_vs_eager_
//! dqkv_divergence_grows_with_depth_bf16_cuda` (`jammi-encoders`,
//! `src/modernbert.rs`) drives the REAL `forward_training_attention`/
//! `forward_eager_training_attention_composition` 28 layers deep from
//! ONE tracked `qkv` `Var` and asserts the fused/eager divergence does
//! NOT grow with depth. It is RED with the original three
//! `.contiguous()` calls restored on `dv`/`dp`/`dkr`, GREEN at this
//! file's tip — both readings reported, not merely asserted, in this
//! round's hand-off. Measured on the SAME pod run: reverting ONLY
//! `dqs`/`dkr`'s operand form to the pre-round-4 form (`dv`/`dp` left at
//! this round's fix) does NOT redden that oracle — its step-1 `r(1)` is
//! exactly `0.0` for every slot, bit-identical to the fully-fixed build,
//! at `b=8, s=512, h=16, d=64` bf16. The `dqs`/`dkr` change above is
//! therefore justified STRUCTURALLY (the layout test, previous
//! paragraph), not as a numerically-demonstrated-necessary fix at this
//! shape; whether it matters at another shape or another GPU/driver is
//! UNCONFIRMED — stated here, not claimed.
//!
//! ### The two-armed rule: which GEMMs stay a VIEW, which get materialized
//!
//! An earlier revision of this doc claimed every `bwd` GEMM operand is
//! "either genuinely contiguous or a bare transposed VIEW — never a copy
//! this op chose to make for its own sake". That is false on its face:
//! `dctx` (`grad_res.reshape(...).transpose(1, 2)?.contiguous()?`) IS a
//! copy `bwd` makes, unconditionally, every call. The REAL rule has two
//! arms, and both are matches to a SPECIFIC real thing, never "for its
//! own sake": (1) a GEMM that RECOMPUTES a `fwd` GEMM (`scores`, the one
//! row in the table above) keeps `fwd`'s OWN operand form — a view stays
//! a view, because that is what `fwd` itself issues, and `bwd`'s job
//! there is to reproduce it exactly (see "fwd/bwd GEMM shape match"
//! below). (2) every OTHER GEMM's operand form matches whatever
//! PRODUCTION's OWN autograd would hand it, and production's own
//! `Op::Matmul` backward NEVER hands a bare, unaccumulated view for an
//! UPSTREAM gradient (as opposed to a forward-graph operand): candle
//! accumulates every gradient through `GradStore::or_insert`
//! (`backprop.rs:768-777`), which starts from `tensor.zeros_like()` (a
//! fresh, CONTIGUOUS buffer) and `.add()`s into it — so the `grad`
//! `Op::Matmul`'s own backward formula receives is ALWAYS contiguous on
//! production's side too, whether or not the value it wraps came from a
//! transposed view somewhere upstream. `dctx` (this op's own analogue of
//! that `grad`) and `kt_contig` (`scores`'s forward operand, materialized
//! on production via `crate::contiguous_matmul`) are both materialized
//! for exactly this reason — matching a specific, cited real behaviour on
//! the production side, not "for its own sake" and not merely "because it
//! was needed for admissibility".
//!
//! ### fwd/bwd GEMM shape match (`scores`, the one recomputed GEMM)
//!
//! `fwd` computes `scores` from a TRANSPOSED VIEW of `k` (cuBLAS `OP_T`,
//! no separate materialize — see `attention_fwd_f32`'s `k_t_layout` on
//! CPU, `cuda::attention_block::cuda_fwd`'s `k_t_l` on CUDA). `bwd`'s own
//! `scores` recompute uses the SAME transposed VIEW on `k_rot` (no
//! `.contiguous()` call) rather than materializing a contiguous transpose
//! (cuBLAS `OP_N` on a copy) — the two GEMMs `fwd` and `bwd` issue for
//! `scores` are therefore identical in shape AND transpose mode on both
//! devices, not merely equal in VALUE. `Tensor::matmul`'s public API
//! (`candle_core::Tensor::matmul`, `tensor.rs`) passes each operand's own
//! `Layout` straight to `BackendStorage::matmul` without forcing
//! contiguity first, so a transposed VIEW here is exactly as valid a GEMM
//! operand as an explicit `.contiguous()` copy — `is_gemm_operand_admissible`
//! accepts both forms, and the module's own oracle
//! (`bwd_every_gemm_operand_is_admissible_at_boundary_and_production_ranks`)
//! proves the VIEW form specifically.
//!
//! Two SEPARATE claims sit in this section, and only one of them is
//! guaranteed: (1) `fwd`'s `scores` GEMM and `bwd`'s `scores` recompute issue
//! the IDENTICAL operand shape/transpose-mode pair — ARCHITECTURAL,
//! guaranteed by this op's own source on every device, checked by
//! `bwd_every_gemm_operand_is_admissible_at_boundary_and_production_ranks`.
//! (2) that this op's OWN `scores` GEMM produces byte-identical output to
//! a DIFFERENT implementation's equivalent GEMM (an eager reference, on
//! either device) — CONTINGENT on the underlying `gemm`/cuBLAS library
//! choosing the same accumulation order for the two operand forms being
//! compared, which is NOT guaranteed in general (see "GEMM operand form
//! is a determinism concern" above) and does not hold for every shape:
//! `tests::attention_block_diag_bf16_fused_cublas_cross_form_determinism_probe_cuda`
//! (`tests/cuda_parity.rs`) is a labelled, derived-bound probe of exactly
//! this non-guarantee, deliberately keeping ONE eager reference in the
//! materialized (non-matching) form; every OTHER fused-vs-eager oracle in
//! this crate matches operand forms on both sides specifically so it
//! tests VALUE correctness, not cuBLAS's own internal determinism.
//!
//! ### `bwd` runs DETACHED: why the drops below free anything at all
//!
//! In production `qkv` is a TRACKED graph node (downstream of the `Wqkv`
//! LoRA `Var`s), and candle records an `Op` on every result whose input
//! tracks — `BackpropOp::new1/new2/new3` (candle-core 0.11.0 `op.rs`)
//! clone each tracking argument INTO the result's `Op`. Had `bwd` composed
//! its chain from the tracked `qkv` directly, every intermediate (`scores`
//! inside `p`'s `Op::CustomOp2`, `p`/`dp` inside `ds`'s, `ds` inside
//! `dqs`'s `Op::Matmul` …) would have stayed alive through its consumer's
//! `Op` no matter when the local binding was dropped, the four explicit
//! `drop`s would have freed nothing, and the entire chain would have been
//! handed back to the engine inside `dqkv`'s own `Op` — where it lives
//! until the engine detaches the `Var`'s accumulated grad, i.e. until the
//! backward pass finishes with that `Var`'s node. Four `[B, H, S, S]`-
//! shaped tensors (`scores`, `p`, `dp`, `ds` — `pᵀ`/`dsᵀ`/`vᵀ` are VIEWS,
//! not separate allocations, see "GEMM operand form is a determinism
//! concern" above) per layer, ≈ 268.4 MB at the shape below, would have
//! been the true sustained cost.
//!
//! `bwd` therefore starts by calling `Tensor::detach` on all four of its
//! tensor inputs (`qkv`, `rope_pack`, `mask`, `grad_res`) — the SAME move
//! candle's own engine makes on the incoming `grad` before calling any
//! `bwd` (`backprop.rs`: "call `.detach` to avoid computing the backprop
//! graph of the backprop itself"). `detach` shares storage (no copy) and
//! yields an untracked tensor, so NOTHING built downstream records an
//! `Op`, no input is cloned anywhere, `dqkv` returns with an EMPTY tape
//! (`!track_op()`, `sorted_nodes().len() == 0` — the test
//! `bwd_from_a_tracked_qkv_returns_an_untracked_dqkv_with_no_tape` drives
//! this from a `Var`, since a leaf fixture could not observe retention),
//! and the drop points below are the REAL free points. Second-order
//! derivatives through this op are therefore not supported — the same
//! standing limitation candle's engine already imposes on every `bwd`.
//!
//! ### Transient scoping: the live set, derived from the drop points
//!
//! With every intermediate untracked, a `[B, H, S, S]`-shaped tensor is
//! freed exactly when its last `Tensor` handle goes away. Walking `bwd`'s
//! statements in order (each row names what is alive AFTER the statement):
//!
//! | statement | `[B, H, S, S]` tensors alive | count |
//! |---|---|---|
//! | `scores = q_scaled · k_rotᵀ` | `scores` | 1 |
//! | `p = softmax(scores, mask)` | `scores`, `p` | 2 |
//! | `drop(scores)` | `p` | 1 |
//! | `dv = pᵀ (view) · dctx` | `p` (`pᵀ` is a transposed VIEW of `p`'s own storage — zero new bytes) | 1 |
//! | `dp = dctx · vᵀ (view)` | `p`, `dp` | 2 |
//! | `ds = softmax_bwd(p, dp)` | `p`, `dp`, `ds` | 3 |
//! | `drop(p); drop(dp)` | `ds` | 1 |
//! | `dqs = ds · k_rot` | `ds` (already contiguous — no clone at all) | 1 |
//! | `dkr = dsᵀ (view) · q_scaled` | `ds` (`dsᵀ` is a transposed VIEW, zero new bytes — as for `dv`) | 1 |
//! | `drop(ds)` | — | 0 |
//!
//! The SUSTAINED maximum is therefore TWO (`p` alongside `dp`), the
//! momentary maximum THREE (while `ds` is being written from `p` and
//! `dp` — the one point three GENUINE `[B, H, S, S]` buffers are alive at
//! once; `dv`/`dkr`'s transposed operands add none, since a transpose is a
//! VIEW), and NOTHING `[B, H, S, S]`-shaped survives past `drop(ds)` — in
//! particular nothing is returned to the engine. At `b=8, h=16, s=512`
//! (BF16, `2` bytes/element — the training arm's own dtype) one
//! `[8, 16, 512, 512]` tensor is `8·16·512·512·2 = 67_108_864` bytes
//! (≈ 67.1 MB): sustained ≈ 134.2 MB, momentary peak ≈ 201.3 MB, against
//! the ≈ 268.4 MB (four tensors, above) the tracked composition would have
//! held until the engine released the `Var`'s grad. These are DERIVATIONS
//! from the drop points, not measurements; the CUDA run artifact
//! (`artifacts/cuda-runs/`) records the measured `peak_vram` per bench
//! leg.
//!
//! `rope_pack`/`mask` are asserted `!track_op()` at the top of `bwd` — this
//! op computes no `dcos`/`dsin`/`dmask` (unlike `RopeFused`/
//! `SoftmaxLastDimFused`, which DO compute those for the dead-in-practice
//! case a future trainable table/mask would need); a caller that
//! nonetheless makes either argument trainable gets a loud, typed refusal
//! here rather than a silently-missing gradient (family D) — this is the
//! op contract's "construction-time `!track_op()` asserts on args 2, 3",
//! enforced at the one point real `Tensor` values (rather than merely
//! construction data) are actually available to check it against.
//!
//! ## Rounding (CPU / F32, and the composed-CUDA arm's identical order)
//!
//! Forward: rotate `Q`, `K` (bit-exact to [`super::RopeFused`], since this
//! op reuses that op's own row math on CPU and that op's own kernel on
//! CUDA) → fold `scale` into `Q` (exact, see "Fixed domain" above) → `QKᵀ`
//! (`f32` accumulate throughout on this op's F32-only CPU domain) → add the
//! caller's already-combined mask (padding alone, or padding-plus-band —
//! either way exactly `0.0` or a value at the `MASKED_LOGIT` magnitude —
//! no meaningful rounding at F32 for scores of `O(1)`-`O(10)` combined
//! with either exact term) → softmax (bit-exact to
//! [`super::SoftmaxLastDimFused`]'s own row math, reused directly) → `PV`
//! (`f32` accumulate). No `BF16` rounding points exist on the CPU arm at
//! all (this op's CPU domain is F32-only); the CUDA arm's `BF16` rounding
//! points are exactly [`super::RopeFused`]'s and
//! [`super::SoftmaxLastDimFused`]'s own documented ones, reused unchanged.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp3, Error, Layout, Result, Shape, Tensor, D};

use super::rope::rope_fwd_row_f32;
use super::softmax::{softmax_row_f32, SoftmaxBwdDScores};
use super::{apply2, apply3, FullyMaskedPolicy, RopeFused, SoftmaxLastDimFused};

/// The only `head_dim` this op accepts. See the module doc's "Fixed
/// domain" section for why this is load-bearing (the scale fold's
/// bit-exactness), not a validated-coverage ceiling like most other
/// `MAX_*` constants in this crate.
pub const HEAD_DIM: usize = 64;

/// The largest `seq` this op accepts. A conservative, VALIDATED ceiling —
/// not a hardware limit — mirroring every other `MAX_*` constant in this
/// crate (see e.g. `ops::softmax::MAX_LAST_DIM`'s doc for the same status).
pub const MAX_SEQ: usize = 4096;

/// `jammi_encoders::mask::MASKED_LOGIT`'s numeric value, cited BY VALUE —
/// the encoder-side sliding-window band this op's callers pre-combine into
/// `mask` (see the module doc's "window is construction data at the call
/// site" section) uses this exact sentinel for an out-of-window key, and
/// this crate's own `< 0.0` fully-masked-row rule
/// ([`super::softmax::row_is_fully_masked`]) only needs the two crates'
/// sentinels to agree in SIGN, not in magnitude — but pinning the same
/// magnitude here means a doubly-masked key's combined value
/// (`WINDOW_MASKED_VALUE * 2` in the worst case) stays comfortably clear
/// of the F32/BF16 rounding noise floor around `0.0`, not merely on the
/// correct side of it by an arbitrarily thin margin. See
/// `jammi-encoders`' own test pinning `MASKED_LOGIT == WINDOW_MASKED_VALUE`
/// (family L: this crate names no consumer, so the pin lives on the
/// encoder side, which DOES know this constant by name).
pub const WINDOW_MASKED_VALUE: f32 = -10_000.0;

/// Fused attention block. See the module doc for the full design.
///
/// Constructed ONLY through [`AttentionBlockFused::new`] — `scale` is a
/// private field (see its own doc), so a struct literal does not compile
/// from outside this module, and the power-of-two guard in `new` cannot be
/// bypassed:
///
/// ```compile_fail,E0451
/// use jammi_kernels::ops::{AttentionBlockFused, FullyMaskedPolicy};
/// // `0.1` is exactly the value `new` refuses; a literal would smuggle it in.
/// let _op = AttentionBlockFused {
///     scale: 0.1,
///     fully_masked: FullyMaskedPolicy::Zeros,
///     rope: true,
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AttentionBlockFused {
    /// The scaled-dot-product scale, `1 / sqrt(head_dim)` — folded into
    /// `Q` before `QKᵀ` (see the module doc's "Fixed domain" section for
    /// why this is bit-exact only because `head_dim == ``HEAD_DIM`).
    ///
    /// PRIVATE, exactly as [`super::SoftmaxLastDimFused`]'s own `scale`
    /// is, and for the same reason: `f32` has invalid inhabitants for this
    /// field (`0.0`, negative, `NaN`, `±inf`, and — specific to this op —
    /// any value that is not an exact power of two, which voids the
    /// fold-into-`Q` bit-exactness argument). A `pub` field would let a
    /// struct literal set one directly, bypassing [`Self::new`]'s guard
    /// entirely. The only ways to set or read it from outside this module
    /// are [`Self::new`] (construction, checked) and [`Self::scale`] (the
    /// accessor). [`Self::fully_masked`] and [`Self::rope`] stay `pub`
    /// because neither type has an invalid inhabitant (every enum variant
    /// is a valid policy; both `bool`s are valid), mirroring
    /// `SoftmaxLastDimFused`'s `pub fully_masked` / private `scale` split.
    scale: f32,
    /// See [`super::FullyMaskedPolicy`]'s own doc; reused unchanged.
    pub fully_masked: FullyMaskedPolicy,
    /// Whether `rope_pack` is applied to `Q`/`K` at all. `false` lets a
    /// future caller with no positional embedding reuse this op — the
    /// `rope_pack` argument is then present but ignored (never read).
    pub rope: bool,
}

impl AttentionBlockFused {
    /// `scale.is_finite()` alone only rules out `NaN`/`±inf` — it says
    /// nothing about the bit-exactness argument the module doc's "Fixed
    /// domain" section actually rests on: that folding `scale` into `Q`
    /// BEFORE `QKᵀ` is bit-exact to dividing the score matrix by
    /// `sqrt(head_dim)` AFTER, which holds ONLY when `scale` is an EXACT
    /// power of two (an exponent-only shift introduces no mantissa
    /// rounding; any other value would). The DETERMINANT for "is this f32
    /// value an exact power of two" is its mantissa bits: IEEE 754
    /// binary32 stores a normal value as `(1 + mantissa/2^23) * 2^exponent`,
    /// so a value is an exact power of two (or `0.0`/subnormal, both
    /// refused separately below) iff its 23-bit mantissa field is entirely
    /// zero — `scale.to_bits() & 0x007f_ffff == 0`. Refuses any `scale`
    /// that fails this check with a typed error rather than silently
    /// accepting a value whose bit-exactness claim this op cannot actually
    /// make (family D): `0.125` (`1/sqrt(64)`, this op's own production
    /// value) passes; `0.1` (used by earlier test fixtures as an
    /// "arbitrary" scale) is refused.
    pub fn new(scale: f32, fully_masked: FullyMaskedPolicy, rope: bool) -> Result<Self> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(Error::Msg(format!(
                "attention_block_fused: scale must be finite and strictly positive (matching \
                 SoftmaxLastDimFused::with_scale's identical domain), got {scale}"
            )));
        }
        if scale.to_bits() & 0x007f_ffff != 0 {
            return Err(Error::Msg(format!(
                "attention_block_fused: scale must be an EXACT power of two (nonzero mantissa \
                 bits: {:#x}) — the fold-scale-into-Q bit-exactness argument (module doc's \
                 \"Fixed domain\" section) depends on it; got {scale}",
                scale.to_bits() & 0x007f_ffff
            )));
        }
        Ok(Self {
            scale,
            fully_masked,
            rope,
        })
    }

    /// Reads the validated [`Self::scale`] — always a value that already
    /// passed [`Self::new`]'s finite/positive/power-of-two check; with the
    /// field private, no struct literal or field assignment outside this
    /// module can put a refused value here.
    pub fn scale(&self) -> f32 {
        self.scale
    }
}

impl super::sealed::Sealed for AttentionBlockFused {}

/// Validates `qkv`'s domain (module doc). Returns `(batch, seq, heads,
/// head_dim)`.
///
/// State table (one test per row, `tests::attention_dims_*` /
/// `tests::head_dim_other_than_64_is_refused`):
///
/// | rank | `dims[2]` | `head_dim` | `seq` | outcome |
/// |---|---|---|---|---|
/// | ≠ 5 (`dims[2] == 3` still) | — | — | — | refused (rank) |
/// | 5 | ≠ 3 | — | — | refused (3-axis) |
/// | 5 | 3 | ≠ `HEAD_DIM` | — | refused (head_dim) |
/// | 5 | 3 | `HEAD_DIM` | `> MAX_SEQ` (`MAX_SEQ + 1`, `MAX_SEQ + 2`) | refused (seq) |
/// | 5 | 3 | `HEAD_DIM` | `== MAX_SEQ` (the boundary itself) | accepted |
/// | 5 | 3 | `HEAD_DIM` | `< MAX_SEQ` (incl. `0`) | accepted |
pub(crate) fn attention_dims(
    l_qkv: &Layout,
    op: &'static str,
) -> Result<(usize, usize, usize, usize)> {
    let dims = l_qkv.dims();
    if dims.len() != 5 || dims[2] != 3 {
        return Err(Error::Msg(format!(
            "{op}: qkv must be rank 5 [batch, seq, 3, heads, head_dim], got {dims:?}"
        )));
    }
    let (b, s, h, d) = (dims[0], dims[1], dims[3], dims[4]);
    if d != HEAD_DIM {
        return Err(Error::Msg(format!(
            "{op}: head_dim must be exactly {HEAD_DIM} (see this op's module doc's \"Fixed \
             domain\" section — the scale fold's bit-exactness depends on it), got {d}"
        )));
    }
    if s > MAX_SEQ {
        return Err(Error::Msg(format!(
            "{op}: seq={s} exceeds MAX_SEQ={MAX_SEQ} (a conservative validated ceiling, not a \
             hardware limit)"
        )));
    }
    Ok((b, s, h, d))
}

/// Validates `mask`'s domain (module doc). Returns the mask's own leading
/// (batch) axis size and its query-row axis size (`1` or `s`) — see the
/// module doc's "window is construction data at the call site" section:
/// this op's `mask` is now [`super::SoftmaxLastDimFused`]'s OWN broadcast
/// class (padding alone, `[batch|1, 1, 1, seq]`, or padding-plus-band,
/// `[batch|1, 1, seq, seq]`), not a narrower padding-only shape.
///
/// State table (one test per row, `tests::check_mask_*`; `b`/`s` are
/// `qkv`'s own batch/seq):
///
/// | rank | `dims[0]` | `dims[1]` | `dims[2]` | `dims[3]` | outcome |
/// |---|---|---|---|---|---|
/// | ≠ 4 (every axis it has otherwise valid) | — | — | — | — | refused (rank) |
/// | 4 | — | ≠ 1 | — | — | refused (heads axis) |
/// | 4 | — | 1 | — | ≠ `s` | refused (key axis) |
/// | 4 | — | 1 | ∉ {1, `s`} | `s` | refused (query-row axis) |
/// | 4 | ∉ {1, `b`} | 1 | ∈ {1, `s`} | `s` | refused (leading axis) |
/// | 4 | 1 | 1 | 1 | `s` | accepted (padding-only, broadcast over batch) |
/// | 4 | `b` | 1 | `s` | `s` | accepted (padding-plus-band, per batch) |
pub(crate) fn check_mask(
    l_mask: &Layout,
    b: usize,
    s: usize,
    op: &'static str,
) -> Result<(usize, usize)> {
    let dims = l_mask.dims();
    if dims.len() != 4 || dims[1] != 1 || dims[3] != s {
        return Err(Error::Msg(format!(
            "{op}: mask must be [batch|1, 1, {s}|1, {s}], got {dims:?}"
        )));
    }
    if dims[2] != 1 && dims[2] != s {
        return Err(Error::Msg(format!(
            "{op}: mask's query-row axis must be 1 or seq={s}, got {}",
            dims[2]
        )));
    }
    if dims[0] != 1 && dims[0] != b {
        return Err(Error::Msg(format!(
            "{op}: mask's leading axis must be 1 or batch={b}, got {}",
            dims[0]
        )));
    }
    Ok((dims[0], dims[2]))
}

/// Validates `rope_pack`'s domain (module doc, only when `self.rope`).
/// Returns the RoPE table's own leading position-axis size (the module
/// doc's shape notation calls it seq_max, without backticks — not a
/// bound identifier anywhere in this crate's own source).
///
/// State table (one test per row, `tests::check_rope_pack_*`; `s`/`d` are
/// `qkv`'s own seq/head_dim):
///
/// | rank | `dims[0]` | `dims[1]` | `dims[2]` | `dims[3]` (seq_max) | `dims[4]` | outcome |
/// |---|---|---|---|---|---|---|
/// | ≠ 5 (every axis it has otherwise valid) | — | — | — | — | — | refused (rank) |
/// | 5 | ≠ 2 | — | — | — | — | refused (cos/sin axis) |
/// | 5 | 2 | ≠ 1 | — | — | — | refused (axis 1) |
/// | 5 | 2 | 1 | ≠ 1 | — | — | refused (axis 2) |
/// | 5 | 2 | 1 | 1 | — | ≠ `d` | refused (head_dim) |
/// | 5 | 2 | 1 | 1 | `< s` (`s - 1`) | `d` | refused (short table) |
/// | 5 | 2 | 1 | 1 | `== s` (boundary) | `d` | accepted |
/// | 5 | 2 | 1 | 1 | `> s` | `d` | accepted (the op reads rows `[0, s)`) |
pub(crate) fn check_rope_pack(l: &Layout, s: usize, d: usize, op: &'static str) -> Result<usize> {
    let dims = l.dims();
    if dims.len() != 5 || dims[0] != 2 || dims[1] != 1 || dims[2] != 1 || dims[4] != d {
        return Err(Error::Msg(format!(
            "{op}: rope_pack must be [2, 1, 1, seq_max, {d}], got {dims:?}"
        )));
    }
    if dims[3] < s {
        return Err(Error::Msg(format!(
            "{op}: rope_pack's seq_max={} must be >= seq={s}",
            dims[3]
        )));
    }
    Ok(dims[3])
}

/// The `lhs`-gradient half of candle's own generic `Op::Matmul` backward,
/// extracted as a named, standalone function rather than re-derived at
/// each call site: `backprop.rs`'s `Op::Matmul(lhs, rhs) => { let
/// lhs_grad = grad.matmul(&rhs.t()?)?; ... }` (candle-core 0.11.0,
/// `src/backprop.rs:461`). Any GEMM's `lhs`-gradient — this op's `bwd`
/// included — is `d(A@B)/dA` in matrix form: `grad @ Bᵀ`; this function
/// computes exactly that, with `rhs.t()` a transposed VIEW of whatever
/// `rhs` physically is (a bare view if `rhs` is a view, a view of a
/// materialized buffer if `rhs` was `.contiguous()`-ed first — the caller
/// controls which by choosing `rhs`'s own form, never this function).
/// [`matmul_grad_rhs`] is the other half of the pair. Two things that are
/// the same computation at different call sites — production's eager
/// training composition's autograd, `crate::contiguous_matmul`'s implicit
/// backward via candle's own `Op::Matmul`, and this op's hand-written
/// `bwd` — are one thing: this function IS that one definition, so `bwd`
/// and production's eager arm cannot silently drift into issuing
/// different GEMMs for the same gradient again (P3 fix round 4,
/// deliverable 3 — see `bwd`'s own doc comment on `dqs`/`dkr` for why
/// this mattered there specifically).
pub fn matmul_grad_lhs(grad: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    grad.matmul(&rhs.t()?)
}

/// The `rhs`-gradient half of the pair (`backprop.rs`: `let rhs_grad =
/// lhs.t()?.matmul(&grad)?;`, `src/backprop.rs:465`) — `d(A@B)/dB = Aᵀ @
/// grad`. See [`matmul_grad_lhs`]'s doc for the shared-definition
/// rationale.
pub fn matmul_grad_rhs(lhs: &Tensor, grad: &Tensor) -> Result<Tensor> {
    lhs.t()?.matmul(grad)
}

impl CustomOp3 for AttentionBlockFused {
    fn name(&self) -> &'static str {
        "attention_block_fused"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let op = self.name();
        let (b, s, h, d) = attention_dims(l1, op)?;
        let out_shape = Shape::from((b, s, h * d));
        // No empty fast path on this arm: `b == 0 || s == 0 || h == 0`
        // flows through the general path below (zero-trip gather loops,
        // zero-extent GEMMs — `tests::empty_{batch,seq,heads}_is_a_no_op_
        // not_a_panic`), so the domain checks below run on empty inputs
        // too. An earlier revision returned an empty buffer here first; it
        // was byte-equivalent to the general path on every empty cell
        // (verified by disabling it under all three tests) — dead weight
        // of the same class as the deleted dtype-mismatch arm below. The
        // CUDA arm keeps its own early return: cuBLAS is never handed a
        // zero-extent GEMM there.
        let (mask_b, mask_q) = check_mask(l3, b, s, op)?;
        if s1.dtype() != s3.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s3.dtype(),
                op,
            });
        }
        if self.rope && s1.dtype() != s2.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op,
            });
        }
        match (s1, s3) {
            (CpuStorage::F32(qkv), CpuStorage::F32(mask)) => {
                let (o1, o2) = l1
                    .contiguous_offsets()
                    .ok_or(Error::RequiresContiguous { op })?;
                let (m1, m2) = l3
                    .contiguous_offsets()
                    .ok_or(Error::RequiresContiguous { op })?;
                let rope_slice = if self.rope {
                    let s_max = check_rope_pack(l2, s, d, op)?;
                    match s2 {
                        CpuStorage::F32(rp) => {
                            let (r1, r2) = l2
                                .contiguous_offsets()
                                .ok_or(Error::RequiresContiguous { op })?;
                            Some((&rp[r1..r2], s_max))
                        }
                        other => return Err(Error::UnsupportedDTypeForOp(other.dtype(), op)),
                    }
                } else {
                    None
                };
                let out = attention_fwd_f32(&AttentionFwdF32Params {
                    qkv: &qkv[o1..o2],
                    rope: rope_slice,
                    mask: &mask[m1..m2],
                    mask_batch: mask_b,
                    mask_query_rows: mask_q,
                    b,
                    s,
                    h,
                    d,
                    scale: self.scale,
                    policy: self.fully_masked,
                })?;
                Ok((CpuStorage::F32(out), out_shape))
            }
            // A `qkv`/`mask` dtype MISMATCH never reaches this match — it is
            // refused by the explicit `DTypeMismatchBinaryOp` check above,
            // so the only non-`F32` pair left here is a MATCHING non-`F32`
            // pair (an earlier revision carried a second, unreachable
            // mismatch arm at this point; deleted).
            //
            // `BF16` (or any other dtype) on CPU: candle-core 0.11's CPU
            // backend has no `BF16` `MatMul` impl — the same pre-existing
            // limitation `LowRankResidualLinear`'s module doc discloses.
            // Refused here, loudly and immediately, rather than failing
            // three calls deep inside `BackendStorage::matmul`.
            (s1, _) => Err(Error::UnsupportedDTypeForOp(s1.dtype(), op)),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
        s3: &candle_core::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        crate::cuda::attention_block::cuda_fwd(self, s1, l1, s2, l2, s3, l3)
    }

    /// See the module doc's "`bwd`: ordinary `Tensor` composition" section.
    /// Delegates to `bwd_core` (private, this module), which this op's `dqkv` output and
    /// [`bwd_gradient_gemm_layouts`]'s test-side layout introspection both
    /// call — the SAME code path, not two independently maintained copies
    /// of it (P3 fix round 4: a fixture reconstructed separately from
    /// `bwd`'s own logic cannot regress when `bwd` does — see
    /// `bwd_gradient_gemm_layouts`'s own doc).
    fn bwd(
        &self,
        qkv: &Tensor,
        rope_pack: &Tensor,
        mask: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let op = self.name();
        if rope_pack.track_op() || mask.track_op() {
            return Err(Error::Msg(format!(
                "{op}: this op computes no gradient for the RoPE table or the mask — asserted \
                 here rather than silently returning None (family D): rope_pack/mask must never \
                 be tracked (never a Var, never downstream of one)"
            )));
        }
        let (dqkv, _gemm_operands) = bwd_core(BwdCoreParams {
            op,
            rope: self.rope,
            scale: self.scale,
            fully_masked: self.fully_masked,
            qkv,
            rope_pack,
            mask,
            grad_res,
        })?;
        Ok((Some(dqkv), None, None))
    }
}

/// `bwd_core`'s inputs, bundled into one struct rather than passed
/// positionally — the SAME hazard `AttentionFwdF32Params` exists to
/// remove for `attention_fwd_f32` (see that struct's own doc): four
/// ADJACENT `&Tensor` parameters (`qkv`, `rope_pack`, `mask`, `grad_res`)
/// are silently swappable at a call site with no compiler help, and
/// `bwd_core` has exactly that shape. Named fields make a transposition
/// a compile error instead of a silent wrong-tensor bug (P3 fix round 4
/// closing round: the file's own precedent, reintroduced by this round's
/// refactor, now closed the same way as the forward path).
struct BwdCoreParams<'a> {
    op: &'static str,
    rope: bool,
    scale: f32,
    fully_masked: FullyMaskedPolicy,
    qkv: &'a Tensor,
    rope_pack: &'a Tensor,
    mask: &'a Tensor,
    grad_res: &'a Tensor,
}

/// `bwd`'s real computation, factored out of the `CustomOp3::bwd` trait
/// method so it has exactly one caller-visible definition: the trait
/// method above (production) and [`bwd_gradient_gemm_layouts`] (this
/// file's own layout-identity test oracle, `tests/cuda_parity.rs`) both
/// call THIS function, never a re-derived copy of its logic. Returns
/// `dqkv` plus the four `(lhs, rhs)` operand pairs actually issued to
/// `.matmul()` for `dv`/`dp`/`dqs`/`dkr` respectively — cheap to capture
/// (a `Tensor` clone is a handle/refcount bump, not a data copy) and
/// otherwise unused in production, where the trait method above discards
/// them.
fn bwd_core(params: BwdCoreParams<'_>) -> Result<(Tensor, [(Tensor, Tensor); 4])> {
    let BwdCoreParams {
        op,
        rope,
        scale,
        fully_masked,
        qkv,
        rope_pack,
        mask,
        grad_res,
    } = params;
    // DETACH every input before composing anything (module doc's
    // "`bwd` runs DETACHED" section): `qkv` is a tracked graph node in
    // production (downstream of the `Wqkv` LoRA `Var`s), so without
    // this every `Tensor` built below would carry a `BackpropOp`
    // cloning its inputs (candle-core 0.11.0 `op.rs`'s
    // `BackpropOp::new1/new2/new3` — `if arg.track_op() {
    // Some(f(arg.clone())) }`), the explicit `drop`s below would free
    // NOTHING (the consumer's `Op` still owns a clone), and the whole
    // chain would be handed back to the engine inside `dqkv`'s own
    // `Op`. candle's engine detaches the incoming `grad` for exactly
    // this reason (`backprop.rs`'s `grad.detach()` — "to avoid
    // computing the backprop graph of the backprop itself"); this op
    // does the same for its OTHER three inputs. `detach` shares
    // storage (no copy) and is a no-op clone on an already-untracked
    // tensor (`tensor.rs`'s `Tensor::detach`).
    let qkv = qkv.detach();
    let rope_pack = rope_pack.detach();
    let mask = mask.detach();
    let grad_res = grad_res.detach();
    let (b, s, three, h, d) = qkv.dims5()?;
    if three != 3 {
        return Err(Error::Msg(format!(
            "{op}: qkv must be rank 5 [batch, seq, 3, heads, head_dim], got 3-axis size {three}"
        )));
    }

    let q0 = qkv.narrow(2, 0, 1)?.squeeze(2)?.transpose(1, 2)?;
    let k0 = qkv.narrow(2, 1, 1)?.squeeze(2)?.transpose(1, 2)?;
    let v0 = qkv.narrow(2, 2, 1)?.squeeze(2)?.transpose(1, 2)?;

    let (q_rot, k_rot, cos_sin) = if rope {
        let cos_full = rope_pack.narrow(0, 0, 1)?.squeeze(0)?;
        let sin_full = rope_pack.narrow(0, 1, 1)?.squeeze(0)?;
        let cos = cos_full.narrow(2, 0, s)?;
        let sin = sin_full.narrow(2, 0, s)?;
        let qr = apply3(&q0.contiguous()?, &cos, &sin, RopeFused::new(false))?;
        let kr = apply3(&k0.contiguous()?, &cos, &sin, RopeFused::new(false))?;
        (qr, kr, Some((cos, sin)))
    } else {
        (q0.contiguous()?, k0.contiguous()?, None)
    };

    // Materialized ONCE here (not just at the `scores` call site) so
    // the SAME contiguous buffer feeds both `scores`'s recompute below
    // and `dkr`'s gradient GEMM further down — see that GEMM's own
    // comment for why it needs a materialized `q_scaled`, not a view.
    let q_scaled = (&q_rot * f64::from(scale))?.contiguous()?;
    let v_c = v0.contiguous()?;

    // `mask` is ALREADY the caller's combined padding-plus-band sum
    // (module doc's "window is construction data at the call site"
    // section) — no band to rebuild here, unlike earlier revisions of
    // this op.
    //
    // `scores` uses the SAME transposed-VIEW `k_rot` GEMM shape `fwd`
    // issues (`.transpose(...)` with NO trailing `.contiguous()` — see
    // the module doc's "fwd/bwd GEMM shape match" section). This is a
    // RECOMPUTE of `fwd`'s own GEMM, matched to `fwd`, not to
    // production's `contiguous_matmul` — see the module doc's
    // "two-armed rule" section for why `bwd` deliberately keeps two
    // different operand-form targets for two different GEMMs.
    let scores = q_scaled.matmul(&k_rot.transpose(D::Minus1, D::Minus2)?)?;
    let p = apply2(&scores, &mask, SoftmaxLastDimFused::new(fully_masked))?;
    // `scores`'s only use was building `p` above (module doc's
    // "transient scoping" section) — drop it now instead of letting it
    // live to the end of `bwd`.
    drop(scores);

    let dctx = grad_res
        .reshape((b, s, h, d))?
        .transpose(1, 2)?
        .contiguous()?;

    // `dv`/`dp` differentiate `ctx = p · v_c` (`fwd`'s SECOND GEMM,
    // never itself recomputed — see the table above) through
    // [`matmul_grad_rhs`]/[`matmul_grad_lhs`], candle's own generic
    // `Op::Matmul` backward formula. `p` and `v_c` are ALREADY
    // naturally contiguous (fresh op outputs), the exact form
    // `crate::contiguous_matmul`'s `a.contiguous()?.matmul(&b.
    // contiguous()?)` would leave them in on the production eager
    // path too — the shared definition needs no extra materialize
    // here to match it.
    let dv_operands = (p.transpose(D::Minus1, D::Minus2)?, dctx.clone());
    let dv = matmul_grad_rhs(&p, &dctx)?;
    let dp_operands = (dctx.clone(), v_c.transpose(D::Minus1, D::Minus2)?);
    let dp = matmul_grad_lhs(&dctx, &v_c)?;
    let ds = apply2(&p, &dp, SoftmaxBwdDScores)?;
    // `p`'s last use (its second, after `dv` above) and `dp`'s only
    // use were both building `ds` — drop both now.
    drop(p);
    drop(dp);

    // `dqs`/`dkr` differentiate `scores = q_scaled · k_rotᵀ` — but
    // PRODUCTION's forward for THIS GEMM is
    // `jammi_encoders::contiguous_matmul` (`crates/jammi-encoders/
    // src/lib.rs:139-141`, called from `forward_eager_training_
    // attention_composition` at `modernbert.rs:1016`), which
    // materializes BOTH operands, NOT the transposed-VIEW `k_rotᵀ`
    // `scores`'s recompute above uses. Matching `contiguous_matmul`'s
    // materialization here (deliberately NOT `fwd`'s own form, unlike
    // `scores` above) is the fix: differentiating a materialized
    // `kt_contig` through [`matmul_grad_lhs`]/[`matmul_grad_rhs`]
    // issues the IDENTICAL GEMM (rows/cols/row_stride/col_stride)
    // production's own autograd issues for `dQ`. `dK` needs one more
    // step to get there — candle differentiates `kt_contig =
    // k_rotᵀ.contiguous()` itself through `Op::Copy` (identity —
    // candle-core 0.11.0 `backprop.rs:525-527`) then `Op::Transpose`
    // (`grad.transpose(dim1, dim2)` — `backprop.rs:710-713`), which is
    // exactly the trailing `.transpose()` below. Candle's OWN engine
    // would leave THAT particular transpose's output a view too (`Op::
    // Transpose`'s backward, `backprop.rs:710-713`, is a bare
    // `grad.transpose(...)`), because the engine's `GradStore::or_insert`
    // (`backprop.rs:768-777`) materializes it downstream via `zeros_like`
    // + `.add()` when `dK`'s OWN gradient is later accumulated — a step
    // `bwd_core` has no equivalent of (it composes ordinary `Tensor` ops,
    // not the engine's accumulation machinery). The trailing
    // `.contiguous()` below reproduces that materialization directly,
    // and doubles as satisfying [`super::RopeFused`]'s own admission
    // contract (contiguous input only) when `rope` is `true`.
    // (P3 fix round 4, deliverable 3: an earlier revision of `bwd`
    // computed `dqs`/`dkr` from `k_rot` directly — `fwd`'s operand form,
    // not production's — issuing a DIFFERENT cuBLAS call from
    // production's eager arm for this GEMM specifically, even after the
    // round-3 fix aligned `dv`/`dp`. Measured on the pod (A100, `b=8,
    // s=512, h=16, d=64`, bf16): reverting ONLY this — `dqs`/`dkr` back
    // to the pre-round-4 form, `dv`/`dp` left at this round's shared
    // definition — does NOT redden `tests::attention_block_fused_vs_
    // eager_dqkv_divergence_grows_with_depth_bf16_cuda`
    // (`jammi-encoders`): `r(1)` stays exactly `0.0` for every slot, IDENTICAL
    // to the fully-fixed build, at this shape. Restoring the ORIGINAL
    // esc-044 defect too (`dv`/`dp`/`dkr`'s three `.contiguous()` calls)
    // reddens it regardless of `dqs`/`dkr`'s form. This round's
    // `kt_contig`/shared-definition change for `dqs`/`dkr` is therefore
    // justified STRUCTURALLY here — it provably issues the identical GEMM
    // production's own autograd would (checked by `tests::attention_
    // block_bwd_dqs_dkr_gemm_layouts_match_production_orientation_cuda`)
    // — not as a numerically-demonstrated-necessary fix at this shape;
    // whether it matters at another shape or on another cuBLAS/driver
    // version is unconfirmed, reported honestly rather than claimed.)
    let kt_contig = k_rot.transpose(D::Minus1, D::Minus2)?.contiguous()?;
    let dqs_operands = (ds.clone(), kt_contig.transpose(D::Minus1, D::Minus2)?);
    let dqs = matmul_grad_lhs(&ds, &kt_contig)?;
    let dkr_operands = (q_scaled.transpose(D::Minus1, D::Minus2)?, ds.clone());
    let dkr = matmul_grad_rhs(&q_scaled, &ds)?
        .transpose(D::Minus1, D::Minus2)?
        .contiguous()?;
    // `ds`'s last use (its second, after `dqs`) was building `dkr`,
    // and `kt_contig` was single-use — drop both; no `[B,H,S,S]`- or
    // `[B,H,D,S]`-shaped tensor remains live for the rest of `bwd`.
    drop(ds);
    drop(kt_contig);
    let dqr = (&dqs * f64::from(scale))?;

    let (dq0, dk0) = if let Some((cos, sin)) = cos_sin {
        let dq0 = apply3(&dqr, &cos, &sin, RopeFused::new(true))?;
        let dk0 = apply3(&dkr, &cos, &sin, RopeFused::new(true))?;
        (dq0, dk0)
    } else {
        (dqr, dkr)
    };

    let to_qkv_slot = |t: &Tensor| -> Result<Tensor> {
        t.transpose(1, 2)?.contiguous()?.reshape((b, s, 1, h, d))
    };
    let dqkv = Tensor::cat(
        &[&to_qkv_slot(&dq0)?, &to_qkv_slot(&dk0)?, &to_qkv_slot(&dv)?],
        2,
    )?;

    Ok((dqkv, [dv_operands, dp_operands, dqs_operands, dkr_operands]))
}

/// [`bwd_gradient_gemm_layouts`]'s inputs, bundled for the SAME reason
/// [`BwdCoreParams`] exists (see that struct's own doc) — this one is
/// `pub` because the function it feeds is a cross-crate test entry point
/// (`tests/cuda_parity.rs`), so the transposition hazard is a real
/// cross-crate call-site risk, not just an in-module one.
#[doc(hidden)]
pub struct BwdGemmLayoutsParams<'a> {
    pub rope: bool,
    pub scale: f32,
    pub fully_masked: FullyMaskedPolicy,
    pub qkv: &'a Tensor,
    pub rope_pack: &'a Tensor,
    pub mask: &'a Tensor,
    pub grad_res: &'a Tensor,
}

/// Test/introspection support: captures the four gradient-GEMM operand
/// `Layout`s `bwd_core` (private, this module) ACTUALLY builds, never a
/// fixture reconstructed independently of it. Order matches the module
/// doc's GEMM table: `dv`, `dp`, `dqs`, `dkr`. Used by
/// `tests/cuda_parity.rs`'s `attention_block_bwd_dqs_dkr_gemm_layouts_
/// match_production_orientation_cuda` (P3 fix round 4, deliverable 3's
/// "mechanism pin" — see that test's own doc).
#[doc(hidden)]
pub fn bwd_gradient_gemm_layouts(
    params: BwdGemmLayoutsParams<'_>,
) -> Result<[(Layout, Layout); 4]> {
    let (_dqkv, operands) = bwd_core(BwdCoreParams {
        op: "bwd_gradient_gemm_layouts",
        rope: params.rope,
        scale: params.scale,
        fully_masked: params.fully_masked,
        qkv: params.qkv,
        rope_pack: params.rope_pack,
        mask: params.mask,
        grad_res: params.grad_res,
    })?;
    Ok(operands.map(|(a, b)| (a.layout().clone(), b.layout().clone())))
}

/// [`attention_fwd_f32`]'s inputs, already validated by `cpu_fwd`
/// (`attention_dims`/`check_mask`/`check_rope_pack` plus the contiguity
/// and dtype checks): every slice is the exact contiguous range of its
/// storage, every extent is `qkv`'s own, and `mask_batch`/`mask_query_rows`
/// are `check_mask`'s two returned axis sizes (`1` or `b`, `1` or `s`).
#[derive(Clone, Copy)]
struct AttentionFwdF32Params<'a> {
    /// `[b, s, 3, h, d]`, contiguous.
    qkv: &'a [f32],
    /// `(cos-then-sin table, seq_max)` — the flat `[2, 1, 1, seq_max, d]`
    /// pack — or `None` when the op was built with `rope == false`.
    rope: Option<(&'a [f32], usize)>,
    /// `[mask_batch, 1, mask_query_rows, s]`, contiguous.
    mask: &'a [f32],
    mask_batch: usize,
    mask_query_rows: usize,
    b: usize,
    s: usize,
    h: usize,
    d: usize,
    /// Folded into `Q` before `QKᵀ` (module doc's "Fixed domain").
    scale: f32,
    policy: FullyMaskedPolicy,
}

/// The composed CPU forward: gather `Q`/`K`/`V` out of `qkv` into
/// `[batch*heads, seq, head_dim]` contiguous buffers (fixed ascending
/// `(batch, seq, heads)` gather order — family J), RoPE-rotate `Q`/`K`
/// (reusing [`rope_fwd_row_f32`] directly — bit-exact to
/// [`super::RopeFused`]'s own CPU math), fold `scale` into `Q`, batched
/// `QKᵀ` via [`BackendStorage::matmul`] (the SAME call
/// `candle_core::Tensor::matmul` issues), per-row mask-add-then-softmax
/// (reusing [`softmax_row_f32`] directly — bit-exact to
/// [`super::SoftmaxLastDimFused`]'s own CPU math), batched `PV`, then
/// scatter back to `[batch, seq, heads*head_dim]`. `mask` here is ALREADY
/// the caller's combined padding-plus-band sum (see the module doc's
/// "window is construction data at the call site" section) — this
/// function computes no band predicate of its own, unlike earlier
/// revisions.
///
/// Inputs arrive as ONE [`AttentionFwdF32Params`] (named fields at the
/// call site) rather than eleven positional arguments — the shape that
/// let an earlier revision's `mask_batch`/`mask_query_rows` pair be
/// silently swappable and needed a `clippy::too_many_arguments` allow.
fn attention_fwd_f32(params: &AttentionFwdF32Params<'_>) -> Result<Vec<f32>> {
    let AttentionFwdF32Params {
        qkv,
        rope,
        mask,
        mask_batch,
        mask_query_rows,
        b,
        s,
        h,
        d,
        scale,
        policy,
    } = *params;
    let bh = b * h;
    let sd = s * d;
    let mut q = vec![0f32; bh * sd];
    let mut k = vec![0f32; bh * sd];
    let mut v = vec![0f32; bh * sd];
    for bi in 0..b {
        for si in 0..s {
            let base = (bi * s + si) * 3 * h * d;
            for hi in 0..h {
                let q_src = base + hi * d;
                let k_src = base + h * d + hi * d;
                let v_src = base + 2 * h * d + hi * d;
                let dst = (bi * h + hi) * sd + si * d;
                q[dst..dst + d].copy_from_slice(&qkv[q_src..q_src + d]);
                k[dst..dst + d].copy_from_slice(&qkv[k_src..k_src + d]);
                v[dst..dst + d].copy_from_slice(&qkv[v_src..v_src + d]);
            }
        }
    }

    if let Some((table, s_max)) = rope {
        let cos = &table[0..s_max * d];
        let sin = &table[s_max * d..2 * s_max * d];
        let mut qr = vec![0f32; bh * sd];
        let mut kr = vec![0f32; bh * sd];
        for bh_i in 0..bh {
            for si in 0..s {
                let off = bh_i * sd + si * d;
                let cos_row = &cos[si * d..(si + 1) * d];
                let sin_row = &sin[si * d..(si + 1) * d];
                rope_fwd_row_f32(
                    &q[off..off + d],
                    cos_row,
                    sin_row,
                    1.0,
                    &mut qr[off..off + d],
                );
                rope_fwd_row_f32(
                    &k[off..off + d],
                    cos_row,
                    sin_row,
                    1.0,
                    &mut kr[off..off + d],
                );
            }
        }
        q = qr;
        k = kr;
    }

    for qv in q.iter_mut() {
        *qv *= scale;
    }

    let q_layout = Layout::contiguous((bh, s, d));
    let k_layout = Layout::contiguous((bh, s, d));
    let k_t_layout = k_layout.transpose(1, 2)?;
    // PIN the convention: this is a transpose VIEW (cuBLAS/
    // `gemm`'s OP_T), never a materialized contiguous copy — the module
    // doc's "fwd/bwd GEMM shape match" and "GEMM operand form is a
    // determinism concern" sections both depend on `fwd` and `bwd` issuing
    // this EXACT operand form, and `eager_reference`'s own `k_t` VIEW
    // (this module's `#[cfg(test)]` section) is now held to the same
    // shape by construction — this assertion is what would catch either
    // side silently regressing back to `.contiguous()`.
    debug_assert!(
        d <= 1 || !k_t_layout.is_contiguous(),
        "attention_block_fused: k_t_layout must be a transpose VIEW, not a materialized copy \
         (d>1 makes a genuine transpose of a [bh,s,d] contiguous tensor itself non-contiguous)"
    );
    let scores_storage =
        CpuStorage::F32(q).matmul(&CpuStorage::F32(k), (bh, s, s, d), &q_layout, &k_t_layout)?;
    let CpuStorage::F32(scores) = scores_storage else {
        return Err(Error::Msg(
            "attention_block_fused: internal matmul returned a non-F32 storage for an F32 input"
                .into(),
        ));
    };

    let mut p = vec![0f32; bh * s * s];
    // `mask`'s own flat layout is `[mask_batch, 1, mask_query_rows, s]`
    // row-major (the SAME leading-axis broadcast class
    // `super::softmax::softmax_dims` validates for `SoftmaxLastDimFused`
    // — see `check_mask`'s doc): a batch element's own row block starts at
    // `bi * mask_query_rows * s` (`0` when `mask_batch == 1`, broadcasting
    // over every batch element — this is the exact indexing audit item B2
    // named: an `mrow_base` that stayed hardcoded to `0` regardless of
    // `mask_batch` would silently broadcast batch element 0's mask onto
    // every OTHER batch element too, a bug this function's own oracle
    // (`cpu_fwd_per_batch_mask_row_indexing_is_not_hardcoded_to_zero`)
    // exercises directly), and a query row's own row within that block
    // starts at `qi * s` (`0` when `mask_query_rows == 1`, broadcasting
    // over every query position — the padding-only global-layer shape).
    for bh_i in 0..bh {
        let bi = bh_i / h;
        let mrow_base = if mask_batch == 1 {
            0
        } else {
            bi * mask_query_rows * s
        };
        for qi in 0..s {
            let qrow_base = if mask_query_rows == 1 { 0 } else { qi * s };
            let mrow = &mask[mrow_base + qrow_base..mrow_base + qrow_base + s];
            let srow = &scores[(bh_i * s + qi) * s..(bh_i * s + qi + 1) * s];
            let prow = &mut p[(bh_i * s + qi) * s..(bh_i * s + qi + 1) * s];
            // scale is already folded into `q` above (the same fold the
            // caller's rounding contract requires — see this function's
            // doc); softmax's own `scale` here is exactly 1.0 so it applies
            // no second scaling, matching the module doc's "fold 1/√d into
            // Q, pass scale=1.0 to softmax" resolution (exact power of two,
            // bit-exact either way it is applied).
            softmax_row_f32(srow, mrow, prow, policy, 1.0);
        }
    }

    let p_layout = Layout::contiguous((bh, s, s));
    let v_layout = Layout::contiguous((bh, s, d));
    let ctx_storage =
        CpuStorage::F32(p).matmul(&CpuStorage::F32(v), (bh, s, d, s), &p_layout, &v_layout)?;
    let CpuStorage::F32(ctx) = ctx_storage else {
        return Err(Error::Msg(
            "attention_block_fused: internal matmul returned a non-F32 storage for an F32 input"
                .into(),
        ));
    };

    let mut out = vec![0f32; b * s * h * d];
    for bi in 0..b {
        for hi in 0..h {
            for si in 0..s {
                let src = ((bi * h + hi) * s + si) * d;
                let dst = (bi * s + si) * h * d + hi * d;
                out[dst..dst + d].copy_from_slice(&ctx[src..src + d]);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Var};

    /// Mirrors `LowRankResidualLinear`'s own `is_gemm_operand_admissible`
    /// (candle-core 0.11.0's `cuda_backend::gemm_config`,
    /// `cuda_backend/mod.rs:1398-1422`), extended to the TRAILING two axes
    /// of a possibly-batched operand: admissible iff row-major contiguous
    /// (cuBLAS's N mode) OR a single transposed view of a row-major
    /// contiguous matrix (cuBLAS's T mode) on those last two axes —
    /// device-independent (the same `Layout` any backend's `matmul`
    /// receives). (cuBLAS's own CUBLAS_OP_N/CUBLAS_OP_T enum names are an
    /// external C API this repo's Rust source never defines — cited by mode
    /// letter above instead of backtick-quoting them, so this
    /// citation-resolution check does not need an allowlist entry for a
    /// vendored constant it has no way to see.)
    fn is_gemm_operand_admissible(l: &Layout) -> bool {
        let dims = l.dims();
        let stride = l.stride();
        if dims.len() < 2 {
            return false;
        }
        let r = dims.len();
        let (p, q) = (dims[r - 2], dims[r - 1]);
        let (sp, sq) = (stride[r - 2], stride[r - 1]);
        (sq == 1 && sp == q) || (sp == 1 && sq == p)
    }

    /// Every operand `bwd` hands `Tensor::matmul`, reconstructed via the
    /// EXACT same shape/transpose sequence `bwd` builds (module doc's
    /// "`bwd`: ordinary `Tensor` composition" section), at a boundary rank
    /// (`heads=1`) and a production-scale rank (ModernBERT-large's own
    /// `heads=16`) — proves `bwd`'s `.contiguous()` placement leaves no
    /// operand a raw doubly-strided view `gemm_config` would refuse.
    ///
    /// DEMOTED from "the" regression oracle for the round-4 GEMM-
    /// operand-FORM defect (P3 fix round 4, deliverable 3): this test
    /// answers ADMISSIBILITY ("would cuBLAS accept this operand at all"),
    /// which the pre-round-4 `dqs`/`dkr` forms ALSO satisfied — a wrong
    /// but ADMISSIBLE GEMM is exactly the failure mode that shipped.
    /// Because this test reconstructs its own operands from a hardcoded
    /// shape/transpose sequence rather than reading them FROM `bwd_core`,
    /// it stays green under a `bwd_core` regression by construction.
    /// `tests::attention_block_bwd_dqs_dkr_gemm_layouts_match_production_
    /// orientation_cuda` (`tests/cuda_parity.rs`, via
    /// [`bwd_gradient_gemm_layouts`]) is the oracle that actually reads
    /// `bwd_core`'s own operands.
    #[test]
    fn bwd_every_gemm_operand_is_admissible_at_boundary_and_production_ranks() {
        let device = Device::Cpu;
        for &(b, s, h, d) in &[
            (2usize, 3usize, 1usize, 4usize),
            (2usize, 8usize, 16usize, 4usize),
        ] {
            let q = Tensor::randn(0f32, 1.0, (b, h, s, d), &device).unwrap();
            let k = Tensor::randn(0f32, 1.0, (b, h, s, d), &device).unwrap();
            let v = Tensor::randn(0f32, 1.0, (b, h, s, d), &device).unwrap();
            let p = Tensor::randn(0f32, 1.0, (b, h, s, s), &device).unwrap();
            let dctx = Tensor::randn(0f32, 1.0, (b, h, s, d), &device).unwrap();
            let ds = Tensor::randn(0f32, 1.0, (b, h, s, s), &device).unwrap();

            // scores = q_scaled.contiguous() @ k_rot.transpose(-1,-2) — NO
            // `.contiguous()` on the rhs (see the module doc's "fwd/bwd
            // GEMM shape match" section): this must be the SAME
            // transpose-VIEW operand `fwd` issues, not a materialized
            // contiguous copy.
            let lhs = q.contiguous().unwrap();
            let rhs = k.transpose(D::Minus1, D::Minus2).unwrap();
            assert!(
                is_gemm_operand_admissible(lhs.layout()),
                "scores lhs @ ({b},{s},{h},{d})"
            );
            assert!(
                is_gemm_operand_admissible(rhs.layout()),
                "scores rhs @ ({b},{s},{h},{d})"
            );
            // The mutation this guards against: reintroducing
            // `.contiguous()` on `rhs` (as an earlier revision of `bwd`
            // did) would make THIS assertion pass too (a materialized
            // contiguous tensor is still an admissible GEMM operand — the
            // shape mismatch is invisible to `is_gemm_operand_admissible`
            // alone) but flip `rhs.is_contiguous()` from `false` to `true`,
            // which is exactly what this assertion catches: `s > 1` here
            // (`s=3`/`s=8`) makes a genuine transpose of a contiguous
            // `[b,h,s,d]` tensor itself non-contiguous whenever `d > 1`
            // (verified: both fixtures use `d=4`).
            assert!(
                !rhs.is_contiguous(),
                "scores rhs @ ({b},{s},{h},{d}) must be a transpose VIEW (cuBLAS OP_T), not a \
                 materialized contiguous copy (OP_N) — bwd would then issue a DIFFERENT GEMM \
                 shape than fwd's own QKᵀ"
            );
            let _ = lhs.matmul(&rhs).unwrap();

            // dv = p.transpose(-1,-2) (VIEW, no `.contiguous()`) @ dctx —
            // see the module doc's "GEMM operand form is a determinism
            // concern" section: this is the SAME view-vs-materialize
            // distinction as `scores` above, also applying to `dv`/`dp`/
            // `dkr`.
            let lhs = p.transpose(D::Minus1, D::Minus2).unwrap();
            assert!(
                is_gemm_operand_admissible(lhs.layout()),
                "dv lhs @ ({b},{s},{h},{d})"
            );
            assert!(
                is_gemm_operand_admissible(dctx.layout()),
                "dv rhs @ ({b},{s},{h},{d})"
            );
            assert!(
                !lhs.is_contiguous(),
                "dv lhs @ ({b},{s},{h},{d}) must be a transpose VIEW of `p`, not a materialized \
                 copy — a materialized copy is a DIFFERENT gemm operand form from what candle's \
                 own generic `Op::Matmul` backward issues for the eager composition's equivalent \
                 gradient (backprop.rs: `rhs_grad = lhs.t().matmul(&grad)`, always a view)"
            );
            let _ = lhs.matmul(&dctx).unwrap();

            // dp = dctx @ v.transpose(-1,-2) (VIEW, no `.contiguous()`)
            let rhs = v.transpose(D::Minus1, D::Minus2).unwrap();
            assert!(
                is_gemm_operand_admissible(dctx.layout()),
                "dp lhs @ ({b},{s},{h},{d})"
            );
            assert!(
                is_gemm_operand_admissible(rhs.layout()),
                "dp rhs @ ({b},{s},{h},{d})"
            );
            assert!(
                !rhs.is_contiguous(),
                "dp rhs @ ({b},{s},{h},{d}) must be a transpose VIEW of `v`, not a materialized \
                 copy — same determinism concern as `dv`'s `lhs` above"
            );
            let _ = dctx.matmul(&rhs).unwrap();

            // dqs = ds @ k — both operands ALREADY contiguous (`ds` a
            // fresh `CustomOp2` output, `k` this fixture's own contiguous
            // tensor), no `.contiguous()` needed or issued.
            assert!(
                is_gemm_operand_admissible(ds.layout()),
                "dqs lhs @ ({b},{s},{h},{d})"
            );
            assert!(
                is_gemm_operand_admissible(k.layout()),
                "dqs rhs @ ({b},{s},{h},{d})"
            );
            let _ = ds.matmul(&k).unwrap();

            // dkr = ds.transpose(-1,-2) (VIEW, no `.contiguous()`) @ q
            let lhs = ds.transpose(D::Minus1, D::Minus2).unwrap();
            assert!(
                is_gemm_operand_admissible(lhs.layout()),
                "dkr lhs @ ({b},{s},{h},{d})"
            );
            assert!(
                is_gemm_operand_admissible(q.layout()),
                "dkr rhs @ ({b},{s},{h},{d})"
            );
            assert!(
                !lhs.is_contiguous(),
                "dkr lhs @ ({b},{s},{h},{d}) must be a transpose VIEW of `ds`, not a materialized \
                 copy — same determinism concern as `dv`'s `lhs` above"
            );
            let _ = lhs.matmul(&q).unwrap();
        }
    }

    fn fused(
        qkv: &Tensor,
        rope_pack: &Tensor,
        mask: &Tensor,
        op: AttentionBlockFused,
    ) -> Result<Tensor> {
        qkv.apply_op3(rope_pack, mask, op)
    }

    /// TEST-ONLY: builds the `[1, 1, seq, seq]` sliding-window band a
    /// local-attention CALLER combines with its padding mask before
    /// invoking [`AttentionBlockFused`] — this op itself has no `window`
    /// construction data and computes no band predicate of its own (see
    /// the module doc's "window is construction data at the call site"
    /// section); this is the test suite's own stand-in for what
    /// `jammi_encoders::mask::sliding_window_mask` does at the real call
    /// site, kept here (not imported — family L: this crate names no
    /// consumer) purely to build EXPECTED values.
    fn test_sliding_window_band(
        s: usize,
        half_window: usize,
        dtype: candle_core::DType,
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let mut band = vec![0f32; s * s];
        for qi in 0..s {
            for ki in 0..s {
                if qi.abs_diff(ki) > half_window {
                    band[qi * s + ki] = WINDOW_MASKED_VALUE;
                }
            }
        }
        Tensor::from_vec(band, (1, 1, s, s), device)?.to_dtype(dtype)
    }

    /// A small deterministic eager reference, built from the SAME
    /// conceptual steps this op's forward composes (RoPE, scale-fold,
    /// `QKᵀ`, mask-add, softmax, `PV`), via ordinary `Tensor` ops —
    /// EXACTLY the shape `ops::softmax::tests::eager`/`ops::rope::tests`
    /// use as their own comparison targets. Assembled here rather than
    /// imported from `jammi-encoders` (family L: this crate names no
    /// consumer). `mask` here is the CALLER's already-combined mask (the
    /// SAME value [`fused`]'s own `mask` argument gets) — this function
    /// does no band-building itself; callers that want a window arm build
    /// one via [`test_sliding_window_band`] and combine it in first,
    /// mirroring [`AttentionBlockFused`]'s own real call site.
    #[allow(clippy::too_many_arguments)]
    fn eager_reference(
        q0: &Tensor,
        k0: &Tensor,
        v0: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        mask: &Tensor,
        scale: f32,
        policy: FullyMaskedPolicy,
    ) -> Result<Tensor> {
        let (b, h, s, d) = q0.dims4()?;
        let (q, k) = match (cos, sin) {
            (Some(cos), Some(sin)) => (
                apply3(q0, cos, sin, RopeFused::new(false))?,
                apply3(k0, cos, sin, RopeFused::new(false))?,
            ),
            _ => (q0.clone(), k0.clone()),
        };
        // `k`'s transpose is passed as a VIEW (no `.contiguous()`) — the
        // SAME operand form `AttentionBlockFused`'s own `cpu_fwd`/`cuda_fwd`
        // use for this GEMM (an earlier revision of this reference used to
        // MATERIALIZE the transpose, which put a DIFFERENT operand form
        // into `gemm`'s
        // packing/blocking decision on x86_64/AVX than the op under test —
        // `is_gemm_operand_admissible` accepts both forms, but only the
        // VIEW form is what `fwd` itself issues, so only it is a genuine
        // "does this op compute the right VALUE" oracle rather than an
        // accidental "do two DIFFERENT gemm calls happen to agree" one).
        let k_t = k.transpose(D::Minus1, D::Minus2)?;
        // PIN the convention this reference now shares with `fwd`: a
        // transpose VIEW's `(row_stride, col_stride)` on its trailing two
        // axes is `(1, d)` for a `[.., s, d]`-contiguous `k` (cuBLAS reads
        // `rhs_cs=d, rhs_rs=1` in `attention_fwd_f32`'s own `k_t_layout` —
        // see that function's doc). Asserting the SHAPE here (rather than
        // re-deriving `attention_fwd_f32`'s private `Layout` to compare
        // byte-for-byte) still catches the relevant class of regression:
        // reintroducing `.contiguous()` here would flip `k_t` from
        // a `(d, s, d, 1)`-strided view to a `(d, s, s, 1)`-strided fresh
        // buffer — i.e. exactly the `!is_contiguous()` flip the module's
        // own `bwd_every_gemm_operand_is_admissible_at_boundary_and_production_ranks`
        // test already exercises for `bwd`'s own transposed operands.
        debug_assert!(
            !k_t.is_contiguous(),
            "eager_reference's k transpose must be a VIEW (matching fwd's own k_t_layout), not \
             a materialized copy"
        );
        let scores = (q.contiguous()?.matmul(&k_t)? * f64::from(scale))?;
        let p = apply2(&scores, mask, SoftmaxLastDimFused::new(policy))?;
        let ctx = p.matmul(&v0.contiguous()?)?;
        ctx.transpose(1, 2)?.contiguous()?.reshape((b, s, h * d))
    }

    fn pack_rope(cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        Tensor::stack(&[cos, sin], 0)
    }

    fn qkv_from(q0: &Tensor, k0: &Tensor, v0: &Tensor) -> Result<Tensor> {
        // q0/k0/v0: [B,H,S,D] -> qkv: [B,S,3,H,D].
        let stacked = Tensor::stack(&[q0, k0, v0], 2)?; // [B,H,3,S,D]
        stacked.permute((0, 3, 2, 1, 4))?.contiguous()
    }

    fn rope_tables(s_max: usize, d: usize, device: &Device) -> (Tensor, Tensor) {
        let half = d / 2;
        let mut cos_v = Vec::with_capacity(s_max * d);
        let mut sin_v = Vec::with_capacity(s_max * d);
        for pos in 0..s_max {
            for _ in 0..2 {
                for i in 0..half {
                    let theta = (pos as f64) * (10_000f64.powf(-2.0 * i as f64 / d as f64));
                    cos_v.push(theta.cos() as f32);
                    sin_v.push(theta.sin() as f32);
                }
            }
        }
        let cos = Tensor::from_vec(cos_v, (1, 1, s_max, d), device).unwrap();
        let sin = Tensor::from_vec(sin_v, (1, 1, s_max, d), device).unwrap();
        (cos, sin)
    }

    fn zero_mask(b: usize, s: usize, device: &Device) -> Tensor {
        Tensor::from_vec(vec![0f32; b * s], (b, 1, 1, s), device).unwrap()
    }

    #[test]
    fn cpu_fwd_bit_exact_vs_eager_reference_global_no_rope() {
        let device = Device::Cpu;
        let (b, h, s, d) = (2usize, 2usize, 5usize, HEAD_DIM);
        let n = b * h * s * d;
        let q0v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.13).sin()).collect();
        let k0v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.19).cos()).collect();
        let v0v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.29).sin()).collect();
        let q0 = Tensor::from_vec(q0v, (b, h, s, d), &device).unwrap();
        let k0 = Tensor::from_vec(k0v, (b, h, s, d), &device).unwrap();
        let v0 = Tensor::from_vec(v0v, (b, h, s, d), &device).unwrap();
        let mask = zero_mask(b, s, &device);
        let scale = 1.0 / (d as f32).sqrt();

        let expected = eager_reference(
            &q0,
            &k0,
            &v0,
            None,
            None,
            &mask,
            scale,
            FullyMaskedPolicy::Propagate,
        )
        .unwrap();

        let qkv = qkv_from(&q0, &k0, &v0).unwrap();
        let (cos, sin) = rope_tables(s, d, &device); // unused (rope=false) but still a valid pack
        let rope_pack = pack_rope(&cos, &sin).unwrap();
        let op = AttentionBlockFused::new(scale, FullyMaskedPolicy::Propagate, false).unwrap();
        let got = fused(&qkv, &rope_pack, &mask, op).unwrap();

        let e: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();
        let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(e.len(), g.len());
        for (a, bb) in e.iter().zip(g.iter()) {
            assert!((a - bb).abs() < 1e-6, "{a} vs {bb}");
        }
    }

    #[test]
    fn cpu_fwd_bit_exact_vs_eager_reference_with_rope_and_window() {
        let device = Device::Cpu;
        let (b, h, s, d) = (1usize, 3usize, 9usize, HEAD_DIM);
        let n = b * h * s * d;
        let q0v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.11).sin()).collect();
        let k0v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.23).cos()).collect();
        let v0v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.31).sin()).collect();
        let q0 = Tensor::from_vec(q0v, (b, h, s, d), &device).unwrap();
        let k0 = Tensor::from_vec(k0v, (b, h, s, d), &device).unwrap();
        let v0 = Tensor::from_vec(v0v, (b, h, s, d), &device).unwrap();
        let mask = zero_mask(b, s, &device);
        let scale = 1.0 / (d as f32).sqrt();
        let half_window = 2usize;

        let (cos, sin) = rope_tables(s, d, &device);
        let band = test_sliding_window_band(s, half_window, mask.dtype(), mask.device()).unwrap();
        let combined_mask = mask.broadcast_add(&band).unwrap();
        let expected = eager_reference(
            &q0,
            &k0,
            &v0,
            Some(&cos),
            Some(&sin),
            &combined_mask,
            scale,
            FullyMaskedPolicy::Zeros,
        )
        .unwrap();

        let qkv = qkv_from(&q0, &k0, &v0).unwrap();
        let rope_pack = pack_rope(&cos, &sin).unwrap();
        let op = AttentionBlockFused::new(scale, FullyMaskedPolicy::Zeros, true).unwrap();
        let got = fused(&qkv, &rope_pack, &combined_mask, op).unwrap();

        let e: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();
        let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        for (a, bb) in e.iter().zip(g.iter()) {
            assert!((a - bb).abs() < 1e-6, "{a} vs {bb}");
        }
    }

    /// The fwd+bwd bit-exact proof under a WINDOW arm: a boundary shape
    /// `(3, 9, 2, 64)` and a production-scale shape `(2, 128, 16, 64)`,
    /// with a per-batch-VARYING padding mask (a `mrow_base` bug would
    /// silently broadcast one batch element's row onto every other) and a
    /// genuinely non-uniform `dy` seed (`sum_all().backward()`'s all-ones
    /// seed would cancel exactly the class of bug this oracle exists to
    /// catch). `assert_eq!` throughout: F32 CPU, the SAME op sequence
    /// (`RopeFused` + matmul + `SoftmaxLastDimFused` + matmul) either way,
    /// so the rounding model predicts bit-exact equality, not a tolerance.
    #[test]
    fn cpu_fwd_and_bwd_bit_exact_vs_eager_with_window_nonuniform_dy_and_per_batch_mask() {
        for &(b, h, s, d) in &[
            (3usize, 2usize, 9usize, HEAD_DIM),
            (2usize, 16usize, 128usize, HEAD_DIM),
        ] {
            let device = Device::Cpu;
            let n = b * s * 3 * h * d;
            let half_window = (s / 4).max(1);

            // Per-batch-VARYING padding: batch `bi` pads its last
            // `bi % (s/2).max(1)` keys — batch 0 is unpadded, later
            // batches pad an increasing, DIFFERENT number of keys, so a
            // hardcoded `mrow_base = 0` (always reading batch 0's row)
            // would diverge from the correct per-batch mask visibly.
            let mut mask_v = vec![0f32; b * s];
            for bi in 0..b {
                let pad = (bi % (s / 2).max(1)).min(s.saturating_sub(1));
                for ki in (s - pad)..s {
                    mask_v[bi * s + ki] = -10_000.0;
                }
            }
            let mask = Tensor::from_vec(mask_v, (b, 1, 1, s), &device).unwrap();
            let band =
                test_sliding_window_band(s, half_window, mask.dtype(), mask.device()).unwrap();
            let combined_mask = mask.broadcast_add(&band).unwrap();

            let qkv0: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin() * 0.4).collect();
            let qkv = Var::from_tensor(
                &Tensor::from_vec(qkv0.clone(), (b, s, 3, h, d), &device).unwrap(),
            )
            .unwrap();
            let (cos, sin) = rope_tables(s, d, &device);
            let rope_pack = pack_rope(&cos, &sin).unwrap();
            let scale = 1.0 / (d as f32).sqrt();
            let dy_v: Vec<f32> = (0..(b * s * h * d))
                .map(|i| ((i as f32) * 0.029).cos() * 0.6 + 0.05)
                .collect();
            let dy = Tensor::from_vec(dy_v, (b, s, h * d), &device).unwrap();

            // Fused op under test.
            let op = AttentionBlockFused::new(scale, FullyMaskedPolicy::Zeros, true).unwrap();
            let out_fused = qkv
                .as_tensor()
                .apply_op3(&rope_pack, &combined_mask, op)
                .unwrap();
            let loss_fused = (&out_fused * &dy).unwrap().sum_all().unwrap();
            let grads_fused = loss_fused.backward().unwrap();
            let dqkv_fused: Vec<f32> = grads_fused
                .get(&qkv)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let out_fused_v: Vec<f32> = out_fused.flatten_all().unwrap().to_vec1().unwrap();

            // Independent eager reference, driven from the SAME `qkv` Var
            // (narrow/transpose q/k/v out of it) so gradients accumulate
            // into the SAME tensor `AttentionBlockFused`'s own `dqkv`
            // scatter is compared against.
            let qkv_eager =
                Var::from_tensor(&Tensor::from_vec(qkv0, (b, s, 3, h, d), &device).unwrap())
                    .unwrap();
            let q0 = qkv_eager
                .as_tensor()
                .narrow(2, 0, 1)
                .unwrap()
                .squeeze(2)
                .unwrap()
                .transpose(1, 2)
                .unwrap()
                .contiguous()
                .unwrap();
            let k0 = qkv_eager
                .as_tensor()
                .narrow(2, 1, 1)
                .unwrap()
                .squeeze(2)
                .unwrap()
                .transpose(1, 2)
                .unwrap()
                .contiguous()
                .unwrap();
            let v0 = qkv_eager
                .as_tensor()
                .narrow(2, 2, 1)
                .unwrap()
                .squeeze(2)
                .unwrap()
                .transpose(1, 2)
                .unwrap()
                .contiguous()
                .unwrap();
            let out_eager = eager_reference(
                &q0,
                &k0,
                &v0,
                Some(&cos),
                Some(&sin),
                &combined_mask,
                scale,
                FullyMaskedPolicy::Zeros,
            )
            .unwrap();
            let loss_eager = (&out_eager * &dy).unwrap().sum_all().unwrap();
            let grads_eager = loss_eager.backward().unwrap();
            let dqkv_eager: Vec<f32> = grads_eager
                .get(&qkv_eager)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let out_eager_v: Vec<f32> = out_eager.flatten_all().unwrap().to_vec1().unwrap();

            assert_eq!(
                out_fused_v, out_eager_v,
                "fwd not bit-exact @ (b={b},h={h},s={s},d={d})"
            );
            assert_eq!(
                dqkv_fused, dqkv_eager,
                "dqkv not bit-exact @ (b={b},h={h},s={s},d={d})"
            );
        }
    }

    #[test]
    fn fully_masked_row_under_zeros_policy_outputs_zero_context() {
        // A short sequence with a wide window and a padding mask that
        // masks every key at one batch element (position >= 1) makes
        // every row of that batch element fully masked.
        let device = Device::Cpu;
        let (b, h, s, d) = (1usize, 1usize, 3usize, HEAD_DIM);
        let n = b * h * s * d;
        let q0 = Tensor::from_vec(vec![0.3f32; n], (b, h, s, d), &device).unwrap();
        let k0 = Tensor::from_vec(vec![0.7f32; n], (b, h, s, d), &device).unwrap();
        let v0 = Tensor::from_vec(
            (0..n as i64).map(|i| i as f32).collect::<Vec<_>>(),
            (b, h, s, d),
            &device,
        )
        .unwrap();
        let mask = Tensor::from_vec(vec![-10_000.0f32; s], (1, 1, 1, s), &device).unwrap();
        let scale = 1.0 / (d as f32).sqrt();

        let qkv = qkv_from(&q0, &k0, &v0).unwrap();
        let (cos, sin) = rope_tables(s, d, &device);
        let rope_pack = pack_rope(&cos, &sin).unwrap();
        let op = AttentionBlockFused::new(scale, FullyMaskedPolicy::Zeros, false).unwrap();
        let got = fused(&qkv, &rope_pack, &mask, op).unwrap();
        let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        assert!(g.iter().all(|&x| x == 0.0), "{g:?}");
    }

    /// The WINDOW-restricted fully-masked row: not every key in the
    /// sequence is padding (unlike the global case above), but for a
    /// query row deep enough that its ENTIRE window neighborhood happens
    /// to fall on padded keys, the row is still fully masked — the exact
    /// case `jammi_encoders::mask::sliding_window_mask`'s own doc proves
    /// ("a deep pad-query row in a local layer"). `s=6`, `half_window=1`:
    /// keys `4`/`5` are padding, keys `0..4` are real. Query row `5`'s
    /// window is `{4, 5}` (both padded) — fully masked, zero context row.
    /// Query row `0`'s window is `{0, 1}` (both real) — NOT fully masked,
    /// a non-vacuity check that this fixture's OTHER rows are ordinary.
    #[test]
    fn window_restricted_fully_masked_row_outputs_zero_context_row() {
        let device = Device::Cpu;
        let (b, h, s, d) = (1usize, 1usize, 6usize, HEAD_DIM);
        let half_window = 1usize;
        let n = b * h * s * d;
        let q0v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.19).sin() * 0.4).collect();
        let k0v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.23).cos() * 0.4).collect();
        let v0v: Vec<f32> = (0..n).map(|i| (i as f32) + 1.0).collect();
        let q0 = Tensor::from_vec(q0v, (b, h, s, d), &device).unwrap();
        let k0 = Tensor::from_vec(k0v, (b, h, s, d), &device).unwrap();
        let v0 = Tensor::from_vec(v0v, (b, h, s, d), &device).unwrap();
        let mut mask_v = vec![0f32; s];
        mask_v[4] = -10_000.0;
        mask_v[5] = -10_000.0;
        let mask = Tensor::from_vec(mask_v, (1, 1, 1, s), &device).unwrap();
        let band = test_sliding_window_band(s, half_window, mask.dtype(), mask.device()).unwrap();
        let combined_mask = mask.broadcast_add(&band).unwrap();
        let scale = 1.0 / (d as f32).sqrt();

        let qkv = qkv_from(&q0, &k0, &v0).unwrap();
        let (cos, sin) = rope_tables(s, d, &device);
        let rope_pack = pack_rope(&cos, &sin).unwrap();
        let op = AttentionBlockFused::new(scale, FullyMaskedPolicy::Zeros, true).unwrap();
        let got = fused(&qkv, &rope_pack, &combined_mask, op).unwrap();
        // got: [batch=1, seq=6, heads*head_dim=64].
        let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        let row = |qi: usize| &g[qi * h * d..(qi + 1) * h * d];

        assert!(
            row(5).iter().all(|&x| x == 0.0),
            "query row 5 (window {{4,5}}, both padded) must be all-zero: {:?}",
            row(5)
        );
        assert!(
            row(0).iter().any(|&x| x != 0.0),
            "query row 0 (window {{0,1}}, both real) must NOT be all-zero — otherwise this \
             fixture is vacuously all-masked and proves nothing about the WINDOW-restricted \
             case specifically: {:?}",
            row(0)
        );
    }

    /// `0.125` (`1/sqrt(64)`, this op's own production scale) is an exact
    /// power of two — accepted; `0.1` has a nonzero f32 mantissa (a
    /// terminating-but-not-power-of-two binary fraction) — refused. See
    /// `new`'s own doc for the DETERMINANT this guards (mantissa bits, not
    /// just finiteness).
    #[test]
    fn new_accepts_an_exact_power_of_two_scale_and_refuses_a_non_power_of_two_one() {
        let op = AttentionBlockFused::new(0.125, FullyMaskedPolicy::Propagate, false)
            .expect("0.125 = 2^-3 is an exact power of two");
        // The op's tape name is what error messages and the consumer-side
        // dispatch counters key on — pinned here (cargo-mutants' `""` /
        // `"xyzzy"` replacements were otherwise unobserved).
        assert_eq!(op.name(), "attention_block_fused");
        assert_eq!(
            op.scale(),
            0.125,
            "the accessor reads back the validated value"
        );
        let err = AttentionBlockFused::new(0.1, FullyMaskedPolicy::Propagate, false)
            .expect_err("0.1 has a nonzero f32 mantissa — not an exact power of two");
        assert!(matches!(err, Error::Msg(_)));
    }

    #[test]
    fn head_dim_other_than_64_is_refused() {
        let device = Device::Cpu;
        let (b, s, h, d) = (1usize, 2usize, 1usize, 8usize);
        let qkv = Tensor::zeros((b, s, 3, h, d), DType::F32, &device).unwrap();
        let mask = zero_mask(b, s, &device);
        let (cos, sin) = rope_tables(s, d, &device);
        let rope_pack = pack_rope(&cos, &sin).unwrap();
        let op = AttentionBlockFused::new(0.125, FullyMaskedPolicy::Propagate, false).unwrap();
        let err = fused(&qkv, &rope_pack, &mask, op).expect_err("head_dim != 64 must be refused");
        assert!(matches!(err, Error::Msg(_)));
    }

    /// `qkv` must be contiguous on BOTH devices (`cpu_fwd`'s own
    /// `l1.contiguous_offsets()` refusal; `cuda_fwd` carries the SAME
    /// check even though `gather_bhsd`'s `copy_strided_src` could
    /// structurally tolerate a strided source — so this op's public domain
    /// contract does not depend on which device runs it).
    #[test]
    fn non_contiguous_qkv_is_refused_not_silently_misread() {
        let device = Device::Cpu;
        let (b, s, h, d) = (1usize, 3usize, 1usize, HEAD_DIM);
        let mask = zero_mask(b, s, &device);
        let (cos, sin) = rope_tables(s, d, &device);
        let rope_pack = pack_rope(&cos, &sin).unwrap();
        let scale = 1.0 / (d as f32).sqrt();
        // `[batch, 3, seq, heads, head_dim]` transposed to the CORRECT
        // shape `[batch, seq, 3, heads, head_dim]` but non-contiguous.
        let big = Tensor::zeros((b, 3, s, h, d), DType::F32, &device).unwrap();
        let qkv = big.transpose(1, 2).unwrap();
        assert!(!qkv.is_contiguous());
        assert_eq!(qkv.dims(), &[b, s, 3, h, d]);
        let op = AttentionBlockFused::new(scale, FullyMaskedPolicy::Propagate, true).unwrap();
        let err = fused(&qkv, &rope_pack, &mask, op)
            .expect_err("a non-contiguous qkv must be refused, not silently misread");
        assert!(matches!(err, Error::RequiresContiguous { .. }));
    }

    /// A mask whose query-row axis is neither `1` nor `seq` (the ONE new
    /// domain this op's `mask` gained under the band-as-input redesign —
    /// see `check_mask`'s doc) is refused, not silently mis-broadcast.
    #[test]
    fn mask_query_row_axis_outside_one_or_seq_is_refused() {
        let device = Device::Cpu;
        let (b, s, h, d) = (1usize, 6usize, 1usize, HEAD_DIM);
        let qkv = Tensor::zeros((b, s, 3, h, d), DType::F32, &device).unwrap();
        // query-row axis = 3, neither 1 nor seq=6.
        let mask = Tensor::zeros((b, 1, 3, s), DType::F32, &device).unwrap();
        let (cos, sin) = rope_tables(s, d, &device);
        let rope_pack = pack_rope(&cos, &sin).unwrap();
        let op = AttentionBlockFused::new(0.125, FullyMaskedPolicy::Propagate, false).unwrap();
        let err =
            fused(&qkv, &rope_pack, &mask, op).expect_err("mask query-row axis 3 must be refused");
        assert!(matches!(err, Error::Msg(_)));
    }

    // ── Validator lattices: one typed-refusal test per state-table row
    // (see `attention_dims`/`check_mask`/`check_rope_pack`'s docs). Each
    // calls the validator DIRECTLY on a `Layout`, so a `MAX_SEQ`-sized
    // boundary cell costs no `[seq, seq]` buffer. ──

    fn lay(dims: &[usize]) -> Layout {
        Layout::contiguous(dims)
    }

    fn is_msg(r: &Result<impl std::fmt::Debug>) -> bool {
        matches!(r, Err(Error::Msg(_)))
    }

    /// `attention_dims`: rank ≠ 5 while the 3-axis IS 3 (so a `||`→`&&`
    /// mutation of the rank/3-axis check is what this cell reddens).
    #[test]
    fn attention_dims_rank_other_than_5_is_refused_even_with_a_valid_3_axis() {
        let r = attention_dims(&lay(&[1, 2, 3, HEAD_DIM]), "t");
        assert!(is_msg(&r), "rank 4 with dims[2]==3 must be refused: {r:?}");
        let r = attention_dims(&lay(&[1, 2, 3, 1, HEAD_DIM, 1]), "t");
        assert!(is_msg(&r), "rank 6 must be refused: {r:?}");
    }

    /// `attention_dims`: rank 5 but the 3-axis is not 3.
    #[test]
    fn attention_dims_three_axis_other_than_3_is_refused() {
        let r = attention_dims(&lay(&[1, 2, 2, 1, HEAD_DIM]), "t");
        assert!(is_msg(&r), "3-axis of 2 must be refused: {r:?}");
        let r = attention_dims(&lay(&[1, 2, 4, 1, HEAD_DIM]), "t");
        assert!(is_msg(&r), "3-axis of 4 must be refused: {r:?}");
    }

    /// `attention_dims`: the `MAX_SEQ` boundary from BOTH sides —
    /// `MAX_SEQ` itself accepted (a `>=` mutation would refuse it),
    /// `MAX_SEQ + 1` AND `MAX_SEQ + 2` refused (an `==` mutation would
    /// accept the latter).
    #[test]
    fn attention_dims_seq_at_max_seq_is_accepted_and_past_it_is_refused() {
        let ok = attention_dims(&lay(&[1, MAX_SEQ, 3, 1, HEAD_DIM]), "t").unwrap();
        assert_eq!(ok, (1, MAX_SEQ, 1, HEAD_DIM));
        for over in [MAX_SEQ + 1, MAX_SEQ + 2, 2 * MAX_SEQ] {
            let r = attention_dims(&lay(&[1, over, 3, 1, HEAD_DIM]), "t");
            assert!(is_msg(&r), "seq={over} > MAX_SEQ must be refused: {r:?}");
        }
        let ok = attention_dims(&lay(&[0, 0, 3, 0, HEAD_DIM]), "t").unwrap();
        assert_eq!(ok, (0, 0, 0, HEAD_DIM), "all-zero extents are in-domain");
    }

    /// `check_mask`: rank ≠ 4 while every axis it DOES have is valid
    /// (`dims[1] == 1`, `dims[3] == s`) — the cell a `||`→`&&` mutation
    /// of the rank/heads/key-axis check reddens on.
    #[test]
    fn check_mask_rank_other_than_4_is_refused_even_with_valid_axes() {
        let (b, s) = (2usize, 4usize);
        let r = check_mask(&lay(&[1, 1, 1, s, 1]), b, s, "t");
        assert!(is_msg(&r), "rank-5 mask must be refused: {r:?}");
        let r = check_mask(&lay(&[1, 1, s]), b, s, "t");
        assert!(is_msg(&r), "rank-3 mask must be refused: {r:?}");
    }

    /// `check_mask`: a heads axis other than 1 (the mask never carries
    /// heads — see `SoftmaxLastDimFused`'s broadcast class).
    #[test]
    fn check_mask_heads_axis_other_than_1_is_refused() {
        let (b, s) = (2usize, 4usize);
        let r = check_mask(&lay(&[1, 2, 1, s]), b, s, "t");
        assert!(is_msg(&r), "heads axis 2 must be refused: {r:?}");
    }

    /// `check_mask`: a key axis other than `seq`.
    #[test]
    fn check_mask_key_axis_other_than_seq_is_refused() {
        let (b, s) = (2usize, 4usize);
        for bad in [s - 1, s + 1, 1] {
            let r = check_mask(&lay(&[1, 1, 1, bad]), b, s, "t");
            assert!(
                is_msg(&r),
                "key axis {bad} != seq={s} must be refused: {r:?}"
            );
        }
    }

    /// `check_mask`: the query-row axis must be exactly 1 or `seq`; the
    /// two accepted cells return the axis size the caller indexes with.
    #[test]
    fn check_mask_query_row_axis_cells() {
        let (b, s) = (2usize, 4usize);
        assert_eq!(check_mask(&lay(&[1, 1, 1, s]), b, s, "t").unwrap(), (1, 1));
        assert_eq!(check_mask(&lay(&[1, 1, s, s]), b, s, "t").unwrap(), (1, s));
        for bad in [2usize, 3, s + 1] {
            let r = check_mask(&lay(&[1, 1, bad, s]), b, s, "t");
            assert!(is_msg(&r), "query-row axis {bad} must be refused: {r:?}");
        }
    }

    /// `check_mask`: the leading axis must be exactly 1 or `batch`; both
    /// accepted cells are returned so the caller can pick its row block.
    #[test]
    fn check_mask_leading_axis_cells() {
        let (b, s) = (2usize, 4usize);
        assert_eq!(check_mask(&lay(&[1, 1, 1, s]), b, s, "t").unwrap(), (1, 1));
        assert_eq!(check_mask(&lay(&[b, 1, s, s]), b, s, "t").unwrap(), (b, s));
        for bad in [b + 1, b + 2, 0] {
            let r = check_mask(&lay(&[bad, 1, 1, s]), b, s, "t");
            assert!(is_msg(&r), "leading axis {bad} must be refused: {r:?}");
        }
    }

    /// `check_rope_pack`: rank ≠ 5 while every axis it DOES have is valid.
    #[test]
    fn check_rope_pack_rank_other_than_5_is_refused_even_with_valid_axes() {
        let (s, d) = (4usize, HEAD_DIM);
        let r = check_rope_pack(&lay(&[2, 1, 1, s, d, 1]), s, d, "t");
        assert!(is_msg(&r), "rank-6 rope_pack must be refused: {r:?}");
        let r = check_rope_pack(&lay(&[2, 1, 1, s]), s, d, "t");
        assert!(is_msg(&r), "rank-4 rope_pack must be refused: {r:?}");
    }

    /// `check_rope_pack`: each of the three fixed leading axes and the
    /// head_dim axis, violated one at a time.
    #[test]
    fn check_rope_pack_each_fixed_axis_is_refused_when_violated_alone() {
        let (s, d) = (4usize, HEAD_DIM);
        for (bad, why) in [
            ([3, 1, 1, s, d], "cos/sin axis 3"),
            ([1, 1, 1, s, d], "cos/sin axis 1"),
            ([2, 2, 1, s, d], "axis 1 == 2"),
            ([2, 1, 2, s, d], "axis 2 == 2"),
            ([2, 1, 1, s, 32], "head_dim 32"),
            ([2, 1, 1, s, d + 2], "head_dim d+2"),
        ] {
            let r = check_rope_pack(&lay(&bad), s, d, "t");
            assert!(is_msg(&r), "{why} must be refused: {r:?}");
        }
    }

    /// `check_rope_pack`: the table's position axis from both sides of
    /// `seq` — `s - 1` refused, `s` (boundary) and `s + 1` accepted and
    /// returned as seq_max.
    #[test]
    fn check_rope_pack_short_table_is_refused_and_boundary_is_accepted() {
        let (s, d) = (4usize, HEAD_DIM);
        let r = check_rope_pack(&lay(&[2, 1, 1, s - 1, d]), s, d, "t");
        assert!(is_msg(&r), "seq_max = s - 1 must be refused: {r:?}");
        assert_eq!(
            check_rope_pack(&lay(&[2, 1, 1, s, d]), s, d, "t").unwrap(),
            s
        );
        assert_eq!(
            check_rope_pack(&lay(&[2, 1, 1, s + 1, d]), s, d, "t").unwrap(),
            s + 1
        );
    }

    /// `cpu_fwd`'s empty fast path, `b == 0` cell (`s == 0`'s cell is
    /// `empty_seq_is_a_no_op_not_a_panic`, `h == 0`'s is below).
    #[test]
    fn empty_batch_is_a_no_op_not_a_panic() {
        let device = Device::Cpu;
        let (b, s, h, d) = (0usize, 3usize, 2usize, HEAD_DIM);
        let qkv = Tensor::zeros((b, s, 3, h, d), DType::F32, &device).unwrap();
        let mask = Tensor::zeros((1, 1, 1, s), DType::F32, &device).unwrap();
        let (cos, sin) = rope_tables(s, d, &device);
        let rope_pack = pack_rope(&cos, &sin).unwrap();
        let op = AttentionBlockFused::new(0.125, FullyMaskedPolicy::Propagate, true).unwrap();
        let got = fused(&qkv, &rope_pack, &mask, op).unwrap();
        assert_eq!(got.dims(), &[b, s, h * d]);
        assert_eq!(got.elem_count(), 0);
    }

    /// `cpu_fwd`'s empty fast path, `h == 0` cell.
    #[test]
    fn empty_heads_is_a_no_op_not_a_panic() {
        let device = Device::Cpu;
        let (b, s, h, d) = (2usize, 3usize, 0usize, HEAD_DIM);
        let qkv = Tensor::zeros((b, s, 3, h, d), DType::F32, &device).unwrap();
        let mask = Tensor::zeros((b, 1, 1, s), DType::F32, &device).unwrap();
        let (cos, sin) = rope_tables(s, d, &device);
        let rope_pack = pack_rope(&cos, &sin).unwrap();
        let op = AttentionBlockFused::new(0.125, FullyMaskedPolicy::Propagate, true).unwrap();
        let got = fused(&qkv, &rope_pack, &mask, op).unwrap();
        assert_eq!(got.dims(), &[b, s, 0]);
        assert_eq!(got.elem_count(), 0);
    }

    /// Family D boundary oracle (campaign #443 D1): the CPU domain is
    /// `F32`-only, unaffected by this campaign's CUDA-side `F16` widening
    /// (`crate::cuda::attention_block`'s dtype check) — candle-core 0.11's
    /// CPU backend has no `BF16`/`F16` `MatMul` impl (module doc's "Domain"
    /// section). Both 16-bit dtypes must be refused with a TYPED
    /// `UnsupportedDTypeForOp` naming the OFFENDING dtype, never a silent
    /// upcast and never a generic error three calls deep inside a matmul —
    /// pinned on BOTH dtypes so a future CPU `BF16`-only carve-out could not
    /// silently leave `F16` refused via some OTHER, undocumented path.
    #[test]
    fn cpu_fwd_refuses_both_16_bit_dtypes_with_a_typed_error_naming_the_dtype() {
        let device = Device::Cpu;
        let (b, s, h, d) = (1usize, 3usize, 2usize, HEAD_DIM);
        for dtype in [DType::BF16, DType::F16] {
            let qkv = Tensor::zeros((b, s, 3, h, d), dtype, &device).unwrap();
            let mask = Tensor::zeros((1, 1, 1, s), dtype, &device).unwrap();
            let (cos, sin) = rope_tables(s, d, &device);
            let rope_pack = pack_rope(&cos, &sin).unwrap().to_dtype(dtype).unwrap();
            let op = AttentionBlockFused::new(0.125, FullyMaskedPolicy::Propagate, true).unwrap();
            let err = fused(&qkv, &rope_pack, &mask, op)
                .expect_err(&format!("{dtype:?} must be refused on the CPU arm"));
            assert!(
                matches!(err, Error::UnsupportedDTypeForOp(got, _) if got == dtype),
                "{dtype:?}: expected UnsupportedDTypeForOp naming {dtype:?}, got {err:?}"
            );
        }
    }

    #[test]
    fn empty_seq_is_a_no_op_not_a_panic() {
        let device = Device::Cpu;
        let (b, s, h, d) = (2usize, 0usize, 3usize, HEAD_DIM);
        let qkv = Tensor::zeros((b, s, 3, h, d), DType::F32, &device).unwrap();
        let mask = Tensor::zeros((b, 1, 1, s), DType::F32, &device).unwrap();
        let (cos, sin) = rope_tables(1, d, &device);
        let rope_pack = pack_rope(&cos, &sin).unwrap();
        let op = AttentionBlockFused::new(0.125, FullyMaskedPolicy::Propagate, false).unwrap();
        let got = fused(&qkv, &rope_pack, &mask, op).unwrap();
        assert_eq!(got.elem_count(), 0);
    }

    /// `dqkv == cat(dq, dk, dv)` — the op contract's own oracle: gradcheck
    /// via finite differences on a small fixture proves `bwd`'s SCATTER
    /// (`Tensor::cat` of the three per-slot gradients back into `qkv`'s own
    /// `[B,S,3,H,D]` layout) lines up with the forward's own `[Q|K|V]`
    /// gather order.
    #[test]
    fn gradcheck_dqkv_vs_central_finite_differences() {
        let device = Device::Cpu;
        let (b, h, s, d) = (1usize, 2usize, 3usize, HEAD_DIM);
        let n = b * s * 3 * h * d;
        let qkv0: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.07).sin() * 0.5).collect();
        let qkv =
            Var::from_tensor(&Tensor::from_vec(qkv0.clone(), (b, s, 3, h, d), &device).unwrap())
                .unwrap();
        let mask = zero_mask(b, s, &device);
        let (cos, sin) = rope_tables(s, d, &device);
        let rope_pack = pack_rope(&cos, &sin).unwrap();
        let scale = 1.0 / (d as f32).sqrt();
        let op = AttentionBlockFused::new(scale, FullyMaskedPolicy::Propagate, true).unwrap();

        let out = fused(qkv.as_tensor(), &rope_pack, &mask, op).unwrap();
        let grads = out.sum_all().unwrap().backward().unwrap();
        let dqkv: Vec<f32> = grads
            .get(&qkv)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let sum_fwd = |v: &[f32]| -> f64 {
            let t = Tensor::from_vec(v.to_vec(), (b, s, 3, h, d), &device).unwrap();
            fused(&t, &rope_pack, &mask, op)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap() as f64
        };
        let eps = 2e-3f32;
        // DERIVED tolerance, not a round number: the central difference
        // divides the difference of two F32 sums (each rounded to within
        // one F32 ULP of `|sum|`, `f32::EPSILON * |sum|` as the upper
        // bound on that ULP) by `2 * eps`, so its own rounding noise is
        // bounded by `2 * ulp(|sum|) / (2 * eps)`; `8 *` that is the
        // floor (a few ULPs of accumulation slack in each 384-element
        // sum), plus `eps^2` for the central difference's truncation term
        // (order `eps^2` times the third derivative, which is `O(1)`
        // here). Measured on this fixture: `|sum| = 2.46`, max
        // `|numeric - analytic| = 4.7e-5` against `tol = 5.9e-4` (ratio
        // 0.08); the previous flat `5e-2` was ~1000x the observed error.
        // Discrimination (verified and reverted): dropping the scale from
        // `dq` (`let dqr = dqs.clone()` in `bwd`) moves `dqkv[0]` from
        // `0.166` to `1.33`, `|Δ| = 1.16`, ~2000x this tolerance (and only
        // ~23x the old one).
        let sum0 = sum_fwd(&qkv0);
        let tol = 8.0 * (f64::from(f32::EPSILON) * sum0.abs()) / (2.0 * f64::from(eps))
            + f64::from(eps) * f64::from(eps);
        // Sample one index per `qkv` slot (Q: 0 and 1, K: n/2, V: n-1)
        // rather than every one of `n` (cheap, still a real
        // finite-difference proof of the scatter/gather round-trip and the
        // RoPE/scale/softmax chain feeding it).
        let mut max_ratio = 0f64;
        for &i in &[0usize, 1, n / 2, n - 1] {
            let mut vp = qkv0.clone();
            vp[i] += eps;
            let mut vm = qkv0.clone();
            vm[i] -= eps;
            let numeric = (sum_fwd(&vp) - sum_fwd(&vm)) / (2.0 * f64::from(eps));
            let err = (numeric - f64::from(dqkv[i])).abs();
            max_ratio = max_ratio.max(err / tol);
            assert!(
                err < tol,
                "dqkv[{i}]: numeric {numeric} vs analytic {} (|Δ| = {err:e} > tol {tol:e})",
                dqkv[i]
            );
        }
        assert!(
            max_ratio > 0.0,
            "a zero error ratio would mean the finite difference never moved — vacuous"
        );
    }

    #[test]
    fn track_op_asserted_on_rope_pack_and_mask() {
        let device = Device::Cpu;
        let (b, h, s, d) = (1usize, 1usize, 2usize, HEAD_DIM);
        let qkv = Var::from_tensor(&Tensor::zeros((b, s, 3, h, d), DType::F32, &device).unwrap())
            .unwrap();
        let (cos, sin) = rope_tables(s, d, &device);
        let rope_pack = Var::from_tensor(&pack_rope(&cos, &sin).unwrap()).unwrap();
        let mask = zero_mask(b, s, &device);
        let scale = 1.0 / (d as f32).sqrt();
        let op = AttentionBlockFused::new(scale, FullyMaskedPolicy::Propagate, true).unwrap();
        let out = fused(qkv.as_tensor(), rope_pack.as_tensor(), &mask, op).unwrap();
        let err = out.sum_all().unwrap().backward().expect_err(
            "a tracked rope_pack must make backward fail loudly, not silently drop its gradient",
        );
        let _ = err;
    }

    /// `bwd` runs DETACHED (module doc's "`bwd` runs DETACHED" section):
    /// driven from a TRACKED `qkv` (a `Var`, exactly as production's
    /// `Wqkv` output is downstream of the LoRA `Var`s — standing rule: a
    /// leaf-tensor fixture cannot see tape retention at all), the `dqkv`
    /// it returns must carry NO `BackpropOp` — `!track_op()` and an EMPTY
    /// `sorted_nodes()` — both when `bwd` is called directly and when
    /// candle's engine stores it in the `GradStore` (`grads.get(&var)`,
    /// which is `zeros_like(var) + dqkv` and therefore tracks iff `dqkv`
    /// does). Without the four `detach()` calls at the top of `bwd`, every
    /// intermediate clones into its consumer's `Op` (candle-core 0.11.0
    /// `op.rs`'s `BackpropOp::new2`), `dqkv`'s graph reaches all the way
    /// back to the `Var`, and this test's `sorted_nodes()` count is in
    /// the dozens — the mutation this test reddens under (verified:
    /// deleting the four `detach()` lines).
    #[test]
    fn bwd_from_a_tracked_qkv_returns_an_untracked_dqkv_with_no_tape() {
        let device = Device::Cpu;
        let (b, h, s, d) = (2usize, 2usize, 5usize, HEAD_DIM);
        let n = b * s * 3 * h * d;
        let qkv0: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.05).sin() * 0.5).collect();
        let qkv =
            Var::from_tensor(&Tensor::from_vec(qkv0, (b, s, 3, h, d), &device).unwrap()).unwrap();
        let (cos, sin) = rope_tables(s, d, &device);
        let rope_pack = pack_rope(&cos, &sin).unwrap();
        let mask = zero_mask(b, s, &device);
        let scale = 1.0 / (d as f32).sqrt();
        let op = AttentionBlockFused::new(scale, FullyMaskedPolicy::Zeros, true).unwrap();
        let dy_v: Vec<f32> = (0..(b * s * h * d))
            .map(|i| ((i as f32) * 0.031).cos() * 0.6 + 0.05)
            .collect();
        let dy = Tensor::from_vec(dy_v, (b, s, h * d), &device).unwrap();

        // Non-vacuity: the forward output IS tracked (the fixture reaches
        // `bwd` through a Var), so an untracked `dqkv` is `bwd`'s doing.
        let res = fused(qkv.as_tensor(), &rope_pack, &mask, op).unwrap();
        assert!(qkv.as_tensor().track_op());
        assert!(res.track_op());
        assert!(!res.sorted_nodes().is_empty());

        // Direct call, exactly as candle's engine makes it (the engine
        // hands `bwd` an already-DETACHED `grad`, mirrored here by `dy`
        // being a leaf).
        let (dqkv, d_rope, d_mask) =
            CustomOp3::bwd(&op, qkv.as_tensor(), &rope_pack, &mask, &res, &dy).unwrap();
        let dqkv = dqkv.expect("dqkv is always Some");
        assert!(d_rope.is_none() && d_mask.is_none());
        assert!(
            !dqkv.track_op(),
            "dqkv must be detached: bwd's intermediates would otherwise all be retained by \
             their consumers' Ops and handed back to the engine"
        );
        assert_eq!(
            dqkv.sorted_nodes().len(),
            0,
            "dqkv must carry an EMPTY tape — every intermediate freed at its drop point"
        );

        // Through the engine: the stored grad for the Var is
        // `zeros_like + dqkv`, tracked iff `dqkv` is.
        let loss = (&res * &dy).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let stored = grads.get(&qkv).unwrap();
        assert!(!stored.track_op());
        assert_eq!(stored.sorted_nodes().len(), 0);
        // And the VALUES are the same either way (detaching changes no
        // byte — `detach` shares storage).
        let direct: Vec<f32> = dqkv.flatten_all().unwrap().to_vec1().unwrap();
        let via_engine: Vec<f32> = stored.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(direct, via_engine);
    }

    /// `cuda_fwd`'s `cos_l`/`sin_l` derivation assumes `rope_pack`
    /// is itself contiguous from its own `start_offset` (`sin`'s offset is
    /// computed by ADDING `s_max * d` to that start offset) — a narrowed
    /// or transposed `rope_pack` would silently read the WRONG elements
    /// rather than error, the same "missing-offset" class this crate's
    /// CUDA glue idioms guard against elsewhere. `cpu_fwd` shares the
    /// SAME `check_rope_pack` shape check plus its own
    /// `l2.contiguous_offsets()` refusal, so both devices refuse a
    /// transposed-view `rope_pack` identically — this is the CPU half of
    /// that proof; `cuda_parity.rs`'s CUDA leg proves the other.
    #[test]
    fn transposed_view_rope_pack_is_refused_not_silently_misread() {
        let device = Device::Cpu;
        let (b, h, s, d) = (1usize, 1usize, 3usize, HEAD_DIM);
        let qkv = Tensor::zeros((b, s, 3, h, d), DType::F32, &device).unwrap();
        let mask = zero_mask(b, s, &device);
        let scale = 1.0 / (d as f32).sqrt();
        // Built as `[2, 1, 1, d, s]` then transposed to the correct SHAPE
        // (`[2, 1, 1, s, d]`) but now a non-contiguous VIEW, not a fresh
        // contiguous buffer — `check_rope_pack`'s shape check alone would
        // accept this; only the contiguity check catches it.
        let big = Tensor::zeros((2, 1, 1, d, s), DType::F32, &device).unwrap();
        let rope_pack = big.transpose(3, 4).unwrap();
        assert!(!rope_pack.is_contiguous());
        assert_eq!(rope_pack.dims(), &[2, 1, 1, s, d]);
        let op = AttentionBlockFused::new(scale, FullyMaskedPolicy::Propagate, true).unwrap();
        let err = fused(&qkv, &rope_pack, &mask, op)
            .expect_err("a transposed-view rope_pack must be refused, not silently misread");
        assert!(matches!(err, Error::RequiresContiguous { .. }));
    }
}
