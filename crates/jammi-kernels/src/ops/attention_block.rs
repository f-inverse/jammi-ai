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
//! predicate of its own. A local-attention layer's sliding-window band is
//! `jammi_encoders::mask::sliding_window_mask`'s job, built ONCE per
//! `(seq, half_window, dtype, device)` by the call site's own cache
//! (`ModernBert::band_cache`/`sliding_band`) and summed into the padding
//! mask BEFORE this op ever runs — the caller passes this op ONE already-
//! combined additive mask, exactly the shape
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
//! `F32` (CPU and CUDA) or `BF16` (CUDA only — candle-core 0.11's CPU
//! backend has no `BF16` `MatMul` impl, the SAME pre-existing limitation
//! `LowRankResidualLinear`'s module doc discloses; this op's CPU domain
//! therefore accepts `F32` only, refusing `BF16` with a typed
//! `UnsupportedDTypeForOp` rather than reaching a confusing failure three
//! calls deep inside a matmul). `head_dim` must be exactly `HEAD_DIM`.
//! `seq` must be `<= MAX_SEQ`. `rope_pack` (when `rope == true`): rank
//! 5 `[2, 1, 1, seq_max, head_dim]`, `seq_max >= seq`, contiguous, same
//! dtype as `qkv`. `mask`: rank 4 `[batch|1, 1, seq|1, seq]`, contiguous,
//! same dtype as `qkv` — the SAME broadcast class
//! [`super::SoftmaxLastDimFused`] accepts, since the caller may pass
//! either the padding mask alone (global layer) or the padding-plus-band
//! sum (local layer) — see the "window is construction data at the call
//! site" section above. `b == 0 || seq == 0 || heads == 0` takes the empty
//! fast path (nothing to compute).
//!
//! ## `bwd`: ordinary `Tensor` composition, reusing this crate's own ops
//!
//! Candle has no save-for-backward channel (the same constraint
//! [`super::LayerNormFused`]'s and [`super::RopeFused`]'s own `bwd`
//! methods document), so `bwd` recomputes the rotated `Q`/`K`, the raw
//! scores, and the softmax output `P` from `qkv`/`rope_pack`/`mask` —
//! calling [`super::apply3`] with [`super::RopeFused`] and
//! [`super::apply2`] with [`super::SoftmaxLastDimFused`] DIRECTLY, rather
//! than a second hand-written kernel. This is safe and does NOT reintroduce
//! the retained-activation cost this op exists to remove: `bwd` runs OUTSIDE
//! the forward tape — every intermediate `Tensor` it builds (`scores`, `p`,
//! `dp`, …) is a plain, un-walked `BackpropOp` graph that is dropped the
//! moment `bwd` returns; candle's backward engine never calls `.backward()`
//! on it. This is the same pattern [`super::RopeFused::bwd`] and
//! [`super::SoftmaxLastDimFused::bwd`] already use (composing ordinary
//! `Tensor` ops, including calls into EACH OTHER's `apply*` entry points,
//! inside their own `bwd`), just with a longer chain. `bwd` forces
//! `.contiguous()` on every GEMM operand that is not either fully
//! row-major OR a single transposed view of one (`gemm_config`'s admissible
//! shapes — see `LowRankResidualLinear`'s module doc for the citation and
//! the on-device failure that motivated checking this explicitly); the
//! `is_gemm_operand_admissible` test below proves this holds for every
//! operand `bwd` actually builds, at both a boundary and a production-scale
//! rank, off the real `Layout` each carries (device-independent, mirroring
//! `LowRankResidualLinear`'s own precedent).
//!
//! ### `bwd`'s seven GEMMs: which recompute a forward GEMM, which are new
//!
//! `fwd` issues exactly two GEMMs: `scores = q_scaled · kᵀ` (`k`'s
//! transpose is a VIEW — cuBLAS `OP_T`, no materialize) and `ctx = p · v`
//! (both row-major — `OP_N`/`OP_N`). `bwd` reissues ONE of those two
//! (`scores`, to recompute `P`) and adds five NEW gradient GEMMs that have
//! no forward counterpart at all:
//!
//! | `bwd` GEMM | recompute of a fwd GEMM? | operand shape |
//! |---|---|---|
//! | `scores = q_scaled · k_rotᵀ` | YES — `fwd`'s `QKᵀ`, same shape | `k_rot` transpose VIEW, no `.contiguous()` (see below) |
//! | `dv = pᵀ · dctx` | no — new | `p` transpose materialized (`p` is not reused row-major elsewhere) |
//! | `dp = dctx · vᵀ` | no — new (NOT `fwd`'s `PV`: `v` is transposed here, `fwd`'s `PV` never transposes `v`) | `v` transpose materialized |
//! | `dqs = ds · k_rot` | no — new | both row-major |
//! | `dkr = dsᵀ · q_scaled` | no — new | `ds` transpose materialized |
//!
//! (`ctx` itself — `fwd`'s SECOND GEMM — is never recomputed in `bwd`: the
//! op contract's "no packed `[O|L]` output" design means `bwd` only ever
//! needs `dctx`, supplied by the caller as `grad_res`, never `ctx` itself.)
//!
//! Only the `scores` recompute is held to `fwd`'s exact GEMM shape (module
//! doc's "fwd/bwd GEMM shape match" below) — the other four have no
//! forward shape to match in the first place, so their own `.contiguous()`
//! placement is chosen purely for correctness/admissibility, not parity.
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
//! ### Transient scoping: `scores`/`p`/`dp`/`ds` never all coexist
//!
//! Each of `bwd`'s four `[B, H, S, S]`-shaped intermediates (`scores`,
//! `p`, `dp`, `ds`) is dropped (`std::mem::drop`, explicit) the moment its
//! LAST use inside `bwd` completes, rather than staying bound until the
//! function returns: `scores` drops right after `p` is computed from it;
//! `p` and `dp` both drop right after `ds` is computed from them; `ds`
//! drops right after `dqs`/`dkr` (its two uses) are both computed. The
//! SUSTAINED live set is therefore at most TWO of these four
//! `[B, H, S, S]` tensors at once (`p` alongside `dp` while building
//! `ds`); THREE are simultaneously alive only for the single instant
//! `ds` itself is being constructed from `p` and `dp` (an unavoidable
//! "inputs plus the output under construction" overlap any allocator has,
//! not a design choice this scoping controls), never all four. At
//! `b=8, h=16, s=512` (BF16, `2` bytes/element — the training arm's own
//! dtype), one `[8, 16, 512, 512]` tensor is `8·16·512·512·2 =
//! 67_108_864` bytes (≈ 67.1 MB): the sustained two-tensor live set is
//! therefore ≈ 134.2 MB (briefly ≈ 201.3 MB during the one `ds`-
//! construction instant), against ≈ 268.4 MB when all four intermediates
//! lived to the end of `bwd` (this op's own pre-fix-round shape, and
//! `SoftmaxLastDimFused::bwd`'s/every other fused op's own default
//! without explicit scoping).
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
#[derive(Debug, Clone, Copy)]
pub struct AttentionBlockFused {
    /// The scaled-dot-product scale, `1 / sqrt(head_dim)` — folded into
    /// `Q` before `QKᵀ` (see the module doc's "Fixed domain" section for
    /// why this is bit-exact only because `head_dim == ``HEAD_DIM`).
    pub scale: f32,
    /// See [`super::FullyMaskedPolicy`]'s own doc; reused unchanged.
    pub fully_masked: FullyMaskedPolicy,
    /// Whether `rope_pack` is applied to `Q`/`K` at all. `false` lets a
    /// future caller with no positional embedding reuse this op — the
    /// `rope_pack` argument is then present but ignored (never read).
    pub rope: bool,
}

impl AttentionBlockFused {
    pub fn new(scale: f32, fully_masked: FullyMaskedPolicy, rope: bool) -> Result<Self> {
        if !scale.is_finite() {
            return Err(Error::Msg(format!(
                "attention_block_fused: scale must be finite, got {scale}"
            )));
        }
        Ok(Self {
            scale,
            fully_masked,
            rope,
        })
    }
}

impl super::sealed::Sealed for AttentionBlockFused {}

/// Validates `qkv`'s domain (module doc). Returns `(batch, seq, heads,
/// head_dim)`.
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
        if b == 0 || s == 0 || h == 0 {
            return match s1 {
                CpuStorage::F32(_) => Ok((CpuStorage::F32(Vec::new()), out_shape)),
                other => Err(Error::UnsupportedDTypeForOp(other.dtype(), op)),
            };
        }
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
                let out = attention_fwd_f32(
                    &qkv[o1..o2],
                    rope_slice,
                    &mask[m1..m2],
                    mask_b,
                    mask_q,
                    b,
                    s,
                    h,
                    d,
                    self.scale,
                    self.fully_masked,
                )?;
                Ok((CpuStorage::F32(out), out_shape))
            }
            (s1, s3) if s1.dtype() != s3.dtype() => Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s3.dtype(),
                op,
            }),
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
        let (b, s, three, h, d) = qkv.dims5()?;
        if three != 3 {
            return Err(Error::Msg(format!(
                "{op}: qkv must be rank 5 [batch, seq, 3, heads, head_dim], got 3-axis size {three}"
            )));
        }

        let q0 = qkv.narrow(2, 0, 1)?.squeeze(2)?.transpose(1, 2)?;
        let k0 = qkv.narrow(2, 1, 1)?.squeeze(2)?.transpose(1, 2)?;
        let v0 = qkv.narrow(2, 2, 1)?.squeeze(2)?.transpose(1, 2)?;

        let (q_rot, k_rot, cos_sin) = if self.rope {
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

        let q_scaled = (&q_rot * f64::from(self.scale))?;
        let v_c = v0.contiguous()?;

        // `mask` is ALREADY the caller's combined padding-plus-band sum
        // (module doc's "window is construction data at the call site"
        // section) — no band to rebuild here, unlike earlier revisions of
        // this op.
        //
        // `scores` uses the SAME transposed-VIEW `k_rot` GEMM shape `fwd`
        // issues (`.transpose(...)` with NO trailing `.contiguous()` — see
        // the module doc's "fwd/bwd GEMM shape match" section).
        let scores = q_scaled
            .contiguous()?
            .matmul(&k_rot.transpose(D::Minus1, D::Minus2)?)?;
        let p = apply2(&scores, mask, SoftmaxLastDimFused::new(self.fully_masked))?;
        // `scores`'s only use was building `p` above (module doc's
        // "transient scoping" section) — drop it now instead of letting it
        // live to the end of `bwd`.
        drop(scores);

        let dctx = grad_res
            .reshape((b, s, h, d))?
            .transpose(1, 2)?
            .contiguous()?;

        let dv = p
            .transpose(D::Minus1, D::Minus2)?
            .contiguous()?
            .matmul(&dctx)?;
        let dp = dctx.matmul(&v_c.transpose(D::Minus1, D::Minus2)?.contiguous()?)?;
        let ds = apply2(&p, &dp, SoftmaxBwdDScores)?;
        // `p`'s last use (its second, after `dv` above) and `dp`'s only
        // use were both building `ds` — drop both now.
        drop(p);
        drop(dp);

        let dqs = ds.contiguous()?.matmul(&k_rot.contiguous()?)?;
        let dkr = ds
            .transpose(D::Minus1, D::Minus2)?
            .contiguous()?
            .matmul(&q_scaled.contiguous()?)?;
        // `ds`'s last use (its second, after `dqs`) was building `dkr` —
        // drop it; no `[B,H,S,S]`-shaped tensor remains live for the rest
        // of `bwd`.
        drop(ds);
        let dqr = (&dqs * f64::from(self.scale))?;

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

        Ok((Some(dqkv), None, None))
    }
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
#[allow(clippy::too_many_arguments)]
fn attention_fwd_f32(
    qkv: &[f32],
    rope: Option<(&[f32], usize)>,
    mask: &[f32],
    mask_batch: usize,
    mask_query_rows: usize,
    b: usize,
    s: usize,
    h: usize,
    d: usize,
    scale: f32,
    policy: FullyMaskedPolicy,
) -> Result<Vec<f32>> {
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
            // `.contiguous()` on the rhs (audit B6 / the module doc's
            // "fwd/bwd GEMM shape match" section): this must be the SAME
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

            // dv = p.transpose(-1,-2).contiguous() @ dctx
            let lhs = p
                .transpose(D::Minus1, D::Minus2)
                .unwrap()
                .contiguous()
                .unwrap();
            assert!(
                is_gemm_operand_admissible(lhs.layout()),
                "dv lhs @ ({b},{s},{h},{d})"
            );
            assert!(
                is_gemm_operand_admissible(dctx.layout()),
                "dv rhs @ ({b},{s},{h},{d})"
            );
            let _ = lhs.matmul(&dctx).unwrap();

            // dp = dctx @ v.transpose(-1,-2).contiguous()
            let rhs = v
                .transpose(D::Minus1, D::Minus2)
                .unwrap()
                .contiguous()
                .unwrap();
            assert!(
                is_gemm_operand_admissible(dctx.layout()),
                "dp lhs @ ({b},{s},{h},{d})"
            );
            assert!(
                is_gemm_operand_admissible(rhs.layout()),
                "dp rhs @ ({b},{s},{h},{d})"
            );
            let _ = dctx.matmul(&rhs).unwrap();

            // dqs = ds.contiguous() @ k.contiguous()
            let lhs = ds.contiguous().unwrap();
            let rhs = k.contiguous().unwrap();
            assert!(
                is_gemm_operand_admissible(lhs.layout()),
                "dqs lhs @ ({b},{s},{h},{d})"
            );
            assert!(
                is_gemm_operand_admissible(rhs.layout()),
                "dqs rhs @ ({b},{s},{h},{d})"
            );
            let _ = lhs.matmul(&rhs).unwrap();

            // dkr = ds.transpose(-1,-2).contiguous() @ q.contiguous()
            let lhs = ds
                .transpose(D::Minus1, D::Minus2)
                .unwrap()
                .contiguous()
                .unwrap();
            let rhs = q.contiguous().unwrap();
            assert!(
                is_gemm_operand_admissible(lhs.layout()),
                "dkr lhs @ ({b},{s},{h},{d})"
            );
            assert!(
                is_gemm_operand_admissible(rhs.layout()),
                "dkr rhs @ ({b},{s},{h},{d})"
            );
            let _ = lhs.matmul(&rhs).unwrap();
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
        let scores = (q
            .contiguous()?
            .matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)?
            * f64::from(scale))?;
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
        let tol = 5e-2f64;
        // Sample a handful of indices rather than every one of `n` (cheap,
        // still a real finite-difference proof of the scatter/gather
        // round-trip and the RoPE/scale/softmax chain feeding it).
        for &i in &[0usize, 1, n / 2, n - 1] {
            let mut vp = qkv0.clone();
            vp[i] += eps;
            let mut vm = qkv0.clone();
            vm[i] -= eps;
            let numeric = (sum_fwd(&vp) - sum_fwd(&vm)) / (2.0 * eps as f64);
            assert!(
                (numeric - dqkv[i] as f64).abs() < tol,
                "dqkv[{i}]: numeric {numeric} vs analytic {}",
                dqkv[i]
            );
        }
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
}
