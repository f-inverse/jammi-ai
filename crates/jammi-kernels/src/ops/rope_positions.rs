//! Fused RoPE rotate-half on the FlashAttention-2-packed `[total, 3, h,
//! d]` `qkv` buffer — P6 Stage B B3-dense. `RopeFused` (`ops::rope`)
//! cannot take this layout directly: its `rope_dims` requires the axis
//! immediately before `hidden` to equal `period` (or `period == 1`), and
//! for a packed `qkv` that axis is `h` (the head axis), which is neither
//! — applying `RopeFused` here would silently read the wrong table row
//! for every row after the first (`ops::rope::rope_dims`'s own doc names
//! this exact family-D hazard). This op decodes `token = row / (3*h)`
//! (`row = flat_idx / d`) directly, so it walks the correct axis
//! regardless of `h`.
//!
//! ## Scope: dense only (`position = token % seq`)
//!
//! The P6 Stage B v5 contract's general mechanism is `positions[r] = r -
//! cu[seq(r)]` (a per-row lookup table, needed once a batch has real
//! padding and `cu_seqlens` is non-uniform). For the DENSE fast path
//! (`cu_seqlens` uniform, every sequence length `== seq`) that reduces to
//! the closed form `position = token % seq` — the SAME modulo
//! [`super::rope::RopeFused`] already uses, just walking a different
//! axis. This commit implements ONLY the dense closed form (one `seq:
//! usize` field, no positions `Tensor`/device array at all) — the
//! general table form is explicitly future work (the padded regime), not
//! implemented here; a future generalization would add a `positions`
//! argument alongside (or instead of) `seq` without changing this op's
//! per-element math, mirroring how `rope_positions.cu`'s shared
//! `rope_rotate` device function already factors the math out from the
//! indexing.
//!
//! ## V slot pass-through
//!
//! `qkv`'s slot 2 (V) is copied through unchanged — RoPE only ever
//! applies to Q/K (contract v5 §3.6) — because this op's OUTPUT is the
//! single tensor `flash_attention_varlen` consumes directly (no separate
//! V tensor to reassemble later): the packed buffer must remain a valid,
//! complete `qkv` after this op runs.
//!
//! ## `bwd`: the same sign-flip reuse `RopeFused` already established
//!
//! Forward is `out = x*cos + rotate_half(x)*sin` on the Q/K slots (module
//! doc of `ops::rope` derives why); the SAME algebra applies per-element
//! here (V is a pure identity map either direction), so `bwd` reuses this
//! exact `KernelOp` with `negate_sin` flipped — no permutation of
//! `grad_res` needed, exactly `RopeFused::bwd`'s own mechanism.
//!
//! ## Why `KernelOp` (`Copy`), not `StatefulKernelOp`
//!
//! Unlike `crate::ops::flash_attention`'s ops, this op holds no `Saved`
//! field and no device array construction data at all — dense-scope
//! `seq: usize` is plain `Copy` data, so this op fits the crate's
//! ordinary stateless-`KernelOp` family (`super::apply3`), the same as
//! `RopeFused` itself. It has no dependency on `crate::flash`'s FFI
//! boundary either (a pure elementwise transform, no CUTLASS), so it is
//! gated by this crate's plain `cuda` feature, not `flash-attn` — usable
//! (and testable) independent of whether the vendored FlashAttention-2
//! kernels are compiled in.
//!
//! ## Domain (family D)
//!
//! `qkv`: rank 4, `[total, 3, h, d]`, contiguous, `d` even, `total ==
//! qkv.dim(0)`. `cos`/`sin`: `[period, d]` (any leading dims of size 1
//! accepted, same convention as `ops::rope`), contiguous, `period == seq`
//! (or `period == 1`, a single shared table row). `seq == 0` is refused
//! (the modulo is undefined) UNLESS `total == 0` too (the fully
//! degenerate empty-batch case, which takes the `d == 0`-style empty
//! fast path). `d == 0` degenerates to an empty output, same as
//! `ops::rope`'s `hidden == 0` case.
//!
//! ## The ragged arm (M1a — varlen positions)
//!
//! [`PositionArm::Ragged`] is a SECOND, arm-selective domain living on the
//! SAME op/kernel: `position` degenerates to the row index itself (the
//! CUDA kernel's `token % seq` closed form with `seq` set to `total`
//! reduces to `token % total == token` for every `token < total`), so a
//! caller who has ALREADY gathered `cos`/`sin` into a per-row `[total, d]`
//! table gets per-row correctness "for free" from the SAME kernel math —
//! no `.cu` edit, no new device function. The ENTRY POINT that does that
//! gathering is [`rope_positions_fused_ragged`] (this module) and
//! `flash_attention::flash_attention_varlen_with_rope_ragged` (a code
//! span, not a doc link: that item lives in a sibling module gated behind
//! the `flash-attn` feature, so a link to it is unresolvable — and
//! therefore silently unrendered, never even checked, per rustdoc's
//! default of only checking `pub`-reachable item docs — whenever this
//! crate's docs build without that feature, which is `docs.yml`'s own
//! `cargo doc --workspace --no-deps` invocation): both take the BASE
//! `cos`/`sin` table plus `lengths: &[usize]` and derive the per-row
//! `positions` (via [`ragged_positions_from_lengths`]) and the gathered
//! `[total, d]` table (via [`gather_ragged_tables`]'s `Tensor::index_select`
//! — a stock op on both CPU and CUDA, so the CPU arm stays "identical by
//! construction") from that ONE `lengths` slice.
//!
//! **The "structurally unconstructible" fence is scoped to the FLASH
//! entry, not this module's own.** `flash_attention_varlen_with_rope_ragged`
//! ALSO derives its own `cu_seqlens` from that SAME `lengths` internally
//! (never a caller-supplied one), so for THAT function a
//! table/segmentation mismatch really is structurally unconstructible:
//! there is exactly one source of truth for both artifacts. THIS module's
//! own [`rope_positions_fused_ragged`] emits no `cu_seqlens` at all — it
//! only produces a rotated `qkv` `Tensor` — so the fence does NOT extend
//! to a caller who composes it by hand: nothing stops a caller from
//! rotating with `rope_positions_fused_ragged(qkv, cos, sin, &lengths_a,
//! ..)` and then feeding the result to the crate's plain
//! `flash_attention_varlen` (the non-fused entry) alongside an
//! INDEPENDENTLY-built `CuSeqlens` derived from a DIFFERENT `lengths_b` —
//! that composition reconstructs the exact table/segmentation mismatch
//! hazard this module's ragged arm exists to close, one level up, outside
//! this crate's ability to detect it (this op has no visibility into
//! whatever `CuSeqlens` a caller builds afterwards). Callers who need the
//! closed fence should use `flash_attention_varlen_with_rope_ragged`
//! directly rather than composing `rope_positions_fused_ragged` with a
//! separately-built `CuSeqlens`.
//!
//! No `crate::flash::CuSeqlens` type is built or threaded through this
//! module — `lengths` arrives as a plain `&[usize]` (this module's
//! documented independence from `crate::flash`'s FFI boundary, stated
//! above, is preserved even by the ragged arm; a caller that ALSO needs a
//! real `CuSeqlens` — `flash_attention_varlen_with_rope_ragged` is the one
//! that does — builds its own, independently, from the SAME `lengths`
//! slice, never derived FROM this module's `positions`).
//!
//! The ragged arm's guards ([`rope_positions_dims`]) are DELIBERATELY
//! tighter than the dense arm's, replacing (never merely supplementing)
//! `crate::flash::CuSeqlens::check_dense_uniform` for this arm: `total`
//! (from `qkv`) must EXACTLY equal `seq` (the ragged constructors store
//! the gathered table's own row-total there, reusing the same field the
//! dense arm calls `seq`) rather than merely being a multiple of it; the
//! table must cover EXACTLY `total` rows (`period == seq`) — the dense
//! arm's `period == 1` shared-row convenience does not apply (a single
//! shared row cannot mean anything once `position` is the row index); and
//! `total == 0` is a REFUSAL, never the dense arm's empty-batch
//! acceptance (a deliberate delta — an empty ragged batch is a caller
//! bug, e.g. `lengths == []`, not a degenerate-but-meaningful shape here).
//! Every `lengths` entry must ALSO be `> 0`: a zero-length segment is
//! refused (see [`ragged_positions_from_lengths`]'s own doc) for the SAME
//! reason `crate::flash`'s `cu_seqlens_from_lengths` already refuses it on
//! the flash side (`flash/mod.rs`'s own comment: "a zero-length sequence
//! has no rows to attend from or to ... refused rather than silently
//! producing an empty ... slice of the batch") — ONE `lengths` contract,
//! shared by both entries, rather than the flash entry being stricter than
//! this module's own for the same input shape.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp3, Error, Layout, Result, Shape, Tensor};
use half::bf16;

const OP: &str = "rope_positions_fused";

/// Which domain [`RopePositionsFused::seq`] means — see the module doc's
/// "The ragged arm" section. `Copy`: this discriminant lives INSIDE
/// [`RopePositionsFused`], which must stay `Copy` (the [`super::KernelOp`]
/// bound) — a plain fieldless enum costs nothing towards that.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PositionArm {
    /// `position = token % seq`. `period == 1` (shared row) or
    /// `period == seq` accepted; `total == 0` accepted when `seq == 0`.
    Dense,
    /// `position = token` (the row index) — `seq` holds `total`, the
    /// caller-gathered table's own row count. `period == seq` (== total)
    /// is required EXACTLY (no shared-row allowance); `total == 0` is a
    /// REFUSAL. See the module doc's "The ragged arm" section.
    Ragged,
}

/// Fused RoPE rotate-half on the packed `[total, 3, h, d]` buffer. See
/// the module doc.
#[derive(Debug, Clone, Copy)]
pub struct RopePositionsFused {
    /// Dense sequence length (`total = batch * seq`), OR — when `arm ==
    /// Ragged` — the gathered table's own row-total (`total`). Every
    /// token's position is `token % seq`, which degenerates to the row
    /// index itself in the ragged arm (`seq == total`). See the module
    /// doc's scope note and "The ragged arm" section.
    pub seq: usize,
    /// Same convention as [`super::rope::RopeFused::negate_sin`]: `true`
    /// is how `bwd` reuses this forward kernel to compute `dx`.
    pub negate_sin: bool,
    arm: PositionArm,
}

impl RopePositionsFused {
    pub fn new(seq: usize, negate_sin: bool) -> Self {
        Self::new_with_arm(seq, negate_sin, PositionArm::Dense)
    }

    /// `pub(crate)`: the only sanctioned way to construct the RAGGED arm.
    /// `seq_or_total` is the gathered table's own row-total when `arm ==
    /// Ragged` — callers outside this module never construct `Ragged`
    /// directly; they go through [`rope_positions_fused_ragged`] (this
    /// module) or `flash_attention::flash_attention_varlen_with_rope_ragged`,
    /// both of which derive it from `gather_ragged_tables`. This
    /// constructor itself is reused by BOTH of those AND by
    /// `flash_attention::FlashVarlenAttentionFusedRope`'s own `bwd`
    /// recompute (which must reuse the SAME arm its `fwd` used).
    pub(crate) fn new_with_arm(seq_or_total: usize, negate_sin: bool, arm: PositionArm) -> Self {
        Self {
            seq: seq_or_total,
            negate_sin,
            arm,
        }
    }
}

impl super::sealed::Sealed for RopePositionsFused {}

/// Validates and derives `(total, h, d)` shared by every arm (CPU, CUDA
/// glue) of this op. `pub(crate)`: `crate::cuda::rope_positions` imports
/// this exact check rather than re-deriving it (the same "shared, not
/// duplicated" choice `ops::rope::rope_dims` documents for itself).
pub(crate) fn rope_positions_dims(
    l_qkv: &Layout,
    l_cos: &Layout,
    l_sin: &Layout,
    seq: usize,
    arm: PositionArm,
) -> Result<(usize, usize, usize)> {
    let dims = l_qkv.dims();
    if dims.len() != 4 || dims[1] != 3 {
        return Err(Error::Msg(format!(
            "{OP}: qkv must be rank-4 [total, 3, h, d], got {dims:?}"
        )));
    }
    let total = dims[0];
    let h = dims[2];
    let d = dims[3];

    if l_cos.dims() != l_sin.dims() {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l_cos.shape().clone(),
            rhs: l_sin.shape().clone(),
            op: OP,
        });
    }
    let Some(&cos_last) = l_cos.dims().last() else {
        return Err(Error::Msg(format!(
            "{OP}: cos/sin must have rank >= 1 to define a last (head_dim) dimension"
        )));
    };
    if cos_last != d {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: l_qkv.shape().clone(),
            rhs: l_cos.shape().clone(),
            op: OP,
        });
    }

    // Ragged-arm structural fence (module doc's "The ragged arm" section):
    // ALL THREE ragged checks below run BEFORE the `d == 0` fast path
    // further down, uniformly -- none can be bypassed by a degenerate
    // head_dim. `total == 0` is refused unconditionally (a deliberate
    // delta from the dense arm's own empty-batch acceptance); `total !=
    // seq` catches a caller who built `qkv` and the gathered table from
    // DIFFERENT `lengths`; the mis-sized-table check (`cos_elems != seq *
    // d`) is the "table must cover EXACTLY `total` rows" guard, written
    // as a MULTIPLICATION rather than the `period = cos_elems / d`
    // division the dense arm's own equivalent check uses further down --
    // deliberately, so it stays well-defined (and reachable) even when
    // `d == 0`, where a division would be undefined; `total * d` cannot
    // overflow `usize` for any shape this op's own rank-4 `qkv` domain
    // check above already bounds (`total`/`d` are each an existing
    // tensor's own dimension, never independently attacker-controlled).
    if arm == PositionArm::Ragged {
        if total == 0 {
            return Err(Error::Msg(format!(
                "{OP}: ragged arm refuses total=0 (a deliberate delta from the dense arm's \
                 empty-batch acceptance) -- call the ragged entry point with a nonempty \
                 `lengths` covering at least one token rather than relying on this op to \
                 silently degenerate to an empty forward"
            )));
        }
        if total != seq {
            return Err(Error::Msg(format!(
                "{OP}: ragged arm requires qkv's total rows (={total}) to exactly equal the \
                 gathered table's row-total (={seq}) -- both must be derived from the SAME \
                 `lengths`, or qkv and the position table disagree on segmentation"
            )));
        }
        let cos_elems = l_cos.shape().elem_count();
        if cos_elems != seq * d {
            return Err(Error::Msg(format!(
                "{OP}: ragged arm requires the pre-gathered table to cover EXACTLY total={seq} \
                 rows (cos/sin element count {cos_elems}, expected {seq} * head_dim={d} = \
                 {expected}) -- position degenerates to the row index in this arm, so the \
                 dense arm's period=1 shared-row allowance does not apply here",
                expected = seq * d,
            )));
        }
    }

    if d == 0 {
        return Ok((total, h, 0));
    }
    if !d.is_multiple_of(2) {
        return Err(Error::Msg(format!(
            "{OP}: head_dim={d} must be even -- rotate-half splits it into two equal halves"
        )));
    }
    let cos_elems = l_cos.shape().elem_count();
    if cos_elems == 0 || !cos_elems.is_multiple_of(d) {
        return Err(Error::Msg(format!(
            "{OP}: cos/sin element count {cos_elems} is not a positive multiple of head_dim={d}"
        )));
    }
    let period = cos_elems / d;
    match arm {
        PositionArm::Dense => {
            if period != 1 && period != seq {
                return Err(Error::Msg(format!(
                    "{OP}: cos/sin table covers {period} positions, expected exactly seq={seq} \
                     (or a single shared row, period=1) -- a table covering a different span \
                     would silently index the wrong position for a dense forward"
                )));
            }
            if total > 0 && seq == 0 {
                return Err(Error::Msg(format!(
                    "{OP}: seq=0 with a nonempty qkv (total={total}) -- position = token % seq is \
                     undefined"
                )));
            }
            // Dense-only scope (module doc): `total` MUST be `batch * seq`
            // for some integer `batch`. `token % seq` is arithmetically
            // well-defined even when `total` is not a multiple of `seq`,
            // but it is then SEMANTICALLY wrong -- the tail rows of the
            // last, incomplete "batch" would wrap into positions that
            // belong to a batch element that never existed, silently
            // misindexing rather than refusing a shape outside this op's
            // domain. Mirrors `rope_dims`'s own "silently misindexed" guard.
            if seq > 0 && !total.is_multiple_of(seq) {
                return Err(Error::Msg(format!(
                    "{OP}: total={total} is not a multiple of seq={seq} -- this op's DENSE \
                     scope requires total == batch * seq for some integer batch, or \
                     `position = token % seq` silently wraps into a batch element that does \
                     not exist"
                )));
            }
        }
        PositionArm::Ragged => {
            // Every ragged check (`total == 0`, `total == seq`, and the
            // mis-sized-table check via `cos_elems == seq * d`) was
            // ALREADY enforced above, uniformly BEFORE the `d == 0` fast
            // path -- nothing left to check here; `period` (computed
            // above for the dense arm's own use) is redundant with the
            // multiplication check already performed, by construction.
        }
    }
    Ok((total, h, d))
}

/// Pure host arithmetic: `positions[r] = r - cu[seg(r)]`, `cu` the
/// exclusive prefix sum of `lengths` -- no `crate::flash::CuSeqlens`
/// (device array) is built here, see the module doc's "The ragged arm"
/// section. A caller that ALSO needs a real `CuSeqlens` builds its own,
/// independently, from the SAME `lengths` slice this function reads —
/// one shared input, two independent derivations, never one built FROM
/// the other.
///
/// Every `len` must be `> 0`: a zero-length segment is REFUSED, not a
/// legitimate degenerate case -- UNIFIED with `crate::flash`'s own
/// `cu_seqlens_from_lengths`, which already refuses this for the exact
/// same reason (`flash/mod.rs`: "a zero-length sequence has no rows to
/// attend from or to ... refused rather than silently producing an empty
/// ... slice of the batch"). An earlier version of this function accepted
/// `len == 0` (reasoning it "contributes zero rows, a legitimate
/// degenerate no-tokens-for-this-batch-element case") while
/// `flash_attention_varlen_with_rope_ragged`'s OWN `CuSeqlens::from_lengths`
/// call refused the identical `lengths` slice -- two different answers to
/// the SAME question depending on which entry point a caller reached
/// (`rope_positions_fused_ragged` silently succeeded on `lengths=[3, 0,
/// 5]`; the flash entry on the exact same slice errored). Since this
/// function is the ONE place both entries derive `positions` from
/// `lengths`, refusing here makes `rope_positions_fused_ragged` and
/// `flash_attention_varlen_with_rope_ragged` agree on ONE `lengths`
/// contract rather than two silently different ones. `lengths.is_empty()`
/// (the wholly-empty-batch case) is NOT refused here -- it correctly
/// yields `total == 0`, which the ragged arm's own guard in
/// `rope_positions_dims` refuses at the op boundary (a distinct check,
/// see the module doc's "The ragged arm" section), not here.
pub(crate) fn ragged_positions_from_lengths(lengths: &[usize]) -> Result<Vec<u32>> {
    let mut positions = Vec::new();
    for (i, &len) in lengths.iter().enumerate() {
        if len == 0 {
            return Err(Error::Msg(format!(
                "{OP}: segment {i} has length 0 -- every segment must be non-empty (a \
                 zero-length segment has no rows to rotate; unified with \
                 crate::flash::cu_seqlens_from_lengths's identical refusal, see this fn's own \
                 doc)"
            )));
        }
        for r in 0..len {
            let p = u32::try_from(r).map_err(|_| {
                Error::Msg(format!(
                    "{OP}: segment length {len} exceeds this op's u32 row-position range"
                ))
            })?;
            positions.push(p);
        }
    }
    Ok(positions)
}

/// Gathers `cos_base`/`sin_base` (the SAME base-table convention
/// [`RopePositionsFused::new`]'s dense arm accepts: `[period_base, d]`,
/// any leading dims of size 1) into per-row `[total, d]` tables via
/// `Tensor::index_select` — a stock tensor op on both CPU and CUDA (the
/// module doc's "CPU arm identical by construction" premise) — using
/// [`ragged_positions_from_lengths`]'s own output. Returns `(total,
/// cos_r, sin_r)`. This is the ONE place both `positions` and (for a
/// flash-side caller) `cu_seqlens` are derived FROM `lengths` — see the
/// module doc's "The ragged arm" section for why that makes a
/// table/segmentation mismatch structurally unconstructible.
pub(crate) fn gather_ragged_tables(
    cos_base: &Tensor,
    sin_base: &Tensor,
    lengths: &[usize],
) -> Result<(usize, Tensor, Tensor)> {
    let Some(&d) = cos_base.dims().last() else {
        return Err(Error::Msg(format!(
            "{OP}: cos_base must have rank >= 1 to define a last (head_dim) dimension"
        )));
    };
    if d == 0 {
        return Err(Error::Msg(format!(
            "{OP}: cos_base's last dimension (head_dim) is 0 -- the ragged gather has no \
             columns to index"
        )));
    }
    if cos_base.dims() != sin_base.dims() {
        return Err(Error::ShapeMismatchBinaryOp {
            lhs: cos_base.shape().clone(),
            rhs: sin_base.shape().clone(),
            op: OP,
        });
    }
    let period_base = cos_base.elem_count() / d;
    let positions = ragged_positions_from_lengths(lengths)?;
    let total = positions.len();
    if let Some(&max_pos) = positions.iter().max() {
        if (max_pos as usize) >= period_base {
            return Err(Error::Msg(format!(
                "{OP}: a derived position ({max_pos}) exceeds cos_base's own period \
                 ({period_base}) -- the base table does not cover every position `lengths` \
                 needs; the longest segment in `lengths` must be <= {period_base}"
            )));
        }
    }
    let device = cos_base.device();
    let idx = Tensor::from_vec(positions, total, device)?;
    let cos_2d = cos_base.reshape((period_base, d))?;
    let sin_2d = sin_base.reshape((period_base, d))?;
    let cos_r = cos_2d.index_select(&idx, 0)?.contiguous()?;
    let sin_r = sin_2d.index_select(&idx, 0)?.contiguous()?;
    Ok((total, cos_r, sin_r))
}

/// Ragged entry point (M1a — varlen positions): rotates Q/K in a packed
/// `[total, 3, h, d]` `qkv` buffer whose `total` rows are the
/// CONCATENATION of `lengths.len()` variable-length segments (no
/// padding). See the module doc's "The ragged arm" section for the full
/// contract; this function is the CPU/eager (device-generic — it never
/// touches `crate::flash`) entry, mirrored by
/// `flash_attention::flash_attention_varlen_with_rope_ragged` for the
/// fused-flash arm.
pub fn rope_positions_fused_ragged(
    qkv: &Tensor,
    cos_base: &Tensor,
    sin_base: &Tensor,
    lengths: &[usize],
    negate_sin: bool,
) -> Result<Tensor> {
    let (total, cos_r, sin_r) = gather_ragged_tables(cos_base, sin_base, lengths)?;
    let op = RopePositionsFused::new_with_arm(total, negate_sin, PositionArm::Ragged);
    super::apply3(qkv, &cos_r, &sin_r, op)
}

#[allow(clippy::too_many_arguments)]
fn rope_positions_fwd_f32(
    qkv: &[f32],
    cos: &[f32],
    sin: &[f32],
    total: usize,
    h: usize,
    d: usize,
    seq: usize,
    sign: f32,
) -> Vec<f32> {
    let half = d / 2;
    let mut out = vec![0.0f32; total * 3 * h * d];
    for token in 0..total {
        let seq_idx = if seq == 0 { 0 } else { token % seq };
        let table_base = seq_idx * d;
        for slot in 0..3usize {
            for h_idx in 0..h {
                let row_base = ((token * 3 + slot) * h + h_idx) * d;
                if slot == 2 {
                    out[row_base..row_base + d].copy_from_slice(&qkv[row_base..row_base + d]);
                    continue;
                }
                for c in 0..d {
                    let xv = qkv[row_base + c];
                    let rh = if c < half {
                        -qkv[row_base + c + half]
                    } else {
                        qkv[row_base + c - half]
                    };
                    let cc = cos[table_base + c];
                    let ss = sin[table_base + c];
                    out[row_base + c] = xv * cc + rh * ss * sign;
                }
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn rope_positions_fwd_bf16(
    qkv: &[bf16],
    cos: &[bf16],
    sin: &[bf16],
    total: usize,
    h: usize,
    d: usize,
    seq: usize,
    sign: f32,
) -> Vec<bf16> {
    let half = d / 2;
    let mut out = vec![bf16::ZERO; total * 3 * h * d];
    for token in 0..total {
        let seq_idx = if seq == 0 { 0 } else { token % seq };
        let table_base = seq_idx * d;
        for slot in 0..3usize {
            for h_idx in 0..h {
                let row_base = ((token * 3 + slot) * h + h_idx) * d;
                if slot == 2 {
                    out[row_base..row_base + d].copy_from_slice(&qkv[row_base..row_base + d]);
                    continue;
                }
                for c in 0..d {
                    let xv = f32::from(qkv[row_base + c]);
                    let rh = if c < half {
                        -f32::from(qkv[row_base + c + half])
                    } else {
                        f32::from(qkv[row_base + c - half])
                    };
                    let cc = f32::from(cos[table_base + c]);
                    let ss = f32::from(sin[table_base + c]);
                    out[row_base + c] = bf16::from_f32(xv * cc + rh * ss * sign);
                }
            }
        }
    }
    out
}

impl CustomOp3 for RopePositionsFused {
    fn name(&self) -> &'static str {
        OP
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
        let (total, h, d) = rope_positions_dims(l1, l2, l3, self.seq, self.arm)?;
        if s1.dtype() != s2.dtype() || s1.dtype() != s3.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: OP,
            });
        }
        if d == 0 {
            return super::empty_like(s1, s1, l1, OP);
        }
        let (x1, x2) = l1
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: OP })?;
        let (c1, c2) = l2
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: OP })?;
        let (s_1, s_2) = l3
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: OP })?;
        let sign = if self.negate_sin { -1.0 } else { 1.0 };
        match (s1, s2, s3) {
            (CpuStorage::F32(x), CpuStorage::F32(cos), CpuStorage::F32(sin)) => {
                let out = rope_positions_fwd_f32(
                    &x[x1..x2],
                    &cos[c1..c2],
                    &sin[s_1..s_2],
                    total,
                    h,
                    d,
                    self.seq,
                    sign,
                );
                Ok((CpuStorage::F32(out), l1.shape().clone()))
            }
            (CpuStorage::BF16(x), CpuStorage::BF16(cos), CpuStorage::BF16(sin)) => {
                let out = rope_positions_fwd_bf16(
                    &x[x1..x2],
                    &cos[c1..c2],
                    &sin[s_1..s_2],
                    total,
                    h,
                    d,
                    self.seq,
                    sign,
                );
                Ok((CpuStorage::BF16(out), l1.shape().clone()))
            }
            (s1, _, _) => Err(Error::UnsupportedDTypeForOp(s1.dtype(), OP)),
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
        crate::cuda::rope_positions::cuda_fwd(
            self.seq,
            self.negate_sin,
            self.arm,
            s1,
            l1,
            s2,
            l2,
            s3,
            l3,
        )
    }

    /// `dx` (the gradient wrt the packed, pre-rotation `qkv`) reuses THIS
    /// op with `sin` negated -- `RopeFused::bwd`'s exact mechanism,
    /// applies unchanged here (V's identity map is its own transpose).
    /// `dcos`/`dsin`: unlike `RopeFused`, this op does NOT implement a
    /// real table gradient -- `cos`/`sin` are non-tracked leaf tables in
    /// every call site this op ships behind (the SAME premise
    /// `RopeFused`'s module doc states for itself), and composing one for
    /// this op's packed-buffer-with-V-passthrough indexing is real,
    /// currently-unexercised work. Rather than silently return `None` for
    /// a hypothetical future trainable table (the exact landmine
    /// `LayerNormFused`'s doc warns a hardcoded `false`/`None` would be),
    /// a caller that DOES pass a TRACKED `cos`/`sin` gets a typed error,
    /// not a silently-missing gradient. Gated on `track_op()`, NOT
    /// `is_variable()` alone: the ragged arm's own entry point
    /// (`rope_positions_fused_ragged`) gathers a per-row table via
    /// `Tensor::index_select` BEFORE calling this op, so a caller who
    /// passes a `Var`-backed BASE table gets a `cos`/`sin` argument here
    /// that is itself `is_variable() == false` (it is the `index_select`
    /// RESULT, not the `Var`) but `track_op() == true` (it carries an
    /// `Op`, and candle's own `sorted_nodes` walk, `backprop.rs`, still
    /// expects a gradient entry for it once its ancestry reaches a
    /// `Var`) -- an `is_variable()`-only gate here would silently return
    /// `Ok` and let `apply3` return `(Some(dx), None, None)`, panicking
    /// downstream at `backprop.rs:175` ("grad not populated") instead of
    /// this typed refusal. The SAME predicate-hole class
    /// `low_rank_residual_linear.rs`'s and `jammi-lora`'s
    /// `frozen_weight_gate` already fixed for their own `w`/`weight`
    /// slots; see `crate::ops::rope_positions`'s own test
    /// `rope_positions_fused_ragged_refuses_a_var_backed_base_table_not_a_panic`
    /// for the live probe.
    fn bwd(
        &self,
        arg1: &Tensor,
        arg2: &Tensor,
        arg3: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let dx = super::apply3(
            grad_res,
            arg2,
            arg3,
            RopePositionsFused::new_with_arm(self.seq, !self.negate_sin, self.arm),
        )?;
        if arg2.track_op() || arg3.track_op() {
            return Err(Error::Msg(format!(
                "{OP}: cos/sin gradient is not implemented -- every call site this op ships \
                 behind treats cos/sin as non-tracked leaf tables (see the module doc); a \
                 caller that made them trainable (a Var, or any tensor carrying an Op that \
                 traces back to one -- checked via track_op(), not is_variable() alone) would \
                 need a real dcos/dsin implementation, not the silently-None gradient this \
                 error replaces"
            )));
        }
        let _ = arg1; // domain-only; `dx` above is the whole gradient wrt qkv.
        Ok((Some(dx), None, None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::RopeFused;
    use candle_core::{DType, Device};

    fn fused(
        seq: usize,
        negate_sin: bool,
        qkv: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        super::super::apply3(qkv, cos, sin, RopePositionsFused::new(seq, negate_sin))
    }

    /// A real (non-trivial-angle) rotary table, `[period, hidden]`,
    /// duplicated-column convention (`RopeFused`'s own module doc,
    /// re-required here since this op consumes the identical table
    /// shape).
    fn rope_table(period: usize, hidden: usize, theta_base: f64) -> (Tensor, Tensor) {
        let half = hidden / 2;
        let mut cos = vec![0f32; period * hidden];
        let mut sin = vec![0f32; period * hidden];
        for pos in 0..period {
            for i in 0..half {
                let theta = theta_base.powf(-2.0 * (i as f64) / (hidden as f64));
                let angle = pos as f64 * theta;
                let (s, c) = angle.sin_cos();
                cos[pos * hidden + i] = c as f32;
                cos[pos * hidden + i + half] = c as f32;
                sin[pos * hidden + i] = s as f32;
                sin[pos * hidden + i + half] = s as f32;
            }
        }
        let device = Device::Cpu;
        (
            Tensor::from_vec(cos, (period, hidden), &device).unwrap(),
            Tensor::from_vec(sin, (period, hidden), &device).unwrap(),
        )
    }

    /// Packs a `[b, h, s, d]` tensor's own values into slot `slot` of a
    /// FRESH `[b*s, 3, h, d]` `qkv` tensor (the other two slots filled
    /// from `filler`, a distinct tensor of the same shape, so a
    /// pass-through bug on the wrong slot cannot hide behind identical
    /// data). Mirrors the real call site's `qkv.reshape((b*s, 3, h,
    /// d))`-after-`[b, s, 3, h, d]`-view shape, but built explicitly here
    /// (the real call site's `qkv` is ALREADY `[b, s, 3*h*d]` from one
    /// `Wqkv` GEMM; this test constructs the THREE logical tensors a
    /// from-scratch oracle needs instead).
    fn pack_bhsd_into_qkv(x_bhsd: &Tensor, filler: &Tensor, slot: usize) -> Tensor {
        let (b, h, s, d) = x_bhsd.dims4().unwrap();
        let x_bshd = x_bhsd.transpose(1, 2).unwrap().contiguous().unwrap(); // [b,s,h,d]
        let filler_bshd = filler.transpose(1, 2).unwrap().contiguous().unwrap();
        let mut slots = Vec::with_capacity(3);
        for i in 0..3 {
            let src = if i == slot { &x_bshd } else { &filler_bshd };
            slots.push(src.reshape((b * s, 1, h, d)).unwrap());
        }
        Tensor::cat(&slots, 1).unwrap()
    }

    fn unpack_qkv_slot(qkv: &Tensor, slot: usize, b: usize, s: usize) -> Tensor {
        let (total, _, h, d) = qkv.dims4().unwrap();
        assert_eq!(total, b * s);
        qkv.narrow(1, slot, 1)
            .unwrap()
            .reshape((b, s, h, d))
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap()
    }

    fn to_bits(t: &Tensor) -> Vec<u32> {
        t.flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    /// THE oracle (P6 Stage B B3-dense, contract v5 §3.6): `rope_positions`
    /// on the packed layout is bit-identical to `RopeFused` applied to the
    /// SAME data in `[b, h, s, d]` form (the block arm's own operand
    /// shape, `gather_bhsd`'s target), for a real (non-trivial-angle)
    /// table, non-trivial data, b1 AND b8. RED control: sign flipped (the
    /// `negate_sin` this op's own `bwd` reuses) must NOT match.
    fn bit_identity_case(b: usize, h: usize, s: usize, d: usize) {
        bit_identity_case_with_red_control(b, h, s, d, true)
    }

    fn bit_identity_case_with_red_control(
        b: usize,
        h: usize,
        s: usize,
        d: usize,
        check_red_control: bool,
    ) {
        let device = Device::Cpu;
        let n = b * h * s * d;
        // Non-trivial, non-symmetric data: distinct per element, includes
        // negative values (`sin(k)`-derived, never all-zero or constant).
        let xv: Vec<f32> = (0..n).map(|k| (k as f32 * 0.037).sin() * 3.0).collect();
        let x_bhsd = Tensor::from_vec(xv, (b, h, s, d), &device).unwrap();
        let fv: Vec<f32> = (0..n)
            .map(|k| (k as f32 * 0.091 + 1.0).cos() * 2.0)
            .collect();
        let filler_bhsd = Tensor::from_vec(fv, (b, h, s, d), &device).unwrap();
        let (cos, sin) = rope_table(s, d, 10_000.0);

        // Reference: RopeFused directly on [b,h,s,d] (the block arm's
        // operand shape).
        let reference = super::super::apply3(&x_bhsd, &cos, &sin, RopeFused::new(false)).unwrap();

        // rope_positions on the packed [b*s,3,h,d] buffer, slot 0 = q.
        let qkv = pack_bhsd_into_qkv(&x_bhsd, &filler_bhsd, 0);
        let out = fused(s, false, &qkv, &cos, &sin).unwrap();
        let got_q = unpack_qkv_slot(&out, 0, b, s);
        assert_eq!(
            to_bits(&got_q),
            to_bits(&reference),
            "rope_positions on slot 0 (q) must be bit-identical to RopeFused on [b,h,s,d], \
             b={b} h={h} s={s} d={d}"
        );

        // V slot (2) must pass through byte-identical to the filler data
        // packed into it (RoPE never touches V) -- `qkv`'s OWN slot 2
        // (not the output) is the expected value, read before the op runs.
        let got_v = unpack_qkv_slot(&out, 2, b, s);
        let expected_v = unpack_qkv_slot(&qkv, 2, b, s);
        assert_eq!(
            to_bits(&got_v),
            to_bits(&expected_v),
            "rope_positions must pass V (slot 2) through unchanged, b={b} h={h} s={s} d={d}"
        );

        // RED control: negate_sin=true must NOT reproduce the (positive-
        // sign) reference on non-trivial data. Skipped at `s=1`: with a
        // single position, `angle = pos*theta = 0` for every `i`, so
        // `sin` is identically zero and flipping its sign is genuinely a
        // no-op -- not a weakened control, a degenerate domain where the
        // control itself is vacuous (the boundary case this test exists
        // to cover on its OWN, `bit_identical_single_position_s_one`,
        // still gets the full bit-identity check above).
        if check_red_control {
            let out_negated = fused(s, true, &qkv, &cos, &sin).unwrap();
            let got_q_negated = unpack_qkv_slot(&out_negated, 0, b, s);
            assert_ne!(
                to_bits(&got_q_negated),
                to_bits(&reference),
                "RED control: sign-flipped rope_positions must NOT match the reference, \
                 b={b} h={h} s={s} d={d}"
            );
        }

        // Slot 1 (K) must ALSO be bit-identical to `RopeFused` on the SAME
        // data -- RoPE applies to Q *and* K (contract v5 §3.6), and the
        // slot-0 check above says nothing about slot 1: a defect that
        // rotates only slot 0 (leaving K a pass-through, e.g. a
        // `slot == 2` condition mutated to `slot >= 1`) would sail through
        // every assertion above while silently returning an unrotated K to
        // every downstream attention call. Packed independently (`x_bhsd`
        // now at slot 1, `filler_bhsd` at slots 0 and 2) so this cannot
        // hide behind slot 0's already-rotated data.
        let qkv_k = pack_bhsd_into_qkv(&x_bhsd, &filler_bhsd, 1);
        let out_k = fused(s, false, &qkv_k, &cos, &sin).unwrap();
        let got_k = unpack_qkv_slot(&out_k, 1, b, s);
        assert_eq!(
            to_bits(&got_k),
            to_bits(&reference),
            "rope_positions on slot 1 (k) must be bit-identical to RopeFused on [b,h,s,d], \
             b={b} h={h} s={s} d={d}"
        );
        if check_red_control {
            let out_k_negated = fused(s, true, &qkv_k, &cos, &sin).unwrap();
            let got_k_negated = unpack_qkv_slot(&out_k_negated, 1, b, s);
            assert_ne!(
                to_bits(&got_k_negated),
                to_bits(&reference),
                "RED control: sign-flipped rope_positions on slot 1 (k) must NOT match the \
                 reference, b={b} h={h} s={s} d={d}"
            );
        }
    }

    #[test]
    fn bit_identical_to_rope_fused_on_bhsd_b1_s_small() {
        bit_identity_case(1, 2, 5, 4);
    }

    #[test]
    fn bit_identical_to_rope_fused_on_bhsd_b8_s_small() {
        bit_identity_case(8, 3, 7, 6);
    }

    #[test]
    fn bit_identical_to_rope_fused_on_bhsd_head_dim_matches_production() {
        // head_dim=64 (ModernBERT-large's real head_dim), a smaller
        // (b, s) so the CPU test stays fast — the pod leg covers full
        // production (b, s) at this head_dim in bf16.
        bit_identity_case(2, 4, 9, 64);
    }

    /// Direct assertion of the module doc's `bwd` claim ("the same
    /// sign-flip reuse `RopeFused` already established"): `bwd` for BOTH
    /// [`RopePositionsFused`] (`rope_positions.rs:350-377`) and
    /// [`super::super::rope::RopeFused`] (`rope.rs:348-368`) delegate to
    /// their OWN forward with `negate_sin` flipped — that makes bit-
    /// identity of the two ops' `bwd` outputs a CONSEQUENCE of the
    /// already-proven forward bit-identity (`bit_identity_case`, above)
    /// PLUS identical cotangents, but it was not itself asserted anywhere
    /// before this test. Drives BOTH ops through candle's real
    /// `Tensor::backward()` (not a hand-replication of the `negate_sin`
    /// mechanism) so this exercises the actual `CustomOp3::bwd` trait
    /// method on each op, on the SAME non-uniform cotangent (a real
    /// backward pass through `sum_all` gives a uniform all-ones cotangent,
    /// which would not distinguish a `bwd` that silently returns the
    /// WRONG per-element gradient shape/permutation from a correct one as
    /// sharply as a non-uniform weight does).
    fn bwd_bit_identity_case(b: usize, h: usize, s: usize, d: usize) {
        use candle_core::Var;
        let device = Device::Cpu;
        let n = b * h * s * d;
        let xv: Vec<f32> = (0..n).map(|k| (k as f32 * 0.037).sin() * 3.0).collect();
        let x_bhsd = Tensor::from_vec(xv, (b, h, s, d), &device).unwrap();
        let fv: Vec<f32> = (0..n)
            .map(|k| (k as f32 * 0.091 + 1.0).cos() * 2.0)
            .collect();
        let filler_bhsd = Tensor::from_vec(fv, (b, h, s, d), &device).unwrap();
        // The cotangent: distinct, non-uniform per-element weights (never
        // all-ones like a bare `sum_all().backward()` would produce).
        let wv: Vec<f32> = (0..n).map(|k| (k as f32 * 0.017 + 0.5).cos()).collect();
        let weight_bhsd = Tensor::from_vec(wv, (b, h, s, d), &device).unwrap();
        let zeros_bhsd = Tensor::zeros((b, h, s, d), DType::F32, &device).unwrap();
        let (cos, sin) = rope_table(s, d, 10_000.0);

        // Reference: RopeFused's own real backward, on [b,h,s,d].
        let x_var_ref = Var::from_tensor(&x_bhsd).unwrap();
        let y_ref =
            super::super::apply3(x_var_ref.as_tensor(), &cos, &sin, RopeFused::new(false)).unwrap();
        let loss_ref = (&y_ref * &weight_bhsd).unwrap().sum_all().unwrap();
        let grads_ref = loss_ref.backward().unwrap();
        let dx_ref = grads_ref
            .get(x_var_ref.as_tensor())
            .expect("RopeFused: x must have a gradient");

        for slot in [0usize, 1usize] {
            let qkv = pack_bhsd_into_qkv(&x_bhsd, &filler_bhsd, slot);
            let qkv_var = Var::from_tensor(&qkv).unwrap();
            let weight_packed = pack_bhsd_into_qkv(&weight_bhsd, &zeros_bhsd, slot);
            let y = fused(s, false, qkv_var.as_tensor(), &cos, &sin).unwrap();
            let loss = (&y * &weight_packed).unwrap().sum_all().unwrap();
            let grads = loss.backward().unwrap();
            let dqkv = grads
                .get(qkv_var.as_tensor())
                .expect("RopePositionsFused: qkv must have a gradient");
            let dx = unpack_qkv_slot(dqkv, slot, b, s);
            assert_eq!(
                to_bits(&dx),
                to_bits(dx_ref),
                "RopePositionsFused's real backward (slot {slot}) must be bit-identical to \
                 RopeFused's real backward on the SAME cotangent, b={b} h={h} s={s} d={d}"
            );
        }
    }

    #[test]
    fn bwd_bit_identical_to_rope_fused_bwd_on_bhsd_b1_head_dim_matches_production() {
        bwd_bit_identity_case(1, 4, 9, 64);
    }

    #[test]
    fn bwd_bit_identical_to_rope_fused_bwd_on_bhsd_b8_head_dim_matches_production() {
        bwd_bit_identity_case(8, 4, 9, 64);
    }

    #[test]
    fn degenerate_d_zero_is_empty_not_a_panic() {
        let device = Device::Cpu;
        let qkv = Tensor::zeros((3, 3, 2, 0), DType::F32, &device).unwrap();
        let cos = Tensor::zeros((4, 0), DType::F32, &device).unwrap();
        let sin = Tensor::zeros((4, 0), DType::F32, &device).unwrap();
        let out = fused(4, false, &qkv, &cos, &sin).unwrap();
        assert_eq!(out.dims(), qkv.dims());
        assert_eq!(out.elem_count(), 0);
    }

    /// Family D: a table whose period disagrees with `seq` must be
    /// refused, not silently misindexed.
    #[test]
    fn table_period_mismatch_is_refused() {
        let device = Device::Cpu;
        let qkv = Tensor::zeros((6, 3, 2, 4), DType::F32, &device).unwrap(); // total=6, seq should be e.g. 3 or 6
        let (cos, sin) = rope_table(5, 4, 10_000.0); // period=5, matches neither seq=2 nor seq=3
        let err = fused(2, false, &qkv, &cos, &sin).unwrap_err();
        assert!(format!("{err}").contains("cos/sin table covers"));
    }

    /// Family D: `total` not a multiple of `seq` must be refused, not
    /// silently misindexed (`token % seq` is well-defined arithmetic even
    /// then, but semantically wrong -- mirrors `rope_dims`'s
    /// `total_rows_not_a_multiple_of_period_is_refused`). The table's own
    /// period matches `seq` exactly, so this exercises ONLY the new
    /// `total % seq` guard, not the pre-existing period check.
    #[test]
    fn total_not_a_multiple_of_seq_is_refused() {
        let device = Device::Cpu;
        // total=5, seq=2: 5 is not a multiple of 2.
        let qkv = Tensor::zeros((5, 3, 2, 4), DType::F32, &device).unwrap();
        let (cos, sin) = rope_table(2, 4, 10_000.0); // period=2, matches seq=2 exactly
        let err = fused(2, false, &qkv, &cos, &sin).unwrap_err();
        assert!(
            format!("{err}").contains("is not a multiple of seq"),
            "expected the total%seq guard's message, got: {err}"
        );
    }

    /// Family D boundary: `seq=0` with a nonempty qkv is refused, not a
    /// division-by-zero / modulo-by-zero panic.
    #[test]
    fn seq_zero_with_nonempty_qkv_is_refused_not_a_panic() {
        let device = Device::Cpu;
        let qkv = Tensor::zeros((2, 3, 1, 4), DType::F32, &device).unwrap();
        let (cos, sin) = rope_table(1, 4, 10_000.0);
        let err = fused(0, false, &qkv, &cos, &sin).unwrap_err();
        assert!(format!("{err}").contains("seq=0"));
    }

    /// A single identical token position (`s=1`, the degenerate/boundary
    /// "one point" case family D asks every op to cover) still matches
    /// `RopeFused` bit-for-bit.
    #[test]
    fn bit_identical_single_position_s_one() {
        bit_identity_case_with_red_control(2, 2, 1, 4, false);
    }

    #[test]
    fn cos_sin_variable_gradient_is_a_typed_error_not_silent_none() {
        use candle_core::Var;
        let device = Device::Cpu;
        let (b, h, s, d) = (1usize, 2usize, 3usize, 4usize);
        let n = b * h * s * d;
        let xv: Vec<f32> = (0..n).map(|k| k as f32 * 0.1).collect();
        let x_bhsd = Tensor::from_vec(xv, (b, h, s, d), &device).unwrap();
        let qkv = pack_bhsd_into_qkv(&x_bhsd, &x_bhsd, 0);
        let (cos, sin) = rope_table(s, d, 10_000.0);
        let cos = Var::from_tensor(&cos).unwrap();
        let out = fused(s, false, &qkv, &cos, &sin).unwrap();
        let loss = out.sum_all().unwrap();
        let err = loss.backward().unwrap_err();
        assert!(format!("{err}").contains("gradient is not implemented"));
    }

    /// PIN (audit's "the dense arm errors correctly" claim, made explicit
    /// for the `track_op()` class fix): a tracked-but-not-`Var` `cos`
    /// (`cos_var.as_tensor() * 1.0` -- the SAME construction
    /// `low_rank_residual_linear.rs`'s own regression test uses) reaching
    /// the DENSE entry directly must ALSO be a typed refusal, not a panic
    /// -- the test above only proves the LITERAL-`Var` case (where
    /// `is_variable()` alone already caught it, before AND after this
    /// round's fix); this test proves the DENSE arm's `track_op()` gate
    /// catches the tracked-non-Var case an `is_variable()`-only gate would
    /// have missed, the same class the ragged arm's own probe (below)
    /// exercises.
    #[test]
    fn cos_sin_tracked_non_variable_gradient_is_a_typed_error_not_silent_none() {
        use candle_core::Var;
        let device = Device::Cpu;
        let (b, h, s, d) = (1usize, 2usize, 3usize, 4usize);
        let n = b * h * s * d;
        let xv: Vec<f32> = (0..n).map(|k| k as f32 * 0.1).collect();
        let x_bhsd = Tensor::from_vec(xv, (b, h, s, d), &device).unwrap();
        let qkv = pack_bhsd_into_qkv(&x_bhsd, &x_bhsd, 0);
        let (cos_base, sin) = rope_table(s, d, 10_000.0);
        let cos_var = Var::from_tensor(&cos_base).unwrap();
        let cos = (cos_var.as_tensor() * 1.0).unwrap();
        assert!(
            !cos.is_variable() && cos.track_op(),
            "cos must be tracked-but-not-a-Var: the exact cell the panic lived in"
        );
        let out = fused(s, false, &qkv, &cos, &sin).unwrap();
        let loss = out.sum_all().unwrap();
        let err = loss
            .backward()
            .expect_err("a tracked-non-Var cos must be a typed refusal, never a panic");
        assert!(format!("{err}").contains("gradient is not implemented"));
    }

    /// THE class fix's own probe (audit): the LIVE hazard a Var-backed
    /// BASE table reaching the PUBLIC ragged entry point,
    /// `rope_positions_fused_ragged`, used to trigger -- `gather_ragged_tables`'s
    /// `index_select` output (`cos_r`) is NEVER `is_variable()` (it is the
    /// `index_select` RESULT, not the `Var` itself) but ALWAYS `track_op()`
    /// whenever the caller's own `cos_base` is a `Var`, so an
    /// `is_variable()`-only gate at `bwd` silently returned `Ok` and let
    /// `apply3` return `(Some(dx), None, None)` -- candle's OWN
    /// `sorted_nodes` walk (`backprop.rs`) still expected a gradient entry
    /// for `cos_r` (its ancestry reaches `cos_base`, the `Var`), so
    /// `Tensor::backward()` PANICKED at `backprop.rs:175` ("grad not
    /// populated") instead of returning this test's clean typed `Err`.
    #[test]
    fn rope_positions_fused_ragged_refuses_a_var_backed_base_table_not_a_panic() {
        use candle_core::Var;
        let device = Device::Cpu;
        let (h, d) = (2usize, 4usize);
        let lengths = [3usize, 2];
        let total: usize = lengths.iter().sum();
        let n = total * 3 * h * d;
        let xv: Vec<f32> = (0..n).map(|k| k as f32 * 0.1).collect();
        let qkv = Tensor::from_vec(xv, (total, 3, h, d), &device).unwrap();
        let (cos_base, sin_base) = rope_table(4, d, 10_000.0); // period_base=4 >= max(lengths)=3
        let cos_var = Var::from_tensor(&cos_base).unwrap();
        assert!(cos_var.as_tensor().is_variable());

        let out =
            rope_positions_fused_ragged(&qkv, cos_var.as_tensor(), &sin_base, &lengths, false)
                .expect("forward itself must succeed -- the refusal is a bwd-only concern");
        let loss = out.sum_all().unwrap();
        let err = loss
            .backward()
            .expect_err("a Var-backed base table must be a typed refusal, never a panic");
        assert!(
            format!("{err}").contains("gradient is not implemented"),
            "expected the typed cos/sin-gradient-not-implemented refusal, got: {err}"
        );
    }

    // =======================================================================
    // M1a — varlen positions (the ragged arm). See the module doc's "The
    // ragged arm" section. TRUTH oracle, DENSE INVARIANCE, and GUARD
    // inventory (contract's oracle inventory items 1/3/4) all live here;
    // item 2 (RETENTION, CUDA-only) lives in `ops::flash_attention`.
    // =======================================================================

    /// Builds a `[total, 3, h, d]` ragged `qkv` (the concatenation of
    /// `lengths.len()` per-segment `[1, h, len_b, d]` chunks, each with its
    /// OWN distinct, non-trivial fill so no segment's correctness can hide
    /// behind another's), and the segment-by-segment `RopeFused` reference
    /// (`[1, h, total, d]`, concatenated along the sequence axis) THE
    /// contract's truth oracle: "rotate each segment b's rows with the
    /// already-proven dense `RopeFused` against a `[len_b, d]` table
    /// slice, concatenate, compare bit-level". `cos_base`/`sin_base` cover
    /// `period_base = max(lengths)` positions (the minimum a caller must
    /// supply).
    fn ragged_truth_fixture(
        lengths: &[usize],
        h: usize,
        d: usize,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let device = Device::Cpu;
        let period_base = lengths.iter().copied().max().unwrap_or(1).max(1);
        let (cos_base, sin_base) = rope_table(period_base, d, 10_000.0);

        let mut q_segments: Vec<Tensor> = Vec::new();
        let mut ref_segments: Vec<Tensor> = Vec::new();
        let mut row_offset = 0usize;
        for &len_b in lengths {
            let n = h * len_b * d;
            let xv: Vec<f32> = (0..n)
                .map(|k| ((row_offset * 97 + k) as f32 * 0.037).sin() * 3.0)
                .collect();
            let fv: Vec<f32> = (0..n)
                .map(|k| ((row_offset * 53 + k) as f32 * 0.091 + 1.0).cos() * 2.0)
                .collect();
            let x_bhsd = Tensor::from_vec(xv, (1, h, len_b, d), &device).unwrap();
            let filler_bhsd = Tensor::from_vec(fv, (1, h, len_b, d), &device).unwrap();
            let cos_slice = cos_base.narrow(0, 0, len_b).unwrap();
            let sin_slice = sin_base.narrow(0, 0, len_b).unwrap();
            let reference =
                super::super::apply3(&x_bhsd, &cos_slice, &sin_slice, RopeFused::new(false))
                    .unwrap();
            ref_segments.push(reference);
            q_segments.push(pack_bhsd_into_qkv(&x_bhsd, &filler_bhsd, 0));
            row_offset += len_b;
        }
        let q_refs: Vec<&Tensor> = q_segments.iter().collect();
        let qkv = Tensor::cat(&q_refs, 0).unwrap(); // [total, 3, h, d]
        let ref_refs: Vec<&Tensor> = ref_segments.iter().collect();
        let reference = Tensor::cat(&ref_refs, 2).unwrap(); // [1, h, total, d]
        (qkv, reference, cos_base, sin_base)
    }

    /// `[total, 3, h, d]` -> `[1, h, total, d]` for slot `slot` — the
    /// ragged analogue of `unpack_qkv_slot` (that one assumes a single
    /// `b`/`s`; this one has neither, only `total`).
    fn unpack_ragged_slot(qkv: &Tensor, slot: usize, h: usize, d: usize) -> Tensor {
        let (total, _, h2, d2) = qkv.dims4().unwrap();
        assert_eq!((h2, d2), (h, d));
        qkv.narrow(1, slot, 1)
            .unwrap()
            .reshape((total, h, d))
            .unwrap()
            .transpose(0, 1)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .contiguous()
            .unwrap()
    }

    /// THE ragged TRUTH oracle (contract oracle inventory item 1): ragged
    /// rotation vs the per-segment dense `RopeFused` reference, bit-level,
    /// on Q (slot 0) and K (slot 1). RED control: an off-by-one segment
    /// boundary in `lengths` (SAME total, DIFFERENT segmentation) must be
    /// caught -- passing `wrong_lengths` (which still sums to the SAME
    /// `total` as `lengths`, so the op does not simply refuse it) must NOT
    /// reproduce the reference built from the CORRECT `lengths`.
    #[test]
    fn ragged_bit_identical_to_per_segment_dense_rope_fused() {
        let lengths = [3usize, 5, 2];
        let wrong_lengths = [4usize, 4, 2]; // same total=10, shifted boundary
        assert_eq!(
            lengths.iter().sum::<usize>(),
            wrong_lengths.iter().sum::<usize>()
        );
        let (h, d) = (3usize, 6usize);
        let (qkv, reference, cos_base, sin_base) = ragged_truth_fixture(&lengths, h, d);

        let out = rope_positions_fused_ragged(&qkv, &cos_base, &sin_base, &lengths, false)
            .expect("ragged rotation must accept the fixture it was built from");
        let got_q = unpack_ragged_slot(&out, 0, h, d);
        assert_eq!(
            to_bits(&got_q),
            to_bits(&reference),
            "ragged rope_positions on slot 0 (q) must be bit-identical to the per-segment \
             dense RopeFused reference, lengths={lengths:?}"
        );

        // V slot pass-through, byte-identical to the qkv's OWN slot 2.
        let got_v = unpack_ragged_slot(&out, 2, h, d);
        let expected_v = unpack_ragged_slot(&qkv, 2, h, d);
        assert_eq!(
            to_bits(&got_v),
            to_bits(&expected_v),
            "ragged rope_positions must pass V (slot 2) through unchanged"
        );

        // Slot 1 (K) must ALSO be bit-identical -- independent packing, the
        // same "cannot hide behind slot 0" discipline `bit_identity_case`
        // uses for the dense arm.
        let (qkv_k, reference_k, cos_base_k, sin_base_k) = ragged_truth_fixture(&lengths, h, d);
        let qkv_k_slot1 = {
            // ragged_truth_fixture packs into slot 0; re-pack the SAME
            // per-segment data into slot 1 instead by re-deriving from the
            // slot-0 qkv's own Q data (narrow it back out, then re-pack).
            let q_bhsd = unpack_ragged_slot(&qkv_k, 0, h, d); // [1,h,total,d]
            let filler_bhsd = unpack_ragged_slot(&qkv_k, 2, h, d);
            pack_bhsd_into_qkv(&q_bhsd, &filler_bhsd, 1)
        };
        let out_k =
            rope_positions_fused_ragged(&qkv_k_slot1, &cos_base_k, &sin_base_k, &lengths, false)
                .unwrap();
        let got_k = unpack_ragged_slot(&out_k, 1, h, d);
        assert_eq!(
            to_bits(&got_k),
            to_bits(&reference_k),
            "ragged rope_positions on slot 1 (k) must be bit-identical to the per-segment \
             dense RopeFused reference, lengths={lengths:?}"
        );

        // RED control: a boundary-shifted `wrong_lengths` (same total,
        // different segmentation) must NOT reproduce the reference built
        // from the CORRECT `lengths`.
        let out_wrong =
            rope_positions_fused_ragged(&qkv, &cos_base, &sin_base, &wrong_lengths, false)
                .expect("wrong_lengths sums to the same total, so the op must not refuse it");
        let got_q_wrong = unpack_ragged_slot(&out_wrong, 0, h, d);
        assert_ne!(
            to_bits(&got_q_wrong),
            to_bits(&reference),
            "RED control: an off-by-one segment boundary in `lengths` must NOT reproduce the \
             correct-segmentation reference -- lengths={lengths:?} vs wrong={wrong_lengths:?}"
        );
    }

    /// Direct assertion of `gather_ragged_tables`' own row selection:
    /// `cos_r[r] == cos_base[positions[r]]` (and same for `sin`), per
    /// segment -- the contract's "the gathered tables are assertable
    /// directly" oracle observable.
    #[test]
    fn gather_ragged_tables_selects_the_correct_base_rows() {
        let d = 4usize;
        let period_base = 6usize;
        let (cos_base, sin_base) = rope_table(period_base, d, 10_000.0);
        let lengths = [3usize, 2, 4]; // max=4 <= period_base=6
        let (total, cos_r, sin_r) = gather_ragged_tables(&cos_base, &sin_base, &lengths).unwrap();
        assert_eq!(total, 9);
        let positions = ragged_positions_from_lengths(&lengths).unwrap();
        assert_eq!(positions, vec![0, 1, 2, 0, 1, 0, 1, 2, 3]);

        let cos_base_v: Vec<f32> = cos_base.flatten_all().unwrap().to_vec1().unwrap();
        let sin_base_v: Vec<f32> = sin_base.flatten_all().unwrap().to_vec1().unwrap();
        let cos_r_v: Vec<f32> = cos_r.flatten_all().unwrap().to_vec1().unwrap();
        let sin_r_v: Vec<f32> = sin_r.flatten_all().unwrap().to_vec1().unwrap();
        for (r, &pos) in positions.iter().enumerate() {
            let pos = pos as usize;
            assert_eq!(
                &cos_r_v[r * d..r * d + d],
                &cos_base_v[pos * d..pos * d + d],
                "cos_r[{r}] must equal cos_base[positions[{r}]={pos}]"
            );
            assert_eq!(
                &sin_r_v[r * d..r * d + d],
                &sin_base_v[pos * d..pos * d + d],
                "sin_r[{r}] must equal sin_base[positions[{r}]={pos}]"
            );
        }
    }

    /// DENSE INVARIANCE (contract oracle inventory item 3): the ragged
    /// entry at UNIFORM lengths (every segment == `s`) must be bit-for-bit
    /// identical to the existing DENSE path, `RopePositionsFused::new`
    /// unchanged -- proving the ragged arm is a strict generalization, not
    /// a parallel, possibly-diverging code path.
    fn ragged_matches_dense_at_uniform_lengths(b: usize, h: usize, s: usize, d: usize) {
        let device = Device::Cpu;
        let total = b * s;
        let n = total * 3 * h * d;
        let qkv_v: Vec<f32> = (0..n).map(|k| (k as f32 * 0.043).sin() * 4.0).collect();
        let qkv = Tensor::from_vec(qkv_v, (total, 3, h, d), &device).unwrap();
        let (cos, sin) = rope_table(s, d, 10_000.0);
        let lengths = vec![s; b];

        let out_ragged = rope_positions_fused_ragged(&qkv, &cos, &sin, &lengths, false).unwrap();
        let out_dense = fused(s, false, &qkv, &cos, &sin).unwrap();
        assert_eq!(
            to_bits(&out_ragged),
            to_bits(&out_dense),
            "ragged entry at uniform lengths (all == {s}) must be bit-identical to the \
             existing dense path, b={b} h={h} s={s} d={d}"
        );
    }

    #[test]
    fn ragged_matches_dense_at_uniform_lengths_b1() {
        ragged_matches_dense_at_uniform_lengths(1, 2, 5, 4);
    }

    #[test]
    fn ragged_matches_dense_at_uniform_lengths_b8() {
        ragged_matches_dense_at_uniform_lengths(8, 3, 7, 6);
    }

    // --- GUARDS (contract oracle inventory item 4) ---------------------

    /// Mis-sized table refusal: qkv's total matches the claimed `seq`
    /// (=total), but the SUPPLIED cos/sin table covers a DIFFERENT number
    /// of rows -- a caller bug the dense arm's `period == 1` shared-row
    /// convenience cannot hide behind in the ragged arm (see the module
    /// doc). Constructs `RopePositionsFused` directly (`new_with_arm`,
    /// `pub(crate)`) rather than through `rope_positions_fused_ragged`,
    /// which always gathers a correctly-sized table by construction --
    /// this test is specifically about a caller who bypasses the gather.
    #[test]
    fn ragged_mis_sized_table_is_refused() {
        let device = Device::Cpu;
        let qkv = Tensor::zeros((5, 3, 2, 4), DType::F32, &device).unwrap(); // total=5
        let (cos, sin) = rope_table(3, 4, 10_000.0); // period=3, NOT 5
        let op = RopePositionsFused::new_with_arm(5, false, PositionArm::Ragged);
        let err = super::super::apply3(&qkv, &cos, &sin, op).unwrap_err();
        assert!(
            format!("{err}").contains("requires the pre-gathered table to cover EXACTLY"),
            "expected the ragged mis-sized-table guard's message, got: {err}"
        );
    }

    /// `qkv`'s own total disagreeing with the claimed table row-total
    /// (`seq`) -- the OTHER half of the ragged arm's structural fence
    /// (independent from the mis-sized-table guard above: here the
    /// table's OWN period matches `seq` exactly; only `qkv`'s total is
    /// wrong).
    #[test]
    fn ragged_qkv_total_disagreeing_with_table_total_is_refused() {
        let device = Device::Cpu;
        let qkv = Tensor::zeros((5, 3, 2, 4), DType::F32, &device).unwrap(); // total=5
        let (cos, sin) = rope_table(4, 4, 10_000.0); // period=4, matches seq=4 exactly
        let op = RopePositionsFused::new_with_arm(4, false, PositionArm::Ragged);
        let err = super::super::apply3(&qkv, &cos, &sin, op).unwrap_err();
        assert!(
            format!("{err}").contains("requires qkv's total rows"),
            "expected the ragged qkv/table total-mismatch guard's message, got: {err}"
        );
    }

    /// `total == 0` is a REFUSAL in the ragged arm -- a deliberate delta
    /// from the dense arm's own empty-batch acceptance, proven by
    /// constructing BOTH arms over the IDENTICAL `total=0` shape and
    /// showing they disagree.
    #[test]
    fn ragged_arm_total_zero_is_refused_while_dense_arm_accepts_it() {
        let device = Device::Cpu;
        let qkv = Tensor::zeros((0, 3, 2, 4), DType::F32, &device).unwrap();
        let (cos, sin) = rope_table(1, 4, 10_000.0); // period=1, dense arm's shared-row table

        let dense_ok = super::super::apply3(&qkv, &cos, &sin, RopePositionsFused::new(0, false));
        assert!(
            dense_ok.is_ok(),
            "dense arm must still accept total=0/seq=0 (unchanged empty-batch behavior)"
        );

        let op = RopePositionsFused::new_with_arm(0, false, PositionArm::Ragged);
        let ragged_err = super::super::apply3(&qkv, &cos, &sin, op).unwrap_err();
        assert!(
            format!("{ragged_err}").contains("ragged arm refuses total=0"),
            "expected the ragged total=0 refusal's message, got: {ragged_err}"
        );
    }

    /// The entry point's own `total == 0` refusal (an empty `lengths`),
    /// reached through the PUBLIC `rope_positions_fused_ragged` surface
    /// rather than a direct `new_with_arm` construction.
    #[test]
    fn rope_positions_fused_ragged_refuses_empty_lengths() {
        let device = Device::Cpu;
        let qkv = Tensor::zeros((0, 3, 2, 4), DType::F32, &device).unwrap();
        let (cos, sin) = rope_table(4, 4, 10_000.0);
        let err = rope_positions_fused_ragged(&qkv, &cos, &sin, &[], false).unwrap_err();
        assert!(format!("{err}").contains("ragged arm refuses total=0"));
    }

    /// UNIFIED `lengths` contract (audit advisory 2): a zero-length
    /// segment is refused by `rope_positions_fused_ragged`, matching
    /// `flash_attention_varlen_with_rope_ragged`'s own (pre-existing,
    /// `CuSeqlens::from_lengths`-derived) refusal of the SAME shape -- an
    /// earlier version of this function silently accepted `lengths=[3, 0,
    /// 5]` (treating a zero-length segment as "contributes zero rows"),
    /// while the flash sibling refused it, giving two different answers
    /// to the same question depending on which entry point a caller
    /// reached. `total=8` (3+0+5) is nonempty, so this exercises ONLY the
    /// per-segment `len == 0` guard, not the separate `total == 0` guard
    /// the test above covers.
    #[test]
    fn rope_positions_fused_ragged_refuses_a_zero_length_segment() {
        let device = Device::Cpu;
        let lengths = [3usize, 0, 5];
        let total: usize = lengths.iter().sum();
        let (h, d) = (2usize, 4usize);
        let qkv = Tensor::zeros((total, 3, h, d), DType::F32, &device).unwrap();
        let (cos, sin) = rope_table(5, d, 10_000.0);
        let err = rope_positions_fused_ragged(&qkv, &cos, &sin, &lengths, false).unwrap_err();
        assert!(
            format!("{err}").contains("has length 0"),
            "expected the zero-length-segment guard's message, got: {err}"
        );
    }

    /// A derived position exceeding the base table's own period (a
    /// segment longer than any position the base table covers) is
    /// refused at the gather boundary, not silently indexed out of range.
    #[test]
    fn gather_ragged_tables_refuses_a_length_exceeding_the_base_period() {
        let d = 4usize;
        let (cos_base, sin_base) = rope_table(3, d, 10_000.0); // period_base=3
        let lengths = [2usize, 5]; // second segment needs position up to 4, > 3
        let err = gather_ragged_tables(&cos_base, &sin_base, &lengths).unwrap_err();
        assert!(
            format!("{err}").contains("exceeds cos_base's own period"),
            "expected the base-period-exceeded guard's message, got: {err}"
        );
    }
}
