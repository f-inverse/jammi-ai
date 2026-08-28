//! Attention-mask conversion shared by every encoder forward pass.

use candle_core::{DType, Device, Tensor};

use crate::error::EncoderError;

/// The additive logit applied to a position attention must not see.
///
/// The canonical HuggingFace / candle-transformers value: large enough that
/// `softmax(x + MASKED_LOGIT)` underflows to zero for F32 inputs, while staying
/// well inside BF16 / F16 dynamic range so the mask can be added before the cast
/// to those dtypes. `f32::NEG_INFINITY` would also zero the position but
/// propagates NaN through any row that ends up fully masked, turning a masking
/// bug into a silent NaN rather than a visible one.
pub(crate) const MASKED_LOGIT: f32 = -10_000.0;

/// Convert a `[batch, seq]` u32 attention mask into a `[batch, 1, 1, seq]` f32
/// additive mask: `0.0` at real tokens, [`MASKED_LOGIT`] at padding.
///
/// Depends only on the *key* position, so a single row broadcasts over every
/// query; contrast [`sliding_window_mask`], whose value depends on the distance
/// between query and key.
pub(crate) fn extended_attention_mask(mask: &Tensor) -> Result<Tensor, EncoderError> {
    let mask_f = mask.to_dtype(DType::F32)?;
    let extended = mask_f.unsqueeze(1)?.unsqueeze(2)?;
    // affine(mul, add) computes self*mul + add, so (mask * -MASKED_LOGIT) +
    // MASKED_LOGIT maps mask=1 -> 0.0 (real) and mask=0 -> MASKED_LOGIT.
    Ok(extended.affine(-(MASKED_LOGIT as f64), MASKED_LOGIT as f64)?)
}

/// Build the additive sliding-window band for a local-attention layer:
/// `[1, 1, seq, seq]`, `0.0` where `|query - key| <= half_window` and
/// `MASKED_LOGIT` outside it.
///
/// Shape note: the padding mask is `[batch, 1, 1, seq]` because it depends only
/// on the *key* position, so one row serves every query. A sliding window
/// depends on the distance between query and key, so it needs the full `seq x
/// seq` grid — but not a batch dimension, since the band is a property of the
/// layer's geometry and not of the data. Keeping it batch-free lets both masks
/// broadcast onto the scores independently, so neither is ever materialised at
/// `[batch, heads, seq, seq]`.
///
/// The two masks compose by addition: a position outside the band *and* on a
/// pad token accumulates `2 * MASKED_LOGIT`, which still drives `softmax` to
/// zero.
///
/// The band alone always keeps a query's own diagonal position in-window
/// (`|i - i| = 0 <= half_window`), so a row masked by the band alone can
/// never be fully masked. But `sliding_window_mask` returns only the band —
/// it says nothing about whether the keys inside that window are themselves
/// padding. [`extended_attention_mask`] depends only on the *key* position,
/// so a query that is itself a pad token has its own diagonal key also
/// masked by padding (`real_mask[i] == 0` for that token maps to
/// `MASKED_LOGIT` there too). When every key within `[i - half_window, i +
/// half_window]` is a pad token — which happens whenever a pad query sits
/// deep enough inside a trailing pad run (longer than `half_window`) that
/// no real token falls within its window — every entry in that combined
/// row is `MASKED_LOGIT` (the band contributes `0.0`, padding contributes
/// `MASKED_LOGIT`, at every one of that row's keys). Row `i` is then fully
/// masked: this is the ordinary consequence of padding a short sequence to
/// a fixed length in a local-attention layer, not a corner case reachable
/// only via synthetic construction (`local_attention = 128` means
/// `half_window = 64`; any batch element with more than ~64 trailing pad
/// tokens hits this on every trailing pad-query row, in every
/// local-attention layer). See `jammi_kernels::ops::softmax`'s module doc
/// for the fused softmax kernel's deliberate behavior on exactly this row
/// class: a "safe softmax" zero output, matching PyTorch SDPA /
/// FlashAttention-2 rather than `candle_nn::ops::softmax`'s own NaN or
/// annihilated-uniform result there.
///
/// This is harmless downstream provided the hidden state this op leaves at
/// row `i` is finite: a zero or any other finite value there is not the
/// same as a `NaN` there for every pooling strategy. Row `i` only arises at
/// a query position that is itself padding, and `crate::pooling`'s two
/// pooling families handle that row differently: `mean_pool`/
/// `weighted_mean_pool` multiply `hidden` by the real `[batch, seq]`
/// attention mask before summing (`pooling.rs`'s
/// `hidden.broadcast_mul(&mask...)`, `mean_pool`) — a multiply, so a `NaN`
/// hidden state at that position is not masked away: `0.0 * NaN == NaN` in
/// IEEE754, and that `NaN` then propagates through `summed =
/// masked.sum(1)`, poisoning the entire pooled embedding for that
/// sequence, not just the one position. `max_pool` is the one that
/// genuinely survives `NaN`: it selects via `where_cond(hidden, sentinel)`
/// rather than multiplying (`pooling.rs`'s `max_pool`), so a `NaN` hidden
/// state at a pad position is discarded outright, never read into the
/// reduction at all. `cls_pool` reads only position `0` (never itself a
/// pad token in a real batch) and never touches row `i` regardless. The
/// safe claim is therefore: a zero or any other finite value at row `i` is
/// masked away identically by every pooling strategy; a `NaN` there is
/// masked away only by `max_pool`'s selection, not by `mean_pool`'s
/// multiply. This is exactly why the `MASKED_LOGIT` constant above picks a
/// finite value over `f32::NEG_INFINITY`: an `-inf` convention is not
/// equally safe, for precisely the `mean_pool` reason stated here.
pub(crate) fn sliding_window_mask(
    seq: usize,
    half_window: usize,
    device: &Device,
) -> Result<Tensor, EncoderError> {
    let mut band = Vec::with_capacity(seq * seq);
    for q in 0..seq {
        for k in 0..seq {
            let within = q.abs_diff(k) <= half_window;
            band.push(if within { 0.0f32 } else { MASKED_LOGIT });
        }
    }
    Ok(Tensor::from_vec(band, (1, 1, seq, seq), device)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `AttentionBlockFused` (`jammi-kernels`) has no `window` construction
    /// data — a caller combines `MASKED_LOGIT`-sentinel padding with a
    /// `sliding_window_mask` band into ONE additive mask before calling it
    /// (see that op's module doc's "window is construction data at the
    /// call site" section), and the op's own fully-masked-row rule reads
    /// that combined value with NO knowledge of which crate contributed
    /// which term. This test is the measured, asserted proof (family F)
    /// the two crates' independently-defined sentinels are the SAME value
    /// — not merely the same sign — so a doubly-masked key (padded AND
    /// out-of-window, `MASKED_LOGIT + jammi_kernels`'s own sentinel) lands
    /// at a magnitude with the same clearance from the `0.0`/masked
    /// boundary as a singly-masked one, rather than silently drifting if
    /// either crate's constant were ever edited independently.
    #[test]
    fn masked_logit_matches_jammi_kernels_window_masked_value() {
        assert_eq!(
            MASKED_LOGIT,
            jammi_kernels::ops::ATTENTION_BLOCK_WINDOW_MASKED_VALUE,
            "jammi-encoders::mask::MASKED_LOGIT and \
             jammi_kernels::ops::ATTENTION_BLOCK_WINDOW_MASKED_VALUE must be the exact same \
             sentinel — AttentionBlockFused combines whichever mask a caller hands it with no \
             way to tell padding from a window band apart"
        );
    }

    /// [`jammi_kernels::ops::MemEfficientAttention`]'s own SECOND copy of
    /// the band predicate (M2 plan v3 delta 3, adversarial audit round 3
    /// advisory: this pin was required and missing) — extends the SAME
    /// clearance-from-zero argument above to this op's `MEM_EFFICIENT_WINDOW_MASKED_VALUE`.
    /// Unlike `AttentionBlockFused` (which reads ONE caller-combined mask
    /// tensor and has no window construction data of its own), memeff
    /// re-derives its band internally and SUMS it with the caller's own
    /// `MASKED_LOGIT`-sentinel `key_mask` per chunk
    /// (`ModernBertAttention::forward_memeff_attention` hands it the raw
    /// padding-only mask, never a pre-combined one — see that op's module
    /// doc's "the band is a `Copy` scalar" section). A doubly-masked key
    /// (padded AND out-of-window) therefore lands at `MASKED_LOGIT +
    /// MEM_EFFICIENT_WINDOW_MASKED_VALUE` — the SAME sum-of-two-sentinels
    /// shape `AttentionBlockFused`'s own combined mask produces, so the
    /// two sentinels must match for the SAME reason: a mismatch would
    /// leave a doubly-masked memeff key at a DIFFERENT clearance from the
    /// `0.0`/masked boundary than a singly-masked one, drifting silently
    /// if either crate's constant were ever edited independently.
    #[test]
    fn masked_logit_matches_jammi_kernels_mem_efficient_window_masked_value() {
        assert_eq!(
            MASKED_LOGIT,
            jammi_kernels::ops::MEM_EFFICIENT_WINDOW_MASKED_VALUE,
            "jammi-encoders::mask::MASKED_LOGIT and \
             jammi_kernels::ops::MEM_EFFICIENT_WINDOW_MASKED_VALUE must be the exact same \
             sentinel — MemEfficientAttention sums its own internally-derived band with \
             whichever key_mask the caller hands it, per chunk"
        );
    }
}
