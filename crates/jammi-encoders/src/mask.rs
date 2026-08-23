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
/// zero. Every query row retains at least its own position (`|i - i| = 0`), so
/// no row can be fully masked and `softmax` can never see an all-masked row.
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
