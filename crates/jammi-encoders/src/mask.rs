//! Attention-mask conversion shared by every encoder forward pass.

use candle_core::{DType, Tensor};

use crate::error::EncoderError;

/// Convert a `[batch, seq]` u32 attention mask into a `[batch, 1, 1, seq]` f32
/// additive mask: `0.0` at real tokens, `-10000.0` at padding.
///
/// The constant `-10000.0` is the canonical HuggingFace / candle-transformers
/// value: large enough to drive `softmax(x + mask)` to zero at masked positions
/// for F32 inputs, while staying well within BF16 / F16 dynamic range so the
/// mask can be added before the cast to those dtypes.
pub(crate) fn extended_attention_mask(mask: &Tensor) -> Result<Tensor, EncoderError> {
    let mask_f = mask.to_dtype(DType::F32)?;
    let extended = mask_f.unsqueeze(1)?.unsqueeze(2)?;
    // affine(mul, add) computes self*mul + add, so (mask*10000) - 10000
    // maps mask=1 -> 0.0 (real) and mask=0 -> -10000.0 (padding).
    Ok(extended.affine(10000.0, -10000.0)?)
}
