//! Sentence-embedding pooling strategies plus the shared
//! `pool_and_normalize` helper used by every encoder's pooled forward.

use candle_core::{DType, Tensor, D};

use crate::error::EncoderError;

/// Sentence-transformer-compatible pooling strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Pooling {
    /// Mean over real (non-padding) tokens. Sentence-transformers default.
    #[default]
    Mean,
    /// First token's hidden state (\[CLS\] for BERT-family backbones).
    Cls,
    /// Element-wise max over real tokens (padding positions are excluded by
    /// substituting `-inf` before the reduce).
    Max,
    /// Linear-position-weighted mean — token at position `i` (1-indexed) is
    /// weighted by `i`, normalised by the sum of effective weights. Matches
    /// sentence-transformers' `WeightedMeanPooling`. No learnable parameters.
    WeightedMean,
}

/// Apply pooling and L2 normalisation to a `[batch, seq, hidden]` tensor with
/// a `[batch, seq]` attention mask.
///
/// Returns a `[batch, hidden]` tensor whose rows are unit-length under L2.
pub fn pool_and_normalize(
    hidden: &Tensor,
    attention_mask: &Tensor,
    strategy: Pooling,
) -> Result<Tensor, EncoderError> {
    let pooled = match strategy {
        Pooling::Mean => mean_pool(hidden, attention_mask)?,
        Pooling::Cls => cls_pool(hidden)?,
        Pooling::Max => max_pool(hidden, attention_mask)?,
        Pooling::WeightedMean => weighted_mean_pool(hidden, attention_mask)?,
    };
    l2_normalize(&pooled)
}

fn mask_f32(hidden: &Tensor, attention_mask: &Tensor) -> Result<Tensor, EncoderError> {
    Ok(attention_mask
        .to_dtype(DType::F32)?
        .unsqueeze(2)?
        .broadcast_as(hidden.shape())?)
}

fn mean_pool(hidden: &Tensor, attention_mask: &Tensor) -> Result<Tensor, EncoderError> {
    let mask = mask_f32(hidden, attention_mask)?;
    let masked = hidden.broadcast_mul(&mask)?;
    let summed = masked.sum(1)?;
    let count = mask.sum(1)?;
    Ok((summed.broadcast_div(&count.clamp(1e-9, f32::MAX as f64)?))?)
}

fn cls_pool(hidden: &Tensor) -> Result<Tensor, EncoderError> {
    Ok(hidden.narrow(1, 0, 1)?.squeeze(1)?)
}

fn max_pool(hidden: &Tensor, attention_mask: &Tensor) -> Result<Tensor, EncoderError> {
    // Push padding positions to a very negative value so they lose the max.
    // Multiplying neg_inf by 0 yields NaN, so we add a large finite negative
    // bias instead: 0 at real tokens, -1e30 at padding.
    let mask = mask_f32(hidden, attention_mask)?;
    let bias = mask.affine(1e30, -1e30)?;
    Ok(hidden.broadcast_add(&bias)?.max(1)?)
}

fn weighted_mean_pool(hidden: &Tensor, attention_mask: &Tensor) -> Result<Tensor, EncoderError> {
    let (_batch, seq, _hidden) = hidden.dims3()?;
    let positions: Vec<f32> = (1..=seq as u32).map(|i| i as f32).collect();
    let weights = Tensor::from_vec(positions, (seq,), hidden.device())?
        .unsqueeze(0)?
        .unsqueeze(2)?;
    let mask = mask_f32(hidden, attention_mask)?;
    let effective = mask.broadcast_mul(&weights)?;
    let weighted_hidden = hidden.broadcast_mul(&effective)?;
    let numerator = weighted_hidden.sum(1)?;
    let denominator = effective.sum(1)?;
    Ok(numerator.broadcast_div(&denominator.clamp(1e-9, f32::MAX as f64)?)?)
}

fn l2_normalize(pooled: &Tensor) -> Result<Tensor, EncoderError> {
    let norm = pooled
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .clamp(1e-12, f32::MAX as f64)?;
    Ok(pooled.broadcast_div(&norm)?)
}
