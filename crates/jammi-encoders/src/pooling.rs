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
    /// substituting the compute dtype's own finite most-negative value
    /// before the reduce, so an all-padding row still yields a finite
    /// result).
    Max,
    /// Linear-position-weighted mean — token at position `i` (1-indexed) is
    /// weighted by `i`, normalised by the sum of effective weights. Matches
    /// sentence-transformers' `WeightedMeanPooling`. No learnable parameters.
    WeightedMean,
}

impl std::fmt::Display for Pooling {
    /// The canonical lowercase token a downstream identity/report reader
    /// records for this strategy (unit-62 F-5', `jammi-bench`'s
    /// `EncodeStepTier::pooling`) — `"mean"`/`"cls"`/`"max"`/
    /// `"weighted_mean"`, mirroring the crate's existing lowercase-token
    /// convention for other resolved-strategy strings (e.g.
    /// `ComputePrecision`'s own `Display`).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Pooling::Mean => "mean",
            Pooling::Cls => "cls",
            Pooling::Max => "max",
            Pooling::WeightedMean => "weighted_mean",
        };
        f.write_str(s)
    }
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
    // The divisor floor must survive the cast to `hidden`'s dtype, not just
    // look right in F32: a `1e-9` eps underflows to `0.0` once cast to F16
    // (whose smallest subnormal is ~6e-8), turning an all-padding row's
    // `0 / 0` into NaN instead of a finite zero. `1.0` is exact in every
    // supported dtype and semantically correct — `count` is always a
    // non-negative integer (a sum of 0/1 mask entries), so a real row's count
    // is already >=1 and the floor only ever engages on the all-padding case,
    // where a `0`-sum divided by `1.0` is the finite zero vector.
    let mask = mask_f32(hidden, attention_mask)?;
    let masked = hidden.broadcast_mul(&mask.to_dtype(hidden.dtype())?)?;
    let summed = masked.sum(1)?;
    let count = mask.sum(1)?.clamp(1.0, f32::MAX as f64)?;
    Ok(summed.broadcast_div(&count.to_dtype(hidden.dtype())?)?)
}

fn cls_pool(hidden: &Tensor) -> Result<Tensor, EncoderError> {
    Ok(hidden.narrow(1, 0, 1)?.squeeze(1)?)
}

fn max_pool(hidden: &Tensor, attention_mask: &Tensor) -> Result<Tensor, EncoderError> {
    // Padding positions must lose the max via *selection*, not an additive
    // bias: a bias of `-1e30` is only finite in F32/F64 — once cast to F16
    // (max magnitude ~65504) it silently saturates to `-inf`, and even a
    // narrower bias risks `hidden + bias` itself overflowing past the
    // dtype's own range. `where_cond` instead keeps every real-token value
    // untouched and substitutes the dtype's own exact, finite sentinel at
    // padding positions, so an all-padding row's max is that finite
    // sentinel, never `-inf`.
    let is_real = attention_mask
        .to_dtype(DType::U8)?
        .unsqueeze(2)?
        .broadcast_as(hidden.shape())?;
    let sentinel = Tensor::new(neg_sentinel(hidden.dtype()), hidden.device())?
        .to_dtype(hidden.dtype())?
        .broadcast_as(hidden.shape())?;
    Ok(is_real.where_cond(hidden, &sentinel)?.max(1)?)
}

/// The most-negative value that is finite and exactly representable in
/// `dtype`, used as the padding sentinel for [`max_pool`]. F16's max
/// magnitude (~65504) is far narrower than F32/F64's, so a universal
/// `-1e30` sentinel would saturate to `-inf` once cast to F16; F16 gets its
/// own dtype-exact floor instead.
fn neg_sentinel(dtype: DType) -> f32 {
    match dtype {
        DType::F16 => -65504.0,
        _ => -1e30,
    }
}

fn weighted_mean_pool(hidden: &Tensor, attention_mask: &Tensor) -> Result<Tensor, EncoderError> {
    let (_batch, seq, _hidden) = hidden.dims3()?;
    let positions: Vec<f32> = (1..=seq as u32).map(|i| i as f32).collect();
    let weights = Tensor::from_vec(positions, (seq,), hidden.device())?
        .unsqueeze(0)?
        .unsqueeze(2)?;
    let mask = mask_f32(hidden, attention_mask)?;
    let effective = mask.broadcast_mul(&weights)?;
    let weighted_hidden = hidden.broadcast_mul(&effective.to_dtype(hidden.dtype())?)?;
    let numerator = weighted_hidden.sum(1)?;
    // Same divisor-floor invariant as `mean_pool`: `1.0` is exact in every
    // supported dtype (unlike a `1e-9` eps, which underflows to `0.0` in
    // F16). Every real token's position weight is `>=1` (positions are
    // 1-indexed), so a non-empty row's denominator is already `>=1` and the
    // floor only engages on the all-padding case, giving a finite zero
    // vector instead of `0 / 0`.
    let denominator = effective.sum(1)?.clamp(1.0, f32::MAX as f64)?;
    Ok(numerator.broadcast_div(&denominator.to_dtype(hidden.dtype())?)?)
}

fn l2_normalize(pooled: &Tensor) -> Result<Tensor, EncoderError> {
    // Same cast-survival invariant as the pooling divisors: `Tensor::clamp`
    // casts its scalar bound to `pooled`'s own dtype before comparing, so a
    // `1e-12` floor (exact in F32/F64) underflows to `0.0` once cast to F16
    // (min positive subnormal ~6e-8). A genuinely-zero pooled row (e.g. an
    // all-padding row through `mean_pool`/`weighted_mean_pool`) would then
    // hit `0 / 0 = NaN` instead of the intended `0 / floor = 0`. `norm_floor`
    // picks a value that is nonzero once cast to `pooled`'s dtype and still
    // far below any real embedding's norm, so it never perturbs a genuine
    // non-degenerate row.
    let norm = pooled
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .clamp(norm_floor(pooled.dtype()) as f64, f32::MAX as f64)?;
    Ok(pooled.broadcast_div(&norm)?)
}

/// The smallest strictly-positive L2-norm floor that survives the cast to
/// `dtype` performed inside [`Tensor::clamp`]. See [`l2_normalize`].
fn norm_floor(dtype: DType) -> f32 {
    match dtype {
        DType::F16 => 1e-4,
        _ => 1e-12,
    }
}
