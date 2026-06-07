//! Mask-aware scaled dot-product (cross-)attention, lifted from the BERT
//! self-attention math into a public, reusable primitive.
//!
//! `BertSelfAttention::forward` computes `softmax(QKᵀ/√d + mask)·V` for the
//! single case where Q, K, and V all derive from one token sequence. The
//! amortized context predictors need the *same* arithmetic in two more shapes:
//!
//! - **AttnCNP** — a *cross*-attention where the query is one target row per
//!   episode and the keys/values are that episode's context members. Q and KV
//!   come from different tensors.
//! - **TNP** — *self*-attention over the `(context ∪ target)` token set, where
//!   absent/padded context members must receive zero weight.
//!
//! Both reduce to one operation: multi-head scaled dot-product attention of a
//! query set against a key/value set, with an additive key mask that drives the
//! softmax weight of padded keys to zero. This module is that operation. The
//! additive mask is produced by [`crate::mask::extended_attention_mask`] so a
//! padded context member is never attended over — the softmax weight on it is
//! `softmax(… + (-10000))` ≈ 0, identical to the encoders' padding convention.

use candle_core::{Tensor, D};

use crate::error::EncoderError;

/// Split the last dim of `[B, S, H]` into heads, giving `[B, heads, S, head_dim]`
/// contiguous (the `.contiguous()` is the same matmul-contiguity fix BERT uses).
fn split_heads(x: &Tensor, num_heads: usize) -> Result<Tensor, EncoderError> {
    let (b, s, hidden) = x.dims3()?;
    if hidden % num_heads != 0 {
        return Err(EncoderError::Config(format!(
            "attention: hidden dim {hidden} not divisible by num_heads {num_heads}"
        )));
    }
    let head_dim = hidden / num_heads;
    Ok(x.reshape((b, s, num_heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()?)
}

/// Multi-head scaled dot-product attention of a query set against a key/value
/// set: `softmax(QKᵀ/√d + key_mask)·V`, recombined across heads.
///
/// Shapes:
/// - `query`: `[B, q_len, hidden]`
/// - `keys`, `values`: `[B, kv_len, hidden]`
/// - `key_mask`: an *additive* `[B, 1, 1, kv_len]` mask (`0.0` at real keys,
///   a large negative at padded keys), as built by
///   [`crate::mask::extended_attention_mask`]. It broadcasts over the query and
///   head axes, so a padded key gets ≈ zero softmax weight for *every* query.
///   Pass `None` for unmasked attention (every key real).
///
/// Returns `[B, q_len, hidden]`. `hidden` must be divisible by `num_heads`.
///
/// The math is exactly BERT's self-attention (`softmax(QKᵀ/√d + mask)·V`)
/// generalised to distinct query and key/value tensors; the parity test pins it
/// to the BERT implementation on a fixed input.
pub fn multi_head_attention(
    query: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    key_mask: Option<&Tensor>,
    num_heads: usize,
) -> Result<Tensor, EncoderError> {
    let (b, q_len, hidden) = query.dims3()?;
    let head_dim = hidden / num_heads;

    let q = split_heads(query, num_heads)?; // [B, heads, q_len, head_dim]
    let k = split_heads(keys, num_heads)?; // [B, heads, kv_len, head_dim]
    let v = split_heads(values, num_heads)?; // [B, heads, kv_len, head_dim]

    let scores = q.matmul(&k.t()?)?; // [B, heads, q_len, kv_len]
    let scores = (scores / (head_dim as f64).sqrt())?;
    let scores = match key_mask {
        Some(mask) => scores.broadcast_add(mask)?,
        None => scores,
    };
    let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;

    let context = probs.matmul(&v)?; // [B, heads, q_len, head_dim]
    let context = context.transpose(1, 2)?.contiguous()?; // [B, q_len, heads, head_dim]
    Ok(context.reshape((b, q_len, hidden))?)
}

/// The softmax attention *weights* `softmax(QKᵀ/√d + key_mask)` for the
/// single-head case (`num_heads == 1`), shape `[B, q_len, kv_len]`.
///
/// Exposed so a caller can inspect *which* context members an attentive
/// predictor concentrates on (the "which context matters" payoff), without
/// re-deriving the scores. Padded keys carry ≈ 0 weight via the additive mask,
/// exactly as in [`multi_head_attention`].
pub fn attention_weights(
    query: &Tensor,
    keys: &Tensor,
    key_mask: Option<&Tensor>,
) -> Result<Tensor, EncoderError> {
    let (_b, _q_len, hidden) = query.dims3()?;
    let scores = query.matmul(&keys.t()?)?; // [B, q_len, kv_len]
    let scores = (scores / (hidden as f64).sqrt())?;
    let scores = match key_mask {
        // The additive mask is [B, 1, 1, kv_len]; squeeze the two singleton
        // head/query axes to broadcast against the [B, q_len, kv_len] scores.
        Some(mask) => scores.broadcast_add(&mask.squeeze(1)?.squeeze(1)?.unsqueeze(1)?)?,
        None => scores,
    };
    Ok(candle_nn::ops::softmax(&scores, D::Minus1)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// The lifted primitive reproduces the BERT self-attention math
    /// (`softmax(QKᵀ/√d + mask)·V`) on a fixed input, within f32 tolerance — the
    /// guarantee that lifting the math out of `bert.rs` changed nothing
    /// numerically. Single-head, self-attention (Q=K=V from one sequence), no
    /// mask: the exact `BertSelfAttention::forward` reference.
    #[test]
    fn reproduces_bert_self_attention_math() {
        let device = Device::Cpu;
        // Fixed [1, 3, 4] input; one head so head_dim = hidden = 4.
        let x = Tensor::from_vec(
            vec![
                0.1f32, 0.2, 0.3, 0.4, //
                -0.5, 0.6, -0.7, 0.8, //
                0.9, -1.0, 1.1, -1.2,
            ],
            (1, 3, 4),
            &device,
        )
        .unwrap();

        // The primitive, used as self-attention (query = keys = values = x).
        let got = multi_head_attention(&x, &x, &x, None, 1).unwrap();

        // Hand-rolled BERT reference: softmax(x·xᵀ / √4) · x (single head, so no
        // head split is needed — the math is identical to the multi-head path
        // with one head).
        let scores = x.matmul(&x.transpose(1, 2).unwrap()).unwrap();
        let scores = (scores / 2.0).unwrap(); // √(head_dim=4) = 2
        let probs = candle_nn::ops::softmax(&scores, D::Minus1).unwrap();
        let reference = probs.matmul(&x).unwrap();

        let g = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let r = reference.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(g.len(), r.len());
        for (a, b) in g.iter().zip(r.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "lifted attention diverges from BERT math: {a} vs {b}"
            );
        }
    }

    /// The additive mask drives a padded key's softmax weight to ≈ 0: a key
    /// masked with `-10000` receives negligible weight regardless of its score,
    /// so the attended output ignores it.
    #[test]
    fn masked_key_gets_zero_weight() {
        let device = Device::Cpu;
        let query = Tensor::ones((1, 1, 2), DType::F32, &device).unwrap();
        // Two keys; the second is far more aligned but will be masked out.
        let keys = Tensor::from_vec(vec![1.0f32, 0.0, 5.0, 5.0], (1, 2, 2), &device).unwrap();
        let values =
            Tensor::from_vec(vec![10.0f32, 10.0, -99.0, -99.0], (1, 2, 2), &device).unwrap();
        // Presence: key 0 real, key 1 padded.
        let presence = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &device).unwrap();
        let mask = crate::mask::extended_attention_mask(&presence).unwrap();

        let out = multi_head_attention(&query, &keys, &values, Some(&mask), 1).unwrap();
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // With key 1 masked out, the attended value is essentially key 0's value
        // (10, 10), not pulled toward the masked -99.
        assert!(v.iter().all(|&x| (x - 10.0).abs() < 0.1), "got {v:?}");
    }
}
