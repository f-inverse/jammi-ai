//! The one attention-probabilities softmax every fused-QKV self-attention
//! module in this crate calls, instead of each duplicating the same
//! training/eval dispatch.
//!
//! `candle_nn::ops::softmax_last_dim` is a `CustomOp1` applied via
//! `Tensor::apply_op1_no_bwd` (candle-nn-0.11.0/src/ops.rs:437-439):
//!
//! ```text
//! pub fn softmax_last_dim(xs: &Tensor) -> Result<Tensor> {
//!     xs.apply_op1_no_bwd(&SoftmaxLastDim)
//! }
//! ```
//!
//! and `apply_op1_no_bwd` (candle-core-0.11.0/src/custom_op.rs:156-159) hands
//! back its result wrapped in `BackpropOp::none()`:
//!
//! ```text
//! pub fn apply_op1_no_bwd<C: CustomOp1>(&self, c: &C) -> Result<Self> {
//!     let (storage, shape) = self.storage().apply_op1(self.layout(), c)?;
//!     Ok(from_storage(storage, shape, BackpropOp::none(), false))
//! }
//! ```
//!
//! A backward walk through a `BackpropOp::none()` node does not error — it
//! just stops traversing there. Every operand strictly upstream of the
//! softmax (e.g. the Q/K slices of a fused `in_proj_weight`, or a
//! relative-position bias table read only into the pre-softmax scores) comes
//! back with either an exactly-zero or an entirely-missing gradient entry,
//! never an error — a silently WRONG gradient, not a loud failure. Whatever
//! sits downstream of the softmax (typically V, through the `probs @ V`
//! matmul) is unaffected, since that matmul's own backward is ordinary and
//! intact.
//!
//! [`attention_softmax`] is the single dispatch point every attention module
//! in this crate composes: `training == false` (the default) takes
//! `softmax_last_dim`, matching every eval path's numerics unchanged from
//! before this module existed; `training == true` takes
//! `candle_nn::ops::softmax(scores, D::Minus1)`, the ordinary differentiable
//! max/sub/exp/sum/div composition, so backward reaches every operand.
//!
//! The two arms are measured bit-identical on CPU at f32 and bf16 (see
//! [`tests::softmax_last_dim_and_composed_softmax_are_cpu_bit_identical`]),
//! so `training == false` costs nothing to keep byte-identical to the
//! pre-existing eval path. This is NOT guaranteed on CUDA: candle's fused
//! softmax kernel and the composed primitive-op reduction can differ in
//! floating-point reduction order there, which is why `training` stays a
//! caller-controlled flag rather than always taking the composed arm.

use candle_core::{IndexOp, Tensor, D};
use candle_nn::{linear, Linear, Module, VarBuilder};

use crate::error::EncoderError;

/// Dispatch the attention-probabilities softmax on the module's training
/// flag — see this module's doc for why the two arms exist and when they
/// are (and are not) numerically interchangeable.
pub fn attention_softmax(scores: &Tensor, training: bool) -> Result<Tensor, EncoderError> {
    if training {
        Ok(candle_nn::ops::softmax(scores, D::Minus1)?)
    } else {
        Ok(candle_nn::ops::softmax_last_dim(scores)?)
    }
}

/// Multi-head self-attention with a fused QKV projection (OpenCLIP's
/// `in_proj_weight`/`in_proj_bias` plus an `out_proj` sub-module), shared by
/// the OpenCLIP text tower ([`crate::clip_text`], causally masked) and the
/// OpenCLIP vision tower ([`crate::open_clip_vision`], unmasked) — the two
/// towers' own copies of this module were byte-for-byte identical except
/// for whether an additive causal mask was applied, so [`Self::forward`]
/// generalizes that to an `Option<&Tensor>` parameter instead of
/// duplicating the module. `None` skips the mask `broadcast_add` entirely
/// (not "add a zero mask"), so the vision tower's op sequence stays exactly
/// what it was before this module existed; `Some(mask)` keeps the text
/// tower's `broadcast_add` unchanged.
pub(crate) struct MultiHeadAttention {
    in_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    /// Selects the softmax arm via [`attention_softmax`]. Defaults to
    /// `false` (eval); flipped by each tower's own `set_training`.
    training: bool,
}

impl MultiHeadAttention {
    pub(crate) fn load(
        vb: VarBuilder,
        width: usize,
        num_heads: usize,
    ) -> Result<Self, EncoderError> {
        let head_dim = width / num_heads;
        let in_proj_weight = vb.get((width * 3, width), "in_proj_weight")?;
        let in_proj_bias = vb.get(width * 3, "in_proj_bias")?;
        let in_proj = Linear::new(in_proj_weight, Some(in_proj_bias));
        let out_proj = linear(width, width, vb.pp("out_proj"))?;
        Ok(Self {
            in_proj,
            out_proj,
            num_heads,
            head_dim,
            training: false,
        })
    }

    pub(crate) fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// The fused `[3*width, width]` QKV projection weight — Q/K/V occupy
    /// row ranges `[0, width)` / `[width, 2*width)` / `[2*width, 3*width)`
    /// respectively. Exposed so a caller's own tests can inspect its
    /// gradient (e.g. the Q/K-vs-V backward-truncation oracles) without
    /// this crate-private struct's fields being `pub(crate)` individually.
    /// `#[cfg(test)]`-only: no production call site needs this, only the
    /// `open_clip_vision`/`clip_text` test modules that construct this
    /// struct directly.
    #[cfg(test)]
    pub(crate) fn in_proj_weight(&self) -> &Tensor {
        self.in_proj.weight()
    }

    /// `causal_mask`: `None` for unmasked (bidirectional) attention, or
    /// `Some(mask)` — an additive `[seq, seq]` tensor with `0.0` at allowed
    /// positions and a large negative value at masked positions,
    /// broadcast over `[batch, heads]`.
    pub(crate) fn forward(
        &self,
        x: &Tensor,
        causal_mask: Option<&Tensor>,
    ) -> Result<Tensor, EncoderError> {
        let (batch, seq_len, _) = x.dims3()?;
        let qkv = self.in_proj.forward(x)?;
        let qkv = qkv.reshape((batch, seq_len, 3, self.num_heads, self.head_dim))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?; // (3, batch, heads, seq, head_dim)

        let q = qkv.i(0)?.contiguous()?;
        let k = qkv.i(1)?.contiguous()?;
        let v = qkv.i(2)?.contiguous()?;

        let scale = (self.head_dim as f64).sqrt();
        let attn_scores =
            (crate::contiguous_matmul(&q, &k.transpose(D::Minus2, D::Minus1)?)? / scale)?;
        let attn_scores = match causal_mask {
            Some(mask) => attn_scores.broadcast_add(mask)?,
            None => attn_scores,
        };
        let attn_weights = attention_softmax(&attn_scores, self.training)?;
        let attn_output = crate::contiguous_matmul(&attn_weights, &v)?;

        let attn_output = attn_output.permute((0, 2, 1, 3))?.reshape((
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        Ok(self.out_proj.forward(&attn_output)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// `softmax_last_dim`'s fused CUSTOM-OP kernel and the composed
    /// max/sub/exp/sum/div `softmax` are bit-identical on CPU — the reason
    /// `training == false` is free to keep byte-identical to the
    /// pre-existing eval path in every caller of [`attention_softmax`].
    /// Only CUDA's fused softmax kernel may fold in a different reduction
    /// order (the training flag is load-bearing there, not here). Moved
    /// here (was duplicated verbatim in `clip_text.rs`) since the claim is
    /// about `attention_softmax`'s two arms, not about any one tower.
    #[test]
    fn softmax_last_dim_and_composed_softmax_are_cpu_bit_identical() {
        let device = Device::Cpu;
        for (rows, cols) in [(8usize, 512usize), (64, 64)] {
            let mut state: u32 = 7;
            let n = rows * cols;
            let values: Vec<f32> = (0..n)
                .map(|_| {
                    state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    let unit = (state >> 8) as f32 / (1u32 << 24) as f32;
                    (unit - 0.5) * 20.0 // wide range to exercise the max-shift
                })
                .collect();
            let x = Tensor::from_vec(values, (rows, cols), &device).unwrap();

            let fused = attention_softmax(&x, false).unwrap();
            let composed = attention_softmax(&x, true).unwrap();
            assert_eq!(
                fused.to_vec2::<f32>().unwrap(),
                composed.to_vec2::<f32>().unwrap(),
                "f32 [{rows},{cols}]: fused vs composed softmax must be bit-identical on CPU"
            );

            let x_bf16 = x.to_dtype(DType::BF16).unwrap();
            let fused_bf16 = attention_softmax(&x_bf16, false).unwrap();
            let composed_bf16 = attention_softmax(&x_bf16, true).unwrap();
            let fused_bits: Vec<u16> = fused_bf16
                .to_vec2::<half::bf16>()
                .unwrap()
                .into_iter()
                .flatten()
                .map(|v| v.to_bits())
                .collect();
            let composed_bits: Vec<u16> = composed_bf16
                .to_vec2::<half::bf16>()
                .unwrap()
                .into_iter()
                .flatten()
                .map(|v| v.to_bits())
                .collect();
            assert_eq!(
                fused_bits, composed_bits,
                "bf16 [{rows},{cols}]: fused vs composed softmax must be bit-identical on CPU"
            );
        }
    }
}
