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
/// OpenCLIP vision tower ([`crate::open_clip_vision`], unmasked): a single
/// [`Self::forward`] parameterized by an `Option<&Tensor>` causal mask
/// instead of two near-identical modules. `None` skips the mask
/// `broadcast_add` entirely (not "add a zero mask"); `Some(mask)` applies
/// it. The Q/K/V split (`qkv.i(0..2)?`) yields non-contiguous slices of the
/// permuted fused projection, but no explicit `.contiguous()` is needed on
/// them here: every consumer is [`crate::contiguous_matmul`], which
/// contiguous-izes both its operands unconditionally, so the vision tower's
/// op sequence is exactly what it was before this module existed (one
/// implicit contiguous copy per operand, made inside the matmul primitive,
/// not two).
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

        let q = qkv.i(0)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;

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
    use candle_nn::VarMap;

    /// `MultiHeadAttention::forward`'s op sequence with the three
    /// `qkv.i(_)?.contiguous()?` calls this fix removed put back in
    /// (`contiguous_matmul` already contiguous-izes both its operands, so
    /// those calls were redundant copies, not a correctness dependency) —
    /// used only by
    /// [`tests::dropping_the_redundant_contiguous_calls_does_not_change_eval_output`]
    /// to prove the removal is a pure no-op on values.
    fn forward_with_redundant_contiguous(
        attn: &MultiHeadAttention,
        x: &Tensor,
        causal_mask: Option<&Tensor>,
    ) -> Result<Tensor, EncoderError> {
        let (batch, seq_len, _) = x.dims3()?;
        let qkv = attn.in_proj.forward(x)?;
        let qkv = qkv.reshape((batch, seq_len, 3, attn.num_heads, attn.head_dim))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?;

        let q = qkv.i(0)?.contiguous()?;
        let k = qkv.i(1)?.contiguous()?;
        let v = qkv.i(2)?.contiguous()?;

        let scale = (attn.head_dim as f64).sqrt();
        let attn_scores =
            (crate::contiguous_matmul(&q, &k.transpose(D::Minus2, D::Minus1)?)? / scale)?;
        let attn_scores = match causal_mask {
            Some(mask) => attn_scores.broadcast_add(mask)?,
            None => attn_scores,
        };
        let attn_weights = attention_softmax(&attn_scores, attn.training)?;
        let attn_output = crate::contiguous_matmul(&attn_weights, &v)?;

        let attn_output = attn_output.permute((0, 2, 1, 3))?.reshape((
            batch,
            seq_len,
            attn.num_heads * attn.head_dim,
        ))?;

        Ok(attn.out_proj.forward(&attn_output)?)
    }

    /// Advisory (i)'s claim, MEASURED: dropping the three redundant
    /// `qkv.i(_)?.contiguous()?` calls (since [`crate::contiguous_matmul`]
    /// already contiguous-izes both its operands) changes NO output byte —
    /// `.contiguous()` only ever materializes a data layout, never a value,
    /// so contiguous-izing an operand once (inside `contiguous_matmul`)
    /// versus twice (once explicitly here, redundantly, then again inside
    /// `contiguous_matmul`, a true no-op the second time) cannot change the
    /// numbers either way. Covers both the causally-masked shape
    /// (`clip_text`'s) and the unmasked shape (`open_clip_vision`'s), at an
    /// odd `seq_len` so no accidental symmetry could mask a real
    /// divergence.
    #[test]
    fn dropping_the_redundant_contiguous_calls_does_not_change_eval_output() {
        let device = Device::Cpu;
        let (width, heads, batch, seq_len) = (16usize, 4usize, 2usize, 7usize);
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attn = MultiHeadAttention::load(vb, width, heads).unwrap();
        {
            let mut state: u32 = 41;
            let data = varmap.data().lock().unwrap();
            let mut entries: Vec<_> = data.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            for (_, var) in entries {
                let n = var.shape().elem_count();
                let values: Vec<f32> = (0..n)
                    .map(|_| {
                        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                        ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.2
                    })
                    .collect();
                var.set(&Tensor::from_vec(values, var.shape().clone(), &device).unwrap())
                    .unwrap();
            }
        }

        let n = batch * seq_len * width;
        let xv: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.037 - 3.0).sin()).collect();
        let x = Tensor::from_vec(xv, (batch, seq_len, width), &device).unwrap();

        // Unmasked (open_clip_vision's shape) and causally masked
        // (clip_text's shape).
        let mut causal = vec![0f32; seq_len * seq_len];
        for row in 0..seq_len {
            for col in (row + 1)..seq_len {
                causal[row * seq_len + col] = f32::MIN;
            }
        }
        let causal_mask = Tensor::from_vec(causal, (seq_len, seq_len), &device).unwrap();

        for mask in [None, Some(&causal_mask)] {
            let current = attn.forward(&x, mask).unwrap();
            let old = forward_with_redundant_contiguous(&attn, &x, mask).unwrap();
            let current_bits: Vec<u32> = current
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect();
            let old_bits: Vec<u32> = old
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect();
            assert_eq!(
                current_bits,
                old_bits,
                "masked={}: dropping the redundant qkv.i(_)?.contiguous()? calls must not \
                 change a single output bit",
                mask.is_some()
            );
        }
    }

    /// `softmax_last_dim`'s fused CUSTOM-OP kernel and the composed
    /// max/sub/exp/sum/div `softmax` are bit-identical on CPU — the reason
    /// `training == false` is free to keep byte-identical to the
    /// pre-existing eval path in every caller of [`attention_softmax`].
    /// Only CUDA's fused softmax kernel may fold in a different reduction
    /// order (the training flag is load-bearing there, not here). Lives
    /// here, not in any one tower's test module, since the claim is about
    /// `attention_softmax`'s two arms and covers every tower that calls it.
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

    /// The bit-identity oracle above uses uniform `[-10, 10)` scores with no
    /// masked entry — every production caller instead feeds this softmax a
    /// row that has been additively masked, either with [`crate::clip_text`]'s
    /// causal-mask convention (`f32::MIN` at disallowed positions, added to
    /// the raw score) or with `HtsatAudio`'s Swin shift-window convention
    /// (`-100.0` added across an entire disallowed row). Reproduces both on
    /// a `[rows, cols]` grid of otherwise-uniform `[-10, 10)` scores: an
    /// upper-triangular `f32::MIN` causal mask, AND one additional row fully
    /// masked with `-100.0`. (`f32::MIN + score` for a `[-10, 10)` score
    /// stays exactly `f32::MIN`: the ULP at that magnitude, ~4e31, dwarfs any
    /// realistic raw score, so this does NOT reach a non-finite softmax
    /// input — it exercises the still-finite extreme-magnitude corner that
    /// the max-shift step of both softmax arms must handle identically.)
    #[test]
    fn softmax_last_dim_and_composed_softmax_agree_on_masked_production_domain() {
        let device = Device::Cpu;
        for (rows, cols) in [(8usize, 8usize), (37usize, 64usize)] {
            let mut state: u32 = 11;
            let n = rows * cols;
            let mut values: Vec<f32> = (0..n)
                .map(|_| {
                    state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    let unit = (state >> 8) as f32 / (1u32 << 24) as f32;
                    (unit - 0.5) * 20.0
                })
                .collect();

            // Causal mask: additive f32::MIN strictly above the diagonal —
            // clip_text.rs's exact convention (see its causal_mask doc).
            for row in 0..rows {
                for col in 0..cols {
                    if col > row {
                        values[row * cols + col] += f32::MIN;
                    }
                }
            }
            // A fully window-masked row (Swin shift-window convention:
            // -100.0 added across the whole disallowed row).
            let masked_row = rows - 1;
            for col in 0..cols {
                values[masked_row * cols + col] += -100.0f32;
            }

            assert!(
                values.contains(&f32::MIN),
                "[{rows},{cols}]: this fixture must actually reach the f32::MIN-masked corner, \
                 or it is not exercising the domain this test exists for"
            );

            let x = Tensor::from_vec(values, (rows, cols), &device).unwrap();
            let fused = attention_softmax(&x, false).unwrap();
            let composed = attention_softmax(&x, true).unwrap();
            let fused_v = fused.to_vec2::<f32>().unwrap();
            let composed_v = composed.to_vec2::<f32>().unwrap();

            // NaN != NaN under `==`, so a plain assert_eq! on rows containing
            // NaN would silently pass past a divergence — compare bit
            // patterns instead so a NaN vs NaN mismatch (different payload,
            // or NaN vs a finite value) is caught the same as any other
            // divergence.
            let fused_bits: Vec<u32> = fused_v.iter().flatten().map(|v| v.to_bits()).collect();
            let composed_bits: Vec<u32> =
                composed_v.iter().flatten().map(|v| v.to_bits()).collect();
            assert_eq!(
                fused_bits, composed_bits,
                "[{rows},{cols}]: fused vs composed softmax must agree bit-for-bit on the masked \
                 production domain (f32::MIN causal mask + a fully -100.0-masked row); a \
                 divergence here means the two arms are NOT interchangeable on real masked \
                 inputs and `training`'s CPU byte-identity claim must be narrowed to the \
                 unmasked case only"
            );
        }
    }
}
