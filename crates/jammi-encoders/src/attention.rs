//! The fused-QKV multi-head attention every cross-modal tower in this crate
//! composes, and the one attention-probabilities softmax it takes.
//!
//! [`attention_softmax`] is `candle_nn::ops::softmax(scores, D::Minus1)`, the
//! max/sub/exp/sum/div composition whose backward reaches every operand.
//! `candle_nn::ops::softmax_last_dim` is not used anywhere in this crate: it
//! is applied through `Tensor::apply_op1_no_bwd`, so its result carries
//! `BackpropOp::none()` and a backward walk stops there silently — every
//! operand strictly upstream of the softmax (the Q/K slices of a fused
//! `in_proj_weight`, a relative-position bias table read only into the
//! pre-softmax scores) would come back with an exactly-zero or missing
//! gradient, never an error. One forward serves training, evaluation and
//! serving, so the softmax it takes must be the differentiable one.

use candle_core::{IndexOp, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder};
use jammi_lora::{FrozenBase, MaybeLoraLinear};

use crate::error::EncoderError;
use crate::lora_site::LoraSite;

/// The attention-probabilities softmax — see this module's doc for why it
/// is the composed form.
pub fn attention_softmax(scores: &Tensor) -> Result<Tensor, EncoderError> {
    Ok(candle_nn::ops::softmax(scores, D::Minus1)?)
}

/// The two LoRA-wrappable linear sites of [`MultiHeadAttention`], in the
/// order their names appear in a checkpoint. Both the selector name a caller
/// writes in `target_modules` AND the adapter subpath are this leaf name —
/// the checkpoint's own vocabulary, never a consumer's.
pub(crate) const IN_PROJ_SITE: &str = "in_proj";
/// See [`IN_PROJ_SITE`].
pub(crate) const OUT_PROJ_SITE: &str = "out_proj";

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
/// contiguous-izes both its operands unconditionally (one implicit
/// contiguous copy per operand, made inside the matmul primitive, not two).
pub(crate) struct MultiHeadAttention {
    /// The FUSED QKV projection as ONE LoRA site. Q/K/V are three row
    /// ranges of a single `[3*width, width]` weight in the OpenCLIP
    /// checkpoint layout, not three separate modules, so there is nothing
    /// to split: one adapter on the fused weight adapts all three
    /// projections jointly. `jammi_lora`'s own primitives place no
    /// `out == in` constraint on a base (`FrozenBase` derives both from the
    /// weight's dims), and PEFT adapts `nn.MultiheadAttention`'s
    /// `in_proj_weight` as one parameter the same way.
    in_proj: MaybeLoraLinear,
    out_proj: MaybeLoraLinear,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    /// Resolve the two frozen bases from the checkpoint. The ONE place the
    /// base-tensor locators live (`crate::lora_site`'s module doc, axis 3):
    /// `in_proj` is the FLAT `in_proj_weight`/`in_proj_bias` pair read
    /// straight off `vb` — there is no `in_proj` sub-module to descend into
    /// — while `out_proj` is an ordinary sub-module linear. Shared by
    /// [`Self::load`] and [`Self::load_with`] so the frozen and
    /// LoRA-capable constructions can never drift in what they read.
    fn load_bases(vb: &VarBuilder, width: usize) -> Result<(FrozenBase, FrozenBase), EncoderError> {
        let in_proj_weight = vb.get((width * 3, width), "in_proj_weight")?;
        let in_proj_bias = vb.get(width * 3, "in_proj_bias")?;
        let in_proj = Linear::new(in_proj_weight, Some(in_proj_bias));
        let out_proj = linear(width, width, vb.pp(OUT_PROJ_SITE))?;
        Ok((FrozenBase::Dense(in_proj), FrozenBase::Dense(out_proj)))
    }

    /// Fully frozen construction — no adapter on either site. Routed through
    /// [`Self::load_with`] with a [`FrozenSiteHolder`]'s decline-everything
    /// site, so there is exactly ONE construction path and an unselected
    /// site is `MaybeLoraLinear::Frozen(FrozenBase::Dense(linear))`, whose
    /// forward is `Linear::forward` unchanged.
    ///
    /// `#[cfg(test)]`-only: both owning towers (`crate::clip_text`,
    /// `crate::open_clip_vision`) construct through [`Self::load_with`] —
    /// their own frozen `load` entry points supply the decline-everything
    /// site once, at the TOWER level, rather than each block re-deriving
    /// one. This crate's test modules build a bare `MultiHeadAttention`
    /// directly and have no `LoraSite` to hand it; this constructor keeps
    /// those attention-level oracles on the same frozen construction the
    /// towers use.
    #[cfg(test)]
    pub(crate) fn load(
        vb: VarBuilder,
        width: usize,
        num_heads: usize,
    ) -> Result<Self, EncoderError> {
        let holder = crate::lora_site::FrozenSiteHolder::new();
        let site = holder.site(&vb);
        Self::load_with(vb.clone(), width, num_heads, &site)
    }

    /// Same bases as [`Self::load`], each offered to `site` for LoRA
    /// wrapping under its own leaf name ([`IN_PROJ_SITE`] /
    /// [`OUT_PROJ_SITE`], used as BOTH the selector name and the adapter
    /// subpath). With a `LoraBuildConfig::frozen()` config every site is
    /// declined and the result is exactly [`Self::load`]'s.
    pub(crate) fn load_with(
        vb: VarBuilder,
        width: usize,
        num_heads: usize,
        site: &LoraSite<'_>,
    ) -> Result<Self, EncoderError> {
        let (in_proj_base, out_proj_base) = Self::load_bases(&vb, width)?;
        Ok(Self {
            in_proj: site.wrap(in_proj_base, IN_PROJ_SITE, IN_PROJ_SITE)?,
            out_proj: site.wrap(out_proj_base, OUT_PROJ_SITE, OUT_PROJ_SITE)?,
            num_heads,
            head_dim: width / num_heads,
        })
    }

    /// Propagate the training parameter to the two LoRA sites — see
    /// `jammi_lora::LoraLinear::set_training` for what it governs.
    pub(crate) fn set_training(&mut self, training: bool) {
        for (_, site) in self.lora_sites_mut() {
            site.set_training(training);
        }
    }

    /// This module's LoRA sites paired with their names — the single source
    /// of the site→name map every owning tower's weight / dropout-position /
    /// restore traversal walks, so a tower never re-spells these leaves.
    pub(crate) fn lora_sites(&self) -> [(&'static str, &MaybeLoraLinear); 2] {
        [
            (IN_PROJ_SITE, &self.in_proj),
            (OUT_PROJ_SITE, &self.out_proj),
        ]
    }

    /// The `&mut` twin of [`Self::lora_sites`], over the same names in the
    /// same order.
    pub(crate) fn lora_sites_mut(&mut self) -> [(&'static str, &mut MaybeLoraLinear); 2] {
        [
            (IN_PROJ_SITE, &mut self.in_proj),
            (OUT_PROJ_SITE, &mut self.out_proj),
        ]
    }

    /// The fused `[3*width, width]` QKV projection weight — Q/K/V occupy
    /// row ranges `[0, width)` / `[width, 2*width)` / `[2*width, 3*width)`
    /// respectively. Exposed so a caller's own tests can inspect its
    /// gradient (e.g. the Q/K-vs-V backward-truncation oracles) without
    /// this crate-private struct's fields being `pub(crate)` individually.
    /// Reads through `MaybeLoraLinear::base`, so it returns the SAME frozen
    /// weight whether or not an adapter is installed on the site.
    /// `#[cfg(test)]`-only: no production call site needs this, only the
    /// `open_clip_vision`/`clip_text` test modules that construct this
    /// struct directly.
    #[cfg(test)]
    pub(crate) fn in_proj_weight(&self) -> &Tensor {
        match self.in_proj.base() {
            FrozenBase::Dense(l) => l.weight(),
            FrozenBase::Quantized(_) => panic!(
                "in_proj_weight: this tower never constructs a GGUF-quantized fused-QKV base \
                 (no weight_source seam is wired here), so a Quantized arm is unreachable"
            ),
        }
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
        let attn_weights = attention_softmax(&attn_scores)?;
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

    /// `MultiHeadAttention::forward`'s op sequence with three redundant
    /// `qkv.i(_)?.contiguous()?` calls put back in that the primary forward
    /// path omits (`contiguous_matmul` already contiguous-izes both its
    /// operands, so those calls are redundant copies, not a correctness
    /// dependency) — used only by
    /// [`tests::dropping_the_redundant_contiguous_calls_does_not_change_output`]
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
        let attn_weights = attention_softmax(&attn_scores)?;
        let attn_output = crate::contiguous_matmul(&attn_weights, &v)?;

        let attn_output = attn_output.permute((0, 2, 1, 3))?.reshape((
            batch,
            seq_len,
            attn.num_heads * attn.head_dim,
        ))?;

        Ok(attn.out_proj.forward(&attn_output)?)
    }

    /// MEASURED: omitting explicit `qkv.i(_)?.contiguous()?` calls (since
    /// [`crate::contiguous_matmul`] already contiguous-izes both its
    /// operands) changes NO output byte — `.contiguous()` only ever
    /// materializes a data layout, never a value, so contiguous-izing an
    /// operand once (inside `contiguous_matmul`) versus twice (a true no-op
    /// the second time) cannot change the numbers either way. Covers both the causally-masked shape
    /// (`clip_text`'s) and the unmasked shape (`open_clip_vision`'s), at an
    /// odd `seq_len` so no accidental symmetry could mask a real
    /// divergence.
    #[test]
    fn dropping_the_redundant_contiguous_calls_does_not_change_output() {
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
}
