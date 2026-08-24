//! Closed-enum dispatch over the encoder families used by `jammi-ai`.
//!
//! The three BERT-family encoders ([`Bert`], [`DistilBert`], [`ModernBert`])
//! share a uniform contract: a `[batch, seq, hidden]` `forward_hidden` plus
//! LoRA-aware training-mode hooks. [`ClipText`] is included for callers that
//! want to hand around any supported encoder, but only its pooled `forward`
//! and `hidden_size` are meaningful — it has no per-token hidden output
//! exposed through this enum, and the OpenCLIP text tower is frozen (no
//! LoRA wrapping in this version), so the training-mode methods are no-ops.

use std::collections::HashMap;

use candle_core::Tensor;

use crate::bert::Bert;
use crate::clip_text::ClipText;
use crate::distilbert::DistilBert;
use crate::error::EncoderError;
use crate::modernbert::ModernBert;

/// Family-erased encoder for callers that need to hand around any of the
/// supported encoder types without trait-object overhead.
pub enum AnyEncoder {
    Bert(Bert),
    DistilBert(DistilBert),
    ModernBert(ModernBert),
    /// OpenCLIP text tower. Produces shared-latent `[batch, embed_dim]`
    /// outputs from `forward`; the per-token hidden states and training
    /// hooks of the BERT-family variants are not exposed for this variant.
    ClipText(ClipText),
}

impl AnyEncoder {
    /// Pooled `[batch, output_dim]` embedding. For BERT-family variants the
    /// output dim is `hidden_size`; for [`Self::ClipText`] it is the shared
    /// CLIP latent `embed_dim`.
    pub fn forward(&self, input_ids: &Tensor, mask: &Tensor) -> Result<Tensor, EncoderError> {
        match self {
            Self::Bert(e) => e.forward(input_ids, mask),
            Self::DistilBert(e) => e.forward(input_ids, mask),
            Self::ModernBert(e) => e.forward(input_ids, mask),
            Self::ClipText(e) => e.forward(input_ids, mask),
        }
    }

    /// Per-token `[batch, seq, hidden]` hidden states. Only the BERT-family
    /// variants expose this; the OpenCLIP text tower returns its pooled
    /// projected output and has no peer hidden-state output through this
    /// enum.
    pub fn forward_hidden(
        &self,
        input_ids: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor, EncoderError> {
        match self {
            Self::Bert(e) => e.forward_hidden(input_ids, mask),
            Self::DistilBert(e) => e.forward_hidden(input_ids, mask),
            Self::ModernBert(e) => e.forward_hidden(input_ids, mask),
            Self::ClipText(_) => Err(EncoderError::Config(
                "ClipText does not expose forward_hidden; use forward for pooled CLIP embeddings"
                    .into(),
            )),
        }
    }

    /// Maximum input sequence length. For BERT-family variants this is
    /// `max_position_embeddings`; for [`Self::ClipText`] it is the fixed
    /// OpenCLIP `context_length` (typically 77).
    pub fn max_seq_length(&self) -> usize {
        match self {
            Self::Bert(e) => e.max_seq_length(),
            Self::DistilBert(e) => e.max_seq_length(),
            Self::ModernBert(e) => e.max_seq_length(),
            Self::ClipText(e) => e.context_length(),
        }
    }

    /// Output dimensionality of [`Self::forward`].
    pub fn hidden_size(&self) -> usize {
        match self {
            Self::Bert(e) => e.hidden_size(),
            Self::DistilBert(e) => e.hidden_size(),
            Self::ModernBert(e) => e.hidden_size(),
            Self::ClipText(e) => e.embed_dim(),
        }
    }

    pub fn trainable_params(&self) -> Vec<&Tensor> {
        match self {
            Self::Bert(e) => e.trainable_params(),
            Self::DistilBert(e) => e.trainable_params(),
            Self::ModernBert(e) => e.trainable_params(),
            Self::ClipText(_) => Vec::new(),
        }
    }

    pub fn named_trainable_weights(&self) -> Result<HashMap<String, Tensor>, EncoderError> {
        match self {
            Self::Bert(e) => e.named_trainable_weights(),
            Self::DistilBert(e) => e.named_trainable_weights(),
            Self::ModernBert(e) => e.named_trainable_weights(),
            Self::ClipText(_) => Ok(HashMap::new()),
        }
    }

    pub fn set_training(&mut self, training: bool) {
        match self {
            Self::Bert(e) => e.set_training(training),
            Self::DistilBert(e) => e.set_training(training),
            Self::ModernBert(e) => e.set_training(training),
            // ClipText has no LoRA-wrapped params to gate, but its attention
            // softmax still has a training-only differentiable arm (see
            // `ClipText::set_training`'s doc) — forward the flag so backward
            // through a frozen tower's activations stays correct even though
            // no callers install trainable weights on it today.
            Self::ClipText(e) => e.set_training(training),
        }
    }

    pub fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<(), EncoderError> {
        match self {
            Self::Bert(e) => e.load_weights(weights),
            Self::DistilBert(e) => e.load_weights(weights),
            Self::ModernBert(e) => e.load_weights(weights),
            Self::ClipText(_) => Ok(()),
        }
    }

    /// Per-site dropout-stream positions for every LoRA-wrapped linear, keyed by
    /// the same site names [`Self::named_trainable_weights`] uses — the resume
    /// state for the adapter's dropout so a resumed run replays each stream to its
    /// epoch-boundary position. Empty for a backbone with no installed adapters or
    /// no dropout.
    pub fn dropout_positions(&self) -> Result<HashMap<String, u64>, EncoderError> {
        match self {
            Self::Bert(e) => e.dropout_positions(),
            Self::DistilBert(e) => e.dropout_positions(),
            Self::ModernBert(e) => e.dropout_positions(),
            Self::ClipText(_) => Ok(HashMap::new()),
        }
    }

    /// Restore each LoRA site's dropout-stream position from a
    /// [`Self::dropout_positions`]-shaped map. Missing keys are no-ops.
    pub fn restore_dropout_positions(
        &self,
        positions: &HashMap<String, u64>,
    ) -> Result<(), EncoderError> {
        match self {
            Self::Bert(e) => e.restore_dropout_positions(positions),
            Self::DistilBert(e) => e.restore_dropout_positions(positions),
            Self::ModernBert(e) => e.restore_dropout_positions(positions),
            Self::ClipText(_) => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip_text::{ClipText, ClipTextConfig};
    use crate::test_support::{assert_finite_nonzero, deterministic_fill_varmap, find_var};
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;
    use candle_nn::VarMap;

    fn tiny_config() -> ClipTextConfig {
        ClipTextConfig {
            context_length: 16,
            vocab_size: 64,
            width: 16,
            layers: 2,
            heads: 2,
            embed_dim: 8,
        }
    }

    fn fixed_batch(cfg: &ClipTextConfig, device: &Device) -> (Tensor, Tensor) {
        let ids: Vec<u32> = vec![
            1,
            2,
            3,
            4,
            (cfg.vocab_size - 1) as u32, // EOT at index 4
            5,
            6,
            (cfg.vocab_size - 1) as u32,
            0,
            0, // EOT at index 2, padded
        ];
        let input_ids = Tensor::from_vec(ids, (2, 5), device).unwrap();
        let mask = Tensor::ones((2, 5), DType::U32, device).unwrap();
        (input_ids, mask)
    }

    fn nonuniform_loss(out: &Tensor, channels: usize, device: &Device) -> Tensor {
        let weights: Vec<f32> = (0..channels).map(|i| 1.0 + i as f32 * 0.37).collect();
        let weights = Tensor::from_vec(weights, channels, device).unwrap();
        out.broadcast_mul(&weights).unwrap().sum_all().unwrap()
    }

    fn slice_grad_norm(grad: &Tensor, row_start: usize, width: usize) -> f32 {
        grad.narrow(0, row_start, width)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt()
    }

    /// Deletion-catching oracle for [`AnyEncoder::set_training`]'s
    /// `Self::ClipText(e) => e.set_training(training)` arm — the ONLY
    /// production entry point that reaches `ClipText`'s training flag
    /// through this enum (`AnyEncoder` is what `jammi-ai` actually holds).
    /// If that line regressed to `Self::ClipText(_) => {}` (a silent no-op,
    /// exactly matching this enum's other frozen-tower arms like
    /// `load_weights`), `ClipText` would stay stuck on its default
    /// `training=false` regardless of what the caller asked for.
    ///
    /// `training=true` through `AnyEncoder`: a forward+backward through
    /// `AnyEncoder::forward` (which for `ClipText` IS the full public
    /// `ClipText::forward`, `ln_final` and all — unlike `ClipText`'s own
    /// `#[cfg(test)]`-only `block0_backward` helper, which deliberately
    /// stops short of `ln_final` to isolate the softmax site — see that
    /// helper's doc) reaches layer-0's Q/K slices of `in_proj_weight` with
    /// a finite, nonzero gradient. `training=false` (the default, no
    /// `set_training` call): `in_proj_weight` gets NO gradient entry AT
    /// ALL, not merely zero Q/K rows — `ln_final`'s own `BackpropOp::none()`
    /// truncates backward before it reaches ANY block, matching `ClipText`'s
    /// own full-forward eval oracle
    /// (`training_false_full_forward_grads_are_none_before_ln_final`), NOT
    /// its block-level one (which bypasses `ln_final` and therefore still
    /// sees a partial, V-slice-only gradient — a materially different
    /// fixture from what `AnyEncoder::forward` actually runs).
    ///
    /// RED-verified: reverting `Self::ClipText(e) => e.set_training(training)`
    /// to `Self::ClipText(_) => {}` flips the training=true half of this
    /// test (`in_proj_weight` comes back with NO gradient entry at all,
    /// same as the eval half, since the tower never actually leaves eval
    /// mode) while the training=false half stays green (both are already
    /// `training=false` in that case).
    #[test]
    fn any_encoder_set_training_reaches_clip_text_q_k_gradient() {
        let device = Device::Cpu;
        let cfg = tiny_config();

        // training = true.
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let clip = ClipText::load(vb, &cfg).unwrap();
        deterministic_fill_varmap(&varmap, &device);
        let mut any = AnyEncoder::ClipText(clip);
        any.set_training(true);

        let (input_ids, mask) = fixed_batch(&cfg, &device);
        let out = any.forward(&input_ids, &mask).unwrap();
        let loss = nonuniform_loss(&out, cfg.embed_dim, &device);
        let grads = loss.backward().unwrap();
        let in_proj_weight = find_var(&varmap, "resblocks.0.attn.in_proj_weight");
        let grad = grads
            .get(in_proj_weight.as_tensor())
            .expect("in_proj_weight must have a gradient under training=true");
        let width = cfg.width;
        let q_norm = slice_grad_norm(grad, 0, width);
        let k_norm = slice_grad_norm(grad, width, width);
        assert_finite_nonzero(q_norm, "Q slice (via AnyEncoder::set_training(true))");
        assert_finite_nonzero(k_norm, "K slice (via AnyEncoder::set_training(true))");

        // training = false (the default; no set_training call at all).
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let clip = ClipText::load(vb, &cfg).unwrap();
        deterministic_fill_varmap(&varmap, &device);
        let any = AnyEncoder::ClipText(clip);

        let (input_ids, mask) = fixed_batch(&cfg, &device);
        let out = any.forward(&input_ids, &mask).unwrap();
        let loss = nonuniform_loss(&out, cfg.embed_dim, &device);
        let grads = loss.backward().unwrap();
        let in_proj_weight = find_var(&varmap, "resblocks.0.attn.in_proj_weight");
        assert!(
            grads.get(in_proj_weight.as_tensor()).is_none(),
            "in_proj_weight grad must be None under eval (default, no set_training call) \
             through AnyEncoder::forward's full public forward — ln_final truncates backward \
             before it reaches any block, not merely zeroing Q/K rows"
        );
    }
}
