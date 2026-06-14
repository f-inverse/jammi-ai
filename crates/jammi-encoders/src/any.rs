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
            Self::ClipText(_) => {}
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
