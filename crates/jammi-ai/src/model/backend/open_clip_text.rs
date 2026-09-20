//! OpenCLIP text-tower wrapper that adapts [`jammi_encoders::ClipText`] to
//! the text-forward contract used by the Candle backend.
//!
//! The OpenCLIP text encoder emits a pooled, L2-normalized `[batch, embed_dim]`
//! tensor in the shared CLIP latent space; this wrapper exposes it via
//! `forward_pooled` and rejects the per-token `forward_hidden` and
//! classification/NER paths (those are BERT-family specific).

use candle_core::{Device, Tensor};
use jammi_db::error::{JammiError, Result};
use jammi_encoders::{ClipText, Pooling};

use crate::model::tokenizer::BatchEncoding;

use super::candle::CandleTextForward;

/// Wraps a loaded [`ClipText`] so the candle backend can dispatch text
/// queries through the same `CandleTextForward` interface as BERT-family
/// encoders.
pub struct OpenClipTextForward(pub ClipText);

impl CandleTextForward for OpenClipTextForward {
    fn max_sequence_length(&self) -> usize {
        self.0.context_length()
    }

    fn forward_hidden(
        &self,
        _input_ids: &Tensor,
        _attention_mask: &Tensor,
        _encoding: &BatchEncoding,
        _device: &Device,
    ) -> Result<Tensor> {
        // The OpenCLIP text tower does not expose per-token hidden states
        // through this contract; only the projected pooled output is
        // meaningful, which the embedding path obtains via `forward_pooled`.
        Err(JammiError::Inference(
            "OpenCLIP text encoder does not support forward_hidden \
                 (classification / NER are BERT-family only)"
                .into(),
        ))
    }

    fn forward_pooled(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        _encoding: &BatchEncoding,
        _device: &Device,
    ) -> Result<Tensor> {
        self.0
            .forward(input_ids, attention_mask)
            .map_err(|e| JammiError::Inference(format!("OpenCLIP text forward failed: {e}")))
    }

    /// `None`: [`Self::forward_pooled`] above bypasses pooling entirely (the
    /// OpenCLIP text tower's output is already pooled-and-projected into the
    /// shared CLIP latent space by `ClipText::forward`, with no separate
    /// mean/CLS reduction step this wrapper applies), so there is no
    /// [`jammi_encoders::Pooling`] strategy to report. Stated explicitly
    /// rather than inherited from the trait default because
    /// `CandleTextForward::forward_pooled`'s pairing rule requires every
    /// wrapper that overrides ONE of the `forward_pooled` /
    /// `resolved_pooling` pair to state BOTH at its own `impl` block, as the
    /// classification wrappers' `classification_resolved_pooling()` does.
    fn resolved_pooling(&self) -> Option<Pooling> {
        None
    }
}
