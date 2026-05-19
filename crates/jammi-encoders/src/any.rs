//! Closed-enum dispatch over the three encoder families. Phase B subagents
//! fill in concrete bodies; the match arms are added once each encoder's
//! method signatures are known.

use std::collections::HashMap;

use candle_core::Tensor;

use crate::bert::Bert;
use crate::distilbert::DistilBert;
use crate::error::EncoderError;
use crate::modernbert::ModernBert;

/// Family-erased encoder for callers that need to hand around any of the
/// three concrete encoder types without trait-object overhead.
pub enum AnyEncoder {
    Bert(Bert),
    DistilBert(DistilBert),
    ModernBert(ModernBert),
}

impl AnyEncoder {
    pub fn forward(&self, input_ids: &Tensor, mask: &Tensor) -> Result<Tensor, EncoderError> {
        match self {
            Self::Bert(e) => e.forward(input_ids, mask),
            Self::DistilBert(e) => e.forward(input_ids, mask),
            Self::ModernBert(e) => e.forward(input_ids, mask),
        }
    }

    pub fn forward_hidden(
        &self,
        input_ids: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor, EncoderError> {
        match self {
            Self::Bert(e) => e.forward_hidden(input_ids, mask),
            Self::DistilBert(e) => e.forward_hidden(input_ids, mask),
            Self::ModernBert(e) => e.forward_hidden(input_ids, mask),
        }
    }

    pub fn max_seq_length(&self) -> usize {
        match self {
            Self::Bert(e) => e.max_seq_length(),
            Self::DistilBert(e) => e.max_seq_length(),
            Self::ModernBert(e) => e.max_seq_length(),
        }
    }

    pub fn hidden_size(&self) -> usize {
        match self {
            Self::Bert(e) => e.hidden_size(),
            Self::DistilBert(e) => e.hidden_size(),
            Self::ModernBert(e) => e.hidden_size(),
        }
    }

    pub fn trainable_params(&self) -> Vec<&Tensor> {
        match self {
            Self::Bert(e) => e.trainable_params(),
            Self::DistilBert(e) => e.trainable_params(),
            Self::ModernBert(e) => e.trainable_params(),
        }
    }

    pub fn named_trainable_weights(&self) -> Result<HashMap<String, Tensor>, EncoderError> {
        match self {
            Self::Bert(e) => e.named_trainable_weights(),
            Self::DistilBert(e) => e.named_trainable_weights(),
            Self::ModernBert(e) => e.named_trainable_weights(),
        }
    }

    pub fn set_training(&mut self, training: bool) {
        match self {
            Self::Bert(e) => e.set_training(training),
            Self::DistilBert(e) => e.set_training(training),
            Self::ModernBert(e) => e.set_training(training),
        }
    }

    pub fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<(), EncoderError> {
        match self {
            Self::Bert(e) => e.load_weights(weights),
            Self::DistilBert(e) => e.load_weights(weights),
            Self::ModernBert(e) => e.load_weights(weights),
        }
    }
}
