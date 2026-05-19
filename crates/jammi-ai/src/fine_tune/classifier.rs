//! Classification head for fine-tuning a sequence classifier on top of a
//! jammi-encoders ModernBERT backbone.

use candle_core::{Result as CandleResult, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use jammi_encoders::ModernBert;

pub struct SeqClassifier {
    backbone: ModernBert,
    head: Linear,
    num_classes: usize,
}

impl SeqClassifier {
    pub fn new(backbone: ModernBert, num_classes: usize, vb: VarBuilder) -> CandleResult<Self> {
        let head = linear(backbone.hidden_size(), num_classes, vb.pp("classifier"))?;
        Ok(Self {
            backbone,
            head,
            num_classes,
        })
    }

    /// CLS-token classification: pull the first-token hidden state and project to logits.
    pub fn forward(&self, input_ids: &Tensor, mask: &Tensor) -> CandleResult<Tensor> {
        let hidden = self
            .backbone
            .forward_hidden(input_ids, mask)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let cls = hidden.narrow(1, 0, 1)?.squeeze(1)?;
        self.head.forward(&cls)
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}
