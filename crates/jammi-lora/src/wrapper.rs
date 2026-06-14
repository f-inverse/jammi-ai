//! Closed-enum dispatch between a plain frozen `Linear` and a LoRA-wrapped one.

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module};

use crate::error::LoraError;
use crate::lora_linear::LoraLinear;

/// Either a frozen `candle_nn::Linear` or a LoRA-augmented one.
///
/// Construction is the only place callers decide which arm applies; once built,
/// the rest of the model holds an opaque `MaybeLoraLinear` and forwards through
/// it without branching.
pub enum MaybeLoraLinear {
    /// Plain frozen linear — no LoRA adapter installed.
    Frozen(Linear),
    /// Frozen base wrapped with a trainable LoRA A/B path.
    Lora(LoraLinear),
}

impl MaybeLoraLinear {
    /// Dispatch the forward pass to the appropriate variant.
    ///
    /// The `Frozen` arm casts the input to the weight's dtype so the underlying
    /// matmul sees matching precisions — this matters when a BF16 backbone is
    /// driven by an F32 input.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, LoraError> {
        match self {
            Self::Frozen(l) => {
                let w_dtype = l.weight().dtype();
                let x_cast = if x.dtype() != w_dtype {
                    x.to_dtype(w_dtype)?
                } else {
                    x.clone()
                };
                Ok(l.forward(&x_cast)?)
            }
            Self::Lora(l) => l.forward(x),
        }
    }

    /// Trainable parameters of this layer. Empty for `Frozen`; the LoRA A and
    /// B tensors for `Lora`.
    pub fn trainable_params(&self) -> Vec<&Tensor> {
        match self {
            Self::Frozen(_) => vec![],
            Self::Lora(l) => l.trainable_params(),
        }
    }

    /// Export the LoRA A and B tensors keyed as `{prefix}.lora_a` /
    /// `{prefix}.lora_b`, moved to CPU for safetensors serialisation.
    /// Returns an empty map for `Frozen`.
    pub fn named_weights(&self, prefix: &str) -> Result<HashMap<String, Tensor>, LoraError> {
        let mut out = HashMap::new();
        if let Self::Lora(l) = self {
            out.insert(
                format!("{prefix}.lora_a"),
                l.lora_a.to_device(&Device::Cpu)?,
            );
            out.insert(
                format!("{prefix}.lora_b"),
                l.lora_b.to_device(&Device::Cpu)?,
            );
        }
        Ok(out)
    }

    /// Toggle training mode; no-op on `Frozen`.
    pub fn set_training(&mut self, training: bool) {
        if let Self::Lora(l) = self {
            l.set_training(training);
        }
    }

    /// Restore the LoRA A and B tensors from a `{prefix}.lora_a` /
    /// `{prefix}.lora_b` pair in `weights`. Missing keys are silently ignored
    /// — the caller controls which prefixes they expect to populate. No-op on
    /// `Frozen`.
    pub fn load_weights(&mut self, weights: &HashMap<String, Tensor>, prefix: &str) {
        if let Self::Lora(l) = self {
            if let Some(a) = weights.get(&format!("{prefix}.lora_a")) {
                l.lora_a = a.clone();
            }
            if let Some(b) = weights.get(&format!("{prefix}.lora_b")) {
                l.lora_b = b.clone();
            }
        }
    }

    /// Insert this layer's dropout-stream position keyed `{prefix}.dropout` into
    /// `out`, if it has a dropout stream. No-op for `Frozen` and for a LoRA layer
    /// with `lora_dropout == 0` (no stream to resume).
    pub fn collect_dropout_position(
        &self,
        prefix: &str,
        out: &mut HashMap<String, u64>,
    ) -> Result<(), LoraError> {
        if let Self::Lora(l) = self {
            if let Some(pos) = l.dropout_position()? {
                out.insert(format!("{prefix}.dropout"), pos);
            }
        }
        Ok(())
    }

    /// Restore this layer's dropout-stream position from `{prefix}.dropout` in
    /// `positions`, if present. No-op for `Frozen` and when the key is absent.
    pub fn restore_dropout_position(
        &self,
        prefix: &str,
        positions: &HashMap<String, u64>,
    ) -> Result<(), LoraError> {
        if let Self::Lora(l) = self {
            if let Some(pos) = positions.get(&format!("{prefix}.dropout")) {
                l.restore_dropout_position(*pos)?;
            }
        }
        Ok(())
    }
}
