//! Closed-enum dispatch between a plain frozen base ([`FrozenBase`]) and a
//! LoRA-wrapped one.

use std::collections::HashMap;

use candle_core::{Device, Tensor};

use crate::error::LoraError;
use crate::frozen_base::FrozenBase;
use crate::lora_linear::LoraLinear;

/// Either a frozen base weight ([`FrozenBase`] — dense or GGUF-quantized) or
/// a LoRA-augmented one.
///
/// Construction is the only place callers decide which arm applies; once built,
/// the rest of the model holds an opaque `MaybeLoraLinear` and forwards through
/// it without branching.
pub enum MaybeLoraLinear {
    /// Plain frozen base — no LoRA adapter installed. Dense or GGUF-quantized
    /// (see [`FrozenBase`]).
    Frozen(FrozenBase),
    /// Frozen base wrapped with a trainable LoRA A/B path.
    Lora(LoraLinear),
}

impl MaybeLoraLinear {
    /// Dispatch the forward pass to the appropriate variant.
    ///
    /// The `Frozen` arm delegates to [`FrozenBase::forward`] — a dense base
    /// casts the input to the weight's dtype so the underlying matmul sees
    /// matching precisions (this matters when a BF16 backbone is driven by an
    /// F32 input; behavior PRESERVED byte-for-byte from every prior release,
    /// see that method's own doc), and a quantized base runs the uniform F32
    /// activation rule (see [`crate::QuantizedLinear`]'s own doc).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, LoraError> {
        match self {
            Self::Frozen(base) => base.forward(x),
            Self::Lora(l) => l.forward(x),
        }
    }

    /// The frozen base weight underneath EITHER arm — the `Frozen` arm's own
    /// base, or the base a `Lora` arm wraps. One accessor for both, so a
    /// consumer that needs the site's base tensor (a test inspecting a fused
    /// QKV weight's gradient, a geometry check) does not have to re-derive
    /// the arm split at every call site and cannot accidentally handle only
    /// the unadapted case.
    pub fn base(&self) -> &FrozenBase {
        match self {
            Self::Frozen(base) => base,
            Self::Lora(l) => l.base(),
        }
    }

    /// Whether this site actually carries an adapter — the `Lora` arm.
    ///
    /// One accessor, for the same reason [`Self::base`] exists: a consumer
    /// that needs to COUNT wrapped sites (a structural census of how many
    /// `lora_linear_fused` admissions one training forward can take, e.g.
    /// `jammi_encoders`' `FusibleSiteCensus`) would otherwise re-derive the
    /// arm split with a `matches!` at every call site. Counting
    /// [`Self::trainable_params`] instead is NOT the same measurement: that
    /// is a count of TENSORS (two per adapted site), and it silently returns
    /// the same `0` for "no adapter installed" as for "an adapter whose A/B
    /// pair is empty" — this answers the structural question directly.
    ///
    /// A `true` here does NOT by itself mean the fused kernel runs: the
    /// adapted site still takes its own admission decision per TRAINING
    /// forward (and takes none at all in eval), and a
    /// [`FrozenBase::Quantized`] base never reaches the fused seam at all
    /// (see [`crate::LoraLinear::forward`]'s own doc). It means exactly that
    /// this site is an adapted one.
    pub fn is_lora(&self) -> bool {
        matches!(self, Self::Lora(_))
    }

    /// Whether this site takes a `lora_linear_fused` admission decision on a
    /// TRAINING forward — `Lora` over a `FrozenBase::Dense` base.
    ///
    /// This is a NARROWER question than [`Self::is_lora`]. `LoraLinear::
    /// forward` branches on `self.base` BEFORE it ever reaches `admit()`
    /// (see that method's own doc, "the fused site is Dense-ONLY"): a `Lora`
    /// site whose base is `FrozenBase::Quantized` — the shape a QLoRA
    /// backbone builds for every adapted site — ALWAYS composes and takes NO
    /// admission decision at all, neither `Fused` nor `Eager`. A `Frozen`
    /// site (either base storage) likewise never calls `LoraLinear::forward`
    /// and so never reaches `admit()` either. This method answers exactly
    /// the predicate a `lora_linear_fused` call-count census needs — see
    /// `jammi_encoders::FusibleSiteCensus::lora_sites_wrapped`'s own doc,
    /// which counts this, not [`Self::is_lora`].
    pub fn takes_lora_linear_admission(&self) -> bool {
        matches!(self, Self::Lora(l) if matches!(l.base(), FrozenBase::Dense(_)))
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

/// [`MaybeLoraLinear::takes_lora_linear_admission`]: the Dense-vs-Quantized
/// split that predicate exists to expose (see #467 F3 — a QLoRA backbone's
/// `FusibleSiteCensus::lora_sites_wrapped` must not count a quantized-base
/// adapted site).
#[cfg(test)]
mod tests {
    use super::*;
    use crate::frozen_base::QuantizedLinear;
    use crate::init::LoraInitMode;
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::DType;
    use candle_nn::{Linear, VarBuilder, VarMap};
    use std::sync::Arc;

    fn dense_lora_site(out_f: usize, in_f: usize) -> MaybeLoraLinear {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let w_v: Vec<f32> = (0..out_f * in_f)
            .map(|i| ((i as f64) * 0.037 + 0.3).sin() as f32)
            .collect();
        let w = Tensor::from_vec(w_v, (out_f, in_f), &device).unwrap();
        let base = Linear::new(w, None);
        let lora = LoraLinear::new(
            base,
            4,
            8.0,
            false,
            LoraInitMode::Gaussian,
            None,
            7,
            &varmap,
            &vb,
        )
        .unwrap();
        MaybeLoraLinear::Lora(lora)
    }

    fn quantized_lora_site(out_f: usize, in_f: usize) -> MaybeLoraLinear {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let w_v: Vec<f32> = (0..out_f * in_f)
            .map(|i| ((i as f64) * 0.029 + 0.7).sin() as f32)
            .collect();
        let w = Tensor::from_vec(w_v, (out_f, in_f), &device).unwrap();
        let q = QTensor::quantize(&w, GgmlDType::Q8_0).unwrap();
        let base = FrozenBase::Quantized(QuantizedLinear::new(Arc::new(q), None).unwrap());
        let lora = LoraLinear::new_with_base(
            base,
            4,
            8.0,
            false,
            LoraInitMode::Gaussian,
            None,
            11,
            &varmap,
            &vb,
        )
        .unwrap();
        MaybeLoraLinear::Lora(lora)
    }

    /// Positive control: a `Lora` site over a `Dense` base — the ordinary
    /// (non-quantized) case every LoRA site used before QLoRA existed — DOES
    /// take the `lora_linear_fused` admission decision.
    #[test]
    fn takes_lora_linear_admission_is_true_for_a_lora_over_dense_base() {
        let site = dense_lora_site(4, 8);
        assert!(site.is_lora(), "sanity: this site is adapted");
        assert!(
            site.takes_lora_linear_admission(),
            "a Lora site over a Dense base takes the lora_linear_fused admission decision"
        );
    }

    /// The domain-validity edge #467 F3 closes: a `Lora` site over a
    /// `Quantized` base is still adapted (`is_lora` stays `true`, this is a
    /// non-vacuous control) but `LoraLinear::forward` composes it
    /// unconditionally and never reaches `admit()` — so it must NOT be
    /// counted as taking the admission decision.
    #[test]
    fn takes_lora_linear_admission_is_false_for_a_lora_over_quantized_base() {
        let site = quantized_lora_site(4, 32);
        assert!(
            site.is_lora(),
            "sanity: this site is adapted -- is_lora must stay true"
        );
        assert!(
            !site.takes_lora_linear_admission(),
            "a Lora site over a Quantized base ALWAYS composes (LoraLinear::forward's \
             Dense-only fused branch) and so never reaches admit() -- must not be counted"
        );
    }

    /// Negative control: an unwrapped `Frozen` site (dense storage) is
    /// neither adapted nor admission-taking.
    #[test]
    fn takes_lora_linear_admission_is_false_for_a_frozen_dense_site() {
        let device = Device::Cpu;
        let w = Tensor::zeros((4, 8), DType::F32, &device).unwrap();
        let site = MaybeLoraLinear::Frozen(FrozenBase::Dense(Linear::new(w, None)));
        assert!(!site.is_lora());
        assert!(!site.takes_lora_linear_admission());
    }
}
