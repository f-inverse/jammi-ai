//! Conditional Neural Process — the DeepSets baseline of the family.
//!
//! A per-member encoder φ maps each context pair `(x ‖ y)` to a representation;
//! those are **mean-pooled per episode** via the differentiable
//! [`crate::segment_aggregate`] (the same pooling operator as the data-plane
//! UDAF, in the autograd graph); a decoder ρ maps `(pooled ‖ target_x ‖
//! context_size)` to the predictive head. Feeding the context size to ρ gives
//! the decoder the count signal an epistemic predictor needs — a sparse context
//! is distinguishable from a dense one even when their pooled vectors coincide.
//!
//! The pool is permutation-invariant over context members (it inherits
//! `segment_aggregate`'s proven invariance) and NaN-free on an empty context
//! (an episode with no present member pools to the documented zero row).

use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;

use crate::aggregate::{segment_aggregate, SegmentReduce};
use crate::error::EncoderError;

use super::{ContextEpisode, ContextPredictorConfig, Mlp};

/// Conditional Neural Process: φ (per-member) → segment-mean pool → ρ → head.
pub struct Cnp {
    /// Per-member encoder over `(context_x ‖ context_y)` → `hidden_dim`.
    phi: Mlp,
    /// Decoder over `(pooled ‖ target_x ‖ context_size)` → `head_width`.
    rho: Mlp,
}

impl Cnp {
    /// Build the CNP. φ ingests `feature_dim + value_dim`; ρ ingests
    /// `hidden_dim + feature_dim + 1` (the pooled representation, the target
    /// features, and the scalar context size).
    pub fn new(cfg: &ContextPredictorConfig, vb: VarBuilder) -> Result<Self, EncoderError> {
        let phi = Mlp::new(
            cfg.feature_dim + cfg.value_dim,
            cfg.hidden_dim,
            cfg.hidden_dim,
            vb.pp("phi"),
        )?;
        let rho = Mlp::new(
            cfg.hidden_dim + cfg.feature_dim + 1,
            cfg.hidden_dim,
            cfg.head_width,
            vb.pp("rho"),
        )?;
        Ok(Self { phi, rho })
    }

    /// Predict the `[B, head_width]` head for the episode.
    pub fn forward(&self, episode: &ContextEpisode) -> Result<Tensor, EncoderError> {
        let (b, k, feature_dim, value_dim) = episode.dims()?;
        let device = episode.target_x.device();

        // Per-member φ over the concatenated `(x ‖ y)`, on the dense `[B, k, …]`
        // form, then flattened to a ragged `[B*k, hidden]` for the segment pool.
        let context_xy = Tensor::cat(&[&episode.context_x, &episode.context_y], 2)?;
        let phi_dense = self
            .phi
            .forward(&context_xy.reshape((b * k, feature_dim + value_dim))?)?;
        let hidden = phi_dense.dim(1)?;

        // Segment ids route each of the `B*k` slots to its episode `0..B`, EXCEPT
        // padded slots, which route to a sink segment `B`. Requesting `B + 1`
        // segments and dropping the sink means: a padded member folds into the
        // sink (never into a real episode), and `Mean` divides each real episode
        // by exactly its present-member count — the genuine ragged-flatten pool,
        // expressed through `segment_aggregate`. The sink also absorbs the empty
        // context: an episode with no present member has no slot routed to it,
        // so it pools to the documented zero row.
        let presence_f = episode.presence.to_dtype(DType::F32)?; // [B, k]
        let episode_ids = Tensor::arange(0u32, b as u32, device)?
            .reshape((b, 1))?
            .broadcast_as((b, k))?
            .to_dtype(DType::F32)?;
        let sink = Tensor::full(b as f32, (b, k), device)?;
        // present → episode index, padded → sink (B).
        let seg_ids = (episode_ids.broadcast_mul(&presence_f)?
            + sink.broadcast_mul(&(1.0 - &presence_f)?)?)?
        .reshape((b * k,))?
        .to_dtype(DType::U32)?;

        let pooled_with_sink = segment_aggregate(&phi_dense, &seg_ids, b + 1, SegmentReduce::Mean)?;
        // Drop the sink row; keep the `B` real episodes' pooled representations.
        let pooled = pooled_with_sink.narrow(0, 0, b)?; // [B, hidden]
        debug_assert_eq!(pooled.dim(1)?, hidden);

        // The decoder also sees the context size, so a sparse context is
        // distinguishable from a dense one (the epistemic signal).
        let context_size = presence_f.sum_keepdim(1)?; // [B, 1]
        let decoder_in = Tensor::cat(&[&pooled, &episode.target_x, &context_size], 1)?;
        self.rho.forward(&decoder_in)
    }

    /// φ and ρ parameters.
    pub fn trainable_params(&self) -> Vec<&Tensor> {
        let mut p = self.phi.trainable_params();
        p.extend(self.rho.trainable_params());
        p
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{cfg, episode, randomize};
    use super::super::ContextArchitecture;
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    fn build(head_width: usize) -> (Cnp, VarMap, Device) {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = Cnp::new(&cfg(ContextArchitecture::Cnp, head_width), vb).unwrap();
        randomize(&varmap, &device);
        (model, varmap, device)
    }

    /// Permuting the k context members of every episode leaves the CNP head
    /// unchanged — it inherits `segment_aggregate`'s permutation-invariance.
    #[test]
    fn permutation_invariant_over_context() {
        let (model, _vm, device) = build(2);
        let ep = episode(3, 4, 3, 1, &device);
        let base = model.forward(&ep).unwrap();

        // Reverse the context member order on every episode.
        let idx = Tensor::from_vec(vec![3u32, 2, 1, 0], (4,), &device).unwrap();
        let permuted = ContextEpisode {
            target_x: ep.target_x.clone(),
            context_x: ep.context_x.index_select(&idx, 1).unwrap(),
            context_y: ep.context_y.index_select(&idx, 1).unwrap(),
            presence: ep.presence.index_select(&idx, 1).unwrap(),
        };
        let after = model.forward(&permuted).unwrap();

        let a = base.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let c = after.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (x, y) in a.iter().zip(c.iter()) {
            assert!(
                (x - y).abs() < 1e-5,
                "CNP not permutation-invariant: {x} vs {y}"
            );
        }
    }

    /// An all-masked (empty) context yields a finite head with no NaN — it
    /// exercises `segment_aggregate`'s empty-segment-zero path (no episode slot
    /// routes to a real segment, so the pool is the zero row).
    #[test]
    fn empty_context_is_finite() {
        let (model, _vm, device) = build(2);
        let mut ep = episode(3, 4, 3, 1, &device);
        ep.presence = Tensor::zeros((3, 4), DType::F32, &device).unwrap();
        let out = model.forward(&ep).unwrap();
        assert!(out
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|x| x.is_finite()));
    }

    /// `k = 0` (no context columns at all) is also finite — every episode pools
    /// to the zero row.
    #[test]
    fn zero_k_context_is_finite() {
        let (model, _vm, device) = build(2);
        let target_x = Tensor::randn(0f32, 1.0, (3, 3), &device).unwrap();
        let context_x = Tensor::zeros((3, 0, 3), DType::F32, &device).unwrap();
        let context_y = Tensor::zeros((3, 0, 1), DType::F32, &device).unwrap();
        let presence = Tensor::zeros((3, 0), DType::F32, &device).unwrap();
        let ep = ContextEpisode {
            target_x,
            context_x,
            context_y,
            presence,
        };
        let out = model.forward(&ep).unwrap();
        assert_eq!(out.dims2().unwrap(), (3, 2));
        assert!(out
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|x| x.is_finite()));
    }
}
