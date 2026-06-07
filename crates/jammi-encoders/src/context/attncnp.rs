//! Attentive Conditional Neural Process — attention pooling over the context.
//!
//! Where [`super::Cnp`] pools the context with a fixed mean, AttnCnp lets the
//! *target* decide which context members matter: the target query attends over
//! the context keys, and the attention-weighted context values are the pooled
//! representation. This is S19's payoff over fixed pooling — "which context
//! matters" is learned, not assumed uniform.
//!
//! - query: a projection of `target_x`, one row per episode.
//! - keys: a per-member key projection of `context_x`.
//! - values: a per-member value projection of `(context_x ‖ context_y)`.
//!
//! The attention is mask-aware ([`super::multi_head_attention`] with the
//! additive presence mask), so a padded context member receives ≈ zero weight
//! and is never attended over.
//!
//! # Empty-context prior path
//!
//! A learned **prior token** (one key/value pair, always present) is prepended
//! to every context. When the real context is empty (all members masked) the
//! query attends entirely to the prior — a finite, learned fallback rather than
//! a `0/0` over an all-masked softmax. When the context is non-empty the prior
//! is one more key among many; the network learns how much to lean on it, which
//! *is* the epistemic behaviour (lean on the prior when the context is sparse).

use candle_core::{DType, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

use super::{
    attention, linear_over_seq, presence_to_additive_mask, ContextEpisode, ContextPredictorConfig,
    Mlp,
};
use crate::error::EncoderError;

/// Attentive CNP: target-query × context-key/value cross-attention pooling → ρ.
pub struct AttnCnp {
    /// Projects `target_x` to the query space (`hidden_dim`).
    query_proj: Linear,
    /// Projects `context_x` to the key space (`hidden_dim`).
    key_proj: Linear,
    /// Projects `(context_x ‖ context_y)` to the value space (`hidden_dim`).
    value_proj: Linear,
    /// Learned prior key/value, each `[1, 1, hidden_dim]`, broadcast per episode.
    prior_key: Tensor,
    prior_value: Tensor,
    /// Decoder over `(attended ‖ target_x)` → `head_width`.
    rho: Mlp,
    num_heads: usize,
}

impl AttnCnp {
    /// Build the AttnCnp. The query/key/value projections share `hidden_dim`;
    /// the prior key/value are registered as trainable tensors.
    pub fn new(cfg: &ContextPredictorConfig, vb: VarBuilder) -> Result<Self, EncoderError> {
        if cfg.hidden_dim % cfg.num_heads != 0 {
            return Err(EncoderError::Config(format!(
                "AttnCnp: hidden_dim {} not divisible by num_heads {}",
                cfg.hidden_dim, cfg.num_heads
            )));
        }
        let query_proj = linear(cfg.feature_dim, cfg.hidden_dim, vb.pp("query"))?;
        let key_proj = linear(cfg.feature_dim, cfg.hidden_dim, vb.pp("key"))?;
        let value_proj = linear(
            cfg.feature_dim + cfg.value_dim,
            cfg.hidden_dim,
            vb.pp("value"),
        )?;
        let prior_key = vb.get((1, 1, cfg.hidden_dim), "prior_key")?;
        let prior_value = vb.get((1, 1, cfg.hidden_dim), "prior_value")?;
        let rho = Mlp::new(
            cfg.hidden_dim + cfg.feature_dim,
            cfg.hidden_dim,
            cfg.head_width,
            vb.pp("rho"),
        )?;
        Ok(Self {
            query_proj,
            key_proj,
            value_proj,
            prior_key,
            prior_value,
            rho,
            num_heads: cfg.num_heads,
        })
    }

    /// Project the context into `(keys, values)`, each `[B, k+1, hidden]`, with
    /// the learned prior prepended as an always-present member, and the matching
    /// additive `[B, 1, 1, k+1]` mask. Shared by `forward` and the
    /// attention-weight inspection path.
    fn project(
        &self,
        episode: &ContextEpisode,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor), EncoderError> {
        let (b, _k, _f, _v) = episode.dims()?;
        let device = episode.target_x.device();

        let query = self.query_proj.forward(&episode.target_x)?.unsqueeze(1)?; // [B, 1, hidden]
        let context_keys = linear_over_seq(&self.key_proj, &episode.context_x)?; // [B, k, hidden]
        let context_xy = Tensor::cat(&[&episode.context_x, &episode.context_y], 2)?;
        let context_values = linear_over_seq(&self.value_proj, &context_xy)?; // [B, k, hidden]

        // Prepend the always-present prior token to keys and values.
        let prior_key = self
            .prior_key
            .broadcast_as((b, 1, self.prior_key.dim(2)?))?;
        let prior_value = self
            .prior_value
            .broadcast_as((b, 1, self.prior_value.dim(2)?))?;
        let keys = Tensor::cat(&[&prior_key, &context_keys], 1)?; // [B, k+1, hidden]
        let values = Tensor::cat(&[&prior_value, &context_values], 1)?;

        // Presence with the prior's leading always-present column, then to the
        // additive mask so the real padded members get ≈ zero weight while the
        // prior is always attendable.
        let presence_f = episode.presence.to_dtype(DType::F32)?; // [B, k]
        let prior_present = Tensor::ones((b, 1), DType::F32, device)?;
        let presence = Tensor::cat(&[&prior_present, &presence_f], 1)?; // [B, k+1]
        let mask = presence_to_additive_mask(&presence)?; // [B, 1, 1, k+1]

        Ok((query, keys, values, mask))
    }

    /// Predict the `[B, head_width]` head for the episode.
    pub fn forward(&self, episode: &ContextEpisode) -> Result<Tensor, EncoderError> {
        let (query, keys, values, mask) = self.project(episode)?;
        let attended =
            attention::multi_head_attention(&query, &keys, &values, Some(&mask), self.num_heads)?;
        let attended = attended.squeeze(1)?; // [B, hidden]
        let decoder_in = Tensor::cat(&[&attended, &episode.target_x], 1)?;
        self.rho.forward(&decoder_in)
    }

    /// The single-head attention weights over `(prior ∪ context)`, shape
    /// `[B, 1, k+1]` (column 0 is the prior). Exposed so a test / caller can
    /// inspect which context member the target concentrates on.
    pub fn attention_over_context(&self, episode: &ContextEpisode) -> Result<Tensor, EncoderError> {
        let (query, keys, _values, mask) = self.project(episode)?;
        attention::attention_weights(&query, &keys, Some(&mask))
    }

    /// Projection, prior, and ρ parameters.
    pub fn trainable_params(&self) -> Vec<&Tensor> {
        let mut p = vec![
            self.query_proj.weight(),
            self.key_proj.weight(),
            self.value_proj.weight(),
            &self.prior_key,
            &self.prior_value,
        ];
        if let Some(b) = self.query_proj.bias() {
            p.push(b);
        }
        if let Some(b) = self.key_proj.bias() {
            p.push(b);
        }
        if let Some(b) = self.value_proj.bias() {
            p.push(b);
        }
        p.extend(self.rho.trainable_params());
        p
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{cfg, episode, randomize};
    use super::super::ContextArchitecture;
    use super::*;
    use candle_core::{DType, Device, IndexOp};
    use candle_nn::VarMap;

    fn build(head_width: usize) -> (AttnCnp, VarMap, Device) {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = AttnCnp::new(&cfg(ContextArchitecture::AttnCnp, head_width), vb).unwrap();
        randomize(&varmap, &device);
        (model, varmap, device)
    }

    /// Permuting the k context members leaves the AttnCnp head unchanged: the
    /// attention is a softmax-weighted sum over members, which is order-free.
    #[test]
    fn permutation_invariant_over_context() {
        let (model, _vm, device) = build(2);
        let ep = episode(3, 4, 3, 1, &device);
        let base = model.forward(&ep).unwrap();

        let idx = Tensor::from_vec(vec![2u32, 0, 3, 1], (4,), &device).unwrap();
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
                "AttnCnp not permutation-invariant: {x} vs {y}"
            );
        }
    }

    /// An all-masked context routes entirely to the learned prior — finite head,
    /// no NaN over the all-masked softmax.
    #[test]
    fn empty_context_uses_prior_and_is_finite() {
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

        // The prior (column 0) should carry essentially all the weight when the
        // real context is empty.
        let weights = model.attention_over_context(&ep).unwrap(); // [B, 1, k+1]
        let w = weights.i((0, 0)).unwrap().to_vec1::<f32>().unwrap();
        assert!(
            w[0] > 0.99,
            "empty context should attend to the prior (col 0), got {w:?}"
        );
    }

    /// `k = 0` (no context columns) routes to the prior token alone — finite
    /// head, no NaN.
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

    /// Selectivity smoke test: with one context member planted to match the
    /// target far more than the others, the attention weight concentrates on it
    /// — the "which context matters" payoff. A soft check, not a training claim:
    /// we use the raw projections, so we plant the match in the *key* space by
    /// making one member's features equal to the target's.
    /// Set the query/key projections to the identity and the prior key to zero,
    /// so the projected key space equals the raw feature space. This isolates
    /// the attention *mechanism* under a known projection — exactly what a smoke
    /// check of "which context matters" needs (the projection itself is learned
    /// in real training; here we hold it fixed to read the selectivity).
    fn set_identity_projections(varmap: &VarMap, device: &Device) {
        let data = varmap.data().lock().unwrap();
        for (name, var) in data.iter() {
            if name == "query.weight" || name == "key.weight" {
                // Identity-on-the-feature-block, zeros elsewhere: a hidden×feature
                // weight whose top feature_dim rows are the identity, so the
                // projected query/key reproduce the raw features in the first
                // feature_dim lanes.
                let (hidden, fdim) = var.dims2().unwrap();
                let mut w = vec![0f32; hidden * fdim];
                for i in 0..fdim.min(hidden) {
                    w[i * fdim + i] = 1.0;
                }
                var.set(&Tensor::from_vec(w, (hidden, fdim), device).unwrap())
                    .unwrap();
            } else if name == "query.bias" || name == "key.bias" || name == "prior_key" {
                var.set(&Tensor::zeros(var.shape().clone(), DType::F32, device).unwrap())
                    .unwrap();
            }
        }
    }

    /// Selectivity smoke test: with one context member's features planted to
    /// equal the target's (and the projections held to the identity), the
    /// target's attention concentrates on that member — the "which context
    /// matters" payoff. A soft check of the mechanism, not a training claim.
    #[test]
    fn attention_concentrates_on_a_planted_member() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = AttnCnp::new(&cfg(ContextArchitecture::AttnCnp, 2), vb).unwrap();
        randomize(&varmap, &device);
        set_identity_projections(&varmap, &device);

        // One episode, k=4. Member 2's features equal the target's; the others
        // are unrelated. Under the identity projection the dot-product score is
        // largest for member 2, so the softmax concentrates there.
        let target = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &device).unwrap();
        let members = vec![
            0.1f32, -0.2, 0.0, // m0
            -0.3, 0.05, 0.1, // m1
            1.0, 2.0, 3.0, // m2 == target
            0.0, -0.1, 0.2, // m3
        ];
        let context_x = Tensor::from_vec(members, (1, 4, 3), &device).unwrap();
        let context_y = Tensor::randn(0f32, 1.0, (1, 4, 1), &device).unwrap();
        let presence = Tensor::ones((1, 4), DType::F32, &device).unwrap();
        let ep = ContextEpisode {
            target_x: target,
            context_x,
            context_y,
            presence,
        };
        let weights = model.attention_over_context(&ep).unwrap(); // [1, 1, k+1]
        let w = weights.i((0, 0)).unwrap().to_vec1::<f32>().unwrap();
        // Column 0 is the (zeroed) prior; columns 1..=4 are members 0..=3.
        let member_weights = &w[1..];
        let argmax = member_weights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(
            argmax, 2,
            "attention should concentrate on the planted member (idx 2), weights {member_weights:?}"
        );
    }
}
