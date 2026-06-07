//! Transformer Neural Process — self-attention over `(context ∪ target)`.
//!
//! The strongest member: every context member and the target are embedded as
//! tokens of one set, and `num_layers` of masked self-attention let the target
//! token attend over the context (and the context over itself). The target
//! token's final representation is decoded to the predictive head — the
//! prior-fitted-network / TabPFN-style point of the spectrum.
//!
//! Tokens carry **no positional encoding**, so the set is order-free: permuting
//! the context tokens permutes the attention rows/columns identically and leaves
//! the target token's output unchanged (permutation-invariance over context).
//!
//! Context and target inhabit the same token space. A context token embeds
//! `(x ‖ y)`; the target has no `y`, so its outcome slot is a learned
//! **query-marker** vector in place of `y` — the network learns to read "this is
//! the token to predict" from the marker, the same role TabPFN's masked target
//! row plays. Padded context members are masked out of every attention row (the
//! additive presence mask), so they are never attended over; an empty context
//! leaves the target attending only to itself — finite, no NaN.

use candle_core::{DType, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

use super::{
    attention, linear_over_seq, presence_to_additive_mask, ContextEpisode, ContextPredictorConfig,
    Mlp,
};
use crate::error::EncoderError;

/// One pre-norm-free transformer block: masked self-attention + residual, then
/// an MLP + residual. Layer norm is omitted deliberately — the family's blocks
/// are small and the residual MLP keeps the forward well-conditioned for the
/// synthetic-tensor tests; the math that matters (masked attention) is shared
/// with the encoders' verified primitive.
struct TnpLayer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    mlp: Mlp,
    num_heads: usize,
}

impl TnpLayer {
    fn new(hidden: usize, num_heads: usize, vb: VarBuilder) -> Result<Self, EncoderError> {
        Ok(Self {
            q_proj: linear(hidden, hidden, vb.pp("q"))?,
            k_proj: linear(hidden, hidden, vb.pp("k"))?,
            v_proj: linear(hidden, hidden, vb.pp("v"))?,
            mlp: Mlp::new(hidden, hidden, hidden, vb.pp("mlp"))?,
            num_heads,
        })
    }

    /// `tokens`: `[B, S, hidden]`; `mask`: additive `[B, 1, 1, S]`.
    fn forward(&self, tokens: &Tensor, mask: &Tensor) -> Result<Tensor, EncoderError> {
        let q = self.q_proj.forward(tokens)?;
        let k = self.k_proj.forward(tokens)?;
        let v = self.v_proj.forward(tokens)?;
        let attended = attention::multi_head_attention(&q, &k, &v, Some(mask), self.num_heads)?;
        let tokens = (tokens + attended)?;
        let ff = self.mlp.forward(&tokens)?;
        Ok((&tokens + ff)?)
    }

    fn trainable_params(&self) -> Vec<&Tensor> {
        let mut p = vec![
            self.q_proj.weight(),
            self.k_proj.weight(),
            self.v_proj.weight(),
        ];
        for proj in [&self.q_proj, &self.k_proj, &self.v_proj] {
            if let Some(b) = proj.bias() {
                p.push(b);
            }
        }
        p.extend(self.mlp.trainable_params());
        p
    }
}

/// Transformer Neural Process: token the context ∪ target, self-attend, decode
/// the target token.
pub struct Tnp {
    /// Embeds a context member `(x ‖ y)` into the token space.
    context_embed: Linear,
    /// Embeds the target `x` into the token space (its `y` slot is the marker).
    target_embed: Linear,
    /// Learned query-marker filling the target's absent outcome slot,
    /// `[1, 1, hidden]`.
    query_marker: Tensor,
    layers: Vec<TnpLayer>,
    /// Decodes the target token's final representation to the head.
    head: Mlp,
}

impl Tnp {
    /// Build the TNP with `cfg.num_layers` self-attention layers.
    pub fn new(cfg: &ContextPredictorConfig, vb: VarBuilder) -> Result<Self, EncoderError> {
        if cfg.hidden_dim % cfg.num_heads != 0 {
            return Err(EncoderError::Config(format!(
                "Tnp: hidden_dim {} not divisible by num_heads {}",
                cfg.hidden_dim, cfg.num_heads
            )));
        }
        let context_embed = linear(
            cfg.feature_dim + cfg.value_dim,
            cfg.hidden_dim,
            vb.pp("context_embed"),
        )?;
        let target_embed = linear(cfg.feature_dim, cfg.hidden_dim, vb.pp("target_embed"))?;
        let query_marker = vb.get((1, 1, cfg.hidden_dim), "query_marker")?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for n in 0..cfg.num_layers {
            layers.push(TnpLayer::new(
                cfg.hidden_dim,
                cfg.num_heads,
                vb.pp(format!("layer.{n}")),
            )?);
        }
        let head = Mlp::new(
            cfg.hidden_dim,
            cfg.hidden_dim,
            cfg.head_width,
            vb.pp("head"),
        )?;
        Ok(Self {
            context_embed,
            target_embed,
            query_marker,
            layers,
            head,
        })
    }

    /// Predict the `[B, head_width]` head for the episode.
    pub fn forward(&self, episode: &ContextEpisode) -> Result<Tensor, EncoderError> {
        let (b, k, _f, _v) = episode.dims()?;
        let device = episode.target_x.device();

        // Context tokens from `(x ‖ y)`, [B, k, hidden].
        let context_xy = Tensor::cat(&[&episode.context_x, &episode.context_y], 2)?;
        let context_tokens = linear_over_seq(&self.context_embed, &context_xy)?;

        // Target token from `x` plus the learned query-marker in its `y` slot.
        // The marker enters additively in the token space so the target embed
        // and the marker share the hidden dim.
        let target_token = self.target_embed.forward(&episode.target_x)?.unsqueeze(1)?; // [B,1,hidden]
        let marker = self
            .query_marker
            .broadcast_as((b, 1, self.query_marker.dim(2)?))?;
        let target_token = (target_token + marker)?;

        // Sequence: target at position 0, then the k context tokens.
        let tokens = Tensor::cat(&[&target_token, &context_tokens], 1)?; // [B, k+1, hidden]

        // Mask: the target (position 0) is always present; context positions
        // follow the presence mask. Padded context tokens get ≈ zero weight in
        // every attention row, so they are never attended over.
        let presence_f = episode.presence.to_dtype(DType::F32)?; // [B, k]
        let target_present = Tensor::ones((b, 1), DType::F32, device)?;
        let presence = Tensor::cat(&[&target_present, &presence_f], 1)?; // [B, k+1]
        let mask = presence_to_additive_mask(&presence)?; // [B, 1, 1, k+1]

        let mut hidden = tokens;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &mask)?;
        }
        debug_assert_eq!(hidden.dim(1)?, k + 1);

        // Read the target token (position 0) and decode it.
        let target_out = hidden.narrow(1, 0, 1)?.squeeze(1)?; // [B, hidden]
        self.head.forward(&target_out)
    }

    /// Embedding, marker, layer, and head parameters.
    pub fn trainable_params(&self) -> Vec<&Tensor> {
        let mut p = vec![
            self.context_embed.weight(),
            self.target_embed.weight(),
            &self.query_marker,
        ];
        for proj in [&self.context_embed, &self.target_embed] {
            if let Some(b) = proj.bias() {
                p.push(b);
            }
        }
        for layer in &self.layers {
            p.extend(layer.trainable_params());
        }
        p.extend(self.head.trainable_params());
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

    fn build(head_width: usize) -> (Tnp, VarMap, Device) {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = Tnp::new(&cfg(ContextArchitecture::Tnp, head_width), vb).unwrap();
        randomize(&varmap, &device);
        (model, varmap, device)
    }

    /// Permuting the context tokens leaves the target token's decoded head
    /// unchanged — the tokens carry no positional encoding, so the set is
    /// order-free.
    #[test]
    fn permutation_invariant_over_context() {
        let (model, _vm, device) = build(2);
        let ep = episode(3, 4, 3, 1, &device);
        let base = model.forward(&ep).unwrap();

        let idx = Tensor::from_vec(vec![1u32, 3, 0, 2], (4,), &device).unwrap();
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
                (x - y).abs() < 1e-4,
                "TNP not permutation-invariant: {x} vs {y}"
            );
        }
    }

    /// An all-masked context leaves the target attending only to itself — finite
    /// head, no NaN over the masked attention rows.
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

    /// `k = 0` (no context tokens) is finite — only the target token in the set.
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
