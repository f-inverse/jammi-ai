//! Transformer Neural Process — self-attention over `(context ∪ target)`.
//!
//! The strongest member: every context member and the target are embedded as
//! tokens of one set, and `num_layers` of masked self-attention let the target
//! token attend over the context (and the context over itself). The target
//! token's final representation is decoded to the predictive head — the
//! prior-fitted-network / TabPFN-style point of the spectrum.
//!
//! Each block is pre-normalised (see [`TnpLayer`]) and a final norm precedes the
//! head. Tokens carry **no positional encoding**, so the set is order-free: permuting
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
use candle_nn::{linear, linear_no_bias, Linear, Module, VarBuilder};

use super::{
    attention, linear_over_seq, presence_to_additive_mask, ContextEpisode, ContextPredictorConfig,
    Mlp,
};
use crate::error::EncoderError;
use crate::layer_norm::LayerNorm;

/// The layer norms' epsilon — PyTorch's `nn.LayerNorm` default, which GPT-2's
/// blocks use.
const NORM_EPS: f64 = 1e-5;

/// One pre-norm transformer block: `tokens + attn(norm(tokens))`, then
/// `tokens + mlp(norm(tokens))` — the Pre-LN placement (Xiong et al. 2020, *On
/// Layer Normalization in the Transformer Architecture*; GPT-2's).
///
/// The normalisation is an invariant of the member, not a tuning choice. A
/// residual stack without it amplifies rounding: over 180 optimizer steps at a
/// learning rate of `5e-3`, the same two-layer stack without these norms turned
/// a one-ulp difference in one weight into a loss difference of `1.9e-1`
/// (×10 every ~23 steps, in `f32` and in `f64` alike), so two trainings from
/// identical weights and batches could not be paired beyond a few dozen steps
/// — not across stacks, not even against a one-ulp copy of themselves. The
/// key projection carries **no bias**: every key of a block goes through it,
/// so a key bias adds the same `q·b` to every score of a query's row, and
/// softmax discards a per-row constant — the bias cannot change the output,
/// its true gradient is zero, and Adam would walk its rounding residue at full
/// size. `AttnCnp`'s key projection keeps its bias: its prior key is not
/// projected, so there the bias does move the prior's score against the
/// members'.
struct TnpLayer {
    attn_norm: LayerNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    mlp_norm: LayerNorm,
    mlp: Mlp,
    num_heads: usize,
}

impl TnpLayer {
    fn new(hidden: usize, num_heads: usize, vb: VarBuilder) -> Result<Self, EncoderError> {
        Ok(Self {
            attn_norm: LayerNorm::new(hidden, NORM_EPS, true, vb.pp("attn_norm"))?,
            q_proj: linear(hidden, hidden, vb.pp("q"))?,
            k_proj: linear_no_bias(hidden, hidden, vb.pp("k"))?,
            v_proj: linear(hidden, hidden, vb.pp("v"))?,
            mlp_norm: LayerNorm::new(hidden, NORM_EPS, true, vb.pp("mlp_norm"))?,
            mlp: Mlp::new(hidden, hidden, hidden, vb.pp("mlp"))?,
            num_heads,
        })
    }

    /// `tokens`: `[B, S, hidden]`; `mask`: additive `[B, 1, 1, S]`.
    fn forward(&self, tokens: &Tensor, mask: &Tensor) -> Result<Tensor, EncoderError> {
        let normed = self.attn_norm.forward(tokens)?;
        let q = self.q_proj.forward(&normed)?;
        let k = self.k_proj.forward(&normed)?;
        let v = self.v_proj.forward(&normed)?;
        let attended = attention::multi_head_attention(&q, &k, &v, Some(mask), self.num_heads)?;
        let tokens = (tokens + attended)?;
        let ff = self.mlp.forward(&self.mlp_norm.forward(&tokens)?)?;
        Ok((&tokens + ff)?)
    }

    fn set_training(&mut self, training: bool) {
        self.attn_norm.set_training(training);
        self.mlp_norm.set_training(training);
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
        for norm in [&self.attn_norm, &self.mlp_norm] {
            p.push(norm.weight());
            p.extend(norm.bias());
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
    /// The final norm before the head (the Pre-LN transformer's closing norm).
    final_norm: LayerNorm,
    /// Decodes the target token's final representation to the head.
    head: Mlp,
}

impl Tnp {
    /// Build the TNP with `cfg.num_layers` self-attention layers.
    pub fn new(cfg: &ContextPredictorConfig, vb: VarBuilder) -> Result<Self, EncoderError> {
        if !cfg.hidden_dim.is_multiple_of(cfg.num_heads) {
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
        let final_norm = LayerNorm::new(cfg.hidden_dim, NORM_EPS, true, vb.pp("final_norm"))?;
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
            final_norm,
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

        // Read the target token (position 0), norm it, and decode it.
        let target_out = hidden.narrow(1, 0, 1)?.squeeze(1)?; // [B, hidden]
        self.head.forward(&self.final_norm.forward(&target_out)?)
    }

    /// Switch every layer norm between its eval forward and its
    /// gradient-carrying training forward; a training step needs the latter.
    pub fn set_training(&mut self, training: bool) {
        for layer in &mut self.layers {
            layer.set_training(training);
        }
        self.final_norm.set_training(training);
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
        p.push(self.final_norm.weight());
        p.extend(self.final_norm.bias());
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

    /// The parameter set is exactly the pre-norm block's: biased `q`/`v`, a
    /// bias-free `k` (softmax discards the per-row constant a key bias would
    /// add), a norm before the attention and one before the MLP, and a final
    /// norm — every one of them trainable.
    #[test]
    fn parameter_set_is_the_pre_norm_blocks() {
        let (model, varmap, _device) = build(2);
        let mut names: Vec<String> = varmap.data().lock().unwrap().keys().cloned().collect();
        names.sort();
        for expected in [
            "layer.0.k.weight",
            "layer.0.q.bias",
            "layer.0.attn_norm.weight",
            "layer.0.attn_norm.bias",
            "layer.0.mlp_norm.weight",
            "layer.1.mlp_norm.bias",
            "final_norm.weight",
            "final_norm.bias",
        ] {
            assert!(
                names.iter().any(|n| n == expected),
                "{expected} missing from {names:?}"
            );
        }
        assert!(!names.iter().any(|n| n.ends_with(".k.bias")), "{names:?}");
        assert_eq!(model.trainable_params().len(), names.len());
    }

    /// In training mode a loss over the head reaches every parameter through
    /// the norms — the gradient-carrying LayerNorm forward is what a training
    /// step runs, and nothing is left without a gradient.
    #[test]
    fn training_mode_backward_reaches_every_parameter() {
        // The training forward reaches the fused LayerNorm seam, whose
        // process-wide dispatch counters every writer serialises on.
        let _seam = crate::test_support::seam_counter_lock();
        let (mut model, varmap, device) = build(2);
        model.set_training(true);
        let ep = episode(3, 4, 3, 1, &device);
        let loss = model
            .forward(&ep)
            .unwrap()
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap();
        let grads = loss.backward().unwrap();
        for (name, var) in varmap.data().lock().unwrap().iter() {
            assert!(
                grads.get(var.as_tensor()).is_some(),
                "{name} received no gradient"
            );
        }
    }
}
