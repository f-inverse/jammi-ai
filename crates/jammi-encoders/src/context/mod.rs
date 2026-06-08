//! The amortized in-context predictor family — `AnyContextPredictor` — the
//! learned-aggregation point of the neural-process spectrum.
//!
//! Each member conditions on a *context set* (retrieved neighbour features and
//! their outcomes) and emits a predictive-distribution head at a target, in one
//! differentiable forward pass. The three curated members are a closed enum,
//! the same curation bar and tier boundary as [`crate::AnyEncoder`]: a consumer
//! selects a member by config, never authors a novel architecture through it.
//!
//! - [`Cnp`] — DeepSets: a per-member MLP, mean-pooled over the context via the
//!   differentiable [`crate::segment_aggregate`], then a decoder MLP. The
//!   baseline; the learned twin of the data-plane's fixed pooling.
//! - [`AttnCnp`] — attention pooling over the context (the learned "which
//!   context matters"), so the target query weights its relevant neighbours.
//! - [`Tnp`] — a transformer over the `(context ∪ target)` token set; the
//!   strongest member, the prior-fitted-network point.
//!
//! Every member's `forward` returns the *same* `[B, head_width]` float head as
//! the distributional output adapter consumes: `head_width = 2` for a Gaussian
//! `(mean, raw_std)`, `head_width = levels` for a quantile head. One output
//! shape, three architectures behind it.
//!
//! # Context representation — the [`ContextEpisode`] input
//!
//! The three members want the context in two natural forms, and the input type
//! carries the one representation from which both derive, so a caller assembles
//! it once:
//!
//! - a **dense** `[B, k, feature_dim]` / `[B, k, value_dim]` form with a
//!   `[B, k]` presence mask, padded to the per-batch maximum `k`. AttnCnp and
//!   Tnp attend over this with an additive mask (built from the presence mask
//!   via the crate-internal `extended_attention_mask`) so a padded member
//!   receives zero attention weight — never attended over.
//! - a **ragged** form — the masked-in members flattened to `[N, …]` with a
//!   `[N]` segment-id mapping each to its episode `0..B`. [`Cnp`] derives this
//!   from the dense form + presence mask and pools it via
//!   [`crate::segment_aggregate`], whose empty-segment-zero and
//!   permutation-invariance this family relies on. A padded member is simply
//!   never emitted into the ragged form, so it contributes nothing to the pool.
//!
//! An *empty* context (`k == 0`, or an episode whose whole row is masked) is
//! NaN-free for every member: Cnp pools to the documented zero row, AttnCnp
//! routes through a learned prior path, Tnp attends over the lone target token.

use candle_core::Tensor;
use candle_nn::{linear, Linear, Module, VarBuilder};

use crate::error::EncoderError;

mod attention;
mod attncnp;
mod cnp;
mod tnp;

pub use attention::{attention_weights, multi_head_attention};
pub use attncnp::AttnCnp;
pub use cnp::Cnp;
pub use tnp::Tnp;

/// Which curated in-context-predictor architecture a [`ContextPredictorConfig`]
/// selects. A closed enum — the tier boundary. A novel neural-process
/// architecture is authored in an external stack and its predictions registered
/// back, never added here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ContextArchitecture {
    /// Conditional Neural Process: DeepSets encoder, segment-mean pooling.
    Cnp,
    /// Attentive CNP: attention pooling over the context members.
    AttnCnp,
    /// Transformer Neural Process: self-attention over `(context ∪ target)`.
    Tnp,
}

/// Configuration for an [`AnyContextPredictor`]. The consumer sets these; no
/// member exposes a tensor op.
#[derive(Debug, Clone)]
pub struct ContextPredictorConfig {
    /// Which curated architecture to build.
    pub architecture: ContextArchitecture,
    /// The maximum context size the dense form is padded to (the `k` of the
    /// retrieval). Members shorter than this are presence-masked.
    pub context_k: usize,
    /// Dimensionality of a context/target feature vector `x`.
    pub feature_dim: usize,
    /// Dimensionality of a context member's outcome `y`.
    pub value_dim: usize,
    /// Hidden width of the MLPs / transformer model dimension.
    pub hidden_dim: usize,
    /// Number of attention heads (AttnCnp, Tnp). Must divide `hidden_dim`.
    pub num_heads: usize,
    /// Number of transformer layers (Tnp).
    pub num_layers: usize,
    /// Width of the predictive-distribution head: `2` for Gaussian
    /// `(mean, raw_std)`, `levels` for a quantile head. The output of `forward`
    /// is exactly `[B, head_width]`.
    pub head_width: usize,
}

/// A target row plus its (padded) context set — the input to every predictor's
/// `forward`. See the [module docs](self) for the dual dense/ragged
/// representation and why this single type serves all three members.
#[derive(Debug, Clone)]
pub struct ContextEpisode {
    /// Target features, one row per episode: `[B, feature_dim]`.
    pub target_x: Tensor,
    /// Context member features, padded: `[B, k, feature_dim]`.
    pub context_x: Tensor,
    /// Context member outcomes, padded: `[B, k, value_dim]`.
    pub context_y: Tensor,
    /// Presence mask, `[B, k]` (f32 or integer): `1` at a real context member,
    /// `0` at a padded slot. An all-zero row is an empty context.
    pub presence: Tensor,
}

impl ContextEpisode {
    /// `(B, k, feature_dim, value_dim)` from the carried tensors, validating that
    /// the dense shapes and the presence mask agree. A shape mismatch is a typed
    /// error rather than a downstream tensor panic.
    fn dims(&self) -> Result<(usize, usize, usize, usize), EncoderError> {
        let (b, feature_dim) = self.target_x.dims2()?;
        let (bx, k, fx) = self.context_x.dims3()?;
        let (by, ky, value_dim) = self.context_y.dims3()?;
        let (bp, kp) = self.presence.dims2()?;
        if bx != b || by != b || bp != b {
            return Err(EncoderError::Config(format!(
                "ContextEpisode: batch mismatch (target {b}, context_x {bx}, context_y {by}, presence {bp})"
            )));
        }
        if ky != k || kp != k {
            return Err(EncoderError::Config(format!(
                "ContextEpisode: context size mismatch (context_x {k}, context_y {ky}, presence {kp})"
            )));
        }
        if fx != feature_dim {
            return Err(EncoderError::Config(format!(
                "ContextEpisode: feature_dim mismatch (target {feature_dim}, context_x {fx})"
            )));
        }
        Ok((b, k, feature_dim, value_dim))
    }
}

/// Family-erased in-context predictor: a closed enum over the three curated
/// architectures, mirroring [`crate::AnyEncoder`]'s per-variant dispatch.
pub enum AnyContextPredictor {
    /// See [`Cnp`].
    Cnp(Cnp),
    /// See [`AttnCnp`].
    AttnCnp(AttnCnp),
    /// See [`Tnp`].
    Tnp(Tnp),
}

impl AnyContextPredictor {
    /// Build the predictor the config selects, registering its trainable tensors
    /// in `vb`'s [`candle_nn::VarMap`].
    pub fn new(cfg: &ContextPredictorConfig, vb: VarBuilder) -> Result<Self, EncoderError> {
        if cfg.head_width == 0 {
            return Err(EncoderError::Config(
                "ContextPredictorConfig: head_width must be at least 1".into(),
            ));
        }
        Ok(match cfg.architecture {
            ContextArchitecture::Cnp => Self::Cnp(Cnp::new(cfg, vb)?),
            ContextArchitecture::AttnCnp => Self::AttnCnp(AttnCnp::new(cfg, vb)?),
            ContextArchitecture::Tnp => Self::Tnp(Tnp::new(cfg, vb)?),
        })
    }

    /// Predict the `[B, head_width]` distributional head for the episode's
    /// targets given their context. The output is exactly the float head the
    /// distributional output adapter consumes.
    pub fn forward(&self, episode: &ContextEpisode) -> Result<Tensor, EncoderError> {
        match self {
            Self::Cnp(m) => m.forward(episode),
            Self::AttnCnp(m) => m.forward(episode),
            Self::Tnp(m) => m.forward(episode),
        }
    }

    /// Every trainable tensor across the predictor's parameters, for an
    /// optimizer to step. The autograd loop (a later PR) drives this.
    pub fn trainable_params(&self) -> Vec<&Tensor> {
        match self {
            Self::Cnp(m) => m.trainable_params(),
            Self::AttnCnp(m) => m.trainable_params(),
            Self::Tnp(m) => m.trainable_params(),
        }
    }
}

/// Build the additive `[B, 1, 1, k]` attention mask from a `[B, k]` presence
/// mask, so a padded context member receives ≈ zero attention weight. Shared by
/// the attentive members; the same `extended_attention_mask` the encoders use.
pub(crate) fn presence_to_additive_mask(presence: &Tensor) -> Result<Tensor, EncoderError> {
    crate::mask::extended_attention_mask(presence)
}

/// Apply a [`Linear`] over the last axis of a `[B, S, in]` tensor, returning
/// `[B, S, out]`, via an explicit flatten/reshape with **fully-specified**
/// output dims.
///
/// candle's `Linear::forward` reshapes a 3-D input through an *inferred* (`()`)
/// dimension, which it cannot resolve when the sequence axis is `0` (an empty
/// context). Driving the matmul through `[B*S, in] → [B*S, out] → [B, S, out]`
/// with every dim named keeps the empty-context (`S == 0`) path NaN-free and
/// shape-correct, the same shape discipline `Cnp` already relies on for its
/// `[B*k, …]` flatten.
pub(crate) fn linear_over_seq(layer: &Linear, x: &Tensor) -> Result<Tensor, EncoderError> {
    let (b, s, in_dim) = x.dims3()?;
    let flat = x.reshape((b * s, in_dim))?;
    let out = layer.forward(&flat)?;
    let out_dim = out.dim(1)?;
    Ok(out.reshape((b, s, out_dim))?)
}

/// A small two-layer MLP (`Linear → GELU → Linear`) — the φ/ρ building block
/// shared by every member. Construction-time widths only; the forward applies
/// the same elementwise `gelu` the BERT FFN uses.
pub(crate) struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    /// `in_dim → hidden → out_dim`, registering both linears under `vb`.
    pub(crate) fn new(
        in_dim: usize,
        hidden: usize,
        out_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self, EncoderError> {
        Ok(Self {
            fc1: linear(in_dim, hidden, vb.pp("fc1"))?,
            fc2: linear(hidden, out_dim, vb.pp("fc2"))?,
        })
    }

    /// Apply the MLP to a `[.., in_dim]` tensor, returning `[.., out_dim]`.
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        let h = self.fc1.forward(x)?.gelu_erf()?;
        Ok(self.fc2.forward(&h)?)
    }

    /// The two linears' weight (and bias) tensors, for the trainable-param set.
    pub(crate) fn trainable_params(&self) -> Vec<&Tensor> {
        let mut p = vec![self.fc1.weight(), self.fc2.weight()];
        if let Some(b) = self.fc1.bias() {
            p.push(b);
        }
        if let Some(b) = self.fc2.bias() {
            p.push(b);
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarMap;

    /// Replace every var with small random noise so the predictor is
    /// non-degenerate (a fresh `VarMap` is zero-initialised, which would make
    /// every head identical and hide real behaviour).
    pub(super) fn randomize(varmap: &VarMap, device: &Device) {
        let data = varmap.data().lock().unwrap();
        for var in data.values() {
            let r = Tensor::randn(0f32, 0.2, var.shape().clone(), device).unwrap();
            var.set(&r).unwrap();
        }
    }

    /// A synthetic episode: `B` targets, each with `k` context members of the
    /// given dims, all present (no padding). Deterministic ramps so tests can
    /// reason about the values.
    pub(super) fn episode(
        b: usize,
        k: usize,
        feature_dim: usize,
        value_dim: usize,
        device: &Device,
    ) -> ContextEpisode {
        let target_x = Tensor::randn(0f32, 1.0, (b, feature_dim), device).unwrap();
        let context_x = Tensor::randn(0f32, 1.0, (b, k, feature_dim), device).unwrap();
        let context_y = Tensor::randn(0f32, 1.0, (b, k, value_dim), device).unwrap();
        let presence = Tensor::ones((b, k), DType::F32, device).unwrap();
        ContextEpisode {
            target_x,
            context_x,
            context_y,
            presence,
        }
    }

    pub(super) fn cfg(arch: ContextArchitecture, head_width: usize) -> ContextPredictorConfig {
        ContextPredictorConfig {
            architecture: arch,
            context_k: 4,
            feature_dim: 3,
            value_dim: 1,
            hidden_dim: 8,
            num_heads: 2,
            num_layers: 2,
            head_width,
        }
    }

    fn built(arch: ContextArchitecture, head_width: usize) -> (AnyContextPredictor, VarMap) {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let predictor = AnyContextPredictor::new(&cfg(arch, head_width), vb).unwrap();
        randomize(&varmap, &device);
        (predictor, varmap)
    }

    /// Every member emits exactly `[B, head_width]` for both the Gaussian
    /// (`k=2`) and quantile (`k=levels`) head widths — the single S18 head
    /// shape behind all three architectures.
    #[test]
    fn forward_shape_is_batch_by_head_width() {
        let device = Device::Cpu;
        for arch in [
            ContextArchitecture::Cnp,
            ContextArchitecture::AttnCnp,
            ContextArchitecture::Tnp,
        ] {
            for head_width in [2usize, 5] {
                let (predictor, _vm) = built(arch, head_width);
                let ep = episode(6, 4, 3, 1, &device);
                let out = predictor.forward(&ep).unwrap();
                assert_eq!(
                    out.dims2().unwrap(),
                    (6, head_width),
                    "{arch:?} head_width {head_width}"
                );
                assert!(
                    out.flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                        .iter()
                        .all(|x| x.is_finite()),
                    "{arch:?} produced a non-finite head"
                );
            }
        }
    }

    /// `.backward()` over a synthetic loss populates a gradient on *every*
    /// trainable tensor — the family is trainable before the autograd loop is
    /// wired. Checked for all three members.
    #[test]
    fn gradients_flow_to_every_trainable_param() {
        let device = Device::Cpu;
        for arch in [
            ContextArchitecture::Cnp,
            ContextArchitecture::AttnCnp,
            ContextArchitecture::Tnp,
        ] {
            let (predictor, _vm) = built(arch, 2);
            let ep = episode(5, 4, 3, 1, &device);
            let out = predictor.forward(&ep).unwrap();
            // A trivial scalar loss: sum of squares of the head.
            let loss = out.sqr().unwrap().sum_all().unwrap();
            let grads = loss.backward().unwrap();
            let params = predictor.trainable_params();
            assert!(!params.is_empty(), "{arch:?} exposed no trainable params");
            for (i, p) in params.iter().enumerate() {
                assert!(
                    grads.get(p).is_some(),
                    "{arch:?} param {i} received no gradient"
                );
            }
        }
    }
}
