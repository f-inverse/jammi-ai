//! Head assemblies over an encoder's pooled output: a single projection
//! layer for an embedding fine-tune over a frozen encoder, or a projection
//! plus a task layer for classification and regression over either encoder.
//! Every builder returns a [`LoraModel`] — a flat sequence of named
//! [`jammi_lora::LoraLinear`] layers ([`HeadLayer`]) that the trainer and
//! serving apply in order.

use candle_core::Tensor;
use candle_nn::{Linear, VarBuilder, VarMap};
use jammi_db::error::{JammiError, Result};
use jammi_lora::LoraLinear;

use super::target::HeadLayer;

/// A sequence of named LoRA layers — the trainable head of a
/// [`crate::fine_tune::target::TrainingTarget`].
pub struct LoraModel {
    /// LoRA layers keyed by their name (e.g. `"projection"`, `"classifier"`).
    pub layers: Vec<(String, LoraLinear)>,
}

impl LoraModel {
    /// Collect all trainable parameters across all LoRA layers.
    pub fn trainable_params(&self) -> Vec<&Tensor> {
        self.layers
            .iter()
            .flat_map(|(_, layer)| layer.trainable_params())
            .collect()
    }
}

/// Build one `ZerosB` head LoRA layer at `vb.pp(name)`, carrying
/// `config.lora_dropout` (seeded dropout). The `varmap` receives the seeded
/// trainable A/B tensors. Centralising this keeps every head builder's
/// per-layer construction identical — seed and dropout thread through one
/// place.
///
/// The weight init is always keyed by `config.seed` alone (identical on
/// every rank of a real gang: every rank starts with identical weights);
/// `dropout_seed` is INDEPENDENT of it —
/// `RankContext::dropout_seed(config.seed)` / the free
/// `rank_dropout_seed(config.seed, rank)` (`trainer.rs`) at a real gang's
/// per-rank call site, and `config.seed` itself (rank 0's identity) at
/// every `build_*_head` function's own W=1 convenience wrapper below — see
/// [`jammi_lora::LoraLinear::new_seeded`]'s doc for why the two draws must
/// use independent seeds.
fn build_head_layer_for_rank(
    base: Linear,
    config: &super::FineTuneConfig,
    varmap: &VarMap,
    vb: &VarBuilder,
    name: &str,
    dropout_seed: u64,
) -> Result<LoraLinear> {
    let dropout = if config.lora_dropout > 0.0 {
        Some(config.lora_dropout as f32)
    } else {
        None
    };
    LoraLinear::new_seeded(
        base,
        config.lora_rank,
        config.lora_alpha,
        config.use_rslora,
        jammi_lora::LoraInitMode::ZerosB,
        dropout,
        config.seed,
        dropout_seed,
        varmap,
        &vb.pp(name),
    )
    .map_err(|e| JammiError::FineTune(format!("{name} LoRA: {e}")))
}

/// Build the head `layers` names over a `hidden`-wide pooled output, each
/// layer's base from [`HeadLayer::base`] and its LoRA from
/// `build_head_layer_for_rank`. A task layer's output width is `task_width`.
fn build_head_for_rank(
    layers: &[HeadLayer],
    hidden_size: usize,
    task_width: usize,
    config: &super::FineTuneConfig,
    varmap: &VarMap,
    vb: &VarBuilder,
    dropout_seed: u64,
) -> Result<LoraModel> {
    let layers = layers
        .iter()
        .map(|&layer| {
            let out = match layer {
                HeadLayer::Projection => hidden_size,
                HeadLayer::Classifier | HeadLayer::Distribution => task_width,
            };
            let base = layer
                .base(out, hidden_size, vb.device())
                .map_err(|e| JammiError::FineTune(format!("{} base: {e}", layer.name())))?;
            let lora =
                build_head_layer_for_rank(base, config, varmap, vb, layer.name(), dropout_seed)?;
            Ok((layer.name().to_string(), lora))
        })
        .collect::<Result<_>>()?;
    Ok(LoraModel { layers })
}

/// Build a projection-plus-classifier head for classification fine-tunes.
///
/// Layer 0 (`projection`): LoRA-wrapped identity, `hidden → hidden`.
/// Layer 1 (`classifier`): LoRA-wrapped zeros, `hidden → num_classes`.
pub fn build_classification_head(
    hidden_size: usize,
    num_classes: usize,
    config: &super::FineTuneConfig,
    varmap: &VarMap,
    vb: &VarBuilder,
) -> Result<LoraModel> {
    build_classification_head_for_rank(hidden_size, num_classes, config, varmap, vb, config.seed)
}

/// [`build_classification_head`] with every layer's dropout mask
/// keyed by `dropout_seed` — see `build_head_layer_for_rank`'s doc.
pub fn build_classification_head_for_rank(
    hidden_size: usize,
    num_classes: usize,
    config: &super::FineTuneConfig,
    varmap: &VarMap,
    vb: &VarBuilder,
    dropout_seed: u64,
) -> Result<LoraModel> {
    build_head_for_rank(
        &[HeadLayer::Projection, HeadLayer::Classifier],
        hidden_size,
        num_classes,
        config,
        varmap,
        vb,
        dropout_seed,
    )
}

/// Build a projection-plus-distribution head for regression fine-tunes.
///
/// Layer 0 (`projection`): LoRA-wrapped identity, `hidden → hidden`.
/// Layer 1 (`distribution`): LoRA-wrapped zeros, `hidden → output_dim`, where
/// `output_dim` is the number of distribution parameters — `2` for the
/// parametric Gaussian head `(mean, raw_std)`, or one per level for the quantile
/// head. The zeros base means the head starts emitting `0`s, so the served
/// `σ = floor + softplus(0)` is a finite, non-collapsed default before training.
pub fn build_distribution_head(
    hidden_size: usize,
    output_dim: usize,
    config: &super::FineTuneConfig,
    varmap: &VarMap,
    vb: &VarBuilder,
) -> Result<LoraModel> {
    build_distribution_head_for_rank(hidden_size, output_dim, config, varmap, vb, config.seed)
}

/// [`build_distribution_head`] with every layer's dropout mask
/// keyed by `dropout_seed` — see `build_head_layer_for_rank`'s doc.
pub fn build_distribution_head_for_rank(
    hidden_size: usize,
    output_dim: usize,
    config: &super::FineTuneConfig,
    varmap: &VarMap,
    vb: &VarBuilder,
    dropout_seed: u64,
) -> Result<LoraModel> {
    build_head_for_rank(
        &[HeadLayer::Projection, HeadLayer::Distribution],
        hidden_size,
        output_dim,
        config,
        varmap,
        vb,
        dropout_seed,
    )
}

/// Build a single-layer head for embedding fine-tunes.
///
/// One `projection` LoRA layer wrapping an identity weight
/// (`hidden → hidden`). At init, `B = 0` and the layer acts as identity,
/// so the fine-tuned model starts producing the same embeddings as the
/// frozen base.
pub fn build_projection_head(
    hidden_size: usize,
    config: &super::FineTuneConfig,
    varmap: &VarMap,
    vb: &VarBuilder,
) -> Result<LoraModel> {
    build_projection_head_for_rank(hidden_size, config, varmap, vb, config.seed)
}

/// [`build_projection_head`] with the dropout mask keyed by
/// `dropout_seed` — see `build_head_layer_for_rank`'s doc. The ONE layer
/// this head builds still inits from `config.seed` alone (identical on
/// every rank), so two ranks built through this function with the SAME
/// `config` and DIFFERENT `dropout_seed` start with byte-identical
/// `lora_a`/`lora_b` weights and diverge only in their dropout Philox
/// stream — the property `gang_determinism_oracle`'s
/// `dropout_seed_split_leaves_init_identical_and_dropout_distinct` pins
/// directly.
pub fn build_projection_head_for_rank(
    hidden_size: usize,
    config: &super::FineTuneConfig,
    varmap: &VarMap,
    vb: &VarBuilder,
    dropout_seed: u64,
) -> Result<LoraModel> {
    build_head_for_rank(
        &[HeadLayer::Projection],
        hidden_size,
        hidden_size,
        config,
        varmap,
        vb,
        dropout_seed,
    )
}
