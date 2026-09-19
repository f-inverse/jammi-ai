//! Projection-head assemblies for
//! [`crate::fine_tune::target::TrainingTarget::ProjectionHead`]: a single
//! projection layer for embedding fine-tunes, or a projection plus a
//! classifier head for classification / NER. All three builders return a
//! [`LoraModel`] — a flat sequence of named [`jammi_lora::LoraLinear`]
//! layers that the trainer consumes uniformly.

use candle_core::{DType, Tensor};
use candle_nn::{Linear, VarBuilder, VarMap};
use jammi_db::error::{JammiError, Result};
use jammi_lora::LoraLinear;

/// A sequence of named LoRA layers — the trainable state of a
/// [`TrainingTarget::ProjectionHead`] target.
///
/// [`TrainingTarget::ProjectionHead`]: crate::fine_tune::target::TrainingTarget::ProjectionHead
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
    // Layer 0: projection (identity base, same as embedding)
    let proj_base = Tensor::eye(hidden_size, DType::F32, vb.device())
        .map_err(|e| JammiError::FineTune(format!("Projection identity: {e}")))?;
    let proj_linear = Linear::new(proj_base, None);
    let projection =
        build_head_layer_for_rank(proj_linear, config, varmap, vb, "projection", dropout_seed)?;

    // Layer 1: classifier (zeros base, trained from scratch via LoRA)
    let cls_base = Tensor::zeros((num_classes, hidden_size), DType::F32, vb.device())
        .map_err(|e| JammiError::FineTune(format!("Classifier zeros: {e}")))?;
    let cls_linear = Linear::new(cls_base, None);
    let classifier =
        build_head_layer_for_rank(cls_linear, config, varmap, vb, "classifier", dropout_seed)?;

    Ok(LoraModel {
        layers: vec![
            ("projection".to_string(), projection),
            ("classifier".to_string(), classifier),
        ],
    })
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
    let proj_base = Tensor::eye(hidden_size, DType::F32, vb.device())
        .map_err(|e| JammiError::FineTune(format!("Regression projection identity: {e}")))?;
    let proj_linear = Linear::new(proj_base, None);
    let projection =
        build_head_layer_for_rank(proj_linear, config, varmap, vb, "projection", dropout_seed)?;

    let head_base = Tensor::zeros((output_dim, hidden_size), DType::F32, vb.device())
        .map_err(|e| JammiError::FineTune(format!("Distribution head zeros: {e}")))?;
    let head_linear = Linear::new(head_base, None);
    let distribution = build_head_layer_for_rank(
        head_linear,
        config,
        varmap,
        vb,
        "distribution",
        dropout_seed,
    )?;

    Ok(LoraModel {
        layers: vec![
            ("projection".to_string(), projection),
            ("distribution".to_string(), distribution),
        ],
    })
}

/// Build a projection-plus-token-classifier head for NER fine-tunes.
///
/// Layer 0 (`projection`): LoRA-wrapped identity, `hidden → hidden`, per token.
/// Layer 1 (`token_classifier`): LoRA-wrapped zeros, `hidden → num_labels`, per token.
pub fn build_ner_head(
    hidden_size: usize,
    num_labels: usize,
    config: &super::FineTuneConfig,
    varmap: &VarMap,
    vb: &VarBuilder,
) -> Result<LoraModel> {
    build_ner_head_for_rank(hidden_size, num_labels, config, varmap, vb, config.seed)
}

/// [`build_ner_head`] with every layer's dropout mask keyed by
/// `dropout_seed` — see `build_head_layer_for_rank`'s doc.
pub fn build_ner_head_for_rank(
    hidden_size: usize,
    num_labels: usize,
    config: &super::FineTuneConfig,
    varmap: &VarMap,
    vb: &VarBuilder,
    dropout_seed: u64,
) -> Result<LoraModel> {
    // Layer 0: projection (identity base, same as embedding/classification)
    let proj_base = Tensor::eye(hidden_size, DType::F32, vb.device())
        .map_err(|e| JammiError::FineTune(format!("NER projection identity: {e}")))?;
    let proj_linear = Linear::new(proj_base, None);
    let projection =
        build_head_layer_for_rank(proj_linear, config, varmap, vb, "projection", dropout_seed)?;

    // Layer 1: token classifier (zeros base, trained from scratch via LoRA)
    let cls_base = Tensor::zeros((num_labels, hidden_size), DType::F32, vb.device())
        .map_err(|e| JammiError::FineTune(format!("NER classifier zeros: {e}")))?;
    let cls_linear = Linear::new(cls_base, None);
    let classifier = build_head_layer_for_rank(
        cls_linear,
        config,
        varmap,
        vb,
        "token_classifier",
        dropout_seed,
    )?;

    Ok(LoraModel {
        layers: vec![
            ("projection".to_string(), projection),
            ("token_classifier".to_string(), classifier),
        ],
    })
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
    let base_weight = Tensor::eye(hidden_size, DType::F32, vb.device())
        .map_err(|e| JammiError::FineTune(format!("Identity weight: {e}")))?;
    let base = Linear::new(base_weight, None);
    let lora = build_head_layer_for_rank(base, config, varmap, vb, "projection", dropout_seed)?;
    Ok(LoraModel {
        layers: vec![("projection".into(), lora)],
    })
}
