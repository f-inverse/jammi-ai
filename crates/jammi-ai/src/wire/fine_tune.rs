//! `FineTuneService` proto↔domain conversions.
//!
//! The request `FineTuneConfig` mirrors the engine's [`FineTuneConfig`] field
//! for field; decode maps it (and its nested loss / enum types) onto the engine
//! struct, leaving the engine's defaults for any field left `UNSPECIFIED` (an
//! absent `config` message → the engine default entirely). Validation stays in
//! the engine (`FineTuneConfig::validate`, called inside `fine_tune`); this is a
//! pure shape map. `task` reuses `jammi.v1.inference.ModelTask` via the shared
//! [`super::model_task_from_proto`].

use tonic::Status;

use crate::fine_tune::{
    BackboneDtype, ClassificationLoss, EarlyStoppingMetric, EmbeddingLoss, FineTuneConfig,
    FineTuneMethod, HardNegativeConfig, LoraInitMode, LrSchedule,
};

use crate::wire::proto::fine_tune as pb;

/// Map the wire [`pb::FineTuneMethod`] discriminant onto the engine's
/// [`FineTuneMethod`]. An unspecified or unknown method is rejected — a request
/// that names no method is a client error, not a silent default.
pub fn method_from_proto(method: i32) -> Result<FineTuneMethod, Status> {
    match pb::FineTuneMethod::try_from(method) {
        Ok(pb::FineTuneMethod::Lora) => Ok(FineTuneMethod::Lora),
        Ok(pb::FineTuneMethod::Unspecified) | Err(_) => {
            Err(Status::invalid_argument("method must be specified"))
        }
    }
}

/// Map the wire [`pb::FineTuneConfig`] onto the engine's [`FineTuneConfig`].
///
/// Every numeric field carries straight through. The optional loss messages map
/// to `Option<…Loss>` (unset → engine auto-selects from the data format). The
/// enum-typed fields fall back to the engine default variant when left
/// `UNSPECIFIED`, so a config that sets only the numeric knobs behaves exactly
/// like `FineTuneConfig::default()` for the rest.
impl TryFrom<pb::FineTuneConfig> for FineTuneConfig {
    type Error = Status;

    fn try_from(c: pb::FineTuneConfig) -> Result<Self, Self::Error> {
        let defaults = FineTuneConfig::default();
        Ok(FineTuneConfig {
            lora_rank: c.lora_rank as usize,
            lora_alpha: c.lora_alpha,
            lora_dropout: c.lora_dropout,
            learning_rate: c.learning_rate,
            epochs: c.epochs as usize,
            batch_size: c.batch_size as usize,
            max_seq_length: c.max_seq_length as usize,
            embedding_loss: c
                .embedding_loss
                .map(embedding_loss_from_proto)
                .transpose()?,
            classification_loss: c
                .classification_loss
                .map(classification_loss_from_proto)
                .transpose()?,
            gradient_accumulation_steps: c.gradient_accumulation_steps as usize,
            validation_fraction: c.validation_fraction,
            early_stopping_patience: c.early_stopping_patience as usize,
            warmup_steps: c.warmup_steps as usize,
            lr_schedule: lr_schedule_from_proto(c.lr_schedule, defaults.lr_schedule)?,
            early_stopping_metric: early_stopping_metric_from_proto(
                c.early_stopping_metric,
                defaults.early_stopping_metric,
            )?,
            target_modules: c.target_modules,
            layers_to_transform: c
                .layers_to_transform
                .map(|l| l.layers.into_iter().map(|n| n as usize).collect()),
            use_rslora: c.use_rslora,
            rank_pattern: c
                .rank_pattern
                .into_iter()
                .map(|(k, v)| (k, v as usize))
                .collect(),
            init_lora_weights: lora_init_mode_from_proto(
                c.init_lora_weights,
                defaults.init_lora_weights,
            )?,
            backbone_dtype: backbone_dtype_from_proto(c.backbone_dtype, defaults.backbone_dtype)?,
            weight_decay: c.weight_decay,
            max_grad_norm: c.max_grad_norm,
            cached: c.cached,
            hard_negatives: c
                .hard_negatives
                .map(hard_negatives_from_proto)
                .unwrap_or_default(),
            matryoshka_dims: c.matryoshka_dims.into_iter().map(|d| d as usize).collect(),
        })
    }
}

/// Map the wire [`pb::HardNegativeConfig`] onto the engine's
/// [`HardNegativeConfig`]. Absent on the wire = mining off (the engine default).
fn hard_negatives_from_proto(h: pb::HardNegativeConfig) -> HardNegativeConfig {
    HardNegativeConfig {
        mine: h.mine,
        k: h.k as usize,
        exclude_hops: h.exclude_hops as usize,
        refresh_every: h.refresh_every as usize,
    }
}

/// Map the wire embedding-loss message onto the engine's [`EmbeddingLoss`]. A
/// present message with no `loss` set is a malformed request.
fn embedding_loss_from_proto(loss: pb::EmbeddingLoss) -> Result<EmbeddingLoss, Status> {
    use pb::embedding_loss::Loss;
    match loss.loss {
        Some(Loss::CoSent(_)) => Ok(EmbeddingLoss::CoSent),
        Some(Loss::Triplet(t)) => Ok(EmbeddingLoss::Triplet { margin: t.margin }),
        Some(Loss::MultipleNegativesRanking(m)) => Ok(EmbeddingLoss::MultipleNegativesRanking {
            temperature: m.temperature,
        }),
        Some(Loss::Angle(_)) => Ok(EmbeddingLoss::AnglE),
        Some(Loss::CosineMse(_)) => Ok(EmbeddingLoss::CosineMse),
        None => Err(Status::invalid_argument(
            "embedding_loss is set but carries no variant",
        )),
    }
}

/// Map the wire [`pb::ClassificationLoss`] onto the engine's
/// [`ClassificationLoss`]. An unspecified value on a present field is a
/// malformed request — omit the field instead to let the engine auto-select.
fn classification_loss_from_proto(loss: i32) -> Result<ClassificationLoss, Status> {
    match pb::ClassificationLoss::try_from(loss) {
        Ok(pb::ClassificationLoss::CrossEntropy) => Ok(ClassificationLoss::CrossEntropy),
        Ok(pb::ClassificationLoss::Unspecified) | Err(_) => Err(Status::invalid_argument(
            "classification_loss is set but unspecified; omit it to auto-select",
        )),
    }
}

/// Map the wire [`pb::LrSchedule`]; `UNSPECIFIED` keeps the engine default.
fn lr_schedule_from_proto(schedule: i32, default: LrSchedule) -> Result<LrSchedule, Status> {
    match pb::LrSchedule::try_from(schedule) {
        Ok(pb::LrSchedule::Unspecified) => Ok(default),
        Ok(pb::LrSchedule::Constant) => Ok(LrSchedule::Constant),
        Ok(pb::LrSchedule::CosineDecay) => Ok(LrSchedule::CosineDecay),
        Ok(pb::LrSchedule::LinearDecay) => Ok(LrSchedule::LinearDecay),
        Err(_) => Err(Status::invalid_argument("unknown lr_schedule")),
    }
}

/// Map the wire [`pb::EarlyStoppingMetric`]; `UNSPECIFIED` keeps the default.
fn early_stopping_metric_from_proto(
    metric: i32,
    default: EarlyStoppingMetric,
) -> Result<EarlyStoppingMetric, Status> {
    match pb::EarlyStoppingMetric::try_from(metric) {
        Ok(pb::EarlyStoppingMetric::Unspecified) => Ok(default),
        Ok(pb::EarlyStoppingMetric::ValLoss) => Ok(EarlyStoppingMetric::ValLoss),
        Ok(pb::EarlyStoppingMetric::TrainLoss) => Ok(EarlyStoppingMetric::TrainLoss),
        Err(_) => Err(Status::invalid_argument("unknown early_stopping_metric")),
    }
}

/// Map the wire [`pb::LoraInitMode`]; `UNSPECIFIED` keeps the default.
fn lora_init_mode_from_proto(mode: i32, default: LoraInitMode) -> Result<LoraInitMode, Status> {
    match pb::LoraInitMode::try_from(mode) {
        Ok(pb::LoraInitMode::Unspecified) => Ok(default),
        Ok(pb::LoraInitMode::ZerosB) => Ok(LoraInitMode::ZerosB),
        Ok(pb::LoraInitMode::Gaussian) => Ok(LoraInitMode::Gaussian),
        Err(_) => Err(Status::invalid_argument("unknown init_lora_weights")),
    }
}

/// Map the wire [`pb::BackboneDtype`]; `UNSPECIFIED` keeps the default.
fn backbone_dtype_from_proto(dtype: i32, default: BackboneDtype) -> Result<BackboneDtype, Status> {
    match pb::BackboneDtype::try_from(dtype) {
        Ok(pb::BackboneDtype::Unspecified) => Ok(default),
        Ok(pb::BackboneDtype::F32) => Ok(BackboneDtype::F32),
        Ok(pb::BackboneDtype::Bf16) => Ok(BackboneDtype::BF16),
        Ok(pb::BackboneDtype::F16) => Ok(BackboneDtype::F16),
        Err(_) => Err(Status::invalid_argument("unknown backbone_dtype")),
    }
}

// ─── domain → proto (the RemoteSession send side) ────────────────────────────
//
// The inverse of the decodes above. The [`crate::RemoteSession`] encodes the
// engine [`FineTuneConfig`] (and its method) onto the wire so the server's
// decode rebuilds the identical config. Every concrete engine value maps to a
// concrete wire value — never `UNSPECIFIED`; the `UNSPECIFIED` arms exist only
// so a client that omits a field gets the engine default, which a fully-formed
// engine config never needs.

/// Encode the engine's [`FineTuneMethod`] onto the wire enum. Total — the engine
/// type has no unspecified variant.
pub fn method_to_proto(method: FineTuneMethod) -> pb::FineTuneMethod {
    match method {
        FineTuneMethod::Lora => pb::FineTuneMethod::Lora,
    }
}

/// Encode the engine [`FineTuneConfig`] onto the wire message. Mirrors
/// [`TryFrom<pb::FineTuneConfig> for FineTuneConfig`] field for field; every
/// enum-typed field encodes to its concrete wire variant (never `UNSPECIFIED`),
/// and the optional losses / layer restriction encode to their `Option`-shaped
/// wire fields.
pub fn config_to_proto(config: &FineTuneConfig) -> pb::FineTuneConfig {
    pb::FineTuneConfig {
        lora_rank: config.lora_rank as u32,
        lora_alpha: config.lora_alpha,
        lora_dropout: config.lora_dropout,
        learning_rate: config.learning_rate,
        epochs: config.epochs as u32,
        batch_size: config.batch_size as u32,
        max_seq_length: config.max_seq_length as u32,
        embedding_loss: config.embedding_loss.as_ref().map(embedding_loss_to_proto),
        classification_loss: config
            .classification_loss
            .as_ref()
            .map(|l| classification_loss_to_proto(l) as i32),
        gradient_accumulation_steps: config.gradient_accumulation_steps as u32,
        validation_fraction: config.validation_fraction,
        early_stopping_patience: config.early_stopping_patience as u32,
        warmup_steps: config.warmup_steps as u32,
        lr_schedule: lr_schedule_to_proto(config.lr_schedule) as i32,
        early_stopping_metric: early_stopping_metric_to_proto(config.early_stopping_metric) as i32,
        target_modules: config.target_modules.clone(),
        layers_to_transform: config.layers_to_transform.as_ref().map(|layers| {
            pb::LayersToTransform {
                layers: layers.iter().map(|n| *n as u32).collect(),
            }
        }),
        use_rslora: config.use_rslora,
        rank_pattern: config
            .rank_pattern
            .iter()
            .map(|(k, v)| (k.clone(), *v as u32))
            .collect(),
        init_lora_weights: lora_init_mode_to_proto(config.init_lora_weights) as i32,
        backbone_dtype: backbone_dtype_to_proto(config.backbone_dtype) as i32,
        weight_decay: config.weight_decay,
        max_grad_norm: config.max_grad_norm,
        cached: config.cached,
        // Always encoded so a round-trip preserves the k/hop/refresh knobs even
        // when mining is off; the decode treats an absent message as "off".
        hard_negatives: Some(hard_negatives_to_proto(&config.hard_negatives)),
        matryoshka_dims: config.matryoshka_dims.iter().map(|d| *d as u32).collect(),
    }
}

fn hard_negatives_to_proto(h: &HardNegativeConfig) -> pb::HardNegativeConfig {
    pb::HardNegativeConfig {
        mine: h.mine,
        k: h.k as u32,
        exclude_hops: h.exclude_hops as u32,
        refresh_every: h.refresh_every as u32,
    }
}

fn embedding_loss_to_proto(loss: &EmbeddingLoss) -> pb::EmbeddingLoss {
    use pb::embedding_loss::Loss;
    let inner = match loss {
        EmbeddingLoss::CoSent => Loss::CoSent(pb::embedding_loss::CoSent {}),
        EmbeddingLoss::Triplet { margin } => {
            Loss::Triplet(pb::embedding_loss::Triplet { margin: *margin })
        }
        EmbeddingLoss::MultipleNegativesRanking { temperature } => {
            Loss::MultipleNegativesRanking(pb::embedding_loss::MultipleNegativesRanking {
                temperature: *temperature,
            })
        }
        EmbeddingLoss::AnglE => Loss::Angle(pb::embedding_loss::AnglE {}),
        EmbeddingLoss::CosineMse => Loss::CosineMse(pb::embedding_loss::CosineMse {}),
    };
    pb::EmbeddingLoss { loss: Some(inner) }
}

fn classification_loss_to_proto(loss: &ClassificationLoss) -> pb::ClassificationLoss {
    match loss {
        ClassificationLoss::CrossEntropy => pb::ClassificationLoss::CrossEntropy,
    }
}

fn lr_schedule_to_proto(schedule: LrSchedule) -> pb::LrSchedule {
    match schedule {
        LrSchedule::Constant => pb::LrSchedule::Constant,
        LrSchedule::CosineDecay => pb::LrSchedule::CosineDecay,
        LrSchedule::LinearDecay => pb::LrSchedule::LinearDecay,
    }
}

fn early_stopping_metric_to_proto(metric: EarlyStoppingMetric) -> pb::EarlyStoppingMetric {
    match metric {
        EarlyStoppingMetric::ValLoss => pb::EarlyStoppingMetric::ValLoss,
        EarlyStoppingMetric::TrainLoss => pb::EarlyStoppingMetric::TrainLoss,
    }
}

fn lora_init_mode_to_proto(mode: LoraInitMode) -> pb::LoraInitMode {
    match mode {
        LoraInitMode::ZerosB => pb::LoraInitMode::ZerosB,
        LoraInitMode::Gaussian => pb::LoraInitMode::Gaussian,
    }
}

fn backbone_dtype_to_proto(dtype: BackboneDtype) -> pb::BackboneDtype {
    match dtype {
        BackboneDtype::F32 => pb::BackboneDtype::F32,
        BackboneDtype::BF16 => pb::BackboneDtype::Bf16,
        BackboneDtype::F16 => pb::BackboneDtype::F16,
    }
}
