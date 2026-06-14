//! `TrainingService` proto↔domain conversions for the transport-neutral config.
//!
//! The request `FineTuneConfig` mirrors the engine's [`FineTuneConfig`] field
//! for field; decode starts from [`FineTuneConfig::default()`] and overrides a
//! field only when the wire carries it — each scalar knob has explicit presence
//! (`optional`), so an unset field resolves to the engine default rather than a
//! literal zero, and an absent `config` message → the engine default entirely.
//! The engine is thus the single source of default values for every client.
//! Validation stays in the engine (the submit verbs call `validate`); this is a
//! pure shape map. The `StartTraining` spec `oneof` ↔ engine `TrainingSpec`
//! conversions touch the engine spec vocabulary and so live in the residual
//! `jammi_ai::wire` module, not here.

use tonic::Status;

use crate::fine_tune::{
    BackboneDtype, ClassificationLoss, EarlyStoppingMetric, EmbeddingLoss, FineTuneConfig,
    FineTuneMethod, HardNegativeConfig, LoraInitMode, LrSchedule, RegressionLoss,
};

use crate::proto::training as pb;

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
/// The engine is the single source of default values: decode starts from
/// [`FineTuneConfig::default()`] and overrides a field only when the wire
/// carries it. Each scalar knob has explicit presence (`optional`), so an unset
/// field is distinguishable from a legal `0`/`false` and resolves to the engine
/// default rather than a literal zero. The optional loss messages map to
/// `Option<…Loss>` (unset → engine auto-selects from the data format). The
/// enum-typed fields fall back to the engine default variant when left
/// `UNSPECIFIED`. A config that sets only a handful of knobs behaves exactly
/// like `FineTuneConfig::default()` for the rest.
impl TryFrom<pb::FineTuneConfig> for FineTuneConfig {
    type Error = Status;

    fn try_from(c: pb::FineTuneConfig) -> Result<Self, Self::Error> {
        let mut cfg = FineTuneConfig::default();

        if let Some(v) = c.lora_rank {
            cfg.lora_rank = v as usize;
        }
        if let Some(v) = c.lora_alpha {
            cfg.lora_alpha = v;
        }
        if let Some(v) = c.lora_dropout {
            cfg.lora_dropout = v;
        }
        if let Some(v) = c.learning_rate {
            cfg.learning_rate = v;
        }
        if let Some(v) = c.epochs {
            cfg.epochs = v as usize;
        }
        if let Some(v) = c.batch_size {
            cfg.batch_size = v as usize;
        }
        if let Some(v) = c.max_seq_length {
            cfg.max_seq_length = v as usize;
        }
        if let Some(loss) = c.embedding_loss {
            cfg.embedding_loss = Some(embedding_loss_from_proto(loss)?);
        }
        if let Some(loss) = c.classification_loss {
            cfg.classification_loss = Some(classification_loss_from_proto(loss)?);
        }
        if let Some(v) = c.gradient_accumulation_steps {
            cfg.gradient_accumulation_steps = v as usize;
        }
        if let Some(v) = c.validation_fraction {
            cfg.validation_fraction = v;
        }
        if let Some(v) = c.early_stopping_patience {
            cfg.early_stopping_patience = v as usize;
        }
        if let Some(v) = c.warmup_steps {
            cfg.warmup_steps = v as usize;
        }
        cfg.lr_schedule = lr_schedule_from_proto(c.lr_schedule, cfg.lr_schedule)?;
        cfg.early_stopping_metric =
            early_stopping_metric_from_proto(c.early_stopping_metric, cfg.early_stopping_metric)?;
        if !c.target_modules.is_empty() {
            cfg.target_modules = c.target_modules;
        }
        if let Some(l) = c.layers_to_transform {
            cfg.layers_to_transform = Some(l.layers.into_iter().map(|n| n as usize).collect());
        }
        if let Some(v) = c.use_rslora {
            cfg.use_rslora = v;
        }
        if !c.rank_pattern.is_empty() {
            cfg.rank_pattern = c
                .rank_pattern
                .into_iter()
                .map(|(k, v)| (k, v as usize))
                .collect();
        }
        cfg.init_lora_weights =
            lora_init_mode_from_proto(c.init_lora_weights, cfg.init_lora_weights)?;
        cfg.backbone_dtype = backbone_dtype_from_proto(c.backbone_dtype, cfg.backbone_dtype)?;
        if let Some(v) = c.weight_decay {
            cfg.weight_decay = v;
        }
        if let Some(v) = c.max_grad_norm {
            cfg.max_grad_norm = v;
        }
        if let Some(v) = c.cached {
            cfg.cached = v;
        }
        if let Some(h) = c.hard_negatives {
            cfg.hard_negatives = hard_negatives_from_proto(h);
        }
        if !c.matryoshka_dims.is_empty() {
            cfg.matryoshka_dims = c.matryoshka_dims.into_iter().map(|d| d as usize).collect();
        }
        if let Some(loss) = c.regression_loss {
            cfg.regression_loss = Some(regression_loss_from_proto(loss)?);
        }
        if !c.quantile_levels.is_empty() {
            cfg.quantile_levels = c.quantile_levels;
        }
        if let Some(v) = c.seed {
            cfg.seed = v;
        }

        Ok(cfg)
    }
}

/// Map the wire regression-loss message onto the engine's [`RegressionLoss`]. A
/// present message with no `loss` set is a malformed request.
fn regression_loss_from_proto(loss: pb::RegressionLoss) -> Result<RegressionLoss, Status> {
    use pb::regression_loss::Loss;
    match loss.loss {
        Some(Loss::GaussianNll(_)) => Ok(RegressionLoss::GaussianNll),
        Some(Loss::BetaNll(b)) => Ok(RegressionLoss::BetaNll { beta: b.beta }),
        Some(Loss::Crps(_)) => Ok(RegressionLoss::Crps),
        Some(Loss::Pinball(_)) => Ok(RegressionLoss::Pinball),
        None => Err(Status::invalid_argument(
            "regression_loss is set but carries no variant",
        )),
    }
}

/// Map the wire [`pb::HardNegativeConfig`] onto the engine's
/// [`HardNegativeConfig`]. Absent on the wire = mining off (the engine default).
///
/// Mirrors the [`FineTuneConfig`] overlay: decode starts from
/// [`HardNegativeConfig::default()`] and overrides a field only when the wire
/// carries it. Each scalar knob has explicit presence (`optional`), so a caller
/// that sets only `mine = true` resolves `k`/`exclude_hops`/`refresh_every` to
/// the engine defaults rather than literal zeros — the engine is the single
/// source of these defaults for both the remote and embedded surfaces.
fn hard_negatives_from_proto(h: pb::HardNegativeConfig) -> HardNegativeConfig {
    let mut cfg = HardNegativeConfig {
        mine: h.mine,
        ..HardNegativeConfig::default()
    };
    if let Some(k) = h.k {
        cfg.k = k as usize;
    }
    if let Some(exclude_hops) = h.exclude_hops {
        cfg.exclude_hops = exclude_hops as usize;
    }
    if let Some(refresh_every) = h.refresh_every {
        cfg.refresh_every = refresh_every as usize;
    }
    cfg
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

// ─── domain → proto (the data-plane client send side) ────────────────────────
//
// The inverse of the decodes above. The data-plane client encodes the engine
// [`FineTuneConfig`] (and its method) onto the wire so the server's decode
// rebuilds the identical config. Every concrete engine value maps to a concrete
// wire value — never `UNSPECIFIED`; the `UNSPECIFIED` arms exist only so a
// client that omits a field gets the engine default, which a fully-formed engine
// config never needs.

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
        lora_rank: Some(config.lora_rank as u32),
        lora_alpha: Some(config.lora_alpha),
        lora_dropout: Some(config.lora_dropout),
        learning_rate: Some(config.learning_rate),
        epochs: Some(config.epochs as u32),
        batch_size: Some(config.batch_size as u32),
        max_seq_length: Some(config.max_seq_length as u32),
        embedding_loss: config.embedding_loss.as_ref().map(embedding_loss_to_proto),
        classification_loss: config
            .classification_loss
            .as_ref()
            .map(|l| classification_loss_to_proto(l) as i32),
        gradient_accumulation_steps: Some(config.gradient_accumulation_steps as u32),
        validation_fraction: Some(config.validation_fraction),
        early_stopping_patience: Some(config.early_stopping_patience as u32),
        warmup_steps: Some(config.warmup_steps as u32),
        lr_schedule: lr_schedule_to_proto(config.lr_schedule) as i32,
        early_stopping_metric: early_stopping_metric_to_proto(config.early_stopping_metric) as i32,
        target_modules: config.target_modules.clone(),
        layers_to_transform: config.layers_to_transform.as_ref().map(|layers| {
            pb::LayersToTransform {
                layers: layers.iter().map(|n| *n as u32).collect(),
            }
        }),
        use_rslora: Some(config.use_rslora),
        rank_pattern: config
            .rank_pattern
            .iter()
            .map(|(k, v)| (k.clone(), *v as u32))
            .collect(),
        init_lora_weights: lora_init_mode_to_proto(config.init_lora_weights) as i32,
        backbone_dtype: backbone_dtype_to_proto(config.backbone_dtype) as i32,
        weight_decay: Some(config.weight_decay),
        max_grad_norm: Some(config.max_grad_norm),
        cached: Some(config.cached),
        // Always encoded so a round-trip preserves the k/hop/refresh knobs even
        // when mining is off; the decode treats an absent message as "off".
        hard_negatives: Some(hard_negatives_to_proto(&config.hard_negatives)),
        matryoshka_dims: config.matryoshka_dims.iter().map(|d| *d as u32).collect(),
        regression_loss: config
            .regression_loss
            .as_ref()
            .map(regression_loss_to_proto),
        quantile_levels: config.quantile_levels.clone(),
        seed: Some(config.seed),
    }
}

fn regression_loss_to_proto(loss: &RegressionLoss) -> pb::RegressionLoss {
    use pb::regression_loss::Loss;
    let inner = match loss {
        RegressionLoss::GaussianNll => Loss::GaussianNll(pb::regression_loss::GaussianNll {}),
        RegressionLoss::BetaNll { beta } => {
            Loss::BetaNll(pb::regression_loss::BetaNll { beta: *beta })
        }
        RegressionLoss::Crps => Loss::Crps(pb::regression_loss::Crps {}),
        RegressionLoss::Pinball => Loss::Pinball(pb::regression_loss::Pinball {}),
    };
    pb::RegressionLoss { loss: Some(inner) }
}

fn hard_negatives_to_proto(h: &HardNegativeConfig) -> pb::HardNegativeConfig {
    // The engine config is fully resolved, so every scalar encodes as present —
    // a round-trip preserves the k/hop/refresh knobs. The decode overlay reads
    // an absent scalar as "apply the engine default".
    pb::HardNegativeConfig {
        mine: h.mine,
        k: Some(h.k as u32),
        exclude_hops: Some(h.exclude_hops as u32),
        refresh_every: Some(h.refresh_every as u32),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// An all-unset wire config — the shape a remote client builds when the
    /// caller omits every hyperparameter. With explicit presence on every
    /// scalar, this decodes to exactly the engine default: an omitted field is
    /// the engine default, never a literal `0`/`false`. This is the regression
    /// the pre-fix raw-cast decode silently broke (it read each scalar as `0`,
    /// disabling weight decay / clipping / dropout / warmup / val split, or
    /// failing validation outright on the count knobs).
    #[test]
    fn all_unset_config_decodes_to_engine_default() {
        let proto = pb::FineTuneConfig::default();
        let decoded = FineTuneConfig::try_from(proto).expect("all-unset config decodes");
        assert_eq!(decoded, FineTuneConfig::default());

        // Spot-check the default-bearing scalars that a literal-zero decode
        // would have silently zeroed out, training a materially different model.
        assert_eq!(decoded.weight_decay, 0.01);
        assert_eq!(decoded.max_grad_norm, 1.0);
        assert_eq!(decoded.lora_dropout, 0.05);
        assert_eq!(decoded.warmup_steps, 100);
        assert_eq!(decoded.validation_fraction, 0.1);
        // And the count knobs a literal-zero decode would have failed validation
        // on resolve to their non-zero engine defaults instead.
        assert_eq!(decoded.lora_rank, 8);
        assert_eq!(decoded.epochs, 3);
        assert_eq!(decoded.batch_size, 8);
    }

    /// A partially-set wire config overrides exactly the present fields and
    /// leaves every other field at the engine default — including a legal `0`
    /// override (`warmup_steps = 0` to disable warmup), which is now
    /// distinguishable from an unset field.
    #[test]
    fn partial_config_overrides_only_present_fields() {
        let proto = pb::FineTuneConfig {
            lora_rank: Some(16),
            learning_rate: Some(1e-3),
            // A legal zero override: explicit "no warmup", distinct from unset.
            warmup_steps: Some(0),
            ..Default::default()
        };
        let decoded = FineTuneConfig::try_from(proto).expect("partial config decodes");

        let defaults = FineTuneConfig::default();
        assert_eq!(decoded.lora_rank, 16);
        assert_eq!(decoded.learning_rate, 1e-3);
        assert_eq!(decoded.warmup_steps, 0);
        // Untouched fields stay at the engine default.
        assert_eq!(decoded.weight_decay, defaults.weight_decay);
        assert_eq!(decoded.max_grad_norm, defaults.max_grad_norm);
        assert_eq!(decoded.epochs, defaults.epochs);
        assert_eq!(decoded.batch_size, defaults.batch_size);
        assert_eq!(decoded.lora_dropout, defaults.lora_dropout);
    }

    /// The full-config send side round-trips: encoding the engine default and
    /// decoding it back yields the engine default. This pins the data-plane
    /// client path (which sends a full config) against the now-optional fields.
    #[test]
    fn config_to_proto_round_trips_through_decode() {
        let original = FineTuneConfig::default();
        let proto = config_to_proto(&original);
        let decoded = FineTuneConfig::try_from(proto).expect("round-trip decodes");
        assert_eq!(decoded, original);
    }

    /// A remote caller that enables mining but omits the count knobs ships a
    /// `HardNegativeConfig{mine: true}` with every scalar unset. With explicit
    /// presence on `k`/`exclude_hops`/`refresh_every`, this overlays onto the
    /// engine default rather than decoding the scalars as `0` — so the resulting
    /// config carries the engine's `k=1, exclude_hops=1, refresh_every=1` and
    /// passes `validate`, instead of the pre-fix `refresh_every = 0` that
    /// `validate` rejected for a knob the caller never set.
    #[test]
    fn hard_negatives_mine_only_overlays_engine_defaults() {
        let proto = pb::HardNegativeConfig {
            mine: true,
            ..Default::default()
        };
        let decoded = hard_negatives_from_proto(proto);

        assert_eq!(
            decoded,
            HardNegativeConfig {
                mine: true,
                ..HardNegativeConfig::default()
            }
        );
        assert!(decoded.mine);
        assert_eq!(decoded.k, 1);
        assert_eq!(decoded.exclude_hops, 1);
        assert_eq!(decoded.refresh_every, 1);

        // The whole point: a mining-on config built from `mine` alone validates.
        let cfg = FineTuneConfig {
            hard_negatives: decoded,
            ..FineTuneConfig::default()
        };
        cfg.validate()
            .expect("mine-only hard-negative config validates");
    }

    /// A partially-set hard-negative config overrides exactly the present knobs
    /// and leaves the rest at the engine default.
    #[test]
    fn hard_negatives_partial_overrides_only_present_fields() {
        let proto = pb::HardNegativeConfig {
            mine: true,
            k: Some(5),
            ..Default::default()
        };
        let decoded = hard_negatives_from_proto(proto);

        assert!(decoded.mine);
        assert_eq!(decoded.k, 5);
        // Untouched knobs stay at the engine default.
        assert_eq!(decoded.exclude_hops, 1);
        assert_eq!(decoded.refresh_every, 1);
    }
}
