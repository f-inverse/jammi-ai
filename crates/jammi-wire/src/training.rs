//! Transport-neutral training-config proto↔domain conversions, shared by the
//! `jammi.v1.job` `JobService.SubmitJob` training kinds.
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
    ClassificationLoss, ComputePrecision, EarlyStoppingMetric, EmbeddingLoss, FineTuneConfig,
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
        if let Some(v) = c.keep_last_n_checkpoints {
            cfg.keep_last_n_checkpoints = Some(v);
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
fn backbone_dtype_from_proto(
    dtype: i32,
    default: ComputePrecision,
) -> Result<ComputePrecision, Status> {
    match pb::BackboneDtype::try_from(dtype) {
        Ok(pb::BackboneDtype::Unspecified) => Ok(default),
        Ok(pb::BackboneDtype::F32) => Ok(ComputePrecision::F32),
        Ok(pb::BackboneDtype::Bf16) => Ok(ComputePrecision::BF16),
        Ok(pb::BackboneDtype::F16) => Ok(ComputePrecision::F16),
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
        keep_last_n_checkpoints: config.keep_last_n_checkpoints,
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

fn backbone_dtype_to_proto(dtype: ComputePrecision) -> pb::BackboneDtype {
    match dtype {
        ComputePrecision::F32 => pb::BackboneDtype::F32,
        ComputePrecision::BF16 => pb::BackboneDtype::Bf16,
        ComputePrecision::F16 => pb::BackboneDtype::F16,
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

    /// NON-BUG GOLDEN for #347.
    ///
    /// #347's first stated root cause is that a serde default overrides an
    /// explicit `validation_fraction = 0`, so a user who asks for no validation
    /// split silently gets 10%. That is false, and this pins why so nobody
    /// "fixes" it later: the proto field carries explicit presence, so a set-to
    /// -zero and an unset field are distinguishable on the wire, and the
    /// converter honours both.
    ///
    /// The existing all-unset test covers only the omitted half — which is
    /// exactly why the misconception was plausible. Both halves are asserted
    /// here. A future change that made the field non-optional, or that coerced
    /// `0.0` to the default, would fail this.
    #[test]
    fn explicit_zero_validation_fraction_survives_the_wire_round_trip() {
        let cfg = FineTuneConfig {
            validation_fraction: 0.0,
            // The zero split is only legal alongside a metric that needs no
            // validation pass; see FineTuneConfig::validate.
            early_stopping_metric: EarlyStoppingMetric::TrainLoss,
            ..Default::default()
        };
        cfg.validate()
            .expect("0.0 + train_loss is a legal combination");

        let proto = config_to_proto(&cfg);
        assert_eq!(
            proto.validation_fraction,
            Some(0.0),
            "an explicit zero must be SENT as present, not omitted"
        );
        let decoded = FineTuneConfig::try_from(proto).expect("decodes");
        assert_eq!(
            decoded.validation_fraction, 0.0,
            "an explicit zero must survive the round trip as zero, not become the default"
        );

        // The other half: an omitted field still resolves to the engine default,
        // so presence is genuinely carrying the distinction.
        let omitted = pb::FineTuneConfig::default();
        assert_eq!(
            FineTuneConfig::try_from(omitted)
                .unwrap()
                .validation_fraction,
            0.1
        );
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

    /// Unit 348: `keep_last_n_checkpoints` overlays like every other optional
    /// scalar — absent on the wire keeps the engine default (`None`, keep
    /// every epoch), present overrides it — and a set value round-trips
    /// through the send side unchanged.
    #[test]
    fn keep_last_n_checkpoints_overlays_and_round_trips() {
        let proto = pb::FineTuneConfig {
            keep_last_n_checkpoints: Some(3),
            ..Default::default()
        };
        let decoded = FineTuneConfig::try_from(proto).expect("decodes");
        assert_eq!(decoded.keep_last_n_checkpoints, Some(3));

        let absent = pb::FineTuneConfig::default();
        let decoded_absent = FineTuneConfig::try_from(absent).expect("decodes");
        assert_eq!(
            decoded_absent.keep_last_n_checkpoints,
            FineTuneConfig::default().keep_last_n_checkpoints,
            "an absent field must resolve to the engine default (None)"
        );

        let cfg = FineTuneConfig {
            keep_last_n_checkpoints: Some(4),
            ..FineTuneConfig::default()
        };
        let round_tripped =
            FineTuneConfig::try_from(config_to_proto(&cfg)).expect("round-trip decodes");
        assert_eq!(round_tripped.keep_last_n_checkpoints, Some(4));
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

/// The per-job `world_size` on `jammi.v1.job.SubmitJobRequest`.
///
/// The rank count is the wire form of the engine's `TrainingCommon.world_size`,
/// so it rides beside the other two `TrainingCommon` members (`base_model`,
/// `config`) on the submit request rather than inside a per-kind spec: both
/// LoRA fine-tune kinds fold the same common block, and one field for both is
/// what makes it impossible for the two kinds to diverge on it.
///
/// These tests pin the three properties the frozen wire surface owes for an
/// appended scalar: the tag is the next free one and every pre-existing tag is
/// unmoved; an unset (`0`) count encodes to bytes byte-for-byte identical to
/// what a caller built before the field existed; and a set count survives the
/// round trip. Resolving `0` to the engine's `1` is the engine's decode step
/// (`jammi_ai::wire::training`), not the wire's — on the wire `0` is simply the
/// absent field.
#[cfg(test)]
mod world_size_tests {
    use prost::Message;
    use prost_types::{field_descriptor_proto::Type, FileDescriptorSet};

    use crate::proto::{job as job_pb, training as pb};
    use crate::FILE_DESCRIPTOR_SET;

    /// `SubmitJobRequest` as it was before `world_size` was appended, in the two
    /// scalar fields a submit carries unconditionally. Encoding through this
    /// gives a genuine pre-field byte string to compare against, rather than
    /// re-encoding the new type and asserting it matches itself.
    #[derive(Clone, PartialEq, prost::Message)]
    struct LegacySubmitJobRequest {
        #[prost(string, tag = "4")]
        base_model: String,
        #[prost(string, tag = "6")]
        idempotency_key: String,
    }

    /// The compiled `jammi.v1.job.SubmitJobRequest` descriptor — the
    /// authoritative description of the emitted wire surface.
    fn submit_job_request() -> prost_types::DescriptorProto {
        let set = FileDescriptorSet::decode(FILE_DESCRIPTOR_SET)
            .expect("the compiled jammi.v1 descriptor must decode");
        set.file
            .iter()
            .filter(|f| f.package() == "jammi.v1.job")
            .flat_map(|f| f.message_type.iter())
            .find(|m| m.name() == "SubmitJobRequest")
            .expect("jammi.v1.job.SubmitJobRequest is in the descriptor")
            .clone()
    }

    /// The `SubmitJobRequest` field names and numbers.
    fn submit_job_request_tags() -> Vec<(String, i32)> {
        submit_job_request()
            .field
            .iter()
            .map(|f| (f.name().to_string(), f.number()))
            .collect()
    }

    /// Every field number the message holds RESERVED, flattened from the
    /// descriptor's half-open ranges.
    fn submit_job_request_reserved_tags() -> Vec<i32> {
        let mut reserved: Vec<i32> = submit_job_request()
            .reserved_range
            .iter()
            .flat_map(|r| r.start()..r.end())
            .collect();
        reserved.sort_unstable();
        reserved.dedup();
        reserved
    }

    /// APPEND-ONLY. Every pre-existing `SubmitJobRequest` tag keeps its number
    /// and `world_size` takes the next free one that is not HELD — 9, because 7
    /// and 8 are reserved for the deferred job-dependency unit (#515). A
    /// renumbering, or taking a held tag, would decode a payload built against
    /// either contract into the wrong field.
    #[test]
    fn world_size_takes_the_next_free_tag_and_moves_no_existing_one() {
        assert_eq!(
            submit_job_request_tags(),
            vec![
                ("fine_tune".to_string(), 1),
                ("graph_fine_tune".to_string(), 2),
                ("context_predictor".to_string(), 3),
                ("base_model".to_string(), 4),
                ("config".to_string(), 5),
                ("idempotency_key".to_string(), 6),
                ("world_size".to_string(), 9),
                // `cache` (below, [`cache_tests`]) took the next free tag
                // after this test was written; listed here too so this
                // exhaustive tag inventory stays the single source of truth
                // for the message's WHOLE field set, not merely the count.
                ("cache".to_string(), 10),
            ],
        );
    }

    /// HELD. Tags 7 and 8 belong to the deferred job-dependency unit
    /// (`depends_on = 7`, `parent_id = 8`, #515): the message reserves them, so
    /// `protoc` refuses any later field that tries to take one and the
    /// cherry-pick reviving that unit cannot collide with a field appended in
    /// the meantime. Same policy as `jammi.v1.error`'s vacant 37/38.
    #[test]
    fn tags_seven_and_eight_are_held_for_the_deferred_dependency_unit() {
        assert_eq!(submit_job_request_reserved_tags(), vec![7, 8]);
        assert!(
            !submit_job_request_tags()
                .iter()
                .any(|(_, number)| *number == 7 || *number == 8),
            "no live field may occupy a reserved tag"
        );
    }

    /// The count is an implicit-presence `uint32`, which is what makes `0` mean
    /// "unset" on the wire: an `optional` field would make an explicit `0` a
    /// distinct, encodable state the engine has no meaning for.
    #[test]
    fn world_size_is_an_implicit_presence_uint32() {
        let set = FileDescriptorSet::decode(FILE_DESCRIPTOR_SET)
            .expect("the compiled jammi.v1 descriptor must decode");
        let field = set
            .file
            .iter()
            .filter(|f| f.package() == "jammi.v1.job")
            .flat_map(|f| f.message_type.iter())
            .find(|m| m.name() == "SubmitJobRequest")
            .expect("jammi.v1.job.SubmitJobRequest is in the descriptor")
            .field
            .iter()
            .find(|f| f.name() == "world_size")
            .expect("SubmitJobRequest carries world_size")
            .clone();

        assert_eq!(field.r#type(), Type::Uint32);
        assert_ne!(
            field.proto3_optional,
            Some(true),
            "world_size must not have explicit presence: 0 IS the unset value"
        );
    }

    /// UNSET. A request that leaves the count at `0` encodes to exactly the
    /// bytes a caller built before the field existed — the appended field costs
    /// a pre-existing client nothing and changes no byte it ever sent.
    #[test]
    fn unset_world_size_encodes_byte_for_byte_as_the_pre_field_request() {
        let legacy = LegacySubmitJobRequest {
            base_model: "local:tiny-bert".to_string(),
            idempotency_key: "dedupe-1".to_string(),
        };
        let current = job_pb::SubmitJobRequest {
            spec: None,
            base_model: "local:tiny-bert".to_string(),
            config: None,
            idempotency_key: "dedupe-1".to_string(),
            world_size: 0,
            cache: 0,
        };

        assert_eq!(current.encode_to_vec(), legacy.encode_to_vec());
    }

    /// BACKWARD. Bytes produced before the field existed decode into the
    /// current type with the count at `0` — the value the engine reads as one
    /// rank — and lose nothing else.
    #[test]
    fn pre_field_bytes_decode_with_the_count_unset() {
        let legacy = LegacySubmitJobRequest {
            base_model: "local:tiny-bert".to_string(),
            idempotency_key: "dedupe-1".to_string(),
        };

        let decoded = job_pb::SubmitJobRequest::decode(legacy.encode_to_vec().as_slice())
            .expect("pre-field bytes decode into the current request");

        assert_eq!(decoded.world_size, 0);
        assert_eq!(decoded.base_model, "local:tiny-bert");
        assert_eq!(decoded.idempotency_key, "dedupe-1");
    }

    /// FORWARD. A peer that predates the field decodes a request carrying it
    /// without error — the appended tag is an unknown field to it, skipped, and
    /// every field it does know survives.
    #[test]
    fn a_pre_field_decoder_skips_a_set_count() {
        let current = job_pb::SubmitJobRequest {
            spec: None,
            base_model: "local:tiny-bert".to_string(),
            config: None,
            idempotency_key: "dedupe-1".to_string(),
            world_size: 4,
            cache: 0,
        };

        let decoded = LegacySubmitJobRequest::decode(current.encode_to_vec().as_slice())
            .expect("a pre-field decoder skips the appended tag");

        assert_eq!(decoded.base_model, "local:tiny-bert");
        assert_eq!(decoded.idempotency_key, "dedupe-1");
    }

    /// SET. A full fine-tune submit carrying a multi-rank count round-trips
    /// unchanged — the count included.
    #[test]
    fn a_set_world_size_round_trips_on_a_full_request() {
        let original = job_pb::SubmitJobRequest {
            spec: Some(job_pb::submit_job_request::Spec::FineTune(
                pb::FineTuneSpec {
                    source: "training".to_string(),
                    columns: vec!["text_a".to_string(), "text_b".to_string()],
                    method: pb::FineTuneMethod::Lora as i32,
                    task: crate::proto::inference::ModelTask::TextEmbedding as i32,
                },
            )),
            base_model: "local:tiny-bert".to_string(),
            config: Some(pb::FineTuneConfig {
                epochs: Some(2),
                ..Default::default()
            }),
            idempotency_key: "dedupe-1".to_string(),
            world_size: 4,
            cache: 0,
        };

        let decoded = job_pb::SubmitJobRequest::decode(original.encode_to_vec().as_slice())
            .expect("a request carrying the count round-trips");

        assert_eq!(decoded.world_size, 4);
        assert_eq!(decoded, original);
    }

    /// The count rides on the request, not the kind, so the SAME field serves a
    /// graph fine-tune: the two LoRA kinds cannot diverge on it, because there
    /// is only one place to put it.
    #[test]
    fn the_same_count_field_serves_the_graph_fine_tune_kind() {
        let original = job_pb::SubmitJobRequest {
            spec: Some(job_pb::submit_job_request::Spec::GraphFineTune(
                pb::GraphFineTuneSpec {
                    sources: None,
                    sample_config: None,
                },
            )),
            base_model: "local:tiny-bert".to_string(),
            config: None,
            idempotency_key: String::new(),
            world_size: 2,
            cache: 0,
        };

        let decoded = job_pb::SubmitJobRequest::decode(original.encode_to_vec().as_slice())
            .expect("a graph fine-tune carrying the count round-trips");

        assert_eq!(decoded.world_size, 2);
        assert_eq!(decoded, original);
    }
}

/// The per-job `cache` on `jammi.v1.job.SubmitJobRequest`.
///
/// Opt-in model-level cache reuse for the two LoRA fine-tune kinds, appended
/// the same way [`world_size_tests`] appended the rank count: the next free
/// tag, an implicit-presence field whose zero value IS the unset/default
/// state, and the SAME `jammi.v1.inference.CachePolicy` enum every other
/// result-table producer verb carries — imported rather than redeclared, so
/// this field can never drift onto a second, differently-numbered cache
/// vocabulary.
#[cfg(test)]
mod cache_tests {
    use prost::Message;
    use prost_types::{field_descriptor_proto::Type, FileDescriptorSet};

    use crate::proto::{job as job_pb, training as pb};
    use crate::FILE_DESCRIPTOR_SET;

    /// `SubmitJobRequest` before `cache` was appended, carrying every scalar a
    /// submit sends unconditionally (`base_model`, `idempotency_key`,
    /// `world_size`) so encoding through this gives genuine pre-field bytes
    /// rather than re-encoding the current type and comparing it with itself.
    #[derive(Clone, PartialEq, prost::Message)]
    struct PreCacheSubmitJobRequest {
        #[prost(string, tag = "4")]
        base_model: String,
        #[prost(string, tag = "6")]
        idempotency_key: String,
        #[prost(uint32, tag = "9")]
        world_size: u32,
    }

    fn submit_job_request_field(name: &str) -> prost_types::FieldDescriptorProto {
        let set = FileDescriptorSet::decode(FILE_DESCRIPTOR_SET)
            .expect("the compiled jammi.v1 descriptor must decode");
        set.file
            .iter()
            .filter(|f| f.package() == "jammi.v1.job")
            .flat_map(|f| f.message_type.iter())
            .find(|m| m.name() == "SubmitJobRequest")
            .expect("jammi.v1.job.SubmitJobRequest is in the descriptor")
            .field
            .iter()
            .find(|f| f.name() == name)
            .unwrap_or_else(|| panic!("SubmitJobRequest carries a `{name}` field"))
            .clone()
    }

    /// APPEND-ONLY. `cache` takes the next free tag after `world_size` (9) —
    /// 10, since 7 and 8 stay reserved for the deferred job-dependency unit
    /// (#515) — and every pre-existing field keeps its number. A renumbering,
    /// or taking a held tag, would decode a payload built against either
    /// contract into the wrong field.
    #[test]
    fn cache_takes_the_next_free_tag() {
        assert_eq!(submit_job_request_field("cache").number(), 10);
    }

    /// `cache` is a proto3 enum, implicit presence: `0` (`UNSPECIFIED`) is
    /// what a request that never set the field carries, matching how
    /// `world_size`'s `0` is its unset value. It resolves to the engine's
    /// `CachePolicy::Bypass` default at the decode
    /// (`jammi_ai::wire::cache::cache_policy_from_proto`), never on the wire.
    #[test]
    fn cache_is_an_implicit_presence_enum_of_the_shared_cache_policy_type() {
        let field = submit_job_request_field("cache");
        assert_eq!(field.r#type(), Type::Enum);
        assert_eq!(field.type_name(), ".jammi.v1.inference.CachePolicy");
        assert_ne!(
            field.proto3_optional,
            Some(true),
            "cache must not have explicit presence: UNSPECIFIED (0) IS the unset value"
        );
    }

    /// UNSET. A request that leaves `cache` at `UNSPECIFIED` (`0`) encodes to
    /// exactly the bytes a caller built before the field existed — the
    /// appended field costs a pre-existing client nothing.
    #[test]
    fn unset_cache_encodes_byte_for_byte_as_the_pre_field_request() {
        let legacy = PreCacheSubmitJobRequest {
            base_model: "local:tiny-bert".to_string(),
            idempotency_key: "dedupe-1".to_string(),
            world_size: 2,
        };
        let current = job_pb::SubmitJobRequest {
            spec: None,
            base_model: "local:tiny-bert".to_string(),
            config: None,
            idempotency_key: "dedupe-1".to_string(),
            world_size: 2,
            cache: 0,
        };

        assert_eq!(current.encode_to_vec(), legacy.encode_to_vec());
    }

    /// BACKWARD. Bytes produced before `cache` existed decode into the current
    /// type with the policy `UNSPECIFIED` and lose nothing else.
    #[test]
    fn pre_field_bytes_decode_with_cache_unspecified() {
        let legacy = PreCacheSubmitJobRequest {
            base_model: "local:tiny-bert".to_string(),
            idempotency_key: "dedupe-1".to_string(),
            world_size: 2,
        };

        let decoded = job_pb::SubmitJobRequest::decode(legacy.encode_to_vec().as_slice())
            .expect("pre-field bytes decode into the current request");

        assert_eq!(decoded.cache, 0);
        assert_eq!(decoded.base_model, "local:tiny-bert");
        assert_eq!(decoded.idempotency_key, "dedupe-1");
        assert_eq!(decoded.world_size, 2);
    }

    /// FORWARD. A peer that predates `cache` decodes a request carrying it
    /// without error — the appended tag is unknown to it, skipped, and every
    /// field it does know survives.
    #[test]
    fn a_pre_field_decoder_skips_a_set_cache() {
        let current = job_pb::SubmitJobRequest {
            spec: None,
            base_model: "local:tiny-bert".to_string(),
            config: None,
            idempotency_key: "dedupe-1".to_string(),
            world_size: 2,
            cache: crate::proto::inference::CachePolicy::Use as i32,
        };

        let decoded = PreCacheSubmitJobRequest::decode(current.encode_to_vec().as_slice())
            .expect("a pre-field decoder skips the appended tag");

        assert_eq!(decoded.base_model, "local:tiny-bert");
        assert_eq!(decoded.idempotency_key, "dedupe-1");
        assert_eq!(decoded.world_size, 2);
    }

    /// SET. A full fine-tune submit carrying `CACHE_POLICY_USE` round-trips
    /// unchanged, alongside a non-default `world_size` — the two independent
    /// scalars do not clobber each other.
    #[test]
    fn a_set_cache_round_trips_on_a_full_request() {
        let original = job_pb::SubmitJobRequest {
            spec: Some(job_pb::submit_job_request::Spec::FineTune(
                pb::FineTuneSpec {
                    source: "training".to_string(),
                    columns: vec!["text_a".to_string(), "text_b".to_string()],
                    method: pb::FineTuneMethod::Lora as i32,
                    task: crate::proto::inference::ModelTask::TextEmbedding as i32,
                },
            )),
            base_model: "local:tiny-bert".to_string(),
            config: Some(pb::FineTuneConfig {
                epochs: Some(2),
                ..Default::default()
            }),
            idempotency_key: "dedupe-1".to_string(),
            world_size: 4,
            cache: crate::proto::inference::CachePolicy::Use as i32,
        };

        let decoded = job_pb::SubmitJobRequest::decode(original.encode_to_vec().as_slice())
            .expect("a request carrying the policy round-trips");

        assert_eq!(
            decoded.cache,
            crate::proto::inference::CachePolicy::Use as i32
        );
        assert_eq!(decoded.world_size, 4);
        assert_eq!(decoded, original);
    }

    /// The policy rides on the request, not the kind, so the SAME field
    /// serves a graph fine-tune: the two LoRA kinds cannot diverge on it,
    /// because there is only one place to put it.
    #[test]
    fn the_same_cache_field_serves_the_graph_fine_tune_kind() {
        let original = job_pb::SubmitJobRequest {
            spec: Some(job_pb::submit_job_request::Spec::GraphFineTune(
                pb::GraphFineTuneSpec {
                    sources: None,
                    sample_config: None,
                },
            )),
            base_model: "local:tiny-bert".to_string(),
            config: None,
            idempotency_key: String::new(),
            world_size: 0,
            cache: crate::proto::inference::CachePolicy::Use as i32,
        };

        let decoded = job_pb::SubmitJobRequest::decode(original.encode_to_vec().as_slice())
            .expect("a graph fine-tune carrying the policy round-trips");

        assert_eq!(
            decoded.cache,
            crate::proto::inference::CachePolicy::Use as i32
        );
        assert_eq!(decoded, original);
    }
}
