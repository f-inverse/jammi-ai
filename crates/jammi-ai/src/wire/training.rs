//! `TrainingService` proto↔domain conversions.
//!
//! The request `FineTuneConfig` mirrors the engine's [`FineTuneConfig`] field
//! for field; decode starts from [`FineTuneConfig::default()`] and overrides a
//! field only when the wire carries it — each scalar knob has explicit presence
//! (`optional`), so an unset field resolves to the engine default rather than a
//! literal zero, and an absent `config` message → the engine default entirely.
//! The engine is thus the single source of default values for every client. The
//! `StartTraining`
//! spec `oneof` mirrors the engine's [`TrainingSpec`] enum variant-for-variant,
//! field-for-field: a decoded request reconstructs the identical engine spec, so
//! a remote-submitted job is byte-identical to one submitted in-process.
//! Validation stays in the engine (the submit verbs call `validate`); this is a
//! pure shape map. `task` reuses `jammi.v1.inference.ModelTask` via the shared
//! [`super::model_task_from_proto`].

use tonic::Status;

use crate::fine_tune::{
    BackboneDtype, ClassificationLoss, EarlyStoppingMetric, EmbeddingLoss, FineTuneConfig,
    FineTuneMethod, HardNegativeConfig, LoraInitMode, LrSchedule, RegressionLoss,
};

// The spec oneof ↔ engine `TrainingSpec` conversions touch the engine-side spec
// vocabulary (`TrainingSpec` / graph sampler / context-predictor config), which
// lives behind the `local` feature; the `FineTuneConfig` config vocabulary below
// stays transport-neutral so a thin wire-only client can still encode it.
#[cfg(feature = "local")]
use crate::fine_tune::graph_sampler::{EdgeProvenance, GraphFineTuneSources, GraphSampleConfig};
#[cfg(feature = "local")]
use crate::fine_tune::spec::{TrainingCommon, TrainingSpec};
#[cfg(feature = "local")]
use crate::pipeline::context_predictor::{
    ContextArchitecture, ContextPredictorTrainConfig, GaussianObjective, PredictiveHead,
};
#[cfg(feature = "local")]
use crate::wire::{model_task_from_proto, model_task_to_proto};

use crate::wire::proto::training as pb;

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

// ─── StartTraining spec oneof ↔ engine TrainingSpec ──────────────────────────
//
// The `oneof` carries the verb that produced the job; decode reconstructs the
// engine [`TrainingSpec`] field-for-field so a worker re-runs the identical job.
// The two LoRA fine-tune kinds carry their base-model + config in the request's
// common `base_model`/`config` fields (folded into [`TrainingCommon`]); the
// context-predictor kind carries its full budget inside `predictor_spec`.

/// Decode a [`pb::StartTrainingRequest`] into the engine [`TrainingSpec`]. The
/// `oneof` selects the variant; `base_model`/`config` fold into the two LoRA
/// kinds' [`TrainingCommon`]. A request with no spec set is malformed.
#[cfg(feature = "local")]
pub fn training_spec_from_proto(req: pb::StartTrainingRequest) -> Result<TrainingSpec, Status> {
    let pb::StartTrainingRequest {
        spec,
        base_model,
        config,
    } = req;
    let spec =
        spec.ok_or_else(|| Status::invalid_argument("StartTraining request carries no spec"))?;
    match spec {
        pb::start_training_request::Spec::FineTune(ft) => {
            let common = lora_common_from_proto(base_model, config)?;
            if ft.source.is_empty() {
                return Err(Status::invalid_argument("source is required"));
            }
            // A column-source fine-tune with no columns has no training data to
            // detect a format from — a client error, rejected at decode rather
            // than deferred to a failing worker.
            if ft.columns.is_empty() {
                return Err(Status::invalid_argument("columns is required"));
            }
            Ok(TrainingSpec::FineTune {
                source: ft.source,
                columns: ft.columns,
                method: method_from_proto(ft.method)?,
                task: model_task_from_proto(ft.task)?,
                common,
            })
        }
        pb::start_training_request::Spec::GraphFineTune(g) => {
            let common = lora_common_from_proto(base_model, config)?;
            let sources = g.sources.ok_or_else(|| {
                Status::invalid_argument("graph_fine_tune spec carries no sources")
            })?;
            let sample_config = g.sample_config.ok_or_else(|| {
                Status::invalid_argument("graph_fine_tune spec carries no sample_config")
            })?;
            Ok(TrainingSpec::GraphFineTune {
                sources: graph_sources_from_proto(sources)?,
                sample_config: graph_sample_config_from_proto(sample_config),
                common,
            })
        }
        pb::start_training_request::Spec::ContextPredictor(cp) => {
            let predictor_spec = cp.predictor_spec.ok_or_else(|| {
                Status::invalid_argument("context_predictor spec carries no predictor_spec")
            })?;
            Ok(TrainingSpec::ContextPredictor {
                source: cp.source,
                predictor_spec: predictor_config_from_proto(predictor_spec)?,
            })
        }
    }
}

/// Encode the engine [`TrainingSpec`] (plus the common base-model + config the
/// LoRA kinds carry) onto a [`pb::StartTrainingRequest`] — the inverse of
/// [`training_spec_from_proto`], for the remote send side. The context-predictor
/// kind ignores `base_model`/`config` (its budget rides in `predictor_spec`), so
/// they are left empty there.
#[cfg(feature = "local")]
pub fn training_spec_to_proto(spec: &TrainingSpec) -> pb::StartTrainingRequest {
    match spec {
        TrainingSpec::FineTune {
            source,
            columns,
            method,
            task,
            common,
        } => pb::StartTrainingRequest {
            spec: Some(pb::start_training_request::Spec::FineTune(
                pb::FineTuneSpec {
                    source: source.clone(),
                    columns: columns.clone(),
                    method: method_to_proto(*method) as i32,
                    task: model_task_to_proto(*task) as i32,
                },
            )),
            base_model: common.base_model.clone(),
            config: Some(config_to_proto(&common.config)),
        },
        TrainingSpec::GraphFineTune {
            sources,
            sample_config,
            common,
        } => pb::StartTrainingRequest {
            spec: Some(pb::start_training_request::Spec::GraphFineTune(
                pb::GraphFineTuneSpec {
                    sources: Some(graph_sources_to_proto(sources)),
                    sample_config: Some(graph_sample_config_to_proto(sample_config)),
                },
            )),
            base_model: common.base_model.clone(),
            config: Some(config_to_proto(&common.config)),
        },
        TrainingSpec::ContextPredictor {
            source,
            predictor_spec,
        } => pb::StartTrainingRequest {
            spec: Some(pb::start_training_request::Spec::ContextPredictor(
                pb::ContextPredictorSpec {
                    source: source.clone(),
                    predictor_spec: Some(predictor_config_to_proto(predictor_spec)),
                },
            )),
            base_model: String::new(),
            config: None,
        },
    }
}

/// Fold the request's common `base_model` + optional `config` into a
/// [`TrainingCommon`] for the two LoRA fine-tune kinds. An empty base model is a
/// client error (the worker has nothing to adapt).
#[cfg(feature = "local")]
fn lora_common_from_proto(
    base_model: String,
    config: Option<pb::FineTuneConfig>,
) -> Result<TrainingCommon, Status> {
    if base_model.is_empty() {
        return Err(Status::invalid_argument("base_model is required"));
    }
    let config = config
        .map(FineTuneConfig::try_from)
        .transpose()?
        .unwrap_or_default();
    Ok(TrainingCommon { base_model, config })
}

#[cfg(feature = "local")]
fn graph_sources_from_proto(s: pb::GraphFineTuneSources) -> Result<GraphFineTuneSources, Status> {
    Ok(GraphFineTuneSources {
        node_source: s.node_source,
        id_column: s.id_column,
        text_column: s.text_column,
        edge_source: s.edge_source,
        src_column: s.src_column,
        dst_column: s.dst_column,
        provenance: edge_provenance_from_proto(s.provenance)?,
    })
}

#[cfg(feature = "local")]
fn graph_sources_to_proto(s: &GraphFineTuneSources) -> pb::GraphFineTuneSources {
    pb::GraphFineTuneSources {
        node_source: s.node_source.clone(),
        id_column: s.id_column.clone(),
        text_column: s.text_column.clone(),
        edge_source: s.edge_source.clone(),
        src_column: s.src_column.clone(),
        dst_column: s.dst_column.clone(),
        provenance: edge_provenance_to_proto(s.provenance) as i32,
    }
}

#[cfg(feature = "local")]
fn edge_provenance_from_proto(p: i32) -> Result<EdgeProvenance, Status> {
    match pb::EdgeProvenance::try_from(p) {
        Ok(pb::EdgeProvenance::Declared) => Ok(EdgeProvenance::Declared),
        Ok(pb::EdgeProvenance::Similarity) => Ok(EdgeProvenance::Similarity),
        Ok(pb::EdgeProvenance::Unspecified) | Err(_) => Err(Status::invalid_argument(
            "edge provenance must be DECLARED or SIMILARITY",
        )),
    }
}

#[cfg(feature = "local")]
fn edge_provenance_to_proto(p: EdgeProvenance) -> pb::EdgeProvenance {
    match p {
        EdgeProvenance::Declared => pb::EdgeProvenance::Declared,
        EdgeProvenance::Similarity => pb::EdgeProvenance::Similarity,
    }
}

#[cfg(feature = "local")]
fn graph_sample_config_from_proto(c: pb::GraphSampleConfig) -> GraphSampleConfig {
    GraphSampleConfig {
        walk_length: c.walk_length as usize,
        walks_per_node: c.walks_per_node as usize,
        return_p: c.return_p,
        in_out_q: c.in_out_q,
        hard_negatives: c.hard_negatives as usize,
        exclude_hops: c.exclude_hops as usize,
        min_negatives: c.min_negatives as usize,
        seed: c.seed,
    }
}

#[cfg(feature = "local")]
fn graph_sample_config_to_proto(c: &GraphSampleConfig) -> pb::GraphSampleConfig {
    pb::GraphSampleConfig {
        walk_length: c.walk_length as u32,
        walks_per_node: c.walks_per_node as u32,
        return_p: c.return_p,
        in_out_q: c.in_out_q,
        hard_negatives: c.hard_negatives as u32,
        exclude_hops: c.exclude_hops as u32,
        min_negatives: c.min_negatives as u32,
        seed: c.seed,
    }
}

#[cfg(feature = "local")]
fn predictor_config_from_proto(
    c: pb::ContextPredictorTrainConfig,
) -> Result<ContextPredictorTrainConfig, Status> {
    let head = c
        .head
        .ok_or_else(|| Status::invalid_argument("context predictor spec carries no head"))?;
    Ok(ContextPredictorTrainConfig {
        model_id: c.model_id,
        architecture: context_architecture_from_proto(c.architecture)?,
        key_column: c.key_column,
        task_column: c.task_column,
        value_column: c.value_column,
        context_k: c.context_k as usize,
        hidden_dim: c.hidden_dim as usize,
        num_heads: c.num_heads as usize,
        num_layers: c.num_layers as usize,
        head: predictive_head_from_proto(head)?,
        epochs: c.epochs as usize,
        learning_rate: c.learning_rate,
        grad_clip: c.grad_clip,
        test_task_fraction: c.test_task_fraction,
        min_task_count: c.min_task_count as usize,
        seed: c.seed,
    })
}

#[cfg(feature = "local")]
fn predictor_config_to_proto(c: &ContextPredictorTrainConfig) -> pb::ContextPredictorTrainConfig {
    pb::ContextPredictorTrainConfig {
        model_id: c.model_id.clone(),
        architecture: context_architecture_to_proto(c.architecture) as i32,
        key_column: c.key_column.clone(),
        task_column: c.task_column.clone(),
        value_column: c.value_column.clone(),
        context_k: c.context_k as u32,
        hidden_dim: c.hidden_dim as u32,
        num_heads: c.num_heads as u32,
        num_layers: c.num_layers as u32,
        head: Some(predictive_head_to_proto(&c.head)),
        epochs: c.epochs as u32,
        learning_rate: c.learning_rate,
        grad_clip: c.grad_clip,
        test_task_fraction: c.test_task_fraction,
        min_task_count: c.min_task_count as u32,
        seed: c.seed,
    }
}

#[cfg(feature = "local")]
fn context_architecture_from_proto(a: i32) -> Result<ContextArchitecture, Status> {
    match pb::ContextArchitecture::try_from(a) {
        Ok(pb::ContextArchitecture::Cnp) => Ok(ContextArchitecture::Cnp),
        Ok(pb::ContextArchitecture::AttnCnp) => Ok(ContextArchitecture::AttnCnp),
        Ok(pb::ContextArchitecture::Tnp) => Ok(ContextArchitecture::Tnp),
        Ok(pb::ContextArchitecture::Unspecified) | Err(_) => Err(Status::invalid_argument(
            "context predictor architecture must be CNP, ATTN_CNP, or TNP",
        )),
    }
}

#[cfg(feature = "local")]
fn context_architecture_to_proto(a: ContextArchitecture) -> pb::ContextArchitecture {
    match a {
        ContextArchitecture::Cnp => pb::ContextArchitecture::Cnp,
        ContextArchitecture::AttnCnp => pb::ContextArchitecture::AttnCnp,
        ContextArchitecture::Tnp => pb::ContextArchitecture::Tnp,
    }
}

#[cfg(feature = "local")]
fn predictive_head_from_proto(h: pb::PredictiveHead) -> Result<PredictiveHead, Status> {
    use pb::predictive_head::Head;
    match h.head {
        Some(Head::Gaussian(g)) => {
            let objective = g.objective.ok_or_else(|| {
                Status::invalid_argument("gaussian predictive head carries no objective")
            })?;
            Ok(PredictiveHead::Gaussian {
                objective: gaussian_objective_from_proto(objective)?,
            })
        }
        Some(Head::Quantile(q)) => Ok(PredictiveHead::Quantile { levels: q.levels }),
        None => Err(Status::invalid_argument(
            "predictive head carries no gaussian or quantile variant",
        )),
    }
}

#[cfg(feature = "local")]
fn predictive_head_to_proto(h: &PredictiveHead) -> pb::PredictiveHead {
    use pb::predictive_head::Head;
    let inner = match h {
        PredictiveHead::Gaussian { objective } => Head::Gaussian(pb::predictive_head::Gaussian {
            objective: Some(gaussian_objective_to_proto(*objective)),
        }),
        PredictiveHead::Quantile { levels } => Head::Quantile(pb::predictive_head::Quantile {
            levels: levels.clone(),
        }),
    };
    pb::PredictiveHead { head: Some(inner) }
}

#[cfg(feature = "local")]
fn gaussian_objective_from_proto(o: pb::GaussianObjective) -> Result<GaussianObjective, Status> {
    use pb::gaussian_objective::Objective;
    match o.objective {
        Some(Objective::Nll(n)) => Ok(GaussianObjective::Nll { beta: n.beta }),
        Some(Objective::Crps(_)) => Ok(GaussianObjective::Crps),
        None => Err(Status::invalid_argument(
            "gaussian objective carries no nll or crps variant",
        )),
    }
}

#[cfg(feature = "local")]
fn gaussian_objective_to_proto(o: GaussianObjective) -> pb::GaussianObjective {
    use pb::gaussian_objective::Objective;
    let inner = match o {
        GaussianObjective::Nll { beta } => Objective::Nll(pb::gaussian_objective::Nll { beta }),
        GaussianObjective::Crps => Objective::Crps(pb::gaussian_objective::Crps {}),
    };
    pb::GaussianObjective {
        objective: Some(inner),
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
    /// decoding it back yields the engine default. This pins the `RemoteSession`
    /// path (which sends a full config) against the now-optional fields.
    #[test]
    fn config_to_proto_round_trips_through_decode() {
        let original = FineTuneConfig::default();
        let proto = config_to_proto(&original);
        let decoded = FineTuneConfig::try_from(proto).expect("round-trip decodes");
        assert_eq!(decoded, original);
    }

    /// Re-encode a [`pb::StartTrainingRequest`] into the spec it decoded from —
    /// `training_spec_to_proto` then `training_spec_from_proto` — so a remote
    /// `GraphFineTune` job is byte-identical to the in-process one. Every field
    /// of the source spec is a distinctive non-default value, and every field of
    /// the decoded spec is asserted individually: a dropped or mis-mapped field
    /// in either direction (proto missing it → the decode resolves a different
    /// value; the encode dropping it → the assert below fires) fails the test.
    #[cfg(feature = "local")]
    #[test]
    fn graph_fine_tune_spec_round_trips_field_for_field() {
        let original = TrainingSpec::GraphFineTune {
            sources: GraphFineTuneSources {
                node_source: "nodes_src".into(),
                id_column: "node_id".into(),
                text_column: "node_text".into(),
                edge_source: "edges_src".into(),
                src_column: "edge_from".into(),
                dst_column: "edge_to".into(),
                provenance: EdgeProvenance::Similarity,
            },
            sample_config: GraphSampleConfig {
                walk_length: 7,
                walks_per_node: 3,
                return_p: 0.25,
                in_out_q: 4.0,
                hard_negatives: 5,
                exclude_hops: 2,
                min_negatives: 9,
                seed: 0xDEAD_BEEF,
            },
            common: TrainingCommon {
                base_model: "graph-base".into(),
                // A non-default config knob so the common config round-trips too.
                config: FineTuneConfig {
                    lora_rank: 32,
                    ..FineTuneConfig::default()
                },
            },
        };

        let proto = training_spec_to_proto(&original);
        let decoded =
            training_spec_from_proto(proto).expect("graph spec round-trips through decode");

        let TrainingSpec::GraphFineTune {
            sources,
            sample_config,
            common,
        } = decoded
        else {
            panic!("decoded spec is not GraphFineTune");
        };

        assert_eq!(sources.node_source, "nodes_src");
        assert_eq!(sources.id_column, "node_id");
        assert_eq!(sources.text_column, "node_text");
        assert_eq!(sources.edge_source, "edges_src");
        assert_eq!(sources.src_column, "edge_from");
        assert_eq!(sources.dst_column, "edge_to");
        assert_eq!(sources.provenance, EdgeProvenance::Similarity);

        assert_eq!(sample_config.walk_length, 7);
        assert_eq!(sample_config.walks_per_node, 3);
        assert_eq!(sample_config.return_p, 0.25);
        assert_eq!(sample_config.in_out_q, 4.0);
        assert_eq!(sample_config.hard_negatives, 5);
        assert_eq!(sample_config.exclude_hops, 2);
        assert_eq!(sample_config.min_negatives, 9);
        assert_eq!(sample_config.seed, 0xDEAD_BEEF);

        assert_eq!(common.base_model, "graph-base");
        assert_eq!(common.config.lora_rank, 32);
    }

    /// The `ContextPredictor` spec round-trips field-for-field through
    /// `training_spec_to_proto` → `training_spec_from_proto`, with the Gaussian
    /// `Nll { beta }` head. Every scalar / column / budget knob is a distinctive
    /// non-default value asserted individually, so a dropped or mis-mapped field
    /// in either conversion direction fails.
    #[cfg(feature = "local")]
    #[test]
    fn context_predictor_spec_round_trips_gaussian_head() {
        let original = TrainingSpec::ContextPredictor {
            source: "episodes_src".into(),
            predictor_spec: ContextPredictorTrainConfig {
                model_id: "ctx-pred-1".into(),
                architecture: ContextArchitecture::Tnp,
                key_column: "row_key".into(),
                task_column: "cohort".into(),
                value_column: "outcome".into(),
                context_k: 13,
                hidden_dim: 256,
                num_heads: 8,
                num_layers: 6,
                head: PredictiveHead::Gaussian {
                    objective: GaussianObjective::Nll { beta: 0.5 },
                },
                epochs: 11,
                learning_rate: 3e-4,
                grad_clip: 2.5,
                test_task_fraction: 0.3,
                min_task_count: 7,
                seed: 0xC0FF_EE42,
            },
        };

        let proto = training_spec_to_proto(&original);
        let decoded =
            training_spec_from_proto(proto).expect("predictor spec round-trips through decode");

        let TrainingSpec::ContextPredictor {
            source,
            predictor_spec,
        } = decoded
        else {
            panic!("decoded spec is not ContextPredictor");
        };

        assert_eq!(source, "episodes_src");
        assert_eq!(predictor_spec.model_id, "ctx-pred-1");
        assert_eq!(predictor_spec.architecture, ContextArchitecture::Tnp);
        assert_eq!(predictor_spec.key_column, "row_key");
        assert_eq!(predictor_spec.task_column, "cohort");
        assert_eq!(predictor_spec.value_column, "outcome");
        assert_eq!(predictor_spec.context_k, 13);
        assert_eq!(predictor_spec.hidden_dim, 256);
        assert_eq!(predictor_spec.num_heads, 8);
        assert_eq!(predictor_spec.num_layers, 6);
        match predictor_spec.head {
            PredictiveHead::Gaussian {
                objective: GaussianObjective::Nll { beta },
            } => assert_eq!(beta, 0.5),
            other => panic!("expected Gaussian Nll head, got {other:?}"),
        }
        assert_eq!(predictor_spec.epochs, 11);
        assert_eq!(predictor_spec.learning_rate, 3e-4);
        assert_eq!(predictor_spec.grad_clip, 2.5);
        assert_eq!(predictor_spec.test_task_fraction, 0.3);
        assert_eq!(predictor_spec.min_task_count, 7);
        assert_eq!(predictor_spec.seed, 0xC0FF_EE42);
    }

    /// The predictor spec's other two head shapes also round-trip: the Gaussian
    /// `Crps` objective (no payload) and the `Quantile { levels }` head (a
    /// distinctive non-default level vector). These exercise the head `oneof`
    /// arms the Nll case above does not.
    #[cfg(feature = "local")]
    #[test]
    fn context_predictor_spec_round_trips_crps_and_quantile_heads() {
        for head in [
            PredictiveHead::Gaussian {
                objective: GaussianObjective::Crps,
            },
            PredictiveHead::Quantile {
                levels: vec![0.1, 0.5, 0.9],
            },
        ] {
            let original = TrainingSpec::ContextPredictor {
                source: "episodes_src".into(),
                predictor_spec: ContextPredictorTrainConfig {
                    model_id: "ctx-pred-2".into(),
                    architecture: ContextArchitecture::AttnCnp,
                    key_column: "row_key".into(),
                    task_column: "cohort".into(),
                    value_column: "outcome".into(),
                    context_k: 4,
                    hidden_dim: 64,
                    num_heads: 4,
                    num_layers: 2,
                    head: head.clone(),
                    epochs: 2,
                    learning_rate: 1e-3,
                    grad_clip: 1.0,
                    test_task_fraction: 0.2,
                    min_task_count: 3,
                    seed: 42,
                },
            };

            let proto = training_spec_to_proto(&original);
            let decoded =
                training_spec_from_proto(proto).expect("predictor spec round-trips through decode");

            let TrainingSpec::ContextPredictor { predictor_spec, .. } = decoded else {
                panic!("decoded spec is not ContextPredictor");
            };

            match (&head, &predictor_spec.head) {
                (
                    PredictiveHead::Gaussian {
                        objective: GaussianObjective::Crps,
                    },
                    PredictiveHead::Gaussian {
                        objective: GaussianObjective::Crps,
                    },
                ) => {}
                (
                    PredictiveHead::Quantile { levels: want },
                    PredictiveHead::Quantile { levels: got },
                ) => assert_eq!(got, want),
                (want, got) => panic!("head mismatch: wanted {want:?}, got {got:?}"),
            }
        }
    }
}
