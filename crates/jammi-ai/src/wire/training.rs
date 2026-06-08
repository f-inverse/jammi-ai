//! `TrainingService` proto↔domain conversions.
//!
//! The request `FineTuneConfig` mirrors the engine's [`FineTuneConfig`] field
//! for field; decode maps it (and its nested loss / enum types) onto the engine
//! struct, leaving the engine's defaults for any field left `UNSPECIFIED` (an
//! absent `config` message → the engine default entirely). The `StartTraining`
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
            regression_loss: c
                .regression_loss
                .map(regression_loss_from_proto)
                .transpose()?,
            quantile_levels: c.quantile_levels,
        })
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
