//! `FineTuneService` gRPC implementation.
//!
//! Two verbs land on the wire: `StartFineTune` and `FineTuneStatus`. Like every
//! other engine-backed service, they are thin adapters over the
//! transport-agnostic [`Session`]/[`LocalSession`] abstraction (never raw
//! [`InferenceSession`] calls): proto in, one `Session::fine_tune` /
//! `Session::fine_tune_status` call, proto out. The service reimplements no
//! training, LoRA, or scan logic.
//!
//! `Session::fine_tune` returns a `FineTuneJobId`; the handler carries its
//! string into the response and the client polls `FineTuneStatus` by that id
//! until a terminal state. There is no progress stream — the abstraction
//! exposes none.
//!
//! The request's `FineTuneConfig` mirrors the engine's `FineTuneConfig` field
//! for field; `config_from_proto` maps it (and its nested loss / enum types)
//! onto the engine struct, leaving the engine's defaults in place for any field
//! a client did not set (an absent `config` message → the engine default
//! entirely). `task` reuses `jammi.v1.inference.ModelTask` via the shared
//! [`crate::grpc::wire::model_task_from_proto`].
//!
//! Tenant scope is read from the request's [`crate::grpc::session::
//! SessionTenant`] extension (set upstream by the shared `TenantInterceptor`)
//! and applied via `with_tenant_scoped`, matching every other engine-backed
//! gRPC surface.

use std::sync::Arc;

use jammi_ai::fine_tune::{
    BackboneDtype, ClassificationLoss, EarlyStoppingMetric, EmbeddingLoss, FineTuneConfig,
    FineTuneMethod, LoraInitMode, LrSchedule,
};
use jammi_ai::local_session::FineTuneJobId;
use jammi_ai::session::InferenceSession;
use jammi_ai::{LocalSession, Session};
use tonic::{Request, Response, Status};

use crate::grpc::proto::fine_tune as pb;
use crate::grpc::proto::fine_tune::fine_tune_service_server::FineTuneService;
use crate::grpc::wire::{
    map_engine_error, model_task_from_proto, require_nonempty, scoped, session_tenant,
};

/// Server-side handler for the fine-tune gRPC surface. Holds a shared engine
/// session it wraps in a [`LocalSession`] per call to reach the unified
/// transport surface.
pub struct FineTuneServer {
    session: Arc<InferenceSession>,
}

impl FineTuneServer {
    pub fn new(session: Arc<InferenceSession>) -> Self {
        Self { session }
    }

    /// A [`Session`] over the shared engine; see [`crate::grpc::inference`] for
    /// the tenant-scope wiring rationale.
    fn local(&self) -> Session {
        Session::Local(LocalSession::new(Arc::clone(&self.session)))
    }
}

#[tonic::async_trait]
impl FineTuneService for FineTuneServer {
    async fn start_fine_tune(
        &self,
        request: Request<pb::StartFineTuneRequest>,
    ) -> Result<Response<pb::StartFineTuneResponse>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        require_nonempty(&req.source_id, "source_id")?;
        require_nonempty(&req.base_model, "base_model")?;
        if req.columns.is_empty() {
            return Err(Status::invalid_argument("columns is required"));
        }
        let method = method_from_proto(req.method)?;
        let task = model_task_from_proto(req.task)?;
        let config = req.config.map(config_from_proto).transpose()?;
        let session = self.local();

        let job_id = scoped(&self.session, tenant, || {
            session.fine_tune(
                &req.source_id,
                &req.base_model,
                &req.columns,
                method,
                task,
                config,
            )
        })
        .await
        .map_err(map_engine_error)?;

        Ok(Response::new(pb::StartFineTuneResponse {
            job_id: job_id.0,
        }))
    }

    async fn fine_tune_status(
        &self,
        request: Request<pb::FineTuneStatusRequest>,
    ) -> Result<Response<pb::FineTuneStatusResponse>, Status> {
        let tenant = session_tenant(&request);
        let req = request.into_inner();
        require_nonempty(&req.job_id, "job_id")?;
        let id = FineTuneJobId(req.job_id);
        let session = self.local();

        let status = scoped(&self.session, tenant, || session.fine_tune_status(&id))
            .await
            .map_err(map_engine_error)?;

        Ok(Response::new(pb::FineTuneStatusResponse { status }))
    }
}

/// Map the wire [`pb::FineTuneMethod`] onto the engine's [`FineTuneMethod`]. An
/// unspecified method is rejected — a request that names no method is a client
/// error, not a silent default.
fn method_from_proto(method: i32) -> Result<FineTuneMethod, Status> {
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
/// like `FineTuneConfig::default()` for the rest. Validation stays in the
/// engine (`FineTuneConfig::validate`, called inside `fine_tune`); this is a
/// pure shape map.
fn config_from_proto(c: pb::FineTuneConfig) -> Result<FineTuneConfig, Status> {
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
    })
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
