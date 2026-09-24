//! Jammi's DataFusion extension: what the engine adds to DataFusion at
//! DataFusion's own seams — [`ExecutionPlan`] nodes, a
//! [`PhysicalOptimizerRule`], and the wire form that carries them to
//! another process — over the vocabulary every model stage shares: what a
//! model computes ([`ModelTask`]), where it is loaded from
//! ([`ModelSource`]) and the kind of device a plan runs on
//! ([`ComputeDeviceKind`]).
//!
//! [`inference`] is the forward stage: a relation's rows ordered, numbered
//! and chunked by a token budget once, each chunk prepared on the host,
//! admitted against its device and forwarded, the output behind a common
//! prefix. It binds to a model through one trait pair,
//! [`ModelRuntime`] and [`BoundModel`], so a consumer brings its own
//! model cache, device admission and forward.
//!
//! [`training`] is the training stage: a claimed training job run as one
//! task where its device is, through a [`TrainingRunner`] the consumer
//! implements with its own claim transfer, training loop and publish. The
//! crate depends on no engine.
//!
//! [`ExecutionPlan`]: datafusion::physical_plan::ExecutionPlan
//! [`PhysicalOptimizerRule`]: datafusion::physical_optimizer::PhysicalOptimizerRule

pub mod device;
pub mod error;
pub mod inference;
pub mod source;
pub mod task;
pub mod training;

pub use device::ComputeDeviceKind;
pub use error::{Error, Result};
pub use inference::exec::{inference_specs, plan_inference, InferenceExec, InferenceFanOut};
pub use inference::numbered::NumberedInputExec;
pub use inference::output::BackendOutput;
pub use inference::runtime::{
    BoundModel, ForwardError, ForwardPermit, InferenceRuntime, ModelRuntime, Prepared,
};
pub use inference::spec::{InferenceSpec, RowOrder};
pub use source::ModelSource;
pub use task::ModelTask;
pub use training::exec::{
    NoTrainingRunner, TrainingExec, TrainingJob, TrainingOutcome, TrainingRunner,
};
