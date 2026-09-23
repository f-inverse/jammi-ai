//! Run a model over a DataFusion relation as a physical stage.
//!
//! [`plan_inference`] builds the one plan every model forward runs in: the
//! rows are ordered, numbered and chunked by a token budget once, below
//! every exchange ([`numbered`]); each chunk is prepared on the host,
//! admitted against its device and forwarded ([`runner`]); the output
//! carries the task's columns behind a common prefix ([`schema`],
//! [`adapter`]); and the plan is placeable on another process through the
//! wire form this crate provides ([`wire`]) and the runtime that process
//! binds ([`runtime`]).
//!
//! The crate extends DataFusion at its own seams — [`ExecutionPlan`]
//! nodes and a [`PhysicalOptimizerRule`] — and binds to a model through one
//! trait pair, [`ModelRuntime`] and [`BoundModel`], so a consumer brings its
//! own model cache, device admission and forward. It depends on no engine.
//!
//! [`ExecutionPlan`]: datafusion::physical_plan::ExecutionPlan
//! [`PhysicalOptimizerRule`]: datafusion::physical_optimizer::PhysicalOptimizerRule

pub mod adapter;
pub mod chunk;
pub mod columns;
pub mod device;
pub mod error;
pub mod exec;
pub mod key_check;
pub mod numbered;
pub mod observer;
pub mod output;
pub mod row_cost;
pub mod runner;
pub mod runtime;
pub mod schema;
pub mod source;
pub mod spec;
pub mod task;
pub mod wire;

pub use device::ComputeDeviceKind;
pub use error::{Error, Result};
pub use exec::{inference_specs, plan_inference, InferenceExec, InferenceFanOut};
pub use numbered::NumberedInputExec;
pub use output::BackendOutput;
pub use runtime::{
    BoundModel, ForwardError, ForwardPermit, InferenceRuntime, ModelRuntime, Prepared,
};
pub use source::ModelSource;
pub use spec::{InferenceSpec, RowOrder};
pub use task::ModelTask;
