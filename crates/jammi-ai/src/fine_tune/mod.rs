//! Fine-tuning: LoRA adapter training on user data.
//!
//! This module provides LoRA-based fine-tuning for embedding and classification
//! models. Training data is read through DataFusion, so any registered source
//! (Parquet, CSV, Postgres) works as long as it has the right schema.

// The candle-backed training engine (data loading, the LoRA model, the trainer
// loop, the job handle). Gated behind the default-on `local` feature; the config
// vocabulary re-exported below lives on the `jammi-wire` substrate so a client
// can encode a fine-tune request without the engine.
#[cfg(feature = "local")]
pub mod adamw;
#[cfg(feature = "local")]
pub mod classifier;
#[cfg(feature = "local")]
pub mod data;
#[cfg(feature = "local")]
pub mod gradcache;
#[cfg(feature = "local")]
pub mod graph_sampler;
#[cfg(feature = "local")]
pub mod hard_negative_miner;
#[cfg(feature = "local")]
pub mod lora;
#[cfg(feature = "local")]
pub mod optimizer;
#[cfg(feature = "local")]
pub(crate) mod regression_loss;
#[cfg(feature = "local")]
pub mod resume;
#[cfg(feature = "local")]
pub mod spec;
#[cfg(feature = "local")]
pub mod target;
#[cfg(feature = "local")]
pub mod trainer;
#[cfg(feature = "local")]
pub mod training_job;
#[cfg(feature = "local")]
pub mod worker;

// The fine-tune request vocabulary — `FineTuneConfig`, the loss / schedule /
// dtype enums, `FineTuneMethod`, and the `jammi_lora` init/dtype re-exports —
// is transport-neutral and lives on the `jammi-wire` substrate (so the gRPC
// converters satisfy the orphan rule). It is re-exported here at its original
// paths so the engine's training modules and SDK consumers reach it as
// `jammi_ai::fine_tune::*`.
pub use jammi_wire::fine_tune::{
    BackboneDtype, ClassificationLoss, EarlyStoppingMetric, EmbeddingLoss, FineTuneConfig,
    FineTuneMethod, HardNegativeConfig, LoraInitMode, LrSchedule, RegressionLoss,
};
