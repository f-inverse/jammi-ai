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
pub mod batch_bucket;
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

/// The rayon global-pool thread count backing the media front end's parallel
/// decode/preprocess stages (`inference::audio_preprocess` /
/// `inference::image_preprocess`'s `decode_*_batch` and `preprocess_*_batch`
/// functions, and `trainer::Trainer`'s `audio_encoder_input` /
/// `image_encoder_input` call sites). This is the POOL size — `rayon::
/// current_num_threads()` — NOT the count of threads that actually ran a
/// given batch's chunks (effective parallelism is `min(pool, n)`, emergent
/// and never recorded); `host.logical_cpus` already covers the machine-wide
/// count separately. Exposed for `FinetuneRunTier.rayon_pool_threads`
/// provenance (never identity).
#[cfg(feature = "local")]
pub fn media_front_end_pool_threads() -> usize {
    rayon::current_num_threads()
}

// The fine-tune request vocabulary — `FineTuneConfig`, the loss / schedule /
// dtype enums, `FineTuneMethod`, and the `jammi_lora` init/dtype re-exports —
// is transport-neutral and lives on the `jammi-wire` substrate (so the gRPC
// converters satisfy the orphan rule). It is re-exported here at its original
// paths so the engine's training modules and SDK consumers reach it as
// `jammi_ai::fine_tune::*`.
pub use jammi_wire::fine_tune::{
    ClassificationLoss, ComputePrecision, EarlyStoppingMetric, EmbeddingLoss, ExampleLoss,
    FineTuneConfig, FineTuneMethod, HardNegativeConfig, HeldOutLoss, LoraInitMode, LrSchedule,
    RegressionLoss,
};
