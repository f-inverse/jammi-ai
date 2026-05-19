//! Error type for the `jammi-lora` crate.

use thiserror::Error;

/// Errors produced by LoRA primitives, persistence, and configuration helpers.
#[derive(Debug, Error)]
pub enum LoraError {
    /// Underlying candle tensor operation failed.
    #[error("LoRA tensor: {0}")]
    Tensor(#[from] candle_core::Error),
    /// Filesystem operation failed while saving or loading an adapter.
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    /// JSON (de)serialisation of adapter metadata failed.
    #[error("Serialisation: {0}")]
    Serde(#[from] serde_json::Error),
    /// Caller passed an invalid configuration value.
    #[error("Configuration: {0}")]
    Config(String),
}
