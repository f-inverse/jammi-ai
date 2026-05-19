//! Error type for the `jammi-encoders` crate.

use thiserror::Error;

/// Errors produced by encoder construction and forward passes.
#[derive(Debug, Error)]
pub enum EncoderError {
    /// Underlying candle tensor operation failed.
    #[error("Tensor: {0}")]
    Tensor(#[from] candle_core::Error),
    /// LoRA primitive operation failed (from `jammi_lora`).
    #[error("LoRA: {0}")]
    Lora(#[from] jammi_lora::LoraError),
    /// Input sequence longer than the model's positional capacity.
    #[error("Sequence length {seq} exceeds model's max_position_embeddings {max}")]
    SequenceTooLong { seq: usize, max: usize },
    /// Caller passed an invalid configuration value or builder selection.
    #[error("Configuration: {0}")]
    Config(String),
    /// Filesystem operation failed while loading weights or adapter files.
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
}
