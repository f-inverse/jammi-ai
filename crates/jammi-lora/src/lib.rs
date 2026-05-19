//! Static-dispatch PEFT-style LoRA primitives for [candle].
//!
//! The crate exposes everything needed to add LoRA adapters to a candle-based
//! model without proc-macros or trait objects:
//!
//! - [`LoraLinear`] — a single LoRA-augmented linear layer.
//! - [`MaybeLoraLinear`] — closed-enum dispatch between a plain `Linear` and a
//!   [`LoraLinear`], so a model can be parameterised at construction and stays
//!   inlinable at every forward call.
//! - [`LoraBuildConfig`] / [`should_apply_lora`] / [`effective_rank`] —
//!   call-site decisions about which modules receive an adapter and at what
//!   rank.
//! - [`AdapterConfig`] + [`save_adapter`] / [`load_adapter`] — adapter
//!   directory persistence.
//!
//! [candle]: https://github.com/huggingface/candle

mod adapter;
mod config;
mod error;
mod init;
mod lora_linear;
mod save_load;
mod wrapper;

pub use adapter::{AdapterConfig, BackboneDtype};
pub use config::{effective_rank, should_apply_lora, LoraBuildConfig};
pub use error::LoraError;
pub use init::LoraInitMode;
pub use lora_linear::LoraLinear;
pub use save_load::{load_adapter, save_adapter};
pub use wrapper::MaybeLoraLinear;
