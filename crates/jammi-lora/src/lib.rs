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
mod init;
// The candle-backed primitives, their error type, and persistence. Gated behind
// the default-on `candle` feature so a config-vocabulary-only consumer (one that
// needs `FineTuneConfig`'s field types but never a tensor) pulls no candle stack.
// `LoraError`'s every constructor lives in these modules, so it rides the gate too.
#[cfg(feature = "candle")]
mod error;
#[cfg(feature = "candle")]
mod lora_linear;
#[cfg(feature = "candle")]
mod save_load;
// Seeded, candle-RNG-free init and dropout for the LoRA path. Only the
// candle-backed `lora_linear` consumes it, so it rides the same feature gate.
#[cfg(feature = "candle")]
mod seeded;
#[cfg(feature = "candle")]
mod wrapper;

pub use adapter::AdapterConfig;
pub use config::{effective_rank, should_apply_lora, LoraBuildConfig};
#[cfg(feature = "candle")]
pub use error::LoraError;
pub use init::LoraInitMode;
// `AdapterConfig`'s `backbone_dtype` field is typed on `ComputePrecision`
// (`jammi-numerics`'s candle-free compute-dtype vocabulary), re-exported here
// so a consumer that only needs the config vocabulary reaches the type
// without its own direct `jammi-numerics` import.
pub use jammi_numerics::ComputePrecision;
#[cfg(feature = "candle")]
pub use lora_linear::{
    lora_dropout_dispatch_snapshot, lora_epilogue_dispatch_snapshot, LoraLinear,
};
#[cfg(feature = "candle")]
pub use save_load::{load_adapter, save_adapter};
// The layer_id collision guard (audit advisory, post-4aa1303 round): a
// standalone, whole-run structural check over every name that will
// construct a `DropoutMasks` — see `seeded::assert_no_layer_id_collisions`'s
// own doc for the call-site contract and why this lives here rather than
// being threaded through `LoraLinear::new`'s per-layer construction.
#[cfg(feature = "candle")]
pub use seeded::assert_no_layer_id_collisions;
#[cfg(feature = "candle")]
pub use wrapper::MaybeLoraLinear;
