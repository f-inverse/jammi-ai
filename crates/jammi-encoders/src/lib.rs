//! Candle-native encoders for sentence and cross-modal embeddings, with
//! built-in PEFT support via [`jammi_lora`].
//!
//! Each of [`Bert`], [`DistilBert`], and [`ModernBert`] is a self-contained
//! text encoder whose attention/FFN linears can be selectively LoRA-augmented
//! via [`jammi_lora::LoraBuildConfig`]. [`ClipText`] is the OpenCLIP text
//! tower that produces shared-latent embeddings compatible with an OpenCLIP
//! vision tower for cross-modal text↔image search. [`AnyEncoder`] is the
//! closed-enum dispatch that lets a single caller hold any of the four.

pub mod bert;
pub mod clip_text;
pub mod distilbert;
pub mod modernbert;

mod any;
mod error;
mod layer_norm;
mod mask;
mod pooling;

pub use any::AnyEncoder;
pub use bert::{Bert, BertConfig};
pub use clip_text::{ClipText, ClipTextConfig};
pub use distilbert::{DistilBert, DistilBertConfig};
pub use error::EncoderError;
pub use modernbert::{ModernBert, ModernBertConfig};
pub use pooling::{pool_and_normalize, Pooling};
