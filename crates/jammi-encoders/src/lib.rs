//! Candle-native BERT-family encoders for sentence embeddings, with built-in
//! PEFT support via [`jammi_lora`].
//!
//! Each of [`Bert`], [`DistilBert`], and [`ModernBert`] is a self-contained
//! encoder whose attention/FFN linears can be selectively LoRA-augmented via
//! [`jammi_lora::LoraBuildConfig`]. [`AnyEncoder`] is the closed-enum
//! dispatch that lets a single caller hold any of the three.

pub mod bert;
pub mod distilbert;
pub mod modernbert;

mod any;
mod error;
mod layer_norm;
mod mask;
mod pooling;

pub use any::AnyEncoder;
pub use bert::{Bert, BertConfig};
pub use distilbert::{DistilBert, DistilBertConfig};
pub use error::EncoderError;
pub use modernbert::{ModernBert, ModernBertConfig};
pub use pooling::{pool_and_normalize, Pooling};
