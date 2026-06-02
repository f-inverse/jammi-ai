//! Candle-native encoders for sentence and cross-modal embeddings, with
//! built-in PEFT support via [`jammi_lora`].
//!
//! Each of [`Bert`], [`DistilBert`], and [`ModernBert`] is a self-contained
//! text encoder whose attention/FFN linears can be selectively LoRA-augmented
//! via [`jammi_lora::LoraBuildConfig`]. [`ClipText`] is the OpenCLIP text
//! tower that produces shared-latent embeddings compatible with an OpenCLIP
//! vision tower for cross-modal text↔image search. [`ClapAudio`] is the
//! CLAP audio tower that produces shared-latent embeddings from a log-mel
//! spectrogram, compatible with the CLAP text tower for cross-modal
//! text↔audio search. [`AnyEncoder`] / [`AnyAudioEncoder`] are the
//! closed-enum dispatchers that let a single caller hold any of the
//! text / audio families respectively.

pub mod audio;
pub mod bert;
pub mod clap_audio;
pub mod clip_text;
pub mod distilbert;
pub mod htsat_audio;
pub mod modernbert;

mod any;
mod error;
mod layer_norm;
mod mask;
mod pooling;

pub use any::AnyEncoder;
pub use audio::{AnyAudioEncoder, AudioEncoder};
pub use bert::{Bert, BertConfig};
pub use clap_audio::{ClapAudio, ClapAudioConfig};
pub use clip_text::{ClipText, ClipTextConfig};
pub use distilbert::{DistilBert, DistilBertConfig};
pub use error::EncoderError;
pub use modernbert::{ModernBert, ModernBertConfig};
pub use pooling::{pool_and_normalize, Pooling};
