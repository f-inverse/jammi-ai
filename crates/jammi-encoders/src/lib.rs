//! Candle-native encoders for sentence and cross-modal embeddings, with
//! built-in PEFT support via [`jammi_lora`].
//!
//! Each of [`Bert`], [`DistilBert`], and [`ModernBert`] is a self-contained
//! text encoder whose attention/FFN linears can be selectively LoRA-augmented
//! via [`jammi_lora::LoraBuildConfig`]. [`ClipText`] is the OpenCLIP text
//! tower that produces shared-latent embeddings compatible with an OpenCLIP
//! vision tower for cross-modal text↔image search. [`HtsatAudio`] is the
//! HTSAT-Swin CLAP audio tower that produces shared-latent embeddings from a
//! 4-channel fusion spectrogram, compatible with the CLAP text tower for
//! cross-modal text↔audio search. [`AnyEncoder`] / [`AnyAudioEncoder`] are the
//! closed-enum dispatchers that let a single caller hold any of the
//! text / audio families respectively.
//!
//! [`AnyContextPredictor`] is the amortized in-context predictor family
//! ([`Cnp`] / [`AttnCnp`] / [`Tnp`]): given a target and its context set, it
//! emits a predictive-distribution head in one differentiable forward pass —
//! the learned-aggregation point of the neural-process spectrum, dispatched by
//! the same closed-enum pattern as the encoder families.

pub mod aggregate;
pub mod audio;
pub mod bert;
pub mod clip_text;
pub mod context;
pub mod distilbert;
pub mod htsat_audio;
pub mod modernbert;

mod any;
mod error;
mod layer_norm;
mod mask;
mod pooling;

pub use aggregate::{segment_aggregate, SegmentReduce};
pub use any::AnyEncoder;
pub use audio::{AnyAudioEncoder, AudioEncoder};
pub use bert::{Bert, BertConfig};
pub use clip_text::{ClipText, ClipTextConfig};
pub use context::{
    attention_weights, multi_head_attention, AnyContextPredictor, AttnCnp, Cnp,
    ContextArchitecture, ContextEpisode, ContextPredictorConfig, Tnp,
};
pub use distilbert::{DistilBert, DistilBertConfig};
pub use error::EncoderError;
pub use htsat_audio::{HtsatAudio, HtsatAudioConfig};
pub use modernbert::{ModernBert, ModernBertConfig};
pub use pooling::{pool_and_normalize, Pooling};
