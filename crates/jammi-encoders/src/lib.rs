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
pub mod precision;

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
pub use precision::compute_precision_to_dtype;

use candle_core::Tensor;

/// Contiguity-safe matmul — the single matmul primitive every encoder uses.
///
/// candle's **CUDA** matmul rejects an arbitrary-strided operand (a view left by
/// `narrow` / `transpose` / a RoPE op chain) with "matmul is only supported for
/// contiguous tensors"; its **CPU** matmul silently tolerates the same tensor.
/// That asymmetry is a whole class of CPU-passes / GPU-fails bug: an attention
/// score or context matmul fed a non-contiguous Q/K/V is correct on CPU yet
/// errors at model load on GPU — and because inference annotates per-row errors,
/// the failure is swallowed into empty output rather than surfaced.
///
/// Materialising both operands here, once, removes the class: no encoder call
/// site has to know which view ops leave which layout, and no future
/// architecture can reintroduce the bug by feeding a matmul a fresh
/// non-contiguous tensor. It is deliberately unconditional rather than
/// "`contiguous()` only when candle would reject it" — guessing candle's accepted
/// layouts would re-encode a dependency on its internal behaviour, the exact
/// fragility this replaces. `contiguous()` is a no-op on an already-contiguous
/// tensor, so a copy happens only where one was genuinely needed.
pub fn contiguous_matmul(a: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    a.contiguous()?.matmul(&b.contiguous()?)
}
