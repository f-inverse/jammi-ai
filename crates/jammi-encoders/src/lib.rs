//! Candle-native encoders for sentence and cross-modal embeddings, with
//! built-in PEFT support via [`jammi_lora`].
//!
//! Each of [`Bert`], [`DistilBert`], and [`ModernBert`] is a self-contained
//! text encoder whose attention/FFN linears can be selectively LoRA-augmented
//! via [`jammi_lora::LoraBuildConfig`]. [`ClipText`] is the OpenCLIP text
//! tower and [`OpenClipVisionTransformer`] the OpenCLIP vision tower it is
//! compatible with, together producing shared-latent embeddings for
//! cross-modal text↔image search. [`HtsatAudio`] is the
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
pub mod open_clip_vision;
pub mod precision;

mod activations;
mod any;
mod attention;
mod error;
mod layer_norm;
mod mask;
mod pooling;
#[cfg(test)]
mod test_support;

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
pub use open_clip_vision::{OpenClipVisionConfig, OpenClipVisionTransformer};
pub use pooling::{pool_and_normalize, Pooling};
pub use precision::compute_precision_to_dtype;

use candle_core::Tensor;

/// A snapshot of the bias-free training-mode LayerNorm's fused/eager
/// dispatch counts (see `crate::layer_norm`'s module doc for the
/// admission mechanism this counts). `layer_norm` is a crate-private
/// module — its dispatch-counter static is otherwise unnameable from
/// outside this crate — so this is the read API a durable job record or
/// a bench report uses to state which kernel path actually ran during a
/// measured run (`jammi-bench`'s `finetune_step` tier reads this around
/// its step loop).
pub fn ln_dispatch_snapshot() -> jammi_kernels::admission::DispatchSnapshot {
    layer_norm::LN_DISPATCH_COUNTERS.snapshot()
}

/// A snapshot of ModernBERT's training-mode fused RoPE (rotate-half)
/// dispatch counts (see `crate::modernbert`'s `RotaryEmbedding` doc for
/// the admission mechanism this counts, and its `apply_training` for the
/// call site). `modernbert::ROPE_DISPATCH_COUNTERS` is `pub(crate)` —
/// this is the read API a durable job record or a bench report uses,
/// mirroring [`ln_dispatch_snapshot`] exactly.
pub fn rope_dispatch_snapshot() -> jammi_kernels::admission::DispatchSnapshot {
    modernbert::ROPE_DISPATCH_COUNTERS.snapshot()
}

/// A snapshot of ModernBERT's training-mode fused masked-softmax
/// (`jammi_kernels::ops::SoftmaxLastDimFused`) dispatch counts (see
/// `crate::modernbert`'s `ModernBertAttention::softmax_apply_training` for
/// the call site this counts). `modernbert::SOFTMAX_DISPATCH_COUNTERS` is
/// `pub(crate)` — this is the read API a durable job record or a bench
/// report uses, mirroring [`ln_dispatch_snapshot`] / [`rope_dispatch_snapshot`]
/// exactly.
pub fn softmax_dispatch_snapshot() -> jammi_kernels::admission::DispatchSnapshot {
    modernbert::SOFTMAX_DISPATCH_COUNTERS.snapshot()
}

/// A snapshot of ModernBERT's training-mode fused GeGLU
/// (`jammi_kernels::ops::GegluFused`) dispatch counts (see
/// `crate::modernbert`'s `geglu_apply_training` for the call site this
/// counts). `modernbert::GEGLU_DISPATCH_COUNTERS` is `pub(crate)` — this
/// is the read API a durable job record or a bench report uses, mirroring
/// [`ln_dispatch_snapshot`] / [`rope_dispatch_snapshot`] /
/// [`softmax_dispatch_snapshot`] exactly.
pub fn geglu_dispatch_snapshot() -> jammi_kernels::admission::DispatchSnapshot {
    modernbert::GEGLU_DISPATCH_COUNTERS.snapshot()
}

/// A snapshot of ModernBERT's training-mode fused whole-attention-block
/// (`jammi_kernels::ops::AttentionBlockFused`) dispatch counts (see
/// `crate::modernbert`'s `ModernBertAttention::forward_training_attention`
/// for the call site this counts).
/// `modernbert::ATTENTION_BLOCK_DISPATCH_COUNTERS` is `pub(crate)` — this
/// is the read API a durable job record or a bench report uses, mirroring
/// [`ln_dispatch_snapshot`] / [`rope_dispatch_snapshot`] /
/// [`softmax_dispatch_snapshot`] / [`geglu_dispatch_snapshot`] exactly.
pub fn attention_block_dispatch_snapshot() -> jammi_kernels::admission::DispatchSnapshot {
    modernbert::ATTENTION_BLOCK_DISPATCH_COUNTERS.snapshot()
}

/// A snapshot of the FlashAttention-2 cascade's dispatch counts
/// (`attention_block_flash`, P6 Stage B B3-dense — see
/// `crate::modernbert`'s `ModernBertAttention::forward_training_attention`
/// for the `admit_cascade` call site this counts). `(fused, eager,
/// declined)`, mirroring [`attention_block_dispatch_snapshot`]'s own
/// read-API shape — `jammi_kernels::admission::cascade_counters_for`
/// already owns the process-wide registry (a cascade counter is keyed by
/// op name, not stored per-crate the way the two-arm ops' statics are),
/// so this is a thin, direct pass-through rather than a local static.
pub fn attention_block_flash_dispatch_snapshot() -> jammi_kernels::admission::CascadeDispatchSnapshot
{
    jammi_kernels::admission::cascade_counters_for("attention_block_flash").snapshot()
}

/// Contiguity-safe matmul — the single matmul primitive every encoder uses.
///
/// candle's **CUDA** matmul rejects two operand layouts its **CPU** matmul
/// silently tolerates: a batched operand with **irregular batch strides** (a
/// batch dim scrambled by `permute` / `index` / `narrow`, e.g. the OpenCLIP
/// `qkv` split), and a **2-D operand whose row stride ≠ its column count** (a
/// `narrow`+`squeeze` slice, e.g. the ModernBERT classifier's CLS row). It does
/// *accept* a plain transpose (`.t()`) and regular batched transposes, so not
/// every view is a problem — but which ones are is candle-internal and
/// version-dependent. An operand of a rejected layout fails at model load on GPU,
/// and because inference annotates per-row errors, the failure is swallowed into
/// empty output rather than surfaced — a silent CPU-passes / GPU-fails class.
///
/// This primitive makes **both** operands contiguous unconditionally rather than
/// reasoning per-site about which view ops leave which layout. That per-site
/// reasoning is the error-prone analysis that let the bug exist — and that even a
/// careful reader gets wrong (a RoPE'd score matmul *looks* non-contiguous but
/// RoPE materialises its operands, so candle accepts it). Encoding the guard once,
/// depending on no candle internal, is worth a bounded cost: `contiguous()` is a
/// no-op on an already-contiguous operand and a single copy on a transposed one
/// candle would otherwise have accepted natively.
pub fn contiguous_matmul(a: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    a.contiguous()?.matmul(&b.contiguous()?)
}
