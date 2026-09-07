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
//! cross-modal text↔audio search. All three towers carry LoRA sites on the
//! same [`jammi_lora::MaybeLoraLinear`] seam the BERT family uses, reached
//! through their own builders ([`ClipText::builder`],
//! [`OpenClipVisionTransformer::builder`], [`HtsatAudio::builder`]).
//!
//! [`AnyEncoder`] is the ONE closed-enum dispatcher across all three
//! modalities: [`EncoderInput`] (owned twin [`OwnedEncoderInput`]) names a
//! batch for a given [`Modality`], and a mismatch between an input's
//! modality and the encoder's own is a typed refusal.
//!
//! [`AnyContextPredictor`] is the amortized in-context predictor family
//! ([`Cnp`] / [`AttnCnp`] / [`Tnp`]): given a target and its context set, it
//! emits a predictive-distribution head in one differentiable forward pass —
//! the learned-aggregation point of the neural-process spectrum, dispatched by
//! the same closed-enum pattern as the encoder families.

pub mod aggregate;
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
// The shared per-layer fused-attention cascade (flash → mem_efficient →
// attention_block_fused → eager) — see its own module doc. Extracted from
// `modernbert` (issue #462) so `bert`/`distilbert` can share it too.
mod attention_cascade;
mod error;
// The wave-3 GGUF-quantized-weight construction seam (`FrozenWeightLookup`)
// shared by `bert`/`distilbert`/`modernbert` — see its own module doc.
mod frozen_weight_source;
mod layer_norm;
mod lora_site;
mod mask;
// The ONE OpenCLIP residual-attention block (`Mlp`, `ResidualAttentionBlock`,
// the site table and the key-prefixed traversals), shared by the `clip_text`
// and `open_clip_vision` towers — see its own module doc, including why the
// two towers must NOT share an adapter key namespace.
mod open_clip_block;
mod pooling;
#[cfg(test)]
mod test_support;

pub use aggregate::{segment_aggregate, SegmentReduce};
// There is deliberately NO second, audio-only dispatcher or trait beside
// `AnyEncoder`: audio is a first-class `AnyEncoder` variant carrying the same
// real training hooks every other variant does, and a parallel audio-only
// dispatcher would be exactly the family-erasure duplication `AnyEncoder`
// exists to avoid.
pub use any::{AnyEncoder, EncoderInput, Modality, OwnedEncoderInput};
pub use bert::{Bert, BertConfig};
pub use clip_text::{ClipText, ClipTextBuilder, ClipTextConfig};
pub use context::{
    attention_weights, multi_head_attention, AnyContextPredictor, AttnCnp, Cnp,
    ContextArchitecture, ContextEpisode, ContextPredictorConfig, Tnp,
};
pub use distilbert::{DistilBert, DistilBertConfig};
pub use error::EncoderError;
pub use frozen_weight_source::FrozenWeightLookup;
pub use htsat_audio::{HtsatAudio, HtsatAudioBuilder, HtsatAudioConfig};
pub use modernbert::{ModernBert, ModernBertConfig};
pub use open_clip_vision::{
    OpenClipVisionBuilder, OpenClipVisionConfig, OpenClipVisionTransformer,
};
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

/// A snapshot of the memory-efficient (chunked) attention cascade's
/// dispatch counts (`mem_efficient_attention`, M2 — see
/// `crate::modernbert`'s `mem_efficient_attention_predicate`/
/// `ModernBertAttention::forward_training_attention` for the `admit_cascade`
/// call site this counts). Mirrors [`attention_block_flash_dispatch_snapshot`]'s
/// own read-API shape and rationale exactly (a cascade counter is keyed by
/// op name in `jammi_kernels::admission`'s process-wide registry, not
/// stored per-crate).
pub fn mem_efficient_attention_dispatch_snapshot(
) -> jammi_kernels::admission::CascadeDispatchSnapshot {
    jammi_kernels::admission::cascade_counters_for("mem_efficient_attention").snapshot()
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
