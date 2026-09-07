//! OpenCLIP text transformer.
//!
//! Loads weights from OpenCLIP safetensors checkpoints under their native
//! key layout (`token_embedding.weight`, `positional_embedding`,
//! `transformer.resblocks.{n}.attn.in_proj_weight`, ..., `ln_final.*`,
//! `text_projection`).
//!
//! The text tower is causally masked (lower-triangular attention), uses
//! QuickGelu in the MLP, pools by selecting the hidden state at the EOT
//! position (the argmax of the input token IDs along the sequence axis),
//! projects the pooled state through `text_projection` into the shared
//! CLIP latent space, and L2-normalizes the result.
//!
//! Forward output is `[batch, embed_dim]` in the same latent space as
//! [`crate::clip_text::ClipTextConfig::embed_dim`] vision-tower outputs,
//! enabling cross-modal cosine similarity.
//!
//! The residual block itself — MLP, fused-QKV attention, the four LoRA site
//! names and the key-prefixed traversals over the stack — is the SINGLE
//! `crate::open_clip_block`, shared verbatim with
//! [`crate::open_clip_vision`]'s tower; this module owns only what is
//! genuinely text-specific (the token/positional embeddings, the fixed 4×
//! MLP width, the cached causal mask, EOT pooling, `text_projection`). See
//! that module's "Two towers, two namespaces" section for why the two
//! towers' ADAPTER key roots must differ even though their block shape does
//! not.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{embedding, Embedding, VarBuilder, VarMap};
use jammi_lora::LoraBuildConfig;

use crate::error::EncoderError;
use crate::layer_norm::LayerNorm;
use crate::lora_site::{FrozenSiteHolder, LoraSite};
use crate::open_clip_block::{self as block, ResidualAttentionBlock};

/// Architecture configuration for the OpenCLIP text transformer.
///
/// `embed_dim` is the shared CLIP latent dimensionality (must match the
/// vision tower's projected output). `width` is the per-token hidden size
/// inside the text transformer; the `text_projection` matrix maps from
/// `width` to `embed_dim`.
#[derive(Debug, Clone)]
pub struct ClipTextConfig {
    /// Fixed sequence length — OpenCLIP uses 77 throughout.
    pub context_length: usize,
    /// Vocabulary size of the BPE tokenizer (49408 for the canonical
    /// `bpe_simple_vocab_16e6` vocabulary).
    pub vocab_size: usize,
    /// Per-token hidden size inside the transformer.
    pub width: usize,
    /// Number of transformer layers.
    pub layers: usize,
    /// Number of attention heads. Must divide `width` evenly.
    pub heads: usize,
    /// Shared CLIP latent dimensionality after `text_projection`.
    pub embed_dim: usize,
}

impl ClipTextConfig {
    /// Parse from an OpenCLIP config JSON (`open_clip_config.json`).
    ///
    /// Reads `model_cfg.embed_dim` and `model_cfg.text_cfg.{context_length,
    /// vocab_size, width, layers, heads}`, applying canonical OpenCLIP
    /// defaults (`context_length=77`, `vocab_size=49408`, `heads=width/64`)
    /// when fields are omitted.
    pub fn from_open_clip_config(config: &serde_json::Value) -> Result<Self, EncoderError> {
        let model_cfg = config
            .get("model_cfg")
            .ok_or_else(|| EncoderError::Config("OpenCLIP config missing 'model_cfg'".into()))?;
        let text_cfg = model_cfg.get("text_cfg").ok_or_else(|| {
            EncoderError::Config("OpenCLIP config missing 'model_cfg.text_cfg'".into())
        })?;
        let embed_dim = model_cfg
            .get("embed_dim")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                EncoderError::Config("OpenCLIP config missing 'model_cfg.embed_dim'".into())
            })? as usize;

        let width = text_cfg
            .get("width")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                EncoderError::Config("OpenCLIP config missing 'model_cfg.text_cfg.width'".into())
            })? as usize;
        let layers = text_cfg
            .get("layers")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                EncoderError::Config("OpenCLIP config missing 'model_cfg.text_cfg.layers'".into())
            })? as usize;
        let heads = text_cfg
            .get("heads")
            .and_then(|v| v.as_u64())
            .unwrap_or((width / 64) as u64) as usize;
        let context_length = text_cfg
            .get("context_length")
            .and_then(|v| v.as_u64())
            .unwrap_or(77) as usize;
        let vocab_size = text_cfg
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(49408) as usize;

        Ok(Self {
            context_length,
            vocab_size,
            width,
            layers,
            heads,
            embed_dim,
        })
    }
}

/// The adapter / `VarMap` key root this tower's LoRA sites live under.
///
/// The OpenCLIP text tower IS the checkpoint root (vision sits under
/// `visual.`), so its keys start at `resblocks` with no tower prefix — the
/// checkpoint's own vocabulary. See `crate::open_clip_block`'s "Two towers,
/// two namespaces" section for why the vision twin must NOT share this root.
pub(crate) const ADAPTER_BLOCK_ROOT: &str = "resblocks";

/// OpenCLIP text transformer.
///
/// Weight keys match the OpenCLIP safetensors layout directly: callers pass
/// a [`VarBuilder`] scoped at the root of the checkpoint (the same root used
/// for the vision tower under `visual.*`); this loader reads
/// `token_embedding.*`, `positional_embedding`, `transformer.resblocks.*`,
/// `ln_final.*`, and `text_projection`.
pub struct ClipText {
    token_embedding: Embedding,
    /// Learned `[context_length, width]` positional embedding (added, not
    /// rotary / sinusoidal). OpenCLIP stores this as a raw tensor at
    /// `positional_embedding`, not as an `Embedding` module.
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln_final: LayerNorm,
    /// `[width, embed_dim]` projection into the shared CLIP latent space.
    text_projection: Tensor,
    config: ClipTextConfig,
    /// Cached `[context_length, context_length]` additive causal mask
    /// (`0.0` lower-triangular, the BACKBONE DTYPE's own most-negative
    /// finite value above the diagonal — see [`build_causal_mask`]). Built
    /// once at load time in the backbone's dtype so the forward path slices
    /// instead of allocating AND never casts.
    causal_mask: Tensor,
    /// Wired through [`Self::set_training`]; read by [`Self::is_training`].
    training: bool,
}

impl ClipText {
    /// Start a builder with default settings: frozen LoRA config, F32
    /// backbone dtype, no adapter file. See [`ClipTextBuilder`].
    pub fn builder() -> ClipTextBuilder<'static> {
        ClipTextBuilder {
            lora: LoraBuildConfig::frozen(),
            backbone_dtype: DType::F32,
            adapter_file: None,
        }
    }

    /// Build the text transformer from a checkpoint-root [`VarBuilder`],
    /// fully frozen (no LoRA sites installed).
    ///
    /// Reads keys at the root level (no `text.` prefix): the OpenCLIP
    /// safetensors layout puts vision under `visual.*` and text under the
    /// root, so callers using the same checkpoint pass `vb` for text and
    /// `vb.pp("visual")` for vision.
    ///
    /// Routed through the SAME `Self::load_with` the builder uses, with a
    /// decline-everything site (`crate::lora_site::FrozenSiteHolder`), so
    /// there is one loader rather than two that could drift; the causal mask
    /// is built in `vb`'s own dtype. `builder().lora(frozen()).build(..)`
    /// therefore produces a bit-identical tower (asserted by
    /// `tests::builder_frozen_output_bits_equal_load`).
    pub fn load(vb: VarBuilder, config: &ClipTextConfig) -> Result<Self, EncoderError> {
        let holder = FrozenSiteHolder::new();
        let dtype = vb.dtype();
        Self::load_with(vb.clone(), config, dtype, &|_n| holder.site(&vb))
    }

    /// The one loader. `site_for` yields the [`LoraSite`] for block `n`
    /// (already scoped to that block's adapter subtree and carrying
    /// `layer_idx = Some(n)`); `mask_dtype` is the dtype the cached causal
    /// mask is materialised in — always the frozen backbone's dtype, so no
    /// forward ever casts it.
    fn load_with<'a>(
        vb: VarBuilder,
        config: &ClipTextConfig,
        mask_dtype: DType,
        site_for: &dyn Fn(usize) -> LoraSite<'a>,
    ) -> Result<Self, EncoderError> {
        let token_embedding = embedding(config.vocab_size, config.width, vb.pp("token_embedding"))?;
        let positional_embedding = vb.get(
            (config.context_length, config.width),
            "positional_embedding",
        )?;

        // OpenCLIP's text transformer uses a fixed 4x MLP ratio.
        let blocks = block::load_blocks(
            &vb,
            config.layers,
            config.width,
            config.heads,
            config.width * 4,
            site_for,
        )?;

        let ln_final = LayerNorm::new(config.width, 1e-5, true, vb.pp("ln_final"))?;
        let text_projection = vb.get((config.width, config.embed_dim), "text_projection")?;

        let causal_mask = build_causal_mask(config.context_length, mask_dtype, vb.device())?;

        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln_final,
            text_projection,
            config: config.clone(),
            causal_mask,
            training: false,
        })
    }

    /// Forward pass: token IDs → L2-normalized shared-latent embeddings.
    ///
    /// `input_ids` shape: `[batch, seq]` with `seq <= context_length`.
    /// `attention_mask` is accepted for API symmetry with [`crate::bert`] but
    /// is unused — OpenCLIP's text tower relies on the EOT-token pool to
    /// ignore padding, not on additive masking. Per-row EOT position is
    /// derived as `argmax(input_ids, dim=1)`: the OpenCLIP BPE tokenizer
    /// assigns the highest token ID (49407) to `<|endoftext|>`, so the
    /// argmax across the sequence is the EOT index even when padding
    /// (token 0) trails it.
    ///
    /// Output: `[batch, embed_dim]`, L2-normalized along the embedding axis.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        _attention_mask: &Tensor,
    ) -> Result<Tensor, EncoderError> {
        let (_batch, seq) = input_ids.dims2()?;
        if seq > self.config.context_length {
            return Err(EncoderError::SequenceTooLong {
                seq,
                max: self.config.context_length,
            });
        }

        // Token + positional embeddings.
        let token_emb = self.token_embedding.forward(input_ids)?;
        let pos_emb = self.positional_embedding.i((..seq, ..))?;
        let mut x = token_emb.broadcast_add(&pos_emb)?;

        // Sliced causal mask: [seq, seq] from the cached [context_length, context_length].
        let causal = self.causal_mask.i((..seq, ..seq))?;

        for block in &self.blocks {
            x = block.forward(&x, Some(&causal))?;
        }
        let x = self.ln_final.forward(&x)?;

        // EOT pooling: argmax of input IDs along sequence axis identifies
        // the `<|endoftext|>` token (highest ID in the OpenCLIP BPE vocab).
        let eot_indices = input_ids.argmax(1)?;
        let pooled = gather_at_indices(&x, &eot_indices)?;

        // Project into the shared CLIP latent space and L2-normalize.
        let projected = crate::contiguous_matmul(&pooled, &self.text_projection)?;
        l2_normalize(&projected)
    }

    /// Shared CLIP latent dimensionality of the output (`embed_dim`).
    pub fn embed_dim(&self) -> usize {
        self.config.embed_dim
    }

    /// Per-token hidden size inside the transformer (`width`). Distinct
    /// from [`Self::embed_dim`] — the `text_projection` matrix maps from
    /// `width` to `embed_dim`.
    pub fn hidden_size(&self) -> usize {
        self.config.width
    }

    /// Fixed input sequence length (`context_length`, 77 for canonical CLIP).
    pub fn context_length(&self) -> usize {
        self.config.context_length
    }

    /// Dtype the FROZEN BACKBONE weights are materialised at — read off a
    /// real weight (the token-embedding table), never a remembered builder
    /// setting, so it stays true for a tower built through [`Self::load`]
    /// from an arbitrary `VarBuilder` as well as one built through the
    /// builder's `backbone_dtype`. It is also the dtype [`Self::forward`]
    /// returns, and the dtype the cached causal mask was built in.
    ///
    /// This tower consumes INTEGER token ids, so unlike the two media towers
    /// it has no caller-supplied floating input whose dtype could disagree —
    /// see `crate::AnyEncoder::dtype` for the one caller class that needs a
    /// tower's dtype and why it is not a harmless default there.
    pub fn dtype(&self) -> DType {
        self.token_embedding.embeddings().dtype()
    }

    /// Switch every block's attention softmax AND every LayerNorm (`ln_1`,
    /// `ln_2` per block, plus `ln_final`) between the eval (no-backward) arm
    /// and the differentiable composed arm — see
    /// `MultiHeadAttention::forward`'s doc for the softmax truncation and
    /// `crate::layer_norm::LayerNorm`'s module doc for the identical
    /// `BackpropOp::none()` truncation in `candle_nn::LayerNorm`'s own fast
    /// path. Both are load-bearing together: before `ln_final` was gated,
    /// backward through the tower's public `forward` truncated at `ln_final`
    /// regardless of the attention fix, giving NO gradient at all (not even
    /// a partial one) to every parameter used earlier — see
    /// `tests::training_true_full_forward_reaches_every_parameter` for the
    /// full end-to-end oracle. Eval output is unaffected either way; only
    /// backward through this tower's gradient is correct in training mode.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        block::set_training(&mut self.blocks, training);
        self.ln_final.set_training(training);
    }

    /// Whether [`Self::set_training`] last set training mode. `false` from
    /// every constructor.
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Trainable tensors across every LoRA-wrapped site. Empty for a fully
    /// frozen tower.
    pub fn trainable_params(&self) -> Vec<&Tensor> {
        block::trainable_params(&self.blocks)
    }

    /// Named LoRA A/B tensors keyed `resblocks.{n}.{site}.lora_{a,b}` —
    /// `{site}` being the checkpoint's own leaf name (`in_proj`, `out_proj`,
    /// `c_fc`, `c_proj`), and `resblocks.{n}` the same subtree the builder
    /// registered them under.
    pub fn named_trainable_weights(&self) -> Result<HashMap<String, Tensor>, EncoderError> {
        block::named_trainable_weights(&self.blocks, ADAPTER_BLOCK_ROOT)
    }

    /// Restore LoRA A/B tensors from a [`Self::named_trainable_weights`]-shaped
    /// map. Missing keys are no-ops.
    pub fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<(), EncoderError> {
        block::load_weights(&mut self.blocks, weights, ADAPTER_BLOCK_ROOT);
        Ok(())
    }

    /// Per-site dropout-stream positions keyed
    /// `resblocks.{n}.{site}.dropout` — the SAME site prefix
    /// [`Self::named_trainable_weights`] uses, with
    /// `jammi_lora::MaybeLoraLinear::collect_dropout_position`'s own
    /// `.dropout` leaf appended. This is the resume state for the adapter's
    /// dropout streams.
    pub fn dropout_positions(&self) -> Result<HashMap<String, u64>, EncoderError> {
        block::dropout_positions(&self.blocks, ADAPTER_BLOCK_ROOT)
    }

    /// Restore each LoRA site's dropout-stream position from a
    /// [`Self::dropout_positions`]-shaped map. Missing keys are no-ops.
    pub fn restore_dropout_positions(
        &self,
        positions: &HashMap<String, u64>,
    ) -> Result<(), EncoderError> {
        block::restore_dropout_positions(&self.blocks, ADAPTER_BLOCK_ROOT, positions)
    }
}

/// Fluent builder for [`ClipText`]. Created via [`ClipText::builder`], on
/// the same shape [`crate::Bert`]'s builder uses.
pub struct ClipTextBuilder<'a> {
    lora: LoraBuildConfig<'a>,
    backbone_dtype: DType,
    adapter_file: Option<&'a Path>,
}

impl<'a> ClipTextBuilder<'a> {
    /// LoRA adapter configuration: which of this tower's sites (`in_proj`,
    /// `out_proj`, `c_fc`, `c_proj`, or `all-linear`) get wrapped, at what
    /// rank.
    pub fn lora(mut self, l: LoraBuildConfig<'a>) -> Self {
        self.lora = l;
        self
    }

    /// Dtype the frozen backbone tensors — and the cached causal mask — are
    /// materialised at. LoRA A/B always live in F32.
    pub fn backbone_dtype(mut self, d: DType) -> Self {
        self.backbone_dtype = d;
        self
    }

    /// Optional safetensors file to load already-trained LoRA A/B tensors
    /// from (inference). When `None`, A/B tensors are registered in the
    /// caller-supplied `VarMap` for training.
    pub fn adapter(mut self, p: Option<&'a Path>) -> Self {
        self.adapter_file = p;
        self
    }

    /// Materialise the tower from frozen safetensors checkpoint files.
    ///
    /// `weights_paths` are mmaped at [`Self::backbone_dtype`]; the trainable
    /// `VarBuilder` is the adapter file (F32) when one was supplied, else a
    /// `VarMap`-backed F32 builder. The frozen builder is scoped at the
    /// checkpoint ROOT, matching [`ClipText::load`]'s own contract (OpenCLIP
    /// puts text at the root and vision under `visual.*`).
    pub fn build(
        self,
        weights_paths: &[&Path],
        config: &ClipTextConfig,
        device: &Device,
        varmap: &VarMap,
    ) -> Result<ClipText, EncoderError> {
        let frozen_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(weights_paths, self.backbone_dtype, device)?
        };
        let trainable_vb = if let Some(adapter) = self.adapter_file {
            unsafe { VarBuilder::from_mmaped_safetensors(&[adapter], DType::F32, device)? }
        } else {
            VarBuilder::from_varmap(varmap, DType::F32, device)
        };
        // One trainable sub-builder per block, so each site's adapter key is
        // `resblocks.{n}.{site}.lora_{a,b}` — the exact keys
        // `ClipText::named_trainable_weights` writes.
        let block_vbs = block::block_var_builders(&trainable_vb, ADAPTER_BLOCK_ROOT, config.layers);
        ClipText::load_with(frozen_vb, config, self.backbone_dtype, &|n| LoraSite {
            lora_vb: &block_vbs[n],
            layer_idx: Some(n),
            lora: &self.lora,
            varmap,
        })
    }
}

/// Build the `[size, size]` additive causal mask in `dtype`: `0.0` on and
/// below the diagonal, `dtype`'s own most-negative finite value above it.
/// Constructed once at load time and sliced per forward.
///
/// # Why the sentinel follows the dtype (family D)
///
/// The mask is an ADDITIVE pre-softmax term, so it must be representable in
/// the dtype the attention scores are computed in. `f32::MIN` written into
/// an F16 tensor is not "a very negative number", it is `-inf` — and
/// `-inf + finite` composed through a softmax whose row is entirely masked
/// yields `NaN`, a confident wrong number rather than an error. Each dtype
/// therefore contributes its OWN most-negative finite value: `f32::MIN`,
/// `half::f16::MIN` (`-65504`), `half::bf16::MIN`. All three are exactly
/// representable as `f32`, so the vector is built once in `f32` with the
/// chosen sentinel and cast EXACTLY by `to_dtype` — which for `F32` is
/// `self.clone()` (candle `tensor.rs`'s same-dtype early return), making the
/// F32 mask byte-for-byte the tensor this function has always produced.
///
/// A CLIP-text row can never be fully masked (the causal diagonal is always
/// allowed), so the composed training softmax always sees a finite row
/// maximum and the masked entries decay to an exact zero with zero backward.
///
/// Integer dtypes are a typed refusal: an additive attention mask has no
/// meaning outside the floating dtypes, and silently reinterpreting the
/// sentinel as an integer would be exactly the confident-wrong-number
/// failure this gate exists to prevent.
fn build_causal_mask(
    size: usize,
    dtype: DType,
    device: &candle_core::Device,
) -> Result<Tensor, EncoderError> {
    let sentinel = dtype_min_sentinel(dtype)?;
    let mut data = vec![0f32; size * size];
    for row in 0..size {
        for col in (row + 1)..size {
            data[row * size + col] = sentinel;
        }
    }
    let mask = Tensor::from_vec(data, (size, size), device)?;
    Ok(mask.to_dtype(dtype)?)
}

/// `dtype`'s most-negative FINITE value, as an `f32` (exact for all three
/// supported floating dtypes — every `f16`/`bf16` value is exactly
/// representable in `f32`). See [`build_causal_mask`] for why this follows
/// the dtype and why a non-floating dtype is refused rather than coerced.
pub(crate) fn dtype_min_sentinel(dtype: DType) -> Result<f32, EncoderError> {
    Ok(match dtype {
        // F64 keeps `f32::MIN` rather than `f64::MIN`: the mask vector is
        // built in `f32` (so the F32 case stays byte-identical), and
        // `f32::MIN` ≈ -3.4e38 is already ~36 orders of magnitude below any
        // attention score this tower can produce. `f64::MIN` would not
        // survive that `f32` build at all.
        DType::F32 | DType::F64 => f32::MIN,
        DType::F16 => half::f16::MIN.to_f32(),
        DType::BF16 => half::bf16::MIN.to_f32(),
        other => {
            return Err(EncoderError::Config(format!(
                "additive attention mask: unsupported dtype {other:?} — a mask sentinel is \
                 only meaningful in a floating dtype (F32/F16/BF16)"
            )))
        }
    })
}

/// Gather one `[width]` row per batch from `hidden` (shape `[batch, seq, width]`)
/// at the per-batch positions in `indices` (shape `[batch]`).
fn gather_at_indices(hidden: &Tensor, indices: &Tensor) -> Result<Tensor, EncoderError> {
    let (batch, _seq, width) = hidden.dims3()?;
    // Expand indices [batch] -> [batch, 1, width] so the gather along dim=1
    // selects a full hidden row per batch.
    let idx = indices
        .unsqueeze(1)?
        .unsqueeze(2)?
        .broadcast_as((batch, 1, width))?
        .contiguous()?;
    let gathered = hidden.gather(&idx, 1)?;
    Ok(gathered.squeeze(1)?)
}

/// L2-normalize each row of a `[batch, dim]` tensor along the last axis.
fn l2_normalize(t: &Tensor) -> Result<Tensor, EncoderError> {
    let norm = t
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .clamp(1e-12, f32::MAX)?;
    Ok(t.broadcast_div(&norm)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        assert_finite_nonzero, deterministic_fill_varmap, find_var, nonuniform_loss,
    };
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    /// Replace every variable in `varmap` with a random tensor of the same
    /// shape so the encoder produces non-degenerate outputs in tests.
    fn randomize_varmap(varmap: &VarMap, device: &Device) {
        let data = varmap.data().lock().unwrap();
        for var in data.values() {
            let shape = var.shape().clone();
            let random = Tensor::randn(0f32, 0.1, shape, device).unwrap();
            var.set(&random).unwrap();
        }
    }

    /// Fixed (non-random) 2-sequence batch, EOT (highest ID) trailing padding
    /// on the second row, for the training/eval backward tests.
    fn fixed_batch(cfg: &ClipTextConfig, device: &Device) -> (Tensor, Tensor) {
        let ids: Vec<u32> = vec![
            1,
            2,
            3,
            4,
            (cfg.vocab_size - 1) as u32, // EOT at index 4
            5,
            6,
            (cfg.vocab_size - 1) as u32,
            0,
            0, // EOT at index 2, padded
        ];
        let input_ids = Tensor::from_vec(ids, (2, 5), device).unwrap();
        let mask = Tensor::ones((2, 5), DType::U32, device).unwrap();
        (input_ids, mask)
    }

    fn tiny_config() -> ClipTextConfig {
        ClipTextConfig {
            context_length: 16,
            vocab_size: 64,
            width: 16,
            layers: 2,
            heads: 2,
            embed_dim: 8,
        }
    }

    /// D6, F32 leg: the dtype-following mask is BYTE-IDENTICAL at F32 to the
    /// hardcoded `f32::MIN` upper triangle this function built before it
    /// took a dtype. This is the construction-level companion to
    /// `tests/bits_snapshot.rs`'s end-to-end K4 hash — it localises a
    /// regression to the mask rather than leaving it to a whole-tower digest.
    #[test]
    fn causal_mask_at_f32_is_byte_identical_to_the_hardcoded_f32_min_triangle() {
        let device = Device::Cpu;
        let size = 7;
        let mut expected = vec![0f32; size * size];
        for row in 0..size {
            for col in (row + 1)..size {
                expected[row * size + col] = f32::MIN;
            }
        }
        let mask = build_causal_mask(size, DType::F32, &device).unwrap();
        assert_eq!(mask.dtype(), DType::F32);
        let got: Vec<u32> = mask
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect();
        let want: Vec<u32> = expected.into_iter().map(f32::to_bits).collect();
        assert_eq!(got, want);
    }

    /// D6, reduced-precision legs: at F16 and BF16 the sentinel is that
    /// dtype's OWN most-negative finite value, and — the point of the whole
    /// exercise — it stays FINITE. Casting `f32::MIN` into either dtype
    /// would produce `-inf` instead, which an additive pre-softmax mask
    /// cannot carry (an `-inf` row max composes to `NaN`). BF16 is covered
    /// at CONSTRUCTION only: candle-core 0.11's CPU matmul refuses BF16, so
    /// there is no honest CPU forward to assert it through.
    #[test]
    fn causal_mask_at_f16_and_bf16_uses_that_dtype_s_own_finite_minimum() {
        let device = Device::Cpu;
        let size = 5;

        let f16_mask = build_causal_mask(size, DType::F16, &device).unwrap();
        assert_eq!(f16_mask.dtype(), DType::F16);
        let rows = f16_mask.to_vec2::<half::f16>().unwrap();
        assert_eq!(rows[0][1], half::f16::MIN, "masked entry must be f16::MIN");
        assert!(rows[0][1].is_finite(), "the sentinel must not be -inf");
        assert_eq!(rows[0][0], half::f16::ZERO, "allowed entry must be 0");
        assert_eq!(rows[4][0], half::f16::ZERO);

        let bf16_mask = build_causal_mask(size, DType::BF16, &device).unwrap();
        assert_eq!(bf16_mask.dtype(), DType::BF16);
        let rows = bf16_mask.to_vec2::<half::bf16>().unwrap();
        assert_eq!(
            rows[0][1],
            half::bf16::MIN,
            "masked entry must be bf16::MIN"
        );
        assert!(rows[0][1].is_finite(), "the sentinel must not be -inf");
        assert_eq!(rows[0][0], half::bf16::ZERO);

        // The regression this guards: the OLD f32 sentinel cast into either
        // dtype is -inf, not a large finite number.
        let overflowed = Tensor::from_vec(vec![f32::MIN], 1, &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
            .to_vec1::<half::f16>()
            .unwrap();
        assert!(
            overflowed[0].is_infinite(),
            "control: f32::MIN cast to f16 must indeed overflow to -inf, or this test is \
             guarding nothing"
        );
    }

    /// Family-D edge: a non-floating dtype has no meaningful additive-mask
    /// sentinel, so it is a typed refusal rather than a silently
    /// reinterpreted bit pattern.
    #[test]
    fn causal_mask_refuses_a_non_floating_dtype() {
        let device = Device::Cpu;
        let err = build_causal_mask(4, DType::U32, &device).unwrap_err();
        assert!(
            matches!(err, EncoderError::Config(ref m) if m.contains("U32")),
            "expected a typed Config refusal naming the dtype, got {err:?}"
        );
    }

    /// The frozen `load` path and the builder's `frozen()` path are ONE
    /// loader (`ClipText::load_with`), so their outputs agree bit-for-bit —
    /// A2-i at the unit level, on a `VarMap`-backed tower (the
    /// fixture-backed leg lives in `tests/tower_lora.rs`). Also pins that a
    /// frozen tower reports no trainable params.
    #[test]
    fn frozen_builder_config_installs_no_sites_and_leaves_training_off() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut model = ClipText::load(vb, &cfg).unwrap();
        deterministic_fill_varmap(&varmap, &device);

        assert!(model.trainable_params().is_empty());
        assert!(model.named_trainable_weights().unwrap().is_empty());
        assert!(model.dropout_positions().unwrap().is_empty());
        assert!(!model.is_training());
        model.set_training(true);
        assert!(model.is_training());
    }

    #[test]
    fn config_from_open_clip_json() {
        let json = serde_json::json!({
            "model_cfg": {
                "embed_dim": 512,
                "text_cfg": {
                    "context_length": 77,
                    "vocab_size": 49408,
                    "width": 512,
                    "heads": 8,
                    "layers": 12
                }
            }
        });
        let cfg = ClipTextConfig::from_open_clip_config(&json).unwrap();
        assert_eq!(cfg.embed_dim, 512);
        assert_eq!(cfg.width, 512);
        assert_eq!(cfg.heads, 8);
        assert_eq!(cfg.layers, 12);
        assert_eq!(cfg.context_length, 77);
        assert_eq!(cfg.vocab_size, 49408);
    }

    #[test]
    fn config_heads_default_from_width() {
        let json = serde_json::json!({
            "model_cfg": {
                "embed_dim": 512,
                "text_cfg": {
                    "width": 512,
                    "layers": 12
                }
            }
        });
        let cfg = ClipTextConfig::from_open_clip_config(&json).unwrap();
        assert_eq!(cfg.heads, 8); // 512 / 64
    }

    #[test]
    fn forward_output_shape_and_l2_norm() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let model = ClipText::load(vb, &cfg).unwrap();
        // VarMap creates zero-initialized vars by default; randomize so the
        // text_projection isn't all-zeros (which would yield an all-zero
        // pooled output and a degenerate L2 norm).
        randomize_varmap(&varmap, &device);

        // 3 token sequences, length 5 each, with EOT (highest ID) at the
        // last position. Pad-token-zero is allowed only after the EOT.
        let ids: Vec<u32> = vec![
            1, 2, 3, 4, 63, // EOT=63 (vocab_size-1) at index 4
            5, 6, 7, 63, 0, // EOT at index 3, padded
            8, 9, 63, 0, 0, // EOT at index 2, padded
        ];
        let input_ids = Tensor::from_vec(ids, (3, 5), &device).unwrap();
        let mask = Tensor::ones((3, 5), DType::U32, &device).unwrap();

        let out = model.forward(&input_ids, &mask).unwrap();
        assert_eq!(out.dims(), &[3, 8]); // (batch, embed_dim)

        let rows = out.to_vec2::<f32>().unwrap();
        for row in &rows {
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "L2 norm should be ~1.0, got {norm}"
            );
        }
    }

    #[test]
    fn sequence_too_long_rejected() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = ClipText::load(vb, &cfg).unwrap();

        let seq = cfg.context_length + 1;
        let ids: Vec<u32> = vec![1; seq];
        let input_ids = Tensor::from_vec(ids.clone(), (1, seq), &device).unwrap();
        let mask = Tensor::from_vec(ids, (1, seq), &device).unwrap();

        match model.forward(&input_ids, &mask) {
            Err(EncoderError::SequenceTooLong { seq: got, max }) => {
                assert_eq!(got, seq);
                assert_eq!(max, cfg.context_length);
            }
            other => panic!("expected SequenceTooLong, got {other:?}"),
        }
    }

    /// L2 norm of a `[rows, width]` sub-block of a `[3*width, width]`
    /// `in_proj_weight` gradient, addressing the Q/K/V slice by row range.
    fn slice_grad_norm(grad: &Tensor, row_start: usize, width: usize) -> f32 {
        let rows = grad.to_vec2::<f32>().unwrap();
        rows[row_start..row_start + width]
            .iter()
            .flatten()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
    }

    /// Run token embedding + positional embedding + ONLY block 0 (not the
    /// whole tower) and return `(in_proj_weight grad, token_embedding grad)`.
    ///
    /// This deliberately stops short of `ClipText::forward`'s `ln_final` /
    /// EOT-gather / `text_projection` tail. `candle_nn::LayerNorm`'s own
    /// fused kernel (`candle_nn::ops::layer_norm`, `apply_op3_no_bwd`) is a
    /// SEPARATE `BackpropOp::none()` truncation — the same family of defect
    /// as `softmax_last_dim` but affecting normalization, not attention, and
    /// independent of the softmax arm this helper isolates. Because it sits
    /// AFTER every block, routing
    /// the loss through it (i.e. through `ClipText::forward`'s public output)
    /// severs backward before it reaches ANY block parameter, independent of
    /// the softmax arm under test here — it would make both the training and
    /// eval oracles below vacuously pass (no gradient at all, in either
    /// mode). Stopping at block 0's own output keeps the softmax defect
    /// isolated and observable: block-internal residual connections
    /// (`shortcut + attn`, `hidden + mlp_out`) keep the gradient path to
    /// `token_embedding` and to the V-slice of `in_proj_weight` alive
    /// regardless of `ln_1`/`ln_2`'s OWN truncation, exactly as the
    /// `probs @ V` matmul keeps V alive despite `softmax_last_dim`'s.
    fn block0_backward(
        model: &ClipText,
        varmap: &VarMap,
        input_ids: &Tensor,
        device: &Device,
    ) -> (Option<Tensor>, Option<Tensor>) {
        let seq = input_ids.dim(1).unwrap();
        let token_emb = model.token_embedding.forward(input_ids).unwrap();
        let pos_emb = model.positional_embedding.i((..seq, ..)).unwrap();
        let x = token_emb.broadcast_add(&pos_emb).unwrap();
        let causal = model.causal_mask.i((..seq, ..seq)).unwrap();
        let block0_out = model.blocks[0].forward(&x, Some(&causal)).unwrap();

        let loss = nonuniform_loss(&block0_out, model.hidden_size(), device);
        let grads = loss.backward().unwrap();

        let in_proj_weight = find_var(varmap, "resblocks.0.attn.in_proj_weight");
        let token_embedding = find_var(varmap, "token_embedding.weight");
        (
            grads.get(in_proj_weight.as_tensor()).cloned(),
            grads.get(token_embedding.as_tensor()).cloned(),
        )
    }

    /// RED oracle: fails if `MultiHeadAttention::forward`'s training arm is
    /// reverted to `softmax_last_dim` (or if `ClipText::set_training` stops
    /// propagating down to the attention module) — under either regression
    /// the Q/K slices of `in_proj_weight` come back exactly zero here, same
    /// as the companion eval-mode test below. Measured on this fixture: Q/K
    /// slice norms are small but strictly positive (softmax's Jacobian damps
    /// them relative to V's direct linear pass-through) — e.g. ~5e-4 / ~1.6e-3
    /// against V's ~7.5, never exactly `0.0` the way eval's are.
    #[test]
    fn training_true_backward_gives_nonzero_grad_to_q_k_and_v_slices() {
        // `block0_backward` forwards through `ln_1`/`ln_2` (biased, training
        // mode) — a counter bumper on `crate::layer_norm::LN_DISPATCH_COUNTERS`
        // even though this test never reads that counter itself. Same lock
        // discipline as `clip_text_training_ln_dispatch_is_now_counted` below
        // (see `crate::layer_norm::DISPATCH_COUNTER_TEST_LOCK`'s doc).
        let _guard = crate::layer_norm::DISPATCH_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let cfg = tiny_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut model = ClipText::load(vb, &cfg).unwrap();
        deterministic_fill_varmap(&varmap, &device);
        model.set_training(true);

        let (input_ids, _mask) = fixed_batch(&cfg, &device);
        let (in_proj_grad, token_embedding_grad) =
            block0_backward(&model, &varmap, &input_ids, &device);

        let grad = in_proj_grad.expect("in_proj_weight must have a gradient under training=true");
        let width = cfg.width;
        let q_norm = slice_grad_norm(&grad, 0, width);
        let k_norm = slice_grad_norm(&grad, width, width);
        let v_norm = slice_grad_norm(&grad, 2 * width, width);

        assert_finite_nonzero(q_norm, "Q slice");
        assert_finite_nonzero(k_norm, "K slice");
        assert_finite_nonzero(v_norm, "V slice (positive control)");
        assert!(
            token_embedding_grad.is_some(),
            "token embedding grad must be Some under training=true"
        );
    }

    /// Documents the defect shape on the SAME fixture as
    /// [`training_true_backward_gives_nonzero_grad_to_q_k_and_v_slices`]:
    /// eval's `softmax_last_dim` (`BackpropOp::none()`) truncates backward
    /// before it ever reaches Q/K, but V still receives a gradient through
    /// the untouched `probs @ V` matmul — a silently WRONG (partially zero),
    /// not erroring, gradient. This test is independent of the training-arm
    /// fix (eval always uses `softmax_last_dim`), so it stays green even
    /// under the fix-verifier's revert; paired with the test above it also
    /// catches a dropped `set_training` propagation line (that regression
    /// would flip the OTHER test red instead, since eval's own arm never
    /// changes). Measured on this fixture: Q/K slice norms are exactly
    /// `0.0`; V's is ~7.5 (unchanged from the training=true measurement,
    /// since V's path never crosses the truncation either way).
    #[test]
    fn training_false_q_and_k_grad_are_exactly_zero_v_nonzero() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = ClipText::load(vb, &cfg).unwrap();
        deterministic_fill_varmap(&varmap, &device);
        // training defaults to false; forward without calling set_training.

        let (input_ids, _mask) = fixed_batch(&cfg, &device);
        let (in_proj_grad, token_embedding_grad) =
            block0_backward(&model, &varmap, &input_ids, &device);

        let grad = in_proj_grad.expect("in_proj_weight still gets a (partial) gradient under eval");
        let width = cfg.width;
        let q_norm = slice_grad_norm(&grad, 0, width);
        let k_norm = slice_grad_norm(&grad, width, width);
        let v_norm = slice_grad_norm(&grad, 2 * width, width);

        assert_eq!(q_norm, 0.0, "Q slice grad must be exactly zero under eval");
        assert_eq!(k_norm, 0.0, "K slice grad must be exactly zero under eval");
        assert_finite_nonzero(v_norm, "V slice (positive control, eval)");
        assert!(
            token_embedding_grad.is_some(),
            "token embedding grad is still reachable via the block's residual stream under eval \
             (only Q/K are severed by softmax_last_dim, not the whole graph)"
        );
    }

    /// End-to-end RED oracle through the FULL public `forward` (not the
    /// block-level bypass above): with BOTH the attention-softmax arm and
    /// every `LayerNorm` (`ln_1`/`ln_2` per block, `ln_final`) gated on
    /// `training`, backward through `model.forward(...)`'s pooled,
    /// L2-normalized output reaches every parameter used anywhere in the
    /// tower. Fails if EITHER gate regresses: reverting only the softmax arm
    /// leaves Q/K severed (same failure this test would show even with
    /// `ln_final` fixed); reverting only `ln_final`'s gate severs backward
    /// entirely before it reaches ANY block (the token embedding assertion
    /// would fail first, since `ln_final` sits strictly downstream of every
    /// block). This is the shape a real training loop would have hit: no
    /// error, just gradients that silently never update Q/K or the earlier
    /// layers.
    #[test]
    fn training_true_full_forward_reaches_every_parameter() {
        // Full-tower forward through `ln_1`/`ln_2`/`ln_final` at
        // `training=true` bumps `crate::layer_norm::LN_DISPATCH_COUNTERS`
        // even though this test never reads it — same lock discipline as
        // `training_true_backward_gives_nonzero_grad_to_q_k_and_v_slices`
        // above.
        let _guard = crate::layer_norm::DISPATCH_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let cfg = tiny_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut model = ClipText::load(vb, &cfg).unwrap();
        deterministic_fill_varmap(&varmap, &device);
        model.set_training(true);

        let (input_ids, mask) = fixed_batch(&cfg, &device);
        let out = model.forward(&input_ids, &mask).unwrap();
        let loss = nonuniform_loss(&out, cfg.embed_dim, &device);
        let grads = loss.backward().unwrap();

        let in_proj_weight = find_var(&varmap, "resblocks.0.attn.in_proj_weight");
        let grad = grads
            .get(in_proj_weight.as_tensor())
            .expect("layer-0 in_proj_weight must have a gradient under training=true");
        let width = cfg.width;
        let q_norm = slice_grad_norm(grad, 0, width);
        let k_norm = slice_grad_norm(grad, width, width);
        let v_norm = slice_grad_norm(grad, 2 * width, width);
        assert_finite_nonzero(q_norm, "Q slice");
        assert_finite_nonzero(k_norm, "K slice");
        assert_finite_nonzero(v_norm, "V slice");

        // BLANKET oracle: every Var in the VarMap — not just the
        // hand-picked token_embedding/in_proj_weight pair above — must
        // receive a Some/finite/nonzero gradient. Measured 29/29 on this
        // fixture: ClipText has no non-differentiable buffer to exclude.
        crate::test_support::assert_every_var_has_gradient(&varmap, &grads, &[]);
    }

    /// The eval-mode observable a user of this tower would actually hit
    /// before either gate existed: `model.forward(...)`'s backward yields NO
    /// gradient entry AT ALL (`grads.get(...).is_none()`, not a partial or
    /// zero one) for the token embedding or `in_proj_weight`, because
    /// `ln_final`'s own `BackpropOp::none()` truncates backward before it
    /// reaches ANY block, independent of the softmax arm (which is a
    /// SEPARATE, strictly-worse truncation one hop earlier). This documents
    /// the pre-fix full-tower failure mode: not "Q/K come back zero" (that's
    /// only visible below `ln_final`, per the block-level tests above) but
    /// "nothing upstream of `ln_final` gets a gradient at all."
    #[test]
    fn training_false_full_forward_grads_are_none_before_ln_final() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = ClipText::load(vb, &cfg).unwrap();
        deterministic_fill_varmap(&varmap, &device);
        // training defaults to false; forward without calling set_training.

        let (input_ids, mask) = fixed_batch(&cfg, &device);
        let out = model.forward(&input_ids, &mask).unwrap();
        let loss = nonuniform_loss(&out, cfg.embed_dim, &device);
        let grads = loss.backward().unwrap();

        let token_embedding = find_var(&varmap, "token_embedding.weight");
        assert!(
            grads.get(token_embedding.as_tensor()).is_none(),
            "token embedding grad must be None under eval through the full forward \
             (ln_final truncates backward before it reaches any block)"
        );
        let in_proj_weight = find_var(&varmap, "resblocks.0.attn.in_proj_weight");
        assert!(
            grads.get(in_proj_weight.as_tensor()).is_none(),
            "in_proj_weight grad must be None under eval through the full forward, not merely \
             zero in its Q/K rows — ln_final severs the V-slice's surviving path too"
        );

        // BLANKET oracle: every trainable Var in the VarMap is severed, not
        // just the two spot-checked above. EXCLUDED: `text_projection`
        // (`crate::contiguous_matmul(&pooled, &self.text_projection)` in
        // `ClipText::forward` — see that method) — it sits DOWNSTREAM of
        // `ln_final`'s truncation, applied by a plain differentiable matmul
        // directly to `ln_final`'s output, so it still receives its own
        // gradient (matmul backward for one operand only needs the OTHER
        // operand's forward value, not a walk back through it) even though
        // everything upstream of `ln_final` is severed.
        crate::test_support::assert_every_var_grad_is_none(&varmap, &grads, &["text_projection"]);
    }

    /// Deletion-catching oracle for the residual-stream LayerNorms
    /// themselves (`ln_1`/`ln_2`, block 0): every OTHER gradient assertion
    /// in this file reaches its target parameter THROUGH the block's
    /// residual bypass (`shortcut + attn`, `hidden + mlp_out` —
    /// [`block0_backward`]'s doc), so a dropped `self.ln_1.set_training(training)`
    /// / `self.ln_2.set_training(training)` line (leaving that ONE
    /// LayerNorm stuck on its fused, `BackpropOp::none()`-truncated eval
    /// arm even when the rest of the tower is `training=true`) would NOT
    /// be caught by any test above: the residual path still carries a
    /// gradient to `in_proj_weight`/`token_embedding` regardless of `ln_1`/
    /// `ln_2`'s own truncation. This test asserts `ln_1`/`ln_2`'s OWN
    /// `weight` — not anything upstream of it — through the full public
    /// `forward`: `Some`/finite/nonzero under `training=true`, `None` under
    /// `training=false` (`(Some(bias), false)`'s fused arm is
    /// `BackpropOp::none()` on ALL three of its operands, including
    /// `weight` itself — see `crate::layer_norm::LayerNorm::forward`).
    /// RED-verified: deleting `self.ln_1.set_training(training)` from
    /// `ResidualAttentionBlock::set_training` flips the training=true half
    /// of this test (ln_1.weight comes back `None` instead of `Some`)
    /// while every other test in this file stays green.
    #[test]
    fn ln_1_and_ln_2_own_weight_gradient_present_under_training_absent_under_eval() {
        let cfg = tiny_config();
        let device = Device::Cpu;

        let training_grad = |name: &str| -> Option<Tensor> {
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            let mut model = ClipText::load(vb, &cfg).unwrap();
            deterministic_fill_varmap(&varmap, &device);
            model.set_training(true);

            // Full-tower forward at `training=true` bumps
            // `crate::layer_norm::LN_DISPATCH_COUNTERS` even though this
            // closure never reads it — same lock discipline as the other
            // training-forward tests in this module.
            let _guard = crate::layer_norm::DISPATCH_COUNTER_TEST_LOCK
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let (input_ids, mask) = fixed_batch(&cfg, &device);
            let out = model.forward(&input_ids, &mask).unwrap();
            let loss = nonuniform_loss(&out, cfg.embed_dim, &device);
            let grads = loss.backward().unwrap();
            let var = find_var(&varmap, name);
            grads.get(var.as_tensor()).cloned()
        };
        let eval_grad = |name: &str| -> Option<Tensor> {
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            let model = ClipText::load(vb, &cfg).unwrap();
            deterministic_fill_varmap(&varmap, &device);
            // training defaults to false; forward without calling set_training.

            let (input_ids, mask) = fixed_batch(&cfg, &device);
            let out = model.forward(&input_ids, &mask).unwrap();
            let loss = nonuniform_loss(&out, cfg.embed_dim, &device);
            let grads = loss.backward().unwrap();
            let var = find_var(&varmap, name);
            grads.get(var.as_tensor()).cloned()
        };

        for name in ["resblocks.0.ln_1.weight", "resblocks.0.ln_2.weight"] {
            let grad = training_grad(name)
                .unwrap_or_else(|| panic!("{name} grad must be Some under training=true"));
            let norm = grad
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                .sqrt();
            assert_finite_nonzero(norm, &format!("{name} (training=true)"));

            assert!(
                eval_grad(name).is_none(),
                "{name} grad must be None under training=false (eval's fused LayerNorm arm is \
                 BackpropOp::none() on every operand, including its own weight)"
            );
        }
    }

    /// #460 (C-LN): CLIP-text's LayerNorms (`ln_1`/`ln_2` per block,
    /// `ln_final`) are ALL biased (`LayerNorm::new(.., with_bias: true,
    /// ..)`) and reachable at `training == true` through
    /// `AnyEncoder::set_training` (see `any::tests::
    /// any_encoder_set_training_reaches_clip_text_q_k_gradient`, which
    /// proves the SAME `set_training` wiring for its attention gradient).
    /// Before this unit, every one of them fell through `slow()` with no
    /// `admit()` call at all — the same `ln` `0/0` gap BERT/DistilBERT
    /// had. This is the one NEW assertion #460 adds here: the training
    /// dispatch is now COUNTED, on top of the gradient oracles above
    /// (which stay green, unmodified, proving this doesn't change their
    /// correctness claim).
    #[test]
    fn clip_text_training_ln_dispatch_is_now_counted() {
        let _guard = crate::layer_norm::DISPATCH_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let cfg = tiny_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut model = ClipText::load(vb, &cfg).unwrap();
        deterministic_fill_varmap(&varmap, &device);
        model.set_training(true);

        let (input_ids, mask) = fixed_batch(&cfg, &device);
        let before = crate::ln_dispatch_snapshot();
        let _ = model.forward(&input_ids, &mask).unwrap();
        let after = crate::ln_dispatch_snapshot();
        assert!(
            after.fused > before.fused && after.eager == before.eager,
            "CLIP-text's biased LayerNorms must dispatch the fused `layer_norm_fused` key \
             at training=true, not fall through slow() uncounted, and must never fall back \
             to eager (before={before:?}, after={after:?})"
        );
    }

    /// Eval output is unaffected by ever having toggled training on and back
    /// off: the two arms are only wired into the *backward* path, so the
    /// eval-mode forward is byte-identical before any `set_training` call and
    /// after `set_training(true); set_training(false)`. Masks in this tower
    /// are hardcoded F32 (`build_causal_mask`), so this oracle stays f32-only
    /// — no bf16 variant is meaningful here.
    #[test]
    fn eval_output_is_bit_identical_across_a_training_toggle_round_trip() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut model = ClipText::load(vb, &cfg).unwrap();
        deterministic_fill_varmap(&varmap, &device);

        let (input_ids, mask) = fixed_batch(&cfg, &device);

        let before = model.forward(&input_ids, &mask).unwrap();
        model.set_training(true);
        model.set_training(false);
        let after = model.forward(&input_ids, &mask).unwrap();

        assert_eq!(
            before.to_vec2::<f32>().unwrap(),
            after.to_vec2::<f32>().unwrap(),
            "eval output must be bit-identical across a training toggle round trip"
        );
    }
}
