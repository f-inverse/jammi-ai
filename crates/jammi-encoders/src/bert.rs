//! BERT-family encoder (BERT, RoBERTa, CamemBERT, XLM-RoBERTa).
//!
//! Mirrors candle-transformers' `BertModel` architecture so the parity test
//! can numerically verify the frozen forward pass against the upstream
//! implementation. The six attention/FFN linears per layer (`query`, `key`,
//! `value`, `attention.output.dense`, `intermediate.dense`, `output.dense`)
//! are wrapped in [`jammi_lora::MaybeLoraLinear`] so a builder-time
//! [`jammi_lora::LoraBuildConfig`] selects which of them carry trainable
//! adapters.
//!
//! Two safetensors layouts are supported transparently:
//!   * Raw `BertModel` checkpoints — keys at the root
//!     (`embeddings.word_embeddings.weight`).
//!   * `BertForX` checkpoints — keys under a `"bert."` prefix.
//!
//! Detection is via a single `contains_tensor` probe at build time.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{embedding, linear, Embedding, Linear, Module, VarBuilder, VarMap};
use jammi_kernels::admission::PredicateOutcome;
use jammi_kernels::ops::FullyMaskedPolicy;
use jammi_lora::{
    effective_rank, should_apply_lora, FrozenBase, LoraBuildConfig, LoraLinear, MaybeLoraLinear,
};

use crate::activations;
use crate::attention_cascade::{
    self, FlashDecision, FusedAttentionMasks, RopeCtx, TrainingMaskInputs,
};
use crate::error::EncoderError;
use crate::frozen_weight_source::{validate_frozen_base_geometry, FrozenWeightLookup};
use crate::layer_norm::LayerNorm;
use crate::mask::extended_attention_mask;
use crate::pooling::{pool_and_normalize, Pooling};

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Architecture configuration deserialised from a HuggingFace `config.json`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct BertConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_type_vocab_size")]
    pub type_vocab_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    /// Detected from the `model_type` field. Determines the safetensors-key
    /// prefix (`"bert."`, `"roberta."`, etc., or `""` for raw `BertModel`).
    #[serde(default)]
    pub model_type: Option<String>,
}

fn default_type_vocab_size() -> usize {
    2
}
fn default_layer_norm_eps() -> f64 {
    1e-12
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-layer sub-structures (mirroring candle-transformers' BertModel)
// ─────────────────────────────────────────────────────────────────────────────

struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
}

impl BertEmbeddings {
    fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self, EncoderError> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        let position_embeddings = embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("position_embeddings"),
        )?;
        let token_type_embeddings = embedding(
            config.type_vocab_size,
            config.hidden_size,
            vb.pp("token_type_embeddings"),
        )?;
        let layer_norm = LayerNorm::new(
            config.hidden_size,
            config.layer_norm_eps,
            true,
            vb.pp("LayerNorm"),
        )?;
        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor, EncoderError> {
        let (_batch, seq) = input_ids.dims2()?;
        let word_emb = self.word_embeddings.forward(input_ids)?;
        let token_type_emb = self.token_type_embeddings.forward(token_type_ids)?;
        let embeddings = (&word_emb + token_type_emb)?;
        let position_ids = Tensor::arange(0u32, seq as u32, input_ids.device())?;
        let position_emb = self.position_embeddings.forward(&position_ids)?;
        let embeddings = embeddings.broadcast_add(&position_emb)?;
        self.layer_norm.forward(&embeddings)
    }
}

struct BertSelfAttention {
    query: MaybeLoraLinear,
    key: MaybeLoraLinear,
    value: MaybeLoraLinear,
    num_attention_heads: usize,
    attention_head_size: usize,
    /// A `[2, 1, 1, 64]` placeholder handed to the shared attention
    /// cascade's `rope_pack` argument — BERT has no RoPE at all (absolute
    /// position embeddings are summed into the input once, in
    /// [`BertEmbeddings::forward`], never applied per layer), so this
    /// tensor is never read (see [`RopeCtx`]'s own doc: the fused ops only
    /// consult `rope_pack` when `rope == true`, which BERT's cascade call
    /// never sets). Allocated ONCE, at build time
    /// (`tests::rope_placeholder_is_allocated_once_per_module_and_never_reallocated`),
    /// not per forward.
    rope_placeholder: Tensor,
}

impl BertSelfAttention {
    /// Reshape `[B, S, h*d]` into `[B, h, S, d]` and make the result contiguous
    /// — the `.contiguous()` here is the no-band-aid fix for the matmul
    /// contiguity panic on transposed inputs (candle issue #1965 / PR #3088).
    fn transpose_for_scores(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        let mut new_shape = x.dims().to_vec();
        new_shape.pop();
        new_shape.push(self.num_attention_heads);
        new_shape.push(self.attention_head_size);
        let x = x.reshape(new_shape.as_slice())?.transpose(1, 2)?;
        Ok(x.contiguous()?)
    }

    fn forward(&self, hidden: &Tensor, extended_mask: &Tensor) -> Result<Tensor, EncoderError> {
        let q = self.query.forward(hidden)?;
        let k = self.key.forward(hidden)?;
        let v = self.value.forward(hidden)?;
        let q = self.transpose_for_scores(&q)?;
        let k = self.transpose_for_scores(&k)?;
        let v = self.transpose_for_scores(&v)?;

        let scores = crate::contiguous_matmul(&q, &k.t()?)?;
        let scores = (scores / (self.attention_head_size as f64).sqrt())?;
        // The additive mask is always built in F32 (see `extended_attention_mask`);
        // cast to the scores' dtype so a F16/BF16 backbone can add it (a no-op
        // when scores are already F32).
        let extended_mask = extended_mask.to_dtype(scores.dtype())?;
        let scores = scores.broadcast_add(&extended_mask)?;
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;

        let context = crate::contiguous_matmul(&probs, &v)?;
        let context = context.transpose(1, 2)?.contiguous()?;
        Ok(context.flatten_from(D::Minus2)?)
    }

    /// Training's arm: routes through the shared
    /// [`attention_cascade::training_attention_cascade`] (issue #462) —
    /// `qkv = Tensor::cat(&[q, k, v], D::Minus1)` (the cat bridge, `[B, S,
    /// 3*hidden]`; see this module's doc for the forward-copy/backward-zero-fill
    /// cost this bridge carries on BOTH the fused and eager arms of the
    /// cascade), `rope` disabled (BERT has no RoPE), `window: None` (BERT
    /// has no sliding-window concept — every layer is "global"), `policy:
    /// Propagate` (reproduces `candle_nn::ops::softmax`'s own behaviour on
    /// an all-padding row exactly, unlike ModernBERT's `Zeros` — see this
    /// module's doc), and `flash` always `Declined { CapabilityMiss,
    /// "flash_transport_not_wired" }` — BERT never wires the encoder-boundary
    /// flash transport protocol (a separate line of work; the flash cascade
    /// counter still fires, declined, on every training forward, never
    /// silently).
    fn forward_training(
        &self,
        hidden: &Tensor,
        extended_mask: &Tensor,
        fused: &FusedAttentionMasks,
        flash: &FlashDecision,
    ) -> Result<Tensor, EncoderError> {
        // No `self.training` field to assert against (audit round item 6:
        // a duplicated per-sub-struct training copy was deleted here) —
        // this method is private and has exactly one call site
        // (`BertAttention::forward_training`, itself only reachable from
        // `Bert::forward_hidden`'s `self.training` branch), so a desync
        // between "this method runs" and "training is true" is
        // unrepresentable by construction rather than checked at runtime.
        let q = self.query.forward(hidden)?;
        let k = self.key.forward(hidden)?;
        let v = self.value.forward(hidden)?;
        let qkv = Tensor::cat(&[&q, &k, &v], D::Minus1)?;
        let (batch, seq, _) = hidden.dims3()?;
        let h = self.num_attention_heads;
        let d = self.attention_head_size;
        let masks = TrainingMaskInputs {
            extended: extended_mask,
            local_band: None,
            fused: Some(fused),
        };
        let rope = RopeCtx::Disabled {
            placeholder: &self.rope_placeholder,
        };
        attention_cascade::training_attention_cascade(
            &qkv,
            batch,
            seq,
            h,
            d,
            masks,
            flash,
            &rope,
            None, // window: BERT has no sliding-window concept.
            None, // half_window: same reason — no scalar to pass memeff.
            FullyMaskedPolicy::Propagate,
            |_admission| {
                Err(EncoderError::Config(
                    "attention_block_flash dispatched Fused but BERT always supplies \
                     FlashDecision::Declined -- unreachable: admit_cascade can only report \
                     Fused when flash.outcome() is Holds, which Declined can never be"
                        .into(),
                ))
            },
        )
    }
}

struct BertSelfOutput {
    dense: MaybeLoraLinear,
    layer_norm: LayerNorm,
}

impl BertSelfOutput {
    fn forward(&self, hidden: &Tensor, input_tensor: &Tensor) -> Result<Tensor, EncoderError> {
        let hidden = self.dense.forward(hidden)?;
        self.layer_norm.forward(&(hidden + input_tensor)?)
    }
}

struct BertAttention {
    self_attention: BertSelfAttention,
    self_output: BertSelfOutput,
}

impl BertAttention {
    fn forward(&self, hidden: &Tensor, extended_mask: &Tensor) -> Result<Tensor, EncoderError> {
        let self_outputs = self.self_attention.forward(hidden, extended_mask)?;
        self.self_output.forward(&self_outputs, hidden)
    }

    fn forward_training(
        &self,
        hidden: &Tensor,
        extended_mask: &Tensor,
        fused: &FusedAttentionMasks,
        flash: &FlashDecision,
    ) -> Result<Tensor, EncoderError> {
        let self_outputs =
            self.self_attention
                .forward_training(hidden, extended_mask, fused, flash)?;
        self.self_output.forward(&self_outputs, hidden)
    }
}

struct BertIntermediate {
    dense: MaybeLoraLinear,
}

impl BertIntermediate {
    /// `training` is a PARAMETER, not a stored copy (audit round item 6):
    /// [`Bert::training`] is the single source of truth, threaded down
    /// through [`BertLayer::forward`]/[`BertLayer::forward_training`] to
    /// this call — a desync between what `Bert::forward_hidden` decided
    /// and what this method's `activations::gelu_erf` call receives is
    /// unrepresentable, since there is no second copy left to drift.
    /// `false` (eval) makes the `activations::gelu_erf` call byte-for-byte
    /// identical to the plain `hidden.gelu_erf()` this method called before
    /// the GELU seam existed — see `activations::gelu_erf`'s own doc.
    fn forward(&self, hidden: &Tensor, training: bool) -> Result<Tensor, EncoderError> {
        let hidden = self.dense.forward(hidden)?;
        activations::gelu_erf(&hidden, training)
    }
}

struct BertOutput {
    dense: MaybeLoraLinear,
    layer_norm: LayerNorm,
}

impl BertOutput {
    fn forward(&self, hidden: &Tensor, input_tensor: &Tensor) -> Result<Tensor, EncoderError> {
        let hidden = self.dense.forward(hidden)?;
        self.layer_norm.forward(&(hidden + input_tensor)?)
    }
}

struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    fn forward(&self, hidden: &Tensor, extended_mask: &Tensor) -> Result<Tensor, EncoderError> {
        let attention_output = self.attention.forward(hidden, extended_mask)?;
        let intermediate_output = self.intermediate.forward(&attention_output, false)?;
        self.output.forward(&intermediate_output, &attention_output)
    }

    /// `training` is threaded down to [`BertIntermediate::forward`] as a
    /// parameter (audit round item 6) — [`Bert::forward_hidden`] passes
    /// its own `self.training` here, the single source, rather than each
    /// sub-struct carrying an independently-set copy that could drift.
    fn forward_training(
        &self,
        hidden: &Tensor,
        extended_mask: &Tensor,
        fused: &FusedAttentionMasks,
        flash: &FlashDecision,
        training: bool,
    ) -> Result<Tensor, EncoderError> {
        let attention_output =
            self.attention
                .forward_training(hidden, extended_mask, fused, flash)?;
        let intermediate_output = self.intermediate.forward(&attention_output, training)?;
        self.output.forward(&intermediate_output, &attention_output)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main encoder
// ─────────────────────────────────────────────────────────────────────────────

/// BERT / RoBERTa / CamemBERT / XLM-RoBERTa encoder with optional LoRA adapters
/// on the six per-layer linears (`query`, `key`, `value`,
/// `attention.output.dense`, `intermediate.dense`, `output.dense`).
pub struct Bert {
    embeddings: BertEmbeddings,
    layers: Vec<BertLayer>,
    pooling: Pooling,
    hidden_size: usize,
    max_position_embeddings: usize,
    /// Wired through [`Self::set_training`]. `forward_hidden` reads this
    /// ONCE per forward, before the layer loop, to decide which of the two
    /// call chains to take and — in training — to build the per-forward
    /// mask bundle and flash-cascade decision once, mirroring
    /// `ModernBert::forward_hidden`'s own once-per-forward construction
    /// (see [`FusedAttentionMasks`]'s doc for why per-layer would be
    /// wasteful).
    training: bool,
}

impl Bert {
    /// Start a builder with default settings: mean pooling, frozen LoRA config,
    /// F32 backbone dtype, no adapter file.
    pub fn builder() -> BertBuilder<'static> {
        BertBuilder {
            pooling: Pooling::default(),
            lora: LoraBuildConfig::frozen(),
            backbone_dtype: DType::F32,
            adapter_file: None,
            weight_source: None,
        }
    }

    /// Hidden dimensionality of the model (and of the pooled embeddings).
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Maximum input sequence length (`max_position_embeddings`).
    pub fn max_seq_length(&self) -> usize {
        self.max_position_embeddings
    }

    /// Dtype the FROZEN BACKBONE weights are materialised at — read off a
    /// real weight (the word-embedding table), never a remembered builder
    /// setting, so it stays true for a model built through `load` from an
    /// arbitrary `VarBuilder` as well as one built through the builder's
    /// `backbone_dtype`. See `crate::AnyEncoder::dtype` for the one caller
    /// class this exists for.
    pub fn dtype(&self) -> candle_core::DType {
        self.embeddings.word_embeddings.embeddings().dtype()
    }

    /// Raw `[batch, seq, hidden]` output before pooling. Eval (the default,
    /// `self.training == false`) is BYTE-FOR-BYTE UNCHANGED — the very same
    /// `for layer in &self.layers { hidden = layer.forward(..) }` loop this
    /// method always ran. Training builds the per-forward mask bundle and
    /// flash-cascade decision ONCE (mirroring `ModernBert::forward_hidden`)
    /// and routes every layer through `BertLayer::forward_training`/
    /// `attention_cascade::training_attention_cascade` instead.
    pub fn forward_hidden(
        &self,
        input_ids: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor, EncoderError> {
        let (_batch, seq) = input_ids.dims2()?;
        if seq > self.max_position_embeddings {
            return Err(EncoderError::SequenceTooLong {
                seq,
                max: self.max_position_embeddings,
            });
        }
        let token_type_ids = Tensor::zeros(input_ids.shape(), DType::U32, input_ids.device())?;
        let mut hidden = self.embeddings.forward(input_ids, &token_type_ids)?;
        let extended = extended_attention_mask(mask)?;
        if self.training {
            // BERT has no local-attention layers at all, so the fused
            // masks bundle only ever populates `global` — see
            // `FusedAttentionMasks`'s own doc.
            let fused = FusedAttentionMasks::build(&extended, None, hidden.dtype())?;
            // BERT never wires the encoder-boundary flash transport
            // protocol (a separate line of work — see
            // `BertSelfAttention::forward_training`'s doc): every training
            // forward reports `attention_block_flash` declined, counted,
            // never silent.
            let flash = FlashDecision::Declined {
                outcome: PredicateOutcome::CapabilityMiss,
                reason: "flash_transport_not_wired",
            };
            for layer in &self.layers {
                hidden =
                    layer.forward_training(&hidden, &extended, &fused, &flash, self.training)?;
            }
        } else {
            for layer in &self.layers {
                hidden = layer.forward(&hidden, &extended)?;
            }
        }
        Ok(hidden)
    }

    /// Pooled-and-L2-normalised `[batch, hidden]` sentence embedding.
    pub fn forward(&self, input_ids: &Tensor, mask: &Tensor) -> Result<Tensor, EncoderError> {
        let hidden = self.forward_hidden(input_ids, mask)?;
        pool_and_normalize(&hidden, mask, self.pooling)
    }

    /// Trainable tensors across every LoRA-wrapped linear. Empty for a fully
    /// frozen encoder.
    pub fn trainable_params(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.attention.self_attention.query.trainable_params());
            params.extend(layer.attention.self_attention.key.trainable_params());
            params.extend(layer.attention.self_attention.value.trainable_params());
            params.extend(layer.attention.self_output.dense.trainable_params());
            params.extend(layer.intermediate.dense.trainable_params());
            params.extend(layer.output.dense.trainable_params());
        }
        params
    }

    /// Named LoRA A/B tensors keyed as `layer.{n}.{module}.lora_a` /
    /// `layer.{n}.{module}.lora_b`, ready for safetensors serialisation.
    pub fn named_trainable_weights(&self) -> Result<HashMap<String, Tensor>, EncoderError> {
        let mut out = HashMap::new();
        for (n, layer) in self.layers.iter().enumerate() {
            out.extend(
                layer
                    .attention
                    .self_attention
                    .query
                    .named_weights(&format!("layer.{n}.query"))?,
            );
            out.extend(
                layer
                    .attention
                    .self_attention
                    .key
                    .named_weights(&format!("layer.{n}.key"))?,
            );
            out.extend(
                layer
                    .attention
                    .self_attention
                    .value
                    .named_weights(&format!("layer.{n}.value"))?,
            );
            out.extend(
                layer
                    .attention
                    .self_output
                    .dense
                    .named_weights(&format!("layer.{n}.dense"))?,
            );
            out.extend(
                layer
                    .intermediate
                    .dense
                    .named_weights(&format!("layer.{n}.intermediate_dense"))?,
            );
            out.extend(
                layer
                    .output
                    .dense
                    .named_weights(&format!("layer.{n}.output_dense"))?,
            );
        }
        Ok(out)
    }

    /// Switch every LoRA-wrapped linear and LayerNorm into / out of training
    /// mode, and (issue #462) `self` itself — the ONE flag
    /// [`Self::forward_hidden`] reads to pick its call chain and thread
    /// `training` down to `BertIntermediate::forward` as a parameter
    /// (audit round item 6: no sub-struct carries its own copy of this
    /// flag any more, so there is nothing left to fall out of step with
    /// `self.training`). LoRA layers gate dropout; LayerNorms switch
    /// between the fused no-bwd eval kernel and the primitive-op
    /// composition whose backward is well-defined.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        self.embeddings.layer_norm.set_training(training);
        for layer in &mut self.layers {
            layer.attention.self_attention.query.set_training(training);
            layer.attention.self_attention.key.set_training(training);
            layer.attention.self_attention.value.set_training(training);
            layer.attention.self_output.dense.set_training(training);
            layer
                .attention
                .self_output
                .layer_norm
                .set_training(training);
            layer.intermediate.dense.set_training(training);
            layer.output.dense.set_training(training);
            layer.output.layer_norm.set_training(training);
        }
    }

    /// Restore LoRA A/B tensors from a `named_trainable_weights`-shaped map.
    pub fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<(), EncoderError> {
        for (n, layer) in self.layers.iter_mut().enumerate() {
            layer
                .attention
                .self_attention
                .query
                .load_weights(weights, &format!("layer.{n}.query"));
            layer
                .attention
                .self_attention
                .key
                .load_weights(weights, &format!("layer.{n}.key"));
            layer
                .attention
                .self_attention
                .value
                .load_weights(weights, &format!("layer.{n}.value"));
            layer
                .attention
                .self_output
                .dense
                .load_weights(weights, &format!("layer.{n}.dense"));
            layer
                .intermediate
                .dense
                .load_weights(weights, &format!("layer.{n}.intermediate_dense"));
            layer
                .output
                .dense
                .load_weights(weights, &format!("layer.{n}.output_dense"));
        }
        Ok(())
    }

    /// Per-site dropout-stream positions keyed `{site}.dropout`, over the same
    /// site names [`Self::named_trainable_weights`] uses — the resume state for
    /// the adapter's dropout so a resumed run replays each stream to its
    /// epoch-boundary position.
    pub fn dropout_positions(&self) -> Result<HashMap<String, u64>, EncoderError> {
        let mut out = HashMap::new();
        for (n, layer) in self.layers.iter().enumerate() {
            for (site, lin) in lora_sites(layer) {
                lin.collect_dropout_position(&format!("layer.{n}.{site}"), &mut out)?;
            }
        }
        Ok(out)
    }

    /// Restore each LoRA site's dropout-stream position from a
    /// [`Self::dropout_positions`]-shaped map. Missing keys are no-ops.
    pub fn restore_dropout_positions(
        &self,
        positions: &HashMap<String, u64>,
    ) -> Result<(), EncoderError> {
        for (n, layer) in self.layers.iter().enumerate() {
            for (site, lin) in lora_sites(layer) {
                lin.restore_dropout_position(&format!("layer.{n}.{site}"), positions)?;
            }
        }
        Ok(())
    }
}

/// The six LoRA-wrappable linear sites of one encoder layer paired with their
/// `named_trainable_weights` site names — the single source of the site→name
/// mapping the weight, dropout-position, and restore traversals all share.
fn lora_sites(layer: &BertLayer) -> [(&'static str, &MaybeLoraLinear); 6] {
    [
        ("query", &layer.attention.self_attention.query),
        ("key", &layer.attention.self_attention.key),
        ("value", &layer.attention.self_attention.value),
        ("dense", &layer.attention.self_output.dense),
        ("intermediate_dense", &layer.intermediate.dense),
        ("output_dense", &layer.output.dense),
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// Builder
// ─────────────────────────────────────────────────────────────────────────────

/// Fluent builder for [`Bert`]. Created via [`Bert::builder`].
pub struct BertBuilder<'a> {
    pooling: Pooling,
    lora: LoraBuildConfig<'a>,
    backbone_dtype: DType,
    adapter_file: Option<&'a Path>,
    /// The wave-3 GGUF-quantized-weight construction seam — see
    /// [`FrozenWeightLookup`]'s own module doc. `None` by default
    /// ([`Bert::builder`]); every EXISTING call site that never calls
    /// [`Self::weight_source`] gets byte-identical Dense-only behavior.
    weight_source: Option<&'a FrozenWeightLookup<'a>>,
}

impl<'a> BertBuilder<'a> {
    /// Pooling strategy applied to the final hidden states by [`Bert::forward`].
    pub fn pooling(mut self, p: Pooling) -> Self {
        self.pooling = p;
        self
    }

    /// LoRA adapter configuration: which linears get wrapped and at what rank.
    pub fn lora(mut self, l: LoraBuildConfig<'a>) -> Self {
        self.lora = l;
        self
    }

    /// Dtype the frozen backbone tensors are mapped at. LoRA A/B always live
    /// in F32.
    pub fn backbone_dtype(mut self, d: DType) -> Self {
        self.backbone_dtype = d;
        self
    }

    /// Optional safetensors file from which to load existing LoRA A/B tensors
    /// (inference mode). When `None`, A/B tensors are registered in the
    /// caller-supplied `VarMap` for training.
    pub fn adapter(mut self, p: Option<&'a Path>) -> Self {
        self.adapter_file = p;
        self
    }

    /// Supply a per-tensor-name GGUF-quantized-weight override — see
    /// [`FrozenWeightLookup`]'s own module doc. Defaulted: a builder that
    /// never calls this stays byte-identical to every prior release (Dense
    /// weights, loaded from `weights_paths`, everywhere).
    pub fn weight_source(mut self, w: &'a FrozenWeightLookup<'a>) -> Self {
        self.weight_source = Some(w);
        self
    }

    /// Materialise the encoder from a frozen safetensors checkpoint.
    pub fn build(
        self,
        weights_paths: &[&Path],
        config: &BertConfig,
        device: &Device,
        varmap: &VarMap,
    ) -> Result<Bert, EncoderError> {
        let frozen_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(weights_paths, self.backbone_dtype, device)?
        };
        let trainable_vb = if let Some(adapter) = self.adapter_file {
            unsafe { VarBuilder::from_mmaped_safetensors(&[adapter], DType::F32, device)? }
        } else {
            VarBuilder::from_varmap(varmap, DType::F32, device)
        };

        // Two checkpoint layouts: raw `BertModel` (no prefix) vs `BertForX`
        // (`"bert."` wrapper). Probe for the embeddings tensor under the
        // wrapped layout; fall back to root if absent.
        let prefix: &str = if frozen_vb.contains_tensor("bert.embeddings.word_embeddings.weight") {
            "bert."
        } else {
            ""
        };
        let base_vb = if prefix.is_empty() {
            frozen_vb.clone()
        } else {
            frozen_vb.pp(prefix.trim_end_matches('.'))
        };

        let embeddings = BertEmbeddings::load(base_vb.pp("embeddings"), config)?;

        let head_dim = config.hidden_size / config.num_attention_heads;
        let h = config.hidden_size;
        let i = config.intermediate_size;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for n in 0..config.num_hidden_layers {
            let layer_vb = base_vb.pp(format!("encoder.layer.{n}"));
            let lora_layer_vb = trainable_vb.pp(format!("layer.{n}"));
            let site = LoraSite {
                layer_vb: &layer_vb,
                lora_layer_vb: &lora_layer_vb,
                layer_idx: n,
                lora: &self.lora,
                varmap,
                weight_source: self.weight_source,
            };

            let query = site.build("attention.self.query", "query", h, h)?;
            let key = site.build("attention.self.key", "key", h, h)?;
            let value = site.build("attention.self.value", "value", h, h)?;
            let attn_output_dense = site.build("attention.output.dense", "dense", h, h)?;
            let attn_output_ln = LayerNorm::new(
                config.hidden_size,
                config.layer_norm_eps,
                true,
                layer_vb.pp("attention.output.LayerNorm"),
            )?;
            let intermediate_dense =
                site.build("intermediate.dense", "intermediate_dense", h, i)?;
            let output_dense = site.build("output.dense", "output_dense", i, h)?;
            let output_ln = LayerNorm::new(
                config.hidden_size,
                config.layer_norm_eps,
                true,
                layer_vb.pp("output.LayerNorm"),
            )?;

            // The attention cascade's `rope_pack` placeholder (issue #462):
            // `[2, 1, 1, 64]`, allocated once here, never read (BERT has no
            // RoPE — see `BertSelfAttention`'s own doc). `64` is
            // `ATTENTION_BLOCK_HEAD_DIM`-shaped by convention only; the
            // fused ops never validate this tensor's shape when `rope ==
            // false` (see `RopeCtx`'s doc), so the literal here need not
            // track `head_dim` at all.
            let rope_placeholder = Tensor::zeros((2, 1, 1, 64), DType::F32, device)?;

            layers.push(BertLayer {
                attention: BertAttention {
                    self_attention: BertSelfAttention {
                        query,
                        key,
                        value,
                        num_attention_heads: config.num_attention_heads,
                        attention_head_size: head_dim,
                        rope_placeholder,
                    },
                    self_output: BertSelfOutput {
                        dense: attn_output_dense,
                        layer_norm: attn_output_ln,
                    },
                },
                intermediate: BertIntermediate {
                    dense: intermediate_dense,
                },
                output: BertOutput {
                    dense: output_dense,
                    layer_norm: output_ln,
                },
            });
        }

        Ok(Bert {
            embeddings,
            layers,
            pooling: self.pooling,
            hidden_size: config.hidden_size,
            max_position_embeddings: config.max_position_embeddings,
            training: false,
        })
    }
}

/// Layer-scoped LoRA injection context. Holds the immutable per-layer state so
/// individual call sites only carry the shape-specific arguments (module name,
/// LoRA subpath, fan-in, fan-out).
struct LoraSite<'a, 'b> {
    layer_vb: &'a VarBuilder<'b>,
    lora_layer_vb: &'a VarBuilder<'b>,
    layer_idx: usize,
    lora: &'a LoraBuildConfig<'a>,
    /// The trainable `VarMap` the seeded LoRA A/B tensors are registered into.
    varmap: &'a VarMap,
    /// The wave-3 GGUF-quantized-weight construction seam — see
    /// [`crate::FrozenWeightLookup`]'s own module doc. `None` at every
    /// EXISTING call site (byte-identical to every prior release).
    weight_source: Option<&'a FrozenWeightLookup<'a>>,
}

impl LoraSite<'_, '_> {
    /// Resolve the frozen base at `layer_vb.pp(module_name)`: consult
    /// `weight_source` first (module doc's `FrozenWeightLookup` contract —
    /// `Ok(Some(base))` uses it directly, `Ok(None)` falls through, `Err`
    /// propagates loudly), falling back to the ORIGINAL Dense `linear(..)`
    /// load when `weight_source` is `None` or misses this name.
    ///
    /// A `weight_source` HIT is geometry-checked
    /// (`validate_frozen_base_geometry`) against the `config.json`-derived
    /// `(in_features, out_features)` this call site expects — a GGUF whose
    /// per-site geometry disagrees with `config.json` fails HERE, at load,
    /// rather than surfacing as a confidently-wrong-shaped matmul at first
    /// inference.
    fn resolve_base(
        &self,
        module_name: &str,
        in_features: usize,
        out_features: usize,
    ) -> Result<FrozenBase, EncoderError> {
        let module_vb = self.layer_vb.pp(module_name);
        if let Some(lookup) = self.weight_source {
            let site = module_vb.prefix();
            if let Some(base) = lookup(&site)? {
                validate_frozen_base_geometry(&site, &base, in_features, out_features)?;
                return Ok(base);
            }
        }
        let base: Linear = linear(in_features, out_features, module_vb)?;
        Ok(FrozenBase::Dense(base))
    }

    /// Resolve the frozen base ([`Self::resolve_base`]) and, if the LoRA
    /// build config matches the site, wrap it in a `LoraLinear`. `lora_subpath`
    /// is the key prefix used to register / load the A/B tensors inside the
    /// trainable `VarBuilder`.
    fn build(
        &self,
        module_name: &str,
        lora_subpath: &str,
        in_features: usize,
        out_features: usize,
    ) -> Result<MaybeLoraLinear, EncoderError> {
        let base = self.resolve_base(module_name, in_features, out_features)?;
        if should_apply_lora(
            module_name,
            self.lora.target_modules,
            // A BERT-family site always belongs to a numbered encoder layer,
            // so the index is always present — see `should_apply_lora`'s own
            // doc for the `None` (unindexed-site) case this family never
            // reaches and the towers with head-side projections do.
            Some(self.layer_idx),
            self.lora.layers_to_transform,
        ) {
            let rank = effective_rank(module_name, self.lora.lora_rank, self.lora.rank_pattern);
            let lora_linear = LoraLinear::new_with_base(
                base,
                rank,
                self.lora.lora_alpha,
                self.lora.use_rslora,
                self.lora.init_mode,
                self.lora.lora_dropout,
                self.lora.seed,
                self.varmap,
                &self.lora_layer_vb.pp(lora_subpath),
            )?;
            Ok(MaybeLoraLinear::Lora(lora_linear))
        } else {
            Ok(MaybeLoraLinear::Frozen(base))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Var;
    use jammi_kernels::ops::ATTENTION_BLOCK_HEAD_DIM;
    use jammi_lora::LoraInitMode;

    use crate::attention_cascade::ATTENTION_BLOCK_COUNTER_TEST_LOCK;

    fn seeded_linear(rows: usize, cols: usize, phase: f32, device: &Device) -> Linear {
        let v: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 + phase) * 0.0137).sin() * 0.2)
            .collect();
        Linear::new(Tensor::from_vec(v, (rows, cols), device).unwrap(), None)
    }

    fn declined_flash() -> FlashDecision {
        FlashDecision::Declined {
            outcome: PredicateOutcome::CapabilityMiss,
            reason: "test_stub_flash_declined",
        }
    }

    fn self_attention_fixture(h: usize, d: usize, device: &Device) -> BertSelfAttention {
        let hd = h * d;
        BertSelfAttention {
            query: MaybeLoraLinear::Frozen(FrozenBase::Dense(seeded_linear(hd, hd, 1.0, device))),
            key: MaybeLoraLinear::Frozen(FrozenBase::Dense(seeded_linear(hd, hd, 2.0, device))),
            value: MaybeLoraLinear::Frozen(FrozenBase::Dense(seeded_linear(hd, hd, 3.0, device))),
            num_attention_heads: h,
            attention_head_size: d,
            rope_placeholder: Tensor::zeros((2, 1, 1, 64), DType::F32, device).unwrap(),
        }
    }

    /// The encoder-level tolerance oracle (contract R6'/R2', the crate's own
    /// precedent — `crate::modernbert::tests::fused_training_attention_block_matches_eager_composition_within_tolerance_global`,
    /// tol `1e-4`): at head64, `BertSelfAttention::forward_training`'s fused
    /// arm (`AttentionBlockFused`, `FullyMaskedPolicy::Propagate`) must
    /// match the shared cascade's own eager composition
    /// (`attention_cascade::forward_eager_training_attention_composition`,
    /// called directly here to force the eager path without disabling any
    /// process-wide admission switch) within `1e-4` on identical `q`/`k`/`v`.
    #[test]
    fn bert_head64_fused_attention_matches_eager_composition_within_tolerance() {
        let _guard = ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let (b, s, h) = (2usize, 5usize, 2usize);
        let d = ATTENTION_BLOCK_HEAD_DIM;
        let attn = self_attention_fixture(h, d, &device);
        let hidden_v: Vec<f32> = (0..b * s * h * d)
            .map(|i| ((i as f32) * 0.031).sin() * 0.4)
            .collect();
        let hidden = Tensor::from_vec(hidden_v, (b, s, h * d), &device).unwrap();
        let extended = Tensor::zeros((b, 1, 1, s), DType::F32, &device).unwrap();
        let fused_masks = FusedAttentionMasks::build(&extended, None, DType::F32).unwrap();
        let flash = declined_flash();

        let before = crate::attention_block_dispatch_snapshot();
        let out_fused = attn
            .forward_training(&hidden, &extended, &fused_masks, &flash)
            .expect("fused training forward");
        let after = crate::attention_block_dispatch_snapshot();
        assert!(
            after.fused > before.fused,
            "head64 must actually dispatch the fused arm (before={before:?}, after={after:?})"
        );
        assert_eq!(
            after.eager, before.eager,
            "head64 must NOT also bump the eager count -- only the fused arm ran \
             (before={before:?}, after={after:?})"
        );

        let q = attn.query.forward(&hidden).unwrap();
        let k = attn.key.forward(&hidden).unwrap();
        let v = attn.value.forward(&hidden).unwrap();
        let qkv = Tensor::cat(&[&q, &k, &v], D::Minus1).unwrap();
        let rope = RopeCtx::Disabled {
            placeholder: &attn.rope_placeholder,
        };
        let out_eager = attention_cascade::forward_eager_training_attention_composition(
            &qkv,
            b,
            s,
            h,
            d,
            &extended,
            None,
            &rope,
            None,
            FullyMaskedPolicy::Propagate,
            true,
        )
        .expect("eager reference composition");

        let fv: Vec<f32> = out_fused.flatten_all().unwrap().to_vec1().unwrap();
        let ev: Vec<f32> = out_eager.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(fv.len(), ev.len());
        assert!(fv.iter().all(|x| x.is_finite()) && ev.iter().all(|x| x.is_finite()));
        let max_diff = fv
            .iter()
            .zip(&ev)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "fused vs eager attention diverged beyond tolerance: max|Δ|={max_diff}"
        );
    }

    /// The all-padding-row case (contract R3'/R6'): one sequence entirely
    /// masked. Under `Propagate`, the fused arm must reproduce the eager
    /// composition's own (finite, `MASKED_LOGIT`-convention) output on that
    /// row exactly like every other row — no `Zeros`-style third numeric
    /// path exists for BERT (see `BertSelfAttention::forward_training`'s
    /// doc: `Propagate` is the ONE policy this crate's BERT/DistilBERT
    /// callers ever construct).
    ///
    /// The mask is built through [`crate::mask::extended_attention_mask`]
    /// (the PRODUCTION builder — `[1, 0]` u32 in, `[0.0, MASKED_LOGIT]`
    /// f32 out) rather than a hand-rolled `affine`: an audit round found
    /// this test previously constructed `affine(-10_000.0, 10_000.0)`,
    /// which is `mask.rs`'s own convention (`affine(-MASKED_LOGIT,
    /// MASKED_LOGIT)` = `affine(10_000.0, -10_000.0)`) SIGN-INVERTED —
    /// padding became `+10_000` rather than `MASKED_LOGIT`
    /// (`-10_000`), so the padding row's raw scores were BOOSTED, not
    /// suppressed, `jammi_kernels::ops::softmax::row_is_fully_masked`
    /// never fired, and a uniform positive additive constant is a
    /// softmax no-op either way — the test asserted a real tolerance
    /// bound while never actually reaching the fully-masked branch it
    /// claimed to cover. The negative control below is exactly the proof
    /// that this version does reach that branch.
    #[test]
    fn bert_head64_all_padding_row_propagate_fused_matches_eager_within_tolerance() {
        let _guard = ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let (b, s, h) = (2usize, 4usize, 1usize);
        let d = ATTENTION_BLOCK_HEAD_DIM;
        let attn = self_attention_fixture(h, d, &device);
        let hidden_v: Vec<f32> = (0..b * s * h * d)
            .map(|i| ((i as f32) * 0.021).cos() * 0.3)
            .collect();
        let hidden = Tensor::from_vec(hidden_v, (b, s, h * d), &device).unwrap();
        // Row 0 real, row 1 entirely padding.
        let mask_u32 = Tensor::from_slice(&[1u32, 1, 1, 1, 0, 0, 0, 0], (b, s), &device).unwrap();
        let extended = crate::mask::extended_attention_mask(&mask_u32).unwrap();
        let fused_masks = FusedAttentionMasks::build(&extended, None, DType::F32).unwrap();
        let flash = declined_flash();

        let out_fused = attn
            .forward_training(&hidden, &extended, &fused_masks, &flash)
            .expect("fused training forward on an all-padding row");

        let q = attn.query.forward(&hidden).unwrap();
        let k = attn.key.forward(&hidden).unwrap();
        let v = attn.value.forward(&hidden).unwrap();
        let qkv = Tensor::cat(&[&q, &k, &v], D::Minus1).unwrap();
        let rope = RopeCtx::Disabled {
            placeholder: &attn.rope_placeholder,
        };
        let out_eager = attention_cascade::forward_eager_training_attention_composition(
            &qkv,
            b,
            s,
            h,
            d,
            &extended,
            None,
            &rope,
            None,
            FullyMaskedPolicy::Propagate,
            true,
        )
        .expect("eager reference composition on an all-padding row");

        let fv: Vec<f32> = out_fused.flatten_all().unwrap().to_vec1().unwrap();
        let ev: Vec<f32> = out_eager.flatten_all().unwrap().to_vec1().unwrap();
        assert!(fv.iter().all(|x| x.is_finite()) && ev.iter().all(|x| x.is_finite()));
        let max_diff = fv
            .iter()
            .zip(&ev)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "all-padding-row Propagate: fused vs eager diverged beyond tolerance: max|Δ|={max_diff}"
        );

        // NEGATIVE CONTROL: re-run the SAME fused forward with
        // `FullyMaskedPolicy::Zeros` in place of `Propagate` (bypassing
        // `forward_training`'s hardcoded `Propagate` by calling the
        // shared cascade directly — BERT's own seam never constructs
        // `Zeros`, but the cascade itself accepts either as construction
        // data). Batch row 1 is entirely padding, so under `Zeros` its
        // output must come out EXACTLY zero, and that exact-zero output
        // MUST differ from the `Propagate` eager reference computed
        // above on the SAME row. If the two policies produced the same
        // values here, this test would not actually be exercising
        // `row_is_fully_masked` at all — exactly the failure mode the
        // sign-inverted mask silently produced before this fix.
        let masks_for_zeros = TrainingMaskInputs {
            extended: &extended,
            local_band: None,
            fused: Some(&fused_masks),
        };
        let out_zeros = attention_cascade::training_attention_cascade(
            &qkv,
            b,
            s,
            h,
            d,
            masks_for_zeros,
            &flash,
            &rope,
            None,
            None,
            FullyMaskedPolicy::Zeros,
            |_admission| {
                unreachable!("BERT always supplies FlashDecision::Declined in this fixture")
            },
        )
        .expect("fused training forward with FullyMaskedPolicy::Zeros on an all-padding row");

        let zv: Vec<f32> = out_zeros.flatten_all().unwrap().to_vec1().unwrap();
        let row_len = s * h * d;
        let zeros_row1 = &zv[row_len..2 * row_len];
        let eager_row1 = &ev[row_len..2 * row_len];
        assert!(
            zeros_row1.iter().all(|&x| x == 0.0),
            "FullyMaskedPolicy::Zeros must zero the fully-masked row EXACTLY: {zeros_row1:?}"
        );
        let row_diff = zeros_row1
            .iter()
            .zip(eager_row1)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            row_diff > 1e-3,
            "Zeros and Propagate must diverge on the fully-masked row (both computed through \
             the fused arm) -- a max|Δ|={row_diff} this small would mean the fully-masked \
             branch was never actually reached, the same silent gap the inverted mask sign left"
        );
    }

    /// LoRA gradients on both `query` and `value` must be finite and
    /// non-zero after a fused-arm training forward+backward at head64 —
    /// the fused whole-attention-block op's own backward reaches every
    /// LoRA `A`/`B` tensor through the SAME `Wqkv`-projection chain the
    /// eager composition would (contract R6' precedent:
    /// `crate::modernbert::tests::fused_attention_block_matches_eager_lora_gradients_at_production_seq_on_head64`).
    #[test]
    fn bert_head64_fused_attention_lora_gradients_are_finite_and_nonzero() {
        let _guard = ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let (b, s, h) = (2usize, 5usize, 2usize);
        let d = ATTENTION_BLOCK_HEAD_DIM;
        let hd = h * d;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let query = LoraLinear::new_with_base(
            FrozenBase::Dense(seeded_linear(hd, hd, 1.0, &device)),
            4,
            8.0,
            false,
            LoraInitMode::Gaussian,
            None,
            11,
            &varmap,
            &vb.pp("query"),
        )
        .unwrap();
        let value = LoraLinear::new_with_base(
            FrozenBase::Dense(seeded_linear(hd, hd, 3.0, &device)),
            4,
            8.0,
            false,
            LoraInitMode::Gaussian,
            None,
            13,
            &varmap,
            &vb.pp("value"),
        )
        .unwrap();
        let attn = BertSelfAttention {
            query: MaybeLoraLinear::Lora(query),
            key: MaybeLoraLinear::Frozen(FrozenBase::Dense(seeded_linear(hd, hd, 2.0, &device))),
            value: MaybeLoraLinear::Lora(value),
            num_attention_heads: h,
            attention_head_size: d,
            rope_placeholder: Tensor::zeros((2, 1, 1, 64), DType::F32, &device).unwrap(),
        };

        let hidden_v: Vec<f32> = (0..b * s * hd)
            .map(|i| ((i as f32) * 0.019).sin() * 0.3)
            .collect();
        let hidden =
            Var::from_tensor(&Tensor::from_vec(hidden_v, (b, s, hd), &device).unwrap()).unwrap();
        let extended = Tensor::zeros((b, 1, 1, s), DType::F32, &device).unwrap();
        let fused_masks = FusedAttentionMasks::build(&extended, None, DType::F32).unwrap();
        let flash = declined_flash();

        let before = crate::attention_block_dispatch_snapshot();
        let out = attn
            .forward_training(hidden.as_tensor(), &extended, &fused_masks, &flash)
            .expect("fused training forward");
        let after = crate::attention_block_dispatch_snapshot();
        assert!(after.fused > before.fused, "must dispatch fused at head64");
        assert_eq!(
            after.eager, before.eager,
            "head64 must NOT also bump the eager count -- only the fused arm ran"
        );

        let loss = out.sum_all().unwrap();
        let grads = loss
            .backward()
            .expect("backward through the fused attention block");

        for (name, lin) in [("query", &attn.query), ("value", &attn.value)] {
            let MaybeLoraLinear::Lora(l) = lin else {
                panic!("{name} must be LoRA-wrapped in this fixture")
            };
            for t in l.trainable_params() {
                let g = grads
                    .get(t)
                    .unwrap_or_else(|| panic!("no gradient reached {name}'s LoRA tensor"));
                let gv: Vec<f32> = g.flatten_all().unwrap().to_vec1().unwrap();
                assert!(
                    gv.iter().all(|x| x.is_finite()),
                    "{name}'s LoRA gradient must be finite: {gv:?}"
                );
                assert!(
                    gv.iter().any(|x| *x != 0.0),
                    "{name}'s LoRA gradient must be non-vacuously non-zero"
                );
            }
        }
    }

    /// The real fused-vs-eager LoRA-gradient EQUALITY oracle (contract
    /// fix-round item 2 — the crate's own precedent,
    /// `crate::modernbert::tests::fused_attention_block_matches_eager_lora_gradients_at_production_seq_on_head64`,
    /// tol `1e-4`, mirrored exactly in shape here): both `query`'s and
    /// `value`'s LoRA `A`/`B` gradients from a fused-arm training
    /// forward+backward must match the SAME gradients from the eager
    /// composition (`attention_cascade::forward_eager_training_attention_composition`,
    /// called directly to force the eager path, never through
    /// `JAMMI_KERNELS_DISABLE`) within `1e-4` — not merely finite/non-zero
    /// (that weaker check is
    /// [`bert_head64_fused_attention_lora_gradients_are_finite_and_nonzero`],
    /// kept alongside as its own, narrower oracle). Both arms share the
    /// SAME LoRA `A`/`B` tensors and the same frozen `key` weights (one
    /// fixture, two independent forward+`backward()` calls — candle's
    /// `GradStore` is fresh per call, so this double-backward through
    /// shared leaves is safe); only the attention arithmetic differs, so
    /// any drift beyond tolerance is a real fused-vs-eager numeric
    /// divergence reaching the LoRA gradient, not a fixture difference. A
    /// non-uniform cotangent (`dy`, the same discipline the modernbert
    /// precedent's own `dy` documents) is used rather than a plain
    /// `sum_all()`, whose uniform gradient would not discriminate a
    /// permutation-shaped defect from a correct one.
    #[test]
    fn bert_head64_fused_attention_lora_gradients_match_eager_within_tolerance() {
        let _guard = ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let (b, s, h) = (2usize, 5usize, 2usize);
        let d = ATTENTION_BLOCK_HEAD_DIM;
        let hd = h * d;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let query = LoraLinear::new_with_base(
            FrozenBase::Dense(seeded_linear(hd, hd, 1.0, &device)),
            4,
            8.0,
            false,
            LoraInitMode::Gaussian,
            None,
            11,
            &varmap,
            &vb.pp("query"),
        )
        .unwrap();
        let value = LoraLinear::new_with_base(
            FrozenBase::Dense(seeded_linear(hd, hd, 3.0, &device)),
            4,
            8.0,
            false,
            LoraInitMode::Gaussian,
            None,
            13,
            &varmap,
            &vb.pp("value"),
        )
        .unwrap();
        let attn = BertSelfAttention {
            query: MaybeLoraLinear::Lora(query),
            key: MaybeLoraLinear::Frozen(FrozenBase::Dense(seeded_linear(hd, hd, 2.0, &device))),
            value: MaybeLoraLinear::Lora(value),
            num_attention_heads: h,
            attention_head_size: d,
            rope_placeholder: Tensor::zeros((2, 1, 1, 64), DType::F32, &device).unwrap(),
        };

        let hidden_v: Vec<f32> = (0..b * s * hd)
            .map(|i| ((i as f32) * 0.019).sin() * 0.3)
            .collect();
        let hidden = Tensor::from_vec(hidden_v, (b, s, hd), &device).unwrap();
        let extended = Tensor::zeros((b, 1, 1, s), DType::F32, &device).unwrap();
        let fused_masks = FusedAttentionMasks::build(&extended, None, DType::F32).unwrap();
        let flash = declined_flash();
        let dy_v: Vec<f32> = (0..b * s * hd)
            .map(|i| ((i as f32) * 0.0071).cos() * 0.6 + 0.05)
            .collect();
        let dy = Tensor::from_vec(dy_v, (b, s, hd), &device).unwrap();

        let before = crate::attention_block_dispatch_snapshot();
        let out_fused = attn
            .forward_training(&hidden, &extended, &fused_masks, &flash)
            .expect("fused training forward");
        let after = crate::attention_block_dispatch_snapshot();
        assert!(after.fused > before.fused, "must dispatch fused at head64");
        assert_eq!(
            after.eager, before.eager,
            "head64 must NOT also bump the eager count -- only the fused arm ran"
        );
        let loss_fused = (&out_fused * &dy).unwrap().sum_all().unwrap();
        let grads_fused = loss_fused
            .backward()
            .expect("backward through the fused attention block");

        let q = attn.query.forward(&hidden).unwrap();
        let k = attn.key.forward(&hidden).unwrap();
        let v = attn.value.forward(&hidden).unwrap();
        let qkv = Tensor::cat(&[&q, &k, &v], D::Minus1).unwrap();
        let rope = RopeCtx::Disabled {
            placeholder: &attn.rope_placeholder,
        };
        let out_eager = attention_cascade::forward_eager_training_attention_composition(
            &qkv,
            b,
            s,
            h,
            d,
            &extended,
            None,
            &rope,
            None,
            FullyMaskedPolicy::Propagate,
            true,
        )
        .expect("eager reference composition");
        let loss_eager = (&out_eager * &dy).unwrap().sum_all().unwrap();
        let grads_eager = loss_eager
            .backward()
            .expect("backward through the eager attention composition");

        const TOL: f32 = 1e-4;
        for (name, lin) in [("query", &attn.query), ("value", &attn.value)] {
            let MaybeLoraLinear::Lora(l) = lin else {
                panic!("{name} must be LoRA-wrapped in this fixture")
            };
            for t in l.trainable_params() {
                let gf: Vec<f32> = grads_fused
                    .get(t)
                    .unwrap_or_else(|| panic!("no fused-arm gradient reached {name}'s LoRA tensor"))
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
                let ge: Vec<f32> = grads_eager
                    .get(t)
                    .unwrap_or_else(|| panic!("no eager-arm gradient reached {name}'s LoRA tensor"))
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
                assert_eq!(gf.len(), ge.len());
                assert!(
                    gf.iter().all(|x| x.is_finite()) && ge.iter().all(|x| x.is_finite()),
                    "{name}'s LoRA gradients must be finite on both arms: fused={gf:?} eager={ge:?}"
                );
                let mut max_delta = 0f32;
                for (&f, &e) in gf.iter().zip(ge.iter()) {
                    max_delta = max_delta.max((f - e).abs());
                }
                assert!(
                    max_delta <= TOL,
                    "{name}'s LoRA gradient: fused vs eager max|Δ|={max_delta:e} > {TOL:e}"
                );
            }
        }
    }

    /// Strict mode on a refused domain (family K2): a `head_dim != 64`
    /// shape is a `false` outcome from `attention_block_admission_predicate`
    /// — `BertSelfAttention::forward_training` reaches it through `admit()`
    /// (a two-arm dispatch: no `DomainMiss`/`CapabilityMiss` split exists at
    /// that level, unlike the cascade arms — see `admit`'s own doc), so
    /// under `Strict` ANY failed predicate is `KernelError::StrictModeFallback`,
    /// unconditionally. `JAMMI_KERNELS_STRICT` is a process-wide `OnceLock`
    /// (`admission_mode`'s own doc), so this runs in a FRESH child process —
    /// the same pattern
    /// `crate::modernbert::tests::strict_mode_padded_flash_dispatch_is_unchanged_in_a_fresh_process`
    /// uses.
    #[test]
    fn bert_strict_mode_on_a_refused_domain_is_a_typed_error_in_a_fresh_process() {
        let exe = std::env::current_exe().expect("test binary path");
        let output = std::process::Command::new(exe)
            .args([
                "bert::tests::strict_mode_child_process_body",
                "--exact",
                "--nocapture",
                "--ignored",
            ])
            .env("JAMMI_KERNELS_STRICT", "1")
            .output()
            .expect("spawn child test binary");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "child process assertion failed: stdout={stdout}\nstderr={}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            stdout.contains("1 passed"),
            "the child process must have actually run (and passed) exactly one test -- \
             stdout={stdout}"
        );
    }

    /// Only meaningful inside the child process the test above spawns.
    /// `#[ignore]`d so the NORMAL (non-Strict) test run never executes it
    /// directly — only the `--ignored --exact` child-process invocation
    /// does.
    #[test]
    #[ignore]
    fn strict_mode_child_process_body() {
        let device = Device::Cpu;
        let (b, s, h, d) = (1usize, 4usize, 1usize, 16usize);
        let attn = self_attention_fixture(h, d, &device);
        let hidden_v: Vec<f32> = (0..b * s * h * d).map(|i| i as f32 * 0.01).collect();
        let hidden = Tensor::from_vec(hidden_v, (b, s, h * d), &device).unwrap();
        let extended = Tensor::zeros((b, 1, 1, s), DType::F32, &device).unwrap();
        let fused_masks = FusedAttentionMasks::build(&extended, None, DType::F32).unwrap();
        let flash = declined_flash();

        let err = attn
            .forward_training(&hidden, &extended, &fused_masks, &flash)
            .expect_err("head_dim=16 under Strict must be a typed refusal, not a silent eager");
        let msg = err.to_string();
        assert!(
            msg.contains("attention_block_fused") || msg.contains("head_dim"),
            "expected a StrictModeFallback naming the refused op/predicate: {msg}"
        );
    }
}
