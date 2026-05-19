//! BERT-family encoder (BERT, RoBERTa, CamemBERT, XLM-RoBERTa).
//!
//! Mirrors candle-transformers' [`BertModel`] architecture so the parity test
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
use candle_nn::{
    embedding, layer_norm, linear, Embedding, LayerNorm, Linear, Module, VarBuilder, VarMap,
};
use jammi_lora::{effective_rank, should_apply_lora, LoraBuildConfig, LoraLinear, MaybeLoraLinear};

use crate::error::EncoderError;
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
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
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
        Ok(self.layer_norm.forward(&embeddings)?)
    }
}

struct BertSelfAttention {
    query: MaybeLoraLinear,
    key: MaybeLoraLinear,
    value: MaybeLoraLinear,
    num_attention_heads: usize,
    attention_head_size: usize,
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

        let scores = q.matmul(&k.t()?)?;
        let scores = (scores / (self.attention_head_size as f64).sqrt())?;
        let scores = scores.broadcast_add(extended_mask)?;
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;

        let context = probs.matmul(&v)?;
        let context = context.transpose(1, 2)?.contiguous()?;
        Ok(context.flatten_from(D::Minus2)?)
    }
}

struct BertSelfOutput {
    dense: MaybeLoraLinear,
    layer_norm: LayerNorm,
}

impl BertSelfOutput {
    fn forward(&self, hidden: &Tensor, input_tensor: &Tensor) -> Result<Tensor, EncoderError> {
        let hidden = self.dense.forward(hidden)?;
        Ok(self.layer_norm.forward(&(hidden + input_tensor)?)?)
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
}

struct BertIntermediate {
    dense: MaybeLoraLinear,
}

impl BertIntermediate {
    fn forward(&self, hidden: &Tensor) -> Result<Tensor, EncoderError> {
        let hidden = self.dense.forward(hidden)?;
        Ok(hidden.gelu_erf()?)
    }
}

struct BertOutput {
    dense: MaybeLoraLinear,
    layer_norm: LayerNorm,
}

impl BertOutput {
    fn forward(&self, hidden: &Tensor, input_tensor: &Tensor) -> Result<Tensor, EncoderError> {
        let hidden = self.dense.forward(hidden)?;
        Ok(self.layer_norm.forward(&(hidden + input_tensor)?)?)
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
        let intermediate_output = self.intermediate.forward(&attention_output)?;
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

    /// Raw `[batch, seq, hidden]` output before pooling.
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
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &extended)?;
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

    /// Switch every LoRA-wrapped linear into / out of training mode (gates
    /// dropout on the adapter path).
    pub fn set_training(&mut self, training: bool) {
        for layer in &mut self.layers {
            layer.attention.self_attention.query.set_training(training);
            layer.attention.self_attention.key.set_training(training);
            layer.attention.self_attention.value.set_training(training);
            layer.attention.self_output.dense.set_training(training);
            layer.intermediate.dense.set_training(training);
            layer.output.dense.set_training(training);
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
            };

            let query = site.build("attention.self.query", "query", h, h)?;
            let key = site.build("attention.self.key", "key", h, h)?;
            let value = site.build("attention.self.value", "value", h, h)?;
            let attn_output_dense = site.build("attention.output.dense", "dense", h, h)?;
            let attn_output_ln = layer_norm(
                config.hidden_size,
                config.layer_norm_eps,
                layer_vb.pp("attention.output.LayerNorm"),
            )?;
            let intermediate_dense =
                site.build("intermediate.dense", "intermediate_dense", h, i)?;
            let output_dense = site.build("output.dense", "output_dense", i, h)?;
            let output_ln = layer_norm(
                config.hidden_size,
                config.layer_norm_eps,
                layer_vb.pp("output.LayerNorm"),
            )?;

            layers.push(BertLayer {
                attention: BertAttention {
                    self_attention: BertSelfAttention {
                        query,
                        key,
                        value,
                        num_attention_heads: config.num_attention_heads,
                        attention_head_size: head_dim,
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
}

impl LoraSite<'_, '_> {
    /// Load the frozen `Linear` at `layer_vb.pp(module_name)` and, if the LoRA
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
        let base: Linear = linear(in_features, out_features, self.layer_vb.pp(module_name))?;
        if should_apply_lora(
            module_name,
            self.lora.target_modules,
            self.layer_idx,
            self.lora.layers_to_transform,
        ) {
            let rank = effective_rank(module_name, self.lora.lora_rank, self.lora.rank_pattern);
            let lora_linear = LoraLinear::new(
                base,
                rank,
                self.lora.lora_alpha,
                self.lora.use_rslora,
                self.lora.init_mode,
                self.lora.lora_dropout,
                &self.lora_layer_vb.pp(lora_subpath),
            )?;
            Ok(MaybeLoraLinear::Lora(lora_linear))
        } else {
            Ok(MaybeLoraLinear::Frozen(base))
        }
    }
}
