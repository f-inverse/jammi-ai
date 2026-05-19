//! DistilBERT encoder with built-in PEFT support via [`jammi_lora`].
//!
//! Mirrors candle-transformers' `models::distilbert::DistilBertModel` in
//! structure, with the following per-spec differences from BERT:
//!
//! - No `token_type_embeddings` — embeddings are word + position only.
//! - Attention linears are named `q_lin` / `k_lin` / `v_lin` / `out_lin`.
//! - FFN linears are named `lin1` (dim → hidden_dim) and `lin2` (hidden_dim → dim).
//! - Per-layer LayerNorms are `sa_layer_norm` (post-attention) and
//!   `output_layer_norm` (post-FFN).
//! - Weight prefix in the safetensors archive is `"distilbert."`.
//! - Post-LayerNorm architecture (residual then LayerNorm).
//! - Activation is `gelu_erf`.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{Embedding, VarBuilder, VarMap};
use jammi_lora::{effective_rank, should_apply_lora, LoraBuildConfig, LoraLinear, MaybeLoraLinear};

use crate::error::EncoderError;
use crate::layer_norm::LayerNorm;
use crate::pooling::{pool_and_normalize, Pooling};

/// DistilBERT architecture configuration.
///
/// Field names match the HuggingFace `config.json` naming for DistilBERT
/// checkpoints (`dim` / `n_layers` / `n_heads` / `hidden_dim`). The serde
/// defaults match canonical HuggingFace DistilBERT.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct DistilBertConfig {
    /// Hidden size of token embeddings and attention output.
    #[serde(rename = "dim")]
    pub hidden_size: usize,
    /// Number of transformer blocks.
    #[serde(rename = "n_layers")]
    pub num_hidden_layers: usize,
    /// Number of attention heads. Must divide `hidden_size` evenly.
    #[serde(rename = "n_heads")]
    pub num_attention_heads: usize,
    /// FFN intermediate size (`lin1` output, `lin2` input).
    #[serde(rename = "hidden_dim")]
    pub intermediate_size: usize,
    /// Vocabulary size of the word-embedding matrix.
    pub vocab_size: usize,
    /// Maximum positional capacity. Inputs longer than this are rejected.
    pub max_position_embeddings: usize,
    /// LayerNorm epsilon — DistilBERT uses 1e-12.
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
}

fn default_layer_norm_eps() -> f64 {
    1e-12
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-structures — mirror candle-transformers' DistilBertModel layout.
// ─────────────────────────────────────────────────────────────────────────────

struct DistilBertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    layer_norm: LayerNorm,
}

impl DistilBertEmbeddings {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor, EncoderError> {
        let (_batch, seq) = input_ids.dims2()?;
        let word_emb = self.word_embeddings.forward(input_ids)?;
        // Position IDs as 1-D [seq], broadcast-added across the batch.
        let position_ids = Tensor::arange(0u32, seq as u32, input_ids.device())?;
        let position_emb = self.position_embeddings.forward(&position_ids)?;
        let embeddings = word_emb.broadcast_add(&position_emb)?;
        self.layer_norm.forward(&embeddings)
    }
}

struct DistilBertSelfAttention {
    q_lin: MaybeLoraLinear,
    k_lin: MaybeLoraLinear,
    v_lin: MaybeLoraLinear,
    out_lin: MaybeLoraLinear,
    num_attention_heads: usize,
    attention_head_size: usize,
}

impl DistilBertSelfAttention {
    /// Reshape `[B, S, H]` to `[B, h, S, d]` and materialise contiguously.
    ///
    /// The `.contiguous()` call is the canonical fix for candle's matmul
    /// contiguity panic on transposed inputs (see candle issue #1965 /
    /// PR #3088); it is load-bearing and must not be removed.
    fn transpose_for_scores(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        let mut new_shape = x.dims().to_vec();
        new_shape.pop();
        new_shape.push(self.num_attention_heads);
        new_shape.push(self.attention_head_size);
        let x = x.reshape(new_shape.as_slice())?.transpose(1, 2)?;
        Ok(x.contiguous()?)
    }

    fn forward(&self, hidden: &Tensor, extended_mask: &Tensor) -> Result<Tensor, EncoderError> {
        let q = self.q_lin.forward(hidden)?;
        let k = self.k_lin.forward(hidden)?;
        let v = self.v_lin.forward(hidden)?;

        let q = self.transpose_for_scores(&q)?;
        let k = self.transpose_for_scores(&k)?;
        let v = self.transpose_for_scores(&v)?;

        let scores = (q.matmul(&k.t()?)? / (self.attention_head_size as f64).sqrt())?;
        let scores = scores.broadcast_add(extended_mask)?;
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;

        let context = probs.matmul(&v)?;
        // Re-materialise contiguously after the transpose, then collapse the
        // head/dim trailing axes back into hidden.
        let context = context.transpose(1, 2)?.contiguous()?;
        let context = context.flatten_from(D::Minus2)?;
        Ok(self.out_lin.forward(&context)?)
    }
}

struct DistilBertFfn {
    lin1: MaybeLoraLinear,
    lin2: MaybeLoraLinear,
}

impl DistilBertFfn {
    fn forward(&self, hidden: &Tensor) -> Result<Tensor, EncoderError> {
        let mid = self.lin1.forward(hidden)?;
        let activated = mid.gelu_erf()?;
        Ok(self.lin2.forward(&activated)?)
    }
}

struct DistilBertLayer {
    attention: DistilBertSelfAttention,
    sa_layer_norm: LayerNorm,
    ffn: DistilBertFfn,
    output_layer_norm: LayerNorm,
}

impl DistilBertLayer {
    fn forward(&self, hidden: &Tensor, extended_mask: &Tensor) -> Result<Tensor, EncoderError> {
        // Post-LN attention: residual then LayerNorm.
        let attn_out = self.attention.forward(hidden, extended_mask)?;
        let attn_residual = (attn_out + hidden)?;
        let attn_normed = self.sa_layer_norm.forward(&attn_residual)?;

        // Post-LN FFN: residual then LayerNorm.
        let ffn_out = self.ffn.forward(&attn_normed)?;
        let ffn_residual = (ffn_out + &attn_normed)?;
        self.output_layer_norm.forward(&ffn_residual)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public encoder
// ─────────────────────────────────────────────────────────────────────────────

/// DistilBERT sentence encoder with selective LoRA adapters on attention and
/// FFN linears.
pub struct DistilBert {
    embeddings: DistilBertEmbeddings,
    layers: Vec<DistilBertLayer>,
    pooling: Pooling,
    hidden_size: usize,
    max_position_embeddings: usize,
}

impl DistilBert {
    /// Start a builder with the default pooling (`Mean`), no adapter, and
    /// F32 backbone dtype.
    pub fn builder() -> DistilBertBuilder<'static> {
        DistilBertBuilder {
            pooling: Pooling::default(),
            lora: LoraBuildConfig::frozen(),
            backbone_dtype: DType::F32,
            adapter_file: None,
        }
    }

    /// Configured backbone hidden size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Configured maximum sequence length (positional capacity).
    pub fn max_seq_length(&self) -> usize {
        self.max_position_embeddings
    }

    /// Raw hidden states `[batch, seq, hidden]` from the final transformer
    /// block. Sequence length is bounded by [`Self::max_seq_length`].
    pub fn forward_hidden(
        &self,
        input_ids: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor, EncoderError> {
        let (_, seq) = input_ids.dims2()?;
        if seq > self.max_position_embeddings {
            return Err(EncoderError::SequenceTooLong {
                seq,
                max: self.max_position_embeddings,
            });
        }

        let mut hidden = self.embeddings.forward(input_ids)?;
        let extended = crate::mask::extended_attention_mask(mask)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &extended)?;
        }
        Ok(hidden)
    }

    /// Pooled + L2-normalised sentence embedding `[batch, hidden]`.
    pub fn forward(&self, input_ids: &Tensor, mask: &Tensor) -> Result<Tensor, EncoderError> {
        let hidden = self.forward_hidden(input_ids, mask)?;
        pool_and_normalize(&hidden, mask, self.pooling)
    }

    /// References to every trainable LoRA parameter (A/B matrices) in layer
    /// order. Empty when no LoRA adapters are installed.
    pub fn trainable_params(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.attention.q_lin.trainable_params());
            params.extend(layer.attention.k_lin.trainable_params());
            params.extend(layer.attention.v_lin.trainable_params());
            params.extend(layer.attention.out_lin.trainable_params());
            params.extend(layer.ffn.lin1.trainable_params());
            params.extend(layer.ffn.lin2.trainable_params());
        }
        params
    }

    /// Trainable LoRA weights keyed by `layer.{n}.{module}.{lora_a|lora_b}`
    /// for safetensors persistence. Frozen layers contribute no entries.
    pub fn named_trainable_weights(&self) -> Result<HashMap<String, Tensor>, EncoderError> {
        let mut out = HashMap::new();
        for (n, layer) in self.layers.iter().enumerate() {
            out.extend(
                layer
                    .attention
                    .q_lin
                    .named_weights(&format!("layer.{n}.q_lin"))?,
            );
            out.extend(
                layer
                    .attention
                    .k_lin
                    .named_weights(&format!("layer.{n}.k_lin"))?,
            );
            out.extend(
                layer
                    .attention
                    .v_lin
                    .named_weights(&format!("layer.{n}.v_lin"))?,
            );
            out.extend(
                layer
                    .attention
                    .out_lin
                    .named_weights(&format!("layer.{n}.out_lin"))?,
            );
            out.extend(layer.ffn.lin1.named_weights(&format!("layer.{n}.lin1"))?);
            out.extend(layer.ffn.lin2.named_weights(&format!("layer.{n}.lin2"))?);
        }
        Ok(out)
    }

    /// Toggle training mode on every LoRA-augmented linear and every LayerNorm.
    /// LoRA layers gate dropout; LayerNorms switch between the fused no-bwd
    /// eval kernel and the primitive-op composition whose backward is well-
    /// defined.
    pub fn set_training(&mut self, training: bool) {
        self.embeddings.layer_norm.set_training(training);
        for layer in &mut self.layers {
            layer.attention.q_lin.set_training(training);
            layer.attention.k_lin.set_training(training);
            layer.attention.v_lin.set_training(training);
            layer.attention.out_lin.set_training(training);
            layer.sa_layer_norm.set_training(training);
            layer.ffn.lin1.set_training(training);
            layer.ffn.lin2.set_training(training);
            layer.output_layer_norm.set_training(training);
        }
    }

    /// Reload LoRA A/B tensors from a `layer.{n}.{module}.{lora_a|lora_b}`
    /// hashmap (as produced by [`Self::named_trainable_weights`]). Keys for
    /// frozen layers are silently ignored.
    pub fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<(), EncoderError> {
        for (n, layer) in self.layers.iter_mut().enumerate() {
            layer
                .attention
                .q_lin
                .load_weights(weights, &format!("layer.{n}.q_lin"));
            layer
                .attention
                .k_lin
                .load_weights(weights, &format!("layer.{n}.k_lin"));
            layer
                .attention
                .v_lin
                .load_weights(weights, &format!("layer.{n}.v_lin"));
            layer
                .attention
                .out_lin
                .load_weights(weights, &format!("layer.{n}.out_lin"));
            layer
                .ffn
                .lin1
                .load_weights(weights, &format!("layer.{n}.lin1"));
            layer
                .ffn
                .lin2
                .load_weights(weights, &format!("layer.{n}.lin2"));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Builder
// ─────────────────────────────────────────────────────────────────────────────

/// Builder for [`DistilBert`]. Construct via [`DistilBert::builder`].
pub struct DistilBertBuilder<'a> {
    pooling: Pooling,
    lora: LoraBuildConfig<'a>,
    backbone_dtype: DType,
    adapter_file: Option<&'a Path>,
}

impl<'a> DistilBertBuilder<'a> {
    /// Select the pooling strategy applied by [`DistilBert::forward`].
    pub fn pooling(mut self, p: Pooling) -> Self {
        self.pooling = p;
        self
    }

    /// Select which linears receive LoRA adapters and at what rank / scaling.
    pub fn lora(mut self, l: LoraBuildConfig<'a>) -> Self {
        self.lora = l;
        self
    }

    /// Set the dtype the frozen backbone weights are loaded as.
    pub fn backbone_dtype(mut self, d: DType) -> Self {
        self.backbone_dtype = d;
        self
    }

    /// Optionally load LoRA A/B tensors from a safetensors adapter file
    /// instead of initialising them from the supplied [`VarMap`].
    pub fn adapter(mut self, p: Option<&'a Path>) -> Self {
        self.adapter_file = p;
        self
    }

    /// Load the frozen backbone from `weights_paths`, construct LoRA wrappers
    /// per the [`LoraBuildConfig`], and assemble the [`DistilBert`] encoder.
    pub fn build(
        self,
        weights_paths: &[&Path],
        config: &DistilBertConfig,
        device: &Device,
        varmap: &VarMap,
    ) -> Result<DistilBert, EncoderError> {
        if config.num_attention_heads == 0 || config.hidden_size % config.num_attention_heads != 0 {
            return Err(EncoderError::Config(format!(
                "hidden_size {} not divisible by num_attention_heads {}",
                config.hidden_size, config.num_attention_heads
            )));
        }
        let attention_head_size = config.hidden_size / config.num_attention_heads;

        let frozen_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(weights_paths, self.backbone_dtype, device)?
        };
        let lora_vb = if let Some(af) = self.adapter_file {
            unsafe { VarBuilder::from_mmaped_safetensors(&[af], DType::F32, device)? }
        } else {
            VarBuilder::from_varmap(varmap, DType::F32, device)
        };

        let base_vb = frozen_vb.pp("distilbert");

        let emb_vb = base_vb.pp("embeddings");
        let word_embeddings = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            emb_vb.pp("word_embeddings"),
        )?;
        let position_embeddings = candle_nn::embedding(
            config.max_position_embeddings,
            config.hidden_size,
            emb_vb.pp("position_embeddings"),
        )?;
        let emb_layer_norm = LayerNorm::new(
            config.hidden_size,
            config.layer_norm_eps,
            true,
            emb_vb.pp("LayerNorm"),
        )?;
        let embeddings = DistilBertEmbeddings {
            word_embeddings,
            position_embeddings,
            layer_norm: emb_layer_norm,
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for n in 0..config.num_hidden_layers {
            let layer_vb = base_vb.pp(format!("transformer.layer.{n}"));
            let lora_layer_vb = lora_vb.pp(format!("layer.{n}"));

            let attn_vb = layer_vb.pp("attention");
            let attn_slot = LoraSlot {
                lora_layer_vb: &lora_layer_vb,
                layer_idx: n,
                lora: &self.lora,
            };
            let q_lin = attn_slot.build_in(
                &attn_vb,
                "q_lin",
                "attention.q_lin",
                config.hidden_size,
                config.hidden_size,
            )?;
            let k_lin = attn_slot.build_in(
                &attn_vb,
                "k_lin",
                "attention.k_lin",
                config.hidden_size,
                config.hidden_size,
            )?;
            let v_lin = attn_slot.build_in(
                &attn_vb,
                "v_lin",
                "attention.v_lin",
                config.hidden_size,
                config.hidden_size,
            )?;
            let out_lin = attn_slot.build_in(
                &attn_vb,
                "out_lin",
                "attention.out_lin",
                config.hidden_size,
                config.hidden_size,
            )?;

            let sa_layer_norm = LayerNorm::new(
                config.hidden_size,
                config.layer_norm_eps,
                true,
                layer_vb.pp("sa_layer_norm"),
            )?;

            let ffn_vb = layer_vb.pp("ffn");
            let ffn_slot = LoraSlot {
                lora_layer_vb: &lora_layer_vb,
                layer_idx: n,
                lora: &self.lora,
            };
            let lin1 = ffn_slot.build_in(
                &ffn_vb,
                "lin1",
                "ffn.lin1",
                config.hidden_size,
                config.intermediate_size,
            )?;
            let lin2 = ffn_slot.build_in(
                &ffn_vb,
                "lin2",
                "ffn.lin2",
                config.intermediate_size,
                config.hidden_size,
            )?;

            let output_layer_norm = LayerNorm::new(
                config.hidden_size,
                config.layer_norm_eps,
                true,
                layer_vb.pp("output_layer_norm"),
            )?;

            layers.push(DistilBertLayer {
                attention: DistilBertSelfAttention {
                    q_lin,
                    k_lin,
                    v_lin,
                    out_lin,
                    num_attention_heads: config.num_attention_heads,
                    attention_head_size,
                },
                sa_layer_norm,
                ffn: DistilBertFfn { lin1, lin2 },
                output_layer_norm,
            });
        }

        Ok(DistilBert {
            embeddings,
            layers,
            pooling: self.pooling,
            hidden_size: config.hidden_size,
            max_position_embeddings: config.max_position_embeddings,
        })
    }
}

/// Per-layer LoRA construction context. Holds the layer-scoped LoRA
/// VarBuilder, the layer index, and the call-site LoRA config so the
/// inner build calls stay narrow.
struct LoraSlot<'a, 'b> {
    lora_layer_vb: &'a VarBuilder<'b>,
    layer_idx: usize,
    lora: &'a LoraBuildConfig<'a>,
}

impl LoraSlot<'_, '_> {
    /// Construct a `MaybeLoraLinear` at the named module path. `module_name`
    /// is the short suffix fed to [`should_apply_lora`]; `module_path` is the
    /// parent-relative path used to address the LoRA A/B tensors inside
    /// `lora_layer_vb`.
    fn build_in(
        &self,
        parent_vb: &VarBuilder,
        module_name: &str,
        module_path: &str,
        in_features: usize,
        out_features: usize,
    ) -> Result<MaybeLoraLinear, EncoderError> {
        // The supplied `module_path` decomposes into `parent.child`; the
        // parent VarBuilder is already positioned at the parent, so the
        // base linear descends by the trailing segment only.
        let child_segment = module_path
            .rsplit_once('.')
            .map(|(_, child)| child)
            .unwrap_or(module_path);
        let linear = candle_nn::linear(in_features, out_features, parent_vb.pp(child_segment))?;

        if should_apply_lora(
            module_name,
            self.lora.target_modules,
            self.layer_idx,
            self.lora.layers_to_transform,
        ) {
            let rank = effective_rank(module_name, self.lora.lora_rank, self.lora.rank_pattern);
            let lora_linear = LoraLinear::new(
                linear,
                rank,
                self.lora.lora_alpha,
                self.lora.use_rslora,
                self.lora.init_mode,
                self.lora.lora_dropout,
                &self.lora_layer_vb.pp(module_path),
            )?;
            Ok(MaybeLoraLinear::Lora(lora_linear))
        } else {
            Ok(MaybeLoraLinear::Frozen(linear))
        }
    }
}
