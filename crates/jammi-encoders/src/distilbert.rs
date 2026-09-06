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
use jammi_kernels::admission::PredicateOutcome;
use jammi_kernels::ops::FullyMaskedPolicy;
use jammi_lora::{
    effective_rank, should_apply_lora, FrozenBase, LoraBuildConfig, LoraLinear, MaybeLoraLinear,
};

use crate::activations;
use crate::attention_cascade::{self, FusedAttentionMasks, RopeCtx, TrainingMaskInputs};
use crate::error::EncoderError;
use crate::frozen_weight_source::{validate_frozen_base_geometry, FrozenWeightLookup};
use crate::layer_norm::LayerNorm;
use crate::modernbert::FlashDecision;
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
    /// Same status as `crate::bert::BertSelfAttention::training`.
    training: bool,
    /// Same status as `crate::bert::BertSelfAttention::rope_placeholder`.
    rope_placeholder: Tensor,
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

        let scores =
            (crate::contiguous_matmul(&q, &k.t()?)? / (self.attention_head_size as f64).sqrt())?;
        // The additive mask is always built in F32 (see `extended_attention_mask`);
        // cast to the scores' dtype so a F16/BF16 backbone can add it (a no-op
        // when scores are already F32).
        let extended_mask = extended_mask.to_dtype(scores.dtype())?;
        let scores = scores.broadcast_add(&extended_mask)?;
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;

        let context = crate::contiguous_matmul(&probs, &v)?;
        // Re-materialise contiguously after the transpose, then collapse the
        // head/dim trailing axes back into hidden.
        let context = context.transpose(1, 2)?.contiguous()?;
        let context = context.flatten_from(D::Minus2)?;
        Ok(self.out_lin.forward(&context)?)
    }

    /// Training's arm — same shape as `crate::bert::BertSelfAttention::forward_training`
    /// (issue #462): `qkv = Tensor::cat(&[q, k, v], D::Minus1)`, `rope`
    /// disabled, `window: None` (no sliding-window concept), `policy:
    /// Propagate`, `flash` always `Declined`. `out_lin` (DistilBERT's own
    /// attention output projection — BERT folds this into `BertSelfOutput`
    /// instead) is applied to the cascade's `[batch, seq, hidden]` output,
    /// exactly where [`Self::forward`] applies it to the eager context.
    fn forward_training(
        &self,
        hidden: &Tensor,
        extended_mask: &Tensor,
        fused: &FusedAttentionMasks,
        flash: &FlashDecision,
    ) -> Result<Tensor, EncoderError> {
        debug_assert!(
            self.training,
            "DistilBertSelfAttention::forward_training called outside training mode"
        );
        let q = self.q_lin.forward(hidden)?;
        let k = self.k_lin.forward(hidden)?;
        let v = self.v_lin.forward(hidden)?;
        let qkv = Tensor::cat(&[&q, &k, &v], D::Minus1)?;
        let (batch, seq, _) = hidden.dims3()?;
        let h = self.num_attention_heads;
        let d = self.attention_head_size;
        let masks = TrainingMaskInputs {
            extended: extended_mask,
            local_band: None,
            fused: Some(fused),
        };
        let rope = RopeCtx {
            pack: &self.rope_placeholder,
            enabled: false,
            apply: None,
        };
        let context = attention_cascade::training_attention_cascade(
            &qkv,
            batch,
            seq,
            h,
            d,
            masks,
            flash,
            &rope,
            None,
            FullyMaskedPolicy::Propagate,
            |_admission| {
                Err(EncoderError::Config(
                    "attention_block_flash dispatched Fused but DistilBERT always supplies \
                     FlashDecision::Declined -- unreachable: admit_cascade can only report \
                     Fused when flash.outcome() is Holds, which Declined can never be"
                        .into(),
                ))
            },
        )?;
        Ok(self.out_lin.forward(&context)?)
    }
}

struct DistilBertFfn {
    lin1: MaybeLoraLinear,
    lin2: MaybeLoraLinear,
    /// Same status as `crate::bert::BertIntermediate::training`.
    training: bool,
}

impl DistilBertFfn {
    fn forward(&self, hidden: &Tensor) -> Result<Tensor, EncoderError> {
        let mid = self.lin1.forward(hidden)?;
        let activated = activations::gelu_erf(&mid, self.training)?;
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

    fn forward_training(
        &self,
        hidden: &Tensor,
        extended_mask: &Tensor,
        fused: &FusedAttentionMasks,
        flash: &FlashDecision,
    ) -> Result<Tensor, EncoderError> {
        let attn_out = self
            .attention
            .forward_training(hidden, extended_mask, fused, flash)?;
        let attn_residual = (attn_out + hidden)?;
        let attn_normed = self.sa_layer_norm.forward(&attn_residual)?;

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
    /// Same status as `crate::bert::Bert::training`.
    training: bool,
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
            weight_source: None,
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
        if self.training {
            // DistilBERT has no local-attention layers, so `local_band_f32`
            // is always `None` — see `FusedAttentionMasks`'s doc.
            let fused = FusedAttentionMasks::build(&extended, None, hidden.dtype())?;
            let flash = FlashDecision::Declined {
                outcome: PredicateOutcome::CapabilityMiss,
                reason: "flash_transport_not_wired",
            };
            for layer in &self.layers {
                hidden = layer.forward_training(&hidden, &extended, &fused, &flash)?;
            }
        } else {
            for layer in &self.layers {
                hidden = layer.forward(&hidden, &extended)?;
            }
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
        self.training = training;
        self.embeddings.layer_norm.set_training(training);
        for layer in &mut self.layers {
            layer.attention.training = training;
            layer.attention.q_lin.set_training(training);
            layer.attention.k_lin.set_training(training);
            layer.attention.v_lin.set_training(training);
            layer.attention.out_lin.set_training(training);
            layer.sa_layer_norm.set_training(training);
            layer.ffn.training = training;
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

    /// Per-site dropout-stream positions keyed `{site}.dropout`, over the same
    /// site names [`Self::named_trainable_weights`] uses — the resume state for
    /// the adapter's dropout.
    pub fn dropout_positions(&self) -> Result<HashMap<String, u64>, EncoderError> {
        let mut out = HashMap::new();
        for (n, layer) in self.layers.iter().enumerate() {
            for (site, lin) in distil_lora_sites(layer) {
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
            for (site, lin) in distil_lora_sites(layer) {
                lin.restore_dropout_position(&format!("layer.{n}.{site}"), positions)?;
            }
        }
        Ok(())
    }
}

/// The six LoRA-wrappable linear sites of one DistilBERT layer paired with their
/// `named_trainable_weights` site names.
fn distil_lora_sites(layer: &DistilBertLayer) -> [(&'static str, &MaybeLoraLinear); 6] {
    [
        ("q_lin", &layer.attention.q_lin),
        ("k_lin", &layer.attention.k_lin),
        ("v_lin", &layer.attention.v_lin),
        ("out_lin", &layer.attention.out_lin),
        ("lin1", &layer.ffn.lin1),
        ("lin2", &layer.ffn.lin2),
    ]
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
    /// The wave-3 GGUF-quantized-weight construction seam — see
    /// [`FrozenWeightLookup`]'s own module doc. `None` by default
    /// ([`DistilBert::builder`]); every EXISTING call site that never calls
    /// [`Self::weight_source`] gets byte-identical Dense-only behavior.
    weight_source: Option<&'a FrozenWeightLookup<'a>>,
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

    /// Supply a per-tensor-name GGUF-quantized-weight override — see
    /// [`FrozenWeightLookup`]'s own module doc. Defaulted: a builder that
    /// never calls this stays byte-identical to every prior release (Dense
    /// weights, loaded from `weights_paths`, everywhere).
    pub fn weight_source(mut self, w: &'a FrozenWeightLookup<'a>) -> Self {
        self.weight_source = Some(w);
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
        if config.num_attention_heads == 0
            || !config
                .hidden_size
                .is_multiple_of(config.num_attention_heads)
        {
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
                varmap,
                weight_source: self.weight_source,
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
                varmap,
                weight_source: self.weight_source,
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

            // The attention cascade's `rope_pack` placeholder (issue #462)
            // — see `crate::bert`'s identical construction site for why
            // this literal shape is safe regardless of `attention_head_size`.
            let rope_placeholder = Tensor::zeros((2, 1, 1, 64), DType::F32, device)?;

            layers.push(DistilBertLayer {
                attention: DistilBertSelfAttention {
                    q_lin,
                    k_lin,
                    v_lin,
                    out_lin,
                    num_attention_heads: config.num_attention_heads,
                    attention_head_size,
                    training: false,
                    rope_placeholder,
                },
                sa_layer_norm,
                ffn: DistilBertFfn {
                    lin1,
                    lin2,
                    training: false,
                },
                output_layer_norm,
            });
        }

        Ok(DistilBert {
            embeddings,
            layers,
            pooling: self.pooling,
            hidden_size: config.hidden_size,
            max_position_embeddings: config.max_position_embeddings,
            training: false,
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
    /// The trainable `VarMap` the seeded LoRA A/B tensors are registered into.
    varmap: &'a VarMap,
    /// The wave-3 GGUF-quantized-weight construction seam — see
    /// [`crate::FrozenWeightLookup`]'s own module doc. `None` at every
    /// EXISTING call site (byte-identical to every prior release).
    weight_source: Option<&'a FrozenWeightLookup<'a>>,
}

impl LoraSlot<'_, '_> {
    /// Construct a `MaybeLoraLinear` at the named module path. `module_name`
    /// is the short suffix fed to [`should_apply_lora`]; `module_path` is the
    /// parent-relative path used to address the LoRA A/B tensors inside
    /// `lora_layer_vb`.
    ///
    /// A `weight_source` HIT is geometry-checked
    /// (`validate_frozen_base_geometry`) against the `config.json`-derived
    /// `(in_features, out_features)` this call site expects — a GGUF whose
    /// per-site geometry disagrees with `config.json` fails HERE, at load,
    /// rather than surfacing as a confidently-wrong-shaped matmul at first
    /// inference.
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
        let child_vb = parent_vb.pp(child_segment);

        // Wave-3 seam (module doc): consult `weight_source` first, falling
        // back to the ORIGINAL Dense `linear(..)` load when unset or missed.
        let base = if let Some(lookup) = self.weight_source {
            let site = child_vb.prefix();
            match lookup(&site)? {
                Some(base) => {
                    validate_frozen_base_geometry(&site, &base, in_features, out_features)?;
                    base
                }
                None => FrozenBase::Dense(candle_nn::linear(in_features, out_features, child_vb)?),
            }
        } else {
            FrozenBase::Dense(candle_nn::linear(in_features, out_features, child_vb)?)
        };

        if should_apply_lora(
            module_name,
            self.lora.target_modules,
            self.layer_idx,
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
                &self.lora_layer_vb.pp(module_path),
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

    fn seeded_linear(rows: usize, cols: usize, phase: f32, device: &Device) -> candle_nn::Linear {
        let v: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 + phase) * 0.0151).sin() * 0.2)
            .collect();
        candle_nn::Linear::new(Tensor::from_vec(v, (rows, cols), device).unwrap(), None)
    }

    fn declined_flash() -> FlashDecision {
        FlashDecision::Declined {
            outcome: PredicateOutcome::CapabilityMiss,
            reason: "test_stub_flash_declined",
        }
    }

    fn self_attention_fixture(h: usize, d: usize, device: &Device) -> DistilBertSelfAttention {
        let hd = h * d;
        DistilBertSelfAttention {
            q_lin: MaybeLoraLinear::Frozen(FrozenBase::Dense(seeded_linear(hd, hd, 1.0, device))),
            k_lin: MaybeLoraLinear::Frozen(FrozenBase::Dense(seeded_linear(hd, hd, 2.0, device))),
            v_lin: MaybeLoraLinear::Frozen(FrozenBase::Dense(seeded_linear(hd, hd, 3.0, device))),
            out_lin: MaybeLoraLinear::Frozen(FrozenBase::Dense(seeded_linear(hd, hd, 4.0, device))),
            num_attention_heads: h,
            attention_head_size: d,
            training: true,
            rope_placeholder: Tensor::zeros((2, 1, 1, 64), DType::F32, device).unwrap(),
        }
    }

    /// Same shape as `crate::bert::tests::bert_head64_fused_attention_matches_eager_composition_within_tolerance`
    /// (contract R6', tol `1e-4`) — DistilBERT's own `out_lin` is applied to
    /// BOTH sides identically, so it drops out of the comparison and this
    /// still isolates the cascade's own fused-vs-eager numerics.
    #[test]
    fn distilbert_head64_fused_attention_matches_eager_composition_within_tolerance() {
        let _guard = ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let (b, s, h) = (2usize, 5usize, 2usize);
        let d = ATTENTION_BLOCK_HEAD_DIM;
        let attn = self_attention_fixture(h, d, &device);
        let hidden_v: Vec<f32> = (0..b * s * h * d)
            .map(|i| ((i as f32) * 0.033).sin() * 0.4)
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

        let q = attn.q_lin.forward(&hidden).unwrap();
        let k = attn.k_lin.forward(&hidden).unwrap();
        let v = attn.v_lin.forward(&hidden).unwrap();
        let qkv = Tensor::cat(&[&q, &k, &v], D::Minus1).unwrap();
        let rope = RopeCtx {
            pack: &attn.rope_placeholder,
            enabled: false,
            apply: None,
        };
        let ctx = attention_cascade::forward_eager_training_attention_composition(
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
        let out_eager = attn.out_lin.forward(&ctx).unwrap();

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

    /// Same shape as `crate::bert::tests::bert_head64_all_padding_row_propagate_fused_matches_eager_within_tolerance`.
    #[test]
    fn distilbert_head64_all_padding_row_propagate_fused_matches_eager_within_tolerance() {
        let _guard = ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let (b, s, h) = (2usize, 4usize, 1usize);
        let d = ATTENTION_BLOCK_HEAD_DIM;
        let attn = self_attention_fixture(h, d, &device);
        let hidden_v: Vec<f32> = (0..b * s * h * d)
            .map(|i| ((i as f32) * 0.027).cos() * 0.3)
            .collect();
        let hidden = Tensor::from_vec(hidden_v, (b, s, h * d), &device).unwrap();
        let mask = Tensor::from_slice(
            &[1.0f32, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            (b, 1, 1, s),
            &device,
        )
        .unwrap();
        let extended = mask.affine(-10_000.0_f64, 10_000.0).unwrap();
        let fused_masks = FusedAttentionMasks::build(&extended, None, DType::F32).unwrap();
        let flash = declined_flash();

        let out_fused = attn
            .forward_training(&hidden, &extended, &fused_masks, &flash)
            .expect("fused training forward on an all-padding row");

        let q = attn.q_lin.forward(&hidden).unwrap();
        let k = attn.k_lin.forward(&hidden).unwrap();
        let v = attn.v_lin.forward(&hidden).unwrap();
        let qkv = Tensor::cat(&[&q, &k, &v], D::Minus1).unwrap();
        let rope = RopeCtx {
            pack: &attn.rope_placeholder,
            enabled: false,
            apply: None,
        };
        let ctx = attention_cascade::forward_eager_training_attention_composition(
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
        let out_eager = attn.out_lin.forward(&ctx).unwrap();

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
    }

    /// Same shape as `crate::bert::tests::bert_head64_fused_attention_lora_gradients_are_finite_and_nonzero`.
    #[test]
    fn distilbert_head64_fused_attention_lora_gradients_are_finite_and_nonzero() {
        let _guard = ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let (b, s, h) = (2usize, 5usize, 2usize);
        let d = ATTENTION_BLOCK_HEAD_DIM;
        let hd = h * d;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let q_lin = LoraLinear::new_with_base(
            FrozenBase::Dense(seeded_linear(hd, hd, 1.0, &device)),
            4,
            8.0,
            false,
            LoraInitMode::Gaussian,
            None,
            21,
            &varmap,
            &vb.pp("q_lin"),
        )
        .unwrap();
        let v_lin = LoraLinear::new_with_base(
            FrozenBase::Dense(seeded_linear(hd, hd, 3.0, &device)),
            4,
            8.0,
            false,
            LoraInitMode::Gaussian,
            None,
            23,
            &varmap,
            &vb.pp("v_lin"),
        )
        .unwrap();
        let attn = DistilBertSelfAttention {
            q_lin: MaybeLoraLinear::Lora(q_lin),
            k_lin: MaybeLoraLinear::Frozen(FrozenBase::Dense(seeded_linear(hd, hd, 2.0, &device))),
            v_lin: MaybeLoraLinear::Lora(v_lin),
            out_lin: MaybeLoraLinear::Frozen(FrozenBase::Dense(seeded_linear(
                hd, hd, 4.0, &device,
            ))),
            num_attention_heads: h,
            attention_head_size: d,
            training: true,
            rope_placeholder: Tensor::zeros((2, 1, 1, 64), DType::F32, &device).unwrap(),
        };

        let hidden_v: Vec<f32> = (0..b * s * hd)
            .map(|i| ((i as f32) * 0.023).sin() * 0.3)
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

        let loss = out.sum_all().unwrap();
        let grads = loss
            .backward()
            .expect("backward through the fused attention block");

        for (name, lin) in [("q_lin", &attn.q_lin), ("v_lin", &attn.v_lin)] {
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

    /// Same shape as
    /// `crate::bert::tests::bert_head64_fused_attention_lora_gradients_match_eager_within_tolerance`
    /// (contract fix-round item 2, precedent
    /// `crate::modernbert::tests::fused_attention_block_matches_eager_lora_gradients_at_production_seq_on_head64`,
    /// tol `1e-4`) — DistilBERT's own `out_lin` is applied to BOTH sides
    /// identically before the loss's `dy` multiply, so it drops out of the
    /// comparison and this still isolates the cascade's own fused-vs-eager
    /// LoRA-gradient numerics, not merely finite/non-zero (that weaker
    /// check is
    /// [`distilbert_head64_fused_attention_lora_gradients_are_finite_and_nonzero`],
    /// kept alongside as its own, narrower oracle).
    #[test]
    fn distilbert_head64_fused_attention_lora_gradients_match_eager_within_tolerance() {
        let _guard = ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let (b, s, h) = (2usize, 5usize, 2usize);
        let d = ATTENTION_BLOCK_HEAD_DIM;
        let hd = h * d;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let q_lin = LoraLinear::new_with_base(
            FrozenBase::Dense(seeded_linear(hd, hd, 1.0, &device)),
            4,
            8.0,
            false,
            LoraInitMode::Gaussian,
            None,
            21,
            &varmap,
            &vb.pp("q_lin"),
        )
        .unwrap();
        let v_lin = LoraLinear::new_with_base(
            FrozenBase::Dense(seeded_linear(hd, hd, 3.0, &device)),
            4,
            8.0,
            false,
            LoraInitMode::Gaussian,
            None,
            23,
            &varmap,
            &vb.pp("v_lin"),
        )
        .unwrap();
        let attn = DistilBertSelfAttention {
            q_lin: MaybeLoraLinear::Lora(q_lin),
            k_lin: MaybeLoraLinear::Frozen(FrozenBase::Dense(seeded_linear(hd, hd, 2.0, &device))),
            v_lin: MaybeLoraLinear::Lora(v_lin),
            out_lin: MaybeLoraLinear::Frozen(FrozenBase::Dense(seeded_linear(
                hd, hd, 4.0, &device,
            ))),
            num_attention_heads: h,
            attention_head_size: d,
            training: true,
            rope_placeholder: Tensor::zeros((2, 1, 1, 64), DType::F32, &device).unwrap(),
        };

        let hidden_v: Vec<f32> = (0..b * s * hd)
            .map(|i| ((i as f32) * 0.023).sin() * 0.3)
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
        let loss_fused = (&out_fused * &dy).unwrap().sum_all().unwrap();
        let grads_fused = loss_fused
            .backward()
            .expect("backward through the fused attention block");

        let q = attn.q_lin.forward(&hidden).unwrap();
        let k = attn.k_lin.forward(&hidden).unwrap();
        let v = attn.v_lin.forward(&hidden).unwrap();
        let qkv = Tensor::cat(&[&q, &k, &v], D::Minus1).unwrap();
        let rope = RopeCtx {
            pack: &attn.rope_placeholder,
            enabled: false,
            apply: None,
        };
        let ctx = attention_cascade::forward_eager_training_attention_composition(
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
        let out_eager = attn.out_lin.forward(&ctx).unwrap();
        let loss_eager = (&out_eager * &dy).unwrap().sum_all().unwrap();
        let grads_eager = loss_eager
            .backward()
            .expect("backward through the eager attention composition");

        const TOL: f32 = 1e-4;
        for (name, lin) in [("q_lin", &attn.q_lin), ("v_lin", &attn.v_lin)] {
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

    /// Same shape as `crate::bert::tests::bert_strict_mode_on_a_refused_domain_is_a_typed_error_in_a_fresh_process`.
    #[test]
    fn distilbert_strict_mode_on_a_refused_domain_is_a_typed_error_in_a_fresh_process() {
        let exe = std::env::current_exe().expect("test binary path");
        let output = std::process::Command::new(exe)
            .args([
                "distilbert::tests::strict_mode_child_process_body",
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
