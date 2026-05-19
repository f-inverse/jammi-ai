//! ModernBERT encoder.
//!
//! ModernBERT differences from classic BERT:
//! - Fused QKV projection `Wqkv` (`hidden * 3` by `hidden`).
//! - Output projection `Wo` (`hidden` by `hidden`).
//! - Rotary Position Embeddings (RoPE) applied to Q and K — no learned
//!   position-embedding table.
//! - GeGLU feed-forward: `Wi` packs gate+up (`intermediate * 2` by `hidden`),
//!   `mlp.Wo` projects back (`hidden` by `intermediate`).
//! - Pre-norm attention via `attn_norm`, except layer 0 where the embedding
//!   `norm` is the pre-norm (`attn_norm = None`).
//! - LayerNorm without a learned bias (matches the upstream
//!   `layer_norm_no_bias` configuration: mean-removing, weight-only affine).
//! - No token-type IDs.
//!
//! HuggingFace weight-key convention (prefix `model.`):
//! ```text
//! model.embeddings.tok_embeddings.weight
//! model.embeddings.norm.weight
//! model.layers.{n}.attn.Wqkv.weight
//! model.layers.{n}.attn.Wo.weight
//! model.layers.{n}.attn_norm.weight        // absent for layer 0
//! model.layers.{n}.mlp.Wi.weight
//! model.layers.{n}.mlp.Wo.weight
//! model.layers.{n}.mlp_norm.weight
//! model.final_norm.weight
//! ```

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{embedding, linear_no_bias, Embedding, VarBuilder, VarMap};
use jammi_lora::{effective_rank, should_apply_lora, LoraBuildConfig, LoraLinear, MaybeLoraLinear};

use crate::error::EncoderError;
use crate::layer_norm::LayerNorm;
use crate::mask::extended_attention_mask;
use crate::pooling::{pool_and_normalize, Pooling};

const DEFAULT_LAYER_NORM_EPS: f64 = 1e-5;
const DEFAULT_GLOBAL_ROPE_THETA: f64 = 160_000.0;
const DEFAULT_LOCAL_ROPE_THETA: f64 = 10_000.0;
const DEFAULT_LOCAL_ATTENTION: usize = 128;
const DEFAULT_GLOBAL_ATTN_EVERY_N_LAYERS: usize = 3;

fn default_layer_norm_eps() -> f64 {
    DEFAULT_LAYER_NORM_EPS
}
fn default_global_rope_theta() -> f64 {
    DEFAULT_GLOBAL_ROPE_THETA
}
fn default_local_rope_theta() -> f64 {
    DEFAULT_LOCAL_ROPE_THETA
}
fn default_local_attention() -> usize {
    DEFAULT_LOCAL_ATTENTION
}
fn default_global_attn_every_n_layers() -> usize {
    DEFAULT_GLOBAL_ATTN_EVERY_N_LAYERS
}

/// ModernBERT architecture configuration parsed from `config.json`.
///
/// Fields mirror the HuggingFace ModernBERT config schema. The
/// sliding-window-local-attention fields are accepted for round-trip
/// compatibility with stock checkpoints but are not exercised by the
/// forward pass — this port uses a single global RoPE for every layer,
/// which is exact for any checkpoint whose `global_attn_every_n_layers` is
/// `1` (i.e., every layer is a global-attention layer).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModernBertConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_global_rope_theta")]
    pub global_rope_theta: f64,
    #[serde(default = "default_local_rope_theta")]
    pub local_rope_theta: f64,
    #[serde(default = "default_local_attention")]
    pub local_attention: usize,
    #[serde(default = "default_global_attn_every_n_layers")]
    pub global_attn_every_n_layers: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// RoPE
// ─────────────────────────────────────────────────────────────────────────────

/// Precomputed RoPE cos/sin tables of shape `[max_seq_len, head_dim]`.
///
/// We duplicate the `half_dim` frequencies so the tables are usable with the
/// `rotate_half(x) = cat(-x[..,half:], x[..,:half])` formulation, which is
/// the variant the upstream ModernBERT implementation uses.
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(
        head_dim: usize,
        max_seq_len: usize,
        rope_base: f64,
        device: &Device,
    ) -> Result<Self, EncoderError> {
        let half = head_dim / 2;
        let mut cos_vec = Vec::with_capacity(max_seq_len * head_dim);
        let mut sin_vec = Vec::with_capacity(max_seq_len * head_dim);

        for pos in 0..max_seq_len {
            for _half_pass in 0..2 {
                for i in 0..half {
                    let theta = (pos as f64) * (rope_base.powf(-2.0 * i as f64 / head_dim as f64));
                    cos_vec.push(theta.cos() as f32);
                    sin_vec.push(theta.sin() as f32);
                }
            }
        }

        let cos = Tensor::from_vec(cos_vec, (max_seq_len, head_dim), device)?;
        let sin = Tensor::from_vec(sin_vec, (max_seq_len, head_dim), device)?;

        Ok(Self { cos, sin })
    }

    /// Apply RoPE to a `[batch, num_heads, seq, head_dim]` tensor.
    fn apply(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        let (_batch, _heads, seq, head_dim) = x.dims4()?;
        let half = head_dim / 2;
        let x_dtype = x.dtype();

        let cos = self
            .cos
            .i(..seq)?
            .to_dtype(x_dtype)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let sin = self
            .sin
            .i(..seq)?
            .to_dtype(x_dtype)?
            .unsqueeze(0)?
            .unsqueeze(0)?;

        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let neg_x2 = (x2 * -1.0f64)?;
        let rot_half = Tensor::cat(&[&neg_x2, &x1], D::Minus1)?;

        let cos_part = x.broadcast_mul(&cos)?;
        let sin_part = rot_half.broadcast_mul(&sin)?;
        Ok((cos_part + sin_part)?)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Attention
// ─────────────────────────────────────────────────────────────────────────────

struct ModernBertAttention {
    wqkv: MaybeLoraLinear,
    wo: MaybeLoraLinear,
    /// `None` for layer 0 — the embedding `norm` already pre-normalises the
    /// input there, so the layer holds an identity pre-norm.
    attn_norm: Option<LayerNorm>,
    rope: RotaryEmbedding,
    num_heads: usize,
    head_dim: usize,
}

impl ModernBertAttention {
    fn forward(&self, hidden: &Tensor, extended_mask: &Tensor) -> Result<Tensor, EncoderError> {
        let normed = match &self.attn_norm {
            Some(ln) => ln.forward(hidden)?,
            None => hidden.clone(),
        };
        let (batch, seq, _) = normed.dims3()?;
        let h = self.num_heads;
        let d = self.head_dim;

        let qkv = self.wqkv.forward(&normed)?;

        let q = qkv
            .narrow(D::Minus1, 0, h * d)?
            .reshape((batch, seq, h, d))?
            .transpose(1, 2)?;
        let k = qkv
            .narrow(D::Minus1, h * d, h * d)?
            .reshape((batch, seq, h, d))?
            .transpose(1, 2)?;
        // candle's matmul rejects non-contiguous batch layouts; Q and K become
        // contiguous as a side effect of the RoPE op chain, but V skips RoPE,
        // so make the transposed-V contiguity explicit here.
        let v = qkv
            .narrow(D::Minus1, 2 * h * d, h * d)?
            .reshape((batch, seq, h, d))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.rope.apply(&q)?;
        let k = self.rope.apply(&k)?;

        let scale = (d as f64).sqrt();
        let scores = q.matmul(&k.transpose(D::Minus1, D::Minus2)?)?;
        let scores = (scores / scale)?;
        let scores = scores.broadcast_add(extended_mask)?;

        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;

        let ctx = attn
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq, h * d))?;

        let out = self.wo.forward(&ctx)?;
        Ok((out + hidden)?)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GeGLU FFN
// ─────────────────────────────────────────────────────────────────────────────

struct ModernBertMlp {
    /// Packed gate+up projection. LoRA target name: `"Wi"`.
    wi: MaybeLoraLinear,
    /// Down projection. LoRA target name: `"mlp.Wo"` (kept namespaced so
    /// `ends_with("Wo")` targeting can distinguish it from the attention
    /// output projection when callers want both).
    wo: MaybeLoraLinear,
    mlp_norm: LayerNorm,
}

impl ModernBertMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        let normed = self.mlp_norm.forward(x)?;

        let up_gate = self.wi.forward(&normed)?;
        let intermediate = up_gate.dim(D::Minus1)? / 2;

        let gate = up_gate.narrow(D::Minus1, 0, intermediate)?;
        let up = up_gate.narrow(D::Minus1, intermediate, intermediate)?;

        let act = (gate.gelu_erf()? * up)?;
        let out = self.wo.forward(&act)?;

        Ok((out + x)?)
    }
}

struct ModernBertLayer {
    attention: ModernBertAttention,
    mlp: ModernBertMlp,
}

impl ModernBertLayer {
    fn forward(&self, hidden: &Tensor, extended_mask: &Tensor) -> Result<Tensor, EncoderError> {
        let after_attn = self.attention.forward(hidden, extended_mask)?;
        self.mlp.forward(&after_attn)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Encoder
// ─────────────────────────────────────────────────────────────────────────────

/// ModernBERT encoder with selectable LoRA adapters on attention and FFN
/// linears.
///
/// Construct via [`ModernBert::builder`]; see [`ModernBertBuilder`] for the
/// configurable surface.
pub struct ModernBert {
    word_embeddings: Embedding,
    emb_norm: LayerNorm,
    layers: Vec<ModernBertLayer>,
    final_norm: LayerNorm,
    pooling: Pooling,
    hidden_size: usize,
    max_position_embeddings: usize,
}

impl ModernBert {
    /// Start configuring a `ModernBert` instance.
    pub fn builder() -> ModernBertBuilder<'static> {
        ModernBertBuilder {
            pooling: Pooling::default(),
            lora: LoraBuildConfig::frozen(),
            backbone_dtype: DType::F32,
            adapter_file: None,
        }
    }

    /// Output dimensionality of the encoder.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Maximum sequence length the model supports (`max_position_embeddings`).
    pub fn max_seq_length(&self) -> usize {
        self.max_position_embeddings
    }

    /// Run the encoder and pool + L2-normalise the output, returning
    /// `[batch, hidden]`.
    pub fn forward(&self, input_ids: &Tensor, mask: &Tensor) -> Result<Tensor, EncoderError> {
        let hidden = self.forward_hidden(input_ids, mask)?;
        pool_and_normalize(&hidden, mask, self.pooling)
    }

    /// Run the encoder and return the raw last-layer hidden states
    /// `[batch, seq, hidden]`.
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

        let word_emb = self.word_embeddings.forward(input_ids)?;
        let mut hidden = self.emb_norm.forward(&word_emb)?;

        let extended = extended_attention_mask(mask)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &extended)?;
        }

        self.final_norm.forward(&hidden)
    }

    /// Borrowed references to every trainable LoRA tensor in the encoder.
    pub fn trainable_params(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.attention.wqkv.trainable_params());
            params.extend(layer.attention.wo.trainable_params());
            params.extend(layer.mlp.wi.trainable_params());
            params.extend(layer.mlp.wo.trainable_params());
        }
        params
    }

    /// CPU-side export of every LoRA `A` and `B` tensor, keyed by
    /// `layer.{n}.{site}.lora_{a|b}`.
    pub fn named_trainable_weights(&self) -> Result<HashMap<String, Tensor>, EncoderError> {
        let mut out = HashMap::new();
        for (n, layer) in self.layers.iter().enumerate() {
            out.extend(
                layer
                    .attention
                    .wqkv
                    .named_weights(&format!("layer.{n}.Wqkv"))?,
            );
            out.extend(layer.attention.wo.named_weights(&format!("layer.{n}.Wo"))?);
            out.extend(layer.mlp.wi.named_weights(&format!("layer.{n}.Wi"))?);
            out.extend(layer.mlp.wo.named_weights(&format!("layer.{n}.mlp.Wo"))?);
        }
        Ok(out)
    }

    /// Toggle training mode on every LoRA-augmented linear and every LayerNorm.
    /// ModernBERT's LayerNorms use the bias-free variant whose forward stays
    /// on the slow primitive-op path in both modes, but propagating the flag
    /// keeps the surface consistent with [`Bert`] and [`DistilBert`].
    pub fn set_training(&mut self, training: bool) {
        self.emb_norm.set_training(training);
        for layer in &mut self.layers {
            layer.attention.wqkv.set_training(training);
            layer.attention.wo.set_training(training);
            if let Some(attn_norm) = layer.attention.attn_norm.as_mut() {
                attn_norm.set_training(training);
            }
            layer.mlp.wi.set_training(training);
            layer.mlp.wo.set_training(training);
            layer.mlp.mlp_norm.set_training(training);
        }
        self.final_norm.set_training(training);
    }

    /// Restore LoRA `A`/`B` tensors from a `named_trainable_weights`-shaped map.
    /// Missing keys are silently skipped — see
    /// [`MaybeLoraLinear::load_weights`].
    pub fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<(), EncoderError> {
        for (n, layer) in self.layers.iter_mut().enumerate() {
            layer
                .attention
                .wqkv
                .load_weights(weights, &format!("layer.{n}.Wqkv"));
            layer
                .attention
                .wo
                .load_weights(weights, &format!("layer.{n}.Wo"));
            layer.mlp.wi.load_weights(weights, &format!("layer.{n}.Wi"));
            layer
                .mlp
                .wo
                .load_weights(weights, &format!("layer.{n}.mlp.Wo"));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Builder
// ─────────────────────────────────────────────────────────────────────────────

/// Builder for [`ModernBert`]. Mirrors `BertBuilder` so callers can swap
/// encoder families without touching their builder pipeline.
pub struct ModernBertBuilder<'a> {
    pooling: Pooling,
    lora: LoraBuildConfig<'a>,
    backbone_dtype: DType,
    adapter_file: Option<&'a Path>,
}

impl<'a> ModernBertBuilder<'a> {
    /// Select the sentence-embedding pooling strategy used by
    /// [`ModernBert::forward`].
    pub fn pooling(mut self, p: Pooling) -> Self {
        self.pooling = p;
        self
    }

    /// Provide a LoRA build configuration; defaults to
    /// [`LoraBuildConfig::frozen`].
    pub fn lora(mut self, l: LoraBuildConfig<'a>) -> Self {
        self.lora = l;
        self
    }

    /// Override the backbone dtype (default `F32`).
    pub fn backbone_dtype(mut self, d: DType) -> Self {
        self.backbone_dtype = d;
        self
    }

    /// Provide an optional path to a pre-trained LoRA adapter safetensors
    /// file. When `None`, LoRA tensors are initialised via the supplied
    /// [`VarMap`] at build time.
    pub fn adapter(mut self, p: Option<&'a Path>) -> Self {
        self.adapter_file = p;
        self
    }

    /// Load the backbone (and optional adapter) and assemble a [`ModernBert`].
    pub fn build(
        self,
        weights_paths: &[&Path],
        config: &ModernBertConfig,
        device: &Device,
        varmap: &VarMap,
    ) -> Result<ModernBert, EncoderError> {
        if config.num_attention_heads == 0 || config.hidden_size % config.num_attention_heads != 0 {
            return Err(EncoderError::Config(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                config.hidden_size, config.num_attention_heads
            )));
        }

        let frozen_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(weights_paths, self.backbone_dtype, device)?
        };
        let lora_vb = if let Some(adapter) = self.adapter_file {
            unsafe { VarBuilder::from_mmaped_safetensors(&[adapter], DType::F32, device)? }
        } else {
            VarBuilder::from_varmap(varmap, DType::F32, device)
        };

        let head_dim = config.hidden_size / config.num_attention_heads;

        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            frozen_vb.pp("model.embeddings.tok_embeddings"),
        )?;
        let emb_norm = LayerNorm::new(
            config.hidden_size,
            config.layer_norm_eps,
            false,
            frozen_vb.pp("model.embeddings.norm"),
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for n in 0..config.num_hidden_layers {
            let layer_vb = frozen_vb.pp(format!("model.layers.{n}"));
            let lora_layer_vb = lora_vb.pp(format!("layer.{n}"));
            let site = LoraSite {
                layer_vb: &layer_vb,
                lora_layer_vb: &lora_layer_vb,
                layer_idx: n,
                lora: self.lora,
            };

            let wqkv = site.build(
                "Wqkv",
                "attn.Wqkv",
                config.hidden_size,
                config.hidden_size * 3,
            )?;
            let wo = site.build("Wo", "attn.Wo", config.hidden_size, config.hidden_size)?;

            let attn_norm = if n == 0 {
                None
            } else {
                Some(LayerNorm::new(
                    config.hidden_size,
                    config.layer_norm_eps,
                    false,
                    layer_vb.pp("attn_norm"),
                )?)
            };

            let rope = RotaryEmbedding::new(
                head_dim,
                config.max_position_embeddings,
                config.global_rope_theta,
                device,
            )?;

            let wi = site.build(
                "Wi",
                "mlp.Wi",
                config.hidden_size,
                config.intermediate_size * 2,
            )?;
            let mlp_wo = site.build(
                "mlp.Wo",
                "mlp.Wo",
                config.intermediate_size,
                config.hidden_size,
            )?;
            let mlp_norm = LayerNorm::new(
                config.hidden_size,
                config.layer_norm_eps,
                false,
                layer_vb.pp("mlp_norm"),
            )?;

            layers.push(ModernBertLayer {
                attention: ModernBertAttention {
                    wqkv,
                    wo,
                    attn_norm,
                    rope,
                    num_heads: config.num_attention_heads,
                    head_dim,
                },
                mlp: ModernBertMlp {
                    wi,
                    wo: mlp_wo,
                    mlp_norm,
                },
            });
        }

        let final_norm = LayerNorm::new(
            config.hidden_size,
            config.layer_norm_eps,
            false,
            frozen_vb.pp("model.final_norm"),
        )?;

        Ok(ModernBert {
            word_embeddings,
            emb_norm,
            layers,
            final_norm,
            pooling: self.pooling,
            hidden_size: config.hidden_size,
            max_position_embeddings: config.max_position_embeddings,
        })
    }
}

/// Per-layer scratchpad that captures the shared inputs of every LoRA-site
/// load — the frozen and adapter VarBuilders, the layer index, and the
/// caller's `LoraBuildConfig` — so the per-site call only varies in the four
/// values that actually differ between sites.
struct LoraSite<'a, 'b> {
    layer_vb: &'a VarBuilder<'b>,
    lora_layer_vb: &'a VarBuilder<'b>,
    layer_idx: usize,
    lora: LoraBuildConfig<'b>,
}

impl<'a, 'b> LoraSite<'a, 'b> {
    fn build(
        &self,
        target_name: &str,
        safetensors_sub: &str,
        in_features: usize,
        out_features: usize,
    ) -> Result<MaybeLoraLinear, EncoderError> {
        let frozen = linear_no_bias(in_features, out_features, self.layer_vb.pp(safetensors_sub))?;
        if should_apply_lora(
            target_name,
            self.lora.target_modules,
            self.layer_idx,
            self.lora.layers_to_transform,
        ) {
            let rank = effective_rank(target_name, self.lora.lora_rank, self.lora.rank_pattern);
            let lora_linear = LoraLinear::new(
                frozen,
                rank,
                self.lora.lora_alpha,
                self.lora.use_rslora,
                self.lora.init_mode,
                self.lora.lora_dropout,
                &self.lora_layer_vb.pp(target_name),
            )?;
            Ok(MaybeLoraLinear::Lora(lora_linear))
        } else {
            Ok(MaybeLoraLinear::Frozen(frozen))
        }
    }
}
