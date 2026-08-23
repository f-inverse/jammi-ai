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
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{embedding, linear_no_bias, Embedding, VarBuilder, VarMap};
use jammi_lora::{effective_rank, should_apply_lora, LoraBuildConfig, LoraLinear, MaybeLoraLinear};

use crate::error::EncoderError;
use crate::layer_norm::LayerNorm;
use crate::mask::{extended_attention_mask, sliding_window_mask};
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
/// Fields mirror the HuggingFace ModernBERT config schema, including the
/// sliding-window-local-attention set, which the forward pass honours:
/// `global_attn_every_n_layers` selects which layers are global, and a local
/// layer attends within `local_attention / 2` positions either side using
/// `local_rope_theta` as its RoPE base. See [`ModernBertConfig::is_local_layer`].
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

impl ModernBertConfig {
    /// Whether layer `idx` uses sliding-window local attention.
    ///
    /// Panics if `global_attn_every_n_layers` is 0; [`ModernBertBuilder::build`]
    /// refuses such a config before any layer is constructed, so this is
    /// unreachable from a loaded model.
    ///
    /// ModernBERT's rule, matching upstream
    /// (`layer_types[i] = "sliding_attention" if i % global_attn_every_n_layers
    /// else "full_attention"`): layer 0 and every `global_attn_every_n_layers`-th
    /// layer thereafter are global, and the rest are local. A checkpoint with
    /// `global_attn_every_n_layers == 1` is therefore all-global — which is why
    /// a single-layer fixture cannot distinguish an implementation that honours
    /// the window from one that ignores it.
    pub fn is_local_layer(&self, idx: usize) -> bool {
        !idx.is_multiple_of(self.global_attn_every_n_layers)
    }

    /// Half-width of the sliding window: a local layer's query at position `i`
    /// attends to keys `j` with `|i - j| <= half_window`. Upstream stores the
    /// full width and halves it (`sliding_window = local_attention // 2`).
    pub fn half_window(&self) -> usize {
        self.local_attention / 2
    }
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
    /// The RoPE table for this layer's attention type. Shared, because a model
    /// has exactly two tables (global and local) however many layers it has.
    rope: Arc<RotaryEmbedding>,
    /// `true` when this layer attends within a sliding window rather than over
    /// the whole sequence. The band itself is built once per forward and passed
    /// in, since it depends only on the sequence length.
    is_local: bool,
    num_heads: usize,
    head_dim: usize,
}

impl ModernBertAttention {
    /// `local_band` is the `[1, 1, seq, seq]` sliding-window mask, supplied
    /// whenever the model has any local layer. A global layer ignores it.
    fn forward(
        &self,
        hidden: &Tensor,
        extended_mask: &Tensor,
        local_band: Option<&Tensor>,
    ) -> Result<Tensor, EncoderError> {
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
        let v = qkv
            .narrow(D::Minus1, 2 * h * d, h * d)?
            .reshape((batch, seq, h, d))?
            .transpose(1, 2)?;

        let q = self.rope.apply(&q)?;
        let k = self.rope.apply(&k)?;

        let scale = (d as f64).sqrt();
        let scores = crate::contiguous_matmul(&q, &k.transpose(D::Minus1, D::Minus2)?)?;
        let scores = (scores / scale)?;
        // The additive mask is always built in F32 (see `extended_attention_mask`);
        // cast to the scores' dtype so a F16/BF16 backbone can add it (a no-op
        // when scores are already F32).
        let extended_mask = extended_mask.to_dtype(scores.dtype())?;
        let scores = scores.broadcast_add(&extended_mask)?;

        // A local layer additionally masks everything outside its band. Added
        // straight onto the scores rather than pre-combined with the padding
        // mask, so the two never materialise a joint `[batch, heads, seq, seq]`
        // tensor: each broadcasts from its own smaller shape.
        let scores = match (self.is_local, local_band) {
            (true, Some(band)) => scores.broadcast_add(&band.to_dtype(scores.dtype())?)?,
            (true, None) => {
                return Err(EncoderError::Config(
                    "local-attention layer reached without a sliding-window band".into(),
                ))
            }
            (false, _) => scores,
        };

        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;

        let ctx = crate::contiguous_matmul(&attn, &v)?
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
    /// Passes `local_band` through to attention; whether it is consulted is the
    /// attention's own per-layer property.
    fn forward(
        &self,
        hidden: &Tensor,
        extended_mask: &Tensor,
        local_band: Option<&Tensor>,
    ) -> Result<Tensor, EncoderError> {
        let after_attn = self.attention.forward(hidden, extended_mask, local_band)?;
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
    /// Half-width of the sliding window, `Some` only when the model actually
    /// has a local layer. `None` means every layer is global and no band is
    /// built.
    local_half_window: Option<usize>,
    /// Sliding-window bands, keyed by sequence length.
    ///
    /// The band is a pure function of `(seq, half_window, device)` and constant
    /// for the life of the model, but the sequence length varies per batch
    /// (padding is batch-longest), so it is memoised per length rather than
    /// built once. Without this, every forward allocated and uploaded a
    /// `seq * seq` host buffer — 268 MB per forward at this family's
    /// `max_position_embeddings` of 8192 — which is the same host-generated
    /// per-forward mask cost recorded as esc-032 for LoRA dropout.
    ///
    /// A `Mutex` rather than a `RefCell`: the model is held across threads.
    band_cache: Mutex<HashMap<usize, Tensor>>,
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
        // Built once per forward, not per layer: the band depends only on the
        // sequence length and the window, so every local layer shares it.
        let local_band = match self.local_half_window {
            None => None,
            Some(half) => Some(self.sliding_band(seq, half, input_ids.device())?),
        };
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &extended, local_band.as_ref())?;
        }

        self.final_norm.forward(&hidden)
    }

    /// The sliding-window band for `seq`, built once per length and reused.
    fn sliding_band(
        &self,
        seq: usize,
        half: usize,
        device: &Device,
    ) -> Result<Tensor, EncoderError> {
        let mut cache = self
            .band_cache
            .lock()
            .map_err(|_| EncoderError::Config("sliding-window band cache poisoned".into()))?;
        if let Some(band) = cache.get(&seq) {
            return Ok(band.clone());
        }
        let band = sliding_window_mask(seq, half, device)?;
        cache.insert(seq, band.clone());
        Ok(band)
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
    /// keeps the surface consistent with [`crate::Bert`] and [`crate::DistilBert`].
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

    /// Per-site dropout-stream positions keyed `{site}.dropout`, over the same
    /// site names [`Self::named_trainable_weights`] uses — the resume state for
    /// the adapter's dropout.
    pub fn dropout_positions(&self) -> Result<HashMap<String, u64>, EncoderError> {
        let mut out = HashMap::new();
        for (n, layer) in self.layers.iter().enumerate() {
            for (site, lin) in modern_lora_sites(layer) {
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
            for (site, lin) in modern_lora_sites(layer) {
                lin.restore_dropout_position(&format!("layer.{n}.{site}"), positions)?;
            }
        }
        Ok(())
    }
}

/// The four LoRA-wrappable linear sites of one ModernBERT layer paired with their
/// `named_trainable_weights` site names.
fn modern_lora_sites(layer: &ModernBertLayer) -> [(&'static str, &MaybeLoraLinear); 4] {
    [
        ("Wqkv", &layer.attention.wqkv),
        ("Wo", &layer.attention.wo),
        ("Wi", &layer.mlp.wi),
        ("mlp.Wo", &layer.mlp.wo),
    ]
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
        // Refuse a config this port cannot honour rather than reinterpreting it.
        // Upstream raises on `i % 0`; silently treating it as all-global would
        // be the same silent-wrong-function class this sliding-window support
        // exists to remove.
        if config.global_attn_every_n_layers == 0 {
            return Err(EncoderError::Config(
                "global_attn_every_n_layers must be > 0 (1 = every layer global)".into(),
            ));
        }
        if config.num_attention_heads == 0
            || !config
                .hidden_size
                .is_multiple_of(config.num_attention_heads)
        {
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

        // Exactly two RoPE tables per model, shared by every layer of the
        // matching attention type. Building one per layer would allocate
        // `num_hidden_layers` identical tables.
        let global_rope = Arc::new(RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.global_rope_theta,
            device,
        )?);
        let local_rope = Arc::new(RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.local_rope_theta,
            device,
        )?);

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
                varmap,
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

            let is_local = config.is_local_layer(n);
            let rope = if is_local {
                Arc::clone(&local_rope)
            } else {
                Arc::clone(&global_rope)
            };

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
                    is_local,
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
            local_half_window: (0..config.num_hidden_layers)
                .any(|n| config.is_local_layer(n))
                .then(|| config.half_window()),
            band_cache: Mutex::new(HashMap::new()),
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
    /// The trainable `VarMap` the seeded LoRA A/B tensors are registered into.
    varmap: &'a VarMap,
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
                self.lora.seed,
                self.varmap,
                &self.lora_layer_vb.pp(target_name),
            )?;
            Ok(MaybeLoraLinear::Lora(lora_linear))
        } else {
            Ok(MaybeLoraLinear::Frozen(frozen))
        }
    }
}
