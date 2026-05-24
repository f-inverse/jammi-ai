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

use candle_core::{IndexOp, Module, Tensor, D};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, VarBuilder};

use crate::error::EncoderError;

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

/// QuickGelu activation: `x * sigmoid(1.702 * x)`. OpenCLIP uses this
/// in the text and vision MLPs (not the standard erf-based GELU).
fn quick_gelu(xs: &Tensor) -> Result<Tensor, EncoderError> {
    Ok((xs * candle_nn::ops::sigmoid(&(xs * 1.702f64)?)?)?)
}

/// Multi-head self-attention with fused QKV projection (OpenCLIP layout:
/// `in_proj_weight`/`in_proj_bias` plus an `out_proj` sub-module).
struct MultiHeadAttention {
    in_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    fn load(vb: VarBuilder, width: usize, num_heads: usize) -> Result<Self, EncoderError> {
        let head_dim = width / num_heads;
        let in_proj_weight = vb.get((width * 3, width), "in_proj_weight")?;
        let in_proj_bias = vb.get(width * 3, "in_proj_bias")?;
        let in_proj = Linear::new(in_proj_weight, Some(in_proj_bias));
        let out_proj = linear(width, width, vb.pp("out_proj"))?;
        Ok(Self {
            in_proj,
            out_proj,
            num_heads,
            head_dim,
        })
    }

    /// Causal self-attention. `causal_mask` is an additive `[seq, seq]` tensor
    /// with `0.0` at allowed positions and `-inf` (or large negative) at
    /// masked positions; it is broadcast over `[batch, heads]`.
    fn forward(&self, x: &Tensor, causal_mask: &Tensor) -> Result<Tensor, EncoderError> {
        let (batch, seq_len, _) = x.dims3()?;
        let qkv = self.in_proj.forward(x)?;
        let qkv = qkv.reshape((batch, seq_len, 3, self.num_heads, self.head_dim))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?; // (3, batch, heads, seq, head_dim)

        let q = qkv.i(0)?.contiguous()?;
        let k = qkv.i(1)?.contiguous()?;
        let v = qkv.i(2)?.contiguous()?;

        let scale = (self.head_dim as f64).sqrt();
        let attn_scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? / scale)?;
        let attn_scores = attn_scores.broadcast_add(causal_mask)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_scores)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output.permute((0, 2, 1, 3))?.reshape((
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        Ok(self.out_proj.forward(&attn_output)?)
    }
}

/// Feed-forward MLP with QuickGelu activation.
struct Mlp {
    c_fc: Linear,
    c_proj: Linear,
}

impl Mlp {
    fn load(vb: VarBuilder, width: usize, intermediate_size: usize) -> Result<Self, EncoderError> {
        let c_fc = linear(width, intermediate_size, vb.pp("c_fc"))?;
        let c_proj = linear(intermediate_size, width, vb.pp("c_proj"))?;
        Ok(Self { c_fc, c_proj })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        let x = self.c_fc.forward(x)?;
        let x = quick_gelu(&x)?;
        Ok(self.c_proj.forward(&x)?)
    }
}

/// Residual transformer block: LN → MHSA → residual → LN → MLP → residual.
struct ResidualAttentionBlock {
    ln_1: LayerNorm,
    attn: MultiHeadAttention,
    ln_2: LayerNorm,
    mlp: Mlp,
}

impl ResidualAttentionBlock {
    fn load(vb: VarBuilder, width: usize, heads: usize) -> Result<Self, EncoderError> {
        // OpenCLIP text transformer uses a fixed 4x MLP ratio.
        let intermediate_size = width * 4;
        let ln_1 = layer_norm(width, 1e-5, vb.pp("ln_1"))?;
        let attn = MultiHeadAttention::load(vb.pp("attn"), width, heads)?;
        let ln_2 = layer_norm(width, 1e-5, vb.pp("ln_2"))?;
        let mlp = Mlp::load(vb.pp("mlp"), width, intermediate_size)?;
        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor, causal_mask: &Tensor) -> Result<Tensor, EncoderError> {
        let residual = x;
        let x = self.ln_1.forward(x)?;
        let x = self.attn.forward(&x, causal_mask)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.ln_2.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        Ok((residual + x)?)
    }
}

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
    /// (`0.0` lower-triangular, `f32::MIN` above the diagonal). Built once
    /// at load time so the forward path slices instead of allocating.
    causal_mask: Tensor,
}

impl ClipText {
    /// Build the text transformer from a checkpoint-root [`VarBuilder`].
    ///
    /// Reads keys at the root level (no `text.` prefix): the OpenCLIP
    /// safetensors layout puts vision under `visual.*` and text under the
    /// root, so callers using the same checkpoint pass `vb` for text and
    /// `vb.pp("visual")` for vision.
    pub fn load(vb: VarBuilder, config: &ClipTextConfig) -> Result<Self, EncoderError> {
        let token_embedding = embedding(config.vocab_size, config.width, vb.pp("token_embedding"))?;
        let positional_embedding = vb.get(
            (config.context_length, config.width),
            "positional_embedding",
        )?;

        let mut blocks = Vec::with_capacity(config.layers);
        for i in 0..config.layers {
            let block = ResidualAttentionBlock::load(
                vb.pp(format!("transformer.resblocks.{i}")),
                config.width,
                config.heads,
            )?;
            blocks.push(block);
        }

        let ln_final = layer_norm(config.width, 1e-5, vb.pp("ln_final"))?;
        let text_projection = vb.get((config.width, config.embed_dim), "text_projection")?;

        let causal_mask = build_causal_mask(config.context_length, vb.device())?;

        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln_final,
            text_projection,
            config: config.clone(),
            causal_mask,
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
            x = block.forward(&x, &causal)?;
        }
        let x = self.ln_final.forward(&x)?;

        // EOT pooling: argmax of input IDs along sequence axis identifies
        // the `<|endoftext|>` token (highest ID in the OpenCLIP BPE vocab).
        let eot_indices = input_ids.argmax(1)?;
        let pooled = gather_at_indices(&x, &eot_indices)?;

        // Project into the shared CLIP latent space and L2-normalize.
        let projected = pooled.matmul(&self.text_projection)?;
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
}

/// Build the `[size, size]` additive causal mask: `0.0` on and below the
/// diagonal, `f32::MIN` above it. Constructed once at load time and sliced
/// per forward.
fn build_causal_mask(size: usize, device: &candle_core::Device) -> Result<Tensor, EncoderError> {
    let mut data = vec![0f32; size * size];
    for row in 0..size {
        for col in (row + 1)..size {
            data[row * size + col] = f32::MIN;
        }
    }
    Ok(Tensor::from_vec(data, (size, size), device)?)
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
}
