//! DistilBERT deep LoRA encoder.
//!
//! DistilBERT differences from BERT:
//! - No `token_type_ids` embeddings.
//! - Attention projection names: `q_lin`, `k_lin`, `v_lin`, `out_lin`.
//! - Weight prefix: `distilbert.transformer.layer.{n}.attention.*`.
//! - No intermediate dense (uses a two-layer FFN: `ffn.lin1` / `ffn.lin2`).
//! - Layer norm names: `sa_layer_norm` (post-attention) and `output_layer_norm` (post-FFN).

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder, VarMap};
use jammi_engine::error::{JammiError, Result};

use crate::fine_tune::lora::{LoraInitMode, LoraLinear};

use super::{effective_rank, should_apply_lora, DeepLoraEncoder};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers (re-used from bert — inline copies to avoid cross-module coupling)
// ─────────────────────────────────────────────────────────────────────────────

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn load(vb: &VarBuilder, eps: f64) -> Result<Self> {
        let weight = vb
            .get_with_hints(0, "weight", candle_nn::init::Init::Const(1.0))
            .map_err(|e| JammiError::FineTune(format!("LN weight: {e}")))?;
        let bias = vb
            .get_with_hints(0, "bias", candle_nn::init::Init::Const(0.0))
            .map_err(|e| JammiError::FineTune(format!("LN bias: {e}")))?;
        Ok(Self { weight, bias, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let x = x
            .to_dtype(DType::F32)
            .map_err(|e| JammiError::FineTune(format!("LN dtype: {e}")))?;
        let mean = x
            .mean_keepdim(candle_core::D::Minus1)
            .map_err(|e| JammiError::FineTune(format!("LN mean: {e}")))?;
        let diff = x
            .broadcast_sub(&mean)
            .map_err(|e| JammiError::FineTune(format!("LN diff: {e}")))?;
        let var = diff
            .sqr()
            .map_err(|e| JammiError::FineTune(format!("LN sqr: {e}")))?
            .mean_keepdim(candle_core::D::Minus1)
            .map_err(|e| JammiError::FineTune(format!("LN var: {e}")))?;
        let norm = diff
            .broadcast_div(
                &(var + self.eps)
                    .map_err(|e| JammiError::FineTune(format!("LN +eps: {e}")))?
                    .sqrt()
                    .map_err(|e| JammiError::FineTune(format!("LN sqrt: {e}")))?,
            )
            .map_err(|e| JammiError::FineTune(format!("LN div: {e}")))?;
        let weight_f32 = self
            .weight
            .to_dtype(DType::F32)
            .map_err(|e| JammiError::FineTune(format!("LN weight dtype: {e}")))?;
        let bias_f32 = self
            .bias
            .to_dtype(DType::F32)
            .map_err(|e| JammiError::FineTune(format!("LN bias dtype: {e}")))?;
        norm.broadcast_mul(&weight_f32)
            .map_err(|e| JammiError::FineTune(format!("LN scale: {e}")))?
            .broadcast_add(&bias_f32)
            .map_err(|e| JammiError::FineTune(format!("LN shift: {e}")))?
            .to_dtype(orig_dtype)
            .map_err(|e| JammiError::FineTune(format!("LN out dtype: {e}")))
    }
}

fn load_linear(vb: &VarBuilder) -> Result<Linear> {
    let weight = vb
        .get_with_hints(0, "weight", candle_nn::init::Init::Const(0.0))
        .map_err(|e| JammiError::FineTune(format!("linear weight: {e}")))?;
    let bias = vb
        .get_with_hints(0, "bias", candle_nn::init::Init::Const(0.0))
        .ok();
    Ok(Linear::new(weight, bias))
}

enum MaybeLoraLinear {
    Frozen(Linear),
    Lora(LoraLinear),
}

impl MaybeLoraLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Frozen(l) => {
                let w_dtype = l.weight().dtype();
                let x_cast = if x.dtype() != w_dtype {
                    x.to_dtype(w_dtype)
                        .map_err(|e| JammiError::FineTune(format!("frozen linear cast: {e}")))?
                } else {
                    x.clone()
                };
                l.forward(&x_cast)
                    .map_err(|e| JammiError::FineTune(format!("frozen linear: {e}")))
            }
            Self::Lora(l) => l.forward(x),
        }
    }

    fn trainable_params(&self) -> Vec<&Tensor> {
        match self {
            Self::Frozen(_) => vec![],
            Self::Lora(l) => l.trainable_params(),
        }
    }

    fn named_weights(&self, prefix: &str) -> Result<HashMap<String, Tensor>> {
        let mut out = HashMap::new();
        if let Self::Lora(l) = self {
            out.insert(
                format!("{prefix}.lora_a"),
                l.lora_a
                    .to_device(&Device::Cpu)
                    .map_err(|e| JammiError::FineTune(format!("save lora_a: {e}")))?,
            );
            out.insert(
                format!("{prefix}.lora_b"),
                l.lora_b
                    .to_device(&Device::Cpu)
                    .map_err(|e| JammiError::FineTune(format!("save lora_b: {e}")))?,
            );
        }
        Ok(out)
    }

    fn set_training(&mut self, training: bool) {
        if let Self::Lora(l) = self {
            l.set_training(training);
        }
    }

    fn load_weights(&mut self, weights: &HashMap<String, Tensor>, prefix: &str) {
        if let Self::Lora(l) = self {
            if let Some(a) = weights.get(&format!("{prefix}.lora_a")) {
                l.lora_a = a.clone();
            }
            if let Some(b) = weights.get(&format!("{prefix}.lora_b")) {
                l.lora_b = b.clone();
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DistilBERT attention layer
// ─────────────────────────────────────────────────────────────────────────────

struct DistilAttention {
    q_lin: MaybeLoraLinear,
    k_lin: MaybeLoraLinear,
    v_lin: MaybeLoraLinear,
    out_lin: MaybeLoraLinear,
    sa_layer_norm: LayerNorm,
    num_heads: usize,
    head_dim: usize,
}

impl DistilAttention {
    fn forward(&self, hidden: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (batch, seq, _) = hidden
            .dims3()
            .map_err(|e| JammiError::FineTune(format!("attn dims: {e}")))?;
        let h = self.num_heads;
        let d = self.head_dim;

        let q = self
            .q_lin
            .forward(hidden)?
            .reshape((batch, seq, h, d))
            .map_err(|e| JammiError::FineTune(format!("Q reshape: {e}")))?
            .transpose(1, 2)
            .map_err(|e| JammiError::FineTune(format!("Q t: {e}")))?;
        let k = self
            .k_lin
            .forward(hidden)?
            .reshape((batch, seq, h, d))
            .map_err(|e| JammiError::FineTune(format!("K reshape: {e}")))?
            .transpose(1, 2)
            .map_err(|e| JammiError::FineTune(format!("K t: {e}")))?;
        let v = self
            .v_lin
            .forward(hidden)?
            .reshape((batch, seq, h, d))
            .map_err(|e| JammiError::FineTune(format!("V reshape: {e}")))?
            .transpose(1, 2)
            .map_err(|e| JammiError::FineTune(format!("V t: {e}")))?;

        let scale = (d as f64).sqrt();
        let scores = q
            .matmul(
                &k.transpose(candle_core::D::Minus1, candle_core::D::Minus2)
                    .map_err(|e| JammiError::FineTune(format!("K^T: {e}")))?,
            )
            .map_err(|e| JammiError::FineTune(format!("QK: {e}")))?;
        let scores = (scores / scale).map_err(|e| JammiError::FineTune(format!("scale: {e}")))?;

        let mask_f = mask
            .to_dtype(DType::F32)
            .map_err(|e| JammiError::FineTune(format!("mask dtype: {e}")))?
            .unsqueeze(1)
            .map_err(|e| JammiError::FineTune(format!("mask u1: {e}")))?
            .unsqueeze(2)
            .map_err(|e| JammiError::FineTune(format!("mask u2: {e}")))?;
        let additive = ((mask_f - 1.0f64)
            .map_err(|e| JammiError::FineTune(format!("mask -1: {e}")))?
            * 1e9f64)
            .map_err(|e| JammiError::FineTune(format!("mask *1e9: {e}")))?;
        let scores = scores
            .broadcast_add(&additive)
            .map_err(|e| JammiError::FineTune(format!("mask add: {e}")))?;

        let attn = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)
            .map_err(|e| JammiError::FineTune(format!("softmax: {e}")))?;

        let ctx = attn
            .matmul(&v)
            .map_err(|e| JammiError::FineTune(format!("attn@V: {e}")))?
            .transpose(1, 2)
            .map_err(|e| JammiError::FineTune(format!("ctx t: {e}")))?
            .reshape((batch, seq, h * d))
            .map_err(|e| JammiError::FineTune(format!("ctx reshape: {e}")))?;

        let out = self.out_lin.forward(&ctx)?;
        let residual =
            (out + hidden).map_err(|e| JammiError::FineTune(format!("residual: {e}")))?;
        self.sa_layer_norm.forward(&residual)
    }
}

struct DistilFfn {
    lin1: MaybeLoraLinear,
    lin2: MaybeLoraLinear,
    output_layer_norm: LayerNorm,
}

impl DistilFfn {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mid = self
            .lin1
            .forward(x)?
            .gelu_erf()
            .map_err(|e| JammiError::FineTune(format!("ffn gelu: {e}")))?;
        let out = self.lin2.forward(&mid)?;
        let residual = (out + x).map_err(|e| JammiError::FineTune(format!("ffn residual: {e}")))?;
        self.output_layer_norm.forward(&residual)
    }
}

struct DistilLayer {
    attention: DistilAttention,
    ffn: DistilFfn,
}

impl DistilLayer {
    fn forward(&self, hidden: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let after_attn = self.attention.forward(hidden, mask)?;
        self.ffn.forward(&after_attn)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Embeddings (word + position only — no token type)
// ─────────────────────────────────────────────────────────────────────────────

struct DistilEmbeddings {
    word: Tensor,
    position: Tensor,
    ln: LayerNorm,
}

impl DistilEmbeddings {
    fn load(vb: &VarBuilder) -> Result<Self> {
        let word = vb
            .get_with_hints(
                0,
                "word_embeddings.weight",
                candle_nn::init::Init::Const(0.0),
            )
            .map_err(|e| JammiError::FineTune(format!("word emb: {e}")))?;
        let position = vb
            .get_with_hints(
                0,
                "position_embeddings.weight",
                candle_nn::init::Init::Const(0.0),
            )
            .map_err(|e| JammiError::FineTune(format!("pos emb: {e}")))?;
        let ln = LayerNorm::load(&vb.pp("LayerNorm"), 1e-12)?;
        Ok(Self { word, position, ln })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (batch, seq) = input_ids
            .dims2()
            .map_err(|e| JammiError::FineTune(format!("emb dims: {e}")))?;
        let device = input_ids.device();

        let word_emb = self
            .word
            .embedding(input_ids)
            .map_err(|e| JammiError::FineTune(format!("word embed: {e}")))?;

        let pos_ids = Tensor::arange(0u32, seq as u32, device)
            .map_err(|e| JammiError::FineTune(format!("pos ids: {e}")))?
            .unsqueeze(0)
            .map_err(|e| JammiError::FineTune(format!("pos unsqueeze: {e}")))?
            .expand((batch, seq))
            .map_err(|e| JammiError::FineTune(format!("pos expand: {e}")))?;
        let pos_emb = self
            .position
            .embedding(&pos_ids)
            .map_err(|e| JammiError::FineTune(format!("pos embed: {e}")))?;

        let emb =
            (word_emb + pos_emb).map_err(|e| JammiError::FineTune(format!("emb sum: {e}")))?;
        self.ln.forward(&emb)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main encoder
// ─────────────────────────────────────────────────────────────────────────────

/// DistilBERT encoder with selective LoRA adapters on attention linears.
pub struct DistilBertLoraEncoder {
    embeddings: DistilEmbeddings,
    layers: Vec<DistilLayer>,
}

impl DistilBertLoraEncoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frozen_vb: &VarBuilder,
        lora_vb: &VarBuilder,
        num_layers: usize,
        num_heads: usize,
        hidden_size: usize,
        target_modules: &[String],
        layers_to_transform: &Option<Vec<usize>>,
        lora_rank: usize,
        lora_alpha: f64,
        use_rslora: bool,
        lora_dropout: Option<f32>,
        rank_pattern: &HashMap<String, usize>,
        init_mode: LoraInitMode,
    ) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        let base_vb = frozen_vb.pp("distilbert");

        let embeddings = DistilEmbeddings::load(&base_vb.pp("embeddings"))?;

        let mut layers = Vec::with_capacity(num_layers);
        for n in 0..num_layers {
            let layer_vb = base_vb.pp(format!("transformer.layer.{n}"));
            let lora_layer_vb = lora_vb.pp(format!("layer.{n}"));

            macro_rules! maybe_lora {
                ($name:expr, $sub:expr) => {{
                    let frozen_lin = load_linear(&layer_vb.pp($sub))?;
                    if should_apply_lora($name, target_modules, n, layers_to_transform) {
                        let rank = effective_rank($name, lora_rank, rank_pattern);
                        let l = LoraLinear::new(
                            frozen_lin,
                            rank,
                            lora_alpha,
                            use_rslora,
                            init_mode,
                            lora_dropout,
                            &lora_layer_vb.pp($name),
                        )?;
                        MaybeLoraLinear::Lora(l)
                    } else {
                        MaybeLoraLinear::Frozen(frozen_lin)
                    }
                }};
            }

            let q_lin = maybe_lora!("q_lin", "attention.q_lin");
            let k_lin = maybe_lora!("k_lin", "attention.k_lin");
            let v_lin = maybe_lora!("v_lin", "attention.v_lin");
            let out_lin = maybe_lora!("out_lin", "attention.out_lin");
            let sa_layer_norm = LayerNorm::load(&layer_vb.pp("sa_layer_norm"), 1e-12)?;

            let lin1 = maybe_lora!("lin1", "ffn.lin1");
            let lin2 = maybe_lora!("lin2", "ffn.lin2");
            let output_layer_norm = LayerNorm::load(&layer_vb.pp("output_layer_norm"), 1e-12)?;

            layers.push(DistilLayer {
                attention: DistilAttention {
                    q_lin,
                    k_lin,
                    v_lin,
                    out_lin,
                    sa_layer_norm,
                    num_heads,
                    head_dim,
                },
                ffn: DistilFfn {
                    lin1,
                    lin2,
                    output_layer_norm,
                },
            });
        }

        Ok(Self { embeddings, layers })
    }
}

impl DeepLoraEncoder for DistilBertLoraEncoder {
    fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let mut hidden = self.embeddings.forward(input_ids)?;

        for layer in &self.layers {
            hidden = layer.forward(&hidden, attention_mask)?;
        }

        // Mean pooling
        let mask = attention_mask
            .to_dtype(DType::F32)
            .map_err(|e| JammiError::FineTune(format!("mask dtype: {e}")))?
            .unsqueeze(2)
            .map_err(|e| JammiError::FineTune(format!("mask unsqueeze: {e}")))?;
        let masked = hidden
            .broadcast_mul(&mask)
            .map_err(|e| JammiError::FineTune(format!("masked: {e}")))?;
        let sum = masked
            .sum(1)
            .map_err(|e| JammiError::FineTune(format!("sum: {e}")))?;
        let count = mask
            .sum(1)
            .map_err(|e| JammiError::FineTune(format!("count: {e}")))?;
        let pooled = sum
            .broadcast_div(&count)
            .map_err(|e| JammiError::FineTune(format!("pooled: {e}")))?;

        // L2 normalise
        let norm = pooled
            .sqr()
            .map_err(|e| JammiError::FineTune(format!("norm sqr: {e}")))?
            .sum_keepdim(1)
            .map_err(|e| JammiError::FineTune(format!("norm sum: {e}")))?
            .sqrt()
            .map_err(|e| JammiError::FineTune(format!("norm sqrt: {e}")))?
            .clamp(1e-12, f64::MAX)
            .map_err(|e| JammiError::FineTune(format!("norm clamp: {e}")))?;
        pooled
            .broadcast_div(&norm)
            .map_err(|e| JammiError::FineTune(format!("l2: {e}")))
    }

    fn trainable_params(&self) -> Vec<&Tensor> {
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

    fn named_trainable_weights(&self) -> Result<HashMap<String, Tensor>> {
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

    fn set_training(&mut self, training: bool) {
        for layer in &mut self.layers {
            layer.attention.q_lin.set_training(training);
            layer.attention.k_lin.set_training(training);
            layer.attention.v_lin.set_training(training);
            layer.attention.out_lin.set_training(training);
            layer.ffn.lin1.set_training(training);
            layer.ffn.lin2.set_training(training);
        }
    }

    fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
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
// Public builder
// ─────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn build(
    weights_paths: &[&Path],
    model_config: &serde_json::Value,
    target_modules: &[String],
    layers_to_transform: &Option<Vec<usize>>,
    lora_rank: usize,
    lora_alpha: f64,
    use_rslora: bool,
    lora_dropout: Option<f32>,
    rank_pattern: &HashMap<String, usize>,
    init_mode: LoraInitMode,
    device: &Device,
    varmap: &VarMap,
    adapter_file: Option<&Path>,
    backbone_dtype: DType,
) -> Result<DistilBertLoraEncoder> {
    let frozen_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(weights_paths, backbone_dtype, device)
            .map_err(|e| JammiError::FineTune(format!("Load DistilBERT weights: {e}")))?
    };
    let lora_vb = if let Some(af) = adapter_file {
        unsafe {
            VarBuilder::from_mmaped_safetensors(&[af], DType::F32, device)
                .map_err(|e| JammiError::FineTune(format!("Load DistilBERT adapter: {e}")))?
        }
    } else {
        VarBuilder::from_varmap(varmap, DType::F32, device)
    };

    let hidden_size = model_config
        .get("dim")
        .and_then(|v| v.as_u64())
        .unwrap_or(768) as usize;
    let num_heads = model_config
        .get("n_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(12) as usize;
    let num_layers = model_config
        .get("n_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(6) as usize;

    DistilBertLoraEncoder::new(
        &frozen_vb,
        &lora_vb,
        num_layers,
        num_heads,
        hidden_size,
        target_modules,
        layers_to_transform,
        lora_rank,
        lora_alpha,
        use_rslora,
        lora_dropout,
        rank_pattern,
        init_mode,
    )
}
