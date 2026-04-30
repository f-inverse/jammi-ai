//! BERT / RoBERTa / CamemBERT / XLM-RoBERTa deep LoRA encoder.
//!
//! Loads frozen base weights from safetensors, wraps any attention linear
//! whose name matches `target_modules` with a trainable `LoraLinear`, and
//! implements a manual multi-head-attention + FFN forward pass so that
//! gradients flow through the LoRA A/B matrices.
//!
//! Weight key convention (HuggingFace layout):
//!   `{prefix}.encoder.layer.{n}.attention.self.query.weight`
//!   `{prefix}.encoder.layer.{n}.attention.self.key.weight`
//!   `{prefix}.encoder.layer.{n}.attention.self.value.weight`
//!   `{prefix}.encoder.layer.{n}.attention.output.dense.weight`
//!   `{prefix}.encoder.layer.{n}.intermediate.dense.weight`
//!   `{prefix}.encoder.layer.{n}.output.dense.weight`
//! where `{prefix}` is `bert`, `roberta`, `camembert`, or `xlm_roberta`.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder, VarMap};
use jammi_engine::error::{JammiError, Result};

use crate::fine_tune::lora::{LoraInitMode, LoraLinear};

use super::{effective_rank, should_apply_lora, DeepLoraEncoder};

// ─────────────────────────────────────────────────────────────────────────────
// Layer-norm helper
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
            .map_err(|e| JammiError::FineTune(format!("LayerNorm weight: {e}")))?;
        let bias = vb
            .get_with_hints(0, "bias", candle_nn::init::Init::Const(0.0))
            .map_err(|e| JammiError::FineTune(format!("LayerNorm bias: {e}")))?;
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
                    .map_err(|e| JammiError::FineTune(format!("LN var+eps: {e}")))?
                    .sqrt()
                    .map_err(|e| JammiError::FineTune(format!("LN sqrt: {e}")))?,
            )
            .map_err(|e| JammiError::FineTune(format!("LN div: {e}")))?;
        let weight_f32 = self.weight.to_dtype(DType::F32)
            .map_err(|e| JammiError::FineTune(format!("LN weight dtype: {e}")))?;
        let bias_f32 = self.bias.to_dtype(DType::F32)
            .map_err(|e| JammiError::FineTune(format!("LN bias dtype: {e}")))?;
        norm.broadcast_mul(&weight_f32)
            .map_err(|e| JammiError::FineTune(format!("LN scale: {e}")))?
            .broadcast_add(&bias_f32)
            .map_err(|e| JammiError::FineTune(format!("LN shift: {e}")))?
            .to_dtype(orig_dtype)
            .map_err(|e| JammiError::FineTune(format!("LN out dtype: {e}")))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear-or-LoRA wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// Either a plain frozen linear layer or a LoRA-wrapped one.
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
// Per-layer representation
// ─────────────────────────────────────────────────────────────────────────────

struct BertAttention {
    query: MaybeLoraLinear,
    key: MaybeLoraLinear,
    value: MaybeLoraLinear,
    output_dense: MaybeLoraLinear,
    output_ln: LayerNorm,
    num_heads: usize,
    head_dim: usize,
}

impl BertAttention {
    fn forward(&self, hidden: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (batch, seq, _hidden) = hidden.dims3().map_err(|e| {
            JammiError::FineTune(format!("BertAttention dims: {e}"))
        })?;
        let h = self.num_heads;
        let d = self.head_dim;

        let q = self
            .query
            .forward(hidden)?
            .reshape((batch, seq, h, d))
            .map_err(|e| JammiError::FineTune(format!("Q reshape: {e}")))?
            .transpose(1, 2)
            .map_err(|e| JammiError::FineTune(format!("Q transpose: {e}")))?;

        let k = self
            .key
            .forward(hidden)?
            .reshape((batch, seq, h, d))
            .map_err(|e| JammiError::FineTune(format!("K reshape: {e}")))?
            .transpose(1, 2)
            .map_err(|e| JammiError::FineTune(format!("K transpose: {e}")))?;

        let v = self
            .value
            .forward(hidden)?
            .reshape((batch, seq, h, d))
            .map_err(|e| JammiError::FineTune(format!("V reshape: {e}")))?
            .transpose(1, 2)
            .map_err(|e| JammiError::FineTune(format!("V transpose: {e}")))?;

        let scale = (d as f64).sqrt();
        let scores = q
            .matmul(
                &k.transpose(candle_core::D::Minus1, candle_core::D::Minus2)
                    .map_err(|e| JammiError::FineTune(format!("K^T: {e}")))?,
            )
            .map_err(|e| JammiError::FineTune(format!("QK^T: {e}")))?;
        let scores = (scores / scale)
            .map_err(|e| JammiError::FineTune(format!("QK scale: {e}")))?;

        // Causal / padding mask: mask shape [batch, seq], expand to [batch, 1, 1, seq]
        let mask_f = mask
            .to_dtype(DType::F32)
            .map_err(|e| JammiError::FineTune(format!("mask dtype: {e}")))?
            .unsqueeze(1)
            .map_err(|e| JammiError::FineTune(format!("mask unsqueeze1: {e}")))?
            .unsqueeze(2)
            .map_err(|e| JammiError::FineTune(format!("mask unsqueeze2: {e}")))?;
        let additive = ((mask_f - 1.0f64)
            .map_err(|e| JammiError::FineTune(format!("mask sub: {e}")))?
            * 1e9f64)
            .map_err(|e| JammiError::FineTune(format!("mask scale: {e}")))?;
        let scores = scores
            .broadcast_add(&additive)
            .map_err(|e| JammiError::FineTune(format!("mask add: {e}")))?;

        let attn = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)
            .map_err(|e| JammiError::FineTune(format!("softmax: {e}")))?;

        let ctx = attn
            .matmul(&v)
            .map_err(|e| JammiError::FineTune(format!("attn@V: {e}")))?
            .transpose(1, 2)
            .map_err(|e| JammiError::FineTune(format!("ctx transpose: {e}")))?
            .reshape((batch, seq, h * d))
            .map_err(|e| JammiError::FineTune(format!("ctx reshape: {e}")))?;

        let out = self.output_dense.forward(&ctx)?;
        let residual = (out + hidden)
            .map_err(|e| JammiError::FineTune(format!("attn residual: {e}")))?;
        self.output_ln.forward(&residual)
    }
}

struct BertFfn {
    intermediate: MaybeLoraLinear,
    output: MaybeLoraLinear,
    output_ln: LayerNorm,
}

impl BertFfn {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mid = self.intermediate.forward(x)?;
        let act = mid
            .gelu_erf()
            .map_err(|e| JammiError::FineTune(format!("GELU: {e}")))?;
        let out = self.output.forward(&act)?;
        let residual = (out + x)
            .map_err(|e| JammiError::FineTune(format!("FFN residual: {e}")))?;
        self.output_ln.forward(&residual)
    }
}

struct BertLayer {
    attention: BertAttention,
    ffn: BertFfn,
}

impl BertLayer {
    fn forward(&self, hidden: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let after_attn = self.attention.forward(hidden, mask)?;
        self.ffn.forward(&after_attn)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Embeddings
// ─────────────────────────────────────────────────────────────────────────────

struct BertEmbeddings {
    word: Tensor,
    position: Tensor,
    token_type: Tensor,
    ln: LayerNorm,
}

impl BertEmbeddings {
    fn load(vb: &VarBuilder) -> Result<Self> {
        let word = vb
            .get_with_hints(0, "word_embeddings.weight", candle_nn::init::Init::Const(0.0))
            .map_err(|e| JammiError::FineTune(format!("word embeddings: {e}")))?;
        let position = vb
            .get_with_hints(
                0,
                "position_embeddings.weight",
                candle_nn::init::Init::Const(0.0),
            )
            .map_err(|e| JammiError::FineTune(format!("position embeddings: {e}")))?;
        let token_type = vb
            .get_with_hints(
                0,
                "token_type_embeddings.weight",
                candle_nn::init::Init::Const(0.0),
            )
            .map_err(|e| JammiError::FineTune(format!("token_type embeddings: {e}")))?;
        let ln = LayerNorm::load(&vb.pp("LayerNorm"), 1e-12)?;
        Ok(Self {
            word,
            position,
            token_type,
            ln,
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (batch, seq) = input_ids
            .dims2()
            .map_err(|e| JammiError::FineTune(format!("embed dims: {e}")))?;
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

        let type_ids = Tensor::zeros((batch, seq), DType::U32, device)
            .map_err(|e| JammiError::FineTune(format!("type zeros: {e}")))?;
        let type_emb = self
            .token_type
            .embedding(&type_ids)
            .map_err(|e| JammiError::FineTune(format!("type embed: {e}")))?;

        let emb = (word_emb + pos_emb)
            .map_err(|e| JammiError::FineTune(format!("emb sum1: {e}")))?;
        let emb = (emb + type_emb)
            .map_err(|e| JammiError::FineTune(format!("emb sum2: {e}")))?;
        self.ln.forward(&emb)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main encoder struct
// ─────────────────────────────────────────────────────────────────────────────

/// BERT / RoBERTa encoder with selective LoRA adapters on attention linears.
pub struct BertLoraEncoder {
    embeddings: BertEmbeddings,
    layers: Vec<BertLayer>,
}

impl BertLoraEncoder {
    /// Build a `BertLoraEncoder` from frozen safetensors weights.
    ///
    /// Frozen linear layers are loaded from `frozen_vb` (backed by mmaped safetensors).
    /// LoRA A/B matrices are registered in `lora_vb` (backed by a `VarMap`).
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
        prefix: &str,
    ) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        let base_vb = frozen_vb.pp(prefix);

        let embeddings = BertEmbeddings::load(&base_vb.pp("embeddings"))?;

        let mut layers = Vec::with_capacity(num_layers);
        for n in 0..num_layers {
            let layer_vb = base_vb.pp(format!("encoder.layer.{n}"));
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

            let query = maybe_lora!("query", "attention.self.query");
            let key = maybe_lora!("key", "attention.self.key");
            let value = maybe_lora!("value", "attention.self.value");
            let output_dense = maybe_lora!("dense", "attention.output.dense");

            let output_ln =
                LayerNorm::load(&layer_vb.pp("attention.output.LayerNorm"), 1e-12)?;

            let intermediate = maybe_lora!("intermediate_dense", "intermediate.dense");
            let ffn_output = maybe_lora!("output_dense", "output.dense");
            let ffn_ln = LayerNorm::load(&layer_vb.pp("output.LayerNorm"), 1e-12)?;

            layers.push(BertLayer {
                attention: BertAttention {
                    query,
                    key,
                    value,
                    output_dense,
                    output_ln,
                    num_heads,
                    head_dim,
                },
                ffn: BertFfn {
                    intermediate,
                    output: ffn_output,
                    output_ln: ffn_ln,
                },
            });
        }

        Ok(Self { embeddings, layers })
    }
}

/// Load a `Linear` layer from `vb` — weight required, bias optional.
fn load_linear(vb: &VarBuilder) -> Result<Linear> {
    let weight = vb
        .get_with_hints(0, "weight", candle_nn::init::Init::Const(0.0))
        .map_err(|e| JammiError::FineTune(format!("linear weight: {e}")))?;
    let bias = vb
        .get_with_hints(0, "bias", candle_nn::init::Init::Const(0.0))
        .ok();
    Ok(Linear::new(weight, bias))
}

impl DeepLoraEncoder for BertLoraEncoder {
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
            .map_err(|e| JammiError::FineTune(format!("masked mul: {e}")))?;
        let sum = masked
            .sum(1)
            .map_err(|e| JammiError::FineTune(format!("sum: {e}")))?;
        let count = mask
            .sum(1)
            .map_err(|e| JammiError::FineTune(format!("count: {e}")))?;
        let pooled = sum
            .broadcast_div(&count)
            .map_err(|e| JammiError::FineTune(format!("pooled div: {e}")))?;

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
            .map_err(|e| JammiError::FineTune(format!("l2 norm: {e}")))
    }

    fn trainable_params(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.attention.query.trainable_params());
            params.extend(layer.attention.key.trainable_params());
            params.extend(layer.attention.value.trainable_params());
            params.extend(layer.attention.output_dense.trainable_params());
            params.extend(layer.ffn.intermediate.trainable_params());
            params.extend(layer.ffn.output.trainable_params());
        }
        params
    }

    fn named_trainable_weights(&self) -> Result<HashMap<String, Tensor>> {
        let mut out = HashMap::new();
        for (n, layer) in self.layers.iter().enumerate() {
            out.extend(layer.attention.query.named_weights(&format!("layer.{n}.query"))?);
            out.extend(layer.attention.key.named_weights(&format!("layer.{n}.key"))?);
            out.extend(layer.attention.value.named_weights(&format!("layer.{n}.value"))?);
            out.extend(
                layer
                    .attention
                    .output_dense
                    .named_weights(&format!("layer.{n}.dense"))?,
            );
            out.extend(
                layer
                    .ffn
                    .intermediate
                    .named_weights(&format!("layer.{n}.intermediate_dense"))?,
            );
            out.extend(
                layer
                    .ffn
                    .output
                    .named_weights(&format!("layer.{n}.output_dense"))?,
            );
        }
        Ok(out)
    }

    fn set_training(&mut self, training: bool) {
        for layer in &mut self.layers {
            layer.attention.query.set_training(training);
            layer.attention.key.set_training(training);
            layer.attention.value.set_training(training);
            layer.attention.output_dense.set_training(training);
            layer.ffn.intermediate.set_training(training);
            layer.ffn.output.set_training(training);
        }
    }

    fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        for (n, layer) in self.layers.iter_mut().enumerate() {
            layer.attention.query.load_weights(weights, &format!("layer.{n}.query"));
            layer.attention.key.load_weights(weights, &format!("layer.{n}.key"));
            layer.attention.value.load_weights(weights, &format!("layer.{n}.value"));
            layer.attention.output_dense.load_weights(weights, &format!("layer.{n}.dense"));
            layer.ffn.intermediate.load_weights(weights, &format!("layer.{n}.intermediate_dense"));
            layer.ffn.output.load_weights(weights, &format!("layer.{n}.output_dense"));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Constructor helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Detect the weight prefix for BERT-family models from `config.json`'s
/// `model_type` field.  Returns `"bert"` as a safe default.
pub fn bert_weight_prefix(model_type: &str) -> &'static str {
    match model_type {
        "roberta" => "roberta",
        "camembert" => "camembert",
        "xlm-roberta" | "xlm_roberta" => "roberta",
        _ => "bert",
    }
}

/// Build a [`BertLoraEncoder`] from raw safetensors paths and a config JSON.
///
/// `adapter_file` — when `Some`, load LoRA A/B tensors from that safetensors
///   file (inference mode).  When `None`, allocate fresh tensors in `varmap`
///   using the chosen `init_mode` (training mode).
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
) -> Result<BertLoraEncoder> {
    let frozen_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(weights_paths, backbone_dtype, device)
            .map_err(|e| JammiError::FineTune(format!("Load BERT weights: {e}")))?
    };
    let lora_vb = if let Some(af) = adapter_file {
        unsafe {
            VarBuilder::from_mmaped_safetensors(&[af], DType::F32, device)
                .map_err(|e| JammiError::FineTune(format!("Load BERT adapter: {e}")))?
        }
    } else {
        VarBuilder::from_varmap(varmap, DType::F32, device)
    };

    let model_type = model_config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("bert");
    let prefix = bert_weight_prefix(model_type);

    let hidden_size = model_config
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(768) as usize;
    let num_heads = model_config
        .get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(12) as usize;
    let num_layers = model_config
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(12) as usize;

    BertLoraEncoder::new(
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
        prefix,
    )
}
