//! ModernBERT deep LoRA encoder.
//!
//! ModernBERT differences from classic BERT:
//! - Fused QKV projection `Wqkv` (3·hidden × hidden).
//! - Output projection `Wo` (hidden × hidden).
//! - Rotary Position Embeddings (RoPE) applied to Q and K (no position embedding table).
//! - GeGLU feed-forward: `Wi` is (2·intermediate × hidden) split into gate+up; `Wo` is (hidden × intermediate).
//! - Layer norms have no bias (weight only).
//! - No token-type IDs.
//!
//! HuggingFace weight key convention:
//!   `model.layers.{n}.attn.Wqkv.weight`
//!   `model.layers.{n}.attn.Wo.weight`
//!   `model.layers.{n}.mlp.Wi.weight`
//!   `model.layers.{n}.mlp.Wo.weight`
//!   `model.layers.{n}.attn_norm.weight`
//!   `model.layers.{n}.mlp_norm.weight`
//!   `model.embeddings.tok_embeddings.weight`

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{Linear, VarBuilder, VarMap};
use jammi_engine::error::{JammiError, Result};

use crate::fine_tune::lora::{LoraInitMode, LoraLinear};

use super::{effective_rank, should_apply_lora, DeepLoraEncoder};

// ─────────────────────────────────────────────────────────────────────────────
// LayerNorm without bias (ModernBERT style)
// ─────────────────────────────────────────────────────────────────────────────

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(vb: &VarBuilder, eps: f64, dim: usize) -> Result<Self> {
        let weight = vb
            .get_with_hints((dim,), "weight", candle_nn::init::Init::Const(1.0))
            .map_err(|e| JammiError::FineTune(format!("RmsNorm weight: {e}")))?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let x = x
            .to_dtype(DType::F32)
            .map_err(|e| JammiError::FineTune(format!("RmsNorm dtype: {e}")))?;
        // True RMSNorm: normalise by root-mean-square, no mean subtraction.
        // rms = sqrt(mean(x²) + eps)
        let rms = x
            .sqr()
            .map_err(|e| JammiError::FineTune(format!("RmsNorm sqr: {e}")))?
            .mean_keepdim(candle_core::D::Minus1)
            .map_err(|e| JammiError::FineTune(format!("RmsNorm mean_sq: {e}")))?;
        let rms = (rms + self.eps)
            .map_err(|e| JammiError::FineTune(format!("RmsNorm +eps: {e}")))?
            .sqrt()
            .map_err(|e| JammiError::FineTune(format!("RmsNorm sqrt: {e}")))?;
        let norm = x
            .broadcast_div(&rms)
            .map_err(|e| JammiError::FineTune(format!("RmsNorm div: {e}")))?;
        // Cast weight to F32 to match the normalised tensor, then cast the
        // result back to the original activation dtype (e.g. BF16).
        let weight_f32 = self
            .weight
            .to_dtype(DType::F32)
            .map_err(|e| JammiError::FineTune(format!("RmsNorm weight dtype: {e}")))?;
        norm.broadcast_mul(&weight_f32)
            .map_err(|e| JammiError::FineTune(format!("RmsNorm scale: {e}")))?
            .to_dtype(orig_dtype)
            .map_err(|e| JammiError::FineTune(format!("RmsNorm out dtype: {e}")))
    }
}

fn load_linear_no_bias(
    vb: &VarBuilder,
    out_features: usize,
    in_features: usize,
) -> Result<Linear> {
    let weight = vb
        .get_with_hints(
            (out_features, in_features),
            "weight",
            candle_nn::init::Init::Const(0.0),
        )
        .map_err(|e| JammiError::FineTune(format!("linear weight: {e}")))?;
    Ok(Linear::new(weight, None))
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear-or-LoRA
// ─────────────────────────────────────────────────────────────────────────────

enum MaybeLoraLinear {
    Frozen(Linear),
    Lora(LoraLinear),
}

impl MaybeLoraLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Frozen(l) => {
                // Cast input to the weight's dtype so BF16 frozen linears
                // accept BF16 activations (and F32 activations when the
                // backbone is loaded in F32).
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
// RoPE helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Precomputed RoPE cos/sin tables: [max_seq_len, rotary_dim].
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    #[allow(dead_code)]
    rotary_dim: usize,
}

impl RotaryEmbedding {
    fn new(
        head_dim: usize,
        max_seq_len: usize,
        rope_base: f64,
        device: &Device,
    ) -> Result<Self> {
        // Standard HF-style RoPE table. Build cos/sin at full `head_dim`
        // by duplicating the `half` frequencies; this matches the
        // `rotate_half` formulation `(q*cos) + (rotate_half(q)*sin)` where
        // `rotate_half(x) = cat(-x[..,half:], x[..,:half])`.
        let rotary_dim = head_dim;
        let half = rotary_dim / 2;
        let mut cos_vec = Vec::with_capacity(max_seq_len * head_dim);
        let mut sin_vec = Vec::with_capacity(max_seq_len * head_dim);

        for pos in 0..max_seq_len {
            // First half: cos(pos·θ_i), sin(pos·θ_i) for i in 0..half.
            for i in 0..half {
                let theta = (pos as f64)
                    * (rope_base.powf(-2.0 * i as f64 / head_dim as f64));
                cos_vec.push(theta.cos() as f32);
                sin_vec.push(theta.sin() as f32);
            }
            // Second half: same values duplicated (HF `cat([freqs, freqs])`).
            for i in 0..half {
                let theta = (pos as f64)
                    * (rope_base.powf(-2.0 * i as f64 / head_dim as f64));
                cos_vec.push(theta.cos() as f32);
                sin_vec.push(theta.sin() as f32);
            }
        }

        let cos = Tensor::from_vec(cos_vec, (max_seq_len, head_dim), device)
            .map_err(|e| JammiError::FineTune(format!("RoPE cos: {e}")))?;
        let sin = Tensor::from_vec(sin_vec, (max_seq_len, head_dim), device)
            .map_err(|e| JammiError::FineTune(format!("RoPE sin: {e}")))?;

        Ok(Self { cos, sin, rotary_dim })
    }

    /// Apply RoPE to a [batch, num_heads, seq, head_dim] tensor.
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (_batch, _heads, seq, head_dim) = x.dims4().map_err(|e| {
            JammiError::FineTune(format!("RoPE apply dims: {e}"))
        })?;
        let half = head_dim / 2;
        let x_dtype = x.dtype();

        // cos/sin: [seq, head_dim] → [1, 1, seq, head_dim]; cast to activation
        // dtype so BF16/F16 backbones don't produce a dtype mismatch in mul.
        let cos = self
            .cos
            .i(..seq)
            .map_err(|e| JammiError::FineTune(format!("RoPE cos slice: {e}")))?
            .to_dtype(x_dtype)
            .map_err(|e| JammiError::FineTune(format!("RoPE cos dtype: {e}")))?
            .unsqueeze(0)
            .map_err(|e| JammiError::FineTune(format!("RoPE cos u0: {e}")))?
            .unsqueeze(0)
            .map_err(|e| JammiError::FineTune(format!("RoPE cos u1: {e}")))?;
        let sin = self
            .sin
            .i(..seq)
            .map_err(|e| JammiError::FineTune(format!("RoPE sin slice: {e}")))?
            .to_dtype(x_dtype)
            .map_err(|e| JammiError::FineTune(format!("RoPE sin dtype: {e}")))?
            .unsqueeze(0)
            .map_err(|e| JammiError::FineTune(format!("RoPE sin u0: {e}")))?
            .unsqueeze(0)
            .map_err(|e| JammiError::FineTune(format!("RoPE sin u1: {e}")))?;

        // Split x into first and second half along last dim
        let x1 = x
            .narrow(candle_core::D::Minus1, 0, half)
            .map_err(|e| JammiError::FineTune(format!("RoPE x1: {e}")))?;
        let x2 = x
            .narrow(candle_core::D::Minus1, half, half)
            .map_err(|e| JammiError::FineTune(format!("RoPE x2: {e}")))?;

        // rotate_half: [-x2, x1]
        let neg_x2 = (x2 * -1.0f64)
            .map_err(|e| JammiError::FineTune(format!("RoPE neg_x2: {e}")))?;
        let rot_half = Tensor::cat(&[&neg_x2, &x1], candle_core::D::Minus1)
            .map_err(|e| JammiError::FineTune(format!("RoPE cat: {e}")))?;

        // rotated = x * cos + rotate_half(x) * sin
        let out = x
            .broadcast_mul(&cos)
            .map_err(|e| JammiError::FineTune(format!("RoPE x*cos: {e}")))?;
        let sin_part = rot_half
            .broadcast_mul(&sin)
            .map_err(|e| JammiError::FineTune(format!("RoPE rh*sin: {e}")))?;

        (out + sin_part).map_err(|e| JammiError::FineTune(format!("RoPE add: {e}")))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Attention layer
// ─────────────────────────────────────────────────────────────────────────────

struct ModernBertAttention {
    wqkv: MaybeLoraLinear,
    wo: MaybeLoraLinear,
    /// `None` for layer 0, where ModernBERT replaces `attn_norm` with
    /// `nn.Identity()` because the embedding RMSNorm already normalises the
    /// input. All other layers carry a learned RMSNorm.
    attn_norm: Option<RmsNorm>,
    rope: RotaryEmbedding,
    num_heads: usize,
    head_dim: usize,
}

impl ModernBertAttention {
    fn forward(&self, hidden: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let act_dtype = hidden.dtype();
        let normed = match &self.attn_norm {
            Some(ln) => ln.forward(hidden)?,
            // Layer 0: embedding RMSNorm already normalised the input; pass
            // through without a dtype cast so BF16 activations are preserved.
            None => hidden.clone(),
        };
        let (batch, seq, _) = normed
            .dims3()
            .map_err(|e| JammiError::FineTune(format!("attn dims: {e}")))?;
        let h = self.num_heads;
        let d = self.head_dim;

        // Project to QKV: [batch, seq, 3*hidden]
        let qkv = self.wqkv.forward(&normed)?;

        // Split into Q, K, V each [batch, seq, hidden]
        let q = qkv
            .narrow(candle_core::D::Minus1, 0, h * d)
            .map_err(|e| JammiError::FineTune(format!("Q narrow: {e}")))?
            .reshape((batch, seq, h, d))
            .map_err(|e| JammiError::FineTune(format!("Q reshape: {e}")))?
            .transpose(1, 2)
            .map_err(|e| JammiError::FineTune(format!("Q t: {e}")))?;
        let k = qkv
            .narrow(candle_core::D::Minus1, h * d, h * d)
            .map_err(|e| JammiError::FineTune(format!("K narrow: {e}")))?
            .reshape((batch, seq, h, d))
            .map_err(|e| JammiError::FineTune(format!("K reshape: {e}")))?
            .transpose(1, 2)
            .map_err(|e| JammiError::FineTune(format!("K t: {e}")))?;
        let v = qkv
            .narrow(candle_core::D::Minus1, 2 * h * d, h * d)
            .map_err(|e| JammiError::FineTune(format!("V narrow: {e}")))?
            .reshape((batch, seq, h, d))
            .map_err(|e| JammiError::FineTune(format!("V reshape: {e}")))?
            .transpose(1, 2)
            .map_err(|e| JammiError::FineTune(format!("V t: {e}")))?
            // candle's matmul rejects non-contiguous batch layouts; Q and K
            // are made contiguous as a side effect of the RoPE op chain, but
            // V skips RoPE, so make it explicit here.
            .contiguous()
            .map_err(|e| JammiError::FineTune(format!("V contiguous: {e}")))?;

        // Apply RoPE to Q and K (cos/sin are cast to act_dtype inside apply).
        let q = self.rope.apply(&q)?;
        let k = self.rope.apply(&k)?;

        // Upcast Q/K/V to F32 for numerically stable dot-product attention and
        // softmax, then cast the context back to the backbone activation dtype.
        let q_f32 = q
            .to_dtype(DType::F32)
            .map_err(|e| JammiError::FineTune(format!("Q f32: {e}")))?;
        let k_f32 = k
            .to_dtype(DType::F32)
            .map_err(|e| JammiError::FineTune(format!("K f32: {e}")))?;
        let v_f32 = v
            .to_dtype(DType::F32)
            .map_err(|e| JammiError::FineTune(format!("V f32: {e}")))?;

        // Scaled dot-product attention (all in F32)
        let scale = (d as f64).sqrt();
        let scores = q_f32
            .matmul(
                &k_f32
                    .transpose(candle_core::D::Minus1, candle_core::D::Minus2)
                    .map_err(|e| JammiError::FineTune(format!("K^T: {e}")))?,
            )
            .map_err(|e| JammiError::FineTune(format!("QK: {e}")))?;
        let scores = (scores / scale)
            .map_err(|e| JammiError::FineTune(format!("scale: {e}")))?;

        // Additive attention mask in F32: 0 → -1e9 for padding positions.
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
            .matmul(&v_f32)
            .map_err(|e| JammiError::FineTune(format!("attn@V: {e}")))?
            .transpose(1, 2)
            .map_err(|e| JammiError::FineTune(format!("ctx t: {e}")))?
            .contiguous()
            .map_err(|e| JammiError::FineTune(format!("ctx contiguous: {e}")))?
            .reshape((batch, seq, h * d))
            .map_err(|e| JammiError::FineTune(format!("ctx reshape: {e}")))?
            // Cast back to the activation dtype before the output projection.
            .to_dtype(act_dtype)
            .map_err(|e| JammiError::FineTune(format!("ctx dtype: {e}")))?;

        let out = self.wo.forward(&ctx)?;
        // Pre-norm residual: hidden (before norm) + attention output
        (out + hidden).map_err(|e| JammiError::FineTune(format!("attn residual: {e}")))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GeGLU FFN
// ─────────────────────────────────────────────────────────────────────────────

struct ModernBertMlp {
    /// Gate+up projection — LoRA-eligible via target name "Wi".
    wi: MaybeLoraLinear,
    /// Down projection — LoRA-eligible via target name "Wo" (matches suffix).
    wo: MaybeLoraLinear,
    mlp_norm: RmsNorm,
}

impl ModernBertMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let normed = self.mlp_norm.forward(x)?;

        // Wi: [batch, seq, 2*intermediate] → split gate and up
        let up_gate = self.wi.forward(&normed)?;
        let intermediate = up_gate.dim(candle_core::D::Minus1).map_err(|e| {
            JammiError::FineTune(format!("Wi dim: {e}"))
        })? / 2;

        let gate = up_gate
            .narrow(candle_core::D::Minus1, 0, intermediate)
            .map_err(|e| JammiError::FineTune(format!("gate narrow: {e}")))?;
        let up = up_gate
            .narrow(candle_core::D::Minus1, intermediate, intermediate)
            .map_err(|e| JammiError::FineTune(format!("up narrow: {e}")))?;

        let act = (gate.gelu_erf().map_err(|e| {
            JammiError::FineTune(format!("GELU: {e}"))
        })? * up)
            .map_err(|e| JammiError::FineTune(format!("gate*up: {e}")))?;

        let out = self.wo.forward(&act)?;

        (out + x).map_err(|e| JammiError::FineTune(format!("mlp residual: {e}")))
    }
}

struct ModernBertLayer {
    attention: ModernBertAttention,
    mlp: ModernBertMlp,
}

impl ModernBertLayer {
    fn forward(&self, hidden: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let after_attn = self.attention.forward(hidden, mask)?;
        self.mlp.forward(&after_attn)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main encoder
// ─────────────────────────────────────────────────────────────────────────────

/// ModernBERT encoder with selective LoRA adapters on attention linears.
pub struct ModernBertLoraEncoderInner {
    embeddings: Tensor,
    emb_norm: RmsNorm,
    layers: Vec<ModernBertLayer>,
}

impl ModernBertLoraEncoderInner {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frozen_vb: &VarBuilder,
        lora_vb: &VarBuilder,
        num_layers: usize,
        num_heads: usize,
        hidden_size: usize,
        vocab_size: usize,
        intermediate_size: usize,
        max_seq_len: usize,
        rope_base: f64,
        target_modules: &[String],
        layers_to_transform: &Option<Vec<usize>>,
        lora_rank: usize,
        lora_alpha: f64,
        use_rslora: bool,
        lora_dropout: Option<f32>,
        rank_pattern: &HashMap<String, usize>,
        init_mode: LoraInitMode,
        device: &Device,
    ) -> Result<Self> {
        let head_dim = hidden_size / num_heads;

        // Embedding matrix is [vocab_size, hidden_size]; pass the full 2-D shape so
        // candle does not interpret `0` as a 1-D shape `[0]` and reject the tensor.
        let embeddings = frozen_vb
            .get_with_hints(
                (vocab_size, hidden_size),
                "model.embeddings.tok_embeddings.weight",
                candle_nn::init::Init::Const(0.0),
            )
            .map_err(|e| JammiError::FineTune(format!("ModernBERT tok_embeddings: {e}")))?;

        let emb_norm = RmsNorm::load(
            &frozen_vb.pp("model.embeddings.norm"),
            1e-5,
            hidden_size,
        )?;

        let mut layers = Vec::with_capacity(num_layers);

        for n in 0..num_layers {
            let layer_vb = frozen_vb.pp(format!("model.layers.{n}"));
            let lora_layer_vb = lora_vb.pp(format!("layer.{n}"));

            macro_rules! maybe_lora {
                ($name:expr, $sub:expr, $out:expr, $in:expr) => {{
                    let frozen_lin =
                        load_linear_no_bias(&layer_vb.pp($sub), $out, $in)?;
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

            // attn.Wqkv: fused QKV → out=3*hidden, in=hidden
            let wqkv = maybe_lora!("Wqkv", "attn.Wqkv", 3 * hidden_size, hidden_size);
            // attn.Wo:  hidden × hidden
            let wo   = maybe_lora!("Wo",   "attn.Wo",   hidden_size,     hidden_size);

            // ModernBERT replaces layer 0's `attn_norm` with `nn.Identity()`
            // because the embedding RMSNorm already pre-normalises the input.
            let attn_norm = if n == 0 {
                None
            } else {
                Some(RmsNorm::load(
                    &layer_vb.pp("attn_norm"),
                    1e-5,
                    hidden_size,
                )?)
            };

            let rope = RotaryEmbedding::new(head_dim, max_seq_len, rope_base, device)?;

            // MLP: use a distinct internal name "mlp.Wo" so that it still matches
            // the target-module string "Wo" via ends_with() but has a unique VarMap
            // key that does not collide with the attention "Wo".
            // GeGLU: Wi packs gate+up → out=2*intermediate, in=hidden
            let wi     = maybe_lora!("Wi",     "mlp.Wi",
                                     2 * intermediate_size, hidden_size);
            // mlp.Wo: hidden × intermediate
            let mlp_wo = maybe_lora!("mlp.Wo", "mlp.Wo",
                                     hidden_size,           intermediate_size);
            let mlp_norm = RmsNorm::load(&layer_vb.pp("mlp_norm"), 1e-5, hidden_size)?;

            layers.push(ModernBertLayer {
                attention: ModernBertAttention {
                    wqkv,
                    wo,
                    attn_norm,
                    rope,
                    num_heads,
                    head_dim,
                },
                mlp: ModernBertMlp {
                    wi,
                    wo: mlp_wo,
                    mlp_norm,
                },
            });
        }

        Ok(Self {
            embeddings,
            emb_norm,
            layers,
        })
    }
}

impl DeepLoraEncoder for ModernBertLoraEncoderInner {
    fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // candle 0.9 `Tensor::embedding` requires rank-1 indexes; manually
        // flatten → index_select → reshape so we can feed [batch, seq] ids.
        let (batch, seq) = input_ids
            .dims2()
            .map_err(|e| JammiError::FineTune(format!("input_ids dims: {e}")))?;
        let hidden_size = self
            .embeddings
            .dim(1)
            .map_err(|e| JammiError::FineTune(format!("embeddings hidden dim: {e}")))?;
        let flat_ids = input_ids
            .flatten_all()
            .map_err(|e| JammiError::FineTune(format!("flatten ids: {e}")))?;
        let word_emb = self
            .embeddings
            .index_select(&flat_ids, 0)
            .map_err(|e| JammiError::FineTune(format!("tok embed: {e}")))?
            .reshape((batch, seq, hidden_size))
            .map_err(|e| JammiError::FineTune(format!("tok embed reshape: {e}")))?;
        let mut hidden = self.emb_norm.forward(&word_emb)?;

        for layer in &self.layers {
            hidden = layer.forward(&hidden, attention_mask)?;
        }

        // Mean pooling: cast mask to hidden's dtype to avoid BF16/F32 mismatch.
        let hidden_dtype = hidden.dtype();
        let mask = attention_mask
            .to_dtype(hidden_dtype)
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
            params.extend(layer.attention.wqkv.trainable_params());
            params.extend(layer.attention.wo.trainable_params());
            params.extend(layer.mlp.wi.trainable_params());
            params.extend(layer.mlp.wo.trainable_params());
        }
        params
    }

    fn named_trainable_weights(&self) -> Result<HashMap<String, Tensor>> {
        let mut out = HashMap::new();
        for (n, layer) in self.layers.iter().enumerate() {
            out.extend(layer.attention.wqkv.named_weights(&format!("layer.{n}.Wqkv"))?);
            out.extend(layer.attention.wo.named_weights(&format!("layer.{n}.Wo"))?);
            out.extend(layer.mlp.wi.named_weights(&format!("layer.{n}.Wi"))?);
            out.extend(layer.mlp.wo.named_weights(&format!("layer.{n}.mlp.Wo"))?);
        }
        Ok(out)
    }

    fn set_training(&mut self, training: bool) {
        for layer in &mut self.layers {
            layer.attention.wqkv.set_training(training);
            layer.attention.wo.set_training(training);
            layer.mlp.wi.set_training(training);
            layer.mlp.wo.set_training(training);
        }
    }

    fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        for (n, layer) in self.layers.iter_mut().enumerate() {
            layer.attention.wqkv.load_weights(weights, &format!("layer.{n}.Wqkv"));
            layer.attention.wo.load_weights(weights, &format!("layer.{n}.Wo"));
            layer.mlp.wi.load_weights(weights, &format!("layer.{n}.Wi"));
            layer.mlp.wo.load_weights(weights, &format!("layer.{n}.mlp.Wo"));
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
) -> Result<ModernBertLoraEncoderInner> {
    let frozen_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(weights_paths, backbone_dtype, device)
            .map_err(|e| JammiError::FineTune(format!("Load ModernBERT weights: {e}")))?
    };
    let lora_vb = if let Some(af) = adapter_file {
        unsafe {
            VarBuilder::from_mmaped_safetensors(&[af], DType::F32, device)
                .map_err(|e| JammiError::FineTune(format!("Load ModernBERT adapter: {e}")))?
        }
    } else {
        VarBuilder::from_varmap(varmap, DType::F32, device)
    };

    let hidden_size = model_config
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(768) as usize;
    let vocab_size = model_config
        .get("vocab_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(50368) as usize;
    let num_heads = model_config
        .get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(12) as usize;
    let num_layers = model_config
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(22) as usize;
    let intermediate_size = model_config
        .get("intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(1152) as usize;
    let max_seq_len = model_config
        .get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .unwrap_or(8192) as usize;
    let rope_base = model_config
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(10_000.0);

    ModernBertLoraEncoderInner::new(
        &frozen_vb,
        &lora_vb,
        num_layers,
        num_heads,
        hidden_size,
        vocab_size,
        intermediate_size,
        max_seq_len,
        rope_base,
        target_modules,
        layers_to_transform,
        lora_rank,
        lora_alpha,
        use_rslora,
        lora_dropout,
        rank_pattern,
        init_mode,
        device,
    )
}
