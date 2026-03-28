//! OpenCLIP-compatible Vision Transformer (ViT) implementation.
//!
//! Loads weights from OpenCLIP safetensors files directly, without key remapping.
//! Supports global average pooling (used by PatentCLIP) instead of CLS token pooling.

use candle_core::{IndexOp, Module, Result as CandleResult, Tensor, D};
use candle_nn::{
    conv2d_no_bias, layer_norm, linear, Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder,
};
use jammi_engine::error::{JammiError, Result};

/// Configuration for an OpenCLIP vision transformer.
/// Default CLIP normalization constants (used when preprocess_cfg is absent).
#[allow(clippy::excessive_precision)]
const DEFAULT_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
#[allow(clippy::excessive_precision)]
const DEFAULT_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

#[derive(Debug, Clone)]
pub struct OpenClipVisionConfig {
    /// Width of the transformer (hidden dimension).
    pub width: usize,
    /// Number of transformer layers.
    pub layers: usize,
    /// Number of attention heads.
    pub heads: usize,
    /// MLP intermediate size ratio.
    pub mlp_ratio: f64,
    /// Input image size (square).
    pub image_size: usize,
    /// Patch size for patch embedding.
    pub patch_size: usize,
    /// Output embedding dimension (projection from width -> embed_dim).
    pub embed_dim: usize,
    /// Whether to use global average pooling (true) or CLS token pooling (false).
    pub global_average_pool: bool,
    /// Per-channel normalization mean (from preprocess_cfg).
    pub preprocess_mean: [f32; 3],
    /// Per-channel normalization std (from preprocess_cfg).
    pub preprocess_std: [f32; 3],
}

impl OpenClipVisionConfig {
    /// Parse from an OpenCLIP config JSON (`open_clip_config.json`).
    pub fn from_open_clip_config(config: &serde_json::Value) -> Result<Self> {
        let model_cfg = config.get("model_cfg").ok_or_else(|| JammiError::Model {
            model_id: String::new(),
            message: "OpenCLIP config missing 'model_cfg'".into(),
        })?;
        let vision_cfg = model_cfg
            .get("vision_cfg")
            .ok_or_else(|| JammiError::Model {
                model_id: String::new(),
                message: "OpenCLIP config missing 'model_cfg.vision_cfg'".into(),
            })?;

        let embed_dim = model_cfg
            .get("embed_dim")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| JammiError::Model {
                model_id: String::new(),
                message: "OpenCLIP config missing 'model_cfg.embed_dim'".into(),
            })? as usize;

        let width = vision_cfg
            .get("width")
            .and_then(|v| v.as_u64())
            .unwrap_or(768) as usize;

        Ok(Self {
            width,
            layers: vision_cfg
                .get("layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(12) as usize,
            // Default to width/64 (ViT convention: head_dim=64)
            heads: vision_cfg
                .get("heads")
                .and_then(|v| v.as_u64())
                .unwrap_or((width / 64) as u64) as usize,
            mlp_ratio: vision_cfg
                .get("mlp_ratio")
                .and_then(|v| v.as_f64())
                .unwrap_or(4.0),
            image_size: vision_cfg
                .get("image_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(224) as usize,
            patch_size: vision_cfg
                .get("patch_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(32) as usize,
            embed_dim,
            global_average_pool: vision_cfg
                .get("global_average_pool")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            preprocess_mean: parse_f32_array(config.pointer("/preprocess_cfg/mean"), DEFAULT_MEAN),
            preprocess_std: parse_f32_array(config.pointer("/preprocess_cfg/std"), DEFAULT_STD),
        })
    }
}

/// Parse a 3-element f32 array from JSON, falling back to a default.
fn parse_f32_array(value: Option<&serde_json::Value>, default: [f32; 3]) -> [f32; 3] {
    value
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            if arr.len() >= 3 {
                Some([
                    arr[0].as_f64()? as f32,
                    arr[1].as_f64()? as f32,
                    arr[2].as_f64()? as f32,
                ])
            } else {
                None
            }
        })
        .unwrap_or(default)
}

/// QuickGelu activation: x * sigmoid(1.702 * x).
fn quick_gelu(xs: &Tensor) -> CandleResult<Tensor> {
    xs * candle_nn::ops::sigmoid(&(xs * 1.702f64)?)?
}

/// Multi-head self-attention with fused in_proj.
struct MultiHeadAttention {
    in_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    fn load(vb: VarBuilder, width: usize, num_heads: usize) -> CandleResult<Self> {
        let head_dim = width / num_heads;
        // OpenCLIP uses `in_proj_weight` / `in_proj_bias` (underscore, not dot-separated).
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

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        // Fused QKV projection: (batch, seq, width) -> (batch, seq, 3*width)
        let qkv = self.in_proj.forward(x)?;
        let qkv = qkv.reshape((batch, seq_len, 3, self.num_heads, self.head_dim))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?; // (3, batch, heads, seq, head_dim)

        let q = qkv.i(0)?.contiguous()?;
        let k = qkv.i(1)?.contiguous()?;
        let v = qkv.i(2)?.contiguous()?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? / scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, width)
        let attn_output = attn_output.permute((0, 2, 1, 3))?.reshape((
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        self.out_proj.forward(&attn_output)
    }
}

/// Feed-forward MLP with QuickGelu activation.
struct Mlp {
    c_fc: Linear,
    c_proj: Linear,
}

impl Mlp {
    fn load(vb: VarBuilder, width: usize, intermediate_size: usize) -> CandleResult<Self> {
        let c_fc = linear(width, intermediate_size, vb.pp("c_fc"))?;
        let c_proj = linear(intermediate_size, width, vb.pp("c_proj"))?;
        Ok(Self { c_fc, c_proj })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x = self.c_fc.forward(x)?;
        let x = quick_gelu(&x)?;
        self.c_proj.forward(&x)
    }
}

/// Residual attention block: LN -> MHSA -> residual -> LN -> MLP -> residual.
struct ResidualAttentionBlock {
    ln_1: LayerNorm,
    attn: MultiHeadAttention,
    ln_2: LayerNorm,
    mlp: Mlp,
}

impl ResidualAttentionBlock {
    fn load(vb: VarBuilder, width: usize, heads: usize, mlp_ratio: f64) -> CandleResult<Self> {
        let intermediate_size = (width as f64 * mlp_ratio) as usize;
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

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let residual = x;
        let x = self.ln_1.forward(x)?;
        let x = self.attn.forward(&x)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.ln_2.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        residual + x
    }
}

/// OpenCLIP Vision Transformer.
///
/// Weight keys match the OpenCLIP safetensors layout under the `visual.*` prefix.
pub struct OpenClipVisionTransformer {
    conv1: Conv2d,
    class_embedding: Tensor,
    positional_embedding: Tensor,
    ln_pre: LayerNorm,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: LayerNorm,
    proj: Tensor,
    config: OpenClipVisionConfig,
}

impl OpenClipVisionTransformer {
    /// Load from a VarBuilder scoped to the `visual` prefix.
    ///
    /// Expects the VarBuilder to be created with `vb.pp("visual")` so that
    /// weight keys like `conv1.weight`, `class_embedding`, etc. resolve correctly.
    pub fn load(vb: VarBuilder, config: &OpenClipVisionConfig) -> CandleResult<Self> {
        let conv_config = Conv2dConfig {
            stride: config.patch_size,
            ..Default::default()
        };
        let conv1 = conv2d_no_bias(
            3,
            config.width,
            config.patch_size,
            conv_config,
            vb.pp("conv1"),
        )?;

        let class_embedding = vb.get(&[config.width], "class_embedding")?;
        let grid_size = config.image_size / config.patch_size;
        let num_positions = grid_size * grid_size + 1; // +1 for CLS token
        let positional_embedding =
            vb.get(&[num_positions, config.width], "positional_embedding")?;

        let ln_pre = layer_norm(config.width, 1e-5, vb.pp("ln_pre"))?;

        let mut blocks = Vec::with_capacity(config.layers);
        for i in 0..config.layers {
            let block = ResidualAttentionBlock::load(
                vb.pp(format!("transformer.resblocks.{i}")),
                config.width,
                config.heads,
                config.mlp_ratio,
            )?;
            blocks.push(block);
        }

        let ln_post = layer_norm(config.width, 1e-5, vb.pp("ln_post"))?;
        let proj = vb.get(&[config.width, config.embed_dim], "proj")?;

        Ok(Self {
            conv1,
            class_embedding,
            positional_embedding,
            ln_pre,
            blocks,
            ln_post,
            proj,
            config: config.clone(),
        })
    }

    /// Forward pass: pixel values → embedding vector.
    ///
    /// Input: `(batch, 3, image_size, image_size)` tensor.
    /// Output: `(batch, embed_dim)` tensor.
    pub fn forward(&self, pixel_values: &Tensor) -> CandleResult<Tensor> {
        let batch_size = pixel_values.dim(0)?;

        // Patch embedding: (batch, 3, H, W) -> (batch, width, grid, grid)
        let x = self.conv1.forward(pixel_values)?;

        // Flatten spatial dims: (batch, width, grid*grid) -> (batch, grid*grid, width)
        let x = x.flatten_from(2)?.permute((0, 2, 1))?;

        // Prepend CLS token: (batch, grid*grid+1, width)
        let cls = self.class_embedding.unsqueeze(0)?.unsqueeze(0)?.expand((
            batch_size,
            1,
            self.config.width,
        ))?;
        let x = Tensor::cat(&[&cls, &x], 1)?;

        // Add positional embedding
        let x = x.broadcast_add(&self.positional_embedding)?;

        // Pre-LayerNorm
        let x = self.ln_pre.forward(&x)?;

        // Transformer blocks
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // Pooling
        let pooled = if self.config.global_average_pool {
            // Global average pool over patch tokens (exclude CLS at index 0)
            let patch_tokens = x.i((.., 1.., ..))?;
            patch_tokens.mean(1)?
        } else {
            // CLS token pooling
            x.i((.., 0, ..))?
        };

        // Post-LayerNorm
        let pooled = self.ln_post.forward(&pooled)?;

        // Linear projection: (batch, width) -> (batch, embed_dim)
        pooled.matmul(&self.proj)
    }

    /// Return the output embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.config.embed_dim
    }

    /// Return the expected input image size.
    pub fn image_size(&self) -> usize {
        self.config.image_size
    }

    /// Return the preprocessing normalization mean.
    pub fn preprocess_mean(&self) -> [f32; 3] {
        self.config.preprocess_mean
    }

    /// Return the preprocessing normalization std.
    pub fn preprocess_std(&self) -> [f32; 3] {
        self.config.preprocess_std
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    fn tiny_config() -> OpenClipVisionConfig {
        OpenClipVisionConfig {
            width: 32,
            layers: 2,
            heads: 4,
            mlp_ratio: 4.0,
            image_size: 8,
            patch_size: 4,
            embed_dim: 16,
            global_average_pool: true,
            preprocess_mean: DEFAULT_MEAN,
            preprocess_std: DEFAULT_STD,
        }
    }

    #[test]
    fn test_config_from_open_clip_json() {
        let json = serde_json::json!({
            "model_cfg": {
                "embed_dim": 512,
                "vision_cfg": {
                    "image_size": 224,
                    "patch_size": 32,
                    "width": 768,
                    "layers": 12,
                    "heads": 24,
                    "mlp_ratio": 4.0,
                    "global_average_pool": true
                },
                "text_cfg": {}
            },
            "preprocess_cfg": {}
        });

        let config = OpenClipVisionConfig::from_open_clip_config(&json).unwrap();
        assert_eq!(config.embed_dim, 512);
        assert_eq!(config.width, 768);
        assert_eq!(config.layers, 12);
        assert_eq!(config.heads, 24);
        assert_eq!(config.image_size, 224);
        assert_eq!(config.patch_size, 32);
        assert!(config.global_average_pool);
    }

    #[test]
    fn test_forward_output_shape() {
        let config = tiny_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Initialize all weights with random values
        let visual_vb = vb.pp("visual");

        let model = OpenClipVisionTransformer::load(visual_vb, &config).unwrap();

        // Create a random input image: (batch=2, channels=3, 8, 8)
        let input = Tensor::randn(0f32, 1.0, (2, 3, 8, 8), &device).unwrap();
        let output = model.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 16]); // (batch=2, embed_dim=16)
    }

    #[test]
    fn test_forward_cls_pooling() {
        let config = OpenClipVisionConfig {
            global_average_pool: false,
            ..tiny_config()
        };
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let model = OpenClipVisionTransformer::load(vb.pp("visual"), &config).unwrap();

        let input = Tensor::randn(0f32, 1.0, (1, 3, 8, 8), &device).unwrap();
        let output = model.forward(&input).unwrap();

        assert_eq!(output.dims(), &[1, 16]);
    }
}
