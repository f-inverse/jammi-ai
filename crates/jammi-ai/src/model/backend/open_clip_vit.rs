//! OpenCLIP-compatible Vision Transformer (ViT) implementation.
//!
//! Loads weights from OpenCLIP safetensors files directly, without key remapping.
//! Supports global average pooling (used by PatentCLIP) instead of CLS token pooling.

use candle_core::{IndexOp, Module, Result as CandleResult, Tensor, D};
use candle_nn::{
    conv2d_no_bias, layer_norm, linear, Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder,
};
use jammi_db::error::{JammiError, Result};

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

/// Applies `ln`, taking the ordinary differentiable composition
/// (`candle_nn::ops::layer_norm_slow`) under `training` instead of
/// `candle_nn::LayerNorm::forward`'s own dispatch.
///
/// This is the same root cause as `MultiHeadAttention::training` (see that
/// doc comment) with a second home: `candle_nn::LayerNorm::forward`
/// (candle-nn-0.11.0/src/layer_norm.rs:117-122) takes
/// `candle_nn::ops::layer_norm` — itself `apply_op3_no_bwd`
/// (candle-nn-0.11.0/src/ops.rs:944), whose result also carries
/// `BackpropOp::none()` — unconditionally whenever the input is contiguous,
/// with no training/eval distinction of its own. Every `LayerNorm` call in
/// this tower's forward path sits on a contiguous tensor, so left
/// unrouted this stops backward at the very first `LayerNorm` node
/// (`ln_pre`, ahead of every block) and no gradient is ever reachable at
/// `MultiHeadAttention`'s softmax, regardless of that flag.
fn apply_layer_norm(ln: &LayerNorm, x: &Tensor, training: bool) -> CandleResult<Tensor> {
    if !training {
        return ln.forward(x);
    }
    match ln.bias() {
        Some(bias) => candle_nn::ops::layer_norm_slow(x, ln.weight(), bias, ln.eps() as f32),
        None => ln.forward(x),
    }
}

/// Multi-head self-attention with fused in_proj.
struct MultiHeadAttention {
    in_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    /// Selects the attention-probabilities softmax kernel.
    ///
    /// `candle_nn::ops::softmax_last_dim` is `apply_op1_no_bwd`
    /// (candle-nn-0.11.0/src/ops.rs:437-439): the `Tensor` it returns carries
    /// `BackpropOp::none()` (candle-core-0.11.0/src/custom_op.rs:156-159), so a
    /// backward walk through it does not error — it silently stops traversing
    /// at that node. Through the P·V matmul immediately below, the V slice of
    /// the fused `in_proj_weight` still receives a gradient (the matmul's own
    /// operands remain differentiable), so the observable under this kernel is
    /// not "no gradient" but a *wrong* one: the Q and K slices of
    /// `in_proj_weight` come back exactly zero instead of reflecting how the
    /// attention pattern affects the loss.
    ///
    /// `training == true` therefore takes the ordinary differentiable
    /// composition (same arm as `crates/jammi-encoders/src/modernbert.rs:575`)
    /// instead. On CPU the two kernels are bit-identical (measured 0 differing
    /// bits over 4096 f32 and over 4096 bf16 elements — see
    /// `tests::softmax_last_dim_and_composed_softmax_are_bit_identical_on_cpu`),
    /// so eval output is unaffected by which arm ran; on CUDA the fused kernel
    /// and the composed reduction can differ in floating-point reduction
    /// order, which is why eval keeps the fast kernel unconditionally rather
    /// than switching this flag on for every forward pass. No production
    /// caller can set this flag today — `CandleVisionForward::forward_image`
    /// takes `&self` (crates/jammi-ai/src/model/backend/candle.rs:76-97), and
    /// the trainer only holds an `Arc<LoadedModel>`
    /// (crates/jammi-ai/src/fine_tune/trainer.rs:120) — but the composition is
    /// load-bearing correctness for whichever caller acquires mutable/owned
    /// access to train this tower.
    training: bool,
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
            training: false,
        })
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        // Fused QKV projection: (batch, seq, width) -> (batch, seq, 3*width)
        let qkv = self.in_proj.forward(x)?;
        let qkv = qkv.reshape((batch, seq_len, 3, self.num_heads, self.head_dim))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?; // (3, batch, heads, seq, head_dim)

        let q = qkv.i(0)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;

        // Scaled dot-product attention. `contiguous_matmul` guarantees both
        // operands are contiguous — candle's CUDA matmul rejects the strided
        // views left by the qkv split/transpose that its CPU matmul tolerates.
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights =
            (jammi_encoders::contiguous_matmul(&q, &k.transpose(D::Minus2, D::Minus1)?)? / scale)?;
        // See the doc comment on `training` above for why these two arms exist.
        let attn_weights = if self.training {
            candle_nn::ops::softmax(&attn_weights, D::Minus1)?
        } else {
            candle_nn::ops::softmax_last_dim(&attn_weights)?
        };
        let attn_output = jammi_encoders::contiguous_matmul(&attn_weights, &v)?;

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
    /// See `apply_layer_norm`'s doc comment: routes `ln_1`/`ln_2` to the
    /// differentiable composition under training, since
    /// `candle_nn::LayerNorm::forward` truncates backward on its own.
    training: bool,
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
            training: false,
        })
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.attn.set_training(training);
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let residual = x;
        let x = apply_layer_norm(&self.ln_1, x, self.training)?;
        let x = self.attn.forward(&x)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = apply_layer_norm(&self.ln_2, &x, self.training)?;
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
    /// See `apply_layer_norm`'s doc comment: routes `ln_pre`/`ln_post` to the
    /// differentiable composition under training, since
    /// `candle_nn::LayerNorm::forward` truncates backward on its own.
    training: bool,
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
            training: false,
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
        let x = apply_layer_norm(&self.ln_pre, &x, self.training)?;

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
        let pooled = apply_layer_norm(&self.ln_post, &pooled, self.training)?;

        // Linear projection: (batch, width) -> (batch, embed_dim)
        jammi_encoders::contiguous_matmul(&pooled, &self.proj)
    }

    /// Sets whether attention backward walks through the differentiable
    /// softmax composition (`true`) or the fast fused kernel that truncates
    /// backward (`false`, the default from `load`). See the doc comment on
    /// `MultiHeadAttention::training` for why the two arms exist and produce
    /// bit-identical eval output on CPU. Propagates to every block's
    /// attention sublayer.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        for block in &mut self.blocks {
            block.set_training(training);
        }
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

    /// Deterministic (non-RNG) `(2, 3, 8, 8)` pixel fixture built from a
    /// fixed formula so every element differs — an accidentally symmetric
    /// input could mask a dropped propagation path.
    fn fixed_pixel_values(device: &Device) -> Tensor {
        let n = 2 * 3 * 8 * 8;
        let values: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.017 - 1.0).sin() * 0.5)
            .collect();
        Tensor::from_slice(&values, (2, 3, 8, 8), device).unwrap()
    }

    /// Fixed non-uniform per-element loss weights. `Tensor::backward`'s
    /// implicit seed gradient is `ones_like(self)`, so calling it directly on
    /// a non-scalar output would supply a uniform dy that could accidentally
    /// cancel a dropped propagation path; folding through these weights
    /// before summing keeps dy non-uniform.
    fn fixed_nonuniform_weights(dims: &[usize], device: &Device) -> Tensor {
        let n: usize = dims.iter().product();
        let values: Vec<f32> = (0..n).map(|i| 0.37 + (i as f32) * 0.19).collect();
        Tensor::from_slice(&values, dims, device).unwrap()
    }

    /// L2 norm of `grad`'s rows `[start, start + width)` — used to inspect
    /// the Q / K / V slices of the fused `in_proj_weight` gradient, which
    /// stacks them at row offsets `0`, `width`, and `2 * width`.
    fn row_slice_norm(grad: &Tensor, start: usize, width: usize) -> f32 {
        grad.i((start..start + width, ..))
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt()
    }

    /// `VarBuilder::from_varmap`'s implicit `Init` for a bare `vb.get(...)`
    /// call (no hints) is `Init::Const(0.)` (candle-nn-0.11.0/src/init.rs:
    /// 143-146), which is what this file's `MultiHeadAttention::load` uses
    /// for `in_proj_weight`/`in_proj_bias`. Left at that default, Q, K, and V
    /// are all identically the zero tensor, which zeros the *true*
    /// mathematical gradient at Q and K for reasons that have nothing to do
    /// with the softmax kernel under test: with V ≡ 0, `d(attn_weights @ V)
    /// / d(attn_weights)` is the zero map, so the chain rule gives an exact
    /// zero gradient at the softmax input (hence at Q and K) regardless of
    /// which softmax kernel ran, while V's own gradient stays nonzero since
    /// it depends on `attn_weights` (not on V) through that same product.
    /// Overwriting every `Var` with a fixed nonzero value (sorted by name for
    /// determinism, since `VarMap`'s backing `HashMap` iteration order is
    /// not stable) breaks that degeneracy so these tests observe the fix's
    /// actual behavior instead of an initialization artifact.
    fn break_zero_init_symmetry(varmap: &VarMap, device: &Device) {
        let mut entries: Vec<(String, candle_core::Var)> = varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, var)| (name.clone(), var.clone()))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        for (i, (_, var)) in entries.iter().enumerate() {
            let dims = var.as_tensor().dims().to_vec();
            let n: usize = dims.iter().product();
            let offset = (i as f32) * 0.7 + 0.11;
            let values: Vec<f32> = (0..n)
                .map(|j| (((j as f32) + offset) * 0.043 - 0.9).sin() * 0.2)
                .collect();
            var.set(&Tensor::from_slice(&values, dims.as_slice(), device).unwrap())
                .unwrap();
        }
    }

    /// RED oracle: under `training = true`, `MultiHeadAttention::forward`
    /// must use the differentiable softmax composition, so gradient flows
    /// from the loss back through the attention weights to the Q and K
    /// slices of `in_proj_weight`. Reverting that arm to `softmax_last_dim`
    /// makes this fail: `BackpropOp::none()` stops the walk at the softmax
    /// node before it reaches Q/K (see the doc comment on
    /// `MultiHeadAttention::training`).
    #[test]
    fn training_true_backward_reaches_qk_through_softmax() {
        let config = tiny_config();
        let width = config.width;
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut model = OpenClipVisionTransformer::load(vb.pp("visual"), &config).unwrap();
        break_zero_init_symmetry(&varmap, &device);
        model.set_training(true);

        let input = fixed_pixel_values(&device);
        let output = model.forward(&input).unwrap();
        let weights = fixed_nonuniform_weights(output.dims(), &device);
        let loss = (output * weights).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();

        let in_proj_weight = model.blocks[0].attn.in_proj.weight();
        let grad = grads
            .get(in_proj_weight)
            .expect("in_proj_weight must have a gradient under training=true");

        let q_norm = row_slice_norm(grad, 0, width);
        let k_norm = row_slice_norm(grad, width, width);
        let v_norm = row_slice_norm(grad, 2 * width, width);

        assert!(
            q_norm > 0.0,
            "Q slice grad norm must be nonzero, got {q_norm}"
        );
        assert!(
            k_norm > 0.0,
            "K slice grad norm must be nonzero, got {k_norm}"
        );
        assert!(
            v_norm > 0.0,
            "V slice grad norm must be nonzero (positive control), got {v_norm}"
        );

        assert!(
            grads.get(model.conv1.weight()).is_some(),
            "patch-conv weight must receive a gradient — the whole graph stays connected"
        );
    }

    /// Documents the measured full-model shape at `training = false` (the
    /// default from `load`): `in_proj_weight` gets no gradient entry at
    /// all — not even on the V slice. This is *not* the softmax site's own
    /// signature in isolation (that is
    /// `multi_head_attention_eval_zeros_qk_leaves_v_nonzero` below); at the
    /// full-model level, `apply_layer_norm`'s eval arm calls
    /// `candle_nn::LayerNorm::forward` unchanged, and that independently
    /// truncates backward (see its doc comment) at `ln_pre`, ahead of every
    /// block, before `MultiHeadAttention::forward` ever runs. Asserting the
    /// honest, measured full-model behavior — rather than the softmax
    /// site's shape in isolation — also catches a dropped propagation call
    /// (e.g. `set_training` failing to reach a block would not change this
    /// assertion, since eval is `training = false` either way, but a
    /// regression that accidentally made eval reach the softmax site would
    /// still show up as a change here).
    #[test]
    fn training_false_backward_has_no_in_proj_gradient_at_all() {
        let config = tiny_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = OpenClipVisionTransformer::load(vb.pp("visual"), &config).unwrap();
        break_zero_init_symmetry(&varmap, &device);

        let input = fixed_pixel_values(&device);
        let output = model.forward(&input).unwrap();
        let weights = fixed_nonuniform_weights(output.dims(), &device);
        let loss = (output * weights).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();

        let in_proj_weight = model.blocks[0].attn.in_proj.weight();
        assert!(
            grads.get(in_proj_weight).is_none(),
            "eval's own LayerNorm kernel must still truncate backward before block 0's \
             attention runs, matching pre-fix behavior for this tower's eval path"
        );
    }

    /// Isolated reproduction of the softmax site's own defect shape,
    /// bypassing this file's `LayerNorm` calls entirely (see
    /// `training_false_backward_has_no_in_proj_gradient_at_all`'s doc
    /// comment for why the full-model fixture cannot isolate it).
    /// Constructs `MultiHeadAttention` directly and feeds it a fixed
    /// `(batch, seq, width)` tensor: under `training = false` (the default
    /// from `load`), `softmax_last_dim`'s truncated backward leaves the Q
    /// and K slices of `in_proj_weight` at an *exact* zero gradient (not
    /// merely small), while the V slice — which only ever passes through
    /// the differentiable P·V matmul — still receives one.
    #[test]
    fn multi_head_attention_eval_zeros_qk_leaves_v_nonzero() {
        let config = tiny_config();
        let width = config.width;
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let attn = MultiHeadAttention::load(vb.pp("attn"), width, config.heads).unwrap();
        break_zero_init_symmetry(&varmap, &device);

        let seq_len = 5;
        let n = 2 * seq_len * width;
        let xv: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.023 - 0.7).cos() * 0.4)
            .collect();
        let x = Tensor::from_slice(&xv, (2, seq_len, width), &device).unwrap();

        let out = attn.forward(&x).unwrap();
        let weights = fixed_nonuniform_weights(out.dims(), &device);
        let loss = (out * weights).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();

        let grad = grads.get(attn.in_proj.weight()).expect(
            "V slice must still receive a gradient through the differentiable P·V matmul \
             even under the truncated eval kernel",
        );
        let q_norm = row_slice_norm(grad, 0, width);
        let k_norm = row_slice_norm(grad, width, width);
        let v_norm = row_slice_norm(grad, 2 * width, width);

        assert_eq!(
            q_norm, 0.0,
            "Q slice grad must be exactly zero under the truncated eval kernel"
        );
        assert_eq!(
            k_norm, 0.0,
            "K slice grad must be exactly zero under the truncated eval kernel"
        );
        assert!(v_norm > 0.0, "V slice grad must stay nonzero, got {v_norm}");
    }

    /// Eval output must be byte-identical whether or not `set_training` was
    /// ever called, as long as the tower ends up back in eval mode
    /// (`training == false`, the default). Guards against the training arm
    /// leaking into eval through shared state.
    #[test]
    fn eval_output_bit_identical_regardless_of_set_training_history() {
        let config = tiny_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut model = OpenClipVisionTransformer::load(vb.pp("visual"), &config).unwrap();

        let input = fixed_pixel_values(&device);

        let bits_of = |t: &Tensor| -> Vec<u32> {
            t.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect()
        };

        let before = model.forward(&input).unwrap();
        let before_bits = bits_of(&before);

        model.set_training(true);
        model.set_training(false);
        let after = model.forward(&input).unwrap();
        let after_bits = bits_of(&after);

        assert_eq!(before_bits, after_bits);
    }

    /// Measures, rather than assumes, that `softmax_last_dim` and the
    /// composed `softmax(.., D::Minus1)` agree bit-for-bit on CPU — the
    /// premise the `training` field's doc comment relies on to claim eval is
    /// unaffected by which arm ran. bf16 bit-identity is checked via a
    /// lossless upcast to f32 (bf16 -> f32 zero-pads the mantissa, an
    /// injective map, so f32 equality after the upcast implies bf16 bit
    /// equality before it).
    #[test]
    fn softmax_last_dim_and_composed_softmax_are_bit_identical_on_cpu() {
        let device = Device::Cpu;
        let n = 8 * 512;
        let values: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * 0.0007 - 3.0).sin() * 5.0)
            .collect();

        let bits_of = |t: &Tensor| -> Vec<u32> {
            t.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect()
        };

        let x_f32 = Tensor::from_slice(&values, (8, 512), &device).unwrap();
        let a_f32 = candle_nn::ops::softmax_last_dim(&x_f32).unwrap();
        let b_f32 = candle_nn::ops::softmax(&x_f32, D::Minus1).unwrap();
        assert_eq!(
            bits_of(&a_f32),
            bits_of(&b_f32),
            "f32: softmax_last_dim vs composed softmax must match bit-for-bit on CPU"
        );

        let x_bf16 = x_f32.to_dtype(DType::BF16).unwrap();
        let a_bf16 = candle_nn::ops::softmax_last_dim(&x_bf16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let b_bf16 = candle_nn::ops::softmax(&x_bf16, D::Minus1)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        assert_eq!(
            bits_of(&a_bf16),
            bits_of(&b_bf16),
            "bf16 (compared via lossless f32 upcast): softmax_last_dim vs composed softmax must match bit-for-bit on CPU"
        );
    }
}
