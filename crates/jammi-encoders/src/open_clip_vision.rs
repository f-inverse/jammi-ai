//! OpenCLIP-compatible Vision Transformer (ViT).
//!
//! Loads weights from OpenCLIP safetensors files directly, without key
//! remapping. Supports global average pooling over patch tokens or
//! CLS-token pooling, selected by `global_average_pool`.
//!
//! The attention softmax goes through
//! `crate::attention::attention_softmax`; the fused-QKV `MultiHeadAttention`
//! is the SAME struct [`crate::clip_text`]'s text tower shares (parameterised
//! by an optional causal mask —
//! `crate::attention::MultiHeadAttention::forward`'s doc); every
//! `candle_nn::LayerNorm` is `crate::layer_norm::LayerNorm` (which has its
//! own bias-free/training gradient oracle —
//! `crate::layer_norm::tests::fused_training_path_matches_slow_within_tolerance_fwd_and_bwd`);
//! and `quick_gelu` is the single shared `crate::activations::quick_gelu`.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{conv2d_no_bias, Conv2d, Conv2dConfig, VarBuilder, VarMap};
use jammi_lora::{FrozenBase, LoraBuildConfig, MaybeLoraLinear};

use crate::attention::MultiHeadAttention;
use crate::error::EncoderError;
use crate::layer_norm::LayerNorm;
use crate::lora_site::{FrozenSiteHolder, LoraSite};

/// Default CLIP normalization constants (used when preprocess_cfg is absent).
#[allow(clippy::excessive_precision)]
const DEFAULT_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
#[allow(clippy::excessive_precision)]
const DEFAULT_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Configuration for an OpenCLIP vision transformer.
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
    pub fn from_open_clip_config(config: &serde_json::Value) -> Result<Self, EncoderError> {
        let model_cfg = config
            .get("model_cfg")
            .ok_or_else(|| EncoderError::Config("OpenCLIP config missing 'model_cfg'".into()))?;
        let vision_cfg = model_cfg.get("vision_cfg").ok_or_else(|| {
            EncoderError::Config("OpenCLIP config missing 'model_cfg.vision_cfg'".into())
        })?;

        let embed_dim = model_cfg
            .get("embed_dim")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                EncoderError::Config("OpenCLIP config missing 'model_cfg.embed_dim'".into())
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

/// The MLP's two LoRA-wrappable sites, named as the OpenCLIP checkpoint
/// names them — see `crate::clip_text`'s sibling pair (the two towers share
/// the block shape and therefore the site vocabulary).
const C_FC_SITE: &str = "c_fc";
/// See [`C_FC_SITE`].
const C_PROJ_SITE: &str = "c_proj";

/// Feed-forward MLP with QuickGelu activation.
struct Mlp {
    c_fc: MaybeLoraLinear,
    c_proj: MaybeLoraLinear,
}

impl Mlp {
    /// One construction path (`crate::lora_site`'s module doc): bases
    /// resolved exactly as before, then offered to `site`, which declines
    /// every one of them under a `LoraBuildConfig::frozen()` config.
    fn load_with(
        vb: VarBuilder,
        width: usize,
        intermediate_size: usize,
        site: &LoraSite<'_>,
    ) -> Result<Self, EncoderError> {
        let c_fc = candle_nn::linear(width, intermediate_size, vb.pp(C_FC_SITE))?;
        let c_proj = candle_nn::linear(intermediate_size, width, vb.pp(C_PROJ_SITE))?;
        Ok(Self {
            c_fc: site.wrap(FrozenBase::Dense(c_fc), C_FC_SITE, C_FC_SITE)?,
            c_proj: site.wrap(FrozenBase::Dense(c_proj), C_PROJ_SITE, C_PROJ_SITE)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        let x = self.c_fc.forward(x)?;
        let x = crate::activations::quick_gelu(&x)?;
        Ok(self.c_proj.forward(&x)?)
    }

    fn lora_sites(&self) -> [(&'static str, &MaybeLoraLinear); 2] {
        [(C_FC_SITE, &self.c_fc), (C_PROJ_SITE, &self.c_proj)]
    }

    fn lora_sites_mut(&mut self) -> [(&'static str, &mut MaybeLoraLinear); 2] {
        [(C_FC_SITE, &mut self.c_fc), (C_PROJ_SITE, &mut self.c_proj)]
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
    fn load_with(
        vb: VarBuilder,
        width: usize,
        heads: usize,
        mlp_ratio: f64,
        site: &LoraSite<'_>,
    ) -> Result<Self, EncoderError> {
        let intermediate_size = (width as f64 * mlp_ratio) as usize;
        // `with_bias=true`: no `remove_mean=false` (RMSNorm-style) variant
        // exists in this tower's config — see `crate::clip_text`'s sibling
        // note on the same class.
        let ln_1 = LayerNorm::new(width, 1e-5, true, vb.pp("ln_1"))?;
        let attn = MultiHeadAttention::load_with(vb.pp("attn"), width, heads, site)?;
        let ln_2 = LayerNorm::new(width, 1e-5, true, vb.pp("ln_2"))?;
        let mlp = Mlp::load_with(vb.pp("mlp"), width, intermediate_size, site)?;
        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }

    /// Propagates to the attention module (softmax arm AND its two LoRA
    /// sites' dropout), both residual-stream LayerNorms, and the MLP's two
    /// LoRA sites.
    fn set_training(&mut self, training: bool) {
        self.attn.set_training(training);
        self.ln_1.set_training(training);
        self.ln_2.set_training(training);
        for (_, site) in self.mlp.lora_sites_mut() {
            site.set_training(training);
        }
    }

    /// This block's four LoRA sites paired with their names — the single
    /// source of the site→name map every [`OpenClipVisionTransformer`]
    /// traversal walks. Same fixed order as `crate::clip_text`'s twin.
    fn lora_sites(&self) -> [(&'static str, &MaybeLoraLinear); 4] {
        let [in_proj, out_proj] = self.attn.lora_sites();
        let [c_fc, c_proj] = self.mlp.lora_sites();
        [in_proj, out_proj, c_fc, c_proj]
    }

    /// The `&mut` twin of `Self::lora_sites`, same names, same order.
    fn lora_sites_mut(&mut self) -> [(&'static str, &mut MaybeLoraLinear); 4] {
        let [in_proj, out_proj] = self.attn.lora_sites_mut();
        let [c_fc, c_proj] = self.mlp.lora_sites_mut();
        [in_proj, out_proj, c_fc, c_proj]
    }

    /// Unmasked (bidirectional) attention: passes `None` to the shared
    /// [`MultiHeadAttention::forward`] — see that method's doc for why
    /// `None` keeps this tower's op sequence exactly what it was before it
    /// shared the struct with the causally-masked text tower.
    fn forward(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        let residual = x;
        let x = self.ln_1.forward(x)?;
        let x = self.attn.forward(&x, None)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.ln_2.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        Ok((residual + x)?)
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
    /// Wired through [`Self::set_training`]; read by [`Self::is_training`].
    training: bool,
}

impl OpenClipVisionTransformer {
    /// Start a builder with default settings: frozen LoRA config, F32
    /// backbone dtype, no adapter file. See [`OpenClipVisionBuilder`].
    pub fn builder() -> OpenClipVisionBuilder<'static> {
        OpenClipVisionBuilder {
            lora: LoraBuildConfig::frozen(),
            backbone_dtype: DType::F32,
            adapter_file: None,
        }
    }

    /// Load from a VarBuilder scoped to the `visual` prefix, fully frozen
    /// (no LoRA sites installed).
    ///
    /// Expects the VarBuilder to be created with `vb.pp("visual")` so that
    /// weight keys like `conv1.weight`, `class_embedding`, etc. resolve
    /// correctly. Routed through the SAME `Self::load_with` the builder
    /// uses, with a decline-everything site
    /// (`crate::lora_site::FrozenSiteHolder`), so there is one loader rather
    /// than two that could drift — `builder().lora(frozen()).build(..)`
    /// produces a bit-identical tower (asserted by
    /// `tests::builder_frozen_output_bits_equal_load`).
    pub fn load(vb: VarBuilder, config: &OpenClipVisionConfig) -> Result<Self, EncoderError> {
        let holder = FrozenSiteHolder::new();
        Self::load_with(vb.clone(), config, &|_n| holder.site(&vb))
    }

    /// The one loader. `site_for` yields the [`LoraSite`] for block `n`
    /// (already scoped to that block's adapter subtree and carrying
    /// `layer_idx = Some(n)`). This tower has no additive attention mask at
    /// all (its attention is unmasked — `MultiHeadAttention::forward`'s
    /// `None` arm), so unlike `crate::clip_text` it carries no dtype-following
    /// mask to build.
    fn load_with<'a>(
        vb: VarBuilder,
        config: &OpenClipVisionConfig,
        site_for: &dyn Fn(usize) -> LoraSite<'a>,
    ) -> Result<Self, EncoderError> {
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

        let ln_pre = LayerNorm::new(config.width, 1e-5, true, vb.pp("ln_pre"))?;

        let mut blocks = Vec::with_capacity(config.layers);
        for i in 0..config.layers {
            let block = ResidualAttentionBlock::load_with(
                vb.pp(format!("transformer.resblocks.{i}")),
                config.width,
                config.heads,
                config.mlp_ratio,
                &site_for(i),
            )?;
            blocks.push(block);
        }

        let ln_post = LayerNorm::new(config.width, 1e-5, true, vb.pp("ln_post"))?;
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
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor, EncoderError> {
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
        Ok(crate::contiguous_matmul(&pooled, &self.proj)?)
    }

    /// Sets whether attention backward walks through the differentiable
    /// softmax composition (`true`) or the fast fused kernel that truncates
    /// backward (`false`, the default from `load`) — see
    /// `crate::attention::attention_softmax`'s module doc for why the two
    /// arms exist and produce bit-identical eval output on CPU. Also
    /// switches every `LayerNorm` (`ln_pre`, `ln_1`/`ln_2` per block,
    /// `ln_post`) between the eval (no-backward) arm and the differentiable
    /// composed arm — see `crate::layer_norm::LayerNorm`'s module doc for
    /// the identical `BackpropOp::none()` truncation `candle_nn::LayerNorm`'s
    /// own fast path has. Propagates to every block's attention sublayer and
    /// both norms this method doesn't own directly.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        self.ln_pre.set_training(training);
        for block in &mut self.blocks {
            block.set_training(training);
        }
        self.ln_post.set_training(training);
    }

    /// Whether [`Self::set_training`] last set training mode. `false` from
    /// every constructor.
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Trainable tensors across every LoRA-wrapped site. Empty for a fully
    /// frozen tower.
    pub fn trainable_params(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for block in &self.blocks {
            for (_, lin) in block.lora_sites() {
                params.extend(lin.trainable_params());
            }
        }
        params
    }

    /// Named LoRA A/B tensors keyed `resblocks.{n}.{site}.lora_{a,b}` —
    /// the same layout `crate::clip_text`'s twin uses, `{site}` being the
    /// checkpoint's own leaf name (`in_proj`, `out_proj`, `c_fc`, `c_proj`).
    /// Note the key is NOT prefixed with `visual.`: an adapter file belongs
    /// to ONE tower (`jammi_lora::Tower` records which), so the tower
    /// prefix would be redundant in every key.
    pub fn named_trainable_weights(&self) -> Result<HashMap<String, Tensor>, EncoderError> {
        let mut out = HashMap::new();
        for (n, block) in self.blocks.iter().enumerate() {
            for (site, lin) in block.lora_sites() {
                out.extend(lin.named_weights(&format!("resblocks.{n}.{site}"))?);
            }
        }
        Ok(out)
    }

    /// Restore LoRA A/B tensors from a [`Self::named_trainable_weights`]-shaped
    /// map. Missing keys are no-ops.
    pub fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<(), EncoderError> {
        for (n, block) in self.blocks.iter_mut().enumerate() {
            for (site, lin) in block.lora_sites_mut() {
                lin.load_weights(weights, &format!("resblocks.{n}.{site}"));
            }
        }
        Ok(())
    }

    /// Per-site dropout-stream positions keyed `{site}.dropout` under the
    /// same site names [`Self::named_trainable_weights`] uses.
    pub fn dropout_positions(&self) -> Result<HashMap<String, u64>, EncoderError> {
        let mut out = HashMap::new();
        for (n, block) in self.blocks.iter().enumerate() {
            for (site, lin) in block.lora_sites() {
                lin.collect_dropout_position(&format!("resblocks.{n}.{site}"), &mut out)?;
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
        for (n, block) in self.blocks.iter().enumerate() {
            for (site, lin) in block.lora_sites() {
                lin.restore_dropout_position(&format!("resblocks.{n}.{site}"), positions)?;
            }
        }
        Ok(())
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

/// Fluent builder for [`OpenClipVisionTransformer`]. Created via
/// [`OpenClipVisionTransformer::builder`], on the same shape
/// [`crate::Bert`]'s and [`crate::ClipText`]'s builders use.
pub struct OpenClipVisionBuilder<'a> {
    lora: LoraBuildConfig<'a>,
    backbone_dtype: DType,
    adapter_file: Option<&'a Path>,
}

impl<'a> OpenClipVisionBuilder<'a> {
    /// LoRA adapter configuration: which of this tower's sites (`in_proj`,
    /// `out_proj`, `c_fc`, `c_proj`, or `all-linear`) get wrapped, at what
    /// rank.
    pub fn lora(mut self, l: LoraBuildConfig<'a>) -> Self {
        self.lora = l;
        self
    }

    /// Dtype the frozen backbone tensors are materialised at. LoRA A/B
    /// always live in F32.
    pub fn backbone_dtype(mut self, d: DType) -> Self {
        self.backbone_dtype = d;
        self
    }

    /// Optional safetensors file to load already-trained LoRA A/B tensors
    /// from (inference). When `None`, A/B tensors are registered in the
    /// caller-supplied `VarMap` for training.
    pub fn adapter(mut self, p: Option<&'a Path>) -> Self {
        self.adapter_file = p;
        self
    }

    /// Materialise the tower from frozen safetensors checkpoint files.
    ///
    /// `weights_paths` are the CHECKPOINT ROOT files; this builder applies
    /// the `visual` scoping itself, matching
    /// [`OpenClipVisionTransformer::load`]'s documented contract that its
    /// `VarBuilder` is already `vb.pp("visual")`. The trainable
    /// `VarBuilder` is NOT `visual`-scoped: an adapter file belongs to one
    /// tower, so its keys start at `resblocks.{n}` (see
    /// [`OpenClipVisionTransformer::named_trainable_weights`]).
    pub fn build(
        self,
        weights_paths: &[&Path],
        config: &OpenClipVisionConfig,
        device: &Device,
        varmap: &VarMap,
    ) -> Result<OpenClipVisionTransformer, EncoderError> {
        let frozen_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(weights_paths, self.backbone_dtype, device)?
        };
        let trainable_vb = if let Some(adapter) = self.adapter_file {
            unsafe { VarBuilder::from_mmaped_safetensors(&[adapter], DType::F32, device)? }
        } else {
            VarBuilder::from_varmap(varmap, DType::F32, device)
        };
        // One trainable sub-builder per block, materialised UP FRONT so each
        // site's adapter key is `resblocks.{n}.{site}.lora_{a,b}` — the exact
        // keys `named_trainable_weights` writes.
        let block_vbs: Vec<VarBuilder> = (0..config.layers)
            .map(|n| trainable_vb.pp(format!("resblocks.{n}")))
            .collect();
        OpenClipVisionTransformer::load_with(frozen_vb.pp("visual"), config, &|n| LoraSite {
            lora_vb: &block_vbs[n],
            layer_idx: Some(n),
            lora: &self.lora,
            varmap,
        })
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
    /// 143-146), which is what `crate::attention::MultiHeadAttention::load`
    /// uses for `in_proj_weight`/`in_proj_bias`. Left at that default, Q, K,
    /// and V are all identically the zero tensor, which zeros the *true*
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
    /// node before it reaches Q/K (see
    /// [`crate::attention::attention_softmax`]'s module doc). Beyond the
    /// Q/K/V slice check, a BLANKET loop over every `Var` in the `VarMap`
    /// asserts the whole tower — not a hand-picked subset — receives a
    /// `Some`/finite/nonzero gradient (measured 32/32 on this fixture, no
    /// exclusions: this tower has no non-differentiable buffer like
    /// HTSAT's `BatchNorm` running stats).
    #[test]
    fn training_true_backward_reaches_qk_through_softmax() {
        // Full-tower forward through `ln_1`/`ln_2`/`ln_pre`/`ln_post`
        // (biased, training mode) bumps
        // `crate::layer_norm::LN_DISPATCH_COUNTERS` even though this test
        // never reads that counter itself (see
        // `crate::layer_norm::DISPATCH_COUNTER_TEST_LOCK`'s doc).
        let _guard = crate::layer_norm::DISPATCH_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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

        let in_proj_weight = model.blocks[0].attn.in_proj_weight();
        let grad = grads
            .get(in_proj_weight)
            .expect("in_proj_weight must have a gradient under training=true");

        let q_norm = row_slice_norm(grad, 0, width);
        let k_norm = row_slice_norm(grad, width, width);
        let v_norm = row_slice_norm(grad, 2 * width, width);

        crate::test_support::assert_finite_nonzero(q_norm, "Q slice");
        crate::test_support::assert_finite_nonzero(k_norm, "K slice");
        crate::test_support::assert_finite_nonzero(v_norm, "V slice (positive control)");

        crate::test_support::assert_every_var_has_gradient(&varmap, &grads, &[]);
    }

    /// Documents the measured full-model shape at `training = false` (the
    /// default from `load`): `in_proj_weight` gets no gradient entry at
    /// all — not even on the V slice. This is *not* the softmax site's own
    /// signature in isolation (that is
    /// `multi_head_attention_eval_zeros_qk_leaves_v_nonzero` below); at the
    /// full-model level, `ln_pre`'s own eval arm truncates backward there,
    /// ahead of every block, before `MultiHeadAttention::forward` ever runs.
    /// Asserting the honest, measured full-model behavior — rather than the
    /// softmax site's shape in isolation — also catches a dropped
    /// propagation call (e.g. `set_training` failing to reach a block would
    /// not change this assertion, since eval is `training = false` either
    /// way, but a regression that accidentally made eval reach the softmax
    /// site would still show up as a change here). A BLANKET loop over
    /// every `Var` in the `VarMap` proves this is the whole tower's
    /// behavior, not just `in_proj_weight`'s.
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

        let in_proj_weight = model.blocks[0].attn.in_proj_weight();
        assert!(
            grads.get(in_proj_weight).is_none(),
            "eval's own LayerNorm kernel must still truncate backward before block 0's \
             attention runs, matching pre-fix behavior for this tower's eval path"
        );

        // EXCLUDED: `visual.proj` (`crate::contiguous_matmul(&pooled,
        // &self.proj)` in `OpenClipVisionTransformer::forward`) — it sits
        // DOWNSTREAM of `ln_pre`'s truncation, applied by a plain
        // differentiable matmul directly to `ln_post`'s output, so it
        // still receives its own gradient (matmul backward for one
        // operand only needs the OTHER operand's forward value, not a
        // walk back through it) even though everything upstream of
        // `ln_pre` is severed.
        crate::test_support::assert_every_var_grad_is_none(&varmap, &grads, &["visual.proj"]);
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

        let out = attn.forward(&x, None).unwrap();
        let weights = fixed_nonuniform_weights(out.dims(), &device);
        let loss = (out * weights).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();

        let grad = grads.get(attn.in_proj_weight()).expect(
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
        crate::test_support::assert_finite_nonzero(v_norm, "V slice");
    }

    /// Deletion-catching oracle for [`ResidualAttentionBlock`]'s
    /// residual-stream LayerNorms themselves (`ln_1`/`ln_2`, block 0):
    /// every gradient assertion above reaches its target parameter through
    /// the block's residual bypass (`x + attn`, `x + mlp_out` in
    /// [`ResidualAttentionBlock::forward`]), so a dropped
    /// `self.ln_1.set_training(training)` / `self.ln_2.set_training(training)`
    /// line — leaving that ONE LayerNorm stuck on its fused,
    /// `BackpropOp::none()`-truncated eval arm even with the rest of the
    /// tower `training=true` — would NOT be caught by any test above: the
    /// residual path still carries a gradient to `in_proj_weight`/
    /// `conv1.weight` regardless of `ln_1`/`ln_2`'s own truncation. This
    /// test asserts `ln_1`/`ln_2`'s OWN `weight` — not anything upstream —
    /// through the full public `forward`: `Some`/finite/nonzero under
    /// `training=true`, `None` under `training=false` (`ln_pre` already
    /// truncates backward ahead of every block under eval — see
    /// `training_false_backward_has_no_in_proj_gradient_at_all`'s doc — so
    /// the eval half of this assertion holds independent of `ln_1`/`ln_2`'s
    /// own gate; the training=true half is what a dropped propagation line
    /// actually flips). RED-verified: deleting
    /// `self.ln_1.set_training(training)` from
    /// `ResidualAttentionBlock::set_training` flips the training=true half
    /// of this test (`ln_1.weight` comes back `None` instead of `Some`)
    /// while every other test in this file stays green.
    #[test]
    fn ln_1_and_ln_2_own_weight_gradient_present_under_training_absent_under_eval() {
        let config = tiny_config();
        let device = Device::Cpu;

        for suffix in [
            "transformer.resblocks.0.ln_1.weight",
            "transformer.resblocks.0.ln_2.weight",
        ] {
            let training_grad = {
                // Full-tower forward at `training=true` bumps
                // `crate::layer_norm::LN_DISPATCH_COUNTERS` even though this
                // block never reads it — same lock discipline as the other
                // training-forward tests in this module.
                let _guard = crate::layer_norm::DISPATCH_COUNTER_TEST_LOCK
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
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
                let var = crate::test_support::find_var(&varmap, suffix);
                grads.get(var.as_tensor()).cloned()
            };
            let grad = training_grad
                .unwrap_or_else(|| panic!("{suffix} grad must be Some under training=true"));
            let norm = grad
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                .sqrt();
            crate::test_support::assert_finite_nonzero(norm, &format!("{suffix} (training=true)"));

            let eval_grad = {
                let varmap = VarMap::new();
                let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
                let model = OpenClipVisionTransformer::load(vb.pp("visual"), &config).unwrap();
                break_zero_init_symmetry(&varmap, &device);
                // training defaults to false; forward without calling set_training.

                let input = fixed_pixel_values(&device);
                let output = model.forward(&input).unwrap();
                let weights = fixed_nonuniform_weights(output.dims(), &device);
                let loss = (output * weights).unwrap().sum_all().unwrap();
                let grads = loss.backward().unwrap();
                let var = crate::test_support::find_var(&varmap, suffix);
                grads.get(var.as_tensor()).cloned()
            };
            assert!(
                eval_grad.is_none(),
                "{suffix} grad must be None under training=false"
            );
        }
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
}
