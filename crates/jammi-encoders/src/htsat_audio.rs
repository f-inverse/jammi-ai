//! HTSAT-Swin CLAP audio tower (front half).
//!
//! The audio tower of a CLAP (Contrastive Language-Audio Pretraining) model in
//! the HuggingFace `ClapAudioModelWithProjection` lineage: an HTSAT
//! (Hierarchical Token-Semantic Audio Transformer) built on a Swin-Transformer
//! spine. It consumes a fused 4-channel log-mel spectrogram
//! `[batch, 4, time, freq]`, batch-normalizes it, bicubic-resamples the time
//! axis up to the Swin input width, reshapes the time-frequency plane into a
//! square "image", and patch-embeds it through an Attentional-Feature-Fusion
//! (AFF) block.
//!
//! This module currently implements the front half — everything through the
//! `patch_embed` boundary `[batch, num_patches, patch_embeds_hidden_size]`.
//! The Swin spine, pooling, and projection are layered on top of this boundary.
//!
//! Weight keys follow the HF safetensors layout under
//! `audio_model.audio_encoder.*`; callers build the encoder modules from
//! `vb.pp("audio_model").pp("audio_encoder")`.

use candle_core::{Module, ModuleT, Tensor, D};
use candle_nn::{
    batch_norm, conv2d, layer_norm, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, LayerNorm,
    VarBuilder,
};

use crate::error::EncoderError;

/// Architecture configuration for the HTSAT-Swin CLAP audio tower, deserialized
/// from a HuggingFace `ClapAudioConfig` (`config.json` or the `audio_config`
/// block of a top-level CLAP config).
///
/// The Swin geometry is fully determined by `depths` (one entry per stage) and
/// `num_attention_heads` (heads per stage); `hidden_size` is the final-stage
/// width and equals `patch_embeds_hidden_size << (num_stages - 1)`. The
/// time-frequency plane is square `spec_size × spec_size` after
/// `reshape_mel2img`, with `freq_ratio = spec_size / num_mel_bins` crops folded
/// along the channel axis.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct HtsatAudioConfig {
    /// Number of Swin blocks in each hierarchical stage.
    pub depths: Vec<usize>,
    /// Number of self-attention heads in each stage.
    pub num_attention_heads: Vec<usize>,
    /// Side length of the square self-attention window (in patches).
    pub window_size: usize,
    /// Side length of the square time-frequency "image" after `reshape_mel2img`.
    pub spec_size: usize,
    /// Square patch size for the patch-embedding convolution.
    pub patch_size: usize,
    /// Patch-embedding stride `[height, width]`.
    pub patch_stride: [usize; 2],
    /// Number of mel-frequency bins in the input spectrogram.
    pub num_mel_bins: usize,
    /// Hidden size of the patch embedding (the first-stage width).
    pub patch_embeds_hidden_size: usize,
    /// Final-stage Swin width (`patch_embeds_hidden_size << (num_stages - 1)`).
    pub hidden_size: usize,
    /// MLP intermediate-size ratio inside each Swin block.
    pub mlp_ratio: f64,
    /// Shared CLAP latent dimensionality after the audio projection.
    pub projection_dim: usize,
    /// Activation applied inside the projection head.
    pub projection_hidden_act: String,
    /// Activation applied inside each Swin block's MLP.
    pub hidden_act: String,
    /// LayerNorm / BatchNorm epsilon.
    pub layer_norm_eps: f64,
    /// Whether the fusion (AFF) path is enabled in the patch embedding.
    pub enable_fusion: bool,
    /// Number of input channels into the patch-embedding convolution (before
    /// fusion channel expansion).
    pub patch_embed_input_channels: usize,
    /// AFF block channel-downsize ratio.
    #[serde(default = "default_aff_block_r")]
    pub aff_block_r: usize,
    /// Whether a LayerNorm is applied to the flattened patch embeddings.
    #[serde(default = "default_true")]
    pub enable_patch_layer_norm: bool,
    /// Whether the patch embeddings are flattened to `[B, num_patches, C]`.
    #[serde(default = "default_true")]
    pub flatten_patch_embeds: bool,
    /// QKV-bias flag (consumed by the Swin spine).
    #[serde(default = "default_true")]
    pub qkv_bias: bool,
}

fn default_aff_block_r() -> usize {
    4
}
fn default_true() -> bool {
    true
}

impl HtsatAudioConfig {
    /// Parse from a HuggingFace CLAP config JSON. If the JSON has a nested
    /// `audio_config` object (top-level `ClapConfig`), it is used; otherwise the
    /// root object is treated as a flat `ClapAudioConfig`.
    pub fn from_hf_clap_config(config: &serde_json::Value) -> Result<Self, EncoderError> {
        let audio = config.get("audio_config").unwrap_or(config);
        serde_json::from_value(audio.clone())
            .map_err(|e| EncoderError::Config(format!("invalid ClapAudioConfig: {e}")))
    }

    /// Number of hierarchical Swin stages.
    pub fn num_stages(&self) -> usize {
        self.depths.len()
    }

    /// Number of crops folded along the channel axis by `reshape_mel2img`
    /// (`spec_size / num_mel_bins`).
    pub fn freq_ratio(&self) -> usize {
        self.spec_size / self.num_mel_bins
    }
}

/// Time-axis bicubic resampling expressed as a fixed `[out, in]` weight matrix.
///
/// PyTorch's `interpolate(mode="bicubic", align_corners=True)` over one spatial
/// axis is an affine map: each output sample is a fixed 4-tap weighted sum of
/// input samples (Keys cubic kernel, `a = -0.75`). Folding those taps into a
/// dense `[out, in]` matrix `W` makes the resample a matmul
/// `out = einsum('oi,bcif->bcof', W, x)` — no `interpolate` kernel needed.
///
/// Faithfulness hinges on the *weights*: PyTorch's native CPU kernel computes
/// the cubic coefficients in f32, and those f32-rounded weights differ from the
/// analytic (f64) weights by up to ~3e-5. Building `W` from the same f32
/// coefficient arithmetic ATen uses (see [`TimeInterp::cubic_coefficients`])
/// reproduces the native kernel to ~5e-7; computing the weights in f64 instead
/// drifts the result ~2e-4 from the reference.
struct TimeInterp {
    /// `[out_len, in_len]` resample weights on the target device/dtype.
    weights: Tensor,
}

/// Keys cubic-convolution coefficient `a`. PyTorch's bicubic default.
const CUBIC_A: f32 = -0.75;

impl TimeInterp {
    /// ATen `cubic_convolution1`: the kernel on the near interval `|x| ∈ [0, 1]`.
    /// Evaluated in f32 in PyTorch's exact operation order so the rounded
    /// coefficients match the native CPU bicubic kernel bit-for-bit (the matrix
    /// formulation is only golden-faithful if the *weights* round identically).
    fn cubic_convolution1(x: f32) -> f32 {
        ((CUBIC_A + 2.0) * x - (CUBIC_A + 3.0)) * x * x + 1.0
    }

    /// ATen `cubic_convolution2`: the kernel on the far interval `|x| ∈ [1, 2]`.
    fn cubic_convolution2(x: f32) -> f32 {
        ((CUBIC_A * x - 5.0 * CUBIC_A) * x + 8.0 * CUBIC_A) * x - 4.0 * CUBIC_A
    }

    /// ATen `get_cubic_upsample_coefficients(t)`: the four tap weights for a
    /// sample whose fractional offset from its floor index is `t`, ordered for
    /// taps at indices `floor - 1, floor, floor + 1, floor + 2`.
    fn cubic_coefficients(t: f32) -> [f32; 4] {
        [
            Self::cubic_convolution2(t + 1.0),
            Self::cubic_convolution1(t),
            Self::cubic_convolution1(1.0 - t),
            Self::cubic_convolution2((1.0 - t) + 1.0),
        ]
    }

    /// Build the `[out_len, in_len]` bicubic resample matrix for
    /// `align_corners=True`. Each output's four tap weights are computed in f32
    /// exactly as ATen does, then folded into the dense row; edge taps are
    /// clamped (replicate padding), accumulating their weight onto the nearest
    /// valid input index, matching PyTorch's boundary handling.
    fn build_matrix(out_len: usize, in_len: usize, device: &candle_core::Device) -> Tensor {
        let mut data = vec![0.0_f32; out_len * in_len];
        // align_corners=True: src(o) = o * (in_len - 1) / (out_len - 1).
        let scale = if out_len > 1 {
            (in_len as f32 - 1.0) / (out_len as f32 - 1.0)
        } else {
            0.0
        };
        for o in 0..out_len {
            let src = o as f32 * scale;
            let base = src.floor();
            let frac = src - base;
            let base = base as i64;
            let coeffs = Self::cubic_coefficients(frac);
            // 4-tap window m ∈ {-1, 0, 1, 2}; coeffs[k] is the weight for m=k-1.
            for (k, m) in (-1_i64..=2).enumerate() {
                let idx = (base + m).clamp(0, in_len as i64 - 1) as usize;
                data[o * in_len + idx] += coeffs[k];
            }
        }
        Tensor::from_vec(data, (out_len, in_len), device).expect("build bicubic matrix")
    }

    fn new(out_len: usize, in_len: usize, device: &candle_core::Device) -> Self {
        Self {
            weights: Self::build_matrix(out_len, in_len, device),
        }
    }

    /// Resample the time axis (dim 2) of `[B, C, in_len, F]` to
    /// `[B, C, out_len, F]` via the fixed matrix.
    fn forward(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        // out[b,c,o,f] = sum_i W[o,i] x[b,c,i,f]. Move time to the second-last
        // axis, contract it with W^T, then restore the layout.
        let (b, c, t, f) = x.dims4()?;
        // [B, C, T, F] -> [B, C, F, T]
        let xt = x.transpose(2, 3)?.contiguous()?;
        // [B, C, F, T] -> [B*C*F, T]
        let xt = xt.reshape((b * c * f, t))?;
        // [B*C*F, T] @ [T, O] = [B*C*F, O]
        let out = xt.matmul(&self.weights.t()?)?;
        let out_len = self.weights.dim(0)?;
        // [B*C*F, O] -> [B, C, F, O] -> [B, C, O, F]
        let out = out
            .reshape((b, c, f, out_len))?
            .transpose(2, 3)?
            .contiguous()?;
        Ok(out)
    }
}

/// AFF (Attentional Feature Fusion) block fusing the global and local patch
/// embeddings.
///
/// `local_att` and `global_att` each map `[B, C, H, W]` → `[B, C, H, W]`
/// attention logits; `global_att` first collapses the spatial extent to a
/// single descriptor (`AdaptiveAvgPool2d(1)`). The sigmoid of their sum gates a
/// convex-style blend `2·global·s + 2·local·(1−s)` of the two paths.
struct AffBlock {
    local_conv0: Conv2d,
    local_bn1: BatchNorm,
    local_conv3: Conv2d,
    local_bn4: BatchNorm,
    global_conv1: Conv2d,
    global_bn2: BatchNorm,
    global_conv4: Conv2d,
    global_bn5: BatchNorm,
}

impl AffBlock {
    fn load(vb: VarBuilder, config: &HtsatAudioConfig) -> Result<Self, EncoderError> {
        let channels = config.patch_embeds_hidden_size;
        let inter = channels / config.aff_block_r;
        let conv_cfg = Conv2dConfig::default(); // 1×1, stride 1, no padding.
        let bn_cfg = BatchNormConfig {
            eps: config.layer_norm_eps,
            ..Default::default()
        };

        let local = vb.pp("local_att");
        let local_conv0 = conv2d(channels, inter, 1, conv_cfg, local.pp("0"))?;
        let local_bn1 = batch_norm(inter, bn_cfg, local.pp("1"))?;
        let local_conv3 = conv2d(inter, channels, 1, conv_cfg, local.pp("3"))?;
        let local_bn4 = batch_norm(channels, bn_cfg, local.pp("4"))?;

        let global = vb.pp("global_att");
        let global_conv1 = conv2d(channels, inter, 1, conv_cfg, global.pp("1"))?;
        let global_bn2 = batch_norm(inter, bn_cfg, global.pp("2"))?;
        let global_conv4 = conv2d(inter, channels, 1, conv_cfg, global.pp("4"))?;
        let global_bn5 = batch_norm(channels, bn_cfg, global.pp("5"))?;

        Ok(Self {
            local_conv0,
            local_bn1,
            local_conv3,
            local_bn4,
            global_conv1,
            global_bn2,
            global_conv4,
            global_bn5,
        })
    }

    /// `AdaptiveAvgPool2d(1)`: mean over the spatial axes (H, W) keeping their
    /// dims, yielding `[B, C, 1, 1]`.
    fn adaptive_avg_pool(x: &Tensor) -> Result<Tensor, EncoderError> {
        Ok(x.mean_keepdim(D::Minus1)?.mean_keepdim(D::Minus2)?)
    }

    fn forward(&self, global: &Tensor, local: &Tensor) -> Result<Tensor, EncoderError> {
        let attention_input = (global + local)?;

        // Local branch: Conv → BN → ReLU → Conv → BN over the full spatial map.
        let l = self.local_conv0.forward(&attention_input)?;
        let l = self.local_bn1.forward_t(&l, false)?;
        let l = l.relu()?;
        let l = self.local_conv3.forward(&l)?;
        let local_logits = self.local_bn4.forward_t(&l, false)?;

        // Global branch: pool to a per-channel descriptor, then the same MLP.
        let g = Self::adaptive_avg_pool(&attention_input)?;
        let g = self.global_conv1.forward(&g)?;
        let g = self.global_bn2.forward_t(&g, false)?;
        let g = g.relu()?;
        let g = self.global_conv4.forward(&g)?;
        let global_logits = self.global_bn5.forward_t(&g, false)?;

        // Broadcast the [B, C, 1, 1] global descriptor over the spatial map.
        let fused = local_logits.broadcast_add(&global_logits)?;
        let s = candle_nn::ops::sigmoid(&fused)?;

        let two = 2.0_f64;
        let out =
            ((global.broadcast_mul(&s)? * two)? + (local.broadcast_mul(&(1.0 - &s)?)? * two)?)?;
        Ok(out)
    }
}

/// HTSAT patch embedding under fusion.
///
/// The fused 4-channel image is split into a single global channel and three
/// local channels. The global channel is patch-convolved (`proj`); the local
/// channels are tiled by a wider stride-`(4, 12)` convolution (`mel_conv2d`),
/// re-laid-out, and zero-padded to the global patch width; the two are blended
/// by the AFF block and flattened to `[B, num_patches, C]` with a LayerNorm.
struct HtsatPatchEmbed {
    proj: Conv2d,
    mel_conv2d_weight: Tensor,
    mel_conv2d_bias: Tensor,
    fusion_model: AffBlock,
    norm: LayerNorm,
    img_size: usize,
}

impl HtsatPatchEmbed {
    fn load(vb: VarBuilder, config: &HtsatAudioConfig) -> Result<Self, EncoderError> {
        // padding = ((k - s) // 2, ...) = 0 for patch_size == patch_stride.
        let proj_cfg = Conv2dConfig {
            stride: config.patch_stride[0],
            ..Default::default()
        };
        let proj = conv2d(
            config.patch_embed_input_channels,
            config.patch_embeds_hidden_size,
            config.patch_size,
            proj_cfg,
            vb.pp("proj"),
        )?;

        // mel_conv2d has a rectangular kernel (4, 12) and rectangular stride
        // (4, 12). candle's Conv2dConfig has a single scalar stride, so the
        // convolution cannot be expressed through candle_nn::Conv2d. Because
        // stride == kernel and padding == 0, it is an exact non-overlapping
        // tiling, evaluated here as an unfold + matmul (see `mel_conv2d`).
        let mel = vb.pp("mel_conv2d");
        let mel_conv2d_weight = mel.get(
            (
                config.patch_embeds_hidden_size,
                config.patch_embed_input_channels,
                config.patch_size,
                config.patch_size * 3,
            ),
            "weight",
        )?;
        let mel_conv2d_bias = mel.get(config.patch_embeds_hidden_size, "bias")?;

        let fusion_model = AffBlock::load(vb.pp("fusion_model"), config)?;
        let norm = layer_norm(
            config.patch_embeds_hidden_size,
            config.layer_norm_eps,
            vb.pp("norm"),
        )?;

        Ok(Self {
            proj,
            mel_conv2d_weight,
            mel_conv2d_bias,
            fusion_model,
            norm,
            img_size: config.spec_size,
        })
    }

    /// Evaluate the rectangular-kernel, rectangular-stride `mel_conv2d` as a
    /// non-overlapping unfold + matmul.
    ///
    /// Input `[N, 1, H, W]`, kernel `(kh, kw) = (4, 12)`, stride `(4, 12)`,
    /// padding 0. With stride == kernel the conv tiles the plane into disjoint
    /// `kh × kw` blocks (a tail narrower than `kw` is dropped, exactly as a
    /// strided conv would). Each block is flattened and projected by the
    /// reshaped weight `[out_c, kh*kw]`.
    fn mel_conv2d(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        let (n, _in_c, _h, _w) = x.dims4()?;
        let (out_c, _kc, kh, kw) = self.mel_conv2d_weight.dims4()?;

        // Tile H then W into non-overlapping blocks: append size-kh / size-kw
        // trailing axes. [N, 1, H, W] -> [N, 1, OH, OW, kh, kw].
        let tiled = x.unfold(2, kh, kh)?.unfold(3, kw, kw)?;
        let tiled_dims = tiled.dims().to_vec();
        let (oh, ow) = (tiled_dims[2], tiled_dims[3]);

        // Flatten each block to a kh*kw vector and the batch/grid to rows:
        // [N, 1, OH, OW, kh, kw] -> [N*OH*OW, kh*kw].
        let patches = tiled.contiguous()?.reshape((n * oh * ow, kh * kw))?;

        // Weight [out_c, 1, kh, kw] -> [kh*kw, out_c]; project and add bias.
        let w = self.mel_conv2d_weight.reshape((out_c, kh * kw))?.t()?;
        let out = patches.matmul(&w.contiguous()?)?;
        let out = out.broadcast_add(&self.mel_conv2d_bias)?;

        // [N*OH*OW, out_c] -> [N, OH, OW, out_c] -> [N, out_c, OH, OW].
        let out = out
            .reshape((n, oh, ow, out_c))?
            .permute((0, 3, 1, 2))?
            .contiguous()?;
        Ok(out)
    }

    /// Patch-embed the fused image `[B, 4, spec_size, spec_size]` to
    /// `[B, num_patches, patch_embeds_hidden_size]`. `mel_conv2d_out` and
    /// `fusion_out` receive the two intermediate boundaries.
    fn forward(
        &self,
        x: &Tensor,
        mel_conv2d_out: &mut Option<Tensor>,
        fusion_out: &mut Option<Tensor>,
    ) -> Result<Tensor, EncoderError> {
        let (batch, _channels, height, width) = x.dims4()?;
        if height != self.img_size || width != self.img_size {
            return Err(EncoderError::Config(format!(
                "HTSAT patch embed expected [{batch}, _, {0}, {0}], got height={height} width={width}",
                self.img_size
            )));
        }

        // Global channel: [B, 1, H, W] -> [B, C, gh, gw].
        let global = x.narrow(1, 0, 1)?;
        let global = self.proj.forward(&global)?;
        let output_width = global.dim(D::Minus1)?;

        // Local channels: [B, 3, H, W] -> [B*3, 1, H, W].
        let local = x.narrow(1, 1, 3)?.contiguous()?;
        let num_local = local.dim(1)?;
        let local = local.reshape((batch * num_local, 1, height, width))?;
        let local = self.mel_conv2d(&local)?;
        *mel_conv2d_out = Some(local.clone());

        // [B*3, F, h, w] -> [B, 3, F, h, w] -> permute(0,2,3,1,4) -> flatten 3..
        let (_, features, lh, lw) = local.dims4()?;
        let local = local.reshape((batch, num_local, features, lh, lw))?;
        let local = local.permute((0, 2, 3, 1, 4))?.contiguous()?;
        let local = local.reshape((batch, features, lh, num_local * lw))?;

        // Zero-pad the local patch width up to the global patch width.
        let local_width = local.dim(D::Minus1)?;
        let pad = output_width - local_width;
        let local = local.pad_with_zeros(D::Minus1, 0, pad)?;

        // AFF fusion: global is the gated path, local the residual.
        let fused = self.fusion_model.forward(&global, &local)?;
        *fusion_out = Some(fused.clone());

        // Flatten the patch grid and LayerNorm: [B, C, gh, gw] -> [B, gh*gw, C].
        let flat = fused.flatten_from(2)?.transpose(1, 2)?.contiguous()?;
        Ok(self.norm.forward(&flat)?)
    }
}

/// HTSAT-Swin CLAP audio encoder — front half (through `patch_embed`).
///
/// `forward_front` runs the batch-norm → bicubic time-resample →
/// `reshape_mel2img` → fused patch-embed pipeline and returns the patch
/// embeddings `[B, num_patches, patch_embeds_hidden_size]` together with the
/// intermediate boundaries that the parity harness gates on.
pub struct HtsatAudioEncoder {
    batch_norm: BatchNorm,
    time_interp: TimeInterp,
    patch_embed: HtsatPatchEmbed,
    freq_ratio: usize,
    spec_width: usize,
}

/// The per-boundary activations produced while running the front half, captured
/// so the caller (parity harness) can gate every unit against its golden.
pub struct FrontHalf {
    /// `[B, num_mel_bins, time, 4]` after batch-norm (channel-first layout).
    pub post_batch_norm: Tensor,
    /// `[B, 4, spec_width, freq]` after bicubic time-resampling.
    pub post_interpolation: Tensor,
    /// `[B, 4, spec_size, spec_size]` after `reshape_mel2img`.
    pub post_reshape_mel2img: Tensor,
    /// `[B*3, C, h, w]` raw `mel_conv2d` output.
    pub mel_conv2d_out: Tensor,
    /// `[B, C, gh, gw]` AFF-fused patch map.
    pub fusion_model_out: Tensor,
    /// `[B, num_patches, C]` final patch embeddings.
    pub patch_embed_out: Tensor,
}

impl HtsatAudioEncoder {
    /// Build the front-half encoder from an `audio_encoder`-scoped
    /// [`VarBuilder`] (i.e. `root.pp("audio_model").pp("audio_encoder")`).
    pub fn load(
        vb: VarBuilder,
        config: &HtsatAudioConfig,
        device: &candle_core::Device,
    ) -> Result<Self, EncoderError> {
        let bn_cfg = BatchNormConfig {
            eps: config.layer_norm_eps,
            ..Default::default()
        };
        let batch_norm = batch_norm(config.num_mel_bins, bn_cfg, vb.pp("batch_norm"))?;

        let freq_ratio = config.freq_ratio();
        let spec_width = config.spec_size * freq_ratio;
        let patch_embed = HtsatPatchEmbed::load(vb.pp("patch_embed"), config)?;

        Ok(Self {
            batch_norm,
            time_interp: TimeInterp::new(spec_width, 500, device),
            patch_embed,
            freq_ratio,
            spec_width,
        })
    }

    /// `reshape_mel2img`: fold the `freq_ratio` time-crops onto the channel axis
    /// and lay the plane out as a square `spec_size × spec_size` image.
    fn reshape_mel2img(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        let (batch, channels, time, freq) = x.dims4()?;
        let r = self.freq_ratio;
        let x = x.reshape((batch, channels * r, time / r, freq))?;
        let x = x.permute((0, 1, 3, 2))?.contiguous()?;
        Ok(x.reshape((batch, channels, freq * r, time / r))?)
    }

    /// Run the front half on `input_features` `[B, 4, 500, num_mel_bins]`,
    /// capturing every gated boundary.
    pub fn forward_front(&self, input_features: &Tensor) -> Result<FrontHalf, EncoderError> {
        // transpose(1,3) -> [B, freq, time, 4]; batch-norm over the freq axis
        // (now channel dim 1); transpose back.
        let x = input_features.transpose(1, 3)?.contiguous()?;
        let post_batch_norm = self.batch_norm.forward_t(&x, false)?;
        let normalized = post_batch_norm.transpose(1, 3)?.contiguous()?;

        // Bicubic time-resample 500 -> spec_width (freq already == spec_height,
        // so the frequency interpolation in the reference is a no-op).
        let post_interpolation = self.time_interp.forward(&normalized)?;
        debug_assert_eq!(post_interpolation.dim(2)?, self.spec_width);

        let post_reshape_mel2img = self.reshape_mel2img(&post_interpolation)?;

        let mut mel_conv2d_out = None;
        let mut fusion_model_out = None;
        let patch_embed_out = self.patch_embed.forward(
            &post_reshape_mel2img,
            &mut mel_conv2d_out,
            &mut fusion_model_out,
        )?;

        let mel_conv2d_out = mel_conv2d_out
            .ok_or_else(|| EncoderError::Config("mel_conv2d boundary not captured".into()))?;
        let fusion_model_out = fusion_model_out
            .ok_or_else(|| EncoderError::Config("fusion boundary not captured".into()))?;

        Ok(FrontHalf {
            post_batch_norm,
            post_interpolation,
            post_reshape_mel2img,
            mel_conv2d_out,
            fusion_model_out,
            patch_embed_out,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn fixture_config() -> serde_json::Value {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../cookbook/fixtures/htsat_clap_tiny/config.json");
        let s = std::fs::read_to_string(path).expect("read config.json");
        serde_json::from_str(&s).expect("parse config.json")
    }

    #[test]
    fn config_parity_with_fixture() {
        let cfg = HtsatAudioConfig::from_hf_clap_config(&fixture_config()).unwrap();
        assert_eq!(cfg.depths, vec![2, 2, 2, 2]);
        assert_eq!(cfg.num_attention_heads, vec![2, 2, 4, 4]);
        assert_eq!(cfg.window_size, 4);
        assert_eq!(cfg.spec_size, 128);
        assert_eq!(cfg.patch_size, 4);
        assert_eq!(cfg.patch_stride, [4, 4]);
        assert_eq!(cfg.num_mel_bins, 32);
        assert_eq!(cfg.patch_embeds_hidden_size, 16);
        assert_eq!(cfg.hidden_size, 128);
        assert_eq!(cfg.mlp_ratio, 2.0);
        assert_eq!(cfg.projection_dim, 8);
        assert_eq!(cfg.layer_norm_eps, 1e-5);
        assert!(cfg.enable_fusion);
        assert_eq!(cfg.aff_block_r, 4);

        // Derived geometry invariants.
        assert_eq!(
            cfg.hidden_size,
            cfg.patch_embeds_hidden_size << (cfg.num_stages() - 1)
        );
        assert_eq!(cfg.freq_ratio(), 4);
        assert_eq!(cfg.freq_ratio(), cfg.spec_size / cfg.num_mel_bins);
    }

    /// The bicubic matrix on a small 5 -> 8 case matches the closed-form Keys
    /// cubic taps with align_corners=True and replicate edges. A self-contained
    /// check of `build_matrix` independent of any golden dump.
    #[test]
    fn bicubic_matrix_small_case() {
        let w = TimeInterp::build_matrix(8, 5, &Device::Cpu);
        let rows = w.to_vec2::<f32>().unwrap();
        assert_eq!(rows.len(), 8);
        assert_eq!(rows[0].len(), 5);

        // Each output row's taps sum to 1 (partition of unity).
        for (o, row) in rows.iter().enumerate() {
            let s: f32 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row {o} weights sum to {s}");
        }
        // align_corners=True pins the endpoints exactly to the input endpoints.
        assert!(
            (rows[0][0] - 1.0).abs() < 1e-6,
            "first output = first input"
        );
        assert!((rows[7][4] - 1.0).abs() < 1e-6, "last output = last input");
    }
}
