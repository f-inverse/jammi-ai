//! HTSAT-Swin CLAP audio tower.
//!
//! The audio tower of a CLAP (Contrastive Language-Audio Pretraining) model in
//! the HuggingFace `ClapAudioModelWithProjection` lineage: an HTSAT
//! (Hierarchical Token-Semantic Audio Transformer) built on a Swin-Transformer
//! spine. It consumes a fused 4-channel log-mel spectrogram
//! `[batch, 4, time, freq]`, batch-normalizes it, bicubic-resamples the time
//! axis up to the Swin input width, reshapes the time-frequency plane into a
//! square "image", and patch-embeds it. The patch embedding is gated per sample
//! by `is_longer`: a longer clip's embedding is the Attentional-Feature-Fusion
//! (AFF) blend of the global patch-conv and the local mel channels, while a
//! short clip uses the global patch-conv alone.
//!
//! This module implements the complete tower: the front half (batch-norm →
//! bicubic time-resample → `reshape_mel2img` → fused patch-embed) through the
//! `patch_embed` boundary, the four-stage Swin spine (W-MSA / SW-MSA blocks with
//! recomputed relative-position bias and shift-window masks, plus patch-merging
//! downsamples), the final LayerNorm and group-2D pooling, and the projection
//! head with L2-normalization producing the shared-latent audio embedding.
//!
//! Weight keys follow the HF safetensors layout. The encoder modules live under
//! `audio_model.audio_encoder.*` (built from
//! `vb.pp("audio_model").pp("audio_encoder")`); the projection head lives at
//! `audio_projection.*`, a sibling of `audio_model`. [`HtsatAudio::load`] takes
//! the safetensors root and wires both.

use candle_core::{IndexOp, Module, ModuleT, Tensor, D};
use candle_nn::{
    batch_norm, conv2d, layer_norm, linear, linear_no_bias, BatchNorm, BatchNormConfig, Conv2d,
    Conv2dConfig, LayerNorm, Linear, VarBuilder,
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
    /// Target time width (`spec_size * freq_ratio`) every input is resampled to.
    /// The `[out_len, in_len]` weight matrix is built per forward from the
    /// input's actual time length, so the tower handles any clip length (e.g.
    /// the tiny fixture's 500 and the real checkpoint's 1001), not a fixed one.
    out_len: usize,
    device: candle_core::Device,
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
    fn build_matrix(
        out_len: usize,
        in_len: usize,
        device: &candle_core::Device,
    ) -> Result<Tensor, EncoderError> {
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
        Ok(Tensor::from_vec(data, (out_len, in_len), device)?)
    }

    fn new(out_len: usize, device: &candle_core::Device) -> Self {
        Self {
            out_len,
            device: device.clone(),
        }
    }

    /// Resample the time axis (dim 2) of `[B, C, T, F]` to `[B, C, out_len, F]`,
    /// building the bicubic matrix for this input's `T` (matching the reference,
    /// which interpolates the time axis up to `spec_width`). When `T == out_len`
    /// no resample is needed — the reference skips it too.
    fn forward(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        let (b, c, t, f) = x.dims4()?;
        if t == self.out_len {
            return Ok(x.clone());
        }
        // out[b,c,o,f] = sum_i W[o,i] x[b,c,i,f], W = [out_len, T] built for this
        // input length. Move time to the second-last axis, contract with W^T.
        let weights = Self::build_matrix(self.out_len, t, &self.device)?;
        // [B, C, T, F] -> [B, C, F, T] -> [B*C*F, T]
        let xt = x.transpose(2, 3)?.contiguous()?.reshape((b * c * f, t))?;
        // [B*C*F, T] @ [T, out_len] = [B*C*F, out_len]
        let out = xt.matmul(&weights.t()?)?;
        // [B*C*F, out_len] -> [B, C, F, out_len] -> [B, C, out_len, F]
        let out = out
            .reshape((b, c, f, self.out_len))?
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
    ///
    /// Fusion is gated per sample by `is_longer` (HF `ClapAudioPatchEmbed`): an
    /// `is_longer=true` sample's patch embedding is the AFF blend of the global
    /// patch-conv and the local `mel_conv2d` channels, while an `is_longer=false`
    /// sample uses the global patch-conv alone. Both paths are computed for the
    /// whole batch and selected per sample by a `[B, 1, 1, 1]` mask, matching
    /// HF's `global_hidden_states[is_longer_idx] = fusion(...)` index-assignment
    /// without per-row scatter.
    fn forward(
        &self,
        x: &Tensor,
        is_longer: &[bool],
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
        if is_longer.len() != batch {
            return Err(EncoderError::Config(format!(
                "HTSAT patch embed: is_longer has {} flags for a batch of {batch}",
                is_longer.len()
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

        // Per-sample select: fused where is_longer, global patch-conv otherwise.
        // Mask is [B, 1, 1, 1] broadcasting over channels and the patch grid.
        let mask: Vec<f32> = is_longer
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect();
        let mask = Tensor::from_vec(mask, (batch, 1, 1, 1), global.device())?;
        let patch_map = mask
            .broadcast_mul(&fused)?
            .add(&(1.0 - &mask)?.broadcast_mul(&global)?)?;

        // Flatten the patch grid and LayerNorm: [B, C, gh, gw] -> [B, gh*gw, C].
        let flat = patch_map.flatten_from(2)?.transpose(1, 2)?.contiguous()?;
        Ok(self.norm.forward(&flat)?)
    }
}

/// `window_partition`: tile `[B, H, W, C]` into non-overlapping `ws × ws`
/// windows, returning `[B*nW, ws, ws, C]`.
fn window_partition(x: &Tensor, ws: usize) -> Result<Tensor, EncoderError> {
    let (b, h, w, c) = x.dims4()?;
    let x = x.reshape((b, h / ws, ws, w / ws, ws, c))?;
    let x = x.permute((0, 1, 3, 2, 4, 5))?.contiguous()?;
    Ok(x.reshape((b * (h / ws) * (w / ws), ws, ws, c))?)
}

/// `window_reverse`: merge `[B*nW, ws, ws, C]` windows back into `[B, H, W, C]`.
fn window_reverse(x: &Tensor, ws: usize, h: usize, w: usize) -> Result<Tensor, EncoderError> {
    let c = x.dim(D::Minus1)?;
    let x = x.reshape((x.dim(0)? / ((h / ws) * (w / ws)), h / ws, w / ws, ws, ws, c))?;
    let x = x.permute((0, 1, 3, 2, 4, 5))?.contiguous()?;
    Ok(x.reshape((x.dim(0)?, h, w, c))?)
}

/// Self-attention inside a Swin window (W-MSA / SW-MSA), with the recomputed
/// relative-position bias and an optional precomputed shift-window mask.
struct SwinSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    /// `[(2·ws−1)², num_heads]` learned relative-position bias table, sized by
    /// the config window (HF sizes the table by `config.window_size`, not the
    /// block's effective window).
    rel_bias_table: Tensor,
    /// `[(ws·ws)²]` flattened relative-position index (U32), recomputed over the
    /// config window.
    rel_index: Tensor,
    num_heads: usize,
    head_size: usize,
}

impl SwinSelfAttention {
    /// `ws` is the config window size (`config.window_size`): HF constructs
    /// `ClapAudioSelfAttention` with `window_size=config.window_size` and never
    /// re-sizes it when a block's effective window is clamped to a smaller grid,
    /// so the bias table and relative-position index are both sized by the config
    /// window. (Token count per window equals the effective window squared, which
    /// coincides with `ws·ws` in every reachable config since the deepest stage's
    /// grid equals the window.)
    fn load(vb: VarBuilder, dim: usize, num_heads: usize, ws: usize) -> Result<Self, EncoderError> {
        let query = linear(dim, dim, vb.pp("query"))?;
        let key = linear(dim, dim, vb.pp("key"))?;
        let value = linear(dim, dim, vb.pp("value"))?;
        let table_rows = (2 * ws - 1) * (2 * ws - 1);
        let rel_bias_table = vb.get((table_rows, num_heads), "relative_position_bias_table")?;
        let rel_index = Self::build_rel_index(ws, vb.device())?;
        Ok(Self {
            query,
            key,
            value,
            rel_bias_table,
            rel_index,
            num_heads,
            head_size: dim / num_heads,
        })
    }

    /// Recompute the pairwise relative-position index over a `ws × ws` window
    /// (verified bit-exact against the stored buffer), flattened to U32 for the
    /// bias-table gather.
    fn build_rel_index(ws: usize, device: &candle_core::Device) -> Result<Tensor, EncoderError> {
        let n = ws * ws;
        // coords_flatten[axis][token], token = h*ws + w.
        let mut idx = vec![0u32; n * n];
        for i in 0..n {
            let (hi, wi) = (i / ws, i % ws);
            for j in 0..n {
                let (hj, wj) = (j / ws, j % ws);
                // relative_coords (permuted to [i, j, axis]): coord_i - coord_j.
                let mut rh = (hi as i64) - (hj as i64);
                let mut rw = (wi as i64) - (wj as i64);
                rh += ws as i64 - 1;
                rw += ws as i64 - 1;
                rh *= 2 * ws as i64 - 1;
                idx[i * n + j] = (rh + rw) as u32;
            }
        }
        Ok(Tensor::from_vec(idx, n * n, device)?)
    }

    /// Split the last dim into heads: `[BnW, L, C]` -> `[BnW, heads, L, head]`.
    fn heads(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        let (bnw, l, _c) = x.dims3()?;
        Ok(x.reshape((bnw, l, self.num_heads, self.head_size))?
            .transpose(1, 2)?
            .contiguous()?)
    }

    /// `hidden`: `[B*nW, L, C]` (`L = ws·ws` tokens per window); `mask`: optional
    /// `[nW, L, L]`; `num_windows` is nW (needed to fold the mask over the batch
    /// axis).
    fn forward(
        &self,
        hidden: &Tensor,
        mask: Option<&Tensor>,
        num_windows: usize,
    ) -> Result<Tensor, EncoderError> {
        let (bnw, l, c) = hidden.dims3()?;
        let q = self.heads(&self.query.forward(hidden)?)?;
        let k = self.heads(&self.key.forward(hidden)?)?;
        let v = self.heads(&self.value.forward(hidden)?)?;

        let scale = 1.0 / (self.head_size as f64).sqrt();
        let scores = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)? * scale)?;

        // bias = rel_bias_table[rel_index][L,L,heads].permute(2,0,1) -> [heads,L,L].
        let bias = self
            .rel_bias_table
            .index_select(&self.rel_index, 0)?
            .reshape((l, l, self.num_heads))?
            .permute((2, 0, 1))?
            .contiguous()?;
        let scores = scores.broadcast_add(&bias.unsqueeze(0)?)?;

        let scores = match mask {
            Some(mask) => {
                // scores [B//nW, nW, heads, L, L] + mask [1, nW, 1, L, L].
                let scores =
                    scores.reshape((bnw / num_windows, num_windows, self.num_heads, l, l))?;
                let mask = mask.unsqueeze(1)?.unsqueeze(0)?;
                scores
                    .broadcast_add(&mask)?
                    .reshape((bnw, self.num_heads, l, l))?
            }
            None => scores,
        };

        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // [BnW, heads, L, head]
        let ctx = ctx.transpose(1, 2)?.contiguous()?.reshape((bnw, l, c))?;
        Ok(ctx)
    }
}

/// A single Swin block (pre-norm, W-MSA or SW-MSA) with the two residual MLP.
struct SwinBlock {
    layernorm_before: LayerNorm,
    attention: SwinSelfAttention,
    attention_output: Linear,
    layernorm_after: LayerNorm,
    intermediate: Linear,
    output: Linear,
    /// `(height, width)` patch resolution this block operates on.
    input_resolution: (usize, usize),
    /// Cyclic shift (0 for W-MSA, window/2 for SW-MSA; forced 0 when the grid is
    /// no larger than the window).
    shift_size: usize,
    /// Effective window size (clamped to the grid when the grid is smaller).
    window_size: usize,
    /// Precomputed `[nW, L, L]` attention mask for SW-MSA, or `None`.
    attn_mask: Option<Tensor>,
}

impl SwinBlock {
    fn load(
        vb: VarBuilder,
        config: &HtsatAudioConfig,
        dim: usize,
        num_heads: usize,
        input_resolution: (usize, usize),
        block_index: usize,
        device: &candle_core::Device,
    ) -> Result<Self, EncoderError> {
        let eps = config.layer_norm_eps;
        let layernorm_before = layer_norm(dim, eps, vb.pp("layernorm_before"))?;
        let attention = SwinSelfAttention::load(
            vb.pp("attention").pp("self"),
            dim,
            num_heads,
            config.window_size,
        )?;
        let attention_output = linear(dim, dim, vb.pp("attention").pp("output").pp("dense"))?;
        let layernorm_after = layer_norm(dim, eps, vb.pp("layernorm_after"))?;
        let inter = (config.mlp_ratio * dim as f64) as usize;
        let intermediate = linear(dim, inter, vb.pp("intermediate").pp("dense"))?;
        let output = linear(inter, dim, vb.pp("output").pp("dense"))?;

        // set_shift_and_window_size: window/2 for odd blocks, forced 0 (with the
        // window clamped to the grid) when the grid is no larger than the window.
        let mut window_size = config.window_size;
        let mut shift_size = if block_index % 2 == 0 {
            0
        } else {
            config.window_size / 2
        };
        if input_resolution.0.min(input_resolution.1) <= config.window_size {
            shift_size = 0;
            window_size = input_resolution.0.min(input_resolution.1);
        }

        let attn_mask = if shift_size > 0 {
            Some(Self::build_attn_mask(
                input_resolution.0,
                input_resolution.1,
                window_size,
                shift_size,
                device,
            )?)
        } else {
            None
        };

        Ok(Self {
            layernorm_before,
            attention,
            attention_output,
            layernorm_after,
            intermediate,
            output,
            input_resolution,
            shift_size,
            window_size,
            attn_mask,
        })
    }

    /// Build the SW-MSA attention mask `[nW, L, L]` from the 9-region label map.
    fn build_attn_mask(
        h: usize,
        w: usize,
        ws: usize,
        shift: usize,
        device: &candle_core::Device,
    ) -> Result<Tensor, EncoderError> {
        // img_mask[1, H, W, 1] labelled by the 3×3 slice regions.
        let region = |i: usize, len: usize| -> usize {
            // slices: (0..len-ws), (len-ws..len-shift), (len-shift..len).
            if i < len - ws {
                0
            } else if i < len - shift {
                1
            } else {
                2
            }
        };
        let mut img = vec![0f32; h * w];
        for hi in 0..h {
            for wi in 0..w {
                img[hi * w + wi] = (region(hi, h) * 3 + region(wi, w)) as f32;
            }
        }
        let img = Tensor::from_vec(img, (1, h, w, 1), device)?;
        let mask_windows = window_partition(&img, ws)?; // [nW, ws, ws, 1]
        let nw = mask_windows.dim(0)?;
        let mask_windows = mask_windows.reshape((nw, ws * ws))?;
        // attn_mask = mask[:, None, :] - mask[:, :, None]
        let a = mask_windows.unsqueeze(1)?; // [nW, 1, L]
        let b = mask_windows.unsqueeze(2)?; // [nW, L, 1]
        let diff = a.broadcast_sub(&b)?; // [nW, L, L]
                                         // (diff != 0) * -100.0
        let mask = (diff.ne(0f32)?.to_dtype(candle_core::DType::F32)? * -100.0)?;
        Ok(mask)
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor, EncoderError> {
        let (b, _l, c) = hidden.dims3()?;
        let (h, w) = self.input_resolution;
        let ws = self.window_size;

        let shortcut = hidden;
        let x = self.layernorm_before.forward(hidden)?;
        let x = x.reshape((b, h, w, c))?;

        // Cyclic shift (two single-dim rolls compose the 2-D torch.roll).
        let x = if self.shift_size > 0 {
            x.roll(-(self.shift_size as i32), 1)?
                .roll(-(self.shift_size as i32), 2)?
        } else {
            x
        };

        let windows = window_partition(&x, ws)?; // [B*nW, ws, ws, C]
        let num_windows = (h / ws) * (w / ws);
        let windows = windows.reshape((b * num_windows, ws * ws, c))?;

        let ctx = self
            .attention
            .forward(&windows, self.attn_mask.as_ref(), num_windows)?;
        let attn = self.attention_output.forward(&ctx)?;

        // window_reverse -> [B, H, W, C].
        let attn = attn.reshape((b * num_windows, ws, ws, c))?;
        let attn = window_reverse(&attn, ws, h, w)?;

        // Reverse cyclic shift.
        let attn = if self.shift_size > 0 {
            attn.roll(self.shift_size as i32, 1)?
                .roll(self.shift_size as i32, 2)?
        } else {
            attn
        };
        let attn = attn.reshape((b, h * w, c))?;

        // Residual 1.
        let hidden = (shortcut + attn)?;

        // MLP with residual 2.
        let y = self.layernorm_after.forward(&hidden)?;
        let y = self.intermediate.forward(&y)?;
        let y = y.gelu_erf()?;
        let y = self.output.forward(&y)?;
        Ok((&hidden + y)?)
    }
}

/// Swin patch-merging downsample: `2×` spatial reduction with a `4C → 2C`
/// linear over the concatenated `2×2` neighbourhood.
struct PatchMerging {
    norm: LayerNorm,
    reduction: Linear,
    input_resolution: (usize, usize),
}

impl PatchMerging {
    fn load(
        vb: VarBuilder,
        config: &HtsatAudioConfig,
        dim: usize,
        input_resolution: (usize, usize),
    ) -> Result<Self, EncoderError> {
        let norm = layer_norm(4 * dim, config.layer_norm_eps, vb.pp("norm"))?;
        let reduction = linear_no_bias(4 * dim, 2 * dim, vb.pp("reduction"))?;
        Ok(Self {
            norm,
            reduction,
            input_resolution,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor, EncoderError> {
        let (b, _l, c) = hidden.dims3()?;
        let (h, w) = self.input_resolution;
        // [B, H, W, C] -> [B, H/2, 2, W/2, 2, C] for strided ::2 slicing.
        let x = hidden.reshape((b, h / 2, 2, w / 2, 2, c))?.contiguous()?;
        // f0=(0,0), f1=(1,0), f2=(0,1), f3=(1,1) on (row-parity, col-parity).
        let pick = |kr: usize, kc: usize| -> Result<Tensor, EncoderError> {
            Ok(x.i((.., .., kr, .., kc, ..))?.contiguous()?)
        };
        let f0 = pick(0, 0)?;
        let f1 = pick(1, 0)?;
        let f2 = pick(0, 1)?;
        let f3 = pick(1, 1)?;
        let cat = Tensor::cat(&[f0, f1, f2, f3], D::Minus1)?; // [B, H/2, W/2, 4C]
        let cat = cat.reshape((b, (h / 2) * (w / 2), 4 * c))?;
        let cat = self.norm.forward(&cat)?;
        Ok(self.reduction.forward(&cat)?)
    }
}

/// One hierarchical Swin stage: `depth` blocks then an optional downsample.
struct SwinStage {
    blocks: Vec<SwinBlock>,
    downsample: Option<PatchMerging>,
}

impl SwinStage {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: VarBuilder,
        config: &HtsatAudioConfig,
        dim: usize,
        num_heads: usize,
        depth: usize,
        input_resolution: (usize, usize),
        has_downsample: bool,
        device: &candle_core::Device,
    ) -> Result<Self, EncoderError> {
        let blocks_vb = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            blocks.push(SwinBlock::load(
                blocks_vb.pp(i),
                config,
                dim,
                num_heads,
                input_resolution,
                i,
                device,
            )?);
        }
        let downsample = if has_downsample {
            Some(PatchMerging::load(
                vb.pp("downsample"),
                config,
                dim,
                input_resolution,
            )?)
        } else {
            None
        };
        Ok(Self { blocks, downsample })
    }
}

/// HTSAT-Swin CLAP audio encoder — front half (through `patch_embed`) plus the
/// full Swin spine, final LayerNorm, and group-2D pooling.
///
/// `forward_front` runs the batch-norm → bicubic time-resample →
/// `reshape_mel2img` → fused patch-embed pipeline and returns the patch
/// embeddings `[B, num_patches, patch_embeds_hidden_size]` together with the
/// intermediate boundaries that the parity harness gates on. `forward_spine`
/// continues from the patch embeddings through to the pooled `[B, hidden_size]`.
pub struct HtsatAudioEncoder {
    batch_norm: BatchNorm,
    time_interp: TimeInterp,
    patch_embed: HtsatPatchEmbed,
    stages: Vec<SwinStage>,
    norm: LayerNorm,
    freq_ratio: usize,
    spec_width: usize,
    /// First-stage patch grid resolution `(height, width)`.
    grid: (usize, usize),
    num_stages: usize,
    patch_stride: [usize; 2],
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

        // First-stage patch grid: spec_size / patch_stride per axis.
        let grid = (
            config.spec_size / config.patch_stride[0],
            config.spec_size / config.patch_stride[1],
        );
        let num_stages = config.num_stages();
        let layers_vb = vb.pp("layers");
        let mut stages = Vec::with_capacity(num_stages);
        for i in 0..num_stages {
            let dim = config.patch_embeds_hidden_size << i;
            let input_resolution = (grid.0 >> i, grid.1 >> i);
            stages.push(SwinStage::load(
                layers_vb.pp(i),
                config,
                dim,
                config.num_attention_heads[i],
                config.depths[i],
                input_resolution,
                i < num_stages - 1,
                device,
            )?);
        }
        let norm = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("norm"))?;

        Ok(Self {
            batch_norm,
            time_interp: TimeInterp::new(spec_width, device),
            patch_embed,
            stages,
            norm,
            freq_ratio,
            spec_width,
            grid,
            num_stages,
            patch_stride: config.patch_stride,
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

    /// Run the front half on `input_features` `[B, 4, T, num_mel_bins]` (any T),
    /// capturing every gated boundary. `is_longer` gates the per-sample fusion in
    /// the patch embedding (`true` → AFF blend, `false` → global patch-conv only).
    pub fn forward_front(
        &self,
        input_features: &Tensor,
        is_longer: &[bool],
    ) -> Result<FrontHalf, EncoderError> {
        // transpose(1,3) -> [B, freq, time, 4]; batch-norm over the freq axis
        // (now channel dim 1); transpose back.
        let x = input_features.transpose(1, 3)?.contiguous()?;
        let post_batch_norm = self.batch_norm.forward_t(&x, false)?;
        let normalized = post_batch_norm.transpose(1, 3)?.contiguous()?;

        // Bicubic time-resample T -> spec_width (freq already == spec_height,
        // so the frequency interpolation in the reference is a no-op).
        let post_interpolation = self.time_interp.forward(&normalized)?;
        debug_assert_eq!(post_interpolation.dim(2)?, self.spec_width);

        let post_reshape_mel2img = self.reshape_mel2img(&post_interpolation)?;

        let mut mel_conv2d_out = None;
        let mut fusion_model_out = None;
        let patch_embed_out = self.patch_embed.forward(
            &post_reshape_mel2img,
            is_longer,
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

    /// Run the Swin spine from the patch embeddings `[B, num_patches, C0]`
    /// through the final LayerNorm and group-2D pooling, capturing every gated
    /// boundary. `frames_num` is the spatial height of `post_reshape_mel2img`
    /// (the post-fold image side fed to `patch_embed`); it drives the pooling
    /// reshape (= `spec_size` for the standard config).
    pub fn forward_spine(
        &self,
        patch_embed_out: &Tensor,
        frames_num: usize,
    ) -> Result<Spine, EncoderError> {
        let mut blocks: Vec<Vec<Tensor>> = Vec::with_capacity(self.num_stages);
        let mut downsamples: Vec<Option<Tensor>> = Vec::with_capacity(self.num_stages);

        let mut hidden = patch_embed_out.clone();
        for stage in &self.stages {
            let mut stage_blocks = Vec::with_capacity(stage.blocks.len());
            for block in &stage.blocks {
                hidden = block.forward(&hidden)?;
                stage_blocks.push(hidden.clone());
            }
            blocks.push(stage_blocks);
            match &stage.downsample {
                Some(ds) => {
                    hidden = ds.forward(&hidden)?;
                    downsamples.push(Some(hidden.clone()));
                }
                None => downsamples.push(None),
            }
        }

        let final_norm_out = self.norm.forward(&hidden)?;

        // Group-2D pooling: permute to channel-first, fold the spatial plane into
        // freq/temporal, regroup by `c_freq_bin`, then adaptive-avg-pool to [B, C].
        let (batch, _l, n_channels) = final_norm_out.dims3()?;
        let pow = 2usize.pow((self.num_stages - 1) as u32);
        let freq_shape = frames_num / pow / self.patch_stride[0];
        let temporal_shape = frames_num / pow / self.patch_stride[1];
        let c_freq_bin = freq_shape / self.freq_ratio;

        let h = final_norm_out.permute((0, 2, 1))?.contiguous()?.reshape((
            batch,
            n_channels,
            freq_shape,
            temporal_shape,
        ))?;
        let h = h.reshape((
            batch,
            n_channels,
            freq_shape / c_freq_bin,
            c_freq_bin,
            temporal_shape,
        ))?;
        let pre_pool = h.permute((0, 1, 3, 2, 4))?.contiguous()?.reshape((
            batch,
            n_channels,
            c_freq_bin,
            freq_shape / c_freq_bin * temporal_shape,
        ))?;

        // AdaptiveAvgPool1d(1) over the flattened spatial tail.
        let pooler_out = pre_pool.flatten_from(2)?.mean(D::Minus1)?;

        Ok(Spine {
            blocks,
            downsamples,
            final_norm_out,
            pre_pool,
            pooler_out,
        })
    }

    /// First-stage patch grid resolution `(height, width)`.
    pub fn grid(&self) -> (usize, usize) {
        self.grid
    }
}

/// The per-boundary activations produced while running the Swin spine, captured
/// so the parity harness can gate every unit against its golden. `blocks[s][b]`
/// is stage `s` block `b`'s output; `downsamples[s]` is stage `s`'s
/// patch-merging output (`None` for the final stage).
pub struct Spine {
    /// `blocks[stage][block]` block outputs `[B, L, C]`.
    pub blocks: Vec<Vec<Tensor>>,
    /// `downsamples[stage]` patch-merging output (`None` for the last stage).
    pub downsamples: Vec<Option<Tensor>>,
    /// `[B, num_patches_final, hidden_size]` after the final LayerNorm.
    pub final_norm_out: Tensor,
    /// `[B, hidden_size, c_freq_bin, *]` regrouped pre-pool tensor.
    pub pre_pool: Tensor,
    /// `[B, hidden_size]` pooled audio descriptor.
    pub pooler_out: Tensor,
}

/// The CLAP audio projection head: `linear1 → act → linear2`, then L2-normalize.
pub struct ClapAudioProjection {
    linear1: Linear,
    linear2: Linear,
    act: String,
}

impl ClapAudioProjection {
    /// Build from a root-scoped [`VarBuilder`] (projection lives at
    /// `audio_projection.*`, a sibling of `audio_model`).
    pub fn load(vb: VarBuilder, config: &HtsatAudioConfig) -> Result<Self, EncoderError> {
        let linear1 = linear(config.hidden_size, config.projection_dim, vb.pp("linear1"))?;
        let linear2 = linear(
            config.projection_dim,
            config.projection_dim,
            vb.pp("linear2"),
        )?;
        Ok(Self {
            linear1,
            linear2,
            act: config.projection_hidden_act.clone(),
        })
    }

    /// Project `[B, hidden_size]` to the unnormalized latent `[B, projection_dim]`.
    pub fn forward_unnormalized(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        let x = self.linear1.forward(x)?;
        let x = match self.act.as_str() {
            "relu" => x.relu()?,
            "gelu" => x.gelu_erf()?,
            other => {
                return Err(EncoderError::Config(format!(
                    "unsupported projection activation '{other}'"
                )))
            }
        };
        Ok(self.linear2.forward(&x)?)
    }
}

/// L2-normalize each row of a `[B, D]` tensor along the last axis
/// (`F.normalize(p=2, dim=-1, eps=1e-12)`).
pub fn l2_normalize(t: &Tensor) -> Result<Tensor, EncoderError> {
    let norm = t
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .clamp(1e-12, f32::MAX)?;
    Ok(t.broadcast_div(&norm)?)
}

/// The full HTSAT-Swin CLAP audio tower: encoder spine + projection head,
/// returning the L2-normalized shared-latent embedding.
pub struct HtsatAudio {
    encoder: HtsatAudioEncoder,
    projection: ClapAudioProjection,
    projection_dim: usize,
    num_mel_bins: usize,
}

impl HtsatAudio {
    /// Build the full tower from a root-scoped [`VarBuilder`] (the safetensors
    /// root holding both `audio_model` and `audio_projection`).
    pub fn load(
        vb: VarBuilder,
        config: &HtsatAudioConfig,
        device: &candle_core::Device,
    ) -> Result<Self, EncoderError> {
        let encoder =
            HtsatAudioEncoder::load(vb.pp("audio_model").pp("audio_encoder"), config, device)?;
        let projection = ClapAudioProjection::load(vb.pp("audio_projection"), config)?;
        Ok(Self {
            encoder,
            projection,
            projection_dim: config.projection_dim,
            num_mel_bins: config.num_mel_bins,
        })
    }

    /// Shared CLAP latent dimensionality of the output (`projection_dim`).
    pub fn projection_dim(&self) -> usize {
        self.projection_dim
    }

    /// Number of mel bins the input fusion spectrogram must carry.
    pub fn num_mel_bins(&self) -> usize {
        self.num_mel_bins
    }

    /// Borrow the underlying encoder (for boundary-level parity checks).
    pub fn encoder(&self) -> &HtsatAudioEncoder {
        &self.encoder
    }

    /// Borrow the projection head (for boundary-level parity checks).
    pub fn projection(&self) -> &ClapAudioProjection {
        &self.projection
    }

    /// Full forward on `input_features` `[B, 4, T, num_mel_bins]` (any T), with
    /// the per-sample `is_longer` fusion gate, returning the L2-normalized audio
    /// embedding `[B, projection_dim]`.
    pub fn forward(
        &self,
        input_features: &Tensor,
        is_longer: &[bool],
    ) -> Result<Tensor, EncoderError> {
        let front = self.encoder.forward_front(input_features, is_longer)?;
        let frames_num = front.post_reshape_mel2img.dim(2)?;
        let spine = self
            .encoder
            .forward_spine(&front.patch_embed_out, frames_num)?;
        let unnorm = self.projection.forward_unnormalized(&spine.pooler_out)?;
        l2_normalize(&unnorm)
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
        let w = TimeInterp::build_matrix(8, 5, &Device::Cpu).unwrap();
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

    /// FIX-1 generality: the Swin self-attention's relative-position bias-table
    /// row count and recomputed index are sized purely by the window size, with
    /// no hardcoded `window=4`. Both the tiny config (window=4 → 49 rows) and the
    /// real `laion/clap-htsat-fused` config (window=8 → 225 rows) must derive
    /// correctly. The index is `ws·ws` tokens squared, with every entry in range
    /// for the `(2·ws−1)²`-row table (HF's `relative_position_index` bound).
    #[test]
    fn rel_pos_table_and_index_are_window_sized() {
        for ws in [4usize, 8] {
            let table_rows = (2 * ws - 1) * (2 * ws - 1);
            assert_eq!(
                table_rows,
                match ws {
                    4 => 49,
                    8 => 225,
                    _ => unreachable!(),
                },
                "ws={ws}: table rows"
            );

            let index = SwinSelfAttention::build_rel_index(ws, &Device::Cpu).unwrap();
            let n = ws * ws;
            assert_eq!(index.dims(), &[n * n], "ws={ws}: index length = (ws·ws)²");
            let max = index.max(0).unwrap().to_scalar::<u32>().unwrap();
            assert!(
                (max as usize) < table_rows,
                "ws={ws}: index max {max} must address within {table_rows} table rows"
            );
            // The self-position (token i to itself) maps to the table centre,
            // index (2·ws−1)·(ws−1) + (ws−1) = 2·(ws−1)·ws, for every token.
            let centre = 2 * (ws - 1) * ws;
            let flat = index.to_vec1::<u32>().unwrap();
            for i in 0..n {
                assert_eq!(
                    flat[i * n + i] as usize,
                    centre,
                    "ws={ws}: token {i} self-index must be the table centre"
                );
            }
        }
    }
}
