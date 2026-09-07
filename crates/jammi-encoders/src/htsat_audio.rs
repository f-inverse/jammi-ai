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
//!
//! # Every fusible activation goes through the house seam
//!
//! This tower evaluates GELU-erf at exactly two places — each Swin block's MLP
//! (`SwinBlock::forward`, one call per block, so `sum(depths)` calls per
//! forward) and the projection head's `"gelu"` activation arm
//! ([`ClapAudioProjection::forward_unnormalized_with_training`], one call per
//! forward, and only when `projection_hidden_act == "gelu"`; the `"relu"` arm
//! is a different activation and is untouched). Neither calls `Tensor::gelu_erf` directly:
//! both route through `crate::activations::gelu_erf(x, training)`, the same
//! house seam `crate::bert`'s `BertIntermediate::forward` and
//! `crate::distilbert`'s `DistilBertFfn::forward` use. The invariant that seam
//! carries, and that this tower therefore inherits:
//!
//! * `training == false` is the UNCHANGED `x.gelu_erf()` call, byte for byte —
//!   no admission machinery runs at all, so eval bytes (and every golden-parity
//!   / bits-snapshot row taken in eval) are exactly what they were before the
//!   seam existed. Asserted by
//!   `tests::eval_output_is_bit_identical_across_a_training_toggle_round_trip`.
//! * `training == true` makes the fused-vs-eager choice a COUNTED admission
//!   decision on tensor state (dtype/contiguity/device/non-emptiness), never on
//!   model identity. Asserted by
//!   `tests::training_true_full_forward_dispatches_the_gelu_seam_once_per_swin_block`
//!   and `tests::projection_gelu_arm_dispatches_the_seam_exactly_once`.
//!
//! The `training` flag both sites read is a call-chain PARAMETER sourced from
//! [`HtsatAudio::set_training`]'s single stored flag — threaded
//! `HtsatAudio::forward` → [`HtsatAudioEncoder::forward_spine_with_training`]
//! → `SwinBlock::forward`, and `HtsatAudio::forward` →
//! [`ClapAudioProjection::forward_unnormalized_with_training`] — NOT a
//! per-sub-struct stored copy (`crate::activations::gelu_erf`'s own doc
//! records the drift defect that rule exists to prevent). The two
//! flag-less public entry points ([`HtsatAudioEncoder::forward_spine`],
//! [`ClapAudioProjection::forward_unnormalized`]) are EVAL conveniences for
//! boundary-parity harnesses, defined as their `_with_training(.., false)`
//! twins.
//!
//! One counter, two sites: both sites report to the SAME process-wide
//! `gelu_erf_fused` registry entry, so a full-tower forward's dispatch delta
//! is their SUM — `sum(depths)` (one per Swin block) plus one more when
//! `projection_hidden_act == "gelu"`. On the `htsat_clap_tiny` fixture
//! (`depths = [2, 2, 2, 2]`, projection act `"relu"`) that is exactly 8; the
//! projection's own `+1` is pinned separately by
//! `tests::projection_gelu_arm_dispatches_the_seam_exactly_once` on an
//! explicitly-`"gelu"` fixture, with
//! `tests::projection_relu_arm_never_touches_the_gelu_seam` as its negative
//! control.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, IndexOp, Module, ModuleT, Tensor, D};
use candle_nn::{
    batch_norm, conv2d, linear, linear_no_bias, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig,
    VarBuilder, VarMap,
};
use jammi_lora::{FrozenBase, LoraBuildConfig, MaybeLoraLinear};

use crate::activations;
use crate::error::EncoderError;
use crate::layer_norm::LayerNorm;
use crate::lora_site::{FrozenSiteHolder, LoraSite};

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
    /// Activation applied inside each Swin block's MLP. Every config field
    /// that NAMES a computation is either genuinely dispatched on or
    /// refused as unsupported — never silently ignored (family D). Unlike
    /// `projection_hidden_act` (dispatched at forward — see
    /// [`ClapAudioProjection::forward_unnormalized_with_training`]),
    /// `SwinBlock::forward` is unconditionally GELU-erf (this module's own
    /// doc), so [`HtsatAudioEncoder::load`] REFUSES any value other than
    /// `"gelu"` (HF `ClapAudioConfig`'s only shipped value) at load, naming
    /// this field, rather than silently ignoring a checkpoint that declares
    /// a different one.
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
        //
        // The tap weights are computed in f32 in ATen's exact operation
        // order (see this type's own doc — building them in f64 instead
        // drifts the result ~2e-4 from the reference), then cast ONCE to
        // `x`'s dtype so a reduced-precision backbone's matmul sees matching
        // operands. For an F32 backbone that cast is `self.clone()`
        // (candle's same-dtype early return): the f32 path's bits are
        // unchanged.
        let weights = Self::build_matrix(self.out_len, t, &self.device)?.to_dtype(x.dtype())?;
        // [B, C, T, F] -> [B, C, F, T] -> [B*C*F, T]
        let xt = x.transpose(2, 3)?.contiguous()?.reshape((b * c * f, t))?;
        // [B*C*F, T] @ [T, out_len] = [B*C*F, out_len]
        let out = crate::contiguous_matmul(&xt, &weights.t()?)?;
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

/// Tile `x` (`[N, C, H, W]`) into non-overlapping `kh × kw` blocks:
/// `[N, C, OH, OW, kh, kw]`, via `narrow` (to the largest divisible prefix
/// of each tiled dim) followed by a `reshape` + `permute`.
///
/// Any tail narrower than `kh`/`kw` is dropped, exactly as a strided conv
/// with `stride == kernel` would drop it — narrowing to `floor(dim/size) *
/// size` first makes both the forward tiling AND its gradient exact: the
/// narrowed tensor reshapes losslessly to `[N, C, OH, kh, OW, kw]` (its
/// element count already matches, since every axis is now an exact
/// multiple of its tile size), and `permute((0, 1, 2, 4, 3, 5))` reorders
/// the `kw` axis ahead of `OW` to land on `[N, C, OH, OW, kh, kw]`. Both
/// `reshape` and `permute` have exact, structural backward passes in
/// candle (a reshape's backward is the inverse reshape; a permute's
/// backward is the inverse permute), so gradient flows back through every
/// element at the position it actually came from — unlike `Tensor::unfold`,
/// whose backward is registered as a single flat `Op::Reshape`
/// (candle-core-0.11.0 `tensor.rs:2931-2969` builds the unfolded view with
/// `op: BackpropOp::new1(self, Op::Reshape)`; its backward,
/// `backprop.rs:602-606`, does `let arg_grad = grad.reshape(arg.dims())?`)
/// regardless of which dim is unfolded. That flat reshape is only a valid
/// gradient when the unfolded dim is the tensor's LAST dim (unfolding dim 2
/// of a 4-D tensor into a NEW trailing dim, as `mel_conv2d` does twice in a
/// row, permutes elements between the reshape's input and output layout —
/// `grad.reshape(arg.dims())` silently reinterprets the flat buffer under
/// the wrong strides instead of scattering each gradient element back to
/// its source position). PyTorch's own `Tensor.unfold` backward
/// (`aten/src/ATen/native/UnfoldBackward.cpp`, `unfold_backward`) is a true
/// scatter for exactly this reason — the semantics this tiling reproduces
/// with `reshape`/`permute`, whose backward passes are correct scatters by
/// construction, avoids depending on candle's unfold backward at all. See
/// [`tests::unfold_backward_is_a_plain_reshape_in_candle`] for the
/// premise-pin on candle's flat-reshape behavior, and
/// [`tests::unfold_backward_on_a_non_last_dim_is_wrong_but_finite`] for the
/// pin on the permutation failure this function avoids. Forward stays
/// bit-identical to the old unfold-based tiling — see
/// [`tests::tile_nonoverlapping_is_forward_bit_identical_to_unfold_without_narrow_first`].
fn tile_nonoverlapping(x: &Tensor, kh: usize, kw: usize) -> Result<Tensor, EncoderError> {
    let (n, c, h, w) = x.dims4()?;
    let (oh, ow) = (h / kh, w / kw);
    // Domain guard: h < kh or w < kw makes oh or ow zero, which would
    // silently narrow to a zero-length tile axis and return an empty
    // tensor with no error — a confident-wrong shape rather than a
    // refusal. Reject before narrowing.
    if oh == 0 || ow == 0 {
        return Err(EncoderError::Config(format!(
            "tile_nonoverlapping: input spatial dims ({h}, {w}) are smaller than the tile \
             kernel ({kh}, {kw}); at least one output tile dimension would be zero (oh={oh}, \
             ow={ow})"
        )));
    }
    let x = x.narrow(2, 0, oh * kh)?.narrow(3, 0, ow * kw)?;
    Ok(x.contiguous()?
        .reshape((n, c, oh, kh, ow, kw))?
        .permute((0, 1, 2, 4, 3, 5))?)
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
        // `with_bias=true`: this config never sets a `remove_mean=false`
        // (RMSNorm-style) variant anywhere — see
        // `crate::layer_norm::LayerNorm::new`'s doc for the "weight"/"bias"
        // safetensors names this reads, unchanged from `candle_nn::layer_norm`.
        let norm = LayerNorm::new(
            config.patch_embeds_hidden_size,
            config.layer_norm_eps,
            true,
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
    /// strided conv would — see [`tile_nonoverlapping`]'s doc for why that
    /// drop is narrowed explicitly rather than left implicit in `unfold`'s
    /// own window count). Each block is flattened and projected by the
    /// reshaped weight `[out_c, kh*kw]`.
    fn mel_conv2d(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        let (n, _in_c, _h, _w) = x.dims4()?;
        let (out_c, _kc, kh, kw) = self.mel_conv2d_weight.dims4()?;

        // Tile H then W into non-overlapping blocks: append size-kh / size-kw
        // trailing axes. [N, 1, H, W] -> [N, 1, OH, OW, kh, kw].
        let tiled = tile_nonoverlapping(x, kh, kw)?;
        let tiled_dims = tiled.dims().to_vec();
        let (oh, ow) = (tiled_dims[2], tiled_dims[3]);

        // Flatten each block to a kh*kw vector and the batch/grid to rows:
        // [N, 1, OH, OW, kh, kw] -> [N*OH*OW, kh*kw].
        let patches = tiled.contiguous()?.reshape((n * oh * ow, kh * kw))?;

        // Weight [out_c, 1, kh, kw] -> [kh*kw, out_c]; project and add bias.
        let w = self.mel_conv2d_weight.reshape((out_c, kh * kw))?.t()?;
        let out = crate::contiguous_matmul(&patches, &w)?;
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
        // Built in the GLOBAL path's own dtype so a reduced-precision
        // backbone can `broadcast_mul` it without a dtype mismatch. `0.0`
        // and `1.0` are exact in every floating dtype, and for an F32
        // backbone `to_dtype` is `self.clone()` (candle's same-dtype early
        // return), so this tensor's bits are unchanged there.
        let mask =
            Tensor::from_vec(mask, (batch, 1, 1, 1), global.device())?.to_dtype(global.dtype())?;
        let patch_map = mask
            .broadcast_mul(&fused)?
            .add(&(1.0 - &mask)?.broadcast_mul(&global)?)?;

        // Flatten the patch grid and LayerNorm: [B, C, gh, gw] -> [B, gh*gw, C].
        let flat = patch_map.flatten_from(2)?.transpose(1, 2)?.contiguous()?;
        self.norm.forward(&flat)
    }

    fn set_training(&mut self, training: bool) {
        self.norm.set_training(training);
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

/// The six per-block LoRA-wrappable sites, plus the two unindexed
/// projection-head ones and the stage downsample. Each name is the leaf the
/// HF CLAP checkpoint itself uses (`attention.self.query` →
/// [`QUERY_SITE`], `attention.output.dense` → [`ATTENTION_OUTPUT_SITE`],
/// `intermediate.dense` → [`INTERMEDIATE_DENSE_SITE`], `output.dense` →
/// [`OUTPUT_DENSE_SITE`]) collapsed to the BERT-family site-name convention
/// this workspace already uses for the same six roles — so a caller writes
/// the same `target_modules` string against either family.
const QUERY_SITE: &str = "query";
/// See [`QUERY_SITE`].
const KEY_SITE: &str = "key";
/// See [`QUERY_SITE`].
const VALUE_SITE: &str = "value";
/// See [`QUERY_SITE`].
const ATTENTION_OUTPUT_SITE: &str = "attention_output";
/// See [`QUERY_SITE`].
const INTERMEDIATE_DENSE_SITE: &str = "intermediate_dense";
/// See [`QUERY_SITE`].
const OUTPUT_DENSE_SITE: &str = "output_dense";
/// The patch-merging downsample's `4C → 2C` linear (`layers.{s}.downsample.
/// reduction` in the checkpoint). Bias-free.
const REDUCTION_SITE: &str = "reduction";
/// The audio projection head's two linears (`audio_projection.linear{1,2}`)
/// — UNINDEXED sites: they belong to no Swin stage, so their
/// `LoraSite::layer_idx` is `None`.
const LINEAR1_SITE: &str = "linear1";
/// See [`LINEAR1_SITE`].
const LINEAR2_SITE: &str = "linear2";
/// The ONE `projection_hidden_act` value that routes the projection head
/// through the house GELU seam (`crate::activations::gelu_erf`) and so adds
/// one `gelu_erf_fused` decision per training forward.
///
/// Named once and used in BOTH places that must agree about it — the
/// dispatch site ([`ClapAudioProjection::forward_unnormalized_with_training`]'s
/// match arm) and the census that predicts how often that site fires
/// ([`HtsatAudio::fusible_site_census`]) — so the witness and the thing it
/// witnesses cannot drift to two different spellings.
const GELU_PROJECTION_ACT: &str = "gelu";

/// The selector names a caller may write in `target_modules` to reach this
/// tower's LoRA sites — the nine constants above, in the order the
/// traversals visit them (the six per-block roles, then the stage
/// downsample, then the two unindexed projection-head linears).
///
/// Built from the constants themselves, so it cannot drift from the names
/// the sites are actually wrapped under; `AnyEncoder::lora_site_names`
/// returns it, and a test asserts every entry selects at least one real
/// site on a fixture while the union of all of them is exactly what
/// `all-linear` selects. Two of these are UNINDEXED head sites, so an
/// active `layers_to_transform` filter refuses them — see
/// [`jammi_lora::should_apply_lora`]'s own doc.
pub(crate) const LORA_SITE_NAMES: &[&str] = &[
    QUERY_SITE,
    KEY_SITE,
    VALUE_SITE,
    ATTENTION_OUTPUT_SITE,
    INTERMEDIATE_DENSE_SITE,
    OUTPUT_DENSE_SITE,
    REDUCTION_SITE,
    LINEAR1_SITE,
    LINEAR2_SITE,
];

/// Self-attention inside a Swin window (W-MSA / SW-MSA), with the recomputed
/// relative-position bias and an optional precomputed shift-window mask.
struct SwinSelfAttention {
    query: MaybeLoraLinear,
    key: MaybeLoraLinear,
    value: MaybeLoraLinear,
    /// `[(2·ws−1)², num_heads]` learned relative-position bias table, sized by
    /// the config window (HF sizes the table by `config.window_size`, not the
    /// block's effective window).
    rel_bias_table: Tensor,
    /// `[(ws·ws)²]` flattened relative-position index (U32), recomputed over the
    /// config window.
    rel_index: Tensor,
    num_heads: usize,
    head_size: usize,
    /// Selects the softmax arm — see [`Self::forward`]'s doc for why the two
    /// arms exist. Defaults to `false` (eval); flipped by
    /// [`HtsatAudio::set_training`].
    training: bool,
}

impl SwinSelfAttention {
    /// `ws` is the config window size (`config.window_size`): HF constructs
    /// `ClapAudioSelfAttention` with `window_size=config.window_size` and never
    /// re-sizes it when a block's effective window is clamped to a smaller grid,
    /// so the bias table and relative-position index are both sized by the config
    /// window. (Token count per window equals the effective window squared, which
    /// coincides with `ws·ws` in every reachable config since the deepest stage's
    /// grid equals the window.)
    fn load_with(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        ws: usize,
        site: &LoraSite<'_>,
    ) -> Result<Self, EncoderError> {
        let query = linear(dim, dim, vb.pp(QUERY_SITE))?;
        let key = linear(dim, dim, vb.pp(KEY_SITE))?;
        let value = linear(dim, dim, vb.pp(VALUE_SITE))?;
        let query = site.wrap(FrozenBase::Dense(query), QUERY_SITE, QUERY_SITE)?;
        let key = site.wrap(FrozenBase::Dense(key), KEY_SITE, KEY_SITE)?;
        let value = site.wrap(FrozenBase::Dense(value), VALUE_SITE, VALUE_SITE)?;
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
            training: false,
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
    ///
    /// See [`crate::attention::attention_softmax`]'s module doc for the
    /// `BackpropOp::none()` truncation this dispatch works around.
    /// `query`/`key` (and, most starkly, `rel_bias_table`) sit EXCLUSIVELY
    /// upstream of this softmax: `rel_bias_table` is read only at the
    /// index-select/reshape/permute above that feeds `scores`, with no other
    /// use anywhere in the tower. With no other path ever contributing to its
    /// gradient accumulator, `grads.get(&rel_bias_table_var)` comes back
    /// `None` under eval, not merely zero — a strictly worse failure mode
    /// than CLIP's fused-QKV case (there, V's surviving path at least forces
    /// the shared `in_proj_weight` accumulator to exist, so Q/K only read
    /// back as zero rows of it). `training=true` routes through the
    /// composed, differentiable arm — same arm ModernBERT's attention uses.
    ///
    /// This flag is load-bearing on CUDA: candle's fused softmax kernel and
    /// the composed primitive-op reduction can differ in fold order there.
    /// On CPU the two arms are bit-identical (measured 0/N differing
    /// mantissa bits — see
    /// `crate::attention::tests::softmax_last_dim_and_composed_softmax_are_cpu_bit_identical`,
    /// which covers every tower's shared claim), so `training=false` costs
    /// nothing to keep byte-identical to today's eval path.
    ///
    /// No caller in this codebase can set `training=true` on an `HtsatAudio`
    /// tower today (`CandleAudioForward` towers are reached through `&self`
    /// trait objects behind `Arc<LoadedModel>`). The flag exists so the
    /// tower is trainable-CORRECT for when a caller does.
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
        let scores = (crate::contiguous_matmul(&q, &k.transpose(D::Minus1, D::Minus2)?)? * scale)?;

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

        let probs = crate::attention::attention_softmax(&scores, self.training)?;
        let ctx = crate::contiguous_matmul(&probs, &v)?; // [BnW, heads, L, head]
        let ctx = ctx.transpose(1, 2)?.contiguous()?.reshape((bnw, l, c))?;
        Ok(ctx)
    }

    /// Propagates to the softmax arm AND to the three QKV LoRA sites
    /// (whose dropout is training-gated).
    fn set_training(&mut self, training: bool) {
        self.training = training;
        for (_, lin) in self.lora_sites_mut() {
            lin.set_training(training);
        }
    }

    fn lora_sites(&self) -> [(&'static str, &MaybeLoraLinear); 3] {
        [
            (QUERY_SITE, &self.query),
            (KEY_SITE, &self.key),
            (VALUE_SITE, &self.value),
        ]
    }

    fn lora_sites_mut(&mut self) -> [(&'static str, &mut MaybeLoraLinear); 3] {
        [
            (QUERY_SITE, &mut self.query),
            (KEY_SITE, &mut self.key),
            (VALUE_SITE, &mut self.value),
        ]
    }
}

/// A single Swin block (pre-norm, W-MSA or SW-MSA) with the two residual MLP.
struct SwinBlock {
    layernorm_before: LayerNorm,
    attention: SwinSelfAttention,
    attention_output: MaybeLoraLinear,
    layernorm_after: LayerNorm,
    intermediate: MaybeLoraLinear,
    output: MaybeLoraLinear,
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

/// The three load-time values EVERY level of this tower's loader threads
/// unchanged from the root down to an individual Swin block, bundled so each
/// signature carries one context instead of three parallel arguments that a
/// call site could reorder (`config`/`device` are both references; a swapped
/// pair of same-shaped arguments is exactly the class a bundle removes).
///
/// `mask_dtype` is NOT re-derivable from `vb` at every level and is not the
/// same thing as "whatever dtype this `VarBuilder` happens to hand back": it
/// is the FROZEN BACKBONE dtype every load-time shift-window mask is
/// materialised in, so the forward path never casts one. An additive
/// attention mask must be representable in the dtype the scores are computed
/// in — see `crate::clip_text::dtype_min_sentinel` for the same family-D
/// argument on the CLIP side.
#[derive(Clone, Copy)]
struct HtsatLoadCtx<'a> {
    config: &'a HtsatAudioConfig,
    device: &'a candle_core::Device,
    mask_dtype: DType,
}

impl SwinBlock {
    fn load_with(
        vb: VarBuilder,
        ctx: HtsatLoadCtx<'_>,
        dim: usize,
        num_heads: usize,
        input_resolution: (usize, usize),
        block_index: usize,
        site: &LoraSite<'_>,
    ) -> Result<Self, EncoderError> {
        let config = ctx.config;
        let eps = config.layer_norm_eps;
        // `with_bias=true`: no `remove_mean=false` variant exists in this
        // config — see `HtsatPatchEmbed::load`'s note on the same class.
        let layernorm_before = LayerNorm::new(dim, eps, true, vb.pp("layernorm_before"))?;
        let attention = SwinSelfAttention::load_with(
            vb.pp("attention").pp("self"),
            dim,
            num_heads,
            config.window_size,
            site,
        )?;
        let attention_output = linear(dim, dim, vb.pp("attention").pp("output").pp("dense"))?;
        let attention_output = site.wrap(
            FrozenBase::Dense(attention_output),
            ATTENTION_OUTPUT_SITE,
            ATTENTION_OUTPUT_SITE,
        )?;
        let layernorm_after = LayerNorm::new(dim, eps, true, vb.pp("layernorm_after"))?;
        let inter = (config.mlp_ratio * dim as f64) as usize;
        let intermediate = linear(dim, inter, vb.pp("intermediate").pp("dense"))?;
        let intermediate = site.wrap(
            FrozenBase::Dense(intermediate),
            INTERMEDIATE_DENSE_SITE,
            INTERMEDIATE_DENSE_SITE,
        )?;
        let output = linear(inter, dim, vb.pp("output").pp("dense"))?;
        let output = site.wrap(
            FrozenBase::Dense(output),
            OUTPUT_DENSE_SITE,
            OUTPUT_DENSE_SITE,
        )?;

        // set_shift_and_window_size: window/2 for odd blocks, forced 0 (with the
        // window clamped to the grid) when the grid is no larger than the window.
        let mut window_size = config.window_size;
        let mut shift_size = if block_index.is_multiple_of(2) {
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
                ctx.device,
                ctx.mask_dtype,
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

    /// Build the SW-MSA attention mask `[nW, L, L]` from the 9-region label
    /// map, in `dtype` — the frozen backbone's own dtype, built ONCE at load
    /// so no forward ever casts it.
    ///
    /// # The `-100.0` magnitude is HF parity, not a free parameter
    ///
    /// This is the value HF's `ClapAudioLayer` itself adds across a
    /// disallowed shift-window row, and `exp(-100) ≈ 3.7e-44` is a
    /// DENORMAL in f32 — output-affecting, so it must not be "improved" to
    /// a dtype minimum the way `crate::clip_text`'s causal sentinel is.
    /// `-100.0` is exactly representable in F32, F16 and BF16 alike (it is
    /// `-1.5625 × 2^6`, well inside every one of their normal ranges), so
    /// casting the f32-built tensor with `to_dtype` is exact in all three
    /// — and for F32 that cast is `self.clone()` (candle's same-dtype early
    /// return), keeping this tensor byte-for-byte what it has always been.
    fn build_attn_mask(
        h: usize,
        w: usize,
        ws: usize,
        shift: usize,
        device: &candle_core::Device,
        dtype: DType,
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
        Ok(mask.to_dtype(dtype)?)
    }

    /// `training` is a PARAMETER, not a stored copy — the same rule
    /// `crate::bert::BertIntermediate::forward` and
    /// `crate::distilbert::DistilBertFfn::forward` follow, and the one
    /// `crate::activations::gelu_erf`'s own doc mandates:
    /// [`HtsatAudio::set_training`]'s flag is the single source of truth,
    /// threaded down through
    /// [`HtsatAudioEncoder::forward_spine_with_training`] to this call, so a
    /// desync between what the tower's own forward dispatched on and what
    /// this block's MLP activation receives is unrepresentable — there is no
    /// second copy of it left to drift.
    fn forward(&self, hidden: &Tensor, training: bool) -> Result<Tensor, EncoderError> {
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

        // MLP with residual 2. The activation goes through the house GELU
        // seam (`crate::activations::gelu_erf`), never `Tensor::gelu_erf`
        // directly — see this module's own doc: `training == false` is that
        // same call byte for byte, `true` is a counted admission decision on
        // tensor state.
        let y = self.layernorm_after.forward(&hidden)?;
        let y = self.intermediate.forward(&y)?;
        let y = activations::gelu_erf(&y, training)?;
        let y = self.output.forward(&y)?;
        Ok((&hidden + y)?)
    }

    /// Propagates to the attention module (softmax arm + its three QKV LoRA
    /// sites), both LayerNorms, and this block's own three LoRA sites. It
    /// deliberately does NOT reach the MLP's GELU seam arm: that one is a
    /// [`Self::forward`] PARAMETER, not a stored copy — see that method's
    /// own doc.
    fn set_training(&mut self, training: bool) {
        self.attention.set_training(training);
        self.layernorm_before.set_training(training);
        self.layernorm_after.set_training(training);
        for lin in [
            &mut self.attention_output,
            &mut self.intermediate,
            &mut self.output,
        ] {
            lin.set_training(training);
        }
    }

    /// This block's six LoRA sites paired with their names — the single
    /// source of the site→name map every [`HtsatAudio`] traversal walks.
    /// Fixed order: the three QKV projections, then the attention output,
    /// then the two MLP linears.
    fn lora_sites(&self) -> [(&'static str, &MaybeLoraLinear); 6] {
        let [query, key, value] = self.attention.lora_sites();
        [
            query,
            key,
            value,
            (ATTENTION_OUTPUT_SITE, &self.attention_output),
            (INTERMEDIATE_DENSE_SITE, &self.intermediate),
            (OUTPUT_DENSE_SITE, &self.output),
        ]
    }

    /// The `&mut` twin of `Self::lora_sites`, same names, same order.
    fn lora_sites_mut(&mut self) -> [(&'static str, &mut MaybeLoraLinear); 6] {
        let [query, key, value] = self.attention.lora_sites_mut();
        [
            query,
            key,
            value,
            (ATTENTION_OUTPUT_SITE, &mut self.attention_output),
            (INTERMEDIATE_DENSE_SITE, &mut self.intermediate),
            (OUTPUT_DENSE_SITE, &mut self.output),
        ]
    }
}

/// Swin patch-merging downsample: `2×` spatial reduction with a `4C → 2C`
/// linear over the concatenated `2×2` neighbourhood.
struct PatchMerging {
    norm: LayerNorm,
    reduction: MaybeLoraLinear,
    input_resolution: (usize, usize),
}

impl PatchMerging {
    fn load_with(
        vb: VarBuilder,
        config: &HtsatAudioConfig,
        dim: usize,
        input_resolution: (usize, usize),
        site: &LoraSite<'_>,
    ) -> Result<Self, EncoderError> {
        // `with_bias=true`: no `remove_mean=false` variant exists in this
        // config — see `HtsatPatchEmbed::load`'s note on the same class.
        let norm = LayerNorm::new(4 * dim, config.layer_norm_eps, true, vb.pp("norm"))?;
        // BIAS-FREE (`linear_no_bias`), unlike every other site in this
        // tower — `jammi_lora`'s `bias_gate` returns `Ok(None)` for a
        // bias-free base, so this site wraps exactly like the others.
        let reduction = linear_no_bias(4 * dim, 2 * dim, vb.pp(REDUCTION_SITE))?;
        let reduction = site.wrap(FrozenBase::Dense(reduction), REDUCTION_SITE, REDUCTION_SITE)?;
        Ok(Self {
            norm,
            reduction,
            input_resolution,
        })
    }

    fn set_training(&mut self, training: bool) {
        self.norm.set_training(training);
        self.reduction.set_training(training);
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
    /// `site_for` yields the [`LoraSite`] for block `b` of THIS stage: its
    /// `lora_vb` is scoped to `layers.{s}.blocks.{b}` and its `layer_idx` is
    /// `Some(s)` — the STAGE index, not the block index. That is PEFT's own
    /// rule: `check_target_module_exists`
    /// (`peft/src/peft/tuners/tuners_utils.py:2353-2389`) matches the FIRST
    /// numbered segment of the key (`re.match(r".*?\.[^.]*\.(?P<idx>\d+)\.")`),
    /// which for `layers.{s}.blocks.{b}.…` is `s`. `downsample_site` is the
    /// stage's own patch-merging site (subpath `layers.{s}.downsample`, same
    /// stage index).
    ///
    /// Still `#[allow(clippy::too_many_arguments)]` after the
    /// [`HtsatLoadCtx`] bundle absorbed `config`/`device`/`mask_dtype`: the
    /// five that remain (`dim`, `num_heads`, `depth`, `input_resolution`,
    /// `has_downsample`) are the stage's own GEOMETRY, each derived
    /// differently from the stage index by the caller
    /// (`patch_embeds_hidden_size << i`, `num_attention_heads[i]`,
    /// `depths[i]`, `grid >> i`, `i < num_stages - 1`). Bundling them would
    /// only move that derivation behind a constructor with the same five
    /// fields, not remove a single one.
    #[allow(clippy::too_many_arguments)]
    fn load_with<'a>(
        vb: VarBuilder,
        ctx: HtsatLoadCtx<'_>,
        dim: usize,
        num_heads: usize,
        depth: usize,
        input_resolution: (usize, usize),
        has_downsample: bool,
        site_for: &dyn Fn(usize) -> LoraSite<'a>,
        downsample_site: &LoraSite<'_>,
    ) -> Result<Self, EncoderError> {
        let blocks_vb = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            blocks.push(SwinBlock::load_with(
                blocks_vb.pp(i),
                ctx,
                dim,
                num_heads,
                input_resolution,
                i,
                &site_for(i),
            )?);
        }
        let downsample = if has_downsample {
            Some(PatchMerging::load_with(
                vb.pp("downsample"),
                ctx.config,
                dim,
                input_resolution,
                downsample_site,
            )?)
        } else {
            None
        };
        Ok(Self { blocks, downsample })
    }

    fn set_training(&mut self, training: bool) {
        for block in &mut self.blocks {
            block.set_training(training);
        }
        if let Some(downsample) = &mut self.downsample {
            downsample.set_training(training);
        }
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
    /// Dtype the frozen backbone weights are materialised at, read off the
    /// patch-embedding `proj` convolution kernel — the first WEIGHT any
    /// input meets after the batch-norm, and one that exists in every HTSAT
    /// checkpoint.
    pub fn dtype(&self) -> DType {
        self.patch_embed.proj.weight().dtype()
    }

    /// Build the front-half encoder from an `audio_encoder`-scoped
    /// [`VarBuilder`] (i.e. `root.pp("audio_model").pp("audio_encoder")`),
    /// fully frozen. Routed through `Self::load_with` with a
    /// decline-everything site, so there is one loader (see
    /// `crate::lora_site::FrozenSiteHolder`).
    pub fn load(
        vb: VarBuilder,
        config: &HtsatAudioConfig,
        device: &candle_core::Device,
    ) -> Result<Self, EncoderError> {
        let holder = FrozenSiteHolder::new();
        let ctx = HtsatLoadCtx {
            config,
            device,
            mask_dtype: vb.dtype(),
        };
        Self::load_with(vb.clone(), ctx, &|_s, _b| holder.site(&vb), &|_s| {
            holder.site(&vb)
        })
    }

    /// The one loader. `block_site(stage, block)` yields the [`LoraSite`]
    /// for that Swin block and `downsample_site(stage)` the one for that
    /// stage's patch-merging downsample — two closures rather than one with
    /// a sentinel index, because a downsample genuinely has no block index
    /// (both carry `layer_idx == Some(stage)`; see [`SwinStage::load_with`]'s
    /// doc for the PEFT rule that makes the STAGE the index). `ctx` carries
    /// the config, device and the dtype every load-time shift-window mask is
    /// materialised in (see [`HtsatLoadCtx`]).
    ///
    /// The config-validity edge for `config.hidden_act` (see that field's
    /// own doc): both [`HtsatAudio::load`] and `HtsatAudioBuilder::build`
    /// (`crate::htsat_audio`'s builder) funnel through this one loader, so
    /// checking it here — before any tensor is touched — refuses an
    /// unsupported checkpoint through EITHER entry point rather than
    /// silently computing gelu-erf under a different declared activation.
    fn load_with<'a>(
        vb: VarBuilder,
        ctx: HtsatLoadCtx<'_>,
        block_site: &dyn Fn(usize, usize) -> LoraSite<'a>,
        downsample_site: &dyn Fn(usize) -> LoraSite<'a>,
    ) -> Result<Self, EncoderError> {
        let HtsatLoadCtx { config, device, .. } = ctx;
        if config.hidden_act != "gelu" {
            return Err(EncoderError::Config(format!(
                "HtsatAudioConfig.hidden_act = {:?} is unsupported: SwinBlock's MLP is \
                 compiled to gelu-erf only (HF ClapAudioConfig's only shipped value is \
                 \"gelu\") — refusing rather than silently computing gelu-erf under a \
                 different declared activation",
                config.hidden_act
            )));
        }
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
            stages.push(SwinStage::load_with(
                layers_vb.pp(i),
                ctx,
                dim,
                config.num_attention_heads[i],
                config.depths[i],
                input_resolution,
                i < num_stages - 1,
                &|b| block_site(i, b),
                &downsample_site(i),
            )?);
        }
        // `with_bias=true`: no `remove_mean=false` variant exists in this
        // config — see `HtsatPatchEmbed::load`'s note on the same class.
        let norm = LayerNorm::new(
            config.hidden_size,
            config.layer_norm_eps,
            true,
            vb.pp("norm"),
        )?;

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
    ///
    /// # Input-dtype domain: cast at the seam
    ///
    /// `input_features` may arrive in ANY floating dtype. This method is
    /// the tower's whole floating-input edge — the leading `BatchNorm` and
    /// the patch-embedding `Conv2d` are the first weights a fusion
    /// spectrogram meets, and neither promotes a mismatched input the way
    /// a dense linear does (an F32 batch against an F16 backbone is a hard
    /// `dtype mismatch in sub` inside the batch-norm centring). Every
    /// production front end (decode → resample → log-mel →
    /// `Tensor::from_vec` of `f32`) emits F32, so casting here — the one
    /// place that knows
    /// [`Self::dtype`] — is what lets an F16 backbone consume the only
    /// batch a caller actually has. `Tensor::to_dtype` on the SAME dtype is
    /// `self.clone()`, so an already-matching (F32-on-F32) batch takes a
    /// byte-identical path: a widening of the accepted input domain, never
    /// a change to any output bit.
    pub fn forward_front(
        &self,
        input_features: &Tensor,
        is_longer: &[bool],
    ) -> Result<FrontHalf, EncoderError> {
        let input_features = input_features.to_dtype(self.dtype())?;
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
        self.forward_spine_with_training(patch_embed_out, frames_num, false)
    }

    /// [`Self::forward_spine`] with each Swin block's MLP GELU-seam arm
    /// chosen by the caller's `training` flag, a PARAMETER threaded down to
    /// every `SwinBlock::forward` rather than a stored per-block copy (the
    /// rule `crate::activations::gelu_erf`'s own doc mandates —
    /// [`HtsatAudio::set_training`]'s flag is the single source of truth,
    /// and [`HtsatAudio::forward`] is the caller that supplies it).
    ///
    /// Note what this flag does NOT select: the attention softmax arm and
    /// every LayerNorm arm are propagated STATE (this encoder's own
    /// crate-private `set_training`), not parameters, so
    /// [`Self::forward_spine`] (the eval convenience) on a spine that
    /// `set_training(true)` has already touched still takes the training
    /// softmax arm — it only puts the GELU seam on its eval arm, which on
    /// every supported device is numerically the same function. The
    /// production path never mixes the two: `HtsatAudio::forward` passes its
    /// own flag here.
    pub fn forward_spine_with_training(
        &self,
        patch_embed_out: &Tensor,
        frames_num: usize,
        training: bool,
    ) -> Result<Spine, EncoderError> {
        let mut blocks: Vec<Vec<Tensor>> = Vec::with_capacity(self.num_stages);
        let mut downsamples: Vec<Option<Tensor>> = Vec::with_capacity(self.num_stages);

        let mut hidden = patch_embed_out.clone();
        for stage in &self.stages {
            let mut stage_blocks = Vec::with_capacity(stage.blocks.len());
            for block in &stage.blocks {
                hidden = block.forward(&hidden, training)?;
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

    /// Propagate the training flag to every stage's blocks (attention
    /// softmax AND `layernorm_before`/`layernorm_after`/`PatchMerging::norm`
    /// — see [`SwinSelfAttention::forward`]'s doc for the softmax truncation
    /// and [`crate::layer_norm::LayerNorm`]'s module doc for the identical
    /// `BackpropOp::none()` truncation candle_nn's own LayerNorm fast path
    /// has), plus the patch-embed norm and this encoder's own final norm —
    /// every LayerNorm the front-half + spine + pooling boundary touches.
    /// Before the final norm was gated, backward through
    /// `HtsatAudio::forward`'s public output truncated there regardless of
    /// the attention fix, giving NO gradient at all to anything upstream —
    /// see `tests::training_true_full_forward_reaches_every_parameter`.
    fn set_training(&mut self, training: bool) {
        self.patch_embed.set_training(training);
        for stage in &mut self.stages {
            stage.set_training(training);
        }
        self.norm.set_training(training);
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
    linear1: MaybeLoraLinear,
    linear2: MaybeLoraLinear,
    act: String,
}

impl ClapAudioProjection {
    /// Build from a root-scoped [`VarBuilder`] (projection lives at
    /// `audio_projection.*`, a sibling of `audio_model`), fully frozen.
    pub fn load(vb: VarBuilder, config: &HtsatAudioConfig) -> Result<Self, EncoderError> {
        let holder = FrozenSiteHolder::new();
        Self::load_with(vb.clone(), config, &holder.site(&vb))
    }

    /// The one loader. `site`'s `layer_idx` is `None`: this head belongs to
    /// no Swin stage, so an active `layers_to_transform` filter can never
    /// select it (`jammi_lora::should_apply_lora`'s own doc — PEFT's rule
    /// for a key with no numbered segment).
    fn load_with(
        vb: VarBuilder,
        config: &HtsatAudioConfig,
        site: &LoraSite<'_>,
    ) -> Result<Self, EncoderError> {
        let linear1 = linear(
            config.hidden_size,
            config.projection_dim,
            vb.pp(LINEAR1_SITE),
        )?;
        let linear2 = linear(
            config.projection_dim,
            config.projection_dim,
            vb.pp(LINEAR2_SITE),
        )?;
        Ok(Self {
            linear1: site.wrap(FrozenBase::Dense(linear1), LINEAR1_SITE, LINEAR1_SITE)?,
            linear2: site.wrap(FrozenBase::Dense(linear2), LINEAR2_SITE, LINEAR2_SITE)?,
            act: config.projection_hidden_act.clone(),
        })
    }

    fn lora_sites(&self) -> [(&'static str, &MaybeLoraLinear); 2] {
        [(LINEAR1_SITE, &self.linear1), (LINEAR2_SITE, &self.linear2)]
    }

    fn lora_sites_mut(&mut self) -> [(&'static str, &mut MaybeLoraLinear); 2] {
        [
            (LINEAR1_SITE, &mut self.linear1),
            (LINEAR2_SITE, &mut self.linear2),
        ]
    }

    /// Propagates to this head's two LoRA sites (dropout gating). It
    /// deliberately does NOT store a GELU-seam arm flag: that one is a
    /// [`Self::forward_unnormalized_with_training`] PARAMETER — see that
    /// method's own doc.
    fn set_training(&mut self, training: bool) {
        for (_, lin) in self.lora_sites_mut() {
            lin.set_training(training);
        }
    }

    /// Project `[B, hidden_size]` to the unnormalized latent
    /// `[B, projection_dim]` in EVAL mode — the exact
    /// [`Self::forward_unnormalized_with_training`]`(x, false)` call, kept
    /// as its own entry point because this head's public callers (a
    /// boundary-parity harness such as `tests/golden_parity.rs`) are eval
    /// harnesses that hold no training flag of their own. A caller that IS
    /// training calls the `_with_training` twin with the tower's own flag,
    /// which is exactly what [`HtsatAudio::forward`] does.
    pub fn forward_unnormalized(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        self.forward_unnormalized_with_training(x, false)
    }

    /// [`Self::forward_unnormalized`] with the GELU seam's arm chosen by the
    /// caller's `training` flag, a PARAMETER rather than a stored copy (the
    /// rule `crate::activations::gelu_erf`'s own doc mandates —
    /// [`HtsatAudio::set_training`]'s flag is the single source of truth).
    ///
    /// The `"gelu"` arm goes through the house GELU seam
    /// (`crate::activations::gelu_erf`), never `Tensor::gelu_erf` directly —
    /// see this module's own doc. At `training == false` that seam IS the
    /// unchanged `Tensor::gelu_erf` call, byte for byte. The `"relu"` arm is
    /// a different activation with no fused seam and is untouched by
    /// `training` entirely.
    pub fn forward_unnormalized_with_training(
        &self,
        x: &Tensor,
        training: bool,
    ) -> Result<Tensor, EncoderError> {
        let x = self.linear1.forward(x)?;
        let x = match self.act.as_str() {
            "relu" => x.relu()?,
            GELU_PROJECTION_ACT => activations::gelu_erf(&x, training)?,
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
    /// Wired through [`Self::set_training`]; read by [`Self::is_training`].
    training: bool,
}

impl HtsatAudio {
    /// Start a builder with default settings: frozen LoRA config, F32
    /// backbone dtype, no adapter file. See [`HtsatAudioBuilder`].
    pub fn builder() -> HtsatAudioBuilder<'static> {
        HtsatAudioBuilder {
            lora: LoraBuildConfig::frozen(),
            backbone_dtype: DType::F32,
            adapter_file: None,
        }
    }

    /// Build the full tower from a root-scoped [`VarBuilder`] (the safetensors
    /// root holding both `audio_model` and `audio_projection`), fully frozen.
    ///
    /// Routed through the SAME `Self::load_with` the builder uses, with a
    /// decline-everything site, so `builder().lora(frozen()).build(..)` is
    /// bit-identical (asserted by
    /// `tests::builder_frozen_output_bits_equal_load`).
    pub fn load(
        vb: VarBuilder,
        config: &HtsatAudioConfig,
        device: &candle_core::Device,
    ) -> Result<Self, EncoderError> {
        let holder = FrozenSiteHolder::new();
        let ctx = HtsatLoadCtx {
            config,
            device,
            mask_dtype: vb.dtype(),
        };
        Self::load_with(
            vb.clone(),
            ctx,
            &|_s, _b| holder.site(&vb),
            &|_s| holder.site(&vb),
            &holder.site(&vb),
        )
    }

    /// The one loader — see [`HtsatAudioEncoder::load_with`] for the two
    /// stage-scoped site closures; `projection_site` is the UNINDEXED
    /// (`layer_idx == None`) site for the `audio_projection.*` head.
    fn load_with<'a>(
        vb: VarBuilder,
        ctx: HtsatLoadCtx<'_>,
        block_site: &dyn Fn(usize, usize) -> LoraSite<'a>,
        downsample_site: &dyn Fn(usize) -> LoraSite<'a>,
        projection_site: &LoraSite<'_>,
    ) -> Result<Self, EncoderError> {
        let config = ctx.config;
        let encoder = HtsatAudioEncoder::load_with(
            vb.pp("audio_model").pp("audio_encoder"),
            ctx,
            block_site,
            downsample_site,
        )?;
        let projection =
            ClapAudioProjection::load_with(vb.pp("audio_projection"), config, projection_site)?;
        Ok(Self {
            encoder,
            projection,
            projection_dim: config.projection_dim,
            num_mel_bins: config.num_mel_bins,
            training: false,
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

    /// Dtype the FROZEN BACKBONE weights are materialised at — read off a
    /// real weight (the patch-embedding `proj` convolution kernel, via
    /// [`HtsatAudioEncoder::dtype`]), never a remembered builder setting,
    /// so it stays true for a tower built through `load` from an arbitrary
    /// `VarBuilder` as well as one built through the builder's
    /// `backbone_dtype`.
    ///
    /// This is the dtype [`HtsatAudioEncoder::forward_front`] casts its
    /// input TO, and the dtype of the embedding [`Self::forward`] returns.
    /// It is NOT a precondition on the input: this tower's front half is a
    /// `BatchNorm` followed by a `Conv2d`, and neither promotes its input
    /// the way `jammi_lora::FrozenBase::Dense` does, which is exactly why
    /// the cast lives at that one seam rather than in every caller. A
    /// caller that MANUFACTURES a floating input (a probe batch, a warm-up)
    /// can still read it here to build one that needs no cast at all.
    pub fn dtype(&self) -> DType {
        self.encoder.dtype()
    }

    /// Borrow the underlying encoder (for boundary-level parity checks).
    pub fn encoder(&self) -> &HtsatAudioEncoder {
        &self.encoder
    }

    /// Borrow the projection head (for boundary-level parity checks).
    pub fn projection(&self) -> &ClapAudioProjection {
        &self.projection
    }

    /// Switch every Swin block's attention softmax between the eval
    /// (no-backward) arm and the differentiable composed arm — see
    /// `SwinSelfAttention::forward`'s doc for why the two arms exist. Eval
    /// output is unaffected either way; only backward through this tower's
    /// gradient (most visibly `rel_bias_table`, which otherwise gets no
    /// gradient at all) is correct in training mode.
    /// Also propagates to every LoRA site (Swin blocks, patch-merging
    /// downsamples, and the projection head) so their dropout is gated.
    ///
    /// The tower's `BatchNorm` deliberately does NOT change: it stays on
    /// `forward_t(x, false)` — eval statistics — even at `training == true`
    /// (`HtsatAudioEncoder::forward_front`). HF's own HTSAT pins it the
    /// same way, and a LoRA fine-tune leaves the backbone frozen, so
    /// recomputing batch statistics from a fine-tuning batch would silently
    /// change the frozen base's own numerics. Asserted by
    /// `tests::batch_norm_stays_on_eval_statistics_under_set_training`.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        self.encoder.set_training(training);
        self.projection.set_training(training);
    }

    /// Whether [`Self::set_training`] last set training mode. `false` from
    /// every constructor.
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Every LoRA site in the tower paired with its adapter key prefix, in
    /// one fixed traversal order — the single source
    /// [`Self::trainable_params`], [`Self::named_trainable_weights`],
    /// [`Self::dropout_positions`] and [`Self::restore_dropout_positions`]
    /// all walk, so a key can never be spelled two ways.
    ///
    /// Prefixes mirror the checkpoint's own path shape:
    /// `layers.{s}.blocks.{b}.{site}`, `layers.{s}.downsample.reduction`,
    /// `audio_projection.{linear1,linear2}`.
    fn lora_sites(&self) -> Vec<(String, &MaybeLoraLinear)> {
        let mut out = Vec::new();
        for (s, stage) in self.encoder.stages.iter().enumerate() {
            for (b, block) in stage.blocks.iter().enumerate() {
                for (site, lin) in block.lora_sites() {
                    out.push((format!("layers.{s}.blocks.{b}.{site}"), lin));
                }
            }
            if let Some(ds) = &stage.downsample {
                out.push((
                    format!("layers.{s}.downsample.{REDUCTION_SITE}"),
                    &ds.reduction,
                ));
            }
        }
        for (site, lin) in self.projection.lora_sites() {
            out.push((format!("audio_projection.{site}"), lin));
        }
        out
    }

    /// The `&mut` twin of `Self::lora_sites`, same prefixes, same order.
    fn lora_sites_mut(&mut self) -> Vec<(String, &mut MaybeLoraLinear)> {
        let mut out = Vec::new();
        for (s, stage) in self.encoder.stages.iter_mut().enumerate() {
            for (b, block) in stage.blocks.iter_mut().enumerate() {
                for (site, lin) in block.lora_sites_mut() {
                    out.push((format!("layers.{s}.blocks.{b}.{site}"), lin));
                }
            }
            if let Some(ds) = &mut stage.downsample {
                out.push((
                    format!("layers.{s}.downsample.{REDUCTION_SITE}"),
                    &mut ds.reduction,
                ));
            }
        }
        for (site, lin) in self.projection.lora_sites_mut() {
            out.push((format!("audio_projection.{site}"), lin));
        }
        out
    }

    /// This tower's [`crate::FusibleSiteCensus`], walked off the built Swin
    /// spine and projection head (see that type's own doc for what each
    /// field means).
    ///
    /// Every term is structure, and on this tower that matters more than on
    /// any other:
    ///
    /// * **LoRA sites** — `Self::lora_sites`, the same traversal the adapter
    ///   export walks, filtered to the wrapped arms. It already handles the
    ///   two irregularities a formula would have to encode by hand: the LAST
    ///   stage has no `downsample`, so it contributes no `reduction` site,
    ///   and the projection head's two linears belong to no stage at all.
    /// * **LayerNorms** — the patch-embed norm, then per stage the two norms
    ///   of every Swin block plus that stage's `downsample` norm when it has
    ///   one, then the encoder's final norm. The AFF fusion block's norms are
    ///   NOT here and must not be: they are `candle_nn::BatchNorm`, not the
    ///   house LayerNorm, so they cannot reach the `layer_norm_fused` seam.
    /// * **GELU** — one seam call per Swin block (`SwinBlock::forward`'s
    ///   MLP), plus ONE more when the projection head's activation is
    ///   `"gelu"`. The projection term is read off the BUILT head's own `act`
    ///   string, not off a config: `"relu"` (the committed tiny fixture and
    ///   HF's own default) contributes nothing, and `"gelu"` contributes
    ///   exactly one. Both site classes report to the SAME `gelu_erf_fused`
    ///   key, so this single number is what one forward's counter delta must
    ///   equal — see this module's own doc for the measured `sum(depths)` /
    ///   `sum(depths) + 1` oracles behind it.
    pub(crate) fn fusible_site_census(&self) -> crate::FusibleSiteCensus {
        crate::FusibleSiteCensus {
            lora_sites_wrapped: self
                .lora_sites()
                .into_iter()
                .filter(|(_, lin)| lin.is_lora())
                .count(),
            layer_norms: std::iter::once(&self.encoder.patch_embed.norm)
                .chain(self.encoder.stages.iter().flat_map(|stage| {
                    stage
                        .blocks
                        .iter()
                        .flat_map(|block| [&block.layernorm_before, &block.layernorm_after])
                        .chain(stage.downsample.iter().map(|merge| &merge.norm))
                }))
                .chain(std::iter::once(&self.encoder.norm))
                .count(),
            gelu_seam_calls_per_forward: self
                .encoder
                .stages
                .iter()
                .map(|stage| stage.blocks.len())
                .sum::<usize>()
                + usize::from(self.projection.act == GELU_PROJECTION_ACT),
        }
    }

    /// Trainable tensors across every LoRA-wrapped site. Empty for a fully
    /// frozen tower.
    pub fn trainable_params(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for (_, lin) in self.lora_sites() {
            params.extend(lin.trainable_params());
        }
        params
    }

    /// Named LoRA A/B tensors keyed `{site_prefix}.lora_{a,b}` — see
    /// `Self::lora_sites` for the prefix layout.
    pub fn named_trainable_weights(&self) -> Result<HashMap<String, Tensor>, EncoderError> {
        let mut out = HashMap::new();
        for (prefix, lin) in self.lora_sites() {
            out.extend(lin.named_weights(&prefix)?);
        }
        Ok(out)
    }

    /// Restore LoRA A/B tensors from a [`Self::named_trainable_weights`]-shaped
    /// map. Missing keys are no-ops.
    pub fn load_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<(), EncoderError> {
        for (prefix, lin) in self.lora_sites_mut() {
            lin.load_weights(weights, &prefix);
        }
        Ok(())
    }

    /// Per-site dropout-stream positions keyed `{site_prefix}.dropout`.
    pub fn dropout_positions(&self) -> Result<HashMap<String, u64>, EncoderError> {
        let mut out = HashMap::new();
        for (prefix, lin) in self.lora_sites() {
            lin.collect_dropout_position(&prefix, &mut out)?;
        }
        Ok(out)
    }

    /// Restore each LoRA site's dropout-stream position from a
    /// [`Self::dropout_positions`]-shaped map. Missing keys are no-ops.
    pub fn restore_dropout_positions(
        &self,
        positions: &HashMap<String, u64>,
    ) -> Result<(), EncoderError> {
        for (prefix, lin) in self.lora_sites() {
            lin.restore_dropout_position(&prefix, positions)?;
        }
        Ok(())
    }

    /// Full forward on `input_features` `[B, 4, T, num_mel_bins]` (any T,
    /// any floating dtype), with the per-sample `is_longer` fusion gate,
    /// returning the L2-normalized audio embedding `[B, projection_dim]` in
    /// this tower's own [`Self::dtype`].
    ///
    /// The input-dtype cast is NOT repeated here: this method's whole
    /// floating input goes to `HtsatAudioEncoder::forward_front`, which is
    /// the one seam that owns it (see that method's "Input-dtype domain"
    /// section). Everything downstream of it — the Swin spine, the
    /// projection head — consumes tensors this tower produced, already in
    /// the backbone dtype.
    pub fn forward(
        &self,
        input_features: &Tensor,
        is_longer: &[bool],
    ) -> Result<Tensor, EncoderError> {
        let front = self.encoder.forward_front(input_features, is_longer)?;
        let frames_num = front.post_reshape_mel2img.dim(2)?;
        let spine = self.encoder.forward_spine_with_training(
            &front.patch_embed_out,
            frames_num,
            self.training,
        )?;
        let unnorm = self
            .projection
            .forward_unnormalized_with_training(&spine.pooler_out, self.training)?;
        l2_normalize(&unnorm)
    }
}

/// Fluent builder for [`HtsatAudio`]. Created via [`HtsatAudio::builder`],
/// on the same shape [`crate::Bert`]'s and the OpenCLIP towers' builders use.
pub struct HtsatAudioBuilder<'a> {
    lora: LoraBuildConfig<'a>,
    backbone_dtype: DType,
    adapter_file: Option<&'a Path>,
}

impl<'a> HtsatAudioBuilder<'a> {
    /// LoRA adapter configuration: which of this tower's sites (`query`,
    /// `key`, `value`, `attention_output`, `intermediate_dense`,
    /// `output_dense`, `reduction`, `linear1`, `linear2`, or `all-linear`)
    /// get wrapped, at what rank.
    pub fn lora(mut self, l: LoraBuildConfig<'a>) -> Self {
        self.lora = l;
        self
    }

    /// Dtype the frozen backbone tensors — and every load-time shift-window
    /// mask — are materialised at. LoRA A/B always live in F32.
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

    /// Materialise the tower from frozen safetensors checkpoint files, the
    /// `VarBuilder` scoped at the checkpoint ROOT (which holds both
    /// `audio_model` and `audio_projection`), matching
    /// [`HtsatAudio::load`]'s own contract.
    pub fn build(
        self,
        weights_paths: &[&Path],
        config: &HtsatAudioConfig,
        device: &Device,
        varmap: &VarMap,
    ) -> Result<HtsatAudio, EncoderError> {
        let frozen_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(weights_paths, self.backbone_dtype, device)?
        };
        let trainable_vb = if let Some(adapter) = self.adapter_file {
            unsafe { VarBuilder::from_mmaped_safetensors(&[adapter], DType::F32, device)? }
        } else {
            VarBuilder::from_varmap(varmap, DType::F32, device)
        };
        // Trainable sub-builders materialised UP FRONT (so they outlive the
        // site closures), one per Swin block / per stage downsample / one for
        // the projection head — giving exactly the adapter keys
        // `HtsatAudio::named_trainable_weights` writes:
        // `layers.{s}.blocks.{b}.{site}`, `layers.{s}.downsample.reduction`,
        // `audio_projection.{linear1,linear2}`.
        let block_vbs: Vec<Vec<VarBuilder>> = config
            .depths
            .iter()
            .enumerate()
            .map(|(s, &depth)| {
                (0..depth)
                    .map(|b| trainable_vb.pp(format!("layers.{s}.blocks.{b}")))
                    .collect()
            })
            .collect();
        let downsample_vbs: Vec<VarBuilder> = (0..config.num_stages())
            .map(|s| trainable_vb.pp(format!("layers.{s}.downsample")))
            .collect();
        let projection_vb = trainable_vb.pp("audio_projection");
        let ctx = HtsatLoadCtx {
            config,
            device,
            mask_dtype: self.backbone_dtype,
        };
        HtsatAudio::load_with(
            frozen_vb,
            ctx,
            &|s, b| LoraSite {
                lora_vb: &block_vbs[s][b],
                // The STAGE index, not the block index — PEFT's rule; see
                // `SwinStage::load_with`'s doc.
                layer_idx: Some(s),
                lora: &self.lora,
                varmap,
            },
            &|s| LoraSite {
                lora_vb: &downsample_vbs[s],
                layer_idx: Some(s),
                lora: &self.lora,
                varmap,
            },
            &LoraSite {
                lora_vb: &projection_vb,
                // UNINDEXED: the projection head belongs to no stage.
                layer_idx: None,
                lora: &self.lora,
                varmap,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        assert_finite_nonzero, deterministic_fill_varmap, find_var, nonuniform_loss,
    };
    use candle_core::{DType, Device, Var};
    use candle_nn::VarMap;

    /// Deterministic `[rows, cols]` tensor via the same LCG, wide enough
    /// (`[-10, 10)`) to exercise the softmax max-shift and window-mask paths.
    fn deterministic_tensor(rows: usize, cols: usize, seed: u32, device: &Device) -> Tensor {
        let mut state = seed;
        let n = rows * cols;
        let values: Vec<f32> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let unit = (state >> 8) as f32 / (1u32 << 24) as f32;
                (unit - 0.5) * 20.0
            })
            .collect();
        Tensor::from_vec(values, (rows, cols), device).unwrap()
    }

    /// A minimal config exercising a `SwinStage` directly: `window_size=2`
    /// against a `4×4` grid (`min(4,4) > window_size`) forces `shift_size=0`
    /// on block 0 (W-MSA) and `shift_size=window/2=1` on block 1 (SW-MSA,
    /// masked) — both attention arms in one 2-block stage.
    fn tiny_stage_config() -> HtsatAudioConfig {
        HtsatAudioConfig {
            depths: vec![2],
            num_attention_heads: vec![2],
            window_size: 2,
            spec_size: 8,
            patch_size: 4,
            patch_stride: [4, 4],
            num_mel_bins: 8,
            patch_embeds_hidden_size: 8,
            hidden_size: 8,
            mlp_ratio: 2.0,
            projection_dim: 8,
            projection_hidden_act: "relu".to_string(),
            hidden_act: "gelu".to_string(),
            layer_norm_eps: 1e-5,
            enable_fusion: false,
            patch_embed_input_channels: 1,
            aff_block_r: 4,
            enable_patch_layer_norm: true,
            flatten_patch_embeds: true,
            qkv_bias: true,
        }
    }

    /// Build a 2-block `SwinStage` (`[4,4]` grid, `dim=8`, `heads=2`): block 0
    /// is W-MSA (`shift_size=0`, unmasked), block 1 is SW-MSA
    /// (`shift_size=1`, masked) — both attention arms in one stage.
    fn build_tiny_stage(varmap: &VarMap, device: &Device) -> SwinStage {
        let cfg = tiny_stage_config();
        let vb = VarBuilder::from_varmap(varmap, DType::F32, device);
        // Frozen (adapter-free) construction, through the same decline-
        // everything site the towers' own `load` entry points use.
        let holder = FrozenSiteHolder::new();
        let ctx = HtsatLoadCtx {
            config: &cfg,
            device,
            mask_dtype: DType::F32,
        };
        SwinStage::load_with(
            vb.clone(),
            ctx,
            8,
            2,
            2,
            (4, 4),
            false,
            &|_b| holder.site(&vb),
            &holder.site(&vb),
        )
        .unwrap()
    }

    /// Run `stage`'s blocks sequentially on a FIXED (non-`Var`) input tensor
    /// and return each block's `rel_bias_table` gradient plus the final
    /// output.
    ///
    /// `hidden` is deliberately a plain tensor with no computation history
    /// (not derived from any VarMap leaf): `SwinBlock::forward`'s residual
    /// connections make each block's OWN in-block leaf weights (query, key,
    /// value, `rel_bias_table`, `attention_output`) independently reachable
    /// from a downstream loss regardless of whether `hidden` itself is
    /// tracked — see [`SwinSelfAttention::forward`]'s doc: `rel_bias_table`'s
    /// reachability is a purely LOCAL property of the softmax arm, not of
    /// what feeds the block. This sidesteps HTSAT's SEPARATE, out-of-scope
    /// `candle_nn::LayerNorm` `BackpropOp::none()` truncation (its own
    /// `norm`/`layernorm_before`/`layernorm_after`/`PatchMerging::norm`),
    /// which would otherwise sever backward the moment the loss is routed
    /// through the tower's FINAL norm or a cross-stage `PatchMerging`
    /// downsample — this test stays scoped to a single stage's own blocks
    /// rather than the full `HtsatAudio::forward` for exactly that reason.
    ///
    /// `training` is passed through to [`SwinBlock::forward`] explicitly —
    /// that flag is a call-chain parameter, not stage state (see that
    /// method's own doc), so this helper's caller must supply the SAME value
    /// it handed `SwinStage::set_training`.
    fn run_stage_backward(
        stage: &SwinStage,
        varmap: &VarMap,
        device: &Device,
        training: bool,
    ) -> (Vec<Option<Tensor>>, Tensor) {
        let hidden = deterministic_tensor(16, 8, 11, device)
            .reshape((1, 16, 8))
            .unwrap();
        let mut x = hidden;
        for block in &stage.blocks {
            x = block.forward(&x, training).unwrap();
        }
        let loss = nonuniform_loss(&x, 8, device);
        let grads = loss.backward().unwrap();

        let rel_bias_grads = (0..stage.blocks.len())
            .map(|i| {
                let var = find_var(
                    varmap,
                    &format!("blocks.{i}.attention.self.relative_position_bias_table"),
                );
                grads.get(var.as_tensor()).cloned()
            })
            .collect();
        (rel_bias_grads, x)
    }

    /// RED oracle: fails if `SwinSelfAttention::forward`'s training arm is
    /// reverted to `softmax_last_dim` (or if `HtsatAudio::set_training`'s
    /// propagation down to `SwinStage`/`SwinBlock` is dropped) — under
    /// either regression `rel_bias_table`'s gradient comes back `None` for
    /// every block, same as the companion eval-mode test below. Exercises
    /// BOTH attention arms: block 0 is W-MSA (`shift_size=0`, unmasked),
    /// block 1 is SW-MSA (`shift_size=1`, masked).
    #[test]
    fn training_true_rel_bias_table_grad_is_some_for_every_block() {
        // `run_stage_backward` forwards each block's `layernorm_before`/
        // `layernorm_after` (biased, training mode) — a counter bumper on
        // `crate::layer_norm::LN_DISPATCH_COUNTERS` even though this test
        // never reads that counter itself (see
        // `crate::layer_norm::DISPATCH_COUNTER_TEST_LOCK`'s doc).
        let _guard = crate::layer_norm::DISPATCH_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let mut stage = build_tiny_stage(&varmap, &device);
        deterministic_fill_varmap(&varmap, &device);
        stage.set_training(true);

        let (rel_bias_grads, _) = run_stage_backward(&stage, &varmap, &device, true);
        for (i, grad) in rel_bias_grads.iter().enumerate() {
            let g = grad.as_ref().unwrap_or_else(|| {
                panic!("block {i}: rel_bias_table grad must be Some under training=true")
            });
            let norm: f32 = g
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                .sqrt();
            assert_finite_nonzero(norm, &format!("block {i}: rel_bias_table"));
        }
    }

    /// Documents the defect shape on the SAME fixture as
    /// [`training_true_rel_bias_table_grad_is_some_for_every_block`]: eval's
    /// `softmax_last_dim` never records a link back to `rel_bias_table` (its
    /// ONLY use anywhere in the tower — see `SwinSelfAttention::forward`'s
    /// doc), so `grads.get` comes back `None`, not zero — a step worse than
    /// CLIP's shared-weight case, where V's surviving path at least forces
    /// the accumulator to exist. Independent of the training-arm fix (eval
    /// always uses `softmax_last_dim`), so this stays green under the
    /// fix-verifier's revert; paired with the test above it also catches a
    /// dropped `set_training` propagation line.
    #[test]
    fn training_false_rel_bias_table_grad_is_none_for_every_block() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let stage = build_tiny_stage(&varmap, &device);
        deterministic_fill_varmap(&varmap, &device);
        // training defaults to false; forward without calling set_training.

        let (rel_bias_grads, _) = run_stage_backward(&stage, &varmap, &device, false);
        for (i, grad) in rel_bias_grads.iter().enumerate() {
            assert!(
                grad.is_none(),
                "block {i}: rel_bias_table grad must be None under eval (its only use is upstream \
                 of softmax_last_dim, which records no backward link at all)"
            );
        }
    }

    /// Build a VarMap-backed full `HtsatAudio` tower sized by the real
    /// `htsat_clap_tiny` config, and a fixed `[1, 4, t, num_mel_bins]` input.
    fn build_full_tower_and_input(
        varmap: &VarMap,
        device: &Device,
        seed: u32,
    ) -> (HtsatAudio, HtsatAudioConfig, Tensor) {
        let cfg = HtsatAudioConfig::from_hf_clap_config(&fixture_config()).unwrap();
        let vb = VarBuilder::from_varmap(varmap, DType::F32, device);
        let tower = HtsatAudio::load(vb, &cfg, device).unwrap();
        deterministic_fill_varmap(varmap, device);

        let t = 40;
        let input = deterministic_tensor(4 * t, cfg.num_mel_bins, seed, device)
            .reshape((1, 4, t, cfg.num_mel_bins))
            .unwrap();
        (tower, cfg, input)
    }

    /// L2 norm of a gradient tensor, via `grads.get(...)`.
    fn grad_norm(g: &Tensor) -> f32 {
        g.sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt()
    }

    /// Run the REAL front half (`HtsatAudioEncoder::forward_front`, public —
    /// batch-norm → bicubic time-resample → `reshape_mel2img` → the fused
    /// `patch_embed`, exercising BOTH the global-conv path (`proj` runs
    /// unconditionally inside `HtsatPatchEmbed::forward` regardless of
    /// `is_longer`) and the `mel_conv2d`/AFF-fusion local branch when
    /// `is_longer[i]` is `true`), then the full Swin spine (`forward_spine`,
    /// public), returning the pooled descriptor. With `mel_conv2d`'s tiling
    /// backward-sound (see [`tile_nonoverlapping`]), this is the full
    /// production composition — no caller of this helper needs to route
    /// around the fused local branch.
    ///
    /// The spine is entered through `forward_spine_with_training` with the
    /// TOWER'S OWN flag (`tower.is_training()`), not the flag-less eval
    /// convenience: this helper stands in for the production composition
    /// `HtsatAudio::forward` performs, and that one sources the GELU seam's
    /// arm from the same single flag.
    fn run_front_and_spine(tower: &HtsatAudio, input: &Tensor, is_longer: &[bool]) -> Spine {
        let encoder = tower.encoder();
        let front = encoder.forward_front(input, is_longer).unwrap();
        let frames_num = front.post_reshape_mel2img.dim(2).unwrap();
        encoder
            .forward_spine_with_training(&front.patch_embed_out, frames_num, tower.is_training())
            .unwrap()
    }

    /// End-to-end RED oracle through the REAL front half (`is_longer=[true]`,
    /// exercising the fused `mel_conv2d`/AFF path — see
    /// [`run_front_and_spine`]'s doc) and the full Swin spine. With BOTH the
    /// attention-softmax arm and every `LayerNorm` (patch-embed norm,
    /// `layernorm_before`/`after` and `PatchMerging::norm` per stage, the
    /// encoder's final norm) gated on `training`, AND `mel_conv2d`'s tiling
    /// backward-sound (its own reshape/permute have exact scatter
    /// backward passes — see [`tile_nonoverlapping`]), backward through the
    /// pooled descriptor reaches the patch-embed conv, `mel_conv2d`'s own
    /// conv weight, layer-0 Q/K, and EVERY stage/block's `rel_bias_table` —
    /// all 4 stages, all `sum(depths)=8` blocks. Fails if any of the THREE
    /// independent invariants regresses: reverting the softmax arm leaves
    /// every `rel_bias_table` severed the same way the eval companion below
    /// shows; reverting the final norm's gate severs backward before it
    /// reaches ANY stage (the patch-embed conv assertion fails first);
    /// reverting [`tile_nonoverlapping`] to a plain `unfold` either errors
    /// `loss.backward()` outright (the non-dividing case) or SUCCEEDS with a
    /// silently-permuted gradient (a dividing case, or any case where
    /// candle's flat-reshape backward happens not to error) — this test
    /// alone cannot distinguish "wrong but finite" from "correct"; see
    /// [`tests::mel_conv2d_gradient_matches_the_hand_built_scatter_at_production_geometry`]
    /// for the oracle that catches the wrong-gradient failure mode this one
    /// cannot.
    #[test]
    fn training_true_full_forward_reaches_every_parameter() {
        // Full-tower forward through the patch-embed norm, every stage's
        // `layernorm_before`/`layernorm_after`/`PatchMerging::norm`, and the
        // encoder's final norm at `training=true` bumps
        // `crate::layer_norm::LN_DISPATCH_COUNTERS` even though this test
        // never reads it (see `crate::layer_norm::DISPATCH_COUNTER_TEST_LOCK`'s
        // doc).
        let _guard = crate::layer_norm::DISPATCH_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let (mut tower, cfg, input) = build_full_tower_and_input(&varmap, &device, 101);
        tower.set_training(true);

        let spine = run_front_and_spine(&tower, &input, &[true]);
        let loss = nonuniform_loss(&spine.pooler_out, cfg.hidden_size, &device);
        let grads = loss.backward().unwrap();

        let patch_conv = find_var(&varmap, "patch_embed.proj.weight");
        assert!(
            grads.get(patch_conv.as_tensor()).is_some(),
            "patch-embed conv weight grad must be Some under training=true through the full forward"
        );

        let mel_conv_weight = find_var(&varmap, "mel_conv2d.weight");
        let mel_conv_grad = grads.get(mel_conv_weight.as_tensor()).expect(
            "mel_conv2d.weight grad must be Some under training=true through the full forward \
             (the fused/AFF local branch, is_longer=true) — this is the assertion that FAILS \
             (loss.backward() itself returns Err before reaching it) without \
             tile_nonoverlapping's backward-sound reshape/permute tiling",
        );
        let mel_conv_norm = grad_norm(mel_conv_grad);
        assert_finite_nonzero(mel_conv_norm, "mel_conv2d.weight");

        let q0 = find_var(&varmap, "layers.0.blocks.0.attention.self.query.weight");
        let k0 = find_var(&varmap, "layers.0.blocks.0.attention.self.key.weight");
        let q_grad = grads
            .get(q0.as_tensor())
            .expect("layer-0 query.weight grad must be Some under training=true");
        let k_grad = grads
            .get(k0.as_tensor())
            .expect("layer-0 key.weight grad must be Some under training=true");
        // A bare `norm > 0.0` positive control leaves a hole open at `+inf`,
        // not at `NaN`: `NaN > 0.0` is `false`, so `assert!(norm > 0.0)`
        // already panics (correctly fails) on a NaN-poisoned gradient, but
        // `+inf > 0.0` is `true`, so an exploded gradient would silently
        // satisfy a bare `> 0.0` check. `assert_finite_nonzero` checks
        // `is_finite()` first, closing that hole with its own message.
        let q_norm = grad_norm(q_grad);
        let k_norm = grad_norm(k_grad);
        assert_finite_nonzero(q_norm, "layer-0 query.weight");
        assert_finite_nonzero(k_norm, "layer-0 key.weight");

        for s in 0..cfg.num_stages() {
            for b in 0..cfg.depths[s] {
                let suffix =
                    format!("layers.{s}.blocks.{b}.attention.self.relative_position_bias_table");
                let var = find_var(&varmap, &suffix);
                let g = grads.get(var.as_tensor()).unwrap_or_else(|| {
                    panic!(
                        "stage {s} block {b}: rel_bias_table grad must be Some under training=true"
                    )
                });
                let norm = grad_norm(g);
                assert_finite_nonzero(norm, &format!("stage {s} block {b}: rel_bias_table"));
            }
        }

        // BLANKET oracle: every Var in the VarMap — not just the
        // hand-picked patch-embed/mel_conv2d/layer-0-Q-K/rel_bias_table
        // subset above — must receive a Some/finite/nonzero gradient.
        // EXCLUDED, two disjoint reasons:
        //  - `running_mean`/`running_var`: the five `BatchNorm` instances'
        //    buffers (the top-level `batch_norm` plus the fused AFF
        //    block's `local_bn1`/`local_bn4`/`global_bn2`/`global_bn5`).
        //    Non-differentiable BY CONSTRUCTION: candle's
        //    `BatchNorm::forward_t(_, train=false)` (this tower always
        //    runs eval-mode batch-norm statistics, regardless of the
        //    tower's own `training` flag — see
        //    `HtsatAudioEncoder::forward_front`'s `forward_t(&x, false)`
        //    call) reads `running_mean`/`running_var` as plain tensors,
        //    never through an autodiff op, so no `Op` node ever links a
        //    loss back to them.
        //  - `audio_projection.*` (the `ClapAudioProjection` head's two
        //    linears): OUT OF SCOPE for this test's own composition, not a
        //    tower defect. [`run_front_and_spine`] stops at
        //    `spine.pooler_out`, one step short of `HtsatAudio::forward`'s
        //    own `self.projection.forward_unnormalized(...)` call (see
        //    that method's doc) — `audio_projection`'s weights are simply
        //    never read by the graph this test builds, so `grads.get`
        //    correctly returns `None` for them independent of anything
        //    this test is actually measuring.
        // Measured on this fixture: 171 of 185 Vars present, exactly the
        // union of the 10 BatchNorm buffers and the 4 audio_projection
        // weights excluded above.
        crate::test_support::assert_every_var_has_gradient(
            &varmap,
            &grads,
            &["running_mean", "running_var", "audio_projection"],
        );
    }

    /// The eval-mode observable a user of this tower would actually hit
    /// before either gate existed: the full spine's backward (same REAL
    /// front half as the companion test above, `is_longer=[true]`, for a
    /// like-for-like comparison — see [`run_front_and_spine`]'s doc) yields
    /// NO gradient entry AT ALL for the patch-embed conv, `mel_conv2d`'s
    /// conv weight, layer-0 Q/K, or any `rel_bias_table`, because the
    /// encoder's final `norm` (`candle_nn::LayerNorm`'s `BackpropOp::none()`
    /// truncation) severs backward before it reaches ANY stage — independent
    /// of the softmax arm, which is a SEPARATE, strictly-worse truncation
    /// one hop earlier inside each block (see
    /// `training_false_rel_bias_table_grad_is_none_for_every_block`, which
    /// isolates that one), and independent of `mel_conv2d`'s tiling fix
    /// (eval never runs `loss.backward()` past `norm`, so it never reaches
    /// far enough upstream to observe whether `mel_conv2d`'s own backward
    /// would have succeeded).
    #[test]
    fn training_false_full_forward_grads_are_none_before_final_norm() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let (tower, cfg, input) = build_full_tower_and_input(&varmap, &device, 101);
        // training defaults to false; forward without calling set_training.

        let spine = run_front_and_spine(&tower, &input, &[true]);
        let loss = nonuniform_loss(&spine.pooler_out, cfg.hidden_size, &device);
        let grads = loss.backward().unwrap();

        let patch_conv = find_var(&varmap, "patch_embed.proj.weight");
        assert!(
            grads.get(patch_conv.as_tensor()).is_none(),
            "patch-embed conv weight grad must be None under eval through the full forward \
             (the encoder's final norm truncates backward before it reaches any stage)"
        );

        let mel_conv_weight = find_var(&varmap, "mel_conv2d.weight");
        assert!(
            grads.get(mel_conv_weight.as_tensor()).is_none(),
            "mel_conv2d.weight grad must be None under eval through the full forward"
        );

        let q0 = find_var(&varmap, "layers.0.blocks.0.attention.self.query.weight");
        let k0 = find_var(&varmap, "layers.0.blocks.0.attention.self.key.weight");
        assert!(
            grads.get(q0.as_tensor()).is_none(),
            "layer-0 query.weight grad must be None under eval"
        );
        assert!(
            grads.get(k0.as_tensor()).is_none(),
            "layer-0 key.weight grad must be None under eval"
        );

        for s in 0..cfg.num_stages() {
            for b in 0..cfg.depths[s] {
                let suffix =
                    format!("layers.{s}.blocks.{b}.attention.self.relative_position_bias_table");
                let var = find_var(&varmap, &suffix);
                assert!(
                    grads.get(var.as_tensor()).is_none(),
                    "stage {s} block {b}: rel_bias_table grad must be None under eval"
                );
            }
        }

        // BLANKET oracle: every trainable Var in the VarMap is severed
        // (same `running_mean`/`running_var` exclusion as the training=true
        // companion oracle — those buffers are non-differentiable by
        // construction either way, see that test's doc).
        crate::test_support::assert_every_var_grad_is_none(
            &varmap,
            &grads,
            &["running_mean", "running_var"],
        );
    }

    /// Deletion-catching oracle for [`SwinBlock`]'s residual-stream
    /// LayerNorms themselves (`layernorm_before`/`layernorm_after`, layer-0
    /// block-0): every OTHER gradient assertion above reaches its target
    /// parameter through the block's residual bypass (`shortcut + attn` /
    /// `hidden + mlp_out` in [`SwinBlock::forward`]), so a dropped
    /// `self.layernorm_before.set_training(training)` /
    /// `self.layernorm_after.set_training(training)` line — leaving that
    /// ONE LayerNorm stuck on its fused, `BackpropOp::none()`-truncated
    /// eval arm even with the rest of the tower `training=true` — would
    /// NOT be caught by any test above: the residual path still carries a
    /// gradient to `mel_conv2d.weight`/`query.weight`/`rel_bias_table`
    /// regardless of `layernorm_before`/`layernorm_after`'s own truncation
    /// (this is the SAME shape as `training_false_full_forward_grads_are_none_before_final_norm`'s
    /// note that the encoder's final norm is a SEPARATE truncation from
    /// this one). This test asserts `layernorm_before`/`layernorm_after`'s
    /// OWN `weight` — not anything upstream — through
    /// [`run_front_and_spine`] (`forward_front` composed with
    /// `forward_spine`, stopping at `spine.pooler_out`), NOT the full
    /// public [`HtsatAudio::forward`]: the skipped tail is
    /// `ClapAudioProjection::forward_unnormalized` (linear, activation,
    /// linear) plus `l2_normalize`, none of which touch `layernorm_before`/
    /// `layernorm_after` either way, so composing up to `pooler_out` is
    /// sufficient for this assertion without pulling in the projection
    /// head's own weights as noise. `Some`/finite/nonzero under
    /// `training=true`, `None` under `training=false`. RED-verified: deleting
    /// `self.layernorm_before.set_training(training)` from
    /// `SwinBlock::set_training` flips the training=true half of this test
    /// (`layernorm_before.weight` comes back `None` instead of `Some`)
    /// while every other test in this file stays green.
    #[test]
    fn layernorm_before_and_after_own_weight_gradient_present_under_training_absent_under_eval() {
        let device = Device::Cpu;

        for name in [
            "layers.0.blocks.0.layernorm_before.weight",
            "layers.0.blocks.0.layernorm_after.weight",
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
                let (mut tower, cfg, input) = build_full_tower_and_input(&varmap, &device, 303);
                tower.set_training(true);
                let spine = run_front_and_spine(&tower, &input, &[true]);
                let loss = nonuniform_loss(&spine.pooler_out, cfg.hidden_size, &device);
                let grads = loss.backward().unwrap();
                let var = find_var(&varmap, name);
                grads.get(var.as_tensor()).cloned()
            };
            let grad = training_grad
                .unwrap_or_else(|| panic!("{name} grad must be Some under training=true"));
            let norm = grad_norm(&grad);
            assert_finite_nonzero(norm, &format!("{name} (training=true)"));

            let eval_grad = {
                let varmap = VarMap::new();
                let (tower, cfg, input) = build_full_tower_and_input(&varmap, &device, 303);
                // training defaults to false; forward without calling set_training.
                let spine = run_front_and_spine(&tower, &input, &[true]);
                let loss = nonuniform_loss(&spine.pooler_out, cfg.hidden_size, &device);
                let grads = loss.backward().unwrap();
                let var = find_var(&varmap, name);
                grads.get(var.as_tensor()).cloned()
            };
            assert!(
                eval_grad.is_none(),
                "{name} grad must be None under training=false"
            );
        }
    }

    /// Direct, isolated backward test for [`HtsatPatchEmbed::forward`]'s
    /// fused (`is_longer=true`) branch reaching `mel_conv2d`'s own conv
    /// weight — narrower than the full-tower oracle above (stops at
    /// `patch_embed`, not the whole Swin spine), and the exact test that
    /// FAILED before [`tile_nonoverlapping`]'s narrow-before-unfold fix.
    ///
    /// Pre-fix failure signature (reproduced in isolation by
    /// [`unfold_backward_is_a_plain_reshape_in_candle`] on a minimal
    /// fixture; MEASURED here by temporarily reverting `mel_conv2d`'s tiling
    /// to plain `x.unfold(2, kh, kh)?.unfold(3, kw, kw)?` with no `narrow`
    /// first): on the real `htsat_clap_tiny` geometry (`spec_size=128`,
    /// `kw=patch_size*3=12`, `128 / 12 = 10` windows with an 8-wide tail
    /// dropped, `patch_embed_input_channels=3` local channels, batch=1),
    /// `loss.backward()` itself returned
    /// `Err("shape mismatch in reshape, lhs: [3, 1, 32, 10, 4, 12], rhs:
    /// [3, 1, 32, 128, 4]")` — `unfold`'s `Op::Reshape` backward trying to
    /// reshape the width-unfold's gradient (`[..., 10, 4, 12]`, sized by the
    /// DROPPED-tail 120-of-128 output) back to the pre-width-unfold input
    /// shape (`[..., 128, 4]`, the full un-narrowed 128) — a loud `Err`, not
    /// a panic, but this test's `.expect(...)` on that `Result` is what
    /// turns it into a failing assertion — before any `grads.get(...)` call
    /// was even reached.
    #[test]
    fn mel_conv2d_backward_reaches_conv_weight_through_the_fused_patch_embed_path() {
        // `patch_embed.forward` below ends in `self.norm.forward(&flat)`
        // (biased, training mode) — a counter bumper on
        // `crate::layer_norm::LN_DISPATCH_COUNTERS` even though this test
        // never reads that counter itself (see
        // `crate::layer_norm::DISPATCH_COUNTER_TEST_LOCK`'s doc).
        let _guard = crate::layer_norm::DISPATCH_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let (mut tower, cfg, input) = build_full_tower_and_input(&varmap, &device, 202);
        tower.set_training(true);
        let encoder = tower.encoder();

        // Front half up to patch_embed's own input, matching
        // HtsatAudioEncoder::forward_front's composition exactly (see that
        // method), stopping one step short of it so this test can reach
        // into HtsatPatchEmbed::forward's own boundary outputs directly.
        let x = input.transpose(1, 3).unwrap().contiguous().unwrap();
        let post_bn = encoder.batch_norm.forward_t(&x, false).unwrap();
        let normalized = post_bn.transpose(1, 3).unwrap().contiguous().unwrap();
        let post_interp = encoder.time_interp.forward(&normalized).unwrap();
        let post_reshape = encoder.reshape_mel2img(&post_interp).unwrap();

        let mut mel_conv2d_out = None;
        let mut fusion_out = None;
        let patch_embed_out = encoder
            .patch_embed
            .forward(&post_reshape, &[true], &mut mel_conv2d_out, &mut fusion_out)
            .unwrap();

        let loss = nonuniform_loss(&patch_embed_out, cfg.patch_embeds_hidden_size, &device);
        let grads = loss.backward().expect(
            "backward through the fused (is_longer=true) patch-embed path must succeed now \
             that mel_conv2d's tiling narrows before unfolding — see this test's doc for the \
             exact pre-fix Err this .expect(...) turns into a failing assertion for",
        );

        let mel_conv_weight = find_var(&varmap, "mel_conv2d.weight");
        let grad = grads
            .get(mel_conv_weight.as_tensor())
            .expect("mel_conv2d.weight grad must be Some through the fused patch-embed path");
        let norm = grad_norm(grad);
        assert_finite_nonzero(norm, "mel_conv2d.weight");
    }

    /// Gradient-VALUE oracle (a): central finite differences on a small
    /// non-dividing fixture prove [`tile_nonoverlapping`]'s analytic
    /// gradient is not just finite and nonzero but numerically CORRECT —
    /// the oracle the old unfold-based tiling never had. `tile_nonoverlapping`
    /// is a pure linear map of `x` (narrow, reshape, permute — no
    /// nonlinearity anywhere), so `loss = sum(w * tile(x))` for a
    /// non-uniform `w` is exactly linear in `x`: central differences have
    /// ZERO truncation error here (unlike a general nonlinear function),
    /// so a tight tolerance is legitimate — any residual gap is purely
    /// f32 rounding. `[1, 1, 9, 10]`, `kh=2, kw=3`: neither `9 / 2` (row
    /// 8 dropped) nor `10 / 3` (column 9 dropped) divides exactly, so
    /// this fixture is genuinely non-dividing on BOTH axes — the FD loop
    /// below runs over every element including the dropped row/column, so
    /// it also proves the dropped-tail elements get exactly zero gradient.
    /// The production-geometry scatter oracle below covers a
    /// dropped-tail-on-one-axis-only case at full production scale, which
    /// does not check EVERY element (72x more elements) the way this one
    /// does.
    #[test]
    fn tile_nonoverlapping_gradient_matches_central_finite_differences() {
        let device = Device::Cpu;
        let (n, c, h, w) = (1usize, 1usize, 9usize, 10usize);
        let (kh, kw) = (2usize, 3usize);
        let numel = n * c * h * w;

        let mut state: u32 = 3;
        let mut next = || {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 4.0 // [-2, 2)
        };
        let x_vals: Vec<f32> = (0..numel).map(|_| next()).collect();

        let (oh, ow) = (h / kh, w / kw);
        let tile_numel = n * c * oh * ow * kh * kw;
        let w_vals: Vec<f32> = (0..tile_numel).map(|_| next()).collect();

        let loss_value = |xs: &[f32]| -> f32 {
            let x = Tensor::from_vec(xs.to_vec(), (n, c, h, w), &device).unwrap();
            let tiled = tile_nonoverlapping(&x, kh, kw).unwrap();
            let weights = Tensor::from_vec(w_vals.clone(), tiled.dims().to_vec(), &device).unwrap();
            (&tiled * &weights)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        };

        let xv =
            Var::from_tensor(&Tensor::from_vec(x_vals.clone(), (n, c, h, w), &device).unwrap())
                .unwrap();
        let tiled = tile_nonoverlapping(xv.as_tensor(), kh, kw).unwrap();
        let weights = Tensor::from_vec(w_vals.clone(), tiled.dims().to_vec(), &device).unwrap();
        let loss = (&tiled * &weights).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let analytic: Vec<f32> = grads
            .get(xv.as_tensor())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let eps = 1e-2f32;
        for i in 0..numel {
            let mut plus = x_vals.clone();
            plus[i] += eps;
            let mut minus = x_vals.clone();
            minus[i] -= eps;
            let fd = (loss_value(&plus) - loss_value(&minus)) / (2.0 * eps);

            let diff = (analytic[i] - fd).abs();
            let tol = 1e-4 * fd.abs().max(1.0); // relative-with-floor — no-producer: the floor guards the fd ~ 0 regime, a chosen margin, not a measurement.
            assert!(
                diff <= tol,
                "element {i}: analytic grad {} vs central-difference {fd} differ by {diff} \
                 (tol {tol})",
                analytic[i]
            );
        }
    }

    /// Shared body for the gradient-VALUE scatter oracle (b): builds a
    /// deterministic `x` and non-uniform weight tensor at the given
    /// geometry, runs `tile_nonoverlapping` forward+backward, and asserts
    /// the analytic gradient exactly equals a hand-built scatter:
    /// `dx[n, c, oh*kh+i, ow*kw+j] = w[n, c, oh, ow, i, j]` for every
    /// retained element and exactly `0.0` for any dropped tail. This is a
    /// pure scatter — every retained `x` element contributes to EXACTLY
    /// one tile slot, so there is no floating-point accumulation to give
    /// tolerance to: `assert_eq!` on the f32 values is the correct
    /// comparison here, not an approximation.
    fn assert_tile_nonoverlapping_gradient_matches_scatter(
        (n, c, h, w): (usize, usize, usize, usize),
        (kh, kw): (usize, usize),
        seed: u32,
        label: &str,
    ) {
        let device = Device::Cpu;
        let (oh, ow) = (h / kh, w / kw);

        let x_vals = deterministic_tensor(n * c * h, w, seed, &device)
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let tile_numel = n * c * oh * ow * kh * kw;
        // Bounded, non-uniform weights (not the unbounded `1.0 + i*0.37`
        // idiom used elsewhere — at this element count that would reach
        // into the tens of thousands and lose f32 precision).
        let w_vals: Vec<f32> = (0..tile_numel)
            .map(|i| 1.0 + (i % 97) as f32 * 0.013)
            .collect();

        let xv =
            Var::from_tensor(&Tensor::from_vec(x_vals, (n, c, h, w), &device).unwrap()).unwrap();
        let tiled = tile_nonoverlapping(xv.as_tensor(), kh, kw).unwrap();
        let weights = Tensor::from_vec(w_vals.clone(), (n, c, oh, ow, kh, kw), &device).unwrap();
        let loss = (&tiled * &weights).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let analytic: Vec<f32> = grads
            .get(xv.as_tensor())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let mut expected = vec![0f32; n * c * h * w];
        for ni in 0..n {
            for ci in 0..c {
                for ohi in 0..oh {
                    for owi in 0..ow {
                        for i in 0..kh {
                            for j in 0..kw {
                                let src_h = ohi * kh + i;
                                let src_w = owi * kw + j;
                                let x_idx = ((ni * c + ci) * h + src_h) * w + src_w;
                                let wt_idx =
                                    ((((ni * c + ci) * oh + ohi) * ow + owi) * kh + i) * kw + j;
                                expected[x_idx] = w_vals[wt_idx];
                            }
                        }
                    }
                }
            }
        }

        assert_eq!(
            analytic, expected,
            "tile_nonoverlapping's gradient at the {label} geometry must exactly equal the \
             hand-built scatter (element-for-element — this is a pure copy, no arithmetic \
             accumulation, so bit-exact equality is the correct comparison)"
        );
    }

    /// Gradient-VALUE oracle (b): at the exact production geometry
    /// (`[3, 1, 128, 128]`, `kh=4, kw=12` — `mel_conv2d`'s real shape, see
    /// `config_parity_with_fixture`), [`tile_nonoverlapping`]'s analytic
    /// gradient matches the hand-built scatter, including exactly `0.0`
    /// for the dropped 8-column tail (`w=128`, `kw=12`, `128 / 12 = 10`
    /// windows, columns `120..128` never read; `h=128`, `kh=4` divides
    /// exactly, so height has no dropped tail here — the complementary
    /// dropped-ROW case is covered by
    /// [`mel_conv2d_gradient_matches_the_hand_built_scatter_with_dropped_row_tail`]
    /// below, since narrowing on dim 2 (height) exercises a different
    /// stride/permute path than narrowing on dim 3 (width) alone).
    #[test]
    fn mel_conv2d_gradient_matches_the_hand_built_scatter_at_production_geometry() {
        assert_tile_nonoverlapping_gradient_matches_scatter(
            (3, 1, 128, 128),
            (4, 12),
            71,
            "production (W-tail dropped, H exact)",
        );
    }

    /// Gradient-VALUE oracle (b'): the complementary dropped-ROW-tail case
    /// — `[2, 1, 9, 8]`, `kh=2, kw=2`: `h=9, kh=2` drops row `8` (narrow on
    /// dim 2, the H axis), while `w=8, kw=2` divides exactly (no dropped
    /// column). Case (b) above only ever exercises a dropped-tail on the W
    /// axis (dim 3); `tile_nonoverlapping`'s narrow-then-reshape-then-permute
    /// composition treats dim 2 and dim 3 asymmetrically (the permute order
    /// is fixed), so a narrow-on-dim-2 backward is a materially different
    /// code path from narrow-on-dim-3 and must be measured on its own
    /// fixture, not assumed to work because (b) passed.
    #[test]
    fn mel_conv2d_gradient_matches_the_hand_built_scatter_with_dropped_row_tail() {
        assert_tile_nonoverlapping_gradient_matches_scatter(
            (2, 1, 9, 8),
            (2, 2),
            97,
            "dropped-row-tail (H narrowed, W exact)",
        );
    }

    /// Domain-validity oracle: when the input spatial dims are smaller than
    /// the tile kernel (`h < kh` or `w < kw`), `oh` or `ow` floors to zero
    /// and a narrow-then-reshape without a guard would silently produce an
    /// empty tensor instead of surfacing the caller's mistake. Both the
    /// height-starved and the width-starved case must return a typed
    /// [`EncoderError`], not an empty `Ok`.
    #[test]
    fn tile_nonoverlapping_rejects_input_smaller_than_the_kernel() {
        let device = Device::Cpu;

        // h=1 < kh=2: oh would floor to 0.
        let x = Tensor::zeros((1usize, 1usize, 1usize, 10usize), DType::F32, &device).unwrap();
        let err = tile_nonoverlapping(&x, 2, 3)
            .expect_err("h < kh must be rejected, not silently return an empty tensor");
        assert!(
            matches!(err, EncoderError::Config(_)),
            "h < kh must fail with EncoderError::Config, got {err:?}"
        );

        // w=1 < kw=3: ow would floor to 0.
        let x = Tensor::zeros((1usize, 1usize, 10usize, 1usize), DType::F32, &device).unwrap();
        let err = tile_nonoverlapping(&x, 2, 3)
            .expect_err("w < kw must be rejected, not silently return an empty tensor");
        assert!(
            matches!(err, EncoderError::Config(_)),
            "w < kw must fail with EncoderError::Config, got {err:?}"
        );
    }

    /// Reference tiling: `unfold` applied directly, with no `narrow` first —
    /// reproduces exactly what `mel_conv2d` did before [`tile_nonoverlapping`]
    /// existed. Used only to prove `tile_nonoverlapping`'s forward stays
    /// bit-identical to this reference.
    fn unfold_only_tiling(x: &Tensor, kh: usize, kw: usize) -> Tensor {
        x.unfold(2, kh, kh).unwrap().unfold(3, kw, kw).unwrap()
    }

    /// Forward bit-identity oracle: [`tile_nonoverlapping`]'s
    /// narrow-then-reshape-then-permute tiling must read the IDENTICAL
    /// elements as `unfold` did (the plain-`unfold` reference tiling above),
    /// on the exact non-dividing shape `mel_conv2d` hits in production
    /// (`spec_size=128`, `kw=patch_size*3=12`; see `config_parity_with_fixture`
    /// for where these numbers come from — `128 / 12 = 10` windows, an
    /// 8-wide tail dropped either way; `n=3` matches the fused patch-embed's
    /// local branch, `patch_embed_input_channels=3` local channels folded
    /// into the batch axis). This is the oracle behind `golden_parity`'s
    /// 6/6 staying green: both tilings only ever read the same
    /// `floor(dim/size) * size` prefix, so changing HOW that prefix is
    /// reshaped into tiles changes nothing about which elements FORWARD
    /// reads — only whether BACKWARD is exact.
    #[test]
    fn tile_nonoverlapping_is_forward_bit_identical_to_unfold_without_narrow_first() {
        let device = Device::Cpu;
        let (n, c, h, w) = (3usize, 1usize, 128usize, 128usize);
        let (kh, kw) = (4usize, 12usize);
        let x = deterministic_tensor(n * c * h, w, 55, &device)
            .reshape((n, c, h, w))
            .unwrap();

        let narrowed = tile_nonoverlapping(&x, kh, kw).unwrap();
        let direct = unfold_only_tiling(&x, kh, kw);

        assert_eq!(
            narrowed.dims(),
            direct.dims(),
            "reshape/permute tiling and unfold-only must tile to the identical shape"
        );
        let a: Vec<f32> = narrowed
            .contiguous()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let b: Vec<f32> = direct
            .contiguous()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(
            a, b,
            "reshape/permute tiling must read the identical elements as unfold alone (the \
             dropped-tail, non-dividing case)"
        );
    }

    /// Premise-pin (the repo idiom for a documented upstream limitation,
    /// e.g. `jammi-kernels`' `cpu_matmul_still_cannot_do_bf16`): pins the
    /// exact candle-core behavior that makes `Tensor::unfold`'s backward
    /// unsafe to depend on directly — a flat `Op::Reshape` regardless of
    /// which dim was unfolded (see [`tile_nonoverlapping`]'s own doc for
    /// the exact candle-core source citations this pins,
    /// `tensor.rs:2931-2969`, `backprop.rs:602-606`, `tensor.rs:2523-2531`)
    /// — so a future candle upgrade that fixes `unfold`'s backward flips
    /// THIS test red, a loud signal that `tile_nonoverlapping` could go
    /// back to `unfold` directly, instead of silently leaving the
    /// workaround unexplained and untested.
    #[test]
    fn unfold_backward_is_a_plain_reshape_in_candle() {
        let device = Device::Cpu;
        // dim length 5, size=2, step=2: 2 windows read 4 of the 5 elements
        // (an element dropped) — the minimal non-dividing case.
        let x =
            Var::from_tensor(&Tensor::from_vec(vec![1f32, 2., 3., 4., 5.], 5, &device).unwrap())
                .unwrap();
        let tiled = x.as_tensor().unfold(0, 2, 2).unwrap(); // [2, 2]
        let loss = tiled.sum_all().unwrap();
        let err = loss.backward().expect_err(
            "unfold's backward on a non-dividing dim must still error today (the candle \
             premise this crate avoids depending on) — if this now succeeds, candle may have \
             fixed unfold's backward",
        );
        let msg = err.to_string();
        assert!(
            msg.contains("shape mismatch") && msg.contains("reshape"),
            "expected candle's reshape-shape-mismatch error from unfold's Op::Reshape backward; \
             got: {msg}"
        );
    }

    /// Premise-pin, companion to [`unfold_backward_is_a_plain_reshape_in_candle`]:
    /// on a NON-last-dim unfold whose size evenly divides the unfolded dim
    /// (so `grad.reshape(arg.dims())` does NOT error — the element counts
    /// match), candle's backward still returns a WRONG gradient, silently.
    /// `x` is `[4, 2]`; `x.unfold(0, 2, 2)` unfolds dim 0 (NOT `x`'s last
    /// dim, which is dim 1) into a new trailing dim, giving `[2, 2, 2]`
    /// (`windows, orig_dim1, in_window`) — structurally the same shape as
    /// `tile_nonoverlapping`'s width-then-height unfold chain, just 2-D
    /// instead of 4-D. `loss = sum(w * unfold(x))` for a distinct-valued
    /// `w` backward-succeeds (8 elements both sides), but the resulting
    /// `x` gradient does NOT match the hand-derived correct gradient
    /// `dx[row, col] = w[row / 2, col, row % 2]` (derived from unfold's own
    /// FORWARD index arithmetic, `tensor.rs:2931-2969`) at 4 of its 8
    /// positions — this is the exact silent-wrong-gradient failure mode a
    /// width/height `unfold` chain is exposed to, the reason
    /// [`tile_nonoverlapping`] instead uses `reshape`/`permute` (whose
    /// backward passes are exact scatters, not a flat reshape). If this
    /// test ever starts PASSING (the two grads agreeing), candle has fixed
    /// unfold's backward
    /// to account for which dim was unfolded, and the day this happens is
    /// the day `tile_nonoverlapping` could safely go back to `unfold`.
    #[test]
    fn unfold_backward_on_a_non_last_dim_is_wrong_but_finite() {
        let device = Device::Cpu;
        let x_vals: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let xv = Var::from_tensor(&Tensor::from_vec(x_vals, (4, 2), &device).unwrap()).unwrap();
        let tiled = xv.as_tensor().unfold(0, 2, 2).unwrap(); // [2, 2, 2]; dim 0, not the last dim.

        let w_vals: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let w = Tensor::from_vec(w_vals.clone(), (2, 2, 2), &device).unwrap();
        let loss = (&tiled * &w).unwrap().sum_all().unwrap();
        let grads = loss.backward().expect(
            "this dividing case must NOT error (element counts match: 8 both sides) — the \
             point of this pin is that candle's naive reshape backward succeeds here, silently, \
             with the wrong answer, unlike the non-dividing case above which errors loudly",
        );
        let got: Vec<f32> = grads
            .get(xv.as_tensor())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let w_at = |window: usize, col: usize, in_window: usize| {
            w_vals[(window * 2 + col) * 2 + in_window]
        };
        let mut correct = vec![0f32; 8];
        for row in 0..4 {
            for col in 0..2 {
                correct[row * 2 + col] = w_at(row / 2, col, row % 2);
            }
        }

        assert_ne!(
            got, correct,
            "candle's unfold backward on a non-last dim is expected to be WRONG (see this \
             test's doc) — if it now matches the hand-derived correct gradient, candle has \
             fixed this and the premise this test pins no longer holds"
        );
        assert!(
            got.iter().all(|v| v.is_finite()),
            "candle's wrong gradient here must still be FINITE (not NaN/Inf) — the whole point \
             is that it is a SILENT wrong answer, not a loud failure: got {got:?}"
        );
    }

    /// D10: the tower's `BatchNorm` stays pinned to EVAL statistics even
    /// under `set_training(true)` (`forward_front`'s `forward_t(&x, false)`).
    /// HF's own HTSAT pins it the same way, and a LoRA fine-tune leaves the
    /// backbone frozen — recomputing batch statistics from a fine-tuning
    /// batch would silently change the frozen base's own numerics.
    ///
    /// Asserted as a MECHANISM, both directions: the post-batch-norm
    /// boundary is bit-identical with the flag on and off, AND the
    /// train-statistics arm of the very same `BatchNorm` on the very same
    /// input produces a DIFFERENT tensor (the non-vacuity control — without
    /// it, "the two agree" would also hold if the two arms happened to be
    /// the same computation on this fixture).
    #[test]
    fn batch_norm_stays_on_eval_statistics_under_set_training() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let (mut tower, _cfg, input) = build_full_tower_and_input(&varmap, &device, 31);

        let eval_boundary = tower
            .encoder
            .forward_front(&input, &[true])
            .unwrap()
            .post_batch_norm;
        tower.set_training(true);
        let training_boundary = tower
            .encoder
            .forward_front(&input, &[true])
            .unwrap()
            .post_batch_norm;

        let as_bits = |t: &Tensor| -> Vec<u32> {
            t.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect()
        };
        assert_eq!(
            as_bits(&training_boundary),
            as_bits(&eval_boundary),
            "the BatchNorm must stay on eval statistics under set_training(true)"
        );

        // Non-vacuity: the train-statistics arm really is a different
        // computation on this fixture.
        let x = input.transpose(1, 3).unwrap().contiguous().unwrap();
        let train_arm = tower.encoder.batch_norm.forward_t(&x, true).unwrap();
        assert_ne!(
            as_bits(&train_arm),
            as_bits(&eval_boundary),
            "control: BatchNorm's train arm must differ from its eval arm here, or this test \
             asserts nothing"
        );
    }

    /// D6 for this tower's shift-window mask: it follows the backbone dtype
    /// and keeps HF's own `-100.0` magnitude, which is exactly representable
    /// in F32, F16 and BF16 alike — so the cast is lossless and the F32
    /// tensor's bits are unchanged. Unlike `crate::clip_text`'s causal
    /// sentinel this value must NOT become a dtype minimum: `exp(-100)` is
    /// an f32 denormal that the reference implementation's numerics depend
    /// on, so changing the magnitude would be an eval-bit change.
    #[test]
    fn shift_window_mask_follows_the_dtype_and_keeps_the_minus_100_magnitude() {
        let device = Device::Cpu;
        let (h, w, ws, shift) = (8usize, 8usize, 4usize, 2usize);

        let f32_mask = SwinBlock::build_attn_mask(h, w, ws, shift, &device, DType::F32).unwrap();
        assert_eq!(f32_mask.dtype(), DType::F32);
        let f32_values: Vec<f32> = f32_mask.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            f32_values.contains(&-100.0f32),
            "the fixture must actually reach a masked entry"
        );
        assert!(f32_values.iter().all(|v| *v == 0.0 || *v == -100.0));

        for (dtype, name) in [(DType::F16, "f16"), (DType::BF16, "bf16")] {
            let mask = SwinBlock::build_attn_mask(h, w, ws, shift, &device, dtype).unwrap();
            assert_eq!(mask.dtype(), dtype, "{name}: mask must carry the dtype");
            // Widening back to f32 is exact (every f16/bf16 value is
            // representable in f32), so an exact comparison here IS a
            // comparison of the stored bit patterns.
            let widened: Vec<f32> = mask
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(
                widened, f32_values,
                "{name}: -100.0 and 0.0 are exact in this dtype, so the cast must be lossless"
            );
        }
    }

    /// Eval output is unaffected by ever having toggled training on and back
    /// off: the two softmax arms are only wired into the *backward* path, so
    /// the eval-mode forward is byte-identical before any `set_training`
    /// call and after `set_training(true); set_training(false)`. Uses a
    /// VarMap-backed tower sized by the real `htsat_clap_tiny` config (not
    /// the minimal `tiny_stage_config` above) so this exercises the full
    /// front-half + all four Swin stages + patch-merging + pooling +
    /// projection, not just one stage. The shift-window masks follow the
    /// backbone dtype (`build_attn_mask`); this VarMap-backed tower is F32,
    /// so this oracle stays f32-only.
    #[test]
    fn eval_output_is_bit_identical_across_a_training_toggle_round_trip() {
        let device = Device::Cpu;
        let cfg = HtsatAudioConfig::from_hf_clap_config(&fixture_config()).unwrap();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut tower = HtsatAudio::load(vb, &cfg, &device).unwrap();
        deterministic_fill_varmap(&varmap, &device);

        let t = 40;
        let input = deterministic_tensor(4 * t, cfg.num_mel_bins, 99, &device)
            .reshape((1, 4, t, cfg.num_mel_bins))
            .unwrap();
        let is_longer = [true];

        let before = tower.forward(&input, &is_longer).unwrap();
        tower.set_training(true);
        tower.set_training(false);
        let after = tower.forward(&input, &is_longer).unwrap();

        // Compared as RAW BITS, not as `f32` values: `==` on `f32` calls
        // `-0.0` equal to `+0.0` and every `NaN` unequal to itself, so a
        // value-level comparison neither proves "byte for byte" nor fails
        // loudly on a degenerate output. Non-vacuity is asserted first —
        // an empty (or all-zero) descriptor would make any equality below
        // trivially true.
        let before_bits: Vec<u32> = before
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let after_bits: Vec<u32> = after
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(before_bits.len(), cfg.projection_dim);
        let norm = before
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt();
        assert_finite_nonzero(norm, "eval-mode audio embedding");
        assert_eq!(
            before_bits, after_bits,
            "eval output must be bit-identical across a training toggle round trip -- \
             every arm `set_training` flips (the attention softmax arms, both LayerNorms, \
             and the MLP/projection GELU seam arms) must leave the EVAL forward's bytes \
             exactly as they were"
        );
    }

    /// The GELU seam's exact per-forward dispatch count (family F: an
    /// independently-known number, computed live and asserted, not an
    /// "advanced by some amount" smoke check). On the real `htsat_clap_tiny`
    /// geometry `depths = [2, 2, 2, 2]`, so the spine has `2+2+2+2 = 8` Swin
    /// blocks and `SwinBlock::forward` evaluates GELU exactly ONCE each
    /// (`crate::activations::gelu_erf` on the MLP's intermediate output);
    /// `projection_hidden_act` is `"relu"` in this fixture, so the
    /// projection head contributes ZERO — total 8 per full-tower forward.
    ///
    /// This is the oracle that makes a broken flag thread fail loudly: if
    /// `HtsatAudio::forward` ever stops passing its own flag down through
    /// `forward_spine_with_training` (or a block stops forwarding it), those
    /// blocks silently take the seam's EVAL arm — which makes no admission
    /// decision at all — and the count comes in below 8 rather than
    /// producing any visibly wrong number.
    ///
    /// # Why K4 (eval bytes) is untouched, and where CUDA is proved
    ///
    /// On CPU F32 every one of those 8 tensors is contiguous and non-empty,
    /// so `gelu_admission_predicate` HOLDS and all 8 go down the FUSED arm —
    /// and `GeluErfFused`'s CPU F32 forward is BIT-IDENTICAL to
    /// `Tensor::gelu_erf` by that op's own documented contract (see
    /// `crate::activations::dispatch_gelu_erf_fused`, and
    /// `tests::projection_gelu_arm_dispatches_the_seam_exactly_once` for
    /// this crate's own per-element bit assertion of it). So CPU bits are
    /// unchanged in BOTH modes: eval because the seam short-circuits to the
    /// unchanged call, training because the fused CPU arm reproduces it
    /// exactly. Eval bytes crate-wide are pinned by `tests/bits_snapshot.rs`
    /// and `tests/golden_parity.rs`, neither of which sets training at all.
    /// The fused-vs-eager question that CANNOT be settled bit-exactly — CUDA
    /// F32/BF16/F16 — is not this test's to settle: it is
    /// `jammi-kernels`'s own `tests/cuda_parity.rs::gelu_erf_parity_*` suite
    /// (`gelu_erf_parity_contiguous_small`,
    /// `gelu_erf_parity_production_width`,
    /// `gelu_erf_parity_non_multiple_of_launch_block`,
    /// `gelu_erf_parity_narrowed_with_nonzero_offset`,
    /// `gelu_erf_parity_empty_input_is_refused_on_both_devices`), which
    /// bounds the same kernel this seam dispatches.
    ///
    /// The eager counter must stay exactly where it was. Both halves are
    /// asserted — a one-sided "fused went up" check would pass even if half
    /// the sites had fallen back.
    ///
    /// # Why two locks, in THIS order
    ///
    /// This test needs both crate-shared counter locks: the full forward
    /// bumps `crate::layer_norm::LN_DISPATCH_COUNTERS` and the softmax
    /// cascade counters as well as the `gelu_erf_fused` counters it reads
    /// exactly, and every OTHER test that can bump `gelu_erf_fused` holds
    /// one of the two (this module's own training-mode forwards hold the
    /// `layer_norm` one; `crate::activations`/`crate::bert`/
    /// `crate::distilbert`'s GELU counter tests hold the
    /// `attention_cascade` one).
    ///
    /// The ORDER is not free: `crate::modernbert`'s own tests already
    /// acquire both, `attention_cascade` FIRST and `layer_norm` second
    /// (e.g. `modernbert::tests::
    /// forward_hidden_with_lengths_none_is_bit_identical_to_forward_hidden`,
    /// which takes `ATTENTION_BLOCK_COUNTER_TEST_LOCK` and then, a few lines
    /// later, `crate::layer_norm::DISPATCH_COUNTER_TEST_LOCK`). Acquiring
    /// them the other way round here is a lock-order INVERSION against those
    /// tests, and the default parallel test runner deadlocks the whole
    /// `cargo test --lib` binary on it — observed, not hypothesized, while
    /// writing this test (every htsat/layer_norm/modernbert thread parked in
    /// `__psynch_mutexwait` forever). Every multi-lock test in this crate
    /// therefore takes `attention_cascade` before `layer_norm`, and any new
    /// one must too.
    #[test]
    fn training_true_full_forward_dispatches_the_gelu_seam_once_per_swin_block() {
        // Lock order: attention_cascade THEN layer_norm -- see this test's
        // own doc.
        let _attn_guard = crate::attention_cascade::ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _ln_guard = crate::layer_norm::DISPATCH_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let (mut tower, cfg, input) = build_full_tower_and_input(&varmap, &device, 101);

        // The expected count, re-derived from the config this tower was
        // actually built from (not a bare literal), then pinned against the
        // independently-known fixture value.
        assert_eq!(cfg.depths, vec![2, 2, 2, 2], "fixture geometry changed");
        assert_eq!(
            cfg.projection_hidden_act, "relu",
            "fixture projection activation changed -- the projection head would then \
             contribute one more GELU call per forward"
        );
        let expected: usize = cfg.depths.iter().sum();
        assert_eq!(expected, 8, "sum(depths) for htsat_clap_tiny");

        tower.set_training(true);
        let before = crate::activations::GELU_DISPATCH_COUNTERS.snapshot();
        let out = tower.forward(&input, &[true]).unwrap();
        let after = crate::activations::GELU_DISPATCH_COUNTERS.snapshot();

        // Non-vacuity: the forward really produced a descriptor.
        assert_eq!(out.dims(), &[1, cfg.projection_dim]);

        assert_eq!(
            after.fused - before.fused,
            expected as u64,
            "the GELU seam must dispatch FUSED exactly once per Swin block \
             (sum(depths)={expected}) at training=true on CPU F32"
        );
        assert_eq!(
            after.eager, before.eager,
            "every one of those sites is contiguous, non-empty CPU F32 -- none may fall \
             back to the eager arm"
        );
    }

    /// The eval half of the same mechanism, and the reason it is a SEPARATE
    /// assertion: at `training == false` — which is what `HtsatAudio::forward`
    /// passes down when nothing has called `set_training` — the seam
    /// short-circuits to the unchanged `Tensor::gelu_erf` call before any
    /// admission decision is taken, so NEITHER counter may move. Without this, the count oracle
    /// above could be satisfied by a seam that admitted on every forward
    /// regardless of mode — which would be a silent behavior change to
    /// eval bytes on any device/dtype where fused and eager differ.
    #[test]
    fn eval_forward_takes_no_gelu_admission_decision_at_all() {
        // Lock order: attention_cascade THEN layer_norm -- see this test's
        // own doc.
        let _attn_guard = crate::attention_cascade::ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _ln_guard = crate::layer_norm::DISPATCH_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let (tower, cfg, input) = build_full_tower_and_input(&varmap, &device, 101);
        // `training` defaults to false; no `set_training` call at all.
        assert!(!tower.is_training());

        let before = crate::activations::GELU_DISPATCH_COUNTERS.snapshot();
        let out = tower.forward(&input, &[true]).unwrap();
        let after = crate::activations::GELU_DISPATCH_COUNTERS.snapshot();

        assert_eq!(out.dims(), &[1, cfg.projection_dim]);
        assert_eq!(
            (after.fused, after.eager),
            (before.fused, before.eager),
            "eval must take no admission decision on ANY of the {} Swin-block GELU sites",
            cfg.depths.iter().sum::<usize>()
        );
    }

    /// The COMPOSITE count both site classes sum to, measured rather than
    /// inferred (family F): the same `htsat_clap_tiny` geometry with
    /// `projection_hidden_act` flipped to `"gelu"` must dispatch the seam
    /// `sum(depths) + 1 = 9` times per full-tower forward — 8 Swin-block
    /// MLPs plus the projection head, all reporting to the SAME process-wide
    /// `gelu_erf_fused` counter. The sibling tests pin the two terms
    /// separately; this one pins that they ADD, on one real tower, so the
    /// arithmetic in this module's own doc is a measured claim and not a
    /// hypothesis.
    #[test]
    fn training_true_full_forward_with_a_gelu_projection_dispatches_sum_depths_plus_one() {
        // Lock order: attention_cascade THEN layer_norm -- see this test's
        // own doc.
        let _attn_guard = crate::attention_cascade::ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _ln_guard = crate::layer_norm::DISPATCH_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let mut cfg = HtsatAudioConfig::from_hf_clap_config(&fixture_config()).unwrap();
        cfg.projection_hidden_act = "gelu".to_string();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut tower = HtsatAudio::load(vb, &cfg, &device).unwrap();
        deterministic_fill_varmap(&varmap, &device);
        let t = 40;
        let input = deterministic_tensor(4 * t, cfg.num_mel_bins, 101, &device)
            .reshape((1, 4, t, cfg.num_mel_bins))
            .unwrap();

        let expected = cfg.depths.iter().sum::<usize>() + 1;
        assert_eq!(
            expected, 9,
            "sum(depths) + 1 projection for htsat_clap_tiny"
        );

        tower.set_training(true);
        let before = crate::activations::GELU_DISPATCH_COUNTERS.snapshot();
        let out = tower.forward(&input, &[true]).unwrap();
        let after = crate::activations::GELU_DISPATCH_COUNTERS.snapshot();

        assert_eq!(out.dims(), &[1, cfg.projection_dim]);
        assert_eq!(
            after.fused - before.fused,
            expected as u64,
            "both site classes report to one counter: {} Swin-block MLP dispatches plus \
             the projection head's one",
            cfg.depths.iter().sum::<usize>()
        );
        assert_eq!(after.eager, before.eager, "no eager fallback on CPU F32");
    }

    /// Build a standalone [`ClapAudioProjection`] whose activation is `act`,
    /// over the minimal `tiny_stage_config` geometry, plus a deterministic
    /// `[1, hidden_size]` pooled-descriptor input.
    fn build_projection(
        varmap: &VarMap,
        device: &Device,
        act: &str,
    ) -> (ClapAudioProjection, Tensor) {
        let mut cfg = tiny_stage_config();
        cfg.projection_hidden_act = act.to_string();
        let vb = VarBuilder::from_varmap(varmap, DType::F32, device);
        let proj = ClapAudioProjection::load(vb, &cfg).unwrap();
        deterministic_fill_varmap(varmap, device);
        let x = deterministic_tensor(1, cfg.hidden_size, 7, device);
        (proj, x)
    }

    /// The projection head's own GELU site (the `"gelu"` arm of
    /// `projection_hidden_act`, which the `htsat_clap_tiny` fixture does not
    /// select — hence this separate, explicitly-`"gelu"` fixture): exactly
    /// ONE seam dispatch per `forward_unnormalized_with_training(.., true)`,
    /// and ZERO counter movement through the flag-less
    /// `forward_unnormalized` eval convenience. Together with the Swin-block
    /// count above, this is the `+1` term in the tower's per-forward
    /// dispatch total (`sum(depths) + 1` when `projection_hidden_act` is
    /// `"gelu"` — both sites report to the SAME `gelu_erf_fused` counter). The eval output is asserted
    /// bit-identical to the training-mode one as well: on CPU F32 the fused
    /// arm IS `Tensor::gelu_erf` bit for bit
    /// (`crate::activations::dispatch_gelu_erf_fused`'s own contract), so
    /// this is a genuine numeric equality, not merely a counter equality.
    #[test]
    fn projection_gelu_arm_dispatches_the_seam_exactly_once() {
        // Lock order: attention_cascade THEN layer_norm -- see this test's
        // own doc.
        let _attn_guard = crate::attention_cascade::ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _ln_guard = crate::layer_norm::DISPATCH_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let (proj, x) = build_projection(&varmap, &device, "gelu");

        // Eval arm, through the flag-less public convenience: no admission
        // decision at all.
        let before_eval = crate::activations::GELU_DISPATCH_COUNTERS.snapshot();
        let eval_out = proj.forward_unnormalized(&x).unwrap();
        let after_eval = crate::activations::GELU_DISPATCH_COUNTERS.snapshot();
        assert_eq!(
            (after_eval.fused, after_eval.eager),
            (before_eval.fused, before_eval.eager),
            "the projection's eval arm must take no admission decision"
        );

        // Training arm: exactly one fused dispatch, no eager fallback.
        let before = crate::activations::GELU_DISPATCH_COUNTERS.snapshot();
        let train_out = proj.forward_unnormalized_with_training(&x, true).unwrap();
        let after = crate::activations::GELU_DISPATCH_COUNTERS.snapshot();
        assert_eq!(
            after.fused - before.fused,
            1,
            "the projection head evaluates GELU exactly once per forward"
        );
        assert_eq!(after.eager, before.eager, "no eager fallback on CPU F32");

        let eval_v: Vec<f32> = eval_out.flatten_all().unwrap().to_vec1().unwrap();
        let train_v: Vec<f32> = train_out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(eval_v.len(), 8, "tiny_stage_config projection_dim");
        let norm = eval_v.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert_finite_nonzero(norm, "projection output");
        for (i, (&e, &t)) in eval_v.iter().zip(train_v.iter()).enumerate() {
            assert_eq!(
                e.to_bits(),
                t.to_bits(),
                "elem[{i}]: eval {e} vs training {t} -- CPU F32 fused GELU is bit-identical \
                 to the eager call it replaces"
            );
        }
    }

    /// Negative control for the site above: the `"relu"` arm is a DIFFERENT
    /// activation with no fused seam, so a training-mode projection forward
    /// on it must move NEITHER GELU counter. Without this, a seam wired at
    /// the wrong place (outside the `match`, say) would still satisfy the
    /// `"gelu"`-arm count oracle while silently applying GELU admission
    /// machinery to a ReLU network.
    #[test]
    fn projection_relu_arm_never_touches_the_gelu_seam() {
        // Lock order: attention_cascade THEN layer_norm -- see this test's
        // own doc.
        let _attn_guard = crate::attention_cascade::ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _ln_guard = crate::layer_norm::DISPATCH_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let (proj, x) = build_projection(&varmap, &device, "relu");

        let before = crate::activations::GELU_DISPATCH_COUNTERS.snapshot();
        let out = proj.forward_unnormalized_with_training(&x, true).unwrap();
        let after = crate::activations::GELU_DISPATCH_COUNTERS.snapshot();

        assert_eq!(out.dims(), &[1, 8]);
        assert_eq!(
            (after.fused, after.eager),
            (before.fused, before.eager),
            "the relu arm must not reach the GELU seam"
        );
    }

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

    /// (a) Positive control: the `htsat_clap_tiny` fixture's own
    /// `hidden_act` ("gelu" — the only HF `ClapAudioConfig` value the Swin
    /// MLP's unconditional gelu-erf actually computes) loads cleanly
    /// through [`HtsatAudio::load`]. Without this, the negative control
    /// below could be satisfied by a check that rejects EVERY config, not
    /// just an unsupported one.
    #[test]
    fn hidden_act_gelu_fixture_loads() {
        let cfg = HtsatAudioConfig::from_hf_clap_config(&fixture_config()).unwrap();
        assert_eq!(cfg.hidden_act, "gelu", "fixture activation changed");
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        HtsatAudio::load(vb, &cfg, &device)
            .expect("hidden_act='gelu' (the fixture's own value) must load");
    }

    /// (b) The domain-validity edge this unit closes: `SwinBlock::forward`
    /// is unconditionally gelu-erf (this module's own doc), so a config
    /// declaring any OTHER `hidden_act` must be REFUSED at load — named,
    /// not silently computed as gelu-erf regardless — through BOTH
    /// `HtsatAudio::load` (a `VarMap`-backed, no-checkpoint build, the
    /// shape every other `HtsatAudio` unit test in this module uses) and
    /// `HtsatAudio::builder().build(..)` (the mmaped-safetensors-backed
    /// production entry point). The check in
    /// `HtsatAudioEncoder::load_with` runs before ANY tensor is read, so an
    /// EMPTY safetensors file is enough to prove the builder path refuses
    /// too — a full checkpoint's worth of weights would prove nothing more
    /// about this particular edge.
    #[test]
    fn hidden_act_other_than_gelu_is_refused_at_load_through_both_entry_points() {
        let device = Device::Cpu;
        let mut cfg = tiny_stage_config();
        assert_eq!(cfg.hidden_act, "gelu", "tiny_stage_config's own default");
        cfg.hidden_act = "relu".to_string();

        // Through `HtsatAudio::load` (`VarBuilder::from_varmap`).
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let err = HtsatAudio::load(vb, &cfg, &device)
            .err()
            .expect("hidden_act='relu' must be refused through HtsatAudio::load");
        match err {
            EncoderError::Config(msg) => assert!(
                msg.contains("hidden_act") && msg.contains("relu"),
                "error must name the field and the offending value, got: {msg}"
            ),
            other => panic!("expected EncoderError::Config, got {other:?}"),
        }

        // Through `HtsatAudio::builder().build(..)` (mmaped safetensors) —
        // an EMPTY checkpoint file: the refusal must fire before any tensor
        // is even looked up.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("hidden_act_relu_probe.safetensors");
        let empty: HashMap<String, Tensor> = HashMap::new();
        candle_core::safetensors::save(&empty, &path).unwrap();
        let build_varmap = VarMap::new();
        let err = HtsatAudio::builder()
            .build(&[&path], &cfg, &device, &build_varmap)
            .err()
            .expect("hidden_act='relu' must be refused through HtsatAudio::builder().build(..)");
        match err {
            EncoderError::Config(msg) => assert!(
                msg.contains("hidden_act") && msg.contains("relu"),
                "error must name the field and the offending value, got: {msg}"
            ),
            other => panic!("expected EncoderError::Config, got {other:?}"),
        }
    }

    /// #421 P1-a3, the HTSAT leg — the same oracle as
    /// `crate::bert::tests::
    /// fusible_site_census_is_the_exact_per_forward_seam_call_count`, run
    /// on the committed `htsat_clap_tiny` checkpoint through the REAL
    /// builder, at BOTH projection activations.
    ///
    /// This tower is where the census earns the most, because three of its
    /// counts are genuinely irregular:
    ///
    /// * the LAST Swin stage has no `downsample`, so it contributes neither
    ///   a `reduction` LoRA site nor a merging LayerNorm;
    /// * the projection head's two linears belong to no stage at all; and
    /// * the GELU count depends on a STRING in the built head
    ///   (`projection_hidden_act`), which is why this test runs the `"relu"`
    ///   fixture AND a `"gelu"` twin over the same geometry and asserts the
    ///   two differ by exactly one. The sibling tests
    ///   `training_true_full_forward_dispatches_the_gelu_seam_once_per_swin_block`
    ///   and
    ///   `training_true_full_forward_with_a_gelu_projection_dispatches_sum_depths_plus_one`
    ///   already MEASURE those two counter deltas (8 and 9 on this fixture);
    ///   this test is what ties them to the census, so the profile's `calls`
    ///   term and the measured dispatch count are the same number by
    ///   assertion rather than by coincidence.
    #[test]
    fn fusible_site_census_is_the_exact_per_forward_seam_call_count() {
        // Lock order: attention_cascade THEN layer_norm — see this module's
        // own multi-lock test doc.
        let _attn_guard = crate::attention_cascade::ATTENTION_BLOCK_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _ln_guard = crate::layer_norm::DISPATCH_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let device = Device::Cpu;
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../cookbook/fixtures/htsat_clap_tiny");
        let weights = dir.join("model.safetensors");
        let config = HtsatAudioConfig::from_hf_clap_config(&fixture_config()).unwrap();
        assert_eq!(config.projection_hidden_act, "relu", "fixture act changed");
        let lora = crate::test_support::AllLinearGaussianLora::new();

        let varmap = VarMap::new();
        let adapted = HtsatAudio::builder()
            .lora(lora.config())
            .build(&[weights.as_path()], &config, &device, &varmap)
            .expect("build htsat with an all-linear Gaussian adapter");
        let mut any = crate::AnyEncoder::Htsat(Box::new(adapted));
        let census = crate::test_support::assert_fusible_site_census_is_exact(
            &mut any,
            &device,
            "htsat/all-linear/relu",
        );

        // Re-derived from the geometry this tower was built from: six sites
        // per Swin block, one `reduction` per stage EXCEPT the last, two
        // projection linears.
        let blocks: usize = config.depths.iter().sum();
        let stages = config.num_stages();
        assert_eq!(census.lora_sites_wrapped, 6 * blocks + (stages - 1) + 2);
        assert_eq!(
            census.layer_norms,
            1 + 2 * blocks + (stages - 1) + 1,
            "patch-embed norm + two per Swin block + one per downsample + the final norm \
             (the AFF block's BatchNorms are NOT house LayerNorms and must not be counted)"
        );
        assert_eq!(
            census.gelu_seam_calls_per_forward, blocks,
            "one seam call per Swin block, and none from a \"relu\" projection head"
        );

        // The `"gelu"` projection twin: identical weights and geometry, one
        // string changed, exactly one more seam call per forward.
        let mut gelu_config = config.clone();
        gelu_config.projection_hidden_act = GELU_PROJECTION_ACT.to_string();
        let gelu_varmap = VarMap::new();
        let gelu_tower = HtsatAudio::builder()
            .lora(lora.config())
            .build(&[weights.as_path()], &gelu_config, &device, &gelu_varmap)
            .expect("build htsat with a gelu projection head");
        let mut any_gelu = crate::AnyEncoder::Htsat(Box::new(gelu_tower));
        let gelu_census = crate::test_support::assert_fusible_site_census_is_exact(
            &mut any_gelu,
            &device,
            "htsat/all-linear/gelu",
        );
        assert_eq!(
            gelu_census.gelu_seam_calls_per_forward,
            census.gelu_seam_calls_per_forward + 1,
            "a \"gelu\" projection head adds exactly ONE seam call per forward"
        );
        assert_eq!(gelu_census.layer_norms, census.layer_norms);
        assert_eq!(gelu_census.lora_sites_wrapped, census.lora_sites_wrapped);

        // Frozen twin: no wrapped site, everything else unchanged.
        let frozen_varmap = VarMap::new();
        let frozen = HtsatAudio::builder()
            .build(&[weights.as_path()], &config, &device, &frozen_varmap)
            .expect("build a frozen htsat");
        let mut any_frozen = crate::AnyEncoder::Htsat(Box::new(frozen));
        let frozen_census = crate::test_support::assert_fusible_site_census_is_exact(
            &mut any_frozen,
            &device,
            "htsat/frozen",
        );
        assert_eq!(
            frozen_census.lora_sites_wrapped, 0,
            "a frozen backbone wraps no site and takes no lora_linear_fused decision"
        );
        assert_eq!(frozen_census.layer_norms, census.layer_norms);
        assert_eq!(
            frozen_census.gelu_seam_calls_per_forward,
            census.gelu_seam_calls_per_forward
        );
    }
}
