//! CLAP-family audio transformer.
//!
//! The audio tower of a CLAP (Contrastive Language-Audio Pretraining) model:
//! it consumes a log-mel spectrogram `[batch, n_mels, n_frames]`, embeds it
//! with a strided patch convolution (a 1-channel ViT over the time-frequency
//! plane), runs a stack of residual attention blocks, mean-pools over the
//! patch tokens, projects into the shared CLAP latent space, and
//! L2-normalizes.
//!
//! Forward output is `[batch, embed_dim]` in the same latent space as the
//! CLAP text tower, enabling cross-modal text↔audio cosine similarity — the
//! audio analogue of [`crate::clip_text::ClipText`].
//!
//! The transformer block layout (fused `in_proj_weight`/`in_proj_bias`,
//! `out_proj`, QuickGelu MLP) matches the OpenCLIP/CLAP safetensors key
//! convention so checkpoints load without key remapping.

use candle_core::{IndexOp, Module, Tensor, D};
use candle_nn::{
    conv2d_no_bias, layer_norm, linear, Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder,
};

use crate::error::EncoderError;

/// Architecture configuration for the CLAP audio transformer.
///
/// The audio tower treats the `[n_mels, n_frames]` log-mel spectrogram as a
/// single-channel image and patch-embeds it with a `patch_size`-strided
/// convolution. `embed_dim` is the shared CLAP latent dimensionality (must
/// match the text tower's projected output); `width` is the per-token hidden
/// size inside the transformer, projected to `embed_dim` by `audio_projection`.
#[derive(Debug, Clone)]
pub struct ClapAudioConfig {
    /// Number of mel-frequency bins in the input spectrogram (the height of
    /// the time-frequency plane). Must be divisible by `patch_size`.
    pub n_mels: usize,
    /// Number of time frames the spectrogram is padded/truncated to (the
    /// width of the time-frequency plane). Must be divisible by `patch_size`.
    pub n_frames: usize,
    /// Square patch size for the patch-embedding convolution.
    pub patch_size: usize,
    /// Per-token hidden size inside the transformer.
    pub width: usize,
    /// Number of transformer layers.
    pub layers: usize,
    /// Number of attention heads. Must divide `width` evenly.
    pub heads: usize,
    /// MLP intermediate-size ratio (`intermediate = width * mlp_ratio`).
    pub mlp_ratio: f64,
    /// Shared CLAP latent dimensionality after `audio_projection`.
    pub embed_dim: usize,
    /// Target sample rate (Hz) the decoder resamples raw audio to before the
    /// log-mel transform. Feature extraction is config-driven, not hardcoded.
    pub sample_rate: u32,
    /// FFT window size (samples) for the short-time Fourier transform.
    pub n_fft: usize,
    /// Hop length (samples) between successive STFT frames.
    pub hop_length: usize,
}

impl ClapAudioConfig {
    /// Parse from a CLAP config JSON (`open_clip_config.json`-shape with an
    /// `audio_cfg` block).
    ///
    /// Reads `model_cfg.embed_dim` and `model_cfg.audio_cfg.{n_mels, n_frames,
    /// patch_size, width, layers, heads, mlp_ratio, sample_rate, n_fft,
    /// hop_length}`, applying CLAP-family defaults (`patch_size=16`,
    /// `mlp_ratio=4.0`, `heads=width/64`, `sample_rate=48000`, `n_fft=1024`,
    /// `hop_length=480`) when fields are omitted.
    pub fn from_clap_config(config: &serde_json::Value) -> Result<Self, EncoderError> {
        let model_cfg = config
            .get("model_cfg")
            .ok_or_else(|| EncoderError::Config("CLAP config missing 'model_cfg'".into()))?;
        let audio_cfg = model_cfg.get("audio_cfg").ok_or_else(|| {
            EncoderError::Config("CLAP config missing 'model_cfg.audio_cfg'".into())
        })?;
        let embed_dim = model_cfg
            .get("embed_dim")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                EncoderError::Config("CLAP config missing 'model_cfg.embed_dim'".into())
            })? as usize;

        let width = audio_cfg
            .get("width")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                EncoderError::Config("CLAP config missing 'model_cfg.audio_cfg.width'".into())
            })? as usize;
        let layers = audio_cfg
            .get("layers")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                EncoderError::Config("CLAP config missing 'model_cfg.audio_cfg.layers'".into())
            })? as usize;
        let n_mels = audio_cfg
            .get("n_mels")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                EncoderError::Config("CLAP config missing 'model_cfg.audio_cfg.n_mels'".into())
            })? as usize;
        let n_frames = audio_cfg
            .get("n_frames")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                EncoderError::Config("CLAP config missing 'model_cfg.audio_cfg.n_frames'".into())
            })? as usize;

        let patch_size = audio_cfg
            .get("patch_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(16) as usize;
        let heads = audio_cfg
            .get("heads")
            .and_then(|v| v.as_u64())
            .unwrap_or((width / 64).max(1) as u64) as usize;
        let mlp_ratio = audio_cfg
            .get("mlp_ratio")
            .and_then(|v| v.as_f64())
            .unwrap_or(4.0);
        let sample_rate = audio_cfg
            .get("sample_rate")
            .and_then(|v| v.as_u64())
            .unwrap_or(48_000) as u32;
        let n_fft = audio_cfg
            .get("n_fft")
            .and_then(|v| v.as_u64())
            .unwrap_or(1024) as usize;
        let hop_length = audio_cfg
            .get("hop_length")
            .and_then(|v| v.as_u64())
            .unwrap_or(480) as usize;

        if n_mels % patch_size != 0 || n_frames % patch_size != 0 {
            return Err(EncoderError::Config(format!(
                "CLAP audio_cfg: n_mels ({n_mels}) and n_frames ({n_frames}) must each be \
                 divisible by patch_size ({patch_size})"
            )));
        }

        Ok(Self {
            n_mels,
            n_frames,
            patch_size,
            width,
            layers,
            heads,
            mlp_ratio,
            embed_dim,
            sample_rate,
            n_fft,
            hop_length,
        })
    }
}

/// QuickGelu activation: `x * sigmoid(1.702 * x)`.
fn quick_gelu(xs: &Tensor) -> Result<Tensor, EncoderError> {
    Ok((xs * candle_nn::ops::sigmoid(&(xs * 1.702f64)?)?)?)
}

/// Multi-head self-attention with fused QKV projection (OpenCLIP/CLAP layout).
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

    /// Full (non-causal) self-attention over the patch tokens.
    fn forward(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        let (batch, seq_len, _) = x.dims3()?;
        let qkv = self.in_proj.forward(x)?;
        let qkv = qkv.reshape((batch, seq_len, 3, self.num_heads, self.head_dim))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?; // (3, batch, heads, seq, head_dim)

        let q = qkv.i(0)?.contiguous()?;
        let k = qkv.i(1)?.contiguous()?;
        let v = qkv.i(2)?.contiguous()?;

        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? / scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
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
    fn load(
        vb: VarBuilder,
        width: usize,
        heads: usize,
        mlp_ratio: f64,
    ) -> Result<Self, EncoderError> {
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

    fn forward(&self, x: &Tensor) -> Result<Tensor, EncoderError> {
        let residual = x;
        let x = self.ln_1.forward(x)?;
        let x = self.attn.forward(&x)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.ln_2.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        Ok((residual + x)?)
    }
}

/// CLAP audio transformer.
///
/// Weight keys follow the OpenCLIP/CLAP safetensors layout under the `audio.*`
/// prefix: `audio.conv1.weight`, `audio.positional_embedding`,
/// `audio.ln_pre.*`, `audio.transformer.resblocks.{n}.*`, `audio.ln_post.*`,
/// and `audio.audio_projection`. Callers build this from `vb.pp("audio")`,
/// the audio analogue of the vision tower's `vb.pp("visual")`.
pub struct ClapAudio {
    conv1: Conv2d,
    /// Learned `[num_patches, width]` positional embedding (added per patch
    /// token; no class token — the tower mean-pools over patches).
    positional_embedding: Tensor,
    ln_pre: LayerNorm,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: LayerNorm,
    /// `[width, embed_dim]` projection into the shared CLAP latent space.
    audio_projection: Tensor,
    config: ClapAudioConfig,
}

impl ClapAudio {
    /// Build the audio transformer from an `audio`-scoped [`VarBuilder`].
    pub fn load(vb: VarBuilder, config: &ClapAudioConfig) -> Result<Self, EncoderError> {
        let conv_config = Conv2dConfig {
            stride: config.patch_size,
            ..Default::default()
        };
        // Single-channel input (the mel spectrogram) → `width` feature maps.
        let conv1 = conv2d_no_bias(
            1,
            config.width,
            config.patch_size,
            conv_config,
            vb.pp("conv1"),
        )?;

        let grid_h = config.n_mels / config.patch_size;
        let grid_w = config.n_frames / config.patch_size;
        let num_patches = grid_h * grid_w;
        let positional_embedding = vb.get((num_patches, config.width), "positional_embedding")?;

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
        let audio_projection = vb.get((config.width, config.embed_dim), "audio_projection")?;

        Ok(Self {
            conv1,
            positional_embedding,
            ln_pre,
            blocks,
            ln_post,
            audio_projection,
            config: config.clone(),
        })
    }

    /// Forward pass: log-mel spectrogram → L2-normalized shared-latent embeddings.
    ///
    /// Input: `[batch, n_mels, n_frames]` (the decoder/feature extractor pads
    /// or truncates every clip to the fixed `n_frames` window). Output:
    /// `[batch, embed_dim]`, L2-normalized along the embedding axis — the same
    /// latent space as the CLAP text tower.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor, EncoderError> {
        let (batch, n_mels, n_frames) = mel.dims3()?;
        if n_mels != self.config.n_mels || n_frames != self.config.n_frames {
            return Err(EncoderError::Config(format!(
                "CLAP audio forward expected mel [batch, {}, {}], got [{batch}, {n_mels}, {n_frames}]",
                self.config.n_mels, self.config.n_frames
            )));
        }

        // Add the single channel axis: [batch, 1, n_mels, n_frames].
        let x = mel.unsqueeze(1)?;

        // Patch embedding: [batch, 1, H, W] -> [batch, width, gh, gw].
        let x = self.conv1.forward(&x)?;

        // Flatten the patch grid: [batch, width, gh*gw] -> [batch, gh*gw, width].
        let x = x.flatten_from(2)?.permute((0, 2, 1))?;

        // Add positional embedding (broadcast over batch).
        let x = x.broadcast_add(&self.positional_embedding)?;

        let mut x = self.ln_pre.forward(&x)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // Mean-pool over patch tokens, post-LayerNorm, project, L2-normalize.
        let pooled = x.mean(1)?;
        let pooled = self.ln_post.forward(&pooled)?;
        let projected = pooled.matmul(&self.audio_projection)?;
        l2_normalize(&projected)
    }

    /// Shared CLAP latent dimensionality of the output (`embed_dim`).
    pub fn embed_dim(&self) -> usize {
        self.config.embed_dim
    }

    /// Number of mel bins the input spectrogram must carry.
    pub fn n_mels(&self) -> usize {
        self.config.n_mels
    }

    /// Number of time frames every clip is padded/truncated to.
    pub fn n_frames(&self) -> usize {
        self.config.n_frames
    }

    /// Target sample rate (Hz) for decoding/resampling before feature extraction.
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// FFT window size (samples) for the short-time Fourier transform.
    pub fn n_fft(&self) -> usize {
        self.config.n_fft
    }

    /// Hop length (samples) between successive STFT frames.
    pub fn hop_length(&self) -> usize {
        self.config.hop_length
    }
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

    /// Replace every variable with a small random tensor so the encoder
    /// produces non-degenerate (non-all-zero) outputs in tests.
    fn randomize_varmap(varmap: &VarMap, device: &Device) {
        let data = varmap.data().lock().unwrap();
        for var in data.values() {
            let shape = var.shape().clone();
            let random = Tensor::randn(0f32, 0.1, shape, device).unwrap();
            var.set(&random).unwrap();
        }
    }

    fn tiny_config() -> ClapAudioConfig {
        ClapAudioConfig {
            n_mels: 8,
            n_frames: 16,
            patch_size: 4,
            width: 16,
            layers: 2,
            heads: 2,
            mlp_ratio: 4.0,
            embed_dim: 8,
            sample_rate: 16_000,
            n_fft: 256,
            hop_length: 128,
        }
    }

    #[test]
    fn config_from_clap_json() {
        let json = serde_json::json!({
            "model_cfg": {
                "embed_dim": 512,
                "audio_cfg": {
                    "n_mels": 64,
                    "n_frames": 1024,
                    "patch_size": 16,
                    "width": 768,
                    "layers": 12,
                    "heads": 12,
                    "mlp_ratio": 4.0,
                    "sample_rate": 48000,
                    "n_fft": 1024,
                    "hop_length": 480
                },
                "text_cfg": {}
            }
        });
        let cfg = ClapAudioConfig::from_clap_config(&json).unwrap();
        assert_eq!(cfg.embed_dim, 512);
        assert_eq!(cfg.width, 768);
        assert_eq!(cfg.layers, 12);
        assert_eq!(cfg.n_mels, 64);
        assert_eq!(cfg.n_frames, 1024);
        assert_eq!(cfg.patch_size, 16);
        assert_eq!(cfg.sample_rate, 48000);
        assert_eq!(cfg.n_fft, 1024);
        assert_eq!(cfg.hop_length, 480);
    }

    #[test]
    fn config_defaults_when_optional_fields_omitted() {
        let json = serde_json::json!({
            "model_cfg": {
                "embed_dim": 256,
                "audio_cfg": {
                    "n_mels": 64,
                    "n_frames": 256,
                    "width": 512,
                    "layers": 6
                }
            }
        });
        let cfg = ClapAudioConfig::from_clap_config(&json).unwrap();
        assert_eq!(cfg.patch_size, 16);
        assert_eq!(cfg.heads, 8); // 512 / 64
        assert_eq!(cfg.mlp_ratio, 4.0);
        assert_eq!(cfg.sample_rate, 48000);
    }

    #[test]
    fn config_rejects_indivisible_patch_grid() {
        let json = serde_json::json!({
            "model_cfg": {
                "embed_dim": 256,
                "audio_cfg": {
                    "n_mels": 30, // not divisible by patch_size 16
                    "n_frames": 256,
                    "width": 512,
                    "layers": 6
                }
            }
        });
        assert!(ClapAudioConfig::from_clap_config(&json).is_err());
    }

    #[test]
    fn forward_output_shape_and_l2_norm() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let model = ClapAudio::load(vb.pp("audio"), &cfg).unwrap();
        randomize_varmap(&varmap, &device);

        // batch=3 random log-mel spectrograms of the configured shape.
        let mel = Tensor::randn(0f32, 1.0, (3, cfg.n_mels, cfg.n_frames), &device).unwrap();
        let out = model.forward(&mel).unwrap();
        assert_eq!(out.dims(), &[3, cfg.embed_dim]);

        for row in out.to_vec2::<f32>().unwrap() {
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "L2 norm should be ~1.0, got {norm}"
            );
        }
    }

    #[test]
    fn forward_rejects_wrong_mel_shape() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = ClapAudio::load(vb.pp("audio"), &cfg).unwrap();

        let mel = Tensor::zeros((1, cfg.n_mels + 1, cfg.n_frames), DType::F32, &device).unwrap();
        assert!(model.forward(&mel).is_err());
    }
}
