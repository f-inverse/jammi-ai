//! Closed-enum dispatch over the audio encoder families used by `jammi-ai`.
//!
//! Parallel to the text-side [`crate::any::AnyEncoder`]: a family-erased
//! handle so callers can hold any supported audio encoder without trait-object
//! overhead. Every audio encoder maps a log-mel spectrogram batch
//! `[batch, n_mels, n_frames]` to pooled shared-latent embeddings
//! `[batch, embed_dim]`, and reports the feature-extraction parameters the
//! decode/resample/mel front-end needs (`n_mels`, `n_frames`, `sample_rate`,
//! `n_fft`, `hop_length`) so the front-end is config-driven, not hardcoded.

use candle_core::Tensor;

use crate::clap_audio::ClapAudio;
use crate::error::EncoderError;

/// The contract every audio encoder satisfies: spectrogram in, pooled
/// L2-normalized embedding out, plus the feature-extraction geometry the
/// front-end must match.
pub trait AudioEncoder {
    /// Pooled, L2-normalized `[batch, embed_dim]` embedding for a batch of
    /// log-mel spectrograms shaped `[batch, n_mels, n_frames]`.
    fn embed_batch(&self, mel: &Tensor) -> Result<Tensor, EncoderError>;
    /// Output embedding dimensionality (the shared cross-modal latent).
    fn embedding_dim(&self) -> usize;
    /// Mel bins the input spectrogram must carry.
    fn n_mels(&self) -> usize;
    /// Time frames every clip is padded/truncated to.
    fn n_frames(&self) -> usize;
    /// Target sample rate (Hz) for decode/resample before feature extraction.
    fn sample_rate(&self) -> u32;
    /// FFT window size (samples) for the short-time Fourier transform.
    fn n_fft(&self) -> usize;
    /// Hop length (samples) between successive STFT frames.
    fn hop_length(&self) -> usize;
}

impl AudioEncoder for ClapAudio {
    fn embed_batch(&self, mel: &Tensor) -> Result<Tensor, EncoderError> {
        self.forward(mel)
    }
    fn embedding_dim(&self) -> usize {
        self.embed_dim()
    }
    fn n_mels(&self) -> usize {
        self.n_mels()
    }
    fn n_frames(&self) -> usize {
        self.n_frames()
    }
    fn sample_rate(&self) -> u32 {
        self.sample_rate()
    }
    fn n_fft(&self) -> usize {
        self.n_fft()
    }
    fn hop_length(&self) -> usize {
        self.hop_length()
    }
}

/// Family-erased audio encoder for callers that need to hand around any of the
/// supported audio encoder types without trait-object overhead.
pub enum AnyAudioEncoder {
    /// The CLAP audio tower — shared-latent `[batch, embed_dim]` outputs from
    /// a log-mel spectrogram, compatible with the CLAP text tower for
    /// cross-modal text↔audio search.
    Clap(ClapAudio),
}

impl AudioEncoder for AnyAudioEncoder {
    fn embed_batch(&self, mel: &Tensor) -> Result<Tensor, EncoderError> {
        match self {
            Self::Clap(e) => e.embed_batch(mel),
        }
    }
    fn embedding_dim(&self) -> usize {
        match self {
            Self::Clap(e) => AudioEncoder::embedding_dim(e),
        }
    }
    fn n_mels(&self) -> usize {
        match self {
            Self::Clap(e) => AudioEncoder::n_mels(e),
        }
    }
    fn n_frames(&self) -> usize {
        match self {
            Self::Clap(e) => AudioEncoder::n_frames(e),
        }
    }
    fn sample_rate(&self) -> u32 {
        match self {
            Self::Clap(e) => AudioEncoder::sample_rate(e),
        }
    }
    fn n_fft(&self) -> usize {
        match self {
            Self::Clap(e) => AudioEncoder::n_fft(e),
        }
    }
    fn hop_length(&self) -> usize {
        match self {
            Self::Clap(e) => AudioEncoder::hop_length(e),
        }
    }
}
