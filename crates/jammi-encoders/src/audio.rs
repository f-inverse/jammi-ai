//! Closed-enum dispatch over the audio encoder families used by `jammi-ai`.
//!
//! Parallel to the text-side [`crate::any::AnyEncoder`]: a family-erased
//! handle so callers can hold any supported audio encoder without trait-object
//! overhead. Every audio encoder maps a 4-channel CLAP fusion spectrogram
//! `[batch, 4, time, num_mel_bins]` to pooled shared-latent embeddings
//! `[batch, embedding_dim]`, and reports the mel-bin count the spectrogram must
//! carry. The bytes-to-spectrogram front-end geometry (sample rate, FFT size,
//! hop, window length) is a feature-extractor concern owned by the caller's
//! `ClapFrontendConfig`, not the tower — the tower consumes the packed features
//! and is agnostic to how they were produced.

use candle_core::Tensor;

use crate::error::EncoderError;
use crate::htsat_audio::HtsatAudio;

/// The contract every audio encoder satisfies: 4-channel fusion spectrogram in,
/// pooled L2-normalized embedding out, plus the mel-bin count the input must
/// carry so the front-end can be config-driven.
pub trait AudioEncoder {
    /// Pooled, L2-normalized `[batch, embedding_dim]` embedding for a batch of
    /// CLAP fusion spectrograms shaped `[batch, 4, time, num_mel_bins]`,
    /// gated per clip by `is_longer` (whether the source clip exceeded the
    /// fixed fusion window).
    fn embed_batch(
        &self,
        input_features: &Tensor,
        is_longer: &[bool],
    ) -> Result<Tensor, EncoderError>;
    /// Output embedding dimensionality (the shared cross-modal latent).
    fn embedding_dim(&self) -> usize;
    /// Mel bins the input spectrogram must carry.
    fn num_mel_bins(&self) -> usize;
}

impl AudioEncoder for HtsatAudio {
    fn embed_batch(
        &self,
        input_features: &Tensor,
        is_longer: &[bool],
    ) -> Result<Tensor, EncoderError> {
        self.forward(input_features, is_longer)
    }
    fn embedding_dim(&self) -> usize {
        self.projection_dim()
    }
    fn num_mel_bins(&self) -> usize {
        self.num_mel_bins()
    }
}

/// Family-erased audio encoder for callers that need to hand around any of the
/// supported audio encoder types without trait-object overhead.
pub enum AnyAudioEncoder {
    /// The HTSAT-Swin CLAP audio tower — shared-latent `[batch, embedding_dim]`
    /// outputs from a 4-channel fusion spectrogram, compatible with the CLAP
    /// text tower for cross-modal text↔audio search.
    Htsat(HtsatAudio),
}

impl AudioEncoder for AnyAudioEncoder {
    fn embed_batch(
        &self,
        input_features: &Tensor,
        is_longer: &[bool],
    ) -> Result<Tensor, EncoderError> {
        match self {
            Self::Htsat(e) => e.embed_batch(input_features, is_longer),
        }
    }
    fn embedding_dim(&self) -> usize {
        match self {
            Self::Htsat(e) => AudioEncoder::embedding_dim(e),
        }
    }
    fn num_mel_bins(&self) -> usize {
        match self {
            Self::Htsat(e) => AudioEncoder::num_mel_bins(e),
        }
    }
}
