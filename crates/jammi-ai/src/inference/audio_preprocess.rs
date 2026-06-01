//! Audio preprocessing for the audio-embedding path.
//!
//! Owns the bytes-in front-end the CLAP-family audio tower consumes: decode
//! encoded audio (WAV/FLAC/MP3/Ogg-Vorbis) to mono PCM, resample to the
//! model's target sample rate, compute a log-mel spectrogram, and pad or
//! truncate to the model's fixed time-frame window. Parallel to
//! [`super::image_preprocess`] — the caller hands raw bytes, the backend
//! produces a model-ready tensor; no DSP knobs are exposed.
//!
//! Feature-extraction geometry (sample rate, FFT size, hop, mel-bin count,
//! frame count) is supplied by the model config, never hardcoded.

use std::io::Cursor;

use candle_core::{Device, Tensor};
use jammi_db::error::{JammiError, Result};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Decoded mono PCM plus its native sample rate.
pub struct DecodedAudio {
    /// Mono PCM samples in `[-1.0, 1.0]` (multi-channel sources are averaged).
    pub samples: Vec<f32>,
    /// Native sample rate of the decoded stream, in Hz.
    pub sample_rate: u32,
}

/// Decode encoded audio bytes (WAV/FLAC/MP3/Ogg-Vorbis) to mono PCM.
///
/// Symphonia probes the container from the byte stream, decodes every packet,
/// and averages multi-channel frames to mono. Returns the native sample rate
/// so the caller can resample to the model's target rate.
pub fn decode_audio_bytes(bytes: &[u8]) -> Result<DecodedAudio> {
    let owned = bytes.to_vec();
    let source = Box::new(Cursor::new(owned));
    let mss = MediaSourceStream::new(source, Default::default());

    let probed = symphonia::default::get_probe()
        .format(
            &Hint::new(),
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| JammiError::Inference(format!("Failed to probe audio container: {e}")))?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| JammiError::Inference("Audio stream has no decodable track".into()))?;
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| JammiError::Inference(format!("Failed to construct audio decoder: {e}")))?;

    let mut sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut samples: Vec<f32> = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            // Clean end-of-stream: symphonia surfaces this as an UnexpectedEof
            // I/O error once the last packet is consumed.
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(e) => {
                return Err(JammiError::Inference(format!(
                    "Failed to read audio packet: {e}"
                )))
            }
        };
        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => {
                return Err(JammiError::Inference(format!(
                    "Failed to decode audio packet: {e}"
                )))
            }
        };

        let spec = *decoded.spec();
        if sample_rate == 0 {
            sample_rate = spec.rate;
        }
        let channels = spec.channels.count().max(1);

        let buf = sample_buf
            .get_or_insert_with(|| SampleBuffer::<f32>::new(decoded.capacity() as u64, spec));
        buf.copy_interleaved_ref(decoded);
        // Average interleaved channels down to mono.
        for frame in buf.samples().chunks(channels) {
            let sum: f32 = frame.iter().copied().sum();
            samples.push(sum / channels as f32);
        }
    }

    if sample_rate == 0 {
        return Err(JammiError::Inference(
            "Audio stream reported no sample rate".into(),
        ));
    }
    if samples.is_empty() {
        return Err(JammiError::Inference(
            "Audio stream decoded to zero samples".into(),
        ));
    }

    Ok(DecodedAudio {
        samples,
        sample_rate,
    })
}

/// Resample mono PCM from `from_rate` to `to_rate` by linear interpolation.
///
/// Linear interpolation is the right primitive for a feature-extraction
/// front-end: the downstream log-mel transform is robust to the mild
/// high-frequency rolloff it introduces, and it adds no codec dependency.
/// A no-op when the rates already match.
pub fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let out_len = ((samples.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let left = src_pos.floor() as usize;
        let frac = (src_pos - left as f64) as f32;
        let a = samples[left.min(samples.len() - 1)];
        let b = samples[(left + 1).min(samples.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

/// In-place iterative radix-2 Cooley-Tukey FFT over interleaved complex data.
///
/// `data` holds `n` complex numbers as `[re0, im0, re1, im1, ...]` where `n`
/// is a power of two. Iterative (no recursion) — bounded work, no stack risk
/// regardless of `n`.
fn fft_in_place(data: &mut [f32]) {
    let n = data.len() / 2;
    if n < 2 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "FFT length must be a power of two");

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 0..n {
        if i < j {
            data.swap(2 * i, 2 * j);
            data.swap(2 * i + 1, 2 * j + 1);
        }
        let mut m = n >> 1;
        while m >= 1 && j & m != 0 {
            j ^= m;
            m >>= 1;
        }
        j |= m;
    }

    // Danielson-Lanczos butterflies.
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let theta = -2.0 * std::f32::consts::PI / len as f32;
        let (wm_re, wm_im) = (theta.cos(), theta.sin());
        let mut start = 0;
        while start < n {
            let (mut w_re, mut w_im) = (1.0f32, 0.0f32);
            for k in 0..half {
                let i = start + k;
                let l = i + half;
                let (ir, ii) = (data[2 * i], data[2 * i + 1]);
                let (lr, li) = (data[2 * l], data[2 * l + 1]);
                let tr = w_re * lr - w_im * li;
                let ti = w_re * li + w_im * lr;
                data[2 * l] = ir - tr;
                data[2 * l + 1] = ii - ti;
                data[2 * i] = ir + tr;
                data[2 * i + 1] = ii + ti;
                let nw_re = w_re * wm_re - w_im * wm_im;
                w_im = w_re * wm_im + w_im * wm_re;
                w_re = nw_re;
            }
            start += len;
        }
        len <<= 1;
    }
}

/// Convert a frequency in Hz to the HTK mel scale.
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert a value on the HTK mel scale back to Hz.
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

/// Build a `[n_mels, n_fft/2 + 1]` triangular mel filterbank (HTK convention)
/// spanning `0..sample_rate/2`. Row-major: `filters[m * n_bins + b]`.
fn mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: u32) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let f_max = sample_rate as f32 / 2.0;
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(f_max);

    // n_mels + 2 mel points → n_mels triangular filters.
    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    // FFT bin index (fractional) for each mel point.
    let bin_points: Vec<f32> = hz_points
        .iter()
        .map(|&hz| hz * (n_fft as f32) / sample_rate as f32)
        .collect();

    let mut filters = vec![0f32; n_mels * n_bins];
    for m in 0..n_mels {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];
        for (b, slot) in filters[m * n_bins..(m + 1) * n_bins].iter_mut().enumerate() {
            let bf = b as f32;
            let w = if bf >= left && bf <= center && center > left {
                (bf - left) / (center - left)
            } else if bf > center && bf <= right && right > center {
                (right - bf) / (right - center)
            } else {
                0.0
            };
            *slot = w;
        }
    }
    filters
}

/// Compute a `[n_mels, n_frames]` log-mel spectrogram for one mono clip.
///
/// Frames the signal with a Hann-windowed STFT (`n_fft` window, `hop_length`
/// stride), maps each power spectrum through the triangular mel filterbank,
/// and takes the natural log. The clip is padded or truncated to exactly
/// `n_frames` frames so every clip yields a fixed-shape spectrogram (the
/// fixed-window pooling strategy).
#[allow(clippy::too_many_arguments)]
fn log_mel_spectrogram(
    samples: &[f32],
    n_mels: usize,
    n_frames: usize,
    n_fft: usize,
    hop_length: usize,
    filters: &[f32],
) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let hann: Vec<f32> = (0..n_fft)
        .map(|i| {
            let x = 2.0 * std::f32::consts::PI * i as f32 / n_fft as f32;
            0.5 * (1.0 - x.cos())
        })
        .collect();

    // Row-major mel spectrogram [n_mels, n_frames], log floor 1e-10.
    let mut mel = vec![(1e-10f32).ln(); n_mels * n_frames];
    let mut fft_buf = vec![0f32; 2 * n_fft];
    let mut power = vec![0f32; n_bins];

    for frame in 0..n_frames {
        let offset = frame * hop_length;
        // Window into the complex FFT buffer (imag = 0); zero-pad past the end.
        for k in 0..n_fft {
            let s = samples.get(offset + k).copied().unwrap_or(0.0);
            fft_buf[2 * k] = s * hann[k];
            fft_buf[2 * k + 1] = 0.0;
        }
        fft_in_place(&mut fft_buf);

        for (b, slot) in power.iter_mut().enumerate() {
            let re = fft_buf[2 * b];
            let im = fft_buf[2 * b + 1];
            *slot = re * re + im * im;
        }

        for m in 0..n_mels {
            let mut energy = 0f32;
            let row = &filters[m * n_bins..(m + 1) * n_bins];
            for (b, &w) in row.iter().enumerate() {
                energy += w * power[b];
            }
            mel[m * n_frames + frame] = energy.max(1e-10).ln();
        }
    }
    mel
}

/// Preprocess a batch of decoded clips into a model-ready `[batch, n_mels,
/// n_frames]` log-mel tensor.
///
/// Each clip is resampled to `sample_rate`, transformed to a fixed-window
/// log-mel spectrogram, and stacked. Clips shorter than `n_frames * hop`
/// samples are zero-padded; longer clips are truncated — fixed-window pooling
/// keyed on the model's configured frame count.
#[allow(clippy::too_many_arguments)]
pub fn preprocess_audio_batch(
    clips: &[DecodedAudio],
    n_mels: usize,
    n_frames: usize,
    n_fft: usize,
    hop_length: usize,
    sample_rate: u32,
    device: &Device,
) -> Result<Tensor> {
    if clips.is_empty() {
        return Err(JammiError::Inference(
            "Cannot preprocess empty audio batch".into(),
        ));
    }
    if !n_fft.is_power_of_two() {
        return Err(JammiError::Inference(format!(
            "Audio n_fft ({n_fft}) must be a power of two for the radix-2 FFT"
        )));
    }

    let filters = mel_filterbank(n_mels, n_fft, sample_rate);
    let per_clip = n_mels * n_frames;
    let mut flat = Vec::with_capacity(clips.len() * per_clip);

    for clip in clips {
        let resampled = resample_linear(&clip.samples, clip.sample_rate, sample_rate);
        let mel = log_mel_spectrogram(&resampled, n_mels, n_frames, n_fft, hop_length, &filters);
        flat.extend_from_slice(&mel);
    }

    Tensor::from_vec(flat, (clips.len(), n_mels, n_frames), device)
        .map_err(|e| JammiError::Inference(format!("Failed to create audio tensor: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal 16-bit PCM mono WAV in memory at the given sample rate.
    fn wav_bytes(samples: &[i16], sample_rate: u32) -> Vec<u8> {
        let data_len = (samples.len() * 2) as u32;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&(36 + data_len).to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&1u16.to_le_bytes()); // mono
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
        buf.extend_from_slice(&2u16.to_le_bytes()); // block align
        buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_len.to_le_bytes());
        for &s in samples {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        buf
    }

    fn sine_wav(freq: f32, sample_rate: u32, n: usize) -> Vec<u8> {
        let samples: Vec<i16> = (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (0.5 * (2.0 * std::f32::consts::PI * freq * t).sin() * i16::MAX as f32) as i16
            })
            .collect();
        wav_bytes(&samples, sample_rate)
    }

    #[test]
    fn decode_wav_round_trips_sample_rate_and_length() {
        let bytes = sine_wav(440.0, 16_000, 1600);
        let decoded = decode_audio_bytes(&bytes).unwrap();
        assert_eq!(decoded.sample_rate, 16_000);
        assert_eq!(decoded.samples.len(), 1600);
        assert!(decoded.samples.iter().all(|s| (-1.0..=1.0).contains(s)));
    }

    #[test]
    fn decode_rejects_garbage() {
        assert!(decode_audio_bytes(b"not audio at all").is_err());
    }

    #[test]
    fn resample_is_noop_when_rates_match() {
        let s = vec![0.1, 0.2, 0.3, 0.4];
        assert_eq!(resample_linear(&s, 16_000, 16_000), s);
    }

    #[test]
    fn resample_doubles_length_when_upsampling_2x() {
        let s = vec![0.0, 1.0, 0.0, 1.0];
        let out = resample_linear(&s, 8_000, 16_000);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn fft_matches_naive_dft_on_small_input() {
        // Impulse at index 1 → known DFT (cos/-sin ramp).
        let n = 8;
        let mut data = vec![0f32; 2 * n];
        data[2] = 1.0; // real part of sample index 1
        fft_in_place(&mut data);
        for k in 0..n {
            let angle = -2.0 * std::f32::consts::PI * k as f32 / n as f32;
            assert!((data[2 * k] - angle.cos()).abs() < 1e-4);
            assert!((data[2 * k + 1] - angle.sin()).abs() < 1e-4);
        }
    }

    #[test]
    fn mel_filterbank_rows_are_nonnegative_and_nonempty() {
        let filters = mel_filterbank(8, 64, 16_000);
        assert_eq!(filters.len(), 8 * (64 / 2 + 1));
        assert!(filters.iter().all(|&w| w >= 0.0));
        // At least one filter must have positive weight somewhere.
        assert!(filters.iter().any(|&w| w > 0.0));
    }

    #[test]
    fn preprocess_batch_produces_fixed_shape_and_distinguishes_pitch() {
        let device = Device::Cpu;
        let low = decode_audio_bytes(&sine_wav(220.0, 16_000, 4000)).unwrap();
        let high = decode_audio_bytes(&sine_wav(3000.0, 16_000, 4000)).unwrap();

        let n_mels = 16;
        let n_frames = 24;
        let tensor =
            preprocess_audio_batch(&[low, high], n_mels, n_frames, 256, 128, 16_000, &device)
                .unwrap();
        assert_eq!(tensor.dims(), &[2, n_mels, n_frames]);

        // The two clips differ in pitch → their mel spectrograms differ.
        let rows = tensor.to_vec3::<f32>().unwrap();
        let diff: f32 = rows[0]
            .iter()
            .flatten()
            .zip(rows[1].iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1.0,
            "distinct pitches must yield distinct spectrograms"
        );
    }

    #[test]
    fn preprocess_empty_batch_errors() {
        assert!(preprocess_audio_batch(&[], 16, 24, 256, 128, 16_000, &Device::Cpu).is_err());
    }

    #[test]
    fn preprocess_rejects_non_power_of_two_fft() {
        let clip = decode_audio_bytes(&sine_wav(440.0, 16_000, 2000)).unwrap();
        assert!(preprocess_audio_batch(&[clip], 16, 24, 200, 128, 16_000, &Device::Cpu).is_err());
    }
}
