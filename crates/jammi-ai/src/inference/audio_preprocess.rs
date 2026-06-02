//! Audio preprocessing for the audio-embedding path.
//!
//! Owns the bytes-in front-end the CLAP-family audio tower consumes: decode
//! encoded audio (WAV/FLAC/MP3/Ogg-Vorbis) to mono PCM, resample to the
//! model's target sample rate, and produce a model-ready spectrogram tensor.
//! Parallel to [`super::image_preprocess`] — the caller hands raw bytes, the
//! backend produces a model-ready tensor; no DSP knobs are exposed.
//!
//! Two spectrogram front-ends live here:
//!
//! - [`preprocess_clap_fusion`] reproduces HuggingFace `ClapFeatureExtractor`
//!   (truncation `"fusion"`, padding `"repeatpad"`) exactly: repeatpad to the
//!   fixed window, reflect-centered Hann STFT (power 2), an HTK mel filterbank
//!   built in Hz space (`norm=None`), the `10·log10` dB nonlinearity, and the
//!   4-channel fusion packing. It emits `[batch, 4, time, n_mels]` and the
//!   per-clip `is_longer` flag the HTSAT tower's fusion path keys on. Every
//!   numeric is derived from a [`ClapFrontendConfig`] read off the model /
//!   feature-extractor config — nothing is hardcoded.
//! - [`preprocess_audio_batch`] is the fixed-window `[batch, n_mels, n_frames]`
//!   front-end of the flat-ViT `ClapAudio` tower. It remains until the audio
//!   backend is migrated onto the HTSAT tower + [`preprocess_clap_fusion`]; at
//!   that point this function and the flat-ViT tower are removed together.

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

// ===========================================================================
// CLAP fusion front-end — exact `ClapFeatureExtractor` reproduction.
// ===========================================================================

/// Feature-extraction geometry for the CLAP fusion front-end, read off the
/// model / feature-extractor config (never hardcoded).
///
/// Mirrors the `ClapFeatureExtractor` constructor arguments that affect the
/// numeric output for `truncation="fusion"`, `padding="repeatpad"`,
/// `top_db=None`: HTK mel scale, `norm=None`, Hann window (periodic), power 2,
/// `log_mel="dB"`, reflect-centered STFT.
#[derive(Debug, Clone, Copy)]
pub struct ClapFrontendConfig {
    /// Number of mel filters (`feature_size`), the `n_mels` output dimension.
    pub n_mels: usize,
    /// Target sample rate (Hz); input clips are resampled to it.
    pub sample_rate: u32,
    /// STFT window / FFT size in samples (`fft_window_size`). Must be a power
    /// of two for the radix-2 transform.
    pub fft_window_size: usize,
    /// Hop length (samples) between successive STFT frames.
    pub hop_length: usize,
    /// Lowest mel-filter frequency of interest, in Hz (`frequency_min`).
    pub frequency_min: f64,
    /// Highest mel-filter frequency of interest, in Hz (`frequency_max`).
    pub frequency_max: f64,
    /// Maximum input length in seconds (`max_length_s`); the fixed window is
    /// `max_length_s * sample_rate` samples.
    pub max_length_s: u32,
}

impl ClapFrontendConfig {
    /// The fixed-window length in samples (`nb_max_samples`): clips at or below
    /// it are repeatpadded up to it, longer clips take the fusion-crop path.
    fn nb_max_samples(&self) -> usize {
        self.max_length_s as usize * self.sample_rate as usize
    }

    /// The per-channel chunk frame count `T` of the packed output:
    /// `nb_max_samples // hop_length + 1`. The `+1` matches how the reference
    /// counts STFT frames over the padded window.
    fn chunk_frames(&self) -> usize {
        self.nb_max_samples() / self.hop_length + 1
    }
}

/// One clip's CLAP fusion features: the 4-channel dB mel `[4, time, n_mels]`
/// (row-major) and whether the source clip exceeded the fixed window.
pub struct ClapFusionFeatures {
    /// `[4, time, n_mels]` dB log-mel, row-major over `(channel, time, mel)`.
    pub features: Vec<f32>,
    /// Time-frame count `T` (equal to [`ClapFrontendConfig::chunk_frames`]).
    pub time: usize,
    /// `is_longer`: the source clip was longer than the fixed window, so the
    /// fusion (`mel_conv2d` + AFF) tower path applies.
    pub is_longer: bool,
}

/// Preprocess a batch of decoded clips into the CLAP fusion tensor
/// `[batch, 4, time, n_mels]` plus the per-clip `is_longer` flags.
///
/// Reproduces `ClapFeatureExtractor.__call__` for `truncation="fusion"`,
/// `padding="repeatpad"`: each clip is resampled to `config.sample_rate`,
/// repeatpadded (or, when longer than the window, left whole), run through the
/// reflect-centered Hann STFT and the HTK/`norm=None` mel filterbank to a dB
/// log-mel, then packed into 4 channels. The unbatched `_get_input_mel` path is
/// reproduced per clip; the batch-level "mark one clip longer at random" branch
/// is intentionally not — it is RNG the engine must not introduce, and the
/// tower keys fusion on the deterministic per-clip flag.
pub fn preprocess_clap_fusion(
    clips: &[DecodedAudio],
    config: &ClapFrontendConfig,
    device: &Device,
) -> Result<(Tensor, Vec<bool>)> {
    if clips.is_empty() {
        return Err(JammiError::Inference(
            "Cannot preprocess empty audio batch".into(),
        ));
    }
    if !config.fft_window_size.is_power_of_two() {
        return Err(JammiError::Inference(format!(
            "Audio fft_window_size ({}) must be a power of two for the radix-2 FFT",
            config.fft_window_size
        )));
    }

    let filters = mel_filterbank_hz(config);
    let window = hann_periodic(config.fft_window_size);
    let time = config.chunk_frames();
    let per_clip = 4 * time * config.n_mels;

    let mut flat = Vec::with_capacity(clips.len() * per_clip);
    let mut is_longer = Vec::with_capacity(clips.len());

    for clip in clips {
        let resampled = resample_linear(&clip.samples, clip.sample_rate, config.sample_rate);
        let feat = clap_fusion_features(&resampled, config, &filters, &window);
        debug_assert_eq!(feat.time, time);
        debug_assert_eq!(feat.features.len(), per_clip);
        flat.extend_from_slice(&feat.features);
        is_longer.push(feat.is_longer);
    }

    let tensor = Tensor::from_vec(flat, (clips.len(), 4, time, config.n_mels), device)
        .map_err(|e| JammiError::Inference(format!("Failed to create audio tensor: {e}")))?;
    Ok((tensor, is_longer))
}

/// `ClapFeatureExtractor._get_input_mel` for one resampled clip: repeatpad or
/// fusion-crop, dB log-mel, 4-channel packing.
fn clap_fusion_features(
    samples: &[f32],
    config: &ClapFrontendConfig,
    filters: &MelFilterbank,
    window: &[f64],
) -> ClapFusionFeatures {
    let max_length = config.nb_max_samples();
    let chunk = config.chunk_frames();
    let n_mels = config.n_mels;

    if samples.len() > max_length {
        // Fusion branch: mel over the WHOLE waveform, then crop + downsample.
        let whole: Vec<f64> = samples.iter().map(|&s| s as f64).collect();
        let mel = db_log_mel(&whole, config, filters, window); // [total, n_mels]
        let total = mel.len() / n_mels;
        if total == chunk {
            // Corner case (window < clip <= window + hop): use the whole mel
            // four times, marked not-longer — `_get_input_mel`'s
            // `chunk_frames == total_frames` branch.
            let features = stack4(&mel);
            ClapFusionFeatures {
                features,
                time: chunk,
                is_longer: false,
            }
        } else {
            let features = random_mel_fusion(&mel, total, chunk, n_mels);
            ClapFusionFeatures {
                features,
                time: chunk,
                is_longer: true,
            }
        }
    } else {
        // repeatpad branch: tile then zero-pad to the fixed window, one mel
        // stacked four times.
        let padded = repeatpad(samples, max_length);
        let mel = db_log_mel(&padded, config, filters, window); // [chunk, n_mels]
        debug_assert_eq!(mel.len() / n_mels, chunk);
        let features = stack4(&mel);
        ClapFusionFeatures {
            features,
            time: chunk,
            is_longer: false,
        }
    }
}

/// `repeatpad`: `tile(wave, int(max_length/len))` then zero-pad to `max_length`.
fn repeatpad(samples: &[f32], max_length: usize) -> Vec<f64> {
    let len = samples.len();
    // `int(max_length / len)` — floor division, matching numpy's `int(...)`.
    let n_repeat = max_length / len;
    let mut out = Vec::with_capacity(max_length);
    for _ in 0..n_repeat {
        out.extend(samples.iter().map(|&s| s as f64));
    }
    out.resize(max_length, 0.0);
    out
}

/// Stack one `[time, n_mels]` mel four times into `[4, time, n_mels]`.
fn stack4(mel: &[f64]) -> Vec<f32> {
    let mut out = Vec::with_capacity(4 * mel.len());
    for _ in 0..4 {
        out.extend(mel.iter().map(|&v| v as f32));
    }
    out
}

/// `_random_mel_fusion`: channel 0 is the full mel bilinearly downsampled in
/// time to `chunk` frames; channels 1-3 are three deterministic crops at the
/// front/middle/back of the `total - chunk + 1` valid offsets.
///
/// The crop offsets reproduce `np.array_split(range(0, total-chunk+1), 3)`
/// followed by the first element of each split (the deterministic anchor the
/// golden pins by choosing a waveform length that collapses each split to a
/// single index). `mel` is row-major `[total, n_mels]`; output is row-major
/// `[4, chunk, n_mels]`.
fn random_mel_fusion(mel: &[f64], total: usize, chunk: usize, n_mels: usize) -> Vec<f32> {
    let (front, middle, back) = fusion_crop_offsets(total, chunk);

    let mut out = vec![0f32; 4 * chunk * n_mels];

    // Channel 0: bilinear time-downsample `total -> chunk` (mel axis unchanged
    // since the target mel size equals the source). `align_corners=False`.
    let shrink = bilinear_time_downsample(mel, total, chunk, n_mels);
    for (d, &s) in out[..chunk * n_mels].iter_mut().zip(shrink.iter()) {
        *d = s as f32;
    }

    // Channels 1-3: the front/middle/back crops, `chunk` rows each.
    for (ch, off) in [(1, front), (2, middle), (3, back)] {
        let src = &mel[off * n_mels..(off + chunk) * n_mels];
        let dst = &mut out[ch * chunk * n_mels..(ch + 1) * chunk * n_mels];
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = s as f32;
        }
    }
    out
}

/// The deterministic crop anchors for `_random_mel_fusion`: the first index of
/// each of the three `np.array_split(range(0, n), 3)` parts, where
/// `n = total - chunk + 1`. `np.array_split(range(n), 3)` makes the first
/// `n % 3` parts one element longer; an empty 2nd/3rd part falls back to `[0]`.
fn fusion_crop_offsets(total: usize, chunk: usize) -> (usize, usize, usize) {
    let n = total - chunk + 1;
    // Sizes of the three array_split parts (first `n % 3` are `ceil`, rest
    // `floor`); the first index of part `i` is the running sum of prior sizes.
    let base = n / 3;
    let rem = n % 3;
    let size = |i: usize| base + usize::from(i < rem);
    let start0 = 0;
    let start1 = size(0);
    let start2 = size(0) + size(1);
    // Empty 2nd/3rd parts (n < 2 / n < 3) fall back to offset 0, mirroring the
    // reference's `ranges[1] = [0]` / `ranges[2] = [0]` guards.
    let front = start0;
    let middle = if size(1) == 0 { 0 } else { start1 };
    let back = if size(2) == 0 { 0 } else { start2 };
    (front, middle, back)
}

/// Bilinear downsample a `[total, n_mels]` mel along time to `[chunk, n_mels]`,
/// `align_corners=False` — the time axis of torch's
/// `F.interpolate(mode="bilinear", align_corners=False)`. The mel axis is left
/// untouched (target size equals source size, an identity along that axis).
fn bilinear_time_downsample(mel: &[f64], total: usize, chunk: usize, n_mels: usize) -> Vec<f64> {
    let scale = total as f64 / chunk as f64;
    let mut out = vec![0f64; chunk * n_mels];
    for ti in 0..chunk {
        // align_corners=False source coordinate.
        let src = (ti as f64 + 0.5) * scale - 0.5;
        let src_clamped = src.max(0.0);
        let lo = src_clamped.floor() as usize;
        let hi = (lo + 1).min(total - 1);
        let frac = src_clamped - lo as f64;
        let lo = lo.min(total - 1);
        for m in 0..n_mels {
            let a = mel[lo * n_mels + m];
            let b = mel[hi * n_mels + m];
            out[ti * n_mels + m] = a + (b - a) * frac;
        }
    }
    out
}

/// Floor applied to the linear-power mel before the dB nonlinearity
/// (`mel_floor` in the reference). With `reference=1.0` and `min_value=1e-10`,
/// the reference's dB clip is a no-op on top of this floor, so the dB log-mel
/// is exactly `10·log10` of the floored linear mel.
const MEL_FLOOR: f64 = 1e-10;

/// Compute the linear-power mel `[time, n_mels]` (row-major) for one mono clip
/// — the spectrogram BEFORE the dB nonlinearity.
///
/// Reproduces `spectrogram(power=2, mel_filters=…, log_mel=None)` followed by
/// the `.T`: reflect-center the waveform, Hann-window each frame, take the power
/// FFT, project through the mel filterbank, and floor at `mel_floor`.
/// Accumulated in f64 to match the reference's float64 FFT and dot product.
fn linear_mel(
    samples: &[f64],
    config: &ClapFrontendConfig,
    filters: &MelFilterbank,
    window: &[f64],
) -> Vec<f64> {
    let n_fft = config.fft_window_size;
    let n_mels = config.n_mels;
    let hop = config.hop_length;
    let n_bins = n_fft / 2 + 1;

    // Reflect-center pad by n_fft/2 each side.
    let padded = reflect_pad(samples, n_fft / 2);
    let num_frames = if padded.len() < n_fft {
        0
    } else {
        1 + (padded.len() - n_fft) / hop
    };

    let mut out = vec![0f64; num_frames * n_mels];
    let mut re = vec![0f64; n_fft];
    let mut im = vec![0f64; n_fft];
    let mut power = vec![0f64; n_bins];

    for frame in 0..num_frames {
        let offset = frame * hop;
        for k in 0..n_fft {
            re[k] = padded[offset + k] * window[k];
            im[k] = 0.0;
        }
        fft_f64(&mut re, &mut im);
        for (b, slot) in power.iter_mut().enumerate() {
            // power=2: |rfft|^2.
            *slot = re[b] * re[b] + im[b] * im[b];
        }
        // mel projection: max(mel_floor, filter . power).
        for m in 0..n_mels {
            let row = &filters.weights[m * n_bins..(m + 1) * n_bins];
            let mut energy = 0f64;
            for (b, &w) in row.iter().enumerate() {
                energy += w * power[b];
            }
            out[frame * n_mels + m] = energy.max(MEL_FLOOR);
        }
    }
    out
}

/// Compute the dB log-mel `[time, n_mels]` (row-major) for one mono clip:
/// [`linear_mel`] followed by the `log_mel="dB"` map `10·log10(·)` (the floor
/// already applied in [`linear_mel`], reference `1.0`).
fn db_log_mel(
    samples: &[f64],
    config: &ClapFrontendConfig,
    filters: &MelFilterbank,
    window: &[f64],
) -> Vec<f64> {
    let mut mel = linear_mel(samples, config, filters, window);
    for v in &mut mel {
        *v = 10.0 * v.log10();
    }
    mel
}

/// Reflect-pad a signal by `pad` samples on each side (numpy `mode="reflect"`:
/// the edge sample is not repeated). Requires `len > pad`.
fn reflect_pad(samples: &[f64], pad: usize) -> Vec<f64> {
    let len = samples.len();
    let mut out = Vec::with_capacity(len + 2 * pad);
    // Left reflection: samples[pad], samples[pad-1], ..., samples[1].
    for i in 0..pad {
        out.push(samples[pad - i]);
    }
    out.extend_from_slice(samples);
    // Right reflection: samples[len-2], samples[len-3], ..., samples[len-1-pad].
    for i in 0..pad {
        out.push(samples[len - 2 - i]);
    }
    out
}

/// Periodic Hann window of length `n` (numpy `hanning(n+1)[:-1]`):
/// `0.5 - 0.5·cos(2π·i/n)` for `i in 0..n`.
fn hann_periodic(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
        .collect()
}

/// A `[n_mels, n_bins]` triangular mel filterbank, row-major.
struct MelFilterbank {
    weights: Vec<f64>,
}

/// HTK-scale mel filterbank built in Hz space with `norm=None`, matching
/// `mel_filter_bank(..., norm=None, mel_scale="htk")`.
///
/// FFT-bin centre frequencies are `linspace(0, sample_rate/2, n_bins)` in Hz;
/// filter edge frequencies are `mel_to_hz(linspace(mel(fmin), mel(fmax),
/// n_mels+2))`. The triangular weights follow `_create_triangular_filter_bank`:
/// `max(0, min(up_slope, down_slope))` per (bin, filter).
fn mel_filterbank_hz(config: &ClapFrontendConfig) -> MelFilterbank {
    let n_bins = config.fft_window_size / 2 + 1;
    let n_mels = config.n_mels;

    // FFT bin frequencies: linspace(0, sample_rate/2, n_bins), endpoint=True.
    // numpy uses sample_rate // 2 (integer) as the stop, matching the reference.
    let fft_stop = (config.sample_rate / 2) as f64;
    let fft_freqs: Vec<f64> = (0..n_bins)
        .map(|i| fft_stop * i as f64 / (n_bins - 1) as f64)
        .collect();

    // Filter edge frequencies in Hz: n_mels + 2 points equally spaced in mel.
    let mel_min = hz_to_mel_htk(config.frequency_min);
    let mel_max = hz_to_mel_htk(config.frequency_max);
    let filter_freqs: Vec<f64> = (0..n_mels + 2)
        .map(|i| {
            let mel = mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64;
            mel_to_hz_htk(mel)
        })
        .collect();

    // _create_triangular_filter_bank: for each fft bin and each filter,
    // down = (f - left)/(center - left), up = (right - f)/(right - center),
    // weight = max(0, min(down, up)). Expressed via filter_diff like the source.
    let filter_diff: Vec<f64> = (0..filter_freqs.len() - 1)
        .map(|i| filter_freqs[i + 1] - filter_freqs[i])
        .collect();

    let mut weights = vec![0f64; n_mels * n_bins];
    for m in 0..n_mels {
        for (b, &f) in fft_freqs.iter().enumerate() {
            // slopes[b, j] = filter_freqs[j] - f; down uses j=m, up uses j=m+2.
            let down = -(filter_freqs[m] - f) / filter_diff[m];
            let up = (filter_freqs[m + 2] - f) / filter_diff[m + 1];
            weights[m * n_bins + b] = down.min(up).max(0.0);
        }
    }
    MelFilterbank { weights }
}

/// Hz -> HTK mel: `2595·log10(1 + hz/700)`.
fn hz_to_mel_htk(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// HTK mel -> Hz: `700·(10^(mel/2595) - 1)`.
fn mel_to_hz_htk(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}

/// In-place iterative radix-2 Cooley-Tukey FFT over split real/imag f64 arrays.
///
/// `re`/`im` hold `n` complex samples (`n` a power of two). f64 throughout to
/// match numpy's float64 FFT — the reference accumulates the STFT and the mel
/// dot product in float64, and an f32 transform diverges past the derived
/// pre-dB tolerance. Iterative (no recursion) — bounded work, no stack risk.
fn fft_f64(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    if n < 2 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "FFT length must be a power of two");
    debug_assert_eq!(re.len(), im.len());

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 0..n {
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
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
        let theta = -2.0 * std::f64::consts::PI / len as f64;
        let (wm_re, wm_im) = (theta.cos(), theta.sin());
        let mut start = 0;
        while start < n {
            let (mut w_re, mut w_im) = (1.0f64, 0.0f64);
            for k in 0..half {
                let i = start + k;
                let l = i + half;
                let tr = w_re * re[l] - w_im * im[l];
                let ti = w_re * im[l] + w_im * re[l];
                re[l] = re[i] - tr;
                im[l] = im[i] - ti;
                re[i] += tr;
                im[i] += ti;
                let nw_re = w_re * wm_re - w_im * wm_im;
                w_im = w_re * wm_im + w_im * wm_re;
                w_re = nw_re;
            }
            start += len;
        }
        len <<= 1;
    }
}

// ===========================================================================
// Flat-ViT front-end — fixed-window `[batch, n_mels, n_frames]` log-mel.
//
// Serves the `ClapAudio` (flat-ViT) tower the audio backend currently runs.
// Removed together with that tower once the backend migrates onto the HTSAT
// tower + `preprocess_clap_fusion` above.
// ===========================================================================

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
/// n_frames]` log-mel tensor for the flat-ViT `ClapAudio` tower.
///
/// Each clip is resampled to `sample_rate`, transformed to a fixed-window
/// log-mel spectrogram, and stacked. Clips shorter than `n_frames * hop`
/// samples are zero-padded; longer clips are truncated — fixed-window pooling
/// keyed on the model's configured frame count.
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
    fn fft_f64_matches_naive_dft_on_small_input() {
        // Impulse at index 1 → known DFT (cos/-sin ramp), f64 precision.
        let n = 8;
        let mut re = vec![0f64; n];
        let mut im = vec![0f64; n];
        re[1] = 1.0;
        fft_f64(&mut re, &mut im);
        for k in 0..n {
            let angle = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
            assert!((re[k] - angle.cos()).abs() < 1e-12);
            assert!((im[k] - angle.sin()).abs() < 1e-12);
        }
    }

    #[test]
    fn reflect_pad_mirrors_without_repeating_edge() {
        // numpy.pad([1,2,3,4], 2, mode="reflect") == [3,2,1,2,3,4,3,2].
        let s = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            reflect_pad(&s, 2),
            vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
        );
    }

    #[test]
    fn fusion_crop_offsets_collapse_to_singletons_for_split3() {
        // total - chunk + 1 == 3 -> array_split(range(3),3) == [[0],[1],[2]].
        assert_eq!(fusion_crop_offsets(5, 3), (0, 1, 2));
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

    // -- CLAP fusion front-end parity against the committed golden -----------
    //
    // Oracle: `cookbook/fixtures/htsat_clap_frontend/goldens.safetensors`,
    // dumped from the real HuggingFace `ClapFeatureExtractor`
    // (laion/clap-htsat-fused geometry, truncation="fusion") by
    // `tests/fixtures/generate_clap_frontend.py`. Hermetic: the golden is a
    // committed binary, no torch / network at test time.
    //
    // Two metrics, each with a bound derived from the measured error (printed
    // below; see the test report for the numbers that justify each bound):
    //
    //  * pre-dB LINEAR-power mel — the faithful parity space. dB values span
    //    ~-100..+40, so a max-abs on dB would be dominated by the log's
    //    amplification of tiny linear differences near the 1e-10 floor. The
    //    linear mel is the honest space; bound is RELATIVE because the linear
    //    magnitudes span many orders of magnitude.
    //  * final dB packed `input_features` — checked separately with its own
    //    max-abs bound, covering the dB nonlinearity AND the crop/bilinear
    //    fusion packing (which the reference performs in dB space).

    use candle_core::{DType, Device as CandleDevice};
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn frontend_fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../cookbook/fixtures/htsat_clap_frontend")
    }

    /// Read the feature-extractor geometry from the committed manifest so the
    /// front-end under test is driven by config, not hardcoded numbers.
    fn fixture_config() -> ClapFrontendConfig {
        let manifest = std::fs::read_to_string(frontend_fixture_dir().join("golden_manifest.json"))
            .expect("read golden_manifest.json");
        let v: serde_json::Value = serde_json::from_str(&manifest).expect("parse manifest");
        let p = &v["params"];
        ClapFrontendConfig {
            n_mels: p["feature_size"].as_u64().unwrap() as usize,
            sample_rate: p["sampling_rate"].as_u64().unwrap() as u32,
            fft_window_size: p["fft_window_size"].as_u64().unwrap() as usize,
            hop_length: p["hop_length"].as_u64().unwrap() as usize,
            frequency_min: p["frequency_min"].as_f64().unwrap(),
            frequency_max: p["frequency_max"].as_f64().unwrap(),
            max_length_s: p["max_length_s"].as_u64().unwrap() as u32,
        }
    }

    fn load_goldens() -> HashMap<String, Tensor> {
        candle_core::safetensors::load(
            frontend_fixture_dir().join("goldens.safetensors"),
            &CandleDevice::Cpu,
        )
        .expect("load frontend goldens.safetensors")
    }

    fn tensor_f64(t: &Tensor) -> Vec<f64> {
        t.to_dtype(DType::F64)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f64>()
            .unwrap()
    }

    /// Reproduce, in f64, the exact waveform handling `clap_fusion_features`
    /// applies before the dB nonlinearity, returning the UNPACKED linear-power
    /// mel `[time, n_mels]` — the faithful pre-dB parity space.
    fn unpacked_linear_mel(wave: &[f64], config: &ClapFrontendConfig) -> Vec<f64> {
        let filters = mel_filterbank_hz(config);
        let window = hann_periodic(config.fft_window_size);
        let max_length = config.nb_max_samples();
        if wave.len() > max_length {
            linear_mel(wave, config, &filters, &window)
        } else {
            // repeatpad operates on the f32 PCM contract; reproduce it in f64
            // directly from the f64 golden waveform to isolate the front-end
            // algorithm error from the f32 input-quantization error.
            let len = wave.len();
            let n_repeat = max_length / len;
            let mut padded = Vec::with_capacity(max_length);
            for _ in 0..n_repeat {
                padded.extend_from_slice(wave);
            }
            padded.resize(max_length, 0.0);
            linear_mel(&padded, config, &filters, &window)
        }
    }

    /// Max absolute and max relative (vs |golden|, floored at the mel floor)
    /// element error between two equal-length flat slices.
    fn errors(got: &[f64], want: &[f64]) -> (f64, f64) {
        assert_eq!(got.len(), want.len(), "length mismatch");
        let mut max_abs = 0f64;
        let mut max_rel = 0f64;
        for (&g, &w) in got.iter().zip(want.iter()) {
            let abs = (g - w).abs();
            max_abs = max_abs.max(abs);
            // Floor the denominator at the mel floor so near-floor cells (which
            // carry no signal) don't manufacture a huge relative error.
            let rel = abs / w.abs().max(MEL_FLOOR);
            max_rel = max_rel.max(rel);
        }
        (max_abs, max_rel)
    }

    #[test]
    fn clap_fusion_pre_db_linear_mel_matches_golden() {
        let config = fixture_config();
        let goldens = load_goldens();

        let mut worst_abs = 0f64;
        let mut worst_rel = 0f64;
        for tag in ["short", "long"] {
            let wave = tensor_f64(&goldens[&format!("{tag}_waveform")]);
            let want = tensor_f64(&goldens[&format!("{tag}_mel_linear")]);
            let got = unpacked_linear_mel(&wave, &config);
            let (abs, rel) = errors(&got, &want);
            println!("[{tag}] pre-dB linear mel: max_abs={abs:.3e} max_rel={rel:.3e}");
            worst_abs = worst_abs.max(abs);
            worst_rel = worst_rel.max(rel);
        }
        println!("pre-dB linear mel WORST: max_abs={worst_abs:.3e} max_rel={worst_rel:.3e}");

        // Bound derived from the measured worst-case relative error across both
        // clips: max_rel = 1.13e-7 (short 1.08e-7, long 1.13e-7), at the f64
        // FFT-butterfly-ordering floor — the f64 algorithm reproduces the
        // reference to ~1 part in 1e7. The linear-power mel spans many orders of
        // magnitude, so a RELATIVE bound is the honest metric. 5e-7 sits ~4x
        // above the measured worst (margin for platform FFT-order variation); a
        // wrong filterbank scale, window, or FFT diverges by >=1e-2 relative and
        // cannot pass.
        assert!(
            worst_rel < 5e-7,
            "pre-dB linear mel relative error {worst_rel:.3e} exceeds 5e-7"
        );
    }

    #[test]
    fn clap_fusion_db_input_features_match_golden() {
        let config = fixture_config();
        let goldens = load_goldens();

        let mut worst_abs = 0f64;
        for tag in ["short", "long"] {
            let wave = tensor_f64(&goldens[&format!("{tag}_waveform")]);
            // Drive the realistic jammi path: f32 PCM samples in a DecodedAudio
            // at the target rate (no resample), through the full public API.
            let clip = DecodedAudio {
                samples: wave.iter().map(|&s| s as f32).collect(),
                sample_rate: config.sample_rate,
            };
            let (tensor, is_longer) =
                preprocess_clap_fusion(&[clip], &config, &Device::Cpu).unwrap();

            let want_t = &goldens[&format!("{tag}_input_features")];
            assert_eq!(
                tensor.dims(),
                &[1, 4, want_t.dim(1).unwrap(), config.n_mels],
                "{tag}: shape mismatch"
            );
            let expect_longer = tag == "long";
            assert_eq!(is_longer, vec![expect_longer], "{tag}: is_longer mismatch");

            let got = tensor_f64(&tensor);
            let want = tensor_f64(want_t);
            let (abs, _) = errors(&got, &want);
            println!("[{tag}] dB input_features: max_abs={abs:.3e}");
            worst_abs = worst_abs.max(abs);
        }
        println!("dB input_features WORST: max_abs={worst_abs:.3e}");

        // Bound derived from the measured worst-case max-abs: 1.61e-3 (short
        // 2.67e-5, long 1.61e-3). The long clip's worst is set by the f32 PCM
        // input contract — f32-quantized samples amplified through the dB log at
        // low-energy cells — which is jammi's REAL front-end path, plus the
        // f64→f32 output cast. 5e-3 sits ~3x above the measured worst (margin
        // for input-dependent f32-rounding variation); a wrong dB map, crop
        // offset, or bilinear downsample diverges by whole dB and cannot pass.
        assert!(
            worst_abs < 5e-3,
            "dB input_features max-abs error {worst_abs:.3e} exceeds 5e-3"
        );
    }
}
