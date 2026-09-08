//! Audio preprocessing for the audio-embedding path.
//!
//! Owns the bytes-in front-end the CLAP-family audio tower consumes: decode
//! encoded audio (WAV/FLAC/MP3/Ogg-Vorbis) to mono PCM, resample to the
//! model's target sample rate, and produce a model-ready spectrogram tensor.
//! Parallel to [`super::image_preprocess`] — the caller hands raw bytes, the
//! backend produces a model-ready tensor; no DSP knobs are exposed.
//!
//! [`preprocess_clap_fusion`] reproduces HuggingFace `ClapFeatureExtractor`
//! (truncation `"fusion"`, padding `"repeatpad"`) exactly: repeatpad to the
//! fixed window, reflect-centered Hann STFT (power 2), an HTK mel filterbank
//! built in Hz space (`norm=None`), the `10·log10` dB nonlinearity, and the
//! 4-channel fusion packing. It emits `[batch, 4, time, n_mels]` plus a
//! `is_longer` flag per clip that is DETERMINISTICALLY `true` for every clip
//! (see [`preprocess_clap_fusion`]), feeding the HTSAT-Swin CLAP audio tower.
//! Every numeric is derived from a [`ClapFrontendConfig`] read off the model /
//! feature-extractor config — nothing is hardcoded.

use std::io::Cursor;

use candle_core::{Device, Tensor};
use jammi_db::error::{JammiError, Result};
use rayon::prelude::*;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Decoded mono PCM plus its native sample rate.
#[derive(Debug)]
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

/// Decode a batch of encoded audio byte buffers to mono PCM in PARALLEL,
/// across whichever rayon pool the call runs under (its global pool in
/// production; candle installs no private pool of its own, so this is the
/// one pool the process ever schedules media-batch work on).
///
/// Used by `fine_tune::trainer`'s `audio_encoder_input`, whose items are
/// already the row space the caller wants reported (no compaction ever
/// happens ahead of it — every `MediaTriplet` blob is present by
/// construction), so this thin wrapper reports positions `0..items.len()`
/// via [`decode_audio_batch_indexed`]. A caller that compacts rows first
/// (dropping nulls before decoding, as [`super::arrow_to_audio`] and
/// `CandleBackend::forward_audio_embedding` both do) must call
/// [`decode_audio_batch_indexed`] directly with the ORIGINAL row ids, or
/// every error message after the first null row reports the wrong (compacted)
/// position instead of the caller's row.
pub fn decode_audio_batch<T>(items: &[T]) -> Result<Vec<DecodedAudio>>
where
    T: AsRef<[u8]> + Sync,
{
    let row_ids: Vec<usize> = (0..items.len()).collect();
    decode_audio_batch_indexed(&row_ids, items)
}

/// [`decode_audio_batch`], reporting `row_ids[i]` (not the position `i`) in
/// every error — for a caller that has already compacted its batch (dropped
/// null rows) before decoding, so the position in `items` and the row the
/// caller (and its own caller, in turn) means by "row N" have diverged.
/// `row_ids.len()` must equal `items.len()`.
///
/// One decode task per item — chunk count is `items.len()`, no thread-count
/// knob, so the effective parallelism is `min(pool_size, items.len())`,
/// emergent from whichever pool is installed.
///
/// Errors are collected per row and the LOWEST-INDEX failing row is the one
/// surfaced, with a row-indexed message — the same selection the pre-unit
/// sequential decode loop made by construction (it returned on the first
/// failure it walked into, in row order).
pub fn decode_audio_batch_indexed<T>(row_ids: &[usize], items: &[T]) -> Result<Vec<DecodedAudio>>
where
    T: AsRef<[u8]> + Sync,
{
    crate::inference::lowest_index_result(decode_audio_results(row_ids, items)?)
}

/// [`decode_audio_batch_indexed`]'s per-row outcomes, WITHOUT collapsing to
/// the lowest-index failure — for a caller (serving's `arrow_to_audio`) that
/// marks each row's OWN status rather than refusing the whole batch because
/// one row's bytes were corrupt. Every element is independently `Ok` or a
/// row-indexed `Err`, in the same order as `items`/`row_ids`.
pub fn decode_audio_batch_per_row_indexed<T>(
    row_ids: &[usize],
    items: &[T],
) -> Result<Vec<Result<DecodedAudio>>>
where
    T: AsRef<[u8]> + Sync,
{
    decode_audio_results(row_ids, items)
}

/// The parallel per-row decode shared by [`decode_audio_batch_indexed`] and
/// [`decode_audio_batch_per_row_indexed`] — the only difference between the
/// two public entry points is whether the caller wants ALL per-row outcomes
/// or just the lowest-index failure.
///
/// `row_ids.len() != items.len()` is a typed error, not a `debug_assert!`: in
/// a release build a `debug_assert!` compiles out, and `.zip()` silently
/// truncates to the SHORTER of the two slices — a short `row_ids` would then
/// return zero outcomes for every item past its length, not an error.
fn decode_audio_results<T>(row_ids: &[usize], items: &[T]) -> Result<Vec<Result<DecodedAudio>>>
where
    T: AsRef<[u8]> + Sync,
{
    if row_ids.len() != items.len() {
        return Err(JammiError::Inference(format!(
            "decode_audio: row_ids has {} entries, expected one per item ({})",
            row_ids.len(),
            items.len()
        )));
    }
    Ok(items
        .par_iter()
        .zip(row_ids.par_iter())
        .map(|(item, &row)| {
            decode_audio_bytes(item.as_ref()).map_err(|e| {
                JammiError::Inference(format!("Failed to decode audio at row {row}: {e}"))
            })
        })
        .collect())
}

/// The sample count [`resample_linear`] produces for `len` input samples
/// resampled from `from_rate` to `to_rate`, WITHOUT doing the resample —
/// shared by [`resample_linear`] itself and by
/// [`preprocess_clap_fusion_indexed`]'s pre-check, which needs this exact
/// arithmetic to refuse a clip that would round to zero output samples
/// BEFORE the parallel per-clip stage ever runs `repeatpad` on it (a
/// zero-length `repeatpad` input is an integer-division-by-zero panic:
/// `max_length / len` with `len == 0`).
///
/// `from_rate == 0` is a degenerate source rate a `DecodedAudio` should never
/// carry (its own decoder refuses one), but this function has no `Result` to
/// refuse through, so it treats a zero `from_rate` as producing zero output
/// samples rather than dividing by it: `to_rate as f64 / 0.0` is `+inf`, and
/// `(len as f64 * inf).round() as usize` saturates to `usize::MAX` — a huge,
/// confidently wrong length that then feeds `Vec::with_capacity(usize::MAX)`
/// in [`resample_linear`] instead of the empty-clip guard below ever seeing
/// it. Returning `0` here lets that same guard catch it by the ordinary
/// "rounds to zero samples" path.
fn resampled_len(len: usize, from_rate: u32, to_rate: u32) -> usize {
    if from_rate == 0 {
        return 0;
    }
    if from_rate == to_rate || len == 0 {
        return len;
    }
    let ratio = to_rate as f64 / from_rate as f64;
    ((len as f64) * ratio).round() as usize
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
    let out_len = resampled_len(samples.len(), from_rate, to_rate);
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

    /// Validate every numeric field the fusion front-end's math requires to
    /// stay in a well-defined domain — called at the PARSE edge
    /// (`clap_frontend_from_preprocessor`, right after a
    /// `preprocessor_config.json` is read), never deferred to the point of
    /// failure deep inside the transform:
    ///
    /// - `hop_length == 0` makes `chunk_frames`'s `nb_max_samples /
    ///   hop_length` an integer-division-by-zero panic.
    /// - `sample_rate == 0` makes [`resample_linear`]'s ratio zero, so the
    ///   resampled clip is empty and `repeatpad`'s `max_length / len`
    ///   panics on the resulting `0 / 0`.
    /// - `max_length_s == 0` collapses `nb_max_samples` to zero
    ///   without panicking anywhere, so every clip silently takes the
    ///   fusion-crop branch over a near-empty window instead of failing —
    ///   a confident-wrong shape, not a crash.
    /// - `fft_window_size` must be a power of two (for the radix-2 FFT) AND
    ///   at least 2: at `fft_window_size == 1`, `mel_filterbank_hz`'s bin
    ///   count minus one is zero and its FFT-bin-frequency division degrades
    ///   to `0.0 / 0.0` (NaN), silently, not a panic.
    /// - `frequency_min`/`frequency_max` must be finite, ordered, and the
    ///   upper edge must not exceed the Nyquist frequency implied by
    ///   `sample_rate` — a filter edge past Nyquist is not a meaningful mel
    ///   band for real audio.
    pub fn validate(&self) -> Result<()> {
        if self.n_mels == 0 {
            return Err(JammiError::Inference(format!(
                "Audio n_mels (feature_size) must be positive, got {}",
                self.n_mels
            )));
        }
        if self.sample_rate == 0 {
            return Err(JammiError::Inference(
                "Audio sample_rate (sampling_rate) must be positive, got 0".into(),
            ));
        }
        if self.fft_window_size < 2 || !self.fft_window_size.is_power_of_two() {
            return Err(JammiError::Inference(format!(
                "Audio fft_window_size ({}) must be a power of two >= 2 for the radix-2 FFT",
                self.fft_window_size
            )));
        }
        if self.hop_length == 0 {
            return Err(JammiError::Inference(
                "Audio hop_length must be positive, got 0".into(),
            ));
        }
        if self.max_length_s == 0 {
            return Err(JammiError::Inference(
                "Audio max_length_s must be positive, got 0".into(),
            ));
        }
        if !self.frequency_min.is_finite() || self.frequency_min < 0.0 {
            return Err(JammiError::Inference(format!(
                "Audio frequency_min must be finite and non-negative, got {}",
                self.frequency_min
            )));
        }
        if !self.frequency_max.is_finite() || self.frequency_max <= self.frequency_min {
            return Err(JammiError::Inference(format!(
                "Audio frequency_max ({}) must be finite and greater than frequency_min ({})",
                self.frequency_max, self.frequency_min
            )));
        }
        let nyquist = self.sample_rate as f64 / 2.0;
        if self.frequency_max > nyquist {
            return Err(JammiError::Inference(format!(
                "Audio frequency_max ({}) must not exceed the Nyquist frequency ({nyquist}) \
                 implied by sample_rate ({})",
                self.frequency_max, self.sample_rate
            )));
        }
        Ok(())
    }
}

/// One clip's CLAP fusion features: the 4-channel dB mel `[4, time, n_mels]`
/// (row-major). The channel construction is length-determined (short clips →
/// the repeatpad mel stacked four times; long clips → crops + downsample); the
/// emitted `is_longer` flag the tower gates on is constant `true` (set at the
/// batch level in [`preprocess_clap_fusion`]), so it is not carried here.
pub struct ClapFusionFeatures {
    /// `[4, time, n_mels]` dB log-mel, row-major over `(channel, time, mel)`.
    pub features: Vec<f32>,
    /// Time-frame count `T` (equal to the frontend's fixed `chunk_frames`).
    pub time: usize,
}

/// Preprocess a batch of decoded clips into the CLAP fusion tensor
/// `[batch, 4, time, n_mels]` plus the `is_longer` flags (always `true`).
///
/// Reproduces `ClapFeatureExtractor.__call__` for `truncation="fusion"`,
/// `padding="repeatpad"`: each clip is resampled to `config.sample_rate`,
/// repeatpadded (or, when longer than the window, left whole), run through the
/// reflect-centered Hann STFT and the HTK/`norm=None` mel filterbank to a dB
/// log-mel, then packed into 4 channels.
///
/// `is_longer` policy — read carefully. HF's `ClapFeatureExtractor`
/// deterministically promotes a single clip to `is_longer=True` even when the
/// whole batch fits inside the fixed window (its `_get_input_mel` marks the
/// global-only mel "longer" so the AFF fusion path runs), so `get_audio_features`
/// returns the FUSION embedding — the vector the CLAP ecosystem indexes and
/// searches with. To reproduce that canonical embedding, jammi marks EVERY clip
/// `is_longer=true` (its deterministic analogue of the feature extractor's
/// promotion), rather than RNG-promoting one clip or emitting the global-only
/// flag (which yields a different, ecosystem-incompatible embedding ~0.73 cosine
/// off). The CHANNEL construction stays length-determined (short → the repeatpad
/// mel stacked four times; long → crops + downsample); only the emitted gate the
/// tower keys fusion on is forced on, so the flags are simply `vec![true; n]`.
///
/// Reports row positions `0..clips.len()` in any per-row error via
/// [`preprocess_clap_fusion_indexed`]; see that function's doc for the
/// compacted-batch case (`CandleBackend::forward_audio_embedding` calls it
/// directly with the original Arrow row ids for exactly that reason).
pub fn preprocess_clap_fusion(
    clips: &[DecodedAudio],
    config: &ClapFrontendConfig,
    device: &Device,
) -> Result<(Tensor, Vec<bool>)> {
    let row_ids: Vec<usize> = (0..clips.len()).collect();
    preprocess_clap_fusion_indexed(&row_ids, clips, config, device)
}

/// [`preprocess_clap_fusion`], reporting `row_ids[i]` (not the position `i`)
/// in every per-row error — for a caller that has already compacted its
/// batch (dropped null rows) before preprocessing, so the position in
/// `clips` and the row its own caller means by "row N" have diverged.
/// `row_ids.len()` must equal `clips.len()`.
///
/// Preallocates the whole batch's flat buffer, then writes each clip's
/// disjoint, fixed-stride `per_clip` chunk in PARALLEL across whichever
/// rayon pool this call runs under (`par_chunks_mut` zipped with the clips —
/// one task per clip, no thread-count knob). `filters` and `window` are
/// computed once and shared read-only by every task. `row_ids` only changes
/// what number a row is called in an error message; it never reorders the
/// chunk a given clip writes to, so it cannot change the numeric output.
pub fn preprocess_clap_fusion_indexed(
    row_ids: &[usize],
    clips: &[DecodedAudio],
    config: &ClapFrontendConfig,
    device: &Device,
) -> Result<(Tensor, Vec<bool>)> {
    if clips.is_empty() {
        return Err(JammiError::Inference(
            "Cannot preprocess empty audio batch".into(),
        ));
    }
    // The comprehensive domain check ([`ClapFrontendConfig::validate`]) is
    // also run at the config's PARSE edge
    // (`clap_frontend_from_preprocessor`), but re-running it here defends a
    // caller that builds/mutates a config directly (as this module's own
    // tests do) rather than going through that funnel — a corrupt-domain
    // config must never reach `mel_filterbank_hz`/`chunk_frames`/`repeatpad`
    // below, whichever path constructed it.
    config.validate()?;
    // A typed error, not a `debug_assert!`: in a release build the
    // three-way `.zip()` below (chunks / clips / row_ids) silently truncates
    // to the SHORTEST of the three, so a short `row_ids` would leave every
    // clip past its length UNWRITTEN in `flat` (still its zero-initialized
    // placeholder) rather than failing.
    if row_ids.len() != clips.len() {
        return Err(JammiError::Inference(format!(
            "preprocess_clap_fusion: row_ids has {} entries, expected one per clip ({})",
            row_ids.len(),
            clips.len()
        )));
    }
    // A clip with zero raw samples, or one so short that resampling to
    // `config.sample_rate` rounds it to zero samples, feeds `repeatpad`'s
    // `max_length / len` with `len == 0` — an integer-division-by-zero panic.
    // Refuse both causes here, in a SEQUENTIAL pass over every clip, before
    // the parallel per-clip stage below ever dispatches a closure over them
    // (this is a check before the parallel stage, not inside it: the
    // `par_chunks_mut` writer and `clap_fusion_row` stay untouched).
    for (clip, &row) in clips.iter().zip(row_ids.iter()) {
        if clip.samples.is_empty() {
            return Err(JammiError::Inference(format!(
                "CLAP fusion row {row}: clip has zero samples"
            )));
        }
        // Named separately from the "rounds to zero samples" refusal below:
        // `clip.sample_rate == 0` makes `resampled_len`'s ratio `to_rate /
        // 0.0`, which (pre-fix) rounds to `usize::MAX` rather than `0` and
        // slips past that guard, then feeds `Vec::with_capacity(usize::MAX)`
        // in `resample_linear`. A decoded clip's own decoder already refuses
        // `sample_rate == 0`, but this is a defense-in-depth check for any
        // other `DecodedAudio` producer (including this module's own tests).
        if clip.sample_rate == 0 {
            return Err(JammiError::Inference(format!(
                "CLAP fusion row {row}: clip has sample_rate 0"
            )));
        }
        if resampled_len(clip.samples.len(), clip.sample_rate, config.sample_rate) == 0 {
            return Err(JammiError::Inference(format!(
                "CLAP fusion row {row}: resampling {} sample(s) from {} Hz to {} Hz rounds to \
                 zero samples",
                clip.samples.len(),
                clip.sample_rate,
                config.sample_rate
            )));
        }
    }

    let filters = mel_filterbank_hz(config);
    let window = hann_periodic(config.fft_window_size);
    let time = config.chunk_frames();
    let per_clip = 4 * time * config.n_mels;

    // Preallocate the whole batch's flat buffer, then write each clip's
    // disjoint, fixed-stride `per_clip` chunk in PARALLEL across whichever
    // rayon pool this call runs under (`par_chunks_mut` zipped with the
    // clips — one task per clip, no thread-count knob). `filters` and
    // `window` are computed once above and shared read-only by every task.
    let mut flat = vec![0f32; clips.len() * per_clip];
    let results: Vec<Result<()>> = flat
        .par_chunks_mut(per_clip)
        .zip(clips.par_iter())
        .zip(row_ids.par_iter())
        .map(|((chunk, clip), &row)| {
            let row_values = clap_fusion_row(row, clip, config, &filters, &window, chunk.len())?;
            chunk.copy_from_slice(&row_values);
            Ok(())
        })
        .collect();
    crate::inference::lowest_index_result(results)?;

    let tensor = Tensor::from_vec(flat, (clips.len(), 4, time, config.n_mels), device)
        .map_err(|e| JammiError::Inference(format!("Failed to create audio tensor: {e}")))?;
    // Always-fusion: every clip is marked `is_longer=true` so the tower runs the
    // AFF fusion path and reproduces HF's canonical `get_audio_features` vector
    // (see the policy note above). The flag is constant, not data-dependent.
    Ok((tensor, vec![true; clips.len()]))
}

/// Compute one clip's resampled CLAP-fusion feature row and verify its length
/// against `expected_len` — the caller's preallocated, fixed-stride chunk —
/// before handing it back to be copied in. This is the release-mode per-item
/// length check the parallel writer in [`preprocess_clap_fusion`] depends on
/// (a plain `if`, not `debug_assert!`, so it still runs in release builds):
/// `clap_fusion_features`'s output length is always `expected_len` under a
/// consistent config in practice, but a future change to its branch
/// arithmetic must fail loudly here rather than silently
/// truncate/short-`copy_from_slice`/misalign another clip's chunk.
fn clap_fusion_row(
    row_index: usize,
    clip: &DecodedAudio,
    config: &ClapFrontendConfig,
    filters: &MelFilterbank,
    window: &[f64],
    expected_len: usize,
) -> Result<Vec<f32>> {
    let resampled = resample_linear(&clip.samples, clip.sample_rate, config.sample_rate);
    let feat = clap_fusion_features(&resampled, config, filters, window);
    if feat.features.len() != expected_len {
        return Err(JammiError::Inference(format!(
            "CLAP fusion row {row_index}: produced {} features, expected {expected_len} \
             (time={}, n_mels={})",
            feat.features.len(),
            config.chunk_frames(),
            config.n_mels
        )));
    }
    Ok(feat.features)
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
            // four times — `_get_input_mel`'s `chunk_frames == total_frames`
            // branch (channel construction; the emitted gate is set in
            // `preprocess_clap_fusion`).
            let features = stack4(&mel);
            ClapFusionFeatures {
                features,
                time: chunk,
            }
        } else {
            let features = random_mel_fusion(&mel, total, chunk, n_mels);
            ClapFusionFeatures {
                features,
                time: chunk,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time `Send` assertion (K4/family-J precedent:
    /// `jammi-kernels/src/ops/saved.rs`'s `saved_is_send_and_sync`): the types
    /// that cross the parallel decode/preprocess boundary in
    /// [`decode_audio_batch`] and [`preprocess_clap_fusion`] must be `Send`,
    /// or the crate would not compile — this only names the requirement, it
    /// never silently loosens it.
    #[test]
    fn decoded_audio_and_fusion_features_are_send() {
        fn assert_send<T: Send>() {}
        assert_send::<DecodedAudio>();
        assert_send::<ClapFusionFeatures>();
        assert_send::<Result<DecodedAudio>>();
    }

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
    fn mel_filterbank_hz_rows_are_nonnegative_and_nonempty() {
        let config = ClapFrontendConfig {
            n_mels: 8,
            sample_rate: 16_000,
            fft_window_size: 64,
            hop_length: 32,
            frequency_min: 0.0,
            frequency_max: 8_000.0,
            max_length_s: 1,
        };
        let filters = mel_filterbank_hz(&config);
        assert_eq!(filters.weights.len(), 8 * (64 / 2 + 1));
        assert!(filters.weights.iter().all(|&w| w >= 0.0));
        // At least one filter must have positive weight somewhere.
        assert!(filters.weights.iter().any(|&w| w > 0.0));
    }

    /// A tiny fusion front-end config whose fixed window comfortably exceeds a
    /// short clip, so the repeatpad branch runs and the output shape is fixed.
    fn tiny_fusion_config() -> ClapFrontendConfig {
        ClapFrontendConfig {
            n_mels: 16,
            sample_rate: 16_000,
            fft_window_size: 256,
            hop_length: 128,
            frequency_min: 0.0,
            frequency_max: 8_000.0,
            max_length_s: 1,
        }
    }

    #[test]
    fn fusion_produces_fixed_shape_and_distinguishes_pitch() {
        let device = Device::Cpu;
        let low = decode_audio_bytes(&sine_wav(220.0, 16_000, 4000)).unwrap();
        let high = decode_audio_bytes(&sine_wav(3000.0, 16_000, 4000)).unwrap();
        let config = tiny_fusion_config();

        let (tensor, is_longer) = preprocess_clap_fusion(&[low, high], &config, &device).unwrap();
        // [batch, 4, time, n_mels]; short clips are repeatpadded.
        let time = config.chunk_frames();
        assert_eq!(tensor.dims(), &[2, 4, time, config.n_mels]);
        // Always-fusion policy: every clip is marked is_longer=true (jammi's
        // deterministic analogue of ClapFeatureExtractor's promotion) so the
        // tower reproduces HF's canonical get_audio_features embedding.
        assert_eq!(is_longer, vec![true, true]);

        // The two clips differ in pitch → their fusion spectrograms differ.
        let rows = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let half = rows.len() / 2;
        let diff: f32 = rows[..half]
            .iter()
            .zip(&rows[half..])
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1.0,
            "distinct pitches must yield distinct spectrograms"
        );
    }

    #[test]
    fn fusion_empty_batch_errors() {
        assert!(preprocess_clap_fusion(&[], &tiny_fusion_config(), &Device::Cpu).is_err());
    }

    #[test]
    fn fusion_rejects_non_power_of_two_fft() {
        let clip = decode_audio_bytes(&sine_wav(440.0, 16_000, 2000)).unwrap();
        let mut config = tiny_fusion_config();
        config.fft_window_size = 200; // not a power of two
        assert!(preprocess_clap_fusion(&[clip], &config, &Device::Cpu).is_err());
    }

    #[test]
    fn fusion_zero_n_mels_is_a_typed_error_not_a_panic() {
        // `par_chunks_mut` panics on a zero chunk size — `n_mels == 0` makes
        // `per_clip` zero, so this guard at the input edge is load-bearing,
        // not decorative.
        let clip = decode_audio_bytes(&sine_wav(440.0, 16_000, 2000)).unwrap();
        let mut config = tiny_fusion_config();
        config.n_mels = 0;
        let err = preprocess_clap_fusion(&[clip], &config, &Device::Cpu)
            .expect_err("n_mels == 0 must be a typed error, not a panic");
        let msg = err.to_string();
        assert!(
            msg.contains("n_mels") && msg.contains('0'),
            "error must name the field and the offending value: {msg}"
        );
    }

    // -- `ClapFrontendConfig::validate` — per-field domain checks ------------

    #[test]
    fn validate_accepts_the_real_htsat_clap_config() {
        // The actual `cookbook/fixtures/htsat_clap_tiny/preprocessor_config.json`
        // values must never be rejected by the domain check.
        let config = ClapFrontendConfig {
            n_mels: 32,
            sample_rate: 48_000,
            fft_window_size: 1024,
            hop_length: 577,
            frequency_min: 50.0,
            frequency_max: 14_000.0,
            max_length_s: 6,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn validate_rejects_zero_n_mels() {
        let mut config = tiny_fusion_config();
        config.n_mels = 0;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("n_mels"));
    }

    #[test]
    fn validate_rejects_zero_sample_rate() {
        let mut config = tiny_fusion_config();
        config.sample_rate = 0;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("sample_rate"));
    }

    #[test]
    fn validate_rejects_zero_hop_length() {
        // `chunk_frames`'s `nb_max_samples / hop_length` is an
        // integer-division-by-zero panic when `hop_length == 0`.
        let mut config = tiny_fusion_config();
        config.hop_length = 0;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("hop_length"));
    }

    #[test]
    fn validate_rejects_zero_max_length_s() {
        // `max_length_s == 0` collapses `nb_max_samples` to zero without
        // panicking anywhere — a silent wrong-shape output, not a crash.
        let mut config = tiny_fusion_config();
        config.max_length_s = 0;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("max_length_s"));
    }

    #[test]
    fn validate_rejects_non_power_of_two_fft_window_size() {
        let mut config = tiny_fusion_config();
        config.fft_window_size = 200;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("fft_window_size"));
    }

    #[test]
    fn validate_rejects_fft_window_size_of_one() {
        // A power of two (`2^0`) that still breaks `mel_filterbank_hz`'s
        // `n_bins - 1` (== 0) division.
        let mut config = tiny_fusion_config();
        config.fft_window_size = 1;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("fft_window_size"));
    }

    #[test]
    fn validate_rejects_non_finite_frequency_min() {
        let mut config = tiny_fusion_config();
        config.frequency_min = f64::NAN;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("frequency_min"));
    }

    #[test]
    fn validate_rejects_negative_frequency_min() {
        let mut config = tiny_fusion_config();
        config.frequency_min = -1.0;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("frequency_min"));
    }

    #[test]
    fn validate_rejects_frequency_max_at_or_below_frequency_min() {
        let mut config = tiny_fusion_config();
        config.frequency_min = 100.0;
        config.frequency_max = 100.0;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("frequency_max"));
    }

    #[test]
    fn validate_rejects_non_finite_frequency_max() {
        let mut config = tiny_fusion_config();
        config.frequency_max = f64::INFINITY;
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("frequency_max"));
    }

    #[test]
    fn validate_rejects_frequency_max_past_nyquist() {
        let mut config = tiny_fusion_config();
        config.sample_rate = 16_000;
        config.frequency_max = 9_000.0; // > 16_000 / 2
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("Nyquist"));
    }

    #[test]
    fn validate_accepts_frequency_max_exactly_at_nyquist() {
        let mut config = tiny_fusion_config();
        config.sample_rate = 16_000;
        config.frequency_max = 8_000.0; // == 16_000 / 2, the boundary
        assert!(config.validate().is_ok());
    }

    #[test]
    fn preprocess_clap_fusion_rejects_zero_sample_rate_without_panicking() {
        // Pre-fix, `resample_linear`'s ratio-zero output feeds `repeatpad`'s
        // `max_length / len` with `len == 0`, an integer-division-by-zero
        // panic — the config-level `validate()` call inside
        // `preprocess_clap_fusion_indexed` must refuse this before that code
        // ever runs.
        let clip = decode_audio_bytes(&sine_wav(440.0, 16_000, 2000)).unwrap();
        let mut config = tiny_fusion_config();
        config.sample_rate = 0;
        let err = preprocess_clap_fusion(&[clip], &config, &Device::Cpu)
            .expect_err("sample_rate == 0 must be a typed error, not a panic");
        assert!(err.to_string().contains("sample_rate"));
    }

    #[test]
    fn preprocess_clap_fusion_rejects_zero_hop_length_without_panicking() {
        let clip = decode_audio_bytes(&sine_wav(440.0, 16_000, 2000)).unwrap();
        let mut config = tiny_fusion_config();
        config.hop_length = 0;
        let err = preprocess_clap_fusion(&[clip], &config, &Device::Cpu)
            .expect_err("hop_length == 0 must be a typed error, not a panic");
        assert!(err.to_string().contains("hop_length"));
    }

    #[test]
    fn preprocess_clap_fusion_rejects_zero_max_length_s() {
        let clip = decode_audio_bytes(&sine_wav(440.0, 16_000, 2000)).unwrap();
        let mut config = tiny_fusion_config();
        config.max_length_s = 0;
        let err = preprocess_clap_fusion(&[clip], &config, &Device::Cpu)
            .expect_err("max_length_s == 0 must be a typed error, not a silent shape");
        assert!(err.to_string().contains("max_length_s"));
    }

    // -- Empty-clip refusal --------------------------------------------------
    //
    // Both causes below feed `repeatpad`'s `max_length / len` with `len ==
    // 0`, an integer-division-by-zero panic — the fixed config here passes
    // `ClapFrontendConfig::validate()` (unlike the zero-sample_rate case
    // above), so only a per-clip length check catches them.

    #[test]
    fn preprocess_clap_fusion_rejects_a_zero_sample_clip_without_panicking() {
        let clip = DecodedAudio {
            samples: Vec::new(),
            sample_rate: 16_000,
        };
        let config = tiny_fusion_config();
        let err = preprocess_clap_fusion(&[clip], &config, &Device::Cpu)
            .expect_err("a zero-sample clip must be a typed error, not a division-by-zero panic");
        assert!(err.to_string().contains("zero samples"));
    }

    #[test]
    fn preprocess_clap_fusion_rejects_a_clip_that_resamples_to_zero_samples() {
        // 1 sample at 1,000,000 Hz resampled down to 16,000 Hz: ratio 0.016,
        // round(1 * 0.016) == 0.
        let clip = DecodedAudio {
            samples: vec![0.5],
            sample_rate: 1_000_000,
        };
        let config = tiny_fusion_config();
        assert_eq!(config.sample_rate, 16_000);
        let err = preprocess_clap_fusion(&[clip], &config, &Device::Cpu)
            .expect_err("a clip that resamples to zero samples must be a typed error, not a panic");
        assert!(err.to_string().contains("rounds to zero samples"));
    }

    #[test]
    fn preprocess_clap_fusion_indexed_names_the_row_of_a_zero_sample_clip() {
        let good = decode_audio_bytes(&sine_wav(440.0, 16_000, 2000)).unwrap();
        let bad = DecodedAudio {
            samples: Vec::new(),
            sample_rate: 16_000,
        };
        let config = tiny_fusion_config();
        let err = preprocess_clap_fusion(&[good, bad], &config, &Device::Cpu)
            .expect_err("the second clip's zero samples must be refused");
        assert!(err.to_string().contains("row 1"));
    }

    /// A clip carrying `sample_rate == 0` (a
    /// decoder's own decode path already refuses this, but a directly
    /// constructed `DecodedAudio` — as every test here does — is not routed
    /// through that decoder) must be a NAMED typed refusal, never a
    /// division-by-zero-derived `usize::MAX` that then feeds
    /// `Vec::with_capacity` downstream. Verified by temporarily removing the
    /// `clip.sample_rate == 0` check in the sequential pre-check above: this
    /// test goes RED (`resample_linear`'s `Vec::with_capacity(usize::MAX)`
    /// aborts the process rather than returning the typed `Err` asserted
    /// below).
    #[test]
    fn preprocess_clap_fusion_indexed_rejects_a_zero_sample_rate_clip_by_name() {
        let clip = DecodedAudio {
            samples: vec![0.5; 100],
            sample_rate: 0,
        };
        let config = tiny_fusion_config();
        let err = preprocess_clap_fusion(&[clip], &config, &Device::Cpu)
            .expect_err("sample_rate == 0 must be a typed error, not a huge-allocation abort");
        assert!(err.to_string().contains("sample_rate 0"), "{err}");
    }

    /// `resampled_len`'s own domain edge: a zero
    /// `from_rate` must return `0`, never round `to_rate / 0.0 == +inf` up
    /// to `usize::MAX` (a confidently wrong length the empty-clip guard above
    /// would then fail to catch, since `usize::MAX != 0`).
    #[test]
    fn resampled_len_returns_zero_for_a_zero_from_rate() {
        assert_eq!(resampled_len(100, 0, 16_000), 0);
    }

    // -- Media front-end parallelization (#421 follow-on) --------------------

    #[test]
    fn decode_audio_batch_empty_is_ok_empty() {
        // Decode-stage empty is a no-op (K2's "empty batch refused" guard
        // lives at the PREPROCESS stage, which still refuses — see
        // `fusion_empty_batch_errors` above — matching the pre-unit
        // sequential decode loop, which also never rejected zero rows).
        let items: Vec<Vec<u8>> = Vec::new();
        let decoded = decode_audio_batch(&items).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn decode_audio_batch_two_bad_rows_surfaces_the_lowest_index() {
        let good = sine_wav(440.0, 16_000, 2000);
        let items: Vec<Vec<u8>> = vec![
            good.clone(),
            good.clone(),
            b"not audio at all".to_vec(), // row 2: bad
            good.clone(),
            good.clone(),
            b"also not audio".to_vec(), // row 5: bad
            good,
        ];
        let err = decode_audio_batch(&items).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("row 2"),
            "expected the lowest-index (row 2) failure, got: {msg}"
        );
        assert!(
            !msg.contains("row 5"),
            "row 5's failure must not be the one surfaced when row 2 also failed: {msg}"
        );
    }

    /// A caller that COMPACTS its batch (drops null rows) before decoding
    /// must get the ORIGINAL row back in the error, not the position in the
    /// compacted `items` slice.
    #[test]
    fn decode_audio_batch_indexed_reports_the_given_row_id_not_the_position() {
        let good = sine_wav(440.0, 16_000, 2000);
        let items: Vec<Vec<u8>> = vec![good.clone(), b"not audio at all".to_vec(), good];
        let row_ids = vec![10usize, 42, 99];

        let err = decode_audio_batch_indexed(&row_ids, &items).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("row 42"),
            "expected the caller's row id (42), got: {msg}"
        );
        assert!(
            !msg.contains("row 1:"),
            "must never report the compacted position instead of the row id: {msg}"
        );
    }

    #[test]
    fn decode_audio_results_rejects_short_row_ids_instead_of_dropping_tail_items() {
        // `.zip()` silently truncates to the shorter of `items`/`row_ids` in
        // a release build (`debug_assert!` compiles out there), so a short
        // `row_ids` must be a typed error, not zero outcomes for the
        // uncovered tail items.
        let good = sine_wav(440.0, 16_000, 2000);
        let items: Vec<Vec<u8>> = vec![good.clone(), good];
        let row_ids = vec![0usize]; // one entry short of `items.len()`
        let err = decode_audio_batch_indexed(&row_ids, &items)
            .expect_err("a short row_ids must be a typed error, not silent truncation");
        assert!(err.to_string().contains("row_ids"));
    }

    #[test]
    fn decode_audio_batch_per_row_indexed_rejects_short_row_ids() {
        let good = sine_wav(440.0, 16_000, 2000);
        let items: Vec<Vec<u8>> = vec![good.clone(), good];
        let row_ids = vec![0usize];
        let err = decode_audio_batch_per_row_indexed(&row_ids, &items)
            .expect_err("a short row_ids must be a typed error, not silent truncation");
        assert!(err.to_string().contains("row_ids"));
    }

    #[test]
    fn preprocess_clap_fusion_indexed_rejects_short_row_ids_instead_of_leaving_a_tail_unwritten() {
        // The three-way `.zip()` (chunks / clips / row_ids) would otherwise
        // leave every clip past `row_ids.len()` as its zero-initialized
        // placeholder in `flat`, silently, rather than failing.
        let bytes = sine_wav(220.0, 16_000, 2000);
        let clips = vec![
            decode_audio_bytes(&bytes).unwrap(),
            decode_audio_bytes(&bytes).unwrap(),
        ];
        let row_ids = vec![0usize];
        let err =
            preprocess_clap_fusion_indexed(&row_ids, &clips, &tiny_fusion_config(), &Device::Cpu)
                .expect_err(
                    "a short row_ids must be a typed error, not a silently truncated tensor",
                );
        assert!(err.to_string().contains("row_ids"));
    }

    #[test]
    fn clap_fusion_row_wrong_expected_len_is_a_typed_error_not_a_panic() {
        // The release-mode length check `clap_fusion_row` performs before
        // handing its row back to `copy_from_slice` — exercised directly
        // (not via `debug_assert!`, which would compile out in release) by
        // asking for a length the real feature row can never have.
        let config = tiny_fusion_config();
        let clip = decode_audio_bytes(&sine_wav(220.0, 16_000, 2000)).unwrap();
        let filters = mel_filterbank_hz(&config);
        let window = hann_periodic(config.fft_window_size);
        let real_len = 4 * config.chunk_frames() * config.n_mels;

        let err = clap_fusion_row(3, &clip, &config, &filters, &window, real_len + 1)
            .expect_err("a deliberately wrong expected_len must be a typed error");
        let msg = err.to_string();
        assert!(msg.contains("row 3"), "error must be row-indexed: {msg}");
    }

    /// One synthetic clip of `kind` at batch position `idx`, at the fixed
    /// sample rate `config.sample_rate` (so `preprocess_clap_fusion` never
    /// resamples — isolating the parallel-write mechanism under test from
    /// `resample_linear`'s own arithmetic). `kind` selects which of
    /// `clap_fusion_features`'s three length-determined branches the clip
    /// hits: repeatpad (short), the generic fusion crop (long), and the
    /// `total == chunk` fusion corner case (long, at the exact boundary
    /// length derived from `config`).
    #[derive(Clone, Copy)]
    enum ClipKind {
        Repeatpad,
        FusionGeneric,
        FusionCornerTotalEqChunk,
    }

    fn synthetic_clip(kind: ClipKind, idx: usize, config: &ClapFrontendConfig) -> DecodedAudio {
        let max_length = config.nb_max_samples();
        let n = match kind {
            ClipKind::Repeatpad => max_length / 4,
            ClipKind::FusionGeneric => max_length + max_length / 4,
            // Derived so `linear_mel`'s frame count lands exactly on `chunk`:
            // num_frames = 1 + len / hop (floor); solving for the smallest
            // `len > max_length` that keeps `len / hop == chunk - 1`.
            ClipKind::FusionCornerTotalEqChunk => {
                (config.chunk_frames() - 1) * config.hop_length + config.hop_length - 1
            }
        };
        let freq = 110.0 * (1.0 + idx as f32 * 0.37);
        let sample_rate = config.sample_rate;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                0.5 * (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect();
        DecodedAudio {
            samples,
            sample_rate,
        }
    }

    fn batch_24(config: &ClapFrontendConfig) -> Vec<DecodedAudio> {
        (0..24)
            .map(|i| {
                let kind = match i % 3 {
                    0 => ClipKind::Repeatpad,
                    1 => ClipKind::FusionGeneric,
                    _ => ClipKind::FusionCornerTotalEqChunk,
                };
                synthetic_clip(kind, i, config)
            })
            .collect()
    }

    /// [`batch_24`] plus one clip decoded at HALF the config's target sample
    /// rate — every other synthetic clip is generated AT `config.sample_rate`
    /// specifically to isolate the parallel-write mechanism from
    /// `resample_linear`'s own arithmetic (see `synthetic_clip`'s doc), which
    /// means the K4 bit-identity oracle below would otherwise never exercise
    /// per-clip resampling at all. Appending one clip that genuinely needs
    /// resampling closes that gap without disturbing the other 24 clips'
    /// existing branch coverage.
    fn batch_25_with_a_resampled_clip(config: &ClapFrontendConfig) -> Vec<DecodedAudio> {
        let mut clips = batch_24(config);
        let half_rate = config.sample_rate / 2;
        let n = config.nb_max_samples() / 4; // repeatpad branch at the native rate
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / half_rate as f32;
                0.5 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
            })
            .collect();
        clips.push(DecodedAudio {
            samples,
            sample_rate: half_rate,
        });
        clips
    }

    /// Run [`preprocess_clap_fusion`] under a rayon pool of exactly
    /// `pool_size` threads and return the flattened `f32` output plus the
    /// `is_longer` flags.
    fn run_at_pool_size(
        pool_size: usize,
        clips: &[DecodedAudio],
        config: &ClapFrontendConfig,
    ) -> (Vec<f32>, Vec<bool>) {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(pool_size)
            .build()
            .expect("build a fixed-size rayon pool");
        pool.install(|| {
            let (tensor, is_longer) = preprocess_clap_fusion(clips, config, &Device::Cpu).unwrap();
            let flat = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            (flat, is_longer)
        })
    }

    /// The PRE-UNIT sequential implementation, transcribed verbatim from
    /// `c1b0b0ba`'s `preprocess_clap_fusion` (before the `par_chunks_mut`
    /// rewrite) — kept ONLY as a test oracle, INDEPENDENT of the shipped
    /// parallel code path, so the bit-identity assertions below compare the
    /// new code against a second implementation rather than against itself
    /// at a different pool size (a pool-1-vs-pool-k comparison alone cannot
    /// catch a bug the SAME code makes at every pool size). Reuses
    /// `mel_filterbank_hz`, `hann_periodic`, `resample_linear` and
    /// `clap_fusion_features`, all unchanged by this unit.
    fn preprocess_clap_fusion_reference_sequential(
        clips: &[DecodedAudio],
        config: &ClapFrontendConfig,
        device: &Device,
    ) -> (Tensor, Vec<bool>) {
        let filters = mel_filterbank_hz(config);
        let window = hann_periodic(config.fft_window_size);
        let time = config.chunk_frames();
        let per_clip = 4 * time * config.n_mels;
        let mut flat = Vec::with_capacity(clips.len() * per_clip);
        for clip in clips {
            let resampled = resample_linear(&clip.samples, clip.sample_rate, config.sample_rate);
            let feat = clap_fusion_features(&resampled, config, &filters, &window);
            debug_assert_eq!(feat.time, time);
            debug_assert_eq!(feat.features.len(), per_clip);
            flat.extend_from_slice(&feat.features);
        }
        let tensor = Tensor::from_vec(flat, (clips.len(), 4, time, config.n_mels), device).unwrap();
        (tensor, vec![true; clips.len()])
    }

    /// Oracle 1 (K4): element-wise bit identity between the PRE-UNIT
    /// sequential reference above and the shipped parallel code at pool sizes
    /// {1, 5, 7, 24} (non-dividing counts included), on a batch that hits all
    /// three of `clap_fusion_features`'s length-determined branches
    /// (repeatpad; fusion crop; fusion crop's `total == chunk` corner case)
    /// PLUS one clip that genuinely needs resampling.
    #[test]
    fn clap_fusion_parallel_matches_sequential_bit_identical_across_pool_sizes() {
        let config = tiny_fusion_config();
        let clips = batch_25_with_a_resampled_clip(&config);

        let (reference_tensor, reference_is_longer) =
            preprocess_clap_fusion_reference_sequential(&clips, &config, &Device::Cpu);
        let reference = reference_tensor
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(reference_is_longer, vec![true; clips.len()]);

        for &k in &[1usize, 5, 7, 24] {
            let (got, got_is_longer) = run_at_pool_size(k, &clips, &config);
            assert_eq!(
                got_is_longer, reference_is_longer,
                "is_longer flags must match the pre-unit reference at pool size {k}"
            );
            assert_eq!(
                got.len(),
                reference.len(),
                "output length must match the pre-unit reference at pool size {k}"
            );
            for (idx, (&a, &b)) in got.iter().zip(reference.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "element {idx} differs from the pre-unit sequential reference at pool \
                     size {k}: {a} (bits {:x}) vs {b} (bits {:x})",
                    a.to_bits(),
                    b.to_bits()
                );
            }
        }
    }

    /// `row_ids` only changes what number a row is called in an error
    /// message; it must never change the numeric output. Uses row ids that
    /// are neither `0..n` nor monotonic-by-one.
    #[test]
    fn preprocess_clap_fusion_indexed_row_ids_do_not_affect_numeric_output() {
        let config = tiny_fusion_config();
        let clips = batch_24(&config);

        let (sequential_tensor, sequential_is_longer) =
            preprocess_clap_fusion(&clips, &config, &Device::Cpu).unwrap();
        let sequential = sequential_tensor
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let row_ids: Vec<usize> = (0..clips.len()).map(|i| i * 7 + 1000).collect();
        let (indexed_tensor, indexed_is_longer) =
            preprocess_clap_fusion_indexed(&row_ids, &clips, &config, &Device::Cpu).unwrap();
        let indexed = indexed_tensor
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(sequential_is_longer, indexed_is_longer);
        assert_eq!(sequential.len(), indexed.len());
        for (idx, (&a, &b)) in sequential.iter().zip(indexed.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "element {idx} changed when row_ids were non-sequential"
            );
        }
    }

    /// Oracle 2 (determinism): two independent parallel runs at the same pool
    /// size reproduce the identical output — no unseeded RNG, no
    /// timing-dependent behavior in the front end.
    #[test]
    fn clap_fusion_parallel_is_deterministic_across_two_runs() {
        let config = tiny_fusion_config();
        let clips = batch_24(&config);

        let (first, first_is_longer) = run_at_pool_size(7, &clips, &config);
        let (second, second_is_longer) = run_at_pool_size(7, &clips, &config);

        assert_eq!(first_is_longer, second_is_longer);
        assert_eq!(first.len(), second.len());
        for (idx, (&a, &b)) in first.iter().zip(second.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "element {idx} differs between two runs at the same pool size"
            );
        }
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
            let want_time = want_t.dim(1).unwrap();
            assert_eq!(
                tensor.dims(),
                &[1, 4, want_time, config.n_mels],
                "{tag}: shape mismatch"
            );
            // Always-fusion policy: the emitted gate is constant `true` for both
            // the short and long clip (jammi's deterministic analogue of
            // ClapFeatureExtractor's promotion).
            assert_eq!(is_longer, vec![true], "{tag}: is_longer must be true");

            let got = tensor_f64(&tensor);

            // The length-based channel construction is still distinct even though
            // both flag true: a short (repeatpad) clip stacks one mel into 4
            // identical channels, while a long clip's channels 1-3 are crops that
            // differ from channel 0's downsample.
            let chan = want_time * config.n_mels;
            let ch0 = &got[..chan];
            let ch1 = &got[chan..2 * chan];
            let channels_identical = ch0 == ch1;
            if tag == "short" {
                assert!(
                    channels_identical,
                    "short clip's 4 channels must be the same repeatpad mel"
                );
            } else {
                assert!(
                    !channels_identical,
                    "long clip's fusion crops must differ across channels"
                );
            }
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
