#!/usr/bin/env python3
"""Generate the CLAP audio front-end oracle: deterministic waveforms run through
the real HuggingFace `ClapFeatureExtractor` (laion/clap-htsat-fused geometry,
truncation="fusion"), with both the final 4-channel dB `input_features` and the
linear-power mel BEFORE the dB nonlinearity committed as goldens.

The Rust front-end (`crates/jammi-ai/src/inference/audio_preprocess.rs`) is
parity-tested against these goldens hermetically — CI needs no torch.

Two waveforms pin both `_get_input_mel` branches that a fusion-truncation
extractor takes, with every source of randomness removed:

  short  (len < nb_max_samples): repeatpad -> single mel -> stacked x4,
         is_longer=False. Fully deterministic; no random anywhere.
  long   (len  > nb_max_samples): `_random_mel_fusion`. The waveform length is
         chosen so `total_frames - chunk_frames + 1 == 3`, hence
         `np.array_split(range(3), 3) == [[0],[1],[2]]` and every
         `np.random.choice` collapses to its single element (front=0, middle=1,
         back=2) INDEPENDENT of the RNG. is_longer=True. We additionally seed
         numpy so the run is bit-reproducible, but the crop indices are forced
         by geometry, not by the seed.

The final `__call__`-level "if no clip is_longer, randomly mark one" branch is
sidestepped: we call the per-waveform `_get_input_mel` directly (which is what
`__call__` maps over) so no batch-level RNG is touched. The dB golden produced
this way is identical to `fe(raw_speech=wave, ...)["input_features"]` for these
two pinned waveforms.

The 4-channel fusion packing (crop + bilinear downsample) happens AFTER the dB
nonlinearity, on the dB mel — so the pre-dB *linear* space is faithful only for
the UNPACKED mel (the STFT + filterbank output before crop/stack). Each waveform
therefore yields:
  <tag>_mel_linear      the pre-dB linear-power mel, f64, UNPACKED [T_full,n_mels]
                        — the faithful parity space for STFT + filterbank.
  <tag>_input_features  the final dB packed `input_features`, f32, [4,T,n_mels]
                        — validates the dB nonlinearity AND the crop/bilinear
                        packing that operates in dB space.
Plus the raw waveform `<tag>_waveform` (f64) so the Rust test drives the exact
same samples through its own pipeline.

Environment:  pip install "transformers==4.57.6" "torch" "numpy" "safetensors"
Regenerate:   python tests/fixtures/generate_clap_frontend.py
"""

import json
import os

import numpy as np
from safetensors.numpy import save_file
from transformers.audio_utils import spectrogram, window_function
from transformers.models.clap.feature_extraction_clap import ClapFeatureExtractor

# tests/fixtures/generate_clap_frontend.py -> repo root is two parents up.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO_ROOT, "cookbook", "fixtures", "htsat_clap_frontend")

# laion/clap-htsat-fused feature-extractor geometry.
PARAMS = dict(
    feature_size=64,
    sampling_rate=48_000,
    hop_length=480,
    fft_window_size=1024,
    frequency_min=50.0,
    frequency_max=14_000.0,
    truncation="fusion",
    padding="repeatpad",
    max_length_s=10,
)


def linear_mel(fe: ClapFeatureExtractor, waveform: np.ndarray) -> np.ndarray:
    """The pre-dB linear-power mel, transposed to [time, n_mels] — exactly what
    `_np_extract_fbank_features` produces minus the final dB nonlinearity."""
    lin = spectrogram(
        waveform,
        window_function(fe.fft_window_size, "hann"),
        frame_length=fe.fft_window_size,
        hop_length=fe.hop_length,
        power=2.0,
        mel_filters=fe.mel_filters,
        log_mel=None,  # stop BEFORE the dB nonlinearity
    )
    return lin.T  # [time, n_mels]


def make_short(fe: ClapFeatureExtractor):
    """A < nb_max_samples waveform: 5s tone+noise, repeatpad path. Returns the
    waveform and the UNPACKED pre-dB linear mel of the repeatpadded waveform."""
    sr = fe.sampling_rate
    n = sr * 5  # 240000 < 480000
    t = np.arange(n) / sr
    rng = np.random.default_rng(0)
    wave = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.standard_normal(n)).astype(np.float64)
    max_length = fe.nb_max_samples
    # repeatpad, mirroring _get_input_mel's shorter-than-max branch.
    n_repeat = int(max_length / len(wave))
    padded = np.tile(wave, n_repeat)
    padded = np.pad(padded, (0, max_length - padded.shape[0]), mode="constant", constant_values=0)
    return wave, linear_mel(fe, padded), False


def make_long(fe: ClapFeatureExtractor):
    """A > nb_max_samples waveform whose length forces deterministic crops.
    Returns the waveform and the UNPACKED pre-dB linear mel over the whole audio
    (the fusion branch computes the mel on the entire waveform, no padding)."""
    sr = fe.sampling_rate
    hop = fe.hop_length
    max_length = fe.nb_max_samples
    chunk_frames = max_length // hop + 1
    # Solve for waveform length L s.t. total_frames == chunk_frames + 2, i.e.
    # total_frames - chunk_frames + 1 == 3 -> array_split(range(3),3)=[[0],[1],[2]].
    # total_frames = 1 + (L + fft - fft)//hop = 1 + L//hop.
    target_total = chunk_frames + 2
    n = (target_total - 1) * hop  # smallest L giving that total_frames
    assert n > max_length, f"long waveform {n} must exceed max_length {max_length}"
    t = np.arange(n) / sr
    rng = np.random.default_rng(1)
    wave = (0.3 * np.sin(2 * np.pi * 660 * t) + 0.05 * rng.standard_normal(n)).astype(np.float64)
    lin = linear_mel(fe, wave)
    assert lin.shape[0] == target_total, f"total_frames {lin.shape[0]} != {target_total}"
    return wave, lin, True


def db_input_features(fe: ClapFeatureExtractor, wave: np.ndarray):
    """The real extractor's final 4-channel dB input_features for one waveform,
    via the per-waveform `_get_input_mel` (no batch-level RNG)."""
    mel, longer = fe._get_input_mel(  # noqa: SLF001 — oracle reproduces the exact op
        np.asarray(wave, dtype=np.float64), fe.nb_max_samples, fe.truncation, fe.padding
    )
    return np.asarray(mel, dtype=np.float32), bool(longer)


def main():
    os.makedirs(OUT, exist_ok=True)
    fe = ClapFeatureExtractor(**PARAMS)

    goldens = {}
    meta = {}
    for tag, maker in (("short", make_short), ("long", make_long)):
        wave, mel_linear, expect_longer = maker(fe)
        db_feats, longer = db_input_features(fe, wave)
        assert longer == expect_longer, f"{tag}: is_longer {longer} != {expect_longer}"
        # Cross-check the unpacked dB mel == 10*log10(clip(linear,1e-10)) within
        # fp32 — the dB nonlinearity the front-end applies before packing.
        unpacked_db = (10.0 * np.log10(np.clip(mel_linear, 1e-10, None))).astype(np.float32)
        # For the short (repeatpad) branch every channel IS the unpacked dB mel,
        # so the packed dB golden's channel 0 must equal it exactly. For the long
        # branch channel 1 (front crop, idx 0) is the unpacked dB mel's first
        # chunk_frames rows. Both checks tie the packed golden back to the linear
        # mel, proving the linear golden is the faithful pre-dB of the dB golden.
        if not longer:
            err = float(np.max(np.abs(db_feats[0] - unpacked_db)))
        else:
            cf = db_feats.shape[1]
            err = float(np.max(np.abs(db_feats[1] - unpacked_db[0:cf])))
        assert err < 1e-3, f"{tag}: packed dB vs 10log10(linear) mismatch {err}"

        goldens[f"{tag}_waveform"] = wave.astype(np.float64)
        goldens[f"{tag}_mel_linear"] = mel_linear.astype(np.float64)
        goldens[f"{tag}_input_features"] = db_feats.astype(np.float32)
        meta[tag] = {
            "is_longer": longer,
            "mel_linear_shape": list(mel_linear.shape),
            "input_features_shape": list(db_feats.shape),
            "n_samples": int(wave.shape[0]),
            "packed_db_vs_log_linear_max_abs": err,
        }

    save_file(goldens, os.path.join(OUT, "goldens.safetensors"))
    manifest = {
        "params": PARAMS,
        "clips": meta,
        "tensors": {
            name: {"shape": list(t.shape), "dtype": str(t.dtype)}
            for name, t in goldens.items()
        },
    }
    with open(os.path.join(OUT, "golden_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[frontend] {OUT}")
    total = 0
    for f in sorted(os.listdir(OUT)):
        size = os.path.getsize(os.path.join(OUT, f))
        total += size
        print(f"  {f}: {size:,} bytes")
    print(f"  total: {total:,} bytes")
    for tag, m in meta.items():
        print(f"  {tag}: {m}")


if __name__ == "__main__":
    main()
