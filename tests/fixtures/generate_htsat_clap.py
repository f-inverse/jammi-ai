#!/usr/bin/env python3
"""Generate the HTSAT-Swin CLAP audio-tower oracle: a tiny real-HTSAT fixture
plus per-boundary golden activations, for hermetic numerical parity testing.

This builds a real HuggingFace `transformers` `ClapAudioModelWithProjection` so
the committed `model.safetensors` carries the exact `audio_model.audio_encoder.*`
+ `audio_projection.*` key layout of `laion/clap-htsat-fused`. The Rust HTSAT
tower is parity-tested against the golden activations dumped here at every
forward boundary, so any divergence is localized to the unit that produced it.

Two artifact sets are written under cookbook/fixtures/:

  htsat_clap_tiny/   (committed; hermetic — CI needs no torch)
    config.json            tiny ClapAudioConfig (real `model_type`/key layout)
    model.safetensors      tiny real-HTSAT weights (~1.6 MB)
    pinned_input.safetensors   the deterministic `input_features` the tower is
                               tested on, independent of the audio front-end
    goldens.safetensors    per-boundary activations (mel_in .. projected_*)
    golden_manifest.json   name -> shape -> dtype index of the goldens

  htsat_clap_real/   (written only by `--real`; small)
    goldens.safetensors    pinned `input_features` + `is_longer`, the real
                           checkpoint's L2-normalized `embedding`, and the
                           int16 `waveform_i16` that produced them, for the
                           live-gated numerical-acceptance + e2e tests.
    golden_manifest.json   name -> shape -> dtype index of the goldens
    model_id.txt           the checkpoint id the golden was dumped from

Scope: the fixture reproduces the REAL forward of `laion/clap-htsat-fused`. Its
default feature extractor (`truncation="fusion"`) returns `is_longer=True` for
every clip, and a clip's mel frame count is below `spec_size*freq_ratio`, so a
normal forward ALWAYS runs (a) bicubic time-interpolation in `reshape_mel2img`
and (b) the AFF fusion path (`mel_conv2d` + `fusion_model`). The pinned input is
therefore `is_longer=True` with 4 distinct channels and a time dim that triggers
interpolation, so the goldens cover both. The Rust tower ports both paths; only
`*.relative_position_index` (recomputed) and `*.num_batches_tracked` are excluded
from the weight-coverage test.

Environment (run on Linux; x86_64 macOS caps torch too low for current ClapModel):

    pip install "torch==2.8.0" "transformers==4.57.6" "safetensors"

Regenerate (deterministic, fixed seed):

    python tests/fixtures/generate_htsat_clap.py            # tiny fixture (hermetic)
    python tests/fixtures/generate_htsat_clap.py --real     # + real-model golden (downloads laion/clap-htsat-fused)
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import ClapAudioConfig, ClapAudioModelWithProjection

# tests/fixtures/generate_htsat_clap.py -> repo root is two parents up.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURES = os.path.join(REPO_ROOT, "cookbook", "fixtures")
TINY_OUT = os.path.join(FIXTURES, "htsat_clap_tiny")
REAL_OUT = os.path.join(FIXTURES, "htsat_clap_real")

REAL_MODEL_ID = "laion/clap-htsat-fused"

# Tiny HTSAT geometry. Chosen so the forward exercises every codepath the Rust
# port must reproduce: a patch-merge downsample between all 4 stages, and a
# genuine shifted-window attention mask in stages 0-2 (window < grid). Stage 3's
# grid equals the window, so its shift degenerates to 0 — correct Swin behavior,
# locked by its own golden. `fusion_type=None` matches the real checkpoint and
# keeps `scale_factor=1` (single-channel patch-embed conv).
TINY_CONFIG = dict(
    patch_embeds_hidden_size=16,
    depths=[2, 2, 2, 2],
    num_attention_heads=[2, 2, 4, 4],
    window_size=4,
    spec_size=128,
    patch_size=4,
    patch_stride=[4, 4],
    num_mel_bins=32,
    hidden_size=128,
    projection_dim=8,
    projection_hidden_act="relu",
    mlp_ratio=2.0,
    flatten_patch_embeds=True,
    enable_patch_layer_norm=True,
    enable_fusion=True,
    fusion_type=None,
)

BATCH = 2
# enable_fusion=True forward takes a 4-channel `input_features` [B, 4, T, mel].
# freq_ratio = spec_size // num_mel_bins = 4, so spec_width = spec_size*freq_ratio
# = 512. T = 500 < 512 triggers the bicubic time-interpolation in reshape_mel2img
# (mel = num_mel_bins = 32 = spec_height, so freq is NOT interpolated). The 4
# channels are distinct (randn, not repeated) so the AFF fusion path on channels
# 1:4 is genuinely exercised.
PINNED_SHAPE = (BATCH, 4, 500, 32)
SPEC_WIDTH = 512  # time axis is bicubic-interpolated up to this (spec_size*freq_ratio)


def build_tiny_model():
    config = ClapAudioConfig(**TINY_CONFIG)
    # Seed BEFORE construction so the randomly-initialized weights — and hence
    # the committed model.safetensors and every golden — are reproducible.
    torch.manual_seed(0)
    # .eval() is REQUIRED: the tower's single-channel reproduction of the
    # reference channel-0 path is exact only when batch_norm uses running
    # stats (eval), not per-batch stats.
    model = ClapAudioModelWithProjection(config).eval()
    return model, config


def pinned_input():
    g = torch.Generator().manual_seed(0)
    feats = torch.randn(*PINNED_SHAPE, generator=g, dtype=torch.float32)
    # is_longer=True for the whole batch -> is_longer_idx = [0,1] -> the AFF
    # fusion path (mel_conv2d + fusion_model) executes on every row, matching the
    # real fused checkpoint's default truncation="fusion" feature extractor.
    is_longer = torch.ones(BATCH, 1, dtype=torch.bool)
    return feats, is_longer


def _as_tensor(value):
    """Normalize a hooked module output to a detached fp32 CPU tensor."""
    if isinstance(value, tuple):
        value = value[0]
    return value.detach().to(torch.float32).cpu().contiguous()


def collect_goldens(model, feats, is_longer):
    """Run a forward with hooks at every boundary the Rust port must match."""
    goldens = {}
    handles = []

    encoder = model.audio_model.audio_encoder

    def save_out(name):
        def hook(_module, _inp, out):
            goldens[name] = _as_tensor(out)

        return hook

    def save_pre_input(name):
        def hook(_module, inp):
            goldens[name] = _as_tensor(inp[0])

        return hook

    def save_encoder_last_hidden(name):
        def hook(_module, _inp, out):
            # ClapAudioEncoder returns last_hidden_state as the 4D [B,C,1,F] grid.
            lhs = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            goldens[name] = _as_tensor(lhs)

        return hook

    # mel_in: the raw pinned input_features (channel 0 is the global view used).
    goldens["mel_in"] = _as_tensor(feats)

    handles.append(encoder.batch_norm.register_forward_hook(save_out("post_batch_norm")))
    # reshape_mel2img is a method, not a module: its output is patch_embed's input.
    handles.append(encoder.patch_embed.register_forward_pre_hook(save_pre_input("post_reshape_mel2img")))
    handles.append(encoder.patch_embed.register_forward_hook(save_out("patch_embed_out")))
    # AFF fusion path (executes because is_longer=True): the local mel conv and
    # the attention-feature-fusion block.
    handles.append(encoder.patch_embed.mel_conv2d.register_forward_hook(save_out("mel_conv2d_out")))
    handles.append(encoder.patch_embed.fusion_model.register_forward_hook(save_out("fusion_model_out")))

    for s, stage in enumerate(encoder.layers):
        for b, block in enumerate(stage.blocks):
            handles.append(block.register_forward_hook(save_out(f"stage{s}.block{b}_out")))
        if stage.downsample is not None:
            handles.append(stage.downsample.register_forward_hook(save_out(f"stage{s}.downsample_out")))

    handles.append(encoder.norm.register_forward_hook(save_out("final_norm_out")))
    handles.append(encoder.register_forward_hook(save_encoder_last_hidden("pre_pool")))
    # pooler_out is the pooled vector fed into the projection MLP.
    handles.append(model.audio_projection.register_forward_pre_hook(save_pre_input("pooler_out")))

    with torch.inference_mode():
        out = model(input_features=feats, is_longer=is_longer)

    for h in handles:
        h.remove()

    # is_longer=False patch-embed boundary: the SAME post_reshape_mel2img input
    # the main forward fed patch_embed, but with an empty is_longer_idx so the
    # fusion (mel_conv2d + AFF) path is skipped and every sample is the global
    # patch-conv alone. This pins the short-clip branch the Rust tower selects
    # per sample. The existing is_longer=True goldens above are untouched.
    with torch.inference_mode():
        empty_idx = torch.where(torch.zeros(BATCH, dtype=torch.bool))[0]
        goldens["patch_embed_out_global_only"] = _as_tensor(
            encoder.patch_embed(goldens["post_reshape_mel2img"], empty_idx)
        )

    # post_interpolation: the bicubic time-interpolation reshape_mel2img performs
    # internally (1001->spec_width in the real model; 500->512 here). It equals
    # F.interpolate of reshape_mel2img's input = the batch_norm output transposed
    # back to [B,4,T,mel]. Captured here as the exact op the Rust port must match.
    reshape_input = goldens["post_batch_norm"].transpose(1, 3)  # [2,32,500,4] -> [2,4,500,32]
    goldens["post_interpolation"] = _as_tensor(
        F.interpolate(
            reshape_input,
            size=(SPEC_WIDTH, TINY_CONFIG["num_mel_bins"]),
            mode="bicubic",
            align_corners=True,
        )
    )

    # projected: audio_projection MLP output (linear1 -> relu -> linear2),
    # UNNORMALIZED, and its explicit L2-normalized form (the jammi tower
    # contract target — ClapAudioModelWithProjection does not normalize, and
    # get_audio_features is absent on this class).
    embeds = out.audio_embeds.detach().to(torch.float32).cpu().contiguous()
    goldens["projected_unnormalized"] = embeds
    goldens["projected_normalized"] = F.normalize(embeds, dim=-1)
    return goldens


def write_manifest(goldens, path):
    manifest = {
        name: {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}
        for name, t in goldens.items()
    }
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def generate_tiny():
    os.makedirs(TINY_OUT, exist_ok=True)
    model, _config = build_tiny_model()
    model.save_pretrained(TINY_OUT)  # config.json + model.safetensors

    feats, is_longer = pinned_input()
    save_file({"input_features": feats.contiguous()}, os.path.join(TINY_OUT, "pinned_input.safetensors"))

    goldens = collect_goldens(model, feats, is_longer)
    save_file(goldens, os.path.join(TINY_OUT, "goldens.safetensors"))
    write_manifest(goldens, os.path.join(TINY_OUT, "golden_manifest.json"))

    print(f"[tiny] {TINY_OUT}")
    total = 0
    for f in sorted(os.listdir(TINY_OUT)):
        size = os.path.getsize(os.path.join(TINY_OUT, f))
        total += size
        print(f"  {f}: {size:,} bytes")
    print(f"  total: {total:,} bytes")
    print(f"  goldens: {', '.join(goldens)}")


def generate_real(model_id):
    """Commit a SMALL real-model golden (final embedding + the pinned input) for
    the live numerical-acceptance tests, the CANONICAL CLAP audio embedding.

    The golden is the L2-normalized `get_audio_features` embedding of a pinned 5s
    clip, computed from the feature extractor's NATURAL output. The same
    int16-quantized waveform that produces it is committed as `waveform_i16`, so
    the e2e front-end+tower test can rebuild a byte-identical 48 kHz mono WAV
    in-Rust and exercise the production decode → front-end → tower path against
    this committed embedding. The torch path is fed the *dequantized* int16 signal
    (not the raw float) so the committed embedding corresponds to the exact
    samples the Rust WAV carries.

    `is_longer` policy — read carefully. `ClapFeatureExtractor.__call__` with
    `truncation="fusion"` DETERMINISTICALLY marks a single short clip
    `is_longer=True` (its `_get_input_mel` promotes the global-only mel so the AFF
    fusion path runs), so `get_audio_features` returns the FUSION embedding — the
    vector the CLAP ecosystem indexes and searches with. We build the golden from
    that natural output: `model.get_audio_features(**inputs)` with the processor's
    own `is_longer` (True for this clip). jammi's front-end reproduces this by
    deterministically marking every clip `is_longer=True` (its analogue of the
    extractor's promotion), so both the tower-isolation and e2e tests target this
    same canonical, reproducible embedding — not the global-only one (which sits
    ~0.73 cosine off and is ecosystem-incompatible).
    """
    from transformers import ClapModel, ClapProcessor

    os.makedirs(REAL_OUT, exist_ok=True)
    model = ClapModel.from_pretrained(model_id).eval()
    processor = ClapProcessor.from_pretrained(model_id)

    sr = processor.feature_extractor.sampling_rate
    g = torch.Generator().manual_seed(0)
    audio_f32 = (torch.randn(sr * 5, generator=g) * 0.1).to(torch.float32)
    # Round-trip through int16 PCM so the committed embedding corresponds to the
    # exact waveform the Rust e2e test reconstructs from `waveform_i16`.
    waveform_i16 = torch.clamp((audio_f32 * 32767.0).round(), -32768.0, 32767.0).to(torch.int16)
    audio = (waveform_i16.to(torch.float32) / 32767.0).numpy()
    inputs = processor(audios=audio, sampling_rate=sr, return_tensors="pt")

    # The 5s clip is shorter than the 10s window, so its 4 channels are the
    # repeatpad-stacked mel; the feature extractor nonetheless marks the clip
    # is_longer=True (deterministic fusion promotion), so get_audio_features runs
    # the AFF fusion path and returns the canonical embedding.
    input_features = inputs["input_features"]
    ch = input_features[0]
    assert torch.equal(ch[0], ch[1]) and torch.equal(ch[0], ch[2]) and torch.equal(ch[0], ch[3]), (
        "short-clip mel must be the repeatpad-stacked 4-identical-channel form"
    )
    is_longer = inputs["is_longer"].to(torch.bool)
    assert bool(is_longer.any()) is True, (
        "ClapFeatureExtractor must mark the clip is_longer=True (canonical fusion path)"
    )

    with torch.inference_mode():
        embedding = model.get_audio_features(**inputs)

    goldens = {
        "input_features": input_features.to(torch.float32).cpu().contiguous(),
        # Fusion gate as u8 (0/1): candle's safetensors loader has no BOOL dtype,
        # so the Rust live test reads this and maps != 0 -> bool. This is the
        # canonical always-fusion flag (True) jammi's front-end emits for every
        # clip, matching the feature extractor's deterministic promotion.
        "is_longer": is_longer.to(torch.uint8).cpu().contiguous(),
        "embedding": embedding.to(torch.float32).cpu().contiguous(),
        "waveform_i16": waveform_i16.cpu().contiguous(),
    }
    save_file(goldens, os.path.join(REAL_OUT, "goldens.safetensors"))
    write_manifest(goldens, os.path.join(REAL_OUT, "golden_manifest.json"))
    with open(os.path.join(REAL_OUT, "model_id.txt"), "w") as f:
        f.write(model_id + "\n")
    print(
        f"[real] {REAL_OUT}: embedding {tuple(embedding.shape)} "
        f"is_longer={is_longer.tolist()} (canonical fusion) from {model_id}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", nargs="?", const=REAL_MODEL_ID, default=None,
                        help=f"also generate the real-model golden (default id: {REAL_MODEL_ID})")
    args = parser.parse_args()

    generate_tiny()
    if args.real is not None:
        generate_real(args.real)
