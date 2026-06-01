#!/usr/bin/env python3
"""Generate a minimal CLAP audio-tower fixture for hermetic testing.

Creates cookbook/fixtures/tiny_clap/ with:
  - open_clip_config.json       (CLAP config: model_cfg.audio_cfg block)
  - open_clip_model.safetensors (random weights matching the CLAP audio
    transformer layout under the `audio.*` prefix)

The model is tiny (~50KB) — small enough to commit to the repo. It produces
garbage embeddings (random weights) but exercises the full
CandleBackend::load() -> ClapAudio::forward() -> L2_normalize pipeline. It is
the audio analogue of `generate_tiny_open_clip.py` and is what the offline
`audio_search` cookbook recipe loads (`local:cookbook/fixtures/tiny_clap`).

Re-run with `python tests/fixtures/generate_tiny_clap.py`. Output is fully
deterministic (fixed seed).
"""

import json
import os

import numpy as np
from safetensors.numpy import save_file

# tests/fixtures/generate_tiny_clap.py -> repo root is two parents up.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO_ROOT, "cookbook", "fixtures", "tiny_clap")

# CLAP audio tower dimensions (tiny). The mel/frame geometry below must match
# what the audio_search fixtures feed and stay divisible by PATCH_SIZE.
N_MELS = 16        # mel-frequency bins (spectrogram height)
N_FRAMES = 32      # time frames every clip is padded/truncated to (width)
PATCH_SIZE = 4     # square patch-embedding kernel/stride
WIDTH = 32         # hidden width inside the transformer
HEADS = 4          # attention heads (head_dim=8)
LAYERS = 1         # transformer layers
MLP_RATIO = 4.0    # MLP intermediate = WIDTH * MLP_RATIO
EMBED_DIM = 16     # shared latent dim (audio projection output)

# Feature-extraction geometry (drives decode -> resample -> log-mel).
SAMPLE_RATE = 16_000
N_FFT = 256        # power of two for the radix-2 FFT
HOP_LENGTH = 128

GRID_H = N_MELS // PATCH_SIZE     # 4
GRID_W = N_FRAMES // PATCH_SIZE   # 8
NUM_PATCHES = GRID_H * GRID_W     # 32
INTERMEDIATE = int(WIDTH * MLP_RATIO)


def rand(shape, scale=0.02):
    return np.random.randn(*shape).astype(np.float32) * scale


def ones(shape):
    return np.ones(shape, dtype=np.float32)


def zeros(shape):
    return np.zeros(shape, dtype=np.float32)


def generate_config():
    config = {
        "model_cfg": {
            "embed_dim": EMBED_DIM,
            "audio_cfg": {
                "n_mels": N_MELS,
                "n_frames": N_FRAMES,
                "patch_size": PATCH_SIZE,
                "width": WIDTH,
                "layers": LAYERS,
                "heads": HEADS,
                "mlp_ratio": MLP_RATIO,
                "sample_rate": SAMPLE_RATE,
                "n_fft": N_FFT,
                "hop_length": HOP_LENGTH,
            },
        },
    }
    with open(os.path.join(OUT, "open_clip_config.json"), "w") as f:
        json.dump(config, f, indent=2)


def generate_weights():
    """Create every tensor that ClapAudio::load() expects under `audio.*`."""
    tensors = {}

    # Patch embedding (Conv2D: 1 channel -> WIDTH, kernel=stride=PATCH_SIZE).
    tensors["audio.conv1.weight"] = rand((WIDTH, 1, PATCH_SIZE, PATCH_SIZE))

    # Per-patch positional embedding (no class token — the tower mean-pools).
    tensors["audio.positional_embedding"] = rand((NUM_PATCHES, WIDTH))

    # Pre-LayerNorm.
    tensors["audio.ln_pre.weight"] = ones((WIDTH,))
    tensors["audio.ln_pre.bias"] = zeros((WIDTH,))

    for i in range(LAYERS):
        prefix = f"audio.transformer.resblocks.{i}"

        tensors[f"{prefix}.attn.in_proj_weight"] = rand((WIDTH * 3, WIDTH))
        tensors[f"{prefix}.attn.in_proj_bias"] = zeros((WIDTH * 3,))
        tensors[f"{prefix}.attn.out_proj.weight"] = rand((WIDTH, WIDTH))
        tensors[f"{prefix}.attn.out_proj.bias"] = zeros((WIDTH,))

        tensors[f"{prefix}.ln_1.weight"] = ones((WIDTH,))
        tensors[f"{prefix}.ln_1.bias"] = zeros((WIDTH,))
        tensors[f"{prefix}.ln_2.weight"] = ones((WIDTH,))
        tensors[f"{prefix}.ln_2.bias"] = zeros((WIDTH,))

        tensors[f"{prefix}.mlp.c_fc.weight"] = rand((INTERMEDIATE, WIDTH))
        tensors[f"{prefix}.mlp.c_fc.bias"] = zeros((INTERMEDIATE,))
        tensors[f"{prefix}.mlp.c_proj.weight"] = rand((WIDTH, INTERMEDIATE))
        tensors[f"{prefix}.mlp.c_proj.bias"] = zeros((WIDTH,))

    # Post-LayerNorm + projection into the shared latent.
    tensors["audio.ln_post.weight"] = ones((WIDTH,))
    tensors["audio.ln_post.bias"] = zeros((WIDTH,))
    tensors["audio.audio_projection"] = rand((WIDTH, EMBED_DIM))

    save_file(tensors, os.path.join(OUT, "open_clip_model.safetensors"))


if __name__ == "__main__":
    np.random.seed(42)
    os.makedirs(OUT, exist_ok=True)
    generate_config()
    generate_weights()

    total = 0
    for f in ["open_clip_config.json", "open_clip_model.safetensors"]:
        size = os.path.getsize(os.path.join(OUT, f))
        total += size
        print(f"  {f}: {size:,} bytes")
    print(f"  Total model: {total:,} bytes")
    print(f"Tiny CLAP fixture generated in {OUT}")
