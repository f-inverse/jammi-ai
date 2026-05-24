#!/usr/bin/env python3
"""Generate a minimal OpenCLIP (vision + text) model fixture for hermetic testing.

Creates tests/fixtures/tiny_open_clip/ with:
  - open_clip_config.json  (OpenCLIP config, width=32, 1 layer, image_size=8, patch_size=4)
  - open_clip_model.safetensors (random weights matching OpenCLIP ViT + text layout)

The model is tiny (~50KB) — small enough to commit to the repo.
It produces garbage embeddings (random weights) but exercises the full
CandleBackend::load() → OpenClip{Vision,Text}Transformer::forward() → L2_normalize
pipeline for both modalities. Vision and text share the EMBED_DIM latent space.

Also generates:
  - tests/fixtures/figures.parquet with inline PNG image data
  - tests/fixtures/tiny_open_clip/tokenizer.json — a tiny HF-shape tokenizer
    that maps the printable ASCII vocabulary to token IDs and reserves the
    highest ID for <|endoftext|> (matches OpenCLIP's EOT-pool convention).
"""

import io
import json
import os

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors.numpy import save_file

FIXTURES = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(FIXTURES, "tiny_open_clip")

# Model dimensions (tiny)
WIDTH = 32        # Hidden dimension (vision tower)
HEADS = 4         # Attention heads (head_dim=8)
LAYERS = 1        # Transformer layers (vision)
MLP_RATIO = 4.0   # MLP intermediate = WIDTH * MLP_RATIO = 128
IMAGE_SIZE = 8    # Input image size
PATCH_SIZE = 4    # Patch embedding kernel
EMBED_DIM = 16    # Shared latent dim (vision proj output = text proj output)
GRID = IMAGE_SIZE // PATCH_SIZE  # 2
NUM_PATCHES = GRID * GRID        # 4
NUM_POSITIONS = NUM_PATCHES + 1  # 5 (4 patches + CLS)
INTERMEDIATE = int(WIDTH * MLP_RATIO)

# Text tower dimensions
TEXT_WIDTH = 32          # Hidden width inside text transformer
TEXT_HEADS = 4           # Attention heads (head_dim=8)
TEXT_LAYERS = 1          # Transformer layers (text)
TEXT_INTERMEDIATE = TEXT_WIDTH * 4   # OpenCLIP text MLP ratio is fixed at 4x
CONTEXT_LENGTH = 32      # Max text seq length (small but >= test inputs)
# Vocab covers <pad>=0, 95 printable-ASCII codepoints (chr 32..=126), and
# <|endoftext|>=96 as the highest token ID (the OpenCLIP EOT-pool path
# selects argmax(input_ids), so EOT must be the highest ID).
VOCAB_SIZE = 97


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
            "vision_cfg": {
                "image_size": IMAGE_SIZE,
                "patch_size": PATCH_SIZE,
                "width": WIDTH,
                "layers": LAYERS,
                "heads": HEADS,
                "mlp_ratio": MLP_RATIO,
                "global_average_pool": True,
            },
            "text_cfg": {
                "context_length": CONTEXT_LENGTH,
                "vocab_size": VOCAB_SIZE,
                "width": TEXT_WIDTH,
                "heads": TEXT_HEADS,
                "layers": TEXT_LAYERS,
            },
        },
        "preprocess_cfg": {
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
            "interpolation": "bicubic",
        },
    }
    with open(os.path.join(OUT, "open_clip_config.json"), "w") as f:
        json.dump(config, f, indent=2)


def generate_weights():
    """Create all tensors that OpenClipVisionTransformer::load() expects.

    Weight keys match the OpenCLIP safetensors layout under the 'visual.*' prefix.
    """
    tensors = {}

    # Patch embedding (Conv2D: 3 -> WIDTH, kernel=PATCH_SIZE, stride=PATCH_SIZE)
    tensors["visual.conv1.weight"] = rand((WIDTH, 3, PATCH_SIZE, PATCH_SIZE))

    # CLS token and positional embedding
    tensors["visual.class_embedding"] = rand((WIDTH,))
    tensors["visual.positional_embedding"] = rand((NUM_POSITIONS, WIDTH))

    # Pre-LayerNorm
    tensors["visual.ln_pre.weight"] = ones((WIDTH,))
    tensors["visual.ln_pre.bias"] = zeros((WIDTH,))

    # Transformer blocks
    for i in range(LAYERS):
        prefix = f"visual.transformer.resblocks.{i}"

        # Attention: fused in_proj (QKV) + out_proj
        tensors[f"{prefix}.attn.in_proj_weight"] = rand((WIDTH * 3, WIDTH))
        tensors[f"{prefix}.attn.in_proj_bias"] = zeros((WIDTH * 3,))
        tensors[f"{prefix}.attn.out_proj.weight"] = rand((WIDTH, WIDTH))
        tensors[f"{prefix}.attn.out_proj.bias"] = zeros((WIDTH,))

        # LayerNorms
        tensors[f"{prefix}.ln_1.weight"] = ones((WIDTH,))
        tensors[f"{prefix}.ln_1.bias"] = zeros((WIDTH,))
        tensors[f"{prefix}.ln_2.weight"] = ones((WIDTH,))
        tensors[f"{prefix}.ln_2.bias"] = zeros((WIDTH,))

        # MLP
        tensors[f"{prefix}.mlp.c_fc.weight"] = rand((INTERMEDIATE, WIDTH))
        tensors[f"{prefix}.mlp.c_fc.bias"] = zeros((INTERMEDIATE,))
        tensors[f"{prefix}.mlp.c_proj.weight"] = rand((WIDTH, INTERMEDIATE))
        tensors[f"{prefix}.mlp.c_proj.bias"] = zeros((WIDTH,))

    # Post-LayerNorm
    tensors["visual.ln_post.weight"] = ones((WIDTH,))
    tensors["visual.ln_post.bias"] = zeros((WIDTH,))

    # Projection: WIDTH -> EMBED_DIM
    tensors["visual.proj"] = rand((WIDTH, EMBED_DIM))

    # ─── Text tower (root-level keys: token_embedding, transformer.*, ln_final, text_projection) ──
    tensors["token_embedding.weight"] = rand((VOCAB_SIZE, TEXT_WIDTH))
    tensors["positional_embedding"] = rand((CONTEXT_LENGTH, TEXT_WIDTH))

    for i in range(TEXT_LAYERS):
        prefix = f"transformer.resblocks.{i}"

        tensors[f"{prefix}.attn.in_proj_weight"] = rand((TEXT_WIDTH * 3, TEXT_WIDTH))
        tensors[f"{prefix}.attn.in_proj_bias"] = zeros((TEXT_WIDTH * 3,))
        tensors[f"{prefix}.attn.out_proj.weight"] = rand((TEXT_WIDTH, TEXT_WIDTH))
        tensors[f"{prefix}.attn.out_proj.bias"] = zeros((TEXT_WIDTH,))

        tensors[f"{prefix}.ln_1.weight"] = ones((TEXT_WIDTH,))
        tensors[f"{prefix}.ln_1.bias"] = zeros((TEXT_WIDTH,))
        tensors[f"{prefix}.ln_2.weight"] = ones((TEXT_WIDTH,))
        tensors[f"{prefix}.ln_2.bias"] = zeros((TEXT_WIDTH,))

        tensors[f"{prefix}.mlp.c_fc.weight"] = rand((TEXT_INTERMEDIATE, TEXT_WIDTH))
        tensors[f"{prefix}.mlp.c_fc.bias"] = zeros((TEXT_INTERMEDIATE,))
        tensors[f"{prefix}.mlp.c_proj.weight"] = rand((TEXT_WIDTH, TEXT_INTERMEDIATE))
        tensors[f"{prefix}.mlp.c_proj.bias"] = zeros((TEXT_WIDTH,))

    tensors["ln_final.weight"] = ones((TEXT_WIDTH,))
    tensors["ln_final.bias"] = zeros((TEXT_WIDTH,))

    # Text projection: TEXT_WIDTH -> EMBED_DIM (shared latent with vision)
    tensors["text_projection"] = rand((TEXT_WIDTH, EMBED_DIM))

    save_file(tensors, os.path.join(OUT, "open_clip_model.safetensors"))


def generate_tokenizer():
    """Emit an HF-shape tokenizer.json built from the canonical `tokenizers`
    library so deserialization round-trips through any compatible runtime.

    The tokenizer maps printable ASCII codepoints to single-token IDs and
    reserves the highest ID for `<|endoftext|>` — the OpenCLIP EOT marker
    that the text tower's pool path locates via argmax(input_ids).
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Split
    from tokenizers.processors import TemplateProcessing
    from tokenizers import AddedToken

    eot_id = VOCAB_SIZE - 1
    vocab = {"<pad>": 0}
    for i, cp in enumerate(range(32, 127)):
        vocab[chr(cp)] = i + 1
    vocab["<|endoftext|>"] = eot_id

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<pad>"))
    tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated", invert=True)
    tokenizer.post_processor = TemplateProcessing(
        single="$A <|endoftext|>",
        special_tokens=[("<|endoftext|>", eot_id)],
    )
    tokenizer.add_special_tokens([
        AddedToken("<pad>", special=True),
        AddedToken("<|endoftext|>", special=True),
    ])
    tokenizer.save(os.path.join(OUT, "tokenizer.json"))


def generate_figures_parquet():
    """Create a Parquet file with inline PNG image data for testing."""
    images = []
    ids = []

    for i in range(5):
        # Create distinct colored rectangles so each image is different
        img = Image.new("RGB", (20, 30), (255, 255, 255))
        for y in range(5 + i * 3, 15 + i * 3):
            for x in range(5, 15):
                img.putpixel((x, y), (i * 50, 100, 200 - i * 30))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        images.append(buf.getvalue())
        ids.append(f"fig_{i}")

    table = pa.table(
        {
            "figure_id": pa.array(ids, type=pa.utf8()),
            "image": pa.array(images, type=pa.binary()),
        }
    )
    pq.write_table(table, os.path.join(FIXTURES, "figures.parquet"))


if __name__ == "__main__":
    np.random.seed(42)
    os.makedirs(OUT, exist_ok=True)
    generate_config()
    generate_weights()
    generate_tokenizer()
    generate_figures_parquet()

    # Report sizes
    total = 0
    for f in ["open_clip_config.json", "open_clip_model.safetensors", "tokenizer.json"]:
        size = os.path.getsize(os.path.join(OUT, f))
        total += size
        print(f"  {f}: {size:,} bytes")
    print(f"  Total model: {total:,} bytes")

    fig_size = os.path.getsize(os.path.join(FIXTURES, "figures.parquet"))
    print(f"  figures.parquet: {fig_size:,} bytes")
    print(f"Tiny OpenCLIP fixture generated in {OUT}")
