#!/usr/bin/env python3
"""Generate a `head_dim == 64` BERT model fixture for the K4 GPU device leg.

Creates cookbook/fixtures/tiny_bert_head64/ with:
  - config.json      (BERT config, hidden=64, 1 layer, 1 head -> head_dim=64)
  - model.safetensors (random weights in the correct tensor layout)
  - tokenizer.json    (the SAME 256-token WordPiece vocab as `tiny_bert`,
                       copied verbatim so both fixtures tokenize identically)

Unit 62 / CONTRACT.md E5 needs a `head_dim == 64` checkpoint for the K4
transport-only device leg (`grpc_remote_session_gpu.rs`): the tiny BERT-arch
fixtures the rest of the suite uses (`tiny_bert`, `tiny_modernbert_*`) are all
`hidden_size / num_heads == 16`, deliberately small enough that flash/block
attention arms never dispatch (see PLAN.md v2 G3) — irrelevant here since this
leg asserts *transport* bitwise parity, not kernel-arm dispatch, but head_dim
64 is the shape a real embedding-model checkpoint uses, and this fixture keeps
the same BertModel loader path `tiny_bert` already exercises (single self-
attention head is head_dim==hidden_size, so `num_attention_heads=1` is the
minimal way to hit 64 without adding a second architecture family).

This is otherwise a direct copy of `generate_tiny_bert.py`'s tensor-layout
logic at a wider hidden size; see that file's docstring for the general
fixture-generation rationale.
"""

import json
import os
import shutil

import numpy as np
from safetensors.numpy import save_file

SELF_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SELF_DIR, "..", "..", "cookbook", "fixtures", "tiny_bert_head64")
TINY_BERT_DIR = os.path.join(SELF_DIR, "..", "..", "cookbook", "fixtures", "tiny_bert")

# Model dimensions — `hidden_size / num_attention_heads == 64` is the whole
# point of this fixture (`head_dim == 64`).
HIDDEN = 64
INTERMEDIATE = 256
HEADS = 1
LAYERS = 1
VOCAB = 256
MAX_POS = 128
TYPE_VOCAB = 2


def generate_config():
    config = {
        "architectures": ["BertModel"],
        "model_type": "bert",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "intermediate_size": INTERMEDIATE,
        "vocab_size": VOCAB,
        "max_position_embeddings": MAX_POS,
        "type_vocab_size": TYPE_VOCAB,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "layer_norm_eps": 1e-12,
        "initializer_range": 0.02,
        "position_embedding_type": "absolute",
        "pad_token_id": 0,
    }
    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def rand(shape, scale=0.02):
    """Small random weights matching BERT's initializer_range."""
    return np.random.randn(*shape).astype(np.float32) * scale


def ones(shape):
    return np.ones(shape, dtype=np.float32)


def zeros(shape):
    return np.zeros(shape, dtype=np.float32)


def generate_weights():
    """Create all tensors that candle's BertModel::load() expects."""
    tensors = {}

    tensors["embeddings.word_embeddings.weight"] = rand((VOCAB, HIDDEN))
    tensors["embeddings.position_embeddings.weight"] = rand((MAX_POS, HIDDEN))
    tensors["embeddings.token_type_embeddings.weight"] = rand((TYPE_VOCAB, HIDDEN))
    tensors["embeddings.LayerNorm.weight"] = ones((HIDDEN,))
    tensors["embeddings.LayerNorm.bias"] = zeros((HIDDEN,))

    for i in range(LAYERS):
        prefix = f"encoder.layer.{i}"

        tensors[f"{prefix}.attention.self.query.weight"] = rand((HIDDEN, HIDDEN))
        tensors[f"{prefix}.attention.self.query.bias"] = zeros((HIDDEN,))
        tensors[f"{prefix}.attention.self.key.weight"] = rand((HIDDEN, HIDDEN))
        tensors[f"{prefix}.attention.self.key.bias"] = zeros((HIDDEN,))
        tensors[f"{prefix}.attention.self.value.weight"] = rand((HIDDEN, HIDDEN))
        tensors[f"{prefix}.attention.self.value.bias"] = zeros((HIDDEN,))

        tensors[f"{prefix}.attention.output.dense.weight"] = rand((HIDDEN, HIDDEN))
        tensors[f"{prefix}.attention.output.dense.bias"] = zeros((HIDDEN,))
        tensors[f"{prefix}.attention.output.LayerNorm.weight"] = ones((HIDDEN,))
        tensors[f"{prefix}.attention.output.LayerNorm.bias"] = zeros((HIDDEN,))

        tensors[f"{prefix}.intermediate.dense.weight"] = rand((INTERMEDIATE, HIDDEN))
        tensors[f"{prefix}.intermediate.dense.bias"] = zeros((INTERMEDIATE,))
        tensors[f"{prefix}.output.dense.weight"] = rand((HIDDEN, INTERMEDIATE))
        tensors[f"{prefix}.output.dense.bias"] = zeros((HIDDEN,))
        tensors[f"{prefix}.output.LayerNorm.weight"] = ones((HIDDEN,))
        tensors[f"{prefix}.output.LayerNorm.bias"] = zeros((HIDDEN,))

    save_file(tensors, os.path.join(OUT, "model.safetensors"))


def copy_tokenizer():
    """Reuse `tiny_bert`'s tokenizer verbatim — same 256-token vocab, so the
    two fixtures tokenize identically and this fixture needs no new tokenizer
    generation logic."""
    src = os.path.join(TINY_BERT_DIR, "tokenizer.json")
    shutil.copyfile(src, os.path.join(OUT, "tokenizer.json"))


if __name__ == "__main__":
    np.random.seed(42)  # Reproducible fixtures
    os.makedirs(OUT, exist_ok=True)
    generate_config()
    generate_weights()
    copy_tokenizer()

    total = 0
    for f in ["config.json", "model.safetensors", "tokenizer.json"]:
        size = os.path.getsize(os.path.join(OUT, f))
        total += size
        print(f"  {f}: {size:,} bytes")
    print(f"  Total: {total:,} bytes")
    print(f"tiny_bert_head64 fixture generated in {OUT}")
