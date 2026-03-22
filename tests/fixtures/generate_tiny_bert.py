#!/usr/bin/env python3
"""Generate a minimal BERT model fixture for hermetic end-to-end testing.

Creates tests/fixtures/tiny_bert/ with:
  - config.json      (BERT config, hidden=32, 1 layer, 2 heads, vocab=256)
  - model.safetensors (random weights in the correct tensor layout)
  - tokenizer.json    (minimal WordPiece tokenizer with 256-token vocab)

The model is ~85KB total — small enough to commit to the repo.
It produces garbage embeddings (random weights) but exercises the full
CandleBackend::load() → BertModel::forward() → mean_pool → L2_normalize pipeline.
"""

import json
import os

import numpy as np
from safetensors.numpy import save_file
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_bert")

# Model dimensions
HIDDEN = 32
INTERMEDIATE = 128
HEADS = 2
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

    # Embeddings
    tensors["embeddings.word_embeddings.weight"] = rand((VOCAB, HIDDEN))
    tensors["embeddings.position_embeddings.weight"] = rand((MAX_POS, HIDDEN))
    tensors["embeddings.token_type_embeddings.weight"] = rand((TYPE_VOCAB, HIDDEN))
    tensors["embeddings.LayerNorm.weight"] = ones((HIDDEN,))
    tensors["embeddings.LayerNorm.bias"] = zeros((HIDDEN,))

    # Encoder layers
    for i in range(LAYERS):
        prefix = f"encoder.layer.{i}"

        # Self-attention
        tensors[f"{prefix}.attention.self.query.weight"] = rand((HIDDEN, HIDDEN))
        tensors[f"{prefix}.attention.self.query.bias"] = zeros((HIDDEN,))
        tensors[f"{prefix}.attention.self.key.weight"] = rand((HIDDEN, HIDDEN))
        tensors[f"{prefix}.attention.self.key.bias"] = zeros((HIDDEN,))
        tensors[f"{prefix}.attention.self.value.weight"] = rand((HIDDEN, HIDDEN))
        tensors[f"{prefix}.attention.self.value.bias"] = zeros((HIDDEN,))

        # Attention output
        tensors[f"{prefix}.attention.output.dense.weight"] = rand((HIDDEN, HIDDEN))
        tensors[f"{prefix}.attention.output.dense.bias"] = zeros((HIDDEN,))
        tensors[f"{prefix}.attention.output.LayerNorm.weight"] = ones((HIDDEN,))
        tensors[f"{prefix}.attention.output.LayerNorm.bias"] = zeros((HIDDEN,))

        # Feed-forward
        tensors[f"{prefix}.intermediate.dense.weight"] = rand(
            (INTERMEDIATE, HIDDEN)
        )
        tensors[f"{prefix}.intermediate.dense.bias"] = zeros((INTERMEDIATE,))
        tensors[f"{prefix}.output.dense.weight"] = rand((HIDDEN, INTERMEDIATE))
        tensors[f"{prefix}.output.dense.bias"] = zeros((HIDDEN,))
        tensors[f"{prefix}.output.LayerNorm.weight"] = ones((HIDDEN,))
        tensors[f"{prefix}.output.LayerNorm.bias"] = zeros((HIDDEN,))

    save_file(tensors, os.path.join(OUT, "model.safetensors"))


def generate_tokenizer():
    """Create a minimal WordPiece tokenizer with 256 tokens."""
    # Build vocabulary: special tokens + ASCII printable + subword pieces
    vocab = {}
    # Special tokens
    vocab["[PAD]"] = 0
    vocab["[UNK]"] = 1
    vocab["[CLS]"] = 2
    vocab["[SEP]"] = 3
    vocab["[MASK]"] = 4

    # Single characters (a-z, 0-9, common punctuation)
    idx = 5
    for c in "abcdefghijklmnopqrstuvwxyz":
        vocab[c] = idx
        idx += 1
    for c in "0123456789":
        vocab[c] = idx
        idx += 1
    for c in " .,;:!?-'\"()[]{}/@#$%^&*+=<>~`_\\|":
        vocab[c] = idx
        idx += 1

    # Common English words
    common_words = [
        "the", "is", "a", "an", "of", "in", "to", "and", "for", "on",
        "with", "this", "that", "we", "our", "new", "from", "by", "at",
        "are", "as", "be", "or", "not", "it", "can", "has", "was", "but",
        "all", "its", "one", "two", "more", "use", "been", "will", "each",
        "about", "how", "up", "out", "if", "do", "no", "so", "what", "when",
        "quantum", "model", "data", "error", "test", "patent", "method",
        "system", "using", "based", "learning", "network", "neural",
        "approach", "results", "novel", "present", "structure", "high",
        "energy", "cell", "protein", "gene", "surface", "design",
    ]
    for word in common_words:
        if word not in vocab and idx < VOCAB:
            vocab[word] = idx
            idx += 1

    # WordPiece subword tokens
    subwords = [
        "##s", "##ed", "##ing", "##er", "##ly", "##tion", "##al", "##ment",
        "##ive", "##ous", "##ity", "##ness", "##able", "##ful", "##less",
        "##ize", "##ical", "##ence", "##ance",
    ]
    for sw in subwords:
        if idx < VOCAB:
            vocab[sw] = idx
            idx += 1

    # Fill remaining slots
    while idx < VOCAB:
        vocab[f"[unused{idx}]"] = idx
        idx += 1

    # Build tokenizer
    tokenizer = Tokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.Lowercase(), normalizers.StripAccents()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )

    tokenizer.save(os.path.join(OUT, "tokenizer.json"))


if __name__ == "__main__":
    np.random.seed(42)  # Reproducible fixtures
    os.makedirs(OUT, exist_ok=True)
    generate_config()
    generate_weights()
    generate_tokenizer()

    # Report sizes
    total = 0
    for f in ["config.json", "model.safetensors", "tokenizer.json"]:
        size = os.path.getsize(os.path.join(OUT, f))
        total += size
        print(f"  {f}: {size:,} bytes")
    print(f"  Total: {total:,} bytes")
    print(f"Tiny BERT fixture generated in {OUT}")
