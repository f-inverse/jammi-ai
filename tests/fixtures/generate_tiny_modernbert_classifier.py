#!/usr/bin/env python3
"""Generate a minimal ModernBERT classification model fixture for hermetic testing.

Creates tests/fixtures/tiny_modernbert_classifier/ with:
  - config.json      (ModernBERT config with classifier fields, hidden=32, 1 layer, 2 heads, vocab=256)
  - model.safetensors (random weights including head and classifier layers)
  - tokenizer.json    (minimal WordPiece tokenizer with 256-token vocab)

The model is ~20KB total — small enough to commit to the repo.
It produces garbage predictions (random weights) but exercises the full
ModernBertForSequenceClassification::load() → forward() → softmax pipeline.
"""

import json
import os

import numpy as np
from safetensors.numpy import save_file
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_modernbert_classifier")

# Model dimensions
HIDDEN = 32
INTERMEDIATE = 64
HEADS = 2
LAYERS = 1
VOCAB = 256
MAX_POS = 128
NUM_CLASSES = 2


def generate_config():
    config = {
        "architectures": ["ModernBertForSequenceClassification"],
        "model_type": "modernbert",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "intermediate_size": INTERMEDIATE,
        "vocab_size": VOCAB,
        "max_position_embeddings": MAX_POS,
        "layer_norm_eps": 1e-5,
        "pad_token_id": 0,
        "global_attn_every_n_layers": 1,
        "global_rope_theta": 160000.0,
        "local_attention": 64,
        "local_rope_theta": 10000.0,
        "id2label": {"0": "physics", "1": "biology"},
        "label2id": {"physics": "0", "biology": "1"},
        "classifier_pooling": "cls",
    }
    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def rand(shape, scale=0.02):
    """Small random weights matching typical initializer_range."""
    return np.random.randn(*shape).astype(np.float32) * scale


def ones(shape):
    return np.ones(shape, dtype=np.float32)


def zeros(shape):
    return np.zeros(shape, dtype=np.float32)


def generate_weights():
    """Create all tensors that candle's ModernBertForSequenceClassification::load() expects.

    Base model layers use linear_no_bias and layer_norm_no_bias — no bias tensors.
    The classifier head uses linear (with bias).
    """
    tensors = {}

    # === Base ModernBERT model weights (same as tiny_modernbert) ===

    # Embeddings
    tensors["model.embeddings.tok_embeddings.weight"] = rand((VOCAB, HIDDEN))
    tensors["model.embeddings.norm.weight"] = ones((HIDDEN,))

    # Encoder layers
    for i in range(LAYERS):
        prefix = f"model.layers.{i}"

        # Attention norm (optional in candle, loaded with .ok())
        tensors[f"{prefix}.attn_norm.weight"] = ones((HIDDEN,))

        # Fused QKV projection (no bias)
        tensors[f"{prefix}.attn.Wqkv.weight"] = rand((HIDDEN * 3, HIDDEN))

        # Output projection (no bias)
        tensors[f"{prefix}.attn.Wo.weight"] = rand((HIDDEN, HIDDEN))

        # MLP norm
        tensors[f"{prefix}.mlp_norm.weight"] = ones((HIDDEN,))

        # GLU gate+up fused (no bias)
        tensors[f"{prefix}.mlp.Wi.weight"] = rand((INTERMEDIATE * 2, HIDDEN))

        # MLP output (no bias)
        tensors[f"{prefix}.mlp.Wo.weight"] = rand((HIDDEN, INTERMEDIATE))

    # Final norm
    tensors["model.final_norm.weight"] = ones((HIDDEN,))

    # === Head weights (loaded via vb.pp("head")) ===
    # ModernBertHead: dense (linear_no_bias) + norm (layer_norm_no_bias)
    tensors["head.dense.weight"] = rand((HIDDEN, HIDDEN))
    tensors["head.norm.weight"] = ones((HIDDEN,))

    # === Classifier weights (loaded via vb.pp("classifier")) ===
    # ModernBertClassifier: linear (with bias)
    tensors["classifier.weight"] = rand((NUM_CLASSES, HIDDEN))
    tensors["classifier.bias"] = zeros((NUM_CLASSES,))

    save_file(tensors, os.path.join(OUT, "model.safetensors"))


def generate_tokenizer():
    """Create a minimal WordPiece tokenizer with 256 tokens."""
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
    print(f"Tiny ModernBERT classifier fixture generated in {OUT}")
