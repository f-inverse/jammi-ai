#!/usr/bin/env python3
"""Generate a minimal ModernBERT NER (token classification) model fixture for hermetic testing.

Creates tests/fixtures/tiny_modernbert_ner/ with:
  - config.json      (ModernBERT config with NER label fields, hidden=32, 1 layer, 2 heads, vocab=256)
  - model.safetensors (random weights including token-level classifier)
  - tokenizer.json    (minimal WordPiece tokenizer with 256-token vocab)

The model is ~20KB total — small enough to commit to the repo.
It produces garbage predictions (random weights) but exercises the full
ModernBertForTokenClassification::load() -> forward() pipeline.
"""

import json
import os

import numpy as np
from safetensors.numpy import save_file
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_modernbert_ner")

# Model dimensions
HIDDEN = 32
INTERMEDIATE = 64
HEADS = 2
LAYERS = 1
VOCAB = 256
MAX_POS = 128
NUM_LABELS = 5


def generate_config():
    config = {
        "architectures": ["ModernBertForTokenClassification"],
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
        # NER-specific fields
        "id2label": {
            "0": "O",
            "1": "B-PER",
            "2": "I-PER",
            "3": "B-ORG",
            "4": "I-ORG",
        },
        "label2id": {
            "O": "0",
            "B-PER": "1",
            "I-PER": "2",
            "B-ORG": "3",
            "I-ORG": "4",
        },
        "num_labels": NUM_LABELS,
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
    """Create all tensors that candle's ModernBertForTokenClassification::load() expects.

    Base model layers use linear_no_bias and layer_norm_no_bias — no bias tensors.
    The token classifier head uses linear (with bias) directly on hidden states.
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

    # === Token classifier weights (loaded via vb.pp("classifier")) ===
    # NER uses a simple Linear layer directly on hidden states (no pooling head)
    tensors["classifier.weight"] = rand((NUM_LABELS, HIDDEN))
    tensors["classifier.bias"] = zeros((NUM_LABELS,))

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
    print(f"Tiny ModernBERT NER fixture generated in {OUT}")
