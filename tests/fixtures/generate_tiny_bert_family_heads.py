#!/usr/bin/env python3
"""Generate tiny BERT-family checkpoints whose outputs `transformers` itself computed.

Creates, under tests/fixtures/:
  - tiny_bert_classifier/     `BertForSequenceClassification`: encoder under `bert.`,
                              the pooler (`bert.pooler.dense`), a `classifier` linear.
  - tiny_roberta_classifier/  `RobertaForSequenceClassification`: encoder under
                              `roberta.`, no pooler, the `classifier.dense` +
                              `classifier.out_proj` head.
  - tiny_roberta/             a raw `RobertaModel` (no key prefix), the layout a
                              sentence-transformers RoBERTa / XLM-R checkpoint ships.

Each directory holds config.json, model.safetensors, tokenizer.json and
goldens.json: the fixed input texts, their token ids, and what `transformers` produced for them,
padded as one batch (class probabilities for a classifier, the last hidden state
of every real token for the encoder). The engine's tests assert against those
numbers, so they hold the engine to the reference implementation with no torch
and no network.

Weights are random (seeded) at `initializer_range` 0.2 rather than the usual
0.02, so the heads move the output far beyond the tests' tolerance: the script
refuses to write a classifier whose probabilities without its pooled head stay
within 0.05 of the real ones, and an encoder whose hidden states with BERT's
0-based positions stay within 0.05 of RoBERTa's offset ones.

Environment:  pip install "torch==2.8.0" "transformers==4.57.6" "safetensors" "tokenizers"
Run:          python3 tests/fixtures/generate_tiny_bert_family_heads.py
"""

import json
import os

import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaModel,
)

FIXTURES = os.path.dirname(os.path.abspath(__file__))

HIDDEN = 32
INTERMEDIATE = 64
HEADS = 2
LAYERS = 2
VOCAB = 256
MAX_TOKENS = 64
INIT_RANGE = 0.2
LABELS = {0: "negative", 1: "neutral", 2: "positive"}
MIN_HEAD_EFFECT = 0.05

# Lengths differ so the batch carries padding.
TEXTS = [
    "the model is good",
    "a novel approach based on neural network learning with high energy results",
    "this is not the data we use",
    "quantum",
    "the protein structure of each cell surface was not what we present",
]

COMMON_WORDS = [
    "the", "is", "a", "an", "of", "in", "to", "and", "for", "on",
    "with", "this", "that", "we", "our", "new", "from", "by", "at",
    "are", "as", "be", "or", "not", "it", "can", "has", "was", "but",
    "all", "its", "one", "two", "more", "use", "been", "will", "each",
    "about", "how", "up", "out", "if", "do", "no", "so", "what", "when",
    "quantum", "model", "data", "error", "test", "patent", "method",
    "system", "using", "based", "learning", "network", "neural",
    "approach", "results", "novel", "present", "structure", "high",
    "energy", "cell", "protein", "gene", "surface", "design", "good",
]


def build_tokenizer(specials):
    """A WordPiece tokenizer whose special tokens sit at the ids `specials` maps.

    `specials` is ordered (cls, pad, sep, unk, mask) → id; BERT and RoBERTa
    differ only in those ids, which is what RoBERTa's positions depend on.
    """
    vocab = {name: idx for name, idx in specials.items()}
    idx = len(vocab)
    for word in list("abcdefghijklmnopqrstuvwxyz0123456789") + COMMON_WORDS:
        if word not in vocab and idx < VOCAB:
            vocab[word] = idx
            idx += 1
    while idx < VOCAB:
        vocab[f"[unused{idx}]"] = idx
        idx += 1
    cls, pad, sep, unk, mask = specials
    tokenizer = Tokenizer(models.WordPiece(vocab=vocab, unk_token=unk))
    tokenizer.normalizer = normalizers.Sequence([normalizers.Lowercase(), normalizers.StripAccents()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{cls} $A {sep}",
        special_tokens=[(cls, specials[cls]), (sep, specials[sep])],
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        cls_token=cls,
        pad_token=pad,
        sep_token=sep,
        unk_token=unk,
        mask_token=mask,
    )
    return tokenizer, fast


def batch(fast):
    enc = fast(TEXTS, padding=True, return_tensors="pt", return_token_type_ids=False)
    return enc["input_ids"], enc["attention_mask"]


def save(name, model, tokenizer, goldens):
    out = os.path.join(FIXTURES, name)
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save(os.path.join(out, "tokenizer.json"))
    with open(os.path.join(out, "goldens.json"), "w") as f:
        json.dump(goldens, f, indent=1)
        f.write("\n")
    total = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
    print(f"  {name}: {total:,} bytes")


def max_abs(a, b):
    return float((a - b).abs().max())


def tokens(ids, mask):
    """Each text's token ids, unpadded: what the tokenizer must reproduce."""
    return [row[: int(m.sum())].tolist() for row, m in zip(ids, mask)]


def classifier_goldens(ids, mask, probs):
    return {
        "texts": TEXTS,
        "input_ids": tokens(ids, mask),
        "id2label": {str(k): v for k, v in LABELS.items()},
        "probabilities": [[round(float(p), 7) for p in row] for row in probs],
    }


@torch.no_grad()
def bert_classifier():
    tokenizer, fast = build_tokenizer({"[CLS]": 2, "[PAD]": 0, "[SEP]": 3, "[UNK]": 1, "[MASK]": 4})
    config = BertConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, num_hidden_layers=LAYERS,
        num_attention_heads=HEADS, intermediate_size=INTERMEDIATE,
        max_position_embeddings=MAX_TOKENS, type_vocab_size=2, pad_token_id=0,
        initializer_range=INIT_RANGE, num_labels=len(LABELS),
        id2label=LABELS, label2id={v: k for k, v in LABELS.items()},
    )
    model = BertForSequenceClassification(config).eval()
    ids, mask = batch(fast)
    probs = model(input_ids=ids, attention_mask=mask).logits.softmax(-1)
    # What the engine computed before: the classifier over the raw [CLS] state.
    cls = model.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
    unpooled = model.classifier(cls).softmax(-1)
    effect = max_abs(probs, unpooled)
    assert effect > MIN_HEAD_EFFECT, f"BERT pooler moves probabilities by only {effect}"
    print(f"  BERT pooler moves probabilities by up to {effect:.3f}")
    save("tiny_bert_classifier", model, tokenizer, classifier_goldens(ids, mask, probs))


def roberta_config(**extra):
    return RobertaConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, num_hidden_layers=LAYERS,
        num_attention_heads=HEADS, intermediate_size=INTERMEDIATE,
        max_position_embeddings=MAX_TOKENS + 2, type_vocab_size=1,
        pad_token_id=1, bos_token_id=0, eos_token_id=2,
        initializer_range=INIT_RANGE, **extra,
    )


ROBERTA_SPECIALS = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4}


@torch.no_grad()
def roberta_classifier():
    tokenizer, fast = build_tokenizer(ROBERTA_SPECIALS)
    config = roberta_config(
        num_labels=len(LABELS), id2label=LABELS, label2id={v: k for k, v in LABELS.items()},
    )
    model = RobertaForSequenceClassification(config).eval()
    ids, mask = batch(fast)
    probs = model(input_ids=ids, attention_mask=mask).logits.softmax(-1)
    cls = model.roberta(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
    # The nearest thing the engine could compute: its single linear over [CLS]
    # does not exist in this checkpoint, so compare against the out_proj alone.
    without_dense = model.classifier.out_proj(cls).softmax(-1)
    effect = max_abs(probs, without_dense)
    assert effect > MIN_HEAD_EFFECT, f"RoBERTa head's dense moves probabilities by only {effect}"
    print(f"  RoBERTa head's dense + tanh moves probabilities by up to {effect:.3f}")
    save("tiny_roberta_classifier", model, tokenizer, classifier_goldens(ids, mask, probs))


@torch.no_grad()
def roberta_encoder():
    tokenizer, fast = build_tokenizer(ROBERTA_SPECIALS)
    model = RobertaModel(roberta_config(), add_pooling_layer=False).eval()
    ids, mask = batch(fast)
    hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
    # BERT's positions, 0..seq, against RoBERTa's padding_idx + 1 onwards.
    zero_based = torch.arange(ids.shape[1]).expand_as(ids)
    shifted = model(input_ids=ids, attention_mask=mask, position_ids=zero_based).last_hidden_state
    real = mask.bool()
    effect = max_abs(hidden[real], shifted[real])
    assert effect > MIN_HEAD_EFFECT, f"RoBERTa's position offset moves hidden states by only {effect}"
    print(f"  RoBERTa's position offset moves hidden states by up to {effect:.3f}")
    goldens = {
        "texts": TEXTS,
        "input_ids": tokens(ids, mask),
        # Each text's real tokens only: row i holds attention_mask[i].sum() vectors.
        "last_hidden_state": [
            [[round(float(x), 7) for x in tok] for tok in hidden[i][real[i]]]
            for i in range(len(TEXTS))
        ],
    }
    save("tiny_roberta", model, tokenizer, goldens)


if __name__ == "__main__":
    torch.manual_seed(42)
    bert_classifier()
    roberta_classifier()
    roberta_encoder()
