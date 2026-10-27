#!/usr/bin/env python3
"""Record what `transformers` computes for real BERT-family Hub checkpoints.

Writes tests/fixtures/bert_family_hub_goldens.json: for each checkpoint, the
commit it was measured at and, for a fixed set of texts run as one padded
batch, the class probabilities of a sequence classifier or the mean-pooled,
L2-normalized sentence embedding of an encoder. The engine's live tests
(`live-hub-tests`) serve the same checkpoints and assert they reproduce
these numbers.

  - bhadresh-savani/bert-base-uncased-emotion: BERT, classifier over the pooler.
  - mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis:
    RoBERTa, `classifier.dense` + `classifier.out_proj` head.
  - papluca/xlm-roberta-base-language-detection: XLM-RoBERTa, the same head.
  - sentence-transformers/all-distilroberta-v1: RoBERTa sentence embeddings
    (its own `1_Pooling` is mean, and it normalizes).

Environment:  pip install "torch==2.8.0" "transformers==4.57.6"
Run:          python3 tests/fixtures/generate_bert_family_hub_goldens.py
"""

import json
import os

import torch
from huggingface_hub import HfApi
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bert_family_hub_goldens.json")

TEXTS = [
    "The service was wonderful and the food was even better.",
    "Quarterly revenue fell sharply after the merger was cancelled.",
    "I am so angry that the delivery is late again",
    "Je ne sais pas si je viendrai demain.",
    "Das Wetter ist heute schön.",
    "¿Dónde está la estación de tren?",
]

CLASSIFIERS = [
    "bhadresh-savani/bert-base-uncased-emotion",
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
    "papluca/xlm-roberta-base-language-detection",
]
ENCODERS = ["sentence-transformers/all-distilroberta-v1"]


def batch(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer(TEXTS, padding=True, return_tensors="pt")


def rounded(rows):
    return [[round(float(x), 7) for x in row] for row in rows]


@torch.no_grad()
def classifier(model_id):
    model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
    probs = model(**batch(model_id)).logits.softmax(-1)
    return {
        "id2label": {str(k): v for k, v in model.config.id2label.items()},
        "probabilities": rounded(probs),
    }


@torch.no_grad()
def encoder(model_id):
    model = AutoModel.from_pretrained(model_id).eval()
    enc = batch(model_id)
    hidden = model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    mean = (hidden * mask).sum(1) / mask.sum(1)
    return {"embeddings": rounded(torch.nn.functional.normalize(mean, dim=-1))}


if __name__ == "__main__":
    api = HfApi()
    models = {}
    for model_id in CLASSIFIERS + ENCODERS:
        entry = {"revision": api.model_info(model_id).sha}
        entry.update(classifier(model_id) if model_id in CLASSIFIERS else encoder(model_id))
        models[model_id] = entry
        print(f"  {model_id} @ {entry['revision'][:12]}")
    with open(OUT, "w") as f:
        json.dump({"texts": TEXTS, "models": models}, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {OUT}")
