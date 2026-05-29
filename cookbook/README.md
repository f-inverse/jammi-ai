# Jammi AI Cookbook

Runnable recipes for the OSS `jammi-ai` Python package. Every example here
lives in a single `example.py` file, runs end-to-end against the local
fixtures in `cookbook/fixtures/`, and is gated by CI so a broken recipe
blocks the merge.

## Where to start

| If you want to... | Open |
|---|---|
| Run your first vector query in 5 minutes | [`quickstart/`](./quickstart/) |
| See create/insert/select on a mutable table | [`recipes/mutable_tables/`](./recipes/mutable_tables/) |
| Publish + subscribe on a topic | [`recipes/trigger_streams/`](./recipes/trigger_streams/) |
| Store transient data that is auto-deleted on session end | [`recipes/session_lifecycle/`](./recipes/session_lifecycle/) |
| Measure recall@k / nDCG against a golden set | [`recipes/eval_embeddings/`](./recipes/eval_embeddings/) |
| Measure classification accuracy against gold labels | [`recipes/eval_inference/`](./recipes/eval_inference/) |
| Measure NER precision/recall/F1 against gold spans | [`recipes/eval_inference_ner/`](./recipes/eval_inference_ner/) |
| Fine-tune an encoder with LoRA | [`recipes/fine_tune/`](./recipes/fine_tune/) |
| Connect to a remote `jammi-server` via Flight SQL | [`recipes/flight_sql/`](./recipes/flight_sql/) |

## Fixtures

Every example loads from `cookbook/fixtures/`:

- `tiny_corpus.parquet` — 20 synthetic patents with `id`, `title`, `content`,
  `year`, `category`, `assignee_id`
- `tiny_golden.json` — 5 queries with relevance judgments for
  `eval_embeddings`
- `tiny_labels.csv` — per-row classification labels for `eval_inference`
- `tiny_pairs.csv` — contrastive (text_a, text_b, score) pairs for
  `fine_tune`
- `tiny_ner_corpus.parquet` — 20 generic PER/ORG sentences for the
  `eval_inference_ner` recipe
- `tiny_ner_gold.csv` — per-span gold entities (`id`, `label`, `start`,
  `end`) for the NER corpus
- `tiny_bert/` — 32-dim BERT encoder weights (88 KB) for every text
  embedding example
- `tiny_modernbert_classifier/` — tiny ModernBERT-for-classification
  weights for `eval_inference`
- `tiny_modernbert_ner/` — tiny ModernBERT-for-token-classification
  weights (PER + ORG labels) for `eval_inference_ner`

The fixture tree is under 5 MB total so CI checkouts stay fast. Regenerate
the data files with `python cookbook/fixtures/generate.py`; the encoder
weights are committed as-is and generated separately by the scripts under
`tests/fixtures/generate_tiny_*.py`.

## Real vs. fixture models

Every `example.py` uses the local fixture model so it runs without network
access. Each recipe's README points at the corresponding Hugging Face Hub
model ID that a user would substitute in production — for instance
`sentence-transformers/all-MiniLM-L6-v2` for text embeddings or
`answerdotai/ModernBERT-base` for classification.

## Running

```bash
pip install jammi-ai
# OR, from a source checkout:
maturin develop --release

# Run one recipe
python cookbook/quickstart/quickstart.py
python cookbook/recipes/mutable_tables/example.py

# Run the whole smoke suite (CI does this on every PR)
python tests/cookbook_smoke.py

# Include the slow + integration recipes (fine_tune, flight_sql)
JAMMI_COOKBOOK_SLOW=1 python tests/cookbook_smoke.py
```

## Mirroring

These recipes are the OSS source of truth. The platform docs site at
`docs.jammi.cloud/sdk/python/recipes/` ingests this tree read-only; do
not author recipe content in two places.
