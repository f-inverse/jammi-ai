# Cookbook Recipes (runnable)

Every recipe under [`cookbook/`](https://github.com/f-inverse/jammi-ai/tree/main/cookbook)
ships as a runnable `example.py` next to a markdown README and is wired
into CI via `tests/cookbook_smoke.py` — a broken recipe blocks the
merge. The cookbook is the OSS source of truth; this page mirrors each
README below.

The recipes shipped at MVP:

| Recipe | Demonstrates |
|---|---|
| [`mutable_tables`](#mutable-tables) | Create/insert/select/drop on a mutable companion table |
| [`trigger_streams`](#trigger-streams) | Publish + subscribe on a topic via the in-process broker |
| [`eval_embeddings`](#evaluate-retrieval-quality) | recall@k, MRR, nDCG against a golden set |
| [`eval_inference`](#evaluate-inference-classification) | Accuracy + macro F1 against gold labels |
| [`eval_inference_ner`](#evaluate-inference-ner) | Entity-level precision / recall / F1 against gold spans |
| [`fine_tune`](#fine-tune-an-encoder) | LoRA fine-tune end-to-end |
| [`flight_sql`](#connect-via-flight-sql) | Query a remote `jammi serve` over Arrow Flight SQL |

---

## Mutable tables

{{#include ../../../cookbook/recipes/mutable_tables/README.md:3:}}

---

## Trigger streams

{{#include ../../../cookbook/recipes/trigger_streams/README.md:3:}}

---

## Evaluate retrieval quality

{{#include ../../../cookbook/recipes/eval_embeddings/README.md:3:}}

---

## Evaluate inference (classification)

{{#include ../../../cookbook/recipes/eval_inference/README.md:3:}}

---

## Evaluate inference (NER)

{{#include ../../../cookbook/recipes/eval_inference_ner/README.md:3:}}

---

## Fine-tune an encoder

{{#include ../../../cookbook/recipes/fine_tune/README.md:3:}}

---

## Connect via Flight SQL

{{#include ../../../cookbook/recipes/flight_sql/README.md:3:}}
