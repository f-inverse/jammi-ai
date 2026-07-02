# Runnable Recipes

Every recipe under [`cookbook/recipes/`](https://github.com/f-inverse/jammi-ai/tree/main/cookbook/recipes)
ships as a runnable `example.py` next to a markdown README and is wired
into CI via `tests/cookbook_smoke.py` — a broken recipe blocks the
merge. These recipes are the OSS source of truth; this page mirrors each
README below.

For the long-form, *measured* companion — **The Cookbook**, a Theory↔Computation
book that shows one recipe = one equation in the graph-signal-processing
monograph = one line of the GNN canon, executed against committed goldens — see
[The Cookbook](https://f-inverse.github.io/jammi-ai/cookbook/). The two are
complementary: these How-To Guides are the short, dual-language (Rust + Python),
compile-tested "how do I call this verb" reference; The Cookbook is the
long-form, Python, executed-and-measured narrative.

The recipes shipped at MVP:

| Recipe | Demonstrates |
|---|---|
| [`mutable_tables`](#mutable-tables) | Create/insert/select/drop on a mutable companion table |
| [`trigger_streams`](#trigger-streams) | Publish + subscribe on a topic via the in-process broker |
| [`eval_embeddings`](#evaluate-retrieval-quality) | recall@k, MRR, nDCG against a golden set |
| [`image_search`](#image-search) | Image-to-image search with PatentCLIP + Recall@K / MRR eval |
| [`eval_inference`](#evaluate-inference-classification) | Accuracy + macro F1 against gold labels |
| [`eval_inference_ner`](#evaluate-inference-ner) | Entity-level precision / recall / F1 against gold spans |
| [`fine_tune`](#fine-tune-an-encoder) | LoRA fine-tune end-to-end |
| [`flight_sql`](#connect-via-flight-sql) | Query a remote `jammi-server` over Arrow Flight SQL |
| [`audio_search`](#audio-search) | Audio-to-audio search with a CLAP encoder |
| [`search_audit`](#per-query-search-audit) | Per-query provenance audit of a search |
| [`session_lifecycle`](#ephemeral-session-storage) | Ephemeral session storage with scoped cleanup |

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

## Image search

{{#include ../../../cookbook/recipes/image_search/README.md:3:}}

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

---

## Audio search

{{#include ../../../cookbook/recipes/audio_search/README.md:3:}}

---

## Per-query search audit

{{#include ../../../cookbook/recipes/search_audit/README.md:3:}}

---

## Ephemeral session storage

{{#include ../../../cookbook/recipes/session_lifecycle/README.md:3:}}
