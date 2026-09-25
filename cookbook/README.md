# Jammi AI Cookbook

Learn every jammi capability by running it. The cookbook has two halves, and
every capability the Python client ships is run by at least one of them
(`ci/scripts/check_cookbook_coverage.py` fails a change that adds a verb
nothing runs):

- **Recipes** (`quickstart/`, `recipes/`) — one short `example.py` per
  capability, run end to end against the committed fixtures, in seconds, on a
  CPU. Start here.
- **The book** (`book/`) — long-form chapters that run a capability over a real
  dataset and check what they measured against frozen goldens: graph
  construction, propagation and fine-tuning on ogbn-arxiv, conformal
  prediction and calibration, retrieval, tenancy, point-in-time joins,
  incremental refresh, precision and more. Every chapter runs at `small` scale
  on a CPU in seconds and at `full` scale on a GPU. See
  [`book/README.md`](./book/README.md).

## Recipes

| If you want to… | Open |
|---|---|
| Run your first embedding and search in five minutes | [`quickstart/`](./quickstart/) |
| Fine-tune an encoder, with LoRA or over a graph | [`recipes/fine_tune/`](./recipes/fine_tune/) |
| Measure retrieval quality — recall@k, nDCG, per query | [`recipes/eval_embeddings/`](./recipes/eval_embeddings/) |
| Measure classification accuracy against labels | [`recipes/eval_inference/`](./recipes/eval_inference/) |
| Measure NER precision/recall/F1 against gold spans | [`recipes/eval_inference_ner/`](./recipes/eval_inference_ner/) |
| Build neighbour graphs, propagate, follow lineage | [`recipes/graph_and_lineage/`](./recipes/graph_and_lineage/) |
| Predict from retrieved context | [`recipes/context_predictor/`](./recipes/context_predictor/) |
| Search images | [`recipes/image_search/`](./recipes/image_search/) |
| Search audio | [`recipes/audio_search/`](./recipes/audio_search/) |
| Submit, watch and cancel jobs | [`recipes/jobs/`](./recipes/jobs/) |
| List, describe, preload and delete models | [`recipes/model_catalog/`](./recipes/model_catalog/) |
| Create, insert into and query a mutable table | [`recipes/mutable_tables/`](./recipes/mutable_tables/) |
| Publish and subscribe on a topic | [`recipes/trigger_streams/`](./recipes/trigger_streams/) |
| Store data deleted when the session ends | [`recipes/session_lifecycle/`](./recipes/session_lifecycle/) |
| Audit every search a session runs | [`recipes/search_audit/`](./recipes/search_audit/) |
| Run one program embedded and against a server | [`recipes/remote_session/`](./recipes/remote_session/) |
| Query a `jammi-server` over Flight SQL | [`recipes/flight_sql/`](./recipes/flight_sql/) |
| Embed and search with a model served at an endpoint | [`recipes/remote_model/`](./recipes/remote_model/) |

Every recipe uses a committed fixture model, so it runs without network
access; each recipe's README names the Hugging Face model a production caller
would use instead.

## Fixtures

`fixtures/` holds everything the recipes and the book's `small` scale read:
tiny text, image and audio encoders (`tiny_bert/`, `tiny_open_clip/`,
`htsat_clap_tiny/`, …), small corpora with their goldens (`tiny_corpus.parquet`,
`tiny_golden.json`, the image and audio corpora), and small excerpts of the
book's datasets (`arxiv_small/`, `air_routes/`, `finetune_heldout/`). The data
files regenerate with `python cookbook/fixtures/generate.py`; the encoder
weights are committed as built by `tests/fixtures/generate_tiny_*.py`.

## Running

```bash
pip install jammi-ai
# or, from a source checkout: build the engine into the current environment
maturin develop --release -m crates/jammi-python/Cargo.toml

python cookbook/quickstart/quickstart.py
python cookbook/recipes/fine_tune/example.py

# Every recipe, as CI runs them on every change. The server recipes start a
# `jammi-server` from PATH (`cargo build --release -p jammi-server`).
python tests/cookbook_smoke.py
```
