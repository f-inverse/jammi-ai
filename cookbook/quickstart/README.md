# 5-minute quickstart

Goal: a fresh user goes from `pip install jammi-ai` to a successful vector
query in five minutes. The end-to-end script lives next to this file in
[`quickstart.py`](./quickstart.py) — copy-paste it, run it, then read the
four step-by-step pages for the explanation.

## Steps

1. [Install](./01_install.md) — `pip install jammi-ai`
2. [Connect](./02_connect.md) — open a session against a local artifact dir
3. [Register a source](./03_register_source.md) — attach a Parquet file
4. [Generate embeddings + search](./04_vector_search.md) — build a vector
   index and run a similarity query

## Run it

```bash
python cookbook/quickstart/quickstart.py
```

Expected output: a header row and three top-3 matches with cosine similarity
scores. The script exits 0 in under 30 seconds on CPU.

## Production substitution

The script uses the local `cookbook/fixtures/tiny_bert/` model (32-dim, 88 KB,
single-layer) so the example needs no network access. In a real workload
you would swap in a Hub model — for example
`sentence-transformers/all-MiniLM-L6-v2` (384-dim, English) — by changing
the `MODEL` constant. Everything else stays the same.
