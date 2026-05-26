# Evaluate retrieval quality

Measure recall@k, precision@k, MRR, and nDCG of an embedding index against
a golden relevance set.

**When to use this pattern.** You have a corpus and a small set of
(query, expected document) judgments, and you need a number that tells
you "is my new encoder better than the one I shipped last month?" The
same loop powers nightly regression dashboards and A/B model comparison.

## What `example.py` does

1. Connects to a temporary artifact dir
2. Registers the tiny corpus as a Parquet source
3. Builds 32-dim embeddings over the `content` column with the local
   `tiny_bert` fixture
4. Reads `cookbook/fixtures/tiny_golden.json`, expands it into the
   `(query_id, query_text, relevant_id)` CSV shape `eval_embeddings`
   consumes, and registers it as a `golden` source
5. Calls `db.eval_embeddings(source="corpus", golden_source="golden.public.golden", k=5)`
6. Asserts each returned metric is in `[0.0, 1.0]`

## API surface exercised

- `Database.generate_text_embeddings(source, *, model, columns, key)`
- `Database.eval_embeddings(*, source, golden_source, model=None, k=10)`

The returned dict carries the aggregate metrics `recall_at_k`,
`precision_at_k`, `mrr`, and `ndcg` (averaged across queries). For
per-query drill-down, call `RetrievalMetrics::compute_query` directly
from `jammi-numerics`; the OSS Python surface returns the aggregate only.

## Golden source shape

`eval_embeddings` requires a registered source with these columns:

| column        | type | example                              |
|---------------|------|--------------------------------------|
| `query_id`    | utf8 | `q1`                                 |
| `query_text`  | utf8 | `quantum computing applications`     |
| `relevant_id` | utf8 | `1` (matches `corpus.id` as a string)|

Image queries are supported via a `query_image` BLOB column instead of
`query_text`; cross-modal eval is out of scope for this recipe.

## Run it

```bash
python cookbook/recipes/eval_embeddings/example.py
```

Exits 0 on success, prints the metrics dict + `eval_embeddings: OK`.
