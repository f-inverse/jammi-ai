# Evaluate retrieval quality

Measure recall@k, precision@k, MRR, and nDCG of an embedding index against
a golden relevance set.

**When to use this pattern.** You have a corpus and a small set of
(query, expected document) judgments, and you need a number that tells
you "is my new encoder better than the one I shipped last month?" The
same loop powers nightly regression dashboards and A/B model comparison.

## API surface exercised

- `Session.generate_embeddings(*, source, model, columns, key, modality="text")`
- `Session.eval_embeddings(*, source, golden_source, model=None, k=10)`

The returned dict carries `aggregate` (mean across queries — `recall_at_k`,
`precision_at_k`, `mrr`, `ndcg`) and `per_query` (one entry per query with
`query_id` and a `metrics` sub-dict of the same four names, un-averaged).

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

It prints the four aggregate scores, the stored run, a cohort tag read back, and the smoothed table's recall delta.
