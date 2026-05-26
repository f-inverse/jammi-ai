# 4. Generate embeddings and search

## Build the embedding table

```python
db.generate_text_embeddings(
    source="corpus",
    model=MODEL,        # e.g. "local:cookbook/fixtures/tiny_bert"
    columns=["content"],
    key="id",
)
```

This runs the encoder over every row in `corpus`, writes the vectors plus
the `key` column to a Parquet result table, and builds a USEARCH ANN
sidecar. The job is checkpointed — interrupting and re-running picks up
where it left off.

`model` accepts:

- `local:/abs/path/to/model_dir` — a directory with `config.json`,
  `model.safetensors`, `tokenizer.json` (the fixture under `cookbook/fixtures/tiny_bert`)
- `hf:<user>/<model>` or a bare Hugging Face Hub ID like
  `sentence-transformers/all-MiniLM-L6-v2`
- `onnx:/abs/path/to/model.onnx` for an ONNX backend

## Encode the query

```python
query_vec = db.encode_text_query(MODEL, "how does quantum computing work?")
```

`encode_text_query` runs the same model that built the index and returns a
`list[float]`. The dimension must match the index — 32 for `tiny_bert`,
384 for `all-MiniLM-L6-v2`, and so on.

## Search

```python
results = db.search("corpus", query=query_vec, k=3).run()
for row in results.to_pylist():
    print(f"id={row['_row_id']}  score={row['similarity']:.4f}  {row['title']}")
```

`db.search` returns a `SearchBuilder`. Call `.run()` to execute, or chain
`.filter("year > 2020")`, `.sort("similarity", descending=True)`,
`.limit(n)`, `.select([...])`, `.join("other_source", on="...")`,
`.annotate(model=..., task=..., columns=...)` first.

Every result row carries:

- the original columns from `corpus`
- `_row_id` — the deduplicated row identifier
- `similarity` — cosine similarity to the query (1.0 = identical)
- `retrieved_by` — provenance JSON: which model + which index served this
  row. The same column shows up for every annotate/join you add.

## Run the full script

```bash
python cookbook/quickstart/quickstart.py
```

That's the 5-minute path. From here, jump to a recipe that matches your
workload — `mutable_tables`, `trigger_streams`, `eval_embeddings`,
`eval_inference`, `fine_tune`, or `flight_sql`.
