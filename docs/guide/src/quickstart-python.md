# Quickstart: Python

This walkthrough registers a local data file, runs a SQL query, generates embeddings, and performs a semantic search.

## Full example

```python
import jammi

# 1. Connect
db = jammi.connect(gpu_device=-1)  # CPU-only; omit for GPU

# 2. Register a data source
db.add_source("patents", path="patents.parquet", format="parquet")

# 3. Query with SQL
table = db.sql("SELECT id, title, year FROM patents.public.patents WHERE year > 2020 LIMIT 5")
print(table.to_pandas())

# 4. Generate embeddings
db.generate_text_embeddings(
    source="patents",
    model="sentence-transformers/all-MiniLM-L6-v2",
    columns=["title"],
    key="id",
)

# 5. Semantic search
query_vec = db.encode_text_query("sentence-transformers/all-MiniLM-L6-v2", "quantum computing applications")

search = db.search("patents", query=query_vec, k=5)
search.sort("similarity", descending=True)
results = search.run()

print(results.to_pandas())
```

## What's happening

1. **`jammi.connect()`** creates a `Database` backed by an `InferenceSession` with a shared tokio runtime
2. **`add_source`** registers a local file — Parquet, CSV, and JSON are supported
3. **`sql`** returns a `pyarrow.Table` (zero-copy from Rust via pyo3-arrow)
4. **`generate_text_embeddings`** runs the model and persists vectors to Parquet with an ANN index
5. **`encode_text_query`** encodes text into the same vector space, returns `list[float]`
6. **`search`** returns a `SearchBuilder` — call methods to filter, sort, join, then `.run()` to execute

## SearchBuilder

The Python `SearchBuilder` uses imperative-style calls (each method mutates in place):

```python
search = db.search("patents", query=query_vec, k=20)
search.filter("year >= 2020")
search.sort("similarity", descending=True)
search.limit(5)
search.select(["id", "title", "similarity"])
results = search.run()  # pyarrow.Table
```

After `.run()`, the builder is consumed and cannot be reused.

## Working with results

All query and search methods return `pyarrow.Table`:

```python
table = db.sql("SELECT * FROM patents.public.patents LIMIT 10")

# Convert to pandas
df = table.to_pandas()

# Access columns directly
titles = table.column("title").to_pylist()

# Metadata
print(table.column_names)
print(table.num_rows)
```

## Next steps

- [Query Your Data with SQL](./query-data.md) — SQL features, joins, aggregations
- [Generate Embeddings](./generate-embeddings.md) — persistence, multiple models, crash recovery
- [Semantic Search](./semantic-search.md) — filtering, joins, annotations, evidence provenance
