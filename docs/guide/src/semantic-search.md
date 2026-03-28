# Semantic Search

Perform ANN vector similarity search over embedding tables. Results include all original source columns, similarity scores, and evidence provenance.

## Basic search

### Rust

```rust
use std::sync::Arc;

let session = Arc::new(InferenceSession::new(config).await?);

// Encode a query
let query = session.encode_text_query(
    "sentence-transformers/all-MiniLM-L6-v2",
    "quantum computing applications",
).await?;

// Search — returns top 10 results
let results = session.search("patents", query, 10).await?
    .run().await?;
```

### Python

```python
query_vec = db.encode_text_query("sentence-transformers/all-MiniLM-L6-v2", "quantum computing applications")

search = db.search("patents", query=query_vec, k=10)
results = search.run()
print(results.to_pandas())
```

## What search returns

Results are `RecordBatch` / `pyarrow.Table` with:

- All original source columns (e.g., `id`, `title`, `abstract`, `year`)
- `_row_id` — the source key
- `_source_id` — which source the row came from
- `similarity` — cosine similarity score (1.0 = identical, 0.0 = orthogonal)
- `retrieved_by` — `List<Utf8>` provenance: which channels found this row
- `annotated_by` — `List<Utf8>` provenance: which channels added evidence post-retrieval

## SearchBuilder

The SearchBuilder composes query operations. Each method adds a node to a DataFusion execution plan — no data is processed until `.run()`.

### Filter

Apply a SQL WHERE clause to hydrated columns:

#### Rust

```rust
session.search("patents", query, 20).await?
    .filter("year > 2020")?
    .run().await?
```

#### Python

```python
search = db.search("patents", query=query_vec, k=20)
search.filter("year > 2020")
results = search.run()
```

### Sort

#### Rust

```rust
session.search("patents", query, 20).await?
    .sort("similarity", true)?  // descending
    .run().await?
```

#### Python

```python
search = db.search("patents", query=query_vec, k=20)
search.sort("similarity", descending=True)
results = search.run()
```

### Limit

#### Rust

```rust
session.search("patents", query, 20).await?
    .sort("similarity", true)?
    .limit(5)
    .run().await?
```

#### Python

```python
search = db.search("patents", query=query_vec, k=20)
search.sort("similarity", descending=True)
search.limit(5)
results = search.run()
```

### Select

Project specific columns. Note that `retrieved_by` and `annotated_by` evidence columns are always appended to the output regardless of your selection:

#### Rust

```rust
session.search("patents", query, 10).await?
    .select(&["_row_id".into(), "title".into(), "similarity".into()])?
    .run().await?
```

#### Python

```python
search = db.search("patents", query=query_vec, k=10)
search.select(["_row_id", "title", "similarity"])
results = search.run()
```

### Chaining everything

#### Rust

```rust
let results = session.search("patents", query, 100).await?
    .filter("year > 2020")?
    .sort("similarity", true)?
    .limit(10)
    .select(&["title".into(), "similarity".into()])?
    .run().await?;
```

#### Python

```python
search = db.search("patents", query=query_vec, k=100)
search.filter("year > 2020")
search.sort("similarity", descending=True)
search.limit(10)
search.select(["title", "similarity"])
results = search.run()
```

## ANN vs exact search

Search automatically selects the best path:

- **ANN (fast)** — when sidecar index files (`.usearch` + `.rowmap` + `.manifest.json`) exist and load successfully
- **Exact (brute-force)** — fallback when sidecar files are missing or corrupt

The caller never knows the difference. Deleting sidecar files degrades performance but not correctness.

## Embedding table resolution

When multiple embedding tables exist for a source, search uses the most recently created "ready" table. The resolution order:

1. Explicit table name (if provided)
2. Latest ready embedding table for the source (by `created_at`)
3. Error if no embedding table exists
