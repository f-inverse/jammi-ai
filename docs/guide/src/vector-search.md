# Vector Search

`search()` performs ANN vector similarity search over embedding tables and returns results with structured evidence provenance.

## Basic search

```rust
use std::sync::Arc;

let session = Arc::new(InferenceSession::new(config).await?);

// Encode a query
let query = session.encode_query(
    "sentence-transformers/all-MiniLM-L6-v2",
    "quantum computing applications",
).await?;

// Search — returns top 10 results with all source columns
let results = session.search("patents", query, 10).await?
    .run().await?;
```

`search()` requires `Arc<InferenceSession>` because the `SearchBuilder` lives across async boundaries.

## What search returns

Search results are `Vec<RecordBatch>` with:

- **All original source columns** (e.g., `id`, `title`, `abstract`, `year`, `assignee_id`) — automatically hydrated by joining ANN results back to the source table
- **`_row_id`** — the source key as Utf8
- **`_source_id`** — which source the row came from
- **`similarity`** — cosine similarity score (1.0 = identical, 0.0 = orthogonal), descending
- **`retrieved_by`** — `List<Utf8>` provenance: which channels found this row (e.g., `["vector"]`)
- **`annotated_by`** — `List<Utf8>` provenance: which channels added evidence post-retrieval

## SearchBuilder fluent API

The `SearchBuilder` lets you compose query operations. Each method adds a node to a DataFusion execution plan — no data is processed until `.run()`.

### Sort

```rust
session.search("patents", query, 20).await?
    .sort("similarity", true)?   // descending
    .run().await?
```

### Limit

```rust
session.search("patents", query, 20).await?
    .sort("similarity", true)?
    .limit(5)                     // top 5 only
    .run().await?
```

### Filter

```rust
session.search("patents", query, 20).await?
    .filter("year > 2020")?      // SQL predicate on hydrated columns
    .run().await?
```

### Join

Join search results with another registered source:

```rust
session.add_source("assignees", SourceType::Local, SourceConnection {
    url: Some("file:///path/to/assignees.csv".into()),
    format: Some(FileFormat::Csv),
    ..Default::default()
}).await?;

let results = session.search("patents", query, 10).await?
    .join("assignees", "assignee_id=id", None).await?  // left join by default
    .run().await?;
// Results now include company_name, country from assignees
```

The `on` parameter is `"left_col=right_col"`. The optional third parameter is `"inner"` or `"left"` (default).

### Annotate

Run model inference over search results to add new columns:

```rust
let results = session.search("patents", query, 10).await?
    .annotate(
        "sentence-transformers/all-MiniLM-L6-v2",
        "embedding",
        &["abstract".to_string()],
    ).await?
    .run().await?;
// annotated_by = ["inference"], retrieved_by = ["vector"]
```

### Select

Project specific columns:

```rust
let results = session.search("patents", query, 10).await?
    .select(&["_row_id".into(), "title".into(), "similarity".into()])?
    .run().await?;
```

### Chaining

All operations compose:

```rust
let results = session.search("patents", query, 100).await?
    .join("assignees", "assignee_id=id", None).await?
    .filter("country = 'US'")?
    .sort("similarity", true)?
    .limit(10)
    .select(&["title".into(), "company_name".into(), "similarity".into()])?
    .run().await?;
```

## Evidence provenance

Every search result carries provenance tracking:

| Scenario | `retrieved_by` | `annotated_by` |
|----------|---------------|----------------|
| Plain search | `["vector"]` | `[]` |
| Search + annotate | `["vector"]` | `["inference"]` |

These are `List<Utf8>` columns — each row has its own list of contributing channels.

## Embedding table resolution

When multiple embedding tables exist for a source, `search()` uses the most recently created "ready" table. The resolution order:

1. Explicit table name (if provided)
2. Latest ready embedding table for the source (by `created_at`)
3. Error if no embedding table exists

## ANN vs exact search

Search automatically selects the best path:

- **ANN (fast)** — when the sidecar index files (`.usearch` + `.rowmap` + `.manifest.json`) exist and load successfully
- **Exact (brute-force)** — fallback when sidecar files are missing or corrupt

The caller never knows the difference. Deleting sidecar files degrades performance but not correctness.
