# Semantic Search

> **Measured companion:** for the long-form, executed-and-measured Python treatment, see [The Cookbook → Retrieval](https://f-inverse.github.io/jammi-ai/cookbook/chapters/10-retrieval/retrieval.html).

Perform ANN vector similarity search over embedding tables. Results include all original source columns, similarity scores, and evidence provenance.

## Basic search

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# use jammi_db::config::JammiConfig;
# async fn ex(config: JammiConfig) -> jammi_db::error::Result<()> {
use std::sync::Arc;

let session = Arc::new(InferenceSession::new(config).await?);

// Encode a query
let query = session.encode_text_query(
    "sentence-transformers/all-MiniLM-L6-v2",
    "quantum computing applications",
).await?;

// Search — returns top 10 results
let results = session.search("patents", query, 10, None, None).await?
    .run().await?;
# Ok(()) }
```

### Python

```python
query_vec = db.encode_query(model="sentence-transformers/all-MiniLM-L6-v2", query="quantum computing applications")

results = db.search("patents", query=query_vec, k=10)  # pyarrow.Table
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

## Refining a search

`search` carries the two knobs the bounded primitive owns directly: a SQL `filter`
predicate over the hydrated columns and a `select` column projection. In Python they
are keyword arguments and `search` returns the table: with a `filter`, the `k`
nearest rows that satisfy it (fewer only when fewer exist) — the engine widens the
ranked candidates until `k` rows pass, ending in an exact search over the whole
table. In Rust they are methods on the fluent `QueryBuilder` (`session.search(...)`
returns the builder, which also carries `sort` / `limit` / `join` / `annotate` and a
`.run()`); there `filter` composes over the `k` rows the search ranked, as every
builder step does.

### Filter and select

#### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# async fn ex(session: &std::sync::Arc<InferenceSession>, query: Vec<f32>) -> jammi_db::error::Result<()> {
session.search("patents", query, 20, None, None).await?
    .filter("year > 2020")?
    .sort("similarity", true)?  // descending
    .limit(5)
    .select(&["_row_id".into(), "title".into(), "similarity".into()])?
    .run().await?;
# Ok(()) }
```

#### Python

```python
results = db.search(
    "patents", query=query_vec, k=20,
    filter="year > 2020",
    select=["_row_id", "title", "similarity"],
)  # pyarrow.Table
```

### Compound query (join, annotate)

Joining other sources and running a model over the results is open composition, so
in Python and over the wire it is SQL — `db.sql(...)`, with the `annotate(...)` table
function for inference. In Rust the same operations compose on the fluent builder:

#### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# async fn ex(session: &std::sync::Arc<InferenceSession>, query: Vec<f32>) -> jammi_db::error::Result<()> {
let results = session.search("patents", query, 100, None, None).await?
    .filter("year > 2020")?
    .sort("similarity", true)?
    .limit(10)
    .select(&["title".into(), "similarity".into()])?
    .run().await?;
# Ok(()) }
```

#### Python

```python
results = db.sql("""
    SELECT title, vector
    FROM annotate('local:/models/all-MiniLM-L6-v2', 'text_embedding',
                  'patents.public.patents', 'id', 'abstract') AS a
    JOIN patents.public.patents AS p ON a._row_id = arrow_cast(p.id, 'Utf8')
    WHERE p.year > 2020
    LIMIT 10
""")
```

See [Compound Retrieval and Inference over Flight SQL](./remote-compound-query.md) for the full compound surface — it runs the same SQL in-process or against a remote engine over Flight SQL.

## Approximate vs exact search

By default a search is **approximate**: it walks the table's ANN sidecar
index (`.usearch` + `.rowmap` + `.manifest.json`), and falls back to scoring
every vector when the sidecar is missing or corrupt — deleting sidecar files
degrades speed, never correctness. On a quantized table (`int8` / `binary`
`storage_precision`) the index retrieves `k * oversample` candidates and
rescores them exactly; `oversample` widens that breadth for one call.

An **exact** search scores every vector and returns the true nearest
neighbours — the baseline an approximate search's recall is measured
against:

```python
approximate = db.search("patents", query=vector, k=10)
exact = db.search("patents", query=vector, k=10, exact=True)
recall = len(set(approximate.column("_row_id").to_pylist())
             & set(exact.column("_row_id").to_pylist())) / 10
```

In Rust the choice is a `SearchMethod`: `SearchMethod::Approximate {
oversample }` or `SearchMethod::Exact`. An exact search takes no oversample.

## Embedding table resolution

When multiple embedding tables exist for a source, search uses the most recently created "ready" table. The resolution order:

1. Explicit table name (if provided)
2. Latest ready embedding table for the source (by `created_at`)
3. Error if no embedding table exists

## Search over gRPC (edge runtimes)

`EmbeddingService` exposes `Search` on the typed gRPC surface, so a process that reaches the engine over gRPC-web — an edge function that cannot speak Flight SQL's bidirectional HTTP/2 — can run the same similarity search it already uses for `AddSource`, `GenerateAudioEmbeddings`, and `EncodeAudioQuery`. It is the same engine capability on an additional transport, not a second search path.

A `SearchRequest` carries the source, a `k`, an optional SQL `filter` (the `k` nearest rows that satisfy it), and an optional `select` column list. The query is a `oneof`:

- **`query_vector`** — a precomputed vector. The usual flow is encode-then-search: call `EncodeAudioQuery` (or any client-side encoder) to get the vector, then feed it back as the query.
- **`row_key`** — query-by-example. The engine resolves that row's stored vector **internally** and ranks by it ("rows like this row"). The vector never crosses the wire.

```text
// encode-then-search
embedding = EncodeAudioQuery{ model_id, audio_bytes }.embedding
result    = Search{ source_id, query_vector: { values: embedding }, k: 10 }.result

// query-by-example (no re-encode round-trip; vector stays in the engine)
result    = Search{ source_id, row_key: "clip_1", k: 10 }.result
```

`result` is the hydrated rows as one Arrow IPC stream — the key (`_row_id`), the source's columns (or exactly the `select`ed ones), the retrieval provenance, and the `similarity` — typed as the engine typed them, the same table `db.search` returns. `method` chooses an exact search (`exact`) or overrides the approximate search's `oversample`; unset is an approximate search at the table's own oversample.
