# Enrich Results with Joins and Annotations

Search results can be enriched by joining with other data sources and annotating with additional model inference.

There are two surfaces for this, and they are deliberately different:

- **`search`** is the bounded, jammi-defined primitive: nearest-neighbor top-k with optional `filter`/`select`, returning a table directly. Same call, same shape, embedded or remote.
- **Compound query** — open, caller-shaped composition (`join` / `filter` / `select` and model inference over the results) — rides **SQL**. In Rust the fluent `QueryBuilder` (returned by `session.search(...)`) builds the same plan in-process; in Python and over the wire the surface is `db.sql(...)`, where the `annotate(...)` **table function** runs a model over a relation. Both descend through the one inference operator, so an in-process query and a Flight-SQL query run the same plan node.

The fluent Rust builder tracks every enrichment step in the evidence provenance columns (`retrieved_by` / `annotated_by`).

## Join with another source

Join search results with a registered source to add context columns (e.g., company name, category labels):

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use std::sync::Arc;
# use jammi_ai::session::InferenceSession;
# use jammi_db::source::{FileFormat, SourceConnection, SourceType};
# async fn ex(session: &Arc<InferenceSession>, query: Vec<f32>) -> jammi_db::error::Result<()> {
session.add_source("assignees", SourceType::File, SourceConnection {
    url: Some("file:///data/assignees.csv".into()),
    format: Some(FileFormat::Csv),
    ..Default::default()
}).await?;

let results = session.search("patents", query, 10).await?
    .join("assignees", "assignee_id=id", None).await?  // left join by default
    .run().await?;
// Results now include company_name, country from assignees
# Ok(()) }
```

### Python

In Python the compound query is SQL. `search` returns a `pyarrow.Table` directly; to join, run SQL that the engine plans (in-process for the embed wheel, over Flight SQL for a remote engine — same SQL either way):

```python
db.add_source("assignees", path="/data/assignees.csv", format="csv")

results = db.sql("""
    SELECT p.title, a.company_name, a.country
    FROM patents.public.patents AS p
    JOIN assignees.public.assignees AS a ON p.assignee_id = a.id
""")
# Results now include company_name, country from assignees
```

In Rust, the fluent builder's `on` parameter is `"left_col=right_col"` and the optional join type is `"inner"` or `"left"` (default).

## Annotate with model inference

Run a model over search results to add new columns:

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use std::sync::Arc;
# use jammi_ai::session::InferenceSession;
use jammi_db::ModelTask;
# async fn ex(session: &Arc<InferenceSession>, query: Vec<f32>) -> jammi_db::error::Result<()> {
let results = session.search("patents", query, 10).await?
    .annotate(
        "sentence-transformers/all-MiniLM-L6-v2",
        ModelTask::TextEmbedding,
        &["abstract".to_string()],
    ).await?
    .run().await?;
# Ok(()) }
```

### Python

`annotate(model, task, relation, key_column, content_column, …)` is a SQL **table function**: it runs the model over the named relation's columns and returns the inference output (`_row_id` keyed from `key_column`, plus the task's columns — e.g. `vector`). Join it back to the source on `_row_id` to enrich:

```python
results = db.sql("""
    SELECT p.title, a.vector
    FROM annotate('sentence-transformers/all-MiniLM-L6-v2', 'text_embedding',
                  'patents.public.patents', 'id', 'abstract') AS a
    JOIN patents.public.patents AS p ON a._row_id = arrow_cast(p.id, 'Utf8')
""")
```

The same SQL — the same `annotate` function — runs in-process (embed wheel) or over the Flight SQL lane (remote engine, `jammi-client`), so compound retrieval + inference is one round-trip.

## Evidence provenance

Every search result carries provenance tracking that records how each row was found and enriched:

| Scenario | `retrieved_by` | `annotated_by` |
|----------|---------------|----------------|
| Plain search | `["vector"]` | `[]` |
| Search + annotate | `["vector"]` | `["inference"]` |

These are `List<Utf8>` columns — each row has its own list of contributing channels.

## Composing everything

All operations compose freely:

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use std::sync::Arc;
# use jammi_ai::session::InferenceSession;
use jammi_db::ModelTask;
# async fn ex(session: &Arc<InferenceSession>, query: Vec<f32>) -> jammi_db::error::Result<()> {
let results = session.search("patents", query, 100).await?
    .join("assignees", "assignee_id=id", None).await?
    .annotate("all-MiniLM-L6-v2", ModelTask::TextEmbedding, &["abstract".into()]).await?
    .filter("country = 'US'")?
    .sort("similarity", true)?
    .limit(10)
    .select(&["title".into(), "company_name".into(), "similarity".into()])?
    .run().await?;
# Ok(()) }
```

### Python

The compound query is SQL, so everything composes as `JOIN` / `WHERE` / `ORDER BY` / `LIMIT` / projection over `annotate(...)` and registered sources:

```python
results = db.sql("""
    SELECT p.title, a.company_name, ann.vector
    FROM annotate('all-MiniLM-L6-v2', 'text_embedding',
                  'patents.public.patents', 'id', 'abstract') AS ann
    JOIN patents.public.patents  AS p ON ann._row_id = arrow_cast(p.id, 'Utf8')
    JOIN assignees.public.assignees AS a ON p.assignee_id = a.id
    WHERE a.country = 'US'
    LIMIT 10
""")
```

Each surface plans a DataFusion execution plan under the hood. No data is processed until the result is read.
