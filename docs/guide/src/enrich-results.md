# Enrich Results with Joins and Annotations

Search results can be enriched by joining with other data sources and annotating with additional model inference. Every enrichment step is tracked in the evidence provenance columns.

## Join with another source

Join search results with a registered source to add context columns (e.g., company name, category labels):

### Rust

```rust
session.add_source("assignees", SourceType::Local, SourceConnection {
    url: Some("file:///data/assignees.csv".into()),
    format: Some(FileFormat::Csv),
    ..Default::default()
}).await?;

let results = session.search("patents", query, 10).await?
    .join("assignees", "assignee_id=id", None).await?  // left join by default
    .run().await?;
// Results now include company_name, country from assignees
```

### Python

```python
db.add_source("assignees", path="/data/assignees.csv", format="csv")

search = db.search("patents", query=query_vec, k=10)
search.join("assignees", on="assignee_id=id")
results = search.run()
# Results now include company_name, country from assignees
```

The `on` parameter is `"left_col=right_col"`. The optional join type is `"inner"` or `"left"` (default).

## Annotate with model inference

Run a model over search results to add new columns:

### Rust

```rust
let results = session.search("patents", query, 10).await?
    .annotate(
        "sentence-transformers/all-MiniLM-L6-v2",
        "embedding",
        &["abstract".to_string()],
    ).await?
    .run().await?;
```

### Python

```python
search = db.search("patents", query=query_vec, k=10)
search.annotate(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="embedding",
    columns=["abstract"],
)
results = search.run()
```

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

```rust
let results = session.search("patents", query, 100).await?
    .join("assignees", "assignee_id=id", None).await?
    .annotate("all-MiniLM-L6-v2", "embedding", &["abstract".into()]).await?
    .filter("country = 'US'")?
    .sort("similarity", true)?
    .limit(10)
    .select(&["title".into(), "company_name".into(), "similarity".into()])?
    .run().await?;
```

### Python

```python
search = db.search("patents", query=query_vec, k=100)
search.join("assignees", on="assignee_id=id")
search.annotate(model="all-MiniLM-L6-v2", task="embedding", columns=["abstract"])
search.filter("country = 'US'")
search.sort("similarity", descending=True)
search.limit(10)
search.select(["title", "company_name", "similarity"])
results = search.run()
```

The pipeline builds a DataFusion execution plan under the hood. No data is processed until `.run()` — so adding more steps doesn't cost anything until execution.
