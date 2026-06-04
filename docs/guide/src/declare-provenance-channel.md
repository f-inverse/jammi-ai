# Declare a Custom Provenance Channel

Every row that flows through Jammi carries provenance — `retrieved_by` and `annotated_by` lists that record *how* the row was found and *what was added after retrieval*. Jammi ships two built-in channels — `vector` (declares `similarity`) and `inference` (declares `inference_model`, `inference_task`, `inference_confidence`) — but the catalog accepts any channel a consumer wants to register. Each channel declares the columns it contributes; the engine merges those columns into every result `RecordBatch` at query time.

This recipe walks through registering a third channel, `scored_by`, for a multi-stage retrieval pipeline where a federated reranker rescores the vector hits. The same shape applies to any non-built-in provenance signal: a citation graph, an attribution chain, a quality-grading pass.

## Setup

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate arrow;
# extern crate tokio;
# use jammi_db::config::JammiConfig;
# async fn ex(config: JammiConfig) -> jammi_db::error::Result<()> {
use std::sync::Arc;
use arrow::array::{ArrayRef, Float32Array, StringArray};
use jammi_ai::evidence::{merge_channels, ChannelContribution};
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::channel_repo::{ChannelColumn, ChannelColumnType, ChannelSpec};
use jammi_db::ChannelId;
use jammi_db::source::{FileFormat, SourceConnection, SourceType};

let session = Arc::new(InferenceSession::new(config).await?);
session.add_source("patents", SourceType::File, SourceConnection {
    url: Some("file:///data/patents.parquet".into()),
    format: Some(FileFormat::Parquet),
    ..Default::default()
}).await?;
# Ok(()) }
```

### Python

```python
import jammi_ai

db = jammi_ai.connect("file:///var/lib/jammi")
db.add_source("patents", path="/data/patents.parquet", format="parquet")
```

## Declare the channel

Channel declarations are catalog rows. Each declared column has a name, an Arrow type, and an ordinal. The set is append-only — once `ranker: Utf8` is declared, the engine refuses to redeclare it as `Int32` or drop it.

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use std::sync::Arc;
# use jammi_ai::session::InferenceSession;
# use jammi_db::catalog::channel_repo::{ChannelColumn, ChannelColumnType, ChannelSpec};
# use jammi_db::ChannelId;
# async fn ex(session: &Arc<InferenceSession>) -> jammi_db::error::Result<()> {
session.catalog().channels().register(&ChannelSpec {
    id: ChannelId::new("scored_by")?,
    priority: 3,
    columns: vec![
        ChannelColumn {
            name: "ranker".into(),
            data_type: ChannelColumnType::Utf8,
        },
        ChannelColumn {
            name: "rank_score".into(),
            data_type: ChannelColumnType::Float32,
        },
    ],
}).await?;
# Ok(()) }
```

### Python

```python
db.register_channel(
    "scored_by",
    priority=3,
    columns=[("ranker", "Utf8"), ("rank_score", "Float32")],
)
```

`priority` controls the order columns appear in the merged output — `vector` (1) and `inference` (2) come first, then `scored_by` (3).

To add more columns to an already-registered channel:

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use std::sync::Arc;
# use jammi_ai::session::InferenceSession;
# use jammi_db::catalog::channel_repo::{ChannelColumn, ChannelColumnType};
# use jammi_db::ChannelId;
# async fn ex(session: &Arc<InferenceSession>) -> jammi_db::error::Result<()> {
session.catalog().channels().add_columns(
    &ChannelId::new("scored_by")?,
    &[ChannelColumn { name: "scored_at".into(), data_type: ChannelColumnType::Utf8 }],
).await?;
# Ok(()) }
```

```python
db.add_channel_columns("scored_by", columns=[("scored_at", "Utf8")])
```

`add_columns` is append-only by construction. Trying to redeclare `ranker` with the same or a different type returns `JammiError::EvidenceChannel(_)`.

## Use the channel

Build a `ChannelContribution` for each batch your reranker produces. The arrays must align 1:1 with the channel's declared columns (`ranker` first, `rank_score` second) and have the same length as the batch's row count.

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate arrow;
# extern crate tokio;
# use std::sync::Arc;
# use arrow::array::{ArrayRef, Float32Array, RecordBatch, StringArray};
# use jammi_ai::evidence::{merge_channels, ChannelContribution};
# use jammi_ai::session::InferenceSession;
# use jammi_db::ChannelId;
# fn rerank_scores(_batch: &RecordBatch) -> Vec<f32> { vec![] }
# async fn ex(session: &Arc<InferenceSession>, batches: Vec<RecordBatch>) -> jammi_db::error::Result<()> {
let scored_by = ChannelId::new("scored_by")?;
let vector = ChannelId::new("vector")?;

let mut contributions = Vec::with_capacity(batches.len());
for batch in &batches {
    let n = batch.num_rows();
    let ranker: ArrayRef = Arc::new(StringArray::from(vec!["bm25"; n]));
    let rank_score: ArrayRef = Arc::new(Float32Array::from(rerank_scores(batch)));
    contributions.push(vec![ChannelContribution {
        channel: scored_by.clone(),
        columns: vec![ranker, rank_score],
    }]);
}

let merged = merge_channels(
    session.catalog(),
    &batches,
    &[vector.clone(), scored_by.clone()],
    &[vector, scored_by],   // retrieved_by
    &[],                     // annotated_by
    &contributions,
).await?;
# Ok(()) }
```

## Verify

The merged output schema includes the declared columns. Rows where the channel did not supply a value carry NULL.

### Rust

```rust,no_run
# extern crate arrow;
# use arrow::array::RecordBatch;
# fn ex(merged: Vec<RecordBatch>) {
let schema = merged[0].schema();
assert!(schema.field_with_name("ranker").is_ok());
assert!(schema.field_with_name("rank_score").is_ok());

for batch in &merged {
    let ranker = batch.column_by_name("ranker").unwrap();
    println!("first ranker: {:?}", ranker);
}
# }
```

### Python

From the SQL surface, the declared columns show up in any query that touches the result table — Python sees them as plain Arrow columns:

```python
table = db.sql(
    "SELECT _row_id, similarity, ranker, rank_score FROM results LIMIT 3"
)
for row in table.to_pylist():
    print(row["ranker"], row["rank_score"])
```

## What you cannot do

The channel declaration is append-only. Once `scored_by` ships with `ranker: Utf8`, you cannot:

- Redeclare `ranker` as `Int32` — `add_columns` rejects with `JammiError::EvidenceChannel("channel 'scored_by': column 'ranker' was declared Utf8, cannot redeclare as Int32")`. From Python, the same call raises `RuntimeError` carrying the identical message:

  ```python
  db.add_channel_columns("scored_by", columns=[("ranker", "Int32")])
  # RuntimeError: channel 'scored_by': column 'ranker' was declared Utf8, cannot redeclare as Int32
  ```

- Add a second column with the same name — `add_columns` rejects with `JammiError::EvidenceChannel("channel 'scored_by': column 'ranker' already declared")`.
- Drop `ranker` from the channel — there is no `drop_column` method by design.

If a column needs to change shape, declare a new column under a new name and migrate consumers. This preserves byte-for-byte readability of any backing table or downstream artifact that already references the original column.
