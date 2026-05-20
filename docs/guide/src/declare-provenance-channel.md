# Declare a Custom Provenance Channel

Every row that flows through Jammi carries provenance — `retrieved_by` and `annotated_by` lists that record *how* the row was found and *what was added after retrieval*. Jammi ships two built-in channels — `vector` (declares `similarity`) and `inference` (declares `inference_model`, `inference_task`, `inference_confidence`) — but the catalog accepts any channel a consumer wants to register. Each channel declares the columns it contributes; the engine merges those columns into every result `RecordBatch` at query time.

This recipe walks through registering a third channel, `scored_by`, for a multi-stage retrieval pipeline where a federated reranker rescores the vector hits. The same shape applies to any non-built-in provenance signal: a citation graph, an attribution chain, a quality-grading pass.

## Setup

### Rust

```rust
use std::sync::Arc;
use arrow::array::{ArrayRef, Float32Array, StringArray};
use jammi_ai::evidence::{merge_channels, ChannelContribution};
use jammi_ai::session::InferenceSession;
use jammi_engine::catalog::channel_repo::{ChannelColumn, ChannelColumnType, ChannelSpec};
use jammi_engine::ChannelId;
use jammi_engine::source::{FileFormat, SourceConnection, SourceType};

let session = Arc::new(InferenceSession::new(config).await?);
session.add_source("patents", SourceType::Local, SourceConnection {
    url: Some("file:///data/patents.parquet".into()),
    format: Some(FileFormat::Parquet),
    ..Default::default()
}).await?;
```

## Declare the channel

Channel declarations are catalog rows. Each declared column has a name, an Arrow type, and an ordinal. The set is append-only — once `ranker: Utf8` is declared, the engine refuses to redeclare it as `Int32` or drop it.

### Rust

```rust
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
})?;
```

`priority` controls the order columns appear in the merged output — `vector` (1) and `inference` (2) come first, then `scored_by` (3).

To add more columns to an already-registered channel:

```rust
session.catalog().channels().add_columns(
    &ChannelId::new("scored_by")?,
    &[ChannelColumn { name: "scored_at".into(), data_type: ChannelColumnType::Utf8 }],
)?;
```

`add_columns` is append-only by construction. Trying to redeclare `ranker` with the same or a different type returns `JammiError::EvidenceChannel(_)`.

## Use the channel

Build a `ChannelContribution` for each batch your reranker produces. The arrays must align 1:1 with the channel's declared columns (`ranker` first, `rank_score` second) and have the same length as the batch's row count.

### Rust

```rust
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
)?;
```

## Verify

The merged output schema includes the declared columns. Rows where the channel did not supply a value carry NULL.

### Rust

```rust
let schema = merged[0].schema();
assert!(schema.field_with_name("ranker").is_ok());
assert!(schema.field_with_name("rank_score").is_ok());

for batch in &merged {
    let ranker = batch.column_by_name("ranker").unwrap();
    println!("first ranker: {:?}", ranker);
}
```

## What you cannot do

The channel declaration is append-only. Once `scored_by` ships with `ranker: Utf8`, you cannot:

- Redeclare `ranker` as `Int32` — `add_columns` rejects with `JammiError::EvidenceChannel("channel 'scored_by': column 'ranker' was declared Utf8, cannot redeclare as Int32")`.
- Add a second column with the same name — `add_columns` rejects with `JammiError::EvidenceChannel("channel 'scored_by': column 'ranker' already declared")`.
- Drop `ranker` from the channel — there is no `drop_column` method by design.

If a column needs to change shape, declare a new column under a new name and migrate consumers. This preserves byte-for-byte readability of any backing table or downstream artifact that already references the original column.
