# Assemble a Context Set for Conditioned Prediction

A prediction is often best made *conditioned on a neighbourhood*: not "what is
the label of this row" in the abstract, but "given the `k` most similar
labelled rows, what is the label of this row." That neighbourhood is a **context
set** — `C = {(xᵢ, yᵢ)}` — and in a database it is not an abstraction to invent.
It is a [search](./semantic-search.md) joined to its labels.

`assemble_context` makes that first-class. It retrieves a target's `k` nearest
neighbours, pairs them with their outcome columns, and pools the neighbour
vectors — permutation-invariantly — into one fixed-width *context vector* a
predictor can condition on. It is the encode-and-aggregate half of a Neural
Process; the decode half (a learned predictor over the representation) composes
on top.

## When this is the right tool

Reach for `assemble_context` when you want a **reusable set representation** of a
target's retrieved neighbourhood — a conditioning vector, a prototype/centroid,
a bag-of-evidence summary. If you only need to aggregate one specific, already
known set of rows once, that is a SQL `GROUP BY` you already have; this is the
operator that turns *any* target's retrieval into a representation,
reproducibly, with the leakage guards a prediction context needs.

## Assemble and encode

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# use jammi_ai::pipeline::context_set::{ContextRequest, SetAggregator};
# async fn ex(session: &std::sync::Arc<InferenceSession>, query: Vec<f32>) -> jammi_db::error::Result<()> {
let mut request = ContextRequest::new("patents", query, 10);
request.value_columns = vec!["category".into()];     // the labels carried per context row
request.aggregator = SetAggregator::Mean;             // mean | sum | max pooling

let context = session.assemble_context(&request).await?;

if let Some(vector) = &context.context_vector {
    // condition a predictor on `vector` + `context.context_size`
    let _ = (vector, context.context_size);
}
# Ok(()) }
```

The result carries:

- `context_vector` — the pooled `ρ(Σ φ(xᵢ))` representation, or `None` for an
  empty context (no neighbour survived the guards below — treat as
  low-confidence / fall back to the prior, never as a one-element average).
- `context_size` — the number of neighbours that entered the pool, carried
  **separately** so a decoder can use the count signal without it corrupting the
  pooled vector.
- `context_keys` — the context members' keys, in retrieval (descending
  similarity) order.
- `value_rows` — the requested `value_columns` of each member, in the same order.

## The leakage guards (on by default)

A target that retrieves **itself** as its own context trivially leaks the answer
when a value column is the prediction target. So:

- **`exclude_self`** defaults **on**. Set `exclude_key` to the target's own row
  key (when the query vector belongs to a stored row) and that same-key neighbour
  is dropped before pooling; the retrieval over-fetches by one so a self-hit
  never shrinks the context below `k`.
- **`split`** scopes the context to a train split. When the context feeds a
  training or evaluation target, pass e.g. `split = Some("split = 'train'".into())`
  so the target's own outcome stays held out — the same train/target line the
  evaluation harness enforces.

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# use jammi_ai::pipeline::context_set::ContextRequest;
# fn ex(query: Vec<f32>) {
let mut request = ContextRequest::new("patents", query, 10);
request.exclude_key = Some("US-1234567".into());   // drop the target's own row
request.split = Some("split = 'train'".into());     // context from the train split only
# let _ = request;
# }
```

## Pooling: fixed, permutation-invariant, deterministic

The encoder pools through the engine's vector-aggregation functions
(`vector_mean` / `vector_sum` / `vector_max`) — the *same* element-wise
aggregation the engine ships for grouped vector reduction. The pool is:

- **permutation-invariant** — shuffling the context rows yields a byte-identical
  vector (the aggregate folds with a commutative, associative operator);
- **deterministic** under exact retrieval — the pooled vector is reproducible
  across runs.

`mean` discards set size; `sum` encodes it; `max` is robust but lossy. None is
universally right, which is why the aggregator is a knob and `context_size` is
always carried alongside.

This is **fixed** pooling — the DeepSets / Conditional-Neural-Process
expressiveness ceiling. It cannot model *which* context element matters; learned
attention pooling (the AttnCNP point on the spectrum) is a separate, downstream
capability, not a silent extension of this one.

## Materialise for batch workflows

For batch pipelines, pool every target once and land the results as a normal
embedding-shaped result table — searchable and joinable like any other embedding
table, with its own sidecar ANN index:

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# use jammi_ai::pipeline::context_set::MaterializedContext;
# async fn ex(session: &std::sync::Arc<InferenceSession>, rows: Vec<(String, Vec<f32>)>) -> jammi_db::error::Result<()> {
let table = session
    .materialize_context("patents", MaterializedContext { rows: &rows, dimensions: 32 })
    .await?;
// `table` is a normal embedding result table: search it, join it, index it.
let _ = table;
# Ok(()) }
```

Each target's key becomes the table's `_row_id`; the pooled context vector
becomes its `vector`. A materialised context set is a first-class member of the
same table family every embedding table belongs to.
