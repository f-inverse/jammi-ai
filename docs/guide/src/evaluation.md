# Evaluate and Compare Models

Measure embedding quality and classification accuracy against golden datasets. Results are recorded in the catalog for tracking over time.

## Prepare a golden dataset

A golden dataset is any registered source with the right columns. No special format required.

### Retrieval golden set

```csv
query_id,query_text,relevant_id
q1,quantum computing applications,1
q1,quantum computing applications,4
q2,machine learning for science,2
```

| Column | Type | Required |
|--------|------|----------|
| `query_id` | Utf8 | yes |
| `query_text` | Utf8 | yes |
| `relevant_id` | Utf8 or Int | yes |
| `relevance_grade` | Int32 | no (default: 1 = binary) |

Register it as a source:

```python
db.add_source("golden", path="/data/golden_relevance.csv", format="csv")
```

## Evaluate embedding quality

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# async fn ex(session: &InferenceSession) -> jammi_db::error::Result<()> {
let report = session.eval_embeddings(
    "patents",
    None,                                // use latest embedding table
    "golden.public.golden_relevance",    // golden dataset
    10,                                  // k for recall@k, precision@k
    &std::collections::HashMap::new(),   // no cohort tags
).await?;

println!("recall@10:    {}", report.aggregate.recall_at_k);
println!("precision@10: {}", report.aggregate.precision_at_k);
println!("MRR:          {}", report.aggregate.mrr);
println!("nDCG:         {}", report.aggregate.ndcg);
# Ok(()) }
```

### Python

```python
metrics = db.eval_embeddings(
    source="patents",
    golden_source="golden.public.golden_relevance",
    k=10,
)

agg = metrics["aggregate"]
print(f"recall@10:    {agg['recall_at_k']:.3f}")
print(f"precision@10: {agg['precision_at_k']:.3f}")
print(f"MRR:          {agg['mrr']:.3f}")
print(f"nDCG:         {agg['ndcg']:.3f}")
```

### Per-query drill-down

The report also carries a `per_query` array — one record per golden-set query, in golden order. This is what sample-based statistical rules (Welch's t, Mann-Whitney U) consume at gate time.

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# async fn ex(session: &InferenceSession) -> jammi_db::error::Result<()> {
# let report = session.eval_embeddings("patents", None, "golden.public.golden_relevance", 10, &std::collections::HashMap::new()).await?;
for record in &report.per_query {
    println!("{}: recall={:.3} ndcg={:.3}",
        record.query_id, record.metrics.recall, record.metrics.ndcg);
}
# Ok(()) }
```

```python
for record in metrics["per_query"]:
    m = record["metrics"]
    print(f"{record['query_id']}: recall={m['recall']:.3f} ndcg={m['ndcg']:.3f}")
```

### Retrieval metrics

| Metric | What it measures |
|--------|-----------------|
| `recall_at_k` | Fraction of relevant docs found in top-k |
| `precision_at_k` | Fraction of top-k that are relevant |
| `mrr` | Reciprocal rank of the first relevant result |
| `ndcg` | Normalized discounted cumulative gain (uses graded relevance if provided) |

All metrics are in [0, 1]. Higher is better.

## Compare models (A/B)

Compare a base model against a fine-tuned model:

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# async fn ex(session: &InferenceSession, base_table: String, finetuned_table: String) -> jammi_db::error::Result<()> {
let comparison = session.eval_compare(
    &[base_table.clone(), finetuned_table.clone()],
    "patents",
    "golden.public.golden_relevance",
    10,
).await?;

// The first entry is the baseline (`delta: None`); every subsequent entry
// carries a delta against it.
for entry in comparison.per_table.iter().skip(1) {
    let delta = entry.delta.as_ref().expect("non-baseline entries carry a delta");
    println!(
        "{}: recall@10 delta {:+.3} ({:+.1}%)",
        entry.table_name,
        delta.recall_at_k.absolute,
        delta.recall_at_k.relative * 100.0,
    );
}
# Ok(()) }
```

### Python

```python
comparison = db.eval_compare(
    embedding_tables=[base_table, finetuned_table],
    source="patents",
    golden_source="golden.public.golden_relevance",
    k=10,
)
# `per_table[0]` is the baseline (`delta` is None); subsequent entries
# carry a `delta` dict keyed by metric name (recall_at_k, precision_at_k,
# mrr, ndcg) with `absolute` and `relative` sub-keys.
for entry in comparison["per_table"][1:]:
    d = entry["delta"]["recall_at_k"]
    print(f"{entry['table_name']}: recall@10 delta {d['absolute']:+.3f} ({d['relative']*100:+.1f}%)")
```

The first table is the baseline. Deltas (absolute and relative) are computed for all subsequent tables.

## Evaluate classification

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# async fn ex(session: &InferenceSession) -> jammi_db::error::Result<()> {
use jammi_ai::eval::{EvalTask, InferenceAggregate};

let report = session.eval_inference(
    "facebook/bart-large-mnli",
    "test_data",
    &["text".into()],
    EvalTask::Classification,
    "golden.public.labels",
    "category",
).await?;

match &report.aggregate {
    InferenceAggregate::Classification(c) => {
        println!("Accuracy: {}", c.accuracy);
        println!("Macro F1: {}", c.f1);
    }
    InferenceAggregate::Ner(n) => {
        println!("NER F1: {}", n.f1);
    }
}
println!("per_record predictions: {}", report.per_record.len());
# Ok(()) }
```

### Python

```python
metrics = db.eval_inference(
    model="facebook/bart-large-mnli",
    source="test_data",
    columns=["text"],
    task="classification",
    golden_source="golden.public.labels",
    label_column="category",
)

# `aggregate` is tagged by `task`; for classification it carries
# `accuracy`, `f1`, and `per_class`.
agg = metrics["aggregate"]
print(f"Accuracy: {agg['accuracy']:.3f}")
print(f"Macro F1: {agg['f1']:.3f}")
# `per_record` is one entry per aligned predicted/gold pair.
print(f"per_record predictions: {len(metrics['per_record'])}")
```

## Eval runs in the catalog

Every evaluation is recorded automatically:

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# async fn ex(session: &InferenceSession) -> jammi_db::error::Result<()> {
let runs = session.catalog().list_eval_runs().await?;
for run in &runs {
    println!("{}: {} on {} (k={:?})", run.eval_run_id, run.eval_type, run.golden_source, run.k);
}
# Ok(()) }
```

## Schema validation

Golden datasets are validated before evaluation starts. Missing or wrong-type columns produce clear errors:

```text
Eval error: Golden dataset missing required column 'query_text'
Eval error: Golden dataset column 'query_id' has type Boolean, expected Utf8
```

Integer ID columns (Int32, Int64) are accepted where Utf8 is expected.
