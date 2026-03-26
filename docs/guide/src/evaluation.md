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

```rust
let metrics = session.eval_embeddings(
    "patents",
    None,                              // use latest embedding table
    "golden.public.golden_relevance",  // golden dataset
    10,                                // k for recall@k, precision@k
).await?;

println!("recall@10:    {}", metrics["recall_at_k"]);
println!("precision@10: {}", metrics["precision_at_k"]);
println!("MRR:          {}", metrics["mrr"]);
println!("nDCG:         {}", metrics["ndcg"]);
```

### Python

```python
metrics = db.eval_embeddings(
    source="patents",
    golden_source="golden.public.golden_relevance",
    k=10,
)

print(f"recall@10:    {metrics['recall_at_k']:.3f}")
print(f"precision@10: {metrics['precision_at_k']:.3f}")
print(f"MRR:          {metrics['mrr']:.3f}")
print(f"nDCG:         {metrics['ndcg']:.3f}")
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

```rust
let comparison = session.eval_compare(
    &[base_table.clone(), finetuned_table.clone()],
    "patents",
    "golden.public.golden_relevance",
    10,
).await?;

let deltas = &comparison["delta"];
for (table, metrics) in deltas.as_object().unwrap() {
    println!("{table}: recall@10 delta {:+.3} ({:+.1}%)",
        metrics["recall_at_k"]["absolute"],
        metrics["recall_at_k"]["relative"].as_f64().unwrap() * 100.0,
    );
}
```

### Python

```python
comparison = db.eval_compare(
    embedding_tables=[base_table, finetuned_table],
    source="patents",
    golden_source="golden.public.golden_relevance",
    k=10,
)
print(comparison)
```

The first table is the baseline. Deltas (absolute and relative) are computed for all subsequent tables.

## Evaluate classification

### Rust

```rust
use jammi_ai::eval::EvalTask;

let metrics = session.eval_inference(
    "facebook/bart-large-mnli",
    "test_data",
    &["text".into()],
    EvalTask::Classification,
    "golden.public.labels",
    "category",
).await?;

println!("Accuracy: {}", metrics["accuracy"]);
println!("Macro F1: {}", metrics["f1"]);
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

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Macro F1: {metrics['f1']:.3f}")
```

## Eval runs in the catalog

Every evaluation is recorded automatically:

```rust
let runs = session.catalog().list_eval_runs()?;
for run in &runs {
    println!("{}: {} on {} (k={:?})", run.eval_run_id, run.eval_type, run.golden_source, run.k);
}
```

## Schema validation

Golden datasets are validated before evaluation starts. Missing or wrong-type columns produce clear errors:

```
Eval error: Golden dataset missing required column 'query_text'
Eval error: Golden dataset column 'query_id' has type Boolean, expected Utf8
```

Integer ID columns (Int32, Int64) are accepted where Utf8 is expected.
