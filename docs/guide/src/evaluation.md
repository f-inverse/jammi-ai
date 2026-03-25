# Evaluation

Measure embedding quality, classification accuracy, and summarization quality against golden datasets. Results are recorded in the catalog for tracking and model comparison.

## Golden datasets

A golden dataset is any registered source with the right columns. No special format — CSV, Parquet, or database table all work.

### Retrieval (for eval_embeddings)

| Column | Type | Required |
|--------|------|----------|
| `query_id` | Utf8 | yes |
| `query_text` | Utf8 | yes |
| `relevant_id` | Utf8 or Int | yes |
| `relevance_grade` | Int32 | no (default: 1 = binary) |

```
query_id,query_text,relevant_id
q1,quantum computing applications,1
q1,quantum computing applications,4
q2,machine learning for science,2
```

### Classification (for eval_inference)

| Column | Type | Required |
|--------|------|----------|
| `id` | Utf8 or Int | yes |
| `label` | Utf8 | yes |

### Summarization (for eval_inference)

| Column | Type | Required |
|--------|------|----------|
| `id` | Utf8 or Int | yes |
| `reference` | Utf8 | yes |

Register the golden dataset as a source:

```rust
session.add_source("golden", SourceType::Local, SourceConnection {
    url: Some("file:///path/to/golden_relevance.csv".into()),
    format: Some(FileFormat::Csv),
    ..Default::default()
}).await?;
```

## Evaluate embedding quality

```rust
let metrics = session.eval_embeddings(
    "patents",                           // source with embeddings
    None,                                // use latest embedding table
    "golden.public.golden_relevance",    // golden dataset (schema.table)
    10,                                  // k for recall@k, precision@k
).await?;

println!("recall@10:    {}", metrics["recall_at_k"]);
println!("precision@10: {}", metrics["precision_at_k"]);
println!("MRR:          {}", metrics["mrr"]);
println!("nDCG:         {}", metrics["ndcg"]);
```

All metrics are in [0, 1]. Higher is better.

### Retrieval metrics

| Metric | What it measures |
|--------|-----------------|
| `recall_at_k` | Fraction of relevant docs found in top-k |
| `precision_at_k` | Fraction of top-k that are relevant |
| `mrr` | Reciprocal rank of the first relevant result |
| `ndcg` | Normalized discounted cumulative gain (uses graded relevance if provided) |

For each query in the golden set: the query text is encoded through the same model that generated the embeddings, ANN search is performed, and retrieved IDs are compared against the golden relevance judgments.

## Compare models

Compare a base model against a fine-tuned model (or any two embedding tables):

```rust
let comparison = session.eval_compare(
    &[base_table.clone(), finetuned_table.clone()],
    "patents",
    "golden.public.golden_relevance",
    10,
).await?;

println!("Baseline: {}", comparison["baseline"]);

let deltas = &comparison["delta"];
for (table, metrics) in deltas.as_object().unwrap() {
    println!("{table}:");
    println!("  recall@10 delta: {:+.3} ({:+.1}%)",
        metrics["recall_at_k"]["absolute"],
        metrics["recall_at_k"]["relative"].as_f64().unwrap() * 100.0,
    );
}
```

The first table is the baseline. Deltas (absolute and relative) are computed for all subsequent tables.

## Eval runs in the catalog

Every evaluation is recorded automatically:

```rust
let runs = session.catalog().list_eval_runs()?;
for run in &runs {
    println!("{}: {} on {} (k={:?})",
        run.eval_run_id, run.eval_type, run.golden_source, run.k);
}

// Get the latest eval for a specific model
let latest = session.catalog()
    .latest_eval_run("sentence-transformers/all-MiniLM-L6-v2::1", "embedding")?;
```

## Schema validation

Golden datasets are validated before evaluation starts. Missing or wrong-type columns produce clear errors:

```
Eval error: Golden dataset missing required column 'query_text'
Eval error: Golden dataset column 'query_id' has type Boolean, expected Utf8
```

Integer ID columns (Int32, Int64) are accepted where Utf8 is expected — common for datasets with numeric IDs.

## Evaluate classification

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

Per-class precision, recall, and F1 are available in the `per_class` field.

## Evaluate summarization

```rust
let metrics = session.eval_inference(
    "facebook/bart-large-cnn",
    "articles",
    &["text".into()],
    EvalTask::Summarization,
    "golden.public.summaries",
    "reference",
).await?;

println!("ROUGE-L precision: {}", metrics["precision"]);
println!("ROUGE-L recall:    {}", metrics["recall"]);
println!("ROUGE-L F1:        {}", metrics["f1"]);
```

ROUGE-L uses longest common subsequence (LCS) after normalization (lowercase, strip punctuation).
