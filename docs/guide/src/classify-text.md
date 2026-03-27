# Classify Text

Run a classification model over text columns to assign labels and confidence scores. Any HuggingFace model with `id2label` in its config works out of the box.

## Basic usage

### Rust

```rust
use jammi_ai::model::{ModelSource, ModelTask};

let model = ModelSource::hf("answerdotai/ModernBERT-base-classification");
let results = session.infer(
    "patents",
    &model,
    ModelTask::Classification,
    &["abstract".to_string()],
    "id",
).await?;
```

### Python

```python
results = db.infer(
    source="patents",
    model="answerdotai/ModernBERT-base-classification",
    columns=["abstract"],
    task="classification",
    key="id",
)
```

## Output schema

Each `RecordBatch` has prefix columns plus classification-specific columns:

| Column | Type | Description |
|--------|------|-------------|
| `_row_id` | Utf8 | Key column value |
| `_source` | Utf8 | Source identifier |
| `_model` | Utf8 | Model identifier |
| `_status` | Utf8 | `"ok"` or `"error"` |
| `_error` | Utf8 (nullable) | Error message if failed |
| `_latency_ms` | Float32 | Inference latency |
| `label` | Utf8 (nullable) | Predicted class label |
| `confidence` | Float32 (nullable) | Confidence score (0-1) |
| `all_scores_json` | Utf8 (nullable) | JSON with all class scores |

## Supported model architectures

Classification models must have `id2label` in their `config.json`. Supported architectures:

**BERT family** — BERT, RoBERTa, DistilBERT, CamemBERT, XLM-RoBERTa:
- Loads `classifier.weight` + `classifier.bias` from safetensors
- CLS token pooling + linear classifier + softmax

**ModernBERT** — uses the built-in `ModernBertForSequenceClassification`:
- CLS or MEAN pooling (configured via `classifier_pooling` in config)
- Head (dense + GELU + LayerNorm) + classifier + softmax

## Fine-tuning for classification

Train a LoRA adapter with a classification head on your labeled data:

### Prepare training data

```csv
text,label
"quantum error correction","physics"
"CRISPR gene editing","biology"
```

### Rust

```rust
use jammi_ai::fine_tune::FineTuneMethod;

let job = session.fine_tune(
    "training",
    "sentence-transformers/all-MiniLM-L6-v2",
    &["text".into(), "label".into()],
    FineTuneMethod::Lora,
    "classification",
    None,
).await?;

job.wait().await?;
```

### Python

```python
job = db.fine_tune(
    source="training",
    base_model="sentence-transformers/all-MiniLM-L6-v2",
    columns=["text", "label"],
    method="lora",
    task="classification",
)
job.wait()
```

The fine-tuned model trains a LoRA projection plus a linear classification head using cross-entropy loss. Both are saved to `adapter.safetensors`.

## Error handling

Same per-row error tracking as embeddings:

| Condition | `_status` | `label` | `confidence` |
|-----------|-----------|---------|--------------|
| Valid text | `"ok"` | Predicted label | 0-1 score |
| Null/empty text | `"error"` | null | null |
