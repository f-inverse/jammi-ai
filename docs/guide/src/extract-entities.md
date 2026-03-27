# Extract Entities (NER)

Run a Named Entity Recognition model over text columns to extract person names, organizations, locations, and other entities. Results are returned as JSON arrays of entity spans with character positions and confidence scores.

## Basic usage

### Rust

```rust
use jammi_ai::model::{ModelSource, ModelTask};

let model = ModelSource::hf("dslim/bert-base-NER");
let results = session.infer(
    "patents",
    &model,
    ModelTask::Ner,
    &["abstract".to_string()],
    "id",
).await?;
```

### Python

```python
results = db.infer(
    source="patents",
    model="dslim/bert-base-NER",
    columns=["abstract"],
    task="ner",
    key="id",
)
```

## Output schema

| Column | Type | Description |
|--------|------|-------------|
| `_row_id` | Utf8 | Key column value |
| `_source` | Utf8 | Source identifier |
| `_model` | Utf8 | Model identifier |
| `_status` | Utf8 | `"ok"` or `"error"` |
| `_error` | Utf8 (nullable) | Error message if failed |
| `_latency_ms` | Float32 | Inference latency |
| `entities` | Utf8 (nullable) | JSON array of entity spans |

## Entity span format

Each entity in the JSON array has:

```json
{
  "text": "Google",
  "label": "ORG",
  "start": 15,
  "end": 21,
  "confidence": 0.97
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | The entity text extracted from the input |
| `label` | string | Entity type (PER, ORG, LOC, etc.) without B-/I- prefix |
| `start` | integer | Character start position (inclusive) |
| `end` | integer | Character end position (exclusive) |
| `confidence` | float | Average softmax confidence across entity tokens |

## Supported models

NER models must have `id2label` with BIO-tagged labels (e.g., `B-PER`, `I-PER`, `O`) in their `config.json`.

**BERT family** — loads `classifier.weight` + `classifier.bias` on top of the encoder:
- `dslim/bert-base-NER` (English, 4 entity types)
- `dbmdz/bert-large-cased-finetuned-conll03-english`

**ModernBERT** — same pattern, modern encoder architecture.

## How it works

```
text → tokenize (with character offsets)
     → encoder forward → hidden states [batch, seq_len, hidden]
     → Linear(hidden, num_labels) per token → logits
     → softmax → argmax → BIO tag per token
     → merge consecutive B-/I- tags into entity spans
     → map character offsets back to original text
```

The BIO decoding handles:
- **B-TYPE**: starts a new entity of that type
- **I-TYPE**: continues the current entity (must match type)
- **O**: outside any entity
- Special tokens ([CLS], [SEP], padding) are automatically skipped
