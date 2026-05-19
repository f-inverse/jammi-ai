# Fine-Tune for Your Domain

Train LoRA adapters on your data to improve embedding quality for your domain. The base model stays frozen — only a small projection layer is trained and saved.

## Prepare training data

Create contrastive pairs with a similarity score:

```csv
text_a,text_b,score
"quantum error correction","superconducting qubit stabilization",0.88
"quantum error correction","medieval poetry analysis",0.08
```

High scores mean similar; low scores mean dissimilar.

Register the training data as a source:

### Rust

```rust
session.add_source("training", SourceType::Local, SourceConnection {
    url: Some("file:///data/training_pairs.csv".into()),
    format: Some(FileFormat::Csv),
    ..Default::default()
}).await?;
```

### Python

```python
db.add_source("training", path="/data/training_pairs.csv", format="csv")
```

## Start a fine-tuning job

### Rust

```rust
use jammi_ai::fine_tune::FineTuneMethod;

let job = session.fine_tune(
    "training",
    "sentence-transformers/all-MiniLM-L6-v2",
    &["text_a".into(), "text_b".into(), "score".into()],
    FineTuneMethod::Lora,
    "embedding",
    None,  // default config
).await?;

println!("Job: {}", job.job_id);
job.wait().await?;
println!("Model: {}", job.model_id());
```

### Python

```python
job = db.fine_tune(
    source="training",
    base_model="sentence-transformers/all-MiniLM-L6-v2",
    columns=["text_a", "text_b", "score"],
    method="lora",
    task="embedding",
)

job.wait()
print(f"Model: {job.model_id}")
```

## Custom configuration

### Rust

```rust
use jammi_ai::fine_tune::FineTuneConfig;

let config = FineTuneConfig {
    lora_rank: 4,
    learning_rate: 5e-4,
    epochs: 5,
    batch_size: 4,
    warmup_steps: 10,
    lr_schedule: LrSchedule::CosineDecay,
    early_stopping_patience: 2,
    validation_fraction: 0.2,
    gradient_accumulation_steps: 4,  // effective batch = 4 x 4 = 16
    ..Default::default()
};

let job = session.fine_tune(
    "training", model, &columns, FineTuneMethod::Lora, "embedding", Some(config),
).await?;
```

## Configuration reference

| Field | Default | Description |
|-------|---------|-------------|
| `lora_rank` | 8 | Low-rank dimension |
| `lora_alpha` | 16.0 | Scaling factor |
| `lora_dropout` | 0.05 | Dropout probability |
| `learning_rate` | 2e-4 | Base learning rate |
| `epochs` | 3 | Training epochs |
| `batch_size` | 8 | Micro-batch size |
| `max_seq_length` | 512 | Max tokens per text |
| `gradient_accumulation_steps` | 1 | Steps before optimizer update |
| `validation_fraction` | 0.1 | Holdout fraction for early stopping |
| `early_stopping_patience` | 3 | Epochs without improvement before stopping |
| `warmup_steps` | 100 | Linear warmup from 0 to base LR |
| `lr_schedule` | CosineDecay | Decay after warmup: Constant, CosineDecay, LinearDecay |
| `embedding_loss` | auto | CoSent (pairs+scores), Triplet, MultipleNegativesRanking |

## Use the fine-tuned model

The fine-tuned model is automatically registered and can be used anywhere a model ID is accepted:

### Rust

```rust
let model_id = job.model_id();

let embedding = session.encode_text_query(model_id, "quantum computing").await?;
session.generate_text_embeddings("patents", model_id, &["abstract".into()], "id").await?;
```

### Python

```python
model_id = job.model_id

query_vec = db.encode_text_query(model_id, "quantum computing")
db.generate_text_embeddings(source="patents", model=model_id, columns=["abstract"], key="id")
```

## How it works

```
text -> encoder (frozen) -> base embedding -> LoRA projection (trained) -> output
```

1. The base encoder model (BERT, ModernBERT, etc.) is loaded and frozen
2. A LoRA projection layer (identity + low-rank A/B matrices) is added after pooling
3. For each batch: text is encoded, projected through LoRA, and loss is computed
4. Only the A/B matrices receive gradients
5. The adapter is saved as `adapter.safetensors` in the artifact directory

## Deep LoRA (PEFT-style adapter injection)

The default flow above (shallow LoRA) trains a single low-rank projection sitting *outside* the frozen encoder. For higher capacity at the same parameter budget, Jammi also supports **deep LoRA** — adapters injected into named linear layers *inside* the encoder stack, matching the PEFT convention.

Trigger deep LoRA by populating `target_modules` on `FineTuneConfig`:

```rust
let config = FineTuneConfig {
    lora_rank: 8,
    lora_alpha: 16.0,
    // Inject LoRA into BERT's attention query and value projections.
    target_modules: vec!["query".to_string(), "value".to_string()],
    ..Default::default()
};
```

```python
job = db.fine_tune(
    source="training",
    base_model="sentence-transformers/all-MiniLM-L6-v2",
    columns=["text_a", "text_b", "score"],
    method="lora",
    task="embedding",
    target_modules=["query", "value"],
)
```

### Target-module conventions

Pick `target_modules` per the architecture you're fine-tuning:

| Architecture | Common target_modules |
|---|---|
| BERT / RoBERTa / CamemBERT / XLM-RoBERTa | `["query", "value"]` (recommended) or `["query", "key", "value", "dense"]` |
| DistilBERT | `["q_lin", "v_lin"]` or `["q_lin", "k_lin", "v_lin", "out_lin"]` |
| ModernBERT | `["Wqkv", "Wo"]` (fused QKV + output) |
| Any encoder | `["all-linear"]` — every linear layer gets an adapter (largest capacity) |

Names match the trailing module-name segment in the HuggingFace weight layout. Suffix matching is the rule, so `"query"` matches `"attention.self.query"`.

### Layer ranges and per-module ranks

Two optional refinements:

- **`layers_to_transform`** — restrict injection to specific 0-based layer indices. `None` (default) applies to every layer.
- **`rank_pattern`** — override `lora_rank` for individual modules. Keys are substring matches against the module name; values are the override rank.

```rust
let mut rank_pattern = std::collections::HashMap::new();
rank_pattern.insert("query".to_string(), 16);  // higher capacity on Q
rank_pattern.insert("value".to_string(), 4);   // lower on V

let config = FineTuneConfig {
    lora_rank: 8,                                     // default rank
    target_modules: vec!["query".into(), "value".into()],
    layers_to_transform: Some(vec![6, 7, 8, 9, 10, 11]), // top half only
    rank_pattern,
    ..Default::default()
};
```

### On-disk artifact

Deep-LoRA adapters save `adapter.safetensors` plus an `adapter_config.json` carrying the build-time settings:

```json
{
  "adapter_type": "deep_lora",
  "model_type": "bert",
  "lora_rank": 8,
  "lora_alpha": 16.0,
  "use_rslora": false,
  "target_modules": ["query", "value"],
  "layers_to_transform": [6, 7, 8, 9, 10, 11],
  "rank_pattern": {"query": 16, "value": 4},
  "backbone_dtype": "f32"
}
```

The Candle inference backend reads `adapter_config.json` on model load. If `adapter_type` is `"deep_lora"`, it rebuilds the encoder with frozen backbone weights plus the LoRA A/B from `adapter.safetensors`. Shallow-LoRA adapters (no `adapter_config.json`, or `adapter_type` set to anything else) continue using the external-projection inference path.

### When to choose deep vs shallow

- **Shallow** — fastest training, smallest artifact, lowest memory. The default when `target_modules` is empty. Best for adapting embedding direction without per-token attention reshape.
- **Deep** — higher representational ceiling per adapter parameter; required if the task needs to reshape attention behaviour (e.g., a domain where the base attention pattern mismatches the query distribution). Costs a slightly slower forward pass since LoRA paths run per layer.

## Training safety

- **Divergence detection:** if loss is NaN or >100 for 3 consecutive batches, the job fails with a clear error
- **Early stopping:** training stops when validation loss doesn't improve for `patience` epochs, best checkpoint weights are restored
- **Checkpoints:** saved at ~10% intervals for crash recovery
