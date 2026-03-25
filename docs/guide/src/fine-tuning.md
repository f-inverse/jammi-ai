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

let embedding = session.encode_query(model_id, "quantum computing").await?;
session.generate_embeddings("patents", model_id, &["abstract".into()], "id").await?;
```

### Python

```python
model_id = job.model_id

query_vec = db.encode_query(model_id, "quantum computing")
db.generate_embeddings(source="patents", model=model_id, columns=["abstract"], key="id")
```

## How it works

```
text -> BertModel (frozen) -> base embedding -> LoRA projection (trained) -> output
```

1. The base model is loaded and frozen
2. A LoRA projection layer (identity + low-rank A/B matrices) is added after pooling
3. For each batch: text is encoded, projected through LoRA, and loss is computed
4. Only the A/B matrices receive gradients
5. The adapter is saved as `adapter.safetensors` in the artifact directory

## Training safety

- **Divergence detection:** if loss is NaN or >100 for 3 consecutive batches, the job fails with a clear error
- **Early stopping:** training stops when validation loss doesn't improve for `patience` epochs, best checkpoint weights are restored
- **Checkpoints:** saved at ~10% intervals for crash recovery
