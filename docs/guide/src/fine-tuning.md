# Fine-Tuning

`fine_tune()` trains LoRA adapters on your data to improve embedding quality for your domain. The base model stays frozen — only a small projection layer is trained and saved.

## Training data

Prepare contrastive pairs (CSV, Parquet, or any registered source):

```
text_a,text_b,score
"quantum error correction","superconducting qubit stabilization",0.88
"quantum error correction","medieval poetry analysis",0.08
```

High scores mean similar; low scores mean dissimilar. Triplet format (`anchor, positive, negative`) is also supported.

Register the training data as a source:

```rust
session.add_source("training", SourceType::Local, SourceConnection {
    url: Some("file:///path/to/training_pairs.csv".into()),
    format: Some(FileFormat::Csv),
    ..Default::default()
}).await?;
```

## Start a fine-tuning job

```rust
use jammi_ai::fine_tune::FineTuneConfig;

let job = session.fine_tune(
    "training",                                          // source
    "sentence-transformers/all-MiniLM-L6-v2",           // base model
    &["text_a".into(), "text_b".into(), "score".into()], // columns
    "lora",                                              // method
    "embedding",                                         // task
    None,                                                // default config
).await?;

println!("Job: {}", job.job_id);
println!("Model: {}", job.model_id());  // jammi:fine-tuned:{uuid}

job.wait().await?;
```

The job runs asynchronously. `wait()` blocks until completion.

## Custom configuration

```rust
let config = FineTuneConfig {
    lora_rank: 4,
    learning_rate: 5e-4,
    epochs: 5,
    batch_size: 4,
    warmup_steps: 10,
    lr_schedule: LrSchedule::CosineDecay,
    early_stopping_patience: 2,
    validation_fraction: 0.2,
    gradient_accumulation_steps: 4,  // effective batch = 4 × 4 = 16
    ..Default::default()
};

let job = session.fine_tune(
    "training", model, &columns, "lora", "embedding", Some(config),
).await?;
```

All config fields are validated before training starts. Invalid values (e.g., `batch_size: 0`, `validation_fraction: 1.5`) produce clear errors.

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

```rust
let model_id = job.model_id();  // "jammi:fine-tuned:{uuid}"

// Encode a query with the fine-tuned model
let embedding = session.encode_query(model_id, "quantum computing").await?;

// Generate embeddings for a whole source
session.generate_embeddings("patents", model_id, &["abstract".into()], "id").await?;
```

The fine-tuned model loads the original base model plus the saved LoRA adapter. Embeddings differ from the base model because the LoRA projection has been trained on your domain data.

## How it works

```
text → BertModel (frozen) → base embedding → LoRA projection (trained) → output
```

1. The base model (e.g., MiniLM) is loaded from safetensors and frozen
2. A LoRA projection layer (identity + low-rank A/B matrices) is added after pooling
3. For each batch: text is encoded through the base model, projected through LoRA, and loss is computed
4. Only the A/B matrices receive gradients — the base model weights never change
5. The adapter (A/B matrices) is saved as `adapter.safetensors` in the artifact directory

## Training safety

- **Divergence detection:** if loss is NaN or >100 for 3 consecutive batches, the job fails with a clear error
- **Early stopping:** training stops when validation loss doesn't improve for `patience` epochs, and the best checkpoint weights are restored
- **Checkpoints:** saved at ~10% intervals for crash recovery

## QLoRA

Not supported. `method="qlora"` returns a clear error.
