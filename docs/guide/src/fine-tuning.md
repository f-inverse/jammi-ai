# Fine-Tune for Your Domain

> **Measured companion:** for the long-form, executed-and-measured Python treatment, see [The Cookbook → Fine-Tuning Methods](https://f-inverse.github.io/jammi-ai/cookbook/chapters/08-finetune-methods/finetune-methods.html).

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

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# use jammi_db::source::{FileFormat, SourceConnection, SourceType};
# async fn ex(session: &InferenceSession) -> jammi_db::error::Result<()> {
session.add_source("training", SourceType::File, SourceConnection {
    url: Some("file:///data/training_pairs.csv".into()),
    format: Some(FileFormat::Csv),
    ..Default::default()
}).await?;
# Ok(()) }
```

### Python

```python
db.add_source("training", path="/data/training_pairs.csv", format="csv")
```

## Start a fine-tuning job

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# async fn ex(session: &InferenceSession) -> jammi_db::error::Result<()> {
use jammi_ai::fine_tune::FineTuneMethod;
use jammi_db::ModelTask;

let job = session.fine_tune(
    "training",
    "sentence-transformers/all-MiniLM-L6-v2",
    &["text_a".into(), "text_b".into(), "score".into()],
    FineTuneMethod::Lora,
    ModelTask::TextEmbedding,
    None,  // default config
).await?;

println!("Job: {}", job.job_id);
job.wait().await?;
println!("Model: {}", job.model_id());
# Ok(()) }
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

`base_model` accepts any model reference form, including a local checkpoint
(`local:/path`, `file:///path`, or a bare filesystem path) — see [Use a Local
Model Checkpoint](./local-models.md).

## Custom configuration

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# use jammi_ai::fine_tune::{FineTuneMethod, LrSchedule};
# async fn ex(session: &InferenceSession, model: &str, columns: Vec<String>) -> jammi_db::error::Result<()> {
use jammi_ai::fine_tune::FineTuneConfig;
use jammi_db::ModelTask;

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
    "training", model, &columns, FineTuneMethod::Lora, ModelTask::TextEmbedding, Some(config),
).await?;
# Ok(()) }
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
| `backbone_dtype` | f32 | Frozen-backbone dtype: f32, f16, or bf16 (bf16 requires CUDA). Applies only when `target_modules` is non-empty (encoder-adapters) — see [Memory](#memory) |

## Use the fine-tuned model

The fine-tuned model is automatically registered and can be used anywhere a model ID is accepted:

### Rust

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# use jammi_ai::fine_tune::training_job::TrainingJob;
# use jammi_db::store::CachePolicy;
# async fn ex(session: &InferenceSession, job: &TrainingJob) -> jammi_db::error::Result<()> {
let model_id = job.model_id();

let embedding = session.encode_text_query(model_id, "quantum computing").await?;
println!("query embedding has {} dims", embedding.len());
session.generate_text_embeddings("patents", model_id, &["abstract".into()], "id", CachePolicy::Bypass).await?;
# Ok(()) }
```

### Python

```python
model_id = job.model_id

query_vec = db.encode_query(model=model_id, query="quantum computing")
db.generate_embeddings(source="patents", model=model_id, columns=["abstract"], key="id", modality="text")
```

## Run metrics

`job.metrics()` (Python) returns the run summary recorded on the job — a
dict with `final_loss` (the best value seen, on whichever metric
`early_stopping_metric` monitored), `early_stopping_metric` (`"train_loss"`
or `"val_loss"`), `total_steps`, and `started_at`/`completed_at`
timestamps. It returns `{}` for a job that has not recorded anything yet
(still queued, or running before its first stamp), and carries
`error_message` instead when the job failed.

```python
job.wait()
metrics = job.metrics()
print(f"final loss: {metrics['final_loss']} ({metrics['early_stopping_metric']})")
```

This is a run summary, not a per-epoch curve: the trainer computes and logs
`avg_train_loss`/`avg_val_loss` at every epoch boundary but does not retain
them past that boundary, so no per-epoch trajectory is available through
this surface today.

## How it works

```text
text -> encoder (frozen) -> base embedding -> LoRA projection (trained) -> output
```

1. The base encoder model (BERT, ModernBERT, etc.) is loaded and frozen
2. A LoRA projection layer (identity + low-rank A/B matrices) is added after pooling
3. For each batch: text is encoded, projected through LoRA, and loss is computed
4. Only the A/B matrices receive gradients
5. The adapter is saved as `adapter.safetensors` in the artifact directory

## Encoder-adapters fine-tuning (PEFT-style adapter injection)

The default flow above trains a single low-rank **projection head** sitting *outside* the frozen encoder. For higher capacity at the same parameter budget, Jammi also supports **encoder adapters** — LoRA injected into named linear layers *inside* the encoder stack, matching the PEFT convention.

Switch to encoder adapters by populating `target_modules` on `FineTuneConfig`:

```rust,no_run
# extern crate jammi_ai;
# use jammi_ai::fine_tune::FineTuneConfig;
# fn make() -> FineTuneConfig {
let config = FineTuneConfig {
    lora_rank: 8,
    lora_alpha: 16.0,
    // Inject LoRA into BERT's attention query and value projections.
    target_modules: vec!["query".to_string(), "value".to_string()],
    ..Default::default()
};
# config }
```

```python
job = db.fine_tune(
    source="training",
    base_model="sentence-transformers/all-MiniLM-L6-v2",
    columns=["text_a", "text_b", "score"],
    method="lora",
    task="text_embedding",
    target_modules=["query", "value"],
)
```

### Target-module conventions

Pick `target_modules` per the architecture you're fine-tuning:

| Architecture | Task | Common target_modules |
|---|---|---|
| BERT / RoBERTa / CamemBERT / XLM-RoBERTa | text | `["query", "value"]` (recommended) or `["query", "key", "value", "dense"]` |
| DistilBERT | text | `["q_lin", "v_lin"]` or `["q_lin", "k_lin", "v_lin", "out_lin"]` |
| ModernBERT | text | `["Wqkv", "Wo"]` (fused QKV + output) |
| OpenCLIP text tower | `text_embedding` | `["in_proj", "out_proj"]` (attention) or `["in_proj", "out_proj", "c_fc", "c_proj"]` |
| OpenCLIP vision tower | `image_embedding` | the same four names |
| HTSAT-CLAP audio tower | `audio_embedding` | `["query", "value"]` or `["query", "key", "value", "attention_output", "intermediate_dense", "output_dense"]`; plus `["reduction"]` (patch-merging) and `["linear1", "linear2"]` (projection head) |
| Any encoder | any | `["all-linear"]` — every linear layer gets an adapter (largest capacity) |

Names match the trailing module-name segment in the HuggingFace weight layout. Suffix matching is the rule, so `"query"` matches `"attention.self.query"`.

`in_proj` on the two OpenCLIP towers is the **fused QKV** projection — one site covering
query, key and value, the way `Wqkv` does on ModernBERT.

A `target_modules` list that matches nothing on the selected tower fails the job with an
error naming that tower's real site names, rather than training an adapter with zero
parameters.

### Fine-tuning an image or audio tower

The tower is selected by the job's **task**, not by a separate flag: an
`image_embedding` job on an OpenCLIP checkpoint fine-tunes its vision tower, a
`text_embedding` job on the same checkpoint fine-tunes its text tower, and an
`audio_embedding` job on an HF-CLAP checkpoint fine-tunes its HTSAT audio tower. A task
the base checkpoint has no tower for is refused before training starts, with a message
naming the towers it does have.

Media jobs read **triplets of encoded bytes** — three binary columns
`anchor`, `positive`, `negative` holding whole files (PNG/JPEG/… for images, WAV/FLAC/
MP3/Ogg for audio). The modality comes from the declared task and is never sniffed from
the bytes, so passing an image corpus to an `audio_embedding` job is a decoding error,
not a silently mis-encoded run. The three groups are encoded as one joined forward pass
and then split.

```python
job = db.fine_tune(
    source="image_triplets",
    base_model="local:/models/open_clip_vit_b32",
    columns=["anchor", "positive", "negative"],
    method="lora",
    task="image_embedding",
    target_modules=["in_proj", "out_proj"],
)
```

What makes a blob a "positive" — an augmentation of the anchor, a co-occurring item — is
your data's concern; the triplet loss only separates whatever pairs you supply.

### Layer ranges and per-module ranks

Two optional refinements:

- **`layers_to_transform`** — restrict injection to specific 0-based layer indices. `None` (default) applies to every layer.
- **`rank_pattern`** — override `lora_rank` for individual modules. Keys are substring matches against the module name; values are the override rank.

`layers_to_transform` indexes the **first numbered segment** of the weight name, matching
PEFT's own rule. On the BERT family and the two OpenCLIP towers that is the transformer
layer. On the HTSAT audio tower, whose blocks are named `layers.{stage}.blocks.{block}`,
it is the **stage** index. A site that sits in no numbered unit at all — the CLAP audio
projection head's `linear1`/`linear2` — is **excluded** whenever `layers_to_transform` is
set, again matching PEFT: a restriction to specific layers cannot be satisfied by a module
that belongs to no layer.

```rust,no_run
# extern crate jammi_ai;
# use jammi_ai::fine_tune::FineTuneConfig;
# fn make() -> FineTuneConfig {
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
# config }
```

### On-disk artifact

Every fine-tuned model writes `adapter.safetensors` plus an
`adapter_config.json` whose `adapter_type` tag discriminates between the two
adapter shapes Jammi produces.

Encoder-adapters example:

```json
{
  "adapter_type": "encoder_adapters",
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

Projection-head example:

```json
{
  "adapter_type": "projection_head",
  "lora_rank": 8,
  "lora_alpha": 16.0,
  "head_layers": ["projection"]
}
```

`model_type` records the base architecture the adapter was trained on — one of
`bert`, `distilbert`, `modernbert`, `open_clip` (an OpenCLIP checkpoint, which ships no
`model_type` field of its own) or `clap_audio_model`.

A checkpoint that holds more than one tower carries one extra key, `tower`, naming which
one the adapter installs on:

```json
{
  "adapter_type": "encoder_adapters",
  "model_type": "open_clip",
  "lora_rank": 8,
  "lora_alpha": 16.0,
  "use_rslora": false,
  "target_modules": ["in_proj", "out_proj"],
  "layers_to_transform": null,
  "rank_pattern": {},
  "backbone_dtype": "f32",
  "tower": "vision"
}
```

The key is written only when it applies: a single-tower adapter's
`adapter_config.json` carries no `tower` key at all.

The Candle inference backend reads `adapter_config.json` on model load and
dispatches on `adapter_type`: `encoder_adapters` rebuilds the encoder with
frozen backbone weights plus the LoRA A/B from `adapter.safetensors`;
`projection_head` loads the saved projection weights as a `LoraLinear`
applied after pooling.

Before any of that, the backend checks the adapter and the base agree on
**architecture family**: an `open_clip` adapter on a CLAP base, a `clap_audio_model`
adapter on a BERT base, or a `tower` the base checkpoint does not have, is refused with a
typed error rather than loaded onto whatever the base happens to be. On an OpenCLIP base
the adapted tower is rebuilt with the adapter's weights and the sibling tower is rebuilt
frozen at the same backbone precision — a fine-tuned model has one identity and one
precision.

### When to use each

- **Projection head** — fastest training, smallest artifact, lowest memory.
  The default when `target_modules` is empty. Best for adapting embedding
  direction without changing per-token attention behaviour.
- **Encoder adapters** — higher representational ceiling per adapter
  parameter; required if the task needs to reshape attention behaviour
  (e.g. a domain where the base attention pattern mismatches the query
  distribution). Costs a slightly slower forward pass since the LoRA path
  runs per layer.

### QLoRA (encoder adapters over a quantized base)

`base_model` accepts a GGUF checkpoint (see [Quantized (GGUF)
checkpoints](./local-models.md#quantized-gguf-checkpoints)) the same way it
accepts a safetensors checkpoint. When the resolved base is `model.gguf`, an
encoder-adapters job trains its LoRA A/B matrices over the frozen quantized
backbone automatically — the base artifact selects this, not a separate flag
or config field. The quantized weights themselves are never trained (LoRA
never updates a frozen base, quantized or dense); only the low-rank adapters
receive gradients, exactly as with a dense base.

## Training safety

- **Divergence detection:** if loss is NaN or >100 for 3 consecutive batches, the job fails with a clear error
- **Early stopping:** training stops when validation loss doesn't improve for `patience` epochs, best checkpoint weights are restored
- **Checkpoints:** saved at ~10% intervals for crash recovery

## Memory

Training memory is dominated by the frozen backbone's weights and
activations, scaled by `batch_size` and `max_seq_length`. `backbone_dtype`
defaults to `f32` for numerical conservatism — every job runs the backbone
at full precision unless you opt into a lower-precision dtype; the trained
LoRA A/B matrices always stay `f32` regardless of `backbone_dtype`, for
numerical stability.

`backbone_dtype` only takes effect on the **encoder-adapters** arm
(`target_modules` non-empty) — the projection-head arm (the default, empty
`target_modules`) never re-dtypes the frozen backbone, so `bf16` is not an
available remedy there.

This guidance applies to the **fine-tune training kinds** (embedding and
classification training), and only when the failure's own error text carries
a recognized out-of-memory spelling. Two cases it does NOT cover: a host
OOM-kill that terminates the worker process outright leaves no error message
at all to classify — the job is picked up by lease reclaim instead, not this
guidance; and a `ContextPredictor` training run carries no OOM guidance at
all (it doesn't route through this classifier).

When it applies, the job's terminal error is rewritten to name the exact
`batch_size`, `max_seq_length`, and (on the encoder-adapters arm)
`backbone_dtype` it ran with, and suggests remedies in the order they're
cheapest to try:

- **Encoder adapters:** (1) `backbone_dtype: bf16` — substantially reduces
  memory on this arm. Requires a CUDA device; a `bf16` backbone on a
  non-CUDA device is refused before training starts, rather than silently
  falling back to `f32`. (2) A smaller `batch_size`, or trade batch size for
  `gradient_accumulation_steps` to hold the same effective batch size while
  shrinking the per-step activation memory. (3) A smaller `max_seq_length`.
- **Projection head (default):** `backbone_dtype` does not apply and is
  omitted from the message — the message says so outright. (1) A smaller
  `batch_size`, or trade batch size for `gradient_accumulation_steps`. (2) A
  smaller `max_seq_length`.

For a fine-tune job whose failure was classified this way, `jammi train
status` (and the Python `job.status()`) surfaces the rewritten message
directly, so you don't need to read raw driver output to find the fix.
