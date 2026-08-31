# Use a Local Model Checkpoint

Every model-accepting argument in the engine — `generate_embeddings`' `model`,
fine-tune's `base_model`, `annotate()`'s first argument, the Python `Session`
API — takes a model reference string. A reference that names a filesystem
location loads the checkpoint from local disk, with no Hub access and no
network: this is the supported path for air-gapped hosts and for checkpoints
you have already downloaded or trained elsewhere.

## Reference forms

`ModelSource::parse` recognizes four spellings; the first three are local:

| Form | Example | Resolves to |
|---|---|---|
| `local:` prefix | `local:/models/bioclinical-modernbert-large` | the directory after the prefix |
| `file://` URI | `file:///models/bioclinical-modernbert-large` | the URI's path |
| Bare filesystem path | `/models/m`, `./m`, `../m` | that path (must start with `/`, `./`, or `../`) |
| Anything else | `sentence-transformers/all-MiniLM-L6-v2`, `hf://owner/repo` | a HuggingFace Hub repo id |

A local path is resolved against the filesystem of the host **running the
engine** — for a remote client that is the server, not the client machine — so
the directory must exist there.

## What the directory must contain

- A config: `config.json` (or `open_clip_config.json` for OpenCLIP models).
- Weights: `model.safetensors` (or `open_clip_model.safetensors`),
  `model.onnx`, and/or `model.gguf`.

When both safetensors and ONNX weights are present, the ONNX file wins and the
model runs on the ORT backend; safetensors alone selects the Candle backend. An
explicit backend hint overrides this choice. `model.gguf` is considered only
when neither safetensors nor ONNX weights are present — see [Quantized (GGUF)
checkpoints](#quantized-gguf-checkpoints) below.

Resolution is fail-loud: a nonexistent directory, a directory with no config,
and a directory with no recognized weights file each produce a typed error
naming what is missing — there is no silent fallback to the Hub.

## Quantized (GGUF) checkpoints

A directory with no `model.safetensors`/`model.onnx` but a `model.gguf` loads
on the Candle backend as a quantized checkpoint — matmul-site weights stored
at a k-quant format (`q4_0` through `q6k`) stay resident in that compressed
form; everything else (embeddings, norms, classifier/NER heads, and any
matmul-site weight that happens to be stored densely) is dequantized to the
model's compute dtype at load.

Requirements and limits:

- The weights file must be named exactly `model.gguf` — any other `*.gguf`
  filename in the directory produces a typed error naming the convention.
- `config.json` is still required, and still the source of the model's
  architecture and layer count — a GGUF file's own embedded metadata (the
  convention some GGUF exporters use in place of a sidecar `config.json`) is
  never read for this. A directory with a `model.gguf` but no `config.json`
  is not a supported checkpoint shape.
- Supported architectures: BERT and its variants (RoBERTa, CamemBERT,
  XLM-RoBERTa), DistilBERT, and ModernBERT. Any other architecture (OpenCLIP,
  CLAP) is a typed refusal, not a best-effort load.
- Every tensor in the file must be a supported k-quant format (`q4_0`,
  `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q2k`, `q3k`, `q4k`, `q5k`, `q6k`) or stored
  densely as `f32`/`f16`/`bf16` — any other GGML dtype is a typed refusal.

`base_model` in [fine-tuning](./fine-tuning.md) accepts a `model.gguf`
checkpoint the same way: an encoder-adapters LoRA job trains its low-rank
adapters over the frozen quantized backbone automatically when the resolved
base is GGUF — there is no separate QLoRA flag or config field.

## Examples

Embeddings over a local checkpoint (Python):

```python
db.generate_embeddings(
    source="patents",
    model="local:/models/all-MiniLM-L6-v2",
    columns=["abstract"],
    key="id",
    modality="text",
)
```

Fine-tune from a local base model (see [Fine-Tune for Your
Domain](./fine-tuning.md)):

```python
db.fine_tune(
    base_model="local:/models/bioclinical-modernbert-large",
    ...
)
```

SQL, over Flight SQL (see [Compound Retrieval and
Inference](./remote-compound-query.md)):

```sql
SELECT * FROM annotate('local:/models/all-MiniLM-L6-v2', 'text_embedding',
                       ARRAY['abstract'])
```

Model **registration** — a durable catalog entry with an id, stage
transitions, and evidence — is a platform concern and lives outside the OSS
engine; the engine consumes local checkpoints directly through the reference
forms above.
