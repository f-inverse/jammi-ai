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
- Weights: `model.safetensors` (or `open_clip_model.safetensors`), and/or
  `model.onnx`.

When both safetensors and ONNX weights are present, the ONNX file wins and the
model runs on the ORT backend; safetensors alone selects the Candle backend. An
explicit backend hint overrides this choice.

Resolution is fail-loud: a nonexistent directory, a directory with no config,
and a directory with no recognized weights file each produce a typed error
naming what is missing — there is no silent fallback to the Hub.

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
