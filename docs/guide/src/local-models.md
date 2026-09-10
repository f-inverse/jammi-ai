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

## Configuring the Hub cache, endpoint, and token

Every other spelling (a bare repo id, `hf://owner/repo`) resolves against the
Hugging Face Hub through one client built once from `[models]`:

```toml
[models]
hub_endpoint = "https://huggingface.co"
hub_cache_dir = "/var/cache/jammi"
hub_token = { file = "/run/secrets/hf-token" }
offline = false
```

Every field is also settable through the standard `JAMMI_MODELS__HUB_*` /
`JAMMI_MODELS__OFFLINE` env-override layer (e.g. `JAMMI_MODELS__HUB_CACHE_DIR`,
`JAMMI_MODELS__OFFLINE=true`) — that tier sits ABOVE the `HF_*` fallbacks
below: a `JAMMI_*` override behaves exactly like the equivalent TOML key, one
precedence step above `HF_HUB_CACHE`/`HF_HOME`/`HF_ENDPOINT`/`HF_TOKEN`/
`HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`, not alongside them.

| Field | Precedence |
|---|---|
| Cache root | `hub_cache_dir` (`hub/` appended) → `HF_HUB_CACHE` (used AS the cache root directly, nothing appended, matching `huggingface_hub`'s own convention) → `HF_HOME` (`hub/` appended) → the platform home directory's `.cache/huggingface` (`hub/` appended) |
| Endpoint | `hub_endpoint` → `HF_ENDPOINT` → the Hub's own default |
| Token | `hub_token` → `HF_TOKEN` → the cache's own `token` file (`huggingface-cli login`'s file) |
| Offline | `offline`, when explicitly set → `HF_HUB_OFFLINE` → `TRANSFORMERS_OFFLINE` (only when `HF_HUB_OFFLINE` is itself unset — `huggingface_hub`'s own alias) → `false`; truthy for any of `"1"`, `"on"`, `"yes"`, `"true"` (case-insensitive, whitespace trimmed — `huggingface_hub`'s own `ENV_VARS_TRUE_VALUES`) |

No home directory, no `HF_HUB_CACHE`, no `HF_HOME`, and no `hub_cache_dir` is
a typed `JammiError::Config` at session construction — never a panic. A token
that resolves from none of the three sends no `Authorization` header, the
same as an anonymous `huggingface-cli` session.

`offline = true` refuses every Hub *network* fetch: a `HuggingFace`-sourced
model loads only when it resolves against `local:` or an already-populated
catalog row — a warm, on-disk Hub cache directory with no matching catalog
row is still a miss, because the catalog (not the cache) is offline's source
of truth. It does not reach the fine-tune worker's adapter fetch for an
already-trained model, which always reads the adapter bundle from the
artifact store, offline or not. An explicit `offline = false` wins over
`HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` in the environment (and vice versa
for `offline = true`) — only an OMITTED `offline` key falls back to either
environment variable at all.

Every `HF_*`/`HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` fallback above is read
from the **server** process's own environment, at session construction —
never from a remote client's environment (a gRPC/Flight SQL/Python client
connecting to a running `jammi-server` has no way to influence these; only
that server process's own `[models]`/env decides resolution).

A configured `hub_endpoint`/mirror is not part of a resolved model's
identity: two endpoints can serve different bytes for the same repo id (a
stale or divergent mirror), so once a model resolves, its catalog
`artifact_path` pins the actual bytes fetched — re-resolving under a
different endpoint later never silently swaps them out from under an
already-registered model.
