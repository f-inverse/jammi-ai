# Configuration

Jammi loads configuration from three sources, in priority order:

1. **Config file** (TOML) — explicit path, `$JAMMI_CONFIG` env var, `./jammi.toml`, or `~/.config/jammi/config.toml`
2. **Environment variables** — `JAMMI_GPU__DEVICE=0`, `JAMMI_INFERENCE__BATCH_SIZE=64`
3. **Defaults** — sensible defaults for all fields

```rust
// Load with defaults
let config = JammiConfig::load(None)?;

// Load from a specific file
let config = JammiConfig::load(Some(Path::new("/path/to/jammi.toml")))?;
```

## Full reference

```toml
# Where Jammi stores artifacts (catalog DB, model cache, embeddings)
# Default: platform-specific data directory (~/.local/share/jammi on Linux)
artifact_dir = "/path/to/artifacts"

[engine]
# Number of DataFusion execution threads. Default: number of CPUs.
execution_threads = 8
# Memory limit for the query engine. Default: "75%".
memory_limit = "75%"
# Maximum rows per DataFusion batch. Default: 8192.
batch_size = 8192

[gpu]
# GPU device index. -1 for CPU only. Default: 0.
device = -1
# GPU memory limit. Default: "auto".
memory_limit = "auto"
# Fraction of GPU memory Jammi may use. Default: 0.9.
memory_fraction = 0.9

[inference]
# Default backend selection strategy. Default: "auto".
default_backend = "auto"
# Maximum rows per inference batch. Default: 32.
batch_size = 32
# Timeout for batch accumulation in server mode (seconds). Default: 300.
batch_timeout_secs = 300
# Maximum models kept loaded simultaneously. 0 = unlimited. Default: 0.
max_loaded_models = 0

[inference.vllm]
# vLLM server host. Optional.
host = "localhost"
# vLLM server port. Optional.
port = 8000
# Extra CLI args passed to vLLM. Default: [].
extra_args = []

[inference.http]
# HTTP request timeout (seconds). Default: 60.
timeout_secs = 60
# Custom headers for HTTP model endpoints.
[inference.http.headers]
# Authorization = "Bearer sk-..."

[embedding]
# Distance metric for vector indices. Default: "cosine".
default_distance_metric = "cosine"
# Index type for vector storage. Default: "ivf_hnsw_sq".
default_index_type = "ivf_hnsw_sq"
# Rows between embedding index checkpoints. Default: 1000.
checkpoint_interval = 1000

[fine_tuning]
# LoRA rank for fine-tuning. Default: 8.
default_lora_rank = 8
# Learning rate. Default: 0.0002.
default_learning_rate = 0.0002
# Training epochs. Default: 3.
default_epochs = 3
# Training batch size. Default: 8.
default_batch_size = 8
# Checkpoint every N fraction of training. Default: 0.1.
checkpoint_fraction = 0.1

[cache]
# Enable ANN query cache. Default: true.
ann_cache_enabled = true
# Max cached ANN queries. Default: 10000.
ann_cache_max_entries = 10000
# Enable embedding cache. Default: true.
embedding_cache_enabled = true
# Embedding cache size. Default: "1GB".
embedding_cache_size = "1GB"

[server]
# Health probe listen address. Default: "0.0.0.0:8080".
health_listen = "0.0.0.0:8080"
# Arrow Flight SQL listen address. Default: "0.0.0.0:8081".
flight_listen = "0.0.0.0:8081"
# Models to preload on server start. Default: [].
preload_models = ["sentence-transformers/all-MiniLM-L6-v2"]

[logging]
# Log level: "trace", "debug", "info", "warn", "error". Default: "info".
level = "info"
# Log format: "text" or "json". Default: "text".
format = "text"
```

## Environment variable overrides

Every config field can be overridden with an environment variable using the pattern `JAMMI_<SECTION>__<FIELD>`:

| Variable | Overrides |
|----------|-----------|
| `JAMMI_ARTIFACT_DIR` | `artifact_dir` |
| `JAMMI_ENGINE__BATCH_SIZE` | `engine.batch_size` |
| `JAMMI_GPU__DEVICE` | `gpu.device` |
| `JAMMI_INFERENCE__BATCH_SIZE` | `inference.batch_size` |
| `JAMMI_LOGGING__LEVEL` | `logging.level` |

Note the double underscore (`__`) separating section and field.
