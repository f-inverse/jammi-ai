# Configuration

Jammi loads configuration from three sources, in priority order:

1. **Config file** (TOML) — explicit path, `$JAMMI_CONFIG` env var, `./jammi.toml`, or `~/.config/jammi/config.toml`
2. **Environment variables** — `JAMMI_GPU__DEVICE=0`, `JAMMI_INFERENCE__BATCH_SIZE=64`
3. **Defaults** — sensible defaults for all fields

```rust,no_run
# extern crate jammi_db;
# use std::path::Path;
# use jammi_db::config::JammiConfig;
# fn ex() -> jammi_db::error::Result<()> {
// Load with defaults
let config = JammiConfig::load(None)?;

// Load from a specific file
let config = JammiConfig::load(Some(Path::new("/path/to/jammi.toml")))?;
# Ok(()) }
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
# Fail fast if the requested GPU is unavailable instead of falling back to CPU.
# Default: false (degrade to CPU with a warning).
require_gpu = false
# Default inference compute precision: "f32" or "f16". A model may override
# this with its own "compute_precision" in config.json; the per-model value
# wins. "bf16" is a valid value for fine-tune's frozen-backbone dtype but is
# rejected at inference load time (not yet supported). Default: "f32".
compute_precision = "f32"

[inference]
# Default backend selection strategy. Default: "auto".
default_backend = "auto"
# Maximum rows per inference batch. Default: 32.
batch_size = 32
# Timeout for batch accumulation in server mode (seconds). Default: 300.
batch_timeout_secs = 300
# Maximum models kept loaded simultaneously. 0 = unlimited. Default: 0.
max_loaded_models = 0

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

[training]
# Whether THIS process runs the training claim loop. Default: true.
# true  - the process claims queued jobs, renews the lease while they run,
#         and reclaims leases that expired under a dead claimant.
# false - the process still mounts and serves the training surface and still
#         accepts submissions, but never claims. Submitted jobs stay queued
#         until some process with run_worker = true opens the catalog. The
#         SQLite catalog is single-process, so this process must close the
#         catalog before that one can open it; a Postgres catalog is
#         multi-process and can run both at once.
run_worker = true
# How long a claim leases a job before it is reclaimable. Default: 30.
lease_duration_secs = 30
# How often the worker renews the lease while a job runs. Must leave a real
# margin under the lease (heartbeat_interval_secs * 2 < lease_duration_secs),
# so a single missed beat does not drop a live worker's lease. Default: 10.
heartbeat_interval_secs = 10
# How often an idle worker polls for a queued job (and reclaims expired
# leases). Must be > 0 - a zero poll is a busy-loop. Default: 1.
idle_poll_secs = 1

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

The loader reads the environment variables below, and only these. Each name
follows the pattern `JAMMI_<SECTION>__<FIELD>` — note the double underscore
(`__`) separating section from field — but the pattern describes the names
that exist, it does not generate them. A config field with no row here has no
environment override, and setting a plausible-looking name for one
(`JAMMI_TRAINING__LEASE_DURATION_SECS`, say) does nothing at all rather than
failing. Set it in the file.

| Variable | Overrides |
|----------|-----------|
| `JAMMI_ARTIFACT_DIR` | `artifact_dir` |
| `JAMMI_ENGINE__BATCH_SIZE` | `engine.batch_size` |
| `JAMMI_ENGINE__EXECUTION_THREADS` | `engine.execution_threads` |
| `JAMMI_ENGINE__MEMORY_LIMIT` | `engine.memory_limit` |
| `JAMMI_GPU__DEVICE` | `gpu.device` |
| `JAMMI_GPU__MEMORY_FRACTION` | `gpu.memory_fraction` |
| `JAMMI_GPU__MEMORY_LIMIT` | `gpu.memory_limit` |
| `JAMMI_GPU__REQUIRE_GPU` | `gpu.require_gpu` |
| `JAMMI_INFERENCE__BATCH_SIZE` | `inference.batch_size` |
| `JAMMI_INFERENCE__BATCH_TIMEOUT_SECS` | `inference.batch_timeout_secs` |
| `JAMMI_INFERENCE__DEFAULT_BACKEND` | `inference.default_backend` |
| `JAMMI_INFERENCE__MAX_LOADED_MODELS` | `inference.max_loaded_models` |
| `JAMMI_LOGGING__FORMAT` | `logging.format` |
| `JAMMI_LOGGING__LEVEL` | `logging.level` |
| `JAMMI_SERVER__FLIGHT_LISTEN` | `server.flight_listen` |
| `JAMMI_SERVER__HEALTH_LISTEN` | `server.health_listen` |
| `JAMMI_SERVER__SERVICES` | `server.services` |
| `JAMMI_TRAINING__RUN_WORKER` | `training.run_worker` |

`JAMMI_CONFIG` is not in the table because it is not an override: it names
which config *file* to load.

`JAMMI_TRAINING__RUN_WORKER` is a boolean override (`TrainingConfig::run_worker`
in `crates/jammi-db/src/config.rs`). It accepts `true`, `false`, `1`, and `0`,
case-insensitively and with surrounding whitespace trimmed. Any other value —
including an empty one — fails the config load with an error naming the
variable, the rejected value, and the accepted set. It is not ignored and does
not fall back to the file's value: a yes/no question about what the process
will do has no safe direction to guess in, and silently dropping the override
would leave the process doing the opposite of what was written, with nothing
in the config file to explain it.
