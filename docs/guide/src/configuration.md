# Configuration

Jammi loads configuration by layering three sources — the file layer is deep
merged with the environment layer (an environment value wins field-by-field;
see [Environment variable overrides](#environment-variable-overrides)),
falling back to defaults for anything neither layer sets:

1. **Config file** (TOML) — resolved, first match wins: an explicit path,
   `$JAMMI_CONFIG`, `./jammi.toml`, `/etc/jammi/jammi.toml`, then the platform
   per-user config directory (`config.toml` under
   `directories::ProjectDirs::from("ai", "jammi", "jammi").config_dir()` —
   e.g. `~/.config/jammi/config.toml` on Linux).
2. **Environment variables** — `JAMMI_GPU__DEVICE=0`, `JAMMI_INFERENCE__BATCH_SIZE=64`.
3. **Defaults** — sensible defaults for every field.

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

`JammiConfig::load` is `JammiConfig::load_from` against the real process
environment; `load_from(file, env)` takes an explicit `env` map instead (used
by every hermetic config test in the workspace — and by the doc example
below — so nothing here ever touches a real environment variable).
`JammiConfig::parse_from(toml_src, env)` is the parse-only core underneath
both: it runs `${VAR}` interpolation, the layering, and deserialization, but
skips the post-load validation (`storage.cloud.validate()`, the worker
timing invariants) `load_from` runs afterward.

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

[lease]
# The one lease timing every leased catalog row shares: a claimed training
# job and a `building` result table are both owned under a lease their holder
# heartbeats, and both are reclaimed by a sweep once it expires.
# How long a claim owns its row before it is reclaimable. Default: 30.
duration_secs = 30
# How often the holder renews the lease. Must leave a real margin under the
# lease (heartbeat_secs * 2 < duration_secs), so a single missed beat does
# not drop a live holder's lease. Default: 10.
heartbeat_secs = 10

[worker]
# Whether THIS process runs the job claim loop. Default: true.
# true  - the process claims queued jobs (of `kinds`, below), renews the
#         lease while they run, and reclaims leases that expired under a
#         dead claimant.
# false - the process still mounts and serves the submission surface and
#         still accepts submissions, but never claims. Submitted jobs stay
#         queued until some process with enabled = true opens the catalog.
#         The SQLite catalog is single-process, so this process must close
#         the catalog before that one can open it; a Postgres catalog is
#         multi-process and can run both at once.
enabled = true
# Which job kinds this worker claims. "all" (the default) claims every kind
# compiled into the binary; a comma list or array claims only those named.
kinds = "all"
# How often an idle worker polls for a queued job (and reclaims expired
# leases). Must be > 0 - a zero poll is a busy-loop. Default: 1.
# The lease a claim is held under is `[lease]` above; a `[worker]` section
# naming the former `lease_duration_secs` / `heartbeat_interval_secs`
# keys is refused at load (no alias), never silently defaulted.
idle_poll_secs = 1

[jobs]
# How many days a terminal (completed/failed) job row survives before the
# retention sweep may delete it, and before it stops blocking `delete_model`
# on the model(s) it references. A non-terminal job blocks indefinitely,
# regardless of age. Default: 30.
retention_days = 30

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

[server.limits]
# Request-bounds and refusal policy for the combined gRPC + Flight SQL
# surface (also applied to the Flight-only listener). A request exceeding
# any of these is refused at the edge -- before any tenant-scoped catalog
# read runs, so a refusal never leaks cross-tenant existence -- with a typed
# gRPC status and a jammi_grpc_refused_total{reason} counter increment.
# Maximum inbound message size, in bytes. Must be > 0. Default: 67108864
# (64 MiB). There is no outbound cap.
max_message_bytes = 67108864
# Global cap on unary requests in flight across every connection.
# 0 = unbounded. Default: 256.
max_in_flight = 256
# Cap on unary requests in flight on a SINGLE connection. 0 = unbounded;
# when both this and max_in_flight are non-zero (bounded), this must be
# <= max_in_flight. Default: 64.
max_in_flight_per_connection = 64
# Maximum duration a unary request may run before this server cancels it
# with DEADLINE_EXCEEDED. Unset (the default) means no server-imposed
# timeout. Unary methods only -- Subscribe/WaitJob use wait_timeout_secs
# and the stream budgets below instead.
# request_timeout_secs = 30
# Maximum grpc-timeout a CLIENT may request on TriggerService.Subscribe or
# JobService.WaitJob; a longer request is refused at the edge, before the
# stream opens. Unset (the default) means no cap.
# wait_timeout_secs = 300
# Cap on concurrently open TriggerService.Subscribe streams. 0 = unbounded.
# Default: 256.
max_subscriptions = 256
# Cap on concurrently open JobService.WaitJob streams. 0 = unbounded.
# Default: 1024.
max_job_waits = 1024

[logging]
# Log level: "trace", "debug", "info", "warn", "error". Default: "info".
level = "info"
# Log format: "text" or "json". Default: "text".
format = "text"
```

## Catalog, broker, signing key, storage, and model source

Five sections select a backend rather than tune a fixed set of knobs, so each
is an **externally tagged enum**: the variant name is its own TOML table (or a
bare string for a variant with no required fields), never a `kind =` key
inside one shared table — `[catalog.postgres]`, not `[catalog]` with
`kind = "postgres"`. Selecting an unrecognised variant, or naming a key that
does not belong to the selected one, is a load-time error naming the
offending name; two variants of the same section both present (in the same
layer) is a load-time error too.

**`catalog`** — the models/sources/eval-runs/mutable-table backend. Default:
SQLite under `artifact_dir`.

```toml
[catalog.sqlite]
# path = "/var/lib/jammi/catalog.db"   # optional override
```

```toml
[catalog.postgres]
url = "${POSTGRES_URL}?sslmode=verify-full&sslrootcert=/etc/ssl/certs/ca-certificates.crt"
pool_size = 16
max_lifetime_secs = 1800
```

**`broker`** — the trigger/provenance-channel backend. Default: the
in-process broker.

```toml
broker = "in_memory"
```

```toml
[broker.jet_stream]
url = "nats://${NATS_HOST}:4222"
retention_seconds = 604800
credentials = { file = "/var/run/secrets/nats.creds" }
```

`[broker.jet_stream]` requires the `jetstream-broker` cargo feature on
`jammi-db`; selecting it without the feature is a load-time
`JammiError::Config`, never a panic at session construction.

```toml
[broker.postgres]
# url = "postgres://user:pass@host:5432/jammi"   # optional; defaults to
#                                                 # `catalog.postgres.url`
idle_poll_secs = 5
```

`[broker.postgres]` is a `LISTEN`/`NOTIFY` wake-up transport over the topic's
own mutable backing table — it carries no cargo feature, no bytes, and no
separate log: `url` defaults to `catalog.postgres.url` and MUST name the SAME
Postgres database on every replica (`NOTIFY` is scoped to one instance; a
replica pointed elsewhere silently degrades to `idle_poll`-only delivery,
never data loss). A SQLite catalog with no explicit `url` here is a load-time
`JammiError::Config` naming both keys. `idle_poll_secs` (default 5, must be
`>= 1`) bounds how long a lost `NOTIFY` can go undetected before the next
poll wakes every topic. The broker itself opens up to three dedicated
Postgres connections (one `PgListener`, up to two for `NOTIFY`) — never the
catalog's own pool — but every trigger-stream replay (a tail's own
driver-triggered replay, a lagging subscriber's own catch-up, and a fresh
subscriber's subscribe-time drain) runs one STEP at a time, and each step
borrows one CATALOG-pool connection only for its own duration: the permit is
released between steps, so a long multi-step catch-up never monopolises a
connection. Concurrent replay STEPS across the process are bounded at
`pool_size − 2` (minimum 1), one connection per step, leaving two
connections for publishers; size `catalog.postgres.pool_size` for the
number of `(topic, tenant)` tails you expect to be replaying at the same
moment plus ordinary writers, independent of this broker's fixed
three-connection budget. See
[Catalog Backend and Trigger Broker](./catalog-and-broker.md) for the full
trade-off discussion, the health probe, and the SQLite single-process
contract.

**`signing_key`** — where the audit HMAC master key comes from. Default:
`env` (`JAMMI_AUDIT_MASTER_KEY`, a runtime knob outside this config layer's
own `JAMMI_*` namespace — see [below](#environment-variable-overrides)).

```toml
signing_key = "env"
```

```toml
[signing_key.file]
path = "/run/secrets/jammi-audit-master-key"
```

The file form is read at each signing request (not at config load), so a
rotated mount is picked up without a restart.

**`storage`** — the object-storage root for result tables, and the default
cloud driver credentials for both the result root and any cloud data source
whose registration carries no inline credentials. Default: unset (result
tables live on local disk under `artifact_dir`; cloud sources fall back to
the SDK's own ambient credential chain).

```toml
[storage]
result_root = "s3://jammi-results/prod"

[storage.cloud.s3]
region = "us-east-1"
```

`[storage.cloud]` is itself externally tagged over `s3` / `r2` / `gcs` /
`azure`; a bare `storage.cloud = "s3"` also selects a variant with its
per-field defaults. See
[Store Sources and Results in Cloud Object Storage](./cloud-storage.md) for
every provider's fields, the credential precedence, and the segment layout on
disk.

**`models`** — the Hugging Face Hub cache root, endpoint, token, and offline
switch. Default: every field unset (cache root falls back to `HF_HOME`, then
the platform home directory; endpoint falls back to `HF_ENDPOINT`, then the
Hub's own default; token falls back to `HF_TOKEN`, then the cache's own
`token` file).

```toml
[models]
hub_endpoint = "https://huggingface.co"
hub_cache_dir = "/var/cache/jammi/hub"
hub_token = { file = "/run/secrets/hf-token" }
offline = false
```

`offline = true` refuses every Hub network fetch: a model loads only from a
`local:` reference or an already-resolved catalog row (a warm, on-disk Hub
cache with no catalog row is still a miss). It does not reach the fine-tune
worker's adapter fetch, which always reads from the artifact store, offline or
not. See [Use a Local Model Checkpoint](./local-models.md).

## Environment variable overrides

Every field in the config tree is overridable — not a hand-enumerated subset
— through one namespace rule, deep-merged over the file layer (an
environment value wins field-by-field; see the layering order above and
`JammiConfig::parse_from`/`load_from`).

**Namespace.** `JAMMI_<X>__<path>` (segments joined by `__`) is **always**
config: an unknown `X` — one that does not name a top-level `JammiConfig`
field (`artifact_dir`, `engine`, `gpu`, `inference`, `embedding`,
`fine_tuning`, `lease`, `worker`, `jobs`, `cache`, `server`, `logging`,
`catalog`, `broker`, `signing_key`, `storage`, `models`) — is a load-time
error naming the
variable, never a silent no-op (`JAMMI_CATALOG__KIND=postgres`, a typo one
segment short of `JAMMI_CATALOG__POSTGRES__URL`, refuses rather than quietly
running SQLite with nothing to explain why). A bare `JAMMI_<X>` with **no**
`__` is config *iff* `X` exactly names one of those same top-level fields
(`JAMMI_ARTIFACT_DIR`, `JAMMI_CATALOG=sqlite`, …); every other `JAMMI_*` name
is a runtime knob outside this layer's namespace and is silently ignored here
— `JAMMI_AUDIT_MASTER_KEY`, `JAMMI_CONFIG` (which names the config *file* to
load, not a field override), `JAMMI_TEST_PG_URL`, and similar single-purpose
variables all pass through untouched.

**Path segments and TOML syntax.** Everything after the first segment is
lowercased on the way in, matching every config struct's `snake_case` field
names — `JAMMI_STORAGE__CLOUD__S3__REGION` reaches `storage.cloud.s3.region`
regardless of case. A leaf value is parsed as the field's own type; a value
naming a list or map field is parsed as **TOML** —
`JAMMI_SERVER__PRELOAD_MODELS='["a", "b"]'`,
`JAMMI_INFERENCE__HTTP__HEADERS='{ X-Api-Key = "v" }'` — so a map value's own
keys keep the case written in the TOML (only the path segments that route to
the map are lowercased). `services` is the one field with its own grammar
instead of TOML syntax — see below.

**Refusals name the variable.** An unknown section, an unknown key, or a
value outside a field's domain is a load-time error naming the offending
`JAMMI_*` variable — never a silent drop and never a fall-back to the file's
value or the default. `JAMMI_WORKER__ENABLED` (boolean) accepts `true`,
`false`, `1`, `0`, case-insensitively and with surrounding whitespace
trimmed; any other value — including an empty one — is refused by name: a
yes/no question about what the process will do has no safe direction to
guess in.

**`services`.** `JAMMI_SERVER__SERVICES` takes the same grammar the TOML
field does: exactly `all` (case-sensitive — `ALL` is a one-token tier list,
rejected as an unknown tier name) selects all-in-one; a comma-separated list
(`event,eval`, empty tokens filtered, so `""` means serve-only) selects
exactly those tiers. See [Service tiers](./deploy-server.md#service-tiers).

**Secrets.** A `Secret`-typed field (`catalog.postgres.url`,
`broker.jet_stream.credentials`, `broker.postgres.url`,
`inference.http.headers` values, the cloud
credential fields, `models.hub_token`) takes the value inline
(`JAMMI_CATALOG__POSTGRES__URL=…`)
or as a file reference via the `__FILE` suffix
(`JAMMI_CATALOG__POSTGRES__URL__FILE=/run/secrets/pg-url`) — the TOML-side
mirror of `url = { file = "…" }`. Both spellings at the same path is a
collision error.

**No `${VAR}` interpolation in environment values.** `${VAR}` substitution
(see the loading order above) runs once, over the TOML *file* text, before
that layer is parsed — an environment variable's own value is taken
**verbatim**, never re-interpolated.

**Resolution order** (for the config *file* itself, not the override layer):
an explicit path, `JAMMI_CONFIG`, `./jammi.toml`, `/etc/jammi/jammi.toml`,
then the platform per-user config directory. First existing path wins; when
none exists the config is defaults-plus-environment-overrides only.
