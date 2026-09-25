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
# The engine's CPU parallelism budget — the one setting that bounds all of it:
# the query engine's partitions, the model forwards a CPU device runs at once,
# and the process-wide pool CPU tensor math and media preprocessing run on
# (`jammi-server` and the Python engine size that pool from this at startup; a
# Rust process embedding the engine as a library owns the pool itself).
# Default: the CPU count the OS reports. Set it where a container is allotted
# fewer cores than it can see — the engine cannot detect that from inside.
execution_threads = 8
# Memory limit for the query engine's DataFusion session: this becomes the
# byte size of the pool every plan and every engine-side memory reservation
# (a training-set stream's chunk, an eager read's collected batches) is
# bounded by. An operator that can spill (a sort, a sort-merge join) is held
# to an equal share among the spilling operators holding memory at that
# moment, and spills at it rather than growing past it. Three forms:
#   - "<n>%"      -- that percentage (1-100) of the HOST's total physical
#                    memory (a Linux cgroup ceiling is honoured when it is
#                    lower than the host total and readable), resolved once
#                    at session build.
#   - "<n>GB"/"<n>MB"/"<n>KB" -- n binary (1024-based) units.
#   - "<n>"       -- n bytes, unadorned.
# Anything else, or a resolved value below the 64 MiB floor, is refused at
# load, naming the key and (for the floor) the floor. A query or engine-side
# reservation that would grow the pool past this limit fails with a typed
# `ResourcesExhausted` error rather than silently exceeding it. Default: "75%".
memory_limit = "75%"
# Maximum rows per DataFusion batch. Default: 8192.
# Deployment rule: a single batch larger than `memory_limit` above cannot be
# sorted (a spilling external sort still needs one batch resident), which
# matters most for a training-set materialization's full-tuple sort over wide
# rows — size this down (and `memory_limit` up) when a row is large.
batch_size = 8192

[gpu]
# GPU device index. -1 for CPU only. Default: 0 in a build with an accelerator
# (the CUDA server or wheel, a Metal build), -1 in a CPU-only build.
device = -1
# Each device's model-residency budget, in the same grammar as
# [engine] memory_limit: "<n>%" of the device's total memory, or an absolute
# "<n>GB"/"<n>MB"/"<n>KB"/"<n>". What the budget leaves of the card is
# headroom for activations and workspace. An absolute budget larger than a
# device is refused when the session opens on it. Default: "90%".
memory_limit = "90%"
# Fail fast if the requested GPU is unavailable instead of falling back to CPU.
# Default: false (degrade to CPU with a warning).
require_gpu = false
# Default inference compute precision: "f32" or "f16". A model may override
# this with its own "compute_precision" in config.json; the per-model value
# wins. "bf16" is a valid value for fine-tune's frozen-backbone dtype but is
# rejected at inference load time (not yet supported). Default: "f32".
compute_precision = "f32"

[inference]
# The chunk budget: what bounds one model forward. A forward chunk is cut
# from the input ordered by row cost — a text row's token count, then the
# key — under both caps, whatever the fan-out below, so rows that share a
# forward are nearly equal in length and pad to little more than their real
# length. batch_size is the most rows one forward takes; batch_tokens the
# most padded tokens (the chunk's rows times the width they are padded to:
# a multiple of 8, within an eighth of the longest row, up to the model's
# own sequence limit) — the bound on a
# forward's activation memory. A row longer than batch_tokens still
# forwards alone. 0 is refused at load for either.
# Defaults: 32 rows, 16384 tokens (32 rows of a 512-token encoder).
batch_size = 32
batch_tokens = 16384
# The most IDLE models kept loaded, across the process's devices. Past it
# the least recently used idle model is evicted; a model in use is never
# evicted, so the count can exceed this while models are held. 0 = unbounded
# (device memory still bounds what is admitted). Default: 0.
max_loaded_models = 0
# The most model descriptions kept memoized — what a plan is built against
# without loading a model, a few kilobytes each. Past it the least recently
# used is dropped and recomputed (its files re-hashed) if described again.
# 0 = unbounded. Default: 1024.
max_described_models = 1024
# The inference fan-out: how many partitions of one plan forward chunks
# concurrently — threads of one process, or tasks of a cluster when the plan
# is submitted to one. The rows a model forwards together are decided by
# the row costs and the chunk budget alone, so the written bytes are
# identical at every value. The
# DEVICE admits forwards — one at a time on a GPU, the core count on the CPU —
# across every plan and partition running on it, so a fan-out wider than the
# device admits queues rather than oversubscribes. 1 is the default and the
# minimum: 0 is refused at load, never silently treated as 1. Default: 1.
partitions = 1

[embedding]
# Batches between two progress checkpoints on a building embedding table's
# catalog row; 0 never checkpoints. Default: 1000.
checkpoint_interval = 1000
# Rows per ANN segment of a written embedding table. The segments are
# consecutive runs of the table's rows (key order) at this budget, each built
# on its own thread as its rows are written; a query fans out over them, so a
# smaller budget builds sooner and wider, a larger one searches fewer graphs.
# Default: 4096.
index_segment_rows = 4096

[embedding.ann]
# The HNSW sidecar index built beside every embedding table. 0 = the
# backend's default for each graph knob.
# Edges per graph node (HNSW M): larger builds a bigger, slower-to-build,
# higher-recall graph.
connectivity = 0
# Candidate-list width while building the graph (ef_construction).
build_expansion = 0
# Candidate-list width while searching (ef_search), at least k: wider finds
# more of the true neighbours per query at more work. A query-time setting —
# it applies to indexes already built. Measure it against `search(exact=True)`.
search_expansion = 0
# Precision new tables' indexes are built at: "f32", "f16", "int8", "binary".
# A quantized index retrieves k * oversample candidates and rescores them
# exactly. Default: "f32".
storage_precision = "f32"
# That candidate multiplier; unset uses the precision's own default
# (32 for binary, 4 otherwise). A table keeps the value it was created with.
# oversample = 4

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
# How often a worker-enabled process samples the queue-depth gauges
# (`jammi_jobs_queued{kind}` / `jammi_jobs_running{kind}`) from the catalog:
# one grouped count per tick on a dedicated task, never on a `/metrics`
# scrape and never on the claim loop. Must be >= 1. Default: 5.
metrics_sample_secs = 5
# How many ranks THIS HOST places on its own `[gpu] devices` for a
# distributed training job it runs entirely in-process - one rank per
# device, rank `i` on `[gpu] devices[i]`. Must be >= 1 (the default, 1, is
# the single-rank deployment: no gang, no collective) and never more than
# the configured device count. Orthogonal to a submitted job's own
# `world_size` (a separate, per-job knob) and to `[distributed]
# max_world_size` (the fleet-wide bound on a `Peer` gang across hosts) -
# the three knobs load independently, with no cross-check between any
# pair. A claimed job whose `world_size` is within `local_ranks` runs every
# rank in this process over a `Local` gang; one wider than `local_ranks`
# makes this process rank 0 of a `Peer` gang whose other ranks are fleet
# members it assembles and dials.
local_ranks = 1
# Which collective a multi-rank worker reduces gradients over. Default:
# "auto" (the best collective this process can actually reach: NCCL on a
# CUDA build, the host CPU reduction otherwise). "nccl" on a build without
# the `cuda` feature is refused at session open. Configuration, not a build
# feature.
collective = "auto"
# How long a rank waits on its peers at a gang boundary before the wait is a
# failure: on the coordinator, a member silent this long retires the attempt
# (requeued from its checkpoint, one attempt spent). Must be > 0 - a zero
# deadline expires before any peer can answer, turning every gang into an
# immediate failure. Default: 120.
rank_timeout_secs = 120

[distributed]
# The widest `Peer` gang any coordinator on this deployment may admit,
# bounding a job's own `world_size` ACROSS FLEET MEMBERS: a submitted
# `world_size` past it is refused at submit, from configuration alone; one
# within it submits even when it is wider than this host's own `[gpu]
# devices`, and is decided by assembly on the claiming coordinator. Loads
# independently of `[worker]`'s own per-host rank count -- the two knobs
# are checked against each other by nothing. Must be >= 1 (1, the default,
# admits no fleet gang at all).
max_world_size = 1

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
# Models to load into the cache before /readyz reports ready and before this
# process's claim loop claims anything. A bare id takes its task from the
# catalog's `models` row; `{ id, task }` names it (required for a `local:`
# path). A model that cannot load, a bare id with no row, or an unknown task
# token is a startup error (the server exits non-zero). Default: [].
preload_models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    { id = "local:/models/bge-small", task = "text_embedding" },
]
# The INTERNAL peer listener for beyond-one-node retrieval and multi-host
# gang admission: the address this replica serves `jammi.v1.peer.PeerService`
# (segment search for the segments it owns) AND `jammi.v1.gang.GangService`
# (RunRank -- a coordinator admitting this replica into a multi-host training
# run) on, to OTHER replicas/coordinators of the same deployment. Unset (the
# default) = no third listener = single node = no gang admission surface.
# A replica is a segment owner and a gang admission member iff this is set.
# Must differ from health_listen and flight_listen at a fixed port (`:0`
# never collides). I-PEER / I-GANG: every client of this listener is a jammi
# coordinator -- the owner/member trusts the channel; the peer side binds no
# tenant and enforces only that each requested segment belongs to the named
# table (the coordinator resolved that table through its own tenant-scoped
# catalog read before fanning out); the gang side never reads the caller's
# tenant either -- a rank's tenant is derived from the verified `jobs` row,
# and its training set is resolved under that tenant alone. Bind it on a
# private interface behind
# network policy / mTLS from the runtime: on a routable interface without
# them it exposes cross-tenant reads. See security.md "The peer listener"
# and "The gang listener".
# peer_bind = "10.0.0.5:8082"
# The address OTHER replicas dial THIS process's `peer_bind` listener at
# (`peer_bind` is commonly `0.0.0.0:PORT`, unusable as a dial target).
# Unset (the default) = this process never advertises a gang-membership row:
# its `instances.peer_addr`/`result_root` columns stay NULL regardless of
# whether `peer_bind` is set. Setting it means: this process ADVERTISES
# itself as a gang member. Requires `peer_bind` to be set too -- refused,
# naming both keys, by `InstanceRegistration::from_config`, called once by
# every session construction path (and, for this early-failure check alone,
# by `JammiConfig::load_from` at config load time too).
# The membership root rule: `instances.result_root` carries the VERBATIM,
# byte-for-byte output of `resolved_result_root()` -- the exact same string
# the result store is rooted at ({artifact_dir}/jammi_db when [storage]
# result_root is unset, else result_root itself) -- and
# `instances.result_root_identity` carries that root's IDENTITY across
# spellings, computed once by this process at registration from the same
# config the store reads: scheme aliases folded by the store's own URL
# parser (gcs://=gs://, abfss://=azure://), the bucket as spelled and
# the key normalised by the store's own key parser, the location determinants
# (the endpoint/account/base URL the driver dials) read back from the very
# builder the store constructs -- environment first, [storage.cloud] on top,
# every spelling object_store accepts -- a local root CREATED
# (as the store creates it at open) and canonicalised on this host's
# filesystem (symlinks, ./.., the filesystem's own spelling). Only members
# whose identity equals this process's are its gang members. A memory://
# root is refused here: it lives in this process alone and can never be
# shared with a peer. See "The gang listener (I-GANG)" in security.md.
# peer_advertise = "10.0.4.7:9000"
# MARGINAL-LOAD ADMISSION per query, in bytes (a plain integer): the maximum
# estimated bytes ONE query may load locally for segments it does not own,
# when their owners are unreachable -- the last rung of the placed-search
# failure ladder (see "Beyond one node" in reference-topologies.md). Unset
# (the default) = unbounded. It is NOT a memory cap: the
# segment cache never evicts, earlier queries' loads are invisible to the
# check (each query loads afresh and frees on completion; the on-disk copy of
# a remote bundle persists), distinct remote segments accumulate on disk, and
# concurrent queries admit independently, so peak heap is
# concurrency x budget. The estimate per segment is
# row_count x (dimensions x bytes(precision) + 32 + 64) -- 4 (F32) / 2 (F16)
# / 1 (Int8) / ceil(d/8)/d (Binary) bytes per component, 32 bytes of row-id
# strings and 64 bytes of graph link overhead per row -- a LOWER bound for
# the quantized precisions: the rawf32 companion is excluded (it is a
# positioned read, never resident), but usearch's level-0 links and the
# row-id HashMap are unmodelled, so the true resident size exceeds it.
# Prescribe headroom: set the budget to at most half the memory you are
# willing to give one query's fallback loads. 0 is refused. Read by the
# result store; a library embedder sets it through the same config.
# peer_local_load_bytes = 268435456
# Which segment placement this session builds: "local" (the default -- every
# segment is this process's own, a single node regardless of what else is
# configured) or "rendezvous" (beyond-one-node retrieval over the LIVE
# `instances` ring: every segment is scored per live, root-sharing member --
# self included -- by a rendezvous hash, so membership changes move a near-
# minimal share of segments and every replica agrees on the owner with no
# coordination round). "rendezvous" REQUIRES peer_advertise to be set too --
# refused by name at the same membership choke point peer_advertise's own
# requirement is (InstanceRegistration::from_config / JammiConfig::load_from);
# unset peer_bind/peer_advertise is unaffected -- this knob changes nothing
# about an existing single-node deployment. See "Beyond one node" in
# reference-topologies.md.
# placement = "rendezvous"

[server.limits]
# Request-bounds and refusal policy for the combined gRPC + Flight SQL
# surface (also applied to the Flight-only listener). A request exceeding
# any of these is refused at the edge -- before any tenant-scoped catalog
# read runs, so a refusal never leaks cross-tenant existence -- with a typed
# gRPC status and a jammi_grpc_refused_total{reason} counter increment.
# Maximum inbound message size, in bytes, on EVERY listener: the public
# chain and the internal `peer_bind` listener (a gang round's chunks are
# sized to it). Must be > 0. Default: 67108864 (64 MiB). There is no
# outbound cap.
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
# Bounds a TriggerService.Subscribe or JobService.WaitJob stream. The
# server budget bounds the stream; the client imposes no deadline of its
# own by default (jammi-client's wait_job/subscribe send no grpc-timeout
# header). Three arms:
#   * a grpc-timeout header ABOVE this budget is refused at the edge,
#     before the stream ever opens (DEADLINE_EXCEEDED).
#   * a grpc-timeout header WITHIN this budget is ENFORCED by the server
#     itself, at the caller's own declared deadline -- the stream ends
#     with DEADLINE_EXCEEDED once that (shorter) duration elapses, not
#     the wider budget. This is deliberate: nothing else bounds a
#     streaming response body already returned, so a caller that declares
#     a deadline and then ignores it would otherwise hold the stream open
#     (and its permit held) past its own declared timeout.
#   * NO grpc-timeout header at all (the default for jammi-client, and for
#     any header-less caller) is NOT refused -- this budget itself becomes
#     the stream's own deadline, ending it with DEADLINE_EXCEEDED once it
#     elapses, wherever the stream then stands.
# Unset (the default) means no cap -- a stream runs until terminal
# (WaitJob) or indefinitely (Subscribe).
# wait_timeout_secs = 300
# Cap on concurrently open TriggerService.Subscribe streams. 0 = unbounded.
# Default: 256.
max_subscriptions = 256
# Cap on concurrently open JobService.WaitJob streams. 0 = unbounded.
# Default: 1024.
max_job_waits = 1024

# [ballista]
# The three compute-plane roles, held in any combination: a process hosts
# a Ballista scheduler iff `[ballista.scheduler]` is present, an executor
# iff `[ballista.executor]` is present, and is a client of a scheduler iff
# `[ballista.client]` is present. Unset (the default, the whole `[ballista]`
# table absent) means no role -- the process runs exactly as it always has,
# byte-for-byte. A role is a listener-shaped knob: the scheduler and the
# executor bind what they serve, the client names what it dials. A
# scheduler and an executor on one process is the single-node cluster; a
# process that also names itself as a client submits its own claims and
# materializations to the scheduler it hosts, its own executor excluded
# from a training attempt it submits (a claimant's host is never bound its
# own attempt; a materialization may run on it).
# Trust class: every listener this table opens (the scheduler's gRPC below,
# the executor's task gRPC and Flight shuffle in `[ballista.executor]`) is
# the peer listener's class, I-PEER -- unauthenticated, every client a jammi
# role, tenant scope enforced at the submitting session (see the security
# guide, "The Ballista listeners"). Bind them on the cluster-internal
# network and owe them the same network policy as `[server] peer_bind`.

# [ballista.scheduler]
# This process hosts a Ballista scheduler iff this table is present.
# The scheduler's gRPC listener. Default: "0.0.0.0:50050".
# bind = "0.0.0.0:50050"
# The host executors dial to report a placed task's status back to this
# scheduler: the scheduler stamps `advertise_host:port` into every task it
# places. REQUIRED when `bind`'s host is unspecified (`0.0.0.0`/`::`) -- an
# executor can never dial an unspecified host, and a task whose completion
# is never reported holds its executor slot forever. Unset (the default)
# means the `bind` host, valid only when `bind` already names a real
# interface.
# advertise_host = "10.0.4.7"

# [ballista.executor]
# This process hosts a Ballista executor iff this table is present. Unset
# (the default, table absent) means no executor role.
# The scheduler this executor registers with and takes tasks from,
# `host:port` -- a `SocketAddr` literal or a DNS name and port (the
# Kubernetes case). Required whenever `[ballista.executor]` is present.
# scheduler_address = "10.0.4.7:50050"
# This executor's Arrow Flight (shuffle) listener. Default: "0.0.0.0:50051".
# bind = "0.0.0.0:50051"
# This executor's gRPC (task) listener. Default: "0.0.0.0:50052".
# grpc_bind = "0.0.0.0:50052"
# The host other executors/the scheduler dial to reach this executor.
# REQUIRED when `bind`'s host is unspecified (`0.0.0.0`/`::`) -- the
# scheduler dials this address back to register the executor and push
# tasks, and an unspecified host never resolves on the scheduler's side of
# that connection. Unset (the default) means the `bind` host, valid only
# when `bind` already names a real interface.
# advertise_host = "10.0.4.8"
# Local directory Ballista's shuffle writer stages files under. Unset (the
# default) means a fresh temporary directory per process (no object-store
# shuffle in v1).
# work_dir = "/var/lib/jammi/shuffle"
# Concurrent task slots this executor offers the scheduler. Must be >= 1.
# Default: 1.
# task_slots = 1

# [ballista.client]
# This process is a client of a Ballista scheduler iff this table is
# present: a result-table materialization -- `CREATE TABLE … AS`, an
# embedding, inference, refresh, as-of join or training-set build -- runs
# WHOLE on that scheduler's executors when a live executor holds every
# device kind the plan requires (the same admission a claimed training
# attempt gets): the compute AND the write, as one plan rooted in the
# result-table sink, which writes the table's bytes on the executor under
# the row's lease (taken from this process for the write, handed back
# after) and streams one summary back; this process then finishes the
# catalog side. A claimed training attempt of any kind -- a fine-tune, a
# graph fine-tune, a context predictor -- is placed there as one task, on an
# executor other than this process that lists the plan's device kind. A plan
# no live executor can hold -- or that the wire cannot carry -- runs in this
# process, logged as such, never parked. Wherever it runs, the table's
# manifest records the environment of the process that ran it: that
# process's device and the models it loaded. A statement that serves rows
# inline (a `SELECT`, a search) never leaves this process. Unset (the
# default) means every statement and claim runs in this process.
# The scheduler this client submits to, `host:port` -- a `SocketAddr`
# literal or a DNS name and port (the Kubernetes case). A dial target: it
# binds nothing and joins no collision check. Required whenever
# `[ballista.client]` is present.
# scheduler_address = "10.0.4.7:50050"
# The device kind (`cpu`, `cuda`, `metal`) this process's model plans are
# placed onto -- a deployment fact, not this process's hardware: a CPU query
# tier placing onto a GPU compute tier names `cuda`. Unset (the default)
# means the kind of this process's own compute device.
# device_kind = "cuda"
#
# `scheduler.bind`, `executor.bind`, `executor.grpc_bind`,
# `[server] health_listen`/`flight_listen`/`peer_bind` (configuration.md's
# `[server]` block) may never share a fixed port -- a collision is refused
# at load time naming both keys. Two addresses collide iff their ports are
# equal and non-zero AND their hosts are equal or either host is
# unspecified (`0.0.0.0`/`::` overlaps every interface, including
# `127.0.0.1`); an ephemeral `:0` never collides with anything.

[logging]
# Log level: "trace", "debug", "info", "warn", "error". Default: "info".
level = "info"
# Log format: "text" or "json". Default: "text".
format = "text"

[observability]
# OTLP/gRPC collector endpoint spans export to. Unset (the default) means:
# build no exporter and open no network connection at all -- a process with
# no configured endpoint attempts zero egress for tracing, whether or not
# the `telemetry-otlp` cargo feature is compiled in.
# otlp_endpoint = "http://localhost:4317"
# `service.name` resource attribute stamped on every exported span.
# Default: "jammi".
service_name = "jammi"
# Fraction of traces kept by the parent-based ratio sampler, in [0.0, 1.0].
# Default: 1.0 (sample everything).
sample_ratio = 1.0

# [observability.otlp_headers]
# Request headers the exporter attaches to every export call (e.g. a
# collector auth token). Each value is a secret -- a plain string inline, or
# `{ file = "/run/secrets/otlp-token" }` -- and is never logged. Default:
# empty.
# x-api-key = { file = "/run/secrets/otlp-token" }
```

`JobService.SubmitJob`'s `idempotency_key` is bounded to 256 bytes
(`MAX_IDEMPOTENCY_KEY_BYTES`, `jammi_db::catalog::jobs_repo`) — a fixed
engine bound, not a `[server.limits]` key. A longer key is refused with
`INVALID_ARGUMENT` naming the bound, never the key's own value. This closes
a real backend divergence: Postgres's btree index has a hard row-size
ceiling an oversize key can exceed (`index row size ... exceeds btree
version 4 maximum ...`), while SQLite silently accepts a key of any size —
without the bound, the same `idempotency_key` would be accepted on one
backend and refused on the other.

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
[broker.postgres]
# url = "postgres://user:pass@host:5432/jammi?sslmode=verify-full&sslrootcert=/etc/ssl/certs/ca-certificates.crt"
#                                                 # optional; defaults to
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
switch. Default: every field unset (cache root falls back to a non-empty
`HF_HUB_CACHE` — used directly as the cache root, nothing appended — then a
non-empty `HF_HOME`, then the platform home directory; endpoint falls back
to a non-empty `HF_ENDPOINT`, then the Hub's own default; token falls back
to a non-empty `HF_TOKEN`, then a non-empty `HUGGING_FACE_HUB_TOKEN`
(`huggingface_hub`'s own live legacy alias), then the token FILE
(`HF_TOKEN_PATH`, naming the file directly, else the `<HF_HOME>/token` file
— resolved independently of whichever tier won the cache-root fallback,
never derived from the cache root itself); offline falls back to a
non-empty `HF_HUB_OFFLINE`, then `TRANSFORMERS_OFFLINE` when
`HF_HUB_OFFLINE` is itself unset or present-but-empty, then `false`). A
present-but-empty `HF_*` value — the shape a Compose/Kubernetes env block or
a shell `export FOO=` produces — is treated identically to an unset one at
every one of these tiers, never as a literal empty value; the value used is
also always the TRIMMED string, never the raw one (a padded `HF_HOME` must
not silently resolve to a current-working-directory-relative root). One
exception is a genuine divergence from `huggingface_hub`, not merely a
stricter reading of it: a *whitespace-only* `HF_HUB_OFFLINE` falls through
to `TRANSFORMERS_OFFLINE` here, failing toward offline, where
`huggingface_hub` itself stops at the whitespace-only value and resolves
online.

```toml
[models]
hub_endpoint = "https://huggingface.co"
hub_cache_dir = "/var/cache/jammi"
hub_token = { file = "/run/secrets/hf-token" }
offline = false
```

`offline = true` refuses every Hub network fetch: a model loads only from a
`local:` reference or an already-resolved catalog row (a warm, on-disk Hub
cache with no catalog row is still a miss). It does not reach the fine-tune
worker's adapter fetch, which always reads from the artifact store, offline or
not. `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` are truthy for any of
`huggingface_hub`'s own `ENV_VARS_TRUE_VALUES` — `"1"`, `"on"`, `"yes"`,
`"true"`, case-insensitively, surrounding whitespace trimmed. See
[Use a Local Model Checkpoint](./local-models.md).

A model served at a remote endpoint is declared under
`[models.remote.<name>]` and referenced as `remote:<name>`:

```toml
[models.remote.hosted-encoder]
protocol = "openai_embeddings"            # the only protocol: text embedding
url = "https://api.example.com/v1/embeddings"
model = "text-embedding-small"            # the name the endpoint is asked for
dimensions = 1536                         # every response is held to this width
revision = "2026-01"                      # your pin of what the name serves
headers = { Authorization = { file = "/run/secrets/embeddings-key" } }
timeout_secs = 60                         # per request; default 60
max_in_flight = 4                         # requests at once; default 4
max_retries = 2                           # 429 / 5xx / timeout; default 2
```

A declaration that no request could be built from is refused at load, naming
the key: a `url` that is not an http(s) URL naming a host, an empty `model` or
`revision`, or a header that is not a header name. A credential that cannot be
read is refused when the session opens. See [Remote Models](./remote-models.md).

## Environment variable overrides

Every field in the config tree is overridable — not a hand-enumerated subset
— through one namespace rule, deep-merged over the file layer (an
environment value wins field-by-field; see the layering order above and
`JammiConfig::parse_from`/`load_from`).

**Namespace.** `JAMMI_<X>__<path>` (segments joined by `__`) is **always**
config: an unknown `X` — one that does not name a top-level `JammiConfig`
field (`artifact_dir`, `engine`, `gpu`, `inference`, `embedding`,
`fine_tuning`, `lease`, `worker`, `jobs`, `cache`, `server`, `logging`,
`observability`, `catalog`, `broker`, `signing_key`, `storage`, `models`) — is a load-time
error naming the
variable, never a silent no-op (`JAMMI_CATALOG__KIND=postgres`, a typo one
segment short of `JAMMI_CATALOG__POSTGRES__URL`, refuses rather than quietly
running SQLite with nothing to explain why). A bare `JAMMI_<X>` with **no**
`__` is config *iff* `X` exactly names one of those same top-level fields
(`JAMMI_ARTIFACT_DIR`, `JAMMI_CATALOG=sqlite`, …); every other `JAMMI_*` name
is a runtime knob outside this layer's namespace and is silently ignored here
— `JAMMI_AUDIT_MASTER_KEY`, `JAMMI_CONFIG` (which names the config *file* to
load, not a field override), `JAMMI_WORKER_ID` (below), `JAMMI_TEST_PG_URL`,
and similar single-purpose variables all pass through untouched.

**`JAMMI_WORKER_ID` is a label.** Every process mints its own
`instances.instance_id` (a UUID) at session construction — that id is what
`jobs.claimed_by` records and what the lease/liveness machinery keys on.
`JAMMI_WORKER_ID`, when set and non-empty (trimmed), is only the
`instances.label` shown beside that id by `ListWorkers` / `jammi workers` and
in logs: an operator-chosen, non-unique name (a node, a replica slot). Two
processes given the same label are two instances, so a replacement process
never inherits — or keeps alive — a dead namesake's claims.

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
`broker.postgres.url`, the cloud
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
