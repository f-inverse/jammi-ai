# Operability

How to run a Jammi server in production: what it exposes for observability, how
it shuts down cleanly, the resource limits it enforces, and how it behaves when
a dependency fails. Everything below describes the system as it ships today.

## Observability surface

The server exposes three HTTP side-channel endpoints, independent of the gRPC
and Flight SQL data paths:

| Endpoint   | Meaning                                                                                         | Status |
|------------|-------------------------------------------------------------------------------------------------|--------|
| `/healthz` | Liveness — dependency-free `200` with the build version. The process is up and serving.         | `200`  |
| `/readyz`  | Readiness — pings the catalog backend the session is bound to. Use it for load-balancer admission. | `200` ready / `503` not ready |
| `/metrics` | Prometheus text-format snapshot of the substrate metric registry.                               | `200`  |

`/healthz` answers without touching any dependency, so an orchestrator uses it
to decide whether to *restart* the container. `/readyz` goes one step further
and pings the catalog — a transient catalog outage returns `503` so the load
balancer removes the instance from rotation rather than restarting it.

```bash
$ curl -s localhost:8080/healthz
{"status":"ok","version":"0.29.0"}

$ curl -s localhost:8080/readyz          # catalog reachable
{"status":"ready"}

$ curl -s localhost:8080/readyz          # catalog unreachable → 503
{"status":"not_ready","detail":"catalog ping failed: connection refused"}
```

### Metrics

`/metrics` emits four substrate-level metrics that the gRPC services and Flight
SQL layer feed:

| Metric                          | Type      | Incremented by                                              |
|---------------------------------|-----------|-------------------------------------------------------------|
| `jammi_grpc_requests_total`     | counter   | Any `/jammi.v1.*` gRPC request.                             |
| `jammi_flight_queries_total`    | counter   | A Flight SQL `DoGet` query.                                 |
| `jammi_eval_invocations_total`  | counter   | An `EvalService/*` RPC.                                     |
| `jammi_search_latency_seconds`  | histogram | End-to-end `EmbeddingService/Search` request latency.       |

```text
# HELP jammi_grpc_requests_total Total number of gRPC requests served across all jammi.v1 services.
# TYPE jammi_grpc_requests_total counter
jammi_grpc_requests_total 1432
# HELP jammi_search_latency_seconds Vector-search request latency, in seconds.
# TYPE jammi_search_latency_seconds histogram
jammi_search_latency_seconds_bucket{le="0.05"} 311
jammi_search_latency_seconds_bucket{le="0.1"} 402
jammi_search_latency_seconds_sum 27.41
jammi_search_latency_seconds_count 418
```

### Tracing

The server installs a global `tracing` subscriber. Spans carry the correlation
fields that let you follow a request across the gRPC surface and the worker
fleet:

- **gRPC handler spans** carry `tenant_id` — recorded once the handler has
  resolved the request's tenant scope.
- **`run_claimed_job`** (the worker dispatching a claimed job) carries
  `worker_id`, `job_id`, and `tenant_id`.
- **`run_spec`** (the training run inside a claimed job) carries `job_id` and
  `worker_id`.

Logs are emitted as structured records, JSON or human-readable text per
`logging.format` (`LogFormat`). The filter comes from `logging.level`, with
`RUST_LOG` as an optional override. Output always goes to stdout — a server runs
non-interactively by design — and ANSI colour is enabled only when stdout is a
terminal.

```json
{"timestamp":"2026-06-15T04:17:33.114Z","level":"INFO","fields":{"message":"job completed"},"target":"jammi_ai::fine_tune::worker","span":{"job_id":"job-7af3","worker_id":"worker-2","tenant_id":"acme","name":"run_claimed_job"}}
```

## Graceful shutdown

`run_with_shutdown` drives both the HTTP side-channel and the gRPC surface and
drains them in parallel: the call returns once both have stopped accepting new
connections and finished serving in-flight requests. The standalone binary wires
both `SIGINT` (Ctrl+C) and `SIGTERM`, so `docker stop` — which sends
`SIGTERM` — triggers the same clean drain as an interactive Ctrl+C.

## Backpressure and resource limits

The engine enforces these limits. They are the only ones it enforces — there is
**no configured gRPC message-size cap** and **no in-memory worker-queue-depth
bound**; the work queue is durable, not buffered (see below).

### Worker timing

The training worker drives its loop on three intervals, defaulting to 30 s lease
/ 10 s heartbeat / 1 s idle-poll:

- **Lease (30 s default)** — how long a claimed job is exclusively owned before
  it becomes reclaimable.
- **Heartbeat (10 s default)** — renews the lease well inside the window.
- **Idle-poll (1 s default)** — how often an idle worker checks for new work;
  reclaim runs on each idle tick, so a dead worker's job is recovered within
  roughly one poll plus one lease.

The config layer enforces the invariant `heartbeat × 2 < lease` (and rejects a
zero heartbeat or zero idle-poll). This guarantees a live worker renews at least
twice per lease, so a single missed beat still leaves one in-window renewal that
lands strictly before expiry — never coincident with it, which would race an
idle-polling worker's reclaim. Bad values are rejected at config time, never
silently clamped.

### Job attempts cap

A job is retried at most **3 times**. After the third attempt the expired-lease
reclaim path fails the job for good rather than re-queueing it indefinitely.

### GPU admission — a memory budget

GPU admission is a **memory budget**, not a max-concurrent-job count. The
scheduler admits work against a budget of
`total_gpu_memory × (1 − headroom_fraction)`: a reservation is admitted by a
compare-and-swap against the reserved total, and released via RAII when the
permit drops. Many small jobs can run concurrently while one large job is
admitted only when its memory fits the remaining budget.

### Work queue

The work queue is the durable Postgres `training_jobs` table, drained with a
`SELECT … FOR UPDATE SKIP LOCKED` claim so concurrent workers each lock a
distinct row. It is bounded by the lease plus the attempts cap, not by an
in-memory buffer — there is no in-process queue-depth limit to overflow, and a
worker crash leaves the row claimable again after the lease expires.

## Failure-mode matrix

| Failure | Observed behavior | Recovery mechanism | Signal (metric/log) | Proving test |
|---------|-------------------|--------------------|---------------------|--------------|
| **Storage dies mid-publish** | No half-written committed artifact. The crashed worker's per-attempt prefix is orphaned because its finalize CAS never ran. | Winner-only commit: each attempt writes a unique `{job}/{worker}/{attempt}` prefix; the served `artifact_path` is written solely by the finalize CAS, so the committed pointer roots under the winner's prefix. | Final `training_jobs` row's `artifact_path` resolves to the winner's prefix; reload returns the winner's bytes. | **Proven** by `tests/distributed/artifact_crash_window.rs`. |
| **Worker dies** | The claimed job is reclaimed by a different worker after the lease expires and completes exactly once. | Lease expiry + idle-tick reclaim; the `FOR UPDATE SKIP LOCKED` claim guarantees a single new owner. | The finalized row's `claimed_by` is a different worker id; reclaim runs each idle tick (worker log). | **Proven** by `tests/distributed/kill9_reclaim.rs` (plus `exactly_one_claim.rs` for the N-worker claim race and `cross_tenant_isolation.rs` for tenant scope). |
| **GPU dies** | — | Memory-budget admission releases the permit via RAII on the failing path, but in-flight GPU-fault recovery is not yet validated end-to-end. | — | **Honest gap: not yet proven (1.0-deferred).** The distributed lane is CPU-only, so no chaos test exercises a GPU fault. |
| **Broker dies** | The trigger stream is a **separate subsystem** from the training worker fleet — claim and lease are pure Postgres, with no broker coupling — so a broker outage does not stall training. A broker fan-out failure is best-effort: the publisher has already committed the augmented event (with its engine `_offset`) to the durable backing table before fanning out, so the event is never lost — subscribers replay it from the backing table on reconnect. | **At-least-once + replay-completeness**: the backing table is the authoritative log; a subscriber attaches at an engine `_offset`, replays `[from..last_replayed]` from the table, then joins the live broker tail *with overlap* and dedups by engine `_offset` so no committed offset is ever skipped across the replay/live seam. The seam is keyed on the engine `_offset` alone — never on a broker-native sequence (JetStream's stream sequence is an independent counter that skews permanently after any post-commit fan-out failure). | Replayed offsets are contiguous from `from_offset`; the live tail resumes with no gap (broker integration test + in-mem property test). | **At-least-once + replay-completeness PROVEN.** In-memory + crash-mid-publish: `jammi-db/tests/it/trigger.rs` (`crash_mid_publish_replays_committed_offsets_with_no_loss`, `live_tail_resumes_with_no_loss_after_post_commit_fan_out_failure`, `at_least_once_no_skip_property_over_randomized_states`). Live JetStream consumer-recreate resume: `jammi-db/tests/it/trigger_jetstream.rs` (`consumer_recreate_resumes_engine_offsets_with_no_loss`, gated `live-broker-tests`). **Exactly-once is NOT provided by either backend** — dedup downstream by the `(_offset, _row_idx)` composite key. At-least-once is bounded by the backing log's durability (see below): process-crash-durable always; full on Postgres; on SQLite a host power-loss can lose the last committed backing-table row(s) since the previous checkpoint. |
| **Crash mid-publish of a result table** | No half-written result table is ever queryable. On restart every table left `building` is reconciled to exactly one terminal state — `ready` if its Parquet is a fully-valid closed file (promoted with the *true* footer row count, and the ANN sidecar rebuilt from the Parquet so an embedding table self-heals), `failed` otherwise (missing or torn bytes, which are then reaped). | Crash-consistent eventual reconciliation: the bytes are written first, then a single catalog row flips `building → ready`; the startup sweep classifies each `building` orphan against the bytes on disk. The sweep runs cross-tenant (admin scan), so every tenant's orphan is reconciled and each keeps its own `tenant_id`. | `Recovery: …` `WARN` logs name each reconciled table and its disposition; no row remains `building`. | **Proven** by `jammi-db/tests/it/recovery.rs` — each torn state (missing bytes, truncated Parquet, valid-but-unfinalized Parquet, finalize-ordering window, ready-but-missing-bytes, two tenants' orphans) is constructed directly, then the real `recover()` + `load_existing_tables` asserts invariants I1–I6. |

### Catalog durability under crash vs. power loss

Result-table crash-consistency reconciles whatever the catalog *durably retained* against the bytes on disk, so the catalog's own durability setting bounds the guarantee:

- **Process crash** (the engine dies, the host survives): both backends replay their write-ahead log on restart, so a `building → ready` (or the `building` insert recovery later reconciles) that committed before the crash is present after it. No committed catalog state is lost.
- **Host power loss**: Postgres commits synchronously (`fsync` per commit by default), so a committed transaction survives. SQLite runs `synchronous=NORMAL` under WAL — it fsyncs at checkpoint, not on every commit — so a power loss can lose the last committed transaction(s) since the previous checkpoint. A row that was *not* durably retained simply isn't seen by recovery; the bytes it would have pointed at are reaped as an orphan on a later sweep. This is a property of the catalog backend's durability configuration, not of the reconciliation.

The trigger stream's at-least-once guarantee inherits exactly this bound, because its durable log is the same kind of backing table written through the same backend transaction. Under a **process crash** the at-least-once guarantee is unconditional — the committed backing-table rows replay on reconnect. Under a **host power loss** the durable log itself is power-loss-bounded: on Postgres a committed publish survives; on SQLite (`synchronous=NORMAL` under WAL) the last committed backing-table row(s) since the previous checkpoint can be lost, and an offset whose row was not durably retained will not replay. At-least-once is therefore *full* on Postgres and *power-loss-bounded* on SQLite — never weaker than the backing log's own durability.
