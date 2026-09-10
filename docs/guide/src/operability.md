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

`/metrics` emits these substrate-level metrics that the gRPC services, Flight
SQL layer, and request-bounds layer stack feed:

| Metric                          | Type      | Incremented by                                              |
|---------------------------------|-----------|-------------------------------------------------------------|
| `jammi_grpc_requests_total`     | counter   | Any `/jammi.v1.*` gRPC request.                             |
| `jammi_flight_queries_total`    | counter   | A Flight SQL `DoGet` query.                                 |
| `jammi_eval_invocations_total`  | counter   | An `EvalService/*` RPC.                                     |
| `jammi_search_latency_seconds`  | histogram | End-to-end `EmbeddingService/Search` request latency.       |
| `jammi_grpc_refused_total{reason}` | counter | A request refused by `[server.limits]` — `reason` ∈ `message_size`, `in_flight`, `in_flight_per_connection`, `subscriptions`, `job_waits`, `timeout`. See [Request bounds](#request-bounds-serverlimits). |

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

The server installs a global `tracing` subscriber (a `Registry` layered with
the `fmt` formatter and, when configured, an OTLP export layer). Spans carry
the correlation fields that let you follow a request across the gRPC surface
and the worker fleet:

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

#### OTLP trace export

Setting `[observability] otlp_endpoint` sends spans to any vendor-neutral OTLP
collector over gRPC (`opentelemetry-otlp`, tonic transport). Every span from
this process carries the `service.name` resource attribute (default
`"jammi"`), and the parent-based ratio sampler keeps `sample_ratio` (default
`1.0`, i.e. everything) of the traces this process ROOTS — a span whose parent
was already sampled by the caller is always kept, regardless of the local
ratio. Request headers a collector requires (an auth token, a tenant header)
go under `[observability.otlp_headers]`; each value is a secret — inline or
`{ file = "…" }` — and is never logged (see [Configuration](./configuration.md)).

Leaving `otlp_endpoint` unset installs no exporter and opens no network
connection for tracing at all — the zero-egress default. The exporter lives
behind the `telemetry-otlp` cargo feature (on by default in every published
`jammi-server` build and in the `jammi-python` embed wheel); a build compiled
WITHOUT the feature refuses to start with a typed configuration error if
`otlp_endpoint` is set, rather than silently dropping every span.

A whole-server tower layer extracts the incoming W3C `traceparent` (and
`tracestate`) from every gRPC/Flight request's HTTP headers and continues that
trace: the span this process opens for the request shares the CALLER's trace
id, so a request that entered through an edge proxy or gateway stays one trace
end-to-end across process boundaries. A request with no `traceparent` header
starts a fresh, unparented trace, exactly as today.

## Graceful shutdown

`run_with_shutdown` drives both the HTTP side-channel and the gRPC surface and
drains them in parallel: the call returns once both have stopped accepting new
connections and finished serving in-flight requests. The standalone binary wires
both `SIGINT` (Ctrl+C) and `SIGTERM`, so `docker stop` — which sends
`SIGTERM` — triggers the same clean drain as an interactive Ctrl+C.

## Backpressure and resource limits

The engine enforces these limits. There is **no in-memory worker-queue-depth
bound** — the work queue is durable, not buffered (see below) — but the
combined gRPC + Flight SQL surface (and, separately, the request-bounds knobs
below) DOES enforce a configured inbound message-size cap, request-concurrency
bounds, and two long-lived-stream budgets.

### Request bounds (`[server.limits]`)

Every inbound request to the combined gRPC + Flight SQL listener (and, for
message size, the Flight-only listener too) is checked against
`[server.limits]` before any tenant-scoped catalog read runs — a refused
request never reaches a handler, so a refusal leaks nothing about
cross-tenant existence. A request that would exceed any of these is refused
at the edge with a typed gRPC status and a
`jammi_grpc_refused_total{reason}` counter increment (see
[Metrics](#metrics) below); see [Configuration](./configuration.md) for the
full `[server.limits]` reference and every default.

| Knob | Refusal | `reason` label | Notes |
|------|---------|-----------------|-------|
| `max_message_bytes` (default 64 MiB) | `OUT_OF_RANGE` | `message_size` | Enforced by tonic's own per-service codec (`max_decoding_message_size`), below the refusal-counting layer stack — verified against the vendored tonic 0.14.5 source; this is NOT `RESOURCE_EXHAUSTED`. No outbound cap: a large result set is never truncated. |
| `max_in_flight` (default 256) | `RESOURCE_EXHAUSTED` | `in_flight` | Global, UNARY methods only. `0` = unbounded. |
| `max_in_flight_per_connection` (default 64) | `RESOURCE_EXHAUSTED` | `in_flight_per_connection` | Per TCP connection, UNARY methods only. `0` = unbounded. |
| `request_timeout_secs` (default unset) | `DEADLINE_EXCEEDED` | `timeout` | UNARY methods only; unset means no server-imposed timeout. |
| `wait_timeout_secs` (default unset) | `DEADLINE_EXCEEDED` | `timeout` (edge refusal only) | Bounds a `TriggerService.Subscribe` / `JobService.WaitJob` stream: a `grpc-timeout` ABOVE the budget is refused at the edge (before the stream opens, counted under `timeout`); a header WITHIN the budget is ENFORCED by the server itself, ending the stream with `DEADLINE_EXCEEDED` at the caller's own declared deadline (uncounted — fires mid-stream, after the edge; tonic's own `GrpcTimeout` never bounds a streaming response body already returned, so this is deliberate, not merely honoured as-is); NO header at all is NOT refused — the budget itself becomes the stream's deadline, ending it with `DEADLINE_EXCEEDED` once elapsed (also uncounted, same reason). Unset means no cap. |
| `max_subscriptions` (default 256) | `RESOURCE_EXHAUSTED` | `subscriptions` | Concurrently open `TriggerService.Subscribe` streams; released when the stream ends or the client disconnects. `0` = unbounded. |
| `max_job_waits` (default 1024) | `RESOURCE_EXHAUSTED` | `job_waits` | Concurrently open `JobService.WaitJob` streams; same release rule. `0` = unbounded. |

### Lease and worker timing

One lease primitive (`[lease]`, `crates/jammi-db/src/catalog/lease.rs`) owns
every leased catalog row — a claimed job and a `building` result table
alike — defaulting to a 30 s lease renewed every 10 s; the job worker adds a
1 s idle-poll (`[worker]`):

- **Lease (30 s default)** — how long a claimed job, or a result table being
  written, is exclusively owned by its holder before it becomes reclaimable.
- **Heartbeat (10 s default)** — renews the lease well inside the window,
  from the process's shared lease-keeper thread: one registration per running
  job and one per `BuildingTable` between `create_table` and `finish`.
- **Idle-poll (1 s default)** — how often an idle worker checks for new work;
  reclaim runs on each idle tick, so a dead worker's job is recovered within
  roughly one poll plus one lease. A dead writer's `building` result table is
  reclaimed by the next session's startup recovery sweep once its lease has
  expired — never before, so a replica restarting beside a live writer leaves
  that writer's table alone.

The config layer enforces the invariant `heartbeat × 2 < lease` (and rejects a
zero lease, zero heartbeat, or zero idle-poll). This guarantees a live holder
renews at least twice per lease, so a single missed beat still leaves one
in-window renewal that lands strictly before expiry — never coincident with
it, which would race a reclaim. Bad values are rejected at config time, never
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

The work queue is the durable `jobs` table (migration 029) — one kind-agnostic
table for every training AND compute kind, distinguishing a `queued` row a
`[worker]` claim loop may pick up from an `inline` row a submitting call
claims once, by id, in its own task. The queued path is drained with a
`SELECT … FOR UPDATE SKIP LOCKED` claim (Postgres; SQLite's single serialised
writer) so concurrent workers each lock a distinct row. It is bounded by the
lease plus the attempts cap, not by an in-memory buffer — there is no
in-process queue-depth limit to overflow, and a worker crash leaves the row
claimable again after the lease expires.

### Close() and the successor handoff

Closing a session (`InferenceSession::close` / `PyDatabase.close`) shuts the
process's dedicated lease-keeper thread down and joins it — bounded by a 30s
window — before closing the shared catalog pool. This ordering matters
specifically for the SQLite backend: the keeper holds its own catalog
connection on its own OS thread, independent of the shared pool, so closing
only the shared pool while the keeper's thread stays up leaves the
`unix-excl` VFS's process-scoped exclusive lock held and a successor process
opening the same directory refused within its busy timeout, even after every
other handle has let go. A keeper that does not exit within the shutdown
window is logged and does not block the shared-pool close that follows —
`close()` must never hang, even at the cost of leaving that one connection's
fate unresolved in the (unexpected) case the keeper's thread is wedged.

## Failure-mode matrix

| Failure | Observed behavior | Recovery mechanism | Signal (metric/log) | Proving test |
|---------|-------------------|--------------------|---------------------|--------------|
| **Storage dies mid-publish** | No half-written committed artifact. The crashed worker's per-attempt prefix is orphaned because its finalize CAS never ran. | Winner-only commit: each attempt writes a unique `{job}/{worker}/{attempt}` prefix; the served `artifact_path` is written solely by the finalize CAS, so the committed pointer roots under the winner's prefix. | Final `jobs` row's `artifact_path` (nested in its `result` payload) resolves to the winner's prefix; reload returns the winner's bytes. | **Proven** by `tests/distributed/artifact_crash_window.rs`. |
| **Worker dies** | The claimed job is reclaimed by a different worker after the lease expires and completes exactly once. | Lease expiry + idle-tick reclaim; the `FOR UPDATE SKIP LOCKED` claim guarantees a single new owner. | The finalized row's `claimed_by` is a different instance id; reclaim runs each idle tick (worker log); the process's own `workers` row is deleted once its claim loop stops. | **Proven** by `tests/distributed/kill9_reclaim.rs` (plus `exactly_one_claim.rs` for the N-worker claim race and `cross_tenant_isolation.rs` for tenant scope). |
| **Compute tier down: jobs stay queued** | A deployment where the request-facing node runs `[worker] enabled = false` and only a separate compute-tier node runs `[worker] enabled = true` submits jobs normally even while every compute node is down: `SubmitJob` only inserts the `queued` row, it never requires a live claimant. `JobStatus`/`WaitJob` report the row `queued` indefinitely — never a fabricated failure — until a compute node's claim loop comes back and claims it. | No special recovery needed: `reclaim_expired_jobs`'s arms match only `running`-status rows past their lease — a `queued` row that has never been claimed has no lease to expire, so it is simply the row a `[worker]` claim loop resumes from once any process with a matching `kinds` entry is running again. | `ListWorkers` shows zero rows for the down kind; the job's row stays `queued` (never `running`, never `failed`) for the outage's whole duration. | The reclaim arms' `running`-only scope is asserted by `crates/jammi-db/tests/it/jobs_queue.rs`'s `reclaim_leaves_live_leases_untouched` and the claim-ordering tests (a job the claim loop has not reached yet is untouched by reclaim); `[worker] enabled` gating is asserted by `crates/jammi-db/src/config/tests.rs`'s `worker_config_toml_enabled_false_parses_to_false`. No single test submits a job with zero live workers and asserts it stays `queued` indefinitely end to end — an **honest gap**, tracked as a follow-up. |
| **GPU dies** | — | Memory-budget admission releases the permit via RAII on the failing path, but in-flight GPU-fault recovery is not yet validated end-to-end. | — | **Honest gap: not yet proven (1.0-deferred).** The distributed lane is CPU-only, so no chaos test exercises a GPU fault. |
| **Broker dies** | The trigger stream is a **separate subsystem** from the training worker fleet — claim and lease are pure Postgres, with no broker coupling — so a broker outage does not stall training. A broker fan-out failure is best-effort: the publisher has already committed the augmented event (with its engine `_offset`) to the durable backing table before fanning out, so the event is never lost — subscribers replay it from the backing table on reconnect. The Postgres broker driver carries no data of its own (it is a `LISTEN`/`NOTIFY` wake-up transport over the same backing table every driver replays), so its own connection dying is doubly harmless: a lost `NOTIFY` (a listener disconnect, a dropped notification) is bounded by the `idle_poll_secs` tick, which wakes every topic and triggers a replay regardless — nothing is lost, only delayed up to one poll interval. | **At-least-once + replay-completeness**: the backing table is the authoritative log; a subscriber attaches at an engine `_offset`, replays `[from..last_replayed]` from the table, then joins the live broker tail *with overlap* and dedups by engine `_offset` so no committed offset is ever skipped across the replay/live seam. The seam is keyed on the engine `_offset` alone — never on a broker-native sequence (JetStream's stream sequence is an independent counter that skews permanently after any post-commit fan-out failure). | Replayed offsets are contiguous from `from_offset`; the live tail resumes with no gap (broker integration test + in-mem property test); for the Postgres driver, every offset still arrives via replay within `idle_poll_secs` of the listener connection being killed or a `NOTIFY` being silently dropped. | **At-least-once + replay-completeness PROVEN.** In-memory + crash-mid-publish: `jammi-db/tests/it/trigger.rs` (`crash_mid_publish_replays_committed_offsets_with_no_loss`, `live_tail_resumes_with_no_loss_after_post_commit_fan_out_failure`, `at_least_once_no_skip_property_over_randomized_states`). Live JetStream consumer-recreate resume: `jammi-db/tests/it/trigger_jetstream.rs` (`consumer_recreate_resumes_engine_offsets_with_no_loss`, gated `live-broker-tests`). Postgres listener-kill and lost-NOTIFY recovery: `jammi-db/tests/it/broker_parity.rs` (`postgres_listener_killed_recovers_via_replay`, `postgres_suppressed_notify_recovers_via_idle_tick`, runtime-skip on `JAMMI_TEST_PG_URL`, run in the `test-pg` CI job). **Exactly-once is NOT provided by any backend** — dedup downstream by the `(_offset, _row_idx)` composite key. At-least-once is bounded by the backing log's durability (see below): process-crash-durable always; full on Postgres; on SQLite a host power-loss can lose the last committed backing-table row(s) since the previous checkpoint. |
| **Crash mid-publish of a result table** | No half-written result table is ever queryable. On restart every table left `building` by a **dead** writer — its lease absent or expired — is reconciled to exactly one terminal state — `ready` if its Parquet is a fully-valid closed file whose manifest sidecar landed (promoted with the *true* footer row count, and the ANN sidecar rebuilt from the Parquet so an embedding table self-heals), `failed` otherwise (missing or torn bytes, or a valid Parquet with no manifest; the objects are reaped only after the row's `building → failed` compare-and-set). A `building` row under a **live** lease belongs to a writer that is still producing it and is left alone. | Crash-consistent eventual reconciliation over lease-owned rows: a writer's `BuildingTable` stamps its `writer_id` and heartbeats a lease; every transition on the row is a compare-and-set naming the owner; the startup sweep enumerates only expired-lease rows, claims a promotable row (becoming its writer) before it rebuilds, and fails a reapable row before it deletes. The sweep runs cross-tenant (the one implicit-admin pass) — it deletes expired-lease bytes across every tenant, even from a tenant-bound session — and each row keeps its own `tenant_id`. | `Recovery: …` `WARN` logs name each reconciled table and its disposition; no expired-lease row remains `building`. | **Proven** by `jammi-db/tests/it/recovery.rs` — each torn state (missing bytes, truncated Parquet, valid-but-unfinalized Parquet, finalize-ordering window, ready-but-missing-bytes, two tenants' orphans) is constructed directly as a dead writer's, then the real `recover()` + `load_existing_tables` asserts invariants I1–I6; the two-writer oracles (`live_writer_survives_peer_recover_w1`/`_w2`, `live_writer_survives_peer_reconcile_apply_u2`, feature `test-hooks`) park a live writer while a peer session sweeps or reconciles beside it and prove its row, bytes, and segments survive and the table completes with the true count. |
| **Replica restarts during another replica's materialization** | The restarting replica's recovery sweep skips the live writer's `building` row (its lease is live); the writer finishes normally. If the writer instead stalls past its lease, the sweep claims the row (the writer's next heartbeat reports `LeaseLost`, it stops and deletes nothing) and either promotes it from the writer's own sidecar or reaps it. Lease time is the catalog database's clock (`now()` on Postgres); replica clock skew does not matter — see "Multi-writer safety" above. | Lease ownership + compare-and-set on `(table_name, writer_id, status = 'building')`; the deletion arms are a writer's own `abort()` after its one-row CAS, the reaper's claim/fail CAS (from both `recover()`'s startup sweep and `reconcile(apply=true)`'s expired-lease pass), and `delete_result_tables_for_source`/`remove_source`'s atomic delete-with-live-guard. | A writer that lost its lease surfaces `LeaseLost` / `CasFailed` typed errors; `Recovery: … skipped` `WARN` logs name a row a sweep declined. | **Proven** by `recovery.rs::live_writer_survives_peer_recover_*` / `live_writer_survives_peer_reconcile_apply_u2`, the zero-row outcome tests (`RowGone` / `TenantMismatch` / `CasFailed` / `LeaseLost`, none deletes), `expired_lease_building_row_is_claimed_before_reconcile_reaps_it_u2b`, and `two_recoverers_race_on_one_expired_row` (exactly one recoverer promotes). |
| **Client floods the port** (oversize messages, unbounded concurrency, or a long-held stream budget) | Every excess request is refused at the edge — before a tenant-scoped catalog read runs — rather than exhausting memory, file descriptors, or the catalog connection pool. A well-behaved caller elsewhere is unaffected: the per-connection bound isolates one noisy peer, and the two stream budgets (`Subscribe`/`WaitJob`) are independent of each other and of the unary bounds. | `[server.limits]`'s refusal-at-the-edge design (see [Request bounds](#request-bounds-serverlimits)) — no queueing, no silent degrade; a caller that needs more headroom raises the config knob. | `jammi_grpc_refused_total{reason}` — non-zero for a specific `reason` names exactly which knob a caller is hitting. | **Proven** by `crates/jammi-server/src/limits.rs`'s own `tests` module (the concurrency/timeout/budget refusal paths, deterministically) and `crates/jammi-server/tests/it/grpc_limits.rs` (the oversize-message, stream-budget-drop-releases-the-permit, and wait-timeout-at-the-edge cases, live over the wire — including the Flight SQL parity case). |
| **Catalog restored to an earlier point (storage unchanged)** | Startup recovery only reconciles `building` rows — a `ready` row whose objects the storage side has since moved past (or a `ready` row the restored catalog no longer has any record of writing) is invisible to it. | `jammi reconcile` catches what startup recovery does not: a `ready` row whose required objects are missing is flipped to `failed` (by CAS, after a live `exists()` per object); any storage object the restored catalog no longer references is reported as an orphan (or `unattributed` if its key predates the tenant-layout convention). See [Backup and Restore](./backup-and-restore.md) for the restore-ordering rule this failure mode motivates. | `reconcile`'s report (`rows_failed`, `orphans`, `unattributed`, `bytes_reclaimed`). | **Proven** by `reconcile.rs::missing_object_fails_the_row_and_is_reaped_past_grace` and `apply_false_never_mutates` (a dry run reports without touching anything, so re-running `reconcile` against a mismatched restore is always safe to inspect first). |
| **Storage restored to an earlier point (catalog unchanged)** | Symmetric to the row above: a `ready` row's catalog entry survives but its Parquet or a sidecar the row requires is gone from the rolled-back storage snapshot. | Same `reconcile` pass, same row→object completeness check — the missing-object arm does not care which side of the pair moved. | Same as above. | **Proven** by the same `reconcile.rs` suite; `unattributed_key_never_deleted_regardless_of_grace` additionally guards against a rolled-back storage snapshot's pre-layout keys being mistaken for reclaimable orphans (see below). |

### `reconcile` boot cost, pre-layout keys, and `delete_model`

- **`reconcile_ready_manifests`'s boot cost is one `exists()` per post-contract `ready` row** (`definition_hash IS NOT NULL`) — it checks only that row's `.materialization.json` sidecar is still present, never a full object-store `LIST`. A pre-contract row (`definition_hash IS NULL`) is skipped entirely; it legitimately has no sidecar to check.
- **A key written before the tenant-prefixed layout landed is permanently `unattributed`.** `reconcile`'s row→object and object→row checks both key off `TenantSegment::parse`, which only recognizes `{seg}/{table}.parquet` and `models/{seg}/{job}/…` shapes; an older flat key never round-trips through that parser, so it is reported as `unattributed` and **never deleted at any `grace`** — the allowlist is a safety property, not a migration path. Such an object stays reachable only by direct URL, exactly as it was before the tenant-prefixed layout existed. `unattributed` is reported only by the admin cross-tenant pass (`reconcile --all`); a tenant-scoped pass never lists another tenant's — or nobody's — stray keys.
- **`reconcile`'s `grace` gate compares two DIFFERENT clocks.** The age check
  (is an orphan candidate old enough to reclaim?) reads `last_modified` off
  the OBJECT STORE's own clock and compares it against the replica running
  `reconcile`'s `Utc::now()` — never the catalog database's clock the way a
  lease predicate does. `apply=true` requiring `grace >= ` the configured
  lease duration is exact only when the object store's clock and the
  replica's clock agree; ordinary NTP-level skew erodes the safety margin,
  it does not remove the mechanism — set `grace` to several multiples of the
  lease duration as the practical guard against both a slow writer and clock
  skew together.
- **`delete_model` deletes the catalog row only** — it never touches the artifact bytes at `models/{seg}/{job_id}/…`, and it refuses (`JammiError::ModelReferenced`) while another row still references it. A specific `{model_id}:epoch_N` checkpoint row is independent of the model's own row: deleting one leaves the other's row and bytes untouched. An artifact prefix no `models` row names any longer becomes a `reconcile` orphan candidate on the next pass, aged against `grace` like any other.

### Catalog durability under crash vs. power loss

Result-table crash-consistency reconciles whatever the catalog *durably retained* against the bytes on disk, so the catalog's own durability setting bounds the guarantee:

- **Process crash** (the engine dies, the host survives): both backends replay their write-ahead log on restart, so a `building → ready` (or the `building` insert recovery later reconciles) that committed before the crash is present after it. No committed catalog state is lost.
- **Host power loss**: Postgres commits synchronously (`fsync` per commit by default), so a committed transaction survives. SQLite runs `synchronous=NORMAL` under WAL — it fsyncs at checkpoint, not on every commit — so a power loss can lose the last committed transaction(s) since the previous checkpoint. A row that was *not* durably retained simply isn't seen by recovery; the bytes it would have pointed at are reaped as an orphan on a later sweep. This is a property of the catalog backend's durability configuration, not of the reconciliation.

The trigger stream's at-least-once guarantee inherits exactly this bound, because its durable log is the same kind of backing table written through the same backend transaction. Under a **process crash** the at-least-once guarantee is unconditional — the committed backing-table rows replay on reconnect. Under a **host power loss** the durable log itself is power-loss-bounded: on Postgres a committed publish survives; on SQLite (`synchronous=NORMAL` under WAL) the last committed backing-table row(s) since the previous checkpoint can be lost, and an offset whose row was not durably retained will not replay. At-least-once is therefore *full* on Postgres and *power-loss-bounded* on SQLite — never weaker than the backing log's own durability.
