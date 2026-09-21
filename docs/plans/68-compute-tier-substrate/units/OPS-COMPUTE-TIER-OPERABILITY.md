# Compute-tier operability — two-mode shutdown, lease release, gauges, warm-before-ready

How a worker-bearing `jammi-server` (or an embedded library worker) stops without losing in-flight
work, hands a running job to a successor at no attempt cost, reports queue depth and liveness, and
refuses to claim or report ready before its models are loaded.

The design has five parts:

- **Two shutdown modes.** SIGTERM is DRAIN (finish the in-flight job, then exit); SIGINT, or any
  signal while draining, is RELEASE (hand every lease back, exit at once). `jammi-server release` is
  the actuator that sends SIGINT.
- **Lease release in the catalog.** A released job stays `running` with a NULL lease and
  `releases + 1`; the retry cap counts `attempts − releases`. A compute job's building-table lease is
  released through its `jobs.partial_result` linkage. Every renewal statement carries a
  lease-present guard, so a released lease is never re-armed.
- **Warm-before-ready.** `[server] preload_models` is loaded before `/readyz` reports ready and
  before the claim loop makes its first claim.
- **Gauges.** Queue depth per kind, in-flight, claim-loop-up and heartbeat age, sampled off the
  scrape path.
- **Liveness.** `/healthz` is 503 when the lease keeper thread is dead or the claim loop panicked.

Decision labels (D1…D23) are stable anchors that sibling documents cite.

---

## 1. Decisions

| # | Decision | Why |
|---|---|---|
| D1 | **Two shutdown modes with PostgreSQL's mapping (R1): SIGTERM = DRAIN (smart), SIGINT = RELEASE (fast); any signal received while draining = RELEASE.** There is no engine-side drain timeout. | The engine ships the actuator, never the control loop that pulls it; the runtime owns the bound (`terminationGracePeriodSeconds`, `stop_grace_period`). The engine cannot abort a `spawn_blocking` training thread — aborting the loop task only cancels it at its next `.await` — and a tokio runtime drop waits for blocking work (R3), so an in-engine timeout could only duplicate SIGKILL. Orchestrators send one stop signal and then SIGKILL (R4, R5): a second SIGTERM never arrives from Kubernetes, so RELEASE must be reachable by a distinct signal and by an actuator (D16). |
| D2 | **DRAIN keeps the lease keeper alive, never flips `cancel`, finishes the in-flight job, then closes the session.** | A rollout must not lose in-flight training. Closing the session first would stop the keeper, whose exit guard flips every hold's `lost` flag (`crates/jammi-db/src/catalog/lease_keeper.rs::ExitGuard`); the trainer reads that flag as `cancel`, skips its epoch bundle and bails. So the worker is joined before `session.close()`. |
| D3 | **The worker guard is owned by `BoundServer`.** `OssServer::bind` calls `BoundChain::take_worker()`; chain-level serve paths keep Drop-abort semantics; a downstream embedding the chain takes the guard through `ChainParts::worker` or `take_worker`. | Ownership decides who can drain: the guard must outlive the gRPC serve future for the two-mode shutdown to join or release it. `EmbeddedWorker`'s `Drop` aborts only a handle nobody has joined, so a graceful path followed by `Drop` is a no-op. |
| D4 | **Loop termination is observed through a `watch<LoopState>` written by a Drop guard inside the task** (`LoopState { Running, Stopped, Aborted, Failed }`, `LoopExitGuard`). `stop_and_join` and `release_and_stop` await the watch; the `JoinHandle` is awaited only after the watch reports terminal. | A writer placed after the loop's `.await` never runs on abort or panic; a Drop guard runs on every exit path. |
| D5 | **The worker's stop is a level-triggered `watch<bool>` in `WorkerShared`**, not an `AtomicBool` and not a `Notify`. The idle sleep is `select!{ sleep(idle_poll), stop_rx.wait_for(|v| *v) }`. Every watch consumer uses `wait_for(|v| *v)`, never `changed()`. | Level-triggered wakeups cannot be lost, and a receiver subscribed after the send still resolves. A cooperative stop of an idle loop is immediate instead of costing up to `idle_poll_secs`. |
| D6 | **RELEASE is one mechanism, `EmbeddedWorker::release_and_stop` (§2.4): phase flip and stop together → keeper releases registered job holds → sweep #1 → cooperative-or-abort keyed on the slot holder → observe the watch → sweep #2 unconditionally → `delete_worker`. Never abort the loop task while a claim transaction may be in flight.** | `Catalog::claim_next` is a client-driven BEGIN/UPDATE/COMMIT. A COMMIT already flushed when tokio drops the loop future is still processed by the backend afterwards, and a sweep evaluates the last-committed row version — so "the task is aborted, hence no claim can land" is false. Separately, the zombie gate must flip before the hold disappears: `LeaseHold`'s `Drop` removes the hold without setting `lost`, so aborting first would let a detached trainer write a doomed epoch's bundle under the job's `{job_id}/_checkpoints/` prefixes. Sweep #2 runs before `delete_worker` so the `workers` row outlives this instance's last lease write. |
| D7 | **"A loop-claimed job is running under a live hold" is a slot state, `Holder::JobRun`, not a counter.** The holder moves `ClaimProbe → JobRun` inside `register_job_hold_or_release` once the hold is registered, and back to `Free` when the iteration's `ClaimGuard` drops. Invariant: `holder == JobRun ⇒ the loop is not inside claim_next`. Inline `run_now` never touches the cell. | The claim→hold prologue can hold an uncommitted or just-committed claim, so it must take RELEASE's cooperative arm; a job under a hold cannot be inside `claim_next`, so it may be aborted. The decision reads the holder kind (§2.5). |
| D8 | **One helper registers the hold at both loop hold sites** (fine-tune and compute): `register_job_hold_or_release(session, catalog, shared, job_id, attempts, holder) -> Option<LeaseHold>`. | One mechanism, two callers. A hold registered after the keeper's release pass snapshotted must self-release, not train. |
| D9 | **RELEASE is lease-level.** `Catalog::release_job_lease` is the heartbeat CAS plus `lease_expires_at = NULL, releases = releases + 1`; no job status is invented. Both `heartbeat_job` and the release statements carry `AND lease_expires_at IS NOT NULL`. The building-table class has the symmetric guard: `ResultTableCas::lease_present`, which `Catalog::renew_lease` always sets on a copy of its argument, so all three renewal callers carry it (§2.6). | Mutable state is CRUD through DML, with no invented transition. Every `running` row has a non-NULL lease by construction (`claim_next` and `claim_by_id` write the deadline), so the predicate is safe; without it any holder could re-arm a NULLed row. A per-hold in-memory flag cannot close the hazard alone, because a fresh hold registered after a sweep has no flag — the guarantee sits at the SQL edge. |
| D10 | **A release is not a failure: `jobs.releases` offsets `MAX_ATTEMPTS`.** The reclaim cap compares `attempts − releases`; `attempts` still bumps on every claim; a release is counted at most once per attempt, because the `IS NOT NULL` predicate makes every release statement idempotent. A rolling restart therefore costs zero net attempts, for every kind (§2.7). | A deploy storm must not burn the cap; three real crashes still fail the job. |
| D11 | **Inline rows are never released — for both lease classes.** Every jobs release statement carries `execution = 'queued'`; the keeper's `release_job_holds` releases `LeaseTarget::Job` holds only and leaves one alone on `Ok(false)`; the building-lease sweep is scoped through the jobs linkage, never through `writer_id` alone; the keeper never releases a `ResultTable` hold. An inline row is failed by the reclaim's inline-liveness arm when its instance goes stale. | No status is invented for shutdown, and inline rows have no requeue arm. The store's `writer_id` is minted once per store and shared by every materialization on the session, and `result_tables` has no `execution` column, so a writer-scoped release would NULL an inline or library materialization's lease. The only column that can tell them apart is `jobs.execution`, reached through `jobs.partial_result`. |
| D12 | **The server's DRAIN**: `/readyz` 503 "draining", then concurrently `{tonic graceful shutdown, with idle WaitJob/Subscribe streams ended by a typed UNAVAILABLE "server draining" trailer}` and `{gated worker join}`, the release signal racing the whole sequence; `session.close()` after. | `[server.limits]` timeouts default to unbounded, so an idle `Subscribe` would hold a serial drain open forever. The join half is gated on the drain signal because `stop_and_join` requests stop at its first poll — an ungated join would stop the worker at t = 0. |
| D13 | **DRAIN refuses no RPC submissions; the HTTP side-channel (and the peer listener) stay up through a drain.** | The query tier may be the same process. tonic's graceful shutdown already stops accepting connections and finishes in-flight requests; `JobService` mounts regardless of `[worker] enabled`. |
| D14 | **`workers.state TEXT NOT NULL DEFAULT 'claiming'`, CHECK-constrained to `warming | claiming | draining`.** The loop task's first statement upserts `warming`; the same task writes `claiming` when the worker gate opens (one sequential chain, §2.1); `begin_drain` writes `draining`. `WorkerSummary.state` carries it on the wire. | The row is the only truth another process can read. Writing it from the loop task itself, never from a detached spawn-time task, means `claiming` can never precede `warming` and the post-exit delete is ordered after the upsert. A rank-running peer is `claiming` with its slot busy; no further state is needed. |
| D15 | **Warm-before-ready**: `PreloadEntry` with a hand-written `Deserialize` (a bare id string, or a table `{ id, task }`); bare ids resolve their task from the `models` row at the server's startup edge; a preload failure is a hard startup error before serving; the signal watcher is running before preload starts; the worker is gated on warm. | Invalid input is refused with a typed error at the edge it enters. The model cache key is task-free while the head is chosen from the resolved task, so a guessed task would poison the cache — hence the explicit task or the catalog's. A worker spawned at bind would otherwise claim while `/readyz` says "preloading". |
| D16 | **`jammi-server release [--pid N]`** sends `SIGINT` with `libc::kill` (default pid 1). | A generic actuator that sends a signal and knows nothing about jobs, mirroring `probe`. The CUDA image has a shell, so `preStop: ["/bin/sh","-c","kill -INT 1"]` works there; the distroless CPU images have none, so the subcommand is the uniform actuator. `lifecycle.stopSignal` is behind an alpha feature gate (R4b). |
| D17 | **Gauges are sampled by a dedicated task every `[worker] metrics_sample_secs` (default 5, ≥ 1)** — never on scrape, never on the claim loop. Index `idx_jobs_kind_status(status, execution, kind)`. The worker families are absent on non-worker processes. No tenant label. | A scrape storm must not become a catalog storm, and the claim loop does not tick during a run. `idx_jobs_claim` lacks `kind`, so `GROUP BY kind` would be a heap aggregate. Tenant scope is a listing predicate, not a metrics dimension; kinds are bounded by `COMPILED_KINDS`. Naming follows R2. |
| D18 | **`/healthz` is 503 on keeper death or loop `Failed`, on every process; 200 while draining or stopped.** No slow-step detection. | The runtime owns "how long is too long". The keeper exists in every session, so the check applies to non-worker processes too. |
| D19 | **Remote equals embedded at the release write**: the server, the Rust library and Python issue the same statements through the same `release_and_stop`. The one named divergence is library-only: a process that survives a release can still land the finalize CAS. | The finalize CAS is guarded by `claimed_by / status / attempts`, never by the lease. After `process::exit` nothing finalizes on the server; in the library a surviving training thread may reach finalize before a successor claims and land `completed`. It is attempt-guarded, so a successor's claim bump makes it match zero rows. Cross-surface row equality is deliberately not claimed — outcome (iii) of §2.7 licenses the rows to differ. |
| D20 | **Compute overlay `terminationGracePeriodSeconds: 600`** (the cluster-autoscaler's `--max-graceful-termination-sec` default, R11) with the operative rule and rollout arithmetic in the README; compose `stop_grace_period: 120s`, CI override `30s`; spot handling is README prose. The Shape C base keeps the Kubernetes default. | The grace knob ships with the drain mechanism that makes it meaningful. A documented number, not a magic default (§2.12). |
| D21 | **One appended migration, `031_jobs_releases_workers_state`**; tests assert its presence and its order after `030_jobs_idempotency_key`, never that it is last. | Append-only migrations; later migrations follow it. |
| D22 | **A confirmed RELEASE exits 0; SIGINT is RELEASE on a laptop too** (Ctrl+C releases the running job and exits promptly). `main` calls `std::process::exit` after a release. | The library never exits a process; the binary must, because a normal return would wait on the runtime drop for the detached training thread (R3). Safe because `ArtifactStore::stage_checkpoint` writes each epoch under its own prefix, manifest last, and the trainer gates the write on `!cancel`, so an exit mid-put leaves the previous epoch's bundle the newest complete one. |
| D23 | **Serve entries**: `BoundChain::serve_with_shutdown(future)` = drain flip only, worker Drop-abort retained; `BoundServer::serve_with_shutdown(future)` = drain flip + gated join; `BoundServer::serve_with_signals(drain_rx, release_rx) -> Result<ShutdownOutcome, ServerError>` is the two-signal form; `BoundServer::serve()` feeds it from the signal watcher. | One entry carries the two-mode shutdown; chain-level embedders keep their single-future teardown. The worker gate is session state, so `GrpcChain` and `assemble_grpc_chain` keep their shape. |

---

## 2. Design

### 2.1 State and readers

| Value | Lives in | Written by | Read by |
|---|---|---|---|
| `WorkerPhase { Running, Draining, Releasing }` — one `watch` cell | `HostAdmission`, owned by the session | `HostAdmission::begin_drain` (`Running → Draining`, never regresses `Releasing`); `HostAdmission::begin_release` (→ `Releasing`, then bumps `release_epoch`); `try_claim_loop` resets it to `Running` for a new loop generation | the loop's claim gate; `probe_claim`; every held gang rank (a phase leaving `Running` ends the rank with the `Drain` reason) |
| `Holder { Free, ClaimProbe, JobRun, Awaiting{..}, Rank{..} }` — one `watch` cell, every transition a `send_if_modified` compare-and-set | `HostAdmission` | the claim loop (`probe_claim`, `job_running`, `ClaimGuard` drop); gang admission (`try_hold_rank`) | `release_and_stop` step 2e; `jammi_worker_jobs_in_flight`; rank admission |
| `release_epoch: AtomicU64` | `HostAdmission` | `begin_release`, on every call | `WorkerShared::released_since_birth` (compared with the loop's birth snapshot) |
| `WorkerShared { admission, stop: watch<bool>, state_tx: watch<LoopState>, sample, samples_taken, instance_id, spawn_release_epoch }` | one `Arc`, held by the loop task and the guard; `EmbeddedWorker::shared()` hands out a `Weak` | stop: `begin_drain`, `stop_and_join`, `release_and_stop`, `Drop`; state: `LoopExitGuard`; sample: the sampler task | loop gate and idle sleep; `/healthz`; `/metrics` |
| `LoopState { Running, Stopped, Aborted, Failed }` | `watch` in `WorkerShared` | `LoopExitGuard` — `Stopped` on return, `Failed` when `std::thread::panicking()`, `Aborted` otherwise | `release_and_stop`, `stop_and_join`, `/healthz`, `jammi_worker_claim_loop_up` |
| `HeldState.released` | the keeper's hold map | `release_job_holds`, on `Ok(true)`, for `LeaseTarget::Job` holds only | `renew_all` skips a released hold — an optimisation; the SQL predicate is the guarantee |
| `jobs.releases`, `jobs.lease_expires_at = NULL` | catalog | `release_job_lease`, `release_jobs_claimed_by` | the reclaim's queued arm (`attempts − releases`), every `lease_expired_clause` reader |
| `result_tables.lease_expires_at = NULL` | catalog | `release_building_tables_of_claimant` only — there is no per-target release and the keeper never writes it | `claim_expired_building_table` (`IS NULL OR < now`, so claimable at once); `renew_lease` through its `lease_present` arm (0 rows → `CasFailed` → `lost`) |
| `workers.state` | catalog | the loop task (`warming`, then `claiming`), `EmbeddedWorker::begin_drain` (`draining`); row deleted by `stop_and_join` / `release_and_stop` | `ListWorkers`, `jammi workers`, admin |
| `ReadinessProbe { warm, draining, preloaded, preload_total }` | `OssServer` | the preload step; the drain and release arms | `/readyz` |
| worker gate: `watch<bool>`, initialised open | `InferenceSession` | `OssServer::new` closes it before `bind` when `preload_models` is non-empty; `serve_with_signals` opens it after preload. Library callers never touch it | the loop task, before its first claim |
| `MetricsRegistry` worker families | registered by `attach_worker(Weak<WorkerShared>)`; `attach_keeper` on every process | the `/metrics` handler copies the snapshot | Prometheus |

The `workers` row is written by one sequential chain on the loop task
(`crates/jammi-ai/src/fine_tune/worker.rs::JobWorker::run_until`): upsert `warming` →
`select!{ gate_rx.wait_for(|open| *open), stop_rx.wait_for(|stop| *stop) }` → upsert `claiming` on the
gate arm, or return without claiming on the stop arm → the claim loop. Row and registration cell are
written as one fact (`write_worker_facts`): the cell is set first, and reverted if the upsert fails,
so a keeper re-registration racing the write never publishes facts no row write succeeded with. No
detached spawn-time upsert exists, so the post-exit `delete_worker` is ordered after the upsert by
construction. Two orders cannot be pinned: a timeout abort landing while the upsert itself is still in
flight past one heartbeat (catalog unreachable), and `EmbeddedWorker`'s synchronous `Drop`, which
aborts the task and deletes the row from a detached task without observing the exit guard. Either can
leave a phantom `warming` row; it falls to the `instances` staleness cascade, the existing
catalog-unreachable class.

Ownership of the loop task's `JoinHandle` is total: `LoopTask { Running, Joined, Abandoned }` behind
a mutex, read only through `TakenHandle`. A stop attempt whose own future is cancelled mid-wait (a
RELEASE preempting a DRAIN's `stop_and_join`) restores the handle as `Abandoned` instead of detaching
the task; the next `release_and_stop`, or `Drop`, aborts it unconditionally.

### 2.2 Startup and warm-before-ready (`preload_models`)

`ServerConfig.preload_models: Vec<PreloadEntry>`, `PreloadEntry { id: String, task: Option<ModelTask> }`.
The hand-written `Deserialize` accepts a bare id string or a map with exactly `id` and optional
`task` (`ModelTask::try_from_db_str`), rejects an empty id and an unknown task token with a typed
error, and keeps the config layer's unknown-field refusal.

`OssServer::new` builds the session and closes the worker gate when the list is non-empty.
`OssServer::bind` binds the health listener, assembles and binds the gRPC chain (the worker is spawned
inside `assemble_grpc_chain` and subscribes to the session's gate), moves the guard into
`BoundServer` with `take_worker()`, and attaches the worker metrics. Then
`BoundServer::serve_with_signals(drain_rx, release_rx)`:

1. `BoundServer::serve()` has already spawned `signal_watcher`: one task owns both signal streams for
   the process's lifetime, because tokio coalesces signals only before a stream's first poll (R8).
   First SIGTERM → `drain_tx`; SIGINT at any time, or any later SIGTERM → `release_tx`. The two
   watches are the only coupling between the watcher and the serve function.
2. The health task starts on its own stop channel: `/healthz` 200, `/readyz` 503
   `"preloading 0/n"`.
3. Preload inline, per entry (`crates/jammi-server/src/runtime.rs::preload_models`):
   `ModelSource::parse(id)`; task = the entry's explicit task, else the `models` row's, else
   `ServerError::Preload { id, reason: "no models row; give { id, task }" }`;
   `ModelCache::preload(&source, task, None)`. The preload is raced against both signals.
   - A signal aborts the preload and the server never serves. DRAIN: `stop_and_join` (the gate is
     closed and its wait is selected against stop, so the loop returns without claiming). RELEASE:
     `release_and_stop`, its outcome built from its own evidence exactly as in §2.4.
   - A preload `Err` is a startup error: the worker is stopped and joined, the health task stopped,
     the session closed, `Err(Preload)` returned; `main` exits non-zero. The listener never served
     and the worker never claimed.
   - On both exits the `workers` row is deleted and awaited before `session.close()` — never left
     to `Drop`'s detached delete, which `process::exit` and the runtime drop would race.
   - An empty list is warm immediately.
4. `readiness.set_warm()`, `session.open_worker_gate()` — the loop passes its wait, writes
   `claiming`, and claims.
5. gRPC serves through `BoundChain::serve_with_drain(drain_rx)`.

Liveness is 200 throughout, so no `startupProbe` is needed. A `test-hooks` park point,
`preload_test_hooks::ParkPoint::BeforeLoad` inside `ModelCache::preload`, is the rendezvous the
preload tests use.

### 2.3 DRAIN (first SIGTERM)

1. The gated join half wakes on `drain_rx`: `readiness.begin_drain()` (`/readyz` 503 `"draining"`),
   `HostAdmission::begin_drain()` (held gang ranks end `Drain`), a Ballista executor role stops
   admitting tasks, then `EmbeddedWorker::begin_drain()` = phase `Draining` + stop requested +
   best-effort `workers.state = 'draining'`. The health side-channel and the peer listener are not
   signalled.
2. The same `drain_rx` feeds tonic's `serve_with_incoming_shutdown` — listener closed, in-flight
   requests finish — and the stream ender: `MethodClassLayer` wraps the two streaming paths
   (`WaitJob`, `Subscribe`) in `PermitBody::Draining`, which on drain synthesises an `UNAVAILABLE`
   "server draining" trailer through the same path the deadline body uses and counts
   `jammi_grpc_refused_total{reason="draining"}` (`RefusedBound::Draining`). A stream can carry both
   a deadline and a drain ender. Unaries are untouched: an in-flight unary, including an inline
   `run_now`, is bounded only by the runtime's grace.
3. `drain_sequence = join(grpc_serve, gated_join)` — concurrent. `stop_and_join` requests stop,
   awaits `state_rx.wait_for(|s| *s != Running)` — the in-flight job's own terminal write lands
   first: keeper alive, heartbeats continue, every epoch bundle lands — then joins the task and
   deletes the `workers` row. With no worker the join half reports `worker_joined: false`.
4. `select!{ drain_sequence → Drained { worker_joined }, release_rx → §2.4 }` — a release races the
   whole sequence.
5. Executor drain, scheduler stop, health and peer tasks stopped and awaited, `session.close()`,
   `telemetry::flush_otlp()`, `Ok(ShutdownOutcome::Drained { .. })` → `main` returns success (no
   blocking work remains).

### 2.4 RELEASE (SIGINT at any time, or any signal while draining)

Server arm, in order:

1. The `select!` drops the drain sequence and with it the gRPC serve future — connections severed
   (PostgreSQL "fast"). An inline `run_now` future dies here; its row is left to the reclaim's
   inline-liveness arm (D11).
2. `EmbeddedWorker::release_and_stop()` — the one release mechanism, identical on the library:
   - **2a** `HostAdmission::begin_release()` and the stop request, as one statement pair with no
     genuine yield between them. `begin_release` flips the phase to `Releasing` and then bumps
     `release_epoch`; the flip-before-bump order is load-bearing (§2.5). The stop belongs here, not in
     a later step: deferring it past 2b/2c opens a window in which the loop reclaims and re-claims
     the same row under `Releasing` without ever tripping the cap (`attempts − releases` nets to 0
     on every self-release), spinning for up to one keeper pass plus one sweep.
   - **2b** `LeaseKeeper::release_job_holds(bound = heartbeat)`: on the keeper thread, serialised
     with `renew_all`, snapshot the registered holds and release every not-yet-released
     `LeaseTarget::Job` — only that class — with `Catalog::release_job_lease`. `Ok(true)` →
     `released = true`, `lost = true` (the trainer bails at its next boundary without writing a
     bundle); `Ok(false)` (an inline row, or already gone) → hold untouched; `Err` → logged, hold
     untouched (expiry path). `ResultTable` holds are never touched here (D11). The pass returns
     `HoldRelease { attempted, released, not_required, failed }`, the three outcomes summing to
     `attempted`; a pass that could not be confirmed to run is `HoldReleaseOutcome::Unobserved`,
     never a count of zero. It runs before any abort, so every live job hold's `lost` flips while
     the hold still exists.
   - **2c** sweep #1 (`release_sweep`), two statements in this order:
     `Catalog::release_jobs_claimed_by(instance_id)` then
     `Catalog::release_building_tables_of_claimant(instance_id, writer_id)` (§2.6). It covers a
     claim, or a building row, that committed after 2b snapshotted. An `Err` from either statement
     is logged and recorded as `None` in `ReleaseSweep`; the arm continues.
   - **2e** a total match on the loop task's state. An `Abandoned` handle is aborted
     unconditionally. A `Running` handle whose holder is not `JobRun` — `Free` (idle or inside
     `reclaim_expired_jobs`), `ClaimProbe` (inside `claim_next` or the claim→hold prologue), or a
     `Rank` (never loop work) — is **not aborted**: wait `state_rx.wait_for(|s| *s != Running)`
     bounded by one heartbeat (`WorkerIntervals.heartbeat`, fed by `[lease] heartbeat_secs`, default
     10 s) and join on the cooperative exit. On timeout, abort — the only path where a claim
     transaction can be in flight when the abort lands; the bound also spans
     `reclaim_expired_jobs`, so a large backlog can trip it: cost only, never corruption. Dropping
     the loop future inside `claim_next` before COMMIT always rolls back on both backends (the sqlx
     transaction rolls back on drop); a COMMIT already flushed, or an abort landing between COMMIT
     and hold registration, is outcome (iii) of §2.7. Holder `JobRun`: the loop is inside a job
     under a registered hold and not inside `claim_next` — abort now; the dropped future runs the
     hold's and the cancel watcher's `Drop`.
   - **2f** observe the terminal `LoopState`. `ReleaseReport::stop_witnessed` is true when 2e itself
     resolved the task (joined or aborted) or this observation is a genuine watch-fired terminal
     value — never the fallback read on a timeout, which alone can read `Running` on an abort whose
     guard has not published yet. `wait_for` treats an already-terminal value as witnessed, which
     is the usual case: 2a's gate stops an idle loop before 2b/2c even run.
   - **2g** sweep #2, unconditionally: the same two statements in the same order, idempotent by
     `lease_expires_at IS NOT NULL`. It catches a claim or a building row that committed after
     sweep #1, on every path.
   - **2h** `delete_worker(instance_id)`. 2g-before-2h is load-bearing: the `workers` row outlives
     this instance's last lease write, so a reader keyed on the row never sees a claimant vanish
     while its rows still hold live leases.
   - **2i** release the session's claim-loop slot (`HostAdmission::release_loop_claim`, a
     compare-and-set on this guard's own generation), so a successor loop can be spawned without
     waiting for the guard's drop.
3. With no worker: `InferenceSession::release_job_leases()` = `begin_release` + 2b + 2c on the
   session. Every statement matches nothing by construction — only inline holds and inline rows
   exist — but the surface stays uniform and held gang ranks still end.
4. Executor and scheduler roles stop; health and peer tasks are stopped and awaited;
   `session.close()`.
5. `telemetry::flush_otlp()`.
6. The outcome is built only from the release's own evidence
   (`crates/jammi-server/src/runtime.rs::release_outcome`): `ShutdownOutcome::Released` iff the call
   returned `Ok`, the hold pass `confirms_release()`, `stop_witnessed` is true (vacuous with no
   loop), and the **final** sweep's `jobs` and `building` are both `Some`. Sweep #1 is never a
   determinant — its `None` on a transient race is legitimately covered by the idempotent sweep
   #2. Anything else is `ShutdownOutcome::ReleaseDegraded`. `main` calls `process::exit(0)` on
   `Released` and `process::exit(3)` on `ReleaseDegraded`; a degraded release is a value, never a
   propagated `Err`, because a normal return would wait on the detached trainer past the grace
   period and turn a degraded release into a kill. What a degraded release establishes depends on
   which determinant is missing: a sweep field reading `None` means that table's lease was not
   written and falls to expiry (one attempt for a `jobs` row, one back-off for a building row);
   with both sweep fields `Some`, every row the sweep matched is released, and nothing is
   established about a row under an active hold (hold pass unobserved or failed) or about a claim
   that commits after the sweep (`stop_witnessed == false`).

The whole arm is bounded by 2 × heartbeat + the keeper's pass + up to 2 × SQLite `busy_timeout` (5 s
each) while a claim transaction is parked open: sweep #1 issues two write statements, each able to
block on the parked `BEGIN IMMEDIATE`. By sweep #2 the loop has exited or been aborted and its
transaction rolled back.

**Library.** `EmbeddedWorker::release_and_stop` and `InferenceSession::release_job_leases` issue the
identical statements. The in-flight training thread keeps running until its next epoch boundary,
then bails without a bundle; Python cannot exit its host process. If that thread reaches finalize
before a successor claims, the attempt-guarded CAS may still land `completed` (D19).

### 2.5 The claim loop and its single slot

`JobWorker::run_until(shared: Arc<WorkerShared>)` is a bounded `loop`, spawned only through
`EmbeddedWorker::spawn` / `spawn_worker`.

**One claim loop per session.** `HostAdmission::try_claim_loop` is a compare-and-set on
`loop_owner` that hands out a monotonic, never-reused generation id; a second spawn while a
generation is live is refused with a typed error before any task exists. The slot is freed by
`release_loop_claim(generation)` — a compare-and-set against the caller's own generation — from a
completed `release_and_stop` or from `Drop`, so a late release never steals a successor's slot. The
phase barrier alone is necessary but not sufficient: it cannot stop a second loop from existing in
the first place.

**The claim gate.** `WorkerShared::admits_claim` = `!stop_requested() && phase() == Running &&
!released_since_birth()`. The loop reads this one predicate at two sites: the top of the
iteration, and again immediately after `reclaim_expired_jobs` returns, with no `.await` between
that second read and the `claim_next` call. A RELEASE or DRAIN landing during the reclaim round trip
is caught by the second read even though the first, now-stale read admitted the iteration. The one
residual is a claim whose own catalog round trip is already in flight when the phase flips: under
RELEASE it self-releases at the hold site; under DRAIN it dispatches and runs to completion.

**The release epoch.** "Have I been released" is an epoch comparison, not a phase read.
`try_claim_loop` resets the phase to `Running` for each new generation, so a stale task whose
execution lags behind an `abort()` could otherwise read a later generation's fresh `Running` phase.
Each `WorkerShared` snapshots `release_epoch` at birth, and `begin_release` bumps it on every call.
`begin_release` flips the phase strictly before it bumps the epoch, so "the bump is visible ⇒ the
flip already happened"; a placed-attempt run relies on exactly that (it snapshots the epoch before
`probe_claim`, whose phase check then refuses directly). Pinned by
`begin_release_bumps_the_epoch_strictly_after_the_phase_flip_is_already_visible`.

**The slot holder.** `Free → ClaimProbe` (`HostAdmission::probe_claim`, immediately before
`claim_next`; refused when the phase is not `Running` or the slot is held, in which case the loop
sleeps an idle poll) `→ JobRun` (`job_running`, at the hold site, once the lease hold is registered
— never earlier, so the claim→hold prologue stays a `ClaimProbe`) `→ Free` (the iteration's
`ClaimGuard` drop, on every exit path including abort and panic). A peer never claims while it
holds a gang `Rank`, and never receives a rank while it runs a loop-claimed job; `Awaiting` is a
claimant waiting on a placed attempt's result, which admits a rank exactly as `Free` does. An inline
`run_now` and a direct `run_claimed_job` hold no probe and leave the cell alone.

**The hold site.** `register_job_hold_or_release` registers the hold with the keeper, then reads
`released_since_birth()`. If a release landed, the claim raced it (it committed after the keeper's
pass snapshotted, or after a sweep): the helper releases its own row with `release_job_lease`
(idempotent — 0 rows if a sweep already took it), drops the hold and returns `None`; the caller
returns without dispatching. Otherwise it moves the holder to `JobRun` and returns the hold.
Fine-tune keeps `cancel = hold.lost_flag()`; compute drops the hold after `execute_compute`. On the
compute path the check lands after `dispatch_partial_result`'s catalog reads, which is harmless.

**Idle sleep.** `select!{ sleep(idle_poll), stop_rx.wait_for(|stop| *stop) }` (D5).

**Rejected.** An `AtomicBool` stop with an uninterruptible sleep (a cooperative stop costs up to
`idle_poll_secs`). A count of in-flight jobs as the abort key (the holder kind carries the same fact
and also serves rank admission and the in-flight gauge; a count cannot distinguish a claim probe
from idle). A bare `phase == Releasing` test at the hold site (unsound across loop generations).
Making `claim_next` abort-safe (needs a server-side claim or a two-phase claim token — a different
design; outcome (iii) is the accepted residual).

Test hooks (`test-hooks` feature): `loop_test_hooks::ParkPoint::BeforeHold`, the notify-only
`Rendezvous::ReleaseAt2e` fired as 2e's first statement, `arm_after_reclaim`, a per-instance
`claim_next` call counter, and `crates/jammi-db/src/catalog/claim_test_hooks.rs::ParkPoint::ClaimBeforeCommit`.
They let the shutdown arms be pinned against the mechanism rather than raced against a wall clock.
On SQLite a `ClaimBeforeCommit` park holds the write lock, so each sweep statement logs
`database is locked` after its `busy_timeout`; the tests key the unpark on the 2e rendezvous, never
on those logs.

### 2.6 Lease release in the catalog

Migration `031_jobs_releases_workers_state`:

```sql
ALTER TABLE jobs ADD COLUMN releases INTEGER NOT NULL DEFAULT 0;
CREATE INDEX idx_jobs_kind_status ON jobs(status, execution, kind);
ALTER TABLE workers ADD COLUMN state TEXT NOT NULL DEFAULT 'claiming'
    CHECK (state IN ('warming', 'claiming', 'draining'));
```

Statements (both backends; `crates/jammi-db/src/catalog/jobs_repo.rs` and `result_repo.rs`):

- `Catalog::heartbeat_job`: the heartbeat CAS `… AND lease_expires_at IS NOT NULL`.
- `Catalog::release_job_lease(job_id, instance_id, attempts) -> bool`:
  `UPDATE jobs SET lease_expires_at = NULL, releases = releases + 1, updated_at = $now WHERE job_id = $1 AND status = 'running' AND execution = 'queued' AND claimed_by = $2 AND attempts = $3 AND lease_expires_at IS NOT NULL`.
- `Catalog::release_jobs_claimed_by(instance_id) -> usize`: the same `SET`,
  `WHERE claimed_by = $1 AND status = 'running' AND execution = 'queued' AND lease_expires_at IS NOT NULL`.
- `Catalog::reclaim_expired_jobs`, queued arm: requeue when `attempts - releases < $max`, fail when
  `>= $max`. The inline arm is unchanged: an inline row is failed when its instance is stale past
  2 × lease, never requeued.
- `Catalog::count_jobs_by_kind_status() -> Vec<(kind, status, n)>`:
  `SELECT kind, status, COUNT(*) FROM jobs WHERE execution = 'queued' AND status IN ('queued','running') GROUP BY kind, status`
  — index-only on `idx_jobs_kind_status`; held (`claimable = false`) rows count as queued.
- `Catalog::upsert_worker(.., state, ..)`, `set_worker_state`, `WorkerState { Warming, Claiming, Draining }`,
  `WorkerRecord.state`, `JobRecord.releases`.

**The `releases` counter (D10).** A released row stays `running` and `claimed_by` this instance,
with a NULL lease. `lease_expired_clause` treats `IS NULL` as expired, so the next reclaim requeues
it at once and a successor claims it within one `idle_poll_secs` instead of one
`[lease] duration_secs`. The claim bumps `attempts`; the release bumped `releases`; the cap compares
the difference. No cap on `releases` exists.

**The `lease_present` guard on renewals (D9).** `ResultTableCas` has a pub `lease_present: bool`
field, `false` in every builder (`writer`, `writer_any_tenant`, and `expired`, whose owner arm is
`IS NULL OR < now` and must never carry it), set by `with_lease_present()`, rendered as
`AND lease_expires_at IS NOT NULL`. `Catalog::renew_lease` clone-and-sets it on a copy of its
argument unconditionally, so all three callers carry it:

- the keeper's `LeaseTarget::ResultTable` renewal — a released lease renews 0 rows → `CasFailed` →
  the hold's `lost` flips. For a `ResultTable` hold this renewal is the only path to `lost`, within
  one heartbeat of the sweep;
- the writer's pre-promote renewal in `BuildingTable::finish` — an old writer errors instead of
  promoting;
- recovery's pre-rebuild renewal in `ResultStore::reconcile_expired_building_row` — the one caller
  where the arm gates a destructive purge: a miss is `is_cas_miss`, the row is detached and left
  `Untouched`, so a NULLed building lease is never purged by recovery either.

A miss on this arm re-reads as `building` with the owner arm holding, so `classify_cas_miss`
reports `CasFailed { status: "building" }` — a status-named error whose status did not change, the
fifth cause in that function's contract. Every caller that matches `CasFailed` already treats it as
a miss.

**Building-table lease release through the `jobs.partial_result` linkage (D11).**

```sql
UPDATE result_tables SET lease_expires_at = NULL
WHERE writer_id = $2 AND status = 'building' AND lease_expires_at IS NOT NULL
  AND table_name IN (SELECT partial_result FROM jobs
                     WHERE claimed_by = $1 AND execution = 'queued'
                       AND status = 'running' AND partial_result IS NOT NULL)
```

`$1 = instance_id`, `$2 = ResultStore::writer_id()`. It is a sweep over rows, not a CAS on one; the
`IN (SELECT ..)` is uncorrelated and portable on both backends. It runs after the jobs sweep, which
is safe because that sweep only NULLs `lease_expires_at` and the subquery's predicate is untouched by
it. It matches exactly the building rows of this instance's loop-claimed compute jobs: an inline
`run_now` row is excluded by `execution = 'inline'`; a library materialization has no jobs row; a row
recovery re-adopted under `"{writer_id}/claim-…"` matches neither arm. Any of those expires as usual.
The loop's own `ResultTable` hold is neither flagged nor released here; its next renewal misses and
flips `lost`. A NULL building lease is claimable at once through `claim_expired_building_table`, so
the successor's `dispatch_partial_result` takes the claim-and-fail arm → `MaterializeAnew` instead of
`BackOff`. A `BackOff` returns without finishing and costs a lease window plus an attempt, which is
what the linked sweep exists to avoid.

**`jobs.partial_result` is cleared when the building table is not adoptable.**
`create_result_table` records `partial_result` inside its transaction with a CAS that carries
`partial_result IS NULL`, so the column is written once per job row. A successor attempt that cannot
adopt its predecessor's table must therefore clear the column before it creates its own, or its CAS
matches zero rows and the job fails `JobAttemptSuperseded` although the successor did everything
right. `dispatch_partial_result` (`crates/jammi-ai/src/jobs.rs`) calls
`Catalog::clear_partial_result(job_id, instance_id, attempts, table)` on every arm that ends in
`MaterializeAnew` with a named table — the row vanished, the table is `failed`, or an expired
`building` row was just claimed and failed:

```sql
UPDATE jobs SET partial_result = NULL, updated_at = $now
WHERE job_id = $1 AND claimed_by = $2 AND status = 'running'
  AND attempts = $3 AND partial_result = $4
```

It is the attempt-guarded inverse of `record_partial_result`, and idempotent. Zero rows means a peer
already cleared it or a successor has taken over this attempt, which the existing
`JobAttemptSuperseded` path then reports; an error is logged and the attempt proceeds to the same
CAS. The invariant holds on the expiry path as well as after a RELEASE. A `ready` predecessor table
is adopted (the job finishes with it); a `building` row under a live lease is `BackOff`. Compute
producers are single-shot writes with no mid-table resume, so an expired `building` row is always
failed, never adopted.

### 2.7 Outcomes

n = `attempts`, r = `releases`; the row is shown after the event.

| Event | Row after | Successor | Cap consumed |
|---|---|---|---|
| DRAIN, job finishes | `completed`, lease NULL, n, r | — | 0 |
| DRAIN then SIGKILL (grace undersized) | `running`, live lease | expiry → requeue → claim n+1, r; resumes from the last bundle | 1 |
| RELEASE during a compute materialization (job hold and `ResultTable` hold registered) | `running`, `claimed_by = me`, jobs lease NULL, r+1; the linked `result_tables` row `building`, `writer_id = me`, lease NULL; the hold's `lost` flips at the keeper's next renewal | claims within one `idle_poll` (n+1, r+1); `dispatch_partial_result` claims the NULL-lease building row at once, fails it, clears `partial_result`, materializes anew and runs to `completed`; neither renewal re-arms either released lease | 0 net |
| RELEASE mid-epoch (hold registered) — outcome (ii) | `running`, `claimed_by = me`, lease NULL, r+1 | next tick → requeue → claim n+1, r+1; resumes from the last bundle; the aborted attempt never writes a bundle | 0 net |
| RELEASE with a claim committed and visible before sweep #1 or #2 — outcome (i) | `running`, lease NULL, r+1 | as (ii) | 0 net |
| RELEASE with the loop in the claim→hold prologue, including a claim parked before COMMIT whose unpark commits | `running`, `claimed_by = me`, lease NULL, n, r+1 — the hold site self-releases (or a sweep already did); no dispatch | as (ii) | 0 net |
| RELEASE with the loop idle or in `reclaim_expired_jobs`, cooperative arm | the claim never started, or `claim_next` found nothing → row untouched: `queued`, n = 0 | — | 0 |
| RELEASE, timeout-arm abort — outcome (iii): the 2e wait timed out with the loop between COMMIT and hold registration, or inside a claim whose COMMIT was already flushed | `running`, `claimed_by = me`, **live lease**, no hold, r unchanged; sweep #2 matches 0 rows | the queued arm requeues only once the lease expires → up to one `[lease] duration_secs` (30 s default) + one `idle_poll`; claim n+1, r; never `failed` | 1 |
| SIGTERM between `claim_next` returning and hold registration | the job runs under DRAIN to completion | — | 0 |
| RELEASE racing the finalize CAS | either order is safe (the CAS ignores the lease); if a successor already claimed, finalize matches 0 rows and the artifact prefix is orphaned for the existing GC | — | 0 |
| Keeper dead at SIGTERM | holds lost, trainer bails, lease stale → expiry; `/healthz` already 503 | as expiry | 1 |
| Catalog unreachable during DRAIN | heartbeats error; `lost` after one lease window → as keeper-dead | — | 1 |
| Catalog unreachable during RELEASE | 2b/2c/2g error and are logged; bounded exit (`ReleaseDegraded`); leases of both classes stay live and expire together, so the successor's `claim_expired_building_table` matches at once → `MaterializeAnew`, no `BackOff` | — | 1 |
| RELEASE, the linked building sweep errors while the jobs sweep succeeded (SQLite `database is locked` past `busy_timeout` on the second statement; a transient failure between the two) | jobs row `running`, lease NULL, r+1; the linked building row keeps a **live lease** the keeper renews until it dies at `session.close()`; `ReleaseDegraded` | the jobs row is claimed within one `idle_poll`; the successor's first `dispatch_partial_result` sees the live building lease → `BackOff` once, then claims after one `duration_secs` | release: 0 net; plus the successor's one `BackOff` attempt |
| Signal during preload | preload aborted, nothing served, worker never claimed, `workers` row deleted (awaited), exit 0 | — | 0 |
| Preload `Err` | `workers` row deleted (awaited), health down, session closed, non-zero exit | — | 0 |
| Loop task panics | guard → `Failed`; `/healthz` 503; `jammi_worker_claim_loop_up` 0 | — | — |
| Rollout cadence shorter than one epoch | no bundle ever advances; `releases` grows; visible only in the row (README operative rule) | — | 0 |
| Inline `run_now` in flight at RELEASE (server; the RPC future is dropped) | `running` with a live lease; hold skipped; the inline-liveness arm fails it once the instance is stale (2 × lease) | — | terminal |
| Inline `run_now` compute materializing at RELEASE (library; the process survives) | jobs row `running`, `execution = 'inline'`, live lease, r; building row `building`, live lease — both untouched by 2b, by the jobs sweep and by the building sweep's subquery; its `ResultTable` hold keeps renewing | none — the inline call runs to completion (`completed`, table `ready`), or falls to the row above if its caller's future is dropped | — |
| A reader-side reclaim (`InferenceSession::reclaim_job_on_read`, from `WaitJob` / `JobStatus`) sees a released row first | requeues immediately — harmless | — | 0 |

Outcome (iii) and the errored-building-sweep row are the only paths on which a released job is not
claimable within one `idle_poll_secs` under a reachable catalog. (iii) needs 2e's heartbeat-bounded
wait to time out with the loop between COMMIT and hold registration, or inside a claim whose COMMIT
was already flushed; a claim still before COMMIT when the abort lands rolls back and is the idle
row. Outcome (iii)'s exact row (live lease, `releases` unchanged) needs the COMMIT to land after
sweep #2 — a backend-internal race no park point can construct: a claim parked at `BeforeHold` has
already committed before sweep #1, so a park-driven timeout abort leaves `releases` already bumped.
The test pins the arm's invariant on the reachable row instead: a timeout abort never fails the job,
and the queued reclaim arm recovers it.

### 2.8 Gauges (`/metrics`)

| Metric | Type / labels | Source | Present on |
|---|---|---|---|
| `jammi_jobs_queued{kind}` | IntGaugeVec | the sampler: `count_jobs_by_kind_status` every `metrics_sample_secs`; a kind seen last sample but absent now is carried at 0 | worker-enabled processes |
| `jammi_jobs_running{kind}` | IntGaugeVec | same query | worker-enabled |
| `jammi_worker_jobs_in_flight` | IntGauge | 1 iff the slot holder is `JobRun` — never a `ClaimProbe`, never a held gang `Rank`, never an inline `run_now` | worker-enabled |
| `jammi_worker_claim_loop_up` | IntGauge | `LoopState == Running` (a failed `Weak` upgrade → 0) | worker-enabled |
| `jammi_lease_heartbeat_age_seconds` | Gauge | `keeper.last_renewed_at().elapsed()` | every process |

`MetricsRegistry::attach_worker` is called in `OssServer::bind` after `take_worker()`, so non-worker
processes omit the families (absent ≠ 0). The `/metrics` handler calls
`MetricsRegistry::refresh_gauges`, which upgrades the `Weak`, copies the sampler's snapshot and the
keeper's in-memory stamp into the gauges, then gathers — zero catalog statements per scrape,
observable through `WorkerShared::samples_taken`. The sampler (`sample_loop`) is its own task, spawned
with the loop and aborted on every stop path. `[worker] metrics_sample_secs` is validated ≥ 1 beside
`idle_poll_secs`.

### 2.9 Liveness (`/healthz`)

`jammi_server::build_health_router(readiness, metrics, liveness)` is the one router builder.
`LivenessCheck` mirrors `ReadinessCheck`; the production impl, `EngineLiveness`, holds the keeper and
a `Weak<WorkerShared>`. Unhealthy iff `!keeper.is_alive() || loop_state == Failed` → 503
`{"status":"unhealthy","lease_keeper":bool,"claim_loop":"running|stopped|aborted|failed|none"}`.
`Stopped`, `Aborted`, `none`, a dropped shared state (`stopped`) and draining are all 200. `alive`
flips in the keeper thread's exit guard the instant the thread dies, so the next probe sees it. With
a Kubernetes `livenessProbe` on `/healthz`, `failureThreshold × periodSeconds` is the restart-rate
knob for a dead keeper: each such restart consumes one attempt through the expiry path.

### 2.10 Library and Python surfaces

- Rust: `EmbeddedWorker::{spawn, begin_drain, stop_and_join -> StopOutcome, release_and_stop -> ReleaseReport, shared}`;
  `InferenceSession::{release_job_leases, close_worker_gate, open_worker_gate, host_admission}`;
  `LeaseKeeper::release_job_holds(bound)`;
  `Catalog::{release_job_lease, release_jobs_claimed_by, release_building_tables_of_claimant, clear_partial_result}`;
  `ResultTableCas::with_lease_present` and the pub `lease_present` field — every literal
  construction of `ResultTableCas` names the field.
- Python: `close(release: bool = False)` on `Database`, the embedded backend and the backend
  protocol. `release=False` is DRAIN (`stop_and_join`, then `session.close()`); `release=True` is
  `release_and_stop` (or `release_job_leases` with no worker), then `session.close()`. The remote
  arm accepts and ignores the flag: leases live in the server process.

### 2.11 `jammi-server release`

`Command::Release(ReleaseArgs { pid: i32 })` beside `Serve` and `Probe`
(`crates/jammi-server/src/main.rs`): `--pid` defaults to 1 and is validated ≥ 1 by the argument
parser; on unix `libc::kill(pid, libc::SIGINT)`; exit 0 on success, failure with the errno text
otherwise; unsupported off unix. PID 1 is `jammi-server` on every image (the `ENTRYPOINT`), and every
image runs non-root, which is sufficient to signal a process of the same user.

### 2.12 Runtime grace: `terminationGracePeriodSeconds` on the compute overlay

- `deploy/kubernetes/overlays/shape-d/statefulset-compute.yaml` — the compute tier is a
  `StatefulSet` behind a headless Service — carries `terminationGracePeriodSeconds: 600`, as does
  `deployment-scheduler.yaml`. 600 s is the cluster-autoscaler's `--max-graceful-termination-sec`
  default (R11), the upper anchor a scale-down honours; a larger value is honoured only by rollouts
  and `kubectl delete`, so the two are raised together.
- `deploy/kubernetes/base/deployment.yaml` (Shape C) spells no grace key: the 30 s Kubernetes
  default applies, its query replicas hold no training job, and a comment says a job in flight past
  the grace resumes from its last bundle after one lease window and consumes one attempt.
- `deploy/kubernetes/README.md`, "Shutdown: DRAIN and RELEASE":
  - **Operative rule** — the grace must cover one epoch's wall time, otherwise the drain never lands
    its final bundle before SIGKILL and the job takes the expiry path (one `[lease] duration_secs`
    window, one attempt).
  - **Rollout arithmetic** — a `StatefulSet` `RollingUpdate` has no `maxSurge`, and
    `maxUnavailable` sits behind the alpha `MaxUnavailableStatefulSet` gate, so assume strictly
    serial, descending ordinals: worst case replicas × 600 s (2 × 600 s = 20 min for the overlay's
    two replicas). `rollingUpdate.partition` is the only lever; `podManagementPolicy: Parallel`
    affects scaling only. The single-replica scheduler `Deployment` rolls in one drain (≤ 600 s),
    with `maxSurge: 25%` rounded up to 1 (R12).
  - **Caps a DRAIN cannot cross** — kubelet graceful node shutdown is off by default (R7); AWS Spot
    gives a two-minute notice (R13); GCP Spot ≤ 30 s (R14). On spot capacity use RELEASE:
    `preStop: exec: ["/usr/local/bin/jammi-server","release"]` (uniform) or
    `["/bin/sh","-c","kill -INT 1"]` (CUDA image only). A preStop hook runs before TERM is sent and
    inside the grace countdown (R9).
  - **Alternatives** — a derived image with `STOPSIGNAL SIGINT` (R4); `lifecycle.stopSignal` once
    `ContainerStopSignals` leaves alpha (R4b); `docker kill --signal=INT` /
    `docker compose kill -s SIGINT` (R10) — the daemon records a kill as a manual stop, so a
    `restart` policy does not bring that container back.
  - **Autoscaling input** — `jammi_jobs_queued{kind}` is the HPA/KEDA signal;
    `jammi_worker_jobs_in_flight` says whether a replica is busy.
- `deploy/docker-compose.yml`: `stop_grace_period: 120s`; `deploy/docker-compose.ci.yml`: `30s`.
- The user-facing description is `docs/guide/src/deploy-server.md#shutdown-drain-and-release`
  (modes, row outcomes, exit codes 0 / 3, the inline-unary bound, the `release` subcommand) and
  `#preloading-models`.

---

## 3. Principles preserved

- **Engine, not platform.** The engine ships generic actuators — signals, a subcommand, gauges — and
  no control loop. The README names Kubernetes and Docker features, never a consumer.
- **One binary, every topology; the library is never less capable than the server.** Both modes
  reach the library and the server through the same `release_and_stop`; the only server-only act is
  `process::exit`, a binary concern.
- **Tenant isolation.** No tenant label on any gauge. Both sweeps are instance-scoped and
  tenant-unscoped exactly like the claim; the building sweep reaches its rows through
  `claimed_by` + `execution = 'queued'` on `jobs`, never through the shared `writer_id` alone.
- **Typed refusal at the input edge.** Preload entries are validated at the startup edge;
  `metrics_sample_secs ≥ 1`; `workers.state` is CHECK-constrained; the re-arm guard sits in SQL for
  both lease classes, never in a per-hold flag alone.
- **Remote equals embedded.** `WorkerSummary.state` equals `workers.state`, written by one
  sequential chain, so no wire value precedes or outlives its row; the release surfaces share one
  mechanism and identical statements (D19).
- **Append-only migrations**, **lockstep versions across crates**. `releases` is not
  output-affecting, so replayability of producers is untouched.

---

## 4. Properties the tests hold

Catalog (`crates/jammi-db/tests/it/`, both backends where parameterised):

- A released job is requeued by the next reclaim without waiting for expiry —
  `released_job_is_requeued_by_the_next_reclaim_without_waiting_for_expiry`.
- The cap counts `attempts − releases`: 3 claims with 2 releases requeue, 3 claims with 0 releases
  fail — `reclaim_cap_counts_attempts_minus_releases`. A double release counts once —
  `a_double_release_increments_releases_once`.
- The release write is exactly lease → NULL, `releases + 1`, `updated_at`, every other column and
  every other instance's row untouched —
  `release_job_lease_write_is_exactly_lease_null_releases_plus_one_and_timestamp`.
- A hold of either class registered after a release never re-arms it —
  `a_hold_registered_after_release_never_re_arms_the_lease`,
  `a_released_building_lease_is_never_re_armed_by_the_keeper`.
- The keeper releases job holds only and skips inline rows —
  `release_job_holds_flips_lost_and_skips_inline_holds`.
- A released building table is claimable by the successor at once, a second sweep matches nothing,
  and inline, job-unlinked and `ready` rows are untouched —
  `released_building_table_is_claimable_by_the_successor_at_once`.
- Clearing `partial_result` lets the next attempt record its own table; a stale attempt or a
  different table is refused — `clear_partial_result_lets_the_next_attempt_record_its_own_table`.
- The finalize CAS still matches a released lease (the library divergence, pinned) —
  `finalize_cas_still_matches_a_released_lease`.
- `count_jobs_by_kind_status_matches_row_counts`; `set_worker_state_round_trips_through_list_workers`;
  the migration is present and ordered after `030_jobs_idempotency_key`
  (`crates/jammi-db/tests/it/migrations.rs`).

Library (`crates/jammi-ai/tests/it/jobs_shutdown.rs`, `jobs_compute.rs`):

- DRAIN lets the epoch bundle land; an idle stop returns within the idle poll —
  `stop_and_join_lets_the_epoch_bundle_land`, `stop_and_join_returns_within_idle_poll_when_idle`.
- RELEASE leaves `running` / NULL lease / no new bundle, and the resume manifest epoch does not
  advance — `release_and_stop_leaves_running_with_null_lease_and_no_new_bundle`.
- The 2e arms: `release_with_the_loop_paused_inside_claim_next_does_not_abort` (`Stopped`, never
  `Aborted`; the claim commits on unpark and self-releases; n = 1, r = 1),
  `release_before_the_loop_reaches_claim_next_leaves_the_row_untouched`,
  `release_with_the_loop_paused_in_the_claim_to_hold_prologue_self_releases`,
  `release_timeout_arm_leaves_the_honest_row_recovered_by_arm_1a`,
  `release_landing_during_the_reclaim_window_is_caught_by_the_second_gate_read`,
  `release_gate_refuses_every_claim_when_phase_flips_without_a_stop`,
  `every_phase_setter_pairs_the_stop_in_the_same_statement_group`.
- A compute job released mid-materialization resumes on the successor without a back-off and runs to
  `completed` — `release_mid_materialization_resumes_on_the_successor_without_backoff`; the same
  chain through plain expiry — `expired_compute_attempt_re_materializes_on_the_successor`.
- Inline work survives a release — `run_now_under_release_and_stop_does_not_change_in_flight`,
  `run_now_materialization_survives_release_and_stop`.
- One loop per session, and a completed release frees the slot —
  `a_second_spawn_on_the_same_session_is_refused_structurally`,
  `release_and_stop_completing_frees_the_slot_for_a_successor_not_stopped_by_the_old_release`.
- A cancelled stop never detaches the task —
  `dropping_the_guard_after_a_cancelled_stop_and_join_aborts_the_task`,
  `a_release_racing_an_in_flight_drain_reads_stop_unwitnessed`.
- Release evidence is never fabricated —
  `release_and_stop_report_matches_release_job_leases_on_the_pair_that_actually_differs`,
  `release_job_leases_is_unobserved_when_the_keeper_thread_is_dead`,
  `release_and_stops_second_sweep_reports_jobs_none_building_some_from_a_real_fault`,
  `release_job_leases_one_sweep_reports_building_none_jobs_some_from_a_real_fault`.

Server (`crates/jammi-server/tests/it/`):

- `serve_shutdown_modes.rs`: `sigterm_drains_the_in_flight_job_and_exits_drained`,
  `sigint_while_draining_releases_and_returns_released_within_two_heartbeats`,
  `release_preempts_a_drain_blocked_on_an_in_flight_unary`,
  `an_idle_subscribe_stream_ends_with_unavailable_draining_on_sigterm`,
  `drain_runs_rpc_drain_and_worker_join_concurrently`,
  `worker_does_not_stop_before_drain_is_signalled`,
  `released_server_never_finalizes_the_aborted_job` (also asserts the resume manifest epoch is
  unchanged), `bound_server_holds_the_worker_guard_when_worker_enabled`,
  `list_workers_reports_state_over_the_wire`,
  `release_subcommand_sends_sigint_and_the_child_exits_zero_within_two_heartbeats`.
- `health.rs`: `metrics_show_the_queued_count_change_within_one_tick_after_submit`,
  `metrics_omit_worker_gauges_when_no_worker_is_attached`,
  `a_thousand_scrapes_issue_zero_catalog_statements`,
  `in_flight_gauge_is_one_during_a_loop_claimed_job_and_zero_during_run_now`.
- `liveness.rs`: `healthz_flips_to_503_within_one_heartbeat_after_the_keeper_thread_dies`,
  `healthz_stays_200_while_draining`, `healthz_503_when_the_claim_loop_task_panics`.
- `readiness_preload.rs`: `readyz_is_503_until_preload_models_are_loaded_then_200`,
  `preload_of_an_unloadable_model_is_a_startup_error`,
  `preload_of_a_bare_id_with_no_models_row_is_a_startup_error`,
  `signal_during_preload_exits_without_serving`,
  `release_signal_during_preload_actually_releases`; and in
  `crates/jammi-db/src/config/tests.rs`,
  `preload_entry_rejects_an_unknown_task_token_with_a_typed_error`.
- `release_outcome`'s truth table is unit-tested beside it in `crates/jammi-server/src/runtime.rs`.

Python: `crates/jammi-python/tests/test_close_release.py::test_close_release_true_leaves_the_job_claimable`.
Compose: `tests/compose/shape_b_release.py`, run by `.github/workflows/compose-smoke.yml` — a long
job, SIGINT, exit 0, lease NULL with `releases = 1`, and the restarted process reclaims it.

---

## 5. Out of scope

Resume for `context_predictor` and compute kinds; HPA / KEDA / PDB / PriorityClass manifests; a
`releases` / `attempts` field on `JobSummary`; tenant-labelled gauges; an engine drain timeout;
slow-step detection; per-cloud spot overlays; a cap on `releases`; bounding inline unaries (a
`[server.limits]` concern); closing outcome (iii) by making `claim_next` abort-safe (a server-side
claim or a two-phase claim token — a different design).

---

## 6. References

- R1 https://www.postgresql.org/docs/current/server-shutdown.html — SIGTERM smart (disallow new
  connections, let sessions finish), SIGINT fast (abort current work, exit promptly), SIGQUIT
  immediate.
- R2 https://prometheus.io/docs/practices/naming/ — `jammi_` prefix, base units (`_seconds`), no
  `_total` on gauges, no high-cardinality labels.
- R3 https://docs.rs/tokio/latest/tokio/runtime/struct.Runtime.html — `spawn_blocking` tasks keep
  running; a runtime drop waits for them.
- R4 https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/ — TERM to PID 1, `STOPSIGNAL`
  honoured, SIGKILL at grace expiry; R4a core/v1 `terminationGracePeriodSeconds` default 30 s; R4b
  the `ContainerStopSignals` feature gate is alpha, default off.
- R5 https://docs.docker.com/reference/cli/docker/container/stop/ — SIGTERM, then SIGKILL after
  `--time` (default 10 s); `--signal`.
- R7 https://kubernetes.io/docs/concepts/cluster-administration/node-shutdown/ — graceful node
  shutdown defaults to 0 s (off).
- R8 https://docs.rs/tokio/latest/tokio/signal/unix/struct.Signal.html — signals coalesce only before
  the first poll; after a poll every further signal yields an item.
- R9 https://kubernetes.io/docs/concepts/containers/container-lifecycle-hooks/ — PreStop must
  complete before TERM is sent; the grace countdown begins before PreStop runs.
- R10 https://docs.docker.com/reference/cli/docker/container/kill/ and `docker compose kill -s`.
- R11 https://github.com/kubernetes/autoscaler/blob/master/cluster-autoscaler/FAQ.md —
  `--max-graceful-termination-sec` default 600.
- R12 https://kubernetes.io/docs/concepts/workloads/controllers/deployment/ — `maxSurge` /
  `maxUnavailable` default 25%; surge is rounded up, unavailable is rounded down.
- R13 https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-interruptions.html — two-minute
  interruption notice.
- R14 https://cloud.google.com/compute/docs/instances/spot — ≤ 30 s preemption notice.
