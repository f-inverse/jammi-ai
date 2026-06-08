# Remote ML surface — durable training jobs + compute-to-data parity

**Status:** plan (authoritative spec for implementing subagents). Forced by the Theory↔Computation Cookbook (`docs/plans/40-cookbook`), which needs GPU ML through the published surface and surfaced that the remote surface is incomplete and one training verb is modeled inconsistently. Per the philosophy, the cookbook *informs* this roadmap; the engine gains generic primitives, not the consumer.

Run through the cookbook's per-spec lifecycle: plan → adversarial pressure-test (with research) → implement → independent audit → open PR → watch CI to green → merge → release. Coordinated centrally; each PR implemented in its own worktree. Tracked in `EXECUTION-STATUS.md`.

---

## The principle that decides the whole surface: compute goes to data

A verb belongs on the **remote (server) surface** iff its **inputs live in the engine** (result-tables, indexes, models). Then compute runs where the data is, and moving a consumer from embedded (Shape A) to remote (Shape C/D) is *configuration, not a fork* — their verb calls don't change. GPU-ness is incidental to this decision.

A verb whose inputs are **caller-supplied arrays** (not engine state) is a **stateless numeric utility**: the server holds none of its inputs, so a wire hop would only transport data the caller already has. These live as one shared client/local module so embedded and pure-client agree by construction.

This replaces any "expose it if a consumer asks" deferral with a decided split. (Confirmed compute facts: `propagate_embeddings` and `build_neighbor_graph(exact)` are *deliberately* CPU + fixed-order/`f64`/in-memory for byte-identical, auditable determinism — GPU would break that contract; so "needs GPU" is not the axis.)

### The decided surface

| Verb(s) | Inputs | Decision |
|---|---|---|
| `generate_embeddings`, `encode_query`, `search`, `infer` | engine table / index / model | **remote** (already on the wire) |
| `assemble_context`, `build_neighbor_graph`, `propagate_embeddings` | engine tables | **remote** (compute-to-data; CPU + deterministic, runs server-side) |
| `eval_embeddings/compare/inference/per_query`, `eval_calibration` | engine tables | **remote** (most on the wire; add `eval_calibration`) |
| `predict_with_context_predictor` | engine model + source | **remote** |
| `fine_tune`, `fine_tune_graph`, `train_context_predictor` | engine tables → model | **remote, as durable training jobs** |
| `conformalize`, `conformalize_cqr`, `conformalize_interval`, `rrf_fuse` | caller-supplied arrays | **client/local numeric utility** (principled exclusion from the server) |

The library keeps the full surface in-process (Shape A); the server reaches parity for every engine-state verb; conformal/rrf are shared numerics.

## Two structural fixes this forces

1. **Two training verbs, modeled inconsistently.** `fine_tune`/`fine_tune_graph` are durable catalog jobs (`fine_tune_jobs`, status queued→running→completed/failed, `tokio::spawn_blocking`, `FineTuneJob` handle polling the catalog). `train_context_predictor` is GPU (candle `select_device`) but runs **synchronously** and returns a `model_id` string. Two callers justify a generic **durable training-job** primitive; the predictor becomes a job (greenfield API change to a job handle).
2. **Orphan recovery is fleet-unsafe.** `cleanup_stale_fine_tune_jobs` marks *all* `Running` jobs → `Failed` at session init, tenant-scoped — so in a fleet a *starting* instance fails another instance's *live* jobs. A latent Shape C/D correctness bug. Replace with **lease-based** detection: a job is orphaned only when its lease expires, and an expired lease **re-queues** it (bounded by an attempts cap) so a dead worker's job is retried.

## The training-job model (the right abstraction)

A training job is **durable, self-describing work claimed under a lease**. A worker claims a queued job, reconstructs its data loader from a persisted self-contained `TrainingSpec`, trains while **heartbeating** the lease, and on completion records `model_id` (and on failure, the error). The worker runs identically in the embedded engine (Shape A/B) and the server `train` tier (Shape C/D); a "GPU worker pool" is just *N processes claiming from the shared catalog*. Stays training-specific (three callers); not a generic job system (no other caller — would fail the discipline test).

`TrainingSpec` (persisted JSON on the job): a tagged `kind` ∈ `{fine_tune, graph_fine_tune, context_predictor}` plus the kind's reconstruction inputs (tabular: source/columns/method/task; graph: node/edge sources, columns, provenance, `sample_config` incl. `sample_seed` for deterministic re-sampling; predictor: source/key/task/value columns + predictor spec) and common `{base_model, config}`. Determinism: graph re-sampling is seeded; metrics asserted to tolerance downstream.

## Implementation PRs (capability-split; each atomic across the crates it touches; sequential unless noted)

- **T1 — catalog training-job primitives (`jammi-db`).** Append-only migration: rename `fine_tune_jobs` → `training_jobs`; add `kind`, `claimed_by`, `lease_expires_at`, `attempts`, `training_spec`. Methods: `create_training_job`, `claim_next_training_job(worker_id, lease)` (atomic; SQLite single-writer txn / Postgres `FOR UPDATE SKIP LOCKED`), `heartbeat_training_job(job_id, worker_id, lease)`, `reclaim_expired_training_jobs(now, max_attempts)` (re-queue, or fail past cap) **replacing** `cleanup_stale_fine_tune_jobs`. SQLite + Postgres impls. Catalog + migration tests, incl. concurrent-claim exactly-one-wins and forced-expiry reclaim. **No callers yet** (introduce the primitive first). Update all existing `fine_tune_jobs` references to the renamed table.
- **T2 — submit + `TrainingWorker` (`jammi-ai`, `jammi-server`).** `fine_tune`/`fine_tune_graph`/`train_context_predictor` persist a typed `TrainingSpec` and **submit** (no inline spawn); all three return a uniform **`TrainingJob`** handle (`model_id`/`status()`/`wait()`) — `train_context_predictor` becomes a job. One `TrainingWorker` claims → reconstructs by `kind` → trains with a heartbeat task → sets `completed`+`model_id` / `failed`+error / releases on abort; loops in the embedded engine and the server `train` tier; `reclaim_expired` replaces startup cleanup. Atomic across `jammi-ai`/`jammi-server` (+ `jammi-python` handle rename). Test: a queued job submitted by one session runs to completion on a fresh worker (proves durability).
- **T3 — remote training + predict surface (`proto`, `wire`, `jammi-server`, `clients/python`, `jammi-python` conformance, docs).** Generic training service: `StartTraining(kind, spec) → {job_id, model_id}`, `TrainingStatus(job_id) → {status, model_id, error}` (graph-expressible). `RemoteTrainingJob` in `jammi-client` mirroring local `TrainingJob`. Add `predict_with_context_predictor` to the inference surface. Extend the M2 conformance test (`crates/jammi-python/tests/test_conformance.py`) to the training/predict verbs (remote shape == local). Correct `docs/guide/src/deploy-server.md` ("or CLI" → "library or Python package"; the CLI has no ML verbs).
- **T4 — compute-to-data parity for the remaining engine-state verbs (`proto`/`wire`/`server`/`client`).** Put `assemble_context`, `build_neighbor_graph`, `propagate_embeddings`, `eval_calibration` on the wire + client (they return result-tables/handles the client references). Committed (not demand-gated); lower priority than T1–T3.
- **N — shared numeric utilities (`clients/python` + a shared spot).** `conformalize*` and `rrf_fuse` as one implementation usable from both the embedded wheel and the pure client (no server round-trip). Parallelizable with T1.

Phase order for the cookbook's unblock: **T1 → T2 → T3** (training jobs + predict = the GPU cluster the keystone needs), then cookbook emit via `connect("grpc://…")`; **T4/N** complete parity in parallel/after.

## Research map (evidenced)
See `EXECUTION-STATUS.md` "Research map" for file:line citations of the current subsystem (catalog schema/repo, `session.fine_tune`/`fine_tune_graph`, `pipeline/context_predictor.rs`, proto services, `jammi-client`, conformance test, migrations + test conventions).

## Determinism / hermeticity
SQLite single-writer txn / Postgres `FOR UPDATE SKIP LOCKED` for atomic claims; lease expiry → re-queue (attempts-capped) → truthful failure; worker reconstruction is in-memory-carryover-free; graph re-sampling seeded. Default-hermetic tests (tiny fixture model), no live network.
