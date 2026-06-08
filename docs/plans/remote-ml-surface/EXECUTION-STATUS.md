# EXECUTION-STATUS — remote ML surface

Coordinator-maintained. Per-PR status, decisions log, research map, per-branch audit history. Each PR is implemented in its own worktree by a delegated session; the coordinator owns the plan, reviews, gates CI-to-green, and merges.

## Per-PR status

| PR | Status | Branch | PR | Notes |
|---|---|---|---|---|
| plan | in review | `plan/remote-ml-surface` | — | this spec set |
| T1 catalog training-job primitives | pending | — | — | rename→training_jobs, kind+lease+claim/heartbeat/reclaim, no callers |
| T2 submit + TrainingWorker | pending | — | — | three training verbs submit; uniform TrainingJob handle; reclaim replaces cleanup |
| T3 remote training + predict surface | pending | — | — | TrainingService, RemoteTrainingJob, predict on wire, conformance, doc fix |
| T4 compute-to-data parity | pending | — | — | assemble_context/neighbor_graph/propagate/eval_calibration on wire+client |
| N shared numeric utilities | pending | — | — | conformalize*/rrf_fuse as one shared impl (parallel with T1) |

## Decisions log

- **Surface decided by data residency, not GPU/CPU.** Engine-state verbs → remote (compute-to-data, no-fork); caller-array verbs (conformalize*, rrf_fuse) → shared client/local numerics. No demand-gated deferral.
- **propagate_embeddings / build_neighbor_graph(exact) stay CPU** — their fixed-order/`f64`/in-memory determinism is the auditability feature; GPU would break byte-identical reproducibility. They go remote for *data locality*, not GPU.
- **Generalize fine_tune_jobs → training_jobs** with a `kind`; one durable queue + one worker serves fine_tune, graph_fine_tune, context_predictor (2+ callers justify the abstraction; not a generic job system — no other caller).
- **train_context_predictor becomes a durable job** returning a `TrainingJob` handle (was synchronous → model_id). Greenfield API change; sync ergonomics preserved via `wait()`.
- **Lease-based orphan recovery** replaces fleet-unsafe `cleanup_stale_*` (which fails other instances' live jobs). Expired lease → re-queue (attempts-capped) → truthful failure.
- **Job stays a job, not a sync RPC** (pressure-test): durable + catalog-tracked survives connection/instance loss; enables backpressure and disaggregation; `wait()` is client-side polling sugar.
- **Working mode:** coordinator + delegated worktree sessions per PR (keep design coherent, context light).

## Research map (evidenced)

- `fine_tune_jobs` table `crates/jammi-db/src/catalog/schema.rs:47-59` (MIGRATION_001); tenant_id `:129,136` (MIGRATION_005). Status enum (Queued/Running/Completed/Failed) `crates/jammi-db/src/catalog/status.rs:47-84`.
- Job repo `crates/jammi-db/src/catalog/fine_tune_repo.rs`: create `:64`, get `:106`, update_status `:140`, set_output_model `:174`, **cleanup_stale (fleet-unsafe) `:207`**, list `:234`; `FineTuneJobRecord` `:8-22`.
- `session.fine_tune` `crates/jammi-ai/src/session.rs:808`; `spawn_fine_tune` (shared; derives `jammi:fine-tuned:{job_id}`, spawn_blocking) `:845-970`; `fine_tune_graph` `:989-1084`; `run_fine_tune_blocking` `:1534`; `record_failed` `:1521`.
- `FineTuneJob` handle `crates/jammi-ai/src/fine_tune/job.rs:11-76` (wait polls 100ms).
- Context predictor (GPU candle, **synchronous**, registers catalog model): `crates/jammi-ai/src/pipeline/context_predictor.rs` — `select_device` `:372,419`; `train_context_predictor` `:398`; reload-for-inference `:880`. propagate (deterministic CPU f64 fold) `crates/jammi-ai/src/pipeline/graph_propagation.rs:35-60`; exact neighbor-graph (in-memory deterministic) `crates/jammi-ai/src/pipeline/neighbor_graph.rs:28,151`.
- conformalize (caller arrays, empirical quantile) `crates/jammi-ai/src/predict/conformal.rs`.
- gRPC: proto `crates/jammi-ai/proto/jammi/v1/fine_tune.proto` (Start→{job_id} `:218`, Status→{status} `:228`); handler `crates/jammi-server/src/grpc/fine_tune.rs:62-111`; conversions `crates/jammi-ai/src/wire/fine_tune.rs`; mount (feature `train`) `crates/jammi-server/src/runtime.rs:489-494`. EmbeddingService/InferenceService/EvalService rpcs in `proto/jammi/v1/{embedding,inference,eval}.proto`.
- Client `clients/python/jammi_client/_database.py` (embedding+session only); stubs `clients/python/Makefile`; conformance `crates/jammi-python/tests/test_conformance.py` (`_REMOTE_VERBS` lacks training).
- Migrations: const in `crates/jammi-db/src/catalog/schema.rs` + tuple in `crates/jammi-db/src/catalog/migrations.rs:17-47`. Tests: `crates/jammi-db/tests/it/migrations.rs`, `crates/jammi-ai/tests/it/fine_tune.rs` (lifecycle `:250`, catalog CRUD `:730`).

## Audit history

_(Per branch, before each merge.)_
