# EXECUTION-STATUS — remote ML surface

Coordinator-maintained. Per-PR status, decisions log, research map, per-branch audit history. Each PR is implemented in its own worktree by a delegated session; the coordinator owns the plan, reviews, gates CI-to-green, and merges.

## Per-PR status

| PR | Status | Branch | PR # | Notes |
|---|---|---|---|---|
| plan | merged | `plan/remote-ml-surface` | #107 | spec set committed to `docs/plans/remote-ml-surface/` |
| T1 catalog training-job primitives | merged | `t1-training-job-catalog` | #108 | lease-based training-job queue primitives in `jammi-db`; `training_jobs` table, kind+lease+claim/heartbeat/reclaim; no callers yet. Hardened by Postgres queue test coverage + true-parallel claim test (#114). |
| T2a rename → training_jobs/TrainingJob | merged | `t2a-rename-training-jobs` | #109 | rename `fine_tune_jobs` infrastructure to `training_jobs/TrainingJob`; no behavior change |
| T2b durable training worker + submit | merged | `t2b-training-worker` | #112 | durable `TrainingWorker`; `submit` verb; context-predictor promoted to a durable job returning a `TrainingJob` handle; `reclaim` replaces fleet-unsafe `cleanup_stale_*` |
| T3 remote training + predict surface | merged | `t3-remote-training-surface` | #115 | `TrainingService` gRPC + `RemoteTrainingJob` handle; predict on the wire (`InferenceService.Predict`); conformance test updated |
| T4 compute-to-data wire parity | merged | `t4-compute-to-data-wire-parity` | #119 | `assemble_context` / `build_neighbor_graph` / `propagate_embeddings` / `eval_calibration` on gRPC (`PipelineService` + `EvalService`); matching client wrappers in `jammi_client._database` |
| N shared numeric utilities | merged | `n-client-conformal-numerics` | #110 | `conformalize*` / `rrf_fuse` as pure-Python in `jammi_client`; verb-surface parity with the embedded engine; conformance test pins numeric identity across transports |
| H1 residual: stub-generator pin + runtime floors | merged | — | #168 | `grpcio-tools==1.80.0` pinned; runtime floors coupled to the pinned stubs; import-time crash at declared minima eliminated |
| H1 residual: `RemoteDatabase.infer` | merged | — | #169 | bulk inference on the wire (`InferenceService.Infer`); returns `pyarrow.Table`; same call surface as embedded `Database.infer`; conformance-guarded by `_INFERENCE_VERBS` |
| H1 residual: channel catalog tenant-scope + TOCTOU | merged | — | #170 | cross-tenant channel leak D1: migration 020 adds `tenant_id` to `evidence_channels` / `evidence_channel_columns`; `UNIQUE (tenant_id, channel_name)`; partial unique index on `channel_name WHERE tenant_id IS NULL` closes global-channel-name TOCTOU race |
| H1 residual: eval family on `RemoteDatabase` | merged | — | #171 | `eval_embeddings` / `eval_per_query` / `eval_inference` / `eval_compare` on `RemoteDatabase` (`EvalService`); `eval_embeddings` `model` → `embedding_table` rename; conformance-guarded by `_EVAL_VERBS` |
| H1 residual: channel family on `RemoteDatabase` | merged | — | #172 | `register_channel` / `add_channel_columns` / `list_channels` on `RemoteDatabase` (`CatalogService`); `Database.list_channels` added to embedded binding; conformance-guarded by `_CHANNEL_VERBS` |
| H1 residual: `eval_compare` order-invariance fix | merged | — | #173 | `bootstrap_ci` canonicalizes input (sort) before seeded resample; CI is now a function of the sample multiset, not emission order; unit test pins order-invariance directly; `delta_significance` pinned on a non-degenerate paired set |
| H1 residual: v0.26.5 lockstep release | merged | — | #174 | lockstep release across all publish lanes; tags `v0.26.5` + `py-v0.26.5` |

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
- gRPC: protos live under `crates/jammi-wire/proto/jammi/v1/` — training surface `training.proto` (`rpc StartTraining` `:34`, `rpc TrainingStatus` `:40`); handler `crates/jammi-server/src/grpc/training.rs`; proto↔domain conversions `crates/jammi-wire/src/training.rs` (engine-side spec conversions in `crates/jammi-ai/src/wire/training.rs`); mount (feature `train`) `crates/jammi-server/src/runtime.rs:497-514`. EmbeddingService/InferenceService/EvalService/PipelineService rpcs in `crates/jammi-wire/proto/jammi/v1/{embedding,inference,eval,pipeline}.proto`.
- Client `clients/python/jammi_client/_database.py` (as of 0.26.5: embedding+session, training, pipeline, eval (full family), bulk inference, channel family, conformal/RRF numerics, mutable+topic+pubsub — the full verb surface is now on `RemoteDatabase`); conformance `crates/jammi-python/tests/test_conformance.py` — training verbs pinned in `_TRAINING_VERBS`; pipeline verbs (`build_neighbor_graph`, `propagate_embeddings`, `assemble_context`) pinned in `_PIPELINE_VERBS`; eval verbs (`eval_embeddings`, `eval_per_query`, `eval_inference`, `eval_compare`, `eval_calibration`) pinned in `_EVAL_VERBS`; inference verb (`infer`) pinned in `_INFERENCE_VERBS`; channel verbs (`register_channel`, `add_channel_columns`, `list_channels`) pinned in `_CHANNEL_VERBS`; numeric verbs (`conformalize*`, `rrf_fuse`) pinned in `_NUMERIC_VERBS`; mutable+topic+pubsub verbs pinned in `_MUTABLE_TOPIC_VERBS`. Open: the channel gRPC error-taxonomy (`JammiError::EvidenceChannel` → `Code::Internal` instead of `AlreadyExists` / `NotFound` / `InvalidArgument`) is an honest deferred item, tracked in the roadmap at H3 §3.8.
- Migrations: const in `crates/jammi-db/src/catalog/schema.rs` + tuple in `crates/jammi-db/src/catalog/migrations.rs:17-47`. Tests: `crates/jammi-db/tests/it/migrations.rs`, `crates/jammi-ai/tests/it/fine_tune.rs` (lifecycle `:250`, catalog CRUD `:730`).

## Audit history

_(Per branch, before each merge.)_

### `t1-training-job-catalog` (T1, #108) and T1-hardening (#114)

T1 introduced the `training_jobs` catalog table with lease-based claim/heartbeat/reclaim in `jammi-db`. T1-hardening (#114) added Postgres queue test coverage, a true-parallel claim test, and a sharper error string.

### `n-client-conformal-numerics` (N, #110)

`conformalize` / `conformalize_interval` / `conformalize_cqr` / `rrf_fuse` ported to pure Python in `jammi_client._conformal`; verb surface and numeric output verified identical to the embedded engine by a new hermetic conformance test.

### `t2a-rename-training-jobs` (T2a, #109)

Renamed `fine_tune_jobs` infrastructure to `training_jobs` / `TrainingJob` across the workspace; no behavior change. The CI docs fix for the broken intra-doc links caused by the rename is #116.

### `t2b-training-worker` (T2b, #112)

Durable `TrainingWorker` with lease-guarded claim and heartbeat; `submit` on the session; `train_context_predictor` promoted to a durable job returning a `TrainingJob` handle; `reclaim_stale_jobs` replaces fleet-unsafe `cleanup_stale_*`.

### `t3-remote-training-surface` (T3, #115)

`TrainingService` gRPC with `StartTraining` / `TrainingStatus`; `RemoteTrainingJob` in `jammi_client`; `predict_with_context_predictor` wired through `InferenceService.Predict`; conformance test updated. Base-model FK fix (#118) landed the same day — the context predictor was not using the catalogued model PK for the FK, caught by the post-merge correctness sweep.

### `t4-compute-to-data-wire-parity` (T4, #119)

`PipelineService` gRPC: `BuildNeighborGraph` / `PropagateEmbeddings` / `AssembleContext`; `EvalService`: `EvalCalibration` — four engine-state verbs added to both the gRPC server and the pure-Python `RemoteDatabase` client. Conformance tests pin pipeline and numeric verb surfaces across transports.

### Follow-up fixes after T4 (same workstream, post-#119 on `main`)

The following fixes belong to the T1–T4+N workstreams and landed between the T4 merge and the 0.26.4 release:

- **#120 CLI full-UX** — prebuilt binaries, turnkey GPU image, docs (packaging/deployment surface for the training tier).
- **#121 Configurable worker lease/heartbeat/poll intervals** — `TrainingWorker` intervals made configurable; closes the lease-interval tunability gap the T2b spec left open.
- **#122 Fix: `jammi serve` emitted no logs** — the `jammi` CLI never installed a tracing subscriber, so every `tracing::*` record was dropped (silent regardless of TTY); fixed by extracting a shared `jammi_server::telemetry::init_tracing` called from both the standalone `jammi-server` binary and the `jammi serve` subcommand.
- **#123 Artifacts through the object store** — model artifacts routed through the configured object-store backend instead of the local FS; required for multi-instance worker fleets.
- **#124 Gated multi-process distributed-validation lane** — a parallel validation pass the durable worker can run across a fleet; gated behind a feature flag.
- **#128 Fix: worker fleet S3 driver + diagnosability** — the worker binary was built without the S3 storage driver (artifact publish failed with `SchemeNotEnabled`); the lane was also made diagnosable — errors now surface to the test harness (`Fleet::dump_diagnostics` + `harness::await_job`, which fails loudly on an unexpected worker exit and dumps the job's final catalog row on timeout) rather than silently running out the poll timeout.
- **#140 Fix: tenant-qualify model catalog PK (D1)** — the model catalog PK was not tenant-qualified; a cross-tenant model-overwrite was possible.
- **#148 Fix: context-predictor z-space target standardization** — the amortized context predictor collapsed on high-offset targets (year/price/count); z-space standardization of the predictor's target + in-context members' y restores correct fitting. (Completes the 0.26.2 A3 bidirectional win.)
- **#158 RemoteDatabase mutable-table + topic + pub/sub gap** — closes the cp9 client gap: `create_mutable_table` / `drop_mutable_table` / `list_mutable_tables`, `register_topic` / `drop_topic` / `list_topics`, `publish_topic`, `subscribe_collect` added to `RemoteDatabase`, with a conformance guard pinning the set (`_MUTABLE_TOPIC_VERBS`). Also fixes multi-chunk `publish_topic` parity: the remote client collapses a multi-chunk `pyarrow.Table` to one batch (`combine_chunks`) before the wire hop, mirroring the embedded `concat_batches` path.
- **#160 Fix: hard-negative default-overlay + bounded mining; epoch-count oracle** — hard-negative mining defaults now overlay correctly on the proto wire (zeros no longer ship as literal values that validation rejects); mining is memory-bounded (no second full corpus-embedding copy). On the epoch question (#64): not a bug — there is exactly one `for epoch in 0..epochs` loop shared by the tabular and graph paths (no warmup/zeroth epoch); the engine's epoch count is verified exact and pinned by a regression-guard oracle (`optimizer steps == epochs × ceil(batches/grad_accum)`).
- **#161 Tenant scope-safe context manager + unambiguous setter** — `with_tenant` (bind-in-place, returned `None`) replaced by `set_tenant` (unambiguous `-> None` setter) + `tenant_scope` (block-scoped context manager restoring the prior tenant); both embedded and remote surfaces updated atomically.

### H1 residual closure — v0.26.5 bundle (#168–#174)

The seven embedded-only verbs flagged in the tracker as of 0.26.4 (`infer`, `eval_embeddings`, `eval_per_query`, `eval_inference`, `eval_compare`, `register_channel`, `add_channel_columns`) are now ALL on `RemoteDatabase`, plus `list_channels` which was also added. Released in v0.26.5; validated by cookbook chapter 14 (`jammi-cookbook` PR #15).

- **#168 — stub-generator pin + honest runtime floors.** `grpcio-tools==1.80.0` pinned in the `dev` extra; runtime floors (`grpcio` / `protobuf`) coupled to the generated stubs. Eliminates the import-time crash a wheel installed at its own declared minima would hit.
- **#169 — `RemoteDatabase.infer`.** Bulk inference on the wire (`InferenceService.Infer`); returns `pyarrow.Table`; same call surface as embedded `Database.infer`. Conformance-guarded by the new `_INFERENCE_VERBS` set in `test_conformance.py`.
- **#170 — evidence-channel catalog tenant-scoped (D1 bug fix) + TOCTOU close.** Migration 020 reshapes `evidence_channels` / `evidence_channel_columns` to carry `tenant_id` with `UNIQUE (tenant_id, channel_name)` and composite FK. `register` / `add_channel_columns` / `get` / `list` all read and write `tenant_id`. A partial unique index on `channel_name WHERE tenant_id IS NULL` enforces global-channel-name uniqueness atomically, closing the race where two concurrent unbound registrations of the same name could both commit. Adversarial cross-tenant isolation test added.
- **#171 — eval family on `RemoteDatabase`.** `eval_embeddings` / `eval_per_query` / `eval_inference` / `eval_compare` added to `RemoteDatabase` (`EvalService`); return the same nested report dicts as the embedded `Database`. `eval_embeddings` kwarg renamed `model` → `embedding_table` (the former name misdescribed the lookup). Conformance-guarded by the new `_EVAL_VERBS` set; projection shape pinned against a shared golden fixture (`tests/fixtures/eval_report_projection.json`).
- **#172 — channel family on `RemoteDatabase`.** `register_channel` / `add_channel_columns` / `list_channels` added to `RemoteDatabase` (`CatalogService`); `Database.list_channels` added to the embedded binding (read half was missing). Both return `{"channel_id", "priority", "columns": [{"name", "data_type"}]}` ordered by `(priority, channel_id)`. Conformance-guarded by the new `_CHANNEL_VERBS` set.
- **#173 — `eval_compare` significance-CI order-invariance fix.** `bootstrap_ci` in `jammi-numerics` now sorts its input before the seeded resample, making the CI a function of the sample multiset rather than emission order. Surfaced by cookbook chapter 14 (a self-comparison on the same run in different orders produced differing intervals while all point metrics agreed). Unit test pins `bootstrap_ci` order-invariance directly; `delta_significance` pinned on a non-degenerate paired set.
- **#174 — v0.26.5 lockstep release.** All publish lanes green: crates ×12, PyPI (`jammi-ai`, `jammi-client`, `jammi-server`, `jammi-server-cu12`), npm, GHCR images, release binaries. Tags `v0.26.5` + `py-v0.26.5`.

**Validation:** cookbook chapter 14 (`jammi-cookbook` PR #15) drives all the new verbs on both transports asserting `remote == embedded`. It surfaced #173 (the eval_compare CI determinism bug) and an NER-confidence representation question (resolved as f32-precision parity) before release.

**Honestly deferred (not claimed done):** the channel gRPC error-taxonomy — `JammiError::EvidenceChannel` variants return `Code::Internal` instead of `AlreadyExists` / `NotFound` / `InvalidArgument`. Known open item; sequenced to H3 §3.8 in the roadmap.
