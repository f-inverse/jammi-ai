# UNITS — per-unit phase-2 contracts (#500), v2

Each unit: `files_in_scope` (write-owner), `invariants_to_preserve`, `acceptance` (RED at the
base commit, GREEN on the branch, asserting a criterion), `lane`, `depends_on`, `size` (S/M/L/XL
by blast radius). Full CI gate for every commit: the verbatim step list of `ci.yml` with
per-step `$?`. Naming per README ruling 23.

## U1 — DataFusion 54 line upgrade (PR-A, alone)

- **files_in_scope**: (docs-ci) `Cargo.toml` workspace pins — `datafusion 54`,
  `arrow`/`arrow-array`/`arrow-schema`/`arrow-ipc`/`arrow-flight`/`parquet 58`, `object_store
  0.13`, `datafusion-federation 0.5.5`, `datafusion-flight-sql-server 0.4.18`, `pyo3-arrow` to the
  arrow-58 line; `Cargo.lock`; `deny.toml`; `.github/workflows/ci.yml` — a NEW step
  `cargo clippy -p jammi-db --features postgres,mysql --all-targets -- -D warnings` registered
  with `ci/scripts/check_lint_surface_closure.py`. (db) `crates/jammi-db/Cargo.toml:55`
  `datafusion-table-providers` 0.10.1 → 0.13 and its compat surface: `PostgresTableFactory`,
  `PostgresConnectionPool` (`source/postgres.rs:10-11`), `MySQLTableFactory`,
  `MySQLConnectionPool` (`source/mysql.rs:10-11`); `tenant_scope.rs` analyzer API. (ai-core,
  wire-server, python, cli, bench) compile fixes in their crates.
- **invariants_to_preserve**: B6, K6, K4 (`grpc_remote_session.rs` green), B5, K1.
- **acceptance**: workspace gate green including the new db-features clippy step (RED at
  base: the step does not exist and the features do not compile on 54); cookbook 6.5 zero
  divergence; `cargo tree -d` shows one `arrow` and one `datafusion` line; a
  `DATAFUSION_VERSION` guard test (guard, not the oracle).
- **lane**: hermetic + cookbook; `distributed.yml` dispatched manually before merge.
- **depends_on**: S3 (sizes it). **size**: L if S3 compiles with local fixes only; XL otherwise.

## U7a — `gpu-gang.yml` pod leg (PR-B commit 1)

- **files_in_scope** (docs-ci): `.github/workflows/gpu-gang.yml` (label `run-gang`, nightly,
  manual; never `push`/`workflow_call`), `ci/scripts/runpod_lib.sh` (`gpuCount` becomes a
  parameter of the shared deploy payload, default 1 — three-lane blast radius: gpu-prove,
  gpu-perf-ab, gpu-dev), `ci/scripts/runpod_gpu_gang.sh` (pod leg: 1 pod × 2 GPU
  A100-SXM4-80GB), `ci/scripts/check_cuda_run_artifacts.py` (a `gang` artifact kind: world,
  collective, per-rank device, digest pair, per-step loss delta, ε), `docs/maintainer/dev-gpu.md`.
- **invariants_to_preserve**: `check_gpu_prove_once.py` P1 rules; `check_ci_guard_wiring.py`; B2.
- **acceptance**: the gang artifact-schema test (RED at base); `check_gpu_prove_once.py`
  applied to the new workflow's `on:` block; `check_ci_guard_wiring.py` green; every existing
  lane still deploys `gpuCount: 1`. Provisioning proof is spike S4, not an acceptance.
- **lane**: gate scripts. **depends_on**: S4. **size**: L.
- **cost ceiling** (human-approved before first run): ≤ 1 h × 2 GPU × $1.59 ≈ $3.2 per run.

## U2a — `TrainingSet` producer (PR-B commit 2)

- **files_in_scope**: (db) `store/manifest.rs` (`ProducingDescriptor::TrainingSet`,
  `canonical_bytes`), `catalog/result_repo.rs` (`ResultTableKind::TrainingSet`), tests.
  (ai-core) `fine_tune/worker.rs::run_spec` (materialize-or-reuse, then read the table into
  today's loader — a compiling intermediate), `fine_tune/graph_sampler.rs` (pairs → table),
  `pipeline/recompute.rs` (arm = re-materialize), the cookbook fixture golden.
- **invariants_to_preserve**: K1, K7 (descriptor: source anchors, columns, task, format,
  order rule — no topology, no split), K2 (empty training set refused), B1/B2 naming, B6.
- **acceptance**: (a) `fine_tune` creates a `TrainingSet` result table with a definition hash
  and attestation (RED at base); (b) two jobs over the same source/columns/task reuse one table
  (RED at base); (c) refactor parity: adapter bytes identical to base on every cookbook
  fine-tune fixture.
- **lane**: hermetic + cookbook. **depends_on**: U1. **size**: M.

## U4a — `Collective` trait; device-plural session; `CacheKey`; config refusals (PR-B commit 3)

- **files_in_scope**: (ai-core) new `fine_tune/collective/{mod.rs, noop.rs, local.rs, nccl.rs}`,
  `model/cache.rs` (`CacheKey { model_id, device, task: Option, backend: Option }` — shared
  shape with plan 65, recorded in both ledgers), `model/backend/mod.rs` (`DeviceConfig` plural),
  `concurrency/gpu_scheduler.rs` (per device), `session.rs`, `jammi-ai/Cargo.toml` (`cuda`
  adds `candle-core/nccl`). (db) `config/mod.rs` (`[gpu] devices`, `[training] world_size`,
  `collective`), tests.
- **invariants_to_preserve**: B4 (topology is configuration), K2 (`world_size > devices`,
  `nccl` without CUDA, `world_size > 1` with `cached` or `hard_negatives.refresh_every > 0`
  refused with typed errors), K4 (remote parity suite unchanged), B6.
- **acceptance** (hermetic): (a) `Local` over 2 ranks: `all_gather` layout and rank-ordered
  `all_reduce_sum` are deterministic and equal a serial reference bit-for-bit (RED at base);
  (b) each refusal above (RED at base); (c) two devices in one session hold two cache entries
  for one model id (RED at base: keyed by id alone). (pod leg smoke) `Nccl` over 2 devices
  reduces a known vector.
- **lane**: hermetic (+ pod leg smoke). **depends_on**: S1. **size**: L. No loader contact.

## U2b — Streaming loader; partition rule; scaler; whole-set arms (PR-B commit 4)

- **files_in_scope** (ai-core): `fine_tune/data.rs` (stream + per-batch converters for every
  format), `fine_tune/trainer.rs` (epoch loop over the stream; `batches_per_epoch` from
  `train_count`), `fine_tune/worker.rs::run_spec` (`PartitionSpec { rank, world, batch, rule }`),
  `fine_tune/regression_loss.rs` (scaler from the streamed train prefix),
  `fine_tune/hard_negative_miner.rs` and `fine_tune/gradcache.rs` (stream-sourced, W=1-only),
  `fine_tune/batch_bucket.rs` (rung pinning option), tests. (db) `store/mod.rs` reader slicing.
- **invariants_to_preserve**: K3 (scaler over the train prefix, bit-identical), K2, B6.
- **acceptance**: (a) resident-row high-water mark ≤ `batch × prefetch` on a fixture larger
  than the bound, on every non-whole-set arm (RED at base); (b) partition rule: for W ∈
  {1,2,4} the multiset of rows over ranks at each global step equals the W=1 batch (RED at
  base); (c) refactor parity holds on every cookbook fixture including regression; (d)
  mining/GradCache runs at W=1 produce bytes identical to base.
- **lane**: hermetic + cookbook. **depends_on**: U2a. **size**: XL.

## U3 — `FineTune` producer; migration 029; cache reuse (PR-B commit 5, concurrent with U2b)

- **files_in_scope**: (db) `store/manifest.rs` (`ProducingDescriptor::FineTune`;
  `MaterializationEnv` kernel-profile), `catalog/{schema.rs, migrations.rs}` (029, nullable
  columns), `catalog/model_repo.rs` (`probe_model_by_definition`, NULL never matches),
  `store/artifact.rs` (manifest last), `store/reconcile.rs` (prefix reaped only when
  unreferenced), `tests/it/migrations.rs`. (ai-core) `fine_tune/worker.rs::publish_and_finalize`
  (materialization; probe before training; own name → reused prefix), `pipeline/recompute.rs`
  (arm = retrain), `model/resolver.rs` (manifest on `ModelRecord`).
- **invariants_to_preserve**: K5, K7 (exhaustive destructuring of `FineTuneConfig` and
  `TrainingCommon`, fields existing at this commit), K1, B6, B1 (no `register_*`).
- **acceptance**: (a) same spec on the same training-set digest with `CachePolicy::Use` trains
  once, two model rows share one prefix, deleting one leaves the prefix (RED at base); (b) the
  exhaustive-destructuring completeness test (RED at base); (c) 029 append-only test.
- **lane**: hermetic. **depends_on**: U2a. **size**: M. Co-ownership: `manifest.rs` with U4b.

## U4b — Rank context; gather rule; lockstep; single-node gang (PR-B commit 6)

- **files_in_scope** (ai-core): `fine_tune/trainer.rs` (`RankContext`; gather-then-global-loss;
  local-slot gather backward; canonical-order reduce; lockstep flags; rank-0-only checkpoint),
  `fine_tune/optimizer.rs` (zero-filled reduce set), `fine_tune/worker.rs::run_spec` (spawn W
  local ranks), `fine_tune/spec.rs` (`TrainingCommon.world_size`), `store/manifest.rs` (topology
  fields; extends U3's completeness test), tests.
- **invariants_to_preserve**: B4, K7, K2, K4, B6.
- **acceptance** (hermetic, `Local`): (a) W=2 twice → identical bytes (RED at base); (b)
  gather exactness for CoSENT/AnglE/MNRL bit-for-bit vs W=1 on the same rows (RED at base);
  (c) W=2 × B vs W=1 × 2B within pre-registered ε at `lora_dropout=0`, rung pinned (RED at
  base); (d) lockstep: forced divergence on one rank; a Var absent from one rank's `GradStore`
  — the gang completes (RED at base); (e) W=1 via `Noop` byte-identical to U2b's golden.
  (pod leg, `Nccl`, 2×A100): (a) as a digest pair + per-step delta, (c) with GPU-measured ε;
  artifact committed as PR-B commit 7.
- **lane**: hermetic + gpu-gang pod leg. **depends_on**: U2b, U3, U4a. **size**: L.

## U7b — cluster leg + cluster reap (PR-C commit 1)

- **files_in_scope** (docs-ci): `ci/scripts/runpod_lib.sh` (cluster create/teardown primitive
  with deadline), `ci/scripts/runpod_gpu_gang.sh` (cluster leg: TRAINING cluster, 2 pods × 2
  GPUs), `.github/workflows/gpu-reap.yml` (clusters enumerated and reaped), `gpu-gang.yml`.
- **acceptance**: reap enumerates clusters (RED at base: pods only); P1 rules; guard wiring.
- **lane**: gate scripts. **depends_on**: U7a, S4. **size**: M.
- **cost ceiling**: ≤ 1 h × 4 GPU × $1.59 ≈ $6.4 per run; label-only until a flake-free streak.

## U5a — `gang.proto`; `GangService`; `FetchPartition`; authorization (PR-C commit 2)

- **files_in_scope**: (wire-server) `crates/jammi-wire/proto/jammi/v1/gang.proto`
  (`RunRank(RankAssignment) returns (stream RankEvent)`; `FetchPartition(PartitionRequest)
  returns (stream ArrowIpc)`), `crates/jammi-wire/src/{lib.rs, gang.rs}`,
  `crates/jammi-server/src/grpc/gang.rs` (tenant-scoped mount via `mount_tenant_scoped`; the
  job/lease/claimed_by verification; ids resolved through the catalog; #485 bounds),
  `crates/jammi-server/src/runtime.rs`, `crates/jammi-server/tests/it/{gang_partition.rs,
  gang_authz.rs}`. (db) `config/mod.rs` (`peers`, `rank_timeout_secs`).
- **invariants_to_preserve**: B5, K2, B1 (no `stage`/`register` stems), B6.
- **acceptance**: (a) the streamed bytes of partition r equal the local read of partition r
  (RED at base); (b) `RunRank` for a job that is not running / not claimed by the named worker
  / lease expired is refused with a typed status (RED at base); (c) a lesser-or-equal attempt is
  refused, a greater one supersedes (RED at base).
- **lane**: hermetic + server it-suite. **depends_on**: U4a. **size**: L.

## U6 — Partition-aware inference operator; distributed frozen forward (PR-C commit 3)

- **files_in_scope** (ai-core): `operator/inference_exec.rs` (inherit input partitioning),
  `operator/runner.rs` (one load per process), `pipeline/embedding.rs` (stream into the sink;
  fan-out through `FetchPartition`), `fine_tune/worker.rs::run_fine_tune_blocking` head-target
  arm (`worker.rs:2375`), tests. (db) `store/result_sink.rs` (partition-ordered append).
- **invariants_to_preserve**: K4 (4-partition table == 1-partition table, bytes), K1, B3, B6.
- **acceptance**: (a) byte parity across partition counts (RED at base); (b) resident batches
  ≤ a bound (RED at base: `collect`); (c) distributed: 2 workers compute disjoint halves via
  `FetchPartition`, merged table equals single-worker.
- **lane**: hermetic + distributed. **depends_on**: U2b, U5a. **size**: M.

## U5b — Coordinator; `Peer` collective; attempt fence; chaos (PR-C commit 4)

- **files_in_scope** (ai-core): `fine_tune/collective/peer.rs`, `fine_tune/worker.rs`
  (coordinator: peer resolution, id mint, dispatch, watchdog, attempt abort),
  `tests/distributed/{harness.rs (peer TOML, `[training] peers`), gang_deterministic.rs,
  gang_chaos.rs}`, `.github/workflows/distributed.yml` (three new test names). (wire-server)
  `grpc/gang.rs` peer-side rank runner and attempt fence.
- **invariants_to_preserve**: K4 real (W=1 via the gang path == embedded bytes), B4, K2, B6.
- **acceptance**: (a) server it-suite: W=1 coordinator dispatching to itself → bytes identical
  to the in-process trainer (RED at base); (b) distributed deterministic leg: 2 processes, W=2,
  `Peer` → bytes identical to the single-process W=2 `Local` run (RED at base); (c) chaos:
  SIGKILL a peer → requeued with `attempts+1`, completed by a new gang from the checkpoint,
  exactly one model, no orphan prefix promoted; SIGKILL the coordinator → same via lease; split
  brain: attempt N+1 dispatched while N is live → N aborted, N+1 completes (RED at base); (d)
  cluster leg: 2 pods × 2 GPUs, W=4, `Nccl` `from_rank` → digest pair + deltas; artifact as
  PR-C commit 5.
- **lane**: hermetic + distributed (dispatched manually; deterministic leg green before merge;
  chaos advisory as today) + gpu-gang cluster leg. **depends_on**: U4b, U5a. **size**: XL.

## U8 — Ballista scheduler/executor roles + extension codec (PR-D commit 1; mandatory, last)

- **files_in_scope**: (wire-server) `crates/jammi-server/Cargo.toml` (`ballista` feature:
  `ballista-core`/`-scheduler`/`-executor` 54.x, `datafusion-proto 54`), `runtime.rs` (roles
  `scheduler`/`executor` via `[server] services` and `ServiceTier`), `ballista/{codec.rs,
  roles.rs}` (codec for `InferenceExec`, `AnnSearchExec`, `AsofJoinExec`, `GangExec` ↔ U5
  descriptor messages; task retry attempts = 0 for gang jobs), `tests/it/ballista_codec.rs`.
  (ai-core) `operator/gang_exec.rs` (single-partition operator whose `execute` runs the U5b
  coordinator), `tests/distributed/{harness.rs (scheduler/executor TOML), ballista_parity.rs}`.
  (docs-ci) `ci.yml` gated-surface clippy for `ballista`.
- **invariants_to_preserve**: B4, K4, B6, K6, B1.
- **acceptance**: hermetic: codec round-trip for every operator (RED at base). Distributed
  (three processes, one binary): (a) an embedding job via `submit_physical_plan` across two
  executors → bytes identical to U6's peer path (RED at base); (b) a W=2 gang job through the
  scheduler → bytes identical to U5b's, never task-retried (RED at base); (c) killing an
  executor mid-gang fails the job and it requeues through jammi's lease path.
- **lane**: hermetic codec arm + distributed three-process arm. **depends_on**: U1, U5b, U6,
  S2. **size**: L.

## U9 — Docs (PR-D commit 2)

- **files_in_scope** (docs-ci / doc-updater): `docs/guide/src/{philosophy.md,
  reference-topologies.md, fine-tune pages}`, `docs/maintainer/MAINTAINER-GUIDE.md`
  (ProducingDescriptor enumeration — `check_doc_parity.py`), `CHANGELOG.md`, `deploy/` note.
- **acceptance**: `check_doc_parity.py` green with the two new variants; docs gates.
- **lane**: docs. **depends_on**: all. **size**: S.
