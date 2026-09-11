# UNITS — per-unit phase-2 contracts (#500), v4 (on the jobs fleet, `main` @ 4ecc0230 = PR #501; `wt-C:` citations hold at the same lines)

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
- **acceptance**: the RED-at-base oracle is the `DATAFUSION_VERSION` test (the only criterion
  RED at base by construction); the new db-features clippy step is a **coverage lane** (the
  0.10.1 pin compiles today — record the base-commit run of the exact command in the ledger);
  workspace gate green including that lane; cookbook 6.5 zero divergence; `cargo tree -d`
  shows one `arrow` and one `datafusion` line.
- **lane**: hermetic + cookbook; `distributed.yml` dispatched manually before merge.
- **depends_on**: S3 (sizes it). **size**: L if S3 compiles with local fixes only; XL otherwise.

## U7a — `gpu-gang.yml` pod leg (PR-B commit 1)

- **files_in_scope** (docs-ci): `.github/workflows/gpu-gang.yml` (label `run-gang`, nightly,
  manual; never `push`/`workflow_call`), `ci/scripts/runpod_lib.sh` (`gpuCount` becomes a
  parameter of the shared deploy payload, default 1 — three-lane blast radius: gpu-prove,
  gpu-perf-ab, gpu-dev), `ci/scripts/runpod_gpu_gang.sh` (pod leg: 1 pod × 2 GPU
  A100-SXM4-80GB), `ci/scripts/execution_surface_reachability_allowlist.txt` (every `cargo`
  tuple in the new script, with residual notes — the registry is derived from every tracked
  `ci/scripts` file), `ci/scripts/check_cuda_run_artifacts.py` (a `gang` artifact kind: world,
  collective, per-rank device, digest pair, measured per-step loss delta, and the
  **pre-registered** ε with its derivation), `docs/maintainer/dev-gpu.md`.
- **invariants_to_preserve**: `check_gpu_prove_once.py` P1 rules; `check_ci_guard_wiring.py`; B2.
- **acceptance**: the gang artifact-schema test (RED at base); `check_gpu_prove_once.py`
  applied to the new workflow's `on:` block; `check_ci_guard_wiring.py` and
  `check_execution_surface_reachability.py` green with the new tuples allowlisted; every
  existing lane still deploys `gpuCount: 1`. Provisioning proof is spike S4, not an acceptance.
- **lane**: gate scripts. **depends_on**: S4. **size**: L.
- **cost ceiling** (human-approved before first run): ≤ 1 h × 2 GPU × $1.59 ≈ $3.2 per run.

## U2a — `TrainingSet` producer (PR-B commit 2)

- **files_in_scope**: (db) `store/manifest.rs` (`ProducingDescriptor::TrainingSet` with
  `format` as a canonical string), `catalog/result_repo.rs` (`ResultTableKind::TrainingSet`),
  tests. (wire-server) `crates/jammi-wire/proto/jammi/v1/embedding.proto` (`TRAINING_SET = 4`,
  append-only) and `crates/jammi-wire/src/embedding.rs:95-116` (the two exhaustive
  `ResultTableKind` mirrors). (ai-core) `fine_tune/worker.rs::run_spec` (materialize-or-reuse,
  then read the table back through `session.sql` over the registered `jammi.{name}` result
  table — `store/mod.rs:855`, `session.rs:944` precedent — with the canonical `ORDER BY`
  re-applied, into today's loader: a compiling intermediate), `fine_tune/graph_sampler.rs`
  (pairs → table), `pipeline/recompute.rs` (arm = re-materialize). (docs-ci) the
  `PRODUCING-DESCRIPTOR-VARIANTS` block of `docs/maintainer/MAINTAINER-GUIDE.md`
  (`check_doc_parity.py` runs on every PR with no path filter; co-owned with U3, U9). The
  cookbook fixture golden.
- **invariants_to_preserve**: K1, K7 (descriptor: source anchors, columns, task, format,
  order rule — no topology, no split), K2 (empty training set refused), B5 (wire mirror),
  B1/B2 naming, B6, doc parity.
- **acceptance**: (a) `fine_tune` creates a `TrainingSet` result table with a definition hash
  and attestation (RED at base); (b) two jobs over the same source/columns/task reuse one table
  (RED at base); (c) refactor parity: adapter bytes identical to base on every cookbook
  fine-tune fixture; (d) order: the read-back re-applies the canonical `ORDER BY` and matches
  the committed order on a fixture with > 1 row group at `execution_threads > 1` (RED at base
  for an unordered scan; `session.rs:189` sets `target_partitions`).
- **lane**: hermetic + cookbook. **depends_on**: U1. **size**: M.

## U4a — `Collective` trait; device-plural session; `CacheKey`; config refusals (PR-B commit 3)

- **files_in_scope**: (ai-core) new `fine_tune/collective/{mod.rs, noop.rs, local.rs, nccl.rs}`,
  `model/cache.rs` (`CacheKey { model_id, device, task: Option, backend: Option }` — shared
  shape with plan 65, recorded in both ledgers), `model/backend/mod.rs` (`DeviceConfig` plural),
  `concurrency/gpu_scheduler.rs` (per device), `session.rs`, `fine_tune/spec.rs`
  (`TrainingCommon.world_size`, `#[serde(default)]` = 1 — the field lands here so the
  refusals below are testable at this commit; queued specs still deserialize), every
  `TrainingCommon { .. }` construction site (`wire/training.rs:176`, `session.rs:1172`,
  `:1300`, the `tests/it` sites), `jammi-ai/Cargo.toml` (`cuda` adds `candle-core/nccl`).
  (wire-server, co-owner) `proto/jammi/v1/training.proto` + `crates/jammi-wire/src/training.rs`
  (the per-job `world_size` field, append-only). (db) `config/mod.rs` (`[gpu] devices`; `[worker] world_size`, `collective`), tests. Test targets: hermetic
  tests in the crate's unit tests; the `Nccl` smoke in the existing `gpu_capability` target.
- **invariants_to_preserve**: B4 (topology is configuration), K2 (`world_size > devices`,
  `nccl` without CUDA, `world_size > 1` with `cached == true` or `hard_negatives.mine == true`
  refused with typed errors at the submit edge), K4 (remote parity suite unchanged), B6.
- **acceptance** (hermetic): (a) `Local` over 2 ranks: `all_gather` layout and rank-ordered
  `all_reduce_sum` are deterministic and equal a serial reference bit-for-bit (RED at base);
  (b) each refusal above (RED at base); (c) two devices in one session hold two cache entries
  for one model id (RED at base: keyed by id alone). (pod leg smoke) `Nccl` over 2 devices
  reduces a known vector.
- **lane**: hermetic (+ pod leg smoke). **depends_on**: S1. **size**: L. No loader contact;
  U2b takes `world` as a partition-rule argument fed from this field (U4a → U2b).

## U2b — Streaming loader; partition rule; scaler; whole-set arms (PR-B commit 4)

- **files_in_scope** (ai-core): `fine_tune/data.rs` (stream + per-batch converters for every
  format; the tests-only `Precomputed` arm unchanged), `fine_tune/trainer.rs` (epoch loop over
  the stream; `batches_per_epoch = ceil(train_count / (W·B))` and every step quantity indexed
  by global batch, at W=1), `fine_tune/worker.rs::run_spec` (`PartitionSpec { rank, world,
  batch, rule }`), `fine_tune/regression_loss.rs` (scaler from ONE collected `Vec<f32>` of the
  train prefix, `from_targets` once),
  `fine_tune/hard_negative_miner.rs` and `fine_tune/gradcache.rs` (stream-sourced, W=1-only),
  `fine_tune/batch_bucket.rs` (rung pinning option), tests. (db) `store/mod.rs` reader slicing.
- **invariants_to_preserve**: K3 (scaler over the train prefix, bit-identical), K2, B6.
- **acceptance**: (a) resident-row high-water mark ≤ `batch × prefetch` (+ the named 4-byte
  per row scaler exemption) on a fixture larger than the bound, on every non-whole-set arm
  (RED at base); (b) partition rule: for W ∈ {1,2,4} the multiset of rows over ranks at each
  global step equals the W=1 batch, on a fixture whose `train_count` is not a multiple of W·B
  (RED at base); (c) refactor parity holds on every cookbook fixture including regression;
  (d) mining/GradCache runs at W=1 produce bytes identical to base; (e) order: the streamed
  row order equals the committed materialization order for `target_partitions ∈ {1, N}` from
  the row-group reader, with no blocking sort (RED at base).
- **lane**: hermetic + cookbook. **depends_on**: U2a, U4a (the `world` argument). **size**: XL.

## U3 — `FineTune` producer; `model_materialization` migration; cache reuse (PR-B commit 5, concurrent with U2b)

- **files_in_scope**: (db) `store/manifest.rs` (`ProducingDescriptor::FineTune`;
  `MaterializationEnv` kernel-profile), `catalog/{schema.rs, migrations.rs}` (029, nullable
  columns), `catalog/model_repo.rs` (`probe_model_by_definition`, NULL never matches),
  `store/artifact.rs` (manifest last), `store/reconcile.rs` (prefix reaped only when
  unreferenced), `tests/it/migrations.rs`. (ai-core) `fine_tune/worker.rs::publish_and_finalize`
  (materialization; probe before training; own name → reused prefix), `pipeline/recompute.rs`
  (arm = retrain), `model/resolver.rs` (manifest on `ModelRecord`), the canonical-encoding
  producer and the exhaustive-destructuring completeness test in `jammi-ai` (and `jammi-wire`
  for `FineTuneConfig`). (docs-ci) the `PRODUCING-DESCRIPTOR-VARIANTS` guide block (co-owned).
- **invariants_to_preserve**: K5, K7 (exhaustive destructuring of `FineTuneConfig` and
  `TrainingCommon`, fields existing at this commit; the descriptor holds an opaque versioned
  canonical encoding, never a foreign type), K1, B6, B1 (no `register_*`), doc parity.
- **acceptance**: (a) same spec on the same training-set digest with `CachePolicy::Use` trains
  once, two model rows share one prefix, deleting one leaves the prefix (RED at base); (b) the
  exhaustive-destructuring completeness test (RED at base); (c) 029 append-only test.
- **lane**: hermetic. **depends_on**: U2a. **size**: L (migration, NULL-probe semantics,
  reference-counted reaping, artifact ordering, ten files across two crates). Co-ownership:
  `manifest.rs` and `recompute.rs` with U2a/U4b.

## U4b — Rank context; gather rule; lockstep; single-node gang (PR-B commit 6)

- **files_in_scope** (ai-core): `fine_tune/trainer.rs` (`RankContext`; per-arm gather points
  — encoder outputs / classification **logits** / regression head output; identical global
  loss; local-slot gather backward; canonical-order reduce; lockstep flags; rank-0 checkpoint
  with per-rank `dropout_positions` gathered), `fine_tune/resume.rs` (per-rank positions,
  schema-version bump), `fine_tune/target.rs` (per-rank dropout seed derivation),
  `fine_tune/optimizer.rs` (zero-filled reduce set), `fine_tune/worker.rs::run_spec` (spawn W
  local ranks), tests. (db, co-owned with U2a/U3) `store/manifest.rs` (topology fields;
  extends U3's completeness test). Test targets: hermetic tests in the crate's unit tests;
  pod-leg tests in `gpu_capability`.
- **invariants_to_preserve**: B4, K7, K2, K4, B6.
- **acceptance** (hermetic, `Local`): (a) W=2 twice → identical bytes, including across a
  resume (kill at epoch k, resume, compare to uninterrupted) (RED at base); (b) gather
  exactness for CoSENT, AnglE, MNRL, classification and quantile regression — global loss AND
  summed adapter gradient bit-for-bit vs W=1 on the same rows, on a fixture whose
  `train_count` is not a multiple of W·B (RED at base); (c) W=2 × B vs W=1 × 2B within
  pre-registered ε at `lora_dropout=0`, rung pinned (RED at base); (d) lockstep: forced
  divergence on one rank; a Var absent from one rank's `GradStore`; a zero-row rank — the gang
  completes (RED at base); (e) W=1 via `Noop` byte-identical to U2b's golden. (pod leg,
  `Nccl`, 2×A100): (a) as a digest pair + per-step delta against the pre-registered ε, (c)
  with GPU ε; artifact committed as PR-B commit 7.
- **lane**: hermetic + gpu-gang pod leg. **depends_on**: U2b, U3, U4a, S1. **size**: XL (the
  plan's mathematical core; five hermetic oracles).

## U7b — cluster leg + cluster reap (PR-C commit 1)

- **files_in_scope** (docs-ci): `ci/scripts/runpod_lib.sh` (cluster create/teardown primitive
  with deadline), `ci/scripts/runpod_gpu_gang.sh` (cluster leg: TRAINING cluster, 2 pods × 2
  GPUs), `ci/scripts/execution_surface_reachability_allowlist.txt` (the cluster-leg tuples;
  U7a's rows re-verified if their command lines change), `.github/workflows/gpu-reap.yml`
  (clusters enumerated and reaped), `gpu-gang.yml`.
- **acceptance**: reap enumerates clusters (RED at base: pods only); P1 rules; guard wiring;
  `check_execution_surface_reachability.py` green with the cluster-leg tuples allowlisted.
- **lane**: gate scripts. **depends_on**: U7a, S4. **size**: M.
- **cost ceiling**: ≤ 1 h × 4 GPU × $1.59 ≈ $6.4 per run; label-only until a flake-free streak.

## U5a — `GangService` on `peer_bind`; I-GANG authorization (PR-C commit 2)

- **files_in_scope**: (wire-server) `crates/jammi-wire/proto/jammi/v1/gang.proto`
  (`RunRank(RankAssignment) returns (stream RankEvent)` with `RankEvent::Released`;
  `FetchPartition(PartitionRequest) returns (stream ArrowIpc)`), `crates/jammi-wire/build.rs:22-33`,
  `crates/jammi-wire/src/{lib.rs, gang.rs}`, `crates/jammi-server/src/grpc/gang.rs` (handler on
  the **peer listener** — 68 DIST D7's routes built outside `assemble_grpc_chain`; the I-GANG
  verification through `get_job_for_rank`, `status = 'running'`, `claimed_by`, live lease, tenant
  derived from the row and pinned; peer addresses resolved from `instances.peer_addr` by
  instance id; `FetchPartition` belongs-to-job check; handler order fence-then-slot; #485
  bounds), (db) `crates/jammi-db/src/catalog/jobs_repo.rs` (`get_job_for_rank(job_id)`: primary
  key, no tenant predicate, never admin scope; `list_gang_members(kind)`), `crates/jammi-server/src/runtime.rs` (mount on the peer
  routes), `crates/jammi-server/tests/it/{api_freeze_baseline.txt (RPC + PACKAGE lines),
  api_freeze.rs (package count prose), tenant_isolation_oracle.rs (`GANG_LISTENER_ALLOWLIST`,
  unioned into `covered_on_wire` and the partition assertion; public-listener `UNIMPLEMENTED`
  probe), gang_partition.rs, gang_authz.rs}`. (ai-core) `fine_tune/worker.rs` (`JobSlot`: taken before
  `claim_next` (`wt-C: worker.rs:346`), held across `run_claimed_job` (`:355`), released before
  the idle sleep (`:363`)). (db) `config/mod.rs` (`rank_timeout_secs`
  if not already in U4a).
- **invariants_to_preserve**: B5 (I-GANG written invariant; no double binder — never mounted under
  `TenantResolverLayer`), K2, B1 (no `stage`/`register` stems), B6, OPS D6 (no abort while a claim
  may be in flight — the slot is taken outside the transaction).
- **acceptance**: (a) the streamed bytes of partition r equal the local read of partition r (RED
  at base); (b) `RunRank` for a job not running / not claimed by the named instance / lease
  expired / wrong `FetchPartition` table is refused with a typed status (RED at base); (c) fence
  before slot: a greater attempt for the job whose stale runner holds the slot aborts it and
  takes the slot; lesser-or-equal refused (RED at base); (d) a peer busy with its *own* claimed
  job refuses with `Unavailable`; an idle peer accepts; the claim loop never claims while a rank
  runs (RED at base); (e) `api_freeze` and `every_rpc_is_covered` green with the new lines; the
  public listener answers `UNIMPLEMENTED` for `GangService/*`; `get_job_for_rank` is unreachable
  from any public RPC and ignores any caller tenant (invariant oracles); (f) an assignment naming
  an instance id that is not a fresh member is refused (RED at base).
- **lane**: hermetic + server it-suite. **depends_on**: U4a, **68 DIST unit 1** (the
  `peer_bind` listener; if unmerged, its listener commit is carried verbatim as this unit's first
  commit, co-owned). **size**: L.

## U6 — Partition-aware inference operator; distributed frozen forward (PR-C commit 3)

- **files_in_scope** (ai-core): `operator/inference_exec.rs` (inherit input partitioning),
  `operator/runner.rs` (one load per process), `pipeline/embedding.rs` (stream into the sink;
  fan-out through `FetchPartition`), `fine_tune/worker.rs::run_fine_tune_blocking` head-target
  arm (`worker.rs:2375`), tests. (db) `store/result_sink.rs` (partition-ordered append).
- **invariants_to_preserve**: K4 (4-partition table == 1-partition table, bytes), K1, B3, B6.
- **acceptance**: (a) byte parity across partition counts (RED at base); (b) resident batches
  ≤ a bound (RED at base: `collect`). The two-worker `FetchPartition` leg is U5b's acceptance
  (e), because the harness, `tests/distributed/main.rs` and the `distributed.yml` matrix are
  U5b's.
- **lane**: hermetic. **depends_on**: U2b, U5a. **size**: L (operator repartitioning, peer
  fan-out client, ordered sink).

## U5b-1 — Coordinator; `Peer` collective; membership substrate; determinism (PR-C commit 4)

- **files_in_scope** (ai-core): `fine_tune/collective/peer.rs`, `fine_tune/worker.rs` (coordinator
  on the `JobWorker`: members via `list_gang_members`, id mint, dispatch by instance id),
  `session.rs` (`instances.peer_addr` write site, `wt-C: session.rs:247-251`),
  `tests/distributed/{main.rs, harness.rs (peer TOML with `peer_bind`/`peer_advertise`),
  gang_deterministic.rs, gang_forward.rs}`, `.github/workflows/distributed.yml` (test names).
  (db) `config/mod.rs` (`[server] peer_advertise`; validate `peer_advertise ⇒ peer_bind ⇒
  storage.result_root`), `catalog/{schema.rs, migrations.rs}` (`instances_peer_addr`, number at
  rebase, both pin sites), `catalog/jobs_repo.rs` (`upsert_instance` gains `peer_addr`;
  `list_gang_members(kind)`: `workers ⋈ instances`, kinds split on `,` in Rust, `peer_addr` set,
  `last_seen_at` within `[lease] duration_secs`, sorted in Rust). (docs-ci) configuration.md for
  `peer_advertise`. This is the substrate 68 DIST §5.8 sketches; DIST's `RendezvousPlacement`
  builds on it later (recorded in `68/RECONCILIATION-WITH-67.md`).
- **invariants_to_preserve**: K4 real, B4, K2 (validate chain), K5, B6.
- **acceptance**: (a) K4 real: 2 processes, W=2, `Peer` → bytes identical to the single-process
  W=2 `Local` run (RED at base; rank 0 is always in-process); (b) `list_gang_members` excludes a
  stale instance, an instance without `peer_addr`, and a worker whose kinds contain only
  `graph_fine_tune` when `fine_tune` is asked (RED at base); (c) `peer_advertise` without
  `peer_bind` or without `result_root` is refused at load (RED at base); (d) two workers compute
  disjoint halves of a head-target feature table via `FetchPartition` and the merged table
  equals the single-worker table (U6's operator; RED at base). Test target: `distributed`.
- **lane**: hermetic + distributed (dispatched manually; deterministic leg green before merge).
  **depends_on**: U4b, U5a, U6, S1. **size**: L.

## U5b-2 — Watchdog; abort with no terminal write; released-vs-failed; chaos (PR-C commit 5)

- **files_in_scope** (ai-core): `fine_tune/worker.rs` (watchdog; attempt abort by flipping the
  hold's `lost` flag so the run exits through the leave-for-reclaim arm, `wt-C: worker.rs:670-676`;
  `Released` → `release_job_lease` first, then the flag; `BackOff` on a live same-named
  `building` training-set row), `tests/distributed/gang_chaos.rs`. (wire-server) `grpc/gang.rs`
  (the drain hook that emits `RankEvent::Released`).
- **invariants_to_preserve**: OPS D6 (no abort while a claim may be in flight), OPS D10 (a
  peer-tier rolling restart costs zero net attempts), B6.
- **acceptance**: (a) after a rank failure the row is `running` (no terminal write) until reclaim
  requeues it, then `attempts+1` at the successor's claim (RED at base); (b) chaos: SIGKILL a
  peer → requeued within the lease window, completed by a new gang from the checkpoint, one
  model, no orphan prefix; SIGKILL the coordinator → same via lease; split brain → older attempt
  aborted; SIGTERM (drain) on a peer host → `Released`, `releases+1`, net attempts unchanged, job
  completes; a crashed coordinator's live `building` row → successor backs off and reclaims after
  expiry (RED at base); (c) cluster leg: 2 pods × 2 GPUs, W=4, `Nccl` `from_rank` → digest pair +
  deltas against the pre-registered ε; artifact as PR-C commit 6. Test targets: `distributed`;
  `gpu_capability` for the cluster leg.
- **lane**: distributed (chaos advisory as today) + gpu-gang cluster leg. **depends_on**: U5b-1,
  **68 OPS merged** (`release_job_lease`, drain hooks, C2's loop shape). **size**: L.

## U8a — `jammi-ballista`: crate, codecs, execution engine, role knobs (PR-D commit 1)

- **files_in_scope**: (wire-server; new crate) `crates/jammi-ballista/{Cargo.toml (ballista-core /
  -scheduler / -executor 54.x, datafusion-proto 54; depends on jammi-ai, jammi-db, jammi-wire;
  publishable, lockstep), src/{lib.rs, codec.rs (`JammiCodec`: `InferenceExec`, `AnnSearchExec`,
  `AsofJoinExec`, `GangExec` ↔ U5 descriptor messages), engine.rs (`JammiExecutionEngine`:
  model cache across plans, device pinned to `[gpu] devices`, shuffle reader rewrite + writer
  wrap), roles.rs (scheduler via `start_server(cluster, …)` with Ballista's in-memory cluster;
  executor via `ExecutorProcessConfig { override_execution_engine, override_*_codec }`;
  `task_max_failures = stage_max_failures = 0`), config.rs (`BallistaConfig { scheduler_bind:
  Option, executor: Option<{scheduler_address, work_dir}> }`)}}`, `Cargo.toml` (workspace member;
  `deny.toml`), `crates/jammi-server/{Cargo.toml (depends on jammi-ballista, no feature),
  src/runtime.rs (host the roles from config)}`, `crates/jammi-db/src/config/mod.rs`
  (`[ballista]` section, `deny_unknown_fields`, listener collision checks). (ai-core)
  `crates/jammi-ai/src/operator/gang_exec.rs` (single-partition operator whose `execute` runs the
  U5b coordinator). `tests/distributed/{main.rs, harness.rs (scheduler/executor TOML),
  ballista_parity.rs}`; `.github/workflows/distributed.yml` matrix entries;
  `ci/scripts/publish_crates.sh:40-50` (`jammi-ballista` inserted before `jammi-server` in the
  topological publish list — a `v*` tag would otherwise half-publish). Shuffle stays Ballista's
  local `work_dir` (no object-store shuffle in v1).
- **invariants_to_preserve**: B4 (roles are config; no cargo feature; the library keeps the
  capability through the crate), B2 (dep direction), K4 (bytes through Ballista == bytes through
  the peer path), B6, K6 (publishable, lockstep, in `publish_crates.sh`'s ordered list), B1
  (pre-swept names, README r45).
- **acceptance**: hermetic: codec round-trip for every operator (RED at base); config: `[ballista]`
  parses, unset = no roles, `scheduler_bind == peer_bind/flight_listen/health_listen` refused (RED
  at base). Distributed (three processes, one binary): (a) an embedding job via
  `submit_physical_plan` across two executors → bytes identical to U6's peer path (RED at base);
  (b) a W=2 gang job through the scheduler → bytes identical to U5b's, never task-retried (RED at
  base); (c) killing an executor mid-gang fails the job and requeues it through jammi's lease path.
- **lane**: hermetic + distributed. **depends_on**: U1, U5b, U6, S6. **size**: L.

## U8b — Catalog-backed cluster state; device-aware placement (PR-D commit 2; the completion gate)

- **files_in_scope**: (wire-server) `crates/jammi-ballista/src/{cluster.rs (`CatalogClusterState`,
  `CatalogJobState` over jammi-db's public backend — the Ballista-shaped repo lives here, not in
  jammi-db), placement.rs (`DevicePlacement`: executor id ↔ `workers.devices`; a task is GPU-bound
  iff its stage plan — read from `active_jobs`' execution graph and decoded through `JammiCodec` —
  contains a `GangExec` or an `InferenceExec` whose descriptor names a CUDA device; installed
  through `bind_schedulable_tasks` / `TaskDistributionPolicy::Custom`)}`. (db)
  `catalog/{schema.rs, migrations.rs}` (`compute_cluster_state`: distributor-neutral tables —
  `compute_executors(executor_id, instance_id, heartbeat_at, slots)`, `compute_jobs(job_id,
  graph, status)` — plus `workers.devices TEXT` JSON `[{kind, ordinal, memory}]` written by
  `upsert_worker` from the session's device list, `wt-C: jobs_repo.rs:1556`; number at rebase,
  both pin sites), `catalog/jobs_repo.rs` (`WorkerRecord.devices`), `catalog/compute_repo.rs`
  (new, generic CRUD). Tests in `tests/distributed/ballista_state.rs`.
- **invariants_to_preserve**: K5 (neutral names; append-only), B1/B2 (no distributor vocabulary
  in the engine's catalog), B4, K4, B6, the actuator-rule disposition (README r42), OPS D6.
- **acceptance**: (a) restart the scheduler process: registered executors and in-flight job state
  survive (RED at base: in-memory); (b) two schedulers over one catalog serve one cluster (RED at
  base); (c) a GPU-bound task never binds to an executor whose `devices` is empty; a device-less
  cluster refuses it with a typed error (RED at base); (d) `list_workers` returns `devices` as
  registered (RED at base); (e) the gang job of U8a (b) still byte-matches U5b-1 under catalog
  state.
- **lane**: distributed. **depends_on**: U8a, S6 (restart, two-scheduler and executor-identity
  probes recorded). **size**: L.

## U9a — Docs (PR-D commit 3)

- **files_in_scope** (docs-ci / doc-updater): `docs/guide/src/{philosophy.md,
  reference-topologies.md, configuration.md ([worker]/[ballista]/peer_advertise knobs), fine-tune
  pages}`, `docs/maintainer/MAINTAINER-GUIDE.md` (prose; the `PRODUCING-DESCRIPTOR-VARIANTS` block
  lands with U2a and U3), `CHANGELOG.md`.
- **acceptance**: docs gates green; reference-topologies states the StatefulSet consequence.
- **lane**: docs. **depends_on**: all. **size**: M.

## U9b — shape-d overlay (PR-D commit 4)

- **files_in_scope** (docs-ci): `deploy/kubernetes/overlays/shape-d/**` (68 K's compute
  Deployment becomes a StatefulSet with a headless service and `nvidia.com/gpu: N`, keeping OPS
  C6's `terminationGracePeriodSeconds` observable green; K's own header invites this edit),
  `deploy/docker-compose*.yml` note, the K README's `issues/500` provisional note (replaced, not
  duplicated).
- **acceptance**: kubeconform strict + kind smoke on the amended overlay; K3/K4b guards green;
  the `issues/500` note count as K's oracle expects after replacement.
- **lane**: kubeconform + kind smoke. **depends_on**: 68 K merged, 68 OPS merged. **size**: M.
