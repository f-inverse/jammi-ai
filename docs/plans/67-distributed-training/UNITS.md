# UNITS — per-unit phase-2 contracts (#500)

Each unit: `files_in_scope` (write-owner in parentheses), `invariants_to_preserve` (constitution
IDs), `acceptance` (the feature RED oracle: RED at the base commit, GREEN on the branch, asserts
the criterion not an implementation detail), `lane`, `depends_on`, `size` (S/M/L/XL by blast
radius, not calendar). Full CI gate for every commit: the verbatim step list of `ci.yml`
(hermetic + clippy gated-surface matrix) with per-step `$?`, never a subset. Naming per
README ruling 19.

---

## U1 — DataFusion 54 line upgrade (PR-A, alone)

- **files_in_scope** (docs-ci for the shared manifests; db, ai-core, wire-server, python, cli,
  bench for compile fixes in their crates): `Cargo.toml` (workspace pins: `datafusion 54`,
  `arrow`/`arrow-array`/`arrow-schema`/`arrow-ipc`/`arrow-flight`/`parquet 58`, `object_store
  0.13`, `datafusion-federation 0.5.5`, `datafusion-flight-sql-server 0.4.18`,
  `datafusion-table-providers 0.13`, `pyo3-arrow` to the arrow-58 line), `Cargo.lock`, every
  `crates/*/src/**` site the API bump breaks, `deny.toml` if licenses move.
- **invariants_to_preserve**: B6 (one PR, workspace-atomic), K6 (lockstep version bump), K4
  (`crates/jammi-server/tests/it/grpc_remote_session.rs` stays green), B5
  (`TenantScopeAnalyzerRule` semantics unchanged under the new analyzer API), K1 (replay arms
  compile unchanged).
- **acceptance**: a test asserting `datafusion::DATAFUSION_VERSION` begins with `54` (RED at
  base); cookbook 6.5 re-emit shows zero golden divergence; the gated-surface clippy matrix is
  green; `cargo tree -d` shows one `arrow` and one `datafusion` line.
- **lane**: hermetic + cookbook; `distributed.yml` dispatched manually once before merge.
- **depends_on**: —. **size**: L (touches every crate; mechanical).
- **risk**: `pyo3-arrow` may lag arrow 58 — if so, pin the newest compatible and record it in
  the unit's ledger; `datafusion-federation` analyzer trait changes at 0.5.x.

## U2 — Training set as a producer; streaming loader; partition rule (PR-B commit 1)

- **files_in_scope** (db): `crates/jammi-db/src/store/manifest.rs` (`ProducingDescriptor::
  TrainingSet`, `canonical_bytes`), `crates/jammi-db/src/catalog/result_repo.rs`
  (`ResultTableKind::TrainingSet`), `crates/jammi-db/src/store/mod.rs` (writer row-group
  sizing), tests. (ai-core): `crates/jammi-ai/src/fine_tune/{data.rs, worker.rs, trainer.rs,
  graph_sampler.rs}` (materialize → stream; per-batch converters; `PartitionSpec {rank, world,
  batch, rule}`; row-count-driven `batches_per_epoch`), `crates/jammi-ai/src/pipeline/recompute.rs`
  (replay arm: re-materialize), `crates/jammi-ai/src/fine_tune/regression_loss.rs`
  (`TargetScaler` from the manifest), tests + the cookbook fixture golden.
- **invariants_to_preserve**: K1 (new variant → replay arm), K7 (descriptor completeness:
  source anchors, columns, task, format, order rule), K3 (scaler persisted in data space), K2
  (typed refusal on an empty training set, a partition with zero rows, `world > rows/batch`),
  B1/B2 (naming), B6.
- **acceptance**: (a) `fine_tune` on the cookbook fixture creates a `TrainingSet` result table
  with a definition hash and attestation (RED at base: no such kind exists); (b) the trainer's
  resident-row high-water mark during an epoch ≤ `batch × prefetch` on a fixture larger than
  that bound (RED at base: the whole set is resident); (c) refactor parity — adapter bytes
  identical to the base commit's on the fixture (cookbook golden); (d) partition-rule test: for
  W ∈ {1,2,4}, the multiset of rows over ranks at each global step t equals the W=1 batch t.
- **lane**: hermetic + cookbook. **depends_on**: U1. **size**: XL (every head constructor).

## U3 — Model as a producer; migration 029; cache reuse (PR-B commit 2, concurrent with U4)

- **files_in_scope** (db): `crates/jammi-db/src/store/manifest.rs` (`ProducingDescriptor::
  FineTune`, `MaterializationEnv` kernel-profile field), `crates/jammi-db/src/catalog/{schema.rs,
  migrations.rs}` (029 `model_materialization`), `crates/jammi-db/src/catalog/model_repo.rs`
  (columns; `probe_model_by_definition`), `crates/jammi-db/src/store/artifact.rs` (manifest
  written last), `crates/jammi-db/tests/it/migrations.rs`. (ai-core): `crates/jammi-ai/src/fine_tune/
  worker.rs` (`publish_and_finalize` writes the materialization; `CachePolicy::Use` probe before
  training), `crates/jammi-ai/src/pipeline/recompute.rs` (arm = retrain), `crates/jammi-ai/src/
  model/resolver.rs` (unchanged handle; manifest surfaced on `ModelRecord`).
- **invariants_to_preserve**: K5 (029 appended, names unique), K7 (the DESIGN §3 table is a
  test: every field flips the hash), K1, B6, B1 (no `register_*` pub items).
- **acceptance**: (a) submitting the same spec twice against the same training-set digest with
  `CachePolicy::Use` runs the trainer once and returns the existing model (RED at base: no
  probe exists); (b) per-field completeness test over the DESIGN §3 table (RED at base: no
  descriptor); (c) `migrations.rs` append-only test admits 029.
- **lane**: hermetic. **depends_on**: U2 (training-set digest is an anchor). **size**: M.

## U4 — `Collective` trait; device-plural session; single-node gang (PR-B commit 3)

- **files_in_scope** (ai-core): new `crates/jammi-ai/src/fine_tune/collective/{mod.rs, noop.rs,
  local.rs, nccl.rs}`, `crates/jammi-ai/src/fine_tune/{trainer.rs (RankContext, reduce at step
  boundary, rank-0-only checkpoint/publish), optimizer.rs (count-weighted mean), worker.rs
  (spawn W local ranks), spec.rs (`TrainingCommon.world_size`)}`, `crates/jammi-ai/src/model/
  {cache.rs (key gains device), backend/mod.rs (DeviceConfig plural)}`, `crates/jammi-ai/src/
  concurrency/gpu_scheduler.rs` (per-device), `crates/jammi-ai/src/session.rs` (device-plural),
  `crates/jammi-ai/Cargo.toml` (`cuda` feature adds `candle-core/nccl`, `candle-nn` unchanged).
  (db): `crates/jammi-db/src/config/mod.rs` (`[gpu] devices`, `[training] world_size /
  collective`), config tests. (numerics, read-only consult): reduction-order determinism note.
- **invariants_to_preserve**: B4 (one trainer, no cfg fork; W=1 through `Noop`), K7 (world size,
  rule version, collective backend in the hash — lands with U3), K2 (world > devices refused;
  `nccl` requested without CUDA refused with a typed error), K4 (unchanged remote parity suite),
  B6.
- **acceptance** (hermetic, CPU, `Local`): (a) W=2 twice → identical adapter bytes (RED at base:
  no W); (b) W=2 × B versus W=1 × 2B → per-step loss within ε on the fixture, ε pre-registered
  from the fixture's measured fp spread (RED at base); (c) W=1 through `Noop` byte-identical to
  U2's golden. (gpu-gang pod leg, `Nccl`, 2×A100): (a) and (b) again, plus a committed artifact
  passing `check_cuda_run_artifacts.py`.
- **lane**: hermetic + gpu-gang pod leg (U7a). **depends_on**: U2 (partition rule), S1.
  **size**: L.

## U5 — `GangService`; coordinator; multi-node gang; chaos (PR-C commit 1)

- **files_in_scope** (wire-server): `crates/jammi-wire/proto/jammi/v1/gang.proto`
  (`GangService { RunRank(RankAssignment) returns (stream RankEvent); FetchPartition(PartitionRequest)
  returns (stream ArrowIpc); }`), `crates/jammi-wire/src/{lib.rs, gang.rs}` (typed conversions),
  `crates/jammi-server/src/grpc/gang.rs` (service impl; tenant-scoped mount in
  `crates/jammi-server/src/runtime.rs` via `mount_tenant_scoped`; request bounds per #485),
  `crates/jammi-server/tests/it/gang_remote_parity.rs`. (ai-core): `crates/jammi-ai/src/fine_tune/
  collective/peer.rs`, `crates/jammi-ai/src/fine_tune/{worker.rs (coordinator: peer resolution,
  NCCL id mint, dispatch, watchdog, attempt abort), trainer.rs (heartbeat hook)}`, `crates/jammi-ai/
  tests/distributed/{harness.rs (peer TOML, `[training] peers`), gang_deterministic.rs,
  gang_chaos.rs}`, `.github/workflows/distributed.yml` (two new test names in the matrix).
  (db): `crates/jammi-db/src/config/mod.rs` (`peers`, `rank_timeout_secs`).
- **invariants_to_preserve**: K4 (real: W=1 via the gang path == embedded bytes), B5
  (tenant-scoped mount; a peer serves only the job's tenant), B4 (the library can run a gang
  in-process with `Local`; the server adds a transport, not a capability), K2 (typed refusal
  when `peers.len() + 1 < world_size`, on timeout), B1 (no `stage`/`register` stems), B6.
- **acceptance**: (a) server it-suite: W=1 coordinator dispatching to itself produces bytes
  identical to the in-process trainer (RED at base: no service); (b) distributed deterministic
  leg: 2 processes, W=2, `Peer` collective → bytes identical to the single-process W=2 `Local`
  run (RED at base); (c) chaos leg: SIGKILL a peer mid-epoch → job requeued with `attempts+1`,
  completed by a new gang resuming from the checkpoint, `list_models` shows exactly one model,
  no orphan attempt prefix promoted (RED at base); SIGKILL the coordinator → same via lease
  expiry; (d) gpu-gang cluster leg: 2 pods × 2 GPUs, W=4, `Nccl` with `Comm::from_rank` →
  reproducibility (two runs identical) + committed artifact.
- **lane**: hermetic (unit tests) + distributed nightly (deterministic leg required green before
  merge; chaos advisory as today) + gpu-gang cluster leg (U7b). **depends_on**: U4. **size**: XL.

## U6 — Partition-aware inference operator; distributed frozen forward (PR-C commit 2, concurrent with U5)

- **files_in_scope** (ai-core): `crates/jammi-ai/src/operator/inference_exec.rs`
  (`properties()` inherits input partitioning; `benefits_from_input_partitioning`),
  `crates/jammi-ai/src/operator/runner.rs` (model guard per partition, one load per process),
  `crates/jammi-ai/src/pipeline/embedding.rs` (stream into `ResultSink`, no `collect`; optional
  fan-out through `FetchPartition`), `crates/jammi-ai/src/fine_tune/worker.rs` (head target:
  features from the embedding table over the training set), tests incl. a multi-partition
  byte-parity test. (db): `crates/jammi-db/src/store/result_sink.rs` (partition-ordered append).
- **invariants_to_preserve**: K4 (multi-partition table == single-partition table, bytes; row
  order by partition index then row), K1 (no descriptor change), B3 (features consumed by the
  head via the SQL surface, no vector verb), B6.
- **acceptance**: (a) an embedding table produced with 4 input partitions is byte-identical to
  the 1-partition table on the fixture (RED at base: `UnknownPartitioning(1)` makes the 4-partition
  plan degenerate); (b) resident batches during production ≤ a bound (RED at base: `collect`);
  (c) distributed leg: 2 workers each compute a disjoint half via `FetchPartition` and the
  merged table equals the single-worker table.
- **lane**: hermetic + distributed. **depends_on**: U2, U5 (`FetchPartition`). **size**: M.

## U7a / U7b — `gpu-gang.yml` pod leg; cluster leg (PR-B commit 4; PR-C commit 3)

- **files_in_scope** (docs-ci): `.github/workflows/gpu-gang.yml` (label `run-gang`, nightly,
  manual; never `push`/`workflow_call`; concurrency lesson from `gpu-prove.yml`),
  `ci/scripts/runpod_gpu_gang.sh` (pod leg: one pod, `gpuCount 2`, A100-SXM4-80GB; cluster leg:
  `create-cluster` type TRAINING, 2 pods × 2 GPUs, deadline + reap), `ci/scripts/check_cuda_run_
  artifacts.py` (a `gang` artifact kind: world, collective, per-rank device, reproducibility
  digest pair, W-invariance ε), `.github/workflows/gpu-reap.yml` (clusters reaped too),
  `docs/maintainer/dev-gpu.md` (cost and cadence).
- **invariants_to_preserve**: the gate-script rules `check_gpu_prove_once.py` pins (no
  auto-start); K6 untouched; B2 (no consumer names in workflow inputs).
- **acceptance**: workflow `act`-style dry run or `workflow_dispatch` with `--dry-run` proving
  the script provisions, runs, reaps; the artifact schema test RED at base; the first real
  artifacts land with U4 (pod) and U5 (cluster).
- **lane**: workflow dry-run; real runs by label. **depends_on**: U7a → U7b. **size**: M + M.
- **cost ceiling** (stated, human-approved before first run): pod leg ≤ 1 h × 2 GPU ×
  $1.59 ≈ $3.2 per run; cluster leg ≤ 1 h × 4 GPU × $1.59 ≈ $6.4 per run; nightly only after a
  flake-free streak, label-only before.

## U8 — Ballista scheduler/executor roles + extension codec (PR-D commit 1; mandatory, last)

- **files_in_scope** (wire-server): `crates/jammi-server/Cargo.toml` (`ballista` feature:
  `ballista-core`, `ballista-scheduler`, `ballista-executor` 54.x, `datafusion-proto 54`),
  `crates/jammi-server/src/{runtime.rs (roles `scheduler` / `executor` via `[server] services`
  and `ServiceTier`), ballista/{codec.rs, roles.rs}}` (a `PhysicalExtensionCodec` mapping
  `InferenceExec`, `AnnSearchExec`, `AsofJoinExec`, and a `GangExec` wrapper to and from the U5
  descriptor messages; scheduler config with task retry attempts = 0 for gang jobs),
  `crates/jammi-server/tests/it/ballista_parity.rs`. (ai-core): `crates/jammi-ai/src/operator/
  gang_exec.rs` (a single-partition operator whose `execute` runs the U5 coordinator; under
  Ballista the gang stays a jammi mechanism scheduled as one task), tests. (docs-ci):
  `.github/workflows/ci.yml` gated-surface clippy for the `ballista` feature.
- **invariants_to_preserve**: B4 (roles are `[server] services` values in the one binary),
  K4 (bytes through Ballista == bytes through the peer path), B6, K6 (feature-gated deps still
  lockstep), B1 (the codec and roles name no consumer).
- **acceptance**: with the `ballista` feature, the jammi binary runs as one scheduler + two
  executors (three processes, same binary); (a) an embedding job submitted via
  `submit_physical_plan` executes across both executors and yields a table byte-identical to
  U6's peer path (RED at base: no feature, no codec); (b) a W=2 gang job submitted through the
  scheduler completes with bytes identical to U5's and is never retried at task level (RED at
  base); (c) killing an executor mid-gang fails the job (no task retry) and the job requeues
  through jammi's lease path.
- **lane**: hermetic (`ballista` feature in the gated-surface matrix) + distributed.
  **depends_on**: U1, U5, U6, S2. **size**: L.

## U9 — Docs (PR-D commit 2)

- **files_in_scope** (docs-ci / doc-updater): `docs/guide/src/{philosophy.md (deployment shapes:
  a gang is a configuration; no sixth backend), reference-topologies.md (#482 consequence:
  StatefulSet/indexed Job, `peers`), fine-tune pages}`, `docs/maintainer/MAINTAINER-GUIDE.md`
  (ProducingDescriptor enumeration — `check_doc_parity.py` binding), `CHANGELOG.md`, `deploy/`
  overlay note. Docs reflect current state; no journey markers.
- **invariants_to_preserve**: doc parity gate (ProducingDescriptor ⇄ guide), B2.
- **acceptance**: `ci/scripts/check_doc_parity.py` green with the two new variants; docs gates.
- **lane**: docs. **depends_on**: all. **size**: S.
