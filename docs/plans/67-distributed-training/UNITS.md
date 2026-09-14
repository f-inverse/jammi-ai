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
  with `ci/scripts/check_lint_surface_closure.py`. (db) `crates/jammi-db/Cargo.toml::[dependencies].datafusion-table-providers`
  (already 0.13.1) and its compat surface: `PostgresTableFactory`,
  `PostgresConnectionPool` (`crates/jammi-db/src/source/postgres.rs`), `MySQLTableFactory`,
  `MySQLConnectionPool` (`crates/jammi-db/src/source/mysql.rs`); `tenant_scope.rs` analyzer API. (ai-core,
  wire-server, python, cli, bench) compile fixes in their crates.
- **invariants_to_preserve**: B6, K6, K4 (`grpc_remote_session.rs` green), B5, K1.
- **acceptance**: the RED-at-base oracle is the `DATAFUSION_VERSION` test (the only criterion
  RED at base by construction); the new db-features clippy step is a **coverage lane** (the
  current `datafusion-table-providers` pin compiles today — record the base-commit run of the exact command in the ledger);
  workspace gate green including that lane; cookbook 6.5 zero divergence; `cargo tree -d`
  shows one `arrow` and one `datafusion` line.
- **lane**: hermetic + cookbook; `distributed.yml` dispatched manually before merge.
- **depends_on**: S3 (sizes it). **size**: L if S3 compiles with local fixes only; XL otherwise.

## U7a — `gpu-gang.yml` pod leg (PR-B1 commit 1)

- **files_in_scope** (docs-ci): `.github/workflows/gpu-gang.yml` (label `run-gang` or manual
  `workflow_dispatch` only — never `push`/`workflow_call`; U7b-A1-pull deletes the `schedule:`
  block for the window before U7b-A3 re-adds it 6-hourly alongside that unit's never-vacuous
  writer, so no cron of any kind fires this lane between the two), `ci/scripts/runpod_lib.sh` (`gpuCount` becomes a
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
- **cost ceiling** (human-approved before first run), basis `$3.18/h` (the SECURE 2-GPU
  `A100-SXM4-80GB` pod's measured rate, not `$1.59 × 2`): terminate-succeeds bound (i)
  `4 × 300s × $3.18/h + 1h × $3.18/h = $4.24`/run; sweep-only bound (ii)
  `(4 + 1) × 1h × $3.18/h = $15.90`/run (`docs/maintainer/dev-gpu.md#the-gang-leg--two-gpus-in-one-pod`).

## U2a — `TrainingSet` producer (PR-B1 commit 2)

- **files_in_scope**: (db) `store/manifest.rs` (`ProducingDescriptor::TrainingSet` with
  `format` as a canonical string), `catalog/result_repo.rs` (`ResultTableKind::TrainingSet`),
  tests. (wire-server) `crates/jammi-wire/proto/jammi/v1/embedding.proto` (`TRAINING_SET = 4`,
  append-only) and `crates/jammi-wire/src/embedding.rs::result_table_kind_to_proto`/
  `::result_table_kind_from_proto` (the two exhaustive
  `ResultTableKind` mirrors). (ai-core) `fine_tune/worker.rs::run_spec` (materialize-or-reuse,
  then read the table back through `session.sql` over the registered `jammi.{name}` result
  table — `crates/jammi-db/src/store/mod.rs::ResultStore::register_table`
  and `crates/jammi-ai/src/session.rs::infer_ordered_read_back_sql` precedent — with the canonical `ORDER BY`
  re-applied, into today's loader: a compiling intermediate). The `GraphFineTune` arm does
  not route through this producer: `fine_tune/graph_sampler.rs`'s pairs stay in-memory, and a
  graph training set's own table is https://github.com/f-inverse/jammi-ai/issues/538.
  `pipeline/recompute.rs` (arm = re-materialize). (docs-ci) the
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
  for an unordered scan; `crates/jammi-db/src/session.rs::JammiSession::build` sets `target_partitions`).
- **lane**: hermetic + cookbook. **depends_on**: U1. **size**: M.

## U4a — `Collective` trait; device-plural session; `CacheKey`; config refusals (PR-B1 commit 3)

- **files_in_scope**: (ai-core) new `fine_tune/collective/{mod.rs, noop.rs, local.rs, nccl.rs}`,
  `model/cache.rs` (`CacheKey { model_id, device, task: Option, backend: Option }` — shared
  shape with plan 65, recorded in both ledgers), `model/backend/mod.rs` (`DeviceConfig` plural),
  `concurrency/gpu_scheduler.rs` (per device), `session.rs`, `fine_tune/spec.rs`
  (`TrainingCommon.world_size`, `#[serde(default)]` = 1 — the field lands here so the
  refusals below are testable at this commit; queued specs still deserialize), every
  `TrainingCommon { .. }` construction site (`crates/jammi-ai/src/wire/training.rs::lora_common_from_proto`,
  `crates/jammi-ai/src/session.rs::InferenceSession::fine_tune`,
  `InferenceSession::fine_tune_graph`, the `tests/it` sites), `jammi-ai/Cargo.toml` (`cuda` adds `candle-core/nccl`).
  (wire-server, co-owner) `proto/jammi/v1/training.proto` + `crates/jammi-wire/src/training.rs`
  (the per-job `world_size` field, append-only). (db) `config/mod.rs` (`[gpu] devices`; `[worker] world_size`, `collective`), tests. Test targets: hermetic
  tests in the crate's unit tests; the `Nccl` smoke in the existing `gpu_capability` target.
  (docs-ci) the six cu12 packaging sites that name the CUDA runtime library set, so it gains
  `libnccl`:
  `.github/workflows/release-binaries.yml::server-cu12-build` (the comment naming `libnccl`
  alongside the CUDA runtime's other hard `DT_NEEDED` entries) and its packaging step's soname
  loop (`release-binaries.yml::server-cu12-build`'s `Package` step; a note recording that the prior six-name hand list missed
  the `DT_NEEDED libnccl.so.2` `candle-core/nccl` adds — why the list gained a seventh, still
  hand-listed, name rather than a `DT_NEEDED` closure walk, issue #535), `packaging/server-cu12/
  verify_link_set.py`, `packaging/server-cu12/jammi_server/_entry.py::_CUDA_COMPONENTS`,
  `packaging/server-cu12/pyproject.toml` (the `nvidia-*-cu12` pins), `packaging/
  server-cu12/README.md#jammi-server-cu12`; and `.github/workflows/ci.yml`'s `flash-attn-compile` job, which
  gains a preflight step (`ci.yml::flash-attn-compile`: `Preflight — the image carries NCCL`, `rpm -q libnccl
  libnccl-devel && test -e /usr/include/nccl.h && test -e /usr/lib64/libnccl.so`) so a `:latest`
  published before B0's Dockerfile change reds this job before any `--features cuda` step tries
  to link.
- **precondition (S1) — satisfied by B0** (`ci/500-cuda-image-nccl`, merged to `main`): the CI
  CUDA image carries NCCL (`.docker/ci-cuda.Dockerfile` pins `libnccl-2.23.4-1+cuda12.6` and
  `libnccl-devel-2.23.4-1+cuda12.6`, the version `nvidia/cuda:12.6.3-runtime-ubi8` already ships)
  and the `flash-attn-compile` job's preflight step above (`ci.yml::flash-attn-compile`) asserts
  it before this unit's own `cargo clippy -p jammi-ai --features cuda --tests -- -D warnings`
  step (`ci.yml::flash-attn-compile`'s `Clippy jammi-ai --features cuda (nvcc, no GPU)` step) compiles the `Nccl` arm. The CUDA-tarball soname set the
  cu12 packaging above bundles is a fixed HAND LIST of seven stems (`libcudart libcublas
  libcublasLt libcurand libnvrtc libnvrtc-builtins libnccl`), searched first in the CUDA 12.6
  toolkit's lib dir then in `/usr/lib64`, fail-closed per name — a name absent from both
  locations fails the build rather than silently shipping a tarball missing it
  (`.github/workflows/release-binaries.yml::server-cu12-build`, the `Package` step's soname
  loop). Deriving the set from the binary's transitive `DT_NEEDED` closure was excised under
  this unit's stop rule — a closure walk cannot be trusted to reach `libnvrtc-builtins` on its
  own, since it is `dlopen`'d by `libnvrtc` rather than linked, a MEASURED fact (`readelf -d`
  against the toolkit's `libnvrtc.so.12` names no such `NEEDED` entry) — and is filed as issue
  #535, not built by this unit. The post-copy check is filesystem presence only, never the real
  runtime loader (`release-binaries.yml::server-cu12-build`'s `Package` step, the
  `no versioned object for '${soname}' found` failure arm); a runtime loader verification is filed as
  issue #534, not established by this unit.
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

## U2b — Eager loader; partition rule; scaler; whole-set arms (PR-B commit 4)

The streaming arm this unit originally carried is EXCISED (design fix round 3, second BLOCK on
the residency mechanism, pre-committed stop rule fired): PR-B2 ships the eager `TextRows` path
only. A residency-bounded, per-rank streaming reader is its own unit, **U2c** (below), filed as
issue #544 and scheduled before U4b binds a per-rank reader to it.

- **files_in_scope** (ai-core): `fine_tune/data.rs` (per-batch converters over the eager
  `TextRows` path read back through `read_back_sql` — applies `training_set_order_by`; the
  tests-only `Precomputed` arm unchanged; a reader-class allow-list oracle enumerates every
  reader of `sql_relation()` and asserts each either applies the order or pins
  `target_partitions = 1`), `fine_tune/trainer.rs` (the epoch loop over `TextRows`,
  byte-identical to its pre-U2b shape; `batches_per_epoch = ceil(train_count / (W·B))` and every
  step quantity indexed by global batch, at W=1), `fine_tune/worker.rs::run_spec`
  (`PartitionSpec { rank, world, batch, rule }`), `fine_tune/regression_loss.rs` (scaler from
  ONE collected `Vec<f32>` of the train prefix, `from_targets` once), `fine_tune/hard_negative_miner.rs`
  and `fine_tune/gradcache.rs` (whole-prefix consumers of the eager loader, W=1-only),
  `fine_tune/batch_bucket.rs` (rung pinning option), the one `TrainingSetSpec` constructor
  shared by every construction site (or a field-by-field oracle asserting the sites agree),
  tests. (db) `store/mod.rs` reader slicing over the result-table `ListingTable` (no injectable
  row-group knob: `crates/jammi-db/src/storage/writer.rs::ObjectParquetWriter::open`'s `set_max_row_group_row_count`
  stays hardcoded at `65_536`; a multi-row-group fixture is simply >65,536 rows
  through that one writer).
- **invariants_to_preserve**: K3 (scaler over the train prefix, bit-identical), K2, B6.
- **acceptance**: (a) partition rule: for W ∈ {1,2,4} the multiset of rows over ranks at each
  global step equals the W=1 batch, on a fixture whose `train_count` is not a multiple of W·B
  (RED at base); (b) refactor parity holds on every cookbook fixture including regression; (c)
  mining/GradCache runs at W=1 produce bytes identical to base; (d) order: `read_back_sql`
  applies the canonical order on a multi-row-group fixture at `execution_threads > 1` (RED at
  base for an unordered scan), and the reader-class oracle covers every reader of
  `sql_relation()`; (e) the one `TrainingSetSpec` constructor is shared by every construction
  site, asserted by a byte-parity test. This unit's acceptance carries no residency-bound row —
  that criterion moved to U2c.
- **lane**: hermetic + cookbook. **depends_on**: U2a, U4a (the `world` argument). **size**: L.

## U2c — Streaming training-set loader with a residency bound (PR-B2, before U4b's commit; wave 3; issue #544)

Excised from U2b by that unit's fix round 3 (a second BLOCK on the residency mechanism: a lease
held across the excised `BatchChunker`'s carry-over deadlocked at `prefetch = 2` on any
multi-row-group table). Rebuilt here against issue #544's constraints, never carried over
verbatim from the excised arm. The root defect the excised arm never named: the result-table
provider (`store/mod.rs`'s `build_result_table_provider`, a plain `ListingTable` with no
declared file sort order) makes `read_back_sql`'s `ORDER BY` plan a pipeline-breaking `SortExec`
at `target_partitions` ∈ {1, N} — the whole table arrives on the first poll, so no stream built
on top of it can ever be bounded. This unit fixes the provider FIRST, then builds the stream.

- **files_in_scope** (db): `store/mod.rs` (`build_result_table_provider` declares
  `ListingOptions::with_file_sort_order`, rendered from the single source of truth
  `training_set_order_by` — never a third hand-spelling of the order, NULLS placement included),
  `storage/reader.rs` (the per-rank slicing reader the stream is built over), `config/mod.rs`
  (`engine.memory_limit`, DEAD — no `MemoryPool`/`RuntimeEnvBuilder` reads it — wired to a
  bounded `MemoryPool` on the session's `RuntimeEnv`, so exceeding the bound is a TYPED error,
  never an assertion over the loader's own counters), `tests/it/pinned_source_gate.rs`
  (`session_registration_literal_sites` gets its reviewed entry if the stream spells the
  relation literal). (ai-core) `fine_tune/training_set.rs`'s reader-class allow-list (a direct
  `parquet_path` reader — issue #551's `RelationKey` — is a covered member of this allow-list,
  never a silent second route around the engine's `MemoryPool`), `fine_tune/data.rs` (a per-rank
  `StreamSource`: a WHOLE-PREFIX ORDERED stream per rank, filtered by a per-rank `rows_for_step`
  predicate — row groups are 65,536 rows, the partition rule strides at `W·B`, so every rank
  scans and decodes the WHOLE prefix and keeps only its own rows; this `W×` read/decode
  amplification is STATED, not hidden, and is the accepted cost of a per-rank stream over a
  provider with no row-group-level partition pushdown; the residency-accounting types are
  re-designed against this unit's own contract — never the excised `BatchChunker`/`ChunkLease`/
  `ResidencyBound`/`StreamConfig` shapes carried over unchanged), `fine_tune/trainer.rs` (the
  per-rank stream consumer U4b's rank body binds to), tests. No injectable row-group knob: the
  multi-row-group fixture is simply >65,536 rows written through the one existing writer
  (`crates/jammi-db/src/storage/writer.rs::ObjectParquetWriter::open`'s hardcoded `set_max_row_group_row_count(Some(65_536))` — its absence
  of a knob is itself pinned by `default_row_group_row_count_is_65_536`); the two headline
  oracles (the residency bound and the no-`SortExec` provider plan) share ONE materialized
  fixture per test binary.
- **invariants_to_preserve**: K3 (the scaler stays U2b's whole-prefix, one-pass reduction —
  never itself streamed), K2, B6.
- **acceptance**: (a) the provider's physical plan for `read_back_sql` contains NO `SortExec` at
  `target_partitions` ∈ {1, N} (RED at base: the `SortExec` is present); a mutation that
  declares NULLS LAST instead of the canonical placement kills this oracle; (b) the residency
  bound is stated as `live_bytes(rank) ≤ f(B, prefetch, carry_over) + Σ named_exemptions`, each
  exemption its own separately asserted term — the K3 scaler's one collected `Vec<f32>` over the
  train prefix (4 bytes/row) and the whole-set mining/GradCache arms — never a flat
  `batch × prefetch` term alone with the exemptions folded in unstated; on a multi-row-group
  fixture with a consumer holding each chunk ≥ 50 ms the high-water mark stays within the stated
  bound (RED at base: U2c does not exist); a whole-set read under a `MemoryPool` sized below the
  bound fails TYPED, and the streamed read under the SAME pool completes (RED at base: no
  `MemoryPool` is wired); (c) liveness is a property over EVERY held lease, not only the
  steady-state case: the same multi-row-group fixture completes within a wall-clock timeout with
  no deadlock, at every named prefetch value — this is the regression pin for the excised arm's
  `prefetch = 2` deadlock; (d) the slicing is per-rank: a whole-table row-group SCAN with a
  per-rank `rows_for_step` FILTER is what this DEMANDS; a stream SHARED across ranks (rank r
  observing rank r′'s rows) is what it FORBIDS — an oracle that can fail both ways: two ranks'
  streams are independent objects, and swapping the filter for a shared cursor dies it; (e) the
  parity oracle: streamed rows == `read_back_sql`'s rows, in committed order, at
  `target_partitions` ∈ {1, N}; at W=1 the streamed training bytes == U2b's eager golden,
  byte-for-byte; (f) every wired refusal (a chunk-length mismatch, `ResidencyBound::new(0)`, a
  `prefetch` floor, …) has a BEHAVIOURAL oracle — a dying test exercised through the loader's
  public path, never a deletable dead branch.
- **lane**: hermetic + cookbook. **depends_on**: U2a, U2b, U4a. **size**: L (the provider fix,
  the `MemoryPool` instrument, and the per-rank stream are three separately-oracled mechanisms
  sharing one fixture). Scheduled BEFORE U4b binds a per-rank reader to it — **U4b depends_on
  U2c** — even though its own base is the PR-B2 branch after U2b/U3 land (before U4b's own
  commit); the implementation wave is 3 (built concurrently with PR-C(67) once PR-B1 merges).

## U3 — `FineTune` producer; `model_materialization` migration; cache reuse (PR-B2 commit 2, concurrent with U2b)

- **files_in_scope**: (db) `store/manifest.rs` (`ProducingDescriptor::FineTune`;
  `MaterializationEnv` kernel-profile field — declared, UNCOVERED, no writer;
  https://github.com/f-inverse/jammi-ai/issues/546), `catalog/{schema.rs, migrations.rs}`
  (`model_materialization` migration 033, both pin sites; two nullable columns —
  `definition_hash`, `input_anchors_json`; no `manifest_path` column, since the sidecar path
  is always derived from `artifact_path`), `catalog/model_repo.rs` (`probe_model_by_definition`,
  NULL never matches), `store/artifact.rs` (manifest last), `store/reconcile.rs`
  (`prefix_is_referenced`, an admin-scoped whole-catalog scan, guards every `models/`-prefix
  byte-delete — the reap and `delete_unreferenced_prefix` both consult it; `ReconcileReport.
  referenced`/`referenced_count`, carried on the wire, name a prefix found still referenced),
  `tests/it/migrations.rs`. (ai-core) `fine_tune/worker.rs::publish_and_finalize`
  (materialization; probe before training; own name → reused prefix; every byte-delete on the
  abandon path routes through the guarded `PrefixReferences` port, never the unguarded
  primitive), `pipeline/recompute.rs` (arm = retrain), `model/resolver.rs` (manifest on
  `ModelRecord`), `fine_tune/spec.rs` (`cache` lives on `TrainingSpec::FineTune` itself, not
  `TrainingCommon`, so `TrainingSpec::GraphFineTune` cannot carry one — a stray `cache` key
  under `graph_fine_tune` in a persisted `jobs.spec` row is dropped at deserialize, not
  refused; a hard error on unknown keys is a separate reshape,
  https://github.com/f-inverse/jammi-ai/issues/548), `wire/training.rs`
  (`lora_common_from_proto` refuses `cache = Use` for `GraphFineTune`, typed, at decode), the
  canonical-encoding producer and the exhaustive-destructuring completeness test in `jammi-ai`
  (and `jammi-wire` for `FineTuneConfig`). (docs-ci) the `PRODUCING-DESCRIPTOR-VARIANTS` guide
  block (co-owned).
- **invariants_to_preserve**: K5, K7 (exhaustive destructuring of `FineTuneConfig` and
  `TrainingCommon`; the descriptor holds an opaque versioned canonical encoding, never a
  foreign type), K1, B6, B1 (no `register_*`), doc parity.
- **acceptance**: (a) same spec on the same training-set digest with `CachePolicy::Use` is
  refused on every durable submit edge (reuse is https://github.com/f-inverse/jammi-ai/issues/562);
  `Bypass` trains, two model rows may share one prefix; deleting either row is always allowed (no catalog edge
  enforces which is the original — the ownership edge is
  https://github.com/f-inverse/jammi-ai/issues/547) and leaves the prefix servable through the
  other row; the prefix itself is reclaimed only once no live row, in any tenant, still names
  it; (b) the exhaustive-destructuring completeness test; (c) the append-only migration test
  admits the appended migration.
- **lane**: hermetic. **depends_on**: U2a. **size**: L (migration, NULL-probe semantics,
  reference-counted reaping, artifact ordering, ten files across two crates). Co-ownership:
  `manifest.rs` and `recompute.rs` with U2a/U4b.

## U4b — Rank context; gather rule; lockstep; single-node gang (PR-B2 commit 3)

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
  completes (RED at base); (e) W=1 via `Noop` byte-identical to U2b's golden; (f) two real
  devices in one session hold two entries in the production model-cache map for one model id —
  U4a's own hermetic assertion pins the key-type fact against a mirror map, never the production
  insert (`crates/jammi-ai/src/model/cache.rs::ModelCache::do_load`'s `cache.entries.insert`); only one device
  exists off the pod, so a device-collapse mutation at that insert site is hermetically
  UNCOVERED and this determinant is a pod-leg obligation, not a hermetic one. (pod leg,
  `Nccl`, 2×A100): (a) as a digest pair + per-step delta against the pre-registered ε, (c)
  with GPU ε; artifact committed as PR-B commit 7.
- **lane**: hermetic + gpu-gang pod leg. **depends_on**: U2b, U2c (the per-rank residency-bounded
  stream this unit's rank body binds to), U3, U4a, S1. **size**: XL (the
  plan's mathematical core; five hermetic oracles).

## U7b — cluster leg + cluster reap (PR-C commit 1)

The cluster leg is a SEPARATE driver from the pod-tier smoke — never `runpod_gpu_gang.sh`, which
stays the pod-tier driver end to end. The cluster leg's own driver script (built on
`runpod_lib.sh`'s `rp_cluster_*` primitives) launches, ships the NCCL id, and assembles the one
committed artifact for a **2×1 shape**: 2 hosts, 1 GPU each (never `gpuCount: 4`) — a genuine
two-HOST NCCL smoke over `ens1`, not a second copy of the pod-tier's two-process bootstrap.

- **files_in_scope** (docs-ci): `ci/scripts/runpod_lib.sh` (cluster create/get/pods/delete/list
  primitives with the same self-terminating entrypoint deadline a pod already carries), the
  cluster leg's OWN driver script (never `runpod_gpu_gang.sh`) — launch, id-ship, report-ship,
  single-writer assembly (both members only ever REPORT; the driver alone assembles and writes
  the one artifact) — `ci/scripts/execution_surface_reachability_allowlist.txt` (the cluster-leg
  tuples; U7a's rows re-verified if their command lines change), `.github/workflows/gpu-reap.yml`
  (clusters enumerated and reaped; the reaper's second enumeration is fail-closed on a failed
  GET, mirroring the pod arm), `gpu-gang.yml` (never carries a `schedule:` trigger for the
  cluster leg without the allow-listed exception below).
- **invariants_to_preserve**: U7b's OWN obligation — extend `ci/scripts/check_gpu_prove_once.py`
  with a new P7 arm, landing with U7b-A1's own PR beside the script's existing P1–P6, so a
  `schedule:` trigger on ANY paid-lane workflow is refused unless that workflow is on a reviewed
  cron allow-list with its never-vacuous arm named — a rule this unit BUILDS, never one it finds
  already enforced; B2.
- **acceptance**: reap enumerates clusters (RED at base: pods only); P1 rules; guard wiring;
  `check_execution_surface_reachability.py` green with the cluster-leg tuples allowlisted.
- **acceptance (id-secrecy)**: the NCCL id's out-of-band crossing is backstopped by a scan over
  its full carrier set — the pulled artifact directory, the run log, the committed `gang.reason`
  field, and the driver's own staging copy of the id file created for the ship step — asserting
  none of them ever carries the id's 128 bytes (base64 or raw), the staging copy deleted after
  the pull scan runs; an unexaminable carrier (unreadable log, missing staging path) is a
  refusal, never a silent pass (RED at base: the cluster leg and its driver do not exist yet, so
  this scan has nothing to run against).
- **acceptance (schedule visibility)**: the new P7 arm this unit adds to
  `check_gpu_prove_once.py` refuses a `schedule:` trigger on ANY paid-lane workflow — pod or
  cluster — unless that workflow is on a reviewed cron allow-list with its own never-vacuous arm
  named; a planted cron on `gpu-gang.yml` is a FINDING under P7, an allow-listed cron is not —
  RED-then-GREEN within this unit: RED before P7 exists (a planted cron passes unrefused, since
  no arm reads `schedule:` at all), GREEN once this unit's P7 and its fixture land together.
- **lane**: gate scripts. **depends_on**: U7a, S4. **size**: M.
- **cost ceiling** (human-approved before first run, committed figures — never re-derived per
  run): at S4's MEASURED cluster rate, $1.908/GPU/h (README.md#units-and-order (~:305) — never the 2-GPU pod rate),
  the 2×1 shape bills $3.816/h; ≤ 1 h billed wall per run, ≤ 2 runs per authorization; label-only
  until a flake-free streak.

## U5a — `GangService` on `peer_bind`; I-GANG authorization; admit-and-hold (PR-C commit 2)

One bidi RPC, `GangService.RunRank(stream RankControl) returns (stream
RankEvent)` (`crates/jammi-wire/proto/jammi/v1/gang.proto`) — no
`RankAssignment`/`FetchPartition`/`RankEvent::Released`/`RankEvent::Progress`;
message shape and arity are protected by review and
`docs/rigor/contracts/feat_500-C-U5a-1.md`, not by
`api_freeze` (which tracks only `PACKAGE`/`RPC` tokens). Two sub-units, split
across two branches: U5a-1 freezes the wire, I-GANG, and the training-set
identity; U5a-2 builds `HostAdmission` (the admit-and-hold session, drain,
re-verification) on top of it.

### U5a-1 — wire freeze, I-GANG, training-set identity (wire-server → db → docs-ci)

- **files_in_scope**: (wire) `crates/jammi-wire/proto/jammi/v1/gang.proto`,
  `crates/jammi-wire/build.rs`, `crates/jammi-wire/src/lib.rs`. (wire-server)
  `crates/jammi-server/src/grpc/gang.rs` (`GangServer::run_rank`: wire K2
  before any row read; the `world_size == 1` admission lattice, keyed on the
  ROW's own `world_size` (`WorldSizeFact::{Decoded, Undecodable}`,
  `get_job_for_rank`), never the caller's `assign.world` —
  `assign.world != row.world_size` refuses
  (`GangRefusalReason::WorldMismatch`); a row whose `world_size` decodes but
  is not `1` refuses (`GangRefusalReason::MultiHostUnsupported` — the
  training-set pair conjunct and its sidecar verify that would admit a
  genuine multi-host row are U5a-2's to build, filed at
  <https://github.com/f-inverse/jammi-ai/issues/566>); a row whose `spec`
  does not decode a `world_size` at all refuses as a row fact
  (`GangRefusalReason::SpecUndecodable`), never a fault of the read that
  found it; `fresh_instance` for the coordinator's own liveness; every
  admission-time catalog read maps its `Err` through
  `admission_catalog_fault` to `Unavailable`, never `map_engine_error`;
  every determinant collapses to one `FailedPrecondition` with a fixed
  message; ends `Unimplemented` — no `HostAdmission` session exists yet in
  this sub-unit), `crates/jammi-server/src/runtime.rs` (`GangServiceServer`
  mounted beside `PeerServiceServer` on `[server] peer_bind`, never the
  public listener), `crates/jammi-server/src/metrics_layer.rs` +
  `crates/jammi-server/src/routes/health.rs` (`jammi_gang_requests_total{rpc}`).
  (db) `crates/jammi-db/src/catalog/jobs_repo.rs` (`get_job_for_rank(job_id)`:
  primary key, no tenant predicate, never admin scope; `fill_training_set_identity`,
  the write-once CAS over `training_set_ref`/`training_set_location`;
  `fresh_instance(instance_id, lease)`), `crates/jammi-db/src/catalog/lease.rs`
  (`instance_liveness_margin(lease)` = `2 × lease`, shared with
  `reclaim_expired_jobs`'s inline-execution arm), `crates/jammi-db/src/catalog/result_repo.rs`
  (`get_result_table_for_tenant`: the strict tenant predicate, never the
  relaxed `get_result_table` read), `crates/jammi-db/src/catalog/schema.rs` +
  `migrations.rs` (migration adding `jobs.training_set_ref`/
  `training_set_location`, both nullable TEXT, a paired-nullability `CHECK`
  constraint — numbered 033 (`jobs_training_set_identity`) on this branch's
  base; when PR-B2's `033_model_materialization` is on `main` first, PR-C1's
  rebase renumbers `033_jobs_training_set_identity` to 034 (four pin sites +
  the ordered-after oracle: the tuple in `catalog/migrations.rs`'s const
  list, the `MIGRATION_033_JOBS_TRAINING_SET_IDENTITY` constant name in
  `catalog/schema.rs`, the const `tests/it/migrations.rs::EXPECTED_MIGRATION_NAMES`,
  the SQL `IN`-list literal inside
  `tests/it/migrations.rs::migration_029_copies_training_jobs_rows_into_jobs_as_queued`
  (`tests/it/migrations.rs:848`, `DELETE FROM applied_migrations WHERE name
  IN (…, '033_jobs_training_set_identity')` — the ledger rows this test
  clears so the reopen replays every later migration that alters `jobs`, and
  `migration_033_is_ordered_after_032_and_pins_the_pair_at_the_schema_edge`'s
  own `position("033_jobs_training_set_identity")` literal,
  `tests/it/migrations.rs:1572-1746`). (docs-ci)
  `crates/jammi-server/tests/it/api_freeze_baseline.txt` +  `api_freeze.rs`
  (`PACKAGE jammi.v1.gang` / `RPC GangService/RunRank`),
  `tenant_isolation_oracle.rs` (`GANG_LISTENER_ALLOWLIST`, never appended to
  `PEER_LISTENER_ALLOWLIST`), `gang_rank_admission_oracle.rs` (the
  enumerating caller oracle: `get_job_for_rank`'s only production caller is
  the gang `RunRank` handler — the `get_result_table_for_tenant` caller
  surface this file once also enumerated is gone, not merely retired, since
  this sub-unit's handler no longer reaches that verb at all),
  `gang_admission_catalog_fault_oracle.rs` (a source-scan oracle: no
  admission-time catalog read in `run_rank` reaches `map_engine_error`,
  every one reaches `admission_catalog_fault`), `gang_training_spec_parity.rs`
  (producer→consumer parity: `get_job_for_rank`'s `world_size` decode reads
  exactly what a real `TrainingSpec::FineTune`'s `TrainingCommon` producer
  writes, never a hand-written spec literal on either side).
- **invariants_to_preserve**: B5 (I-GANG beside I-PEER, never under
  `TenantResolverLayer`, tenant from the row, non-disclosure on refusal), K2
  (`world == 0`, `rank >= world`; the lease-NULL edge), B6 (docs same commit
  set), B4 (no admitted `RunRank` session reaches a rank body in this
  sub-unit or U5a-2 — every session parks and ends `Aborted{NoBody}` absent
  an earlier exit, once U5a-2 lands), K4 (this sub-unit's instance: the
  `training_set_ref`/`training_set_location` write-once CAS, exercised
  directly at the db layer; no caller on this sub-unit's wire path reaches
  it yet, since the row-keyed world gate refuses every `world_size != 1` row
  outright — verify-at-read is U5a-2's K4 instance, built once a rank
  actually reads the pair it names), B1 (`RunRank`/`Assign`/`Outcome` are
  mechanism names), `api_freeze` additive-only (`Outcome` is a new oneof
  arm, not a new RPC).
- **acceptance**: a1' (the `training_set_ref`/`training_set_location` CAS
  fires exactly once per job, and a concurrent racer reuses the winner's
  pair rather than overwriting it — both asserted directly against the CAS
  at the db layer); b1' (one refusal test per I-GANG determinant — ten
  total: `AdminScope`, `NotFound`, `NotRunning`, `WrongClaimant`,
  `WrongAttempt`, `LeaseDead`, `SpecUndecodable`, `WorldMismatch`,
  `MultiHostUnsupported`, `CoordinatorNotFresh` — status + `test-hooks`
  reason pairwise-identical across every determinant, on both the plain and
  `--features test-hooks` lanes, counted separately; plus wire K2; plus the
  producer→consumer `world_size` parity oracle; plus the admission-catalog-
  fault-maps-to-`Unavailable` source-scan oracle); e1 (`api_freeze` green
  with the gang lines; `every_rpc_is_covered` green with the new allowlist;
  public-listener `Unimplemented` probe; the `get_job_for_rank` caller
  oracle); f1' (a call satisfying every I-GANG determinant still ends
  `Unimplemented` — this sub-unit has no `HostAdmission` session to hand it
  to).
- **lane**: hermetic + server it-suite. **depends_on**: U2b, U4a, PR-B2 merged
  (the `peer_bind` listener this mounts beside `PeerService` on). GRAPH does
  not depend_on: it is deferred to #515. **size**: M.

### U5a-2 — `HostAdmission`, drain, re-verification, admit-and-hold (ai-core → wire-server)

- **files_in_scope**: (ai-core) `fine_tune/worker.rs` (`phase` moves from
  `WorkerShared` to a session-owned cell; `HostAdmission { phase, holder,
  registry }`; the holder CAS lattice — `Free`/`ClaimProbe`/`JobRun`/
  `Rank{job_id,attempt}` — replacing the attempt-keyed fence), `session.rs`
  (`worker_gate_receiver`, the same gate reader a claim loop already
  `wait_for`s on). (wire-server) `crates/jammi-server/src/grpc/gang.rs` (the
  bidi handler's order: admission decision before the CAS, `Admitted` emitted
  only after it, the guard moved into a spawned HOLD-loop future driving
  re-verification and the park-bound timer; the `select!`'s four arms —
  inbound stream, drain signal, re-verification tick, park-bound timer — no
  fifth "quiesce" arm, since `Quiesce` is never emitted; **carried whole from
  U5a-1** — the `world_size > 1` conjunct U5a-1 refuses outright under
  `GangRefusalReason::MultiHostUnsupported`: the training-set pair conjunct
  against the already-written `training_set_ref`/`training_set_location` CAS
  and the sidecar verify that would admit a genuine multi-host row, filed at
  <https://github.com/f-inverse/jammi-ai/issues/566>).
- **invariants_to_preserve**: OPS D6/D10 (a rank is not loop work; the
  release decision reads the holder kind, never `in_flight`; a peer-tier
  rolling restart costs zero net attempts), B4, K2 (a second `Assign` on an
  already-admitted stream is a protocol violation, `InvalidArgument`), K4
  (the peer-local-vs-coordinator-local partition-parity row is U5b-1b-i's K4
  instance, not this one's).
- **acceptance**: c2' (holder contention: `Free` admits, `ClaimProbe` waits
  ≤ one heartbeat then admits-if-freed or `Unavailable`, `JobRun`/another
  `Rank` refuses `Unavailable` at once; the second-`Assign` K2 row); d2'
  (exclusion: a `JobRun`-holding peer refuses, an idle peer admits, inline
  `run_now` is outside the exclusion); g2' (terminal-write scope: the peer
  writes nothing terminal on behalf of a rank; a reclaim/drain/park ends the
  stream with the matching reason, the job row otherwise untouched); i2'
  (re-verification: row refutation vs. catalog-unavailable vs.
  object-store-unavailable are pairwise distinguishable, each with its own
  scope and count-toward-`assembly_attempts` rule); j2' (an admitted
  `RunRank`, either `world_size` arm, receives `Admitted`, is held under
  re-verification, and — absent an earlier exit — ends `Aborted{NoBody}` at
  the park bound; no acceptance row in this sub-unit asserts a body running
  end to end).
- **lane**: hermetic + server it-suite (rows needing `world_size > 1` are
  built below the submit edge). **depends_on**: U5a-1 (cut from its merge),
  68 OPS merged (the claim loop and `WorkerShared` shape this reshapes).
  **size**: L.

## U5b-1 — Coordinator; `Peer` collective; membership substrate; determinism (PR-C commit 3)

Split into five units by capability (design round 4 REFINE, 19 findings folded): the membership
substrate, the row-group attestation inventory, the `Peer` collective + round protocol, the
coordinator, and the `world_size == 1` rank body. Two merge orders are pinned: **U5a-1 lands
before U5b-1a** (U5a-1 creates `instance_liveness_margin()`; U5b-1a only consumes it — U5a-1/
U5a-2 are the wire-server unit's own internal split, `CONTRACT-U5a-v10.md`) and **U4b lands
before U5b-1b-ii** (`spec.rs`'s admission fields are co-owned; the `[worker] world_size` →
`[worker] local_ranks` rename lands in U4b S8, and U5b-1b-ii rebases onto U4b's
`admit_and_place`). "U5b-1" stays the name for the whole capability; each acceptance/oracle
reference to "U5b-1's peer-based run" below means the assembled behaviour of all five units.

### U5b-1a — Membership substrate (PR-C commit 3a)

- **files_in_scope**: (db) `catalog/{schema.rs, migrations.rs}` (`instances_peer_addr_result_root`
  migration, number at rebase, three pin sites incl. the ordered-after oracle in
  `crates/jammi-db/tests/it/migrations.rs` (added by U5a-1) — the same
  `migration_031_is_ordered_after_030_and_adds_releases_and_workers_state`'s pattern
  (`crates/jammi-db/tests/it/migrations.rs::migration_031_is_ordered_after_030_and_adds_releases_and_workers_state`, on `main`) repeated for this migration, cited by
  construct rather than by an offset on a branch this fold cannot read), `catalog/jobs_repo.rs`
  (`upsert_instance` gains `peer_addr` + a
  canonicalized `result_root`; `peer_addr_of(instance_id, window) -> Option<PeerAddr>` — the ONE
  by-id resolution verb, fresh-only under the same margin, no kind/root/self filter — is the
  address-resolution surface DESIGN.md §4 names; `list_gang_members(GangListing { kind,
  self_instance, canonical_root, window })` excludes self, stale (freshness via U5a-1's
  `instance_liveness_margin()`, consumed here, never recomputed), draining/warming, other-kind
  (kinds split on `,`, matched as whole tokens), and root-divergent instances; the member order is
  byte order on `instance_id`, sorted and compared in Rust — never a SQL `ORDER BY`, whose
  collation is backend-dependent; root divergence is likewise a byte-exact Rust comparison of the
  canonicalized string, never a SQL `=`; `prune_instances`' window (`crates/jammi-ai/src/session.rs::InferenceSession::wrap_with`'s call
  site, exactly `lease().saturating_mul(2)` — the same value `instance_liveness_margin()`
  will return) moves to STRICTLY BEYOND the margin, so a member judged merely stale is never also
  eligible for deletion; the lease keeper's `Instance` arm (`crates/jammi-db/src/catalog/lease_keeper.rs::renew_all`, folded into
  the generic `Some(false) → lost` dispatch at `renew_all`) RE-UPSERTS the row on a failed touch
  instead of only flipping `lost` — `touch_instance` (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::touch_instance`) is a pure `UPDATE`
  that can never resurrect a pruned row, so a process whose row was pruned during a transient
  outage now rejoins on its next heartbeat with no restart), `config/mod.rs` (`[server]
  peer_advertise` validated at load — requires `peer_bind`; `canonicalize_result_root()`
  canonicalizes the RESOLVED result-table root — `[storage] result_root` when set, else
  `{artifact_dir}/jammi_db` (`crates/jammi-db/src/config/mod.rs::StorageConfig`'s documented default, mirroring
  `session.rs`'s `build_result_store`, `crates/jammi-ai/src/session.rs::build_result_store`) — scheme-aliased, trailing slash
  trimmed, a non-existent or relative `file://` root refused at load with the row never written;
  canonical-root equality is NECESSARY, never SUFFICIENT, for shared storage — sufficiency is
  established only by the attestation VERIFY, U5a-1's whole-artifact sidecar / U5b-0's and
  U5b-1b-i's per-partition inventory, never by this predicate alone). (ai-core) `session.rs` (the
  ONLY production call site of `upsert_instance`, `crates/jammi-ai/src/session.rs::InferenceSession::wrap_with` — gains the `peer_addr` /
  canonicalized-`result_root` arguments; `InferenceSession::open_with_placement`'s shared-`artifact_dir` topology note stays
  configurable for a gang; `build_result_store` stays the one place the effective
  root is actually computed at runtime). (docs-ci) `docs/guide/src/{configuration.md, security.md,
  deploy-server.md, reference-topologies.md}`.
- **invariants_to_preserve**: B6, K2 (validate chain), K5 (migration, three pin sites).
- **acceptance**: (a) `list_gang_members` excludes a stale, draining/warming, other-kind and
  root-divergent instance on both backends, plus one fresh multi-kind worker included, from a DB
  return order permuted away from `instance_id` order — the returned list is still sorted (RED at
  base); (b) `peer_advertise` without `peer_bind`, or with a non-existent or relative `file://`
  root, is refused at load, each its own typed error; `peer_advertise` with `result_root` UNSET is
  ACCEPTED and canonicalizes `{artifact_dir}/jammi_db` (RED at base: the prior refusal sentence
  is dropped); (c) the migration's ordered-after oracle on both backends; (d) a config with
  `peer_advertise` set (result root set OR unset) produces a non-NULL `peer_addr`/`canonical_root`
  `instances` row through the real session-construction path (`InferenceSession::open` /
  `open_with_placement`), never a direct db write (RED at base: no caller threads the new
  arguments); (e) `peer_addr_of` resolves a busy or other-kind fresh member and returns
  `None` for a stale one; it is unreachable from any public RPC and ignores any caller tenant (the
  invariant oracle, mirroring `get_job_for_rank`'s, RED at base: the verb does not exist); (f) two
  members whose canonicalized `result_root` strings are byte-identical but sit on different
  filesystems land in the member-scoped `StoreUnavailable` arm at the attestation VERIFY, never
  silently — root equality alone never green-lights a round; (g) a `2×`-lease heartbeat gap
  followed by recovery makes the member fresh again without a process restart, via the keeper's
  re-upsert on a failed touch (RED at base: `touch_instance` never resurrects a pruned row); (h)
  `gang_instance_freshness` runs on BOTH backends — the SQLite-only file is widened, and the
  live-postgres lane exercises it too.
- **lane**: hermetic + distributed. **depends_on**: **U5a-1** (creates
  `instance_liveness_margin()`; merge order pinned, U5a-1 lands first), PR-B1. **size**: M.

### U5b-0 — Partitioned attestation inventory (PR-C, new db unit)

- **files_in_scope** (db): `store/manifest.rs` (`MaterializationManifest` gains `leaves:
  Vec<LeafDigest { row_group: u32, digest: ArtifactDigest }>`; `manifest.artifact` becomes the
  FOLD over `leaves` in row-group order — never a second, independently-computed whole-artifact
  digest; `MANIFEST_VERSION` bump), the training-set materialization writer (one leaf per row
  group, written as each row group is written), the freshness/probe readers that consume
  `MaterializationManifest` (an old sidecar with no `leaves` field is a CACHE MISS —
  re-materialize — never a whole-artifact read accepted in its place).
- **invariants_to_preserve**: K5 (append-only manifest shape), B6.
- **acceptance**: (a) leaf count == row-group count, asserted via a pyarrow/parquet metadata
  oracle over a fixture with N row groups (RED at base: no `leaves` field exists); (b)
  `manifest.artifact` == the stated fold of `leaves`, recomputed independently by the test; (c)
  an old-format sidecar round-trips through the freshness/probe reader as a MISS, never a hit
  that treats the whole artifact as one leaf.
- **lane**: hermetic. **depends_on**: none; base `main` after PR-B2. **size**: S. U5b-1b-i
  depends_on this unit (its per-partition verify reads the leaf inventory); U5a-1's own
  admission-time sidecar VERIFY is unaffected — it stays whole-artifact/admission-time-only.

### U5b-1b-i — The `Peer` collective + round protocol (PR-C commit 3b-i)

- **files_in_scope**: (ai-core) `fine_tune/collective/mod.rs` (the lifted `Descriptor` — verb,
  world, root, counts, per-tensor signature, a new extensible `agreement` slot bound to U4b's
  canonical key-name digest), `fine_tune/collective/local.rs` (the `Descriptor` struct and
  `agrees_with` move to `mod.rs`, re-exported), `fine_tune/collective/peer.rs` (NEW — chunking/
  reassembly, dtype/residency oracles, a `BlockingCall` witness token minted only at the
  `spawn_blocking` boundary as a REQUIRED argument of every `Collective` verb — a call from a
  runtime-worker thread is a COMPILE error, never a runtime refusal — and the two-phase round
  protocol: a round applies on a rank only after `RoundCommit`, sent after the coordinator
  observes W ACKs; a fault before the W-th ACK leaves no rank applied), a per-partition
  incremental verify in the rank's read path (reads U5b-0's leaf inventory one row group at a
  time — bounded memory, never whole-artifact buffering; a failure here is MEMBER-scoped,
  `StoreUnavailable`, never an assembly refutation), `tests/distributed/{main.rs, harness.rs}`
  (two-process harness driving `Peer` directly — no job, no claim). (wire-server)
  `crates/jammi-wire/proto/jammi/v1/gang.proto` (additive: `RoundDescriptor` — a closed `verb`
  enum, a wrapped `Counts` message, an exhaustive `DType` match — plus `RoundAck`/`RoundCommit`),
  `crates/jammi-server/src/runtime.rs` (the peer-only listener's `Routes`, built outside
  `assemble_grpc_chain`, gain `.max_decoding_message_size(max_message_bytes)` from the SAME
  `[server.limits]` value the public chain already applies), `limits.rs` (its decode-cap
  invariant restated to quantify over every listener, not only the public chain).
- **invariants_to_preserve**: K4 real, K2 (every numeric wire edge), B5, B6, `api_freeze`
  additive-only.
- **acceptance**: (a) the `Peer` fold over the wire equals `Local`'s fold over fixed tensor
  inputs, byte-for-byte, at f32/f16/bf16 (RED at base: `Peer` does not exist); (b) Peer-W2
  adapter bytes equal Local-W2 on a regression and a contrastive fixture; (c) every round wait
  on every rank expires at the gang deadline naming the round; a disconnect between publish and
  the last ACK leaves no rank applied for that round and the fault names it; (d) a gang message
  of `max_message_bytes − 1` bytes decodes on the peer listener (RED at base: no explicit cap —
  the peer Routes decodes at tonic's 4 MiB default) and one of `max_message_bytes + 1` is
  refused naming the CONFIGURED cap on both listeners; (e) a descriptor disagreement (root,
  counts, the `agreement` slot, an unknown wire `verb`) is a typed refusal naming both sides;
  (f) a corrupted leaf is caught before any collective step, MEMBER-scoped, `StoreUnavailable`,
  never counted against assembly; (g) a `Collective` verb called from a runtime-worker thread is
  a compile error (`trybuild`).
- **lane**: hermetic + distributed. **depends_on**: U5a-1 (frozen `RoundContribution`/
  `RoundResult`), U5b-0 (the leaf inventory its verify reads); does NOT depend on U5b-1a — the
  harness drives `Peer` directly. **size**: L.

### U5b-1b-ii — The coordinator: membership → assignment → dispatch → assembly (PR-C commit 3b-ii)

- **files_in_scope**: (ai-core) `fine_tune/worker.rs` (the coordinator/assembly body; the
  assembly cooldown/counter SPLIT — `next_assembly_after` written on every non-proceeding
  attempt as `backoff(k)`, `assembly_failures` counting only the terminal-refusal class over a
  TOTAL reason table — `Refuted`/all-root-divergent counted and cooled down; `Unavailable`/
  `StoreUnavailable`/short-listing cooled but not counted; `NoBody`/`Drain`/`Cancelled` neither;
  a success resets the counter), `fine_tune/spec.rs` (`RankAdmission`'s `world_size > devices`
  check REPLACED by `world_size > serveable_world`, sourced from a NEW `[distributed]
  max_world_size` — **co-owned with U4b: merge order pinned, U4b lands first**, since the
  `[worker] world_size` → `[worker] local_ranks` rename lands in U4b S8, and this unit rebases
  onto U4b's `admit_and_place`). (db) `config/mod.rs` (`[distributed] max_world_size`, default
  1, loads independently of `[worker] local_ranks` — no cross-check between the two knobs),
  `catalog/migrations.rs` (`jobs_assembly_failures_next_after` migration, three pin sites),
  `catalog/jobs_repo.rs` (`claim_next`'s CANDIDATE subselect gains the cooldown term on the SAME
  backend clock the lease columns use, never the outer UPDATE guard or a process clock; the
  counter+cooldown UPDATE; the `training_set_ref`/`training_set_location` CAS call site and its
  ABORT arm — a moved claim exits without a write). (wire-server) `grpc/gang.rs` (the dispatch
  entrypoint into the coordinator body).
- **invariants_to_preserve**: K2, K4, K5 (migration), OPS D10 (a member-scoped or transient
  assembly outcome costs zero net attempts).
- **acceptance**: (a) a higher-priority job inside its cooldown does not block a lower-priority
  ready job on either backend (RED at base: no cooldown term exists); (b) a skewed process
  clock never changes when a cooldown expires (RED at base); (c) one row per reason in the total
  counting table (RED at base); (d) a `world_size` within `serveable_world` but beyond this
  host's own devices submits and is decided by assembly, never a submit-time refusal; a
  `world_size > serveable_world` refuses at submit with no catalog read (RED at base: no
  `serveable_world`/`max_world_size` exist); (e) rank assignment is a pure function of the
  sorted membership listing — no substitution: a member answering `Unavailable` ends the
  CURRENT attempt (cooled down, not counted), the NEXT attempt re-lists; (f) the CAS pair is
  never set partially; a moved claim aborts with no write, a concurrent CAS sees zero rows and
  REUSEs.
- **lane**: hermetic + server it-suite. **depends_on**: U5b-1a (`list_gang_members`,
  `canonicalize_result_root`), U5b-1b-i (the `Peer` collective it dispatches to), U4b (`spec.rs`
  co-ownership, merge order pinned), U5a-1, U5a-2. **size**: L.

### U5b-1b-iii — The `world_size == 1` rank body; runner-role writer split; `Outcome`; resume pin (PR-C commit 3b-iii)

- **files_in_scope** (ai-core): `fine_tune/worker.rs` (every job-row-writing site on the run
  path reparameterized on RUNNER ROLE in one diff — the lease-hold registration/renewal,
  `in_flight` counting, the `Releasing` self-release arm, the acceleration-report write, EVERY
  `record_failed` call site (a DERIVED enumeration — `grep -n 'record_failed(' worker.rs` minus
  doc lines — never a hand list; a missed site is a COMPILE error, the runner-role parameter is
  REQUIRED), and `finish_job_with_model`'s call site; the enumerating doc table regenerated as a
  per-site table in the same diff), `fine_tune/trainer.rs` (`save_resume_checkpoint`/its write —
  the runner-role gate for resume-checkpoint writing lives HERE, never inside `store/artifact.rs`,
  which stays role-agnostic), the `RankEvent::Outcome` producer at the rank body's natural end
  (built in the SAME diff as U5b-1b-ii's terminal write on receipt — never shipped alone).
- **invariants_to_preserve**: K4 (`W == 1` stays byte-identical to today's loop path), B6, the
  single-writer rule (the gang's output is one artifact written by the lease holder; no per-rank
  fragment, no peer-side write into the artifact prefix).
- **acceptance**: a per-determinant table (never "the mutation fails a test") — each derived
  `record_failed` site, `finish_job_with_model`'s call site, `in_flight` counting, the
  `Releasing` arm, the acceleration-report write, and the resume-checkpoint pin, exercised once
  under the loop-claiming role (a regression oracle, unchanged behaviour) and once under the
  coordinator/rank role (RED at base: no runner-role parameter exists — a missed site is silent,
  not a compile error); `RankEvent::Outcome` end-to-end reaches the same terminal state as
  today's loop-claimed run, byte-for-byte, via a hermetic two-member `Local` gang (`W == 1`
  itself never traverses the coordinator — a pinned row); `W > 1` resume is refused at assembly
  with a typed reason (resume-state broadcast across ranks is its own follow-up, issue #543,
  never built here).
- **lane**: hermetic. **depends_on**: U5b-1b-ii (`Outcome`'s only consumer is that unit's
  terminal write on receipt; they land together, never `Outcome` shipped alone). **size**: L
  (the largest unit in the split — eleven `record_failed` sites × two runner roles × the K4
  byte-identity oracle).

## U5b-2 — Watchdog; abort with no terminal write; released-vs-failed; chaos (PR-C commit 4)

- **files_in_scope** (ai-core): `fine_tune/worker.rs` (watchdog; attempt abort by flipping the
  hold's `lost` flag so the run exits through the leave-for-reclaim arm,
  `crates/jammi-ai/src/fine_tune/worker.rs::JobWorker::run_claimed_job_under` (the lease-lost arm);
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
- **lane**: distributed (chaos advisory as today) + gpu-gang cluster leg. **depends_on**:
  U5b-1b-ii, U5b-1b-iii (the coordinator/rank-body split it wraps a watchdog around),
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
  `ci/scripts/publish_crates.sh`'s `PUBLISH_ORDER` array (`jammi-ballista` inserted before `jammi-server` in the
  topological publish list — a `v*` tag would otherwise half-publish); `.claude/agents/
  wire-server.md` `owns:` gains `crates/jammi-ballista/**` (`check_swarm_bijection.py` asserts a
  total partition of `crates/`; this edit trips `SWARM_GATE_TOUCHED`, so **PR-D is an admin
  merge**); `docs/maintainer/MAINTAINER-GUIDE.md` dep-DAG block regenerated by
  `ci/scripts/gen_dep_dag.py` (advisory by standing). Shuffle stays Ballista's local `work_dir`
  (no object-store shuffle in v1).
- **invariants_to_preserve**: B4 (roles are config; no cargo feature; the library keeps the
  capability through the crate), B2 (dep direction), K4 (bytes through Ballista == bytes through
  the peer path), B6, K6 (publishable, lockstep, in `publish_crates.sh`'s ordered list), B1
  (pre-swept names, README r45).
- **acceptance**: hermetic: codec round-trip for every operator (RED at base); config: `[ballista]`
  parses, unset = no roles, `scheduler_bind == peer_bind/flight_listen/health_listen` refused (RED
  at base). Distributed (three processes, one binary): (a) an embedding job via
  `submit_physical_plan` across two executors → bytes identical to the same query's
  single-executor plan (RED at base: no `jammi-ballista` crate exists to submit through); (b) a
  W=2 gang job through the scheduler → bytes identical to U5b's, never task-retried (RED at
  base); (c) killing an executor mid-gang fails the job and requeues it through jammi's lease path.
- **lane**: hermetic + distributed. **depends_on**: U1, U5b-1b-ii, U5b-1b-iii (the full U5b-1
  coordinator/rank-body split), S6. **size**: L.

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
  `upsert_worker` from the session's device list, `crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::upsert_worker`; number at rebase,
  three pin sites incl. an ordered-after oracle on both backends — this migration lands after
  PR-C(67)'s, so the oracle asserts ordered-after BOTH `instances_peer_addr_result_root` (U5b-1a)
  and `jobs_assembly_failures_next_after` (U5b-1b-ii)), `catalog/jobs_repo.rs`
  (`WorkerRecord.devices`), `catalog/compute_repo.rs`
  (new, generic CRUD). Tests in `tests/distributed/cluster_state.rs`.
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
