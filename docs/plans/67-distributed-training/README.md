# 67 — Multi-GPU and multi-node training as engine mechanism, on DataFusion (#500)

**Status:** PLANNED, v3 — scoped (gap-analyzer: invariant-crossing, 2026-09-10), pressure-tested
in three rounds (design lens and sizing lens; rounds 1 and 2 REFINE, every finding folded;
round 3 is the disposition check — see `PRESSURE.md`), NOT implemented. This directory is the hand-off artifact for the lead that
implements it in a fresh session: read this file, then `DESIGN.md`, then `UNITS.md`, then
`SIZING.md`, then `PRESSURE.md`.

**Posture:** greenfield. Nothing that exists is a constraint; whatever needs rebuilding the
right way is rebuilt. Every fork is resolved by deriving from
`docs/guide/src/philosophy.md` and `docs/swarm/CONSTITUTION.md` and from solid outside
references (`DESIGN.md` §References), never by asking.

## The ask (issue #500, body rewritten 2026-09-10)

Training runs today in one process on one device. The worker fleet is job-level parallelism
only. Build multi-GPU and multi-node training as engine mechanism: a training job declares a
world size, N ranks train one job cooperatively, gang failure semantics are explicit, identity
and parity are scoped, and placement stays the deployer's runtime.

## The position (DESIGN.md §1)

"DataFusion for training" is correct at three layers and wrong at a fourth:

1. **The training set is a plan and a producer.** Materialized once per job as a
   content-addressed result table (`ProducingDescriptor::TrainingSet`); its committed row order
   is pinned by its artifact digest; sharding is partitioning of the train prefix of that table
   by a fixed rule; epochs re-read it with bounded memory.
2. **A trained model is a producer.** `ProducingDescriptor::FineTune` folds the training-set
   digest, the base-model identity, the canonical training spec, the seed and the topology into
   a definition hash; the artifact is manifest-attested; replay is retrain.
3. **The gang is a co-scheduled unit, not a per-step query.** N ranks launched together; every
   rank gathers the global batch's representations and computes the identical global loss, so
   batch-coupled objectives (CoSENT, AnglE, in-batch negatives) keep their semantics; adapter
   gradients are summed; collectives run in-binary behind one trait; any rank failure fails the
   attempt; the existing lease, reclaim and job-level resume checkpoint give restart.
4. **Not** the SGD step or the gradient exchange as DataFusion operators or aggregates (the
   refuted 2014 MLlib `treeAggregate` design).

Distributor-agnosticism is proven last by hosting Ballista scheduler and executor roles in the
jammi binary and executing the same operators unchanged through a physical extension codec.

## Decisions taken by the user (2026-09-10)

| Fork | Decision |
|---|---|
| Ballista posture | Distributor-agnostic build on DataFusion `ExecutionPlan` + `PhysicalExtensionCodec`; the jammi binary is the worker over its own gRPC surface. The Ballista-role unit is **in plan, mandatory, last** (gates completion). |
| Hardware proof | CPU-collective gang protocol hermetic in CI always. Single-node multi-GPU NCCL on a RunPod pod and multi-node NCCL on a 2-node RunPod instant cluster, both as committed artifacts. |
| Unit 0 scope | The greenfield rebuild of the training-data path and model identity is **in scope**. |

## Lead rulings (scope verdict + both pressure-test rounds, resolved; principle in parentheses)

1. **Streaming boundary.** The loader becomes a per-epoch stream of `TrainingBatch` read from
   the training-set table by row group, bounded by `batch × prefetch`; each head constructor
   becomes a per-batch converter. A stream that re-buffers into the old `Vec` is rejected.
   (Right abstraction.)
2. **Epochs re-read the same committed table**, identical order, no shuffle. Two arms derive
   rows from the model rather than the table — hard-negative mining rebuilds a mined loader
   at each refresh epoch, and GradCache treats the whole train prefix as one in-batch-negative
   batch (`trainer.rs:1120`, `trainer.rs:1616-1626`). Both stay **W=1-only in this plan**: a
   typed refusal at submit time when `world_size > 1` and (`hard_negatives.mine == true` or
   `cached == true`) — `mine` is the real gate (`trainer.rs:1451`; `refresh_every` defaults to
   1 and is not a predicate) — plus an explicit exemption from the residency bound. The gather
   primitive (ruling 6) is what a follow-on would use to lift this. (K2; honest scope.)
3. **Split, order and partition rule.** Materialization orders by the full projected tuple
   (identical tuples are identical rows). The split is today's arithmetic
   (`data.rs:477-481`: `val_count = round(rows × validation_fraction)`, train prefix
   `[0, rows − val_count)`), recorded in the FineTune descriptor through `validation_fraction`.
   Partition rule v1 over the train prefix: global batch t is rows `[t·W·B, (t+1)·W·B)`; rank r
   takes `[t·W·B + r·B, t·W·B + (r+1)·B)`; the union over ranks at step t is exactly the W=1
   batch of size W·B. The trailing global batch is **kept** (today's trailing-window scaling
   stays); when `train_count mod (W·B) ≤ r·B` rank r holds zero rows, encodes nothing, and
   contributes a 0-row tensor to every gather with a zero in the counts vector. Every step
   quantity is a function of the **global** batch: `batches_per_epoch = ceil(train_count /
   (W·B))`, the global step, the LR horizon and the trailing-window scale
   (`trainer.rs:843-852`, `:2505-2514`). No row-group alignment requirement. The rule has a
   version tag folded into identity. The tests-only `Precomputed` loader arm
   (`data.rs:439-442`, split by batch count at `:493-497`) stays outside the table path,
   unchanged. (Design correctness; K7 on the TrainingSet descriptor.)
4. **`TargetScaler` over the train prefix**: rank 0 collects the target column of rows
   `[0, train_count)` in committed order into **one** `Vec<f32>` on the trainer's device and
   calls `from_targets` **once** (`regression_loss.rs:169-190`, a two-pass whole-tensor
   reduction; a chunked accumulation would move the low bits), so it is bit-identical to today
   (`trainer.rs:824-836`) and refactor parity holds (K3; no validation leakage). This is a
   named exemption from the residency bound (4 bytes per row). It travels in the
   `RankAssignment` and persists for resume as today.
5. **Model identity.** The catalog name `jammi:fine-tuned:{job_id}` stays (re-claim
   idempotency key). The model gains a materialization (definition hash + input anchors +
   artifact digest) persisted by migration 029 and written last into the artifact prefix.
   The descriptor folds the **canonical serialization of the whole `TrainingSpec` variant**
   (`FineTuneConfig` included — `use_rslora`, `rank_pattern`, `init_lora_weights`,
   `max_seq_length`, `matryoshka_dims`, `quantile_levels`, `validation_fraction`, early-stopping,
   `cached`, `hard_negatives`, …) plus the topology fields. `jammi-db` depends on no jammi
   crate but `jammi-numerics`, so the descriptor carries an **opaque, versioned canonical
   encoding** (`spec_canonical: String`, `spec_schema_version: u32`) produced by the crate
   that owns the type (`jammi-ai`, over `jammi-wire`'s config), plus db-local primitives; the
   same shape holds for `TrainingSet.format`. The completeness test is an exhaustive
   destructuring (no `..`) sited in the owning crate, so a new field breaks the build rather
   than escaping the hash. Cache reuse: the new job registers its own name pointing at the reused prefix; a prefix
   is reaped only when no model row references it (reconcile attribution, `store/reconcile.rs:14-17`);
   029's columns are nullable for `ContextPredictor`, and the probe never matches NULL. (K7, K1.)
6. **The gather rule.** jammi's default embedding loss is batch-coupled (`trainer.rs:4356`
   routes `CoSent | None` to a pairwise log-sum-exp over the (n,n) batch; AnglE and MNRL
   likewise), so per-rank gradient averaging is not the global-batch gradient. Rule: every
   rank `all_gather`s the step's representations (anchor/positive/negative columns, and the
   scores), computes the **identical global loss** over the gathered batch, backpropagates with
   a gather whose backward keeps only the local slots (no backward communication), and the
   adapter gradients are **summed** across ranks. Invariant: **the gather point is
   downstream of every trainable parameter; nothing trainable consumes a gathered remote
   slot.** Per batch arm: contrastive/pairs/triplet gather the encoder outputs (already
   post-projection, `trainer.rs:2195-2225`); classification gathers the **logits** from
   `classify()` (the head is applied inside the loss today, `trainer.rs:2676-2679`, so gathering
   `embeddings` would W×-inflate the head gradient); regression gathers the head output
   (`head_forward` already runs pre-loss, `:2245`); NER stays refused as today. This is exact
   for row-decomposable and batch-coupled objectives alike, needs no count weighting, and its
   payload is
   `W·B·d` floats per representation column plus the adapter gradients — kilobytes to a few
   megabytes per step. (Design correctness; the treeAggregate refutation still stands because
   the collective is never the query engine.)
7. **Lockstep control flow.** Every loop-control predicate that crosses a collective is itself
   collective: the step boundary, `global_step`, the LR horizon and the trailing-window scale
   are functions of the global batch index (`batches_per_epoch = ceil(train_count / (W·B))`,
   landed by U2b at W=1), never of a rank-local `batch_count`; the divergence skip (`trainer.rs:2560-2567`), the 3-strikes abort, early
   stopping and epoch exit are decided by an `all_reduce_max` of flags (rank 0 decides
   validation-based stops and broadcasts). The reduce set is the canonical ordered
   `trainable_vars` list, zero-filled where a rank's `GradStore` has no entry
   (`optimizer.rs:600-611`). (Feasibility: no rank can wait at a barrier alone.)
8. **Rendezvous supersedes the issue's design; zero catalog *schema* for ranks.** The claiming
   worker is the coordinator (rank 0) with the only lease; it resolves peers from
   `[training] peers`, mints the NCCL id, and sends each peer a `RankAssignment`. Peers fence
   on **`job_id`**: a `RunRank` at attempt N aborts every local runner of that job at attempt
   < N whatever rank it held (the rank→peer mapping may shift between attempts); a lesser or
   equal attempt is refused; `(job_id, rank)` is the runner's identity only. Peer death fails
   the attempt and the job requeues (existing `attempts`/reclaim); coordinator death expires
   the lease. Resume from the job-level resume checkpoint rank 0 writes; the bundle's
   `dropout_positions` map (`resume.rs:107`, per-process Philox positions) becomes **per
   rank**, gathered to rank 0 at the epoch boundary, and each rank's dropout seed derives as
   `f(seed, rank)` with `world_size` already in identity — so a resumed gang reproduces an
   uninterrupted one at equal topology. (Feasibility; K5 crossed once, by 029.)
9. **Authorization of the peer surface.** Tenant scoping authenticates the caller, not the
   verb, so `RunRank`/`FetchPartition` bind to a real leased job: the assignment carries
   `(job_id, tenant, attempt, coordinator_worker_id)`; the peer verifies through a tenant-scoped
   **read-only** catalog read that the job is `running`, claimed by that worker, lease live; it
   resolves the training-set table and base model by id through the catalog and derives every
   URL itself — no URL is ever trusted from the wire; `FetchPartition` takes a result-table id
   and partition index. Unauthorized calls are refused with a typed status (a U5a RED oracle).
   Mutual transport auth remains the deployer's, as for every surface today. (B5; K2.)
10. **Transport before Ballista is descriptors**, never serialized physical plans (the engine
    plans no `LogicalPlan` for its compute verbs, `pipeline/asof/exec.rs:4-6`).
    `datafusion-proto` enters only with the Ballista unit. (True dependency graph.)
11. **Peer surface shape.** `jammi.v1.GangService` (server-streaming Arrow IPC), mounted
    tenant-scoped through `AssembledChain::mount_tenant_scoped`. A second Arrow Flight service
    would collide on the Flight service name. #485 request bounds apply. (B5; K4 real.)
12. **Operator partitioning** is an explicit sub-unit (U6): `InferenceExec` declares one
    partition today (`operator/inference_exec.rs:144`).
13. **One code path, four collectives.** `Collective` trait — `all_gather`, `all_reduce_sum`,
    `all_reduce_max_flags`, `broadcast`, `barrier` — with `Noop` (W=1), `Local` (in-process
    ranks, rank-ordered reduce), `Peer` (cross-process over the gang stream, coordinator-reduce
    in rank order, exact) and `Nccl` (`candle-core/nccl` under the existing `cuda` feature).
    W=1 runs the same trainer with `Noop`. (B4.)
14. **Model-cache key.** Plan 65 already proposes rekeying the cache from `ModelId` alone to
    `(ModelId, ModelTask, Option<BackendType>)` (`docs/plans/65-resolve-witness/README.md:23-26`).
    U4a introduces one `CacheKey` struct carrying `device` now and the plan-65 fields as
    `Option`s so the two plans share one key shape; recorded in both ledgers. `None` is a
    distinct key value, never a wildcard (stated so plan 65's `Some(task)` lookups are not
    surprised), and the single-flight `in_flight` map (`cache.rs:45`, keyed by id alone) is
    rekeyed with the entries map.
15. **K4 is the remote-equals-embedded invariant**; the issue misused the ID. Real crossing:
    W=1 through the gang path equals the in-process trainer byte-for-byte. Equal-topology
    reproducibility and W-invariance are two NEW oracles (`DESIGN.md` §6).
16. **GPU byte-equality is not claimed until proven.** FlashAttention-2's backward accumulates
    `dQ` with a non-deterministic `atomicAdd` unless its deterministic path is forced
    (`crates/jammi-kernels/third_party/flash-attention/src/flash_bwd_kernel.h:122-123`), and
    NCCL's summation split depends on channel and buffer settings. Byte oracles hold on the
    hermetic `Local`/`Peer` legs; GPU legs record digest pairs (a fact, never a failure until
    S5 promotes) and compare per-step loss deltas against an ε **pre-registered per leg before
    the first gating run** (derivation: S5, or the max delta over ≥ 3 same-seed baseline runs
    on that box). (Feasibility.)
17. **Config surface is topology, not a sixth backend.** `[gpu] devices`, `[training]
    world_size`, `peers`, `rank_timeout_secs`, `collective`; per-job `world_size` in the spec.
18. **Exactly one migration, 029 `model_materialization`.** Migrations end at 028; the issue's
    029/030 claim was false and is corrected in its body.
19. **DataFusion 54 upgrade is a unit, first**, sized by spike S3. Compatible third-party lines
    exist (`datafusion-federation` 0.5.5 → ^54, `datafusion-table-providers` 0.13.1 → ^54 — a
    0.10.1 → 0.13 jump on the db-owned pin at `crates/jammi-db/Cargo.toml:55`,
    `datafusion-flight-sql-server` 0.4.18 → ^54; crates.io 2026-09-10). (B6/K6.)
20. **Committed artifact means the repo's proof convention**: a `gpu-gang.yml` lane modeled on
    `gpu-prove.yml` (label, nightly, manual; off the merge path), sha-stamped JSON under
    `crates/jammi-kernels/artifacts/cuda-runs/`. Box A100-SXM4-80GB; cluster 2 pods × 2 GPUs.
    `ci/scripts/runpod_lib.sh` hardcodes `gpuCount: 1` (line 1263) and has no cluster
    primitive, so the lane unit owns those changes and spike S4 proves provisioning first.
    Every `cargo` invocation in a non-merge-path script must be allowlisted in
    `ci/scripts/execution_surface_reachability_allowlist.txt` (the registry is derived from
    every tracked `ci/scripts` file). GPU and gang tests land in the existing gated targets
    (`gpu_capability` under `live-gpu-tests`, `distributed` under `live-distributed-tests`,
    `crates/jammi-ai/Cargo.toml:240-272`); no new `[[test]]` target.
21. **#482 consequence kept**: StatefulSet with a headless service (or indexed Job) for stable
    rank addresses and ordered startup.
22. **Unit split rubric** (SIZING.md): a unit is the smallest change with its own RED-able
    acceptance; the producer/consumer seam is a valid cut; units sharing files ship as ordered
    commits of one PR; each PR boundary states its honest reason.
23. **Naming.** No new `pub` items with any of the seven governance stems `promote`, `retire`,
    `register`, `approve`, `gate`, `stage`, `transition` (`ci/scripts/check_no_consumer_names.py:78-80`,
    prefix-anchored). Every pub noun this plan names was swept against the matcher and is clean. Use `gang`, `rank`, `world`, `collective`,
    `exchange`, `partition`, `assignment`, `training_set`.
24. **ContextPredictor is out of the gang scope**; it touches none of `data.rs`/the loader.
25. **This plan is itself a docs-only PR** on `feat/500-distributed-training-plan`.

## Units and order (SIZING.md has the analysis)

| PR | Commit | Unit | Name | Lane | Depends on |
|---|---|---|---|---|---|
| A | 1 | U1 | DataFusion 54 line upgrade (workspace-atomic) | hermetic + cookbook + new db-features clippy step | S3 |
| B | 1 | U7a | `gpu-gang.yml` pod leg; `runpod_lib.sh` gpuCount; artifact schema | gate scripts | S4 |
| B | 2 | U2a | `TrainingSet` producer (worker still reads into today's loader) | hermetic + cookbook | U1 |
| B | 3 | U4a | `Collective` trait + `Noop`/`Local`/`Nccl`; device-plural session; `CacheKey`; config refusals | hermetic (+ pod leg for `Nccl` smoke) | S1 |
| B | 4 | U2b | Streaming loader; partition rule; scaler over train prefix; mining/GradCache adaptation | hermetic + cookbook | U2a |
| B | 5 | U3 | `FineTune` producer; migration 029; cache reuse | hermetic | U2a |
| B | 6 | U4b | Rank context; gather rule; lockstep control; single-node gang | hermetic + gpu-gang pod leg | U2b, U3, U4a |
| B | 7 | — | pod-leg artifact JSON (lane unit follow-up) | gpu-gang | U4b |
| C | 1 | U7b | cluster leg + cluster reap primitive | gate scripts | U7a, S4 |
| C | 2 | U5a | `gang.proto`; `GangService` mount; `FetchPartition`; authorization | hermetic + server it-suite | U4a |
| C | 3 | U6 | Partition-aware inference operator; distributed frozen forward | hermetic (the two-worker leg is U5b's (e)) | U2b, U5a |
| C | 4 | U5b | Coordinator; `Peer` collective; attempt fence; chaos | distributed (manual dispatch before merge) + cluster leg | U4b, U5a |
| C | 5 | — | cluster-leg artifact JSON | gpu-gang | U5b |
| D | 1 | U8 | Ballista scheduler/executor roles + extension codec (mandatory, last) | hermetic codec round-trip + distributed three-process arm | U1, U5b, U6, S2 |
| D | 2 | U9 | Docs (guide, reference topologies, maintainer guide, CHANGELOG) | docs gates | all |

Spikes (time-boxed, no PR, results in the ledger before the dependent unit is briefed):
S1 (→ U4a, U5b) `candle-core/nccl` under jammi's `cuda` feature; `Comm::from_devices` over two
candle CUDA devices; `all_gather` on a candle tensor's storage with equal and unequal counts
(rank-ordered concat layout); a two-process `Comm::from_rank` rendezvous on the S4 pod; S2 a Ballista 54 executor runs a custom `ExecutionPlan` via
`with_ballista_physical_extension_codec`, task retry attempts settable to 0 per job; S3 a
throwaway workspace compile on datafusion 54 / arrow 58 / object_store 0.13 including
`-p jammi-db --features postgres,mysql` and `pyo3-arrow`; S4 one 2-GPU pod through a
`gpuCount: 2` variant of `rp_deploy_live` plus a create-cluster/teardown probe — **spends real
money**: ≈ one $3 pod-hour plus cluster minutes, human-approved before it runs; S5 CUDA
bit-reproducibility with the flash deterministic path forced, `CUBLAS_WORKSPACE_CONFIG` set and
NCCL channels pinned (decides whether GPU byte oracles are promoted).

## Hand-off: how a fresh lead kicks this off

1. Read `docs/swarm/CONSTITUTION.md`, `docs/swarm/SELF-FAILURE-MODES.md`, `.claude/agents/lead.md`.
2. Phase 0: seed the ledger at `.jammi/ledger/distributed-training-500-<date>.jsonl` from
   `UNITS.md` (this session's rows are in `.jammi/ledger/distributed-training-500-20260910.jsonl`,
   gitignored — copy what you need). Phase 0.5 is done for the plan; re-run `gap-analyzer`
   only for a unit you change.
3. Run S3 immediately (it sizes PR-A); S1, S2, S4, S5 concurrently, each recorded in the ledger.
4. PR-A (U1) alone, based on `main`, one worktree, one commit; `docs-ci` for shared manifests,
   `db` for the table-providers pin and the new clippy step, other owners for compile fixes.
   PR bases: PR-B branches from `main` after PR-A merges; PR-C after PR-B; PR-D after PR-C.
   Parallel worktrees inside a PR are rebased onto the PR branch in the stated commit order
   before the gate run that counts; no PR stacks on an unmerged PR.
5. PR-B in the commit order above. Concurrency: U7a ∥ U2a ∥ U4a (three worktrees, disjoint
   seams); U2b after U2a; U3 ∥ U2b (U3 owns `manifest.rs` and `worker.rs::publish_and_finalize`;
   U2b owns `data.rs`, `trainer.rs`, `worker.rs::run_spec`); U4b last (it extends U3's
   hash-flip test with the topology fields and touches `manifest.rs` — co-ownership recorded).
   Label the PR to fire the pod leg; the artifact lands as commit 7.
6. PR-C in the commit order above. Concurrency: U7b ∥ U5a; U6 ∥ U5b after U5a.
   `distributed.yml` is dispatched manually and must be green on the deterministic leg before
   merge (it has no PR trigger).
7. PR-D: U8 then U9. U8 is the completion gate.
8. Every commit runs phases 3–6.5; every PR runs phase 7. Amend subagent commits with the
   session trailers.
