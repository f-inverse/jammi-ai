# 67 — Multi-GPU and multi-node training as engine mechanism, on DataFusion (#500)

**Status:** PLANNED — scoped (gap-analyzer: invariant-crossing, 2026-09-10), pressure-tested
(see `PRESSURE.md`), NOT implemented. This directory is the hand-off artifact for the lead that
implements it in a fresh session: read this file, then `DESIGN.md`, then `UNITS.md`, then
`SIZING.md`. `PRESSURE.md` records what the pressure-testers attacked and what was folded.

**Posture:** greenfield. Nothing that exists is a constraint; whatever needs rebuilding the
right way is rebuilt. Every fork is resolved by deriving from
`docs/guide/src/philosophy.md` and `docs/swarm/CONSTITUTION.md`, and by solid outside
references (recorded in `DESIGN.md` §References), never by asking.

## The ask (issue #500, as rewritten on 2026-09-10)

Training runs today in one process on one device. The worker fleet is job-level parallelism
only. Build multi-GPU and multi-node training as engine mechanism: a training job declares a
world size, N ranks train one job cooperatively, gang failure semantics are explicit, identity
and parity are scoped, and placement stays the deployer's runtime.

## The position (reviewed against outside evidence; DESIGN.md §1)

"DataFusion for training" is correct at three layers and wrong at a fourth:

1. **The training set is a plan and a producer.** A training set is materialized once per job
   as a content-addressed result table produced from a DataFusion plan (`ProducingDescriptor::
   TrainingSet`). Its committed row order is pinned by its artifact digest. Sharding is
   partitioning of that table by a fixed rule; rank r consumes partition r; epochs re-read the
   table with bounded memory. The `ORDER BY`-string-as-determinism mechanism is retired.
2. **A trained model is a producer.** `ProducingDescriptor::FineTune` folds the training-set
   digest, the base-model identity, the spec, the seed, and the topology (world size, partition
   rule, collective backend) into a definition hash; the model artifact is manifest-attested
   like a result table; the recompute pipeline's replay arm for it is retrain.
3. **The gang is a co-scheduled unit, not a per-step query.** N ranks launched together; the
   rank table, peer addresses and the NCCL id are carried in the rank assignment the
   coordinator hands each peer; collectives run in-binary (`candle-core/nccl` = `cudarc/nccl`
   for CUDA; a deterministic in-process and cross-process CPU collective for everything else);
   any rank failure fails the attempt; the existing job lease, reclaim and job-level resume
   checkpoint give restart-from-checkpoint with no new catalog state for ranks.
4. **Not** expressing the SGD step or the gradient exchange as DataFusion operators or
   aggregates. That is the 2014 MLlib `treeAggregate` design; the record (Sparker
   measurements, Project Hydrogen's barrier-mode SPIP, TorchDistributor) shows it failed for
   deep learning and was replaced by gang scheduling plus NCCL.

Distributor-agnosticism is proven last, by hosting a Ballista scheduler and executor role in the
jammi binary and executing the same operators unchanged through a physical extension codec.

## Decisions taken by the user (2026-09-10)

| Fork | Decision |
|---|---|
| Ballista posture | Distributor-agnostic build on DataFusion `ExecutionPlan` + `PhysicalExtensionCodec`; the jammi binary is the worker over its own gRPC surface. A Ballista scheduler-role unit is **in plan, mandatory, last** (gates completion). |
| Hardware proof | CPU collective gang protocol hermetic in CI always. Single-node multi-GPU NCCL proven on a RunPod pod as a committed artifact. Multi-node NCCL proven on a 2-node RunPod instant cluster as a committed artifact. |
| Unit 0 scope | The greenfield rebuild of the training-data path and model identity is **in scope**. |

## Lead rulings on the scope verdict (every ambiguity the gap-analyzer surfaced, resolved)

Each ruling names the principle it derives from. The unit contracts in `UNITS.md` carry them.

1. **Streaming boundary.** The loader becomes a per-epoch stream of `TrainingBatch` read from
   the materialized training-set table by row group, bounded by `batch × prefetch`; every head
   constructor becomes a per-batch converter. A stream that re-buffers into the old `Vec` would
   be a no-op refactor, so that reading is rejected. (Right abstraction; greenfield.)
2. **Multi-epoch semantics.** Epochs re-read the same committed table, identical order every
   epoch, no shuffle (unchanged policy, `docs/plans/64-torch-training-twin/README.md:17`).
   Re-executing the plan per epoch is rejected: a plan over a live source is not a pinned
   data version; the table's artifact digest is. (Feasibility; K7.)
3. **Total order and partition rule.** Materialization orders by the full projected tuple
   (rows with identical tuples are identical, so their relative order is immaterial), then the
   digest pins the order forever. Partition rule v1: global batch t is rows
   `[t·W·B, (t+1)·W·B)` of the committed order; rank r takes rows `[t·W·B + r·B, t·W·B + (r+1)·B)`.
   The union over ranks at step t is exactly the W=1 batch of size W·B at step t. The rule has
   a version tag folded into identity. (Design correctness: makes the trajectory oracle exact.)
4. **Model identity.** Two identities, exactly as result tables have them: the catalog *name*
   `jammi:fine-tuned:{job_id}` stays (it is the re-claim idempotency key, a correct property),
   and the model gains a **materialization** (definition hash + input anchors + artifact
   digest) persisted by migration 029 and written next to the artifact. Cache probe by
   definition hash and anchors reuses a trained model instead of retraining. (K7; K1 arm =
   retrain, since equal-topology training is reproducible by construction.)
5. **Rendezvous supersedes the issue's design.** No rank claims, no rank columns, no
   rendezvous rows. The worker that claims the job is the **coordinator** (rank 0) and holds
   the only lease; it resolves peers from `[training] peers`, mints the NCCL id, and dispatches
   a `RankAssignment` to each peer over the gang gRPC service. Peer death fails the attempt and
   the job requeues for reclaim (existing `attempts` and reclaim machinery); coordinator death
   expires the lease as today; both resume from the job-level resume checkpoint rank 0
   writes. No partial artifact is possible: publish stays a CAS by the coordinator after the
   gang completes. Zero catalog schema for ranks. (Feasibility; K5 crossed once, by 029 only.)
6. **Transport before Ballista.** The gang and the distributed forward are dispatched as
   **descriptors** (typed protobuf in `jammi-wire`), never as serialized physical plans — the
   engine plans no `LogicalPlan` for its compute verbs (`crates/jammi-ai/src/pipeline/asof/exec.rs:4-6`).
   `datafusion-proto` and the extension codec therefore enter only with the Ballista unit,
   where the codec maps the operators to and from the same descriptor messages. The ordering
   "Ballista last" stands on a true dependency graph.
7. **Peer surface shape.** A jammi gRPC service (`jammi.v1.GangService`) with server-streaming
   Arrow IPC for partition exchange, mounted **tenant-scoped** through the existing
   `AssembledChain::mount_tenant_scoped` seam; the coordinator presents the job's tenant as any
   client does. A second Arrow Flight service is rejected: it would collide on the Flight
   service name. Request-level bounds from #485 apply to this surface. (B5; K4 real.)
8. **Operator partitioning.** The distributed forward needs a partition-aware inference
   operator; `InferenceExec` declares one partition today
   (`crates/jammi-ai/src/operator/inference_exec.rs:144`). That change is an explicit
   sub-unit (U6), not "reuse as is".
9. **One code path, two collectives.** A `Collective` trait with `Noop` (W=1), `Local`
   (in-process ranks, deterministic rank-ordered reduce), `Peer` (cross-process over the gang
   service, coordinator-reduce in rank order, exact) and `Nccl` (`candle-core/nccl`, behind the
   existing `cuda` feature). W=1 runs through the same trainer with `Noop`. Topology is
   configuration, never a cfg fork of the training loop. (B4.)
10. **Failure detection.** Per-rank heartbeat on the `RunRank` stream; `[training]
    rank_timeout_secs` (default 120) detects hung ranks; a watchdog aborts the NCCL
    communicator on timeout. Checkpoint anchor = the existing job-level resume checkpoint
    (`{tenant}/{job_id}/_resume/`), written by rank 0 at epoch boundaries (all ranks hold
    identical adapter weights after the all-reduce).
11. **K4 is the remote-equals-embedded invariant.** The issue misused the ID. The real K4
    crossing: W=1 through the gang path must produce adapter bytes identical to the in-process
    trainer. Equal-topology reproducibility (two runs, same W and plan, identical bytes) and
    W-invariance of the trajectory (W ranks × B versus W=1 × W·B, within a tolerance) are two
    NEW oracles, named as such in `DESIGN.md` §6.
12. **Config surface is topology, not a sixth backend.** `[gpu] devices = [..]` (plural
    superset of `device`), `[training] world_size`, `peers`, `rank_timeout_secs`, `collective`;
    per-job `world_size` in the spec (identity-relevant, K7). Placement (which hosts run which
    rank) is the deployer's runtime.
13. **Exactly one migration, 029 `model_materialization`.** The issue's claim that #485 landed
    a jobs table at 029/030 is false in this tree (migrations end at 028; #485 is the open
    request-bounds issue). Corrected in the issue body.
14. **DataFusion 54 upgrade is a unit, first.** Ballista 54.1 pins datafusion ^54,
    arrow-flight ^58, object_store ^0.13, tonic ^0.14; the workspace is on 52.3 / 57 / 0.12 /
    0.14. The third-party crates have compatible releases (`datafusion-federation` 0.5.5 → ^54,
    `datafusion-table-providers` 0.13.1 → ^54, `datafusion-flight-sql-server` 0.4.18 → ^54;
    verified on crates.io 2026-09-10). It lands first so no later unit is written twice. (B6/K6.)
15. **Committed artifact means the repo's proof convention.** A `gpu-gang.yml` lane modeled on
    `gpu-prove.yml` (label, nightly, manual; off the merge path), writing sha-stamped JSON
    artifacts under `crates/jammi-kernels/artifacts/cuda-runs/` that pass
    `ci/scripts/check_cuda_run_artifacts.py`. Evidence of the commit that produced it, never a
    merge gate; the hermetic CPU lane is the standing gate. Box: `NVIDIA A100-SXM4-80GB`
    (matches existing artifacts; cluster availability MEDIUM on 2026-09-10 at $1.59/GPU-hour
    secure). Multi-node: 2 pods × 2 GPUs (W=4, both intra- and inter-node links exercised).
16. **#482 consequence is kept.** A gang needs stable per-rank addresses and ordered startup:
    the Kubernetes compute-tier overlay becomes a StatefulSet with a headless service (or an
    indexed Job), and the reference-topologies page says so. Placement is still the
    deployer's; the engine only consumes `peers`.
17. **Unit split rubric** (SIZING.md): a unit is the smallest change with its own RED-able
    acceptance criterion; units sharing files ship as ordered commits of one PR; a PR boundary
    exists only where a merge is required by construction (the upgrade; a lane that later
    commits' proofs need) or where hardware proofs are gate-adjacent.
18. **This plan is itself a PR** on `feat/500-distributed-training-plan` (docs only).
19. **Naming.** `ci/scripts/check_no_consumer_names.py:78` fails closed on new `pub` items whose
    stem is `stage` or `register`. Identifiers use `gang`, `rank`, `world`, `collective`,
    `exchange`, `partition`, `assignment`, `training_set`; never `Stage*`/`stage_*`/`register_*`.
    Prose may say "stage".
20. **ContextPredictor is out of this plan's gang scope.** It keeps its own path; the two LoRA
    fine-tune kinds (`FineTune`, `GraphFineTune`) unify on the training-set producer. Declared,
    not silent.

## Units and order (SIZING.md has the analysis)

| PR | Unit | Name | Lane | Depends on |
|---|---|---|---|---|
| A | U1 | DataFusion 54 line upgrade (workspace-atomic) | hermetic + cookbook | — |
| B | U2 | Training set as a producer; streaming loader; partition rule | hermetic + cookbook | U1 |
| B | U3 | Model as a producer; migration 029; cache reuse | hermetic | U2 |
| B | U4 | `Collective` trait; device-plural session; single-node gang | hermetic + gpu-gang pod leg | U2 |
| B | U7a | `gpu-gang.yml` pod leg + artifact schema | workflow dry-run | — |
| C | U5 | `GangService`; coordinator; multi-node gang; chaos | distributed nightly + gpu-gang cluster leg | U4 |
| C | U6 | Partition-aware inference operator; distributed frozen forward for the head target | hermetic + distributed | U2, U5 |
| C | U7b | `gpu-gang.yml` cluster leg + reap | workflow dry-run | U7a |
| D | U8 | Ballista scheduler/executor roles + extension codec (mandatory, last) | hermetic (ballista feature) + distributed | U1, U5, U6 |
| D | U9 | Docs: guide + reference topologies (#482 consequence) + maintainer guide | docs gates | all |

Spikes (time-boxed, no PR, premises the pressure-test demanded reproduced before PR-B starts):
S1 `candle-core/nccl` builds under jammi's `cuda` feature and `Comm::from_devices` works with
candle tensors on a 2-GPU pod; S2 a Ballista 54 executor runs a custom `ExecutionPlan` through
`with_ballista_physical_extension_codec` in a scratch crate.

## Hand-off: how a fresh lead kicks this off

1. Read `docs/swarm/CONSTITUTION.md`, `docs/swarm/SELF-FAILURE-MODES.md`, `.claude/agents/lead.md`.
2. Phase 0 ground: seed the ledger from `UNITS.md` (the invariants per unit are listed there).
   Phase 0.5 is done for the plan; run `gap-analyzer` per unit brief only if you change a unit.
3. Run S1 and S2 in parallel with PR-A (they need no code in the tree).
4. PR-A (U1) first, alone. Phase 2 contract is in `UNITS.md`; dispatch `docs-ci` for the shared
   manifests and each crate owner for compile fixes, all in one worktree, one commit.
5. PR-B: U2 → U4 serial (both edit `trainer.rs`/`worker.rs`); U3 concurrent with U4 in its
   own worktree; U7a concurrent (docs-ci). One PR, ordered commits U2, U3, U4, U7a. Label the PR
   to fire the pod leg; commit the artifact.
6. PR-C: U5 and U6 concurrent worktrees (wire-server + ai-core; ai-core operator), U7b
   concurrent (docs-ci); ordered commits U5, U6, U7b. Distributed lane must be green on the
   deterministic leg before merge; chaos leg advisory as today.
7. PR-D: U8 then U9. U8 is the completion gate.
8. Every commit runs phases 3–6.5; every PR runs phase 7. Amend subagent commits with the
   session trailers (memory: subagent trailers).
