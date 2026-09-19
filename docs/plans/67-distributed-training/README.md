# 67 — Multi-GPU and multi-node training as engine mechanism

A training job declares a world size, N ranks train one job cooperatively, gang failure
semantics are explicit, identity and parity are scoped, and placement stays the deployer's
runtime. The design is in [`DESIGN.md`](./DESIGN.md); this page states the position, what is
built, and what is deliberately not.

## The position

1. **The training set is a plan and a producer** — a content-addressed result table. Sharding
   is partitioning of its train prefix by a fixed rule; epochs re-read it with bounded memory.
2. **A trained model is a producer** — its definition folds the training-set digest, the
   base-model identity, the canonical spec, the seed, the topology and the kernel-admission
   profile. Replay is retrain; an equal definition over pinned inputs is a cache hit.
3. **The gang is a co-scheduled unit, not a per-step query** — every rank gathers the global
   batch's representations and computes the identical global loss, so batch-coupled objectives
   keep their semantics; adapter gradients are summed; collectives run in-binary behind one
   trait; any rank failure fails the attempt; the jobs fleet's lease, reclaim and resume
   checkpoint give restart.
4. **Not** the SGD step or gradient exchange as DataFusion operators or aggregates.

Distributor-agnosticism is shown by `jammi-ballista`, a workspace crate that extends Ballista
at the seams it exposes and executes the engine's operators unchanged.

## What is built

| Capability | Where |
|---|---|
| Training set as a materialised, replayable result table; eager and residency-bounded streaming readers; one ordered reader class | `crates/jammi-ai/src/fine_tune/training_set.rs`, `stream.rs` |
| Fine-tune as a producer; the model artifact as a catalog entity that N model rows may reference; cache reuse of an equal definition; byte deletion only under a reclaim licence | `crates/jammi-db/src/catalog/artifact_repo.rs`, `crates/jammi-ai/src/fine_tune/worker.rs` |
| The `Collective` trait and its `noop` / `local` / `peer` / NCCL implementations; the gather rule; lockstep control flow | `crates/jammi-ai/src/fine_tune/collective/` |
| Single-host gangs (`[worker] local_ranks`) and multi-host `Peer` gangs (`[distributed] max_world_size`), for `fine_tune` and `graph_fine_tune` | `crates/jammi-ai/src/fine_tune/worker.rs`, `crates/jammi-server/src/grpc/gang.rs` |
| Membership from the catalog (`peer_advertise`, root identity), host admission, the attempt fence, the two-phase round protocol, the assembly-outcome table, released-versus-failed settlement, the per-attempt watchdog | `DESIGN.md` §4 |
| Partitioned attestation inventory (per-row-group leaf digests) | `crates/jammi-db/src/store/manifest.rs` |
| Partitioned inference: one plan shape in one process and across a cluster | `crates/jammi-ai/src/operator/inference_exec.rs`, `numbered_input_exec.rs` |
| The Ballista compute plane: codecs, execution engine, catalog-backed cluster and job state, device placement, a gang as one placed task | `crates/jammi-ballista/` |
| The `shape-d` Kubernetes topology: scheduler `Deployment`, compute `StatefulSet` with a headless service | `deploy/kubernetes/overlays/shape-d/` |

Hardware proof runs in three legs (`DESIGN.md` §10): the CPU collective hermetically in CI; a
single-node two-GPU gang on a rented pod; a two-host gang over a private network. Their
committed evidence is under `crates/jammi-kernels/artifacts/cuda-runs/`
(`*-gang-pod-*.json`, `*-gang-cluster-*.json`).

## What is not

Declared non-goals (`DESIGN.md` §8): sharded model or optimizer state; elastic gangs;
`context_predictor` on a gang; hard-negative mining and gradient caching at `world_size > 1`
(a typed refusal).

Blocked on upstream releases and tracked in one place
([#613](https://github.com/f-inverse/jammi-ai/issues/613)): moving the workspace to the
DataFusion 55 line, and distributed SQL through `datafusion-distributed`. Ballista's executor
specification has no accelerator dimension; `jammi-ballista` carries device inventory in the
engine's own catalog (`compute_executors.devices`) and places device-bound tasks with its own
distribution policy — an extension at the seam Ballista exposes, the way `jammi-kernels`
extends candle, never a fork or an upstream request. Two unrelated backlog items are parked,
not pending: Metal/f16 acceleration (#445) and a Kafka-protocol trigger broker (#478).
