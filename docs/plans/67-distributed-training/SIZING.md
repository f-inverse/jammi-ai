# SIZING — how the units were cut, ordered and grouped (#500), v2

The user asked for the sizing itself to go through the rigor chain. This file records the
dimensions, the alternatives, the choice, and what the sizing pressure-test changed (PRESSURE.md).

## Dimensions

| # | Dimension | What it decides |
|---|---|---|
| D1 | Seam overlap (files, crates) | Units sharing files ship as ordered commits of one PR; never split a file across PRs. |
| D2 | Independent RED oracle | A unit is the smallest change whose acceptance is RED at base and GREEN after, asserting a criterion. The producer/consumer seam is a valid cut. |
| D3 | Lane | Hermetic / cookbook / distributed (manual dispatch, nightly) / gpu-gang pod / gpu-gang cluster. A unit whose proof needs a lane is committed after the lane unit. |
| D4 | Dependency edges | Topological order; waves. |
| D5 | Risk-first | Premises that could kill the design are reproduced by spikes before the expensive units. |
| D6 | Bisectability | One commit per unit; each commit passes the full gate. Artifacts produced by a lane land in a follow-up commit of the lane unit, never inside the compute unit's commit. |
| D7 | CI iteration cost | Few PRs. |
| D8 | Concurrency | Seam-disjoint units run in separate worktrees by different owners; no unit hides a serial handoff inside itself. |
| D9 | Hardware cost and availability | GPU legs label-triggered; cluster leg after the pod leg is green; A100 cluster stock MEDIUM on 2026-09-10. |
| D10 | Greenfield blast radius | Identity changes are breaking for existing catalogs; migrations append-only (K5); no shims. |

## Alternatives considered

1. **One PR for everything.** Rejected: a long-lived shared base branch would force every
   unit's worktree off `main` for the whole program (D8), and the hardware artifacts would
   have nowhere to land between proofs (D6). (The earlier "two API lines" reason was wrong —
   ordered commits already prevent that — and is withdrawn.)
2. **One PR per unit (thirteen PRs).** Rejected on D7 and D1: U2b/U3/U4b share `trainer.rs`,
   `worker.rs` and `manifest.rs`; U5b/U6 share `worker.rs` and the harness.
3. **Ladder-per-PR** with the training-set and identity rebuild folded into the multi-GPU PR.
   Chosen. Honest reasons per boundary: PR-A because a base branch carrying a workspace-wide
   API bump must not be the parent of every other worktree; PR-B and PR-C on reviewability
   (each ≈ 6–7 commits) plus hardware-proof adjacency (the pod leg proves PR-B, the cluster leg
   proves PR-C); PR-D because U8 introduces a new optional dependency tree (Ballista crates,
   `deny.toml`, a new clippy lane) best reviewed in isolation and is the user's completion gate.
   A lead who prefers may fold PR-D into PR-C; nothing requires the split by construction.
4. **Gang first, identity later.** Rejected on D1 (double write of `trainer.rs`/`worker.rs`),
   not on oracle definability: U4a's oracles need no loader and are scheduled early; U4b's
   need the partition rule, which is U2b.
5. **Skip the DataFusion upgrade until Ballista.** Rejected: arrow 57 → 58 crosses every
   `RecordBatch` site U2 and U6 touch (`worker.rs:1620-1762`, the operator layer); the upgrade
   is isolated and goes first, sized by S3.
6. **No cut inside U2/U4/U5.** Rejected by the sizing pressure-test: the producer/consumer
   seam (U2a/U2b), the trait/rank-context seam (U4a/U4b) and the service/coordinator seam
   (U5a/U5b) each yield a compiling, gate-passing, bisectable intermediate with its own RED
   oracle and restore real concurrency (U4a ∥ U2a; U6 ∥ U5b).

## The schedule

```
PR-A  [U1]                                            S3 sizes it; S1, S2, S4, S5 run concurrently (no PR)
PR-B  U7a ∥ U2a ∥ U4a  →  U2b ∥ U3  →  U4b  →  artifact      commits: U7a, U2a, U4a, U2b, U3, U4b, artifact
PR-C  U7b ∥ U5a  →  U6 ∥ U5b  →  artifact                    commits: U7b, U5a, U6, U5b, artifact
PR-D  U8 → U9                                                  U8 is the completion gate
```

Serial edges: U1 → everything (API line); U2a → U2b, U3; U2b + U3 + U4a → U4b; U4a → U5a;
U4b + U5a → U5b; U2b + U5a → U6; U5b + U6 → U8.

Co-ownership recorded: `manifest.rs` (U2a, U3, U4b — U3's completeness test is extended by
U4b), `worker.rs` (U2a/U2b own `run_spec`; U3 owns `publish_and_finalize`; U5b owns the
coordinator region; U6 owns the head-target arm at `run_fine_tune_blocking`), `runpod_lib.sh`
(U7a then U7b), `config/mod.rs` (U4a then U5a).

## Size and owner map

| Unit | Size | Owners | Worktree |
|---|---|---|---|
| U1 | L (XL if S3 needs API rework) | docs-ci + db + every crate owner for fixes | one shared worktree, one commit |
| U7a | L | docs-ci | own |
| U2a | M | db + ai-core | own |
| U4a | L | ai-core (+ db config) | own |
| U2b | XL | ai-core (+ db reader) | own |
| U3 | M | db + ai-core | own |
| U4b | L | ai-core | own |
| U7b | M | docs-ci | own |
| U5a | L | wire-server (+ db config) | own |
| U6 | M | ai-core (+ db sink) | own |
| U5b | XL | ai-core + wire-server | own |
| U8 | L | wire-server + ai-core + docs-ci | own |
| U9 | S | docs-ci / doc-updater | own |

U2b and U5b remain XL because their acceptance criteria are not separable further: a loader
that streams for some heads is not an acceptance; a coordinator without the chaos leg has no
failure oracle.

## Spikes (results in the ledger before the dependent unit is briefed)

- **S1** (→ U4a): `candle-core = { features = ["cuda", "nccl"] }` on a 2-GPU pod;
  `Comm::from_devices` over two candle CUDA devices; `all_reduce_in_place` on a candle tensor's
  storage equals the CPU sum for f32 sums of two operands.
- **S2** (→ U8): a scratch crate on Ballista 54.1 registers a `PhysicalExtensionCodec` via
  `with_ballista_physical_extension_codec`, executes a custom `ExecutionPlan` on an executor
  process, and sets task retry attempts to 0 per job.
- **S3** (→ U1): a throwaway workspace compile on datafusion 54 / arrow 58 / object_store 0.13
  including `-p jammi-db --features postgres,mysql` and `pyo3-arrow`; record the delta.
- **S4** (→ U7a/U7b): one 2-GPU pod through a `gpuCount: 2` variant of `rp_deploy_live`; a
  create-cluster reachability and teardown probe (≈ one $3 pod-hour + cluster minutes).
- **S5** (→ U4b/U5b GPU oracles): CUDA bit-reproducibility with the flash deterministic path
  forced, `CUBLAS_WORKSPACE_CONFIG` set and NCCL channels pinned; decides whether GPU byte
  oracles are promoted from digest-pair records.
