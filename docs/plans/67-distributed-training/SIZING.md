# SIZING — how the units were cut, ordered and grouped (#500)

The user asked for the sizing itself to go through the rigor chain. This file records the
dimensions, the alternatives, the choice, and what the pressure-test changed (PRESSURE.md).

## Dimensions

| # | Dimension | What it decides |
|---|---|---|
| D1 | Seam overlap (files, crates) | PR grouping: units sharing files ship as ordered commits of one PR; never split a file across PRs (memory: consolidate PRs). |
| D2 | Independent RED oracle | Unit boundary: a unit is the smallest change whose acceptance is RED at base and GREEN after, asserting a criterion. |
| D3 | Lane | Hermetic / cookbook / distributed nightly / gpu-gang pod / gpu-gang cluster. A unit whose proof needs a lane depends on the lane unit. |
| D4 | Dependency edges | Topological order; waves. |
| D5 | Risk-first | Premises that could kill the design are reproduced before the expensive units (spikes S1, S2). |
| D6 | Bisectability | One commit per unit; each commit passes the full gate. |
| D7 | CI iteration cost | Few PRs (each PR costs merge-in regen commits and slow CI). |
| D8 | Concurrency | Seam-disjoint units run in separate worktrees by different owners. |
| D9 | Hardware cost and availability | GPU legs label-triggered; cluster leg only after the pod leg is green; A100 cluster stock MEDIUM on 2026-09-10. |
| D10 | Greenfield blast radius | Identity changes are breaking for existing catalogs; migrations stay append-only (K5); no compatibility shims. |

## Alternatives considered

1. **One PR for everything.** Rejected on D3/D4: the DataFusion 54 upgrade must merge before
   the operator work or every later unit is written against two API lines; the cluster leg's
   proof depends on the pod leg's lane existing on main; a single PR would carry a month of
   commits with no merge-in resets.
2. **One PR per unit (nine PRs).** Rejected on D7 and D1: U2/U3/U4 all edit `trainer.rs` and
   `worker.rs`; U5/U6 both edit `worker.rs` and the harness; nine PRs multiply regen commits
   and CI hours with no isolation gained.
3. **Ladder-per-PR (multi-GPU PR, multi-node PR, Ballista PR).** Close, but it puts the
   training-set and identity rebuild (U2/U3) either into the multi-GPU PR (correct) or alone
   (an extra PR with no independent lane). Chosen grouping is ladder-per-PR with U2/U3 folded
   into the multi-GPU PR because the gang oracles need the pinned training set to mean anything.
4. **Gang first, identity later.** Rejected on D2: without the pinned training set and the
   identity hash, "identical bytes for equal world size and plan" has no definition of "plan".
5. **Skip the DataFusion upgrade until Ballista.** Rejected on D4: U6 changes operator
   properties against the DataFusion API; doing it on 52 then re-porting to 54 is the double
   write D1 forbids. The upgrade is mechanical and isolated; it goes first.
6. **CPU collective as the only hermetic proof, NCCL unproven until the cluster.** Rejected on
   D5: the single-process `Comm::from_devices` path is the cheapest NCCL proof and catches the
   candle/cudarc integration risk before the multi-node work; hence S1 and the pod leg in PR-B.

## The schedule

```
PR-A  [U1]                                      alone; S1, S2 run concurrently (no PR)
PR-B  [U2] -> [U4] ; [U3] ∥ U4 ; [U7a] ∥ U2..U4    ordered commits: U2, U3, U4, U7a
PR-C  [U5] ∥ [U6] ; [U7b] ∥ U5                     ordered commits: U5, U6, U7b
PR-D  [U8] -> [U9]                               U8 is the completion gate
```

Serial edges: U1 → everything (API line); U2 → U4 (partition rule and loader are the gang's
input); U4 → U5 (the `Collective` trait and rank context are the peer transport's substrate);
U5 → U6's fan-out (`FetchPartition`), though U6's operator change can start on U2 alone;
U5 + U6 → U8.

Concurrency: within PR-B, U3 (db-owned manifest, migration, publish) and U4 (ai-core-owned
trainer, collective, session) overlap only in `worker.rs` — the lead assigns U3 the
`publish_and_finalize` region and U4 the `run_spec`/rank-spawn region and merges U3 first.
Within PR-C, U5 (wire-server + worker coordinator) and U6 (operator + embedding pipeline) are
seam-disjoint except `worker.rs`'s head-target arm, assigned to U6.

## Size and owner map

| Unit | Size | Owners | Worktree |
|---|---|---|---|
| U1 | L | docs-ci (manifests) + every crate owner for its fixes | one shared worktree, one commit |
| U2 | XL | db + ai-core | one worktree (serial handoff db → ai-core) |
| U3 | M | db + ai-core | own worktree |
| U4 | L | ai-core (+ db for config) | own worktree |
| U5 | XL | wire-server + ai-core | own worktree |
| U6 | M | ai-core (+ db for sink) | own worktree |
| U7a/b | M/M | docs-ci | own worktree |
| U8 | L | wire-server + ai-core + docs-ci | own worktree |
| U9 | S | docs-ci / doc-updater | own worktree |

XL units are XL because the loader rewrite (U2) touches every head constructor and the gang
service (U5) spans proto, server, coordinator, harness and lane. They are not split further
because their acceptance criteria are not separable: a loader that streams for some heads is
not an acceptance, and a gang service without a coordinator has no RED oracle.

## What must be true before PR-B starts (spike results recorded in the ledger)

- S1: on a 2-GPU pod, a scratch binary with `candle-core = { features = ["cuda", "nccl"] }`
  runs `Comm::from_devices` over two candle CUDA devices and an `all_reduce_in_place` on a
  candle tensor's storage; result matches the CPU sum bit-for-bit for f32 sums of two operands.
- S2: a scratch crate on Ballista 54.1 registers a `PhysicalExtensionCodec` via
  `with_ballista_physical_extension_codec` and executes a trivial custom `ExecutionPlan` on an
  executor process; task retry attempts can be set to 0 per job.
