# PROGRAM — one schedule for plans 67 (training gangs) and 68 (compute-tier substrate)

Owned by one lead since 2026-09-10. Each unit keeps its own contract (`67-distributed-training/UNITS.md`;
`units/*.md` here) and its own pressure record; this file only orders them and names the shared
files. Base: `main` @ `4ecc0230` (PR #501, the jobs fleet, merged). PR-K = #502 (in CI).

## Waves

| Wave | Runs concurrently | Preconditions | Notes |
|---|---|---|---|
| 0 | spikes S1, S3, S4, S5, S6 (67); PR-K #502 CI | — | S3 on `main`; S4 spends money (human-approved); S6 gates U8a and U8b |
| 1 | **PR-A** (67 U1, DataFusion 54) ∥ **K** (#502) ∥ **DELTA** ∥ **GRAPH** ∥ **DIST-1** ∥ **OPS** | #501 | disjoint seams except the migration pin sites and `worker.rs` (OPS C2, GRAPH `claim_next`); rebase whoever lands second |
| 2 | **PR-B1** (67 U7a ∥ U2a ∥ U4a → artifact) | PR-A | independent of every 68 unit; U2a edits `worker.rs` regions OPS C2 does not (`run_spec`) — if OPS lands first, rebase. PR-B2 (U2b, U3, U2c, U4b) finishes during wave 3 below, since U4b now depends_on U2c |
| 3 | **PR-B2 tail** (67 U2b ∥ U3 → **U2c** → U4b → artifact) ∥ **PR-C(67)** (U7b ∥ U5a-1 → U5b-1a → U5b-0 → U5b-1b-i → U5b-1b-ii → U5b-1b-iii → U5a-2 → U5b-2 → artifact) | DIST-1, OPS, GRAPH, PR-B1 | Two merge-order pins: **U5a-1 lands before U5b-1a** (U5a-1 creates `instance_liveness_margin()`; U5b-1a only consumes it) and **U4b lands before U5b-1b-ii** (`spec.rs` admission fields are co-owned; the `[worker] world_size` → `[worker] local_ranks` rename lands in U4b S8, and U5b-1b-ii rebases onto U4b's `admit_and_place`). U2c (streaming loader with a residency bound, issue #544) lands before U4b binds a per-rank reader to it. U5a-1's `JobSlot` wraps the loop OPS C2 rewrites and the `claim_next` GRAPH rewrites; U5b-1a builds the membership substrate DIST §5.8 sketches; U5b-0 (new db unit) carries the row-group attestation inventory U5b-1b-i's per-partition verify reads; U5b-2 uses OPS's `release_job_lease`. **C1** (cookbook, AST session-lifecycle gate, issue #539) runs independently after PR-B2 merges |
| 4 | **PR-D** (U8a → U8b → U9a → U9b) | K, OPS, PR-B2, PR-C(67) | admin merge (domain-card edit); `distributed.yml` three-process arm green before merge (dated correction 2026-09-16: SHIPPED on `feat/500-wave4` as one consolidated PR, contract `docs/rigor/contracts/feat_500-wave4.md`; the actual build order was U8a-cfg + U8b's db slice + U9b concurrently, then U8a's crate and `GangExec` concurrently, then U8b's `CatalogClusterState`/`DevicePlacement`, then U9a last — U9b (the shape-d overlay) is built CONCURRENTLY with U8a, not after U8b/before U9a as this row's arrow reads; U9a also completes the shape-d scheduler role the pressure round's contract §9 B7 decided after U9b's own wave-A build) |
| later | DIST-2 `RendezvousPlacement` (68) on U5b-1a's substrate; DIST-3 `datafusion-distributed` behind the DF-55 gate | PR-C(67); flight-sql-server on DF 55 | 68's, unchanged |

A lead-gate anticipation proposal runs concurrently, independent of every wave above: it amends
the swarm's own gate mechanism (`SWARM_GATE_TOUCHED`), so it lands as its own human-merged PR,
never folded into a wave's admin merge.

## Shared files (merge order inside each)

- `crates/jammi-ai/src/fine_tune/worker.rs`: OPS C2 (loop) and GRAPH (`claim_next`) before
  U5a-1/U5b-1b-*; U2c and U4b before U5b-1b-ii/iii (the coordinator/rank-body split reuses the
  run path); other 67 edits are region-disjoint.
- `crates/jammi-ai/src/session.rs`: DIST-1 (`build_result_store`, `open_with_placement`), OPS C2,
  then U4a, U5b-1a (`upsert_instance`'s `peer_addr` call site).
- `crates/jammi-db/src/config/mod.rs`: DIST-1 (`peer_bind`, `peer_local_load_bytes`), OPS knobs,
  U4a `[worker]` fields (renamed `[worker] local_ranks` in U4b S8), U5b-1a `peer_advertise`,
  U5b-1b-ii `[distributed] max_world_size`, U8a `[ballista]`.
- `crates/jammi-server/src/runtime.rs`: DIST-1 peer routes, OPS C2–C5, U5a, U5b-1b-i (the
  peer-only listener's decode cap), U8a.
- `crates/jammi-server/tests/it/{tenant_isolation_oracle.rs, api_freeze_baseline.txt, main.rs}`: DIST-1 PEER bucket, then U5a GANG bucket; mod lines DIST/OPS/U5a.
- `crates/jammi-db/src/catalog/{migrations.rs, ../../tests/it/migrations.rs}`: every migration-appending unit (OPS, GRAPH, DELTA, U3 — the standard two pin sites), **U5b-1a**
  `instances_peer_addr_result_root`, **U5b-1b-ii** `jobs_assembly_failures_next_after`, and U8b
  `compute_cluster_state` (each three pin sites: the standard two PLUS an ordered-after oracle —
  U8b's asserts ordered-after BOTH of PR-C(67)'s migrations, since PR-D lands after PR-C(67)
  merges) — next free number at rebase, second merger renumbers. U5b-0's leaf-digest
  inventory is sidecar-object-only and appends no migration.
- `crates/jammi-wire/proto/jammi/v1/error.proto` + `src/error.rs`: DIST-1 `unavailable = 30`, DELTA 32–36, GRAPH 37–38, then 67's typed refusals at the next free tags.
- `crates/jammi-db/src/catalog/jobs_repo.rs`: OPS `release_job_lease`, GRAPH `claim_next`, U5a
  `get_job_for_rank`, U5b-1a `upsert_instance`/`list_gang_members`, U5b-1b-ii `claim_next`'s
  cooldown term (candidate subselect, the SAME backend clock as leases) and the assembly
  counter, U8b `WorkerRecord.devices`.
- `deploy/kubernetes/overlays/shape-d/**`: K, OPS C6, then U9b (StatefulSet, `nvidia.com/gpu: N`).

## Ballista, stated once

68 DIST D2 withdrew Ballista **for the data plane**; that stands. 67 adopts it **for the compute/gang
plane** as a seam-level extension in `crates/jammi-ballista` (codecs, `override_execution_engine`,
`ClusterState`/`JobState` implementations over the catalog, `TaskDistributionPolicy::Custom`,
retries off), the way `jammi-kernels` extends candle — never a fork. D2's conditions: (1) the
accelerator dimension has no seam and is carried out of band (`workers.devices`) — the one
upstream PR owed; (2) pluggable cluster state exists as public traits; (3) object-store shuffle
stays unproven and out of v1. Full record: `67-distributed-training/DESIGN.md` §9 and README
r38–r45.

## Issues

- #500: body rewritten 2026-09-10; the v4 comment supersedes the withdrawn "planned training"
  draft in `ISSUE-COMMENTS.md`.
- #482: the draft in `ISSUE-COMMENTS.md` posts with its Ballista paragraph replaced by the
  section above; the compute tier's provisional Deployment becomes a StatefulSet in U9b.
