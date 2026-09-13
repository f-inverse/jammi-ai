# PROGRAM — one schedule for plans 67 (training gangs) and 68 (compute-tier substrate)

Owned by one lead since 2026-09-10. Each unit keeps its own contract (`67-distributed-training/UNITS.md`;
`units/*.md` here) and its own pressure record; this file only orders them and names the shared
files. Base: `main` @ `4ecc0230` (PR #501, the jobs fleet, merged). PR-K = #502 (in CI).

## Waves

| Wave | Runs concurrently | Preconditions | Notes |
|---|---|---|---|
| 0 | spikes S1, S3, S4, S5, S6 (67); PR-K #502 CI | — | S3 on `main`; S4 spends money (human-approved); S6 gates U8a and U8b |
| 0.5 | **fix/500-release-claim-spin** | — | its own PR, before PR-B1: a pre-existing RELEASE defect on `main` (a claim→self-release spin in `release_and_stop`'s window, found by U2a's flake investigation) — PR-B1 rebases onto it |
| 1 | **PR-A** (67 U1, DataFusion 54) ∥ **K** (#502) ∥ **DELTA** ∥ **DIST-1** ∥ **OPS** | #501 | GRAPH is deferred to #515, not scheduled in this wave; disjoint seams except the migration pin sites and `worker.rs` (OPS C2 — GRAPH's `claim_next` rewrite is not part of this wave, so `worker.rs` orders only against OPS C2); rebase whoever lands second |
| 2 | **PR-B1** (67 U7a ∥ U2a ∥ U4a) | PR-A, fix/500-release-claim-spin | independent of every 68 unit; U2a/U4a edit `worker.rs` regions OPS C2 does not (`run_spec`) — if OPS lands first, rebase; merges to `main` right after `fix/500-release-claim-spin` |
| 2.5 | **PR-B2** (67 U2b ∥ U3 → U4b → artifact) | PR-B1 | cut from PR-B1's merge; U2b/U3/U4b edit `worker.rs` regions OPS C2 does not (`run_spec`, `publish_and_finalize`, head-target arm) — if OPS lands first, rebase; U4b needs U2b's `PartitionSpec` and a two-GPU pod leg |
| 3 | **PR-C(67)** (U7b ∥ U5a → U6 → U5b-1 → U5b-2 → artifact) | DIST-1, OPS, PR-B1 | GRAPH is deferred to #515 and is NOT a precondition: `claim_next` is unrewritten in this wave, so U5a's `JobSlot` wraps only the loop OPS C2 rewrites; U5b-1 builds the membership substrate DIST §5.8 sketches; U5b-2 uses OPS's `release_job_lease`. Wave-3 base is `main` after PR-B1 for U5a-1 (the `JobSlot`/authorization acceptance items, none of which run a multi-rank job) and U7b; U5a-2 (any acceptance item its own re-attack finds does need the multi-rank run path) additionally needs PR-B2 |
| 4 | **PR-D** (U8a → U8b → U9a → U9b) | K, OPS, PR-C(67) | admin merge (domain-card edit); `distributed.yml` three-process arm green before merge |
| later | DIST-2 `RendezvousPlacement` (68) on U5b-1's substrate; DIST-3 `datafusion-distributed` behind the DF-55 gate | PR-C(67); flight-sql-server on DF 55 | 68's, unchanged |

A lead-gate anticipation proposal runs concurrently, independent of every wave above: it amends
the swarm's own gate mechanism (`SWARM_GATE_TOUCHED`), so it lands as its own human-merged PR,
never folded into a wave's admin merge.

## Shared files (merge order inside each)

- `crates/jammi-ai/src/fine_tune/worker.rs`: OPS C2 (loop) before U5a/U5b-*; other 67 edits are region-disjoint. GRAPH's `claim_next` rewrite is deferred to #515 and is not part of this ordering.
- `crates/jammi-ai/src/session.rs`: DIST-1 (`build_result_store`, `open_with_placement`), OPS C2, then U4a, U5b-1.
- `crates/jammi-db/src/config/mod.rs`: DIST-1 (`peer_bind`, `peer_local_load_bytes`), OPS knobs, U4a `[worker]` fields, U5b-1 `peer_advertise`, U8a `[ballista]`.
- `crates/jammi-server/src/runtime.rs`: DIST-1 peer routes, OPS C2–C5, U5a, U8a.
- `crates/jammi-server/tests/it/{tenant_isolation_oracle.rs, api_freeze_baseline.txt, main.rs}`: DIST-1 PEER bucket, then U5a GANG bucket; mod lines DIST/OPS/U5a.
- `crates/jammi-db/src/catalog/{migrations.rs, ../../tests/it/migrations.rs}`: every migration-appending unit (OPS, GRAPH, DELTA, U3, U5b-1, U8b) — next free number at rebase, both pin sites, second merger renumbers.
- `crates/jammi-wire/proto/jammi/v1/error.proto` + `src/error.rs`: DIST-1 `unavailable = 30`, DELTA 32–36, GRAPH 37–38, then 67's typed refusals at the next free tags.
- `crates/jammi-db/src/catalog/jobs_repo.rs`: OPS `release_job_lease`, U5a `get_job_for_rank`, U5b-1 `upsert_instance`/`list_gang_members`, U8b `WorkerRecord.devices`. GRAPH's `claim_next` rewrite is deferred to #515 and does not order against these.
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
