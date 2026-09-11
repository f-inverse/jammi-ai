# PROGRAM — one schedule for plans 67 (training gangs) and 68 (compute-tier substrate)

Owned by one lead since 2026-09-10. Each unit keeps its own contract (`67-distributed-training/UNITS.md`;
`units/*.md` here) and its own pressure record; this file only orders them and names the shared
files. Base: `main` @ `4ecc0230` (PR #501, the jobs fleet, merged). PR-K = #502 (in CI).

## Waves

| Wave | Runs concurrently | Preconditions | Notes |
|---|---|---|---|
| 0 | spikes S1, S3, S4, S5, S6 (67); PR-K #502 CI | — | S3 on `main`; S4 spends money (human-approved); S6 gates U8a and U8b |
| 1 | **PR-A** (67 U1, DataFusion 54) ∥ **K** (#502) ∥ **DELTA** ∥ **GRAPH** ∥ **DIST-1** ∥ **OPS** | #501 | disjoint seams except the migration pin sites and `worker.rs` (OPS C2, GRAPH `claim_next`); rebase whoever lands second |
| 2 | **PR-B** (67 U7a ∥ U2a ∥ U4a → U2b ∥ U3 → U4b → artifact) | PR-A | independent of every 68 unit; U2a/U2b/U3/U4b edit `worker.rs` regions OPS C2 does not (`run_spec`, `publish_and_finalize`, head-target arm) — if OPS lands first, rebase |
| 3 | **PR-C(67)** (U7b ∥ U5a → U6 → U5b-1 → U5b-2 → artifact) | DIST-1, OPS, GRAPH, PR-B | U5a's `JobSlot` wraps the loop OPS C2 rewrites and the `claim_next` GRAPH rewrites; U5b-1 builds the membership substrate DIST §5.8 sketches; U5b-2 uses OPS's `release_job_lease` |
| 4 | **PR-D** (U8a → U8b → U9a → U9b) | K, OPS, PR-C(67) | admin merge (domain-card edit); `distributed.yml` three-process arm green before merge |
| later | DIST-2 `RendezvousPlacement` (68) on U5b-1's substrate; DIST-3 `datafusion-distributed` behind the DF-55 gate | PR-C(67); flight-sql-server on DF 55 | 68's, unchanged |

## Shared files (merge order inside each)

- `crates/jammi-ai/src/fine_tune/worker.rs`: OPS C2 (loop) and GRAPH (`claim_next`) before U5a/U5b-*; other 67 edits are region-disjoint.
- `crates/jammi-ai/src/session.rs`: DIST-1 (`build_result_store`, `open_with_placement`), OPS C2, then U4a, U5b-1.
- `crates/jammi-db/src/config/mod.rs`: DIST-1 (`peer_bind`, `peer_local_load_bytes`), OPS knobs, U4a `[worker]` fields, U5b-1 `peer_advertise`, U8a `[ballista]`.
- `crates/jammi-server/src/runtime.rs`: DIST-1 peer routes, OPS C2–C5, U5a, U8a.
- `crates/jammi-server/tests/it/{tenant_isolation_oracle.rs, api_freeze_baseline.txt, main.rs}`: DIST-1 PEER bucket, then U5a GANG bucket; mod lines DIST/OPS/U5a.
- `crates/jammi-db/src/catalog/{migrations.rs, ../../tests/it/migrations.rs}`: every migration-appending unit (OPS, GRAPH, DELTA, U3, U5b-1, U8b) — next free number at rebase, both pin sites, second merger renumbers.
- `crates/jammi-wire/proto/jammi/v1/error.proto` + `src/error.rs`: DIST-1 `unavailable = 24`, DELTA 24–28, then 67's typed refusals at the next free tags.
- `crates/jammi-db/src/catalog/jobs_repo.rs`: OPS `release_job_lease`, GRAPH `claim_next`, U5a `get_job_for_rank`, U5b-1 `upsert_instance`/`list_gang_members`, U8b `WorkerRecord.devices`.
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
