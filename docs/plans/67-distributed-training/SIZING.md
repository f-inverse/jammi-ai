# SIZING — how the units were cut, ordered and grouped (#500), v4

v3.1's dimensions D1–D10, alternatives 1–6 and the producer/consumer cuts stand (PRESSURE.md
records them). v4 adds the sibling-plan edges, the substrate change and the Ballista extension.

## Order against the sibling work (measured)

PR #501 (`feat/deploy-shapes-C-jobs` @ 95993a06 → merge `4ecc0230`, 2026-09-10) versus the
previous main 7561658e: 242 files, +21 704 / −7 921 lines, both `Cargo.toml` and `Cargo.lock`,
seven crates. It is merged; every 68 unit cuts after it and so does every 67 unit.

- **PR-A starts now from `main`.** The earlier fork (PR-A before or after PR-C) is closed by the
  merge; had it been open, PR-C first was right — U1 is mechanical and cheap to redo on top,
  a 26-commit branch rebasing onto a workspace-wide API bump is not. 68's PR-K is in CI; PR-A
  and PR-K have no ordering constraint (disjoint files; rebase whichever lands second).
- **Edges into 68**: U5a needs DIST unit 1 (`peer_bind`), OPS (C2 rewrites the claim loop
  `JobSlot` wraps) and GRAPH (rewrites `claim_next`); U5b-2 needs OPS (`release_job_lease`,
  drain hooks); U9b needs K and OPS C6. DIST "unit 2" is a sketch, so U5b-1a builds the
  membership substrate itself (README r28). U2a no longer needs OPS C1 (`job_attempt: None`).
  With #501 merged, K, OPS, GRAPH, DELTA and DIST-1 are all unblocked and run concurrently with
  PR-A and PR-B1. The single cross-plan schedule is `../68-compute-tier-substrate/PROGRAM.md`.
- **Migrations**: 67 appends four (`model_materialization` U3, `instances_peer_addr_result_root`
  U5b-1a, `jobs_assembly_failures_next_after` U5b-1b-ii, `compute_cluster_state` U8b); U5b-0's
  per-row-group leaf digests are a sidecar-object change and append none. No number is
  reserved — each PR takes the next free at rebase and updates both pin sites
  (`catalog/migrations.rs` const list; `tests/it/migrations.rs:23-54` `EXPECTED_MIGRATION_NAMES`)
  plus OPS's relative-position oracle; the second merger renumbers.

```
#501 (merged) → PR-A [U1] ∥ K (#502, in CI) ∥ DIST-1, OPS, GRAPH, DELTA (68)
        → PR-B1 [U7a ∥ U2a ∥ U4a → artifact]                            (needs PR-A only)
        → PR-B2 [U2b ∥ U3 → U2c → U4b → artifact]                       (needs PR-B1; U2c before
                                                                           U4b binds a per-rank
                                                                           reader to it — #544)
        → PR-C(67) [U7b ∥ U5a-1 → U5b-1a → U5b-0 → U5b-1b-i → U5b-1b-ii → U5b-1b-iii → U5a-2
                     → U5b-2 → artifact]                                 (needs DIST-1, OPS, GRAPH,
                                                                           PR-B1; U4b lands before
                                                                           U5b-1b-ii — spec.rs
                                                                           co-ownership)
        → PR-D [U8a → U8b → U9a → U9b]                                  (needs K, OPS; admin merge)
```

PR-B2 and PR-C(67) build concurrently once PR-B1 merges (wave 3); the only cross-branch edge is
the pin above (U4b before U5b-1b-ii).

Serial edges: #501 → U1 → everything; U2a → U2b, U3; U4a → U2b (the `world` argument); U2b +
U4a → U2c (the residency-bounded per-rank stream, before U4b binds a per-rank reader to it,
issue #544); U2c + U3 + U4a → U4b; U5a-1 → U5b-1a (`instance_liveness_margin()`, merge order
pinned); U5a-1 → U5b-1b-i (frozen `RoundContribution`/`RoundResult`); U5b-0 → U5b-1b-i (the leaf
inventory its per-partition verify reads); U4b + U5b-1a + U5b-1b-i + U5a-1 + U5a-2 →
U5b-1b-ii (`spec.rs` co-owned with U4b, merge order pinned); U5b-1b-ii → U5b-1b-iii; U5b-1b-ii +
U5b-1b-iii + OPS → U5b-2; U1 + U5b-1b-ii + U5b-1b-iii + S6 → U8a; U8a + S6 → U8b; all → U9a;
K + OPS → U9b.

## Alternatives added in v4

7. **PR-A before PR-C.** Moot: #501 merged first. Recorded because the measurement decided it.
8. **Static peer list in `[worker]`.** Rejected: 68 DIST D9 already owns membership
   (`peer_advertise` → `instances.peer_addr`); two membership mechanisms would be a knob 67 does
   not need. Cost: U5b waits for DIST unit 2.
9. **Keep U8 as one unit / demote it to a conformance proof.** Rejected: the seams verified at
   source carry the codec + engine + roles (U8a) and the persistent, device-aware cluster state
   (U8b) as two RED-able units, and the user's decision keeps Ballista mandatory. The
   demotion proposal is withdrawn.
10. **`jammi-ballista` as a `jammi-server` module or behind a cargo feature.** Rejected: B4's
    anchor forbids a library-vs-server feature gate; a crate lets a library embedder host the
    roles too; publishable and lockstep like every workspace crate.
11. **Gang ranks as Ballista tasks with all-or-nothing binding.** Rejected for v1: it would be a
    second gang mechanism with its own K4 oracle; the gang is one placed task (README r41).

## Size and owner map (deltas from v3.1)

| Unit | Size | Owners | Note |
|---|---|---|---|
| U4a | L | ai-core + db (+ wire-server co-owner for the spec field) | no migration |
| U2c | M | ai-core + db | streaming training-set loader with a residency bound; issue #544 |
| U5a | L | wire-server + ai-core + db | on the peer listener; DIST-1 merged is a hard precondition |
| U5b-1a | M | db + docs-ci | membership substrate; depends_on U5a-1 |
| U5b-0 | S | db | partitioned attestation inventory (per-row-group leaf digests) |
| U5b-1b-i | L | wire-server + ai-core | `Peer` collective + round protocol |
| U5b-1b-ii | L | wire-server + ai-core + db | coordinator: assembly + dispatch; `spec.rs` co-owned with U4b |
| U5b-1b-iii | L | ai-core | the largest unit in the split — eleven `record_failed` sites × two runner roles |
| U5b-2 | L | ai-core + wire-server | failure semantics + chaos + cluster artifact |
| U8a | L | wire-server + ai-core + docs-ci | new crate + three registration sites; admin merge |
| U8b | L | wire-server + db | the completion gate; neutral migration |
| U9a | M | docs-ci / doc-updater | docs |
| U9b | M | docs-ci | shape-d overlay after K and OPS |
| C1 (cookbook) | S | cookbook | AST session-lifecycle gate; issue #539; after PR-B2 |

Co-ownership (order = merge order): `crates/jammi-ai/src/fine_tune/worker.rs` — **OPS C2 rewrites
the claim loop and GRAPH rewrites `claim_next`; both merge before any 67 unit that touches the
loop** (U5a `JobSlot`, U5b-1b-ii coordinator, U5b-1b-iii rank body, U5b-2 abort path);
U2a/U2b/U2c/U3/U4b edit other regions and may precede them. `crates/jammi-ai/src/session.rs`
(DIST c1/c2 `build_result_store` + `open_with_placement`, OPS C2 release/worker gate, U4a
device-plural session and the `TrainingCommon` sites, U5b-1a `peer_addr` write).
`crates/jammi-db/src/config/mod.rs` (DIST `peer_bind`/`peer_local_load_bytes`, OPS knobs, U4a
`[worker]` fields (renamed `[worker] local_ranks` in U4b S8), U5b-1a `peer_advertise`, U5b-1b-ii
`[distributed] max_world_size`, U8a `[ballista]`). `crates/jammi-server/src/runtime.rs` (DIST
peer routes, OPS C2–C5, U5a, U5b-1b-i decode cap, U8a).
`tenant_isolation_oracle.rs` + `api_freeze_baseline.txt` (DIST PEER bucket, then U5a GANG bucket).
`crates/jammi-server/tests/it/main.rs` mod lines (DIST, OPS, U5a). `crates/jammi-db/src/catalog/
{migrations.rs, tests/it/migrations.rs}` (every migration-appending unit; two pin sites).
`jammi-wire` error-tag space (DIST takes `unavailable = 30`, DELTA 32–36, GRAPH 37–38 — 67's typed refusals
take the next free tags at rebase; `error.proto` + `src/error.rs`). `jobs_repo.rs` (OPS
`release_job_lease`, GRAPH `claim_next`, U5a `get_job_for_rank`, U5b-1a `upsert_instance` /
`list_gang_members`, U5b-1b-ii's cooldown term on `claim_next`'s candidate subselect, U8b
`WorkerRecord.devices`). `deploy/kubernetes/overlays/shape-d/**` (K, OPS
C6, then U9b).

## Spikes

S1, S4, S5 as in v3.1; **S3** on `main` (which carries #501), also recording `cargo tree -d`
for `tonic`/`prost`; **S6** (supersedes S2): a scratch crate on Ballista 54.1 installing
`override_execution_engine`, a custom `ClusterState`/`JobState` through `start_server(cluster,
…)`, and the codec; run a custom `ExecutionPlan` on one scheduler + two executors; confirm
`task_max_failures = 0` disables retry and `expire_dead_executors` only removes executors; kill
and restart the scheduler (state survives); two schedulers over one store; print the executor
identity a custom policy receives and read the stage plan from `active_jobs`. S6's results gate
U8a (engine/codec) and U8b (restart, two-scheduler, identity).
