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
- **Edges into 68**: U5a needs DIST unit 1 (`peer_bind`); U5b needs DIST unit 2
  (`instances.peer_addr`) and OPS (`release_job_lease`, drain hooks); U9 needs K (the shape-d
  overlay). U2a no longer needs OPS C1 (`job_attempt: None`).
- **Migrations**: 67 appends three (`model_materialization` U3, `workers_devices` U4a,
  `ballista_state` U8b), numbered at rebase after 68's five; K5 renumber-on-second-merge.

```
#501 (merged) → PR-A [U1] ∥ K (in CI) → DIST-1 (68) → PR-B [U7a ∥ U2a ∥ U4a → U2b ∥ U3 → U4b → artifact]
        → DIST-2, OPS (68) → PR-C(67) [U7b ∥ U5a → U6 → U5b → artifact]
        → PR-D [U8a → U8b → U9]           (DELTA, GRAPH (68) are independent of 67)
```

Serial edges: #501 → U1 → everything; U2a → U2b, U3; U4a → U2b (the `world` argument);
U2b + U3 + U4a → U4b; U4a + DIST-1 → U5a; U2b + U5a → U6; U4b + U5a + U6 + DIST-2 + OPS → U5b;
U1 + U5b + U6 + S6 → U8a; U8a + U4a → U8b; K → U9.

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
| U4a | L | ai-core + db (+ wire-server co-owner for the spec field) | adds the `workers_devices` migration |
| U5a | L | wire-server + ai-core | on the peer listener; carries DIST's listener commit only if DIST-1 is unmerged |
| U5b | XL | ai-core + wire-server + db | membership, released-vs-failed |
| U8a | L | wire-server + ai-core + docs-ci | new crate |
| U8b | L | wire-server + db | the completion gate |
| U9 | M | docs-ci / doc-updater | shape-d overlay after K |

Co-ownership additions: `crates/jammi-db/src/config/mod.rs` (68 DIST `peer_bind`, then 67 U4a
`[worker]` fields, then U8a `[ballista]`); `crates/jammi-server/src/runtime.rs` (DIST peer routes,
then U5a, then U8a); `crates/jammi-server/tests/it/tenant_isolation_oracle.rs` and
`api_freeze_baseline.txt` (DIST's PEER bucket, then U5a's GANG bucket);
`deploy/kubernetes/overlays/shape-d/**` (K, then U9); `crates/jammi-db/src/catalog/jobs_repo.rs`
(OPS `release_job_lease`, then U5b reads, then U4a `WorkerRecord.devices`).

## Spikes

S1, S3 (on top of PR-C), S4, S5 as in v3.1; **S6** (supersedes S2): a scratch crate on Ballista
54.1 installing `override_execution_engine`, a custom `ClusterState`/`JobState` through
`start_server(cluster, …)`, and the codec; run a custom `ExecutionPlan` on one scheduler + two
executors; confirm `task_max_failures = 0` disables retry and that `expire_dead_executors` only
removes executors. Results in the ledger before U8a is briefed.
