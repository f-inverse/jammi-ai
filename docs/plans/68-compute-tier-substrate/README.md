# 68 — Compute-tier substrate: deployment shapes, distributed data plane, incremental embedding, job dependencies, operability (#482 follow-ups)

**Status:** PLANNED, five units each at PROCEED after plan + pressure-test (2026-09-10), NOT implemented.
PR-C (the jobs fleet) **merged as #501** (`4ecc0230`, 2026-09-10); PR-K is **#502, in CI**. Owned
together with `67-distributed-training/` since 2026-09-10; the single cross-plan schedule is
`PROGRAM.md` in this directory. Multi-GPU / multi-node training is 67's (out of scope here).
This directory is the hand-off artifact for the lead that implements it in a fresh session: read this file,
then the unit plan you are cutting under `units/`, then its verdicts under `pressure/verdicts/` and the
measurement scripts under `pressure/measurements/`. The per-round rulings are in the repo ledger
`.jammi/ledger/deploy-shapes-20260910-collate.jsonl` (73 rows, `unit` field = K / DIST / DELTA / GRAPH / OPS).

**Posture:** greenfield. Every fork was resolved by deriving from `docs/guide/src/philosophy.md`
and `docs/swarm/CONSTITUTION.md` and from solid outside references (each plan's References
section), never by asking. Every plan is a consolidated, stated-once document: no amendment
layers, every `path:line` re-derived on the named tree at the named sha.

**Trees.** The plans' `wt-C` citations were derived on `feat/deploy-shapes-C-jobs` @ 95993a06,
which merged into `main` as #501 (`4ecc0230`); the merge added three cookbook files only, so
every cited `crates/`, `ci/` and `docs/` line holds on `main` at the same number. All five units
cut from `main` now. DIST §5.8 "unit 2" (membership) is **built by 67's U5b-1** (see PROGRAM.md);
DIST's `RendezvousPlacement` builds on it.

## Units (`units/`)

| Unit | File | Outcome | Rounds |
|---|---|---|---|
| K | `K-KUBERNETES.md` | kustomize base + overlays; base = engine default (`[worker] enabled = true`), shape-d replaces the ConfigMap; nothing sized (BestEffort, no grace; only `nvidia.com/gpu: 1` on compute); `:latest` kept with the honest floor note (cu12 `:latest` == 0.49.1, predates `[worker]`); downward-API `JAMMI_WORKER_ID`; kubeconform strict 1.34 + kind smoke; compute tier a provisional Deployment | 5 |
| DIST | `DIST-DATA-PLANE.md` | Ballista WITHDRAWN (task_slots-only resources, Memory cluster state, local shuffle, DF54 pin, scheduler control loop; Spice batch-only on forks) → future option behind three upstream contributions. Batch inference = jobs fleet; online retrieval beyond one node = in-house `PeerService` on `[server] peer_bind` (two-phase per precision, coordinator-only tenant, failure ladder + marginal-load admission + typed `Unavailable`); distributed SQL = `datafusion-distributed` deferred behind the DF-55 gate (sole blocker: flight-sql-server pins DF ^54). 3 commits, A1–A13; membership (unit 2) after PR-C | 7 |
| DELTA | `DELTA-INCREMENTAL-EMBEDDING.md` | versioned tables (monotonic `next_version`), base published in one txn with identity = artifact digest, per-VERSION deletion horizon, masked provider (`limit = None` into fragments; `MaskExec` never pushes a fetch), nullable `_content_hash` fifth column hashed inside the UDF with the runner's kernel, total-order sort; NULL keys = typed `InvalidKey` on base/refresh/infer via `KeyCheckExec` below the blocking sort (zero model calls by construction); structural error classification (the manual `From<DataFusionError>` is the classifier; `#[source]` kept; object-store NotFound recovered through `ParquetError::External`); five new variants get wire detail arms; `DefinitionDrift` → `FailedPrecondition` | 9 |
| GRAPH | `GRAPH-JOB-DEPENDENCIES.md` | `blocked` = projection; `claimable = TRUE` fixes a pre-existing SQLite TEMP B-TREE defect in `claim_next`; `cancelled` terminal; strict tenant at submit; the outcome push is bounded per TRANSACTION (terminal write retires DIRECT dependants with one CTE-VALUES set statement, chunk 500, bind indices from the actual chunk length; deeper levels via Arm 3, a retire-only fourth reclaim arm riding tick/boot/reclaim_on_read, MIN-named dependency, literal LIMIT); honest cost model (a pass is O(gated queued rows), 6–8.5 buffers/row measured on PG, the claim predicate's class; the gate asserts the cost class, not the join strategy); no python CANCELLED arm | 8 |
| OPS | `OPS-COMPUTE-TIER-OPERABILITY.md` | DRAIN (SIGTERM) / RELEASE (SIGINT) with PostgreSQL's mapping; `jammi-server release`; grace 600 s on the compute overlay; `release_and_stop` stated once (never abort while a claim txn can be in flight — `claim_next`'s COMMIT is client-driven); two lease classes (jobs: `releases` offsets `MAX_ATTEMPTS`; building tables: released only through the `jobs.partial_result` linkage; `lease_present` SQL guard on every renewal); warm-before-ready (`preload_models` wired, typed `PreloadEntry`, worker gate as session state); gauges sampled off-scrape; `/healthz` liveness | 12 |

## Cross-cutting facts that surfaced (each refuted a plan at least once)

- `claim_next` is a client-driven BEGIN/UPDATE/COMMIT: "the task is aborted, so no claim can land" is false — a flushed COMMIT lands after the future is dropped. Hence OPS's never-abort-while-claiming rule.
- `RETURNING` order is unspecified on both backends (measured inverted on PG 16.15 and SQLite 3.46.0); every consumer sorts in Rust.
- sqlx-sqlite binds an out-of-range placeholder as NULL silently (`arguments.rs:106-114`); bind indices must be computed from the actual chunk length.
- A bound `LIMIT` flips PG to a generic plan from the sixth execution (sqlx persistent statements); house style interpolates every `LIMIT` (zero `LIMIT $n` in the tree).
- `[worker] idle_poll_secs` defaults to 1 s; the 5 s in `config/mod.rs:593-595` belongs to the superseded `[training]` block.
- The store's `writer_id` is minted once per store and shared by every materialization on a session; nothing keyed on it alone can distinguish a loop-claimed table from an inline or library one.
- Any "X never happens in CI" claim must be checked against an artifact that would exist if it did (the cu12 `:latest` claim was refuted by the registry).

## Pre-existing engine defect found (escape to open at the OPS cut)

`jobs.partial_result` is written exactly once (`create_result_table`'s CAS `… AND partial_result IS NULL`,
`result_repo.rs:599-603`) and never cleared, so every compute attempt ≥ 2 that reaches
`MaterializeAnew` has its own `create_result_table` CAS match 0 rows → `JobAttemptSuperseded`
→ terminal `failed`. Today's expiry path hits it; `jobs_compute.rs:351` covers only the
`ready`-adopt arm. OPS C1 fixes it with `Catalog::clear_partial_result` (attempt-guarded CAS)
on `dispatch_partial_result`'s fail arm and pins the successor's terminal state.
Symptom spec for the escape row: intended = a compute job whose earlier attempt recorded
`partial_result` and whose building table is not adoptable re-materializes on the next attempt;
observable = `MaterializeAnew` then `JobAttemptSuperseded` then `failed`; control = a first
attempt (NULL `partial_result`) records its table and completes.

## Sequencing and human items

- PR-C merged (#501). PR-K is #502 (in CI). OPS, GRAPH and DELTA each append one migration, and 67 appends three (`model_materialization`, `instances_peer_addr`, `compute_cluster_state`); no plan reserves a number — each PR takes the next free at rebase and updates both pin sites (`crates/jammi-db/src/catalog/migrations.rs` const list and `crates/jammi-db/tests/it/migrations.rs:23-54` `EXPECTED_MIGRATION_NAMES`) plus OPS's relative-position oracle; the second merger renumbers (K5).
- Cut a `v*` release right after PR-C + PR-K merge so cu12 `:latest` carries `[worker]`.
- Separate human-merged items: widen `check_no_consumer_names.py`'s `SCAN_TREE_ROOTS`; pin `datafusion-federation = "=0.5.1"` (0.5.6 pulls DataFusion 55 — cargo-update hazard); a `check_doc_parity.py` binding for `JammiError` (DIST proposes, never folds in).
- `ISSUE-COMMENTS.md` holds the unposted #482 draft (its Ballista paragraph is superseded by 67's position — see PROGRAM.md §Ballista) and the withdrawn #500 draft.

## Corrections recorded by 67 (68's decisions unchanged)


1. **DIST D2, the three withdrawal grounds, re-read at Ballista 54.1 source.** Ground (2)
   "`ClusterStorage` is `Memory` only" describes the *config enum*; `ClusterState` and `JobState`
   are public traits and `BallistaCluster::new(Arc<dyn ClusterState>, Arc<dyn JobState>)` +
   `start_server(cluster, …)` accept any implementation (`ballista/scheduler/src/cluster/mod.rs`,
   `scheduler_process.rs`). Shuffle: `ExecutorProcessConfig.override_execution_engine` receives
   each stage's plan and rewrites `ShuffleReaderExec` nodes and wraps the writer
   (`executor/src/execution_engine.rs`), which is where an object-store shuffle would be
   installed, unproven. Condition (1), the accelerator dimension, stands (`ExecutorSpecification { vcores }`); 67 carries it out of band
   (`workers.devices`) and owes the upstream PR. D2's separate control-loop ground (`expire_dead_executors` started in `init()`): 67
   disposes it as executor-membership liveness (the class of `reclaim_expired_jobs` /
   `prune_instances`) with `task_max_failures = stage_max_failures = 0`, so it never makes
   consumer work runnable (67 README r40, r42). D2's condition (3), object-store shuffle, **stands**:
   67 v4 keeps Ballista's local `work_dir` and names the `ExecutionEngine` rewrite only as the
   seam a future spike would prove. **D2's decision for the data plane is not
   contested**: 67 uses Ballista only for its own compute/gang plane, in `jammi-ballista`, as the
   distributor-agnosticism proof the #500 decision requires.
2. **`ISSUE-COMMENTS.md` #500 draft ("third design option: planned training").** Superseded, not
   posted. Per-round parameter averaging is the MLlib `treeAggregate` shape 67 rejects; 67's
   gather rule makes synchronous data parallelism exact for adapters with kilobyte payloads, so
   the local-SGD convergence risk is unnecessary. Its "what Ballista lacks" list is folded into
   67 DESIGN §9 with the corrections above. The #482 consequence stays the StatefulSet (67 r21),
   not "remove the provisional note".
3. **K D5 / `:182`.** The compute overlay's `nvidia.com/gpu: 1` becomes `N` and the Deployment a
   StatefulSet with a headless service when 67 U9 lands, after K merges; co-owned, K first.


## Folded from 67 (2026-09-10)


- OPS: a peer host's DRAIN/RELEASE ends a running rank with `RankEvent::Released`; 67 counts it
  through `release_job_lease` (D10's zero-net-cost guarantee holds for gangs).
- OPS D14 `workers.state`: a rank-running peer is `claiming` with its `JobSlot` busy; no new
  state (67 r27).
- The actuator rule (D5) has no constitution ID; 67 proposes a human-merged constitution row as
  a follow-on.
- Migration numbering: 67 appends three after 68's four (OPS, GRAPH, DELTA, DIST unit 2; 67 r30).
- `instances.peer_addr` does double duty: a replica that sets `peer_advertise` to be
  gang-reachable also joins DIST's retrieval ring (unit 2's ring is every row with `peer_addr`
  set, no capability filter). Capability-scoping the ring is a 68 follow-on to consider.

## Directory

- `PROGRAM.md` — the single schedule across 67 and 68.
- `units/` — the five consolidated plans (final versions: K-v3, DIST-v3, DELTA-v5, GRAPH-v5, OPS-v4).
- `briefs/` — the unit briefs and the common principles brief the planners worked from.
- `pressure/verdicts/` — the early-round pressure verdicts kept as files; later rounds are summarised row-by-row in the ledger.
- `pressure/measurements/pt4..pt7/` — the SQL the testers ran on live PG 16.15 (`docker jammi-pg`, TEMP tables inside `BEGIN … ROLLBACK`) and SQLite 3.46.0 (a CLI built from the bundled `libsqlite3-sys` source); the plans cite these paths.
