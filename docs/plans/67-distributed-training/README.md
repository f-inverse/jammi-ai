# 67 — Multi-GPU and multi-node training as engine mechanism, on DataFusion (#500)

**Status:** PLANNED, v4 — rebased on the jobs fleet, which is now on `main` (PR #501, merge
`4ecc0230`, 2026-09-10: `jobs` table, migrations 029/030, `[worker]`, `JobService`; the #485/#486
work), reconciled with `docs/plans/68-compute-tier-substrate/` (its PR-K is in CI), U8 reshaped as
a Ballista **extension** unit. Scoped twice (gap-analyzer 2026-09-10, v1 and v4
briefs), pressure-tested in four rounds (`PRESSURE.md`), NOT implemented. Hand-off order: this
file, `DESIGN.md`, `UNITS.md`, `SIZING.md`, `PRESSURE.md`; then `68-compute-tier-substrate/README.md`
for the sibling units 67 depends on.

**Posture:** greenfield. Nothing that exists is a constraint; whatever needs rebuilding the
right way is rebuilt. Every fork is resolved by deriving from `docs/guide/src/philosophy.md`
and `docs/swarm/CONSTITUTION.md` and from solid outside references (`DESIGN.md` §References),
never by asking. Plans 67 and 68 are owned together from 2026-09-10; 68's five units keep their PROCEED
verdicts unchanged, and the single cross-plan schedule is `68-compute-tier-substrate/PROGRAM.md`.

## The ask (issue #500)

Training runs today in one process on one device; the worker fleet is job-level parallelism
only. Build multi-GPU and multi-node training as engine mechanism: a training job declares a
world size, N ranks train one job cooperatively, gang failure semantics are explicit, identity
and parity are scoped, placement stays the deployer's runtime.

## The position (DESIGN.md §1)

1. **The training set is a plan and a producer** — a content-addressed result table; sharding
   is partitioning of its train prefix by a fixed rule; epochs re-read it with bounded memory.
2. **A trained model is a producer** — `ProducingDescriptor::FineTune` folds the training-set
   digest, base-model identity, the canonical spec, the seed and the topology; replay is retrain.
3. **The gang is a co-scheduled unit, not a per-step query** — every rank gathers the global
   batch's representations and computes the identical global loss (batch-coupled objectives keep
   their semantics); adapter gradients are summed; collectives run in-binary behind one trait;
   any rank failure fails the attempt; the jobs fleet's lease, reclaim and resume checkpoint give
   restart.
4. **Not** the SGD step or gradient exchange as DataFusion operators or aggregates.

Distributor-agnosticism is proven last by a `jammi-ballista` crate that extends Ballista at
its seams the way `jammi-kernels` extends candle, and executes the same operators unchanged.

## Decisions taken by the user

| Fork | Decision |
|---|---|
| Ballista posture (2026-09-10) | Distributor-agnostic build; a Ballista unit is **in plan, mandatory, last**. Reaffirmed after 68's withdrawal: extend Ballista at its seams like candle, never fork. |
| Hardware proof | CPU collective hermetic in CI; single-node multi-GPU on a RunPod pod; multi-node on a 2-node RunPod cluster; committed artifacts. |
| Unit 0 scope | The greenfield rebuild of the training-data path and model identity is in scope. |
| Plan hygiene | 67 modified in place to v4; 68 modified in place, additively. |

## Lead rulings

Rulings 1–25 are the v3.1 set, kept verbatim in `PRESSURE.md` §"v3.1 rulings" where superseded;
those still in force are restated here in their v4 form. Principle in parentheses.

**Substrate (the jobs fleet)**

26. **The claimant is the coordinator.** A training-kind job is claimed by a `JobWorker` through
    `claim_next` (`jobs_repo.rs:658-719`); that process is rank 0 and holds the only lease
    (`heartbeat_job`, `jobs_repo.rs:772`). Kinds eligible for `world_size > 1`: `fine_tune` and
    `graph_fine_tune`; `context_predictor` is refused at `world_size > 1` (K2, typed, at submit).
27. **A peer is a fleet worker with a busy slot.** A peer is a `JobWorker` process whose
    `[worker] kinds` include the job's kind and whose `peer_bind` is set. `RunRank` takes the
    worker's single job slot: a `JobSlot` mutex the claim loop takes **before** `claim_next`,
    **holds across** the inline `run_claimed_job` (`worker.rs:355`), and **releases before** the
    idle sleep (`:363`) — so a peer never claims while it runs a rank, never aborts a claim
    transaction (OPS D6), never receives a rank while training its own job, and is reachable
    whenever idle. Handler order: same `job_id` with a lesser attempt → abort that runner and
    take the slot; lesser-or-equal attempt → refuse; otherwise try-lock; busy → typed
    `Unavailable`, and the coordinator picks another member or fails the attempt. No new worker
    state. (B1; OPS D6.)
28. **Membership substrate is built by 67, used by both plans.** 68 DIST "unit 2" is a design
    sketch (`DIST-DATA-PLANE.md:208-215`), not a plannable unit, so U5b-1 lands the substrate it
    sketches: `[server] peer_advertise` (validate: `peer_advertise ⇒ peer_bind ⇒ result_root`),
    the `instances.peer_addr` column (migration, numbered at rebase), the `upsert_instance`
    signature and the session write site (`session.rs:247-251`). DIST's `RendezvousPlacement`
    builds on it later. Peers are resolved from the catalog: `workers.kinds` ∋ kind,
    `instances.peer_addr` set, `last_seen_at` within `[lease] duration_secs`, and — from U8b on —
    `workers.devices` sufficient — through a new joined listing `list_gang_members(kind)` that splits
    `workers.kinds` on `,` and compares whole tokens in Rust (`fine_tune` must not match
    `graph_fine_tune`). No static peer list. Consequence recorded in 68's reconciliation: a
    replica that sets `peer_advertise` to be gang-reachable also joins DIST's retrieval ring;
    capability-scoping the ring is a 68 follow-on. (One membership mechanism; DIST D9.)
29. **Knobs.** `[gpu] devices = [..]`; `[worker] world_size = 1`, `rank_timeout_secs = 120`,
    `collective = "auto"`; per-job `world_size` on `TrainingCommon` (`#[serde(default)]` = 1).
    `[training]` no longer exists on the branch.
30. **Migrations.** 67 appends exactly three: `model_materialization` (U3, PR-B),
    `instances_peer_addr` (U5b-1, PR-C), `compute_cluster_state` (U8b, PR-D — distributor-neutral
    name and columns; `workers.devices` rides in it, since U8b is its first reader). No plan
    reserves a number: each PR takes the next free number at rebase and updates **both pin
    sites** — the const list in `crates/jammi-db/src/catalog/migrations.rs` and
    `EXPECTED_MIGRATION_NAMES` in `crates/jammi-db/tests/it/migrations.rs:23-54` (exact-equality
    asserts) — and OPS's relative-position oracle; the second merger renumbers (K5). Three 68
    units (OPS, GRAPH, DELTA) also append one each.
31. **The training set is not this attempt's partial result.** U2a materializes it with
    `job_attempt: None`: it is a shared producer output reused by definition hash, not an
    attempt-owned table, so the `jobs.partial_result` attempt≥2 defect (68 OPS C1) is never
    reached. It is still lease-guarded (`writer_id`/`lease_expires_at` are independent of the
    jobs CAS, `result_repo.rs:99-130`) but outside OPS's linked release sweep, so a crashed or
    released coordinator leaves a live `building` row: the successor (or a second job over the
    same training set) that finds a live same-named `building` row **backs off** — returns the
    `BackOff` disposition (`jobs.rs:248-261`), leaving the job `running` for the next tick — and
    reclaims it through `claim_expired_building_table` (`store/mod.rs:1443-1470`) once the
    lease expires. The model artifact remains the job's result through `finish_job_with_model`.
32. **Row order from the catalog is never trusted.** No 67 query consumes `RETURNING` order;
    every listing sorts in Rust (68 cross-cutting fact).

**Peer surface (68's listener)**

33. **`GangService` is a second service on `[server] peer_bind`** (DIST D7's third listener),
    never on the tenant-scoped public chain, never advertised by `GetServerInfo`. **68 DIST unit
    1 merged is a hard precondition of U5a** (its listener commit is ~22 files on top of ~10;
    there is no verbatim-carry fallback). PR-C(67) also waits for OPS and GRAPH, because OPS C2
    rewrites the claim loop U5a's `JobSlot` wraps and GRAPH rewrites `claim_next`.
34. **Authorization: the job row is the capability (invariant I-GANG).** The peer reads the
    `jobs` row through a **new db-owned verb `get_job_for_rank(job_id)`** — by primary key, no
    tenant predicate, never admin scope (`get_job` is tenant-filtered, `jobs_repo.rs:580-596`,
    and D7 forbids `with_admin_scope` on the peer path), reachable only from the gang handler —
    verifies `status = 'running'`, `claimed_by = coordinator_instance_id`, lease live, and
    **derives the tenant from the row** (`jobs.tenant_id`), pinning every subsequent catalog
    read to it. Nothing dialable travels on the wire: the assignment carries
    `peers[rank → instance_id]` and each peer resolves addresses through `instances.peer_addr`,
    refusing a rank whose instance is not a fresh member (the NCCL id, an opaque secret, is the
    only out-of-band value). `FetchPartition` verifies the named table belongs to the job's
    training set (the analogue of D7's segment-belongs-to-table check). The RPCs get their own
    `GANG_LISTENER_ALLOWLIST` bucket in `tenant_isolation_oracle.rs` (text: "served only on
    peer_bind; tenant derived from the verified job row; deliberately not caller-scoped"),
    unioned like D7's, with the public-listener `UNIMPLEMENTED` assertion, and their
    `api_freeze_baseline.txt` lines land in the same commit. (B5; D7's single-binder rule.)
35. **Attempt fence on `job_id`**: a `RunRank` at attempt N aborts every local runner of that
    job at attempt < N; lesser-or-equal refused.

**Operability (68 OPS)**

36. **An aborted attempt lands no terminal write.** `fail_job` is terminal (`jobs_repo.rs:
    1057-1100`: `status = 'failed'`, no attempt bump); the only requeue path on the fleet is the
    leave-`running`-for-reclaim arm (`worker.rs:670-676`) → reclaim arm 1a → `attempts + 1` at
    the successor's claim. So on any rank failure the coordinator cancels every rank, aborts the
    attempt (no publish, no finalize), flips its hold's `lost` flag (the cancel flag *is*
    `hold.lost_flag()`, `worker.rs:531-547`) so its own run exits through that arm, and the job
    is requeued by reclaim within the remaining lease window (≤ `[lease] duration_secs`, 30 s
    default) — no new verb. **A released rank is a release, not a failure**: DRAIN/RELEASE on a
    peer host ends the rank with `RankEvent::Released`; the coordinator first calls OPS's
    `release_job_lease` (`releases + 1`, lease NULL — the CAS admits the holder), then flips the
    same flag; the keeper's `lease_present` guard would flip it within one heartbeat anyway. A
    peer-tier rolling restart costs zero net attempts (OPS D10). U5b depends on OPS.
37. **The watchdog is allowed under the actuator rule.** It is per attempt, bounded by the
    attempt's lifetime, created by the claimant for the job it holds, and only retires that
    attempt (requeue is the pre-existing reclaim semantics) — the lease keeper's shape, not a
    standing loop. The rule (`recompute.rs:29-35`; 68 DIST D5) has no constitution ID; a
    human-merged constitution row is a proposed follow-on, not assumed.

**The Ballista extension (U8a, U8b)**

38. **Discipline, not dependency shape, is what `jammi-kernels` teaches**: extend at the seam
    the library exposes; one call path; vendor-and-shim only where no seam exists; oracles
    before belief; upstream-only, never a fork. `jammi-ballista` is a workspace crate that
    depends on `jammi-ai`, `jammi-db`, `jammi-wire` and the Ballista crates and is depended on by
    `jammi-server`; publishable, lockstep (K6); no cargo feature (B4's "no library-vs-server
    gate"): the library never loses the capability because a library embedder adds the same
    crate. No vendored subtree in v1. A new workspace crate has three registration sites, two
    of which red at merge: the domain card's `owns:` globs (`.claude/agents/wire-server.md` takes
    `crates/jammi-ballista/**`; `check_swarm_bijection.py` asserts a total partition, and editing
    a card trips `SWARM_GATE_TOUCHED` → **PR-D needs an admin merge**), `ci/scripts/publish_crates.sh`'s
    ordered publish list (insert before `jammi-server`), and — advisory by standing, `dep-dag.yml`
    is outside `ci-summary` — the generated dep-DAG block in `docs/maintainer/MAINTAINER-GUIDE.md`
    (`gen_dep_dag.py`).
    `check_dep_direction.py` encodes no layering and is not touched.
39. **Roles are listener-shaped knobs, like `peer_bind`**: a replica hosts the scheduler iff
    `[ballista] scheduler_bind` is set, and an executor iff `[ballista] executor.scheduler_address`
    is set. Not a tier, not a CLI role. Topology is configuration. (B4; DIST D7's "owner iff
    `peer_bind` set".)
40. **Retries are jammi's, never Ballista's.** The scheduler role runs with
    `task_max_failures = 0` and `stage_max_failures = 0` (scheduler-global fields; no per-job
    scoping exists or is needed). Re-running work is the jobs table's `attempts`/reclaim.
41. **One gang mechanism.** Under Ballista a gang job is ONE task (`GangExec`, single partition)
    whose `execute` runs the U5b coordinator; Ballista's job is to place that task on a
    device-bearing executor. No all-or-nothing rank binding in Ballista; a gang stage kind stays
    a future upstream option. (K4 parity: bytes == U5b.)
42. **Executor liveness is membership, and it resets tasks with no knob to stop it (S6 probe
    5).** `expire_dead_executors` (started unconditionally in `SchedulerServer::init`) sweeps
    Ballista executor heartbeats — the same class as `reclaim_expired_jobs` and `prune_instances`
    the fleet already runs each tick — but the same loop also posts `ExecutorLost`
    (`scheduler_server/mod.rs:395`) → `reset_stages_on_lost_executor`: `RunningStage::reset_tasks`
    frees the lost task's slot and `SuccessfulStage::reset_tasks` re-fails its COMPLETED tasks as
    `ResultLost` (`retryable: true, count_to_failures: false`), which `update_task_status` resets
    **without consulting `task_max_failures`**. With both retry knobs at 0, a `GangExec` on a
    killed executor is re-launched by Ballista on a surviving executor and the job succeeds on
    its own — in parallel with jammi's own reclaim. U8b needs an explicit bind-time guard in
    `CatalogClusterState::bind_schedulable_tasks` / `DevicePlacement` that refuses to re-bind a
    task whose `(job_id, stage_id, partition)` was already launched, keyed on jammi's own job row
    (Ballista's `task_attempt` counter is not bumped by this reset, so the refusal cannot key on
    it) — or an upstream "no reset on lost executor" option; there is no knob for this in 54.1.
    This still disposes the control-loop ground of 68 DIST D2 for jammi's compute plane (D2's
    numbered future-option conditions are the accelerator dimension, pluggable cluster storage
    and object-store shuffle); DIST's decision for the data plane is untouched.
43. **The seams and the one true gap.** Codecs (`override_{logical,physical}_codec`); the
    `ExecutionEngine` (`override_execution_engine`; receives each stage's plan, rewrites
    `ShuffleReaderExec` nodes and wraps the writer — the seam an object-store shuffle would use,
    **not adopted in v1**: shuffle stays Ballista's local `work_dir`, and D2's object-store-shuffle
    condition stands until a spike proves the cross-executor read); `BallistaCluster::new(Arc<dyn
    ClusterState>, Arc<dyn JobState>)` + `start_server(cluster, …)` — executor registrations and
    heartbeats survive a scheduler restart from a custom state (S6 probe 6); an in-flight job does
    not (`ExecutionGraphBox` has no serialisation in 54.1, `JobState::try_acquire_job` is never
    called by the scheduler), so recovery is a re-submit of the stored plan as a *new* Ballista job
    id — jammi's own `attempts`/reclaim semantics (r36/r40), not Ballista-side job survival; two
    schedulers over one store serve one cluster only for sequential jobs (S6 probe 7) — concurrent
    jobs hang on whichever scheduler loses the slot race, with no public path to wake it
    (`revive_offers` and `query_stage_event_loop` are `pub(crate)`, `job_resubmit_interval_ms` has
    no readers, `cluster_state_events` is unconsumed): active/standby, not Spice's active/active
    HA, until a second upstream contribution lands; `TaskDistributionPolicy::Custom` (a custom
    `ClusterState` must bring its own binder — the built-in `bind_task_bias` / `bind_task_round_robin`
    are `pub(crate)` — and in pull-staged mode `ClusterState::bind_schedulable_tasks` is bypassed
    entirely and only the `Custom` policy is honoured, so the role knobs pin push-staged
    scheduling); `TaskLauncher`. `DevicePlacement` learns that a task is GPU-bound from the stage's
    physical plan in `active_jobs`' execution graph, decoded through `JammiCodec` (a `GangExec` or
    an `InferenceExec` whose descriptor names a CUDA device); S6 proves that read. The accelerator
    dimension has no seam (Rust `ExecutorSpecification { task_slots: u32 }`, not `vcores`; the
    proto side's `oneof resource { TaskSlots(u32) }` is extensible); jammi carries it out of band
    in `workers.devices`, and the upstream accelerator variant is the only PR 67 owes.
44. **Device pinning does not move bytes.** The executor process runs on its configured
    `[gpu] devices`; device *kind* is already in `MaterializationEnv`; the ordinal is not
    output-affecting; the shuffle writer never reorders a partition-ordered sink. K4 and K7 hold
    through the Ballista path per device kind.
45. **Naming pre-swept.** New pub items: `JammiCodec`, `JammiExecutionEngine`, `CatalogClusterState`,
    `CatalogJobState`, `DevicePlacement`, `BallistaConfig`, `GangExec`, `JobSlot`, `RankEvent::Released`
    — none carries the seven governance stems (`check_no_consumer_names.py:78-80`); trait-impl
    methods such as `create_query_stage_exec` are not `pub` declarations and are not scanned.

**Still in force from v3.1 (restated)**: streaming loader and per-batch converters (r1);
mining/GradCache W=1-only with the `mine`/`cached` predicate (r2); split, order and partition
rule with zero-row ranks and the global-batch formula (r3); scaler from one collected `Vec`
(r4); model identity with the opaque canonical encoding (r5); the gather rule with per-arm
gather points (r6); lockstep control flow (r7); resume with per-rank `dropout_positions` (r8);
descriptors not physical plans before Ballista (r10); U6's partition-aware operator (r12); one
trainer, four collectives (r13); the shared `CacheKey` (r14); K4 is remote-equals-embedded
(r15); GPU byte oracles downgraded until S5 (r16); DataFusion 54 first (r19, ordering amended
in r46); committed-artifact convention (r20); StatefulSet consequence, now owned by U9 after
68 K merges (r21); split rubric (r22); naming (r23); ContextPredictor out of gang scope (r24).

46. **Order against the sibling work.** PR-C(68) merged as #501 (242 files, both manifests, seven
    crates) before any 67 unit — so PR-A (U1) starts now from `main`; 68's K (PR in CI) and DIST
    unit 1 precede PR-B; PR-C(67) needs DIST unit 2 and OPS; PR-D needs K. `SIZING.md` carries
    the edge list. Correction to v3.1 ruling 18: the issue's "jobs table, migrations 029/030"
    was describing #485/#486's branch, which is now merged; only the "#485 is unrelated" remark
    in the 2026-09-10 issue comment was wrong.

## Units and order

| PR | Commit | Unit | Name | Lane | Depends on |
|---|---|---|---|---|---|
| A | 1 | U1 | DataFusion 54 line upgrade (workspace-atomic) | hermetic + cookbook + db-features clippy lane | S3 (main already carries #501) |
| B | 1 | U7a | `gpu-gang.yml` pod leg; `runpod_lib.sh` gpuCount; allowlist; artifact schema | gate scripts | S4 |
| B | 2 | U2a | `TrainingSet` producer (`job_attempt: None`; wire mirror; guide block) | hermetic + cookbook | U1 |
| B | 3 | U4a | `Collective` trait; device-plural session; `CacheKey`; refusals | hermetic (+ pod smoke) | S1 |
| B | 4 | U2b | Streaming loader; partition rule; scaler; whole-set arms | hermetic + cookbook | U2a, U4a |
| B | 5 | U3 | `FineTune` producer; `model_materialization` migration; cache reuse | hermetic | U2a |
| B | 6 | U4b | Rank context; gather rule; lockstep; single-node gang | hermetic + pod leg | U2b, U3, U4a, S1 |
| B | 7 | — | pod-leg artifact | gpu-gang | U4b |
| C | 1 | U7b | cluster leg + cluster reap | gate scripts | U7a, S4 |
| C | 2 | U5a | `GangService` on `peer_bind`; I-GANG authorization; allowlist + freeze lines | hermetic + server it-suite | U4a, 68 DIST unit 1 |
| C | 3 | U6 | Partition-aware inference operator | hermetic | U2b, U5a |
| C | 4 | U5b-1 | Coordinator; `Peer` collective; membership substrate (`peer_advertise`, `instances.peer_addr`, `list_gang_members`); determinism; two-worker forward leg | distributed | U4b, U5a, U6, S1 |
| C | 5 | U5b-2 | Watchdog; abort with no terminal write; released-vs-failed; chaos; cluster leg | distributed + cluster leg | U5b-1, 68 OPS |
| C | 6 | — | cluster-leg artifact | gpu-gang | U5b-2 |
| D | 1 | U8a | `jammi-ballista`: crate (+ card globs, publish list, dep-DAG), codecs, `JammiExecutionEngine`, role knobs; in-memory cluster | hermetic + distributed three-process arm | U1, U5b-1, U6, S6 |
| D | 2 | U8b | `CatalogClusterState`/`CatalogJobState`, `DevicePlacement`, `compute_cluster_state` migration (+ `workers.devices`), gang as one placed task | distributed | U8a, S6 |
| D | 3 | U9a | Docs (guide, maintainer, CHANGELOG) | docs gates | all |
| D | 4 | U9b | shape-d overlay → StatefulSet + headless service + `nvidia.com/gpu: N` (after 68 K, keeping OPS C6's grace) | kubeconform + kind smoke | 68 K, 68 OPS |

Spikes (no PR; results in the ledger before the dependent unit is briefed): **S1** (→ U4a,
U5b) `candle-core/nccl` under the `cuda` feature; `Comm::from_devices`; `all_gather` equal and
unequal counts; two-process `from_rank`. **Runtime result:** the NCCL host call itself never
blocks — the hang is in `stream.synchronize()`, and NCCL does not detect a dead peer, so a
watchdog `ncclCommAbort` from another thread is what unblocks it; the sync then returns `Ok` with
a garbage buffer, so the abort flag (not the sync's return value) is the failure signal, and the
`Nccl` arm needs an `Aborted` state and must never drop after an explicit abort (a drop after
abort double-aborts and SIGSEGVs). Unequal-count `all_gather` via pad-to-max plus narrow is
bitwise equal to a CPU concat for f32 and bf16. **S3** (→ U1) throwaway DataFusion 54 workspace
compile on top of PR-C, incl. `-p jammi-db --features postgres,mysql`. **S4** (→ U7) a
`gpuCount: 2` pod + create-cluster probe — spends money, human-approved. **Result:** a 2-GPU pod
and a 2-pod × 2-GPU TRAINING cluster both provision from the repo's own payload shape; cluster
members expose `actions: []`, so the reaper deletes the cluster (not a member pod), and
`list-pods` needs `includeClusterPods=true` to see them; cluster GPUs bill at $1.908/GPU/h, so
U7b's ceiling is $7.63/h (4 GPUs), not $6.4. **S5** (→ GPU byte oracles) CUDA
bit-reproducibility pin. **Result:** candle 0.11 LoRA-shaped forward/backward/SGD is
byte-identical across processes and across two A100s with no env pins; `CUBLAS_WORKSPACE_CONFIG`
is a kernel-selection input that must merely be *consistent* across ranks (`:4096:8` reproduces
the unset default, `:16:8` differs), so it belongs in the `MaterializationEnv` kernel profile,
not a default setting; the `NCCL_ALGO`/`PROTO`/`NCHANNELS` pin set is untested — at world size 2
the reduction is commutative, so it needs world size ≥ 3 on the cluster leg. **S6** (→ U8a/U8b;
supersedes S2) a scratch crate on Ballista 54.1 with `override_execution_engine`, a custom
`ClusterState`/`JobState` passed to `start_server`, and the codec, running a custom
`ExecutionPlan` on one scheduler + two executors. **Result:** retries are off for jammi operators
by error classification, not by the knob (README r43); `expire_dead_executors` removes executors
but the same loop also resets and re-launches their tasks with no knob to stop it (README r42);
executor registrations survive a scheduler restart, in-flight jobs do not (README r43); two
schedulers over one store serve one cluster only for sequential jobs, not concurrent ones (README
r43). U8b is gated on all of these. **S3** also records `cargo tree -d` for `tonic` and `prost`
with the OpenTelemetry family #501 added.

## Hand-off: how a fresh lead kicks this off

1. Read `docs/swarm/CONSTITUTION.md`, `docs/swarm/SELF-FAILURE-MODES.md`, `.claude/agents/lead.md`,
   then `68-compute-tier-substrate/README.md` (its PR-C, K, DIST, OPS units are 67's
   preconditions).
2. Ledger: `.jammi/ledger/distributed-training-500-<date>.jsonl`; this session's rows are in
   `distributed-training-500-20260910.jsonl` (gitignored).
3. Run S3 on `main` (it carries #501); S1, S4, S5, S6 concurrently.
4. PR-A (U1) from `main` now; one worktree, one commit. 68's PR-K is in CI and may merge
   before PR-A — rebase, no ordering constraint between them.
5. PR-B after PR-A merges; no 68 dependency. Commit order as in the table; PR-B edits
   `ci/scripts/check_cuda_run_artifacts.py` (U7a), which trips `SWARM_GATE_TOUCHED`, so PR-B is
   an admin merge too;
   U7a ∥ U2a ∥ U4a; U2b ∥ U3; U4b last; label the PR for the pod leg.
6. PR-C(67) after DIST unit 1, OPS and GRAPH merge (they rewrite the claim loop and
   `claim_next`). U7b ∥ U5a; U6; U5b-1; U5b-2; `distributed.yml` dispatched manually,
   deterministic leg green before merge.
7. PR-D after 68 K: U8a, U8b (the completion gate), U9a, U9b. `distributed.yml` dispatched
   manually with the three-process arm green before merge. PR-D edits a swarm domain card, so it
   needs an admin merge (`SWARM_GATE_TOUCHED`).
8. Every commit runs phases 3–6.5; every PR runs phase 7; amend subagent commits with the
   session trailers. PR bases: each PR from `main` after its preconditions merged; never stacked.
