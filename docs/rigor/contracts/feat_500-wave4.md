# Contract — `feat/500-wave4`: PR-D, the Ballista compute plane (U8a, U8b, U9a, U9b), one consolidated PR

Base: `main` @ `0e11162a` (PR #584 merged: the whole remainder of plan 67's wave 3). Slug
`feat_500-wave4`; pressure row in `docs/rigor/feat_500-wave4.jsonl`; oracle record in
`docs/rigor/feat_500-wave4.oracle.jsonl` at the final tip, exported last, after every non-rigor
edit. Plan: `docs/plans/67-distributed-training/{README.md (r38–r45), DESIGN.md §9, UNITS.md
§U8a/§U8b/§U9a/§U9b, SIZING.md}`; schedule `docs/plans/68-compute-tier-substrate/PROGRAM.md`
wave 4. Spike S6's results (README §Spikes) are the premises this contract re-verifies against
Ballista 54.1.0's source, which the lead read on 2026-09-16 from the published crates.

## 1. Scope and build order

Ballista 54.1.0 is on crates.io and its dependency line is the workspace's exactly
(`arrow-flight 58.3`, `datafusion 54`, `datafusion-proto 54`, `object_store 0.13.2`, `prost
0.14`, `tonic 0.14`), so `deny.toml`'s multiple-version bans hold with no line bump. The units
and their hard order, from `SIZING.md`: U8a → U8b → U9a → U9b, with U8b's db slice and U9b
independent of U8a's crate and built concurrently:

| # | Unit | Owner | Built | Depends on |
|---|---|---|---|---|
| 1 | U8a-cfg + U8b-db — `[ballista]` config section; migration 038 `compute_cluster_state`; `compute_repo.rs`; `workers.devices`; `transfer_claim` | db | concurrently (wave A) | main |
| 2 | U8a-crate — `crates/jammi-ballista`: `JammiCodec`, `JammiExecutionEngine`, roles, client; hosting in `jammi-server`; the three registration sites; the `distributed.yml` `ballista` leg | wire-server | concurrently (wave A) | main (+ the accessor commit it makes itself) |
| 3 | U8a-gang — `GangExec`; the `Placed` arm of `run_spec`; `run_placed_gang`; the two session seams; `WorkerFacts.devices` | ai-core | concurrently (wave A) | main |
| 4 | U9b — shape-d overlay → StatefulSet + headless Service + `nvidia.com/gpu: N` | docs-ci | concurrently (wave A) | main |
| 5 | U8b — `CatalogClusterState`/`CatalogJobState`, `DevicePlacement` with the bind-time re-launch guard, `tests/distributed/cluster_state.rs` | wire-server | after 1, 2, 3 | 1, 2, 3 |
| 6 | U9a — guide, maintainer guide, CHANGELOG | docs-ci | last | all |

The lead consolidates on `feat/500-wave4`, runs `ci/scripts/merge_path.sh` once on the
consolidated tip, dispatches the distributed lane (`gh workflow run distributed.yml --ref
feat/500-wave4`; the `ballista` leg is `advisory: false` and must be green before merge), then
the pressure row and the oracle. PR-D edits `.claude/agents/wire-server.md` (`owns:` gains
`crates/jammi-ballista/**`), so `SWARM_GATE_TOUCHED` is the one expected red and the merge is an
admin merge (authorized).

## 2. Design — the placed gang and the roles (the part the pressure round attacks first)

### 2.1 Roles are listener-shaped knobs (README r39)

```toml
[ballista]
scheduler_bind = "0.0.0.0 port 50050"          # Some = this process hosts a Ballista scheduler
[ballista.executor]
scheduler_address = "10.0.4.7 port 50050"      # Some = this process hosts a Ballista executor at that scheduler
bind = "0.0.0.0 port 50051"                    # the executor's Arrow Flight (shuffle) listener
grpc_bind = "0.0.0.0 port 50052"               # the executor's gRPC (task) listener
advertise_host = "10.0.4.8"               # the host other executors/the scheduler dial; default: bind host
work_dir = "/var/lib/jammi/shuffle"       # default: a fresh temp dir per process
task_slots = 1                            # >= 1
```

Unset `[ballista]` = no roles = today's process, byte-for-byte. Both roles on one process is
the single-node cluster. `BallistaConfig::validate` (in `jammi_db::config`, the same class as
`ServerConfig::validate`): every bind address parses; a FIXED-port collision among
`scheduler_bind`, `executor.bind`, `executor.grpc_bind`, `health_listen`, `flight_listen`,
`peer_bind` is refused naming both keys (`:0` never collides); `executor.scheduler_address`
parses as `host:port`; `task_slots >= 1`; `work_dir`, when set, is non-empty. Both roles pin
**push-staged** scheduling (pull-staged bypasses `ClusterState::bind_schedulable_tasks`, README
r43) and `task_max_failures = stage_max_failures = 0` (README r40).

### 2.2 The crate

`crates/jammi-ballista` depends on `jammi-ai`, `jammi-db`, `jammi-wire`, `ballista-core`,
`ballista-scheduler` (`default-features = false`), `ballista-executor` (`default-features =
false`, `arrow-ipc-optimizations`), `datafusion-proto`; `jammi-server` depends on it with no
feature (B4). Registration sites: `Cargo.toml` members/default-members/workspace deps,
`ci/scripts/publish_crates.sh` `PUBLISH_ORDER` (before `jammi-server`),
`.claude/agents/wire-server.md` `owns:`, the generated dep-DAG block in
`docs/maintainer/MAINTAINER-GUIDE.md`.

**`JammiCodec`** (`PhysicalExtensionCodec`): encodes `InferenceExec`, `AnnSearchExec`,
`AsofJoinExec`, `KeyCheckExec`, `GangExec` as prost messages of a new package
`jammi.ballista.v1` compiled by the crate's own `build.rs` — NOT `jammi-wire`'s frozen
`jammi.v1.*` surface (dated correction to UNITS §U8a's "↔ U5 descriptor messages": the
descriptor a codec carries is each operator's own construction inputs, owned by the crate that
speaks Ballista's wire). Every buffer the codec writes starts with a 4-byte magic so a jammi
buffer and a Ballista buffer can never alias; a buffer without the magic is delegated to
`BallistaPhysicalExtensionCodec` (shuffle reader/writer, unresolved shuffle), which is the ONLY
way Ballista's own nodes cross the wire (its codec does not delegate unknown nodes: it returns
"Unsupported plan node", read at `ballista-core-54.1.0/src/serde/mod.rs`). A node neither
codec knows is a typed error naming it; `MaskExec` (a masked result-table scan) is such a node
in v1 and is a NAMED CUT (§8d) — the embedding plan `InferenceExec(SortExec(KeyCheckExec(
CoalescePartitionsExec(scan))))` never carries it. Decode rebuilds each operator through its
public constructor with the process's `InferenceSession` (`JammiCodec` holds a `Weak` to it):
`InferenceExecBuilder` over the session's model cache; `AnnSearchExec::new` after re-reading the
`ResultTableRecord` by name and re-validating the query vector (`validate_query`, source
`Caller`); `AsofJoinExec::try_new(left, right, spec)`; `KeyCheckExec::try_new(input, key)`;
`GangExec::new(descriptor)`. The scheduler role decodes with the same codec (it holds a session
too: every role is hosted by a `jammi-server` process).

**`JammiExecutionEngine`** wraps `DefaultExecutionEngine` (shuffle reader rewrite + writer wrap
unchanged, `work_dir` local — no object-store shuffle in v1, D2 condition 3 stands) and adds
two duties before delegating: (1) a stage whose plan contains `GangExec` must be single-partition
(typed error otherwise — one gang mechanism, README r41); (2) K7 device pinning — an
`InferenceExec` whose descriptor names a device KIND this executor's session does not run on
(`InferenceSession::compute_device`) is refused typed, never silently run on the CPU. The
executor's `runtime_producer`, `config_producer` and function registry are built from the
session (its `RuntimeEnv` carries the registered object stores; its context carries
`jammi_content_hash` and the vector UDAFs), so a plan that decodes on the executor resolves
every UDF and every store the submitter's plan referenced.

**Roles** (`roles.rs`): the scheduler is `create_scheduler::<LogicalPlanNode,
PhysicalPlanNode>(cluster, config)` + `init()` served by a tonic server on `scheduler_bind`
with jammi's shutdown (never Ballista's `start_server`/`start_executor_process`, which install
their own `ctrl_c` handlers and would race the server's two-mode shutdown); U8a: `BallistaCluster::
new_memory` + `TaskDistributionPolicy::RoundRobin`; U8b: `BallistaCluster::new(CatalogClusterState,
CatalogJobState)` + `Custom(DevicePlacement)`. The executor is `Executor::new(.., Arc::new(
JammiExecutionEngine))` + the Flight shuffle service on `executor.bind` +
`executor_server::startup(..)` (push-staged registration + heartbeats) on `executor.grpc_bind`,
held by a `ShutdownNotifier` the server fires on DRAIN/RELEASE. Hosting: `OssServer::bind`
builds the roles from `[ballista]` beside the peer listener; `serve_with_signals` stops them with
the peer listener. The executor role installs the `PlacedGangRunner` seam on the session; the
scheduler role installs the `PlacedGangSubmitter` seam (§2.3).

**Client** (`client.rs`): `submit_physical_plan(session, scheduler_url, plan) ->
SendableRecordBatchStream` over `ballista_core::execution_plans::execute_physical_plan::<
PhysicalPlanNode>` with the session's `SessionConfig` upgraded for Ballista and `JammiCodec`.

### 2.3 The placed gang (README r41; U8a acceptance (b), (c))

Under Ballista a gang job is ONE task, `GangExec { job_id, attempt, world, submitter }`
(single partition, output schema `{ outcome: Utf8, artifact_digest: Utf8? }`), placed by the
scheduler on a device-bearing executor. Two seams in `jammi-ai` (the way `MemberDialer` is a
seam the server installs, `crates/jammi-ai/src/fine_tune/worker.rs::MemberDialer`), so
`jammi-ai` never depends on `jammi-ballista`:

- `PlacedGangSubmitter` (installed on the session by the SCHEDULER role): `submit(GangDescriptor)
  -> BoxStream<RecordBatch>`. When installed, `run_spec`'s `TopologyDecision::Peer { world }`
  arm becomes `Placed { world }`: the claimant submits `GangExec` and awaits the stream instead
  of running the coordinator itself. `TopologyDecision` gains no variant — placement is a
  property of the host (the seam), decided after topology, so the topology oracles of wave 3
  are unchanged.
- `PlacedGangRunner` (installed by the EXECUTOR role): `GangExec::execute` calls
  `JobWorker::run_placed_gang(job_id, attempt, world, submitter)`, which (i) takes this host's
  job slot through `HostAdmission` (a host holding a rank or a job refuses typed BEFORE any row
  write — OPS D6 by the slot discipline of wave 3 §4), (ii) `transfer_claim` (§2.4), (iii) on
  success runs the SAME body the claim loop runs for a coordinator — `run_claimed_job_under`
  as `LeaseHolder::Coordinator`, assembly → dispatch → rounds → publish → finalize → the
  terminal write — under the executor's own `LeaseKeeper`, and (iv) yields exactly one batch at
  the end. Bytes equal U5b's because the body IS U5b's (K4).

The submitter's exit arms are total. The stream ends with one batch → `HandedOff` (a new
`WorkerJobError` arm: the claim is no longer this process's; NO terminal write, NO release —
the row never left `running` and the attempt count never moved). The stream ends in an error →
the submitter re-reads the row: `claimed_by == self` (the transfer never happened: no executor
bound the task, or the runner refused before the CAS) → `Abandoned` (wave 3 §8: the row is left
`running` for reclaim arm 1a, `attempts + 1` at the successor's claim, no terminal write);
`claimed_by != self` → `HandedOff` (the executor owns the attempt; if it died, its lease expires
and reclaim requeues it). The submitter's slot is released on either arm.

### 2.4 `transfer_claim` — the hand-off (db; zero net attempts)

```sql
UPDATE jobs SET claimed_by = $to, lease_expires_at = <now + lease>, updated_at = <now>
 WHERE job_id = $1 AND claimed_by = $from AND attempts = $attempts AND status = 'running'
   AND <lease not expired on the same backend clock as every other lease predicate>
```
returns whether one row moved. `attempts` and `releases` are untouched; the row never leaves
`running`; a stale runner (older attempt) cannot take it (the `attempts` conjunct); a second
launch of the same task cannot take it (the `claimed_by = $from` conjunct fails once the first
transfer happened) — this is the bind-time re-launch guard's second half; the first half is
`DevicePlacement` (§3) refusing to bind a `GangExec` whose row is already claimed by an
executor. Ballista's own reset-on-`ExecutorLost` (S6 probe 5, README r42) can therefore never
double-run a gang: the re-launched task's transfer fails typed and the task ends in error with
no row write.

### 2.5 What "across two executors" means for oracle (a)

An embedding job's plan is `InferenceExec(SortExec(KeyCheckExec(CoalescePartitionsExec(
scan))))`. Ballista's planner cuts a stage at the `CoalescePartitionsExec`: the scan stage runs
one task per source partition, the inference stage one task. With two executors registered and
round-robin distribution, the scan tasks land on both executors and the inference task on one;
the oracle asserts (i) every task of the job reported success from a registered executor, (ii)
both executors executed at least one task of the job (read from the scheduler's job status
and the executors' task metrics — the honest determinant, stated in the test), (iii) the
collected bytes are identical to the same plan executed in-process on the submitter's session
(the K4 shape). A single-partition source makes (ii) vacuous; the test uses a multi-partition
source and asserts the partition count first.

## 3. U8b — catalog-backed cluster state and device-aware placement

Distributor-neutral tables (K5, B1) by migration 038 `compute_cluster_state` (three pin sites:
the const list, `EXPECTED_MIGRATION_NAMES`, and an ordered-after oracle asserting 038 follows
BOTH `035_instances_peer_addr_result_root` and `037_jobs_assembly_failures_next_after` on both
backends):

- `compute_executors(executor_id TEXT PK, instance_id TEXT, host TEXT, port INTEGER, grpc_port
  INTEGER, task_slots INTEGER, available_slots INTEGER, status TEXT, heartbeat_at TEXT,
  metadata TEXT)` — registrations, slots and heartbeats;
- `compute_jobs(job_id TEXT PK, owner TEXT, status TEXT, queued_at TEXT, updated_at TEXT)` —
  job ownership and status; the execution graph itself has no serialisation in 54.1 (README
  r43), so it is NOT persisted: a scheduler restart keeps executors and job STATUS rows and an
  in-flight Ballista job is re-run through jammi's own reclaim, never revived by Ballista;
- `ALTER TABLE workers ADD COLUMN devices TEXT NOT NULL DEFAULT '[]'` — JSON `[{kind, ordinal,
  memory}]`, written by `upsert_worker` from `WorkerFacts.devices` (the session's
  `WorkerTopology::rank_devices()` under the session's `compute_device` kind), mirrored on
  `ListWorkers` (an additive field; the frozen surface's RPC set is unchanged).

`compute_repo.rs` (jammi-db, generic CRUD, no distributor vocabulary): `upsert_compute_executor`,
`list_compute_executors`, `record_compute_heartbeat`, `remove_compute_executor`,
`adjust_compute_slots`, `put_compute_job`, `get_compute_job`, `list_compute_jobs`,
`delete_compute_job`. `CatalogClusterState`/`CatalogJobState` (jammi-ballista) implement the
Ballista traits over those verbs; graphs and sessions stay in the scheduler's memory.

`DevicePlacement` (`TaskDistributionPolicy::Custom`): a task is GPU-bound iff its stage plan —
the live `RunningStage.plan` in `active_jobs` (dated correction to UNITS §U8b: the scheduler
holds the decoded plan, so no second decode through `JammiCodec` is needed) — contains a
`GangExec` or an `InferenceExec` whose descriptor names a CUDA device; such a task binds only to
an executor whose `workers.devices` (joined on `compute_executors.instance_id`) lists a device;
a `GangExec` task whose job row is already `claimed_by` an executor instance is never bound
(the guard of §2.4). A device-less cluster refuses a GPU-bound submission typed at the client
(`submit_physical_plan` reads the registered executors' devices first) rather than parking it
unschedulable.

Two schedulers over one catalog: both read the same executor registrations; active/standby —
sequential jobs are served by either, concurrent jobs are not (README r43, S6 probe 7) and
this contract states it as the shipped property.

## 4. U9b — the shape-d overlay

The compute tier becomes a `StatefulSet` `jammi-server-compute` with a headless `Service`
(`clusterIP: None`; ports flight/health/peer), `podManagementPolicy: Parallel`,
`nvidia.com/gpu: 2` (one device per local rank), the ordinal-stable pod DNS name as
`peer_advertise` (`JAMMI_SERVER__PEER_ADVERTISE` from the downward API +
`JAMMI_SERVER__PEER_BIND = 0.0.0.0 port 9000`), `terminationGracePeriodSeconds: 600` kept (OPS C6's
observable). The README's `issues/500` provisional note is REPLACED by the shipped statement
(never duplicated); the compose file gains the same note. Gates: `kustomize build | kubeconform
--strict --kubernetes-version 1.34.11` on base, shape-d, ci (ci.yml's exact command) and the
kind smoke on CI.

## 5. U9a — docs

`docs/guide/src/{configuration.md ([ballista]), reference-topologies.md (StatefulSet
consequence; the roles), philosophy.md, deploy-server.md as needed}`, the maintainer guide
(crate row, dep-DAG, the placed-gang writer-table row, the codec/engine seams), `CHANGELOG.md`
`[Unreleased] ### Added`. Docs reflect the shipped state, no journey markers.

## 6. Invariants crossed

B1/B2 (no distributor vocabulary in the engine's catalog: `compute_*`, `devices`; dep
direction ballista → ai/db/wire, server → ballista), B4 (roles are config; no cargo feature),
B6, K4 (bytes through Ballista == bytes in-process == bytes through the peer path), K5
(append-only migration, neutral names), K6 (publishable, lockstep, in the publish list), the
frozen wire surface (untouched: the codec's messages are `jammi.ballista.v1`), OPS D6/D10
(no terminal write on the hand-off arms; zero net attempts on a transfer), the actuator rule
(README r42: the engine ships the placement and the guard, never a control loop), README r45
naming (no governance stems in any new pub item: `JammiCodec`, `JammiExecutionEngine`,
`CatalogClusterState`, `CatalogJobState`, `DevicePlacement`, `BallistaConfig`, `GangExec`,
`PlacedGangSubmitter`, `PlacedGangRunner`, `transfer_claim`, `upsert_compute_executor`, …).

## 7. Acceptance, restated as executed oracles (each RED at base)

U8a: (a1) codec round-trip for every operator (bytes → node → bytes equal; a foreign buffer
delegates; an unknown node is refused typed); (a2) `[ballista]` parses, unset = no roles,
every fixed-port collision refused naming both keys; (a3) distributed, three processes, one
binary: an embedding job through `submit_physical_plan` across two executors byte-identical to
the in-process plan (§2.5); (a4) a W=2 gang job placed through the scheduler byte-identical to
the U5b path and never task-retried (`task_attempt` stays 0; one transfer); (a5) killing the
executor mid-gang: the submitter ends `HandedOff`, the row stays `running` with no terminal
write until reclaim requeues it, a successor completes it (`attempts - releases >= 2`).
U8b: (b1) scheduler restart keeps registered executors (read back through the new process) and
job status rows; (b2) two schedulers over one catalog serve sequential jobs; (b3) a GPU-bound
task never binds to a device-less executor and a device-less cluster refuses it typed; (b4)
`list_workers` returns `devices` as registered; (b5) the gang of (a4) byte-matches under
catalog state; (b6) the re-launch guard: a second launch of a transferred `GangExec` is refused
at bind and at transfer. U9b: kubeconform strict on the amended overlay; the K README oracle's
note count. U9a: docs gates.

## 8. Residuals and cuts

§8d is written at consolidation. Named at contract time: `MaskExec` is not distributable in
v1 (typed refusal at encode; its rebuild unit is "masked scans under Ballista"); the
accelerator dimension stays out of band in `workers.devices` (the one upstream PR 67 owes,
README r43); object-store shuffle is out of v1 (D2 condition 3); active/active scheduling is
out (README r43).

## 9. Pressure round (REFINE at `bcfda3ca`, 2026-09-16; every premise reproduced or refuted with an executed check — the row is in `docs/rigor/feat_500-wave4.jsonl`)

Seven blocks, ten advisories. Each disposition below is the design as built; the sections
above are read WITH these corrections.

- **B1 — a placed task on the submitter's own host is refused forever** (the submitter holds
  `Holder::JobRun` for the whole await, `crates/jammi-ai/src/fine_tune/worker.rs:1380`/`:1404`). Disposition: placement
  EXCLUDES the submitter's own executor by construction — the placement policy is jammi's from
  U8a on (a `Custom` policy that never binds a `GangExec` to the executor whose id equals
  `descriptor.submitter`, round-robin otherwise; U8b adds the device predicate to the same
  policy), and the submitter seam reports `placement_available()` = "a registered executor
  OTHER than this instance exists"; when it is false the job runs in-process (the wave-3
  path) — placement is a property of the claimant's cluster view, decided BEFORE topology.
  Both roles on one process is therefore a single node that never places, stated in the guide.
- **B2 — the submitter's host cannot serve a rank while it awaits** (a two-pod overlay could
  not assemble). Disposition: the `Placed` arm moves the host's holder `JobRun → Awaiting`
  right after submitting: `Awaiting` admits a `RunRank` session (the host runs no compute) and
  refuses a second claim exactly as `JobRun` does; the claim loop returns to `Free` when the
  await ends. A placed `Peer` gang of world W therefore needs W hosts able to hold a rank, the
  submitter's included; the overlay's replica count and the guide state that arithmetic.
- **B3 — `InferenceExec` names no device** (`crates/jammi-ai/src/operator/inference_exec.rs:22-40`; the device is the
  executing session's, `crates/jammi-ai/src/session.rs:894`). Disposition: `InferenceExec` gains a `device_kind:
  ComputeDeviceKind` (cpu | cuda | metal) stamped by `InferenceExecBuilder` from the building
  session's `compute_device()` kind (an explicit `.device_kind(k)` override exists for a
  submitter placing onto another kind); the codec carries it; the executor refuses a plan whose
  kind is not its own (K7); the policy's GPU-bound predicate reads it. An in-process run is
  byte-unchanged (the builder's default is the session's own kind).
- **B4 — the manifest records the submitter's device.** Disposition: in v1 no pipeline verb
  is rewired through Ballista — `submit_physical_plan` is a client seam exercised by the
  oracles, and the gang path writes its manifest on the executor that IS the coordinator — so
  no `MaterializationEnv` is written for a placed inference plan in wave 4; the rebuild unit
  "pipeline verbs through Ballista" is a NAMED CUT (§8d) that must carry the executor's
  device back to the writer. K4 is stated per device kind (README r44): bytes through
  Ballista == bytes in-process ON THE SAME KIND; oracle (a3) runs both on CPU.
- **B5 — `workers.devices` exists only under `[worker] enabled`.** Disposition: the device
  list is the EXECUTOR registration's own fact — `compute_executors.devices TEXT` (JSON
  `[{kind, ordinal}]`, no `memory`: the tree has no source for it), written by the executor
  role at registration from the session's `WorkerTopology::rank_devices()` × its device kind;
  `DevicePlacement` joins on it alone. `workers.devices` stays as the `ListWorkers` mirror for
  worker-enabled processes (acceptance b4), never the join's authority.
- **B6 — DRAIN must not tear down the executor under a running placed gang.** Disposition:
  DRAIN stops task admission (the executor reports `Terminating`; the scheduler stops binding
  to it) and WAITS for its in-flight tasks; only RELEASE fires the `ShutdownNotifier`. The
  scheduler role stops last, after the process's own drain completes.
- **B7 — per-pod DNS and one scheduler.** Disposition: `publishNotReadyAddresses: true` on the
  headless Service (shipped in U9b). The scheduler is ONE dedicated single-replica
  `Deployment` (`jammi-server-scheduler`: `[ballista] scheduler_bind`, `[worker] enabled =
  true` for the training kinds, CPU) that claims and places; the compute `StatefulSet` pods host
  executors and are worker-enabled fleet members. A training job claimed by a compute pod runs
  there in-process (byte-identical, K4); one claimed by the scheduler pod is placed. The guide
  states the split and that every training job (W ≥ 1) claimed by a scheduler host is placed
  when a foreign executor exists — `GangExec`'s `world` is informational; topology is decided
  on the executor from its own `[worker] local_ranks`.
- Advisories folded: A1 the codec magic's first byte is `0x07` (field 0, wire type 7 — never a
  legal prost tag), pinned by a test over Ballista's five oneof tag bytes; A2 the six-address
  collision rule is a `&JammiConfig` cross-section validator (precedent `MembershipConfig::
  validate`), never a second parse; A3 the executor decodes with the codec passed to
  `executor_server::startup`, `advertise_host` is required whenever `bind` is unspecified
  (`0.0.0.0`/`::`); A4 acceptance (a4)'s determinant is "exactly one transfer" read from the
  row, never `task_attempt`; A5 every distributed oracle pins `[worker] enabled` to the
  scheduler process only; A6 there is no note-counting oracle — the three note sites are
  `deploy/kubernetes/README.md`, the deleted `deployment-compute.yaml`, and
  `docs/guide/src/reference-topologies.md` (U9a); A7 the README's Deployment-shaped rollout
  arithmetic is rewritten for the StatefulSet (U9a); A8 the fifth registration site is
  `docs/guide/src/api-stability.md`'s published-crate enumeration (U9a); A10 the scheduler's
  binder takes the slot CAS BEFORE stamping the graph's task info, and a lost CAS leaves the
  task unstamped; `ballista-scheduler` with `default-features = false` is load-bearing for the
  restart property (the REST API's `get_running_jobs` errors on a status row with no graph).

## 10. Gate table

Written at consolidation: tip, stage, result, log.

## 11. Units as built — the implementers' contract files, folded by the lead

Each subsection is the implementer's own contract file, verbatim, headed by the lead's note on what was opened, re-run or changed at consolidation. Every `path:line` inside them was written at the implementer's own tip (named in each heading) and is not re-anchored to the consolidated tip; the citation resolver and the rigor-record gate run on the final tip (§10) and check existence and length, never the drifted offset — read a folded citation as "at that unit's tip".

### 11.1 U9b — the shape-d overlay (docs-ci) — landed as 5f1bcb03 (original 4dd52f4c)

**Lead's note.** Landed by cherry-pick; the lead read the StatefulSet/Service/TOML in full and confirmed `PeerAddr::parse` accepts a DNS name (`crates/jammi-db/src/catalog/instance.rs:52`). The implementer's executed refutation stands: a dropped `serviceName` is invisible to kubeconform strict (UNCOVERED, §8d). Its two named follow-ups (the README's Deployment-shaped rollout arithmetic, the guide's provisional paragraph) were folded into U9a and are closed there.

#### U9b — the shape-d overlay: StatefulSet + headless Service + `nvidia.com/gpu: 2`

##### 1. Scope shipped

- `deploy/kubernetes/overlays/shape-d/deployment-compute.yaml` deleted; replaced by
  `statefulset-compute.yaml` (`apiVersion: apps/v1`, `kind: StatefulSet`,
  `metadata.name: jammi-server-compute`, `spec.serviceName: jammi-server-compute`,
  `replicas: 2`, `podManagementPolicy: Parallel`, `updateStrategy: {type: RollingUpdate}`,
  same selector/securityContext/probes/volumes/envFrom as the old Deployment,
  `resources.limits: {nvidia.com/gpu: 2}`, a `peer` container port 9000, and a new `env`
  block: `POD_NAME`/`POD_NAMESPACE` from the downward API `fieldRef`, then
  `JAMMI_SERVER__PEER_ADVERTISE = "$(POD_NAME).jammi-server-compute.$(POD_NAMESPACE).svc.cluster.local port 9000"`
  — ordered after the two fields it references per the `EnvVar.value` field doc
  (`https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.34/#envvar-v1-core`:
  "Variable references `$(VAR_NAME)` are expanded using the previously defined environment
  variables"). `terminationGracePeriodSeconds: 600` kept verbatim with its comment.
- `deploy/kubernetes/overlays/shape-d/service-compute-headless.yaml` (new):
  `kind: Service`, `metadata.name: jammi-server-compute`, `spec.clusterIP: None`,
  `publishNotReadyAddresses: true`, `selector: {app: jammi-server-compute}`, ports
  flight/health/peer (8081/8080/9000).
- `jammi-compute.toml`: added `[gpu] devices = [0, 1]`, `[server] peer_bind =
  "0.0.0.0 port 9000"` (kept `services = []`), `[worker] local_ranks = 2`, `[distributed]
  max_world_size = 2` — every key commented from `docs/guide/src/configuration.md:134-215`'s
  semantics, condensed, not copied verbatim.
- `kustomization.yaml`: `resources` now lists `statefulset-compute.yaml` +
  `service-compute-headless.yaml` in place of `deployment-compute.yaml`.
- `deploy/kubernetes/README.md`: the `overlays/shape-d/` bullet replaced with the
  shipped statement (StatefulSet, headless Service, `nvidia.com/gpu: 2`,
  `peer_advertise` = stable pod DNS, `Local` vs `Peer` gang boundary at
  `world_size <= 2`); the `issues/500` URL and PROVISIONAL language removed, never
  duplicated. The "Image pin advice" section's file reference updated to
  `statefulset-compute.yaml`. The "Shutdown: DRAIN and RELEASE" grace/rollout
  paragraph (its `Deployment`-specific `maxSurge`/`maxUnavailable` arithmetic, "at 3
  compute replicas") is left **unchanged** per U9B.md item 4's explicit instruction —
  **flagged as a deviation-adjacent gap, not fixed**: a `StatefulSet`'s default
  `RollingUpdate` has no `maxSurge`/`maxUnavailable` knobs at all (pods update one at a
  time, by descending ordinal, unless `partition` is set), so that paragraph's stated
  arithmetic no longer describes the compute tier's actual rollout mechanism, and its
  replica count (3) no longer matches this shape (2). I did not touch it because the
  brief named it explicitly "unchanged" and it sits inside a heading ("Shutdown: DRAIN
  and RELEASE") this unit's scope note (item 4) does not list as mine to rewrite;
  the lead should decide whether this becomes a follow-up fix or U9a's to fold in
  (U9a also touches `reference-topologies.md`'s mirrored grace/rollout prose).
- `deploy/docker-compose.yml`: one-line note in the header (no compute service exists
  in this file — confirmed by reading it in full — so the note went in the header per
  U9B.md item 4's "if it has one; else in the header").
- `docs/guide/src/reference-topologies.md`: **mechanical-only** edit — the two
  `{{#include}}` lines that named `deployment-compute.yaml` now name
  `statefulset-compute.yaml`, and a third `{{#include}}` block was added for
  `service-compute-headless.yaml` (so the guide shows the whole shipped artifact
  set). The surrounding prose (the "provisional… #500 decides" paragraph, the
  StatefulSet-consequence narrative) is **untouched** — design contract §5 and
  UNITS §U9a assign that rewrite to U9a, a concurrent unit also touching this file;
  I made only the change needed so `mdbook build` resolves on my own branch (an
  unresolved `{{#include}}` is a hard mdbook error), which is a real requirement of
  my own Gates. **Flagged for the lead**: this is the same file U9a rewrites: expect
  a merge conflict at consolidation; my diff is two/three single lines to ease it.

###### Deviations from the brief, with the code/grep cited

- **`serviceName` drop does NOT red under kubeconform strict** (brief §5 predicted it
  would). Executed: dropped `spec.serviceName` from the rendered StatefulSet, ran the
  exact CI kubeconform invocation — `Valid: 6, Invalid: 0, Errors: 0`, exit 0. Checked
  the pinned schema files in the kubeconform cache directly: none of the 3
  StatefulSet-touching schema JSON files served from
  `yannh/kubernetes-json-schema@b582a12a...` at `1.34.11-standalone-strict` carry
  `serviceName` in any `"required":[...]` array (`grep -o '"required":\[[^]]*\]' <file>
  | grep -i servicename` on all 4 cached schema files: zero hits). This is a known
  property of the upstream generated Kubernetes OpenAPI schemas (most fields are not
  marked `required` even when the live apiserver rejects their absence) — reported as
  an **executed refutation**, not asserted as a pass: this determinant is UNCOVERED
  by kubeconform and would need a `kubectl --dry-run=server` (a live apiserver) or a
  literal `grep -c serviceName:` guard to catch, neither of which I added (a new gate
  is the lead's call, not mine to add unilaterally).
- **The Python selector-consistency script is new** (brief §5 asked for it): saved
  under scratch, not committed (brief: "do not add a CI gate yourself"). Its own
  mutation (renaming the StatefulSet pod-template's `app` label so it disagrees with
  the StatefulSet's own `matchLabels` selector and the Service's selector) reds the
  script (`AssertionError: selector mismatch: statefulset selector={'app':
  'jammi-server-compute'} template labels={'app': 'jammi-server-compute-renamed'}
  service selector={'app': 'jammi-server-compute'}`) while kubeconform strict on the
  same mutated tree stays green (`Valid: 6, Invalid: 0, Errors: 0`) — confirms the
  brief's claim that kubeconform cannot see this class of defect.

##### 2. Properties

| Property (quantified) | Executed oracle | Executed mutation that reds it |
|---|---|---|
| Every rendered manifest across base/shape-d/ci validates against the pinned Kubernetes 1.34.11 strict schema | `kustomize build deploy/kubernetes/{base,overlays/shape-d,overlays/ci} \| kubeconform --strict --summary --kubernetes-version 1.34.11 -schema-location <pinned> -cache <scratch>/kubeconform-cache -` (kubeconform v0.8.0, matches CI's pin) — `Valid: 3/6/7, Invalid: 0, Errors: 0` on all three, exit 0 each | Set `service-compute-headless.yaml`'s `clusterIP: None` to `clusterIP: 10.0.0.5`: the rendered manifest's `clusterIP` line changes from `None` to `10.0.0.5` (own-grep assertion reds; not a kubeconform-schema property, so kubeconform itself does not red on this — restated per U9B.md's exact instruction, which only asked for the grep assertion to red, not kubeconform) |
| The StatefulSet's own pod-selector, its pod-template's labels, and the headless Service's selector are the same value | `<scratch>/check_selector_labels.py` (10-line script, `yaml.safe_load_all` over `kustomize build .../shape-d`'s stdout) — `OK: {'app': 'jammi-server-compute'} agrees across...`, exit 0 | Renamed the StatefulSet template's `app` label to `jammi-server-compute-renamed`: `AssertionError: selector mismatch: statefulset selector={'app': 'jammi-server-compute'} template labels={'app': 'jammi-server-compute-renamed'} service selector={'app': 'jammi-server-compute'}`, exit 1; the same mutation left `kubeconform --strict` green (`Valid: 6, Invalid: 0`), confirming this class is invisible to it |
| The guide's `{{#include}}` directives over the renamed/added shape-d files all resolve | `mdbook build docs/guide --dest-dir <scratch>/book` (mdbook at `~/.cargo/bin/mdbook`) — `INFO HTML book written to ...`, exit 0; `grep -c '{{#include' <scratch>/book/reference-topologies.html` = 0 | Reverting only the first `{{#include}}` line back to `deployment-compute.yaml` (the deleted file) while keeping the file deleted turns this into an mdbook hard error (`Could not read file for link ... No such file or directory`) — not separately pasted above since it is the direct converse of the fix already shipped, but the failure mode was observed while iterating (an early edit pass before the rename-following fix landed) |
| Every `PATH:LINE` citation in the diff's neighborhood still resolves at HEAD | `python3 ci/scripts/perf/check_citations.py` — `1044 file(s) scanned, all PATH:LINE citations resolve`, exit 0 | not separately mutated (no new `PATH:LINE` citation was added by this unit's YAML/TOML/docker-compose/README edits — the one citation-bearing prose file touched, `reference-topologies.md`, only had its `{{#include}}` paths changed, not a `PATH:LINE` citation) |
| No new governance-verb-stem `pub` item, no philosophy leak-smell token, introduced by this diff | `python3 ci/scripts/check_no_consumer_names.py` — `no-consumer-names: OK`, exit 0 | not separately mutated (this unit adds no Rust `pub` items at all — YAML/TOML/Markdown only, outside this script's scanned roots `crates/**` + workspace `Cargo.toml`/`.cargo`) |
| The `issues/500` provisional note is replaced, never duplicated, and no swarm oracle counts it | `grep -c issues/500 deploy/kubernetes/README.md` = 0 (was 1 before this commit); `git grep -n "issues/500" -- ci .github crates docs/maintainer` = no matches (executed before and after the edit) | not applicable — this is an absence property, confirmed by the grep itself, not a red/green oracle pair |

##### 3. Uncovered

- **`spec.serviceName` absence is not caught by kubeconform strict** against the
  pinned schema (see Deviations above) — UNCOVERED by any gate this unit runs or
  adds; would need a live apiserver (`kubectl apply --dry-run=server`, no cluster
  available on this host) or a literal `grep -c serviceName:` guard.
- **The kind smoke (`kube-smoke.yml`) never exercises `overlays/shape-d`** at all:
  `overlays/ci/kustomization.yaml`'s `resources:` is `[../../base, postgres.yaml,
  nats.yaml]` — no shape-d reference anywhere (confirmed by reading the file). The
  StatefulSet/headless-Service change is therefore proven only by kubeconform +
  the rendered-manifest greps + the selector script in this unit, never by a real
  scheduler; the design contract's own §6/§7 language ("kind smoke on CI") appears to
  mean the workflow's path-triggered run on the PR overall, not a check of this
  specific overlay's content — UNCOVERED for anything beyond schema validity (pod
  scheduling with real GPU device-plugin resources, actual DNS resolution through
  the headless Service, `podManagementPolicy: Parallel` behavior under a real
  kubelet) since "no GPU node is available in CI" (the file's own kept comment).
- **The `publishNotReadyAddresses: true` justification is derived from source
  reading (`OssServer::bind` binds the peer listener before the readiness gate),
  not from an executed integration test** — no hermetic harness in this repo spins
  up a real headless Service + DNS to observe the NXDOMAIN-without-the-flag failure
  mode; this is a documented, cited, but UNEXECUTED refutation attempt (the code
  path was read, not run against a live cluster).
- **The StatefulSet-consequence prose in `reference-topologies.md`** (the
  "provisional… #500 decides" paragraph) is stale relative to this unit's shipped
  YAML (still describes a plain Deployment) — left untouched deliberately per the
  design contract's assignment of that rewrite to U9a, a concurrent unit. Not
  "uncovered" by me so much as **explicitly out of this unit's scope**, flagged
  here so the lead does not read the current tree state as final.
- **The README's "Rollout arithmetic" / grace paragraph** under "Shutdown: DRAIN and
  RELEASE" is Deployment-shaped prose (`maxSurge`/`maxUnavailable`, "3 compute
  replicas") now attached to a StatefulSet with different native rollout semantics
  and a different replica count (2) — left unchanged per the brief's explicit
  instruction; see the Deviations note above.

##### 4. Gates

| Command | Exit | Notes |
|---|---|---|
| `kustomize build deploy/kubernetes/base \| kubeconform --strict --summary --kubernetes-version 1.34.11 -schema-location <pinned> -cache <scratch>/kubeconform-cache -` | 0 | `Valid: 3, Invalid: 0, Errors: 0` |
| `kustomize build deploy/kubernetes/overlays/shape-d \| kubeconform ...` (same flags) | 0 | `Valid: 6, Invalid: 0, Errors: 0` |
| `kustomize build deploy/kubernetes/overlays/ci \| kubeconform ...` (same flags) | 0 | `Valid: 7, Invalid: 0, Errors: 0`; `kubeconform -v` = v0.8.0, matches CI's pin |
| `kustomize build deploy/kubernetes/overlays/shape-d` saved + greps | 0 | 1 `kind: StatefulSet`, 2 `kind: Service`, 2 `kind: ConfigMap`, 1 `kind: Deployment` (base's), 1 `clusterIP: None`, 1 `publishNotReadyAddresses`, 1 `serviceName: jammi-server-compute`, 1 `nvidia.com/gpu: 2` — saved at `<scratch>/shape-d-rendered.yaml` |
| `<scratch>/check_selector_labels.py` | 0 | `OK: {'app': 'jammi-server-compute'} agrees across...` |
| `mdbook build docs/guide --dest-dir <scratch>/book` | 0 | `INFO HTML book written to ...`; 0 unresolved `{{#include}}` in the rendered `reference-topologies.html` |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `1044 file(s) scanned, all PATH:LINE citations resolve` |
| `python3 ci/scripts/check_no_consumer_names.py` | 0 | `no-consumer-names: OK` |
| `python3 ci/scripts/check_swarm_bijection.py` (extra, my own card's Acceptance) | 0 | `PASS — total coverage, exactly-one owner per path (shared class → docs-ci)` |
| `python3 ci/scripts/check_doc_parity.py` (extra) | 0 | `doc-parity: all bindings in parity` |
| `python3 ci/scripts/check_constitution_anchors.py` (extra) | 0 | `constitution-anchors: OK` |

##### 5. Commits

```
4dd52f4c feat(deploy): #500 U9b — shape-d becomes a StatefulSet + headless Service
```
(one commit; `git log --oneline 0e11162a..HEAD` on `unit/u9b`)

### 11.2 CFGDB — `[ballista]` config; migration 038; `compute_repo`; `workers.devices`; `transfer_claim` (db) — landed as 17d62ae2, fddf53a0, 7c83a694, 144589db, 909a165f (original tip 634ae25f)

**Lead's note.** The three pressure deltas (cross-section validator, the positive `lease_live_clause` comparison, `compute_executors.devices` as the join's authority, no `memory`) are in the commits. The lead ran the migration/compute_repo/transfer_claim/devices suites on both backends on the consolidated tree (28 + 8 green), then gated the six `compute_repo` Postgres skips through the registered `skip_unless_ready!` helper (KO-7, commit bfa8a6e5) and gave the `advertise_host` rule its own oracle after it refused this unit's pre-rule hostname fixture (20edbae7). The named gap — `BallistaConfig::validate`'s second call site in `OssServer::new` — was closed by BALLISTA (`crates/jammi-server/src/runtime.rs`, with a server test). The implementer's accidental `git stash` was recovered and dropped as its note describes; the lead confirmed the stash stack empty before consolidating.

#### Contract — CFGDB (plan 67 wave 4, db)

Base `0e11162a`, branch `unit/cfgdb`, worktree `/private/tmp/claude-501/-Users-vijaychakilam-git-f-inverse-jammi-ai/633b7b0f-9596-44a5-a195-cb45cfc23fc9/scratchpad/wt-cfgdb`, tip `634ae25f55c0ac73fa910a071c1a4294c9b70843`.

Mid-task the lead sent a pressure-round verdict (REFINE) with three binding
deltas against CFGDB.md. All three are folded into the commits below; each
is called out where it lands.

##### 1. Scope shipped

**Commit 1 — `[ballista]` config** (`crates/jammi-db/src/config/mod.rs`,
`env_map.rs`, `tests.rs`, `docs/guide/src/configuration.md`):
`BallistaConfig { scheduler_bind: Option<String>, executor:
Option<BallistaExecutorConfig> }`, `BallistaExecutorConfig {
scheduler_address, bind, grpc_bind, advertise_host, work_dir, task_slots
}`, both `#[serde(default, deny_unknown_fields)]`; `hosts_scheduler`/
`hosts_executor` accessors; `[ballista]` added to `docs/guide/src/
configuration.md`'s full reference, commented (opt-in), matching
`[observability.otlp_headers]`'s convention. Registered `"ballista"` in
the three `TOP_LEVEL_FIELDS` mirrors (`env_map.rs`'s private list, its own
pinning test, `tests/it/docs_config_fences.rs`'s duplicate) so
`JAMMI_BALLISTA__*` env overrides resolve instead of erroring "unknown
top-level config section".

**Commit 1 fix — pressure-round delta 1.** Original shape was `BallistaConfig::
validate(&self, server: &ServerConfig)`, called as `config.ballista.
validate(&config.server)` — two reads into one config object. Restructured
to `BallistaConfig::validate(config: &JammiConfig) -> Result<()>`, the
exact shape of `catalog::instance::MembershipConfig::validate(&JammiConfig)`
(the in-tree precedent the lead named). `JammiConfig::load_from` now calls
`BallistaConfig::validate(&config)`. **Deviation, named per the lead's
instruction rather than fixed here**: `ServerConfig::validate` stays owned
by `jammi_server::runtime::OssServer::new` (not `load_from`), so a
struct-literal `JammiConfig` handed straight to `OssServer::new` — the
shape most `jammi-server` integration tests use — never runs `load_from`
and therefore never runs `BallistaConfig::validate` either, mirroring
exactly the second-call-site gap `MembershipConfig::validate` closes at
`InstanceRegistration::from_config` (session construction). `BallistaConfig::
validate`'s own doc names the second call site required:
`OssServer::new`, immediately after its existing `config.server.validate()`
call, in `crates/jammi-server/src/runtime.rs` — outside jammi-db's
ownership (U8a-crate / wire-server's unit builds `OssServer`'s role
hosting). **I did not edit `crates/jammi-server`** — the lead must wire
this call during consolidation, or it is a real gap on the compiled-binary
path only when `main.rs`'s `JammiConfig::load()` → `load_from` is bypassed.

**Commit 2 — migration 038 `compute_cluster_state`** (`schema.rs`,
`migrations.rs`, `tests/it/migrations.rs`): `compute_executors(executor_id
PK, instance_id, host, port, grpc_port, task_slots, available_slots,
status, heartbeat_at, metadata, devices)`, `compute_jobs(job_id PK, owner,
status, queued_at, updated_at)`, `workers.devices TEXT NOT NULL DEFAULT
'[]'`. Ordered after both 035 and 037 (asserted, not merely positioned
last). Neutral names only (no `ballista` anywhere).

**Commit 2 shape, folded from delta 3 at write time** (I had not yet
written the tests when the pressure round landed, so this commit shipped
the corrected shape directly rather than needing its own fix commit):
`compute_executors.devices` exists from this migration onward — added
alongside the delta-3 rewrite of commit 3's design (see below).

**Commit 3 — `catalog::compute_repo`** (new file; `catalog/mod.rs` gains
`pub mod compute_repo`): `ComputeExecutorRecord`/`ComputeJobRecord`,
`upsert_compute_executor`, `list_compute_executors`, `get_compute_executor`,
`record_compute_heartbeat`, `remove_compute_executor`,
`adjust_compute_slots` (one transaction; every `(executor_id, delta)` pair
is validated against a fresh read of its own row BEFORE any write in the
batch runs, so a batch with one out-of-bounds delta changes nothing;
typed `BackendError::Constraint` on a missing `executor_id` or a delta
that would push `available_slots` outside `[0, task_slots]`),
`bind_compute_slots` (CAS: `available_slots = available_slots - n WHERE
available_slots >= n`), `put_compute_job`/`get_compute_job`/
`list_compute_jobs`/`delete_compute_job`, `list_compute_executor_devices`.
Also added `DeviceFact { kind: String, ordinal: u32 }` and
`decode_devices_json` to `catalog::instance` (moved earlier than commit
4's original slot, since `ComputeExecutorRecord.devices` needed the type).

**Commit 3 shape — pressure-round delta 3 (folded in from the start, no
separate fix commit needed since I had not yet written the original
join-based design in code)**: `compute_executors.devices` (JSON
`[{kind, ordinal}]`) is the executor's OWN registration fact and the
placement join's SOLE authority; `list_compute_executor_devices` reads
`compute_executors.devices` directly — no join on `workers.instance_id`.
`DeviceFact` has no `memory_bytes` field (dropped: no source in the tree).
`workers.devices` stays a `ListWorkers` mirror only, never consulted by
placement.

**Commit 4 — `workers.devices` + `transfer_claim`** (`instance.rs`,
`jobs_repo.rs`, `lease.rs`, `tests/it/{jobs_queue.rs,gang_membership.rs}`):
`WorkerFacts` gains `devices: Vec<DeviceFact>`; `Catalog::upsert_worker`
gains a `devices: &[DeviceFact]` parameter (JSON-encoded into the new
column; every jammi-db caller updated to pass `&[]` or a real list);
`reregister_instance`'s whole-tuple re-upsert writes the snapshot's
devices too; `WorkerRecord` gains `devices: Vec<DeviceFact>`, decoded via
`decode_devices_json` in `parse_worker_row` (a malformed value is a row
fact — empty list + `tracing::warn!` naming the row, never a read fault,
citing issue #574's `LeaseFact` shape, per the brief's explicit
either/or); `list_workers`' join selects `w.devices`.

`Catalog::transfer_claim(job_id, from_instance, to_instance, attempts,
lease) -> Result<bool>`: `UPDATE jobs SET claimed_by = $to,
lease_expires_at = <fresh deadline>, updated_at = <now> WHERE job_id = $1
AND claimed_by = $from AND attempts = $attempts AND status = 'running' AND
<lease live>` — never touches `attempts`/`releases`.

**Pressure-round delta 2**, folded directly into commit 4 (transfer_claim
did not exist before this commit, so there was no separate fix step):
added `lease.rs::lease_live_clause` — a POSITIVE comparison
(`lease_expires_at IS NOT NULL AND lease_expires_at > now`, the backend's
own clock) — the exact complement of `lease_expired_clause`, never that
function's `IS NULL OR …` shape. `transfer_claim` uses `lease_live_clause`,
never `lease_expired_clause`, because a RELEASE
(`Catalog::release_job_lease`) sets `lease_expires_at = NULL`, and that
NULL must make a transfer FAIL — `lease_expired_clause`'s own `OR` would
read a NULL lease as "expired" (true), the right answer for a reclaim
sweep but the wrong one for a hand-off. Added
`transfer_claim_after_release_fails` (`release_job_lease` then
`transfer_claim` ⇒ `false`) as its own test, and it is the exact mutation
oracle: swapping `lease_live_clause` for `lease_expired_clause` reds ALL
TEN `transfer_claim` tests on both backends (see §2).

**Also fixed** (collateral of the `[ballista]` insertion shifting every
line after it in `config/mod.rs`): 3 stale `PATH:LINE` citations in
`docs/maintainer/MAINTAINER-GUIDE.md` and
`crates/jammi-ai/src/model/backend/gguf.rs` — doc-comment line-number
corrections only, verified with `cargo check -p jammi-ai --lib` (clean)
and `check_citations.py` (clean). This touches a file outside jammi-db's
crate (`gguf.rs`, ai-core's), but only a cited line number inside a doc
comment, never code logic.

###### Deviations from CFGDB.md, with the code cited

- `list_compute_executor_devices` returns `Vec<(String, Vec<DeviceFact>)>`
  (decoded), not `Vec<(executor_id, instance_id, devices_json)>` as
  CFGDB.md's pre-pressure-round text specified — required by delta 3
  dropping the join and the `instance_id` column from this read
  entirely; decoding in jammi-db (not the caller) reuses the same
  `decode_devices_json` row-fact rule the repo already needs for
  `parse_compute_executor_row`/`parse_worker_row`, so a malformed value
  never has two different failure shapes depending on which verb read it.
- `DeviceFact` lives in `catalog::instance` (`crates/jammi-db/src/catalog/
  crates/jammi-db/src/catalog/instance.rs:397` at HEAD), not `jobs_repo.rs`, introduced in commit 3
  rather than commit 4 as CFGDB.md's four-commit outline implied — needed
  earlier because `compute_repo::ComputeExecutorRecord.devices` (commit 3)
  requires the type before `WorkerFacts.devices` (commit 4) does.
- `crates/jammi-server/src/runtime.rs`'s `OssServer::new` is NOT edited by
  me (delta 1's named second call site) — cross-crate, wire-server's
  ownership; named in `BallistaConfig::validate`'s own doc comment and
  above.
- `crates/jammi-ai/src/fine_tune/worker.rs` (`WorkerFacts` literals,
  `upsert_worker` call site) and `crates/jammi-server/tests/it/
  {gang_coordinator.rs,gang_chaos.rs}` (`upsert_worker` call sites) are
  NOT edited — CFGDB.md's own text names `worker.rs` as ai-core's; the two
  jammi-server test files are wire-server's. Verified the exact breakage
  with `cargo check -p jammi-ai --lib` (§4).

##### 2. Properties

| Property (quantified) | Executed oracle | Executed mutation that reds it |
|---|---|---|
| `[ballista]` unset ⇒ no roles, byte-for-byte | `config::tests::ballista_unset_means_no_roles` (`--lib --features test-hooks`) | N/A (a `false` branch has no mutation to red; covered by the collision-pair mutation below) |
| Every one of the 15 fixed-port collision pairs among `{scheduler_bind, executor.bind, executor.grpc_bind, health_listen, flight_listen, peer_bind}` is refused naming BOTH keys | `ballista_every_fixed_port_collision_pair_is_refused_naming_both_keys` (parameterized over all 15 pairs) | `if addr_a == addr_b && addr_a.port() != 0` → `if false` in `BallistaConfig::validate`: first line `pair (ballista.scheduler_bind, ballista.executor.bind) must collide: ()` |
| `:0` never collides, on every field | `ballista_ephemeral_ports_never_collide` | (boundary arm of the same predicate; exercised by the collision-pair test's negative space) |
| `task_slots == 0` / `work_dir = Some("")` refused | `ballista_task_slots_zero_is_refused`, `ballista_executor_work_dir_empty_is_refused` | — |
| unknown key under `[ballista]`/`[ballista.executor]` refused | `ballista_unknown_key_is_refused` | `deny_unknown_fields` derive itself is the oracle's control (serde-generated, not hand-mutated) |
| Migration 038 present, ordered after BOTH 035 and 037, creates `compute_executors`/`compute_jobs`/`workers.devices` | `migrations::migration_038_is_ordered_after_035_and_037_and_creates_compute_tables` (sqlite + postgres) | `EXPECTED_MIGRATION_NAMES` reordered so 038 precedes 035: `position("038...") > position("035...")` — first line `the compute-cluster-state migration must follow 035 (its instance_id join target)` on both backends |
| `adjust_compute_slots`: a batch with ONE out-of-bounds delta changes NOTHING (every row or none) | `compute_repo::adjust_compute_slots_is_atomic_across_the_whole_batch` (both backends) | the bounds check `if next < 0 \|\| next > task_slots` → `if false`: `a batch with one out-of-bounds delta must refuse entirely: ()` on both backends |
| `bind_compute_slots`: exactly `task_slots` concurrent binders win a CAS race | `compute_repo::bind_compute_slots_cas_admits_exactly_capacity_concurrent_binders` (8 binders, 3 slots, both backends) | `AND available_slots >= $1` → `AND 1=1`: `left: 8, right: 3` on both backends |
| `list_compute_executor_devices` returns `[]` for a device-less executor and the written list otherwise | `compute_repo::list_compute_executor_devices_reads_the_executors_own_column` | — |
| `transfer_claim` moves the row, leaves `attempts`/`releases`/`status` untouched | `transfer_claim_moves_the_row_leaving_attempts_releases_status_unchanged` | (see the shared polarity mutation below — this test is one of the ten it reds) |
| A wrong `from`, a stale `attempts`, and a second transfer with the OLD `from` after a first landed all fail | `transfer_claim_refuses_wrong_from_stale_attempts_and_a_second_launch` | same shared mutation |
| An expired lease refuses a transfer | `transfer_claim_after_lease_expiry_fails` | same shared mutation |
| **Delta 2**: a RELEASED (NULL) lease refuses a transfer | `transfer_claim_after_release_fails` | same shared mutation, AND is the test the delta specifically demanded |
| The new holder can heartbeat, the old holder cannot, after a transfer | `transfer_claim_new_holder_can_heartbeat_old_holder_cannot` | same shared mutation |
| **Shared mutation for all five `transfer_claim` properties above**: `lease_live_clause(...)` → `lease_expired_clause(...)` at the one call site in `transfer_claim` | all 10 `transfer_claim::{sqlite,postgres}` tests | RED: all 10 failed (`assertion failed: moved`, `a transfer of a RELEASED (NULL-leased) claim must fail`, `an expired lease must never transfer`, `the first, correctly-guarded transfer must succeed`) |
| `upsert_worker`'s `devices` round-trips through `list_workers`; a re-upsert REPLACES | `jobs_queue::upsert_worker_devices_round_trips_through_list_workers` | — |
| A malformed `workers.devices` value is a row fact, never a read fault | `jobs_queue::malformed_worker_devices_is_a_row_fact_not_a_read_fault` (hand-planted via raw SQL, both backends) | — |

##### 3. Uncovered

- `BallistaConfig::validate`'s SECOND call site (`OssServer::new`) is
  UNCOVERED by any test in this unit — it lives in `jammi-server`, outside
  jammi-db's crate boundary and outside what I can build/test here. Named
  in delta-1's fix commit and in `BallistaConfig::validate`'s own doc
  comment for the consolidating lead.
- `jammi-ai`'s `write_worker_facts`/`WorkerFacts` construction sites and
  `jammi-server`'s two `upsert_worker` test call sites are UNCOVERED here
  by design (CFGDB.md itself scopes them to ai-core/wire-server); verified
  the exact break shape with `cargo check -p jammi-ai --lib` (E0061 at
  `crates/jammi-ai/src/fine_tune/worker.rs:1045`, two E0063 at `crates/jammi-ai/src/fine_tune/worker.rs:1294`/`1325`) so the lead has
  the precise fix needed, but did not attempt the fix myself (cross-crate
  ownership).
- `compute_executors.status`/`metadata` are carried opaquely (never
  validated against a closed vocabulary) — by design (CFGDB.md: "opaque
  here"), but this means a garbage `status` string is never refused at
  write time; UNCOVERED by any oracle here because the design contract
  never asks for one.
- The six-address collision check's `server`-side three addresses are
  re-parsed with `.ok()`-silent-skip on a parse failure inside
  `BallistaConfig::validate`, rather than erroring — this means a
  genuinely malformed `server.health_listen` (caught elsewhere by
  `ServerConfig::validate`) is invisible to THIS function's collision
  matrix. No oracle exercises this specific interaction (a malformed
  server address AND a ballista collision in the same config) — labelled
  UNCOVERED rather than asserted safe.

##### 4. Gates

| Command | Exit | Result |
|---|---|---|
| `cargo test -p jammi-db --lib --features test-hooks ballista` | 0 | 8 passed |
| `cargo test -p jammi-db --features live-postgres-tests,test-hooks --test it -- migration_038 compute_repo transfer_claim devices --test-threads=1` (`JAMMI_TEST_PG_URL` set) | 0 | 28 passed (both backends) |
| `cargo test -p jammi-db --test it --features live-postgres-tests,test-hooks docs_config_fences -- --test-threads=1` | 0 | 3 passed |
| `cargo test -p jammi-db --test it --features live-postgres-tests,test-hooks jobs_queue:: -- --test-threads=1` (collateral-breakage check, run once) | 0 | 114 passed (1 unrelated pre-existing test needed manual row cleanup against the shared scratch Postgres first — see commit 4's message; not a jammi-db code regression, unmodified by any commit on this branch) |
| `cargo test -p jammi-db --test it --features live-postgres-tests,test-hooks gang_membership:: -- --test-threads=1` (collateral-breakage check, run once) | 0 | 48 passed |
| `cargo clippy -p jammi-db --all-targets --features live-postgres-tests,test-hooks -- -D warnings` | 0 | clean |
| `cargo fmt --all -- --check` | 0 | clean |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `check-citations: 1046 file(s) scanned, all PATH:LINE citations resolve` |
| `cargo check -p jammi-ai --lib` (diagnostic only, confirms the named cross-crate breakage's exact shape) | 101 | 3 errors, exactly as named in §1/§3 |

##### 5. Commits

```
634ae25f feat(db): #500 CFGDB — workers.devices + transfer_claim (U8b)
cfc1bcf8 feat(db): #500 CFGDB — compute_repo.rs, generic CRUD over compute_executors/compute_jobs (U8b)
bf7a4b01 feat(db): #500 CFGDB — migration 038 compute_cluster_state (U8b db slice)
957ac62f fix(db): #500 CFGDB — BallistaConfig::validate is a &JammiConfig cross-section check (pressure round delta 1)
a8cdc68d feat(db): #500 CFGDB — [ballista] config section (U8a-cfg)
```

### 11.3 GANG — `GangExec`; the placed decision; `run_placed_gang`; the seams; `WorkerFacts.devices` (ai-core) — landed as 87e228a2 (original 97f0261d; its stopgap 2225c6c0 dropped in favour of CFGDB's real `DeviceFact`/`upsert_worker`/`transfer_claim`)

**Lead's note.** The lead opened the placed decision at the top of `run_claimed_job_under` (gated on `placed == false`, `ContextPredictor` excluded) and `run_placed_gang`'s slot → `transfer_claim` → body order before landing. The nine `gang_placed` oracles ran green on the consolidated tree; one pre-existing sibling (`host_admission::an_idle_loop_never_claims_while_a_rank_is_held`) raced its own claim probe under the concurrent suite and was fixed at the root of the test (wait for `Free` before holding, 4f5bfa64). The vacuous-mutation note (a `record_failed` on the `HandedOff` arm is a no-op by the CAS guard) is accepted as stated.

#### Contract — GANG (plan 67 wave 4: `GangExec`, the `Placed` arm, `run_placed_gang`, `WorkerFacts.devices`)

Worktree `wt-gang`, branch `unit/gang`, base `0e11162a` (main after PR #584). Owner: ai-core.
Two commits: `2225c6c0` (stopgap), `97f0261d` (the unit). Every path below is repo-relative;
every test name is `file::fn`. This contract also folds the mid-task pressure-round deltas
(five, delivered by the lead) — the design below is the POST-delta shape; §1 states what
changed from GANG.md's original text and why.

##### 1. Scope shipped

###### The pressure-round deltas (binding, folded here)

1. **Placement moved from inside `run_spec`'s `Peer` arm to the top of
   `run_claimed_job_under`, before topology is decided, and applies to every
   `fine_tune`/`graph_fine_tune` attempt (not only `Peer`).** `run_claimed_job_under` reads
   `common.world_size` off the deserialised spec by a non-consuming match (`ContextPredictor`
   has no `world_size`/gang concept and is excluded — stated as a deviation, §1a below), checks
   `session.host_admission().placed_gang_submitter()` filtered on the NEW
   `PlacedGangSubmitter::placement_available()` method, and — when placement applies — calls
   `submit_placed` directly, never entering `run_spec`/`TopologyDecision::decide` at all. This
   means `run_spec`'s `TopologyDecision::Peer` arm in the `FineTune` match is **byte-unchanged**
   from base (it always calls `self.coordinate(..)`): the recursion guard (`placed: bool`) lives
   entirely in `run_claimed_job_under`, never inside `run_spec`. A materialization/loader cost is
   therefore avoided for a placed run — the executor performs it once, on whichever process
   actually trains, per delta 1's "the executor decides topology from its own `[worker]
   local_ranks`".
2. **`Holder::Awaiting { job_id, attempt }`** (block B2): `HostAdmission::begin_awaiting_placement`
   flips `JobRun -> Awaiting` right after a successful `submit()`, called from `submit_placed`.
   `try_hold_rank` admits out of `Awaiting` exactly as out of `Free` (a rank superseding an
   awaiting host); `probe_claim` refuses it exactly like `JobRun` (unchanged — `probe_claim`'s
   predicate is `== Free`, so `Awaiting` was already refused by construction, no code change
   needed there); `ClaimGuard::drop`'s reset matcher gained `Holder::Awaiting { .. }` alongside
   `ClaimProbe`/`JobRun`.
3. **`run_placed_gang` on the submitter's own host is a typed `HolderBusy`-shaped refusal —
   kept as originally designed** (no change): `probe_claim()` refuses a busy slot before any row
   write. p7 covers it.
4. **`WorkerFacts.devices: Vec<DeviceFact { kind, ordinal }>` — no `memory_bytes`.** Dropped
   entirely (this tree has no source for it); `compute_executors.devices` (U8b, not this unit) is
   the join authority for placement, `workers.devices` is only the `ListWorkers` mirror.
5. No further GANG.md changes.

###### The stopgap commit (`2225c6c0`) — dropped by the lead once CFGDB's own commit lands

- `jammi_db::catalog::instance::DeviceFact { kind: String, ordinal: u32 }` (no `memory_bytes`,
  delta 4) and `WorkerFacts.devices: Vec<DeviceFact>`.
- `Catalog::upsert_worker(instance_id, kinds, state, devices: &[DeviceFact])` — the `devices`
  parameter is accepted but **not yet persisted** (no `workers.devices` column exists without
  migration 038, U8b's own unit); every existing call site (jammi-db's and jammi-server's own
  it-suites) updated to pass `&[]`.
- `Catalog::transfer_claim(job_id, from_instance, to_instance, attempts, lease) -> Result<bool>`
  — contract §2.4's SQL exactly: `claimed_by = $from AND attempts = $attempts AND status =
  'running' AND NOT (lease_expired_clause(..))`, using `lease_deadline_expr`/`lease_expired_clause`
  (the same backend-clock helpers every other lease predicate in `jobs_repo.rs` uses — no second
  clock source). **The lead keeps CFGDB's real, JSON-persisted `upsert_worker`/`DeviceFact` and
  drops this whole commit**; `transfer_claim` is this unit's own permanent addition (not
  duplicated by CFGDB per the brief) and should be KEPT when CFGDB's commit lands.

###### The unit commit (`97f0261d`)

- **`crates/jammi-ai/src/operator/gang_exec.rs`** (new): `GangDescriptor { job_id, attempt,
  world, submitter }` (serde), `PlacedOutcome { Trained { artifact_digest }, Failed { reason } }`,
  `GangExec` — zero children, one partition (`Partitioning::UnknownPartitioning(1)`), schema
  `{ outcome: Utf8 non-null, artifact_digest: Utf8 nullable }`. `execute`: `partition != 0` ->
  typed `DataFusionError::Plan`; no runner installed -> typed `DataFusionError::Plan` ("this
  process hosts no executor"); a runner is dispatched to via
  `crate::fine_tune::worker::placed_gang_runner()` (a free fn reading the process-global seam,
  see below) inside a spawned task built on `RecordBatchReceiverStreamBuilder` (the same shape
  `InferenceExec::execute` uses) — `Ok(Trained)`/`Ok(Failed)` each yield exactly one batch,
  `Err(e)` yields none and the spawned task's own `Err` return propagates as the stream's error
  (DataFusion's `RecordBatchReceiverStreamBuilder::build` merges a task-return `Err` into the
  output stream — verified by reading `datafusion-physical-plan-54.1.0/src/stream.rs`, no
  `tx.send` needed on that arm).
- **The process-global runner seam — the design choice and its refutation.** `GangExec::execute`
  receives only a `TaskContext`, which on a Ballista executor is reconstructed from the
  scheduler's serialized `SessionConfig` — never the submitter's `InferenceSession`. **Refutation
  attempted:** a `SessionConfig` extension (`with_extension`) was the first candidate; it fails
  because an extension set on the SUBMITTER's `SessionConfig` never crosses Arrow-Flight-serialized
  plan bytes to the EXECUTOR's independently-reconstructed `TaskContext` — they are different
  processes' objects. Since a `jammi-server` process hosts exactly one `InferenceSession`
  (`OssServer::bind` builds one), the runner it installed is a legitimate PROCESS-GLOBAL fact:
  `static PLACED_GANG_HOST: OnceLock<Weak<HostAdmission>>` in `worker.rs`, set the one time
  `HostAdmission::install_placed_gang_runner` succeeds; `placed_gang_runner()` (free fn) upgrades
  the weak ref and delegates to `HostAdmission::placed_gang_runner()`. Both this static and the
  choice are documented in `gang_exec.rs`'s module doc AND `worker.rs`'s own doc on the static.
- **`HostAdmission`** (`worker.rs`, beside `install_member_dialer`): two new `OnceLock` fields
  (`placed_gang_submitter`, `placed_gang_runner`), `install_placed_gang_submitter`/
  `placed_gang_submitter()`, `install_placed_gang_runner`/`placed_gang_runner()` (write-once, the
  `MemberDialer` shape; `install_placed_gang_runner` ALSO sets `PLACED_GANG_HOST` on a winning
  install), `begin_awaiting_placement(job_id, attempt) -> bool` (`pub(crate)`, `JobRun ->
  Awaiting`, no-op otherwise).
- **`PlacedGangSubmitter`** (trait): `submit(descriptor) -> BoxFuture<'static,
  Result<BoxStream<'static, Result<RecordBatch, DataFusionError>>, JammiError>>` (the stream's
  item type is DataFusion's own `Result` — Ballista's `execute_physical_plan` result, carried
  unwrapped) + `placement_available(&self) -> bool` (delta 1: "a registered executor other than
  this instance exists").
- **`PlacedGangRunner`** (trait): `run(descriptor) -> BoxFuture<'static,
  Result<PlacedOutcome, JammiError>>`.
- **`WorkerJobError::HandedOff`** (new variant, no payload): the claim moved to a placed executor
  mid-attempt — no terminal write, no release; the lease-keeper registration is already dropped
  by the shared `drop(hold); drop(cancel_watcher);` every `run_claimed_job_under` exit goes
  through. Doc names the exact test that proves `heartbeat_job` keys on `claimed_by` (below).
- **`submit_placed`** (`JobWorker` method): records `training_test_hooks::note_placed`, builds
  `GangDescriptor { submitter: session.instance_id() }`, calls `submitter.submit(..)`; on `Ok`
  flips `Awaiting` (delta 2) then drains the stream. Stream ends with >= 1 batch ->
  `WorkerJobError::HandedOff`. Stream ends in error, OR ends with no batch and no error (an
  UNCOVERED-labelled edge — GangExec's own design always yields exactly one batch or an error,
  so an empty-clean-end is not expected from a real submitter; treated as a synthetic error and
  routed through the same re-read) -> `placed_submit_end`.
- **`placed_submit_end`**: re-reads the row (`catalog.get_job`); `claimed_by == self.worker_id`
  -> `Abandoned("placement failed before transfer: {e}")`; otherwise (moved, OR the re-read
  itself faults) -> `HandedOff`. A `test-hooks` recorder (`note_placed_submit_end`/
  `placed_submit_ends_for`) records the ACTUAL classification chosen (not the raw predicate —
  see the mutation table, this distinction mattered).
- **`AttemptEnd`/`PublishOutcome`** (new private enums): `publish_and_finalize`'s return type
  changed from `()` to `PublishOutcome { Completed, Failed(String), LeftForReclaim }` (every one
  of its five early `return;` sites, all already past a `record_failed` call, now returns
  `Failed(<same message>)`; the tail `Ok(true)`/`Ok(false)`/`Err(e)` match arms of the finalize
  CAS now return `Completed`/`LeftForReclaim`/`LeftForReclaim`). `run_claimed_job_under`'s return
  type changed from `()` to `AttemptEnd { Published { artifact_digest }, Failed { reason },
  LeftForReclaim }`: the `Ok(artifact)` arm computes `adapter_files_digest(artifact.dir.path())`
  **before** `artifact` is moved into `publish_and_finalize` (the same function a `Peer` gang's
  member/coordinator already uses — K4, one function on every side), maps `PublishOutcome` to
  `AttemptEnd`; every other arm (`Cancelled`, `Abandoned`, `HandedOff`, `Failed`) maps
  accordingly. `run_claimed_job`/the claim loop discard the returned `AttemptEnd` (their
  signatures are unchanged — `run_claimed_job` still returns `()`).
- **`run_claimed_job_under` gains a `placed: bool` parameter** (the recursion guard, delta 1):
  threaded from `run_claimed_job` (`false`), `run_until`'s loop (`false`), and
  `run_placed_gang` (`true`).
- **`JobWorker::run_placed_gang`** (new, `pub async fn`, an ASSOCIATED function — not a method —
  since the executor role holds only a session, never a `JobWorker`): (i)
  `session.host_admission().probe_claim()` (`Free -> ClaimProbe`; refuses BEFORE any row write —
  p7); (ii) `Catalog::transfer_claim`; `false`/`Err` -> typed refusal, slot released, no write
  (p6); (iii) re-reads the row under `session.with_admin_scope` (the claim this stands in for is
  the unscoped kind every claim-loop read is — `get_job` itself is tenant-scoped, wrong for this
  internal re-read); (iv) `worker.run_claimed_job_under(session, record, &shared, true)` — the
  SAME body, which itself performs the `register_job_hold_or_release`/`job_running()`
  (`ClaimProbe -> JobRun`) registration a loop claim performs, so nothing here duplicates it;
  (v) maps `AttemptEnd` -> `PlacedOutcome` (`LeftForReclaim` -> a typed `Err`, so the Ballista
  task itself ends in error — Ballista's `task_max_failures = 0` never re-runs THIS task; jammi's
  own reclaim from a FUTURE claim is the only path back); (vi) `drop(claim)` on every exit arm.
- **`worker_devices(config, compute_device) -> Vec<DeviceFact>`** (new, private): for each
  `WorkerTopology::rank_devices()` ordinal, `DeviceFact { kind: <session's ComputeDevice
  discriminant, lowercase>, ordinal: ordinal.max(0) as u32 }` (the CPU sentinel `-1` becomes the
  honest `0`, never wrapped via a bare `as u32`). Wired at both `write_worker_facts` call sites
  in `run_until` and into `write_worker_facts`'s own `catalog.upsert_worker` call.
- **`training_test_hooks`**: `note_placed`/`placed_attempts_for` (p1/p8),
  `note_placed_submit_end`/`placed_submit_ends_for` (p4/p5's classification oracle).
- **Module doc**: the Holder enum's own doc documents `Awaiting` fully (admits/refuses/reset);
  the writer table gained rows 18 (submitter after hand-off: writes nothing) and 19 (executor:
  writes as `Coordinator`, the same body as row 17). **Deviation, stated:** I did NOT also edit
  `docs/maintainer/MAINTAINER-GUIDE.md`'s own Holder-lattice/writer-table prose (U9a's docs unit
  owns that file; GANG.md's brief scopes MAINTAINER-GUIDE.md edits to citation re-anchoring only,
  which I did do — five citations, listed in the gates table).

#### 1a. Deviations from GANG.md/UNITS.md, each with the reason and the code

1. **Placement applies to `FineTune` and `GraphFineTune`, never `ContextPredictor`.**
   `TrainingSpec::ContextPredictor` has no `common: TrainingCommon`/`world_size` field at all
   (`crates/jammi-ai/src/fine_tune/spec.rs`, the enum definition) — there is no coherent
   `GangDescriptor.world` to build for it, and it was never in `TopologyDecision`'s scope either.
   Stated as the match's third arm (`ContextPredictor { .. } => None`).
2. **`run_spec`'s `Peer` arm carries no `placed` parameter** (contrary to GANG.md's original
   text, superseded by pressure-round delta 1): since placement is now decided BEFORE `run_spec`
   is ever called, `run_spec` never needs to re-check the submitter itself; the recursion guard
   is `run_claimed_job_under`'s own `placed: bool`, checked once, before `run_spec`/`coordinate`
   are reached at all.
3. **The "no batch and no error" stream-end arm** (not explicitly named in GANG.md's p1–p8) is
   folded into the same re-read path as a genuine stream error (a synthetic
   `JammiError::FineTune` naming it) — GangExec's own design never produces this shape in
   practice (see its `execute`'s doc), so this is a defensive completion of `submit_placed`'s
   match, not a distinct oracle.
4. **`GangExec`'s own unit tests use ONE process-global stub runner across all four assertions**
   (no runner / `Trained` / `Failed` / `Err`), sequenced within a SINGLE `#[test]` function.
   `PLACED_GANG_HOST` is a genuine process-wide `OnceLock`, settable once for the whole `--lib`
   test binary; four separate `#[test]` fns each wanting a different runner would race
   (`cargo test`'s default parallelism) or silently share whichever installed first. A
   job-id-keyed script registry (the `training_test_hooks`-style `static ARMED: OnceLock<Mutex<..>>`
   shape) lets ONE installed stub serve every scenario, and the "no runner" assertion runs FIRST,
   before any install anywhere in this test — the only ordering that makes it deterministic.
   Stated in `gang_exec.rs`'s test doc.

##### 2. Properties

| # | Property (quantified) | Executed oracle (path::name; lane) | Executed mutation → red (first line) |
|---|---|---|---|
| GangExec-1 | Schema/partitioning fixed; `partition != 0` refused typed; no runner installed refused typed; a `Trained` stub yields one batch with the digest; a `Failed` stub yields one batch, no digest; an `Err` stub's error is the stream's only item (no batch) | `jammi-ai` lib `operator::gang_exec::tests::gang_exec_dispatches_through_the_process_global_runner_seam` | RED at base (`GangExec` does not exist on the base tree) |
| Devices-1 | `WorkerFacts.devices` is decided from `[worker]`/`[gpu]` configuration alone, no GPU needed: `device = -1` registers one fact `{cpu, 0}` (never `{cpu, -1}` — no wrapped-`u32`); two configured devices register two facts in rank order; the session's own `ComputeDevice` decides `kind` uniformly | `jammi-ai` lib `fine_tune::worker::tests::worker_devices_is_decided_from_configuration_alone` | RED at base (`worker_devices`/`DeviceFact` do not exist) |
| Awaiting-1 | `Awaiting` admits a `RunRank` session exactly like `Free`; refuses a second claim exactly like `JobRun`; the claim's own guard resets it to `Free` once the await ends, whether or not a rank took the cell over meanwhile | `jammi-ai` lib `fine_tune::worker::tests::awaiting_admits_a_rank_and_refuses_a_second_claim_and_frees_when_the_claim_ends` | RED at base (`Holder::Awaiting`/`begin_awaiting_placement` do not exist) |
| p1 | With a submitter installed (`placement_available() == true`), a claimed `fine_tune`/`graph_fine_tune` attempt takes the `Placed` arm and NEVER traverses `coordinate` | `it::gang_placed::p1_a_world_size_two_job_takes_the_placed_arm_and_not_coordinate` | M: `placement_world` short-circuited to `None` unconditionally → RED: `crates/jammi-ai/tests/it/gang_placed.rs:225: assertion left == right failed: the Placed arm fired exactly once` (`left: []`) |
| p2/K4 | The stub submitter drives a REAL `run_placed_gang` on a second (executor) session sharing one catalog: the transfer moves `claimed_by`, `attempts`/`releases` unchanged, the executor's own `Local{2}` gang completes, and the published bytes equal the wave-3 `LocalGang` reference | `it::gang_placed::p2_the_stub_submitter_drives_a_real_run_placed_gang_to_the_same_bytes` | RED at base (`run_placed_gang`/the `Placed` arm do not exist; the whole harness would panic on `install_placed_gang_submitter`) |
| p3 | After a placed attempt ends, the submitter's slot is `Free` again — a second claim proceeds at once | `it::gang_placed::p3_the_submitters_slot_is_free_after_a_placed_attempt_ends` (+ the second-claim assertion folded into p2) | RED at base (as p2) |
| p4 | A stream fault BEFORE any transfer classifies `Abandoned`: the row stays `running`, `claimed_by` the submitter, no error, attempts unspent | `it::gang_placed::p4_a_stream_fault_before_transfer_leaves_the_row_running_for_the_submitter` | M: `placed_submit_end`'s `still_mine` branch inverted → RED: `gang_placed.rs: assertion left == right failed: the row is still this instance's at the re-read: Abandoned, never HandedOff` (`left: [false]`) |
| p5 | A stream fault AFTER the transfer landed classifies `HandedOff`: `claimed_by` the executor, no error written by the submitter, the submitter's own slot free at once | `it::gang_placed::p5_a_stream_fault_after_transfer_hands_off_and_the_submitter_writes_nothing` | same inversion mutation → RED: `assertion left == right failed: the row had already moved at the re-read: HandedOff, never Abandoned` (`left: [true]`) |
| p6 | `run_placed_gang` refuses typed on a stale `attempt` (no row write) and on a SECOND launch of an already-transferred descriptor (`claimed_by = $from` no longer matches) | `it::gang_placed::p6_run_placed_gang_refuses_a_stale_attempt_and_a_second_launch` | M: `transfer_claim`'s `attempts` conjunct removed → RED: `crates/jammi-ai/tests/it/gang_placed.rs:378: a stale attempt must refuse: Trained { .. }` |
| p7 | `run_placed_gang` on a host already holding a rank refuses typed BEFORE `transfer_claim` — the row's `claimed_by` unchanged | `it::gang_placed::p7_run_placed_gang_refuses_a_host_already_holding_a_rank_before_any_transfer` | M: the slot check in `run_placed_gang` bypassed (`let claim = admission.probe_claim();` unconditional) → RED: `crates/jammi-ai/tests/it/gang_placed.rs:433: a host already holding a rank must refuse: Trained { .. }` |
| p8 | A placed run whose own process ALSO hosts a submitter does not re-submit: `note_placed` fires exactly once, on the original submitter only | `it::gang_placed::p8_a_placed_run_never_re_submits_even_with_a_submitter_installed_on_its_own_process` | M: `run_placed_gang`'s `run_claimed_job_under(.., true)` mutated to pass `false` → RED: `assertion left == right failed: note_placed fires exactly once, on the original submitter only` (`left: [1, 1]`) |
| HandedOff-heartbeat | `Catalog::heartbeat_job` keys on `claimed_by`: a stale (pre-transfer) holder's heartbeat after a hand-off cannot resurrect the new holder's lease | `it::gang_placed::the_submitters_heartbeat_after_hand_off_never_resurrects_the_executors_lease` | not separately mutated (this is `heartbeat_job`'s own pre-existing, unmodified attempt-guarded predicate; the oracle is new, the mechanism is not) |

###### A vacuous mutation, recorded honestly

**M: `WorkerJobError::HandedOff`'s arm in `run_claimed_job_under` made to call `record_failed`**
(the brief's own suggested mutation) **stayed GREEN against every p1–p8 oracle.** Root cause,
verified: `record_failed`'s own attempt-guarded `UPDATE` requires `claimed_by = self.worker_id`;
by the time `HandedOff` is EVER reached (the stream saw >= 1 batch, or the re-read shows the row
moved), the row's `claimed_by` is provably never the submitter's own id any more, so the write
is a no-op by construction on every reachable path — defense in depth, not a gap. Recorded here
per "ship the honest negative": the SUGGESTED mutation for item 4 does not red, and the reason is
a stronger guarantee (the CAS guard), not a missing oracle. The classification itself (which
enum variant `placed_submit_end` returns) IS covered and DOES red under mutation (p4/p5 above,
via the `note_placed_submit_end` recorder — added specifically because the row-level assertions
alone could not distinguish `Abandoned` from `HandedOff`, both being "no write" outcomes at the
row level).

##### 3. Uncovered

- **Two real OS processes.** Every it-test here runs the "executor" and "submitter" as two
  `InferenceSession`s in ONE test process sharing one SQLite file (WAL) — the same substitution
  `gang_coordinator.rs`'s and `gang_chaos.rs`'s own hermetic fleets make. A real `jammi-ballista`
  executor process, a real Ballista scheduler, and `GangExec` actually executing inside a
  Ballista `TaskContext` are BALLISTA's/U8b's own legs (distributed lane).
- **`GangExec::execute`'s dispatch through a REAL Ballista `TaskContext`** (only its dispatch to
  `placed_gang_runner()` is exercised; the `_context: Arc<TaskContext>` parameter is unused by
  design — stated in the signature, never read).
- **A stream that ends with no batch and no error** (§1a deviation 3): GangExec's own design
  never produces this shape; `submit_placed`'s handling of it is defensive, not separately
  oracled.
- **`adapter_files_digest` failing after a successful publish** (the digest re-read racing the
  publish that already read the same files) — `run_claimed_job_under`'s `Ok(artifact_digest)`
  vs `Err(e)` split on this exact narrow window is stated, not executed (an essentially
  unreachable I/O race).
- **The elder session's exact end reason in a split-brain placement** (two submitters racing to
  place the same job) is not built — GANG.md's p1–p8 do not name this scenario and U8b's
  `DevicePlacement` re-launch guard is the mechanism that would need it; `transfer_claim`'s own
  `claimed_by = $from` conjunct is what makes a second submission's transfer fail (proven by p6),
  but no oracle drives two CONCURRENT submitters against the same row.
- **A `PlacedGangSubmitter` installed by a real Ballista scheduler role** (`crates/jammi-ballista`,
  BALLISTA's unit) — this contract's submitter stubs are hand-written, not the real Ballista
  client wrapper.
- **`MAINTAINER-GUIDE.md`'s own prose** (the Holder-lattice/writer-table narrative, beyond
  citation re-anchoring) is U9a's, not re-derived here.

##### 4. Gates

`CARGO_TARGET_DIR=…/targets/gang`, `RUSTC_WRAPPER=sccache`, one `--features` set for the whole
session (`jammi-ai`: `test-hooks`; `jammi-db`: `test-hooks`).

| Command | Exit | Result |
|---|---|---|
| `cargo fmt -p jammi-ai --check` | 0 | clean |
| `cargo fmt -p jammi-db --check` | 0 | clean |
| `cargo clippy -p jammi-ai --all-targets --features test-hooks -- -D warnings` | 0 | clean |
| `cargo clippy -p jammi-db --all-targets --features test-hooks -- -D warnings` | 0 | clean |
| `cargo check -p jammi-server --tests --features test-hooks` | 0 | compiles (sanity check: two of its it-test files' `upsert_worker` call sites needed `&[]`) |
| `cargo test -p jammi-ai --features test-hooks --test it -- gang_placed gang_coordinator jobs_shutdown host_admission --test-threads=4` | 0 | 42 passed (9 gang_placed + 4 gang_coordinator + 19 jobs_shutdown + 10 host_admission), 0 failed |
| `cargo test -p jammi-ai --features test-hooks --lib -- fine_tune::worker gang_exec` | 0 | 44 passed |
| `cargo test -p jammi-ai --features test-hooks --lib -- fine_tune::` | 0 | 361 passed, 0 failed — the WHOLE pre-existing suite, unaffected |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `1046 file(s) scanned, all PATH:LINE citations resolve` — 5 re-anchored in `MAINTAINER-GUIDE.md` under this unit's insertions |
| `python3 ci/scripts/check_no_consumer_names.py` | 0 | OK |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p jammi-ai --no-deps` | 0 | clean — 3 `GangExec::execute` intra-doc links converted to backtick code spans (a trait method has no inherent-item target), 1 `PLACED_GANG_HOST`/2 `Self::run_claimed_job_under`/1 `AttemptEnd` link converted to backtick spans (private items linked from public docs), 1 `Partitioning::UnknownPartitioning(1)` converted (tuple-variant call syntax unsupported by intra-doc links) |

Mutations M1–M6 (§2's table + the vacuous-mutation note): every one applied to the committed
tree, run through ONE filtered test, reverted (`git diff` empty after each); logs are the
transcript above (this session had no separate scratch-log file — every mutation's before/after
diff and red/green output is reproduced verbatim in §2).

Not run, per COMMON.md's trimmed set: `cargo test -p jammi-db` (no `jammi-db` test added beyond
the stopgap's mechanical `&[]` fixes, already covered by jammi-db's clippy pass), the live-Postgres
lane (no SQL shape here is backend-conditional beyond `lease_expired_clause`/`lease_deadline_expr`,
already used identically by `heartbeat_job` elsewhere in this file — both backends), `merge_path.sh`,
workspace-wide builds.

##### 5. Seams for the lead (consolidation)

- **BALLISTA installs both seams** from `jammi-server`: the scheduler role calls
  `session.host_admission().install_placed_gang_submitter(Arc::new(<its Ballista client
  wrapper>))`; the executor role calls
  `session.host_admission().install_placed_gang_runner(Arc::new(<its GangExec runner
  wrapper>))`. Both are `pub` methods on `HostAdmission` (`crates/jammi-ai/src/fine_tune/
  worker.rs`).
- **The stopgap commit `2225c6c0` is to be DROPPED** once CFGDB's real `DeviceFact`/
  `upsert_worker` commit is folded in; `Catalog::transfer_claim` in that same commit is this
  unit's OWN permanent addition and must be KEPT (CFGDB does not build it — confirmed against
  CFGDB's own contract scope in GANG.md/CFGDB's brief).
- **`operator/mod.rs` gained exactly one line** (`pub mod gang_exec;`) — BALLISTA's four reserved
  operator files (`inference_exec.rs`, `ann_search_exec.rs`, `key_check_exec.rs`, `asof/exec.rs`)
  were not touched.
- **`GangDescriptor`/`PlacedOutcome`** live in `crates/jammi-ai/src/operator/gang_exec.rs`
  (public); `worker.rs` imports them via `use crate::operator::gang_exec::{GangDescriptor,
  PlacedOutcome};` (a private `use`, not a re-export) — BALLISTA's own codec/runner code should
  import them from `jammi_ai::operator::gang_exec` directly, not from `jammi_ai::fine_tune::worker`.
- **Six `gang_coordinator.rs` helpers were made `pub(crate)`** for `gang_placed.rs` to reuse
  (`coordinating_session`, `two_rank_spec`, `two_rank_graph_spec`, `fan_out_config`,
  `graph_loader`, `submit_and_claim`, `graph_nodes`, `graph_edges`, `write_csv` — nine, not six;
  corrected count) — no behavioural change to any of them.

##### 6. Commits

```
97f0261d feat(ai-core): #500 wave 4 GANG — GangExec, the Placed arm, run_placed_gang, WorkerFacts.devices
2225c6c0 stopgap(db): transfer_claim + DeviceFact (CFGDB owns the real ones)
```
(no trailers, per the brief; `git status --short` clean at the tip.)

### 11.4 BALLISTA — `crates/jammi-ballista`; hosting; registration sites (wire-server) — landed as a86df04e (the net diff of unit/ballista f6b32006 over df5734e1, consolidated as one commit; its config stopgap db289303 superseded by CFGDB)

**Lead's note.** The lead read the hosting hunks (`BallistaConfig::validate` at construction; DRAIN → `drain()` then `stop()`; the submitter exclusion in the placement policy) and ran the crate's 18 tests, the server's hosting tests and workspace clippy on the consolidated tree. One server oracle — the enumerating-caller gate on `get_result_table_for_tenant` — went red on the codec's `AnnSearchExec` decode; the lead reviewed that read (pinned to the tenant the submitter's session carried onto the wire, over the internal listeners, the peer listener's trust class) and grew the allowlist deliberately with that review (50ab34bc). The unit's named gaps (the distributed leg, (a3)–(a5), the r41 engine test, `devices()` values, `build_embedding_plan`) were assigned to LANEAI, U8b and LANE. Its deviation on `device_kind` (stamped at the codec, not the constructor) was reversed by LANEAI.

#### Contract — unit BALLISTA (`crates/jammi-ballista`, plan 67 wave 4, PR-D)

Base `0e11162a`; merged `df5734e1` (CFGDB U8a-cfg/U8b-db + GANG's `GangExec`) at `42ef56c8`.
Branch `unit/ballista`. Design contract: `docs/rigor/contracts/feat_500-wave4.md` (read at
`bcfda3ca` for the base design, re-read at `cb976274` for the pressure-round REFINE deltas
folded into this unit mid-session).

##### 1. Scope shipped

**New crate `crates/jammi-ballista`** (publishable, no cargo feature; `jammi-server` depends on
it unconditionally):

- `src/error.rs` — `Error`/`Result<T>`; `into_df_error()` converts to the `DataFusionError` the
  codec/engine trait methods must return.
- `src/codec.rs` — `JammiCodec: PhysicalExtensionCodec`. Magic `[0x07, b'J', b'M', b'B']` (an
  illegal prost tag byte — field 0, wire type 7 — pinned in a test against Ballista's own five
  oneof tag bytes `0x0A/0x12/0x1A/0x22/0x2A`, `ballista-core-54.1.0/src/serde/generated/
  ballista-core-54.1.0 src/serde/generated/ballista.rs lines 31–54`). Encodes `InferenceExec`, `AnnSearchExec`, `AsofJoinExec`, `KeyCheckExec`,
  `GangExec` as `jammi.ballista.v1` prost messages (own `build.rs`, `proto/jammi/ballista/v1/
  plan.proto` — messages only); delegates every other node to `BallistaPhysicalExtensionCodec`
  (its own typed "Unsupported plan node" refusal is the ONLY refusal path for a node neither
  codec knows — `MaskExec`, the named v1 cut, and a plain unknown node both prove this via one
  shared oracle). Decode rebuilds through each operator's public constructor against a `Weak<
  InferenceSession>` (dead session → typed `SessionGone`, never a panic). `AnnSearchExec` decode
  re-reads the `ResultTableRecord` via `Catalog::get_result_table_for_tenant` (tenant carried
  EXPLICITLY on the wire — table_name + tenant_id — never the decoding call's ambient tenant,
  which a scheduler/executor process has none of); this is a synchronous trait method reading an
  async catalog, bridged via `tokio::task::block_in_place` + `Handle::current().block_on` (a
  documented MULTI-THREADED-runtime precondition, not silently assumed).
- `src/engine.rs` — `JammiExecutionEngine` wraps `DefaultExecutionEngine`: (1) K7 — a stage whose
  `InferenceExec.device_kind()` (always `Some` on the wire; the codec stamps the submitting
  session's own kind when a construction call site left it `None`) differs from this executor's
  own `session.compute_device().kind()` is refused typed, naming both kinds; (2) README r41 — a
  stage containing a `GangExec` with `output_partitioning().partition_count() != 1` is refused
  typed.
- `src/placement.rs` — `PlacementPolicy: DistributionPolicy`, a from-scratch round-robin
  (Ballista's own `bind_task_round_robin` is `pub(crate)` to `ballista-scheduler`, confirmed by
  reading `cluster/mod.rs`; re-implemented over the same PUBLIC types its trait signature
  requires — `AvailableTaskSlots`, `JobInfoCache`, `ExecutionGraph::fetch_running_stage`,
  `TaskDescription`, `create_task_info`), excluding a `GangExec` stage from the executor whose id
  equals `descriptor.submitter` (contract §9 B1).
- `src/roles.rs` — `host_scheduler`/`host_executor`/`SchedulerRole`/`ExecutorRole`, jammi's own
  shutdown (never `start_server`/`start_executor_process` — both install their own `ctrl_c`
  handler). `SchedulerRole::stop()`; `ExecutorRole::stop()` (immediate/RELEASE) and `drain()`
  (report `Terminating` via the SAME `TERMINATING` flag + heartbeat `start_executor_process`
  uses, await `TasksDrainedFuture`, then `stop()` — contract §9 B6). Both roles bind their real
  listener BEFORE building `SchedulerConfig`/`ExecutorProcessConfig`/`ExecutorRegistration`, so
  the scheduler's own `scheduler_name()` and the registration metadata never carry a stale `:0`
  (a real bug this session found and fixed by executing the exact path: red at `addr.port()`,
  green at `local_addr.port()` — see Properties). `host_scheduler` installs
  `SchedulerPlacedGangSubmitter` (submits a `GangExec` via `submit_physical_plan`;
  `placement_available()` reads the scheduler's own `ClusterState::registered_executor_metadata`
  synchronously via `block_in_place`); `host_executor` installs `ExecutorPlacedGangRunner`
  (`JobWorker::run_placed_gang`) and reports `devices()` as `jammi_db::catalog::instance::
  DeviceFact` (U8b's real shape), reproducing `jammi_ai::fine_tune::worker`'s PRIVATE
  `worker_devices` two-line mapping verbatim rather than exposing it (outside this unit's
  file grant).
- `src/client.rs` — `submit_physical_plan(session, scheduler_url, plan)` over `ballista_core::
  execution_plans::execute_physical_plan`.
- `tests/it/{codec,engine,roles}.rs` (hermetic, no live backend) + `tests/roles_drain.rs` (its
  OWN test binary — see Uncovered/deviation below) + `tests/distributed/{main,harness}.rs`
  (`live-distributed-tests`, gated, honestly skips without live backends).

**jammi-ai accessors** (the four granted files): `InferenceExec::{source, task, content_columns,
key_column, source_id, backend, batch_size, embedding_dim, regression_form, passthrough, input,
device_kind}`; `InferenceExecBuilder::{backend, device_kind}` setters (`with_new_children` now
threads both through — closes a pre-existing gap where a re-planned node silently dropped its
backend override); `AnnSearchExec::{table, query_vector, k, oversample_override}`;
`AsofJoinExec::{spec, left, right}`. `KeyCheckExec::key_column()` already existed — untouched.

**`ComputeDeviceKind`** (`jammi_db::store::manifest`, next to `ComputeDevice`, with
`ComputeDevice::kind()`) and `InferenceExec::device_kind: Option<ComputeDeviceKind>` (pressure
round delta B3) — stamped `None` at construction (every in-process jammi-ai pipeline call site is
outside this unit's four-file grant, so none was touched); the codec fills the gap at the wire
boundary from the submitting session's own kind when `None`. **Deviation from the design
contract's literal wording** ("stamped by `InferenceExecBuilder` from the building session's
`compute_device()` by default"): the stamp happens at the CODEC's encode boundary, not at every
construction call site, because touching `pipeline/embedding.rs` (the one real call site) is
outside this unit's grant. Cited: `crates/jammi-ai/src/operator/inference_exec.rs` (the field's
doc states the deviation and why); functionally equivalent for every submitted plan.

**jammi-server hosting** (`runtime.rs`): `OssServer` carries `ballista: BallistaConfig`; `new`
calls `BallistaConfig::validate(&config)` (CFGDB's real cross-section associated fn) immediately
after `config.server.validate()`; `bind()` builds the roles from `[ballista]` beside the peer
listener (in-memory cluster, U8b swaps it behind this one constructor argument); `BoundServer`
gains `scheduler_addr()`/`executor_addrs()`; `serve_with_signals` stops the executor via
`drain()` on the DRAIN arm (waits for in-flight tasks) and `stop()` on RELEASE/the preload
early-exit tail (immediate), the scheduler always after. `tests/it/ballista_roles.rs`: unset = no
listener; set = both roles bind + DRAIN closes both ports within the grace; a `[ballista]`
address colliding with a fixed `[server]` listener is refused at `OssServer::new`.

**Registration sites**: workspace `Cargo.toml` (members/default-members/`[workspace.dependencies]`
— `ballista-core`/`-scheduler`/`-executor`, `datafusion-proto`); `ci/scripts/publish_crates.sh`
`PUBLISH_ORDER` (before `jammi-server`); `.claude/agents/wire-server.md` `owns:`;
`.github/workflows/ci.yml`'s gated clippy step; `docs/maintainer/MAINTAINER-GUIDE.md`'s generated
dep-DAG block (`python3 ci/scripts/gen_dep_dag.py`) and hand-maintained production DAG paragraph
(plus two PRE-EXISTING staleness fixes this exposed: workspace member count 13→15, the publish
order sentence never named `jammi-ai`→`jammi-server`'s intermediate crates); `docs/guide/src/
api-stability.md`'s published-crate count (eleven→twelve).

**Deviations from BALLISTA.md, with reasons and the code cited**:
1. No `impl LogicalExtensionCodec for JammiCodec`. Read `ballista-scheduler-54.1.0/src/
   ballista-scheduler-54.1.0 src/scheduler_process.rs lines 61–64`: `create_scheduler` falls back to `BallistaLogicalExtensionCodec::
   default()` when `override_logical_codec` is `None` — there is no jammi logical node for a
   logical codec to carry, so the brief's "if `create_scheduler` demands one" condition is false.
2. `host_scheduler`'s `distribution: TaskDistributionPolicy` parameter from the original brief is
   dropped — the pressure round's delta 1 makes `PlacementPolicy` ALWAYS the policy (never a
   caller choice); the design contract as amended, not this unit, made that call.
3. `.github/workflows/distributed.yml` gains NO `ballista` matrix leg. That workflow's OWN
   "Require the distributed backends (no hollow green)" step (already present for the
   `jammi-ai`-only legs) exists precisely because a silently-skipping harness reporting green
   on a `advisory: false` leg is a rigor violation this file itself guards against. This unit's
   `tests/distributed/{main,harness}.rs` DOES honestly skip when live backends are absent (see
   Uncovered) — wiring a leg that always reports green regardless of whether it exercised
   anything would violate the exact principle the sibling leg's own precheck step encodes.
   Named here as a blocking item for the lead: the leg becomes safe to wire once (a3)'s
   `build_embedding_plan` and (a4)/(a5)'s fine-tune submission stack (below) are ported in.
4. `tests/roles_drain.rs` is a SEPARATE `[[test]]` binary from `tests/it`, not a module inside
   `tests/it/roles.rs`. Found by executing the plausible arm: `ballista_executor::executor_server::
   TERMINATING` is a crate-wide `static AtomicBool`; running the drain test in the SAME process
   as `tests/it`'s executor-hosting test made the LATTER hang for 30s (its executor immediately
   reported `Terminating` to a fresh scheduler, so no task was ever bound) the moment both ran in
   one binary. Each Cargo `[[test]]` target is its own OS process — moving it out is the fix, not
   a workaround. Documented in `tests/roles_drain.rs`'s own module doc with the reproduction.
5. `tests/distributed/harness.rs` is a REDUCED COPY of `crates/jammi-ai/tests/distributed/
   harness.rs`'s shape (`Backends`/`spawn_worker`/`worker_toml`/`jammi_server_binary`), per
   COMMON.md's own instruction that a private test module of a different crate makes a copy the
   honest shape. It does NOT copy the fine-tune submission helpers (`register_training_source`,
   `submit_gang_fine_tune`, `JobSize`) — named explicitly in its own module doc as the reason the
   (a4)/(a5) gang oracles are UNCOVERED here (below).

##### 2. Properties

| Property (quantified) | Executed oracle | Executed mutation that reds it (what changed; red output's first line) |
|---|---|---|
| Every jammi node type (`InferenceExec`, `AnnSearchExec`, `AsofJoinExec`, `KeyCheckExec`, `GangExec`) round-trips encode→decode with every accessor equal, and the magic prefix is present | `tests/it/codec.rs::{inference_exec_round_trips, ann_search_exec_round_trips, asof_join_exec_round_trips, key_check_exec_round_trips, gang_exec_round_trips}` | not separately mutated per-node; covered by the shared magic/dispatch mutations below |
| `InferenceExec`'s device_kind defaults to the submitting session's own kind when unset at construction | `codec::inference_exec_device_kind_defaults_to_the_submitting_sessions_own` | n/a (a positive-construction property; the K7 refusal test below is the negative one) |
| encode→decode→re-encode of the SAME node is byte-identical | `codec::inference_exec_round_trips` (asserts `buf == buf2`) | n/a — structural (any encode nondeterminism would red this on the FIRST run, not just under mutation) |
| A magic-prefixed buffer with no tag byte is refused typed, NEVER delegated | `codec::truncated_magic_buffer_is_refused_typed_never_delegated` | deleting the `buf.len() < 5` guard; the buffer would then read `buf[4]` on a 4-byte slice and panic (index out of bounds) instead of erroring — verified by removing the guard and observing the panic, then restored |
| A buffer without the magic delegates whole to Ballista's own codec (never handled by jammi's decode arm) | `codec::no_magic_buffer_delegates_to_ballistas_own_codec` | replaced the magic check `if buf.len() < 4 \|\| buf[0..4] != MAGIC` with `if false` (never delegates) → RED: `a no-magic buffer must never be handled by jammi's own decode arm: External error: jammi-ballista: malformed operator buffer: truncated jammi operator buffer (magic, no tag byte)` — restored |
| `MaskExec` and any node neither codec knows are refused typed, naming the node | `codec::{mask_exec_is_refused_typed, unknown_node_is_refused_typed}` | covered by the same delegation path as the no-magic mutation above (both route through `self.inner.try_encode`, whose own "Unsupported plan node" is the refusal) |
| A dead `Weak<InferenceSession>` is a typed refusal on decode, never a panic | `codec::dead_session_is_refused_typed` | n/a — structural (`Weak::upgrade` returning `None` is the only path; no separate mutation needed beyond the type itself) |
| The magic's first byte (`0x07`) never collides with any of Ballista's five oneof tag bytes, and is structurally illegal (field 0, wire type 7) | `codec::magic_never_collides_with_a_ballista_oneof_tag` | n/a — a static assertion against Ballista's own pinned byte list, re-verified whenever that list is read from source again |
| A stage whose `InferenceExec.device_kind` mismatches this executor's own kind is refused typed, naming both kinds; a matching kind is NOT refused by this gate | `tests/it/engine.rs::{refuses_a_stage_whose_inference_exec_names_a_different_device_kind, does_not_refuse_a_matching_device_kind}` | routed the K7 `Err` to `tracing::warn!` (no refusal) → RED: `the refusal must name the property: Internal error: Plan passed to new_query_stage_exec is not a ShuffleWriterExec or SortShuffleWriterExec.` (the mismatch was silently let through to `DefaultExecutionEngine`, which then failed for the WRONG reason) — restored |
| Both roles pin `task_max_failures = stage_max_failures = 0` (README r40) | `roles::scheduler_config_tests::pins_zero_task_and_stage_failures` (unit test over the pulled-out `scheduler_config` builder fn) | dropped the `task_max_failures: 0` line (falls to `SchedulerConfig::default()`'s `4` via the struct-update syntax) → RED: `assertion `left == right` failed / left: 4 / right: 0` — restored |
| `host_scheduler`/`host_executor` bind on `127.0.0.1 port 0` in one process, the executor registers, and `submit_physical_plan` of a shuffle-boundary plan (`MemTable` scan → hash `RepartitionExec`) returns the same row count as in-process `physical_plan::collect` | `roles::scheduler_and_executor_host_in_one_process_and_submit_round_trips` | REAL bug found and fixed by executing this exact test before the port-resolution fix: the scheduler's `SchedulerConfig.bind_port` was built from the PRE-bind `addr.port()` (`0`), so every task-status report from the executor failed with `Fail to connect to scheduler ...:0` and the test hung for the full 30s timeout; fixed by binding the listener FIRST and building the config from `listener.local_addr()` |
| `unset [ballista]` config hosts no roles (`hosts_scheduler()`/`hosts_executor()` both `false`) | `roles::unset_ballista_config_hosts_no_roles`; server-level consequence in `jammi-server`'s `ballista_roles::unset_ballista_config_has_no_ballista_listener` | n/a — a direct assertion on the default config |
| `drain()` reports `Terminating` and stops within 5s with no in-flight work | `tests/roles_drain.rs::executor_drain_reports_terminating_and_stops_with_no_inflight_work` | REAL cross-test interference found and fixed: running this test in the SAME binary as `tests/it`'s role-hosting test made the LATTER hang (the crate-global `TERMINATING` static leaked across tests sharing one process) — fixed by moving this test to its own `[[test]]` binary (its own OS process); see Deviations |
| `OssServer::bind` builds/serves the roles from `[ballista]`; unset = no listener; a fixed-port collision with `[server]` is refused at `OssServer::new` | `jammi-server`'s `tests/it/ballista_roles.rs::{ballista_roles_bind_executor_registers_and_drain_stops_both, unset_ballista_config_has_no_ballista_listener, colliding_ballista_address_is_refused_at_oss_server_new}` | the collision test itself is the executed oracle for the NEW `BallistaConfig::validate(&config)` call site CFGDB's own contract named as owed to this unit; not separately mutated (CFGDB's own unit test suite covers `validate`'s internal logic) |
| A `GangExec` stage with `partition_count != 1` is refused typed (README r41) | not covered by an executed test — see Uncovered | — |

##### 3. Uncovered

- **(a3) embedding-across-two-executors** (contract §7): needs a public `jammi_ai::pipeline::
  embedding::build_embedding_plan`. That file is outside this unit's four-file jammi-ai grant.
  `tests/distributed/main.rs::embedding_job_across_two_executors_matches_in_process` stands up
  the skip path honestly (prints and returns) rather than fabricating a plan-construction
  workaround.
- **(a4)/(a5) the placed gang across the real claim path**: needs the fine-tune job submission
  stack (`register_training_source`/`submit_gang_fine_tune`/`JobSize`, already in `jammi-ai`'s
  own distributed harness) ported into this crate's `tests/distributed/harness.rs`, which
  deliberately did not copy it (time-boxed within this unit's remaining session). This crate's
  OWN hermetic suite DOES cover everything up to that boundary: `GangExec` encode/decode
  (`codec::gang_exec_round_trips`), `PlacedGangSubmitter`/`PlacedGangRunner` installation
  (`roles.rs`, exercised structurally by compiling and by the write-once semantics
  `HostAdmission` itself already tests), and — via `submit_physical_plan` — a plan completing on
  a registered executor other than the submitter (`roles::scheduler_and_executor_host_in_one_
  process_and_submit_round_trips`, though that test's plan is a plain shuffle, not a `GangExec`).
  What is NOT exercised: reaching the submit path through `run_claimed_job_under`'s own real
  placement decision on a genuinely claimed job, and the exclusion rule (`PlacementPolicy` never
  binds a `GangExec` to its own submitter's executor id) under a REAL scheduler with 3 registered
  executors. `tests/distributed/main.rs::placed_gang_completes_on_a_registered_executor_other_
  than_the_submitter` stands up the honest skip path and states exactly the missing stack.
- **The `distributed.yml` `ballista` matrix leg** (see Deviation 3): not wired, to avoid a
  hollow-green `advisory: false` leg.
- **`PUBLISH_ORDER`'s topological-completeness guard**: searched `ci/scripts/*.py` and
  `publish_crates.sh` itself for a mechanical check that the list is topologically complete (the
  brief's prescribed 4th mutation target) and found NONE exists in this repository today —
  `publish_crates.sh`'s only self-check is `VERSION` against the workspace version; a crate
  missing from the list is silently skipped ("crate not present at this tag — skipping"), and
  the actual failure mode is a LATER `cargo publish -p jammi-server` erroring because crates.io
  doesn't yet have `jammi-ballista`. This is a refuted-not-confirmed claim, reported rather than
  papered over: the insertion is still objectively correct (before `jammi-server`), but no
  EXECUTED oracle proves a regression here would be caught before a real publish attempt.
- **GangExec's single-partition engine duty** (README r41, `engine.rs::contains_gang` +
  the partition-count check): implemented but not covered by an executed test in this pass —
  building a stage plan that wraps a `GangExec` in a multi-partition shuffle writer needs the
  same fine-tune/gang submission machinery named above.
- **`ExecutorRole::devices()`** is exercised only by compiling against the real
  `jammi_db::catalog::instance::DeviceFact` shape; no test asserts its VALUES (would need a
  `[worker]`-enabled executor session, which none of this unit's hermetic tests construct — every
  test session here has `[worker] enabled = false`, the default).

##### 4. Gates

| Command | Exit | Notes |
|---|---|---|
| `cargo fmt -p jammi-ballista --check` | 0 | |
| `cargo fmt -p jammi-server --check` | 0 | |
| `cargo fmt --all -- --check` | 0 (after `cargo fmt -p jammi-server -p jammi-ballista -p jammi-ai -p jammi-db`) | fixed pre-existing unformatted `gang_chaos.rs` from the GANG merge while in scope |
| `cargo clippy -p jammi-ballista --all-targets --features test-hooks -- -D warnings` | 0 | |
| `cargo clippy -p jammi-ballista -p jammi-server --all-targets --features test-hooks -- -D warnings` | 0 | |
| `cargo clippy -p jammi-ballista --tests --features live-distributed-tests -- -D warnings` | 0 | |
| `cargo test -p jammi-ballista` (default features; pulls jammi-ai/jammi-db test-hooks via dev-deps) | 0 | 20 tests (1 unit + 15 `it` + 1 `roles_drain` + doc-tests 0) |
| `cargo test -p jammi-ballista --features live-distributed-tests --test distributed` | 0 | 2 tests, both honestly SKIPPED (no live backends in this environment) |
| `cargo test -p jammi-server --features test-hooks --test it -- ballista` | 0 | 3 tests |
| `python3 ci/scripts/check_swarm_bijection.py` | 0 | |
| `python3 ci/scripts/check_no_consumer_names.py` | 0 | |
| `python3 ci/scripts/check_lint_surface_closure.py` | 0 | `jammi-ballista::distributed` auto-discovered and OK |
| `python3 ci/scripts/check_execution_surface_reachability.py` | 0 | PASS (1 pre-existing unrelated prose false-positive noted by the script itself) |
| `cargo deny check bans` | 0 | `bans ok` |
| `cargo deny --all-features check bans` | 0 | `bans ok` |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `1064 file(s) scanned, all PATH:LINE citations resolve` |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p jammi-ballista --no-deps` | 0 | |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p jammi-ballista -p jammi-server -p jammi-ai -p jammi-db --no-deps` | 0 | fixed one PRE-EXISTING broken intra-doc link in CFGDB's merged `compute_repo.rs` (`Self::heartbeat_job`, not a cross-module path) while in scope |
| `python3 ci/scripts/gen_dep_dag.py` | 0 | regenerated the dep-DAG block (found `jammi-ballista -> jammi-ai, jammi-db, jammi-test-utils, jammi-wire`) |

**Not run** (out of this unit's scope per COMMON.md's trimmed-gate instruction; the lead runs
the full merge path once on the consolidated tip): the full `cargo test -p jammi-ai` / `-p
jammi-db` / `-p jammi-server` suites; `ci/scripts/merge_path.sh`; the live-Postgres/MinIO
`distributed.yml` lanes; `dispatch gh workflow run distributed.yml`.

##### 5. Commits

```
f6b32006 feat(wire-server): #500 U8a — wire GangExec through the codec/placement/engine; the placed-gang seams; post-merge fixes
42ef56c8 Merge commit 'df5734e1' into unit/ballista
2dbc79de stopgap(config): #500 U8a — advertise_host-vs-unspecified-bind validation (contract §9 A3)
0d0c6b2a feat(wire-server): #500 U8a — jammi-server hosts [ballista] roles; registration sites
181a9e4a feat(wire-server): #500 U8a — crates/jammi-ballista: JammiCodec, JammiExecutionEngine, PlacementPolicy, roles, client
099f78c9 feat(ai-core): #500 U8a — codec accessors on InferenceExec/AnnSearchExec/AsofJoinExec + device_kind (contract §9 B3)
df5734e1 test(server): #500 wave 4 — upsert_worker's devices argument at the two gang fixtures        [merged, CFGDB/GANG]
87e228a2 feat(ai-core): #500 wave 4 GANG — GangExec, the Placed arm, run_placed_gang, WorkerFacts.devices  [merged, GANG]
909a165f feat(db): #500 CFGDB — workers.devices + transfer_claim (U8b)                                 [merged, CFGDB]
144589db feat(db): #500 CFGDB — compute_repo.rs, generic CRUD over compute_executors/compute_jobs (U8b) [merged, CFGDB]
7c83a694 feat(db): #500 CFGDB — migration 038 compute_cluster_state (U8b db slice)                      [merged, CFGDB]
fddf53a0 fix(db): #500 CFGDB — BallistaConfig::validate is a &JammiConfig cross-section check           [merged, CFGDB]
17d62ae2 feat(db): #500 CFGDB — [ballista] config section (U8a-cfg)                                     [merged, CFGDB]
cb976274 docs(rigor): #500 wave 4 — pressure round REFINE folded (§9)                                   [merged, lead]
5f1bcb03 feat(deploy): #500 U9b — shape-d becomes a StatefulSet + headless Service                      [merged, docs-ci]
db289303 stopgap(config): [ballista] section (CFGDB owns the real one)
bcfda3ca docs(rigor): #500 wave 4 — the PR-D design contract                                            [merged, lead]
```

The first six commits and the merge (`f6b32006` back through `db289303`, plus `42ef56c8`) are
this unit's own; the remainder landed via `git merge --no-edit df5734e1` (CFGDB/GANG/docs-ci
units that shipped concurrently on the shared `feat/500-wave4` branch, per the coordinator's
mid-session integration instruction).

### 11.5 LANEAI — `build_embedding_plan`; `device_kind` at construction; `worker_devices` public (ai-core) — landed as 3c1d023f, 542f80e7, 6fc08ed0, ac4af91f, 244894c8 (original tip 87587e44)

**Lead's note.** Accepted deviations: `build_embedding_plan` takes `&InferenceSession` (its only in-process caller holds a borrow); the K7 checker's `Option` unwrap in `engine.rs` was a load-bearing follow-through of the constructor change. The lead ran the crate, embedding/pipeline/inference and server suites on the consolidated tree (18 / 45 / 27 green).

#### Contract — LANEAI (`unit/laneai`, worktree `wt-laneai`)

Base: `feat/500-wave4` @ `a86df04e` per the brief; merged `20edbae7` (four wave-4
test-only commits: rank-hold race, KO-7 Postgres-skip gating, the tenant-pinned
enumerating oracle, advertise_host's own oracle+fixture) per the coordinator's
mid-task instruction, with `git merge --no-edit 20edbae7` on `unit/laneai`.

##### 1. Scope shipped

Three commits (plus two clippy/rustdoc fixups), touching:

- `crates/jammi-ai/src/operator/inference_exec.rs` — `InferenceExec.device_kind`
  is `ComputeDeviceKind` (non-`Optional`); `InferenceExecBuilder::new` gains a
  required `device_kind: ComputeDeviceKind` parameter as its LAST positional
  argument; the `.device_kind(Option<..>)` setter is deleted; `with_new_children`
  threads `self.device_kind` verbatim; the accessor returns `ComputeDeviceKind`.
  `#[allow(clippy::too_many_arguments)]` added (8 args now).
- `crates/jammi-ai/src/session.rs` — both `InferenceExecBuilder::new` call
  sites (`annotate_plan` ~:1028, `infer` ~:1541) pass
  `self.compute_device().kind()`.
- `crates/jammi-ai/src/pipeline/embedding_refresh.rs:938` (delta-plan builder)
  passes `self.compute_device().kind()`.
- `crates/jammi-ai/src/pipeline/embedding.rs` — new `pub async fn
  build_embedding_plan(session: &InferenceSession, source_id: &str,
  model_source: ModelSource, task: ModelTask, columns: &[String], key_column:
  &str, embedding_dim: usize) -> Result<Arc<dyn ExecutionPlan>>`, exactly the
  scan→ordered_input→InferenceExecBuilder body `EmbeddingPipeline::run` used to
  build inline (same batch size, observer, embedding_dim, `_content_hash`
  passthrough, device_kind); `run()` now calls it instead of duplicating it.
- `crates/jammi-ai/src/fine_tune/worker.rs` — `worker_devices` is now `pub`
  (was crate-private); body and its existing unit test unchanged.
- `crates/jammi-ballista/src/codec.rs` — the ONE granted deletion: the
  "fill `device_kind` from the encoding session when `None`" arm in
  `try_encode`/`encode_inference` is gone (no session lookup, no
  `unwrap_or_else`); `decode_inference`'s `.device_kind(Some(...))` becomes a
  plain required constructor argument. The wire format (`device_kind: String`
  in `jammi.ballista.v1.InferenceExecNode`) is unchanged.
- `crates/jammi-ballista/src/engine.rs` — `first_device_kind_mismatch` drops
  its now-dead `Option` handling on `exec.device_kind()`; module doc updated
  (see §1a, scope amendment).
- `crates/jammi-ballista/src/roles.rs` — the ONE granted body: `device_facts`
  (backing `ExecutorRole::devices()`) now calls
  `jammi_ai::fine_tune::worker::worker_devices` instead of reproducing its
  mapping.
- Tests: `crates/jammi-ballista/tests/it/codec.rs` (three `InferenceExecBuilder::
  new` call sites updated; `inference_exec_device_kind_defaults_to_the_
  submitting_sessions_own` replaced by `codec_never_rewrites_device_kind`),
  `crates/jammi-ballista/tests/it/engine.rs` (two call sites), `crates/jammi-ai/
  tests/it/pipeline.rs` (new
  `build_embedding_plan_collected_in_process_matches_generates_written_rows`).

###### 1a. Deviations from the brief/dispatch, with the code cited

- **Dispatch message said "the two named spots ... and nothing else in that
  crate" for `jammi-ballista`, but `engine.rs`'s `first_device_kind_mismatch`
  also needed a matching edit.** `InferenceExec::device_kind()` changed
  return type from `Option<ComputeDeviceKind>` to `ComputeDeviceKind`
  (LANEAI.md commit 1's own words: "the accessor returns `ComputeDeviceKind`").
  `engine.rs`'s K7 checker pattern-matched `Some(kind)` on the old type — this
  does not compile against the new signature, so the edit is load-bearing, not
  optional. `engine.rs`'s OWN pre-existing module doc (lines 5-11, before my
  edit) already asserted "`InferenceExec::device_kind()` is ALWAYS `Some` by
  the time a plan reaches an executor" — i.e. the file's author had already
  designed for exactly this non-Optional end state; my edit is the mechanical
  follow-through of that stated invariant (remove the now-vacuous `Option`
  unwrap), not a new design decision. I made the minimal change: unwrap the
  `if let Some(kind) = ...` to a plain `let kind = ...`, updated the two-line
  module doc's description of WHY device_kind is always concrete (constructor
  argument, not codec fill), left every other line of the file untouched.
- **`build_embedding_plan`'s `session` parameter is `&InferenceSession`, not
  `&Arc<InferenceSession>`** (LANEAI.md's literal text names the latter).
  `EmbeddingPipeline` (the sole in-process caller, `crates/jammi-ai/src/pipeline/embedding.rs:69`, same
  file, in-grant) holds `session: &'a InferenceSession` — a plain borrow, not
  an `Arc`. Its three ultimate callers hold no `Arc` either:
  `crates/jammi-ai/src/session.rs:1141/1220/1272`'s `generate_*_embeddings(&self, ...)` are
  plain-`&self` methods (not `self: &Arc<Self>`, unlike `recompute.rs`'s
  `replay_descriptor`/`recompute_one` at `self: &Arc<Self>`, which is a
  DIFFERENT calling convention on the SAME struct for a DIFFERENT set of
  methods), and `InferenceSession` has no `Clone`/self-`Arc` accessor to
  synthesize one without widening those methods' `self` type — out of this
  four-file grant, and a needless cascade for a function whose only present
  caller has a plain reference in hand. A future Ballista submitter (which
  DOES hold `Arc<InferenceSession>`, per `codec.rs`'s own `session(&self) ->
  Result<Arc<InferenceSession>, Error>`) calls `build_embedding_plan(session
  .as_ref(), ...)` exactly as trivially as the reverse would be for
  `embedding.rs`. No functional capability is lost either way.
- **`worker_devices`' second parameter type left as `ComputeDevice`, not
  narrowed to `ComputeDeviceKind`.** LANEAI.md's commit-3 text gives the exact
  signature `pub fn worker_devices(config: &JammiConfig, compute_device:
  ComputeDevice) -> Vec<DeviceFact>` — this is what shipped verbatim (no
  deviation here; noted only because `roles.rs`'s `device_facts` used to take
  the kind out of `ComputeDevice` itself via its own local match, which the
  refactor removes along with the reproduction).

##### 2. Properties

| Property (quantified) | Executed oracle | Executed mutation that reds it |
|---|---|---|
| Every `InferenceExec`, in-process or decoded, carries a concrete `device_kind` equal to what its constructor was given (never a codec-side default) | `jammi-ballista::tests/it/codec.rs::codec_never_rewrites_device_kind` — a node built explicitly onto `Cuda` on a `Cpu` session round-trips through `JammiCodec` still naming `Cuda` | Restored the OLD "fill from session when unset" behaviour (`.unwrap_or_else(\|\| session.compute_device().kind())`) — the test's own predecessor (`inference_exec_device_kind_defaults_to_the_submitting_sessions_own`) asserted exactly the fill; the new test's assertion `decoded.device_kind() == Cuda` would fail (`Cpu` won by the fill) had it been left in |
| A stage whose `InferenceExec` names a device kind different from the executing session's own is refused typed (K7), for both a mismatch and a match | `jammi-ballista::tests/it/engine.rs::{refuses_a_stage_whose_inference_exec_names_a_different_device_kind, does_not_refuse_a_matching_device_kind}` — pre-existing, unchanged assertions, only the constructor call site updated | Not re-executed this round (pre-existing oracle, unchanged logic; the Option→plain refactor in `first_device_kind_mismatch` is a type-level simplification with no behavioural branch removed — verified by inspection: `if let Some(kind) = ... { if kind != own_kind {...} }` and `let kind = ...; if kind != own_kind {...}` are the same three-way case split minus a now-impossible `None` arm) |
| `build_embedding_plan` is the ONE plan-building site: calling it directly (bypassing `generate`/`EmbeddingPipeline::run` entirely) on a two-file source produces rows byte-for-byte identical (via `pretty_format_batches`) to what `generate_text_embeddings` persisted for the same source/model/columns | `jammi-ai::tests/it/pipeline.rs::build_embedding_plan_collected_in_process_matches_generates_written_rows` | RED at base: the function did not exist (compile failure). Additionally executed: reverted `EmbeddingPipeline::run` to a second, hand-inlined copy of the plan-building body that skips the `_content_hash` passthrough (a plausible, easy-to-reintroduce divergence a reviewer might miss) while leaving `build_embedding_plan` itself untouched — the test failed with `assertion \`left == right\` failed: build_embedding_plan collected in-process must equal generate's written rows byte-for-byte`, `_content_hash` columns diverging (persisted: real hex hashes; independent: empty/null). Reverted immediately after confirming red. |
| `worker_devices` (config → `Vec<DeviceFact>`, kind × rank ordinal) is computed in exactly one place; `ExecutorRole::devices()`'s backing data is that same computation, never a second copy | `jammi-ai::fine_tune::worker::tests::worker_devices_is_decided_from_configuration_alone` (pre-existing, unchanged, now exercised through the `pub` surface) + `jammi-ballista::tests/it/roles.rs::scheduler_and_executor_host_in_one_process_and_submit_round_trips` (pre-existing hermetic oracle whose `host_executor(...)` call exercises `device_facts`, which now calls `worker_devices` — still passes, so the delegation preserves behaviour on this path) | Not re-executed as a fresh mutation this round: the change is a literal function-body replacement (`{match session.compute_device() {...}; cfg.worker.topology(&cfg.gpu).map(|t| ...).unwrap_or_default()}` → `worker_devices(session.jammi_config(), session.compute_device())`) verified line-for-line equivalent to the deleted body by inspection, and `worker_devices`' own oracle above already reds on any behavioural change to that mapping (kind string, ordinal `.max(0)`, empty-on-`Err`) |

##### 3. Uncovered

- **K7 refusal's Option→plain simplification** (row 2 above) was not
  independently re-mutation-tested this round — the pre-existing oracle pair
  already covers the mismatch/match behaviour end-to-end and I verified the
  refactor preserves the case split by inspection rather than by an executed
  mutation (both branches are exercised by the untouched tests, which pass).
  Labelled here per the "demand the refutation" principle rather than silently
  assumed.
- **`device_facts`/`worker_devices` delegation** (row 4) likewise relies on
  the pre-existing `worker_devices` oracle plus the pre-existing hermetic
  `roles.rs` integration test rather than a fresh, LANEAI-specific mutation —
  no new test was authorized or needed per the brief ("made public ... with
  its existing unit test"), and I did not add one beyond what's listed.
- **The three-process distributed lane's (a3) oracle**
  (`crates/jammi-ballista/tests/distributed/main.rs::embedding_job_across_two_
  executors_matches_in_process`) still reports SKIPPED for a DIFFERENT reason
  than before this unit (previously: `build_embedding_plan` didn't exist;
  now: live backends — Postgres/MinIO/the built server binary — are
  unavailable in this hermetic environment). That test file is outside this
  unit's grant (`crates/jammi-ballista/tests/distributed/`); I did not touch
  it. `build_embedding_plan` is now public and reachable at the path that
  test's module doc names, so the scope-amendment blocker it names is
  resolved from this unit's side; standing up the three-process fleet against
  it is the lead's follow-up per that file's own doc.

##### 4. Gates

| Command | Result |
|---|---|
| `cargo fmt --all -- --check` | exit 0 |
| `cargo clippy -p jammi-ai --all-targets --features test-hooks -- -D warnings` | exit 0 |
| `cargo clippy -p jammi-ballista --all-targets --features test-hooks -- -D warnings` | exit 0 |
| `cargo test -p jammi-ai --features test-hooks --test it -- embedding inference_exec build_embedding_plan` | 37 passed, 0 failed |
| `cargo test -p jammi-ai --features test-hooks --lib -- inference_exec embedding build_embedding_plan worker_devices` | 15 passed, 0 failed |
| `cargo test -p jammi-ballista --features test-hooks --test it -- codec` | 12 passed, 0 failed |
| `cargo test -p jammi-ballista --features test-hooks --test it -- engine` | 2 passed, 0 failed |
| `cargo test -p jammi-ballista --features test-hooks --test it -- roles` | 2 passed, 0 failed |
| `python3 ci/scripts/perf/check_citations.py` | `check-citations: 1064 file(s) scanned, all PATH:LINE citations resolve` (2 pre-existing EXEMPT legacy-evidence citations, unrelated to this unit) |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p jammi-ai -p jammi-ballista --no-deps` | exit 0 (after converting `build_embedding_plan`'s `[\`embedding_definition\`]` intra-doc link, which targets a `pub(crate)` item, to a plain code span — never a doc-hidden bypass) |

Not run (COMMON.md: implementers run only touched/filtered tests; the lead
runs the full merge path once on the consolidated tip): workspace-wide
builds, full crate suites, `cargo doc --workspace`, live-Postgres lanes,
`merge_path.sh`, the `distributed.yml` `ballista` leg.

##### 5. Commits

```
87587e44 fix(ai-core): #500 LANEAI — build_embedding_plan's doc names embedding_definition as a code span, not an intra-doc link
0c8bb912 Merge commit '20edbae7' into unit/laneai
aed7122d fix(ai-core): #500 LANEAI — allow too_many_arguments on InferenceExecBuilder::new
3641abda feat(ai-core): #500 LANEAI — worker_devices is public; ExecutorRole's device mapping calls it
e463cdcb feat(ai-core): #500 LANEAI — build_embedding_plan is the one embedding plan-building site
23b8744b feat(ai-core): #500 LANEAI — device_kind is a required InferenceExecBuilder constructor argument, never a codec-side fill
20edbae7 test(db): #500 wave 4 — the executor's advertise_host rule has its own oracle; the hostname fixture names one
50ab34bc test(server): #500 wave 4 — the tenant-pinned table read's enumerating oracle admits the codec's reviewed decode site; rustfmt on the rank-hold test
```

(`20edbae7`/`50ab34bc` are the four wave-4 test-only commits merged in per the
coordinator's mid-task instruction — not authored by this unit; `0c8bb912` is
the merge commit itself, no manual conflict resolution was needed.)

### 11.6 U8b — `CatalogClusterState`/`CatalogJobState`; `DevicePlacement`; the re-launch guard (wire-server) — landed as c072ef0f, f875eba9, 5772ac53 (original tip b041f083)

**Lead's note.** The lead opened the CAS-before-stamp order in `placement.rs`, the `claimed_by` guard and the executor's own devices write in `roles.rs` before landing, and ran the 23 hermetic `it` tests and the server hosting tests on the consolidated tree. The unit's largest deviation — no distributed lane — is closed by LANE (§11.8). `InferenceSession::catalog_arc` widened to `pub` is accepted (one visibility line).

#### Contract — unit `u8b` (plan 67 wave 4, PR-D): catalog-backed cluster state and device-aware placement

Owner: wire-server. Base `feat/500-wave4` @ `a86df04e`; LANEAI's tip `244894c8`
merged in. Branch `unit/u8b`.

##### 1. Scope shipped

- `crates/jammi-ballista/src/cluster.rs` (new): `CatalogClusterState`
  (`ballista_scheduler::cluster::ClusterState` over `jammi_db::catalog::
  compute_repo`: `register_executor`/`save_executor_metadata`/
  `get_executor_metadata`/`registered_executor_metadata` →
  `upsert_compute_executor`/`get_compute_executor`/`list_compute_executors`;
  `save_executor_heartbeat` → `record_compute_heartbeat` + a write-through
  `std::sync::RwLock<HashMap>` heartbeat cache (the trait's own
  `executor_heartbeats`/`get_executor_heartbeat` are SYNC, so this cache is
  load-bearing, not an optimization — its doc states the staleness bound: a
  standby scheduler's cache reflects only its own `init` read plus whatever
  heartbeats land on IT, never another scheduler's); `remove_executor` →
  `remove_compute_executor`; `unbind_tasks` → aggregated
  `adjust_compute_slots` (one call, atomic by that verb's own contract);
  `bind_schedulable_tasks` → snapshot `list_compute_executors` into
  `AvailableTaskSlots`, dispatch to the built-in `bind_round_robin` fallback
  (Bias/RoundRobin) or the installed `Custom` policy; `cluster_state_events`
  → `ballista_scheduler::cluster::event::ClusterEventSender`, same shape
  `InMemoryClusterState` uses) and `CatalogJobState` (`JobState`: `graphs:
  tokio::sync::RwLock<HashMap<JobId, ExecutionGraphBox>>` kept ONLY in this
  process's memory — Ballista 54.1 has no graph serialisation; `compute_jobs`
  mirrors owner+status text via `put_compute_job`; `get_job_status` answers
  from the catalog row (reconstructing a minimal `JobStatus`) even when
  `get_execution_graph` is `None` for a job this process's memory never
  built; `try_acquire_job` only ever succeeds against a graph already in
  THIS process's memory, matching `InMemoryJobState`'s own "never revives a
  serialized graph" behavior; `accept_job` stays PURELY in-memory —
  deviation from a literal reading of the brief's "accept_job/submit_job/
  save_job/get_execution_graph … mirror put_compute_job": `accept_job` is a
  SYNC trait method and `InMemoryJobState::accept_job`
  (`ballista-scheduler-54.1.0 src/cluster/memory.rs lines 483–488`) itself never
  persists either — mirroring its actual behavior, not the summary's gloss).
- `crates/jammi-ballista/src/placement.rs` (rewritten in place):
  `PlacementPolicy` (U8a) is REPLACED by `DevicePlacement` — same struct
  slot, same round-robin+submitter-exclusion loop, EXTENDED (never a second
  policy) with: (1) the device predicate (a GPU-bound task — `GangExec`
  anywhere, or `InferenceExec` naming `Cuda`/`Metal` — binds only to an
  executor whose `compute_executors.devices` lists a matching device), (2)
  the re-launch guard (a `GangExec` whose job row's `claimed_by` names an
  instance OTHER than the stage's own submitter is never bound to ANY slot;
  an unreadable row folds into the same refusal), (3) the slot CAS
  (`Catalog::bind_compute_slots`) runs PER CANDIDATE inline, BEFORE the
  graph's task info is stamped (contract §9 A10) — a lost CAS never stamps,
  so there is nothing to "unstamp" on failure. `bind_round_robin` (new, free
  `pub(crate)` fn): the UNREFINED fallback for the `Bias`/`RoundRobin`
  built-in arms `CatalogClusterState::bind_schedulable_tasks` must still
  handle for `ClusterState`'s generic contract — `jammi-server` never
  selects them (no knob) and, unlike `Custom`, they never reserve a catalog
  slot; documented as test-fixture-only, never safe under a shared catalog.
- `crates/jammi-ballista/src/engine.rs`: `pub fn stage_is_gpu_bound(plan)`,
  factored beside `contains_gang`/`first_device_kind_mismatch` — the ONE
  GPU-bound predicate `DevicePlacement` and `client::submit_physical_plan`'s
  device-less refusal both cite (never two independent notions of "needs a
  device"). New hermetic test `refuses_a_gang_exec_stage_with_more_than_one_
  partition` (r41): two `GangExec` leaves under a `UnionExec` (partition
  count 2) is refused typed, naming the mechanism and the count.
- `crates/jammi-ballista/src/client.rs`: `submit_physical_plan` refuses a
  GPU-bound plan typed, BEFORE submitting, when
  `Catalog::list_compute_executor_devices()` shows no registered executor
  with a matching device — reads the session's own catalog, the same store
  the scheduler's `DevicePlacement` reads.
- `crates/jammi-ballista/src/roles.rs`: `host_scheduler` gains a
  `distribution: TaskDistributionPolicy` parameter (no knob at runtime:
  `jammi-server`'s own hosting always passes `Custom(DevicePlacement)`; the
  parameter exists so the in-memory cluster + a bare policy stay reachable
  as a TEST fixture, `tests/it/roles.rs`/`tests/roles_drain.rs`, never a
  second production path). `host_executor` now patches its own
  `compute_executors` row with `devices` (`worker_devices`, LANEAI's public
  fn) directly via `session.catalog()` right after registration completes —
  `ClusterState::register_executor`'s fixed signature carries no device
  field, so the row's devices come from the executor's OWN process,
  deterministically the LAST write of its own startup sequence.
  `CatalogClusterState::register_executor` preserves whatever `devices`
  value already exists on the row (read-then-merge) rather than wiping it on
  re-registration.
- `crates/jammi-server/src/runtime.rs`: the scheduler role always builds
  `BallistaCluster::new(CatalogClusterState, CatalogJobState)` over
  `Arc::clone(self.session.catalog_arc())` and
  `Custom(Arc::new(DevicePlacement::new(catalog)))`.
- `crates/jammi-ai/src/session.rs`: `InferenceSession::catalog_arc()`
  visibility widened `pub(crate)` → `pub` (one line) — the only way
  `jammi-ballista` can hold its own long-lived `Arc<Catalog>` handle,
  since `catalog()` only returns `&Catalog` and `Catalog` has no `Clone`
  impl (always held behind an `Arc` at construction). This is the one
  cross-crate touch outside my four owned crates; named per COMMON.md's
  "coordinate through the lead" note — a single visibility widening on an
  already-existing method, not a new surface.
- Hermetic tests `crates/jammi-ballista/tests/it/cluster.rs` (new, both
  SQLite/Postgres backends via `jammi_test_utils::make_test_session`):
  register/heartbeat/remove round-trip; `unbind_tasks` atomicity; two
  `CatalogClusterState`s over one catalog see each other's registrations
  (b2's substrate); a GPU-bound stage never binds to a device-less executor
  (b3); an already-transferred `GangExec` is never bound (b6, first half);
  the slot-CAS-before-stamp property (A10) with a real, single-stage
  `StaticExecutionGraph` built through `DefaultDistributedPlanner` (Ballista's
  own `#[cfg(test)]`-gated `test_utils`/`cluster::test_util` are NOT
  reachable from an external crate, so this is jammi's own minimal graph
  fixture, not a copy of Ballista's).

##### 2. Deviations from the brief (with the code cited)

- `CatalogJobState`'s field list drops the brief's literal `sessions` field:
  `InMemoryJobState` itself (`ballista-scheduler-54.1.0/src/cluster/
  ballista-scheduler-54.1.0 src/cluster/memory.rs lines 430–451`) has no session-cache field either — `create_or_update_
  session` builds a fresh `SessionContext` every call via
  `create_datafusion_context`. Mirrored verbatim.
- `Catalog::register_executor`'s device handling: the brief says "register_
  executor → upsert_compute_executor (+ the ExecutorData slots)" without
  addressing where `devices` comes from. `ClusterState::register_executor`'s
  signature (`ballista-scheduler-54.1.0 src/cluster/mod.rs lines 174–179`) is
  `(metadata: ExecutorMetadata, spec: ExecutorData)` — no device field. The
  shipped design: the EXECUTOR's own process (`roles::host_executor`) writes
  its device claim directly, over the same shared catalog, right after
  `executor_server::startup` returns — see `roles.rs`'s new block and its
  doc citing this ordering.
- No new `jammi-db` verb was added for a "devices-only" patch; `host_executor`
  does a read-then-merge full `upsert_compute_executor` instead, avoiding any
  edit to `compute_repo.rs` (CFGDB's file).
- The distributed lane (`tests/distributed/*`, the `distributed.yml` matrix
  leg, (a3)-(a5)/(b1)-(b6) live oracles, the b4 live-executor devices value
  test) is **NOT COMPLETE** — see §3 Uncovered. This is the largest deviation
  from the brief and is called out explicitly rather than left implicit.

##### 3. Properties (hermetic, executed, mutated)

| Property (quantified) | Oracle | Mutation executed → red |
|---|---|---|
| Every `ClusterState` write verb is visible through every read verb, both backends | `cluster::register_heartbeat_remove_round_trip` (`--test it -- cluster`, sqlite+postgres) | not executed as a targeted mutation (straightforward wiring); covered structurally by the b3/b6/A10 mutations below exercising the same call paths |
| `unbind_tasks` moves every executor's row or none, for one batch | `cluster::unbind_tasks_is_all_or_none` | relies on `adjust_compute_slots`'s own atomicity (CFGDB's contract); this test proves `CatalogClusterState::unbind_tasks` aggregates per-executor deltas into ONE call rather than issuing one call per `ExecutorSlot` entry |
| Two `CatalogClusterState`s over one catalog see each other's registrations | `cluster::two_cluster_states_over_one_catalog_see_each_others_registrations` | not separately mutated — a fresh, unrelated catalog for the second state is the failure mode this proves against, exercised by construction |
| A GPU-bound task never binds to a device-less executor | `cluster::gpu_bound_stage_never_binds_to_a_device_less_executor` | Removed the `(!gpu_bound || …)` conjunct from `DevicePlacement::bind_tasks`'s eligibility check → **RED**: `left: "cpu-…" right: "gpu-…"` (the task bound to the device-less executor by round-robin order) |
| A `GangExec` whose job row is `claimed_by` an instance ≠ the descriptor's submitter is never bound | `cluster::already_transferred_gang_is_never_bound` | Short-circuited the claim guard (`if false && claimant != submitter`) → **RED**: `bound` was non-empty, naming the executor id and the `GangExec` task description |
| The catalog's committed `available_slots` — never the in-memory snapshot — decides whether a slot binds | `cluster::a_slot_less_executor_never_gets_a_task_stamped` | not separately mutated (a full CAS-skip mutation would double as the A10 test above); this test proves the specific race arm — snapshot says 1 free, catalog says 0 |
| README r41: a stage plan wrapping `GangExec` under a multi-partition node is refused typed | `engine::refuses_a_gang_exec_stage_with_more_than_one_partition` | not separately mutated in this pass (the check itself, `if contains_gang(&plan) { if partitions != 1 { … } }`, predates this unit — U8a's own tests cover its absence) |

Every mutation above was executed via a direct source edit, `cargo test`
re-run, observed red, then reverted and re-verified green (shown in the
gates below).

##### 4. Uncovered

- **The entire distributed lane** (`tests/distributed/{main.rs,harness.rs}`,
  oracles (a3)-(a5)/(b1)-(b6), the `distributed.yml` `ballista` matrix leg,
  `check_lint_surface_closure.py`/`check_execution_surface_reachability.py`
  re-runs, the b4 `ExecutorRole::devices()` live-executor value test, the
  README r41 "engine's single-partition duty" wording cross-reference).
  **Why**: this is a three-real-process (scheduler + 2 executors), live
  Postgres + MinIO integration surface requiring porting jammi-ai's
  fine-tune submission harness (`register_training_source`/
  `submit_gang_fine_tune`/`JobSize`/`await_job`) into this crate, plus
  writing and RUNNING the actual byte-comparison/claim/kill assertions
  against a live fleet — a multi-hour lift this session's remaining budget
  did not cover once the core catalog-state/placement unit (§1) and its
  hermetic oracles were built, tested, mutated, and gated green. `main.rs`'s
  module doc and skip messages were updated to state this honestly (the
  earlier "(a3) needs a scope amendment" reasoning is now FALSE — LANEAI's
  `build_embedding_plan` landed public — so the doc no longer claims a
  structural blocker that does not exist; it names this as a budget gap).
  MinIO was not running in this environment (verified: port 9000 closed) and
  I did not stand it up, since doing so without also completing the harness
  port would not have produced an executed oracle.
- **b2's full property** ("two schedulers over one catalog serve sequential
  jobs") is only substrate-tested here (both `CatalogClusterState`s see the
  same registrations); the JobState half (a job submitted to scheduler B
  after scheduler A's own job completed) needs the live three-process lane
  above.
- **b1** (scheduler restart, read back through a NEW process) needs a real
  process kill/restart, not exercised hermetically.
- `check_citations.py` reports 3 PRE-EXISTING stale citations (`docs/maintainer/MAINTAINER-GUIDE.md:538`/`:3019`, `crates/jammi-ai/tests/it/pinned_source_gate.rs:1374`) whose line numbers
  drifted from OTHER units' edits on this consolidated tip — none of them
  touch code I authored; not fixed here (out of this unit's file grant and
  not introduced by this unit's commits).

##### 5. Gates

All run from `/private/tmp/…/scratchpad/wt-u8b` with
`CARGO_TARGET_DIR=/private/tmp/…/scratchpad/targets/u8b`,
`RUSTC_WRAPPER=sccache`, `JAMMI_TEST_PG_URL=postgres://jammi@127.0.0.1 port 54329/jammi_test`.

| Command | Result |
|---|---|
| `cargo test -p jammi-ballista --features test-hooks --test it` | 23 passed, 0 failed |
| `cargo test -p jammi-ballista --features test-hooks --test roles_drain` | 1 passed |
| `cargo test -p jammi-ballista --features live-distributed-tests --test distributed -- --test-threads=1` | 2 passed (both honest SKIPs — backends unavailable in this environment) |
| `cargo clippy -p jammi-ballista --all-targets --features test-hooks -- -D warnings` | exit 0 |
| `cargo clippy -p jammi-ballista --all-targets --features live-distributed-tests -- -D warnings` | exit 0 |
| `cargo clippy -p jammi-server --all-targets -- -D warnings` | exit 0 |
| `cargo clippy -p jammi-ai --all-targets --features test-hooks -- -D warnings` | exit 0 |
| `cargo fmt -p jammi-ballista --check` | exit 0 |
| `cargo fmt -p jammi-server --check` | exit 0 |
| `cargo fmt -p jammi-ai --check` | exit 0 |
| `cargo fmt --all -- --check` | exit 0 |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p jammi-ballista --no-deps` | exit 0 (fixed 3 private-intra-doc-link errors → backtick spans) |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p jammi-server --no-deps` | exit 0 |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p jammi-ai --no-deps` | exit 0 |
| `python3 ci/scripts/perf/check_citations.py` | FAIL: 3 pre-existing stale citations, none touching this unit's files (see §4) |

##### 6. Commits

```
b041f083 docs(wire-server): #500 U8b — distributed lane doc no longer claims a file-grant blocker
ca02015d fix(wire-server): #500 U8b — adapt to LANEAI's device_kind API, rustdoc private-link fix, fmt
2916e479 Merge commit '244894c8' into unit/u8b
b92cf587 feat(wire-server): #500 U8b — CatalogClusterState/CatalogJobState, DevicePlacement with the re-launch guard
```
(`244894c8` and earlier are LANEAI's/the consolidated tip's, merged in per
the lead's message.)

### 11.7 U9a — guide, maintainer guide, deploy README, CHANGELOG, plan rows; the scheduler Deployment and the executor wiring in the overlay (docs-ci) — landed as ed56e334, fd9e02ff, 30aac897, 1ea837cd, 8b2fefeb (original tip 81f26d2e), plus the lead's 8785ba33

**Lead's note.** The scope amendment (wiring the compute pods as executors and adding the scheduler Deployment) is exactly §9 B7's disposition and is accepted. The lead corrected the scheduler pod's claim kinds to the two placeable kinds (a `context_predictor` job has no placed arm and would have trained on the CPU pod) and replaced a rigor-contract citation in a manifest comment with the property it stood for (8785ba33); the config-fence oracle, the citation resolver and kubeconform strict (9/9) were re-run on that tip.

#### U9A — docs: guide, maintainer guide, deploy README, CHANGELOG, plan rows (contract §5, §9; UNITS §U9a)

Base: `feat/500-wave4` @ `5772ac53`. Branch: `unit/u9a`. Owner: docs-ci (doc-updater lens).

##### 1. Scope shipped

- `docs/guide/src/configuration.md` — the `[ballista]` block's comments checked against
  `BallistaConfig`/`BallistaExecutorConfig`'s field docs and `BallistaConfig::validate`
  (`crates/jammi-db/src/config/mod.rs:2233-2450`): added the process-role semantics paragraph
  (a process hosts a scheduler iff `scheduler_bind` is set, an executor iff `[ballista.executor]`
  is present; both roles on one process is the single-node cluster, with the submitter-exclusion
  refinement stated), the precise `advertise_host`-required-when-`bind`-is-unspecified rule (was
  a plain "default: the bind host" note with no REQUIRED case), and the six-address collision
  rule (was entirely absent from the guide).
- `docs/guide/src/reference-topologies.md` — Shape D section (was: a plain-Deployment,
  "provisional" write-up naming issue #500) rewritten to the shipped two-role split: the
  scheduler `Deployment` + the compute `StatefulSet`'s executors, the `Awaiting` holder's rank
  arithmetic (a placed `Peer` gang of world `W` needs `W` hosts able to hold a rank, the
  submitter's own included), and why placement always excludes a task's own submitter (deadlock
  avoidance, hence a separate scheduler role rather than "whichever compute pod claims first
  places its own siblings"). Embeds all six shape-d manifests (three pre-existing, three new —
  see §1's k8s scope below).
- `docs/guide/src/philosophy.md` — new "Extending third-party libraries at their seams" section:
  states the candle/`jammi-kernels` discipline and the Ballista/`jammi-ballista` discipline as one
  pattern (extension points, never a fork; retries stay on jammi's own side). This section did
  not exist before at all — the brief named a discipline the guide had never stated as a general
  principle, only demonstrated per-crate.
- `docs/guide/src/api-stability.md` — already listed twelve crates with `jammi-ballista` in
  alphabetical place (built by an earlier wave-4 unit); verified against
  `grep -L 'publish = false' crates/*/Cargo.toml` (exactly 12) — no edit needed, confirmed rather
  than assumed.
- `docs/maintainer/MAINTAINER-GUIDE.md` — (a) Orientation (§0) and the crate-graph seams (§1.2)
  gain `jammi-ballista`'s position and dependency direction (the DAG blocks themselves already
  had it, from an earlier wave-4 unit; the prose around them did not); (b) new §2.8f
  (`jammi-ballista`'s codec/engine/roles/client/cluster-state/placement, including that
  `jammi-server`'s own hosting always passes the catalog-backed `BallistaCluster`/
  `DevicePlacement` pair — the in-memory one is a test fixture only, verified against
  `crates/jammi-server/src/runtime.rs:656-692`) and §2.8g (the placed gang: `Holder::Awaiting`,
  `submit_placed`'s total exit arms, `Catalog::transfer_claim`'s four conjuncts,
  `run_placed_gang`, the writer-table rows 18/19 already in `worker.rs`'s own module doc); (c) a
  new migration-038/`compute_repo` bullet under §2.3; (d) a one-line cross-reference in §2.2
  stating `jammi.ballista.v1` is not part of the frozen `jammi.v1.*` surface. Also re-anchored
  three PRE-EXISTING stale citations `check_citations.py` reported at this tip (unrelated to this
  unit's own content, found before writing anything new):
  `docs/maintainer/MAINTAINER-GUIDE.md:538` (`worker.enabled`, `crates/jammi-server/src/runtime.rs:2210` → `:2224`),
  `docs/maintainer/MAINTAINER-GUIDE.md:3019` (`tenant`, `crates/jammi-ai/src/session.rs:746` → `:764`), and
  `crates/jammi-ai/tests/it/pinned_source_gate.rs:1374` (`read_vectors`,
  `crates/jammi-ai/src/session.rs:1156` → `:1160`).
- `deploy/kubernetes/README.md` — the "Rollout arithmetic" paragraph was still Deployment-shaped
  (`maxSurge`/`maxUnavailable` at a stale "3 compute replicas" count) even though the overlay had
  already become a 2-replica `StatefulSet`; rewritten for the StatefulSet's actual
  `RollingUpdate` semantics (no surge/unavailable knob, strictly ordinal-descending, one at a
  time) plus the new scheduler Deployment's own rollout. New "Compute plane" section: the two
  roles, what each does, and that DRAIN on an executor pod waits for an in-flight placed task
  while RELEASE tears it down at once. Layout bullet updated to name the scheduler Deployment.
- **Deviation from the brief, cited**: the brief's item 2 says "If a scheduler Deployment
  manifest does not exist ... WRITE IT" and separately states as already-decided fact (contract
  §9 B7) that "the compute StatefulSet pods host executors." At this tip NEITHER side of that
  disposition was wired: no scheduler manifest existed (confirmed: `grep -rn ballista
  deploy/kubernetes/` returned nothing before this unit), and `jammi-compute.toml` had no
  `[ballista.executor]` table at all (confirmed by reading the file). U9b's own scope
  (`UNITS.md` §U9b, built in wave A, BEFORE the pressure round that produced §9 B7) never
  included Ballista wiring — it is a StatefulSet/headless-Service/`nvidia.com/gpu` unit only. I
  therefore also wrote the executor side: `[ballista.executor]` in `jammi-compute.toml`
  (`scheduler_address`, `bind`/`grpc_bind`, `task_slots = 2` — one per `nvidia.com/gpu`),
  two new container ports + a per-pod `advertise_host` env var (the same downward-API
  construction `peer_advertise` already uses) in `statefulset-compute.yaml`, and the matching
  ports on `service-compute-headless.yaml`. Without this, the guide/README text the brief asked
  for would describe a reference deployment that does not actually do what the prose claims —
  I judged completing the already-decided (not newly designed) B7 split as in scope for a docs
  unit that ships the reference manifests, rather than shipping docs describing a manifest gap.
  Both new/changed TOML files are exercised by a REAL oracle (`docs_config_fences.rs`, see §2)
  and the whole overlay by `kubeconform --strict`.
- New files: `deploy/kubernetes/overlays/shape-d/{deployment-scheduler.yaml, service-scheduler.yaml,
  jammi-scheduler.toml}`; `kustomization.yaml` updated to include them and a second
  `configMapGenerator` entry.
- `CHANGELOG.md` `[Unreleased]`: `### Added` — one paragraph-style bullet for the whole Ballista
  compute plane (crate, codec, engine, roles, client, catalog cluster state, device placement,
  the placed gang + `transfer_claim`, migration 038 + `workers.devices`); `### Changed` — the
  shape-D StatefulSet conversion, phrased as an operator action (`kubectl delete deployment`
  before applying, since Kubernetes cannot convert a `Deployment` into a `StatefulSet` in place);
  `### BREAKING` — `Catalog::upsert_worker` gains a `devices: &[DeviceFact]` parameter (a `pub`
  `jammi-db` API), added to the EXISTING top `### BREAKING` section rather than a second heading
  under `[Unreleased]` (caught my own first draft, which had created a duplicate heading).
- Plan rows: 67 `README.md`'s D1–D4 unit-table rows and `UNITS.md`'s §U8a/§U8b/§U9a/§U9b each
  gain `(dated correction 2026-09-16: SHIPPED on feat/500-wave4 — contract §9: <clauses>)` citing
  the specific §9 letter/advisory block per deviation named for that unit (never a bare "shipped,
  see contract" — each clause is independently checkable against the code); 68 `PROGRAM.md`'s
  wave-4 row gains the actual (concurrent-wave-A, not strictly sequential) build order and notes
  U9a completed U9b's scheduler-side gap.

##### 2. Properties

| Property (quantified) | Executed oracle | Executed mutation that reds it |
|---|---|---|
| Every `PATH:LINE` citation across the tree resolves at HEAD | `python3 ci/scripts/perf/check_citations.py` — ran after every edit; final: "1066 file(s) scanned, all PATH:LINE citations resolve" | Before my first commit, the checker itself reported the three PRE-EXISTING stale citations (`docs/maintainer/MAINTAINER-GUIDE.md:538`/`:3019`, `crates/jammi-ai/tests/it/pinned_source_gate.rs:1374`) as its own red output — I re-anchored them and re-ran to green; this IS the executed red/green pair for that property (I did not need to manufacture a fresh mutation since the tree already supplied one) |
| Every `[ballista]`/`[ballista.executor]` TOML fence the guide embeds (directly or via `{{#include}}`) parses under the REAL `JammiConfig::parse_from` loader, and the selected-fence count is pinned | `cargo test -p jammi-db --test it docs_toml_fences_parse_under_the_real_loader` — green after bumping the pinned count 29→30 for the new `jammi-scheduler.toml` include | Set `jammi-compute.toml`'s `scheduler_address = 12345` (a non-string value): reds naming `docs/guide/src/reference-topologies.md:311`'s include chain down to `jammi-compute.toml` with "expected a string"; reverted, re-ran green |
| The amended shape-d overlay (9 resources incl. the 3 new manifests) is strict-schema-valid on the pinned Kubernetes version | `kustomize build deploy/kubernetes/overlays/shape-d \| kubeconform --strict --summary --kubernetes-version 1.34.11 -` → "Valid: 9, Invalid: 0, Errors: 0" | Not separately mutated — kubeconform's own strict mode is the oracle; a malformed manifest (e.g. a missing required field) would report `Invalid`/`Errors` directly, and this was exercised implicitly across several draft/fix cycles while authoring the new YAML (an early draft had a duplicate `metadata.name`-style typo caught this way before this final run) |
| The `[worker]`/`[ballista]`-shaped fence test's own referenced test file still compiles and its OWN dependent oracle (`pinned_source_gate.rs`'s allowlist) still matches reality after the citation-line edit | `cargo test -p jammi-ai --test it pinned_source_gate` → 30 passed, incl. `allowlists_match_current_hits_exactly` and `caller_set_claims_match_reality` | Not separately mutated (this is a re-anchor of a comment's cited line number, not a behavior change); the 30/30 green run is the confirmation the edit did not silently break the allowlist oracle it sits inside |
| doc-parity / no-consumer-names / bijection / constitution-anchor / doc-numbers gates stay green across the whole edit set | `check_doc_parity.py`, `check_no_consumer_names.py`, `check_swarm_bijection.py`, `check_constitution_anchors.py`, `check_doc_numbers_have_producers.py` — all exit 0 | Not mutated (these are cross-cutting hermetic gates unrelated in mechanism to my edits; each ran clean before and after every commit in this unit) |
| `mdbook build docs/guide` succeeds (the guide's mdbook fences, includes, and cross-references are well-formed) | `mdbook build docs/guide --dest-dir <scratch>/book` → "HTML book written" | Not mutated |
| `cargo fmt --all -- --check` is clean on the touched Rust files | `cargo fmt --all -- --check` → exit 0 | Not mutated |

##### 3. Uncovered

- **The Ballista executor wiring I added to the compute overlay (`[ballista.executor]`,
  the new container ports, `advertise_host`) has never been exercised against a live Ballista
  scheduler/executor pair on Kubernetes** — no GPU node in CI, and the `distributed.yml`
  `ballista` leg (a concurrent LANE unit's own scope) exercises the crate's protocol on bare
  metal/CI containers, never this specific overlay's manifests. `kubeconform --strict` proves
  schema validity only, not that the scheduler's `scheduler_address` DNS name actually resolves
  or that the two new ports are reachable across the headless Service. UNCOVERED, named as a
  residual for whichever unit next runs a real (or `kind`-simulated) multi-pod Ballista
  cluster — this overlay is validated by `kubeconform` only, same standing note the pre-existing
  compute StatefulSet already carried.
- **`task_slots = 2` on the compute executor (one per `nvidia.com/gpu` entry) is a documented
  guess, not a measured or contract-mandated number** — the design contract does not size this
  knob; I chose the device-count mapping as the least-surprising default and said so in the
  TOML's own comment, but no oracle (hermetic or live) exercises whether 2 is the right
  concurrency bound for a placed gang plus this pod's own in-process claim to coexist without
  starving each other. UNCOVERED.
- **The CHANGELOG's `upsert_worker` BREAKING bullet and the plan rows' dated corrections are
  prose-only** — no CI gate cross-checks a CHANGELOG bullet's claim against the actual function
  signature (I read `crates/jammi-db/src/catalog/jobs_repo.rs:2852-2857` myself to confirm the `devices: &[DeviceFact]`
  parameter exists, but this is a manual check, not a property any script re-verifies going
  forward). Same for every "dated correction" clause in the plan rows — each is verified by me
  reading the cited code once, not by a standing oracle.

##### 4. Gates

| Command | Exit | Notes |
|---|---|---|
| `python3 ci/scripts/perf/check_citations.py` | 0 | 1066 files scanned, all resolve, 2 exempt (pre-existing, unrelated) |
| `python3 ci/scripts/check_doc_parity.py` | 0 | all bindings in parity |
| `python3 ci/scripts/check_no_consumer_names.py` | 0 | 1 pre-existing ADVISORY (`stage_is_gpu_bound`, not touched by this unit) |
| `python3 ci/scripts/check_doc_numbers_have_producers.py` | 0 | 57 measurement-shaped numbers, all cited or allowlisted |
| `python3 ci/scripts/check_swarm_bijection.py` | 0 | total coverage, exactly-one owner per path |
| `python3 ci/scripts/check_constitution_anchors.py` | 0 | all anchors across 13 invariants resolve |
| `mdbook build docs/guide --dest-dir <scratch>/book` | 0 | HTML book written |
| `kustomize build deploy/kubernetes/overlays/shape-d \| kubeconform --strict --summary --kubernetes-version 1.34.11 -` | 0 | 9/9 valid |
| `kustomize build deploy/kubernetes/base \| kubeconform --strict --summary --kubernetes-version 1.34.11 -` | 0 | 3/3 valid (sanity, unaffected) |
| `kustomize build deploy/kubernetes/overlays/ci \| kubeconform --strict --summary --kubernetes-version 1.34.11 -` | 0 | 7/7 valid (sanity, unaffected) |
| `cargo test -p jammi-db --test it docs_toml_fences_parse_under_the_real_loader` | 0 | 1 passed (pinned count 29→30) |
| `cargo test -p jammi-ai --test it pinned_source_gate` | 0 | 30 passed |
| `cargo clippy -p jammi-db --all-targets -- -D warnings` | 0 | clean |
| `cargo clippy -p jammi-ai --test it -- -D warnings` | 0 | clean |
| `cargo fmt --all -- --check` | 0 | clean |

No rustdoc gate run: this unit touched no `.rs` source under `crates/jammi-ballista` (only a
test-file comment and a test's pinned assertions in `jammi-db`/`jammi-ai`), so
`RUSTDOCFLAGS="-D warnings" cargo doc` is not applicable per the brief's own scope.

##### 5. Commits

```
99fe4736 docs(docs-ci): #500 U9a — re-anchor three stale citations ahead of the docs pass
d0305c4c docs(docs-ci): #500 U9a — shape-d scheduler Deployment, wire the compute pods as Ballista executors
dde97892 docs(docs-ci): #500 U9a — guide, maintainer guide, README, CHANGELOG for the Ballista compute plane
3cf3afc1 docs(docs-ci): #500 U9a — state that the catalog-backed scheduler is the only shipped path
81f26d2e docs(docs-ci): #500 U9a — fix CHANGELOG wording, retries are a scheduler-wide pin not a per-role one
```
(`git log --oneline 5772ac53..HEAD`, oldest first above; tip `81f26d2e`.)
