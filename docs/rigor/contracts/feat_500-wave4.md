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
scheduler_bind = "0.0.0.0:50050"          # Some = this process hosts a Ballista scheduler
[ballista.executor]
scheduler_address = "10.0.4.7:50050"      # Some = this process hosts a Ballista executor at that scheduler
bind = "0.0.0.0:50051"                    # the executor's Arrow Flight (shuffle) listener
grpc_bind = "0.0.0.0:50052"               # the executor's gRPC (task) listener
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
`JAMMI_SERVER__PEER_BIND = 0.0.0.0:9000`), `terminationGracePeriodSeconds: 600` kept (OPS C6's
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

## 9. Pressure round

Recorded here after the verdict, with each block's disposition.

## 10. Gate table

Written at consolidation: tip, stage, result, log.
