# Issue comment drafts (unposted; 2026-09-10)

## #482

## Design note from the 2026-09-10 review: Kubernetes unit outcome, the distributed data plane, and a correction

**Kubernetes unit (this issue) — plan at PROCEED after five pressure rounds; cuts after PR-C.** Shape unchanged: kustomize base + overlays. What the pressure test changed:
- The base ships the engine default, `[worker] enabled = true` with kinds/services "all"; the query-tier overlay (shape d) *replaces* the base ConfigMap with `enabled = false` and adds a compute ConfigMap (`configMapGenerator` `files: [jammi.toml=<file>]`, `behavior: replace`). `enabled = false` in the base would strand every job kind, since `enqueue` always writes `execution = queued`.
- Nothing is sized for the deployer: pods ship BestEffort, no `terminationGracePeriodSeconds`; the compute tier carries only the device-plugin request `nvidia.com/gpu: 1`. Sizing, QoS and grace are documented seams with a copy-paste strategic-merge patch, not shipped values.
- Images stay `:latest`. Correction for the record: the earlier claim that the cu12 `:latest` tag is never published was wrong. docker-publish passes an empty `flavor`, so metadata-action's `latest=auto` emits `latest` on every non-prerelease `v*` tag; v0.49.1 already did. The honest floor note is that cu12 `:latest` is 0.49.1 today, predates `[worker]`, and CrashLoops on the new ConfigMap until the next `v*` release, which should be cut right after PR-C and PR-K merge.
- Worker identity comes from the downward API (`JAMMI_WORKER_ID` = pod name); kubeconform runs strict against the pinned 1.34 schema plus a kind smoke; the compute tier stays a provisional Deployment pending #500.

**Distributed data plane — Ballista withdrawn for the data plane; adopted by 67 for the compute/gang plane as a seam-level extension (`jammi-ballista`, see `PROGRAM.md` §Ballista). Two of the grounds below were revised at 54.1 source (README §Corrections).** Two corrections first. (1) An earlier statement here that Ballista's client submits only logical plans was wrong: at 54.1.0 `execute_physical_plan` submits a pre-built physical plan. (2) The follow-up position, "embed Ballista in the binary, upstream-only", did not survive the second round either. Verified against the 54.1.0 source: the executor resource model is `task_slots` only (no accelerator dimension), cluster state is `ClusterStorage::Memory` only, shuffle is a local `work_dir` (no object-store stage), the scheduler is a sensor-to-actuator control loop in its own process, and the pin is DataFusion 54 against our 52.3. Spice AI's production use corroborates the limits: batch-only, sub-second stays single-node, and on forks. Ballista stays a future option behind three upstream contributions (accelerator resource + placement, pluggable cluster storage, object-store shuffle), which is where a Spark-MLlib-shaped relationship would start.

**Position adopted instead, by plane:**
- Batch inference and eval = the jobs fleet (PR-C), with a shard primitive from the job-dependency and incremental-embedding units. Job-level parallelism only, as recorded on #500.
- Online retrieval beyond one node = in-house scatter-gather between replicas on an internal `[server] peer_bind` listener (`jammi.v1.peer.PeerService`): coordinator enforces tenant scope, owners are tenant-free with a segment-belongs-to-table check, two-phase per precision (approximate + exact rescore for F16/Int8, final for Binary), width = over_fetch(k·oversample, N). Default unset; binding it on a routable interface without network policy or mTLS is documented as a cross-tenant exposure.
- Distributed SQL over large result tables = `datafusion-distributed` (library, no scheduler process), deferred behind the DataFusion 55 gate; the only blocker today is `datafusion-flight-sql-server` 0.4.18 pinning DF ^54.

**Follow-up units from this review**, each planned and pressure-tested separately, to be filed as their own issues: compute-tier operability (DRAIN/RELEASE shutdown, `jammi-server release`, queue and liveness gauges, warm-before-ready); job dependencies on the jobs table (blocked as a projection, outcome pushed by the terminal write); incremental embedding over versioned tables and immutable segments.

## #500

> **SUPERSEDED 2026-09-10 — do not post.** Replaced by the 67 v4 comment on #500 (see `README.md` §Corrections recorded by 67).

## Third design option: planned training (DataFusion/Ballista), alongside the fleet and the gang

Recorded from the 2026-09-10 design session so the target-workload decision can pick between three shapes, not two.

**Shape.** One training round is one physical plan: scan the pairs/neighbourhoods (the worker already assembles them by SQL today, `fine_tune/worker.rs` `session.sql`), repartition across executors, a partial-gradient operator per partition (the existing precomputed-tensor loop `pipeline/parallel_train.rs` wrapped as an `ExecutionPlan`, the way `InferenceExec` already is, with the frozen tower held in the executor's model cache across plans), an aggregate that sums the adapter deltas, and the coordinator applies the optimizer step and issues the next round. Gradients and parameters travel as Arrow batches through the same operators, shuffles and Flight as data. A trained adapter is a materialized result with a definition hash and replayable lineage, like an embedding table.

**Where it fits.** A plan executes once, so per-step synchronous SGD becomes per-round parameter averaging (local SGD / federated averaging). That converges for adapters and heads (megabytes of parameters, frozen tower); it does not for full fine-tunes, whose sharded weights need an all-reduce every step. So this option answers the first open question by construction:

- adapters on encoder towers → planned training; Ballista (or an in-process DataFusion distribution) carries it
- full fine-tunes of large towers → the gang (NCCL in the binary); no DataFusion substrate helps

**What Ballista lacks today** (verified at 54.1.0): no accelerator dimension in the executor resource model (`ExecutorResource { vcores }`), no barrier stage, no executor-side state across plans. Those are the upstream contributions this option would need, in that order. Ballista is therefore not adopted now for any plane (see the #482 note of the same date); this option is recorded so the target-workload decision can weigh it, not as a commitment. Spice AI's production use of Ballista is batch-only (sub-second stays single-node) and lives in a fork; jammi's rule would be upstream-only.

**Risk to measure before any distribution work.** Convergence of local SGD vs synchronous SGD for the target adapter workloads — measurable now with a single-process two-partition simulation on the distributed lane's fixtures.

**Consequence for #482.** Unchanged: the compute tier stays a plain Deployment marked provisional; planned training needs no stable rank identity, so if the workload decision lands on adapters, the provisional note may simply be removed rather than replaced by a StatefulSet.
