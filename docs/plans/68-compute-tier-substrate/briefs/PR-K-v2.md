# Brief PR-K-v2 — amendments to the Kubernetes reference unit (#482 item 2)

Plan of record: `PLAN-K.md` (copy at $S/plans/PLAN-K.md) with refinements V1–V10, pressure-tested to PROCEED on 2026-09-10; brief `PR-K.md` (copy at $S/briefs/PR-K-orig.md). Do NOT re-plan it. Produce PLAN-K-v2 = the delta only, so the amendments can be pressure-tested alone.

## What surfaced since (fold in, or refuse with the principle)
A1. Worker naming: `JAMMI_WORKER_ID` is the worker's human label shown by `ListWorkers` (wt-C crates/jammi-ai/src/fine_tune/worker.rs, `WORKER_LABEL_ENV`). Set it from the pod name via the downward API on the compute Deployment so the listing reads as pod names. Base query tier: not needed ([worker] enabled=false).
A2. Kind-specific pools: `[worker] kinds` is the pool seam (wt-C config/mod.rs WorkerConfig). Do not add a second overlay; one README paragraph states how a second compute Deployment with a different `kinds` list is a copy of shape-d with one TOML line changed.
A3. Deployer seams to NAME in the README, not ship: PodDisruptionBudget on the query tier, PriorityClass separating query from compute, a ReadWriteMany PVC or init-container pre-pull for `HF_HOME` (today `emptyDir` → every pod restart re-downloads the tower), HPA/KEDA on queue depth (once the OPS unit exposes the gauges). Principle: orchestration/autoscaling/observability stack are the consumer's runtime (philosophy.md "How it deploys"). Decide per item whether it is a seam (name it) or a shape the engine cares about (ship it) and say why.
A4. terminationGracePeriodSeconds on the compute tier: today SIGTERM stops the claim loop and drains RPCs but does NOT wait for the in-flight training thread; the process exits when serve returns (wt-C crates/jammi-server/src/runtime.rs ~:476-566, session.close()), the lease expires, the job is re-queued from scratch by reclaim. A long grace period is therefore useless until the OPS unit ships drain mode. Decision to validate: PR-K ships the compute overlay WITHOUT a grace period and with a comment naming the behaviour; the OPS unit adds the grace period and the drain mechanism in the same PR (B6: a manifest knob and the mechanism it depends on land together).
A5. Spot/preemptible GPU node overlay: same dependency on drain mode → OPS unit, not here. Confirm or refute.
A6. reference-topologies.md still says `services = ["train"]` for Shape D (main :243, :278); PR-C's todo already carries that fix. Confirm PLAN-K commit 3 supersedes it and that the two do not conflict at merge.
A7. The pin table's open item: `yannh/kubernetes-json-schema` commit sha for `-schema-location` — PLAN-K's table says b582a12a09aa9b5d1edad577a964c504130214fc; PR-K.md says OPEN. Resolve which is current by reading the plan and, if needed, the GitHub API; state the resolved pin.

## Precondition (unchanged)
Cuts after PR-C merges. State whether any amendment changes that.

Output: PLAN-K-v2 as a delta: for each of A1–A7, decision + principle + exact file/line edits to PLAN-K's commits + the oracle that proves the amendment (RED first where a test exists). Keep the pinned-versions discipline: every SHA copied from PLAN-K, never refreshed.
