# Brief GRAPH — job dependencies and fan-in on the jobs table (job-graph coordination without gangs)

Unit: feature. Cuts after PR-C merges. One PR.

## Why
Today the jobs table (wt-C crates/jammi-db/src/catalog/schema.rs ~:991-1032, migrations 029/030) holds independent jobs claimed by interchangeable workers (skip-locked claim, lease, CAS finalize). A user with a large workload — embed a corpus in shards, mine negatives then fine-tune then eval, a sweep of N fine-tunes with a winner — sequences it from a client script today. The engine should carry the dependency edges so that N workers cooperate through the catalog on one workload while staying interchangeable (no rank identity, no rendezvous — those are #500 and out of scope). Job kinds already include compute kinds `embedding`, `infer`, `neighbor_graph`, `propagate`, `asof_join` (wt-C crates/jammi-ai/src/jobs.rs ~:171-175) plus the three training kinds.

## Decisions taken (validate or refute)
- Mechanism, not composition: the engine ships (a) `depends_on: [job_id]` on submit — a job is `blocked` until every dependency is terminal-success; a dependency's failure or cancel fails the dependant with a typed reason naming the dependency; (b) `parent_id` for grouping: `ListJobs` filters by parent, `CancelJob` on a parent cascades to unfinished children; (c) fan-in is nothing new: a job whose `depends_on` lists its siblings. NO sweep verb, NO pipeline DSL, NO scheduler process: consumers compose those above (philosophy: engine ships the actuator, not the control loop; discipline test).
- Unblocking is decided inside the existing claim query (a job is claimable iff no non-terminal-success dependency exists), not by a separate sweep or a status transition performed by a coordinator — no new process, no new control loop, consistent with skip-locked claiming on both backends. Evaluate the cost on the claim query and index it.
- Edges live in an append-only child table `job_dependencies(job_id, depends_on_job_id)` (K5, new migration 031 on top of wt-C's 030), not an array column (SQLite/Postgres parity, indexability). `parent_id` is a nullable column on `jobs`.
- Tenant scope: a dependency may only name a job visible to the submitter's tenant (B5); cross-tenant references are refused at submit with INVALID_ARGUMENT (K2 at the input edge).
- Idempotency keys already exist; a resubmitted parent with the same key returns the same handle — state how children interact with dedupe.
- Wire: `SubmitJobRequest` gains `repeated string depends_on` and `string parent_id`; `JobStatusResponse` gains `blocked` as a status and the failing-dependency reason; freeze baseline updated in the same PR; K4 parity: embedded (`run_now`/library submit) and remote produce identical rows.
- Deletion/pruning (`PruneJobs`) must not orphan edges or delete a parent with live children — define it.

## Forks
F1. Cycle detection at submit (a job cannot depend on itself transitively): the dependencies must already exist at submit (ids are minted before), so a cycle is impossible by construction — verify that claim; if a child can be submitted before its parent, state the rule.
F2. Should a `blocked` job hold a lease? No (nothing runs). Confirm the reclaim sweep ignores `blocked`.
F3. Cancel cascade semantics for a running child: request cancel (existing `cancel_requested`), never kill.
F4. Whether `parent_id` is a real FK (ON DELETE?) — choose and justify against PruneJobs.
Reference systems to validate the semantics (cite): PostgreSQL-backed queues with dependency support (e.g. River, Graphile Worker, pg-boss), and note what they do on dependency failure. Do NOT import an Airflow/Temporal-shaped DAG model (platform).

## Acceptance
- Two workers, three jobs (A, B independent; C depends on A and B): C is never claimed before both are `completed`; if A fails, C lands `failed` with a reason naming A (both backends).
- Cancel on a parent: queued children land `cancelled`, running children get `cancel_requested`, completed children untouched.
- Cross-tenant `depends_on` refused at submit; same-tenant accepted.
- K4: identical `jobs` rows (minus timestamps/ids) from the embedded submit and the wire submit.
