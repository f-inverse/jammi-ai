# Job dependencies and grouping — decision record

**Decision.** The `jobs` table does not model dependencies between jobs (`depends_on`) or job
groups (`parent_id`). Sequencing work — "run B after A completes", "cancel everything under this
group" — belongs to the consumer's runtime. What the engine provides is the vocabulary a consumer
needs to sequence jobs safely from outside: a durable job row, idempotent submission, terminal
statuses that distinguish *could not run* from *was asked to stop*, a status stream, and
cooperative cancel. Discussion: <https://github.com/f-inverse/jammi-ai/issues/515>.

## Why the engine declines

**It does not pass the discipline test as a primitive.** Orchestration is on the consumer's side
of the boundary `docs/guide/src/philosophy.md` draws ("load balancing, ingress, TLS, secrets, IAM,
observability stack, orchestration, autoscaling — is the consumer's runtime, not the engine's").
A dependency graph is a statement about a consumer's workflow; the engine's jobs are independent
units of durable compute.

**A correct implementation needs a control loop the engine does not ship.** The actuator rule
(`crates/jammi-ai/src/pipeline/recompute.rs`, module doc: "the engine ships the actuator; it never
ships the control loop that pulls it") allows one bounded sweep on one explicit request. Evaluating
a design for dependencies on the two catalog backends showed that every workable shape pulls the
engine past that line:

- **Readiness.** Storing a `blocked` status needs something to move rows `blocked → queued` when a
  dependency completes — a coordinator deciding *when*. Projecting readiness inside the claim
  predicate instead avoids the stored transition, but every claim must then examine every
  higher-priority blocked row: measured at about 6 buffers per blocked-ahead row on PostgreSQL
  16.15 and about 1.2 µs per row on SQLite 3.46.0, a cost paid on every poll by every worker.
- **Outcome propagation is unbounded in one transaction.** When a dependency fails, its dependants
  must be durably retired (typed error text, retention, prunability and waiter termination all key
  on stored state). Pushing the outcome transitively inside the failing write's transaction was
  measured at 32,766 rows in one transaction for a fan-out-2 × depth-14 tree; that transaction
  blocks the claim path's compare-and-set and holds SQLite's single writer for its whole duration.
  Bounding the transaction leaves deeper levels, and anything a crash interrupted, to a
  reconciliation pass. A whole-system pass has no frontier — a queued row with a
  terminal-unsuccessful dependency spans two tables, and nothing about that predicate shrinks when
  the system is idle — so on the worker tick it re-scans to discover nothing (measured at 24,166
  buffers with nothing to retire on a 113,000-row table), and off the tick it is a standing sweep
  in all but name.
- **Attach-time races.** A design that converges only at write time lets a later submit attach a
  child under an ancestor that is already doomed and never retire it. Closing that needs the
  submit to walk the ancestor chain under lock, serialised against every terminal write
  (`FOR SHARE` on PostgreSQL — the foreign key's own `KEY SHARE` does not block a status `UPDATE` —
  and the single writer on SQLite), which couples submit latency to graph shape and makes the
  claim path's `SKIP LOCKED` skip a dependency row for the window of every submit naming it.
- **Pruning.** A terminal row becomes unprunable while any dependant or child is live, so retention
  has to reason about graph reachability, and the edge table's `ON DELETE CASCADE` foreign keys
  become load-bearing for correctness rather than hygiene.

Each item is solvable; together they make the jobs table a workflow scheduler. Other job systems
that offer dependencies pay for it the same way: pg-boss releases blocked dependants from a
background resolver, River Pro and Oban Pro ship workflow engines on top of their queues, and
Graphile Worker declines the feature.

## What the engine provides instead

- **`cancelled` is a terminal status distinct from `failed`**
  (`crates/jammi-db/src/catalog/status.rs::JobStatus`). A cancel on a job no worker has claimed
  retires it in the request's own transaction, `queued → cancelled`; a `running` job is flagged
  and its executor ends the row at its next checkpoint
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::cancel_request`,
  `crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::cancel_job`). An `inline` row is never
  retired by the request: its submitter claims it synchronously and owns its outcome. A waiter on
  a cancelled job receives the typed `JobCancelled` error on both transports; the Python client
  raises `jammi.JobCancelled`.
- **One terminality predicate.** Every terminality decision, in Rust and in the Python client,
  derives from `JobStatus::is_terminal` / `JobStatus::is_terminal_unsuccessful` or its mirror
  rather than comparing status strings; a source-tree gate holds the rule
  (`crates/jammi-db/tests/it/terminality_source_gate.rs`). A consumer's sequencer can therefore key
  on "terminal and unsuccessful" without enumerating statuses.
- **Idempotent submission** by per-tenant idempotency key
  (`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::submit_job_deduped`), so a sequencer that
  crashes between observing A's completion and submitting B can resubmit B safely.
- **Status as a stream** (`WaitJob`) and as a read (`get_job`), tenant-scoped.

## Backend facts measured during the evaluation

These hold independently of the decision and apply to any code that writes set-shaped SQL against
the two catalog backends (PostgreSQL 16.15; SQLite 3.46.0 as bundled by `libsqlite3-sys`).

- `RETURNING` order is unspecified on both backends and tracks neither the input order nor an
  upstream `ORDER BY` (a descending input came back ascending on both). Every consumer of
  `RETURNING` rows sorts in Rust.
- sqlx-sqlite 0.8.6 binds a placeholder whose index exceeds the argument count as NULL, silently
  (`src/arguments.rs`: "SQLite treats unbound variables as NULL; we reproduce this here"). A
  chunked statement must compute its bind indices from the actual chunk length, or a partial final
  chunk binds NULLs.
- A bound `LIMIT` flips PostgreSQL to a generic plan once sqlx's persistent statement has run a
  few times (measured turning an index-driven plan into a sequential scan); the catalog
  interpolates every `LIMIT` as a literal.
- The portable spelling of a set-shaped update is `WITH v(id, msg) AS (VALUES …) UPDATE … FROM v`;
  SQLite rejects `FROM (VALUES …) AS v(id, msg)`. `UPDATE … FROM` needs SQLite ≥ 3.33.
- PostgreSQL rejects `FOR UPDATE` combined with `UNION`/`INTERSECT`/`EXCEPT`.
- PostgreSQL does not define `AND`/`OR` operand evaluation order; a nested `CASE` forces it.
- A scalar sublink `(SELECT … LIMIT 1) IS NULL` is evaluated per row with index probes and stops at
  the first match; the equivalent `NOT EXISTS` was planned as a hashed whole-catalog subplan.
- SQLite plans a correlated `ORDER BY … LIMIT 1` sublink with a temporary B-tree; `MIN(…)` over the
  same rows plans without one and is deterministic on both backends.
- On SQLite a bare boolean conjunct (`AND claimable`) is not an index-usable equality: a claim
  candidate query spelled that way materialises and sorts the `(status, execution)` range
  (`USE TEMP B-TREE FOR ORDER BY`, 4.0 ms at 20,000 rows), whereas `claimable = TRUE` makes it a
  three-column index constraint whose remaining columns satisfy the `ORDER BY` (16 µs). PostgreSQL
  plans both spellings identically.

## References

- pg-boss `docs/api/jobs.md` — flow / `job_dependency` table; dependants created blocked and
  released by a background resolver; a parent failure leaves the child blocked.
- River `rivertype.JobState` and River Pro Workflows — `pending` is never worked unless moved;
  dependants cancelled by default when an upstream is discarded, cancelled or deleted.
- Oban Pro Workflow — holds until dependencies complete; cancels dependants of a cancelled,
  discarded or deleted upstream by default.
- Graphile Worker FAQ — no dependency feature.
- PostgreSQL "Explicit Locking" — https://www.postgresql.org/docs/current/explicit-locking.html —
  `FOR SHARE` blocks `UPDATE` and is compatible with `FOR SHARE`; `FOR KEY SHARE` blocks `DELETE`
  but not `FOR NO KEY UPDATE`.
- PostgreSQL 9.3 release notes — https://www.postgresql.org/docs/release/9.3.0/ — foreign-key
  checks use `KEY SHARE`, which does not conflict with `NO KEY UPDATE`.
- PostgreSQL "Expression Evaluation Rules" —
  https://www.postgresql.org/docs/current/sql-expressions.html
