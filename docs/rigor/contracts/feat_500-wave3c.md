# Contract — `feat/500-wave3c`: the remainder of plan 67's wave 3, one consolidated PR

Base: `main` @ `b5a7aab9` (PR #580 merged: U5b-1a-A2 root identity, U5b-0 leaf inventory,
the merge-path runner). Slug `feat_500-wave3c`; pressure row in
`docs/rigor/feat_500-wave3c.jsonl`; oracle record in `docs/rigor/feat_500-wave3c.oracle.jsonl`
at the final tip, exported last, after every non-rigor edit.

## 1. Scope and build order

The plan's sizing doc pins two hard orders inside the remainder (U4b before U5b-1b-ii;
U5b-1b-ii before U5b-1b-iii and U5b-2); everything else is independent and was built
concurrently, one implementer per unit in its own worktree off `856ec8dd`, consolidated here
as one commit per unit:

| # | Unit | Built | Depends on |
|---|---|---|---|
| 1 | U4b — rank context; gather rule; lockstep; single-node `Local` gang; `[worker] local_ranks` | concurrently (wave A) | main |
| 2 | U5b-1b-i — `Peer` collective + two-phase round protocol; peer-listener decode cap | concurrently (wave A) | main |
| 3 | U5a-2 — `HostAdmission`, drain, re-verification, admit-and-hold; the world>1 admission conjunct (#566) | concurrently (wave A) | main |
| 4 | U5b-1b-ii (db) — `[distributed] max_world_size`; migration 037; assembly reason table, cooldown/counter; claim-side cooldown term; CAS call-site helper | concurrently (wave A) | main |
| 5 | U5b-1b-ii (body) — coordinator: membership → assignment → dispatch → assembly; `serveable_world` at submit | after 1–4 | 1, 2, 3, 4 |
| 6 | U5b-1b-iii — `world_size == 1` rank body; runner-role writer split; `Outcome`; resume pin | after 5 | 5 |
| 7 | U5b-2 — watchdog; abort with no terminal write; released-vs-failed; chaos | after 6 | 5, 6 |

Not in this PR: U7b-A2b live (RunPod cluster runs, with the user); the pod and cluster legs'
artifacts (label-gated, run after merge).

Each unit's section below is the implementer's contract file, folded by the lead after the
lead opened every cited line and re-ran every named oracle on the consolidated tree; every
deviation from `docs/plans/67-distributed-training/UNITS.md` is stated with its reason and the
plan rows carry a dated correction in the same commit.

## 2. U4b

(folded at consolidation)

## 3. U5b-1b-i

(folded at consolidation)

## 4. U5a-2

(folded at consolidation)

## 5. U5b-1b-ii — database slice (landed as one commit on this branch; original `6dab678b`)

The implementer's contract, folded by the lead after opening the cited lines (the cooldown term is `lease_expired_clause("next_assembly_after", ..)` verbatim; `lease.rs` carries a zero diff; `AssemblyOutcome::effect` has no wildcard arm; the pressure round's block 8 is closed by this shape).

### 1. Scope shipped

Files touched (all inside `crates/jammi-db` and its docs; `[worker]` untouched, per the brief):

- `crates/jammi-db/src/config/mod.rs` — new `DistributedConfig { max_world_size: u32 }`
  section (default 1, `deny_unknown_fields`), `JammiConfig.distributed` field, `Default`
  wiring, and a `config.distributed.validate()?` call in `load_from` (refuses `0`).
- `crates/jammi-db/src/config/tests.rs` — 4 tests for the new section (default, TOML
  round-trip, zero-refusal at `validate()` and at `load_from`, independence from
  `[worker]`).
- `crates/jammi-db/src/catalog/schema.rs` — `MIGRATION_037_JOBS_ASSEMBLY_FAILURES_NEXT_AFTER`:
  `jobs.assembly_failures INTEGER NOT NULL DEFAULT 0`, `jobs.next_assembly_after` (nullable
  `TEXT`, same representation `lease_expires_at` uses).
- `crates/jammi-db/src/catalog/migrations.rs` — the `("037_jobs_assembly_failures_next_after",
  schema::MIGRATION_037_JOBS_ASSEMBLY_FAILURES_NEXT_AFTER)` tuple appended.
- `crates/jammi-db/src/catalog/jobs_repo.rs`:
  - `AssemblyOutcome` (9 variants: `Refuted`, `AllRootDivergent`, `Unavailable`,
    `StoreUnavailable`, `ShortListed`, `NoBody`, `Drain`, `Cancelled`, `Success`) and a
    private `AssemblyEffect` (`Neither | CooldownOnly | CooldownAndCounted | Reset`) — a
    proper 4-way enum rather than a `(bool, bool)` pair, so "counted but not cooled" (a
    combination the design never calls for) is unrepresentable. `AssemblyOutcome::effect`
    is an exhaustive match (no wildcard): a new variant with no rule is a compile error.
  - `assembly_backoff(failures: u32) -> Duration` — bounded exponential (`2s * 2^k`,
    `k` clamped to 8 before the shift to avoid overflow regardless of `failures`, final
    ceiling 300s / 5 minutes, both named constants).
  - `Catalog::record_assembly_outcome(job_id, attempt, outcome) -> Result<bool>` — a
    locking read of the current `assembly_failures` (guarded by `job_id`+`attempts`, `FOR
    UPDATE` on Postgres; SQLite's `BEGIN IMMEDIATE` already serializes), then ONE `UPDATE`
    that writes both columns per the rule, using `super::lease::lease_deadline_expr` (the
    SAME helper `claim_next`'s own lease stamp uses) for the cooldown deadline. Returns
    `false` with no write when `attempt` no longer names the row's current attempt.
  - `claim_next`'s candidate subselect gains `AND {cooldown_clause}`, where
    `cooldown_clause = lease_expired_clause("next_assembly_after", kind, &mut params)` —
    **reused verbatim**, not a new function (see the pressure-round correction below).
  - `TrainingSetAssembly { Won, Reused, Moved }` and
    `Catalog::materialize_or_reuse_training_set(job_id, claimed_by, attempts, ref,
    location) -> Result<TrainingSetAssembly>` — a thin, DESIGN.md-vocabulary wrapper of
    `fill_training_set_identity`, mapping `Filled → Won`, `Reused → Reused`, `Aborted →
    Moved`.
- `crates/jammi-db/tests/it/migrations.rs` — `037_...` appended to `EXPECTED_MIGRATION_NAMES`
  and the 029-replay `DELETE FROM applied_migrations ... IN (...)` list (037 ALTERs `jobs`);
  a new column-presence assertion after the replayed reopen (the fifth K5 pin site's teeth);
  a new `migration_037_is_ordered_after_036_and_adds_assembly_failures_next_after` oracle
  (sqlite+postgres) mirroring 034/035/036's own shape, plus a default-value assertion
  (`assembly_failures = 0`, `next_assembly_after = NULL` on a row naming neither column).
- `crates/jammi-db/tests/it/assembly_outcome.rs` (new) — every behavioural oracle below.
- `crates/jammi-db/tests/it/main.rs` — `mod assembly_outcome;`.
- `docs/guide/src/configuration.md`, `docs/maintainer/MAINTAINER-GUIDE.md` — `[distributed]
  max_world_size` documented (the guide fence is inside the ONE big TOML fence
  `docs_config_fences.rs`'s real-loader oracle already walks, so it is exercised, not just
  asserted in prose).

#### Deviations from the brief, with the cited reason

1. **No `cooldown_ready_clause` function.** The brief (`U5b-1b-ii-db.md`) asked for a new
   `(next_assembly_after IS NULL OR next_assembly_after <= <backend now>)` clause. Mid-task
   the lead relayed a pressure-round finding, verified against `crates/jammi-db/src/catalog/lease.rs:13-26,
   47-56, 60-73, 147-158`: SQLite's arm deliberately binds the APPLICATION clock (never a
   SQL clock function, which truncates precision) while Postgres's reads the DATABASE's
   `now()`, and the two backends store different representations. The binding correction:
   reuse `lease_expired_clause` VERBATIM (its `<` predicate, not a new `<=` one) rather than
   invent a second, near-duplicate clock helper. I cut the `cooldown_ready_clause` function
   and its lease.rs tests entirely and wired `claim_next`'s cooldown term straight to
   `lease_expired_clause("next_assembly_after", kind, &mut params)` — `lease.rs` ends the
   unit with a NET-ZERO diff (`git diff --stat crates/jammi-db/src/catalog/lease.rs` is
   empty), so it is not part of this unit's shipped diff at all.
2. **Acceptance (b) restated, per the same relay:** "no second clock source: the cooldown
   predicate is rendered by the lease module's helpers" (a source-scan oracle), plus the
   process-clock-skew oracle scoped to the Postgres arm only. Both built exactly as
   directed — see properties 5 and 6 below.
3. **`record_assembly_outcome` is a locking SELECT + one UPDATE, not a single bare UPDATE.**
   The brief's prose says "one UPDATE applying that rule"; I read that as "one coherent,
   atomically-guarded write" rather than "zero reads permitted" — embedding the exponential-
   backoff arithmetic (`2^k` clamped, ceiling, per-dialect `LEAST`/`MIN`) directly in raw SQL
   for two dialects risked exactly the overflow/precision class of bug `assembly_backoff`'s
   own doc comment warns about, for no benefit: the read is a locked, guarded, same-
   transaction SELECT (`FOR UPDATE` on Postgres; SQLite's `BEGIN IMMEDIATE` already
   serializes per `catalog::migrations`' own module doc and the concurrent-migrate tests),
   and the SUBSEQUENT write is exactly one `UPDATE` statement per call, guarded by the same
   `job_id`/`attempts` CAS `fill_training_set_identity` uses. `assembly_backoff` itself is a
   pure, independently testable Rust function (see property 4's escalation oracle), not
   duplicated per-dialect SQL.
4. **`materialize_or_reuse_training_set` maps `Aborted → Moved`, not a bespoke third state.**
   `fill_training_set_identity` (existing, `crates/jammi-db/src/catalog/jobs_repo.rs`) already
   distinguishes "won"/"reused"/"aborted, no write" exhaustively; the brief's ask ("returns
   whether it WON or REUSED; a moved claim ... aborts with NO write, a distinct typed
   outcome") is satisfied by a thin rename into the coordinator's own vocabulary
   (`TrainingSetAssembly`), not a new state machine.

### 2. Properties

| Property (quantified) | Executed oracle (test path::name, lane/features) | Executed mutation that reds it (change, red output first line) |
|---|---|---|
| (a) A higher-priority job inside its cooldown never blocks a lower-priority ready job, on either backend. | `assembly_outcome::cooldown_job_never_blocks_a_lower_priority_ready_job::{sqlite,postgres}` (`--features live-postgres-tests,test-hooks`) | Removed `AND {cooldown_clause}` from both `candidate` branches in `claim_next` (jobs_repo.rs). Red (both dialects, verified separately with/without `JAMMI_TEST_PG_URL`): `assertion \`left == right\` failed: the higher-priority job must be skipped while cooling down ... left: "cooling-high" right: "ready-low"`. |
| (b1) No second clock source: every cooldown-bearing SQL fragment in `jobs_repo.rs` is rendered through `catalog::lease`'s helpers, never a hand-written `CURRENT_TIMESTAMP`/`datetime('now'`/`chrono::Utc::now()`. | `assembly_outcome::cooldown_sql_has_no_second_clock_source` (hermetic, no backend) | Added `const MUTATION_PROBE_TOKEN: &str = "CURRENT_TIMESTAMP";` to jobs_repo.rs. Red: `jobs_repo.rs must never hand-write a clock-bearing SQL fragment ... found the literal "CURRENT_TIMESTAMP"`. |
| (b2, Postgres-only) The cooldown predicate is governed ENTIRELY by the Postgres SERVER's own clock — a value set via `now() ± interval` and compared via the same `now()`, never a value this test computes from its own process clock. | `assembly_outcome::postgres_cooldown_is_governed_by_the_server_clock_alone` (`--features live-postgres-tests`, requires `JAMMI_TEST_PG_URL`) | Same mutation as (a) (cooldown clause removed). Red: `the hour-ahead cooldown must still hold the second row back`. |
| (c1) The assembly outcome table is exhaustive and total: every one of the 8 non-success variants gets exactly its documented (counted?, cooled?) pair, on both backends. | `assembly_outcome::assembly_outcome_table_is_exhaustive_and_each_rule_fires_once::{sqlite,postgres}` | Reassigned `AllRootDivergent` from `CooldownAndCounted` to `CooldownOnly` in `AssemblyOutcome::effect`. Red: `assertion \`left == right\` failed: all-root-divergent: counted=true left: 0 right: 1`. |
| (c2) Consecutive COUNTED failures escalate the backoff (bounded exponential on the counter), on both backends. | `assembly_outcome::consecutive_counted_failures_escalate_the_backoff::{sqlite,postgres}` | Pinned `assembly_backoff`'s exponent to `0` regardless of `failures` (constant backoff). Red: `the second counted failure's backoff (1.999...s remaining) must exceed the first's (1.999...s remaining)`. |
| (c3) A success resets BOTH the counter and the cooldown, even after a counted failure armed both. | `assembly_outcome::success_resets_the_counter_and_the_cooldown::{sqlite,postgres}` | Remapped `Success` to `AssemblyEffect::Neither` (no reset). Red: `assertion \`left == right\` failed: success must reset the counter to 0 left: 1 right: 0`. |
| (d) `record_assembly_outcome`: a moved claim (stale `attempts`) writes NOTHING, on either backend. | `assembly_outcome::record_assembly_outcome_moved_claim_aborts_without_a_write::{sqlite,postgres}` | Dropped the `attempts = $2` guard from the internal locking SELECT (job_id-only). Red: `a stale attempt number must write nothing at all`. |
| Migration 037 ordered after 036; adds `jobs.assembly_failures` (NOT NULL) / `jobs.next_assembly_after` (nullable TEXT), on both backends. | `migrations::migration_037_is_ordered_after_036_and_adds_assembly_failures_next_after::{sqlite,postgres}` | RED at base: the migration, the const, and the test are all new in this diff — the test cannot even compile against the pre-change tree (`schema::MIGRATION_037_...` does not exist). |
| The fifth K5 pin site: 037 is on the 029-replay's ledger-clearing `IN (...)` list, so a manufactured pre-029 reopen replays it and the recreated `jobs` table carries both columns. | `migrations::migration_029_copies_training_jobs_rows_into_jobs_as_queued` | Dropped `'037_jobs_assembly_failures_next_after'` from the `DELETE FROM applied_migrations ... IN (...)` list. Red: `reopened jobs must carry 'assembly_failures' (migration 037 replayed): []`. |
| (f1) `materialize_or_reuse_training_set`: first call WINS; a repeat under the same claim, same values, REUSEs; a concurrent racer REUSEs (never a second write); a moved claim aborts with NO write. | `assembly_outcome::materialize_or_reuse_training_set_{first_call_wins,second_call_same_values_reuses,concurrent_racer_reuses_never_overwrites,moved_claim_aborts_without_a_write}::{sqlite,postgres}` | Remapped `TrainingSetFillOutcome::Aborted → TrainingSetAssembly::Reused` (instead of `Moved`). Red: `assertion \`left == right\` failed left: Reused right: Moved`. |
| `[distributed] max_world_size` defaults to 1, round-trips through TOML, refuses `0` at both `validate()` and `load_from`, and loads independently of `[worker]`. | `config::tests::distributed_config_{default_is_single_rank,toml_round_trips_max_world_size,zero_max_world_size_is_refused_at_load,loads_independently_of_worker}` (`--features test-hooks`, `--lib`) | Disabled the `max_world_size == 0` branch (`if false && ...`) in `DistributedConfig::validate`. Red: `called \`Result::unwrap_err()\` on an \`Ok\` value: ()`. |
| The `[distributed]` guide fence parses under the real loader (doc-parity). | `docs_config_fences::docs_toml_fences_parse_under_the_real_loader` (pre-existing oracle, exercised by the new fence content) | Not separately mutated — this is the SAME generic oracle every guide fence already rides; a broken `[distributed]` fence would fail it exactly as any other section's would (verified green with the addition in place). |

### 3. Uncovered

- **The body's own call sites** (`jammi-ai`'s `fine_tune/worker.rs`/`spec.rs`, `jammi-server`'s
  `grpc/gang.rs`) that will actually call `record_assembly_outcome` and
  `materialize_or_reuse_training_set`, and the `serveable_world`/`[distributed]
  max_world_size` submit-time cross-check — explicitly out of scope for this unit (U4b,
  U5b-1b-i, U5a-2 build them); labelled DEFERRED per the brief, not a gap in this unit's own
  verb-level oracles.
- **A true OS-clock-skew injection** (actually setting the test process's wall clock away
  from the Postgres server's) is not attempted — no clock-mocking harness exists in this
  crate, and the lead's relayed correction explicitly reframes acceptance (b) as the
  structural "no second clock source" source-scan plus the Postgres-server-clock-authority
  behavioural test, which together are the achievable, hermetic proxy for the same property.
  Labelled, not silently assumed.
- **`assembly_backoff`'s behaviour beyond the clamp exponent (8)** — failures counts above
  8 all saturate at the same 300s ceiling; not separately probed at e.g. `failures = 1000`
  for overflow-freedom beyond the `checked_shl`/`expect` invariant already encoded (the
  `expect` message states why it cannot panic: the exponent is clamped to `< 32` before the
  shift). No test drives `failures` near `u32::MAX`; the `unwrap_or(u32::MAX)` fallback in
  `record_assembly_outcome`'s `u32::try_from(new_failures)` (i32 → u32) is likewise
  unexercised at that boundary (`new_failures` is bounded by `i32::MAX` well before it, via
  `saturating_add`, so the fallback arm is dead in practice but retained as a defensive
  total function).

### 4. Gates

| Command | Exit | Notes |
|---|---|---|
| `cargo fmt --all -- --check` | 0 | |
| `cargo clippy -p jammi-db --all-targets --features live-postgres-tests,test-hooks -- -D warnings` | 0 | |
| `cargo test -p jammi-db --features live-postgres-tests,test-hooks --test it -- assembly_outcome migrations docs_config_fences --test-threads=1` | 0 | 53 passed (20 `assembly_outcome`, 30 `migrations`, 3 `docs_config_fences`), 0 failed, `JAMMI_TEST_PG_URL=postgres://jammi@127.0.0.1:54329/jammi_test` set (live scratch Postgres 16, reachable on 54329) |
| `cargo test -p jammi-db --features live-postgres-tests,test-hooks --lib config::tests::distributed -- --test-threads=1` | 0 | 4 passed |
| Every mutation in §2 | red confirmed, then reverted | `git status --short` clean of mutation artifacts after each revert; `git diff --stat crates/jammi-db/src/catalog/lease.rs` empty (net zero touch, per the pressure-round correction) |

One test binary reused for the whole session (`CARGO_TARGET_DIR=<scratch>/targets/u5b1bii-db`,
one `--features live-postgres-tests,test-hooks` set throughout).

### 5. Commits

```
6dab678b feat(db): #500 U5b-1b-ii (db) — assembly cooldown/counter, [distributed] max_world_size, the CAS call-site helper
```
(one commit on `unit/u5b1bii-db`, off tip `856ec8dd`)


## 6. U5b-1b-ii — coordinator body

(built after wave A lands)

## 7. U5b-1b-iii

(built after §6)

## 8. U5b-2

(built after §7)

## 9. Pressure round (phase 1, executed at `856ec8dd` before the code landed) — REFINE, eight blocks folded

The design of the four wave-A units was attacked read-only against the base tree; the lead
opened every cited line before relaying. Each block below names the decision that folds it,
sent to the implementer while its unit was still being built; the commit closing each is
recorded at consolidation.

| # | Finding (cited) | Decision folded |
|---|---|---|
| 1 | U4b (b) "bit-for-bit summed adapter gradient vs W=1" is unachievable under f32 reassociation (`batch_bucket.rs:304-315` already documents 1e-5 for the padding half) and is satisfiable only by every rank computing the whole global gradient and skipping the reduce | (b) restated as a pre-registered ε whose discriminator is the W× hazard; bit-identity kept only for W=2-twice and resume-vs-uninterrupted; the "trainable op after the gather" mutation stays the hazard oracle |
| 2 | Zero ≠ absent for AdamW (`adamw.rs:245` skips an absent Var, steps a zero-gradient one; `optimizer.rs:626-633` documents absence as designed) | the reduce reduces the presence set (union across ranks) and restores "absent on every rank ⇒ absent" before `clip_and_step`; never a `world > 1` guard |
| 3 | Both epoch loops end on THIS rank's first empty chunk (`trainer.rs:1400-1404`, `1453-1456`); the optimizer-step boundary counts only non-diverged batches (`2881-2893`, `2903`) | epoch end = the global step count from the partition rule; a zero-row rank takes the step with a synthesized `[0, hidden]` tensor; the divergence decision is flag-reduced BEFORE accumulation so all ranks skip together (at W=1 `Noop` returns the rank's own flags — the W=1 window is unchanged); the one-rank divergence oracle uses the `cfg(test)` poke seam and says so |
| 4 | Hard-negative mining (`1303-1315`), GradCache (`1344-1362`) and the Precomputed arm (`1317-1343`) have no gather story; `gradcache.rs`/`hard_negative_miner.rs` in no scope | typed refusal at the spec admission edge for `world_size > 1` with mining, GradCache or a precomputed loader; their gather is a named follow-up; (b)/(d) oracles use the real loaders |
| 5 | The lifted `Descriptor` drops the round generation `Local` keeps outside it (`local.rs:174-193`); a commit-phase fault leaves one rank un-applied and the next fold is silently wrong | the round index is bound into the agreed descriptor; any commit-phase fault is fatal on every rank |
| 6 | `tests/distributed` already exists behind `live-distributed-tests` with a Postgres+MinIO harness and never gates a PR (`Cargo.toml:335-338`, `distributed.yml:24-26`) — (a),(c),(e),(f) would report green without executing | the Peer-vs-Local fold, deadline, commit-phase fault, descriptor disagreement, corrupted leaf and decode cap move to a hermetic two-process target in the ordinary test lane; fleet-dependent rows join the matrix by name |
| 7 | `RankAdmissionRow` carries no tenant (`jobs_repo.rs:212-250`, deleted in #566 round 2); `get_job` is tenant-filtered; admin scope forbidden — no route to the job's tenant for R2(b) | the admission row carries the row's OWN `jobs.tenant_id` (rebuilt carrier, its own oracle on both backends); the strict resolver pins to it |
| 8 | "Backend SQL clock on both dialects" contradicts the lease discipline: SQLite binds the application clock by design (`lease.rs:13-26`), and the two backends store different representations (`47-56`) | the cooldown reuses `lease_expired_clause`/`lease_deadline_expr`/`lease_now`/`lease_deadline` — no second clock source; the process-skew oracle is Postgres-only; a source-scan oracle pins "no literal clock outside the lease helpers" |

Advisories folded in the same messages: no rung exchange and no pin in U4b (the gather is dim 0
over pooled `[rows, hidden]`; `batch_bucket.rs:96-104`'s rationale rewritten; (c)'s ε covers
padding variance); the `agreement` slot is bound by U4b's canonical key-name digest; the
`BlockingCall` witness is a per-call trait argument owned by U5b-1b-i, threaded through U4b's
single collective helper at consolidation; an older resume bundle reads as no checkpoint (warn),
and W=1's byte-identity property is the artifact, not the bundle; the decode cap is per service
(both services on the peer listener, and the client recv side), pinned at exactly
`max_message_bytes`; classification logits are gathered by moving `classify()` into batch
construction (`TrainingBatch::Classification` reshaped, `data.rs` in scope); the
`[worker] world_size` → `local_ranks` rename touches the config field and accessors only and
the guide block is an ADD; stale premises corrected (U5a-1 froze no round messages;
`get_result_table_for_tenant` is deleted); the run_rank seam is pinned — U5a-2 owns admission,
the holder CAS and the hold loop; U5b-1b-i's round machinery lives in its own module and is wired
into the inbound arm at consolidation.

## 10. Gates (the merge path)

`bash ci/scripts/merge_path.sh` on the consolidated tip with `JAMMI_TEST_PG_URL` pointing at a
local PostgreSQL 16 in the CI lane's shape — run ONCE by the lead, never per implementer; the
phase-5 oracle dispatched after every other stage is green; only `docs/rigor/**` committed
after it.
