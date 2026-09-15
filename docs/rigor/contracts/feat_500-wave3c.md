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

## 5. U5b-1b-ii — database slice

(folded at consolidation)

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
