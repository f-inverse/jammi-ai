# CONTRACT — fix/500-release-claim-spin: RELEASE admits no new claims from the instant its phase flips

**Contract of record.** slug: `fix_500-release-claim-spin` · base: `main` @ `fe5ac560` · this file is
the committed mechanism contract `ci/scripts/check_rigor_record.py` requires under
`docs/rigor/contracts/**` (its check 3) before this unit's rigor record at
`docs/rigor/fix_500-release-claim-spin.jsonl` (the lead's export, landed separately) satisfies check 1/2.
Citations tagged **(at fe5ac560)** below were read against the commit this branch was cut from; this
branch's own later commits (`c7c31141`, `c75452b0`, `d471541a`, `5922787b`, `eea07e5d`, `a8c33000`,
`04433abc`) moved lines in the files those citations name. The checker's cost-floor pass
(`check_path_line_citations`) only verifies a cited file has *at least* that many lines at HEAD — never
that the cited content still matches — so a tag here is a provenance note, not a claim the line still
reads the same; re-open each tagged citation by symbol before relying on its line number for anything
beyond "this file is long enough."

Owner: **ai-core**. Worktree `scratchpad/wt-RS`, branch `fix/500-release-claim-spin`, cut from `main` @ fe5ac560.
Own PR, merged BEFORE PR-B (PR-B rebases). Design round: DONE (pressure-tester on U2a's M3, 2026-09-13 — the
mechanism was reproduced on main, not argued; its `mechanism_contract` is this file's source). Found by
U2a's fix round: `jobs_shutdown::release_and_stop_report_matches_release_job_leases_on_the_pair_that_actually_differs`
fails on main 1/16 under load with `attempts == releases == N` in a 22–38 ms window.

## The defect (reproduced; citations on main fe5ac560 ≈ 9db8d395 for worker.rs)
`EmbeddedWorker::release_and_stop` sets `WorkerPhase::Releasing` at
`crates/jammi-ai/src/fine_tune/worker.rs:~2592` (2a, at fe5ac560) and requests the loop's stop only at
`:~2604` (2d, at fe5ac560 — same file); between them run 2b `release_job_holds(self.heartbeat)` (bounded
by a FULL heartbeat) and 2c's two sweeps. The loop's top-of-iteration gate (`:~808`, at fe5ac560) reads
only `stop_requested()`; `register_job_hold_or_release`'s self-release arm (`:~498`, at fe5ac560) reads
only `phase()`. So for the whole [2a, 2d] window the loop free-spins: reclaim (arm 1a requeues the
NULL-lease row, `crates/jammi-db/src/catalog/jobs_repo.rs:~1632` (at fe5ac560); `attempts − releases`
stays 0 so the cap never trips) → `claim_next` (`attempts+1`) → hold registration sees `Releasing` →
`release_job_lease` (`releases+1`) → repeat, with no sleep in the `Some(record)` arm (`:~838-852`, at
fe5ac560). The spin's own catalog traffic delays 2b, lengthening the window. Production heartbeat is
10 s: every graceful shutdown/rollout spins over the live queue for up to ~10 s, inflating
`attempts`/`releases` on real rows. `begin_drain` (`:~2449-2455`, at fe5ac560) sets phase AND stop
together — the in-file precedent. Class sweep (done): exactly four phase/stop setters; this is the
only one that sets a claim-inhibiting phase without the stop.

## Property (quantified over EVERY `release_and_stop` call, every worker, any load, any lease/heartbeat)
Between 2a and the loop's exit the loop executes ZERO `Catalog::claim_next` calls; on return, every
`jobs` row this instance touched has `attempts` equal to its value at 2a and `releases` equal to the
number of leases the instance actually held at 2a (1 in the escape's scenario). The genuine single race —
a claim whose COMMIT preceded 2a — stays covered by the Releasing arm and its existing oracle
`release_with_the_loop_paused_in_the_claim_to_hold_prologue_self_releases`
(`crates/jammi-ai/tests/it/jobs_shutdown.rs:~673`, at fe5ac560, (attempts, releases) == (1, 1)).

## Mechanism
The loop's claim gate is ONE predicate over the phase read at the loop top: `Running` admits claims;
`Draining`/`Releasing` do not (`:~808`, at fe5ac560, worker.rs — breaks on `phase() != Running` as well as
on `stop_requested()`), AND `release_and_stop` requests the stop at 2a beside the phase flip, as
`begin_drain` does. `stop` stays the wakeup. Every doc line that describes the [2a, 2d] window
(`:~2460-2485`, at fe5ac560, worker.rs; OPS D6 prose in `docs/`) is re-read and corrected where it
implies claims continue until 2d.

## Oracle (deterministic — no load, no timing)
Drive the loop to re-reach the loop top while `phase == Releasing` and stop is not yet set (a
`Rendezvous`/test-hook at 2a, the idiom the sibling tests use; or hold 2b's keeper pass) and assert
`(attempts, releases) == (1, 1)` on the in-flight row and that `claim_next` was called ZERO times after 2a
(a `test-hooks` counter). RED at main fe5ac560 (reproduce the spin: N cycles), GREEN after. Per-determinant:
revert the loop-gate change alone → RED; revert the 2a stop alone → RED (say which test dies for each).
The sibling oracles at `crates/jammi-ai/tests/it/jobs_shutdown.rs:~645` (0,0), `:~673` (1,1), `:~759`
(timeout arm) — all at fe5ac560 — stay green — name any that dies. The escape test itself stays as is
(it is a correct oracle of the invariant).

## Gates
`cargo test -p jammi-ai` (whole crate) + ≥ 3 full `--test it` runs (name every failing test per run; the
two pre-named jobs_cancel tests excluded), `cargo test -p jammi-server --test it` (the server's drain/
release tests), workspace clippy, gated clippy lane, rustdoc, fmt. Commit without trailers; never push.
Report: common `<eval-verdict>` with the RED spin counts at main and the per-determinant table.

## Stop rule (pre-committed): if the closing audit BLOCKs on the gate predicate, the fix reduces to the
`begin_drain` shape alone (stop requested at 2a, no gate change) with the property re-measured; if that
blocks, the unit ships the oracle marked `#[ignore]` with the defect filed — never a weakened assertion.

## Design-pass folds (pressure-tester on fe5ac560, REFINE — binding; supersede the sections above where they differ)
- **Citations on fe5ac560:** 2a `crates/jammi-ai/src/fine_tune/worker.rs:2500` (at fe5ac560), 2d `:2512`
  (at fe5ac560), `begin_drain` `:2356-2363` (at fe5ac560), loop gate `:807` (at fe5ac560), hold arm `:497`
  (at fe5ac560), the single production `claim_next` initiation `:826` (at fe5ac560), `WorkerPhase`
  `:239-243` (at fe5ac560).
- **Property, corrected quantifier:** no `Catalog::claim_next` call is INITIATED at or after 2a; at most the ONE claim
  already in flight at 2a commits and is matched by exactly one release (the sibling oracles at
  `crates/jammi-ai/tests/it/jobs_shutdown.rs:592` (at fe5ac560) and `:673` (at fe5ac560) end (1,1) and stay
  as they are); and no `jobs` row OTHER than that one has its `attempts`, `releases`,
  `status` or `claimed_by` changed by this instance after 2a — quantified over ALL rows (the spin steals the head of the
  global priority queue: `claim_next` has no `claimed_by` predicate,
  `crates/jammi-db/src/catalog/jobs_repo.rs:744-753` (at fe5ac560); reclaim 1a is fleet-wide).
- **Two properties, not one per-determinant pair:** P1 (safety, phase-keyed): a `claim_next` is initiated only if
  `phase() == Running` was read at the top of that same iteration. P2 (wakeup/latency, stop-keyed): every phase setter
  requests the stop in the same statement pair, so exit latency after a flip is bounded by the in-flight job, not by
  `idle_poll` — discharged by a check over the two setters that exist (`:2357`, `:2500` — both at fe5ac560),
  never by a RED mutation.
- **Oracle for P1 (gate-direct, deterministic):** a `#[cfg(feature = "test-hooks")]` phase setter on `WorkerShared`
  (field and `set_phase` are private, `:286`/`:382` — at fe5ac560; the it-tests are external); set `Releasing` WITHOUT the stop, run
  `run_until` over a session with TWO queued rows, assert: `LoopState::Stopped`, a test-hooks `claim_next` counter == 0,
  BOTH rows still `(attempts, releases) == (0, 0)` / `queued` / `claimed_by NULL`. RED at fe5ac560 is deterministic in
  the liveness direction (the loop never exits; exit wait times out; `attempts > 0`).
- **Not a licence to abort earlier:** 2e (`:2516-2567`, at fe5ac560) is untouched; state it in the doc.
- **Doc edit sites (by name):** `docs/plans/68-compute-tier-substrate/units/OPS-COMPUTE-TIER-OPERABILITY.md:18`
  (at fe5ac560; D6's order puts `stop` in position 4) and `:116` (at fe5ac560);
  `crates/jammi-ai/src/fine_tune/worker.rs:2444-2493` (at fe5ac560; the 2a/2d bullets); `:2483-2489`
  (at fe5ac560; P-2F wording — the transition is now usually an already-terminal read). OPS doc edits
  are docs-ci's: report the exact sentences.
- **F7 pre-committed disposition:** if `release_and_stop_leaves_running_with_null_lease_and_no_new_bundle`
  (`crates/jammi-ai/tests/it/jobs_shutdown.rs:399-421`, at fe5ac560) dies, park the trainer to pin
  `in_flight == 1` as the compute sibling at `:514` (at fe5ac560) does — never weaken its assertion.
- **Abstraction option (advisory, decide and state):** either commit the setter pairing as an enumerated invariant
  (P2's check) or carry the phase on its own `watch` selected in the idle sleep. Safety is unaffected either way.
- **Sibling, filed not fixed here:** `InferenceSession::release_job_leases`
  (`crates/jammi-ai/src/session.rs:350-371`, at fe5ac560) hands leases back with no claim inhibition;
  every caller is prose-guarded "only when there is no worker" (`crates/jammi-server/src/runtime.rs:1069`
  (at fe5ac560), `:1237` (at fe5ac560), `crates/jammi-python/src/database.rs:335` (at fe5ac560)).
  Recorded for U5b-2 / OPS.

## Fix round 1 (citation BLOCK at d471541a) — two citation folds, no mechanism change
- ai-core: `crates/jammi-ai/tests/it/jobs_shutdown.rs:833-834` — the doc comment on
  `every_phase_setter_pairs_the_stop_in_the_same_statement_group` cites `begin_drain (:2357)` and 2a
  `(:2500)` — both the fe5ac560 anchors from the design-pass fold above (at fe5ac560), stale by this
  commit; re-anchor to the HEAD lines (`:2383-2390`, `:2547`, both at c75452b0) or cite by symbol
  without a line. Doc comment only.
- docs-ci: `docs/plans/68-compute-tier-substrate/units/OPS-COMPUTE-TIER-OPERABILITY.md` D6 row cites
  `crates/jammi-ai/src/fine_tune/worker.rs:620` (at fe5ac560) for "aborting first would drop the hold —
  runs on the dropped future"; find the real site at HEAD (the 2e cooperative-vs-abort decision / the
  hold's `Drop`) and re-anchor, or drop the number and cite the symbol.
Serialized AFTER the running audit/acceptance/oracle report (doc-only; no re-run of those beyond citation-checker).

## Fix round 1 — REVISED (audit BLOCK at c75452b0 + citation BLOCK at d471541a): one round, both owners
**Ruling.** The audit executed the falsification: the gate is read at the loop top, then `reclaim_expired_jobs` is
awaited (`:842-844`, at c75452b0), then `claim_next` (`:853`, at c75452b0) — a RELEASE that lands during the reclaim still lets THIS iteration
initiate one claim, self-released by the hold arm. The fold's wide property ("no `claim_next` INITIATED at or after
2a … over ALL rows") was over-claimed; the narrow P1 holds. This is a LOCAL correction of where the existing predicate
is read, not a new mechanism (no design round): the property becomes **P1'**: `claim_next` is called only after a
read of `stop_requested() || phase() != Running` that returned false with NO await between that read and the call;
the residual — a claim whose catalog round trip is already in flight when 2a runs — is stated in both doc sentences
as exactly that, and is what the hold arm's self-release (:509) exists for.

### ai-core — `worker.rs`, `tests/it/jobs_shutdown.rs`
1. Re-read the same predicate immediately before `claim_next` (`:853`, at c75452b0), after `reclaim_expired_jobs`; `break` on it.
   No await between the read and the call (`record_claim_next` is sync and stays after the read). Both gate sites
   share ONE private fn (`fn admits_claim(&self) -> bool`) so the predicate cannot drift between them.
2. Doc sentences: `crates/jammi-ai/src/fine_tune/worker.rs:2478-2480` ("a claim that nonetheless lands (it began before this instant)") → the
   residual as stated in P1'; the `run_until` doc names both read sites.
3. Oracle (RED at c75452b0 — the audit's probe): park the loop INSIDE `reclaim_expired_jobs` via the existing
   `loop_test_hooks` rendezvous (or a `test-hooks` park at that await), run 2a, release the park, assert
   `claim_next_calls` delta == 0 and the queued row untouched (`status`, `attempts`, `releases`, `claimed_by`).
4. Counter positive control: at `jobs_shutdown.rs:~955` (at c75452b0; the loop has claimed and parked)
   assert `claim_next_calls(id) >= 1` before the release — so a dead counter (M4) dies. Landed at HEAD
   as `release_timeout_arm_leaves_the_honest_row_recovered_by_arm_1a`'s own positive control
   (`crates/jammi-ai/tests/it/jobs_shutdown.rs:1025-1030`), not at the estimated line.
5. `jobs_shutdown.rs:~838`: delete the false sentence ("cannot be made RED through observable behaviour") — M2
   kills `release_before_the_loop_reaches_claim_next_leaves_the_row_untouched` (`:671`, at c75452b0; `:660`
   at cd807a5b) as well; say that instead.
6. Citations `:833-834` → `:2383-2390` / `:2547` (or cite by symbol; all four at c75452b0); stale "2a–2h"
   labels at `:1079`, `:1112` (at c75452b0).
7. Gates: `cargo test -p jammi-ai --features test-hooks --test it -- jobs_shutdown` (all), the whole `--test it`
   once (name the load-sensitive `jobs_cancel` test if it fails and re-run it alone), clippy (workspace + gated
   lanes), fmt, rustdoc. Mutations: remove the second read → the new oracle dies (name it); M4 → the positive
   control dies.
### docs-ci — `OPS-COMPUTE-TIER-OPERABILITY.md`
- `:108` 2a sentence → P1' residual wording (from the code, after ai-core commits — poll `git log`); `:231` "2a–2h"
  label; D6 row's `crates/jammi-ai/src/fine_tune/worker.rs:620` (at fe5ac560) citation re-anchored to the
  real site at HEAD or cited by symbol. Doc gates.
The P2 setter reshape (`enter_phase`) is recorded, not taken (mechanism; contract-scoped advisory). Commit by
pathspec; no trailers; never push.
**Stop rule (pre-committed, binding):** if closing audit #2 BLOCKs on the gate placement or P1', the second read is
reverted, the property is reduced to the fold's narrow P1 ("read at the top of that iteration") and both doc
sentences state the reclaim-window residual verbatim; no further gate reads are added.

## Salvage from PR #524 (closed as superseded, 2026-09-13) — folded into this unit
- **db:** `crates/jammi-db/tests/it/lease_keeper.rs` gains the catalog-level idempotency oracle from
  `fix/release-stop-ordering` (`git diff main...fix/release-stop-ordering -- crates/jammi-db/tests/it/lease_keeper.rs`):
  the three RELEASE statements in production order (2b `LeaseKeeper::release_job_holds` on the keeper's own
  connection, then 2c/2g `Catalog::release_jobs_claimed_by` on the pool connection) against a REAL backend with a real
  elapsed heartbeat between claim and release; `jobs.releases` lands at exactly 1. Both backends (sqlite and the
  `::postgres` arm, which runs in CI under `--features live-postgres-tests` with `JAMMI_REQUIRE_PG: "1"`
  (`.github/workflows/ci.yml:1296-1300`) and locally under `JAMMI_TEST_PG_URL`; the lead's executed runs are recorded
  in the session ledger, not here). Re-anchor its doc citations to THIS branch's
  worker.rs (2a at :2547 → after the fix round, re-read) — the branch's own numbers are stale here. RED: revert to
  the pre-#524 shape is not required (an oracle over an invariant main already satisfies is a PIN — say so); mutate
  the test's release path once (drop the second statement) to show it bites.
- **docs-ci:** `.jammi/escapes.jsonl` gains `esc-111` (symptom spec from the #524 branch, `evidence_ref` → this unit's
  contract and record; `status: eval_added` naming `release_gate_refuses_every_claim_when_phase_flips_without_a_stop` and
  the AfterReclaim oracle). Lands with the rigor record commit at PR time.
  **Correction (docs-ci, at HEAD `04433abc`):** `fix/release-stop-ordering` (PR #524, closed head
  `a00c62a07cb7ebb901bc26430af68725d35606fa`) never touched `.jammi/escapes.jsonl` — `git diff
  fe5ac560 a00c62a0...` on that path is empty, and no `esc-111` row (or any row past `esc-110`) exists on
  that branch at any commit in its history. The PR's own fix commit
  (`4ce58b0e fix(ai,db): release_and_stop requests stop before releasing any hold, closing a double-release
  race`) documents the identical defect in its commit message but was never filed as an escape row. The
  `esc-111` row this unit adds is therefore built from that commit message and this contract's own defect
  section, not copied verbatim from a pre-existing ledger row — flagged for the lead.
- **later, own unit:** the cancel-request-watcher observed-event wait (a00c62a0) after this unit merges.
