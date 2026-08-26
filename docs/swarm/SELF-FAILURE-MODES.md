# Self-Failure Modes — the lead's phase-0 pre-flight catalog

The recurring ways an agentic run goes wrong *in this repo*. The lead loads this at
phase 0 (Ground) before dispatching any work, and re-checks the relevant entries at the
gate a failure mode targets. Every entry is tied to a **real incident** from
`docs/plans/51-marathon-learnings/` — an agent trusts a rule with a failure attached, not
an abstraction. Each is **trigger → symptom → root cause → prevention**; keep them tight.

The one-sentence spine: **green CI is the floor, not the ceiling** — the two steps a
hurried run skips (pressure-test, independent audit) are the two that catch the bugs CI
cannot (`AGENTIC-PLAYBOOK.md` §1). Everything below is a specific way that spine breaks.

---

## F1 · Wrong design shipped on green (pressure-test skip)

- **Trigger.** A plausible plan that will compile and pass its own tests is approved and
  implemented without attacking it against first principles first.
- **Symptom.** The change is green everywhere and still wrong — the bug lives in the
  *design*, not the code, so no test asserts against it.
- **Root cause.** The most expensive bugs are in the plan; a wrong design's tests all pass
  because they test the wrong thing.
- **Prevention.** Run phase 1 pressure-test (`pressure-tester`) *before* any code; a
  design that survives only with a named premise is dead.
- **Incident.** A plan to fix a collapsing regression head by **rescaling the loss** was
  killed at pressure-test: Adam divides each parameter's step by its own gradient RMS, so
  a raw parameter moves ~`lr`/step *regardless of loss scale* — loss-rescaling is a no-op.
  The fix had to standardize the *data space*. Caught before a line was written.
  (`CASE-STUDIES.md` §1; constitution K3.)

## F2 · Plausible-wrong fix past CI (audit skip)

- **Trigger.** A merged-quality, green PR goes to merge without a fresh agent trying to
  *refute* the diff.
- **Symptom.** A correct-looking fix that CI accepts but that breaks a state the assertions
  forgot to construct.
- **Root cause.** The author is invested in their solution; the suite only checks what
  someone thought to assert. Neither sees the unexercised state.
- **Prevention.** Phase 4 independent adversarial audit (`adversarial-audit`), told "refute
  this; default BLOCK if uncertain," reading `git diff <base>...<head>` on live state.
- **Incident.** On a green PR, `__exit__` on the new `tenant_scope` context manager used
  `.take().flatten()`, collapsing "never entered" and "entered-while-unscoped" into one
  `unbind` arm — a stray exit **cleared a live tenant scope**, a data-scope leak on the
  very feature meant to make scoping safe. Audit found it; the fix matches the full
  `Option<Option<TenantId>>`. (`CASE-STUDIES.md` §3b.)

## F3 · "Done" claimed but uncommitted / gate not fully run (verify-the-artifact)

- **Trigger.** A delegated agent reports completion.
- **Symptom.** No PR appears, or CI reddens on a lane the agent never ran; the work is
  uncommitted in a worktree.
- **Root cause.** Delegated agents end mid-task and narrate success anyway; "done" is a
  claim, not a result.
- **Prevention.** Gate on the *artifact*: a pushed commit, an open PR, and the **full**
  gate actually run — re-verify, don't accept the word. If it's not on the remote, it
  didn't happen.
- **Incident.** An implementer did the full implementation, ran the gate, then **ended
  without committing** — final message a confused "I'll wait for notifications." Two emit
  agents backgrounded a long job and stopped. The coordinator had to detect it (the PR
  never appeared), verify the diff, run the gate, and push. (`AGENTIC-PLAYBOOK.md` §3.)

## F4 · Full gate vs subset

- **Trigger.** The delegation brief lists a convenient subset (build/test/clippy) of the
  gate.
- **Symptom.** Agents pass local checks and redden CI on the omitted lane every time.
- **Root cause.** The subset is the *coordinator's* spec error, not the agent's fault — an
  agent runs the brief it was given.
- **Prevention.** Put the **entire** gate in every delegation brief (build, test, clippy,
  fmt, the `cargo doc -D warnings` Docs lane, the Postgres lane — roadmap §7.2), and run
  every touched crate's it-suite when a change adds a wire RPC (the tenant-isolation oracle
  lives there).
- **Incident.** Agents repeatedly passed local checks and reddened CI because the brief
  omitted the `cargo doc -D warnings` Docs lane and the Postgres lane. (`AGENTIC-PLAYBOOK.md`
  §3.)

## F5 · Build-dir contention + never override `RUSTFLAGS`/`RUSTC_WRAPPER` (the ~100-min cache-miss)

- **Trigger.** Parallel agents share one `CARGO_TARGET_DIR`, or a brief exports
  `RUSTFLAGS` / `RUSTC_WRAPPER`.
- **Symptom.** The gate crawls: cargo's build lock serialises contending `cargo test` runs,
  or a changed sccache key re-misses the whole cache into a full recompile.
- **Root cause.** One target dir → one build lock the runs fight over; overriding
  `RUSTFLAGS`/`RUSTC_WRAPPER` changes the sccache key so nothing hits cache (`CLAUDE.md`:
  sccache is always active, `RUSTFLAGS` overrides `config.toml`).
- **Prevention.** Give every agent/worktree a **unique** `CARGO_TARGET_DIR=/mnt/…/ct-<uid>`;
  **never** override `RUSTC_WRAPPER`/`RUSTFLAGS`; if an agent goes rogue, `TaskStop` it and
  kill its stray `cargo`/`rustc`. This is the highest-leverage build-env hook
  (`hooks/build-env-guard.sh`, advisory).
- **Incident.** A stuck agent spawned `cargo test` into the coordinator's target dir → the
  build lock serialised them → the gate crawled; separately, an `RUSTC_WRAPPER`/`RUSTFLAGS`
  override once turned a gate into **~100 min** of redundant compiles.
  (`AGENTIC-PLAYBOOK.md` §3, §6.)

## F6 · Stale-worktree audit / fabricated citations

- **Trigger.** A verifier audits inside a worktree, or cites `path:line` it did not re-read.
- **Symptom.** An audit blesses stale state, or a citation points at a line that doesn't say
  what's claimed (or no longer exists).
- **Root cause.** Verifiers must audit *live* state at an explicit diff, not a private
  worktree snapshot; recalled memory reflects when it was written, not now.
- **Prevention.** Verifiers are read-only on the main checkout (never a worktree) and audit
  at `git diff <base>...<head>`; `citation-checker` re-reads every cited `path:line`; verify
  any file/flag/path a memory names still exists before acting.
- **Incident.** A worktree checked out under a full NVMe disk **silently dropped a tracked
  fixture** and produced a *spurious* test failure — treat remote CI (full checkout) as
  authoritative. Recalled memory naming stale paths is a standing trap.
  (`AGENTIC-PLAYBOOK.md` §6.)

## F7 · Non-finite negative control flaking vacuously

- **Trigger.** A "the bug still reproduces" control asserts a bounded predicate
  (`mean far OR σ floored`).
- **Symptom.** The control flakes red on an *unrelated* PR; suspicion falls on a regression
  that isn't there.
- **Root cause.** The bad path fails in an *unmodelled* way (diverges to NaN), and
  `NaN > 100 || NaN < 0.05` is `false || false` — a divergence (the strongest collapse) reads
  as a *successful fit*.
- **Prevention.** `fix-verifier` treats a non-finite served value as collapse; make "failed
  to fit" include the failure modes you didn't enumerate, or seed deterministically. When a
  reproduce-control flakes, suspect the *predicate* before a regression.
- **Incident.** Exactly this control flaked on a sibling crate's PR off the same base; the
  fix made non-finite count as collapse. (`CASE-STUDIES.md` §4; `AGENTIC-PLAYBOOK.md` §2.)

## F8 · Spec not gospel

- **Trigger.** A spec asserts *why* something is broken and prescribes a fix.
- **Symptom.** An unnecessary (or wrong) change is built on a premise nobody reproduced —
  and it passes its own tests.
- **Root cause.** "The spec said so" is a dodge; you read the spec, so catching its
  wrongness is your job. A fix on a wrong premise is a plausible-wrong change.
- **Prevention.** Reproduce the *why* before building the fix; a doc-only change is the
  correct output when the claim is false.
- **Incident.** A spec claimed `transaction()` hangs on a current-thread runtime and
  prescribed a fix; the implementer **reproduced the claim and found it false**, and shipped
  a doc-only change instead of the unnecessary code. (`AGENTIC-PLAYBOOK.md` §4.)

## F9 · Honesty as a release gate

- **Trigger.** A chapter/benchmark is *expected* to show a win; the measured result doesn't.
- **Symptom.** A staged crux, a transcribed-not-computed number, or an overclaimed guarantee
  — a confident-wrong result that reads as success.
- **Root cause.** A measured result can be real *and* mean something other than you claim; a
  manufactured win is a blocker, not a nuance — the core is trustworthy only because nothing
  untrue shipped.
- **Prevention.** Numbers are computed live and asserted against a golden; verify the
  *mechanism* produces the number, not just that the number appears; if the honest answer is
  "it doesn't help," that is the finding — ship it.
- **Incident.** A tenancy chapter's "hard-zero leak prevented by per-tenant sources" was a
  **loader pre-filter artifact, not engine isolation** (tenant A could still `SELECT` tenant
  B's source by name); the audit caught the overclaim and the chapter was reframed to the
  true two-layer model. The keystone likewise **refused a manufactured crux** when weighted
  conformal did not restore coverage under a location shift. (`AGENTIC-PLAYBOOK.md` §7;
  `CASE-STUDIES.md` §3a; constitution B5, K2.)

## F10 · Lead passivity — relaying verifier findings round-by-round

- **Trigger.** A verifier (`adversarial-audit`/`fix-verifier`/etc.) returns BLOCK on a unit.
  The lead has a standing instruction ("on every BLOCK, the auditor's findings are a SAMPLE
  from a class — probe the class yourself, in parallel with the fix, before dispatching") in
  three places: memory, `CLAUDE.md`, and this agent's own card.
- **Symptom.** The lead forwards the verifier's finding list to the implementer verbatim,
  round after round, without probing for sibling/class members the verifier's own sweep did
  not name. The instruction sat in context and lost at the decision point every time.
- **Root cause.** A norm in context competes with everything else in context (the diff, the
  verdict, the phase machine, the user's last message) and loses — prose the lead "should"
  remember is not infrastructure. This is the general failure the whole swarm exists to fix
  (`ARCHITECTURE.md §1`), caught here in its most expensive instance: the audit-round loop.
- **Prevention.** Two parts, today. (1) Every verifier card now REQUIRES a `class_enumeration`
  field in its verdict (the union of every sibling site its own sweep found for each
  BLOCK-severity finding, or an explicit `sweep_method: "none"` when it did not sweep) — the
  verifier no longer hands the lead one instance and calls it done; it enumerates the class it
  found, by default. (2) The lead's own brief to an implementer states the class in one
  sentence and cites every member it is briefing against (`lead.md` "The class, not the
  instance"), so the omission is visible in the artifact the lead writes, not just in its own
  head. **There is no mechanical gate yet** — nothing currently blocks a dispatch that skips
  this; a hook-based gate that makes the round-by-round relay hard to satisfy by rewording a
  prompt was designed and built across three audit rounds but is not yet merged (see #402,
  held in draft on outstanding review findings). Until #402 lands, this is discipline, not
  enforcement — named honestly, not overclaimed.
- **Incident.** One session: ~43 audit/BLOCK rows across three units (`check_kernel_oracles.py`
  7 rounds, `check_perf_claims.py` 5 rounds, the pod-build substrate 6 rounds) against 3 lead
  probe rows total. When the lead finally probed, it found unnamed class members within
  minutes on every unit it checked (a sibling of a named finding in the same file; a
  745-instance unit-wrapper class the audit itself later independently found; 3 of 4 unpinned
  CUDA kernels when the audit named one) — the class was findable the whole time; it was never
  looked for. A prior session showed the same ratio (11 BLOCK rows, 0 probes).

---

## Using this catalog

At phase 0 the lead names which of F1–F10 the brief is exposed to and pins the gate that
catches each (F1→phase 1, F2→phase 4, F3/F4→phase 7, F5→build-env hook, F6→citation-checker,
F7→fix-verifier, F8→pressure-test, F9→audit + honesty gate, F10→no mechanical gate yet; see
#402 — today, the verifier-card `class_enumeration` requirement and the lead's own audit-brief
practice are discipline, not enforcement). A defect that slips every gate and is caught later
is logged to the escape ledger (`.jammi/escapes.jsonl`) and clustered by the retrospective loop
into a new gate — so each failure mode compounds into infrastructure instead of re-teaching
itself every session.
