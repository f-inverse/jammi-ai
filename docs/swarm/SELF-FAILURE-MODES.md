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
  (`hooks/build-env-guard.sh`, advisory — stays advisory by design; a build-env hazard is a
  recoverable nudge, unlike F10 below, whose hook (`hooks/lead-gate-pre.sh`) is fail-closed
  because a norm is dodgeable by rewording and a denied dispatch is not).
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
- **Prevention.** Two layers. (1) Every verifier card REQUIRES a `class_enumeration`
  field in its verdict (the union of every sibling site its own sweep found for each
  BLOCK-severity finding, or an explicit `sweep_method: "none"` when it did not sweep) —
  the verifier enumerates the class it found, by default. (2) The mechanical gate:
  `hooks/lead-gate-{start,stop,pre}.sh` — a `PreToolUse` hook, fail-closed by design. Two rounds of adversarial audit against the first two designs both found the SAME
  shape of bug: a predicate over FREE TEXT (a `unit:`/`intent:` label in v1; site regexes,
  worktree/sha token scans, write-verb walks, and tag scans in v2) whose input domain is
  unbounded, so every fix moved the squeeze between jamming legitimate traffic and being
  dodged by a rewording (round 1: 9 findings, round 2: 9 MORE findings, each a live sibling
  of an r1 instance the patch had only closed one-for-one). v3 is a mechanism change, not a
  third patch: it stopped reading free text entirely and narrowed to ONE choke point — a
  second dispatch of the SAME verifier type, whole-token-bound (worktree/head_sha/
  unit_branch, strings the VERIFIER itself emitted, matched as whole tokens, never raw
  substrings) to an open BLOCK, is denied unless an
  ACCEPTED RELAY ARTIFACT exists — a structured file the lead writes explicitly
  (`.jammi/gate-state/<slug>.relay.<agent_type>.<block_ts>.json`, exact-string set inclusion
  of the verifier's own `class_enumeration`), never scanned from message prose. Under a
  usage-limit scope cut mid-round-3, `SendMessage` gating, implementer-dispatch binding, and
  the Bash backstop were DROPPED ENTIRELY (not log-only) rather than patched a third time —
  round 1 and 2 both proved free-text detection on those channels is undecidable without
  jamming legitimate freeze/status/stand-down/hygiene/advisory-fold traffic or ordinary
  compound Bash reads. Three documented, visible residuals remain (relaying by `SendMessage`,
  an unlabeled same-type verifier re-dispatch, `disableAllHooks`), each with the SAME runtime
  tell (the next verdict row with no accepted relay artifact between it and the prior BLOCK).
  See `.claude/agents/lead.md` "The class, not the instance" and
  `.claude/hooks/README.md`'s "mechanical vs. visible-only" paragraph for the exact statement.
- **Incident.** One session: ~43 audit/BLOCK rows across three units (`check_kernel_oracles.py`
  7 rounds, `check_perf_claims.py` 5 rounds, the pod-build substrate 6 rounds) against 3 lead
  probe rows total. When the lead finally probed, it found unnamed class members within
  minutes on every unit it checked (`pod_seed_target.sh:711`/`:770` at
  `ci/pod-build-substrate-r5@63bf905`, sibling of a named finding; a 745-instance `as %`/
  unit-wrapper class the audit itself later independently found; 3 of 4 unpinned CUDA
  kernels when the audit named one) — the class was findable the whole time; it was never
  looked for. A prior session showed the same ratio (11 BLOCK rows, 0 probes). TWO further
  rounds of this same failure shape hit the gate's OWN implementation: round-1 audit found 9
  BLOCK-severity findings (a label-keyed predicate or an operator applied outside its
  domain); round-2 audit, against the round-1 fix, found 9 MORE — every "closed" instance had
  a live sibling (a jam where a dodge had been closed, or vice versa) — because a free-text
  predicate's input domain is never actually bounded by patching the instances found so far.
  The mechanism meant to close F10 needed F10's own lesson (the class, not the instance)
  applied to ITSELF twice before it stopped trying to read free text at all.

---

## F11 · Sequence non-termination / mechanism regress

- **Trigger.** An audit-round sequence keeps finding defects — increasingly *inside the
  verification mechanisms the sequence itself built* — and no party owns a criterion for
  stopping; or an anti-escape mechanism ships and its own defects seed the next escapes.
- **Symptom.** Rounds continue past the point where marginal findings are latent-only; or a
  sequence is declared closed on prose ("chain closes") and the next rounds refute it; or ship
  happens by exhaustion rather than decision. The record: round-19 PASS "the chain closes"
  (ledger row 131), followed by three rounds of defects entirely inside the newly built claims
  oracle (rows 134–140) — the closed chain had not audited its own mechanism.
- **Root cause.** The regress is irreducible (Knight–Leveson correlated checker failures;
  equivalent-mutant undecidability; the obfuscated-arguments non-convergence result): every
  mechanism joins the set of artifacts needing verification, so "keep going until clean" never
  terminates, and "stop when green" ships latent mechanism defects. Termination is not
  discoverable from the data; it is a human-owned acceptance of enumerated residuals — the
  consensus terminal move of every mature assurance field (ALARP, GSN assurance-deficit
  registers, UL 4600, seL4/CompCert trusted-assumption lists, SRE error budgets).
- **Prevention.** Constitution T1/T2. The sequence terminates only in operator acceptance of an
  assurance-deficit register (`.jammi/registers/`, `check_ship_register.py` G1–G8,
  `REGISTER_TOUCHED` admin-merge); liveness is auditor-owned, fail-closed default-live,
  mechanism findings classified by the UNSOUND-NOW test; acceptance is metered by a reopen
  budget the retrospective settles via `registered_class_hit`. The portfolio grows only through
  the ALARP table + decorrelation fixtures (T2). Honest limit, on the record: mid-sequence
  build-vs-design conformance is covered ONLY by the lead's per-delta disposition discipline
  (row 140) — the terminus catches only what the closing auditor independently re-finds; the
  un-mechanized per-wave gate is a registered residual standing on ALARP grounds, and its
  absence is exactly esc-066.
- **Incident.** Unit 63: 8 of the first 9 mechanical adversarial-audit rounds BLOCKed (the sole
  PASS, mech round 6, carried 2 advisories); the defect stream migrated from live artifact
  defects to latent mechanism-model defects (row 138: "ZERO live false numbers anywhere … both
  blocks are the coverage MODEL itself"); the first liveness-classified round (mech round 10,
  row 145) still found 3 live blocks in fresh wave code. esc-066 (`open` at seeding): REFINE
  deltas had no merge-time conformance check — the lead merged greenness, not spec-conformance,
  and rounds 20–21 were the one-round-late catches.

---

## Using this catalog

At phase 0 the lead names which of F1–F11 the brief is exposed to and pins the gate that
catches each (F1→phase 1, F2→phase 4, F3/F4→phase 7, F5→build-env hook, F6→citation-checker,
F7→fix-verifier, F8→pressure-test, F9→audit + honesty gate, F10→`hooks/lead-gate-pre.sh`, the
fail-closed dispatch/relay gate, F11→constitution T1/T2, `check_ship_register.py`, the
operator-accepted assurance-deficit register). A defect that slips every gate and is caught
later is logged to the escape ledger (`.jammi/escapes.jsonl`) and clustered by the retrospective
loop into a new gate — so each failure mode compounds into infrastructure instead of
re-teaching itself every session.
