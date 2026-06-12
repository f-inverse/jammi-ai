# Agentic Engineering Playbook — what the 0.25→0.26.4 marathon taught

How to do rigorous agentic engineering in this codebase. Read it alongside
`CLAUDE.md` (the rules), `PHILOSOPHY.md` (engine-not-platform), and the hardening
roadmap's §7 (the execution substrate: build env, gate commands, infra walls).

This is **evidence-graded**: every lesson carries the real incident that taught it,
because an agent — like a human — should trust a rule with a failure attached, not an
abstraction. The incidents are from one continuous run: it set out to write a consumer
cookbook against `jammi_ai` v0.25.0, and authoring real measured workflows turned the
cookbook into a bug-discovery engine that drove five patch releases (0.26.0 → 0.26.4)
and dozens of fixes.

The one-sentence thesis: **the rigor chain is not bureaucracy — it is the mechanism
that converts a discovered anomaly into a *correct* fix instead of a plausible-but-wrong
one.** Skip a step and you ship the plausible-wrong version on green CI.

---

## 1. The rigor chain is a discovery mechanism

`plan → pressure-test → implement → independent adversarial audit → CI → merge`.
Two steps are load-bearing and the ones an agent is most tempted to skip:

**Pressure-test (before any code) catches *wrong designs*.** The most expensive bugs
aren't in the implementation — they're in the plan. Pressure-testing the plan against
first principles, *before* writing code, repeatedly killed designs that would have
compiled, passed tests, and been wrong:
- A target-standardization design was going to rescale the *loss*. Pressure-test caught
  that **Adam normalizes each parameter by its own gradient RMS, so a raw parameter
  travels ~lr/step regardless of loss scale** — loss-rescaling cannot move it. The fix
  had to standardize the *data/representation space* (z-space), not the loss. This was
  caught before a line was written; had it been "implemented to spec," every test of the
  wrong design would have passed.
- An over-engineered schema design and two wrong *spec premises* (see §4) were also
  killed at pressure-test.

**Independent adversarial audit catches what the implementer + green CI miss.** A fresh
agent told "try to *refute* this; default to BLOCK if uncertain," reading the diff
against the contract, finds what the author (invested in their solution) and the test
suite (only checks what someone thought to assert) cannot:
- It found a **tenant-scope-restore leak** in a merged-quality PR: `__exit__` collapsed
  "never entered" and "entered-while-unscoped" into the same `unbind` arm, so a stray
  exit cleared a live tenant scope — a data-scope leak, on green CI.
- It found a remote `publish_topic` that silently diverged from the embedded path on
  multi-chunk tables (the server rejected what the client sent), invisible to the
  single-chunk happy-path test.
- It found an abstraction smell — a fetch cap that bounded the *total* instead of the
  thing that actually blew up — that would have silently capped a large `k`.

**So: never merge on green CI alone.** Green CI is necessary, not sufficient (§2).

---

## 2. Green CI is necessary, not sufficient

CI checks what someone thought to assert. It does not check the claim's *premise*, the
unexercised path, or the silent divergence. Every audit catch above was on a green PR.
Treat the green check as the floor, then ask the audit questions CI cannot:
- Does the negative/control arm actually reproduce the bug, or can it pass vacuously?
  (A negative control asserting `mean far OR σ floored` flaked because the un-standardized
  path sometimes **diverged to NaN**, and `NaN > 100 || NaN < 0.05` is `false || false` —
  a divergence read as a fit. The control must count non-finite as collapse.)
- Does the remote surface match the embedded one *byte-for-byte*, not just "both respond"?
- Is the number in the prose actually computed live and asserted, or transcribed?
- Does the change hold at the boundary/degenerate input, not just the typical one?

---

## 3. Delegation discipline (the coordinator pattern)

The pattern that scaled this run: a **coordinator** owns plan / pressure-test / gating
and keeps its own context light; heavy implementation is delegated to **fresh agents in
isolated git worktrees**. It works, but only with these guardrails:

**Verify the agent actually finished — never trust "done."** Delegated agents end
mid-task and report completion anyway:
- One implementer did the full implementation, ran the gate, then **ended without
  committing** — its final message was a confused "I'll wait for notifications." The work
  was uncommitted in the worktree. The coordinator had to detect this (the PR never
  appeared), verify the diff itself, run the gate, and commit/push/PR.
- Two emit agents backgrounded a long job and stopped; the coordinator took over the wait.

Before trusting a delegated result, confirm the *artifact*: a pushed commit, an open PR,
and the **full** gate actually run (not the agent's word). If it's not on the remote, it
didn't happen.

**Run the FULL gate, not a subset — and that's the coordinator's spec, not the agent's
fault.** Agents kept passing local checks and reddening CI because the delegation brief
listed a subset (build/test/clippy) and omitted the `cargo doc -D warnings` Docs lane and
the Postgres lane. Put the *entire* gate (roadmap §7.2) in the delegation brief.

**Don't let parallel agents contend on one build dir.** A stuck agent spawned its own
`cargo test` runs into the *same* `CARGO_TARGET_DIR` the coordinator's gate was using →
cargo's build lock serialized them → the gate crawled. Give every agent/worktree a
**unique** `CARGO_TARGET_DIR=/mnt/sagemaker-nvme/ct-<unique>`, and if an agent goes rogue,
`TaskStop` it and kill its stray `cargo`/`rustc` before they fight your build.

**Keep the loop alive.** The cookbook→engine loop never stopped surfacing real findings —
even the CI run for a *fix* surfaced a latent flaky test. Expect each fix to beget the
next finding; budget for it rather than treating "done" as terminal.

---

## 4. The spec is not gospel

"The spec said so" is a dodge, not a defense — you read the spec, so catching its
wrongness is your job. This run disproved spec premises empirically more than once:
- A spec asserted `transaction()` hangs on a current-thread runtime and prescribed a fix.
  The implementer **reproduced the claim and found it false** — shipped a doc-only change
  instead of the prescribed (unnecessary) code.
- A sweep spec carried wrong premises about how a subsystem failed; the implementer's
  pressure-test corrected them before coding.

When a spec tells you *why* something is broken, reproduce the *why* before you build the
fix. A fix built on a wrong premise is a plausible-wrong change that passes its own tests.

---

## 5. Numerical and test gotchas

- **Standardize in data space, not the loss (the Adam lesson, §1).** Any trainable head
  on a high-offset / low-variance / large-magnitude target needs a standardization +
  domain contract. This bug was found *twice* (fine-tune head, then the amortized context
  predictor) because the first fix covered one subsystem and the cookbook re-emit exposed
  the second. Add a high-offset oracle for *every* trainable head.
- **Domain-validity is the unifying invariant.** The sweep's root cause across negative
  LR, unbounded mean, inflated degree, and global identity was one thing: *the engine
  computed confidently past its valid input domain*. Validate / clamp / normalize at every
  numeric and catalog input edge; make it a review checklist item, not a per-bug patch.
- **Flaky negative controls.** A control that asserts "the bad path fails" can pass
  vacuously when the bad path fails in an *unmodelled* way (NaN). Make "failed to fit"
  include non-finite, or seed the run deterministically — don't let a borderline assert
  rerun-flake your gate.
- **Cap the thing that blows up, not the aggregate.** A bound on `k + excluded + 1` as a
  whole silently capped `k`; the correct bound was on the *excluded headroom* alone,
  leaving the `k+1` base uncapped. When you add a limit, bound the unbounded term, not the
  sum it lives in.

---

## 6. Operational traps (this host)

Detail and recipes live in roadmap §7 and the memory files; the agent-facing summary:
- **This is not the `CLAUDE.md` devcontainer.** Use the §7.1 build env; **do not override
  `RUSTC_WRAPPER`/`RUSTFLAGS`** (it changes the sccache key and re-misses the cache — once
  turned a gate into ~100 min of redundant compiles).
- **The NVMe build volume fills.** Unique target dirs, remove merged-PR worktrees, watch
  `df -h /mnt/sagemaker-nvme`. A worktree checked out under a full disk silently dropped a
  tracked fixture and produced a *spurious* test failure — treat remote CI (full checkout)
  as authoritative.
- **`maturin`/embedded `jammi_ai` cross-worktree shadowing.** `import jammi_ai` can resolve
  to a *different* worktree's install; pin `PYTHONPATH` to the worktree's `python/` and
  confirm `jammi_ai.__file__` + that its `_native.abi3.so` is the fresh build.
- **The server's JSON log writes nothing to a non-TTY stdout** — prove GPU use with
  `nvidia-smi` compute-apps, not the log.
- **Recalled memory reflects when it was written.** Verify any file/flag/path a memory
  names still exists before acting on it.

---

## 7. Honesty is a release gate

A manufactured benchmark, a mismeasured fix, or an overclaimed guarantee is a *blocker*,
not a nuance — the entire reason the core is trustworthy is that nothing untrue shipped.
This run repeatedly chose the honest negative:
- The keystone chapter **refused a manufactured crux**: importance-weighted conformal did
  *not* restore coverage under a location/orthogonal shift, so the chapter teaches that
  honestly instead of staging a fake win.
- A "restored coverage" number turned out to be an APS-convention artifact; aligning the
  local computation to the engine's admission rule made `local == engine` and dissolved
  the illusion.
- A tenancy chapter's "hard-zero leak prevented by source separation" was a **loader
  pre-filter artifact, not engine isolation**; the audit caught the overclaim and the
  chapter was reframed to the true two-layer model.

If the honest result is "it doesn't work / doesn't help," that is the finding. Ship it.

---

## The meta-pattern

Authoring a faithful consumer (the cookbook) is the most powerful integration test you can
write — it exercises real call sequences with real data and real expected numbers, which
is exactly where engines compute confidently past their valid domain. The rigor chain is
what makes the resulting discoveries *land correctly*: pressure-test kills the wrong fix
before it's built, independent audit kills the plausible-wrong fix before it merges, and
honesty kills the fake win before it ships. The loop plus the chain, run continuously, is
the method — not any single fix.
