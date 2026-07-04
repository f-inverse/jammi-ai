# The engine swarm — operating model

This is the human-facing overview of the agentic swarm that runs Jammi's rigor chain
(`plan → pressure-test → implement → independent adversarial audit → CI → merge`) as
committed infrastructure instead of prose the lead has to remember. The full design is
[`docs/plans/53-agentic-swarm/ARCHITECTURE.md`](../docs/plans/53-agentic-swarm/ARCHITECTURE.md);
the lesson→mechanism map is [`LESSONS.md`](../docs/plans/53-agentic-swarm/LESSONS.md); the
invariant index the swarm cites by ID is [`docs/swarm/CONSTITUTION.md`](../docs/swarm/CONSTITUTION.md).

## The phase machine

The **lead** (the main loop) runs a fixed pipeline. Each phase clears a named gate before
the next begins — skip a step and you ship the plausible-wrong version on green.

Two front doors, symmetric on their RED oracle: a **defect** is triaged (0.7) into a
`symptom_spec` that drives `fix-verifier`; a **feature** is scoped (0.5) and its `acceptance`
criteria drive `acceptance-verifier`. A **question** mutates nothing → no phase machine.

| Phase | Name | Agent(s) | Gate |
|---|---|---|---|
| 0 | Ground | `lead` | facts ledger; load the constitution invariants the brief crosses + `SELF-FAILURE-MODES.md` |
| 0.5 | Scope (all mutating work) | [`gap-analyzer`](agents/gap-analyzer.md) | enumerate what's asked, flag ambiguities, name the invariants crossed; `clear / ambiguous / invariant-crossing` |
| 0.7 | Triage (defect only) | [`issue-triage`](agents/issue-triage.md) | classify the RAW issue; **valid-defect emits the `symptom_spec` as an `open` escape row** (the RED for phase 6); misconception → halt; constitution-challenge → escalate |
| 1 | Plan + pressure-test | `lead` + [`pressure-tester`](agents/pressure-tester.md) | a written plan; kill wrong designs *before code* |
| 2 | Contract | `lead` | per-domain scope + invariants + acceptance (the *feature*'s RED oracle); embed CI's exact full gate |
| 3 | Implement | owning **domain agent** (worktree) or `general-purpose` | the change + the full gate run locally |
| 4 | Audit | [`adversarial-audit`](agents/adversarial-audit.md) + [`discipline-test-auditor`](agents/discipline-test-auditor.md) + [`citation-checker`](agents/citation-checker.md) | independent refutation; BLOCK on any Stands |
| 5 | Oracle | [`oracle`](agents/oracle.md) | hard-block on frozen-seam / boundary / lockstep / tenant-iso — **not overridable** |
| 6 | Verify red→green | **defect:** [`fix-verifier`](agents/fix-verifier.md) · **feature:** [`acceptance-verifier`](agents/acceptance-verifier.md) | the test asserts the RED oracle, was RED at base, GREEN on branch |
| 6.5 | Cookbook | `cookbook` | re-emit chapters whose goldens the diff could move; block Ship on divergence |
| 7 | Ship + publish | `lead` | push, PR, watch CI green, merge, watch post-merge green; lockstep publish |
| — | Learn + hygiene | [`retrospective`](agents/retrospective.md) | periodic: cluster escapes → **one** general tightening PR (human-merged); own the escape-ledger lifecycle (promote → cluster → archive to `.jammi/escapes-archive.jsonl`, never delete) |

## The roster

- **[`lead`](agents/lead.md)** — orchestrator; sole `Task` holder (no subagent spawns a
  subagent); owns the ledger, per-axis consensus, and git/PR/publish; **re-verifies every
  cited `path:line` and every "gate passed" claim**; never edits code on swarm work.
- **[`gap-analyzer`](agents/gap-analyzer.md)** — the scope front door (phase 0.5, all
  mutating work); enumerates exactly what's asked, flags ambiguities, names the invariants
  crossed (`clear / ambiguous / invariant-crossing`).
- **[`issue-triage`](agents/issue-triage.md)** — the defect front door (phase 0.7); classifies
  the raw issue text and, for a valid-defect, emits the `symptom_spec` as the `open` escape row
  that seeds red-green.
- **[`pressure-tester`](agents/pressure-tester.md)** — attacks the *plan* before code;
  treats every spec premise as a claim to reproduce.
- **[`adversarial-audit`](agents/adversarial-audit.md)** — refutes the *diff* across the
  correctness lenses; default BLOCK.
- **[`discipline-test-auditor`](agents/discipline-test-auditor.md)** — the engine-not-platform
  judgment lens; pairs with the mechanical `check_no_consumer_names.py`.
- **[`citation-checker`](agents/citation-checker.md)** — re-reads every cited location; catches
  fabricated or stale citations.
- **[`fix-verifier`](agents/fix-verifier.md)** — the *defect*-path phase-6 exit gate; red-green
  + non-finite control; requires a `closes_escape` id on a defect fix.
- **[`acceptance-verifier`](agents/acceptance-verifier.md)** — the *feature*-path phase-6 exit
  gate, symmetric to `fix-verifier`; proves the phase-2 acceptance test was RED at base and GREEN
  on the branch, asserting the acceptance criterion (not an implementation detail).
- **[`retrospective`](agents/retrospective.md)** — out-of-band Learn + escape-ledger hygiene;
  clusters escapes into a principle and opens **one** general tightening PR (human-merged), and
  owns the ledger lifecycle (promote → cluster → archive, never delete).
- **[`oracle`](agents/oracle.md)** — the hard-blocks (dep-direction, lockstep, append-only
  migrations, embedded⇄remote parity, per-RPC tenant isolation, frozen surface); **not
  consensus-overridable**.
- **Domain agents** (per crate, ARCHITECTURE §6) — the write-owners, worktree-isolated, each
  carrying its crate's invariants. **Doc-currency** (`build-graph`, `graph-navigator`,
  `doc-updater`, `doc-parity`) and **cookbook** complete the roster.

## Consensus is per-axis, never a vote

A phase advances only when **no axis Stands as BLOCK** — one unrefuted block-severity
finding on any single axis blocks it; a "majority PASS" never overrides it. An `oracle`
HARD_BLOCK is not overridable by anything. Complementary lenses do not substitute: a design
PASS is not a correctness PASS is not a boundary PASS is not an evidence PASS.

## Honest enforcement

The only **hard** teeth are (a) native per-agent `tools:` — a verifier has no `Edit`, so it
*cannot* write — and (b) fail-closed CI gates, **and a CI gate only blocks once a human adds
it to branch-protection required-checks.** Committing a workflow ≠ committing enforcement.
Everything else — routing nudges, the build-env hook, file-scope conventions — is
*discipline*, labeled as such. The swarm may *propose* to tighten itself via a human-merged
PR but never *weakens* itself: the constitution and every gate definition are human-amend-only
(`CONSTITUTION_TOUCHED` / `SWARM_GATE_TOUCHED` fail closed → admin-merge; anti-Goodhart is
human-in-the-loop, not a direction classifier).

## The generalization principle (the overriding rule)

Every verifier card is a **principle-level rubric, not a lookup table of past bugs**
(ARCHITECTURE §2.8). A swarm that memorizes past bugs has high precision on the seen and
~zero recall on the unseen — the opposite of fool-proof. So each card phrases its checklist
as a *general principle applied to code the agent has never seen*, carries historical
incidents only as one-line calibration, and closes with the standing instruction:

> Apply these principles to the diff in front of you; a novel-but-analogous smell is in
> scope; default to BLOCK when uncertain. Do not limit yourself to the illustrative instances.

Instances are *evidence* (frozen in `.jammi/escapes.jsonl` as regressions); principles are
*mechanism*. This is also the audit lens on the swarm's own files: any card that reads as
instance-memorization is sent back.
