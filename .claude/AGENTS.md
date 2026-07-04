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

| Phase | Name | Agent(s) | Gate |
|---|---|---|---|
| 0 | Ground | `lead` | facts ledger; load the constitution invariants the brief crosses + `SELF-FAILURE-MODES.md` |
| 1 | Plan + pressure-test | `lead` + [`pressure-tester`](agents/pressure-tester.md) | a written plan; kill wrong designs *before code* |
| 2 | Contract | `lead` | per-domain scope + invariants + acceptance; embed CI's exact full gate |
| 3 | Implement | owning **domain agent** (worktree) or `general-purpose` | the change + the full gate run locally |
| 4 | Audit | [`adversarial-audit`](agents/adversarial-audit.md) + [`discipline-test-auditor`](agents/discipline-test-auditor.md) + [`citation-checker`](agents/citation-checker.md) | independent refutation; BLOCK on any Stands |
| 5 | Oracle | [`oracle`](agents/oracle.md) | hard-block on frozen-seam / boundary / lockstep / tenant-iso — **not overridable** |
| 6 | Verify-fix | [`fix-verifier`](agents/fix-verifier.md) | on a defect fix: red-green + non-finite control; the test must bite |
| 6.5 | Cookbook | `cookbook` | re-emit chapters whose goldens the diff could move; block Ship on divergence |
| 7 | Ship + publish | `lead` | push, PR, watch CI green, merge, watch post-merge green; lockstep publish |
| — | Learn | `retrospective` | periodic: cluster escapes → a *general* tightening PR (human-merged) |

## The roster

- **[`lead`](agents/lead.md)** — orchestrator; sole `Task` holder (no subagent spawns a
  subagent); owns the ledger, per-axis consensus, and git/PR/publish; **re-verifies every
  cited `path:line` and every "gate passed" claim**; never edits code on swarm work.
- **[`pressure-tester`](agents/pressure-tester.md)** — attacks the *plan* before code;
  treats every spec premise as a claim to reproduce.
- **[`adversarial-audit`](agents/adversarial-audit.md)** — refutes the *diff* across the
  correctness lenses; default BLOCK.
- **[`discipline-test-auditor`](agents/discipline-test-auditor.md)** — the engine-not-platform
  judgment lens; pairs with the mechanical `check_no_consumer_names.py`.
- **[`citation-checker`](agents/citation-checker.md)** — re-reads every cited location; catches
  fabricated or stale citations.
- **[`fix-verifier`](agents/fix-verifier.md)** — the phase-6 exit gate; red-green + non-finite
  control; requires a `closes_escape` id on a defect fix.
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
*discipline*, labeled as such. The swarm may *tighten* itself but never *weaken* itself: the
constitution and every gate definition are human-amend-only (`CONSTITUTION_TOUCHED` /
`SWARM_GATE_TOUCHED` fail closed → admin-merge).

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
