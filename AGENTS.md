# Working in this repo — the agentic swarm

**Mutating work runs through the swarm** — the rigor chain as a committed phase
machine, not prose to remember. If you are an agent about to change this repo, this
is your entry point (`CLAUDE.md` is not tracked here, so this file is the portable
one).

- **Operating model:** [`.claude/AGENTS.md`](.claude/AGENTS.md) — the phase machine,
  the roster, honest-enforcement, and the generalization principle.
- **The phase machine you execute:** [`.claude/agents/lead.md`](.claude/agents/lead.md).
  You, the main agent, **are the `lead`** — sole `Task` holder; you dispatch the
  verifiers and domain agents, own the ledger and consensus, and drive git/PR/publish.

**How work enters:**
- A **question** mutates nothing → skip the machine; answer directly.
- **Every** mutating change is **scoped** first — phase 0.5, `gap-analyzer`: the ask,
  its ambiguities, the invariants it crosses (feature *and* fix alike).
- A **defect** then *also* passes **triage** — phase 0.7, `issue-triage` (defect-only):
  validity + the `symptom_spec` that seeds red-green (`fix-verifier`).
- A **feature** proceeds from scope to a plan; its phase-2 acceptance criteria drive
  `acceptance-verifier`.

So the two doors aren't symmetric on *entry* (0.5 is universal); they're symmetric on
their **RED oracle** — a defect's is a triaged `symptom_spec`, a feature's is its
acceptance criteria.

Design rationale lives in `docs/plans/53-agentic-swarm/ARCHITECTURE.md`. The
enforcement is honest: the hard teeth are native per-agent `tools:` + fail-closed CI
gates (`.github/workflows/swarm.yml`); everything else is labeled discipline.
