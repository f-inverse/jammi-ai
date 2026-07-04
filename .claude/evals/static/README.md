# Static evals (tier 1) — the CI gates

Tier 1 is not a separate harness: it **is** the repo's fail-closed CI gates, run on every
PR, deterministic and $0. There is nothing to author here beyond this pointer — a static
eval is a gate script that already exists under `ci/scripts/` and is wired in a workflow.

The gates that serve as tier-1 evals (`ARCHITECTURE.md §7`, `LESSONS.md` CI-gate lines):

| Gate script | Asserts (the invariant, over a class — not a bug signature) |
|---|---|
| `ci/scripts/check_swarm_bijection.py`      | every tracked source path under `crates/` maps to exactly one domain owner and coverage is total; an unowned path is a **P0**. |
| `ci/scripts/check_constitution_anchors.py` | every constitution `code anchor` still resolves (typed: `rust_symbol` / `gate_script` referenced in `swarm.yml` / `doc_heading`); every boundary invariant anchors to a live wired gate. |
| `ci/scripts/check_doc_parity.py`           | a documented enum enumeration ⇄ the code enum it mirrors (set-equality + exception arms). SHIPPED (#245). |
| `check_dep_direction.py` / `check_cookbook_one_way.sh` | references point one way only (engine names no consumer). |
| `CONSTITUTION_TOUCHED` / `SWARM_GATE_TOUCHED` (in `swarm.yml`) | the constitution and gate definitions are human-amend-only, fail-closed (anti-Goodhart). |

**Why these are evals, not just checks:** each asserts a *property over a class* (all
drift, all unowned paths, all stale anchors), so it generalizes to inputs never seen —
the same standard the golden tier holds the LLM verifiers to. A grep for one known-bad
string would not qualify.

These are the deterministic floor. The `golden/` tier proves the *LLM* verifiers (which
reason from principles, not greps) fire; the judge/Monte-Carlo tier runs local/on-demand.
