# `.claude/evals/` — the swarm's held-out test set

This is how the swarm **proves its verifiers actually fire** and, more deeply, how it
proves they **generalize** rather than memorize (`ARCHITECTURE.md §9a`, §2.8). It mirrors
the engine's own test tiers.

## The three tiers

1. **static** (`static/*`) — the fail-closed CI gates (`check_swarm_bijection.py`,
   `check_constitution_anchors.py`, `check_doc_parity.py`, the `*_TOUCHED` guards).
   Deterministic, $0, runs on every PR. See `static/README.md`.
2. **golden set** (`golden/*`) — each past **escape** in `.jammi/escapes.jsonl` becomes a
   case: given the situation, does the verifier that should catch it *fire*? These back the
   citation discipline (below). Runnable-shaped scaffold — one YAML per case; note no runner
   executes them yet.
3. **judge / Monte-Carlo** (local / on-demand, not in the PR gate) — LLM-judge scoring of
   verifier verdicts and stochastic mutation sweeps. Runs when a human asks; never gates a
   PR, because a nondeterministic judge must not block merge.

## The rule (citation discipline → golden eval)

> An escape is only `closed` when a **golden eval proves the catch.**

A diff that transitions an escape's `status` to `closed`/`eval_added` **should add, in the
same diff, a golden eval that cites that escape id** (`ARCHITECTURE.md §9`). A green test is
not enough — the golden must demonstrate the *verifier fires on the situation*. This is
**discipline today**, enforced by the `fix-verifier` card's review: there is no committed
script that greps the transition, and no runner executes these goldens. A mechanical
`check_escape_citations.py` grep gate (transition-only) is a **candidate tightening**
(`ARCHITECTURE.md §12`, G-e), not a wired gate.

## Generalization by mutation (not replay)

A golden case does **not** replay the logged bug — a verifier that only catches the exact
recorded repro has high precision on the seen and ~zero recall on the unseen, the opposite
of fool-proof (§2.8). So every case carries a **`mutation`**: a NOVEL perturbation of the
*principle* the verifier must *also* catch. The verifier passes the case only if it fires
on both the seeded situation **and** the mutation — i.e. it reasons from the principle, not
a lookup of the past signature. This is the same red-green primitive `fix-verifier` uses:
the mutation is the held-out mutant; a verifier that misses it is memorizing.

## Case shape (see `golden/*.yaml`)

Each golden case names:
- `escape` — the real id from `.jammi/escapes.jsonl` it derives from.
- `principle` — the LESSONS.md family/principle it tests (the thing that must generalize).
- `verifier` — the gate agent (or CI gate) that MUST fire on this situation.
- `situation` — the seeded scenario (the principle instantiated).
- `expect` — the required verdict (BLOCK / hard-block / red-green verified …).
- `mutation` — a novel perturbation of the *principle* the verifier must also catch, with
  its own `expect`. Memorizing the seeded signature must not pass the mutation.

Keep the set small and deep: few principle-level cases over many instance replays
(`§2.8` bias–variance). New escapes add a case here as they are closed.
