# Jammi agentic swarm — architecture

The complete design, rooted in Jammi's engineering philosophy. `README.md` is the
capability-1 narrative; `LESSONS.md` is the lesson→mechanism map; `CONSTITUTION.md`
and `SELF-FAILURE-MODES.md` are the authored foundations. Every choice below is
justified against a Jammi principle — Lace is prior art we adapt, not a template.

This revision incorporates the pre-build pressure-test (which was REFINE with two
BLOCK findings) and the generalization mandate (§2.8). Where the first draft was
wrong, this says so plainly.

## 1. Thesis

Jammi already *practices* a rigorous method — the rigor chain
(`plan → pressure-test → implement → independent adversarial audit → CI → merge`),
"never merge on green CI alone," "a delegated agent's *done* is a claim to verify."
It lives as **prose the lead must remember**. The swarm mechanizes it into
committed, CI-enforced infrastructure so it executes by construction, and it is
**evidence-graded**: every mechanism traces to a real past incident (`LESSONS.md`).

## 2. First principles (the swarm obeys Jammi's own rules)

1. **Engine names no consumer — including the swarm's own files.** No engine agent
   card, gate, doc, fixture, or **data file** (e.g. `.jammi/escapes.jsonl`) may name
   a consumer. The pressure-test caught the first draft seeding a consumer-named
   incident into the tracked engine ledger — a boundary violation in the swarm's own
   foundation. Engine escapes seed from engine incidents only; consumer-specific
   lessons are carried as *generic patterns*, never by name.
2. **Domains are crates, owned by exhaustive path manifests.** Cargo partitions the
   workspace; each domain owns a 100%-coverage glob manifest of its crate's tree
   (not a marquee-subsystem sketch — that leaves modules unowned and the bijection
   unpassable). Shared declaration files (`lib.rs`, `Cargo.toml`, `error.rs`) are a
   `shared`/lead-owned class, exempt from the domain mutex.
3. **Enforcement is honest, and a gate is soft until wired.** The only hard teeth
   are (a) native per-agent `tools:`, (b) fail-closed CI gates — **and a CI gate
   only blocks once it is added to branch-protection required-checks, a manual admin
   step that is not in any PR** — and (c) one named exception: the lead-proactivity
   gate (`hooks/lead-gate-{start,stop,pre}.sh`, §7) is a fail-closed `PreToolUse`
   hook, armed by default, whose decider DENIES on internal error rather than
   allowing (F10, `SELF-FAILURE-MODES.md`). Committing `swarm.yml` ≠ committing
   enforcement. The design labels every "gate" as advisory-until-required.
   Everything else (routing nudges, file-scope allowlists, the *other* hooks) is
   discipline, labeled as such — a single named hard hook does not relabel the
   *default* posture, it is the documented exception to it.
4. **DRY across the swarm's own knowledge.** A fact lives in one place. The
   constitution does not re-copy `philosophy.md`; it indexes citable invariants.
   Across repos, the repo-agnostic shape is not mirrored (mirroring drifts with no
   gate) — it is a **swarm-kit the consumer repo vendors one-way** (§3).
5. **The cookbook loop is the crown jewel — and it gets a blocking phase.** It is
   Jammi's real bug-discovery engine (drove five releases). Phase 6.5 re-emits the
   chapters whose goldens a diff could move and **blocks Ship on any divergence**.
   A crown jewel with no phase is decorative; the first draft's was.
6. **No band-aids in the swarm.** Its own files pass `CLAUDE.md`'s self-check.
7. **Self-evolution is human-gated; anti-Goodhart via human-in-the-loop, not a
   classifier.** A Markdown constitution has no mechanical tighten-vs-weaken
   discriminator, so *all* constitution and gate-definition edits are human-amend
   (`CONSTITUTION_TOUCHED`/`SWARM_GATE_TOUCHED` make them red → admin-merge). The
   only autonomous surface is: append to `escapes.jsonl`, and have `retrospective`
   open a *tightening PR* a human merges. The first draft's "may tighten
   autonomously" was false — adding a gate touches a gate-definition → blocked.
8. **Generalize, don't overfit — encode the principle, not the instance.** The
   deepest principle. A swarm that memorizes past bugs has high precision on the
   seen and ~zero recall on the unseen — the opposite of fool-proof. So:
   - **Instances are evidence; principles are mechanism.** Each lesson is stored
     twice — the instance frozen in the escape ledger (regression), the *principle*
     in a rubric/invariant (generalization). Mechanisms name the class, never the
     instance.
   - **Verifiers reason from principles, not lookup tables** — this is why the audit
     layer is LLM reviewers, not greps. Cards are principle-first, with history as
     illustration only, and instruct the agent to apply the principle to novel code
     and default-BLOCK on a novel-but-analogous smell.
   - **Gates assert invariants over a class, never bug-signatures.** `doc-parity`
     is the standard (property: doc-list == code-enum → catches all drift, not just
     one variant). A grep for a known-bad string does not ship. Denylists exist only
     as cheap backstops behind a generalizing judgment layer.
   - **The retrospective abstracts, not accumulates** — it clusters escapes into a
     common principle and proposes ONE general gate, not N narrow checks. A pile of
     hyper-specific gates is the overfitting smell.
   - **Generalization is validated by mutation** — a verifier is trusted only if it
     catches *unseen* perturbations (mutation-adequacy), not replays of the log.
   - **Bias–variance for the swarm:** prefer few deep principle-mechanisms over many
     shallow instance-checks; fool-proofness = broad rubrics + default-block, not
     enumeration.
   This is also the audit lens on the swarm's *own* files: any card/gate reading as
   instance-memorization is sent back.

## 3. Topology — two swarms, one boundary, a vendored kit

Jammi is two repos with a one-way dependency (enterprise consumes the engine;
engine depends on no consumer). A single spanning swarm would violate that — an
engine-repo constitution enumerating consumer invariants would itself name a
consumer. So two swarms:
- **Engine swarm** (`jammi-ai/.claude/`) — constitution = engine invariants;
  oracle enforces names-no-consumer + the frozen seams.
- **Consumer swarm** (`jammi-enterprise/.claude/`) — its own constitution
  (consumer-of-engine rules), may reference the engine, adds the **triangular**
  engine↔enterprise↔cookbook loop and its own escape ledger.

The repo-agnostic shape (lead, phase machine, verifier cards that name no
invariant, hook scripts, the eval harness runner) is factored into a **swarm-kit**
that the consumer repo **vendors one-way with a drift-check gate** — the exact
pattern the enterprise repo already uses to vendor the engine's audit proto. Each
repo then owns only its constitution + domain manifests. Not mirrored (mirroring
drifts ungated); vendored (on-philosophy: references point one way).

## 4. The rigor chain as a phase machine

The lead (the main loop) runs a fixed pipeline; each phase names the agent(s) it
dispatches and the gate it clears.

| Phase | Name | Agent(s) | Gate |
|---|---|---|---|
| 0 | Ground | lead | facts ledger; load the constitution invariants the brief crosses + `SELF-FAILURE-MODES.md` |
| 0.5 | Scope (all work) | `gap-analyzer` | enumerate exactly what's asked, flag ambiguities, name which invariants it crosses; verdict `clear / ambiguous / invariant-crossing` |
| 0.7 | Triage (defect only) | `issue-triage` | ingest the RAW issue text; classify `valid-defect / misconception / constitution-challenge / enhancement`. **valid-defect EMITS the `symptom_spec` (intended/observable/control) as an `open` escape row — this is the RED that seeds red-green.** misconception → halt (+ optional non-bug golden); constitution-challenge → escalate to a human |
| 1 | Plan + pressure-test | lead + `pressure-tester` | a written plan; pressure-test kills wrong designs *before code* |
| 2 | Contract | lead | per-domain: files_in_scope, invariants_to_preserve, **acceptance criteria (the *feature*'s RED oracle)**; embeds CI's EXACT full gate (per-step `$?`, no pipe-masking) |
| 3 | Implement | owning **domain agent** (worktree, unique `CARGO_TARGET_DIR`) or `general-purpose` on an existing branch | the change + the full gate run locally |
| 4 | Audit | `adversarial-audit` + `discipline-test-auditor` + `citation-checker` | independent refutation; BLOCK on any Stands |
| 5 | Oracle | `oracle` | hard-block on frozen-seam / boundary / lockstep violation (not overridable) |
| 6 | Verify red→green | **defect:** `fix-verifier` (the test asserts the triaged `symptom_spec.observable`; revert fix → RED → GREEN; non-finite control; cite `closes_escape`). **feature:** `acceptance-verifier` (the phase-2 acceptance test was RED at the base commit, GREEN on the branch; asserts the acceptance criterion, not an implementation detail) | the test must have been RED and now bites |
| 6.5 | Cookbook | `cookbook` (`cookbook-emit`) | re-emit chapters whose goldens the diff could move; **block Ship on divergence** (route back as an engine bug) |
| 7 | Ship + publish | lead | push, PR, watch CI green, merge, watch post-merge green; own the lockstep crates.io+PyPI publish |
| — | Learn + hygiene (out-of-band) | `retrospective` | periodic, not per-unit: cluster escapes into a *principle* → **one** general tightening PR (human-merged); own escape-ledger **lifecycle** — promote `open→eval_added→closed`, cluster (never N narrow gates), and **archive** long-green `closed` escapes to `escapes-archive.jsonl` (never delete — the row is its golden's oracle) |

Two front doors, symmetric on their RED oracle: a **defect** is triaged into a
`symptom_spec` (0.7) that drives `fix-verifier`; a **feature** is scoped (0.5) and
its `acceptance` criteria (phase 2) drive `acceptance-verifier`. A **question**
mutates nothing → no phase machine. The lead re-verifies every cited `path:line`
and every "gate passed" claim — a subagent result is evidence to audit, never a
fact to accept.

## 5. Roles

**Lead** — main loop; sole `Task` holder (no subagent spawns a subagent); owns
ledger, phases, consensus, git/PR/publish; never edits on swarm work.

**Verifiers** (read-only, main checkout, JSON verdict, audit an explicit
`git diff <base>...<head>`). Each card is a **principle-level rubric** (§2.8), with
its checklist lifted from `LESSONS.md` §Per-mechanism:
- `gap-analyzer` — the scope front door (phase 0.5, feature *and* fix): re-reads the
  brief, enumerates exactly what's asked, flags ambiguities, names which constitution
  invariants it crosses. Verdict `clear / ambiguous / invariant-crossing`.
- `issue-triage` — the defect front door (phase 0.7): ingests the RAW issue text
  (the lead runs `gh issue view`; the agent has no Bash), classifies validity, and
  for a valid-defect **emits the `symptom_spec` as an `open` escape row** — the RED
  seed that `fix-verifier` later verifies faithfulness against.
- `acceptance-verifier` — the *feature*-path exit gate (phase 6), symmetric to
  `fix-verifier`: proves the phase-2 acceptance test was RED at the base commit and
  GREEN on the branch, and asserts the acceptance criterion (not an implementation
  detail). `fix-verifier` is the *defect*-path exit; both share the red-green primitive.
- `retrospective` — out-of-band Learn + hygiene: clusters escapes into a principle
  and opens **one** general tightening PR (human-merged; it *proposes*, it does not
  self-modify a gate — `SWARM_GATE_TOUCHED` blocks that), and owns the escape-ledger
  lifecycle (promote → cluster → archive, never delete).
- `pressure-tester` — attacks the plan (spec-as-claim-to-reproduce; principle
  violations: wrong abstraction, band-aid shape, non-atomic split).
- `adversarial-audit` — refutes the diff (states the happy path never constructs;
  remote⇄embedded parity; cap-the-unbounded-term; domain-validity at every edge;
  nullable-clearable → 3-state; verify-the-mechanism/honesty; `CLAUDE.md`
  self-check + dodges). Default BLOCK.
- `discipline-test-auditor` — the engine-not-platform lens, split into a mechanical
  part (§7: consumer-name denylist + governance-verb-stem tripwire) and an LLM
  judgment part (the discipline test; governance-is-platform-vs-mechanism-is-open-core).
- `citation-checker` — re-reads every cited `path:line`.
- `fix-verifier` — the exit gate (red-green; a test that doesn't fail pre-fix is
  tautological → BLOCK; a control that passes on NaN/±inf is vacuous → BLOCK).
- `oracle` — hard-blocks: dep-direction, cookbook-one-way, migration monotonicity,
  lockstep version, embedded⇄remote parity, tenant-iso (every wire RPC gets a
  cross-tenant-denial case), and the frozen wire surface (§ H4 below).

**Domain agents** (write-owners; worktree + unique `CARGO_TARGET_DIR`; domain
mutex; exhaustive path manifest; `<eval-verdict>` hand-off) — one per crate (§6),
each carrying its crate-specific invariants from `LESSONS.md`.

**Doc-currency** (shipped, #245): `build-graph`, `graph-navigator`, `doc-updater`,
`doc-parity`.

**Cookbook loop** — phase 6.5 above; measured-not-asserted, read-the-cache/assert-
goldens, teach-the-honest-negative.

## 6. Domain decomposition (engine) — exhaustive, per-crate

One write-owner per crate; ownership is a **total partition** checked by
`ci/scripts/check_swarm_bijection.py`, which walks the filesystem (not the
crate-level `build-graph`, which cannot express within-crate ownership) and asserts
every tracked source path under `crates/` maps to exactly one owner **and** that
coverage is total (an unowned path is a P0). Manifests are exhaustive globs.

| Agent | Owns (crate) |
|---|---|
| `db` | `jammi-db` (all 18 modules; shared `lib.rs`/`Cargo.toml`/`error.rs` = shared-class) |
| `ai-core` | `jammi-ai` (all 15 modules) |
| `numerics` | `jammi-numerics`, `jammi-encoders`, `jammi-lora`, `jammi-kernels` |
| `wire-server` | `jammi-wire`, `jammi-admin`, `jammi-client`, `jammi-server` |
| `cli` | `jammi-cli` |
| `python` | `jammi-python` |
| `bench` | `jammi-bench`, `jammi-test-utils` |
| `cookbook` | `cookbook/` |
| `docs-ci` | `docs/`, `ci/`, `.github/`, `.claude/` (the shared-class + swarm files) |

Per-crate ownership (not the first draft's within-`jammi-db`/`jammi-ai` split)
makes the bijection trivially total and avoids the shared-file co-edit collision.
Finer within-crate splits are a later refinement that must ship an explicit module
manifest, not a prose sketch.

## 7. Enforcement (honest inventory)

**Hard (gates, fail-closed; advisory until required-in-branch-protection):**
- Existing: `check_dep_direction.py`, `check_cookbook_one_way.sh`, `gen_dep_dag.py`
  freshness (`dep-dag.yml`), `check_doc_parity.py` (`doc-parity.yml`).
- New: `check_swarm_bijection.py` (total ownership); `check_constitution_anchors.py`
  (typed anchors, §8); `check_no_consumer_names.py` (denylist of known consumer
  crate/repo names + governance-verb-stem tripwire + philosophy leak-smells — the
  mechanical half of the discipline test); `CONSTITUTION_TOUCHED` /
  `SWARM_GATE_TOUCHED` (in `swarm.yml`).
- **Construction mandate (from the #245 trap):** every gate workflow — especially
  the TOUCHED guards — **always runs** (no `paths:` filter) and detects its touched
  path set *inside* the job via `git diff <base>...<head>`, exiting green when
  untouched. A path-filtered required check hangs unrelated PRs.

**Hard, hook-shaped (the one named exception to "hooks are advisory"):**
- `hooks/lead-gate-{start,stop,pre}.sh` (the lead-proactivity gate, v3 —
  `CONTRACT-v3.md` after round-1 AND round-2 audits both found every prior arm was
  a predicate over FREE TEXT with an unbounded input domain: site regexes,
  worktree/sha/token scans, write-verb walks, tag scans — every fix moved the
  squeeze between jamming legitimate traffic and being dodged by a rewording. v3
  is a mechanism change, not a third patch, narrowed under a usage-limit scope
  cut mid-round to **one choke point**: a fresh `Agent`/`Task` dispatch of a
  verifier-exit type (`adversarial-audit`/`fix-verifier`/`acceptance-verifier`)
  is denied iff its prompt names, as a WHOLE TOKEN (never a raw substring —
  `ci/gpu` does not gate `ci/gpu-dev`), an open BLOCK's recorded `worktree` (or a
  path under it), `head_sha` (full or a >=7-char prefix), or `unit_branch` of the SAME
  `agent_type`, AND no **accepted relay artifact**
  (`.jammi/gate-state/<slug>.relay.<agent_type>.<block_ts>.json`, written by the
  lead directly — never scanned from message prose — whose `sites` keys are an
  exact-string superset of the verifier's `class_enumeration`) exists for that
  `(unit, agent_type, block_ts)`. A first dispatch of any type is structurally
  never gated. `SendMessage` gating, implementer-dispatch binding, and the Bash
  backstop are DROPPED ENTIRELY (not log-only): round 1 and round 2 both proved
  free-text relay/write detection is undecidable without jamming legitimate
  freeze/status/stand-down/hygiene/advisory-fold traffic or ordinary compound
  Bash reads; the mechanical control for hook-file protection is
  `permissions.deny` on Edit/Write/MultiEdit, unchanged. The agent-type lattice
  is closed-world (an unrecognized `subagent_type` is DENIED, not allowed). The
  deciding state is written by `SubagentStart`/`SubagentStop` from the verifier's
  OWN reported verdict, never by the lead's prose — `SubagentStop` takes the LAST
  fenced ` ```json ` block with `"kind": "verdict"` (a one-release `<verdict>`
  tag fallback when none exists), both paths JSON-string-aware from right after
  the opening marker so a `</verdict>`/`}` inside a quoted `notes` string can
  never truncate the region early. A same-type PASS clears its own BLOCK; a
  fix-/acceptance-verifier PASS also clears an older `adversarial-audit` BLOCK
  when that BLOCK's own relay was accepted. The `PreToolUse` decider FAILS
  CLOSED on the two-value `{0,2}` lattice: an internal error, missing `python3`,
  unreadable state, or any other non-zero interpreter exit all DENY, never exit
  1 — proven by `ci/scripts/check_lead_gate.py --self-test` and a one-time
  fresh-session execution-provenance log (`ci/hook-acceptance/`). Motivated by
  F10 (`SELF-FAILURE-MODES.md`). Three documented, visible residuals remain
  (relaying by `SendMessage`, an unlabeled same-agent_type re-dispatch,
  `disableAllHooks`) — not claimed closed; see `hooks/README.md` for the exact
  mechanical-vs-visible-only statement and each residual's runtime tell.
  Operator escape hatch: `rm .jammi/gate-state/<slug>.*`.

**Soft (discipline, advisory, fail-open — labeled as such):**
- `hooks/build-env-guard.sh` (PreToolUse Bash, opt-in) — warns on
  `RUSTFLAGS`/`RUSTC_WRAPPER` override (the ~100-min cache-miss), non-unique
  `CARGO_TARGET_DIR` (build-lock contention), NVMe disk pressure, and maturin
  cross-worktree `PYTHONPATH` shadowing.
- `hooks/stop-gate.sh` (opt-in) — P0 static checks on a dirty tree at Stop.
- `hooks/agent-routing-gate.sh` (PreToolUse `Agent|Task`, advisory, armed by
  default) — nudges a phase-shaped dispatch to its gate agent; fails open.
  Re-matched from `Task`-only to `Agent|Task` in the lead-proactivity-gate PR: the
  pressure-test's census of this session's own transcripts found the MODEL-side
  dispatch tool named `Agent` 475 times and `Task` 0 times, so the original
  matcher had almost certainly been silently dead the entire time it was wired —
  the hook PAYLOAD's own `tool_name` field is a separate claim, confirmed only by
  the fresh-session log (`ci/hook-acceptance/`).

Native per-agent `tools:` are the real capability boundary (a verifier has no
`Edit`). **No ported permission/deny lists** beyond the two narrow,
self-protection-scoped `permissions.deny` entries above (the CONSTITUTION.md
edit-deny, and the `.claude/hooks/**`/`settings.json` edit-deny) — Jammi runs
auto-mode, lighter than Lace, and does not port Lace's broader permission
deny-list machinery.

## 8. The constitution + typed anchors

`docs/swarm/CONSTITUTION.md` (authored; human-amend-only; tool-blocked in
`settings.json`). An **index**: each invariant is `ID | statement | canonical
source | code anchor | checked by`. Boundary invariants B1–B6, correctness K1–K7,
every anchor grep-verified.

`check_constitution_anchors.py` uses **typed** anchors so it stays real for every
class (the first draft went vacuous for boundary invariants):
- `rust_symbol` → the symbol still parses (as `doc-parity` does for the enum).
- `gate_script` → the script exists **and is referenced in `swarm.yml`**.
- `doc_heading` → the heading exists.
**Every boundary invariant must anchor to a live `gate_script`** — its enforcer. A
boundary invariant with no wired gate is itself a finding (an unenforced invariant),
turning the anchor check into a completeness check over the enforcement surface.

## 9. State: ledgers + the escape/eval loop

- **Facts ledger** — `.jammi/ledger/<session>.jsonl`, per-session, gitignored.
- **Escape ledger** — `.jammi/escapes.jsonl`, tracked, engine-incidents-only (19
  seeded). Each row: `symptom_spec{intended,observable,control}`, `which_gate_missed`
  (naming the mechanism that now catches it), `status`. **Citation discipline (not a wired
  gate):** when a diff transitions an escape to `closed`/`eval_added`, a golden eval citing
  that escape id should be present in the same diff (grep the id in changed eval files). This
  is **discipline today**, enforced by the `fix-verifier` card's review — no committed script
  checks it and no runner executes `.claude/evals/golden/*`. A mechanical
  `check_escape_citations.py` grep gate is a **candidate tightening** (§12, G-e). Not "every
  test cites an escape."

## 9a. Evals — the swarm's held-out test set

`.claude/evals/` — how the swarm proves its verifiers actually fire, and the
generalization test (§2.8). Layered like the engine's own test tiers:
- **static** (`evals/static/*` — the CI gates above; deterministic, $0, every PR).
- **golden set** (`evals/golden/*`) — each past **escape becomes a case**: given the
  situation, does the verifier that should catch it fire? An escape is only `closed`
  when a golden eval proves the catch — citation discipline enforced by the `fix-verifier`
  card today (a mechanical `check_escape_citations.py` grep gate is a candidate tightening,
  §12 G-e), not a wired gate.
- **generalization by mutation** — cases hold out *novel* perturbations of each
  principle, not replays of the logged bug (shared red-green primitive with
  `fix-verifier`). A verifier passes only if it catches unseen mutants.
- **judge / Monte-Carlo** — LLM-judge and stochastic tiers run local/on-demand, not
  in the PR gate.

## 10. File layout (engine)

```
.claude/
  agents/   lead + 6 verifiers + 9 domain agents + doc-currency(4) + cookbook
  hooks/    build-env-guard.sh, stop-gate.sh, agent-routing-gate.sh (advisory);
            lead-gate-{start,stop,pre}.sh + lead-gate-lib.py (fail-closed,
            armed by default); README.md
  evals/    static/ (gates) , golden/ (escape-derived) , README.md
  settings.json   wires the advisory routing nudge + the lead-gate hook family
                  (SubagentStart/SubagentStop/PreToolUse); deny edits to
                  CONSTITUTION.md and to hooks/**+settings.json themselves
  AGENTS.md       operating-model overview
docs/swarm/  CONSTITUTION.md , SELF-FAILURE-MODES.md
ci/scripts/  new: check_swarm_bijection.py, check_constitution_anchors.py,
             check_no_consumer_names.py, check_lead_gate.py ; existing gates
.github/workflows/  swarm.yml (always-run gates + git-diff-scoped TOUCHED guards)
.jammi/      escapes.jsonl (tracked) , ledger/ (gitignored) , gate-state/ (gitignored)
docs/plans/53-agentic-swarm/  README, ARCHITECTURE (this), LESSONS
```

## 11. Build plan — one engine PR (ordered commits) + one consumer PR

The engine swarm is one rigor-chain unit, so per the repo's PR-sizing rule it is
**one PR**; the steps below are its **ordered commits**, giving bisectability and
per-commit audit focus without the CI-minutes and merge overhead of five PRs. The
bijection/TOUCHED wiring is not required-at-merge (§7), so a separate PR would
isolate no real risk. The only justified split is cross-repo: the consumer swarm is
a second PR in the enterprise repo.

**PR-1 (jammi-ai) — the complete engine swarm.** Ordered commits:
1. **Foundation** — `CONSTITUTION.md` (typed anchors), `SELF-FAILURE-MODES.md`,
   `LESSONS.md`, `check_constitution_anchors.py`.
2. **Enforcement spine** — the 9 domain manifests, `check_swarm_bijection.py`,
   `check_no_consumer_names.py` (generic patterns only — no committed consumer
   name), `swarm.yml` (always-run gates + git-diff-scoped `CONSTITUTION_TOUCHED`/
   `SWARM_GATE_TOUCHED`).
3. **Operating core** — `lead.md`, the six verifier cards, `AGENTS.md`.
4. **Domain agents + ledger + evals** — the nine crate cards, `.jammi/escapes.jsonl`,
   the `.claude/evals/` scaffold + golden seeds.
5. **Cookbook loop + hooks** — the cookbook agent's phase-6.5 emit behavior, the
   three hooks, `settings.json`.

The independent audit reviews PR-1 commit-by-commit. (Genesis note: this PR adds the
`CONSTITUTION_TOUCHED`/`SWARM_GATE_TOUCHED` guards, so they flag their own
introduction — the genesis PR is human-/admin-merged by design; the swarm's own
creation is a human act.)

**PR-2 (jammi-enterprise) — the consumer swarm.** Vendor the swarm-kit + author the
enterprise constitution/manifests/ledger. Separate repo, so a separate PR.

## 12. Known gaps (honest; candidate future gates — not silently omitted)

From `LESSONS.md` §Gaps: **G-a** one-root-cause-two-homes has no mechanical
sibling-finder (candidate: retrospective greps structurally-similar sites on
escape-close); **G-b** the honesty catches are audit judgment, not fail-closed
(candidate: a dual-recompute oracle for cookbook headline numbers); **G-c** teaching
invariants (supervision-ceiling, shift-geometry) stay discipline (no oracle, flagged
so no one mistakes them for enforced); **G-d** the discipline-test semantic lens has
no ground-truth oracle beyond dep-level (partially covered by the governance-verb-
stem tripwire in `check_no_consumer_names.py`); **G-e** the escape→golden citation is
`fix-verifier` discipline, not a wired gate — candidate gate: on a PR that transitions an
escape's status to `closed`/`eval_added` in `escapes.jsonl`, require a golden citing that id
in the same diff (transition-only, so a newly-added row on genesis needs none). These are
labeled discipline today, tracked as candidate tightenings.
```
