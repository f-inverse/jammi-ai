# PROPOSAL (human-merge): make a class of lead mistakes structurally impossible — pre-flight cards (mechanism 1) + a committed, fresh phase-5 oracle record (mechanism 2)

Status: PROPOSED. Four patch files, proven applying IN SEQUENCE (mech1, then the
three mech2 patches, in the order below) against a pristine, independently-cloned
copy of `main` (`0fc0370edf4b71dc47bc0826064162e48aafd462`) — see "Proven", below.
**Corrected**: `mech2-lead-gate-lib-export-oracle.patch` and
`mech2-swarm-yml-wiring.patch` are cut against the REAL merge-order base, not
bare `main` — `main` carries R7's and R11's designs only as `.patch` FILES, not
applied; a human applying this proposal in practice applies `R7-patch{1,2,3}`
and `R11-patch1` first (they are the shipped, prior proposals touching the same
two files, `.claude/hooks/lead-gate-lib.py` and `.github/workflows/swarm.yml`),
so these two patches are rebuilt against that real, fully-applied base and
re-verified there (see "Proven" and "A base-mismatch class this program
measured", below). The design is unchanged by the correction: the narrow
`--export-oracle` command (never reusing R7's not-yet-landed `cmd_export`) and
the separate `docs/rigor/<slug>.oracle.jsonl` record path remain exactly as
designed. No file is edited directly in this PR; every change ships as a
`.patch` (agent cards, `.claude/hooks/lead-gate-lib.py`, and
`.github/workflows/swarm.yml` are all human-amend-only per
`SWARM_GATE_TOUCHED`/the constitution, so the diff a human reviews at merge IS
the patch, never a fait-accompli edit):

- `docs/plans/53-agentic-swarm/proposals/LEAD-INVARIANTS-mech1-agent-cards.patch`
  → all 9 implementer agent cards (`ai-core`, `bench`, `cli`, `cookbook`, `db`,
  `docs-ci`, `numerics`, `python`, `wire-server`).
- `docs/plans/53-agentic-swarm/proposals/LEAD-INVARIANTS-mech2-lead-gate-lib-export-oracle.patch`
  → `.claude/hooks/lead-gate-lib.py` (adds `--export-oracle`).
- `docs/plans/53-agentic-swarm/proposals/LEAD-INVARIANTS-mech2-new-check-files.patch`
  → `ci/scripts/_swarm_diff_shape.py` (new), `ci/scripts/check_oracle_gate.py`
  (new), `ci/scripts/oracle_gate_allowlist.txt` (new).
- `docs/plans/53-agentic-swarm/proposals/LEAD-INVARIANTS-mech2-swarm-yml-wiring.patch`
  → `.github/workflows/swarm.yml`.

`SWARM_GATE_TOUCHED` (`.claude/agents/*.md`, `ci/scripts/**/check_*.py`,
`.claude/hooks/**`, `.github/workflows/swarm.yml`) → every one of the four
patches, proposal only, human admin-merge.

## The problem, and the principle

Stated by the user after a program that took roughly twenty fix rounds: operational
knowledge that keeps being rediscovered lives in the lead's memory and in
per-dispatch briefs. Both are advisory — a fresh lead in a new session reads
neither reliably. **Anything the lead has to remember is a defect.** Three places
enforce rather than advise: agent cards (load automatically for every agent), the
lead-gate hook (sees dispatch shape), and CI gates (see the tree). Everything else
is a note. This proposal is one mechanism per enforcement point that still had a
gap: cards (mechanism 1) and CI (mechanism 2). The lead-gate hook's own
enforcement point (esc-097, `hooks/lead-gate-pre.sh`) is untouched here — nothing
in this proposal's scope needed it.

---

## Mechanism 1 — operational invariants belong in agent cards, not lead briefs

### Measured, not asserted: the real distribution (R-A)

The user's own quick grep found "five of twenty-four" cards carrying a
build-directory rule, and specifically not the card dispatched most heavily this
program. That number is **wrong**, measured directly, and the truth is a
different and more serious defect than a coverage gap:

```
$ grep -lIE '^##+ *Pre-flight' .claude/agents/*.md | wc -l
       9
$ grep -lI 'CARGO_TARGET_DIR' .claude/agents/*.md | wc -l
      12
```

The 9 are exactly the 9 domain **implementer** cards: `ai-core`, `bench`, `cli`,
`cookbook`, `db`, `docs-ci`, `numerics`, `python`, `wire-server` — every single
one of them already has a `## Pre-flight` section and already names
`CARGO_TARGET_DIR` in its first bullet. (The other 3 of the 12 —
`acceptance-verifier.md`, `fix-verifier.md`, `lead.md` — mention
`CARGO_TARGET_DIR` in a different role: a verifier's own isolated re-run copy, or
the phase table's one-line description of dispatch. They are not "pre-flight for
an implementer" and are out of scope for this mechanism.) Coverage is total: **9
of 9 implementer cards, not 5 of 24.**

The real defect is in the CONTENT of that rule, uniformly, across all 9:

```
$ grep -n -B1 -A1 'CARGO_TARGET_DIR' .claude/agents/{ai-core,bench,cli,cookbook,db,docs-ci,numerics,python,wire-server}.md
```
(full output cited inline below) — every one of the 9 cards says, verbatim or
near-verbatim: *"Work in your isolated worktree with a **unique** `CARGO_TARGET_DIR`
(e.g. `target/wt-<agent-name>-$$`)."* The example path is keyed by **agent name +
PID**, i.e. per-DISPATCH, not per-worktree. `db.md`'s own version makes the
(overly broad) reading explicit: *"never share a target dir (a shared one serves
stale artifacts...)"* — never, not "never across worktrees."

`docs/swarm/SELF-FAILURE-MODES.md:86` (family F5, the incident this rule exists
to prevent) says *"Give every agent/worktree a **unique** `CARGO_TARGET_DIR`"* —
"agent/worktree" reads as either scope, and every card that operationalized it
picked the narrower, wrong one: unique per agent. That is the exact ambiguity the
user's incident report names: *"Six agents each built the workspace cold in a
private directory, thirty gigabytes, and it was a large part of why every task
took forty-five minutes."* The isolation boundary `build-env-guard.sh` (the one
mechanical enforcement point that touches this rule at all) actually checks is
weaker still — advisory-only, and it warns only when `CARGO_TARGET_DIR` is
**totally unset**, never on whether it is shared correctly:

```
.claude/hooks/build-env-guard.sh:70-83 — warns iff no CARGO_TARGET_DIR is set at
all (inline, exported, or --target-dir); does not check per-agent vs per-worktree.
```
So nothing currently distinguishes "this agent has ITS OWN target dir" (good, if
a different tree) from "this agent has a target dir private to ITS OWN dispatch
on a worktree five other agents are also building in" (the incident). This
mechanism fixes the SENTENCE, not the hook — the hook's job (warn when totally
absent) is unaffected and still correct.

### The fix

`LEAD-INVARIANTS-mech1-agent-cards.patch` rewrites item 1 of every one of the 9
implementer cards' `## Pre-flight` section (preserving each card's own
domain-specific clause — `db.md`'s sccache-miss timing, `cookbook.md`'s temp
output dir, `python.md`'s pinned `PYTHONPATH`) to:

> **Build into your worktree's own `CARGO_TARGET_DIR`** (e.g.
> `target/wt-<worktree-basename>`), **shared by every agent working THIS
> worktree** — the isolation boundary is the worktree, not the dispatch: a fresh
> private directory per agent means a cold rebuild of the whole workspace on
> every dispatch. Use a genuinely private directory only for a genuinely
> different tree (an archive, a separate clone, a checkout at another commit) —
> never for a second agent sharing this SAME worktree.

and adds three new, previously-nowhere-encoded pre-flight items, each tied to a
concrete cost from the user's own incident list:

- **Scope your gates to what you changed** — a markdown-only or single-surface
  edit does not need the full workspace suite.
- **On a shared worktree: commit by explicit pathspec, re-read `HEAD` in the same
  command as any amend, and verify the commit's contents again at the end of the
  task.** Two agents on the last program damaged each other's work this way: one
  amended past a `HEAD` that had moved, one overwrote an edit no one had
  committed yet. `git commit -- <files>` (never `git commit -a`/`git add -A`);
  `git rev-parse HEAD && git commit --amend …` in ONE invocation; `git show
  --stat HEAD` / `git diff HEAD~1 HEAD -- <files>` once more before reporting
  done.
- **Never pipe a build or test through another command when you read its exit
  status** — `cargo test … | tail` hands you the LAST command's `$?`, never the
  build/test's own. An agent on the last program reported a passing build that
  had actually failed underneath a `tail`.

Every implementer card gets the identical wording for items 3–5 (a uniform,
mechanically-greppable template) and a card-specific item 1 that keeps its own
domain clause. `## Pre-flight` numbering is preserved (`2.` — load the
constitution invariants — is untouched; the new items are `3.`–`5.`).

### What this mechanism does NOT do, stated plainly

It is still prose a card carries — nothing MECHANICALLY prevents an agent from
ignoring it, the same limit every other agent-card invariant already has (cards
are advisory to the model reading them; they are enforced only in the sense that
they load automatically into every dispatch, never skipped by a stale brief). The
actual teeth against a genuinely shared, corrupted target dir remain
`build-env-guard.sh`'s advisory warning and the agent's own worktree isolation.
This mechanism's contribution is narrower and real: it removes the AMBIGUITY that
caused 9-for-9 cards to encode the wrong reading of an already-existing rule, and
it puts three previously-nowhere-encoded, cost-measured operational rules where
every future dispatch will see them without the lead having to remember to say
so in a brief.

---

## Mechanism 2 — a CI gate that makes running the oracle late structurally impossible

### The problem, measured against the real tree

On the last program, phase 5 (`oracle`, `.claude/agents/oracle.md`,
`docs/swarm/CONSTITUTION.md` B2/B6/K4–K6 via `.claude/agents/lead.md`'s own phase
table) — the one gate whose verdict is never consensus-overridable — did not run
until a 57-commit branch's very end, where it hard-blocked immediately. Nothing
in the phase machine, the lead-gate hook, or CI prevented that ordering: the
oracle's own verdict lives only in `.jammi/gate-state/<slug>.jsonl`, which is
gitignored and lead-writable — the same class of ledger
`docs/plans/53-agentic-swarm/proposals/R7-committed-rigor-record.md` (already on
`main`, merged as PR #518) was designed to stop trusting, for the pressure-tester
row specifically. Measured directly, before designing anything on top of it:

```
$ git show origin/main:ci/scripts/check_rigor_record.py
fatal: path 'ci/scripts/check_rigor_record.py' does not exist in 'origin/main'
$ git show origin/main:docs/plans/53-agentic-swarm/proposals/R7-committed-rigor-record.md | head -3
# PROPOSAL (human-merge): tighten(lead-gate) — a committed rigor record...
Status: PROPOSED. Five designs have gone through this problem...
```

So: `main` carries R7's DESIGN (the proposal doc + its three `.patch` files,
merged as documentation) but has **not** applied it — `check_rigor_record.py`,
`docs/rigor/`, and `lead-gate-lib.py`'s `--export` do not exist as live code on
`main` today. This proposal's mechanism 2 is built to work **whether or not R7
ever lands**: it adds its OWN, narrower `--export-oracle` subcommand (a distinct
name and a distinct output path, `docs/rigor/<slug>.oracle.jsonl` vs R7's
`docs/rigor/<slug>.jsonl`) rather than reusing R7's not-yet-applied
`cmd_export`/`--export`, so the two proposals can be merged in either order, or
only one of them ever, without a textual collision. It reuses R7's DESIGN
LANGUAGE and its measured, hard-won lessons (arm on diff shape, never a
lead-reported count; committed artifacts, not a gitignored ledger; state the
residual, never imply it closed) because those lessons are general, not
specific to the pressure-tester row R7 itself checks.

### The rule

`ci/scripts/check_oracle_gate.py`, wired into `swarm.yml` (the hermetic,
no-`paths:`-filter workflow that already hosts `check_lead_gate.py --self-test`,
matching family T's own rule that every gate workflow always runs and detects
its touched set inside the job): if `origin/<base>...HEAD` (three-dot, pinned)
touches any path under `crates/**` or `cookbook/**` — the surfaces
`oracle.md`'s own principle rubric reasons about (dep-direction, cookbook
one-way, lockstep version, append-only migrations, tenant isolation,
embedded↔remote parity, per-variant safety oracles) — this check is ARMED and
requires, from COMMITTED files only:

1. A record at `docs/rigor/<unit_slug>.oracle.jsonl` (`unit_slug =
   slugify(<head branch>)`, the SAME `slugify()` `lead-gate-lib.py` already
   uses, loaded dynamically so the two can never drift) — produced by
   `lead-gate-lib.py --export-oracle <slug>`, never hand-typed, and exists and
   parses at HEAD.
2. That record carries at least one row with `agent_type == "oracle"` **and**
   `verdict == "PASS"`. A `HARD_BLOCK` row does not satisfy this — its own
   `verdict_raw` says so, and `oracle.md`'s hard-blocks are never overridable,
   so a record whose only rows are `HARD_BLOCK` correctly fails rather than
   being read as "an oracle ran."
3. **At least one such PASS row is FRESH at the branch's current head** — the
   part that actually answers the task: *"fails when a branch has no phase-5
   oracle verdict recorded at its current head."*

### Freshness: exact content, never ancestry — and why

The obvious first design (reuse R7's own `merge-base --is-ancestor <head_sha>
HEAD` check) is wrong for two independent reasons, one already measured by R7
itself and one new:

- **R7 already measured that ancestry-checking is unreliable in this repo's own
  workflow.** `R7-committed-rigor-record.md`'s own "Proven"/design-history
  section: an amend-after-verification orphans the pre-amend `head_sha`
  (`merge-base --is-ancestor` exits 1 on the honest path), and a transport clone
  (what `actions/checkout` produces) can be missing the object entirely. R7 made
  its OWN ancestry check advisory-only for exactly this reason.
- **Ancestry is not even the right property for THIS check, independent of
  reliability.** "Some ancestor of HEAD passed the oracle" is compatible with
  HEAD having since drifted arbitrarily far from what the oracle actually read —
  which is precisely the failure this mechanism exists to close (*"A verdict
  recorded at an older commit must not satisfy a later head, or the gate is
  theatre"*). Ancestry answers "was this commit on the branch," not "is the code
  the oracle reviewed still the code that's here."

The check instead asks the stronger, and — measured directly, below — simpler
question directly: **does the code match, byte for byte, outside the record
file itself.**

```python
git diff --name-only <recorded head_sha> HEAD -- . ':(exclude)docs/rigor/**'
```
Empty → fresh (the code the oracle saw is exactly the code that's here; how many
follow-up commits sit on top, or whether `head_sha` is even a literal ancestor,
is irrelevant). Non-empty → stale, and the check names every changed path (up to
5) in its failure message. This also resolves — for free, with no special case —
the "committing the record changes HEAD" bootstrap the naive "recorded sha ==
HEAD, exactly" design would hit: the lead commits code, dispatches `oracle`
against that commit, exports the PASS row (`head_sha` = that commit), and
commits the record as a follow-up commit (or the same commit); either way the
diff between the recorded sha and the new HEAD touches only `docs/rigor/**`, so
freshness holds. Proven directly by fixture (`OG5`, `OG6`, `OG4`; see "Proven,"
below) — including the specific case the task named: `OG4` commits a REAL
further `crates/**` change on top of an exported PASS row and asserts the check
FAILS, naming "stale," never silently passing on the older recording.

**This check fails CLOSED on an unresolvable `head_sha`** (`OG7`) — the opposite
of R7's own choice for its unrelated ancestry check, and stated as a deliberate
divergence, not an inconsistency: R7 made unresolvability advisory because
ancestry there was only ever a disclosure, never the enforcement mechanism. Here,
resolvability plus content-equality **is** the entire freshness proof; an
unresolvable sha means the proof cannot be constructed at all, and
`scratchpad/CONTRACT-RULES.md`'s own R-I ("a state defined by missing evidence
cannot be given a definite consequence") cuts toward BLOCK here, not silent
pass, because the state in question — "cannot be shown fresh" — is exactly the
state this check exists to catch.

**A branch with no oracle-reviewed unit in it is not blocked** (`OG1`, `OG8f`):
a diff touching neither `crates/**` nor `cookbook/**` never arms in the first
place — there is nothing for `oracle` to have reviewed. The known, structural
no-op shapes (not a `pull_request` event, `dependabot[bot]`, a revert, a
release-shaped diff, a purely human-authored range) are factored into a new,
shared module, `ci/scripts/_swarm_diff_shape.py`, rather than copy-pasted from
`check_rigor_record.py`'s inline version — `docs-ci`'s own DRY invariant ("a
fact lives in exactly one place") applied to this proposal's own code, not just
prose. If/when R7 lands, `check_rigor_record.py` should be refactored to import
this same module instead of keeping its own inline copy of the identical five
predicates; not required for this mechanism to work today, flagged as a
follow-up.

A shrink-only allowlist (`ci/scripts/oracle_gate_allowlist.txt`,
`--check-allowlist-only-shrinks`, the exact ratchet `check_perf_claims.py`/
(would-be) `check_rigor_record.py` already use for their own allowlists) exists
for a genuinely exceptional unit — a human decision made directly on `main`,
never an autonomous addition on a swarm branch.

### The residual, stated plainly, never implied closed

This check proves a SHAPE: a committed row claims `agent_type: "oracle"`,
`verdict: "PASS"`, and a `head_sha` whose tree matches HEAD outside
`docs/rigor/**`. **It does not prove the `oracle` subagent actually ran, or ran
honestly, or that its `checks[]` array reflects a real gate execution** —
`.jammi/gate-state/` carries no `permissions.deny` entry and the lead already
writes into it by design (the exact residual `check_rigor_record.py` already
states for its own committed record). Concretely, and stated as plainly as R7
states its own: **a lead CAN satisfy this check dishonestly** by hand-appending a
row `{"agent_type": "oracle", "verdict": "PASS", "head_sha": "<current HEAD
sha>"}` to `docs/rigor/<slug>.oracle.jsonl` and committing it, without ever
dispatching `oracle` at all — the freshness check finds the tree diff trivially
empty (`head_sha` IS `HEAD`) and passes. Making the STALENESS proof strong does
not make the underlying CLAIM any harder to fabricate; it only makes fabricating
a *stale-but-claimed-fresh* verdict impossible, which is the one failure mode
this mechanism was asked to close. **ARMED BY THE DIFF, SATISFIED BY DISCLOSURE,
JUDGED BY THE HUMAN. TAMPER-EVIDENT, NEVER TAMPER-PROOF** — R7's own three-clause
honesty statement, repeated verbatim here because it is still true here, not
because it is decorative.

One further, narrower residual specific to this mechanism: the arming scope
(`crates/**`, `cookbook/**`) is a judgment call, stated as one — a change that
violates an oracle invariant through some OTHER path (e.g. a `.github/
workflows/**` change that alters how a wire RPC gets deployed, without touching
`crates/**` itself) would not arm this check. `oracle.md`'s own principle rubric
is about source-code seams, and every invariant it names is checkable from a
`crates/**`/`cookbook/**` diff; I could not find a case in the card's own text
where a change confined to CI/workflow files alone would trip one of its
hard-blocks without also touching `crates/**`, but I state this as my own
reading, not a proof.

### Proven

Every step below ran against a pristine, independently-cloned copy of `main`
(`git clone --no-local --branch main`, verified `HEAD ==
0fc0370edf4b71dc47bc0826064162e48aafd462` before applying anything), never
against this worktree's own live `.claude/agents/*.md` / `.claude/hooks/**`
(both are tool-permission-denied for direct edits in this worktree by design —
the patches were authored by editing a scratch copy, diffing, and reverting the
live tree, or by hand-constructing the diff from `Read`-tool output against the
exact line numbers, then verified applying against the pristine clone, never
against the live denied path).

1. **Mechanism 1 alone**, applied to pristine clone #1:
   `git apply --check` then `git apply` clean; spot-checked `ai-core.md`'s
   resulting `## Pre-flight` block reads as designed (quoted above).
2. **Mechanism 2's `--export-oracle` addition alone**, applied to pristine
   clone #1 (already carrying mechanism 1, unrelated file set): `git apply
   --check`/`git apply` clean; `python3 -m py_compile` clean;
   `python3 lead-gate-lib.py --export-oracle nonexistent-slug` → `exported 0
   oracle row(s)`, exit 0; a driven `handle_stop` call with a real `oracle`
   `SubagentStop`-shaped payload (`verdict: PASS`, a `head_sha`) writes exactly
   one row to `.jammi/gate-state/<slug>.jsonl` with `"agent_type": "oracle",
   "verdict": "PASS"`, and `--export-oracle` reproduces it verbatim to stdout.
3. **All three mech2 patches**, applied sequentially to pristine clone #1 (now
   carrying all four patches together): `git apply --check`/`git apply` clean,
   in order (`--export-oracle` addition, then the two new files, then the
   `swarm.yml` wiring); `python3 -m py_compile` clean on all three touched/new
   Python files; `check_oracle_gate.py --self-test` → **10/10 fixtures pass**
   (`OG1`–`OG10`: not-armed docs-only no-op; armed-with-no-record fails naming
   the missing record; a record with only a `HARD_BLOCK` row fails naming "no
   oracle PASS row"; a PASS row recorded at an OLDER commit with a REAL further
   `crates/**` change on top fails naming "stale"; a record-only follow-up
   commit over unchanged code stays fresh; `head_sha == HEAD` directly is
   fresh; an unresolvable `head_sha` FAILS CLOSED (not advisory); dependabot/
   revert/release-shaped/human-authored/non-PR-event/no-oracle-reviewed-path
   diffs all no-op; an allowlisted unit no-ops even though armed; the
   allowlist-only-shrinks ratchet passes unchanged and fails on a new,
   not-yet-on-`origin/main` entry).
4. **Full sequential re-application from scratch**, all four patches, on a
   SECOND, separately and independently cloned pristine copy (never the same
   tree the patches were authored/tested against): all four `git apply
   --check`/`git apply` pairs clean, in the documented order; `python3
   -m py_compile` clean on `.claude/hooks/lead-gate-lib.py`,
   `ci/scripts/_swarm_diff_shape.py`, `ci/scripts/check_oracle_gate.py`;
   `python3 ci/scripts/check_lead_gate.py --self-test` → **66/66** (unaffected
   — `--export-oracle` is additive, touches no existing code path);
   `python3 ci/scripts/test_check_lead_gate.py` → **OK, 5 tests**; `python3
   ci/scripts/check_oracle_gate.py --self-test` → **10/10**; `python3
   ci/scripts/check_swarm_bijection.py` → **PASS** (`ci/**`'s new files fall
   under `docs-ci`'s existing glob ownership, no bijection change needed);
   `python3 ci/scripts/check_constitution_anchors.py` → **OK, 13
   invariant(s)**; `python3 ci/scripts/check_no_consumer_names.py` → **OK**.

Every command above and its real exit code were captured directly, per
`scratchpad/CONTRACT-RULES.md` R-A — no step is narrated without having been
run.

5. **Corrected re-verification against the REAL 8-patch merge order**
   (`R7-patch1`, `R7-patch2`, `R7-patch3`, `R11-patch1`, then this proposal's
   own four, in that order), on a pristine clone of `origin/main`
   (`0fc0370edf4b71dc47bc0826064162e48aafd462`): all eight `git apply
   --check`/`git apply` pairs clean, in sequence — including the two patches
   this correction rebuilt, which FAILED in this exact sequence before the
   rebuild (`mech2-lead-gate-lib-export-oracle` conflicted with R7-patch2's
   own addition of `cmd_export`/the `main()` dispatch at
   `.claude/hooks/lead-gate-lib.py:1394`; `mech2-swarm-yml-wiring` conflicted
   with R7-patch2's own appended self-test and guard steps at
   `.github/workflows/swarm.yml:70`/`:113`). Reproduced via
   `scratchpad/apply-swarm-proposals.sh --dry-run` — see the fix commit's own
   record for the exact output.

## What I could not verify myself, asked for explicitly

- **Mechanism 2's arming scope** (`crates/**`/`cookbook/**` only, not
  `.github/workflows/**`/`ci/**`) is my own reading of `oracle.md`'s principle
  rubric, not a value the card states in code-shaped form — flagged above,
  under "The residual."
- **I did not run `ci/scripts/check_ci_guard_wiring.py`** against a full copy of
  the patched repository, for the same proportionality reason R7's own author
  gave for the identical decision: I instead confirmed directly, by reading
  `swarm.yml`'s new lines, that they are literal, non-comment `run: python3
  ci/scripts/check_oracle_gate.py` lines (twice more, with flags) — the exact
  shape that script's comment-stripped-line-scan requires.
- **I have not run the real CI workflow** (`swarm.yml` on a real GitHub Actions
  runner) — only its exact commands, locally, against pristine clones. The
  `fetch-depth: 0` requirement `swarm.yml` already sets for its OTHER
  git-diff-scoped guards is what makes `origin/<base>` resolvable for this
  check too; I did not independently re-verify GitHub Actions' own checkout
  behavior beyond reading the existing job's `with: fetch-depth: 0`.

## A base-mismatch class this program measured, and a cheap mechanical check for it

Verifying each of this proposal's four patches by applying it, alone, to a
pristine `main` is exactly what let two of them (`mech2-lead-gate-lib-export-
oracle`, `mech2-swarm-yml-wiring`) pass that check and then fail to apply in
the order a human actually runs: `R7-patch{1,2,3}` and `R11-patch1` are
shipped, prior proposals that also edit `.claude/hooks/lead-gate-lib.py` and
`.github/workflows/swarm.yml`, and `main` carries their DESIGN as `.patch`
files without applying them (see "The problem, measured against the real
tree," above, for the identical observation about R7 specifically). A human
merging this stack applies those first, so the real base for this proposal's
two overlapping patches is "main + R7 + R11," not bare main — and nothing
before this correction checked that.

This is a real, general property of shipping enforcement as patch files
rather than direct commits, not an artifact of this one proposal: a patch is
cut against A base; two sibling proposals that touch the same file go stale
against EACH OTHER the moment either one's base assumption stops holding
(one lands, or a third proposal lands between them), and nothing currently
re-checks that the declared apply order still holds once it is written down.
A per-patch "does this apply to pristine main" check cannot catch it by
construction — it is checking the wrong base.

**A cheap mechanical fix exists, and this proposal ships a first cut of it
rather than leaving the ordering to memory**: `scratchpad/apply-swarm-
proposals.sh` (untracked — this repo's own `/scratchpad/` convention for
session-local tooling, per `.gitignore:112-113`) takes a declared
`name:ref:path` SEQUENCE, sandboxes at `origin/main` in a throwaway clone, and
re-applies every patch in order, reading each one's text from its OWN
proposal branch via `git show <ref>:<path>` — never assuming a not-yet-merged
patch's text already lives on `main`. It is cheap because the entire check is
`git apply --check` in a loop; no compilation, no toolchain. The residual is
that the SEQUENCE itself is still hand-maintained (a new sibling proposal must
be added to it by a human), which is a strictly smaller memory burden than
re-deriving the correct order from scratch every time, and — unlike the
un-checked assumption this correction fixes — a missing or wrong entry in a
committed sequence is a visible diff at review time, not a silent one. Folding
this into a required CI gate (so the sequence is asserted fresh on every push
to any `proposal/*` branch, the same "always runs, detects its own touched set"
shape `swarm.yml`'s other gates already use) is a natural next step, not done
here: it would need the SEQUENCE to live in a committed, human-amend-only
manifest rather than a gitignored script, and deciding that manifest's home is
a scope question for whoever picks this up, not answered by this proposal.

## Ledger lifecycle

No escape id is assigned yet. `.jammi/escapes.jsonl` carries no
`esc-lead-invariants-1`/`esc-lead-invariants-2` row as of this doc (the code
comments in the mech2 patches use `esc-lead-invariants-2` as a forward-looking
identifier only); one is appended, following the
`esc-097-relay-form-satisfied-without-probing-the-fix` /
`esc-lead-gate-R7`-style precedent, once a human applies all four patches and
every self-test goes green on `main`.
