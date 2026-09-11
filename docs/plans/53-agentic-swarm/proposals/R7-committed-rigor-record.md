# PROPOSAL (human-merge): tighten(lead-gate) — a committed rigor record, armed by the diff's own shape (R7)

Status: PROPOSED. Third design on this problem — R4/R5 were KILLED, R6 was KILLED, R7 came back
REFINE with binding corrections (v2, below the first `---`; where v1 and v2 differ, v2 wins — this
doc states the v2 design directly and cites v1 only where the design history matters). Two patch
files, proven applying and self-test-green against a pristine, independently-reconstructed copy of
the real tree (see "Proven", below):

- `docs/plans/53-agentic-swarm/proposals/R7-patch1-per-type-pass-vocabulary.patch` → `.claude/hooks/
  lead-gate-lib.py`, `.claude/hooks/README.md`, `ci/scripts/check_lead_gate.py`.
- `docs/plans/53-agentic-swarm/proposals/R7-patch2-committed-rigor-record.patch` → `.claude/hooks/
  lead-gate-lib.py`, `.claude/agents/adversarial-audit.md`, `.github/workflows/swarm.yml`,
  `ci/scripts/check_rigor_record.py` (new), `ci/scripts/rigor_record_allowlist.txt` (new).

`SWARM_GATE_TOUCHED` (`.claude/hooks/**`, `.claude/agents/*.md`, `ci/scripts/**/check_*.py`,
`.github/workflows/swarm.yml`) → both patches, proposal only, human admin-merge — every file either
patch touches is human-amend-only or itself a swarm gate definition.

## Applying the patches (human step)

```
git apply --check docs/plans/53-agentic-swarm/proposals/R7-patch1-per-type-pass-vocabulary.patch && \
  git apply docs/plans/53-agentic-swarm/proposals/R7-patch1-per-type-pass-vocabulary.patch
python3 ci/scripts/check_lead_gate.py --self-test

git apply --check docs/plans/53-agentic-swarm/proposals/R7-patch2-committed-rigor-record.patch && \
  git apply docs/plans/53-agentic-swarm/proposals/R7-patch2-committed-rigor-record.patch
python3 ci/scripts/check_lead_gate.py --self-test
python3 ci/scripts/check_rigor_record.py --self-test
```

Patches apply independently and in either order against each other's OWN file set (patch 1 never
touches a file patch 2 also touches except `lead-gate-lib.py`, where the two edit disjoint regions —
proven applying sequentially, patch 1 then patch 2, in that order, below); ship patch 1 now, patch 2
next, per the binding sequencing below.

## Design history — three rounds, cited by artifact, not narrated

**R4/R5 — KILLED.** A verifiable-sweep mechanism (re-executing a lead-authored shell command) and a
round cap billed as enforceable against the lead were both killed on reproduced evidence, not argued
down:
`{scratchpad}/tasks/abe09018c16727106.output`'s verdict (`"verdict": "KILL"`) found R5a's own unlock
condition (a `pressure-tester` row of `PROCEED` *or* `REFINE`) is R5c's own poison — `REFINE` is not
in `_PASS_LIKE`, so `normalize_verdict` classifies it `BLOCK`, and a `REFINE` row therefore counts
*toward* the very cap R5a's unlock was supposed to relieve; replayed against the live state
directory, **16 of 28 unit files were already past R5c's cap on the day it would have shipped**,
clearable only by the evidence-destroying `rm .jammi/gate-state/<slug>.*` escape hatch. R4a (execute
a lead-authored `sweep.cmd` in the decider) was killed on a direct measurement: of 7 plausible
legitimate sweep commands, 4 were DENIED by the proposed write-verb denylist (`git grep -n
'transform'`/`'confirm'` both contain the substring `rm`), while `curl … | sh`, `git clean -fdx`,
and `git reset --hard` were ALLOWED — the exact class of predicate `.claude/hooks/README.md:209-213`
already records as deleted after round 2 found it undecidable.

**R6 — KILLED.** A site-overlap intersection counter (`class_enumeration` entries intersecting
across consecutive `adversarial-audit` BLOCK rows) reproduced its own claimed arithmetic exactly (2
stops, 0 false stops over the corpus as it stood) — and then a fresh, real round of BLOCKs on
`feat_500-A-datafusion-54` (six `adversarial-audit` rounds, a human-certified single-class loop) gave
the counter a direct test it failed: peak chain **2**, never arming, across rounds whose real
`class_enumeration` entries were `deny.toml:bans/multiple-versions` → `deny.toml:103,117,
ci.yml:959` → ten `deny.toml`/`ci.yml` entries → seven more — the SAME `deny.toml`/cargo-features
class, relocated round to round, with exactly one exact-string collision (`deny.toml:66`, between
rounds 4 and 6) in the whole chain. `{scratchpad}/tasks/a51ebcc1234dd552f.output`'s verdict searched
72 parameterizations (4 granularities × 3 site sources × 3 thresholds × 2 relations × 2 caps) over
the live corpus and found none that catches this real loop without also stopping a unit before the
round that produced its own PASS; the one setting that separates does so by a bridging site
(`crates/jammi-ai/tests/it/import_pipeline.rs:56`) unrelated to the `deny.toml` class it happened to
catch — right answer, wrong reason, selected from 72 candidates against one positive example. **Sites
are not classes**: a class relocated rather than closed moves its sites by construction, which is
exactly what every site-overlap relation is blind to.

**R7 — REFINE, v2 binding.** The design contract's own thesis (enforcement moves to CI and the human
at merge, because no `PreToolUse` rule can hold against a lead who holds both `Bash` and `Write` —
the decider is code the same constrained party can rewrite, over state it can author, in a process
it can disable) reproduced true, independently, in the design round: a hand-written `{"agent_type":
"adversarial-audit","verdict":"PASS"}` line appended to a gitignored state file flips the hook's
own decision from deny to allow, end to end, against the unmodified live library. But v1's own check
2 (arm when the record shows ≥3 BLOCK rounds) had the SAME root flaw R6 had: the lead authors the
record, so the cheap dodge is writing two rows instead of three, not editing session state — v1's own
claim that this "cannot be dodged by editing session state" is false and is deleted. **The single
binding correction that turns this from a disclosure form into a rule with teeth: arm on the DIFF's
own shape, never on a count the lead reports about itself.** Three further corrections, all binding
(v2 of the session's own design contract, a scratchpad artifact, not repo-tracked): check 3 (ancestry) is RED on the honest
path — this session's OWN documented workflow (amend an implementer's commit to add session
trailers) orphans every pre-amend `head_sha`, reproduced twice (`git merge-base --is-ancestor` exits
1; the object is ABSENT in a `git clone file://` transport clone, what `actions/checkout` produces)
— so it becomes advisory; the carrier moves from a hand-typed markdown table to JSONL (the hook's
own row schema, exported by a new `--export` subcommand); the commit trailer is dropped entirely
(equally forgeable, less visible than a file in the diff, fragile under the amend-trailers workflow).

## What R7 is, as bound by v2

**Patch 1 — R7d, ship independently.** `_PASS_LIKE` was a single GLOBAL set (`{"PASS", "verified",
"PROCEED"}`) checked against every agent type identically — so an `adversarial-audit` row whose raw
verdict text was literally `"verified"` (`fix-verifier`'s own spelling) or `"PROCEED"`
(`pressure-tester`'s own spelling) also cleared an `adversarial-audit` BLOCK, independent of anything
R1–R3 check. Replaced with `_PASS_LIKE_BY_AGENT_TYPE`, a per-type map (`fix-verifier`/
`acceptance-verifier` → `"verified"`, `pressure-tester` → `"PROCEED"`, every other type → `"PASS"`),
threaded through `normalize_verdict(data, agent_type)` (both real call sites — `handle_stop`'s own
trusted payload field, and `_diagnose_row`'s stored `row["agent_type"]` — already carried the type in
scope; this is a two-call-site change) and `_diagnose_row`'s own unrecognized-value check. Confirmed
independently, twice, against real data: a census of all 261 (round-2 KILL) then all 221 (round-1
KILL) live verdict rows found every real clearing already uses its own card's spelling — no real
clearing depends on the pooled set. The self-test's `fixture_v4_v5_v6_pass_like_pinned` keeps its
three positives (`"PASS"`/adversarial-audit, `"verified"`/fix-verifier, `"PROCEED"`/pressure-tester)
and gains three negatives (`"verified"`/adversarial-audit, `"PROCEED"`/adversarial-audit,
`"verified"`/oracle — none is that card's own spelling and must NOT clear); each reproduces RED
against the unpatched, pooled `_PASS_LIKE` (verified directly, below) and GREEN once patch 1 lands.
`.claude/hooks/README.md`'s mechanical-vs-visible-only list (already four items, not three — this
doc corrects the session's own design contract's count, written against an earlier state of that file)
gains a fifth: a hand-appended verdict row in `.jammi/gate-state/<slug>.jsonl` itself, true today,
independent of everything else in this proposal.

**Patch 2 — R7a/R7b (v2), the piece that serves the user's own standing requirement.**

*The rule.* If a PR's `base...HEAD` (a three-dot range, pinned explicitly — never two-dot) touches
any path under `crates/**`, `ci/**`, or `.github/workflows/**`, `ci/scripts/check_rigor_record.py`
(wired into `swarm.yml`, the hermetic, no-`paths:`-filter workflow that already hosts
`check_lead_gate.py --self-test` — never `ci.yml`, whose concurrency cancels are routine) is ARMED
and requires, from COMMITTED files only: (1) a rigor record at `docs/rigor/<unit_slug>.jsonl` — the
hook's OWN row schema, produced by `lead-gate-lib.py --export <slug>`, never hand-typed; (2) that
record carries ≥1 row with `agent_type == "pressure-tester"` — `.claude/agents/lead.md:41`'s own
"design-before-mechanism… before a non-local fix" rule, now visible in a reviewed diff, not merely
asserted in a transcript; (3) this SAME diff adds or modifies ≥1 file under `docs/plans/*/proposals/**`
or `docs/rigor/contracts/**` — the committed mechanism contract itself; (4) that contract passes four
anti-vacuity cost-floor checks (below). `unit_slug` is `slugify(<head branch>)` — the same
`slugify()` `lead-gate-lib.py` already uses, loaded dynamically rather than reimplemented, so the two
can never drift.

*Anti-vacuity is a cost floor, stated as one, never as a judge of quality.* Nothing mechanical can
tell a good design pass from a weak one — this proposal says so plainly rather than implying check 2
covers it. What CI CAN impose: (1) every `path:line` citation in the contract resolves at HEAD (file
exists; a line number does not exceed the file's own line count); (2) the contract file's
introducing commit (`git log --follow --diff-filter=A`) is an ancestor of HEAD; (3) the contract is
not near-identical (normalized-whitespace sha256) to another unit's own committed contract — this
alone kills the two-line copy-paste; (4) exemptions live in `ci/scripts/rigor_record_allowlist.txt`,
a shrink-only allowlist (`--check-allowlist-only-shrinks`, the exact ratchet
`ci/scripts/check_perf_claims.py --check-allowlist-only-shrinks` already uses for a different file —
current entries must be a subset of `origin/main`'s; a genuinely new exemption is a human decision
made directly on `main`, never an autonomous addition on a swarm branch). Never a length or
word-count rule.

*Ancestry (check 3, historical numbering) is advisory, never a failure.* Every recorded `head_sha` in
the rigor record is checked with the exact git shape `_relay_rejection` already uses
(`rev-parse --verify`, `merge-base --is-ancestor`) — but only REPORTED. Reproduced directly, twice:
an amend-after-verification orphans the pre-amend sha (`merge-base --is-ancestor` exits 1) and a
`git clone file://` transport clone (what `actions/checkout` produces) is missing the object
entirely; `_relay_rejection`'s identical shape only works locally, where the object store still
holds the orphan.

*Known, structural no-op shapes* (never an author-declared marker/trailer/commit-message convention
— every such predicate is a one-line opt-out): not a `pull_request` event; `dependabot[bot]` as
actor; a revert (`HEAD`'s own commit subject starts with `Revert "`, git's own convention); a
release-shaped diff (every `crates/**` touch is `Cargo.toml`/`Cargo.lock` only, no `ci/**`/
`.github/workflows/**` touch); a docs-only diff (implied structurally — it never touches the three
armed roots, no special case needed); a purely human-authored range (no commit in `base..HEAD`
carries this repo's own `Co-Authored-By: Claude` trailer — one swarm-touched commit is enough to
arm). **Known, undetectable residual, stated honestly, not closed:** a retargeted stacked branch
whose three-dot range still carries a parent unit's own commits is not detectable from the diff
alone — three-dot ranges are the correct GitHub PR semantics, and there is no mechanical way to tell
"this diff is this PR's own work" from "this branch needs a rebase" without knowing the intended
base.

*`recurrence_of_round` — a verifier-authored field, never a lead-computed counter key.* Added to the
`adversarial-audit` card's verdict schema and to `parse_verdict_fields`: if a BLOCK is the SAME class
as an earlier round on the SAME unit — relocated, not new — the auditor names that earlier round's
number, having read `.jammi/gate-state/<slug>.jsonl` itself (never trusting the lead's brief, which
is exactly the channel that fails open when it conceals a prior round). No gate reads this field to
decide anything; it FAILS OPEN when the auditor does not notice, and it is forgeable by the same
ledger-append path every other row field already is — it is carried, verbatim, into the committed
rigor record for the human at merge to read. No existing card asks for anything equivalent (checked
against all seven verdict schemas).

**The honest words, binding, used exactly once, as instructed, and nowhere else in this document:**
ARMED BY THE DIFF, SATISFIED BY DISCLOSURE, JUDGED BY THE HUMAN. TAMPER-EVIDENT, NEVER TAMPER-PROOF.

## Proven

Built and tested against a from-scratch, byte-for-byte pristine reconstruction of the real files
(`cp` from the primary checkout, never a `git worktree`, deleted after use) — never against this
worktree's own live `.claude/hooks/**`, which tools are denied from editing:

1. `python3 ci/scripts/check_lead_gate.py --self-test` on the UNPATCHED library: the three new
   `fixture_v4_v5_v6_pass_like_pinned` negatives reproduce RED (`adversarial-audit`/`"verified"`
   clears when it must not) — proving the fixture is a real test, not vacuous.
2. Patch 1 applied alone (`git apply --check` then `git apply`, clean) to a fresh pristine copy:
   `python3 -m py_compile` clean; `check_lead_gate.py --self-test` → **66/66 fixtures pass**.
3. Patch 1 THEN patch 2, applied sequentially to a SEPARATE, independently-reconstructed pristine
   copy (never the same tree the patches were authored against): both `git apply --check` and
   `git apply` clean, in that order; `python3 -m py_compile` clean on all three touched/new Python
   files; `check_lead_gate.py --self-test` → **66/66**; `check_rigor_record.py --self-test` → **11/11**
   (`RR1`–`RR11`: not-armed docs-only no-ops; armed-with-no-record fails naming the missing record;
   armed-with-record-but-no-pressure-row fails; armed-with-everything-but-no-contract-file fails;
   full disclosure allows; a path:line citation past a file's own line count fails; a
   whitespace-normalized copy of an existing contract fails as near-identical; dependabot/revert/
   release-shaped/human-authored/non-PR-event diffs all no-op; an unresolvable `head_sha` WARNS,
   never fails the exit code; an allowlisted unit no-ops even though armed; the allowlist-only-shrinks
   ratchet passes unchanged and fails on a new, not-yet-on-`origin/main` entry).
4. `python3 .claude/hooks/lead-gate-lib.py --export <slug>` end to end against a real fixture state
   directory: exports the unit's own rows verbatim, including a `recurrence_of_round` value written
   by a real `SubagentStop` payload through `handle_stop` (verified by reading the resulting
   `.jammi/gate-state/<slug>.jsonl` row directly).

Every fixture in `check_rigor_record.py`'s own self-test builds a REAL `origin` + feature-branch
clone pair (`git fetch origin <base>` really resolves `origin/<base>`, exactly as a CI checkout
would) and calls the REAL `run_check()`/`check_allowlist_only_shrinks()`, never a reimplementation —
the same discipline `check_lead_gate.py`'s own G20–G40 arm already established for this hook family.

## Replay evidence against REAL behaviour (not narrated)

**R1–R3 baseline (unchanged by this proposal — carried forward from the prior `docs/plans/
53-agentic-swarm/proposals/` cut on this branch, now superseded in its R4/R5/R6 content but not in
its harness or this evidence):** `ci/hook-acceptance/replay_relays.py --state-dir .jammi/gate-state
--project-dir . --lib .claude/hooks/lead-gate-lib.py` against the real, live `.jammi/gate-state/`:
255 recorded verdict rows, 5 ALLOW under today's R1–R3, none of the 5 carrying a `sweep` field — that
finding is now moot (R4a is dead), kept only as the record of what was checked; full table:
`ci/hook-acceptance/2026-09-11-r4-replay-baseline.log`.

**R7b arming, replayed against REAL merge history — the counter this harness was built for is dead;
what R7 needs replayed is which real PRs would have armed.** `ci/hook-acceptance/replay_relays.py
--project-dir . --rigor-record-ranges ci/scripts/check_rigor_record.py --limit 20` walks the last 20
real two-parent merges into `main`, computes each merge's own `parent1...parent2` (a real,
already-merged `base...head` range — no fetch needed), and evaluates `check_rigor_record.py`'s own
arming/no-op functions against it, read-only. Full table:
`ci/hook-acceptance/2026-09-11-r7-rigor-record-ranges-baseline.log`. Result: **20 of 20 real recent
merges would ARM** under R7b (every one touches `crates/**` and/or `ci/**`); **19 of the 20 carry NO
mechanism-contract file in their own merge range** — expected, since no unit has authored one under
this scheme yet, exactly the migration-shaped bite esc-097 itself described for its own `fix_head`
field ("a relay written to disk before this patch lands… stops being acceptable the moment the patch
IS applied — there is no grandfathering"). The ONE exception, `227ce61` ("merge: main (esc-097
lead-gate proposal) into feat/broker-postgres"), DOES carry a contract file in its own range — this
is the real `esc-097-probe-the-fix.md` proposal merge, correctly detected by the SAME glob match this
check uses for `docs/plans/*/proposals/**`, real confirmation the detector works on a real positive,
not only on fixtures.

Neither the R1–R3 baseline nor the R7b range replay is evidence that any of the flagged rounds was
itself a dodge — every real unit up to today predates `docs/rigor/`, `--export`, and this check
entirely. The evidence is that the mechanism bites on real shapes, real merges, real diffs — not that
it has caught anyone.

## What I could not verify myself, and what I believe may still be wrong (asked for explicitly)

- **The exact scope of "adds or rewrites a mechanism"** (v2's own rule text) is, as written, satisfied
  by the SIMPLER reading I implemented — "touches `crates/**`/`ci/**`/`.github/workflows/**` at all"
  — rather than a narrower "genuinely adds/rewrites, not a one-line value change" reading. v2's own
  "Scope by diff shape only" sentence immediately follows "the rule" and names exactly those three
  roots, which I read as DEFINING the arming predicate operationally, not merely bounding where a
  separate, narrower judgment applies — but the contract never states this equivalence explicitly,
  and I could not find a fixture or a replay case in either KILL or the REFINE verdict that
  disambiguates the two readings. Under my reading, this is deliberately OVER-inclusive (every
  substantive `crates/**` change arms, not only a new/rewritten module), relying on the no-op list
  and the allowlist to carry the exceptions — I believe this is the more defensible, more mechanical
  reading (a semantic "is this really a mechanism rewrite" judgment is not something a diff-shape
  scan can make honestly), but it is MY interpretation, not a value the contract states in code-shaped
  form, and it should be confirmed or overridden before this ships.
- **The release-shaped and human-authored no-op heuristics are mine, not the contract's** — v2 names
  these as categories that "must no-op" but gives no detection algorithm for either. I built the
  narrowest, most literal ones I could defend (a version-bump-only `crates/**` diff; zero commits
  carrying this repo's own `Co-Authored-By: Claude` trailer) and stated their limits in the script's
  own module doc, but neither is exhaustive — a release PR that ALSO edits a gate script is correctly
  NOT exempted by my rule, but a human PR that touches `ci/**` heavily while collaborating with the
  swarm in the same range would still arm (correctly, I believe, since the rule is "does the swarm's
  own rigor process apply to this range", not "is a human involved at all").
- **I did not run the real `ci/scripts/check_ci_guard_wiring.py` against a full copy of the patched
  repository** — building a complete, disk-heavy clone of every `check_*.py`/`test_*.py` this repo
  owns was out of proportion to what this proposal needs to prove, so I verified the ONE property
  that matters by direct inspection instead: `swarm.yml`'s new steps contain `run: python3
  ci/scripts/check_rigor_record.py` as a literal, non-comment line (twice more, with flags), which is
  exactly what `check_ci_guard_wiring.py`'s own comment-stripped-line-scan requires (confirmed by
  reading that script's algorithm directly, `ci/scripts/check_ci_guard_wiring.py:41-57`, and by
  running it, unmodified, read-only, against the REAL primary checkout as a baseline — 67/67 gate
  scripts wired, exit 0). This is a real but indirect check, not a full run against the patched tree.
- **The four cost-floor checks are my own construction from the contract's prose** ("every `path:line`
  … resolves", "introducing commit is an ancestor", "not near-identical by normalized hash", "a
  shrink-only allowlist") — the contract names these properties but not an algorithm; my
  implementations (a `path:line` extraction regex reusing `_probe_path`'s grammar family; `git log
  --follow --diff-filter=A`; a normalized-whitespace sha256 comparison against every other committed
  contract under the same two roots) are reasonable, tested against their own fixtures, but should be
  read as a first cut, not a spec-derived certainty.
- **I believe the "human-authored" no-op is the single largest remaining gap in the DESIGN, not the
  implementation**: it currently exempts a range the instant NO commit carries the trailer, which
  means a human contributor who rebases their own PR onto a branch a swarm session previously touched
  (inheriting an old swarm commit into their own range) would incorrectly ARM the check. This is the
  same class of residual v2 already names for stacked branches ("undetectable from the diff alone")
  — I flag it rather than claim it closed.

## Ledger lifecycle

No escape id is assigned yet. `.jammi/escapes.jsonl` carries no `lead-gate-R7`/`esc-lead-gate-R7` row
as of this doc; one is appended, following the `esc-097-relay-form-satisfied-without-probing-the-fix`
precedent, once a human applies the patches and both self-tests go green on `main`.

## Session-local evidence

Every `scratchpad/…`/`{scratchpad}/tasks/…` path cited above is session-local working state — per
this repo's own `.gitignore`, `scratchpad/` is never tracked and none of it is citable after the
session ends; it is named here only to show how this doc's own numbers were produced. The durable
evidence is: the two `.patch` files under this same directory; `ci/hook-acceptance/replay_relays.py`
and its two baseline logs; and the real `.jammi/gate-state/*`/`.jammi/ledger/*` rows this proposal's
own design-history section cites by exact path and timestamp.
