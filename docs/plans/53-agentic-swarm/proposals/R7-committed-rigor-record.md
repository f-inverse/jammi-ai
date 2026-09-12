# PROPOSAL (human-merge): tighten(lead-gate) — a committed rigor record, armed by the diff's own shape (R7); one relay probe entry must be an open question (R10)

Status: PROPOSED. Five designs have gone through this problem — R4/R5 were KILLED, R6 was KILLED, R7
came back REFINE with binding corrections (v2, below the first `---`; where v1 and v2 differ, v2
wins — this doc states the v2 design directly and cites v1 only where the design history matters),
R9 was KILLED (see "R10 — the one thing in this corpus that worked ahead of time", below), and R10
came back REFINE, the smallest surviving thing. Three patch files, proven applying and
self-test-green against a pristine, independently-reconstructed copy of the real tree (see "Proven",
below):

- `docs/plans/53-agentic-swarm/proposals/R7-patch1-per-type-pass-vocabulary.patch` → `.claude/hooks/
  lead-gate-lib.py`, `.claude/hooks/README.md`, `ci/scripts/check_lead_gate.py`.
- `docs/plans/53-agentic-swarm/proposals/R7-patch2-committed-rigor-record.patch` → `.claude/hooks/
  lead-gate-lib.py`, `.claude/agents/adversarial-audit.md`, `.github/workflows/swarm.yml`,
  `ci/scripts/check_rigor_record.py` (new), `ci/scripts/rigor_record_allowlist.txt` (new).
- `docs/plans/53-agentic-swarm/proposals/R7-patch3-open-question-probe.patch` → `.claude/hooks/
  lead-gate-lib.py`, `.claude/hooks/README.md`, `ci/scripts/check_lead_gate.py`. Applies AFTER patch 1
  and patch 2, in that order (proven sequentially, below) — the smallest of the three, touching only
  `_relay_rejection`'s R2 block, two docstring sentences, and the harness's own relay-fixture helper.

`SWARM_GATE_TOUCHED` (`.claude/hooks/**`, `.claude/agents/*.md`, `ci/scripts/**/check_*.py`,
`.github/workflows/swarm.yml`) → all three patches, proposal only, human admin-merge — every file any
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

git apply --check docs/plans/53-agentic-swarm/proposals/R7-patch3-open-question-probe.patch && \
  git apply docs/plans/53-agentic-swarm/proposals/R7-patch3-open-question-probe.patch
python3 ci/scripts/check_lead_gate.py --self-test
```

Patches 1 and 2 apply independently and in either order against each other's OWN file set (patch 1
never touches a file patch 2 also touches except `lead-gate-lib.py`, where the two edit disjoint
regions — proven applying sequentially, patch 1 then patch 2, in that order, below). Patch 3 is
ORDER-DEPENDENT on both: it is built against, and proven applying only against, the tree AFTER patch
1 and patch 2 already landed (its own `_relay_rejection` edit sits between R2's block, which patch 1
and patch 2 leave untouched, and R3's — proven sequentially, patch 1 then patch 2 then patch 3, in
that order, below). Ship patch 1 now, patch 2 next, patch 3 last, per the binding sequencing below.

## Design history — five rounds, cited by artifact, not narrated

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

**R9 — KILLED, including a case the round caught against the lead itself.** R9 proposed a
re-executable, verifier-authored sweep predicate; the design round's own verdict killed it on three
independent measurements, one of which corrected a claim this session had relayed as measured when it
was fitted. The predicate credited with finding a site "one round early" was round 4's OWN sweep
instrument, replayed at round 4's OWN audit head — the round that would actually have had to author it
never mentions the term: its five real patterns yield 54, 45, 123, 20, and 444 matches at that same
head and NONE reaches the site; the honest, non-hindsight construction hands the lead about 686
dispositions and still misses. The showcase was also MISDIAGNOSED: the class was named prospectively —
by the LEAD, in an earlier round's own relay, as the one site it could not close by reading, with an
explicit "attack it" — and the next round attacked it and closed it, which is the hand-off chain
working, not a proactivity failure the predicate caught. Three further measurements killed the
mechanism independently: the jam is the MEDIAN (executing 101 real verifier-authored patterns at their
own recorded fix heads gives a median of 25 matches whole-tree and 10 scoped, 31% over 25 even when
scoped; 13 of 101 are FATAL under the extended-regex the design demanded and 5 more silently match
nothing); and the theory of change is wrong (over 45 relay-carrying consecutive pairs, 67% of the next
round's findings land in a file the relay ALREADY dispositioned and only 6% repeat a site — the gap is
DEPTH per site, not breadth, and R9 multiplied breadth tenfold while leaving depth alone; median
disposition length falls from 180 characters at eight sites or fewer to 45 at thirty-four sites, and
14% of all live dispositions are under forty characters).

**R10 — REFINE, the one thing in this corpus that worked ahead of time.** Every relay already carries,
per R2, at least two probe sites "examined and found clean." The single best hand-off in the entire
corpus was not one of those — it was a probe entry that said: here is a site I examined and could NOT
close, here is the attack; the next round took it and closed it (the same hand-off R9's showcase
misdiagnosed as a predicate's finding, above). **The rule:** alongside the two examined-clean sites R2
already requires, at least ONE probe entry must be an OPEN QUESTION — a site examined and explicitly
not closed, carrying the attack to run. The relay schema gains this distinction; R2's own count
requirement is otherwise unchanged. This is PROSPECTIVE — checked when the relay is written, before
the next verifier exists, changing what the lead must produce at the moment of re-dispatch (the user's
own bar), rather than annotating the round after. It does not JAM: one line of relay text, no
subprocess, no regex, no new field on the verifier's write path, no match count, no cap, no shadow
period — nothing scales with the size of the class (patch 3, below, is the smallest of the three
patches this proposal ships). It creates no bad incentive: it cannot be satisfied by narrowing the
verifier's brief, because it is about the LEAD's own examination, never the verifier's scope — R9's
worst residual (a narrower round-one audit becomes cheaper for the lead) does not arise here.

**The residual, stated and never implied away.** This is a SCHEMA requirement over lead-authored
text. It is a cost floor, not proof of examination — the same limit the hook already records for its
existing probe rule. A lead can write a hollow open question. What it cannot do is write nothing and
pass, and what the human reads at merge is whether the open questions were real.

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

## What patch 3 (R10) is

**The mechanism.** `_relay_rejection`'s R2 block (`.claude/hooks/lead-gate-lib.py`) already requires
`probe` to name ≥2 distinct sites outside the enumeration/findings. Patch 3 adds, immediately after
that block and immediately before R3, ONE further ARMED-ALWAYS check: `data.get("open_question")`
must be a non-empty string. That is the entire mechanism — no adjacency check against `probe` or the
enumeration, no format requirement beyond non-empty, no cap. R2's own ≥2-distinct-non-reactive
counting is untouched, byte-for-byte, by this patch — a decision recorded in the diff itself: the new
check reads a field R2 never looks at, so a regression in one cannot silently mask a regression in the
other. The module docstring's "Arm order" sentence and its `esc-064` summary paragraph gain one
clause each naming the new arm; `.claude/hooks/README.md` gains one new numbered sub-point, (2b),
between (2) Proactivity and (3) Probe-the-fix, restating the exact same limit in the human-facing doc.

**The harness.** `ci/scripts/check_lead_gate.py`'s `_write_relay_exact` test helper gains one new
keyword parameter, `open_question`, defaulting to a fixed, non-empty placeholder string
(`_DEFAULT_OPEN_QUESTION`) — so every one of the ~30 EXISTING call sites, none of which passes this
kwarg, keeps satisfying the new always-armed requirement without a single one of them being edited.
This is the identical backward-compatible shape `fix_head`'s own `None`-means-omitted default already
established in this same helper. Two new fixtures exercise the arm directly: `OQ1` writes a relay that
already satisfies R1+R2+R3 in full but passes `open_question=None` (explicitly suppressing the
default) and asserts DENY, naming the missing field; `OQ2` is the identical relay with a real
`open_question` string and asserts ALLOW. Both are non-vacuous — confirmed directly by temporarily
removing the new check and observing `OQ1` turn RED (`check-lead-gate[OQ1]: FAIL — ... got 0`) while
every other fixture stayed unaffected, then restoring it.

**What this patch does NOT do**, stated as plainly as the mechanism itself: it does not check that the
named open question is adjacent to anything, that it differs from any `probe` entry, that it names a
real file, or that its prose is substantive — any non-empty string satisfies it, including a copied
placeholder. This is not an oversight; it is the same limit (2)'s own ≥2-clean-sites rule already
carries, restated rather than hidden. See "The residual, stated and never implied away," above.

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
5. Patch 1 THEN patch 2 THEN patch 3, applied sequentially to a THIRD, independently-reconstructed
   pristine copy: all three `git apply --check`/`git apply` pairs clean, in that order (patch 3's own
   `index` lines chain from patch 2's post-apply blob hashes, confirmed by hash, not merely by
   re-running the check); `python3 -m py_compile` clean; `check_lead_gate.py --self-test` →
   **68/68** (66 + `OQ1` + `OQ2`); `check_rigor_record.py --self-test` → **11/11**, unaffected (patch 3
   touches no file that script imports or reads). `OQ1`/`OQ2` confirmed non-vacuous by temporarily
   deleting patch 3's own check and re-running: `OQ1` alone turns RED, every other fixture (including
   every pre-existing ALLOW fixture that never passes `open_question`) stays GREEN — proof the
   backward-compatible default in `_write_relay_exact` does not also hide a vacuous new assertion.

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

No escape id is assigned yet. `.jammi/escapes.jsonl` carries no `lead-gate-R7`/`esc-lead-gate-R7`/
`esc-lead-gate-R10` row as of this doc; one is appended, following the
`esc-097-relay-form-satisfied-without-probing-the-fix` precedent, once a human applies all three
patches and every self-test goes green on `main`.

## Session-local evidence

Every `scratchpad/…`/`{scratchpad}/tasks/…` path cited above is session-local working state — per
this repo's own `.gitignore`, `scratchpad/` is never tracked and none of it is citable after the
session ends; it is named here only to show how this doc's own numbers were produced. The durable
evidence is: the three `.patch` files under this same directory; `ci/hook-acceptance/replay_relays.py`
and its two baseline logs; and the real `.jammi/gate-state/*`/`.jammi/ledger/*` rows this proposal's
own design-history section cites by exact path and timestamp. Patch 3 (R10) adds no new replay
evidence of its own — it is a pure schema-presence check with no arming condition to replay against
real merge history (unlike R7b/patch 2); its own evidence is exclusively "Proven" item 5, above.
