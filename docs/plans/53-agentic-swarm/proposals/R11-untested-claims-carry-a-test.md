# PROPOSAL (human-merge): tighten(lead-gate) — untested claims carry a test (R11)

Status: PROPOSED. One patch file, proven applying SEQUENTIALLY after `R7-patch1-per-type-
pass-vocabulary.patch`, `R7-patch2-committed-rigor-record.patch`, and
`R7-patch3-open-question-probe.patch` (in that order) against a pristine,
independently-reconstructed copy of `main` (see "Proven", below), and self-test-green
before and after:

- `docs/plans/53-agentic-swarm/proposals/R11-patch1-untested-claims.patch` → `.claude/hooks/
  lead-gate-lib.py`, `.claude/hooks/README.md`, `ci/scripts/check_lead_gate.py`.

`SWARM_GATE_TOUCHED` (`.claude/hooks/**`, `ci/scripts/**/check_*.py`) → the patch, proposal
only, human admin-merge — every file it touches is human-amend-only or itself a swarm gate
definition.

## Applying the patch (human step)

```
git apply --check docs/plans/53-agentic-swarm/proposals/R7-patch1-per-type-pass-vocabulary.patch && \
  git apply docs/plans/53-agentic-swarm/proposals/R7-patch1-per-type-pass-vocabulary.patch
git apply --check docs/plans/53-agentic-swarm/proposals/R7-patch2-committed-rigor-record.patch && \
  git apply docs/plans/53-agentic-swarm/proposals/R7-patch2-committed-rigor-record.patch
git apply --check docs/plans/53-agentic-swarm/proposals/R7-patch3-open-question-probe.patch && \
  git apply docs/plans/53-agentic-swarm/proposals/R7-patch3-open-question-probe.patch
git apply --check docs/plans/53-agentic-swarm/proposals/R11-patch1-untested-claims.patch && \
  git apply docs/plans/53-agentic-swarm/proposals/R11-patch1-untested-claims.patch
python3 ci/scripts/check_lead_gate.py --self-test
python3 ci/scripts/check_rigor_record.py --self-test
```

This patch is ORDER-DEPENDENT on all three R7/R10 patches: it edits `_fix_window` (added by
esc-097, already on `main` before any of these patches) at the SAME point R3's own `diff
--name-only` call sits, and it edits `_relay_rejection` immediately after (2b)'s
`open_question` check (patch 3) and R3's probe-path check — both of which must already be
in the file for this patch's own hunks to find their context. It does **not** depend on
patch 2 (`check_rigor_record.py`/the committed rigor record) for its OWN mechanism — the
hook-level DENY this proposal adds works independent of whether a rigor record is ever
committed — but it is proven applying after all three in sequence because that is the
order this program's own patches ship in, and because `check_rigor_record.py --self-test`
must still pass unaffected (this patch touches no file that script imports or reads).

## The diagnosis this rule comes from

`{scratchpad}/CONTRACT-RULES.md` (session-local; cited here only for provenance — never
re-copied, per this program's own DRY discipline) is a retrospective written after three
units failed the SAME way in consecutive rounds: an implementer's fix asserts a CLAIM in
prose — "this cannot be driven from a test", "there is no injection point", "routes through
the safe path", "its only in-tree callers are X and Y", "re-verified" — and the claim is
false, refuted by an auditor who tried instead of argued, usually in seconds, usually with
a technique already in the tree. Four shapes cost a full audit round each:

1. **A caller-set claim** ("its only in-tree caller is…") that was actually THREE callers,
   the two omitted being the destructive ones (R-H).
2. **A re-verification** labelled as such that was wrong when written (R-A's own
   `_PASS_LIKE` spelling-collision incident, and the generic "verified"/"re-verified" shape
   R7d's per-type-vocabulary fix independently found and closed for a DIFFERENT reason —
   the CLAIM shape this rule catches is broader than that one spelling bug).
3. **A determinant declared impossible to drive** — an auditor wrote the test in about
   forty lines using a pattern already in the tree and drove it in 2.03 seconds (R-A).
4. **A universal about every lease** that was false on the arm whose defining property is
   that nothing is established (R-I).

The existing lead-gate rules (R1 coverage, R2 adjacent probing, R3 probe-the-fix, R10 open
question) gate the SHAPE of a dispatch — sites, counts, an explicit unclosed question — and
they have caught the lead three times in this program (R-A/R-F/R-H's own retrospective
entries are examples of exactly that). What none of them do is examine a CLAIM the fix's
own diff asserts. R1's `sites`/`class_enumeration` coverage is the closest existing
mechanism in shape — an auditor-derived enumeration the relay must cover — but it is the
AUDITOR's own enumeration of SITES, never a claim the FIX itself makes. This rule is that
same coverage discipline, turned onto the fix's own diff, with the enumeration derived by
the HOOK instead of asserted by anyone.

**The rule:** a relay may not be accepted while the fix's own diff contains an untested
claim of a testable shape. For each claim-shaped line the hook finds ADDED by the fix, the
relay carries either a test — the command that tests it and the hash of that command's
output, which the hook re-executes and compares — or an explicit UNCOVERED marking with a
reason. The obligation is DERIVED BY THE HOOK from the diff, never declared by the lead —
a lead who under-enumerates is exactly the failure being fixed, and a hook that trusted the
lead's own list of claims would reproduce R-H's exact defect (a hand-written sweep that
kept missing members) one level up.

## Design decisions, made and justified

### Which sentence family the hook greps for

The phrase family is the exact, finite set this program's own retrospective names, not an
invented one: **impossibility/unreachability** ("cannot be driven", "no injection point",
"is unreachable", "no caller can", "not producer-driven", "no reliable way to force", "is
not established", "structurally unreachable" — R-A and R-F, verbatim), **mechanism claims**
("is safe because", "routes through" — R-A and the diagnosis's own second example),
**caller-SET totality** ("its only in-tree caller", "its one in-tree caller", "every caller
is" — R-H, verbatim), and **completed re-verification** ("re-verified" — the diagnosis's
own second example). Fourteen phrases, `.claude/hooks/lead-gate-lib.py`'s own `_CLAIM_
PHRASES` tuple.

**A naive version was tried and measured, not merely imagined.** R-I's own tell for the
fourth diagnosed shape is a bare quantifier — "the sentence contains 'every', 'only', or
'always' about an outcome whose type exists to represent uncertainty." A bare `grep -ciE
'\b(every|only|always)\b'` was run directly against this repo: **91 hits in
`.claude/hooks/lead-gate-lib.py` alone**, **46 in `.claude/hooks/README.md`**, and **1291
across `ci/scripts/*.py`** combined — this hook's own prose ("ARMED UNCONDITIONALLY", "every
invocation", "only when") is exactly the kind of legitimate, hedged engineering prose this
naive form cannot distinguish from a claim. R-I's own shape is therefore **deliberately NOT
shipped** in this patch — a known, stated residual (see "What this does NOT do", below), not
an oversight; mechanizing "an outcome whose type exists to represent uncertainty" would
require the hook to understand a TYPE's own semantics, the same non-mechanizable judgment
`check_rigor_record.py`'s own module doc already declines for contract QUALITY ("nothing
mechanical can tell a good design pass from a weak one").

**Precision comes from TWO things together, and both are measured, not asserted:**
multi-word, space-delimited phrases (never a bare word, so an identifier like
`is_unreachable_state()` can never match — a phrase requires a literal space between two of
its words, and identifiers never contain one), and restricting matches to **PROSE LINES**
— reusing the exact WHERE-scoping TECHNIQUE `ci/scripts/check_doc_numbers_have_producers.py`
already established for precision (doc comments and whole-prose files, never bare code) —
the CONCEPT, not the file: this hook must stay dependency-free and must never import a
CI-only script, so `_is_prose_line` is a small, self-contained reimplementation of the same
idea (a per-extension single-line-comment marker, or "the whole line" for `.md`/`.mdx`/
`.rst`/`.txt`). Measured: the fourteen-phrase family, run UNSCOPED (no prose restriction at
all) against this repo's entire `ci/scripts` + `.claude/hooks` + `.claude/agents` combined,
returns **26 hits** — a tractable number, every one of them inspected by hand (see the raw
list in "Proven", below): several are legitimate DOCUMENTATION of this exact rule family
(`.claude/agents/lead.md`, `citation-checker.md`) rather than a violation in fix code — a
real, honestly-stated false-positive class (see "What this does NOT do").

### Whether the claim's test command should be re-executed by the hook

**Yes — the hook re-executes it**, using the exact hardened `Popen`-into-its-own-
process-group + `tempfile.TemporaryFile()` + bounded-`wait()` shape `_run_git` already
established (`.claude/hooks/lead-gate-lib.py:656` region) for the identical
escaped-grandchild reason, duplicated rather than shared with `_run_git` itself — R3 is an
adversarially-hardened, heavily-reproduced code path (G20-G40 across four rounds), and this
patch does not touch it. The command runs `/bin/sh -c <command>` sharing R3's OWN
per-decision git deadline (`_new_git_deadline()`, now minted ONCE by `_relay_rejection` and
threaded into both `_fix_window` and the claim-command loop — never a second, separate
budget: the whole arm, including claim-command execution, stays bounded by `_GIT_BUDGET_S`
total, never `_GIT_BUDGET_S` per arm). `sha256(f"{returncode}\n{stdout}")` is compared
against the relay's own recorded `output_hash`; a mismatch DENIES, naming that the claim is
not established.

**Before execution, the command is checked against a write-verb denylist — this is a
SECOND attempt at a mechanism a design round already killed once, and it is built to fix
BOTH of that round's own measured failures, not just avoid repeating the more visible one.**
R4a (a DIFFERENT, killed proposal — a verifiable-sweep predicate, `R7-committed-rigor-
record.md`'s own "Design history" section) measured, directly: of 7 plausible legitimate
sweep commands, 4 were wrongly DENIED because its denylist checked for the SUBSTRING `rm`
(`git grep -n 'transform'`/`'confirm'` both CONTAIN `rm`), while `curl … | sh`, `git clean
-fdx`, and `git reset --hard` were wrongly ALLOWED. This patch's `_command_denied` fixes
both, and both fixes are reproduced directly (not merely argued), in "Proven" below:
tokens are compared WHOLE, via `shlex.shlex(command, posix=True, punctuation_chars=True)`
(which recognizes shell operators — `|`, `&&`, `;`, `>`, … — as their OWN tokens while still
respecting quoting), so `transform`/`confirm` are never confused with `rm` as a substring;
and every shell-operator-delimited SEGMENT's own first token is checked (never only the
whole command's first word), so a denied program hiding after a pipe (`curl … | sh`) or
inside a nested shell (`bash -c 'rm -rf /'`) is caught, and specific destructive git
subcommands (`push`, `reset`, `clean`, `gc`) are denied outright regardless of flags.

**A deliberate simplification, stated plainly:** the command runs against the worktree's
CURRENT state, never an ephemeral `git worktree add --detach <fix_head>` checkout. If the
worktree has advanced past `fix_head` since the relay was written (further edits), the
re-executed output may reflect a state the lead never actually measured. This is the SAME
trust boundary R3's own `probe` field already carries (a lead can satisfy R3 by pasting a
path out of its own diff without reading it) — extended here rather than closed, because
creating and tearing down an ephemeral worktree inside a `PreToolUse` hook's hot path is a
new operational failure mode (a stray worktree left behind on a crash, a path collision, a
locked ref) for a benefit (testing exactly `fix_head` rather than "the worktree as it
currently stands, usually very close to `fix_head`") that is marginal in the common case —
a re-dispatch happens shortly after the fix commit, before further edits.

### How an honest uncovered marking is distinguished from a lazy one, knowing it cannot be fully

It cannot be fully, and this proposal does not claim otherwise — the same honest limit R10
already states for its own `open_question` field ("a lead can write a hollow open
question"). Two things ARE mechanically enforced, both cost floors, neither a quality
judge: (1) a non-empty `reason` string is required — a lead cannot skip a claim by omission,
because the hook enumerates the claims, never the lead; (2) two or more UNCOVERED reasons in
the SAME relay may not be IDENTICAL after normalization (the exact `_probe_normalize`
zero-width/whitespace stripping R2's own adjacency check already uses, reused directly) —
the honest-limit cousin of `check_rigor_record.py`'s own near-identical-CONTRACT check (the
CONCEPT reused, not the code: a normalized-text collision, never a length/word-count rule,
which would be Goodhart by another name). This closes exactly the templated-copy-paste
dodge a lazy disposition takes (ten claims, one boilerplate sentence) without pretending to
verify that any SINGLE reason is true.

## The residual, stated plainly, not softened

**A lead can discharge a claim with a weak command.** `claims["path:line"]["command"]` can
be `"true"` or `"echo ok"` — any command that runs, is not denylisted, and produces a
reproducible hash satisfies the mechanism. **What it cannot do is skip a claim, because the
hook enumerates them, never the lead** — the obligation is derived from the diff's own
shape, and a relay missing even one hook-derived key is denied outright, naming it. What a
human reads at merge to catch the weak discharge is exactly what `R7-committed-rigor-
record.md`'s own committed-rigor-record mechanism already proposes making legible in a
reviewed diff: `docs/rigor/<unit_slug>.jsonl` (once patch 2 lands) carries this unit's own
verdict rows, and a relay's `claims` object — committed alongside it, or read directly from
the gitignored state at review time before that lands — is prose a human reads, the same
way `open_question`'s own hollow-vs-real distinction is judged by a human today. This
mechanism does not, and cannot, replace that read. It replaces "the claim was never even
named" with "the claim was named, and here is exactly what discharged it" — the same
category of improvement esc-097 made for "the fix was probed" over "the fix was mentioned."

## What this does NOT do

- **It does not catch R-I's bare-quantifier shape** ("every"/"only"/"always" about an
  uncertain outcome) — deliberately, measured as unworkable at 91-1291 hits above. R-I's
  own diagnosis remains a `.claude/agents/lead.md`-level discipline, not a hook, the same
  category `.claude/hooks/README.md` already carries for R3's own "design-before-mechanism"
  substantive rule.
- **It does not distinguish a fix's OWN new claim from a diff that merely DISCUSSES the
  phrase family** — a fix that edits `.claude/agents/lead.md`, `citation-checker.md`, or
  this very proposal doc to talk ABOUT "re-verified" or "routes through" as a CONCEPT will
  have that prose flagged exactly like a real claim, because the regex cannot tell "this
  line USES the pattern to clear a finding" from "this line DESCRIBES the pattern." The
  remedy is the same as everywhere else in this mechanism: an honest `uncovered` marking
  costs nothing (`"reason": "this line documents the rule, not a claim about this fix's own
  code"`), and the human at merge reads whether that reason is true — this is not a new
  category of trust, it is the SAME one (4) already carries for every other uncovered claim.
- **It does not verify a `tested` command is a GOOD test**, only that it is a NAMED one
  whose output is reproducible — see "The residual", above.
- **It does not check out `fix_head` before executing** — see the design-decision section
  above.
- **It does not claim the write-verb denylist is exhaustive.** It closes the two SPECIFIC
  shapes a prior, killed design measured wrongly (substring collision, pipe/nested-shell
  evasion) and adds the destructive git subcommands that incident named — it does not claim
  no other evasion exists, the same honest posture `.claude/hooks/README.md` already states
  for its own now-deleted Bash backstop ("undecidable without jamming legitimate traffic").

## Proven

Built and tested against a from-scratch, byte-for-byte pristine reconstruction of `main`
(`git archive main | tar -x`, never a `git worktree` of this repo, deleted after use), with
patches 1-3 applied first, then this patch, on TWO SEPARATE, independently-reconstructed
copies (never the same tree the patch was authored against, for the second run):

1. **Baseline, before this patch:** patches 1-3 applied to a pristine copy —
   `python3 ci/scripts/check_lead_gate.py --self-test` → **68/68**; `python3 ci/scripts/
   check_rigor_record.py --self-test` → **11/11**.
2. **This patch applied on top** (`git apply --check` then `git apply`, both clean, in
   sequence after all three R7/R10 patches): `python3 -m py_compile` clean on all three
   touched Python files; `check_lead_gate.py --self-test` → **75/75** (68 + 7 new: UC1-UC7);
   `check_rigor_record.py --self-test` → **11/11**, unaffected (this patch touches no file
   that script imports or reads).
3. **Repeated on a SECOND, independently-reconstructed copy** (`git archive main` run a
   second time into a fresh directory, patches 1-3 then this patch applied again from
   scratch): all four `git apply --check`/`git apply` pairs clean, in order;
   `check_lead_gate.py --self-test` → **75/75**; `check_rigor_record.py --self-test` →
   **11/11**.
4. **Non-vacuity, direct:** `_claims_rejection`'s own call in `_relay_rejection` was
   temporarily replaced with `claims_why = None` (the check disabled, nothing else touched)
   and the suite re-run: **UC1, UC3, UC4, UC6 (every fixture that asserts a DENY) turned
   RED**; UC2, UC5, UC7 (the ALLOW fixtures) and every one of the other 68 pre-existing
   fixtures stayed GREEN — proof the four DENY fixtures are real tests of this arm, not
   vacuously passing, and proof this patch does not perturb any pre-existing fixture's own
   pass/fail outcome even under a broken mechanism.
5. **The phrase-family measurement, run for real, not estimated:** `grep -ciE
   '\b(every|only|always)\b' .claude/hooks/lead-gate-lib.py .claude/hooks/README.md` → 91,
   46; the same across `ci/scripts/*.py`, summed → 1291. The fourteen-phrase `_CLAIM_
   PHRASES` family, unscoped, across `ci/scripts` + `.claude/hooks` + `.claude/agents`
   combined → **26 hits**, every one inspected: `re-verified` (8×, one an allowlist
   comment), `routes through` (9×), `is unreachable` (3×), `structurally unreachable` (4×),
   `cannot be driven` (1×), `is safe because` (1×) — none a raw identifier collision,
   confirming the phrase-plus-prose-scoping precision claim directly rather than asserting
   it.
6. **The write-verb denylist, run against R4a's own two measured failure classes
   directly:** `git grep -n 'transform'`/`'confirm'` → ALLOWED (not denied); `curl … | sh`,
   `git reset --hard`, `git clean -fdx`, `bash -c 'rm -rf /'` → DENIED, each naming the
   specific program or subcommand; `printf hello`, `python3 -c "print(1)"`, `git log
   --oneline | head -5` → ALLOWED; an unbalanced quote → DENIED (fails closed, monotone
   toward DENY). Eleven cases, all correct, run directly against `_command_denied` in
   isolation (not merely inferred from the fixtures).
7. **The diff-parser's own hardening, run against the exact self-referential edge case this
   repo's own convention creates:** a synthetic `git diff -U0` payload was constructed
   where a fix adds BOTH a real claim-shaped line in `bar.py` AND a brand-new `.patch` file
   (this repo's own `docs/plans/*/proposals/*.patch` convention) whose OWN embedded content
   contains literal `+++`/`---`/`@@` diff-header-shaped text. `_parse_claim_sites` returned
   exactly `{"bar.py:2": "# this cannot be driven from a test"}` — the embedded patch
   file's own diff-shaped text did not create a phantom second file boundary or corrupt
   `bar.py`'s own line numbering, confirming the `diff --git`-anchored state machine (never
   a bare `+++`-line scan) was worth its extra ~10 lines over the naive version.

## What I could not verify myself, and what I believe may still be wrong (asked for explicitly)

- **The fourteen-phrase family is a first cut from the retrospective's own text, not a
  closed set.** A future round's own claim-shaped failure may use different words entirely
  ("must be true because", "guaranteed by construction") that this list does not catch —
  the hook enumerates a FIXED, human-amend-only list (`.claude/hooks/**` is agent-write-
  denied by design), so widening it is a human decision on a future PR, never something
  this mechanism grows on its own. I believe this is correct (an autonomously-growing
  phrase list is exactly the kind of self-tightening this program's own anti-Goodhart
  principle reserves for a human), but it means the list's OWN completeness is not
  something this proposal can prove, only measure against the corpus it had.
- **I did not measure this mechanism against the SAME real historical wave the task's own
  diagnosis cites** (the "26 claim-shaped lines... at least four of the blocks that cost a
  full audit round" figure) — that wave's own fix commits are session-local
  (`{scratchpad}/tasks/…`) and not citable after this session ends, the same limit
  `R7-committed-rigor-record.md`'s own "Session-local evidence" section already states for
  its own citations. The 26-hit figure this proposal DOES cite (measured against this
  repo's `ci/scripts`/`.claude/hooks`/`.claude/agents` today) is a DIFFERENT corpus that
  happens to produce a similar-looking number — I state this explicitly so the two 26s are
  never conflated as the same measurement.
- **The per-extension comment-marker map (`_LINE_COMMENT_MARKERS`) is a first cut, not
  exhaustive** — an unrecognized extension defaults to trying BOTH `#` and `//`, which is
  the right FAIL-OPEN-TOWARD-SCANNING direction (never silently exempting a whole language)
  but could still miss a real claim written in, say, a `.proto`/`.sql`/`.hcl` file using a
  different single-line-comment marker (`--`, for SQL) that is not in the map and does not
  start with `#`/`//` either — a real, narrow gap I did not close, since this repo's own
  fix commits are overwhelmingly Rust/Python/Markdown/shell/YAML/TOML.

## Ledger lifecycle

No escape id is assigned yet. `.jammi/escapes.jsonl` carries no `lead-gate-R11`/
`esc-lead-gate-R11` row as of this doc; one is appended, following the `esc-097`/
`esc-lead-gate-R10` precedent, once a human applies this patch (after `R7-patch1`,
`R7-patch2`, `R7-patch3`) and every self-test goes green on `main`.
