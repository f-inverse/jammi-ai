#!/usr/bin/env python3
"""lead-gate-lib.py — the lead-proactivity gate's shared mechanism (v3, core).

Round 1 and round 2 both found the SAME class: a predicate over FREE TEXT
(site regexes, worktree/sha/slug token scans, write-verb walks, tag scans)
whose input domain is unbounded, so every fix moved the squeeze between
"jams legitimate traffic" and "dodged by a rewording". v3 removes free-text
parsing from the DENY decision entirely and narrows the gate to ONE choke
point (a scope cut from round-3's original wider design, made under a
usage-limit constraint — see the commit message for what was dropped and
why the remaining design still closes the expensive loop, F10):

  THE ONE GATE (§3): a same-`agent_type` VERIFIER `Agent` dispatch whose
  prompt names, as a WHOLE TOKEN (never a raw substring — audit-r3 finding
  1: `ci/gpu` must not gate `ci/gpu-dev`), an open BLOCK's recorded
  `worktree` (or a path under it), `head_sha` (full, or any >=7-char prefix
  — this repo's short-sha convention), or exact `unit_branch` is DENIED
  unless an
  ACCEPTED RELAY ARTIFACT exists for that (unit, agent_type, block_ts). A
  first dispatch of any agent_type is never gated (no prior row exists to
  match). The relay artifact (`.jammi/gate-state/<slug>.relay.<agent_type>.
  <block_ts>.json`) is written by the LEAD directly (Write is not gated);
  the hook only ever READS it fresh, every time — never appends an
  "accepted" row itself, so there is no phantom-acceptance state to corrupt. The
  relay's requirements are a CONJUNCTION (esc-064): coverage (`sites`) is
  armed by a non-empty `class_enumeration` — the data, never the recorded
  `enumeration_missing` flag — and adjacent probing (`probe`, >=2 distinct
  sites outside enumeration+findings) is armed ALWAYS. esc-lead-gate-R10's
  own `open_question` field (a schema cost floor over lead-authored text,
  never an executed check) is RETIRED — esc-lead-gate-R12 (below) replaces
  it with a mechanized, RE-EXECUTED pre-fix anticipation artifact plus a
  post-fix differential, because `open_question` could be satisfied by
  merely NAMING an attack "for the next round to run," never running one.

esc-097 (R3, "PROBE THE FIX"): the relay gains ONE further requirement,
`fix_head` (a full sha, `re.fullmatch(r"[0-9a-f]{7,40}")`) plus a `probe`
entry naming a file the fix actually changed. Armed ONLY from
`_decide_verifier_dispatch`'s own repeat-dispatch branch — never on a FIRST
dispatch (no prior row exists to reach this arm) — and, per decision, for AT
MOST ONE targeted unit even when a prompt whole-token-names more than one
open BLOCK of the SAME type; every OTHER targeted unit named in that same
prompt DENIES outright, naming it, rather than silently skipping its own
R3 (a prompt naming more than one open BLOCK must be split into separate
dispatches — see `_decide_verifier_dispatch`). There is exactly ONE
predicate, `_relay_rejection`, and exactly ONE caller that ever runs it:
the repeat-dispatch branch above. AN ADVERSARIAL-AUDIT BLOCK IS CLOSED ONLY
BY A LATER ADVERSARIAL-AUDIT PASS ON THE SAME UNIT (which itself requires a
relay that passes R1+R2+R3 on that direct path) OR BY THE DOCUMENTED `rm`
ESCAPE HATCH — there is no cross-type clearing arm; an earlier draft's
`_adversarial_audit_cleared_by_verifier_pass` (a `fix-verifier`/
`acceptance-verifier` PASS clearing an older `adversarial-audit` BLOCK
without ever checking R3) is deleted, not merely tightened — a relay
lacking `fix_head` entirely could satisfy that arm, because it never ran
git at all. Reachability is bound in TWO parts (V18 — round-3 closure):
the relay's own `unit_branch` field must first `slugify()` to EXACTLY the
`unit_slug` this BLOCK's own state file is already filed under — the
identity the dispatch itself resolved, never an arbitrary branch the relay
merely asserts (this binds the NAME); THEN that same `unit_branch` is
resolved UNDER `refs/heads/` ONLY (`git rev-parse --verify --end-of-options
refs/heads/<name>^{commit}`; a name that merely slugifies to the right
string but names no real branch fails here too, closing the
slugify-collision advisory), and `fix_head` must `git merge-base
--is-ancestor --end-of-options fix_head <resolved tip>` (this binds
`fix_head`'s POSITION on that same history — the NAME check alone does
not: an amended-away, orphaned sha, or a sha that is a real commit on some
OTHER branch, both slugify-match a unit_branch that names the right unit
while never being reachable from its tip). An amended commit IS on the tip
and allows; only a STALE relay naming the pre-amend sha is denied —
"fix_head <sha> is not on <unit_branch>; if the fix was amended, name the
amended sha; if it was committed on a child branch, commit or merge it
onto <unit_branch>". A BLOCK row filed under the `UNBOUND` fallback bucket
can never be satisfied by any relay (no real branch name slugifies to the
literal string `UNBOUND`); the remedy is to re-dispatch naming the unit so
the verdict lands on the unit's own file, then hand-remove the stale row
for this block from `UNBOUND.jsonl` (never `rm` the shared file — it holds
every other unit's UNBOUND rows too).

ROUND-4 (two adversarial reproducers, both closed structurally): (1)
resolving `unit_branch` as a BARE `<name>^{commit}` — rather than
explicitly under `refs/heads/` — lets a TAG (or any other
`refs/<kind>/<name>`) literally named like the branch win
gitrevisions(7)'s own refs/tags-before-refs/heads disambiguation and
shadow the real branch's tip with the tag's own target (G39); the fix
resolves `refs/heads/<name>^{commit}` explicitly, so nothing but a real
branch of that name is ever consulted — the documented cost is that a unit
whose worktree is on a DETACHED HEAD has no `refs/heads/` entry to bind to
and can never be relayed this way. (2) `block_sha` and `fix_head` are
sha-shaped, not branch names, so a ref (branch or tag) literally NAMED
like one of them — or like a short PREFIX of one — can shadow the object
it abbreviates the identical way (G40); both are now resolved via `git
rev-parse --verify --end-of-options <hex>^{commit}` and the resolved,
FULL 40-hex object is required to START WITH the caller-supplied hex,
never merely to resolve to SOME commit — the RESOLVED, full sha is what
every later git argv (`merge-base`, `diff`) actually uses. Either check
failing DENIES, naming which hex resolved to what, and that a ref named
like a sha shadows it. On the ACCEPT path, if any of these calls still
wrote to stderr despite succeeding (e.g. git's own `warning: refname
'...' is ambiguous.` when a shadow happened to resolve to a
prefix-matching, correct object anyway), that text is appended to the
operator-facing ALLOW reason (and the `hook.log` row it is written into),
so shadowing stays visible even when it did not change the outcome.

Arm order: artifact-exists -> schema -> R1 (coverage) -> R2 (proactivity,
always) -> CLAUDE_PROJECT_DIR is set ->
`row.head_sha` ("block_sha")
matches the sha shape and resolves (STARTS WITH the given hex) -> `fix_head`
matches the sha shape, differs from block_sha, and resolves (STARTS WITH
the given hex) -> relay `unit_branch` slugifies to this BLOCK's own
`unit_slug` -> that `unit_branch` resolves under `refs/heads/` -> `fix_head`
is an ancestor of `unit_branch`'s resolved tip -> R3 (>=1 probe path names
a file the fix actually changed, per `git diff --name-only -z block_sha
fix_head` — TRUSTED and COMPUTED, never lead-supplied) -> R11 (esc-lead-
gate-R11, "untested claims carry a test": every claim-shaped line the
fix's OWN diff adds, per `git diff -U0 block_sha fix_head` — the hook's
OWN derived enumeration, `_parse_claim_sites` — carries a `claims`
disposition: TESTED, with a re-executed, hash-matching command, or
explicitly UNCOVERED with a reason). git runs ONLY here, ONLY in
`$CLAUDE_PROJECT_DIR` (required explicitly), SIX calls total per decision
(`rev-parse --verify` block_sha, `rev-parse --verify` fix_head,
`rev-parse --verify --end-of-options refs/heads/...` unit_branch,
`merge-base --is-ancestor`, `diff --name-only`, and — esc-lead-gate-R11 —
`diff -U0`), every invocation carrying `--end-of-options` immediately
before its revision arguments (git >= 2.24) so a value shaped like an
option (e.g. `--output=/tmp/x`) can never be read as one, ALL SIX sharing
ONE per-decision monotonic deadline (`_GIT_BUDGET_S = 5.0`, an absolute
`time.monotonic()` value threaded through every `_run_git` call, never a
fresh 5s per call — the whole arm, INCLUDING R11's own claim-command
re-execution below, is bounded by 5s total, not 6x5s)
(`Popen` into its own process group, `Popen.wait(timeout=<time left on the
shared deadline>)` — never `.communicate()`, never `with Popen(...)` —
stdout/stderr captured to `tempfile.TemporaryFile()`s, BOTH read back only
after `wait()` returns, so a failing call's stderr text is appended to the
deny reason, and a SUCCEEDING call's stderr text — round-4, above — is
appended to the accept-side reason instead; on timeout, `os.killpg` the
WHOLE group, then a bounded second `wait(timeout=1)` — this reap tail is a
flat 1s, NOT drawn from the shared budget — then close the files and
`.kill()` if even that does not return); any git failure (non-zero exit,
timeout, budget exhaustion, an unresolvable/malformed sha, or a resolved
sha that does not start with the hex given) DENIES, naming the failing
command and its stderr, and states that `rm .jammi/gate-state/<slug>.*` is
the escape hatch but destroys the unit's evidence rather than fixing the
underlying git problem. HONEST LIMIT: R3 cannot verify examination, only that a named
path is a real member of the fix's own diff — a lead can still satisfy it
by pasting a path out of its own change; the substantive rule
(design-before-mechanism, one fix round per BLOCK) is a
`.claude/agents/lead.md` discipline, not a hook. A PRE-EXISTING wrinkle
this arm never introduced is now CLOSED (esc-lead-gate-R7d): the PASS
vocabulary used to be checked as one GLOBAL set (`_PASS_LIKE`), so an
adversarial-audit row whose OWN verdict text was literally "verified" or
"PROCEED" — words that clear a DIFFERENT card's row, never this one's —
also closed a BLOCK, independent of R3 entirely. `_pass_word_for(agent_type)`
(below) replaces the pooled set with a PER-AGENT-TYPE map, so a raw value
only clears when it is that type's OWN card spelling.

esc-lead-gate-R11 ("UNTESTED CLAIMS CARRY A TEST"): the retrospective this
arm formalises (`{scratchpad}/CONTRACT-RULES.md`, session-local, cited only
for provenance) named the SAME failure four times running — a caller-set
claim that was false on its destructive callers, a "re-verified" claim that
was wrong when written, an impossibility claim an auditor refuted in two
seconds, and a universal about an uncertain outcome — none of them a wrong
SITE; each a CLAIM the fix's own diff asserted and never tested. The
obligation is DERIVED BY THE HOOK from the diff's own shape (`_parse_claim_
sites`, run from `_fix_window`, sharing its git deadline), never declared by
the lead — exactly the posture R1 already takes toward the auditor's own
`class_enumeration`, here applied to the lead's OWN fix. HONEST LIMIT,
stated as plainly as R3's own: this cannot verify that a `tested` claim's
command is a GOOD test, only that it is a NAMED one whose recorded output
hash REPRODUCES under fresh execution, and it cannot verify that an
`uncovered` reason is TRUE, only that it is present and not a copy-paste
duplicate of another uncovered reason in the same relay — a lead can still
discharge a real claim with a weak command; what it cannot do is skip a
claim, because the hook enumerates them, never the lead. See the proposal
doc for the full design history and the residual, stated without softening.

esc-lead-gate-R12 v2 ("ANTICIPATE BEFORE THE FIX", fix round 1, M1'-M6'):
esc-lead-gate-R10's `open_question` was checked ONLY on the RELAY — written
after a fix commit already exists — so it was fillable after the fact (v1
of this rule, which attached every obligation to that same post-fix relay,
was KILLED by its own pressure round for the identical reason; the FIRST
v2 implementation keyed the pre-fix artifact by a single block's `ts` and
required the tip to equal that block's OWN `head_sha`, which denies
PERMANENTLY the moment two open second-round BLOCKs of different types sit
at different shas — fix round 1 replaces that shape). v2 moves the
load-bearing check to the PRE-FIX moment: `_decide_implementer_dispatch`
(READER 1) denies a dispatch onto a unit with ANY open verifier-type BLOCK
unless a `.jammi/gate-state/<slug>.anticipation.<tip_sha>.json` artifact
exists AT THE BRANCH'S CURRENT TIP covering the UNION of every such open
block's own derived keys, reduced to one entry PER FILE, whose commands the
hook re-executes and hash-matches — and whose ordering evidence is: the
tip has not moved since the decision began, the resolved worktree's own
HEAD equals it, and the tree is clean. `_relay_rejection` (READER 2) then
requires `attacks_post`, the SAME per-file keys (widened by every file the
fix changed or added a new surface to) re-run at `fix_head`, and a measured
DIFFERENCE from the pre-fix witness for every fix-changed file the pre-fix
artifact covers — an outcome DERIVED from hash inequality, never lead-
declared. Attack-command execution runs in a COMPUTED cwd (`git worktree
list --porcelain`'s own resolution for the unit branch, never
`$CLAUDE_PROJECT_DIR`) and has its OWN budget (`_ATTACK_BUDGET_S`,
separate from `_GIT_BUDGET_S`, run under its own per-phase deadline —
never inside the git-metadata deadline the ordering checks use).
`ci/scripts/check_rigor_record.py` is READER 3, the half the lead cannot
forge: a committed-record SHAPE check (every second-round BLOCK row's
export carries a matching anticipation record covering every required
file) that HARD FAILS on shape alone; re-execution against a real
`pre_fix_sha` checkout is ADVISORY, never a hard fail (BSD/GNU and path
divergence between the lead's machine and CI, measured). HONEST LIMITS,
stated as plainly as R3's/R11's own: the hook cannot judge that an attack
is a GOOD attack — only that it ran, at this instant, against a worktree
whose HEAD really was the tip with nothing uncommitted, and that its
measured output moved after the fix; this does NOT establish that no fix
exists off-tree (a stash, another branch or worktree) nor that the attack
necessarily preceded any later fix commit — "true by construction" is not
claimed. A changed file covered by no pre-fix attack key is not
differential-checked by the hook (Reader 3's real checkout is the only
place that gap could close); the decoy-unit-line residual (the hook trusts
the lead's own `unit:` line) is not detected by anything here. See the
proposal doc for the full v1-to-v2 design history and CONTRACT-R12-fix1.md
for the fix-round-1 design verdict this implementation follows.

Explicitly OUT OF SCOPE by this cut (dropped entirely, not log-only):
`SendMessage` gating and all message-prose parsing; the Bash backstop (the
mechanical control is `permissions.deny` on the hook files in
`.claude/settings.json`, unchanged); tell rows beyond the existing one-line
`hook.log` entry every invocation already writes. Implementer-dispatch
binding now covers thirteen agent_types (M6'): the nine domain-implementer
`IMPLEMENTER_TYPES` (a dispatch with NO unit line DENIES, naming the
missing line) and four generic/harness `_R12_EXTRA_GATED_TYPES` —
`general-purpose`/`claude`/`fork`/`doc-updater` — which stay ALLOWED with
no unit line (a lead may dispatch these for reasons unrelated to
implementing a fix) but are gated identically to the nine the moment their
prompt DOES name a unit with an open second-round BLOCK; a `unit:` line
naming a branch that does not resolve under `refs/heads/` DENIES either
way.
Visible-only residuals, each with a runtime tell (a verifier's next round
reproducing the same BLOCK with no relay artifact between it and the
prior one): relaying to a running agent by `SendMessage` (out of scope by
design, not merely undetected); an unlabeled fresh-worktree re-dispatch
(DODGE-5); `disableAllHooks` in local settings.

Two entry points still write state:
  start — SubagentStart (all agent types). Appends an agent_id ->
          unit_branch binding to `.jammi/gate-state/bindings.jsonl`.
          ADVISORY ONLY — used to file an UNPARSEABLE verdict row (§2)
          under the right unit, never for a gate decision. Never denies.
  stop  — SubagentStop (matcher: the verifier agent types). The verdict
          JSON is the LAST fenced ```json block of the message whose
          object has `"kind": "verdict"`; `<verdict>…</verdict>` tags are
          accepted as a one-release fallback when no fenced block exists.
          Both paths hand off to the SAME JSON-string-aware object
          extractor starting right AFTER the opening marker — neither path
          ever searches for a CLOSING marker, so a `</verdict>` or a stray
          `}` sitting inside a quoted `notes` string can never truncate the
          region early (round-2 finding 6). An UNPARSEABLE row is filed
          under the agent's OWN SubagentStart binding, never under a
          shared "UNBOUND" bucket, unless no binding exists either. Never
          denies.
  pre   — PreToolUse (matcher: `Agent|Task`). THE decider — exit 0 = allow,
          exit 2 + stderr = deny. Every other `tool_name` (including
          `SendMessage`/`Bash`) is a pass-through allow; this hook does not
          gate them.

Fail-closed doctrine: never exit 1. Every internal error in `pre` exits 2
with a reason (`sys.stdin.buffer.read()` + decode + JSON parse are ALL
inside the try/except boundary — a UTF-8-valid but JSON-invalid payload, an
empty payload, and a non-object payload each deny, never coerce to `{}` and
allow; no `errors="replace"` fallback). `start`/`stop` are best-effort
writers that never block a subagent lifecycle event and always exit 0.

No git subprocess ANYWHERE except the relay-validation arm added by esc-097
(R3) and esc-lead-gate-R12's two readers — reader 1
(`_decide_implementer_dispatch` -> `_pre_fix_anticipation_rejection`), armed
ONLY when an implementer-type dispatch names a unit branch with an open
verifier-type BLOCK, and reader 2 (inside `_relay_rejection`, alongside R3)
— never on the hot, ungated first-dispatch path, never outside
`$CLAUDE_PROJECT_DIR` for git METADATA calls (`rev-parse`, `worktree
list`; the ATTACK COMMANDS themselves run in the separately resolved
worktree cwd, never `$CLAUDE_PROJECT_DIR` — see `_resolve_worktree_cwd`),
and never for more than one targeted unit per verifier-dispatch decision
(see the esc-097 paragraph above; this amends the prior "no git subprocess
anywhere" doctrine to these two narrower, explicit exceptions).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# esc-097 (R3, "probe the fix"): the version marker `check_lead_gate.py
# --self-test`'s guarded arm probes for via `hasattr` — its presence tells
# the fixture harness this patch is applied, so the arm runs for real
# instead of reporting SKIPPED.
RELAY_R3 = True

# --------------------------------------------------------------------------
# Agent-type lattice (closed world, deny-unknown).
# --------------------------------------------------------------------------

GATED_TYPES = {
    "ai-core", "bench", "cli", "cookbook", "db", "docs-ci", "numerics",
    "python", "wire-server", "general-purpose", "claude", "fork",
    "adversarial-audit", "fix-verifier", "acceptance-verifier", "doc-updater",
}
NEVER_GATED_TYPES = {
    "citation-checker", "graph-navigator", "build-graph", "Explore", "Plan",
    "claude-code-guide", "doc-parity", "retrospective", "issue-triage",
    "gap-analyzer", "pressure-tester", "statusline-setup", "oracle",
    "discipline-test-auditor",
}
# The second-round rule (§3) names exactly these three verifier types.
VERIFIER_SECOND_ROUND_TYPES = {"adversarial-audit", "fix-verifier", "acceptance-verifier"}
# SubagentStop matcher: the agent types whose verdict writes gate state.
STOP_MATCH_TYPES = {
    "adversarial-audit", "fix-verifier", "acceptance-verifier",
    "pressure-tester", "oracle", "citation-checker", "discipline-test-auditor",
}
# The harness's built-in agent types that are not `.claude/agents/*.md`
# cards at all (no `tools:` frontmatter to cross-check).
HARNESS_BUILTIN_TYPES = {"general-purpose", "Explore", "Plan", "claude", "statusline-setup",
                          "claude-code-guide", "fork"}

# --------------------------------------------------------------------------
# Verdict normalization — parse the JSON field, never substring-grep.
# --------------------------------------------------------------------------

# esc-lead-gate-R7d: the PASS vocabulary is PER AGENT TYPE, never pooled —
# a global set let an adversarial-audit row whose raw verdict text was
# literally "verified" (fix-verifier's own spelling) or "PROCEED"
# (pressure-tester's own spelling) also clear an adversarial-audit BLOCK,
# independent of anything R3 checks. `_PASS_LIKE_BY_AGENT_TYPE` maps a TYPE
# to the ONE literal string its own card spells as PASS; every type not
# listed (adversarial-audit, citation-checker, discipline-test-auditor,
# oracle, and any type outside STOP_MATCH_TYPES) defaults to "PASS" — the
# closed-world membership check that decides whether a type is gated AT ALL
# happens elsewhere, at dispatch time (GATED_TYPES/NEVER_GATED_TYPES), never
# here. Confirmed against all seven STOP_MATCH_TYPES cards' own verdict
# vocabularies and a census of every live verdict row: every real clearing
# already uses its own card's spelling, so no clearing depended on the
# pooled set.
_PASS_LIKE_BY_AGENT_TYPE: dict[str, str] = {
    "fix-verifier": "verified",
    "acceptance-verifier": "verified",
    "pressure-tester": "PROCEED",
}
_DEFAULT_PASS_WORD = "PASS"


def _pass_word_for(agent_type: str) -> str:
    """The ONE literal string `agent_type`'s own card spells as its
    PASS-equivalent verdict. Every STOP_MATCH_TYPES card not in
    `_PASS_LIKE_BY_AGENT_TYPE` (adversarial-audit, citation-checker,
    discipline-test-auditor, oracle) spells it "PASS"; an unrecognized
    agent_type also defaults to "PASS" — never a KeyError, since the
    closed-world membership check belongs to `_decide_dispatch`, not here."""
    return _PASS_LIKE_BY_AGENT_TYPE.get(agent_type, _DEFAULT_PASS_WORD)
_TEMPLATE_UNIT_BRANCH = "_branch_"

# The verdict is the LAST fenced ```json block — or, for one release, the
# LAST `<verdict>` tag (tolerating a markdown-escaped backslash before
# either angle bracket) when no fenced block exists. BOTH paths hand off to
# the SAME JSON-string-aware object extractor starting right after the
# marker; neither ever searches for a closing marker, so a `</verdict>` or
# a stray `}` sitting inside a quoted string can never end the region
# early (round-2 finding 6). The OPENING-marker scan is string-aware too
# (audit-r3 finding 2): a marker occurrence INSIDE a successfully parsed
# object's own quoted strings (e.g. a PASS whose `notes` mentions the
# marker) never counts as a later opening marker — the scan jumps past
# every object it parses (via the same brace walker) before looking again.
_FENCE_JSON_RE = re.compile(r"```json\s*", re.IGNORECASE)
_VERDICT_TAG_RE = re.compile(r"\\?<verdict\\?>")
# The lead's dispatch prompt names the unit under one of several shapes.
# `^[ \t]*unit:` (line-anchored) is tried first; `unit_branch:` / bare
# `unit_branch ` occurring ANYWHERE in the prompt (not line-anchored) are
# tried next, in that order — these are the shapes an audit of real
# dispatch prompts found the lead actually writing (the line-anchored form
# alone bound 0/126 real Starts; 12/625 prompts carried `^unit:` at all).
_UNIT_LINE_RE = re.compile(r"^[ \t]*unit:[ \t]*(\S+)", re.MULTILINE)
_UNIT_BRANCH_COLON_RE = re.compile(r"unit_branch:\s*(\S+)")
_UNIT_BRANCH_BARE_RE = re.compile(r"unit_branch\s+(\S+)")


def _last_marker_end_string_aware(pattern: re.Pattern, text: str) -> int | None:
    """The end offset of the LAST opening marker that is NOT inside a
    successfully parsed JSON object begun at an earlier marker. After each
    marker whose following object parses, the scan resumes AFTER that
    object's closing brace, so markers quoted inside the object's own
    strings are never counted (audit-r3 finding 2). After a marker whose
    object does NOT parse, the scan resumes right after the marker (there
    is no object to skip)."""
    last = None
    pos = 0
    while True:
        m = pattern.search(text, pos)
        if m is None:
            return last
        last = m.end()
        obj, obj_end = _extract_json_object_span(text, m.end())
        pos = obj_end if (obj is not None and obj_end is not None and obj_end > m.end()) else m.end()


def _extract_json_object_span(s: str, start: int = 0) -> tuple[dict | None, int | None]:
    """Find the first `{` in `s` at or after `start` and its JSON-string-
    aware MATCHING `}` (braces and angle brackets inside a quoted JSON
    string never count, an escaped `\\"` never ends a string early), parse
    exactly that substring, and tolerate trailing noise (a stray ``` fence,
    a `</verdict>` tag, more prose) between the object's closing brace and
    the end of `s`. There is no "find the closing marker" step — content
    inside a quoted field can never trick this into truncating early.
    Returns `(object, end_offset_just_past_the_closing_brace)`; `(None,
    None)` when no complete object parses."""
    brace = s.find("{", start)
    if brace == -1:
        return None, None
    depth = 0
    in_string = False
    escape = False
    for i in range(brace, len(s)):
        c = s[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(s[brace:i + 1])
                except Exception:
                    return None, None
                return (obj, i + 1) if isinstance(obj, dict) else (None, None)
    return None, None


def _extract_first_json_object(s: str, start: int = 0) -> dict | None:
    return _extract_json_object_span(s, start)[0]


def _normalize_unit_branch(raw: str | None) -> tuple[str | None, str | None]:
    """Normalize a reported `unit_branch` to its LEADING (first
    whitespace-delimited) token — BEFORE template-checking and BEFORE
    binding. Every verifier card's own schema line instructs "say which"
    (a provenance parenthetical after the branch, e.g. `"feat/x (from
    git)"`), and the annotated shape is exactly what real verifiers
    produce; the bare `" " in ub` template check used to reject it
    outright. The annotation — everything after the first whitespace run —
    is preserved separately as a note, never folded into the anchor a
    second-round dispatch is matched against (`_block_named_in_text`
    whole-token-matches on the BARE token only). A raw value that is only
    whitespace normalizes to `(None, None)` exactly like an absent field."""
    if not isinstance(raw, str):
        return None, None
    parts = raw.strip().split(None, 1)
    if not parts:
        return None, None
    token = parts[0]
    note = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
    return token, note


def _looks_like_template(data: dict) -> bool:
    """A parsed object that is the verifier CARD'S OWN SCHEMA TEMPLATE
    (echoed after — or instead of — a real verdict) is not a real verdict,
    even though it parses as valid JSON. `unit_branch` is checked on its
    LEADING TOKEN (see `_normalize_unit_branch`) so a real, annotated
    branch — `"feat/x (from git)"` — is never mistaken for the literal
    `"<...>"` schema placeholder merely because it, too, contains a space;
    a leading token that still carries `<`/`>` (the literal placeholder
    itself always does — it opens with `<`) remains template."""
    ub = data.get("unit_branch")
    if isinstance(ub, str):
        token, _note = _normalize_unit_branch(ub)
        if token is not None and (token == _TEMPLATE_UNIT_BRANCH or "<" in token or ">" in token):
            return True
    ce = data.get("class_enumeration")
    if isinstance(ce, list) and any(isinstance(x, str) and x.strip() == "path:line" for x in ce):
        return True
    return False


def extract_verdict_json(last_assistant_message: str) -> tuple[dict | None, str | None]:
    """Returns `(data, invalid_reason)`: the LAST fenced ```json block
    whose object has `"kind": "verdict"`; falling back (one release only)
    to the LAST `<verdict>` tag when no fenced block exists. `invalid_
    reason` is `None` on a normal parse, `"unparseable"`, or `"template"`.
    """
    if not last_assistant_message:
        return None, "unparseable"
    text = last_assistant_message

    fence_end = _last_marker_end_string_aware(_FENCE_JSON_RE, text)
    if fence_end is not None:
        data = _extract_first_json_object(text, fence_end)
        if data is None:
            return None, "unparseable"
        if data.get("kind") != "verdict":
            return None, "unparseable"
        if _looks_like_template(data):
            return None, "template"
        return data, None

    # One-release legacy fallback: no fenced ```json block found at all.
    tag_end = _last_marker_end_string_aware(_VERDICT_TAG_RE, text)
    if tag_end is not None:
        data = _extract_first_json_object(text, tag_end)
        if data is None:
            return None, "unparseable"
        if _looks_like_template(data):
            return None, "template"
        return data, None

    return None, "unparseable"


def normalize_verdict(data: dict | None, agent_type: str) -> tuple[str, str | None]:
    """Returns `(verdict, verdict_raw)`; `verdict` is one of "PASS" |
    "BLOCK" | "UNPARSEABLE". `agent_type` is the CALLER's own trusted value
    (`handle_stop`'s SubagentStop payload field, or a stored row's own
    `agent_type` via `_diagnose_row`) — never read out of `data` itself,
    which is verifier-authored prose the same way every other verdict field
    is (esc-lead-gate-R7d: PASS-like is PER CARD, never pooled)."""
    if data is None:
        return "UNPARSEABLE", None
    raw = data.get("verdict")
    if raw is None:
        raw = data.get("overall")
    if raw is None:
        return "UNPARSEABLE", None
    if not isinstance(raw, str):
        return "BLOCK", repr(raw)
    if raw == _pass_word_for(agent_type):
        return "PASS", raw
    return "BLOCK", raw


def is_open(verdict: str) -> bool:
    return verdict in ("BLOCK", "UNPARSEABLE")


def parse_verdict_fields(data: dict | None) -> dict:
    class_enum = data.get("class_enumeration") if data else None
    if not isinstance(class_enum, list) or not all(isinstance(x, str) for x in class_enum):
        class_enum = []
    sweep_method = data.get("sweep_method") if data else None
    exhaustive = bool(data.get("exhaustive", False)) if data else False
    # DIAGNOSTIC ONLY — no gate decision reads this field (esc-064: a flag
    # with a weak-arm default once selected the weaker relay requirement;
    # `_relay_rejection` derives its arms from the DATA instead).
    enumeration_missing = (not data) or ("class_enumeration" not in data) or (len(class_enum) == 0)
    unit_branch_raw = data.get("unit_branch") if data else None
    unit_branch, unit_branch_note = _normalize_unit_branch(unit_branch_raw)
    head_sha = data.get("head_sha") if data else None
    worktree = data.get("worktree") if data else None

    finding_locations: list[str] = []
    if data:
        for v in data.values():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        loc = item.get("location")
                        if isinstance(loc, str):
                            finding_locations.append(loc)
    # esc-lead-gate-R7 (v2): a VERIFIER-authored field, never a lead-computed
    # counter key — the auditor's own judgement that THIS round is a
    # recurrence of an earlier one on the SAME unit, carried verbatim into
    # the committed rigor record (`--export`) for a human to read at merge.
    # No gate decision reads this field: it FAILS OPEN when the auditor does
    # not notice a recurrence, and it is forgeable by the same ledger-append
    # path every other row field already is — DIAGNOSTIC/DISCLOSURE ONLY,
    # the same trust boundary as `class_enumeration` itself. `bool` is
    # excluded explicitly (`isinstance(True, int)` is true in Python).
    recurrence_of_round = data.get("recurrence_of_round") if data else None
    if isinstance(recurrence_of_round, bool) or not isinstance(recurrence_of_round, int):
        recurrence_of_round = None
    return {
        "class_enumeration": class_enum,
        "sweep_method": sweep_method,
        "exhaustive": exhaustive,
        "enumeration_missing": enumeration_missing,
        "unit_branch": unit_branch,
        "unit_branch_note": unit_branch_note,
        "head_sha": head_sha if isinstance(head_sha, str) else None,
        "worktree": worktree if isinstance(worktree, str) else None,
        "finding_locations": finding_locations,
        "recurrence_of_round": recurrence_of_round,
    }


# --------------------------------------------------------------------------
# State I/O — `.jammi/gate-state/`, gitignored, hook-written only (except
# the relay artifact, which the LEAD writes directly).
# --------------------------------------------------------------------------

def repo_root() -> Path:
    root = os.environ.get("CLAUDE_PROJECT_DIR")
    if root:
        return Path(root)
    return Path.cwd()


def state_dir() -> Path:
    return repo_root() / ".jammi" / "gate-state"


def slugify(branch: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", branch.strip()) or "UNBOUND"


def _fs_safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s.strip()) or "_"


def unit_file(sdir: Path, unit_branch: str) -> Path:
    return sdir / f"{slugify(unit_branch)}.jsonl"


def bindings_file(sdir: Path) -> Path:
    return sdir / "bindings.jsonl"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True))
        f.write("\n")


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                rows.append({"_unparseable_raw": line})
        except Exception:
            rows.append({"_unparseable_raw": line})
    return rows


def _unit_rows_by_agent_type(sdir: Path, unit_slug: str) -> dict[str, tuple[dict, int]]:
    """`agent_type -> (latest verdict row for that agent_type, its index in
    the full row list)`. A corrupted (unparseable-JSONL) row is attributed
    to the sentinel agent_type `""` and is always open."""
    rows = read_rows(unit_file(sdir, unit_slug))
    latest: dict[str, tuple[dict, int]] = {}
    for i, r in enumerate(rows):
        if "_unparseable_raw" in r:
            latest[""] = ({
                "verdict": "BLOCK", "verdict_raw": None, "class_enumeration": [],
                "enumeration_missing": True, "finding_locations": [], "agent_type": "",
                "_corrupted": True,
            }, i)
            continue
        if "verdict" in r:
            latest[r.get("agent_type") or ""] = (r, i)
    return latest


def open_blocks_for_unit(sdir: Path, unit_slug: str) -> list[tuple[str, dict, int]]:
    """`[(agent_type, row, idx), …]` for every agent_type whose LATEST row
    on this unit is BLOCK-equivalent. esc-097 (V10): there is no cross-type
    clearing arm — an `adversarial-audit` BLOCK stays open until a LATER
    `adversarial-audit` row on the SAME unit is itself a PASS (which can
    only happen after a relay that passed R1+R2+R3 on the direct
    repeat-dispatch path), or until the operator `rm`s the state by hand.
    An earlier draft's `_adversarial_audit_cleared_by_verifier_pass` (a
    `fix-verifier`/`acceptance-verifier` PASS clearing an older
    `adversarial-audit` BLOCK without ever running R3) is deleted, not
    tightened — deleting it is what closes the round-2 reproducer (a relay
    with no `fix_head` at all previously cleared via that path, because it
    never touched git)."""
    by_type = _unit_rows_by_agent_type(sdir, unit_slug)
    out: list[tuple[str, dict, int]] = []
    for atype, (row, idx) in by_type.items():
        if not is_open(row.get("verdict", "UNPARSEABLE")):
            continue
        out.append((atype, row, idx))
    return out


def all_open_blocks(sdir: Path) -> list[tuple[str, str, dict, int]]:
    """`[(unit_slug, agent_type, row, idx), …]` across every unit file."""
    out: list[tuple[str, str, dict, int]] = []
    if not sdir.exists():
        return out
    for entry in sdir.iterdir():
        if entry.name in ("bindings.jsonl", "hook.log") or entry.suffix != ".jsonl":
            continue
        if ".relay." in entry.name:
            continue
        for atype, row, idx in open_blocks_for_unit(sdir, entry.stem):
            out.append((entry.stem, atype, row, idx))
    return out


def any_unit_has_open_block(sdir: Path) -> bool:
    return len(all_open_blocks(sdir)) > 0


def _lookup_binding_unit(agent_id: str, sdir: Path) -> str | None:
    """ADVISORY ONLY (never used for a gate decision): the latest
    SubagentStart binding for `agent_id`, used only to file an UNPARSEABLE
    verdict row under the right unit."""
    for r in reversed(read_rows(bindings_file(sdir))):
        if r.get("agent_id") == agent_id:
            ub = r.get("unit_branch")
            if isinstance(ub, str) and ub:
                return ub
    return None


# --------------------------------------------------------------------------
# The relay artifact. Written by the LEAD directly (Write is not gated);
# the hook only ever READS it, fresh, every time it gates — there is no
# "append relay_accepted then discover later it should have been rejected"
# step, so a DENY can never leave a phantom acceptance behind.
# --------------------------------------------------------------------------

def relay_artifact_path(sdir: Path, unit_slug: str, agent_type: str, block_ts: str) -> Path:
    return sdir / f"{unit_slug}.relay.{_fs_safe(agent_type)}.{_fs_safe(block_ts)}.json"


def _probe_normalize(s: str) -> str:
    """Zero-width/control characters (Unicode categories Cf/Cc) dropped, then
    surrounding whitespace stripped — applied to BOTH sides of the
    probe/reactive comparison. This is NOT the normalization the `sites`
    doctrine bans: normalizing `sites` would make acceptance EASIER, while
    this normalization is provably monotone-toward-DENY — it can only shrink
    the adjacent set (a padded or invisible-char entry either collapses onto
    a reactive site or onto its own duplicate). esc-064 mutant C and its
    zero-width sibling class."""
    import unicodedata
    cleaned = "".join(ch for ch in s if unicodedata.category(ch) not in ("Cf", "Cc"))
    return cleaned.strip()


# --------------------------------------------------------------------------
# esc-097 (R3, "probe the fix"): the ONLY git subprocess in this module, run
# ONLY from `_relay_rejection` (in turn reachable ONLY from
# `_decide_verifier_dispatch`'s repeat-dispatch branch) — never on a first
# dispatch's hot path. `$CLAUDE_PROJECT_DIR` is REQUIRED explicitly here
# (this arm never falls back to `repo_root()`'s cwd), with a shared,
# per-decision `_GIT_BUDGET_S` git budget (V18); any failure DENIES, naming
# the failing command.
# --------------------------------------------------------------------------

_GIT_BUDGET_S = 5.0
_ESCAPE_HATCH_NOTE = (
    "rm .jammi/gate-state/<slug>.* is the operator escape hatch, but it "
    "destroys the unit's evidence rather than fixing the underlying git problem"
)
# V15: every git-bound sha (the BLOCK row's own `head_sha` AND the relay's
# `fix_head`) must match this shape BEFORE it is ever placed in a git argv —
# a value shaped like an option (e.g. `--output=/tmp/x`) is rejected here,
# never merely relied on `--end-of-options` (below) to neutralize.
_SHA_RE = re.compile(r"[0-9a-f]{7,40}")


def _new_git_deadline() -> float:
    """One `_GIT_BUDGET_S`-wide monotonic deadline, minted ONCE per decision
    (by `_fix_window`) and threaded through every `_run_git` call it makes —
    a decision that runs five git calls (round-4: `fix_head` gained its own
    `rev-parse --verify` resolution alongside `block_sha`'s) is bounded by
    `_GIT_BUDGET_S` total, never `5 * _GIT_BUDGET_S`."""
    return time.monotonic() + _GIT_BUDGET_S


def _run_git(args: list[str], cwd: str, deadline: float,
             warnings: list[str] | None = None) -> tuple[bool, str, int | None]:
    """Runs `git -C <cwd> <args>`, bounded by the TIME LEFT on the shared,
    per-decision `deadline` (an absolute `time.monotonic()` value minted
    once by `_new_git_deadline()`, never a fresh timeout per call) — a
    decision that makes five such calls is bounded by `_GIT_BUDGET_S`
    total, not `5 * _GIT_BUDGET_S`. Returns `(True, stdout, rc)` on
    success (`rc == 0`); `(False, reason, rc)` on any failure — `rc` is
    git's own exit code when the process ran to completion (so a caller
    that must distinguish, e.g., `git merge-base --is-ancestor`'s `rc == 1`
    "not an ancestor" from a real error can inspect it), or `None` when no
    process ever produced one (budget already exhausted, failed to spawn,
    or timed out). The caller always turns a `False` into a DENY, never a
    silent allow.

    `warnings`, when given, is a caller-owned list this call APPENDS to
    (never replaces) whenever the process still SUCCEEDED (`rc == 0`) but
    wrote to stderr anyway — e.g. git's own `warning: refname '...' is
    ambiguous.` when a ref shadows a sha or another ref of the same literal
    name and still resolves to a prefix-matching, correct object (round-4).
    A failing call's stderr is never routed here — it is already folded
    into that call's own `(False, reason, rc)` return value, read back the
    same way, below.

    Never `with Popen(...)`, never `subprocess.run` (V12): `Popen` is
    opened into its OWN process group (`start_new_session=True`) with
    stdout/stderr captured to real `tempfile.TemporaryFile()`s — never
    `subprocess.PIPE` — so the parent NEVER DRAINS A PIPE AT ALL; the bound
    comes from `Popen.wait(timeout=…)` alone. This is the load-bearing fix,
    not `killpg`: a grandchild that escapes the process group entirely (its
    own `os.setsid()`, e.g. a detached credential helper) is NOT reachable
    by `os.killpg(proc.pid, ...)` either — measured, a manual `Popen(...,
    stdout=PIPE).communicate(timeout=5)` that catches `TimeoutExpired`,
    `killpg`s, and then calls a SECOND, UN-timed `.communicate()` to "drain"
    still blocks ~30s against exactly such a shim, because that second call
    waits for every pipe writer (including the escaped grandchild) to close
    its end — `killpg` reaching the ORIGINAL group does not help when the
    holder already left it. Switching to `TemporaryFile` + bounded `wait()`
    sidesteps the question entirely: nothing is ever read from a pipe, so
    it does not matter whether the grandchild is reachable or not. On
    `TimeoutExpired`, `os.killpg` the WHOLE group anyway (real, complementary
    benefit for the COMMON case: a hung child that never escaped the group,
    e.g. a stuck-but-still-in-group credential helper, is reaped rather
    than left running), then a SECOND, independently bounded `wait(timeout=
    1)` — this reap tail is a flat 1s, NOT drawn from the shared budget, and
    is unchanged from the pre-V18 shape; if even that does not return,
    close the temp files ourselves and `proc.kill()` before giving up.

    Measured at `_GIT_BUDGET_S == T == 5.0` (the value actually shipped;
    an earlier pressure probe at T=1 had reported "~1.0s", which no longer
    describes this deadline): against the escaped-grandchild shim
    (`python3 -c "import os,time;os.setsid();time.sleep(30)" &` then `exec
    sleep 7`) this shape returns in **~5.00s** (vs. ~30s for the
    un-timed-drain shape it replaced); against a CHILDLESS shim (`exec
    sleep 30`, no escaped grandchild at all) it also returns in **~5.00s**
    — the bound holds regardless of whether a grandchild escapes the group,
    because nothing is ever read from a pipe either way. A 549 KB `diff
    --name-only` payload (18000+9000 added files, measured directly, not
    the earlier "540 KB in ~0.01s" guess) round-trips through the temp
    files in **~0.02-0.03s**."""
    cmd = ["git", "-C", cwd] + list(args)
    printable = "git " + " ".join(args)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return (
            False,
            f"the shared {_GIT_BUDGET_S:g}s git budget for this decision was already "
            f"exhausted before `{printable}` could run — {_ESCAPE_HATCH_NOTE}",
            None,
        )
    try:
        out_f = tempfile.TemporaryFile()
        err_f = tempfile.TemporaryFile()
    except OSError as exc:
        return False, f"could not open temp files for `{printable}` ({exc}) — {_ESCAPE_HATCH_NOTE}", None
    try:
        proc = subprocess.Popen(
            cmd, stdout=out_f, stderr=err_f, start_new_session=True,
        )
    except OSError as exc:
        out_f.close()
        err_f.close()
        return False, f"`{printable}` failed to run ({exc}) — {_ESCAPE_HATCH_NOTE}", None
    try:
        rc = proc.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        out_f.close()
        err_f.close()
        return (
            False,
            f"`{printable}` timed out (shared {_GIT_BUDGET_S:g}s git budget for this "
            f"decision) — {_ESCAPE_HATCH_NOTE}",
            None,
        )
    try:
        out_f.seek(0)
        out = out_f.read().decode("utf-8", errors="replace")
        err_f.seek(0)
        err = err_f.read().decode("utf-8", errors="replace").strip()
    finally:
        out_f.close()
        err_f.close()
    if rc != 0:
        detail = f" ({err})" if err else ""
        return False, f"`{printable}` exited {rc}{detail} — {_ESCAPE_HATCH_NOTE}", rc
    if err and warnings is not None:
        warnings.append(f"`{printable}` wrote to stderr on a call that still succeeded: {err}")
    return True, out.strip(), rc


# A probe entry's PATH is the first whitespace-delimited token (V17: NOT a
# quote- or backtick-span rule — a fix touching only a path containing a
# space is a documented limit; no such tracked path exists today), a
# surrounding backtick or parenthesis and trailing punctuation stripped,
# then an optional trailing `:<n>[-<n>][,<n>]*` line spec stripped.
# Unparseable -> None, which counts for nothing toward R3 (monotone toward
# DENY, never toward a false ALLOW).
_PROBE_LINESPEC_RE = re.compile(r":\d+(?:-\d+)?(?:,\d+)*$")


def _probe_path(entry: str) -> str | None:
    if not isinstance(entry, str):
        return None
    parts = entry.strip().split(None, 1)
    if not parts:
        return None
    tok = parts[0].strip("`()").rstrip(",;:.)")
    tok = _PROBE_LINESPEC_RE.sub("", tok)
    return tok or None


def _resolve_sha_exact(hexstr: str, label: str, cwd: str, deadline: float,
                        warnings: list[str]) -> tuple[str | None, str | None]:
    """Resolves a sha-shaped value (`block_sha`/`fix_head` — already
    `_SHA_RE`-validated by the caller, so 7-40 hex chars) via `git rev-parse
    --verify --end-of-options <hexstr>^{commit}` and requires the resolved,
    FULL 40-hex object name to START WITH the given `hexstr` — never merely
    that it resolves to SOME commit. `--end-of-options` alone does not stop
    a REF literally NAMED like a hex string from shadowing the object it
    abbreviates: measured, a branch literally named `da137e40f501` pointing
    at an unrelated commit makes `git rev-parse --verify
    da137e40f501^{commit}` resolve to the BRANCH's tip (not the abbreviated
    object), printing `warning: refname '...' is ambiguous.` to stderr
    while still exiting 0 — silently wrong, not merely noisy (round-4
    adversarial reproducer G40). Returns `(resolved_full_sha, None)` on
    success — the CALLER uses this resolved, full value at every later
    git-argv site (`merge-base`, `diff`), never the caller-supplied,
    possibly-abbreviated form again — or `(None, deny_reason)` otherwise.
    Any stderr text on a call that still resolved a prefix-matching object
    is appended to `warnings` (by `_run_git` itself) so a shadow that
    happens to resolve safely is still visible on the accept-side
    reason/log, not only on a DENY."""
    ok, resolved, _rc = _run_git(
        ["rev-parse", "--verify", "--end-of-options", f"{hexstr}^{{commit}}"], cwd, deadline, warnings)
    if not ok:
        return None, f"{label} {hexstr} does not resolve (an amend can orphan it) — {resolved}"
    if not resolved.startswith(hexstr):
        return None, (
            f"{hexstr} resolves to {resolved}, which is not that object — a ref named like "
            "a sha shadows it"
        )
    return resolved, None


def _fix_window(row: dict, unit_slug: str, data: dict, deadline: float) -> tuple[
        list[str] | None, dict[str, str] | None, str | None, list[str], dict[str, str] | None]:
    """Resolves the trusted, COMPUTED `fix_changed` set for the relay arm,
    (esc-lead-gate-R11) the trusted, COMPUTED `claim_sites` map — every
    claim-shaped line the fix's OWN diff adds, keyed `path:line`, mapped to
    its own verbatim text — and (esc-lead-gate-R12 reader 2) the trusted,
    COMPUTED `new_surfaces` map, derived from that SAME `-U0` diff payload
    (no extra git call). `(fix_changed, claim_sites, None, warnings,
    new_surfaces)` on success; `(None, None, deny_reason, warnings, None)`
    otherwise — `warnings` carries any accept-side git stderr text noticed
    along the way (round-4: surfaced by the caller even when it did not
    change the outcome). Only ever called from `_relay_rejection`'s own
    repeat-dispatch caller, once R1/R2 have already passed.

    esc-lead-gate-R11: `deadline` is now a PARAMETER, minted ONCE by the
    caller (`_relay_rejection`) via `_new_git_deadline()`, rather than
    minted internally here as before — R3's own git calls and R11's
    claim-scan (the new SIXTH call, below) share the SAME one-decision
    budget this way; the arm never grows `_GIT_BUDGET_S` per-caller. Every
    internal use of `deadline` below is otherwise UNCHANGED from before
    this parameter move.

    esc-097 V18 (round-3 closure) plus the round-4 closure below:
    reachability binds TWO things, not one. First, git-free: the relay's
    own `unit_branch` must `slugify()` to EXACTLY this BLOCK's own
    `unit_slug` (the identity the dispatch already resolved) — this binds
    the unit's NAME, never an arbitrary branch the relay merely asserts. A
    row filed under the `UNBOUND` fallback bucket can never be satisfied —
    no real branch name slugifies to the literal string `UNBOUND`. Second,
    binding the name is not enough: `fix_head`'s POSITION on that name's
    OWN history must also hold, or a relay could name the right unit's
    branch while citing a `fix_head` that an amend orphaned, or that landed
    on some unrelated branch — both slugify-match while never being
    reachable from the named branch's tip. So `unit_branch` is resolved
    UNDER `refs/heads/` ONLY (`git rev-parse --verify --end-of-options
    refs/heads/<name>^{commit}`, never a bare `<name>^{commit}`) and
    `fix_head` must `git merge-base --is-ancestor` that resolved tip.

    Round-4 (two adversarial reproducers, both closed structurally, not by
    patching a symptom): (1) a bare `<name>^{commit}` lookup for
    `unit_branch` is resolved per gitrevisions(7)'s own disambiguation
    order, which tries `refs/tags/<name>` BEFORE `refs/heads/<name>` — a
    TAG literally named like the unit branch, pointing at a DIFFERENT
    branch's commit, silently wins and the ancestry check then runs
    against the TAG's target instead of the real branch's own tip (G39).
    Fixed by resolving `refs/heads/<name>^{commit}` explicitly — a tag (or
    any other `refs/<kind>/<name>`) sharing the branch's literal name no
    longer participates in the lookup at all. A side effect, stated
    honestly: a unit whose worktree is on a DETACHED HEAD has no
    `refs/heads/` entry to bind to and can never be relayed this way. (2)
    `block_sha`/`fix_head` are sha-shaped, not branch names, so they are
    resolved through `_resolve_sha_exact` instead, which requires the
    resolved FULL 40-hex object to START WITH the caller-supplied hex — a
    branch (or tag) literally NAMED like a sha, or a short prefix of one,
    can shadow the object it abbreviates the exact same way `unit_branch`
    could (G40); `_resolve_sha_exact`'s prefix check catches it regardless
    of which of the two git happened to pick. The RESOLVED, full sha is
    what every later git argv (`merge-base`, `diff`) actually uses, never
    the original, possibly-abbreviated caller-supplied value — so a
    `fix_head` given as a short prefix that resolves to the SAME commit as
    `block_sha` is caught as a re-roll even when the two strings differed
    before resolution. SIX git calls total (`rev-parse --verify` for
    block_sha, `rev-parse --verify` for fix_head, `rev-parse --verify
    --end-of-options refs/heads/...` for unit_branch, `merge-base
    --is-ancestor`, `diff --name-only`, and — esc-lead-gate-R11 — `diff
    -U0` for the claim-scan), each sharing ONE per-decision deadline
    (`_new_git_deadline()`, now minted by the CALLER and threaded in as a
    parameter — `_GIT_BUDGET_S` total, not `6 * _GIT_BUDGET_S`)."""
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR")
    if not project_dir:
        return None, None, "hook needs CLAUDE_PROJECT_DIR for the relay arm", [], None

    warnings: list[str] = []

    block_sha = row.get("head_sha")
    if not (isinstance(block_sha, str) and _SHA_RE.fullmatch(block_sha)):
        return None, None, "BLOCK row's head_sha is not a valid sha", warnings, None
    block_sha, why = _resolve_sha_exact(
        block_sha, "BLOCK row's head_sha", project_dir, deadline, warnings)
    if why is not None:
        return None, None, why, warnings, None

    fix_head = data.get("fix_head")
    if not (isinstance(fix_head, str) and _SHA_RE.fullmatch(fix_head)):
        return None, None, "relay carries no valid `fix_head` — the lead must write the fix commit's full sha", warnings, None

    if fix_head == block_sha:
        return None, None, "no fix commit since the BLOCK; a second dispatch without a fix is a re-roll", warnings, None

    fix_head, why = _resolve_sha_exact(fix_head, "relay `fix_head`", project_dir, deadline, warnings)
    if why is not None:
        return None, None, why, warnings, None
    if fix_head == block_sha:
        # Round-4: a fix_head given as a SHORT prefix can resolve to the
        # SAME full commit as block_sha even when the two caller-supplied
        # strings differed — still a re-roll, only visible after both are
        # resolved to their full form.
        return None, None, "no fix commit since the BLOCK; a second dispatch without a fix is a re-roll", warnings, None

    relay_ub = data.get("unit_branch")
    if not (isinstance(relay_ub, str) and relay_ub.strip()):
        return None, None, "relay carries no `unit_branch` naming the unit this fix landed on", warnings, None
    if unit_slug == "UNBOUND":
        return None, None, (
            "this BLOCK was recorded without a unit binding (the UNBOUND fallback bucket) — "
            "re-dispatch naming the unit so the verdict lands on the unit's own file, then "
            "hand-remove the stale row for this block from UNBOUND.jsonl (never `rm` the "
            "shared file — it holds every other unit's UNBOUND rows too)"
        ), warnings, None
    if slugify(relay_ub) != unit_slug:
        return None, None, (
            f"relay `unit_branch` {relay_ub!r} does not name this BLOCK's own unit "
            f"(recorded under slug {unit_slug!r}) — reachability is bound to the file this "
            "BLOCK is filed under, not an arbitrary branch the relay names"
        ), warnings, None

    # V18 + round-4: the slug equality above binds the NAME; fix_head must
    # also be bound to that same name's own HISTORY. `unit_branch` is
    # resolved under refs/heads/ ONLY — a bare `<name>^{commit}` lookup lets
    # a same-named tag (or any other refs/<kind>/<name>) win the
    # disambiguation and shadow the real branch (G39).
    ok, tip, _rc = _run_git(
        ["rev-parse", "--verify", "--end-of-options", f"refs/heads/{relay_ub}^{{commit}}"],
        project_dir, deadline, warnings)
    if not ok:
        return None, None, (
            f"relay `unit_branch` {relay_ub!r} slugifies to this BLOCK's own unit but does not "
            f"resolve under refs/heads/ — {tip} (a unit whose worktree is on a detached HEAD "
            "cannot be relayed this way — there is no refs/heads entry to bind to)"
        ), warnings, None
    ok, why, rc = _run_git(
        ["merge-base", "--is-ancestor", "--end-of-options", fix_head, tip], project_dir, deadline, warnings)
    if not ok:
        if rc == 1:
            return None, None, (
                f"fix_head {fix_head} is not on {relay_ub!r}; if the fix was amended, name "
                "the amended sha; if it was committed on a child branch, commit or merge it "
                f"onto {relay_ub!r}"
            ), warnings, None
        return None, None, f"could not check whether fix_head is on {relay_ub!r} — {why}", warnings, None

    ok, out, _rc = _run_git(
        ["diff", "--name-only", "-z", "--end-of-options", block_sha, fix_head], project_dir, deadline, warnings)
    if not ok:
        return None, None, f"could not compute the fix window: {out}", warnings, None
    fix_changed = [seg for seg in out.split("\0") if seg]

    # esc-lead-gate-R11 — the SIXTH git call, sharing the SAME deadline as
    # the five above: the fix's own added-line diff, zero context (`-U0`),
    # from which `_parse_claim_sites` derives the hook's OWN, mechanically
    # DERIVED claim enumeration — never lead-asserted, the same posture R1
    # already takes toward the auditor's own `class_enumeration`, applied
    # here to the fix's diff instead of a finding.
    ok, diff_out, _rc = _run_git(
        ["diff", "-U0", "--end-of-options", block_sha, fix_head], project_dir, deadline, warnings)
    if not ok:
        return None, None, f"could not compute the fix's added-line diff (esc-lead-gate-R11): {diff_out}", warnings, None
    claim_sites = _parse_claim_sites(diff_out)
    new_surfaces = _parse_new_surfaces(diff_out)
    return fix_changed, claim_sites, None, warnings, new_surfaces


# --------------------------------------------------------------------------
# esc-lead-gate-R11 — "UNTESTED CLAIMS CARRY A TEST". The obligation is
# DERIVED BY THE HOOK from the fix's own diff, never declared by the lead:
# `_parse_claim_sites` (above, run from `_fix_window`, sharing its git
# deadline) is the hook's OWN enumeration of every claim-shaped line the fix
# ADDS. A relay may not be accepted while any such line lacks a disposition
# in `claims` — see `_claims_rejection`, the sole caller of everything below.
#
# THE PHRASE FAMILY is the exact, finite set of claim shapes this program's
# own retrospective (`{scratchpad}/CONTRACT-RULES.md` R-A/R-F/R-H, plus the
# diagnosis's own "re-verified" example) named as the shape that kept costing
# a full audit round: an impossibility/unreachability claim, a "safe
# because"/"routes through" mechanism claim, a caller-SET totality claim, and
# a completed-re-verification claim. Measured, not asserted: scoping to
# PROSE LINES (below) plus literal, space-delimited multi-word phrases —
# never a bare word — is what keeps this precise. A bare quantifier grep
# (`every`/`only`/`always`, R-I's own tell) was tried first and produces
# garbage — `grep -ciE '\\b(every|only|always)\\b'` alone returns 91 hits in
# this very file and 1291 across `ci/scripts/*.py` — and is DELIBERATELY NOT
# shipped; R-I's shape is a known, stated residual (see the proposal doc's
# "what this does NOT do"). The phrase family below, unscoped, returns 26
# hits across this repo's `ci/scripts` + `.claude/hooks` + `.claude/agents`
# combined — a tractable number, not a flood.
_CLAIM_PHRASES = (
    "cannot be driven", "no injection point", "is unreachable", "is safe because",
    "no caller can", "not producer-driven", "no reliable way to force",
    "is not established", "structurally unreachable", "its only in-tree caller",
    "its one in-tree caller", "every caller is", "routes through", "re-verified",
)
_CLAIM_RE = re.compile("|".join(re.escape(p) for p in _CLAIM_PHRASES), re.IGNORECASE)

# WHERE: reuses the exact WHERE-scoping TECHNIQUE
# `ci/scripts/check_doc_numbers_have_producers.py` already established for
# precision (doc comments / whole-prose files, never bare code) — the
# concept, not the file, since this hook must stay dependency-free and must
# never import a CI-only script. A "prose line" is a whole line in a
# whole-prose file extension, or a line whose STRIPPED text starts with the
# file extension's own single-line-comment marker; an unknown extension
# tries both `#` and `//` (never neither — failing OPEN here would silently
# exempt an entire language from the rule, which is the wrong direction to
# fail for a DENY-arming enumeration).
_LINE_COMMENT_MARKERS: dict[str, tuple[str, ...]] = {
    ".rs": ("//",), ".py": ("#",), ".sh": ("#",), ".bash": ("#",), ".toml": ("#",),
    ".yml": ("#",), ".yaml": ("#",), ".js": ("//",), ".ts": ("//",), ".go": ("//",),
    ".c": ("//",), ".h": ("//",), ".cpp": ("//",), ".hpp": ("//",), ".java": ("//",),
}
_WHOLE_LINE_PROSE_EXTS = {".md", ".mdx", ".rst", ".txt"}
_DEFAULT_COMMENT_MARKERS = ("#", "//")


def _is_prose_line(path: str, stripped: str) -> bool:
    ext = Path(path).suffix.lower()
    if ext in _WHOLE_LINE_PROSE_EXTS:
        return True
    markers = _LINE_COMMENT_MARKERS.get(ext, _DEFAULT_COMMENT_MARKERS)
    return any(stripped.startswith(m) for m in markers)


_DIFF_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _parse_claim_sites(diff_text: str) -> dict[str, str]:
    """The hook's OWN derived claim enumeration from a `git diff -U0`
    payload: every `path:line` the diff ADDS (never a removed or context
    line — there are no context lines under `-U0`) whose text is BOTH a
    prose line (`_is_prose_line`) and matches `_CLAIM_RE`. The value is the
    line's own verbatim text (comment marker included, only the line's own
    leading/trailing whitespace stripped).

    A state machine, not a bare line-by-line grep: `+++`/`---` are only
    read as FILE HEADERS between a `diff --git` line and this file's first
    `@@` hunk — never afterward. Without this, a fix that adds a NEW
    `.patch` file (this very repo's own convention — see
    `docs/plans/*/proposals/*.patch`) would have its OWN embedded `+++ b/…`
    text misread as a second file boundary the instant a content line
    inside that patch starts with three literal `+` characters (the
    patched file's own added line, prefixed by this diff's own `+`)."""
    sites: dict[str, str] = {}
    current_path: str | None = None
    in_header = False
    new_lineno = 0
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            current_path = None
            in_header = True
            continue
        if in_header:
            if line.startswith("+++ ") or line == "+++":
                raw = line[4:] if line.startswith("+++ ") else ""
                if raw == "/dev/null":
                    current_path = None
                elif raw[:2] in ("a/", "b/"):
                    current_path = raw[2:]
                else:
                    current_path = raw or None
                in_header = False
            continue
        if line.startswith("@@"):
            m = _DIFF_HUNK_RE.match(line)
            new_lineno = int(m.group(1)) if m else 0
            continue
        if current_path is None:
            continue
        if line.startswith("+"):
            text = line[1:]
            stripped = text.strip()
            if stripped and _is_prose_line(current_path, stripped) and _CLAIM_RE.search(stripped):
                sites[f"{current_path}:{new_lineno}"] = text.strip()
            new_lineno += 1
    return sites


# --------------------------------------------------------------------------
# esc-lead-gate-R12 reader 2 — every NEW definition the fix's own diff ADDS
# widens the post-fix required-attack-key set. Same `-U0` payload
# `_parse_claim_sites` already parses, no extra git call. K2 (fail
# direction): an unparseable hunk enumerates NOTHING extra for that line —
# monotone toward requiring LESS, never toward silently exempting a real
# surface (the pre-fix `finding_locations`/`class_enumeration` keys are
# UNCHANGED by this and remain fully required regardless).
# --------------------------------------------------------------------------

_RUST_DEF_RE = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:default\s+)?(?:async\s+|unsafe\s+|const\s+|"
    r"extern\s+\"[^\"]*\"\s+)*(fn|struct|enum|trait)\s+(\w+)"
)
_RUST_IMPL_RE = re.compile(r"^\s*impl(?:<[^>]*>)?\s+(?:[\w:<>, ]+?\s+for\s+)?([\w:]+)")
_RUST_MOD_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)")
_PY_DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)")
_PY_CLASS_RE = re.compile(r"^\s*class\s+(\w+)")
_BASH_FN_RE = re.compile(r"^\s*(?:function\s+)?(\w+)\s*\(\)\s*\{")
_WORKFLOW_STEP_RE = re.compile(r"^\s*-\s*name:\s*(.+?)\s*$")


def _new_surface_def(path: str, added_text: str) -> tuple[str, str] | None:
    """`(kind, identifier)` iff `added_text` (one `+` line's own text) is a
    new-definition line, scoped by `path`'s own extension; `None`
    otherwise."""
    ext = Path(path).suffix.lower()
    if ext == ".rs":
        m = _RUST_DEF_RE.match(added_text)
        if m:
            return m.group(1), m.group(2)
        m = _RUST_IMPL_RE.match(added_text)
        if m:
            return "impl", m.group(1)
        m = _RUST_MOD_RE.match(added_text)
        if m:
            return "mod", m.group(1)
        return None
    if ext == ".py":
        m = _PY_DEF_RE.match(added_text)
        if m:
            return "def", m.group(1)
        m = _PY_CLASS_RE.match(added_text)
        if m:
            return "class", m.group(1)
        return None
    if ext in (".sh", ".bash"):
        m = _BASH_FN_RE.match(added_text)
        if m:
            return "function", m.group(1)
        return None
    if ext in (".yml", ".yaml"):
        m = _WORKFLOW_STEP_RE.match(added_text)
        if m:
            return "step", m.group(1)
        return None
    return None


def _parse_new_surfaces(diff_text: str) -> dict[str, str]:
    """The hook's OWN derived new-surface enumeration from a `git diff -U0`
    payload — every `path:line` the diff ADDS whose text is a new
    definition (`_new_surface_def`). IDENTICAL state machine to
    `_parse_claim_sites` (same reason: a fix that adds a new `.patch` file
    must never have its own embedded `+++ b/…` misread as a second file
    boundary) — duplicated deliberately, the same choice `_run_claim_
    command` already makes against `_run_git`."""
    sites: dict[str, str] = {}
    current_path: str | None = None
    in_header = False
    new_lineno = 0
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            current_path = None
            in_header = True
            continue
        if in_header:
            if line.startswith("+++ ") or line == "+++":
                raw = line[4:] if line.startswith("+++ ") else ""
                if raw == "/dev/null":
                    current_path = None
                elif raw[:2] in ("a/", "b/"):
                    current_path = raw[2:]
                else:
                    current_path = raw or None
                in_header = False
            continue
        if line.startswith("@@"):
            m = _DIFF_HUNK_RE.match(line)
            new_lineno = int(m.group(1)) if m else 0
            continue
        if current_path is None:
            continue
        if line.startswith("+"):
            text = line[1:]
            defn = _new_surface_def(current_path, text)
            if defn is not None:
                kind, ident = defn
                sites[f"{current_path}:{new_lineno}"] = f"{kind} {ident}"
            new_lineno += 1
    return sites


# WRITE-VERB DENYLIST, WHOLE-TOKEN matched (never substring — R4a's own
# killed design measured `git grep -n 'transform'`/`'confirm'` wrongly
# DENIED by a substring-`rm` check; this denylist is checked against
# shell-tokenized argv, one token at a time, so `transform` is never
# confused with `rm`), and it covers the THREE shapes that SAME killed
# design measured as wrongly ALLOWED (`curl … | sh`, `git clean -fdx`,
# `git reset --hard`) — by denying bare shell invocations and specific
# destructive git subcommands outright, never by pattern-matching flags.
_SHELL_OPERATOR_TOKENS = {"|", "||", "&&", ";", ">", ">>", "<", "&"}
_DENIED_PROGRAMS = {
    "rm", "mv", "dd", "shred", "mkfs", "truncate", "chmod", "chown", "chgrp",
    "sudo", "su", "kill", "killall", "pkill", "reboot", "shutdown", "halt",
    "curl", "wget", "ssh", "scp", "rsync", "nc", "ncat", "telnet",
    "docker", "kubectl", "eval", "sh", "bash", "zsh", "dash", "ksh",
    "xargs", "env",
}
_DENIED_GIT_SUBCOMMANDS = {"push", "reset", "clean", "gc"}


def _command_denied(command: object) -> str | None:
    """`None` iff `command` is safe to re-execute; otherwise the deny
    reason. Splits on shell operators (`|`, `&&`, `;`, `>`, …) via
    `shlex.shlex(..., punctuation_chars=True)`, which recognizes them as
    their OWN tokens while still respecting quoting — so a denied program
    hiding after a pipe (`curl … | sh`) or a redirect (`echo x > f`) is
    caught by inspecting EVERY segment's own first token, never only the
    command string's first word. `bash -c 'rm -rf /'` is denied on `bash`
    itself, closing the nested-shell evasion without needing to parse
    inside the quoted argument at all. An unbalanced quote (unparseable)
    denies too — monotone toward DENY, the same posture `_probe_path`
    already takes on an unparseable probe entry."""
    if not isinstance(command, str) or not command.strip():
        return "empty or non-string command"
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        tokens = list(lex)
    except ValueError as exc:
        return f"command does not tokenize as a shell command ({exc})"
    if not tokens:
        return "empty command"
    segments: list[list[str]] = [[]]
    for tok in tokens:
        if tok in _SHELL_OPERATOR_TOKENS:
            segments.append([])
        else:
            segments[-1].append(tok)
    for seg in segments:
        if not seg:
            continue
        prog = seg[0].lower()
        if prog in _DENIED_PROGRAMS:
            return f"`{prog}` is a denied program (esc-lead-gate-R11 write-verb denylist)"
        if prog == "git" and len(seg) > 1 and seg[1].lower() in _DENIED_GIT_SUBCOMMANDS:
            return f"`git {seg[1]}` is a denied git subcommand (esc-lead-gate-R11 write-verb denylist)"
        if prog == "find" and any(t in ("-delete", "-exec", "-execdir") for t in seg[1:]):
            return "`find` with -delete/-exec/-execdir is denied (esc-lead-gate-R11 write-verb denylist)"
    return None


def _run_claim_command(command: str, cwd: str, deadline: float) -> tuple[bool, str, str]:
    """Re-executes `command` via `/bin/sh -c`, bounded by the TIME LEFT on
    the SAME shared per-decision `deadline` R3's own git calls already
    share (esc-lead-gate-R11 never grows the budget per-arm). The identical
    hardened shape to `_run_git` (own process group, real
    `tempfile.TemporaryFile()`s, never `subprocess.PIPE`/`.communicate()`,
    `killpg` on timeout then a flat, un-budgeted 1s reap) for the exact
    same escaped-grandchild reason `_run_git`'s own docstring measures —
    duplicated rather than shared with `_run_git` itself, deliberately: R3
    is an adversarially-hardened, heavily-reproduced code path (G20-G40),
    and this arm does not touch it. Returns `(True, sha256_hex, "")` on a
    completed run — `sha256(f"{returncode}\\n{stdout}")`, stdout and exit
    code only; stderr is deliberately EXCLUDED from the hash, since
    incidental warnings/progress text on stderr is frequently
    non-deterministic and would break reproducibility for reasons unrelated
    to the claim itself (an HONEST LIMIT, stated as one: a claim command
    whose STDOUT itself is non-deterministic is not caught by this hash at
    all) — or `(False, "", reason)` otherwise. Runs in `cwd` AS IT
    CURRENTLY STANDS, never an ephemeral checkout of `fix_head` (a further
    HONEST LIMIT: if the worktree has moved past `fix_head` since the relay
    was written, the re-executed output may reflect a state the lead never
    measured — the same trust boundary R3's own `probe` already carries)."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False, "", f"the shared decision budget was already exhausted — {_ESCAPE_HATCH_NOTE}"
    try:
        out_f = tempfile.TemporaryFile()
        err_f = tempfile.TemporaryFile()
    except OSError as exc:
        return False, "", f"could not open temp files ({exc})"
    try:
        proc = subprocess.Popen(["/bin/sh", "-c", command], stdout=out_f, stderr=err_f,
                                 cwd=cwd, start_new_session=True)
    except OSError as exc:
        out_f.close()
        err_f.close()
        return False, "", f"command failed to run ({exc})"
    try:
        rc = proc.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        out_f.close()
        err_f.close()
        return False, "", "command timed out (shared decision budget)"
    try:
        out_f.seek(0)
        out = out_f.read().decode("utf-8", errors="replace")
    finally:
        out_f.close()
        err_f.close()
    digest = hashlib.sha256(f"{rc}\n{out}".encode("utf-8")).hexdigest()
    return True, digest, ""


_OUTPUT_HASH_RE = re.compile(r"[0-9a-f]{64}")


# R12-BEGIN
# ==========================================================================
# esc-lead-gate-R12 v2 ("ANTICIPATE BEFORE THE FIX", fix round 1) — a
# differential witness, checked while the fix does not yet exist.
#
# v1 attached every obligation to the RELAY — written only after a fix
# commit already exists — so v1 was fillable after the fact (killed by its
# own pressure round). The FIRST implementation of v2 (this comment's own
# prior revision) keyed the pre-fix artifact by `block_ts` and required the
# unit branch's tip to equal that ONE BLOCK row's own `head_sha` exactly —
# which denies PERMANENTLY the moment two open second-round BLOCKs of
# different types sit at different shas (an acceptance-verifier BLOCK
# provoked at the tip, alongside an older open adversarial-audit BLOCK,
# cannot both be "the current tip" at once). CONTRACT-R12-fix1.md's design
# round (v2, M1'-M6') replaces that with the shape below.
#
# THE ARTIFACT is keyed by the unit branch's CURRENT TIP, never a block's
# own ts or sha: `.jammi/gate-state/<slug>.anticipation.<tip_sha>.json` =
# `{unit_branch, pre_fix_sha, covers: [block_ts, ...], attacks: {file:
# {command, hash, keys: [...]}}, residual_risk}`. READER 1 (an implementer
# dispatch onto a unit with an open second-round BLOCK) requires ONE such
# artifact at the branch's CURRENT tip that covers the UNION of every open
# BLOCK's own derived keys (`finding_locations` UNION `class_enumeration`),
# reduced to one required entry PER FILE (`_key_to_file`) — never one entry
# per raw line-key, and never "the newest block only" (that shape is
# exactly what produced the permanent-deny bug above: a lead-provoked
# acceptance-verifier BLOCK at the tip could otherwise retire an older,
# unrelated audit's 10-key obligation for free). Every recorded command is
# RE-EXECUTED by the hook itself, and its witness must hash-match, in the
# RESOLVED worktree for that branch — the git worktree `git worktree list
# --porcelain` reports for it, never `$CLAUDE_PROJECT_DIR`. Ordering
# evidence is: the branch's tip has not moved since this decision began,
# the resolved worktree's OWN `HEAD` equals that tip, and `git status
# --porcelain` in that worktree is EMPTY (an uncommitted fix denies, naming
# the dirty paths). This establishes exactly one thing, stated precisely
# everywhere it is claimed: the hook re-executed each recorded command at
# THIS INSTANT on a worktree whose HEAD is the tip with no uncommitted
# modifications. It does NOT establish that no fix exists off-tree (a
# stash, another branch, another worktree of the same repo) nor that the
# attack necessarily preceded any particular fix commit that will later
# land — "true by construction" is not claimed.
#
# READER 2 (the closing-verifier relay, post-fix) requires `attacks_post`
# with the SAME per-file keys (widened by every FILE the fix's own diff
# changed, `_fix_window`'s `fix_changed`, and every FILE a new-surface
# definition landed in), the SAME commands, re-run at `fix_head`; for every
# fix-changed file the PRE-FIX artifact covers, at least one such file's
# post-fix hash must differ from its own recorded pre-fix hash — the fix
# must have measurably moved something an attack measures. `outcome` is
# DERIVED from that inequality, never a lead-declared field.
#
# READER 3 (`ci/scripts/check_rigor_record.py`) is the half the lead cannot
# forge: a committed-record SHAPE check (every second-round BLOCK row's
# exported record carries a matching `docs/rigor/<slug>.anticipation.jsonl`
# whose every key is covered and whose every command passes THIS module's
# own `_r12_attack_command_denied`) that HARD FAILS on shape alone;
# re-execution against a real `block_sha`/`pre_fix_sha` checkout is
# ADVISORY (BSD/GNU divergence, path divergence — measured, never portable)
# and reported per row, never a hard fail.
#
# HONEST LIMITS (stated as plainly as R3's/R11's own, never softened):
# attack QUALITY is never judged by any of this — only that a command ran,
# against a worktree whose HEAD really was the broken tip with nothing
# uncommitted, and that its output measurably moved after the fix. The
# hook trusts the lead's own `unit:` line naming which branch is under
# work (a decoy line is not detected). A key that does not parse as
# `path[:line]` still requires an attack (mapped to itself as its own
# "file") rather than being silently exempted or denied outright — a
# malformed key is a re-keying problem for the NEXT relay, not a reason to
# block anticipation from ever being satisfiable. `rm
# .jammi/gate-state/<slug>.*` remains the operator escape hatch and
# destroys evidence rather than fixing anything; no R12 deny message ever
# names it as a remedy.
# ==========================================================================

# The nine domain-implementer agent_types READER 1 gates unconditionally
# (M6' widens gating to four more GATED-but-generic types, below, but ONLY
# when their dispatch actually names a unit — these nine are gated even
# with NO unit line at all, denied naming the missing line).
IMPLEMENTER_TYPES = {
    "ai-core", "bench", "cli", "cookbook", "db", "docs-ci", "numerics",
    "python", "wire-server",
}

# M6': `general-purpose`/`claude`/`fork`/`doc-updater` are harness/generic
# types a lead may dispatch for reasons unrelated to implementing a fix —
# so a dispatch of one of these with NO unit line stays allowed (bricking
# concern: ~40 units may carry open blocks at any time; a lead using one of
# these types for something unrelated must not be gated). When the prompt
# DOES name a unit with an open second-round BLOCK, the SAME anticipation
# requirement as `IMPLEMENTER_TYPES` applies — this closes the fail-open
# enumeration these four types previously sat in unconditionally.
_R12_EXTRA_GATED_TYPES = {"general-purpose", "claude", "fork", "doc-updater"}

# Attack-command execution has its OWN budget, separate from `_GIT_BUDGET_S`
# (5s, shared by the ref/worktree-resolution git calls — far too little to
# re-run the 10-53 attack keys the design round counted as definition-
# shaped added lines on the three real U4a fix windows: `git diff -U0
# <range> | grep -Ec '(fn|struct|enum|mod|impl|def|class|name:)'` returned
# 53, 28 and 10). `_ATTACK_BUDGET_S` bounds the WHOLE decision's worth of
# attack re-execution; `_ATTACK_PER_COMMAND_CAP_S` additionally caps any ONE
# command so a single hung attack cannot eat the whole shared budget
# silently. Per M1', git-bound work runs under its OWN, FRESH
# `_GIT_BUDGET_S` window BEFORE attack execution begins and again AFTER it
# ends — never inside the attack window, so a 6s attack command cannot
# exhaust the deadline the NEXT phase's git calls need.
_ATTACK_BUDGET_S = 300.0
_ATTACK_PER_COMMAND_CAP_S = 120.0

# M1': because a cancelled PreToolUse `command` hook has its output
# DISCARDED (fail-open — no decision at all, not a deny), this module
# SELF-BOUNDS well inside the harness's own cancellation deadline
# (`.claude/settings.json` pins `timeout: 420` on this hook's entry) via
# `signal.alarm`, so a hung or over-budget decision produces a real,
# logged DENY instead of silently vanishing into the harness's cancel path.
_SELF_ALARM_S = 330


def _install_self_alarm() -> None:
    """Installs a `SIGALRM` handler that DENIES (prints a reason to stderr,
    exits 2) if this process is still running `_SELF_ALARM_S` seconds after
    this call. Only ever called from `main()`'s `pre` branch — `start`/
    `stop` are best-effort writers with no attack-execution phase. A
    platform with no `SIGALRM` (this module targets POSIX; there is no
    Windows-hook deployment of this hook family today) silently skips
    installing it — an HONEST LIMIT, not a workaround: the harness's own
    `timeout` setting remains the only bound in that case, stated here
    rather than pretended away.

    esc-lead-gate-R12 fix round 2 advisory: BEFORE `os._exit(2)` (which
    skips normal interpreter cleanup — no atexit handlers, no child-process
    reaping), the handler `killpg`s `_CURRENT_ATTACK_PID`'s own process
    group when one is live — an attack subprocess mid-execution at the
    moment the alarm fires would otherwise be orphaned, continuing to run
    detached from the now-dead hook process."""
    def _handler(signum: int, frame: object) -> None:  # noqa: ARG001
        pid = _CURRENT_ATTACK_PID
        if pid is not None:
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        sys.stderr.write(
            f"lead-gate: internal self-bound ({_SELF_ALARM_S:g}s) exceeded during "
            "esc-lead-gate-R12 attack re-execution — denying rather than risk the "
            "harness's own PreToolUse cancellation (fail-open, output discarded) "
            "leaving this decision unresolved\n"
        )
        sys.stderr.flush()
        os._exit(2)
    try:
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(_SELF_ALARM_S)
    except (ValueError, AttributeError, OSError):
        pass


def _new_attack_deadline() -> float:
    return time.monotonic() + _ATTACK_BUDGET_S


def _derived_attack_keys(row: dict) -> set[str]:
    """The hook's OWN derived pre-fix attack-key set for ONE BLOCK row:
    `finding_locations` UNION `class_enumeration`, exact strings — never a
    lead-supplied list, matched as exact dict keys the same way R11's
    `claim_sites` already are, never by substring."""
    locs = {s for s in (row.get("finding_locations") or []) if isinstance(s, str)}
    enum = {s for s in (row.get("class_enumeration") or []) if isinstance(s, str)}
    return locs | enum


_KEY_LINE_SUFFIX_RE = re.compile(r":\d+(?:-\d+)?$")


def _key_to_file(key: str) -> str:
    """Strips a trailing `:<line>[-<line>]` from an attack key, best-effort
    — a `path:line` key maps to its own file; a key with no such suffix (an
    unusual finding_locations entry, or prose in parentheses) maps to
    ITSELF, unresolved — M1'/M2': this NEVER denies; it still requires an
    attack, filed under the key's own literal text as its "file"."""
    return _KEY_LINE_SUFFIX_RE.sub("", key)


def anticipation_artifact_path(sdir: Path, unit_slug: str, tip_sha: str) -> Path:
    """M1': keyed by the unit branch's CURRENT TIP — never a block's own
    `ts` (the pre-fix-round-1 shape, which denied permanently the moment
    two open blocks of different types sat at different shas)."""
    return sdir / f"{unit_slug}.anticipation.{_fs_safe(tip_sha)}.json"


# esc-lead-gate-R12 fix round 2 item 12: substrings a TEST RUNNER's own
# summary line prints that vary by wall-clock alone, never by the mechanism
# under attack — an UNMUTATED, execution-class command (e.g. `cargo test -p
# foo`) run TWICE naturally differs in ITS OWN timing text every run, so
# hashing RAW output would make reader 2's pre/post differential trivially
# "pass" (a hash inequality) on pure noise, never a real behavioural change.
_R12_VOLATILE_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    # cargo/libtest: "... finished in 0.42s" / "... finished in 1.03s".
    (re.compile(r"finished in [\d.]+s"), "finished in <T>s"),
    # pytest's own summary line: "=== 3 passed in 0.12s ===".
    (re.compile(r"in [\d.]+s ==="), "in <T>s ==="),
    # A bare "... in 0.12s" trailer some runners print without the "===".
    (re.compile(r"\bin [\d.]+s\b"), "in <T>s"),
)


def _r12_normalize_output(text: str) -> str:
    """esc-lead-gate-R12 fix round 2 item 12 — the ONE committed
    normalizer every witness hash runs its `stdout` through before
    hashing. Strips ONLY the volatile timing substrings above; every other
    byte (pass/fail counts, assertion text, stdout content) is untouched —
    a genuine behavioural difference still changes the normalized hash.
    This is deliberately narrow: it does not attempt to normalize paths,
    PIDs, or any other source of run-to-run variance not measured as an
    actual false-negative/false-positive risk (see the fixture's own
    docstring for the measured case this closes)."""
    normalized = text
    for pattern, replacement in _R12_VOLATILE_PATTERNS:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def _witness_hash(rc: int, stdout: str, stderr_first_line: str) -> str:
    """`sha256(rc + "\\n" + normalize(stdout) + "\\n" + first line of
    stderr)`. The stderr line is INCLUDED (unlike R11's
    `_run_claim_command`, which excludes stderr entirely): the design round
    measured that two DIFFERENT commands — a real script absent from a
    stale `$CLAUDE_PROJECT_DIR` checkout, and a nonexistent script — both
    produce `rc=127` with EMPTY stdout, and hash IDENTICALLY (`743c7850…`)
    under `sha256(rc+stdout)` alone; `rc=1` (a real, meaningful failure)
    does NOT collide with `rc=127` under either hashing scheme — the two
    are already distinguished by `rc` alone. The independent `rc in (126,
    127)`/empty-stdout vacuity check below still applies even with the
    stderr-disambiguated hash: a command that is simply broken can
    reproduce its own broken witness consistently, which proves nothing
    about the mechanism under attack. `stdout` is run through
    `_r12_normalize_output` FIRST (item 12) — an execution-class command's
    own wall-clock timing text must never be the thing that makes a pre/
    post differential "pass"."""
    return hashlib.sha256(
        f"{rc}\n{_r12_normalize_output(stdout)}\n{stderr_first_line}".encode("utf-8")
    ).hexdigest()


# esc-lead-gate-R12 fix round 2 advisory: the PID of the currently in-
# flight attack subprocess's OWN process group (it is always started with
# `start_new_session=True`, so its pgid equals its pid) — set only for the
# duration of one `_run_attack_command` call, read by `_install_self_alarm`'s
# handler so a SIGALRM firing mid-attack kills the child rather than
# leaking it as an orphan when `os._exit(2)` skips normal cleanup.
_CURRENT_ATTACK_PID: int | None = None


def _run_attack_command(command: str, cwd: str, deadline: float) -> tuple[bool, int | None, str, str, str]:
    """Re-executes `command` via `/bin/sh -c` in `cwd` (the RESOLVED
    worktree — see `_resolve_worktree_cwd` — never `$CLAUDE_PROJECT_DIR`),
    bounded by the LESSER of the time left on the shared `_ATTACK_BUDGET_S`
    decision deadline and the flat `_ATTACK_PER_COMMAND_CAP_S` per-command
    cap. The identical hardened Popen/TemporaryFile/killpg shape
    `_run_claim_command`/`_run_git` already use (own process group, real
    tempfiles, never `subprocess.PIPE`), duplicated deliberately for the
    same reason those two never share code. Returns `(True, rc, stdout,
    stderr_first_line, "")` on a completed run, or `(False, None, "", "",
    reason)` otherwise. `_CURRENT_ATTACK_PID` is published for the
    duration of the child's lifetime so the self-alarm handler can kill it
    too — cleared in a `finally`, never left stale after this returns."""
    global _CURRENT_ATTACK_PID
    remaining = min(deadline - time.monotonic(), _ATTACK_PER_COMMAND_CAP_S)
    if remaining <= 0:  # R12-RESIDUAL: requires the shared attack budget to already be exhausted BEFORE this specific command's own turn; not exercised by a fast self-test fixture (the caller-level equivalent at :2091 is the same residual)
        return False, None, "", "", "the shared attack budget (or the per-command cap) was already exhausted"
    try:
        out_f = tempfile.TemporaryFile()
        err_f = tempfile.TemporaryFile()
    except OSError as exc:
        return False, None, "", "", f"could not open temp files ({exc})"
    try:
        proc = subprocess.Popen(["/bin/sh", "-c", command], stdout=out_f, stderr=err_f,
                                 cwd=cwd, start_new_session=True)
    except OSError as exc:
        out_f.close()
        err_f.close()
        return False, None, "", "", f"command failed to run ({exc})"
    _CURRENT_ATTACK_PID = proc.pid
    try:
        rc = proc.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        out_f.close()
        err_f.close()
        return (False, None, "", "",
                f"command timed out (per-command cap {_ATTACK_PER_COMMAND_CAP_S:g}s / shared "
                f"{_ATTACK_BUDGET_S:g}s attack budget) — name the slow key and narrow the "
                "command")
    finally:
        _CURRENT_ATTACK_PID = None
    try:
        out_f.seek(0)
        stdout = out_f.read().decode("utf-8", errors="replace")
        err_f.seek(0)
        stderr = err_f.read().decode("utf-8", errors="replace")
    finally:
        out_f.close()
        err_f.close()
    stderr_first_line = stderr.splitlines()[0] if stderr.splitlines() else ""
    return True, rc, stdout, stderr_first_line, ""


def _resolve_worktree_cwd(unit_branch: str, project_dir: str, deadline: float) -> tuple[str | None, str | None]:
    """esc-lead-gate-R12: the COMPUTED cwd attack commands run in — the
    linked worktree `git worktree list --porcelain` resolves for
    `unit_branch`, NEVER `$CLAUDE_PROJECT_DIR` (a possibly different,
    possibly stale checkout of this same repository that may not carry
    the fix's own files at all — this git command itself still runs
    against `$CLAUDE_PROJECT_DIR`'s repo, since worktree metadata is
    shared across every linked worktree of one repository). `(path,
    None)` on success; `(None, deny_reason)` otherwise — no worktree
    registered for the branch is an HONEST LIMIT, not a workaround: an
    attack command needs real checked-out files to run against."""
    ok, out, _rc = _run_git(["worktree", "list", "--porcelain"], project_dir, deadline, [])
    if not ok:  # R12-RESIDUAL: requires the `git worktree list` subprocess call itself to fail (e.g. a broken git binary); not exercised by a fast self-test fixture
        return None, f"could not resolve a worktree for {unit_branch!r} (`git worktree list`) — {out}"
    found = None
    current_path = None
    for line in out.splitlines():
        if line.startswith("worktree "):
            current_path = line[len("worktree "):].strip()
        elif line.startswith("branch "):
            if line[len("branch "):].strip() == f"refs/heads/{unit_branch}":
                found = current_path
        elif line == "":
            current_path = None
    if found is None:
        return None, (
            f"no linked worktree resolves for {unit_branch!r} (`git worktree list --porcelain`) "
            "— an attack command needs a real checkout of the branch's own files, and none is "
            "registered (esc-lead-gate-R12)"
        )
    return found, None


def _r12_attack_command_denied(command: object, cwd: str, *, require_tracked_at: str | None = None,
                                project_dir: str | None = None, deadline: float | None = None) -> str | None:
    """The RELAXED sibling of R11's `_command_denied`, for esc-lead-gate-
    R12 attack commands ONLY (R11's own `claims` commands are UNCHANGED —
    this function is never called from there): `sh <path>` / `bash <path>`
    — where `<path>` resolves to a REAL, EXISTING file under `cwd` — is
    ALLOWED even though `bash`/`sh` are in R11's own `_DENIED_PROGRAMS`
    (today they are unconditionally denied, so `bash
    ci/scripts/test_bundle_cuda_libs.sh` — the exact command that found
    this escape — is unrecordable). Every OTHER denied program/subcommand
    (`rm`, `curl`, `docker`, `git push/reset/clean/gc`, `find -delete`, a
    BARE `bash -c '...'`, `curl … | sh`, …) stays denied exactly as R11
    leaves it — this widens NOTHING but the one 'invoke a real repo
    script directly, with no further arguments' shape.

    M3': when `require_tracked_at` (a resolved sha) is given, `<path>` must
    ALSO be git-TRACKED at that sha (`git ls-tree`) — READER 3's own,
    stricter form, so CI never executes an untracked lead-written file from
    a detached checkout. Tracked-ness buys REVIEWABILITY, not safety: a
    script the lead committed or modified in the SAME PR is tracked and CI
    still runs it; the controls remain the write-verb denylist below, the
    human SWARM_GATE_TOUCHED review, and same-repo `on: pull_request`."""
    if not isinstance(command, str) or not command.strip():  # R12-RESIDUAL: both real callers (_r12_validate_and_run_entry) already reject an empty/non-string `command` before this function is ever called; dead code from every reachable caller, kept as defense-in-depth
        return "empty or non-string command"
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        tokens = list(lex)
    except ValueError as exc:
        return f"command does not tokenize as a shell command ({exc})"
    if not tokens:  # R12-RESIDUAL: a string non-empty after `.strip()` that tokenizes to zero shlex tokens is not constructible by any real caller's own upstream validation; not exercised by a fast self-test fixture
        return "empty command"
    segments: list[list[str]] = [[]]
    for tok in tokens:
        if tok in _SHELL_OPERATOR_TOKENS:
            segments.append([])
        else:
            segments[-1].append(tok)
    for seg in segments:
        if not seg:
            continue
        prog = seg[0].lower()
        if prog in ("bash", "sh") and len(seg) == 2 and not seg[1].startswith("-"):
            try:
                base = Path(cwd).resolve()
                candidate = (base / seg[1]).resolve()
                is_real = candidate.is_file() and str(candidate).startswith(str(base) + os.sep)
            except OSError:
                is_real = False
            if not is_real:
                return (f"`{prog} {seg[1]}` does not resolve to a real, existing file under {cwd} "
                         "(esc-lead-gate-R12's relaxed denylist only admits a real repo script)")
            if require_tracked_at is not None and project_dir is not None and deadline is not None:
                ok, out, _rc = _run_git(["ls-tree", "-r", "--name-only", require_tracked_at],
                                         project_dir, deadline, [])
                tracked = set(out.splitlines()) if ok else set()
                if seg[1] not in tracked:  # R12-RESIDUAL: this arm is only ever reached with `require_tracked_at` set, which only Reader 3 (ci/scripts/check_rigor_record.py) passes; it is exercised there (its own tracked-vs-untracked bash path fixtures), never through check_lead_gate.py's own FIXTURES list this sweep credits from
                    return (f"`{prog} {seg[1]}` is not git-TRACKED at {require_tracked_at[:12]}… "
                             "(esc-lead-gate-R12 M3': CI never executes an untracked lead-written "
                             "file from a detached checkout)")
            continue
        if prog in _DENIED_PROGRAMS:
            return f"`{prog}` is a denied program (esc-lead-gate-R11 write-verb denylist)"
        if prog == "git" and len(seg) > 1 and seg[1].lower() in _DENIED_GIT_SUBCOMMANDS:
            return f"`git {seg[1]}` is a denied git subcommand (esc-lead-gate-R11 write-verb denylist)"
        if prog == "find" and any(t in ("-delete", "-exec", "-execdir") for t in seg[1:]):
            return "`find` with -delete/-exec/-execdir is denied (esc-lead-gate-R11 write-verb denylist)"
    return None


# M2': "an artifact of only inspectors ... denies" — the precise condition
# is EVERY attack command being drawn exclusively from this list, never
# merely "lacks a command from the execution allowlist" (a `printf`/`echo`
# placeholder is neither an inspector nor on the execution allowlist, and
# is NOT itself grounds for this specific deny).
_R12_INSPECTOR_PROGRAMS = {"sed", "grep", "cat", "head", "awk", "rg", "wc", "tail", "ls"}
_R12_EXECUTION_PROGRAMS = {"cargo", "python3", "pytest", "make"}


def _r12_is_inspector_only(command: str) -> bool:
    """True iff EVERY shell segment of `command` invokes ONLY a program in
    `_R12_INSPECTOR_PROGRAMS` (a pipeline `grep -c foo | wc -l` is
    inspector-only end to end; `cargo test && sed -n 1p out` is NOT,
    because one segment is not an inspector at all)."""
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        tokens = list(lex)
    except ValueError:
        return False
    segments: list[list[str]] = [[]]
    for tok in tokens:
        if tok in _SHELL_OPERATOR_TOKENS:
            segments.append([])
        else:
            segments[-1].append(tok)
    real_segments = [seg for seg in segments if seg]
    if not real_segments:  # R12-RESIDUAL: `_r12_validate_and_run_entry`'s own denylist check (`_r12_attack_command_denied`, "empty command") already rejects a command with zero real segments before this is ever consulted; dead code from every reachable caller
        return False
    return all(seg[0].lower() in _R12_INSPECTOR_PROGRAMS for seg in real_segments)


def _r12_is_execution_class(command: str) -> bool:
    """esc-lead-gate-R12 fix round 2 F2: True iff AT LEAST ONE shell segment
    of `command` invokes a program in `_R12_EXECUTION_PROGRAMS`, or `bash`/
    `sh <path>` (the relaxed denylist's own real-tracked-script shape,
    already validated by `_r12_attack_command_denied` before this is ever
    consulted). This is the POSITIVE membership test the "at least one
    execution-class attack" requirement needs — `_R12_EXECUTION_PROGRAMS`
    was previously declared but never actually membership-tested anywhere,
    so the callers below inferred "execution-class" from merely "not
    inspector-only", which let a `printf`/`echo` placeholder (neither an
    inspector NOR an execution program) slip through both the inspector-
    only deny (it isn't one) and the execution requirement (nothing ever
    checked it WAS one)."""
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        tokens = list(lex)
    except ValueError:
        return False
    segments: list[list[str]] = [[]]
    for tok in tokens:
        if tok in _SHELL_OPERATOR_TOKENS:
            segments.append([])
        else:
            segments[-1].append(tok)
    for seg in segments:
        if not seg:
            continue
        prog = seg[0].lower()
        if prog in _R12_EXECUTION_PROGRAMS:
            return True
        if prog in ("bash", "sh") and len(seg) == 2 and not seg[1].startswith("-"):
            return True
    return False


def _r12_targeted_open_blocks(sdir: Path, unit_slug: str) -> list[tuple[str, dict]]:
    """M1': the UNION arm — every open `VERIFIER_SECOND_ROUND_TYPES` BLOCK
    on this unit, never "the newest only" (which produced the permanent-
    deny bug the module docstring above describes)."""
    return [
        (atype, row) for atype, row, _idx in open_blocks_for_unit(sdir, unit_slug)
        if atype in VERIFIER_SECOND_ROUND_TYPES
    ]


def _r12_required_by_file(targeted: list[tuple[str, dict]]) -> tuple[dict[str, set[str]], set[str]]:
    """`(by_file, covers)`: `by_file` maps EVERY FILE named by the UNION of
    every targeted open block's own derived keys to the set of raw keys
    that map to it (`_key_to_file`) — the per-file reduction M1' requires
    ("this unit's 50 keys cover 7 files"). `covers` is the set of every
    covered block's own `ts`, recorded into the artifact for provenance —
    never itself gated on."""
    by_file: dict[str, set[str]] = {}
    covers: set[str] = set()
    for _atype, row in targeted:
        ts = row.get("ts")
        if isinstance(ts, str) and ts:
            covers.add(ts)
        for key in _derived_attack_keys(row):
            by_file.setdefault(_key_to_file(key), set()).add(key)
    return by_file, covers


def _r12_find_pre_fix_artifact(sdir: Path, unit_slug: str, block_ts: str,
                                block_sha: str | None) -> dict | None:
    """esc-lead-gate-R12 fix round 2 F1: locates the pre-fix anticipation
    artifact that COVERS this BLOCK's own `ts`, never by re-deriving the
    artifact's filename from the BLOCK row's own `head_sha` alone. M1' keys
    the artifact by the branch's CURRENT TIP at the moment Reader 1 required
    it; when TWO open blocks of different types sit at different shas, the
    union's own covering artifact sits at the NEWER tip, and the OLDER
    block's own `head_sha` names no artifact file at all — the previous
    lookup treated that as "no pre-fix witness required" instead of "the
    witness this block needed is filed under a different name", silently
    allowing the older block's relay through with no differential check at
    all.

    Scans every `<unit_slug>.anticipation.*.json` artifact on disk — there
    is normally at most one live at a time (Reader 1 re-records at the
    CURRENT tip on every dispatch), but a prior dispatch's artifact can
    still be sitting on disk uncollected — and returns the most recently
    WRITTEN one (by mtime) whose own `covers` list names `block_ts`. An
    artifact recorded with NO `covers` field at all (the pre-union, single-
    artifact shape every existing fixture and every anticipation artifact
    written before a second block opened still produces) is treated as
    implicitly covering exactly the block it was filed under — i.e. it
    counts only when its own `pre_fix_sha` equals `block_sha`, which is
    what "filed under" means for that shape; this is the ONLY place
    `block_sha` is still consulted, and only as a fallback for artifacts
    that predate the `covers` field.

    `None` when nothing on disk covers this block at all — the caller
    treats that as a DENY (a pre-fix witness Reader 1 required is simply
    missing), never a silent skip."""
    candidates: list[tuple[float, dict]] = []
    for p in sorted(sdir.glob(f"{unit_slug}.anticipation.*.json")):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        covers = data.get("covers")
        if isinstance(covers, list) and covers:
            if block_ts in covers:
                candidates.append((p.stat().st_mtime, data))
            continue
        if isinstance(block_sha, str) and block_sha and data.get("pre_fix_sha") == block_sha:
            candidates.append((p.stat().st_mtime, data))
    if not candidates:
        return None
    candidates.sort(key=lambda mc: mc[0])
    return candidates[-1][1]


_R12_MAIN_REF_CANDIDATES = ("main", "master")


def _r12_changed_file_set(project_dir: str, deadline: float, tip: str) -> tuple[set[str] | None, str | None]:
    """M2': the unit's own `merge-base(main, tip)..tip` changed-file set,
    used ONLY to constrain lead-chosen keys on the empty-derived-set path
    (below) — never to narrow the by-file requirement itself. Tries `main`
    then `master`; `(None, reason)` if neither resolves (an honest failure,
    denied naming it, never silently bypassed)."""
    for ref in _R12_MAIN_REF_CANDIDATES:
        ok, base, _rc = _run_git(["merge-base", "--end-of-options", ref, tip], project_dir, deadline, [])
        if not ok:
            continue
        base = base.strip()
        if not base:
            continue
        ok2, out, _rc2 = _run_git(["diff", "--name-only", "-z", "--end-of-options", base, tip],
                                   project_dir, deadline, [])
        if ok2:
            return {p for p in out.split("\0") if p}, None
    return None, (
        f"could not resolve a merge-base against {'/'.join(_R12_MAIN_REF_CANDIDATES)} to derive "
        f"the unit's own changed-file set for {tip[:12]}… (esc-lead-gate-R12)"
    )


def _r12_validate_and_run_entry(artifact_name: str, key: str, entry: object, cwd: str,
                                 attack_deadline: float, seen_pairs: dict[tuple[str, str], str],
                                 *, require_tracked_at: str | None = None,
                                 project_dir: str | None = None,
                                 git_deadline: float | None = None) -> tuple[str | None, str | None, bool]:
    """Validates, denylist-checks, RE-EXECUTES and hash-matches ONE
    `{command, hash}` attack entry keyed by `key` (a file, or a raw key
    mapped to itself). Returns `(error_or_None, command_or_None,
    is_inspector_only)` — shared by every R12 caller (reader 1's by-file
    loop, reader 1's empty-set fallback, and — via its own `hash` field
    name — reader 3's shape check) so a fix to this one function fixes
    every caller identically."""
    if not isinstance(entry, dict):
        return f"anticipation artifact {artifact_name} attacks[{key!r}] is not an object (esc-lead-gate-R12)", None, False
    command = entry.get("command")
    if not isinstance(command, str) or not command.strip():
        return f"anticipation artifact {artifact_name} attacks[{key!r}] has no `command` (esc-lead-gate-R12)", None, False
    recorded_hash = entry.get("hash")
    if not (isinstance(recorded_hash, str) and _OUTPUT_HASH_RE.fullmatch(recorded_hash)):
        return f"anticipation artifact {artifact_name} attacks[{key!r}] has no valid `hash` (esc-lead-gate-R12)", None, False
    pair = (command, recorded_hash)
    if pair in seen_pairs:
        return (f"anticipation artifact {artifact_name} attacks[{key!r}] and "
                 f"attacks[{seen_pairs[pair]!r}] reuse the IDENTICAL (command, hash) pair — a "
                 "templated attack is not a per-site examination (esc-lead-gate-R12)"), None, False
    seen_pairs[pair] = key
    deny = _r12_attack_command_denied(command, cwd, require_tracked_at=require_tracked_at,
                                       project_dir=project_dir, deadline=git_deadline)
    if deny is not None:
        return f"anticipation artifact {artifact_name} attacks[{key!r}] command is denied: {deny}", None, False
    remaining = attack_deadline - time.monotonic()
    if remaining <= 0:  # R12-RESIDUAL: requires a real attack to consume the FULL _ATTACK_BUDGET_S (300s default) between two entries to trigger through a live fixture; not exercised by a fast self-test fixture
        return (f"the shared {_ATTACK_BUDGET_S:g}s attack budget for this decision was already "
                 f"exhausted before attacks[{key!r}]'s command could run (esc-lead-gate-R12)"), None, False
    ok, rc, stdout, stderr_line, run_why = _run_attack_command(command, cwd, attack_deadline)
    if not ok:  # R12-RESIDUAL: the OSError/tempfile-open-failure paths inside _run_attack_command require breaking the filesystem or process-creation itself; not exercised by a fast self-test fixture (the budget-exhaustion sub-case of `ok=False` IS covered indirectly by the arm above, which returns before this line is even reached)
        return f"anticipation artifact {artifact_name} attacks[{key!r}] command could not be re-executed: {run_why}", None, False
    if rc in (126, 127) or stdout == "":
        return (f"anticipation artifact {artifact_name} attacks[{key!r}] command's re-executed "
                 f"run is VACUOUS (rc={rc}) — this proves nothing about the mechanism "
                 "(esc-lead-gate-R12)"), None, False
    actual_hash = _witness_hash(rc, stdout, stderr_line)
    if actual_hash != recorded_hash:
        return (f"anticipation artifact {artifact_name} attacks[{key!r}] command's re-executed "
                 f"output does not reproduce the recorded hash (recorded {recorded_hash[:12]}…, "
                 f"got {actual_hash[:12]}…) — the attack is not established (esc-lead-gate-R12)"), None, False
    return None, command, _r12_is_inspector_only(command)


def _r12_empty_set_rejection(artifact_name: str, attacks: object, unit_branch: str, tip: str,
                              project_dir: str, git_deadline: float, attack_deadline: float,
                              cwd: str) -> str | None:
    """M2': an EMPTY derived set (an `uncertain` BLOCK with no findings; an
    unparseable verifier row) can never be satisfied by writing an empty
    file — the artifact must instead carry >=2 LEAD-CHOSEN keys, each
    naming a file in the unit's own `merge-base(main, tip)..tip` changed-
    file set, with distinct commands, distinct hashes, and at least one
    execution-class. Two keys in an untouched file, or two lines of one
    file read with two different `sed -n` invocations, do not satisfy
    this."""
    if not isinstance(attacks, dict) or len(attacks) < 2:
        return (f"anticipation artifact {artifact_name} carries no derived keys (an uncertain "
                 "BLOCK with no findings) and fewer than 2 lead-chosen attack keys — an "
                 "anticipation can never be satisfied by writing an empty file (esc-lead-gate-R12)")
    changed, why = _r12_changed_file_set(project_dir, git_deadline, tip)
    if why is not None:
        return why
    seen_pairs: dict[tuple[str, str], str] = {}
    all_inspector = True
    any_execution_class = False
    for key, entry in attacks.items():
        f = _key_to_file(key)
        if f not in changed:
            return (f"anticipation artifact {artifact_name} attacks[{key!r}] names {f!r}, which "
                     f"is not in the unit's own changed-file set against "
                     f"{'/'.join(_R12_MAIN_REF_CANDIDATES)} — a lead-chosen key on the empty-set "
                     "path must attack a file the unit actually touches (esc-lead-gate-R12)")
        err, command, is_inspector = _r12_validate_and_run_entry(
            artifact_name, key, entry, cwd, attack_deadline, seen_pairs)
        if err is not None:
            return err
        all_inspector = all_inspector and is_inspector
        if command is not None and _r12_is_execution_class(command):
            any_execution_class = True
    if all_inspector:
        return (f"anticipation artifact {artifact_name} carries only inspector-class commands — "
                 "at least one lead-chosen attack must be execution-class (sh|bash <tracked "
                 "path>, cargo, python3, pytest, make) (esc-lead-gate-R12)")
    if not any_execution_class:
        return (f"anticipation artifact {artifact_name} carries no execution-class attack — "
                 "esc-lead-gate-R12 F2: at least one command's first token must actually be "
                 "cargo/python3/pytest/make or a `sh|bash <tracked path>` invocation; a "
                 "placeholder command (e.g. printf/echo) is neither inspector-class nor "
                 "execution-class and proves nothing about the mechanism (esc-lead-gate-R12)")
    return None


def _pre_fix_anticipation_rejection(sdir: Path, unit_slug: str, unit_branch: str, tip: str,
                                     by_file: dict[str, set[str]], covers: set[str],
                                     project_dir: str, git_deadline: float,
                                     attack_deadline: float) -> str | None:
    """esc-lead-gate-R12 READER 1 (M1'/M2'). `None` iff a complete, EXECUTED
    anticipation artifact exists at the branch's CURRENT tip covering the
    UNION of every open second-round BLOCK's derived keys, reduced to one
    entry per FILE, plus non-empty `residual_risk` and clean ordering
    evidence (see the module doc above for exactly what that evidence
    establishes and what it does not)."""
    path = anticipation_artifact_path(sdir, unit_slug, tip)
    if not path.exists():
        return (f"no anticipation artifact ({path.name}) at the current tip {tip[:12]}… — the "
                 "lead must attack the fix's own broken tip BEFORE dispatching the implementer "
                 "(esc-lead-gate-R12)")
    try:
        data = json.loads(path.read_text())
    except Exception:
        return f"anticipation artifact {path.name} is not valid JSON"
    if not isinstance(data, dict):
        return f"anticipation artifact {path.name} is not a JSON object"
    if data.get("pre_fix_sha") != tip:
        return (f"anticipation artifact {path.name} `pre_fix_sha` does not match its own "
                 f"filename's tip {tip[:12]}… (esc-lead-gate-R12)")
    relay_ub = data.get("unit_branch")
    if not (isinstance(relay_ub, str) and relay_ub.strip() and slugify(relay_ub) == unit_slug):
        return f"anticipation artifact {path.name} `unit_branch` does not name this unit"

    # Fix round 5 Z7: residual_risk (item 8c) and item 8a's gates SHAPE
    # (the VALUE of `rc` is never judged here — the pre-fix tip is
    # expected to be broken) both run through the ONE shared anticipation
    # validator every R12 reader now calls — never re-implemented per
    # reader.
    required_commands, required_commands_deny_reason = _r12_required_commands_or_deny()
    shape_why = _r12_anticipation_rejection([data], required_commands, check_attacks=False,
                                             gates_row=data, judge_gates_rc=False,
                                             required_commands_deny_reason=required_commands_deny_reason)
    if shape_why is not None:
        return shape_why

    # M1' ordering evidence: the tip has not moved since this decision
    # began, the resolved worktree's own HEAD equals it, and the tree is
    # clean — a moved tip or a dirty tree denies, never silently degrades.
    ok, tip_now, out = _run_git(
        ["rev-parse", "--verify", "--end-of-options", f"refs/heads/{relay_ub}^{{commit}}"],
        project_dir, git_deadline, [])
    if not ok:  # R12-RESIDUAL: requires the `git rev-parse` subprocess call itself to fail; not exercised by a fast self-test fixture
        return f"could not re-resolve {relay_ub!r} under refs/heads/ to confirm the pre-fix tip — {out}"
    if tip_now != tip:  # R12-RESIDUAL: requires the branch to advance in the narrow window BETWEEN the caller's own tip resolution and this internal re-check, inside ONE synchronous hook invocation — a genuine race condition, not exercised by a fast self-test fixture (R12P5 tests a DIFFERENT, easier-to-construct case: the artifact simply not existing at an already-moved tip)
        return (f"{relay_ub!r}'s tip moved to {tip_now[:12]}… since this decision began (was "
                 f"{tip[:12]}…) — re-record the anticipation artifact at the new tip "
                 "(esc-lead-gate-R12)")

    cwd, why = _resolve_worktree_cwd(relay_ub, project_dir, git_deadline)
    if why is not None:
        return why
    ok, status_out, _rc = _run_git(["status", "--porcelain"], cwd, git_deadline, [])
    if not ok:  # R12-RESIDUAL: requires the `git status` subprocess call itself to fail; not exercised by a fast self-test fixture
        return f"could not check {cwd} for a clean tree (`git status --porcelain`) — {status_out}"
    if status_out.strip():
        dirty = [ln[3:].strip() if len(ln) > 3 else ln for ln in status_out.splitlines()[:5]]
        return (f"the worktree at {cwd} carries uncommitted changes ({', '.join(dirty)}) — "
                 "commit them or move them outside the worktree before anticipation attacks can "
                 "prove they ran against the tip exactly (esc-lead-gate-R12)")
    ok, head_now, _rc = _run_git(["rev-parse", "HEAD"], cwd, git_deadline, [])
    if not ok or head_now != tip:  # R12-RESIDUAL: `_resolve_worktree_cwd` only ever returns a path git ITSELF reports as `branch refs/heads/{unit_branch}` in `git worktree list --porcelain`, which makes that worktree's own HEAD (a symref to the SAME branch ref) equal `tip` by construction barring exotic repo corruption; not exercised by a fast self-test fixture
        return (f"the worktree at {cwd}'s HEAD ({head_now if ok else '?'}) does not match the "
                 f"branch tip {tip[:12]}… (esc-lead-gate-R12)")

    attacks = data.get("attacks")
    if not isinstance(attacks, dict):
        return f"anticipation artifact {path.name} carries no `attacks` object"

    if not by_file:
        return _r12_empty_set_rejection(path.name, attacks, relay_ub, tip, project_dir,
                                         git_deadline, attack_deadline, cwd)

    missing_files = [f for f in by_file if f not in attacks]
    if missing_files:
        return (f"anticipation artifact {path.name} omits {len(missing_files)} file(s) the open "
                 f"BLOCK(s)' own finding_locations/class_enumeration name, e.g. "
                 f"{sorted(missing_files)[:3]} (esc-lead-gate-R12)")

    # Every entry is individually validated (shape, denylist, RE-EXECUTED,
    # hash-matched) FIRST, in file order — a SPECIFIC per-entry failure
    # (hash mismatch, VACUOUS, a denylist hit) must be reported before any
    # aggregate judgement about the artifact's commands AS A SET. Only
    # once every entry individually passes does the shared shape validator
    # (Fix round 5 Z7 — the SAME function reader 3 calls, over the SAME
    # `attacks`, without ever re-executing) judge the pair-reuse and
    # execution-class-requirement arms that only make sense in aggregate.
    seen_pairs: dict[tuple[str, str], str] = {}
    for f in sorted(by_file):
        err, _command, _is_inspector = _r12_validate_and_run_entry(
            path.name, f, attacks.get(f), cwd, attack_deadline, seen_pairs)
        if err is not None:
            return err
    return _r12_anticipation_rejection([data], [], check_attacks=True, required_files=set(by_file))


def _post_fix_attacks_rejection(row: dict, data: dict, project_dir: str, cwd: str,
                                 attack_deadline: float, pre_by_file: dict[str, set[str]],
                                 new_surfaces: dict[str, str] | None, fix_changed: list[str] | None,
                                 anticipation_data: dict | None) -> str | None:
    """esc-lead-gate-R12 READER 2 (M4'), post-fix. `attacks_post` REPLACES
    `open_question` (a relay still carrying the old field with no
    `attacks_post` is denied naming the migration). The required key set is
    `keys(ARTIFACT) ∪ new_surfaces`, PER FILE (M4' widens by file for
    every changed file — the finer, per-rewritten-definition refinement
    M4' also describes for `.rs`/`.py` is NOT implemented here; this is a
    documented simplification, never a weaker per-file guarantee: every
    changed file still needs its own covering key). Every required file's
    `command` must match the pre-fix artifact's own command for that SAME
    file, re-execute non-vacuously at `fix_head`, and — for every fix-
    changed file the PRE-FIX artifact covers — at least one such file's
    post-fix hash must differ from its own recorded pre-fix hash. `outcome`
    is DERIVED from that inequality, never a lead-declared field. HONEST
    LIMIT: a changed file covered by NO pre-fix key (only brand-new attack
    sites) is not differential-checked here — there is no pre-fix witness
    for a key that did not exist before the fix, and this hook does not
    stand up an ephemeral `block_sha` checkout to synthesize one (see
    Reader 3, `check_rigor_record.py`, which independently re-derives from
    a real checkout instead)."""
    new_surface_files = {_key_to_file(k) for k in (new_surfaces or {}).keys()}
    # F3: widen by EVERY file the fix's own diff changed, not merely the
    # artifact's keys and new-surface definitions — a fix-changed file with
    # no covering key at all was previously invisible to this check.
    required = set(pre_by_file) | new_surface_files | set(fix_changed or [])
    if "open_question" in data and not (isinstance(data.get("attacks_post"), dict) and data.get("attacks_post")):
        return (
            "relay carries `open_question`, which esc-lead-gate-R12 REPLACES with "
            "`attacks_post` (post-fix, keyed like the pre-fix anticipation artifact's own "
            "`attacks`, per FILE) — see the anticipation artifact for the schema "
            "(esc-lead-gate-R12)"
        )
    if not required:
        return None
    if pre_by_file and not (isinstance(anticipation_data, dict) and isinstance(anticipation_data.get("attacks"), dict)):
        # Defensive re-check: Reader 1 should already have required a real
        # pre-fix anticipation artifact before the implementer was ever
        # dispatched — if it is missing HERE, the differential below would
        # be vacuously satisfied (nothing to compare against), so this
        # denies explicitly rather than silently skipping the check.
        return (
            "no pre-fix anticipation artifact found for this BLOCK — the differential below "
            "requires a real pre-fix witness to compare against (esc-lead-gate-R12)"
        )
    attacks_post = data.get("attacks_post")
    if not isinstance(attacks_post, dict):
        return (f"relay carries no `attacks_post` object, but {len(required)} attack site(s) "
                f"are required, e.g. {sorted(required)[:3]} (esc-lead-gate-R12)")
    missing = [k for k in required if k not in attacks_post]
    if missing:
        return f"relay `attacks_post` omits {len(missing)} required site(s), e.g. {sorted(missing)[:3]} (esc-lead-gate-R12)"

    pre_attacks = (anticipation_data or {}).get("attacks") if isinstance(anticipation_data, dict) else None
    if not isinstance(pre_attacks, dict):
        pre_attacks = {}

    seen_pairs: dict[tuple[str, str], str] = {}
    post_hashes: dict[str, str] = {}
    for f in sorted(required):
        entry = attacks_post.get(f)
        if not isinstance(entry, dict):
            return f"relay `attacks_post[{f!r}]` is not an object (esc-lead-gate-R12)"
        command = entry.get("command")
        pre_entry = pre_attacks.get(f)
        if isinstance(pre_entry, dict) and isinstance(pre_entry.get("command"), str):
            if isinstance(command, str) and pre_entry["command"] != command:
                return (f"relay `attacks_post[{f!r}]` command differs from the pre-fix "
                         "anticipation's own command for the SAME file — the differential "
                         "requires the SAME command (esc-lead-gate-R12)")
        err, _command, _is_exec = _r12_validate_and_run_entry(
            "relay", f, entry, cwd, attack_deadline, seen_pairs)
        if err is not None:
            return err.replace("anticipation artifact relay", "relay `attacks_post`")
        post_hashes[f] = entry.get("hash")

    for f in (fix_changed or []):
        if f not in pre_by_file:
            continue
        pre_hash = pre_attacks.get(f, {}).get("hash") if isinstance(pre_attacks.get(f), dict) else None
        if pre_hash is not None and pre_hash == post_hashes.get(f):
            return (
                f"the fix changed {f!r} but its pre-fix attack site reproduces the IDENTICAL "
                "post-fix hash — the fix did not observably move anything these attacks "
                "measure (esc-lead-gate-R12)"
            )
    return None


# ==========================================================================
# esc-lead-gate-R12 fix round 3, item 8 (the fold-9 replacement: 8a/8b/8c).
# ==========================================================================

_R12_REQUIRED_COMMANDS_FILENAME = "ci/lead-gate-required-commands.txt"


def _r12_required_commands_path() -> Path:
    return repo_root() / _R12_REQUIRED_COMMANDS_FILENAME


def _r12_required_commands_or_deny() -> tuple[list[str], str | None]:
    """item 8a, fix round 5 Z4: `ci/lead-gate-required-commands.txt` is a
    gate file IN FACT (human-amend-only, ratcheted, SWARM_GATE_TOUCHED-
    guarded), so a MISSING, EMPTY or all-comment file is now a hard DENY
    — never the honest-but-silently-disarming `[]` fix round 3 returned
    (which let deleting or emptying this committed file weaken item 8a
    with ZERO CI signal: every reader simply stopped requiring anything).
    Returns `(commands, deny_reason)`; `deny_reason` is `None` ONLY when
    the file exists and names at least one real command line."""
    path = _r12_required_commands_path()
    if not path.exists():
        return [], (f"{_R12_REQUIRED_COMMANDS_FILENAME} does not exist — item 8a's committed, "
                     "human-amend-only gate-command list must be present (esc-lead-gate-R12 "
                     "fix round 5 Z4)")
    out: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        command = stripped.split("  #", 1)[0].rstrip()
        if command:
            out.append(command)
    if not out:
        return [], (f"{_R12_REQUIRED_COMMANDS_FILENAME} exists but names no command line (empty "
                     "or all-comment) — item 8a's gate-command list must never silently disarm "
                     "(esc-lead-gate-R12 fix round 5 Z4)")
    return out, None


def _r12_gates_shape_rejection(artifact_name: str, gates: object, required_commands: list[str],
                                *, judge_rc: bool) -> str | None:
    """item 8a: `None` iff `gates` is a dict naming EVERY command in
    `required_commands` VERBATIM, each mapping to an object with an
    integer `rc`. `judge_rc=True` (Reader 2's post-fix posture) ALSO
    requires `rc == 0` for every line; `judge_rc=False` (Reader 1's
    pre-fix posture) never inspects the VALUE of `rc` — a lead-attested
    checklist against the tip BEFORE the fix, whose own rc is not itself
    a defect. The hook never runs these commands itself (see the
    committed file's own header for the honest-limit statement)."""
    if not required_commands:
        return None
    if not isinstance(gates, dict):
        return (f"{artifact_name} carries no `gates` object, but {len(required_commands)} "
                f"committed gate command(s) are required (esc-lead-gate-R12 item 8a)")
    missing = [c for c in required_commands if c not in gates]
    if missing:
        return (f"{artifact_name} `gates` omits {len(missing)} committed command(s), e.g. "
                f"{missing[:3]} (esc-lead-gate-R12 item 8a)")
    for c in required_commands:
        entry = gates.get(c)
        if not isinstance(entry, dict):
            return f"{artifact_name} `gates`[{c!r}] is not an object (esc-lead-gate-R12 item 8a)"
        rc = entry.get("rc")
        if not isinstance(rc, int) or isinstance(rc, bool):
            return f"{artifact_name} `gates`[{c!r}] carries no integer `rc` (esc-lead-gate-R12 item 8a)"
        if judge_rc and rc != 0:
            return (f"{artifact_name} `gates`[{c!r}] recorded rc={rc} (non-zero) — every "
                     "committed gate must be green at fix_head (esc-lead-gate-R12 item 8a)")
    return None


def _r12_anticipation_rejection(rows: list[dict], required_commands: list[str], *,
                                 check_attacks: bool,
                                 required_files: set[str] | None = None,
                                 gates_row: dict | None = None,
                                 judge_gates_rc: bool = False,
                                 required_commands_deny_reason: str | None = None) -> str | None:
    """esc-lead-gate-R12 fix round 5 Z7: the ONE anticipation-record
    shape validator shared by READER 1 (the hook, at dispatch — a
    singleton `rows=[<the live artifact>]`, called twice: once for
    `residual_risk`/`gates`, once more with `check_attacks=True` for the
    attacks shape/coverage, matching its own pre-existing check ORDER
    exactly) and READER 3 (`check_rigor_record.py`, every row the
    committed `docs/rigor/<slug>.anticipation.jsonl` carries, loaded from
    THIS module by path so the two validators can never drift). `rows`
    is never empty when called — the caller already denied on a missing/
    unparseable record before reaching here.

    Checks, in order:
      1. every row's `unit_branch` is a non-empty string, and every row's
         `residual_risk` is a non-empty string (item 8c) — a GENERIC
         non-empty check; reader 1 additionally requires its own
         `unit_branch` to match the dispatching unit's own slug, a
         stronger check it makes itself, before ever reaching here.
      2. `gates_row` (the SINGLE governing row — a checklist against one
         moment in time, never a union), when given, has `gates` shape-
         complete against `required_commands` (item 8a, via the ALREADY-
         shared `_r12_gates_shape_rejection`); `rc`'s VALUE is judged
         only when `judge_gates_rc`.
      3. (only when `check_attacks`) every attacks[*] entry, across the
         union of every row, is shape-valid (`command` non-empty, `hash`
         a valid 64-hex digest) — SHAPE only, never re-executed (a caller
         needing live re-execution, e.g. reader 1's real dispatch path,
         does that itself afterward via `_r12_validate_and_run_entry`).
      4. no (command, hash) pair repeats across ANY two keys in ANY row
         — a templated attack proves nothing about a per-site
         examination.
      5. at least one command, across the union, is EXECUTION-class
         (never only inspector-class) — enforced only when at least one
         attacks entry exists across `rows`.
      6. `required_files` (when non-empty) is fully covered by the union
         of every row's own `attacks` keys — the omits-a-command arm: an
         anticipation record that never attacks a required file at all
         denies here, identically in EITHER reader."""
    for row in rows:
        unit_branch = row.get("unit_branch")
        if not (isinstance(unit_branch, str) and unit_branch.strip()):  # R12-RESIDUAL: unreachable from reader 1's own call — its stronger pre-check (unit_branch must ALSO slugify to match the dispatching unit) already denies before this generic non-empty check is ever reached; exercised instead by check_rigor_record.py's own RR21 fixture, which the check_lead_gate.py-scoped sweep cannot see
            return "anticipation record carries a row with no non-empty `unit_branch` (esc-lead-gate-R12)"
        residual_risk = row.get("residual_risk")
        if not (isinstance(residual_risk, str) and residual_risk.strip()):
            return ("anticipation record carries no non-empty `residual_risk` — the one field "
                    "where the lead admits an unclosed site: which case does the attack you just "
                    "ran NOT cover? (esc-lead-gate-R12 item 8c)")

    if gates_row is not None:
        # Fix round 5 Z4: a MISSING/EMPTY/all-comment committed
        # required-commands file is a hard DENY here — never silently
        # treated as "no gate obligation" (which is what an empty
        # `required_commands` list, on its own, would otherwise mean).
        if required_commands_deny_reason is not None:
            return required_commands_deny_reason
        gates_why = _r12_gates_shape_rejection("the governing anticipation row", gates_row.get("gates"),
                                                required_commands, judge_rc=judge_gates_rc)
        if gates_why is not None:
            return gates_why

    if not check_attacks:
        return None

    covered: set[str] = set()
    all_inspector = True
    any_execution_class = False
    any_entry = False
    for row in rows:
        attacks = row.get("attacks")
        if not isinstance(attacks, dict):
            continue
        # Pair-reuse is scoped to WITHIN this ONE row — a templated attack
        # is two keys of the SAME artifact sharing one (command, hash);
        # two DIFFERENT rounds' rows (reader 3's `rows` spans every
        # exported round) legitimately re-attacking the same file the
        # same way is not that smell, and reader 1 only ever validates a
        # singleton `rows=[data]` anyway, so this is a no-op narrowing
        # for reader 1's own call.
        seen_pairs: dict[tuple[str, str], str] = {}
        for key, entry in attacks.items():
            if not isinstance(entry, dict):  # R12-RESIDUAL: reader 1 only ever reaches this on a key OUTSIDE `by_file` (a real per-file entry is already validated by `_r12_validate_and_run_entry`'s OWN identical check first) — narrow; exercised for a REQUIRED key via reader 1's real loop, and for reader 3 via its own RR27 fixture (fix round 6 Z14 deleted reader 3's earlier inline duplicate of this exact check, which had been shadowing it), invisible to this check_lead_gate.py-scoped sweep
                return f"anticipation record attacks[{key!r}] is not an object (esc-lead-gate-R12)"
            command = entry.get("command")
            if not isinstance(command, str) or not command.strip():  # R12-RESIDUAL: same narrowing as the arm above — only reachable via a key OUTSIDE `by_file` from reader 1's own call; exercised for reader 3 via its own RR28 fixture (fix round 6 Z14)
                return f"anticipation record attacks[{key!r}] has no `command` (esc-lead-gate-R12)"
            recorded_hash = entry.get("hash")
            if not (isinstance(recorded_hash, str) and _OUTPUT_HASH_RE.fullmatch(recorded_hash)):  # R12-RESIDUAL: same narrowing as the two arms above; exercised for reader 3 via its own RR29 fixture (fix round 6 Z14)
                return f"anticipation record attacks[{key!r}] has no valid `hash` (esc-lead-gate-R12)"
            pair = (command, recorded_hash)
            if pair in seen_pairs:  # R12-RESIDUAL: reader 1's real per-file loop already threads its OWN `seen_pairs` across `by_file` (`_r12_validate_and_run_entry`'s identical check fires first there); this arm fires only when a pair repeats via a key OUTSIDE `by_file`, exercised by check_rigor_record.py's own RR23 fixture (reader 3 has no `by_file` restriction at all), invisible to this sweep
                return (f"anticipation record attacks[{key!r}] and attacks[{seen_pairs[pair]!r}] "
                        "reuse the IDENTICAL (command, hash) pair — a templated attack is not a "
                        "per-site examination (esc-lead-gate-R12)")
            seen_pairs[pair] = key
            any_entry = True
            covered.add(_key_to_file(key))
            if _r12_is_execution_class(command):
                any_execution_class = True
            if not _r12_is_inspector_only(command):
                all_inspector = False

    if required_files:
        missing = required_files - covered
        if missing:  # R12-RESIDUAL: unreachable from reader 1's own second call — its OWN inline `missing_files` check (run before this function is ever called on the non-empty by_file path) already guarantees `required_files ⊆ covered` by construction; exercised instead by check_rigor_record.py's own RR20 fixture (reader 3 has no such pre-check), invisible to this check_lead_gate.py-scoped sweep
            return (f"anticipation record omits {len(missing)} file(s) the open BLOCK(s)' own "
                     f"finding_locations/class_enumeration name, e.g. {sorted(missing)[:3]} "
                     "(esc-lead-gate-R12)")

    if any_entry:
        if all_inspector:
            return ("anticipation record carries only inspector-class commands "
                     "(sed/grep/cat/head/awk/rg/wc/tail/ls) — at least one attack must be "
                     "execution-class (sh|bash <tracked path>, cargo, python3, pytest, make) "
                     "(esc-lead-gate-R12)")
        if not any_execution_class:
            return ("anticipation record carries no execution-class attack — esc-lead-gate-R12 F2: "
                     "at least one command's first token must actually be cargo/python3/pytest/"
                     "make or a `sh|bash <tracked path>` invocation; a placeholder command (e.g. "
                     "printf/echo) is neither inspector-class nor execution-class and proves "
                     "nothing about the mechanism (esc-lead-gate-R12)")
    return None


_R12_TEST_FAILURE_MARKERS = ("test result: FAILED", "FAIL —", "= FAILURES =")


def _mutations_rejection(row: dict, data: dict, new_surfaces: dict[str, str] | None) -> str | None:
    """item 8b: scoped to call sites in files the open BLOCK's own
    `finding_locations` name — armed by the DATA, never merely by
    `finding_locations` being non-empty (which is true of nearly every
    BLOCK and would require a `mutations` row on units whose fix never
    touched a finding's own file at all): the fix's own diff must ADD at
    least one new definition (`new_surfaces`, the hook's own derived
    enumeration, reduced to files) inside a file `finding_locations` also
    names. `None` iff the relay's `mutations` array then carries 1..3 rows
    — LABELED a sample, never exhaustive — each shaped `{site, command,
    rc_before, rc_after, marker_after}` with EITHER an ACCEPTED mutation
    (`rc_before == 0 ∧ rc_after != 0 ∧ marker_after` matches a committed
    TEST-failure marker, distinct from a build-failure marker) OR an
    explicit `uncovered` reason (R11's own disposition precedent, see
    `_claims_rejection`). NO hash-reproduction here — like `gates`, this
    is a lead-attested record, never re-executed by the hook."""
    finding_files = {_key_to_file(s) for s in (row.get("finding_locations") or []) if isinstance(s, str)}
    new_surface_files = {_key_to_file(k) for k in (new_surfaces or {}).keys()}
    scoped_files = finding_files & new_surface_files
    if not scoped_files:
        return None
    mutations = data.get("mutations")
    if not isinstance(mutations, list) or not mutations:
        return (f"relay carries no `mutations` array, but the fix's own diff adds a new "
                f"definition in {sorted(scoped_files)[:3]} — a file the BLOCK's own "
                "finding_locations also names — esc-lead-gate-R12 item 8b: record 1-3 "
                "lead-chosen mutation rows (a sample, never exhaustive)")
    if len(mutations) > 3:
        return (f"relay `mutations` carries {len(mutations)} row(s) — esc-lead-gate-R12 item 8b "
                "caps this at K<=3, a LABELED sample, never an exhaustive sweep")
    seen_uncovered: dict[str, int] = {}
    for i, entry in enumerate(mutations):
        if not isinstance(entry, dict):
            return f"relay `mutations`[{i}] is not an object (esc-lead-gate-R12 item 8b)"
        site = entry.get("site")
        if not isinstance(site, str) or not site.strip():
            return f"relay `mutations`[{i}] carries no `site` (esc-lead-gate-R12 item 8b)"
        command = entry.get("command")
        if not isinstance(command, str) or not command.strip():
            return f"relay `mutations`[{i}] carries no `command` (esc-lead-gate-R12 item 8b)"
        uncovered = entry.get("uncovered")
        if uncovered is not None:
            if not isinstance(uncovered, str) or not uncovered.strip():
                return f"relay `mutations`[{i}] `uncovered` is present but empty (esc-lead-gate-R12 item 8b)"
            # Fix round 5 Z11 (audit advisory 8): three identical `uncovered`
            # strings would otherwise satisfy the row-shape obligation three
            # times over with ONE real disposition — reuse R11's own
            # distinctness precedent (`_claims_rejection`'s uncovered-reason
            # check), never a length/word-count proxy.
            norm = _probe_normalize(uncovered)
            if norm in seen_uncovered:
                return (f"relay `mutations`[{i}] and `mutations`[{seen_uncovered[norm]}] carry the "
                         "IDENTICAL `uncovered` reason (normalized) — a templated disposition is "
                         "not a per-site examination (esc-lead-gate-R12 item 8b)")
            seen_uncovered[norm] = i
            continue
        rc_before = entry.get("rc_before")
        rc_after = entry.get("rc_after")
        marker_after = entry.get("marker_after")
        if not (isinstance(rc_before, int) and not isinstance(rc_before, bool)):
            return f"relay `mutations`[{i}] carries no integer `rc_before` (esc-lead-gate-R12 item 8b)"
        if not (isinstance(rc_after, int) and not isinstance(rc_after, bool)):
            return f"relay `mutations`[{i}] carries no integer `rc_after` (esc-lead-gate-R12 item 8b)"
        if not (isinstance(marker_after, str) and marker_after.strip()):
            return f"relay `mutations`[{i}] carries no `marker_after` (esc-lead-gate-R12 item 8b)"
        accepted = (rc_before == 0 and rc_after != 0
                    and any(m in marker_after for m in _R12_TEST_FAILURE_MARKERS))
        if not accepted:
            return (f"relay `mutations`[{i}] does not satisfy the ACCEPT rule (rc_before==0, "
                     "rc_after!=0, marker_after names a committed TEST-failure marker) and "
                     "carries no `uncovered` reason either (esc-lead-gate-R12 item 8b)")
    return None


def _r12_new_test_surfaces(new_surfaces: dict[str, str] | None) -> dict[str, str]:
    """item 8c: the SUBSET of `_parse_new_surfaces`' own enumeration
    (already the hook's derived, never lead-supplied, new-definition map)
    whose file path or definition name looks like a TEST — the file path
    contains "test" (case-insensitive) or the definition's own name
    starts with `test_`/`fixture_` or contains "test". A heuristic,
    stated as one: this can under- or over-include relative to a human's
    own judgment of "is this a test", the same class of limit
    `_parse_new_surfaces` itself already carries for "is this a
    definition"."""
    out: dict[str, str] = {}
    for key, name in (new_surfaces or {}).items():
        file_part = _key_to_file(key)
        looks_like_test_file = "test" in file_part.lower()
        looks_like_test_name = isinstance(name, str) and (
            name.startswith("test_") or name.startswith("fixture_") or "test" in name.lower()
        )
        if looks_like_test_file or looks_like_test_name:
            out[key] = name
    return out


def _r12_previous_relay_row(sdir: Path, unit_slug: str, row: dict) -> dict | None:
    """item 8c: the unit's own PREVIOUS row of the SAME `agent_type` (an
    earlier `ts`) — used only to compare `exclusions` text across rounds,
    never to gate anything else. `None` when there is no earlier row."""
    agent_type = row.get("agent_type") or ""
    cur_ts = row.get("ts") or ""
    if not agent_type or not cur_ts:
        return None
    path = sdir / f"{unit_slug}.jsonl"
    candidates = [r for r in read_rows(path)
                  if r.get("agent_type") == agent_type and isinstance(r.get("ts"), str) and r["ts"] < cur_ts]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["ts"])


def _exclusions_rejection(new_test_surfaces: dict[str, str], data: dict, unit_slug: str,
                           sdir: Path, row: dict) -> str | None:
    """item 8c: `None` iff every NEW test definition the fix's own diff
    adds (`new_test_surfaces`, derived from `_parse_new_surfaces`, never
    lead-supplied) carries a non-empty entry in the relay's `exclusions`
    object, each NORMALIZED-DISTINCT from its siblings in THIS relay and
    from the unit's own PREVIOUS relay of the SAME agent_type (an earlier
    `ts`) — the anti-templating cousin of `_claims_rejection`'s own
    uncovered-reason check. LIMIT, stated as plainly as R11's own: this
    cannot prove an exclusion is TRUE, only that two are not one."""
    if not new_test_surfaces:
        return None
    exclusions = data.get("exclusions")
    if not isinstance(exclusions, dict):
        return (f"relay carries no `exclusions` object, but the fix's own diff adds "
                f"{len(new_test_surfaces)} new test definition(s), e.g. "
                f"{list(new_test_surfaces)[:3]} — esc-lead-gate-R12 item 8c: each must name "
                "the case the attack you just ran does NOT cover")
    missing = [k for k in new_test_surfaces if k not in exclusions]
    if missing:
        return (f"relay `exclusions` omits {len(missing)} new test definition(s) the fix's own "
                f"diff adds, e.g. {missing[:3]} (esc-lead-gate-R12 item 8c)")
    normalized: dict[str, str] = {}
    for key in new_test_surfaces:
        reason = exclusions.get(key)
        if not isinstance(reason, str) or not reason.strip():
            return (f"relay `exclusions`[{key!r}] is empty — esc-lead-gate-R12 item 8c: name "
                     "the case this test does NOT cover")
        normalized[key] = _probe_normalize(reason)
    seen: dict[str, str] = {}
    for key, norm in normalized.items():
        if norm in seen:
            return (f"relay `exclusions`[{key!r}] and `exclusions`[{seen[norm]!r}] carry the "
                     "IDENTICAL exclusion (normalized) — a templated exclusion is not a "
                     "per-test examination (esc-lead-gate-R12 item 8c)")
        seen[norm] = key
    prev_row = _r12_previous_relay_row(sdir, unit_slug, row)
    if prev_row is not None:
        prev_path = relay_artifact_path(sdir, unit_slug, prev_row.get("agent_type") or "",
                                         prev_row.get("ts") or "")
        if not prev_path.exists():
            # Fix round 5 Z9: the ledger records a PRIOR round for this
            # unit (`prev_row` is real), but its own relay artifact is
            # gone — the cross-round distinctness check above cannot run
            # at all, which would silently DEGRADE to "no prior text to
            # collide with" instead of denying. The SAME shape as F1's
            # missing-pre-fix-artifact arm (lib:2122): a witness the
            # ledger says should exist, but doesn't, is a DENY, never a
            # silent skip.
            return (f"this unit's own previous relay row (agent_type={prev_row.get('agent_type')!r}, "
                    f"ts={prev_row.get('ts')!r}) names no on-disk artifact at {prev_path.name} — "
                    "the cross-round `exclusions` distinctness check cannot compare against a "
                    "missing prior witness (esc-lead-gate-R12 fix round 5 Z9)")
        try:
            prev_data = json.loads(prev_path.read_text())
        except Exception:
            prev_data = None
        if isinstance(prev_data, dict) and isinstance(prev_data.get("exclusions"), dict):
            prev_norms = {_probe_normalize(v) for v in prev_data["exclusions"].values() if isinstance(v, str)}
            for key, norm in normalized.items():
                if norm in prev_norms:
                    return (f"relay `exclusions`[{key!r}] repeats the IDENTICAL exclusion "
                             "(normalized) from this unit's own PREVIOUS relay — a templated "
                             "exclusion carried across rounds is not a per-round examination "
                             "(esc-lead-gate-R12 item 8c)")
    return None
# R12-END


def _claims_rejection(claim_sites: dict[str, str] | None, data: dict, project_dir: str,
                       deadline: float) -> str | None:
    """esc-lead-gate-R11: `None` iff every claim-shaped line the fix's own
    diff adds (`claim_sites`, the HOOK's OWN derived enumeration — never
    lead-supplied, the same posture R1 already takes toward the auditor's
    own `class_enumeration`, applied here to the fix instead of a finding)
    carries a disposition in the relay's `claims` object: either TESTED (a
    command whose re-executed output hash matches the one recorded) or
    explicitly marked UNCOVERED (a non-empty reason). Armed only when
    `claim_sites` is non-empty — a fix that adds no claim-shaped line
    carries no obligation, the same "armed by the DATA" posture R1 already
    takes toward `class_enumeration`."""
    if not claim_sites:
        return None
    claims = data.get("claims")
    if not isinstance(claims, dict):
        return (
            f"relay carries no `claims` object, but the fix's own diff adds "
            f"{len(claim_sites)} claim-shaped line(s), e.g. {list(claim_sites)[:3]} — every "
            "one must be TESTED (a command + its output hash) or explicitly marked "
            "UNCOVERED with a reason (esc-lead-gate-R11)"
        )
    missing = [k for k in claim_sites if k not in claims]
    if missing:
        return (
            f"relay `claims` omits {len(missing)} claim-shaped line(s) the fix's own diff "
            f"adds, e.g. {missing[:3]} — the obligation is DERIVED FROM THE DIFF, never "
            "declared by the lead (esc-lead-gate-R11)"
        )

    uncovered_reasons: dict[str, str] = {}
    for key in claim_sites:
        entry = claims.get(key)
        if not isinstance(entry, dict):
            return f"relay `claims[{key!r}]` is not an object (esc-lead-gate-R11)"
        status = entry.get("status")
        if status == "uncovered":
            reason = entry.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                return f"relay `claims[{key!r}]` is marked uncovered with no `reason` (esc-lead-gate-R11)"
            uncovered_reasons[key] = _probe_normalize(reason)
        elif status == "tested":
            command = entry.get("command")
            output_hash = entry.get("output_hash")
            if not isinstance(command, str) or not command.strip():
                return f"relay `claims[{key!r}]` is marked tested with no `command` (esc-lead-gate-R11)"
            if not (isinstance(output_hash, str) and _OUTPUT_HASH_RE.fullmatch(output_hash)):
                return (f"relay `claims[{key!r}]` is marked tested with no valid "
                        "`output_hash` (64 hex chars, sha256) (esc-lead-gate-R11)")
            deny = _command_denied(command)
            if deny is not None:
                return f"relay `claims[{key!r}]` command is denied: {deny}"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return (f"the shared git/claim budget for this decision was already exhausted "
                        f"before claims[{key!r}]'s command could run — {_ESCAPE_HATCH_NOTE}")
            ok, actual_hash, run_why = _run_claim_command(command, project_dir, deadline)
            if not ok:
                return f"relay `claims[{key!r}]` command could not be re-executed: {run_why}"
            if actual_hash != output_hash:
                return (
                    f"relay `claims[{key!r}]` command's re-executed output does not reproduce "
                    f"the recorded hash (recorded {output_hash[:12]}…, got {actual_hash[:12]}…) "
                    "— the claim is not established (esc-lead-gate-R11)"
                )
        else:
            return (f"relay `claims[{key!r}]` has an unrecognized `status` {status!r} "
                     "(must be `tested` or `uncovered`) (esc-lead-gate-R11)")

    # Anti-vacuity: two or more IDENTICAL (normalized) uncovered reasons in
    # the SAME relay is a templated, copy-pasted disposition, never a real
    # per-claim examination — the honest-limit cousin of `check_rigor_
    # record.py`'s own near-identical-CONTRACT check (the CONCEPT reused,
    # not the code: a normalized-text collision, never a length/word-count
    # rule). This CANNOT prove a reason is true; it can only prove two
    # reasons are not two.
    seen: dict[str, str] = {}
    for key, norm in uncovered_reasons.items():
        if norm in seen:
            return (
                f"relay `claims[{key!r}]` and `claims[{seen[norm]!r}]` carry the IDENTICAL "
                "uncovered reason (normalized) — a templated reason is not a per-claim "
                "examination (esc-lead-gate-R11)"
            )
        seen[norm] = key
    return None


def _relay_rejection(sdir: Path, unit_slug: str, row: dict,
                      warnings: list[str] | None = None) -> str | None:
    """`None` == accepted; otherwise the operator-facing REASON the relay is
    missing or insufficient. ONE predicate, ONE reachable caller
    (`_decide_verifier_dispatch`'s repeat-dispatch branch) — esc-097 (V10)
    deletes the earlier cross-type clearing arm that used to call a
    `check_fix=False` variant of this same check; there is no other caller
    left, so this always runs the FULL conjunction: R1 (coverage), R2
    (adjacent probing, always armed), and R3 (probe the fix).

    The relay's requirements are a CONJUNCTION, never a two-arm disjunction
    (esc-064): R1 (coverage) is armed by the DATA — a non-empty
    `class_enumeration` — and R2 (adjacent probing) is armed ALWAYS, for
    every gated verifier type's relay. The recorded `enumeration_missing`
    flag is diagnostic only and is never read here: a flag with a weak-arm
    default let a legacy/hand-edited row take the fallback arm and drop a
    requirement entirely.

    `warnings`, when given, is a caller-owned list this call EXTENDS (never
    replaces) with any accept-side git stderr text `_fix_window` observed
    (round-4) — the ONLY caller, `_decide_verifier_dispatch`, folds it into
    the operator-facing ALLOW reason so a ref that shadowed a sha or a
    branch name, even one that still resolved correctly, stays visible."""
    agent_type = row.get("agent_type") or ""
    block_ts = row.get("ts") or ""
    if not block_ts or not agent_type:
        return "BLOCK row carries no ts/agent_type to match a relay against"
    path = relay_artifact_path(sdir, unit_slug, agent_type, block_ts)
    if not path.exists():
        return "no relay artifact exists"
    try:
        data = json.loads(path.read_text())
    except Exception:
        return "relay artifact is not valid JSON"
    if not isinstance(data, dict):
        return "relay artifact is not a JSON object"
    if data.get("agent_type") != agent_type:
        return "relay `agent_type` does not match the BLOCK row's"
    if data.get("block_ts") != block_ts:
        return "relay `block_ts` does not match the BLOCK row's own ts"

    class_enum = [s for s in (row.get("class_enumeration") or []) if isinstance(s, str)]
    finding_locs = [s for s in (row.get("finding_locations") or []) if isinstance(s, str)]

    # R1 COVERAGE — content supplied by the enumeration; never gates R2.
    if class_enum:
        sites = data.get("sites")
        if not isinstance(sites, dict):
            return "relay has no `sites` object, but the BLOCK enumerated a class"
        if not all(isinstance(v, str) and v.strip() for v in sites.values()):
            return "relay `sites` has an empty/non-string disposition value"
        missing = [s for s in class_enum if s not in sites]
        if missing:
            return f"relay `sites` omits {len(missing)} enumerated site(s), e.g. {missing[:3]}"

    # R2 PROACTIVITY — ARMED UNCONDITIONALLY (the esc-064 fix).
    probe = data.get("probe")
    if not isinstance(probe, list):
        return ("relay carries no `probe` array — on a BLOCK the lead must probe "
                "ADJACENT to the class and name >=2 sites it EXAMINED and found "
                "clean (outside the enumeration/findings) before re-dispatching")
    reactive = {_probe_normalize(s) for s in class_enum} | {_probe_normalize(s) for s in finding_locs}
    adjacent = {_probe_normalize(p) for p in probe if isinstance(p, str) and _probe_normalize(p)} - reactive
    if len(adjacent) < 2:
        return (f"relay `probe` names {len(adjacent)} distinct site(s) outside the verifier's "
                "own class_enumeration/findings (>=2 required) — restating the enumeration "
                "is reactive acknowledgment, not adjacent probing; name >=2 sites you "
                "EXAMINED and found clean (esc-064)")

    # esc-lead-gate-R10 is REPLACED by esc-lead-gate-R12's `attacks_post`
    # (below) — `open_question` is no longer read here at all; a relay
    # still carrying it with no `attacks_post` is caught by
    # `_post_fix_attacks_rejection`'s own migration message.

    # R3 PROBE-THE-FIX (esc-097) — the only git subprocess in this module;
    # see its own module-doc paragraph for the arm order. esc-lead-gate-R11
    # shares this SAME deadline (minted ONCE, here) for its own claim-scan —
    # never a second, separate budget.
    deadline = _new_git_deadline()
    fix_changed, claim_sites, why, fix_warnings, new_surfaces = _fix_window(row, unit_slug, data, deadline)
    if warnings is not None:
        warnings.extend(fix_warnings)
    if why is not None:
        return why
    probe_paths = {p for p in (_probe_path(e) for e in probe) if p}
    if not any(p in fix_changed for p in probe_paths):
        n = len(fix_changed)
        block7 = (row.get("head_sha") or "")[:7]
        fix_head = data.get("fix_head")
        fix7 = fix_head[:7] if isinstance(fix_head, str) else "?"
        if n:
            sample = ", ".join(fix_changed[:3])
            example = f" — e.g. {sample}"
        else:
            # An empty fix window (fix_head differs from block_sha, e.g. an
            # empty/no-op commit) — no dangling "e.g." with nothing after it.
            example = " — the fix commit changed no files"
        return (
            f"relay `probe` names none of the {n} files the fix changed "
            f"(block {block7}..fix {fix7}); probe the fix, not the neighbourhood{example}"
        )

    # esc-lead-gate-R11 — "UNTESTED CLAIMS CARRY A TEST", required alongside
    # R3: the relay may not be accepted while the fix's OWN diff still
    # carries an untested claim of a testable shape. `claim_sites` is the
    # HOOK's own derived enumeration (`_parse_claim_sites`, run inside
    # `_fix_window`, sharing its git deadline) — never lead-asserted.
    claims_why = _claims_rejection(claim_sites, data, project_dir=os.environ.get("CLAUDE_PROJECT_DIR"),
                                    deadline=deadline)
    if claims_why is not None:
        return claims_why

    # esc-lead-gate-R12 fix round 3 item 8a — UNIVERSAL, never scoped to
    # only R12-anticipation-covered units: the relay's own `gates` object
    # must name every committed `ci/lead-gate-required-commands.txt` line
    # with `rc == 0` at fix_head. Fix round 6 Z13: this MUST go through the
    # same `_r12_required_commands_or_deny()` reader 1 already uses, never
    # the collapsing `_r12_required_commands()` accessor (`[]` on a
    # missing/empty/all-comment file) — that collapse fed straight into
    # `_r12_gates_shape_rejection`'s own `if not required_commands: return
    # None` early-out, so a relay with NO `gates` object at all was
    # ALLOWED the instant the committed required-commands file vanished or
    # was emptied, even though the identical relay DENIES with the file
    # present. A missing/empty/all-comment file is a hard DENY for every
    # relay here too, `gates` or not.
    required_commands, required_commands_deny_reason = _r12_required_commands_or_deny()
    if required_commands_deny_reason is not None:
        return required_commands_deny_reason
    gates_post_why = _r12_gates_shape_rejection("relay", data.get("gates"), required_commands,
                                                 judge_rc=True)
    if gates_post_why is not None:
        return gates_post_why

    # item 8b — armed by the DATA: a new definition in a file the BLOCK's
    # own finding_locations also names.
    mutations_why = _mutations_rejection(row, data, new_surfaces)
    if mutations_why is not None:
        return mutations_why

    # item 8c — armed by the fix's own diff adding a new TEST definition.
    exclusions_why = _exclusions_rejection(_r12_new_test_surfaces(new_surfaces), data, unit_slug, sdir, row)
    if exclusions_why is not None:
        return exclusions_why

    # esc-lead-gate-R12 READER 2 (M4', fix round 2 F1/F3) — the post-fix
    # differential. Runs in the SAME resolved worktree the fix landed in
    # (never $CLAUDE_PROJECT_DIR), bounded by its OWN `_ATTACK_BUDGET_S`,
    # separate from the git budget above. F1: the pre-fix artifact is
    # located by SCANNING for the one whose own `covers` list names this
    # BLOCK's `ts` (`_r12_find_pre_fix_artifact`) — never by re-deriving the
    # filename from the BLOCK row's own `head_sha`, which only ever equals
    # ONE covering artifact's filename when exactly one block was open at
    # dispatch time; a second, OLDER open block's own `head_sha` names no
    # artifact at all once the tip has moved past it, and the old lookup
    # silently treated "no file at that name" as "nothing required" rather
    # than "the witness is filed under a different name". `by_file_this_
    # row` is this ONE block's own derived-keys-to-files map, computed
    # independent of whatever the artifact contains, so a genuinely MISSING
    # (deleted, never-written) covering artifact still leaves `pre_by_file`
    # non-empty and the guard below fires instead of silently emptying the
    # requirement it exists to enforce. The required set is `keys(ARTIFACT)
    # ∪ new_surfaces ∪ fix_changed` (F3: every file the fix touched, not
    # only new-surface files), PER FILE — never merely this row's own
    # derived keys — so lead-chosen keys the empty-set path added, and any
    # file the fix changed with no covering key at all, are both
    # differentiated too.
    relay_ub = data.get("unit_branch")
    agent_type_local = row.get("agent_type") or ""
    by_file_this_row, _covers_this_row = _r12_required_by_file([(agent_type_local, row)])
    anticipation_data = None
    if isinstance(relay_ub, str) and relay_ub.strip():
        block_ts_local = row.get("ts")
        block_sha_local = row.get("head_sha")
        if isinstance(block_ts_local, str) and block_ts_local:
            anticipation_data = _r12_find_pre_fix_artifact(sdir, unit_slug, block_ts_local,
                                                             block_sha_local if isinstance(block_sha_local, str) else None)
    if isinstance(anticipation_data, dict) and isinstance(anticipation_data.get("attacks"), dict):
        pre_by_file: dict[str, set[str]] = {f: set() for f in anticipation_data["attacks"].keys()}
    elif by_file_this_row:
        # F1: no covering artifact was found (deleted, or never written) —
        # keep `pre_by_file` non-empty from this row's OWN derived keys so
        # the "no pre-fix anticipation artifact found" guard below actually
        # fires, instead of a missing file quietly emptying the set it was
        # supposed to enforce.
        pre_by_file = dict(by_file_this_row)
    else:
        pre_by_file = {}
    required_r12 = (set(pre_by_file) | {_key_to_file(k) for k in (new_surfaces or {}).keys()}
                    | set(fix_changed or []))
    if required_r12:
        project_dir = os.environ.get("CLAUDE_PROJECT_DIR")
        if not project_dir:
            return "hook needs CLAUDE_PROJECT_DIR for the esc-lead-gate-R12 reader-2 arm"
        cwd, cwd_why = _resolve_worktree_cwd(relay_ub, project_dir, deadline) if isinstance(relay_ub, str) else (None, "relay carries no `unit_branch`")
        if cwd_why is not None:
            return cwd_why
        attack_deadline = _new_attack_deadline()
        post_why = _post_fix_attacks_rejection(row, data, project_dir, cwd, attack_deadline,
                                                pre_by_file, new_surfaces, fix_changed, anticipation_data)
        if post_why is not None:
            return post_why
    return None


def _diagnose_row(row: dict) -> str:
    if row.get("verdict") == "UNPARSEABLE":
        reason = row.get("unparseable_reason") or "no valid verdict block found"
        return f" [UNPARSEABLE: {reason}]"
    raw = row.get("verdict_raw")
    # `raw == "BLOCK"` is the RECOGNIZED literal spelling for every card
    # whose vocabulary is `BLOCK | PASS` (adversarial-audit,
    # citation-checker, discipline-test-auditor) — it must never be
    # reported as "unrecognized ... defaulted to BLOCK"; that label is for
    # a raw value outside a card's own vocabulary, not the vocabulary's own
    # BLOCK spelling normalizing to the BLOCK verdict by construction.
    if raw is not None and raw != "BLOCK" and raw != _pass_word_for(row.get("agent_type") or ""):
        return f" [unrecognized verdict value {raw!r} defaulted to BLOCK]"
    if row.get("_corrupted"):
        return " [state row corrupted — treated as BLOCK]"
    return ""


# --------------------------------------------------------------------------
# THE ONE GATE. WHOLE-TOKEN binding only; no free-text parsing.
# --------------------------------------------------------------------------

# The character class a branch name / worktree path / sha token is drawn
# from. An anchor "names" a BLOCK only when it appears in the prompt as a
# WHOLE token over this class — never as a raw substring of a longer token
# (audit-r3 finding 1: an open BLOCK on `ci/gpu` must not deny the FIRST
# audit of `ci/gpu-dev`; `feat/x` vs `feat/x2`; a unit named `main` vs the
# word "domain"). `.` stays inside the class so `release-1` can never match
# inside `release-1.2`; the cost (a sha butted against a sentence-final `.`
# is not recognized) fails toward ALLOW, the same direction as the
# documented DODGE-5 residual, never toward a false DENY.
_TOKEN_CHARS = "A-Za-z0-9._/\\-"


def _whole_token_present(anchor: str, text: str, *, allow_path_under: bool = False) -> bool:
    """True iff `anchor` occurs in `text` delimited by non-token characters
    (or string edges) on both sides. With `allow_path_under`, a `/` may
    follow the anchor (a path UNDER the recorded worktree still names it)."""
    if not anchor:
        return False
    tail = rf"(?:(?=/)|(?![{_TOKEN_CHARS}]))" if allow_path_under else rf"(?![{_TOKEN_CHARS}])"
    pat = rf"(?<![{_TOKEN_CHARS}]){re.escape(anchor)}{tail}"
    return re.search(pat, text) is not None


def _sha_named(sha: str, text: str) -> bool:
    """True iff `text` carries, as a whole token, the full `sha` or any
    prefix of it that is at least 7 characters (this repo's short-sha
    convention). A hex token that merely STARTS with the 7-char prefix but
    is not itself a prefix of the recorded sha (e.g. a different commit
    sharing 7 leading characters) does not match."""
    if len(sha) < 7:
        return _whole_token_present(sha, text)
    pat = rf"(?<![{_TOKEN_CHARS}]){re.escape(sha[:7])}[0-9a-fA-F]*(?![{_TOKEN_CHARS}])"
    for m in re.finditer(pat, text):
        if sha.startswith(m.group(0)):
            return True
    return False


def _block_named_in_text(text: str, row: dict) -> bool:
    """WHOLE-TOKEN binding on strings the VERIFIER ITSELF emitted — the
    recorded `worktree` (or a path under it), the recorded `head_sha`
    (full, or a >=7-char prefix of it — this repo's short-sha convention),
    or the exact `unit_branch`. No path/site/message parsing at all; never
    a raw-substring match (audit-r3 finding 1)."""
    wt = row.get("worktree")
    if isinstance(wt, str) and wt and _whole_token_present(wt.rstrip("/"), text, allow_path_under=True):
        return True
    sha = row.get("head_sha")
    if isinstance(sha, str) and sha and _sha_named(sha, text):
        return True
    ub = row.get("unit_branch")
    if isinstance(ub, str) and ub and _whole_token_present(ub, text):
        return True
    return False


def _decide_verifier_dispatch(subtype: str, prompt: str, sdir: Path) -> tuple[bool, str]:
    """Denied iff the prompt whole-token-names an open BLOCK of the
    SAME agent_type (by worktree/head_sha/unit_branch) with no accepted
    relay artifact for that (unit, agent_type, block_ts). A first dispatch
    of this type is structurally never gated (no prior row exists to
    match). Residual: an unlabeled re-dispatch naming none of the three
    anchors (DODGE-5) — allowed, visible; the tell is a verifier row whose
    `worktree` differs from every binding on record.

    esc-097 (V11): this is the ONLY caller of `_relay_rejection` (which, in
    turn, is the ONLY place R3's git-dependent check runs). If the prompt
    whole-token-names MORE THAN ONE open BLOCK of this type, the whole
    dispatch is denied outright, naming every targeted unit — R3 is
    evaluated for exactly ONE unit per dispatch, never silently skipped for
    the others (an earlier draft ran R1/R2-only for every extra unit and
    R3 for just one, which meant a prompt naming two units could clear the
    SECOND one's BLOCK without ever probing its own fix). The remedy is to
    dispatch each named unit separately."""
    targeted = [
        (unit_slug, row) for unit_slug, atype, row, idx in all_open_blocks(sdir)
        if atype == subtype and _block_named_in_text(prompt, row)
    ]
    if not targeted:
        return True, (
            f"no open {subtype} BLOCK named (worktree/head_sha/unit_branch) in this "
            "dispatch — a first dispatch of this type, or an unlabeled re-dispatch "
            "(the documented visible residual)"
        )
    if len(targeted) > 1:
        names = ", ".join(u for u, _r in targeted)
        return False, (
            f"R3 is evaluated for one unit per dispatch; this prompt names {names} — "
            "dispatch them separately (a scratchpad path under a sibling worktree also names it)"
        )
    relay_warnings: list[str] = []
    unresolved = [(u, r, why) for u, r in targeted
                  for why in [_relay_rejection(sdir, u, r, relay_warnings)] if why is not None]
    if unresolved:
        names = "; ".join(f"{u}/{r.get('agent_type')}{_diagnose_row(r)}: {why}"
                           for u, r, why in unresolved)
        return False, (
            f"a second {subtype} dispatch naming {names} is denied — the relay artifact "
            "for that (unit, agent_type, block_ts) is missing or insufficient"
        )
    allowed_reason = f"every named {subtype} BLOCK has an accepted relay artifact — allowed"
    if relay_warnings:
        # Round-4: a ref shadowed a sha or a branch name and still resolved
        # to a correct, prefix-matching object — the ALLOW stands, but the
        # shadow is surfaced here rather than silently discarded.
        allowed_reason += " [git stderr on an otherwise-successful call: " + "; ".join(relay_warnings) + "]"
    return True, allowed_reason


# The dispatch payload's agent-type field, by every name it is known to (or
# may plausibly) travel under. The real harness schema is settled by the
# fresh-session acceptance log (`ci/hook-acceptance/README.md`); until then
# every spelling is checked, and a payload carrying NONE of them is a
# DISTINCT deny arm with its own remedy — never collapsed into the
# unknown-agent-type arm (audit-r3 finding 4, whose remedy "add '' to
# GATED_TYPES" was unrepresentable).
_SUBTYPE_KEYS = ("subagent_type", "agent_type", "subagentType", "agentType")


def _decide_dispatch(tool_input: dict, sdir: Path) -> tuple[bool, str]:
    subtype = _first_str(tool_input, _SUBTYPE_KEYS)
    if subtype is None:
        return False, (
            "dispatch payload carries no agent-type field (checked "
            f"{'/'.join(_SUBTYPE_KEYS)}; tool_input keys: {sorted(tool_input.keys())!r}) "
            "— failing closed. If the harness's real payload schema spells the field "
            "differently, add that spelling to _SUBTYPE_KEYS in lead-gate-lib.py; the "
            "fresh-session acceptance run (ci/hook-acceptance/README.md) settles the "
            "real schema"
        )
    if subtype in NEVER_GATED_TYPES:
        return True, f"never-gated agent type {subtype!r}"
    if subtype not in GATED_TYPES:
        return False, (
            f"unknown agent type {subtype!r} — add it to GATED_TYPES or "
            "NEVER_GATED_TYPES in lead-gate-lib.py (closed-world lattice, deny-unknown)"
        )
    if subtype not in VERIFIER_SECOND_ROUND_TYPES:
        if subtype in IMPLEMENTER_TYPES:
            # esc-lead-gate-R12 reader 1 — the pre-fix moment. A `unit:`
            # line is REQUIRED for these nine (M6') — an implementer
            # dispatch with no unit line is denied, naming the missing
            # line.
            prompt = _first_str(tool_input, ("prompt", "description")) or ""
            return _decide_implementer_dispatch(subtype, prompt, sdir, require_unit_line=True)
        if subtype in _R12_EXTRA_GATED_TYPES:
            # M6': these four GATED-but-generic types stay allowed with NO
            # unit line (a lead may dispatch them for reasons unrelated to
            # implementing a fix — ~40 units may carry an open block at any
            # time; gating an unlabeled dispatch of these types would brick
            # the lead). When the prompt DOES name a unit with an open
            # second-round BLOCK, the SAME anticipation requirement as
            # IMPLEMENTER_TYPES applies — this closes the fail-open
            # enumeration these four previously sat in unconditionally.
            prompt = _first_str(tool_input, ("prompt", "description")) or ""
            return _decide_implementer_dispatch(subtype, prompt, sdir, require_unit_line=False)
        # Implementer-dispatch binding is OUT OF SCOPE by the round-3 core
        # cut — every OTHER GATED non-verifier, non-implementer type
        # dispatch is always allowed.
        return True, f"agent type {subtype!r} is GATED but dispatch-binding is out of scope (§3 core)"
    prompt = _first_str(tool_input, ("prompt", "description")) or ""
    return _decide_verifier_dispatch(subtype, prompt, sdir)


def _decide_implementer_dispatch(subtype: str, prompt: str, sdir: Path, *,
                                  require_unit_line: bool) -> tuple[bool, str]:
    """esc-lead-gate-R12 READER 1 (M1'/M2'/M6'), the pre-fix moment. A
    dispatch that names, via a `unit_branch:`/`unit:` line (the SAME parse
    `handle_start` already uses to bind a branch — this is about WHICH unit
    the agent works on, never a whole-token scan of the whole prompt the
    way the verifier branch's anchor-matching is), a unit branch carrying
    ANY open `VERIFIER_SECOND_ROUND_TYPES` BLOCK is denied unless a valid,
    complete, PRE-FIX anticipation artifact exists at the branch's CURRENT
    tip covering the UNION of every such open block's derived keys
    (`_r12_required_by_file`). `require_unit_line=True` (the nine
    `IMPLEMENTER_TYPES`) denies a dispatch with no unit line at all, naming
    the missing line (M6' — a decoy-free zone: EVERY implementer dispatch
    must say which unit it works on); `require_unit_line=False` (the four
    `_R12_EXTRA_GATED_TYPES`) allows a dispatch naming no unit — nothing to
    check yet. A `unit:` line naming a branch that does not resolve under
    `refs/heads/` DENIES either way (M6') — never silently treated as "no
    unit named"."""
    unit_branch = None
    m = _UNIT_LINE_RE.search(prompt)
    if m:
        unit_branch = m.group(1)
    else:
        for pat in (_UNIT_BRANCH_COLON_RE, _UNIT_BRANCH_BARE_RE):
            m = pat.search(prompt)
            if m:
                unit_branch = m.group(1)
                break
    if unit_branch:
        unit_branch, _note = _normalize_unit_branch(unit_branch)
    if not unit_branch:
        if require_unit_line:
            return False, (
                f"implementer dispatch ({subtype!r}) names no unit branch — esc-lead-gate-R12 "
                "M6' requires every implementer dispatch to name the unit it works on via a "
                "`unit:` line"
            )
        return True, (
            f"dispatch ({subtype!r}) names no unit branch — nothing to check for an open "
            "verifier-type BLOCK (esc-lead-gate-R12 reader 1)"
        )
    unit_slug = slugify(unit_branch)
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR")
    if not project_dir:
        return False, "hook needs CLAUDE_PROJECT_DIR for the esc-lead-gate-R12 reader-1 arm"
    git_deadline = _new_git_deadline()
    ok, tip, git_out = _run_git(
        ["rev-parse", "--verify", "--end-of-options", f"refs/heads/{unit_branch}^{{commit}}"],
        project_dir, git_deadline, [])
    if not ok:
        return False, (
            f"dispatch ({subtype!r}) names unit branch {unit_branch!r}, which does not resolve "
            f"under refs/heads/ ({git_out}) — esc-lead-gate-R12 M6'"
        )
    targeted = _r12_targeted_open_blocks(sdir, unit_slug)
    if not targeted:
        return True, (
            f"dispatch ({subtype!r}) names unit {unit_slug!r} with no open verifier-type BLOCK "
            "— nothing to anticipate yet (esc-lead-gate-R12 reader 1)"
        )
    by_file, _covers = _r12_required_by_file(targeted)
    # A fresh, SEPARATE git-budget window for the ordering/artifact checks
    # below, minted AFTER the branch-resolution call above and BEFORE any
    # attack execution — M1': per-phase deadlines, never one budget shared
    # across ref resolution, attack execution, and any later git call.
    git_deadline = _new_git_deadline()
    attack_deadline = _new_attack_deadline()
    why = _pre_fix_anticipation_rejection(sdir, unit_slug, unit_branch, tip, by_file, _covers,
                                           project_dir, git_deadline, attack_deadline)
    if why is not None:
        return False, f"implementer dispatch onto unit {unit_slug!r} denied — {why}"
    return True, (
        f"dispatch onto unit {unit_slug!r} allowed — the anticipation artifact at tip "
        f"{tip[:12]}… covers every open verifier-type BLOCK's derived keys (esc-lead-gate-R12)"
    )


def decide_pre(payload: dict, sdir: Path) -> tuple[bool, str]:
    """`SendMessage` and `Bash` are OUT OF SCOPE by the round-3 core cut —
    both pass through as an unconditional allow. Only a fresh `Agent`/
    `Task` dispatch is decided."""
    tool_name = _first_str(payload, ("tool_name",)) or ""
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        tool_input = {}

    if tool_name in ("Agent", "Task"):
        return _decide_dispatch(tool_input, sdir)

    return True, f"n/a ({tool_name!r} is not gated in this design — §3 core cut)"


# --------------------------------------------------------------------------
# SubagentStart — write the ADVISORY-ONLY binding.
# --------------------------------------------------------------------------

def _first_str(d: dict, keys: tuple[str, ...]) -> str | None:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v:
            return v
    return None


def handle_start(payload: dict, sdir: Path) -> None:
    agent_id = _first_str(payload, ("agent_id", "id", "subagent_id")) or "UNKNOWN"
    agent_type = _first_str(payload, ("agent_type", "subagent_type")) or ""
    worktree = _first_str(payload, ("cwd", "worktree", "workdir", "working_directory"))

    unit_branch = _first_str(payload, ("unit_branch", "branch"))
    if not unit_branch:
        prompt = _first_str(payload, ("prompt", "description")) or ""
        if not prompt:
            tool_input = payload.get("tool_input")
            if isinstance(tool_input, dict):
                prompt = _first_str(tool_input, ("prompt", "description")) or ""
        m = _UNIT_LINE_RE.search(prompt)
        if m:
            unit_branch = m.group(1)
        else:
            for pat in (_UNIT_BRANCH_COLON_RE, _UNIT_BRANCH_BARE_RE):
                m = pat.search(prompt)
                if m:
                    unit_branch = m.group(1)
                    break
    if unit_branch:
        unit_branch, _note = _normalize_unit_branch(unit_branch)
    if not unit_branch:
        unit_branch = "UNBOUND"

    append_jsonl(bindings_file(sdir), {
        "ts": now_iso(), "agent_id": agent_id, "agent_type": agent_type,
        "unit_branch": unit_branch, "head_sha": None, "worktree": worktree,
    })


# --------------------------------------------------------------------------
# SubagentStop — write exactly one validated row per stop, and ONLY for a
# verifier-typed stop (STOP_MATCH_TYPES) — the settings.json SubagentStop
# `matcher` is meant to restrict invocation to this same set already, but
# the closed-world check is re-asserted HERE, inside the handler, as
# defense in depth: a matcher that does not filter as intended (e.g. because
# the payload's agent-type field travels under a spelling the matcher
# pattern doesn't anticipate) must not turn every non-verifier Stop event
# into an UNBOUND-bucket write. This is a closed-world MEMBERSHIP check
# only — never a free-text predicate — and it only ever SUPPRESSES a write;
# it can never deny anything (`stop` remains exit-0 always).
# --------------------------------------------------------------------------

# `all_open_blocks` re-reads and JSON-parses the WHOLE contents of every
# `<unit>.jsonl` file on every dispatch. An UNBOUND.jsonl that has
# accumulated to megabytes (every stop whose unit could not be resolved —
# most commonly, historically, a non-verifier stop that reached this
# handler before the STOP_MATCH_TYPES filter above existed — lands here)
# turns every future dispatch into a multi-megabyte parse. The cap is
# generous (2 MiB — comfortably past any single session's worth of
# legitimately-UNBOUND verifier rows, so it never fires on ordinary
# traffic) purely to bound the READ cost of a state directory that
# accumulated before this fix landed.
_UNBOUND_ROTATE_CAP_BYTES = 2 * 1024 * 1024


def _rotate_unbound_if_oversized(sdir: Path) -> None:
    """One-time migration, safe to call on every `stop` invocation: if the
    live `UNBOUND.jsonl` has grown past the cap, move it out of the LIVE
    `.jsonl` glob to a `.jsonl.1` sibling — `all_open_blocks`'s `entry.
    suffix != ".jsonl"` check already skips any file whose suffix is not
    exactly `.jsonl`, so the rotated sibling is invisible to it without any
    further change. This is a no-op for gate correctness: UNBOUND rows were
    never load-bearing for any gate decision (advisory-only), only for
    per-invocation read cost. Runs BEFORE the STOP_MATCH_TYPES filter below
    so a session that already carries an oversized file is rotated on its
    very first post-deploy `stop`, not stuck re-parsing it until its NEXT
    growth past the cap."""
    path = unit_file(sdir, "UNBOUND")
    try:
        if path.exists() and path.stat().st_size > _UNBOUND_ROTATE_CAP_BYTES:
            path.replace(path.parent / (path.name + ".1"))
    except OSError:
        pass


def handle_stop(payload: dict, sdir: Path) -> None:
    _rotate_unbound_if_oversized(sdir)

    agent_type = _first_str(payload, ("agent_type", "subagent_type")) or ""
    if agent_type not in STOP_MATCH_TYPES:
        # Closed-world membership, never a free-text RED: a non-verifier
        # stop writes NOTHING (no unit row, no UNBOUND append) — early
        # return before any parsing or binding-lookup work. A session
        # carries roughly 2700 non-subagent Stop-shaped events; a RED arm
        # here (rather than a silent skip) would jam nearly all of them.
        return

    agent_id = _first_str(payload, ("agent_id", "id", "subagent_id")) or "UNKNOWN"
    last_msg = _first_str(payload, ("last_assistant_message", "last_assistant_message_text")) or ""

    data, invalid_reason = extract_verdict_json(last_msg)
    fields = parse_verdict_fields(data)
    verdict, verdict_raw = normalize_verdict(data, agent_type)

    # UNPARSEABLE rows are filed under the agent's SubagentStart binding,
    # never under a shared "UNBOUND" bucket, unless no binding exists
    # either (the last resort — there is genuinely nothing else to file it
    # under). A parseable verdict's own unit_branch always wins first.
    unit_branch = fields["unit_branch"] or _lookup_binding_unit(agent_id, sdir) or "UNBOUND"

    prior = [
        r for r in read_rows(unit_file(sdir, unit_branch))
        if "verdict" in r and r.get("agent_type") == agent_type
    ]
    round_no = 1 + len(prior)

    append_jsonl(unit_file(sdir, unit_branch), {
        "ts": now_iso(), "agent_id": agent_id, "agent_type": agent_type,
        "unit_branch": unit_branch, "unit_branch_note": fields["unit_branch_note"],
        "head_sha": fields["head_sha"],
        "worktree": fields["worktree"],
        "verdict": verdict, "verdict_raw": verdict_raw,
        "unparseable_reason": invalid_reason if verdict == "UNPARSEABLE" else None,
        "round": round_no,
        "class_enumeration": fields["class_enumeration"],
        "enumeration_missing": fields["enumeration_missing"],
        "sweep_method": fields["sweep_method"], "exhaustive": fields["exhaustive"],
        "finding_locations": fields["finding_locations"],
        "recurrence_of_round": fields["recurrence_of_round"],
    })


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def _log_line(sdir: Path, event: str, tool_name: str, agent_type: str, decision: str,
              payload_keys: list[str], reason: str = "") -> None:
    try:
        line = {
            "ts": now_iso(), "event": event, "tool_name": tool_name,
            "agent_type": agent_type, "decision": decision,
            "payload_keys": sorted(payload_keys), "reason": reason,
        }
        sdir.mkdir(parents=True, exist_ok=True)
        with (sdir / "hook.log").open("a") as f:
            f.write(json.dumps(line, sort_keys=True))
            f.write("\n")
    except Exception:
        pass


def cmd_export(argv: list[str]) -> int:
    """esc-lead-gate-R7a (v2): `--export <unit_slug>` prints
    `unit_file(sdir, unit_slug)`'s own rows, one JSON object per line, to
    stdout — the hook's OWN row schema, verbatim, never hand-typed. A pure
    READ: no stdin, no payload, never touches `decide_pre` or any gate
    decision. The lead runs `python3 .claude/hooks/lead-gate-lib.py --export
    <slug> > docs/rigor/<slug>.jsonl` and commits the result — the carrier
    `ci/scripts/check_rigor_record.py` reads. Rows with no `verdict` key
    (a corrupted line `read_rows` already flags via `_unparseable_raw`) are
    skipped rather than exported malformed."""
    if len(argv) < 3 or not argv[2].strip():
        sys.stderr.write("lead-gate-lib: usage: lead-gate-lib.py --export <unit_slug>\n")
        return 2
    slug = argv[2]
    sdir = state_dir()
    rows = read_rows(unit_file(sdir, slug))
    n = 0
    for row in rows:
        if "verdict" not in row:
            continue
        sys.stdout.write(json.dumps(row, sort_keys=True))
        sys.stdout.write("\n")
        n += 1
    sys.stderr.write(f"lead-gate-lib: exported {n} row(s) for {slug!r}\n")
    return 0


def cmd_export_oracle(argv: list[str]) -> int:
    """esc-lead-invariants-2: `--export-oracle <unit_slug>` prints every
    `agent_type == "oracle"` row from `unit_file(sdir, unit_slug)`'s own
    rows, one JSON object per line, to stdout — the hook's OWN row schema,
    verbatim, never hand-typed. A pure READ: no stdin, no payload, never
    touches `decide_pre` or any gate decision.

    The lead runs `python3 .claude/hooks/lead-gate-lib.py --export-oracle
    <slug> > docs/rigor/<slug>.oracle.jsonl` and commits the result — the
    carrier `ci/scripts/check_oracle_gate.py` reads. Deliberately narrower
    than the general `--export` (it exports only oracle rows, to a distinct
    `*.oracle.jsonl` path) so this addition never collides with
    `cmd_export`'s own `--export` form — both coexist; a human may later
    fold this into the general form, which is not required for this one
    to work today."""
    if len(argv) < 3 or not argv[2].strip():
        sys.stderr.write("lead-gate-lib: usage: lead-gate-lib.py --export-oracle <unit_slug>\n")
        return 2
    slug = argv[2]
    sdir = state_dir()
    rows = read_rows(unit_file(sdir, slug))
    n = 0
    for row in rows:
        if "verdict" not in row or row.get("agent_type") != "oracle":
            continue
        sys.stdout.write(json.dumps(row, sort_keys=True))
        sys.stdout.write("\n")
        n += 1
    sys.stderr.write(f"lead-gate-lib: exported {n} oracle row(s) for {slug!r}\n")
    return 0


def cmd_export_anticipation(argv: list[str]) -> int:
    """esc-lead-gate-R12 READER 3: `--export-anticipation <unit_slug>`
    prints every `<unit_slug>.anticipation.*.json` artifact under
    `.jammi/gate-state/`, one JSON object per line, to stdout — each row
    carries `agent_type: "lead-anticipation"` and DELIBERATELY NO `verdict`
    key (a synthetic agent_type carrying a verdict would enter the
    closed-world gate lattice at `_unit_rows_by_agent_type`/`normalize_
    verdict`; this export is read-only evidence for
    `ci/scripts/check_rigor_record.py`, never gate state). A pure READ: no
    stdin, no payload, never touches `decide_pre` or any gate decision.

    Fix round 5 Z5: every exported row is stamped with `ts` (ISO-8601 UTC,
    derived from the artifact FILE's own mtime — the moment the lead's
    attack actually ran and wrote it, never the export's own wall-clock)
    and `head_sha` (the artifact's own `pre_fix_sha`, i.e. the tip its
    filename already encodes — the artifact IS keyed by this tip, so no
    extra bookkeeping is needed to know it). These two fields are what let
    `check_rigor_record.py`'s governing-row selection be ORDER-INDEPENDENT
    (select by `head_sha` match, else by the greatest `ts`) instead of by
    append/sort position, which the artifact's OWN filename (a content
    hash, not a timestamp) cannot support.

    Fix round 6 Z12: this command writes TWO files, never one. stdout (the
    operator redirects it into `docs/rigor/<slug>.anticipation.jsonl`)
    carries ONLY `lead-anticipation` rows. Any relay's non-empty
    `mutations`/`exclusions` — the lead's own attestations — are written
    DIRECTLY, by this command, to `docs/rigor/<slug>.attestation.jsonl`
    (`agent_type: "lead-relay-attestation"`, also no `verdict` key), never
    interleaved into the anticipation stream: two distinct row kinds
    committed as one file let an attestation row become the anticipation
    stream's own "governing" row, hiding a real `gates` object entirely —
    the bug `check_rigor_record.py`'s reader 3 now REFUSES on sight."""
    if len(argv) < 3 or not argv[2].strip():
        sys.stderr.write("lead-gate-lib: usage: lead-gate-lib.py --export-anticipation <unit_slug>\n")
        return 2
    slug = argv[2]
    sdir = state_dir()
    n = 0
    prefix = f"{slug}.anticipation."
    if sdir.exists():
        for entry in sorted(sdir.iterdir()):
            if not (entry.name.startswith(prefix) and entry.name.endswith(".json")):
                continue
            try:
                data = json.loads(entry.read_text())
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            row = dict(data)
            row.pop("verdict", None)
            row["agent_type"] = "lead-anticipation"
            try:
                mtime = entry.stat().st_mtime
                row["ts"] = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            except OSError:  # R12-RESIDUAL: requires the artifact file to vanish between the `iterdir()` listing and this `stat()` call, a TOCTOU race not exercised by a fast self-test fixture
                row["ts"] = now_iso()
            pre_fix_sha = data.get("pre_fix_sha")
            if isinstance(pre_fix_sha, str) and pre_fix_sha:
                row["head_sha"] = pre_fix_sha
            sys.stdout.write(json.dumps(row, sort_keys=True))
            sys.stdout.write("\n")
            n += 1

    # Fix round 5 Z8 / fix round 6 Z12: `mutations`/`exclusions` are the
    # lead's OWN attestations, written into the (gitignored, CI-invisible)
    # RELAY artifact — `.claude/hooks/README.md`/`lead.md`'s own "the
    # control is the human at merge, reading the exported record" sentence
    # was false until this export actually carried them anywhere a human
    # reviewing a diff could see them. Every `<slug>.relay.*.json` that
    # carries a non-empty `mutations` or `exclusions` field is exported —
    # but Z12 (closing audit #4) found that dumping these `lead-relay-
    # attestation` rows into the SAME STDOUT STREAM the operator redirects
    # into `docs/rigor/<slug>.anticipation.jsonl` put TWO ROW KINDS in ONE
    # committed file: neither reader-3 site filtered by `agent_type`, so an
    # attestation row (no `residual_risk`, no `gates`) could become the
    # governing row `check_required_gates` selects, or deny `check_
    # anticipation_witnesses` outright. PROPERTY (Z12): every committed
    # rigor stream carries exactly ONE row kind. Attestation rows are
    # therefore written to their OWN committed stream, `docs/rigor/<slug>.
    # attestation.jsonl` — a SEPARATE FILE, written directly here (never via
    # stdout, since one redirect cannot populate two distinct committed
    # artifacts) — DELIBERATELY NO `verdict` key for the same closed-world-
    # lattice reason as above.
    m = 0
    relay_prefix = f"{slug}.relay."
    attestation_rows: list[dict] = []
    if sdir.exists():
        for entry in sorted(sdir.iterdir()):
            if not (entry.name.startswith(relay_prefix) and entry.name.endswith(".json")):
                continue
            try:
                data = json.loads(entry.read_text())
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            mutations = data.get("mutations")
            exclusions = data.get("exclusions")
            has_mutations = isinstance(mutations, list) and mutations
            has_exclusions = isinstance(exclusions, dict) and exclusions
            if not (has_mutations or has_exclusions):
                continue
            row = {
                "agent_type": "lead-relay-attestation",
                "unit_branch": data.get("unit_branch"),
                "relay_agent_type": data.get("agent_type"),
                "block_ts": data.get("block_ts"),
                "fix_head": data.get("fix_head"),
            }
            if has_mutations:
                row["mutations"] = mutations
            if has_exclusions:
                row["exclusions"] = exclusions
            try:
                mtime = entry.stat().st_mtime
                row["ts"] = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            except OSError:  # R12-RESIDUAL: requires the artifact file to vanish between the `iterdir()` listing and this `stat()` call, a TOCTOU race not exercised by a fast self-test fixture
                row["ts"] = now_iso()
            fix_head = data.get("fix_head")
            if isinstance(fix_head, str) and fix_head:
                row["head_sha"] = fix_head
            attestation_rows.append(row)
            m += 1
    attestation_path = repo_root() / "docs" / "rigor" / f"{slug}.attestation.jsonl"
    if attestation_rows:
        attestation_path.parent.mkdir(parents=True, exist_ok=True)
        attestation_path.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in attestation_rows))
    sys.stderr.write(
        f"lead-gate-lib: exported {n} anticipation row(s) to stdout and {m} relay-attestation "
        f"row(s) to {attestation_path} for {slug!r}\n")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "--export-oracle":
        return cmd_export_oracle(argv)
    if len(argv) >= 2 and argv[1] == "--export-anticipation":
        return cmd_export_anticipation(argv)
    if len(argv) >= 2 and argv[1] == "--export":
        return cmd_export(argv)
    if len(argv) < 2 or argv[1] not in ("start", "stop", "pre"):
        sys.stderr.write(
            "lead-gate-lib: usage: lead-gate-lib.py {start|stop|pre} < payload.json "
            "| --export <unit_slug> | --export-oracle <unit_slug> | "
            "--export-anticipation <unit_slug>\n")
        return 2
    cmd = argv[1]

    sdir = state_dir()
    tool_name = ""
    agent_type = ""
    payload_keys: list[str] = []
    try:
        raw = sys.stdin.buffer.read()
        text = raw.decode("utf-8")
        # A payload that is not a JSON object RAISES into the fail-closed
        # boundary below — `pre` exits 2, `start`/`stop` exit 0 (best-effort
        # writers). Never silently coerced to `{}`: a UTF-8-valid but
        # JSON-invalid payload used to collapse to `{}` and ALLOW, the
        # decode-error sibling's fail-open twin (audit-r3 finding 3).
        if not text.strip():
            raise ValueError("empty hook payload on stdin")
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(
                f"hook payload is JSON {type(payload).__name__}, not an object")

        tool_name = _first_str(payload, ("tool_name",)) or ""
        agent_type = _first_str(payload, ("agent_type", "subagent_type")) or ""
        payload_keys = list(payload.keys())

        if cmd == "start":
            handle_start(payload, sdir)
            _log_line(sdir, "SubagentStart", tool_name, agent_type, "n/a (writer)", payload_keys)
            return 0
        if cmd == "stop":
            handle_stop(payload, sdir)
            _log_line(sdir, "SubagentStop", tool_name, agent_type, "n/a (writer)", payload_keys)
            return 0
        # cmd == "pre" — esc-lead-gate-R12 M1': self-bound well inside the
        # harness's own PreToolUse cancellation deadline (settings.json
        # pins it above `_SELF_ALARM_S`), since a cancelled hook's output
        # is DISCARDED (fail-open) rather than denied.
        _install_self_alarm()
        allow, reason = decide_pre(payload, sdir)
        _log_line(sdir, "PreToolUse", tool_name, agent_type, "allow" if allow else "deny",
                   payload_keys, reason)
        if allow:
            return 0
        sys.stderr.write(reason + "\n")
        return 2
    except Exception as exc:  # noqa: BLE001 — this IS the fail-closed boundary
        try:
            _log_line(sdir, cmd, tool_name, agent_type, "error", payload_keys, str(exc))
        except Exception:
            pass
        if cmd == "pre":
            sys.stderr.write(f"lead-gate: internal error — failing closed: {exc}\n")
            return 2
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
