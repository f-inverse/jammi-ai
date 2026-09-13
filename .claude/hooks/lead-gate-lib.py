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
  sites outside enumeration+findings) is armed ALWAYS. esc-lead-gate-R10:
  alongside those >=2 examined-clean sites, the relay must ALSO carry a
  non-empty `open_question` — a site examined and explicitly NOT closed,
  naming the attack for the next round to run. A one-line schema
  requirement over lead-authored text: a COST FLOOR, not proof of
  examination, the same limit `probe` itself already carries — it cannot
  be satisfied by writing nothing, and a human reads at merge whether the
  open question was real.

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
always) -> R10 (open question, always) -> CLAUDE_PROJECT_DIR is set ->
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

Explicitly OUT OF SCOPE by this cut (dropped entirely, not log-only):
`SendMessage` gating and all message-prose parsing; implementer-dispatch
binding; the Bash backstop (the mechanical control is `permissions.deny`
on the hook files in `.claude/settings.json`, unchanged); tell rows beyond
the existing one-line `hook.log` entry every invocation already writes.
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
(R3) — never on the hot first-dispatch path, never outside
`$CLAUDE_PROJECT_DIR`, and never for more than one targeted unit per
decision (see the esc-097 paragraph above; this amends the prior "no git
subprocess anywhere" doctrine to this narrower, explicit exception).
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


def _fix_window(row: dict, unit_slug: str, data: dict,
                 deadline: float) -> tuple[list[str] | None, dict[str, str] | None, str | None, list[str]]:
    """Resolves the trusted, COMPUTED `fix_changed` set for the relay arm,
    AND (esc-lead-gate-R11) the trusted, COMPUTED `claim_sites` map — every
    claim-shaped line the fix's OWN diff adds, keyed `path:line`, mapped to
    its own verbatim text. `(fix_changed, claim_sites, None, warnings)` on
    success; `(None, None, deny_reason, warnings)` otherwise — `warnings`
    carries any accept-side git stderr text noticed along the way (round-4:
    surfaced by the caller even when it did not change the outcome). Only
    ever called from `_relay_rejection`'s own repeat-dispatch caller, once
    R1/R2 have already passed.

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
        return None, None, "hook needs CLAUDE_PROJECT_DIR for the relay arm", []

    warnings: list[str] = []

    block_sha = row.get("head_sha")
    if not (isinstance(block_sha, str) and _SHA_RE.fullmatch(block_sha)):
        return None, None, "BLOCK row's head_sha is not a valid sha", warnings
    block_sha, why = _resolve_sha_exact(
        block_sha, "BLOCK row's head_sha", project_dir, deadline, warnings)
    if why is not None:
        return None, None, why, warnings

    fix_head = data.get("fix_head")
    if not (isinstance(fix_head, str) and _SHA_RE.fullmatch(fix_head)):
        return None, None, "relay carries no valid `fix_head` — the lead must write the fix commit's full sha", warnings

    if fix_head == block_sha:
        return None, None, "no fix commit since the BLOCK; a second dispatch without a fix is a re-roll", warnings

    fix_head, why = _resolve_sha_exact(fix_head, "relay `fix_head`", project_dir, deadline, warnings)
    if why is not None:
        return None, None, why, warnings
    if fix_head == block_sha:
        # Round-4: a fix_head given as a SHORT prefix can resolve to the
        # SAME full commit as block_sha even when the two caller-supplied
        # strings differed — still a re-roll, only visible after both are
        # resolved to their full form.
        return None, None, "no fix commit since the BLOCK; a second dispatch without a fix is a re-roll", warnings

    relay_ub = data.get("unit_branch")
    if not (isinstance(relay_ub, str) and relay_ub.strip()):
        return None, None, "relay carries no `unit_branch` naming the unit this fix landed on", warnings
    if unit_slug == "UNBOUND":
        return None, None, (
            "this BLOCK was recorded without a unit binding (the UNBOUND fallback bucket) — "
            "re-dispatch naming the unit so the verdict lands on the unit's own file, then "
            "hand-remove the stale row for this block from UNBOUND.jsonl (never `rm` the "
            "shared file — it holds every other unit's UNBOUND rows too)"
        ), warnings
    if slugify(relay_ub) != unit_slug:
        return None, None, (
            f"relay `unit_branch` {relay_ub!r} does not name this BLOCK's own unit "
            f"(recorded under slug {unit_slug!r}) — reachability is bound to the file this "
            "BLOCK is filed under, not an arbitrary branch the relay names"
        ), warnings

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
        ), warnings
    ok, why, rc = _run_git(
        ["merge-base", "--is-ancestor", "--end-of-options", fix_head, tip], project_dir, deadline, warnings)
    if not ok:
        if rc == 1:
            return None, None, (
                f"fix_head {fix_head} is not on {relay_ub!r}; if the fix was amended, name "
                "the amended sha; if it was committed on a child branch, commit or merge it "
                f"onto {relay_ub!r}"
            ), warnings
        return None, None, f"could not check whether fix_head is on {relay_ub!r} — {why}", warnings

    ok, out, _rc = _run_git(
        ["diff", "--name-only", "-z", "--end-of-options", block_sha, fix_head], project_dir, deadline, warnings)
    if not ok:
        return None, None, f"could not compute the fix window: {out}", warnings
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
        return None, None, f"could not compute the fix's added-line diff (esc-lead-gate-R11): {diff_out}", warnings
    claim_sites = _parse_claim_sites(diff_out)
    return fix_changed, claim_sites, None, warnings


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

    # esc-lead-gate-R10 — ARMED UNCONDITIONALLY, alongside R2 above. A ONE-LINE
    # schema requirement over lead-authored text, never a count or a match: a
    # cost floor, not proof of examination — the same limit the >=2-clean
    # rule above already carries. Alongside the >=2 examined-CLEAN sites,
    # the relay must ALSO name a site examined and explicitly NOT closed,
    # carrying the attack the next round should run.
    open_question = data.get("open_question")
    if not isinstance(open_question, str) or not open_question.strip():
        return ("relay carries no `open_question` — alongside the >=2 examined-clean "
                "sites above, one further entry must be a site you examined and could "
                "NOT close, naming the attack for the next round to run (esc-lead-gate-R10)")

    # R3 PROBE-THE-FIX (esc-097) — the only git subprocess in this module;
    # see its own module-doc paragraph for the arm order. esc-lead-gate-R11
    # shares this SAME deadline (minted ONCE, here) for its own claim-scan —
    # never a second, separate budget.
    deadline = _new_git_deadline()
    fix_changed, claim_sites, why, fix_warnings = _fix_window(row, unit_slug, data, deadline)
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
        # Implementer-dispatch binding is OUT OF SCOPE by the round-3 core
        # cut — a GATED non-verifier type dispatch is always allowed.
        return True, f"agent type {subtype!r} is GATED but dispatch-binding is out of scope (§3 core)"
    prompt = _first_str(tool_input, ("prompt", "description")) or ""
    return _decide_verifier_dispatch(subtype, prompt, sdir)


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


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "--export-oracle":
        return cmd_export_oracle(argv)
    if len(argv) >= 2 and argv[1] == "--export":
        return cmd_export(argv)
    if len(argv) < 2 or argv[1] not in ("start", "stop", "pre"):
        sys.stderr.write(
            "lead-gate-lib: usage: lead-gate-lib.py {start|stop|pre} < payload.json "
            "| --export <unit_slug> | --export-oracle <unit_slug>\n")
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
        # cmd == "pre"
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
