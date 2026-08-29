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
  sites outside enumeration+findings) is armed ALWAYS.

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

No git subprocess anywhere in this module (hot-path speed).
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

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

# "_PASS_LIKE = exactly each card's PASS vocabulary" — "PASS" is every card
# except fix-verifier ("verified") and pressure-tester ("PROCEED"). Any
# other value defaults to BLOCK.
_PASS_LIKE = {"PASS", "verified", "PROCEED"}
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


def normalize_verdict(data: dict | None) -> tuple[str, str | None]:
    """Returns `(verdict, verdict_raw)`; `verdict` is one of "PASS" |
    "BLOCK" | "UNPARSEABLE"."""
    if data is None:
        return "UNPARSEABLE", None
    raw = data.get("verdict")
    if raw is None:
        raw = data.get("overall")
    if raw is None:
        return "UNPARSEABLE", None
    if not isinstance(raw, str):
        return "BLOCK", repr(raw)
    if raw in _PASS_LIKE:
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


def _adversarial_audit_cleared_by_verifier_pass(sdir: Path, unit_slug: str, aa_row: dict,
                                                  aa_idx: int, by_type: dict) -> bool:
    """A fix-verifier/acceptance-verifier PASS ALSO clears an older
    adversarial-audit BLOCK on the same unit IF an accepted relay artifact
    for that BLOCK exists (the normal workflow's resolution path).
    Inherits the esc-064 conjunction via `_relay_accepted` — deliberately
    the predicate, not this caller (though per the G13 note this arm has
    no independently observable effect in any reachable state)."""
    if not _relay_accepted(sdir, unit_slug, aa_row):
        return False
    for t in ("fix-verifier", "acceptance-verifier"):
        entry = by_type.get(t)
        if entry is None:
            continue
        vrow, vidx = entry
        if vrow.get("verdict") == "PASS" and vidx > aa_idx:
            return True
    return False


def open_blocks_for_unit(sdir: Path, unit_slug: str) -> list[tuple[str, dict, int]]:
    """`[(agent_type, row, idx), …]` for every agent_type whose LATEST row
    on this unit is BLOCK-equivalent (and, for `adversarial-audit`, not
    cross-type-cleared)."""
    by_type = _unit_rows_by_agent_type(sdir, unit_slug)
    out: list[tuple[str, dict, int]] = []
    for atype, (row, idx) in by_type.items():
        if not is_open(row.get("verdict", "UNPARSEABLE")):
            continue
        if atype == "adversarial-audit" and _adversarial_audit_cleared_by_verifier_pass(
                sdir, unit_slug, row, idx, by_type):
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


def _relay_rejection(sdir: Path, unit_slug: str, row: dict) -> str | None:
    """`None` == accepted; otherwise the operator-facing REASON the relay is
    missing or insufficient.

    The relay's requirements are a CONJUNCTION, never a two-arm disjunction
    (esc-064): R1 (coverage) is armed by the DATA — a non-empty
    `class_enumeration` — and R2 (adjacent probing) is armed ALWAYS, for
    every gated verifier type's relay. The recorded `enumeration_missing`
    flag is diagnostic only and is never read here: a flag with a weak-arm
    default let a legacy/hand-edited row take the fallback arm and drop a
    requirement entirely."""
    agent_type = row.get("agent_type") or ""
    block_ts = row.get("ts") or ""
    unit_branch = row.get("unit_branch")
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
    if data.get("unit_branch") != unit_branch:
        return "relay `unit_branch` does not match the BLOCK row's"
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
    return None


def _relay_accepted(sdir: Path, unit_slug: str, row: dict) -> bool:
    return _relay_rejection(sdir, unit_slug, row) is None


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
    if raw is not None and raw != "BLOCK" and raw not in _PASS_LIKE:
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
    `worktree` differs from every binding on record."""
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
    unresolved = [(u, r, why) for u, r in targeted
                  for why in [_relay_rejection(sdir, u, r)] if why is not None]
    if unresolved:
        names = "; ".join(f"{u}/{r.get('agent_type')}{_diagnose_row(r)}: {why}"
                           for u, r, why in unresolved)
        return False, (
            f"a second {subtype} dispatch naming {names} is denied — the relay artifact "
            "for that (unit, agent_type, block_ts) is missing or insufficient"
        )
    return True, f"every named {subtype} BLOCK has an accepted relay artifact — allowed"


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
    verdict, verdict_raw = normalize_verdict(data)

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


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in ("start", "stop", "pre"):
        sys.stderr.write("lead-gate-lib: usage: lead-gate-lib.py {start|stop|pre} < payload.json\n")
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
